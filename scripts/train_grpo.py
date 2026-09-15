"""Explicit rollout-only diagnostics or gated local smoke/custom GRPO training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from grpo.schemas import GRPOConfig
from grpo.tasks import build_task_manifest, coordinates, training_schedule
from grpo.policy import GRPOPolicy
from grpo.rollouts import collect_group, variance_report
from grpo.rewards import REWARD_CONFIG
from scripts.evaluate import project_git_state
from sft.preprocessing import fingerprint
from sft.training import write_json_once


def runtime_record(policy, manifest, mode):
    import torch, transformers, peft
    return {"schema_version":"grpo-run-v1","mode":mode,"config":asdict(policy.config),
            "identity":policy.backend.model_identity(),"lora":policy.source_record["lora"],
            "initial_sft_adapter_sha256":policy.source_hash,"task_manifest_sha256":fingerprint(manifest),
            "initial_sft_dataset":policy.source_record["dataset"],
            "optimizer":None if mode == "probe" else {"name":"AdamW","learning_rate":policy.config.learning_rate,
                "betas":[0.9,0.999],"eps":1e-8,"weight_decay":0,"gradient_clip_norm":1.0},
            "loss_normalization":"equal group weight; sampled assistant-token mean within each group",
            "policy_iterations_per_batch":1,
            "reward_config":REWARD_CONFIG,"generation_config":dict(policy.backend._generation_kwargs,
                top_k=0,use_cache=True,output_scores=True,return_dict_in_generate=True),
            "transformers_version":transformers.__version__,"peft_version":peft.__version__,
            "torch_version":torch.__version__,"cuda_version":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "trainable_parameters":sum(p.numel() for p in policy.trainable_parameters()),
            **project_git_state()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-turn GRPO; probe is mandatory before smoke/train")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe-only",action="store_true")
    mode.add_argument("--smoke",action="store_true")
    mode.add_argument("--train",action="store_true")
    parser.add_argument("--sft-adapter",type=str,required=True)
    parser.add_argument("--sft-manifest",type=str,required=True)
    parser.add_argument("--output-dir",type=str,required=True)
    parser.add_argument("--probe-report",type=Path)
    parser.add_argument("--model",default=GRPOConfig.__dataclass_fields__["model"].default)
    parser.add_argument("--revision",default=GRPOConfig.__dataclass_fields__["revision"].default)
    parser.add_argument("--cache-dir")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--dtype",choices=("float32","bfloat16"),default="bfloat16")
    parser.add_argument("--local-files-only",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--group-size",type=int,default=4)
    parser.add_argument("--temperature",type=float,default=0.8)
    parser.add_argument("--max-new-tokens",type=int,default=256)
    parser.add_argument("--max-steps",type=int,default=16)
    parser.add_argument("--prompts-per-step",type=int,default=2)
    parser.add_argument("--optimizer-steps",type=int,default=10)
    parser.add_argument("--learning-rate",type=float,default=1e-5)
    parser.add_argument("--reward",choices=("outcome_only","shaped"),default="outcome_only")
    parser.add_argument("--alignment-tolerance",type=float,default=0.05)
    args = parser.parse_args(argv)
    run_started = False
    try:
        config = GRPOConfig(args.sft_adapter,args.output_dir,args.sft_manifest,model=args.model,revision=args.revision,
            cache_dir=args.cache_dir,tokenizer_path=args.tokenizer_path,device=args.device,dtype=args.dtype,
            local_files_only=args.local_files_only,seed=args.seed,group_size=args.group_size,temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,max_steps=args.max_steps,prompts_per_step=args.prompts_per_step,
            optimizer_steps=args.optimizer_steps,learning_rate=args.learning_rate,reward_variant=args.reward,
            alignment_tolerance=args.alignment_tolerance)
        directory = Path(config.output_dir)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Output directory must be new/empty; never overwrite rollouts/checkpoints")
        if not args.probe_only:
            from grpo.trainer import require_probe
            require_probe(args.probe_report,config)
            if args.smoke and (config.optimizer_steps != 10 or config.prompts_per_step != 2):
                raise ValueError("Local smoke uses exactly 10 optimizer steps and 2 prompts/step")
        manifest = build_task_manifest(Path(config.sft_manifest))
        directory.mkdir(parents=True,exist_ok=True)
        run_started = True
        write_json_once(directory/"tasks.json",manifest)
        policy = GRPOPolicy(config)
        run_record = runtime_record(policy,manifest,"probe" if args.probe_only else "smoke" if args.smoke else "train")
        write_json_once(directory/"run_config.json",run_record)
        import torch
        if config.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        if args.probe_only:
            picked,seeds = [],set()
            for c in coordinates(manifest,"train"):
                if c.difficulty == 4 and c.seed not in seeds:
                    picked.append(c)
                    seeds.add(c.seed)
                if len(picked) == 20:
                    break
            groups = []
            for i,c in enumerate(picked):
                groups.append(collect_group(policy,c,0,directory))
                print(f"Probe group {i+1}/20 complete",file=sys.stderr,flush=True)
            members = [m for g in groups for m in g.members]
            report = variance_report(groups,config.std_floor)
            alignment_error = None
            try:
                alignment = policy.validate_alignment(members)
            except ValueError as exc:
                alignment_error = str(exc)
                alignment = dict(getattr(exc,"result",{"passed":False}),error=alignment_error)
            report.update(alignment=alignment,task_ids=[c.task_id for c in picked],
                          coordinates=[asdict(c) for c in picked],
                          config=asdict(config),identity=policy.backend.model_identity(),
                          initial_sft_adapter_sha256=policy.source_hash,
                          peak_memory_allocated_bytes=torch.cuda.max_memory_allocated() if config.device.startswith("cuda") else 0,
                          peak_memory_reserved_bytes=torch.cuda.max_memory_reserved() if config.device.startswith("cuda") else 0)
            report["passed"] = alignment["passed"] and report["mixed_reward_group_count"] > 0 and report["zero_advantage_group_count"] < 20
            write_json_once(directory/"probe_report.json",report)
            if alignment_error:
                print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))
                raise ValueError(alignment_error)
        else:
            from grpo.trainer import train_grpo
            report = train_grpo(policy,manifest,directory,args.probe_report)
        print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))
    except (ValueError,RuntimeError,OSError,ImportError,KeyError,TypeError) as exc:
        if run_started and not (directory/"run_failure.json").exists():
            write_json_once(directory/"run_failure.json",{"error_type":type(exc).__name__,"error":str(exc)})
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
