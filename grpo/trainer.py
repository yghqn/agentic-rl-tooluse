"""Single-device custom GRPO: collect, cache reference, locked policy update.

Each group has equal batch weight. Its assistant tokens share that weight;
context (including previous assistant turns and tool observations) has no loss.
No oracle data, evaluation metrics, or test answers are inputs to the loss.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

from grpo.loss import group_advantages, token_loss
from grpo.policy import adapter_hash
from grpo.rollouts import ROLLOUT_ENGINE, append_jsonl, collect_group, variance_report
from grpo.schemas import TaskCoordinate, task_from_coordinate
from grpo.tasks import coordinates, training_schedule
from sft.preprocessing import fingerprint
from sft.training import validate_adapter_identity, write_json_once


def require_probe(path, config):
    """Bind a real no-update, train-only variance/alignment gate to this run."""
    if path is None:
        raise ValueError("--probe-report is mandatory before any optimizer update")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("model","revision","dtype","seed","group_size","temperature","top_p",
                "max_new_tokens","max_steps","reward_variant","alignment_tolerance"):
        if report["config"].get(key) != getattr(config,key):
            raise ValueError(f"Probe configuration mismatch: {key}")
    if (report.get("rollout_engine") != ROLLOUT_ENGINE
        or report.get("passed") is not True or report.get("optimizer_updates") != 0
        or report.get("group_count") != 20 or report.get("rollout_count") != 20*config.group_size
        or report.get("mixed_reward_group_count",0) <= 0
        or report.get("zero_advantage_group_count",20) >= 20
        or report.get("alignment",{}).get("passed") is not True):
        raise ValueError("Probe failed variance/alignment/no-update gate")
    if report.get("initial_sft_adapter_sha256") != adapter_hash(Path(config.sft_adapter)):
        raise ValueError("Probe used a different starting SFT adapter")
    manifest = json.loads((Path(path).parent/"tasks.json").read_text(encoding="utf-8"))
    train = {c.task_id:c for c in coordinates(manifest,"train")}
    rows = report.get("coordinates",[])
    if len(rows) != 20 or len({r["seed"] for r in rows}) != 20:
        raise ValueError("Probe must contain 20 distinct train-only L4 environments")
    for row in rows:
        c = TaskCoordinate(**row)
        if c.difficulty != 4 or train.get(c.task_id) != c or not 40000 <= c.seed < 40256:
            raise ValueError("Probe coordinate is not a train-only L4 task")
        task_from_coordinate(c)
    if report.get("task_ids") != [r["task_id"] for r in rows]:
        raise ValueError("Probe identity mismatch")
    return report


def weights_digest(policy, *, frozen):
    """Hash one tensor at a time, with no full-model CPU/GPU duplicate."""
    import torch
    digest = hashlib.sha256()
    for name,p in policy.model.named_parameters():
        is_policy = ".default." in name and (".lora_A." in name or ".lora_B." in name)
        if is_policy != frozen:
            digest.update(name.encode())
            digest.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def update_groups(policy, groups, optimizer, reference_writer=None):
    """All reference forwards precede ALL policy graphs and the single step.

    L = mean_groups [ sum_sampled_tokens(-min(r*A,clip(r)*A)+beta*KL) / N_group ].
    Backward per turn releases its graph, accumulating one batch's gradients.
    """
    import torch
    config = policy.config
    members = [m for g in groups for m in g.members]
    if not groups or any(len(g.members) != config.group_size for g in groups):
        raise ValueError("Incomplete GRPO batch/group")
    policy.cache_reference(members)  # no_grad; cache CPU numbers before switch
    if reference_writer:
        for member in members:
            reference_writer(member)
    parameters = policy.trainable_parameters()
    frozen_versions = {name:p._version for name,p in policy.model.named_parameters()
                       if id(p) not in {id(x) for x in parameters}}
    stats = {"loss":0.0,"policy_loss":0.0,"kl":0.0,"clip_count":0,"sampled_token_count":0,
             "nonzero_advantage_group_count":0,"max_policy_old_logprob_error":0.0}
    optimizer.zero_grad(set_to_none=True)
    with policy.update_phase():  # switch once, LOCK through optimizer.step
        for group in groups:
            advantages = group_advantages([m.reward.total for m in group.members],config.std_floor)
            stats["nonzero_advantage_group_count"] += any(a != 0 for a in advantages)
            token_count = sum(len(t.generated_ids) for m in group.members for t in m.turns)
            if not token_count:
                raise ValueError("Empty supervised batch")
            denominator = len(groups)*token_count
            for member,advantage in zip(group.members,advantages,strict=True):
                if len(member.reference_logprobs) != len(member.turns):
                    raise ValueError("Incomplete reference cache")
                for turn,ref in zip(member.turns,member.reference_logprobs,strict=True):
                    current = policy.score(turn)
                    old = torch.tensor(turn.old_logprobs,device=current.device)
                    reference = torch.tensor(ref,device=current.device)
                    delta = (current.detach()-old).abs().max().item()
                    stats["max_policy_old_logprob_error"] = max(stats["max_policy_old_logprob_error"],delta)
                    if delta > config.alignment_tolerance:
                        raise ValueError("Train-mode sampled logprob alignment gate failed; no optimizer step")
                    loss,details = token_loss(current,old,reference,advantage,config.kl_beta,config.clip_epsilon)
                    weighted = loss/denominator
                    stats["loss"] += weighted.detach().item()
                    stats["policy_loss"] += details["pg_sum"]/denominator
                    stats["kl"] += details["kl_sum"]/denominator
                    stats["clip_count"] += details["clip_count"]
                    stats["sampled_token_count"] += details["token_count"]
                    weighted.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
        stats["gradient_norm"] = norm.item()
        if not all(math.isfinite(v) for v in stats.values()):
            raise ValueError("Non-finite training statistics")
        for name,p in policy.model.named_parameters():
            if name in frozen_versions and p.grad is not None:
                raise ValueError("Frozen base/reference received gradients")
        optimizer.step()  # Still in locked policy phase; never switch here.
    for name,p in policy.model.named_parameters():
        if name in frozen_versions and p._version != frozen_versions[name]:
            raise ValueError("Frozen base/reference changed during optimizer update")
    return stats


class GreedyBackend:
    """Same normal HF messages/generate path; no sampling during evaluation."""
    def __init__(self, backend):
        self.backend = backend

    def generate(self, messages):
        backend = self.backend
        kwargs = dict(backend._generation_kwargs)
        kwargs.pop("temperature",None)
        kwargs.pop("top_p",None)
        kwargs["do_sample"] = False
        original_kwargs,original_config = backend._generation_kwargs,backend._generation_config
        try:
            backend._generation_kwargs = kwargs
            backend._generation_config = backend._transformers.GenerationConfig(**kwargs)
            return backend.generate(messages)
        finally:
            backend._generation_kwargs,backend._generation_config = original_kwargs,original_config


def evaluate_dev(policy, manifest, directory, *, small):
    from agent.agent import PromptAgent
    from scripts.evaluate import evaluate_tasks, save_report
    policy.switch("policy")
    cs = coordinates(manifest,"dev")
    if small:
        cs = [c for c in cs if 50000 <= c.seed < 50004]
    tasks = [task_from_coordinate(c) for c in cs]
    seeds = {task.environment_id:c.seed for task,c in zip(tasks,cs,strict=True)}
    report = evaluate_tasks(tasks,lambda _:PromptAgent(GreedyBackend(policy.backend),policy.config.max_steps),seeds)
    save_report(directory,report,{"mode":"grpo_greedy_dev","identity":policy.backend.model_identity(),
                "coordinates":[asdict(c) for c in cs],"generation":{"do_sample":False,"num_beams":1},
                "max_steps":policy.config.max_steps,"max_new_tokens":policy.config.max_new_tokens})
    return report.metrics


def dev_guards(before, after):
    """Observable held-out dev guards, never reference-path or test selection."""
    return {"l1_l3_preserved":all(after["by_difficulty"][level]["task_success_rate"]["value"] >= before["by_difficulty"][level]["task_success_rate"]["value"] for level in ("L1","L2","L3")),
            "parse_errors_not_increased":after["counts"]["parse_error_count"] <= before["counts"]["parse_error_count"],
            "invalid_calls_not_increased":after["counts"]["invalid_tool_call_count"] <= before["counts"]["invalid_tool_call_count"],
            "redundant_calls_not_increased":after["counts"]["redundant_tool_call_count"] <= before["counts"]["redundant_tool_call_count"],
            "trajectory_validity_preserved":after["counts"]["valid_trajectory_count"] >= before["counts"]["valid_trajectory_count"]}


def dev_selection_key(metrics):
    return (metrics["by_difficulty"]["L4"]["task_success_rate"]["value"],
            metrics["task_success_rate"]["value"],-metrics["average_agent_steps"]["value"])


def save_checkpoint(policy, optimizer, directory, step, schedule, history):
    import torch
    checkpoint = directory/f"checkpoint-{step:04d}"
    checkpoint.mkdir()
    policy.switch("policy")
    policy.model.save_pretrained(checkpoint/"adapter",selected_adapters=[policy.policy_adapter],
                                 safe_serialization=True,save_embedding_layers=False)
    policy.tokenizer.save_pretrained(checkpoint/"tokenizer")
    run = json.loads((directory/"run_config.json").read_text(encoding="utf-8"))
    run.update(schema_version="lora-grpo-run-v1",status="complete",optimizer_step=step,
               adapter_sha256=adapter_hash(checkpoint/"adapter"),
               adapter_precision_strategy="FP32 LoRA initialized from effective HF-loaded SFT values",
               schedule_sha256=fingerprint([asdict(c) for c in schedule]))
    write_json_once(checkpoint/"run_config.json",run)
    write_json_once(checkpoint/"loss_history.json",history)
    # Full resume/audit state is saved; resumption is deliberately not implicit.
    torch.save({"optimizer":optimizer.state_dict(),"optimizer_step":step,
                "schedule":[asdict(c) for c in schedule],"next_coordinate_index":step*policy.config.prompts_per_step,
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng":random.getstate()},checkpoint/"training_state.pt")
    return checkpoint


def train_grpo(policy, manifest, directory, probe_path):
    import torch
    config = policy.config
    probe = require_probe(probe_path,config)
    validate_adapter_identity(probe["identity"],policy.backend.model_identity())
    if fingerprint(manifest) != fingerprint(json.loads((Path(probe_path).parent/"tasks.json").read_text(encoding="utf-8"))):
        raise ValueError("Probe/train task manifests differ")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(),lr=config.learning_rate,weight_decay=0)
    schedule = training_schedule(manifest,config.seed,config.optimizer_steps*config.prompts_per_step)
    write_json_once(directory/"schedule.json",[asdict(c) for c in schedule])
    small = json.loads((directory/"run_config.json").read_text(encoding="utf-8"))["mode"] == "smoke"
    baseline = evaluate_dev(policy,manifest,directory/"dev-before",small=small)
    frozen_before = weights_digest(policy,frozen=True)
    policy_before = weights_digest(policy,frozen=False)
    history,all_groups,checkpoints,selection = [],[],[],[]
    best_key,best_checkpoint = dev_selection_key(baseline),None
    for step in range(1,config.optimizer_steps+1):
        cs = schedule[(step-1)*config.prompts_per_step:step*config.prompts_per_step]
        groups = [collect_group(policy,c,step-1,directory) for c in cs]
        all_groups.extend(groups)
        members = [m for g in groups for m in g.members]
        alignment = policy.validate_alignment(members)
        stats = update_groups(policy,groups,optimizer,lambda m:append_jsonl(directory/"reference_logprobs.jsonl",
            {"rollout_id":m.rollout_id,"sampled_token_reference_logprobs":m.reference_logprobs}))
        rewards = variance_report(groups,config.std_floor)
        row = {"optimizer_step":step,**stats,"reward_mean":rewards["reward_mean"],
               "reward_std":rewards["reward_std"],"mixed_reward_group_count":rewards["mixed_reward_group_count"],
               "alignment":alignment,"task_ids":[c.task_id for c in cs],
               "peak_memory_allocated_bytes":torch.cuda.max_memory_allocated() if config.device.startswith("cuda") else 0}
        history.append(row)
        append_jsonl(directory/"loss_history.jsonl",row)
        print(f"GRPO optimizer step {step}/{config.optimizer_steps}, loss={stats['loss']:.6f}, grad={stats['gradient_norm']:.6f}",flush=True)
        if step % 50 == 0 or step == config.optimizer_steps:
            checkpoint = save_checkpoint(policy,optimizer,directory,step,schedule,history)
            checkpoints.append(str(checkpoint))
            after = evaluate_dev(policy,manifest,directory/("dev-after" if step == config.optimizer_steps else f"dev-step-{step:04d}"),small=small)
            guards = dev_guards(baseline,after)
            key = dev_selection_key(after)
            eligible = all(guards.values())
            if eligible and key >= best_key:
                best_key,best_checkpoint = key,str(checkpoint)
            selection.append({"checkpoint":str(checkpoint),"eligible":eligible,"guards":guards,"dev_selection_key":list(key)})
    write_json_once(directory/"checkpoint_selection.json",{"selected_checkpoint":best_checkpoint,
        "selection_split":"RL dev only; no test access","candidates":selection,
        "fallback":"Retain initial SFT adapter if no guarded candidate matches/improves its dev selection key"})
    frozen_after = weights_digest(policy,frozen=True)
    final_policy_hash = weights_digest(policy,frozen=False)
    stats = variance_report(all_groups,config.std_floor)
    stats["optimizer_updates"] = config.optimizer_steps
    guards = {"finite_losses_and_gradients":all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in history),
              "nonzero_advantages":any(r["nonzero_advantage_group_count"] > 0 for r in history),
              "policy_updated":policy_before != final_policy_hash,"base_and_reference_unchanged":frozen_before == frozen_after,
              "sft_source_unchanged":adapter_hash(Path(config.sft_adapter)) == policy.source_hash,
              "generated_protocol_valid":stats["parse_error_count"] == 0 and stats["invalid_tool_call_count"] == 0,
              "dev_l1_l3_preserved":dev_guards(baseline,after)["l1_l3_preserved"]}
    report = {"optimizer_updates":config.optimizer_steps,"checks":guards,"passed":all(guards.values()),
              "checkpoints":checkpoints,"reward_statistics":stats,"dev_before":baseline,"dev_after":after,
              "selected_checkpoint":best_checkpoint,
              "frozen_weights_sha256_before":frozen_before,"frozen_weights_sha256_after":frozen_after,
              "policy_weights_sha256_before":policy_before,"policy_weights_sha256_after":final_policy_hash,
              "probe_report_sha256":hashlib.sha256(Path(probe_path).read_bytes()).hexdigest(),
              "peak_memory_allocated_bytes":torch.cuda.max_memory_allocated() if config.device.startswith("cuda") else 0,
              "peak_memory_reserved_bytes":torch.cuda.max_memory_reserved() if config.device.startswith("cuda") else 0}
    write_json_once(directory/("smoke_report.json" if small else "training_report.json"),report)
    return report
