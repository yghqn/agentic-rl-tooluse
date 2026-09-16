"""One-step, no-checkpoint GRPO throughput diagnostic for serial vs batched rollouts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grpo.policy import GRPOPolicy
from grpo.rollouts import ROLLOUT_ENGINE, collect_group_batched, collect_group_serial, variance_report
from grpo.schemas import GRPOConfig
from grpo.tasks import build_task_manifest, coordinates
from grpo.trainer import update_groups, weights_digest
from sft.training import write_json_once


class _GpuMonitor:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.values: list[int] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5,
                ).stdout.splitlines()[0]
                self.values.append(int(output.strip()))
            except Exception as exc:  # Utilization is optional diagnostic evidence.
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=10)

    def report(self) -> dict:
        values = self.values
        return {
            "sample_count": len(values),
            "mean_percent": statistics.mean(values) if values else None,
            "median_percent": statistics.median(values) if values else None,
            "max_percent": max(values) if values else None,
            "error": self.error,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Diagnostic one-step GRPO throughput benchmark")
    parser.add_argument("--engine", choices=("serial", "batched"), required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--sft-manifest", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-seed", type=int, default=40000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args(argv)
    directory = Path(args.output_dir)
    if directory.exists() and any(directory.iterdir()):
        parser.error("Output directory must be new/empty")
    directory.mkdir(parents=True, exist_ok=True)
    config = GRPOConfig(
        args.sft_adapter, str(directory), args.sft_manifest,
        model=args.model, revision=args.revision, cache_dir=args.cache_dir,
        tokenizer_path=args.tokenizer_path, device=args.device, dtype=args.dtype,
        seed=args.seed, group_size=args.group_size, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens, max_steps=args.max_steps,
        prompts_per_step=1, optimizer_steps=1, learning_rate=args.learning_rate,
    )
    manifest = build_task_manifest(Path(config.sft_manifest))
    candidates = [c for c in coordinates(manifest, "train")
                  if c.difficulty == 4 and c.seed == args.task_seed]
    if not candidates:
        parser.error("No train-only L4 coordinate matches --task-seed")
    coordinate = candidates[0]
    write_json_once(directory / "tasks.json", {"coordinate": asdict(coordinate)})
    policy = GRPOPolicy(config)
    import torch
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=config.learning_rate, weight_decay=0)
    frozen_before = weights_digest(policy, frozen=True)
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    collect = collect_group_serial if args.engine == "serial" else collect_group_batched
    with _GpuMonitor() as monitor:
        started = time.perf_counter()
        group = collect(policy, coordinate, 0, directory)
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
        rollout_seconds = time.perf_counter() - started
        alignment = policy.validate_alignment(group.members)
        stats = update_groups(policy, [group], optimizer)
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
        step_seconds = time.perf_counter() - started
    if weights_digest(policy, frozen=True) != frozen_before:
        raise RuntimeError("Frozen base/reference changed during benchmark update")
    diagnostics = variance_report([group], config.std_floor)
    report = {
        "schema_version": "grpo-throughput-benchmark-v1",
        "engine": args.engine,
        "production_rollout_engine": ROLLOUT_ENGINE,
        "diagnostic_only": True,
        "checkpoint_saved": False,
        "config": asdict(config),
        "coordinate": asdict(coordinate),
        "rollout_seconds": rollout_seconds,
        "time_per_optimizer_step_seconds": step_seconds,
        "alignment": alignment,
        "update": stats,
        "diagnostics": diagnostics,
        "gpu_utilization": monitor.report(),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated() if config.device.startswith("cuda") else 0,
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved() if config.device.startswith("cuda") else 0,
    }
    write_json_once(directory / "benchmark_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
