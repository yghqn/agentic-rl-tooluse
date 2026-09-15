"""Prompt Agent evaluation CLI: offline replay or optional HF inference, no training.

Run: python -m scripts.evaluate --seed 42 --count-per-level 2 --responses replay.jsonl
Each JSONL row: {"task_id": "...", "responses": ["<model output>", ...]}.
Responses are fixtures, NOT generated solutions or evidence of model capability.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import hashlib
from pathlib import Path
import sys
import subprocess
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent import Agent, PromptAgent, ScriptedBackend
from environment.database import SyntheticCompanyDatabase
from environment.environment import Environment
from evaluation.failure_analysis import analyze_failures
from evaluation.metrics import EvaluationRecord, compute_metrics
from evaluation.verifier import verify_final_answer
from tasks.generator import TaskGenerator, agent_task_view
from tasks.schemas import Task
from tasks.validators import environment_id_for_seed, validate_task


RESULT_FILES = ("trajectories.jsonl", "metrics.json", "run_config.json", "comparison.json")


def check_output_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError("output-dir must be a directory")
    if any((path / name).exists() for name in RESULT_FILES):
        raise ValueError("Result files already exist; choose a new output-dir")


def project_git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"git_commit_hash": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit_hash": None, "git_dirty": None}


def save_report(path: Path, report: EvaluationReport, config: dict[str, Any]) -> None:
    check_output_directory(path)
    path.mkdir(parents=True, exist_ok=True)
    exported = report.to_dict()
    # Exclusive creation protects previous results, including concurrent writers.
    with (path / "trajectories.jsonl").open("x", encoding="utf-8") as stream:
        for row in exported["runs"]:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    for name, value in (("metrics.json", report.metrics), ("run_config.json", config)):
        with (path / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


@dataclass(slots=True)
class EvaluationReport:
    records: list[EvaluationRecord]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Explicit export: never serialize a full privileged Task record."""

        return {
            "metrics": self.metrics,
            "runs": [{
                "task_id": record.task.task_id,
                "difficulty": record.task.difficulty,
                "environment_id": record.task.environment_id,
                "run": asdict(record.run),
                "verification": asdict(record.verification),
                "failure_analysis": asdict(record.failure_analysis),
            } for record in self.records],
        }


def evaluate_tasks(
    tasks: Sequence[Task],
    agent_factory: Callable[[str], Agent],
    environment_seeds: Mapping[str, int],
) -> EvaluationReport:
    """Run in (difficulty, task_id) order using a fresh Agent and environment.

    The factory receives only task_id for fixture selection, never the full Task.
    It must create independent conversation state (stateless model weights may
    be shared). All benchmark tasks are validated
    before any Agent executes, so bad benchmark configuration fails fast.
    """

    task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, Task) or not isinstance(task.environment_id, str):
            raise ValueError("Expected a Task with a string environment_id")
        if task.environment_id not in environment_seeds:
            raise ValueError(f"No seed configured for environment: {task.environment_id}")
        validate_task(task, SyntheticCompanyDatabase(environment_seeds[task.environment_id]))
        if task.task_id in task_ids:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        task_ids.add(task.task_id)

    records: list[EvaluationRecord] = []
    agents: list[Agent] = []
    for task in sorted(tasks, key=lambda item: (item.difficulty, item.task_id)):
        agent = agent_factory(task.task_id)
        if any(agent is previous for previous in agents):
            raise ValueError("agent_factory must return a fresh Agent for every task")
        agents.append(agent)
        environment = Environment(seed=environment_seeds[task.environment_id])
        run = agent.run(agent_task_view(task), environment)
        verification = (
            verify_final_answer(task, run.final_answer) if run.has_final_answer
            else verify_final_answer(task)
        )
        analysis = analyze_failures(task, run, verification)
        records.append(EvaluationRecord(task, run, verification, analysis))
    return EvaluationReport(records, compute_metrics(records))


def load_scripted_responses(path: Path) -> dict[str, list[str]]:
    scripts: dict[str, list[str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}") from exc
        if not isinstance(row, dict) or set(row) != {"task_id", "responses"}:
            raise ValueError(f"Invalid replay row at line {line_number}")
        task_id, responses = row["task_id"], row["responses"]
        if (
            not isinstance(task_id, str) or not task_id.strip()
            or not isinstance(responses, list)
            or any(not isinstance(output, str) for output in responses)
        ):
            raise ValueError(f"Invalid replay fields at line {line_number}")
        if task_id in scripts:
            raise ValueError(f"Duplicate replay task_id: {task_id}")
        scripts[task_id] = responses
    return scripts


def load_manifest_benchmark(path: Path, split: str = "test") -> tuple[list[Task], dict[str, int], dict[str, Any]]:
    """Read coordinate identity ONLY. Never open sibling SFT/oracle exports."""
    if split not in {"dev", "test"}:
        raise ValueError("Manifest benchmark split must be dev or test")
    content = path.read_bytes()
    manifest = json.loads(content)
    if manifest.get("schema_version") != "sft-manifest-v1":
        raise ValueError("Unsupported benchmark manifest")
    coordinates = manifest["splits"][split]["coordinates"]
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("Empty benchmark coordinates")
    tasks, seeds, identities = [], {}, []
    seen = set()
    for row in coordinates:
        # Whitelist: even if a manifest has extra privileged fields, they are ignored.
        coordinate = {key:row[key] for key in ("seed", "difficulty", "task_index", "task_id")}
        if (any(type(coordinate[key]) is not int for key in ("seed", "difficulty", "task_index"))
            or coordinate["difficulty"] not in {1,2,3,4} or coordinate["task_index"] < 0):
            raise ValueError("Invalid benchmark coordinate")
        task = TaskGenerator(coordinate["seed"]).generate_task(coordinate["difficulty"], coordinate["task_index"])
        if task.task_id != coordinate["task_id"] or task.task_id in seen:
            raise ValueError("Benchmark task identity mismatch/duplicate")
        seen.add(task.task_id)
        tasks.append(task)
        seeds[task.environment_id] = coordinate["seed"]
        identities.append(coordinate)
    return tasks, seeds, {"benchmark_split":split, "benchmark_coordinates":identities,
                          "benchmark_manifest_sha256":hashlib.sha256(content).hexdigest()}


def validate_comparison(base: dict[str, Any], adapted: dict[str, Any]) -> None:
    from sft.training import validate_adapter_identity

    if (base.get("mode") != "hf_benchmark" or adapted.get("mode") != "hf_benchmark"
        or base.get("adapter_path") is not None or adapted.get("adapter_identity_validated") is not True):
        raise ValueError("Comparison requires a real base benchmark and a provenance-validated adapter benchmark")
    validate_adapter_identity(base["model_identity"], adapted["model_identity"])
    for key in ("task_ids", "environment_seeds", "max_steps", "generation_seed", "generation_arguments",
                "dtype", "chat_template_strategy", "tool_message_strategy", "transformers_version", "torch_version"):
        if key not in base or base[key] != adapted.get(key):
            raise ValueError(f"Base/SFT comparison configuration mismatch: {key}")
    if base["requested_backend_config"]["device"] != adapted["requested_backend_config"]["device"]:
        raise ValueError("Base/SFT comparison device mismatch")
    # Dataset identities are immutable; paths are not needed to compare benchmark semantics.
    if base.get("benchmark_coordinates") != adapted.get("benchmark_coordinates"):
        raise ValueError("Base/SFT benchmark coordinates mismatch")


def comparison_metrics(base: dict[str, Any], adapted: dict[str, Any]) -> dict[str, Any]:
    keys = ("task_success_rate", "invalid_tool_call_rate", "average_tool_calls", "average_agent_steps")
    return {"base":base, "sft":adapted,
            "delta":{key:adapted[key]["value"] - base[key]["value"]
                     if adapted[key]["value"] is not None and base[key]["value"] is not None else None for key in keys},
            "parse_error_count_delta":adapted["counts"]["parse_error_count"] - base["counts"]["parse_error_count"],
            "success_by_difficulty_delta":{
                level:adapted["by_difficulty"][level]["task_success_rate"]["value"] - values["task_success_rate"]["value"]
                if values["task_success_rate"]["value"] is not None and adapted["by_difficulty"][level]["task_success_rate"]["value"] is not None
                else None for level, values in base["by_difficulty"].items()}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt Agent scripted replay or Hugging Face benchmark")
    parser.add_argument("--backend", choices=("scripted", "hf"), default="scripted")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--count-per-level", type=int)
    parser.add_argument("--benchmark-manifest", type=Path)
    parser.add_argument("--benchmark-split", choices=("dev", "test"), default="test")
    parser.add_argument("--generation-seed", type=int, help="Defaults to benchmark seed, or 42 for manifests")
    parser.add_argument("--adapter-path", help="Saved adapter directory; requires exact training model revision")
    parser.add_argument("--tokenizer-path", help="Optional saved training tokenizer directory")
    parser.add_argument("--compare-to", type=Path, help="Base HF result directory; refuse incompatible configurations")
    parser.add_argument("--cache-dir")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--responses", type=Path, help="Per-task JSONL model-output fixtures (scripted only)")
    parser.add_argument("--output-dir", type=Path, help="Save trajectories, metrics and run configuration")
    parser.add_argument("--model", help="HF model name or local path")
    parser.add_argument("--revision", help="Requested HF revision; pin a commit for reproducibility")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-fallback-template", action="store_true", help="DEBUG ONLY: allow absent chat template")
    arguments = parser.parse_args(argv)
    try:
        if arguments.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if arguments.output_dir is not None:
            check_output_directory(arguments.output_dir)
        benchmark = {}
        if arguments.benchmark_manifest:
            if arguments.seed is not None or arguments.count_per_level is not None:
                raise ValueError("Manifest benchmark cannot be combined with --seed/--count-per-level")
            tasks, environment_seeds, benchmark = load_manifest_benchmark(arguments.benchmark_manifest, arguments.benchmark_split)
        else:
            arguments.seed = 42 if arguments.seed is None else arguments.seed
            arguments.count_per_level = 2 if arguments.count_per_level is None else arguments.count_per_level
            tasks = TaskGenerator(arguments.seed).generate_tasks(arguments.count_per_level)
            environment_seeds = {environment_id_for_seed(arguments.seed):arguments.seed}
        configuration = {
            "backend": arguments.backend,
            "seed": arguments.seed,
            "count_per_level": arguments.count_per_level,
            "max_steps": arguments.max_steps,
            "order": "difficulty_then_task_id",
            "task_ids": [task.task_id for task in sorted(tasks, key=lambda item: (item.difficulty, item.task_id))],
            "environment_seeds": environment_seeds,
            **benchmark,
            **project_git_state(),
        }
        if arguments.backend == "scripted":
            if arguments.adapter_path or arguments.tokenizer_path or arguments.compare_to:
                raise ValueError("Adapter evaluation/comparison requires --backend hf")
            if arguments.responses is None or arguments.model is not None:
                raise ValueError("scripted requires --responses and does not accept --model")
            scripts = load_scripted_responses(arguments.responses)
            if set(scripts) != {task.task_id for task in tasks}:
                raise ValueError("Replay task IDs must exactly match the generated benchmark")
            agent_factory = lambda task_id: PromptAgent(ScriptedBackend(scripts[task_id]), arguments.max_steps)
            mode = "scripted_replay"
        else:
            if not arguments.model or arguments.output_dir is None or arguments.responses is not None:
                raise ValueError("hf requires --model and --output-dir, and does not accept --responses")
            from agent.hf_backend import HuggingFaceBackend, HuggingFaceConfig

            backend_config = HuggingFaceConfig(
                model_name_or_path=arguments.model, revision=arguments.revision, device=arguments.device,
                max_new_tokens=arguments.max_new_tokens, temperature=arguments.temperature,
                top_p=arguments.top_p, do_sample=arguments.do_sample,
                local_files_only=arguments.local_files_only,
                allow_fallback_template=arguments.allow_fallback_template,
                adapter_path=arguments.adapter_path, tokenizer_path=arguments.tokenizer_path,
                dtype=arguments.dtype, cache_dir=arguments.cache_dir,
            )
            backend = HuggingFaceBackend(backend_config)
            # Seed sampling once for the whole deterministic task order. Greedy
            # remains the default; no per-task conversational state is shared.
            from transformers import set_seed

            generation_seed = (arguments.generation_seed if arguments.generation_seed is not None
                               else (arguments.seed if arguments.seed is not None else 42)) % (2**32)
            set_seed(generation_seed)
            configuration["generation_seed"] = generation_seed
            configuration.update({"requested_backend_config": asdict(backend_config), **backend.runtime_config()})
            mode = "hf_debug_fallback" if backend.chat_template_strategy == "debug_plaintext_fallback" else "hf_benchmark"
            agent_factory = lambda task_id: PromptAgent(backend, arguments.max_steps)
        configuration["mode"] = mode
        if arguments.compare_to:
            if arguments.output_dir is None:
                raise ValueError("Comparison requires --output-dir")
            base_config = json.loads((arguments.compare_to / "run_config.json").read_text(encoding="utf-8"))
            validate_comparison(base_config, configuration)  # BEFORE any task runs.
            base_metrics = json.loads((arguments.compare_to / "metrics.json").read_text(encoding="utf-8"))
        report = evaluate_tasks(
            tasks, agent_factory, environment_seeds,
        )
        if arguments.output_dir is not None:
            save_report(arguments.output_dir, report, configuration)
            if arguments.compare_to:
                with (arguments.output_dir / "comparison.json").open("x", encoding="utf-8") as stream:
                    json.dump(comparison_metrics(base_metrics, report.metrics), stream, indent=2, sort_keys=True, allow_nan=False)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    output = {
        "mode": mode,
        "configuration": configuration,
        **report.to_dict(),
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
