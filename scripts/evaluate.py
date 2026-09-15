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


RESULT_FILES = ("trajectories.jsonl", "metrics.json", "run_config.json")


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt Agent scripted replay or Hugging Face benchmark")
    parser.add_argument("--backend", choices=("scripted", "hf"), default="scripted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-per-level", type=int, default=2)
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
        tasks = TaskGenerator(arguments.seed).generate_tasks(arguments.count_per_level)
        configuration = {
            "backend": arguments.backend,
            "seed": arguments.seed,
            "count_per_level": arguments.count_per_level,
            "max_steps": arguments.max_steps,
            "order": "difficulty_then_task_id",
            "task_ids": [task.task_id for task in sorted(tasks, key=lambda item: (item.difficulty, item.task_id))],
            **project_git_state(),
        }
        if arguments.backend == "scripted":
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
            )
            backend = HuggingFaceBackend(backend_config)
            # Seed sampling once for the whole deterministic task order. Greedy
            # remains the default; no per-task conversational state is shared.
            from transformers import set_seed

            generation_seed = arguments.seed % (2**32)
            set_seed(generation_seed)
            configuration["generation_seed"] = generation_seed
            configuration.update({"requested_backend_config": asdict(backend_config), **backend.runtime_config()})
            mode = "hf_debug_fallback" if backend.chat_template_strategy == "debug_plaintext_fallback" else "hf_benchmark"
            agent_factory = lambda task_id: PromptAgent(backend, arguments.max_steps)
        configuration["mode"] = mode
        report = evaluate_tasks(
            tasks, agent_factory,
            {environment_id_for_seed(arguments.seed): arguments.seed},
        )
        if arguments.output_dir is not None:
            save_report(arguments.output_dir, report, configuration)
    except (OSError, ValueError, RuntimeError) as exc:
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
