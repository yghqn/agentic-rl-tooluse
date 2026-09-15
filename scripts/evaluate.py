"""Evaluation harness and offline replay CLI, with no training/model backend.

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
    It must create independent backend state. All benchmark tasks are validated
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
    parser = argparse.ArgumentParser(description="Offline Prompt Agent pipeline replay (no real model)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-per-level", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--responses", type=Path, required=True, help="Per-task JSONL model-output fixtures")
    arguments = parser.parse_args(argv)
    try:
        if arguments.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        tasks = TaskGenerator(arguments.seed).generate_tasks(arguments.count_per_level)
        scripts = load_scripted_responses(arguments.responses)
        expected_ids = {task.task_id for task in tasks}
        if set(scripts) != expected_ids:
            raise ValueError("Replay task IDs must exactly match the generated benchmark")
        report = evaluate_tasks(
            tasks,
            lambda task_id: PromptAgent(ScriptedBackend(scripts[task_id]), arguments.max_steps),
            {environment_id_for_seed(arguments.seed): arguments.seed},
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = {
        "mode": "scripted_replay",
        "configuration": {
            "seed": arguments.seed,
            "count_per_level": arguments.count_per_level,
            "max_steps": arguments.max_steps,
            "order": "difficulty_then_task_id",
        },
        **report.to_dict(),
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
