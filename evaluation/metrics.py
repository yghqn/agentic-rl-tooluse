"""Micro-aggregated diagnostics and independent task success, with raw counts.

    N = runs, C = parsed/dispatched calls, S = backend invocations.
    Selection: required-role calls / C (diagnostic only).
    Arguments: correct assessed calls / assessed known calls; execution errors
    and unknown tool names are unassessed. Calculator correctness here means
    argument validity, not inference of the expression's reasoning provenance.
    Invalid: rejected call/argument codes / C; parser errors are counted separately.
    Redundant: repeated successful calls / C.
    Averages: C / N and S / N. Success and validity: successful/valid runs / N.
    Zero denominators return None. Categories count runs, not evidence events.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent.agent import AgentRun
from agent.parser import FinalAnswer, ParseError, parse_model_output
from evaluation.failure_analysis import FAILURE_CATEGORIES, FailureAnalysis
from evaluation.verifier import VerificationResult
from tasks.schemas import Task
from tasks.validators import KNOWN_TOOLS


@dataclass(slots=True)
class EvaluationRecord:
    task: Task
    run: AgentRun
    verification: VerificationResult
    failure_analysis: FailureAnalysis


def trajectory_is_valid(run: AgentRun) -> bool:
    """Protocol/record validity, not reference-path adherence or task success."""

    if type(run.max_steps) is not int or run.max_steps <= 0 or not 0 < len(run.events) <= run.max_steps:
        return False
    final_count = 0
    for index, event in enumerate(run.events, start=1):
        if event.step_index != index or event.parse_error is not None or event.backend_error is not None:
            return False
        if (event.interaction is None) == (event.final_action is None):
            return False
        try:
            parsed = parse_model_output(event.raw_output)
        except ParseError:
            return False
        if event.interaction is not None:
            step = event.interaction
            if (
                step.step_index != index or parsed != step.tool_call
                or step.tool_call.tool_name != step.tool_result.tool_name
            ):
                return False
        else:
            final_count += 1
            if not isinstance(parsed, FinalAnswer) or parsed != event.final_action or index != len(run.events):
                return False
    if run.termination_reason == "final_answer":
        return final_count == 1
    return run.termination_reason == "max_steps" and final_count == 0 and len(run.events) == run.max_steps


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _aggregate(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    calls = [call for record in records for call in record.failure_analysis.calls]
    events = [event for record in records for event in record.run.events]
    steps = [step for record in records for step in record.run.tool_steps]
    assessed = [call for call in calls if call.argument_correct is not None]
    counts = {
        "task_count": len(records),
        "task_success_count": sum(record.verification.success for record in records),
        "task_failure_count": sum(not record.verification.success for record in records),
        "tool_call_count": len(steps),
        "known_tool_call_count": sum(step.tool_call.tool_name in KNOWN_TOOLS for step in steps),
        "agent_step_count": len(events),
        "tool_selection_correct_count": sum(call.selection_correct for call in calls),
        "argument_assessed_count": len(assessed),
        "argument_correct_count": sum(call.argument_correct is True for call in assessed),
        "invalid_tool_call_count": sum(call.invalid for call in calls),
        "redundant_tool_call_count": sum(call.redundant for call in calls),
        "parse_error_count": sum(event.parse_error is not None for event in events),
        "backend_error_count": sum(event.backend_error is not None for event in events),
        "execution_error_count": sum(step.tool_result.error_code == "EXECUTION_ERROR" for step in steps),
        "valid_trajectory_count": sum(trajectory_is_valid(record.run) for record in records),
    }
    metrics = {
        "task_success_rate": _ratio(counts["task_success_count"], counts["task_count"]),
        "tool_selection_accuracy": _ratio(counts["tool_selection_correct_count"], counts["tool_call_count"]),
        "argument_accuracy": _ratio(counts["argument_correct_count"], counts["argument_assessed_count"]),
        "invalid_tool_call_rate": _ratio(counts["invalid_tool_call_count"], counts["tool_call_count"]),
        "redundant_tool_call_rate": _ratio(counts["redundant_tool_call_count"], counts["tool_call_count"]),
        "average_tool_calls": _ratio(counts["tool_call_count"], counts["task_count"]),
        "average_agent_steps": _ratio(counts["agent_step_count"], counts["task_count"]),
        "trajectory_validity_rate": _ratio(counts["valid_trajectory_count"], counts["task_count"]),
    }
    category_counts = {category: 0 for category in FAILURE_CATEGORIES}
    for record in records:
        for category in record.failure_analysis.categories:
            category_counts[category] += 1
    return {"counts": counts, **metrics, "failure_category_counts": category_counts}


def compute_metrics(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    """All rates carry numerator/denominator; every level is present, even empty."""

    return {
        **_aggregate(records),
        "tool_selection_accuracy_is_diagnostic": True,
        "by_difficulty": {
            f"L{difficulty}": _aggregate([record for record in records if record.task.difficulty == difficulty])
            for difficulty in (1, 2, 3, 4)
        },
    }
