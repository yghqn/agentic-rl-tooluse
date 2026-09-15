"""Observable diagnostics, never inferred model intentions or reward signals."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any

from agent.agent import AgentRun
from evaluation.verifier import VerificationResult
from tasks.schemas import Task, ToolCall
from tasks.validators import KNOWN_TOOLS, REQUIRED_TOOLS


FAILURE_CATEGORIES: tuple[str, ...] = (
    "wrong_tool", "wrong_argument", "invalid_tool_call", "missing_information",
    "planning_failure", "calculation_failure", "redundant_tool_use", "loop",
    "premature_termination", "final_answer_error", "unclassified_failure",
)
INVALID_CALL_CODES = frozenset({
    "UNKNOWN_TOOL", "MISSING_ARGUMENT", "INVALID_ARGUMENT", "UNKNOWN_COMPANY",
    "UNKNOWN_FIELD", "INVALID_EXPRESSION",
})


@dataclass(slots=True)
class FailureEvidence:
    category: str
    reason: str
    step_indices: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CallAssessment:
    step_index: int
    selection_correct: bool
    argument_correct: bool | None
    invalid: bool
    redundant: bool


@dataclass(slots=True)
class FailureAnalysis:
    evidence: list[FailureEvidence] = field(default_factory=list)
    calls: list[CallAssessment] = field(default_factory=list)

    @property
    def categories(self) -> list[str]:
        return sorted({item.category for item in self.evidence})


def _target_queries(task: Task) -> set[tuple[str, str]]:
    task_type = task.verification_spec["task_type"]
    if task_type == "single_retrieval":
        return {(task.metadata["company"], task.metadata["field"])}
    if task_type in {"profit_margin", "highest_profit_margin"}:
        companies = (
            [task.metadata["company"]] if task_type == "profit_margin"
            else task.metadata["companies"]
        )
        return {(company, field) for company in companies for field in ("profit", "revenue")}
    return set()


def _call_key(call: ToolCall) -> str:
    arguments = dict(call.arguments)
    if isinstance(arguments.get("expression"), str):
        arguments["expression"] = arguments["expression"].strip()
    return json.dumps([call.tool_name, arguments], sort_keys=True, allow_nan=False)


def _matches(task: Task, actual: Any, expected: Any) -> bool:
    if type(actual) not in (int, float) or type(expected) not in (int, float):
        return False
    try:
        return math.isfinite(actual) and math.isfinite(expected) and math.isclose(
            actual, expected,
            rel_tol=task.verification_spec["rel_tol"],
            abs_tol=task.verification_spec["abs_tol"],
        )
    except (OverflowError, ValueError):
        return False


def analyze_failures(
    task: Task, run: AgentRun, verification: VerificationResult
) -> FailureAnalysis:
    """Analyze validated tasks using actions, observations and answer mismatches.

    Tags may coexist and may describe inefficient successful runs. They never
    override the independent Verifier. Missing calculator calls alone are NOT
    evidence of planning failure: alternative solutions need not match a path.
    """

    analysis = FailureAnalysis()
    task_type = task.verification_spec["task_type"]
    required = REQUIRED_TOOLS[task_type]
    targets = _target_queries(task)
    observed: dict[tuple[str, str], tuple[Any, int]] = {}
    seen_successful: set[str] = set()
    previous_key: str | None = None
    consecutive: list[int] = []

    def add(category: str, reason: str, indices: list[int] | None = None, **details: Any) -> None:
        analysis.evidence.append(FailureEvidence(category, reason, indices or [], details))

    for event in run.events:
        if event.parse_error is not None:
            add("invalid_tool_call", "action_protocol_error", [event.step_index],
                error_code=event.parse_error.error_code)
        if event.interaction is None:
            previous_key = None
            consecutive = []
            continue
        step = event.interaction
        call, result = step.tool_call, step.tool_result
        selection_correct = call.tool_name in required
        invalid = result.error_code in INVALID_CALL_CODES
        argument_correct: bool | None = None
        if call.tool_name in KNOWN_TOOLS and result.error_code != "EXECUTION_ERROR":
            argument_correct = result.success
            if result.success and call.tool_name == "lookup_company" and targets:
                argument_correct = (call.arguments["company"], call.arguments["field"]) in targets

        key = _call_key(call)
        redundant = result.success and key in seen_successful
        analysis.calls.append(CallAssessment(
            step.step_index, selection_correct, argument_correct, invalid, redundant
        ))
        if not selection_correct:
            add("wrong_tool", "tool_not_in_required_role_set", [step.step_index], tool_name=call.tool_name)
        if argument_correct is False:
            add("wrong_argument", "invalid_or_off_target_arguments", [step.step_index],
                error_code=result.error_code, tool_name=call.tool_name)
        if invalid:
            add("invalid_tool_call", "environment_rejected_call", [step.step_index], error_code=result.error_code)
        if result.error_code == "INVALID_EXPRESSION":
            add("calculation_failure", "calculator_rejected_expression", [step.step_index])
        if redundant:
            add("redundant_tool_use", "repeated_successful_call", [step.step_index])
        if result.success:
            seen_successful.add(key)
            if call.tool_name == "lookup_company":
                observed[(call.arguments["company"], call.arguments["field"])] = (result.output, step.step_index)

        consecutive = consecutive + [step.step_index] if key == previous_key else [step.step_index]
        previous_key = key
        if len(consecutive) == 3:
            add("loop", "three_consecutive_identical_calls", list(consecutive))

    if verification.success:
        return analysis

    missing = targets - observed.keys()
    if task_type == "list_companies":
        missing_roster = not any(
            step.tool_call.tool_name == "list_companies" and step.tool_result.success
            for step in run.tool_steps
        )
    else:
        missing_roster = False
    if missing or missing_roster:
        details = {"missing_queries": [list(query) for query in sorted(missing)]}
        if missing_roster:
            details["missing_company_list"] = True
        add("missing_information", "required_observations_not_present", **details)
        if run.has_final_answer:
            add("premature_termination", "failed_answer_before_required_observations", **details)
    if not run.has_final_answer:
        add("premature_termination", "no_final_answer", termination_reason=run.termination_reason)
    if run.has_final_answer and verification.failure_category == "final_answer_error":
        add("final_answer_error", "independent_verifier_rejected_answer", error_code=verification.error_code)

    normalized = verification.normalized_answer
    if task_type in {"profit_margin", "highest_profit_margin"} and not missing:
        company = task.metadata["company"] if task_type == "profit_margin" else (
            normalized.get("company") if isinstance(normalized, dict) else None
        )
        if (company, "profit") in observed and (company, "revenue") in observed:
            profit, profit_step = observed[(company, "profit")]
            revenue, revenue_step = observed[(company, "revenue")]
            if type(profit) in (int, float) and type(revenue) in (int, float) and revenue != 0:
                ratio = profit / revenue
                actual = normalized if task_type == "profit_margin" else normalized["profit_margin"]
                if type(actual) in (int, float) and not _matches(task, actual, ratio):
                    add("calculation_failure", "answer_ratio_differs_from_observed_ratio",
                        [profit_step, revenue_step], actual=actual, observed_ratio=ratio)
                elif task_type == "highest_profit_margin" and verification.error_code == "WRONG_COMPANY":
                    add("planning_failure", "nonwinning_company_with_its_observed_margin",
                        [profit_step, revenue_step], selected_company=company)
    elif task_type == "arithmetic" and verification.error_code == "NUMERIC_MISMATCH":
        for step in run.tool_steps:
            if (
                step.tool_call.tool_name == "calculator" and step.tool_result.success
                and _matches(task, normalized, step.tool_result.output)
            ):
                add("calculation_failure", "submitted_calculator_result_differs_from_target",
                    [step.step_index], calculator_output=step.tool_result.output)
                break

    if not analysis.categories:
        add("unclassified_failure", "no_supported_deterministic_rule_matched",
            verification_error_code=verification.error_code)
    return analysis
