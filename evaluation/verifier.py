"""Independent final-answer verification. No trajectory or training reward."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from tasks.schemas import Task


NO_FINAL_ANSWER = object()


@dataclass(slots=True)
class VerificationResult:
    success: bool
    error_code: str | None = None
    failure_category: str | None = None
    normalized_answer: Any | None = None
    message: str | None = None


class VerificationConfigError(ValueError):
    """Invalid benchmark data, not an Agent error."""


def _number(value: Any) -> int | float:
    if type(value) not in (int, float):
        raise ValueError("Expected a JSON number")
    try:
        if not math.isfinite(value):
            raise ValueError("Number must be finite")
    except OverflowError as exc:
        raise ValueError("Number is too large") from exc
    return value


def _company(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty company name")
    return value.strip()


def _normalize(answer_type: str, value: Any) -> Any:
    if answer_type == "number":
        return _number(value)
    if answer_type == "company_list":
        if not isinstance(value, list):
            raise ValueError("Expected an array of company names")
        names = [_company(item) for item in value]
        if len(names) != len(set(names)):
            raise ValueError("Company names must not repeat")
        return sorted(names)
    if not isinstance(value, dict) or set(value) != {"company", "profit_margin"}:
        raise ValueError("Expected company and profit_margin fields")
    return {"company": _company(value["company"]), "profit_margin": _number(value["profit_margin"])}


def _failure(code: str, normalized: Any = None, message: str | None = None) -> VerificationResult:
    category = "premature_termination" if code == "NO_FINAL_ANSWER" else "final_answer_error"
    return VerificationResult(False, code, category, normalized, message)


def verify_final_answer(task: Task, answer: Any = NO_FINAL_ANSWER) -> VerificationResult:
    """Compare only the final answer against the privileged answer contract.

    No calls, ordering, information coverage, or efficiency enter Task Success.
    """

    spec = task.verification_spec
    contracts = {
        "number": "numeric",
        "company_list": "unordered_exact",
        "company_margin": "company_and_numeric",
    }
    if not isinstance(spec, dict):
        raise VerificationConfigError("verification_spec must be a dictionary")
    answer_type = spec.get("answer_type")
    if (
        not isinstance(answer_type, str) or answer_type not in contracts
        or spec.get("comparison") != contracts[answer_type]
    ):
        raise VerificationConfigError("Unsupported answer contract")
    try:
        expected = _normalize(answer_type, task.ground_truth)
        if answer_type != "company_list":
            rel_tol = _number(spec.get("rel_tol"))
            abs_tol = _number(spec.get("abs_tol"))
            if rel_tol < 0 or abs_tol < 0:
                raise ValueError("Tolerances must be non-negative")
    except ValueError as exc:
        raise VerificationConfigError(str(exc)) from exc

    if answer is NO_FINAL_ANSWER:
        return _failure("NO_FINAL_ANSWER", message="No final answer was submitted")
    try:
        normalized = _normalize(answer_type, answer)
    except ValueError as exc:
        return _failure("MALFORMED_FINAL_ANSWER", message=str(exc))

    if answer_type == "company_list":
        if normalized != expected:
            return _failure("COMPANY_LIST_MISMATCH", normalized)
    elif answer_type == "number":
        if not math.isclose(normalized, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            return _failure("NUMERIC_MISMATCH", normalized)
    else:
        if normalized["company"] != expected["company"]:
            return _failure("WRONG_COMPANY", normalized)
        if not math.isclose(
            normalized["profit_margin"], expected["profit_margin"],
            rel_tol=rel_tol, abs_tol=abs_tol,
        ):
            return _failure("MARGIN_MISMATCH", normalized)
    return VerificationResult(True, normalized_answer=normalized)
