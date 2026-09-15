"""Validate task semantics and answers, not Agent trajectories."""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Any

from environment.database import SUPPORTED_FIELDS, SyntheticCompanyDatabase
from environment.tools import InvalidExpressionError, calculator
from tasks.schemas import Task


KNOWN_TOOLS: tuple[str, ...] = (
    "lookup_company", "calculator", "list_companies"
)
REQUIRED_TOOLS: dict[str, frozenset[str]] = {
    "single_retrieval": frozenset({"lookup_company"}),
    "list_companies": frozenset({"list_companies"}),
    "arithmetic": frozenset({"calculator"}),
    "profit_margin": frozenset({"lookup_company", "calculator"}),
    "highest_profit_margin": frozenset({"lookup_company", "calculator"}),
}
_DIFFICULTIES = {
    "single_retrieval": 1,
    "list_companies": 2,
    "arithmetic": 2,
    "profit_margin": 3,
    "highest_profit_margin": 4,
}
_METADATA_KEYS = {
    "single_retrieval": {"company", "field"},
    "list_companies": set(),
    "arithmetic": {"expression"},
    "profit_margin": {"company", "fields"},
    "highest_profit_margin": {"companies", "fields"},
}
REL_TOL = 1e-9
ABS_TOL = 1e-12


class TaskValidationError(ValueError):
    """A task is malformed or inconsistent with its environment."""


def environment_id_for_seed(seed: int) -> str:
    """Identify the current synthetic database version and seed."""

    return f"company-db-v1-seed-{seed}"


def verification_spec_for_type(task_type: str) -> dict[str, Any]:
    """Describe the answer contract without specifying any tool-call order."""

    if task_type not in REQUIRED_TOOLS:
        raise TaskValidationError(f"Unknown task type: {task_type}")
    if task_type == "list_companies":
        return {
            "task_type": task_type,
            "answer_type": "company_list",
            "comparison": "unordered_exact",
        }

    spec: dict[str, Any] = {
        "task_type": task_type,
        "answer_type": "number",
        "comparison": "numeric",
        "rel_tol": REL_TOL,
        "abs_tol": ABS_TOL,
    }
    if task_type in {"profit_margin", "highest_profit_margin"}:
        spec.update(formula="profit / revenue", unit="ratio")
    if task_type == "highest_profit_margin":
        spec.update(
            answer_type="company_margin",
            comparison="company_and_numeric",
            objective="maximum",
            tie_break="alphabetical_first",
        )
    return spec


def validate_task(task: Task, database: SyntheticCompanyDatabase) -> None:
    """Raise TaskValidationError unless the task has a recomputable answer.

    Questions may be paraphrased. This validates the structured task contract,
    not the meaning of arbitrary natural language or a reference trajectory.
    """

    if not isinstance(task, Task):
        raise TaskValidationError("task must be a Task")
    for name in ("task_id", "question", "environment_id"):
        _require_nonempty_string(getattr(task, name), name)
    if type(task.difficulty) is not int or task.difficulty not in {1, 2, 3, 4}:
        raise TaskValidationError("difficulty must be one of 1, 2, 3, 4")
    if task.environment_id != environment_id_for_seed(database.seed):
        raise TaskValidationError("environment_id does not match the database seed")
    if not isinstance(task.verification_spec, dict):
        raise TaskValidationError("verification_spec must be a dictionary")

    task_type = task.verification_spec.get("task_type")
    if not isinstance(task_type, str) or task_type not in REQUIRED_TOOLS:
        raise TaskValidationError("Unknown or missing task_type")
    if task.difficulty != _DIFFICULTIES[task_type]:
        raise TaskValidationError("difficulty does not match task_type")
    if task.verification_spec != verification_spec_for_type(task_type):
        raise TaskValidationError("verification_spec is inconsistent with task_type")

    _validate_tools(task.available_tools, task_type)
    _validate_metadata(task.metadata, task_type, database)
    expected = _recompute_ground_truth(task.metadata, task_type, database)
    _validate_ground_truth(task.ground_truth, expected, task_type)


def _require_nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TaskValidationError(f"{name} must be a non-empty string")


def _validate_tools(tools: Any, task_type: str) -> None:
    if not isinstance(tools, list) or any(
        not isinstance(tool, str) or tool not in KNOWN_TOOLS for tool in tools
    ):
        raise TaskValidationError("available_tools must contain only known tool names")
    if len(tools) != len(set(tools)):
        raise TaskValidationError("available_tools must not contain duplicates")
    if not REQUIRED_TOOLS[task_type].issubset(tools):
        raise TaskValidationError("available_tools is missing required tools")


def _validate_metadata(
    metadata: Any, task_type: str, database: SyntheticCompanyDatabase
) -> None:
    # Type-specific allowlists keep hidden numbers out of structural metadata.
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS[task_type]:
        raise TaskValidationError("metadata has missing or unsupported keys")

    valid_companies = database.list_companies()
    if "company" in metadata:
        company = metadata["company"]
        if not isinstance(company, str) or company not in valid_companies:
            raise TaskValidationError("metadata references an invalid company")
    if "companies" in metadata:
        companies = metadata["companies"]
        if (
            not isinstance(companies, list)
            or len(companies) != 3
            or any(
                not isinstance(company, str) or company not in valid_companies
                for company in companies
            )
            or len(set(companies)) != 3
        ):
            raise TaskValidationError("metadata must reference three distinct valid companies")
    if "field" in metadata:
        field = metadata["field"]
        if not isinstance(field, str) or field not in SUPPORTED_FIELDS:
            raise TaskValidationError("metadata references an invalid field")
    if "fields" in metadata:
        fields = metadata["fields"]
        if (
            not isinstance(fields, list)
            or len(fields) != 2
            or any(not isinstance(field, str) for field in fields)
            or set(fields) != {"profit", "revenue"}
        ):
            raise TaskValidationError("profit margin requires profit and revenue fields")
    if "expression" in metadata:
        _require_nonempty_string(metadata["expression"], "expression")


def _recompute_ground_truth(
    metadata: dict[str, Any], task_type: str, database: SyntheticCompanyDatabase
) -> Any:
    if task_type == "single_retrieval":
        return database.lookup(metadata["company"], metadata["field"])
    if task_type == "list_companies":
        return database.list_companies()
    if task_type == "arithmetic":
        try:
            return calculator(metadata["expression"])
        except InvalidExpressionError as exc:
            raise TaskValidationError("metadata contains an invalid expression") from exc

    def margin(company: str) -> Fraction:
        try:
            return Fraction(
                database.lookup(company, "profit"),
                database.lookup(company, "revenue"),
            )
        except (TypeError, ZeroDivisionError) as exc:
            raise TaskValidationError("environment cannot provide a valid profit margin") from exc

    if task_type == "profit_margin":
        return float(margin(metadata["company"]))
    margins = {company: margin(company) for company in metadata["companies"]}
    winner = min(margins, key=lambda company: (-margins[company], company))
    return {"company": winner, "profit_margin": float(margins[winner])}


def _numbers_match(actual: Any, expected: int | float) -> bool:
    if type(actual) not in (int, float):
        return False
    try:
        return math.isfinite(actual) and math.isclose(
            actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL
        )
    except (OverflowError, TypeError):
        return False


def _validate_ground_truth(actual: Any, expected: Any, task_type: str) -> None:
    if task_type == "list_companies":
        valid = (
            isinstance(actual, list)
            and all(isinstance(company, str) for company in actual)
            and len(actual) == len(expected)
            and set(actual) == set(expected)
        )
    elif task_type == "highest_profit_margin":
        valid = (
            isinstance(actual, dict)
            and set(actual) == {"company", "profit_margin"}
            and actual["company"] == expected["company"]
            and _numbers_match(actual["profit_margin"], expected["profit_margin"])
        )
    else:
        valid = _numbers_match(actual, expected)
    if not valid:
        raise TaskValidationError("ground_truth does not match the environment/task semantics")
