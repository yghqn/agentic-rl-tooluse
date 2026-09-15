"""Tests for reproducible generation and trajectory-independent validation."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import re
from typing import Any

import pytest

from environment.database import SyntheticCompanyDatabase
from environment.tools import calculator
from tasks.generator import TaskGenerator, agent_task_view
from tasks.validators import TaskValidationError, validate_task


ALL_TOOLS = ["lookup_company", "calculator", "list_companies"]
COORDINATES = [(1, 0), (2, 0), (2, 1), (3, 0), (4, 0)]


def test_generation_is_deterministic() -> None:
    first = TaskGenerator(seed=42)
    second = TaskGenerator(seed=42)

    assert first.generate_tasks(5) == second.generate_tasks(5)
    assert first.generate_tasks(5) == first.generate_tasks(5)


def test_ids_are_stable_unique_and_independent_of_call_order() -> None:
    generator = TaskGenerator(seed=42)
    expected = generator.generate_task(3, 7)
    generator.generate_task(4, 100)
    generator.generate_task(1, 2)

    assert expected.task_id == "task-v1-seed-42-l3-index-7"
    assert expected == generator.generate_task(3, 7)
    assert expected == TaskGenerator(seed=42).generate_task(3, 7)
    tasks = generator.generate_tasks(20)
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert expected.task_id != TaskGenerator(seed=43).generate_task(3, 7).task_id


@pytest.mark.parametrize(
    ("difficulty", "index", "task_type"),
    [
        (1, 0, "single_retrieval"),
        (2, 0, "list_companies"),
        (2, 1, "arithmetic"),
        (3, 0, "profit_margin"),
        (4, 0, "highest_profit_margin"),
    ],
)
def test_each_task_type_and_tool_set(
    difficulty: int, index: int, task_type: str
) -> None:
    task = TaskGenerator(seed=42).generate_task(difficulty, index)

    assert task.difficulty == difficulty
    assert task.verification_spec["task_type"] == task_type
    assert task.available_tools == ALL_TOOLS
    assert task.environment_id == "company-db-v1-seed-42"
    validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize(("difficulty", "index"), COORDINATES)
def test_ground_truth_correctness(difficulty: int, index: int) -> None:
    task = TaskGenerator(seed=42).generate_task(difficulty, index)
    database = SyntheticCompanyDatabase(seed=42)
    metadata = task.metadata

    if difficulty == 1:
        assert task.ground_truth == database.lookup(metadata["company"], metadata["field"])
    elif difficulty == 2 and index == 0:
        assert task.ground_truth == database.list_companies()
    elif difficulty == 2:
        assert task.ground_truth == calculator(metadata["expression"])
    elif difficulty == 3:
        company = metadata["company"]
        expected = database.lookup(company, "profit") / database.lookup(company, "revenue")
        assert task.ground_truth == pytest.approx(expected)
    else:
        ratios = {
            company: Fraction(
                database.lookup(company, "profit"), database.lookup(company, "revenue")
            )
            for company in metadata["companies"]
        }
        winner = sorted(ratios, key=lambda company: (-ratios[company], company))[0]
        assert task.ground_truth["company"] == winner
        assert task.ground_truth["profit_margin"] == pytest.approx(float(ratios[winner]))


def test_different_seeds_change_environment_values_and_task_structures() -> None:
    first = TaskGenerator(seed=1).generate_tasks(10)
    second = TaskGenerator(seed=2).generate_tasks(10)

    assert {task.task_id for task in first}.isdisjoint(task.task_id for task in second)
    assert all(a.environment_id != b.environment_id for a, b in zip(first, second))
    assert any(a.metadata != b.metadata for a, b in zip(first, second))
    assert any(a.ground_truth != b.ground_truth for a, b in zip(first, second))
    # The company roster does not depend on seed; this task may stay unchanged.
    assert first[10].question == second[10].question
    assert first[10].ground_truth == second[10].ground_truth


def _assert_no_numeric_metadata(value: Any) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_numeric_metadata(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_numeric_metadata(nested)
    else:
        assert isinstance(value, str)
        assert not re.search(r"\d", value)


def test_question_and_metadata_do_not_include_hidden_numbers() -> None:
    for task in TaskGenerator(seed=42).generate_tasks(20):
        if task.verification_spec["task_type"] == "arithmetic":
            # These operands are intentionally public and are not company data.
            expression = task.metadata["expression"]
            assert set(task.metadata) == {"expression"}
            assert re.fullmatch(r"[1-9]\d? [+-/*] [1-9]\d?", expression)
            assert task.question == f"Calculate {expression}."
        else:
            assert not re.search(r"\d", task.question)
            _assert_no_numeric_metadata(task.metadata)


def test_public_content_is_independent_of_hidden_attribute_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_lookup = SyntheticCompanyDatabase.lookup
    before = TaskGenerator(seed=42).generate_tasks(10)

    def changed_lookup(
        database: SyntheticCompanyDatabase, company: str, field: str
    ) -> int | float:
        return original_lookup(database, company, field) + {
            "profit": 1, "revenue": 11, "employees": 17, "growth_rate": 0.01
        }[field]

    monkeypatch.setattr(SyntheticCompanyDatabase, "lookup", changed_lookup)
    after = TaskGenerator(seed=42).generate_tasks(10)

    assert [task.question for task in before] == [task.question for task in after]
    assert [task.metadata for task in before] == [task.metadata for task in after]
    assert any(a.ground_truth != b.ground_truth for a, b in zip(before, after))
    for a, b in zip(before, after):
        if a.difficulty == 2:
            assert a.ground_truth == b.ground_truth


@pytest.mark.parametrize(("difficulty", "index"), COORDINATES)
def test_agent_view_excludes_privileged_fields(difficulty: int, index: int) -> None:
    task = TaskGenerator(seed=42).generate_task(difficulty, index)

    visible = agent_task_view(task)

    assert visible == {"question": task.question, "available_tools": ALL_TOOLS}
    assert not {"ground_truth", "verification_spec", "metadata"}.intersection(visible)
    visible["available_tools"].clear()
    assert task.available_tools == ALL_TOOLS


@pytest.mark.parametrize("difficulty", [0, 5, -1, True, 1.0, "1", None])
def test_invalid_difficulty_is_rejected(difficulty: Any) -> None:
    with pytest.raises(ValueError):
        TaskGenerator(seed=42).generate_task(difficulty)
    task = replace(TaskGenerator(seed=42).generate_task(1), difficulty=difficulty)
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize("value", [-1, True, 1.0, "1", None])
def test_invalid_generation_indices_and_counts_are_rejected(value: Any) -> None:
    generator = TaskGenerator(seed=42)
    with pytest.raises(ValueError):
        generator.generate_task(1, value)
    with pytest.raises(ValueError):
        generator.generate_tasks(value)


def test_empty_batch_is_allowed() -> None:
    assert TaskGenerator(seed=42).generate_tasks(0) == []


@pytest.mark.parametrize("name", ["task_id", "question", "environment_id"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_required_strings_are_rejected(name: str, value: Any) -> None:
    task = replace(TaskGenerator(seed=42).generate_task(1), **{name: value})
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize(
    ("difficulty", "metadata"),
    [
        (1, {"company": "Unknown Corp", "field": "revenue"}),
        (1, {"company": "Company A", "field": "market_cap"}),
        (1, {"company": [], "field": "revenue"}),
        (1, {"company": "Company A"}),
        (1, {"company": "Company A", "field": "revenue", "value": 100}),
        (3, {"company": "Unknown Corp", "fields": ["profit", "revenue"]}),
        (3, {"company": "Company A", "fields": ["profit", "employees"]}),
        (4, {"companies": ["Company A", "Company B", "Unknown Corp"], "fields": ["profit", "revenue"]}),
        (4, {"companies": ["Company A", "Company A", "Company B"], "fields": ["profit", "revenue"]}),
        (4, {"companies": ["Company A", "Company B"], "fields": ["profit", "revenue"]}),
        (1, None),
    ],
)
def test_invalid_metadata_is_rejected(difficulty: int, metadata: Any) -> None:
    task = replace(TaskGenerator(seed=42).generate_task(difficulty), metadata=metadata)
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize(
    ("difficulty", "index", "required_tool"),
    [(1, 0, "lookup_company"), (2, 0, "list_companies"), (2, 1, "calculator"),
     (3, 0, "lookup_company"), (3, 0, "calculator"),
     (4, 0, "lookup_company"), (4, 0, "calculator")],
)
def test_insufficient_tool_sets_are_rejected(
    difficulty: int, index: int, required_tool: str
) -> None:
    task = TaskGenerator(seed=42).generate_task(difficulty, index)
    task.available_tools.remove(required_tool)
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize(
    ("difficulty", "index", "tools"),
    [(1, 0, ["lookup_company"]), (2, 0, ["list_companies"]),
     (2, 1, ["calculator"]), (3, 0, ["lookup_company", "calculator"]),
     (4, 0, ["lookup_company", "calculator"])],
)
def test_validator_checks_required_tools_independently_of_exposed_tools(
    difficulty: int, index: int, tools: list[str]
) -> None:
    task = replace(
        TaskGenerator(seed=42).generate_task(difficulty, index), available_tools=tools
    )
    validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize("tools", [["unknown"], ALL_TOOLS + ["calculator"], [123], None])
def test_unknown_duplicate_or_malformed_tools_are_rejected(tools: Any) -> None:
    task = replace(TaskGenerator(seed=42).generate_task(1), available_tools=tools)
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize(("difficulty", "index"), COORDINATES)
def test_incorrect_ground_truth_is_rejected(difficulty: int, index: int) -> None:
    task = TaskGenerator(seed=42).generate_task(difficulty, index)
    if difficulty == 4:
        incorrect = {**task.ground_truth, "profit_margin": 999.0}
    elif difficulty == 2 and index == 0:
        incorrect = task.ground_truth[:-1]
    else:
        incorrect = task.ground_truth + 1
    with pytest.raises(TaskValidationError):
        validate_task(replace(task, ground_truth=incorrect), SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize("answer", [True, "10", float("nan"), float("inf"), None, []])
def test_malformed_numeric_ground_truth_is_rejected(answer: Any) -> None:
    task = replace(TaskGenerator(seed=42).generate_task(3), ground_truth=answer)
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


def test_incorrect_winning_company_is_rejected() -> None:
    task = TaskGenerator(seed=42).generate_task(4)
    other = next(company for company in task.metadata["companies"] if company != task.ground_truth["company"])
    task.ground_truth["company"] = other
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize("expression", ["1 / 0", "2 ** 3", "x + 1", "", None])
def test_invalid_arithmetic_metadata_is_rejected(expression: Any) -> None:
    task = replace(TaskGenerator(seed=42).generate_task(2, 1), metadata={"expression": expression})
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize("change", ["unknown_type", "difficulty", "formula", "tolerance", "trajectory"])
def test_inconsistent_verification_spec_is_rejected(change: str) -> None:
    task = TaskGenerator(seed=42).generate_task(3)
    if change == "unknown_type":
        task.verification_spec["task_type"] = "unknown"
    elif change == "difficulty":
        task.difficulty = 4
    elif change == "formula":
        task.verification_spec["formula"] = "revenue / profit"
    elif change == "tolerance":
        task.verification_spec["rel_tol"] = 1.0
    else:
        task.verification_spec["reference_trajectory"] = []
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


@pytest.mark.parametrize("spec", [None, [], {}, {"task_type": []}])
def test_malformed_verification_spec_is_rejected(spec: Any) -> None:
    task = replace(TaskGenerator(seed=42).generate_task(1), verification_spec=spec)
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=42))


def test_wrong_environment_seed_is_rejected() -> None:
    task = TaskGenerator(seed=42).generate_task(1)
    with pytest.raises(TaskValidationError):
        validate_task(task, SyntheticCompanyDatabase(seed=43))


@pytest.mark.parametrize(("difficulty", "index"), COORDINATES)
def test_questions_can_be_paraphrased(difficulty: int, index: int) -> None:
    task = TaskGenerator(seed=42).generate_task(difficulty, index)
    task.question = "Please solve this task: " + task.question
    validate_task(task, SyntheticCompanyDatabase(seed=42))


def test_validation_does_not_require_field_or_company_order() -> None:
    task = TaskGenerator(seed=42).generate_task(4)
    task.metadata["companies"].reverse()
    task.metadata["fields"].reverse()
    validate_task(task, SyntheticCompanyDatabase(seed=42))
    assert not {"trajectory", "reference_trajectory", "tool_calls"}.intersection(task.verification_spec)


def test_company_list_order_is_not_required() -> None:
    task = TaskGenerator(seed=42).generate_task(2, 0)
    task.ground_truth.reverse()
    validate_task(task, SyntheticCompanyDatabase(seed=42))


def test_l4_ties_use_alphabetical_first_company(monkeypatch: pytest.MonkeyPatch) -> None:
    def tied_lookup(
        database: SyntheticCompanyDatabase, company: str, field: str
    ) -> int:
        return {"profit": 10, "revenue": 100}[field]

    monkeypatch.setattr(SyntheticCompanyDatabase, "lookup", tied_lookup)
    task = TaskGenerator(seed=42).generate_task(4)

    assert task.ground_truth == {
        "company": min(task.metadata["companies"]), "profit_margin": 0.1
    }
    validate_task(task, SyntheticCompanyDatabase(seed=42))
