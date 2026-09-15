"""Unit tests for the deterministic V1 tool environment."""

from __future__ import annotations

import pytest

from environment.database import (
    COMPANY_NAMES,
    SyntheticCompanyDatabase,
    UnknownCompanyError,
    UnknownFieldError,
)
from environment.environment import Environment
from environment.tools import (
    InvalidExpressionError,
    InvalidToolArgumentError,
    calculator,
    list_companies,
    lookup_company,
)
from tasks.schemas import ToolCall, ToolErrorCode


def test_list_companies_returns_names_only() -> None:
    database = SyntheticCompanyDatabase(seed=7)

    companies = list_companies(database=database)

    assert companies == list(COMPANY_NAMES)
    assert all(isinstance(company, str) for company in companies)


def test_list_companies_returns_a_fresh_list() -> None:
    database = SyntheticCompanyDatabase(seed=7)
    first_result = list_companies(database=database)

    first_result.pop()

    assert list_companies(database=database) == list(COMPANY_NAMES)


def test_lookup_company_returns_each_supported_field() -> None:
    database = SyntheticCompanyDatabase(seed=7)

    revenue = lookup_company("Company A", "revenue", database=database)
    profit = lookup_company("Company A", "profit", database=database)
    employees = lookup_company("Company A", "employees", database=database)
    growth_rate = lookup_company("Company A", "growth_rate", database=database)

    assert isinstance(revenue, int)
    assert isinstance(profit, int)
    assert isinstance(employees, int)
    assert isinstance(growth_rate, float)
    assert 0 < profit < revenue


@pytest.mark.parametrize("company", ["Unknown Corp", "", 123, None])
def test_lookup_company_rejects_invalid_company(company: object) -> None:
    database = SyntheticCompanyDatabase(seed=7)
    expected_exception = (
        UnknownCompanyError if company == "Unknown Corp" else InvalidToolArgumentError
    )

    with pytest.raises(expected_exception):
        lookup_company(company, "revenue", database=database)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["market_cap", "", 123, None])
def test_lookup_company_rejects_invalid_field(field: object) -> None:
    database = SyntheticCompanyDatabase(seed=7)
    expected_exception = (
        UnknownFieldError if field == "market_cap" else InvalidToolArgumentError
    )

    with pytest.raises(expected_exception):
        lookup_company("Company A", field, database=database)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("-8 / +2", -4.0),
        ("1.5 + 2.25", 3.75),
    ],
)
def test_calculator_evaluates_supported_arithmetic(
    expression: str, expected: int | float
) -> None:
    assert calculator(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "2 ** 3",
        "7 // 2",
        "7 % 2",
        "abs(-1)",
        "unknown + 1",
        "__import__('os').getcwd()",
        "1 / 0",
    ],
)
def test_calculator_rejects_invalid_expressions(expression: str) -> None:
    with pytest.raises(InvalidExpressionError):
        calculator(expression)


def test_same_seed_produces_identical_company_data() -> None:
    first = SyntheticCompanyDatabase(seed=42)
    second = SyntheticCompanyDatabase(seed=42)

    first_values = [
        lookup_company(company, field, database=first)
        for company in COMPANY_NAMES
        for field in ("revenue", "profit", "employees", "growth_rate")
    ]
    second_values = [
        lookup_company(company, field, database=second)
        for company in COMPANY_NAMES
        for field in ("revenue", "profit", "employees", "growth_rate")
    ]

    assert first_values == second_values


def test_different_seeds_change_data_but_not_company_names() -> None:
    first = SyntheticCompanyDatabase(seed=1)
    second = SyntheticCompanyDatabase(seed=2)

    assert list_companies(database=first) == list_companies(database=second)
    assert lookup_company("Company A", "revenue", database=first) != lookup_company(
        "Company A", "revenue", database=second
    )


def test_environment_executes_each_registered_tool() -> None:
    environment = Environment(seed=7)

    companies_result = environment.execute_tool_call(ToolCall("list_companies"))
    lookup_result = environment.execute_tool_call(
        ToolCall("lookup_company", {"company": "Company A", "field": "revenue"})
    )
    calculator_result = environment.execute_tool_call(
        ToolCall("calculator", {"expression": "6 * (4 - 1)"})
    )

    assert companies_result.success is True
    assert companies_result.output == list(COMPANY_NAMES)
    assert lookup_result.success is True
    assert isinstance(lookup_result.output, int)
    assert calculator_result.success is True
    assert calculator_result.output == 18


@pytest.mark.parametrize(
    ("tool_call", "expected_code"),
    [
        (ToolCall("not_a_tool"), ToolErrorCode.UNKNOWN_TOOL),
        (ToolCall("calculator"), ToolErrorCode.MISSING_ARGUMENT),
        (
            ToolCall("calculator", {"expression": "2 ** 8"}),
            ToolErrorCode.INVALID_EXPRESSION,
        ),
        (
            ToolCall(
                "lookup_company",
                {"company": "Unknown Corp", "field": "revenue"},
            ),
            ToolErrorCode.UNKNOWN_COMPANY,
        ),
        (
            ToolCall(
                "lookup_company",
                {"company": "Company A", "field": "market_cap"},
            ),
            ToolErrorCode.UNKNOWN_FIELD,
        ),
        (
            ToolCall("list_companies", {"unexpected": True}),
            ToolErrorCode.INVALID_ARGUMENT,
        ),
    ],
)
def test_environment_returns_stable_error_codes(
    tool_call: ToolCall, expected_code: ToolErrorCode
) -> None:
    result = Environment(seed=7).execute_tool_call(tool_call)

    assert result.success is False
    assert result.output is None
    assert result.error_code == expected_code.value
    assert result.error_message


def test_environment_converts_unexpected_exception_to_execution_error() -> None:
    environment = Environment(seed=7)

    def broken_tool() -> None:
        raise RuntimeError("boom")

    environment._tool_handlers["list_companies"] = broken_tool

    result = environment.execute_tool_call(
        ToolCall("list_companies")
    )

    assert result.success is False
    assert result.output is None
    assert result.error_code == ToolErrorCode.EXECUTION_ERROR.value
    assert result.error_message == "boom"