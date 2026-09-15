"""Tools exposed by the deterministic V1 environment."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from environment.database import SyntheticCompanyDatabase


MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 64
MAX_ABSOLUTE_VALUE = 1_000_000_000_000_000

_DEFAULT_DATABASE = SyntheticCompanyDatabase(seed=0)
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[int | float, int | float], int | float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[int | float], int | float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class InvalidToolArgumentError(ValueError):
    """Raised when a tool argument has an invalid type or value."""


class InvalidExpressionError(ValueError):
    """Raised when an expression is invalid or outside calculator limits."""


def lookup_company(
    company: str,
    field: str,
    *,
    database: SyntheticCompanyDatabase | None = None,
) -> int | float:
    """Look up one field without exposing the complete company record."""

    if not isinstance(company, str) or not company.strip():
        raise InvalidToolArgumentError("company must be a non-empty string")
    if not isinstance(field, str) or not field.strip():
        raise InvalidToolArgumentError("field must be a non-empty string")

    active_database = database if database is not None else _DEFAULT_DATABASE
    return active_database.lookup(company, field)


def list_companies(
    *, database: SyntheticCompanyDatabase | None = None
) -> list[str]:
    """List available company names without returning hidden attributes."""

    active_database = database if database is not None else _DEFAULT_DATABASE
    return active_database.list_companies()


def calculator(expression: str) -> int | float:
    """Safely evaluate a small arithmetic expression without using eval()."""

    if not isinstance(expression, str) or not expression.strip():
        raise InvalidExpressionError("expression must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise InvalidExpressionError("expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise InvalidExpressionError("expression has invalid syntax") from exc

    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise InvalidExpressionError("expression is too complex")

    try:
        return _evaluate_node(tree)
    except InvalidExpressionError:
        raise
    except (ArithmeticError, OverflowError, RecursionError) as exc:
        raise InvalidExpressionError("expression could not be evaluated") from exc


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise InvalidExpressionError("only numeric literals are supported")
        return _validate_number(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        operation = _BINARY_OPERATORS[type(node.op)]
        return _validate_number(operation(left, right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _evaluate_node(node.operand)
        operation = _UNARY_OPERATORS[type(node.op)]
        return _validate_number(operation(operand))

    raise InvalidExpressionError("expression contains an unsupported operation")


def _validate_number(value: int | float) -> int | float:
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidExpressionError("expression result must be finite")
    if abs(value) > MAX_ABSOLUTE_VALUE:
        raise InvalidExpressionError("expression result is too large")
    return value
