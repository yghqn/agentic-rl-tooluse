"""Unified execution boundary for V1 tool calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from environment.database import (
    SyntheticCompanyDatabase,
    UnknownCompanyError,
    UnknownFieldError,
)
from environment.tools import (
    InvalidExpressionError,
    InvalidToolArgumentError,
    calculator,
    list_companies,
    lookup_company,
)
from tasks.schemas import ToolCall, ToolErrorCode, ToolResult


class Environment:
    """Own hidden state and execute only registered tools against it."""

    _EXPECTED_ARGUMENTS: dict[str, frozenset[str]] = {
        "lookup_company": frozenset({"company", "field"}),
        "calculator": frozenset({"expression"}),
        "list_companies": frozenset(),
    }

    def __init__(self, seed: int = 0) -> None:
        self._database = SyntheticCompanyDatabase(seed=seed)
        self._tool_handlers: dict[str, Callable[..., Any]] = {
            "lookup_company": self._lookup_company,
            "calculator": calculator,
            "list_companies": self._list_companies,
        }

    @property
    def available_tools(self) -> tuple[str, ...]:
        """Return tool names, not the database or its hidden records."""

        return tuple(self._tool_handlers)

    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Validate, execute, and convert a tool call to a stable result."""

        tool_name = tool_call.tool_name
        if not isinstance(tool_name, str) or tool_name not in self._tool_handlers:
            return self._error_result(
                str(tool_name),
                ToolErrorCode.UNKNOWN_TOOL,
                f"Unknown tool: {tool_name}",
            )

        if not isinstance(tool_call.arguments, dict):
            return self._error_result(
                tool_name,
                ToolErrorCode.INVALID_ARGUMENT,
                "arguments must be a dictionary",
            )

        argument_error = self._validate_arguments(tool_name, tool_call.arguments)
        if argument_error is not None:
            return argument_error

        try:
            output = self._tool_handlers[tool_name](**tool_call.arguments)
        except UnknownCompanyError as exc:
            return self._error_result(
                tool_name, ToolErrorCode.UNKNOWN_COMPANY, str(exc)
            )
        except UnknownFieldError as exc:
            return self._error_result(
                tool_name, ToolErrorCode.UNKNOWN_FIELD, str(exc)
            )
        except InvalidExpressionError as exc:
            return self._error_result(
                tool_name, ToolErrorCode.INVALID_EXPRESSION, str(exc)
            )
        except (InvalidToolArgumentError, TypeError, ValueError) as exc:
            return self._error_result(
                tool_name, ToolErrorCode.INVALID_ARGUMENT, str(exc)
            )
        except Exception as exc:  # Defensive boundary: callers always receive ToolResult.
            return self._error_result(
                tool_name, ToolErrorCode.EXECUTION_ERROR, str(exc)
            )

        return ToolResult(tool_name=tool_name, success=True, output=output)

    def _lookup_company(self, company: str, field: str) -> int | float:
        return lookup_company(company, field, database=self._database)

    def _list_companies(self) -> list[str]:
        return list_companies(database=self._database)

    def _validate_arguments(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult | None:
        expected = self._EXPECTED_ARGUMENTS[tool_name]
        provided = set(arguments)
        missing = sorted(expected - provided)
        if missing:
            return self._error_result(
                tool_name,
                ToolErrorCode.MISSING_ARGUMENT,
                f"Missing argument(s): {', '.join(missing)}",
            )

        unexpected = sorted(provided - expected)
        if unexpected:
            return self._error_result(
                tool_name,
                ToolErrorCode.INVALID_ARGUMENT,
                f"Unexpected argument(s): {', '.join(unexpected)}",
            )
        return None

    @staticmethod
    def _error_result(
        tool_name: str, error_code: ToolErrorCode, message: str
    ) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error_code=error_code.value,
            error_message=message,
        )
