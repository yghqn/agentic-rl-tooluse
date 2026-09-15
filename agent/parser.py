"""Strict JSON actions; no free-text extraction or Python evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

from tasks.schemas import ToolCall


MAX_OUTPUT_LENGTH = 16_384
MAX_JSON_DEPTH = 64


@dataclass(slots=True)
class FinalAnswer:
    answer: Any


@dataclass(slots=True)
class ParseFailure:
    error_code: str
    message: str


class ParseError(ValueError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParseError("DUPLICATE_KEY", "JSON keys must be unique")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ParseError("NON_FINITE_NUMBER", "JSON numbers must be finite")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        _reject_constant(value)
    return number


def parse_model_output(output: str) -> ToolCall | FinalAnswer:
    """Parse exactly one action. Tool argument semantics belong to Environment."""

    if not isinstance(output, str) or not output.strip():
        raise ParseError("INVALID_OUTPUT", "Output must be a non-empty JSON string")
    if len(output) > MAX_OUTPUT_LENGTH:
        raise ParseError("OUTPUT_TOO_LONG", "Output exceeds the length limit")
    try:
        action = json.loads(
            output,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except ParseError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ParseError("INVALID_JSON", "Output must be exactly one valid JSON object") from exc

    pending: list[tuple[Any, int]] = [(action, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise ParseError("OUTPUT_TOO_DEEP", "JSON nesting exceeds the depth limit")
        if isinstance(value, dict):
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)

    if not isinstance(action, dict):
        raise ParseError("INVALID_ACTION", "Action must be a JSON object")
    if action.get("type") == "tool_call":
        if set(action) != {"type", "tool_name", "arguments"}:
            raise ParseError("INVALID_ACTION", "Tool action requires type, tool_name, arguments")
        if not isinstance(action["tool_name"], str) or not action["tool_name"].strip():
            raise ParseError("INVALID_ACTION", "tool_name must be a non-empty string")
        if not isinstance(action["arguments"], dict):
            raise ParseError("INVALID_ACTION", "arguments must be a JSON object")
        return ToolCall(action["tool_name"], action["arguments"])
    if action.get("type") == "final" and set(action) == {"type", "answer"}:
        return FinalAnswer(action["answer"])
    raise ParseError("INVALID_ACTION", "Expected a tool_call or final action")
