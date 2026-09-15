"""Core data structures shared by tasks and the tool environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolErrorCode(StrEnum):
    """Stable error codes returned by the environment."""

    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    MISSING_ARGUMENT = "MISSING_ARGUMENT"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNKNOWN_COMPANY = "UNKNOWN_COMPANY"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    INVALID_EXPRESSION = "INVALID_EXPRESSION"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass(slots=True)
class Task:
    """A reproducible tool-use task."""

    task_id: str
    question: str
    difficulty: int
    environment_id: str
    available_tools: list[str]
    ground_truth: Any
    verification_spec: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    """A request to invoke one named tool with structured arguments."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """The structured outcome of a tool invocation."""

    tool_name: str
    success: bool
    output: Any | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class TrajectoryStep:
    """One completed environment interaction in a trajectory."""

    step_index: int
    tool_call: ToolCall
    tool_result: ToolResult
