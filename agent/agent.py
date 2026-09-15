"""Minimal synchronous Agent orchestration with an injectable text backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from agent.parser import FinalAnswer, ParseError, ParseFailure, parse_model_output
from agent.prompts import Message, build_messages, validate_agent_view
from tasks.schemas import ToolCall, ToolResult, TrajectoryStep


class Backend(Protocol):
    def generate(self, messages: list[Message]) -> str: ...


class ToolExecutor(Protocol):
    def execute_tool_call(self, tool_call: ToolCall) -> ToolResult: ...


class Agent(Protocol):
    def run(self, agent_view: dict[str, Any], environment: ToolExecutor) -> AgentRun: ...


@dataclass(slots=True)
class AgentEvent:
    step_index: int
    raw_output: str | None = None
    interaction: TrajectoryStep | None = None
    final_action: FinalAnswer | None = None
    parse_error: ParseFailure | None = None
    backend_error: str | None = None


@dataclass(slots=True)
class AgentRun:
    max_steps: int
    termination_reason: Literal["final_answer", "max_steps", "backend_error"]
    events: list[AgentEvent] = field(default_factory=list)

    @property
    def tool_steps(self) -> list[TrajectoryStep]:
        return [event.interaction for event in self.events if event.interaction is not None]

    @property
    def has_final_answer(self) -> bool:
        return bool(self.events and self.events[-1].final_action is not None)

    @property
    def final_answer(self) -> Any | None:
        return self.events[-1].final_action.answer if self.has_final_answer else None

    @property
    def agent_steps(self) -> int:
        return len(self.events)


class ScriptedBackend:
    """Offline replay only: instantiate a fresh backend for each task."""

    def __init__(self, responses: Sequence[str]) -> None:
        if isinstance(responses, str) or any(not isinstance(item, str) for item in responses):
            raise ValueError("responses must be a sequence of model-output strings")
        self._responses = tuple(responses)
        self._position = 0

    def generate(self, messages: list[Message]) -> str:
        if self._position >= len(self._responses):
            raise RuntimeError("Scripted responses exhausted")
        output = self._responses[self._position]
        self._position += 1
        return output


class PromptAgent:
    def __init__(self, backend: Backend, max_steps: int = 16) -> None:
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self._backend = backend
        self._max_steps = max_steps

    def run(self, agent_view: dict[str, Any], environment: ToolExecutor) -> AgentRun:
        validate_agent_view(agent_view)
        run = AgentRun(self._max_steps, "max_steps")
        for step_index in range(1, self._max_steps + 1):
            messages = build_messages(agent_view, run.events)
            event = AgentEvent(step_index)
            run.events.append(event)
            try:
                output = self._backend.generate(messages)
            except Exception as exc:
                event.backend_error = f"{type(exc).__name__}: {exc}"
                run.termination_reason = "backend_error"
                return run
            event.raw_output = output if isinstance(output, str) else None
            try:
                action = parse_model_output(output)
            except ParseError as exc:
                event.parse_error = ParseFailure(exc.error_code, str(exc))
                continue
            if isinstance(action, FinalAnswer):
                event.final_action = action
                run.termination_reason = "final_answer"
                return run
            result = environment.execute_tool_call(action)
            event.interaction = TrajectoryStep(step_index, action, result)
        return run
