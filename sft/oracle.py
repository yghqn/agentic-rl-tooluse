"""Privileged, deterministic demonstration policy. Never used for verification."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

from agent.agent import AgentEvent, AgentRun
from agent.parser import FinalAnswer, parse_model_output
from environment.database import SyntheticCompanyDatabase
from environment.environment import Environment
from tasks.schemas import Task, ToolCall, TrajectoryStep
from tasks.validators import validate_task


ORACLE_VERSION = "oracle-v1"


def generate_oracle_trajectory(task: Task, database: SyntheticCompanyDatabase) -> AgentRun:
    """Use structural task information, but obtain answer values ONLY via tools.

    The database validates the task and supplies the seed for a fresh matching
    Environment. Neither its records nor Task.ground_truth are copied into actions.
    This is one demonstration policy, not a definition of all valid solutions.
    """

    validate_task(task, database)
    environment = Environment(seed=database.seed)
    run = AgentRun(max_steps=16, termination_reason="max_steps")

    def invoke(name: str, arguments: dict[str, Any]) -> Any:
        raw = json.dumps({"type": "tool_call", "tool_name": name, "arguments": arguments},
                         sort_keys=True, allow_nan=False)
        call = parse_model_output(raw)
        if not isinstance(call, ToolCall):
            raise ValueError("Oracle did not produce a ToolCall")
        result = environment.execute_tool_call(call)
        if not result.success:
            raise ValueError(f"Oracle tool failed: {result.error_code}")
        index = len(run.events) + 1
        run.events.append(AgentEvent(index, raw, TrajectoryStep(index, call, result)))
        return result.output

    task_type = task.verification_spec["task_type"]
    metadata = task.metadata
    if task_type == "single_retrieval":
        answer = invoke("lookup_company", {"company": metadata["company"], "field": metadata["field"]})
    elif task_type == "list_companies":
        answer = sorted(invoke("list_companies", {}))
    elif task_type == "arithmetic":
        answer = invoke("calculator", {"expression": metadata["expression"]})
    else:
        companies = [metadata["company"]] if task_type == "profit_margin" else sorted(metadata["companies"])
        exact_margins: dict[str, Fraction] = {}
        margins: dict[str, int | float] = {}
        calculated: dict[str, int | float] = {}
        for company in companies:
            profit = invoke("lookup_company", {"company": company, "field": "profit"})
            revenue = invoke("lookup_company", {"company": company, "field": "revenue"})
            expression = f"{profit} / {revenue}"
            if expression not in calculated:
                calculated[expression] = invoke("calculator", {"expression": expression})
            margins[company] = calculated[expression]
            exact_margins[company] = Fraction(profit, revenue)
        if task_type == "profit_margin":
            answer = margins[companies[0]]
        else:
            winner = min(companies, key=lambda company: (-exact_margins[company], company))
            answer = {"company": winner, "profit_margin": margins[winner]}
    raw = json.dumps({"type": "final", "answer": answer}, sort_keys=True, allow_nan=False)
    final = parse_model_output(raw)
    if not isinstance(final, FinalAnswer):
        raise ValueError("Oracle did not produce a final answer")
    run.events.append(AgentEvent(len(run.events) + 1, raw, final_action=final))
    run.termination_reason = "final_answer"
    return run
