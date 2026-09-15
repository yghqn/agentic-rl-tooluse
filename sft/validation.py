"""Replay and dataset-quality checks, independent of the oracle policy."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict
import re
from typing import Any

from agent.agent import AgentRun, PromptAgent, ScriptedBackend
from agent.prompts import Message, build_messages
from environment.database import SyntheticCompanyDatabase
from environment.environment import Environment
from evaluation.failure_analysis import analyze_failures
from evaluation.metrics import trajectory_is_valid
from evaluation.verifier import VerificationResult, verify_final_answer
from tasks.generator import agent_task_view
from tasks.schemas import Task
from tasks.validators import validate_task


def to_sft_messages(messages: list[Message]) -> list[Message]:
    """Match the existing HF backend adapter without loading optional packages."""

    converted: list[Message] = []
    for message in messages:
        if message["role"] == "tool":
            converted.append({"role": "user", "content":
                              f"Tool observation ({message['name']}): {message['content']}"})
        else:
            converted.append({"role": message["role"], "content": message["content"]})
    return converted


def arithmetic_key(expression: str) -> str:
    """Ignore spelling whitespace, but retain operators, operands and grouping."""

    return ast.dump(ast.parse(expression.strip(), mode="eval"), include_attributes=False)


def _literal(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("Margin operands must be previously observed numeric literals")


def _check_observation_sources(task: Task, run: AgentRun) -> None:
    """Dataset-only causal/data coverage requirements, NOT verifier semantics.

    Accept different lookup orders and equivalent literal formatting; do not
    regenerate an oracle trajectory or compare against one reference sequence.
    """

    kind = task.verification_spec["task_type"]
    if kind not in {"profit_margin", "highest_profit_margin"}:
        if kind == "arithmetic":
            target = arithmetic_key(task.metadata["expression"])
            if any(arithmetic_key(step.tool_call.arguments["expression"]) != target for step in run.tool_steps):
                raise ValueError("Arithmetic call differs from the public expression")
        if not run.tool_steps:
            raise ValueError("A demonstration must include its legitimate tool observation")
        outputs = [step.tool_result.output for step in run.tool_steps]
        if kind == "list_companies":
            if not any(sorted(output) == sorted(run.final_answer) for output in outputs):
                raise ValueError("Final company list must come from a tool observation")
        elif run.final_answer not in outputs:
            raise ValueError("Final answer must come from a tool observation")
        return

    companies = [task.metadata["company"]] if kind == "profit_margin" else task.metadata["companies"]
    observed: dict[tuple[str, str], int | float] = {}
    calculated: dict[tuple[int | float, int | float], int | float] = {}
    for step in run.tool_steps:
        call = step.tool_call
        if call.tool_name == "lookup_company":
            observed[(call.arguments["company"], call.arguments["field"])] = step.tool_result.output
        elif call.tool_name == "calculator":
            expression = ast.parse(call.arguments["expression"].strip(), mode="eval").body
            if not isinstance(expression, ast.BinOp) or not isinstance(expression.op, ast.Div):
                raise ValueError("Margin demonstration must calculate observed profit / revenue")
            pair = (_literal(expression.left), _literal(expression.right))
            available_pairs = {
                (observed[(company, "profit")], observed[(company, "revenue")])
                for company in companies
                if (company, "profit") in observed and (company, "revenue") in observed
            }
            if pair not in available_pairs:
                raise ValueError("Calculator uses values before legitimate lookup observations")
            calculated[pair] = step.tool_result.output
    for company in companies:
        if (company, "profit") not in observed or (company, "revenue") not in observed:
            raise ValueError("Missing target lookup observations")
        pair = (observed[(company, "profit")], observed[(company, "revenue")])
        if pair not in calculated:
            raise ValueError("Missing margin calculation (cached results are allowed)")
    winner = companies[0] if kind == "profit_margin" else run.final_answer["company"]
    answer = run.final_answer if kind == "profit_margin" else run.final_answer["profit_margin"]
    pair = (observed[(winner, "profit")], observed[(winner, "revenue")])
    if answer != calculated[pair]:
        raise ValueError("Final margin must be the observed calculator result")


def validate_demonstration(
    task: Task,
    database: SyntheticCompanyDatabase,
    run: AgentRun,
    *,
    messages: list[Message] | None = None,
    sft_sample: dict[str, Any] | None = None,
) -> VerificationResult:
    """Reject inconsistent or low-quality demonstrations, then return verification."""

    validate_task(task, database)
    if not isinstance(run, AgentRun) or not trajectory_is_valid(run) or not run.has_final_answer:
        raise ValueError("Expected a valid complete trajectory ending in one final action")
    kind = task.verification_spec["task_type"]
    # Existing synthetic non-arithmetic questions contain no numbers. This is a
    # leakage guard for this dataset, not exact natural-language template matching.
    if kind != "arithmetic" and re.search(r"\d", task.question):
        raise ValueError("Numeric information in a non-arithmetic question")
    if kind == "arithmetic":
        numbers = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
        if Counter(re.findall(numbers, task.question)) - Counter(re.findall(numbers, task.metadata["expression"])):
            raise ValueError("Arithmetic question contains numbers outside its public expression")
    if any(word in task.question for word in ("ground_truth", "verification_spec", "metadata")):
        raise ValueError("Privileged fields in question")
    if any(not step.tool_result.success for step in run.tool_steps):
        raise ValueError("Demonstrations must contain only successful tool calls")
    replay = PromptAgent(ScriptedBackend([event.raw_output for event in run.events]), run.max_steps).run(
        agent_task_view(task), Environment(seed=database.seed),
    )
    if asdict(replay) != asdict(run):
        raise ValueError("Replay differs from recorded trajectory or observations")
    verification = verify_final_answer(task, replay.final_answer)
    if not verification.success:
        raise ValueError(f"Independent verifier rejected final answer: {verification.error_code}")
    analysis = analyze_failures(task, replay, verification)
    if analysis.categories:
        raise ValueError(f"Demonstration quality failure: {', '.join(analysis.categories)}")
    _check_observation_sources(task, replay)
    expected = build_messages(agent_task_view(task), replay.events)
    if messages is not None and messages != expected:
        raise ValueError("Messages differ from public inputs and actual observations")
    if sft_sample is not None:
        keys = {"schema_version", "sample_id", "task_id", "split", "messages", "assistant_message_indices"}
        if set(sft_sample) != keys or sft_sample["schema_version"] != "sft-chat-v1":
            raise ValueError("Unexpected SFT sample schema")
        if sft_sample["task_id"] != task.task_id or sft_sample["split"] not in {"train", "dev", "test"}:
            raise ValueError("Invalid SFT sample identity or split")
        if sft_sample["sample_id"] != f"oracle-v1:{task.task_id}":
            raise ValueError("Invalid stable sample ID")
        converted = to_sft_messages(expected)
        if sft_sample["messages"] != converted:
            raise ValueError("SFT messages contain unobserved information or hidden reasoning")
        indices = [i for i, message in enumerate(converted) if message["role"] == "assistant"]
        if sft_sample["assistant_message_indices"] != indices:
            raise ValueError("Incorrect assistant supervision indices")
    return verification
