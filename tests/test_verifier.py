"""Verifier, privacy, protocol, diagnostics and offline evaluation tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from agent.agent import AgentEvent, PromptAgent, ScriptedBackend
from agent.parser import FinalAnswer, ParseError, parse_model_output
from agent.prompts import build_messages, tool_schemas
from environment.database import SyntheticCompanyDatabase
from environment.environment import Environment
from evaluation.failure_analysis import FAILURE_CATEGORIES, analyze_failures
from evaluation.metrics import EvaluationRecord, compute_metrics, trajectory_is_valid
from evaluation.verifier import VerificationConfigError, VerificationResult, verify_final_answer
from scripts.evaluate import evaluate_tasks, load_scripted_responses, main
from tasks.generator import TaskGenerator, agent_task_view
from tasks.schemas import Task, ToolCall
from tasks.validators import environment_id_for_seed


SEED = 42
ENVIRONMENTS = {environment_id_for_seed(SEED): SEED}


def tool_output(name: str, **arguments: Any) -> str:
    return json.dumps({"type": "tool_call", "tool_name": name, "arguments": arguments})


def final_output(answer: Any) -> str:
    return json.dumps({"type": "final", "answer": answer})


def replay_for(task: Task, reverse: bool = False) -> list[str]:
    """Evaluator-side fixtures, NOT a policy/backend that solves tasks."""

    database = SyntheticCompanyDatabase(SEED)
    metadata = task.metadata
    task_type = task.verification_spec["task_type"]
    outputs: list[str] = []
    if task_type == "single_retrieval":
        outputs.append(tool_output("lookup_company", **metadata))
    elif task_type == "list_companies":
        outputs.append(tool_output("list_companies"))
    elif task_type == "arithmetic":
        outputs.append(tool_output("calculator", **metadata))
    else:
        companies = [metadata["company"]] if task_type == "profit_margin" else list(metadata["companies"])
        if reverse:
            companies.reverse()
        fields = ["revenue", "profit"] if reverse else ["profit", "revenue"]
        for company in companies:
            outputs.extend(tool_output("lookup_company", company=company, field=field) for field in fields)
            profit = database.lookup(company, "profit")
            revenue = database.lookup(company, "revenue")
            outputs.append(tool_output("calculator", expression=f"{profit} / {revenue}"))
    outputs.append(final_output(task.ground_truth))
    return outputs


def record_for(task: Task, outputs: list[str], max_steps: int = 16) -> EvaluationRecord:
    run = PromptAgent(ScriptedBackend(outputs), max_steps).run(agent_task_view(task), Environment(SEED))
    verification = verify_final_answer(task, run.final_answer) if run.has_final_answer else verify_final_answer(task)
    return EvaluationRecord(task, run, verification, analyze_failures(task, run, verification))


@pytest.mark.parametrize(("difficulty", "index"), [(1, 0), (2, 1), (3, 0)])
def test_correct_numeric_answer(difficulty: int, index: int) -> None:
    task = TaskGenerator(SEED).generate_task(difficulty, index)
    result = verify_final_answer(task, task.ground_truth)
    assert result.success
    assert result.normalized_answer == task.ground_truth
    assert result.error_code is None


@pytest.mark.parametrize(
    ("rel_tol", "abs_tol", "answer", "success"),
    [(0.0, 0.1, 10.05, True), (0.0, 0.1, 10.2, False),
     (0.01, 0.0, 10.05, True), (0.01, 0.0, 10.2, False)],
)
def test_numeric_tolerances_come_from_spec(
    rel_tol: float, abs_tol: float, answer: float, success: bool
) -> None:
    task = TaskGenerator(SEED).generate_task(3)
    task.ground_truth = 10.0
    task.verification_spec.update(rel_tol=rel_tol, abs_tol=abs_tol)
    assert verify_final_answer(task, answer).success is success


def test_wrong_numeric_answer() -> None:
    task = TaskGenerator(SEED).generate_task(3)
    result = verify_final_answer(task, task.ground_truth + 1)
    assert not result.success
    assert result.error_code == "NUMERIC_MISMATCH"
    assert result.failure_category == "final_answer_error"


def test_unordered_company_list_and_normalization() -> None:
    task = TaskGenerator(SEED).generate_task(2, 0)
    answer = [f" {company} " for company in reversed(task.ground_truth)]
    result = verify_final_answer(task, answer)
    assert result.success
    assert result.normalized_answer == sorted(task.ground_truth)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "wrong_name"])
def test_wrong_company_list(change: str) -> None:
    task = TaskGenerator(SEED).generate_task(2, 0)
    answer = list(task.ground_truth)
    if change == "missing":
        answer.pop()
    elif change == "extra":
        answer.append("Unknown Corp")
    elif change == "duplicate":
        answer.append(answer[0])
    else:
        answer[0] = "Unknown Corp"
    result = verify_final_answer(task, answer)
    assert not result.success
    assert result.error_code == ("MALFORMED_FINAL_ANSWER" if change == "duplicate" else "COMPANY_LIST_MISMATCH")


def test_correct_l4_structured_answer() -> None:
    task = TaskGenerator(SEED).generate_task(4)
    assert verify_final_answer(task, task.ground_truth).success
    near = {**task.ground_truth, "profit_margin": task.ground_truth["profit_margin"] + 1e-12}
    assert verify_final_answer(task, near).success


@pytest.mark.parametrize(("key", "value", "code"), [
    ("company", "Unknown Corp", "WRONG_COMPANY"),
    ("profit_margin", 999.0, "MARGIN_MISMATCH"),
])
def test_wrong_l4_answer(key: str, value: Any, code: str) -> None:
    task = TaskGenerator(SEED).generate_task(4)
    result = verify_final_answer(task, {**task.ground_truth, key: value})
    assert not result.success
    assert result.error_code == code


@pytest.mark.parametrize("answer", [None, True, "0.1", float("nan"), float("inf"), [], {}])
def test_malformed_numeric_final_answer(answer: Any) -> None:
    assert verify_final_answer(TaskGenerator(SEED).generate_task(3), answer).error_code == "MALFORMED_FINAL_ANSWER"


@pytest.mark.parametrize("answer", ["Company A", [], {}, {"company": "Company A"},
    {"company": "Company A", "profit_margin": True},
    {"company": "Company A", "profit_margin": 0.1, "extra": 1}])
def test_malformed_l4_final_answer(answer: Any) -> None:
    assert verify_final_answer(TaskGenerator(SEED).generate_task(4), answer).error_code == "MALFORMED_FINAL_ANSWER"


def test_missing_and_null_final_answers_are_distinct() -> None:
    task = TaskGenerator(SEED).generate_task(1)
    assert verify_final_answer(task).error_code == "NO_FINAL_ANSWER"
    assert verify_final_answer(task, None).error_code == "MALFORMED_FINAL_ANSWER"


@pytest.mark.parametrize("change", ["contract", "ground_truth", "negative_tolerance", "nan_tolerance"])
def test_invalid_verification_configuration_raises(change: str) -> None:
    task = TaskGenerator(SEED).generate_task(3)
    if change == "contract":
        task.verification_spec["comparison"] = "unknown"
    elif change == "ground_truth":
        task.ground_truth = None
    elif change == "negative_tolerance":
        task.verification_spec["abs_tol"] = -1
    else:
        task.verification_spec["rel_tol"] = float("nan")
    with pytest.raises(VerificationConfigError):
        verify_final_answer(task, 0.1)


def test_verifier_has_no_trajectory_parameter_or_dependency() -> None:
    assert list(inspect.signature(verify_final_answer).parameters) == ["task", "answer"]
    task = TaskGenerator(SEED).generate_task(4)
    first = record_for(task, replay_for(task))
    second = record_for(task, replay_for(task, reverse=True))
    direct = record_for(task, [final_output(task.ground_truth)])
    assert first.verification.success and second.verification.success and direct.verification.success
    assert first.run.tool_steps != second.run.tool_steps
    assert direct.failure_analysis.categories == []


def test_prompt_rejects_full_task_and_privileged_dict() -> None:
    task = TaskGenerator(SEED).generate_task(4)
    with pytest.raises(ValueError):
        build_messages(task)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_messages(asdict(task))
    for field in ("ground_truth", "verification_spec", "metadata"):
        with pytest.raises(ValueError):
            build_messages({**agent_task_view(task), field: "SECRET_SENTINEL"})


@pytest.mark.parametrize("difficulty", [1, 4])
def test_no_privileged_field_leakage_in_backend_messages(difficulty: int) -> None:
    task = TaskGenerator(SEED).generate_task(difficulty)
    responses = replay_for(task)
    task.ground_truth = "SECRET_GT_SENTINEL"
    task.verification_spec = {"private": "SECRET_SPEC_SENTINEL"}
    task.metadata = {"private": "SECRET_METADATA_SENTINEL"}

    class RecordingBackend(ScriptedBackend):
        def __init__(self) -> None:
            super().__init__(responses)
            self.messages: list[Any] = []

        def generate(self, messages: list[dict[str, str]]) -> str:
            self.messages.append(deepcopy(messages))
            return super().generate(messages)

    backend = RecordingBackend()
    run = PromptAgent(backend).run(agent_task_view(task), Environment(SEED))
    serialized = json.dumps(backend.messages)
    assert "SECRET_" not in serialized
    assert "ground_truth" not in serialized and "verification_spec" not in serialized
    assert "metadata" not in serialized and "database" not in serialized
    assert len(backend.messages) == (2 if difficulty == 1 else 10)
    assert any(message["role"] == "tool" for message in backend.messages[1])
    assert str(run.tool_steps[0].tool_result.output) in serialized


def test_execution_error_details_are_not_echoed_to_prompt() -> None:
    task = TaskGenerator(SEED).generate_task(2, 0)
    environment = Environment(SEED)

    def broken() -> None:
        raise RuntimeError("SECRET_INTERNAL_SENTINEL")

    environment._tool_handlers["list_companies"] = broken
    record = PromptAgent(ScriptedBackend([tool_output("list_companies"), final_output([])])).run(
        agent_task_view(task), environment
    )
    assert "SECRET_INTERNAL_SENTINEL" not in json.dumps(build_messages(agent_task_view(task), record.events[:1]))


def test_tool_schemas_hide_internal_injection_and_are_copied() -> None:
    schemas = tool_schemas(["lookup_company", "calculator", "list_companies"])
    assert "database" not in json.dumps(schemas)
    schemas[0]["parameters"]["properties"].clear()
    assert "company" in tool_schemas(["lookup_company"])[0]["parameters"]["properties"]


def test_parser_accepts_tool_and_final_actions() -> None:
    assert parse_model_output(tool_output("calculator", expression="1 + 2")) == ToolCall("calculator", {"expression": "1 + 2"})
    assert parse_model_output(final_output(3)) == FinalAnswer(3)
    assert parse_model_output(tool_output("not_a_tool")) == ToolCall("not_a_tool", {})


@pytest.mark.parametrize("output", [
    "", "some text", '```json\n{"type":"final","answer":1}\n```',
    '{"type":"final","answer":1} {"type":"final","answer":2}',
    '{"type":"final","answer":1,"answer":2}',
    '{"type":"final","answer":NaN}', '{"type":"final","answer":1e999}',
    '{"type":"tool_call","tool_name":"calculator","arguments":[]}',
    '{"type":"final"}', '{"type":"final","answer":1,"extra":1}', "[]",
    '{"type":"tool_call","tool_name":"calculator","arguments":{"x":1,"x":2}}',
    "__import__('os').getcwd()", "x" * 16_385, None,
])
def test_parser_fails_safely(output: Any) -> None:
    with pytest.raises(ParseError):
        parse_model_output(output)


def test_parser_rejects_excessively_nested_output() -> None:
    output = '{"type":"final","answer":' + '[' * 70 + '0' + ']' * 70 + '}'
    with pytest.raises(ParseError, match="depth limit"):
        parse_model_output(output)


def test_agent_recovers_from_parser_error_and_stops_at_first_final() -> None:
    task = TaskGenerator(SEED).generate_task(1)
    record = record_for(task, ["bad output", final_output(task.ground_truth), "must not execute"])
    assert record.verification.success
    assert record.run.agent_steps == 2
    assert record.run.events[0].parse_error is not None
    assert not trajectory_is_valid(record.run)
    metrics = compute_metrics([record])
    assert metrics["counts"]["parse_error_count"] == 1
    assert metrics["invalid_tool_call_rate"] == {"value": None, "numerator": 0, "denominator": 0}


def test_agent_max_steps_and_backend_exhaustion() -> None:
    task = TaskGenerator(SEED).generate_task(2, 0)
    maxed = record_for(task, [tool_output("list_companies")] * 4, max_steps=3)
    assert maxed.run.termination_reason == "max_steps"
    assert maxed.run.agent_steps == 3
    assert maxed.verification.error_code == "NO_FINAL_ANSWER"
    assert trajectory_is_valid(maxed.run)
    exhausted = record_for(task, [])
    assert exhausted.run.termination_reason == "backend_error"
    assert exhausted.run.agent_steps == 1
    assert not trajectory_is_valid(exhausted.run)
    assert compute_metrics([exhausted])["counts"]["backend_error_count"] == 1


@pytest.mark.parametrize("max_steps", [0, -1, True, 1.5])
def test_invalid_agent_step_limits(max_steps: Any) -> None:
    with pytest.raises(ValueError):
        PromptAgent(ScriptedBackend([]), max_steps)


def test_success_is_independent_of_diagnostic_tool_selection() -> None:
    task = TaskGenerator(SEED).generate_task(1)
    record = record_for(task, [tool_output("list_companies"), final_output(task.ground_truth)])
    assert record.verification.success
    assert record.failure_analysis.categories == ["wrong_tool"]
    metrics = compute_metrics([record])
    assert metrics["task_success_rate"]["value"] == 1
    assert metrics["tool_selection_accuracy"] == {"value": 0.0, "numerator": 0, "denominator": 1}


def test_wrong_argument_and_invalid_tool_call_evidence() -> None:
    task = TaskGenerator(SEED).generate_task(1)
    outputs = [
        tool_output("lookup_company", company="Unknown Corp", field="profit"),
        tool_output("lookup_company", company=task.metadata["company"], field="market_cap"),
        tool_output("calculator"), tool_output("unknown"), final_output(-999),
    ]
    record = record_for(task, outputs)
    assert {"wrong_argument", "invalid_tool_call", "missing_information", "premature_termination", "final_answer_error"}.issubset(record.failure_analysis.categories)
    assert all(item.reason for item in record.failure_analysis.evidence)
    metrics = compute_metrics([record])
    assert metrics["counts"]["tool_call_count"] == 4
    assert metrics["counts"]["argument_assessed_count"] == 3
    assert metrics["invalid_tool_call_rate"]["value"] == 1
    assert metrics["argument_accuracy"]["value"] == 0


def test_valid_but_off_target_lookup_is_wrong_argument_not_invalid_call() -> None:
    task = TaskGenerator(SEED).generate_task(1)
    off_target = next(company for company in SyntheticCompanyDatabase(SEED).list_companies() if company != task.metadata["company"])
    record = record_for(task, [tool_output("lookup_company", company=off_target, field=task.metadata["field"]), final_output(task.ground_truth)])
    assert record.verification.success
    assert record.failure_analysis.categories == ["wrong_argument"]
    assert compute_metrics([record])["invalid_tool_call_rate"]["value"] == 0


def test_redundancy_loop_and_counts_are_per_call_or_per_task_as_documented() -> None:
    task = TaskGenerator(SEED).generate_task(2, 0)
    record = record_for(task, [tool_output("list_companies")] * 4 + [final_output(task.ground_truth)])
    metrics = compute_metrics([record])
    assert record.verification.success
    assert metrics["redundant_tool_call_rate"] == {"value": 0.75, "numerator": 3, "denominator": 4}
    assert metrics["failure_category_counts"]["redundant_tool_use"] == 1
    assert metrics["failure_category_counts"]["loop"] == 1
    assert metrics["average_agent_steps"]["value"] == 5


def test_repeated_failed_calls_are_not_redundant_but_can_loop() -> None:
    task = TaskGenerator(SEED).generate_task(2, 1)
    record = record_for(task, [tool_output("calculator", expression="1 / 0")] * 3, max_steps=3)
    assert "loop" in record.failure_analysis.categories
    assert "calculation_failure" in record.failure_analysis.categories
    assert compute_metrics([record])["redundant_tool_call_rate"]["numerator"] == 0


def test_execution_error_is_not_argument_or_invalid_call_error() -> None:
    task = TaskGenerator(SEED).generate_task(2, 0)
    record = record_for(task, replay_for(task))
    result = record.run.tool_steps[0].tool_result
    result.success = False
    result.output = None
    result.error_code = "EXECUTION_ERROR"
    record.failure_analysis = analyze_failures(task, record.run, record.verification)
    metrics = compute_metrics([record])
    assert metrics["counts"]["execution_error_count"] == 1
    assert metrics["argument_accuracy"]["denominator"] == 0
    assert metrics["invalid_tool_call_rate"]["numerator"] == 0


def test_ratio_mismatch_has_observation_evidence() -> None:
    task = TaskGenerator(SEED).generate_task(3)
    record = record_for(task, replay_for(task)[:-1] + [final_output(task.ground_truth + 1)])
    evidence = next(item for item in record.failure_analysis.evidence if item.category == "calculation_failure")
    assert len(evidence.step_indices) == 2
    assert evidence.details["observed_ratio"] == pytest.approx(task.ground_truth)


def test_nonwinner_with_its_actual_margin_has_comparison_evidence() -> None:
    task = TaskGenerator(SEED).generate_task(4)
    company = next(name for name in task.metadata["companies"] if name != task.ground_truth["company"])
    database = SyntheticCompanyDatabase(SEED)
    margin = database.lookup(company, "profit") / database.lookup(company, "revenue")
    record = record_for(task, replay_for(task)[:-1] + [final_output({"company": company, "profit_margin": margin})])
    assert "planning_failure" in record.failure_analysis.categories
    assert "calculation_failure" not in record.failure_analysis.categories


def test_absent_calculator_alone_does_not_infer_planning_failure() -> None:
    task = TaskGenerator(SEED).generate_task(3)
    outputs = [output for output in replay_for(task)[:-1] if '"calculator"' not in output]
    record = record_for(task, outputs + [final_output(None)])
    assert record.failure_analysis.categories == ["final_answer_error"]


def test_unclassified_failure_fallback() -> None:
    task = TaskGenerator(SEED).generate_task(2, 0)
    record = record_for(task, replay_for(task))
    unknown = VerificationResult(False, "FUTURE_FAILURE")
    analysis = analyze_failures(task, record.run, unknown)
    assert analysis.categories == ["unclassified_failure"]
    assert analysis.evidence[0].details["verification_error_code"] == "FUTURE_FAILURE"


def test_empty_metrics_have_raw_zero_counts_and_none_ratios() -> None:
    metrics = compute_metrics([])
    assert metrics["counts"]["task_count"] == 0
    assert metrics["task_success_rate"] == {"value": None, "numerator": 0, "denominator": 0}
    assert metrics["failure_category_counts"] == dict.fromkeys(FAILURE_CATEGORIES, 0)
    assert set(metrics["by_difficulty"]) == {"L1", "L2", "L3", "L4"}


def test_micro_aggregation_not_average_of_task_rates() -> None:
    task = TaskGenerator(SEED).generate_task(1)
    one = record_for(task, [tool_output("list_companies"), final_output(task.ground_truth)])
    three = record_for(task, [tool_output("lookup_company", **task.metadata)] * 3 + [final_output(task.ground_truth)])
    metrics = compute_metrics([one, three])
    assert metrics["tool_selection_accuracy"] == {"value": 0.75, "numerator": 3, "denominator": 4}
    assert metrics["average_tool_calls"] == {"value": 2.0, "numerator": 4, "denominator": 2}
    assert metrics["average_agent_steps"] == {"value": 3.0, "numerator": 6, "denominator": 2}


@pytest.mark.parametrize("change", ["index", "result_name", "termination", "raw_output", "after_final"])
def test_inconsistent_trajectory_is_invalid(change: str) -> None:
    task = TaskGenerator(SEED).generate_task(1)
    record = record_for(task, replay_for(task))
    assert trajectory_is_valid(record.run)
    if change == "index":
        record.run.events[0].step_index = 2
    elif change == "result_name":
        record.run.tool_steps[0].tool_result.tool_name = "calculator"
    elif change == "termination":
        record.run.termination_reason = "max_steps"
    elif change == "raw_output":
        record.run.events[0].raw_output = tool_output("list_companies")
    else:
        record.run.events.append(AgentEvent(3, final_output(1), final_action=FinalAnswer(1)))
    assert not trajectory_is_valid(record.run)


def test_l1_l4_offline_end_to_end_and_independent_task_state() -> None:
    tasks = [TaskGenerator(SEED).generate_task(4), TaskGenerator(SEED).generate_task(1)]
    scripts = {task.task_id: replay_for(task) for task in tasks}
    instances: list[ScriptedBackend] = []

    def factory(task_id: str) -> PromptAgent:
        backend = ScriptedBackend(scripts[task_id])
        instances.append(backend)
        return PromptAgent(backend)

    report = evaluate_tasks(tasks, factory, ENVIRONMENTS)
    assert [record.task.difficulty for record in report.records] == [1, 4]
    assert instances[0] is not instances[1]
    assert report.metrics["task_success_rate"]["value"] == 1
    assert report.metrics["counts"]["tool_call_count"] == 10
    assert report.metrics["counts"]["agent_step_count"] == 12
    assert report.metrics["by_difficulty"]["L1"]["task_success_rate"]["value"] == 1
    assert report.metrics["by_difficulty"]["L4"]["task_success_rate"]["value"] == 1
    exported = json.dumps(report.to_dict(), allow_nan=False)
    assert "ground_truth" not in exported and "verification_spec" not in exported and "metadata" not in exported


def test_scripted_backends_do_not_share_cursor_or_mutable_response_list() -> None:
    responses = [final_output(1), final_output(2)]
    first, second = ScriptedBackend(responses), ScriptedBackend(responses)
    responses.clear()
    assert first.generate([]) == final_output(1)
    assert first.generate([]) == final_output(2)
    assert second.generate([]) == final_output(1)


def test_harness_validates_all_tasks_before_running_and_rejects_duplicates() -> None:
    tasks = TaskGenerator(SEED).generate_tasks(1)
    calls: list[str] = []

    def factory(task_id: str) -> PromptAgent:
        calls.append(task_id)
        return PromptAgent(ScriptedBackend([]))

    tasks[-1].ground_truth = None
    with pytest.raises(ValueError):
        evaluate_tasks(tasks, factory, ENVIRONMENTS)
    assert calls == []
    task = TaskGenerator(SEED).generate_task(1)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_tasks([task, task], factory, ENVIRONMENTS)


def test_harness_rejects_reused_agent_and_unconfigured_environment() -> None:
    tasks = TaskGenerator(SEED).generate_tasks(1)
    reused = PromptAgent(ScriptedBackend([final_output(None)] * 4))
    with pytest.raises(ValueError, match="fresh Agent"):
        evaluate_tasks(tasks, lambda _: reused, ENVIRONMENTS)
    with pytest.raises(ValueError, match="No seed"):
        evaluate_tasks(tasks, lambda _: reused, {})


def test_harness_evaluation_is_deterministic() -> None:
    tasks = TaskGenerator(SEED).generate_tasks(2)
    scripts = {task.task_id: replay_for(task) for task in tasks}

    def factory(task_id: str) -> PromptAgent:
        return PromptAgent(ScriptedBackend(scripts[task_id]))

    first = evaluate_tasks(tasks, factory, ENVIRONMENTS)
    second = evaluate_tasks(list(reversed(tasks)), factory, ENVIRONMENTS)
    assert first.to_dict() == second.to_dict()
    assert first.metrics["task_success_rate"]["value"] == 1


def test_jsonl_cli_offline_pipeline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tasks = TaskGenerator(SEED).generate_tasks(1)
    path = tmp_path / "replay.jsonl"
    path.write_text("\n".join(json.dumps({"task_id": task.task_id, "responses": replay_for(task)}) for task in tasks), encoding="utf-8")
    assert main(["--seed", str(SEED), "--count-per-level", "1", "--responses", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "scripted_replay"
    assert output["metrics"]["task_success_rate"]["value"] == 1
    assert len(output["runs"]) == 4


@pytest.mark.parametrize("row", ["bad json", "{}", '{"task_id":"id","responses":[1]}'])
def test_malformed_replay_rows_are_rejected(tmp_path: Path, row: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(row, encoding="utf-8")
    with pytest.raises(ValueError):
        load_scripted_responses(path)


def test_duplicate_replay_rows_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    row = json.dumps({"task_id": "id", "responses": []})
    path.write_text(row + "\n" + row, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_scripted_responses(path)
