"""Real deterministic Environment fixtures, never production oracle trajectories."""
import json
import pytest

from agent.agent import PromptAgent,ScriptedBackend
from agent.parser import FinalAnswer
from environment.environment import Environment
from evaluation.verifier import verify_final_answer
from grpo.rewards import compute_reward,complete_l4_evidence,_margin_expression
from tasks.generator import TaskGenerator,agent_task_view
from tasks.schemas import ToolCall


def action(tool,**arguments):
    return json.dumps({"type":"tool_call","tool_name":tool,"arguments":arguments})


def fixture_run(task,seed=40000,order="normal",missing=None,winner=True,margin=True):
    env = Environment(seed)
    responses = []
    companies = task.metadata.get("companies",[])
    if order == "reverse": companies = list(reversed(companies))
    calculations = []
    for company in companies:
        p = env.execute_tool_call(ToolCall("lookup_company",{"company":company,"field":"profit"})).output
        r = env.execute_tool_call(ToolCall("lookup_company",{"company":company,"field":"revenue"})).output
        for field in ("revenue","profit") if order == "reverse" else ("profit","revenue"):
            if missing != (company,field):
                responses.append(action("lookup_company",company=company,field=field))
        expression = f"({p} * 2) / ({r} * 2)" if order == "reverse" else f"{p}/{r}"
        if missing != (company,"calculator"):
            if order == "all-lookups": calculations.append(action("calculator",expression=expression))
            else: responses.append(action("calculator",expression=expression))
    responses.extend(calculations)
    answer = dict(task.ground_truth)
    if not winner: answer["company"] = next(c for c in companies if c != answer["company"])
    if not margin: answer["profit_margin"] += 0.1
    responses.append(json.dumps({"type":"final","answer":answer}))
    run = PromptAgent(ScriptedBackend(responses),max_steps=32).run(agent_task_view(task),Environment(seed))
    return run,verify_final_answer(task,run.final_answer)


@pytest.mark.parametrize("order",["normal","reverse","all-lookups"])
def test_complete_evidence_without_reference_order(order):
    task = TaskGenerator(40000).generate_task(4)
    run,v = fixture_run(task,order=order)
    reward = compute_reward(task,run,v,"shaped")
    assert v.success and complete_l4_evidence(task,run)
    assert reward.counts["selection_with_evidence"] == 1
    assert reward.components["selection_with_evidence"] == 0.15
    assert compute_reward(task,run,v,"outcome_only").total == 1


def test_guessing_correct_winner_has_no_selection_shaping():
    task = TaskGenerator(40000).generate_task(4)
    run = PromptAgent(ScriptedBackend([json.dumps({"type":"final","answer":task.ground_truth})])).run(agent_task_view(task),Environment(40000))
    v = verify_final_answer(task,run.final_answer)
    assert v.success
    assert compute_reward(task,run,v,"outcome_only").total == 1
    assert compute_reward(task,run,v,"shaped").counts["selection_with_evidence"] == 0


@pytest.mark.parametrize("field",["profit","revenue","calculator"])
def test_incomplete_evidence_denies_c(field):
    task = TaskGenerator(40000).generate_task(4)
    run,v = fixture_run(task,missing=(task.metadata["companies"][0],field))
    assert v.success and compute_reward(task,run,v,"shaped").counts["selection_with_evidence"] == 0


def test_wrong_winner_right_company_margin_has_no_c():
    task = TaskGenerator(40000).generate_task(4)
    run,v = fixture_run(task,winner=False)
    assert not v.success
    assert compute_reward(task,run,v,"shaped").counts["selection_with_evidence"] == 0


def test_correct_selection_wrong_margin_partial_credit_only_with_evidence():
    task = TaskGenerator(40000).generate_task(4)
    run,v = fixture_run(task,margin=False)
    assert not v.success and v.error_code == "MARGIN_MISMATCH"
    reward = compute_reward(task,run,v,"shaped")
    assert reward.counts["selection_with_evidence"] == 1 and reward.total == pytest.approx(0.17)


def test_selection_is_not_a_numeric_format_success_requirement():
    task = TaskGenerator(40000).generate_task(4)
    run,_ = fixture_run(task)
    answer = {"company":task.ground_truth["company"],"profit_margin":"not numeric"}
    run.events[-1].raw_output = json.dumps({"type":"final","answer":answer})
    run.events[-1].final_action = FinalAnswer(answer)
    v = verify_final_answer(task,answer)
    assert not v.success
    assert compute_reward(task,run,v,"shaped").counts["selection_with_evidence"] == 1


def test_selection_normalizes_company_whitespace_like_verifier():
    task = TaskGenerator(40000).generate_task(4)
    run,_ = fixture_run(task)
    answer = dict(task.ground_truth,company=" "+task.ground_truth["company"]+" ")
    run.events[-1].raw_output = json.dumps({"type":"final","answer":answer})
    run.events[-1].final_action = FinalAnswer(answer)
    v = verify_final_answer(task,answer)
    assert v.success
    assert compute_reward(task,run,v,"shaped").counts["selection_with_evidence"] == 1


def test_margin_constants_after_lookups_do_not_count_as_calculation_evidence():
    task = TaskGenerator(40000).generate_task(4)
    reference,_ = fixture_run(task)
    outputs = []
    for event in reference.events:
        if event.interaction and event.interaction.tool_call.tool_name == "calculator":
            outputs.append(action("calculator",expression=str(event.interaction.tool_result.output)))
        else: outputs.append(event.raw_output)
    run = PromptAgent(ScriptedBackend(outputs),32).run(agent_task_view(task),Environment(40000))
    v = verify_final_answer(task,run.final_answer)
    assert v.success and not complete_l4_evidence(task,run)
    assert compute_reward(task,run,v,"shaped").counts["selection_with_evidence"] == 0


def test_calculation_before_obtaining_inputs_is_not_complete_evidence():
    task = TaskGenerator(40000).generate_task(4)
    reference,_ = fixture_run(task)
    calls = reference.events[:-1]
    outputs = [e.raw_output for e in calls if e.interaction.tool_call.tool_name == "calculator"]
    outputs += [e.raw_output for e in calls if e.interaction.tool_call.tool_name == "lookup_company"]
    outputs += [reference.events[-1].raw_output]
    run = PromptAgent(ScriptedBackend(outputs),32).run(agent_task_view(task),Environment(40000))
    v = verify_final_answer(task,run.final_answer)
    assert v.success and not complete_l4_evidence(task,run)


@pytest.mark.parametrize("tool,arguments",[("lookup_company",{}),
    ("lookup_company",{"company":"missing","field":"profit"}),
    ("lookup_company",{"company":"Company A","field":"bad"}),
    ("calculator",{"expression":"2**3"})])
def test_illegal_arguments_are_penalties_not_execution_rewards(tool,arguments):
    task = TaskGenerator(40000).generate_task(1)
    outputs = [action(tool,**arguments),json.dumps({"type":"final","answer":task.ground_truth})]
    run = PromptAgent(ScriptedBackend(outputs)).run(agent_task_view(task),Environment(40000))
    reward = compute_reward(task,run,verify_final_answer(task,run.final_answer),"shaped")
    assert reward.counts["invalid"] == 1 and reward.components["invalid"] == -.1
    assert reward.counts["legal_completion"] == 0


def test_backend_failure_aborts_instead_of_creating_zero_reward():
    task = TaskGenerator(40000).generate_task(1)
    run = PromptAgent(ScriptedBackend([])).run(agent_task_view(task),Environment(40000))
    with pytest.raises(RuntimeError,match="Infrastructure failure"):
        compute_reward(task,run,verify_final_answer(task),"outcome_only")


@pytest.mark.parametrize("expression,p,r,expected",[
    ("99/1376",99,1376,True),("99*(1/1376)",99,1376,True),
    ("(99*2)/(1376*2)",99,1376,True),("+(99)/+(1376)",99,1376,True),
    ("0.07194767441860465",99,1376,False),("99/1376 + 1",99,1376,False),
    ("99 - 1376",99,1376,False),("1*(1/7)",1,7,True),("0/7",0,7,True),
    ("5/5",5,5,True),("__import__('os')",99,1376,False),
    ("(99/1376)+(99-2)*(99-7)*(99-3)*(99-17)",99,1376,False)])
def test_exact_margin_provenance(expression,p,r,expected):
    assert _margin_expression(expression,p,r) == expected


@pytest.mark.parametrize("difficulty",[1,2,3,4])
def test_outcome_only_independent_of_trajectory(difficulty):
    task = TaskGenerator(40000).generate_task(difficulty,1)
    run = PromptAgent(ScriptedBackend([json.dumps({"type":"final","answer":task.ground_truth})])).run(agent_task_view(task),Environment(40000))
    v = verify_final_answer(task,run.final_answer)
    assert compute_reward(task,run,v,"outcome_only").total == 1


def test_invalid_redundant_and_no_final_counts():
    task = TaskGenerator(40000).generate_task(2,1)
    responses = ["not json",action("unknown"),action("list_companies"),action("list_companies")]
    run = PromptAgent(ScriptedBackend(responses),4).run(agent_task_view(task),Environment(40000))
    reward = compute_reward(task,run,verify_final_answer(task),"shaped")
    assert reward.counts["invalid"] == 2 and reward.counts["redundant"] == 1
    assert reward.total == pytest.approx(-0.27)
