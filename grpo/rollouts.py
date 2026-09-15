"""Full PromptAgent rollouts, deterministic replay, streaming audit and variance."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics

from agent.agent import PromptAgent
from agent.parser import parse_model_output, ParseError
from agent.prompts import build_messages
from environment.environment import Environment
from evaluation.verifier import verify_final_answer
from grpo.loss import group_advantages
from grpo.rewards import compute_reward
from grpo.schemas import RolloutRecord, RolloutGroup, TurnTrace, task_from_coordinate
from tasks.generator import agent_task_view


def derived_seed(seed, version, task_id, member):
    return int.from_bytes(hashlib.sha256(f"{seed}:{version}:{task_id}:{member}".encode()).digest()[:4],"big")


def append_jsonl(path: Path, value):
    with path.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")
        stream.flush()


class RecordingBackend:
    def __init__(self, backend):
        self.backend = backend
        self.turns = []

    def generate(self, messages):
        turn = TurnTrace(**self.backend.generate_with_trace(messages))
        turn.validate()
        self.turns.append(turn)
        return turn.raw_output


def validate_replay(task, run, seed):
    """Replay failures too: consistency is distinct from task/protocol success."""
    environment = Environment(seed=seed)
    for index,event in enumerate(run.events,start=1):
        if event.step_index != index:
            raise ValueError("Non-consecutive replay events")
        if event.backend_error:
            raise RuntimeError("Backend failure; abort whole batch, do not manufacture reward")
        try:
            parsed = parse_model_output(event.raw_output)
        except ParseError as exc:
            if (event.parse_error is None or event.parse_error.error_code != exc.error_code
                or event.parse_error.message != str(exc) or event.interaction is not None or event.final_action is not None):
                raise ValueError("Parse replay mismatch")
            continue
        if event.interaction:
            if parsed != event.interaction.tool_call or environment.execute_tool_call(parsed) != event.interaction.tool_result:
                raise ValueError("Environment replay mismatch")
        elif parsed != event.final_action:
            raise ValueError("Final replay mismatch")
        elif index != len(run.events):
            raise ValueError("Replay continued after final answer")
    if (run.termination_reason == "final_answer") != run.has_final_answer:
        raise ValueError("Replay termination mismatch")


def collect_group(policy, coordinate, version: int, directory: Path) -> RolloutGroup:
    import torch
    config = policy.config
    task = task_from_coordinate(coordinate)
    view = agent_task_view(task)
    policy.switch("policy")
    group_id = f"group-{version}-{coordinate.task_id}"
    members = []
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    for index in range(config.group_size):
        seed = derived_seed(config.seed,version,coordinate.task_id,index)
        backend = RecordingBackend(policy.backend)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            run = PromptAgent(backend,config.max_steps).run(view,Environment(seed=coordinate.seed))
        try:
            validate_replay(task,run,coordinate.seed)
            if len(backend.turns) != len(run.events):
                raise ValueError("Missing token traces")
            for event,turn in zip(run.events,backend.turns,strict=True):
                expected = policy.backend._copy_messages(build_messages(view,run.events[:event.step_index-1]))
                if turn.messages != expected or turn.raw_output != event.raw_output:
                    raise ValueError("Messages/token trace event alignment failure")
            verification = verify_final_answer(task,run.final_answer) if run.has_final_answer else verify_final_answer(task)
            reward = compute_reward(task,run,verification,config.reward_variant)
        except Exception as exc:
            append_jsonl(directory/"rollout_errors.jsonl",{"group_id":group_id,"member_index":index,"seed":seed,
                         "run":asdict(run),"turns":[asdict(t) for t in backend.turns],"error":str(exc)})
            raise
        record = RolloutRecord(f"{group_id}-member-{index}",coordinate,group_id,index,seed,version,
                               run,backend.turns,verification,reward)
        append_jsonl(directory/"rollouts.jsonl",asdict(record))
        members.append(record)
    advantages = group_advantages([m.reward.total for m in members],config.std_floor)
    append_jsonl(directory/"groups.jsonl",{"group_id":group_id,"rollout_ids":[m.rollout_id for m in members],
                 "rewards":[m.reward.total for m in members],"advantages":advantages})
    return RolloutGroup(group_id,members)


def variance_report(groups: list[RolloutGroup], std_floor: float) -> dict:
    mixed,mixed_outcome,all_success,all_failure,equal,zero = 0,0,0,0,0,0
    per_group, rewards, all_trajectories = [],[],set()
    for group in groups:
        rs = [m.reward.total for m in group.members]
        successes = sum(m.verification.success for m in group.members)
        mixed += max(rs)-min(rs) > 1e-8
        mixed_outcome += 0 < successes < len(rs)
        all_success += successes == len(rs)
        all_failure += successes == 0
        equal += max(rs)-min(rs) <= 1e-8
        advantages = group_advantages(rs,std_floor)
        zero += all(a == 0 for a in advantages)
        answers = {json.dumps(m.run.final_answer,sort_keys=True,allow_nan=False) if m.run.has_final_answer else "NO_FINAL"
                   for m in group.members}
        trajectories = {json.dumps(asdict(m.run),sort_keys=True,allow_nan=False) for m in group.members}
        all_trajectories.update(trajectories)
        per_group.append({"group_id":group.group_id,"unique_final_answer_count":len(answers),
                          "unique_final_answers":sorted(answers),"unique_trajectory_count":len(trajectories)})
        rewards.extend(rs)
    count = len(groups)
    parse_errors = sum(e.parse_error is not None for g in groups for m in g.members for e in m.run.events)
    invalid_actions = sum(m.reward.counts["invalid"] for g in groups for m in g.members)
    return {"group_count":count,"rollout_count":len(rewards),"mixed_reward_group_count":mixed,
            "mixed_outcome_group_count":mixed_outcome,
            "mixed_reward_group_rate":mixed/count if count else 0,"all_success_group_count":all_success,
            "all_failure_group_count":all_failure,"equal_reward_group_count":equal,"zero_advantage_group_count":zero,
            "unique_trajectory_count":len(all_trajectories),"per_group":per_group,
            "reward_mean":statistics.mean(rewards) if rewards else 0,
            "reward_std":statistics.pstdev(rewards) if rewards else 0,
            "parse_error_count":parse_errors,"invalid_action_count":invalid_actions,
            "invalid_tool_call_count":invalid_actions-parse_errors,
            "backend_error_count":sum(e.backend_error is not None for g in groups for m in g.members for e in m.run.events),
            "execution_error_count":sum(s.tool_result.error_code == "EXECUTION_ERROR" for g in groups for m in g.members for s in m.run.tool_steps),
            "redundant_tool_call_count":sum(m.reward.counts["redundant"] for g in groups for m in g.members),
            "task_success_count":sum(m.verification.success for g in groups for m in g.members),
            "optimizer_updates":0}
