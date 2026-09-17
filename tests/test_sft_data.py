"""Offline data construction, strict splits, causal leakage checks and replay."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from agent.agent import PromptAgent, ScriptedBackend
from agent.hf_backend import HuggingFaceBackend
from agent.parser import FinalAnswer, parse_model_output
from agent.prompts import build_messages
from environment.database import SyntheticCompanyDatabase
from environment.environment import Environment
from evaluation.verifier import verify_final_answer
from scripts.build_sft_data import main
from sft.dataset import (
    EXPORT_FILES, SPLITS, DatasetConfig, Demonstration, audit_export, build_dataset,
    chat_fingerprint, check_split_isolation, export_dataset,
)
from sft.oracle import generate_oracle_trajectory
from sft.validation import arithmetic_key, to_sft_messages, validate_demonstration
from tasks.generator import TaskGenerator, agent_task_view


TYPES = [(1, 0, "single_retrieval", 1), (2, 0, "list_companies", 1),
         (2, 1, "arithmetic", 1), (3, 0, "profit_margin", 3), (4, 0, "highest_profit_margin", 9)]


def demonstration(level=3, index=0, seed=42):
    task = TaskGenerator(seed).generate_task(level, index)
    database = SyntheticCompanyDatabase(seed)
    run = generate_oracle_trajectory(task, database)
    verification = validate_demonstration(task, database, run)
    return Demonstration(task, seed, index, run, verification)


@pytest.fixture(scope="module")
def smoke():
    return build_dataset()


@pytest.mark.parametrize("level,index,kind,calls", TYPES)
def test_all_task_types_replay_parser_and_verifier(level, index, kind, calls):
    item = demonstration(level, index)
    assert item.task.verification_spec["task_type"] == kind
    assert len(item.run.tool_steps) == calls
    assert len(item.run.events) == calls + 1
    assert all(step.tool_result.success for step in item.run.tool_steps)
    assert isinstance(parse_model_output(item.run.events[-1].raw_output), FinalAnswer)
    assert verify_final_answer(item.task, item.run.final_answer).success
    sample = item.chat_sample("train")
    assert validate_demonstration(item.task, SyntheticCompanyDatabase(42), item.run,
                                  messages=item.raw_record("train")["messages"], sft_sample=sample).success
    replay = PromptAgent(ScriptedBackend([event.raw_output for event in item.run.events])).run(
        agent_task_view(item.task), Environment(42))
    assert asdict(replay) == asdict(item.run)


def test_oracle_determinism_and_ordering():
    item = demonstration(4)
    assert asdict(generate_oracle_trajectory(item.task, SyntheticCompanyDatabase(42))) == asdict(item.run)
    companies = sorted(item.task.metadata["companies"])
    for offset, company in enumerate(companies):
        calls = item.run.tool_steps[3 * offset:3 * offset + 3]
        assert [step.tool_call.tool_name for step in calls] == ["lookup_company", "lookup_company", "calculator"]
        assert calls[0].tool_call.arguments == {"company":company,"field":"profit"}
        assert calls[1].tool_call.arguments == {"company":company,"field":"revenue"}


def test_oracle_does_not_copy_tolerance_accepted_ground_truth():
    task = TaskGenerator(42).generate_task(3)
    true_value = task.ground_truth
    task.ground_truth += 1e-12  # Valid within the existing answer contract.
    run = generate_oracle_trajectory(task, SyntheticCompanyDatabase(42))
    assert run.final_answer == true_value
    assert run.final_answer != task.ground_truth
    assert validate_demonstration(task, SyntheticCompanyDatabase(42), run).success


def test_tie_breaking_and_identical_calculator_cache(monkeypatch):
    original = SyntheticCompanyDatabase.lookup

    def tied(database, company, field):
        return {"profit":10,"revenue":100}.get(field, original(database, company, field))

    monkeypatch.setattr(SyntheticCompanyDatabase, "lookup", tied)
    item = demonstration(4)
    assert item.run.final_answer == {"company":min(item.task.metadata["companies"]),"profit_margin":0.1}
    assert len(item.run.tool_steps) == 7  # six lookups, one reusable calculator call
    assert validate_demonstration(item.task, SyntheticCompanyDatabase(42), item.run).success


def test_matching_environment_is_required():
    task = TaskGenerator(42).generate_task(1)
    with pytest.raises(ValueError, match="seed"):
        generate_oracle_trajectory(task, SyntheticCompanyDatabase(43))
    item = demonstration()
    with pytest.raises(ValueError, match="seed"):
        validate_demonstration(item.task, SyntheticCompanyDatabase(43), item.run)


def test_validation_does_not_require_oracle_order_or_call_oracle(monkeypatch):
    item = demonstration()
    responses = [event.raw_output for event in item.run.events]
    responses[0], responses[1] = responses[1], responses[0]
    alternate = PromptAgent(ScriptedBackend(responses)).run(agent_task_view(item.task), Environment(42))

    def forbidden(*args):
        raise AssertionError("Verifier/replay must not call the oracle")

    monkeypatch.setattr("sft.oracle.generate_oracle_trajectory", forbidden)
    assert verify_final_answer(item.task, alternate.final_answer).success
    assert validate_demonstration(item.task, SyntheticCompanyDatabase(42), alternate).success


def test_verifier_accepts_answer_without_oracle_trajectory():
    item = demonstration()
    final_only = PromptAgent(ScriptedBackend([item.run.events[-1].raw_output])).run(agent_task_view(item.task), Environment(42))
    assert verify_final_answer(item.task, final_only.final_answer).success
    # Dataset quality is intentionally stricter, without changing Task Success.
    with pytest.raises(ValueError, match="Missing"):
        validate_demonstration(item.task, SyntheticCompanyDatabase(42), final_only)


def test_calculator_cannot_use_hidden_values_before_observation():
    item = demonstration()
    responses = [event.raw_output for event in item.run.events]
    responses = [responses[2], responses[0], responses[1], responses[3]]
    run = PromptAgent(ScriptedBackend(responses)).run(agent_task_view(item.task), Environment(42))
    assert verify_final_answer(item.task, run.final_answer).success
    with pytest.raises(ValueError, match="before legitimate"):
        validate_demonstration(item.task, SyntheticCompanyDatabase(42), run)


@pytest.mark.parametrize("mutation", ["observation", "answer", "parse", "redundant", "invalid", "off_target", "incomplete"])
def test_bad_demonstrations_rejected(mutation):
    item = demonstration()
    responses = [event.raw_output for event in item.run.events]
    if mutation == "observation":
        item.run.tool_steps[0].tool_result.output += 1
    elif mutation == "answer":
        responses[-1] = '{"type":"final","answer":999}'
    elif mutation == "parse":
        responses[0] = "thinking then a JSON action"
    elif mutation == "redundant":
        responses.insert(1, responses[0])
    elif mutation == "invalid":
        responses.insert(0, '{"type":"tool_call","tool_name":"unknown","arguments":{}}')
    elif mutation == "off_target":
        responses.insert(0, '{"type":"tool_call","tool_name":"list_companies","arguments":{}}')
    else:
        responses.pop()
    if mutation != "observation":
        item.run = PromptAgent(ScriptedBackend(responses), max_steps=len(responses)).run(agent_task_view(item.task), Environment(42))
    with pytest.raises(ValueError):
        validate_demonstration(item.task, SyntheticCompanyDatabase(42), item.run)


def test_public_numeric_arithmetic_is_not_leakage():
    item = demonstration(2, 1)
    assert any(char.isdigit() for char in item.task.question)
    assert validate_demonstration(item.task, SyntheticCompanyDatabase(42), item.run).success


def test_arithmetic_question_cannot_include_a_supplied_answer():
    item = demonstration(2, 1)
    item.task.question += f" Answer: {item.task.ground_truth}"
    with pytest.raises(ValueError, match="outside its public expression"):
        validate_demonstration(item.task, SyntheticCompanyDatabase(42), item.run)


@pytest.mark.parametrize("mutation", ["extra_key", "reasoning", "ground_truth", "mask", "raw_messages", "question", "metadata", "sample_id"])
def test_leakage_and_schema_rejected(mutation):
    item = demonstration()
    sample = item.chat_sample("train")
    messages = item.raw_record("train")["messages"]
    if mutation == "extra_key":
        sample["ground_truth"] = item.task.ground_truth
    elif mutation == "reasoning":
        sample["messages"].insert(2, {"role":"assistant","content":"hidden reasoning"})
    elif mutation == "ground_truth":
        sample["messages"][1]["content"] += f" Answer: {item.task.ground_truth}"
    elif mutation == "mask":
        sample["assistant_message_indices"] = [0, 1]
    elif mutation == "raw_messages":
        messages.append({"role":"user","content":"verification_spec"})
    elif mutation == "question":
        item.task.question += f" Answer: {item.task.ground_truth}"
    elif mutation == "sample_id":
        sample["sample_id"] = "unstable-id"
    else:
        item.task.metadata["profit"] = 1234
    with pytest.raises(ValueError):
        validate_demonstration(item.task, SyntheticCompanyDatabase(42), item.run,
                               messages=messages, sft_sample=sample)


def test_chat_format_matches_hf_adapter_and_has_no_privileged_fields(smoke):
    for split in SPLITS:
        for item in smoke.records[split]:
            raw = item.raw_record(split)
            sample = item.chat_sample(split)
            assert sample["messages"] == HuggingFaceBackend._copy_messages(raw["messages"])
            assert sample["messages"][:2] == build_messages(agent_task_view(item.task))
            assert all(set(message) == {"role","content"} for message in sample["messages"])
            assert len(sample["assistant_message_indices"]) == len(item.run.events)
            for message in sample["messages"]:
                if message["role"] == "assistant":
                    parse_model_output(message["content"])
            serialized = json.dumps(raw) + json.dumps(sample)
            for forbidden in ("ground_truth", "verification_spec", '"metadata"', "<think>", "chain_of_thought"):
                assert forbidden not in serialized


def test_public_messages_independent_of_hidden_values(monkeypatch):
    before = demonstration()
    lookup = SyntheticCompanyDatabase.lookup

    def changed(database, company, field):
        return lookup(database, company, field) + {"profit":1,"revenue":11,"employees":17,"growth_rate":0.01}[field]

    monkeypatch.setattr(SyntheticCompanyDatabase, "lookup", changed)
    after = demonstration()
    assert before.chat_sample("train")["messages"][:2] == after.chat_sample("train")["messages"][:2]
    assert before.run.final_answer != after.run.final_answer


def test_smoke_quotas_strict_splits_and_content_deduplication(smoke):
    assert [len(smoke.records[split]) for split in SPLITS] == [64, 16, 16]
    for split in SPLITS:
        info = smoke.manifest["splits"][split]
        per_level = info["sample_count"] // 4
        assert info["counts_by_difficulty"] == {f"L{level}":per_level for level in range(1,5)}
        assert info["counts_by_task_type"]["list_companies"] == (1 if split == "train" else 0)
        assert info["counts_by_task_type"]["arithmetic"] == per_level - (1 if split == "train" else 0)
        order = [(item.task.difficulty, item.seed, item.task_index) for item in smoke.records[split]]
        assert order == sorted(order)
    assert all(not any(counts.values()) for counts in check_split_isolation(smoke.records).values())
    # Audit identity is not a proxy for content isolation.
    sample = smoke.records["train"][0].chat_sample("train")
    changed = {**sample, "sample_id":"another", "task_id":"another", "split":"test"}
    assert chat_fingerprint(sample) == chat_fingerprint(changed)
    assert arithmetic_key(" 2 + 3 ") == arithmetic_key("2+3")


@pytest.mark.parametrize("mutation", ["seed", "task_id", "chat", "arithmetic", "list", "within_split"])
def test_split_overlap_rejected(smoke, mutation):
    records = deepcopy(smoke.records)
    train, dev = records["train"][0], records["dev"][0]
    if mutation == "seed":
        dev.seed = train.seed
    elif mutation == "task_id":
        dev.task.task_id = train.task.task_id
    elif mutation == "chat":
        replacement = deepcopy(train)
        replacement.seed = dev.seed
        replacement.task.task_id = dev.task.task_id
        records["dev"][0] = replacement
    elif mutation == "arithmetic":
        source = next(item for item in records["train"] if item.task.verification_spec["task_type"] == "arithmetic")
        target = next(item for item in records["dev"] if item.task.verification_spec["task_type"] == "arithmetic")
        target.task.metadata["expression"] = source.task.metadata["expression"]
    elif mutation == "list":
        target = next(item for item in records["train"] if item.task.verification_spec["task_type"] == "list_companies")
        records["dev"].append(deepcopy(target))
    else:
        records["train"].append(deepcopy(train))
    with pytest.raises(ValueError):
        check_split_isolation(records)


@pytest.mark.parametrize("kwargs", [{"train_size":0}, {"dev_size":17}, {"test_size":True},
                                   {"max_candidates_per_level":0}, {"train_seed_start":True},
                                   {"dev_seed_start":10000}])
def test_invalid_dataset_config(kwargs):
    with pytest.raises(ValueError):
        DatasetConfig(**kwargs)


def test_candidate_limit_is_not_silent():
    with pytest.raises(ValueError, match="Candidate limit"):
        build_dataset(DatasetConfig(max_candidates_per_level=1))


def test_export_reproducibility_manifest_and_oracle_free_audit(smoke, tmp_path, monkeypatch):
    second = build_dataset()
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    manifest = export_dataset(smoke, first_dir)
    export_dataset(second, second_dir)
    for name in EXPORT_FILES:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    for name, info in manifest["files"].items():
        assert info["sha256"] == hashlib.sha256((first_dir / name).read_bytes()).hexdigest()
        assert info["row_count"] == len((first_dir / name).read_text().splitlines())
    assert "git_commit_hash" in manifest and "git_dirty" in manifest
    assert "timestamp" not in manifest

    def forbidden(*args):
        raise AssertionError("Audit must replay saved outputs, not regenerate oracle answers")

    monkeypatch.setattr("sft.dataset.generate_oracle_trajectory", forbidden)
    audit = audit_export(first_dir)
    assert [audit["splits"][split]["validation"]["replay_success_count"] for split in SPLITS] == [64,16,16]
    assert audit["split_overlap"] == manifest["split_overlap"]
    with pytest.raises(ValueError, match="already exist"):
        export_dataset(smoke, first_dir)


def test_export_rejects_mutated_order_and_verification(smoke, tmp_path):
    changed = deepcopy(smoke)
    changed.records["train"].reverse()
    with pytest.raises(ValueError, match="ordering"):
        export_dataset(changed, tmp_path / "order")
    changed = deepcopy(smoke)
    changed.records["train"][0].verification.normalized_answer = "tampered"
    with pytest.raises(ValueError, match="verification"):
        export_dataset(changed, tmp_path / "verification")


@pytest.mark.parametrize("mutation", ["checksum", "observation", "chat", "manifest", "order", "version"])
def test_saved_export_tampering_rejected(smoke, tmp_path, mutation):
    export_dataset(smoke, tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    name = "train.trajectories.jsonl" if mutation == "observation" else "train.sft.jsonl"
    file_path = tmp_path / name
    rows = [json.loads(line) for line in file_path.read_text().splitlines()]
    if mutation == "manifest":
        manifest["splits"]["train"]["validation"]["verifier_success_count"] = 0
    elif mutation == "version":
        manifest["schema_version"] = "unknown-version"
    else:
        if mutation == "observation":
            rows[0]["run"]["events"][0]["interaction"]["tool_result"]["output"] += 1
        elif mutation == "chat":
            rows[0]["messages"][1]["content"] += " ground_truth"
        elif mutation == "order":
            rows[0], rows[1] = rows[1], rows[0]
        file_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        if mutation == "checksum":
            manifest["files"][name]["sha256"] = "0" * 64
        else:
            manifest["files"][name]["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        audit_export(tmp_path)


def test_cli_smoke_and_audit_only(tmp_path, capsys):
    arguments = ["--output-dir",str(tmp_path / "smoke")]
    assert main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["splits"]["train"]["sample_count"] == 64
    assert main(arguments + ["--audit-only"]) == 0
    assert json.loads(capsys.readouterr().out) == output
    with pytest.raises(SystemExit):
        main(arguments)
