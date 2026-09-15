"""Training gates and configuration validation without HF weights or a GPU."""
from dataclasses import asdict,replace
import json
from pathlib import Path

import pytest

from grpo.policy import adapter_hash
from grpo.schemas import GRPOConfig,TaskCoordinate
from grpo.trainer import require_probe,dev_guards,dev_selection_key
from scripts.train_grpo import main
from tasks.generator import TaskGenerator


@pytest.fixture
def valid_probe(tmp_path):
    adapter = tmp_path/"adapter"
    adapter.mkdir()
    (adapter/"adapter_config.json").write_text("{}")
    (adapter/"adapter_model.safetensors").write_bytes(b"fixture")
    config = GRPOConfig(str(adapter),str(tmp_path/"train"),"fixture")
    rows = [asdict(TaskCoordinate(seed,4,0,TaskGenerator(seed).generate_task(4).task_id)) for seed in range(40000,40020)]
    report = {"passed":True,"optimizer_updates":0,"group_count":20,"rollout_count":80,
        "mixed_reward_group_count":5,"zero_advantage_group_count":15,"alignment":{"passed":True},
        "config":asdict(config),"initial_sft_adapter_sha256":adapter_hash(adapter),
        "coordinates":rows,"task_ids":[r["task_id"] for r in rows]}
    directory = tmp_path/"probe"
    directory.mkdir()
    path = directory/"probe_report.json"
    path.write_text(json.dumps(report))
    (directory/"tasks.json").write_text(json.dumps({"splits":{"train":{"coordinates":rows}}}))
    return config,path,report


def test_probe_required_before_optimizer(valid_probe):
    config,path,report = valid_probe
    assert require_probe(path,config) == report
    with pytest.raises(ValueError,match="mandatory"): require_probe(None,config)


@pytest.mark.parametrize("key,value",[("passed",False),("optimizer_updates",1),("group_count",19),
    ("rollout_count",79),("mixed_reward_group_count",0),("zero_advantage_group_count",20),("alignment",{"passed":False})])
def test_failed_probe_rejected(valid_probe,key,value):
    config,path,report = valid_probe
    report[key] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError): require_probe(path,config)


@pytest.mark.parametrize("change",[{"group_size":8},{"temperature":1.},{"dtype":"float32"},{"reward_variant":"shaped"},{"seed":3}])
def test_probe_cannot_silently_change_diagnostic_config(valid_probe,change):
    config,path,_ = valid_probe
    with pytest.raises(ValueError,match="configuration mismatch"):
        require_probe(path,replace(config,**change))


def test_probe_cannot_use_test_coordinate(valid_probe):
    config,path,report = valid_probe
    task = TaskGenerator(30000).generate_task(4)
    report["coordinates"][0] = asdict(TaskCoordinate(30000,4,0,task.task_id))
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="train-only"): require_probe(path,config)


def test_probe_rejects_changed_sft_weights(valid_probe):
    config,path,_ = valid_probe
    (Path(config.sft_adapter)/"adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError,match="different starting"): require_probe(path,config)


@pytest.mark.parametrize("options",[{"group_size":1},{"temperature":0},{"top_p":.9},{"top_p":True},
    {"seed":-1},{"max_steps":0},{"learning_rate":float("nan")},{"clip_epsilon":1},{"kl_beta":-1}])
def test_invalid_training_configuration(options):
    with pytest.raises(ValueError): GRPOConfig("adapter","out","manifest",**options)


def test_training_without_probe_exits_before_loading_or_updates(tmp_path,monkeypatch):
    monkeypatch.setattr("scripts.train_grpo.GRPOPolicy",lambda *a:pytest.fail("Model must not load before gate"))
    with pytest.raises(SystemExit):
        main(["--smoke","--sft-adapter","adapter","--sft-manifest","manifest","--output-dir",str(tmp_path/"out")])
    assert not (tmp_path/"out").exists()


def test_refused_existing_run_is_not_modified(tmp_path):
    (tmp_path/"original.json").write_text("preserve")
    with pytest.raises(SystemExit):
        main(["--probe-only","--sft-adapter","adapter","--sft-manifest","manifest","--output-dir",str(tmp_path)])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["original.json"]


def dev_metrics():
    return {"by_difficulty":{f"L{l}":{"task_success_rate":{"value":1. if l < 4 else .5}} for l in (1,2,3,4)},
        "task_success_rate":{"value":.875},"average_agent_steps":{"value":4.},
        "counts":{"parse_error_count":0,"invalid_tool_call_count":0,"redundant_tool_call_count":0,"valid_trajectory_count":16}}


@pytest.mark.parametrize("key,value",[("parse_error_count",1),("invalid_tool_call_count",1),
    ("redundant_tool_call_count",1),("valid_trajectory_count",15)])
def test_dev_selection_rejects_observable_protocol_regression(key,value):
    before,after = dev_metrics(),dev_metrics()
    after["counts"][key] = value
    assert not all(dev_guards(before,after).values())


def test_dev_selection_preserves_early_levels_and_uses_l4_first():
    before,after = dev_metrics(),dev_metrics()
    after["by_difficulty"]["L4"]["task_success_rate"]["value"] = .75
    after["task_success_rate"]["value"] = .9375
    assert all(dev_guards(before,after).values())
    assert dev_selection_key(after) > dev_selection_key(before)
    after["by_difficulty"]["L1"]["task_success_rate"]["value"] = .75
    assert not dev_guards(before,after)["l1_l3_preserved"]


def test_tool_selection_diagnostic_is_not_task_success_or_eligibility():
    before,after = dev_metrics(),dev_metrics()
    before["tool_selection_accuracy"] = {"value":1.}
    after["tool_selection_accuracy"] = {"value":0.}
    assert all(dev_guards(before,after).values())
    assert dev_selection_key(before) == dev_selection_key(after)
