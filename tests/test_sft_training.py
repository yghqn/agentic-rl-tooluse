"""Network-free tests of explicit labels, audit gates and LoRA configuration."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from sft.dataset import DatasetConfig, build_dataset, export_dataset
from sft.preprocessing import (
    AssistantOnlyCollator, SpanAlignmentError, preprocess_dataset,
    require_preprocessing_gate, tokenize_sample, validate_sample,
)
from sft.training import (
    TrainingConfig, build_lora_model, lora_configuration, load_training_tokenizer,
    parameter_counts, require_persisted_gate, run_preprocessing, train, validate_adapter_identity,
)


class FixtureTokenizer:
    """A fixture-owned template, not a production fallback training template."""

    is_fast = True
    eos_token = "<end>"
    eos_token_id = 1
    pad_token_id = 1
    all_special_tokens = ["<start>", "<end>"]
    special_tokens_map = {"eos_token": "<end>"}
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"fixture":true}')

    def get_chat_template(self):
        return "fixture-owned-template"

    def apply_chat_template(self, messages, *, chat_template, tokenize, add_generation_prompt):
        assert chat_template == self.get_chat_template() and tokenize is False
        text = "".join(f"<start>{m['role']}\n{m['content']}<end>\n" for m in messages)
        return text + ("<start>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, *, add_special_tokens, truncation, return_attention_mask):
        assert add_special_tokens is False and truncation is False
        ids = []
        while text:
            special = next((s for s in self.all_special_tokens if text.startswith(s)), None)
            if special:
                ids.append(1 if special == self.eos_token else 2)
                text = text[len(special):]
            else:
                ids.append(ord(text[0]) + 10)
                text = text[1:]
        return {"input_ids": ids}


@pytest.fixture(scope="module")
def bundle():
    return build_dataset(DatasetConfig(train_size=16, dev_size=16, test_size=16))


@pytest.fixture
def dataset(tmp_path, bundle):
    export_dataset(bundle, tmp_path)
    return tmp_path


@pytest.mark.parametrize("difficulty,index", [(1,0), (2,0), (2,1), (3,0), (4,0)])
def test_assistant_only_masking_all_task_types(bundle, difficulty, index):
    sample = [d for d in bundle.records["train"] if d.task.difficulty == difficulty][index].chat_sample("train")
    tokenizer = FixtureTokenizer()
    features, spans = tokenize_sample(sample, tokenizer)
    owned = set()
    for span in spans:
        start, end = span["start"], span["end"]
        owned.update(range(start, end))
        assert features["labels"][start:end] == features["input_ids"][start:end]
        assert features["labels"][end - 1] == tokenizer.eos_token_id
        decoded = "".join(chr(t - 10) for t in features["labels"][start:end - 1])
        assert decoded == sample["messages"][span["message_index"]]["content"]
        assert features["labels"][start - 1] == -100  # header newline
        if end < len(features["labels"]):
            assert features["labels"][end] == -100  # separator newline
    assert all(label == -100 for i, label in enumerate(features["labels"]) if i not in owned)
    assert spans[-1]["message_index"] == len(sample["messages"]) - 1
    assert spans[-1]["action_type"] == "final"


@pytest.mark.parametrize("mutation", ["indices", "final_index", "ordering", "empty", "reasoning", "control", "observation"])
def test_bad_samples_rejected(bundle, mutation):
    sample = deepcopy(bundle.records["train"][0].chat_sample("train"))
    if mutation == "indices":
        sample["assistant_message_indices"][0] = True
    elif mutation == "final_index":
        sample["assistant_message_indices"].pop()
    elif mutation == "ordering":
        sample["messages"][3]["role"] = "tool"
    elif mutation == "empty":
        sample["messages"][2]["content"] = ""
    elif mutation == "reasoning":
        sample["messages"][2]["content"] = 'Thinking... {"type":"final","answer":2}'
    elif mutation == "control":
        sample["messages"][1]["content"] += "<end>"
    else:
        sample["messages"][3]["content"] = "Tool observation (calculator): {}"
    with pytest.raises(ValueError):
        tokenize_sample(sample, FixtureTokenizer())


def test_no_test_file_reads_and_deterministic_preprocessing(dataset, monkeypatch):
    original = Path.read_bytes

    def guarded(path):
        assert path.name not in {"test.sft.jsonl", "test.trajectories.jsonl", "dev.trajectories.jsonl", "train.trajectories.jsonl"}
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    first = preprocess_dataset(dataset, FixtureTokenizer(), 10000, {"revision":"pinned"})
    second = preprocess_dataset(dataset, FixtureTokenizer(), 10000, {"revision":"pinned"})
    assert first == second and first.audit["passed"]
    require_preprocessing_gate(first)
    assert set(first.features) == {"train", "dev"}
    for stats in first.audit["splits"].values():
        assert stats["sample_count"] == 16
        assert set(stats["sequence_length"]) == {"min", "median", "p95", "max"}
        assert stats["supervised_token_count"] > 0
        assert 0 < stats["supervised_token_ratio"] < 1


def test_overlength_not_truncated_and_gate_rejects(dataset):
    result = preprocess_dataset(dataset, FixtureTokenizer(), 1, {})
    assert not result.audit["passed"]
    assert result.audit["splits"]["train"]["overlength_sample_count"] == 16
    assert len(result.features["train"][0]["input_ids"]) > 1
    with pytest.raises(ValueError, match="hard gate"):
        require_preprocessing_gate(result)


def test_span_alignment_failure_is_fail_closed(dataset):
    class BadTokenizer(FixtureTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            return text + "BAD" if kwargs["add_generation_prompt"] else text

    result = preprocess_dataset(dataset, BadTokenizer(), 10000, {})
    assert result.audit["splits"]["train"]["assistant_span_alignment_failure_count"] == 16
    with pytest.raises(ValueError, match="hard gate"):
        require_preprocessing_gate(result)


def test_token_prefix_alignment_is_checked(bundle):
    class BadTokenizer(FixtureTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text.endswith("assistant\n"):
                result["input_ids"][-1] = 999
            return result

    with pytest.raises(SpanAlignmentError):
        tokenize_sample(bundle.records["train"][0].chat_sample("train"), BadTokenizer())


def test_changed_features_and_zero_supervision_fail_gate(dataset):
    result = preprocess_dataset(dataset, FixtureTokenizer(), 10000, {})
    result.features["train"][0]["labels"] = [-100] * len(result.features["train"][0]["labels"])
    with pytest.raises(ValueError, match="hard gate"):
        require_preprocessing_gate(result)


def test_checksum_change_rejected(dataset):
    with (dataset / "train.sft.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        preprocess_dataset(dataset, FixtureTokenizer(), 10000, {})


def test_declared_split_overlap_rejected(dataset):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["split_overlap"]["train_test"]["task_id_overlap_count"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlap"):
        preprocess_dataset(dataset, FixtureTokenizer(), 10000, {})


def test_inconsistent_manifest_seed_rejected(dataset):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["test"]["coordinates"][0]["seed"] = 10000
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coordinate"):
        preprocess_dataset(dataset, FixtureTokenizer(), 10000, {})


@pytest.mark.parametrize("duplicate", ["chat", "arithmetic"])
def test_cross_split_content_duplicates_fail_gate(dataset, duplicate):
    train_rows = [json.loads(line) for line in (dataset / "train.sft.jsonl").read_text().splitlines()]
    dev_path = dataset / "dev.sft.jsonl"
    dev_rows = [json.loads(line) for line in dev_path.read_text().splitlines()]
    if duplicate == "chat":
        dev_rows[0]["messages"] = deepcopy(train_rows[0]["messages"])
    else:
        dev_rows[4]["messages"][2] = deepcopy(train_rows[5]["messages"][2])
    dev_path.write_text("\n".join(json.dumps(row) for row in dev_rows) + "\n", encoding="utf-8")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["dev.sft.jsonl"]["sha256"] = hashlib.sha256(dev_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result = preprocess_dataset(dataset, FixtureTokenizer(), 10000, {})
    assert result.audit["splits"]["dev"]["invalid_sample_count"] >= 1
    with pytest.raises(ValueError, match="hard gate"):
        require_preprocessing_gate(result)


def test_test_samples_cannot_be_preprocessed(bundle):
    with pytest.raises(ValueError, match="train/dev"):
        validate_sample(bundle.records["test"][0].chat_sample("test"), "test")


def test_padding_does_not_mask_real_eos(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(long="long", tensor=lambda value, dtype: value))
    items = [{"input_ids":[9,1], "labels":[9,1], "attention_mask":[1,1]},
             {"input_ids":[1], "labels":[1], "attention_mask":[1]}]
    result = AssistantOnlyCollator(1)(items)
    assert result["input_ids"] == [[9,1],[1,1]]
    assert result["labels"] == [[9,1],[1,-100]]
    assert result["attention_mask"] == [[1,1],[1,0]]


@pytest.fixture
def gated(dataset, tmp_path):
    config = TrainingConfig(str(dataset), str(tmp_path / "training"), device="cpu", dtype="float32", max_seq_length=10000)
    tokenizer = FixtureTokenizer()
    from sft.preprocessing import tokenizer_fingerprint
    identity = {"base_model_name_or_path":config.model_name_or_path, "base_model_revision":"a" * 40,
                "tokenizer_identity":config.model_name_or_path, "tokenizer_revision":"a" * 40,
                "chat_template_sha256":hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
                "tokenizer_sha256":tokenizer_fingerprint(tokenizer)}
    result = preprocess_dataset(dataset, tokenizer, config.max_seq_length, identity)
    return config, tokenizer, identity, result


def test_prior_preprocess_only_is_required_before_trainer(gated, monkeypatch):
    config, tokenizer, identity, result = gated
    monkeypatch.setitem(sys.modules, "transformers", None)  # gate must precede imports
    with pytest.raises(ValueError, match="preprocess-only"):
        train(config, tokenizer, result, identity)


def test_persisted_gate_and_stale_audit(gated):
    config, tokenizer, identity, result = gated
    assert run_preprocessing(config, tokenizer, identity) == result
    require_persisted_gate(config, result, identity)
    changed = deepcopy(result)
    changed.audit["identity"]["base_model_revision"] = "b" * 40
    with pytest.raises(ValueError, match="stale"):
        require_persisted_gate(config, changed, changed.audit["identity"])


def test_failed_gate_never_imports_trainer(gated, monkeypatch):
    config, tokenizer, identity, result = gated
    result.audit["splits"]["train"]["assistant_span_alignment_failure_count"] = 1
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="hard gate"):
        train(config, tokenizer, result, identity)


def test_different_tokenizer_rejected_before_trainer(gated, monkeypatch):
    config, tokenizer, identity, result = gated
    run_preprocessing(config, tokenizer, identity)
    tokenizer.get_chat_template = lambda: "changed-template"
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="tokenizer differs"):
        train(config, tokenizer, result, identity)


@pytest.mark.parametrize("key", ["base_model_name_or_path", "base_model_revision", "tokenizer_identity",
                                 "tokenizer_revision", "chat_template_sha256", "tokenizer_sha256"])
def test_adapter_identity_mismatch_rejected(gated, key):
    identity = gated[2]
    changed = dict(identity, **{key:"different"})
    with pytest.raises(ValueError, match=key):
        validate_adapter_identity(identity, changed)
    validate_adapter_identity(identity, identity)


def test_lora_defaults_and_actual_count_check(gated):
    lora = lora_configuration(gated[0])
    assert (lora["r"], lora["lora_alpha"], lora["lora_dropout"]) == (8,16,0)
    assert set(lora["target_modules"]) == {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    assert lora["bias"] == "none" and lora["modules_to_save"] is None
    params = [("base.weight", SimpleNamespace(requires_grad=False, numel=lambda:100)),
              ("base.q.lora_A.default.weight", SimpleNamespace(requires_grad=True, numel=lambda:8))]
    model = SimpleNamespace(parameters=lambda:[p for _,p in params], named_parameters=lambda:params)
    assert parameter_counts(model)["trainable_parameters"] == 8
    params[0][1].requires_grad = True
    with pytest.raises(ValueError, match="Non-LoRA"):
        parameter_counts(model)


def test_training_tokenizer_uses_resolved_own_snapshot(gated, monkeypatch):
    from sft import training
    config, tokenizer, identity, _ = gated
    calls = []
    fake = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(
            _commit_hash=identity["base_model_revision"], model_type="qwen2", max_position_embeddings=32768)),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda name, **kw: calls.append((name,kw)) or tokenizer))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(training, "tokenizer_source", lambda name, revision, loading: ("exact-snapshot",revision))
    actual_tokenizer, actual_identity = load_training_tokenizer(config)
    assert actual_tokenizer is tokenizer and actual_identity == identity
    assert calls[0][0] == "exact-snapshot" and calls[0][1]["use_fast"] is True
    assert calls[0][1]["trust_remote_code"] is False


def test_preprocess_only_cli_never_loads_model_or_trainer(gated, monkeypatch, capsys):
    from scripts import train_sft
    config, tokenizer, identity, _ = gated
    monkeypatch.setattr(train_sft, "load_training_tokenizer", lambda c:(tokenizer,identity))
    monkeypatch.setattr(train_sft, "train", lambda *a:pytest.fail("Trainer must not start"))
    monkeypatch.setattr(train_sft, "inspect_model", lambda *a:pytest.fail("Model must not load"))
    assert train_sft.main(["--dataset-dir",config.dataset_dir,"--output-dir",config.output_dir,
                           "--max-seq-length","10000","--preprocess-only"]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert (Path(config.output_dir) / "length_stats.json").is_file()


@pytest.mark.parametrize("kwargs", [{"max_seq_length":0}, {"lora_rank":True}, {"lora_dropout":1},
                                    {"learning_rate":float("nan")}, {"dtype":"int8"}, {"seed":-1}])
def test_invalid_training_configuration(kwargs, tmp_path):
    with pytest.raises(ValueError):
        TrainingConfig(str(tmp_path / "data"), str(tmp_path / "out"), **kwargs)


def test_mock_trainer_preserves_explicit_labels_and_split_isolation(gated, monkeypatch):
    """Exercise train wiring with fake packages only; never start a real Trainer."""
    from sft import training
    config, tokenizer, identity, result = gated
    run_preprocessing(config, tokenizer, identity)
    state = SimpleNamespace(calls=[], log_history=[])
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True), dtype="float32",
                            save_pretrained=lambda *a, **k:state.calls.append("save_adapter"))
    monkeypatch.setattr(training, "build_lora_model", lambda *a:(model,{"trainable_parameters":8}))
    tokenizer.save_pretrained = lambda *a:state.calls.append("save_tokenizer")

    class FakeArguments:
        def __init__(self, **kw):
            self.kw = kw
            self.device = SimpleNamespace(type="cpu", index=None)

        def to_dict(self):
            return self.kw

    class FakeTrainer:
        def __init__(self, **kw):
            state.kw = kw
            self.state = state

        def train(self):
            state.log_history.append({"loss":1.0})

        def evaluate(self):
            state.log_history.append({"eval_loss":1.1})

        def save_state(self):
            state.calls.append("save_state")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fixture",version=SimpleNamespace(cuda=None),
                         device=lambda _:SimpleNamespace(type="cpu",index=None)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="fixture",Trainer=FakeTrainer,TrainingArguments=FakeArguments))
    for name in ("accelerate", "peft"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__version__="fixture"))
    record = train(config, tokenizer, result, identity)
    assert state.kw["train_dataset"] is result.features["train"]
    assert state.kw["eval_dataset"] is result.features["dev"]
    assert isinstance(state.kw["data_collator"], AssistantOnlyCollator)
    assert state.kw["args"].kw["label_names"] == ["labels"]
    assert record["dataset"]["manifest_sha256"] == result.audit["manifest_sha256"]
    assert record["identity"] == identity
    assert state.calls == ["save_adapter","save_tokenizer","save_state"]
    assert len((Path(config.output_dir) / "loss_history.jsonl").read_text().splitlines()) == 2
