"""All HF tests use fake packages/tensors: no weights or optional dependencies."""

from contextlib import nullcontext
from copy import deepcopy
import json
from types import ModuleType, SimpleNamespace
import sys
from pathlib import Path

import pytest

from agent.agent import PromptAgent
from agent.hf_backend import HuggingFaceBackend, HuggingFaceConfig
from scripts.evaluate import main, evaluate_tasks, load_manifest_benchmark, validate_comparison
from sft.dataset import DatasetConfig, build_dataset, export_dataset
from sft.training import TrainingConfig, lora_configuration
from tasks.generator import TaskGenerator
from tasks.validators import environment_id_for_seed


class Tensor:
    def __init__(self, tokens):
        self.tokens = tokens
        self.shape = (1, len(tokens))
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, index):
        row, selected = index
        assert row == 0
        return self.tokens[selected]


@pytest.fixture
def fake_hf(monkeypatch):
    state = SimpleNamespace(loads=[], rendered=[], tokenized=[], generated=[], decoded=[], seeds=[], inference=0)
    tokenizer = SimpleNamespace(
        chat_template="own-template", name_or_path="fake/tokenizer",
        init_kwargs={"_commit_hash": "tokenizer-commit"},
        eos_token_id=2, pad_token_id=None, model_max_length=4096,
    )
    tokenizer.get_chat_template = lambda: tokenizer.chat_template

    def render(messages, **kwargs):
        state.rendered.append((deepcopy(messages), kwargs))
        return "rendered prompt"

    def tokenize(prompt, **kwargs):
        state.tokenized.append((prompt, kwargs))
        state.inputs = {"input_ids": Tensor([10, 11, 12]), "attention_mask": Tensor([1, 1, 1])}
        return state.inputs

    def decode(tokens, **kwargs):
        state.decoded.append((tokens, kwargs))
        return state.output

    # SimpleNamespace is not callable; use a tiny callable proxy for tokenization.
    class Tokenizer:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def __call__(self, *args, **kwargs):
            return tokenize(*args, **kwargs)

    tokenizer.apply_chat_template = render
    tokenizer.decode = decode
    state.tokenizer = tokenizer
    state.output = '{"type":"final","answer":0}'
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=2048, is_encoder_decoder=False,
                               _commit_hash="model-commit", _name_or_path="fake/model"),
        generation_config=SimpleNamespace(eos_token_id=[2, 3], pad_token_id=None),
        dtype="torch.float32", eval_count=0,
    )

    def model_to(device):
        state.model_device = device
        return model

    def model_eval():
        model.eval_count += 1

    def generate(**kwargs):
        state.generated.append(kwargs)
        return Tensor([10, 11, 12, 90, 91])

    model.to, model.eval, model.generate = model_to, model_eval, generate
    state.model = model

    def load(kind, value, name, **kwargs):
        state.loads.append((kind, name, kwargs))
        return value

    class GenerationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return self.kwargs.copy()

    torch = ModuleType("torch")
    torch.__version__, torch.float32 = "fake-torch", "torch.float32"
    torch.version = SimpleNamespace(cuda="fake-cuda")

    def inference():
        state.inference += 1
        return nullcontext()

    torch.inference_mode = inference
    transformers = ModuleType("transformers")
    transformers.__version__ = "fake-transformers"
    transformers.GenerationConfig = GenerationConfig
    transformers.set_seed = state.seeds.append
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name, **kw: load("tokenizer", Tokenizer(), name, **kw))
    transformers.AutoConfig = SimpleNamespace(from_pretrained=lambda *args, **kwargs: model.config)
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda name, **kw: load("model", model, name, **kw))
    hub = ModuleType("transformers.utils.hub")
    hub.cached_file = lambda *args, **kwargs: "fixture/tokenizer_config.json"
    hub.extract_commit_hash = lambda *args: tokenizer.init_kwargs["_commit_hash"]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", hub)
    return state


def make_backend(**kwargs):
    return HuggingFaceBackend(HuggingFaceConfig("fake/model", **kwargs))


def test_greedy_loading_tokens_and_raw_continuation(fake_hf):
    backend = make_backend(revision="pinned", device="cuda:0", local_files_only=True)
    fake_hf.output = ' \n```json\n{"type":"final","answer":2}\n``` '
    for _ in range(2):
        assert backend.generate([{"role": "user", "content": "question"}]) == fake_hf.output
    assert len(fake_hf.loads) == 2  # one tokenizer and one model across calls
    assert fake_hf.loads[1][2] == {
        "revision": "pinned", "local_files_only": True, "trust_remote_code": False,
        "dtype": "torch.float32", "use_safetensors": True,
        "config": fake_hf.model.config,
    }
    assert fake_hf.model.eval_count == 1 and fake_hf.inference == 2
    assert fake_hf.model_device == "cuda:0"
    arguments = fake_hf.generated[0]
    assert arguments["do_sample"] is False and arguments["num_beams"] == 1
    assert "temperature" not in arguments and "top_p" not in arguments
    assert "temperature" not in arguments["generation_config"].to_dict()
    assert arguments["eos_token_id"] == [2, 3] and arguments["pad_token_id"] == 2
    assert fake_hf.inputs["input_ids"].device == "cuda:0"
    assert fake_hf.inputs["attention_mask"].device == "cuda:0"
    assert fake_hf.tokenized[0][1] == {"return_tensors": "pt", "add_special_tokens": False}
    assert fake_hf.decoded[0] == ([90, 91], {"skip_special_tokens": True, "clean_up_tokenization_spaces": False})


def test_sampling_arguments(fake_hf):
    make_backend(do_sample=True, temperature=0.7, top_p=0.9).generate([{"role": "user", "content": "q"}])
    assert fake_hf.generated[0]["temperature"] == 0.7
    assert fake_hf.generated[0]["top_p"] == 0.9


@pytest.mark.parametrize("model_pad,tokenizer_pad,expected", [(9, 8, 9), (None, 8, 8), (None, None, 2)])
def test_pad_precedence(fake_hf, model_pad, tokenizer_pad, expected):
    fake_hf.model.generation_config.pad_token_id = model_pad
    fake_hf.tokenizer.pad_token_id = tokenizer_pad
    assert make_backend().runtime_config()["generation_arguments"]["pad_token_id"] == expected


def test_tokenizer_eos_and_missing_tokens(fake_hf):
    fake_hf.model.generation_config.eos_token_id = None
    assert make_backend().runtime_config()["generation_arguments"]["eos_token_id"] == 2
    fake_hf.tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="pad token"):
        make_backend()


@pytest.mark.parametrize("eos,pad", [([], None), ([-1], None), ([True], None), (2, -1), (2, True)])
def test_invalid_special_token_ids(fake_hf, eos, pad):
    fake_hf.model.generation_config.eos_token_id = eos
    fake_hf.model.generation_config.pad_token_id = pad
    with pytest.raises(ValueError, match="token_id"):
        make_backend()


def test_template_required_and_explicit_debug_fallback(fake_hf):
    fake_hf.tokenizer.chat_template = None
    with pytest.raises(ValueError, match="no chat template"):
        make_backend()
    assert [row[0] for row in fake_hf.loads] == ["tokenizer"]
    backend = make_backend(allow_fallback_template=True)
    assert backend.chat_template_strategy == "debug_plaintext_fallback"
    backend.generate([{"role": "user", "content": "q"}])
    assert fake_hf.tokenized[0][0] == "user: q\nassistant:"
    assert not fake_hf.rendered


def test_template_render_error_never_falls_back(fake_hf):
    def fail(*args, **kwargs):
        raise ValueError("template failed")

    fake_hf.tokenizer.apply_chat_template = fail
    with pytest.raises(ValueError, match="template failed"):
        make_backend(allow_fallback_template=True).generate([{"role": "user", "content": "q"}])
    assert not fake_hf.tokenized and not fake_hf.generated


def test_template_resolution_error_never_falls_back(fake_hf):
    def fail():
        raise ValueError("no default template")

    fake_hf.tokenizer.get_chat_template = fail
    with pytest.raises(ValueError, match="no default template"):
        make_backend(allow_fallback_template=True)


def test_empty_resolved_template_rejected(fake_hf):
    fake_hf.tokenizer.get_chat_template = lambda: ""
    with pytest.raises(ValueError, match="non-empty string"):
        make_backend(allow_fallback_template=True)


def test_message_copy_and_tool_observation(fake_hf):
    messages = [
        {"role": "system", "content": "rules"}, {"role": "user", "content": "q"},
        {"role": "assistant", "content": "action"},
        {"role": "tool", "name": "lookup_company", "content": '{"output":100}'},
    ]
    original = deepcopy(messages)
    make_backend().generate(messages)
    assert messages == original
    rendered, kwargs = fake_hf.rendered[0]
    assert rendered[:3] == messages[:3]
    assert rendered[3] == {"role": "user", "content": 'Tool observation (lookup_company): {"output":100}'}
    assert kwargs == {"chat_template": "own-template", "tokenize": False, "add_generation_prompt": True}
    assert "tools" not in kwargs


@pytest.mark.parametrize("message", [{"role":"unknown","content":"q"}, {"role":"user","content":1}, {"role":"tool","content":"q"}])
def test_invalid_messages(fake_hf, message):
    with pytest.raises(ValueError):
        make_backend().generate([message])


def test_context_budget_is_checked_without_truncation(fake_hf):
    fake_hf.model.config.max_position_embeddings = 10
    with pytest.raises(ValueError, match="context"):
        make_backend(max_new_tokens=8).generate([{"role": "user", "content": "q"}])
    assert not fake_hf.generated


def test_audit_config(fake_hf):
    config = make_backend(revision="requested").runtime_config()
    assert config["requested_model_revision"] == "requested"
    assert config["resolved_model_revision"] == "model-commit"
    assert config["resolved_tokenizer_revision"] == "tokenizer-commit"
    assert config["tokenizer_name"] == "fake/tokenizer"
    assert config["transformers_version"] == "fake-transformers"
    assert config["torch_version"] == "fake-torch"
    assert config["cuda_version"] == "fake-cuda" and config["dtype"] == "torch.float32"
    assert config["chat_template_strategy"] == "model_chat_template"
    assert len(config["chat_template_sha256"]) == 64
    json.dumps(config, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens":0}, {"max_new_tokens":True}, {"temperature":0}, {"temperature":float("nan")},
    {"top_p":1.1}, {"top_p":False}, {"device":""}, {"revision":""}, {"do_sample":1},
    {"local_files_only":1}, {"allow_fallback_template":1},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        HuggingFaceConfig("fake/model", **kwargs)


def test_empty_model_name():
    with pytest.raises(ValueError):
        HuggingFaceConfig("")


def test_missing_optional_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="requirements-hf"):
        make_backend()


def test_decoder_only_required(fake_hf):
    fake_hf.model.config.is_encoder_decoder = True
    with pytest.raises(ValueError, match="decoder-only"):
        make_backend()


def test_unmodified_parser_counts_and_no_task_leakage(fake_hf):
    fake_hf.output = '```json\n{"type":"final","answer":0}\n```'
    tasks = TaskGenerator(42).generate_tasks(1)
    backend = make_backend()
    report = evaluate_tasks(tasks, lambda _: PromptAgent(backend, max_steps=1), {environment_id_for_seed(42):42})
    assert len(report.records) == 4
    assert all(record.run.events[0].parse_error for record in report.records)
    assert all(not record.verification.success for record in report.records)
    rendered = json.dumps(fake_hf.rendered)
    for forbidden in ("ground_truth", "verification_spec", "metadata", "task_id", "environment_id"):
        assert forbidden not in rendered
    assert len(fake_hf.loads) == 2
    # Each new task starts with only system + user, never another task's history.
    assert all(len(messages) == 2 for messages, _ in fake_hf.rendered)
    assert report.metrics["counts"]["parse_error_count"] == 4


def test_template_backend_errors_are_counted(fake_hf):
    def fail(*args, **kwargs):
        raise ValueError("unsupported chat history")

    fake_hf.tokenizer.apply_chat_template = fail
    backend = make_backend(allow_fallback_template=True)
    tasks = TaskGenerator(42).generate_tasks(1)
    report = evaluate_tasks(tasks, lambda _: PromptAgent(backend, max_steps=2), {environment_id_for_seed(42):42})
    assert report.metrics["counts"]["backend_error_count"] == 4
    assert report.metrics["counts"]["parse_error_count"] == 0
    assert all(record.run.termination_reason == "backend_error" for record in report.records)
    assert not fake_hf.tokenized


def test_cli_hf_save_and_no_overwrite(fake_hf, tmp_path, capsys):
    directory = tmp_path / "run"
    arguments = ["--backend","hf","--model","fake/model","--output-dir",str(directory),
                 "--count-per-level","1","--max-steps","1","--revision","pinned"]
    assert main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    config = json.loads((directory / "run_config.json").read_text())
    rows = [json.loads(line) for line in (directory / "trajectories.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert config["mode"] == "hf_benchmark" and config["seed"] == 42
    assert config["task_ids"] == [row["task_id"] for row in rows]
    assert "git_commit_hash" in config and "git_dirty" in config
    assert config["requested_model_revision"] == "pinned" and fake_hf.seeds == [42]
    assert json.loads((directory / "metrics.json").read_text()) == output["metrics"]
    before = (directory / "run_config.json").read_bytes()
    with pytest.raises(SystemExit):
        main(arguments)
    assert (directory / "run_config.json").read_bytes() == before
    assert len(fake_hf.loads) == 2  # overwrite rejected before loading again


def test_cli_labels_fallback_debug(fake_hf, tmp_path, capsys):
    fake_hf.tokenizer.chat_template = None
    main(["--backend","hf","--model","fake/model","--output-dir",str(tmp_path / "debug"),
          "--allow-fallback-template","--count-per-level","1","--max-steps","1"])
    assert json.loads(capsys.readouterr().out)["mode"] == "hf_debug_fallback"


@pytest.mark.parametrize("args", [[], ["--backend","hf"], ["--backend","hf","--model","fake/model"],
                                  ["--backend","scripted","--model","fake/model"]])
def test_cli_required_arguments(fake_hf, args):
    with pytest.raises(SystemExit):
        main(args)
    assert not fake_hf.loads


def prepare_adapter(fake_hf, tmp_path, monkeypatch):
    fake_hf.tokenizer.is_fast = True
    fake_hf.tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda:'{"fixture":true}')
    fake_hf.tokenizer.special_tokens_map = {"eos_token":"fixture-eos"}
    fake_hf.tokenizer.init_kwargs["_commit_hash"] = "model-commit"
    from sft.preprocessing import tokenizer_fingerprint
    import hashlib

    identity = {"base_model_name_or_path":"fake/model", "base_model_revision":"model-commit",
                "tokenizer_identity":"fake/model", "tokenizer_revision":"model-commit",
                "chat_template_sha256":hashlib.sha256(b"own-template").hexdigest(),
                "tokenizer_sha256":tokenizer_fingerprint(fake_hf.tokenizer)}
    lora = lora_configuration(TrainingConfig("data", "out"))
    directory = tmp_path / "training"
    directory.mkdir()
    record = {"schema_version":"lora-sft-run-v1", "identity":identity, "lora":lora,
              "dataset":{"manifest_sha256":"fixture"}}
    (directory / "run_config.json").write_text(json.dumps(record))
    peft_config = SimpleNamespace(base_model_name_or_path="fake/model", revision="model-commit", peft_type="LORA", **lora)
    fake_hf.adapter_loads = []

    def load(model, path, **kw):
        fake_hf.adapter_loads.append((model,path,kw))
        return model

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftConfig=SimpleNamespace(from_pretrained=lambda *a, **k:peft_config),
        PeftModel=SimpleNamespace(from_pretrained=load)))
    adapter = directory / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fake weights; no model download")
    return adapter, peft_config


def prepare_grpo_adapter(fake_hf,tmp_path,monkeypatch):
    from grpo.policy import adapter_hash
    adapter,_ = prepare_adapter(fake_hf,tmp_path,monkeypatch)
    path = adapter.parent/"run_config.json"
    record = json.loads(path.read_text())
    record.update(schema_version="lora-grpo-run-v1",status="complete",
                  initial_sft_adapter_sha256="a"*64,adapter_sha256=adapter_hash(adapter))
    path.write_text(json.dumps(record))
    reloads = []
    monkeypatch.setitem(sys.modules,"peft.utils.save_and_load",SimpleNamespace(
        load_peft_weights=lambda *a,**k:{"fake":"weights"},
        set_peft_model_state_dict=lambda *a,**k:reloads.append((a,k))))
    return adapter,record,reloads


def test_grpo_adapter_validated_and_exact_weights_reloaded(fake_hf,tmp_path,monkeypatch):
    adapter,record,reloads = prepare_grpo_adapter(fake_hf,tmp_path,monkeypatch)
    backend = make_backend(adapter_path=str(adapter),revision="model-commit")
    runtime = backend.runtime_config()
    assert runtime["adapter_schema"] == "lora-grpo-run-v1"
    assert runtime["initial_sft_adapter_sha256"] == "a"*64
    assert len(reloads) == 1 and reloads[0][1] == {"adapter_name":"default"}


@pytest.mark.parametrize("mutation",["status","adapter_hash","source_hash","revision","template"])
def test_grpo_adapter_tampering_rejected(fake_hf,tmp_path,monkeypatch,mutation):
    adapter,record,_ = prepare_grpo_adapter(fake_hf,tmp_path,monkeypatch)
    if mutation == "status": record["status"] = "incomplete"
    elif mutation == "adapter_hash": record["adapter_sha256"] = "0"*64
    elif mutation == "source_hash": record["initial_sft_adapter_sha256"] = "invalid"
    elif mutation == "revision": record["identity"]["base_model_revision"] = "wrong"
    else: record["identity"]["chat_template_sha256"] = "wrong"
    (adapter.parent/"run_config.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        make_backend(adapter_path=str(adapter),revision="model-commit")


def test_sft_grpo_comparison_requires_audited_start_and_equal_identity():
    common = {"mode":"hf_benchmark","adapter_identity_validated":True,"adapter_path":"adapter",
        "model_identity":{"base_model_name_or_path":"model","base_model_revision":"r",
            "tokenizer_identity":"model","tokenizer_revision":"r","chat_template_sha256":"t","tokenizer_sha256":"v"},
        "task_ids":["task"],"environment_seeds":{"env":1},"max_steps":16,"generation_seed":42,
        "generation_arguments":{"do_sample":False},"dtype":"bf16","chat_template_strategy":"model_chat_template",
        "tool_message_strategy":"user_observation_with_tool_name","transformers_version":"4","torch_version":"2",
        "requested_backend_config":{"device":"cuda:0"},"benchmark_coordinates":[{"seed":1}]}
    sft = dict(common,adapter_schema="lora-sft-run-v1",adapter_sha256="a"*64)
    rl = dict(common,adapter_schema="lora-grpo-run-v1",initial_sft_adapter_sha256="a"*64)
    validate_comparison(sft,rl,"sft-vs-grpo")
    for key,value in (("initial_sft_adapter_sha256","wrong"),("task_ids",["test-task"]),("generation_arguments",{"do_sample":True})):
        with pytest.raises(ValueError): validate_comparison(sft,dict(rl,**{key:value}),"sft-vs-grpo")


def test_adapter_loading_is_frozen_and_provenance_checked(fake_hf, tmp_path, monkeypatch):
    adapter, _ = prepare_adapter(fake_hf, tmp_path, monkeypatch)
    backend = make_backend(adapter_path=str(adapter), revision="model-commit", tokenizer_path=str(adapter.parent / "tokenizer"))
    assert backend.runtime_config()["adapter_identity_validated"] is True
    assert fake_hf.adapter_loads[0][2] == {"is_trainable":False,"local_files_only":True}
    assert backend.generate([{"role":"user","content":"normal task"}]) == fake_hf.output
    assert "temperature" not in fake_hf.generated[0]


@pytest.mark.parametrize("mutation", ["requested_revision", "model_revision", "tokenizer_revision", "template", "tokenizer", "lora"])
def test_adapter_mismatches_are_rejected(fake_hf, tmp_path, monkeypatch, mutation):
    adapter, peft_config = prepare_adapter(fake_hf, tmp_path, monkeypatch)
    revision = "model-commit"
    if mutation == "requested_revision":
        revision = "main"
    elif mutation == "model_revision":
        fake_hf.model.config._commit_hash = "wrong"
    elif mutation == "tokenizer_revision":
        fake_hf.tokenizer.init_kwargs["_commit_hash"] = "wrong"
    elif mutation == "template":
        fake_hf.tokenizer.chat_template = "different-template"
    elif mutation == "tokenizer":
        fake_hf.tokenizer.special_tokens_map = {}
    else:
        peft_config.r = 64
    with pytest.raises(ValueError):
        make_backend(adapter_path=str(adapter), revision=revision)
    assert not fake_hf.adapter_loads


@pytest.mark.parametrize("kwargs", [{"dtype":"int8"}, {"adapter_path":""}, {"tokenizer_path":"tok"},
                                    {"adapter_path":"adapter","allow_fallback_template":True}])
def test_invalid_adapter_configuration(kwargs):
    with pytest.raises(ValueError):
        HuggingFaceConfig("fake/model", **kwargs)


def test_manifest_benchmark_reads_coordinates_only(fake_hf, tmp_path, monkeypatch):
    directory = tmp_path / "dataset"
    export_dataset(build_dataset(DatasetConfig(train_size=16,dev_size=16,test_size=16)), directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["test"]["coordinates"][0]["messages"] = "FORBIDDEN_ORACLE_OUTPUT"
    manifest_path.write_text(json.dumps(manifest))
    # Delete every sample/trajectory export. Coordinates alone are sufficient.
    for path in directory.glob("*.jsonl"):
        path.unlink()
    original = Path.read_bytes

    def guarded(path):
        assert not path.name.endswith(".jsonl")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    tasks, seeds, configuration = load_manifest_benchmark(manifest_path)
    assert len(tasks) == 16 and list(seeds.values()) == [30000]
    assert all(set(c) == {"seed","difficulty","task_index","task_id"} for c in configuration["benchmark_coordinates"])
    report = evaluate_tasks(tasks, lambda _:PromptAgent(make_backend(), max_steps=1), seeds)
    assert len(report.records) == 16
    rendered = json.dumps(fake_hf.rendered)
    assert "FORBIDDEN_ORACLE_OUTPUT" not in rendered
    for forbidden in ("ground_truth", "verification_spec", "metadata", "coordinates"):
        assert forbidden not in rendered
    assert all(len(messages) == 2 for messages, _ in fake_hf.rendered)


def test_manifest_identity_mismatch_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version":"sft-manifest-v1", "splits":{"test":{"coordinates":[
        {"seed":42,"difficulty":1,"task_index":0,"task_id":"wrong"}]}}}))
    with pytest.raises(ValueError, match="identity"):
        load_manifest_benchmark(path)


def test_cli_base_adapter_comparison_and_real_parse_errors(fake_hf, tmp_path, monkeypatch, capsys):
    adapter, _ = prepare_adapter(fake_hf, tmp_path, monkeypatch)
    fake_hf.output = '```json\n{"type":"final","answer":0}\n```'
    base, sft = tmp_path / "base", tmp_path / "sft"
    common = ["--backend","hf","--model","fake/model","--revision","model-commit", "--count-per-level","1","--max-steps","1"]
    assert main(common + ["--output-dir",str(base)]) == 0
    capsys.readouterr()
    assert main(common + ["--output-dir",str(sft),"--adapter-path",str(adapter),"--compare-to",str(base)]) == 0
    capsys.readouterr()
    result = json.loads((sft / "comparison.json").read_text())
    assert result["base"]["counts"]["parse_error_count"] == 4
    assert result["sft"]["counts"]["parse_error_count"] == 4
    assert result["parse_error_count_delta"] == 0
    base_config = json.loads((base / "run_config.json").read_text())
    sft_config = json.loads((sft / "run_config.json").read_text())
    sft_config["model_identity"]["chat_template_sha256"] = "wrong"
    with pytest.raises(ValueError, match="chat_template_sha256"):
        validate_comparison(base_config,sft_config)


def test_cli_manifest_benchmark(fake_hf, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    task = TaskGenerator(30000).generate_task(2,1)
    manifest.write_text(json.dumps({"schema_version":"sft-manifest-v1", "splits":{"test":{"coordinates":[
        {"seed":30000,"difficulty":2,"task_index":1,"task_id":task.task_id}]}}}))
    directory = tmp_path / "eval"
    assert main(["--backend","hf","--model","fake/model","--benchmark-manifest",str(manifest),
                 "--output-dir",str(directory),"--max-steps","1"]) == 0
    capsys.readouterr()
    record = json.loads((directory / "run_config.json").read_text())
    assert record["generation_seed"] == 42 and record["seed"] is None
    assert record["environment_seeds"] == {task.environment_id:30000}
    with pytest.raises(SystemExit):
        main(["--benchmark-manifest",str(manifest),"--seed","42"])
