"""All HF tests use fake packages/tensors: no weights or optional dependencies."""

from contextlib import nullcontext
from copy import deepcopy
import json
from types import ModuleType, SimpleNamespace
import sys

import pytest

from agent.agent import PromptAgent
from agent.hf_backend import HuggingFaceBackend, HuggingFaceConfig
from scripts.evaluate import main, evaluate_tasks
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
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda name, **kw: load("model", model, name, **kw))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
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
