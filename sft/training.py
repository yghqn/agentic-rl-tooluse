"""Optional PEFT training behind a persisted, reproducible preprocessing gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from sft.preprocessing import (
    AssistantOnlyCollator, PreprocessingResult, fingerprint, preprocess_dataset,
    require_preprocessing_gate, tokenizer_fingerprint,
)


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
IDENTITY_KEYS = ("base_model_name_or_path", "base_model_revision", "tokenizer_identity",
                 "tokenizer_revision", "chat_template_sha256", "tokenizer_sha256")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    dataset_dir: str
    output_dir: str
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    cache_dir: str | None = None
    local_files_only: bool = False
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    seed: int = 42
    max_seq_length: int = 2048
    epochs: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    def __post_init__(self) -> None:
        for key in ("dataset_dir", "output_dir", "model_name_or_path", "revision", "device"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be non-empty")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Unsupported dtype")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for key in ("max_seq_length", "batch_size", "gradient_accumulation_steps", "lora_rank", "lora_alpha"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("epochs", "learning_rate"):
            value = getattr(self, key)
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if type(self.lora_dropout) not in {int, float} or not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0,1)")
        if Path(self.dataset_dir).resolve() == Path(self.output_dir).resolve():
            raise ValueError("Training output must not overwrite the dataset")


def tokenizer_source(model: str, revision: str | None, loading: dict[str, Any]) -> tuple[str, str | None]:
    """Load tokenizers from the exact cached snapshot (also avoids remote probes).

    This does not change the model-owned template or tokenizer implementation.
    For a local model we cannot assert an upstream revision without provenance.
    """
    if Path(model).is_dir():
        return model, None
    from transformers.utils.hub import cached_file, extract_commit_hash

    path = cached_file(model, "tokenizer_config.json", revision=revision, **loading)
    return str(Path(path).parent), extract_commit_hash(path, None)


def load_training_tokenizer(config: TrainingConfig) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoConfig, AutoTokenizer

    loading = {"cache_dir": config.cache_dir, "local_files_only": config.local_files_only,
               "trust_remote_code": False}
    model_config = AutoConfig.from_pretrained(config.model_name_or_path, revision=config.revision, **loading)
    resolved = getattr(model_config, "_commit_hash", None)
    if not isinstance(resolved, str) or not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("Training requires a resolved 40-character HF model commit; use an HF ID and pinned revision")
    if getattr(model_config, "model_type", None) != "qwen2":
        raise ValueError("This stage supports Qwen2/Qwen2.5 causal models only")
    if config.max_seq_length > model_config.max_position_embeddings:
        raise ValueError("max_seq_length exceeds the model context")
    source, tokenizer_revision = tokenizer_source(config.model_name_or_path, resolved, loading)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **loading)
    if tokenizer_revision != resolved:
        raise ValueError("Tokenizer snapshot does not match the base model revision")
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Model-owned chat template required")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Explicit tokenizer EOS and PAD IDs are required")
    identity = {"base_model_name_or_path": config.model_name_or_path, "base_model_revision": resolved,
                "tokenizer_identity": config.model_name_or_path, "tokenizer_revision": tokenizer_revision,
                "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "tokenizer_sha256": tokenizer_fingerprint(tokenizer)}
    return tokenizer, identity


def write_json_once(path: Path, value: Any) -> None:
    """Permit identical audit re-runs, never silently overwrite a different run."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Different artifact already exists: {path}; choose a new output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run_preprocessing(config: TrainingConfig, tokenizer: Any, identity: dict[str, Any]) -> PreprocessingResult:
    result = preprocess_dataset(Path(config.dataset_dir), tokenizer, config.max_seq_length, identity)
    output = Path(config.output_dir)
    write_json_once(output / "preprocessing_audit.json", result.audit)
    write_json_once(output / "length_stats.json", {
        split: {key: value for key, value in stats.items() if key not in {"sample_spans", "errors"}}
        for split, stats in result.audit["splits"].items()
    })
    require_preprocessing_gate(result)
    return result


def require_persisted_gate(config: TrainingConfig, result: PreprocessingResult, identity: dict[str, Any]) -> None:
    require_preprocessing_gate(result)
    if result.audit["identity"] != identity or result.audit["max_seq_length"] != config.max_seq_length:
        raise ValueError("Preprocessing identity/length mismatch")
    path = Path(config.output_dir) / "preprocessing_audit.json"
    if not path.is_file():
        raise ValueError("Run --preprocess-only successfully before loading a training model or starting Trainer")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded != result.audit or recorded.get("passed") is not True:
        raise ValueError("Persisted preprocessing audit is failed or stale; run --preprocess-only in a new output directory")


def lora_configuration(config: TrainingConfig) -> dict[str, Any]:
    return {"task_type": "CAUSAL_LM", "r": config.lora_rank, "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout, "target_modules": list(TARGET_MODULES),
            "bias": "none", "modules_to_save": None, "use_rslora": False}


def parameter_counts(model: Any) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise ValueError("LoRA has no trainable parameters")
    if any(p.requires_grad and ".lora_A." not in name and ".lora_B." not in name
           for name, p in model.named_parameters()):
        raise ValueError("Non-LoRA parameters unexpectedly trainable")
    return {"trainable_parameters": trainable, "total_parameters": total,
            "trainable_parameter_ratio": trainable / total}


def load_causal_model(model_name: str, loading: dict[str, Any], dtype: Any) -> Any:
    """Avoid PEFT auto-detection probing remote adapters during offline loading."""
    from transformers import AutoConfig, AutoModelForCausalLM

    source = model_name
    options = dict(loading)
    if loading["local_files_only"] and not Path(model_name).is_dir():
        from transformers.utils.hub import cached_file

        path = cached_file(model_name, "config.json", **loading)
        source = str(Path(path).parent)
        options["config"] = AutoConfig.from_pretrained(model_name, **loading)
    model = AutoModelForCausalLM.from_pretrained(source, **options, dtype=dtype, use_safetensors=True)
    # PEFT records this identity, not the machine-specific cache folder.
    model.name_or_path = model_name
    model.config._name_or_path = model_name
    return model


def build_lora_model(config: TrainingConfig, identity: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import torch
    from transformers import set_seed
    from peft import LoraConfig, get_peft_model

    device = torch.device(config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Training supports one CPU or CUDA device")
    if device.type == "cpu" and config.dtype != "float32":
        raise ValueError("CPU inspection/training requires float32")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable: install a CUDA-enabled PyTorch build before GPU training")
        torch.cuda.set_device(device)
        if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16")
    set_seed(config.seed)
    model = load_causal_model(config.model_name_or_path,
                             {"revision":identity["base_model_revision"], "cache_dir":config.cache_dir,
                              "local_files_only":config.local_files_only, "trust_remote_code":False},
                             getattr(torch, config.dtype))
    if getattr(model.config, "_commit_hash", None) != identity["base_model_revision"]:
        raise ValueError("Loaded model revision differs from preprocessing")
    leaves = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if not set(TARGET_MODULES) <= leaves:
        raise ValueError("Expected Qwen LoRA target modules are missing")
    lora = LoraConfig(**lora_configuration(config), revision=identity["base_model_revision"])
    model = get_peft_model(model, lora)
    model.to(device)
    return model, parameter_counts(model)


def inspect_model(config: TrainingConfig, result: PreprocessingResult, identity: dict[str, Any]) -> dict[str, Any]:
    """Count actual initialized LoRA parameters. No Trainer or optimizer is created."""
    require_persisted_gate(config, result, identity)
    _, counts = build_lora_model(config, identity)
    record = {"identity": identity, "lora": lora_configuration(config), "dtype": config.dtype,
              "device": config.device, **counts, "trainer_started": False}
    write_json_once(Path(config.output_dir) / "model_inspection.json", record)
    return record


def train(config: TrainingConfig, tokenizer: Any, result: PreprocessingResult, identity: dict[str, Any]) -> dict[str, Any]:
    # This check precedes even optional Trainer imports, model loading and optimizer creation.
    require_persisted_gate(config, result, identity)
    if (tokenizer_fingerprint(tokenizer) != identity["tokenizer_sha256"]
        or hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest() != identity["chat_template_sha256"]):
        raise ValueError("Training tokenizer differs from the preprocessing gate")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This minimal training stage is single-process only")
    output = Path(config.output_dir)
    if any((output / name).exists() for name in ("run_config.json", "adapter", "checkpoints", "loss_history.jsonl")):
        raise ValueError("Training artifacts already exist; choose a new output directory")
    import accelerate
    import peft
    import torch
    import transformers
    from transformers import Trainer, TrainingArguments
    from scripts.evaluate import project_git_state

    model, counts = build_lora_model(config, identity)
    model.config.use_cache = False
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=config.epochs,
        learning_rate=config.learning_rate, per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim="adamw_torch", lr_scheduler_type="linear", warmup_ratio=0.05,
        weight_decay=0.0, max_grad_norm=1.0, seed=config.seed, data_seed=config.seed,
        use_cpu=config.device.startswith("cpu"), bf16=config.dtype == "bfloat16", fp16=config.dtype == "float16",
        tf32=False, full_determinism=True, dataloader_num_workers=0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, logging_steps=1,
        report_to=[], push_to_hub=False, prediction_loss_only=True, label_names=["labels"],
    )
    requested_device = torch.device(config.device)
    if args.device.type != requested_device.type or (
        requested_device.type == "cuda" and args.device.index != (requested_device.index or 0)
    ):
        raise ValueError("Trainer device differs from --device; select one GPU with CUDA_VISIBLE_DEVICES and use cuda:0")
    record = {"schema_version": "lora-sft-run-v1", "training_config": asdict(config), "identity": identity,
              "requested_model_revision": config.revision, "resolved_model_revision": identity["base_model_revision"],
              "dataset": {key: result.audit[key] for key in ("manifest_sha256", "file_sha256", "preprocessing_sha256")},
              "lora": lora_configuration(config), "training_arguments": args.to_dict(),
              "transformers_version": transformers.__version__, "peft_version": peft.__version__,
              "accelerate_version": accelerate.__version__, "torch_version": torch.__version__,
              "cuda_version": torch.version.cuda, "dtype": str(model.dtype),
              "chat_template_strategy": "model_chat_template", **project_git_state(), **counts}
    write_json_once(output / "run_config.json", record)
    trainer = Trainer(model=model, args=args, train_dataset=result.features["train"],
                      eval_dataset=result.features["dev"], data_collator=AssistantOnlyCollator(tokenizer.pad_token_id))
    # Keep the history even if training fails; never attempt JSON repair or benchmark changes.
    try:
        trainer.train()
        trainer.evaluate()
        model.save_pretrained(output / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(output / "tokenizer")
        trainer.save_state()
    finally:
        with (output / "loss_history.jsonl").open("x", encoding="utf-8") as stream:
            for entry in trainer.state.log_history:
                stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
    return record


def validate_adapter_identity(recorded: dict[str, Any], actual: dict[str, Any]) -> None:
    """Exact provenance gate; revision/template mismatches are not comparable."""
    for key in IDENTITY_KEYS:
        if not recorded.get(key) or recorded[key] != actual.get(key):
            raise ValueError(f"Adapter identity mismatch: {key}")
