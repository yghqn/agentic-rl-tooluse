"""Optional, single-device Hugging Face text backend (no execution or Task access)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import json
from pathlib import Path
from typing import Any

from agent.prompts import Message


@dataclass(frozen=True, slots=True)
class HuggingFaceConfig:
    model_name_or_path: str
    revision: str | None = None
    device: str = "cpu"
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    local_files_only: bool = False
    allow_fallback_template: bool = False
    cache_dir: str | None = None
    dtype: str = "float32"
    adapter_path: str | None = None
    tokenizer_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name_or_path, str) or not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path must be non-empty")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("revision must be non-empty when supplied")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty")
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        for name in ("do_sample", "local_files_only", "allow_fallback_template"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.top_p > 1:
            raise ValueError("top_p must be <= 1")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Unsupported dtype")
        for name in ("cache_dir", "adapter_path", "tokenizer_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty when supplied")
        if self.tokenizer_path and not self.adapter_path:
            raise ValueError("tokenizer_path requires an adapter")
        if self.adapter_path and self.allow_fallback_template:
            raise ValueError("Adapter evaluation cannot use a fallback template")


class HuggingFaceBackend:
    """Loads once; every generate call derives all conversation state from messages.

    Missing templates may use an explicitly enabled DEBUG transcript. Template
    rendering errors always propagate, even when debugging fallback is enabled.
    """

    def __init__(self, config: HuggingFaceConfig) -> None:
        # Scripted evaluation and unit tests do not require these optional packages.
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError("HF dependencies missing; install requirements-hf.txt") from exc
        self.config = config
        self._torch = torch
        self._transformers = transformers
        loading = {
            "revision": config.revision,
            "local_files_only": config.local_files_only,
            "trust_remote_code": False,
        }
        if config.cache_dir is not None:
            loading["cache_dir"] = config.cache_dir
        self._adapter_record = None
        if config.adapter_path:
            record = json.loads((Path(config.adapter_path).parent / "run_config.json").read_text(encoding="utf-8"))
            if record.get("schema_version") != "lora-sft-run-v1":
                raise ValueError("Adapter requires training run provenance")
            self._adapter_record = record
            identity = record["identity"]
            if (config.model_name_or_path != identity["base_model_name_or_path"]
                or config.revision != identity["base_model_revision"]):
                raise ValueError("Adapter requires its exact training base model ID and pinned revision")
        # Resolve tokenizer provenance and load its exact local snapshot. Some
        # Transformers versions probe remote model metadata even with local_files_only.
        from sft.training import tokenizer_source

        tokenizer_loading = {k:v for k,v in loading.items() if k != "revision"}
        if config.tokenizer_path:
            source, self._tokenizer_revision = config.tokenizer_path, config.revision
        else:
            source, self._tokenizer_revision = tokenizer_source(config.model_name_or_path, config.revision, tokenizer_loading)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(source, **tokenizer_loading)
        # Resolve the default template explicitly (including tokenizers with a
        # template dictionary); do not silently choose a tool-specific template.
        if getattr(self._tokenizer, "chat_template", None):
            self._template = self._tokenizer.get_chat_template()
            if not isinstance(self._template, str) or not self._template.strip():
                raise ValueError("Model chat template must resolve to a non-empty string")
            self.chat_template_strategy = "model_chat_template"
        elif config.allow_fallback_template:
            self._template = None
            self.chat_template_strategy = "debug_plaintext_fallback"
        else:
            raise ValueError("Model tokenizer has no chat template; debug fallback requires explicit opt-in")
        from sft.training import load_causal_model

        self._model = load_causal_model(config.model_name_or_path, loading, getattr(torch, config.dtype))
        if config.adapter_path:
            from peft import PeftConfig, PeftModel
            from sft.training import validate_adapter_identity

            validate_adapter_identity(self._adapter_record["identity"], self.model_identity())
            adapter_config = PeftConfig.from_pretrained(config.adapter_path, local_files_only=True)
            if (adapter_config.base_model_name_or_path != config.model_name_or_path
                or adapter_config.revision != config.revision or adapter_config.peft_type != "LORA"
                or adapter_config.task_type != "CAUSAL_LM"):
                raise ValueError("PEFT adapter configuration differs from training provenance")
            expected_lora = self._adapter_record["lora"]
            for key in ("r", "lora_alpha", "lora_dropout", "bias", "modules_to_save", "use_rslora"):
                if getattr(adapter_config, key, None) != expected_lora[key]:
                    raise ValueError(f"PEFT adapter configuration mismatch: {key}")
            if set(adapter_config.target_modules) != set(expected_lora["target_modules"]):
                raise ValueError("PEFT adapter target modules mismatch")
            self._model = PeftModel.from_pretrained(self._model, config.adapter_path, is_trainable=False,
                                                     local_files_only=True)
        self._model.to(config.device)
        self._model.eval()
        if getattr(self._model.config, "is_encoder_decoder", False):
            raise ValueError("HuggingFaceBackend requires a decoder-only causal language model")
        generation = self._model.generation_config
        eos = getattr(generation, "eos_token_id", None)
        if eos is None:
            eos = self._tokenizer.eos_token_id
        if eos is not None:
            ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("eos_token_id must be a non-negative integer or non-empty list of integers")
            eos = ids if isinstance(eos, (list, tuple)) else eos
        pad = getattr(generation, "pad_token_id", None)
        if pad is None:
            pad = self._tokenizer.pad_token_id
        if pad is None and eos is not None:
            pad = eos[0] if isinstance(eos, (list, tuple)) else eos
        if pad is None:
            raise ValueError("Neither a pad token nor an EOS token is configured")
        if type(pad) is not int or pad < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        self._generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "num_beams": 1,
            "num_return_sequences": 1,
            "return_dict_in_generate": False,
            "eos_token_id": eos,
            "pad_token_id": pad,
        }
        if config.do_sample:
            self._generation_kwargs.update(temperature=config.temperature, top_p=config.top_p)
        # Reset pretrained sampling/beam settings instead of inheriting hidden
        # model-specific repetition penalties or stopping options in a baseline.
        self._generation_config = transformers.GenerationConfig(**self._generation_kwargs)

    @staticmethod
    def _copy_messages(messages: list[Message]) -> list[Message]:
        converted: list[Message] = []
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("Each message must contain string content")
            role = message.get("role")
            if role == "tool":
                name = message.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Tool messages require a tool name")
                converted.append({"role": "user", "content": f"Tool observation ({name}): {message['content']}"})
            elif role in ("system", "user", "assistant"):
                converted.append({"role": role, "content": message["content"]})
            else:
                raise ValueError("Unsupported message role")
        return converted

    def generate(self, messages: list[Message]) -> str:
        converted = self._copy_messages(messages)
        if self._template is not None:
            prompt = self._tokenizer.apply_chat_template(
                converted, chat_template=self._template,
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in converted) + "\nassistant:"
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_length = inputs["input_ids"].shape[-1]
        limits = [
            getattr(self._model.config, "max_position_embeddings", None),
            getattr(self._tokenizer, "model_max_length", None),
        ]
        known_limits = [limit for limit in limits if type(limit) is int and 0 < limit < 10**9]
        if known_limits and input_length + self.config.max_new_tokens > min(known_limits):
            raise ValueError("Prompt plus generation budget exceeds model context; no silent truncation")
        with self._torch.inference_mode():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"].to(self.config.device),
                attention_mask=inputs["attention_mask"].to(self.config.device),
                generation_config=self._generation_config,
                **self._generation_kwargs,
            )
        continuation = outputs[0, input_length:]
        # Raw continuation only: no JSON repair, whitespace stripping or fences removal.
        return self._tokenizer.decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )

    def runtime_config(self) -> dict[str, Any]:
        """Best-effort audit information, never claimed resolved when unavailable."""
        return {
            "requested_model_revision": self.config.revision,
            "resolved_model_revision": getattr(self._model.config, "_commit_hash", None),
            "resolved_tokenizer_revision": self._tokenizer_revision,
            "model_name": getattr(self._model.config, "_name_or_path", self.config.model_name_or_path),
            "tokenizer_name": self._tokenizer.name_or_path,
            "transformers_version": self._transformers.__version__,
            "torch_version": self._torch.__version__,
            "cuda_version": self._torch.version.cuda,
            "dtype": str(self._model.dtype),
            "generation_config": self._generation_config.to_dict(),
            "generation_arguments": dict(self._generation_kwargs),
            "chat_template_strategy": self.chat_template_strategy,
            "chat_template_sha256": hashlib.sha256(self._template.encode()).hexdigest() if self._template else None,
            "tool_message_strategy": "user_observation_with_tool_name",
            "model_identity": self.model_identity(),
            "adapter_path": self.config.adapter_path,
            "adapter_identity_validated": self._adapter_record is not None,
            "adapter_training_dataset": self._adapter_record.get("dataset") if self._adapter_record else None,
        }

    def model_identity(self) -> dict[str, Any]:
        from sft.preprocessing import tokenizer_fingerprint

        return {"base_model_name_or_path": self.config.model_name_or_path,
                "base_model_revision": getattr(self._model.config, "_commit_hash", None),
                "tokenizer_identity": self.config.model_name_or_path,
                "tokenizer_revision": self._tokenizer_revision,
                "chat_template_sha256": hashlib.sha256(self._template.encode()).hexdigest() if self._template else None,
                "tokenizer_sha256": tokenizer_fingerprint(self._tokenizer) if getattr(self._tokenizer, "is_fast", False) else None}
