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
            if record.get("schema_version") not in {"lora-sft-run-v1","lora-grpo-run-v1"}:
                raise ValueError("Adapter requires training run provenance")
            if record["schema_version"] == "lora-grpo-run-v1":
                from grpo.policy import adapter_hash
                source = record.get("initial_sft_adapter_sha256")
                if (record.get("status") != "complete" or not isinstance(source,str)
                    or len(source) != 64 or any(c not in "0123456789abcdef" for c in source)
                    or record.get("adapter_sha256") != adapter_hash(Path(config.adapter_path))):
                    raise ValueError("GRPO adapter checkpoint identity/status mismatch")
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
            if self._adapter_record["schema_version"] == "lora-grpo-run-v1":
                # PEFT may initialize a first adapter in the BF16 base dtype and
                # only then promote LoRA to FP32. Reload into those FP32 tensors
                # so an RL checkpoint is evaluated with its exact trained values.
                from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
                state = load_peft_weights(config.adapter_path,device="cpu",local_files_only=True)
                set_peft_model_state_dict(self._model,state,adapter_name="default")
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

    def _prepare_inputs(self, messages: list[Message]) -> tuple[Any, int, list[Message]]:
        prompt, converted = self._render_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_length = inputs["input_ids"].shape[-1]
        self._validate_context_length(input_length)
        return inputs,input_length,converted

    def _render_prompt(self, messages: list[Message]) -> tuple[str, list[Message]]:
        converted = self._copy_messages(messages)
        if self._template is not None:
            prompt = self._tokenizer.apply_chat_template(
                converted, chat_template=self._template,
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in converted) + "\nassistant:"
        return prompt, converted

    def _validate_context_length(self, input_length: int) -> None:
        limits = [
            getattr(self._model.config, "max_position_embeddings", None),
            getattr(self._tokenizer, "model_max_length", None),
        ]
        known_limits = [limit for limit in limits if type(limit) is int and 0 < limit < 10**9]
        if known_limits and input_length + self.config.max_new_tokens > min(known_limits):
            raise ValueError("Prompt plus generation budget exceeds model context; no silent truncation")

    def _prepare_batch_inputs(self, messages_batch: list[list[Message]]) -> tuple[Any, Any, list[list[int]], list[list[Message]]]:
        """Render independently, then left-pad for decoder-only batched decoding."""
        if not messages_batch:
            raise ValueError("messages_batch must be non-empty")
        prompt_ids: list[list[int]] = []
        converted_batch: list[list[Message]] = []
        for messages in messages_batch:
            prompt, converted = self._render_prompt(messages)
            encoded = self._tokenizer(prompt, add_special_tokens=False)
            ids = encoded["input_ids"]
            if ids and isinstance(ids[0], list):
                if len(ids) != 1:
                    raise ValueError("Tokenizer returned an unexpected batch")
                ids = ids[0]
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("Tokenizer returned invalid or empty prompt IDs")
            self._validate_context_length(len(ids))
            prompt_ids.append(list(ids))
            converted_batch.append(converted)
        pad = self._generation_kwargs["pad_token_id"]
        width = max(map(len, prompt_ids))
        input_ids = self._torch.full((len(prompt_ids), width), pad, dtype=self._torch.long,
                                     device=self.config.device)
        attention_mask = self._torch.zeros_like(input_ids)
        for row, ids in enumerate(prompt_ids):
            input_ids[row, -len(ids):] = self._torch.tensor(ids, dtype=self._torch.long,
                                                            device=self.config.device)
            attention_mask[row, -len(ids):] = 1
        return input_ids, attention_mask, prompt_ids, converted_batch

    def generate(self, messages: list[Message]) -> str:
        inputs,input_length,_ = self._prepare_inputs(messages)
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

    def generate_with_trace(self, messages: list[Message]) -> dict[str, Any]:
        """RL-only stochastic generation. Baseline generate and its defaults stay intact."""
        if not self.config.do_sample or self.config.top_p != 1 or self._template is None:
            raise ValueError("Traced RL generation requires sampling, top_p=1 and model-owned template")
        inputs,length,converted = self._prepare_inputs(messages)
        kwargs = dict(self._generation_kwargs, top_k=0, use_cache=True,
                      return_dict_in_generate=True, output_scores=True)
        generation_config = self._transformers.GenerationConfig(**kwargs)
        with self._torch.inference_mode():
            output = self._model.generate(
                input_ids=inputs["input_ids"].to(self.config.device),
                attention_mask=inputs["attention_mask"].to(self.config.device),
                generation_config=generation_config, **kwargs)
            ids = output.sequences[0,length:].tolist()
            if len(ids) != len(output.scores) or not ids:
                raise ValueError("Generated IDs/scores length mismatch")
            # Scores are the actual processed sampling logits, including temperature.
            logprobs = self._torch.stack([
                scores[0,token].float() - self._torch.logsumexp(scores[0].float(),dim=-1)
                for scores,token in zip(output.scores,ids,strict=True)]).cpu().tolist()
        eos = kwargs["eos_token_id"]
        eos_ids = list(eos) if isinstance(eos,(list,tuple)) else [eos]
        raw = self._tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        return {"messages":converted,"prompt_ids":inputs["input_ids"][0].tolist(),
                "generated_ids":ids,"old_logprobs":logprobs,"raw_output":raw,
                "ended_with_eos":ids[-1] in eos_ids,
                "token_limit":len(ids) == self.config.max_new_tokens and ids[-1] not in eos_ids}

    def generate_batch_with_trace(
        self, messages_batch: list[list[Message]], *, seeds: list[int] | None = None,
        generators: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Sample one assistant turn for many independent trajectories.

        Prompts are left-padded only for the forward pass. Returned prompt IDs
        never contain padding, so teacher-forced policy/reference scoring keeps
        exactly the same assistant-token boundary as single-example rollout.
        Each row owns a generator, making its sampling stream independent of
        other rows finishing early or having a different prompt length.
        """
        if not self.config.do_sample or self.config.top_p != 1 or self._template is None:
            raise ValueError("Batched traced RL generation requires sampling, top_p=1 and model-owned template")
        if not messages_batch or (seeds is None) == (generators is None):
            raise ValueError("Supply exactly one seed or generator per non-empty batch row")
        if seeds is not None and (len(messages_batch) != len(seeds)
                                  or any(type(seed) is not int or seed < 0 for seed in seeds)):
            raise ValueError("Batch seeds must be non-negative integers")
        if generators is not None and len(messages_batch) != len(generators):
            raise ValueError("Each batch row requires one generator")
        eos = self._generation_kwargs["eos_token_id"]
        if eos is None:
            raise ValueError("Batched traced generation requires eos_token_id")
        eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        input_ids, attention_mask, prompt_ids, converted = self._prepare_batch_inputs(messages_batch)
        torch = self._torch
        batch_size = len(messages_batch)
        device = torch.device(self.config.device)
        if generators is None:
            generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        old_logprobs: list[list[float]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_tensor = torch.tensor(eos_ids, dtype=torch.long, device=device)
        pad = self._generation_kwargs["pad_token_id"]
        # Explicit positions are essential for left padding: real prompt tokens
        # retain the same positions/logits as their unpadded serial counterpart.
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        past = None
        next_ids = input_ids
        with torch.inference_mode():
            for _ in range(self.config.max_new_tokens):
                output = self._model(
                    input_ids=next_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                past = output.past_key_values
                processed = output.logits[:, -1, :].float() / self.config.temperature
                log_probs = torch.log_softmax(processed, dim=-1)
                probabilities = torch.softmax(processed, dim=-1)
                active_before = ~finished
                sampled = torch.full((batch_size,), pad, dtype=torch.long, device=device)
                for row in active_before.nonzero(as_tuple=False).flatten().tolist():
                    token = torch.multinomial(probabilities[row], 1, generator=generators[row]).squeeze(0)
                    sampled[row] = token
                    token_id = int(token.item())
                    generated[row].append(token_id)
                    old_logprobs[row].append(float(log_probs[row, token].item()))
                finished |= active_before & (sampled[:, None] == eos_tensor[None, :]).any(dim=1)
                if bool(finished.all()):
                    break
                attention_mask = torch.cat((attention_mask, active_before[:, None].long()), dim=1)
                next_ids = sampled[:, None]
                position_ids = (attention_mask.long().sum(dim=-1) - 1)[:, None]
        traces = []
        for row, ids in enumerate(generated):
            if not ids or len(ids) != len(old_logprobs[row]):
                raise ValueError("Generated IDs/scores length mismatch")
            ended = ids[-1] in eos_ids
            traces.append({
                "messages": converted[row],
                "prompt_ids": prompt_ids[row],
                "generated_ids": ids,
                "old_logprobs": old_logprobs[row],
                "raw_output": self._tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                ),
                "ended_with_eos": ended,
                "token_limit": len(ids) == self.config.max_new_tokens and not ended,
            })
        return traces

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
            "rl_batch_generation_strategy": "same_turn_left_pad_independent_rng_v1",
            "chat_template_strategy": self.chat_template_strategy,
            "chat_template_sha256": hashlib.sha256(self._template.encode()).hexdigest() if self._template else None,
            "tool_message_strategy": "user_observation_with_tool_name",
            "model_identity": self.model_identity(),
            "adapter_path": self.config.adapter_path,
            "adapter_identity_validated": self._adapter_record is not None,
            "adapter_schema": self._adapter_record.get("schema_version") if self._adapter_record else None,
            "adapter_sha256": self._adapter_hash() if self._adapter_record else None,
            "initial_sft_adapter_sha256": self._adapter_record.get("initial_sft_adapter_sha256") if self._adapter_record else None,
            "adapter_precision_strategy": "exact_checkpoint_fp32_lora" if self._adapter_record and self._adapter_record["schema_version"] == "lora-grpo-run-v1" else "existing_hf_sft_loading",
            "adapter_training_dataset": self._adapter_record.get("dataset") if self._adapter_record else None,
        }

    def _adapter_hash(self) -> str:
        from grpo.policy import adapter_hash
        return adapter_hash(Path(self.config.adapter_path))

    def model_identity(self) -> dict[str, Any]:
        from sft.preprocessing import tokenizer_fingerprint

        return {"base_model_name_or_path": self.config.model_name_or_path,
                "base_model_revision": getattr(self._model.config, "_commit_hash", None),
                "tokenizer_identity": self.config.model_name_or_path,
                "tokenizer_revision": self._tokenizer_revision,
                "chat_template_sha256": hashlib.sha256(self._template.encode()).hexdigest() if self._template else None,
                "tokenizer_sha256": tokenizer_fingerprint(self._tokenizer) if getattr(self._tokenizer, "is_fast", False) else None}
