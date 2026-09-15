"""Explicit assistant-only labels and a fail-closed train/dev preprocessing gate."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from agent.parser import FinalAnswer, parse_model_output
from tasks.schemas import ToolCall
from tasks.validators import KNOWN_TOOLS


def fingerprint(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_fingerprint(tokenizer: Any) -> str:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("SFT requires a fast tokenizer with auditable configuration")
    return fingerprint({"backend": json.loads(tokenizer.backend_tokenizer.to_str()),
                        "special_tokens": tokenizer.special_tokens_map})


def validate_sample(sample: dict[str, Any], split: str) -> None:
    keys = {"schema_version", "sample_id", "task_id", "split", "messages", "assistant_message_indices"}
    if not isinstance(sample, dict) or set(sample) != keys or sample["schema_version"] != "sft-chat-v1":
        raise ValueError("Invalid SFT sample schema")
    if sample["split"] != split or split not in {"train", "dev"}:
        raise ValueError("Only correctly labelled train/dev samples may be loaded")
    if not isinstance(sample["task_id"], str) or not sample["task_id"].strip():
        raise ValueError("Empty task ID")
    if sample["sample_id"] != f"oracle-v1:{sample['task_id']}":
        raise ValueError("Invalid sample ID")
    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) < 5 or len(messages) % 2 != 1:
        raise ValueError("Expected a complete multi-turn demonstration")
    for i, message in enumerate(messages):
        role = "system" if i == 0 else ("user" if i % 2 else "assistant")
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
            or message["role"] != role or not isinstance(message["content"], str) or not message["content"].strip()):
            raise ValueError("Invalid message ordering/content")
    indices = sample["assistant_message_indices"]
    expected = list(range(2, len(messages), 2))
    if not isinstance(indices, list) or any(type(i) is not int for i in indices) or indices != expected:
        raise ValueError("assistant_message_indices must exactly identify all assistant messages including final")
    for i in indices:
        action = parse_model_output(messages[i]["content"])
        if i == len(messages) - 1:
            if not isinstance(action, FinalAnswer):
                raise ValueError("Last supervised assistant message must be final")
        else:
            if not isinstance(action, ToolCall):
                raise ValueError("Non-final assistant messages must be tool-call JSON")
            if action.tool_name not in KNOWN_TOOLS:
                raise ValueError("Unknown tool in supervised demonstration")
            prefix = f"Tool observation ({action.tool_name}): "
            observation = messages[i + 1]["content"]
            if not observation.startswith(prefix):
                raise ValueError("Observation tool identity mismatch")
            result = json.loads(observation[len(prefix):])
            if (not isinstance(result, dict) or set(result) != {"tool_name", "success", "output", "error_code", "error_message"}
                or result["tool_name"] != action.tool_name or result["success"] is not True
                or result["error_code"] is not None or result["error_message"] is not None):
                raise ValueError("Invalid tool observation")


class SpanAlignmentError(ValueError):
    pass


def _render(tokenizer: Any, messages: list[dict[str, str]], generation_prompt: bool = False) -> str:
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template.strip():
        raise ValueError("A model-owned chat template is required; no fallback")
    return tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False,
                                         add_generation_prompt=generation_prompt)


def _ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]


def tokenize_sample(sample: dict[str, Any], tokenizer: Any) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    validate_sample(sample, sample["split"])
    messages = sample["messages"]
    for message in messages:
        if any(token in message["content"] for token in tokenizer.all_special_tokens):
            raise ValueError("Chat control token in message content")
    full_text = _render(tokenizer, messages)  # Full training input always uses False.
    input_ids = _ids(tokenizer, full_text)
    labels = [-100] * len(input_ids)
    spans: list[dict[str, Any]] = []
    for i in sample["assistant_message_indices"]:
        # True is used ONLY to locate the model-owned assistant header, never to
        # construct a separate training input or introduce a custom template.
        before = _render(tokenizer, messages[:i], True)
        after = _render(tokenizer, messages[:i + 1])
        before_ids, after_ids = _ids(tokenizer, before), _ids(tokenizer, after)
        start, end = len(before_ids), len(after_ids)
        if (not full_text.startswith(before) or not full_text.startswith(after)
            or input_ids[:start] != before_ids or input_ids[:end] != after_ids):
            raise SpanAlignmentError("Rendered/tokenized message boundaries are not exact prefixes")
        response = after[len(before):]
        content = messages[i]["content"]
        suffix = response[len(content):] if response.startswith(content) else ""
        eos = tokenizer.eos_token
        if not eos or not suffix.startswith(eos) or suffix[len(eos):].strip():
            raise SpanAlignmentError("Assistant response/terminator differs from supported Qwen template")
        eos_positions = [j for j in range(start, end) if input_ids[j] == tokenizer.eos_token_id]
        if len(eos_positions) != 1 or eos_positions[0] <= start:
            raise SpanAlignmentError("Empty JSON region or ambiguous assistant terminator")
        supervised_end = eos_positions[0] + 1
        if labels[start:supervised_end] != [-100] * (supervised_end - start):
            raise SpanAlignmentError("Overlapping assistant spans")
        labels[start:supervised_end] = input_ids[start:supervised_end]
        action = parse_model_output(content)
        spans.append({"message_index": i, "start": start, "end": supervised_end,
                      "supervised_token_count": supervised_end - start,
                      "action_type": "final" if isinstance(action, FinalAnswer) else "tool_call",
                      "tool_name": action.tool_name if isinstance(action, ToolCall) else None})
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}, spans


@dataclass(slots=True)
class PreprocessingResult:
    features: dict[str, list[dict[str, list[int]]]]
    audit: dict[str, Any]


def _statistics(lengths: list[int]) -> dict[str, int | float | None]:
    ordered = sorted(lengths)
    return {"min": min(ordered) if ordered else None,
            "median": statistics.median(ordered) if ordered else None,
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
            "max": max(ordered) if ordered else None}


def preprocess_dataset(directory: Path, tokenizer: Any, max_seq_length: int,
                       identity: dict[str, Any]) -> PreprocessingResult:
    if type(max_seq_length) is not int or max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "sft-manifest-v1":
        raise ValueError("Unsupported dataset manifest")
    if any(value for pair in manifest["split_overlap"].values() for value in pair.values()):
        raise ValueError("Dataset manifest declares split overlap")
    for split in ("train", "dev", "test"):
        summary = manifest["splits"][split]
        coordinates = summary["coordinates"]
        if not isinstance(coordinates, list) or len(coordinates) != summary["sample_count"] or not coordinates:
            raise ValueError("Invalid manifest coordinate count")
        identities = set()
        for coordinate in coordinates:
            seed, level, index = (coordinate[key] for key in ("seed", "difficulty", "task_index"))
            if (any(type(value) is not int for value in (seed, level, index)) or level not in {1,2,3,4}
                or index < 0 or coordinate["task_id"] != f"task-v1-seed-{seed}-l{level}-index-{index}"):
                raise ValueError("Inconsistent manifest task coordinate")
            identities.add(coordinate["task_id"])
        if len(identities) != len(coordinates) or {c["seed"] for c in coordinates} != set(summary["environment_seeds"]):
            raise ValueError("Duplicate task identity or inconsistent manifest environment seeds")
    # Manifest identities may be read, but test messages/trajectories NEVER are.
    split_ids = {split: {c["task_id"] for c in manifest["splits"][split]["coordinates"]}
                 for split in ("train", "dev", "test")}
    split_seeds = {split: set(manifest["splits"][split]["environment_seeds"]) for split in split_ids}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if split_ids[left] & split_ids[right] or split_seeds[left] & split_seeds[right]:
            raise ValueError("Split task/environment identities overlap")
    features: dict[str, list[dict[str, list[int]]]] = {}
    audit: dict[str, Any] = {"schema_version": "sft-preprocessing-v1", "max_seq_length": max_seq_length,
                             "identity": identity, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                             "file_sha256": {}, "splits": {}}
    seen_chats: set[str] = set()
    seen_expressions: set[str] = set()
    for split in ("train", "dev"):
        name = f"{split}.sft.jsonl"
        content = (directory / name).read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        if file_hash != manifest["files"][name]["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
        samples = [json.loads(line) for line in content.decode("utf-8").splitlines()]
        coordinates = manifest["splits"][split]["coordinates"]
        if ([row["task_id"] for row in samples] != [c["task_id"] for c in coordinates]
            or len(samples) != manifest["files"][name]["row_count"] or not samples):
            raise ValueError("Dataset identity/order/count differs from manifest")
        audit["file_sha256"][name] = file_hash
        features[split] = []
        lengths: list[int] = []
        stats: dict[str, Any] = {"sample_count": len(samples), "supervised_token_count": 0,
                                "zero_supervision_sample_count": 0, "overlength_sample_count": 0,
                                "assistant_span_alignment_failure_count": 0, "invalid_sample_count": 0,
                                "errors": [], "sample_spans": []}
        for sample, coordinate in zip(samples, coordinates, strict=True):
            try:
                validate_sample(sample, split)
                chat_hash = fingerprint(sample["messages"])
                if chat_hash in seen_chats:
                    raise ValueError("Duplicate chat content across train/dev")
                seen_chats.add(chat_hash)
                if coordinate["difficulty"] == 2:
                    first = parse_model_output(sample["messages"][2]["content"])
                    if isinstance(first, ToolCall) and first.tool_name == "calculator":
                        expression = ast.dump(ast.parse(first.arguments["expression"].strip(), mode="eval"), include_attributes=False)
                        if expression in seen_expressions:
                            raise ValueError("Duplicate arithmetic expression across train/dev")
                        seen_expressions.add(expression)
                length = len(_ids(tokenizer, _render(tokenizer, sample["messages"])))
                lengths.append(length)
                stats["overlength_sample_count"] += length > max_seq_length
                encoded, spans = tokenize_sample(sample, tokenizer)
                supervised = sum(label != -100 for label in encoded["labels"])
                stats["supervised_token_count"] += supervised
                stats["zero_supervision_sample_count"] += supervised == 0
                features[split].append(encoded)
                stats["sample_spans"].append({"task_id": sample["task_id"], "difficulty": coordinate["difficulty"],
                                              "sequence_length": length, "supervised_token_count": supervised,
                                              "spans": spans})
            except (ValueError, KeyError, TypeError) as exc:
                key = "assistant_span_alignment_failure_count" if isinstance(exc, SpanAlignmentError) else "invalid_sample_count"
                stats[key] += 1
                stats["errors"].append({"task_id": sample["task_id"], "error": str(exc)})
        total = sum(lengths)
        stats.update(sequence_length=_statistics(lengths), total_token_count=total,
                     supervised_token_ratio=stats["supervised_token_count"] / total if total else 0)
        audit["splits"][split] = stats
    audit["preprocessing_sha256"] = fingerprint(features)
    result = PreprocessingResult(features, audit)
    audit["passed"] = _gate_passes(result)
    return result


def _gate_passes(result: PreprocessingResult) -> bool:
    for split in ("train", "dev"):
        stats = result.audit["splits"][split]
        if any(stats[key] for key in ("zero_supervision_sample_count", "overlength_sample_count",
                                     "assistant_span_alignment_failure_count", "invalid_sample_count")):
            return False
        items = result.features[split]
        if not items or len(items) != stats["sample_count"]:
            return False
        for item in items:
            if (not 0 < len(item["input_ids"]) <= result.audit["max_seq_length"]
                or len(item["labels"]) != len(item["input_ids"]) or len(item["attention_mask"]) != len(item["input_ids"])
                or not any(label != -100 for label in item["labels"])):
                return False
    return result.audit["preprocessing_sha256"] == fingerprint(result.features)


def require_preprocessing_gate(result: PreprocessingResult) -> None:
    if not _gate_passes(result):
        raise ValueError("Preprocessing hard gate failed; Trainer must not start")


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch
        width = max(len(item["input_ids"]) for item in items)
        batch: dict[str, list[list[int]]] = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for item in items:
            padding = width - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
