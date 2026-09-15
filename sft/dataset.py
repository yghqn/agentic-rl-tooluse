"""Deterministic split construction and auditable JSONL export; no training."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any

from agent.agent import AgentRun, PromptAgent, ScriptedBackend
from agent.prompts import build_messages
from environment.database import SyntheticCompanyDatabase
from environment.environment import Environment
from evaluation.verifier import VerificationResult
from scripts.evaluate import project_git_state
from sft.oracle import ORACLE_VERSION, generate_oracle_trajectory
from sft.validation import arithmetic_key, to_sft_messages, validate_demonstration
from tasks.generator import TaskGenerator, agent_task_view
from tasks.schemas import Task


SPLITS = ("train", "dev", "test")
TASK_TYPES = ("single_retrieval", "list_companies", "arithmetic", "profit_margin", "highest_profit_margin")
EXPORT_FILES = tuple(f"{split}.{kind}.jsonl" for split in SPLITS for kind in ("trajectories", "sft")) + ("manifest.json",)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    train_size: int = 64
    dev_size: int = 16
    test_size: int = 16
    train_seed_start: int = 10000
    dev_seed_start: int = 20000
    test_seed_start: int = 30000
    max_candidates_per_level: int = 1000

    def __post_init__(self) -> None:
        # Four examples per level per environment: keeps environment diversity
        # and exact L1-L4 quotas explicit, without a generic sampling framework.
        for split in SPLITS:
            size = getattr(self, f"{split}_size")
            if type(size) is not int or size < 16 or size % 16:
                raise ValueError("Split sizes must be positive multiples of 16")
            if type(getattr(self, f"{split}_seed_start")) is not int:
                raise ValueError("Seed starts must be integers")
        if type(self.max_candidates_per_level) is not int or self.max_candidates_per_level <= 0:
            raise ValueError("max_candidates_per_level must be a positive integer")
        for left, right in combinations(SPLITS, 2):
            if set(self.seeds(left)) & set(self.seeds(right)):
                raise ValueError("Environment seed ranges overlap")

    def seeds(self, split: str) -> range:
        start = getattr(self, f"{split}_seed_start")
        return range(start, start + getattr(self, f"{split}_size") // 16)


@dataclass(slots=True)
class Demonstration:
    task: Task
    seed: int
    task_index: int
    run: AgentRun
    verification: VerificationResult

    @property
    def sample_id(self) -> str:
        return f"{ORACLE_VERSION}:{self.task.task_id}"

    def chat_sample(self, split: str) -> dict[str, Any]:
        messages = to_sft_messages(build_messages(agent_task_view(self.task), self.run.events))
        return {
            "schema_version": "sft-chat-v1", "sample_id": self.sample_id,
            "task_id": self.task.task_id, "split": split, "messages": messages,
            "assistant_message_indices": [i for i, m in enumerate(messages) if m["role"] == "assistant"],
        }

    def raw_record(self, split: str) -> dict[str, Any]:
        # Never asdict(Task): privileged fields have no place in the export.
        return {
            "schema_version": "oracle-trajectory-v1", "sample_id": self.sample_id,
            "task_id": self.task.task_id, "split": split,
            "environment_id": self.task.environment_id, "environment_seed": self.seed,
            "difficulty": self.task.difficulty, "oracle_version": ORACLE_VERSION,
            "agent_view": agent_task_view(self.task), "run": asdict(self.run),
            "messages": build_messages(agent_task_view(self.task), self.run.events),
            "verification": asdict(self.verification),
        }


@dataclass(slots=True)
class DatasetBundle:
    records: dict[str, list[Demonstration]]
    manifest: dict[str, Any]


def chat_fingerprint(sample: dict[str, Any]) -> str:
    """Only actual model-visible conversation content, no audit fields."""

    return hashlib.sha256(_json(sample["messages"]).encode("utf-8")).hexdigest()


def check_split_isolation(records: dict[str, list[Demonstration]]) -> dict[str, Any]:
    sets: dict[str, dict[str, set[Any]]] = {}
    for split in SPLITS:
        items = records[split]
        ids = [item.task.task_id for item in items]
        chats = [chat_fingerprint(item.chat_sample(split)) for item in items]
        expressions = [arithmetic_key(item.task.metadata["expression"]) for item in items
                       if item.task.verification_spec["task_type"] == "arithmetic"]
        if len(ids) != len(set(ids)) or len(chats) != len(set(chats)) or len(expressions) != len(set(expressions)):
            raise ValueError("Duplicate task, chat content or arithmetic expression within split")
        sets[split] = {
            "environment_seed": {item.seed for item in items},
            "task_coordinate": {(item.seed, item.task.difficulty, item.task_index) for item in items},
            "task_id": set(ids), "chat_content": set(chats), "arithmetic_expression": set(expressions),
        }
        lists = sum(item.task.verification_spec["task_type"] == "list_companies" for item in items)
        if lists != (1 if split == "train" else 0):
            raise ValueError("Exactly one list demonstration is allowed, in train only")
    overlaps: dict[str, Any] = {}
    for left, right in combinations(SPLITS, 2):
        counts = {f"{key}_overlap_count": len(sets[left][key] & sets[right][key]) for key in sets[left]}
        if any(counts.values()):
            raise ValueError(f"Split overlap between {left} and {right}: {counts}")
        overlaps[f"{left}_{right}"] = counts
    return overlaps


def _split_summary(items: list[Demonstration]) -> dict[str, Any]:
    difficulty = Counter(item.task.difficulty for item in items)
    kinds = Counter(item.task.verification_spec["task_type"] for item in items)
    return {
        "sample_count": len(items),
        "counts_by_difficulty": {f"L{level}": difficulty[level] for level in range(1, 5)},
        "counts_by_task_type": {kind: kinds[kind] for kind in TASK_TYPES},
        "environment_seeds": sorted({item.seed for item in items}),
        "coordinates": [{"seed": item.seed, "difficulty": item.task.difficulty,
                         "task_index": item.task_index, "task_id": item.task.task_id} for item in items],
        "validation": {
            "checked_count": len(items), "replay_success_count": len(items),
            "verifier_success_count": sum(item.verification.success for item in items),
            "leakage_check_success_count": len(items),
        },
    }


def build_dataset(config: DatasetConfig = DatasetConfig()) -> DatasetBundle:
    records: dict[str, list[Demonstration]] = {split: [] for split in SPLITS}
    seen_chats: set[str] = set()
    seen_expressions: set[str] = set()
    skips: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        skips[split] = {"chat_duplicates": 0, "arithmetic_duplicates": 0}
        for seed in config.seeds(split):
            generator = TaskGenerator(seed)
            database = SyntheticCompanyDatabase(seed)
            for level in range(1, 5):
                accepted = 0
                for candidate in range(config.max_candidates_per_level):
                    is_list = split == "train" and seed == config.train_seed_start and level == 2 and candidate == 0
                    index = candidate if level != 2 else (0 if is_list else 2 * candidate + 1)
                    task = generator.generate_task(level, index)
                    expression_key = arithmetic_key(task.metadata["expression"]) if task.verification_spec["task_type"] == "arithmetic" else None
                    if expression_key is not None and expression_key in seen_expressions:
                        skips[split]["arithmetic_duplicates"] += 1
                        continue
                    run = generate_oracle_trajectory(task, database)
                    verification = validate_demonstration(task, database, run)
                    item = Demonstration(task, seed, index, run, verification)
                    sample = item.chat_sample(split)
                    validate_demonstration(task, database, run, sft_sample=sample)
                    fingerprint = chat_fingerprint(sample)
                    if fingerprint in seen_chats:
                        skips[split]["chat_duplicates"] += 1
                        continue
                    records[split].append(item)
                    seen_chats.add(fingerprint)
                    if expression_key is not None:
                        seen_expressions.add(expression_key)
                    accepted += 1
                    if accepted == 4:
                        break
                if accepted != 4:
                    raise ValueError(f"Candidate limit exhausted: {split}, seed={seed}, L{level}")
        records[split].sort(key=lambda item: (item.task.difficulty, item.seed, item.task_index))
    overlaps = check_split_isolation(records)
    manifest = {
        "schema_version": "sft-manifest-v1", "oracle_version": ORACLE_VERSION,
        "configuration": asdict(config), "ordering": "difficulty_then_seed_then_task_index",
        "message_strategy": "hf_user_observation_with_tool_name",
        "supervision": "assistant_tool_calls_and_final_only_no_cot",
        "deduplication_strategy": "global_chat_sha256_and_arithmetic_ast",
        "list_policy": "one_train_example_no_dev_or_test_examples",
        "splits": {split: {**_split_summary(records[split]), "deduplication": skips[split]} for split in SPLITS},
        "split_overlap": overlaps,
        **project_git_state(),
    }
    return DatasetBundle(records, manifest)


def check_export_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError("Output path must be a directory")
    if any((path / name).exists() for name in EXPORT_FILES):
        raise ValueError("Dataset files already exist; choose a new output directory")


def export_dataset(bundle: DatasetBundle, path: Path) -> dict[str, Any]:
    check_export_directory(path)
    # Revalidate mutable records immediately before exporting.
    for split, items in bundle.records.items():
        order = [(item.task.difficulty, item.seed, item.task_index) for item in items]
        if order != sorted(order):
            raise ValueError("Non-deterministic dataset ordering")
        for item in items:
            verification = validate_demonstration(item.task, SyntheticCompanyDatabase(item.seed), item.run,
                                                 messages=item.raw_record(split)["messages"], sft_sample=item.chat_sample(split))
            if verification != item.verification:
                raise ValueError("Recorded verification differs from independent verification")
    overlaps = check_split_isolation(bundle.records)
    for split in SPLITS:
        summary = _split_summary(bundle.records[split])
        if any(bundle.manifest["splits"][split].get(key) != value for key, value in summary.items()):
            raise ValueError("Manifest differs from actual dataset")
    if bundle.manifest["split_overlap"] != overlaps:
        raise ValueError("Manifest overlap report is inconsistent")
    path.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for split in SPLITS:
        for kind in ("trajectories", "sft"):
            name = f"{split}.{kind}.jsonl"
            with (path / name).open("x", encoding="utf-8", newline="\n") as stream:
                for item in bundle.records[split]:
                    row = item.raw_record(split) if kind == "trajectories" else item.chat_sample(split)
                    stream.write(_json(row) + "\n")
            files[name] = {"sha256": hashlib.sha256((path / name).read_bytes()).hexdigest(),
                           "row_count": len(bundle.records[split])}
    manifest = {**bundle.manifest, "files": files}
    with (path / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def audit_export(path: Path) -> dict[str, Any]:
    """Re-read exported files, reconstruct tasks, replay and verify without oracle."""

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sft-manifest-v1" or manifest.get("oracle_version") != ORACLE_VERSION:
        raise ValueError("Unsupported dataset manifest version")
    if set(manifest.get("files", {})) != set(EXPORT_FILES) - {"manifest.json"}:
        raise ValueError("Unexpected dataset file manifest")
    records: dict[str, list[Demonstration]] = {split: [] for split in SPLITS}
    config = DatasetConfig(**manifest["configuration"])
    for split in SPLITS:
        rows: dict[str, list[dict[str, Any]]] = {}
        for kind in ("trajectories", "sft"):
            name = f"{split}.{kind}.jsonl"
            content = (path / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != manifest["files"][name]["sha256"]:
                raise ValueError(f"Export checksum mismatch: {name}")
            rows[kind] = [json.loads(line) for line in content.decode("utf-8").splitlines()]
            if len(rows[kind]) != manifest["files"][name]["row_count"]:
                raise ValueError("Export row count mismatch")
        coordinates = manifest["splits"][split]["coordinates"]
        if not len(rows["trajectories"]) == len(rows["sft"]) == len(coordinates) == getattr(config, f"{split}_size"):
            raise ValueError("Export size mismatch")
        for coordinate, raw, sample in zip(coordinates, rows["trajectories"], rows["sft"], strict=True):
            seed, level, index = coordinate["seed"], coordinate["difficulty"], coordinate["task_index"]
            if seed not in config.seeds(split):
                raise ValueError("Coordinate seed outside configured split")
            task = TaskGenerator(seed).generate_task(level, index)
            if coordinate["task_id"] != task.task_id:
                raise ValueError("Coordinate task ID mismatch")
            run = PromptAgent(ScriptedBackend([event["raw_output"] for event in raw["run"]["events"]]),
                              raw["run"]["max_steps"]).run(agent_task_view(task), Environment(seed))
            verification = validate_demonstration(task, SyntheticCompanyDatabase(seed), run,
                                                 messages=raw["messages"], sft_sample=sample)
            item = Demonstration(task, seed, index, run, verification)
            if raw != item.raw_record(split) or sample != item.chat_sample(split):
                raise ValueError("Export differs from replayed records")
            records[split].append(item)
        order = [(item.task.difficulty, item.seed, item.task_index) for item in records[split]]
        if order != sorted(order):
            raise ValueError("Non-deterministic export ordering")
        summary = _split_summary(records[split])
        if any(manifest["splits"][split].get(key) != value for key, value in summary.items()):
            raise ValueError("Manifest validation/distribution mismatch")
    overlaps = check_split_isolation(records)
    if overlaps != manifest["split_overlap"]:
        raise ValueError("Manifest overlap mismatch")
    return {"splits": {split: _split_summary(records[split]) for split in SPLITS}, "split_overlap": overlaps}
