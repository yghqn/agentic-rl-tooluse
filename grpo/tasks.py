"""Coordinate-only train/dev construction. Never opens SFT chat/trajectory files."""
from __future__ import annotations

import ast
from dataclasses import asdict
import json
from pathlib import Path
import random

from grpo.schemas import TaskCoordinate
from tasks.generator import TaskGenerator


def expression_key(expression: str) -> str:
    return ast.dump(ast.parse(expression.strip(), mode="eval"), include_attributes=False)


def build_task_manifest(sft_manifest: Path) -> dict:
    original = json.loads(sft_manifest.read_text(encoding="utf-8"))
    if original.get("schema_version") != "sft-manifest-v1":
        raise ValueError("Need the existing SFT coordinate manifest")
    reserved_seeds, reserved_ids, expressions = set(), set(), set()
    for split in ("train", "dev", "test"):
        for c in original["splits"][split]["coordinates"]:
            task = TaskGenerator(c["seed"]).generate_task(c["difficulty"], c["task_index"])
            if task.task_id != c["task_id"]:
                raise ValueError("Invalid SFT task identity")
            reserved_seeds.add(c["seed"])
            reserved_ids.add(c["task_id"])
            if task.verification_spec["task_type"] == "arithmetic":
                expressions.add(expression_key(task.metadata["expression"]))
    output = {"schema_version":"grpo-task-manifest-v1", "splits":{}, "overlap_counts":{
        "environment_seed":0, "task_coordinate":0, "task_id":0, "arithmetic_expression":0}}
    all_ids, all_seeds = set(reserved_ids), set(reserved_seeds)
    for split, start, count in (("train",40000,256), ("dev",50000,64)):
        coordinates = []
        for offset, seed in enumerate(range(start, start + count)):
            if seed in all_seeds:
                raise ValueError("Reserved environment seed overlaps")
            all_seeds.add(seed)
            generator = TaskGenerator(seed)
            quotas = {1:1 if split == "dev" or offset % 2 == 0 else 0,
                      2:1 if split == "dev" or offset % 2 == 1 else 0,
                      3:1, 4:1 if split == "dev" else 8}
            for difficulty, quota in quotas.items():
                seen_questions = set()
                accepted = 0
                for index in range(1000):
                    if accepted == quota:
                        break
                    if difficulty == 2 and index % 2 == 0:
                        continue  # No repeated list task in RL.
                    task = generator.generate_task(difficulty,index)
                    if task.question in seen_questions:
                        continue
                    if difficulty == 2:
                        key = expression_key(task.metadata["expression"])
                        if key in expressions:
                            continue
                        expressions.add(key)
                    if task.task_id in all_ids:
                        raise ValueError("Duplicate task identity")
                    all_ids.add(task.task_id)
                    seen_questions.add(task.question)
                    coordinates.append(asdict(TaskCoordinate(seed,difficulty,index,task.task_id)))
                    accepted += 1
                if accepted != quota:
                    raise ValueError("Task candidate budget exhausted")
        output["splits"][split] = {"coordinates":coordinates, "environment_seeds":list(range(start,start+count)),
                                   "sample_count":len(coordinates)}
    return output


def coordinates(manifest: dict, split: str) -> list[TaskCoordinate]:
    return [TaskCoordinate(**{k:c[k] for k in ("seed","difficulty","task_index","task_id")})
            for c in manifest["splits"][split]["coordinates"]]


def training_schedule(manifest: dict, seed: int, count: int) -> list[TaskCoordinate]:
    if type(count) is not int or count <= 0:
        raise ValueError("Schedule length must be positive integer")
    pools = {level:[c for c in coordinates(manifest,"train") if c.difficulty == level] for level in (1,2,3,4)}
    if any(not pool for pool in pools.values()):
        raise ValueError("Training schedule requires all four train difficulty pools")
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    order, positions = [], {level:0 for level in pools}
    pattern = [1,2,3,3] + [4] * 16
    while len(order) < count:
        block = pattern.copy()
        rng.shuffle(block)
        for level in block:
            pool = pools[level]
            order.append(pool[positions[level] % len(pool)])
            positions[level] += 1
            if len(order) == count:
                break
    return order
