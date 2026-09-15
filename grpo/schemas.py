"""RL-specific records deliberately kept out of shared environment schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from agent.agent import AgentRun
from evaluation.verifier import VerificationResult
from tasks.schemas import Task


@dataclass(frozen=True, slots=True)
class TaskCoordinate:
    seed: int
    difficulty: int
    task_index: int
    task_id: str


@dataclass(slots=True)
class TurnTrace:
    messages: list[dict[str, str]]
    prompt_ids: list[int]
    generated_ids: list[int]
    old_logprobs: list[float]
    raw_output: str
    ended_with_eos: bool
    token_limit: bool

    def validate(self) -> None:
        if (not self.prompt_ids or not self.generated_ids
            or len(self.generated_ids) != len(self.old_logprobs)
            or any(type(i) is not int or i < 0 for i in self.prompt_ids + self.generated_ids)
            or any(not math.isfinite(p) for p in self.old_logprobs)):
            raise ValueError("Malformed sampled-token trace")

    @property
    def policy_mask(self) -> list[int]:
        return [0] * len(self.prompt_ids) + [1] * len(self.generated_ids)


@dataclass(frozen=True, slots=True)
class RewardResult:
    total: float
    components: dict[str, float]
    counts: dict[str, int]


@dataclass(slots=True)
class RolloutRecord:
    rollout_id: str
    coordinate: TaskCoordinate
    group_id: str
    member_index: int
    generation_seed: int
    policy_version: int
    run: AgentRun
    turns: list[TurnTrace]
    verification: VerificationResult
    reward: RewardResult
    reference_logprobs: list[list[float]] = field(default_factory=list)


@dataclass(slots=True)
class RolloutGroup:
    group_id: str
    members: list[RolloutRecord]


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    sft_adapter: str
    output_dir: str
    sft_manifest: str
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    cache_dir: str | None = None
    tokenizer_path: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    local_files_only: bool = True
    seed: int = 42
    group_size: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    max_new_tokens: int = 256
    max_steps: int = 16
    max_scoring_length: int = 8192
    prompts_per_step: int = 2
    optimizer_steps: int = 10
    learning_rate: float = 1e-5
    reward_variant: str = "outcome_only"
    kl_beta: float = 0.02
    clip_epsilon: float = 0.2
    std_floor: float = 0.1
    alignment_tolerance: float = 0.05

    def __post_init__(self) -> None:
        for name in ("sft_adapter", "output_dir", "sft_manifest", "model", "revision", "device"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("group_size", "max_new_tokens", "max_steps", "max_scoring_length", "prompts_per_step", "optimizer_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive integer")
        if self.group_size < 2 or type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Need G >= 2 and a valid seed")
        for name in ("temperature", "learning_rate", "std_floor", "alignment_tolerance"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        if self.top_p != 1.0:
            raise ValueError("V1 requires top_p=1 and top_k=0 for matching scoring distributions")
        if type(self.top_p) not in (int,float) or type(self.local_files_only) is not bool:
            raise ValueError("Invalid sampling/loading configuration type")
        if self.reward_variant not in {"outcome_only", "shaped"} or self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("Unsupported reward/dtype")
        if not math.isfinite(self.kl_beta) or self.kl_beta < 0 or not 0 < self.clip_epsilon < 1:
            raise ValueError("Invalid KL/clipping configuration")


def task_from_coordinate(coordinate: TaskCoordinate) -> Task:
    from tasks.generator import TaskGenerator
    task = TaskGenerator(coordinate.seed).generate_task(coordinate.difficulty, coordinate.task_index)
    if task.task_id != coordinate.task_id:
        raise ValueError("Task coordinate identity mismatch")
    return task
