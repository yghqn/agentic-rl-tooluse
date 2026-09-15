"""Qwen LoRA SFT CLI. Preprocessing is mandatory and never silently truncates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft.preprocessing import preprocess_dataset
from sft.training import (
    TrainingConfig, inspect_model, load_training_tokenizer, run_preprocessing, train,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess/audit, inspect, or explicitly train Qwen LoRA")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preprocess-only", action="store_true")
    mode.add_argument("--inspect-model-only", action="store_true", help="Load base + LoRA and count parameters; NO Trainer")
    arguments = parser.parse_args(argv)
    try:
        config = TrainingConfig(
            dataset_dir=str(arguments.dataset_dir), output_dir=str(arguments.output_dir),
            model_name_or_path=arguments.model, revision=arguments.revision, cache_dir=arguments.cache_dir,
            local_files_only=arguments.local_files_only, device=arguments.device, dtype=arguments.dtype,
            seed=arguments.seed, max_seq_length=arguments.max_seq_length, epochs=arguments.epochs,
            learning_rate=arguments.learning_rate, batch_size=arguments.batch_size,
            gradient_accumulation_steps=arguments.gradient_accumulation_steps,
            lora_rank=arguments.lora_rank, lora_alpha=arguments.lora_alpha, lora_dropout=arguments.lora_dropout,
        )
        tokenizer, identity = load_training_tokenizer(config)
        if arguments.preprocess_only:
            result = run_preprocessing(config, tokenizer, identity)
            output = {"mode":"preprocess_only", "passed":result.audit["passed"],
                      "splits":{k:{a:b for a,b in v.items() if a not in {"sample_spans", "errors"}}
                                for k,v in result.audit["splits"].items()}}
        else:
            # Recompute instead of trusting a stale preprocessing report or cached labels.
            result = preprocess_dataset(arguments.dataset_dir, tokenizer, config.max_seq_length, identity)
            output = inspect_model(config, result, identity) if arguments.inspect_model_only else train(config, tokenizer, result, identity)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
