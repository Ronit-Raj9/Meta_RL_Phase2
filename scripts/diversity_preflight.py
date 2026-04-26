"""Diversity preflight gate between SFT and GRPO (2026-04 spec, FIX 2).

Loads the SFT-warm-started LoRA adapter on top of the 4-bit NF4 quantised
``unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit`` base, picks 5 prompts
from ``data/sft_validation.jsonl`` (mix of trivial and non-trivial),
generates 8 completions per prompt at temperature=1.2, and counts the
unique completions per prompt.

PASS: at least 3 of 5 prompts have >= 3 unique completions.
FAIL: anything less; the model is too collapsed for GRPO to recover.

Exit codes:
    0  PASS  -> safe to launch GRPO
    2  FAIL  -> redo SFT with higher LoRA dropout (0.15) or more
              label smoothing.
    1  Other error (model load failed, validation file missing, etc.)

This script reuses the shared ``_diversity_preflight`` function from
``scripts.train_grpo`` so the gate logic is identical to the one GRPO
runs at its own startup (defence in depth: even if a caller skips this
script, GRPO will refuse to start on a collapsed model).

Usage::

    python -m scripts.diversity_preflight \
        --sft-checkpoint checkpoints/sft_warmup \
        --val data/sft_validation.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft-checkpoint", type=str, default="checkpoints/sft_warmup",
        help="LoRA adapter directory (or any HuggingFace model id) to load. "
             "Defaults to checkpoints/sft_warmup (the rolling SFT save).",
    )
    parser.add_argument(
        "--base-model", type=str,
        default="unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit",
        help="Base model to load if --sft-checkpoint is missing.",
    )
    parser.add_argument(
        "--val", type=str, default="data/sft_validation.jsonl",
        help="JSONL of validation rows (must contain 'prompt' and "
             "'had_errors' fields).",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=5,
        help="How many prompts to probe (default 5).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=8,
        help="Completions to draw per prompt at high temperature.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.2,
        help="Sampling temperature for the diversity probe.",
    )
    parser.add_argument(
        "--min-unique", type=int, default=3,
        help="A prompt PASSES if it produces at least this many unique "
             "completions out of n-samples.",
    )
    parser.add_argument(
        "--min-passing", type=int, default=3,
        help="The run PASSES if at least this many prompts pass.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="Generation cap per completion.",
    )
    args = parser.parse_args(list(argv))

    # Lazy heavy imports.
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. "
              "Run `pip install -r requirements-train.txt`", file=sys.stderr)
        return 1

    val = Path(args.val)
    if not val.exists():
        print(f"ERROR: validation file not found: {val}", file=sys.stderr)
        return 1

    sft = Path(args.sft_checkpoint)
    base_for_load = str(sft) if sft.exists() else args.base_model
    if not sft.exists():
        print(f"[preflight] WARN: {sft} not found; loading base "
              f"{args.base_model} only (no SFT adapter applied). The "
              f"diversity result still says something about the base "
              f"model's behaviour but will not reflect the SFT run.",
              file=sys.stderr)

    print(f"[preflight] loading model: {base_for_load}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_for_load,
        max_seq_length=1500 + args.max_new_tokens,
        load_in_4bit=True,
        dtype=None,
    )

    # Reuse the shared preflight function from train_grpo so this gate
    # is byte-identical to the one GRPO will run at its own startup.
    from scripts.train_grpo import _diversity_preflight
    ok = _diversity_preflight(
        model, tokenizer,
        val_path=str(val),
        n_prompts=args.n_prompts,
        n_samples_per_prompt=args.n_samples,
        temperature=args.temperature,
        min_unique=args.min_unique,
        min_passing=args.min_passing,
        max_new_tokens=args.max_new_tokens,
    )

    # Optional: write the result to W&B summary if a run is open.
    try:
        from qubit_medic import wandb_utils
        wandb_utils.update_summary({
            "preflight/passed": bool(ok),
            "preflight/sft_checkpoint": str(args.sft_checkpoint),
        })
    except Exception:
        pass

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
