"""scripts/train_grpo.py - GRPO RL phase (Section 7 of the plan).

Loads the SFT-warm-started model, connects to the OpenEnv server (local or
remote via ``QUBIT_MEDIC_URL``), and runs TRL's :class:`GRPOTrainer` for
2,000 steps with five reward functions registered separately.

Each reward is a Python callable that maps ``(prompts, completions)`` to a
list of floats; TRL aggregates them internally and logs each as its own
column. We keep the weights in :mod:`qubit_medic.config` and apply them
ourselves so the per-component lines on the W&B chart are interpretable.

Run::

    python -m scripts.train_grpo \
        --sft-checkpoint checkpoints/sft_warmup \
        --output checkpoints/grpo \
        --steps 2000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# --------------------------------------------------------------------------- #
# Lightweight rollout buffer - one syndrome per (prompt, completion_group).    #
# --------------------------------------------------------------------------- #


@dataclass
class _PromptContext:
    episode_id: int
    obs_dict: dict
    used: bool = False


@dataclass
class _RolloutTracker:
    """Maps each prompt back to its env episode so reward fns can score."""
    by_prompt: dict[str, _PromptContext] = field(default_factory=dict)


_TRACKER = _RolloutTracker()


def _seed_everything(seed: int) -> None:
    import numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Dataset of prompts (TRL's GRPOTrainer expects a HuggingFace Dataset)        #
# --------------------------------------------------------------------------- #


def _build_prompt_iterator(env_client, n: int):
    """Pre-generate ``n`` syndromes via the env so each training prompt has a
    cached scoring context. We don't call env.reset() inside the trainer's
    loop because TRL re-tokenises the same prompt across generations."""
    prompts = []
    for _ in range(n):
        obs = env_client.reset()
        ctx = _PromptContext(
            episode_id=obs.episode_id,
            obs_dict=obs.model_dump(),
        )
        _TRACKER.by_prompt[obs.prompt] = ctx
        prompts.append({"prompt": obs.prompt, "episode_id": obs.episode_id})
    return prompts


def _score_completion(env_client, prompt: str, completion: str) -> dict:
    """Step the env for a single (prompt, completion) pair.

    Each completion needs its own episode (single-step episodes), so we
    call reset() once per scoring call. The cached prompt-context is only
    used to log episode counts for W&B.
    """
    obs = env_client.reset()
    result = env_client.step(raw_response=completion, episode_id=obs.episode_id)
    return result.info["rewards"]


# --------------------------------------------------------------------------- #
# Reward callables - one per Section 3 component, plus the weighted total.    #
# --------------------------------------------------------------------------- #


def _make_reward_fns(env_client):
    component_names = (
        "logical_correction",
        "syndrome_consistency",
        "hamming_overlap",
        "format_compliance",
        "pymatching_beat",
    )

    def _score_batch(prompts, completions):
        return [_score_completion(env_client, p, c) for p, c in zip(prompts, completions)]

    def _factory(name):
        def fn(prompts, completions, **_unused):
            scores = _score_batch(prompts, completions)
            return [s[name] for s in scores]
        fn.__name__ = f"reward_{name}"
        return fn

    return [_factory(n) for n in component_names]


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-checkpoint", type=str, default="checkpoints/sft_warmup")
    parser.add_argument("--output", type=str, default="checkpoints/grpo")
    parser.add_argument("--model", type=str,
                        default=os.getenv("QUBIT_MEDIC_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--gen-per-prompt", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--kl-coef", type=float, default=None)
    parser.add_argument("--max-prompt-len", type=int, default=None)
    parser.add_argument("--max-completion-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--prompt-pool", type=int, default=512,
                        help="number of pre-generated prompts to draw training "
                             "questions from. Each prompt is paired with a "
                             "fresh syndrome on every step.")
    args = parser.parse_args(list(argv))

    # Lazy heavy imports.
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. Run `pip install -r requirements-train.txt`",
              file=sys.stderr)
        return 1
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from qubit_medic.client.client import make_default_client
    from qubit_medic.config import (
        GRPO_CHECKPOINT_EVERY, GRPO_GEN_PER_PROMPT, GRPO_KL_COEF, GRPO_LOG_EVERY,
        GRPO_LR, GRPO_MAX_COMPLETION_LEN, GRPO_MAX_PROMPT_LEN, GRPO_STEPS,
        LORA_ALPHA, LORA_R, LORA_TARGET_MODULES, MODEL_ID, PRIMARY_SEED,
    )

    steps = args.steps if args.steps is not None else GRPO_STEPS
    gen_per_prompt = args.gen_per_prompt if args.gen_per_prompt is not None else GRPO_GEN_PER_PROMPT
    lr = args.lr if args.lr is not None else GRPO_LR
    kl_coef = args.kl_coef if args.kl_coef is not None else GRPO_KL_COEF
    max_p = args.max_prompt_len if args.max_prompt_len is not None else GRPO_MAX_PROMPT_LEN
    max_c = args.max_completion_len if args.max_completion_len is not None else GRPO_MAX_COMPLETION_LEN
    seed = args.seed if args.seed is not None else PRIMARY_SEED

    _seed_everything(seed)

    env_client = make_default_client()
    print(f"using env client: {type(env_client).__name__}; "
          f"health = {env_client.health()}")

    print(f"pre-generating {args.prompt_pool} prompts ...")
    prompts = _build_prompt_iterator(env_client, args.prompt_pool)
    dataset = Dataset.from_list(prompts)
    print(f"  built dataset with {len(dataset)} prompts")

    print(f"loading model: {args.sft_checkpoint or args.model}")
    base = args.sft_checkpoint if args.sft_checkpoint and Path(args.sft_checkpoint).exists() else args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=max_p + max_c,
        load_in_4bit=True,
        dtype=None,
    )
    if not Path(base).is_dir():
        # No SFT checkpoint - attach a fresh LoRA adapter.
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=list(LORA_TARGET_MODULES),
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

    Path(args.output).mkdir(parents=True, exist_ok=True)
    config = GRPOConfig(
        output_dir=args.output,
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=gen_per_prompt,
        max_prompt_length=max_p,
        max_completion_length=max_c,
        learning_rate=lr,
        beta=kl_coef,  # KL coefficient in TRL's GRPOConfig
        logging_steps=GRPO_LOG_EVERY,
        save_steps=GRPO_CHECKPOINT_EVERY,
        save_total_limit=4,
        seed=seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=args.report_to,
    )

    reward_fns = _make_reward_fns(env_client)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fns,
    )

    print(f"running GRPO for {steps} steps...")
    started = time.time()
    trainer.train()
    print(f"finished in {time.time() - started:.1f}s")

    print(f"saving adapters to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
