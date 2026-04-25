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
        --steps 2000 \
        --report-to wandb

W&B logging
-----------
On every GRPO step:

* ``rl/reward/<component>_mean|std|min|max`` for all 5 components and total
* ``rl/parse/{success,partial,failure}_rate``
* ``rl/curriculum/<level>_mean`` and ``..._samples``
* ``rl/env/{episodes_started,active_episodes,timeout_rate}``

Every :data:`qubit_medic.config.WANDB_LOG_GENERATIONS_EVERY` steps a
generation table is uploaded with prompt, completion, parse status, and
each reward component.

Every :data:`qubit_medic.config.WANDB_INLOOP_EVAL_EVERY` steps a small
held-out greedy eval pass logs ``eval/*`` scalars so the GRPO curve and
the actual eval curve are visible side-by-side. The trained LoRA adapter
directory is uploaded as a W&B artifact at the end of training.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# --------------------------------------------------------------------------- #
# Per-batch scoring cache                                                     #
# --------------------------------------------------------------------------- #
#
# The original implementation called the env 5 times per (prompt, completion)
# - once per reward function. That's 5x wasted env work AND each reward
# function would see a different syndrome (since reset() picks a fresh one),
# so the per-component scores wouldn't be comparable. We fix both with a
# single (prompt, completion) -> breakdown cache keyed inside one GRPO step.
# --------------------------------------------------------------------------- #


@dataclass
class _ScoredCompletion:
    """One scored (prompt, completion) pair, keyed by the env episode."""
    rewards: dict
    parse_success: bool
    parse_partial: bool
    x_pred: list
    z_pred: list
    actual_flip: int
    pm_flip: int
    elapsed: float
    timed_out: bool
    curriculum_level: str


@dataclass
class _BatchScoringCache:
    """Caches per-(prompt, completion) scores within one GRPO step."""
    env_client: object
    _cache: dict = field(default_factory=dict)
    _step_keys: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _all_curriculum_stats: dict = field(default_factory=dict)
    _episodes: int = 0
    _timeouts: int = 0

    def score(self, prompt: str, completion: str) -> _ScoredCompletion:
        key = (prompt, completion)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                return entry
        # Env work is independent across (p, c) so it's safe to release the
        # lock during the network round-trip.
        obs = self.env_client.reset()
        result = self.env_client.step(raw_response=completion,
                                      episode_id=obs.episode_id)
        info = result.info
        action = info.get("parsed_action", {})
        scored = _ScoredCompletion(
            rewards=info["rewards"],
            parse_success=bool(action.get("parse_success", False)),
            parse_partial=False,
            x_pred=list(action.get("x_error_qubits", []) or []),
            z_pred=list(action.get("z_error_qubits", []) or []),
            actual_flip=int(info.get("actual_observable_flip", 0)),
            pm_flip=int(info.get("pymatching_observable_pred", 0)),
            elapsed=float(info.get("elapsed_seconds", 0.0)),
            timed_out=bool(info.get("timed_out", False)),
            curriculum_level=str(getattr(obs, "curriculum_level", "")),
        )
        with self._lock:
            self._cache[key] = scored
            self._step_keys.append(key)
            self._all_curriculum_stats = info.get("curriculum_stats", {}) or {}
            self._episodes += 1
            if scored.timed_out:
                self._timeouts += 1
        return scored

    def drain_step(self):
        """Pop everything cached since the last drain_step() call."""
        with self._lock:
            entries = [self._cache[k] for k in self._step_keys]
            keys = list(self._step_keys)
            self._step_keys.clear()
            # Bound memory use - long runs with unique strings.
            if len(self._cache) > 4096:
                self._cache.clear()
            return entries, keys


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Reward function factory - all 5 share one cache                             #
# --------------------------------------------------------------------------- #


def _make_reward_fns(cache: _BatchScoringCache):
    component_names = (
        "logical_correction",
        "syndrome_consistency",
        "hamming_overlap",
        "format_compliance",
        "pymatching_beat",
    )

    def _factory(name):
        def fn(prompts, completions, **_unused):
            scored = [cache.score(p, c) for p, c in zip(prompts, completions)]
            return [s.rewards[name] for s in scored]
        fn.__name__ = f"reward_{name}"
        return fn

    return [_factory(n) for n in component_names]


# --------------------------------------------------------------------------- #
# In-loop W&B callback                                                        #
# --------------------------------------------------------------------------- #


def _build_wandb_callback(cache, model, tokenizer, env_client,
                          *, sample_every: int, sample_n: int,
                          inloop_every: int, inloop_episodes: int,
                          inloop_max_new_tokens: int):
    """Returns a TrainerCallback that drains ``cache`` after each step."""
    from transformers import TrainerCallback

    from qubit_medic import wandb_utils

    if not wandb_utils.is_available():
        return None

    class _RolloutCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
            entries, keys = cache.drain_step()
            if not entries:
                return
            step = state.global_step

            # 1. Per-component reward stats (mean/std/min/max).
            wandb_utils.log_reward_breakdown(
                [e.rewards for e in entries], step=step, prefix="rl",
            )

            # 2. Parse stats.
            n = max(1, len(entries))
            n_success = sum(1 for e in entries if e.parse_success)
            n_partial = sum(1 for e in entries
                            if not e.parse_success and e.parse_partial)
            wandb_utils.log({
                "rl/parse/success_rate": n_success / n,
                "rl/parse/partial_rate": n_partial / n,
                "rl/parse/failure_rate": (n - n_success - n_partial) / n,
                "rl/parse/sample_count": n,
            }, step=step)

            # 3. Curriculum stats (env-reported snapshot).
            if cache._all_curriculum_stats:
                wandb_utils.log_curriculum(cache._all_curriculum_stats,
                                           step=step, prefix="rl")

            # 4. Env health (cumulative).
            wandb_utils.log({
                "rl/env/episodes_started_total": cache._episodes,
                "rl/env/timeouts_total": cache._timeouts,
                "rl/env/timeout_rate": cache._timeouts / max(1, cache._episodes),
                "rl/env/mean_elapsed_seconds": (
                    sum(e.elapsed for e in entries) / n
                ),
            }, step=step)

            # 5. Curriculum-level distribution within this batch.
            level_counts: dict[str, int] = {}
            for e in entries:
                level_counts[e.curriculum_level] = level_counts.get(
                    e.curriculum_level, 0) + 1
            wandb_utils.log({
                f"rl/batch_level_count/{lvl}": cnt
                for lvl, cnt in level_counts.items()
            }, step=step)

            # 6. Sample-completion table every N steps.
            if sample_every and step > 0 and step % sample_every == 0:
                rows = []
                for (prompt, completion), e in list(zip(keys, entries))[:sample_n]:
                    rows.append({
                        "step": step,
                        "prompt": prompt[:600],
                        "completion": completion[:300],
                        "x_pred": ",".join(map(str, e.x_pred)),
                        "z_pred": ",".join(map(str, e.z_pred)),
                        "logical_correction": e.rewards["logical_correction"],
                        "syndrome_consistency": e.rewards["syndrome_consistency"],
                        "hamming_overlap": e.rewards["hamming_overlap"],
                        "format_compliance": e.rewards["format_compliance"],
                        "pymatching_beat": e.rewards["pymatching_beat"],
                        "total": e.rewards["total"],
                        "parse_success": e.parse_success,
                        "actual_obs_flip": e.actual_flip,
                        "pm_obs_flip": e.pm_flip,
                        "curriculum_level": e.curriculum_level,
                    })
                wandb_utils.log_generation_table(
                    rows, step=step, table_name="rl/generations",
                    columns=[
                        "step", "prompt", "completion", "x_pred", "z_pred",
                        "logical_correction", "syndrome_consistency",
                        "hamming_overlap", "format_compliance",
                        "pymatching_beat", "total", "parse_success",
                        "actual_obs_flip", "pm_obs_flip", "curriculum_level",
                    ],
                )

            # 7. In-loop greedy eval every N steps.
            if inloop_every and step > 0 and step % inloop_every == 0:
                self._run_inloop_eval(step)

        def on_train_end(self, args, state, control, **kwargs):  # noqa: D401
            self._run_inloop_eval(state.global_step, table_name="rl/final_eval")

        def _run_inloop_eval(self, step: int, table_name: str = "rl/inloop_eval"):
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(model)
            except Exception:
                model.eval()  # type: ignore[attr-defined]

            stats = {
                "logical_correction": 0,
                "format_success": 0,
                "format_partial": 0,
                "pymatching_beat": 0,
                "total_reward_sum": 0.0,
            }
            rows = []
            for ep in range(inloop_episodes):
                obs = env_client.reset()
                chat = [{"role": "user", "content": obs.prompt}]
                try:
                    text = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    text = "<|im_start|>user\n" + obs.prompt + "\n<|im_end|>\n<|im_start|>assistant\n"
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                try:
                    out = model.generate(
                        **inputs,
                        max_new_tokens=inloop_max_new_tokens,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    completion = tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                except Exception as exc:  # pragma: no cover
                    completion = f"<gen-error: {exc}>"
                result = env_client.step(raw_response=completion,
                                         episode_id=obs.episode_id)
                rwd = result.info["rewards"]
                action = result.info.get("parsed_action", {})
                stats["logical_correction"] += int(rwd["logical_correction"] >= 0.5)
                stats["format_success"] += int(action.get("parse_success", False))
                stats["format_partial"] += int(rwd["format_compliance"] >= 0.5
                                               and not action.get("parse_success", False))
                stats["pymatching_beat"] += int(rwd["pymatching_beat"] >= 0.5)
                stats["total_reward_sum"] += float(rwd["total"])
                if ep < 4:  # keep tables tiny
                    rows.append({
                        "step": step,
                        "episode": ep,
                        "completion": completion[:300],
                        "logical_correction": rwd["logical_correction"],
                        "total": rwd["total"],
                    })

            n = max(1, inloop_episodes)
            wandb_utils.log({
                "eval/logical_correction_rate": stats["logical_correction"] / n,
                "eval/format_success_rate": stats["format_success"] / n,
                "eval/format_partial_rate": stats["format_partial"] / n,
                "eval/pymatching_beat_rate": stats["pymatching_beat"] / n,
                "eval/mean_total_reward": stats["total_reward_sum"] / n,
                "eval/episodes": n,
            }, step=step)
            if rows:
                wandb_utils.log_generation_table(
                    rows, step=step, table_name=table_name,
                )

            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_training(model)
            except Exception:
                model.train()  # type: ignore[attr-defined]

    return _RolloutCallback()


# --------------------------------------------------------------------------- #
# Dataset of prompts                                                          #
# --------------------------------------------------------------------------- #


def _build_prompt_pool(env_client, n: int):
    """Pre-generate ``n`` syndromes so TRL has a Dataset to iterate over.

    The actual scoring still calls ``env_client.reset()`` on every
    completion (single-step episodes), which is the right behaviour - we
    don't want the trainer to repeatedly score the same syndrome. This
    pool is just TRL's prompt sampling source.
    """
    prompts = []
    for _ in range(n):
        obs = env_client.reset()
        prompts.append({"prompt": obs.prompt, "episode_id": obs.episode_id})
    return prompts


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
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--prompt-pool", type=int, default=512)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=("grpo",))
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--sample-every", type=int, default=None,
                        help="Override config WANDB_LOG_GENERATIONS_EVERY")
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Override config WANDB_SAMPLE_GENERATIONS")
    parser.add_argument("--inloop-eval-every", type=int, default=None,
                        help="Override config WANDB_INLOOP_EVAL_EVERY (0 to disable)")
    parser.add_argument("--inloop-eval-episodes", type=int, default=None,
                        help="Override config WANDB_INLOOP_EVAL_EPISODES")
    parser.add_argument("--no-artifact", action="store_true")
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

    from qubit_medic import wandb_utils
    from qubit_medic.client.client import make_default_client
    from qubit_medic.config import (
        GRPO_CHECKPOINT_EVERY, GRPO_GEN_PER_PROMPT, GRPO_KL_COEF, GRPO_LOG_EVERY,
        GRPO_LR, GRPO_MAX_COMPLETION_LEN, GRPO_MAX_PROMPT_LEN, GRPO_STEPS,
        LORA_ALPHA, LORA_R, LORA_TARGET_MODULES, MODEL_ID, PRIMARY_SEED,
        WANDB_INLOOP_EVAL_EPISODES, WANDB_INLOOP_EVAL_EVERY,
        WANDB_LOG_GENERATIONS_EVERY, WANDB_SAMPLE_GENERATIONS,
    )

    steps = args.steps if args.steps is not None else GRPO_STEPS
    gen_per_prompt = args.gen_per_prompt if args.gen_per_prompt is not None else GRPO_GEN_PER_PROMPT
    lr = args.lr if args.lr is not None else GRPO_LR
    kl_coef = args.kl_coef if args.kl_coef is not None else GRPO_KL_COEF
    max_p = args.max_prompt_len if args.max_prompt_len is not None else GRPO_MAX_PROMPT_LEN
    max_c = args.max_completion_len if args.max_completion_len is not None else GRPO_MAX_COMPLETION_LEN
    seed = args.seed if args.seed is not None else PRIMARY_SEED
    sample_every = args.sample_every if args.sample_every is not None else WANDB_LOG_GENERATIONS_EVERY
    sample_n = args.sample_n if args.sample_n is not None else WANDB_SAMPLE_GENERATIONS
    inloop_every = args.inloop_eval_every if args.inloop_eval_every is not None else WANDB_INLOOP_EVAL_EVERY
    inloop_episodes = args.inloop_eval_episodes if args.inloop_eval_episodes is not None else WANDB_INLOOP_EVAL_EPISODES

    _seed_everything(seed)

    # ---- Env client --------------------------------------------------- #
    env_client = make_default_client()
    print(f"using env client: {type(env_client).__name__}; "
          f"health = {env_client.health()}")

    # ---- W&B init (no-op if unavailable / disabled) -------------------- #
    report_to = wandb_utils.derive_report_to(args.report_to)
    run_name = args.wandb_run_name or wandb_utils.make_run_name("grpo")
    wandb_utils.init_run(
        run_name=run_name,
        job_type="grpo",
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        group=args.wandb_group,
        extra_config={
            "cli": {
                "steps": steps,
                "gen_per_prompt": gen_per_prompt,
                "lr": lr,
                "kl_coef": kl_coef,
                "max_prompt_len": max_p,
                "max_completion_len": max_c,
                "prompt_pool": args.prompt_pool,
                "sample_every": sample_every,
                "sample_n": sample_n,
                "inloop_eval_every": inloop_every,
                "inloop_eval_episodes": inloop_episodes,
                "sft_checkpoint": args.sft_checkpoint,
                "model": args.model,
                "seed": seed,
                "report_to": report_to,
            },
        },
    )

    # ---- Build prompt pool -------------------------------------------- #
    print(f"pre-generating {args.prompt_pool} prompts ...")
    prompts = _build_prompt_pool(env_client, args.prompt_pool)
    dataset = Dataset.from_list(prompts)
    print(f"  built dataset with {len(dataset)} prompts")

    # ---- Load model --------------------------------------------------- #
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

    # ---- TRL GRPOConfig ----------------------------------------------- #
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
        beta=kl_coef,
        logging_steps=GRPO_LOG_EVERY,
        save_steps=GRPO_CHECKPOINT_EVERY,
        save_total_limit=4,
        seed=seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=report_to,
        run_name=run_name,
    )

    # ---- Reward functions + scoring cache ----------------------------- #
    cache = _BatchScoringCache(env_client=env_client)
    reward_fns = _make_reward_fns(cache)

    callbacks = []
    cb = _build_wandb_callback(
        cache, model, tokenizer, env_client,
        sample_every=sample_every, sample_n=sample_n,
        inloop_every=inloop_every, inloop_episodes=inloop_episodes,
        inloop_max_new_tokens=max_c,
    )
    if cb is not None:
        callbacks.append(cb)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fns,
        callbacks=callbacks,
    )

    print(f"running GRPO for {steps} steps...")
    started = time.time()
    train_result = trainer.train()
    elapsed = time.time() - started
    print(f"finished in {elapsed:.1f}s")

    metrics = getattr(train_result, "metrics", {}) or {}
    wandb_utils.update_summary({
        "grpo/wall_seconds": elapsed,
        "grpo/total_episodes": cache._episodes,
        "grpo/total_timeouts": cache._timeouts,
        **{f"grpo/final/{k}": v for k, v in metrics.items()
           if isinstance(v, (int, float))},
    })

    print(f"saving adapters to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    if not args.no_artifact:
        wandb_utils.log_artifact(
            args.output,
            name=f"grpo-adapter-{run_name}",
            artifact_type="model",
            description="GRPO-trained Qwen2.5-3B + LoRA adapter (Qubit-Medic).",
        )

    wandb_utils.finish_run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
