"""scripts/train_grpo.py - GRPO RL phase (2026-04 spec rewrite).

Loads the SFT-warm-started LoRA adapter at
``checkpoints/sft_warmup/checkpoint-50`` on top of the 4-bit NF4 quantised
``unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit`` base, connects to the
OpenEnv server (local or remote via ``QUBIT_MEDIC_URL``), and runs TRL's
:class:`GRPOTrainer` for 1,500 steps with diversity-focused rollout
sampling (temperature=1.2, top_p=0.95, top_k=50, repetition_penalty=1.1)
and a weighted 5-component reward bounded to [0, 1].

Why diversity-focused sampling
------------------------------
The first GRPO attempt (temperature=0.7) collapsed inside 100 steps to a
constant ``X_ERRORS=[] Z_ERRORS=[]`` policy: every group of 4 generations
was byte-identical, so within-group reward variance was zero and the
GRPO advantage was exactly zero - no gradient. The new sampler defaults
broaden per-token entropy enough to keep within-group variance positive,
which is what GRPO needs to learn anything.

Major spec features wired up here
---------------------------------
* ``_diversity_preflight`` - 5 prompts x 8 completions at T=1.2; abort if
  fewer than 3 prompts hit >=3 unique completions. The model is too
  collapsed for GRPO to recover.
* Frozen 200-syndrome eval set seeded ``4284`` (matches SFT validation
  seed). Cached to ``data/grpo_validation.jsonl`` so reruns and offline
  inspection see the same prompts.
* Tier-1 training metrics (every 5 steps): total_reward_mean,
  reward_std_within_group, completion_uniqueness, advantage_mean_abs,
  kl_divergence, grad_norm, policy_loss, learning_rate.
* Tier-2 eval metrics (every 100 steps, greedy at T=0): logical
  correction rate, pymatching beat rate, format compliance, exact-match
  pymatching, hard-syndrome (>=2 errors) LCR, syndrome consistency,
  avg_completion_length, output_diversity at T=1.0.
* Tier-3 (every eval): per-round logical-error rate at d=3 p=0.001 plus
  log10 transform.
* Sample-completion table every 50 steps: 5 random eval prompts, the 4
  rollouts each, per-component rewards, parsed action.
* Anti-hacking: 30s per-episode timeout (server-side), reward bounds
  enforced both pre-multiply and post-sum, mode-collapse inspection
  every 100 steps that auto-raises temperature by 0.2 if >7 of the
  last 10 prompts produced 4 byte-identical generations.
* Wall-clock cap: 13h. Saves+exits cleanly if exceeded.
* Best-checkpoint tracking: writes ``output/best/`` whenever a new best
  ``eval/total_reward_mean`` is observed. Final state always saves to
  ``output/final/`` regardless of rank.
* Decision rules (warnings only, no auto-fix): step-50 reward variance
  floor, step-500 pymatching-beat sanity, format-compliance floor, and
  3-consecutive-log grad-norm runaway alarm.

Usage::

    python -m scripts.train_grpo \
        --sft-checkpoint checkpoints/sft_warmup/checkpoint-50 \
        --output checkpoints/grpo \
        --report-to wandb
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# --------------------------------------------------------------------------- #
# Pre-flight: detect the unsloth / unsloth_zoo signature skew that crashes    #
# GRPO at step 0 with a misleading TypeError.                                 #
# --------------------------------------------------------------------------- #
#
# unsloth==2025.11.1's GRPO trainer template calls
#     grpo_accumulated_loss(..., old_hidden_states=..., ref_hidden_states=...)
# but unsloth_zoo>=2026.x renamed those positional args to old_logps / ref_logps
# with no compat shim. Pip's resolver (with the unpinned `unsloth` line in
# requirements-train.txt) silently couples the two: it picks
#     unsloth==2025.11.1  +  unsloth_zoo==2026.4.9
# and that pair crashes at the first training step with:
#     TypeError: grpo_accumulated_loss() missing 2 required positional
#     arguments: 'old_logps' and 'ref_logps'
#
# SFT does not exercise this code path, so SFT finishes cleanly first, the
# checkpoint gets saved, and only then GRPO blows up - wasting the whole SFT
# run. This guard runs in well under a second, before any GPU work, and
# prints the exact pip command to fix it instead of the cryptic TypeError.
# --------------------------------------------------------------------------- #


def _assert_grpo_signature_compatible() -> None:
    """Abort early if the installed unsloth_zoo signature does not match
    the call pattern baked into the installed unsloth.
    """
    try:
        import unsloth  # noqa: F401  (force the patches to apply first)
        import unsloth_zoo
        from unsloth_zoo.rl_replacements import grpo_accumulated_loss
    except Exception as exc:
        print(f"[grpo-guard] WARNING: could not introspect unsloth_zoo "
              f"({exc!r}); skipping signature check.", file=sys.stderr)
        return

    params = list(inspect.signature(grpo_accumulated_loss).parameters.keys())
    has_hidden = "old_hidden_states" in params and "ref_hidden_states" in params
    has_logps = "old_logps" in params and "ref_logps" in params

    # The unsloth in this repo is pinned to the 2025.11.x lineage (matches
    # what SFT just used). That lineage calls with old_hidden_states= /
    # ref_hidden_states=. If unsloth_zoo has those names, we are fine.
    if has_hidden:
        return

    unsloth_ver = getattr(unsloth, "__version__", "?")
    zoo_ver = getattr(unsloth_zoo, "__version__", "?")
    have_logps_only = has_logps and not has_hidden

    msg = [
        "",
        "=" * 78,
        "[grpo-guard] FATAL: unsloth / unsloth_zoo signature mismatch detected.",
        "=" * 78,
        f"  unsloth     == {unsloth_ver}",
        f"  unsloth_zoo == {zoo_ver}",
        f"  grpo_accumulated_loss parameters: {params}",
        "",
        "  unsloth (this version) calls grpo_accumulated_loss with",
        "    old_hidden_states=... , ref_hidden_states=...",
        "  but the installed unsloth_zoo expects",
        "    old_logps=... , ref_logps=...",
        "  as required positional arguments." if have_logps_only else
        "  but the installed unsloth_zoo signature does not contain those names.",
        "",
        "  Without this fix, GRPO will crash at step 0 with:",
        "    TypeError: grpo_accumulated_loss() missing 2 required positional",
        "    arguments: 'old_logps' and 'ref_logps'",
        "",
        "  Fix on Colab (one-liner):",
        "    pip install --no-deps --force-reinstall unsloth_zoo==2025.11.1 \\",
        "      && rm -rf unsloth_compiled_cache",
        "",
        "  Then re-run:",
        "    python -m scripts.train_grpo --sft-checkpoint "
        "checkpoints/sft_warmup/checkpoint-50 \\",
        "        --output checkpoints/grpo",
        "=" * 78,
        "",
    ]
    raise SystemExit("\n".join(msg))


def _wipe_stale_grpo_cache() -> None:
    """Remove unsloth_compiled_cache/UnslothGRPOTrainer.py if present.

    The cache file is regenerated automatically by unsloth on the next
    GRPO import using the *currently installed* unsloth_zoo source, so
    deleting it is safe and is the only way to recover after fixing
    a previously-mismatched install.
    """
    cache_file = Path("unsloth_compiled_cache") / "UnslothGRPOTrainer.py"
    if cache_file.exists():
        print(f"[grpo-guard] removing stale {cache_file} so it regenerates "
              f"against the current unsloth_zoo install")
        try:
            cache_file.unlink()
        except OSError as exc:
            print(f"[grpo-guard] WARNING: failed to remove {cache_file}: "
                  f"{exc!r}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Per-batch scoring cache + reward bounds enforcement                         #
# --------------------------------------------------------------------------- #
#
# The original implementation called the env 5 times per (prompt, completion)
# - once per reward function. We fix that with a single (prompt, completion)
# -> breakdown cache keyed inside one GRPO step, AND we apply the spec's
# weighted-sum + [0, 1] clip in one place so every reward function returns
# a number that's already correctly weighted.
# --------------------------------------------------------------------------- #


@dataclass
class _ScoredCompletion:
    """One scored (prompt, completion) pair, keyed by the env episode."""
    rewards: dict          # raw per-component rewards from the env (in [0, 1])
    weighted_total: float  # weighted sum, clipped to [0, 1]
    parse_success: bool
    parse_partial: bool
    x_pred: list
    z_pred: list
    actual_flip: int
    pm_flip: int
    elapsed: float
    timed_out: bool
    curriculum_level: str
    bounds_violations: int  # >0 if env returned a component outside [0, 1]


@dataclass
class _BatchScoringCache:
    """Caches per-(prompt, completion) scores within one GRPO step."""
    env_client: object
    reward_weights: dict
    _cache: dict = field(default_factory=dict)
    _step_keys: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _all_curriculum_stats: dict = field(default_factory=dict)
    _episodes: int = 0
    _timeouts: int = 0
    _bounds_violations: int = 0

    def _enforce_bounds(self, name: str, val: float) -> tuple[float, bool]:
        """Clip a reward component to [0, 1]; flag if it was outside."""
        v = float(val)
        if v < 0.0 or v > 1.0:
            return max(0.0, min(1.0, v)), True
        return v, False

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

        # Apply spec weights + [0, 1] bounds enforcement.
        raw = info.get("rewards", {}) or {}
        violations = 0
        weighted_sum = 0.0
        bounded_components: dict = {}
        for name, weight in self.reward_weights.items():
            v, was_oob = self._enforce_bounds(name, raw.get(name, 0.0))
            bounded_components[name] = v
            weighted_sum += weight * v
            if was_oob:
                violations += 1
        # Clip weighted sum to [0, 1] (already in range when components
        # are; defensive against weights that don't sum to 1.0).
        weighted_total = max(0.0, min(1.0, weighted_sum))

        # Preserve env's "total" alongside our weighted total so downstream
        # wandb log_reward_breakdown still works.
        bounded_components["total"] = weighted_total

        scored = _ScoredCompletion(
            rewards=bounded_components,
            weighted_total=weighted_total,
            parse_success=bool(action.get("parse_success", False)),
            parse_partial=False,
            x_pred=list(action.get("x_error_qubits", []) or []),
            z_pred=list(action.get("z_error_qubits", []) or []),
            actual_flip=int(info.get("actual_observable_flip", 0)),
            pm_flip=int(info.get("pymatching_observable_pred", 0)),
            elapsed=float(info.get("elapsed_seconds", 0.0)),
            timed_out=bool(info.get("timed_out", False)),
            curriculum_level=str(getattr(obs, "curriculum_level", "")),
            bounds_violations=violations,
        )
        with self._lock:
            self._cache[key] = scored
            self._step_keys.append(key)
            self._all_curriculum_stats = info.get("curriculum_stats", {}) or {}
            self._episodes += 1
            if scored.timed_out:
                self._timeouts += 1
            if violations:
                self._bounds_violations += violations
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
# Reward function factory                                                     #
# --------------------------------------------------------------------------- #
#
# Spec: total reward = sum of weighted components, clipped to [0, 1].
# Implementation: the cache returns a per-completion weighted_total in
# [0, 1]. We expose ONE TRL reward function that returns that bounded
# total, plus zero-weight per-component observers so wandb gets per-
# component traces without altering the policy gradient.
# --------------------------------------------------------------------------- #


_REWARD_COMPONENTS = (
    "logical_correction",
    "hamming_overlap",
    "syndrome_consistency",
    "format_compliance",
    "pymatching_beat",
)


def _make_reward_fns(cache: _BatchScoringCache):
    def total_fn(prompts, completions, **_unused):
        scored = [cache.score(p, c) for p, c in zip(prompts, completions)]
        return [s.weighted_total for s in scored]
    total_fn.__name__ = "reward_total"

    observers: list = []
    for name in _REWARD_COMPONENTS:
        def _factory(component_name: str):
            def fn(prompts, completions, **_unused):
                scored = [cache.score(p, c) for p, c in zip(prompts, completions)]
                return [s.rewards.get(component_name, 0.0) for s in scored]
            fn.__name__ = f"reward_obs_{component_name}"
            return fn
        observers.append(_factory(name))

    return [total_fn] + observers


# --------------------------------------------------------------------------- #
# Frozen eval set: 200 syndromes seeded GRPO_VAL_SEED.                        #
# --------------------------------------------------------------------------- #
#
# We snapshot the 200 prompts to data/grpo_validation.jsonl on first run so
# reruns hit byte-identical syndromes, and so the file can be inspected /
# diffed offline. If the file already exists with >= n rows, we trust it.
# --------------------------------------------------------------------------- #


def _load_or_build_eval_set(env_client, *, seed: int, n: int, path: str) -> list[dict]:
    p = Path(path)
    if p.exists():
        rows: list[dict] = []
        with p.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if len(rows) >= n:
            print(f"[grpo-eval] reusing cached eval set: {p} ({len(rows)} rows)")
            return rows[:n]
        print(f"[grpo-eval] cached eval set at {p} has {len(rows)} < {n} rows; "
              f"regenerating")

    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[grpo-eval] building frozen eval set seed={seed} n={n} -> {p}")
    cur_seed = seed
    for _ in range(n):
        obs = env_client.reset(seed=cur_seed)
        rows.append({
            "prompt": obs.prompt,
            "episode_id": int(obs.episode_id),
            "curriculum_level": str(getattr(obs, "curriculum_level", "")),
            "distance": int(getattr(obs, "distance", 0)),
            "rounds": int(getattr(obs, "rounds", 0)),
            "p": float(getattr(obs, "p", 0.0)),
            "syndrome_bits": list(getattr(obs, "syndrome_bits", []) or []),
            "seed": cur_seed,
        })
        cur_seed += 1  # deterministic, reproducible
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[grpo-eval] wrote {len(rows)} eval rows to {p}")
    return rows


# --------------------------------------------------------------------------- #
# Diversity preflight                                                         #
# --------------------------------------------------------------------------- #


def _diversity_preflight(model, tokenizer, *, val_path: str, n_prompts: int = 5,
                         n_samples_per_prompt: int = 8, temperature: float = 1.2,
                         min_unique: int = 3, min_passing: int = 3,
                         max_new_tokens: int = 50) -> bool:
    """Generate ``n_samples_per_prompt`` completions per prompt at high temp.

    Returns True iff at least ``min_passing`` of the prompts produced
    >= ``min_unique`` unique completions (byte-equal under skip-special-tokens
    decoding). False -> the model is collapsed past the point where GRPO
    can recover, so we should refuse to start training.
    """
    import torch

    src = Path(val_path)
    if not src.exists():
        print(f"[grpo-preflight] WARNING: {val_path} not found; "
              f"skipping diversity preflight")
        return True  # don't block on missing file

    rows: list[dict] = []
    with src.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if len(rows) < n_prompts:
        print(f"[grpo-preflight] WARNING: only {len(rows)} validation rows "
              f"available, need {n_prompts}; using all")
        n_prompts = len(rows)

    # Mix of trivial (no errors) and non-trivial (errors present), so the
    # diversity probe sees both regimes the model has to handle.
    rng = random.Random(0)
    trivial = [r for r in rows if not r.get("had_errors")]
    non_trivial = [r for r in rows if r.get("had_errors")]
    rng.shuffle(trivial)
    rng.shuffle(non_trivial)
    half = max(1, n_prompts // 2)
    chosen = (non_trivial[:half] + trivial[:n_prompts - half])[:n_prompts]
    if not chosen:
        chosen = rows[:n_prompts]

    print(f"[grpo-preflight] probing diversity at T={temperature} on "
          f"{len(chosen)} prompts x {n_samples_per_prompt} samples each")

    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()
    passing = 0
    per_prompt_unique: list[int] = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for i, row in enumerate(chosen):
        prompt = row["prompt"]
        try:
            chat = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = ("<|im_start|>user\n" + prompt
                    + "\n<|im_end|>\n<|im_start|>assistant\n")
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        completions: list[str] = []
        for _ in range(n_samples_per_prompt):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_id,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            completions.append(txt)
        unique = len(set(completions))
        per_prompt_unique.append(unique)
        verdict = "PASS" if unique >= min_unique else "FAIL"
        print(f"[grpo-preflight]   prompt {i}: {unique}/{n_samples_per_prompt} "
              f"unique  [{verdict}]  examples={completions[:2]!r}")
        if unique >= min_unique:
            passing += 1

    overall = passing >= min_passing
    print(f"[grpo-preflight] {passing}/{len(chosen)} prompts passed "
          f"(threshold: >= {min_passing}). per_prompt_unique={per_prompt_unique}")

    if not overall:
        print("=" * 78)
        print("[grpo-preflight] PRE-FLIGHT FAILED - model is too collapsed; "
              "redo SFT with regularization before launching GRPO")
        print("=" * 78)
    return overall


# --------------------------------------------------------------------------- #
# In-loop W&B callback (tier-1 + tier-2 + tier-3 + sample table + safeguards) #
# --------------------------------------------------------------------------- #


def _build_wandb_callback(cache, model, tokenizer, env_client, eval_rows,
                          *, sample_every: int, sample_n: int,
                          inloop_every: int,
                          inloop_max_new_tokens: int,
                          kl_alarm: float,
                          inspection_every: int, inspection_sample_n: int,
                          inspection_collapse_threshold: int,
                          temp_bump_on_collapse: float,
                          best_dir: Path, output_dir: Path,
                          wall_seconds: float,
                          decision_thresholds: dict):
    from transformers import TrainerCallback

    from qubit_medic import wandb_utils

    if not wandb_utils.is_available():
        return None

    started_at = time.time()

    # Rolling cache for the inspection hook: we record (group_unique_count)
    # per prompt as it streams in, and at every inspection_every-step
    # boundary look at the most recent inspection_sample_n entries.
    recent_uniques = deque(maxlen=max(inspection_sample_n, 16))

    grad_norm_run = deque(maxlen=decision_thresholds["grad_norm_run_len"])
    state = {
        "best_total_reward": float("-inf"),
        "best_step": -1,
        "wall_exceeded": False,
        "step50_warned": False,
        "step500_warned": False,
        "format_warned_at": -1,
        "grad_norm_warned_at": -1,
        "beat_rate_history": [],
    }

    class _RolloutCallback(TrainerCallback):

        # ------------------------------------------------------------------ #
        # Per-step instrumentation                                           #
        # ------------------------------------------------------------------ #
        def on_step_end(self, args, state_, control, **kwargs):  # noqa: D401
            entries, keys = cache.drain_step()
            if not entries:
                return
            step = state_.global_step

            # Group entries by prompt so we can compute within-group stats.
            groups: list[list[_ScoredCompletion]] = []
            current_prompt = None
            current: list[_ScoredCompletion] = []
            for (p, _), e in zip(keys, entries):
                if p != current_prompt and current:
                    groups.append(current)
                    current = []
                current_prompt = p
                current.append(e)
            if current:
                groups.append(current)

            # ----- Tier-1 metrics ----- #
            totals = [e.weighted_total for e in entries]
            if not totals:
                return
            mean_total = sum(totals) / len(totals)

            within_stds: list[float] = []
            uniques: list[int] = []
            for grp in groups:
                if len(grp) < 2:
                    within_stds.append(0.0)
                    uniques.append(1)
                    continue
                vals = [e.weighted_total for e in grp]
                mu = sum(vals) / len(vals)
                var = sum((v - mu) ** 2 for v in vals) / len(vals)
                within_stds.append(var ** 0.5)
                key_set = {(tuple(e.x_pred), tuple(e.z_pred)) for e in grp}
                uniques.append(len(key_set))
            mean_within_std = sum(within_stds) / max(1, len(within_stds))
            mean_unique = sum(uniques) / max(1, len(uniques))

            # GRPO advantage (recomputed locally for the log only).
            adv_abs: list[float] = []
            for grp in groups:
                if len(grp) < 2:
                    continue
                vals = [e.weighted_total for e in grp]
                mu = sum(vals) / len(vals)
                var = sum((v - mu) ** 2 for v in vals) / len(vals)
                std = max((var ** 0.5), 1e-4)
                adv_abs.extend(abs((v - mu) / std) for v in vals)
            mean_adv_abs = sum(adv_abs) / max(1, len(adv_abs))

            wandb_utils.log({
                "train/total_reward_mean": mean_total,
                "train/reward_std_within_group": mean_within_std,
                "train/completion_uniqueness": mean_unique,
                "train/advantage_mean_abs": mean_adv_abs,
                "train/global_step": step,
            }, step=step)

            wandb_utils.log_reward_breakdown(
                [e.rewards for e in entries], step=step, prefix="train",
            )

            wandb_utils.log({
                "train/reward_bounds_violations_total": cache._bounds_violations,
                "train/env_episodes_total": cache._episodes,
                "train/env_timeouts_total": cache._timeouts,
            }, step=step)

            # ----- Decision rule: step 50 within-group variance ----- #
            if (not state["step50_warned"]
                    and step >= decision_thresholds["reward_std_check_step"]):
                if mean_within_std < decision_thresholds["reward_std_floor"]:
                    print(f"\n[grpo-decision] CRITICAL @ step {step}: "
                          f"train/reward_std_within_group={mean_within_std:.4f} "
                          f"< {decision_thresholds['reward_std_floor']}. The "
                          f"within-group reward std has collapsed; GRPO has "
                          f"effectively zero advantage signal. Pausing for "
                          f"manual review (warning only - no auto-action).")
                    wandb_utils.log({
                        "alarms/reward_std_collapse": 1.0,
                        "alarms/reward_std_value": mean_within_std,
                    }, step=step)
                state["step50_warned"] = True

            # ----- Mode-collapse inspection hook ----- #
            for u in uniques:
                recent_uniques.append(u)
            if (inspection_every and step > 0
                    and step % inspection_every == 0
                    and len(recent_uniques) >= inspection_sample_n):
                last = list(recent_uniques)[-inspection_sample_n:]
                collapsed_count = sum(1 for u in last if u == 1)
                if collapsed_count > inspection_collapse_threshold:
                    cur_temp = float(getattr(args, "temperature", 1.2))
                    new_temp = cur_temp + temp_bump_on_collapse
                    print(f"\n[grpo-inspection] WARN @ step {step}: "
                          f"{collapsed_count}/{inspection_sample_n} of the "
                          f"most recent prompts had ALL 4 generations identical. "
                          f"Bumping rollout temperature {cur_temp:.2f} "
                          f"-> {new_temp:.2f}.")
                    try:
                        args.temperature = new_temp
                    except Exception as exc:
                        print(f"[grpo-inspection] could not patch temperature "
                              f"on TRL args: {exc!r}")
                    wandb_utils.log({
                        "alarms/mode_collapse_count": collapsed_count,
                        "train/temperature_after_bump": new_temp,
                    }, step=step)

            # ----- Sample-completion table every sample_every steps ----- #
            if sample_every and step > 0 and step % sample_every == 0:
                rows_out = []
                # First sample_n unique prompts in this batch; emit a row per
                # generation (so the W&B table has gen_idx as a column).
                chosen_groups: list[tuple[str, list[_ScoredCompletion]]] = []
                seen_prompts: set = set()
                for (p, _), e in zip(keys, entries):
                    if p in seen_prompts:
                        for q, grp in chosen_groups:
                            if q == p:
                                grp.append(e)
                                break
                        continue
                    if len(chosen_groups) >= sample_n:
                        continue
                    chosen_groups.append((p, [e]))
                    seen_prompts.add(p)
                for prompt, grp in chosen_groups:
                    for gi, e in enumerate(grp[:4]):
                        rows_out.append({
                            "step": step,
                            "prompt": prompt[:600],
                            "gen_idx": gi,
                            "x_pred": ",".join(map(str, e.x_pred)),
                            "z_pred": ",".join(map(str, e.z_pred)),
                            "logical_correction":
                                e.rewards.get("logical_correction", 0.0),
                            "syndrome_consistency":
                                e.rewards.get("syndrome_consistency", 0.0),
                            "hamming_overlap":
                                e.rewards.get("hamming_overlap", 0.0),
                            "format_compliance":
                                e.rewards.get("format_compliance", 0.0),
                            "pymatching_beat":
                                e.rewards.get("pymatching_beat", 0.0),
                            "weighted_total": e.weighted_total,
                            "parse_success": e.parse_success,
                            "actual_obs_flip": e.actual_flip,
                            "pm_obs_flip": e.pm_flip,
                            "curriculum_level": e.curriculum_level,
                        })
                if rows_out:
                    wandb_utils.log_generation_table(
                        rows_out, step=step, table_name="rl/generations",
                        columns=[
                            "step", "prompt", "gen_idx", "x_pred", "z_pred",
                            "logical_correction", "syndrome_consistency",
                            "hamming_overlap", "format_compliance",
                            "pymatching_beat", "weighted_total",
                            "parse_success", "actual_obs_flip", "pm_obs_flip",
                            "curriculum_level",
                        ],
                    )

            # ----- Wall-clock cap ----- #
            elapsed = time.time() - started_at
            if elapsed > wall_seconds and not state["wall_exceeded"]:
                state["wall_exceeded"] = True
                print(f"\n[grpo-walltime] wall-clock cap hit at step {step} "
                      f"({elapsed:.0f}s > {wall_seconds:.0f}s). "
                      f"Saving and exiting.")
                try:
                    control.should_save = True
                    control.should_training_stop = True
                except Exception:
                    pass
                wandb_utils.log({
                    "alarms/wall_exceeded": 1.0,
                    "alarms/wall_seconds_at_cap": elapsed,
                }, step=step)

            # ----- Tier-2 + tier-3 eval ----- #
            if inloop_every and step > 0 and step % inloop_every == 0:
                self._run_inloop_eval(step)

        def on_log(self, args, state_, control, logs=None, **kwargs):  # noqa: D401
            if not logs:
                return
            step = state_.global_step

            # Tier-1: surface train/* metrics that TRL itself produces.
            extra: dict = {}
            for src_key, dst_key in [
                ("kl", "train/kl_divergence"),
                ("train/kl_divergence", "train/kl_divergence"),
                ("grad_norm", "train/grad_norm"),
                ("loss", "train/policy_loss"),
                ("learning_rate", "train/learning_rate"),
            ]:
                if src_key in logs:
                    try:
                        extra[dst_key] = float(logs[src_key])
                    except (TypeError, ValueError):
                        pass
            if extra:
                wandb_utils.log(extra, step=step)

            # KL alarm.
            kl = logs.get("kl") or logs.get("train/kl_divergence")
            if kl is not None:
                try:
                    kl_v = float(kl)
                except (TypeError, ValueError):
                    kl_v = None
                if kl_v is not None and kl_v > kl_alarm:
                    wandb_utils.log({
                        "alarms/kl_alarm": 1.0,
                        "alarms/kl_alarm_value": kl_v,
                    }, step=step)
                    print(f"[grpo][step {step}] KL ALARM: {kl_v:.3f} "
                          f"> {kl_alarm:.3f} - inspect generations.")

            # Decision rule: grad_norm > ceil for N consecutive logs.
            gn = logs.get("grad_norm")
            if gn is not None:
                try:
                    gn_v = float(gn)
                except (TypeError, ValueError):
                    gn_v = None
                if gn_v is not None:
                    grad_norm_run.append(gn_v)
                    ceil = decision_thresholds["grad_norm_ceil"]
                    run_len = decision_thresholds["grad_norm_run_len"]
                    if (len(grad_norm_run) >= run_len
                            and all(x > ceil for x in grad_norm_run)
                            and step != state["grad_norm_warned_at"]):
                        print(f"\n[grpo-decision] CRITICAL @ step {step}: "
                              f"train/grad_norm > {ceil} for {run_len} "
                              f"consecutive logs ({list(grad_norm_run)}). "
                              f"Recommend reducing LR (warning only - no "
                              f"auto-action).")
                        wandb_utils.log({
                            "alarms/grad_norm_runaway": 1.0,
                            "alarms/grad_norm_value": gn_v,
                        }, step=step)
                        state["grad_norm_warned_at"] = step

        def on_train_end(self, args, state_, control, **kwargs):  # noqa: D401
            self._run_inloop_eval(state_.global_step, table_name="rl/final_eval")

        # ------------------------------------------------------------------ #
        # Tier-2 / tier-3 eval (greedy, T=0)                                 #
        # ------------------------------------------------------------------ #
        def _run_inloop_eval(self, step: int, table_name: str = "rl/inloop_eval"):
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(model)
            except Exception:
                model.eval()  # type: ignore[attr-defined]

            n = len(eval_rows)
            stats = {
                "logical_correction": 0,
                "format_success": 0,
                "format_partial": 0,
                "pymatching_beat": 0,
                "syndrome_consistency_pass": 0,
                "exact_match_pymatching": 0,
                "total_reward_sum": 0.0,
                "completion_len_sum": 0,
                "hard_lcr_num": 0,
                "hard_lcr_den": 0,
                "ler_d3_p001_logical_errors": 0,
                "ler_d3_p001_total": 0,
                "ler_d3_p001_rounds": 0,
            }
            preview_rows = []
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            import torch
            from qubit_medic.config import REWARD_WEIGHTS

            for ep_idx, row in enumerate(eval_rows):
                prompt = row["prompt"]
                episode_id = int(row.get("episode_id", -1))
                try:
                    chat = [{"role": "user", "content": prompt}]
                    text = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    text = ("<|im_start|>user\n" + prompt
                            + "\n<|im_end|>\n<|im_start|>assistant\n")
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                try:
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=inloop_max_new_tokens,
                            do_sample=False,  # greedy at T=0 per spec
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=pad_id,
                        )
                    gen_ids = out[0][inputs["input_ids"].shape[1]:]
                    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    n_tokens = int(gen_ids.shape[0])
                except Exception as exc:  # pragma: no cover
                    completion = f"<gen-error: {exc}>"
                    n_tokens = 0

                # Score against the env. If episode_id has TTL'd we fall back
                # to a fresh reset so the run continues, but log nothing
                # special - the metric arithmetic is still correct.
                try:
                    result = env_client.step(raw_response=completion,
                                             episode_id=episode_id)
                except Exception:
                    obs2 = env_client.reset(seed=row.get("seed"))
                    result = env_client.step(raw_response=completion,
                                             episode_id=obs2.episode_id)

                rwd = result.info.get("rewards", {}) or {}
                action = result.info.get("parsed_action", {}) or {}
                actual = int(result.info.get("actual_observable_flip", 0))
                pm_pred = int(result.info.get("pymatching_observable_pred", 0))
                we_correct = float(rwd.get("logical_correction", 0.0)) >= 0.5
                pm_correct = (pm_pred == actual)

                stats["logical_correction"] += int(we_correct)
                stats["format_success"] += int(action.get("parse_success", False))
                stats["format_partial"] += int(
                    float(rwd.get("format_compliance", 0.0)) >= 0.5
                    and not action.get("parse_success", False)
                )
                stats["pymatching_beat"] += int(
                    float(rwd.get("pymatching_beat", 0.0)) >= 0.5)
                stats["syndrome_consistency_pass"] += int(
                    float(rwd.get("syndrome_consistency", 0.0)) >= 0.999)
                weighted = sum(
                    weight * max(0.0, min(1.0, float(rwd.get(name, 0.0))))
                    for name, weight in REWARD_WEIGHTS.items()
                )
                stats["total_reward_sum"] += max(0.0, min(1.0, weighted))
                stats["completion_len_sum"] += n_tokens

                pm_x = sorted(set(map(int,
                    result.info.get("pymatching_x_errors", []) or [])))
                pm_z = sorted(set(map(int,
                    result.info.get("pymatching_z_errors", []) or [])))
                our_x = sorted(set(map(int,
                    action.get("x_error_qubits", []) or [])))
                our_z = sorted(set(map(int,
                    action.get("z_error_qubits", []) or [])))
                if (action.get("parse_success", False)
                        and pm_x == our_x and pm_z == our_z):
                    stats["exact_match_pymatching"] += 1

                # Hard syndrome: >=2 stabilizers fired (anti-hacking spec
                # forbids exposing true_x/true_z, so we use the syndrome
                # bit count from the cached eval row as the proxy).
                n_active = sum(1 for b in row.get("syndrome_bits", []) if int(b))
                if n_active >= 2:
                    stats["hard_lcr_den"] += 1
                    stats["hard_lcr_num"] += int(we_correct)

                # tier-3: per-round LER for d=3 / p=0.001 only.
                d = int(row.get("distance", 0))
                rnds = max(1, int(row.get("rounds", 0)))
                if d == 3 and abs(float(row.get("p", 0.0)) - 0.001) < 1e-6:
                    stats["ler_d3_p001_total"] += 1
                    stats["ler_d3_p001_rounds"] = rnds
                    if not we_correct:
                        stats["ler_d3_p001_logical_errors"] += 1

                if ep_idx < 4:
                    preview_rows.append({
                        "step": step,
                        "episode": ep_idx,
                        "completion": completion[:300],
                        "logical_correction": rwd.get("logical_correction", 0.0),
                        "syndrome_consistency": rwd.get("syndrome_consistency", 0.0),
                        "format_compliance": rwd.get("format_compliance", 0.0),
                        "pymatching_beat": rwd.get("pymatching_beat", 0.0),
                        "weighted_total": weighted,
                    })

            denom = max(1, n)
            lcr = stats["logical_correction"] / denom
            beat_rate = stats["pymatching_beat"] / denom
            fmt_compliance = stats["format_success"] / denom
            hard_lcr = (stats["hard_lcr_num"] / max(1, stats["hard_lcr_den"])
                        if stats["hard_lcr_den"] else 0.0)
            sync_consistency_rate = stats["syndrome_consistency_pass"] / denom
            avg_completion_len = stats["completion_len_sum"] / denom
            mean_total_reward = stats["total_reward_sum"] / denom
            exact_match = stats["exact_match_pymatching"] / denom

            # Tier-3 LER per round, log10.
            ler_per_round = None
            ler_log10 = None
            if stats["ler_d3_p001_total"] > 0:
                p_logical = (stats["ler_d3_p001_logical_errors"]
                             / stats["ler_d3_p001_total"])
                rounds = max(1, stats["ler_d3_p001_rounds"])
                # Per-round LER: 1 - (1 - p_logical)^(1/rounds).
                ler_per_round = 1.0 - (1.0 - max(0.0, min(1.0, p_logical))) ** (1.0 / rounds)
                if ler_per_round > 0:
                    import math
                    ler_log10 = math.log10(max(ler_per_round, 1e-12))

            # Tier-2 output diversity probe at T=1.0 (8 samples per prompt
            # on a small subset to keep eval fast).
            div_probe_n = min(8, len(eval_rows))
            div_samples = 8
            unique_counts: list[int] = []
            for row in eval_rows[:div_probe_n]:
                prompt = row["prompt"]
                try:
                    chat = [{"role": "user", "content": prompt}]
                    text = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    text = ("<|im_start|>user\n" + prompt
                            + "\n<|im_end|>\n<|im_start|>assistant\n")
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outs = []
                for _ in range(div_samples):
                    try:
                        with torch.no_grad():
                            out = model.generate(
                                **inputs,
                                max_new_tokens=inloop_max_new_tokens,
                                do_sample=True,
                                temperature=1.0,
                                top_p=0.95,
                                top_k=50,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=pad_id,
                            )
                        gen = tokenizer.decode(
                            out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        ).strip()
                    except Exception:
                        gen = ""
                    outs.append(gen)
                unique_counts.append(len(set(outs)))
            output_diversity_t1 = (sum(unique_counts) / max(1, len(unique_counts))
                                   if unique_counts else 0.0)

            eval_metrics = {
                "eval/logical_correction_rate": lcr,
                "eval/pymatching_beat_rate": beat_rate,
                "eval/format_compliance": fmt_compliance,
                "eval/exact_match_pymatching": exact_match,
                "eval/hard_syndrome_lcr": hard_lcr,
                "eval/syndrome_consistency_rate": sync_consistency_rate,
                "eval/avg_completion_length": avg_completion_len,
                "eval/output_diversity_temp_1": output_diversity_t1,
                "eval/total_reward_mean": mean_total_reward,
                "eval/episodes": denom,
            }
            if ler_per_round is not None:
                eval_metrics["eval/ler_per_round_d3_p001"] = ler_per_round
                if ler_log10 is not None:
                    eval_metrics["eval/ler_per_round_log10"] = ler_log10

            print(f"[grpo][eval@{step}] " + ", ".join(
                f"{k.split('/')[-1]}={v:.4f}" if isinstance(v, float)
                else f"{k.split('/')[-1]}={v}" for k, v in eval_metrics.items()
            ))
            wandb_utils.log(eval_metrics, step=step)
            if preview_rows:
                wandb_utils.log_generation_table(
                    preview_rows, step=step, table_name=table_name,
                )

            # Decision rule: step-500 pymatching_beat sanity.
            state["beat_rate_history"].append(beat_rate)
            if len(state["beat_rate_history"]) > 5:
                state["beat_rate_history"] = state["beat_rate_history"][-5:]
            if (not state["step500_warned"]
                    and step >= decision_thresholds["beat_rate_check_step"]
                    and len(state["beat_rate_history"]) >= 5
                    and all(b == 0 for b in state["beat_rate_history"])):
                print(f"\n[grpo-decision] WARN @ step {step}: "
                      f"eval/pymatching_beat_rate has been 0.0 across the last "
                      f"5 evals. The model is never finding syndromes where "
                      f"PyMatching fails - consider increasing the "
                      f"pymatching_beat reward weight (warning only).")
                wandb_utils.log({"alarms/zero_beat_rate": 1.0}, step=step)
                state["step500_warned"] = True

            # Decision rule: format_compliance < floor.
            if (fmt_compliance < decision_thresholds["format_floor"]
                    and step != state["format_warned_at"]):
                print(f"\n[grpo-decision] WARN @ step {step}: "
                      f"eval/format_compliance={fmt_compliance:.3f} < "
                      f"{decision_thresholds['format_floor']}. Consider "
                      f"increasing format_compliance weight (warning only).")
                wandb_utils.log({
                    "alarms/format_below_floor": 1.0,
                    "alarms/format_value": fmt_compliance,
                }, step=step)
                state["format_warned_at"] = step

            # ----- Best-checkpoint tracking ----- #
            if mean_total_reward > state["best_total_reward"]:
                old = state["best_total_reward"]
                state["best_total_reward"] = mean_total_reward
                state["best_step"] = step
                print(f"[grpo][eval@{step}] new best total_reward_mean="
                      f"{mean_total_reward:.4f} (prev {old:.4f}); "
                      f"saving to {best_dir}")
                try:
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    best_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(best_dir))
                    tokenizer.save_pretrained(str(best_dir))
                    wandb_utils.update_summary({
                        "best/total_reward_mean": mean_total_reward,
                        "best/step": step,
                    })
                except Exception as exc:
                    print(f"[grpo] WARN: failed to save best checkpoint: "
                          f"{exc!r}", file=sys.stderr)

            # Switch back to training mode.
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
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="LoRA adapter directory to start GRPO from. "
                             "Defaults to config.SFT_CHECKPOINT_PATH_FOR_GRPO "
                             "(checkpoints/sft_warmup/checkpoint-50).")
    parser.add_argument("--output", type=str, default="checkpoints/grpo")
    parser.add_argument("--model", type=str,
                        default=os.getenv(
                            "QUBIT_MEDIC_MODEL",
                            "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"),
                        help="Base model. Defaults to the 4-bit unsloth bundle "
                             "matching the SFT base.")
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
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--inloop-eval-every", type=int, default=None)
    parser.add_argument("--inloop-eval-episodes", type=int, default=None)
    parser.add_argument("--kl-alarm", type=float, default=None)
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the diversity preflight (DEBUG ONLY)")
    args = parser.parse_args(list(argv))

    # Lazy heavy imports.
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. "
              "Run `pip install -r requirements-train.txt`", file=sys.stderr)
        return 1
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    # Pre-flight signature check + stale-cache wipe.
    _wipe_stale_grpo_cache()
    _assert_grpo_signature_compatible()

    from qubit_medic import wandb_utils
    from qubit_medic.client.client import make_default_client
    from qubit_medic.config import (
        GRPO_BATCH_SIZE, GRPO_CHECKPOINT_EVERY, GRPO_DECISION_BEAT_RATE_CHECK_STEP,
        GRPO_DECISION_FORMAT_FLOOR, GRPO_DECISION_GRAD_NORM_CEIL,
        GRPO_DECISION_GRAD_NORM_RUN_LEN, GRPO_DECISION_REWARD_STD_CHECK_STEP,
        GRPO_DECISION_REWARD_STD_FLOOR, GRPO_DO_SAMPLE, GRPO_GEN_PER_PROMPT,
        GRPO_GRAD_ACCUM, GRPO_INSPECTION_COLLAPSE_THRESHOLD,
        GRPO_INSPECTION_HOOK_EVERY, GRPO_INSPECTION_SAMPLE_N, GRPO_KL_ALARM,
        GRPO_KL_COEF, GRPO_LOG_EVERY, GRPO_LR, GRPO_LR_SCHEDULER,
        GRPO_MAX_COMPLETION_LEN, GRPO_MAX_PROMPT_LEN, GRPO_OPTIMIZER,
        GRPO_REPETITION_PENALTY, GRPO_SAMPLE_LOG_EVERY, GRPO_SAMPLE_LOG_N,
        GRPO_SAVE_TOTAL_LIMIT, GRPO_STEPS, GRPO_TEMP_BUMP_ON_COLLAPSE,
        GRPO_TEMPERATURE, GRPO_TOP_K, GRPO_TOP_P, GRPO_VAL_EPISODES,
        GRPO_VAL_PATH, GRPO_VAL_SEED, GRPO_WALL_SECONDS, LORA_ALPHA, LORA_DROPOUT,
        LORA_R, LORA_TARGET_MODULES, MODEL_ID, PRIMARY_SEED, REWARD_WEIGHTS,
        SFT_CHECKPOINT_PATH_FOR_GRPO, WANDB_INLOOP_EVAL_EPISODES,
        WANDB_INLOOP_EVAL_EVERY,
    )

    sft_ckpt = args.sft_checkpoint or SFT_CHECKPOINT_PATH_FOR_GRPO
    steps = args.steps if args.steps is not None else GRPO_STEPS
    gen_per_prompt = args.gen_per_prompt if args.gen_per_prompt is not None else GRPO_GEN_PER_PROMPT
    lr = args.lr if args.lr is not None else GRPO_LR
    kl_coef = args.kl_coef if args.kl_coef is not None else GRPO_KL_COEF
    max_p = args.max_prompt_len if args.max_prompt_len is not None else GRPO_MAX_PROMPT_LEN
    max_c = args.max_completion_len if args.max_completion_len is not None else GRPO_MAX_COMPLETION_LEN
    seed = args.seed if args.seed is not None else PRIMARY_SEED
    sample_every = args.sample_every if args.sample_every is not None else GRPO_SAMPLE_LOG_EVERY
    sample_n = args.sample_n if args.sample_n is not None else GRPO_SAMPLE_LOG_N
    inloop_every = args.inloop_eval_every if args.inloop_eval_every is not None else WANDB_INLOOP_EVAL_EVERY
    inloop_episodes = args.inloop_eval_episodes if args.inloop_eval_episodes is not None else WANDB_INLOOP_EVAL_EPISODES
    kl_alarm = args.kl_alarm if args.kl_alarm is not None else GRPO_KL_ALARM

    _seed_everything(seed)

    # ---- Env client --------------------------------------------------- #
    env_client = make_default_client()
    print(f"using env client: {type(env_client).__name__}; "
          f"health = {env_client.health()}")

    # ---- W&B init ----------------------------------------------------- #
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
                "kl_alarm": kl_alarm,
                "temperature": GRPO_TEMPERATURE,
                "top_p": GRPO_TOP_P,
                "top_k": GRPO_TOP_K,
                "repetition_penalty": GRPO_REPETITION_PENALTY,
                "do_sample": GRPO_DO_SAMPLE,
                "lr_scheduler": GRPO_LR_SCHEDULER,
                "optimizer": GRPO_OPTIMIZER,
                "grad_accum": GRPO_GRAD_ACCUM,
                "effective_batch": GRPO_BATCH_SIZE * GRPO_GRAD_ACCUM,
                "sft_checkpoint": sft_ckpt,
                "model": args.model,
                "seed": seed,
                "report_to": report_to,
                "wall_seconds": GRPO_WALL_SECONDS,
                "reward_weights": dict(REWARD_WEIGHTS),
                "val_seed": GRPO_VAL_SEED,
                "val_episodes": GRPO_VAL_EPISODES,
            },
        },
    )
    # Use train/global_step as default x-axis for everything we log.
    try:
        run = wandb_utils.get_run()
        if run is not None:
            run.define_metric("train/global_step")
            run.define_metric("train/*", step_metric="train/global_step")
            run.define_metric("eval/*", step_metric="train/global_step")
            run.define_metric("alarms/*", step_metric="train/global_step")
            run.define_metric("rl/*", step_metric="train/global_step")
            run.define_metric("best/*", step_metric="train/global_step")
    except Exception as exc:
        print(f"[wandb] could not define x-axis metric: {exc!r}", file=sys.stderr)

    # ---- Build prompt pool -------------------------------------------- #
    print(f"pre-generating {args.prompt_pool} prompts ...")
    prompts = _build_prompt_pool(env_client, args.prompt_pool)
    dataset = Dataset.from_list(prompts)
    print(f"  built dataset with {len(dataset)} prompts")

    # ---- Frozen eval set --------------------------------------------- #
    eval_rows = _load_or_build_eval_set(
        env_client, seed=GRPO_VAL_SEED, n=inloop_episodes, path=GRPO_VAL_PATH,
    )

    # ---- Load model --------------------------------------------------- #
    print(f"loading base={args.model}, sft adapter={sft_ckpt}")
    base_for_load = sft_ckpt if Path(sft_ckpt).exists() else args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_for_load,
        max_seq_length=max_p + max_c,
        load_in_4bit=True,
        dtype=None,
    )
    if not Path(sft_ckpt).exists():
        print(f"[grpo] WARN: SFT checkpoint {sft_ckpt} not found; "
              f"attaching fresh LoRA on the base model")
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=list(LORA_TARGET_MODULES),
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

    # ---- Diversity preflight ----------------------------------------- #
    if not args.skip_preflight:
        ok = _diversity_preflight(
            model, tokenizer,
            val_path="data/sft_validation.jsonl",
            n_prompts=5, n_samples_per_prompt=8,
            temperature=GRPO_TEMPERATURE,
            min_unique=3, min_passing=3,
            max_new_tokens=max_c,
        )
        if not ok:
            wandb_utils.update_summary({"preflight/passed": False})
            wandb_utils.finish_run()
            return 2
        wandb_utils.update_summary({"preflight/passed": True})
    else:
        print("[grpo] --skip-preflight given; bypassing diversity preflight "
              "(DEBUG ONLY)")

    # ---- TRL GRPOConfig ---------------------------------------------- #
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"
    final_dir = output_dir / "final"
    bf16_supported = (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )

    grpo_kwargs: dict = {
        "output_dir": str(output_dir),
        "max_steps": steps,
        "per_device_train_batch_size": GRPO_BATCH_SIZE,
        "gradient_accumulation_steps": GRPO_GRAD_ACCUM,
        "num_generations": gen_per_prompt,
        "max_prompt_length": max_p,
        "max_completion_length": max_c,
        "learning_rate": lr,
        "beta": kl_coef,
        "lr_scheduler_type": GRPO_LR_SCHEDULER,
        "optim": GRPO_OPTIMIZER,
        "logging_steps": GRPO_LOG_EVERY,
        "save_steps": GRPO_CHECKPOINT_EVERY,
        "save_total_limit": GRPO_SAVE_TOTAL_LIMIT,
        "save_only_model": False,
        "seed": seed,
        "bf16": bf16_supported,
        "fp16": torch.cuda.is_available() and not bf16_supported,
        "report_to": report_to,
        "run_name": run_name,
        # Diversity-focused rollout sampling.
        "temperature": GRPO_TEMPERATURE,
        "top_p": GRPO_TOP_P,
        "top_k": GRPO_TOP_K,
        "repetition_penalty": GRPO_REPETITION_PENALTY,
    }

    # Some TRL versions don't accept every sampling kwarg on GRPOConfig;
    # fall back gracefully so the script still runs.
    config = None
    dropped: list[str] = []
    while config is None:
        try:
            config = GRPOConfig(**grpo_kwargs)
        except TypeError as exc:
            msg = str(exc)
            removed = False
            for k in ("repetition_penalty", "top_k", "top_p", "temperature",
                      "save_only_model"):
                if k in msg and k in grpo_kwargs:
                    grpo_kwargs.pop(k)
                    dropped.append(k)
                    removed = True
                    break
            if not removed:
                raise
    if dropped:
        print(f"[grpo] WARN: TRL did not accept these GRPOConfig kwargs and "
              f"they were dropped: {dropped}. Using TRL defaults for them.")

    # ---- Reward functions + scoring cache ----------------------------- #
    cache = _BatchScoringCache(env_client=env_client,
                               reward_weights=dict(REWARD_WEIGHTS))
    reward_fns = _make_reward_fns(cache)
    # The first reward is the bounded weighted-total used for the gradient;
    # the rest are zero-weight observers used only for per-component traces.
    reward_weights = [1.0] + [0.0] * len(_REWARD_COMPONENTS)

    callbacks = []
    cb = _build_wandb_callback(
        cache, model, tokenizer, env_client, eval_rows,
        sample_every=sample_every, sample_n=sample_n,
        inloop_every=inloop_every,
        inloop_max_new_tokens=max_c,
        kl_alarm=kl_alarm,
        inspection_every=GRPO_INSPECTION_HOOK_EVERY,
        inspection_sample_n=GRPO_INSPECTION_SAMPLE_N,
        inspection_collapse_threshold=GRPO_INSPECTION_COLLAPSE_THRESHOLD,
        temp_bump_on_collapse=GRPO_TEMP_BUMP_ON_COLLAPSE,
        best_dir=best_dir, output_dir=output_dir,
        wall_seconds=GRPO_WALL_SECONDS,
        decision_thresholds={
            "reward_std_floor": GRPO_DECISION_REWARD_STD_FLOOR,
            "reward_std_check_step": GRPO_DECISION_REWARD_STD_CHECK_STEP,
            "beat_rate_check_step": GRPO_DECISION_BEAT_RATE_CHECK_STEP,
            "format_floor": GRPO_DECISION_FORMAT_FLOOR,
            "grad_norm_ceil": GRPO_DECISION_GRAD_NORM_CEIL,
            "grad_norm_run_len": GRPO_DECISION_GRAD_NORM_RUN_LEN,
        },
    )
    if cb is not None:
        callbacks.append(cb)

    # Older TRL versions: GRPOTrainer may not accept reward_weights kw.
    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fns,
        reward_weights=reward_weights,
        callbacks=callbacks,
    )
    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError as exc:
        if "reward_weights" in str(exc):
            print("[grpo] WARN: this TRL does not accept reward_weights= "
                  "on GRPOTrainer; falling back to using only the bounded "
                  "weighted-total reward (and no observers).")
            trainer_kwargs.pop("reward_weights")
            trainer_kwargs["reward_funcs"] = [reward_fns[0]]
            trainer = GRPOTrainer(**trainer_kwargs)
        else:
            raise

    print(f"running GRPO for {steps} steps "
          f"(temperature={GRPO_TEMPERATURE}, top_p={GRPO_TOP_P}, "
          f"top_k={GRPO_TOP_K}, repetition_penalty={GRPO_REPETITION_PENALTY}, "
          f"beta={kl_coef}, lr={lr}) ...")
    started = time.time()
    train_result = trainer.train()
    elapsed = time.time() - started
    print(f"finished in {elapsed:.1f}s")

    metrics = getattr(train_result, "metrics", {}) or {}
    wandb_utils.update_summary({
        "grpo/wall_seconds": elapsed,
        "grpo/total_episodes": cache._episodes,
        "grpo/total_timeouts": cache._timeouts,
        "grpo/reward_bounds_violations": cache._bounds_violations,
        **{f"grpo/final/{k}": v for k, v in metrics.items()
           if isinstance(v, (int, float))},
    })

    # ---- Final + rolling adapter saves ------------------------------- #
    print(f"saving rolling adapter snapshot to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"saving final adapter snapshot to {final_dir}")
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if not args.no_artifact:
        wandb_utils.log_artifact(
            str(final_dir),
            name=f"grpo-final-{run_name}",
            artifact_type="model",
            description="GRPO final LoRA adapter (Qubit-Medic).",
        )
        if best_dir.exists():
            wandb_utils.log_artifact(
                str(best_dir),
                name=f"grpo-best-{run_name}",
                artifact_type="model",
                description="GRPO best-eval LoRA adapter (Qubit-Medic).",
            )

    wandb_utils.finish_run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
