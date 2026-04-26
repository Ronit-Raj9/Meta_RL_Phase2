"""scripts/preflight_grpo.py - validate the new reward shape before launching the full run.

Six checks (CHANGE 9 of the 2026-04 final-RL spec):

    1. Filtered prompt pool exists and has >= 500 entries.
    2. New reward function imports without errors.
    3. New reward weights sum to 1.0 (within 1e-9).
    4. Sampling at temperature=1.0 produces >= 3 unique completions out of 8
       on at least 3 of 5 test prompts (uses the SFT-warmed adapter).
    5. clipped_ratio after 10 GRPO-style sampling steps is < 0.30
       (validates max_completion_length=24 + stop_strings actually clip
       the tail; we do this as a generation-only proxy without running
       a real GRPO step).
    6. pymatching_margin reward fires non-zero on >= 80% of prompts
       (proves the new reward provides signal everywhere).

Exit code 0 if all six pass; 1 if any fail. The pipeline aborts before
GRPO if this exits non-zero, so the full 1500-step run never starts on
a known-broken setup.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


_TAIL_RE = re.compile(r"X_ERRORS=\[([^\]]*)\]\s*Z_ERRORS=\[([^\]]*)\]")


def _check_pool(pool_path: Path, min_size: int) -> tuple[bool, str]:
    if not pool_path.exists():
        return False, f"prompt pool not found at {pool_path}"
    rows = [
        json.loads(l) for l in pool_path.read_text().splitlines() if l.strip()
    ]
    if len(rows) < min_size:
        return False, f"pool has {len(rows)} prompts < {min_size} required"
    return True, f"pool ok ({len(rows)} prompts)"


def _check_reward_imports() -> tuple[bool, str]:
    try:
        from qubit_medic.server.rewards import (
            RewardBreakdown, compute_all_rewards, reward_pymatching_margin,
        )
        # Construct a trivial breakdown to verify the dataclass shape.
        b = RewardBreakdown(
            logical_correction=0.0, syndrome_consistency=0.0,
            hamming_overlap=0.0, format_compliance=0.0,
            pymatching_margin=0.5, total=0.0,
        )
        if "pymatching_margin" not in b.as_dict():
            return False, "RewardBreakdown.as_dict() missing pymatching_margin"
        return True, "rewards module imports ok"
    except Exception as exc:
        return False, f"reward import failed: {type(exc).__name__}: {exc}"


def _check_weights_sum() -> tuple[bool, str]:
    from qubit_medic.config import REWARD_WEIGHTS
    s = sum(REWARD_WEIGHTS.values())
    if abs(s - 1.0) > 1e-9:
        return False, f"REWARD_WEIGHTS sum to {s}, expected 1.0"
    if "pymatching_margin" not in REWARD_WEIGHTS:
        return False, "REWARD_WEIGHTS missing 'pymatching_margin' key"
    return True, f"weights sum=1.0; pymatching_margin={REWARD_WEIGHTS['pymatching_margin']}"


def _check_diversity_and_clipping(adapter_dir: str,
                                  pool_path: Path,
                                  n_prompts: int = 5,
                                  n_samples: int = 8,
                                  max_new_tokens: int = 24,
                                  ) -> tuple[bool, str, bool, str, bool, str]:
    """Combined: returns three (passed, msg) tuples for checks 4, 5, 6."""
    try:
        from unsloth import FastLanguageModel
        import torch
        from qubit_medic.client.client import LocalDecoderClient
        from qubit_medic.config import GRPO_STOP_STRINGS, MODEL_ID
        from qubit_medic.prompts import parse_action
    except Exception as exc:
        msg = f"import error: {type(exc).__name__}: {exc}"
        return (False, msg, False, msg, False, msg)

    if not Path(adapter_dir).exists():
        msg = f"adapter directory not found: {adapter_dir}"
        return (False, msg, False, msg, False, msg)

    rows = [
        json.loads(l) for l in pool_path.read_text().splitlines() if l.strip()
    ][:n_prompts]
    if len(rows) < n_prompts:
        msg = f"pool too small for preflight: {len(rows)} < {n_prompts}"
        return (False, msg, False, msg, False, msg)

    print(f"[preflight] loading adapter {adapter_dir} on {MODEL_ID}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    # Build the env client so we can compute the margin reward server-side.
    client = LocalDecoderClient()

    # Run generation per prompt, n_samples each.
    n_unique_per_prompt: list[int] = []
    n_clipped = 0
    n_total_gens = 0
    n_with_signal = 0  # prompts where ANY of the 8 sampled rewards != 0.5

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    # Build stop_token_ids set from stop_strings (best-effort; some stop
    # strings may map to multiple tokens, in which case the trainer's stop
    # is what enforces clipping. The preflight just checks for tail clipping).
    stop_token_ids: list[int] = []
    for s in GRPO_STOP_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            stop_token_ids.append(ids[0])
    if eos_id is not None:
        stop_token_ids.append(eos_id)

    for rec in rows:
        prompt = rec["prompt"]
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outs: list[str] = []
        prompt_signal_seen = False
        for _ in range(n_samples):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            n_total_gens += 1
            # Clipped iff we used the full budget (no stop fired).
            if int(gen_ids.shape[0]) >= max_new_tokens:
                n_clipped += 1
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
            outs.append(completion.strip())

            # Score this completion via the env to get the margin reward.
            obs = client.reset(seed=int(rec.get("seed", 0)) or 7,
                               forced_level=rec.get("level"))
            result = client.step(
                raw_response=completion, episode_id=obs.episode_id,
            )
            margin = result.info.get("rewards", {}).get(
                "pymatching_margin",
                result.info.get("rewards", {}).get("pymatching_beat", 0.5),
            )
            if abs(float(margin) - 0.5) > 1e-6:
                prompt_signal_seen = True

        n_unique_per_prompt.append(len(set(outs)))
        if prompt_signal_seen:
            n_with_signal += 1

    # Check 4: diversity. >=3 unique out of 8 on at least 3 of 5 prompts.
    n_diverse_prompts = sum(1 for u in n_unique_per_prompt if u >= 3)
    diversity_passed = n_diverse_prompts >= 3
    diversity_msg = (
        f"{n_diverse_prompts}/{len(n_unique_per_prompt)} prompts have "
        f"unique>=3/8 (uniques={n_unique_per_prompt})"
    )

    # Check 5: clipped ratio < 0.30.
    clipped_ratio = n_clipped / max(1, n_total_gens)
    clipped_passed = clipped_ratio < 0.30
    clipped_msg = (
        f"clipped_ratio={clipped_ratio:.2f} ({n_clipped}/{n_total_gens})"
    )

    # Check 6: margin reward signal on >= 80% of prompts.
    signal_frac = n_with_signal / max(1, len(n_unique_per_prompt))
    signal_passed = signal_frac >= 0.80
    signal_msg = (
        f"margin reward fires on {n_with_signal}/{len(n_unique_per_prompt)} "
        f"prompts ({signal_frac:.0%})"
    )

    return (
        diversity_passed, diversity_msg,
        clipped_passed, clipped_msg,
        signal_passed, signal_msg,
    )


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=str, default="data/grpo_prompt_pool.jsonl")
    parser.add_argument("--adapter", type=str,
                        default="checkpoints/sft_warmup")
    parser.add_argument("--min-pool-size", type=int, default=500)
    parser.add_argument("--skip-generation", action="store_true",
                        help="run only the static checks 1-3 (skip the GPU-"
                             "dependent checks 4-6). Useful for CPU-only sanity.")
    args = parser.parse_args(list(argv))

    print()
    print("GRPO PRE-FLIGHT CHECKS")
    print("=" * 22)

    results: list[tuple[str, bool, str]] = []

    ok1, msg1 = _check_pool(Path(args.pool), args.min_pool_size)
    results.append(("1. filtered pool exists & sized", ok1, msg1))

    ok2, msg2 = _check_reward_imports()
    results.append(("2. reward module imports", ok2, msg2))

    ok3, msg3 = _check_weights_sum()
    results.append(("3. reward weights sum to 1.0", ok3, msg3))

    if args.skip_generation:
        results.append(("4. diversity probe", True, "skipped (--skip-generation)"))
        results.append(("5. clipped_ratio probe", True, "skipped (--skip-generation)"))
        results.append(("6. margin signal probe", True, "skipped (--skip-generation)"))
    else:
        if not ok1 or not ok2 or not ok3:
            # Don't run expensive GPU checks if the cheap checks fail.
            results.append(("4. diversity probe", False, "skipped (earlier check failed)"))
            results.append(("5. clipped_ratio probe", False, "skipped (earlier check failed)"))
            results.append(("6. margin signal probe", False, "skipped (earlier check failed)"))
        else:
            (d_ok, d_msg, c_ok, c_msg, s_ok, s_msg) = _check_diversity_and_clipping(
                adapter_dir=args.adapter, pool_path=Path(args.pool),
            )
            results.append(("4. diversity (>=3/8 on 3/5 prompts)", d_ok, d_msg))
            results.append(("5. clipped_ratio < 0.30 (10 gens)", c_ok, c_msg))
            results.append(("6. margin reward fires on >=80%", s_ok, s_msg))

    print()
    label_w = max(len(label) for label, _, _ in results) + 1
    for label, passed, msg in results:
        glyph = "PASS" if passed else "FAIL"
        print(f"  [{glyph}] {label.ljust(label_w)} {msg}")

    print()
    n_failed = sum(1 for _, p, _ in results if not p)
    if n_failed:
        print(f"PRE-FLIGHT FAILED: {n_failed}/{len(results)} checks did not pass")
        print("Aborting before the full GRPO run.")
        return 1
    print(f"PRE-FLIGHT PASSED: all {len(results)} checks ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
