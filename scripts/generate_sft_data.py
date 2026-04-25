"""scripts/generate_sft_data.py - SFT dataset generator (master spec, sec. 1).

Locked configuration:
    * Train split:      3,000 examples (default seed 42).
    * Held-out split:   100 examples (seed 4242 - independent stream).
    * Curriculum mix:   40% L1_warmup, 50% L2_target, 10% L3_stretch.

For each example:
    1. Pick a curriculum level by the locked mixture.
    2. Sample a noisy syndrome from Stim (SI1000 noise model).
    3. Run PyMatching to get the canonical correction (Pauli frame).
    4. Format the locked prompt + target completion.
    5. Emit one JSONL record per sample. Records carry ``true_x_errors``,
       ``true_z_errors``, ``actual_observable_flip``, and curriculum info
       so the SFT validation callback can compute every spec metric
       (logical_correction_rate, exact_match_pymatching, hamming_overlap,
       syndrome_consistency, ...) without re-sampling.

Output:
    data/sft_dataset.jsonl                 - training set (3,000 rows)
    data/sft_validation.jsonl              - held-out validation (100 rows)
    data/sft_dataset_sample.jsonl          - 50-row preview for repo commit

Run::

    python -m scripts.generate_sft_data \
        --n 3000 --val-n 100 \
        --out data/sft_dataset.jsonl \
        --val-out data/sft_validation.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pymatching

from qubit_medic.config import (
    PRIMARY_SEED,
    SFT_DATASET_SIZE,
    SFT_VAL_HOLDOUT,
    level_by_name,
)
from qubit_medic.prompts import build_prompt, format_completion
from qubit_medic.server.physics import (
    build_circuit,
    build_dem,
    extract_layout,
    per_round_x_z_counts,
    pymatching_predicted_pauli_frame,
    rectify_pauli_frame_to_observable,
)


# --------------------------------------------------------------------------- #
# Reasoning-prefix helper                                                     #
# --------------------------------------------------------------------------- #
# The locked prompt (qubit_medic/prompts.py) tells the model to "Reason
# briefly (1-2 sentences), then output your answer on the LAST line in this
# EXACT format". A bare-format-line target contradicts that, so we emit a
# 1-sentence template derived deterministically from (px, pz) and then the
# canonical format line. Prompt and target now agree, eliminating the
# oscillation that drove format_compliance from 0.54 -> 0.04 between
# step 30 and step 50 of the original SFT run.


def _build_reasoning(px: list[int], pz: list[int]) -> str:
    """Deterministic 1-sentence reasoning that matches the format line."""
    if not px and not pz:
        return ("All stabilizer measurements report no detector firings, "
                "indicating no data-qubit errors.")
    if px and not pz:
        ids = ", ".join(str(q) for q in sorted(set(px)))
        return f"Z-stabilizer firings localize X-errors to qubit(s) {ids}."
    if pz and not px:
        ids = ", ".join(str(q) for q in sorted(set(pz)))
        return f"X-stabilizer firings localize Z-errors to qubit(s) {ids}."
    x_ids = ", ".join(str(q) for q in sorted(set(px)))
    z_ids = ", ".join(str(q) for q in sorted(set(pz)))
    return (f"X-stabilizer firings localize Z-errors to qubit(s) {z_ids}, "
            f"and Z-stabilizer firings localize X-errors to qubit(s) {x_ids}.")


# Locked curriculum mixture (master spec, section 1: SFT learns format
# across the full distribution it will face during RL).
LEVEL_MIX: list[tuple[str, float]] = [
    ("L1_warmup", 0.40),
    ("L2_target", 0.50),
    ("L3_stretch", 0.10),
]

# Held-out validation runs from a disjoint seed stream so it is truly
# independent of the train split.
VALIDATION_SEED_OFFSET: int = 4_242


def _pick_level(rng: random.Random) -> str:
    roll = rng.random()
    cum = 0.0
    for name, w in LEVEL_MIX:
        cum += w
        if roll < cum:
            return name
    return LEVEL_MIX[-1][0]


def _build_caches() -> dict[str, dict]:
    """Pre-compile circuits / matchers once per level."""
    caches: dict[str, dict] = {}
    for name, _ in LEVEL_MIX:
        lvl = level_by_name(name)
        c = build_circuit(lvl)
        dem = build_dem(c)
        m = pymatching.Matching.from_detector_error_model(dem)
        layout = extract_layout(c)
        n_x, n_z = per_round_x_z_counts(layout)
        caches[name] = {
            "level": lvl,
            "circuit": c,
            "dem": dem,
            "matching": m,
            "layout": layout,
            "n_x_stab": n_x,
            "n_z_stab": n_z,
        }
    return caches


# Class-balance targets for SFT rejection sampling.
# At p=0.001 / d=3, ~80% of natural draws have no errors. Training on that
# distribution lets the model collapse to "always predict empty" and reach
# ~0.20 loss without ever learning the format. We enforce ~70/30
# non-trivial/trivial here so the cheap escape route is closed.
TARGET_NONEMPTY_FRACTION: float = 0.70
TRIVIAL_KEEP_PROB: float = 0.15  # keep ~15% of further trivial draws once at target


def _generate_split(
    *,
    n: int,
    seed: int,
    caches: dict[str, dict],
    out_path: Path,
    rng: random.Random,
) -> tuple[int, int, int]:
    """Generate ``n`` records to ``out_path``. Returns ``(n_written, n_syndrome, n_errors)``.

    Uses class-balanced rejection sampling so non-trivial syndromes are
    over-represented relative to the natural p=0.001 distribution. Each
    completion is composed of a deterministic 1-sentence reasoning prefix
    (see :func:`_build_reasoning`) followed by the canonical format line,
    so prompt and target stay in agreement.
    """
    written = n_with_syndrome = n_with_errors = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_nonempty = int(round(n * TARGET_NONEMPTY_FRACTION))
    target_empty = n - target_nonempty

    shot = 0
    with out_path.open("w") as f:
        while written < n:
            shot += 1
            level_name = _pick_level(rng)
            cache = caches[level_name]
            layout = cache["layout"]
            sampler = cache["circuit"].compile_detector_sampler(seed=seed + shot)
            det, obs = sampler.sample(1, separate_observables=True)
            det_row = det[0].astype(np.uint8)

            # Optimal correction via PyMatching (X + Z Pauli frame).
            px_stim, pz_stim = pymatching_predicted_pauli_frame(
                cache["matching"], det_row, layout,
            )
            pm_obs = int(cache["matching"].decode(det_row)[0])
            px_stim, pz_stim = rectify_pauli_frame_to_observable(
                px_stim, pz_stim, pm_obs, layout,
            )
            # LLM ID space (consecutive 0..N-1).
            px = layout.stim_to_llm(px_stim)
            pz = layout.stim_to_llm(pz_stim)
            is_nonempty = bool(px or pz)

            # Class-balance rejection: once we have enough trivial rows,
            # reject most further trivial draws so non-trivial rows
            # accumulate. Keep a small fraction of trivial draws so the
            # model still sees clean "no errors" outputs.
            empty_so_far = written - n_with_errors
            if (not is_nonempty
                    and empty_so_far >= target_empty
                    and rng.random() > TRIVIAL_KEEP_PROB):
                continue
            # Surplus non-trivial rows are fine; no rejection on that side.

            prompt = build_prompt(
                distance=cache["level"].distance,
                rounds=cache["level"].rounds,
                p=cache["level"].p,
                syndrome_bits=det_row.tolist(),
                num_x_stabilizers=cache["n_x_stab"],
                num_z_stabilizers=cache["n_z_stab"],
                num_data_qubits=layout.num_data_qubits,
            )
            reasoning = _build_reasoning(px, pz)
            completion = f"{reasoning} {format_completion(px, pz)}"
            record = {
                "prompt": prompt,
                "completion": completion,
                "level": level_name,
                "distance": cache["level"].distance,
                "rounds": cache["level"].rounds,
                "p": cache["level"].p,
                "num_data_qubits": int(layout.num_data_qubits),
                "num_x_stabilizers": int(cache["n_x_stab"]),
                "num_z_stabilizers": int(cache["n_z_stab"]),
                "syndrome_bits": [int(b) for b in det_row.tolist()],
                "true_x_errors": list(map(int, px)),
                "true_z_errors": list(map(int, pz)),
                "actual_observable_flip": int(obs[0, 0]),
                "pymatching_observable_pred": int(pm_obs),
                "had_syndrome": bool(det_row.any()),
                "had_errors": bool(px or pz),
            }
            f.write(json.dumps(record) + "\n")
            written += 1
            if record["had_errors"]:
                n_with_errors += 1
            if record["had_syndrome"]:
                n_with_syndrome += 1
    return written, n_with_syndrome, n_with_errors


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=SFT_DATASET_SIZE,
                        help=f"train split size (default {SFT_DATASET_SIZE})")
    parser.add_argument("--val-n", type=int, default=SFT_VAL_HOLDOUT,
                        help=f"held-out validation size (default {SFT_VAL_HOLDOUT})")
    parser.add_argument("--out", type=str, default="data/sft_dataset.jsonl")
    parser.add_argument("--val-out", type=str, default="data/sft_validation.jsonl")
    parser.add_argument("--sample-out", type=str,
                        default="data/sft_dataset_sample.jsonl",
                        help="optional small JSONL committed to the repo")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED,
                        help=f"deterministic seed (default {PRIMARY_SEED})")
    parser.add_argument("--no-validation", action="store_true",
                        help="skip writing the held-out validation split")
    args = parser.parse_args(list(argv))

    train_path = Path(args.out)
    val_path = Path(args.val_out)
    sample_path = Path(args.sample_out)
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    caches = _build_caches()
    print(f"prepared caches for {len(caches)} levels")

    # ---- training split ------------------------------------------------ #
    train_rng = random.Random(args.seed)
    print(f"writing TRAIN split: n={args.n}, seed={args.seed} -> {train_path}")
    train_written, train_syn, train_err = _generate_split(
        n=args.n, seed=args.seed, caches=caches,
        out_path=train_path, rng=train_rng,
    )
    print(f"  wrote {train_written}; syndrome-fraction={train_syn / max(1, train_written):.3f}; "
          f"non-empty-correction-fraction={train_err / max(1, train_written):.3f}")

    # ---- validation split (disjoint seed stream) ---------------------- #
    if not args.no_validation:
        val_seed = args.seed + VALIDATION_SEED_OFFSET
        val_rng = random.Random(val_seed)
        print(f"writing VAL  split: n={args.val_n}, seed={val_seed} -> {val_path}")
        val_written, val_syn, val_err = _generate_split(
            n=args.val_n, seed=val_seed, caches=caches,
            out_path=val_path, rng=val_rng,
        )
        print(f"  wrote {val_written}; syndrome-fraction={val_syn / max(1, val_written):.3f}; "
              f"non-empty-correction-fraction={val_err / max(1, val_written):.3f}")

    # ---- sample preview (for repo commit / eyeball QC) ---------------- #
    sample_records: list[dict] = []
    with train_path.open() as src:
        for line in src:
            sample_records.append(json.loads(line))
            if len(sample_records) >= args.sample_size:
                break
    with sample_path.open("w") as sf:
        for r in sample_records:
            sf.write(json.dumps(r) + "\n")
    print(f"wrote {len(sample_records)} sample records to {sample_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
