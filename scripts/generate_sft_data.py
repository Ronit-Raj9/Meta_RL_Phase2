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


# Quota-based generation (master spec, section 1, plus the dataset-audit
# fix): instead of weighted sampling + global rejection (which biased L1
# down because L1 produces mostly trivial syndromes), we generate a fixed
# count per level with per-level non-empty floors. This guarantees the
# 40/50/10 curriculum split exactly while still hitting an overall
# non-empty fraction in the 65-75% target band.
LEVEL_QUOTAS_TRAIN: dict[str, int] = {
    "L1_warmup": 1200,   # 40% of 3000
    "L2_target": 1500,   # 50%
    "L3_stretch": 300,   # 10%
}

LEVEL_QUOTAS_VAL: dict[str, int] = {
    "L1_warmup": 40,     # 40% of 100
    "L2_target": 50,     # 50%
    "L3_stretch": 10,    # 10%
}

# Per-level minimum non-empty correction fraction. The math (with the
# configured 40/50/10 quota mix) gives:
#     L1 0.30 + L2 0.80 + L3 0.90 = 0.40*0.30 + 0.50*0.80 + 0.10*0.90 = 0.61
# which lands solidly inside the audit's 55-75% target band. ``None``
# would mean "accept all draws naturally" but the natural non-empty rate
# at L1's p=0.0005 (~3.5%) is too low to satisfy the audit, so we enforce
# an explicit floor here too. L1 keeps a deliberately lower floor so the
# warmup level retains a meaningful share of trivial "no error" cases.
PER_LEVEL_NONEMPTY_FLOOR: dict[str, float | None] = {
    "L1_warmup": 0.30,   # ~360 non-empty + 840 empty per 1200
    "L2_target": 0.80,   # ~1200 non-empty + 300 empty per 1500
    "L3_stretch": 0.90,  # ~270 non-empty + 30 empty per 300
}

# Held-out validation runs from a disjoint seed stream so it is truly
# independent of the train split.
VALIDATION_SEED_OFFSET: int = 4_242


def _quotas_from_total(total: int, base: dict[str, int]) -> dict[str, int]:
    """Scale ``base`` quota proportions to sum to ``total``.

    When the user passes ``--n`` or ``--val-n`` overriding the default
    sizes, we keep the 40/50/10 curriculum proportions and absorb any
    rounding remainder into the largest level (L2) so the file row count
    matches ``total`` exactly.
    """
    base_sum = sum(base.values())
    if base_sum == 0:
        return {k: 0 for k in base}
    scaled = {k: int(round(v * total / base_sum)) for k, v in base.items()}
    diff = total - sum(scaled.values())
    if diff != 0:
        # Largest level absorbs the remainder.
        target = max(scaled, key=scaled.get)
        scaled[target] += diff
    return scaled


def _build_caches() -> dict[str, dict]:
    """Pre-compile circuits / matchers once per level."""
    caches: dict[str, dict] = {}
    for name in LEVEL_QUOTAS_TRAIN.keys():
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


# Per-level seed offsets so each level draws an independent shot stream
# from a distinct RNG. Without this, switching from L1 to L2 with the
# same `seed` would produce identical syndromes (Stim's RNG is per-sampler).
_LEVEL_SEED_OFFSETS: dict[str, int] = {
    "L1_warmup": 0,
    "L2_target": 100_000,
    "L3_stretch": 200_000,
}

# Safety cap on shots per level. With L1 floor=0.50 at p=0.0005 (~3.5%
# natural non-empty rate) we expect ~17k shots; 1M is a generous ceiling
# that triggers a descriptive error if generation can't converge -- e.g.
# someone bumped a level's floor too aggressively for its physical error
# rate.
_MAX_SHOTS_PER_LEVEL: int = 1_000_000

# Stim's compile_detector_sampler is the slow step (~ms per call); once
# compiled, sample(N) is essentially free. We sample in chunks of this
# size to amortise the compile cost across thousands of shots.
_SHOT_BATCH_SIZE: int = 4096


def _level_shot_stream(cache: dict, base_seed: int):
    """Yield ``(det_row, obs_row)`` tuples lazily from a level's circuit.

    Compiles the detector sampler exactly ONCE per level and then pulls
    shots in batches of :data:`_SHOT_BATCH_SIZE`. ``det_row`` is a
    ``np.uint8`` 1-D array (the detector activations); ``obs_row`` is the
    1-D observables vector for the same shot.

    Determinism: the same ``base_seed`` always produces the same shot
    sequence regardless of batch size (Stim's per-sampler RNG advances
    deterministically across each ``sample()`` call).
    """
    sampler = cache["circuit"].compile_detector_sampler(seed=base_seed)
    while True:
        det, obs = sampler.sample(_SHOT_BATCH_SIZE, separate_observables=True)
        for i in range(_SHOT_BATCH_SIZE):
            yield det[i].astype(np.uint8), obs[i]


def _generate_split(
    *,
    quotas: dict[str, int],
    seed: int,
    caches: dict[str, dict],
    out_path: Path,
    rng: random.Random,
) -> tuple[int, int, int]:
    """Quota-based generator with per-level non-empty floors.

    Returns ``(n_written, n_syndrome, n_errors)``.

    For each level in ``quotas`` we generate exactly ``quotas[level]`` rows.
    Within each level, :data:`PER_LEVEL_NONEMPTY_FLOOR` controls the
    non-empty/empty split:

      * ``floor=None`` -> accept every draw until the quota is filled
        (mostly empty for low-p levels).
      * ``floor=f``    -> accept exactly ``round(level_n * f)`` non-empty
        rows and ``level_n - round(level_n * f)`` empty rows. Surplus on
        either side is dropped, draws continue until both sub-quotas are
        filled or :data:`_MAX_SHOTS_PER_LEVEL` is exceeded.

    Stim sampling is batched per level (single ``compile_detector_sampler``
    call, chunked ``sample()``) so generation is ~1 second per level even
    when the floor demands tens of thousands of shots.
    """
    written = n_with_syndrome = n_with_errors = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # rng kept for caller-API compatibility; sampling determinism is
    # driven by ``seed`` + per-level offsets.
    del rng

    with out_path.open("w") as f:
        for level_name, level_n in quotas.items():
            cache = caches[level_name]
            layout = cache["layout"]
            floor = PER_LEVEL_NONEMPTY_FLOOR.get(level_name)

            if floor is None:
                target_nonempty = None
                target_empty = None
            else:
                target_nonempty = int(round(level_n * floor))
                target_empty = level_n - target_nonempty

            level_nonempty = 0
            level_empty = 0
            shots_drawn = 0
            level_seed = seed + _LEVEL_SEED_OFFSETS.get(level_name, 0)
            shots = _level_shot_stream(cache, level_seed)

            while (level_nonempty + level_empty) < level_n:
                if shots_drawn >= _MAX_SHOTS_PER_LEVEL:
                    raise RuntimeError(
                        f"[gen] level {level_name}: exceeded "
                        f"_MAX_SHOTS_PER_LEVEL={_MAX_SHOTS_PER_LEVEL} with "
                        f"only {level_nonempty} non-empty + {level_empty} "
                        f"empty rows (target: {target_nonempty} non-empty + "
                        f"{target_empty} empty). Either lower "
                        f"PER_LEVEL_NONEMPTY_FLOOR[{level_name!r}] or "
                        f"raise the level's physical error rate in "
                        f"qubit_medic/config.py."
                    )
                det_row, obs_row = next(shots)
                shots_drawn += 1

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

                # Per-level quota acceptance:
                if floor is None:
                    pass  # accept anything until level_n is filled
                elif is_nonempty:
                    if level_nonempty >= target_nonempty:
                        continue  # surplus non-empty for this level
                else:
                    if level_empty >= target_empty:
                        continue  # surplus empty for this level

                actual_obs = int(obs_row[0]) if obs_row.shape[0] else 0

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
                    "actual_observable_flip": actual_obs,
                    "pymatching_observable_pred": pm_obs,
                    "had_syndrome": bool(det_row.any()),
                    "had_errors": bool(px or pz),
                }
                f.write(json.dumps(record) + "\n")
                written += 1
                if record["had_errors"]:
                    n_with_errors += 1
                    level_nonempty += 1
                else:
                    level_empty += 1
                if record["had_syndrome"]:
                    n_with_syndrome += 1

            print(f"  [{level_name}] {level_nonempty} non-empty + "
                  f"{level_empty} empty (drew {shots_drawn} shots, "
                  f"natural non-empty rate "
                  f"~{level_nonempty / max(1, shots_drawn):.1%})")

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
    train_quotas = _quotas_from_total(args.n, LEVEL_QUOTAS_TRAIN)
    train_rng = random.Random(args.seed)
    print(f"writing TRAIN split: n={args.n}, seed={args.seed}, "
          f"quotas={train_quotas} -> {train_path}")
    train_written, train_syn, train_err = _generate_split(
        quotas=train_quotas, seed=args.seed, caches=caches,
        out_path=train_path, rng=train_rng,
    )
    print(f"  wrote {train_written}; syndrome-fraction={train_syn / max(1, train_written):.3f}; "
          f"non-empty-correction-fraction={train_err / max(1, train_written):.3f}")

    # ---- validation split (disjoint seed stream) ---------------------- #
    if not args.no_validation:
        val_quotas = _quotas_from_total(args.val_n, LEVEL_QUOTAS_VAL)
        val_seed = args.seed + VALIDATION_SEED_OFFSET
        val_rng = random.Random(val_seed)
        print(f"writing VAL  split: n={args.val_n}, seed={val_seed}, "
              f"quotas={val_quotas} -> {val_path}")
        val_written, val_syn, val_err = _generate_split(
            quotas=val_quotas, seed=val_seed, caches=caches,
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

    # ---- self-audit (fail fast on bad regen) -------------------------- #
    # Run the same audit train_sft.py runs at startup, so a regen that
    # silently produced bad data exits non-zero immediately rather than
    # waiting until the next training launch. Lazy import so we don't
    # pull in train_sft's heavy ML deps at import time.
    if not args.no_validation:
        try:
            from scripts.train_sft import audit_sft_dataset
        except ImportError as exc:
            print(f"[gen] could not run self-audit: {exc}", file=sys.stderr)
            return 0
        print()  # blank line before banner
        audit_sft_dataset(str(train_path), str(val_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
