"""scripts/generate_sft_data.py - SFT dataset generator (Section 5).

Loops 5,000 times, each iteration:
    1. Picks a curriculum level (40% L1, 50% L2, 10% L3 weighted).
    2. Samples a noisy syndrome from Stim (SI1000).
    3. Runs PyMatching to get the canonical correction.
    4. Formats the prompt + target completion.
    5. Writes one JSONL record per sample.

Output:
    data/sft_dataset.jsonl
    data/sft_dataset_sample.jsonl  (50 rows for repo commit + eyeball)

Run::

    .venv/bin/python -m scripts.generate_sft_data --n 5000 \
        --out data/sft_dataset.jsonl
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


LEVEL_MIX: list[tuple[str, float]] = [
    ("L1_warmup", 0.40),
    ("L2_target", 0.50),
    ("L3_stretch", 0.10),
]


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


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=SFT_DATASET_SIZE)
    parser.add_argument("--out", type=str, default="data/sft_dataset.jsonl")
    parser.add_argument("--sample-out", type=str,
                        default="data/sft_dataset_sample.jsonl",
                        help="optional small JSONL committed to the repo")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    args = parser.parse_args(list(argv))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path = Path(args.sample_out)
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    caches = _build_caches()
    print(f"prepared caches for {len(caches)} levels")

    n_with_errors = 0
    n_with_syndrome = 0
    written = 0

    f = out_path.open("w")
    sample_records: list[dict] = []
    try:
        for i in range(args.n):
            level_name = _pick_level(rng)
            cache = caches[level_name]
            layout = cache["layout"]
            sampler = cache["circuit"].compile_detector_sampler(seed=args.seed + i + 1)
            det, obs = sampler.sample(1, separate_observables=True)
            det_row = det[0].astype(np.uint8)

            # Optimal correction via PyMatching (X + Z Pauli frame on data qubits).
            px_stim, pz_stim = pymatching_predicted_pauli_frame(
                cache["matching"], det_row, layout,
            )
            # PyMatching's actual observable prediction (used to rectify the
            # snapped frame so it has the same logical-Z parity as PM).
            pm_obs = int(cache["matching"].decode(det_row)[0])
            px_stim, pz_stim = rectify_pauli_frame_to_observable(
                px_stim, pz_stim, pm_obs, layout,
            )
            # Render targets in LLM ID space (consecutive 0..N-1).
            px = layout.stim_to_llm(px_stim)
            pz = layout.stim_to_llm(pz_stim)

            prompt = build_prompt(
                distance=cache["level"].distance,
                rounds=cache["level"].rounds,
                p=cache["level"].p,
                syndrome_bits=det_row.tolist(),
                num_x_stabilizers=cache["n_x_stab"],
                num_z_stabilizers=cache["n_z_stab"],
                num_data_qubits=layout.num_data_qubits,
            )
            completion = format_completion(px, pz)
            record = {
                "prompt": prompt,
                "completion": completion,
                "level": level_name,
                "distance": cache["level"].distance,
                "rounds": cache["level"].rounds,
                "p": cache["level"].p,
                "actual_observable_flip": int(obs[0, 0]),
                "had_syndrome": bool(det_row.any()),
                "had_errors": bool(px or pz),
            }
            f.write(json.dumps(record) + "\n")

            if record["had_errors"]:
                n_with_errors += 1
            if record["had_syndrome"]:
                n_with_syndrome += 1
            written += 1

            if len(sample_records) < args.sample_size and rng.random() < 0.05:
                sample_records.append(record)
    finally:
        f.close()

    if not sample_records:  # ensure we at least seed the sample
        with out_path.open() as src:
            for line in src:
                sample_records.append(json.loads(line))
                if len(sample_records) >= args.sample_size:
                    break

    with sample_path.open("w") as sf:
        for r in sample_records:
            sf.write(json.dumps(r) + "\n")

    print(f"wrote {written} records to {out_path}")
    print(f"wrote {len(sample_records)} sample records to {sample_path}")
    print(f"  fraction with non-zero syndrome: {n_with_syndrome / max(1, written):.3f}")
    print(f"  fraction with non-empty PyMatching correction: "
          f"{n_with_errors / max(1, written):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
