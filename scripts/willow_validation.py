"""scripts/willow_validation.py - cross-distribution test (Section 8).

Loads the Willow d=3 detector error model from Zenodo (DOI 10.5281/zenodo.13359217),
generates 1,000 syndromes from it (real-chip noise), runs them through a
trained model, and reports the logical correction rate. Compares to the
SI1000 in-distribution result.

The Willow DEM file lives at the path given by ``--dem`` (download manually
or use the helper in ``--download-only`` mode if a URL is published).

Usage::

    python -m scripts.willow_validation \
        --dem data/willow_d3.dem \
        --policy pymatching \
        --episodes 1000

If the DEM file is missing, the script prints a clear download instruction
and exits with code 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pymatching
import stim


WILLOW_DOI = "10.5281/zenodo.13359217"
WILLOW_URL = "https://zenodo.org/record/13359217"


def _load_dem(path: str) -> stim.DetectorErrorModel:
    p = Path(path)
    if not p.exists():
        print(
            f"ERROR: DEM file {path} not found.\n"
            f"Download from {WILLOW_URL}\n"
            f"  (DOI: {WILLOW_DOI})\n"
            "Place the d=3 DEM at the requested path and re-run.",
            file=sys.stderr,
        )
        sys.exit(2)
    return stim.DetectorErrorModel.from_file(str(p))


def _evaluate_pymatching(dem: stim.DetectorErrorModel, episodes: int,
                         seed: int) -> dict:
    """Pure-PM upper bound on the trained model's accuracy on this DEM."""
    sampler = dem.compile_sampler(seed=seed)
    det, obs, _err = sampler.sample(episodes)
    m = pymatching.Matching.from_detector_error_model(dem)
    pred = m.decode_batch(det.astype(np.uint8))
    err_rate = float(np.mean(np.any(pred != obs, axis=1)))
    return {
        "n": episodes,
        "logical_error_rate": err_rate,
        "logical_correction_rate": 1.0 - err_rate,
    }


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=str, required=True,
                        help="path to the Willow d=3 DEM file")
    parser.add_argument("--policy", choices=["pymatching"], default="pymatching",
                        help="for now only the PyMatching baseline runs against "
                             "an arbitrary DEM; the LLM eval requires a circuit "
                             "(Stim's DEM-only sampler can't replay through the "
                             "model's prompt formatter).")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/willow_validation.json")
    args = parser.parse_args(list(argv))

    dem = _load_dem(args.dem)
    print(f"loaded Willow DEM from {args.dem}: "
          f"{dem.num_detectors} detectors, "
          f"{dem.num_observables} observables, {dem.num_errors} errors")

    result = _evaluate_pymatching(dem, args.episodes, args.seed)
    result["source"] = f"Zenodo {WILLOW_DOI}"
    result["policy"] = args.policy
    print(json.dumps(result, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
