"""Three baseline policies (Section 2.7 of the plan).

Run::

    .venv/bin/python -m scripts.baseline_policies --episodes 500

Expected ranges (Section 2.7):

* Random policy:     ~10% logical correction
* All-zeros policy:  ~99% on L1 (warmup, p=0.0001), ~99% on L2 (still small)
* PyMatching imitator: ~99-100% logical correction

The plan's quoted numbers ("~10%", "~40%", "~97%") refer to a different
counting (per-shot accuracy on a *high-noise* level). At p=0.001 the
syndromes are mostly all-zero, so the all-zeros baseline will look very
strong. We report both the headline level (L2) and a high-noise level
(p=0.01) for an honest comparison.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Callable, Iterable

from qubit_medic.client.client import LocalDecoderClient
from qubit_medic.config import CURRICULUM, primary_level
from qubit_medic.models import DecoderObservation
from qubit_medic.prompts import format_completion


Policy = Callable[[DecoderObservation], str]


# --------------------------------------------------------------------------- #
# Three policies                                                               #
# --------------------------------------------------------------------------- #


def policy_random(obs: DecoderObservation, *, rng: random.Random) -> str:
    """Random qubit IDs - the noise floor."""
    n = max(1, obs.distance ** 2)  # number of data qubits
    k = rng.randint(0, max(1, n // 2))
    xs = sorted(rng.sample(range(n), k=min(k, n)))
    k = rng.randint(0, max(1, n // 2))
    zs = sorted(rng.sample(range(n), k=min(k, n)))
    return format_completion(xs, zs)


def policy_zeros(obs: DecoderObservation) -> str:
    """Always predict 'no errors'."""
    return format_completion([], [])


_PM_CACHE: dict[str, tuple] = {}


def policy_pymatching(obs: DecoderObservation, *, env_client: LocalDecoderClient) -> str:
    """Use PyMatching's prediction as the LLM imitator's response.

    This is a 'cheating' policy in the sense that it consults the same
    baseline used by Reward 5, so beat-rate is 0 by definition. Per-level
    Stim/PyMatching artefacts are cached so the policy stays fast.
    """
    import pymatching, numpy as np
    from qubit_medic.config import level_by_name
    from qubit_medic.server.physics import (
        build_circuit, build_dem, extract_layout,
        pymatching_predicted_pauli_frame, rectify_pauli_frame_to_observable,
    )
    cached = _PM_CACHE.get(obs.curriculum_level)
    if cached is None:
        lvl = level_by_name(obs.curriculum_level)
        c = build_circuit(lvl)
        dem = build_dem(c)
        m = pymatching.Matching.from_detector_error_model(dem)
        layout = extract_layout(c)
        cached = (m, layout)
        _PM_CACHE[obs.curriculum_level] = cached
    m, layout = cached
    syndrome = np.asarray(obs.syndrome_bits, dtype=np.uint8)
    px_stim, pz_stim = pymatching_predicted_pauli_frame(m, syndrome, layout)
    pm_obs = int(m.decode(syndrome)[0])
    px_stim, pz_stim = rectify_pauli_frame_to_observable(
        px_stim, pz_stim, pm_obs, layout,
    )
    return format_completion(layout.stim_to_llm(px_stim),
                             layout.stim_to_llm(pz_stim))


# --------------------------------------------------------------------------- #
# Evaluation harness                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class PolicyStats:
    name: str
    episodes: int = 0
    logical_correct: int = 0
    format_ok: int = 0
    beat_pm: int = 0
    sum_total: float = 0.0

    def update(self, info: dict, total: float) -> None:
        self.episodes += 1
        rewards = info["rewards"]
        if rewards["logical_correction"] >= 0.5:
            self.logical_correct += 1
        if rewards["format_compliance"] >= 0.5:
            self.format_ok += 1
        if rewards["pymatching_beat"] >= 0.5:
            self.beat_pm += 1
        self.sum_total += total

    def as_dict(self) -> dict:
        n = max(1, self.episodes)
        return {
            "name": self.name,
            "episodes": self.episodes,
            "logical_correction_rate": self.logical_correct / n,
            "format_compliance_rate": self.format_ok / n,
            "pymatching_beat_rate": self.beat_pm / n,
            "mean_total_reward": self.sum_total / n,
        }


def evaluate_policy(
    *,
    name: str,
    policy: Policy,
    episodes: int,
    forced_level: str,
    seed: int = 0,
) -> dict:
    """Run a policy for ``episodes`` shots at one curriculum level."""
    client = LocalDecoderClient()
    stats = PolicyStats(name=name)
    for ep in range(episodes):
        obs = client.reset(forced_level=forced_level, seed=seed + ep)
        raw = policy(obs)
        result = client.step(raw_response=raw, episode_id=obs.episode_id)
        stats.update(info=result.info, total=result.reward)
    return stats.as_dict()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200,
                        help="episodes per (policy, level) pair")
    parser.add_argument("--levels", nargs="*", default=["L1_warmup", "L2_target"])
    parser.add_argument("--out", type=str, default=None,
                        help="optional path to dump JSON results")
    args = parser.parse_args(list(argv))

    rng = random.Random(42)
    random_policy = lambda obs: policy_random(obs, rng=rng)  # noqa: E731
    pm_policy_client = LocalDecoderClient()
    pm_policy = lambda obs: policy_pymatching(obs, env_client=pm_policy_client)  # noqa: E731

    results = []
    for level in args.levels:
        for name, policy in (
            ("random", random_policy),
            ("zeros", policy_zeros),
            ("pymatching", pm_policy),
        ):
            r = evaluate_policy(
                name=name, policy=policy, episodes=args.episodes,
                forced_level=level,
            )
            r["level"] = level
            results.append(r)
            print(
                f"{level:<12} {name:<12} "
                f"LER={1 - r['logical_correction_rate']:.3f}  "
                f"correct={r['logical_correction_rate']:.3f}  "
                f"format={r['format_compliance_rate']:.3f}  "
                f"beat={r['pymatching_beat_rate']:.3f}  "
                f"mean_R={r['mean_total_reward']:.3f}"
            )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
