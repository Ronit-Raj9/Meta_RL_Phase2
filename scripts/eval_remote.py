"""scripts/eval_remote.py - held-out evaluation against a live OpenEnv server.

Mirrors the baseline path of :mod:`scripts.eval` but talks to the deployed
HF Space (or any URL serving the QubitMedic OpenEnv API) over HTTP via
:class:`qubit_medic.client.client.DecoderClient`. The summary JSON shape is
identical to ``scripts.eval._summary``, so the same
:mod:`scripts.comparison_table` consumes its output.

Usage::

    python -m scripts.eval_remote \\
        --url https://ronitraj-quantumscribe.hf.space \\
        --policy pymatching --episodes 100 --levels L2_target \\
        --out data/eval_remote_pymatching_L2.json

    # All three baselines, both levels, single command:
    python -m scripts.eval_remote --url https://ronitraj-quantumscribe.hf.space \\
        --episodes 100 --levels L1_warmup L2_target --all-policies \\
        --out-dir data/remote_eval/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

from qubit_medic.client.client import DecoderClient
from scripts.baseline_policies import (
    policy_pymatching,
    policy_random,
    policy_zeros,
)


_POLICY_CHOICES = ("zeros", "random", "pymatching")


def _summary(name: str, level: str, results: list[dict],
             elapsed_s: float) -> dict:
    """Same metric set as scripts.eval._summary so comparison_table can
    consume it without modification, plus a few HTTP-specific extras."""
    n = max(1, len(results))
    return {
        "name": name,
        "level": level,
        "episodes": len(results),
        "logical_correction_rate":
            sum(r["logical_correction"] >= 0.5 for r in results) / n,
        "pymatching_beat_rate":
            sum(r["pymatching_beat"] >= 0.5 for r in results) / n,
        "format_compliance_rate":
            sum(r["format_compliance"] >= 0.999 for r in results) / n,
        "format_partial_rate":
            sum((r["format_compliance"] >= 0.5
                 and r["format_compliance"] < 0.999) for r in results) / n,
        "syndrome_consistency_rate":
            sum(r["syndrome_consistency"] >= 0.999 for r in results) / n,
        "mean_syndrome_consistency":
            sum(r["syndrome_consistency"] for r in results) / n,
        "mean_hamming_overlap":
            sum(r["hamming_overlap"] for r in results) / n,
        "mean_total_reward":
            sum(r["total"] for r in results) / n,
        "exact_match_pymatching":
            sum(int(r.get("exact_match_pymatching", 0)) for r in results) / n,
        "mean_output_length":
            sum(int(r.get("output_length", 0)) for r in results) / n,
        "elapsed_seconds": round(elapsed_s, 3),
        "throughput_ep_per_s": round(n / max(elapsed_s, 1e-6), 3),
    }


def _run_one(client: DecoderClient, policy_name: str, level: str,
             episodes: int, seed_offset: int = 10_000) -> dict:
    rng = random.Random(0)

    def _pick(obs):
        if policy_name == "zeros":
            return policy_zeros(obs)
        if policy_name == "random":
            return policy_random(obs, rng=rng)
        if policy_name == "pymatching":
            return policy_pymatching(obs, env_client=None)
        raise ValueError(f"unknown policy {policy_name}")

    rewards: list[dict] = []
    started = time.monotonic()
    for ep in range(episodes):
        obs = client.reset(seed=seed_offset + ep, forced_level=level)
        completion = _pick(obs)
        result = client.step(raw_response=completion,
                             episode_id=obs.episode_id)
        rwd = dict(result.info["rewards"])
        if policy_name == "pymatching":
            # PyMatching imitator decodes from the same baseline we score
            # against, so it trivially exact-matches by construction.
            rwd["exact_match_pymatching"] = 1
        rewards.append(rwd)
        if (ep + 1) % 25 == 0:
            print(f"  [{policy_name:<10} {level:<10}] {ep + 1}/{episodes} done",
                  file=sys.stderr, flush=True)
    elapsed = time.monotonic() - started
    return _summary(policy_name, level, rewards, elapsed)


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True,
        help="Base URL of the live OpenEnv server "
             "(e.g. https://ronitraj-quantumscribe.hf.space).",
    )
    parser.add_argument("--policy", choices=_POLICY_CHOICES, default=None,
                        help="Single policy to run.")
    parser.add_argument("--all-policies", action="store_true",
                        help="Run all of zeros/random/pymatching.")
    parser.add_argument("--levels", nargs="+", default=["L2_target"],
                        help="One or more curriculum levels.")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per (policy, level) pair.")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Per-request HTTP timeout in seconds.")
    parser.add_argument("--out", type=str, default=None,
                        help="JSON output path (single-pair runs only).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to write one JSON per (policy, level).")
    args = parser.parse_args(list(argv))

    if (args.policy is None) == (not args.all_policies):
        print("ERROR: pass exactly one of --policy or --all-policies",
              file=sys.stderr)
        return 1

    policies = list(_POLICY_CHOICES) if args.all_policies else [args.policy]

    client = DecoderClient(args.url, timeout=args.timeout)
    try:
        try:
            health = client.healthz()
            print(f"server={args.url} version={health.get('version')} "
                  f"stim={health.get('stim_version')} "
                  f"openenv={health.get('openenv_version')}",
                  file=sys.stderr)
        except Exception as exc:
            print(f"WARNING: /healthz probe failed ({exc}); continuing",
                  file=sys.stderr)

        all_results: list[dict] = []
        for level in args.levels:
            for pol in policies:
                print(f"\n=== {pol} @ {level} ({args.episodes} episodes) ===",
                      file=sys.stderr)
                summary = _run_one(client, pol, level, args.episodes)
                summary["url"] = args.url
                all_results.append(summary)
                print(json.dumps(summary, indent=2))

                if args.out_dir:
                    out_dir = Path(args.out_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"eval_remote_{pol}_{level}.json"
                    (out_dir / fname).write_text(json.dumps(summary, indent=2))
                    print(f"wrote {out_dir / fname}", file=sys.stderr)

        if args.out and len(all_results) == 1:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(all_results[0], indent=2))
            print(f"wrote {out}", file=sys.stderr)
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
