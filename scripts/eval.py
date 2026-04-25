"""scripts/eval.py - held-out evaluation harness (Sections 6.2 + 7.3).

Runs a model (or one of the deterministic baselines) over a held-out set
of syndromes and reports:

    * format compliance rate
    * logical correction rate
    * mean Hamming-overlap with PyMatching
    * PyMatching beat-rate
    * mean total reward

Usage::

    # Baseline run (no model; uses PyMatching-imitator):
    python -m scripts.eval --policy pymatching --episodes 200

    # Trained model (loads adapters via Unsloth):
    python -m scripts.eval --adapter checkpoints/grpo --episodes 500
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from qubit_medic.client.client import LocalDecoderClient
from qubit_medic.config import primary_level


def _summary(name: str, results: list[dict]) -> dict:
    n = max(1, len(results))
    out = {
        "name": name,
        "episodes": len(results),
        "format_compliance_rate": sum(r["format_compliance"] >= 0.5 for r in results) / n,
        "logical_correction_rate": sum(r["logical_correction"] >= 0.5 for r in results) / n,
        "mean_hamming_overlap": sum(r["hamming_overlap"] for r in results) / n,
        "pymatching_beat_rate": sum(r["pymatching_beat"] >= 0.5 for r in results) / n,
        "mean_total_reward": sum(r["total"] for r in results) / n,
    }
    return out


def _eval_baseline(name: str, episodes: int, level: str) -> dict:
    from scripts.baseline_policies import (
        policy_pymatching, policy_zeros, policy_random,
    )
    import random as _r
    rng = _r.Random(0)
    pol_map = {
        "pymatching": lambda obs: policy_pymatching(obs, env_client=None),
        "zeros": policy_zeros,
        "random": lambda obs: policy_random(obs, rng=rng),
    }
    if name not in pol_map:
        raise ValueError(f"unknown baseline {name}; choose from {sorted(pol_map)}")
    pol = pol_map[name]
    client = LocalDecoderClient()
    rewards = []
    for ep in range(episodes):
        obs = client.reset(forced_level=level, seed=10_000 + ep)
        result = client.step(raw_response=pol(obs), episode_id=obs.episode_id)
        rewards.append(result.info["rewards"])
    return _summary(name, rewards)


def _eval_model(adapter: str, episodes: int, level: str,
                base_model: str, max_new_tokens: int) -> dict:
    """Use Unsloth to load the adapter and generate completions."""
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter if adapter else base_model,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    client = LocalDecoderClient()
    rewards = []
    for ep in range(episodes):
        obs = client.reset(forced_level=level, seed=10_000 + ep)
        chat = [{"role": "user", "content": obs.prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic eval
            temperature=1.0, top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        result = client.step(raw_response=completion, episode_id=obs.episode_id)
        rewards.append(result.info["rewards"])
    return _summary(f"model[{adapter}]", rewards)


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["random", "zeros", "pymatching"],
                        default=None,
                        help="evaluate a deterministic baseline instead of a model")
    parser.add_argument("--adapter", type=str, default=None,
                        help="path to LoRA adapter dir; mutually exclusive with --policy")
    parser.add_argument("--base-model", type=str,
                        default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--level", type=str, default=primary_level().name)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(list(argv))

    if (args.policy is None) == (args.adapter is None):
        print("ERROR: exactly one of --policy and --adapter is required",
              file=sys.stderr)
        return 1

    if args.policy is not None:
        result = _eval_baseline(args.policy, args.episodes, args.level)
    else:
        result = _eval_model(args.adapter, args.episodes, args.level,
                             args.base_model, args.max_new_tokens)
    result["level"] = args.level
    print(json.dumps(result, indent=2))
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
