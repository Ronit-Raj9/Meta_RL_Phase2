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

    # With W&B logging (summary + per-episode table):
    python -m scripts.eval --adapter checkpoints/grpo --episodes 500 \
        --report-to wandb --wandb-group my-experiment
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from qubit_medic.client.client import LocalDecoderClient
from qubit_medic.config import primary_level


def _summary(name: str, results: list[dict]) -> dict:
    """Aggregate per-episode reward dicts into the metrics the master spec
    benchmarks against (sections 6 + 7 of the locked spec).

    Each entry in ``results`` is the env's per-step ``info["rewards"]``
    dict, optionally with extra fields the eval loop decorated:
        * ``exact_match_pymatching`` (model-eval only)
        * ``output_length`` (model-eval only)
        * ``n_true_errors`` (any caller; enables hard-syndrome subset)
    """
    n = max(1, len(results))
    # Hard-syndrome subset = episodes where the simulated truth contains
    # at least 2 X|Z errors. This is the cohort where MWPM ambiguity
    # matters and trained-model contributions are most visible.
    hard = [r for r in results if int(r.get("n_true_errors", 0)) >= 2]
    n_hard = len(hard)
    out = {
        "name": name,
        "episodes": len(results),
        # Headline metrics (master spec, section 6).
        "logical_correction_rate":
            sum(r["logical_correction"] >= 0.5 for r in results) / n,
        "pymatching_beat_rate":
            sum(r["pymatching_beat"] >= 0.5 for r in results) / n,
        "format_compliance_rate":
            sum(r["format_compliance"] >= 0.999 for r in results) / n,
        "format_partial_rate":
            sum((r["format_compliance"] >= 0.5
                 and r["format_compliance"] < 0.999) for r in results) / n,
        # Continuous progress metrics.
        "syndrome_consistency_rate":
            sum(r["syndrome_consistency"] >= 0.999 for r in results) / n,
        "mean_syndrome_consistency":
            sum(r["syndrome_consistency"] for r in results) / n,
        "mean_hamming_overlap":
            sum(r["hamming_overlap"] for r in results) / n,
        "mean_total_reward":
            sum(r["total"] for r in results) / n,
        # Model-eval extras (present iff the model loop populated them).
        "exact_match_pymatching":
            sum(int(r.get("exact_match_pymatching", 0)) for r in results) / n,
        "mean_output_length":
            sum(int(r.get("output_length", 0)) for r in results) / n,
        # Hard-syndrome subset (FIX 5, 2026-04 eval spec). Easy syndromes
        # are where every baseline already hits ~95%+; the hard subset is
        # where differentiation actually shows up.
        "hard_syndrome_count": n_hard,
        "hard_syndrome_lcr":
            (sum(r["logical_correction"] >= 0.5 for r in hard) / n_hard
             if n_hard else 0.0),
        "hard_syndrome_beat_rate":
            (sum(r["pymatching_beat"] >= 0.5 for r in hard) / n_hard
             if n_hard else 0.0),
    }
    return out


def _eval_baseline(name: str, episodes: int, level: str,
                   collect_rows: bool = False):
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
    rows = []
    for ep in range(episodes):
        obs = client.reset(forced_level=level, seed=10_000 + ep)
        completion = pol(obs)
        result = client.step(raw_response=completion, episode_id=obs.episode_id)
        rwd = dict(result.info["rewards"])  # copy so we can decorate
        # Tag with true-error count so _summary can filter the hard subset.
        rwd["n_true_errors"] = (
            len(result.info.get("pymatching_x_errors", []) or [])
            + len(result.info.get("pymatching_z_errors", []) or [])
        )
        rewards.append(rwd)
        if collect_rows and ep < 50:  # cap table size
            rows.append({
                "episode": ep,
                "completion": completion,
                "logical_correction": rwd["logical_correction"],
                "syndrome_consistency": rwd["syndrome_consistency"],
                "hamming_overlap": rwd["hamming_overlap"],
                "format_compliance": rwd["format_compliance"],
                "pymatching_beat": rwd["pymatching_beat"],
                "total": rwd["total"],
                "actual_obs_flip": result.info["actual_observable_flip"],
                "pm_obs_flip": result.info["pymatching_observable_pred"],
            })
    return _summary(name, rewards), rows


def _eval_model(adapter: str, episodes: int, level: str,
                base_model: str, max_new_tokens: int,
                collect_rows: bool = False):
    """Use Unsloth to load the adapter and generate completions.

    Populates ``exact_match_pymatching`` and ``output_length`` on each
    per-episode reward dict so :func:`_summary` can report the master
    spec's full benchmark suite (section 6 + section 7).
    """
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
    rows = []
    for ep in range(episodes):
        obs = client.reset(forced_level=level, seed=10_000 + ep)
        chat = [{"role": "user", "content": obs.prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic / greedy eval
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
        n_tokens = int(gen_ids.shape[0])
        result = client.step(raw_response=completion, episode_id=obs.episode_id)
        rwd = dict(result.info["rewards"])  # copy so we can decorate

        # Decorate with the master-spec extras.
        action = result.info.get("parsed_action", {}) or {}
        pm_x = sorted(set(map(int, result.info.get("pymatching_x_errors", []) or [])))
        pm_z = sorted(set(map(int, result.info.get("pymatching_z_errors", []) or [])))
        our_x = sorted(set(map(int, action.get("x_error_qubits", []) or [])))
        our_z = sorted(set(map(int, action.get("z_error_qubits", []) or [])))
        rwd["exact_match_pymatching"] = int(
            bool(action.get("parse_success", False))
            and our_x == pm_x and our_z == pm_z
        )
        rwd["output_length"] = n_tokens
        rwd["n_true_errors"] = len(pm_x) + len(pm_z)
        rewards.append(rwd)

        if collect_rows and ep < 50:
            rows.append({
                "episode": ep,
                "completion": completion[:300],
                "logical_correction": rwd["logical_correction"],
                "syndrome_consistency": rwd["syndrome_consistency"],
                "hamming_overlap": rwd["hamming_overlap"],
                "format_compliance": rwd["format_compliance"],
                "pymatching_beat": rwd["pymatching_beat"],
                "exact_match_pymatching": rwd["exact_match_pymatching"],
                "output_length": rwd["output_length"],
                "total": rwd["total"],
                "actual_obs_flip": result.info["actual_observable_flip"],
                "pm_obs_flip": result.info["pymatching_observable_pred"],
            })
    return _summary(f"model[{adapter}]", rewards), rows


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
    parser.add_argument("--report-to", type=str, default="none",
                        choices=["wandb", "none"],
                        help="If 'wandb', log summary + per-episode table.")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=("eval",))
    parser.add_argument("--wandb-notes", type=str, default=None)
    args = parser.parse_args(list(argv))

    if (args.policy is None) == (args.adapter is None):
        print("ERROR: exactly one of --policy and --adapter is required",
              file=sys.stderr)
        return 1

    from qubit_medic import wandb_utils

    report_to = wandb_utils.derive_report_to(args.report_to)
    use_wandb = report_to == "wandb"
    if use_wandb:
        slug = args.policy or (args.adapter or "model").replace("/", "_")
        run_name = args.wandb_run_name or wandb_utils.make_run_name(
            "eval", suffix=slug)
        wandb_utils.init_run(
            run_name=run_name,
            job_type="eval",
            tags=tuple(list(args.wandb_tags) + [args.level]),
            notes=args.wandb_notes,
            group=args.wandb_group,
            extra_config={
                "cli": {
                    "policy": args.policy,
                    "adapter": args.adapter,
                    "episodes": args.episodes,
                    "level": args.level,
                    "max_new_tokens": args.max_new_tokens,
                    "base_model": args.base_model,
                },
            },
        )

    if args.policy is not None:
        result, rows = _eval_baseline(args.policy, args.episodes, args.level,
                                      collect_rows=use_wandb)
    else:
        result, rows = _eval_model(args.adapter, args.episodes, args.level,
                                   args.base_model, args.max_new_tokens,
                                   collect_rows=use_wandb)
    result["level"] = args.level
    print(json.dumps(result, indent=2))

    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

    if use_wandb:
        wandb_utils.log_eval_summary(result, prefix="eval")
        if rows:
            wandb_utils.log_generation_table(
                rows, step=None, table_name="eval/episode_breakdown",
            )
        wandb_utils.update_summary({
            "eval/policy_or_adapter": args.policy or args.adapter,
            "eval/episodes": args.episodes,
            "eval/level": args.level,
        })
        wandb_utils.finish_run()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
