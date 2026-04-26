# Remote eval against the live HF Space

Server: `https://ronitraj-quantumscribe.hf.space` (qubit-medic v1.0.0, stim 1.15.0, openenv 0.2.3)
Date: 2026-04-26
Run via: `python -m scripts.eval_remote --url … --all-policies --levels L1_warmup L2_target --episodes 100 --out-dir data/remote_eval/`

100 episodes per (policy, level) cell. Same JSON schema as `scripts/eval.py`, so `scripts/comparison_table.py` consumes these files unmodified (see `results/remote_eval_comparison.md`).

## Headline metrics

| Level | Policy | Logical correction | Format | PM beat | Mean reward | Throughput |
|---|---|---:|---:|---:|---:|---:|
| L1_warmup | zeros      | 97% | 100% | 0% | 0.832 | 1.75 ep/s |
| L1_warmup | random     | 59% | 100% | 0% | 0.469 | 1.68 ep/s |
| L1_warmup | pymatching | **99%** | 100% | 0% | **0.873** | 1.64 ep/s |
| L2_target | zeros      | 92% | 100% | 0% | 0.745 | 1.70 ep/s |
| L2_target | random     | 60% | 100% | 0% | 0.483 | 1.63 ep/s |
| L2_target | pymatching | **99%** | 100% | 0% | **0.874** | 1.72 ep/s |

## Reading

- **The Space scores correctly.** The PyMatching imitator hits ≥99% logical correction at both levels — this is the expected ceiling for the matching decoder at p=5e-4 / p=1e-3 distance-3 SI1000 noise. The zeros and random baselines fall in the right places relative to it.
- **Format compliance 100% across the board** confirms `parse_action()` accepts both the `<answer>X: … | Z: …</answer>` shape (zeros/random) and the bare `X_ERRORS=… Z_ERRORS=…` shape (pymatching imitator).
- **PyMatching beat-rate is 0% for every baseline by construction** — a model that beats this column is the actual differentiator. None of the baselines can; that's the gap a trained adapter has to fill.
- **Distance between zeros and pymatching is small at L1/L2 (97→99%, 92→99%)** because at p≤1e-3 most syndromes are clean; the all-zeros policy is hard to embarrass. Where pymatching pulls ahead is the continuous metrics: hamming overlap (0.87 → 0.97 at L1, 0.69 → 0.98 at L2) and syndrome consistency. These are the channels GRPO actually trains on.
- **Throughput is steady at ~1.7 ep/s** end-to-end including HTTP round-trip — the Space is not a bottleneck for an eval loop of this size.

## Files

- Per-cell JSON: `data/remote_eval/eval_remote_<policy>_<level>.json`
- Literature-comparison table: `results/remote_eval_comparison.md`
- Script: `scripts/eval_remote.py`

## Next step

To put the trained adapter (`ronitraj/quantumscribe`) in the **"Trained Qubit-Medic"** column of the literature table, run the existing `scripts/eval.py --adapter ronitraj/quantumscribe …` on a GPU host (Colab / Kaggle / Hub Inference). It writes the same JSON schema, so swap it into `--eval-json` of `scripts/comparison_table.py`.
