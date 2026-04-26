# Evaluation Protocol

This document describes the held-out evaluation protocol used to populate [`results/comparison_table.md`](../results/comparison_table.md). It was split out of the main README for readability — see the project [README](../README.md) for the high-level overview and headline numbers.

## Episode budget, seeds, and reproducibility commands

End-to-end evaluation protocol used for the figures in [results/comparison_table.md](../results/comparison_table.md). To reproduce, see "Reproducibility commands" below.

### Episode budget

| Cohort | Cells | Episodes / cell | Total |
|---|---|---|---|
| Trained model (SFT-only + SFT+RL × 4 levels) | 8 | 500 | **4,000** |
| Baselines (zeros / random / pymatching × 4 levels) | 12 | 100 | **1,200** |
| **Total** | 20 | — | **5,200 evaluation episodes** |

(The headline 3,200 figure is for a single-adapter run: 2,000 trained + 1,200 baseline.)

### Random seeds

Eval seed range: **5000 – 7199** (held out from training seeds 1–4999 and SFT-validation seeds 4242 + offset). Each (policy, level) cell uses contiguous seeds from this range, so results are bitwise reproducible.

### Confidence intervals

At 500 episodes per cell, a 95% Wilson CI on a 0.85-LCR estimate is approximately **±2.5%**. Baseline cells at 100 episodes carry a wider ±5% CI — they are deliberately cheaper because the metrics there (≥90% LCR for PyMatching, ~95%+ on L1/L2) are well-separated from the trained-model regime where the improvement is tested.

### Hard-syndrome subset definition

A "hard syndrome" is an evaluation episode where the **simulated true error pattern contains ≥ 2 X|Z error qubits**. Easy syndromes (zero or one error) are where every reasonable decoder hits ~95%+ LCR; the hard subset is the cohort where MWPM ambiguity matters and trained-model contributions are most visible. The subset metric is reported as `hard_syndrome_lcr` in each per-cell JSON.

### Curriculum levels (noise-model parameters)

Defined in [`qubit_medic/config.py:CURRICULUM`](../qubit_medic/config.py). All levels use the rotated surface code with a Z-memory experiment under the SI1000 noise model (Gidney & Fowler 2021).

| Level | Distance | Rounds | Physical error rate `p` | Notes |
|---|---|---|---|---|
| `L1_warmup` | 3 | 1 | 0.0005 | trivial; warmup |
| `L2_target` | 3 | 3 | 0.001 | primary benchmark (AlphaQubit Fig. 2b geometry) |
| `L3_stretch` | 5 | 5 | 0.001 | distance-5 stretch goal |
| `L4_stress` | 5 | 5 | 0.005 | 5× higher noise; eval-only stress test where baselines drop and headroom opens |

### Deployed environment

Live OpenEnv server: **[https://ronitraj-quantumscribe.hf.space](https://ronitraj-quantumscribe.hf.space)** — health probe at `/healthz`. The deployed Space currently knows L1/L2/L3 only; `L4_stress` evaluation runs locally via `scripts/eval.py` against the in-process `DecoderEnvironment`.

### Reproducibility commands

End-to-end (12 baseline cells + 4 trained-model cells + table generation) — run from the repo root:

```bash
SPACE_URL=https://ronitraj-quantumscribe.hf.space \
ADAPTER=checkpoints/grpo_v2 \
TRAINED_EPISODES=500 BASELINE_EPISODES=100 \
bash scripts/run_full_eval.sh
```

Outputs:
- `data/remote_eval/eval_remote_{policy}_{level}.json` — 12 baseline cells
- `data/trained_eval/eval_trained_{level}.json` — 4 trained-model cells
- `results/comparison_table.md` — final pivot table

Individual steps if you only need to refresh part of the matrix:

```bash
# Remote baselines on L1/L2/L3 only (Space-known levels)
python -m scripts.eval_remote --url https://ronitraj-quantumscribe.hf.space \
    --episodes 100 --levels L1_warmup L2_target L3_stretch \
    --all-policies --out-dir data/remote_eval/

# L4_stress baselines (local; Space rejects forced_level=L4_stress until redeployed)
for policy in zeros random pymatching; do
    python -m scripts.eval --policy $policy --episodes 100 \
        --level L4_stress \
        --out data/remote_eval/eval_remote_${policy}_L4_stress.json
done

# Trained-model evaluation (local; needs GPU)
for level in L1_warmup L2_target L3_stretch L4_stress; do
    python -m scripts.eval --adapter checkpoints/grpo_v2 \
        --episodes 500 --level $level \
        --out data/trained_eval/eval_trained_${level}.json
done

# Build the comparison table from whatever cells are present
python -m scripts.comparison_table_full \
    --remote-eval-dir data/remote_eval/ \
    --trained-eval-dir data/trained_eval/ \
    --output results/comparison_table.md
```

The runner is idempotent — `SKIP_BASELINES=1` reuses existing baseline JSONs; `SKIP_TRAINED=1` reuses existing trained-model JSONs.

