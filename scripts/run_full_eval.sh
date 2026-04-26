#!/usr/bin/env bash
set -euo pipefail
#
# 2026-04 evaluation spec — chained runner that produces every cell the
# comparison table needs:
#
#   Remote baselines (3 policies × 3 levels)        -> 9 cells
#       on the deployed HF Space at $SPACE_URL
#   Local baselines for L4_stress (3 policies × 1)  -> 3 cells
#       L4_stress is post-deployment; the Space rejects forced_level=L4
#   Trained model (4 levels × 1)                    -> 4 cells × 500 eps
#       runs locally via scripts.eval --adapter
#
# Total: 12 baseline cells (1200 eps) + 4 trained-model cells (2000 eps)
#        = 3200 evaluation episodes.
#
# Outputs:
#   data/remote_eval/eval_remote_{policy}_{level}.json   (12 files)
#   data/trained_eval/eval_trained_{level}.json          (4 files)
#   results/comparison_table.md                          (final pivot)
#
# Optional env vars:
#   SPACE_URL  (default: https://ronitraj-quantumscribe.hf.space)
#   ADAPTER    (default: checkpoints/grpo_v2)
#   BASELINE_EPISODES (default: 100)
#   TRAINED_EPISODES  (default: 500)
#   SKIP_BASELINES   (set 1 to skip baseline eval cells; useful for re-runs)
#   SKIP_TRAINED     (set 1 to skip the trained-model eval cells)

SPACE_URL="${SPACE_URL:-https://ronitraj-quantumscribe.hf.space}"
ADAPTER="${ADAPTER:-checkpoints/grpo_v2}"
BASELINE_EPISODES="${BASELINE_EPISODES:-100}"
TRAINED_EPISODES="${TRAINED_EPISODES:-500}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"
SKIP_TRAINED="${SKIP_TRAINED:-0}"

mkdir -p data/remote_eval data/trained_eval results

if [[ "${SKIP_BASELINES}" != "1" ]]; then
  echo "[full-eval] Step 1/4: remote baselines on L1/L2/L3 (3 policies × 3 levels)"
  python -m scripts.eval_remote \
    --url "${SPACE_URL}" \
    --episodes "${BASELINE_EPISODES}" \
    --levels L1_warmup L2_target L3_stretch \
    --all-policies \
    --out-dir data/remote_eval/

  echo "[full-eval] Step 2/4: local baselines on L4_stress (3 policies × 1 level)"
  echo "[full-eval]   (Space does not know L4_stress; local DecoderClient + new"
  echo "[full-eval]    CurriculumLevel in qubit_medic/config.py handles it)"
  for policy in zeros random pymatching; do
    python -m scripts.eval \
      --policy "${policy}" \
      --episodes "${BASELINE_EPISODES}" \
      --level L4_stress \
      --out "data/remote_eval/eval_remote_${policy}_L4_stress.json"
  done
else
  echo "[full-eval] SKIP_BASELINES=1; reusing existing data/remote_eval/*.json"
fi

if [[ "${SKIP_TRAINED}" != "1" ]]; then
  echo "[full-eval] Step 3/4: trained-model eval (4 levels × ${TRAINED_EPISODES} eps each)"
  for level in L1_warmup L2_target L3_stretch L4_stress; do
    echo "[full-eval]   trained @ ${level} ..."
    python -m scripts.eval \
      --adapter "${ADAPTER}" \
      --episodes "${TRAINED_EPISODES}" \
      --level "${level}" \
      --out "data/trained_eval/eval_trained_${level}.json"
  done
else
  echo "[full-eval] SKIP_TRAINED=1; reusing existing data/trained_eval/*.json"
fi

echo "[full-eval] Step 4/4: building comparison table"
python -m scripts.comparison_table_full \
  --remote-eval-dir data/remote_eval/ \
  --trained-eval-dir data/trained_eval/ \
  --output results/comparison_table.md

echo
echo "[full-eval] DONE. See:"
echo "  - data/remote_eval/  (baseline cells)"
echo "  - data/trained_eval/ (trained-model cells)"
echo "  - results/comparison_table.md"
