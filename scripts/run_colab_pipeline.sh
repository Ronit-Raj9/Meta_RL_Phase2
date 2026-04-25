#!/usr/bin/env bash
set -euo pipefail

# End-to-end Colab pipeline runner for Qubit-Medic.
#
# Usage (inside Colab or any Linux box with GPU):
#   bash scripts/run_colab_pipeline.sh
#
# Optional environment variables:
#   REPO_URL   (default: https://github.com/Ronit-Raj9/Meta_RL_Phase2.git)
#   REPO_DIR   (default: Meta_RL_Phase2)
#   GROUP      (default: qubit-medic-final)
#   EPISODES   (default: 1000)
#   SKIP_WANDB (set to 1 to disable wandb login prompt/reporting)

REPO_URL="${REPO_URL:-https://github.com/Ronit-Raj9/Meta_RL_Phase2.git}"
REPO_DIR="${REPO_DIR:-Meta_RL_Phase2}"
GROUP="${GROUP:-qubit-medic-final}"
EPISODES="${EPISODES:-1000}"
SKIP_WANDB="${SKIP_WANDB:-0}"
# Set to 1 when the caller (e.g. run_lightning_pipeline.sh) has already
# installed a CUDA-matched torch + unsloth stack and we MUST NOT let
# `pip install -r requirements-train.txt` re-resolve them (which would
# pull a default-PyPI torch wheel and break GPU detection on driver-550
# Lightning boxes).
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"

# If the caller is already inside a clone of this repo (e.g. they ran
# `git clone -b murtuza ... && cd Meta_RL_Phase2 && bash scripts/run_colab_pipeline.sh`),
# stay put -- otherwise we'd shadow their checkout with a fresh `main` clone
# nested at Meta_RL_Phase2/Meta_RL_Phase2/, silently running the wrong code.
if [[ -d ".git" ]]; then
  echo "[colab-pipeline] already inside repo at $(pwd); skipping clone"
elif [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
else
  git clone "${REPO_URL}" "${REPO_DIR}"
  cd "${REPO_DIR}"
fi

if [[ "${SKIP_PIP_INSTALL}" != "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install -r requirements-train.txt
else
  echo "[colab-pipeline] SKIP_PIP_INSTALL=1 set; skipping pip install of requirements*.txt"
fi

if [[ "${SKIP_WANDB}" != "1" ]]; then
  wandb login
  REPORT_TO="wandb"
else
  REPORT_TO="none"
fi

python -m scripts.generate_sft_data

python -m scripts.train_sft \
  --dataset data/sft_dataset.jsonl \
  --val-dataset data/sft_validation.jsonl \
  --output checkpoints/sft_warmup \
  --report-to "${REPORT_TO}" \
  --wandb-group "${GROUP}"

python -m scripts.train_grpo \
  --sft-checkpoint checkpoints/sft_warmup \
  --output checkpoints/grpo_final \
  --report-to "${REPORT_TO}" \
  --wandb-group "${GROUP}"

python -m scripts.eval \
  --adapter checkpoints/grpo_final \
  --episodes "${EPISODES}" \
  --out data/eval_grpo.json

echo
echo "Pipeline complete."
echo "Final eval file: ${PWD}/data/eval_grpo.json"
