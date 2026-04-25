#!/usr/bin/env bash
set -euo pipefail

# End-to-end Lightning AI runner for Qubit-Medic.
#
# This is a thin wrapper around scripts/run_colab_pipeline.sh with defaults
# that fit Lightning Studio / Lightning notebooks.
#
# Usage:
#   bash scripts/run_lightning_pipeline.sh
#
# Optional env vars:
#   GROUP=qubit-medic-lightning
#   EPISODES=1000
#   SKIP_WANDB=0
#   REPO_URL=https://github.com/Ronit-Raj9/Meta_RL_Phase2.git
#   REPO_DIR=Meta_RL_Phase2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# If script is already running from inside the repo, avoid re-cloning.
if [[ -f "${ROOT_DIR}/requirements.txt" && -d "${ROOT_DIR}/qubit_medic" ]]; then
  export REPO_DIR="${ROOT_DIR}"
fi

bash "${SCRIPT_DIR}/run_colab_pipeline.sh"
