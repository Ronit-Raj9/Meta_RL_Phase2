#!/usr/bin/env bash
set -euo pipefail

# End-to-end Lightning AI runner for Qubit-Medic.
#
# Lightning Studios (H100 / A10G boxes, as of 2026-04) ship with NVIDIA
# driver 550.x, which supports CUDA up to 12.4. The default `torch` wheel
# on PyPI is currently built against a newer CUDA runtime (12.6+) that
# needs driver 560+, so a plain `pip install torch` produces:
#
#   "The NVIDIA driver on your system is too old (found version 12040)"
#   torch.cuda.is_available() -> False
#
# and Unsloth then fails its own GPU probe:
#
#   "Unsloth cannot find any torch accelerator? You need a GPU."
#
# Fix: pin torch to the cu124 wheel index BEFORE running the rest of the
# install. Once torch is already satisfied with the correct CUDA build,
# `pip install -r requirements-train.txt` (and unsloth's own deps) leaves
# it alone, and the GPU is detected correctly.
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
#   TORCH_CUDA=cu124      # override only if Lightning upgrades the driver
#   TORCH_VERSION=2.5.1   # latest version with stable cu124 wheels for unsloth
#   SKIP_TORCH_INSTALL=0  # set to 1 if you've already installed a CUDA-matched torch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# If script is already running from inside the repo, avoid re-cloning.
if [[ -f "${ROOT_DIR}/requirements.txt" && -d "${ROOT_DIR}/qubit_medic" ]]; then
  export REPO_DIR="${ROOT_DIR}"
fi

TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL:-0}"

if [[ "${SKIP_TORCH_INSTALL}" != "1" ]]; then
  echo "[lightning] upgrading pip"
  python -m pip install --upgrade pip

  echo "[lightning] installing torch==${TORCH_VERSION}+${TORCH_CUDA} from PyTorch's CUDA-matched wheel index"
  echo "[lightning]   (Lightning Studios ship NVIDIA driver 550.x / CUDA 12.4;"
  echo "[lightning]    the default PyPI torch wheel needs driver 560+ and will not init CUDA)"
  python -m pip install \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    "torch==${TORCH_VERSION}" "torchvision" "torchaudio"
fi

echo "[lightning] verifying torch can see the GPU before delegating to the colab pipeline"
python - <<'PY'
import sys
import torch
print(f"[lightning] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[lightning] device={torch.cuda.get_device_name(0)}  "
          f"runtime_cuda={torch.version.cuda}")
else:
    sys.stderr.write(
        "[lightning] FATAL: torch still cannot init CUDA after the cu124 install.\n"
        "[lightning] If `nvidia-smi` shows a driver < 12.4, ask Lightning to bump\n"
        "[lightning] the studio image. Otherwise re-run with TORCH_CUDA matching\n"
        "[lightning] your driver (e.g. TORCH_CUDA=cu121 for driver 535.x).\n"
    )
    sys.exit(2)
PY

bash "${SCRIPT_DIR}/run_colab_pipeline.sh"
