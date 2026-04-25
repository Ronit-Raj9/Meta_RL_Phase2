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
# Why we have to install the GPU stack ourselves (cannot just rely on
# `pip install -r requirements-train.txt`):
#
#   * `requirements-train.txt` lists `unsloth` (unpinned).
#   * Recent unsloth releases depend on `xformers>=0.0.35`, which itself
#     requires `torch>=2.10`.
#   * The latest torch on PyPI (>=2.10) is built against CUDA 12.6+ and
#     needs NVIDIA driver 560+. Lightning ships driver 550 / CUDA 12.4.
#   * So a vanilla `pip install -r requirements-train.txt` SILENTLY
#     UPGRADES torch from 2.5.1+cu124 to 2.11.0 (default PyPI wheel) and
#     re-breaks GPU detection -- exactly what happened in your last run:
#         "Successfully installed ... torch-2.11.0 ..."
#         "torchaudio 2.5.1+cu124 requires torch==2.5.1, but you have
#          torch 2.11.0 which is incompatible"
#         "Unsloth cannot find any torch accelerator? You need a GPU."
#
# So this script does the install in the right order and tells the colab
# pipeline to skip its own pip step (SKIP_PIP_INSTALL=1):
#
#   1. torch 2.5.1 + torchvision + torchaudio from PyTorch's cu124 index
#   2. xformers 0.0.28.post3 from the same cu124 index (the build matched
#      to torch 2.5.1; later xformers releases bump torch to >=2.10)
#   3. unsloth + unsloth_zoo with `--no-deps` so they CANNOT pull a newer
#      xformers / torch (this is what the Unsloth Colab notebook does)
#   4. everything else from requirements-train.txt EXCEPT unsloth (we
#      already installed it manually with --no-deps above)
#   5. project runtime deps from requirements.txt
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
#   TORCH_CUDA=cu124           # override only if Lightning upgrades the driver
#   TORCH_VERSION=2.5.1        # paired with XFORMERS_VERSION below
#   XFORMERS_VERSION=0.0.28.post3   # the build matched to torch 2.5.1+cu124
#   SKIP_TORCH_INSTALL=0       # set to 1 if you've already installed a CUDA-matched torch + unsloth

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# If script is already running from inside the repo, avoid re-cloning.
if [[ -f "${ROOT_DIR}/requirements.txt" && -d "${ROOT_DIR}/qubit_medic" ]]; then
  export REPO_DIR="${ROOT_DIR}"
fi

TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
XFORMERS_VERSION="${XFORMERS_VERSION:-0.0.28.post3}"
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL:-0}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

if [[ "${SKIP_TORCH_INSTALL}" != "1" ]]; then
  echo "[lightning] upgrading pip"
  python -m pip install --upgrade pip

  # Show what's currently installed so we can spot a mismatched CUDA build.
  CURRENT_TORCH="$(python - <<'PY'
try:
    import torch
    print(f"{torch.__version__}|{torch.version.cuda}|{torch.cuda.is_available()}")
except Exception as exc:
    print(f"none|none|{exc}")
PY
)"
  echo "[lightning] currently installed torch: ${CURRENT_TORCH}  (version|runtime_cuda|cuda_available)"

  # ALWAYS uninstall a wrong-CUDA / non-functional torch (and anything
  # that pinned to it: torchvision, torchaudio, xformers, unsloth's
  # private deps). Without this, `pip install torch==X` from the cu124
  # index is a no-op when pip thinks torch==X (any CUDA build) is already
  # installed -- which is exactly how a previous run of this script left
  # you with torch-2.11.0 (default PyPI wheel, CUDA 12.6+).
  if [[ "${CURRENT_TORCH}" != none\|* && "${CURRENT_TORCH}" != *\|True ]]; then
    echo "[lightning] existing torch cannot init CUDA on this driver; force-uninstalling torch + xformers + unsloth"
    python -m pip uninstall -y torch torchvision torchaudio xformers unsloth unsloth_zoo || true
  fi

  echo "[lightning] (1/4) installing torch==${TORCH_VERSION}+${TORCH_CUDA} from PyTorch's CUDA-matched wheel index"
  echo "[lightning]      (Lightning ships NVIDIA driver 550 / CUDA 12.4; default PyPI torch needs driver 560+)"
  python -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" "torchvision" "torchaudio"

  echo "[lightning] (2/4) installing xformers==${XFORMERS_VERSION} (built against torch ${TORCH_VERSION}+${TORCH_CUDA})"
  python -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "xformers==${XFORMERS_VERSION}"

  echo "[lightning] (3/4) installing unsloth + unsloth_zoo with --no-deps"
  echo "[lightning]      (this is the same trick the Unsloth Colab notebook uses, so unsloth"
  echo "[lightning]       cannot drag in a newer xformers / torch and undo step 1+2)"
  python -m pip install --no-deps unsloth unsloth_zoo

  echo "[lightning] (4/4) installing the rest of requirements-train.txt EXCEPT torch / unsloth"
  # Strip torch* and unsloth* lines from requirements-train.txt before
  # piping into pip. We already pinned both manually above; letting pip
  # see them again risks a re-resolve.
  TRAIN_REQS_FILTERED="$(mktemp)"
  trap 'rm -f "${TRAIN_REQS_FILTERED}"' EXIT
  grep -Ev '^\s*(torch|torchvision|torchaudio|xformers|unsloth(_zoo)?)\b' \
    "${REPO_DIR:-${ROOT_DIR}}/requirements-train.txt" \
    > "${TRAIN_REQS_FILTERED}" || true
  python -m pip install -r "${TRAIN_REQS_FILTERED}"

  echo "[lightning] installing project runtime deps from requirements.txt"
  python -m pip install -r "${REPO_DIR:-${ROOT_DIR}}/requirements.txt"
fi

echo "[lightning] verifying torch + unsloth before delegating to the colab pipeline"
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
try:
    import xformers
    print(f"[lightning] xformers={xformers.__version__}")
except ImportError as exc:
    sys.stderr.write(f"[lightning] FATAL: xformers import failed: {exc}\n")
    sys.exit(3)
try:
    from unsloth import FastLanguageModel  # noqa: F401
    print("[lightning] unsloth ok")
except Exception as exc:
    sys.stderr.write(f"[lightning] FATAL: unsloth import failed: {exc}\n")
    sys.exit(4)
PY

# We just installed everything; tell the colab pipeline NOT to redo it
# (otherwise its `pip install -r requirements-train.txt` would re-pull
# unsloth-with-deps and undo our careful pinning).
SKIP_PIP_INSTALL=1 bash "${SCRIPT_DIR}/run_colab_pipeline.sh"
