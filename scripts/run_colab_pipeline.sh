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
#   SFT_FOR_GRPO  (default: checkpoints/sft_warmup/checkpoint-50) — SFT
#            snapshot for GRPO init. If this directory exists and
#            FORCE_SFT=0, generate_sft_data and train_sft are skipped.
#   FORCE_SFT  (default: 0) — if 1, always run SFT first.

REPO_URL="${REPO_URL:-https://github.com/Ronit-Raj9/Meta_RL_Phase2.git}"
REPO_DIR="${REPO_DIR:-Meta_RL_Phase2}"
GROUP="${GROUP:-qubit-medic-final}"
EPISODES="${EPISODES:-1000}"
SKIP_WANDB="${SKIP_WANDB:-0}"
# Step-50 LoRA in repo is the default GRPO initialisation (see config.SFT_CHECKPOINT_PATH_FOR_GRPO).
SFT_FOR_GRPO="${SFT_FOR_GRPO:-checkpoints/sft_warmup/checkpoint-50}"
FORCE_SFT="${FORCE_SFT:-0}"
# Set to 1 when the caller (e.g. run_lightning_pipeline.sh) has already
# installed a CUDA-matched torch + unsloth stack and we MUST NOT let
# `pip install -r requirements-train.txt` re-resolve them (which would
# pull a default-PyPI torch wheel and break GPU detection on driver-550
# Lightning boxes).
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"

if [[ ! -d "${REPO_DIR}" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

if [[ "${SKIP_PIP_INSTALL}" != "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install -r requirements-train.txt

  # ------------------------------------------------------------------ #
  # GRPO compatibility pin (the reason this block exists)
  # ------------------------------------------------------------------ #
  # `requirements-train.txt` lists `unsloth` (unpinned). Pip resolves it
  # to unsloth==2025.11.1, whose metadata declares `unsloth_zoo>=2025.11.1`
  # with no upper bound. Pip therefore picks the latest unsloth_zoo
  # (2026.4.9 at time of writing) -- and that release renamed two
  # required positional args of `grpo_accumulated_loss`:
  #
  #     unsloth_zoo<=2025.11.1 :  old_hidden_states, ref_hidden_states
  #     unsloth_zoo>=2026.x    :  old_logps,         ref_logps
  #
  # unsloth==2025.11.1's GRPO patch still calls the function with the
  # OLD keyword names. The old kwargs slop into **kwargs and the new
  # required positional args are never supplied, so step 0 of GRPO
  # crashes with the cryptic:
  #
  #     TypeError: grpo_accumulated_loss() missing 2 required positional
  #     arguments: 'old_logps' and 'ref_logps'
  #
  # SFT does not exercise this code path, so SFT finishes cleanly; the
  # mismatch only bites the moment GRPO starts training. Fix: lock
  # unsloth_zoo to the matched 2025.11.1 release with --no-deps so pip
  # cannot re-resolve it upward, and force-reinstall so an existing
  # 2026.x install on the Colab box is actually replaced.
  # ------------------------------------------------------------------ #
  echo "[colab-pipeline] pinning unsloth==2025.11.1 + unsloth_zoo==2025.11.1 for GRPO compatibility"
  python -m pip install --no-deps --force-reinstall \
    "unsloth==2025.11.1" \
    "unsloth_zoo==2025.11.1"

  # Verify the pin actually stuck and the function signature is the
  # OLD one (old_hidden_states / ref_hidden_states). Refusing to
  # continue here costs a few seconds; missing the bug here costs
  # the full SFT runtime + a confusing TypeError at GRPO step 0.
  python - <<'PY'
import inspect, sys
import unsloth, unsloth_zoo
from unsloth_zoo.rl_replacements import grpo_accumulated_loss
params = list(inspect.signature(grpo_accumulated_loss).parameters.keys())
print(f"[colab-pipeline]   unsloth     = {unsloth.__version__}")
print(f"[colab-pipeline]   unsloth_zoo = {unsloth_zoo.__version__}")
print(f"[colab-pipeline]   grpo_accumulated_loss params = {params}")
if "old_hidden_states" not in params or "ref_hidden_states" not in params:
    sys.stderr.write(
        "[colab-pipeline] FATAL: unsloth_zoo pin did not stick.\n"
        "[colab-pipeline] grpo_accumulated_loss is missing old_hidden_states/\n"
        "[colab-pipeline] ref_hidden_states -- this is the signature that\n"
        "[colab-pipeline] crashes at GRPO step 0. Aborting BEFORE wasting GPU\n"
        "[colab-pipeline] minutes on SFT.\n"
    )
    sys.exit(20)
print("[colab-pipeline] unsloth/unsloth_zoo signatures match -- safe to train.")
PY

  # Wipe any stale unsloth_compiled_cache/ from a previous run. The
  # cache inlines `grpo_accumulated_loss` source via inspect.getsource
  # at the moment GRPO is first imported, so a cache generated against
  # the old unsloth_zoo==2026.4.9 install would still hold the broken
  # signature even after we fix the pin. Easier to just delete it.
  if [[ -d unsloth_compiled_cache ]]; then
    echo "[colab-pipeline] removing stale unsloth_compiled_cache/ so it regenerates against the pinned unsloth_zoo"
    rm -rf unsloth_compiled_cache
  fi
else
  echo "[colab-pipeline] SKIP_PIP_INSTALL=1 set; skipping pip install of requirements*.txt"
fi

if [[ "${SKIP_WANDB}" != "1" ]]; then
  wandb login
  REPORT_TO="wandb"
else
  REPORT_TO="none"
fi

if [[ "${FORCE_SFT}" == "1" ]] || [[ ! -d "${SFT_FOR_GRPO}" ]]; then
  python -m scripts.generate_sft_data
  python -m scripts.train_sft \
    --dataset data/sft_dataset.jsonl \
    --val-dataset data/sft_validation.jsonl \
    --output checkpoints/sft_warmup \
    --report-to "${REPORT_TO}" \
    --wandb-group "${GROUP}"
else
  echo "[colab-pipeline] found ${SFT_FOR_GRPO}; skipping generate_sft_data and train_sft (FORCE_SFT=1 to run SFT anyway)"
fi

# ------------------------------------------------------------------ #
# Diversity preflight gate (2026-04 spec, FIX 2)
# ------------------------------------------------------------------ #
# Before letting GRPO touch the GPU, verify the SFT model still
# produces diverse outputs at temperature=1.2. If every checkpoint
# converged to one canonical completion (output_diversity=1), GRPO
# cannot compute non-zero within-group reward variance, so the
# advantage vanishes and the model just sits.
#
# On FAIL the pipeline re-runs SFT ONCE with --lora-dropout 0.15
# (an extra step beyond the new 0.10 default) and re-checks. Two
# consecutive failures abort the pipeline so we don't burn GPU time
# on a model that GRPO can't recover.
# ------------------------------------------------------------------ #
echo "[colab-pipeline] running diversity preflight against checkpoints/sft_warmup"
PREFLIGHT_RC=0
python -m scripts.diversity_preflight \
  --sft-checkpoint checkpoints/sft_warmup \
  --val data/sft_validation.jsonl \
  || PREFLIGHT_RC=$?

if [[ "${PREFLIGHT_RC}" != "0" ]]; then
  echo "[colab-pipeline] diversity preflight FAILED (rc=${PREFLIGHT_RC}); re-running SFT with --lora-dropout 0.15"
  python -m scripts.train_sft \
    --dataset data/sft_dataset.jsonl \
    --val-dataset data/sft_validation.jsonl \
    --output checkpoints/sft_warmup \
    --lora-dropout 0.15 \
    --report-to "${REPORT_TO}" \
    --wandb-group "${GROUP}"

  echo "[colab-pipeline] re-running diversity preflight after dropout-0.15 SFT retry"
  PREFLIGHT_RC=0
  python -m scripts.diversity_preflight \
    --sft-checkpoint checkpoints/sft_warmup \
    --val data/sft_validation.jsonl \
    || PREFLIGHT_RC=$?
  if [[ "${PREFLIGHT_RC}" != "0" ]]; then
    echo "[colab-pipeline] FATAL: diversity preflight FAILED again after retry."
    echo "[colab-pipeline] Refusing to launch GRPO on a collapsed model."
    echo "[colab-pipeline] Investigate the SFT data distribution / loss masking"
    echo "[colab-pipeline] before running this pipeline again."
    exit 30
  fi
fi
echo "[colab-pipeline] diversity preflight PASSED; launching GRPO"

python -m scripts.train_grpo \
  --sft-checkpoint "${SFT_FOR_GRPO}" \
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
