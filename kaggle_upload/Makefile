# Convenience targets. Use `make` not `python -m ...` for the obvious flows.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install validate baselines sft-data plots animation tests serve gradio
.PHONY: all clean docker-build docker-run
.PHONY: wandb-login train-sft train-grpo eval eval-baseline format-test
.PHONY: kaggle-sync kaggle-push-create kaggle-push kaggle-status kaggle-download kaggle-help
.PHONY: kaggle-kernel-status kaggle-kernel-output kaggle-auth-test

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

validate:
	$(PY) -m scripts.validate_env

baselines:
	$(PY) -m scripts.baseline_policies --episodes 500 --out data/baseline_results.json

sft-data:
	$(PY) -m scripts.generate_sft_data

plots:
	$(PY) -m scripts.plot_results --baselines data/baseline_results.json --out-dir figures

animation:
	$(PY) -m scripts.animate_grid --frames 30

tests:
	$(PY) -m pytest tests/ -v

serve:
	$(PY) -m qubit_medic.server.app

gradio:
	$(PY) app_gradio.py

# All-in-one: validate, baselines, sft data, plots, animation, tests.
all: validate baselines sft-data plots animation tests

clean:
	rm -rf data/sft_dataset.jsonl data/baseline_results.json
	rm -rf figures/grid_animation.gif figures/grid_hero.png
	rm -rf figures/total_reward.png figures/logical_correction.png
	rm -rf figures/pymatching_beat_rate.png

docker-build:
	docker build -t qubit-medic:latest .

docker-run:
	docker run --rm -p 7860:7860 qubit-medic:latest

# ---- W&B-aware training shortcuts ---------------------------------------- #
# Set WANDB_DISABLED=1 to skip W&B; otherwise these expect `wandb login` to
# have been run once. Use GROUP=my-experiment to bundle SFT+GRPO+eval runs.

GROUP ?= local-$(shell date +%Y%m%d-%H%M%S)

wandb-login:
	$(PY) -m wandb login

format-test:
	$(PY) -m scripts.format_test --backend dummy --report-to wandb \
	    --wandb-group $(GROUP) --out data/format_test.json

train-sft:
	$(PY) -m scripts.train_sft --report-to wandb \
	    --wandb-group $(GROUP) \
	    --output checkpoints/sft_warmup

train-grpo:
	$(PY) -m scripts.train_grpo --report-to wandb \
	    --wandb-group $(GROUP) \
	    --sft-checkpoint checkpoints/sft_warmup \
	    --output checkpoints/grpo

eval:
	$(PY) -m scripts.eval --adapter checkpoints/grpo --episodes 500 \
	    --report-to wandb --wandb-group $(GROUP) \
	    --out data/grpo_eval.json

eval-baseline:
	$(PY) -m scripts.eval --policy pymatching --episodes 500 \
	    --report-to wandb --wandb-group $(GROUP) \
	    --out data/baseline_eval.json

# ---- HF Spaces deployment ----------------------------------------------- #
# Set REPO=your-username/qubit-medic before running.

REPO ?= ronitraj/QuantumScribe
SPACE_URL ?= https://ronitraj-quantumscribe.hf.space

deploy-placeholder:
	$(PY) -m scripts.deploy_to_space --repo $(REPO) --placeholder

deploy:
	$(PY) -m scripts.deploy_to_space --repo $(REPO)

hello-local:
	$(PY) -m scripts.hello_space

wakeup:
	$(PY) -m scripts.wakeup_space --url $(SPACE_URL)

healthz:
	@curl -fsSL $(SPACE_URL)/healthz && echo

# ---- Kaggle Datasets (code sync for notebooks / Colab) ------------------- #
# Publish the repo as a Kaggle Dataset so `kaggle datasets download` always
# pulls the latest sources. One-time: ~/.kaggle/kaggle.json (see kaggle/README.md).
#
# Optional: use a conda env for the CLI:
#   make kaggle-push MSG='...' KAGGLE_CONDA_ENV=aisehack
#
# Override dataset id when syncing:
#   KAGGLE_DATASET_ID=username/my-slug make kaggle-sync

KAGGLE_DATASET ?= ronitraj1/qubit-medic-code
KAGGLE_CONDA_ENV ?=
KAGGLE_DOWNLOAD_DIR ?= kaggle_download
# Prefer the same venv as $(PY) so `make kaggle-*` works without activating .venv.
KAGGLE_BIN ?= $(dir $(PY))kaggle

ifeq ($(strip $(KAGGLE_CONDA_ENV)),)
KAGGLE := $(KAGGLE_BIN)
else
KAGGLE := conda run -n $(KAGGLE_CONDA_ENV) kaggle
endif

kaggle-help:
	@echo "Kaggle code-dataset workflow:"
	@echo "  make kaggle-sync              # populate kaggle_upload/"
	@echo "  make kaggle-push-create       # first publish (datasets create)"
	@echo "  make kaggle-push MSG='...'    # upload a new version"
	@echo "  make kaggle-status            # remote status"
	@echo "  make kaggle-download          # fetch latest zip to $(KAGGLE_DOWNLOAD_DIR)/"
	@echo "  make kaggle-kernel-status     # needs KAGGLE_KERNEL=owner/slug"
	@echo "  make kaggle-kernel-output     # download notebook output files"
	@echo "Variables: KAGGLE_DATASET=$(KAGGLE_DATASET)  KAGGLE_CONDA_ENV=$(KAGGLE_CONDA_ENV)"
	@echo "  KAGGLE_BIN=$(KAGGLE_BIN) (override with KAGGLE_BIN=/path/to/kaggle)"

# Quick auth check — 401 on push means fix ~/.kaggle/kaggle.json (see kaggle/README.md)
kaggle-auth-test:
	@echo "Using: $(KAGGLE)"
	@$(KAGGLE) datasets list --mine 2>&1 | head -8

kaggle-sync:
	KAGGLE_DATASET_ID=$(KAGGLE_DATASET) $(PY) -m scripts.sync_kaggle_bundle

kaggle-push-create: kaggle-sync
	$(KAGGLE) datasets create -p kaggle_upload --dir-mode zip
	@echo "Dataset: https://www.kaggle.com/datasets/$(KAGGLE_DATASET)"

kaggle-push:
	@if [ -z "$(MSG)" ]; then echo "MSG is required: make kaggle-push MSG='describe change'"; exit 1; fi
	$(MAKE) kaggle-sync
	$(KAGGLE) datasets version -p kaggle_upload --dir-mode zip -m "$(MSG)"
	@echo "https://www.kaggle.com/datasets/$(KAGGLE_DATASET)"

kaggle-status:
	$(KAGGLE) datasets status $(KAGGLE_DATASET)

kaggle-download:
	mkdir -p $(KAGGLE_DOWNLOAD_DIR)
	$(KAGGLE) datasets download $(KAGGLE_DATASET) -p $(KAGGLE_DOWNLOAD_DIR) --unzip
	@echo "Extracted under $(KAGGLE_DOWNLOAD_DIR)/"

# Optional: Kaggle *notebook* (kernel) status — set KAGGLE_KERNEL=owner/slug
KAGGLE_KERNEL ?=

kaggle-kernel-status:
	@if [ -z "$(KAGGLE_KERNEL)" ]; then echo "Set KAGGLE_KERNEL=owner/notebook-slug (e.g. ronitraj1/my-kernel)"; exit 1; fi
	$(KAGGLE) kernels status $(KAGGLE_KERNEL)

kaggle-kernel-output:
	@if [ -z "$(KAGGLE_KERNEL)" ]; then echo "Set KAGGLE_KERNEL=owner/notebook-slug"; exit 1; fi
	mkdir -p kaggle_kernel_output
	$(KAGGLE) kernels output $(KAGGLE_KERNEL) -p kaggle_kernel_output/
	@echo "Files in kaggle_kernel_output/"
