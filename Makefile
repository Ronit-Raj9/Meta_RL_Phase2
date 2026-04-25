# Convenience targets. Use `make` not `python -m ...` for the obvious flows.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install validate baselines sft-data plots animation tests serve gradio
.PHONY: all clean docker-build docker-run
.PHONY: wandb-login train-sft train-grpo eval eval-baseline format-test

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

validate:
	$(PY) -m scripts.validate_env

baselines:
	$(PY) -m scripts.baseline_policies --episodes 500 --out data/baseline_results.json

sft-data:
	$(PY) -m scripts.generate_sft_data --n 5000

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
