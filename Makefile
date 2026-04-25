# Convenience targets. Use `make` not `python -m ...` for the obvious flows.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install validate baselines sft-data plots animation tests serve gradio
.PHONY: all clean docker-build docker-run

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
