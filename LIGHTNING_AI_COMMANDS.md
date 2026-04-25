# Lightning AI Commands (SFT + GRPO)

Use this guide to run Qubit-Medic training in Lightning AI Studio.

## 1) Create a Lightning Studio with GPU

- Choose a GPU machine (T4 is enough for this setup).
- Open terminal in the Studio.

## 2) Clone and enter repo

```bash
git clone https://github.com/Ronit-Raj9/Meta_RL_Phase2.git
cd Meta_RL_Phase2
```

## 3) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-train.txt
```

## 4) Login to W&B (recommended)

```bash
wandb login
```

## 5) Run full pipeline (one command)

```bash
bash scripts/run_lightning_pipeline.sh
```

This executes:

1. `python -m scripts.generate_sft_data`
2. `python -m scripts.train_sft ...`
3. `python -m scripts.train_grpo ...`
4. `python -m scripts.eval --adapter checkpoints/grpo_final --episodes 1000 --out data/eval_grpo.json`

## Optional flags

Skip W&B:

```bash
SKIP_WANDB=1 bash scripts/run_lightning_pipeline.sh
```

Custom W&B group:

```bash
GROUP=qubit-medic-lightning bash scripts/run_lightning_pipeline.sh
```

Custom final eval episode count:

```bash
EPISODES=500 bash scripts/run_lightning_pipeline.sh
```

## Resume behavior

- If `Meta_RL_Phase2/` already exists, the script reuses it.
- Checkpoints are written to:
  - `checkpoints/sft_warmup`
  - `checkpoints/grpo_final`

## Quick sanity checks

```bash
python -m scripts.eval --adapter checkpoints/sft_warmup --episodes 200 --out data/eval_sft_smoke.json
python -m scripts.eval --adapter checkpoints/grpo_final --episodes 200 --out data/eval_grpo_smoke.json
```
