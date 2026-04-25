# Colab Commands (SFT + GRPO)

Use this if you want copy-paste cells or one-shot script execution.

## Option A: Run step-by-step in cells

```bash
!git clone https://github.com/Ronit-Raj9/Meta_RL_Phase2.git
%cd Meta_RL_Phase2
```

```bash
!pip install -r requirements.txt
!pip install -r requirements-train.txt
```

```bash
!wandb login
```

```bash
!python -m scripts.generate_sft_data
```

```bash
!python -m scripts.train_sft \
  --dataset data/sft_dataset.jsonl \
  --val-dataset data/sft_validation.jsonl \
  --output checkpoints/sft_warmup \
  --report-to wandb \
  --wandb-group qubit-medic-final
```

```bash
!python -m scripts.train_grpo \
  --sft-checkpoint checkpoints/sft_warmup \
  --output checkpoints/grpo_final \
  --report-to wandb \
  --wandb-group qubit-medic-final
```

```bash
!python -m scripts.eval \
  --adapter checkpoints/grpo_final \
  --episodes 1000 \
  --out data/eval_grpo.json
```

## Option B: One-shot script

After clone:

```bash
%cd Meta_RL_Phase2
!bash scripts/run_colab_pipeline.sh
```

If you want to skip W&B:

```bash
!SKIP_WANDB=1 bash scripts/run_colab_pipeline.sh
```

If you want custom W&B group or eval episodes:

```bash
!GROUP=my-run EPISODES=500 bash scripts/run_colab_pipeline.sh
```
