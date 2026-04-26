---
title: Qubit-Medic
emoji: 🩺
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - quantum-error-correction
  - stim
  - pymatching
  - grpo
  - trl
  - llm
license: mit
short_description: OpenEnv RL env that teaches an LLM to decode quantum errors.
---

# Qubit-Medic

> An LLM trained to decode quantum surface-code syndromes. We follow the
> AlphaQubit-style recipe (*Nature* 2024): a language model as decoder with
> verifiable rewards—implemented on **Stim + PyMatching**, an **OpenEnv**-style
> HTTP contract, **SFT warm-up + GRPO** (TRL/Unsloth), and **multi-component
> rewards** that are hard to game.

![Qubit-Medic decoding a syndrome on the rotated surface code](figures/grid_hero.png)

---

## Quick links

| Resource | URL |
|----------|-----|
| **Hugging Face Space (live demo + API)** | [ronitraj/QuantumScribe](https://huggingface.co/spaces/ronitraj/QuantumScribe) — health: [`/healthz`](https://ronitraj-quantumscribe.hf.space/healthz) |
| **Trained LoRA on the Hub** | [ronitraj/quantumscribe](https://huggingface.co/ronitraj/quantumscribe) (PEFT adapter + tokenizer) |
| **Weights & Biases** | Project `QuantumScribe-GRPO`, entity `ronitraj` — [W&B](https://wandb.ai/ronitraj) |
| **Colab training** | [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) |
| **Local Gradio** | `python app_gradio.py` |
| **OpenEnv manifest** | [`openenv.yaml`](openenv.yaml) |

---

## What this repo does (elevator pitch)

Quantum computers need a **decoder**: classical software that maps **syndromes** (detector results) to **corrections**. DeepMind’s [AlphaQubit](https://www.nature.com/articles/s41586-024-08148-8) showed a transformer can beat a strong **PyMatching** baseline. We reimplement the *idea* with a commodity stack:

- **3B** instruction-tuned **Qwen2.5** in **4-bit** (Unsloth) + **LoRA**
- **SFT** then **GRPO** (reward from a real Stim environment, not offline labels)
- **OpenEnv**-compatible server: `/reset` / `/step` / state & schema
- **Five** logged reward components (aggregate is weighted)

| Dimension | This project (typical) | AlphaQubit (reference) |
|-----------|------------------------|------------------------|
| Decoder | 3B LM + LoRA (off-the-shelf) | Custom architecture, lab-scale data mix |
| Training signal | SFT + GRPO on env reward | Proprietary + SI1000 / Sycamore |
| Baseline | PyMatching (sparse blossom) | Same class of MWM decoder |
| Open source | This repo + Hub weights | Research partial |

---

## Latest measured eval (JSON)

These numbers come from a held-out run written to `data/eval_grpo.json` (1000 episodes, L2 target, adapter path recorded in the file). They are the **source of truth** for submission claims; **do not** substitute synthetic plots for these metrics.

| Metric | Value |
|--------|------:|
| `logical_correction_rate` | 0.964 |
| `pymatching_beat_rate` | 0.0 |
| `format_compliance_rate` | 1.0 |
| `mean_hamming_overlap` | 0.8405 |
| `mean_total_reward` | ~0.821 |
| `exact_match_pymatching` | 0.734 |

`pymatching_beat` is 1 only when **PyMatching is wrong on the observable** and the **LLM is right**; on this eval it is **0.0**—i.e. no "beats" on that slice—so do not claim outperforming PM here without a separate run where that rate is non-zero. High **logical correction** and overlap with the PM frame remain meaningful; interpret with [reward definitions](qubit_medic/server/rewards.py).

Reproduce:

```bash
python -m scripts.eval --adapter /path/to/grpo/adapter --episodes 1000 --out data/eval_grpo.json
```

(Adjust `--adapter` to your checkpoint, e.g. a downloaded [ronitraj/quantumscribe](https://huggingface.co/ronitraj/quantumscribe) adapter.)

---

## Figures in `figures/`

[figures/FIGURES.md](figures/FIGURES.md) documents provenance. The **trajectory** PNGs (`total_reward.png`, `logical_correction.png`, `pymatching_beat_rate.png`) are **illustrative** (synthetic from baselines), not automatic exports from a specific W&B run. To replace them with real curves, use `scripts/plot_results.py` and your run logs (see the script and `FIGURES.md`).

---

## The problem (in one story)

Qubits are noisy. You do not observe errors directly; you get **syndromes** from stabilizer measurements. A **decoder** turns syndromes into a **Pauli correction**. **PyMatching** is a strong classical baseline. We train an LLM to output a parseable correction; the environment checks it with Stim and five reward functions.

---

## The environment

A **FastAPI** app exposes an **OpenEnv**-style flow (see [qubit_medic/server/app.py](qubit_medic/server/app.py) and [qubit_medic/server/openenv_adapter.py](qubit_medic/server/openenv_adapter.py)):

- `reset(seed)` — sample a syndrome (curriculum), return a prompt.
- `step(text)` — parse, score rewards, return reward + per-component `info`.

**Episodes** are **single-step**: one completion per episode. The trainer and W&B see each reward component separately.

```text
+----------+  reset / step  +---------------------------+
| TRL/     | ------------>  | Qubit-Medic (Stim+PM)     |
| Unsloth  |  observation  | parse, 5 rewards, return   |
+----------+ <------------  +---------------------------+
```

---

## Methodology checklist

| Concern | Status | Pointer |
|--------|--------|--------|
| Realistic noise (SI1000) | Used | Gidney & Fowler [arXiv:2108.10457](https://arxiv.org/abs/2108.10457) |
| Real code family | Stim `surface_code:rotated_memory_z` | [Stim](https://github.com/quantumlib/Stim) |
| Strong classical baseline | PyMatching v2 | [arXiv:2303.15933](https://arxiv.org/abs/2303.15933) |
| Policy optimisation | GRPO | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| OOD / Willow (optional) | `scripts/willow_validation.py` + `data/willow_d3.dem` | [Zenodo](https://zenodo.org/record/13359217) |

---

## Baselines (no LLM)

`make baselines` writes `data/baseline_results.json` (random, all-zeros, PyMatching). `make plots` rebuilds the headline figures from that JSON (see [figures/FIGURES.md](figures/FIGURES.md)).

```bash
make baselines
make plots
```

---

## Reward design (config-driven)

Weights are **`qubit_medic/config.py` → `REWARD_WEIGHTS`** (sum **1.0**):

```text
total = 0.35 * logical_correction
      + 0.25 * hamming_overlap
      + 0.20 * syndrome_consistency
      + 0.10 * format_compliance
      + 0.10 * pymatching_beat
```

| Component | Role |
|-----------|------|
| **logical_correction** | 1 if the implied correction matches logical observable (Stim). |
| **hamming_overlap** | Dense credit vs the PyMatching reference frame. |
| **syndrome_consistency** | Implied final detectors vs observed syndrome. |
| **format_compliance** | Parse success / partial / fail. |
| **pymatching_beat** | 1 only if **PM wrong** and **LLM right** (rare; headline for beating PM). |

Details: [qubit_medic/server/rewards.py](qubit_medic/server/rewards.py). GRPO uses a **shared batch cache** so all five components score the *same* `(prompt, completion)` (see W&B section in previous docs—[`qubit_medic/wandb_utils.py`](qubit_medic/wandb_utils.py) and trainer).

---

## Weights & Biases

Defaults: **`WANDB_ENTITY=ronitraj`**, **`WANDB_PROJECT=QuantumScribe-GRPO`**. Trainers use [qubit_medic/wandb_utils.py](qubit_medic/wandb_utils.py). Disable: `WANDB_DISABLED=1` or `QUBIT_MEDIC_WANDB=0`.

```bash
pip install -r requirements-train.txt
wandb login
GROUP=my-exp make train-sft
GROUP=my-exp make train-grpo
GROUP=my-exp make eval
```

---

## Reproducibility (`qubit_medic/config.py`)

| Item | Value |
|------|--------|
| Stim / PyMatching | Pinned in `requirements*.txt` |
| SFT default base | `Qwen/Qwen2.5-3B-Instruct` via Unsloth |
| GRPO default base | `unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit` |
| LoRA | `r=16`, `alpha=32`, `dropout=0.1`, `q/k/v/o` |
| GRPO | **1500** steps, short completions (`max_completion` 50), KL coeff **0.02**, `temperature=1.2` rollouts, etc. |
| Seeds | `42, 1337, 2024` |

**Import from `qubit_medic.config`**—do not duplicate magic numbers in scripts.

---

## Train and eval (local)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make validate

make sft-data
make baselines
make tests

python -m scripts.train_sft --output checkpoints/sft_warmup
python -m scripts.train_grpo \
  --sft-checkpoint checkpoints/sft_warmup/checkpoint-50 \
  --output checkpoints/grpo

python -m scripts.eval --adapter checkpoints/grpo --episodes 1000 --out data/eval_grpo.json
```

End-to-end: [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb). Makefile shortcuts: `make train-sft`, `make train-grpo`, `make eval` (see [Makefile](Makefile)).

---

## Publish the adapter to the Hub

Released weights: **[ronitraj/quantumscribe](https://huggingface.co/ronitraj/quantumscribe)**. Load as PEFT on the same base used for training:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "ronitraj/quantumscribe")
tokenizer = AutoTokenizer.from_pretrained("ronitraj/quantumscribe")
```

Re-upload: `hf upload ronitraj/quantumscribe /path/to/final .` with Hub authentication.

---

## Space deployment

- **Space:** [ronitraj/QuantumScribe](https://huggingface.co/spaces/ronitraj/QuantumScribe)
- **Script:** `python -m scripts.deploy_to_space` — see [scripts/deploy_to_space.py](scripts/deploy_to_space.py)
- For private model pulls, set Space secret `HF_TOKEN`.

---

## Cross-distribution (optional)

`python -m scripts.willow_validation` — see [scripts/willow_validation.py](scripts/willow_validation.py).

---

## Repository layout

```text
qubit_medic/
  config.py, models.py, prompts.py, wandb_utils.py
  client/
  server/   (app, environment, rewards, curriculum, physics, openenv_adapter)
scripts/
  validate_env.py, generate_sft_data.py, train_sft.py, train_grpo.py, eval.py
  baseline_policies.py, plot_results.py, animate_grid.py, willow_validation.py
  format_test.py, diversity_preflight.py, deploy_to_space.py, sync_kaggle_bundle.py
tests/     data/     figures/     checkpoints/     notebooks/colab_train.ipynb
app_gradio.py   Dockerfile   openenv.yaml   Makefile
```

---

## Citations

```bibtex
@article{bausch_alphaqubit_2024,
  title   = {Learning high-accuracy error decoding for quantum processors},
  author  = {Bausch, Johannes and others},
  journal = {Nature},
  volume  = {635},
  pages   = {834},
  year    = {2024},
  doi     = {10.1038/s41586-024-08148-8}
}
@article{acharya_willow_2024,
  title   = {Quantum error correction below the surface code threshold},
  author  = {Acharya, R. and others (Google Quantum AI)},
  journal = {arXiv:2408.13687},
  year    = {2024}
}
@article{gidney_si1000_2021,
  title   = {A fault-tolerant honeycomb memory},
  author  = {Gidney, Craig and Fowler, Austin G.},
  journal = {arXiv:2108.10457},
  year    = {2021}
}
@article{higgott_pymatching_2023,
  title   = {Sparse Blossom: correcting a million errors per core second
             with minimum-weight matching},
  author  = {Higgott, Oscar and Gidney, Craig},
  journal = {arXiv:2303.15933},
  year    = {2023}
}
@article{shao_grpo_2024,
  title   = {DeepSeekMath: pushing the limits of mathematical reasoning
             in open language models},
  author  = {Shao, Zhihong and others},
  journal = {arXiv:2402.03300},
  year    = {2024}
}
```

---

## Acknowledgments

DeepMind (AlphaQubit), Google Quantum AI (Stim, Willow data), Gidney (SI1000), Higgott (PyMatching), Hugging Face, Unsloth, OpenEnv.

---

## License

MIT — [LICENSE](LICENSE).
