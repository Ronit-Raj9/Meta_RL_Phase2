"""Locked experiment configuration (Section 1.4 of the plan).

Every magic number in the project lives here. Do not hard-code circuit
parameters, noise rates, or model identifiers anywhere else; import them
from this module instead.

Cited literature
----------------
Bausch et al., AlphaQubit, *Nature* 635:834 (2024)
    DOI: 10.1038/s41586-024-08148-8
    https://www.nature.com/articles/s41586-024-08148-8
Acharya et al. (Google QAI), *Willow*, arXiv:2408.13687 (2024)
    https://arxiv.org/abs/2408.13687
Gidney & Fowler, *SI1000*, arXiv:2108.10457 (2021)
    https://arxiv.org/abs/2108.10457
Higgott & Gidney, *PyMatching v2*, arXiv:2303.15933 (2023)
    https://arxiv.org/abs/2303.15933
Shao et al., *DeepSeekMath / GRPO*, arXiv:2402.03300 (2024)
    https://arxiv.org/abs/2402.03300
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


# --------------------------------------------------------------------------- #
# Quantum code geometry                                                        #
# --------------------------------------------------------------------------- #

CODE_TASK = "surface_code:rotated_memory_z"
"""Stim task identifier. We always use the rotated surface code with a Z
memory experiment - same family AlphaQubit and Willow report on."""

DISTANCE_PRIMARY: int = 3
"""Distance-3 is the primary benchmark configuration (AlphaQubit Fig. 2b)."""

DISTANCE_STRETCH: int = 5
"""Distance-5 is the stretch-goal configuration for Section 4.3."""

ROUNDS_FACTOR: int = 1
"""rounds = ROUNDS_FACTOR * distance. Value 1 matches the AlphaQubit
distance-equals-rounds protocol."""


# --------------------------------------------------------------------------- #
# Noise model: SI1000 sub-rates (Gidney & Fowler 2021, Table 1)                #
# --------------------------------------------------------------------------- #
# SI1000 maps a single physical error budget ``p`` to four operation-specific
# sub-rates. The factors below come from arXiv:2108.10457 Table 1 and are the
# *same* values Google's QAI uses in their Willow analyses.
#
# Stim's surface_code:rotated_memory_z generator accepts four matching knobs:
#   after_clifford_depolarization        (two-qubit gate noise)
#   before_round_data_depolarization     (idle data-qubit noise per round)
#   before_measure_flip_probability      (measurement noise)
#   after_reset_flip_probability         (reset noise)


@dataclass(frozen=True)
class SI1000Rates:
    """Per-operation error rates derived from a single budget ``p``."""

    after_clifford_depolarization: float
    before_round_data_depolarization: float
    before_measure_flip_probability: float
    after_reset_flip_probability: float

    @classmethod
    def from_p(cls, p: float) -> "SI1000Rates":
        """Build SI1000 sub-rates from the headline budget ``p``.

        The factors are exactly Gidney & Fowler 2021 Table 1.
        """
        return cls(
            after_clifford_depolarization=p,
            before_round_data_depolarization=p / 10.0,
            before_measure_flip_probability=p * 5.0,
            after_reset_flip_probability=p * 2.0,
        )

    def as_stim_kwargs(self) -> Mapping[str, float]:
        """Return the kwargs dict accepted by ``stim.Circuit.generated``."""
        return {
            "after_clifford_depolarization": self.after_clifford_depolarization,
            "before_round_data_depolarization": self.before_round_data_depolarization,
            "before_measure_flip_probability": self.before_measure_flip_probability,
            "after_reset_flip_probability": self.after_reset_flip_probability,
        }


# --------------------------------------------------------------------------- #
# Curriculum levels (Section 4)                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CurriculumLevel:
    """One rung on the difficulty ladder."""

    name: str
    distance: int
    rounds: int
    p: float
    promotion_threshold: float  # logical-correction rate at which we move on
    eval_size: int              # held-out shots used to test promotion


CURRICULUM: tuple[CurriculumLevel, ...] = (
    CurriculumLevel(
        name="L1_warmup",
        distance=DISTANCE_PRIMARY,
        rounds=1,
        # 0.0005 (was 0.0001) — at the original budget, L1 syndromes were
        # almost always trivial, dragging the SFT class balance down even
        # under per-level rejection sampling. Bumping to 0.0005 keeps L1
        # strictly easier than L2 (p=0.001) while giving the model real
        # non-empty examples to learn from at the warmup stage.
        p=0.0005,
        promotion_threshold=0.80,
        eval_size=100,
    ),
    CurriculumLevel(
        name="L2_target",
        distance=DISTANCE_PRIMARY,
        rounds=DISTANCE_PRIMARY,
        p=0.001,
        promotion_threshold=0.70,
        eval_size=200,
    ),
    CurriculumLevel(
        name="L3_stretch",
        distance=DISTANCE_STRETCH,
        rounds=DISTANCE_STRETCH,
        p=0.001,
        promotion_threshold=0.30,  # stretch goal - even partial counts
        eval_size=200,
    ),
)


# --------------------------------------------------------------------------- #
# Reward weights (Section 3) - sum to 1.0 by construction                      #
# --------------------------------------------------------------------------- #

REWARD_WEIGHTS: dict[str, float] = {
    # 2026-04 final-RL spec (margin-reward rewrite). The headline metric
    # (pymatching_margin) now carries 50% of the gradient because the
    # binary pymatching_beat reward fired <5% of the time and gave no
    # gradient on most prompts. Continuous margin in [0,1] (centered at
    # 0.5 for "tied with PyMatching") provides signal on every prompt.
    "logical_correction": 0.20,    # was 0.35; saturates once >= PyMatching
    "hamming_overlap": 0.15,       # was 0.25; saturates once >= PyMatching
    "syndrome_consistency": 0.10,  # was 0.20; flat once non-trivial
    "format_compliance": 0.05,     # was 0.10; already at 0.95+
    "pymatching_margin": 0.50,     # NEW; replaces pymatching_beat at 0.10
}
assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-9, "reward weights must sum to 1"


# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #

SEEDS: tuple[int, ...] = (42, 1337, 2024)
"""Three seeds for error bars - never run with anything else."""

PRIMARY_SEED: int = SEEDS[0]


# --------------------------------------------------------------------------- #
# Model + training                                                             #
# --------------------------------------------------------------------------- #

MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"
"""Locked primary model. 3B params, 4-bit quantised + LoRA fits in a Colab T4.
Backup is ``Qwen/Qwen2.5-7B-Instruct`` - only swap if format-test < 30%."""

MODEL_BACKUP_ID: str = "Qwen/Qwen2.5-7B-Instruct"
"""Only swap to this if the pre-onsite format test fails."""

# ---- LoRA (shared SFT + GRPO) -------------------------------------------- #
LORA_R: int = 16
LORA_ALPHA: int = 32  # 2x rank, standard ratio
LORA_DROPOUT: float = 0.10
"""Bumped 0.05 -> 0.10 (2026-04 SFT regularisation) because the prior
SFT runs converged to a single-output mode (every checkpoint reported
output_diversity=1) which left GRPO unable to compute non-zero
within-group reward variance. 0.10 is the spec's first-pass dropout;
the post-SFT diversity preflight will bump to 0.15 if needed."""
LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

# ---- SFT warmup phase (master spec, section 1; 2026-04 regularisation) -- #
# 2026-04 changes (diversity-preserving regularisation): SFT collapsed to
# a constant-output model under the prior settings (LR=2e-4 + dropout=0.05
# + max_steps=200 left every checkpoint at output_diversity=1). New
# defaults trade some ceiling LCR for diversity headroom so GRPO has a
# reward signal to climb.
SFT_EPOCHS: int = 1
SFT_BATCH_SIZE: int = 4
SFT_GRAD_ACCUM: int = 4              # effective batch = 16
SFT_LR: float = 1e-4
"""Halved 2e-4 -> 1e-4 to slow the slide into mode collapse."""
SFT_LR_SCHEDULER: str = "constant_with_warmup"  # 20-step warmup then constant
SFT_WARMUP_STEPS: int = 20
SFT_WEIGHT_DECAY: float = 0.01
SFT_LABEL_SMOOTHING: float = 0.05
"""TrainingArguments.label_smoothing_factor; spreads the loss across
non-target tokens so the model is less rewarded for memorising the
single highest-likelihood completion."""
SFT_OPTIMIZER: str = "adamw_8bit"
SFT_DATASET_SIZE: int = 3_000        # 3,000 train + 100 held-out validation
SFT_VAL_HOLDOUT: int = 100
SFT_MAX_SEQ_LEN: int = 1024          # ~300 prompt + ~80 completion + headroom
SFT_MAX_STEPS: int = 50
"""Cut 200 -> 50 so SFT stops well before the model can grind itself
into a single-output mode. The format-only knowledge fits in <50
steps and post-SFT diversity preflight is the gate to GRPO."""
SFT_EVAL_EVERY: int = 25             # legacy fallback if no schedule given
SFT_SAVE_EVERY: int = 25
SFT_LOG_EVERY: int = 10
SFT_PREFLIGHT_DIVERSITY_FLOOR: int = 2
"""eval/output_diversity threshold. If two consecutive evals both report
output_diversity below this floor, the diversity-collapse early stop
fires and SFT exits with reason=diversity_collapse."""
SFT_DIVERSITY_COLLAPSE_RUN_LEN: int = 2
"""Number of consecutive sub-floor evals required before stopping."""
SFT_MAX_NEW_TOKENS: int = 200        # generation cap during eval
# Was 128; bumped to 200 because Qwen2.5-Instruct's cold-start reasoning
# (### Analysis: 1. ... 2. ... 3. ...) regularly runs to 100+ tokens
# before reaching the format line in early SFT steps. With 128, every
# step-5 sample truncated mid-reasoning and format_compliance read 0.
# 200 gives ~70 tokens of headroom past a typical reasoning + format
# completion (~70 tokens total) so truncation never masks the model's
# real behaviour.

# --- Variable eval cadence ------------------------------------------------- #
# Early evals are quick sanity checks (small sample, format-only) so a
# broken parser / generation drift gets caught before ~10 min of compute is
# burned. Late evals are real measurements with the full sample size.
# Catching format-compliance failure at step 15 instead of step 50 saves
# ~7 minutes per fire.
#
# Each entry: (step, sample_size, mode) where mode is "format_only" or
# "full". format_only skips the diversity probe and the physics-heavy
# logical_correction / hamming / syndrome metrics, so the eval costs
# ~30 seconds instead of ~2 minutes.
SFT_EVAL_SCHEDULE: tuple[tuple[int, int, str], ...] = (
    # 2026-04: schedule rebuilt to fit the SFT_MAX_STEPS=50 budget. Two
    # full evals plus a fast format probe gives the diversity-collapse
    # early-stop two consecutive data points before the run ends, which
    # is the minimum to fire the new run-length-2 stop rule.
    (5,   30,  "format_only"),
    (15,  50,  "full"),
    (25,  100, "full"),
    (40,  100, "full"),
    (50,  100, "full"),
)
SFT_PRINT_SAMPLE_OUTPUTS: int = 5    # raw outputs printed at each eval

# Early-stop thresholds (master spec, section 3).
SFT_EARLY_STOP_FORMAT: float = 0.95
SFT_EARLY_STOP_CORRECTION: float = 0.80
SFT_EARLY_STOP_DIVERSITY: int = 3
SFT_MAX_WALL_SECONDS: float = 30 * 60.0  # 30-minute hard ceiling

# HuggingFace Trainer subfolder (step-50 save) used to initialise GRPO.
# ``python -m scripts.train_grpo`` defaults to this path; pipeline scripts
# also pass it explicitly.
SFT_CHECKPOINT_PATH_FOR_GRPO: str = "checkpoints/sft_warmup/checkpoint-50"

# ---- GRPO RL phase (master spec, section 5; 2026-04 spec rewrite) -------- #
# All numbers below were re-pinned by the 2026-04 GRPO spec. The previous
# defaults (GRPO_STEPS=2000, LR=1e-5, KL=0.04, max_prompt=512,
# max_completion=256, temperature=0.7) produced a degenerate "always say
# []" policy in <100 steps because reward variance collapsed and KL
# saturated the loss. The new defaults emphasise diversity:
#
#   - higher temperature (1.2) + top_k + repetition_penalty -> non-collapsed rollouts
#   - shorter max_completion_length (50) -> the answer is one short line anyway
#   - longer max_prompt_length (1500) -> distance-3 syndromes already use
#     ~280 tokens; distance-5 / curriculum L3 needs the headroom
#   - lower KL coefficient (0.02) -> reward signal not dominated by KL drift
#   - 1500 steps -> wall-clock fits the 13h cap with margin
GRPO_STEPS: int = 1_500
GRPO_GEN_PER_PROMPT: int = 8         # was 4; 8 keeps reward variance non-degenerate
                                     # even when one or two completions collapse
GRPO_BATCH_SIZE: int = 1             # per-device prompts per step
GRPO_GRAD_ACCUM: int = 8             # effective batch = 8 prompts
GRPO_LR: float = 2e-5                # bumped from 1e-5; reward signal is sparse
GRPO_LR_SCHEDULER: str = "constant"  # no warmup, no decay
GRPO_KL_COEF: float = 0.04           # was 0.02; doubled to keep policy honest
                                     # under the new continuous margin reward
GRPO_MAX_PROMPT_LEN: int = 1_500     # surface-code prompts can run long
GRPO_MAX_COMPLETION_LEN: int = 24    # was 50; tail past 24 tokens is garbage that
                                     # was being optimised against during training
GRPO_STOP_STRINGS: tuple[str, ...] = ("\n\n", "<|im_end|>", "<|endoftext|>")
"""Stop-string set passed to the rollout generator so completions terminate at
EOS / blank line instead of running to ``max_completion_length`` (which made
clipped_ratio 1.0 and broke the train/eval policy match)."""

# Filtered GRPO prompt pool (CHANGE 3): GRPO loads from this path instead of
# data/sft_dataset.jsonl. The filter keeps only PyMatching-fallible prompts
# (PM logical-error distance > 0) or multi-error syndromes (true_x|true_z >= 2),
# concentrating GRPO compute on prompts with actual learning headroom.
GRPO_PROMPT_POOL_PATH: str = "data/grpo_prompt_pool.jsonl"

# Curriculum mix used for GRPO ROLLOUT prompts only (SFT mix unchanged).
# 2026-04 final-RL spec: shift toward harder noise where MWPM breaks down
# and beat-rate can fire.
GRPO_CURRICULUM_MIX: dict[str, float] = {
    "L1_warmup": 0.20,    # was 0.40
    "L2_target": 0.50,    # unchanged
    "L3_stretch": 0.30,   # was 0.10; tripled for beat-rate headroom
}

# ---- Diversity-focused rollout sampling (critical) ----------------------- #
# These apply to GRPO ROLLOUT generation only. Eval uses temperature=0
# (greedy) regardless of these. The combination temperature=1.2 + top_p=0.95
# + top_k=50 + repetition_penalty=1.1 was selected because:
#   * temperature=1.2 broadens the per-token distribution past the SFT
#     mode-collapsed favourite ("X_ERRORS=[] Z_ERRORS=[]").
#   * top_p=0.95 keeps tail tokens in but truncates the long tail.
#   * top_k=50 caps the candidate set so we don't sample garbage.
#   * repetition_penalty=1.1 discourages the model from repeating the
#     exact same byte sequence within a 4-completion group (reduces
#     "all 4 generations identical" rate, which kills GRPO's gradient).
GRPO_TEMPERATURE: float = 1.0       # was 1.2; this is now the STARTING value
                                    # for the schedule below, not a constant
GRPO_TOP_P: float = 0.95
GRPO_TOP_K: int = 50
GRPO_REPETITION_PENALTY: float = 1.1
GRPO_DO_SAMPLE: bool = True

# Linear temperature annealing schedule (replaces the prior auto-bump-on-
# collapse band-aid that escalated to T=2.0 and produced incoherent rollouts).
#   steps  0-300   : T = 1.0          (exploration phase)
#   steps  300-1000: T = 1.0 -> 0.9   (linear, mid-training stabilisation)
#   steps  1000+   : T = 0.9 -> 0.7   (linear, sharpening toward best policy)
GRPO_TEMP_SCHEDULE: tuple[tuple[int, float], ...] = (
    (0,    1.0),
    (300,  1.0),
    (1000, 0.9),
    (1500, 0.7),
)


def temperature_at_step(step: int,
                        schedule: tuple[tuple[int, float], ...] = GRPO_TEMP_SCHEDULE,
                        ) -> float:
    """Piecewise-linear interpolation over the GRPO temperature schedule."""
    if step <= schedule[0][0]:
        return schedule[0][1]
    for (s0, t0), (s1, t1) in zip(schedule, schedule[1:]):
        if s0 <= step <= s1:
            if s1 == s0:
                return t1
            frac = (step - s0) / (s1 - s0)
            return t0 + (t1 - t0) * frac
    return schedule[-1][1]

# ---- Checkpoint cadence + retention -------------------------------------- #
GRPO_CHECKPOINT_EVERY: int = 100
GRPO_SAVE_TOTAL_LIMIT: int = 3       # keep 3 most recent rolling checkpoints
GRPO_LOG_EVERY: int = 5              # real-time visibility (every 5 steps)
GRPO_OPTIMIZER: str = "adamw_8bit"
GRPO_KL_ALARM: float = 0.3           # informational only (no auto-action)
GRPO_KL_HARD_CEIL: float = 0.5       # if sustained for GRPO_KL_HARD_CEIL_RUN_LEN
                                     # steps, halve the LR (was: kill the run)
GRPO_KL_HARD_CEIL_RUN_LEN: int = 5   # consecutive logged steps above the ceiling
                                     # before the LR halves once

# ---- Wall-clock safety --------------------------------------------------- #
GRPO_WALL_SECONDS: float = 46_800.0   # 13 hours. Save+exit if exceeded.

# ---- Frozen eval set ----------------------------------------------------- #
# The 200-syndrome eval set is regenerated from the env at GRPO start with
# this seed. Same seed as SFT validation (sft_validation.jsonl) so eval
# distributions are comparable across SFT and GRPO. The set is cached on
# disk under data/grpo_validation.jsonl so reruns hit identical syndromes.
GRPO_VAL_SEED: int = 4_284
GRPO_VAL_EPISODES: int = 200
GRPO_VAL_PATH: str = "data/grpo_validation.jsonl"

# ---- Sample-table logging ------------------------------------------------ #
GRPO_SAMPLE_LOG_EVERY: int = 50
GRPO_SAMPLE_LOG_N: int = 5

# ---- Anti-hacking: mode-collapse inspection hook ------------------------- #
# Every N steps, we sample the most-recent N rollouts and check what
# fraction of prompts had ALL 4 generations identical. If too many
# prompts collapsed, raise the rollout temperature by a fixed step.
GRPO_INSPECTION_HOOK_EVERY: int = 100
GRPO_INSPECTION_SAMPLE_N: int = 10
# 2026-04 final-RL spec (CHANGE 6): the inspection-hook auto-bump is
# disabled. The previous version escalated temperature to 2.0, which
# produced incoherent training rollouts and didn't translate to better
# eval policies. Threshold > sample_n means the condition can never fire,
# so the hook still LOGS uniqueness but never mutates temperature.
GRPO_INSPECTION_COLLAPSE_THRESHOLD: int = 99
GRPO_TEMP_BUMP_ON_COLLAPSE: float = 0.0

# ---- Decision-rule thresholds (warnings only; no auto-action) ----------- #
GRPO_DECISION_REWARD_STD_FLOOR: float = 0.03
GRPO_DECISION_REWARD_STD_CHECK_STEP: int = 50
GRPO_DECISION_BEAT_RATE_CHECK_STEP: int = 500
GRPO_DECISION_FORMAT_FLOOR: float = 0.95
GRPO_DECISION_GRAD_NORM_CEIL: float = 50.0
GRPO_DECISION_GRAD_NORM_RUN_LEN: int = 3       # consecutive logs

# Decoding sampler defaults at evaluation/format-test time.
# (Used by greedy eval paths: temp/top_p only matter when do_sample=True.)
SAMPLE_TEMPERATURE: float = 0.7
SAMPLE_TOP_P: float = 0.95


# --------------------------------------------------------------------------- #
# Server / deployment                                                          #
# --------------------------------------------------------------------------- #

EPISODE_TIMEOUT_SECONDS: float = 30.0
"""Wall-clock budget per episode (Section 2.6)."""

DEFAULT_HOST: str = "0.0.0.0"
DEFAULT_PORT: int = 7860  # Hugging Face Spaces' default exposed port


# --------------------------------------------------------------------------- #
# Weights & Biases                                                             #
# --------------------------------------------------------------------------- #
# Centralised so the SFT trainer, GRPO trainer, eval script, and notebook
# all log to the same project / dashboard. Override per-run on the CLI.
import os as _os  # noqa: E402  (local import to keep top of module clean)

WANDB_PROJECT: str = _os.environ.get("WANDB_PROJECT", "QuantumScribe-GRPO")
"""Default W&B project name. Override with ``WANDB_PROJECT=...``.

Changed 2026-04 from ``"QuantumScribe"`` to ``"QuantumScribe-GRPO"`` per
the GRPO spec rewrite. SFT runs that should land in the original project
should set ``WANDB_PROJECT=QuantumScribe`` at the shell."""

WANDB_ENTITY: str | None = _os.environ.get("WANDB_ENTITY", "ronitraj") or None
"""W&B team or username. ``None`` -> wandb's default entity for the user."""

WANDB_DEFAULT_TAGS: tuple[str, ...] = (
    "qubit-medic",
    "quantum-error-correction",
    "openenv",
    f"distance-{DISTANCE_PRIMARY}",
    "si1000",
)
"""Tags applied to every W&B run (per-script tags appended on top)."""

WANDB_LOG_GENERATIONS_EVERY: int = 50
"""Log a sample-completion table every N GRPO steps (master spec sec. 7)."""

WANDB_SAMPLE_GENERATIONS: int = 5
"""Number of generations included in each sample-completion table.
Master spec, section 7: 'Save 5 randomly sampled rollouts ... and their rewards.'"""

WANDB_INLOOP_EVAL_EVERY: int = 100
"""Run an in-loop evaluation pass (deterministic, ``WANDB_INLOOP_EVAL_EPISODES``
syndromes) every N GRPO steps. Tightened from 250 -> 100 by the 2026-04 GRPO
spec rewrite so collapse / drift gets caught within a 5-minute window
instead of a 15-minute window."""

WANDB_INLOOP_EVAL_EPISODES: int = 200
"""Held-out syndromes per in-loop eval pass. Bumped from 100 -> 200 by the
2026-04 spec rewrite so eval-stat error bars are tight enough to read
pymatching_beat_rate movement (which is sub-5% in early training)."""

WANDB_COMPARE_EVERY: int = 500
"""Run the PyMatching head-to-head comparison every N steps (master spec sec. 7)."""


# --------------------------------------------------------------------------- #
# Convenience accessors                                                        #
# --------------------------------------------------------------------------- #


def level_by_name(name: str) -> CurriculumLevel:
    for lvl in CURRICULUM:
        if lvl.name == name:
            return lvl
    raise KeyError(f"unknown curriculum level {name!r}")


def primary_level() -> CurriculumLevel:
    """The L2 target benchmark - what the headline numbers come from."""
    return level_by_name("L2_target")
