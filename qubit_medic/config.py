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
        p=0.0001,
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
    "logical_correction": 0.40,    # Reward 1 - the unfakeable ground truth
    "syndrome_consistency": 0.20,  # Reward 2 - prevents lucky-guess attacks
    "hamming_overlap": 0.20,       # Reward 3 - dense partial credit
    "format_compliance": 0.10,     # Reward 4 - parser must succeed
    "pymatching_beat": 0.10,       # Reward 5 - the headline metric
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
"""3B params, 4-bit quantised + LoRA fits in a Colab T4."""

LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

SFT_EPOCHS: int = 1
SFT_BATCH_SIZE: int = 4
SFT_GRAD_ACCUM: int = 4
SFT_LR: float = 2e-4
SFT_DATASET_SIZE: int = 5_000
SFT_MAX_SEQ_LEN: int = 2048

GRPO_STEPS: int = 2_000
GRPO_GEN_PER_PROMPT: int = 4
GRPO_LR: float = 1e-5
GRPO_KL_COEF: float = 0.04
GRPO_MAX_PROMPT_LEN: int = 512
GRPO_MAX_COMPLETION_LEN: int = 256
GRPO_CHECKPOINT_EVERY: int = 250
GRPO_LOG_EVERY: int = 50

# Decoding sampler defaults at evaluation/format-test time.
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
