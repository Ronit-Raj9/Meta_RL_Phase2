"""Pydantic data classes shared by client and server (Section 2.2 of the plan).

Three classes draw the trust boundary:

* ``DecoderObservation`` - what the LLM sees on each step.
* ``DecoderAction``      - what the LLM emits (after parsing).
* ``DecoderState``       - server-side state, never serialised to the client.

Keeping the wire schema explicit is what closes off reward-hacking attacks:
the LLM literally cannot reach into the ``true_error_pattern`` because that
field is not in any class it ever receives.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Wire types - sent across the OpenEnv HTTP boundary                           #
# --------------------------------------------------------------------------- #


class DecoderObservation(BaseModel):
    """The view the LLM (and only the LLM) sees on each step."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(
        ...,
        description=(
            "Pre-formatted prompt string. This is exactly what the trainer "
            "passes to the policy - it appears verbatim in training logs."
        ),
    )
    syndrome_bits: list[int] = Field(
        ...,
        description=(
            "Raw detector activations (0/1). Provided for debugging and "
            "reward-hacking audits; the LLM should be reading the prompt, not "
            "this array."
        ),
    )
    distance: int = Field(..., description="Code distance for this episode.")
    rounds: int = Field(..., description="Number of stabiliser rounds.")
    p: float = Field(..., description="Physical error budget (SI1000 base).")
    curriculum_level: str = Field(..., description="Curriculum level name.")
    episode_id: int = Field(..., description="Monotonic episode counter.")
    dem_digest: str = Field(
        ...,
        description=(
            "Short hash of the detector error model used this episode. The "
            "trainer logs this so we can group rollouts by physics config."
        ),
    )


class DecoderAction(BaseModel):
    """Action emitted by the LLM after parsing.

    ``raw_response`` is preserved exactly so we can satisfy the participant
    guide's *inspect generations* mandate (Section 2.5 of the plan).
    """

    model_config = ConfigDict(frozen=True)

    x_error_qubits: list[int] = Field(default_factory=list)
    z_error_qubits: list[int] = Field(default_factory=list)
    raw_response: str = ""
    parse_success: bool = True


class StepResult(BaseModel):
    """Standard env step return (mirrors OpenEnv core/Gymnasium)."""

    observation: DecoderObservation
    reward: float
    done: bool
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    """Optional knobs the trainer can pass to ``reset``."""

    seed: Optional[int] = None
    forced_level: Optional[str] = Field(
        default=None,
        description=(
            "Override the curriculum scheduler. Used by eval scripts that "
            "want a specific (distance, rounds, p) configuration."
        ),
    )


class StepRequest(BaseModel):
    """The trainer sends the LLM's raw text; the server parses + scores."""

    raw_response: str
    episode_id: int


# --------------------------------------------------------------------------- #
# Server-only state - intentionally NOT a wire type                            #
# --------------------------------------------------------------------------- #


class DecoderState(BaseModel):
    """Per-episode state kept on the server; never sent to the client.

    Pydantic ``arbitrary_types_allowed`` is on because we hold a reference to
    a ``stim.Circuit`` object. The state is not serialised over HTTP - it
    lives in the server's per-episode dict and is discarded on ``done``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    episode_id: int
    seed: int
    curriculum_level: str
    distance: int
    rounds: int
    p: float

    syndrome_bits: list[int]
    true_x_errors: list[int]
    true_z_errors: list[int]
    actual_observable_flip: int  # 0 or 1; the unfakeable ground truth
    pymatching_observable_pred: int  # 0 or 1; baseline's prediction

    # Pre-computed quantities the reward functions need.
    x_observable_support: list[int]  # data qubits whose Z error flips X obs
    z_observable_support: list[int]  # data qubits whose X error flips Z obs
    num_data_qubits: int
    num_stabilizers: int

    # Stim/PyMatching objects - kept opaque to satisfy Pydantic.
    circuit_text: str
    dem_text: str

    # Reward audit log.
    last_reward_breakdown: dict[str, float] = Field(default_factory=dict)
