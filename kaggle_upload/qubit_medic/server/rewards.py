"""The five reward functions (Section 3 of the plan).

Design contract (from Section 3.6):

* Each reward is a pure function ``(action, state, layout) -> float in [0, 1]``.
* Rewards never observe each other - they're independent by construction so
  the LLM can't satisfy one at the expense of another without genuine task
  understanding.
* The combined reward is a weighted sum (weights in :mod:`qubit_medic.config`)
  clamped to ``[0, 1]``.
* Every per-component score is reported in the ``info`` dict so logs can
  surface reward-hacking early (Section 3.7).

A note on Reward 2 and Reward 3 ground truth - see ``physics.py``: the LLM
predicts a *terminal Pauli frame*, which fully determines the logical-Z
observable but only constrains the *final-round* detectors. Earlier rounds'
detectors are intentionally unscored. Reward 3 compares against PyMatching's
near-optimal Pauli-frame prediction (the canonical decoder reference used in
AlphaQubit's Nature paper).
"""
from __future__ import annotations

from dataclasses import dataclass

from qubit_medic.config import REWARD_WEIGHTS
from qubit_medic.prompts import ParseResult
from qubit_medic.server.physics import (
    CircuitLayout,
    SyndromeSample,
    predicted_observable_flip,
)


# --------------------------------------------------------------------------- #
# Reward 1: logical correction success                                         #
# --------------------------------------------------------------------------- #


def reward_logical_correction(
    parsed: ParseResult,
    sample: SyndromeSample,
    layout: CircuitLayout,
) -> float:
    """Did the predicted correction preserve the logical state?

    Apply the predicted X errors as a Pauli frame at end-of-circuit and
    compute the implied observable flip. If this matches the actual
    observable flip recorded by Stim, the logical state was preserved.
    Outputs 1.0 if so, else 0.0.

    This is the unfakeable reward - it depends only on Stim's ground truth.
    """
    implied = predicted_observable_flip(parsed.x_errors, layout)
    return 1.0 if implied == sample.actual_observable_flip else 0.0


# --------------------------------------------------------------------------- #
# Reward 2: syndrome consistency                                               #
# --------------------------------------------------------------------------- #


def _syndrome_from_pauli_frame(
    x_errors: list[int],
    layout: CircuitLayout,
    final_detector_supports: dict[int, frozenset[int]],
) -> dict[int, int]:
    """Compute the implied bits for FINAL-round detectors only.

    A terminal X error on data qubit ``q`` flips a final-round Z-stabiliser
    detector iff ``q`` is in that detector's support.
    """
    out: dict[int, int] = {}
    x_set = set(x_errors)
    for det_idx, support in final_detector_supports.items():
        out[det_idx] = 1 if len(x_set & support) % 2 == 1 else 0
    return out


def reward_syndrome_consistency(
    parsed: ParseResult,
    sample: SyndromeSample,
    layout: CircuitLayout,
    final_detector_supports: dict[int, frozenset[int]],
) -> float:
    """How well does the predicted Pauli frame reproduce the FINAL detectors?

    Computes Hamming similarity between ``predicted_final_bits`` (induced by
    the predicted X errors) and ``observed_final_bits``. Returns
    ``1 - hamming_distance / num_final_detectors``.

    Rationale (Section 3.2): without this term, an LLM that lucky-guesses
    the right qubits could get Reward 1 occasionally; this signal forces it
    to also explain the data the syndrome carries.
    """
    final_dets = layout.final_detectors
    if not final_dets:
        return 0.0
    implied = _syndrome_from_pauli_frame(
        parsed.x_errors, layout, final_detector_supports
    )
    distance = 0
    for det_idx in final_dets:
        observed = sample.syndrome_bits[det_idx]
        predicted = implied.get(det_idx, 0)
        if observed != predicted:
            distance += 1
    return 1.0 - distance / len(final_dets)


def compute_final_detector_supports(
    layout: CircuitLayout,
    syndrome_bits_unused: list[int] | None = None,  # API symmetry
    *,
    detector_to_data_qubits: dict[int, frozenset[int]] | None = None,
) -> dict[int, frozenset[int]]:
    """Map each final-round detector to the set of data qubits whose
    terminal X error flips it.

    For the rotated memory_z code, each Z-stabiliser final detector watches
    the four (or two/one on the boundary) data qubits adjacent to it on the
    grid. We compute adjacency by Euclidean distance; data qubits at
    distance ``sqrt(2)`` from a Z-stabiliser ancilla coordinate are
    incident.
    """
    if detector_to_data_qubits is not None:
        return detector_to_data_qubits

    out: dict[int, frozenset[int]] = {}
    for det_idx in layout.final_detectors:
        dx, dy = layout.detector_coords[det_idx]
        adj: set[int] = set()
        for q, (qx, qy) in zip(layout.data_qubits, layout.data_qubit_coords):
            if abs((qx - dx) ** 2 + (qy - dy) ** 2 - 2.0) < 1e-6:
                adj.add(q)
        out[det_idx] = frozenset(adj)
    return out


# --------------------------------------------------------------------------- #
# Reward 3: Hamming overlap with reference Pauli frame                         #
# --------------------------------------------------------------------------- #


def _jaccard(a: list[int], b: list[int]) -> float:
    """Jaccard index. Returns 1.0 when both sets are empty (perfect agreement)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0


def reward_hamming_overlap(
    parsed: ParseResult,
    sample: SyndromeSample,
    layout: CircuitLayout,
) -> float:
    """Average of Jaccard(X) and Jaccard(Z) against the reference frame.

    Reference is PyMatching's per-edge predicted Pauli frame
    (``sample.pymatching_x_errors`` / ``..._z_errors``). This is the dense
    partial-credit signal of Section 3.3 - even if Reward 1 fires zero,
    being *close* to the canonical solution still gets credit, smoothing
    the reward landscape during early training.
    """
    jx = _jaccard(parsed.x_errors, sample.pymatching_x_errors)
    jz = _jaccard(parsed.z_errors, sample.pymatching_z_errors)
    return 0.5 * (jx + jz)


# --------------------------------------------------------------------------- #
# Reward 4: format compliance                                                  #
# --------------------------------------------------------------------------- #


def reward_format_compliance(parsed: ParseResult) -> float:
    """1.0 if both keys parsed, 0.5 if exactly one, 0.0 if neither."""
    return parsed.format_score


# --------------------------------------------------------------------------- #
# Reward 5: PyMatching beat-rate bonus                                         #
# --------------------------------------------------------------------------- #


def reward_pymatching_beat(
    parsed: ParseResult,
    sample: SyndromeSample,
    layout: CircuitLayout,
) -> float:
    """1.0 iff PyMatching got this syndrome wrong AND the LLM got it right.

    This is the headline metric (Section 3.5). Most of training it'll be
    near zero; the trajectory of its mean over steps is the proof we've
    moved past pure imitation.
    """
    pm_correct = sample.pymatching_observable_pred == sample.actual_observable_flip
    if pm_correct:
        return 0.0
    llm_implied = predicted_observable_flip(parsed.x_errors, layout)
    return 1.0 if llm_implied == sample.actual_observable_flip else 0.0


# --------------------------------------------------------------------------- #
# Combined reward                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-component scores plus the weighted total."""

    logical_correction: float
    syndrome_consistency: float
    hamming_overlap: float
    format_compliance: float
    pymatching_beat: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "logical_correction": self.logical_correction,
            "syndrome_consistency": self.syndrome_consistency,
            "hamming_overlap": self.hamming_overlap,
            "format_compliance": self.format_compliance,
            "pymatching_beat": self.pymatching_beat,
            "total": self.total,
        }


def compute_all_rewards(
    parsed: ParseResult,
    sample: SyndromeSample,
    layout: CircuitLayout,
    final_detector_supports: dict[int, frozenset[int]],
    weights: dict[str, float] = REWARD_WEIGHTS,
) -> RewardBreakdown:
    """Compute all five rewards and the weighted total.

    Returns a :class:`RewardBreakdown` whose ``as_dict`` is what the env's
    ``info`` payload contains. The trainer logs each component separately.
    """
    r1 = reward_logical_correction(parsed, sample, layout)
    r2 = reward_syndrome_consistency(parsed, sample, layout, final_detector_supports)
    r3 = reward_hamming_overlap(parsed, sample, layout)
    r4 = reward_format_compliance(parsed)
    r5 = reward_pymatching_beat(parsed, sample, layout)
    total = (
        weights["logical_correction"] * r1
        + weights["syndrome_consistency"] * r2
        + weights["hamming_overlap"] * r3
        + weights["format_compliance"] * r4
        + weights["pymatching_beat"] * r5
    )
    total = max(0.0, min(1.0, total))
    return RewardBreakdown(
        logical_correction=r1,
        syndrome_consistency=r2,
        hamming_overlap=r3,
        format_compliance=r4,
        pymatching_beat=r5,
        total=total,
    )
