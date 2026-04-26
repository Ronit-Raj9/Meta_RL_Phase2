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

    Computes Hamming similarity between ``predicted_final_bits`` (induced
    by the predicted X errors) and ``observed_final_bits``. Returns
    ``1 - hamming_distance / num_final_detectors``.

    Rationale (Section 3.2): without this term, an LLM that lucky-guesses
    the right qubits could get Reward 1 occasionally; this signal forces
    it to also explain the data the syndrome carries.

    2026-04 anti-collapse cap (FIX 1, RL spec rewrite): if the prediction
    is empty AND the observed syndrome is non-empty (at least one
    detector fired), cap the score at 0.5. Without this cap, the
    "always predict empty" policy can still pull a high syndrome-
    consistency score on the prompts where the implied final-round bits
    happen to coincide with zeros, which kept GRPO trapped in the
    constant-empty mode.
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
    base = 1.0 - distance / len(final_dets)

    # Anti-collapse cap: empty prediction + non-empty observed syndrome
    # is a "did nothing while alarms were firing" failure mode. Cap at
    # 0.5 so the empty policy can never approach the full 1.0 even when
    # the implied final-round bits happen to coincide.
    pred_is_empty = (not parsed.x_errors) and (not parsed.z_errors)
    has_active_syndrome = any(int(b) != 0 for b in sample.syndrome_bits)
    if pred_is_empty and has_active_syndrome:
        return min(base, 0.5)
    return base


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


def _set_aware_jaccard(true_set: list[int], pred_set: list[int]) -> float:
    """Set-aware Jaccard: penalises BOTH false alarms and missed errors.

    2026-04 spec rewrite (FIX 1). The four-case rule is what makes
    "predict empty everywhere" stop being a near-optimal strategy:

    +-------------+-----------+-----------------------------------------+
    | true_set    | pred_set  | score                                   |
    +-------------+-----------+-----------------------------------------+
    | empty       | empty     | 1.0   (perfect, "no errors -> no edit") |
    | empty       | non-empty | 0.0   false alarm                       |
    | non-empty   | empty     | 0.0   missed errors  <-- the key change |
    | non-empty   | non-empty | |inter| / |union|  (standard Jaccard)   |
    +-------------+-----------+-----------------------------------------+

    Critically the third case used to score 1.0 under the prior plain
    Jaccard (because both sets were treated symmetrically; "everything
    correct, just nothing predicted" was indistinguishable from "perfect
    agreement"). Under this rule a missed-error answer scores 0.0,
    which moves the GRPO reward landscape so a non-trivial prediction
    can climb out of the empty-everywhere local optimum.
    """
    sa, sp = set(true_set), set(pred_set)
    if not sa and not sp:
        return 1.0  # perfect agreement: no true errors AND no claimed errors
    if not sa and sp:
        return 0.0  # false alarm: claimed errors that were not there
    if sa and not sp:
        return 0.0  # missed errors: alarms fired but model said nothing
    inter = len(sa & sp)
    union = len(sa | sp)
    return inter / union if union else 1.0


def reward_hamming_overlap(
    parsed: ParseResult,
    sample: SyndromeSample,
    layout: CircuitLayout,
) -> float:
    """Average of set-aware Jaccard(X) and set-aware Jaccard(Z) against
    the GROUND-TRUTH error sets recorded by Stim.

    2026-04 spec rewrite (FIX 1):
      * Reference changed from PyMatching's predicted Pauli frame to
        the actual ``sample.true_x_errors`` / ``sample.true_z_errors``.
        PyMatching's prediction is itself a noisy estimate; comparing
        against it rewarded "imitate PyMatching" rather than "decode
        correctly", which masked the policy when both we AND PyMatching
        were wrong on the same syndrome.
      * Per-axis score uses the set-aware rule (see
        :func:`_set_aware_jaccard`), so missed errors no longer score
        1.0 just because the prediction set is empty.
    """
    jx = _set_aware_jaccard(sample.true_x_errors, parsed.x_errors)
    jz = _set_aware_jaccard(sample.true_z_errors, parsed.z_errors)
    return 0.5 * (jx + jz)


# --------------------------------------------------------------------------- #
# Reward 4: format compliance                                                  #
# --------------------------------------------------------------------------- #


def reward_format_compliance(parsed: ParseResult) -> float:
    """Binary {0.0, 1.0}: 1.0 iff the parser fully extracted both lists.

    2026-04 spec rewrite (FIX 1): partial credit (0.5) is removed. With
    partial credit on, the model could still earn ~half the format
    weight on garbage outputs that resembled the canonical form, which
    is part of what kept the reward landscape too flat for GRPO to
    escape the empty-everywhere mode. The new rule rewards only a
    cleanly-parsed answer.
    """
    return 1.0 if parsed.parse_success else 0.0


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
