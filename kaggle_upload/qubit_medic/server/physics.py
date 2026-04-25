"""Stim + PyMatching wrapper - the physics engine (Section 2.4 of the plan).

This module never makes decoding decisions: it builds circuits, samples
syndromes, computes baselines, and exposes the observable's support on the
data qubits so the reward functions can score predictions deterministically.

Two design choices worth flagging up-front:

* The LLM's action is a **terminal Pauli frame** on data qubits (the X and Z
  errors on each data qubit at the moment of final measurement). This
  representation is exact for the rotated memory_z task and lets us reuse
  Stim/PyMatching ground-truth machinery. The trade-off is documented in
  ``rewards.py``: the syndrome-consistency reward (Reward 2) only constrains
  the *final-round* detectors. Earlier rounds are silent w.r.t. an
  end-of-circuit Pauli frame; that is intentional and made explicit in the
  reward's docstring.

* "Ground-truth error pattern" for Reward 3 is taken to be the
  **PyMatching-most-probable error pattern** explaining the syndrome
  (extracted via ``Matching.decode_to_edges_array``). This is the
  near-optimal canonical choice and matches what the AlphaQubit baseline
  comparison uses. The README's *honesty note* repeats this.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pymatching
import stim

from qubit_medic.config import (
    CODE_TASK,
    CurriculumLevel,
    SI1000Rates,
)


# --------------------------------------------------------------------------- #
# Circuit + DEM construction                                                   #
# --------------------------------------------------------------------------- #


def build_circuit(level: CurriculumLevel) -> stim.Circuit:
    """Generate a Stim ``rotated_memory_z`` circuit at the given level."""
    rates = SI1000Rates.from_p(level.p)
    return stim.Circuit.generated(
        CODE_TASK,
        distance=level.distance,
        rounds=level.rounds,
        **rates.as_stim_kwargs(),
    )


def build_dem(circuit: stim.Circuit) -> stim.DetectorErrorModel:
    """Decompose-errors=True is mandatory for PyMatching."""
    return circuit.detector_error_model(decompose_errors=True)


def dem_digest(dem: stim.DetectorErrorModel) -> str:
    """8-char digest of the DEM, useful for grouping training logs."""
    return hashlib.sha256(str(dem).encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# Layout introspection - figure out data qubits, ancillas, observable support  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CircuitLayout:
    """Static facts about a circuit, computed once per episode.

    Two indexings coexist:

    * **Stim IDs** (``data_qubits``) are the physical qubit IDs Stim emits
      (e.g. ``(1, 3, 5, 8, 10, 12, 15, 17, 19)`` for distance-3). These are
      what Stim/PyMatching speak.
    * **LLM IDs** are consecutive ``0..num_data_qubits-1``. These are what
      the prompt advertises and what the LLM emits, because consecutive
      small ints are dramatically easier for a language model to handle.

    :meth:`llm_to_stim` and :meth:`stim_to_llm` perform the remap. *All*
    server-internal scoring uses Stim IDs; the boundary at the prompt
    formatter / parser converts.
    """

    data_qubits: tuple[int, ...]
    """Stim IDs of data qubits (measured by terminal ``M``), sorted."""

    data_qubit_coords: tuple[tuple[float, float], ...]
    """(x, y) coordinate of each data qubit, in the same order as
    ``data_qubits``. Used by Reward 3 to snap PyMatching edges to qubits."""

    ancilla_qubits: tuple[int, ...]
    """Physical qubit IDs that hold stabiliser measurements (``MR``)."""

    z_observable_support: tuple[int, ...]
    """Data qubits whose Z value is XOR'd into the logical Z observable.
    An X error on any of these flips the observable."""

    detector_round: tuple[int, ...]
    """For each detector index, the round it nominally belongs to (0-based,
    extracted from the ``DETECTOR(x, y, t)`` coordinate)."""

    detector_coords: tuple[tuple[float, float], ...]
    """(x, y) coordinate of each detector, used by Reward 3."""

    detector_is_x_type: tuple[bool, ...]
    """Whether the detector watches an X-stabiliser. For the rotated surface
    code Stim places X-stabilisers at coordinates with ``(x + y) mod 4 == 2``
    and Z-stabilisers at ``(x + y) mod 4 == 0`` (verified empirically against
    Stim 1.15's ``surface_code:rotated_memory_z``)."""

    final_detectors: tuple[int, ...]
    """Indices of detectors that correspond to the *last* timeslice - those
    are the only detectors a terminal Pauli frame can affect (Reward 2)."""

    num_data_qubits: int
    num_ancilla_qubits: int
    num_detectors: int
    num_observables: int

    # ----- LLM <-> Stim qubit-ID remapping ---------------------------------

    def llm_to_stim(self, llm_ids: list[int]) -> list[int]:
        """Convert consecutive LLM IDs to physical Stim IDs.

        Out-of-range IDs are silently dropped (the parser already enforces
        the upper bound, but we double-check here as a defence-in-depth).
        """
        out: list[int] = []
        n = len(self.data_qubits)
        for i in llm_ids:
            if 0 <= i < n:
                out.append(self.data_qubits[i])
        return out

    def stim_to_llm(self, stim_ids: list[int]) -> list[int]:
        """Inverse of :meth:`llm_to_stim` - used to render targets in the
        SFT data and the imitator policy."""
        lookup = {q: i for i, q in enumerate(self.data_qubits)}
        return [lookup[q] for q in stim_ids if q in lookup]


def _walk_measurement_records(
    circuit: stim.Circuit,
) -> tuple[list[int], list[Optional[str]]]:
    """Replay the circuit (no sampling) to map each measurement record to a
    qubit. Returns parallel lists: qubits[i] = qubit id, instr[i] = gate."""
    qubits: list[int] = []
    instrs: list[Optional[str]] = []

    def _walk(c: stim.Circuit, repeats: int = 1) -> None:
        for _ in range(repeats):
            for inst in c:
                if isinstance(inst, stim.CircuitRepeatBlock):
                    _walk(inst.body_copy(), inst.repeat_count)
                    continue
                name = inst.name
                if name in {
                    "M", "MX", "MY", "MZ",
                    "MR", "MRX", "MRY", "MRZ",
                    "MPP",
                }:
                    for t in inst.targets_copy():
                        if t.is_qubit_target:
                            qubits.append(t.qubit_value)
                            instrs.append(name)

    _walk(circuit)
    return qubits, instrs


def extract_layout(circuit: stim.Circuit) -> CircuitLayout:
    """Walk the circuit once to build a full :class:`CircuitLayout`."""
    flat = circuit.flattened()
    measurement_qubits, measurement_instrs = _walk_measurement_records(circuit)

    # Data qubits = those measured by terminal ``M`` (destructive, no reset).
    data_qubits_in_order: list[int] = []
    seen_data = set()
    for q, instr in zip(measurement_qubits, measurement_instrs):
        if instr == "M" and q not in seen_data:
            data_qubits_in_order.append(q)
            seen_data.add(q)
    data_qubits = tuple(sorted(seen_data))

    # Ancilla qubits = everything measured by MR (reset after measurement).
    ancilla_qubits = tuple(
        sorted({q for q, instr in zip(measurement_qubits, measurement_instrs)
                if instr == "MR"})
    )

    # Observable support: walk OBSERVABLE_INCLUDE entries and resolve their
    # rec[-k] back to qubit IDs via the measurement record table.
    obs_support: dict[int, set[int]] = {}
    for inst in flat:
        if inst.name == "OBSERVABLE_INCLUDE":
            args = inst.gate_args_copy()
            obs_idx = int(args[0]) if args else 0
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    actual = len(measurement_qubits) + t.value  # value is negative
                    if 0 <= actual < len(measurement_qubits):
                        obs_support.setdefault(obs_idx, set()).add(
                            measurement_qubits[actual]
                        )
    z_obs = tuple(sorted(obs_support.get(0, set())))

    # Qubit coordinates from QUBIT_COORDS instructions.
    qubit_coords: dict[int, tuple[float, float]] = {}
    for inst in flat:
        if inst.name == "QUBIT_COORDS":
            args = inst.gate_args_copy()
            x = float(args[0]) if len(args) >= 1 else 0.0
            y = float(args[1]) if len(args) >= 2 else 0.0
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    qubit_coords[t.qubit_value] = (x, y)
    data_qubit_coords = tuple(qubit_coords.get(q, (0.0, 0.0)) for q in data_qubits)

    # Detector coordinates - last value of the tuple is the round index.
    det_coords_raw = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors
    rounds_per_det: list[int] = []
    is_x_type: list[bool] = []
    detector_coords: list[tuple[float, float]] = []
    for i in range(num_dets):
        c = det_coords_raw.get(i, ())
        if not c:
            rounds_per_det.append(0)
            is_x_type.append(False)
            detector_coords.append((0.0, 0.0))
            continue
        round_idx = int(c[-1]) if len(c) >= 3 else 0
        rounds_per_det.append(round_idx)
        x = float(c[0]) if len(c) >= 1 else 0.0
        y = float(c[1]) if len(c) >= 2 else 0.0
        detector_coords.append((x, y))
        # X-stabilisers sit at (x + y) % 4 == 2 in Stim's generator.
        is_x_type.append((int(x + y) % 4) == 2)

    final_round = max(rounds_per_det) if rounds_per_det else 0
    final_dets = tuple(i for i, r in enumerate(rounds_per_det) if r == final_round)

    return CircuitLayout(
        data_qubits=data_qubits,
        data_qubit_coords=data_qubit_coords,
        ancilla_qubits=ancilla_qubits,
        z_observable_support=z_obs,
        detector_round=tuple(rounds_per_det),
        detector_coords=tuple(detector_coords),
        detector_is_x_type=tuple(is_x_type),
        final_detectors=final_dets,
        num_data_qubits=len(data_qubits),
        num_ancilla_qubits=len(ancilla_qubits),
        num_detectors=num_dets,
        num_observables=circuit.num_observables,
    )


# --------------------------------------------------------------------------- #
# Sampling and decoding                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SyndromeSample:
    """One noisy episode: detector activations, ground-truth observable
    flip, and PyMatching's prediction (used by Rewards 1 and 5)."""

    syndrome_bits: list[int]
    actual_observable_flip: int
    pymatching_observable_pred: int
    pymatching_x_errors: list[int]   # Pauli frame at end of circuit (X part)
    pymatching_z_errors: list[int]   # Pauli frame at end of circuit (Z part)


def sample_episode(
    circuit: stim.Circuit,
    matching: pymatching.Matching,
    layout: CircuitLayout,
    seed: Optional[int] = None,
) -> SyndromeSample:
    """Sample one shot, decode it with PyMatching, and bundle the result."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    detection, observables = sampler.sample(1, separate_observables=True)
    detection_row = detection[0].astype(np.uint8)
    observable_flip = int(observables[0, 0]) if observables.shape[1] else 0

    # PyMatching's prediction (observable level).
    pred_obs = int(matching.decode(detection_row)[0])

    # PyMatching's predicted physical Pauli frame on data qubits.
    pred_x, pred_z = pymatching_predicted_pauli_frame(
        matching=matching, syndrome=detection_row, layout=layout,
    )

    return SyndromeSample(
        syndrome_bits=detection_row.tolist(),
        actual_observable_flip=observable_flip,
        pymatching_observable_pred=pred_obs,
        pymatching_x_errors=pred_x,
        pymatching_z_errors=pred_z,
    )


def pymatching_predicted_pauli_frame(
    matching: pymatching.Matching,
    syndrome: np.ndarray,
    layout: CircuitLayout,
) -> tuple[list[int], list[int]]:
    """Convert PyMatching's per-edge prediction into a data-qubit Pauli frame.

    The matching graph's edges correspond to error mechanisms in the DEM.
    Each edge connects two detectors (or a detector and a boundary). The
    data qubit responsible for the edge sits geometrically between the two
    detectors on the surface-code grid - we recover it by snapping the
    midpoint of the detector coordinates to the nearest data qubit.

    This frame is used as ground-truth for Reward 3 (Hamming overlap).
    Z-stabiliser endpoints (``(x+y) mod 4 == 0``) catch X errors on data
    qubits; X-stabiliser endpoints catch Z errors. Boundary edges are
    snapped to the unique data qubit adjacent to that boundary.
    """
    try:
        edges = matching.decode_to_edges_array(syndrome)
    except Exception:
        return [], []

    if edges is None or len(edges) == 0:
        return [], []

    data_qubits = layout.data_qubits
    data_coords = layout.data_qubit_coords
    det_coords = layout.detector_coords
    det_is_x = layout.detector_is_x_type
    n_dets = len(det_coords)

    def _snap(x: float, y: float) -> int:
        best_q = data_qubits[0]
        best_d = float("inf")
        for q, (qx, qy) in zip(data_qubits, data_coords):
            d = (qx - x) ** 2 + (qy - y) ** 2
            if d < best_d:
                best_d = d
                best_q = q
        return best_q

    x_errs: set[int] = set()
    z_errs: set[int] = set()
    for edge in edges:
        a, b = int(edge[0]), int(edge[1])
        ca = det_coords[a] if 0 <= a < n_dets else None
        cb = det_coords[b] if 0 <= b < n_dets else None
        if ca is None and cb is None:
            continue
        if cb is None:
            mid_x, mid_y = ca
            ref_is_x = det_is_x[a]
        elif ca is None:
            mid_x, mid_y = cb
            ref_is_x = det_is_x[b]
        else:
            mid_x = (ca[0] + cb[0]) / 2.0
            mid_y = (ca[1] + cb[1]) / 2.0
            ref_is_x = det_is_x[a] if 0 <= a < n_dets else det_is_x[b]
        snap = _snap(mid_x, mid_y)
        if ref_is_x:
            z_errs.add(snap)
        else:
            x_errs.add(snap)

    return sorted(x_errs), sorted(z_errs)


# --------------------------------------------------------------------------- #
# Predicted-observable computation (used by Reward 1)                          #
# --------------------------------------------------------------------------- #


def predicted_observable_flip(
    predicted_x_qubits: list[int],
    layout: CircuitLayout,
) -> int:
    """Compute the implied logical-Z flip from a predicted Pauli frame.

    Only X errors on data qubits in ``z_observable_support`` matter for the
    Z observable - Z errors on data qubits commute with the destructive Z
    measurement and so cannot flip the observable.
    """
    support = set(layout.z_observable_support)
    parity = 0
    for q in predicted_x_qubits:
        if q in support:
            parity ^= 1
    return parity


def rectify_pauli_frame_to_observable(
    x_errors: list[int],
    z_errors: list[int],
    target_observable_flip: int,
    layout: CircuitLayout,
) -> tuple[list[int], list[int]]:
    """Adjust a predicted X-error frame so its implied observable matches.

    Used by the SFT data generator and the PyMatching imitator policy: the
    snap-to-data-qubit edge mapping (:func:`pymatching_predicted_pauli_frame`)
    is only ~95% faithful, but PyMatching's *observable* prediction is exact.
    We patch the X frame by toggling the smallest-degree data qubit on the
    observable support whenever the implied parity disagrees with the
    target. Z errors are untouched because they don't affect a Z observable.
    """
    implied = predicted_observable_flip(x_errors, layout)
    if implied == target_observable_flip:
        return list(x_errors), list(z_errors)

    support = list(layout.z_observable_support)
    if not support:
        return list(x_errors), list(z_errors)

    x_set = set(x_errors)
    intersect = sorted(x_set & set(support))
    if intersect:
        # Remove the smallest one currently flipping the observable.
        x_set.discard(intersect[0])
    else:
        # Add the smallest support qubit to introduce a flip.
        x_set.add(support[0])
    return sorted(x_set), list(z_errors)


# --------------------------------------------------------------------------- #
# Stabiliser counts - derived from layout                                      #
# --------------------------------------------------------------------------- #


def detector_round_split(layout: CircuitLayout, syndrome_bits: list[int]) -> dict[int, list[int]]:
    """Group detector bits by their nominal round (used for prompt formatting)."""
    out: dict[int, list[int]] = {}
    for idx, bit in enumerate(syndrome_bits):
        r = layout.detector_round[idx] if idx < len(layout.detector_round) else 0
        out.setdefault(r, []).append(bit)
    return out


def per_round_x_z_counts(layout: CircuitLayout) -> tuple[int, int]:
    """Best-effort count of X-type and Z-type stabiliser detectors per round.

    For a rotated surface code at distance d there are (d^2-1)/2 of each
    type. We compute that from the layout to be robust.
    """
    # Take one fully-populated round (the one with the most detectors).
    round_counts: dict[int, list[bool]] = {}
    for idx, r in enumerate(layout.detector_round):
        round_counts.setdefault(r, []).append(layout.detector_is_x_type[idx])
    if not round_counts:
        return 0, 0
    full_round = max(round_counts.values(), key=len)
    n_x = sum(1 for v in full_round if v)
    n_z = sum(1 for v in full_round if not v)
    return n_x, n_z
