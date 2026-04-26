"""Tests for the five reward functions.

These run against a real Stim circuit so they double as a regression test
on the physics layer too. Kept fast (<1s total) by reusing one cached
circuit.
"""
from __future__ import annotations

import numpy as np
import pymatching
import pytest

from qubit_medic.config import primary_level
from qubit_medic.prompts import ParseResult
from qubit_medic.server.physics import (
    build_circuit, build_dem, extract_layout, sample_episode,
)
from qubit_medic.server.rewards import (
    compute_all_rewards, compute_final_detector_supports,
    reward_format_compliance, reward_logical_correction,
    reward_pymatching_beat,
)


@pytest.fixture(scope="module")
def env_fixture():
    level = primary_level()
    circuit = build_circuit(level)
    dem = build_dem(circuit)
    matching = pymatching.Matching.from_detector_error_model(dem)
    layout = extract_layout(circuit)
    supports = compute_final_detector_supports(layout)
    return circuit, matching, layout, supports


def test_format_compliance_levels():
    # 2026-04 spec rewrite (FIX 1): format compliance is now binary
    # {0.0, 1.0} - partial credit (0.5) was removed because it left the
    # reward landscape too flat for GRPO to escape the empty-everywhere
    # mode. Full parse -> 1.0; anything else -> 0.0.
    parsed = ParseResult([1], [], parse_success=True, parse_partial=False, raw_response="")
    assert reward_format_compliance(parsed) == 1.0
    parsed = ParseResult([1], [], parse_success=False, parse_partial=True, raw_response="")
    assert reward_format_compliance(parsed) == 0.0
    parsed = ParseResult([], [], parse_success=False, parse_partial=False, raw_response="")
    assert reward_format_compliance(parsed) == 0.0


def test_logical_correction_zero_error_pred(env_fixture):
    circuit, matching, layout, _ = env_fixture
    # Find a syndrome with no observable flip (~99.8% of the time at p=0.001).
    for s in range(20):
        sample = sample_episode(circuit, matching, layout, seed=s)
        if sample.actual_observable_flip == 0:
            break
    parsed = ParseResult([], [], parse_success=True, parse_partial=False, raw_response="")
    r = reward_logical_correction(parsed, sample, layout)
    assert r == 1.0


def test_logical_correction_all_obs_support(env_fixture):
    """Predicting an X error on every observable-support qubit flips the
    parity len(support) times. For odd support that's a flip; for even,
    no flip. d=3 has support of size 3, so this should claim a flip."""
    circuit, matching, layout, _ = env_fixture
    for s in range(20):
        sample = sample_episode(circuit, matching, layout, seed=s)
        if sample.actual_observable_flip == 0:
            break
    # Predict flips on all support qubits (size 3 -> implied parity = 1).
    parsed = ParseResult(list(layout.z_observable_support), [],
                         parse_success=True, parse_partial=False,
                         raw_response="")
    r = reward_logical_correction(parsed, sample, layout)
    # Should disagree with actual_observable_flip == 0.
    assert r == 0.0


def test_pymatching_beat_only_when_pm_wrong(env_fixture):
    """If PM is right, beat is always 0 regardless of the LLM's answer."""
    circuit, matching, layout, _ = env_fixture
    # Pick any seed; with high probability PM is right at p=0.001.
    sample = sample_episode(circuit, matching, layout, seed=11)
    if sample.pymatching_observable_pred == sample.actual_observable_flip:
        parsed = ParseResult([], [], parse_success=True, parse_partial=False,
                             raw_response="")
        assert reward_pymatching_beat(parsed, sample, layout) == 0.0


def test_compute_all_rewards_in_unit_interval(env_fixture):
    circuit, matching, layout, supports = env_fixture
    sample = sample_episode(circuit, matching, layout, seed=7)
    parsed = ParseResult([0, 1], [2], parse_success=True, parse_partial=False,
                         raw_response="X_ERRORS=[0,1] Z_ERRORS=[2]")
    r = compute_all_rewards(parsed, sample, layout, supports)
    assert 0.0 <= r.total <= 1.0
    for v in r.as_dict().values():
        assert 0.0 <= v <= 1.0


def test_empty_prediction_on_nonempty_truth_is_penalised(env_fixture):
    """Regression guard for FIX 1 (2026-04 RL spec rewrite).

    Pre-fix: predicting empty on a non-empty syndrome could score ~0.85
    on the weighted total because hamming_overlap returned 1.0 for two
    empty sets and syndrome_consistency saw plausibly-zero implied bits.
    Post-fix: hamming_overlap is set-aware (missed-error case scores
    0.0) and syndrome_consistency caps empty predictions on non-empty
    syndromes at 0.5. Empty-everywhere should score WELL below 0.85.
    """
    from dataclasses import replace
    from qubit_medic.server.physics import sample_episode
    circuit, matching, layout, supports = env_fixture

    # Find a syndrome with at least one true X or Z error so the new
    # set-aware Jaccard rule has something to bite on.
    sample = None
    for s in range(200):
        cand = sample_episode(circuit, matching, layout, seed=s)
        if cand.true_x_errors or cand.true_z_errors:
            sample = cand
            break
    assert sample is not None, "no non-trivial syndrome in 200 seeds"

    empty_pred = ParseResult([], [], parse_success=True, parse_partial=False,
                             raw_response="X_ERRORS=[] Z_ERRORS=[]",
                             strict_format=True)
    r = compute_all_rewards(empty_pred, sample, layout, supports)
    # Headline claim from the spec: weighted total should be well below
    # the pre-fix ~0.85; the spec calls out "~0.5" as the new ceiling.
    assert r.total <= 0.6, (
        f"empty prediction on non-empty truth should not score above 0.6; "
        f"got total={r.total:.3f} breakdown={r.as_dict()}"
    )
    # Hamming overlap on missed errors is exactly 0.0 by the new rule.
    # Whether jx OR jz hit 0.0 depends on which axis carries the truth;
    # the average is 0.5 only if exactly one axis is non-empty.
    assert r.hamming_overlap <= 0.5, (
        f"hamming_overlap on empty/non-empty axis should be <= 0.5; "
        f"got {r.hamming_overlap:.3f}"
    )


def test_set_aware_jaccard_rule(env_fixture):
    """Direct unit test of the four-case rule in _set_aware_jaccard."""
    from qubit_medic.server.rewards import _set_aware_jaccard
    assert _set_aware_jaccard([], []) == 1.0          # empty / empty
    assert _set_aware_jaccard([], [1]) == 0.0         # false alarm
    assert _set_aware_jaccard([1], []) == 0.0         # missed errors
    assert _set_aware_jaccard([1], [1]) == 1.0        # exact agreement
    assert _set_aware_jaccard([1, 2], [2, 3]) == 1/3  # standard Jaccard


def test_pymatching_imitator_logical_correction_high(env_fixture):
    """Sanity check: PyMatching imitator should hit ~99% LER at p=0.001."""
    from qubit_medic.server.physics import (
        pymatching_predicted_pauli_frame, rectify_pauli_frame_to_observable,
    )
    circuit, matching, layout, supports = env_fixture
    correct = 0
    n = 200
    for s in range(n):
        sample = sample_episode(circuit, matching, layout, seed=1000 + s)
        syndrome = np.asarray(sample.syndrome_bits, dtype=np.uint8)
        px, pz = pymatching_predicted_pauli_frame(matching, syndrome, layout)
        pm_obs = int(matching.decode(syndrome)[0])
        px, pz = rectify_pauli_frame_to_observable(px, pz, pm_obs, layout)
        parsed = ParseResult(px, pz, parse_success=True, parse_partial=False,
                             raw_response="")
        r = reward_logical_correction(parsed, sample, layout)
        correct += int(r > 0.5)
    assert correct / n >= 0.97
