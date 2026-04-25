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
    parsed = ParseResult([1], [], parse_success=True, parse_partial=False, raw_response="")
    assert reward_format_compliance(parsed) == 1.0
    parsed = ParseResult([1], [], parse_success=False, parse_partial=True, raw_response="")
    assert reward_format_compliance(parsed) == 0.5
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
