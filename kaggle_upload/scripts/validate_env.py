"""scripts/validate_env.py - the Section 1.1 environment validation.

Five gates, in order:

    1.  Imports succeed (catches install issues).
    2.  Stim generates a tiny distance-3 surface code.
    3.  PyMatching decodes 100 syndromes.
    4.  Logical-error rate at p=0.001 is in the expected range.
    5.  ``DecoderEnvironment`` reset+step works end-to-end (proves the wire
        contract is intact).

Run with::

    .venv/bin/python -m scripts.validate_env

Exit code is 0 iff every gate passes. The participant guide explicitly
warns: *"if any of these fail on any team member's machine, fix it now -
not at 11pm on Day 1."*
"""
from __future__ import annotations

import sys
import time
from typing import Iterable

GATES = []


def gate(name: str):
    def deco(fn):
        GATES.append((name, fn))
        return fn
    return deco


def _ok(name: str, msg: str = "") -> None:
    extra = f" {msg}" if msg else ""
    print(f"  PASS  {name}{extra}")


def _fail(name: str, msg: str) -> None:
    print(f"  FAIL  {name} -- {msg}")


# --------------------------------------------------------------------------- #


@gate("imports")
def _imports() -> None:
    import stim, pymatching, numpy, fastapi, pydantic  # noqa: F401
    import qubit_medic
    import qubit_medic.config
    import qubit_medic.models
    import qubit_medic.prompts
    import qubit_medic.server.physics
    import qubit_medic.server.rewards
    import qubit_medic.server.curriculum
    import qubit_medic.server.environment
    print(f"      stim={stim.__version__}  pymatching={pymatching.__version__}  "
          f"qubit_medic={qubit_medic.__version__}")


@gate("stim_circuit_generation")
def _stim_gen() -> None:
    from qubit_medic.config import primary_level
    from qubit_medic.server.physics import build_circuit, build_dem, extract_layout
    c = build_circuit(primary_level())
    dem = build_dem(c)
    layout = extract_layout(c)
    assert layout.num_data_qubits == 9, f"expected 9 data qubits, got {layout.num_data_qubits}"
    assert layout.num_ancilla_qubits == 8
    assert layout.z_observable_support == (1, 3, 5)
    print(f"      circuit={len(str(c))} chars, DEM={len(str(dem))} chars, "
          f"obs_support={layout.z_observable_support}")


@gate("pymatching_decoding_100")
def _pm_decoding() -> None:
    import pymatching, numpy as np
    from qubit_medic.config import primary_level
    from qubit_medic.server.physics import build_circuit, build_dem
    c = build_circuit(primary_level())
    dem = build_dem(c)
    sampler = c.compile_detector_sampler(seed=42)
    det, obs = sampler.sample(100, separate_observables=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    pred = m.decode_batch(det)
    err_rate = float(np.mean(np.any(pred != obs, axis=1)))
    print(f"      logical-error rate (100 shots): {err_rate:.4f}")


@gate("ler_in_expected_range")
def _ler_range() -> None:
    """At distance 3, p=0.001, 5000 shots, PyMatching LER should be < 1%."""
    import pymatching, numpy as np
    from qubit_medic.config import primary_level
    from qubit_medic.server.physics import build_circuit, build_dem
    c = build_circuit(primary_level())
    dem = build_dem(c)
    sampler = c.compile_detector_sampler(seed=2024)
    det, obs = sampler.sample(5000, separate_observables=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    pred = m.decode_batch(det)
    err = float(np.mean(np.any(pred != obs, axis=1)))
    expected_lo, expected_hi = 0.0, 0.01
    if not (expected_lo <= err <= expected_hi):
        raise AssertionError(
            f"PyMatching LER {err:.4f} outside [{expected_lo}, {expected_hi}]"
        )
    print(f"      PyMatching LER on 5000 shots: {err:.4f} "
          f"(expected ~0.001 - 0.01)")


@gate("decoder_environment_roundtrip")
def _env_roundtrip() -> None:
    """Reset + step round-trip with three trivial policies."""
    from qubit_medic.client.client import LocalDecoderClient
    from qubit_medic.prompts import format_completion

    client = LocalDecoderClient()
    obs = client.reset(forced_level="L2_target", seed=1)
    assert obs.distance == 3 and obs.rounds == 3
    assert obs.curriculum_level == "L2_target"

    # All-zeros policy: claim no errors.
    result = client.step(
        raw_response=format_completion([], []),
        episode_id=obs.episode_id,
    )
    assert result.done is True
    assert "rewards" in result.info
    print(f"      reset->step round-trip ok; "
          f"all-zeros total reward={result.reward:.3f}, "
          f"breakdown={result.info['rewards']}")

    # Trivial second episode under forced L1.
    obs2 = client.reset(forced_level="L1_warmup", seed=2)
    assert obs2.distance == 3 and obs2.rounds == 1
    print(f"      L1 warmup reset OK; prompt is {len(obs2.prompt)} chars long")


# --------------------------------------------------------------------------- #


def main(argv: Iterable[str] = ()) -> int:
    print("Qubit-Medic environment validation")
    print("=" * 60)
    failures = 0
    started = time.monotonic()
    for name, fn in GATES:
        try:
            fn()
            _ok(name)
        except Exception as exc:  # noqa: BLE001 - we want to keep going
            _fail(name, repr(exc))
            failures += 1
    elapsed = time.monotonic() - started
    print("=" * 60)
    if failures:
        print(f"{failures} gate(s) failed in {elapsed:.2f}s")
        return 1
    print(f"all {len(GATES)} gates passed in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
