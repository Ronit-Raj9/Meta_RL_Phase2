"""DecoderEnvironment: the OpenEnv-style env that the LLM trainer talks to.

This is the heart of the server (Sections 2.4 + 2.5 of the plan):

* ``reset()``: pick a curriculum level, build a circuit, sample a syndrome,
  return a :class:`DecoderObservation`.
* ``step(raw_response)``: parse the LLM's text, score five rewards, return
  a :class:`StepResult` whose ``info`` dict carries the per-component
  breakdown.

Episodes are single-step (Section 2.5): the LLM emits one prediction and
the episode ends.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import pymatching

from qubit_medic.config import (
    EPISODE_TIMEOUT_SECONDS,
    PRIMARY_SEED,
    REWARD_WEIGHTS,
)
from qubit_medic.models import (
    DecoderAction,
    DecoderObservation,
    DecoderState,
    StepResult,
)
from qubit_medic.prompts import build_prompt, parse_action
from qubit_medic.server import physics
from qubit_medic.server.curriculum import CurriculumScheduler
from qubit_medic.server.physics import (
    CircuitLayout,
    SyndromeSample,
    build_circuit,
    build_dem,
    dem_digest,
    extract_layout,
    per_round_x_z_counts,
    sample_episode,
)
from qubit_medic.server.rewards import (
    RewardBreakdown,
    compute_all_rewards,
    compute_final_detector_supports,
)


# --------------------------------------------------------------------------- #
# Per-level cached compilation - building Stim/PyMatching is the slow step    #
# --------------------------------------------------------------------------- #


@dataclass
class _LevelCache:
    """Compiled Stim/PyMatching artefacts for one curriculum level."""
    circuit: object
    dem: object
    matching: pymatching.Matching
    layout: CircuitLayout
    final_detector_supports: dict
    dem_digest: str

    @classmethod
    def build(cls, level) -> "_LevelCache":
        c = build_circuit(level)
        d = build_dem(c)
        m = pymatching.Matching.from_detector_error_model(d)
        layout = extract_layout(c)
        supports = compute_final_detector_supports(layout)
        return cls(
            circuit=c,
            dem=d,
            matching=m,
            layout=layout,
            final_detector_supports=supports,
            dem_digest=dem_digest(d),
        )


# --------------------------------------------------------------------------- #
# DecoderEnvironment                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class _ActiveEpisode:
    """In-flight episode bookkeeping."""
    state: DecoderState
    sample: SyndromeSample
    layout: CircuitLayout
    final_detector_supports: dict
    started_at: float


class DecoderEnvironment:
    """OpenEnv-style env for surface-code decoding.

    Thread-safe by virtue of a single ``_lock``: the FastAPI server is
    expected to be I/O bound, and per-call latency is well under a
    millisecond, so a coarse lock is fine and dramatically simplifies the
    state machine.
    """

    def __init__(self, *, base_seed: int = PRIMARY_SEED) -> None:
        self._lock = threading.Lock()
        self._scheduler = CurriculumScheduler(rng=__import__("random").Random(base_seed))
        self._caches: dict[str, _LevelCache] = {}
        self._episode_counter = 0
        self._base_seed = base_seed
        self._active: dict[int, _ActiveEpisode] = {}

    # ----- cache helpers --------------------------------------------------

    def _cache_for(self, level_name: str):
        cache = self._caches.get(level_name)
        if cache is not None:
            return cache
        from qubit_medic.config import level_by_name
        cache = _LevelCache.build(level_by_name(level_name))
        self._caches[level_name] = cache
        return cache

    # ----- public API -----------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        forced_level: Optional[str] = None,
    ) -> DecoderObservation:
        with self._lock:
            self._episode_counter += 1
            ep_id = self._episode_counter
            shot_seed = seed if seed is not None else self._base_seed + ep_id
            level = self._scheduler.sample(forced_level=forced_level)
            cache = self._cache_for(level.name)

            sample = sample_episode(
                circuit=cache.circuit,
                matching=cache.matching,
                layout=cache.layout,
                seed=shot_seed,
            )

            state = DecoderState(
                episode_id=ep_id,
                seed=shot_seed,
                curriculum_level=level.name,
                distance=level.distance,
                rounds=level.rounds,
                p=level.p,
                syndrome_bits=sample.syndrome_bits,
                true_x_errors=sample.pymatching_x_errors,
                true_z_errors=sample.pymatching_z_errors,
                actual_observable_flip=sample.actual_observable_flip,
                pymatching_observable_pred=sample.pymatching_observable_pred,
                x_observable_support=[],  # memory_z task: no X observable
                z_observable_support=list(cache.layout.z_observable_support),
                num_data_qubits=cache.layout.num_data_qubits,
                num_stabilizers=cache.layout.num_ancilla_qubits,
                circuit_text=str(cache.circuit),
                dem_text=str(cache.dem),
            )
            self._active[ep_id] = _ActiveEpisode(
                state=state,
                sample=sample,
                layout=cache.layout,
                final_detector_supports=cache.final_detector_supports,
                started_at=time.monotonic(),
            )

            n_x, n_z = per_round_x_z_counts(cache.layout)
            prompt = build_prompt(
                distance=level.distance,
                rounds=level.rounds,
                p=level.p,
                syndrome_bits=sample.syndrome_bits,
                num_x_stabilizers=n_x,
                num_z_stabilizers=n_z,
                num_data_qubits=cache.layout.num_data_qubits,
            )

            return DecoderObservation(
                prompt=prompt,
                syndrome_bits=sample.syndrome_bits,
                distance=level.distance,
                rounds=level.rounds,
                p=level.p,
                curriculum_level=level.name,
                episode_id=ep_id,
                dem_digest=cache.dem_digest,
            )

    def step(self, raw_response: str, episode_id: int) -> StepResult:
        with self._lock:
            episode = self._active.pop(episode_id, None)
            if episode is None:
                # Calling step() on an unknown episode ID is a hard error -
                # the trainer didn't follow reset/step pairing.
                raise KeyError(f"unknown or already-finished episode {episode_id}")

            elapsed = time.monotonic() - episode.started_at
            timed_out = elapsed > EPISODE_TIMEOUT_SECONDS

            parsed = parse_action(
                raw_response=raw_response,
                num_data_qubits=episode.layout.num_data_qubits,
            )

            if timed_out:
                # Hard timeout: zero reward, mark format compliance as zero,
                # close the episode cleanly (Section 2.6).
                breakdown = RewardBreakdown(
                    logical_correction=0.0,
                    syndrome_consistency=0.0,
                    hamming_overlap=0.0,
                    format_compliance=0.0,
                    pymatching_beat=0.0,
                    total=0.0,
                )
                action = DecoderAction(
                    raw_response=raw_response,
                    parse_success=False,
                )
            else:
                # Convert LLM-space qubit IDs (0..N-1) to Stim IDs before
                # scoring; rewards operate in the Stim coordinate system.
                from qubit_medic.prompts import ParseResult
                parsed_stim = ParseResult(
                    x_errors=episode.layout.llm_to_stim(parsed.x_errors),
                    z_errors=episode.layout.llm_to_stim(parsed.z_errors),
                    parse_success=parsed.parse_success,
                    parse_partial=parsed.parse_partial,
                    raw_response=parsed.raw_response,
                )
                breakdown = compute_all_rewards(
                    parsed=parsed_stim,
                    sample=episode.sample,
                    layout=episode.layout,
                    final_detector_supports=episode.final_detector_supports,
                    weights=REWARD_WEIGHTS,
                )
                action = DecoderAction(
                    x_error_qubits=parsed.x_errors,
                    z_error_qubits=parsed.z_errors,
                    raw_response=raw_response,
                    parse_success=parsed.parse_success,
                )

            self._scheduler.update(
                episode.state.curriculum_level,
                logical_correction=breakdown.logical_correction,
            )

            episode.state.last_reward_breakdown = breakdown.as_dict()

            n_x, n_z = per_round_x_z_counts(episode.layout)
            prompt = build_prompt(
                distance=episode.state.distance,
                rounds=episode.state.rounds,
                p=episode.state.p,
                syndrome_bits=episode.state.syndrome_bits,
                num_x_stabilizers=n_x,
                num_z_stabilizers=n_z,
                num_data_qubits=episode.layout.num_data_qubits,
            )
            obs = DecoderObservation(
                prompt=prompt,
                syndrome_bits=episode.state.syndrome_bits,
                distance=episode.state.distance,
                rounds=episode.state.rounds,
                p=episode.state.p,
                curriculum_level=episode.state.curriculum_level,
                episode_id=episode.state.episode_id,
                dem_digest=episode.state.dem_text[:8],
            )

            info = {
                "rewards": breakdown.as_dict(),
                "parsed_action": action.model_dump(),
                "actual_observable_flip": episode.sample.actual_observable_flip,
                "pymatching_observable_pred": episode.sample.pymatching_observable_pred,
                "pymatching_x_errors": episode.sample.pymatching_x_errors,
                "pymatching_z_errors": episode.sample.pymatching_z_errors,
                "elapsed_seconds": elapsed,
                "timed_out": timed_out,
                "curriculum_stats": self._scheduler.stats(),
            }

            return StepResult(
                observation=obs,
                reward=breakdown.total,
                done=True,  # single-step episodes
                truncated=timed_out,
                info=info,
            )

    # ----- introspection --------------------------------------------------

    def health(self) -> dict:
        with self._lock:
            return {
                "ok": True,
                "episodes_started": self._episode_counter,
                "active_episodes": len(self._active),
                "curriculum": self._scheduler.stats(),
                "cached_levels": list(self._caches.keys()),
            }
