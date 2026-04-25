"""Two equivalent client implementations:

* :class:`DecoderClient`     - hits an HTTP endpoint (HF Spaces deployment).
  Speaks the **OpenEnv** wire format:
  - ``POST /reset`` body ``{"seed": int?, "episode_id": str?}``
  - ``POST /step``  body ``{"action": {"raw_response": "...", ...},
    "timeout_s": float?, "request_id": str?}``
* :class:`LocalDecoderClient` - calls the in-process env directly. Use this
  in tests, in CI, and during local Colab runs to skip HTTP overhead.

Both expose the same ``reset`` / ``step`` API so the training scripts can
swap between them via a single env var (``QUBIT_MEDIC_URL``).
"""
from __future__ import annotations

import os
from typing import Optional, Protocol

import httpx

from qubit_medic.models import (
    DecoderObservation,
    StepResult,
)


class _ClientProtocol(Protocol):
    def reset(self, *, seed: Optional[int] = None,
              forced_level: Optional[str] = None) -> DecoderObservation: ...
    def step(self, *, raw_response: str, episode_id: int) -> StepResult: ...
    def health(self) -> dict: ...
    def close(self) -> None: ...


def _obs_from_openenv(payload: dict) -> DecoderObservation:
    """Re-hydrate our internal :class:`DecoderObservation` from the
    OpenEnv response body. The OpenEnv wrapper inlines all our fields
    onto the observation, so this is a 1-1 field mapping."""
    return DecoderObservation(
        prompt=payload.get("prompt", ""),
        syndrome_bits=list(payload.get("syndrome_bits", [])),
        distance=int(payload.get("distance", 0)),
        rounds=int(payload.get("rounds", 0)),
        p=float(payload.get("p", 0.0)),
        curriculum_level=payload.get("curriculum_level", ""),
        episode_id=int(payload.get("episode_id", 0)),
        dem_digest=payload.get("dem_digest", ""),
    )


class DecoderClient:
    """HTTP client targeting a deployed FastAPI server (OpenEnv shape)."""

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def reset(self, *, seed: Optional[int] = None,
              forced_level: Optional[str] = None) -> DecoderObservation:
        # OpenEnv's ResetRequest only accepts seed + episode_id. We pass
        # forced_level via the URL query string so adapters that honour
        # it (our QubitMedicEnvironment via **kwargs) pick it up; servers
        # that ignore it just get a default level.
        body: dict = {}
        if seed is not None:
            body["seed"] = seed
        params = {"forced_level": forced_level} if forced_level else None
        r = self._client.post("/reset", json=body, params=params)
        r.raise_for_status()
        payload = r.json()
        # OpenEnv returns {observation: {...}, reward, done}.
        return _obs_from_openenv(payload.get("observation", payload))

    def step(self, *, raw_response: str, episode_id: int) -> StepResult:
        body = {
            "action": {
                "raw_response": raw_response,
                "episode_id": episode_id,
            },
        }
        r = self._client.post("/step", json=body)
        r.raise_for_status()
        payload = r.json()
        obs_payload = payload.get("observation", {})
        return StepResult(
            observation=_obs_from_openenv(obs_payload),
            reward=float(payload.get("reward", 0.0) or 0.0),
            done=bool(payload.get("done", True)),
            truncated=bool(obs_payload.get("info", {}).get("timed_out", False)),
            info=dict(obs_payload.get("info", {})),
        )

    def health(self) -> dict:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def healthz(self) -> dict:
        r = self._client.get("/healthz")
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()


class LocalDecoderClient:
    """In-process client - calls :class:`DecoderEnvironment` directly."""

    def __init__(self, env=None) -> None:
        from qubit_medic.server.environment import DecoderEnvironment
        self._env = env if env is not None else DecoderEnvironment()

    def reset(self, *, seed: Optional[int] = None,
              forced_level: Optional[str] = None) -> DecoderObservation:
        return self._env.reset(seed=seed, forced_level=forced_level)

    def step(self, *, raw_response: str, episode_id: int) -> StepResult:
        return self._env.step(raw_response=raw_response, episode_id=episode_id)

    def health(self) -> dict:
        return self._env.health()

    def close(self) -> None:  # nothing to clean up
        pass


def make_default_client() -> _ClientProtocol:
    """Return :class:`DecoderClient` if ``QUBIT_MEDIC_URL`` is set, else local."""
    url = os.getenv("QUBIT_MEDIC_URL")
    if url:
        return DecoderClient(url)
    return LocalDecoderClient()
