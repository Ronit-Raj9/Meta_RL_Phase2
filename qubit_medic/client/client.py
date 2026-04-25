"""Two equivalent client implementations:

* :class:`DecoderClient`     - hits an HTTP endpoint (HF Spaces deployment).
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
    ResetRequest,
    StepRequest,
    StepResult,
)


class _ClientProtocol(Protocol):
    def reset(self, *, seed: Optional[int] = None,
              forced_level: Optional[str] = None) -> DecoderObservation: ...
    def step(self, *, raw_response: str, episode_id: int) -> StepResult: ...
    def health(self) -> dict: ...
    def close(self) -> None: ...


class DecoderClient:
    """HTTP client targeting a deployed FastAPI server."""

    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def reset(self, *, seed: Optional[int] = None,
              forced_level: Optional[str] = None) -> DecoderObservation:
        body = ResetRequest(seed=seed, forced_level=forced_level).model_dump(
            exclude_none=True
        )
        r = self._client.post("/reset", json=body)
        r.raise_for_status()
        return DecoderObservation.model_validate(r.json())

    def step(self, *, raw_response: str, episode_id: int) -> StepResult:
        body = StepRequest(raw_response=raw_response, episode_id=episode_id).model_dump()
        r = self._client.post("/step", json=body)
        r.raise_for_status()
        return StepResult.model_validate(r.json())

    def health(self) -> dict:
        r = self._client.get("/health")
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
