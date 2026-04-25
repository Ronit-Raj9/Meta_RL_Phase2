"""FastAPI app exposing the OpenEnv contract over HTTP.

Endpoints (Section 9 of the plan):

* ``POST /reset``   - start a new episode, returns a ``DecoderObservation``.
* ``POST /step``    - submit the LLM's raw text, returns a ``StepResult``.
* ``GET  /health``  - liveness + curriculum stats (used by HF Spaces).
* ``POST /decode``  - convenience endpoint for live demo. Takes a syndrome
  in the same wire format and returns the model's prediction. (The model
  is loaded lazily; if no checkpoint is mounted the endpoint returns a
  PyMatching baseline answer.)

Run with ``uvicorn qubit_medic.server.app:app --host 0.0.0.0 --port 7860``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from qubit_medic.config import DEFAULT_HOST, DEFAULT_PORT
from qubit_medic.models import (
    DecoderAction,
    DecoderObservation,
    ResetRequest,
    StepRequest,
    StepResult,
)
from qubit_medic.server.environment import DecoderEnvironment


logger = logging.getLogger("qubit_medic.server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


app = FastAPI(
    title="Qubit-Medic OpenEnv",
    description=(
        "RL training environment for LLM-based quantum error-correction "
        "decoders. Built on Stim + PyMatching with five independent "
        "verifiable rewards (see /health for live curriculum stats)."
    ),
    version=os.getenv("QUBIT_MEDIC_VERSION", "1.0.0"),
)


_env: Optional[DecoderEnvironment] = None


def get_env() -> DecoderEnvironment:
    global _env
    if _env is None:
        _env = DecoderEnvironment()
        # Pre-warm L1 + L2 caches so the first /reset doesn't pay compile cost.
        _env._cache_for("L1_warmup")  # noqa: SLF001 - intentional pre-warm
        _env._cache_for("L2_target")  # noqa: SLF001
    return _env


# --------------------------------------------------------------------------- #
# OpenEnv contract                                                             #
# --------------------------------------------------------------------------- #


@app.post("/reset", response_model=DecoderObservation)
def reset(req: ResetRequest = Body(default=ResetRequest())) -> DecoderObservation:
    """Start a new episode."""
    return get_env().reset(seed=req.seed, forced_level=req.forced_level)


@app.post("/step", response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    """Submit the LLM's raw response and get rewards back."""
    try:
        return get_env().step(raw_response=req.raw_response, episode_id=req.episode_id)
    except KeyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    """Liveness probe + curriculum stats. Used by HF Spaces."""
    return get_env().health()


# --------------------------------------------------------------------------- #
# Demo endpoint - takes a raw syndrome, returns a prediction                   #
# --------------------------------------------------------------------------- #


@app.post("/decode")
def decode(syndrome: list[int] = Body(..., embed=True),
           level: str = Body("L2_target", embed=True)) -> dict:
    """Decode an arbitrary syndrome with PyMatching (baseline) and return its
    predicted Pauli frame and observable flip.

    This endpoint is intended for the live demo (Section 9.2): a Gradio app
    can POST a hand-crafted syndrome here and visualise the result. When a
    trained LLM checkpoint is mounted, replace the body with a call to the
    LLM's predict function.
    """
    import numpy as np
    env = get_env()
    cache = env._cache_for(level)  # noqa: SLF001
    arr = np.asarray(syndrome, dtype=np.uint8)
    if arr.shape[0] != cache.layout.num_detectors:
        raise HTTPException(
            status_code=400,
            detail=f"syndrome length {arr.shape[0]} != "
                   f"{cache.layout.num_detectors} expected for {level}",
        )
    from qubit_medic.server.physics import (
        pymatching_predicted_pauli_frame,
        predicted_observable_flip,
    )
    pm_obs = int(cache.matching.decode(arr)[0])
    px, pz = pymatching_predicted_pauli_frame(cache.matching, arr, cache.layout)
    return {
        "level": level,
        "syndrome": syndrome,
        "pymatching_observable_flip": pm_obs,
        "pymatching_x_errors": px,
        "pymatching_z_errors": pz,
        "implied_observable_from_x_errors": predicted_observable_flip(px, cache.layout),
    }


# --------------------------------------------------------------------------- #
# Local entry point                                                            #
# --------------------------------------------------------------------------- #

def _main() -> None:
    import uvicorn

    uvicorn.run(
        "qubit_medic.server.app:app",
        host=os.getenv("QUBIT_MEDIC_HOST", DEFAULT_HOST),
        port=int(os.getenv("QUBIT_MEDIC_PORT", str(DEFAULT_PORT))),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    _main()
