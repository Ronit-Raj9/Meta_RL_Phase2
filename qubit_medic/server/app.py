"""Qubit-Medic FastAPI server.

Built on **openenv-core** ``create_fastapi_app`` so the canonical OpenEnv
routes (``/reset``, ``/step``, ``/state``, ``/health``, ``/schema``,
``/metadata``, ``/mcp``) are wired automatically by the framework.

We add a few extras on top:

* ``GET  /healthz``   - the Day-0 deployment-substrate liveness probe
  (returns Stim/PyMatching/openenv versions). Used by the recurring
  4-hour HF Spaces wakeup ping.
* ``POST /decode``    - PyMatching baseline demo: takes a hand-crafted
  syndrome and returns the matching-decoder's prediction. Useful for
  the Gradio playground.

Run with ``python -m qubit_medic.server.app`` or
``uvicorn qubit_medic.server.app:app --host 0.0.0.0 --port 7860``.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from fastapi import Body, HTTPException
from fastapi.responses import HTMLResponse
from openenv.core import create_fastapi_app

from qubit_medic.config import DEFAULT_HOST, DEFAULT_PORT
from qubit_medic.server.environment import DecoderEnvironment
from qubit_medic.server.openenv_adapter import (
    QubitMedicAction,
    QubitMedicEnvironment,
    QubitMedicObservation,
)


logger = logging.getLogger("qubit_medic.server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# --------------------------------------------------------------------------- #
# Build the OpenEnv-compliant FastAPI app                                     #
# --------------------------------------------------------------------------- #

app = create_fastapi_app(
    env=QubitMedicEnvironment,
    action_cls=QubitMedicAction,
    observation_cls=QubitMedicObservation,
)
app.title = "Qubit-Medic OpenEnv"
app.version = os.getenv("QUBIT_MEDIC_VERSION", "1.0.0")
app.description = (
    "RL training environment for LLM-based quantum error-correction "
    "decoders. Built on Stim + PyMatching with five independent verifiable "
    "rewards (logical correction, syndrome consistency, Hamming overlap, "
    "format compliance, PyMatching beat-rate). Wraps "
    "qubit_medic.server.environment.DecoderEnvironment in "
    "openenv.core.Environment - see /metadata, /schema, /docs."
)


# --------------------------------------------------------------------------- #
# Day-0 + demo extras                                                          #
# --------------------------------------------------------------------------- #

# Lazy-built *legacy* DecoderEnvironment for /decode demos. The OpenEnv
# adapter has its own per-instance DecoderEnvironment; we keep this one
# around for the simple synchronous `/decode` baseline endpoint.
_legacy_env: Optional[DecoderEnvironment] = None


def _get_legacy_env() -> DecoderEnvironment:
    global _legacy_env
    if _legacy_env is None:
        _legacy_env = DecoderEnvironment()
        _legacy_env._cache_for("L1_warmup")  # noqa: SLF001
        _legacy_env._cache_for("L2_target")  # noqa: SLF001
    return _legacy_env


# --------------------------------------------------------------------------- #
# Compliance Section 2 (audit 2026-04): POST /state and POST /close.          #
# --------------------------------------------------------------------------- #
# OpenEnv's create_fastapi_app already mounts GET /state and (via the
# canonical contract) does not expose /close at all. The participant-guide
# audit explicitly requires POST /state and POST /close, so we surface
# both as additional routes that delegate to the legacy DecoderEnvironment
# singleton (the same one /decode already uses). The OpenEnv-canonical
# GET /state route is preserved untouched.
# --------------------------------------------------------------------------- #


@app.post("/state")
def post_state() -> dict:
    """POST mirror of the OpenEnv GET /state route.

    Returns a JSON-serialisable snapshot of env state. Uses the inner
    :meth:`DecoderEnvironment.state` (added in Section 1 compliance work)
    which excludes ground-truth fields by construction.
    """
    return _get_legacy_env().state()


@app.post("/close")
def post_close() -> dict:
    """POST /close: drop in-flight episodes on the legacy env singleton.

    The singleton is rebuilt lazily on the next /reset, so calling /close
    repeatedly is idempotent. Returns a small JSON dict so the caller can
    confirm the request landed.
    """
    _get_legacy_env().close()
    return {"ok": True, "closed": True}


@app.get("/healthz")
def healthz() -> dict:
    """Lightweight liveness probe (Day-0 deployment-substrate test).

    Returns the versions of Stim, PyMatching, and openenv so curl-ing
    this in a browser or from Colab proves both that networking works
    AND that the heavy quantum + RL deps actually loaded. Used by the
    recurring 4-hour HF Spaces wakeup ping.
    """
    import stim
    try:
        import pymatching as _pm
        pm_v = getattr(_pm, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - defensive
        pm_v = f"import-error: {exc}"
    try:
        import openenv as _oe
        oe_v = getattr(_oe, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - defensive
        oe_v = f"import-error: {exc}"
    return {
        "ok": True,
        "service": "qubit-medic",
        "version": app.version,
        "stim_version": stim.__version__,
        "pymatching_version": pm_v,
        "openenv_version": oe_v,
        "python_version": sys.version.split()[0],
    }


@app.post("/decode")
def decode(
    syndrome: list[int] = Body(..., embed=True),
    level: str = Body("L2_target", embed=True),
) -> dict:
    """Decode an arbitrary syndrome with PyMatching (baseline) and return
    its predicted Pauli frame and observable flip.

    Intended for the live Gradio demo: a notebook or web page can POST a
    hand-crafted syndrome here and visualise the matching-decoder result.
    """
    import numpy as np

    env = _get_legacy_env()
    cache = env._cache_for(level)  # noqa: SLF001
    arr = np.asarray(syndrome, dtype=np.uint8)
    if arr.shape[0] != cache.layout.num_detectors:
        raise HTTPException(
            status_code=400,
            detail=f"syndrome length {arr.shape[0]} != "
                   f"{cache.layout.num_detectors} expected for {level}",
        )
    from qubit_medic.server.physics import (
        predicted_observable_flip,
        pymatching_predicted_pauli_frame,
    )
    pm_obs = int(cache.matching.decode(arr)[0])
    px, pz = pymatching_predicted_pauli_frame(cache.matching, arr, cache.layout)
    return {
        "level": level,
        "syndrome": syndrome,
        "pymatching_observable_flip": pm_obs,
        "pymatching_x_errors": px,
        "pymatching_z_errors": pz,
        "implied_observable_from_x_errors": predicted_observable_flip(
            px, cache.layout
        ),
    }


# --------------------------------------------------------------------------- #
# Root landing page                                                            #
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    """HTML landing page shown in the HF Spaces App tab."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Qubit-Medic — OpenEnv server</title>
  <style>
    body { font-family: sans-serif; max-width: 680px; margin: 60px auto; color: #e0e0e0; background: #0d1117; }
    h1   { font-size: 1.5rem; }
    a    { color: #58a6ff; }
    code { background: #161b22; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
    ul   { line-height: 2; }
  </style>
</head>
<body>
  <h1>Qubit-Medic &mdash; OpenEnv server</h1>
  <p>
    This Space exposes a <strong>JSON API</strong> for the quantum
    error-decoding environment (Stim + PyMatching, OpenEnv contract).
    Use the links below to interact with it.
  </p>
  <ul>
    <li><a href="/docs">Interactive API docs (Swagger)</a></li>
    <li><a href="/redoc">ReDoc</a></li>
    <li><a href="/healthz">Liveness <code>GET /healthz</code></a> &mdash; versions probe</li>
    <li><a href="/metadata">OpenEnv <code>GET /metadata</code></a></li>
  </ul>
  <p>
    Typical flow: <code>POST /reset</code> then <code>POST /step</code>
    with the model&rsquo;s text action &mdash; see the schema in <a href="/docs">/docs</a>.
  </p>
</body>
</html>"""
    return HTMLResponse(content=html)


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
