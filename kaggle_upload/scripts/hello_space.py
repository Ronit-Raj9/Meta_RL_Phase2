"""scripts/hello_space.py - Day-0 deployment-substrate placeholder app.

This is the *minimal* FastAPI app you push to your Hugging Face Space the
night before the hackathon to prove the deployment substrate works:

    Step 1: Create a public HF Space called e.g. ``qubit-medic-hello``
    Step 2: Push our slim Dockerfile + this file (rename to ``app.py``)
    Step 3: Hit ``/healthz`` from your browser  -> proves networking works
    Step 4: Hit ``/healthz`` from a Colab cell  -> proves Colab can reach it

Once all four pass, replace this file with the real env (the real
:mod:`qubit_medic.server.app` already implements the same ``/healthz``
endpoint, so the Day-0 contract carries forward).

Run locally::

    python -m scripts.hello_space          # listens on :7860
"""
from __future__ import annotations

import os
import sys

import stim
from fastapi import FastAPI


app = FastAPI(
    title="Qubit-Medic - Hello Space",
    description="Day-0 deployment-substrate placeholder. Proves Stim imports "
                "and the HTTP server is reachable. Replace with the real env "
                "once Section 2 of the plan passes local validation.",
    version="0.0.1-placeholder",
)


@app.get("/")
def root() -> dict:
    return {
        "service": "qubit-medic-hello",
        "status": "placeholder live",
        "next": "POST /reset and /step will become available once the real "
                "DecoderEnvironment is pushed.",
    }


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "stim_version": stim.__version__,
        "python_version": sys.version.split()[0],
        "service": "qubit-medic-hello",
    }


def _main() -> None:
    import uvicorn
    uvicorn.run(
        "scripts.hello_space:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )


if __name__ == "__main__":
    _main()
