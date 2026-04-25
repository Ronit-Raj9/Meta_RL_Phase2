# Qubit-Medic OpenEnv server container.
#
# This image carries ONLY the inference / env-server code (Stim, PyMatching,
# FastAPI). Heavy ML training deps (torch, transformers, trl, unsloth) are
# deliberately NOT installed - they live in `requirements-train.txt` and
# are installed by Colab notebooks. Keeping the Spaces image small avoids
# the ~10 GB CUDA wheel that would otherwise blow the Spaces free tier.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Stim and PyMatching ship pre-built wheels for manylinux; no system C++ deps
# are needed beyond libstdc++. We still install build-essential for safety
# in case pip falls back to source on an unexpected arch.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt \
 && pip install gradio>=4.0

COPY qubit_medic /app/qubit_medic
COPY app_gradio.py /app/app_gradio.py
COPY README.md /app/README.md

# Pre-warm Stim/PyMatching caches at build time so the first request after
# `docker run` has near-zero latency (Section 9.1 of the plan).
RUN python -c "from qubit_medic.server.environment import DecoderEnvironment; \
               e = DecoderEnvironment(); \
               e._cache_for('L1_warmup'); \
               e._cache_for('L2_target')"

EXPOSE 7860

# Default command: run the FastAPI env server. Override to `python app_gradio.py`
# to launch the Gradio demo instead.
ENV LOG_LEVEL=INFO \
    QUBIT_MEDIC_HOST=0.0.0.0 \
    QUBIT_MEDIC_PORT=7860

CMD ["python", "-m", "qubit_medic.server.app"]
