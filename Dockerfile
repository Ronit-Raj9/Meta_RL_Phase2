# Qubit-Medic OpenEnv server container.
#
# This image ships ONLY the env-server code:
#   * stim + pymatching      - quantum simulation + matching baseline
#   * fastapi + uvicorn      - HTTP transport
#   * openenv-core           - canonical OpenEnv contract (/reset, /step,
#                              /state, /health, /schema, /metadata, /mcp,
#                              /docs)
#
# Heavy ML training deps (torch, transformers, trl, unsloth) are
# deliberately NOT installed - they live in requirements-train.txt and
# are installed only by the Colab training notebook. Keeping the Spaces
# image lean avoids the ~10 GB CUDA wheel that would blow the free tier.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Stim and PyMatching ship manylinux wheels - no system C++ deps needed
# beyond libstdc++. We keep build-essential as a safety net for any
# unexpected source-fallback path on the build host.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

# HF Spaces best-practice: run as non-root user with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --user --upgrade pip \
 && pip install --user -r /app/requirements.txt

COPY --chown=user qubit_medic /app/qubit_medic
COPY --chown=user README.md /app/README.md

# Pre-warm Stim/PyMatching caches at build time so the first request
# after `docker run` has near-zero latency (Section 9.1 of the plan).
RUN python -c "from qubit_medic.server.environment import DecoderEnvironment; \
               e = DecoderEnvironment(); \
               e._cache_for('L1_warmup'); \
               e._cache_for('L2_target')"

EXPOSE 7860

ENV LOG_LEVEL=INFO \
    QUBIT_MEDIC_HOST=0.0.0.0 \
    QUBIT_MEDIC_PORT=7860

# Boots the FastAPI app (qubit_medic.server.app) which is built on top
# of openenv.core.create_fastapi_app.
CMD ["python", "-m", "qubit_medic.server.app"]
