"""Tests for the OpenEnv-style env contract: reset/step round-trips,
episode bookkeeping, curriculum behaviour, and the FastAPI HTTP surface.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from qubit_medic.client.client import LocalDecoderClient
from qubit_medic.prompts import format_completion
from qubit_medic.server.app import app


def test_local_reset_returns_obs():
    client = LocalDecoderClient()
    obs = client.reset(forced_level="L2_target", seed=1)
    assert obs.distance == 3 and obs.rounds == 3
    assert "X_ERRORS" in obs.prompt
    assert obs.episode_id >= 1


def test_local_reset_step_pairs():
    client = LocalDecoderClient()
    obs = client.reset(forced_level="L1_warmup", seed=2)
    res = client.step(raw_response=format_completion([], []),
                      episode_id=obs.episode_id)
    assert res.done is True
    assert "rewards" in res.info
    for k in ("logical_correction", "syndrome_consistency",
              "hamming_overlap", "format_compliance",
              "pymatching_beat", "total"):
        assert k in res.info["rewards"]


def test_step_unknown_episode_raises():
    # Compliance Section 1 (audit, 2026-04): step() on an unknown episode
    # ID must raise a clean ValueError, not a KeyError-Python-traceback.
    client = LocalDecoderClient()
    with pytest.raises(ValueError):
        client.step(raw_response="X_ERRORS=[] Z_ERRORS=[]", episode_id=10**9)


def test_unparseable_response_still_returns_step_result():
    client = LocalDecoderClient()
    obs = client.reset(forced_level="L2_target", seed=3)
    res = client.step(raw_response="i give up", episode_id=obs.episode_id)
    assert res.info["rewards"]["format_compliance"] == 0.0


def test_distinct_episode_ids():
    client = LocalDecoderClient()
    a = client.reset(forced_level="L2_target")
    b = client.reset(forced_level="L2_target")
    assert a.episode_id != b.episode_id


def test_health_returns_curriculum_stats():
    client = LocalDecoderClient()
    h = client.health()
    assert h["ok"] is True
    assert "curriculum" in h
    for level in ("L1_warmup", "L2_target", "L3_stretch"):
        assert level in h["curriculum"]


def test_http_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        # OpenEnv's /health returns a HealthResponse with status + uptime.
        body = r.json()
        assert "status" in body or "ok" in body


def test_http_healthz():
    with TestClient(app) as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "stim_version" in body
        assert "openenv_version" in body


def test_http_metadata_and_schema():
    with TestClient(app) as c:
        m = c.get("/metadata")
        assert m.status_code == 200
        assert m.json()["name"] == "QubitMedicEnvironment"
        s = c.get("/schema")
        assert s.status_code == 200
        # Schema should expose action + observation models.
        body = s.json()
        assert isinstance(body, dict) and len(body) > 0


def test_http_reset_and_step_openenv_shape():
    """Drive the OpenEnv-canonical /reset + /step endpoints."""
    with TestClient(app) as c:
        # OpenEnv ResetRequest = {seed?, episode_id?}. Forced level rides
        # along as a query-string param honoured by our adapter.
        r = c.post("/reset", json={"seed": 42},
                   params={"forced_level": "L1_warmup"})
        assert r.status_code == 200, r.text
        body = r.json()
        obs = body.get("observation", body)
        assert obs["distance"] == 3
        ep = obs["episode_id"]
        # OpenEnv StepRequest wraps the action in {"action": {...}}.
        s = c.post("/step", json={"action": {
            "raw_response": "X_ERRORS=[] Z_ERRORS=[]",
            "episode_id": ep,
        }})
        assert s.status_code == 200, s.text
        sbody = s.json()
        assert sbody["done"] is True
        assert "reward" in sbody
        sobs = sbody.get("observation", {})
        assert "rewards" in sobs.get("info", {})


def test_http_decode_endpoint():
    """The legacy /decode demo endpoint stays mounted alongside OpenEnv."""
    with TestClient(app) as c:
        # We need to know a syndrome length for a level. Easiest: ask the
        # in-process env directly so we don't depend on /reset's shape.
        local = LocalDecoderClient()
        obs = local.reset(forced_level="L1_warmup", seed=1)
        n_dets = len(obs.syndrome_bits)
        d = c.post("/decode",
                   json={"syndrome": [0] * n_dets, "level": "L1_warmup"})
        assert d.status_code == 200, d.text
        body = d.json()
        assert "pymatching_observable_flip" in body
        # All-zero syndrome -> PM predicts no flip.
        assert body["pymatching_observable_flip"] == 0


def test_http_state_endpoint():
    """OpenEnv's /state returns the QubitMedicState dump."""
    with TestClient(app) as c:
        c.post("/reset", json={"seed": 7})
        r = c.get("/state")
        assert r.status_code == 200, r.text
        body = r.json()
        assert "step_count" in body
        assert body["step_count"] >= 1
