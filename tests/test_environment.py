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
    client = LocalDecoderClient()
    with pytest.raises(KeyError):
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
        body = r.json()
        assert body["ok"] is True


def test_http_reset_and_step():
    with TestClient(app) as c:
        r = c.post("/reset", json={"forced_level": "L1_warmup", "seed": 42})
        assert r.status_code == 200
        obs = r.json()
        assert obs["distance"] == 3
        ep = obs["episode_id"]
        s = c.post("/step", json={"raw_response": "X_ERRORS=[] Z_ERRORS=[]",
                                  "episode_id": ep})
        assert s.status_code == 200
        body = s.json()
        assert body["done"] is True


def test_http_decode_endpoint():
    with TestClient(app) as c:
        r = c.post("/reset", json={"forced_level": "L1_warmup", "seed": 1})
        n_dets = len(r.json()["syndrome_bits"])
        d = c.post("/decode",
                   json={"syndrome": [0] * n_dets, "level": "L1_warmup"})
        assert d.status_code == 200
        body = d.json()
        assert "pymatching_observable_flip" in body
        # All-zero syndrome -> PM predicts no flip.
        assert body["pymatching_observable_flip"] == 0
