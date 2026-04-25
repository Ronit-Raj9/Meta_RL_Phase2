"""Tests for the W&B integration shim.

These run regardless of whether the ``wandb`` Python package is actually
installed - the whole point of ``qubit_medic.wandb_utils`` is that it
no-ops cleanly when wandb is missing or disabled. We use the
``WANDB_DISABLED=1`` env var to force the no-op path even on machines
that have wandb installed (e.g. Colab).

We also exercise the eval-script CLI end-to-end with ``--report-to wandb``
under that disabled mode to make sure it doesn't crash.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _force_wandb_disabled(monkeypatch):
    """Hard-disable wandb for every test in this file."""
    monkeypatch.setenv("WANDB_DISABLED", "1")
    # Reload the module so the new env var takes effect.
    if "qubit_medic.wandb_utils" in sys.modules:
        importlib.reload(sys.modules["qubit_medic.wandb_utils"])
    yield


def _w():
    import qubit_medic.wandb_utils as w
    return w


def test_disabled_when_env_var_set():
    w = _w()
    assert w.is_disabled() is True
    assert w.is_available() is False


def test_init_run_returns_none_when_disabled():
    w = _w()
    run = w.init_run("test-run", job_type="test")
    assert run is None
    assert w.get_run() is None


def test_log_helpers_no_op_without_run():
    w = _w()
    # All of these must be safe to call when there's no active run.
    w.log({"foo": 1.0})
    w.update_summary({"bar": 2})
    w.log_reward_breakdown([{
        "logical_correction": 1.0, "syndrome_consistency": 0.5,
        "hamming_overlap": 0.3, "format_compliance": 1.0,
        "pymatching_beat": 0.0, "total": 0.6,
    }])
    w.log_curriculum({"L1_warmup": {"moving_mean": 0.7, "samples": 100}})
    w.log_generation_table([{"prompt": "p", "completion": "c"}], step=0)
    w.log_eval_summary({"logical_correction_rate": 0.9})
    w.log_artifact("/nonexistent/path", name="foo", artifact_type="model")
    w.finish_run()


def test_run_context_no_op():
    w = _w()
    with w.run_context("test-ctx", job_type="test") as run:
        assert run is None
        w.log({"x": 1})


def test_derive_report_to_falls_back_to_none():
    w = _w()
    assert w.derive_report_to("wandb") == "none"
    assert w.derive_report_to("none") == "none"
    assert w.derive_report_to("tensorboard") == "tensorboard"


def test_make_run_name_format():
    w = _w()
    name = w.make_run_name("sft")
    assert name.startswith("sft-")
    parts = name.split("-")
    # "sft-YYYYmmdd-HHMMSS"
    assert len(parts) == 3
    assert len(parts[1]) == 8
    assert len(parts[2]) == 6
    name2 = w.make_run_name("sft", suffix="exp1")
    assert name2.endswith("-exp1")


def test_eval_script_runs_with_wandb_disabled(tmp_path, monkeypatch):
    """End-to-end: `eval.py --report-to wandb` should not crash when
    wandb is disabled (the whole point of ``derive_report_to``)."""
    from scripts.eval import main as eval_main
    out_path = tmp_path / "eval.json"
    rc = eval_main([
        "--policy", "zeros",
        "--episodes", "10",
        "--out", str(out_path),
        "--report-to", "wandb",
        "--wandb-group", "ci-test",
    ])
    assert rc == 0
    import json
    with open(out_path) as f:
        body = json.load(f)
    assert "logical_correction_rate" in body
    assert body["episodes"] == 10


def test_format_test_runs_with_wandb_disabled(tmp_path):
    """Same end-to-end smoke check for the format-test script."""
    from scripts.format_test import main as ft_main
    out_path = tmp_path / "ft.json"
    rc = ft_main([
        "--backend", "dummy",
        "--syndromes", "3",
        "--samples-per", "2",
        "--out", str(out_path),
        "--report-to", "wandb",
    ])
    assert rc == 0
    import json
    with open(out_path) as f:
        body = json.load(f)
    assert body["n"] == 6
    assert "rate" in body
