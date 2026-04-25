"""Central Weights & Biases integration for Qubit-Medic.

Design goals
------------
1. **Single source of truth** for the W&B project name, default tags, and
   the ``config`` dump that every run logs. Trainers, eval scripts, and
   notebooks all funnel through :func:`init_run` so dashboards always
   line up.

2. **Safe to import without wandb installed.** The package's training
   deps (``wandb``) live in ``requirements-train.txt`` and are absent on
   the lean Spaces image. Anything in this module degrades gracefully
   when the import fails - the rest of the project keeps working.

3. **Disable-by-env-var.** Set ``WANDB_DISABLED=1`` (or
   ``QUBIT_MEDIC_WANDB=0``) and every helper here becomes a no-op,
   regardless of whether the package is installed. CI runs and offline
   testing rely on this.

4. **Rich first-class logging.** We expose dedicated helpers for the
   things this project cares about:

       * Per-reward component scalars (5 lines per step, not just total)
       * Curriculum-level moving averages (one line per level)
       * Parse success / partial / failure rates
       * Generation sample tables (prompt / completion / per-reward)
       * Eval summary tables (one row per (policy, level))

   The trainers and eval script only have to call these helpers; the
   Pythonic context manager handles init, summary, and finish.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import socket
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

from qubit_medic.config import (
    CURRICULUM, GRPO_GEN_PER_PROMPT, GRPO_KL_COEF, GRPO_LR, GRPO_MAX_COMPLETION_LEN,
    GRPO_MAX_PROMPT_LEN, GRPO_STEPS, LORA_ALPHA, LORA_R, LORA_TARGET_MODULES,
    MODEL_ID, PRIMARY_SEED, REWARD_WEIGHTS, SFT_BATCH_SIZE, SFT_EPOCHS,
    SFT_GRAD_ACCUM, SFT_LR, SFT_MAX_SEQ_LEN, WANDB_DEFAULT_TAGS, WANDB_ENTITY,
    WANDB_LOG_GENERATIONS_EVERY, WANDB_PROJECT, WANDB_SAMPLE_GENERATIONS,
)


# --------------------------------------------------------------------------- #
# Lazy import + on/off toggle                                                 #
# --------------------------------------------------------------------------- #


_WANDB_MODULE = None
_RUN: Any = None


def _import_wandb():
    """Import wandb on first use. Returns ``None`` if it isn't installed."""
    global _WANDB_MODULE
    if _WANDB_MODULE is None:
        try:
            import wandb  # type: ignore[import-not-found]
            _WANDB_MODULE = wandb
        except ImportError:
            _WANDB_MODULE = False  # sentinel: "tried, failed"
    return _WANDB_MODULE if _WANDB_MODULE is not False else None


def is_disabled() -> bool:
    """Honours ``WANDB_DISABLED`` and ``QUBIT_MEDIC_WANDB=0``."""
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return True
    if os.environ.get("QUBIT_MEDIC_WANDB", "1").lower() in {"0", "false", "no"}:
        return True
    return False


def is_available() -> bool:
    """``True`` iff wandb is importable AND not disabled by env var."""
    return _import_wandb() is not None and not is_disabled()


def get_run():
    """Return the active W&B run object, or ``None`` if not initialised."""
    return _RUN


# --------------------------------------------------------------------------- #
# Init / finish                                                                #
# --------------------------------------------------------------------------- #


def _system_metadata() -> dict:
    """Static metadata that's helpful on the dashboard but isn't a hyperparam."""
    info = {
        "python_version": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "argv": " ".join(sys.argv),
        "pid": os.getpid(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import stim
        info["stim_version"] = stim.__version__
    except Exception:
        pass
    try:
        import pymatching
        info["pymatching_version"] = pymatching.__version__
    except Exception:
        pass
    try:
        import trl, transformers, peft
        info["trl_version"] = trl.__version__
        info["transformers_version"] = transformers.__version__
        info["peft_version"] = peft.__version__
    except Exception:
        pass
    return info


def _build_default_config(extra: Optional[Mapping[str, Any]] = None) -> dict:
    """The config every run logs - hyperparameters + reward weights + curriculum."""
    cfg: dict[str, Any] = {
        "model_id": MODEL_ID,
        "primary_seed": PRIMARY_SEED,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": list(LORA_TARGET_MODULES),
        "sft": {
            "epochs": SFT_EPOCHS,
            "batch_size": SFT_BATCH_SIZE,
            "grad_accum": SFT_GRAD_ACCUM,
            "lr": SFT_LR,
            "max_seq_len": SFT_MAX_SEQ_LEN,
        },
        "grpo": {
            "steps": GRPO_STEPS,
            "gen_per_prompt": GRPO_GEN_PER_PROMPT,
            "lr": GRPO_LR,
            "kl_coef": GRPO_KL_COEF,
            "max_prompt_len": GRPO_MAX_PROMPT_LEN,
            "max_completion_len": GRPO_MAX_COMPLETION_LEN,
        },
        "reward_weights": dict(REWARD_WEIGHTS),
        "curriculum": [
            {
                "name": lvl.name, "distance": lvl.distance, "rounds": lvl.rounds,
                "p": lvl.p, "promotion_threshold": lvl.promotion_threshold,
            }
            for lvl in CURRICULUM
        ],
        "system": _system_metadata(),
    }
    if extra:
        cfg.update(extra)
    return cfg


def init_run(
    run_name: str,
    job_type: str,
    *,
    tags: Optional[Sequence[str]] = None,
    extra_config: Optional[Mapping[str, Any]] = None,
    notes: Optional[str] = None,
    group: Optional[str] = None,
):
    """Initialise (or no-op) a W&B run.

    Parameters
    ----------
    run_name:
        Human-readable run name (e.g. ``"sft-warmup-2026-04-25"``).
    job_type:
        One of ``"sft"``, ``"grpo"``, ``"eval"``, ``"format-test"``,
        ``"baseline"``. Used to group runs on the dashboard.
    tags:
        Extra tags appended to :data:`qubit_medic.config.WANDB_DEFAULT_TAGS`.
    extra_config:
        Hyperparameters specific to this run (e.g. SFT epochs override).
    notes:
        Free-text notes shown on the dashboard.
    group:
        Optional W&B group (used to bundle SFT + GRPO + eval runs of the
        same experiment).

    Returns
    -------
    The wandb Run object, or ``None`` if W&B is unavailable / disabled.
    """
    global _RUN
    wandb = _import_wandb()
    if wandb is None or is_disabled():
        if wandb is None:
            print("[wandb] not installed; skipping init "
                  "(install with `pip install wandb` to enable logging)",
                  file=sys.stderr)
        else:
            print("[wandb] disabled by env var; skipping init", file=sys.stderr)
        _RUN = None
        return None

    all_tags = list(WANDB_DEFAULT_TAGS) + list(tags or [])
    cfg = _build_default_config(extra=extra_config)
    cfg["job_type"] = job_type

    _RUN = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        job_type=job_type,
        tags=all_tags,
        config=cfg,
        notes=notes,
        group=group,
        reinit=True,
    )
    print(f"[wandb] run live at {_RUN.url}", file=sys.stderr)
    return _RUN


def finish_run() -> None:
    """Cleanly close the current W&B run, if any."""
    global _RUN
    wandb = _import_wandb()
    if wandb is None or _RUN is None:
        _RUN = None
        return
    try:
        wandb.finish()
    finally:
        _RUN = None


@contextlib.contextmanager
def run_context(run_name: str, job_type: str, **kwargs):
    """Context-manager wrapper around :func:`init_run` / :func:`finish_run`."""
    init_run(run_name, job_type, **kwargs)
    try:
        yield get_run()
    finally:
        finish_run()


# --------------------------------------------------------------------------- #
# Generic logging helpers                                                     #
# --------------------------------------------------------------------------- #


def log(metrics: Mapping[str, Any], *, step: Optional[int] = None,
        commit: bool = True) -> None:
    """No-op-safe ``wandb.log`` wrapper.

    We store training-step alignment as an explicit scalar
    ``train/global_step`` instead of passing W&B's reserved ``step=`` value.
    HuggingFace/TRL may advance W&B's internal step before our callback logs,
    which otherwise produces "Tried to log to step N that is less than the
    current step N+1" and drops eval metrics.
    """
    wandb = _import_wandb()
    if wandb is None or _RUN is None:
        return
    try:
        payload = dict(metrics)
        if step is not None and "train/global_step" not in payload:
            payload["train/global_step"] = int(step)
        wandb.log(payload, commit=commit)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[wandb] log failed: {exc}", file=sys.stderr)


def update_summary(values: Mapping[str, Any]) -> None:
    """Write to ``run.summary`` (the run's headline numbers)."""
    if _RUN is None:
        return
    try:
        for k, v in values.items():
            _RUN.summary[k] = v
    except Exception as exc:  # pragma: no cover
        print(f"[wandb] summary update failed: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Project-specific helpers                                                    #
# --------------------------------------------------------------------------- #


_REWARD_KEYS = (
    "logical_correction",
    "syndrome_consistency",
    "hamming_overlap",
    "format_compliance",
    "pymatching_beat",
    "total",
)


def log_reward_breakdown(
    breakdowns: Sequence[Mapping[str, float]],
    *,
    step: Optional[int] = None,
    prefix: str = "rl",
) -> None:
    """Log mean / min / max for each of the five reward components.

    ``breakdowns`` is a list of dicts, one per generation in the most-recent
    GRPO step (length = ``GRPO_GEN_PER_PROMPT * batch``). We log mean and
    standard deviation so the dashboard has both signal and noise.
    """
    if not breakdowns or _RUN is None:
        return
    out: dict[str, float] = {}
    for k in _REWARD_KEYS:
        vals = [float(b.get(k, 0.0)) for b in breakdowns]
        n = max(1, len(vals))
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        out[f"{prefix}/reward/{k}_mean"] = mean
        out[f"{prefix}/reward/{k}_std"] = var ** 0.5
        out[f"{prefix}/reward/{k}_max"] = max(vals)
        out[f"{prefix}/reward/{k}_min"] = min(vals)
    log(out, step=step)


def log_parse_stats(
    parse_results: Iterable,  # iterable of qubit_medic.prompts.ParseResult
    *,
    step: Optional[int] = None,
    prefix: str = "rl",
) -> None:
    """Log parse-success / partial / failure rates for the most-recent batch."""
    if _RUN is None:
        return
    parse_results = list(parse_results)
    n = max(1, len(parse_results))
    success = sum(1 for r in parse_results if getattr(r, "parse_success", False))
    partial = sum(1 for r in parse_results
                  if not getattr(r, "parse_success", False)
                  and getattr(r, "parse_partial", False))
    log({
        f"{prefix}/parse/success_rate": success / n,
        f"{prefix}/parse/partial_rate": partial / n,
        f"{prefix}/parse/failure_rate": (n - success - partial) / n,
        f"{prefix}/parse/sample_count": n,
    }, step=step)


def log_curriculum(
    curriculum_stats: Mapping[str, Mapping[str, float]],
    *,
    step: Optional[int] = None,
    prefix: str = "rl",
) -> None:
    """Log the per-level moving-average from the env health endpoint.

    ``curriculum_stats`` is what
    :meth:`qubit_medic.server.curriculum.CurriculumScheduler.snapshot`
    returns; one inner dict per level with keys ``moving_mean`` / ``samples``.
    """
    if _RUN is None or not curriculum_stats:
        return
    out: dict[str, float] = {}
    for level_name, stats in curriculum_stats.items():
        out[f"{prefix}/curriculum/{level_name}_mean"] = float(stats.get("moving_mean", 0.0))
        out[f"{prefix}/curriculum/{level_name}_samples"] = float(stats.get("samples", 0.0))
    log(out, step=step)


def log_generation_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    step: Optional[int],
    table_name: str = "rl/generations",
    columns: Optional[Sequence[str]] = None,
) -> None:
    """Log a wandb.Table of (prompt, completion, reward, ...) rows.

    Each row is a flat dict; the column set is the union of all keys (or
    the explicit ``columns`` arg). Used to surface qualitative samples
    in addition to the scalar curves.
    """
    wandb = _import_wandb()
    if wandb is None or _RUN is None or not rows:
        return
    cols = list(columns) if columns is not None else sorted(
        {k for row in rows for k in row.keys()}
    )
    try:
        table = wandb.Table(columns=cols)
        for row in rows:
            table.add_data(*[row.get(c, None) for c in cols])
        log({table_name: table}, step=step)
    except Exception as exc:  # pragma: no cover
        print(f"[wandb] table log failed: {exc}", file=sys.stderr)


def log_eval_summary(
    summary: Mapping[str, Any],
    *,
    step: Optional[int] = None,
    prefix: str = "eval",
) -> None:
    """Log the dict produced by ``scripts/eval._summary`` as scalars."""
    if _RUN is None:
        return
    out: dict[str, Any] = {}
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            out[f"{prefix}/{k}"] = v
    log(out, step=step)
    update_summary({f"{prefix}/{k}": v for k, v in summary.items()
                    if isinstance(v, (int, float, str, bool))})


def log_artifact(
    path: str, *, name: str, artifact_type: str,
    description: Optional[str] = None,
) -> None:
    """Save a file or directory as a W&B artifact."""
    wandb = _import_wandb()
    if wandb is None or _RUN is None:
        return
    try:
        art = wandb.Artifact(name, type=artifact_type, description=description)
        if os.path.isdir(path):
            art.add_dir(path)
        else:
            art.add_file(path)
        _RUN.log_artifact(art)
    except Exception as exc:  # pragma: no cover
        print(f"[wandb] artifact log failed: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# CLI integration helpers                                                     #
# --------------------------------------------------------------------------- #


def derive_report_to(report_to: str) -> str:
    """Translate the user-facing ``--report-to`` flag.

    If the user passes ``"wandb"`` but wandb is unavailable, fall back to
    ``"none"`` rather than crashing the trainer. Lets the same script run
    on Colab (with wandb) and CI (without).
    """
    if report_to == "wandb" and not is_available():
        print("[wandb] requested via --report-to but unavailable; falling back to 'none'",
              file=sys.stderr)
        return "none"
    return report_to


def make_run_name(prefix: str, suffix: Optional[str] = None) -> str:
    """Build a default run name like ``sft-warmup-20260425-2105``."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bits = [prefix, stamp]
    if suffix:
        bits.append(suffix)
    return "-".join(bits)


__all__ = [
    "derive_report_to",
    "finish_run",
    "get_run",
    "init_run",
    "is_available",
    "is_disabled",
    "log",
    "log_artifact",
    "log_curriculum",
    "log_eval_summary",
    "log_generation_table",
    "log_parse_stats",
    "log_reward_breakdown",
    "make_run_name",
    "run_context",
    "update_summary",
    "WANDB_LOG_GENERATIONS_EVERY",
    "WANDB_SAMPLE_GENERATIONS",
]
