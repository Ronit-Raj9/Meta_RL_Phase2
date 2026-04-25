"""scripts/train_sft.py - SFT warm-up phase (master spec, sections 1-3).

Loads ``Qwen/Qwen2.5-3B-Instruct`` in 4-bit (NF4) via Unsloth, attaches a
LoRA adapter (rank 16, alpha 32, dropout 0.05, on q/k/v/o projections),
and runs a single epoch of supervised fine-tuning on
``data/sft_dataset.jsonl`` (3,000 examples).

Goal: take the base model from ~0% format compliance to >=95% so the GRPO
trainer has a non-zero probability of getting parseable rewards.

Locked hyperparameters (master spec, section 1):
    * batch=4, grad_accum=4 -> effective batch 16
    * lr=2e-4 with 20-step linear warmup -> constant
    * weight_decay=0.01, optimizer=adamw_8bit, mixed precision=bf16
    * max_seq_len=1024, epochs=1, max_steps=200
    * checkpoint every 50, eval every 50, log every 10
    * seed=42

Designed to run on a Colab T4 in <=30 minutes.

Usage::

    pip install -r requirements-train.txt
    python -m scripts.train_sft \
        --dataset data/sft_dataset.jsonl \
        --val-dataset data/sft_validation.jsonl \
        --output checkpoints/sft_warmup \
        --report-to wandb

W&B logging (master spec, section 2)
------------------------------------
* Every 10 steps: TRL's built-in train/loss, learning_rate, grad_norm,
  epoch, global_step.
* Every 50 steps (validation pass on 100 held-out syndromes):

      eval/format_compliance
      eval/logical_correction_rate
      eval/exact_match_pymatching
      eval/hamming_overlap_mean
      eval/output_length_mean
      eval/output_diversity         (10 samples of one prompt @ T=0.7)
      eval/syndrome_consistency

* End-of-train: ``run.summary`` dump of final eval scores; LoRA adapter
  uploaded as a W&B artifact.

Early stopping (master spec, section 3)
---------------------------------------
Training halts as soon as ANY of these is true after a validation pass:

    1. format_compliance >= 0.95 AND logical_correction_rate >= 0.80
       AND output_diversity >= 3                                (success)
    2. global_step >= 200                                       (hard cap)
    3. wall-clock >= 30 minutes                                 (hard cap)
    4. train/loss has NaN or inf                                (failure)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


# --------------------------------------------------------------------------- #
# Pre-flight dataset audit                                                    #
# --------------------------------------------------------------------------- #
# Runs as the FIRST step of main(), before any model/tokenizer/heavy imports.
# Catches dataset regressions (class collapse, format drift, parse breakage,
# size mismatches) in a few seconds, before burning ~30 min of GPU on a run
# that was doomed at row 0.
#
# 9 checks, 3 of them duplicated on the validation split. Any failure raises
# SystemExit(2) so the Colab/Lightning shell pipeline exits with a non-zero
# status and won't proceed to model loading.

_BARE_FORMAT_LITERAL = "X_ERRORS=[] Z_ERRORS=[]"
_FORMAT_ANCHOR_RE = re.compile(r"X_ERRORS=\[[\d,\s]*\]\s*Z_ERRORS=\[[\d,\s]*\]\s*$")
_TAIL_RE = re.compile(r"X_ERRORS=\[([^\]]*)\]\s*Z_ERRORS=\[([^\]]*)\]\s*$")
_LEVEL_P_RE = re.compile(r"Physical error rate:\s*([\d.]+)")
_LEVEL_D_RE = re.compile(r"Code distance:\s*(\d+)")


def _detect_level_from_prompt(prompt: str) -> str:
    """Return ``"L1"``/``"L2"``/``"L3"``/``"unknown"`` for an SFT prompt.

    Used as a fallback for legacy datasets that didn't write a ``level``
    field into each record. We read the L1/L2/L3 ``p`` and ``distance``
    values straight from :mod:`qubit_medic.config` rather than hardcoding
    them, so the audit keeps working when the curriculum is tuned (e.g.
    L1's ``p`` was bumped from 0.0001 -> 0.0005, which broke the old
    hardcoded check and made every L1 row read as ``unknown``).
    """
    m_p = _LEVEL_P_RE.search(prompt)
    m_d = _LEVEL_D_RE.search(prompt)
    if not m_p or not m_d:
        return "unknown"
    p = float(m_p.group(1))
    d = int(m_d.group(1))
    try:
        from qubit_medic.config import level_by_name
        l3 = level_by_name("L3_stretch")
        l2 = level_by_name("L2_target")
        l1 = level_by_name("L1_warmup")
        if d == l3.distance and abs(p - l3.p) < 1e-9:
            return "L3"
        if d == l2.distance and abs(p - l2.p) < 1e-9:
            return "L2"
        if d == l1.distance and abs(p - l1.p) < 1e-9:
            return "L1"
    except Exception:
        pass
    return "unknown"


def _level_label_from_record(rec: dict) -> str:
    """Return ``"L1"``/``"L2"``/``"L3"``/``"unknown"`` for an SFT record.

    Prefers the explicit ``level`` field written by
    ``scripts/generate_sft_data.py`` (e.g. ``"L1_warmup"``). Falls back
    to :func:`_detect_level_from_prompt` for legacy records that lack
    that field.
    """
    raw = rec.get("level")
    if isinstance(raw, str):
        if raw.startswith("L1"):
            return "L1"
        if raw.startswith("L2"):
            return "L2"
        if raw.startswith("L3"):
            return "L3"
    prompt = rec.get("prompt")
    if isinstance(prompt, str):
        return _detect_level_from_prompt(prompt)
    return "unknown"


def _has_nonempty_correction(completion: str) -> bool:
    """True iff the completion's trailing format line predicts at least one
    error (X or Z). Robust to a leading reasoning prefix.
    """
    m = _TAIL_RE.search(completion.rstrip())
    if m is None:
        return False
    return bool(m.group(1).strip()) or bool(m.group(2).strip())


def _audit_file(path: Path) -> dict:
    """Compute raw audit metrics for one JSONL file."""
    if not path.exists():
        return {"error": f"missing file: {path}"}
    rows: list[dict] = []
    parse_failures = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                parse_failures += 1
                continue
            if "prompt" not in rec or "completion" not in rec:
                parse_failures += 1
                continue
            rows.append(rec)
    n = len(rows)
    total_lines = n + parse_failures
    parse_rate = (n / total_lines) if total_lines else 0.0
    nonempty = sum(_has_nonempty_correction(r["completion"]) for r in rows)
    anchor = sum(1 for r in rows if _FORMAT_ANCHOR_RE.search(r["completion"].rstrip()))
    levels = {"L1": 0, "L2": 0, "L3": 0, "unknown": 0}
    for r in rows:
        levels[_level_label_from_record(r)] += 1
    plens = [len(r["prompt"]) for r in rows]
    clens = [len(r["completion"]) for r in rows]
    bare = sum(1 for r in rows if r["completion"].strip() == _BARE_FORMAT_LITERAL)
    return {
        "n": n,
        "parse_failures": parse_failures,
        "parse_rate": parse_rate,
        "nonempty_frac": (nonempty / n) if n else 0.0,
        "anchor_frac": (anchor / n) if n else 0.0,
        "level_pct": {k: ((v / n) if n else 0.0) for k, v in levels.items()},
        "plens": plens,
        "clens": clens,
        "bare_frac": (bare / n) if n else 0.0,
    }


def audit_sft_dataset(
    train_path: str = "data/sft_dataset.jsonl",
    val_path: str = "data/sft_validation.jsonl",
) -> None:
    """Pre-flight audit of the SFT dataset. Halts (SystemExit) on violation.

    Runs 9 checks against ``train_path`` plus 4 parallel checks against
    ``val_path``. Designed to run in seconds on the CPU before any heavy
    ML deps are imported, so a broken dataset never reaches the GPU.

    Locked thresholds:
        Total rows:           train=3000, val=100
        JSON parse rate:      100%
        Non-empty correction: 65-75%
        Format anchor:        100%
        Curriculum L1/L2/L3:  35-45% / 45-55% / 7-15%
        Prompt length:        min>=800, median in [1100,1600], max<=2200
        Completion length:    min>=25, median in [80,250], max<=600
        Bare-format-only:     <=10%
        Validation parallel:  same thresholds applied to val split
    """
    EXPECTED_TRAIN = 3000
    EXPECTED_VAL = 100
    NONEMPTY_LO, NONEMPTY_HI = 0.65, 0.75
    # Tightened to match quota-based per-level generation in
    # scripts/generate_sft_data.py, which produces the 40/50/10 split
    # exactly (no rejection-sampling drift).
    L1_LO, L1_HI = 0.38, 0.42
    L2_LO, L2_HI = 0.48, 0.52
    L3_LO, L3_HI = 0.08, 0.12
    PLEN_MIN, PLEN_MED_LO, PLEN_MED_HI, PLEN_MAX = 800, 1100, 1600, 2200
    CLEN_MIN, CLEN_MED_LO, CLEN_MED_HI, CLEN_MAX = 25, 80, 250, 600
    BARE_MAX = 0.10

    train = _audit_file(Path(train_path))
    if "error" in train:
        print(f"[audit] FATAL: {train['error']}")
        raise SystemExit(2)

    # ------------------------------- train checks ------------------------- #
    checks: list[tuple[str, str, bool]] = []

    checks.append((
        "Total rows",
        f"{train['n']} (expected {EXPECTED_TRAIN})",
        train["n"] == EXPECTED_TRAIN,
    ))
    checks.append((
        "JSON parse rate",
        f"{train['parse_rate'] * 100:.1f}% ({train['parse_failures']} failures)",
        abs(train["parse_rate"] - 1.0) < 1e-9,
    ))
    checks.append((
        "Non-empty correction",
        f"{train['nonempty_frac'] * 100:.1f}% (target 65-75%)",
        NONEMPTY_LO <= train["nonempty_frac"] <= NONEMPTY_HI,
    ))
    checks.append((
        "Format anchor",
        f"{train['anchor_frac'] * 100:.1f}%",
        abs(train["anchor_frac"] - 1.0) < 1e-9,
    ))

    p1 = train["level_pct"]["L1"]
    p2 = train["level_pct"]["L2"]
    p3 = train["level_pct"]["L3"]
    p_unknown = train["level_pct"]["unknown"]
    checks.append((
        "Curriculum L1/L2/L3",
        f"{p1*100:.1f}/{p2*100:.1f}/{p3*100:.1f}% (unknown={p_unknown*100:.1f}%)",
        (L1_LO <= p1 <= L1_HI
         and L2_LO <= p2 <= L2_HI
         and L3_LO <= p3 <= L3_HI),
    ))

    pmin = min(train["plens"]) if train["plens"] else 0
    pmed = int(statistics.median(train["plens"])) if train["plens"] else 0
    pmax = max(train["plens"]) if train["plens"] else 0
    checks.append((
        "Prompt length",
        f"min={pmin} median={pmed} max={pmax}",
        (pmin >= PLEN_MIN
         and PLEN_MED_LO <= pmed <= PLEN_MED_HI
         and pmax <= PLEN_MAX),
    ))

    cmin = min(train["clens"]) if train["clens"] else 0
    cmed = int(statistics.median(train["clens"])) if train["clens"] else 0
    cmax = max(train["clens"]) if train["clens"] else 0
    checks.append((
        "Completion length",
        f"min={cmin} median={cmed} max={cmax}",
        (cmin >= CLEN_MIN
         and CLEN_MED_LO <= cmed <= CLEN_MED_HI
         and cmax <= CLEN_MAX),
    ))

    checks.append((
        "Bare-format completions",
        f"{train['bare_frac'] * 100:.1f}% (max 10%)",
        train["bare_frac"] <= BARE_MAX,
    ))

    # ------------------------------- val parallel ------------------------- #
    val = _audit_file(Path(val_path))
    if "error" in val:
        checks.append(("Validation parallel", val["error"], False))
    else:
        v1 = val["level_pct"]["L1"]
        v2 = val["level_pct"]["L2"]
        v3 = val["level_pct"]["L3"]
        val_pass = (
            val["n"] == EXPECTED_VAL
            and abs(val["parse_rate"] - 1.0) < 1e-9
            and NONEMPTY_LO <= val["nonempty_frac"] <= NONEMPTY_HI
            and abs(val["anchor_frac"] - 1.0) < 1e-9
            and L1_LO <= v1 <= L1_HI
            and L2_LO <= v2 <= L2_HI
            and L3_LO <= v3 <= L3_HI
        )
        val_summary = (
            f"rows={val['n']} parse={val['parse_rate']*100:.0f}% "
            f"nonempty={val['nonempty_frac']*100:.1f}% "
            f"anchor={val['anchor_frac']*100:.0f}% "
            f"L1/L2/L3={v1*100:.1f}/{v2*100:.1f}/{v3*100:.1f}%"
        )
        checks.append(("Validation parallel", val_summary, val_pass))

    # ------------------------------- print banner ------------------------- #
    print()
    print("DATASET AUDIT SUMMARY")
    print("=" * 21)
    label_w = max(len(label) for label, _, _ in checks) + 1
    val_w = max(len(val_str) for _, val_str, _ in checks)
    for label, val_str, passed in checks:
        mark = "✓" if passed else "✗"  # ✓ / ✗
        print(f"{(label + ':').ljust(label_w + 1)} {val_str.ljust(val_w)} [{mark}]")

    all_passed = all(passed for _, _, passed in checks)
    print()
    if all_passed:
        print("ALL CHECKS PASSED — DATASET READY FOR TRAINING")
        print()
        return
    print("AUDIT FAILED — FIX DATASET BEFORE TRAINING")
    print()
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# Validation-record loading                                                   #
# --------------------------------------------------------------------------- #


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _load_train_dataset(path: str, tokenizer):
    """Load the SFT JSONL into a HuggingFace Dataset.

    Master spec (section 4): the chat template is applied via the
    tokenizer (``apply_chat_template``), NOT by manually inserting
    ``<|im_start|>`` markers - that way the same template works across
    Qwen2.5 / Qwen3 / etc. without surprises.
    """
    from datasets import Dataset

    rows = _load_jsonl(path)
    out = []
    for rec in rows:
        messages = [
            {"role": "user", "content": rec["prompt"]},
            {"role": "assistant", "content": rec["completion"]},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            # Defensive fallback if apply_chat_template ever misbehaves.
            text = (
                "<|im_start|>user\n"
                f"{rec['prompt']}\n<|im_end|>\n"
                "<|im_start|>assistant\n"
                f"{rec['completion']}<|im_end|>"
            )
        out.append({
            "prompt": rec["prompt"],
            "completion": rec["completion"],
            "text": text,
        })
    return Dataset.from_list(out)


# --------------------------------------------------------------------------- #
# Per-level physics caches (used by the validation callback)                  #
# --------------------------------------------------------------------------- #


def _build_level_caches(needed_levels: set[str]) -> dict[str, dict]:
    """Pre-build circuit / matching / layout / supports per curriculum level."""
    import pymatching

    from qubit_medic.config import level_by_name
    from qubit_medic.server.physics import (
        build_circuit, build_dem, extract_layout, per_round_x_z_counts,
    )
    from qubit_medic.server.rewards import compute_final_detector_supports

    caches: dict[str, dict] = {}
    for name in needed_levels:
        lvl = level_by_name(name)
        circuit = build_circuit(lvl)
        dem = build_dem(circuit)
        matching = pymatching.Matching.from_detector_error_model(dem)
        layout = extract_layout(circuit)
        n_x, n_z = per_round_x_z_counts(layout)
        supports = compute_final_detector_supports(layout)
        caches[name] = {
            "level": lvl,
            "circuit": circuit,
            "dem": dem,
            "matching": matching,
            "layout": layout,
            "supports": supports,
            "num_x_stab": n_x,
            "num_z_stab": n_z,
        }
    return caches


# --------------------------------------------------------------------------- #
# Validation callback (master spec, section 2 + section 3)                    #
# --------------------------------------------------------------------------- #


def _build_validation_callback(
    *,
    model,
    tokenizer,
    val_records: list[dict],
    eval_every: int,
    eval_schedule: tuple[tuple[int, int, str], ...] | None,
    print_sample_outputs: int,
    output_dir: str,
    max_new_tokens: int,
    diversity_n_samples: int,
    diversity_temperature: float,
    early_stop_format: float,
    early_stop_correction: float,
    early_stop_diversity: int,
    max_wall_seconds: float,
    started_wall: float,
):
    """Returns a ``TrainerCallback`` that:
       * fires at every step in ``eval_schedule`` (or every ``eval_every``
         steps if no schedule is given) with a per-step sample size,
       * logs the spec metrics + new diagnostic metrics to W&B,
       * prints the first ``print_sample_outputs`` raw model outputs to
         stdout AND to ``{output_dir}/eval_samples_step{N}.txt`` so a
         broken parser / generation drift can be diagnosed in seconds,
       * stops training when the success criterion or hard caps fire.

    Metric semantics changed in this revision:
        * Parse failures NO LONGER default to "predict no errors". Failed
          rows contribute logical_correction=0, hamming=0,
          syndrome_consistency=0 to the aggregates. This stops trivial
          syndromes (~95% at p=0.001) from inflating logical_correction_rate
          to 0.98 while format_compliance sits at 0.01.
        * New ``eval/parse_failure_rate`` = 1 - format_compliance, so a
          broken parser is impossible to miss.
        * New ``eval/format_compliance_strict`` reports the share of
          outputs that hit the canonical ``X_ERRORS=[...] Z_ERRORS=[...]``
          form (Reward 4 == 1.0). The looser ``eval/format_compliance``
          reports the share where the model's answer was extractable at all.
    """
    from transformers import TrainerCallback

    from qubit_medic import wandb_utils
    from qubit_medic.prompts import parse_action
    from qubit_medic.server.physics import SyndromeSample
    from qubit_medic.server.rewards import compute_all_rewards

    if not val_records:
        return None

    # Pre-build per-level physics for fast scoring.
    needed = {r["level"] for r in val_records}
    level_caches = _build_level_caches(needed)

    # Pick one stable prompt for the diversity probe (always the same record
    # so the diversity number is comparable across checkpoints).
    diversity_record = val_records[0]
    diversity_messages = [{"role": "user", "content": diversity_record["prompt"]}]

    # Index the schedule: step -> (sample_size, mode). Sample sizes are
    # capped at len(val_records) so a small held-out set still works.
    if eval_schedule:
        schedule = {
            step: (min(size, len(val_records)), mode)
            for step, size, mode in eval_schedule
        }
    else:
        schedule = {}

    sample_dir = Path(output_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    class _ValidationCallback(TrainerCallback):
        # Stamp the most recent eval here so the on_train_end hook can avoid
        # re-running if the eval step coincided with the final step.
        last_eval_step: int = -1

        def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
            now = time.time() - started_wall
            if now >= max_wall_seconds:
                print(f"[sft] wall-clock cap {max_wall_seconds:.0f}s hit at step "
                      f"{state.global_step}; stopping.")
                control.should_training_stop = True
                return

            step = state.global_step
            if step == 0:
                return
            if schedule:
                if step not in schedule:
                    return
            else:
                if step % eval_every != 0:
                    return
            self._run_eval(state, control)

        def on_train_end(self, args, state, control, **kwargs):  # noqa: D401
            if state.global_step != self.last_eval_step:
                self._run_eval(state, control, final=True)

        # ------------------------------------------------------------------ #
        # Core evaluation                                                    #
        # ------------------------------------------------------------------ #
        def _generate_greedy(self, messages: list[dict]) -> tuple[str, int]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            try:
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                gen_ids = out[0][inputs["input_ids"].shape[1]:]
                completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
                return completion, int(gen_ids.shape[0])
            except Exception as exc:
                return f"<gen-error: {exc}>", 0

        def _generate_sampled(self, messages: list[dict]) -> str:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            try:
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=diversity_temperature,
                    top_p=0.95,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                return tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            except Exception as exc:
                return f"<gen-error: {exc}>"

        def _run_eval(self, state, control, *, final: bool = False) -> None:
            self.last_eval_step = state.global_step
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(model)
            except Exception:
                model.eval()  # type: ignore[attr-defined]

            step = state.global_step
            # Resolve sample size + mode for this step.
            if final and step in schedule:
                sample_size, mode = schedule[step]
            elif final:
                sample_size, mode = len(val_records), "full"
            elif step in schedule:
                sample_size, mode = schedule[step]
            else:
                sample_size, mode = len(val_records), "full"

            # Deterministic slice so the same prompts are used across checkpoints.
            records = val_records[:sample_size]
            n = len(records)
            full_eval = (mode == "full")

            n_format = 0          # lenient parse_success
            n_format_strict = 0   # canonical "=" + "[]"
            n_logical = n_exact = 0
            sum_hamming = 0.0
            sum_syndrome = 0.0
            sum_length = 0
            rows: list[dict] = []
            sample_dump_lines: list[str] = [
                f"=== eval samples @ step {step} (mode={mode}, n={n}) ===",
            ]

            for idx, rec in enumerate(records):
                num_data = int(rec["num_data_qubits"])
                messages = [{"role": "user", "content": rec["prompt"]}]
                completion, n_tokens = self._generate_greedy(messages)
                sum_length += n_tokens

                parsed = parse_action(completion, num_data_qubits=num_data)
                fmt_ok = parsed.parse_success
                fmt_strict_ok = bool(parsed.strict_format)
                n_format += int(fmt_ok)
                n_format_strict += int(fmt_strict_ok)

                # Physics-heavy metrics only in "full" mode AND only when
                # the parse actually succeeded. A failed parse means the
                # model didn't produce a usable prediction; we score that
                # as a miss (0) for every downstream metric instead of
                # silently substituting an empty Pauli frame, which would
                # accidentally score correct on the ~95% of trivial
                # syndromes at p=0.001.
                logical_ok = False
                exact_ok = False
                hamming = 0.0
                syndrome = 0.0
                if full_eval and fmt_ok:
                    cache = level_caches[rec["level"]]
                    layout = cache["layout"]
                    supports = cache["supports"]
                    sample = SyndromeSample(
                        syndrome_bits=list(map(int, rec["syndrome_bits"])),
                        actual_observable_flip=int(rec["actual_observable_flip"]),
                        pymatching_observable_pred=int(rec["pymatching_observable_pred"]),
                        pymatching_x_errors=list(map(int, rec["true_x_errors"])),
                        pymatching_z_errors=list(map(int, rec["true_z_errors"])),
                    )
                    breakdown = compute_all_rewards(parsed, sample, layout, supports)
                    logical_ok = breakdown.logical_correction >= 0.5
                    hamming = float(breakdown.hamming_overlap)
                    syndrome = float(breakdown.syndrome_consistency)
                    exact_ok = (
                        parsed.x_errors == sorted(set(rec["true_x_errors"]))
                        and parsed.z_errors == sorted(set(rec["true_z_errors"]))
                    )

                n_logical += int(logical_ok)
                n_exact += int(exact_ok)
                sum_hamming += hamming
                sum_syndrome += syndrome

                if idx < print_sample_outputs:
                    sample_dump_lines.append(
                        f"\n--- sample {idx} (level={rec['level']}, "
                        f"true_x={rec['true_x_errors']}, true_z={rec['true_z_errors']}, "
                        f"fmt_ok={fmt_ok}, fmt_strict={fmt_strict_ok}, "
                        f"n_tokens={n_tokens}) ---\n"
                        f">>> RAW MODEL OUTPUT:\n{completion}\n"
                        f">>> PARSED: x={parsed.x_errors} z={parsed.z_errors}"
                    )

                if idx < 4:  # keep W&B table tiny
                    rows.append({
                        "step": step,
                        "prompt": rec["prompt"][:600],
                        "gold": rec["completion"],
                        "model": completion[:300],
                        "x_pred": ",".join(map(str, parsed.x_errors)),
                        "z_pred": ",".join(map(str, parsed.z_errors)),
                        "format_ok": fmt_ok,
                        "format_strict_ok": fmt_strict_ok,
                        "logical_ok": logical_ok,
                        "exact_match": exact_ok,
                        "hamming_overlap": hamming,
                    })

            # ---------- print + persist raw output samples -------------- #
            sample_blob = "\n".join(sample_dump_lines)
            print(sample_blob)
            try:
                (sample_dir / f"eval_samples_step{step}.txt").write_text(sample_blob)
            except OSError as exc:
                print(f"[sft][eval@{step}] could not persist sample outputs: {exc}")

            # ---------- diversity probe (skip in format_only mode) ------ #
            if full_eval:
                diverse_outputs: list[str] = []
                for _ in range(diversity_n_samples):
                    diverse_outputs.append(self._generate_sampled(diversity_messages))
                output_diversity = len(set(diverse_outputs))
            else:
                output_diversity = 0  # not measured this step

            # ---------- aggregate + log to W&B ------------------------- #
            metrics: dict[str, float | int] = {
                "eval/format_compliance": n_format / max(1, n),
                "eval/format_compliance_strict": n_format_strict / max(1, n),
                "eval/parse_failure_rate": 1.0 - (n_format / max(1, n)),
                "eval/output_length_mean": sum_length / max(1, n),
                "eval/episodes": n,
                "eval/mode_full": int(full_eval),
            }
            if full_eval:
                metrics.update({
                    "eval/logical_correction_rate": n_logical / max(1, n),
                    "eval/exact_match_pymatching": n_exact / max(1, n),
                    "eval/hamming_overlap_mean": sum_hamming / max(1, n),
                    "eval/syndrome_consistency": sum_syndrome / max(1, n),
                    "eval/output_diversity": output_diversity,
                })
            print(f"[sft][eval@{step}] " + ", ".join(
                f"{k.split('/')[-1]}={v:.3f}" if isinstance(v, float) else f"{k.split('/')[-1]}={v}"
                for k, v in metrics.items()
            ))
            wandb_utils.log(metrics, step=step)
            wandb_utils.log_generation_table(
                rows, step=step,
                table_name=("sft/final_validation" if final else "sft/validation"),
                columns=["step", "prompt", "gold", "model", "x_pred", "z_pred",
                         "format_ok", "format_strict_ok", "logical_ok",
                         "exact_match", "hamming_overlap"],
            )

            # ---------- early stop checks ------------------------------ #
            # Only meaningful on full evals: logical_correction_rate and
            # output_diversity are not measured in format_only mode.
            if full_eval:
                success = (
                    metrics["eval/format_compliance"] >= early_stop_format
                    and metrics["eval/logical_correction_rate"] >= early_stop_correction
                    and metrics["eval/output_diversity"] >= early_stop_diversity
                )
                if success and not final:
                    print(f"[sft] success criterion hit at step {state.global_step}: "
                          f"format={metrics['eval/format_compliance']:.3f} >= {early_stop_format}, "
                          f"correction={metrics['eval/logical_correction_rate']:.3f} >= {early_stop_correction}, "
                          f"diversity={int(metrics['eval/output_diversity'])} >= {early_stop_diversity}; "
                          f"stopping.")
                    control.should_training_stop = True
                    wandb_utils.update_summary({"sft/early_stop_reason": "success_criterion"})

            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_training(model)
            except Exception:
                model.train()  # type: ignore[attr-defined]

    return _ValidationCallback()


# --------------------------------------------------------------------------- #
# Loss-divergence guard (failure mode early stop)                             #
# --------------------------------------------------------------------------- #


def _build_loss_guard_callback():
    import math

    from transformers import TrainerCallback

    class _LossGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
            if not logs:
                return
            loss = logs.get("loss")
            if loss is None:
                return
            try:
                lf = float(loss)
            except (TypeError, ValueError):
                return
            if math.isnan(lf) or math.isinf(lf):
                print(f"[sft] loss={loss} is NaN/inf at step {state.global_step}; "
                      f"stopping training.")
                control.should_training_stop = True

    return _LossGuard()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="data/sft_dataset.jsonl")
    parser.add_argument("--val-dataset", type=str,
                        default="data/sft_validation.jsonl",
                        help="held-out validation JSONL (rich records). "
                             "If missing, validation is skipped.")
    parser.add_argument("--output", type=str, default="checkpoints/sft_warmup")
    parser.add_argument("--model", type=str,
                        default=os.getenv("QUBIT_MEDIC_MODEL",
                                          "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="hard cap on training steps (default 200)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=("sft",))
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=None,
                        help="run validation pass every N steps (legacy "
                             "fallback when --no-eval-schedule is set)")
    parser.add_argument("--no-eval-schedule", action="store_true",
                        help="disable the variable-cadence schedule "
                             "(SFT_EVAL_SCHEDULE) and fall back to "
                             "uniform --eval-every spacing")
    parser.add_argument("--print-sample-outputs", type=int,
                        default=None,
                        help="N raw model outputs to print + persist per eval "
                             "(defaults to SFT_PRINT_SAMPLE_OUTPUTS from config)")
    parser.add_argument("--diversity-samples", type=int, default=10,
                        help="N samples for the output_diversity probe")
    parser.add_argument("--diversity-temperature", type=float, default=0.7)
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args(list(argv))

    # Pre-flight dataset audit. Runs in seconds on the CPU before any heavy
    # ML deps are imported, so a broken dataset never reaches the GPU. Halts
    # via SystemExit(2) on any threshold violation.
    audit_sft_dataset(args.dataset, args.val_dataset)

    # Heavy imports are lazy so this module is importable without GPU deps.
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. Run `pip install -r requirements-train.txt`",
              file=sys.stderr)
        return 1
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer

    from qubit_medic import wandb_utils
    from qubit_medic.config import (
        LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGET_MODULES, MODEL_ID,
        PRIMARY_SEED, SFT_BATCH_SIZE, SFT_EARLY_STOP_CORRECTION,
        SFT_EARLY_STOP_DIVERSITY, SFT_EARLY_STOP_FORMAT, SFT_EPOCHS,
        SFT_EVAL_EVERY, SFT_EVAL_SCHEDULE, SFT_GRAD_ACCUM, SFT_LOG_EVERY,
        SFT_LR, SFT_LR_SCHEDULER, SFT_MAX_NEW_TOKENS, SFT_MAX_SEQ_LEN,
        SFT_MAX_STEPS, SFT_MAX_WALL_SECONDS, SFT_OPTIMIZER,
        SFT_PRINT_SAMPLE_OUTPUTS, SFT_SAVE_EVERY, SFT_WARMUP_STEPS,
        SFT_WEIGHT_DECAY,
    )

    epochs = args.epochs if args.epochs is not None else SFT_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else SFT_BATCH_SIZE
    grad_accum = args.grad_accum if args.grad_accum is not None else SFT_GRAD_ACCUM
    lr = args.lr if args.lr is not None else SFT_LR
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else SFT_MAX_SEQ_LEN
    max_steps = args.max_steps if args.max_steps is not None else SFT_MAX_STEPS
    seed = args.seed if args.seed is not None else PRIMARY_SEED
    lora_r = args.lora_r if args.lora_r is not None else LORA_R
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else LORA_ALPHA
    lora_dropout = args.lora_dropout if args.lora_dropout is not None else LORA_DROPOUT
    eval_every = args.eval_every if args.eval_every is not None else SFT_EVAL_EVERY
    print_sample_outputs = (
        args.print_sample_outputs
        if args.print_sample_outputs is not None
        else SFT_PRINT_SAMPLE_OUTPUTS
    )
    model_id = args.model if args.model else MODEL_ID

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- W&B init (no-op if unavailable / disabled) -------------------- #
    report_to = wandb_utils.derive_report_to(args.report_to)
    run_name = args.wandb_run_name or wandb_utils.make_run_name("sft")
    wandb_utils.init_run(
        run_name=run_name,
        job_type="sft",
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        group=args.wandb_group,
        extra_config={
            "cli": {
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "effective_batch": batch_size * grad_accum,
                "lr": lr,
                "lr_scheduler": SFT_LR_SCHEDULER,
                "warmup_steps": SFT_WARMUP_STEPS,
                "weight_decay": SFT_WEIGHT_DECAY,
                "optimizer": SFT_OPTIMIZER,
                "max_seq_len": max_seq_len,
                "max_steps": max_steps,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": list(LORA_TARGET_MODULES),
                "dataset_path": args.dataset,
                "val_dataset_path": args.val_dataset,
                "model": model_id,
                "seed": seed,
                "report_to": report_to,
                "eval_every": eval_every,
                "save_every": SFT_SAVE_EVERY,
                "log_every": SFT_LOG_EVERY,
                "early_stop_format": SFT_EARLY_STOP_FORMAT,
                "early_stop_correction": SFT_EARLY_STOP_CORRECTION,
                "early_stop_diversity": SFT_EARLY_STOP_DIVERSITY,
                "max_wall_seconds": SFT_MAX_WALL_SECONDS,
            },
        },
    )

    # ---- Load model + datasets --------------------------------------- #
    print(f"loading {model_id} via Unsloth (4-bit NF4)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        dtype=None,  # Unsloth auto-selects bf16/fp16
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(LORA_TARGET_MODULES),
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    print(f"loading train dataset from {args.dataset}")
    train_dataset = _load_train_dataset(args.dataset, tokenizer)
    print(f"  {len(train_dataset)} samples; first text len = "
          f"{len(train_dataset[0]['text'])}")

    val_records: list[dict] = []
    val_path = Path(args.val_dataset)
    if val_path.exists():
        val_records = _load_jsonl(args.val_dataset)
        print(f"loaded {len(val_records)} held-out validation records "
              f"from {args.val_dataset}")
    else:
        print(f"WARNING: no validation file at {args.val_dataset}; "
              f"running without eval / early-stop.")

    wandb_utils.log({
        "sft/train_dataset_size": len(train_dataset),
        "sft/val_dataset_size": len(val_records),
        "sft/first_text_len": len(train_dataset[0]["text"]),
    })

    # Dataset preview to W&B (sanity check the chat-template wrapping).
    wandb_utils.log_generation_table(
        [
            {"split": "train", "prompt": train_dataset[i]["prompt"][:600],
             "completion": train_dataset[i]["completion"]}
            for i in range(min(8, len(train_dataset)))
        ],
        step=0,
        table_name="sft/train_preview",
        columns=["split", "prompt", "completion"],
    )

    # ---- TrainingArguments (locked spec) ----------------------------- #
    Path(args.output).mkdir(parents=True, exist_ok=True)
    bf16_supported = (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=epochs,
        max_steps=max_steps,             # hard cap; wins over epochs
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=SFT_WEIGHT_DECAY,
        warmup_steps=SFT_WARMUP_STEPS,
        lr_scheduler_type=SFT_LR_SCHEDULER,
        optim=SFT_OPTIMIZER,
        bf16=bf16_supported,
        fp16=torch.cuda.is_available() and not bf16_supported,
        logging_steps=SFT_LOG_EVERY,
        save_steps=SFT_SAVE_EVERY,
        save_total_limit=4,
        seed=seed,
        report_to=report_to,
        run_name=run_name,
    )

    # ---- Callbacks --------------------------------------------------- #
    started_wall = time.time()
    callbacks = [_build_loss_guard_callback()]
    eval_schedule = None if args.no_eval_schedule else SFT_EVAL_SCHEDULE
    val_cb = _build_validation_callback(
        model=model,
        tokenizer=tokenizer,
        val_records=val_records,
        eval_every=eval_every,
        eval_schedule=eval_schedule,
        print_sample_outputs=print_sample_outputs,
        output_dir=args.output,
        max_new_tokens=SFT_MAX_NEW_TOKENS,
        diversity_n_samples=args.diversity_samples,
        diversity_temperature=args.diversity_temperature,
        early_stop_format=SFT_EARLY_STOP_FORMAT,
        early_stop_correction=SFT_EARLY_STOP_CORRECTION,
        early_stop_diversity=SFT_EARLY_STOP_DIVERSITY,
        max_wall_seconds=SFT_MAX_WALL_SECONDS,
        started_wall=started_wall,
    )
    if val_cb is not None:
        callbacks.append(val_cb)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
        packing=False,
        callbacks=callbacks,
    )

    print(f"training (max_steps={max_steps}, eval_every={eval_every}) ...")
    train_result = trainer.train()
    elapsed = time.time() - started_wall
    metrics = getattr(train_result, "metrics", {}) or {}
    wandb_utils.update_summary({
        "sft/wall_seconds": elapsed,
        **{f"sft/final/{k}": v for k, v in metrics.items()
           if isinstance(v, (int, float))},
    })
    print(f"training finished in {elapsed:.1f}s "
          f"(max_wall_seconds={SFT_MAX_WALL_SECONDS:.0f})")

    print(f"saving adapters to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # ---- Upload adapter as W&B artifact ------------------------------ #
    if not args.no_artifact:
        wandb_utils.log_artifact(
            args.output,
            name=f"sft-adapter-{run_name}",
            artifact_type="model",
            description="SFT-warmed Qwen2.5-3B + LoRA adapter (Qubit-Medic).",
        )

    wandb_utils.finish_run()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
