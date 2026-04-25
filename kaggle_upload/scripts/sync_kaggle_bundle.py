#!/usr/bin/env python3
"""Populate ``kaggle_upload/`` with a clean copy of the repo for Kaggle Datasets.

Run from the repository root::

    python -m scripts.sync_kaggle_bundle

Then::

    kaggle datasets create -p kaggle_upload --dir-mode zip
    # or after edits:
    kaggle datasets version -p kaggle_upload --dir-mode zip -m "sync code"

Environment
-------------
``KAGGLE_DATASET_ID``  Override the ``id`` field in ``dataset-metadata.json``
                       (default: ``ronitraj1/qubit-medic-code``).

The bundle excludes ``.venv``, ``.git``, checkpoints, wandb, and large data
files so the zip stays small enough for Kaggle uploads.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle_upload"
META_SRC = ROOT / "kaggle" / "dataset-metadata.json"

DIRS_TO_COPY = ("qubit_medic", "scripts", "tests", "notebooks")

FILES_TO_COPY = (
    "requirements.txt",
    "requirements-train.txt",
    "openenv.yaml",
    "Dockerfile",
    "Makefile",
    "README.md",
    "app_gradio.py",
    ".env.example",
)

OPTIONAL_DATA_FILES = (
    "data/sft_dataset_sample.jsonl",
)


def _copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ))


def main() -> int:
    if shutil.which("kaggle") is None and os.environ.get("KAGGLE_SKIP_CLI_CHECK") != "1":
        print(
            "NOTE: `kaggle` CLI not found on PATH. Install with:\n"
            "  pip install kaggle\n"
            "and place credentials in ~/.kaggle/kaggle.json\n"
            "Continuing bundle sync anyway...\n",
            file=sys.stderr,
        )

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    missing: list[str] = []
    for name in DIRS_TO_COPY:
        src = ROOT / name
        if not src.is_dir():
            missing.append(name)
            continue
        _copytree(src, OUT / name)

    for name in FILES_TO_COPY:
        src = ROOT / name
        if not src.is_file():
            missing.append(name)
            continue
        shutil.copy2(src, OUT / name)

    data_out = OUT / "data"
    data_out.mkdir(exist_ok=True)
    for rel in OPTIONAL_DATA_FILES:
        src = ROOT / rel
        if src.is_file():
            shutil.copy2(src, OUT / rel)

    # dataset-metadata.json (inject id from env)
    ds_id = os.environ.get("KAGGLE_DATASET_ID", "").strip()
    if not META_SRC.is_file():
        print(f"ERROR: missing {META_SRC}", file=sys.stderr)
        return 1
    with META_SRC.open() as f:
        meta = json.load(f)
    if ds_id:
        meta["id"] = ds_id
    with (OUT / "dataset-metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    # Tiny pointer so Colab users know how to install
    pointer = OUT / "KAGGLE_BUNDLE_README.txt"
    pointer.write_text(
        "This dataset is a code snapshot of Qubit-Medic (META RL Phase 2).\n\n"
        "In a Kaggle notebook or Colab, after adding this dataset as input:\n\n"
        "  !pip install -r /kaggle/input/<this-dataset-folder>/requirements-train.txt\n"
        "  import sys\n"
        "  sys.path.insert(0, '/kaggle/input/<this-dataset-folder>')\n\n"
        "Or download with the Kaggle CLI / API and point PYTHONPATH at the\n"
        "extracted folder.\n",
        encoding="utf-8",
    )

    if missing:
        print("WARNING: skipped missing paths:", ", ".join(missing), file=sys.stderr)

    print(f"Wrote bundle to {OUT} ({sum(1 for _ in OUT.rglob('*'))} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
