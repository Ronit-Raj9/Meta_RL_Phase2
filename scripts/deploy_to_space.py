"""scripts/deploy_to_space.py - upload this repo to a Hugging Face Space.

Two operating modes:

* ``--placeholder``: pushes ONLY the Dockerfile + scripts/hello_space.py
  (renamed to app.py at the destination). Use this for the Day-0
  deployment-substrate test (Section 11 of the plan).

* default: pushes the full env - ``qubit_medic/``, ``app_gradio.py``,
  ``Dockerfile``, ``requirements.txt``, ``README.md``, and the small
  ``figures/``/``data/`` artefacts that the Gradio demo references. The
  README's YAML header tells Spaces to use ``sdk: docker`` and ``app_port:
  7860``; the Dockerfile starts the FastAPI server by default.

Examples::

    # Day-0 placeholder
    huggingface-cli login   # one-time
    python -m scripts.deploy_to_space --repo your-username/qubit-medic-hello \\
        --placeholder

    # Real env
    python -m scripts.deploy_to_space --repo your-username/qubit-medic
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable


def _build_placeholder_payload(target_dir: Path) -> None:
    """Create a minimal Space payload at ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    # Tiny Dockerfile - just install Stim + FastAPI, run hello.
    (target_dir / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "ENV PYTHONUNBUFFERED=1\n"
        "WORKDIR /app\n"
        "RUN pip install --no-cache-dir stim>=1.13 fastapi>=0.110 'uvicorn[standard]>=0.27'\n"
        "COPY app.py /app/app.py\n"
        "EXPOSE 7860\n"
        "CMD [\"python\", \"-m\", \"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"7860\"]\n"
    )
    repo_root = Path(__file__).resolve().parent.parent
    shutil.copy2(repo_root / "scripts" / "hello_space.py", target_dir / "app.py")
    # Minimal Spaces README so the Space page renders something sensible.
    (target_dir / "README.md").write_text(
        "---\n"
        "title: Qubit-Medic Hello\n"
        "emoji: 🩺\n"
        "colorFrom: indigo\n"
        "colorTo: pink\n"
        "sdk: docker\n"
        "app_port: 7860\n"
        "pinned: false\n"
        "license: mit\n"
        "short_description: Day-0 deployment substrate for Qubit-Medic.\n"
        "---\n\n"
        "# Qubit-Medic - Hello\n\n"
        "Placeholder Space. Hit `GET /healthz` to verify it's alive. "
        "Will be replaced by the real Qubit-Medic OpenEnv shortly.\n"
    )


def _full_payload_paths() -> list[tuple[Path, str]]:
    """Return [(local_path, repo_path)] for the full env payload."""
    repo_root = Path(__file__).resolve().parent.parent
    paths: list[tuple[Path, str]] = []

    def _add(local: str, repo: str | None = None):
        p = repo_root / local
        if p.exists():
            paths.append((p, repo or local))

    _add("Dockerfile")
    _add("requirements.txt")
    _add("README.md")
    _add("openenv.yaml")
    _add("app_gradio.py")
    _add("LICENSE")
    _add("qubit_medic")  # whole package
    _add("scripts/hello_space.py")  # keep available as a healthcheck
    # Visuals referenced by the README.
    for f in ("grid_hero.png", "grid_animation.gif", "total_reward.png",
              "logical_correction.png", "pymatching_beat_rate.png",
              "FIGURES.md"):
        _add(f"figures/{f}")
    return paths


def _push(repo_id: str, payload_paths: Iterable[tuple[Path, str]] | None,
          payload_dir: Path | None, *, private: bool, token: str | None) -> int:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    print(f"creating/refreshing Space {repo_id} ...")
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                exist_ok=True, private=private, token=token)

    if payload_dir is not None:
        print(f"uploading folder {payload_dir} -> {repo_id}")
        api.upload_folder(
            repo_id=repo_id, repo_type="space", folder_path=str(payload_dir),
            commit_message="deploy via scripts/deploy_to_space.py",
        )
    else:
        for local, dst in payload_paths or []:
            print(f"  upload  {local}  ->  {dst}")
            if local.is_dir():
                api.upload_folder(
                    repo_id=repo_id, repo_type="space",
                    folder_path=str(local), path_in_repo=dst,
                    commit_message="deploy via scripts/deploy_to_space.py",
                )
            else:
                api.upload_file(
                    repo_id=repo_id, repo_type="space",
                    path_or_fileobj=str(local), path_in_repo=dst,
                    commit_message="deploy via scripts/deploy_to_space.py",
                )
    print(f"done. open https://huggingface.co/spaces/{repo_id}")
    return 0


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True,
                        help="HF Space repo id, e.g. 'your-username/qubit-medic'")
    parser.add_argument("--placeholder", action="store_true",
                        help="Push only the Day-0 hello-world placeholder.")
    parser.add_argument("--private", action="store_true",
                        help="Create the Space as private.")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"),
                        help="HF write token. Defaults to $HF_TOKEN.")
    args = parser.parse_args(list(argv))

    if args.placeholder:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp) / "space"
            _build_placeholder_payload(tmpdir)
            return _push(args.repo, payload_paths=None, payload_dir=tmpdir,
                         private=args.private, token=args.token)
    else:
        return _push(args.repo, payload_paths=_full_payload_paths(),
                     payload_dir=None,
                     private=args.private, token=args.token)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
