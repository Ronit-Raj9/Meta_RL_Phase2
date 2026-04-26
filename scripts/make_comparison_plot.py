"""scripts/make_comparison_plot.py - the "money plot" for Qubit-Medic.

Renders a side-by-side bar chart comparing four conditions on the two
headline metrics emitted by ``scripts.eval`` (and dumped to JSON in
``data/eval/`` or ``data/`` by the training pipeline):

* Random baseline       (uniform-random qubit picks)
* Base Qwen2.5-3B       (un-fine-tuned model; usually format failures)
* SFT-only              (Qwen2.5-3B after supervised fine-tuning)
* SFT + GRPO            (the full Qubit-Medic checkpoint)

Two panels:

* Left: ``logical_correction_rate``    (y-axis 0-1, fraction of shots
        where the predicted Pauli frame yields no logical-Z flip)
* Right: ``pymatching_beat_rate``      (y-axis 0-1, fraction of shots
        where the model corrects but PyMatching does not)

JSON schema expected per condition file (mirrors
``scripts/eval.py::_summary``)::

    {
      "name": str,
      "episodes": int,
      "logical_correction_rate": float,
      "pymatching_beat_rate": float,
      ... (other keys are ignored here)
    }

The script never runs ``scripts.eval`` itself - it just reads JSON.

Usage::

    python scripts/make_comparison_plot.py            # uses defaults
    python scripts/make_comparison_plot.py --eval-dir data/eval
    python scripts/make_comparison_plot.py \
        --random data/eval/random.json \
        --base   data/eval/base_qwen.json \
        --sft    data/eval/sft_only.json \
        --grpo   data/eval/sft_grpo.json \
        --out    figures/before_after_comparison.png
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# --------------------------------------------------------------------------- #
# Plot configuration                                                          #
# --------------------------------------------------------------------------- #

CONDITION_LABELS: tuple[str, ...] = (
    "Random baseline",
    "Base Qwen2.5-3B",
    "SFT-only",
    "SFT + GRPO",
)

# Colour-blind safe-ish palette: greys for the baselines, accent for ours.
CONDITION_COLOURS: tuple[str, ...] = (
    "#9aa0a6",  # random  - light grey
    "#5f6368",  # base    - dark grey
    "#7e57c2",  # sft     - purple
    "#1e88e5",  # sft+grpo - blue (the "after" colour)
)

# Default filenames the script will look for inside --eval-dir if explicit
# per-condition paths are not supplied. Order matches CONDITION_LABELS.
DEFAULT_FILENAMES: tuple[str, ...] = (
    "random.json",
    "base_qwen.json",
    "sft_only.json",
    "sft_grpo.json",
)


# --------------------------------------------------------------------------- #
# Data structures                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Condition:
    """One bar per panel - a single eval JSON read off disk."""

    label: str
    colour: str
    path: Path
    data: Optional[dict]  # None if file missing

    @property
    def lcr(self) -> float:
        if self.data is None:
            return 0.0
        return float(self.data.get("logical_correction_rate", 0.0))

    @property
    def beat(self) -> float:
        if self.data is None:
            return 0.0
        return float(self.data.get("pymatching_beat_rate", 0.0))

    @property
    def episodes(self) -> int:
        if self.data is None:
            return 0
        return int(self.data.get("episodes", 0))


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #


def _load_json(path: Path) -> Optional[dict]:
    """Read a JSON file, returning ``None`` if it does not exist."""
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def _resolve_paths(
    eval_dir: Path,
    explicit: dict[str, Optional[str]],
) -> list[Path]:
    """Resolve a path per condition, preferring explicit overrides."""
    paths: list[Path] = []
    for label, default_name in zip(CONDITION_LABELS, DEFAULT_FILENAMES):
        override = explicit.get(label)
        if override:
            paths.append(Path(override))
        else:
            paths.append(eval_dir / default_name)
    return paths


def load_conditions(
    eval_dir: Path,
    explicit: dict[str, Optional[str]],
) -> list[Condition]:
    """Materialise four ``Condition`` rows in the canonical plot order."""
    paths = _resolve_paths(eval_dir, explicit)
    out: list[Condition] = []
    for label, colour, path in zip(CONDITION_LABELS, CONDITION_COLOURS, paths):
        out.append(
            Condition(
                label=label,
                colour=colour,
                path=path,
                data=_load_json(path),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #


def render_plot(
    conditions: list[Condition],
    out_path: Path,
    title: str,
    dpi: int = 150,
) -> None:
    """Render the two-panel money plot to ``out_path`` at ``dpi``."""
    try:
        import matplotlib.pyplot as plt  # local import: graceful failure path
    except ImportError as exc:  # pragma: no cover - import-time only
        raise SystemExit(
            "matplotlib is required for make_comparison_plot.py. "
            "Install with: pip install matplotlib"
        ) from exc

    labels = [c.label for c in conditions]
    colours = [c.colour for c in conditions]
    lcr_values = [c.lcr for c in conditions]
    beat_values = [c.beat for c in conditions]

    fig, (ax_left, ax_right) = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 5.2), sharey=False
    )

    x = list(range(len(labels)))

    bars_left = ax_left.bar(x, lcr_values, color=colours, edgecolor="black",
                            linewidth=0.6)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=20, ha="right")
    ax_left.set_ylim(0.0, 1.0)
    ax_left.set_ylabel("Logical correction rate (fraction of shots, 0-1)")
    ax_left.set_xlabel("Decoder condition")
    ax_left.set_title("Logical correction rate (per shot)")
    ax_left.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, val in zip(bars_left, lcr_values):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.02, 0.98),
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    bars_right = ax_right.bar(x, beat_values, color=colours, edgecolor="black",
                              linewidth=0.6)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(labels, rotation=20, ha="right")
    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_ylabel("PyMatching beat rate (fraction of shots, 0-1)")
    ax_right.set_xlabel("Decoder condition")
    ax_right.set_title("PyMatching beat rate (model corrects, PM does not)")
    ax_right.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, val in zip(bars_right, beat_values):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.02, 0.98),
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    # One shared legend across both panels.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.6)
        for c in colours
    ]
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02), frameon=False,
    )

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _missing_files_message(conditions: list[Condition]) -> str:
    """Build a helpful error when one or more eval JSONs are absent."""
    missing = [(c.label, c.path) for c in conditions if c.data is None]
    if not missing:
        return ""
    lines = [
        "ERROR: cannot build comparison plot - one or more eval JSON files "
        "were not found.",
        "",
        "Expected files (one per condition):",
    ]
    for label, path in missing:
        lines.append(f"  - {label}: {path}")
    lines.extend([
        "",
        "Generate them with scripts/eval.py, for example:",
        "  python -m scripts.eval --policy random --episodes 1000 \\",
        "      --out data/eval/random.json",
        "  python -m scripts.eval --base-model Qwen/Qwen2.5-3B-Instruct \\",
        "      --adapter '' --episodes 1000 --out data/eval/base_qwen.json",
        "  python -m scripts.eval --adapter checkpoints/sft/best \\",
        "      --episodes 1000 --out data/eval/sft_only.json",
        "  python -m scripts.eval --adapter checkpoints/grpo/best \\",
        "      --episodes 1000 --out data/eval/sft_grpo.json",
        "",
        "Override individual paths with --random / --base / --sft / --grpo.",
    ])
    return "\n".join(lines)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir", type=str, default="data/eval",
        help="Directory holding one JSON per condition "
             "(random.json, base_qwen.json, sft_only.json, sft_grpo.json).",
    )
    parser.add_argument(
        "--random", type=str, default=None,
        help="Override path to the random-baseline eval JSON.",
    )
    parser.add_argument(
        "--base", type=str, default=None,
        help="Override path to the base-Qwen eval JSON.",
    )
    parser.add_argument(
        "--sft", type=str, default=None,
        help="Override path to the SFT-only eval JSON.",
    )
    parser.add_argument(
        "--grpo", type=str, default=None,
        help="Override path to the SFT+GRPO eval JSON.",
    )
    parser.add_argument(
        "--out", type=str, default="figures/before_after_comparison.png",
        help="Where to write the PNG (created at 150 dpi by default).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for the saved PNG.",
    )
    parser.add_argument(
        "--title", type=str,
        default=(
            "Qubit-Medic decoder accuracy: before vs after RLHF training "
            "(distance-3 surface code, p=0.001)"
        ),
        help="Figure suptitle.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = ()) -> int:
    args = parse_args(argv)

    explicit = {
        "Random baseline": args.random,
        "Base Qwen2.5-3B": args.base,
        "SFT-only": args.sft,
        "SFT + GRPO": args.grpo,
    }
    conditions = load_conditions(Path(args.eval_dir), explicit)

    msg = _missing_files_message(conditions)
    if msg:
        print(msg, file=sys.stderr)
        return 1

    render_plot(
        conditions=conditions,
        out_path=Path(args.out),
        title=args.title,
        dpi=args.dpi,
    )
    print(f"Wrote comparison plot to {args.out}")
    for c in conditions:
        print(
            f"  {c.label:>18s}: LCR={c.lcr:.3f}  "
            f"PMbeat={c.beat:.3f}  (n={c.episodes}, src={c.path})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
