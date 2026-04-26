"""Build extra README figures from committed JSON in data/.

Writes:
  figures/eval_metrics_bars.png   — key scalars from data/eval_grpo.json
  figures/sft_curriculum_mix.png  — SFT row counts per level from
                                    data/sft_dataset_analysis.json

Run from repo root::

    python -m scripts.plot_data_figures
    python -m scripts.plot_data_figures --out-dir figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _plot_eval_metrics(data: dict, out: Path) -> bool:
    import matplotlib.pyplot as plt

    keys = [
        ("mean_total_reward", "Mean total reward"),
        ("logical_correction_rate", "Logical correction"),
        ("pymatching_beat_rate", "PyMatching beat"),
        ("format_compliance_rate", "Format"),
        ("syndrome_consistency_rate", "Syndrome cons. (rate)"),
        ("mean_hamming_overlap", "Mean Hamming"),
        ("mean_syndrome_consistency", "Mean synd. consistency"),
        ("exact_match_pymatching", "Exact match PM frame"),
    ]
    labels: list[str] = []
    vals: list[float] = []
    for k, lab in keys:
        if k in data and data[k] is not None:
            v = float(data[k])
            labels.append(lab)
            vals.append(v)
    if not labels:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    def _c(lab: str) -> str:
        if "total reward" in lab.lower():
            return "#059669"
        if "Logical correction" in lab:
            return "#2563eb"
        return "#64748b"

    colors = [_c(lab) for lab in labels]
    ax.barh(labels, vals, color=colors, height=0.55)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Score (0-1)")
    title = f"Held-out eval ({int(data.get('episodes', 0))} ep, {data.get('level', '?')})"
    if data.get("name"):
        title += f" — {data['name']}"
    ax.set_title(title, fontsize=11)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 0.95), i, f"{v:.3f}", va="center", fontsize=9, color="#334155")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_sft_curriculum(data: dict, out: Path) -> bool:
    import matplotlib.pyplot as plt

    train = data.get("train") or {}
    levels = train.get("levels") or {}
    if not levels:
        return False
    order = ["L1_warmup", "L2_target", "L3_stretch"]
    names = [n for n in order if n in levels]
    counts = [int(levels[n]) for n in names]
    if not counts:
        return False
    display = [n.replace("_", " ") for n in names]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = range(len(display))
    ax.bar(x, counts, color=["#0ea5e9", "#8b5cf6", "#f59e0b"], width=0.6)
    ax.set_xticks(list(x), display, rotation=15, ha="right")
    ax.set_ylabel("Rows in SFT train split")
    pct = train.get("level_pct") or {}
    subtitle = " / ".join(f"{k.replace('_', ' ')}: {float(pct.get(k, 0)):.0f}%" for k in names)
    ax.set_title(f"SFT dataset curriculum mix (n={sum(counts)})\n{subtitle}", fontsize=10)
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.02, str(c), ha="center", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--eval", type=Path, dest="eval_path", default=None)
    parser.add_argument("--sft-analysis", type=Path, default=None)
    args = parser.parse_args()
    root = _repo_root()
    out_dir = args.out_dir or (root / "figures")
    eval_path = args.eval_path or (root / "data" / "eval_grpo.json")
    sft_path = args.sft_analysis or (root / "data" / "sft_dataset_analysis.json")
    n_ok = 0
    if eval_path.is_file():
        data = json.loads(eval_path.read_text())
        p = out_dir / "eval_metrics_bars.png"
        if _plot_eval_metrics(data, p):
            print(f"  wrote {p}")
            n_ok += 1
    else:
        print(f"  skip eval bars (missing {eval_path})", file=sys.stderr)

    if sft_path.is_file():
        data = json.loads(sft_path.read_text())
        p = out_dir / "sft_curriculum_mix.png"
        if _plot_sft_curriculum(data, p):
            print(f"  wrote {p}")
            n_ok += 1
    else:
        print(f"  skip SFT mix (missing {sft_path})", file=sys.stderr)

    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
