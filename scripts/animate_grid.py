"""scripts/animate_grid.py - the animated surface-code grid (Section 10.2).

Renders ``--frames`` syndromes as a 2D heatmap of the rotated surface code:

    * Gray dots:   data qubits in their grid positions.
    * Red glow:    data qubits where actual errors occurred (Stim ground truth).
    * Blue glow:   data qubits where the policy predicted errors.
    * Green border: prediction succeeded (logical state preserved).
    * Red border:   prediction failed.

Saves an animated GIF to ``figures/grid_animation.gif`` plus a single
hero-image PNG (``figures/grid_hero.png``) of the most visually striking
frame. Both are committed to the repo.

By default the animation uses the PyMatching imitator policy as the
"trained model" stand-in, which is fine because the script is policy-agnostic
- pass ``--policy zeros`` or ``--policy random`` to see the contrast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pymatching

from qubit_medic.config import level_by_name, primary_level
from qubit_medic.server.physics import (
    build_circuit,
    build_dem,
    extract_layout,
    pymatching_predicted_pauli_frame,
    rectify_pauli_frame_to_observable,
    sample_episode,
)


# --------------------------------------------------------------------------- #
# Policy implementations (operate on raw syndromes, not LLM strings)          #
# --------------------------------------------------------------------------- #


def _policy_pymatching(syndrome, matching, layout):
    px, pz = pymatching_predicted_pauli_frame(matching, syndrome, layout)
    pm_obs = int(matching.decode(syndrome)[0])
    return rectify_pauli_frame_to_observable(px, pz, pm_obs, layout)


def _policy_zeros(syndrome, matching, layout):
    return [], []


def _policy_random(syndrome, matching, layout, *, rng):
    n = len(layout.data_qubits)
    k = rng.integers(0, max(1, n // 2) + 1)
    chosen = rng.choice(n, size=int(k), replace=False).tolist()
    return [layout.data_qubits[i] for i in sorted(chosen)], []


# --------------------------------------------------------------------------- #
# Frame rendering                                                             #
# --------------------------------------------------------------------------- #


def _draw_frame(ax, layout, sample, predicted_x, predicted_z, success, frame_idx):
    ax.clear()
    coords = layout.data_qubit_coords
    qubits = layout.data_qubits
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    ax.scatter(xs, ys, s=300, c="lightgrey", edgecolors="black",
               linewidths=1.5, zorder=1)

    actual_x_set = set(sample.pymatching_x_errors)  # near-optimal proxy
    actual_z_set = set(sample.pymatching_z_errors)
    pred_x_set = set(predicted_x)
    pred_z_set = set(predicted_z)

    for q, (x, y) in zip(qubits, coords):
        if q in actual_x_set or q in actual_z_set:
            ax.scatter([x], [y], s=900, c="red", alpha=0.30, zorder=0)
        if q in pred_x_set or q in pred_z_set:
            ax.scatter([x], [y], s=600, c="blue", alpha=0.30, zorder=2)
        ax.text(x + 0.2, y + 0.2, str(layout.stim_to_llm([q])[0]),
                fontsize=8, color="dimgray")

    # Highlight observable support row.
    for q in layout.z_observable_support:
        idx = layout.data_qubits.index(q)
        x, y = coords[idx]
        ax.scatter([x], [y], s=80, marker="*", c="gold",
                   edgecolors="black", linewidths=0.8, zorder=3)

    border_color = "green" if success else "crimson"
    pad = 1.0
    if xs and ys:
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(border_color)
        s.set_linewidth(4)
    ax.set_title(
        f"Frame {frame_idx}: actual obs flip={sample.actual_observable_flip}, "
        f"pred={'flip' if predicted_x_includes_obs(predicted_x, layout) else 'no flip'}, "
        f"{'OK' if success else 'FAIL'}",
        fontsize=11,
    )
    # Mini legend.
    legend_elems = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor="lightgrey", markeredgecolor="black",
                   label="data qubit"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=14,
                   markerfacecolor="red", alpha=0.4, label="actual error"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=12,
                   markerfacecolor="blue", alpha=0.4, label="predicted"),
        plt.Line2D([0], [0], marker="*", linestyle="", markersize=12,
                   markerfacecolor="gold", markeredgecolor="black",
                   label="logical-Z support"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8,
              framealpha=0.9)


def predicted_x_includes_obs(predicted_x: list[int], layout) -> int:
    support = set(layout.z_observable_support)
    return sum(1 for q in predicted_x if q in support) % 2


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--level", type=str, default=primary_level().name)
    parser.add_argument("--policy", choices=["pymatching", "zeros", "random"],
                        default="pymatching")
    parser.add_argument("--out-dir", type=str, default="figures")
    parser.add_argument("--gif-name", type=str, default="grid_animation.gif")
    parser.add_argument("--hero-name", type=str, default="grid_hero.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args(list(argv))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    level = level_by_name(args.level)
    circuit = build_circuit(level)
    dem = build_dem(circuit)
    matching = pymatching.Matching.from_detector_error_model(dem)
    layout = extract_layout(circuit)
    print(f"layout: {layout.num_data_qubits} data qubits, "
          f"{layout.num_detectors} detectors, level={level.name}")

    samples = []
    predictions = []
    successes = []
    hero_idx = 0
    hero_score = -1
    for i in range(args.frames):
        s = sample_episode(circuit, matching, layout, seed=args.seed + 1 + i)
        if args.policy == "pymatching":
            px, pz = _policy_pymatching(np.asarray(s.syndrome_bits, dtype=np.uint8),
                                         matching, layout)
        elif args.policy == "zeros":
            px, pz = _policy_zeros(None, matching, layout)
        else:
            px, pz = _policy_random(None, matching, layout, rng=rng)
        implied = predicted_x_includes_obs(px, layout)
        success = implied == s.actual_observable_flip
        samples.append(s)
        predictions.append((px, pz))
        successes.append(success)
        # "Hero" frame = the most visually rich one (most overlapping errors).
        score = (
            len(set(px) | set(s.pymatching_x_errors))
            + (10 if not success else 0)
        )
        if score > hero_score:
            hero_score = score
            hero_idx = i

    # Build animation.
    fig, ax = plt.subplots(figsize=(6, 6))
    def _update(i):
        _draw_frame(ax, layout, samples[i], predictions[i][0],
                    predictions[i][1], successes[i], i)
    anim = animation.FuncAnimation(fig, _update, frames=args.frames,
                                   interval=1000 // args.fps)
    gif_path = out_dir / args.gif_name
    try:
        from matplotlib.animation import PillowWriter
        anim.save(gif_path, writer=PillowWriter(fps=args.fps))
        print(f"  wrote {gif_path}")
    except Exception as exc:
        print(f"  GIF save failed ({exc}); writing per-frame PNGs instead")
        for i in range(args.frames):
            _update(i)
            fig.savefig(out_dir / f"grid_frame_{i:03d}.png", dpi=150)

    # Hero frame.
    _update(hero_idx)
    hero_path = out_dir / args.hero_name
    fig.savefig(hero_path, dpi=300, bbox_inches="tight")
    print(f"  wrote {hero_path}  (frame {hero_idx})")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
