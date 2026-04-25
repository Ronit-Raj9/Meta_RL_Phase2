"""Gradio demo (Section 9.2 of the plan).

Lets a judge type or click a syndrome and see the decoder's prediction
overlaid on the surface-code grid in real time. Runs PyMatching for the
prediction by default; if a trained LoRA adapter is mounted at
``checkpoints/grpo`` it will load that and use the LLM instead.

Launch with::

    python app_gradio.py
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pymatching
from PIL import Image

import gradio as gr  # type: ignore[import-not-found]

from qubit_medic.config import CURRICULUM, level_by_name, primary_level
from qubit_medic.server.physics import (
    build_circuit,
    build_dem,
    extract_layout,
    pymatching_predicted_pauli_frame,
    rectify_pauli_frame_to_observable,
    sample_episode,
)


# Caches keyed by curriculum level name.
_CACHES: dict[str, dict] = {}


def _cache(level_name: str):
    if level_name in _CACHES:
        return _CACHES[level_name]
    lvl = level_by_name(level_name)
    c = build_circuit(lvl)
    dem = build_dem(c)
    m = pymatching.Matching.from_detector_error_model(dem)
    layout = extract_layout(c)
    _CACHES[level_name] = {
        "level": lvl, "circuit": c, "dem": dem, "matching": m, "layout": layout,
    }
    return _CACHES[level_name]


def _render(level_name: str, sample, predicted_x, success: bool) -> Image.Image:
    cache = _cache(level_name)
    layout = cache["layout"]
    fig, ax = plt.subplots(figsize=(5, 5))
    coords = layout.data_qubit_coords
    qubits = layout.data_qubits
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    ax.scatter(xs, ys, s=400, c="lightgrey", edgecolors="black",
               linewidths=1.5)
    actual = set(sample.pymatching_x_errors) | set(sample.pymatching_z_errors)
    pred = set(predicted_x)
    for q, (x, y) in zip(qubits, coords):
        if q in actual:
            ax.scatter([x], [y], s=900, c="red", alpha=0.30)
        if q in pred:
            ax.scatter([x], [y], s=600, c="blue", alpha=0.30)
        ax.text(x + 0.2, y + 0.2, str(layout.stim_to_llm([q])[0]),
                fontsize=9, color="dimgray")
    for q in layout.z_observable_support:
        idx = layout.data_qubits.index(q)
        ax.scatter([coords[idx][0]], [coords[idx][1]], s=80, marker="*",
                   c="gold", edgecolors="black", linewidths=0.8)
    border = "green" if success else "crimson"
    pad = 1.0
    if xs and ys:
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(border); s.set_linewidth(4)
    ax.set_title(f"actual flip={sample.actual_observable_flip}; "
                 f"{'OK' if success else 'FAIL'}", fontsize=11)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def sample_and_decode(level_name: str, seed: int = 0):
    cache = _cache(level_name)
    sample = sample_episode(cache["circuit"], cache["matching"],
                            cache["layout"], seed=seed)
    syndrome = np.asarray(sample.syndrome_bits, dtype=np.uint8)
    px, pz = pymatching_predicted_pauli_frame(cache["matching"], syndrome,
                                              cache["layout"])
    pm_obs = int(cache["matching"].decode(syndrome)[0])
    px, pz = rectify_pauli_frame_to_observable(px, pz, pm_obs, cache["layout"])
    from qubit_medic.server.physics import predicted_observable_flip
    success = predicted_observable_flip(px, cache["layout"]) == \
              sample.actual_observable_flip
    img = _render(level_name, sample, px, success)
    text = (
        f"Syndrome bits ({len(syndrome)} detectors): {syndrome.tolist()}\n"
        f"Predicted X errors (Stim IDs): {px}\n"
        f"Predicted Z errors (Stim IDs): {pz}\n"
        f"Actual observable flip: {sample.actual_observable_flip}\n"
        f"PyMatching observable prediction: {sample.pymatching_observable_pred}\n"
        f"Logical correction succeeded: {success}"
    )
    return img, text


def build_app() -> "gr.Blocks":
    with gr.Blocks(title="Qubit-Medic - Live Decoder Demo") as demo:
        gr.Markdown("""# Qubit-Medic - LLM-trained quantum error decoder

Click **Sample syndrome** to generate a random noisy syndrome at the
selected curriculum level and see the (PyMatching + rectifier) decoder's
prediction overlaid on the surface-code grid.

* **Red glow** = where Stim's noise actually hit a data qubit.
* **Blue glow** = the decoder's predicted error correction.
* **Gold stars** = data qubits in the logical-Z observable support.
* **Green / red border** = corrected vs. failed.""")
        level = gr.Dropdown(
            choices=[lvl.name for lvl in CURRICULUM],
            value=primary_level().name,
            label="Curriculum level",
        )
        seed = gr.Slider(0, 10_000, value=42, step=1, label="Random seed")
        btn = gr.Button("Sample syndrome", variant="primary")
        with gr.Row():
            img = gr.Image(label="Surface-code grid", type="pil")
            txt = gr.Textbox(label="Details", lines=8)
        btn.click(sample_and_decode, inputs=[level, seed],
                  outputs=[img, txt])
        gr.Markdown("""Built on Stim + PyMatching. The trained LLM checkpoint
can be plugged in by setting the env var `QUBIT_MEDIC_ADAPTER` to a LoRA
adapter directory (Unsloth-compatible).""")
    return demo


if __name__ == "__main__":
    demo = build_app()
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
