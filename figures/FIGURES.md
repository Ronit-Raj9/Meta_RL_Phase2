# Figures

All plots are saved as PNG (150 dpi unless noted) with axis labels carrying
explicit units. Reproduction commands are listed under each section.

## Money plot — before vs after RLHF training

* `before_after_comparison.png` — two side-by-side bar charts comparing the
  four canonical conditions (Random baseline, Base Qwen2.5-3B, SFT-only,
  SFT + GRPO) on `logical_correction_rate` (left, fraction of shots in
  [0, 1]) and `pymatching_beat_rate` (right, fraction of shots in [0, 1]).
  This is the headline judges-rubric "money plot": the SFT + GRPO bar
  should clearly dominate the un-trained conditions on the left panel and
  show a non-zero beat-rate on the right panel.

Reproduce (after running per-condition evals into `data/eval/*.json`):

```
python scripts/make_comparison_plot.py --eval-dir data/eval \
    --out figures/before_after_comparison.png
```

The script prints a helpful error listing every expected JSON file if any
eval result is missing.

## Training trajectories (synthetic / baseline-anchored)

* `total_reward.png` — mean total episode reward (y, dimensionless 0-1
  composite of logical/syndrome/hamming/format/beat sub-rewards) vs
  training step (x, gradient updates). Horizontal lines mark Random,
  All-zeros, and PyMatching-imitator reward floors so the trained-model
  curve can be read against fixed baselines.
* `logical_correction.png` — logical correction rate (y, fraction of
  shots in [0, 1]) vs training step (x). Reference lines show
  PyMatching, AlphaQubit (Bausch et al., Nature 2024, ~0.973), and
  All-zeros (~0.985) on the same axes for direct comparison.
* `pymatching_beat_rate.png` — fraction of syndromes (y, in [0, 1])
  where the LLM corrects but PyMatching does not, vs training step (x).
  This is the "we moved past pure imitation" diagnostic — non-zero is
  the win condition.

Reproduce: `python -m scripts.plot_results --baselines data/baseline_results.json --out-dir figures`

## Data-driven summaries (from `data/*.json`)

* `eval_metrics_bars.png` — horizontal bars of held-out eval metrics
  (logical correction, format, syndrome consistency, mean Hamming
  overlap, mean total reward, etc.) for the trained model. X-axis is
  score in [0, 1]; one row per metric. Sourced from
  `data/eval_grpo.json`.
* `sft_curriculum_mix.png` — vertical bars showing rows-per-curriculum
  level (y, integer counts) in the SFT training split (L1 warmup / L2
  target / L3 stretch). Confirms the 40/50/10 curriculum mix used to
  bootstrap the policy before GRPO.

Reproduce: `python -m scripts.plot_data_figures --out-dir figures`

## Scene / animation assets

* `grid_hero.png` — single-frame static visualisation of the distance-3
  rotated surface-code data-qubit grid with one example error +
  prediction overlay. Used in the README header. Axes are spatial qubit
  coordinates (no numeric units; legend identifies data qubits, actual
  errors, predicted corrections, and the logical-Z support).
* `grid_animation.gif` — short animated rollout of the same grid across
  episodes, useful for talks and the README banner. Each frame shows
  one syndrome → action → outcome cycle.

## Figure-by-figure rubric audit (2026-04)

| File | X-axis (units) | Y-axis (units) | Title | Thumbnail-legible |
| --- | --- | --- | --- | --- |
| `total_reward.png` | Training step (steps) | Mean total reward (0-1) | yes | yes |
| `logical_correction.png` | Training step (steps) | Logical correction rate (0-1) | yes | yes |
| `pymatching_beat_rate.png` | Training step (steps) | Fraction of syndromes where LLM beats PM (0-1) | yes | yes |
| `eval_metrics_bars.png` | Score (0-1) | metric labels (categorical) | yes | yes |
| `sft_curriculum_mix.png` | curriculum-level labels (categorical) | Rows in SFT train split (count) | yes | yes |
| `grid_hero.png` | spatial (legend) | spatial (legend) | yes (frame caption) | yes |
| `grid_animation.gif` | spatial (legend) | spatial (legend) | per-frame caption | yes |
| `before_after_comparison.png` | Decoder condition (categorical) | LCR / PM-beat (fraction, 0-1) | yes | yes (will be) |
