# Figures

## Training trajectories (synthetic / baseline-anchored)

* `total_reward.png` — mean total episode reward vs step
* `logical_correction.png` — logical correction rate vs step
* `pymatching_beat_rate.png` — PyMatching beat rate vs step

Reproduce: `python -m scripts.plot_results --baselines data/baseline_results.json --out-dir figures`

## Data-driven summaries (from `data/*.json`)

* `eval_metrics_bars.png` — bars for `data/eval_grpo.json` (LCR, PM beat, format, overlaps, mean total reward, …)
* `sft_curriculum_mix.png` — SFT train row counts / mix from `data/sft_dataset_analysis.json`

Reproduce: `python -m scripts.plot_data_figures --out-dir figures`

## Other

* `grid_hero.png` — static grid (also used in README header)
* `grid_animation.gif` — short Stim / layout animation from `scripts/animate_grid.py`
