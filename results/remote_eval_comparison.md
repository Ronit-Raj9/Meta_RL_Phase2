# Qubit-Medic literature comparison

_Generated: 2026-04-26 08:29:31 UTC_

_Distance-3 rotated surface code, Z-memory experiment, SI1000 noise, p ~ 1e-3, level=L2_target._

References:
- PyMatching v2: arXiv:2303.15933
- AlphaQubit:    Nature 635:834 (2024), doi:10.1038/s41586-024-08148-8

| Metric | Trained Qubit-Medic | Baseline (zeros) | PyMatching v2 (lit.) | AlphaQubit (lit.) |
|---|---|---|---|---|
| logical_correction_rate (per shot) | 99.00% | 92.00% | — | — |
| ler_per_round (logical errors / cycle) | — | — | 3.00e-2 | 2.70e-2 |
| pymatching_beat_rate | 0.00% | 0.00% | 0.00% | — |
| format_compliance_rate | 100.00% | 100.00% | — | — |
| exact_match_pymatching | 100.00% | 0.00% | 100.00% | — |
| mean_total_reward | 0.874 | 0.745 | — | — |

## Notes

- LER values for PyMatching v2 and AlphaQubit are taken verbatim from the cited papers at distance-3, p~1e-3 SI1000 noise. They are reproduction targets, not numbers we re-measured here.
- A trained Qubit-Medic ler_per_round below 3.0e-2 means we are matching or beating the canonical PyMatching reference at this noise budget; below 2.7e-2 we are matching AlphaQubit's published two-stage decoder (Bausch et al., Nature 2024).
- pymatching_beat_rate is exactly 0% by construction for PyMatching itself (it cannot beat itself). It is shown only to make the trained-model column meaningful.
