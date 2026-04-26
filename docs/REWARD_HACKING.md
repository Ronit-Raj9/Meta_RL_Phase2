# Reward Hacking Defense

## Threat model

GRPO optimizes a policy directly against the scalar reward signal, so any
exploitable gap between "what the reward measures" and "what the task
actually requires" becomes a high-gradient attractor — the policy will
collapse into the cheapest hack the verifier cannot see. Because our stack
is verifier-style (Stim ground truth + PyMatching reference frame + a text
parser), every reward must be a *physical invariant* or a *cross-checkable
auxiliary*, not a regex of the model's own output.

## The 5-reward composite

All five rewards are pure functions
`(parsed_action, sample, layout) -> float in [0, 1]` evaluated independently
and combined as a weighted sum clamped to `[0, 1]`.

| Name | Weight | What it rewards | What it cannot reward |
|------|--------|-----------------|-----------------------|
| `logical_correction` | 0.40 | Predicted Pauli frame, when applied at end-of-circuit, induces the same logical-Z observable flip Stim recorded as ground truth. | Anything not derivable from Stim's observable trace. Cannot be reverse-engineered from the prompt alone. |
| `syndrome_consistency` | 0.20 | Hamming similarity between *predicted* final-round detector parities (induced by the predicted X errors) and the *observed* final-round detector parities. | Earlier-round detectors are intentionally unscored; partial credit on bit-flipped hallucinations. |
| `hamming_overlap` | 0.20 | Mean of set-aware Jaccard(X_pred, X_ref) and Jaccard(Z_pred, Z_ref) against PyMatching's reference Pauli frame. | Symmetric "predict-empty-when-empty" hacks: the set-aware rule scores 0.0 for missed errors and 0.0 for false alarms. |
| `format_compliance` | 0.10 | 1.0 only when the strict canonical `X_ERRORS=[...] / Z_ERRORS=[...]` form parses BOTH lists cleanly (lenient/partial parses score 0.5; nothing parseable scores 0.0). | Cannot be earned by whitespace tricks alone — the parser validates that every integer is in `[0, num_data_qubits)` and de-duplicates. |
| `pymatching_beat` | 0.10 | 1.0 iff PyMatching got this syndrome wrong AND the model got it right. | Imitation of PyMatching: matching its output exactly forfeits the bonus on every syndrome PyMatching also gets right (most of them). |

Weights sum to 1.00. Source of truth: `openenv.yaml` and
`qubit_medic.config.REWARD_WEIGHTS`. Implementations live in
`qubit_medic/server/rewards.py`.

## Attack/defense matrix

| Hack the model could attempt | Channel(s) that catch it |
|---|---|
| Output empty string | `format_compliance = 0` (no strict pattern, no lists) |
| Memorize one canonical Pauli frame across all syndromes | `hamming_overlap` drops on novel syndromes (Jaccard against per-syndrome PyMatching reference); `logical_correction` drops to chance |
| Match PyMatching exactly on every shot | `pymatching_beat = 0` whenever PyMatching is also correct (which is most syndromes), so the 0.10 channel never fires |
| Output a random valid format string | `logical_correction` collapses to ~chance; `syndrome_consistency` and `hamming_overlap` both drop |
| Skip syndrome reasoning, copy the in-prompt example block | The parser slices from the LAST `X_ERRORS=` key (so the prompt's example doesn't win); `syndrome_consistency` then penalises the stale answer |
| Game the format checker with whitespace / capitalisation tricks | `format_compliance` is parseability-based: `_parse_int_list` rejects out-of-range integers, drops dups, and `strict_format` requires the canonical `=[...]` form for the full 1.0 |
| Inject extra correction operators ("over-correct") | `hamming_overlap` uses set-aware Jaccard whose union grows with false alarms (precision-aware), so over-correction strictly lowers the score |
| Predict an empty frame when the syndrome is non-empty (FIX 1, 2026-04) | `syndrome_consistency` is **capped at 0.5** when prediction is empty AND any detector fired — the empty-everywhere collapse mode can never reach the full 1.0 |
| Output a logical-flipped Pauli frame that *coincidentally* satisfies final-round parities | `logical_correction = 0` because the implied observable flip differs from Stim ground truth; `hamming_overlap` also drops vs PyMatching's reference frame |
| Hallucinate qubit IDs outside `[0, num_data_qubits)` to spoof a long answer | `_parse_int_list` drops out-of-range tokens and flags `parse_success=False`, so `format_compliance` collapses to 0.0/0.5 |
| Exploit per-axis Jaccard (predict X right, Z empty when Z is empty) | The set-aware rule (`true_set` empty AND `pred_set` empty -> 1.0; either non-empty asymmetric -> 0.0) plus the 0.5 mean across axes prevents winning by guessing one axis is empty |
| Time-stall (delay step beyond `EPISODE_TIMEOUT_SECONDS`) to evade scoring | The env builds a zero-reward `RewardBreakdown` and marks the episode `truncated=True`, so timeouts strictly hurt |

## Hard guarantees

These are physical invariants that hold by construction; no policy can
satisfy them via parser games:

- **Logical-Z preservation (Stim ground truth).** `predicted_observable_flip`
  re-applies the predicted X errors as a Pauli frame at end-of-circuit and
  computes the implied flip on the logical Z observable. `logical_correction`
  is 1.0 iff the implied flip equals `sample.actual_observable_flip` recorded
  by Stim. There is no way to fake this without genuinely solving the decoder
  task on this syndrome.
- **Final-round detector arithmetic.** `_syndrome_from_pauli_frame` computes
  the implied final-round detector bits from the predicted X errors and the
  detector-to-data-qubit incidence map (`final_detector_supports`, derived
  from Euclidean adjacency in the rotated memory_z layout). These bits are
  compared against `sample.syndrome_bits` directly — the model never sees
  the comparison target.
- **PyMatching reference frame.** `sample.pymatching_x_errors` /
  `pymatching_z_errors` are computed by the Sparse Blossom matching decoder
  (PyMatching v2) for this exact syndrome; the model has no access to them
  at action time.
- **Hidden ground truth.** `DecoderState` carries `true_x_errors`,
  `true_z_errors`, `actual_observable_flip`, `pymatching_observable_pred`,
  `circuit_text`, and `dem_text`, but the externally-visible `state()`
  endpoint *deliberately omits all of these* (see
  `qubit_medic/server/environment.py` `state()` method). Only the reward
  functions see them.
- **LLM-space → Stim-space conversion.** Predicted qubit ids are mapped from
  LLM-space (0..N-1, the only IDs the prompt advertises) into Stim's
  internal coordinate system before scoring (`layout.llm_to_stim`). The
  model can't gain anything by guessing Stim's internal numbering.
- **Episode pairing enforcement.** `step()` raises a clean `ValueError` for
  unknown episode IDs (compliance audit 2026-04). A trainer cannot replay
  step() against a stale episode to harvest a stale reward.

## Known weaknesses

Honest accounting of what this composite still does **not** catch:

- **Hamming-similarity is not a strict equality on syndrome consistency.**
  A predicted Pauli frame whose final-round implied bits happen to overlap
  the observed bits on most positions (without being correct) still scores
  partial credit on `syndrome_consistency`. The 0.5 cap on
  empty-prediction-vs-active-syndrome closes the worst case, but a *near*
  empty answer that flips one well-chosen qubit can still earn a high
  consistency score on prompts where most final-round detectors quiesced.
- **`hamming_overlap` treats the PyMatching reference as ground truth.**
  PyMatching is itself near-optimal but not optimal; on syndromes where the
  Stim-true correction differs from PyMatching's, a model that found the
  *true* correction is penalised on Reward 3 even though it's right on
  Reward 1. We accept this trade-off because Reward 5 (`pymatching_beat`)
  is the channel that explicitly rewards out-performing PyMatching, and it
  has its own 0.10 weight.
- **No per-round detector scoring.** Earlier-round detectors carry signal
  the LLM could exploit, but we score only the final round to keep the
  Pauli-frame action space tractable. A model could in principle "ignore
  intermediate rounds" without penalty as long as its terminal frame is
  correct — which is the same trade-off AlphaQubit made.
- **Format compliance is binary-ish (0 / 0.5 / 1).** The 2026-04 spec
  rewrite removed full credit for non-canonical resemblances; this is
  intentional, but it means a model that emits beautiful chain-of-thought
  reasoning *and then forgets the final `X_ERRORS=[...]` line* gets
  reduced credit identical to a near-miss. We trade interpretability for
  anti-gaming.
- **`pymatching_beat` is sparse.** Most syndromes are easy and PyMatching
  wins; the bonus only fires on the hard tail. This is by design (the
  trajectory of its mean is the proof of post-imitation behaviour) but it
  means GRPO sees this signal as ~zero-sum noise for most of training.
- **No protection against constant-stream prompts.** If a future trainer
  modification kept episodes alive across `step()` calls, the active-episode
  bookkeeping could in principle leak via observed reward statistics. The
  current single-step-per-episode design (`done=True` after every `step`)
  prevents this; do not relax it without a fresh hacking-surface review.
