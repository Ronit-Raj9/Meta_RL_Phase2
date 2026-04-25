# Architecture - Qubit-Medic

The system has three concentric layers, each behind a clean contract.

```
+-------------------------------------------------------------+
|                       LLM trainer                          |
|                  (TRL GRPOTrainer + Unsloth)                |
|                                                             |
|   for each step:                                            |
|     prompts = sample(prompt_pool)                          |
|     completions = model.generate(prompts, n=4)             |
|     for c in completions:                                   |
|         rewards = env_client.step(c).info["rewards"]       |
+----------------------------+--------------------------------+
                             | HTTP (or in-process)
                             v
+-------------------------------------------------------------+
|              FastAPI server: qubit_medic.server.app         |
|                                                             |
|   POST /reset    -> DecoderObservation                      |
|   POST /step     -> StepResult (reward + info breakdown)    |
|   GET  /health   -> liveness + curriculum stats             |
|   POST /decode   -> baseline PyMatching prediction          |
+----------------------------+--------------------------------+
                             |
                             v
+-------------------------------------------------------------+
|         DecoderEnvironment (qubit_medic.server.environment) |
|                                                             |
|   reset():                                                  |
|     1. CurriculumScheduler.sample()                         |
|     2. cached: stim.Circuit + DEM + pymatching.Matching     |
|     3. compile_detector_sampler().sample(1) -> syndrome     |
|     4. build_prompt(...)  -> DecoderObservation             |
|                                                             |
|   step(raw_response):                                       |
|     1. parse_action()  ->  ParseResult (X/Z error sets)     |
|     2. layout.llm_to_stim()  remap to Stim qubit IDs        |
|     3. compute_all_rewards():                               |
|        - logical_correction (Stim ground truth)             |
|        - syndrome_consistency (final-round detectors)       |
|        - hamming_overlap (vs PyMatching reference frame)    |
|        - format_compliance (parser output)                  |
|        - pymatching_beat (LLM right & PM wrong)             |
|     4. CurriculumScheduler.update(level, logical_correct)   |
|     5. return StepResult                                    |
+-------------------------------------------------------------+
```

## Trust boundaries

```
+-----------+     prompt + syndrome      +--------------+
|    LLM    | <-------------------------- | Observation  |
+-----------+                             +--------------+
      |
      v raw text
+-----------+     parse + remap          +-----------+
|  Action   | --> [LLM ID space] -----> | Stim ID    |
+-----------+                            +-----------+
                                              |
                                              v scoring
                                        +-----------+
                                        |   State   |
                                        | (server)  |
                                        +-----------+
```

The `DecoderState` (server-side) holds the ground-truth observable flip,
the true error pattern (PyMatching reference frame), and the seed used for
sampling. **None** of this is ever returned to the LLM. This is the
participant guide's `"avoid unrestricted global state"` discipline made
concrete by Pydantic schemas.

## Why a terminal Pauli frame, and what it costs

The LLM emits two integer lists: which data qubits suffered an X error and
which suffered a Z error, **at the moment of final measurement** (a
terminal Pauli frame). For the rotated `memory_z` task this is sufficient
for the logical observable - the destructive Z measurement is exactly the
Z observable, and an X error on a data qubit in the observable's support
flips its measurement outcome.

The trade-off is that an end-of-circuit Pauli frame *only* constrains the
final-round detectors (the ones that incorporate the destructive Z
measurement results). Earlier-round detectors fire only in response to
errors that propagate through the stabilizer rounds, and a terminal frame
cannot say anything about them. Reward 2 (syndrome consistency)
explicitly grades only the final-round detector bits, which matches the
representation's expressive power. The remaining detector bits are
implicitly *available* in the prompt for the LLM to reason about, but
unscored.

## Why five rewards instead of one

The participant guide is emphatic: *"use multiple independent reward
functions, not just one."* Each of our five rewards is independently
verifiable in well under a millisecond and disagrees with at least one
other on degenerate inputs:

* All-zeros agent on a syndrome with a logical-but-undetectable error:
  `logical_correction = 0` but `syndrome_consistency = 1`. The R2 - R1
  disagreement exposes the failure case.
* Random-qubit agent that lands on the right observable parity by luck:
  `logical_correction = 1` but `syndrome_consistency` and
  `hamming_overlap` are both low. R1 alone over-rewards; the others
  expose the lack of understanding.

This decomposition is what the guide calls *"hard to game by
construction."*
