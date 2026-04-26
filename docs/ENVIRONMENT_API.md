# Environment API

Qubit-Medic exposes an OpenEnv-compliant HTTP server built on top of
`openenv.core.create_fastapi_app`. The server wraps an internal
`DecoderEnvironment` (Stim + PyMatching) through the standard
`Action` / `Observation` / `State` Pydantic shapes.

Source files:

- `qubit_medic/server/openenv_adapter.py`
- `qubit_medic/server/app.py`
- `qubit_medic/server/environment.py`

## OpenEnv contract

| Method | Path | Request model | Response model |
|--------|------|---------------|----------------|
| POST | `/reset` | `openenv.core.types.ResetRequest` | `openenv.core.types.ResetResponse` |
| POST | `/step` | `openenv.core.types.StepRequest` | `openenv.core.types.StepResponse` |
| GET | `/state` | (none) | `qubit_medic.server.openenv_adapter.QubitMedicState` |
| POST | `/state` | (none) | `dict` (mirror of GET; compliance audit 2026-04) |
| POST | `/close` | (none) | `{"ok": True, "closed": True}` |
| GET | `/schema` | (none) | JSON Schema for action/observation models |
| GET | `/metadata` | (none) | `EnvironmentMetadata` |
| GET | `/health` | (none) | liveness payload |
| GET | `/healthz` | (none) | versions probe (Stim, PyMatching, openenv, Python) |
| POST | `/decode` | `{"syndrome": [int], "level": str}` | PyMatching baseline result |

The OpenEnv canonical routes (`/reset`, `/step`, `/state`, `/health`,
`/schema`, `/metadata`, `/mcp`) are wired automatically by
`create_fastapi_app`. The `/healthz`, `/decode`, `POST /state`,
`POST /close`, and `/` (HTML landing) routes are mounted on top by
`qubit_medic/server/app.py`.

Server entry point: `python -m qubit_medic.server.app` or
`uvicorn qubit_medic.server.app:app --host 0.0.0.0 --port 7860`.

## Action dataclass

```python
class QubitMedicAction(Action):
    """LLM-emitted action: the raw text the model generated."""

    raw_response: str = Field(
        default="",
        description="Raw LLM completion text. Server parses to x/z error lists.",
    )
    parsed_x_errors: Optional[list[int]] = Field(
        default=None,
        description="Optional pre-parsed X-error qubit ids (LLM-space). "
                    "When provided, the server skips text parsing.",
    )
    parsed_z_errors: Optional[list[int]] = Field(
        default=None,
        description="Optional pre-parsed Z-error qubit ids (LLM-space).",
    )
    episode_id: Optional[int] = Field(
        default=None,
        description="Server-assigned episode id from the matching reset(). "
                    "If omitted, the most-recent active episode is used.",
    )
```

Field-level notes:

- `raw_response`: the canonical wire format. The server runs
  `qubit_medic.prompts.parse_action(raw_response, num_data_qubits)` to
  recover both error lists. Keeping the wire format as raw text means the
  server retains full control over parsing, and unparseable outputs surface
  cleanly via `format_compliance = 0`.
- `parsed_x_errors` / `parsed_z_errors`: a trainer-only escape hatch for
  baseline policies and unit tests. When set, the server formats a
  synthetic `<answer>X: ... | Z: ...</answer>` string before parsing — the
  same parser path runs either way, so reward semantics are identical.
- `episode_id`: must match the `episode_id` returned by the matching
  `reset()` call. If `None`, the adapter falls back to the most recent
  active episode (`self._last_episode_id`). Stale or unknown ids raise
  `ValueError` from `DecoderEnvironment.step` (compliance audit 2026-04).

## Observation dataclass

```python
class QubitMedicObservation(Observation):
    """OpenEnv observation - mirrors DecoderObservation plus done/reward."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True,
                              arbitrary_types_allowed=True)

    prompt: str = Field(default="", description="Pre-formatted LLM prompt.")
    syndrome_bits: list[int] = Field(default_factory=list,
                                     description="Detector activations (0/1).")
    distance: int = Field(default=0, description="Code distance for this episode.")
    rounds: int = Field(default=0, description="Number of stabilizer rounds.")
    p: float = Field(default=0.0, description="SI1000 base error rate.")
    curriculum_level: str = Field(default="",
                                  description="Curriculum level name.")
    episode_id: int = Field(default=0,
                            description="Server-assigned episode counter.")
    dem_digest: str = Field(default="",
                            description="Short hash of the detector error model.")
    info: dict[str, Any] = Field(default_factory=dict,
                                 description="Per-step extras (reward "
                                             "breakdown, ground-truth flip, "
                                             "PyMatching baseline, etc.).")
```

Plus the standard inherited OpenEnv fields:

- `done: bool` — `True` after every `step` (single-step episodes).
- `reward: Optional[float]` — `None` on `reset`, the weighted total in
  `[0, 1]` after `step`.

`info` payload (after `step`) carries:

| Key | Type | Meaning |
|-----|------|---------|
| `rewards` | `dict[str, float]` | Per-component breakdown (`logical_correction`, `syndrome_consistency`, `hamming_overlap`, `format_compliance`, `pymatching_beat`, `total`) |
| `parsed_action` | `dict` | Deserialised `DecoderAction` (parsed x/z lists, `parse_success`) |
| `actual_observable_flip` | `int` | Stim ground-truth flip of the logical Z observable |
| `pymatching_observable_pred` | `int` | PyMatching's predicted observable flip |
| `pymatching_x_errors` | `list[int]` | PyMatching reference Pauli frame, X axis |
| `pymatching_z_errors` | `list[int]` | PyMatching reference Pauli frame, Z axis |
| `elapsed_seconds` | `float` | Wall time between `reset` and `step` |
| `timed_out` | `bool` | `True` iff `elapsed > EPISODE_TIMEOUT_SECONDS` |
| `curriculum_stats` | `dict` | Live promotion-tracker counters |

## State dataclass

```python
class QubitMedicState(State):
    """Externally-visible state. Physics-truth fields stay server-side."""

    model_config = ConfigDict(extra="allow", validate_assignment=True,
                              arbitrary_types_allowed=True)

    episodes_started: int = 0
    active_episodes: int = 0
    cached_levels: list[str] = Field(default_factory=list)
    curriculum: dict[str, Any] = Field(default_factory=dict)
    last_reward_breakdown: Optional[dict[str, float]] = None
```

The adapter populates a few inherited base-class fields too: `episode_id`
(stringified) and `step_count` (which equals `episodes_started`).

Crucially, `QubitMedicState` deliberately omits the ground-truth fields
held by the inner `DecoderState`: `true_x_errors`, `true_z_errors`,
`actual_observable_flip`, `pymatching_observable_pred`, `circuit_text`,
`dem_text`. Those are visible only inside the reward functions — see
`docs/REWARD_HACKING.md`.

## Episode lifecycle

Single-step episodes (`done=True` after every `step`):

```
client                                 server
------                                 ------
POST /reset            ────────────►   scheduler.sample(level)
                                       _cache_for(level)            (compile Stim circuit
                                                                     and PyMatching matrix
                                                                     once per level)
                                       sample_episode(seed)         (Stim shot ->
                                                                     syndrome bits +
                                                                     observable flip)
                                       build_prompt(...)
                       ◄────────────   Observation { prompt,
                                                     syndrome_bits,
                                                     distance, rounds, p,
                                                     curriculum_level,
                                                     episode_id,
                                                     dem_digest,
                                                     done=False,
                                                     reward=None }

POST /step (action)    ────────────►   parse_action(raw_response)
                                       compute_all_rewards(...)
                                       scheduler.update(...)        (curriculum promotion)
                       ◄────────────   Observation { ..., done=True,
                                                     reward=total,
                                                     info={rewards: {...},
                                                           ...} }
```

Calling `step()` with an unknown `episode_id` raises `ValueError` (turned
into HTTP 400). Calling `step()` after `EPISODE_TIMEOUT_SECONDS` returns
all-zero rewards and `info["timed_out"] = True`.

## Reward computation

After parsing, the env converts predicted qubit IDs from LLM-space
(`0..num_data_qubits-1`) into Stim's internal coordinate system via
`layout.llm_to_stim`, then runs `compute_all_rewards`
(`qubit_medic/server/rewards.py`). Each of the five rewards is a pure
function over `(parsed, sample, layout, final_detector_supports)`; the
combined total is a weighted sum (weights in
`qubit_medic.config.REWARD_WEIGHTS`, mirrored in `openenv.yaml`) clamped
to `[0, 1]`. The breakdown is exposed in `info["rewards"]`, the curriculum
scheduler is updated using only `logical_correction`, and the episode
bookkeeping is dropped (`self._active.pop(episode_id)`). See
`docs/REWARD_HACKING.md` for the per-reward semantics.

## Curriculum

Source: `openenv.yaml` (`curriculum:` block) plus
`qubit_medic.server.curriculum.CurriculumScheduler`.

| Level | Distance | Rounds | p (SI1000) | Promotion threshold |
|-------|----------|--------|------------|---------------------|
| `L1_warmup` | 3 | 1 | 0.0001 | 0.80 |
| `L2_target` | 3 | 3 | 0.001 | 0.70 |
| `L3_stretch` | 5 | 5 | 0.001 | 0.30 |

The scheduler samples a level on each `reset()`. Promotion thresholds
gate progression via the running `logical_correction` rate at the current
level. Levels `L1_warmup` and `L2_target` are pre-warmed at server boot
(`_get_shared_inner` in the adapter calls `_cache_for` on both);
`L3_stretch` compiles lazily on first selection.

## Local rollout example

```python
from qubit_medic.server.openenv_adapter import (
    QubitMedicAction,
    QubitMedicEnvironment,
)

env = QubitMedicEnvironment()
obs = env.reset(seed=42)                 # QubitMedicObservation
print("level:", obs.curriculum_level, "syndrome bits:", len(obs.syndrome_bits))
print("prompt preview:", obs.prompt[:120], "...")

# Pretend the LLM emitted nothing useful: the parser will return empty
# lists, format_compliance = 0, syndrome_consistency capped at 0.5.
action = QubitMedicAction(
    raw_response="X_ERRORS=[]\nZ_ERRORS=[]",
    episode_id=obs.episode_id,
)
result = env.step(action)
print("reward:", result.reward, "done:", result.done)
print("breakdown:", result.info["rewards"])
print("pymatching reference frame:", result.info["pymatching_x_errors"],
      result.info["pymatching_z_errors"])
```

For HTTP usage, hit the live server with `curl` against `/reset` then
`/step` (see the swagger UI at `/docs`), or use any OpenEnv-compatible
client.
