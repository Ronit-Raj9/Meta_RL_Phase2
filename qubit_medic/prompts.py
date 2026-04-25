"""Locked prompt template + parser (master spec, sections 4 + parser).

This module is the *single source of truth* for what the LLM sees during
SFT and GRPO. The exact wording is fixed: anything that drifts the prompt
between phases throws away the SFT investment because RL builds on the
format SFT taught.

Spec sections honoured:
* Section 4 - "The exact prompt template (locked, for both SFT and RL)"
* Section 4 - "The {syndrome_block} format" (round-by-round, X first then Z)
* Section 4 - "The parser specification (critical)"

Parser highlights
-----------------
* Case-insensitive on ``X_ERRORS``/``Z_ERRORS`` keys.
* Tolerant of trailing chain-of-thought, code fences, and whitespace.
* **Takes the LAST occurrence** of ``X_ERRORS`` so the literal example
  inside the prompt's "Examples:" block is never confused for the answer.
* Validates each id against ``[0, max_qubit_id]`` and dedups within a list.
* Returns a partial-credit score (1.0 / 0.5 / 0.0) for Reward 4.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# --------------------------------------------------------------------------- #
# Prompt template (LOCKED - see master spec, section 4)                       #
# --------------------------------------------------------------------------- #

_PROMPT_TEMPLATE = """You are an expert quantum error correction decoder. Your job is to identify which data qubits experienced errors based on syndrome measurements.

A surface code protects 1 logical qubit using {num_data_qubits} data qubits arranged in a {distance}x{distance} grid. Stabilizer measurements detect errors: a '1' means that stabilizer fired (detected something wrong nearby); a '0' means it looks fine. Errors must be deduced from the pattern of stabilizers that fired.

Code distance: {distance}
Number of stabilizer rounds: {rounds}
Physical error rate: {p}
X-stabilizer count per round: {num_x_stabilizers}
Z-stabilizer count per round: {num_z_stabilizers}

{syndrome_block}

Identify which data qubits (numbered 0-{max_qubit_id}) had X-errors and Z-errors. Most syndromes have 0-2 errors; an empty list means no errors of that type.

Reason briefly (1-2 sentences), then output your answer on the LAST line in this EXACT format:
X_ERRORS=[qubit_ids] Z_ERRORS=[qubit_ids]

Examples:
- No errors: X_ERRORS=[] Z_ERRORS=[]
- One Z error on qubit 4: X_ERRORS=[] Z_ERRORS=[4]
- Two errors: X_ERRORS=[2] Z_ERRORS=[5,6]"""


def format_syndrome_block(
    syndrome_bits: Iterable[int],
    rounds: int,
    num_x_stabilizers: int,
    num_z_stabilizers: int,
) -> str:
    """Render detector activations round-by-round, exactly per the spec.

    Format example for distance-3, rounds=3:

        Round 1 X-stabilizers: 0 0 1 0
        Round 1 Z-stabilizers: 0 0 0 0
        Round 2 X-stabilizers: 0 0 1 0
        Round 2 Z-stabilizers: 0 0 0 0
        Round 3 X-stabilizers: 0 0 0 0
        Round 3 Z-stabilizers: 0 0 0 0

    Every round on its own line, X first then Z, space-separated bits, no
    indent, no commas. Rounds are always emitted in full even when all
    bits are zero so the LLM sees consistent shape.

    Stim's detector layout for the rotated-memory experiment is row-major:
    round 0 stabilizers first, then round 1, and so on. For each round it
    interleaves the per-type detectors in the order Stim's circuit was
    generated (we treat the first ``num_x_stabilizers`` per round as X
    and the rest as Z, matching ``per_round_x_z_counts``).
    """
    bits = list(syndrome_bits)
    per_round = num_x_stabilizers + num_z_stabilizers
    if per_round == 0 or rounds == 0 or len(bits) == 0:
        return "(no detectors fired)"

    lines: list[str] = []
    for r in range(rounds):
        offset = r * per_round
        if offset >= len(bits):
            break
        chunk = bits[offset : offset + per_round]
        x_chunk = chunk[:num_x_stabilizers]
        z_chunk = chunk[num_x_stabilizers : num_x_stabilizers + num_z_stabilizers]
        lines.append(
            f"Round {r + 1} X-stabilizers: " + " ".join(str(int(b)) for b in x_chunk)
        )
        lines.append(
            f"Round {r + 1} Z-stabilizers: " + " ".join(str(int(b)) for b in z_chunk)
        )
    used = rounds * per_round
    if used < len(bits):
        tail = bits[used:]
        lines.append("Final-round detectors: " + " ".join(str(int(b)) for b in tail))
    return "\n".join(lines)


def build_prompt(
    *,
    distance: int,
    rounds: int,
    p: float,
    syndrome_bits: list[int],
    num_x_stabilizers: int,
    num_z_stabilizers: int,
    num_data_qubits: int,
) -> str:
    """Assemble the locked prompt the LLM sees on each step.

    Pure function (no I/O, no globals) so the SFT pipeline and GRPO
    rollout produce byte-identical prompt strings - a critical invariant.
    """
    syndrome_block = format_syndrome_block(
        syndrome_bits=syndrome_bits,
        rounds=rounds,
        num_x_stabilizers=num_x_stabilizers,
        num_z_stabilizers=num_z_stabilizers,
    )
    return _PROMPT_TEMPLATE.format(
        num_data_qubits=num_data_qubits,
        distance=distance,
        rounds=rounds,
        p=p,
        num_x_stabilizers=num_x_stabilizers,
        num_z_stabilizers=num_z_stabilizers,
        syndrome_block=syndrome_block,
        max_qubit_id=num_data_qubits - 1,
    )


# --------------------------------------------------------------------------- #
# Output parsing (LOCKED - see master spec, section 4 "Parser specification") #
# --------------------------------------------------------------------------- #
#
# Two-tier parser:
#   * STRICT  - canonical "X_ERRORS=[...] Z_ERRORS=[...]". Only this form
#     scores 1.0 on Reward 4 (format_compliance), so the GRPO signal still
#     pushes the model toward the locked spec wording.
#   * LENIENT - also accepts ":" instead of "=", "()" instead of "[]",
#               "X-ERRORS" / "X ERRORS" key spellings, and tolerates
#               \boxed{...} / **...** wrapping. Used so eval/metrics see
#               the model's actual *answer* whenever it is extractable,
#               instead of silently treating parse failures as
#               "predict no errors" (which hides the bug at p=0.001 where
#               ~95% of syndromes are trivial and an empty prediction is
#               accidentally correct).

# Strict canonical form: "=" + "[]" - required for Reward 4 = 1.0.
_X_PATTERN_STRICT = re.compile(r"X_ERRORS\s*=\s*\[([^\]]*)\]", re.IGNORECASE)
_Z_PATTERN_STRICT = re.compile(r"Z_ERRORS\s*=\s*\[([^\]]*)\]", re.IGNORECASE)

# Lenient form: "=" or ":" separator, "[]" or "()" brackets, and the key may
# be spelt "X_ERRORS" / "X-ERRORS" / "X ERRORS" / "XERRORS".
_X_PATTERN_LENIENT = re.compile(
    r"X[\s_\-]*ERRORS\s*[=:]\s*[\[\(]([^\]\)]*)[\]\)]",
    re.IGNORECASE,
)
_Z_PATTERN_LENIENT = re.compile(
    r"Z[\s_\-]*ERRORS\s*[=:]\s*[\[\(]([^\]\)]*)[\]\)]",
    re.IGNORECASE,
)

# Key locator (lenient) - finds where any X-errors keyword starts so we can
# slice past in-prompt examples and home in on the model's actual answer.
_X_KEY = re.compile(r"X[\s_\-]*ERRORS", re.IGNORECASE)


@dataclass(frozen=True)
class ParseResult:
    x_errors: list[int]
    z_errors: list[int]
    parse_success: bool        # True iff BOTH X_ERRORS and Z_ERRORS parsed cleanly
    parse_partial: bool        # True iff exactly one of the two parsed cleanly
    raw_response: str
    strict_format: bool = False  # True iff matched the canonical "=" + "[]" form

    @property
    def format_score(self) -> float:
        """Score for Reward 4 (format compliance).

        Only the canonical strict form earns 1.0, so the GRPO reward stays
        anchored to the locked spec wording. Lenient parses or partials
        score 0.5; total miss scores 0.0.
        """
        if self.parse_success and self.strict_format:
            return 1.0
        if self.parse_success or self.parse_partial:
            return 0.5
        return 0.0


def _parse_int_list(s: str, max_qubit: int) -> tuple[list[int], bool]:
    """Parse a comma/space-separated integer list. Drops out-of-range and dups.

    Returns ``(qubits_sorted_unique, all_tokens_were_valid)``.
    A token is "invalid" if it isn't an integer or falls outside ``[0, max_qubit)``.
    Duplicates within a list count as silently de-duped, not invalid.
    """
    if not s.strip():
        return [], True
    raw_tokens = re.split(r"[\s,]+", s.strip())
    out: list[int] = []
    all_clean = True
    for tok in raw_tokens:
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            all_clean = False
            continue
        if 0 <= v < max_qubit:
            out.append(v)
        else:
            all_clean = False
    return sorted(set(out)), all_clean


def parse_action(raw_response: str, num_data_qubits: int) -> ParseResult:
    """Convert the LLM's raw text to a :class:`ParseResult`.

    Two-pass algorithm:
      1. Receive the full model response string; normalise common LaTeX/
         markdown wrappers (``\\boxed{...}``, ``**bold**``).
      2. If the model wrapped output in fenced code blocks, focus on the
         LAST fenced block.
      3. Locate all X-errors keys; slice forward from the LAST one (so the
         example block in the prompt never wins).
      4. Try the STRICT pattern (``X_ERRORS=[...]``) first. If both X and Z
         lists match, ``strict_format=True``.
      5. Otherwise try the LENIENT pattern (``=`` or ``:``, ``[]`` or ``()``)
         so a near-miss like ``X_ERRORS: [1]`` still surfaces the model's
         intended prediction.
      6. Validate every parsed integer is in ``[0, max_qubit_id]``; reject
         duplicates within a list.
      7. ``parse_success`` requires BOTH lists to parse cleanly;
         ``parse_partial`` is set when exactly one parsed cleanly OR both
         keys appear but tokens were dirty.

    The lenient fallback exists for *eval/diagnostic honesty*, not to
    weaken the training signal: ``format_score`` (Reward 4) only returns
    1.0 when ``strict_format`` is also True.
    """
    if not isinstance(raw_response, str):
        return ParseResult([], [], False, False, raw_response="", strict_format=False)

    # 1: normalise common wrappers so the regex sees the inner content.
    normalised = raw_response
    # Strip \boxed{...} (LaTeX) - keep inner text.
    normalised = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", normalised)
    # Strip surrounding **bold** / *italic* markers around the format block.
    normalised = re.sub(r"\*+([A-Za-z_][^*]{0,40})\*+", r"\1", normalised)

    # 2: fence handling - prefer last fenced block if present.
    fenced = re.findall(r"```(?:[^\n]*)\n(.*?)```", normalised, re.DOTALL)
    search_text = fenced[-1] if fenced else normalised

    # 3: find the LAST X-errors key occurrence.
    x_keys = list(_X_KEY.finditer(search_text))
    if x_keys:
        last_x_pos = x_keys[-1].start()
        slice_text = search_text[last_x_pos:]
        # If the last key has no payload (truncated), fall back one.
        if (
            not _X_PATTERN_STRICT.search(slice_text)
            and not _X_PATTERN_LENIENT.search(slice_text)
            and len(x_keys) > 1
        ):
            last_x_pos = x_keys[-2].start()
            slice_text = search_text[last_x_pos:]
    else:
        slice_text = search_text

    # 4-5: try strict, then lenient.
    x_match = _X_PATTERN_STRICT.search(slice_text)
    z_matches_strict = list(_Z_PATTERN_STRICT.finditer(slice_text))
    z_match = z_matches_strict[-1] if z_matches_strict else None
    strict_x = x_match is not None
    strict_z = z_match is not None

    if x_match is None:
        x_match = _X_PATTERN_LENIENT.search(slice_text)
    if z_match is None:
        z_matches_lenient = list(_Z_PATTERN_LENIENT.finditer(slice_text))
        z_match = z_matches_lenient[-1] if z_matches_lenient else None

    # 6: extract + validate qubit IDs.
    x_errors: list[int] = []
    z_errors: list[int] = []
    x_clean = z_clean = False
    if x_match is not None:
        x_errors, x_clean = _parse_int_list(x_match.group(1), num_data_qubits)
    if z_match is not None:
        z_errors, z_clean = _parse_int_list(z_match.group(1), num_data_qubits)

    x_present = x_match is not None and x_clean
    z_present = z_match is not None and z_clean
    parse_success = x_present and z_present
    parse_partial = (x_present ^ z_present) or (
        # Both keys present but at least one had garbage tokens.
        (x_match is not None and z_match is not None) and not parse_success
    )

    # strict_format is true only when BOTH X and Z hit the canonical pattern
    # cleanly (no garbage tokens, no out-of-range qubits).
    strict_format = bool(strict_x and strict_z and parse_success)

    return ParseResult(
        x_errors=x_errors,
        z_errors=z_errors,
        parse_success=parse_success,
        parse_partial=parse_partial,
        raw_response=raw_response,
        strict_format=strict_format,
    )


def format_completion(x_errors: Iterable[int], z_errors: Iterable[int]) -> str:
    """The canonical SFT target string. Inverse of :func:`parse_action`."""
    x_str = ",".join(str(q) for q in sorted(set(x_errors)))
    z_str = ",".join(str(q) for q in sorted(set(z_errors)))
    return f"X_ERRORS=[{x_str}] Z_ERRORS=[{z_str}]"
