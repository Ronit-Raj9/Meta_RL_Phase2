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

Output exactly ONE line and nothing else. Do not write reasoning, markdown, bullets, analysis, or explanations. Your entire response must match this exact format:
X_ERRORS=[qubit_ids] Z_ERRORS=[qubit_ids]

Valid one-line examples:
X_ERRORS=[] Z_ERRORS=[]
X_ERRORS=[] Z_ERRORS=[4]
X_ERRORS=[2] Z_ERRORS=[5,6]"""


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

_X_PATTERN = re.compile(r"X_ERRORS\s*=\s*\[([^\]]*)\]", re.IGNORECASE)
_Z_PATTERN = re.compile(r"Z_ERRORS\s*=\s*\[([^\]]*)\]", re.IGNORECASE)
_X_KEY = re.compile(r"X_ERRORS", re.IGNORECASE)


@dataclass(frozen=True)
class ParseResult:
    x_errors: list[int]
    z_errors: list[int]
    parse_success: bool        # True iff BOTH X_ERRORS and Z_ERRORS parsed cleanly
    parse_partial: bool        # True iff exactly one of the two parsed cleanly
    raw_response: str

    @property
    def format_score(self) -> float:
        """Score for Reward 4 (format compliance)."""
        if self.parse_success:
            return 1.0
        if self.parse_partial:
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

    Algorithm (locked spec):
      1. Receive the full model response string.
      2. If the model wrapped output in fenced code blocks, focus on the
         last fenced block.
      3. Find ALL occurrences of ``X_ERRORS`` (case-insensitive) and take
         the LAST one - this skips the example block in the prompt and
         lands on the model's actual answer.
      4. Slice from that position to end of string and look for the joint
         pattern ``X_ERRORS\\s*=\\s*\\[(...)\\]\\s*Z_ERRORS\\s*=\\s*\\[(...)\\]``.
      5. Validate every parsed integer is in ``[0, max_qubit_id]``; reject
         duplicates within a list.
      6. ``parse_success`` requires BOTH lists to parse cleanly;
         ``parse_partial`` is set when exactly one parsed cleanly OR both
         keys appear but tokens were dirty.
    """
    if not isinstance(raw_response, str):
        return ParseResult([], [], False, False, raw_response="")

    # 1-2: fence handling.
    fenced = re.findall(r"```(?:[^\n]*)\n(.*?)```", raw_response, re.DOTALL)
    search_text = fenced[-1] if fenced else raw_response

    # 3: find the LAST X_ERRORS key occurrence.
    x_keys = list(_X_KEY.finditer(search_text))
    if x_keys:
        last_x_pos = x_keys[-1].start()
        # Slice forward; if the LAST key has no [...] payload (truncated),
        # fall back to the previous key so the parser is forgiving.
        slice_text = search_text[last_x_pos:]
        if not _X_PATTERN.search(slice_text) and len(x_keys) > 1:
            last_x_pos = x_keys[-2].start()
            slice_text = search_text[last_x_pos:]
    else:
        slice_text = search_text

    # 4: pull X and Z lists. Always grab the LAST Z_ERRORS so trailing
    # commentary doesn't fool us either.
    x_match = _X_PATTERN.search(slice_text)
    z_matches = list(_Z_PATTERN.finditer(slice_text))
    z_match = z_matches[-1] if z_matches else None

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

    return ParseResult(
        x_errors=x_errors,
        z_errors=z_errors,
        parse_success=parse_success,
        parse_partial=parse_partial,
        raw_response=raw_response,
    )


def format_completion(x_errors: Iterable[int], z_errors: Iterable[int]) -> str:
    """The canonical SFT target string. Inverse of :func:`parse_action`."""
    x_str = ",".join(str(q) for q in sorted(set(x_errors)))
    z_str = ",".join(str(q) for q in sorted(set(z_errors)))
    return f"X_ERRORS=[{x_str}] Z_ERRORS=[{z_str}]"
