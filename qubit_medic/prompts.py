"""Prompt formatter and action parser (Section 2.3 + Section 2.5 of the plan).

The prompt is engineered around five sections:

    1. Role declaration
    2. Physics summary (~50 tokens, plain English)
    3. Syndrome data (round-by-round, labelled)
    4. Output format spec (one example included)
    5. Reasoning trigger ("think step by step ...")

Total budget ~250-300 tokens for the prompt; ~150 for the response.

The parser is deliberately permissive on whitespace and bracket style but
strict on the existence of the two key tokens ``X_ERRORS`` and ``Z_ERRORS``.
A partial-credit hook is exposed so Reward 4 can hand out 0.5 for "partly
parseable".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# --------------------------------------------------------------------------- #
# Prompt formatting                                                            #
# --------------------------------------------------------------------------- #

_ROLE = (
    "You are a quantum error-correction decoder. You are decoding errors in "
    "a distance-{distance} rotated surface code memory experiment."
)

_PHYSICS_SUMMARY = (
    "Stabilizers are parity checks measured every round. A *syndrome bit* "
    "is 1 when a stabilizer's measurement disagrees with its previous round, "
    "indicating a nearby physical error. Your job is to look at the syndrome "
    "history and output the smallest physical error pattern (X-flips and "
    "Z-flips on data qubits, identified by integer IDs) that explains it."
)

_OUTPUT_SPEC = (
    "Output format (REQUIRED, exact):\n"
    "    X_ERRORS=[id1,id2,...] Z_ERRORS=[id1,id2,...]\n"
    "Use empty lists when no errors of that type. Example with no errors:\n"
    "    X_ERRORS=[] Z_ERRORS=[]"
)

_REASONING_TRIGGER = (
    "Think step by step about which qubits could have caused this syndrome, "
    "then output your prediction in the required format."
)


def format_syndrome_block(
    syndrome_bits: Iterable[int],
    rounds: int,
    num_x_stabilizers: int,
    num_z_stabilizers: int,
) -> str:
    """Render the detector activations round-by-round.

    Stim emits detectors in a flat row-major order: round 0 stabilisers first,
    then round 1, and so on. We label by round and stabiliser type so the LLM
    can read the temporal structure.
    """
    bits = list(syndrome_bits)
    per_round = num_x_stabilizers + num_z_stabilizers
    lines = ["Syndrome (round-by-round):"]
    if per_round == 0 or rounds == 0 or len(bits) == 0:
        lines.append("    (no detectors fired)")
        return "\n".join(lines)

    for r in range(rounds):
        offset = r * per_round
        if offset >= len(bits):
            break
        chunk = bits[offset : offset + per_round]
        x_chunk = chunk[:num_x_stabilizers]
        z_chunk = chunk[num_x_stabilizers : num_x_stabilizers + num_z_stabilizers]
        lines.append(
            f"    Round {r + 1} X-stabilizers: "
            + " ".join(str(b) for b in x_chunk)
        )
        lines.append(
            f"    Round {r + 1} Z-stabilizers: "
            + " ".join(str(b) for b in z_chunk)
        )
    # Trailing block for the final destructive measurement, if any extras.
    used = rounds * per_round
    if used < len(bits):
        tail = bits[used:]
        lines.append("    Final-round detectors: " + " ".join(str(b) for b in tail))
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
    """Assemble the full prompt the LLM sees on each step.

    Keeping this function pure (no I/O, no globals) means the SFT pipeline
    and the GRPO rollout use byte-identical inputs - a critical invariant.
    """
    syndrome_block = format_syndrome_block(
        syndrome_bits=syndrome_bits,
        rounds=rounds,
        num_x_stabilizers=num_x_stabilizers,
        num_z_stabilizers=num_z_stabilizers,
    )
    return (
        _ROLE.format(distance=distance)
        + "\n\n"
        + _PHYSICS_SUMMARY
        + "\n\n"
        + f"Code parameters: distance={distance}, rounds={rounds}, "
        + f"physical_error_rate={p:g}, data_qubits=0..{num_data_qubits - 1}.\n\n"
        + syndrome_block
        + "\n\n"
        + _OUTPUT_SPEC
        + "\n\n"
        + _REASONING_TRIGGER
    )


# --------------------------------------------------------------------------- #
# Output parsing                                                               #
# --------------------------------------------------------------------------- #

_X_PATTERN = re.compile(r"X_ERRORS\s*=\s*\[([^\]]*)\]", re.IGNORECASE)
_Z_PATTERN = re.compile(r"Z_ERRORS\s*=\s*\[([^\]]*)\]", re.IGNORECASE)


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
    """Convert the LLM's raw text to a ``ParseResult``.

    Tolerant of trailing chain-of-thought, surrounding code fences, and
    casing, but strict on the existence of both ``X_ERRORS`` and ``Z_ERRORS``.
    """
    if not isinstance(raw_response, str):
        return ParseResult([], [], False, False, raw_response="")

    # If the model wrapped its answer in ```...``` blocks, focus on the last one.
    fenced = re.findall(r"```(?:[^\n]*)\n(.*?)```", raw_response, re.DOTALL)
    search_text = fenced[-1] if fenced else raw_response

    x_match = _X_PATTERN.search(search_text)
    z_match = _Z_PATTERN.search(search_text)

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
