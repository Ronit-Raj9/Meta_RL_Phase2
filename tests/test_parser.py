"""Tests for the action parser. Covers the cases the participant guide
would care about: chain-of-thought rambling, code-fenced answers, partial
keys, garbage tokens, out-of-range qubits.
"""
from __future__ import annotations

from qubit_medic.prompts import parse_action, format_completion


def test_clean_parse():
    p = parse_action("X_ERRORS=[1,3,5] Z_ERRORS=[2]", num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [1, 3, 5]
    assert p.z_errors == [2]
    assert p.format_score == 1.0


def test_chain_of_thought_then_answer():
    raw = (
        "Let me think step by step. The syndrome bit on round 2 fired, "
        "so qubit 4 likely had an X error.\n\n"
        "Final answer:\nX_ERRORS=[4] Z_ERRORS=[]"
    )
    p = parse_action(raw, num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [4]
    assert p.z_errors == []


def test_code_fenced_answer():
    raw = "Here is the prediction:\n```\nX_ERRORS=[0,2] Z_ERRORS=[5]\n```\n"
    p = parse_action(raw, num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [0, 2]
    assert p.z_errors == [5]


def test_partial_only_x():
    p = parse_action("X_ERRORS=[1,2]", num_data_qubits=9)
    assert p.parse_success is False
    assert p.parse_partial is True
    assert p.x_errors == [1, 2]
    assert p.format_score == 0.5


def test_unparseable():
    p = parse_action("the qubits are around 3 and 7", num_data_qubits=9)
    assert p.parse_success is False
    assert p.parse_partial is False
    assert p.format_score == 0.0


def test_out_of_range_dropped():
    p = parse_action("X_ERRORS=[1,99,3] Z_ERRORS=[]", num_data_qubits=9)
    # 99 is out of range; should be dropped but still considered a clean parse
    # at the structural level if the rest of the tokens are clean.
    assert 99 not in p.x_errors
    # The presence of an out-of-range token marks us as partial, not full.
    assert p.parse_success is False


def test_dedup_and_sort():
    p = parse_action("X_ERRORS=[5,1,3,3,1] Z_ERRORS=[]", num_data_qubits=9)
    assert p.x_errors == [1, 3, 5]
    assert p.parse_success is True


def test_format_completion_inverse():
    s = format_completion([3, 1, 5], [2])
    p = parse_action(s, num_data_qubits=9)
    assert p.x_errors == [1, 3, 5]
    assert p.z_errors == [2]
    assert p.parse_success is True


def test_case_insensitive():
    p = parse_action("x_errors=[1] z_errors=[2]", num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [1]
    assert p.z_errors == [2]


def test_empty_lists():
    p = parse_action("X_ERRORS=[] Z_ERRORS=[]", num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == []
    assert p.z_errors == []
    assert p.strict_format is True
    assert p.format_score == 1.0


# --------------------------------------------------------------------------- #
# Lenient-mode tests (new): parser must surface the model's answer even when
# the format drifts slightly, while keeping `strict_format` False so the GRPO
# Reward 4 still pushes the model toward the canonical wording.
# --------------------------------------------------------------------------- #


def test_lenient_colon_separator():
    p = parse_action("X_ERRORS: [1,3] Z_ERRORS: [2]", num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [1, 3]
    assert p.z_errors == [2]
    assert p.strict_format is False
    assert p.format_score == 0.5  # lenient, not canonical


def test_lenient_parens():
    p = parse_action("X_ERRORS=(1,3) Z_ERRORS=(2)", num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [1, 3]
    assert p.z_errors == [2]
    assert p.strict_format is False


def test_lenient_dashed_key():
    p = parse_action("X-ERRORS=[1] Z-ERRORS=[2]", num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [1]
    assert p.z_errors == [2]
    assert p.strict_format is False


def test_lenient_spaced_key():
    p = parse_action("X ERRORS=[1] Z ERRORS=[2]", num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [1]
    assert p.z_errors == [2]
    assert p.strict_format is False


def test_strict_format_flag_for_canonical():
    p = parse_action("X_ERRORS=[1] Z_ERRORS=[2]", num_data_qubits=9)
    assert p.strict_format is True
    assert p.format_score == 1.0


def test_boxed_latex_wrapper_unwraps():
    raw = (
        "Looking at the syndrome, only one bit fired.\n"
        r"\boxed{X_ERRORS=[3] Z_ERRORS=[]}"
    )
    p = parse_action(raw, num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [3]
    assert p.z_errors == []
    assert p.strict_format is True


def test_reasoning_then_canonical_answer_is_strict():
    raw = (
        "The bottom-right Z stabilizer fired in round 2, suggesting an X on "
        "qubit 7. Final answer: X_ERRORS=[7] Z_ERRORS=[]"
    )
    p = parse_action(raw, num_data_qubits=9)
    assert p.parse_success is True
    assert p.x_errors == [7]
    assert p.z_errors == []
    assert p.strict_format is True


def test_lenient_does_not_rescue_pure_garbage():
    p = parse_action(
        "Looking at this, qubits 3 and 5 might be flipped.",
        num_data_qubits=9,
    )
    assert p.parse_success is False
    assert p.parse_partial is False
    assert p.strict_format is False
    assert p.format_score == 0.0
