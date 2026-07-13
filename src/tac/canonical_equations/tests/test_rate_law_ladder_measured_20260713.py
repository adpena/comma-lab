"""Regression checks for the measured rate-law-ladder successor row."""
from pathlib import Path

import pytest

from tac.canonical_equations import rate_law_ladder_measured_20260713 as M


def test_successor_and_receipt_anchors_exist():
    assert M.EQUATION_ID == "rate_law_ladder_v2_measured"
    assert M.PREDECESSOR_EQUATION_ID == "rate_law_ladder_v1"
    for path in (M.MEMO, M.D36_RECEIPT, M.D37_RECEIPT, M.D39_SPEC):
        assert Path(path).is_file(), f"missing measured ladder anchor: {path}"


def test_d36_measured_gap_arithmetic():
    assert M.D36_CONDITIONAL_GAP_BITS == 147_616
    assert M.D36_UNCONDITIONAL_CODE_BITS - M.D36_CONDITIONAL_GAP_BITS == 15_224
    assert pytest.approx(
        100 * (M.D36_CONDITIONAL_GAP_BITS / 8) / M.ARCHIVE_BYTES
    ) == M.D36_GAP_PERCENT_OF_ARCHIVE_RATE
    assert M.rate_term_for_bits(M.D36_CONDITIONAL_GAP_BITS) == pytest.approx(
        0.012286429403010305
    )
    assert M.D36_NET_SAVING_BITS_AFTER_MODEL < 0


def test_d37_verdict_and_d38_scope_are_not_overpromoted():
    assert M.D37_MI_NET_CI95_BITS[0] > 0
    assert M.D37_PHASE_AWARE_NET_CI95_BITS[1] < 0
    assert M.D38_LOCAL_OBSTRUCTION_CLASS == "neutral"
    assert M.D38_LOCAL_IDEAL_TWIST_BITS == 0
    assert "NOT-TYPED" in M.D38_GLOBAL_EXTENSION_STATUS
    assert "event_marks_telemetry_implementation" in M.REMAINING_OWED
