# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_v15_grammar_parametrized_scorer_solve_20260723 import (
    BREAK_EVEN_SCORE_PER_BYTE,
    N64_RECEIPT,
    N64_RECEIPT_SHA256,
    N600_RECEIPT,
    N600_RECEIPT_SHA256,
    admit_realized_template_step,
    derivation_edges,
    fisher_trace_from_winner_rival_margin,
    measured_v15_anchor,
)

REPO = Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fisher_margin_and_reverse_waterfill_admission_fail_closed() -> None:
    assert fisher_trace_from_winner_rival_margin(0.0) == pytest.approx(0.5)
    assert fisher_trace_from_winner_rival_margin(3.0) < 0.1
    assert admit_realized_template_step(
        target_error_improvement=1,
        harmful_off_target_flips=0,
        score_gain=2.0 * BREAK_EVEN_SCORE_PER_BYTE,
        delta_archive_bytes=1,
    ).admitted
    assert (
        admit_realized_template_step(
            target_error_improvement=1,
            harmful_off_target_flips=1,
            score_gain=1.0,
            delta_archive_bytes=1,
        ).reason
        == "HARD_ZERO_COLLATERAL_VIOLATION"
    )
    assert not admit_realized_template_step(
        target_error_improvement=0,
        harmful_off_target_flips=0,
        score_gain=0.0,
        delta_archive_bytes=1,
    ).admitted


def test_anchor_binds_valid_receipts_and_preserves_pointer_honesty() -> None:
    assert _sha256(REPO / N64_RECEIPT) == N64_RECEIPT_SHA256
    assert _sha256(REPO / N600_RECEIPT) == N600_RECEIPT_SHA256
    anchor = measured_v15_anchor()
    assert anchor["archive_bytes"] == 133_941
    assert anchor["movable_conditional_d_seg"] == pytest.approx(0.291615222639)
    assert anchor["lane_conditional_d_seg"] == pytest.approx(0.435195521828)
    assert anchor["fork_passed"] is False
    assert anchor["pointer_moved"] is False
    assert anchor["score_claim"] is False
    assert len(derivation_edges()) == 7
