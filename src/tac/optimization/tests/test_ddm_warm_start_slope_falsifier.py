from __future__ import annotations

import math

import pytest

from tac.optimization.ddm_warm_start_slope_falsifier import (
    ObjectiveTerms,
    derive_warm_start_gap,
    evaluate_bounded_slope_window,
)


def _gap():
    return derive_warm_start_gap(
        wseg=ObjectiveTerms(seg=2.4, pose=12.0),
        wjoint=ObjectiveTerms(seg=7.0, pose=6.0),
    )


def test_gap_ratio_is_derived_from_score_term_gaps() -> None:
    gap = _gap()
    assert gap.seg_advantage == pytest.approx(4.6)
    assert gap.pose_debt == pytest.approx(6.0)
    assert gap.critical_ratio == pytest.approx(6.0 / 4.6)


def test_pose_progress_with_nonregressing_seg_adopts_wseg() -> None:
    verdict = evaluate_bounded_slope_window(
        gap=_gap(),
        start=ObjectiveTerms(seg=2.4, pose=12.0),
        end=ObjectiveTerms(seg=2.3, pose=10.0),
        steps=4,
    )
    assert verdict.decision == "ADOPT_WSEG"
    assert math.isinf(verdict.observed_ratio)


def test_pose_stall_keeps_wjoint() -> None:
    verdict = evaluate_bounded_slope_window(
        gap=_gap(),
        start=ObjectiveTerms(seg=2.4, pose=12.0),
        end=ObjectiveTerms(seg=2.3, pose=12.0),
        steps=4,
    )
    assert verdict.decision == "KEEP_WJOINT"
    assert verdict.reason == "POSE_STALL_OR_REGRESSION"


def test_any_seg_regression_keeps_wjoint() -> None:
    verdict = evaluate_bounded_slope_window(
        gap=_gap(),
        start=ObjectiveTerms(seg=2.4, pose=12.0),
        end=ObjectiveTerms(seg=2.41, pose=9.0),
        steps=4,
    )
    assert verdict.decision == "KEEP_WJOINT"
    assert verdict.reason == "SEG_REGRESSION"
