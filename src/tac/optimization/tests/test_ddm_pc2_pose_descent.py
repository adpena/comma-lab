# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import pytest

from tac.optimization.ddm_pc2_pose_descent import (
    CRITICAL_RATIO,
    bit_reversal_knot_order,
    constant_slope_horizon,
    fork_verdict,
    four_pair_batch_for_knot,
    realized_slope_row,
    score_domain_action,
    select_realized_candidate,
)


def _verdict(d_seg: float, d_pose: float, archive_bytes: int = 1000):
    return {
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "advisory_action": score_domain_action(
            d_seg=d_seg,
            d_pose=d_pose,
            archive_bytes=archive_bytes,
        ),
    }


def test_bit_reversal_and_four_pair_batches_cover_horizon() -> None:
    order = bit_reversal_knot_order(32)
    assert len(order) == len(set(order)) == 32
    assert order[:8] == (0, 16, 8, 24, 4, 20, 12, 28)
    assert four_pair_batch_for_knot(0) == (0, 1, 2, 3)
    assert four_pair_batch_for_knot(31) == (596, 597, 598, 599)
    assert four_pair_batch_for_knot(16) == (308, 309, 310, 311)


def test_score_domain_action_uses_exact_contest_terms() -> None:
    value = score_domain_action(d_seg=0.02, d_pose=4.0, archive_bytes=1000)
    assert value == pytest.approx(2.0 + math.sqrt(40.0) + 25.0 * 1000 / 37_545_489)


def test_candidate_selection_requires_pose_and_joint_descent() -> None:
    rows = [
        {
            "coordinate_id": "seg-only",
            "pose_delta": 0.1,
            "joint_delta": -1.0,
            "seg_delta": -1.1,
            "archive_bytes": 10,
            "receiver_visible": True,
        },
        {
            "coordinate_id": "pose-only",
            "pose_delta": -1.0,
            "joint_delta": 0.2,
            "seg_delta": 1.2,
            "archive_bytes": 10,
            "receiver_visible": True,
        },
        {
            "coordinate_id": "winner",
            "pose_delta": -0.5,
            "joint_delta": -0.3,
            "seg_delta": 0.2,
            "archive_bytes": 11,
            "receiver_visible": True,
        },
    ]
    assert select_realized_candidate(rows)["coordinate_id"] == "winner"
    assert select_realized_candidate(rows[:2]) is None


def test_slope_and_horizon_report_exact_units() -> None:
    start = _verdict(0.02, 4.0)
    end = _verdict(0.019, 3.0)
    row = realized_slope_row(start=start, end=end, accepted_steps=10)
    assert row["d_pose_delta_per_step"] == pytest.approx(-0.1)
    assert row["seg_term_delta_per_step"] == pytest.approx(-0.01)
    assert math.isinf(row["observed_pose_to_seg_regression_ratio"])
    assert row["ratio_clears_critical"] is True
    assert row["critical_ratio"] == CRITICAL_RATIO
    horizon = constant_slope_horizon(
        start_d_pose=4.0,
        end_d_pose=3.0,
        accepted_steps=10,
        target_d_pose=1.0,
    )
    assert horizon["additional_steps_from_window_end"] == pytest.approx(20.0)
    assert horizon["total_steps_from_window_start"] == pytest.approx(30.0)


def test_fork_verdict_is_formulation_scoped_when_joint_does_not_descend() -> None:
    start = _verdict(0.02, 4.0)
    positive_joint = _verdict(0.03, 3.9)
    verdict, scope = fork_verdict(start=start, end=positive_joint)
    assert verdict == "PC1_POSE_DESCENDS_BUT_JOINT_NOT_NEGATIVE_FORMULATION_STOP"
    assert scope.startswith("FORMULATION:")
    good = _verdict(0.019, 3.9)
    assert fork_verdict(start=start, end=good)[0] == ("PC1_DESCENT_MEASURED_NET_JOINT_NEGATIVE")
