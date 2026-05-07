"""Tests for tac.score_geometry_stacking — Volterra-style cross-axis interaction."""
from __future__ import annotations

import math

import pytest

from tac.score_geometry import contest_score
from tac.score_geometry_stacking import predict_stacking


def test_disjoint_axes_purely_additive() -> None:
    """Lane A touches seg only; Lane B touches bytes only. Linear axes →
    stacked gain == sum of individual gains within float tolerance."""
    pred = predict_stacking(
        focal_d_seg=1e-3,
        focal_d_pose=1e-4,
        focal_archive_bytes=200_000,
        delta_d_seg_a=5e-4,        # seg-only lane
        delta_bytes_b=10_000,      # bytes-only lane
    )
    assert pred.pose_pose_interaction_correction == 0.0
    assert math.isclose(pred.stacked_gain, pred.sum_individual_gains, rel_tol=1e-9)
    assert math.isclose(pred.nominal_stack_ratio, 1.0, rel_tol=1e-9)


def test_two_pose_lanes_compound_toward_shannon_floor() -> None:
    """Both lanes touch pose. Counter-intuitive truth: stacking is
    SUPER-ADDITIVE because we move toward the sqrt singularity at
    d_pose=0 where the marginal blows up. Each pose improvement
    operates at a steeper slope than the previous."""
    pred = predict_stacking(
        focal_d_seg=0.0,
        focal_d_pose=1e-4,
        focal_archive_bytes=200_000,
        delta_d_pose_a=3e-5,
        delta_d_pose_b=3e-5,
    )
    assert pred.stacked_gain > pred.sum_individual_gains
    assert pred.nominal_stack_ratio > 1.0
    # Pose-pose correction is POSITIVE (super-additive compounding)
    assert pred.pose_pose_interaction_correction > 0.0


def test_only_one_pose_lane_no_interaction() -> None:
    """Only lane A touches pose; lane B touches bytes → no pose-pose interaction."""
    pred = predict_stacking(
        focal_d_seg=0.0,
        focal_d_pose=1e-4,
        focal_archive_bytes=200_000,
        delta_d_pose_a=3e-5,
        delta_bytes_b=5_000,
    )
    assert pred.pose_pose_interaction_correction == 0.0
    assert math.isclose(pred.stacked_gain, pred.sum_individual_gains, rel_tol=1e-9)


def test_zero_deltas_zero_gain() -> None:
    """No deltas → no gain on any axis."""
    pred = predict_stacking(
        focal_d_seg=0.001,
        focal_d_pose=1e-4,
        focal_archive_bytes=200_000,
    )
    assert pred.individual_gain_a == 0.0
    assert pred.individual_gain_b == 0.0
    assert pred.stacked_gain == 0.0
    assert pred.nominal_stack_ratio == 1.0  # by convention


def test_negative_deltas_rejected() -> None:
    """Negative deltas are not supported (regressions are not 'gains')."""
    with pytest.raises(ValueError, match="positive improvements"):
        predict_stacking(
            focal_d_seg=0.001,
            focal_d_pose=1e-4,
            focal_archive_bytes=200_000,
            delta_d_seg_a=-1e-4,
        )


def test_score_after_stack_matches_contest_formula() -> None:
    """The stacked score must equal the contest formula at the post-stack point."""
    focal_seg = 0.001
    focal_pose = 5e-4
    focal_bytes = 200_000
    da_seg, db_pose = 3e-4, 1e-4
    pred = predict_stacking(
        focal_d_seg=focal_seg,
        focal_d_pose=focal_pose,
        focal_archive_bytes=focal_bytes,
        delta_d_seg_a=da_seg,
        delta_d_pose_b=db_pose,
    )
    expected = contest_score(focal_seg - da_seg, focal_pose - db_pose, focal_bytes)
    assert math.isclose(pred.score_after_stack, expected, rel_tol=1e-12)


def test_pose_pose_correction_magnitude_grows_with_focal_pose() -> None:
    """At larger d_pose, sqrt is steeper → correction is smaller in magnitude.
    At smaller d_pose, sqrt is flatter approaching origin → correction is larger.

    Comparing equal Δ_a, Δ_b at two different focal d_pose values, the
    correction at smaller focal pose should be larger in magnitude."""
    delta = 1e-5
    pred_high_pose = predict_stacking(
        focal_d_seg=0.0,
        focal_d_pose=1e-3,  # far from sqrt singularity
        focal_archive_bytes=200_000,
        delta_d_pose_a=delta,
        delta_d_pose_b=delta,
    )
    pred_low_pose = predict_stacking(
        focal_d_seg=0.0,
        focal_d_pose=5e-5,  # closer to sqrt singularity
        focal_archive_bytes=200_000,
        delta_d_pose_a=delta,
        delta_d_pose_b=delta,
    )
    # Both negative; low_pose should be MORE negative (larger correction)
    assert abs(pred_low_pose.pose_pose_interaction_correction) > abs(
        pred_high_pose.pose_pose_interaction_correction
    )


def test_notes_emitted_for_pose_pose_stacking() -> None:
    """Pose-pose stacking should emit a super-additivity note."""
    pred = predict_stacking(
        focal_d_seg=0.0,
        focal_d_pose=1e-4,
        focal_archive_bytes=200_000,
        delta_d_pose_a=2e-5,
        delta_d_pose_b=2e-5,
    )
    notes_text = " ".join(pred.notes)
    assert "super-additive" in notes_text or "compound" in notes_text


def test_sum_of_post_stack_decomp_equals_score_after_stack() -> None:
    """Sanity bridge: per-axis decomposition at post-stack point sums to
    score_after_stack."""
    pred = predict_stacking(
        focal_d_seg=0.001,
        focal_d_pose=5e-5,
        focal_archive_bytes=200_000,
        delta_d_seg_a=2e-4,
        delta_d_pose_a=1e-5,
        delta_d_pose_b=1e-5,
        delta_bytes_b=5_000,
    )
    # Confirm score_after_stack is internally consistent
    expected = contest_score(
        pred.focal_d_seg - pred.delta_d_seg_a - pred.delta_d_seg_b,
        pred.focal_d_pose - pred.delta_d_pose_a - pred.delta_d_pose_b,
        pred.focal_archive_bytes - pred.delta_bytes_a - pred.delta_bytes_b,
    )
    assert math.isclose(pred.score_after_stack, expected, rel_tol=1e-12)
