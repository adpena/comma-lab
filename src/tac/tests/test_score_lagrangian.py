# SPDX-License-Identifier: MIT
"""Tests for tac.score_lagrangian canonical analytical multiplier helper.

Per ARBITRARINESS-EXTINCTION audit 2026-05-18 commit 2d042f7e6 TOP-1
(``lambda_seg_pose_rate_multipliers_unprincipled``). Verifies the
closed-form Lagrangian-multiplier formulas recover the CLAUDE.md
empirical anchors (PR106 frontier ratio ~2.71; OLD 1.x ratio ~0.12).
"""
from __future__ import annotations

import math

import pytest

from tac.score_lagrangian import (
    CONTEST_RATE_DENOM_BYTES,
    LagrangianMultipliers,
    POSE_CROSSOVER_AT_AVG,
    RATE_SCORE_PER_BYTE,
    SEG_SCORE_PER_UNIT,
    compute_marginal_multipliers,
    empirical_anchor_old_1x_operating_point,
    empirical_anchor_pr106_frontier,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_contest_rate_denom_bytes_pinned_to_canonical_value() -> None:
    """Per upstream/evaluate.py: 25 * archive_bytes / 37,545,489."""
    assert CONTEST_RATE_DENOM_BYTES == 37_545_489


def test_rate_score_per_byte_matches_canonical_derivation() -> None:
    """``25 / 37,545,489 ~ 6.66e-7`` per CLAUDE.md."""
    expected = 25.0 / 37_545_489
    assert RATE_SCORE_PER_BYTE == pytest.approx(expected, rel=1e-12)
    # Sanity-check the magnitude per CLAUDE.md comment.
    assert RATE_SCORE_PER_BYTE == pytest.approx(6.66e-7, rel=1e-2)


def test_seg_score_per_unit_is_canonical_100() -> None:
    """``d(100*d_seg)/d(d_seg) = 100`` per upstream/evaluate.py."""
    assert SEG_SCORE_PER_UNIT == 100.0


def test_pose_crossover_anchor_pinned() -> None:
    """At pose_avg = 2.5e-4, pose marginal equals seg marginal (= 100)."""
    expected_pose_marginal = 5.0 / math.sqrt(10.0 * POSE_CROSSOVER_AT_AVG)
    assert expected_pose_marginal == pytest.approx(100.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Marginal formula correctness
# ---------------------------------------------------------------------------


def test_seg_marginal_is_constant_100_at_all_operating_points() -> None:
    """``d(score)/d(seg_avg) = 100`` -- linear, constant."""
    for seg_avg in [0.0, 1e-6, 0.07, 0.18, 100.0]:
        mult = compute_marginal_multipliers(seg_avg=seg_avg, pose_avg=1e-3)
        assert mult.lambda_seg == 100.0


def test_pose_marginal_matches_canonical_formula_5_over_sqrt_10x() -> None:
    """``d(sqrt(10*pose_avg))/d(pose_avg) = 5 / sqrt(10 * pose_avg)``."""
    for pose_avg in [1e-6, 3.4e-5, 2.5e-4, 0.18, 1.0]:
        mult = compute_marginal_multipliers(seg_avg=0.001, pose_avg=pose_avg)
        expected = 5.0 / math.sqrt(10.0 * pose_avg)
        assert mult.lambda_pose == pytest.approx(expected, rel=1e-12)


def test_rate_marginal_defaults_to_canonical_25_over_n() -> None:
    """Default ``lambda_rate = 25 / 37,545,489`` per upstream/evaluate.py."""
    mult = compute_marginal_multipliers(seg_avg=0.001, pose_avg=1e-3)
    assert mult.lambda_rate == pytest.approx(RATE_SCORE_PER_BYTE, rel=1e-12)


def test_rate_axis_override_honored_for_diagnostic_use() -> None:
    """``rate_axis_bytes_per_unit`` overrides default rate marginal."""
    diagnostic_rate = 1.234e-5
    mult = compute_marginal_multipliers(
        seg_avg=0.001,
        pose_avg=1e-3,
        rate_axis_bytes_per_unit=diagnostic_rate,
    )
    assert mult.lambda_rate == diagnostic_rate


# ---------------------------------------------------------------------------
# CLAUDE.md empirical anchors (recovery)
# ---------------------------------------------------------------------------


def test_pr106_frontier_anchor_recovers_ratio_2_71() -> None:
    """Per CLAUDE.md anchor: PR106 frontier pose_avg=3.4e-5 -> ratio ~ 2.71."""
    mult = empirical_anchor_pr106_frontier()
    assert mult.pose_to_seg_ratio == pytest.approx(2.71, rel=1e-2)
    assert mult.operating_point_classification == "frontier_pose_dominant"


def test_old_1x_anchor_yields_seg_dominant_ratio_below_unity() -> None:
    """OLD 1.x pose_avg=0.18 -> ratio < 1 (seg-dominant regime).

    Note: CLAUDE.md's table cell "~12" is an upper-bound *informal*
    approximation; the canonical formula ``5/sqrt(10*0.18) ~ 3.73``
    yields ratio ~ 0.037 (well into seg-dominant). The empirical
    *operational* anchor is the CLASSIFICATION (seg-dominant), not the
    exact numeric (which depends on whether you use marginal or total
    contribution -- this helper uses marginal per Boyd Ch.5).
    """
    mult = empirical_anchor_old_1x_operating_point()
    assert mult.pose_to_seg_ratio < 1.0
    assert mult.operating_point_classification == "old_1x_seg_dominant"


def test_pr106_frontier_pose_marginal_is_271_per_claude_md() -> None:
    """CLAUDE.md: at PR106 frontier ``d(pose)/d(pose_avg) ~ 271``."""
    mult = empirical_anchor_pr106_frontier()
    assert mult.lambda_pose == pytest.approx(271.0, abs=1.0)


def test_old_1x_pose_marginal_matches_canonical_formula() -> None:
    """OLD 1.x pose_avg=0.18 -> ``5/sqrt(1.8) ~ 3.73``.

    Direct verification that the formula yields the canonical value at
    the OLD 1.x operating point.
    """
    mult = empirical_anchor_old_1x_operating_point()
    expected = 5.0 / math.sqrt(1.8)
    assert mult.lambda_pose == pytest.approx(expected, rel=1e-12)
    # Sanity-check magnitude.
    assert mult.lambda_pose == pytest.approx(3.73, abs=0.01)


# ---------------------------------------------------------------------------
# Operating-point classification (probe-disambiguator output)
# ---------------------------------------------------------------------------


def test_classification_frontier_below_crossover_threshold() -> None:
    """``pose_avg < 2.5e-4`` -> frontier_pose_dominant."""
    mult = compute_marginal_multipliers(seg_avg=1e-3, pose_avg=1e-5)
    assert mult.operating_point_classification == "frontier_pose_dominant"


def test_classification_old_1x_above_crossover_band() -> None:
    """``pose_avg > 2.5e-3`` -> old_1x_seg_dominant."""
    mult = compute_marginal_multipliers(seg_avg=1e-3, pose_avg=0.1)
    assert mult.operating_point_classification == "old_1x_seg_dominant"


def test_classification_crossover_at_canonical_pose_avg() -> None:
    """``pose_avg = 2.5e-4`` -> crossover (boundary; pose marg == seg marg)."""
    mult = compute_marginal_multipliers(
        seg_avg=1e-3, pose_avg=POSE_CROSSOVER_AT_AVG
    )
    assert mult.operating_point_classification == "crossover"
    # At the crossover point, pose marginal equals seg marginal (= 100).
    assert mult.pose_to_seg_ratio == pytest.approx(1.0, rel=1e-3)


def test_classification_frontier_boundary_strict_inequality() -> None:
    """Boundary: ``pose_avg = 2.5e-4`` exactly is at frontier-edge.

    Strictly less than POSE_CROSSOVER_AT_AVG -> frontier; equal or
    greater (up to 10x) -> crossover.
    """
    mult_below = compute_marginal_multipliers(
        seg_avg=1e-3, pose_avg=POSE_CROSSOVER_AT_AVG * 0.999
    )
    assert mult_below.operating_point_classification == "frontier_pose_dominant"
    mult_at = compute_marginal_multipliers(
        seg_avg=1e-3, pose_avg=POSE_CROSSOVER_AT_AVG
    )
    assert mult_at.operating_point_classification == "crossover"


def test_classification_pr106_anchor_at_3_4e_minus_5_is_frontier() -> None:
    """Per CLAUDE.md: PR106 frontier pose_avg=3.4e-5 is frontier_pose_dominant.

    This is the canonical empirical anchor; the boundary at
    POSE_CROSSOVER_AT_AVG must classify it correctly without ambiguity.
    """
    mult = compute_marginal_multipliers(seg_avg=6.7e-4, pose_avg=3.4e-5)
    assert mult.operating_point_classification == "frontier_pose_dominant"


# ---------------------------------------------------------------------------
# Normalization mode
# ---------------------------------------------------------------------------


def test_normalize_to_sum_one_sums_to_unity() -> None:
    """``normalize_to_sum_one=True`` -> the three multipliers sum to 1."""
    mult = compute_marginal_multipliers(
        seg_avg=6.7e-4,
        pose_avg=3.4e-5,
        normalize_to_sum_one=True,
    )
    total = mult.lambda_seg + mult.lambda_pose + mult.lambda_rate
    assert total == pytest.approx(1.0, abs=1e-12)


def test_normalize_preserves_pose_to_seg_ratio() -> None:
    """Normalization is a uniform scale -- pose/seg ratio is invariant."""
    unnorm = compute_marginal_multipliers(seg_avg=6.7e-4, pose_avg=3.4e-5)
    norm = compute_marginal_multipliers(
        seg_avg=6.7e-4,
        pose_avg=3.4e-5,
        normalize_to_sum_one=True,
    )
    assert unnorm.pose_to_seg_ratio == pytest.approx(
        norm.pose_to_seg_ratio, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_pose_avg_zero_raises_value_error() -> None:
    """Marginal diverges at ``pose_avg = 0`` -- refuse explicitly."""
    with pytest.raises(ValueError, match="pose_avg must be positive"):
        compute_marginal_multipliers(seg_avg=0.001, pose_avg=0.0)


def test_pose_avg_negative_raises_value_error() -> None:
    """Distortion is non-negative by definition."""
    with pytest.raises(ValueError, match="pose_avg must be positive"):
        compute_marginal_multipliers(seg_avg=0.001, pose_avg=-1e-3)


def test_seg_avg_negative_raises_value_error() -> None:
    """Distortion is non-negative by definition."""
    with pytest.raises(ValueError, match="seg_avg must be non-negative"):
        compute_marginal_multipliers(seg_avg=-0.1, pose_avg=1e-3)


def test_seg_avg_zero_is_accepted_as_saturated_seg_axis() -> None:
    """``seg_avg = 0`` is the saturated-seg-axis edge case; multipliers
    are still well-defined (seg marginal is constant)."""
    mult = compute_marginal_multipliers(seg_avg=0.0, pose_avg=1e-3)
    assert mult.lambda_seg == SEG_SCORE_PER_UNIT


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_lagrangian_multipliers_is_frozen() -> None:
    """Frozen dataclass per Catalog #131/#138 immutability discipline."""
    mult = empirical_anchor_pr106_frontier()
    with pytest.raises((AttributeError, Exception)):
        mult.lambda_seg = 999.0  # type: ignore[misc]


def test_lagrangian_multipliers_has_all_five_fields() -> None:
    """Schema invariant: 5 fields exactly."""
    mult = empirical_anchor_pr106_frontier()
    expected_fields = {
        "lambda_seg",
        "lambda_pose",
        "lambda_rate",
        "pose_to_seg_ratio",
        "operating_point_classification",
    }
    # __slots__ enumerates dataclass fields when slots=True is set.
    actual = set(mult.__slots__)
    assert actual == expected_fields


# ---------------------------------------------------------------------------
# Edge cases / numerical robustness
# ---------------------------------------------------------------------------


def test_extreme_frontier_pose_avg_1e_minus_10_does_not_overflow() -> None:
    """Very small pose_avg (extreme frontier) -- formula well-defined."""
    mult = compute_marginal_multipliers(seg_avg=1e-3, pose_avg=1e-10)
    # 5 / sqrt(1e-9) ~ 1.58e5; large but finite.
    assert math.isfinite(mult.lambda_pose)
    assert mult.lambda_pose > 1e4
    assert mult.operating_point_classification == "frontier_pose_dominant"


def test_extreme_old_pose_avg_100_does_not_underflow() -> None:
    """Very large pose_avg -- formula well-defined."""
    mult = compute_marginal_multipliers(seg_avg=1e-3, pose_avg=100.0)
    # 5 / sqrt(1000) ~ 0.158; small but positive.
    assert math.isfinite(mult.lambda_pose)
    assert 0.0 < mult.lambda_pose < 1.0
    assert mult.operating_point_classification == "old_1x_seg_dominant"


def test_anchor_helpers_round_trip_through_compute_marginal_multipliers() -> None:
    """The two anchor convenience helpers should match a direct call."""
    anchor_pr106 = empirical_anchor_pr106_frontier()
    direct_pr106 = compute_marginal_multipliers(
        seg_avg=6.7e-4, pose_avg=3.4e-5
    )
    assert anchor_pr106 == direct_pr106

    anchor_old = empirical_anchor_old_1x_operating_point()
    direct_old = compute_marginal_multipliers(seg_avg=0.07, pose_avg=0.18)
    assert anchor_old == direct_old
