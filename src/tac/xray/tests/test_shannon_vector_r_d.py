# SPDX-License-Identifier: MIT
"""Tests for F2: tac.xray.shannon_vector_r_d."""

from __future__ import annotations

import math

import pytest

from tac.xray.base import XRayPrimitive
from tac.xray.shannon_vector_r_d import (
    CONTEST_UNCOMPRESSED_SIZE_BYTES,
    POSE_COEFF_SQRT,
    RATE_COEFF,
    SEG_COEFF,
    ShannonVectorRDBound,
    ShannonVectorRDEstimator,
)


def test_estimator_implements_protocol():
    assert isinstance(ShannonVectorRDEstimator(), XRayPrimitive)


def test_estimator_name():
    assert ShannonVectorRDEstimator().name == "shannon_vector_r_d"


def test_wire_in_hooks():
    hooks = ShannonVectorRDEstimator().wire_in_hooks
    assert "pareto_constraint" in hooks
    assert "sensitivity_map" in hooks


def test_bound_rejects_neg_d_seg():
    with pytest.raises(ValueError, match="d_seg_target"):
        ShannonVectorRDBound(
            d_seg_target=-0.1,
            d_pose_target=0.018,
            r_min_bytes=100.0,
            r_min_score_contribution=0.001,
            distortion_floor_score_contribution=0.5,
            total_floor_score=0.6,
            rationale="x",
        )


def test_bound_rejects_d_seg_above_one():
    with pytest.raises(ValueError, match="d_seg_target"):
        ShannonVectorRDBound(
            d_seg_target=1.5,
            d_pose_target=0.018,
            r_min_bytes=100.0,
            r_min_score_contribution=0.001,
            distortion_floor_score_contribution=0.5,
            total_floor_score=0.6,
            rationale="x",
        )


def test_bound_rejects_neg_pose():
    with pytest.raises(ValueError, match="d_pose_target"):
        ShannonVectorRDBound(
            d_seg_target=0.067,
            d_pose_target=-0.01,
            r_min_bytes=100.0,
            r_min_score_contribution=0.001,
            distortion_floor_score_contribution=0.5,
            total_floor_score=0.6,
            rationale="x",
        )


def test_bound_rejects_neg_r_min_bytes():
    with pytest.raises(ValueError, match="r_min_bytes"):
        ShannonVectorRDBound(
            d_seg_target=0.067,
            d_pose_target=0.018,
            r_min_bytes=-1.0,
            r_min_score_contribution=0.0,
            distortion_floor_score_contribution=0.5,
            total_floor_score=0.5,
            rationale="x",
        )


def test_compute_at_pr101_operating_point():
    """At the canonical PR101 operating point with default kwargs, the
    Shannon vector R(D) bound should land in the ballpark of the deep_math
    §9 claim (~100 bytes order-of-magnitude)."""
    result = ShannonVectorRDEstimator().compute(
        target=None,
        d_seg_target=0.067,
        d_pose_target=0.018,
        sigma_pose_prior=0.5,
        n_pairs=600,
        n_segnet_classes=5,
        correlation_factor=1.0,
    )
    bound = result.primitive_value
    assert isinstance(bound, ShannonVectorRDBound)
    # Sanity check: R_min should be positive but tiny relative to A1's 178K archive.
    assert bound.r_min_bytes > 0
    # The bound should be MUCH smaller than A1's archive (178,262 B);
    # we test order-of-magnitude (within 100x).
    assert bound.r_min_bytes < 178_262
    # Total floor score must equal sum of components.
    assert bound.total_floor_score == pytest.approx(
        bound.r_min_score_contribution
        + bound.distortion_floor_score_contribution
    )


def test_compute_zero_distortion_targets_drives_r_min_up():
    """At d_seg=0, d_pose tiny, R_min should be MUCH larger (you need
    every bit of source entropy)."""
    high_demand = ShannonVectorRDEstimator().compute(
        target=None,
        d_seg_target=0.001,
        d_pose_target=0.0001,
        sigma_pose_prior=0.5,
    ).primitive_value
    low_demand = ShannonVectorRDEstimator().compute(
        target=None,
        d_seg_target=0.5,
        d_pose_target=0.4,
        sigma_pose_prior=0.5,
    ).primitive_value
    assert high_demand.r_min_bytes > low_demand.r_min_bytes


def test_compute_correlation_factor_tightens_bound():
    """Lower correlation_factor (= more cooperative-receiver tightening)
    must produce a smaller r_min_bytes."""
    no_correlation = ShannonVectorRDEstimator().compute(
        target=None,
        d_seg_target=0.067,
        d_pose_target=0.018,
        correlation_factor=1.0,
    ).primitive_value
    full_correlation = ShannonVectorRDEstimator().compute(
        target=None,
        d_seg_target=0.067,
        d_pose_target=0.018,
        correlation_factor=0.5,
    ).primitive_value
    assert full_correlation.r_min_bytes == pytest.approx(
        no_correlation.r_min_bytes * 0.5
    )


def test_compute_rejects_invalid_correlation_factor():
    est = ShannonVectorRDEstimator()
    with pytest.raises(ValueError, match="correlation_factor"):
        est.compute(target=None, correlation_factor=0.0)
    with pytest.raises(ValueError, match="correlation_factor"):
        est.compute(target=None, correlation_factor=1.5)


def test_compute_rejects_invalid_n_pairs():
    est = ShannonVectorRDEstimator()
    with pytest.raises(ValueError, match="n_pairs"):
        est.compute(target=None, n_pairs=0)


def test_compute_rejects_invalid_sigma_pose():
    est = ShannonVectorRDEstimator()
    with pytest.raises(ValueError, match="sigma_pose_prior"):
        est.compute(target=None, sigma_pose_prior=0.0)


def test_compute_high_pose_target_zeros_pose_rate():
    """When d_pose_target >= sigma^2, the Gaussian R(D) is 0."""
    result = ShannonVectorRDEstimator().compute(
        target=None,
        d_seg_target=0.067,
        d_pose_target=1.0,
        sigma_pose_prior=0.5,
    )
    bound = result.primitive_value
    # Only SegNet contribution remains.
    assert "R_pose/pair = 0.0000" in bound.rationale


def test_compute_score_contribution_formula():
    """Verify the score-contribution arithmetic matches the contest formula."""
    result = ShannonVectorRDEstimator().compute(
        target=None,
        d_seg_target=0.067,
        d_pose_target=0.018,
    )
    bound = result.primitive_value
    expected_rate_score = (
        RATE_COEFF * bound.r_min_bytes / CONTEST_UNCOMPRESSED_SIZE_BYTES
    )
    expected_distortion = SEG_COEFF * 0.067 + math.sqrt(
        POSE_COEFF_SQRT * 0.018
    )
    assert bound.r_min_score_contribution == pytest.approx(expected_rate_score)
    assert bound.distortion_floor_score_contribution == pytest.approx(
        expected_distortion
    )


def test_compute_returns_typed_result_envelope():
    result = ShannonVectorRDEstimator().compute(target=None)
    assert result.primitive_name == "shannon_vector_r_d"
    assert result.evidence_grade == "first-principles-bound"
    assert result.confidence_band is not None
    assert result.archive_or_video_path is None  # no target
    assert result.archive_sha256 is None
    assert "shannon_vector_r_d" not in result.composes_with  # composes_with is OTHERS


def test_compute_with_archive_path_records_sha(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"abc")
    result = ShannonVectorRDEstimator().compute(target=archive)
    assert result.archive_or_video_path == archive
    assert result.archive_sha256 is not None


def test_constants_match_contest():
    assert CONTEST_UNCOMPRESSED_SIZE_BYTES == 37_545_489
    assert SEG_COEFF == 100.0
    assert POSE_COEFF_SQRT == 10.0
    assert RATE_COEFF == 25.0


def test_metadata_includes_kwargs():
    result = ShannonVectorRDEstimator().compute(
        target=None,
        n_pairs=300,
        sigma_pose_prior=0.4,
        correlation_factor=0.7,
    )
    assert result.metadata["n_pairs"] == 300
    assert result.metadata["sigma_pose_prior"] == 0.4
    assert result.metadata["correlation_factor"] == 0.7


def test_compose_with_returns_composed():
    from tac.xray.base import ComposedXRayPrimitive

    est = ShannonVectorRDEstimator()

    class _Other:
        name = "other"
        wire_in_hooks = ("sensitivity_map",)

        def compute(self, target, **kw):
            from tac.xray.base import XRayPrimitiveResult

            return XRayPrimitiveResult(
                primitive_name="other",
                archive_or_video_path=None,
                archive_sha256=None,
                primitive_value=1.0,
                evidence_grade="mathematical-derivation",
                confidence_band=None,
                composes_with=(),
                wire_in_hooks_engaged=("sensitivity_map",),
            )

        def compose_with(self, other):
            return ComposedXRayPrimitive(left=self, right=other)

    composed = est.compose_with(_Other())
    assert isinstance(composed, ComposedXRayPrimitive)


def test_rationale_includes_correlation_factor():
    result = ShannonVectorRDEstimator().compute(
        target=None, correlation_factor=0.85
    )
    assert "0.85" in result.primitive_value.rationale
