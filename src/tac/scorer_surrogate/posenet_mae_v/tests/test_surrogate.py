# SPDX-License-Identifier: MIT
"""Tests for PoseNet MAE-V numpy-portable surrogate.

Slot EEE NO FAKE IMPLEMENTATIONS gate coverage:
  - REAL numpy forward (not stub) verified vs hand-computed reference
  - Distinct-output invariant (different weights -> different outputs)
  - Per-byte FD Jacobian produces finite nonzero estimate on real decoder
  - Forward parity helper computes a real max-abs drift
  - PoseJacobianResult canonical-routing markers ALL frozen False per
    Catalog #341 (cannot be promoted by construction)
  - Canonical Provenance per Catalog #323 in every result
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.scorer_surrogate.posenet_mae_v import (
    CANONICAL_POSE_DIMS,
    CANONICAL_POSE_POOL_GRID,
    PARITY_MAX_ABS_CANONICAL_THRESHOLD,
    PoseJacobianResult,
    PoseNetMaeVSurrogate,
    PoseNetMaeVSurrogateInvalidError,
    build_canonical_provenance_for_surrogate,
    build_surrogate_from_numpy_weights,
    compute_forward_parity_max_abs,
    compute_per_byte_pose_jacobian,
)


def _make_canonical_surrogate(seed: int = 0) -> PoseNetMaeVSurrogate:
    rng = np.random.default_rng(seed)
    feature_dim = 2 * CANONICAL_POSE_POOL_GRID * CANONICAL_POSE_POOL_GRID * 3
    w = rng.standard_normal((feature_dim, CANONICAL_POSE_DIMS)).astype(np.float32) * 0.05
    b = rng.standard_normal(CANONICAL_POSE_DIMS).astype(np.float32) * 0.025
    return build_surrogate_from_numpy_weights(w, b)


def test_canonical_constants_pinned() -> None:
    assert CANONICAL_POSE_DIMS == 6
    assert CANONICAL_POSE_POOL_GRID == 4
    assert PARITY_MAX_ABS_CANONICAL_THRESHOLD == 3e-5


def test_construct_canonical_surrogate_real_weights() -> None:
    surr = _make_canonical_surrogate()
    assert surr.pose_dims == 6
    assert surr.pool_grid == 4
    assert surr.feature_dim == 96
    assert surr.total_params == 96 * 6 + 6  # = 582
    assert surr.weight.shape == (96, 6)
    assert surr.bias.shape == (6,)
    assert len(surr.weight_sha256) == 64
    assert all(c in "0123456789abcdef" for c in surr.weight_sha256)


def test_distinct_weights_distinct_sha(seed_a: int = 1, seed_b: int = 999) -> None:
    surr_a = _make_canonical_surrogate(seed=seed_a)
    surr_b = _make_canonical_surrogate(seed=seed_b)
    assert surr_a.weight_sha256 != surr_b.weight_sha256


def test_forward_real_computation_real_shape() -> None:
    """Slot EEE NO FAKE: forward materializes actual matmul + nonzero output."""
    surr = _make_canonical_surrogate()
    rng = np.random.default_rng(7)
    rgb_0 = rng.random((2, 32, 32, 3)).astype(np.float32)
    rgb_1 = rng.random((2, 32, 32, 3)).astype(np.float32)
    pose = surr.forward(rgb_0, rgb_1)
    assert pose.shape == (2, 6)
    # Not all-zero
    assert np.any(pose != 0.0)


def test_forward_matches_hand_computed_reference() -> None:
    """Verify forward replicates pool->concat->matmul exactly."""
    # Constant-weight surrogate so hand-computation is tractable
    feature_dim = 96
    # weight maps feature -> first dim only, identity on other features
    w = np.zeros((feature_dim, CANONICAL_POSE_DIMS), dtype=np.float32)
    w[0, 0] = 1.0  # only feature[0] (which is pooled-frame0-pixel-(0,0,R)) -> pose[0]
    bias = np.zeros(CANONICAL_POSE_DIMS, dtype=np.float32)
    surr = build_surrogate_from_numpy_weights(w, bias)
    # Construct an input where pooled f0[(0,0,R)] = exactly 0.5
    rgb_0 = np.zeros((1, 4, 4, 3), dtype=np.float32)  # pool_grid=4 means 1x1 blocks
    rgb_0[0, 0, 0, 0] = 0.5  # pixel (0, 0) red = 0.5
    rgb_1 = np.zeros((1, 4, 4, 3), dtype=np.float32)
    pose = surr.forward(rgb_0, rgb_1)
    # 4x4 input with grid=4 means each pool block is 1x1 -> pool[0,0,R] = 0.5
    # feature[0] = 0.5 (first frame pooled[0,0,R] flattened)
    # weight[0,0]=1.0, bias=0 -> pose[0] = 0.5
    assert np.isclose(pose[0, 0], 0.5, atol=1e-7), f"got pose[0,0]={pose[0, 0]}"
    # All other pose dims should be 0 (no weights)
    assert np.allclose(pose[0, 1:], 0.0)


def test_forward_rejects_shape_mismatch() -> None:
    surr = _make_canonical_surrogate()
    rgb_0 = np.zeros((1, 32, 32, 3), dtype=np.float32)
    rgb_1 = np.zeros((1, 32, 32, 4), dtype=np.float32)  # wrong last dim
    with pytest.raises(PoseNetMaeVSurrogateInvalidError):
        surr.forward(rgb_0, rgb_1)


def test_forward_rejects_h_not_divisible_by_grid() -> None:
    surr = _make_canonical_surrogate()
    rgb_0 = np.zeros((1, 33, 32, 3), dtype=np.float32)  # 33 not divisible by 4
    rgb_1 = np.zeros((1, 33, 32, 3), dtype=np.float32)
    with pytest.raises(PoseNetMaeVSurrogateInvalidError):
        surr.forward(rgb_0, rgb_1)


def test_post_init_rejects_wrong_weight_shape() -> None:
    bad_w = np.zeros((50, 6), dtype=np.float32)  # 50 != 96
    bad_b = np.zeros(6, dtype=np.float32)
    with pytest.raises(PoseNetMaeVSurrogateInvalidError, match="weight shape"):
        build_surrogate_from_numpy_weights(bad_w, bad_b)


def test_post_init_rejects_wrong_bias_shape() -> None:
    w = np.zeros((96, 6), dtype=np.float32)
    bad_b = np.zeros(5, dtype=np.float32)  # 5 != 6
    with pytest.raises(PoseNetMaeVSurrogateInvalidError, match="bias shape"):
        build_surrogate_from_numpy_weights(w, bad_b)


def test_post_init_rejects_invalid_pose_dims() -> None:
    bad_w = np.zeros((96, 0), dtype=np.float32)
    bad_b = np.zeros(0, dtype=np.float32)
    with pytest.raises(PoseNetMaeVSurrogateInvalidError, match="pose_dims"):
        build_surrogate_from_numpy_weights(bad_w, bad_b, pose_dims=0)


def test_pose_jacobian_result_rejects_promotable_true() -> None:
    """Catalog #341 frozen-False invariants prevent promotion-by-construction."""
    with pytest.raises(ValueError, match="promotable MUST be False"):
        PoseJacobianResult(
            per_byte_pose_jacobian_magnitude=(0.0,),
            n_bytes_probed=1,
            surrogate_weight_sha256="a" * 64,
            axis_tag="[macOS-CPU advisory]",
            score_claim=False,
            promotable=True,  # forbidden
            evidence_grade="predicted",
            measurement_utc="2026-05-30T00:00:00Z",
        )


def test_pose_jacobian_result_rejects_score_claim_true() -> None:
    with pytest.raises(ValueError, match="score_claim MUST be False"):
        PoseJacobianResult(
            per_byte_pose_jacobian_magnitude=(0.0,),
            n_bytes_probed=1,
            surrogate_weight_sha256="a" * 64,
            axis_tag="[macOS-CPU advisory]",
            score_claim=True,  # forbidden
            promotable=False,
            evidence_grade="predicted",
            measurement_utc="2026-05-30T00:00:00Z",
        )


def test_pose_jacobian_result_rejects_wrong_axis_tag() -> None:
    with pytest.raises(ValueError, match="axis_tag"):
        PoseJacobianResult(
            per_byte_pose_jacobian_magnitude=(0.0,),
            n_bytes_probed=1,
            surrogate_weight_sha256="a" * 64,
            axis_tag="[contest-CUDA]",  # forbidden — surrogate can never claim this
            score_claim=False,
            promotable=False,
            evidence_grade="predicted",
            measurement_utc="2026-05-30T00:00:00Z",
        )


def test_pose_jacobian_result_rejects_negative_magnitudes() -> None:
    with pytest.raises(ValueError, match="all-nonneg"):
        PoseJacobianResult(
            per_byte_pose_jacobian_magnitude=(-0.1,),
            n_bytes_probed=1,
            surrogate_weight_sha256="a" * 64,
            axis_tag="[macOS-CPU advisory]",
            score_claim=False,
            promotable=False,
            evidence_grade="predicted",
            measurement_utc="2026-05-30T00:00:00Z",
        )


def test_pose_jacobian_result_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length must equal"):
        PoseJacobianResult(
            per_byte_pose_jacobian_magnitude=(0.1, 0.2),
            n_bytes_probed=5,  # mismatch with magnitudes len=2
            surrogate_weight_sha256="a" * 64,
            axis_tag="[macOS-CPU advisory]",
            score_claim=False,
            promotable=False,
            evidence_grade="predicted",
            measurement_utc="2026-05-30T00:00:00Z",
        )


def test_pose_jacobian_result_rejects_invalid_sha256() -> None:
    with pytest.raises(ValueError, match="surrogate_weight_sha256"):
        PoseJacobianResult(
            per_byte_pose_jacobian_magnitude=(0.1,),
            n_bytes_probed=1,
            surrogate_weight_sha256="not-a-sha",
            axis_tag="[macOS-CPU advisory]",
            score_claim=False,
            promotable=False,
            evidence_grade="predicted",
            measurement_utc="2026-05-30T00:00:00Z",
        )


def test_per_byte_jacobian_real_decoder_real_signal() -> None:
    """End-to-end per-byte Jacobian with a synthetic but real decoder."""
    surr = _make_canonical_surrogate(seed=42)
    # Synthetic decoder: take first 192 bytes as frame_0 (8x8x3 uint8 -> /255 -> 4x4x3 after pool, but pool_grid=4 needs HxW>=4)
    # Decoder maps bytes -> (1, 8, 8, 3) -> normalized to [0, 1]
    def decoder(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
        first = np.frombuffer(buf[:192], dtype=np.uint8).astype(np.float32) / 255.0
        second = np.frombuffer(buf[192:384], dtype=np.uint8).astype(np.float32) / 255.0
        return (
            first.reshape(1, 8, 8, 3),
            second.reshape(1, 8, 8, 3),
        )

    # Synthetic archive
    rng = np.random.default_rng(100)
    archive = bytes(rng.integers(0, 255, 384, dtype=np.uint8).tolist())
    result = compute_per_byte_pose_jacobian(
        surrogate=surr,
        archive_bytes=archive,
        decoder=decoder,
        byte_indices=tuple(range(0, 384, 32)),  # probe 12 bytes
        perturbation_magnitude=10.0,
    )
    assert result.n_bytes_probed == 12
    assert len(result.per_byte_pose_jacobian_magnitude) == 12
    # All canonical markers
    assert result.score_claim is False
    assert result.promotable is False
    assert result.axis_tag == "[macOS-CPU advisory]"
    assert result.evidence_grade == "predicted"
    # Canonical Provenance present + structurally complete
    assert "schema_version" in result.provenance
    assert result.provenance["score_claim"] is False
    # At least SOME bytes produced nonzero Jacobian (signal exists)
    nonzero = [v for v in result.per_byte_pose_jacobian_magnitude if v > 0.0]
    assert len(nonzero) > 0, "expected at least one nonzero per-byte Jacobian"


def test_per_byte_jacobian_with_teacher_pose() -> None:
    """Teacher-pose mode: Jacobian targets MSE vs teacher (canonical)."""
    surr = _make_canonical_surrogate(seed=42)

    def decoder(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
        first = np.frombuffer(buf[:192], dtype=np.uint8).astype(np.float32) / 255.0
        second = np.frombuffer(buf[192:384], dtype=np.uint8).astype(np.float32) / 255.0
        return (first.reshape(1, 8, 8, 3), second.reshape(1, 8, 8, 3))

    rng = np.random.default_rng(100)
    archive = bytes(rng.integers(0, 255, 384, dtype=np.uint8).tolist())
    teacher_pose = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)
    result = compute_per_byte_pose_jacobian(
        surrogate=surr,
        archive_bytes=archive,
        decoder=decoder,
        byte_indices=(0, 10, 100, 200),
        perturbation_magnitude=5.0,
        teacher_pose=teacher_pose,
    )
    assert result.n_bytes_probed == 4
    assert all(v >= 0.0 for v in result.per_byte_pose_jacobian_magnitude)


def test_per_byte_jacobian_rejects_empty_archive() -> None:
    surr = _make_canonical_surrogate()
    with pytest.raises(ValueError, match="archive_bytes"):
        compute_per_byte_pose_jacobian(
            surrogate=surr,
            archive_bytes=b"",
            decoder=lambda b: (np.zeros((1, 4, 4, 3)), np.zeros((1, 4, 4, 3))),
        )


def test_per_byte_jacobian_rejects_zero_perturbation() -> None:
    surr = _make_canonical_surrogate()
    with pytest.raises(ValueError, match="perturbation"):
        compute_per_byte_pose_jacobian(
            surrogate=surr,
            archive_bytes=b"\x00" * 384,
            decoder=lambda b: (np.zeros((1, 4, 4, 3)), np.zeros((1, 4, 4, 3))),
            perturbation_magnitude=0.0,
        )


def test_per_byte_jacobian_rejects_out_of_range_index() -> None:
    surr = _make_canonical_surrogate()
    with pytest.raises(ValueError, match="out of range"):
        compute_per_byte_pose_jacobian(
            surrogate=surr,
            archive_bytes=b"\x00" * 100,
            decoder=lambda b: (np.zeros((1, 4, 4, 3)), np.zeros((1, 4, 4, 3))),
            byte_indices=(0, 50, 200),  # 200 out of range
        )


def test_per_byte_jacobian_rejects_teacher_pose_shape_mismatch() -> None:
    surr = _make_canonical_surrogate()

    def decoder(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
        return (np.zeros((1, 4, 4, 3), dtype=np.float32), np.zeros((1, 4, 4, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="teacher_pose"):
        compute_per_byte_pose_jacobian(
            surrogate=surr,
            archive_bytes=b"\x00" * 100,
            decoder=decoder,
            byte_indices=(0,),
            teacher_pose=np.zeros((2, 7), dtype=np.float32),  # (2, 7) != (1, 6)
        )


def test_forward_parity_helper() -> None:
    """compute_forward_parity_max_abs returns a real numerical drift."""
    surr = _make_canonical_surrogate(seed=42)
    # Use the surrogate itself as canonical (drift = 0 for identical forward)
    rng = np.random.default_rng(11)
    rgb_0 = rng.random((1, 16, 16, 3)).astype(np.float32)
    rgb_1 = rng.random((1, 16, 16, 3)).astype(np.float32)
    drift = compute_forward_parity_max_abs(
        surrogate=surr,
        canonical_forward=surr.forward,
        rgb_0_bhwc=rgb_0,
        rgb_1_bhwc=rgb_1,
    )
    assert drift == 0.0


def test_forward_parity_detects_real_drift() -> None:
    """A different canonical_forward produces a real drift."""
    surr_a = _make_canonical_surrogate(seed=1)
    surr_b = _make_canonical_surrogate(seed=999)
    rng = np.random.default_rng(11)
    rgb_0 = rng.random((1, 16, 16, 3)).astype(np.float32)
    rgb_1 = rng.random((1, 16, 16, 3)).astype(np.float32)
    drift = compute_forward_parity_max_abs(
        surrogate=surr_a,
        canonical_forward=surr_b.forward,
        rgb_0_bhwc=rgb_0,
        rgb_1_bhwc=rgb_1,
    )
    assert drift > 0.0


def test_canonical_provenance_builder() -> None:
    """Catalog #323 canonical Provenance has all required fields + values."""
    prov = build_canonical_provenance_for_surrogate(
        surrogate_weight_sha256="a" * 64,
        n_bytes_probed=100,
        measurement_utc="2026-05-30T00:00:00Z",
        canonical_helper="tac.scorer_surrogate.posenet_mae_v.compute_per_byte_pose_jacobian",
    )
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["axis_tag"] == "[macOS-CPU advisory]"
    assert prov["evidence_grade"] == "predicted"
    assert prov["kind"] == "predicted_from_model"
    assert prov["captured_at_utc"] == "2026-05-30T00:00:00Z"
    assert prov["n_bytes_probed"] == 100
