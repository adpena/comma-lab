"""Synthetic tests for Lane RAFT/radial pose 6-DoF decomposition.

All tests run on synthetic flow fields with known basis coefficients;
no GPU, no real RAFT inference, no real archive.
Real-anchor empirical measurement is Phase C (Level 2 — out of scope).

References
----------
- Module: src/tac/raft_radial_pose.py
- Design: .omx/research/council_lane_raft_radial_pose_design_20260430.md
- Sibling: src/tac/raft_pose.py (Lane FL — single-DOF dim 0 only)
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.raft_radial_pose import (
    MODE_COMPRESS_TIME_PRIOR,
    MODE_INFLATE_RECOMPUTE,
    RAFT_RADIAL_VERSION,
    PoseCalibration,
    RaftRadialPoseConfig,
    VALID_MODES,
    build_radial_basis,
    calibrate_to_contest_pose,
    compute_radial_basis_from_flow,
    emit_inflate_compliance_banner,
    estimate_pose_compress_time_prior,
    evaluate_disagreement,
)


# ── basis builder ─────────────────────────────────────────────────────


def test_radial_basis_shape():
    """B has shape (h*w*2, 6) for canonical 6-DoF basis."""
    h, w = 8, 16
    B = build_radial_basis(h=h, w=w, n_basis=6)
    assert B.shape == (h * w * 2, 6)
    assert B.dtype == np.float32


def test_radial_basis_first_two_columns_are_pure_translation():
    """B[:, 0] = (1, 0); B[:, 1] = (0, 1) at every pixel."""
    B = build_radial_basis(h=4, w=4, n_basis=6)
    # B_0: u=1, v=0 at every pixel
    assert np.allclose(B[0::2, 0], 1.0)
    assert np.allclose(B[1::2, 0], 0.0)
    # B_1: u=0, v=1
    assert np.allclose(B[0::2, 1], 0.0)
    assert np.allclose(B[1::2, 1], 1.0)


def test_radial_basis_rejects_invalid_dims():
    with pytest.raises(ValueError, match="positive"):
        build_radial_basis(h=0, w=4, n_basis=6)


def test_radial_basis_rejects_non6_n_basis():
    """Higher-order basis is reserved for Lane 11 wavelet residual."""
    with pytest.raises(NotImplementedError, match="Wavelet"):
        build_radial_basis(h=4, w=4, n_basis=10)


# ── decomposition / projection ────────────────────────────────────────


def test_pure_translation_recovers_alpha_0_1():
    """Pure horizontal translation flow projects entirely onto B_0."""
    h, w, T = 16, 32, 5
    flow = np.zeros((T, h, w, 2), dtype=np.float32)
    flow[..., 0] = 0.7  # u = 0.7 everywhere; v = 0
    alpha, residual = compute_radial_basis_from_flow(flow_field=flow, n_basis=6)
    assert alpha.shape == (T, 6)
    # alpha[:, 0] should be ~0.7; other dims ~0
    assert np.allclose(alpha[:, 0], 0.7, atol=1e-5)
    assert np.allclose(alpha[:, 1:], 0.0, atol=1e-5)
    # Residual is (near-)zero
    assert np.max(np.abs(residual)) < 1e-4


def test_pure_roll_recovers_alpha_3():
    """Pure roll flow projects entirely onto B_3."""
    h, w, T = 16, 32, 3
    # Build flow = 0.5 * B_3
    B = build_radial_basis(h=h, w=w, n_basis=6)
    flow_flat = (0.5 * B[:, 3]).reshape(h, w, 2)
    flow = np.tile(flow_flat[None], (T, 1, 1, 1))
    alpha, residual = compute_radial_basis_from_flow(flow_field=flow, n_basis=6)
    assert np.allclose(alpha[:, 3], 0.5, atol=1e-5)
    # Other dims zero
    other_dims = [0, 1, 2, 4, 5]
    for d in other_dims:
        assert np.allclose(alpha[:, d], 0.0, atol=1e-5)


def test_decompose_rejects_wrong_shape():
    flow = np.zeros((3, 4, 5), dtype=np.float32)  # missing trailing 2
    with pytest.raises(ValueError, match="shape"):
        compute_radial_basis_from_flow(flow_field=flow, n_basis=6)


# ── calibration ────────────────────────────────────────────────────────


def test_calibration_recovers_identity_when_alpha_is_pose():
    """If alpha == contest_pose (both shape (T, 6)), A = I, b = 0."""
    T = 50
    rng = np.random.default_rng(seed=42)
    alpha = rng.standard_normal((T, 6)).astype(np.float32)
    contest_pose = alpha.copy()  # identity mapping
    cal = calibrate_to_contest_pose(
        alpha=alpha, contest_pose=contest_pose, calibration_window=20,
    )
    assert cal.A.shape == (6, 6)
    assert cal.b.shape == (6,)
    # A close to identity
    assert np.allclose(cal.A, np.eye(6, dtype=np.float32), atol=1e-4)
    assert np.allclose(cal.b, 0.0, atol=1e-4)
    assert cal.rmse_train < 1e-4


def test_calibration_handles_affine_offset():
    """If pose = 2*alpha + 5, recovered A = 2I, b = 5."""
    T = 50
    rng = np.random.default_rng(seed=7)
    alpha = rng.standard_normal((T, 6)).astype(np.float32)
    contest_pose = (2.0 * alpha + 5.0).astype(np.float32)
    cal = calibrate_to_contest_pose(
        alpha=alpha, contest_pose=contest_pose, calibration_window=20,
    )
    assert np.allclose(cal.A, 2.0 * np.eye(6, dtype=np.float32), atol=1e-3)
    assert np.allclose(cal.b, 5.0, atol=1e-3)


def test_calibration_rejects_short_window():
    alpha = np.zeros((20, 6), dtype=np.float32)
    pose = np.zeros((20, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="calibration_window"):
        calibrate_to_contest_pose(
            alpha=alpha, contest_pose=pose, calibration_window=5,
        )


def test_calibration_rejects_window_larger_than_T():
    alpha = np.zeros((20, 6), dtype=np.float32)
    pose = np.zeros((20, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="calibration_window"):
        calibrate_to_contest_pose(
            alpha=alpha, contest_pose=pose, calibration_window=100,
        )


# ── disagreement ──────────────────────────────────────────────────────


def test_disagreement_zero_for_identical_poses():
    pose = np.random.default_rng(0).standard_normal((50, 6)).astype(np.float32)
    metrics = evaluate_disagreement(pose_estimated=pose, pose_contest=pose)
    assert metrics["overall_mse"] < 1e-9
    assert metrics["kill_threshold_passed"]


def test_disagreement_kill_threshold_fires_above_1e3():
    pose_a = np.zeros((50, 6), dtype=np.float32)
    pose_b = np.full((50, 6), 0.05, dtype=np.float32)  # 0.05^2 = 2.5e-3
    metrics = evaluate_disagreement(pose_estimated=pose_a, pose_contest=pose_b)
    assert metrics["overall_mse"] > 1e-3
    assert not metrics["kill_threshold_passed"]


def test_disagreement_rejects_shape_mismatch():
    a = np.zeros((10, 6), dtype=np.float32)
    b = np.zeros((20, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatch"):
        evaluate_disagreement(pose_estimated=a, pose_contest=b)


# ── config validation ─────────────────────────────────────────────────


def test_config_mode_a_no_marker_required():
    """Mode A is fine without a compliance marker."""
    cfg = RaftRadialPoseConfig(mode=MODE_COMPRESS_TIME_PRIOR)
    assert cfg.mode == MODE_COMPRESS_TIME_PRIOR


def test_config_mode_b_requires_marker():
    """Mode B without marker raises ValueError per CLAUDE.md non-negotiable."""
    with pytest.raises(ValueError, match="Mode B"):
        RaftRadialPoseConfig(mode=MODE_INFLATE_RECOMPUTE)


def test_config_mode_b_with_marker_ok():
    cfg = RaftRadialPoseConfig(
        mode=MODE_INFLATE_RECOMPUTE,
        inflate_compliance_marker="docs/compliance/raft_inflate_approved_2026.md",
    )
    assert cfg.inflate_compliance_marker.endswith("raft_inflate_approved_2026.md")


def test_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        RaftRadialPoseConfig(mode="enthusiastic_mode")


def test_config_rejects_n_basis_below_6():
    with pytest.raises(ValueError, match="n_basis_functions"):
        RaftRadialPoseConfig(mode=MODE_COMPRESS_TIME_PRIOR, n_basis_functions=4)


# ── compliance banner ─────────────────────────────────────────────────


def test_compliance_banner_empty_for_mode_a():
    cfg = RaftRadialPoseConfig(mode=MODE_COMPRESS_TIME_PRIOR)
    assert emit_inflate_compliance_banner(config=cfg) == ""


def test_compliance_banner_emits_for_mode_b():
    cfg = RaftRadialPoseConfig(
        mode=MODE_INFLATE_RECOMPUTE,
        inflate_compliance_marker="docs/raft_compliance.md",
    )
    banner = emit_inflate_compliance_banner(config=cfg)
    assert "[strict-scorer-rule]" in banner
    assert "non-compliant" in banner
    assert "docs/raft_compliance.md" in banner


# ── end-to-end Mode A ─────────────────────────────────────────────────


def test_mode_a_end_to_end_zero_mse_when_pose_is_alpha():
    """If contest_pose is exactly the radial alpha (after the basis decomp),
    end-to-end pipeline should recover near-zero MSE."""
    # Build synthetic flow that's a pure scaling of basis B_0 + B_3 over time
    h, w, T = 16, 16, 60
    rng = np.random.default_rng(seed=123)
    # Random per-frame mix of basis coefficients
    coeffs = rng.standard_normal((T, 6)).astype(np.float32) * 0.3
    B = build_radial_basis(h=h, w=w, n_basis=6)
    flow_flat_per_frame = coeffs @ B.T  # (T, h*w*2)
    flow_field = flow_flat_per_frame.reshape(T, h, w, 2)
    # Contest pose for calibration: linear function of the basis coefficients
    A_true = rng.standard_normal((6, 6)).astype(np.float32)
    b_true = rng.standard_normal(6).astype(np.float32)
    contest_pose = (coeffs @ A_true.T + b_true).astype(np.float32)

    cfg = RaftRadialPoseConfig(
        mode=MODE_COMPRESS_TIME_PRIOR,
        calibration_window=30,
        image_h=h,
        image_w=w,
    )
    pose_estimated, cal, metrics = estimate_pose_compress_time_prior(
        flow_field=flow_field,
        contest_pose_for_calibration=contest_pose,
        config=cfg,
    )
    assert pose_estimated.shape == (T, 6)
    # Pose-TTO would refine from here; for synthetic exact-fit, MSE should be tiny
    assert metrics["overall_mse"] < 1e-3
    assert cal.A.shape == (6, 6)
    assert cal.b.shape == (6,)


def test_mode_a_entrypoint_rejects_mode_b_config():
    """estimate_pose_compress_time_prior is Mode A only."""
    cfg = RaftRadialPoseConfig(
        mode=MODE_INFLATE_RECOMPUTE,
        inflate_compliance_marker="x.md",
        image_h=8,
        image_w=8,
    )
    flow = np.zeros((20, 8, 8, 2), dtype=np.float32)
    contest = np.zeros((20, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="Mode A"):
        estimate_pose_compress_time_prior(
            flow_field=flow,
            contest_pose_for_calibration=contest,
            config=cfg,
        )


# ── version sentinels ─────────────────────────────────────────────────


def test_version_pinned():
    assert RAFT_RADIAL_VERSION == 1


def test_valid_modes_pinned():
    assert MODE_COMPRESS_TIME_PRIOR in VALID_MODES
    assert MODE_INFLATE_RECOMPUTE in VALID_MODES
    assert len(VALID_MODES) == 2
