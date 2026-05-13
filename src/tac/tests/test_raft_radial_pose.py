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
    NormalizedFlowCalibration,
    RaftRadialPoseConfig,
    VALID_MODES,
    build_radial_basis,
    calibrate_to_contest_pose,
    compute_radial_basis_from_flow,
    emit_inflate_compliance_banner,
    estimate_pose_compress_time_prior,
    evaluate_disagreement,
)


def _to_pixel_flow(flow_normalized: np.ndarray, cal: NormalizedFlowCalibration) -> np.ndarray:
    flow_px = flow_normalized.copy()
    flow_px[..., 0] *= np.float32(cal.fx)
    flow_px[..., 1] *= np.float32(cal.fy)
    return flow_px.astype(np.float32)


# ── basis builder ─────────────────────────────────────────────────────


def test_radial_basis_shape():
    """B has shape (h*w*2, 6) for canonical 6-DoF basis."""
    h, w = 8, 16
    B = build_radial_basis(h=h, w=w, n_basis=6)
    assert B.shape == (h * w * 2, 6)
    assert B.dtype == np.float32


def test_radial_basis_first_two_columns_are_pure_translation():
    """B[:, 0] and B[:, 1] are proxy-depth translation fields."""
    cal = NormalizedFlowCalibration(
        fx=100.0, fy=100.0, cx=1.5, cy=1.5, proxy_depth_m=20.0,
    )
    B = build_radial_basis(h=4, w=4, n_basis=6, calibration=cal)
    rho = 1.0 / cal.proxy_depth_m
    # B_0: normalized u=-rho, v=0 at every pixel
    assert np.allclose(B[0::2, 0], -rho)
    assert np.allclose(B[1::2, 0], 0.0)
    # B_1: normalized u=0, v=-rho
    assert np.allclose(B[0::2, 1], 0.0)
    assert np.allclose(B[1::2, 1], -rho)


def test_calibrated_rotation_basis_keeps_normalized_coordinate_constants():
    """Pitch/yaw must include the Longuet-Higgins constant terms.

    At the principal point x=y=0:
      pitch omega_x has normalized flow (0, -1)
      yaw omega_y has normalized flow (1, 0)
      roll omega_z is zero.
    The former proxy basis omitted these constants and therefore could not
    represent calibrated rotation around the optical center.
    """
    cal = NormalizedFlowCalibration(
        fx=100.0, fy=100.0, cx=1.0, cy=1.0, proxy_depth_m=30.0,
    )
    B = build_radial_basis(h=3, w=3, n_basis=6, calibration=cal)
    center = (1 * 3 + 1) * 2
    assert np.allclose(B[center:center + 2, 3], [0.0, 0.0])
    assert np.allclose(B[center:center + 2, 4], [0.0, -1.0])
    assert np.allclose(B[center:center + 2, 5], [1.0, 0.0])


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
    cal = NormalizedFlowCalibration(
        fx=40.0, fy=40.0, cx=(w - 1) / 2, cy=(h - 1) / 2, proxy_depth_m=25.0,
    )
    B = build_radial_basis(h=h, w=w, n_basis=6, calibration=cal)
    flow_flat = (0.7 * B[:, 0]).reshape(h, w, 2)
    flow = np.tile(_to_pixel_flow(flow_flat, cal)[None], (T, 1, 1, 1))
    alpha, residual = compute_radial_basis_from_flow(
        flow_field=flow, n_basis=6, calibration=cal,
    )
    assert alpha.shape == (T, 6)
    # alpha[:, 0] should be ~0.7; other dims ~0
    assert np.allclose(alpha[:, 0], 0.7, atol=1e-5)
    assert np.allclose(alpha[:, 1:], 0.0, atol=1e-5)
    # Residual is (near-)zero
    assert np.max(np.abs(residual)) < 1e-4


def test_pure_roll_recovers_alpha_3():
    """Pure roll flow projects entirely onto B_3."""
    h, w, T = 16, 32, 3
    cal = NormalizedFlowCalibration(
        fx=40.0, fy=40.0, cx=(w - 1) / 2, cy=(h - 1) / 2, proxy_depth_m=30.0,
    )
    # Build flow = 0.5 * B_3
    B = build_radial_basis(h=h, w=w, n_basis=6, calibration=cal)
    flow_flat = (0.5 * B[:, 3]).reshape(h, w, 2)
    flow = np.tile(_to_pixel_flow(flow_flat, cal)[None], (T, 1, 1, 1))
    alpha, residual = compute_radial_basis_from_flow(
        flow_field=flow, n_basis=6, calibration=cal,
    )
    assert np.allclose(alpha[:, 3], 0.5, atol=1e-5)
    # Other dims zero
    other_dims = [0, 1, 2, 4, 5]
    for d in other_dims:
        assert np.allclose(alpha[:, d], 0.0, atol=1e-5)


def test_pixel_flow_is_focal_normalized_before_lsq():
    """A yaw coefficient should recover from raw pixel flow scaled by fx/fy."""
    h, w, T = 5, 5, 2
    cal = NormalizedFlowCalibration(
        fx=80.0, fy=120.0, cx=2.0, cy=2.0, proxy_depth_m=30.0,
    )
    B = build_radial_basis(h=h, w=w, n_basis=6, calibration=cal)
    coeff = 0.25
    flow_normalized = (coeff * B[:, 5]).reshape(h, w, 2)
    flow_px = np.tile(_to_pixel_flow(flow_normalized, cal)[None], (T, 1, 1, 1))

    # At the principal point yaw's normalized u is +coeff, so raw pixel flow
    # is fx*coeff. This guards against fitting pixel magnitudes as if they
    # were normalized coordinates.
    assert flow_px[0, 2, 2, 0] == pytest.approx(cal.fx * coeff)
    alpha, residual = compute_radial_basis_from_flow(
        flow_field=flow_px, n_basis=6, calibration=cal,
    )
    assert np.allclose(alpha[:, 5], coeff, atol=1e-5)
    assert np.max(np.abs(residual)) < 1e-4


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
    cal = NormalizedFlowCalibration(
        fx=40.0, fy=40.0, cx=(w - 1) / 2, cy=(h - 1) / 2, proxy_depth_m=30.0,
    )
    rng = np.random.default_rng(seed=123)
    # Random per-frame mix of basis coefficients
    coeffs = rng.standard_normal((T, 6)).astype(np.float32) * 0.3
    B = build_radial_basis(h=h, w=w, n_basis=6, calibration=cal)
    flow_flat_per_frame = coeffs @ B.T  # (T, h*w*2), normalized flow
    flow_field = _to_pixel_flow(flow_flat_per_frame.reshape(T, h, w, 2), cal)
    # Contest pose for calibration: linear function of the basis coefficients
    A_true = rng.standard_normal((6, 6)).astype(np.float32)
    b_true = rng.standard_normal(6).astype(np.float32)
    contest_pose = (coeffs @ A_true.T + b_true).astype(np.float32)

    cfg = RaftRadialPoseConfig(
        mode=MODE_COMPRESS_TIME_PRIOR,
        calibration_window=30,
        image_h=h,
        image_w=w,
        flow_calibration=cal,
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
    assert RAFT_RADIAL_VERSION == 2


def test_valid_modes_pinned():
    assert MODE_COMPRESS_TIME_PRIOR in VALID_MODES
    assert MODE_INFLATE_RECOMPUTE in VALID_MODES
    assert len(VALID_MODES) == 2
