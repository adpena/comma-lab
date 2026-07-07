# SPDX-License-Identifier: MIT
"""Tests for tac.boundary_math.ground_frame_chart (the #194 / §17.1 chart machinery).

The load-bearing suite: (1) BIT-PARITY of the cumulative homography with the MEASURED FEED-ll
reach tool (the math is REUSED, not re-derived — this test pins it so it cannot drift);
(2) chart[ref] == identity exactly; (3) forward/backward cumulative inverse-consistency;
(4) normalized-coords round-trip vs the pixel-domain map; (5) numpy↔MLX fp32 twin parity
(CPU-stream bit-exact per the standing MLX-GPU-not-bit-identical discipline); (6) identity
fast-path byte-identity (the trainer's --ground-frame-chart-off / ref-pair invariants)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.boundary_math.ground_frame_chart import (  # noqa: E402
    ChartCalibration,
    GroundFrameChart,
    GroundFrameChartError,
    chart_homography_pixel,
    cumulative_plane_motion,
    intrinsics_for_grid,
    normalization_affine,
    plane_motion_step,
    precompose_coords_numpy,
)

SEG_H, SEG_W = 384, 512


def _rand_poses(P: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poses = rng.normal(0.0, 1.0, size=(P, 6))
    poses[:, 0] = np.abs(poses[:, 0]) * 30.0 + 5.0  # dominant forward channel (col0 ~ 33)
    return poses.astype(np.float64)


_CAL = ChartCalibration()  # the MEASURED FEED-ll defaults


# --------------------------------------------------------------------------- #
# (1) bit-parity with the FEED-ll reach tool (the math-reuse pin)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("regime", ["ground", "rotonly", "identity"])
@pytest.mark.parametrize("k", [0, 1, 5, 13])
def test_cumulative_homography_bit_parity_with_reach_tool(regime, k):
    tool = pytest.importorskip("tools.measure_screw_reach_through_R")
    poses = _rand_poses(20)
    K = intrinsics_for_grid(SEG_W, SEG_H)
    Kinv = np.linalg.inv(K)
    a = 2
    H_tool = tool.cumulative_homography(
        poses, a, k, K, Kinv, (_CAL.s_t, _CAL.s_r, _CAL.pitch), regime)
    H_mine = chart_homography_pixel(poses, a, a + k, _CAL, regime, K)
    # same op order end-to-end -> exact equality (not allclose): the reuse pin.
    assert np.array_equal(H_tool, H_mine), f"regime={regime} k={k} drifted from the reach tool"


def test_plane_motion_step_matches_tool_m_step():
    tool = pytest.importorskip("tools.measure_screw_reach_through_R")
    pose = _rand_poses(1)[0]
    for regime in ("ground", "rotonly", "identity"):
        M_tool = tool._m_step(pose, _CAL.s_t, _CAL.s_r, _CAL.pitch, regime)
        M_mine = plane_motion_step(pose, _CAL, regime)
        assert np.array_equal(M_tool, M_mine), regime


def test_intrinsics_for_grid_matches_tool_intrinsics_at():
    tool = pytest.importorskip("tools.measure_pose_warp_dseg")
    assert np.array_equal(tool.intrinsics_at(SEG_W, SEG_H), intrinsics_for_grid(SEG_W, SEG_H))


# --------------------------------------------------------------------------- #
# (2)+(3) chart structure: identity at ref; inverse consistency both directions
# --------------------------------------------------------------------------- #
def test_chart_identity_at_ref_exact():
    poses = _rand_poses(12)
    for ref in (0, 5, 11):
        chart = GroundFrameChart.build(poses, ref_pair=ref, grid_hw=(SEG_H, SEG_W))
        assert np.array_equal(chart.H_chart_norm[ref], np.eye(3))
        assert np.array_equal(chart.H_fwd_pix[ref], np.eye(3))


def test_cumulative_inverse_consistency():
    poses = _rand_poses(10)
    M_fwd = cumulative_plane_motion(poses, 2, 7, _CAL, "ground")
    M_bwd = cumulative_plane_motion(poses, 7, 2, _CAL, "ground")
    assert np.allclose(M_fwd @ M_bwd, np.eye(3), atol=1e-10)


def test_build_backward_ref_matches_direct_cumulative():
    """H_fwd for t < ref must equal K·inv(M_cum(t→ref))·K⁻¹ (the direct formula)."""
    poses = _rand_poses(9)
    ref = 6
    K = intrinsics_for_grid(SEG_W, SEG_H)
    chart = GroundFrameChart.build(poses, ref_pair=ref, grid_hw=(SEG_H, SEG_W), K=K)
    for t in (0, 3, 5):
        M_direct = np.linalg.inv(cumulative_plane_motion(poses, t, ref, _CAL, "ground"))
        H_direct = K @ M_direct @ np.linalg.inv(K)
        assert np.allclose(chart.H_fwd_pix[t], H_direct, atol=1e-9), t


def test_build_forward_matches_reach_tool_per_t():
    """Incremental build == the tool's per-k from-scratch cumulative product, bit-exact."""
    tool = pytest.importorskip("tools.measure_screw_reach_through_R")
    poses = _rand_poses(8)
    K = intrinsics_for_grid(SEG_W, SEG_H)
    chart = GroundFrameChart.build(poses, ref_pair=0, grid_hw=(SEG_H, SEG_W), K=K)
    for k in range(8):
        H_tool = tool.cumulative_homography(
            poses, 0, k, K, np.linalg.inv(K), (_CAL.s_t, _CAL.s_r, _CAL.pitch), "ground")
        assert np.array_equal(chart.H_fwd_pix[k], H_tool), k


# --------------------------------------------------------------------------- #
# (4) normalized chart == pixel-domain chart mapped through the grid affine
# --------------------------------------------------------------------------- #
def test_normalized_chart_matches_pixel_domain():
    poses = _rand_poses(6)
    chart = GroundFrameChart.build(poses, ref_pair=0, grid_hw=(SEG_H, SEG_W))
    A = normalization_affine(SEG_H, SEG_W)
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1.0, 1.0, size=(64, 2)).astype(np.float32)
    t = 4
    out_norm = chart.coords_for_pair_numpy(xy, t)
    # reference: normalized -> pixel -> inv(H_fwd_pix) -> back to normalized (fp64)
    ones = np.ones((64, 1))
    pix = (A @ np.concatenate([xy.astype(np.float64), ones], 1).T).T
    ref = (np.linalg.inv(chart.H_fwd_pix[t]) @ pix.T).T
    ref = ref[:, :2] / ref[:, 2:3]
    back = (np.linalg.inv(A) @ np.concatenate([ref, ones], 1).T).T
    ref_norm = back[:, :2] / back[:, 2:3]
    assert np.allclose(out_norm, ref_norm, atol=1e-4)  # fp32 apply vs fp64 reference


# --------------------------------------------------------------------------- #
# (5) numpy <-> MLX twin parity (CPU stream; bit-exact per the standing discipline)
# --------------------------------------------------------------------------- #
def test_mlx_twin_parity_cpu_bit_exact():
    mx = pytest.importorskip("mlx.core")
    poses = _rand_poses(5)
    chart = GroundFrameChart.build(poses, ref_pair=0, grid_hw=(SEG_H, SEG_W))
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1.0, 1.0, size=(257, 2)).astype(np.float32)
    ref = chart.coords_for_pair_numpy(xy, 3)
    with mx.stream(mx.cpu):
        out = chart.coords_for_pair_mlx(mx.array(xy), 3)
        mx.eval(out)
    got = np.array(out)
    assert np.array_equal(ref, got), "MLX CPU twin not bit-identical to the numpy fp32 reference"


def test_mlx_twin_parity_default_device_close():
    mx = pytest.importorskip("mlx.core")
    poses = _rand_poses(5)
    chart = GroundFrameChart.build(poses, ref_pair=0, grid_hw=(SEG_H, SEG_W))
    xy = np.random.default_rng(2).uniform(-1, 1, size=(64, 2)).astype(np.float32)
    ref = chart.coords_for_pair_numpy(xy, 2)
    out = np.array(chart.coords_for_pair_mlx(mx.array(xy), 2))
    assert np.allclose(ref, out, atol=1e-6)


# --------------------------------------------------------------------------- #
# (6) identity fast-path + guards
# --------------------------------------------------------------------------- #
def test_identity_fast_path_returns_same_object():
    xy = np.zeros((10, 2), np.float32)
    out = precompose_coords_numpy(xy, np.eye(3))
    assert out is xy  # byte-identity by construction (the chart-off / ref-pair invariant)


def test_ref_pair_coords_unchanged_through_chart():
    poses = _rand_poses(7)
    chart = GroundFrameChart.build(poses, ref_pair=3, grid_hw=(SEG_H, SEG_W))
    xy = np.random.default_rng(3).uniform(-1, 1, size=(32, 2)).astype(np.float32)
    assert chart.coords_for_pair_numpy(xy, 3) is xy


def test_bad_inputs_raise():
    poses = _rand_poses(4)
    with pytest.raises(GroundFrameChartError):
        GroundFrameChart.build(poses, ref_pair=9, grid_hw=(SEG_H, SEG_W))
    with pytest.raises(GroundFrameChartError):
        GroundFrameChart.build(poses, ref_pair=0, grid_hw=(SEG_H, SEG_W), regime="warp")
    with pytest.raises(GroundFrameChartError):
        precompose_coords_numpy(np.zeros((5, 3), np.float32), np.eye(3))
    with pytest.raises(GroundFrameChartError):
        cumulative_plane_motion(poses, 0, 9, _CAL, "ground")


def test_z_guard_no_nan_inf():
    # a homography with a near-degenerate projective row must not produce nan/inf
    H = np.eye(3)
    H[2] = [1.0, 1.0, 1.0]  # z = x + y + 1 -> 0 along the x+y=-1 line
    xy = np.array([[-0.5, -0.5], [-0.4, -0.6], [0.3, 0.2]], np.float32)
    out = precompose_coords_numpy(xy, H)
    assert np.all(np.isfinite(out))
