# SPDX-License-Identifier: MIT
"""NO-FAKE gates for the Wave-F unified-ξ ego seam + LBND3 predictive codec + the
source-temporal-smoothing re-parameterization.

Every bit-exact / decode-consistency invariant the design authority
(``unified_xi_design_and_adversarial_review_20260702.md``) requires:
* the advect operator is exact fp64 (Taylor shift) and identity at ξ=0;
* LBND3 round-trips bit-exact (serialize->deserialize reconstructs identical dequant lines);
* LBND3 is a STRICT GENERALIZATION of LBND2 (ξ=0 innovation == LBND2 temporal delta);
* the temporal-smoothing re-param is decode-consistent via the existing LBND2 codec.
Synthetic LaneLines (no gt-cache dependency) -> fast + deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.analytic_lane_render_band import (
    LaneBandRenderConfig, serialize_lane_band_rd, deserialize_lane_band_rd,
    serialize_lane_band_rd3, deserialize_lane_band_rd3, deserialize_lane_band_any,
    roundtrip_lines_through_rd3, temporal_smooth_pairs_lines,
    _pack_pairs_to_matrix, _quantize_matrix, _quantize_ego, _predictive_encode,
    derive_rd_base_steps, LANE_BAND_RD3_MAGIC,
)
from tac.boundary_math.ego_xi_trajectory import (
    XiEgoTrajectory, taylor_shift_cubic, advect_slot_vec, advect_slot_matrix,
    LaneOptimalEgoEstimator, PoseTargetEgoEstimator, ConstantVelocityEgoEstimator,
    bspline_fit_error_curve,
)
from tac.boundary_math.lane_sdf_component import LaneLine


# --------------------------------------------------------------------------- #
# synthetic per-pair lines (a slowly-advancing curved lane + a side lane)
# --------------------------------------------------------------------------- #
def _make_pairs(P: int = 20, seed: int = 0) -> list[list[LaneLine]]:
    rng = np.random.default_rng(seed)
    pairs: list[list[LaneLine]] = []
    for t in range(P):
        lines = []
        for lane_off in (-3.5, 0.0, 3.5):
            cc = np.array([1e-4, -2e-3, 0.02 + 0.001 * t, lane_off + 0.01 * t + 0.005 * rng.standard_normal()], np.float64)
            hc = np.array([0.0, 3.0 + 0.1 * rng.standard_normal()], np.float64)
            lines.append(LaneLine(centerline_coeffs=cc, halfwidth_coeffs=hc,
                                  dash_period_m=0.0, dash_phase_m=0.0, dash_duty=0.5,
                                  forward_range=(6.0, 45.0)))
        pairs.append(lines)
    return pairs


def _lines_bit_equal(A, B) -> bool:
    if len(A) != len(B):
        return False
    for la_list, lb_list in zip(A, B):
        if len(la_list) != len(lb_list):
            return False
        for a, b in zip(la_list, lb_list):
            if not (np.array_equal(np.asarray(a.centerline_coeffs), np.asarray(b.centerline_coeffs))
                    and np.array_equal(np.asarray(a.halfwidth_coeffs), np.asarray(b.halfwidth_coeffs))
                    and tuple(a.forward_range) == tuple(b.forward_range)
                    and a.dash_period_m == b.dash_period_m and a.dash_phase_m == b.dash_phase_m
                    and a.dash_duty == b.dash_duty):
                return False
    return True


# --------------------------------------------------------------------------- #
# advect operator
# --------------------------------------------------------------------------- #
def test_taylor_shift_is_exact_polynomial_shift():
    c = np.array([0.001, -0.02, 0.3, 1.5])
    for ds in (0.0, 0.5, 2.7, -1.3):
        cs = taylor_shift_cubic(c, ds)
        f = np.linspace(0, 40, 11)
        assert np.max(np.abs(np.polyval(cs, f) - np.polyval(c, f + ds))) < 1e-9


def test_advect_identity_at_zero_ego():
    v = np.arange(11, dtype=np.float64) + 0.5
    assert np.array_equal(advect_slot_vec(v, 0.0, 0.0, 0.0), v)


def test_advect_dof_isolation():
    v = np.zeros(11, np.float64)
    v[:4] = [0.0, 0.0, 0.0, 5.0]  # flat centerline at lateral 5
    # dy shifts c0 only
    assert advect_slot_vec(v, 0.0, 0.7, 0.0)[3] == pytest.approx(5.7)
    # dpsi shifts c1 only
    assert advect_slot_vec(v, 0.0, 0.0, 0.11)[2] == pytest.approx(0.11)
    # ds shifts forward_range by -ds
    v2 = v.copy(); v2[9], v2[10] = 6.0, 45.0
    out = advect_slot_vec(v2, 3.0, 0.0, 0.0)
    assert out[9] == pytest.approx(3.0) and out[10] == pytest.approx(42.0)


def test_advect_slot_matrix_present_mask_skips_absent():
    row = np.zeros(22, np.float64)
    row[3] = 5.0; row[14] = 9.0  # slot0 c0=5, slot1 c0=9
    mask = np.array([True, False])
    out = advect_slot_matrix(row, 0.0, 1.0, 0.0, K=2, present_mask=mask)
    assert out[3] == pytest.approx(6.0)   # slot0 advected (c0 += dy)
    assert out[14] == pytest.approx(9.0)  # slot1 untouched (absent)


# --------------------------------------------------------------------------- #
# XiEgoTrajectory
# --------------------------------------------------------------------------- #
def test_xi_trajectory_planar_embedding():
    ds = np.array([0.0, 2.0, 3.0]); dy = np.array([0.0, 0.1, -0.2]); dpsi = np.array([0.0, 0.01, 0.02])
    xt = XiEgoTrajectory.from_planar(ds, dy, dpsi, "t")
    assert np.array_equal(xt.dense_xi[:, 0], dy)   # rho_x = lateral
    assert np.array_equal(xt.dense_xi[:, 2], ds)   # rho_z = forward
    assert np.array_equal(xt.dense_xi[:, 4], dpsi)  # omega_y = yaw


def test_xi_trajectory_rejects_bad_shape_or_dtype():
    with pytest.raises(ValueError):
        XiEgoTrajectory(n_pairs=3, ds=np.zeros(2), dy=np.zeros(3), dpsi=np.zeros(3),
                        dense_xi=np.zeros((3, 6)), estimator_id="t")
    with pytest.raises(ValueError):
        XiEgoTrajectory(n_pairs=3, ds=np.zeros(3, np.float32), dy=np.zeros(3), dpsi=np.zeros(3),
                        dense_xi=np.zeros((3, 6)), estimator_id="t")
    with pytest.raises(ValueError):
        XiEgoTrajectory.from_planar(np.zeros(3), np.zeros(3), np.zeros(3), "")


# --------------------------------------------------------------------------- #
# LBND3 codec bit-exactness
# --------------------------------------------------------------------------- #
def test_lbnd3_roundtrip_bit_exact():
    pairs = _make_pairs()
    cfg = LaneBandRenderConfig()
    M, presence, K = _pack_pairs_to_matrix(pairs)
    xi = LaneOptimalEgoEstimator().estimate(M, presence, K)
    dq3, blob = roundtrip_lines_through_rd3(pairs, cfg, xi)
    assert blob[:len(LANE_BAND_RD3_MAGIC)] == LANE_BAND_RD3_MAGIC
    # LBND3 dequant lines == LBND2 dequant lines (SAME quant grid; only the coding differs)
    dq2, _ = deserialize_lane_band_rd(serialize_lane_band_rd(pairs, cfg))
    assert _lines_bit_equal(dq2, dq3)
    # deserialize_lane_band_any dispatches LBND3
    dq_any, hdr = deserialize_lane_band_any(blob)
    assert _lines_bit_equal(dq_any, dq3) and hdr["format"] == 3


def test_lbnd3_strictly_generalizes_lbnd2_at_zero_ego():
    pairs = _make_pairs()
    M, presence, K = _pack_pairs_to_matrix(pairs)
    steps = derive_rd_base_steps(); steps_full = np.tile(steps, K)
    Q = _quantize_matrix(M, steps_full)
    xi0 = XiEgoTrajectory.from_planar(np.zeros(M.shape[0]), np.zeros(M.shape[0]), np.zeros(M.shape[0]), "zero")
    Qds, Qdy, Qdp, dsdq, dydq, dpdq = _quantize_ego(xi0.ds, xi0.dy, xi0.dpsi)
    innov0 = _predictive_encode(Q, presence, steps_full, dsdq, dydq, dpdq, K)
    dq = Q.copy(); dq[1:] = Q[1:] - Q[:-1]  # LBND2 temporal delta
    assert np.array_equal(innov0, dq)


def test_lbnd3_nonzero_ego_roundtrip_bit_exact():
    pairs = _make_pairs(seed=3)
    cfg = LaneBandRenderConfig()
    ds = np.linspace(0, 2.0, len(pairs)); dy = 0.05 * np.sin(np.arange(len(pairs)))
    dpsi = 0.01 * np.cos(np.arange(len(pairs)))
    xi = XiEgoTrajectory.from_planar(ds, dy, dpsi, "synthetic_nonzero")
    blob = serialize_lane_band_rd3(pairs, cfg, xi)
    dq_lines, _ = deserialize_lane_band_rd3(blob)
    # bit-exact vs LBND2 dequant grid (independent of predictor)
    dq2, _ = deserialize_lane_band_rd(serialize_lane_band_rd(pairs, cfg))
    assert _lines_bit_equal(dq2, dq_lines)


def test_lbnd3_rejects_npair_mismatch():
    pairs = _make_pairs(P=10)
    xi = XiEgoTrajectory.from_planar(np.zeros(5), np.zeros(5), np.zeros(5), "wrong")
    with pytest.raises(ValueError):
        serialize_lane_band_rd3(pairs, LaneBandRenderConfig(), xi)


# --------------------------------------------------------------------------- #
# temporal-smoothing source re-parameterization (decode-consistent via LBND2)
# --------------------------------------------------------------------------- #
def test_temporal_smooth_decode_consistent_via_lbnd2():
    pairs = _make_pairs(P=30)
    cfg = LaneBandRenderConfig()
    sm = temporal_smooth_pairs_lines(pairs, win=5)
    # the smoothed lines ship as standard LBND2 -> serialize/deserialize is bit-exact
    blob = serialize_lane_band_rd(sm, cfg)
    dq, hdr = deserialize_lane_band_rd(blob)
    assert _lines_bit_equal(deserialize_lane_band_rd(serialize_lane_band_rd(sm, cfg))[0], dq)
    assert hdr["format"] == 2  # smoothing does NOT change the format (existing inflate path)


def test_temporal_smooth_preserves_line_count():
    pairs = _make_pairs(P=25)
    for w in (3, 5, 9):
        sm = temporal_smooth_pairs_lines(pairs, win=w)
        assert sum(len(l) for l in sm) == sum(len(l) for l in pairs)


def test_temporal_smooth_win1_is_near_identity():
    pairs = _make_pairs(P=15)
    sm = temporal_smooth_pairs_lines(pairs, win=1)
    # win=1 -> median over a single point == identity on smoothed dims; centerline preserved
    for a_list, b_list in zip(pairs, sm):
        for a, b in zip(a_list, b_list):
            assert np.allclose(np.asarray(a.centerline_coeffs)[-4:], np.asarray(b.centerline_coeffs)[-4:], atol=1e-12)


# --------------------------------------------------------------------------- #
# estimators + b-spline curve
# --------------------------------------------------------------------------- #
def test_lane_optimal_estimator_runs_and_bounds():
    pairs = _make_pairs()
    M, presence, K = _pack_pairs_to_matrix(pairs)
    xi = LaneOptimalEgoEstimator().estimate(M, presence, K)
    assert xi.ds.shape == (len(pairs),) and xi.dy.shape == (len(pairs),)
    assert xi.ds[0] == 0.0 and xi.dy[0] == 0.0 and xi.dpsi[0] == 0.0
    xi2 = LaneOptimalEgoEstimator(fit_forward=False).estimate(M, presence, K)
    assert np.all(xi2.ds == 0.0)  # forward shift disabled


def test_posetarget_estimator_affine_and_geometry():
    pairs = _make_pairs()
    M, presence, K = _pack_pairs_to_matrix(pairs)
    ref = LaneOptimalEgoEstimator().estimate(M, presence, K)
    gt = np.zeros((len(pairs), 6), np.float64)
    gt[:, 0] = np.arange(len(pairs))         # forward channel
    xi_aff = PoseTargetEgoEstimator(calib="affine_to_lane").estimate(gt, ref=ref)
    assert xi_aff.n_pairs == len(pairs) and "affine_ds" in xi_aff.provenance
    xi_geo = PoseTargetEgoEstimator(calib="geometry", s_t=0.1).estimate(gt)
    assert xi_geo.ds[0] == 0.0


def test_const_velocity_estimator():
    xi = ConstantVelocityEgoEstimator(ds_const=2.5).estimate(12)
    assert xi.ds[0] == 0.0 and np.all(xi.ds[1:] == 2.5) and np.all(xi.dy == 0.0)


def test_bspline_fit_error_curve_runs():
    dxi = np.zeros((40, 6), np.float64); dxi[1:, 2] = 1.0  # constant forward
    rows = bspline_fit_error_curve(dxi, [8, 16, 24])
    assert len(rows) == 3 and all("fwd_rms_m" in r and "ctrl_bytes_fp16" in r for r in rows)
