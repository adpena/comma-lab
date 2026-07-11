"""Tests for the SHARED phase primitives (T1 cross-pair phase-advection + #360 forces).

Covers: tie-coordinate correctness + numpy/MLX bit-parity + differentiability; A_ξ tie-field
advection (ξ=0 identity, numpy/MLX parity within the documented warp floor, [0,1] preserved);
the cross-scored-frame screw interpolation (se(3) composition, not a linear sum); and the T1
residual reducer (zero on identical / zero on zero-weight default-off no-op / positive on
phase-mismatch / annulus-weight restricts support / differentiable). numpy-fp32 is the authority.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import phase_primitives as pp
from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    xi_from_pose_calibration,
)

mx = pytest.importorskip("mlx.core")


# --------------------------------------------------------------------------- #
# 0. GT tie-target selection (dominant genuine-V straddle)                     #
# --------------------------------------------------------------------------- #
def test_gt_tie_targets_sentinel_and_active_agree():
    # sentinel (-1) encodes the inactive set: t_full>=0 <=> active.
    rng = np.random.default_rng(20)
    lst = rng.integers(0, 3, (16, 16)).astype(np.int64)
    mg = rng.random((16, 16)).astype(np.float32) * 1.5
    t, d, act = pp.gt_tie_targets_numpy(lst, mg, band=1.0)
    assert np.array_equal(t >= 0.0, act)
    assert np.all((d == 0.0) | (d == 1.0))
    assert t[act].min() >= 0.0 and t[act].max() <= 1.0


def test_gt_tie_targets_none_active_when_no_interclass_edge():
    lst = np.zeros((10, 10), np.int64)  # single class => no straddle
    mg = np.full((10, 10), 0.5, np.float32)
    t, d, act = pp.gt_tie_targets_numpy(lst, mg, band=1.0)
    assert not act.any()
    assert np.all(t == -1.0)


def test_gt_tie_targets_band_gates_by_margin():
    # a genuine inter-class edge but with margins ABOVE the band => not active.
    lst = np.zeros((6, 6), np.int64)
    lst[:, 3:] = 1
    mg = np.full((6, 6), 5.0, np.float32)  # all margins >> band
    _, _, act_hi = pp.gt_tie_targets_numpy(lst, mg, band=1.0)
    assert not act_hi.any()
    mg_lo = np.full((6, 6), 0.3, np.float32)
    _, _, act_lo = pp.gt_tie_targets_numpy(lst, mg_lo, band=1.0)
    assert act_lo.any()


# --------------------------------------------------------------------------- #
# 1. tie coordinate t_wit = Mw[p] / (Mw[p] + Mw[q])                           #
# --------------------------------------------------------------------------- #
def test_tie_coordinate_numpy_mlx_bit_parity():
    rng = np.random.default_rng(0)
    signed = rng.standard_normal((1, 12, 16)).astype(np.float32)
    dirm = (rng.random((1, 12, 16)) < 0.5).astype(np.float32)
    t_np = pp.witness_tie_coordinate_numpy(signed, dirm)
    t_mx = np.asarray(pp.witness_tie_coordinate_mlx(mx.array(signed), mx.array(dirm)))
    assert np.abs(t_np - t_mx).max() == 0.0  # op-for-op identical (both fp32, same ops)


def test_tie_coordinate_range_0_1():
    rng = np.random.default_rng(1)
    signed = rng.standard_normal((1, 20, 20)).astype(np.float32) * 3.0
    dirm = (rng.random((1, 20, 20)) < 0.5).astype(np.float32)
    t = pp.witness_tie_coordinate_numpy(signed, dirm)
    assert t.min() >= 0.0 and t.max() <= 1.0 + 1e-6


def test_tie_coordinate_symmetric_straddle_is_half():
    # Mw[p] == Mw[q] everywhere (constant positive field) => t == 0.5 (up to eps).
    signed = np.full((1, 8, 8), 2.0, np.float32)
    dirm = np.zeros((1, 8, 8), np.float32)  # all-right
    t = pp.witness_tie_coordinate_numpy(signed, dirm, eps=1e-9)
    # interior columns (partner well-defined, not the pad edge)
    assert np.allclose(t[0, :, :-1], 0.5, atol=1e-4)


def test_tie_coordinate_dir_map_selects_partner():
    # Distinct value pattern so RIGHT vs DOWN partner differ; check the selection.
    signed = np.zeros((1, 4, 4), np.float32)
    signed[0, 1, 1] = 1.0  # p
    signed[0, 1, 2] = 3.0  # right partner
    signed[0, 2, 1] = 7.0  # down partner
    t_right = pp.witness_tie_coordinate_numpy(signed, np.zeros((1, 4, 4), np.float32))
    t_down = pp.witness_tie_coordinate_numpy(signed, np.ones((1, 4, 4), np.float32))
    assert abs(float(t_right[0, 1, 1]) - 1.0 / (1.0 + 3.0)) < 1e-5
    assert abs(float(t_down[0, 1, 1]) - 1.0 / (1.0 + 7.0)) < 1e-5


def test_tie_coordinate_negative_signed_is_zero():
    # Mw = relu(signed) = 0 where signed<0 => t = 0 / (0+Mq+eps) -> ~0.
    signed = np.full((1, 6, 6), -5.0, np.float32)
    t = pp.witness_tie_coordinate_numpy(signed, np.zeros((1, 6, 6), np.float32))
    assert np.abs(t).max() < 1e-4


def test_tie_coordinate_mlx_differentiable_in_signed():
    signed = mx.array(np.random.default_rng(2).standard_normal((1, 8, 8)).astype(np.float32))
    dirm = mx.zeros((1, 8, 8))

    def _loss(s):
        return mx.sum(pp.witness_tie_coordinate_mlx(s, dirm))

    g = mx.grad(_loss)(signed)
    mx.eval(g)
    assert np.isfinite(np.asarray(g)).all()
    assert np.abs(np.asarray(g)).max() > 0.0  # non-trivial gradient


# --------------------------------------------------------------------------- #
# 2. A_ξ tie-field advection (shared ground-homography warp)                   #
# --------------------------------------------------------------------------- #
def _geom(h=48, w=64):
    g = GroundHomographyGeom.eon(native_hw=(h, w), pitch=-0.01)
    return g, g.mlx()


def _smooth_field(h=48, w=64):
    # A SMOOTH [0,1] field is the representative tie/boundary-phase input class. (Random noise
    # is a warp-parity worst case: any sub-pixel fp32/fp64 geometry difference samples wildly
    # different neighbours — it measures grid-sampling noise, not the primitive. The real tie
    # field is a smooth boundary phase, so parity is tested on the input class it actually sees.)
    yy, xx = np.mgrid[0:h, 0:w]
    return ((0.5 + 0.5 * np.sin(xx / 9.0) * np.cos(yy / 7.0)) * 0.9 + 0.05).astype(np.float32)[None]


def test_advect_xi_zero_is_identity_numpy():
    g, _ = _geom()
    fld = _smooth_field()
    out = pp.advect_tie_field_numpy(fld, np.zeros(6, np.float32), g)
    assert np.abs(out - fld).max() < 1e-5  # numpy fp64 geometry => exact identity at xi=0


def test_advect_xi_zero_is_identity_mlx():
    g, gm = _geom()
    fld = _smooth_field()
    out = np.asarray(pp.advect_tie_field_mlx(mx.array(fld), mx.zeros(6), gm))
    # MLX uses fp32 geometry => identity holds within the fp32 warp floor (numpy is authority).
    assert np.abs(out - fld).max() < 5e-3


def test_advect_nonzero_numpy_mlx_parity_fraction():
    # numpy oracle = fp64 geometry (AUTHORITY); MLX = fp32-GPU. On the representative smooth tie
    # field the warp module's parity floor holds: >=0.999 of pixels within 0.02, mean << 1e-2.
    g, gm = _geom()
    fld = _smooth_field()
    xi = np.array([0.012, 0.0, 0.18, 0.0, 0.001, 0.0], np.float32)
    a_np = pp.advect_tie_field_numpy(fld, xi, g)
    a_mx = np.asarray(pp.advect_tie_field_mlx(mx.array(fld), mx.array(xi), gm))
    frac_ok = float((np.abs(a_np - a_mx) < 0.02).mean())
    assert frac_ok >= 0.999
    assert np.abs(a_np - a_mx).mean() < 2e-3


def test_advect_preserves_unit_range():
    # bilinear interpolation of a [0,1] field (+ persist fallback of a [0,1] source) stays [0,1].
    g, _ = _geom()
    fld = _smooth_field()
    xi = np.array([0.02, 0.0, 0.3, 0.0, 0.002, 0.0], np.float32)
    out = pp.advect_tie_field_numpy(fld, xi, g)
    assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6


def test_advect_accepts_hw_and_1hw():
    g, _ = _geom(12, 12)
    fld_hw = np.random.default_rng(7).random((12, 12)).astype(np.float32)
    out_hw = pp.advect_tie_field_numpy(fld_hw, np.zeros(6), g)
    out_1hw = pp.advect_tie_field_numpy(fld_hw[None], np.zeros(6), g)
    assert out_hw.shape == (12, 12) and out_1hw.shape == (1, 12, 12)
    assert np.allclose(out_hw, out_1hw[0])


# --------------------------------------------------------------------------- #
# 3. cross-scored-frame screw interpolation (se(3) composition)               #
# --------------------------------------------------------------------------- #
def test_cross_screw_both_zero_is_zero():
    xc = pp.cross_scored_frame_xi_interp(np.zeros(6), np.zeros(6))
    assert np.abs(xc).max() < 1e-9


def test_cross_screw_b_zero_is_half_a():
    # b=0 => xi_gap=0.5a, T_cross = exp(0)·exp(0.5a) = exp(0.5a) => cross = 0.5a.
    a = np.array([0.01, 0.0, 0.2, 0.0, 0.001, 0.0])
    xc = pp.cross_scored_frame_xi_interp(a, np.zeros(6))
    assert np.allclose(xc, 0.5 * a, atol=1e-6)


def test_cross_screw_finite_and_shape():
    rng = np.random.default_rng(8)
    poses = rng.standard_normal((2, 6)) * 0.01
    xa = xi_from_pose_calibration(poses, -0.003, 0.0, -0.01)
    xb = xi_from_pose_calibration(poses * 1.2, -0.003, 0.0, -0.01)
    xc = pp.cross_scored_frame_xi_interp(xa, xb)
    assert xc.shape == (6,) and np.isfinite(xc).all()


def test_cross_screw_is_se3_compose_not_linear_sum():
    # For a rotation-carrying screw the group composition differs from the linear-twist sum
    # 0.5*a + 1.5*b (BCH second-order terms) — proves proper se(3) composition is used.
    a = np.array([0.05, 0.02, 0.30, 0.10, 0.05, 0.02])
    b = np.array([0.04, 0.03, 0.28, 0.08, 0.06, 0.03])
    xc = pp.cross_scored_frame_xi_interp(a, b)
    linear = 0.5 * a + 1.5 * b
    assert np.abs(xc - linear).max() > 1e-4


# --------------------------------------------------------------------------- #
# 4. the T1 residual reducer                                                  #
# --------------------------------------------------------------------------- #
def test_residual_zero_on_identical():
    rng = np.random.default_rng(9)
    fld = rng.random((1, 10, 10)).astype(np.float32)
    w = np.ones((1, 10, 10), np.float32)
    assert pp.phase_advection_weighted_mse_numpy(fld, fld, w) == 0.0


def test_residual_zero_weight_is_no_op():
    # default-OFF / no-annulus => weight all-zero => term is exactly 0 (byte-safe skip).
    rng = np.random.default_rng(10)
    a = rng.random((1, 10, 10)).astype(np.float32)
    b = rng.random((1, 10, 10)).astype(np.float32)
    w = np.zeros((1, 10, 10), np.float32)
    assert pp.phase_advection_weighted_mse_numpy(a, b, w) == 0.0


def test_residual_positive_on_phase_mismatch():
    rng = np.random.default_rng(11)
    a = rng.random((1, 10, 10)).astype(np.float32)
    w = np.ones((1, 10, 10), np.float32)
    assert pp.phase_advection_weighted_mse_numpy(a, 1.0 - a, w) > 0.0


def test_residual_numpy_mlx_parity():
    rng = np.random.default_rng(12)
    a = rng.random((1, 12, 12)).astype(np.float32)
    b = rng.random((1, 12, 12)).astype(np.float32)
    w = (rng.random((1, 12, 12)) < 0.3).astype(np.float32)
    r_np = pp.phase_advection_weighted_mse_numpy(a, b, w)
    r_mx = float(np.asarray(pp.phase_advection_weighted_mse_mlx(mx.array(a), mx.array(b), mx.array(w))))
    assert abs(r_np - r_mx) < 1e-5


def test_residual_annulus_weight_restricts_support():
    # A mismatch OUTSIDE the annulus weight must not contribute to the term.
    a = np.zeros((1, 6, 6), np.float32)
    b = np.zeros((1, 6, 6), np.float32)
    b[0, 0, 0] = 1.0  # mismatch at a single pixel
    w_out = np.ones((1, 6, 6), np.float32)
    w_out[0, 0, 0] = 0.0  # exclude that pixel from the annulus
    assert pp.phase_advection_weighted_mse_numpy(a, b, w_out) == 0.0
    w_in = np.zeros((1, 6, 6), np.float32)
    w_in[0, 0, 0] = 1.0  # include only that pixel
    assert pp.phase_advection_weighted_mse_numpy(a, b, w_in) == pytest.approx(1.0, abs=1e-4)


def test_residual_mlx_differentiable_in_t_wit():
    rng = np.random.default_rng(13)
    t_wit = mx.array(rng.random((1, 10, 10)).astype(np.float32))
    t_ref = mx.array(rng.random((1, 10, 10)).astype(np.float32))
    w = mx.array(np.ones((1, 10, 10), np.float32))

    def _loss(t):
        return pp.phase_advection_weighted_mse_mlx(t, t_ref, w)

    g = mx.grad(_loss)(t_wit)
    mx.eval(g)
    assert np.isfinite(np.asarray(g)).all()
    assert np.abs(np.asarray(g)).max() > 0.0


def test_residual_shrinks_as_t_wit_approaches_ref():
    # A descent step on t_wit toward the (advected) reference must LOWER the residual —
    # the core T1 dynamics: the witness phase is pulled onto the ξ-advected trajectory.
    rng = np.random.default_rng(14)
    t_ref = rng.random((1, 10, 10)).astype(np.float32)
    t0 = t_ref + 0.3
    w = np.ones((1, 10, 10), np.float32)
    r0 = pp.phase_advection_weighted_mse_numpy(t0, t_ref, w)
    t1 = t_ref + 0.15  # one shrink step toward the reference
    r1 = pp.phase_advection_weighted_mse_numpy(t1, t_ref, w)
    assert r1 < r0


# --------------------------------------------------------------------------- #
# 5. the T1 PROVIDER RECIPE (the composition the trainer inlines):             #
#    GT tie -> cross-ξ -> advect prev into p's frame -> annulus∩ground weight  #
# --------------------------------------------------------------------------- #
def _provider_ref(lst_prev, mg_prev, lst_p, mg_p, xi_pair_prev, xi_pair_p, geom, band, ground, eps=1e-6):
    """Replicate the trainer's T1 provider recipe for one pair p (>=1). Returns
    (ref, wmask, dir_p) as the trainer builds them."""
    t_prev, _, a_prev = pp.gt_tie_targets_numpy(lst_prev, mg_prev, band=band, eps=eps)
    _, dir_p, _ = pp.gt_tie_targets_numpy(lst_p, mg_p, band=band, eps=eps)
    xi_cross = pp.cross_scored_frame_xi_interp(xi_pair_prev, xi_pair_p)
    val_prev = np.where(t_prev >= 0.0, t_prev, 0.0).astype(np.float32)
    ref_w = pp.advect_tie_field_numpy(val_prev, xi_cross, geom)  # (H,W) in -> (H,W) out
    act_w = pp.advect_tie_field_numpy(a_prev.astype(np.float32), xi_cross, geom)
    ref_active = act_w >= 0.5
    ref = np.where(ref_active, ref_w, -1.0).astype(np.float32)
    ann = mg_p < band
    grnd = np.isin(lst_p, list(ground))
    wmask = (ann & grnd & ref_active).astype(np.float32)
    return ref, wmask, dir_p


def test_provider_recipe_zero_residual_on_xi_consistent_boundary():
    # If the witness EXACTLY reproduces the ξ-advected reference on the weighted set, the term is 0
    # (the negative: a ξ-consistent boundary contributes no phase penalty).
    g, _ = _geom(48, 64)
    lst_prev = (np.mgrid[0:48, 0:64][1] // 32).astype(np.int64)  # 2 ground classes, a vertical edge
    lst_p = lst_prev.copy()
    mg = np.full((48, 64), 0.4, np.float32)  # inside the band
    xi = np.zeros(6)  # xi=0 => advection is identity => ref == prev GT tie
    ref, wmask, dir_p = _provider_ref(lst_prev, mg, lst_p, mg, xi, xi, g, band=2.0, ground={0, 1})
    # build t_wit that equals the reference where weighted (signed chosen so t_wit == ref there).
    # Directly compare the reference to itself through the residual: zero.
    assert pp.phase_advection_weighted_mse_numpy(ref, ref, wmask) == 0.0
    assert wmask.sum() > 0  # the weighted set is non-empty (there IS an active advected boundary)


def test_provider_recipe_positive_on_phase_shift():
    g, _ = _geom(48, 64)
    lst = (np.mgrid[0:48, 0:64][1] // 32).astype(np.int64)
    mg = np.full((48, 64), 0.4, np.float32)
    ref, wmask, _ = _provider_ref(lst, mg, lst, mg, np.zeros(6), np.zeros(6), g, band=2.0, ground={0, 1})
    # a witness whose phase is shifted by 0.25 everywhere the ref is defined => positive residual.
    t_wit = np.where(wmask > 0, np.clip(ref + 0.25, 0, 1), 0.0).astype(np.float32)
    assert pp.phase_advection_weighted_mse_numpy(t_wit, ref, wmask) > 0.0


def test_provider_recipe_ground_class_restriction():
    # A non-ground class (3=Movable) must be EXCLUDED from the weight even at an active boundary.
    g, _ = _geom(48, 64)
    col = np.mgrid[0:48, 0:64][1]
    lst = np.where(col < 32, 0, 3).astype(np.int64)  # Road | Movable edge
    mg = np.full((48, 64), 0.4, np.float32)
    _, wmask, _ = _provider_ref(lst, mg, lst, mg, np.zeros(6), np.zeros(6), g, band=2.0, ground={0, 1, 2})
    # every weighted pixel must be a ground-class pixel (class 3 excluded).
    assert np.all(lst[wmask > 0.5] != 3)


def test_provider_pair0_is_no_op():
    # Pair 0 has no prior scored frame => the trainer builds an all-zero weight => term is 0.
    wmask0 = np.zeros((1, 12, 12), np.float32)
    a = np.random.default_rng(31).random((1, 12, 12)).astype(np.float32)
    assert pp.phase_advection_weighted_mse_numpy(a, 1.0 - a, wmask0) == 0.0
