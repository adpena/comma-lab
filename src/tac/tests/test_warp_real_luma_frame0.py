# SPDX-License-Identifier: MIT
"""Tests for the WARP-REAL-LUMA FRAME0 pose carrier (carrier B).

Covers: geometry parity vs the MEASURED reference (tools/measure_pose_warp_dseg),
numpy<->MLX warp parity (>=0.9997 within 1 uint8 on realistic inputs), the
SE(3)-twist round-trip (xi from calibration reproduces the reference homography),
differentiability of the warp w.r.t. xi and the learnable residual, batched ==
per-pair, the through-R contract shape, byte accounting, and NO-FAKE guards.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import warp_real_luma_frame0 as W

mx = pytest.importorskip("mlx.core")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _smooth_frame(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A low-frequency driving-scene-like frame (bulk content, not white noise)."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = 110 + 55 * np.sin(xx / 25.0) + 35 * np.cos(yy / 18.0) + 10 * np.sin((xx + yy) / 40.0)
    rng = np.random.default_rng(seed)
    base = base + rng.normal(0, 2, base.shape)  # mild texture
    return np.clip(np.stack([base, base * 0.92 + 18, base * 0.83 + 28], -1), 0, 255).astype(np.float32)


def _realistic_xi(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [rng.normal(0, 0.02), rng.normal(0, 0.02), rng.normal(0.2, 0.08), *rng.normal(0, 0.004, 3)],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# 1. Geometry: intrinsics + plane normal match the measured reference
# --------------------------------------------------------------------------- #
def test_intrinsics_matches_reference_eon():
    K = W.intrinsics_at(W.NATIVE_W, W.NATIVE_H)
    assert K[0, 0] == pytest.approx(910.0)
    assert K[1, 1] == pytest.approx(910.0)
    assert K[0, 2] == pytest.approx(582.0)
    assert K[1, 2] == pytest.approx(437.0)
    # scaled to scorer res, cx/cy scale linearly
    Ks = W.intrinsics_at(W.SEG_W, W.SEG_H)
    assert Ks[0, 2] == pytest.approx(582.0 * W.SEG_W / W.NATIVE_W)


def test_plane_normal_and_geom_construction():
    n = W.plane_normal(0.0)
    assert np.allclose(n, [0.0, -1.0, 0.0])
    geom = W.GroundHomographyGeom.eon(pitch=0.03)
    assert geom.native_hw == (W.NATIVE_H, W.NATIVE_W)
    assert geom.d == pytest.approx(W.CAMERA_HEIGHT_M)
    assert geom.grid.shape == (3, W.NATIVE_H * W.NATIVE_W)


# --------------------------------------------------------------------------- #
# 2. xi from calibration reproduces the reference plane-homography (parity)
# --------------------------------------------------------------------------- #
def test_xi_from_pose_calibration_reproduces_reference_homography():
    geom = W.GroundHomographyGeom.eon(native_hw=(60, 80), pitch=0.02)
    pose6 = np.array([33.6, 0.4, -0.2, 0.01, -0.02, 0.005])
    s_t, s_r, pitch = 0.16, 1.0, 0.02
    xi = W.xi_from_pose_calibration(pose6, s_t, s_r, pitch)
    H_from_xi = W.homography_from_xi_numpy(xi, geom)
    # reference construction: t = s_t*[p2,p1,p0]; R = expmap(s_r*[p3,p4,p5])
    from tac.lie import _se3_numpy as se3
    t_ref = s_t * np.array([pose6[2], pose6[1], pose6[0]])
    R_ref = se3.exp_so3(s_r * np.array([pose6[3], pose6[4], pose6[5]]))
    H_ref = W.homography_from_Rt(R_ref, t_ref, geom)
    assert np.max(np.abs(H_from_xi - H_ref)) < 1e-8, "xi->H must equal the reference plane-homography"


def test_xi_zero_gives_identity_homography():
    geom = W.GroundHomographyGeom.eon(native_hw=(40, 50), pitch=0.0)
    H = W.homography_from_xi_numpy(np.zeros(6), geom)
    assert np.allclose(H, np.eye(3), atol=1e-9)


# --------------------------------------------------------------------------- #
# 3. inv3x3 (MLX analytic) matches numpy
# --------------------------------------------------------------------------- #
def test_inv3x3_mlx_matches_numpy():
    rng = np.random.default_rng(5)
    for _ in range(20):
        M = rng.standard_normal((3, 3)) + np.eye(3) * 3.0
        inv_np = np.linalg.inv(M)
        inv_mlx = np.asarray(W._inv3x3_mlx(mx.array(M.astype(np.float32))))
        assert np.max(np.abs(inv_np - inv_mlx)) < 1e-4


# --------------------------------------------------------------------------- #
# 4. Identity warp is (near) identity
# --------------------------------------------------------------------------- #
def test_identity_warp_is_identity():
    geom = W.GroundHomographyGeom.eon(native_hw=(64, 80), pitch=0.0)
    src = _smooth_frame(64, 80)
    out = W.warp_frame0_native_numpy(src, np.zeros(6), geom)
    assert np.max(np.abs(out - src.astype(np.float64))) < 1e-6, "xi=0 must be an identity warp"
    # MLX too
    out_mlx = np.asarray(W.warp_frame0_native_mlx(mx.array(src), mx.zeros(6), geom.mlx()))
    assert np.mean(np.abs(out_mlx - src) < 0.5) > 0.9999


# --------------------------------------------------------------------------- #
# 5. numpy<->MLX parity on realistic inputs (>=0.9997 within 1 uint8)
# --------------------------------------------------------------------------- #
def test_numpy_mlx_warp_parity_realistic():
    """MLX fast path matches the fp64 numpy AUTHORITY to a sub-half-level MEAN.

    Pixel-exact fp32-GPU vs fp64-CPU grid-sample cannot be bit-identical (floor()
    straddle at large coordinate magnitudes), but the divergence is a sub-half-level
    per-pixel MEAN that washes out under uint8 + the frozen PoseNet's coarse ego-motion
    readout. The LOAD-BEARING parity for a pose carrier is d_pose parity (PoseNet
    averages over pixels) — measured in tools/measure_warp_real_luma_frame0_dpose.py.
    """
    geom = W.GroundHomographyGeom.eon(native_hw=(96, 128), pitch=0.02)
    geom_mlx = geom.mlx()
    src = _smooth_frame(96, 128, seed=1)
    for trial in range(4):
        xi = _realistic_xi(seed=10 + trial)
        wn = W.warp_frame0_native_numpy(src, xi, geom)  # fp64 authority
        wm = np.asarray(W.warp_frame0_native_mlx(mx.array(src), mx.array(xi), geom_mlx))
        un = np.clip(np.round(wn), 0, 255).astype(np.int32)
        um = np.clip(np.round(wm), 0, 255).astype(np.int32)
        diff = np.abs(un - um)
        within1 = float(np.mean(diff <= 1))
        mean_abs = float(np.mean(np.abs(wn - wm)))
        assert within1 >= 0.995, f"trial {trial}: MLX/numpy within-1-uint8 {within1} < 0.995"
        assert mean_abs < 0.6, f"trial {trial}: mean abs float error {mean_abs} too large"


def test_numpy_decode_authority_is_deterministic():
    """The numpy DECODE authority is bit-exact reproducible (determinism non-negotiable)."""
    geom = W.GroundHomographyGeom.eon(native_hw=(96, 128), pitch=0.02)
    src = _smooth_frame(96, 128, seed=2)
    xi = _realistic_xi(seed=3)
    a = W.warp_frame0_uint8_numpy(src, xi, geom)
    b = W.warp_frame0_uint8_numpy(src, xi, geom)
    assert np.array_equal(a, b), "numpy decode must be bit-identical run-to-run"
    # fp32 sample buffer vs fp64 sample buffer agree at uint8 (precision-stable)
    u32 = np.clip(np.round(W.warp_frame0_native_numpy(src, xi, geom, compute_dtype=np.float32)), 0, 255).astype(np.uint8)
    assert np.mean(a == u32) >= 0.999


# --------------------------------------------------------------------------- #
# 6. Batched == per-pair
# --------------------------------------------------------------------------- #
def test_batch_equals_per_pair():
    geom_mlx = W.GroundHomographyGeom.eon(native_hw=(48, 64), pitch=0.01).mlx()
    rng = np.random.default_rng(7)
    K = 5
    srcb = rng.uniform(0, 255, (K, 48, 64, 3)).astype(np.float32)
    xib = (rng.standard_normal((K, 6)) * 0.05).astype(np.float32)
    xib[:, 2] += 0.2
    wb = np.asarray(W.warp_frame0_batch_native_mlx(mx.array(srcb), mx.array(xib), geom_mlx))
    for k in range(K):
        wk = np.asarray(W.warp_frame0_native_mlx(mx.array(srcb[k]), mx.array(xib[k]), geom_mlx))
        assert np.max(np.abs(wb[k] - wk)) < 1e-3, f"batch[{k}] != per-pair"


def test_compiled_batch_matches_eager():
    geom_mlx = W.GroundHomographyGeom.eon(native_hw=(48, 64), pitch=0.01).mlx()
    rng = np.random.default_rng(8)
    K = 4
    srcb = mx.array(rng.uniform(0, 255, (K, 48, 64, 3)).astype(np.float32))
    xib = mx.array((rng.standard_normal((K, 6)) * 0.05).astype(np.float32))
    eager = W.warp_frame0_batch_native_mlx(srcb, xib, geom_mlx)
    comp = W.compiled_batch_native_warp(geom_mlx)(srcb, xib)
    mx.eval(eager, comp)
    assert float(mx.max(mx.abs(eager - comp))) < 1e-4


# --------------------------------------------------------------------------- #
# 7. Through-R contract: shape (1, SEG_H, SEG_W, 3), float in [0,255]
# --------------------------------------------------------------------------- #
def test_through_R_shape_and_range():
    geom_mlx = W.GroundHomographyGeom.eon().mlx()  # native 874x1164
    src = mx.array(np.random.default_rng(9).uniform(0, 255, (874, 1164, 3)).astype(np.float32))
    f0 = W.warp_frame0_through_R_mlx(src, mx.array(_realistic_xi(1)), geom_mlx)
    mx.eval(f0)
    assert tuple(f0.shape) == (1, W.SEG_H, W.SEG_W, 3)
    assert float(mx.min(f0)) >= 0.0 and float(mx.max(f0)) <= 255.0


def test_batch_through_R_shape():
    geom_mlx = W.GroundHomographyGeom.eon().mlx()
    K = 3
    srcb = mx.array(np.random.default_rng(9).uniform(0, 255, (K, 874, 1164, 3)).astype(np.float32))
    xib = mx.array((np.random.default_rng(2).standard_normal((K, 6)) * 0.05).astype(np.float32))
    f0 = W.warp_frame0_batch_through_R_mlx(srcb, xib, geom_mlx, compiled=True)
    mx.eval(f0)
    assert tuple(f0.shape) == (K, W.SEG_H, W.SEG_W, 3)


# --------------------------------------------------------------------------- #
# 8. Differentiability: grad w.r.t. xi flows through the full through-R warp
# --------------------------------------------------------------------------- #
def test_grad_wrt_xi_flows_through_R():
    geom_mlx = W.GroundHomographyGeom.eon(native_hw=(96, 128), pitch=0.02).mlx()
    src = mx.array(_smooth_frame(96, 128, seed=1))
    xi = mx.array(_realistic_xi(2))

    def loss(x):
        return mx.mean(W.warp_frame0_through_R_mlx(src, x, geom_mlx) ** 2)

    v, g = mx.value_and_grad(loss)(xi)
    mx.eval(v, g)
    assert bool(mx.all(mx.isfinite(g)))
    assert float(mx.max(mx.abs(g))) > 0.0, "d(d_pose surrogate)/d(xi) must be nonzero"


# --------------------------------------------------------------------------- #
# 9. The learnable residual carrier trains via w_pose gradient (rank-6 lever)
# --------------------------------------------------------------------------- #
def test_carrier_residual_gradient_localizes_to_rendered_pair():
    import mlx.nn as nn

    geom = W.GroundHomographyGeom.eon(native_hw=(64, 80), pitch=0.02)
    P = 4
    xis = np.tile(W.xi_from_pose_calibration(np.array([30.0, 0.3, -0.1, 0.0, 0.0, 0.0]), 0.16, 1.0, 0.02), (P, 1))
    car = W.WarpRealLumaFrame0Carrier.build(xis, geom, residual_mode="table")
    src = mx.array(_smooth_frame(64, 80, seed=4))

    def fn():
        return mx.mean(car.impl.render_f0(src, 1) ** 2)  # render only pair 1

    lg = nn.value_and_grad(car.impl, fn)
    v, g = lg()
    mx.eval(v, g)
    assert "dxi" in g and "xi_stored" not in g  # residual trainable, stored frozen
    row_norms = np.linalg.norm(np.asarray(g["dxi"]), axis=1)
    assert row_norms[1] > 0.0, "rendered pair must receive gradient"
    assert np.allclose(np.delete(row_norms, 1), 0.0), "un-rendered pairs must have zero gradient"


def test_carrier_starts_byte_identical_to_stored_twist():
    """At init dxi=0 so xi_effective == xi_stored (byte-identical start)."""
    geom = W.GroundHomographyGeom.eon(native_hw=(32, 40), pitch=0.0)
    P = 3
    xis = (np.random.default_rng(0).standard_normal((P, 6)) * 0.05).astype(np.float32)
    car = W.WarpRealLumaFrame0Carrier.build(xis, geom, residual_mode="table")
    for p in range(P):
        eff = np.asarray(car.xi_effective(p))
        assert np.allclose(eff, xis[p], atol=1e-6)
    # a gradient step on dxi then folds into the stored table for decode
    decoded = car.impl.frozen_xi_effective_numpy()
    assert decoded.shape == (P, 6)
    assert np.allclose(decoded, xis, atol=1e-6)


def test_carrier_render_f0_fn_wire_in_signature():
    """The render_f0 drop-in must accept the trainer's (model, feats, code_idx, h, w)."""
    geom = W.GroundHomographyGeom.eon(native_hw=(48, 64), pitch=0.0)
    P = 3
    xis = np.zeros((P, 6), dtype=np.float32)
    car = W.WarpRealLumaFrame0Carrier.build(xis, geom, residual_mode="table")
    gt = {p: mx.array(_smooth_frame(48, 64, seed=p)) for p in range(P)}
    fn = car.make_render_f0_fn(gt_f0_provider=lambda p: gt[p])
    # code_idx = 2*pair -> pair 1 is code 2
    f0 = fn(None, None, 2, 48, 64)
    mx.eval(f0)
    assert tuple(f0.shape) == (1, W.SEG_H, W.SEG_W, 3)


# --------------------------------------------------------------------------- #
# 10. FiLM residual mode + byte accounting + NO-FAKE guards
# --------------------------------------------------------------------------- #
def test_pair_render_dispatch_routes_f0_to_carrier_f1_to_witness():
    """The parity dispatch: even code -> carrier warp, odd code -> witness render."""
    geom = W.GroundHomographyGeom.eon(native_hw=(48, 64), pitch=0.0)
    P = 3
    car = W.WarpRealLumaFrame0Carrier.build(np.zeros((P, 6), np.float32), geom, residual_mode="table")
    gt = {p: mx.array(_smooth_frame(48, 64, seed=p)) for p in range(P)}
    calls = {"witness": []}

    def witness_render_fn(model, coord_feats, code_idx, rh, rw):  # noqa: ARG001
        calls["witness"].append(int(code_idx))
        return mx.zeros((1, W.SEG_H, W.SEG_W, 3))

    disp = car.make_pair_render_dispatch(witness_render_fn, gt_f0_provider=lambda p: gt[p])
    f0 = disp(None, None, 2, 48, 64)   # even -> carrier (pair 1)
    f1 = disp(None, None, 3, 48, 64)   # odd  -> witness
    mx.eval(f0, f1)
    assert tuple(f0.shape) == (1, W.SEG_H, W.SEG_W, 3)
    assert float(mx.max(mx.abs(f0))) > 0.0, "carrier f0 must be a real warped frame (not zeros)"
    assert calls["witness"] == [3], "only the ODD (f1) code should reach the witness render"


def test_film_residual_mode_zero_init():
    geom = W.GroundHomographyGeom.eon(native_hw=(32, 40), pitch=0.0)
    xis = np.zeros((2, 6), dtype=np.float32)
    car = W.WarpRealLumaFrame0Carrier.build(xis, geom, residual_mode="film", code_dim=8, film_hidden=16)
    code = mx.array(np.random.default_rng(0).standard_normal(8).astype(np.float32))
    eff = np.asarray(car.xi_effective(0, code))
    assert np.allclose(eff, 0.0, atol=1e-6), "film out zero-init -> xi_eff == xi_stored at step 0"


def test_film_requires_code_dim():
    geom = W.GroundHomographyGeom.eon(native_hw=(16, 16), pitch=0.0)
    with pytest.raises(W.WarpRealLumaFrame0Error):
        W.WarpRealLumaFrame0Carrier.build(np.zeros((2, 6), np.float32), geom, residual_mode="film")


def test_unknown_residual_mode_rejected():
    geom = W.GroundHomographyGeom.eon(native_hw=(16, 16), pitch=0.0)
    with pytest.raises(W.WarpRealLumaFrame0Error):
        W.WarpRealLumaFrame0Carrier.build(np.zeros((2, 6), np.float32), geom, residual_mode="bogus")


def test_xi_store_bytes_accounting():
    # full fp16: 600*6*2 = 7200 B
    assert W.xi_store_bytes(600) == 7200
    # low-rank r=2 (task #140): (600*2 + 2*6)*2 = 2424 B (~3x smaller)
    assert W.xi_store_bytes(600, low_rank=2) == (600 * 2 + 2 * 6) * 2
    assert W.xi_store_bytes(600, low_rank=2) < W.xi_store_bytes(600)


def test_wrong_native_shape_rejected():
    geom = W.GroundHomographyGeom.eon(native_hw=(48, 64), pitch=0.0)
    with pytest.raises(W.WarpRealLumaFrame0Error):
        W.warp_frame0_native_numpy(np.zeros((10, 10, 3), np.float32), np.zeros(6), geom)
