# SPDX-License-Identifier: MIT
"""Parity-gated tests for the ``tac.lie`` MLX se(3)/SE(3) library.

Three-tier authority per the deterministic-reproducibility non-negotiable:
NumPy-fp64 ``golden`` -> NumPy-fp32 ``authority`` -> MLX-fp32 ``fast path``.
The MLX path must match the NumPy oracle within fp32 tolerance; algebraic
identities are device-agnostic; the Barfoot Q-matrix is verified two
independent ways (series + finite-difference); gradients must stay finite at
``theta -> 0``. No score claim, no MPS; pure geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from tac.lie import _se3_numpy as N
from tac.lie import screw_blend as SB
from tac.lie import se3 as S
from tac.lie import se3_bspline as BS
from tac.lie import so3 as SO

# CPU-only parity: the MLX GPU backend uses fast-math transcendentals (~1e-3
# relative drift; cf. MLX issue #2205 per-chip variation, documented in the
# repo's metal kernels). The numpy-fp32 authority parity is asserted on
# MLX-CPU, which matches numpy fp32 to fp32 round-off.
mx.set_default_device(mx.cpu)

_RNG = np.random.default_rng(20260630)
# fp32 parity tolerances (MLX-fp32 vs NumPy oracle).
_TOL = 2e-5  # SO(3)/SE(3) exp/log/Jacobians/Adjoint
_TOL_Q = 2e-4  # full 6x6 Jacobian (Q is a quartic in skew products -> larger fp32 drift)


# --------------------------------------------------------------------------- #
# sampling helpers
# --------------------------------------------------------------------------- #
def _rand_omega(n, theta=None):
    w = _RNG.standard_normal((n, 3))
    if theta is not None:
        nrm = np.linalg.norm(w, axis=-1, keepdims=True)
        nrm = np.where(nrm < 1e-12, 1.0, nrm)
        w = w / nrm * theta
    return w


def _rand_xi(n, scale=0.4, theta=None):
    return np.concatenate([_RNG.standard_normal((n, 3)) * scale, _rand_omega(n, theta)], axis=-1)


def _rand_T(n, scale=0.4):
    return N.exp_se3(_rand_xi(n, scale=scale))


def _theta_grid():
    # log-spaced over [1e-9, pi-1e-3] (hammer the small-angle branch) + exact 0.
    return list(np.geomspace(1e-9, np.pi - 1e-3, 12)) + [0.0]


def _to32(a):
    return mx.array(np.asarray(a, dtype=np.float32))


def _np32(a):
    return np.asarray(a, dtype=np.float32)


def _maxabs(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


# --------------------------------------------------------------------------- #
# convention
# --------------------------------------------------------------------------- #
def test_convention_constants_match():
    assert N.CONVENTION == SO.CONVENTION == S.CONVENTION == "translation_first_(rho,omega)"


def test_translation_first_block_order():
    # rho drives translation with NO rotation; omega drives rotation.
    xi = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])
    T = N.exp_se3(xi)
    assert _maxabs(N.translation_of(T), xi[..., :3]) < 1e-12
    assert _maxabs(N.rotation_of(T), np.eye(3)) < 1e-12


# --------------------------------------------------------------------------- #
# hat / vee
# --------------------------------------------------------------------------- #
def test_skew_unskew_roundtrip_and_parity():
    v = _RNG.standard_normal((8, 3))
    assert _maxabs(N.unskew(N.skew(v)), v) < 1e-12
    assert _maxabs(np.asarray(SO.skew(_to32(v))), N.skew(_np32(v))) < _TOL


# --------------------------------------------------------------------------- #
# SO(3) exp/log parity across the theta grid
# --------------------------------------------------------------------------- #
def test_exp_so3_parity_across_theta():
    worst = 0.0
    for th in _theta_grid():
        w = _rand_omega(8, theta=th)
        worst = max(worst, _maxabs(np.asarray(SO.exp_so3(_to32(w))), N.exp_so3(_np32(w))))
    assert worst < _TOL, worst


def test_log_so3_parity_across_theta():
    worst = 0.0
    for th in _theta_grid():
        R = N.exp_so3(_rand_omega(8, theta=th))
        worst = max(worst, _maxabs(np.asarray(SO.log_so3(_to32(R))), N.log_so3(_np32(R))))
    assert worst < _TOL, worst


def test_exp_se3_parity_across_theta():
    worst = 0.0
    for th in _theta_grid():
        xi = np.concatenate([_RNG.standard_normal((8, 3)) * 0.5, _rand_omega(8, theta=th)], -1)
        worst = max(worst, _maxabs(np.asarray(S.exp_se3(_to32(xi))), N.exp_se3(_np32(xi))))
    assert worst < _TOL, worst


def test_log_se3_parity_across_theta():
    worst = 0.0
    for th in _theta_grid():
        xi = np.concatenate([_RNG.standard_normal((8, 3)) * 0.5, _rand_omega(8, theta=th)], -1)
        T = N.exp_se3(xi)
        worst = max(worst, _maxabs(np.asarray(S.log_se3(_to32(T))), N.log_se3(_np32(T))))
    assert worst < _TOL, worst


# --------------------------------------------------------------------------- #
# Jacobians parity
# --------------------------------------------------------------------------- #
def test_left_jacobian_so3_parity():
    w = _rand_omega(16)
    assert _maxabs(np.asarray(SO.left_jacobian_so3(_to32(w))), N.left_jacobian_so3(_np32(w))) < _TOL


def test_right_jacobian_so3_parity_and_identity():
    w = _rand_omega(16)
    # parity
    assert _maxabs(np.asarray(SO.right_jacobian_so3(_to32(w))), N.right_jacobian_so3(_np32(w))) < _TOL
    # J_r(w) == J_l(-w)
    assert _maxabs(N.right_jacobian_so3(w), N.left_jacobian_so3(-w)) < 1e-12


def test_jacobian_inverse_parity():
    w = _rand_omega(16)
    assert _maxabs(np.asarray(SO.left_jacobian_inv_so3(_to32(w))), N.left_jacobian_inv_so3(_np32(w))) < _TOL
    assert _maxabs(np.asarray(SO.right_jacobian_inv_so3(_to32(w))), N.right_jacobian_inv_so3(_np32(w))) < _TOL


def test_jacobian_inverse_products_identity():
    w = _rand_omega(16)
    I = np.eye(3)
    assert _maxabs(N.left_jacobian_so3(w) @ N.left_jacobian_inv_so3(w), I) < 1e-10
    assert _maxabs(N.right_jacobian_so3(w) @ N.right_jacobian_inv_so3(w), I) < 1e-10


def test_left_jacobian_inv_small_angle_parity():
    # explicitly hammer near zero where the D-coefficient Taylor must engage.
    for th in [0.0, 1e-9, 1e-7, 5e-7]:
        w = _rand_omega(8, theta=th)
        assert _maxabs(np.asarray(SO.left_jacobian_inv_so3(_to32(w))), N.left_jacobian_inv_so3(_np32(w))) < _TOL


# --------------------------------------------------------------------------- #
# adjoints parity + identities
# --------------------------------------------------------------------------- #
def test_adjoint_T_parity():
    T = _rand_T(16)
    assert _maxabs(np.asarray(S.adjoint_T(_to32(T))), N.adjoint_T(_np32(T))) < _TOL


def test_adjoint_se3_parity():
    xi = _rand_xi(16)
    assert _maxabs(np.asarray(S.adjoint_se3(_to32(xi))), N.adjoint_se3(_np32(xi))) < _TOL


def test_adjoint_homomorphism():
    A, B = _rand_T(16), _rand_T(16)
    assert _maxabs(N.adjoint_T(N.compose(A, B)), N.adjoint_T(A) @ N.adjoint_T(B)) < 1e-10


def test_adjoint_defining_identity():
    # T exp(xi) T^-1 == exp(Ad_T xi) -- independently verifies the Adjoint block form.
    A = _rand_T(16)
    d = _rand_xi(16, scale=0.2, theta=None)
    d[..., 3:] *= 0.3
    lhs = N.compose(N.compose(A, N.exp_se3(d)), N.inverse(A))
    rhs = N.exp_se3((N.adjoint_T(A) @ d[..., None])[..., 0])
    assert _maxabs(lhs, rhs) < 1e-10


# --------------------------------------------------------------------------- #
# group identities (device-agnostic, on the MLX path)
# --------------------------------------------------------------------------- #
def test_exp_log_roundtrip_se3_mlx():
    xi = _rand_xi(16, scale=0.5)
    xi[..., 3:] *= 0.4  # keep omega well away from pi
    out = np.asarray(S.log_se3(S.exp_se3(_to32(xi))))
    assert _maxabs(out, xi) < 1e-4


def test_log_exp_roundtrip_se3_mlx():
    T = _rand_T(16, scale=0.4)
    rt = np.asarray(S.exp_se3(S.log_se3(_to32(T))))
    assert _maxabs(rt, T) < 1e-4


def test_inverse_identity_mlx():
    T = _to32(_rand_T(16))
    I = np.broadcast_to(np.eye(4), (16, 4, 4))
    assert _maxabs(np.asarray(S.compose(T, S.inverse(T))), I) < 1e-4


def test_exp_neg_equals_inverse():
    xi = _rand_xi(16, scale=0.4)
    xi[..., 3:] *= 0.4
    assert _maxabs(N.exp_se3(-xi), N.inverse(N.exp_se3(xi))) < 1e-10


# --------------------------------------------------------------------------- #
# three-tier oracle (fp64 golden / fp32 authority / mlx fast)
# --------------------------------------------------------------------------- #
def test_three_tier_oracle_exp_se3():
    xi64 = _rand_xi(64, scale=0.5)
    xi64[..., 3:] *= 0.5
    golden = N.exp_se3(xi64.astype(np.float64))
    authority = N.exp_se3(xi64.astype(np.float32))
    fast = np.asarray(S.exp_se3(_to32(xi64)))
    assert _maxabs(authority, golden) < 1e-4  # fp32 numpy vs fp64 numpy
    assert _maxabs(fast, authority) < 1e-5  # mlx fp32 vs numpy fp32
    assert _maxabs(fast, golden) < 1e-4  # mlx fp32 vs fp64 golden


# --------------------------------------------------------------------------- #
# external scipy cross-check (independent oracle)
# --------------------------------------------------------------------------- #
def test_scipy_crosscheck_exp_se3():
    from scipy.spatial.transform import RigidTransform

    xi = _rand_xi(32, scale=0.6)
    xi[..., 3:] *= 0.6
    # scipy is rotation-first [r, v]; ours is translation-first (rho, omega) -> swap blocks
    xi_sp = np.concatenate([xi[..., 3:], xi[..., :3]], -1)
    T_sp = RigidTransform.from_exp_coords(xi_sp).as_matrix()
    assert _maxabs(N.exp_se3(xi), T_sp) < 1e-6


def test_scipy_crosscheck_log_se3():
    from scipy.spatial.transform import RigidTransform

    T = _rand_T(32, scale=0.5)
    xi_sp = RigidTransform.from_matrix(T).as_exp_coords()  # rotation-first
    xi_ours = N.log_se3(T)
    xi_ours_sp = np.concatenate([xi_ours[..., 3:], xi_ours[..., :3]], -1)
    assert _maxabs(xi_ours_sp, xi_sp) < 1e-6


# --------------------------------------------------------------------------- #
# Barfoot Q-matrix verification (the flagged formula)
# --------------------------------------------------------------------------- #
def test_Q_matrix_closed_form_vs_series():
    xi = _rand_xi(32, scale=0.4)
    Jcf = N.left_jacobian_se3(xi)
    Jser = N.left_jacobian_se3_series(xi, n_terms=30)
    assert _maxabs(Jcf, Jser) < 1e-10


def test_Q_matrix_closed_form_vs_finite_difference():
    xi0 = _rand_xi(1, scale=0.4)[0]
    eps = 1e-6
    Texp_inv = N.inverse(N.exp_se3(xi0))
    cols = []
    for k in range(6):
        e = np.zeros(6)
        e[k] = eps
        lp = N.log_se3(N.compose(N.exp_se3(xi0 + e), Texp_inv))
        lm = N.log_se3(N.compose(N.exp_se3(xi0 - e), Texp_inv))
        cols.append((lp - lm) / (2 * eps))
    Jfd = np.stack(cols, axis=1)
    assert _maxabs(N.left_jacobian_se3(xi0), Jfd) < 1e-7


def test_left_jacobian_se3_parity_mlx():
    xi = _rand_xi(16, scale=0.4)
    assert _maxabs(np.asarray(S.left_jacobian_se3(_to32(xi))), N.left_jacobian_se3(_np32(xi))) < _TOL_Q


# --------------------------------------------------------------------------- #
# differentiability (the where-NaN-gradient trap)
# --------------------------------------------------------------------------- #
def test_exp_so3_gradient_finite_at_zero():
    def f(w):
        return mx.sum(SO.exp_so3(w))

    g = mx.grad(f)(mx.zeros((3,)))
    assert np.all(np.isfinite(np.asarray(g)))


def test_exp_se3_gradient_finite_small_angle():
    def f(xi):
        return mx.sum(S.exp_se3(xi))

    for th in [0.0, 1e-9, 1e-7]:
        xi = np.concatenate([np.array([0.1, 0.2, 0.3]), _rand_omega(1, theta=th)[0]])
        g = np.asarray(mx.grad(f)(_to32(xi)))
        assert np.all(np.isfinite(g)), (th, g)


def test_log_se3_gradient_finite():
    def f(xi):
        return mx.sum(S.log_se3(S.exp_se3(xi)))

    xi = _to32(_rand_xi(1, scale=0.3)[0])
    g = np.asarray(mx.grad(f)(xi))
    assert np.all(np.isfinite(g))


def test_full_jacobian_gradient_finite_at_zero():
    # The Barfoot Q-matrix has an O(theta^5) term -> needs the double-where trick
    # for a finite gradient at exactly xi=0 (regression guard).
    def f(xi):
        return mx.sum(S.left_jacobian_se3(xi))

    for xi0 in (mx.zeros((6,)), _to32([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])):
        g = np.asarray(mx.grad(f)(xi0))
        assert np.all(np.isfinite(g)), g


def test_log_so3_gradient_finite_at_identity():
    def f(w):
        return mx.sum(SO.log_so3(SO.exp_so3(w)))

    g = np.asarray(mx.grad(f)(mx.zeros((3,))))
    assert np.all(np.isfinite(g))


def test_gradient_value_matches_numpy_finite_difference():
    # A finite gradient can still be WRONG; gradcheck the analytic MLX grad
    # against a numpy-oracle central finite-difference (independent ground truth).
    xi = (_rand_xi(1, scale=0.3)[0]).astype(np.float64)
    h = 1e-6

    def fsum_np(x):
        return float(N.exp_se3(x).sum())

    fd = np.zeros(6)
    for i in range(6):
        e = np.zeros(6)
        e[i] = h
        fd[i] = (fsum_np(xi + e) - fsum_np(xi - e)) / (2 * h)

    g = np.asarray(mx.grad(lambda x: mx.sum(S.exp_se3(x)))(_to32(xi)), dtype=np.float64)
    assert _maxabs(g, fd) < 1e-4, (g, fd)


# --------------------------------------------------------------------------- #
# screw blend: dual-quaternion conversions + DLB + ScLERP
# --------------------------------------------------------------------------- #
def test_se3_dq_roundtrip_and_parity():
    T = _rand_T(32, scale=0.4)
    dq_np = SB.se3_to_dq_numpy(T)
    assert _maxabs(SB.dq_to_se3_numpy(dq_np), T) < 1e-10  # numpy roundtrip
    dq_mlx = np.asarray(SB.se3_to_dq(_to32(T)))
    assert _maxabs(dq_mlx, dq_np) < _TOL  # parity
    back_mlx = np.asarray(SB.dq_to_se3(_to32(dq_np)))
    assert _maxabs(back_mlx, T) < 1e-4


def test_dq_unit_constraints():
    T = _rand_T(32, scale=0.4)
    dq = SB.se3_to_dq_numpy(T)
    qr, qd = dq[..., :4], dq[..., 4:]
    assert np.abs(np.linalg.norm(qr, axis=-1) - 1.0).max() < 1e-10
    assert np.abs(np.sum(qr * qd, -1)).max() < 1e-10  # orthogonality (unit DQ)


def test_dq_scipy_crosscheck():
    from scipy.spatial.transform import RigidTransform

    T = _rand_T(16, scale=0.4)
    ours = SB.se3_to_dq_numpy(T)
    sp = RigidTransform.from_matrix(T).as_dual_quat(scalar_first=True)
    sign = np.sign(np.sum(ours[..., :4] * sp[..., :4], -1))[..., None]
    assert _maxabs(ours * sign, sp) < 1e-7


def test_dlb_single_is_identity():
    T = _rand_T(16, scale=0.4)
    dq1 = SB.se3_to_dq_numpy(T)[..., None, :]
    w = np.ones((16, 1))
    out = SB.dq_to_se3_numpy(SB.dlb_numpy(w, dq1))
    assert _maxabs(out, T) < 1e-10


def test_dlb_endpoints_exact():
    A, B = _rand_T(16, 0.4), _rand_T(16, 0.4)
    dqs = np.stack([SB.se3_to_dq_numpy(A), SB.se3_to_dq_numpy(B)], axis=-2)
    oa = SB.dq_to_se3_numpy(SB.dlb_numpy(np.tile([1.0, 0.0], (16, 1)), dqs))
    ob = SB.dq_to_se3_numpy(SB.dlb_numpy(np.tile([0.0, 1.0], (16, 1)), dqs))
    assert _maxabs(oa, A) < 1e-10 and _maxabs(ob, B) < 1e-10


def test_dlb_midpoint_valid_se3_and_parity():
    A, B = _rand_T(16, 0.4), _rand_T(16, 0.4)
    dqs = np.stack([SB.se3_to_dq_numpy(A), SB.se3_to_dq_numpy(B)], axis=-2)
    w = np.tile([0.5, 0.5], (16, 1))
    mid = SB.dq_to_se3_numpy(SB.dlb_numpy(w, dqs))
    Rm = N.rotation_of(mid)
    assert _maxabs(np.swapaxes(Rm, -1, -2) @ Rm, np.eye(3)) < 1e-9  # orthonormal
    # MLX parity on the blended dual quaternion
    out_mlx = np.asarray(SB.dlb(_to32(w), _to32(dqs)))
    assert _maxabs(out_mlx, SB.dlb_numpy(w, dqs)) < _TOL


def test_sclerp_endpoints():
    A, B = _rand_T(16, 0.4), _rand_T(16, 0.4)
    s0 = SB.dq_to_se3_numpy(SB.sclerp_numpy(A, B, np.zeros(16)))
    s1 = SB.dq_to_se3_numpy(SB.sclerp_numpy(A, B, np.ones(16)))
    assert _maxabs(s0, A) < 1e-10 and _maxabs(s1, B) < 1e-10


def test_sclerp_screw_geodesic_identity():
    A, B = _rand_T(16, 0.4), _rand_T(16, 0.4)
    u = 0.37 * np.ones(16)
    Su = SB.dq_to_se3_numpy(SB.sclerp_numpy(A, B, u))
    lhs = N.log_se3(N.compose(N.inverse(A), Su))
    rhs = u[..., None] * N.log_se3(N.compose(N.inverse(A), B))
    assert _maxabs(lhs, rhs) < 1e-9


def test_sclerp_parity_mlx():
    A, B = _rand_T(8, 0.4), _rand_T(8, 0.4)
    u = np.full(8, 0.42)
    out_mlx = np.asarray(SB.sclerp(_to32(A), _to32(B), _to32(u)))
    out_np = SB.sclerp_numpy(A, B, u)
    sign = np.sign(np.sum(out_mlx[..., :4] * out_np[..., :4], -1))[..., None]
    assert _maxabs(out_mlx * sign, out_np) < _TOL


# --------------------------------------------------------------------------- #
# cumulative SE(3) B-spline
# --------------------------------------------------------------------------- #
def test_bspline_equal_controls_is_constant():
    M = 8
    base = _rand_T(1, 0.3)[0]
    ctrl = np.broadcast_to(base, (M, 4, 4)).copy()
    for t in [0.0, 0.5, 1.7, 3.2, 4.999]:
        assert _maxabs(BS.se3_bspline_eval_numpy(ctrl, t), base) < 1e-12


def test_bspline_on_manifold():
    ctrl = _rand_T(8, 0.25)
    Tq = BS.se3_bspline_eval_numpy(ctrl, 2.3)
    R = N.rotation_of(Tq)
    assert _maxabs(np.swapaxes(R, -1, -2) @ R, np.eye(3)) < 1e-9
    assert _maxabs(Tq[3], np.array([0.0, 0.0, 0.0, 1.0])) < 1e-12


def test_bspline_c0_continuity_at_knots():
    ctrl = _rand_T(8, 0.25)
    for kt in [1, 2, 3, 4]:
        lo = BS.se3_bspline_eval_numpy(ctrl, kt - 1e-7)
        hi = BS.se3_bspline_eval_numpy(ctrl, kt + 1e-7)
        assert _maxabs(lo, hi) < 1e-6, kt


def test_bspline_out_of_domain_clamps_to_boundary():
    # t<0 and t>(M-3) must return the domain-boundary pose, not extrapolate.
    ctrl = _rand_T(8, 0.25)
    n_seg = 8 - 3
    assert _maxabs(BS.se3_bspline_eval_numpy(ctrl, -3.0), BS.se3_bspline_eval_numpy(ctrl, 0.0)) < 1e-12
    hi = BS.se3_bspline_eval_numpy(ctrl, 999.0)
    assert _maxabs(hi, BS.se3_bspline_eval_numpy(ctrl, float(n_seg))) < 1e-12
    assert np.all(np.isfinite(hi))
    # MLX path clamps identically
    assert _maxabs(np.asarray(BS.se3_bspline_eval(_to32(ctrl), -3.0)), BS.se3_bspline_eval_numpy(ctrl, 0.0)) < _TOL


def test_bspline_parity_mlx():
    ctrl = _rand_T(8, 0.25)
    ctrl32 = _to32(ctrl)
    worst = 0.0
    for t in [0.3, 1.5, 2.7, 4.1]:
        worst = max(worst, _maxabs(np.asarray(BS.se3_bspline_eval(ctrl32, t)), BS.se3_bspline_eval_numpy(ctrl, t)))
    assert worst < _TOL, worst


def test_bspline_velocity_autodiff_matches_finite_difference():
    ctrl = _to32(_rand_T(8, 0.25))

    def pose_sum(t_scalar):
        # differentiate the spline pose wrt time via a constant-twist segment.
        M = ctrl.shape[-3]
        s = int(min(max(int(np.floor(float(t_scalar))), 0), M - 4))
        u = t_scalar - s
        c4 = ctrl[..., s : s + 4, :, :]
        return mx.sum(BS.se3_bspline_segment(c4, u))

    t0 = mx.array(2.3, dtype=mx.float32)
    g = float(np.asarray(mx.grad(pose_sum)(t0)))
    h = 1e-3
    fd = (float(np.asarray(pose_sum(mx.array(2.3 + h)))) - float(np.asarray(pose_sum(mx.array(2.3 - h))))) / (2 * h)
    assert abs(g - fd) < 1e-2, (g, fd)


# --------------------------------------------------------------------------- #
# batched shapes + standalone (no residual-pipeline imports)
# --------------------------------------------------------------------------- #
def test_batched_shapes():
    xi = _to32(_rand_xi(5, 0.4).reshape(5, 1, 6))
    T = S.exp_se3(xi)
    assert T.shape == (5, 1, 4, 4)
    assert S.log_se3(T).shape == (5, 1, 6)
    assert S.adjoint_T(T).shape == (5, 1, 6, 6)


def test_lie_package_is_standalone():
    # tac.lie must NOT import the witness residual pipeline / scorer / trainer.
    # Re-import in a clean subprocess so the check is not masked by modules a
    # sibling test already imported into this interpreter.
    import subprocess
    import sys

    code = (
        "import sys; import tac.lie\n"
        "forbidden=('scorer','witness','residual','compose_witness','v2_compose','renderer','trainer','train_')\n"
        "bad=[m for m in sys.modules if m.startswith('tac.') and any(t in m for t in forbidden)]\n"
        "print('BAD:'+ ','.join(bad)); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"tac.lie pulled pipeline modules: {r.stdout}{r.stderr}"
    # all tac.lie submodules really live under the lie package directory
    importlib_path = sys.modules["tac.lie"].__file__ or ""
    assert importlib_path.replace("\\", "/").endswith("tac/lie/__init__.py")
