# SPDX-License-Identifier: MIT
"""NumPy-fp32 reference oracle for the ``tac.lie`` se(3)/SE(3) library.

This module is the **bit-identical verdict authority** per the pact
deterministic-reproducibility non-negotiable: the MLX fast path
(``tac.lie.so3`` / ``tac.lie.se3``) must match these formulas within a tight
tolerance, and these formulas in turn are cross-checked against algebraic
identities, a closed-form-vs-series check (the Barfoot Q-matrix), and the
external ``scipy.spatial.transform.RigidTransform`` SE(3) oracle.

Math is standard textbook (Solà micro-Lie arXiv:1812.01537; Barfoot, *State
Estimation for Robotics*, 2017). Nothing here is video-derived or contest IP;
this is a pure-geometry give-back component (MIT).

Convention (fixed once, asserted everywhere)
--------------------------------------------
A twist is ``xi = (rho, omega) in R^6`` with **translation-first** block order:
``rho = xi[..., :3]`` (the se(3) translation generator coefficients) and
``omega = xi[..., 3:]`` (the so(3) rotation generator coefficients). This
matches Solà micro-Lie, manif, and liegroups. Lynch--Park / Murray--Li--Sastry
use the OPPOSITE ``(omega, v)`` ordering -- swap the blocks when porting those.

An SE(3) element is a ``(..., 4, 4)`` homogeneous matrix ``T = [[R, t],[0, 1]]``.
All functions support a leading batch dimension via NumPy broadcasting.
"""

from __future__ import annotations

import numpy as np

CONVENTION = "translation_first_(rho,omega)"

# Small-angle thresholds. fp32 reference work uses ~1e-6; the float64 "golden"
# layer in the tests uses these same branches (the Taylor series are accurate
# to many digits well past the threshold, so this is safe at both precisions).
_EPS = 1e-6
_EPS2 = _EPS * _EPS


def _asf(x: np.ndarray) -> np.ndarray:
    """As a floating array, preserving fp32/fp64 (ints -> fp64). Keeps this
    oracle usable as both an fp64 ``golden`` and an fp32 ``authority`` layer."""
    x = np.asarray(x)
    return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float64)


def _eye_like(M: np.ndarray) -> np.ndarray:
    n = M.shape[-1]
    return np.broadcast_to(np.eye(n, dtype=M.dtype), M.shape).copy()


# --------------------------------------------------------------------------- #
# so(3) hat / vee
# --------------------------------------------------------------------------- #
def skew(v: np.ndarray) -> np.ndarray:
    """Map ``(..., 3) -> (..., 3, 3)`` skew-symmetric (so(3) hat operator).

    ``skew([a, b, c]) = [[0, -c, b], [c, 0, -a], [-b, a, 0]]``.
    """
    v = _asf(v)
    a, b, c = v[..., 0], v[..., 1], v[..., 2]
    z = np.zeros_like(a)
    row0 = np.stack([z, -c, b], axis=-1)
    row1 = np.stack([c, z, -a], axis=-1)
    row2 = np.stack([-b, a, z], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def unskew(m: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew` for a skew-symmetric ``(..., 3, 3)`` matrix."""
    m = np.asarray(m)
    return np.stack(
        [
            0.5 * (m[..., 2, 1] - m[..., 1, 2]),
            0.5 * (m[..., 0, 2] - m[..., 2, 0]),
            0.5 * (m[..., 1, 0] - m[..., 0, 1]),
        ],
        axis=-1,
    )


def _theta(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta2 = np.sum(omega * omega, axis=-1)
    theta = np.sqrt(np.maximum(theta2, 0.0))
    return theta, theta2


def _bc(a: np.ndarray) -> np.ndarray:
    """Broadcast a ``(...,)`` scalar field to ``(..., 1, 1)`` for matrix scaling."""
    return a[..., None, None]


# --------------------------------------------------------------------------- #
# SO(3) exp / log
# --------------------------------------------------------------------------- #
def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Rodrigues exponential ``so(3) -> SO(3)``: ``(..., 3) -> (..., 3, 3)``."""
    omega = _asf(omega)
    theta, theta2 = _theta(omega)
    K = skew(omega)
    K2 = K @ K
    small = theta < _EPS
    th_safe = np.maximum(theta, _EPS)
    th2_safe = np.maximum(theta2, _EPS2)
    A = np.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, np.sin(theta) / th_safe)
    B = np.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - np.cos(theta)) / th2_safe)
    eye = _eye_like(K)
    return eye + _bc(A) * K + _bc(B) * K2


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithm ``SO(3) -> so(3)``: ``(..., 3, 3) -> (..., 3)``.

    Valid for ``theta in [0, pi)``. The ``theta -> pi`` branch cut (where
    ``2 sin theta -> 0`` while ``R - R^T -> 0``) is out of the ego-motion
    regime; it is guarded against a hard NaN but not given a dedicated eigen
    branch (so parity with the MLX path holds across the tested range).
    """
    R = _asf(R)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )  # = 2 sin(theta) * axis
    wn2 = np.sum(w * w, axis=-1)
    wnorm = np.sqrt(np.maximum(wn2, _EPS2))  # = 2 sin(theta); clamp sqrt ARG
    s = 0.5 * wnorm  # sin(theta) (theta in [0, pi))
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)  # cos(theta)
    theta = np.arctan2(s, c)  # well-conditioned at small theta (unlike arccos)
    small = wn2 < _EPS2
    factor = np.where(
        small,
        0.5 + theta**2 / 12.0 + 7.0 * theta**4 / 720.0,
        theta / np.maximum(wnorm, _EPS),
    )  # = theta / (2 sin theta) = theta / ||w||
    return factor[..., None] * w


# --------------------------------------------------------------------------- #
# SO(3) Jacobians (left/right) and inverses
# --------------------------------------------------------------------------- #
def left_jacobian_so3(omega: np.ndarray) -> np.ndarray:
    """Left Jacobian ``J_l(omega)`` -- also the SE(3) translation V-matrix."""
    omega = _asf(omega)
    theta, theta2 = _theta(omega)
    K = skew(omega)
    K2 = K @ K
    small = theta < _EPS
    th2_safe = np.maximum(theta2, _EPS2)
    th3_safe = np.maximum(theta**3, _EPS**3)
    B = np.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - np.cos(theta)) / th2_safe)
    C = np.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        (theta - np.sin(theta)) / th3_safe,
    )
    eye = _eye_like(K)
    return eye + _bc(B) * K + _bc(C) * K2


def right_jacobian_so3(omega: np.ndarray) -> np.ndarray:
    """Right Jacobian ``J_r(omega) = J_l(-omega)``."""
    return left_jacobian_so3(-_asf(omega))


def _D_coeff(theta: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    small = theta < _EPS
    th_safe = np.maximum(theta, _EPS)
    th2_safe = np.maximum(theta2, _EPS2)
    sin_safe = np.maximum(np.sin(theta), _EPS)
    exact = 1.0 / th2_safe - (1.0 + np.cos(theta)) / (2.0 * th_safe * sin_safe)
    taylor = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    return np.where(small, taylor, exact)


def left_jacobian_inv_so3(omega: np.ndarray) -> np.ndarray:
    """``J_l^{-1}(omega) = I - 1/2 [omega]_x + D [omega]_x^2``."""
    omega = _asf(omega)
    theta, theta2 = _theta(omega)
    K = skew(omega)
    K2 = K @ K
    D = _D_coeff(theta, theta2)
    eye = _eye_like(K)
    return eye - 0.5 * K + _bc(D) * K2


def right_jacobian_inv_so3(omega: np.ndarray) -> np.ndarray:
    """``J_r^{-1}(omega) = J_l^{-1}(-omega) = I + 1/2 [omega]_x + D [omega]_x^2``."""
    return left_jacobian_inv_so3(-_asf(omega))


# --------------------------------------------------------------------------- #
# SE(3) homogeneous-matrix helpers
# --------------------------------------------------------------------------- #
def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble ``(..., 4, 4)`` from rotation ``(..., 3, 3)`` and ``(..., 3)``."""
    R = _asf(R)
    t = _asf(t)
    batch = R.shape[:-2]
    T = np.zeros(batch + (4, 4), dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def rotation_of(T: np.ndarray) -> np.ndarray:
    return np.asarray(T)[..., :3, :3]


def translation_of(T: np.ndarray) -> np.ndarray:
    return np.asarray(T)[..., :3, 3]


# --------------------------------------------------------------------------- #
# SE(3) exp / log
# --------------------------------------------------------------------------- #
def exp_se3(xi: np.ndarray) -> np.ndarray:
    """Exponential ``se(3) -> SE(3)``: ``(..., 6) -> (..., 4, 4)``.

    ``xi = (rho, omega)`` translation-first; ``T = [[exp(omega), V(omega) rho], [0, 1]]``
    with ``V = J_l(omega)``.
    """
    xi = _asf(xi)
    rho = xi[..., :3]
    omega = xi[..., 3:]
    R = exp_so3(omega)
    V = left_jacobian_so3(omega)
    t = (V @ rho[..., None])[..., 0]
    return make_T(R, t)


def log_se3(T: np.ndarray) -> np.ndarray:
    """Logarithm ``SE(3) -> se(3)``: ``(..., 4, 4) -> (..., 6)``."""
    T = _asf(T)
    R = rotation_of(T)
    t = translation_of(T)
    omega = log_so3(R)
    rho = (left_jacobian_inv_so3(omega) @ t[..., None])[..., 0]
    return np.concatenate([rho, omega], axis=-1)


# --------------------------------------------------------------------------- #
# SE(3) group ops (analytic; no general matrix inverse needed)
# --------------------------------------------------------------------------- #
def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    return _asf(T1) @ _asf(T2)


def inverse(T: np.ndarray) -> np.ndarray:
    """Analytic SE(3) inverse: ``[[R^T, -R^T t], [0, 1]]``."""
    T = _asf(T)
    R = rotation_of(T)
    t = translation_of(T)
    Rt = np.swapaxes(R, -1, -2)
    tinv = -(Rt @ t[..., None])[..., 0]
    return make_T(Rt, tinv)


# --------------------------------------------------------------------------- #
# Adjoints
# --------------------------------------------------------------------------- #
def adjoint_T(T: np.ndarray) -> np.ndarray:
    """Group adjoint ``Ad_T`` (6x6), translation-first ordering.

    ``Ad_T = [[R, [t]_x R], [0, R]]`` acting on ``xi = (rho, omega)``.
    """
    T = _asf(T)
    R = rotation_of(T)
    t = translation_of(T)
    tx = skew(t)
    batch = R.shape[:-2]
    Ad = np.zeros(batch + (6, 6), dtype=R.dtype)
    Ad[..., :3, :3] = R
    Ad[..., :3, 3:] = tx @ R
    Ad[..., 3:, 3:] = R
    return Ad


def adjoint_se3(xi: np.ndarray) -> np.ndarray:
    """Algebra adjoint ``ad_xi`` (6x6), translation-first.

    ``ad_xi = [[[omega]_x, [rho]_x], [0, [omega]_x]]``.
    """
    xi = _asf(xi)
    rho = xi[..., :3]
    omega = xi[..., 3:]
    wx = skew(omega)
    px = skew(rho)
    batch = xi.shape[:-1]
    ad = np.zeros(batch + (6, 6), dtype=xi.dtype)
    ad[..., :3, :3] = wx
    ad[..., :3, 3:] = px
    ad[..., 3:, 3:] = wx
    return ad


# --------------------------------------------------------------------------- #
# Full SE(3) left Jacobian + Barfoot Q-matrix (FLAGGED: verified by tests)
# --------------------------------------------------------------------------- #
def left_jacobian_Q(xi: np.ndarray) -> np.ndarray:
    """Barfoot Q-matrix (3x3) -- top-right block of the full SE(3) left Jacobian.

    Transcribed from Barfoot Eq. (7.86b) in the liegroups (utiasSTARS, MIT)
    coefficient form. **Verified by tests** two independent ways: closed-form
    vs the series ``sum_n ad^n/(n+1)!`` and central finite-difference of
    ``log_se3``. Do not trust this formula without those checks passing.
    """
    xi = _asf(xi)
    rho = xi[..., :3]
    omega = xi[..., 3:]
    rx = skew(rho)
    px = skew(omega)
    theta, theta2 = _theta(omega)
    small = theta < _EPS
    # Double-where for parity with the MLX path (value-identical: the Taylor
    # branch is selected wherever small; the benign theta_e only affects the
    # unused exact branch, which matters for autodiff on the MLX side).
    theta_e = np.where(small, np.ones_like(theta), theta)
    sin_e = np.sin(theta_e)
    cos_e = np.cos(theta_e)
    th3 = theta_e**3
    th4 = theta_e**4
    th5 = theta_e**5

    m1 = 0.5
    m2 = np.where(small, 1.0 / 6.0 - theta2 / 120.0, (theta_e - sin_e) / th3)
    m3 = np.where(small, 1.0 / 24.0 - theta2 / 720.0, (0.5 * theta_e * theta_e + cos_e - 1.0) / th4)
    m4 = np.where(
        small,
        1.0 / 120.0 - theta2 / 2520.0,
        (theta_e - 1.5 * sin_e + 0.5 * theta_e * cos_e) / th5,
    )

    pr = px @ rx
    rp = rx @ px
    prp = px @ rx @ px
    ppr = px @ px @ rx
    rpp = rx @ px @ px
    prpp = px @ rx @ px @ px
    pprp = px @ px @ rx @ px

    t1 = rx
    t2 = pr + rp + prp
    t3 = ppr + rpp - 3.0 * prp
    t4 = prpp + pprp
    return m1 * t1 + _bc(m2) * t2 + _bc(m3) * t3 + _bc(m4) * t4


def left_jacobian_se3(xi: np.ndarray) -> np.ndarray:
    """Full SE(3) left Jacobian (6x6): ``[[J_l(omega), Q(xi)], [0, J_l(omega)]]``."""
    xi = _asf(xi)
    omega = xi[..., 3:]
    Jl = left_jacobian_so3(omega)
    Q = left_jacobian_Q(xi)
    batch = xi.shape[:-1]
    J = np.zeros(batch + (6, 6), dtype=Jl.dtype)
    J[..., :3, :3] = Jl
    J[..., :3, 3:] = Q
    J[..., 3:, 3:] = Jl
    return J


def left_jacobian_se3_series(xi: np.ndarray, n_terms: int = 24) -> np.ndarray:
    """Independent oracle for the full left Jacobian: ``sum_{n>=0} ad_xi^n/(n+1)!``.

    Used only to verify :func:`left_jacobian_se3` (and hence the Q-matrix). Not
    a fast path -- it materializes the matrix power series.
    """
    xi = _asf(xi)
    ad = adjoint_se3(xi)
    batch = ad.shape[:-2]
    eye6 = np.broadcast_to(np.eye(6, dtype=ad.dtype), batch + (6, 6)).copy()
    term = eye6.copy()  # ad^0
    J = eye6.copy()  # n=0 term: 1/1! * I
    fact = 1.0
    for n in range(1, n_terms):
        term = term @ ad  # ad^n
        fact *= n + 1  # (n+1)!
        J = J + term / fact
    return J


__all__ = [
    "CONVENTION",
    "skew",
    "unskew",
    "exp_so3",
    "log_so3",
    "left_jacobian_so3",
    "right_jacobian_so3",
    "left_jacobian_inv_so3",
    "right_jacobian_inv_so3",
    "make_T",
    "rotation_of",
    "translation_of",
    "exp_se3",
    "log_se3",
    "compose",
    "inverse",
    "adjoint_T",
    "adjoint_se3",
    "left_jacobian_Q",
    "left_jacobian_se3",
    "left_jacobian_se3_series",
]
