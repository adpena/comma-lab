# SPDX-License-Identifier: MIT
"""MLX-native, autodiff-clean SE(3) primitives (the fast path).

Mirrors :mod:`tac.lie._se3_numpy` formula-for-formula. Homogeneous matrices and
the 6x6 adjoint/Jacobian blocks are assembled by concatenation (not in-place
assignment) so gradients flow cleanly. Convention: twist ``xi = (rho, omega)``
translation-first; ``T`` is ``(..., 4, 4)``. See :data:`CONVENTION`.
"""

from __future__ import annotations

import mlx.core as mx

from .so3 import (
    CONVENTION,
    _bc,
    _theta,
    exp_so3,
    left_jacobian_inv_so3,
    left_jacobian_so3,
    log_so3,
    skew,
)

__all__ = [
    "CONVENTION",
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
]

_EPS = 1e-6


def _bottom_row(R: mx.array) -> mx.array:
    """``(..., 1, 4)`` constant ``[0, 0, 0, 1]`` broadcast to R's batch shape."""
    batch = R.shape[:-2]
    row = mx.array([0.0, 0.0, 0.0, 1.0], dtype=R.dtype).reshape((1,) * len(batch) + (1, 4))
    return mx.broadcast_to(row, batch + (1, 4))


def make_T(R: mx.array, t: mx.array) -> mx.array:
    """Assemble ``(..., 4, 4)`` from ``R (..., 3, 3)`` and ``t (..., 3)``."""
    top = mx.concatenate([R, t[..., None]], axis=-1)  # (..., 3, 4)
    return mx.concatenate([top, _bottom_row(R)], axis=-2)


def rotation_of(T: mx.array) -> mx.array:
    return T[..., :3, :3]


def translation_of(T: mx.array) -> mx.array:
    return T[..., :3, 3]


def exp_se3(xi: mx.array) -> mx.array:
    """Exponential ``se(3) -> SE(3)``: ``(..., 6) -> (..., 4, 4)``."""
    rho = xi[..., :3]
    omega = xi[..., 3:]
    R = exp_so3(omega)
    V = left_jacobian_so3(omega)
    t = (V @ rho[..., None])[..., 0]
    return make_T(R, t)


def log_se3(T: mx.array) -> mx.array:
    """Logarithm ``SE(3) -> se(3)``: ``(..., 4, 4) -> (..., 6)``."""
    R = rotation_of(T)
    t = translation_of(T)
    omega = log_so3(R)
    rho = (left_jacobian_inv_so3(omega) @ t[..., None])[..., 0]
    return mx.concatenate([rho, omega], axis=-1)


def compose(T1: mx.array, T2: mx.array) -> mx.array:
    return T1 @ T2


def inverse(T: mx.array) -> mx.array:
    """Analytic SE(3) inverse ``[[R^T, -R^T t], [0, 1]]``."""
    R = rotation_of(T)
    t = translation_of(T)
    Rt = mx.swapaxes(R, -1, -2)
    tinv = -(Rt @ t[..., None])[..., 0]
    return make_T(Rt, tinv)


def _block6(tl: mx.array, tr: mx.array, bl: mx.array, br: mx.array) -> mx.array:
    """Assemble a ``(..., 6, 6)`` from four ``(..., 3, 3)`` blocks."""
    top = mx.concatenate([tl, tr], axis=-1)
    bot = mx.concatenate([bl, br], axis=-1)
    return mx.concatenate([top, bot], axis=-2)


def adjoint_T(T: mx.array) -> mx.array:
    """Group adjoint ``Ad_T = [[R, [t]_x R], [0, R]]`` (translation-first)."""
    R = rotation_of(T)
    t = translation_of(T)
    z = mx.zeros_like(R)
    return _block6(R, skew(t) @ R, z, R)


def adjoint_se3(xi: mx.array) -> mx.array:
    """Algebra adjoint ``ad_xi = [[[omega]_x, [rho]_x], [0, [omega]_x]]``."""
    rho = xi[..., :3]
    omega = xi[..., 3:]
    wx = skew(omega)
    px = skew(rho)
    z = mx.zeros_like(wx)
    return _block6(wx, px, z, wx)


def left_jacobian_Q(xi: mx.array) -> mx.array:
    """Barfoot Q-matrix (3x3); see the NumPy oracle for provenance + verification."""
    rho = xi[..., :3]
    omega = xi[..., 3:]
    rx = skew(rho)
    px = skew(omega)
    theta, theta2 = _theta(omega)
    small = theta2 < _EPS * _EPS
    # Double-where: evaluate the exact branch at a benign theta (1.0) wherever
    # ``small``, so its (unused) value AND gradient are finite; the outer where
    # then selects the Taylor value. Pure denominator-clamping is insufficient
    # for the O(theta^5) m4 term -- the division's gradient still goes NaN at the
    # clamp boundary. (Canonical JAX/MLX double-where idiom.)
    theta_e = mx.where(small, mx.ones_like(theta), theta)
    sin_e = mx.sin(theta_e)
    cos_e = mx.cos(theta_e)
    th3 = theta_e**3
    th4 = theta_e**4
    th5 = theta_e**5

    m1 = 0.5
    m2 = mx.where(small, 1.0 / 6.0 - theta2 / 120.0, (theta_e - sin_e) / th3)
    m3 = mx.where(small, 1.0 / 24.0 - theta2 / 720.0, (0.5 * theta_e * theta_e + cos_e - 1.0) / th4)
    m4 = mx.where(
        small,
        1.0 / 120.0 - theta2 / 2520.0,
        (theta_e - 1.5 * sin_e + 0.5 * theta_e * cos_e) / th5,
    )

    prp = px @ rx @ px
    t1 = rx
    t2 = px @ rx + rx @ px + prp
    t3 = px @ px @ rx + rx @ px @ px - 3.0 * prp
    t4 = px @ rx @ px @ px + px @ px @ rx @ px
    return m1 * t1 + _bc(m2) * t2 + _bc(m3) * t3 + _bc(m4) * t4


def left_jacobian_se3(xi: mx.array) -> mx.array:
    """Full SE(3) left Jacobian ``[[J_l(omega), Q(xi)], [0, J_l(omega)]]`` (6x6)."""
    omega = xi[..., 3:]
    Jl = left_jacobian_so3(omega)
    Q = left_jacobian_Q(xi)
    z = mx.zeros_like(Jl)
    return _block6(Jl, Q, z, Jl)
