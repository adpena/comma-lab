# SPDX-License-Identifier: MIT
"""MLX-native, autodiff-clean SO(3) primitives (the fast path).

Mirrors :mod:`tac.lie._se3_numpy` formula-for-formula; the NumPy module is the
bit-identical verdict authority and this MLX path must match it within tol
(parity tests in ``tac.lie.tests``). All small-angle branches use the
``where``-with-clamped-denominator idiom so gradients stay finite at ``theta=0``
(see :mod:`tac.lie` docstring). Convention: see :data:`CONVENTION`.
"""

from __future__ import annotations

import mlx.core as mx

CONVENTION = "translation_first_(rho,omega)"

_EPS = 1e-6
_EPS2 = _EPS * _EPS


def skew(v: mx.array) -> mx.array:
    """``(..., 3) -> (..., 3, 3)`` so(3) hat operator."""
    a, b, c = v[..., 0], v[..., 1], v[..., 2]
    z = mx.zeros_like(a)
    row0 = mx.stack([z, -c, b], axis=-1)
    row1 = mx.stack([c, z, -a], axis=-1)
    row2 = mx.stack([-b, a, z], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def unskew(m: mx.array) -> mx.array:
    """Inverse of :func:`skew`."""
    return mx.stack(
        [
            0.5 * (m[..., 2, 1] - m[..., 1, 2]),
            0.5 * (m[..., 0, 2] - m[..., 2, 0]),
            0.5 * (m[..., 1, 0] - m[..., 0, 1]),
        ],
        axis=-1,
    )


def _theta(omega: mx.array) -> tuple[mx.array, mx.array]:
    """Return ``(theta_safe, theta2)``.

    ``theta_safe = sqrt(max(theta2, eps^2))`` clamps the sqrt ARGUMENT (not its
    output) so the gradient stays finite at ``omega=0`` -- otherwise
    ``d sqrt/d arg = inf`` at 0 produces a ``0*inf = NaN`` that poisons even the
    masked Taylor branch. Small-angle branches must select on ``theta2 < eps^2``
    (equivalently ``theta < eps``); the clamped ``theta_safe`` is only ever used
    inside the (unselected) exact branch at small angle, so values are correct.
    """
    theta2 = mx.sum(omega * omega, axis=-1)
    theta = mx.sqrt(mx.maximum(theta2, _EPS2))
    return theta, theta2


def _bc(a: mx.array) -> mx.array:
    return a[..., None, None]


def _eye(shape_mat: mx.array) -> mx.array:
    return mx.broadcast_to(mx.eye(3, dtype=shape_mat.dtype), shape_mat.shape)


def exp_so3(omega: mx.array) -> mx.array:
    """Rodrigues exponential ``so(3) -> SO(3)``."""
    theta, theta2 = _theta(omega)
    K = skew(omega)
    K2 = K @ K
    small = theta2 < _EPS2
    th_safe = mx.maximum(theta, _EPS)
    th2_safe = mx.maximum(theta2, _EPS2)
    A = mx.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, mx.sin(theta) / th_safe)
    B = mx.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - mx.cos(theta)) / th2_safe)
    return _eye(K) + _bc(A) * K + _bc(B) * K2


def log_so3(R: mx.array) -> mx.array:
    """Logarithm ``SO(3) -> so(3)`` (valid for ``theta in [0, pi)``)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = mx.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )  # = 2 sin(theta) * axis
    wn2 = mx.sum(w * w, axis=-1)
    wnorm = mx.sqrt(mx.maximum(wn2, _EPS2))  # = 2 sin(theta); clamp sqrt ARG
    s = 0.5 * wnorm  # sin(theta)
    c = mx.clip((tr - 1.0) * 0.5, -1.0, 1.0)  # cos(theta)
    theta = mx.arctan2(s, c)  # well-conditioned at small theta (unlike arccos)
    small = wn2 < _EPS2
    factor = mx.where(
        small,
        0.5 + theta**2 / 12.0 + 7.0 * theta**4 / 720.0,
        theta / mx.maximum(wnorm, _EPS),
    )  # = theta / (2 sin theta) = theta / ||w||
    return factor[..., None] * w


def left_jacobian_so3(omega: mx.array) -> mx.array:
    """Left Jacobian ``J_l(omega)`` (= the SE(3) translation V-matrix)."""
    theta, theta2 = _theta(omega)
    K = skew(omega)
    K2 = K @ K
    small = theta2 < _EPS2
    th2_safe = mx.maximum(theta2, _EPS2)
    th3_safe = mx.maximum(theta**3, _EPS**3)
    B = mx.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - mx.cos(theta)) / th2_safe)
    C = mx.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        (theta - mx.sin(theta)) / th3_safe,
    )
    return _eye(K) + _bc(B) * K + _bc(C) * K2


def right_jacobian_so3(omega: mx.array) -> mx.array:
    """Right Jacobian ``J_r(omega) = J_l(-omega)``."""
    return left_jacobian_so3(-omega)


def _D_coeff(theta: mx.array, theta2: mx.array) -> mx.array:
    small = theta2 < _EPS2
    th_safe = mx.maximum(theta, _EPS)
    th2_safe = mx.maximum(theta2, _EPS2)
    sin_safe = mx.maximum(mx.sin(theta), _EPS)
    exact = 1.0 / th2_safe - (1.0 + mx.cos(theta)) / (2.0 * th_safe * sin_safe)
    taylor = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    return mx.where(small, taylor, exact)


def left_jacobian_inv_so3(omega: mx.array) -> mx.array:
    """``J_l^{-1}(omega) = I - 1/2 [omega]_x + D [omega]_x^2``."""
    theta, theta2 = _theta(omega)
    K = skew(omega)
    K2 = K @ K
    D = _D_coeff(theta, theta2)
    return _eye(K) - 0.5 * K + _bc(D) * K2


def right_jacobian_inv_so3(omega: mx.array) -> mx.array:
    """``J_r^{-1}(omega) = J_l^{-1}(-omega) = I + 1/2 [omega]_x + D [omega]_x^2``."""
    return left_jacobian_inv_so3(-omega)


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
]
