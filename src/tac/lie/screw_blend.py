# SPDX-License-Identifier: MIT
"""Dual-quaternion screw blending (DLB) and screw interpolation (ScLERP).

Two backends: a NumPy reference oracle (``*_numpy``) and the MLX fast path
(unsuffixed). Both honor the translation-first ``(rho, omega)`` SE(3)
convention and the scalar-first quaternion layout ``[w, x, y, z]``.

* **DLB** (Kavan--Collins--Zara--O'Sullivan, ACM TOG 2008): the approximate
  dual-quaternion linear blend. A naive *linear* blend of rigid transforms
  collapses/tears; the dual-quaternion blend is smooth and seam-free. This is
  the per-class-warp-seam fix for the scored annulus (witness §1.3).
* **ScLERP**: the exact 2-transform screw interpolation, i.e. the constant-screw
  geodesic ``A . exp(u . log(A^-1 B))`` rendered as a dual quaternion. Routed
  through the verified Lie core (one source of truth) rather than a separately
  transcribed dual-quaternion power.

Implemented fresh from the textbook math (dqrobotics is LGPL, basalt is MPL --
neither copied). A dual quaternion is stored as ``(..., 8)`` = real quat
``[:4]`` then dual quat ``[4:]``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import _se3_numpy as _np
from . import se3 as _mx

_EPS = 1e-6
_EPS2 = _EPS * _EPS

__all__ = [
    "quat_mul_numpy",
    "se3_to_dq_numpy",
    "dq_to_se3_numpy",
    "dlb_numpy",
    "sclerp_numpy",
    "quat_mul",
    "se3_to_dq",
    "dq_to_se3",
    "dlb",
    "sclerp",
]


# --------------------------------------------------------------------------- #
# NumPy reference oracle
# --------------------------------------------------------------------------- #
def quat_mul_numpy(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hamilton product of scalar-first quaternions ``(..., 4)``."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    return np.stack(
        [
            qw * pw - qx * px - qy * py - qz * pz,
            qw * px + qx * pw + qy * pz - qz * py,
            qw * py - qx * pz + qy * pw + qz * px,
            qw * pz + qx * py - qy * px + qz * pw,
        ],
        axis=-1,
    )


def se3_to_dq_numpy(T: np.ndarray) -> np.ndarray:
    """SE(3) ``(..., 4, 4)`` -> unit dual quaternion ``(..., 8)``."""
    R = _np.rotation_of(T)
    t = _np.translation_of(T)
    omega = _np.log_so3(R)
    theta, theta2 = _np._theta(omega)
    half = 0.5 * theta
    small = theta < _EPS
    s = np.where(small, 0.5 - theta2 / 48.0 + theta2 * theta2 / 3840.0, np.sin(half) / np.maximum(theta, _EPS))
    qr = np.concatenate([np.cos(half)[..., None], s[..., None] * omega], axis=-1)
    tq = np.concatenate([np.zeros_like(t[..., :1]), t], axis=-1)
    qd = 0.5 * quat_mul_numpy(tq, qr)
    return np.concatenate([qr, qd], axis=-1)


def dq_to_se3_numpy(dq: np.ndarray) -> np.ndarray:
    """Unit dual quaternion ``(..., 8)`` -> SE(3) ``(..., 4, 4)``."""
    qr = dq[..., :4]
    qd = dq[..., 4:]
    w = qr[..., 0]
    v = qr[..., 1:]
    vnorm2 = np.sum(v * v, axis=-1)
    vnorm = np.sqrt(np.maximum(vnorm2, 0.0))
    theta = 2.0 * np.arctan2(vnorm, w)
    small = theta < _EPS
    coef = np.where(small, 2.0 + theta**2 / 12.0, theta / np.maximum(vnorm, _EPS))
    omega = coef[..., None] * v
    R = _np.exp_so3(omega)
    qr_conj = np.concatenate([w[..., None], -v], axis=-1)
    tq = 2.0 * quat_mul_numpy(qd, qr_conj)
    t = tq[..., 1:]
    return _np.make_T(R, t)


def dlb_numpy(weights: np.ndarray, dqs: np.ndarray) -> np.ndarray:
    """Dual-quaternion Linear Blend. ``weights (..., N)``, ``dqs (..., N, 8)``."""
    weights = np.asarray(weights, dtype=float)
    dqs = np.asarray(dqs, dtype=float)
    qr = dqs[..., :4]
    pivot = qr[..., 0:1, :]  # (..., 1, 4)
    dot = np.sum(qr * pivot, axis=-1)  # (..., N)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    dqs_fixed = dqs * sign[..., None]
    b = np.sum(weights[..., None] * dqs_fixed, axis=-2)  # (..., 8)
    real_norm = np.sqrt(np.maximum(np.sum(b[..., :4] ** 2, axis=-1), _EPS2))
    return b / real_norm[..., None]


def sclerp_numpy(A: np.ndarray, B: np.ndarray, u) -> np.ndarray:
    """Screw geodesic ``A . exp(u . log(A^-1 B))`` as a dual quaternion."""
    u = np.asarray(u, dtype=float)
    rel = _np.compose(_np.inverse(A), B)
    xi = _np.log_se3(rel)
    Tu = _np.compose(A, _np.exp_se3(u[..., None] * xi))
    return se3_to_dq_numpy(Tu)


# --------------------------------------------------------------------------- #
# MLX fast path
# --------------------------------------------------------------------------- #
def quat_mul(q: mx.array, p: mx.array) -> mx.array:
    """Hamilton product of scalar-first quaternions ``(..., 4)`` (MLX)."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    return mx.stack(
        [
            qw * pw - qx * px - qy * py - qz * pz,
            qw * px + qx * pw + qy * pz - qz * py,
            qw * py - qx * pz + qy * pw + qz * px,
            qw * pz + qx * py - qy * px + qz * pw,
        ],
        axis=-1,
    )


def se3_to_dq(T: mx.array) -> mx.array:
    """SE(3) ``(..., 4, 4)`` -> unit dual quaternion ``(..., 8)`` (MLX)."""
    R = _mx.rotation_of(T)
    t = _mx.translation_of(T)
    omega = _mx.log_so3(R)
    theta, theta2 = _mx._theta(omega)
    half = 0.5 * theta
    small = theta2 < _EPS2
    s = mx.where(small, 0.5 - theta2 / 48.0 + theta2 * theta2 / 3840.0, mx.sin(half) / mx.maximum(theta, _EPS))
    qr = mx.concatenate([mx.cos(half)[..., None], s[..., None] * omega], axis=-1)
    tq = mx.concatenate([mx.zeros_like(t[..., :1]), t], axis=-1)
    qd = 0.5 * quat_mul(tq, qr)
    return mx.concatenate([qr, qd], axis=-1)


def dq_to_se3(dq: mx.array) -> mx.array:
    """Unit dual quaternion ``(..., 8)`` -> SE(3) ``(..., 4, 4)`` (MLX)."""
    qr = dq[..., :4]
    qd = dq[..., 4:]
    w = qr[..., 0]
    v = qr[..., 1:]
    vnorm2 = mx.sum(v * v, axis=-1)
    vnorm = mx.sqrt(mx.maximum(vnorm2, _EPS2))  # clamp sqrt ARG for finite grad
    theta = 2.0 * mx.arctan2(vnorm, w)
    small = theta < _EPS
    coef = mx.where(small, 2.0 + theta**2 / 12.0, theta / mx.maximum(vnorm, _EPS))
    omega = coef[..., None] * v
    R = _mx.exp_so3(omega)
    qr_conj = mx.concatenate([w[..., None], -v], axis=-1)
    tq = 2.0 * quat_mul(qd, qr_conj)
    t = tq[..., 1:]
    return _mx.make_T(R, t)


def dlb(weights: mx.array, dqs: mx.array) -> mx.array:
    """Dual-quaternion Linear Blend (MLX). ``weights (..., N)``, ``dqs (..., N, 8)``."""
    qr = dqs[..., :4]
    pivot = qr[..., 0:1, :]
    dot = mx.sum(qr * pivot, axis=-1)
    sign = mx.where(dot < 0.0, -1.0, 1.0)
    dqs_fixed = dqs * sign[..., None]
    b = mx.sum(weights[..., None] * dqs_fixed, axis=-2)
    real_norm = mx.sqrt(mx.maximum(mx.sum(b[..., :4] ** 2, axis=-1), _EPS2))
    return b / real_norm[..., None]


def sclerp(A: mx.array, B: mx.array, u: mx.array) -> mx.array:
    """Screw geodesic ``A . exp(u . log(A^-1 B))`` as a dual quaternion (MLX)."""
    rel = _mx.compose(_mx.inverse(A), B)
    xi = _mx.log_se3(rel)
    Tu = _mx.compose(A, _mx.exp_se3(u[..., None] * xi))
    return se3_to_dq(Tu)
