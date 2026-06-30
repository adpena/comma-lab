# SPDX-License-Identifier: MIT
"""Cumulative SE(3) B-spline for a continuous-time trajectory xi_ego(t).

Uniform cubic (k=4) cumulative B-spline on SE(3), Kim--Kim--Shin (SIGGRAPH
1995) generalized to Lie groups by Sommer et al. (CVPR 2020, arXiv:1911.08860).
Implemented FRESH from the paper formulas -- basalt is MPL-2.0 (not copied).

A segment governed by four control poses ``{T_0, T_1, T_2, T_3}`` and local
time ``u in [0, 1)`` evaluates to::

    T(u) = T_0 . prod_{j=1}^{3} exp( B_j(u) . Omega_j ),
    Omega_j = log( T_{j-1}^-1 . T_{j} )

with the uniform cubic cumulative basis::

    B_1(u) = (5 + 3u - 3u^2 +  u^3) / 6
    B_2(u) = (1 + 3u + 3u^2 - 2u^3) / 6
    B_3(u) = (u^3) / 6

The whole thing is a product of ``exp`` of constant twists scaled by scalar
bases, so MLX autodiff gives both ``dT/du`` (velocity) and the control-pose
gradients with NO custom VJP -- only ``exp``/``log`` need stable branches.

NumPy reference oracle (``*_numpy``) + MLX fast path (unsuffixed). Control poses
are passed as a stacked ``(..., M, 4, 4)`` array (M >= 4).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import _se3_numpy as _np
from . import se3 as _mx

__all__ = [
    "cumulative_basis_numpy",
    "se3_bspline_segment_numpy",
    "se3_bspline_eval_numpy",
    "cumulative_basis",
    "se3_bspline_segment",
    "se3_bspline_eval",
]


# --------------------------------------------------------------------------- #
# NumPy reference oracle
# --------------------------------------------------------------------------- #
def cumulative_basis_numpy(u: np.ndarray) -> np.ndarray:
    """Uniform cubic cumulative basis ``(B_1, B_2, B_3)`` -> ``(..., 3)``."""
    u = np.asarray(u, dtype=float)
    u2 = u * u
    u3 = u2 * u
    b1 = (5.0 + 3.0 * u - 3.0 * u2 + u3) / 6.0
    b2 = (1.0 + 3.0 * u + 3.0 * u2 - 2.0 * u3) / 6.0
    b3 = u3 / 6.0
    return np.stack([b1, b2, b3], axis=-1)


def se3_bspline_segment_numpy(controls4: np.ndarray, u) -> np.ndarray:
    """Evaluate one segment. ``controls4 (..., 4, 4, 4)``, ``u`` scalar ``(...,)``."""
    controls4 = np.asarray(controls4, dtype=float)
    T0 = controls4[..., 0, :, :]
    T1 = controls4[..., 1, :, :]
    T2 = controls4[..., 2, :, :]
    T3 = controls4[..., 3, :, :]
    O1 = _np.log_se3(_np.compose(_np.inverse(T0), T1))
    O2 = _np.log_se3(_np.compose(_np.inverse(T1), T2))
    O3 = _np.log_se3(_np.compose(_np.inverse(T2), T3))
    B = cumulative_basis_numpy(u)  # (..., 3)
    T = _np.compose(T0, _np.exp_se3(B[..., 0:1] * O1))
    T = _np.compose(T, _np.exp_se3(B[..., 1:2] * O2))
    T = _np.compose(T, _np.exp_se3(B[..., 2:3] * O3))
    return T


def se3_bspline_eval_numpy(controls: np.ndarray, t) -> np.ndarray:
    """Evaluate the spline at scalar time ``t in [0, M-3)``.

    Segment ``s = floor(t)`` uses controls ``[s, s+1, s+2, s+3]``; ``u = t - s``.
    ``t`` may be a scalar or a 1-D array of query times. ``t`` is clamped to the
    valid domain ``[0, M-3]`` (so ``u`` stays in ``[0, 1]``) -- out-of-domain
    queries return the boundary pose rather than extrapolating wildly.
    """
    controls = np.asarray(controls, dtype=float)
    M = controls.shape[-3]
    n_seg = M - 3
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    s = np.clip(np.floor(t_arr).astype(int), 0, n_seg - 1)
    u = np.clip(t_arr - s, 0.0, 1.0)
    out = []
    for k in range(t_arr.shape[0]):
        c4 = controls[..., s[k] : s[k] + 4, :, :]
        out.append(se3_bspline_segment_numpy(c4, np.asarray(u[k])))
    res = np.stack(out, axis=0)
    return res[0] if np.isscalar(t) or np.asarray(t).ndim == 0 else res


# --------------------------------------------------------------------------- #
# MLX fast path
# --------------------------------------------------------------------------- #
def cumulative_basis(u: mx.array) -> mx.array:
    """Uniform cubic cumulative basis ``(B_1, B_2, B_3)`` -> ``(..., 3)`` (MLX)."""
    u2 = u * u
    u3 = u2 * u
    b1 = (5.0 + 3.0 * u - 3.0 * u2 + u3) / 6.0
    b2 = (1.0 + 3.0 * u + 3.0 * u2 - 2.0 * u3) / 6.0
    b3 = u3 / 6.0
    return mx.stack([b1, b2, b3], axis=-1)


def se3_bspline_segment(controls4: mx.array, u: mx.array) -> mx.array:
    """Evaluate one segment (MLX). ``controls4 (..., 4, 4, 4)``, ``u`` ``(...,)``."""
    T0 = controls4[..., 0, :, :]
    T1 = controls4[..., 1, :, :]
    T2 = controls4[..., 2, :, :]
    T3 = controls4[..., 3, :, :]
    O1 = _mx.log_se3(_mx.compose(_mx.inverse(T0), T1))
    O2 = _mx.log_se3(_mx.compose(_mx.inverse(T1), T2))
    O3 = _mx.log_se3(_mx.compose(_mx.inverse(T2), T3))
    B = cumulative_basis(u)
    T = _mx.compose(T0, _mx.exp_se3(B[..., 0:1] * O1))
    T = _mx.compose(T, _mx.exp_se3(B[..., 1:2] * O2))
    T = _mx.compose(T, _mx.exp_se3(B[..., 2:3] * O3))
    return T


def se3_bspline_eval(controls: mx.array, t: float) -> mx.array:
    """Evaluate the spline at a single scalar time ``t in [0, M-3)`` (MLX).

    Kept scalar-in/matrix-out so it stays differentiable wrt the control poses
    and ``t`` without Python-side gather over a batch.
    """
    M = controls.shape[-3]
    n_seg = M - 3
    s = int(min(max(int(np.floor(float(t))), 0), n_seg - 1))
    u = min(max(float(t) - s, 0.0), 1.0)  # clamp to [0, 1] (boundary pose outside domain)
    c4 = controls[..., s : s + 4, :, :]
    return se3_bspline_segment(c4, mx.array(u, dtype=controls.dtype))
