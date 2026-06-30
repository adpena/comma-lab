# SPDX-License-Identifier: MIT
"""``tac.lie`` -- an MLX-native, autodiff-clean, parity-gated Lie-group library.

The MLX ecosystem has no Lie-group primitives (jaxlie is JAX; Sophus/manif are
C++; liegroups is numpy/torch). This package provides differentiable
``se(3)``/``SE(3)`` operators, dual-quaternion screw blending, and a cumulative
SE(3) B-spline on Apple-silicon MLX, each cross-checked against a NumPy-fp32
reference oracle (the bit-identical verdict authority) and algebraic identities.

Pure geometry; no contest IP and nothing video-derived. Clean-room from textbook
math (Sola micro-Lie arXiv:1812.01537; Barfoot 2017; Sommer et al. CVPR 2020;
Kavan et al. ACM TOG 2008). MIT-licensed give-back component.

Convention (fixed once, asserted in tests)
-------------------------------------------
Twist ``xi = (rho, omega)`` is **translation-first**: ``rho = xi[..., :3]``,
``omega = xi[..., 3:]``. SE(3) elements are ``(..., 4, 4)`` homogeneous
matrices. Quaternions are scalar-first ``[w, x, y, z]``. Dual quaternions are
``(..., 8)`` = real quat then dual quat.

Standalone
----------
This package imports NOTHING from the witness residual pipeline. The witness
design-refine step (canonicalize-to-ground-frame, per-class screw warp, and the
``xi_ego(t)`` spline fit) will CONSUME ``tac.lie``; the wire-in happens there,
not here.

Layout
------
* ``so3`` / ``se3``     -- MLX fast path (SO(3)/SE(3) exp/log/Adjoint/Jacobians).
* ``_se3_numpy``        -- NumPy-fp32 reference oracle (the authority).
* ``screw_blend``       -- dual-quaternion DLB + ScLERP (numpy + MLX).
* ``se3_bspline``       -- cumulative SE(3) B-spline (numpy + MLX).
"""

from __future__ import annotations

from . import _se3_numpy, screw_blend, se3, se3_bspline, so3
from .se3 import (
    CONVENTION,
    adjoint_se3,
    adjoint_T,
    compose,
    exp_se3,
    inverse,
    left_jacobian_se3,
    log_se3,
    make_T,
    rotation_of,
    translation_of,
)
from .so3 import (
    exp_so3,
    left_jacobian_inv_so3,
    left_jacobian_so3,
    log_so3,
    right_jacobian_inv_so3,
    right_jacobian_so3,
    skew,
    unskew,
)

__all__ = [
    "CONVENTION",
    # submodules
    "so3",
    "se3",
    "screw_blend",
    "se3_bspline",
    "_se3_numpy",
    # SO(3) (MLX)
    "skew",
    "unskew",
    "exp_so3",
    "log_so3",
    "left_jacobian_so3",
    "right_jacobian_so3",
    "left_jacobian_inv_so3",
    "right_jacobian_inv_so3",
    # SE(3) (MLX)
    "make_T",
    "rotation_of",
    "translation_of",
    "exp_se3",
    "log_se3",
    "compose",
    "inverse",
    "adjoint_T",
    "adjoint_se3",
    "left_jacobian_se3",
]
