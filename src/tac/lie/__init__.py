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

import importlib

# Eager: the NumPy-fp32 reference oracle (the authority) has no MLX
# dependency and is the only surface the CUDA/torch path consumes.
from . import _se3_numpy

# Lazy (PEP 562): every other submodule imports ``mlx.core`` at module top.
# MLX is Apple-silicon-only, so an eager import made ``import tac.lie`` (and
# thus the CUDA pose-carrier attach, 2026-07-15 r2 smoke, rc=1 at 147.9s on
# H100) impossible on any non-macOS container. Accessing any MLX name below
# still imports the real submodule on demand — behavior on macOS is unchanged.
_MLX_SUBMODULES = frozenset({"so3", "se3", "screw_blend", "se3_bspline"})
_LAZY_EXPORT_OWNERS = {
    # se3 (MLX)
    "CONVENTION": "se3",
    "adjoint_se3": "se3",
    "adjoint_T": "se3",
    "compose": "se3",
    "exp_se3": "se3",
    "inverse": "se3",
    "left_jacobian_se3": "se3",
    "log_se3": "se3",
    "make_T": "se3",
    "rotation_of": "se3",
    "translation_of": "se3",
    # so3 (MLX)
    "exp_so3": "so3",
    "left_jacobian_inv_so3": "so3",
    "left_jacobian_so3": "so3",
    "log_so3": "so3",
    "right_jacobian_inv_so3": "so3",
    "right_jacobian_so3": "so3",
    "skew": "so3",
    "unskew": "so3",
}


def __getattr__(name: str):
    if name in _MLX_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    owner = _LAZY_EXPORT_OWNERS.get(name)
    if owner is not None:
        return getattr(importlib.import_module(f".{owner}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))

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
