# SPDX-License-Identifier: MIT
"""Gauge-fixed Fisher-natural trust-region projection for task #500/#504.

This additive module deliberately consumes the source-sealed typed solve in
``bregman_v9_surfaces`` instead of changing that module's strict #504 binding
bytes.  It is a local geometry primitive, not a V9 optimizer activation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tac.information_geometry.bregman_v9_surfaces import (
    GeometryValidationError,
    solve_fisher_natural_cotangent,
)

FloatVector = NDArray[np.float64]


@dataclass(frozen=True)
class FisherNaturalTrustRegionStep:
    """One gauge-fixed Fisher-natural step projected into an ``H``-norm ball.

    ``step`` has the sign supplied by ``cotangent``. An optimizer seeking descent
    should pass the negative objective cotangent; this helper owns only the
    metric solve and radius projection, not the optimization sign convention.
    """

    step: FloatVector
    unconstrained_step: FloatVector
    unconstrained_hessian_norm: float
    realized_hessian_norm: float
    radius: float
    scale: float
    boundary_active: bool
    solve_residual_linf: float


def fisher_natural_cotangent_trust_region_step(
    hessian: ArrayLike,
    cotangent: ArrayLike,
    *,
    radius: float,
) -> FisherNaturalTrustRegionStep:
    """Return ``min(1, r / ||H^-1 g||_H) H^-1 g`` without forming ``H^-1``.

    The categorical Fisher matrix is singular in the ambient additive-logit
    gauge. This primitive therefore accepts only an already gauge-fixed SPD
    chart, as enforced by the sealed typed solve. Constructing that quotient
    chart and measuring the live renderer/Fisher pullback remain consumer
    obligations; implicit damping here would silently change the geometry.
    """

    try:
        trust_radius = float(radius)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(
            f"radius must be a finite non-negative scalar: {exc}"
        ) from exc
    if not np.isfinite(trust_radius) or trust_radius < 0.0:
        raise GeometryValidationError("radius must be a finite non-negative scalar")

    # The sealed solve performs the authoritative finite/vector/SPD validation.
    unconstrained = solve_fisher_natural_cotangent(hessian, cotangent)
    matrix = np.asarray(hessian, dtype=np.float64)
    vector = np.asarray(cotangent, dtype=np.float64)
    natural_norm_sq = float(vector @ unconstrained)
    zero_tol = np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(vector)) ** 2)
    if natural_norm_sq < -zero_tol:
        raise GeometryValidationError(
            "Fisher-natural squared norm became negative for an SPD Hessian"
        )
    natural_norm = float(np.sqrt(max(0.0, natural_norm_sq)))
    scale = 1.0 if natural_norm == 0.0 else min(1.0, trust_radius / natural_norm)
    step = np.asarray(scale * unconstrained, dtype=np.float64)
    realized_norm_sq = float(step @ matrix @ step)
    if not np.isfinite(realized_norm_sq):
        raise GeometryValidationError(
            "Fisher-natural trust-region projection produced a non-finite norm"
        )
    realized_norm = float(np.sqrt(max(0.0, realized_norm_sq)))
    radius_tol = 32.0 * np.finfo(np.float64).eps * max(1.0, trust_radius)
    if realized_norm > trust_radius + radius_tol:
        raise GeometryValidationError(
            "Fisher-natural trust-region projection exceeded its Hessian-norm radius"
        )
    residual = float(np.max(np.abs(matrix @ unconstrained - vector), initial=0.0))
    return FisherNaturalTrustRegionStep(
        step=step,
        unconstrained_step=np.asarray(unconstrained, dtype=np.float64),
        unconstrained_hessian_norm=natural_norm,
        realized_hessian_norm=realized_norm,
        radius=trust_radius,
        scale=float(scale),
        boundary_active=bool(scale < 1.0),
        solve_residual_linf=residual,
    )


__all__ = [
    "FisherNaturalTrustRegionStep",
    "fisher_natural_cotangent_trust_region_step",
]
