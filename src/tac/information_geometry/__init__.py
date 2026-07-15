"""Deterministic information-geometry primitives."""

from tac.information_geometry.bregman_v9_surfaces import (
    DELTA_ETA_CONSISTENCY_ATOL,
    DELTA_ETA_CONSISTENCY_RTOL,
    GeometryValidationError,
    fisher_natural_cotangent_quadratic,
    local_hessian_dual_geometry_summary,
    primal_hessian_quadratic,
    raw_dual_euclidean_quadratic,
    solve_fisher_natural_cotangent,
    squared_hessian_quadratic,
)

__all__ = [
    "DELTA_ETA_CONSISTENCY_ATOL",
    "DELTA_ETA_CONSISTENCY_RTOL",
    "GeometryValidationError",
    "fisher_natural_cotangent_quadratic",
    "local_hessian_dual_geometry_summary",
    "primal_hessian_quadratic",
    "raw_dual_euclidean_quadratic",
    "solve_fisher_natural_cotangent",
    "squared_hessian_quadratic",
]
