"""Optimization and allocation utilities for contest archive atoms.

This package ranks, allocates, and composes charged archive atoms. It consumes
analysis records and emits planning policies or ledgers. Exact CUDA auth eval
remains the only score authority.
"""

from tac.optimization.scorer_surface_shaking import (
    OperatingPoint,
    ScorerSurfacePlanError,
    SurfaceAtomFamily,
    build_scorer_surface_shaking_plan,
)

__all__ = [
    "OperatingPoint",
    "ScorerSurfacePlanError",
    "SurfaceAtomFamily",
    "build_scorer_surface_shaking_plan",
]
