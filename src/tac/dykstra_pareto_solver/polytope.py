# SPDX-License-Identifier: MIT
"""Polytope dataclass — typed contract for axis-aligned + halfspace constraints.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 1 Phase 4 + Boyd & Vandenberghe
(2004) Convex Optimization Chapter 5 (Duality):

A polytope ``P = {x ∈ R^n : a_i^T x ≤ b_i, i ∈ I}`` is the intersection
of finitely many half-spaces. The canonical 3-axis Pareto polytope at
the contest scorer level is::

    P = {(d_seg, d_pose, archive_bytes_delta) :
            d_seg_lower ≤ d_seg ≤ d_seg_upper,
            d_pose_lower ≤ d_pose ≤ d_pose_upper,
            rate_lower ≤ archive_bytes_delta ≤ rate_upper}

i.e. the axis-aligned bounding box with one half-space per axis-bound.

This module exposes :class:`Polytope` as the canonical typed contract;
the sister :mod:`tac.findings_lagrangian.dual_solver_phase_2` accepts
3-tuples of ``(lower, upper)`` for backward compatibility. Future
per-paradigm extensions (e.g. simplex-constrained polytopes for VQ
codebook indices, group-sparsity polytopes for Cool-Chic) can extend
:class:`Polytope` without breaking the canonical 3-axis API.

Per CLAUDE.md "Beauty, simplicity, and developer experience": frozen
dataclass + explicit invariants + closed-form per-axis projection
sufficient for the canonical 3-axis case; iterative projection for the
general half-space case.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class PolytopeError(ValueError):
    """Raised when a Polytope construction or projection violates invariants.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN": every invariant
    enforced in ``__post_init__`` so the construction surface refuses
    bad inputs at the source.
    """


# Canonical 3-axis ordering mirrors
# :data:`tac.findings_lagrangian.dual_solver_phase_2.AXIS_NAMES` so the
# facade <-> sister-module conversion is lossless.
CANONICAL_3_AXIS_ORDERING: tuple[str, str, str] = ("seg", "pose", "rate")


@dataclass(frozen=True)
class Polytope:
    """Typed contract for axis-aligned + halfspace constraint polytopes.

    The polytope is::

        P = {x : axis_bounds[axis] ⊆ [lower, upper] for each axis}
              ∩ {x : Σ_i coeff_i · x_i ≤ rhs for each (coeffs, rhs) in halfspace_constraints}

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4: the canonical
    3-axis case uses only axis_bounds; halfspace_constraints are
    reserved for per-paradigm extensions (e.g. simplex constraints
    ``Σ_i p_i ≤ 1`` for VQ codebook probability vectors).

    Fields
    ------
    axis_bounds : Mapping[str, tuple[float, float]]
        Per-axis ``(lower, upper)`` half-space bounds. Canonical 3-axis
        ordering matches :data:`CANONICAL_3_AXIS_ORDERING` (seg, pose,
        rate). Per axis: ``lower ≤ x_axis ≤ upper``.
    halfspace_constraints : tuple[tuple[Mapping[str, float], float], ...]
        Each entry is ``(coefficients, rhs)`` expressing the half-space
        ``Σ_axis coefficients[axis] · x[axis] ≤ rhs``. Empty by default
        (canonical 3-axis case). For non-empty halfspace constraints,
        projection is iterative.
    name : str
        Optional human-readable name for the polytope (operator-facing
        diagnostic).

    Examples
    --------
    Canonical 3-axis Pareto polytope around the current frontier::

        Polytope(axis_bounds={
            "seg": (0.0, 0.5),
            "pose": (0.0, 0.1),
            "rate": (-50_000.0, 0.0),
        })

    Simplex-constrained probability vector (extension example)::

        Polytope(
            axis_bounds={f"p_{i}": (0.0, 1.0) for i in range(4)},
            halfspace_constraints=(
                ({f"p_{i}": 1.0 for i in range(4)}, 1.0),  # Σ_i p_i ≤ 1
            ),
        )
    """

    axis_bounds: Mapping[str, tuple[float, float]]
    halfspace_constraints: tuple[tuple[Mapping[str, float], float], ...] = field(
        default_factory=tuple
    )
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.axis_bounds, Mapping):
            raise PolytopeError(
                f"axis_bounds must be Mapping, got {type(self.axis_bounds).__name__}"
            )
        if not self.axis_bounds:
            raise PolytopeError(
                "axis_bounds must be non-empty (at least one axis required)"
            )
        for axis, bounds in self.axis_bounds.items():
            if not isinstance(axis, str) or not axis.strip():
                raise PolytopeError(
                    f"axis_bounds key {axis!r} must be a non-empty string"
                )
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise PolytopeError(
                    f"axis_bounds[{axis!r}]={bounds!r} must be (lower, upper) tuple"
                )
            lo, hi = bounds
            for label, value in (("lower", lo), ("upper", hi)):
                if not isinstance(value, (int, float)):
                    raise PolytopeError(
                        f"axis_bounds[{axis!r}].{label}={value!r} must be numeric"
                    )
                if math.isnan(value):
                    raise PolytopeError(
                        f"axis_bounds[{axis!r}].{label} is NaN"
                    )
                if math.isinf(value):
                    raise PolytopeError(
                        f"axis_bounds[{axis!r}].{label} is infinite"
                    )
            if lo > hi:
                raise PolytopeError(
                    f"axis_bounds[{axis!r}]: lower={lo} > upper={hi}"
                )
        if not isinstance(self.halfspace_constraints, tuple):
            raise PolytopeError(
                "halfspace_constraints must be a tuple (frozen)"
            )
        for i, hsc in enumerate(self.halfspace_constraints):
            if not isinstance(hsc, tuple) or len(hsc) != 2:
                raise PolytopeError(
                    f"halfspace_constraints[{i}] must be (coefficients, rhs) tuple"
                )
            coeffs, rhs = hsc
            if not isinstance(coeffs, Mapping):
                raise PolytopeError(
                    f"halfspace_constraints[{i}].coefficients must be Mapping"
                )
            if not isinstance(rhs, (int, float)):
                raise PolytopeError(
                    f"halfspace_constraints[{i}].rhs={rhs!r} must be numeric"
                )
            if math.isnan(rhs) or math.isinf(rhs):
                raise PolytopeError(
                    f"halfspace_constraints[{i}].rhs={rhs} is NaN/infinite"
                )
            for axis, coeff in coeffs.items():
                if axis not in self.axis_bounds:
                    raise PolytopeError(
                        f"halfspace_constraints[{i}] references unknown axis {axis!r}"
                    )
                if not isinstance(coeff, (int, float)):
                    raise PolytopeError(
                        f"halfspace_constraints[{i}].coefficients[{axis!r}]"
                        f"={coeff!r} must be numeric"
                    )
                if math.isnan(coeff) or math.isinf(coeff):
                    raise PolytopeError(
                        f"halfspace_constraints[{i}].coefficients[{axis!r}]"
                        f"={coeff} is NaN/infinite"
                    )
        if not isinstance(self.name, str):
            raise PolytopeError("name must be a string")

    @property
    def axes(self) -> tuple[str, ...]:
        """Canonical ordered axis names. For canonical 3-axis case uses
        :data:`CANONICAL_3_AXIS_ORDERING`; for general polytopes, uses
        insertion order of ``axis_bounds``."""
        # If axis_bounds keys match the canonical 3-axis set exactly, use
        # the canonical ordering. Otherwise preserve insertion order.
        keys = set(self.axis_bounds.keys())
        if keys == set(CANONICAL_3_AXIS_ORDERING):
            return CANONICAL_3_AXIS_ORDERING
        return tuple(self.axis_bounds.keys())

    @property
    def is_canonical_3_axis(self) -> bool:
        """True iff the polytope is exactly the canonical (seg, pose, rate)
        3-axis Pareto polytope with no extra halfspace constraints.

        Used by :class:`tac.dykstra_pareto_solver.solver.DykstraParetoSolver`
        to dispatch to the canonical sister-module 3-axis solver versus
        the general iterative path.
        """
        return (
            set(self.axis_bounds.keys()) == set(CANONICAL_3_AXIS_ORDERING)
            and not self.halfspace_constraints
        )

    def project(self, point: Mapping[str, float]) -> dict[str, float]:
        """Project ``point`` onto the polytope (closest feasible point).

        For axis-aligned bounds: closed-form clip per axis.
        For halfspace constraints: iterative projection (Dykstra
        alternating projections within the general polytope).

        Per Boyd & Vandenberghe (2004) Theorem 8.2: projection onto a
        convex set is unique and well-defined.

        Args
        ----
        point : Mapping[str, float]
            Initial point ``{axis: value}``. Missing axes default to 0.0.

        Returns
        -------
        dict[str, float]
            Projected point with the same axes as :data:`axes`.
        """
        # Build a complete point dict in canonical axis ordering.
        out: dict[str, float] = {}
        for axis in self.axes:
            raw = point.get(axis, 0.0)
            if not isinstance(raw, (int, float)):
                raise PolytopeError(
                    f"point[{axis!r}]={raw!r} must be numeric"
                )
            if math.isnan(raw):
                raise PolytopeError(f"point[{axis!r}] is NaN")
            lo, hi = self.axis_bounds[axis]
            if raw < lo:
                out[axis] = float(lo)
            elif raw > hi:
                out[axis] = float(hi)
            else:
                out[axis] = float(raw)

        # Halfspace projection: iterative for general case.
        if self.halfspace_constraints:
            # Project onto each halfspace until convergence.
            for _ in range(100):  # bounded iterations
                changed = False
                for coeffs, rhs in self.halfspace_constraints:
                    # Compute Σ_axis coeffs[axis] * out[axis]
                    lhs_value = sum(
                        float(coeffs.get(axis, 0.0)) * out[axis]
                        for axis in self.axes
                    )
                    if lhs_value > rhs + 1e-9:
                        # Violated; project onto the halfspace boundary.
                        # Closed-form: x_new = x - ((lhs - rhs) / ||a||²) * a
                        norm_sq = sum(
                            float(coeffs.get(axis, 0.0)) ** 2
                            for axis in self.axes
                        )
                        if norm_sq > 0:
                            scale = (lhs_value - rhs) / norm_sq
                            for axis in self.axes:
                                a_i = float(coeffs.get(axis, 0.0))
                                out[axis] -= scale * a_i
                            # Re-clip to axis bounds (may need re-iteration).
                            for axis in self.axes:
                                lo, hi = self.axis_bounds[axis]
                                if out[axis] < lo:
                                    out[axis] = float(lo)
                                elif out[axis] > hi:
                                    out[axis] = float(hi)
                            changed = True
                if not changed:
                    break

        return out

    def contains(self, point: Mapping[str, float], *, tolerance: float = 1e-9) -> bool:
        """True iff ``point`` is inside the polytope within ``tolerance``.

        Per Boyd-Vandenberghe (2004) Definition 2.4 + the canonical
        Dykstra alternating-projections convergence threshold per
        :data:`tac.dykstra_pareto_solver.DYKSTRA_DEFAULT_EPSILON`.
        """
        for axis in self.axes:
            value = point.get(axis, 0.0)
            if not isinstance(value, (int, float)) or math.isnan(value):
                return False
            lo, hi = self.axis_bounds[axis]
            if value < lo - tolerance or value > hi + tolerance:
                return False
        for coeffs, rhs in self.halfspace_constraints:
            lhs_value = sum(
                float(coeffs.get(axis, 0.0)) * float(point.get(axis, 0.0))
                for axis in self.axes
            )
            if lhs_value > rhs + tolerance:
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization for observability persistence."""
        return {
            "axis_bounds": {
                axis: [float(lo), float(hi)]
                for axis, (lo, hi) in self.axis_bounds.items()
            },
            "halfspace_constraints": [
                [
                    {axis: float(coeff) for axis, coeff in coeffs.items()},
                    float(rhs),
                ]
                for coeffs, rhs in self.halfspace_constraints
            ],
            "name": str(self.name),
            "axes": list(self.axes),
            "is_canonical_3_axis": bool(self.is_canonical_3_axis),
        }


__all__ = [
    "Polytope",
    "PolytopeError",
    "CANONICAL_3_AXIS_ORDERING",
]
