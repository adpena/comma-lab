# SPDX-License-Identifier: MIT
"""ParetoSolverVerdict — canonical Provenance-bearing solver verdict.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 1 Phase 4 + CLAUDE.md
"Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" +
Catalog #323 canonical Provenance umbrella + Catalog #341 canonical-
routing markers.

The verdict surfaces the canonical compounding-mechanism output:
per-axis dual variables + per-axis KKT residuals + tight-constraint
identification + slack-axis identification + Pareto feasibility verdict.

Per CLAUDE.md "Forbidden score claims" + CLAUDE.md "Apples-to-apples
evidence discipline": every verdict carries ``axis_tag="[predicted]"`` +
``score_claim=False`` + ``promotable=False`` per Catalog #341 canonical-
routing markers. The bounded scalar adjustment factor in [0.95, 1.05]
preserves the META-LAGRANGIAN-WIRE-1 Phase 1 safety envelope contract
per Catalog #355 sister discipline.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class ParetoSolverError(ValueError):
    """Raised when ParetoSolverVerdict inputs violate invariants.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN": every invariant
    enforced in ``__post_init__`` so the construction surface refuses
    bad inputs at the source.
    """


# Per Boyd-Dattorro (2006) § 6.2 + Boyd & Vandenberghe (2004) § 5.5.2:
# the per-axis dual variable extracted from the Dykstra correction vector
# is non-negative for inequality constraints; a dual variable is "tight"
# (binding) iff it is strictly positive. The 1e-6 threshold mirrors the
# canonical sister DYKSTRA_DEFAULT_EPSILON convergence tolerance.
TIGHT_CONSTRAINT_LAMBDA_THRESHOLD: float = 1e-6
"""Threshold below which a per-axis dual variable is considered slack.

Per Boyd-Dattorro (2006) § 6.2: at the KKT saddle point, the per-axis
dual variable ``λ_i ≥ 0`` for inequality constraints. The constraint
is "tight" (binding; the axis IS the next-cycle attack direction) iff
``λ_i > TIGHT_CONSTRAINT_LAMBDA_THRESHOLD``; otherwise the axis is
"slack" (interior; no further attack required on this axis).
"""

PARETO_SOLVER_VERDICT_SCHEMA_VERSION: str = (
    "pareto_solver_verdict_v1_20260528"
)
"""Pinned schema version for ParetoSolverVerdict JSON serialization.

Mirrors :data:`tac.findings_lagrangian.dual_solver_phase_2.PHASE_2_DUAL_SOLVER_SCHEMA_VERSION`
sister versioning discipline. Bump only via explicit migration landing.
"""


@dataclass(frozen=True)
class ParetoSolverVerdict:
    """Canonical Provenance-bearing solver verdict for cathedral autopilot.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 1 Phase 4: this verdict
    is the canonical compounding-mechanism output. The cathedral
    autopilot ranker + canonical equation registry + continual-learning
    posterior all consume this typed contract.

    Per Catalog #323/#341: every verdict is OBSERVABILITY-ONLY at
    landing. The exposed bounded scalar adjustment factor in [0.95, 1.05]
    preserves the Phase 1 META-LAGRANGIAN-WIRE-1 safety envelope per
    Catalog #355 sister discipline.

    Fields
    ------
    candidate_id : str
        Identifier of the candidate this verdict is computed for.
    feasible : bool
        True iff (a) the Dykstra alternating projections converged within
        the iteration cap AND (b) the converged projection point lies
        inside the polytope (within the canonical tolerance). False
        indicates the intersection is empty OR convergence did not
        complete (rare; typically indicates iteration cap too low).
    projection_point : Mapping[str, float]
        The Dykstra-converged projection (closest feasible point to the
        initial point in L2 sense). For feasible verdicts: lies inside
        the polytope. For infeasible verdicts: best-effort projection.
    per_axis_dual_variables : Mapping[str, float]
        Per-axis Lagrangian dual variables ``λ_i ≥ 0``. Extracted from
        the Dykstra correction vector at convergence per Boyd-Dattorro
        (2006) § 6.2.
    tight_constraint_axes : tuple[str, ...]
        Axes where ``λ_i > TIGHT_CONSTRAINT_LAMBDA_THRESHOLD`` (binding
        constraints; next-cycle's highest-EV attack direction). Empty
        when no axis is binding (rare; typically means the initial point
        was already feasible).
    slack_axes : tuple[str, ...]
        Axes where ``λ_i ≈ 0`` (interior; no further attack required on
        this axis). Disjoint from tight_constraint_axes; union covers
        all axes of the polytope.
    per_axis_kkt_residuals : Mapping[str, float]
        Per-axis KKT residuals at the converged projection. Near-zero
        indicates axis feasibility; large positive values indicate
        further iteration would be required.
    per_axis_adjustment_factors : Mapping[str, float]
        Per-axis bounded [0.95, 1.05] adjustment factors per axis.
        Observability-only per Catalog #341.
    adjustment_factor : float
        Exposed scalar adjustment factor in [0.95, 1.05]. Cathedral
        autopilot ranker consumes this value; the per-axis dict is
        observability-only.
    convergence_residual : float
        Final ``||x_{k+1} - x_k||_∞`` at convergence. Should be
        ``≤ epsilon`` for feasible verdicts.
    iteration_count : int
        Number of Dykstra alt-projection iterations performed.
    converged : bool
        True iff the iteration converged before the max-iteration cap.
        Equivalent to ``feasible and iteration_count < max_iterations``.
    canonical_provenance : Mapping[str, Any]
        Catalog #323 canonical Provenance dict (from
        :func:`tac.provenance.builders.build_provenance_for_predicted`
        via :func:`tac.provenance.validator.provenance_to_dict`).
    axis_tag : str
        Canonical axis tag per Catalog #287/#341. Always "[predicted]".
    score_claim : bool
        Always False per Catalog #341.
    promotable : bool
        Always False per Catalog #341.
    """

    candidate_id: str
    feasible: bool
    projection_point: Mapping[str, float]
    per_axis_dual_variables: Mapping[str, float]
    tight_constraint_axes: tuple[str, ...]
    slack_axes: tuple[str, ...]
    per_axis_kkt_residuals: Mapping[str, float]
    per_axis_adjustment_factors: Mapping[str, float]
    adjustment_factor: float
    convergence_residual: float
    iteration_count: int
    converged: bool
    canonical_provenance: Mapping[str, Any] = field(default_factory=dict)
    axis_tag: str = "[predicted]"
    score_claim: bool = False
    promotable: bool = False
    schema_version: str = PARETO_SOLVER_VERDICT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ParetoSolverError("candidate_id must be a non-empty string")
        if not isinstance(self.feasible, bool):
            raise ParetoSolverError("feasible must be bool")
        for mapping_name in (
            "projection_point",
            "per_axis_dual_variables",
            "per_axis_kkt_residuals",
            "per_axis_adjustment_factors",
        ):
            mapping = getattr(self, mapping_name)
            if not isinstance(mapping, Mapping):
                raise ParetoSolverError(
                    f"{mapping_name} must be a Mapping, got {type(mapping).__name__}"
                )
            for axis, value in mapping.items():
                if not isinstance(axis, str):
                    raise ParetoSolverError(
                        f"{mapping_name} key {axis!r} must be a string"
                    )
                if not isinstance(value, (int, float)):
                    raise ParetoSolverError(
                        f"{mapping_name}[{axis!r}]={value!r} must be numeric"
                    )
                if math.isnan(value):
                    raise ParetoSolverError(
                        f"{mapping_name}[{axis!r}] is NaN"
                    )
        # Per-axis dual variables must be non-negative per Boyd-Dattorro § 6.2.
        for axis, dual in self.per_axis_dual_variables.items():
            if dual < -1e-9:
                raise ParetoSolverError(
                    f"per_axis_dual_variables[{axis!r}]={dual} must be >= 0 "
                    "(inequality-constraint dual variables are non-negative "
                    "per Boyd-Dattorro 2006 § 6.2)"
                )
        # tight + slack axes must be disjoint + cover all axes.
        if not isinstance(self.tight_constraint_axes, tuple):
            raise ParetoSolverError(
                "tight_constraint_axes must be a tuple (frozen)"
            )
        if not isinstance(self.slack_axes, tuple):
            raise ParetoSolverError(
                "slack_axes must be a tuple (frozen)"
            )
        tight_set = set(self.tight_constraint_axes)
        slack_set = set(self.slack_axes)
        if tight_set & slack_set:
            raise ParetoSolverError(
                f"tight_constraint_axes and slack_axes must be disjoint; "
                f"overlap: {sorted(tight_set & slack_set)!r}"
            )
        all_axes_in_partition = tight_set | slack_set
        all_axes_in_duals = set(self.per_axis_dual_variables.keys())
        if all_axes_in_partition != all_axes_in_duals:
            raise ParetoSolverError(
                f"tight + slack axes ({sorted(all_axes_in_partition)!r}) must "
                f"partition per_axis_dual_variables axes ({sorted(all_axes_in_duals)!r})"
            )
        # Verify the tight/slack classification respects the threshold.
        for axis, dual in self.per_axis_dual_variables.items():
            if dual > TIGHT_CONSTRAINT_LAMBDA_THRESHOLD:
                if axis not in tight_set:
                    raise ParetoSolverError(
                        f"axis {axis!r} has dual={dual} > threshold "
                        f"{TIGHT_CONSTRAINT_LAMBDA_THRESHOLD} but is not in "
                        "tight_constraint_axes"
                    )
            else:
                if axis not in slack_set:
                    raise ParetoSolverError(
                        f"axis {axis!r} has dual={dual} <= threshold "
                        f"{TIGHT_CONSTRAINT_LAMBDA_THRESHOLD} but is not in "
                        "slack_axes"
                    )
        # Per-axis adjustment factors must be in [0.95, 1.05].
        for axis, af in self.per_axis_adjustment_factors.items():
            if af < 0.95 - 1e-9:
                raise ParetoSolverError(
                    f"per_axis_adjustment_factors[{axis!r}]={af} < 0.95 "
                    "(Catalog #341 routing-marker bound)"
                )
            if af > 1.05 + 1e-9:
                raise ParetoSolverError(
                    f"per_axis_adjustment_factors[{axis!r}]={af} > 1.05 "
                    "(Catalog #341 routing-marker bound)"
                )
        # Exposed scalar adjustment factor must be in [0.95, 1.05].
        if not isinstance(self.adjustment_factor, (int, float)):
            raise ParetoSolverError("adjustment_factor must be numeric")
        if math.isnan(self.adjustment_factor):
            raise ParetoSolverError("adjustment_factor is NaN")
        if self.adjustment_factor < 0.95 - 1e-9:
            raise ParetoSolverError(
                f"adjustment_factor={self.adjustment_factor} < 0.95"
            )
        if self.adjustment_factor > 1.05 + 1e-9:
            raise ParetoSolverError(
                f"adjustment_factor={self.adjustment_factor} > 1.05"
            )
        if not isinstance(self.convergence_residual, (int, float)):
            raise ParetoSolverError("convergence_residual must be numeric")
        if self.convergence_residual < 0:
            raise ParetoSolverError(
                f"convergence_residual={self.convergence_residual} must be >= 0"
            )
        if not isinstance(self.iteration_count, int) or self.iteration_count < 0:
            raise ParetoSolverError(
                "iteration_count must be a non-negative int"
            )
        if not isinstance(self.converged, bool):
            raise ParetoSolverError("converged must be bool")
        if self.axis_tag != "[predicted]":
            raise ParetoSolverError(
                f"axis_tag={self.axis_tag!r} must be '[predicted]' per Catalog #341"
            )
        if self.score_claim is not False:
            raise ParetoSolverError(
                "score_claim must be False per Catalog #341"
            )
        if self.promotable is not False:
            raise ParetoSolverError(
                "promotable must be False per Catalog #341"
            )
        if not isinstance(self.canonical_provenance, Mapping):
            raise ParetoSolverError(
                "canonical_provenance must be a Mapping per Catalog #323"
            )
        if self.schema_version != PARETO_SOLVER_VERDICT_SCHEMA_VERSION:
            raise ParetoSolverError(
                f"schema_version={self.schema_version!r} != canonical "
                f"{PARETO_SOLVER_VERDICT_SCHEMA_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization per Catalog #305 observability surface."""
        return {
            "schema_version": str(self.schema_version),
            "candidate_id": str(self.candidate_id),
            "feasible": bool(self.feasible),
            "projection_point": {
                k: float(v) for k, v in self.projection_point.items()
            },
            "per_axis_dual_variables": {
                k: float(v) for k, v in self.per_axis_dual_variables.items()
            },
            "tight_constraint_axes": list(self.tight_constraint_axes),
            "slack_axes": list(self.slack_axes),
            "per_axis_kkt_residuals": {
                k: float(v) for k, v in self.per_axis_kkt_residuals.items()
            },
            "per_axis_adjustment_factors": {
                k: float(v) for k, v in self.per_axis_adjustment_factors.items()
            },
            "adjustment_factor": float(self.adjustment_factor),
            "convergence_residual": float(self.convergence_residual),
            "iteration_count": int(self.iteration_count),
            "converged": bool(self.converged),
            "axis_tag": str(self.axis_tag),
            "score_claim": bool(self.score_claim),
            "promotable": bool(self.promotable),
            "canonical_provenance": dict(self.canonical_provenance),
        }


__all__ = [
    "ParetoSolverVerdict",
    "ParetoSolverError",
    "TIGHT_CONSTRAINT_LAMBDA_THRESHOLD",
    "PARETO_SOLVER_VERDICT_SCHEMA_VERSION",
]
