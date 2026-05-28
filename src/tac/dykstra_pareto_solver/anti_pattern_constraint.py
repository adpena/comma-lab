# SPDX-License-Identifier: MIT
"""AntiPatternConstraint — Layer 5 anti-pattern polytope exclusion.

Per canonical anti-patterns design memo
``.omx/research/canonical_anti_patterns_registry_design_20260528.md`` §"Layer 5"
+ Layer 1+2 landing memo
``.omx/research/canonical_anti_patterns_registry_layer_1_plus_2_landed_20260528.md``
+ Wave N+2 mandate. This module converts registered :class:`AntiPattern`
matches into ACTIVE :class:`DykstraParetoSolver` polytope-exclusion
constraints so the cathedral autopilot ranker is STEERED AWAY from
known anti-pattern compounding routes per iteration.

Per the design memo §"Mathematical compounding identity"::

    NextCycleAttackDirection = argmax_axis (
        PredictedΔS_axis_i × λ_axis_i_tight_from_Dykstra
    ) subject to (
        NOT any (proposed_stack matches AntiPattern_j AND not waived)
    )

THIS module implements the RIGHT side of the constraint at the Dykstra
solver surface. Where the Layer-1 pattern matcher returns matches at
the consumer surface (observability-only per Catalog #341 Tier A), the
Layer-5 :class:`AntiPatternConstraint` makes those matches FEASIBILITY
CONSTRAINTS that the Pareto polytope solver MUST honor — anti-pattern-
matching regions are excluded from the projection's feasible set.

Per Catalog #287/#323/#341: the constraint surface remains OBSERVABILITY-
SAFE (no score-claim mutation). The Pareto verdict's per-axis dual
variables now include ``anti_pattern_<id>`` rows that surface WHICH
anti-pattern is binding; the cathedral autopilot routes the next-cycle
attack via ``canonical_unwind_path`` per Catalog #125 hook #6
disambiguator semantics.

Per CLAUDE.md "Forbidden premature KILL": a matched anti-pattern
constraint does NOT KILL the candidate's research lane. It excludes
the specific anti-pattern-matching REGION from the polytope feasibility
set; the candidate's projection moves toward the canonical unwind path
which IS a viable alternative the operator can pursue.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST
EMPHASIS" + Boyd & Vandenberghe (2004) Convex Optimization Chapter 5:
the Lagrangian dual ``L(x, λ) = f(x) + Σ_i λ_i · g_i(x)`` extends
naturally to anti-pattern indicator constraints
``g_anti_pattern_j(x) = indicator(x ∈ AntiPatternForbiddenRegion_j) ≤ 0``.
The dual variable ``λ_anti_pattern_j > 0`` when the constraint is
TIGHT (binding); ``≈ 0`` when slack. Per-axis TIGHT identification
extends to per-anti-pattern TIGHT identification.

Cross-references:
  * :mod:`tac.canonical_anti_patterns.pattern_matcher` — Layer 1 matcher
    that returns :class:`AntiPatternMatch` rows the constraint wraps.
  * :class:`tac.dykstra_pareto_solver.solver.DykstraParetoSolver` — the
    solver that consumes these constraints as ACTIVE polytope exclusions.
  * :class:`tac.dykstra_pareto_solver.verdict.ParetoSolverVerdict` —
    per-anti-pattern dual variables surfaced via the
    ``per_axis_dual_variables`` mapping with ``anti_pattern_<id>`` keys.
  * :func:`tools/cathedral_autopilot_autonomous_loop.invoke_dykstra_pareto_solver_on_candidates`
    — the Wave N+2 invoker callsite that derives constraints per
    candidate.

Math contract per Boyd-Vandenberghe Chapter 5
---------------------------------------------

For each ``AntiPatternConstraint(anti_pattern_id, forbidden_region_predicate, severity, canonical_unwind_path)``:

1. The forbidden region ``F_j ⊆ R^n`` is defined by the predicate
   ``forbidden_region_predicate(x) -> bool``.
2. The constraint is the indicator inequality
   ``g_j(x) = 1[x ∈ F_j] · severity_weight_j ≤ 0`` so the feasible set
   is ``P ∩ (F_j)^c`` (polytope minus the forbidden region).
3. The Lagrangian becomes ``L(x, λ) = f(x) + Σ_j λ_j · g_j(x)``.
4. At the KKT saddle, ``λ_j ≥ 0`` with ``λ_j · g_j(x*) = 0``
   (complementary slackness).
5. ``λ_j > 0`` iff the constraint is binding (the projection landed at
   the boundary of ``F_j``); cathedral autopilot routes the canonical
   unwind path for binding anti-patterns per Catalog #125 hook #6.

The MAX-aggregation identity per the design memo §"Mathematical
compounding identity" applies when multiple anti-patterns are matched:
``λ_aggregate = max_j (λ_j × severity_weight_j)`` rather than sum,
because one violated anti-pattern can corrupt the entire stack.

Per CLAUDE.md "Beauty, simplicity, and developer experience": frozen
dataclass + explicit invariants + Callable predicate that closes over
candidate-derived data so the polytope's feasibility check is
deterministic and re-projection on the same point is idempotent.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable


class AntiPatternConstraintError(ValueError):
    """Raised when AntiPatternConstraint inputs violate invariants.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN": every invariant
    enforced in ``__post_init__`` so the construction surface refuses
    bad inputs at the source.
    """


# Canonical severity weights for the MAX-aggregation identity per the
# design memo §"Mathematical compounding identity". Higher severity =
# larger weight contribution to the per-anti-pattern dual variable.
# Mirrors :data:`tac.canonical_anti_patterns.anti_pattern.VALID_SEVERITIES`
# ordering (CRITICAL > HIGH > MEDIUM > LOW) per the canonical taxonomy.
_SEVERITY_WEIGHTS: Mapping[str, float] = {
    "critical_paradigm_blocker": 1.0,
    "high_compound_corruption": 0.75,
    "medium_substrate_regression": 0.50,
    "low_implementation_inefficiency": 0.25,
}
"""Per-severity weight applied to the constraint's dual variable.

The constraint dual ``λ_anti_pattern_j = severity_weight × indicator(F_j)``;
this preserves the MAX-aggregation identity since the per-anti-pattern
dual is monotone in severity.
"""

VALID_SEVERITY_WEIGHTS: frozenset[str] = frozenset(_SEVERITY_WEIGHTS.keys())

ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX: str = "anti_pattern_"
"""Prefix for per-anti-pattern dual variable keys in ParetoSolverVerdict.

Per the Wave N+2 mandate Layer 5 contract: per-anti-pattern dual variables
surface in :attr:`ParetoSolverVerdict.per_axis_dual_variables` with key
``anti_pattern_<id>`` so downstream consumers can distinguish axis-dual
vs anti-pattern-dual via prefix inspection.
"""


def severity_weight_for(severity: str) -> float:
    """Return the canonical severity weight; raises on unknown severity.

    Per the design memo §"Mathematical compounding identity": the
    severity weight is applied multiplicatively to the indicator so the
    per-anti-pattern dual variable preserves the MAX-aggregation
    identity across applicable anti-patterns.
    """
    if severity not in _SEVERITY_WEIGHTS:
        raise AntiPatternConstraintError(
            f"severity={severity!r} must be one of "
            f"{sorted(VALID_SEVERITY_WEIGHTS)!r} per the canonical "
            "anti-patterns severity taxonomy."
        )
    return _SEVERITY_WEIGHTS[severity]


@dataclass(frozen=True)
class AntiPatternConstraint:
    """One anti-pattern polytope-exclusion constraint for the Dykstra solver.

    Per Wave N+2 Layer 5 mandate + canonical anti-patterns design memo:
    converts a registered :class:`AntiPattern` match into an ACTIVE
    feasibility constraint on the :class:`DykstraParetoSolver` polytope.
    Points satisfying ``forbidden_region_predicate(point) is True`` are
    EXCLUDED from the feasible set; the projection step moves them
    toward the canonical unwind path.

    Per Catalog #341 + Catalog #357: constraint contribution is
    OBSERVABILITY-SAFE. The constraint does NOT mutate score; it
    reshapes the polytope feasibility set so subsequent operations
    inherit the anti-pattern boundary structurally.

    Fields
    ------
    anti_pattern_id : str
        References the registered :class:`AntiPattern` per
        :mod:`tac.canonical_anti_patterns.registry`. Required for dual-
        variable surfacing (``anti_pattern_<id>`` key) + per-anti-pattern
        ratification routing per Catalog #125 hook #5.
    forbidden_region_predicate : Callable[[Mapping[str, float]], bool]
        Returns True iff the input point lies INSIDE the anti-pattern's
        forbidden region. The Callable closes over candidate-derived
        data (e.g. compression ops list, quantization parameters) so the
        feasibility check is deterministic and idempotent on the same
        point. Must NOT mutate state per CLAUDE.md "Beauty, simplicity,
        and developer experience".
    severity : str
        One of :data:`VALID_SEVERITY_WEIGHTS` keys (canonical anti-
        pattern severity taxonomy). Used to scale the per-anti-pattern
        dual variable per the MAX-aggregation identity.
    canonical_unwind_path : str
        Pass-through from the matched :class:`AntiPattern` per the
        design memo's "canonical correct alternative" contract. Routed
        as the canonical disambiguator per Catalog #125 hook #6 when
        the constraint is binding.
    """

    anti_pattern_id: str
    forbidden_region_predicate: Callable[[Mapping[str, float]], bool]
    severity: str
    canonical_unwind_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.anti_pattern_id, str) or not self.anti_pattern_id.strip():
            raise AntiPatternConstraintError(
                "anti_pattern_id must be a non-empty string"
            )
        if not callable(self.forbidden_region_predicate):
            raise AntiPatternConstraintError(
                "forbidden_region_predicate must be a Callable returning bool"
            )
        if self.severity not in VALID_SEVERITY_WEIGHTS:
            raise AntiPatternConstraintError(
                f"severity={self.severity!r} must be one of "
                f"{sorted(VALID_SEVERITY_WEIGHTS)!r} per the canonical "
                "anti-patterns severity taxonomy."
            )
        if not isinstance(self.canonical_unwind_path, str) or not self.canonical_unwind_path.strip():
            raise AntiPatternConstraintError(
                "canonical_unwind_path must be a non-empty string"
            )

    @property
    def severity_weight(self) -> float:
        """Canonical severity weight per :data:`_SEVERITY_WEIGHTS`."""
        return severity_weight_for(self.severity)

    @property
    def dual_variable_key(self) -> str:
        """Per-anti-pattern dual variable key for ParetoSolverVerdict.

        Per Wave N+2 Layer 5 mandate: the per-anti-pattern dual variable
        surfaces in :attr:`ParetoSolverVerdict.per_axis_dual_variables`
        with key ``anti_pattern_<id>``. Downstream consumers distinguish
        axis-dual vs anti-pattern-dual via prefix inspection.
        """
        return f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}{self.anti_pattern_id}"

    def evaluate_at(self, point: Mapping[str, float]) -> bool:
        """Return True iff the point lies INSIDE the forbidden region.

        Equivalent to ``self.forbidden_region_predicate(point)`` but
        wrapped to surface the canonical contract + raise a clearer
        error if the predicate returns a non-bool. Per CLAUDE.md
        "Forbidden patterns" section + Catalog #287: the predicate must
        return bool, not a truthy value, so the constraint semantics are
        explicit at the call site.
        """
        verdict = self.forbidden_region_predicate(point)
        if not isinstance(verdict, bool):
            raise AntiPatternConstraintError(
                f"forbidden_region_predicate for anti_pattern_id="
                f"{self.anti_pattern_id!r} returned {type(verdict).__name__}; "
                "must return bool"
            )
        return verdict

    def dual_variable(self, point: Mapping[str, float]) -> float:
        """Return the per-anti-pattern dual variable at the given point.

        Per Boyd-Vandenberghe (2004) Chapter 5 + the design memo's
        MAX-aggregation identity: the dual is
        ``λ_j = severity_weight_j × indicator(point ∈ F_j)``. The
        constraint is TIGHT (binding) iff ``λ_j > 0``; cathedral
        autopilot routes the canonical unwind path for binding
        anti-patterns per Catalog #125 hook #6.

        Returns
        -------
        float
            ``severity_weight`` if the point is inside the forbidden
            region; ``0.0`` otherwise.
        """
        if self.evaluate_at(point):
            return float(self.severity_weight)
        return 0.0


def aggregate_anti_pattern_duals(
    point: Mapping[str, float],
    constraints: tuple["AntiPatternConstraint", ...],
) -> tuple[dict[str, float], float, tuple[str, ...]]:
    """Compute per-anti-pattern duals + the MAX-aggregated dual at a point.

    Per the design memo §"Mathematical compounding identity": the
    aggregation across applicable anti-patterns is MAX (severity-
    weighted), NOT SUM. One violated anti-pattern can corrupt the
    entire stack, so the per-anti-pattern dual is dominated by the
    worst applicable pattern at the given point.

    Args
    ----
    point : Mapping[str, float]
        The candidate point to evaluate constraints at.
    constraints : tuple[AntiPatternConstraint, ...]
        The anti-pattern constraints to evaluate.

    Returns
    -------
    tuple[dict[str, float], float, tuple[str, ...]]
        ``(per_constraint_duals, max_dual, binding_canonical_unwind_paths)``
        where:

        - ``per_constraint_duals`` maps each constraint's
          :attr:`AntiPatternConstraint.dual_variable_key` to its dual.
        - ``max_dual`` is the MAX-aggregated dual across all binding
          constraints (``0.0`` if no constraint is binding).
        - ``binding_canonical_unwind_paths`` is the tuple of canonical
          unwind path strings for every binding constraint, ordered by
          descending severity weight (canonical operator-routable
          recommendation order).
    """
    if not isinstance(point, Mapping):
        raise AntiPatternConstraintError(
            f"point must be a Mapping, got {type(point).__name__}"
        )
    if not isinstance(constraints, tuple):
        raise AntiPatternConstraintError(
            f"constraints must be a tuple (frozen), got {type(constraints).__name__}"
        )
    per_constraint: dict[str, float] = {}
    max_dual = 0.0
    binding: list[tuple[float, str]] = []
    for constraint in constraints:
        if not isinstance(constraint, AntiPatternConstraint):
            raise AntiPatternConstraintError(
                f"constraints entries must be AntiPatternConstraint, got "
                f"{type(constraint).__name__}"
            )
        dual = constraint.dual_variable(point)
        per_constraint[constraint.dual_variable_key] = float(dual)
        if dual > 0:
            binding.append((dual, constraint.canonical_unwind_path))
            if dual > max_dual:
                max_dual = float(dual)
    # Sort binding by descending severity weight for canonical
    # operator-routable order.
    binding.sort(key=lambda row: -row[0])
    binding_paths = tuple(path for _weight, path in binding)
    return per_constraint, max_dual, binding_paths


__all__ = [
    "AntiPatternConstraint",
    "AntiPatternConstraintError",
    "ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX",
    "VALID_SEVERITY_WEIGHTS",
    "aggregate_anti_pattern_duals",
    "severity_weight_for",
]
