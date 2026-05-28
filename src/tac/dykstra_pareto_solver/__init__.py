# SPDX-License-Identifier: MIT
"""tac.dykstra_pareto_solver — canonical Dykstra Pareto polytope solver facade.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 1 Phase 4 + CLAUDE.md "Meta-
Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" + Catalog
#372 STRICT preflight gate (canonical wire-in invocation in
``tools/cathedral_autopilot_autonomous_loop.py::main()`` per Catalog
#355 sister pattern).

This package is a **canonical facade** over the existing Phase 2 Dykstra
alternating-projections solver at :mod:`tac.findings_lagrangian.dual_solver_phase_2`
(landed 2026-05-26 per META-LAGRANGIAN-WIRE-1 Phase 2 advancement). The
canonical Boyd-Vandenberghe + Dykstra-1983 alternating-projections
mathematics, MLX-first + numpy-portable bridge, per-axis dual-variable
extraction, per-axis KKT residuals, and canonical Provenance-bearing
``PerAxisDualSolverResult`` are all imported from the canonical sister
module — duplicate Python implementation would violate CLAUDE.md
"UNIQUE-AND-COMPLETE-PER-METHOD" canonical-vs-unique decision per layer.

What this facade ADDS that the sister does not:

1. **Polytope dataclass** — typed (axis_bounds + halfspace_constraints) +
   ``project(point)`` closed-form projection (axis-aligned) + iterative
   projection (halfspace constraints). The sister's
   ``dykstra_alternating_projections_3_axis`` accepts a 3-tuple of
   ``(lower, upper)`` budgets directly; this facade accepts an explicit
   :class:`Polytope` so future per-paradigm extensions (e.g. simplex-
   constrained polytopes for VQ codebook indices, group-sparsity polytopes
   for Cool-Chic) can extend the contract without breaking the canonical
   3-axis API.
2. **ParetoSolverVerdict** — canonical Provenance-bearing verdict
   dataclass per Catalog #323/#341 with explicit feasibility verdict
   (``feasible: bool``), tight-axis identification (``tight_constraint_axes``
   = axes with λ > 0), and slack-axis identification
   (``slack_axes`` = axes with λ ≈ 0). Mirrors the canonical
   ``PerAxisDualSolverResult`` payload but presents it through the
   compounding-mechanism abstraction the CATHEDRAL-SMARTER-DESIGN-MEMO
   Dim 1 Phase 4 explicitly named.
3. **solve_pareto_polytope_intersection** — convenience wrapper that
   accepts a :class:`Polytope` + initial point, runs the canonical
   sister solver, builds the canonical ParetoSolverVerdict.

This package is a CANONICAL EXTENSION not a duplicate. Per the
operator-routable cathedral autopilot consumer
:mod:`tac.cathedral_consumers.dykstra_pareto_solver_consumer` + the
Catalog #372 STRICT preflight gate, this facade IS the canonical
compounding mechanism per the CATHEDRAL-SMARTER-DESIGN-MEMO.

Math (Boyd & Vandenberghe (2004) + Dykstra (1983))
--------------------------------------------------

Per-axis polytope intersection via alternating projections::

    x_{k+1} = π_{P_{(k mod m)}}(x_k + d_k)
    d_k corrections per Dykstra 1983 (residual feedback)

with per-axis Lagrangian dual ``L(x, λ) = f(x) + Σ_i λ_i · g_i(x)``
where ``g_i(x) ≤ 0`` is the i-th half-space constraint. At the saddle
point ``(x*, λ*)``, the per-axis tight-constraint identification is::

    tight = {i : λ_i > 0}   (binding axis — next-cycle attack direction)
    slack = {i : λ_i ≈ 0}   (interior axis — no further attack)

The sister Phase 2 solver computes ``λ_i`` from the converged Dykstra
correction vector (per Boyd-Dattorro 2006 § 6.2). This facade exposes
those values as a typed verdict for operator + cathedral autopilot
consumption.

Quick start::

    from tac.dykstra_pareto_solver import (
        Polytope,
        DykstraParetoSolver,
        solve_pareto_polytope_intersection,
    )
    from tac.cathedral.consumer_contract import AxisDecomposition

    # Build a 3-axis Pareto polytope around the canonical frontier.
    polytope = Polytope(
        axis_bounds={
            "seg": (0.0, 0.5),
            "pose": (0.0, 0.1),
            "rate": (-50_000.0, 0.0),  # canonical archive_bytes delta
        },
    )

    # Solve from a candidate's per-axis predicted deltas.
    verdict = solve_pareto_polytope_intersection(
        polytope,
        initial_point={"seg": 0.1, "pose": 0.05, "rate": -10_000.0},
        candidate_id="my_candidate_v1",
    )

    if verdict.feasible:
        print(f"feasible at {verdict.projection_point}")
        print(f"tight axes: {verdict.tight_constraint_axes}")
        print(f"slack axes: {verdict.slack_axes}")

Catalog #125 6-hook wire-in declaration
---------------------------------------

- Hook #1 sensitivity-map: ACTIVE — per-axis dual variables surface per-
  axis tight constraints downstream sensitivity-map consumers route through.
- Hook #2 Pareto constraint: ACTIVE PRIMARY — this work IS hook #2;
  Dykstra alternating projections IS the canonical Pareto polytope
  intersection mechanism.
- Hook #3 bit-allocator: ACTIVE — per-axis dual variables map to
  optimal bit allocation per axis (enables Wave N+2 Compound C
  heterogeneous bit allocation).
- Hook #4 cathedral autopilot dispatch: ACTIVE — auto-discovered via
  Catalog #335 + invoked in ``main()`` via Catalog #355 sister pattern
  + Catalog #372 STRICT gate enforces invoker callsite presence.
- Hook #5 continual-learning posterior: ACTIVE — solver verdicts append
  to canonical posterior; canonical equation
  ``dykstra_pareto_polytope_intersection_compounding_v1`` anchors
  accumulate; Catalog #371 auto-recalibrator refits.
- Hook #6 probe-disambiguator: ACTIVE — per-axis tight constraint IS
  the canonical disambiguator (which axis is binding determines the
  next-cycle's highest-EV attack direction).

Cross-references
----------------

- :mod:`tac.findings_lagrangian.dual_solver_phase_2` (canonical sister;
  this facade re-uses the underlying mathematics).
- :mod:`tac.score_composition` (canonical contest formula constants
  this facade's polytope expresses).
- :class:`tac.cathedral.consumer_contract.AxisDecomposition` (per-axis
  decomposition consumed at the Polytope boundary).
- :mod:`tac.provenance` (Catalog #323 canonical Provenance umbrella).
- :mod:`tac.canonical_frontier_pointer` (Catalog #343 canonical
  per-axis epsilon-bounds source).
- :mod:`tac.cathedral_consumers.dykstra_pareto_solver_consumer` (auto-
  discovered cathedral consumer per Catalog #335 canonical contract).
- ``tools/cathedral_autopilot_autonomous_loop.py::invoke_dykstra_pareto_solver_on_candidates``
  (Catalog #355 sister invoker; Catalog #372 STRICT gate enforces
  presence in ``main()``).
- Boyd & Vandenberghe (2004) Convex Optimization, Chapter 5 (Duality).
- Dykstra (1983) An Algorithm for Restricted Least-Squares Regression.
"""
from __future__ import annotations

from tac.dykstra_pareto_solver.anti_pattern_constraint import (
    ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX,
    AntiPatternConstraint,
    AntiPatternConstraintError,
    VALID_SEVERITY_WEIGHTS as ANTI_PATTERN_VALID_SEVERITY_WEIGHTS,
    aggregate_anti_pattern_duals,
    severity_weight_for as anti_pattern_severity_weight_for,
)
from tac.dykstra_pareto_solver.polytope import (
    Polytope,
    PolytopeError,
)
from tac.dykstra_pareto_solver.solver import (
    DykstraParetoSolver,
    solve_pareto_polytope_intersection,
)
from tac.dykstra_pareto_solver.verdict import (
    ParetoSolverVerdict,
    ParetoSolverError,
    TIGHT_CONSTRAINT_LAMBDA_THRESHOLD,
)

# Re-export the canonical sister-module primitives for downstream callers
# that want direct access to the Phase 2 dual-solver API.
from tac.findings_lagrangian.dual_solver_phase_2 import (
    AXIS_NAMES as CANONICAL_3_AXIS_NAMES,
    DYKSTRA_DEFAULT_EPSILON,
    DYKSTRA_DEFAULT_MAX_ITERATIONS,
    MLX_AVAILABLE,
    PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX,
    PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN,
    PerAxisDualSolverResult,
    Phase2SolverError,
    compute_per_axis_dual_variables,
    dykstra_alternating_projections_3_axis,
    kkt_residuals_per_axis,
    per_axis_adjustment_factors,
)


__all__ = [
    # Facade-level typed contracts.
    "Polytope",
    "PolytopeError",
    "DykstraParetoSolver",
    "ParetoSolverVerdict",
    "ParetoSolverError",
    "solve_pareto_polytope_intersection",
    "TIGHT_CONSTRAINT_LAMBDA_THRESHOLD",
    # Layer 5 Wave N+2 anti-pattern constraint integration.
    "AntiPatternConstraint",
    "AntiPatternConstraintError",
    "ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX",
    "ANTI_PATTERN_VALID_SEVERITY_WEIGHTS",
    "aggregate_anti_pattern_duals",
    "anti_pattern_severity_weight_for",
    # Canonical sister-module re-exports (for downstream direct use).
    "CANONICAL_3_AXIS_NAMES",
    "DYKSTRA_DEFAULT_EPSILON",
    "DYKSTRA_DEFAULT_MAX_ITERATIONS",
    "MLX_AVAILABLE",
    "PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX",
    "PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN",
    "PerAxisDualSolverResult",
    "Phase2SolverError",
    "compute_per_axis_dual_variables",
    "dykstra_alternating_projections_3_axis",
    "kkt_residuals_per_axis",
    "per_axis_adjustment_factors",
]
