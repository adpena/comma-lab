# SPDX-License-Identifier: MIT
"""META-LIFT-2 Pareto polytope unified solver via Dykstra alternating projections.

Per operator WAVE-6 stagger directive + META-fractal-lift critique
2026-05-26 verbatim: *"Were those as fractal optimized as possible? We are
making progress but still too leaf and low level when we should be
exploiting patterns from math"*.

This module is the SISTER of META-LIFT-1
(:mod:`tac.cross_substrate_master_gradient_analyzer`) at the
Pareto-polytope-solver-consumer surface. Where META-LIFT-1 ranks
opportunities ACROSS substrates via the Cauchy-Schwarz upper bound,
META-LIFT-2 solves the UNIFIED bit-budget allocation problem:

  Given M substrates × 3 axes (seg, pose, rate) with per-axis
  Cauchy-Schwarz upper bounds from META-LIFT-1, find the canonical
  bit-budget allocation that minimizes aggregate ΔS subject to the
  cross-substrate feasibility polytope (the intersection of axis-wise
  byte-budget convex constraints + the Cauchy-Schwarz aggregate bound).

The canonical formulation is alternating projections (Boyd 2004 §7.2
*Convex Optimization*; Dykstra 1983 *An algorithm for restricted least
squares regression*) onto each constraint set:

  1. Per-substrate per-axis byte-budget box: 0 ≤ b_i_axis ≤ B_i_axis_max
  2. Aggregate Cauchy-Schwarz bound: Σ_i_axis (||∇S_i_axis||_2 · b_i_axis) ≤ M_aggregate
  3. Per-substrate aggregate byte-budget: Σ_axis b_i_axis ≤ B_i_aggregate_max
  4. Non-negativity: b_i_axis ≥ 0

Each iteration projects the current allocation onto each constraint set
sequentially with Dykstra's correction term (sister of Boyd POCS but
with provable convergence to the true intersection point when the
intersection is non-empty).

Mathematical contract (per canonical equation #344 family):

  Per-axis Taylor projection (consumed from META-LIFT-1):

    ΔS_i_axis = <∇S_i_axis, Δθ_i_axis> ≈ ||∇S_i_axis||_2 · b_i_axis

  Aggregate ΔS minimization (objective):

    minimize  Σ_i_axis ΔS_i_axis  (typically negative — score-lowering)
    subject to constraints (1)-(4) above

  Dykstra alternating projections per Boyd 2004 §7.2:

    x^{k+1}_C1 = Π_C1(x^k + p^k_C1)         # corrected projection onto C1
    p^{k+1}_C1 = x^k + p^k_C1 - x^{k+1}_C1   # update Dykstra correction
    ... repeat for C2, C3, ...
    x^{k+1} = x^{k+1}_Cn                     # final projection per round

  Convergence: Boyd 2004 §7.2 Theorem proves convergence to the
  Euclidean projection onto Π C_i for closed convex C_i with non-empty
  intersection. Rate is O(1/sqrt(k)) for general convex; O(1/k²) for
  strongly convex constraints (the box constraints + aggregate
  Cauchy-Schwarz are strongly convex).

All outputs are OBSERVABILITY-ONLY per Catalog #341 + CLAUDE.md
"Apples-to-apples evidence discipline":

  - ``axis_tag = "[predicted]"``
  - ``score_claim = False``
  - ``promotable = False``
  - ``evidence_grade = "[predicted; pareto-polytope-Dykstra-projections]"``

Promotion to a contest score signal REQUIRES paired-CUDA empirical
anchor.

Architecture (Catalog #230 sister-disjoint):

  - Inputs: :class:`tac.cross_substrate_master_gradient_analyzer.CrossSubstrateMasterGradientAnalysis`
    (from META-LIFT-1; READ-ONLY here)
  - Outputs: :class:`PareDLPSolution` persisted to fcntl-locked JSONL at
    ``.omx/state/pareto_polytope_solutions.jsonl``
  - Discipline: Catalog #131 / #138 / #245 (fcntl-locked + strict-load +
    canonical 4-layer ledger) + #287 / #323 (placeholder rejection +
    canonical Provenance) + #341 (routing markers) + #356 (per-axis)
    + #335 (cathedral consumer canonical contract) + #344
    (FORMALIZATION_PENDING canonical equation)

Per CLAUDE.md "MLX-first numpy-portable individually-fractal standing
directive": this module is pure-numpy (no MLX or PyTorch dependency at
analysis time). Inputs are numpy arrays; outputs are numpy + canonical
JSON.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" NON-NEGOTIABLE: this
solver is a CANONICAL Pareto primitive that the meta-Lagrangian solver
in :mod:`tac.findings_lagrangian` will consume when the Phase 3 typed-
atom-flow lands. Phase 1 wire-in is the cathedral consumer sister at
:mod:`tac.cathedral_consumers.pareto_polytope_unified_solver_consumer`.

The 6-hook wire-in declaration per Catalog #125:

  * Hook #1 SENSITIVITY_MAP — ACTIVE (per-substrate per-axis allocations
    feed downstream :mod:`tac.sensitivity_map` axis_weights)
  * Hook #2 PARETO_CONSTRAINT — ACTIVE PRIMARY (this IS the canonical
    Pareto polytope solver per CLAUDE.md "Meta-Lagrangian/Pareto solver"
    non-negotiable; Dim 1 Phase 4 binding implementation)
  * Hook #3 BIT_ALLOCATOR — ACTIVE PRIMARY (the
    :class:`UnifiedBitBudgetAllocation` IS the canonical bit-allocator
    contract per Dim 6 Step 6.5)
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE (sister consumer
    auto-discovered per Catalog #335/#336/#337)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (per-solution canonical
    posterior anchor via :func:`append_solution_locked`)
  * Hook #6 PROBE_DISAMBIGUATOR — ACTIVE (the per-axis allocation IS the
    canonical disambiguator between competing dispatch budget routes — a
    substrate with high seg leverage but low pose leverage receives a
    different allocation than the inverse per CLAUDE.md "SegNet vs
    PoseNet importance — operating-point dependent")
"""
from __future__ import annotations

from tac.pareto_polytope_unified_solver.solver import (
    CANONICAL_EQUATION_ID,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    PARETO_POLYTOPE_SOLUTIONS_LEDGER_PATH,
    PREDICTED_AXIS_TAG,
    SCHEMA_VERSION,
    VALID_AXIS_LABELS,
    PareDLPProblemSpec,
    PareDLPSolution,
    PareDLPSolutionCorruptError,
    UnifiedBitBudgetAllocation,
    append_solution_locked,
    build_problem_spec_from_meta_lift_1_analysis,
    load_solutions_strict,
    solve_pareto_polytope_via_dykstra_projections,
)

__all__ = [
    "CANONICAL_EQUATION_ID",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOLERANCE",
    "PARETO_POLYTOPE_SOLUTIONS_LEDGER_PATH",
    "PREDICTED_AXIS_TAG",
    "SCHEMA_VERSION",
    "VALID_AXIS_LABELS",
    "PareDLPProblemSpec",
    "PareDLPSolution",
    "PareDLPSolutionCorruptError",
    "UnifiedBitBudgetAllocation",
    "append_solution_locked",
    "build_problem_spec_from_meta_lift_1_analysis",
    "load_solutions_strict",
    "solve_pareto_polytope_via_dykstra_projections",
]
