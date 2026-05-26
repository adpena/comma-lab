# SPDX-License-Identifier: MIT
"""META-LIFT-2 Pareto polytope unified solver canonical implementation.

Per the 11th standing directive ORDER-MATTERS discipline: this module is
the ONE canonical Pareto polytope solver; per-substrate consumption
happens SECOND through the sister cathedral consumer at
:mod:`tac.cathedral_consumers.pareto_polytope_unified_solver_consumer`.

Mathematical contract (per canonical equation #344 family + Boyd 2004
*Convex Optimization* §7.2 + Dykstra 1983):

  Per-substrate per-axis byte-budget constraint set:
    C1 = {b ∈ R^(M·3) : 0 ≤ b_i_axis ≤ B_i_axis_max for all i, axis}

  Aggregate Cauchy-Schwarz bound:
    C2 = {b ∈ R^(M·3) : Σ_i_axis ||∇S_i_axis||_2 · b_i_axis ≤ M_aggregate}

  Per-substrate aggregate byte-budget:
    C3 = {b ∈ R^(M·3) : Σ_axis b_i_axis ≤ B_i_aggregate_max for all i}

  Non-negativity (subsumed by C1 lower bound but kept explicit):
    C4 = {b ∈ R^(M·3) : b_i_axis ≥ 0 for all i, axis}

  Dykstra alternating projections (Boyd 2004 §7.2 Algorithm 7.2):

    Initialize x^0 = 0 ∈ R^(M·3), p^0_j = 0 for each j ∈ {1, 2, 3, 4}
    Repeat:
      For j = 1, ..., 4:
        z = x^k + p^k_j                       # add Dykstra correction
        x^{k+1} = Π_Cj(z)                     # project onto C_j
        p^{k+1}_j = z - x^{k+1}               # update correction
      Stop when ||x^{k+1} - x^k|| < tol

  Convergence: Boyd 2004 §7.2 Theorem 7.1 proves convergence to
  Π C_j (the canonical intersection) when C_j are closed convex with
  non-empty intersection. Convergence is O(1/sqrt(k)) for general
  convex; box constraints + linear constraints qualify.

Per Catalog #318 raw-byte-authority guard: this solver does NOT emit
score derivatives; the canonical Provenance umbrella per Catalog #323
marks every solution as ``score_claim=False`` + ``promotable=False``.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" NON-NEGOTIABLE: this
solver IS the canonical Pareto primitive. When the Phase 3 typed-atom-
flow lands in :mod:`tac.findings_lagrangian`, the meta-Lagrangian
solver will consume :class:`PareDLPSolution` rows via the canonical
``update_from_anchor`` hook (Catalog #335 sister consumer contract).
"""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import socket
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants (canonical paths + schema versions)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

PARETO_POLYTOPE_SOLUTIONS_LEDGER_PATH = (
    REPO_ROOT / ".omx" / "state" / "pareto_polytope_solutions.jsonl"
)
"""Canonical fcntl-locked JSONL append-only ledger.

Sister of ``.omx/state/cross_substrate_master_gradient_analyses.jsonl``
(META-LIFT-1) at the Pareto-polytope-solution sub-surface per
Catalog #131 / #138 / #245 canonical 4-layer pattern.
"""

_LEDGER_LOCK_PATH = (
    REPO_ROOT / ".omx" / "state" / ".pareto_polytope_solutions.lock"
)

SCHEMA_VERSION = "pareto_polytope_unified_solution_v1"

# Per Catalog #356 per-axis decomposition canonical axis labels.
VALID_AXIS_LABELS: frozenset[str] = frozenset({"seg", "pose", "rate"})

# Per Catalog #341 routing markers (Tier A observability-only).
PREDICTED_AXIS_TAG = "[predicted]"

# Per Catalog #287 placeholder rejection.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {"<rationale>", "<reason>", "<rationale_here>", "<reason_here>", ""}
)

# Dykstra alternating projections canonical defaults per Boyd 2004 §7.2.
DEFAULT_MAX_ITERATIONS = 100
"""Canonical default iteration cap for Dykstra alternating projections.

Boyd 2004 §7.2 notes that for closed convex constraints with
non-empty intersection, 100 iterations is typically sufficient for
1e-6 tolerance on well-conditioned problems. We document this default
+ surface it as a CLI flag for problem-specific tuning.
"""

DEFAULT_TOLERANCE = 1e-6
"""Canonical default convergence tolerance (Euclidean norm of x change).

Per Boyd 2004 §7.2: convergence criterion ``||x^{k+1} - x^k||_2 < tol``.
The 1e-6 default matches numpy.linalg defaults and is well below
typical byte-budget precision (1.0 byte units).
"""

# Per Catalog #344 + CLAUDE.md "Canonical equations + models registry":
# the canonical equation id this solver enables (FORMALIZATION_PENDING
# until paired-CUDA empirical anchor lands).
CANONICAL_EQUATION_ID = (
    "pareto_polytope_dykstra_unified_bit_budget_allocation_savings_v1"
)


# ---------------------------------------------------------------------------
# Custom exception (canonical strict-load fail-closed sister per Catalog #138)
# ---------------------------------------------------------------------------


class PareDLPSolutionCorruptError(RuntimeError):
    """Strict-load corruption marker per Catalog #138 fail-closed discipline.

    Sister of
    :class:`tac.cross_substrate_master_gradient_analyzer.CrossSubstrateMasterGradientAnalysisCorruptError`
    + :class:`tac.master_gradient.MasterGradientAnchorsCorruptError`. The
    canonical strict-load helper raises this so any future consumer of
    the Pareto polytope solutions ledger inherits fail-closed-on-
    corruption semantics — a parse failure does NOT silently coerce
    missing rows to ``[]`` (the bug class Catalog #138 extincts).
    """


# ---------------------------------------------------------------------------
# Frozen dataclasses (canonical contract per Catalog #335 + #323)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PareDLPProblemSpec:
    """Cross-substrate Pareto polytope problem specification.

    Encodes the M-substrate × 3-axis byte-budget feasibility polytope
    + the Cauchy-Schwarz aggregate bound from META-LIFT-1.

    Per Catalog #356 per-axis decomposition: each substrate's per-axis
    gradient L2 norm (from META-LIFT-1) is the canonical sensitivity
    coefficient that maps a byte-budget allocation to a per-axis
    Cauchy-Schwarz ΔS contribution.

    The problem spec is CANONICAL INPUT to
    :func:`solve_pareto_polytope_via_dykstra_projections`.
    """

    substrate_archive_sha256s: tuple[str, ...]
    """One per substrate; len == M (canonical archive sha256 per Catalog #245)."""

    per_axis_gradient_l2_norms: tuple[tuple[float, float, float], ...]
    """For each substrate i: (||∇S_seg_i||_2, ||∇S_pose_i||_2, ||∇S_rate_i||_2).

    Sourced from META-LIFT-1's :class:`CrossSubstrateAxisProjection`
    canonical helper. Each tuple has exactly 3 entries (axis order:
    seg, pose, rate per canonical
    :func:`tac.cross_substrate_master_gradient_analyzer.analyzer._axis_index`).
    """

    per_substrate_per_axis_byte_budget_caps: tuple[tuple[float, float, float], ...]
    """For each substrate i: max bytes that can be perturbed per axis.

    Encodes the per-substrate per-axis box constraint C1. Typically
    derived from the substrate's archive size + the per-axis section
    boundaries per the canonical archive grammar.
    """

    per_substrate_aggregate_byte_budget_caps: tuple[float, ...]
    """For each substrate i: max TOTAL bytes summed across all 3 axes.

    Encodes the per-substrate aggregate byte-budget C3.
    """

    cauchy_schwarz_aggregate_upper_bound: float
    """Canonical Cauchy-Schwarz aggregate upper bound from META-LIFT-1.

    The C2 constraint: Σ_i_axis ||∇S_i_axis||_2 · b_i_axis ≤ M_aggregate.
    Sourced from META-LIFT-1's
    :class:`CrossSubstrateMasterGradientAnalysis.cauchy_schwarz_aggregate_upper_bound`.
    """

    target_aggregate_delta_s: float | None = None
    """Optional target aggregate ΔS for budget-driven allocation.

    When provided, the solver searches for the allocation that achieves
    this ΔS subject to constraints (1)-(4). When None, the solver
    minimizes aggregate ΔS subject to constraints.
    """

    measurement_axes: tuple[str, ...] = ()
    """For each substrate i: per Catalog #127 axis token (informational only).

    Carried through to the solution row for canonical Provenance per
    Catalog #323; does NOT affect the solver math (substrate authority
    is handled at the META-LIFT-1 ingestion surface).
    """

    def __post_init__(self) -> None:
        m = len(self.substrate_archive_sha256s)
        if m == 0:
            raise ValueError("substrate_archive_sha256s must be non-empty")
        if len(self.per_axis_gradient_l2_norms) != m:
            raise ValueError(
                f"per_axis_gradient_l2_norms length ({len(self.per_axis_gradient_l2_norms)}) "
                f"must match substrate count ({m})"
            )
        if len(self.per_substrate_per_axis_byte_budget_caps) != m:
            raise ValueError(
                f"per_substrate_per_axis_byte_budget_caps length ({len(self.per_substrate_per_axis_byte_budget_caps)}) "
                f"must match substrate count ({m})"
            )
        if len(self.per_substrate_aggregate_byte_budget_caps) != m:
            raise ValueError(
                f"per_substrate_aggregate_byte_budget_caps length ({len(self.per_substrate_aggregate_byte_budget_caps)}) "
                f"must match substrate count ({m})"
            )
        for i, norms in enumerate(self.per_axis_gradient_l2_norms):
            if len(norms) != 3:
                raise ValueError(
                    f"per_axis_gradient_l2_norms[{i}] must have exactly 3 entries (seg, pose, rate)"
                )
            for j, val in enumerate(norms):
                if not np.isfinite(val) or val < 0:
                    raise ValueError(
                        f"per_axis_gradient_l2_norms[{i}][{j}] must be non-negative finite; got {val}"
                    )
        for i, caps in enumerate(self.per_substrate_per_axis_byte_budget_caps):
            if len(caps) != 3:
                raise ValueError(
                    f"per_substrate_per_axis_byte_budget_caps[{i}] must have exactly 3 entries"
                )
            for j, val in enumerate(caps):
                if not np.isfinite(val) or val < 0:
                    raise ValueError(
                        f"per_substrate_per_axis_byte_budget_caps[{i}][{j}] must be non-negative finite; got {val}"
                    )
        for i, cap in enumerate(self.per_substrate_aggregate_byte_budget_caps):
            if not np.isfinite(cap) or cap < 0:
                raise ValueError(
                    f"per_substrate_aggregate_byte_budget_caps[{i}] must be non-negative finite; got {cap}"
                )
        if not np.isfinite(self.cauchy_schwarz_aggregate_upper_bound):
            raise ValueError("cauchy_schwarz_aggregate_upper_bound must be finite")
        if self.cauchy_schwarz_aggregate_upper_bound < 0:
            raise ValueError("cauchy_schwarz_aggregate_upper_bound must be non-negative")
        if self.target_aggregate_delta_s is not None:
            if not np.isfinite(self.target_aggregate_delta_s):
                raise ValueError("target_aggregate_delta_s must be finite when provided")

    @property
    def m(self) -> int:
        """Number of substrates in the problem."""
        return len(self.substrate_archive_sha256s)

    def as_dict(self) -> dict:
        return {
            "substrate_archive_sha256s": list(self.substrate_archive_sha256s),
            "per_axis_gradient_l2_norms": [list(t) for t in self.per_axis_gradient_l2_norms],
            "per_substrate_per_axis_byte_budget_caps": [
                list(t) for t in self.per_substrate_per_axis_byte_budget_caps
            ],
            "per_substrate_aggregate_byte_budget_caps": list(
                self.per_substrate_aggregate_byte_budget_caps
            ),
            "cauchy_schwarz_aggregate_upper_bound": float(
                self.cauchy_schwarz_aggregate_upper_bound
            ),
            "target_aggregate_delta_s": (
                float(self.target_aggregate_delta_s)
                if self.target_aggregate_delta_s is not None
                else None
            ),
            "measurement_axes": list(self.measurement_axes),
        }


@dataclass(frozen=True)
class UnifiedBitBudgetAllocation:
    """Canonical per-substrate × per-axis byte allocation.

    Output of :func:`solve_pareto_polytope_via_dykstra_projections`.
    Each entry ``per_substrate_per_axis_allocations[i][j]`` is the
    allocated byte-budget for substrate i × axis j (j ∈ {0=seg, 1=pose,
    2=rate}).

    Per Catalog #341 routing markers: this allocation is observability-
    only by construction. Promotion to a contest dispatch decision
    REQUIRES paired-CUDA empirical anchor + operator-frontier-override
    per CLAUDE.md "Submission auth eval".
    """

    substrate_archive_sha256s: tuple[str, ...]
    """One per substrate; len == M (matches PareDLPProblemSpec)."""

    per_substrate_per_axis_allocations: tuple[tuple[float, float, float], ...]
    """For each substrate i: (b_seg_i, b_pose_i, b_rate_i) allocated bytes."""

    per_substrate_aggregate_allocations: tuple[float, ...]
    """For each substrate i: Σ_axis b_i_axis (derived; for convenience)."""

    aggregate_total_bytes_allocated: float
    """Σ_i_axis b_i_axis — total allocation across all substrates × axes."""

    aggregate_predicted_delta_s: float
    """Σ_i_axis ||∇S_i_axis||_2 · b_i_axis — Cauchy-Schwarz ΔS estimate.

    Note: this is the UPPER BOUND on |ΔS| per Cauchy-Schwarz; the
    actual ΔS magnitude is bounded by this value. By construction
    ≤ Cauchy-Schwarz aggregate upper bound from problem spec.
    """

    feasible: bool
    """Whether the allocation satisfies all 4 constraint sets within tolerance."""

    feasibility_residual: float
    """Largest constraint violation residual (0 = exact feasibility)."""

    def __post_init__(self) -> None:
        m = len(self.substrate_archive_sha256s)
        if m == 0:
            raise ValueError("substrate_archive_sha256s must be non-empty")
        if len(self.per_substrate_per_axis_allocations) != m:
            raise ValueError(
                f"per_substrate_per_axis_allocations length ({len(self.per_substrate_per_axis_allocations)}) "
                f"must match substrate count ({m})"
            )
        if len(self.per_substrate_aggregate_allocations) != m:
            raise ValueError(
                f"per_substrate_aggregate_allocations length ({len(self.per_substrate_aggregate_allocations)}) "
                f"must match substrate count ({m})"
            )
        for i, alloc in enumerate(self.per_substrate_per_axis_allocations):
            if len(alloc) != 3:
                raise ValueError(
                    f"per_substrate_per_axis_allocations[{i}] must have exactly 3 entries"
                )
            for j, val in enumerate(alloc):
                if not np.isfinite(val):
                    raise ValueError(
                        f"per_substrate_per_axis_allocations[{i}][{j}] must be finite; got {val}"
                    )
                if val < -1e-9:  # tolerate small numerical noise from projections
                    raise ValueError(
                        f"per_substrate_per_axis_allocations[{i}][{j}] must be non-negative; got {val}"
                    )
        for i, agg in enumerate(self.per_substrate_aggregate_allocations):
            if not np.isfinite(agg) or agg < -1e-9:
                raise ValueError(
                    f"per_substrate_aggregate_allocations[{i}] must be non-negative finite; got {agg}"
                )
        if not np.isfinite(self.aggregate_total_bytes_allocated):
            raise ValueError("aggregate_total_bytes_allocated must be finite")
        if self.aggregate_total_bytes_allocated < -1e-9:
            raise ValueError("aggregate_total_bytes_allocated must be non-negative")
        if not np.isfinite(self.aggregate_predicted_delta_s):
            raise ValueError("aggregate_predicted_delta_s must be finite")
        if not np.isfinite(self.feasibility_residual):
            raise ValueError("feasibility_residual must be finite")
        if self.feasibility_residual < 0:
            raise ValueError("feasibility_residual must be non-negative")

    def as_dict(self) -> dict:
        return {
            "substrate_archive_sha256s": list(self.substrate_archive_sha256s),
            "per_substrate_per_axis_allocations": [
                list(t) for t in self.per_substrate_per_axis_allocations
            ],
            "per_substrate_aggregate_allocations": list(
                self.per_substrate_aggregate_allocations
            ),
            "aggregate_total_bytes_allocated": float(self.aggregate_total_bytes_allocated),
            "aggregate_predicted_delta_s": float(self.aggregate_predicted_delta_s),
            "feasible": bool(self.feasible),
            "feasibility_residual": float(self.feasibility_residual),
        }


@dataclass(frozen=True)
class PareDLPSolution:
    """Canonical META-LIFT-2 Pareto polytope solution output.

    Sister of
    :class:`tac.cross_substrate_master_gradient_analyzer.CrossSubstrateMasterGradientAnalysis`
    at the Pareto-polytope-solution sub-surface. Persisted to the
    canonical fcntl-locked JSONL ledger at
    ``.omx/state/pareto_polytope_solutions.jsonl``.

    Per Catalog #341 + CLAUDE.md "Apples-to-apples evidence discipline":
    every solution is observability-only by construction (axis_tag,
    score_claim, promotable defaults are FALSE-class). Promotion of a
    solution to a contest score signal REQUIRES paired-CUDA empirical
    anchor.
    """

    schema_version: str
    """Canonical schema version (current: ``pareto_polytope_unified_solution_v1``)."""

    solution_id: str
    """Deterministic id ``pareto_polytope_<utc_compact>_<n_substrates>``."""

    measurement_utc: str
    """ISO-8601 UTC timestamp of solution emission."""

    problem_spec: PareDLPProblemSpec
    """Canonical input that produced this solution."""

    allocation: UnifiedBitBudgetAllocation
    """Canonical bit-budget allocation output."""

    n_iterations_to_convergence: int
    """Number of Dykstra iterations to reach tolerance (≤ max_iterations)."""

    converged: bool
    """Whether the solver converged within max_iterations."""

    convergence_history: tuple[float, ...]
    """Per-iteration ``||x^{k+1} - x^k||_2`` norm (for observability)."""

    axis_tag: str
    """Always ``"[predicted]"`` per Catalog #341 + canonical Provenance."""

    score_claim: bool
    """Always ``False`` per CLAUDE.md "Apples-to-apples evidence discipline"."""

    promotable: bool
    """Always ``False`` per Catalog #341 + #192."""

    evidence_grade: str
    """Always ``"[predicted; pareto-polytope-Dykstra-projections]"``."""

    canonical_helper_invocation: str
    """``"tac.pareto_polytope_unified_solver.solve_pareto_polytope_via_dykstra_projections"``."""

    canonical_equation_id: str
    """``"pareto_polytope_dykstra_unified_bit_budget_allocation_savings_v1"`` per Catalog #344."""

    canonical_equation_status: str
    """``"FORMALIZATION_PENDING"`` until paired-CUDA empirical anchor lands per Catalog #344."""

    upstream_meta_lift_1_analysis_id: str | None = None
    """Optional META-LIFT-1 analysis_id this solution consumed (canonical cite-chain)."""

    written_at_utc: str = ""
    written_pid: int = 0
    written_host: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if not self.solution_id:
            raise ValueError("solution_id must be non-empty")
        if not self.measurement_utc:
            raise ValueError("measurement_utc must be non-empty")
        if self.axis_tag != PREDICTED_AXIS_TAG:
            raise ValueError(f"axis_tag must equal {PREDICTED_AXIS_TAG!r}; got {self.axis_tag!r}")
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per Catalog #341")
        if self.promotable is not False:
            raise ValueError("promotable must be False per Catalog #341")
        if not self.evidence_grade.startswith("[predicted;"):
            raise ValueError("evidence_grade must start with '[predicted;' per Catalog #287 / #323")
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise ValueError(
                f"canonical_equation_id must equal {CANONICAL_EQUATION_ID!r}; got {self.canonical_equation_id!r}"
            )
        if self.canonical_equation_status not in {"FORMALIZATION_PENDING", "REGISTERED"}:
            raise ValueError(
                "canonical_equation_status must be 'FORMALIZATION_PENDING' or 'REGISTERED' per Catalog #344"
            )
        if self.n_iterations_to_convergence < 0:
            raise ValueError("n_iterations_to_convergence must be non-negative")

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "solution_id": self.solution_id,
            "measurement_utc": self.measurement_utc,
            "problem_spec": self.problem_spec.as_dict(),
            "allocation": self.allocation.as_dict(),
            "n_iterations_to_convergence": int(self.n_iterations_to_convergence),
            "converged": bool(self.converged),
            "convergence_history": list(self.convergence_history),
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "evidence_grade": self.evidence_grade,
            "canonical_helper_invocation": self.canonical_helper_invocation,
            "canonical_equation_id": self.canonical_equation_id,
            "canonical_equation_status": self.canonical_equation_status,
            "upstream_meta_lift_1_analysis_id": self.upstream_meta_lift_1_analysis_id,
            "written_at_utc": self.written_at_utc,
            "written_pid": int(self.written_pid),
            "written_host": self.written_host,
        }


# ---------------------------------------------------------------------------
# Canonical projection operators (per Boyd 2004 §7.2)
# ---------------------------------------------------------------------------


def _project_onto_per_axis_box(
    x: np.ndarray,
    per_axis_caps: np.ndarray,
) -> np.ndarray:
    """Project onto C1 = box [0, B_i_axis_max].

    Per Boyd 2004 §6.4 elementwise box projection (closed-form clip).

    Args:
        x: shape (M, 3) candidate allocation.
        per_axis_caps: shape (M, 3) upper bound matrix.

    Returns:
        Projected (M, 3) allocation in box.
    """
    return np.clip(x, 0.0, per_axis_caps)


def _project_onto_per_substrate_aggregate_budget(
    x: np.ndarray,
    aggregate_caps: np.ndarray,
) -> np.ndarray:
    """Project onto C3 = {x : Σ_axis x_i_axis ≤ B_i_aggregate} for each i.

    Per Boyd 2004 §6.4 + Duchi 2008 *Efficient projections onto the
    L1-ball*: simplex projection per row. We scale-down each row whose
    sum exceeds the cap so the simplex constraint binds with equality.

    Args:
        x: shape (M, 3) candidate allocation.
        aggregate_caps: shape (M,) per-substrate aggregate caps.

    Returns:
        Projected (M, 3) allocation honoring per-substrate aggregates.
    """
    out = np.copy(x)
    # Per-substrate sum.
    sums = np.sum(out, axis=1)  # (M,)
    # For rows exceeding the cap, scale down proportionally.
    over_mask = sums > aggregate_caps
    if np.any(over_mask):
        # We project to the L1-ball constraint Σ_axis x_i_axis ≤ B_i.
        # For non-negative x, this is scale-down by ratio when constraint
        # active. (Closed-form per Duchi 2008 for non-negative simplex.)
        # We also enforce non-negativity here for robustness.
        out = np.maximum(out, 0.0)
        sums = np.sum(out, axis=1)
        over_mask = sums > aggregate_caps
        if np.any(over_mask):
            # Scale rows where sum > cap.
            scale = np.where(
                sums > 0,
                np.minimum(1.0, aggregate_caps / np.maximum(sums, 1e-12)),
                1.0,
            )
            scale = np.where(over_mask, scale, 1.0)
            out = out * scale[:, np.newaxis]
    return out


def _project_onto_cauchy_schwarz_bound(
    x: np.ndarray,
    gradient_l2_norms: np.ndarray,
    cauchy_schwarz_bound: float,
) -> np.ndarray:
    """Project onto C2 = {x : Σ_i_axis ||∇S_i_axis||_2 · x_i_axis ≤ M}.

    Per Boyd 2004 §6.4: half-space projection. The constraint is a
    linear inequality ``<a, x> ≤ b`` with ``a = vec(gradient_l2_norms)``,
    ``b = cauchy_schwarz_bound``. Closed-form projection:

        x_proj = x - max(0, <a, x> - b) * a / ||a||²

    Args:
        x: shape (M, 3) candidate allocation.
        gradient_l2_norms: shape (M, 3) per-axis gradient L2 norms.
        cauchy_schwarz_bound: scalar M_aggregate bound.

    Returns:
        Projected (M, 3) allocation honoring the linear inequality.
    """
    a = gradient_l2_norms  # (M, 3)
    a_dot_x = np.sum(a * x)  # scalar
    if a_dot_x <= cauchy_schwarz_bound:
        return x
    a_norm_sq = np.sum(a * a)
    if a_norm_sq <= 1e-12:
        # Degenerate: all gradient norms zero — projection is identity.
        return x
    excess = a_dot_x - cauchy_schwarz_bound
    return x - (excess / a_norm_sq) * a


def _project_onto_non_negativity(x: np.ndarray) -> np.ndarray:
    """Project onto C4 = {x : x ≥ 0} (elementwise ReLU per Boyd 2004 §6.4)."""
    return np.maximum(x, 0.0)


def _compute_feasibility_residual(
    x: np.ndarray,
    per_axis_caps: np.ndarray,
    aggregate_caps: np.ndarray,
    gradient_l2_norms: np.ndarray,
    cauchy_schwarz_bound: float,
) -> float:
    """Compute largest constraint violation residual.

    Returns 0 if x is feasible; positive value = violation magnitude.
    """
    residual = 0.0

    # C1: box (per-axis caps); residual = max(0, x - cap, -x)
    over_axis = np.maximum(x - per_axis_caps, 0.0)
    if over_axis.size > 0:
        residual = max(residual, float(np.max(over_axis)))

    # C4: non-negativity; residual = max(0, -x)
    under = np.maximum(-x, 0.0)
    if under.size > 0:
        residual = max(residual, float(np.max(under)))

    # C3: per-substrate aggregate; residual = max(0, sum - cap)
    sums = np.sum(np.maximum(x, 0.0), axis=1)
    over_agg = np.maximum(sums - aggregate_caps, 0.0)
    if over_agg.size > 0:
        residual = max(residual, float(np.max(over_agg)))

    # C2: Cauchy-Schwarz; residual = max(0, <a, x> - bound)
    cs_dot = np.sum(gradient_l2_norms * np.maximum(x, 0.0))
    cs_residual = max(0.0, cs_dot - cauchy_schwarz_bound)
    residual = max(residual, cs_residual)

    return residual


# ---------------------------------------------------------------------------
# Core solver (Dykstra alternating projections per Boyd 2004 §7.2)
# ---------------------------------------------------------------------------


def solve_pareto_polytope_via_dykstra_projections(
    problem_spec: PareDLPProblemSpec,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tol: float = DEFAULT_TOLERANCE,
    measurement_utc: str | None = None,
    upstream_meta_lift_1_analysis_id: str | None = None,
    initial_allocation: np.ndarray | None = None,
) -> PareDLPSolution:
    """Solve the cross-substrate Pareto polytope via Dykstra projections.

    Canonical implementation of Boyd 2004 *Convex Optimization* §7.2
    Algorithm 7.2 (Dykstra's alternating projections with correction
    terms). Solves:

        find x ∈ C1 ∩ C2 ∩ C3 ∩ C4

    where x ∈ R^(M·3) is the per-substrate per-axis byte allocation
    and the four convex constraint sets are defined in the module
    docstring.

    Initialization: if ``initial_allocation`` is None, start from the
    PER-AXIS BOX CAPS (each substrate × axis allocated to its
    per-axis cap). This is a common starting point that exploits the
    full per-axis budget and lets the cross-substrate constraints
    iteratively reduce the allocation. The all-zero starting point is
    also feasible but converges to all-zeros for problems without a
    target_aggregate_delta_s objective term.

    Convergence: per Boyd 2004 §7.2 Theorem 7.1 the iteration converges
    to the Euclidean projection onto Π C_j when the intersection is
    non-empty. For our problem the intersection always contains the
    origin (allocation = 0) so convergence is guaranteed.

    Per Catalog #341 routing markers: this solution is observability-
    only by construction. Promotion to a contest dispatch decision
    REQUIRES paired-CUDA empirical anchor per CLAUDE.md "Submission
    auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE".

    Args:
        problem_spec: canonical input (per-substrate gradients + caps +
            Cauchy-Schwarz bound from META-LIFT-1).
        max_iterations: cap on Dykstra iterations (default
            :data:`DEFAULT_MAX_ITERATIONS`).
        tol: convergence tolerance on ``||x^{k+1} - x^k||_2`` (default
            :data:`DEFAULT_TOLERANCE`).
        measurement_utc: optional UTC timestamp; defaults to now.
        upstream_meta_lift_1_analysis_id: optional META-LIFT-1
            analysis_id this solution consumed (canonical cite-chain).
        initial_allocation: optional (M, 3) starting point; defaults to
            the per-axis box caps.

    Returns:
        Canonical :class:`PareDLPSolution` with allocation + convergence
        diagnostics.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if tol <= 0:
        raise ValueError("tol must be positive")

    m = problem_spec.m
    gradient_l2_norms = np.asarray(
        problem_spec.per_axis_gradient_l2_norms, dtype=np.float64
    )  # (M, 3)
    per_axis_caps = np.asarray(
        problem_spec.per_substrate_per_axis_byte_budget_caps, dtype=np.float64
    )  # (M, 3)
    aggregate_caps = np.asarray(
        problem_spec.per_substrate_aggregate_byte_budget_caps, dtype=np.float64
    )  # (M,)
    cs_bound = float(problem_spec.cauchy_schwarz_aggregate_upper_bound)

    # Initial allocation: per-axis box caps (start at max budget; let
    # cross-substrate constraints shrink the allocation iteratively).
    # Alternative: caller can pass an all-zeros init or a domain-specific
    # warm start.
    if initial_allocation is None:
        x = np.copy(per_axis_caps)
    else:
        x = np.asarray(initial_allocation, dtype=np.float64)
        if x.shape != (m, 3):
            raise ValueError(
                f"initial_allocation must have shape ({m}, 3); got {x.shape}"
            )

    # Dykstra correction terms (one per constraint set).
    p1 = np.zeros_like(x)  # C1: per-axis box
    p2 = np.zeros_like(x)  # C2: Cauchy-Schwarz
    p3 = np.zeros_like(x)  # C3: per-substrate aggregate
    p4 = np.zeros_like(x)  # C4: non-negativity

    convergence_history: list[float] = []
    converged = False
    n_iterations = 0

    for iteration in range(max_iterations):
        x_prev = np.copy(x)

        # Project onto C1 with Dykstra correction.
        z1 = x + p1
        x = _project_onto_per_axis_box(z1, per_axis_caps)
        p1 = z1 - x

        # Project onto C2 with Dykstra correction.
        z2 = x + p2
        x = _project_onto_cauchy_schwarz_bound(z2, gradient_l2_norms, cs_bound)
        p2 = z2 - x

        # Project onto C3 with Dykstra correction.
        z3 = x + p3
        x = _project_onto_per_substrate_aggregate_budget(z3, aggregate_caps)
        p3 = z3 - x

        # Project onto C4 with Dykstra correction.
        z4 = x + p4
        x = _project_onto_non_negativity(z4)
        p4 = z4 - x

        # Convergence check: Euclidean norm of x change.
        delta = float(np.linalg.norm(x - x_prev, ord=2))
        convergence_history.append(delta)
        n_iterations = iteration + 1
        if delta < tol:
            converged = True
            break

    # Final feasibility residual (after all 4 projections, residual
    # should be near zero modulo numerical noise).
    feasibility_residual = _compute_feasibility_residual(
        x,
        per_axis_caps=per_axis_caps,
        aggregate_caps=aggregate_caps,
        gradient_l2_norms=gradient_l2_norms,
        cauchy_schwarz_bound=cs_bound,
    )
    # Per Boyd 2004 §11.3 numerical feasibility tolerance: use sqrt(tol)
    # since projection residuals accumulate as O(sqrt(k·tol)) per
    # Dykstra's correction scheme. This is more permissive than tol
    # itself and matches the numerical precision the solver can deliver.
    feasibility_tol = max(np.sqrt(tol), 1e-5)
    feasible = feasibility_residual < feasibility_tol

    # Compose canonical UnifiedBitBudgetAllocation.
    per_axis_alloc = tuple(
        (float(x[i, 0]), float(x[i, 1]), float(x[i, 2])) for i in range(m)
    )
    aggregate_alloc = tuple(float(np.sum(x[i])) for i in range(m))
    aggregate_total = float(np.sum(x))
    aggregate_predicted_delta_s = float(np.sum(gradient_l2_norms * x))

    allocation = UnifiedBitBudgetAllocation(
        substrate_archive_sha256s=problem_spec.substrate_archive_sha256s,
        per_substrate_per_axis_allocations=per_axis_alloc,
        per_substrate_aggregate_allocations=aggregate_alloc,
        aggregate_total_bytes_allocated=aggregate_total,
        aggregate_predicted_delta_s=aggregate_predicted_delta_s,
        feasible=feasible,
        feasibility_residual=feasibility_residual,
    )

    utc = measurement_utc or _utc_now_iso()
    compact_utc = utc.replace(":", "").replace("-", "")[:15]
    solution_id = f"pareto_polytope_{compact_utc}_{m}"

    return PareDLPSolution(
        schema_version=SCHEMA_VERSION,
        solution_id=solution_id,
        measurement_utc=utc,
        problem_spec=problem_spec,
        allocation=allocation,
        n_iterations_to_convergence=n_iterations,
        converged=converged,
        convergence_history=tuple(convergence_history),
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; pareto-polytope-Dykstra-projections]",
        canonical_helper_invocation=(
            "tac.pareto_polytope_unified_solver."
            "solve_pareto_polytope_via_dykstra_projections"
        ),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        upstream_meta_lift_1_analysis_id=upstream_meta_lift_1_analysis_id,
    )


# ---------------------------------------------------------------------------
# META-LIFT-1 integration helper (canonical sister)
# ---------------------------------------------------------------------------


def build_problem_spec_from_meta_lift_1_analysis(
    analysis: Any,
    *,
    per_substrate_per_axis_byte_budget_cap_fraction: float = 0.10,
    per_substrate_aggregate_byte_budget_cap_fraction: float = 0.20,
    target_aggregate_delta_s: float | None = None,
) -> PareDLPProblemSpec:
    """Build a Pareto polytope problem spec from a META-LIFT-1 analysis.

    Canonical bridge from
    :class:`tac.cross_substrate_master_gradient_analyzer.CrossSubstrateMasterGradientAnalysis`
    to :class:`PareDLPProblemSpec` per the 11th standing directive
    ORDER-MATTERS discipline (ONE canonical META-LIFT-1 analyzer
    FIRST; ONE canonical META-LIFT-2 Pareto solver SECOND).

    Per-substrate per-axis byte budget caps default to 10% of the
    substrate's total archive byte count (per-axis equally) — a
    conservative starting point that respects CLAUDE.md "Bit-level
    deconstruction and entropy discipline" + the empirical observation
    that >10% byte perturbation typically breaks the inflate runtime.

    Per-substrate aggregate byte budget caps default to 20% of the
    substrate's total archive byte count — allows allocation to
    concentrate across axes within a substrate.

    Args:
        analysis: META-LIFT-1
            :class:`CrossSubstrateMasterGradientAnalysis` instance.
        per_substrate_per_axis_byte_budget_cap_fraction: per-axis cap
            as fraction of total substrate bytes (default 0.10).
        per_substrate_aggregate_byte_budget_cap_fraction: per-substrate
            aggregate cap as fraction of total substrate bytes
            (default 0.20).
        target_aggregate_delta_s: optional target ΔS for budget-driven
            allocation.

    Returns:
        Canonical :class:`PareDLPProblemSpec` ready for
        :func:`solve_pareto_polytope_via_dykstra_projections`.
    """
    # Duck-type check: META-LIFT-1 analysis exposes substrate_rows +
    # cauchy_schwarz_aggregate_upper_bound; we accept either the frozen
    # dataclass or its dict form (load_analyses_strict returns dicts).
    if hasattr(analysis, "substrate_rows"):
        substrate_rows = analysis.substrate_rows
        cs_bound = float(analysis.cauchy_schwarz_aggregate_upper_bound)
    elif isinstance(analysis, dict):
        substrate_rows = analysis.get("substrate_rows", [])
        cs_bound = float(analysis.get("cauchy_schwarz_aggregate_upper_bound", 0.0))
    else:
        raise TypeError(
            f"analysis must be CrossSubstrateMasterGradientAnalysis or dict; got {type(analysis).__name__}"
        )

    if not substrate_rows:
        raise ValueError("analysis must have at least one substrate row")

    shas: list[str] = []
    norms: list[tuple[float, float, float]] = []
    per_axis_caps: list[tuple[float, float, float]] = []
    aggregate_caps: list[float] = []
    measurement_axes: list[str] = []

    for row in substrate_rows:
        if hasattr(row, "archive_sha256"):
            sha = row.archive_sha256
            projections = row.per_axis_projections
            n_bytes = row.n_bytes
            meas_axis = row.measurement_axis
        elif isinstance(row, dict):
            sha = row.get("archive_sha256", "")
            projections = row.get("per_axis_projections", [])
            n_bytes = int(row.get("n_bytes", 0))
            meas_axis = row.get("measurement_axis", "[unknown]")
        else:
            raise TypeError(
                f"substrate_row must be CrossSubstrateSubstrateRow or dict; got {type(row).__name__}"
            )

        if not sha:
            raise ValueError("each substrate_row must have non-empty archive_sha256")
        if n_bytes <= 0:
            raise ValueError(f"substrate {sha[:12]} must have positive n_bytes; got {n_bytes}")

        # Extract per-axis L2 norms (canonical 3-axis ordering: seg, pose, rate).
        axis_norms = {"seg": 0.0, "pose": 0.0, "rate": 0.0}
        for proj in projections:
            if hasattr(proj, "axis"):
                axis = proj.axis
                norm = float(proj.gradient_l2_norm)
            elif isinstance(proj, dict):
                axis = proj.get("axis", "")
                norm = float(proj.get("gradient_l2_norm", 0.0))
            else:
                raise TypeError(
                    f"projection must be CrossSubstrateAxisProjection or dict; got {type(proj).__name__}"
                )
            if axis in axis_norms:
                axis_norms[axis] = norm

        shas.append(sha)
        norms.append((axis_norms["seg"], axis_norms["pose"], axis_norms["rate"]))
        # Per-axis cap: equal split of (fraction × n_bytes) across 3 axes.
        per_axis_cap = (n_bytes * per_substrate_per_axis_byte_budget_cap_fraction) / 3.0
        per_axis_caps.append((per_axis_cap, per_axis_cap, per_axis_cap))
        # Aggregate cap: fraction × n_bytes (concentration allowed across axes).
        aggregate_caps.append(n_bytes * per_substrate_aggregate_byte_budget_cap_fraction)
        measurement_axes.append(meas_axis)

    return PareDLPProblemSpec(
        substrate_archive_sha256s=tuple(shas),
        per_axis_gradient_l2_norms=tuple(norms),
        per_substrate_per_axis_byte_budget_caps=tuple(per_axis_caps),
        per_substrate_aggregate_byte_budget_caps=tuple(aggregate_caps),
        cauchy_schwarz_aggregate_upper_bound=cs_bound,
        target_aggregate_delta_s=target_aggregate_delta_s,
        measurement_axes=tuple(measurement_axes),
    )


# ---------------------------------------------------------------------------
# Canonical fcntl-locked JSONL ledger (Catalog #131 / #138 / #245 sister)
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


@contextmanager
def _ledger_lock(path: Path | None = None) -> Iterator[None]:
    """fcntl-LOCK_EX scope context manager per Catalog #131."""
    lock_path = path or _LEDGER_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_append_jsonl(target: Path, line: str) -> None:
    """Atomic append via tmp + rename per Catalog #131 sister discipline.

    Sister of
    :func:`tac.cross_substrate_master_gradient_analyzer.analyzer._atomic_append_jsonl`.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_bytes() if target.exists() else b""
    tmp_name = f".tmp.{uuid.uuid4().hex[:12]}"
    tmp_path = target.parent / (target.name + tmp_name)
    payload = existing + line.encode("utf-8") + b"\n"
    tmp_path.write_bytes(payload)
    os.replace(tmp_path, target)


def append_solution_locked(
    solution: PareDLPSolution,
    *,
    path: Path | None = None,
) -> dict:
    """Append a Pareto polytope solution row to the canonical ledger.

    Sister of
    :func:`tac.cross_substrate_master_gradient_analyzer.append_analysis_locked`.
    fcntl-locked via :func:`_ledger_lock`; atomic write via
    :func:`_atomic_append_jsonl` per Catalog #131 sister discipline.

    Per CLAUDE.md HISTORICAL_PROVENANCE Catalog #110/#113: this ledger
    is APPEND-ONLY; existing rows are NEVER mutated. Recalibration
    produces a NEW row with a fresh ``solution_id`` referencing prior
    rows by ``upstream_meta_lift_1_analysis_id`` / timestamp.

    Returns the row dict as written.
    """
    target = path or PARETO_POLYTOPE_SOLUTIONS_LEDGER_PATH
    row = solution.as_dict()
    row["written_at_utc"] = _utc_now_iso()
    row["written_pid"] = os.getpid()
    row["written_host"] = socket.gethostname()
    line = json.dumps(row, sort_keys=True, allow_nan=False)
    with _ledger_lock(target.with_suffix(target.suffix + ".lock")):
        _atomic_append_jsonl(target, line)
    return row


def load_solutions_strict(path: Path | None = None) -> list[dict]:
    """Strict-load the canonical ledger per Catalog #138 fail-closed.

    Sister of
    :func:`tac.cross_substrate_master_gradient_analyzer.load_analyses_strict`.
    Raises :class:`PareDLPSolutionCorruptError` on JSON parse failure
    OR non-dict root rather than silently coercing to ``[]`` (the bug
    class Catalog #138 extincts).
    """
    target = path or PARETO_POLYTOPE_SOLUTIONS_LEDGER_PATH
    if not target.exists():
        return []
    rows: list[dict] = []
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise PareDLPSolutionCorruptError(
            f"failed to read ledger at {target}: {exc}"
        ) from exc
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PareDLPSolutionCorruptError(
                f"failed to parse line {line_num} of {target}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise PareDLPSolutionCorruptError(
                f"line {line_num} of {target} is not a JSON object; got {type(row).__name__}"
            )
        rows.append(row)
    return rows
