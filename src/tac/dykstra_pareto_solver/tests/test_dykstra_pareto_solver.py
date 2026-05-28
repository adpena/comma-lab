# SPDX-License-Identifier: MIT
"""Tests for tac.dykstra_pareto_solver (CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4)."""
from __future__ import annotations

import math

import pytest

from tac.dykstra_pareto_solver import (
    CANONICAL_3_AXIS_NAMES,
    DYKSTRA_DEFAULT_EPSILON,
    DYKSTRA_DEFAULT_MAX_ITERATIONS,
    DykstraParetoSolver,
)
from tac.dykstra_pareto_solver.polytope import (
    CANONICAL_3_AXIS_ORDERING,
    Polytope,
    PolytopeError,
)
from tac.dykstra_pareto_solver.solver import solve_pareto_polytope_intersection
from tac.dykstra_pareto_solver.verdict import (
    PARETO_SOLVER_VERDICT_SCHEMA_VERSION,
    ParetoSolverError,
    ParetoSolverVerdict,
    TIGHT_CONSTRAINT_LAMBDA_THRESHOLD,
)


# -----------------------------------------------------------------------------
# Polytope unit tests
# -----------------------------------------------------------------------------

def test_polytope_canonical_3_axis_construction():
    p = Polytope(
        axis_bounds={"seg": (0.0, 1.0), "pose": (0.0, 0.5), "rate": (-100.0, 100.0)},
    )
    assert p.is_canonical_3_axis
    assert p.axes == CANONICAL_3_AXIS_ORDERING


def test_polytope_general_construction():
    p = Polytope(
        axis_bounds={"x": (0.0, 1.0), "y": (-2.0, 2.0), "z": (10.0, 20.0)},
    )
    assert not p.is_canonical_3_axis
    assert set(p.axes) == {"x", "y", "z"}


def test_polytope_rejects_empty_axis_bounds():
    with pytest.raises(PolytopeError, match="non-empty"):
        Polytope(axis_bounds={})


def test_polytope_rejects_lower_greater_than_upper():
    with pytest.raises(PolytopeError, match="lower=.*upper="):
        Polytope(axis_bounds={"x": (1.0, 0.0)})


def test_polytope_rejects_nan_bound():
    with pytest.raises(PolytopeError, match="NaN"):
        Polytope(axis_bounds={"x": (float("nan"), 1.0)})


def test_polytope_rejects_infinite_bound():
    with pytest.raises(PolytopeError, match="infinite"):
        Polytope(axis_bounds={"x": (0.0, float("inf"))})


def test_polytope_rejects_non_string_axis_key():
    with pytest.raises(PolytopeError):
        Polytope(axis_bounds={0: (0.0, 1.0)})  # type: ignore


def test_polytope_rejects_halfspace_with_unknown_axis():
    with pytest.raises(PolytopeError, match="unknown axis"):
        Polytope(
            axis_bounds={"x": (0.0, 1.0)},
            halfspace_constraints=(({"y": 1.0}, 1.0),),
        )


def test_polytope_project_axis_aligned_closed_form():
    p = Polytope(
        axis_bounds={"seg": (0.0, 0.5), "pose": (0.0, 0.1), "rate": (-100.0, 100.0)},
    )
    # Point inside: returns unchanged.
    inside = p.project({"seg": 0.3, "pose": 0.05, "rate": 50.0})
    assert inside["seg"] == pytest.approx(0.3)
    assert inside["pose"] == pytest.approx(0.05)
    assert inside["rate"] == pytest.approx(50.0)
    # Point outside upper bound on seg: clipped.
    above = p.project({"seg": 1.0, "pose": 0.05, "rate": 50.0})
    assert above["seg"] == pytest.approx(0.5)
    # Point outside lower bound on rate: clipped.
    below = p.project({"seg": 0.3, "pose": 0.05, "rate": -200.0})
    assert below["rate"] == pytest.approx(-100.0)


def test_polytope_project_simplex_halfspace_iterative():
    # 4D simplex: 0 ≤ p_i ≤ 1 AND Σ_i p_i ≤ 1.
    p = Polytope(
        axis_bounds={f"p_{i}": (0.0, 1.0) for i in range(4)},
        halfspace_constraints=(({f"p_{i}": 1.0 for i in range(4)}, 1.0),),
    )
    # Initial point: all 0.5 → sum = 2.0, violates simplex.
    proj = p.project({f"p_{i}": 0.5 for i in range(4)})
    # Each value should be clipped/projected; sum ≤ 1.
    s = sum(proj.values())
    assert s <= 1.0 + 1e-6
    # By symmetry, projection should be 0.25 per component.
    for v in proj.values():
        assert v == pytest.approx(0.25, abs=1e-6)


def test_polytope_contains():
    p = Polytope(axis_bounds={"x": (0.0, 1.0), "y": (0.0, 2.0)})
    assert p.contains({"x": 0.5, "y": 1.0})
    assert not p.contains({"x": 1.5, "y": 1.0})
    assert not p.contains({"x": 0.5, "y": -0.1})
    # Edge values (with tolerance).
    assert p.contains({"x": 1.0, "y": 0.0})


def test_polytope_as_dict_roundtrip():
    p = Polytope(
        axis_bounds={"seg": (0.0, 0.5), "pose": (0.0, 0.1), "rate": (-100.0, 100.0)},
        name="test_polytope",
    )
    d = p.as_dict()
    assert d["name"] == "test_polytope"
    assert d["axes"] == ["seg", "pose", "rate"]
    assert d["is_canonical_3_axis"] is True
    assert d["axis_bounds"]["seg"] == [0.0, 0.5]


# -----------------------------------------------------------------------------
# DykstraParetoSolver unit tests
# -----------------------------------------------------------------------------

def test_solver_constructs_with_canonical_polytope():
    p = Polytope(axis_bounds={"seg": (0.0, 1.0), "pose": (0.0, 0.5), "rate": (-50.0, 50.0)})
    s = DykstraParetoSolver(polytope=p)
    assert s.tolerance == DYKSTRA_DEFAULT_EPSILON
    assert s.max_iterations == DYKSTRA_DEFAULT_MAX_ITERATIONS


def test_solver_rejects_non_polytope():
    with pytest.raises(ParetoSolverError, match="polytope must be Polytope"):
        DykstraParetoSolver(polytope="not_a_polytope")  # type: ignore


def test_solver_rejects_non_positive_tolerance():
    p = Polytope(axis_bounds={"x": (0.0, 1.0)})
    with pytest.raises(ParetoSolverError, match="tolerance"):
        DykstraParetoSolver(polytope=p, tolerance=-1.0)
    with pytest.raises(ParetoSolverError, match="tolerance"):
        DykstraParetoSolver(polytope=p, tolerance=0.0)


def test_solver_rejects_non_positive_max_iterations():
    p = Polytope(axis_bounds={"x": (0.0, 1.0)})
    with pytest.raises(ParetoSolverError, match="max_iterations"):
        DykstraParetoSolver(polytope=p, max_iterations=0)


def test_solver_canonical_3_axis_feasible_initial_point():
    p = Polytope(
        axis_bounds={"seg": (0.0, 0.5), "pose": (0.0, 0.1), "rate": (-100.0, 100.0)},
    )
    s = DykstraParetoSolver(polytope=p)
    v = s.solve(
        {"seg": 0.3, "pose": 0.05, "rate": 50.0},
        candidate_id="feasible_initial",
    )
    assert v.feasible
    assert v.converged
    assert all(v.per_axis_dual_variables[a] < TIGHT_CONSTRAINT_LAMBDA_THRESHOLD for a in CANONICAL_3_AXIS_ORDERING)
    assert v.tight_constraint_axes == ()
    assert set(v.slack_axes) == set(CANONICAL_3_AXIS_ORDERING)
    assert v.adjustment_factor == pytest.approx(1.0, abs=0.01)


def test_solver_canonical_3_axis_infeasible_initial_point():
    p = Polytope(
        axis_bounds={"seg": (0.0, 0.5), "pose": (0.0, 0.1), "rate": (-100.0, 100.0)},
    )
    s = DykstraParetoSolver(polytope=p)
    v = s.solve(
        {"seg": 2.0, "pose": 0.5, "rate": 500.0},  # all outside upper bounds
        candidate_id="infeasible_initial",
    )
    # Even infeasible initial point gets projected to the feasible boundary.
    assert v.feasible  # the projection IS feasible
    # All 3 axes should be tight (each contributed a non-zero correction).
    assert set(v.tight_constraint_axes) == set(CANONICAL_3_AXIS_ORDERING)
    assert v.slack_axes == ()
    # Per-axis dual variables non-negative.
    for axis in CANONICAL_3_AXIS_ORDERING:
        assert v.per_axis_dual_variables[axis] > 0
    # Scalar adjustment factor in [0.95, 1.05].
    assert 0.95 <= v.adjustment_factor <= 1.05


def test_solver_canonical_3_axis_partial_tightness():
    p = Polytope(
        axis_bounds={"seg": (0.0, 0.5), "pose": (0.0, 0.1), "rate": (-100.0, 100.0)},
    )
    s = DykstraParetoSolver(polytope=p)
    # seg is over upper; pose is fine; rate is over lower.
    v = s.solve(
        {"seg": 0.7, "pose": 0.05, "rate": -150.0},
        candidate_id="partial_tightness",
    )
    assert v.feasible
    # seg and rate should be tight; pose should be slack.
    assert "seg" in v.tight_constraint_axes
    assert "rate" in v.tight_constraint_axes
    assert "pose" in v.slack_axes


def test_solver_general_polytope_simplex():
    p = Polytope(
        axis_bounds={f"p_{i}": (0.0, 1.0) for i in range(3)},
        halfspace_constraints=(({f"p_{i}": 1.0 for i in range(3)}, 1.0),),
    )
    s = DykstraParetoSolver(polytope=p)
    v = s.solve(
        {f"p_{i}": 0.6 for i in range(3)},
        candidate_id="simplex_3_test",
    )
    assert v.feasible
    # Sum of projected components ≤ 1.
    s_total = sum(v.projection_point.values())
    assert s_total <= 1.0 + 1e-6


def test_solver_rejects_empty_candidate_id():
    p = Polytope(axis_bounds={"x": (0.0, 1.0)})
    s = DykstraParetoSolver(polytope=p)
    with pytest.raises(ParetoSolverError, match="candidate_id"):
        s.solve({"x": 0.5}, candidate_id="")


# -----------------------------------------------------------------------------
# solve_pareto_polytope_intersection convenience wrapper
# -----------------------------------------------------------------------------

def test_convenience_wrapper_canonical_3_axis():
    p = Polytope(axis_bounds={"seg": (0.0, 0.5), "pose": (0.0, 0.1), "rate": (-100.0, 100.0)})
    v = solve_pareto_polytope_intersection(
        p,
        initial_point={"seg": 0.3, "pose": 0.05, "rate": 50.0},
        candidate_id="convenience_test",
    )
    assert isinstance(v, ParetoSolverVerdict)
    assert v.feasible


def test_convenience_wrapper_threads_provenance():
    p = Polytope(axis_bounds={"seg": (0.0, 0.5), "pose": (0.0, 0.1), "rate": (-100.0, 100.0)})
    prov = {"axis_tag": "[predicted]", "evidence_grade": "predicted", "test_marker": "abc123"}
    v = solve_pareto_polytope_intersection(
        p,
        initial_point={"seg": 0.3, "pose": 0.05, "rate": 50.0},
        candidate_id="prov_test",
        canonical_provenance=prov,
    )
    assert v.canonical_provenance == prov


# -----------------------------------------------------------------------------
# ParetoSolverVerdict unit tests
# -----------------------------------------------------------------------------

def test_verdict_construction_clean():
    v = ParetoSolverVerdict(
        candidate_id="test_v",
        feasible=True,
        projection_point={"seg": 0.5, "pose": 0.1, "rate": 0.0},
        per_axis_dual_variables={"seg": 0.0, "pose": 0.0, "rate": 0.0},
        tight_constraint_axes=(),
        slack_axes=("seg", "pose", "rate"),
        per_axis_kkt_residuals={"seg": 0.0, "pose": 0.0, "rate": 0.0},
        per_axis_adjustment_factors={"seg": 1.0, "pose": 1.0, "rate": 1.0},
        adjustment_factor=1.0,
        convergence_residual=0.0,
        iteration_count=1,
        converged=True,
    )
    assert v.feasible
    assert v.axis_tag == "[predicted]"
    assert v.score_claim is False
    assert v.promotable is False


def test_verdict_rejects_negative_dual_variable():
    with pytest.raises(ParetoSolverError, match="must be >= 0"):
        ParetoSolverVerdict(
            candidate_id="test",
            feasible=True,
            projection_point={"a": 0.0},
            per_axis_dual_variables={"a": -1.0},
            tight_constraint_axes=(),
            slack_axes=("a",),
            per_axis_kkt_residuals={"a": 0.0},
            per_axis_adjustment_factors={"a": 1.0},
            adjustment_factor=1.0,
            convergence_residual=0.0,
            iteration_count=1,
            converged=True,
        )


def test_verdict_rejects_tight_slack_overlap():
    with pytest.raises(ParetoSolverError, match="disjoint"):
        ParetoSolverVerdict(
            candidate_id="test",
            feasible=True,
            projection_point={"a": 0.0},
            per_axis_dual_variables={"a": 0.5},
            tight_constraint_axes=("a",),
            slack_axes=("a",),
            per_axis_kkt_residuals={"a": 0.0},
            per_axis_adjustment_factors={"a": 1.0},
            adjustment_factor=1.0,
            convergence_residual=0.0,
            iteration_count=1,
            converged=True,
        )


def test_verdict_rejects_tight_axes_not_matching_threshold():
    with pytest.raises(ParetoSolverError, match="not in tight_constraint_axes"):
        ParetoSolverVerdict(
            candidate_id="test",
            feasible=True,
            projection_point={"a": 0.0, "b": 0.0},
            per_axis_dual_variables={"a": 0.5, "b": 0.0},  # a is tight but not declared
            tight_constraint_axes=(),
            slack_axes=("a", "b"),
            per_axis_kkt_residuals={"a": 0.0, "b": 0.0},
            per_axis_adjustment_factors={"a": 1.0, "b": 1.0},
            adjustment_factor=1.0,
            convergence_residual=0.0,
            iteration_count=1,
            converged=True,
        )


def test_verdict_rejects_adjustment_factor_out_of_band():
    with pytest.raises(ParetoSolverError, match="< 0.95"):
        ParetoSolverVerdict(
            candidate_id="test",
            feasible=True,
            projection_point={"a": 0.0},
            per_axis_dual_variables={"a": 0.0},
            tight_constraint_axes=(),
            slack_axes=("a",),
            per_axis_kkt_residuals={"a": 0.0},
            per_axis_adjustment_factors={"a": 1.0},
            adjustment_factor=0.5,  # OUT of band
            convergence_residual=0.0,
            iteration_count=1,
            converged=True,
        )


def test_verdict_rejects_nonzero_score_claim():
    with pytest.raises(ParetoSolverError, match="score_claim"):
        ParetoSolverVerdict(
            candidate_id="test",
            feasible=True,
            projection_point={"a": 0.0},
            per_axis_dual_variables={"a": 0.0},
            tight_constraint_axes=(),
            slack_axes=("a",),
            per_axis_kkt_residuals={"a": 0.0},
            per_axis_adjustment_factors={"a": 1.0},
            adjustment_factor=1.0,
            convergence_residual=0.0,
            iteration_count=1,
            converged=True,
            score_claim=True,  # FORBIDDEN per Catalog #341
        )


def test_verdict_rejects_nonzero_promotable():
    with pytest.raises(ParetoSolverError, match="promotable"):
        ParetoSolverVerdict(
            candidate_id="test",
            feasible=True,
            projection_point={"a": 0.0},
            per_axis_dual_variables={"a": 0.0},
            tight_constraint_axes=(),
            slack_axes=("a",),
            per_axis_kkt_residuals={"a": 0.0},
            per_axis_adjustment_factors={"a": 1.0},
            adjustment_factor=1.0,
            convergence_residual=0.0,
            iteration_count=1,
            converged=True,
            promotable=True,  # FORBIDDEN per Catalog #341
        )


def test_verdict_as_dict_serialization():
    v = ParetoSolverVerdict(
        candidate_id="test_serialize",
        feasible=True,
        projection_point={"a": 0.5},
        per_axis_dual_variables={"a": 0.0},
        tight_constraint_axes=(),
        slack_axes=("a",),
        per_axis_kkt_residuals={"a": 0.0},
        per_axis_adjustment_factors={"a": 1.0},
        adjustment_factor=1.0,
        convergence_residual=0.0,
        iteration_count=1,
        converged=True,
    )
    d = v.as_dict()
    assert d["schema_version"] == PARETO_SOLVER_VERDICT_SCHEMA_VERSION
    assert d["candidate_id"] == "test_serialize"
    assert d["axis_tag"] == "[predicted]"
    assert d["score_claim"] is False
    assert d["promotable"] is False
    # Should be JSON-safe.
    import json
    json.dumps(d)


# -----------------------------------------------------------------------------
# Boyd & Vandenberghe (2004) + Dykstra (1983) mathematical correctness
# -----------------------------------------------------------------------------

def test_boyd_alternating_projections_converges_on_simple_intersection():
    """Per Boyd 2004 Theorem 8.2: projection onto convex set is unique +
    well-defined. Verify the 3-axis polytope intersection converges to a
    point in the intersection."""
    p = Polytope(
        axis_bounds={"seg": (-1.0, 1.0), "pose": (-1.0, 1.0), "rate": (-1.0, 1.0)},
    )
    v = solve_pareto_polytope_intersection(
        p,
        initial_point={"seg": 0.0, "pose": 0.0, "rate": 0.0},
        candidate_id="origin_in_box",
    )
    assert v.feasible
    # Origin is inside the box; projection should be the origin itself.
    for axis in CANONICAL_3_AXIS_ORDERING:
        assert v.projection_point[axis] == pytest.approx(0.0, abs=1e-5)


def test_dykstra_correction_extracts_dual_variables_correctly():
    """Per Boyd-Dattorro (2006) § 6.2: the Dykstra correction vector at
    convergence equals λ × constraint_gradient. For axis-aligned half-
    spaces the gradient is the i-th unit vector, so |correction[i]| = λ_i.
    """
    p = Polytope(
        axis_bounds={"seg": (0.0, 0.5), "pose": (0.0, 0.1), "rate": (-100.0, 100.0)},
    )
    # Initial point exactly 1.0 above upper bound on seg.
    v = solve_pareto_polytope_intersection(
        p,
        initial_point={"seg": 1.5, "pose": 0.05, "rate": 0.0},
        candidate_id="seg_lambda_extraction",
    )
    # seg dual should be approximately 1.0 (the distance to upper bound).
    assert v.per_axis_dual_variables["seg"] == pytest.approx(1.0, abs=0.01)
    # Other duals should be near zero.
    assert v.per_axis_dual_variables["pose"] < TIGHT_CONSTRAINT_LAMBDA_THRESHOLD
    assert v.per_axis_dual_variables["rate"] < TIGHT_CONSTRAINT_LAMBDA_THRESHOLD


def test_idempotent_solve_on_feasible_point():
    """Solving from an already-feasible point should yield the same point."""
    p = Polytope(axis_bounds={"seg": (0.0, 1.0), "pose": (0.0, 1.0), "rate": (-1.0, 1.0)})
    start = {"seg": 0.3, "pose": 0.4, "rate": 0.1}
    v1 = solve_pareto_polytope_intersection(p, initial_point=start, candidate_id="idem_1")
    v2 = solve_pareto_polytope_intersection(
        p, initial_point=v1.projection_point, candidate_id="idem_2"
    )
    # Re-projection should land at the same point.
    for axis in CANONICAL_3_AXIS_ORDERING:
        assert v1.projection_point[axis] == pytest.approx(v2.projection_point[axis], abs=1e-5)


def test_general_polytope_dual_extraction_matches_distance():
    """For general polytopes, per-axis dual = |initial - projection| per axis."""
    p = Polytope(axis_bounds={"x": (0.0, 1.0), "y": (0.0, 2.0)})
    v = solve_pareto_polytope_intersection(
        p,
        initial_point={"x": 3.0, "y": 5.0},
        candidate_id="general_dual_test",
    )
    # x clipped from 3.0 to 1.0: dual = 2.0.
    assert v.per_axis_dual_variables["x"] == pytest.approx(2.0, abs=1e-6)
    # y clipped from 5.0 to 2.0: dual = 3.0.
    assert v.per_axis_dual_variables["y"] == pytest.approx(3.0, abs=1e-6)


# -----------------------------------------------------------------------------
# Canonical Provenance threading per Catalog #323
# -----------------------------------------------------------------------------

def test_canonical_provenance_threaded_through_verdict():
    from tac.provenance.builders import build_provenance_for_predicted
    from tac.provenance.validator import provenance_to_dict

    prov = build_provenance_for_predicted(
        model_id="test_canonical_provenance",
        inputs_sha256="a" * 64,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )
    prov_dict = provenance_to_dict(prov)
    p = Polytope(
        axis_bounds={"seg": (0.0, 0.5), "pose": (0.0, 0.1), "rate": (-100.0, 100.0)},
    )
    v = solve_pareto_polytope_intersection(
        p,
        initial_point={"seg": 0.3, "pose": 0.05, "rate": 50.0},
        candidate_id="prov_threading_test",
        canonical_provenance=prov_dict,
    )
    assert v.canonical_provenance == prov_dict


# -----------------------------------------------------------------------------
# Boundary cases
# -----------------------------------------------------------------------------

def test_solve_at_exact_bound():
    p = Polytope(axis_bounds={"x": (0.0, 1.0)})
    v = solve_pareto_polytope_intersection(
        p, initial_point={"x": 1.0}, candidate_id="exact_upper"
    )
    assert v.feasible
    assert v.projection_point["x"] == pytest.approx(1.0, abs=1e-6)


def test_solve_degenerate_polytope_zero_width():
    p = Polytope(axis_bounds={"x": (0.5, 0.5)})  # single point
    v = solve_pareto_polytope_intersection(
        p, initial_point={"x": 0.7}, candidate_id="degenerate_test"
    )
    assert v.feasible
    assert v.projection_point["x"] == pytest.approx(0.5, abs=1e-6)


def test_module_exports_canonical_facade_symbols():
    from tac import dykstra_pareto_solver as dps

    # All facade-level typed contracts must be exposed.
    assert hasattr(dps, "Polytope")
    assert hasattr(dps, "DykstraParetoSolver")
    assert hasattr(dps, "ParetoSolverVerdict")
    assert hasattr(dps, "solve_pareto_polytope_intersection")
    assert hasattr(dps, "TIGHT_CONSTRAINT_LAMBDA_THRESHOLD")
    # Canonical sister re-exports.
    assert hasattr(dps, "CANONICAL_3_AXIS_NAMES")
    assert hasattr(dps, "PerAxisDualSolverResult")
    assert hasattr(dps, "dykstra_alternating_projections_3_axis")
    assert hasattr(dps, "compute_per_axis_dual_variables")
