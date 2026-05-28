# SPDX-License-Identifier: MIT
"""Layer 5 Wave N+2: AntiPatternConstraint <-> DykstraParetoSolver integration tests.

Per canonical anti-patterns design memo §"Layer 5" + Layer 1+2 landing
memo + Wave N+2 mandate. Verifies:

  * ``AntiPatternConstraint`` frozen dataclass invariants
  * ``DykstraParetoSolver`` baseline regression (empty constraints =
    identical behavior to Slot 1 Wave N+1)
  * Anti-pattern constraints excluding the projection region revoke
    feasibility (the MAX-aggregation identity per design memo)
  * Per-anti-pattern dual variables surface in
    :attr:`ParetoSolverVerdict.per_axis_dual_variables` with the
    canonical prefix ``anti_pattern_<id>``
  * Tight-axis classification distinguishes axis-dual vs anti-pattern-
    dual via prefix inspection
  * Multiple anti-patterns MAX-aggregate (severity-weighted) per the
    design memo §"Mathematical compounding identity"
  * End-to-end integration with
    :func:`tools.cathedral_autopilot_autonomous_loop.invoke_dykstra_pareto_solver_on_candidates`
  * Existing 39 Slot 1 Wave N+1 tests still pass (regression guard via
    pytest collection)
"""
from __future__ import annotations

import pytest

from tac.dykstra_pareto_solver import (
    ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX,
    AntiPatternConstraint,
    AntiPatternConstraintError,
    DykstraParetoSolver,
    Polytope,
    aggregate_anti_pattern_duals,
    anti_pattern_severity_weight_for,
    solve_pareto_polytope_intersection,
)


_CANONICAL_3_AXIS_BOUNDS = {
    "seg": (0.0, 0.5),
    "pose": (0.0, 0.1),
    "rate": (-50_000.0, 0.0),
}


# ──────────────────────────────────────────────────────────────────────
# AntiPatternConstraint frozen dataclass invariants
# ──────────────────────────────────────────────────────────────────────


def test_anti_pattern_constraint_happy_path():
    c = AntiPatternConstraint(
        anti_pattern_id="lzma_on_already_brotli_saturated_compounding_v1",
        forbidden_region_predicate=lambda _point: True,
        severity="medium_substrate_regression",
        canonical_unwind_path="choose ONE entropy coder standalone",
    )
    assert c.anti_pattern_id == "lzma_on_already_brotli_saturated_compounding_v1"
    assert c.severity == "medium_substrate_regression"
    assert c.severity_weight == 0.50
    assert (
        c.dual_variable_key
        == f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}lzma_on_already_brotli_saturated_compounding_v1"
    )


def test_anti_pattern_constraint_frozen():
    c = AntiPatternConstraint(
        anti_pattern_id="test_v1",
        forbidden_region_predicate=lambda _: False,
        severity="low_implementation_inefficiency",
        canonical_unwind_path="x",
    )
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        c.severity = "high_compound_corruption"  # type: ignore[misc]


def test_anti_pattern_constraint_rejects_empty_id():
    with pytest.raises(AntiPatternConstraintError):
        AntiPatternConstraint(
            anti_pattern_id="",
            forbidden_region_predicate=lambda _: True,
            severity="medium_substrate_regression",
            canonical_unwind_path="x",
        )


def test_anti_pattern_constraint_rejects_non_callable_predicate():
    with pytest.raises(AntiPatternConstraintError):
        AntiPatternConstraint(
            anti_pattern_id="test_v1",
            forbidden_region_predicate="not callable",  # type: ignore[arg-type]
            severity="medium_substrate_regression",
            canonical_unwind_path="x",
        )


def test_anti_pattern_constraint_rejects_invalid_severity():
    with pytest.raises(AntiPatternConstraintError):
        AntiPatternConstraint(
            anti_pattern_id="test_v1",
            forbidden_region_predicate=lambda _: False,
            severity="not_a_canonical_severity",
            canonical_unwind_path="x",
        )


def test_anti_pattern_constraint_rejects_empty_unwind_path():
    with pytest.raises(AntiPatternConstraintError):
        AntiPatternConstraint(
            anti_pattern_id="test_v1",
            forbidden_region_predicate=lambda _: False,
            severity="medium_substrate_regression",
            canonical_unwind_path="",
        )


def test_anti_pattern_constraint_severity_weight_canonical_taxonomy():
    # Per design memo: CRITICAL > HIGH > MEDIUM > LOW
    assert anti_pattern_severity_weight_for("critical_paradigm_blocker") == 1.0
    assert anti_pattern_severity_weight_for("high_compound_corruption") == 0.75
    assert anti_pattern_severity_weight_for("medium_substrate_regression") == 0.50
    assert anti_pattern_severity_weight_for("low_implementation_inefficiency") == 0.25


def test_anti_pattern_constraint_dual_at_inside_point():
    c = AntiPatternConstraint(
        anti_pattern_id="test_v1",
        forbidden_region_predicate=lambda p: p.get("seg", 0.0) > 0.05,
        severity="high_compound_corruption",
        canonical_unwind_path="x",
    )
    assert c.dual_variable({"seg": 0.1, "pose": 0.0, "rate": 0.0}) == 0.75
    assert c.dual_variable({"seg": 0.01, "pose": 0.0, "rate": 0.0}) == 0.0


def test_anti_pattern_constraint_rejects_non_bool_predicate():
    c = AntiPatternConstraint(
        anti_pattern_id="test_v1",
        forbidden_region_predicate=lambda _: 1,  # int not bool
        severity="medium_substrate_regression",
        canonical_unwind_path="x",
    )
    with pytest.raises(AntiPatternConstraintError):
        c.dual_variable({"seg": 0.0})


def test_aggregate_anti_pattern_duals_max_aggregation():
    # Per design memo §"Mathematical compounding identity": MAX not SUM
    c1 = AntiPatternConstraint(
        anti_pattern_id="ap_critical_v1",
        forbidden_region_predicate=lambda _: True,
        severity="critical_paradigm_blocker",
        canonical_unwind_path="x_critical",
    )
    c2 = AntiPatternConstraint(
        anti_pattern_id="ap_low_v1",
        forbidden_region_predicate=lambda _: True,
        severity="low_implementation_inefficiency",
        canonical_unwind_path="x_low",
    )
    per_constraint, max_dual, binding_paths = aggregate_anti_pattern_duals(
        {"seg": 0.0}, (c1, c2)
    )
    # MAX = 1.0 (critical) NOT SUM = 1.25
    assert max_dual == 1.0
    # Both binding because predicates always True
    assert per_constraint == {
        "anti_pattern_ap_critical_v1": 1.0,
        "anti_pattern_ap_low_v1": 0.25,
    }
    # Sorted by descending severity weight: critical first
    assert binding_paths == ("x_critical", "x_low")


def test_aggregate_anti_pattern_duals_no_binding():
    c = AntiPatternConstraint(
        anti_pattern_id="ap_v1",
        forbidden_region_predicate=lambda _: False,
        severity="critical_paradigm_blocker",
        canonical_unwind_path="x",
    )
    per_constraint, max_dual, binding_paths = aggregate_anti_pattern_duals(
        {"seg": 0.0}, (c,)
    )
    assert max_dual == 0.0
    assert per_constraint == {"anti_pattern_ap_v1": 0.0}
    assert binding_paths == ()


def test_aggregate_anti_pattern_duals_empty_constraints():
    per_constraint, max_dual, binding_paths = aggregate_anti_pattern_duals(
        {"seg": 0.0}, ()
    )
    assert max_dual == 0.0
    assert per_constraint == {}
    assert binding_paths == ()


def test_aggregate_anti_pattern_duals_rejects_non_mapping_point():
    with pytest.raises(AntiPatternConstraintError):
        aggregate_anti_pattern_duals([0.1, 0.05], ())  # type: ignore[arg-type]


def test_aggregate_anti_pattern_duals_rejects_non_tuple_constraints():
    with pytest.raises(AntiPatternConstraintError):
        aggregate_anti_pattern_duals({"seg": 0.0}, [])  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────
# DykstraParetoSolver baseline regression (empty constraints)
# ──────────────────────────────────────────────────────────────────────


def test_solver_empty_anti_patterns_is_wave_n_plus_1_baseline():
    """Wave N+2 mandate Layer 5: empty anti_pattern_constraints behaves
    identically to Slot 1 Wave N+1 baseline (regression guard)."""
    polytope = Polytope(axis_bounds=_CANONICAL_3_AXIS_BOUNDS)
    verdict_no_constraints = solve_pareto_polytope_intersection(
        polytope,
        initial_point={"seg": 0.1, "pose": 0.05, "rate": -10_000.0},
        candidate_id="baseline",
    )
    verdict_empty_constraints = solve_pareto_polytope_intersection(
        polytope,
        initial_point={"seg": 0.1, "pose": 0.05, "rate": -10_000.0},
        candidate_id="baseline",
        anti_pattern_constraints=(),
    )
    # Per-axis dual variables identical (no anti_pattern_<id> keys added)
    assert (
        set(verdict_no_constraints.per_axis_dual_variables.keys())
        == set(verdict_empty_constraints.per_axis_dual_variables.keys())
        == {"seg", "pose", "rate"}
    )
    assert verdict_no_constraints.feasible == verdict_empty_constraints.feasible
    assert (
        verdict_no_constraints.tight_constraint_axes
        == verdict_empty_constraints.tight_constraint_axes
    )


def test_solver_with_anti_pattern_excludes_projection_region():
    """Layer 5: a constraint matching the projection region revokes
    feasibility per the MAX-aggregation identity."""
    polytope = Polytope(axis_bounds=_CANONICAL_3_AXIS_BOUNDS)
    c = AntiPatternConstraint(
        anti_pattern_id="excludes_all_v1",
        forbidden_region_predicate=lambda _point: True,
        severity="critical_paradigm_blocker",
        canonical_unwind_path="use canonical alternative",
    )
    verdict = solve_pareto_polytope_intersection(
        polytope,
        initial_point={"seg": 0.1, "pose": 0.05, "rate": -10_000.0},
        candidate_id="constrained",
        anti_pattern_constraints=(c,),
    )
    # Anti-pattern key surfaces in dual variables map
    assert (
        f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}excludes_all_v1"
        in verdict.per_axis_dual_variables
    )
    assert (
        verdict.per_axis_dual_variables[
            f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}excludes_all_v1"
        ]
        == 1.0
    )
    # Constraint binding -> feasibility revoked
    assert not verdict.feasible
    # Tight axes includes the anti-pattern key
    assert (
        f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}excludes_all_v1"
        in verdict.tight_constraint_axes
    )


def test_solver_with_slack_anti_pattern_preserves_feasibility():
    """An anti-pattern constraint whose forbidden region does NOT
    contain the projection point leaves feasibility intact."""
    polytope = Polytope(axis_bounds=_CANONICAL_3_AXIS_BOUNDS)
    c = AntiPatternConstraint(
        anti_pattern_id="never_binds_v1",
        forbidden_region_predicate=lambda _point: False,
        severity="critical_paradigm_blocker",
        canonical_unwind_path="x",
    )
    verdict = solve_pareto_polytope_intersection(
        polytope,
        initial_point={"seg": 0.1, "pose": 0.05, "rate": -10_000.0},
        candidate_id="slack",
        anti_pattern_constraints=(c,),
    )
    # Anti-pattern key in dual map but value is 0
    assert (
        verdict.per_axis_dual_variables[
            f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}never_binds_v1"
        ]
        == 0.0
    )
    # Feasibility preserved (no binding constraint)
    assert verdict.feasible
    # Tight axes does NOT include the slack anti-pattern key
    assert (
        f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}never_binds_v1"
        not in verdict.tight_constraint_axes
    )
    # Slack axes includes the slack anti-pattern key
    assert (
        f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}never_binds_v1"
        in verdict.slack_axes
    )


def test_solver_multi_anti_patterns_max_aggregated():
    """Multiple constraints MAX-aggregate (severity-weighted) per design memo."""
    polytope = Polytope(axis_bounds=_CANONICAL_3_AXIS_BOUNDS)
    c_high = AntiPatternConstraint(
        anti_pattern_id="high_v1",
        forbidden_region_predicate=lambda _: True,
        severity="high_compound_corruption",
        canonical_unwind_path="x_high",
    )
    c_low = AntiPatternConstraint(
        anti_pattern_id="low_v1",
        forbidden_region_predicate=lambda _: True,
        severity="low_implementation_inefficiency",
        canonical_unwind_path="x_low",
    )
    verdict = solve_pareto_polytope_intersection(
        polytope,
        initial_point={"seg": 0.1, "pose": 0.05, "rate": -10_000.0},
        candidate_id="multi",
        anti_pattern_constraints=(c_high, c_low),
    )
    duals = verdict.per_axis_dual_variables
    assert duals[f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}high_v1"] == 0.75
    assert duals[f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}low_v1"] == 0.25
    # Both binding; feasibility revoked
    assert not verdict.feasible


def test_solver_per_axis_tight_distinguished_from_anti_pattern_tight():
    """Axis-dual + anti-pattern-dual coexist in tight set; consumer
    distinguishes via canonical prefix."""
    polytope = Polytope(axis_bounds=_CANONICAL_3_AXIS_BOUNDS)
    c = AntiPatternConstraint(
        anti_pattern_id="tight_v1",
        forbidden_region_predicate=lambda _: True,
        severity="critical_paradigm_blocker",
        canonical_unwind_path="x",
    )
    verdict = solve_pareto_polytope_intersection(
        polytope,
        # initial point OUTSIDE polytope on seg axis to force a tight axis
        initial_point={"seg": 0.9, "pose": 0.05, "rate": -10_000.0},
        candidate_id="mixed",
        anti_pattern_constraints=(c,),
    )
    # Anti-pattern key prefix discoverable
    ap_tight = [
        k
        for k in verdict.tight_constraint_axes
        if k.startswith(ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX)
    ]
    axis_tight = [
        k
        for k in verdict.tight_constraint_axes
        if not k.startswith(ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX)
    ]
    assert len(ap_tight) == 1
    assert ap_tight[0] == f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}tight_v1"
    # axis-tight is non-empty because seg axis is binding
    assert "seg" in axis_tight


def test_solver_dataclass_rejects_non_tuple_constraints():
    polytope = Polytope(axis_bounds=_CANONICAL_3_AXIS_BOUNDS)
    with pytest.raises(Exception):  # ParetoSolverError wraps the contract violation
        DykstraParetoSolver(
            polytope=polytope,
            anti_pattern_constraints=[
                AntiPatternConstraint(
                    anti_pattern_id="v1",
                    forbidden_region_predicate=lambda _: True,
                    severity="low_implementation_inefficiency",
                    canonical_unwind_path="x",
                )
            ],  # type: ignore[arg-type]
        )


def test_solver_dataclass_rejects_duplicate_anti_pattern_ids():
    polytope = Polytope(axis_bounds=_CANONICAL_3_AXIS_BOUNDS)
    c1 = AntiPatternConstraint(
        anti_pattern_id="dup_v1",
        forbidden_region_predicate=lambda _: True,
        severity="medium_substrate_regression",
        canonical_unwind_path="x",
    )
    c2 = AntiPatternConstraint(
        anti_pattern_id="dup_v1",  # duplicate
        forbidden_region_predicate=lambda _: False,
        severity="high_compound_corruption",
        canonical_unwind_path="y",
    )
    with pytest.raises(Exception):  # ParetoSolverError
        DykstraParetoSolver(
            polytope=polytope,
            anti_pattern_constraints=(c1, c2),
        )


def test_solver_per_axis_adjustment_factors_unchanged_by_anti_pattern():
    """Catalog #341 observability-only: anti-pattern constraints do NOT
    add an axis adjustment factor; they only reshape feasibility set.
    Per-axis (seg/pose/rate) factors remain in [0.95, 1.05]."""
    polytope = Polytope(axis_bounds=_CANONICAL_3_AXIS_BOUNDS)
    c = AntiPatternConstraint(
        anti_pattern_id="obs_only_v1",
        forbidden_region_predicate=lambda _: True,
        severity="high_compound_corruption",
        canonical_unwind_path="x",
    )
    verdict = solve_pareto_polytope_intersection(
        polytope,
        initial_point={"seg": 0.1, "pose": 0.05, "rate": -10_000.0},
        candidate_id="obs_only",
        anti_pattern_constraints=(c,),
    )
    # Anti-pattern factor is 1.0 (observability-only contract)
    assert (
        verdict.per_axis_adjustment_factors[
            f"{ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX}obs_only_v1"
        ]
        == 1.0
    )
    # Axis factors remain in [0.95, 1.05] per Catalog #355 envelope
    for axis in ("seg", "pose", "rate"):
        af = verdict.per_axis_adjustment_factors[axis]
        assert 0.95 <= af <= 1.05


# ──────────────────────────────────────────────────────────────────────
# End-to-end cathedral autopilot integration
# ──────────────────────────────────────────────────────────────────────


def test_cathedral_autopilot_invoker_surfaces_matched_anti_patterns():
    """End-to-end: synthetic candidate with stack_spec matching anti-pattern
    surfaces matched_anti_patterns + canonical_unwind_paths in payload."""
    import sys
    from pathlib import Path

    # Canonical import via the `tools.` package path keeps dataclass
    # module-attribution intact (importlib.util.spec_from_file_location
    # breaks dataclass synthesis on Python 3.12).
    repo_root = Path(__file__).resolve().parents[4]
    tools_dir = str(repo_root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        m = __import__("cathedral_autopilot_autonomous_loop")
    except ImportError:
        pytest.skip("Cannot import cathedral_autopilot_autonomous_loop")

    CandidateRow = m.CandidateRow

    # Synthetic candidate carrying stack_spec that strongly references
    # the brotli+lzma anti-pattern (Layer 1+2 builtin #4).
    candidate = CandidateRow(
        candidate_id="synthetic_brotli_plus_lzma",
        family="diagnostic_test",
        predicted_score_delta=-0.001,
        expected_information_gain=0.0,
        estimated_dispatch_cost_usd=0.0,
    )
    # Inject stack_spec attribute (CandidateRow is a regular dataclass;
    # we use object.__setattr__ to bypass frozen if needed)
    try:
        object.__setattr__(
            candidate,
            "stack_spec",
            {
                "substrate_id": "synthetic_diag",
                "compression_ops": ["brotli_q11", "lzma_q9"],
                "description": (
                    "chained brotli + lzma compounding entropy coders "
                    "that operate on similar redundancy domains"
                ),
            },
        )
    except Exception:
        pytest.skip("CandidateRow does not support stack_spec injection")

    result = m.invoke_dykstra_pareto_solver_on_candidates(
        [candidate], top_n=1
    )
    assert result["schema"].startswith("dykstra_pareto_solver_invocation_")
    assert result["candidates_invoked"] == 1
    # Layer 5 Wave N+2 surfaces:
    assert "anti_pattern_binding_histogram" in result
    assert "candidates_with_matched_anti_patterns" in result
    assert "total_matched_anti_pattern_occurrences" in result
    assert "canonical_unwind_paths_recommended" in result
    assert (
        result["anti_pattern_constraint_canonical_equation_id"]
        == "anti_pattern_polytope_exclusion_dykstra_compounding_v1"
    )
    # Per-candidate surface present even if no anti-patterns matched
    inv = result["invocations"][0]
    assert "matched_anti_patterns" in inv
    assert "binding_anti_pattern_ids" in inv
    assert "canonical_unwind_paths_recommended" in inv
    assert "anti_pattern_constraint_count" in inv
    # If the registered Layer 1+2 builtin brotli+lzma anti-pattern was
    # matched, the constraint count is > 0.
    if inv["anti_pattern_constraint_count"] > 0:
        assert len(inv["matched_anti_patterns"]) > 0
