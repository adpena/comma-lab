# SPDX-License-Identifier: MIT
"""Tests for ``tac.optimization.field_equation_planner.consume_per_pair_lagrangian_duals``.

MEDIUM gap closure wave 2026-05-17 — `lane_medium_gap_closure_3_optimization_
modules_per_pair_consumption_20260517`. Closes GAP-3 from the comprehensive
wiring + integration audit by binding per-pair Lagrangian dual variables
(λ_archive / λ_compute / λ_inflate / ν_per_pair) from the OptimalPerPair-
TreatmentPlan into the field-equation planner's constraint multipliers.

Per CLAUDE.md "Apples-to-apples evidence discipline" + "Meta-Lagrangian/Pareto
solver" non-negotiable: every output carries `[predicted; ...]` and
`score_claim=False`.
"""

from __future__ import annotations

import pytest

from tac.optimization.field_equation_planner import (
    DEFAULT_CONSTRAINTS,
    PER_PAIR_LAGRANGIAN_DUAL_SCHEMA,
    FieldEquationPlannerError,
    consume_per_pair_lagrangian_duals,
)


def test_fallback_path_when_no_plan_available() -> None:
    """No optimal_plan + auto_load disabled → default-constraints fallback."""
    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        auto_load=False,
    )
    assert result["schema"] == PER_PAIR_LAGRANGIAN_DUAL_SCHEMA
    assert result["cascade_path_used"] == "default_constraints_fallback"
    assert result["optimal_plan_consumed"] is False
    assert result["lambda_archive"] is None
    assert result["lambda_compute"] is None
    assert result["lambda_inflate"] is None
    assert result["nu_per_pair"] == ()
    assert result["n_pairs"] == 0
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "predicted" in result["evidence_grade"].lower()
    # Fallback envelope provides the default constraints surface
    assert result["bound_constraints"] == dict(DEFAULT_CONSTRAINTS)


def test_optimal_plan_dict_path_binds_duals() -> None:
    """Optimal-plan dict → binds λ + ν onto constraint multipliers."""
    plan_dict = {
        "lambda_archive": 0.5,
        "lambda_compute": 0.3,
        "lambda_inflate": 0.7,
        "nu_per_pair": [0.1, 0.2, 0.3, 0.4],
        "kkt_residual": 1e-6,
        "feasibility_certificate": {"archive": True, "compute": True, "inflate": True},
        "is_pareto_feasible": True,
        "predicted_score_delta": -0.001,
    }
    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        optimal_plan=plan_dict,
        auto_load=False,
    )
    assert result["cascade_path_used"] == "optimal_plan_binding"
    assert result["optimal_plan_consumed"] is True
    assert result["lambda_archive"] == 0.5
    assert result["lambda_compute"] == 0.3
    assert result["lambda_inflate"] == 0.7
    assert result["n_pairs"] == 4
    assert result["nu_per_pair"] == (0.1, 0.2, 0.3, 0.4)
    assert result["kkt_residual"] == 1e-6
    assert result["is_pareto_feasible"] is True
    assert result["predicted_score_delta"] == -0.001
    assert result["feasibility_certificate"] == {
        "archive": True, "compute": True, "inflate": True
    }


def test_dataclass_surface_binds_duals() -> None:
    """OptimalPerPairTreatmentPlan dataclass-like object also accepted."""

    class FakePlan:
        lambda_archive = 0.5
        lambda_compute = 0.3
        lambda_inflate = 0.7
        nu_per_pair = (0.1, 0.2)
        kkt_residual = 1e-6
        feasibility_certificate = {"archive": True}
        is_pareto_feasible = True
        predicted_score_delta = -0.001

    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        optimal_plan=FakePlan(),
        auto_load=False,
    )
    assert result["cascade_path_used"] == "optimal_plan_binding"
    assert result["lambda_archive"] == 0.5
    assert result["n_pairs"] == 2


def test_bound_constraints_modify_lambda_byte_violation() -> None:
    """λ_archive > 0 → bound_constraints['lambda_byte_violation'] reflects it."""
    plan_dict = {
        "lambda_archive": 2.5,
        "lambda_compute": 0.0,
        "lambda_inflate": 0.0,
        "nu_per_pair": [],
        "kkt_residual": 0.0,
        "feasibility_certificate": {},
        "is_pareto_feasible": True,
        "predicted_score_delta": 0.0,
    }
    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        optimal_plan=plan_dict,
        auto_load=False,
    )
    # lambda_byte_violation bound to lambda_archive (>= 0 floored)
    assert result["bound_constraints"]["lambda_byte_violation"] == 2.5


def test_bound_constraints_scale_compute_to_proxy() -> None:
    """λ_compute scales into lambda_proxy_violation per binding contract."""
    plan_dict = {
        "lambda_archive": 0.0,
        "lambda_compute": 0.5,
        "lambda_inflate": 0.0,
        "nu_per_pair": [],
        "kkt_residual": 0.0,
        "feasibility_certificate": {},
        "is_pareto_feasible": True,
        "predicted_score_delta": 0.0,
    }
    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        optimal_plan=plan_dict,
        auto_load=False,
    )
    # lambda_proxy_violation = lambda_compute * 50.0 (>=0)
    assert result["bound_constraints"]["lambda_proxy_violation"] == 25.0


def test_bound_constraints_scale_inflate_to_kkt_readiness() -> None:
    """λ_inflate scales into lambda_kkt_readiness_violation per binding contract."""
    plan_dict = {
        "lambda_archive": 0.0,
        "lambda_compute": 0.0,
        "lambda_inflate": 0.8,
        "nu_per_pair": [],
        "kkt_residual": 0.0,
        "feasibility_certificate": {},
        "is_pareto_feasible": True,
        "predicted_score_delta": 0.0,
    }
    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        optimal_plan=plan_dict,
        auto_load=False,
    )
    # lambda_kkt_readiness_violation = lambda_inflate * 10.0 (>=0)
    assert result["bound_constraints"]["lambda_kkt_readiness_violation"] == 8.0


def test_negative_duals_floored_at_zero() -> None:
    """Negative dual variables get floored at 0 in bound_constraints."""
    plan_dict = {
        "lambda_archive": -0.5,
        "lambda_compute": -0.3,
        "lambda_inflate": -0.7,
        "nu_per_pair": [],
        "kkt_residual": 0.0,
        "feasibility_certificate": {},
        "is_pareto_feasible": True,
        "predicted_score_delta": 0.0,
    }
    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        optimal_plan=plan_dict,
        auto_load=False,
    )
    assert result["bound_constraints"]["lambda_byte_violation"] == 0.0
    assert result["bound_constraints"]["lambda_proxy_violation"] == 0.0
    assert result["bound_constraints"]["lambda_kkt_readiness_violation"] == 0.0
    # The raw dual variables are preserved as-is (NOT floored at the envelope surface)
    assert result["lambda_archive"] == -0.5


def test_seg_pose_multipliers_NOT_touched_by_binding() -> None:
    """lambda_seg / lambda_pose NOT bound from dual variables (per CLAUDE.md)."""
    plan_dict = {
        "lambda_archive": 5.0,
        "lambda_compute": 5.0,
        "lambda_inflate": 5.0,
        "nu_per_pair": [],
        "kkt_residual": 0.0,
        "feasibility_certificate": {},
        "is_pareto_feasible": True,
        "predicted_score_delta": 0.0,
    }
    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        optimal_plan=plan_dict,
        auto_load=False,
    )
    # lambda_seg_violation + lambda_pose_violation preserved at defaults
    assert (
        result["bound_constraints"]["lambda_seg_violation"]
        == DEFAULT_CONSTRAINTS["lambda_seg_violation"]
    )
    assert (
        result["bound_constraints"]["lambda_pose_violation"]
        == DEFAULT_CONSTRAINTS["lambda_pose_violation"]
    )


def test_invalid_archive_sha_rejected() -> None:
    """Sub-12-char or non-hex sha rejected with FieldEquationPlannerError."""
    with pytest.raises(FieldEquationPlannerError, match="archive_sha256 must be"):
        consume_per_pair_lagrangian_duals(
            archive_sha256="abc",
            auto_load=False,
        )
    with pytest.raises(FieldEquationPlannerError, match="archive_sha256 must be"):
        consume_per_pair_lagrangian_duals(
            archive_sha256="not-a-hex-sha",
            auto_load=False,
        )


def test_invalid_optimal_plan_type_rejected() -> None:
    """Wrong-type optimal_plan rejected with FieldEquationPlannerError."""
    with pytest.raises(FieldEquationPlannerError, match="must be OptimalPerPairTreatmentPlan"):
        consume_per_pair_lagrangian_duals(
            archive_sha256="deadbeef1234567890abcdef",
            optimal_plan=42,  # not dict, not dataclass
            auto_load=False,
        )


def test_envelope_score_claim_false_invariant() -> None:
    """Every envelope carries score_claim=False per Apples-to-apples."""
    plan_dict = {
        "lambda_archive": 0.5,
        "lambda_compute": 0.5,
        "lambda_inflate": 0.5,
        "nu_per_pair": [0.1, 0.2],
        "kkt_residual": 0.0,
        "feasibility_certificate": {},
        "is_pareto_feasible": True,
        "predicted_score_delta": -0.001,
    }
    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        optimal_plan=plan_dict,
        auto_load=False,
    )
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "[predicted;" in result["evidence_grade"]


def test_rationale_explains_cascade_path() -> None:
    """Rationale string mentions binding + dual values."""
    plan_dict = {
        "lambda_archive": 1.5,
        "lambda_compute": 0.5,
        "lambda_inflate": 0.7,
        "nu_per_pair": [0.1, 0.2],
        "kkt_residual": 1e-6,
        "feasibility_certificate": {},
        "is_pareto_feasible": True,
        "predicted_score_delta": -0.001,
    }
    result = consume_per_pair_lagrangian_duals(
        archive_sha256="deadbeef1234567890abcdef",
        optimal_plan=plan_dict,
        auto_load=False,
    )
    assert "OptimalPerPairTreatmentPlan" in result["rationale"]
    assert "lambda_archive" in result["rationale"]
    assert "kkt_residual" in result["rationale"]
