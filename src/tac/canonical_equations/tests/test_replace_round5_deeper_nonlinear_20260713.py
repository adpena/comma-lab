# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.registry import query_equations
from tac.canonical_equations.replace_round5_deeper_nonlinear_20260713 import (
    EQUATION_ID,
    build_replace_round5_deeper_nonlinear_v1,
    equal_exact_call_branch_design,
    populate_replace_round5_deeper_nonlinear_v1,
    post_se_sparse_teacher_economics,
    query_refuse_audit_budget,
)


def test_post_se_economics_matches_sealed_block3_cut() -> None:
    result = post_se_sparse_teacher_economics(
        feature_cut_fraction=0.07129461126470672,
        selected_area_fraction=0.047017415364583336,
        anchor_calls=600,
    )
    assert result["conditional_c_label"] == pytest.approx(0.11495993827820083)
    assert result["conditional_variable_cost_reduction_x"] == pytest.approx(
        8.698682471279858
    )
    assert result["break_even_future_steps_D"] == pytest.approx(677.9354132656225)


def test_branch_design_charges_each_horizon_equally() -> None:
    result = equal_exact_call_branch_design(
        horizons=(0, 1, 2, 4), exact_calls_per_horizon=120
    )
    assert result["baseline_horizon"] == 0
    assert result["total_exact_call_budget"] == 480


def test_query_budget_has_positive_randomized_propensity() -> None:
    result = query_refuse_audit_budget(
        total_cells=49_152,
        targeted_fraction=0.04,
        random_audit_fraction=0.01,
    )
    assert result["targeted_count"] == 1_967
    assert result["random_audit_count"] == 492
    assert result["queried_count"] == 2_459
    assert result["realized_query_fraction"] == pytest.approx(2_459 / 49_152)
    assert result["random_audit_positive_propensity"] > 0.0


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            post_se_sparse_teacher_economics,
            {
                "feature_cut_fraction": -0.1,
                "selected_area_fraction": 0.1,
                "anchor_calls": 1,
            },
        ),
        (
            equal_exact_call_branch_design,
            {"horizons": (1, 2), "exact_calls_per_horizon": 1},
        ),
        (
            query_refuse_audit_budget,
            {
                "total_cells": 2,
                "targeted_fraction": 0.9,
                "random_audit_fraction": 0.9,
            },
        ),
    ],
)
def test_closed_laws_fail_closed(function, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        function(**kwargs)


def test_equation_preserves_feature_source_scope_and_ticket_outcomes() -> None:
    equation = build_replace_round5_deeper_nonlinear_v1()
    anchor = equation.empirical_anchors[0]
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["scope_level"] == (
        "family x feature-source x fixed replay"
    )
    assert anchor.empirical_output["convex_deeper_retained_mass_fraction"] == pytest.approx(
        0.13046753525944724
    )
    assert anchor.empirical_output["nonlinear_ensemble_retained_mass_fraction"] == pytest.approx(
        0.29462633883840517
    )
    assert anchor.empirical_output["query_research_gate_pass"] is True
    assert anchor.empirical_output["branch_horizon_status"] == "blocked-not-identified"
    assert anchor.empirical_output["pay_only_on_support_admitted"] is False
    assert equation.provenance.score_claim_valid is False
    assert equation.provenance.promotion_eligible is False


def test_population_round_trips_through_isolated_locked_registry(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_replace_round5_deeper_nonlinear_v1(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="replace_round5",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line]
    loaded = query_equations(path=registry)
    assert populated.equation_id == EQUATION_ID
    assert [row.equation_id for row in loaded] == [EQUATION_ID]
    assert rows[0]["notes"] == (
        "replace-round5; feature-source-family-kill; query-audit; research-only"
    )
