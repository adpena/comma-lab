# SPDX-License-Identifier: MIT
"""Checks for the Round-2 grokking/undertraining disambiguation law."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.grokking_ridge_undertraining_disambiguation_20260713 import (
    EQUATION_ID,
    build_grokking_ridge_undertraining_disambiguation_v1,
    derive_fixed_quadratic_delay_certificate,
    populate_grokking_ridge_undertraining_disambiguation_v1,
)
from tac.canonical_equations.registry import query_equations


def test_zero_initial_null_component_eliminates_the_slow_mode() -> None:
    result = derive_fixed_quadratic_delay_certificate(
        learning_rate=0.2,
        ridge_lambda=3.0,
        contraction_gamma=1.0 / 3.0,
        steps=15,
        initial_null_norm=0.0,
        terminal_gradient_norm=9e-15,
        strong_curvature_mu=3.0,
    )

    assert result["null_retention_factor"] == pytest.approx(0.4**15)
    assert result["terminal_null_norm"] == 0.0
    assert result["paper_slow_component_present"] is False
    assert result["global_contraction_factor"] == pytest.approx((1.0 / 3.0) ** 15)
    assert result["terminal_parameter_error_bound"] == pytest.approx(3e-15)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("learning_rate", 0.0, "learning rate"),
        ("ridge_lambda", -1.0, "ridge lambda"),
        ("contraction_gamma", 1.0, "gamma"),
        ("steps", -1, "steps"),
        ("steps", True, "steps"),
        ("initial_null_norm", -1.0, "norms"),
        ("terminal_gradient_norm", float("nan"), "finite"),
        ("strong_curvature_mu", 0.0, "curvature"),
    ],
)
def test_invalid_delay_certificate_inputs_fail_closed(
    field: str, value: object, message: str
) -> None:
    kwargs: dict[str, object] = {
        "learning_rate": 0.2,
        "ridge_lambda": 3.0,
        "contraction_gamma": 1.0 / 3.0,
        "steps": 15,
        "initial_null_norm": 0.0,
        "terminal_gradient_norm": 9e-15,
        "strong_curvature_mu": 3.0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        derive_fixed_quadratic_delay_certificate(**kwargs)  # type: ignore[arg-type]


def test_anchor_scopes_feature_poverty_and_refuses_witness_transfer() -> None:
    equation = build_grokking_ridge_undertraining_disambiguation_v1()

    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.inputs["feature_dimension_m"] == 31
    assert anchor.inputs["scalar_training_rows_n"] == 1_474_560
    assert anchor.empirical_output["gd15_reproduced_committed_weights_bitwise"] is True
    assert anchor.empirical_output["gd15_objective_gap"] == 0.0
    assert anchor.empirical_output["gd150_minus_gd15_heldout_cosine"] == pytest.approx(
        4.627647760226061e-11
    )
    assert anchor.empirical_output["best_exact_ridge_ladder_cosine"] == pytest.approx(
        0.007690592649965529
    )
    assert anchor.empirical_output["synthetic_data_used"] is False
    assert anchor.empirical_output["verdict"] == (
        "FEATURE_POVERTY_FORMULATION_NOT_UNDERTRAINED"
    )
    assert equation.domain_of_validity["paper_equation_7_applicability_to_anchor"].startswith(
        "REFUSED"
    )
    assert any(
        "stage-advance" in exclusion
        for exclusion in equation.domain_of_validity["excluded"]
    )
    assert equation.provenance.score_claim_valid is False
    assert equation.provenance.promotion_eligible is False


def test_population_uses_the_locked_registry_helper(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_grokking_ridge_undertraining_disambiguation_v1(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="grokking_ridge_reader",
    )

    rows = [json.loads(line) for line in registry.read_text().splitlines() if line]
    loaded = query_equations(path=registry)
    assert populated.equation_id == EQUATION_ID
    assert [row.equation_id for row in loaded] == [EQUATION_ID]
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["agent"] == "pytest"
    assert rows[0]["subagent_id"] == "grokking_ridge_reader"
    assert not list(tmp_path.glob("*.tmp.*"))

