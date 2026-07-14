# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

from tac.canonical_equations.costate_router_stability_20260714 import (
    EQUATION_ID,
    build_costate_router_stability_v1,
    populate_costate_router_stability_v1,
    self_normalized_ratio_weights,
    sequential_beta_route_match_posterior,
    signed_selection_margin,
)
from tac.canonical_equations.registry import query_equations


def test_margin_and_self_normalized_ratio_law() -> None:
    assert signed_selection_margin(2.0, 1.0) == pytest.approx(1.0)
    assert signed_selection_margin(1.0, 1.0) == 0.0
    weights = self_normalized_ratio_weights([1.5, 0.5, 0.5], [True, True, False], 0.75, 1.25)
    assert sum(weights) == pytest.approx(2.0)
    assert weights[0] > weights[1]
    assert weights[2] == 0.0
    posterior = sequential_beta_route_match_posterior([True, False, True])
    expected = ((2.0, 1.0, 2.0 / 3.0), (2.0, 2.0, 0.5), (3.0, 2.0, 0.6))
    for measured, wanted in zip(posterior, expected, strict=True):
        assert measured == pytest.approx(wanted)


def test_equation_records_real_ties_replay_and_distribution_blocker() -> None:
    equation = build_costate_router_stability_v1()
    anchor = equation.empirical_anchors[0]
    assert equation.equation_id == EQUATION_ID
    assert anchor.empirical_output["exact_zero_margin_folds"] == [75.0, 125.0]
    assert anchor.empirical_output["is_status"] == "BLOCKED_DISTRIBUTION_CUSTODY"
    assert anchor.empirical_output["router_learning_frozen"] is False
    calibration = anchor.empirical_output["forecast_calibration"]
    assert calibration["status"] == "MIS_CALIBRATED_INSTANCE"
    assert calibration["terminal_posterior_match_probability"] == pytest.approx(2.0 / 3.0)
    assert "a_{\\mathrm{apply}}" in equation.latex_form
    assert equation.provenance.score_claim_valid is False


def test_equation_helpers_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        signed_selection_margin(float("nan"), 1.0)
    with pytest.raises(ValueError, match="equally sized"):
        self_normalized_ratio_weights([1.0], [True, False], 0.5, 2.0)
    with pytest.raises(ValueError, match="no positive"):
        self_normalized_ratio_weights([1.0], [False], 0.5, 2.0)
    with pytest.raises(ValueError, match="Beta prior"):
        sequential_beta_route_match_posterior([True], prior_alpha=0.0)


def test_population_uses_temporary_registry_only(tmp_path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    equation = populate_costate_router_stability_v1(
        path=registry,
        lock_path=tmp_path / "registry.lock",
        agent="pytest",
        subagent_id="router-stability",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line]
    assert equation.equation_id == EQUATION_ID
    assert [row.equation_id for row in query_equations(path=registry)] == [EQUATION_ID]
    assert rows[0]["notes"].startswith("costate-router-stability")
