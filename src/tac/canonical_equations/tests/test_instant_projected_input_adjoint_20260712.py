# SPDX-License-Identifier: MIT
"""Triality checks for the isolated INSTANT projected-input-adjoint law."""

from __future__ import annotations

import json

from tac.canonical_equations.instant_projected_input_adjoint_20260712 import (
    INSTANT_PROJECTED_INPUT_ADJOINT_EQUATION_ID,
    build_instant_projected_input_adjoint_v1,
    populate_instant_projected_input_adjoint_v1,
)


def test_instant_equation_pins_formula_citation_and_admission_boundary() -> None:
    equation = build_instant_projected_input_adjoint_v1()
    assert equation.equation_id == INSTANT_PROJECTED_INPUT_ADJOINT_EQUATION_ID
    assert "OpenReview:P2q6Y7UweV" in equation.domain_of_validity["citation"]
    assert "no arXiv identifier or DOI found" in equation.domain_of_validity["citation"]
    assert "oversampling constant 5" in equation.domain_of_validity["calibration"]
    assert "every registered" in equation.domain_of_validity["admission"]
    assert "universal cosine threshold" in equation.domain_of_validity["excluded"][1]
    assert "full_teacher" in equation.domain_of_validity["fallback"]
    assert "t_{validate}" in equation.latex_form
    assert "exact-teacher validation" in equation.units_in["t_validate"]
    assert equation.units_out["S_K"] == "dimensionless_equal_refresh_speedup"
    assert equation.provenance.score_claim_valid is False
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["common_admitted_targets"] == []
    assert anchor.empirical_output["admitted_regime_arms"] == []
    assert anchor.empirical_output["maximum_optimistic_charged_cycle_ratio"] < 1.0
    assert anchor.empirical_output["all_nine_arms_decisive_economic_no_go"] is True
    assert anchor.empirical_output["verdict"] == "NO_GO"
    assert anchor.noise_floor is None
    assert "median-minus-MAD" in anchor.noise_floor_provenance


def test_instant_equation_population_uses_append_only_registry(tmp_path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    populated = populate_instant_projected_input_adjoint_v1(
        path=registry,
        lock_path=tmp_path / "canonical_equations.jsonl.lock",
        agent="codex",
        subagent_id="task449_instant",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    assert populated.equation_id == INSTANT_PROJECTED_INPUT_ADJOINT_EQUATION_ID
    assert len(rows) == 1
    assert rows[0]["event_type"] == "registered"
    assert rows[0]["equation_id"] == INSTANT_PROJECTED_INPUT_ADJOINT_EQUATION_ID
