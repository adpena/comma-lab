# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.registry import query_equations
from tac.canonical_equations.replace_round4_support_ranking_20260713 import (
    EQUATION_ID,
    build_replace_round4_support_ranking_v1,
    conditional_sparse_teacher_economics,
    populate_replace_round4_support_ranking_v1,
    support_retention_law,
)


def test_support_retention_law_matches_receipt() -> None:
    result = support_retention_law(
        retained_l2_square_mass=0.00025184753555350115,
        total_l2_square_mass=0.001248472641573917,
        selected_cells=277_320,
        total_cells=5_898_240,
    )
    assert result["area_fraction"] == pytest.approx(0.047017415364583336)
    assert result["retained_mass_fraction"] == pytest.approx(0.20172451295048283)
    assert result["uplift_over_uniform_area"] == pytest.approx(4.290421142597201)
    assert result["conditional_masked_exact_costate_cosine"] == pytest.approx(
        0.44913752120089323
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "retained_l2_square_mass": -1.0,
            "total_l2_square_mass": 1.0,
            "selected_cells": 1,
            "total_cells": 2,
        },
        {
            "retained_l2_square_mass": 1.0,
            "total_l2_square_mass": 0.0,
            "selected_cells": 1,
            "total_cells": 2,
        },
        {
            "retained_l2_square_mass": 0.5,
            "total_l2_square_mass": 1.0,
            "selected_cells": 3,
            "total_cells": 2,
        },
    ],
)
def test_support_retention_law_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        support_retention_law(**kwargs)  # type: ignore[arg-type]


def test_conditional_economics_matches_sealed_policy_without_wall_claim() -> None:
    result = conditional_sparse_teacher_economics(
        prefix_fraction=0.005714118050141177,
        selected_area_fraction=0.047017415364583336,
    )
    assert result["conditional_composed_label_coefficient"] == pytest.approx(
        0.05246287035291876
    )
    assert result["conditional_variable_cost_reduction_x"] == pytest.approx(
        19.061099655298698
    )


def test_equation_preserves_exact_optimum_and_family_scope() -> None:
    equation = build_replace_round4_support_ranking_v1()
    anchor = equation.empirical_anchors[0]
    assert equation.equation_id == EQUATION_ID
    assert anchor.empirical_output["winning_rung"] == "pairwise-rank-pair-block-44"
    assert anchor.empirical_output["winning_retained_mass_fraction"] == pytest.approx(
        0.20172451295048283
    )
    assert anchor.empirical_output["winning_heldout_ece"] < 0.05
    assert anchor.empirical_output["conditional_wall_clock_claim"] is False
    assert equation.domain_of_validity["scope_level"] == "family x fixed replay"
    assert any("nonlinear" in row for row in equation.domain_of_validity["excluded"])
    assert equation.provenance.score_claim_valid is False
    assert equation.provenance.promotion_eligible is False


def test_population_round_trips_through_isolated_locked_registry(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_replace_round4_support_ranking_v1(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="replace_round4_ranking",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line]
    loaded = query_equations(path=registry)
    assert populated.equation_id == EQUATION_ID
    assert [row.equation_id for row in loaded] == [EQUATION_ID]
    assert rows[0]["notes"] == (
        "replace-round4; family-scoped-no-go; support-ranking; research-only"
    )
