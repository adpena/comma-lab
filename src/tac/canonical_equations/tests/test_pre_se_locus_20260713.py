# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.pre_se_locus_20260713 import (
    EQUATION_ID,
    build_pre_se_locus_tileability_and_localization_v1,
    populate_pre_se_locus_tileability_and_localization_v1,
    post_se_sparse_teacher_economics,
    strict_tileability_from_global_dependencies,
)
from tac.canonical_equations.registry import query_equations


def test_pre_own_se_is_not_sufficient_for_strict_tileability() -> None:
    block2 = strict_tileability_from_global_dependencies(
        upstream_global_reductions=4, own_global_reduction_applied=False
    )
    block3 = strict_tileability_from_global_dependencies(
        upstream_global_reductions=7, own_global_reduction_applied=False
    )
    local = strict_tileability_from_global_dependencies(
        upstream_global_reductions=0, own_global_reduction_applied=False
    )
    assert block2["strict_end_to_end_independently_tileable_from_rgb"] is False
    assert block3["strict_end_to_end_independently_tileable_from_rgb"] is False
    assert local["strict_end_to_end_independently_tileable_from_rgb"] is True


@pytest.mark.parametrize("value", (-1, True, 1.5))
def test_tileability_law_rejects_invalid_dependency_counts(value: object) -> None:
    with pytest.raises(ValueError):
        strict_tileability_from_global_dependencies(
            upstream_global_reductions=value,  # type: ignore[arg-type]
            own_global_reduction_applied=False,
        )


def test_pre_se_cost_reuses_round5_cost_composition() -> None:
    result = post_se_sparse_teacher_economics(
        feature_cut_fraction=0.0670083252029248,
        selected_area_fraction=0.047017415364583336,
        anchor_calls=600,
    )
    assert result["conditional_c_label"] == pytest.approx(0.11087518230855714)
    assert result["conditional_variable_cost_reduction_x"] == pytest.approx(
        9.019150897241158
    )


def test_equation_preserves_measured_scope_and_values() -> None:
    equation = build_pre_se_locus_tileability_and_localization_v1()
    anchor = equation.empirical_anchors[0]
    assert equation.equation_id == EQUATION_ID
    assert anchor.empirical_output["verdict"] == "WIDER-FAMILY-KILL"
    assert anchor.empirical_output["block2_nonlinear_retained_mass_fraction"] == pytest.approx(
        0.2736871496424692
    )
    assert anchor.empirical_output["block3_nonlinear_retained_mass_fraction"] == pytest.approx(
        0.31323809443347944
    )
    assert anchor.empirical_output["block2_strict_tileability"] is False
    assert anchor.empirical_output["block3_strict_tileability"] is False
    assert equation.provenance.score_claim_valid is False
    assert equation.provenance.promotion_eligible is False


def test_population_round_trips_through_isolated_locked_registry(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_pre_se_locus_tileability_and_localization_v1(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="pre_se_locus_builder",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line]
    loaded = query_equations(path=registry)
    assert populated.equation_id == EQUATION_ID
    assert [row.equation_id for row in loaded] == [EQUATION_ID]
    assert rows[0]["notes"] == (
        "pre-se-locus; strict-tileability; wider-family-kill; research-only"
    )
