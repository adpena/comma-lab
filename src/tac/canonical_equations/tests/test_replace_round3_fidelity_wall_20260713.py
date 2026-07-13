# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.registry import query_equations
from tac.canonical_equations.replace_round3_fidelity_wall_20260713 import (
    EQUATION_ID,
    build_replace_round3_fidelity_wall_v1,
    conditional_masked_costate_cosine,
    exact_teacher_call_economics,
    populate_replace_round3_fidelity_wall_v1,
    prefix_conv_compute_fraction,
)


def test_masked_costate_cosine_is_square_root_of_retained_energy() -> None:
    assert conditional_masked_costate_cosine(retained_l2_square_fraction=0.0) == 0.0
    assert conditional_masked_costate_cosine(retained_l2_square_fraction=0.25) == 0.5
    assert conditional_masked_costate_cosine(retained_l2_square_fraction=1.0) == 1.0


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_masked_costate_cosine_rejects_values_outside_domain(value: object) -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        conditional_masked_costate_cosine(  # type: ignore[arg-type]
            retained_l2_square_fraction=value
        )


def test_teacher_call_economics_separates_labels_validation_and_retries() -> None:
    clean = exact_teacher_call_economics(
        training_label_calls=480,
        validation_calls=120,
        effective_cached_label_uses=7200,
    )
    campaign = exact_teacher_call_economics(
        training_label_calls=626,
        validation_calls=120,
        effective_cached_label_uses=7200,
    )
    assert clean["label_only_amortization_x"] == 15.0
    assert clean["inclusive_amortization_x"] == 12.0
    assert campaign["label_only_amortization_x"] == pytest.approx(11.501597444089457)
    assert campaign["inclusive_amortization_x"] == pytest.approx(9.651474530831099)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_label_calls", 0),
        ("training_label_calls", -1),
        ("validation_calls", True),
        ("effective_cached_label_uses", 1.5),
    ],
)
def test_teacher_call_economics_fails_closed(field: str, value: object) -> None:
    values: dict[str, object] = {
        "training_label_calls": 1,
        "validation_calls": 1,
        "effective_cached_label_uses": 2,
    }
    values[field] = value
    with pytest.raises(ValueError):
        exact_teacher_call_economics(**values)  # type: ignore[arg-type]


def test_prefix_compute_fraction_cancels_common_input_vjp_factor() -> None:
    result = prefix_conv_compute_fraction(
        prefix_forward_macs=56_623_104,
        full_forward_macs=9_909_333_952,
    )
    assert result["prefix_fraction_of_full_teacher_conv_flops"] == pytest.approx(
        0.005714118050141177
    )
    assert result["conv_only_ideal_speedup_x"] == pytest.approx(175.0051348650897)


def test_measured_equation_preserves_scoped_negative_and_campaign_accounting() -> None:
    equation = build_replace_round3_fidelity_wall_v1()
    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["verdict"] == "NO_GO_REGISTERED_ROUND3_RUNGS"
    assert anchor.empirical_output["winning_rung"] == "pre_se_prefix_rff"
    assert anchor.empirical_output["rff_heldout_costate_cosine"] == pytest.approx(
        0.0016791964165317613
    )
    assert anchor.empirical_output["campaign_conservative_all_starts"] == 746
    assert anchor.empirical_output["oracle_retained_l2_square_fraction"] > 0.47
    assert equation.predicted_vs_empirical_residual[
        "winning_direction_cosine_shortfall"
    ] == pytest.approx(0.06911047291090586)
    assert equation.domain_of_validity["scope_level"] == "formulation x instance"
    assert equation.domain_of_validity["research_only"] is True
    assert any("deeper frozen" in row for row in equation.domain_of_validity["excluded"])
    assert any("wall-speed" in row for row in equation.domain_of_validity["excluded"])
    assert equation.provenance.score_claim_valid is False
    assert equation.provenance.promotion_eligible is False


def test_population_round_trips_only_through_isolated_locked_registry(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_replace_round3_fidelity_wall_v1(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="replace_round3_fidelity",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line]
    loaded = query_equations(path=registry)
    assert populated.equation_id == EQUATION_ID
    assert [row.equation_id for row in loaded] == [EQUATION_ID]
    assert rows[0]["notes"] == (
        "replace-round3; scoped-no-go; frozen-prefix-rff; research-only"
    )
    assert not list(tmp_path.glob("*.tmp.*"))

