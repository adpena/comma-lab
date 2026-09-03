"""Executable and registry-shape tests for the LV3 current-arc law wave."""
from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_lv3_current_arc_laws_20260901 import (
    AFFINE_ID,
    ALL_LV3_CURRENT_ARC_BUILDERS,
    BHW_ID,
    CEILING_ID,
    CROSS_ID,
    GENERATOR_ID,
    REORDER_ID,
    SHARP_ID,
)
from tac.canonical_equations.evaluators import (
    EvaluatorError,
    populate_lawref_evaluators,
    resolve_equation_value,
)

EXPECTED_IDS = {
    SHARP_ID,
    CROSS_ID,
    AFFINE_ID,
    BHW_ID,
    REORDER_ID,
    GENERATOR_ID,
    CEILING_ID,
}


def test_builders_are_distinct_anchored_advisory_laws() -> None:
    equations = [builder() for builder in ALL_LV3_CURRENT_ARC_BUILDERS]
    assert {equation.equation_id for equation in equations} == EXPECTED_IDS
    for equation in equations:
        assert equation.empirical_anchors
        assert equation.domain_of_validity["score_claim"] is False
        assert equation.domain_of_validity["promotion_eligible"] is False


def test_lawref_evaluators_are_registered_and_executable() -> None:
    registered = set(populate_lawref_evaluators())
    assert registered >= EXPECTED_IDS
    assert resolve_equation_value(
        AFFINE_ID,
        {
            "intercept_argmax_errors": 17_241,
            "marginal_argmax_errors_per_token_error": 1.1435,
            "token_errors": 100,
        },
    ) == pytest.approx(17_355.35)
    assert resolve_equation_value(
        CEILING_ID, {"sampled_gain_bits": 3.322, "sampled_fraction": 0.2}
    ) == pytest.approx(2.07625)


def test_cross_counts_only_joint_predicates() -> None:
    count = resolve_equation_value(
        CROSS_ID,
        {
            "byte_feasible": [True, True, False, False],
            "distortion_feasible": [False, False, True, True],
        },
    )
    assert count == 0
    with pytest.raises(EvaluatorError):
        resolve_equation_value(
            CROSS_ID,
            {"byte_feasible": [True], "distortion_feasible": [True, False]},
        )


def test_bhw_requires_changed_positions_and_matches_fcd1_semantics() -> None:
    row = resolve_equation_value(
        BHW_ID,
        {
            "before_labels": [0, 1, 2],
            "after_labels": [1, 2, 3],
            "ground_truth_labels": [1, 1, 4],
        },
    )
    assert row == {"benefit": 1, "harm": 1, "wash": 1}
    with pytest.raises(EvaluatorError):
        resolve_equation_value(
            BHW_ID,
            {"before_labels": [1], "after_labels": [1], "ground_truth_labels": [1]},
        )


def test_reorder_and_generator_caveats_are_in_the_value() -> None:
    assert resolve_equation_value(
        REORDER_ID,
        {
            "has_context_model": False,
            "generic_coder_savings_bytes": 70_552,
            "context_model_savings_bytes": 0,
        },
    ) == 70_552
    assert resolve_equation_value(
        REORDER_ID,
        {
            "has_context_model": True,
            "generic_coder_savings_bytes": 70_552,
            "context_model_savings_bytes": 0,
        },
    ) == 0
    result = resolve_equation_value(
        GENERATOR_ID,
        {
            "reference_bytes": 103_681,
            "generator_bytes": 47_603,
            "fit_error_fraction": 0.0112324,
        },
    )
    assert result["byte_ratio"] == pytest.approx(103_681 / 47_603)
    assert result["transferable_as_lossless_credit"] is False


def test_sharp_optimum_is_basin_local() -> None:
    assert resolve_equation_value(
        SHARP_ID, {"objective_deltas": [0.0, 0.001, 0.01]}
    ) == 0.0
    with pytest.raises(EvaluatorError):
        resolve_equation_value(SHARP_ID, {"objective_deltas": []})
