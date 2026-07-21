# SPDX-License-Identifier: MIT
"""Triality tests for the G2b predict-project realization admission law."""

from __future__ import annotations

import pytest

from tac.canonical_equations.evaluators import get_evaluator
from tac.canonical_equations.predict_project_realization_admissibility_20260721 import (
    BLOCKER_ID,
    EQUATION_ID,
    build_predict_project_realization_admissibility_v1,
    predict_project_realization_certificate,
)


def _certificate(**overrides):
    values = {
        "pair_count": 600,
        "uint8_factor2_exact_fraction": 1.0,
        "double_decode_identical_pair_count": 600,
        "semantic_cells_to_rgb_exact_pair_count": 600,
        "pose_within_declared_tube_pair_count": 600,
        "additional_seed_bytes": 0,
        "receiver_derived_rgb": True,
    }
    values.update(overrides)
    return predict_project_realization_certificate(**values)


def test_complete_conjunction_admits_and_each_required_predicate_blocks() -> None:
    assert _certificate()["accepted"] is True
    for field, value in (
        ("pair_count", 601),
        ("uint8_factor2_exact_fraction", 0.999),
        ("double_decode_identical_pair_count", 599),
        ("semantic_cells_to_rgb_exact_pair_count", 599),
        ("pose_within_declared_tube_pair_count", 599),
        ("additional_seed_bytes", 1),
        ("receiver_derived_rgb", False),
    ):
        result = _certificate(**{field: value})
        assert result["accepted"] is False
        assert result["status"] == BLOCKER_ID


def test_measured_source_control_is_exact_but_not_admissible() -> None:
    result = _certificate(
        semantic_cells_to_rgb_exact_pair_count=0,
        additional_seed_bytes=707_788_800,
        receiver_derived_rgb=False,
    )
    assert result["predicates"]["factor2_uint8_exact"] is True
    assert result["predicates"]["double_decode_identical"] is True
    assert result["predicates"]["semantic_cells_to_rgb_exact"] is False
    assert result["predicates"]["zero_added_seed_bytes"] is False
    assert result["predicates"]["receiver_derived_rgb"] is False
    assert result["accepted"] is False


def test_lawref_evaluator_and_empirical_anchor_are_registered_in_code() -> None:
    evaluator = get_evaluator(EQUATION_ID)
    result = evaluator(
        {
            "pair_count": 600,
            "uint8_factor2_exact_fraction": 1.0,
            "double_decode_identical_pair_count": 600,
            "semantic_cells_to_rgb_exact_pair_count": 0,
            "pose_within_declared_tube_pair_count": 600,
            "additional_seed_bytes": 707_788_800,
            "receiver_derived_rgb": False,
        }
    )
    assert result["status"] == BLOCKER_ID

    equation = build_predict_project_realization_admissibility_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["blocker_id"] == BLOCKER_ID
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["accepted"] is False
    assert anchor.empirical_output["d_seg_description_vs_frozen_target"] > 0.34
    assert anchor.empirical_output["d_seg_realized_vs_frozen_target"] < 0.001

    assert len(equation.empirical_anchors) == 2
    interior = equation.empirical_anchors[1]
    assert interior.inputs["rung_id"] == "R2_MAX_MARGIN"
    assert interior.inputs["receiver_derived_rgb"] is True
    assert interior.inputs["additional_seed_bytes"] == 0
    assert interior.empirical_output["accepted"] is False
    assert interior.empirical_output["surviving_declared_writes"] == 114
    assert interior.empirical_output["pose_within_declared_tube_pair_count"] == 0
    assert set(interior.empirical_output["failed_predicates"]) == {
        "semantic_cells_to_rgb_exact",
        "pose_within_declared_tube",
    }
    assert "constant-tile interior formulations" in equation.domain_of_validity["verdict_scope"]
    assert "G2b/G2c successor admission" in equation.canonical_consumers


def test_admission_refuses_type_and_range_laundering() -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        _certificate(receiver_derived_rgb=1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _certificate(uint8_factor2_exact_fraction=float("nan"))
    with pytest.raises(ValueError, match="must not exceed"):
        _certificate(semantic_cells_to_rgb_exact_pair_count=601)
