# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_pointfree_program_description_20260723 import (
    EQUATION_ID,
    RECEIPT,
    RECEIPT_SHA256,
    build_ddm_pf1_two_typed_discrete_skeleton_rate_delta_v1,
    discrete_skeleton_formulation_closed,
    evaluate_two_typed_program_vs_flat_bytes,
)

REPO = Path(__file__).resolve().parents[1]


def test_program_rate_evaluator_admits_only_exact_negative_delta() -> None:
    win = evaluate_two_typed_program_vs_flat_bytes(
        program_skeleton_counted_bytes=545,
        program_fiber_counted_bytes=142_253,
        flat_skeleton_counted_bytes=2_886,
        flat_fiber_counted_bytes=142_265,
        semantic_parseback_exact=True,
        same_description_content=True,
    )
    assert win.delta_skeleton_bytes == -2_341
    assert win.delta_fiber_bytes == -12
    assert win.delta_program_minus_flat_bytes == -2_353
    assert win.program_wins is True
    assert win.disposition == "ADMIT_SUBSTITUTIVE_PROGRAM"
    assert win.opaque_native_fibers_counted_separately is True

    loss = evaluate_two_typed_program_vs_flat_bytes(
        program_skeleton_counted_bytes=100,
        program_fiber_counted_bytes=100,
        flat_skeleton_counted_bytes=90,
        flat_fiber_counted_bytes=100,
        semantic_parseback_exact=True,
        same_description_content=True,
    )
    assert loss.delta_program_minus_flat_bytes == 10
    assert loss.program_wins is False
    assert loss.disposition == "KEEP_FLAT_CONTROL"


@pytest.mark.parametrize(
    ("semantic_parseback_exact", "same_description_content"),
    [(False, True), (True, False), (False, False)],
)
def test_program_rate_evaluator_refuses_invalid_comparisons(
    semantic_parseback_exact: bool,
    same_description_content: bool,
) -> None:
    with pytest.raises(ValueError, match="exact same-description replay"):
        evaluate_two_typed_program_vs_flat_bytes(
            program_skeleton_counted_bytes=90,
            program_fiber_counted_bytes=100,
            flat_skeleton_counted_bytes=100,
            flat_fiber_counted_bytes=100,
            semantic_parseback_exact=semantic_parseback_exact,
            same_description_content=same_description_content,
        )


def test_formulation_closure_requires_three_scope_eligible_formulations() -> None:
    assert not discrete_skeleton_formulation_closed({"STRUCTURAL": -30})
    assert not discrete_skeleton_formulation_closed({})
    with pytest.raises(ValueError, match="not scope-eligible"):
        discrete_skeleton_formulation_closed(
            {"LITERAL": 7, "SHARED_LIBRARY": 10, "STRUCTURAL": 1}
        )
    with pytest.raises(ValueError, match="not scope-eligible"):
        discrete_skeleton_formulation_closed({"FIBER_TOKENS": -18})


def test_canonical_equation_binds_receipt_and_authority_firewall() -> None:
    assert hashlib.sha256((REPO / RECEIPT).read_bytes()).hexdigest() == RECEIPT_SHA256
    equation = build_ddm_pf1_two_typed_discrete_skeleton_rate_delta_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.provenance.promotion_eligible is False
    assert equation.provenance.score_claim_valid is False
    assert equation.empirical_anchors[0].residual == 0.0
    output = equation.empirical_anchors[0].empirical_output
    assert output["structural_two_typed_split"] == {
        "composed_stratum": {
            "delta_skeleton_bytes": -2_328,
            "delta_fiber_bytes": -12,
            "delta_total_bytes": -2_340,
        },
        "composed_typed": {
            "delta_skeleton_bytes": -2_341,
            "delta_fiber_bytes": -12,
            "delta_total_bytes": -2_353,
        },
        "dv2_stratum_sentence": {
            "delta_skeleton_bytes": -2_299,
            "delta_fiber_bytes": 0,
            "delta_total_bytes": -2_299,
        },
        "dv2_typed_sentence": {
            "delta_skeleton_bytes": -2_283,
            "delta_fiber_bytes": 0,
            "delta_total_bytes": -2_283,
        },
        "g1_worldsheet": {
            "delta_skeleton_bytes": -30,
            "delta_fiber_bytes": 0,
            "delta_total_bytes": -30,
        },
        "v15_template_bank": {
            "delta_skeleton_bytes": -14,
            "delta_fiber_bytes": -12,
            "delta_total_bytes": -26,
        },
    }
    assert not any(output["formulation_closed"].values())
    assert output["pointer_moved"] is False
    assert output["score_claim"] is False
