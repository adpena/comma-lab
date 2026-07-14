from __future__ import annotations

from tac.canonical_equations.margin_adaptive_integer_waterfill_20260714 import (
    EQUATION_ID,
    build_margin_adaptive_integer_profile_waterfill_v1,
)


def test_margin_adaptive_equation_is_honest_and_registration_inert() -> None:
    equation = build_margin_adaptive_integer_profile_waterfill_v1()
    domain = equation.domain_of_validity

    assert equation.equation_id == EQUATION_ID
    assert equation.empirical_anchors == ()
    assert domain["research_only"] is True
    assert domain["unseen_input_ibp_claim"] is False
    assert domain["spatial_waterfill_native_execution_claim"] is False
    assert domain["score_claim"] is False
    assert domain["pointer_moved"] is False
    assert "n600" in str(domain["req_R"])
    assert "int64" in equation.latex_form
    assert equation.provenance.promotion_eligible is False
    assert equation.provenance.score_claim_valid is False
