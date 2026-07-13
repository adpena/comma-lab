from __future__ import annotations

from tac.canonical_equations.onpolicy_input_costate_surrogate_20260713 import (
    EQUATION_ID,
    build_onpolicy_input_costate_surrogate_v1,
)


def test_onpolicy_equation_is_honest_research_partial_measurement() -> None:
    equation = build_onpolicy_input_costate_surrogate_v1()
    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 4
    first_output = equation.empirical_anchors[0].empirical_output
    corrected_output = equation.empirical_anchors[1].empirical_output
    terminal_output = equation.empirical_anchors[2].empirical_output
    final_output = equation.empirical_anchors[3].empirical_output
    assert first_output["k4_timing_status"] == "diagnostic_only_not_isolated_forward_replacement_economics"
    assert first_output["deterministic_reproduction"].startswith("BLOCKED_")
    assert corrected_output["early_verdict"] == "NO_GO"
    assert corrected_output["common_admitted_formulation"] is False
    assert corrected_output["full_k20_fidelity"] == "UNKNOWN_NOT_MEASURED"
    assert corrected_output["receipt_sha256"] == equation.domain_of_validity["corrected_campaign_sha256"]
    assert terminal_output["verdict"] == "NO_GO"
    assert terminal_output["accepted_exact_prefix_valid"] is True
    assert terminal_output["exact_completion_certified"] is False
    assert terminal_output["completion_reclassification"].startswith("BLOCKED_")
    assert terminal_output["first_failing_step"] == 2
    assert terminal_output["receipt_sha256"] == equation.domain_of_validity["terminal_receipt_sha256"]
    assert final_output["campaign_verdict"] == "NO_GO"
    assert final_output["early_verdict"] == "NO_GO_EMA_NOT_ADMITTED"
    assert final_output["boundary_verdict"] == "NEEDS_MORE_EXACT_ANCHOR_ONLY"
    assert final_output["late_verdict"] == "NO_GO_DSEG_TRAJECTORY_DRIFT"
    assert final_output["full_k20_fidelity"].startswith("UNKNOWN_")
    assert final_output["receipt_sha256"] == equation.domain_of_validity["final_campaign_sha256"]
    assert equation.domain_of_validity["research_only"] is True
    assert equation.domain_of_validity["first_receipt_status"].startswith("MEASURED / raw NEEDS-MORE")
    full_build_blocker = equation.domain_of_validity["full_build_blocker"]
    assert "early EMA admission fails" in full_build_blocker
    assert "late non-anchor CE, d_pose, and d_seg" in full_build_blocker
    assert "boundary is exact-anchor-only NEEDS-MORE" in full_build_blocker
    assert "live-trainer" in full_build_blocker
    assert "full-K20" in full_build_blocker
    assert equation.domain_of_validity["review_status"] == "externally_tracked_by_content_hash"
    assert equation.provenance.score_claim_valid is False
