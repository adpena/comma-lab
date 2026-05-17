# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.master_gradient_feasibility import audit_master_gradient_probe_plan


def test_raw_archive_byte_gradient_is_blocked_for_zip_entropy_packets() -> None:
    verdict = audit_master_gradient_probe_plan(
        mutation_grain="raw_archive_byte",
        axis_label="contest_cpu",
    )

    assert verdict.raw_byte_gradient_valid is False
    assert verdict.operator_response_valid is False
    assert verdict.ready_for_operator_probe is False
    assert verdict.ready_for_provider_dispatch is False
    assert verdict.score_claim is False
    assert verdict.promotion_eligible is False
    assert verdict.ready_for_exact_eval_dispatch is False
    assert verdict.verdict == "blocked_raw_archive_gradient"
    assert "raw_archive_byte_flip_breaks_zip_or_central_directory_grammar" in verdict.blockers
    assert "raw_archive_byte_score_derivative_not_well_defined_for_discrete_packet" not in (
        verdict.blockers
    )
    assert "raw_byte_score_derivative_not_well_defined_for_discrete_packet" in verdict.blockers
    assert verdict.recommended_probe_grain == "grammar_aware_operator"


def test_zip_member_payload_byte_is_blocked_when_inner_stream_is_entropy_coded() -> None:
    verdict = audit_master_gradient_probe_plan(
        mutation_grain="zip_member_payload_byte",
        axis_label="paired_contest_cpu_cuda",
        updates_zip_headers=True,
        updates_crc=True,
        repacks_archive=True,
        proves_inflate_success=True,
    )

    assert verdict.operator_response_valid is False
    assert verdict.ready_for_operator_probe is False
    assert verdict.verdict == "blocked_entropy_stream_payload_gradient"
    assert "zip_member_payload_byte_flip_is_not_local_in_entropy_coded_stream" in (
        verdict.blockers
    )
    assert verdict.recommended_probe_grain == "logical_section_parameter"


def test_grammar_aware_operator_probe_can_run_but_cannot_self_promote() -> None:
    verdict = audit_master_gradient_probe_plan(
        mutation_grain="grammar_aware_operator",
        axis_label="paired_contest_cpu_cuda",
        updates_zip_headers=True,
        updates_crc=True,
        repacks_archive=True,
        proves_inflate_success=True,
    )

    assert verdict.raw_byte_gradient_valid is False
    assert verdict.operator_response_valid is True
    assert verdict.ready_for_operator_probe is True
    assert verdict.ready_for_provider_dispatch is False
    assert verdict.score_claim is False
    assert verdict.promotion_eligible is False
    assert verdict.rank_or_kill_eligible is False
    assert verdict.ready_for_exact_eval_dispatch is False
    assert verdict.blockers == ()
    assert verdict.verdict == "operator_response_probe_ready"


def test_operator_probe_requires_axis_and_packet_proofs() -> None:
    verdict = audit_master_gradient_probe_plan(mutation_grain="logical_section_parameter")

    assert verdict.operator_response_valid is False
    assert verdict.ready_for_operator_probe is False
    assert verdict.verdict == "blocked_until_packet_proofs_land"
    assert "missing_axis_label" in verdict.blockers
    assert "archive_not_repacked_after_mutation" in verdict.blockers
    assert "zip_headers_not_rebuilt" in verdict.blockers
    assert "zip_crc_not_rebuilt" in verdict.blockers
    assert "inflate_success_not_proven" in verdict.blockers
