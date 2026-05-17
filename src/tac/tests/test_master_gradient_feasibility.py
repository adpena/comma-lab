# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.master_gradient_feasibility import (
    audit_master_gradient_anchor_authority,
    audit_master_gradient_probe_plan,
)


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


def test_raw_archive_byte_remains_blocked_even_if_container_proofs_are_claimed() -> None:
    verdict = audit_master_gradient_probe_plan(
        mutation_grain="raw_archive_byte",
        axis_label="contest_cpu",
        updates_zip_headers=True,
        updates_crc=True,
        repacks_archive=True,
        proves_inflate_success=True,
    )

    assert verdict.raw_byte_gradient_valid is False
    assert verdict.operator_response_valid is False
    assert verdict.ready_for_operator_probe is False
    assert verdict.verdict == "blocked_raw_archive_gradient"
    assert "raw_byte_score_derivative_not_well_defined_for_discrete_packet" in verdict.blockers
    assert "raw_archive_byte_flip_is_not_local_in_entropy_coded_payload" in verdict.blockers


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


def test_anchor_authority_blocks_raw_archive_byte_subset_full_axis() -> None:
    verdict = audit_master_gradient_anchor_authority(
        coordinate_system="raw_archive_byte",
        axis_label="contest_cpu",
        n_pairs_used=8,
        n_pairs_total=600,
        rate_column_semantics="uniform_per_archive_byte",
        projection_domain="byte_value_delta",
    )

    assert verdict.anchor_valid is False
    assert verdict.ready_for_operator_probe is False
    assert verdict.ready_for_provider_dispatch is False
    assert verdict.score_claim is False
    assert verdict.rank_or_kill_eligible is False
    assert verdict.verdict == "blocked_anchor_false_authority"
    assert "raw_archive_byte_coordinate_system_not_packet_valid" in verdict.blockers
    assert "subset_gradient_cannot_be_labeled_full_contest_axis" in verdict.blockers
    assert (
        "rate_column_confuses_byte_value_delta_with_archive_byte_count_delta"
        in verdict.blockers
    )


def test_anchor_authority_allows_packet_valid_operator_response_probe() -> None:
    verdict = audit_master_gradient_anchor_authority(
        coordinate_system="grammar_aware_operator_response",
        axis_label="paired_contest_cpu_cuda",
        n_pairs_used=600,
        n_pairs_total=600,
        rate_column_semantics="score_response_measured",
        projection_domain="operator_delta",
    )

    assert verdict.anchor_valid is True
    assert verdict.ready_for_operator_probe is True
    assert verdict.ready_for_provider_dispatch is False
    assert verdict.score_claim is False
    assert verdict.promotion_eligible is False
    assert verdict.rank_or_kill_eligible is False
    assert verdict.ready_for_exact_eval_dispatch is False
    assert verdict.blockers == ()
    assert verdict.verdict == "anchor_ready_for_operator_probe"


def test_anchor_authority_subset_gradient_must_use_diagnostic_axis() -> None:
    full_axis = audit_master_gradient_anchor_authority(
        coordinate_system="codec_symbol_response",
        axis_label="contest_cuda",
        n_pairs_used=16,
        n_pairs_total=600,
        rate_column_semantics="byte_count_delta",
        projection_domain="byte_count_delta",
    )
    diagnostic = audit_master_gradient_anchor_authority(
        coordinate_system="codec_symbol_response",
        axis_label="diagnostic",
        n_pairs_used=16,
        n_pairs_total=600,
        rate_column_semantics="byte_count_delta",
        projection_domain="byte_count_delta",
    )

    assert full_axis.anchor_valid is False
    assert "subset_gradient_cannot_be_labeled_full_contest_axis" in full_axis.blockers
    assert diagnostic.anchor_valid is True
    assert diagnostic.blockers == ()
