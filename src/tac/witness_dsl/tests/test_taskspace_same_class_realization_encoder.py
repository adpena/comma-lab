from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    SCORER_HEIGHT,
    SCORER_WIDTH,
)
from tac.witness_dsl.taskspace_same_class_realization_encoder import (
    N2_STAGE_ABLATION_PRIORITY_AXIS,
    N2_STAGE_ABLATION_PRIORITY_RECEIPT_SHA256,
    N2_STAGE_ABLATION_PRIORITY_SCOPE,
    EncoderOnlySameClassRealizationEvidenceV1,
    SameClassRealizationAcquisitionPlanV1,
    SameClassRealizationBaseInterpretationV1,
    SameClassRealizationEncoderError,
    SameClassRealizationG8CompileSeamV1,
    SameClassRealizationPrefixOrderV1,
    SameClassRealizationProposalFamilyV1,
    acquire_same_class_realization_repair_programs,
    parse_same_class_realization_proposal_receipt,
    validate_same_class_realization_program_against_evidence,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    PACKET_HEADER,
    RUN_ROW,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairRunV1,
    SameClassSemanticRoleV1,
)


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (1, SCORER_HEIGHT, SCORER_WIDTH)
    semantic = np.zeros(shape, dtype=np.uint8)
    target = np.zeros(shape, dtype=np.uint8)
    realized = np.zeros(shape, dtype=np.uint8)
    candidate_rgb = np.zeros((*shape, 3), dtype=np.uint8)
    target_rgb = np.zeros((*shape, 3), dtype=np.uint8)

    road_colours = (
        (10, 20, 30),
        (10, 20, 30),
        (200, 20, 30),
        (200, 20, 30),
    )
    for col, rgb in zip(range(10, 14), road_colours, strict=True):
        realized[0, 10, col] = 1
        candidate_rgb[0, 10, col] = (8, 18, 28)
        target_rgb[0, 10, col] = rgb

    for col, rgb in ((20, (5, 100, 5)), (22, (5, 110, 5))):
        semantic[0, 20, col] = 1
        target[0, 20, col] = 1
        realized[0, 20, col] = 0
        candidate_rgb[0, 20, col] = (5, 10, 5)
        target_rgb[0, 20, col] = rgb

    # Wrong semantic class: realized is also wrong, but this is not G8 debt.
    target[0, 30, 30] = 1
    realized[0, 30, 30] = 2
    target_rgb[0, 30, 30] = (40, 50, 60)

    # Semantic target is already realized: also not debt.
    semantic[0, 31, 31] = 2
    target[0, 31, 31] = 2
    realized[0, 31, 31] = 2
    target_rgb[0, 31, 31] = (60, 50, 40)
    candidate_rgb[0, 31, 31] = (60, 50, 40)

    # Fortunate semantic mismatch: semantic topology differs but H realizes T.
    semantic[0, 32, 32] = 3
    target[0, 32, 32] = 2
    realized[0, 32, 32] = 2
    return semantic, target, realized, candidate_rgb, target_rgb


def _evidence() -> EncoderOnlySameClassRealizationEvidenceV1:
    semantic, target, realized, candidate_rgb, target_rgb = _arrays()
    return EncoderOnlySameClassRealizationEvidenceV1(
        source_pair_ids=(17,),
        base_interpretation=SameClassRealizationBaseInterpretationV1.EXACT_SEMANTIC_G_CONTROL_V1,
        base_semantic_binding_sha256="1" * 64,
        base_p_section_sha256="6" * 64,
        base_p_section_bytes=101,
        base_g_section_sha256="7" * 64,
        base_g_section_bytes=202,
        base_camera_y1_sha256="8" * 64,
        frozen_scorer_sha256="2" * 64,
        candidate_forward_receipt_sha256="3" * 64,
        target_forward_receipt_sha256="4" * 64,
        priority_ordering_receipt_sha256=N2_STAGE_ABLATION_PRIORITY_RECEIPT_SHA256,
        priority_ordering_axis=N2_STAGE_ABLATION_PRIORITY_AXIS,
        priority_ordering_scope=N2_STAGE_ABLATION_PRIORITY_SCOPE,
        current_semantic_labels_sha256=_sha256(semantic),
        target_labels_sha256=_sha256(target),
        realized_labels_sha256=_sha256(realized),
        candidate_scorer_rgb_sha256=_sha256(candidate_rgb),
        target_scorer_rgb_sha256=_sha256(target_rgb),
        current_semantic_labels=semantic,
        target_labels=target,
        realized_labels=realized,
        candidate_scorer_rgb=candidate_rgb,
        target_scorer_rgb=target_rgb,
    )


def _plan() -> SameClassRealizationAcquisitionPlanV1:
    return SameClassRealizationAcquisitionPlanV1(
        palette_sizes_per_class=(2, 3),
        geometric_prefix_ratios=(2, 3),
    )


def _proposal(
    *,
    family: SameClassRealizationProposalFamilyV1,
    order: SameClassRealizationPrefixOrderV1,
    prefix_count: int,
    palette_bound: int | None,
):
    acquisition = acquire_same_class_realization_repair_programs(_evidence(), plan=_plan())
    return next(
        proposal
        for proposal in acquisition.proposals
        if proposal.receipt.family is family
        and proposal.receipt.prefix_order is order
        and proposal.receipt.covered_debt_cell_count == prefix_count
        and proposal.receipt.palette_bound_per_class == palette_bound
    )


def _expanded_rgb(proposal) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    result: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for run in proposal.program.runs:
        for col in range(run.scorer_col_start, run.scorer_col_stop):
            result[(run.source_pair_id, run.scorer_row, col)] = run.rgb_u8
    return result


def test_exact_debt_definition_and_all_families_are_concrete() -> None:
    evidence = _evidence()
    acquisition = acquire_same_class_realization_repair_programs(evidence, plan=_plan())
    partition = evidence.cell_partition_telemetry

    assert evidence.debt_count == 6
    assert evidence.debt_count_by_class == (4, 2, 0, 0, 0)
    assert np.array_equal(
        evidence.debt_mask,
        (evidence.current_semantic_labels == evidence.target_labels)
        & (evidence.realized_labels != evidence.target_labels),
    )
    assert partition.closed_cell_count == SCORER_HEIGHT * SCORER_WIDTH - 8
    assert partition.realization_debt_cell_count == 6
    assert partition.topology_debt_cell_count == 1
    assert partition.fortunate_semantic_mismatch_cell_count == 1
    assert partition.closed_count_by_target_class == (SCORER_HEIGHT * SCORER_WIDTH - 9, 0, 1, 0, 0)
    assert partition.realization_debt_count_by_target_class == (4, 2, 0, 0, 0)
    assert partition.topology_debt_count_by_target_class == (0, 1, 0, 0, 0)
    assert partition.fortunate_semantic_mismatch_count_by_target_class == (0, 0, 1, 0, 0)
    assert (
        len(
            {
                partition.closed_mask_sha256,
                partition.realization_debt_mask_sha256,
                partition.topology_debt_mask_sha256,
                partition.fortunate_semantic_mismatch_mask_sha256,
            }
        )
        == 4
    )
    assert acquisition.total_debt_cell_count == 6
    assert acquisition.total_debt_count_by_class == (4, 2, 0, 0, 0)
    assert acquisition.cell_partition_telemetry == partition
    assert acquisition.prefix_cell_counts == (1, 2, 3, 4, 6)
    assert len(acquisition.proposals) == 40
    assert {proposal.receipt.family for proposal in acquisition.proposals} == set(SameClassRealizationProposalFamilyV1)
    assert {proposal.receipt.prefix_order for proposal in acquisition.proposals} == set(
        SameClassRealizationPrefixOrderV1
    )
    for proposal in acquisition.proposals:
        assert proposal.receipt.cell_partition_telemetry == partition
        validate_same_class_realization_program_against_evidence(
            proposal.program,
            evidence=evidence,
        )


def test_wrong_class_admission_is_rejected() -> None:
    wrong_class = SameClassRealizationRepairProgramV1(
        runs=(
            SameClassRealizationRepairRunV1(
                source_pair_id=17,
                scorer_row=30,
                scorer_col_start=30,
                scorer_col_stop=31,
                semantic_class=0,
                semantic_role=SameClassSemanticRoleV1.ROAD,
                rgb_u8=(1, 2, 3),
            ),
        )
    )
    with pytest.raises(SameClassRealizationEncoderError, match="outside exact"):
        validate_same_class_realization_program_against_evidence(
            wrong_class,
            evidence=_evidence(),
        )


def test_dense_content_and_advisory_custody_mismatches_fail_closed() -> None:
    evidence = _evidence()
    with pytest.raises(SameClassRealizationEncoderError, match="target_labels content hash"):
        replace(evidence, target_labels_sha256="0" * 64)
    with pytest.raises(SameClassRealizationEncoderError, match="advisory ordering"):
        replace(evidence, priority_ordering_axis="[contest-CPU]")  # type: ignore[arg-type]
    with pytest.raises(SameClassRealizationEncoderError, match="advisory ordering"):
        replace(evidence, priority_ordering_receipt_sha256="5" * 64)
    with pytest.raises(SameClassRealizationEncoderError, match="contiguous"):
        replace(evidence, source_pair_ids=(17, 19))


def test_exact_g_and_pass_predictor_bases_are_separate_and_source_bound() -> None:
    exact = _evidence()
    passed = replace(
        exact,
        base_interpretation=(SameClassRealizationBaseInterpretationV1.PASS_PREDICTOR_SEMANTIC_TOPOLOGY_V1),
        base_semantic_binding_sha256="9" * 64,
        base_g_section_sha256=None,
        base_g_section_bytes=None,
        base_camera_y1_sha256="a" * 64,
    )

    exact_result = acquire_same_class_realization_repair_programs(exact, plan=_plan())
    pass_result = acquire_same_class_realization_repair_programs(passed, plan=_plan())
    assert exact_result.evidence_binding_sha256 != pass_result.evidence_binding_sha256
    assert exact_result.proposals[0].receipt.g8_compile_seam is (
        SameClassRealizationG8CompileSeamV1.EXISTING_COUNTED_SEMANTIC_G_V1
    )
    assert exact_result.proposals[0].receipt.base_interpretation_has_frozen_g8_compile_path is True
    assert pass_result.proposals[0].receipt.g8_compile_seam is (
        SameClassRealizationG8CompileSeamV1.PASS_BASE_REQUIRES_NONEMPTY_COUNTED_G8_BASE_ENVELOPE_V1
    )
    assert pass_result.proposals[0].receipt.base_interpretation_has_frozen_g8_compile_path is False
    assert pass_result.proposals[0].receipt.base_g_section_sha256 is None
    assert pass_result.proposals[0].receipt.no_fake_empty_g_section is True

    with pytest.raises(SameClassRealizationEncoderError, match="must not invent"):
        replace(passed, base_g_section_sha256=hashlib.sha256(b"").hexdigest(), base_g_section_bytes=0)
    with pytest.raises(SameClassRealizationEncoderError, match="exact integer"):
        replace(exact, base_g_section_bytes=0)


def test_class_shared_horizontal_runs_are_canonically_merged() -> None:
    proposal = _proposal(
        family=SameClassRealizationProposalFamilyV1.CLASS_SHARED_TARGET_MEDOID_V1,
        order=SameClassRealizationPrefixOrderV1.CANONICAL_ADDRESS_V1,
        prefix_count=6,
        palette_bound=1,
    )
    road_runs = [run for run in proposal.program.runs if run.semantic_class == 0]
    lane_runs = [run for run in proposal.program.runs if run.semantic_class == 1]

    assert len(road_runs) == 1
    assert (road_runs[0].scorer_col_start, road_runs[0].scorer_col_stop) == (10, 14)
    assert len(lane_runs) == 2
    assert proposal.receipt.run_count == 3
    assert proposal.receipt.estimated_standalone_g8_raw_bytes == (PACKET_HEADER.size + 3 * RUN_ROW.size)


def test_bounded_medoids_are_observed_target_colours_and_distinct() -> None:
    proposal = _proposal(
        family=SameClassRealizationProposalFamilyV1.CLASS_BOUNDED_TARGET_MEDOIDS_V1,
        order=SameClassRealizationPrefixOrderV1.CANONICAL_ADDRESS_V1,
        prefix_count=6,
        palette_bound=2,
    )
    evidence = _evidence()
    expanded = _expanded_rgb(proposal)
    road_rgb = {rgb for (pair_id, row, _col), rgb in expanded.items() if pair_id == 17 and row == 10}
    lane_rgb = {rgb for (pair_id, row, _col), rgb in expanded.items() if pair_id == 17 and row == 20}
    observed_road = {tuple(int(value) for value in evidence.target_scorer_rgb[0, 10, col]) for col in range(10, 14)}
    observed_lane = {tuple(int(value) for value in evidence.target_scorer_rgb[0, 20, col]) for col in (20, 22)}

    assert road_rgb == observed_road
    assert lane_rgb == observed_lane
    assert proposal.receipt.actual_palette_count_by_class == (2, 2, 0, 0, 0)
    assert proposal.receipt.selected_target_rgb_sse_after == 0


def test_target_pixel_oracle_is_exact_upper_control_on_every_selected_cell() -> None:
    proposal = _proposal(
        family=SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1,
        order=SameClassRealizationPrefixOrderV1.TARGET_RGB_SSE_DESCENDING_V1,
        prefix_count=6,
        palette_bound=None,
    )
    evidence = _evidence()
    expanded = _expanded_rgb(proposal)

    assert proposal.receipt.target_pixel_oracle_exact_on_selected_cells is True
    assert proposal.receipt.selected_target_rgb_sse_after == 0
    assert proposal.receipt.actual_palette_count_by_class == (0, 0, 0, 0, 0)
    for (pair_id, row, col), rgb in expanded.items():
        assert pair_id == 17
        assert rgb == tuple(int(value) for value in evidence.target_scorer_rgb[0, row, col])


def test_prefix_omission_accounting_closes_globally_and_per_class() -> None:
    proposal = _proposal(
        family=SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1,
        order=SameClassRealizationPrefixOrderV1.CANONICAL_ADDRESS_V1,
        prefix_count=2,
        palette_bound=None,
    )
    receipt = proposal.receipt

    assert receipt.covered_debt_cell_count == 2
    assert receipt.omitted_debt_cell_count == 4
    assert receipt.covered_debt_count_by_class == (2, 0, 0, 0, 0)
    assert receipt.omitted_debt_count_by_class == (2, 2, 0, 0, 0)
    assert (
        tuple(
            covered + omitted
            for covered, omitted in zip(
                receipt.covered_debt_count_by_class,
                receipt.omitted_debt_count_by_class,
                strict=True,
            )
        )
        == receipt.total_debt_count_by_class
    )


def test_prefix_orders_are_non_equivalent_before_full_coverage() -> None:
    address = _proposal(
        family=SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1,
        order=SameClassRealizationPrefixOrderV1.CANONICAL_ADDRESS_V1,
        prefix_count=1,
        palette_bound=None,
    )
    sse = _proposal(
        family=SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1,
        order=SameClassRealizationPrefixOrderV1.TARGET_RGB_SSE_DESCENDING_V1,
        prefix_count=1,
        palette_bound=None,
    )

    assert address.program.runs[0].scorer_row == 10
    assert address.program.runs[0].scorer_col_start == 10
    assert sse.program.runs[0].scorer_row == 10
    assert sse.program.runs[0].scorer_col_start in (12, 13)
    assert address.receipt.program_sha256 != sse.receipt.program_sha256


def test_acquisition_is_deterministic_dense_free_and_receipts_roundtrip() -> None:
    evidence = _evidence()
    first = acquire_same_class_realization_repair_programs(evidence, plan=_plan())
    second = acquire_same_class_realization_repair_programs(evidence, plan=_plan())

    assert first == second
    assert evidence.current_semantic_labels.flags.writeable is False
    assert evidence.target_scorer_rgb.flags.writeable is False
    assert not hasattr(evidence, "to_bytes")
    assert first.proposals[0].receipt.family is (
        SameClassRealizationProposalFamilyV1.TARGET_PIXEL_RGB_ORACLE_CONTROL_V1
    )
    assert first.proposals[0].receipt.covered_debt_cell_count == 6
    assert 0 < len(first.unique_program_proposals) < len(first.proposals)
    assert first.duplicate_program_count == len(first.proposals) - len(first.unique_program_proposals)
    assert len({proposal.receipt.program_sha256 for proposal in first.unique_program_proposals}) == len(
        first.unique_program_proposals
    )
    for proposal in first.proposals:
        payload = proposal.receipt.to_receipt_bytes()
        assert parse_same_class_realization_proposal_receipt(payload) == proposal.receipt
        assert b'"current_semantic_labels":' not in payload
        assert b'"target_scorer_rgb":' not in payload
        assert proposal.receipt.priority_ordering_is_advisory_only is True
        assert proposal.receipt.g8_compile_invoked is False
        assert proposal.receipt.g8_packet_emitted is False
        assert proposal.receipt.scorer_invoked is False
        assert proposal.receipt.through_r_realization_closed is False
        assert proposal.receipt.requires_through_r_scorer_admission is True
        assert proposal.receipt.rgb_sse_is_encoder_coordinate_only is True
        assert proposal.receipt.whole_object_allocator_invoked is False
        assert proposal.receipt.score_claim is False
        assert proposal.receipt.candidate_claim is False
        assert proposal.receipt.public_archive_payload_reused is False

    with pytest.raises(SameClassRealizationEncoderError):
        parse_same_class_realization_proposal_receipt(first.proposals[0].receipt.to_receipt_bytes() + b"\n")


def test_invalid_plan_and_empty_debt_fail_without_silent_default() -> None:
    with pytest.raises(SameClassRealizationEncoderError, match="palette sizes"):
        SameClassRealizationAcquisitionPlanV1(
            palette_sizes_per_class=(1,),
            geometric_prefix_ratios=(2,),
        )
    with pytest.raises(SameClassRealizationEncoderError, match="prefix ratios"):
        SameClassRealizationAcquisitionPlanV1(
            palette_sizes_per_class=(2,),
            geometric_prefix_ratios=(1,),
        )

    semantic, target, _realized, candidate_rgb, target_rgb = _arrays()
    realized = target.copy()
    no_debt = EncoderOnlySameClassRealizationEvidenceV1(
        source_pair_ids=(17,),
        base_interpretation=SameClassRealizationBaseInterpretationV1.EXACT_SEMANTIC_G_CONTROL_V1,
        base_semantic_binding_sha256="1" * 64,
        base_p_section_sha256="6" * 64,
        base_p_section_bytes=101,
        base_g_section_sha256="7" * 64,
        base_g_section_bytes=202,
        base_camera_y1_sha256="8" * 64,
        frozen_scorer_sha256="2" * 64,
        candidate_forward_receipt_sha256="3" * 64,
        target_forward_receipt_sha256="4" * 64,
        priority_ordering_receipt_sha256=N2_STAGE_ABLATION_PRIORITY_RECEIPT_SHA256,
        priority_ordering_axis=N2_STAGE_ABLATION_PRIORITY_AXIS,
        priority_ordering_scope=N2_STAGE_ABLATION_PRIORITY_SCOPE,
        current_semantic_labels_sha256=_sha256(semantic),
        target_labels_sha256=_sha256(target),
        realized_labels_sha256=_sha256(realized),
        candidate_scorer_rgb_sha256=_sha256(candidate_rgb),
        target_scorer_rgb_sha256=_sha256(target_rgb),
        current_semantic_labels=semantic,
        target_labels=target,
        realized_labels=realized,
        candidate_scorer_rgb=candidate_rgb,
        target_scorer_rgb=target_rgb,
    )
    with pytest.raises(SameClassRealizationEncoderError, match="nonempty exact debt"):
        acquire_same_class_realization_repair_programs(no_debt, plan=_plan())
