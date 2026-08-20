# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass, replace

import numpy as np
import pytest

import tac.witness_dsl.taskspace_selective_topology_acquisition as selective
from tac.optimization.direct_description_carrier_compose import ReceiverRealizationProfileV1
from tac.witness_dsl.generative_taskspace_correction import (
    DecodedGenerativeCorrectionV1,
    EncoderOnlyTeacherEvidenceV1,
    PredictorSemanticStateV1,
    parse_generative_taskspace_correction,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_HEADER as A3_PACKET_HEADER,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_MAGIC as A3_PACKET_MAGIC,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_VERSION as A3_PACKET_VERSION,
)
from tac.witness_dsl.taskspace_pass_conditional_a import PACKET_MAGIC as PASS_A_PACKET_MAGIC
from tac.witness_dsl.taskspace_pass_semantic_g import ENVELOPE_MAGIC as PASS_G_ENVELOPE_MAGIC
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    PACKET_HEADER as POST_G8_A_PACKET_HEADER,
)
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    PACKET_MAGIC as POST_G8_A_PACKET_MAGIC,
)
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    PACKET_VERSION as POST_G8_A_PACKET_VERSION,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    parse_generative_taskspace_correction_v2,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    PACKET_MAGIC as G8_REPAIR_PACKET_MAGIC,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    compose_same_class_realization_repair_g_section,
)
from tac.witness_dsl.taskspace_selective_topology_acquisition import (
    EncoderOnlySelectiveTopologyEvidenceV1,
    SelectiveTopologyAcquisitionError,
    SelectiveTopologyAcquisitionPlanV1,
    SelectiveTopologyBlockerCodeV1,
    SelectiveTopologyG7DomainV1,
    SelectiveTopologyG12SeamStatusV1,
    SelectiveTopologyGrammarFamilyV1,
    SelectiveTopologyPrefixOrderV1,
    SelectiveTopologyProductionEnvelopeStatusV1,
    acquire_selective_topology_programs,
    bind_selective_topology_post_forward,
    make_selective_topology_g7_proposals,
    parse_selective_topology_acquisition_receipt,
    parse_selective_topology_fresh_g8_request_receipt,
    parse_selective_topology_program_proposal_receipt,
    validate_selective_topology_g7_bundle,
)
from tac.witness_dsl.taskspace_whole_archive_allocator import TaskspaceSectionBundleV1

PAIR_ID = 17
SHAPE = (1, 384, 512)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One exhaustive partition with both precedence directions in row 3."""

    semantic = np.zeros(SHAPE, dtype=np.uint8)
    target = semantic.copy()
    realized = semantic.copy()

    # Row 2: semantic target, realized wrong.
    semantic[0, 10, 10] = target[0, 10, 10] = 1
    realized[0, 10, 10] = 2

    # Row 3: semantic wrong, realized wrong.  Adjacent target-0 cells have
    # three source classes so family 3 can coalesce the target birth while
    # preserving source-specific precedence deaths.
    rows = (
        (20, 20, 4, 0),
        (20, 21, 3, 0),
        (20, 22, 2, 0),
        (20, 23, 0, 4),
        (20, 24, 0, 4),
        (20, 25, 0, 4),
        (21, 30, 3, 1),
        (22, 40, 2, 3),
    )
    for row, column, source_class, target_class in rows:
        semantic[0, row, column] = source_class
        target[0, row, column] = target_class
        realized[0, row, column] = (target_class + 1) % 5

    # Row 4: semantic wrong, realized target.  One cell is adjacent to row 3
    # and one is remote so fortunate-clearance order is observable.
    semantic[0, 20, 19] = 1
    target[0, 20, 19] = realized[0, 20, 19] = 0
    semantic[0, 100, 100] = 4
    target[0, 100, 100] = realized[0, 100, 100] = 2
    return semantic, target, realized


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (10, 20, 30),
            (40, 50, 60),
            (70, 80, 90),
            (100, 110, 120),
            (130, 140, 150),
        )
    )


@dataclass(frozen=True)
class _Case:
    evidence: EncoderOnlySelectiveTopologyEvidenceV1
    teacher: EncoderOnlyTeacherEvidenceV1
    profile: ReceiverRealizationProfileV1
    target_scorer_rgb: np.ndarray


def _case(*, state_version: str = "v2") -> _Case:
    semantic, target, realized = _arrays()
    target_scorer_rgb = np.repeat(target[..., None], 3, axis=-1).astype(np.uint8)
    predictor_packet = b"LVLS1-selective-topology-synthetic"
    if state_version == "v2":
        state: PredictorSemanticStateV1 | TaskspacePredictorStateV2 = TaskspacePredictorStateV2(
            predictor_program=predictor_packet,
            predictor_renderer_sha256="b" * 64,
            source_archive_sha256="c" * 64,
            source_runtime_sha256="d" * 64,
            source_member_name="0.bin",
            source_pair_ids=(PAIR_ID,),
            labels=semantic,
            transport=NoTransportV2(),
        )
    elif state_version == "v1":
        state = PredictorSemanticStateV1(
            predictor_program_sha256=_bytes_sha256(predictor_packet),
            predictor_renderer_sha256="b" * 64,
            source_pair_ids=(PAIR_ID,),
            labels=semantic,
            pose6_codes=np.zeros((1, 6), dtype=np.int16),
        )
    else:  # pragma: no cover - test helper guard
        raise AssertionError(state_version)
    evidence = EncoderOnlySelectiveTopologyEvidenceV1(
        source_pair_ids=(PAIR_ID,),
        predictor_state=state,
        base_p_section_sha256=state.predictor_program_sha256,
        base_p_section_bytes=len(predictor_packet),
        current_camera_y1_sha256="e" * 64,
        target_scorer_rgb_sha256=_array_sha256(target_scorer_rgb),
        frozen_scorer_sha256="f" * 64,
        current_realization_forward_receipt_sha256="1" * 64,
        target_forward_receipt_sha256="2" * 64,
        current_semantic_labels_sha256=_array_sha256(semantic),
        target_labels_sha256=_array_sha256(target),
        baseline_realized_labels_sha256=_array_sha256(realized),
        current_semantic_labels=semantic,
        target_labels=target,
        baseline_realized_labels=realized,
    )
    teacher = EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="3" * 64,
        pbr2_sha256="4" * 64,
        target_labels_sha256=_array_sha256(target),
        obligation_ir_sha256="5" * 64,
        oracle_evidence_sha256="6" * 64,
        dense_y_sha256="7" * 64,
        target_labels=target,
        teacher_event_count=int(np.count_nonzero(semantic != target)),
    )
    target_scorer_rgb.setflags(write=False)
    return _Case(
        evidence=evidence,
        teacher=teacher,
        profile=_profile(),
        target_scorer_rgb=target_scorer_rgb,
    )


@pytest.fixture(scope="module")
def case_v2() -> _Case:
    return _case()


@pytest.fixture(scope="module")
def acquisition_v2(case_v2: _Case):
    return acquire_selective_topology_programs(
        case_v2.evidence,
        teacher_evidence=case_v2.teacher,
        realization_profile=case_v2.profile,
        plan=SelectiveTopologyAcquisitionPlanV1((1, 2, 4, 99)),
    )


def test_evidence_exposes_exhaustive_four_way_telemetry_and_binds_counted_p(case_v2: _Case) -> None:
    evidence = case_v2.evidence
    telemetry = evidence.cell_partition_telemetry

    assert telemetry.closed_cell_count == 384 * 512 - 11
    assert telemetry.realization_debt_cell_count == 1
    assert telemetry.topology_debt_cell_count == 8
    assert telemetry.fortunate_semantic_mismatch_cell_count == 2
    assert sum(telemetry.closed_count_by_target_class) == telemetry.closed_cell_count
    assert sum(telemetry.realization_debt_count_by_target_class) == 1
    assert sum(telemetry.topology_debt_count_by_target_class) == 8
    assert sum(telemetry.fortunate_semantic_mismatch_count_by_target_class) == 2
    assert (
        len(
            {
                telemetry.closed_mask_sha256,
                telemetry.realization_debt_mask_sha256,
                telemetry.topology_debt_mask_sha256,
                telemetry.fortunate_semantic_mismatch_mask_sha256,
            }
        )
        == 4
    )
    assert evidence.predictor_state_kind == "TASKSPACE_PREDICTOR_STATE_V2"
    assert evidence.serialized_dense_evidence_bytes == 0
    assert evidence.current_semantic_labels.flags.writeable is False
    assert evidence.target_labels.flags.writeable is False
    assert evidence.baseline_realized_labels.flags.writeable is False

    with pytest.raises(SelectiveTopologyAcquisitionError) as exc_info:
        replace(evidence, base_p_section_sha256="a" * 64)
    assert exc_info.value.code is SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH
    with pytest.raises(SelectiveTopologyAcquisitionError) as exc_info:
        replace(evidence, base_p_section_bytes=evidence.base_p_section_bytes + 1)
    assert exc_info.value.code is SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH


def test_all_prefix_order_family_programs_change_row3_only_and_dedupe_exact_bytes(
    case_v2: _Case,
    acquisition_v2: object,
) -> None:
    acquisition = acquisition_v2
    evidence = case_v2.evidence
    assert acquisition.resolved_prefix_sizes == (1, 2, 4, 8)
    assert len(acquisition.proposals) == 4 * 3 * 3
    assert {row.receipt.family for row in acquisition.proposals} == set(SelectiveTopologyGrammarFamilyV1)
    assert {row.receipt.prefix_order for row in acquisition.proposals} == set(SelectiveTopologyPrefixOrderV1)
    assert acquisition.receipt.baseline_cell_partition_telemetry == evidence.cell_partition_telemetry
    assert acquisition.receipt.arbitrary_component_or_byte_thresholds_applied is False
    assert acquisition.production_envelope_status is (
        SelectiveTopologyProductionEnvelopeStatusV1.PRODUCTION_SELECTIVE_TOPOLOGY_ENVELOPE_OWED
    )

    for proposal in acquisition.proposals:
        parsed = parse_generative_taskspace_correction_v2(
            proposal.compiled.packet,
            predictor_state=evidence.predictor_state,
        )
        changed = proposal.compiled.decoded.labels != evidence.current_semantic_labels
        assert np.array_equal(changed, proposal.selected_row3_mask)
        assert np.all(evidence.topology_debt_mask[proposal.selected_row3_mask])
        assert np.array_equal(
            proposal.compiled.decoded.labels[proposal.selected_row3_mask],
            evidence.target_labels[proposal.selected_row3_mask],
        )
        assert np.array_equal(
            proposal.compiled.decoded.labels[evidence.fortunate_semantic_mismatch_mask],
            evidence.current_semantic_labels[evidence.fortunate_semantic_mismatch_mask],
        )
        assert parsed.packet == proposal.compiled.packet
        assert proposal.receipt.baseline_cell_partition_telemetry == evidence.cell_partition_telemetry
        assert proposal.receipt.predictor_state_kind == "TASKSPACE_PREDICTOR_STATE_V2"
        assert proposal.receipt.row4_post_topology_h_preservation_claim is False
        assert proposal.receipt.production_candidate_eligible is False
        assert proposal.receipt.serialized_dense_evidence_bytes == 0
        assert len(proposal.compiled.packet) < evidence.target_labels.nbytes
        assert evidence.target_labels.tobytes() not in proposal.compiled.packet

    first_by_packet: dict[bytes, object] = {}
    for proposal in acquisition.proposals:
        first_by_packet.setdefault(proposal.compiled.packet, proposal)
    assert tuple(first_by_packet.values()) == acquisition.unique_program_proposals
    assert len(acquisition.unique_program_proposals) == len(first_by_packet) < len(acquisition.proposals)
    assert len({row.compiled.packet for row in acquisition.unique_program_proposals}) == len(
        acquisition.unique_program_proposals
    )


def test_legacy_v1_state_uses_the_same_wire_without_fabricated_selective_identity() -> None:
    case = _case(state_version="v1")
    acquisition = acquire_selective_topology_programs(
        case.evidence,
        teacher_evidence=case.teacher,
        realization_profile=case.profile,
        plan=SelectiveTopologyAcquisitionPlanV1((2,)),
    )

    assert len(acquisition.proposals) == 9
    assert case.evidence.predictor_state_kind == "PREDICTOR_SEMANTIC_STATE_V1"
    for proposal in acquisition.proposals:
        parsed = parse_generative_taskspace_correction(
            proposal.compiled.packet,
            predictor_state=case.evidence.predictor_state,
        )
        assert parsed.packet == proposal.compiled.packet
        assert proposal.receipt.predictor_state_kind == "PREDICTOR_SEMANTIC_STATE_V1"


def _proposal(acquisition: object, *, family: object, order: object, prefix: int):
    return next(
        row
        for row in acquisition.proposals
        if row.receipt.family is family
        and row.receipt.prefix_order is order
        and row.receipt.resolved_prefix_cell_count == prefix
    )


def test_precedence_event_families_and_closed_orders_are_observable(acquisition_v2: object) -> None:
    canonical_singleton = _proposal(
        acquisition_v2,
        family=SelectiveTopologyGrammarFamilyV1.SINGLETON_BOX_CONTROL_V1,
        order=SelectiveTopologyPrefixOrderV1.CANONICAL_ADDRESS_V1,
        prefix=8,
    )
    transition_runs = _proposal(
        acquisition_v2,
        family=SelectiveTopologyGrammarFamilyV1.SOURCE_TARGET_ROW_RUN_BOX_V1,
        order=SelectiveTopologyPrefixOrderV1.CANONICAL_ADDRESS_V1,
        prefix=8,
    )
    target_runs = _proposal(
        acquisition_v2,
        family=SelectiveTopologyGrammarFamilyV1.TARGET_BIRTH_SOURCE_DEATH_ROW_RUN_BOX_V1,
        order=SelectiveTopologyPrefixOrderV1.CANONICAL_ADDRESS_V1,
        prefix=8,
    )
    assert canonical_singleton.receipt.birth_event_count == 8
    assert transition_runs.receipt.birth_event_count == 6
    assert target_runs.receipt.birth_event_count == 4
    assert canonical_singleton.receipt.death_event_count == 3

    parsed = parse_generative_taskspace_correction_v2(
        canonical_singleton.compiled.packet,
        predictor_state=_case().evidence.predictor_state,
    )
    events = parsed.program.topology_events
    assert any(row.role == "MyCar" and row.action == "death" and row.x0 == 20 for row in events)
    assert any(row.role == "MyCar" and row.action == "birth" and row.x0 == 23 for row in events)
    assert not any(row.role == "Road" and row.action == "death" and row.x0 == 23 for row in events)

    canonical_first = _proposal(
        acquisition_v2,
        family=SelectiveTopologyGrammarFamilyV1.SINGLETON_BOX_CONTROL_V1,
        order=SelectiveTopologyPrefixOrderV1.CANONICAL_ADDRESS_V1,
        prefix=1,
    )
    transition_first = _proposal(
        acquisition_v2,
        family=SelectiveTopologyGrammarFamilyV1.SINGLETON_BOX_CONTROL_V1,
        order=SelectiveTopologyPrefixOrderV1.TRANSITION_MASS_DESCENDING_V1,
        prefix=1,
    )
    clearance_first = _proposal(
        acquisition_v2,
        family=SelectiveTopologyGrammarFamilyV1.SINGLETON_BOX_CONTROL_V1,
        order=SelectiveTopologyPrefixOrderV1.FORTUNATE_CLEARANCE_DESCENDING_V1,
        prefix=1,
    )
    assert tuple(np.argwhere(canonical_first.selected_row3_mask)[0]) == (0, 20, 20)
    assert tuple(np.argwhere(transition_first.selected_row3_mask)[0]) == (0, 20, 23)
    assert tuple(np.argwhere(clearance_first.selected_row3_mask)[0]) == (0, 22, 40)
    assert all(
        _proposal(
            acquisition_v2,
            family=SelectiveTopologyGrammarFamilyV1.SINGLETON_BOX_CONTROL_V1,
            order=order,
            prefix=8,
        ).receipt.resolved_prefix_cell_count
        == 8
        for order in SelectiveTopologyPrefixOrderV1
    )


def test_receipts_are_canonical_dense_free_and_reject_authority_mutation(acquisition_v2: object) -> None:
    proposal = acquisition_v2.proposals[0]
    proposal_payload = proposal.receipt.to_receipt_bytes()
    acquisition_payload = acquisition_v2.receipt.to_receipt_bytes()
    assert parse_selective_topology_program_proposal_receipt(proposal_payload) == proposal.receipt
    assert parse_selective_topology_acquisition_receipt(acquisition_payload) == acquisition_v2.receipt
    value = json.loads(proposal_payload)
    assert "current_semantic_labels" not in value
    assert "target_labels" not in value

    value["row4_post_topology_h_preservation_claim"] = True
    mutated = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    with pytest.raises(SelectiveTopologyAcquisitionError):
        parse_selective_topology_program_proposal_receipt(mutated)


def test_typed_empty_teacher_unrelated_mutation_and_cardinality_blockers(
    case_v2: _Case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SelectiveTopologyAcquisitionError) as exc_info:
        SelectiveTopologyAcquisitionPlanV1(())
    assert exc_info.value.code is SelectiveTopologyBlockerCodeV1.EMPTY_RESOLVED_PREFIX

    semantic = np.zeros(SHAPE, dtype=np.uint8)
    state = TaskspacePredictorStateV2(
        predictor_program=b"LVLS1-no-debt",
        predictor_renderer_sha256="b" * 64,
        source_archive_sha256="c" * 64,
        source_runtime_sha256="d" * 64,
        source_member_name="0.bin",
        source_pair_ids=(PAIR_ID,),
        labels=semantic,
        transport=NoTransportV2(),
    )
    no_debt = replace(
        case_v2.evidence,
        predictor_state=state,
        base_p_section_sha256=state.predictor_program_sha256,
        base_p_section_bytes=len(state.predictor_program),
        current_semantic_labels_sha256=_array_sha256(semantic),
        target_labels_sha256=_array_sha256(semantic),
        baseline_realized_labels_sha256=_array_sha256(semantic),
        current_semantic_labels=semantic,
        target_labels=semantic,
        baseline_realized_labels=semantic,
    )
    no_debt_teacher = replace(
        case_v2.teacher,
        target_labels_sha256=_array_sha256(semantic),
        target_labels=semantic,
        teacher_event_count=0,
    )
    with pytest.raises(SelectiveTopologyAcquisitionError) as exc_info:
        acquire_selective_topology_programs(
            no_debt,
            teacher_evidence=no_debt_teacher,
            realization_profile=case_v2.profile,
            plan=SelectiveTopologyAcquisitionPlanV1((1,)),
        )
    assert exc_info.value.code is SelectiveTopologyBlockerCodeV1.EMPTY_TOPOLOGY_DEBT

    wrong_teacher = replace(case_v2.teacher, teacher_event_count=case_v2.teacher.teacher_event_count + 1)
    with pytest.raises(SelectiveTopologyAcquisitionError) as exc_info:
        acquire_selective_topology_programs(
            case_v2.evidence,
            teacher_evidence=wrong_teacher,
            realization_profile=case_v2.profile,
            plan=SelectiveTopologyAcquisitionPlanV1((1,)),
        )
    assert exc_info.value.code is SelectiveTopologyBlockerCodeV1.PREDICTOR_EVIDENCE_TEACHER_CUSTODY_MISMATCH

    real_apply = selective.apply_generative_taskspace_correction_v2

    def mutate_outside_selection(packet: bytes, *, predictor_state: TaskspacePredictorStateV2):
        decoded = real_apply(packet, predictor_state=predictor_state)
        labels = decoded.labels.copy()
        labels[0, 0, 0] = 1
        return DecodedGenerativeCorrectionV1(
            labels=labels,
            realization_profile=decoded.realization_profile,
            correction_packet_sha256=decoded.correction_packet_sha256,
        )

    monkeypatch.setattr(selective, "apply_generative_taskspace_correction_v2", mutate_outside_selection)
    with pytest.raises(SelectiveTopologyAcquisitionError) as exc_info:
        acquire_selective_topology_programs(
            case_v2.evidence,
            teacher_evidence=case_v2.teacher,
            realization_profile=case_v2.profile,
            plan=SelectiveTopologyAcquisitionPlanV1((1,)),
        )
    assert exc_info.value.code is SelectiveTopologyBlockerCodeV1.DECODED_MUTATION_OUTSIDE_SELECTED_ROW3
    monkeypatch.setattr(selective, "apply_generative_taskspace_correction_v2", real_apply)

    large_semantic = np.zeros(SHAPE, dtype=np.uint8)
    large_target = large_semantic.copy()
    large_realized = large_semantic.copy()
    flat_count = selective.MAX_SINGLETON_PREFIX_CELLS + 1
    large_semantic.reshape(-1)[:flat_count] = 4
    large_realized.reshape(-1)[:flat_count] = 1
    large_state = TaskspacePredictorStateV2(
        predictor_program=b"LVLS1-cardinality",
        predictor_renderer_sha256="b" * 64,
        source_archive_sha256="c" * 64,
        source_runtime_sha256="d" * 64,
        source_member_name="0.bin",
        source_pair_ids=(PAIR_ID,),
        labels=large_semantic,
        transport=NoTransportV2(),
    )
    large_evidence = replace(
        case_v2.evidence,
        predictor_state=large_state,
        base_p_section_sha256=large_state.predictor_program_sha256,
        base_p_section_bytes=len(large_state.predictor_program),
        current_semantic_labels_sha256=_array_sha256(large_semantic),
        target_labels_sha256=_array_sha256(large_target),
        baseline_realized_labels_sha256=_array_sha256(large_realized),
        current_semantic_labels=large_semantic,
        target_labels=large_target,
        baseline_realized_labels=large_realized,
    )
    selected = np.zeros(SHAPE, dtype=np.bool_)
    selected.reshape(-1)[:flat_count] = True
    with pytest.raises(SelectiveTopologyAcquisitionError) as exc_info:
        selective._event_plan(
            large_evidence,
            selected,
            SelectiveTopologyGrammarFamilyV1.SINGLETON_BOX_CONTROL_V1,
        )
    assert exc_info.value.code is SelectiveTopologyBlockerCodeV1.TACG1C_EVENT_CARDINALITY_OVERFLOW


def test_fresh_post_forward_recomputes_h_prime_debt_and_preserves_trajectory_handoff_custody(
    case_v2: _Case,
    acquisition_v2: object,
) -> None:
    proposal = _proposal(
        acquisition_v2,
        family=SelectiveTopologyGrammarFamilyV1.SINGLETON_BOX_CONTROL_V1,
        order=SelectiveTopologyPrefixOrderV1.CANONICAL_ADDRESS_V1,
        prefix=2,
    )
    post_realized = case_v2.evidence.target_labels.copy()
    selected_coordinate = tuple(np.argwhere(proposal.selected_row3_mask)[0])
    post_realized[selected_coordinate] = (int(case_v2.evidence.target_labels[selected_coordinate]) + 1) % 5
    scorer_rgb = np.full((1, 384, 512, 3), 37, dtype=np.uint8)
    camera_y1 = np.full((1, 874, 1164, 3), 73, dtype=np.uint8)
    request = bind_selective_topology_post_forward(
        case_v2.evidence,
        proposal,
        post_topology_realized_labels=post_realized,
        post_topology_realized_labels_sha256=_array_sha256(post_realized),
        post_topology_scorer_rgb=scorer_rgb,
        post_topology_scorer_rgb_sha256=_array_sha256(scorer_rgb),
        target_scorer_rgb=case_v2.target_scorer_rgb,
        target_scorer_rgb_sha256=_array_sha256(case_v2.target_scorer_rgb),
        post_topology_camera_y1=camera_y1,
        post_topology_camera_y1_sha256=_array_sha256(camera_y1),
        post_topology_forward_receipt_sha256="8" * 64,
    )

    assert request.selective_g_program_proposal is proposal
    assert request.selective_g_packet == proposal.compiled.packet
    assert np.array_equal(request.post_topology_semantic_labels, proposal.compiled.decoded.labels)
    assert request.fresh_realization_debt_mask[selected_coordinate]
    assert request.receipt.fresh_realization_debt_cell_count == 1
    assert request.receipt.fresh_realization_debt_mask_sha256 != (request.receipt.baseline_realization_debt_mask_sha256)
    assert request.receipt.baseline_g8_debt_reused is False
    assert request.receipt.baseline_g8_program_reused is False
    assert request.g12_selective_base_status is (
        SelectiveTopologyG12SeamStatusV1.G12_SELECTIVE_TACG1C_BASE_INTERPRETATION_OWED
    )
    assert request.production_envelope_status is (
        SelectiveTopologyProductionEnvelopeStatusV1.PRODUCTION_SELECTIVE_TOPOLOGY_ENVELOPE_OWED
    )
    assert request.post_topology_camera_y1.flags.writeable is False
    assert request.post_topology_scorer_rgb.flags.writeable is False
    assert request.target_scorer_rgb.flags.writeable is False
    assert request.receipt.target_forward_receipt_sha256 == case_v2.evidence.target_forward_receipt_sha256
    assert (
        sum(
            (
                request.receipt.fresh_cell_partition_telemetry.closed_cell_count,
                request.receipt.fresh_cell_partition_telemetry.realization_debt_cell_count,
                request.receipt.fresh_cell_partition_telemetry.topology_debt_cell_count,
                request.receipt.fresh_cell_partition_telemetry.fortunate_semantic_mismatch_cell_count,
            )
        )
        == 384 * 512
    )
    assert parse_selective_topology_fresh_g8_request_receipt(request.receipt.to_receipt_bytes()) == request.receipt

    with pytest.raises(SelectiveTopologyAcquisitionError) as exc_info:
        bind_selective_topology_post_forward(
            case_v2.evidence,
            proposal,
            post_topology_realized_labels=post_realized,
            post_topology_realized_labels_sha256=_array_sha256(post_realized),
            post_topology_scorer_rgb=scorer_rgb,
            post_topology_scorer_rgb_sha256=_array_sha256(scorer_rgb),
            target_scorer_rgb=case_v2.target_scorer_rgb,
            target_scorer_rgb_sha256=_array_sha256(case_v2.target_scorer_rgb),
            post_topology_camera_y1=camera_y1,
            post_topology_camera_y1_sha256=_array_sha256(camera_y1),
            post_topology_forward_receipt_sha256=(case_v2.evidence.current_realization_forward_receipt_sha256),
        )
    assert exc_info.value.code is SelectiveTopologyBlockerCodeV1.BASELINE_FORWARD_RECEIPT_REUSED


def _a_packet(
    *,
    magic: bytes,
    version: int,
    header: object,
    source_binding_byte: int,
    pair_start: int = PAIR_ID,
    pair_count: int = 1,
) -> bytes:
    source_binding = bytes([source_binding_byte]) * 32
    fixed = (magic, version, 0, pair_start, pair_count, source_binding, 0, 0, header.size)
    zero_crc = header.pack(*fixed, 0)
    return header.pack(*fixed, zlib.crc32(zero_crc) & 0xFFFFFFFF)


def test_g7_hooks_are_lazy_atomic_and_accept_only_matching_plain_or_composite_domains(
    acquisition_v2: object,
) -> None:
    proposal = acquisition_v2.unique_program_proposals[0]
    baseline_a = _a_packet(
        magic=A3_PACKET_MAGIC,
        version=A3_PACKET_VERSION,
        header=A3_PACKET_HEADER,
        source_binding_byte=1,
    )
    baseline = TaskspaceSectionBundleV1(
        predictor_packet=b"LVLS1-selective-topology-synthetic",
        generative_correction_packet=b"TACG1C\x00\x00baseline",
        coupled_preimage_packet=baseline_a,
    )
    calls = 0

    def plain_builder(
        bundle: TaskspaceSectionBundleV1,
        selected_proposal: object,
    ) -> TaskspaceSectionBundleV1:
        nonlocal calls
        calls += 1
        return TaskspaceSectionBundleV1(
            predictor_packet=bundle.predictor_packet,
            generative_correction_packet=selected_proposal.compiled.packet,
            coupled_preimage_packet=_a_packet(
                magic=A3_PACKET_MAGIC,
                version=A3_PACKET_VERSION,
                header=A3_PACKET_HEADER,
                source_binding_byte=2,
            ),
        )

    g7 = make_selective_topology_g7_proposals(
        acquisition_v2,
        whole_bundle_builder=plain_builder,
    )
    assert len(g7) == len(acquisition_v2.unique_program_proposals)
    assert calls == 0
    rebuilt_plain = g7[0].transform(baseline)
    assert calls == 1
    assert (
        validate_selective_topology_g7_bundle(
            baseline,
            rebuilt_plain,
            proposal,
        )
        is SelectiveTopologyG7DomainV1.PLAIN_TACG1C_TACA3P1
    )

    composite = compose_same_class_realization_repair_g_section(
        proposal.compiled.packet,
        G8_REPAIR_PACKET_MAGIC + b"synthetic-repair",
    )
    rebuilt_composite = TaskspaceSectionBundleV1(
        predictor_packet=baseline.predictor_packet,
        generative_correction_packet=composite,
        coupled_preimage_packet=_a_packet(
            magic=POST_G8_A_PACKET_MAGIC,
            version=POST_G8_A_PACKET_VERSION,
            header=POST_G8_A_PACKET_HEADER,
            source_binding_byte=3,
        ),
    )
    assert (
        validate_selective_topology_g7_bundle(
            baseline,
            rebuilt_composite,
            proposal,
        )
        is SelectiveTopologyG7DomainV1.COMPOSITE_TACG8S1_TACA8P1
    )

    stale_a = replace(rebuilt_plain, coupled_preimage_packet=baseline.coupled_preimage_packet)
    with pytest.raises(SelectiveTopologyAcquisitionError):
        validate_selective_topology_g7_bundle(baseline, stale_a, proposal)
    with pytest.raises(SelectiveTopologyAcquisitionError):
        validate_selective_topology_g7_bundle(
            baseline,
            replace(rebuilt_plain, predictor_packet=b"LVLS1-mutated-P"),
            proposal,
        )
    with pytest.raises(SelectiveTopologyAcquisitionError):
        validate_selective_topology_g7_bundle(
            baseline,
            replace(rebuilt_plain, terminal_quotient_packet=b"optional-T"),
            proposal,
        )
    with pytest.raises(SelectiveTopologyAcquisitionError):
        validate_selective_topology_g7_bundle(
            replace(baseline, terminal_quotient_packet=b"preexisting-T"),
            replace(rebuilt_plain, terminal_quotient_packet=b"preexisting-T"),
            proposal,
        )

    pass_domain = TaskspaceSectionBundleV1(
        predictor_packet=baseline.predictor_packet,
        generative_correction_packet=PASS_G_ENVELOPE_MAGIC + b"not-selective",
        coupled_preimage_packet=PASS_A_PACKET_MAGIC + b"not-selective",
    )
    with pytest.raises(SelectiveTopologyAcquisitionError):
        validate_selective_topology_g7_bundle(baseline, pass_domain, proposal)

    foreign = acquisition_v2.unique_program_proposals[-1]
    wrong_inner = TaskspaceSectionBundleV1(
        predictor_packet=baseline.predictor_packet,
        generative_correction_packet=compose_same_class_realization_repair_g_section(
            foreign.compiled.packet,
            G8_REPAIR_PACKET_MAGIC + b"synthetic-repair",
        ),
        coupled_preimage_packet=rebuilt_composite.coupled_preimage_packet,
    )
    with pytest.raises(SelectiveTopologyAcquisitionError):
        validate_selective_topology_g7_bundle(baseline, wrong_inner, proposal)

    wrong_a_domain = replace(
        rebuilt_plain,
        coupled_preimage_packet=_a_packet(
            magic=POST_G8_A_PACKET_MAGIC,
            version=POST_G8_A_PACKET_VERSION,
            header=POST_G8_A_PACKET_HEADER,
            source_binding_byte=4,
        ),
    )
    with pytest.raises(SelectiveTopologyAcquisitionError):
        validate_selective_topology_g7_bundle(baseline, wrong_a_domain, proposal)

    wrong_pair_window = replace(
        rebuilt_plain,
        coupled_preimage_packet=_a_packet(
            magic=A3_PACKET_MAGIC,
            version=A3_PACKET_VERSION,
            header=A3_PACKET_HEADER,
            source_binding_byte=5,
            pair_start=PAIR_ID + 1,
        ),
    )
    with pytest.raises(SelectiveTopologyAcquisitionError):
        validate_selective_topology_g7_bundle(baseline, wrong_pair_window, proposal)
