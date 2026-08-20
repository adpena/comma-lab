# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    ReceiverRealizationProfileV1,
    TopologyEventV1,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_dsl.generative_taskspace_correction import (
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_HEADER as A3_PACKET_HEADER,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PredictorCameraPairSurfaceV1,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    compile_generative_taskspace_correction_v2,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    A3_EXPANDED_RGB_ROW,
    COMPOSITE_G_SECTION_HEADER,
    PACKET_HEADER,
    RUN_ROW,
    CompiledSameClassRealizationRepairV1,
    CountedTaskspaceSectionIdentityV1,
    EncoderOnlyExactTargetLabelCustodyV1,
    SameClassRealizationRepairError,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairRunV1,
    SameClassRealizationRepairSurfaceV1,
    SameClassSemanticRoleV1,
    TaskspaceSectionRoleV1,
    compile_same_class_realization_repair,
    compile_same_class_realization_repair_for_pga_sections,
    compose_same_class_realization_repair_g_section,
    decode_same_class_realization_repair_from_pga_sections,
    decode_same_class_realization_repair_packet,
    derive_same_class_realization_repair_source_binding,
    parse_same_class_realization_repair_g_section,
    parse_same_class_realization_repair_packet,
    parse_same_class_realization_repair_receipt,
)

PAIR_ID = 29
RUN_ROW_INDEX = 123
RUN_COL_START = 100
RUN_COL_STOP = 164
RUN_RGB = (90, 100, 110)


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _section(
    role: TaskspaceSectionRoleV1,
    *,
    index: int,
    payload: bytes,
) -> CountedTaskspaceSectionIdentityV1:
    return CountedTaskspaceSectionIdentityV1(
        role=role,
        section_index=index,
        codec_id=f"synthetic-{role.value.lower()}-codec.v1",
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


@pytest.fixture(scope="module")
def source_surface() -> SameClassRealizationRepairSurfaceV1:
    camera = np.empty((1, 874, 1164, 3), dtype=np.uint8)
    camera[:] = (10, 20, 30)
    labels = np.zeros((1, 384, 512), dtype=np.uint8)
    return SameClassRealizationRepairSurfaceV1(
        source_pair_ids=(PAIR_ID,),
        corrected_y1_camera=camera,
        g_semantic_labels=labels,
        g_semantic_binding_sha256="a" * 64,
        p_section_identity=_section(
            TaskspaceSectionRoleV1.P,
            index=0,
            payload=b"counted-p-section",
        ),
        g_section_identity=_section(
            TaskspaceSectionRoleV1.G,
            index=1,
            payload=b"counted-g-section",
        ),
    )


@pytest.fixture(scope="module")
def target_custody(
    source_surface: SameClassRealizationRepairSurfaceV1,
) -> EncoderOnlyExactTargetLabelCustodyV1:
    labels = source_surface.g_semantic_labels.copy()
    return EncoderOnlyExactTargetLabelCustodyV1(
        source_artifact_sha256="b" * 64,
        source_member_name="gt_n600.npz::lstars",
        source_member_sha256="c" * 64,
        source_pair_ids=source_surface.source_pair_ids,
        target_labels_sha256=_array_sha256(labels),
        target_labels=labels,
    )


def _run(
    *,
    pair_id: int = PAIR_ID,
    row: int = RUN_ROW_INDEX,
    start: int = RUN_COL_START,
    stop: int = RUN_COL_STOP,
    semantic_class: int = 0,
    semantic_role: SameClassSemanticRoleV1 = SameClassSemanticRoleV1.ROAD,
    rgb: tuple[int, int, int] = RUN_RGB,
) -> SameClassRealizationRepairRunV1:
    return SameClassRealizationRepairRunV1(
        source_pair_id=pair_id,
        scorer_row=row,
        scorer_col_start=start,
        scorer_col_stop=stop,
        semantic_class=semantic_class,
        semantic_role=semantic_role,
        rgb_u8=rgb,
    )


def _program(**run_updates: Any) -> SameClassRealizationRepairProgramV1:
    return SameClassRealizationRepairProgramV1((_run(**run_updates),))


@pytest.fixture(scope="module")
def compiled(
    source_surface: SameClassRealizationRepairSurfaceV1,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1,
) -> CompiledSameClassRealizationRepairV1:
    return compile_same_class_realization_repair(
        _program(),
        surface=source_surface,
        target_custody=target_custody,
    )


def test_exact_falsifier_same_semantic_and_target_class_but_realized_oracle_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
    source_surface: SameClassRealizationRepairSurfaceV1,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1,
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    """A semantic-correct cell can still carry distinct camera realization debt."""

    import tac.witness_dsl.taskspace_same_class_realization_repair as module

    cell = (0, RUN_ROW_INDEX, RUN_COL_START)
    synthetic_realized_label_oracle = source_surface.g_semantic_labels.copy()
    synthetic_realized_label_oracle[cell] = 1
    assert int(source_surface.g_semantic_labels[cell]) == 0
    assert int(target_custody.target_labels[cell]) == 0
    assert int(synthetic_realized_label_oracle[cell]) == 1

    calls = {"canonical_reconstruction": 0}
    original = module.apply_disjoint_camera_cell_reconstruction_v1

    def counted_reconstruction(*args: Any, **kwargs: Any):
        calls["canonical_reconstruction"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "apply_disjoint_camera_cell_reconstruction_v1", counted_reconstruction)
    replay = decode_same_class_realization_repair_packet(
        compiled.packet,
        surface=source_surface,
    )
    assert calls == {"canonical_reconstruction": 2}

    assert np.array_equal(replay.semantic_labels, source_surface.g_semantic_labels)
    assert replay.semantic_labels.tobytes() == source_surface.g_semantic_labels.tobytes()
    assert replay.receipt.output_g_labels_sha256 == source_surface.g_labels_sha256
    assert replay.receipt.semantic_labels_bit_identical is True
    assert replay.receipt.coordinate_present is True
    assert replay.receipt.through_r_target_realization_debt_closed is False
    assert replay.receipt.scorer_invoked is False
    assert replay.receipt.score_claim is False
    assert replay.receipt.candidate_claim is False
    assert replay.receipt.exact_eval_invoked is False
    assert replay.receipt.promotion_eligible is False
    assert replay.receipt.research_only is True

    assert not np.array_equal(replay.camera_y1, source_surface.corrected_y1_camera)
    assert replay.receipt.output_camera_y1_sha256 != source_surface.corrected_y1_camera_sha256
    assert replay.receipt.actually_changed_camera_values > 0

    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    before_num, denominator = operator.apply_numerators(source_surface.corrected_y1_camera[0])
    after_num, output_denominator = operator.apply_numerators(replay.camera_y1[0])
    owned_scorer = replay.owned_scorer_mask[0]
    owned_camera = replay.owned_camera_mask[0]
    expected = np.asarray(RUN_RGB, dtype=np.int64) * denominator
    assert denominator == output_denominator == replay.receipt.scorer_denominator
    assert np.all(after_num[RUN_ROW_INDEX, RUN_COL_START] == expected)
    assert np.all(after_num[owned_scorer] == expected)
    assert np.array_equal(after_num[~owned_scorer], before_num[~owned_scorer])
    assert np.array_equal(
        replay.camera_y1[0][~owned_camera],
        source_surface.corrected_y1_camera[0][~owned_camera],
    )
    assert replay.receipt.outside_owned_camera_bytes_identical is True
    assert replay.receipt.outside_owned_scorer_numerators_identical is True


def test_packet_binds_every_live_identity_and_parse_reencodes_exactly(
    source_surface: SameClassRealizationRepairSurfaceV1,
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    source = derive_same_class_realization_repair_source_binding(source_surface)
    parsed = parse_same_class_realization_repair_packet(
        compiled.packet,
        expected_source_binding=source,
    )
    receipt = compiled.decoded.receipt

    assert parsed.packet == compiled.packet
    assert parsed.program == compiled.program
    assert parsed.source_binding_sha256 == source.binding_sha256
    assert receipt.source_binding_sha256 == source.binding_sha256
    assert receipt.corrected_y1_camera_sha256 == source_surface.corrected_y1_camera_sha256
    assert receipt.g_semantic_binding_sha256 == source_surface.g_semantic_binding_sha256
    assert receipt.g_labels_sha256 == source_surface.g_labels_sha256
    assert receipt.p_section_identity_sha256 == source_surface.p_section_identity.identity_sha256
    assert receipt.g_section_identity_sha256 == source_surface.g_section_identity.identity_sha256
    assert receipt.source_pair_ids == source_surface.source_pair_ids
    assert parse_same_class_realization_repair_receipt(receipt.to_receipt_bytes()) == receipt


def test_one_shared_rgb_run_is_exactly_priced_against_expanded_a3_cells(
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    receipt = compiled.decoded.receipt
    expanded_cells = RUN_COL_STOP - RUN_COL_START

    assert PACKET_HEADER.size == 222
    assert RUN_ROW.size == 13
    assert A3_PACKET_HEADER.size == 62
    assert A3_EXPANDED_RGB_ROW.size == 9
    assert receipt.run_count == 1
    assert receipt.expanded_cell_count == expanded_cells == 64
    assert receipt.packet_body_bytes == RUN_ROW.size
    assert receipt.packet_bytes == PACKET_HEADER.size + RUN_ROW.size == 235
    assert receipt.expanded_a3_body_bytes == expanded_cells * A3_EXPANDED_RGB_ROW.size == 576
    assert receipt.expanded_a3_packet_bytes == A3_PACKET_HEADER.size + 576 == 638
    assert receipt.run_body_savings_bytes == 563
    assert receipt.run_packet_savings_bytes == 403
    assert receipt.run_body_compression_observed is True
    assert receipt.score_claim is False


def test_target_custody_changes_outside_admitted_run_do_not_change_packet(
    source_surface: SameClassRealizationRepairSurfaceV1,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1,
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    changed_targets = target_custody.target_labels.copy()
    changed_targets[0, 0, 0] = 4
    changed_custody = replace(
        target_custody,
        target_labels=changed_targets,
        target_labels_sha256=_array_sha256(changed_targets),
    )
    changed = compile_same_class_realization_repair(
        _program(),
        surface=source_surface,
        target_custody=changed_custody,
    )

    assert changed.packet == compiled.packet
    assert np.array_equal(changed.decoded.camera_y1, compiled.decoded.camera_y1)
    assert changed.admission_receipt.target_custody_binding_sha256 != (
        compiled.admission_receipt.target_custody_binding_sha256
    )
    assert changed.admission_receipt.serialized_target_bytes == 0
    assert changed.admission_receipt.target_inputs_encoder_only is True
    assert bytes.fromhex(target_custody.binding_sha256) not in compiled.packet
    assert bytes.fromhex(changed_custody.target_labels_sha256) not in changed.packet
    assert compiled.decoded.receipt.dense_target_labels_serialized is False
    assert compiled.decoded.receipt.dense_g_labels_serialized is False
    assert compiled.decoded.receipt.dense_scorer_state_serialized is False
    assert compiled.decoded.receipt.dense_margins_serialized is False


def test_pga_integration_emits_concrete_composite_g_proposal_and_replays_post_g8_y1() -> None:
    predictor_packet = b"LVLS1\x00synthetic-counted-predictor"
    labels = np.zeros((1, 384, 512), dtype=np.uint8)
    state = TaskspacePredictorStateV2(
        predictor_program=predictor_packet,
        predictor_renderer_sha256="4" * 64,
        source_archive_sha256="5" * 64,
        source_runtime_sha256="6" * 64,
        source_member_name="0.bin",
        source_pair_ids=(PAIR_ID,),
        labels=labels,
        transport=NoTransportV2(),
    )
    frames = np.empty((1, 2, 874, 1164, 3), dtype=np.uint8)
    frames[:, 0] = (5, 15, 25)
    frames[:, 1] = (10, 20, 30)
    predictor_surface = PredictorCameraPairSurfaceV1(
        predictor_state=state,
        chronological_camera_frames=frames,
        upstream_decode_receipt_sha256="7" * 64,
    )
    profile = ReceiverRealizationProfileV1(
        (
            (20, 80, 20),
            (240, 220, 40),
            (30, 30, 30),
            (220, 40, 40),
            (40, 80, 220),
        )
    )
    g_target = labels.copy()
    g_target[0, 10:12, 10:14] = 1
    g_teacher = EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="8" * 64,
        pbr2_sha256="9" * 64,
        target_labels_sha256=_array_sha256(g_target),
        obligation_ir_sha256="a" * 64,
        oracle_evidence_sha256="b" * 64,
        dense_y_sha256="c" * 64,
        target_labels=g_target,
        teacher_event_count=8,
    )
    compiled_g = compile_generative_taskspace_correction_v2(
        state,
        GenerativeCorrectionProgramV1(
            topology_events=(TopologyEventV1(PAIR_ID, "Lane", "birth", "box", 1, 10, 10, 12, 14),),
            realization_profile=profile,
        ),
        teacher_evidence=g_teacher,
    )
    semantic_g_packet = compiled_g.packet
    target = EncoderOnlyExactTargetLabelCustodyV1(
        source_artifact_sha256="1" * 64,
        source_member_name="gt_n600.npz::lstars",
        source_member_sha256="2" * 64,
        source_pair_ids=(PAIR_ID,),
        target_labels_sha256=_array_sha256(compiled_g.decoded.labels),
        target_labels=compiled_g.decoded.labels,
    )
    proposal = compile_same_class_realization_repair_for_pga_sections(
        _program(),
        predictor_section_payload=predictor_packet,
        semantic_g_section_payload=semantic_g_packet,
        predictor_surface=predictor_surface,
        predictor_codec_id="LVLS1.v1",
        semantic_g_codec_id="TACG1C.v2",
        target_custody=target,
    )
    parsed = parse_same_class_realization_repair_g_section(proposal.replacement_g_section_payload)
    replay = decode_same_class_realization_repair_from_pga_sections(
        proposal.replacement_g_section_payload,
        predictor_section_payload=predictor_packet,
        predictor_surface=predictor_surface,
        predictor_codec_id="LVLS1.v1",
        semantic_g_codec_id="TACG1C.v2",
    )

    assert parsed.semantic_g_packet == semantic_g_packet
    assert parsed.repair_packet == proposal.compiled_repair.packet
    assert parsed.section_bytes == proposal.composite_g_section
    assert proposal.replacement_g_section_bytes == (
        COMPOSITE_G_SECTION_HEADER.size + len(semantic_g_packet) + len(proposal.compiled_repair.packet)
    )
    assert proposal.raw_g_section_delta_bytes == (
        COMPOSITE_G_SECTION_HEADER.size + len(proposal.compiled_repair.packet)
    )
    assert proposal.monolithic_receiver_branch_required is True
    assert proposal.existing_a3_rebind_to_post_g8_y1_required is True
    assert np.array_equal(replay.post_g8_corrected_y1, proposal.post_g8_corrected_y1)
    assert np.array_equal(replay.unchanged_g_semantic_labels, compiled_g.decoded.labels)
    assert not np.array_equal(replay.post_g8_corrected_y1, predictor_surface.camera_p1)

    corrupted = proposal.composite_g_section[:-1] + bytes([proposal.composite_g_section[-1] ^ 1])
    with pytest.raises(SameClassRealizationRepairError, match="CRC"):
        parse_same_class_realization_repair_g_section(corrupted)

    changed_semantic_g = semantic_g_packet + b"-different"
    rebound_composite = compose_same_class_realization_repair_g_section(
        changed_semantic_g,
        proposal.compiled_repair.packet,
    )
    with pytest.raises(
        SameClassRealizationRepairError,
        match=r"source binding|G section identity|strict semantic G parse",
    ):
        decode_same_class_realization_repair_from_pga_sections(
            rebound_composite,
            predictor_section_payload=predictor_packet,
            predictor_surface=predictor_surface,
            predictor_codec_id="LVLS1.v1",
            semantic_g_codec_id="TACG1C.v2",
        )

    with pytest.raises(SameClassRealizationRepairError, match="predictor section bytes differ"):
        compile_same_class_realization_repair_for_pga_sections(
            _program(),
            predictor_section_payload=b"LVLS1\x00",
            semantic_g_section_payload=semantic_g_packet,
            predictor_surface=predictor_surface,
            predictor_codec_id="LVLS1.v1",
            semantic_g_codec_id="TACG1C.v2",
            target_custody=target,
        )
    with pytest.raises(SameClassRealizationRepairError, match="strict semantic G parse"):
        compile_same_class_realization_repair_for_pga_sections(
            _program(),
            predictor_section_payload=predictor_packet,
            semantic_g_section_payload=b"TACG1C\x00\x00",
            predictor_surface=predictor_surface,
            predictor_codec_id="LVLS1.v1",
            semantic_g_codec_id="TACG1C.v2",
            target_custody=target,
        )


def test_compile_rejects_target_mismatch_current_class_mismatch_and_foreign_pair(
    source_surface: SameClassRealizationRepairSurfaceV1,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1,
) -> None:
    wrong_target = target_custody.target_labels.copy()
    wrong_target[0, RUN_ROW_INDEX, RUN_COL_START] = 1
    wrong_target_custody = replace(
        target_custody,
        target_labels=wrong_target,
        target_labels_sha256=_array_sha256(wrong_target),
    )
    with pytest.raises(SameClassRealizationRepairError, match="equal target label"):
        compile_same_class_realization_repair(
            _program(),
            surface=source_surface,
            target_custody=wrong_target_custody,
        )

    with pytest.raises(SameClassRealizationRepairError, match="current G semantic label"):
        compile_same_class_realization_repair(
            _program(
                semantic_class=1,
                semantic_role=SameClassSemanticRoleV1.LANE,
            ),
            surface=source_surface,
            target_custody=target_custody,
        )

    with pytest.raises(SameClassRealizationRepairError, match="foreign source pair"):
        compile_same_class_realization_repair(
            _program(pair_id=PAIR_ID + 1),
            surface=source_surface,
            target_custody=target_custody,
        )


def test_direct_decode_rejects_canonical_packet_with_wrong_live_g_class(
    source_surface: SameClassRealizationRepairSurfaceV1,
) -> None:
    import tac.witness_dsl.taskspace_same_class_realization_repair as module

    wrong_class_program = _program(
        semantic_class=1,
        semantic_role=SameClassSemanticRoleV1.LANE,
    )
    source = derive_same_class_realization_repair_source_binding(source_surface)
    forged_but_canonical_packet = module._encode_packet(wrong_class_program, source)

    with pytest.raises(SameClassRealizationRepairError, match="packet-bound live G semantic labels"):
        decode_same_class_realization_repair_packet(
            forged_but_canonical_packet,
            surface=source_surface,
        )


def test_run_grammar_rejects_wrong_role_duplicates_order_overlap_and_split_aliases() -> None:
    with pytest.raises(SameClassRealizationRepairError, match="class and semantic role disagree"):
        _run(semantic_class=0, semantic_role=SameClassSemanticRoleV1.LANE)
    row = _run(start=10, stop=20)
    with pytest.raises(SameClassRealizationRepairError, match="unique canonical"):
        SameClassRealizationRepairProgramV1((row, row))
    earlier = _run(row=2, start=10, stop=20)
    later = _run(row=3, start=10, stop=20)
    with pytest.raises(SameClassRealizationRepairError, match="unique canonical"):
        SameClassRealizationRepairProgramV1((later, earlier))
    overlapping = _run(start=19, stop=30)
    with pytest.raises(SameClassRealizationRepairError, match="overlap"):
        SameClassRealizationRepairProgramV1((row, overlapping))
    adjacent = _run(start=20, stop=30)
    with pytest.raises(SameClassRealizationRepairError, match="must merge"):
        SameClassRealizationRepairProgramV1((row, adjacent))


def test_noop_repair_is_rejected_by_canonical_reconstruction_path(
    source_surface: SameClassRealizationRepairSurfaceV1,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1,
) -> None:
    with pytest.raises(SameClassRealizationRepairError, match="no-op alias"):
        compile_same_class_realization_repair(
            _program(start=RUN_COL_START, stop=RUN_COL_START + 1, rgb=(10, 20, 30)),
            surface=source_surface,
            target_custody=target_custody,
        )


def test_receiver_refuses_missing_canonical_constant_rgb_proof(
    monkeypatch: pytest.MonkeyPatch,
    source_surface: SameClassRealizationRepairSurfaceV1,
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    import tac.witness_dsl.taskspace_same_class_realization_repair as module

    original = module.apply_disjoint_camera_cell_reconstruction_v1

    def stripped_proof(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        return replace(result, constant_rgb_target_numerators_exact=False)

    monkeypatch.setattr(module, "apply_disjoint_camera_cell_reconstruction_v1", stripped_proof)
    with pytest.raises(SameClassRealizationRepairError, match="exact constant-RGB proof"):
        decode_same_class_realization_repair_packet(
            compiled.packet,
            surface=source_surface,
        )


def test_deletion_body_mutation_and_trailing_bytes_fail_closed(
    source_surface: SameClassRealizationRepairSurfaceV1,
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    with pytest.raises(SameClassRealizationRepairError, match="shorter"):
        decode_same_class_realization_repair_packet(b"", surface=source_surface)
    with pytest.raises(SameClassRealizationRepairError, match=r"length accounting|trailing"):
        decode_same_class_realization_repair_packet(compiled.packet[: -RUN_ROW.size], surface=source_surface)
    mutated = compiled.packet[:-1] + bytes([compiled.packet[-1] ^ 1])
    with pytest.raises(SameClassRealizationRepairError, match="CRC"):
        decode_same_class_realization_repair_packet(mutated, surface=source_surface)
    with pytest.raises(SameClassRealizationRepairError, match=r"length accounting|trailing"):
        decode_same_class_realization_repair_packet(compiled.packet + b"\x00", surface=source_surface)


def test_valid_counted_rgb_mutation_and_run_deletion_are_receiver_causal(
    source_surface: SameClassRealizationRepairSurfaceV1,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1,
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    mutated_rgb = compile_same_class_realization_repair(
        _program(rgb=(91, 101, 111)),
        surface=source_surface,
        target_custody=target_custody,
    )
    assert mutated_rgb.packet != compiled.packet
    assert not np.array_equal(mutated_rgb.decoded.camera_y1, compiled.decoded.camera_y1)
    assert np.array_equal(mutated_rgb.decoded.semantic_labels, compiled.decoded.semantic_labels)

    first_run = _run(stop=RUN_COL_START + 32)
    second_run = _run(row=RUN_ROW_INDEX + 1, stop=RUN_COL_START + 32)
    two_run = compile_same_class_realization_repair(
        SameClassRealizationRepairProgramV1((first_run, second_run)),
        surface=source_surface,
        target_custody=target_custody,
    )
    deleted_one_run = compile_same_class_realization_repair(
        SameClassRealizationRepairProgramV1((first_run,)),
        surface=source_surface,
        target_custody=target_custody,
    )
    assert len(two_run.packet) - len(deleted_one_run.packet) == RUN_ROW.size
    assert two_run.packet != deleted_one_run.packet
    assert not np.array_equal(two_run.decoded.camera_y1, deleted_one_run.decoded.camera_y1)
    assert np.array_equal(two_run.decoded.semantic_labels, deleted_one_run.decoded.semantic_labels)
    assert compiled.decoded.receipt.packet_absence_rejected is True
    assert compiled.decoded.receipt.stale_integrity_mutation_rejected is True
    assert compiled.decoded.receipt.counted_rgb_mutation_is_receiver_causal is True
    assert compiled.decoded.receipt.counted_run_deletion_is_receiver_causal_or_empty_rejected is True


def test_stale_y1_g_labels_semantic_binding_and_section_identities_fail_closed(
    source_surface: SameClassRealizationRepairSurfaceV1,
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    changed_camera = source_surface.corrected_y1_camera.copy()
    changed_camera[0, 0, 0, 0] ^= 1
    stale_y1 = replace(source_surface, corrected_y1_camera=changed_camera)
    with pytest.raises(SameClassRealizationRepairError, match=r"source binding|corrected Y1"):
        decode_same_class_realization_repair_packet(compiled.packet, surface=stale_y1)

    changed_labels = source_surface.g_semantic_labels.copy()
    changed_labels[0, 0, 0] = 4
    stale_labels = replace(source_surface, g_semantic_labels=changed_labels)
    with pytest.raises(SameClassRealizationRepairError, match=r"source binding|G labels"):
        decode_same_class_realization_repair_packet(compiled.packet, surface=stale_labels)

    stale_semantics = replace(source_surface, g_semantic_binding_sha256="d" * 64)
    with pytest.raises(SameClassRealizationRepairError, match=r"source binding|semantic binding"):
        decode_same_class_realization_repair_packet(compiled.packet, surface=stale_semantics)

    changed_p_identity = replace(source_surface.p_section_identity, payload_sha256="e" * 64)
    stale_p = replace(source_surface, p_section_identity=changed_p_identity)
    with pytest.raises(SameClassRealizationRepairError, match=r"source binding|P section identity"):
        decode_same_class_realization_repair_packet(compiled.packet, surface=stale_p)

    changed_g_identity = replace(source_surface.g_section_identity, payload_sha256="f" * 64)
    stale_g = replace(source_surface, g_section_identity=changed_g_identity)
    with pytest.raises(SameClassRealizationRepairError, match=r"source binding|G section identity"):
        decode_same_class_realization_repair_packet(compiled.packet, surface=stale_g)


def test_receipt_parser_rejects_numeric_and_truth_smuggling(
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    receipt = compiled.decoded.receipt
    payload = receipt.to_receipt_bytes()

    def encoded(**updates: Any) -> bytes:
        body = json.loads(payload)
        body.update(updates)
        return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")

    with pytest.raises(SameClassRealizationRepairError, match="exact integers"):
        parse_same_class_realization_repair_receipt(encoded(source_pair_ids=[True]))
    with pytest.raises(SameClassRealizationRepairError, match="exact integers"):
        parse_same_class_realization_repair_receipt(encoded(packet_bytes=float(receipt.packet_bytes)))
    with pytest.raises(SameClassRealizationRepairError, match="truth labels"):
        parse_same_class_realization_repair_receipt(encoded(score_claim=True))
    with pytest.raises(SameClassRealizationRepairError, match="recompose"):
        parse_same_class_realization_repair_receipt(encoded(g_labels_sha256="0" * 64))
    with pytest.raises(SameClassRealizationRepairError, match="not canonical JSON"):
        parse_same_class_realization_repair_receipt(payload + b"\n")


def test_compiled_object_closes_admission_source_and_program_counts(
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    other_program = _program(rgb=(91, 101, 111))
    with pytest.raises(SameClassRealizationRepairError, match="re-encode exact packet"):
        replace(compiled, program=other_program)

    foreign_source = replace(
        compiled.admission_receipt,
        source_binding_sha256="f" * 64,
    )
    with pytest.raises(SameClassRealizationRepairError, match="exact compiled source"):
        replace(compiled, admission_receipt=foreign_source)

    foreign_pairs = replace(
        compiled.admission_receipt,
        source_pair_ids=(PAIR_ID + 1,),
    )
    with pytest.raises(SameClassRealizationRepairError, match="exact compiled source"):
        replace(compiled, admission_receipt=foreign_pairs)

    wrong_counts = replace(
        compiled.admission_receipt,
        run_count=7,
        admitted_cell_count=7,
    )
    with pytest.raises(SameClassRealizationRepairError, match="compiled program"):
        replace(compiled, admission_receipt=wrong_counts)


def test_decoded_and_compiled_objects_close_ownership_counts_and_exact_mask(
    compiled: CompiledSameClassRealizationRepairV1,
) -> None:
    receipt = compiled.decoded.receipt
    wrong_camera_count = replace(
        receipt,
        owned_camera_values=receipt.owned_camera_values + 3,
    )
    with pytest.raises(SameClassRealizationRepairError, match="camera ownership count differs from mask"):
        replace(compiled.decoded, receipt=wrong_camera_count)

    moved_mask = compiled.decoded.owned_scorer_mask.copy()
    moved_mask[0, RUN_ROW_INDEX, RUN_COL_START] = False
    moved_mask[0, 0, 0] = True
    moved_receipt = replace(
        receipt,
        owned_scorer_mask_sha256=_array_sha256(moved_mask),
    )
    moved_decoded = replace(
        compiled.decoded,
        owned_scorer_mask=moved_mask,
        receipt=moved_receipt,
    )
    moved_admission = replace(
        compiled.admission_receipt,
        decode_receipt_sha256=moved_receipt.receipt_sha256,
    )
    with pytest.raises(SameClassRealizationRepairError, match="mask differs from packet program"):
        replace(
            compiled,
            decoded=moved_decoded,
            admission_receipt=moved_admission,
        )
