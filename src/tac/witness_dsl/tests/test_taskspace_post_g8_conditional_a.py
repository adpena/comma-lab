# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    ReceiverRealizationProfileV1,
    TopologyEventV1,
)
from tac.witness_dsl.generative_taskspace_correction import (
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CorrectedY1SupportCopyCellV1,
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Mode,
    PredictorPreservingA3ProgramV1,
    SparseConstantRGBCellV1,
)
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    PACKET_HEADER,
    PACKET_MAGIC,
    PostG8ConditionalAError,
    build_post_g8_conditional_a_surface,
    compile_post_g8_conditional_a,
    decode_post_g8_conditional_a_packet,
    derive_post_g8_conditional_a_source_binding,
    parse_post_g8_conditional_a_packet,
    parse_post_g8_conditional_a_receipt,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    compile_generative_taskspace_correction_v2,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    EncoderOnlyExactTargetLabelCustodyV1,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairRunV1,
    SameClassSemanticRoleV1,
    compile_same_class_realization_repair_for_pga_sections,
    decode_same_class_realization_repair_from_pga_sections,
)

PAIR_ID = 29
REPAIR_ROW = 123
REPAIR_COL_START = 100
REPAIR_COL_STOP = 108


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class _Fixture:
    predictor_packet: bytes
    semantic_g_packet: bytes
    predictor_surface: PredictorCameraPairSurfaceV1
    integration: object


def _repair_program(rgb: tuple[int, int, int]) -> SameClassRealizationRepairProgramV1:
    return SameClassRealizationRepairProgramV1(
        (
            SameClassRealizationRepairRunV1(
                source_pair_id=PAIR_ID,
                scorer_row=REPAIR_ROW,
                scorer_col_start=REPAIR_COL_START,
                scorer_col_stop=REPAIR_COL_STOP,
                semantic_class=0,
                semantic_role=SameClassSemanticRoleV1.ROAD,
                rgb_u8=rgb,
            ),
        )
    )


@cache
def _fixture(repair_rgb: tuple[int, int, int] = (90, 100, 110)) -> _Fixture:
    predictor_packet = b"LVLS1\x00synthetic-post-g8-counted-predictor"
    labels = np.zeros((1, 384, 512), dtype=np.uint8)
    state = TaskspacePredictorStateV2(
        predictor_program=predictor_packet,
        predictor_renderer_sha256="1" * 64,
        source_archive_sha256="2" * 64,
        source_runtime_sha256="3" * 64,
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
        upstream_decode_receipt_sha256="4" * 64,
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
    teacher = EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="5" * 64,
        pbr2_sha256="6" * 64,
        target_labels_sha256=_array_sha256(g_target),
        obligation_ir_sha256="7" * 64,
        oracle_evidence_sha256="8" * 64,
        dense_y_sha256="9" * 64,
        target_labels=g_target,
        teacher_event_count=8,
    )
    compiled_g = compile_generative_taskspace_correction_v2(
        state,
        GenerativeCorrectionProgramV1(
            topology_events=(TopologyEventV1(PAIR_ID, "Lane", "birth", "box", 1, 10, 10, 12, 14),),
            realization_profile=profile,
        ),
        teacher_evidence=teacher,
    )
    semantic_g_packet = compiled_g.packet
    target = EncoderOnlyExactTargetLabelCustodyV1(
        source_artifact_sha256="a" * 64,
        source_member_name="gt_n600.npz::lstars",
        source_member_sha256="b" * 64,
        source_pair_ids=(PAIR_ID,),
        target_labels_sha256=_array_sha256(compiled_g.decoded.labels),
        target_labels=compiled_g.decoded.labels,
    )
    proposal = compile_same_class_realization_repair_for_pga_sections(
        _repair_program(repair_rgb),
        predictor_section_payload=predictor_packet,
        semantic_g_section_payload=semantic_g_packet,
        predictor_surface=predictor_surface,
        predictor_codec_id="LVLS1.v1",
        semantic_g_codec_id="TACG1C.v2",
        target_custody=target,
    )
    integration = decode_same_class_realization_repair_from_pga_sections(
        proposal.replacement_g_section_payload,
        predictor_section_payload=predictor_packet,
        predictor_surface=predictor_surface,
        predictor_codec_id="LVLS1.v1",
        semantic_g_codec_id="TACG1C.v2",
    )
    return _Fixture(
        predictor_packet=predictor_packet,
        semantic_g_packet=semantic_g_packet,
        predictor_surface=predictor_surface,
        integration=integration,
    )


def _programs() -> tuple[PredictorPreservingA3ProgramV1, ...]:
    return (
        PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1),
        PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
            constant_rgb_cells=(SparseConstantRGBCellV1(PAIR_ID, REPAIR_ROW, REPAIR_COL_START, (1, 2, 3)),),
        ),
        PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1,
            corrected_y1_copy_cells=(CorrectedY1SupportCopyCellV1(PAIR_ID, REPAIR_ROW, REPAIR_COL_START),),
        ),
    )


@pytest.mark.parametrize("program", _programs(), ids=lambda row: row.mode.value)
def test_all_canonical_a3_program_modes_compile_in_distinct_post_g8_domain(
    program: PredictorPreservingA3ProgramV1,
) -> None:
    fixture = _fixture()
    compiled = compile_post_g8_conditional_a(
        program,
        predictor_surface=fixture.predictor_surface,
        g8_integration=fixture.integration,  # type: ignore[arg-type]
    )
    source = derive_post_g8_conditional_a_source_binding(
        build_post_g8_conditional_a_surface(
            predictor_surface=fixture.predictor_surface,
            g8_integration=fixture.integration,  # type: ignore[arg-type]
        )
    )
    parsed = parse_post_g8_conditional_a_packet(
        compiled.packet,
        expected_source_binding=source,
    )
    decoded = decode_post_g8_conditional_a_packet(
        compiled.packet,
        predictor_surface=fixture.predictor_surface,
        g8_integration=fixture.integration,  # type: ignore[arg-type]
    )

    assert compiled.packet.startswith(PACKET_MAGIC)
    assert parsed.program == program
    assert parsed.packet == compiled.packet
    assert compiled.source_binding == source
    assert compiled.receipt == decoded.receipt
    assert decoded.receipt.packet_sha256 == hashlib.sha256(compiled.packet).hexdigest()
    assert decoded.receipt.composite_g_section_sha256 == (
        fixture.integration.parsed_composite_g_section.section_sha256  # type: ignore[union-attr]
    )
    assert decoded.receipt.repair_receipt_sha256 == (
        fixture.integration.decoded_repair.receipt.receipt_sha256  # type: ignore[union-attr]
    )
    assert decoded.receipt.pre_g8_corrected_y1_sha256 != decoded.receipt.post_g8_corrected_y1_sha256
    assert np.array_equal(decoded.camera_y1, fixture.integration.post_g8_corrected_y1)  # type: ignore[union-attr]
    assert np.array_equal(decoded.chronological_camera_frames[:, 0], decoded.camera_y0)
    assert np.array_equal(decoded.chronological_camera_frames[:, 1], decoded.camera_y1)
    assert decoded.camera_y0.flags.writeable is False
    assert decoded.camera_y1.flags.writeable is False
    assert decoded.receipt.post_g8_y1_unchanged is True
    assert decoded.receipt.g8_semantic_labels_unchanged is True
    assert decoded.receipt.canonical_disjoint_reconstruction_reused is True
    assert decoded.receipt.scorer_invoked is False
    assert decoded.receipt.n600_evidence_claim is False
    assert decoded.receipt.exact_score_claim is False
    assert decoded.receipt.candidate_archive_eligible is False
    assert decoded.receipt.through_r_target_realization_debt_closed is False
    assert parse_post_g8_conditional_a_receipt(decoded.receipt.to_receipt_bytes()) == decoded.receipt


def test_decoder_calls_canonical_reconstruction_and_preserves_outside_owned_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.witness_dsl.taskspace_post_g8_conditional_a as module

    fixture = _fixture()
    compiled = compile_post_g8_conditional_a(
        _programs()[1],
        predictor_surface=fixture.predictor_surface,
        g8_integration=fixture.integration,  # type: ignore[arg-type]
    )
    calls = {"canonical": 0}
    original = module.apply_disjoint_camera_cell_reconstruction_v1

    def counted(*args: object, **kwargs: object):
        calls["canonical"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "apply_disjoint_camera_cell_reconstruction_v1", counted)
    decoded = decode_post_g8_conditional_a_packet(
        compiled.packet,
        predictor_surface=fixture.predictor_surface,
        g8_integration=fixture.integration,  # type: ignore[arg-type]
    )
    assert calls == {"canonical": 2}
    assert np.array_equal(
        decoded.camera_y0[~decoded.owned_camera_mask],
        fixture.predictor_surface.camera_p0[~decoded.owned_camera_mask],
    )


def test_packet_deletion_crc_mutation_trailing_and_foreign_post_g8_source_fail_closed() -> None:
    fixture = _fixture()
    compiled = compile_post_g8_conditional_a(
        _programs()[1],
        predictor_surface=fixture.predictor_surface,
        g8_integration=fixture.integration,  # type: ignore[arg-type]
    )
    with pytest.raises(PostG8ConditionalAError, match="shorter"):
        decode_post_g8_conditional_a_packet(
            b"",
            predictor_surface=fixture.predictor_surface,
            g8_integration=fixture.integration,  # type: ignore[arg-type]
        )
    mutated = compiled.packet[:-1] + bytes([compiled.packet[-1] ^ 1])
    with pytest.raises(PostG8ConditionalAError, match="CRC"):
        decode_post_g8_conditional_a_packet(
            mutated,
            predictor_surface=fixture.predictor_surface,
            g8_integration=fixture.integration,  # type: ignore[arg-type]
        )
    with pytest.raises(PostG8ConditionalAError, match=r"length accounting|trailing"):
        decode_post_g8_conditional_a_packet(
            compiled.packet + b"\x00",
            predictor_surface=fixture.predictor_surface,
            g8_integration=fixture.integration,  # type: ignore[arg-type]
        )

    foreign = _fixture((91, 101, 111))
    with pytest.raises(PostG8ConditionalAError, match="source binding"):
        decode_post_g8_conditional_a_packet(
            compiled.packet,
            predictor_surface=foreign.predictor_surface,
            g8_integration=foreign.integration,  # type: ignore[arg-type]
        )


def test_valid_counted_row_mutation_is_receiver_causal() -> None:
    fixture = _fixture()
    first = compile_post_g8_conditional_a(
        _programs()[1],
        predictor_surface=fixture.predictor_surface,
        g8_integration=fixture.integration,  # type: ignore[arg-type]
    )
    changed_program = PredictorPreservingA3ProgramV1(
        PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
        constant_rgb_cells=(SparseConstantRGBCellV1(PAIR_ID, REPAIR_ROW, REPAIR_COL_START, (2, 3, 4)),),
    )
    second = compile_post_g8_conditional_a(
        changed_program,
        predictor_surface=fixture.predictor_surface,
        g8_integration=fixture.integration,  # type: ignore[arg-type]
    )
    assert first.packet != second.packet
    assert first.receipt.packet_sha256 != second.receipt.packet_sha256
    assert not np.array_equal(first.decoded.camera_y0, second.decoded.camera_y0)
    assert np.array_equal(first.decoded.camera_y1, second.decoded.camera_y1)


def test_receipt_rejects_truth_numeric_and_canonical_smuggling() -> None:
    fixture = _fixture()
    receipt = compile_post_g8_conditional_a(
        _programs()[1],
        predictor_surface=fixture.predictor_surface,
        g8_integration=fixture.integration,  # type: ignore[arg-type]
    ).receipt
    body = json.loads(receipt.to_receipt_bytes())
    body["post_g8_y1_unchanged"] = False
    forged_truth = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(PostG8ConditionalAError, match="truth labels"):
        parse_post_g8_conditional_a_receipt(forged_truth)

    body = json.loads(receipt.to_receipt_bytes())
    body["packet_bytes"] = float(body["packet_bytes"])
    forged_number = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(PostG8ConditionalAError, match="exact integers"):
        parse_post_g8_conditional_a_receipt(forged_number)

    with pytest.raises(PostG8ConditionalAError, match="canonical"):
        parse_post_g8_conditional_a_receipt(receipt.to_receipt_bytes() + b"\n")


def test_packet_header_is_counted_and_versioned_without_legacy_overlay_receipt_semantics() -> None:
    fixture = _fixture()
    compiled = compile_post_g8_conditional_a(
        _programs()[0],
        predictor_surface=fixture.predictor_surface,
        g8_integration=fixture.integration,  # type: ignore[arg-type]
    )
    assert PACKET_HEADER.size == 62
    assert len(compiled.packet) == PACKET_HEADER.size
    assert compiled.receipt.packet_bytes == PACKET_HEADER.size
    assert "overlay_receipt_sha256" not in compiled.receipt.as_dict()
    assert compiled.receipt.repair_receipt_sha256 == (
        fixture.integration.decoded_repair.receipt.receipt_sha256  # type: ignore[union-attr]
    )
