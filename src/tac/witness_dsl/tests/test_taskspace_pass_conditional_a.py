# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest

from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CorrectedY1SupportCopyCellV1,
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Mode,
    PredictorPreservingA3ProgramV1,
    SparseConstantRGBCellV1,
)
from tac.witness_dsl.taskspace_pass_conditional_a import (
    PACKET_HEADER,
    PACKET_MAGIC,
    PassConditionalAError,
    build_pass_conditional_a_surface,
    compile_pass_conditional_a,
    decode_pass_conditional_a_packet,
    derive_pass_conditional_a_source_binding,
    parse_pass_conditional_a_packet,
    parse_pass_conditional_a_receipt,
)
from tac.witness_dsl.taskspace_pass_semantic_g import (
    PassSemanticGEnvelopeMode,
    compile_pass_semantic_g_envelope,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    EncoderOnlyExactTargetLabelCustodyV1,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairRunV1,
    SameClassSemanticRoleV1,
)

PAIR_ID = 29
ROW = 123
COL = 100


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class _Fixture:
    predictor_packet: bytes
    predictor_surface: PredictorCameraPairSurfaceV1
    repair_program: SameClassRealizationRepairProgramV1
    target_custody: EncoderOnlyExactTargetLabelCustodyV1


def _fixture(*, frame1_delta: int = 0) -> _Fixture:
    packet = b"LVLS1\x00synthetic-pass-a-counted-predictor"
    labels = np.zeros((1, 384, 512), dtype=np.uint8)
    state = TaskspacePredictorStateV2(
        predictor_program=packet,
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
    frames[:, 1] = ((10 + frame1_delta) % 256, 20, 30)
    predictor_surface = PredictorCameraPairSurfaceV1(
        predictor_state=state,
        chronological_camera_frames=frames,
        upstream_decode_receipt_sha256="4" * 64,
    )
    repair_program = SameClassRealizationRepairProgramV1(
        (
            SameClassRealizationRepairRunV1(
                source_pair_id=PAIR_ID,
                scorer_row=ROW,
                scorer_col_start=COL,
                scorer_col_stop=COL + 8,
                semantic_class=0,
                semantic_role=SameClassSemanticRoleV1.ROAD,
                rgb_u8=(90, 100, 110),
            ),
        )
    )
    target_custody = EncoderOnlyExactTargetLabelCustodyV1(
        source_artifact_sha256="a" * 64,
        source_member_name="gt_n600.npz::lstars",
        source_member_sha256="b" * 64,
        source_pair_ids=(PAIR_ID,),
        target_labels_sha256=_sha256(memoryview(labels).cast("B")),
        target_labels=labels,
    )
    return _Fixture(packet, predictor_surface, repair_program, target_custody)


def _pass_g(fixture: _Fixture, *, with_g8: bool):
    return compile_pass_semantic_g_envelope(
        predictor_section_payload=fixture.predictor_packet,
        predictor_surface=fixture.predictor_surface,
        repair_program=fixture.repair_program if with_g8 else None,
        target_custody=fixture.target_custody if with_g8 else None,
    ).decoded


def _programs() -> tuple[PredictorPreservingA3ProgramV1, ...]:
    return (
        PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1),
        PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
            constant_rgb_cells=(SparseConstantRGBCellV1(PAIR_ID, ROW, COL, (1, 2, 3)),),
        ),
        PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1,
            corrected_y1_copy_cells=(CorrectedY1SupportCopyCellV1(PAIR_ID, ROW, COL),),
        ),
    )


@pytest.mark.parametrize("with_g8", (False, True), ids=("pass-no-g8", "pass-then-g8"))
@pytest.mark.parametrize("program", _programs(), ids=lambda program: program.mode.value)
def test_all_a_modes_bind_exact_pass_envelope_and_preserve_resulting_y1(
    with_g8: bool,
    program: PredictorPreservingA3ProgramV1,
) -> None:
    fixture = _fixture()
    pass_g = _pass_g(fixture, with_g8=with_g8)
    compiled = compile_pass_conditional_a(
        program,
        predictor_surface=fixture.predictor_surface,
        pass_g=pass_g,
    )
    decoded = decode_pass_conditional_a_packet(
        compiled.packet,
        predictor_surface=fixture.predictor_surface,
        pass_g=pass_g,
    )
    source = derive_pass_conditional_a_source_binding(
        build_pass_conditional_a_surface(
            predictor_surface=fixture.predictor_surface,
            pass_g=pass_g,
        )
    )
    parsed = parse_pass_conditional_a_packet(compiled.packet, expected_source_binding=source)

    assert compiled.packet.startswith(PACKET_MAGIC)
    assert parsed.program == program
    assert compiled.source_binding == source
    assert compiled.decoded.receipt == decoded.receipt
    assert source.pass_g_mode == (
        PassSemanticGEnvelopeMode.PASS_THEN_G8_V1.value if with_g8 else PassSemanticGEnvelopeMode.PASS_NO_G8_V1.value
    )
    assert (source.repair_packet_sha256 is not None) is with_g8
    assert np.array_equal(decoded.camera_y1, pass_g.conditional_camera_y1)
    assert np.array_equal(decoded.chronological_camera_frames[:, 1], pass_g.conditional_camera_y1)
    assert decoded.receipt.conditional_y1_unchanged_by_a is True
    assert decoded.receipt.counted_pass_semantic_g_nonempty is True
    assert decoded.receipt.optional_g8_mode_explicit is True
    assert decoded.receipt.scorer_invoked is False
    assert decoded.receipt.exact_score_claim is False
    assert decoded.receipt.through_r_target_realization_debt_closed is False
    assert parse_pass_conditional_a_receipt(decoded.receipt.to_receipt_bytes()) == decoded.receipt


def test_no_g8_control_is_nonempty_source_bound_pass_a_not_fake_repair() -> None:
    fixture = _fixture()
    pass_g = _pass_g(fixture, with_g8=False)
    compiled = compile_pass_conditional_a(
        PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1),
        predictor_surface=fixture.predictor_surface,
        pass_g=pass_g,
    )
    assert len(compiled.packet) == PACKET_HEADER.size
    assert compiled.source_binding.repair_packet_sha256 is None
    assert compiled.source_binding.repair_receipt_sha256 is None
    assert compiled.source_binding.pre_repair_camera_y1_sha256 == (compiled.source_binding.conditional_camera_y1_sha256)
    assert np.array_equal(compiled.decoded.camera_y0, fixture.predictor_surface.camera_p0)
    assert np.array_equal(compiled.decoded.camera_y1, fixture.predictor_surface.camera_p1)


def test_foreign_same_pair_p_or_envelope_crc_and_trailing_bytes_fail_closed() -> None:
    fixture = _fixture()
    pass_g = _pass_g(fixture, with_g8=False)
    compiled = compile_pass_conditional_a(
        _programs()[1],
        predictor_surface=fixture.predictor_surface,
        pass_g=pass_g,
    )
    foreign = _fixture(frame1_delta=1)
    foreign_pass_g = _pass_g(foreign, with_g8=False)
    with pytest.raises(PassConditionalAError, match="source binding"):
        decode_pass_conditional_a_packet(
            compiled.packet,
            predictor_surface=foreign.predictor_surface,
            pass_g=foreign_pass_g,
        )

    mutated = compiled.packet[:-1] + bytes([compiled.packet[-1] ^ 1])
    with pytest.raises(PassConditionalAError, match="CRC"):
        decode_pass_conditional_a_packet(
            mutated,
            predictor_surface=fixture.predictor_surface,
            pass_g=pass_g,
        )
    with pytest.raises(PassConditionalAError, match=r"length accounting|trailing"):
        decode_pass_conditional_a_packet(
            compiled.packet + b"x",
            predictor_surface=fixture.predictor_surface,
            pass_g=pass_g,
        )


def test_receipt_rejects_truth_numeric_duplicate_and_noncanonical_smuggling() -> None:
    fixture = _fixture()
    compiled = compile_pass_conditional_a(
        _programs()[0],
        predictor_surface=fixture.predictor_surface,
        pass_g=_pass_g(fixture, with_g8=False),
    )
    body = compiled.decoded.receipt.as_dict()
    body["research_only"] = False
    forged_truth = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(PassConditionalAError, match="truth labels"):
        parse_pass_conditional_a_receipt(forged_truth)

    body = compiled.decoded.receipt.as_dict()
    body["packet_bytes"] = float(body["packet_bytes"])
    forged_number = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(PassConditionalAError, match="exact integers"):
        parse_pass_conditional_a_receipt(forged_number)

    receipt = compiled.decoded.receipt.to_receipt_bytes()
    duplicate = receipt[:-1] + b',"schema":"tac.taskspace_pass_conditional_a_receipt.v1"}'
    with pytest.raises(PassConditionalAError, match="repeats key"):
        parse_pass_conditional_a_receipt(duplicate)
    with pytest.raises(PassConditionalAError, match="canonical"):
        parse_pass_conditional_a_receipt(receipt + b"\n")
