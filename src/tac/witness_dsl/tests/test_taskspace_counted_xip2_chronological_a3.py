# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import inspect
import json
import struct
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pytest

from tac.boundary_math.xi_pose_coder import quantize_xi, serialize_xi_payload
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    PredictorCameraPairSurfaceV1,
)
from tac.witness_dsl.taskspace_chronological_a3_encoder import (
    EncoderOnlyA3TargetV1,
    EncoderOnlyXIP2GuidanceV1,
    XIP2Coder,
)
from tac.witness_dsl.taskspace_counted_xip2_chronological_a3 import (
    ABSENT_GUIDANCE_SHA256,
    PACKET_FOOTER,
    PACKET_HEADER,
    PACKET_MAGIC,
    CompiledCountedXIP2ChronologicalA3V1,
    CountedXIP2A3Interpretation,
    CountedXIP2A3Mode,
    CountedXIP2ChronologicalA3Error,
    CountedXIP2ChronologicalA3ProgramV1,
    compile_counted_xip2_chronological_a3,
    compile_counted_xip2_chronological_a3_from_guidance,
    compile_counted_xip2_chronological_a3_pass,
    decode_counted_xip2_chronological_a3_packet,
    parse_counted_xip2_chronological_a3_packet,
    parse_counted_xip2_chronological_a3_receipt,
    reencode_counted_xip2_chronological_a3_packet,
    resume_counted_xip2_chronological_a3,
)
from tac.witness_dsl.taskspace_pass_conditional_a import (
    PassConditionalASourceBindingV1,
    build_pass_conditional_a_surface,
    derive_pass_conditional_a_source_binding,
)
from tac.witness_dsl.taskspace_pass_semantic_g import compile_pass_semantic_g_envelope
from tac.witness_dsl.taskspace_post_g8_conditional_a import PostG8ConditionalASourceBindingV1
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    EncoderOnlyExactTargetLabelCustodyV1,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairRunV1,
    SameClassSemanticRoleV1,
)

PAIR_ID = 37
REPAIR_ROW = 123
REPAIR_COL = 100


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


@lru_cache(maxsize=4)
def _fixture(*, frame1_delta: int = 0) -> _Fixture:
    predictor_packet = b"LVLS1\x00g13-counted-xip2-synthetic-predictor"
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
    frames = np.empty((1, 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS), dtype=np.uint8)
    frames[:, 0] = (5, 15, 25)
    rows = np.arange(CAMERA_HEIGHT, dtype=np.uint16)[:, None]
    cols = np.arange(CAMERA_WIDTH, dtype=np.uint16)[None, :]
    frames[0, 1, :, :, 0] = ((rows + 3 * cols + frame1_delta) % 256).astype(np.uint8)
    frames[0, 1, :, :, 1] = ((2 * rows + cols + 17) % 256).astype(np.uint8)
    frames[0, 1, :, :, 2] = ((rows + cols + 53) % 256).astype(np.uint8)
    predictor_surface = PredictorCameraPairSurfaceV1(
        predictor_state=state,
        chronological_camera_frames=frames,
        upstream_decode_receipt_sha256="4" * 64,
    )
    repair_program = SameClassRealizationRepairProgramV1(
        (
            SameClassRealizationRepairRunV1(
                source_pair_id=PAIR_ID,
                scorer_row=REPAIR_ROW,
                scorer_col_start=REPAIR_COL,
                scorer_col_stop=REPAIR_COL + 8,
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
    return _Fixture(predictor_packet, predictor_surface, repair_program, target_custody)


@lru_cache(maxsize=4)
def _surface(*, with_g8: bool, frame1_delta: int = 0):
    fixture = _fixture(frame1_delta=frame1_delta)
    pass_g = compile_pass_semantic_g_envelope(
        predictor_section_payload=fixture.predictor_packet,
        predictor_surface=fixture.predictor_surface,
        repair_program=fixture.repair_program if with_g8 else None,
        target_custody=fixture.target_custody if with_g8 else None,
    ).decoded
    return build_pass_conditional_a_surface(
        predictor_surface=fixture.predictor_surface,
        pass_g=pass_g,
    )


def _xip2_payload(*, pairs: int = 1) -> bytes:
    xi = np.tile(
        np.array([[0.016, -0.003, 0.002, 0.001, 0.006, -0.009]], dtype=np.float64),
        (pairs, 1),
    )
    if pairs > 1:
        xi[:, 0] += np.arange(pairs, dtype=np.float64) * 0.001
    q, scales = quantize_xi(xi, q_levels=2047)
    return serialize_xi_payload(q, scales, coder="none")


def _active_program(
    interpretation: CountedXIP2A3Interpretation,
    *,
    payload: bytes | None = None,
) -> CountedXIP2ChronologicalA3ProgramV1:
    return CountedXIP2ChronologicalA3ProgramV1(
        mode=CountedXIP2A3Mode.XIP2_WARP_V1,
        interpretation=interpretation,
        pitch=0.03125,
        xip2_payload=_xip2_payload() if payload is None else payload,
        guidance_source_binding_sha256="c" * 64,
    )


@pytest.mark.parametrize("with_g8", (False, True), ids=("pass-no-g8", "pass-then-g8"))
def test_pass_and_global_copy_are_exact_empty_body_chronological_programs(with_g8: bool) -> None:
    surface = _surface(with_g8=with_g8)
    pass_row = compile_counted_xip2_chronological_a3_pass(surface=surface)
    copy_row = compile_counted_xip2_chronological_a3_pass(
        surface=surface,
        mode=CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1,
    )

    assert pass_row.packet.startswith(PACKET_MAGIC)
    assert len(pass_row.packet) == PACKET_HEADER.size + PACKET_FOOTER.size
    assert len(copy_row.packet) == PACKET_HEADER.size + PACKET_FOOTER.size
    assert pass_row.packet != copy_row.packet
    assert pass_row.parsed_packet.xip2_payload == b""
    assert copy_row.parsed_packet.xip2_payload == b""
    assert pass_row.receipt.xip2_payload_bytes == 0
    assert copy_row.receipt.xip2_payload_bytes == 0
    assert pass_row.receipt.guidance_source_binding_sha256 == ABSENT_GUIDANCE_SHA256
    assert copy_row.receipt.guidance_source_binding_sha256 == ABSENT_GUIDANCE_SHA256
    assert np.array_equal(pass_row.decoded.camera_y0, surface.predictor_camera_p0)
    assert np.array_equal(copy_row.decoded.camera_y0, surface.conditional_camera_y1)
    for compiled in (pass_row, copy_row):
        assert np.array_equal(compiled.decoded.camera_y1, surface.conditional_camera_y1)
        assert np.array_equal(compiled.decoded.chronological_camera_frames[:, 0], compiled.decoded.camera_y0)
        assert np.array_equal(compiled.decoded.chronological_camera_frames[:, 1], surface.conditional_camera_y1)
        assert compiled.receipt.raw_counted_bytes == compiled.receipt.packet_bytes
        assert compiled.receipt.packet_bytes == (
            compiled.receipt.header_bytes + compiled.receipt.xip2_payload_bytes + compiled.receipt.footer_bytes
        )
        assert compiled.receipt.dense_camera_y1_serialized is False
        assert compiled.receipt.per_cell_coordinates_serialized is False


def test_copy_and_active_packets_are_bound_to_exact_pass_g_mode_and_foreign_source() -> None:
    no_g8 = _surface(with_g8=False)
    with_g8 = _surface(with_g8=True)
    copy_row = compile_counted_xip2_chronological_a3_pass(
        surface=no_g8,
        mode=CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1,
    )
    active = compile_counted_xip2_chronological_a3(
        _active_program(CountedXIP2A3Interpretation.CAMERA_THEN_R),
        surface=no_g8,
    )
    for packet in (copy_row.packet, active.packet):
        with pytest.raises(CountedXIP2ChronologicalA3Error, match="source binding"):
            decode_counted_xip2_chronological_a3_packet(packet, surface=with_g8)

    foreign = _surface(with_g8=False, frame1_delta=1)
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="source binding"):
        decode_counted_xip2_chronological_a3_packet(copy_row.packet, surface=foreign)


def test_both_interpretations_share_xip2_bytes_but_are_non_aliasing_real_receivers() -> None:
    surface = _surface(with_g8=False)
    camera = compile_counted_xip2_chronological_a3(
        _active_program(CountedXIP2A3Interpretation.CAMERA_THEN_R),
        surface=surface,
    )
    scorer = compile_counted_xip2_chronological_a3(
        _active_program(CountedXIP2A3Interpretation.SCORER_THEN_FACTOR2),
        surface=surface,
    )

    assert camera.parsed_packet.xip2_payload == scorer.parsed_packet.xip2_payload
    assert camera.packet != scorer.packet
    assert camera.receipt.xip2_payload_bytes == len(_xip2_payload())
    assert scorer.receipt.xip2_payload_bytes == len(_xip2_payload())
    assert not np.array_equal(camera.decoded.camera_y0, scorer.decoded.camera_y0)
    assert np.array_equal(camera.decoded.camera_y1, surface.conditional_camera_y1)
    assert np.array_equal(scorer.decoded.camera_y1, surface.conditional_camera_y1)
    assert camera.decoded.camera_y0.dtype == np.uint8
    assert scorer.decoded.camera_y0.dtype == np.uint8


def test_outer_parse_reencode_raw_accounting_and_strict_receipt_round_trip() -> None:
    surface = _surface(with_g8=False)
    compiled = compile_counted_xip2_chronological_a3(
        _active_program(CountedXIP2A3Interpretation.CAMERA_THEN_R),
        surface=surface,
    )
    source = derive_pass_conditional_a_source_binding(surface)
    parsed = parse_counted_xip2_chronological_a3_packet(
        compiled.packet,
        expected_source_binding=source,
    )
    assert (
        reencode_counted_xip2_chronological_a3_packet(
            parsed,
            expected_source_binding=source,
        )
        == compiled.packet
    )
    assert compiled.receipt.packet_bytes == PACKET_HEADER.size + len(_xip2_payload()) + PACKET_FOOTER.size
    assert compiled.receipt.protocol_overhead_bytes == PACKET_HEADER.size + PACKET_FOOTER.size
    assert compiled.receipt.raw_counted_bytes == len(compiled.packet)
    assert parse_counted_xip2_chronological_a3_receipt(compiled.receipt.to_receipt_bytes()) == compiled.receipt


def test_deletion_truncation_crc_header_and_trailing_corruption_fail_closed() -> None:
    surface = _surface(with_g8=False)
    compiled = compile_counted_xip2_chronological_a3(
        _active_program(CountedXIP2A3Interpretation.CAMERA_THEN_R),
        surface=surface,
    )
    source = derive_pass_conditional_a_source_binding(surface)
    mutations = (
        compiled.packet[:-1],
        compiled.packet[:20],
        compiled.packet[:-1] + bytes([compiled.packet[-1] ^ 1]),
        compiled.packet[:9] + bytes([compiled.packet[9] ^ 0x7F]) + compiled.packet[10:],
        compiled.packet + b"trailing",
    )
    for packet in mutations:
        with pytest.raises(CountedXIP2ChronologicalA3Error):
            parse_counted_xip2_chronological_a3_packet(
                packet,
                expected_source_binding=source,
            )


def test_nested_xip2_internal_trailing_pair_mismatch_and_body_mutation_fail_closed() -> None:
    surface = _surface(with_g8=False)
    for payload in (
        _xip2_payload() + b"nested-trailing",
        _xip2_payload(pairs=2),
        b"XIP2\x00",
    ):
        with pytest.raises(CountedXIP2ChronologicalA3Error, match=r"XIP2|nested"):
            compile_counted_xip2_chronological_a3(
                _active_program(CountedXIP2A3Interpretation.CAMERA_THEN_R, payload=payload),
                surface=surface,
            )


def test_diagnostic_post_g8_source_type_is_unrepresentable_in_v1() -> None:
    surface = _surface(with_g8=False)
    compiled = compile_counted_xip2_chronological_a3_pass(surface=surface)
    old_source = object.__new__(PostG8ConditionalASourceBindingV1)
    assert type(old_source) is PostG8ConditionalASourceBindingV1
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="PassConditionalASourceBindingV1"):
        parse_counted_xip2_chronological_a3_packet(
            compiled.packet,
            expected_source_binding=old_source,  # type: ignore[arg-type]
        )
    assert type(compiled.source_binding) is PassConditionalASourceBindingV1


def test_guidance_wrapper_consumes_only_counted_lineage_and_matches_direct_compile() -> None:
    surface = _surface(with_g8=False)
    payload = _xip2_payload()
    target = EncoderOnlyA3TargetV1(
        source_pair_ids=(PAIR_ID,),
        target_camera_y0=surface.conditional_camera_y1,
        custody_sha256="d" * 64,
        evidence_kind="xip2_camera_then_r_guidance_encoder_only.v1",
    )
    scorer_target = EncoderOnlyA3TargetV1(
        source_pair_ids=(PAIR_ID,),
        target_camera_y0=surface.predictor_camera_p0,
        custody_sha256="e" * 64,
        evidence_kind="xip2_scorer_then_factor2_guidance_encoder_only.v1",
    )
    guidance = EncoderOnlyXIP2GuidanceV1(
        source_binding_sha256="c" * 64,
        source_pair_ids=(PAIR_ID,),
        coder=XIP2Coder.NONE,
        q_levels=2047,
        pitch=0.03125,
        xip2_payload=payload,
        camera_then_r=target,
        scorer_then_factor2=scorer_target,
    )
    wrapped = compile_counted_xip2_chronological_a3_from_guidance(
        guidance,
        interpretation=CountedXIP2A3Interpretation.CAMERA_THEN_R,
        surface=surface,
    )
    direct = compile_counted_xip2_chronological_a3(
        _active_program(CountedXIP2A3Interpretation.CAMERA_THEN_R, payload=payload),
        surface=surface,
    )
    assert wrapped.packet == direct.packet
    assert np.array_equal(wrapped.decoded.camera_y0, direct.decoded.camera_y0)
    assert wrapped.receipt.guidance_binding_role == "OPAQUE_ENCODER_LINEAGE_ONLY"
    assert wrapped.receipt.guidance_binding_receiver_verified is False
    assert wrapped.receipt.guidance_binding_source_authority is False
    assert wrapped.receipt.source_binding_sha256 == derive_pass_conditional_a_source_binding(surface).binding_sha256
    assert wrapped.receipt.guidance_source_binding_sha256 != wrapped.receipt.source_binding_sha256


@pytest.mark.parametrize(
    "mode",
    (
        CountedXIP2A3Mode.PASS_P0_V1,
        CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1,
    ),
)
def test_empty_modes_have_one_canonical_encoding(mode: CountedXIP2A3Mode) -> None:
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="canonical"):
        CountedXIP2ChronologicalA3ProgramV1(
            mode=mode,
            interpretation=CountedXIP2A3Interpretation.SCORER_THEN_FACTOR2,
        )
    with pytest.raises(CountedXIP2ChronologicalA3Error, match=r"alias|canonical"):
        CountedXIP2ChronologicalA3ProgramV1(mode=mode, pitch=-0.0)
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="canonical"):
        CountedXIP2ChronologicalA3ProgramV1(mode=mode, xip2_payload=b"x")


def test_decode_is_deterministic_immutable_y1_preserving_and_resumable_by_exact_hash() -> None:
    surface = _surface(with_g8=True)
    compiled = compile_counted_xip2_chronological_a3(
        _active_program(CountedXIP2A3Interpretation.SCORER_THEN_FACTOR2),
        surface=surface,
    )
    replay = decode_counted_xip2_chronological_a3_packet(compiled.packet, surface=surface)
    resumed = resume_counted_xip2_chronological_a3(
        compiled.packet,
        expected_packet_sha256=compiled.receipt.packet_sha256,
        surface=surface,
    )
    assert np.array_equal(replay.camera_y0, resumed.camera_y0)
    assert np.array_equal(replay.camera_y1, surface.conditional_camera_y1)
    assert replay.receipt.to_receipt_bytes() == resumed.receipt.to_receipt_bytes()
    assert replay.camera_y0.flags.writeable is False
    assert replay.camera_y1.flags.writeable is False
    assert replay.chronological_camera_frames.flags.writeable is False
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="hash mismatch"):
        resume_counted_xip2_chronological_a3(
            compiled.packet,
            expected_packet_sha256="f" * 64,
            surface=surface,
        )


def test_receipt_rejects_truth_numeric_duplicate_and_noncanonical_smuggling() -> None:
    compiled = compile_counted_xip2_chronological_a3_pass(surface=_surface(with_g8=False))
    body = compiled.receipt.as_dict()
    body["research_only"] = False
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="truth labels"):
        parse_counted_xip2_chronological_a3_receipt(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
    body = compiled.receipt.as_dict()
    body["guidance_binding_receiver_verified"] = True
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="truth labels"):
        parse_counted_xip2_chronological_a3_receipt(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
    body = compiled.receipt.as_dict()
    body["packet_bytes"] = float(body["packet_bytes"])
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="exact integer"):
        parse_counted_xip2_chronological_a3_receipt(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
    receipt = compiled.receipt.to_receipt_bytes()
    duplicate = receipt[:-1] + b',"schema":"tac.taskspace_counted_xip2_chronological_a3_receipt.v1"}'
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="duplicate"):
        parse_counted_xip2_chronological_a3_receipt(duplicate)
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="canonical"):
        parse_counted_xip2_chronological_a3_receipt(receipt + b"\n")


def test_public_decoder_api_has_no_scorer_target_gt_or_dense_evidence_input() -> None:
    forbidden = ("scorer", "target", "label", "ground_truth", "gt", "dense", "y0", "y1")
    for function in (
        decode_counted_xip2_chronological_a3_packet,
        resume_counted_xip2_chronological_a3,
    ):
        names = tuple(inspect.signature(function).parameters)
        assert not any(token in name.lower() for name in names for token in forbidden)
    assert inspect.signature(decode_counted_xip2_chronological_a3_packet).parameters.keys() == {
        "packet",
        "surface",
    }


def test_packet_header_mode_byte_and_footer_are_inside_crc_custody() -> None:
    compiled = compile_counted_xip2_chronological_a3_pass(
        surface=_surface(with_g8=False),
        mode=CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1,
    )
    header = PACKET_HEADER.unpack_from(compiled.packet)
    assert header[0] == PACKET_MAGIC
    assert header[3] == 1
    assert struct.unpack(">I", compiled.packet[-PACKET_FOOTER.size :])[0] == compiled.receipt.packet_crc32
    assert isinstance(compiled, CompiledCountedXIP2ChronologicalA3V1)
