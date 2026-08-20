# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass

import numpy as np
import pytest

from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    MAX_BOUNDED_PAIRS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    PredictorCameraPairSurfaceV1,
)
from tac.witness_dsl.taskspace_pass_semantic_g import (
    ENVELOPE_HEADER,
    ENVELOPE_MAGIC,
    PASS_PACKET_HEADER,
    PASS_PACKET_MAGIC,
    PassSemanticGEnvelopeMode,
    PassSemanticGError,
    build_pass_semantic_g8_surface,
    compile_pass_semantic_g_envelope,
    compile_pass_semantic_g_packet,
    compose_pass_semantic_g_envelope,
    decode_pass_semantic_g_envelope,
    decode_pass_semantic_g_packet,
    parse_pass_semantic_g_envelope,
    parse_pass_semantic_g_envelope_receipt,
    parse_pass_semantic_g_receipt,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    EncoderOnlyExactTargetLabelCustodyV1,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairRunV1,
    SameClassSemanticRoleV1,
)

PAIR_ID = 29


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
    predictor_packet = b"LVLS1\x00synthetic-pass-semantic-counted-predictor"
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
    frames[:, 1] = ((10 + frame1_delta) % 256, 20, 30)
    surface = PredictorCameraPairSurfaceV1(
        predictor_state=state,
        chronological_camera_frames=frames,
        upstream_decode_receipt_sha256="4" * 64,
    )
    repair_program = SameClassRealizationRepairProgramV1(
        (
            SameClassRealizationRepairRunV1(
                source_pair_id=PAIR_ID,
                scorer_row=123,
                scorer_col_start=100,
                scorer_col_stop=108,
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
    return _Fixture(predictor_packet, surface, repair_program, target_custody)


def test_nonempty_pass_packet_is_source_bound_and_byte_identity_only() -> None:
    fixture = _fixture()
    packet, compiled = compile_pass_semantic_g_packet(predictor_surface=fixture.predictor_surface)
    decoded = decode_pass_semantic_g_packet(packet, predictor_surface=fixture.predictor_surface)

    assert packet.startswith(PASS_PACKET_MAGIC)
    assert len(packet) == PASS_PACKET_HEADER.size
    assert compiled.receipt == decoded.receipt
    assert np.array_equal(decoded.semantic_labels, fixture.predictor_surface.predictor_state.labels)
    assert np.array_equal(decoded.camera_y1, fixture.predictor_surface.camera_p1)
    assert decoded.semantic_labels.flags.writeable is False
    assert decoded.camera_y1.flags.writeable is False
    assert decoded.receipt.packet_nonempty_and_source_bound is True
    assert decoded.receipt.semantic_mutation_count == 0
    assert decoded.receipt.camera_mutation_count == 0
    assert decoded.receipt.scorer_invoked is False
    assert decoded.receipt.score_claim is False
    assert parse_pass_semantic_g_receipt(decoded.receipt.to_receipt_bytes()) == decoded.receipt


def test_pass_no_g8_is_explicit_nonempty_counted_envelope_and_preserves_p1() -> None:
    fixture = _fixture()
    compiled = compile_pass_semantic_g_envelope(
        predictor_section_payload=fixture.predictor_packet,
        predictor_surface=fixture.predictor_surface,
    )
    decoded = decode_pass_semantic_g_envelope(
        compiled.envelope,
        predictor_section_payload=fixture.predictor_packet,
        predictor_surface=fixture.predictor_surface,
    )

    assert compiled.envelope.startswith(ENVELOPE_MAGIC)
    assert compiled.mode is PassSemanticGEnvelopeMode.PASS_NO_G8_V1
    assert compiled.compiled_repair is None
    assert decoded.decoded_repair is None
    assert decoded.parsed_envelope.repair_packet is None
    assert decoded.receipt.repair_packet_sha256 is None
    assert decoded.receipt.repair_packet_bytes == 0
    assert decoded.receipt.output_camera_y1_sha256 == decoded.receipt.pre_repair_camera_y1_sha256
    assert np.array_equal(decoded.conditional_camera_y1, fixture.predictor_surface.camera_p1)
    assert parse_pass_semantic_g_envelope(compiled.envelope).envelope_bytes == compiled.envelope
    assert parse_pass_semantic_g_envelope_receipt(decoded.receipt.to_receipt_bytes()) == decoded.receipt


def test_pass_then_g8_applies_exact_repair_without_changing_semantic_labels() -> None:
    fixture = _fixture()
    compiled = compile_pass_semantic_g_envelope(
        predictor_section_payload=fixture.predictor_packet,
        predictor_surface=fixture.predictor_surface,
        repair_program=fixture.repair_program,
        target_custody=fixture.target_custody,
    )
    decoded = compiled.decoded

    assert compiled.mode is PassSemanticGEnvelopeMode.PASS_THEN_G8_V1
    assert compiled.compiled_repair is not None
    assert decoded.decoded_repair is not None
    assert decoded.receipt.repair_packet_bytes > 0
    assert decoded.receipt.repair_packet_sha256 == compiled.compiled_repair.decoded.receipt.packet_sha256
    assert decoded.receipt.repair_receipt_sha256 == compiled.compiled_repair.decoded.receipt.receipt_sha256
    assert np.array_equal(decoded.semantic_labels, fixture.predictor_surface.predictor_state.labels)
    assert not np.array_equal(decoded.conditional_camera_y1, fixture.predictor_surface.camera_p1)
    assert decoded.receipt.output_camera_y1_sha256 != decoded.receipt.pre_repair_camera_y1_sha256
    assert decoded.receipt.scorer_invoked is False
    assert decoded.receipt.through_r_target_realization_debt_closed is False


def test_pass_packet_and_envelope_reject_foreign_surface_crc_trailing_and_mode_smuggling() -> None:
    fixture = _fixture()
    packet, _ = compile_pass_semantic_g_packet(predictor_surface=fixture.predictor_surface)
    foreign = _fixture(frame1_delta=1)
    with pytest.raises(PassSemanticGError, match="exact live predictor source"):
        decode_pass_semantic_g_packet(packet, predictor_surface=foreign.predictor_surface)

    mutated_packet = packet[:-1] + bytes([packet[-1] ^ 1])
    with pytest.raises(PassSemanticGError, match="CRC"):
        decode_pass_semantic_g_packet(mutated_packet, predictor_surface=fixture.predictor_surface)
    with pytest.raises(PassSemanticGError, match=r"length|trailing"):
        decode_pass_semantic_g_packet(packet + b"x", predictor_surface=fixture.predictor_surface)

    compiled = compile_pass_semantic_g_envelope(
        predictor_section_payload=fixture.predictor_packet,
        predictor_surface=fixture.predictor_surface,
    )
    mutated_envelope = compiled.envelope[:-1] + bytes([compiled.envelope[-1] ^ 1])
    with pytest.raises(PassSemanticGError):
        parse_pass_semantic_g_envelope(mutated_envelope)
    with pytest.raises(PassSemanticGError, match=r"length accounting|trailing"):
        parse_pass_semantic_g_envelope(compiled.envelope + b"x")

    fields = list(ENVELOPE_HEADER.unpack_from(compiled.envelope))
    fields[2] = 1
    forged_mode = ENVELOPE_HEADER.pack(*fields) + compiled.envelope[ENVELOPE_HEADER.size :]
    with pytest.raises(PassSemanticGError, match="omitted"):
        parse_pass_semantic_g_envelope(forged_mode)


def test_public_g8_surface_builder_rejects_foreign_typed_pass_with_same_pair_window() -> None:
    fixture = _fixture()
    foreign = _fixture(frame1_delta=1)
    _, pass_semantic = compile_pass_semantic_g_packet(predictor_surface=fixture.predictor_surface)
    with pytest.raises(PassSemanticGError, match="exact live predictor custody"):
        build_pass_semantic_g8_surface(
            predictor_section_payload=foreign.predictor_packet,
            predictor_surface=foreign.predictor_surface,
            pass_semantic=pass_semantic,
            predictor_codec_id="LVLS1.v1",
        )


def test_envelope_refuses_empty_or_fabricated_legacy_g_and_invalid_optional_repair_inputs() -> None:
    fixture = _fixture()
    with pytest.raises(PassSemanticGError, match="TACGPS1"):
        compose_pass_semantic_g_envelope(b"")
    with pytest.raises(PassSemanticGError, match="TACGPS1"):
        compose_pass_semantic_g_envelope(b"TACG1C\x00\x00fabricated-pass")
    packet, _ = compile_pass_semantic_g_packet(predictor_surface=fixture.predictor_surface)
    with pytest.raises(PassSemanticGError, match="G8R1"):
        compose_pass_semantic_g_envelope(packet, b"not-a-repair")
    with pytest.raises(PassSemanticGError, match="must not consume target custody"):
        compile_pass_semantic_g_envelope(
            predictor_section_payload=fixture.predictor_packet,
            predictor_surface=fixture.predictor_surface,
            target_custody=fixture.target_custody,
        )
    with pytest.raises(PassSemanticGError, match="requires exact encoder-only target custody"):
        compile_pass_semantic_g_envelope(
            predictor_section_payload=fixture.predictor_packet,
            predictor_surface=fixture.predictor_surface,
            repair_program=fixture.repair_program,
        )


def test_receipts_reject_truth_numeric_duplicate_and_noncanonical_smuggling() -> None:
    fixture = _fixture()
    compiled = compile_pass_semantic_g_envelope(
        predictor_section_payload=fixture.predictor_packet,
        predictor_surface=fixture.predictor_surface,
    )
    pass_receipt = compiled.pass_semantic.receipt
    body = pass_receipt.as_dict()
    body["semantic_mutation_count"] = 0.0
    forged_number = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(PassSemanticGError, match="truth labels"):
        parse_pass_semantic_g_receipt(forged_number)

    body = compiled.decoded.receipt.as_dict()
    body["research_only"] = False
    forged_truth = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(PassSemanticGError, match="truth labels"):
        parse_pass_semantic_g_envelope_receipt(forged_truth)

    receipt_bytes = pass_receipt.to_receipt_bytes()
    duplicate = receipt_bytes[:-1] + b',"schema":"tac.taskspace_pass_semantic_g_receipt.v1"}'
    with pytest.raises(PassSemanticGError, match="repeats key"):
        parse_pass_semantic_g_receipt(duplicate)
    with pytest.raises(PassSemanticGError, match="canonical"):
        parse_pass_semantic_g_receipt(receipt_bytes + b"\n")


def test_headers_are_fixed_big_endian_and_no_dense_payload_is_serialized() -> None:
    fixture = _fixture()
    compiled = compile_pass_semantic_g_envelope(
        predictor_section_payload=fixture.predictor_packet,
        predictor_surface=fixture.predictor_surface,
    )
    pass_fields = PASS_PACKET_HEADER.unpack(compiled.pass_packet)
    envelope_fields = ENVELOPE_HEADER.unpack_from(compiled.envelope)

    assert pass_fields[0] == PASS_PACKET_MAGIC
    assert pass_fields[-2] == PASS_PACKET_HEADER.size
    assert envelope_fields[0] == ENVELOPE_MAGIC
    assert envelope_fields[4] == PASS_PACKET_HEADER.size
    assert envelope_fields[5] == 0
    assert len(compiled.envelope) == ENVELOPE_HEADER.size + PASS_PACKET_HEADER.size
    assert struct.calcsize(">8sBBHH32s32s32s32sII") == PASS_PACKET_HEADER.size
    assert fixture.predictor_surface.predictor_state.labels.shape == (
        1,
        SCORER_HEIGHT,
        SCORER_WIDTH,
    )
    assert fixture.predictor_surface.camera_p1.shape == (
        1,
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        CHANNELS,
    )
    assert MAX_BOUNDED_PAIRS >= 1
