# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from functools import cache

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    ReceiverRealizationProfileV1,
    TopologyEventV1,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl import generative_taskspace_receiver as g2_module
from tac.witness_dsl.coupled_preimage_program import (
    CoupledPreimageMode,
    Frame1AnchoredY0FibreControlV1,
    JointSharedSkeletonTwoFibreControlV1,
    compile_coupled_preimage_program,
)
from tac.witness_dsl.generative_taskspace_correction import (
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
    PredictorSemanticStateV1,
    apply_generative_taskspace_correction,
    compile_generative_taskspace_correction,
)
from tac.witness_dsl.generative_taskspace_receiver import (
    CountedTaskspaceRole,
    build_generative_taskspace_receiver_fragment_archive,
    parse_generative_taskspace_receiver_fragment_archive,
)
from tac.witness_dsl.taskspace_pair_fragment_receiver import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    MAX_IN_MEMORY_PAIRS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    TaskspacePairFragmentDecodeV1,
    TaskspacePairFragmentReceiverError,
    build_a_only_counterfactual_fragment_archive,
    build_taskspace_pair_fragment_archive,
    parse_taskspace_pair_fragment_receipt,
    receive_taskspace_pair_fragment,
)


def _state(*, pair_count: int = 1) -> PredictorSemanticStateV1:
    labels = np.zeros((pair_count, SCORER_HEIGHT, SCORER_WIDTH), dtype=np.uint8)
    labels[:, 72:144] = 1
    labels[:, 144:230] = 2
    labels[:, 230:315] = 3
    labels[:, 315:] = 4
    for index in range(pair_count):
        labels[index] = np.roll(labels[index], index * 3, axis=1)
    pose6 = np.arange(pair_count * 6, dtype=np.int16).reshape(pair_count, 6)
    return PredictorSemanticStateV1(
        predictor_program_sha256="a" * 64,
        predictor_renderer_sha256="b" * 64,
        source_pair_ids=tuple(range(20, 20 + pair_count)),
        labels=labels,
        pose6_codes=pose6,
    )


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (12, 24, 36),
            (48, 60, 72),
            (84, 96, 108),
            (120, 132, 144),
            (156, 168, 180),
        )
    )


def _teacher(state: PredictorSemanticStateV1) -> EncoderOnlyTeacherEvidenceV1:
    target = state.labels.copy()
    target[0, 0, 0] = np.uint8((int(target[0, 0, 0]) + 1) % 5)
    target = np.ascontiguousarray(target)
    return EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=hashlib.sha256(memoryview(target).cast("B")).hexdigest(),
        obligation_ir_sha256="4" * 64,
        oracle_evidence_sha256="5" * 64,
        dense_y_sha256="6" * 64,
        target_labels=target,
        teacher_event_count=1,
    )


def _g_packet(state: PredictorSemanticStateV1, *, x0: int = 30) -> bytes:
    program = GenerativeCorrectionProgramV1(
        topology_events=(
            TopologyEventV1(
                state.source_pair_ids[0],
                "Lane",
                "birth",
                "box",
                1,
                35,
                x0,
                48,
                x0 + 19,
            ),
        ),
        realization_profile=_profile(),
    )
    return compile_generative_taskspace_correction(
        state,
        program,
        teacher_evidence=_teacher(state),
    ).packet


def _anchored_controls(state: PredictorSemanticStateV1, *, variant: int = 0):
    return tuple(
        Frame1AnchoredY0FibreControlV1(
            pair_id,
            1,
            -2,
            (3 + variant, -4, 5),
        )
        for pair_id in state.source_pair_ids
    )


def _joint_controls(state: PredictorSemanticStateV1, *, variant: int = 0):
    return tuple(
        JointSharedSkeletonTwoFibreControlV1(
            pair_id,
            2,
            -1,
            (
                (1 + variant, 0, 0),
                (0, 2, 0),
                (0, 0, 3),
                (-4, 0, 0),
                (0, -5, 0),
            ),
        )
        for pair_id in state.source_pair_ids
    )


def _a_packet(
    state: PredictorSemanticStateV1,
    g_packet: bytes,
    mode: CoupledPreimageMode,
    *,
    variant: int = 0,
) -> bytes:
    decoded_g = apply_generative_taskspace_correction(g_packet, predictor_state=state)
    kwargs = (
        {"anchored_controls": _anchored_controls(state, variant=variant)}
        if mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE
        else {"joint_controls": _joint_controls(state, variant=variant)}
    )
    return compile_coupled_preimage_program(
        state,
        decoded_g,
        mode=mode,
        **kwargs,
    ).packet


@dataclass(frozen=True)
class _Fixture:
    state: PredictorSemanticStateV1
    g_packet: bytes
    a_packet: bytes
    archive: bytes
    received: TaskspacePairFragmentDecodeV1


@cache
def _fixture(mode: CoupledPreimageMode) -> _Fixture:
    state = _state()
    g_packet = _g_packet(state)
    a_packet = _a_packet(state, g_packet, mode)
    archive = build_taskspace_pair_fragment_archive(
        g_packet,
        a_packet,
        predictor_state=state,
    )
    received = receive_taskspace_pair_fragment(archive, predictor_state=state)
    return _Fixture(state, g_packet, a_packet, archive, received)


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_one_archive_strictly_receives_both_a_modes(mode: CoupledPreimageMode) -> None:
    fixture = _fixture(mode)
    result = fixture.received
    parsed = parse_generative_taskspace_receiver_fragment_archive(fixture.archive)

    assert tuple(row.role for row in parsed.counted.sections) == (
        CountedTaskspaceRole.GENERATIVE_CORRECTION,
        CountedTaskspaceRole.COUPLED_PREIMAGE,
    )
    assert result.a_program.mode is mode
    assert result.receipt.a_mode is mode
    assert result.receipt.counted_section_descriptors_are_sole_slice_authority is True
    assert result.receipt.semantic_role_order == ("GENERATOR_PARAMETERS", "FRAME0_FROM_EXACT_Y1")
    assert result.receipt.external_pair_manifest_accepted is False
    assert result.receipt.caller_decoded_g_accepted is False
    assert result.receipt.exact_counted_g_reapplied is True
    assert result.receipt.exact_counted_a_strictly_parsed_and_decoded is True
    assert result.receipt.exact_y1_derived_from_counted_g_before_y0 is True
    assert result.receipt.all_counted_sections_semantically_consumed is True
    assert result.receipt.complete_bounded_chronological_pair_output is True
    assert result.receipt.counted_predictor_section_present is False
    assert result.receipt.external_predictor_state_required is True
    assert result.receipt.repository_runtime_dependency is True
    assert result.receipt.complete_data_in_code_lineage_audit is False
    assert result.receipt.n600_evidence_claim is False
    assert result.receipt.exact_score_claim is False
    assert result.receipt.candidate_archive_eligible is False
    assert result.receipt.originality_claim is False
    assert result.receipt.promotion_eligible is False
    assert result.receipt.research_only is True
    assert np.array_equal(result.scorer_chronological_frames[:, 0], result.decoded_a.y0)
    assert np.array_equal(result.scorer_chronological_frames[:, 1], result.decoded_a.y1)
    assert not np.array_equal(result.decoded_a.y0, result.decoded_a.y1)


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_g2_descriptors_are_exact_slice_hash_and_crc_authority(mode: CoupledPreimageMode) -> None:
    fixture = _fixture(mode)
    parsed = parse_generative_taskspace_receiver_fragment_archive(fixture.archive)
    receipt = fixture.received.receipt

    assert len(receipt.sections) == 2
    for retained, descriptor in zip(receipt.sections, parsed.counted.sections, strict=True):
        exact_slice = parsed.counted.payload[retained.offset : retained.stop]
        assert retained.role is descriptor.role
        assert retained.offset == descriptor.offset
        assert retained.byte_length == descriptor.byte_length
        assert retained.payload_sha256 == descriptor.payload_sha256
        assert retained.crc32 == descriptor.crc32
        assert exact_slice == descriptor.payload
        assert hashlib.sha256(exact_slice).hexdigest() == retained.payload_sha256


def _corrupt_exact_section(archive: bytes, role: CountedTaskspaceRole) -> bytes:
    parsed = parse_generative_taskspace_receiver_fragment_archive(archive)
    section = parsed.counted.section(role)
    member = bytearray(parsed.counted.payload)
    member[section.offset + section.byte_length // 2] ^= 0x01
    return g2_module._canonical_zip(bytes(member))


@pytest.mark.parametrize(
    "role",
    (CountedTaskspaceRole.GENERATIVE_CORRECTION, CountedTaskspaceRole.COUPLED_PREIMAGE),
)
def test_corrupt_descriptor_bound_g_or_a_bytes_fail_closed(role: CountedTaskspaceRole) -> None:
    fixture = _fixture(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    corrupted = _corrupt_exact_section(fixture.archive, role)
    with pytest.raises(TaskspacePairFragmentReceiverError, match="strict G2 archive parse"):
        receive_taskspace_pair_fragment(corrupted, predictor_state=fixture.state)


def test_substituted_g_or_a_and_fixed_a_with_changed_g_fail_source_binding() -> None:
    fixture = _fixture(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    changed_g = _g_packet(fixture.state, x0=120)
    changed_a = _a_packet(
        fixture.state,
        changed_g,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )

    fixed_a_changed_g = build_generative_taskspace_receiver_fragment_archive(
        changed_g,
        predictor_state=fixture.state,
        coupled_preimage_packet=fixture.a_packet,
    )
    with pytest.raises(TaskspacePairFragmentReceiverError, match="counted A failed strict reverse-causal"):
        receive_taskspace_pair_fragment(fixed_a_changed_g, predictor_state=fixture.state)

    substituted_a = build_generative_taskspace_receiver_fragment_archive(
        fixture.g_packet,
        predictor_state=fixture.state,
        coupled_preimage_packet=changed_a,
    )
    with pytest.raises(TaskspacePairFragmentReceiverError, match="counted A failed strict reverse-causal"):
        receive_taskspace_pair_fragment(substituted_a, predictor_state=fixture.state)

    with pytest.raises(TaskspacePairFragmentReceiverError, match="counted A failed strict reverse-causal"):
        build_taskspace_pair_fragment_archive(
            changed_g,
            fixture.a_packet,
            predictor_state=fixture.state,
        )


def test_a_only_matched_counterfactual_changes_y0_not_g_or_y1() -> None:
    fixture = _fixture(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    changed_a = _a_packet(
        fixture.state,
        fixture.g_packet,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        variant=7,
    )
    counterfactual_archive = build_a_only_counterfactual_fragment_archive(
        fixture.archive,
        changed_a,
        predictor_state=fixture.state,
    )
    counterfactual = receive_taskspace_pair_fragment(
        counterfactual_archive,
        predictor_state=fixture.state,
    )
    baseline_sections = parse_generative_taskspace_receiver_fragment_archive(fixture.archive).counted.sections
    counterfactual_sections = parse_generative_taskspace_receiver_fragment_archive(
        counterfactual_archive
    ).counted.sections

    assert baseline_sections[0].payload == counterfactual_sections[0].payload
    assert baseline_sections[1].payload != counterfactual_sections[1].payload
    assert fixture.received.receipt.scorer_y0_sha256 != counterfactual.receipt.scorer_y0_sha256
    assert fixture.received.receipt.scorer_y1_sha256 == counterfactual.receipt.scorer_y1_sha256
    assert (
        fixture.received.receipt.camera_frame_sha256_chronological[0::2]
        != (counterfactual.receipt.camera_frame_sha256_chronological[0::2])
    )
    assert (
        fixture.received.receipt.camera_frame_sha256_chronological[1::2]
        == (counterfactual.receipt.camera_frame_sha256_chronological[1::2])
    )


def test_optional_t_is_refused_until_a_real_semantic_consumer_exists() -> None:
    fixture = _fixture(CoupledPreimageMode.JOINT_SHARED_SKELETON_TWO_FIBRE)
    with_t = build_generative_taskspace_receiver_fragment_archive(
        fixture.g_packet,
        predictor_state=fixture.state,
        coupled_preimage_packet=fixture.a_packet,
        terminal_quotient_packet=b"TACT-unconsumed-semantic-terminal-quotient-v1",
    )
    with pytest.raises(TaskspacePairFragmentReceiverError, match="optional T is refused"):
        receive_taskspace_pair_fragment(with_t, predictor_state=fixture.state)


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_camera_raw_is_exact_chronological_y0_y1_factor2_parse_back(mode: CoupledPreimageMode) -> None:
    result = _fixture(mode).received
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )
    raw = np.frombuffer(result.camera_raw, dtype=np.uint8).reshape(
        result.receipt.pair_count,
        2,
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        CHANNELS,
    )

    expected_hashes: list[str] = []
    for pair_index in range(result.receipt.pair_count):
        for chronological_index in range(2):
            camera_frame = raw[pair_index, chronological_index]
            scorer_frame = result.scorer_chronological_frames[pair_index, chronological_index]
            proof = verify_factor2_uint8_scorer_plane(operator, camera_frame, scorer_frame)
            assert proof.certified_exact is True
            assert proof.numerator_exact is True
            expected_hashes.append(hashlib.sha256(camera_frame.tobytes(order="C")).hexdigest())
    assert tuple(expected_hashes) == result.receipt.camera_frame_sha256_chronological
    assert result.receipt.camera_raw_sha256 == hashlib.sha256(result.camera_raw).hexdigest()
    assert result.receipt.chronological_camera_order_y0_then_y1 is True
    assert result.receipt.factor2_camera_parse_back_exact is True


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_repeated_public_receive_is_byte_identical(mode: CoupledPreimageMode) -> None:
    fixture = _fixture(mode)
    replay = receive_taskspace_pair_fragment(fixture.archive, predictor_state=fixture.state)
    assert replay.receipt == fixture.received.receipt
    assert replay.camera_raw == fixture.received.camera_raw
    assert np.array_equal(replay.decoded_g.labels, fixture.received.decoded_g.labels)
    assert np.array_equal(replay.scorer_chronological_frames, fixture.received.scorer_chronological_frames)
    assert replay.receipt.deterministic_double_decode is True


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_receipt_canonical_bytes_roundtrip_and_reemit_identically(mode: CoupledPreimageMode) -> None:
    receipt = _fixture(mode).received.receipt
    first = receipt.to_receipt_bytes()
    second = receipt.to_receipt_bytes()
    reopened = parse_taskspace_pair_fragment_receipt(first)

    assert first == second
    assert reopened == receipt
    assert reopened.to_receipt_bytes() == first
    assert receipt.receipt_sha256 == hashlib.sha256(first).hexdigest()
    assert reopened.sections[-1].stop == reopened.archive_member_bytes
    with pytest.raises(TaskspacePairFragmentReceiverError, match="not canonical JSON"):
        parse_taskspace_pair_fragment_receipt(first + b"\n")


@pytest.mark.parametrize(
    ("field", "substitution"),
    (
        ("scorer_height", float(SCORER_HEIGHT)),
        ("factor2_scorer_values_verified", True),
        ("source_pair_ids", [20.0]),
    ),
)
def test_receipt_parser_rejects_json_numeric_type_smuggling(field: str, substitution: object) -> None:
    receipt = _fixture(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE).received.receipt
    body = json.loads(receipt.to_receipt_bytes().decode("ascii"))
    body[field] = substitution
    mutated = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    with pytest.raises(TaskspacePairFragmentReceiverError):
        parse_taskspace_pair_fragment_receipt(mutated)


def test_public_receipt_constructor_rejects_list_like_tuple_fields() -> None:
    receipt = _fixture(CoupledPreimageMode.JOINT_SHARED_SKELETON_TWO_FIBRE).received.receipt
    with pytest.raises(TaskspacePairFragmentReceiverError, match="sections must be one exact typed tuple"):
        replace(receipt, sections=list(receipt.sections))
    with pytest.raises(TaskspacePairFragmentReceiverError, match="camera-frame hash count"):
        replace(
            receipt,
            camera_frame_sha256_chronological=list(receipt.camera_frame_sha256_chronological),
        )


def test_in_memory_pair_cap_fails_before_attempting_archive_decode() -> None:
    fixture = _fixture(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    assert MAX_IN_MEMORY_PAIRS == 4
    with pytest.raises(TaskspacePairFragmentReceiverError, match="hard-caps in-memory fragments"):
        receive_taskspace_pair_fragment(
            fixture.archive,
            predictor_state=_state(pair_count=MAX_IN_MEMORY_PAIRS + 1),
        )
