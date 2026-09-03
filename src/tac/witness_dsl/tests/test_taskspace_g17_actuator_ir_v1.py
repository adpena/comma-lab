# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryCoefficientDelta,
    ReceiverRealizationProfileV1,
)
from tac.witness_dsl.ep725_levelset_predictor_adapter import (
    EP725_SOURCE_DIRECTORY,
    Ep725EphemeralRuntimeSurfaceV2,
    decode_ep725_prefix_ephemeral_surface,
)
from tac.witness_dsl.generative_taskspace_correction import (
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
)
from tac.witness_dsl.taskspace_g17_actuator_ir_v1 import (
    G17ActuatorCheckpointStageV1,
    G17ActuatorExecutionResultV1,
    G17ActuatorIRError,
    G17ActuatorKindV1,
    G17ActuatorOperandRefV1,
    G17ActuatorPhysicalSpanGroupV1,
    G17ActuatorProgramV1,
    build_g17_actuator_checkpoints,
    execute_g17_actuator_program_v1,
    parse_g17_actuator_checkpoint_receipt,
    parse_g17_actuator_execution_receipt,
    parse_g17_actuator_program_receipt,
    reverify_g17_actuator_execution_receipt,
    reverify_g17_actuator_program_receipt,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    TaskspacePredictorStateV2,
    V9Pose6TransportV2,
)
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    compile_generative_taskspace_correction_v2,
)

# TIMEOUT BUDGET 2026-09-03 (ddm_ql1). These tests drive the ep725 COLD decode, which runs four
# full decodes (canonical NumPy x2 then shipped-runtime subprocess x2, adapter
# _decode_executable_source_materialized). MEASURED: one shipped n2 decode costs 15.6-16.7 s and the
# whole cold path 65.6 s on an idle machine -- over the repo-wide pytest-timeout ceiling of 60 s
# (pyproject.toml). The cold surface is a module-scoped cache, so ANY test in this module can be the
# one that pays for filling it; the budget therefore belongs at module scope, not on one test.
# This module decodes at pair_count=1 (timeout_seconds=90.0), roughly half the n2 cost, so it sits
# right on the 60 s line rather than over it -- it went red only while the ep725 pin was stale and
# is a latent flake under load. 180 s is the value already established for this cold path by
# test_taskspace_monolithic_pga_receiver.py's pre-existing @pytest.mark.timeout(180); it is applied
# here for one budget across one shared cold path. This raises a harness watchdog only -- every
# assertion still runs unchanged.
pytestmark = pytest.mark.timeout(180)


@dataclass(frozen=True)
class _RealPairBundle:
    surface: Ep725EphemeralRuntimeSurfaceV2
    packet: bytes
    member: bytes
    operand: G17ActuatorOperandRefV1
    group: G17ActuatorPhysicalSpanGroupV1
    program: G17ActuatorProgramV1
    result: G17ActuatorExecutionResultV1


def _teacher(state: TaskspacePredictorStateV2) -> EncoderOnlyTeacherEvidenceV1:
    target = state.labels.copy()
    target[0, 0, 0] = np.uint8((int(target[0, 0, 0]) + 1) % 5)
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


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (20, 80, 20),
            (240, 220, 40),
            (30, 30, 30),
            (220, 40, 40),
            (40, 80, 220),
        )
    )


def _operand(
    packet: bytes,
    *,
    state: TaskspacePredictorStateV2,
    operand_id: str = "g.page.000",
    group_id: str = "g.group.000",
    offset: int = 4,
    pair_start: int = 0,
) -> G17ActuatorOperandRefV1:
    return G17ActuatorOperandRefV1(
        operand_id=operand_id,
        kind=G17ActuatorKindV1.EP725_LABEL_LOCAL_SEMANTIC_G,
        physical_coding_group_id=group_id,
        member_name="0.bin",
        member_offset=offset,
        byte_length=len(packet),
        operand_sha256=hashlib.sha256(packet).hexdigest(),
        packet_schema="tac.generative_taskspace_correction.v1",
        section_name="G_LABEL_LOCAL_PAGE",
        pair_start=pair_start,
        pair_count=1,
        predictor_slice_binding_sha256=state.binding_sha256,
    )


@pytest.fixture(scope="module")
def real_pair_bundle() -> _RealPairBundle:
    if not EP725_SOURCE_DIRECTORY.is_dir():
        pytest.skip("frozen ep725 source directory is not mounted")
    surface = decode_ep725_prefix_ephemeral_surface(pair_count=1, timeout_seconds=90.0)
    state = surface.predictor_state
    compiled = compile_generative_taskspace_correction_v2(
        state,
        GenerativeCorrectionProgramV1(
            boundary_coefficients=(BoundaryCoefficientDelta(0, "Road", 0, 3.0),),
            realization_profile=_profile(),
        ),
        teacher_evidence=_teacher(state),
    )
    member = b"HEAD" + compiled.packet + b"TAIL"
    operand = _operand(compiled.packet, state=state)
    group = G17ActuatorPhysicalSpanGroupV1(
        physical_coding_group_id="g.group.000",
        member_name="0.bin",
        member_offset=4,
        byte_length=len(compiled.packet),
    )
    program = G17ActuatorProgramV1(
        predictor_state=state,
        counted_member_name="0.bin",
        counted_member_bytes=member,
        physical_coding_groups=(group,),
        operands=(operand,),
    )
    result = execute_g17_actuator_program_v1(program, ep725_surface=surface)
    return _RealPairBundle(surface, compiled.packet, member, operand, group, program, result)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def test_real_ep725_pair_executes_v2_label_local_g_and_preserves_predictor_signal(
    real_pair_bundle: _RealPairBundle,
) -> None:
    bundle = real_pair_bundle
    state = bundle.surface.predictor_state
    result = bundle.result
    step = result.receipt.step_receipts[0]
    ownership = result.ownership_by_operand[0]
    overlay = result.overlay_by_operand[0]
    source = bundle.surface.chronological_camera_frames

    assert state.transport.kind.value == "NONE"
    assert result.receipt.transport_requirement == "LABEL_LOCAL"
    assert step.changed_semantic_cells == ownership.changed_cells > 0
    assert step.actually_changed_camera_values > 0
    assert np.array_equal(result.chronological_frames[:, 0], source[:, 0])
    assert np.array_equal(
        result.chronological_frames[:, 1][~overlay.owned_camera_mask],
        source[:, 1][~overlay.owned_camera_mask],
    )
    assert not np.array_equal(result.chronological_frames[:, 1], source[:, 1])
    assert result.receipt.output_raw_bytes == 2 * 874 * 1164 * 3
    assert result.receipt.scorer_invoked is False
    assert result.receipt.score_claim is False
    assert result.receipt.candidate_claim is False
    assert result.receipt.public_rgb_output_proven is False
    assert result.receipt.n600_execution_proven is False
    assert result.receipt.teacher_or_gt_payload_bytes == 0
    assert result.receipt.scorer_weight_payload_bytes == 0
    assert result.receipt.byte_vm_impersonation is False


def test_program_and_execution_receipts_strict_parse_and_live_reverify(
    real_pair_bundle: _RealPairBundle,
) -> None:
    program = real_pair_bundle.program
    result = real_pair_bundle.result
    program_bytes = program.receipt.to_receipt_bytes()
    execution_bytes = result.receipt.to_receipt_bytes()

    assert parse_g17_actuator_program_receipt(program_bytes) == program.receipt
    assert reverify_g17_actuator_program_receipt(program_bytes, program=program) == program.receipt
    assert parse_g17_actuator_execution_receipt(execution_bytes) == result.receipt
    assert (
        reverify_g17_actuator_execution_receipt(
            execution_bytes,
            program=program,
            ep725_surface=real_pair_bundle.surface,
        )
        == result.receipt
    )

    forged = json.loads(execution_bytes)
    forged["output_y1_sha256"] = "0" * 64
    forged_bytes = _canonical(forged)
    assert parse_g17_actuator_execution_receipt(forged_bytes).output_y1_sha256 == "0" * 64
    with pytest.raises(G17ActuatorIRError, match="differs from exact receiver replay"):
        reverify_g17_actuator_execution_receipt(
            forged_bytes,
            program=program,
            ep725_surface=real_pair_bundle.surface,
        )


def test_every_counted_g_packet_byte_is_live_under_crc_and_canonical_parse(
    real_pair_bundle: _RealPairBundle,
) -> None:
    bundle = real_pair_bundle
    state = bundle.surface.predictor_state
    for index in range(len(bundle.packet)):
        mutated_packet = bytearray(bundle.packet)
        mutated_packet[index] ^= 1
        packet = bytes(mutated_packet)
        member = b"HEAD" + packet + b"TAIL"
        operand = _operand(packet, state=state)
        with pytest.raises(ValueError):
            G17ActuatorProgramV1(
                predictor_state=state,
                counted_member_name="0.bin",
                counted_member_bytes=member,
                physical_coding_groups=(bundle.group,),
                operands=(operand,),
            )


def test_member_span_hash_gap_overlap_and_unowned_group_bytes_fail_closed(
    real_pair_bundle: _RealPairBundle,
) -> None:
    bundle = real_pair_bundle
    state = bundle.surface.predictor_state
    mutated_member = bytearray(bundle.member)
    mutated_member[4] ^= 1
    with pytest.raises(G17ActuatorIRError, match="span hash"):
        G17ActuatorProgramV1(
            state,
            "0.bin",
            bytes(mutated_member),
            (bundle.group,),
            (bundle.operand,),
        )

    doubled = bundle.packet + bundle.packet
    first = _operand(bundle.packet, state=state, offset=0, operand_id="g.page.000")
    overlapping = _operand(
        bundle.packet,
        state=state,
        offset=len(bundle.packet) - 1,
        operand_id="g.page.001",
    )
    with pytest.raises(G17ActuatorIRError, match="operand overlap"):
        G17ActuatorProgramV1(
            state,
            "0.bin",
            doubled[:-1],
            (G17ActuatorPhysicalSpanGroupV1("g.group.000", "0.bin", 0, len(doubled) - 1),),
            (first, overlapping),
        )

    gapped_member = bundle.packet + b"x" + bundle.packet
    gapped = _operand(
        bundle.packet,
        state=state,
        offset=len(bundle.packet) + 1,
        operand_id="g.page.001",
    )
    with pytest.raises(G17ActuatorIRError, match="operand gap"):
        G17ActuatorProgramV1(
            state,
            "0.bin",
            gapped_member,
            (G17ActuatorPhysicalSpanGroupV1("g.group.000", "0.bin", 0, len(gapped_member)),),
            (first, gapped),
        )

    with pytest.raises(G17ActuatorIRError, match="trailing unowned bytes"):
        G17ActuatorProgramV1(
            state,
            "0.bin",
            bundle.packet + b"x",
            (
                G17ActuatorPhysicalSpanGroupV1(
                    "g.group.000",
                    "0.bin",
                    0,
                    len(bundle.packet) + 1,
                ),
            ),
            (first,),
        )


def test_distinct_physical_groups_cannot_overlap_and_double_own_bytes(
    real_pair_bundle: _RealPairBundle,
) -> None:
    bundle = real_pair_bundle
    overlapping = G17ActuatorPhysicalSpanGroupV1(
        "g.group.001",
        "0.bin",
        5,
        len(bundle.packet) - 1,
    )
    with pytest.raises(G17ActuatorIRError, match="overlap and double-own"):
        G17ActuatorProgramV1(
            bundle.surface.predictor_state,
            "0.bin",
            bundle.member,
            (bundle.group, overlapping),
            (bundle.operand,),
        )


def test_false_public_n600_and_byte_vm_claims_are_unrepresentable(
    real_pair_bundle: _RealPairBundle,
) -> None:
    with pytest.raises(G17ActuatorIRError, match="truth contract"):
        replace(real_pair_bundle.program.receipt, public_archive_proven=True)
    with pytest.raises(G17ActuatorIRError, match="truth contract"):
        replace(real_pair_bundle.result.receipt, n600_execution_proven=True)
    with pytest.raises(G17ActuatorIRError, match="truth contract"):
        replace(real_pair_bundle.result.receipt, byte_vm_impersonation=True)
    with pytest.raises(G17ActuatorIRError, match="truth contract"):
        replace(real_pair_bundle.result.receipt, teacher_or_gt_payload_bytes=1)


def test_v9_pose6_cross_cast_is_refused_before_operand_execution(
    real_pair_bundle: _RealPairBundle,
) -> None:
    state = real_pair_bundle.surface.predictor_state
    pose6 = np.zeros((1, 6), dtype=np.int16)
    transport = V9Pose6TransportV2(
        counted_payload=np.ascontiguousarray(pose6, dtype=">i2").tobytes(),
        pose6_codes=pose6,
        source_pair_ids=state.source_pair_ids,
        predictor_program_sha256=state.predictor_program_sha256,
    )
    wrong_state = TaskspacePredictorStateV2(
        predictor_program=state.predictor_program,
        predictor_renderer_sha256=state.predictor_renderer_sha256,
        source_archive_sha256=state.source_archive_sha256,
        source_runtime_sha256=state.source_runtime_sha256,
        source_member_name=state.source_member_name,
        source_pair_ids=state.source_pair_ids,
        labels=state.labels,
        transport=transport,
    )
    with pytest.raises(G17ActuatorIRError, match="NoTransportV2"):
        G17ActuatorProgramV1(
            wrong_state,
            "0.bin",
            real_pair_bundle.member,
            (real_pair_bundle.group,),
            (real_pair_bundle.operand,),
        )


def test_checkpoint_chain_is_strict_canonical_and_binds_realized_state(
    real_pair_bundle: _RealPairBundle,
) -> None:
    checkpoints = build_g17_actuator_checkpoints(real_pair_bundle.result)
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert checkpoint.stage is G17ActuatorCheckpointStageV1.PAIR_OUTPUT_REALIZED
    assert checkpoint.completed_pair_ids == (0,)
    assert checkpoint.previous_checkpoint_sha256 is None
    assert parse_g17_actuator_checkpoint_receipt(checkpoint.to_receipt_bytes()) == checkpoint

    duplicate = checkpoint.to_receipt_bytes()[:-1] + b',"schema":"tac.g17_actuator_checkpoint_receipt.v1"}'
    with pytest.raises(G17ActuatorIRError, match="repeats JSON key"):
        parse_g17_actuator_checkpoint_receipt(duplicate)
