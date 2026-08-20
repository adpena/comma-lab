# SPDX-License-Identifier: MIT
from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest

import tac.witness_dsl.taskspace_chronological_a3_encoder as encoder
from tac.optimization.direct_description_carrier_compose import ReceiverRealizationProfileV1
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_dsl.generative_taskspace_correction import DecodedGenerativeCorrectionV1
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_HEADER,
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Mode,
    parse_predictor_preserving_a3_packet,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    PredictorPreservingOverlayResultV1,
    overlay_g_on_predictor_camera_y1,
)
from tac.witness_dsl.taskspace_chronological_a3_encoder import (
    A3PrefixPlanV1,
    ChronologicalA3AcquisitionV1,
    ChronologicalA3EncoderError,
    ChronologicalA3Interpretation,
    EncoderOnlyA3TargetV1,
    XIP2Coder,
    XIP2WarpDomain,
    build_xip2_guidance_targets,
    compile_chronological_a3_proposal_groups,
    parse_chronological_a3_encoder_receipt,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_whole_archive_allocator import TaskspaceSectionBundleV1

PAIR_ID = 17
G_CHANGED_CELL = (11, 13)
CONSTANT_CELL = (200, 300)
G_PACKET = b"synthetic-counted-g-packet-v1"
TERMINAL_PACKET = b"synthetic-counted-terminal-packet-v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SyntheticA3Sources:
    surface: PredictorCameraPairSurfaceV1
    decoded_g: DecodedGenerativeCorrectionV1
    overlay: PredictorPreservingOverlayResultV1
    target: EncoderOnlyA3TargetV1


def _paint_cell_support(
    frame: np.ndarray,
    *,
    scorer_cell: tuple[int, int],
    values: np.ndarray | tuple[int, int, int],
) -> None:
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    row_support = operator.row_supports[scorer_cell[0]]
    col_support = operator.col_supports[scorer_cell[1]]
    for camera_row in row_support.indices:
        for camera_col in col_support.indices:
            frame[camera_row, camera_col] = values


@pytest.fixture(scope="module")
def sources() -> SyntheticA3Sources:
    labels = np.zeros((1, 384, 512), dtype=np.uint8)
    state = TaskspacePredictorStateV2(
        predictor_program=b"synthetic-exact-predictor-program-v1",
        predictor_renderer_sha256="1" * 64,
        source_archive_sha256="2" * 64,
        source_runtime_sha256="3" * 64,
        source_member_name="0.bin",
        source_pair_ids=(PAIR_ID,),
        labels=labels,
        transport=NoTransportV2(),
    )
    frames = np.empty((1, 2, 874, 1164, 3), dtype=np.uint8)
    frames[:, 0, :, :, :] = (10, 20, 30)
    frames[:, 1, :, :, :] = (70, 80, 90)
    surface = PredictorCameraPairSurfaceV1(
        predictor_state=state,
        chronological_camera_frames=frames,
    )
    corrected_labels = labels.copy()
    corrected_labels[0, *G_CHANGED_CELL] = 1
    profile = ReceiverRealizationProfileV1(
        (
            (5, 15, 25),
            (40, 50, 60),
            (75, 85, 95),
            (110, 120, 130),
            (145, 155, 165),
        )
    )
    decoded_g = DecodedGenerativeCorrectionV1(
        labels=corrected_labels,
        realization_profile=profile,
        correction_packet_sha256=_sha256(G_PACKET),
    )
    overlay = overlay_g_on_predictor_camera_y1(
        surface.camera_p1,
        state.labels,
        decoded_g,
    )
    target_camera = surface.camera_p0.copy()
    _paint_cell_support(
        target_camera[0],
        scorer_cell=CONSTANT_CELL,
        values=(0, 0, 0),
    )
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    row_support = operator.row_supports[G_CHANGED_CELL[0]]
    col_support = operator.col_supports[G_CHANGED_CELL[1]]
    overlay_cell_rgb = overlay.camera_y1[
        0,
        row_support.indices[0],
        col_support.indices[0],
    ]
    target_overlay_numerators, denominator = operator.apply_numerators(overlay.camera_y1[0])
    assert np.array_equal(
        target_overlay_numerators[G_CHANGED_CELL],
        overlay_cell_rgb.astype(np.int64) * denominator,
    )
    _paint_cell_support(
        target_camera[0],
        scorer_cell=G_CHANGED_CELL,
        values=overlay_cell_rgb,
    )
    target = EncoderOnlyA3TargetV1(
        source_pair_ids=(PAIR_ID,),
        target_camera_y0=target_camera,
        custody_sha256=_sha256(b"synthetic-frozen-target-custody-v1"),
    )
    return SyntheticA3Sources(surface, decoded_g, overlay, target)


@pytest.fixture(scope="module")
def acquisition(sources: SyntheticA3Sources) -> ChronologicalA3AcquisitionV1:
    return compile_chronological_a3_proposal_groups(
        predictor_surface=sources.surface,
        decoded_g=sources.decoded_g,
        corrected_y1_overlay=sources.overlay,
        target=sources.target,
        plans=(
            A3PrefixPlanV1(
                ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1,
                (1, 2),
            ),
            A3PrefixPlanV1(
                ChronologicalA3Interpretation.CORRECTED_Y1_SUPPORT_COPY_V1,
                (1,),
            ),
        ),
        proposal_id_prefix="g9test",
    )


def test_compiles_concrete_counted_packets_with_exact_prefix_arithmetic(
    sources: SyntheticA3Sources,
    acquisition: ChronologicalA3AcquisitionV1,
) -> None:
    assert len(acquisition.pass_p0_control.packet) == PACKET_HEADER.size
    assert len(acquisition.proposals) == 3
    assert tuple(row.interpretation for row in acquisition.proposals) == (
        ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1,
        ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1,
        ChronologicalA3Interpretation.CORRECTED_Y1_SUPPORT_COPY_V1,
    )
    assert tuple(row.receipt.packet_bytes for row in acquisition.proposals) == (
        PACKET_HEADER.size + 9,
        PACKET_HEADER.size + 18,
        PACKET_HEADER.size + 6,
    )
    assert tuple(row.receipt.control_row_count for row in acquisition.proposals) == (1, 2, 1)

    for proposal in acquisition.proposals:
        receipt = proposal.receipt
        assert receipt.selected_numerator_sse_reduction > 0
        assert receipt.selected_numerator_sse_before - receipt.selected_numerator_sse_after == (
            receipt.selected_numerator_sse_reduction
        )
        assert receipt.source_binding_sha256 == acquisition.source_binding.binding_sha256
        assert receipt.target_evidence_binding_sha256 == sources.target.binding_sha256
        assert receipt.serialized_dense_target_camera_bytes == 0
        assert receipt.serialized_target_pose_bytes == 0
        assert receipt.serialized_scorer_evidence_bytes == 0
        assert receipt.through_r_verified is False
        assert receipt.n600_evidence_claim is False
        assert receipt.exact_score_claim is False
        assert receipt.candidate_claim is False
        assert receipt.originality_claim is False
        assert sources.target.binding_sha256.encode("ascii") not in proposal.compiled_a3.packet
        assert sources.target.target_camera_y0.nbytes > len(proposal.compiled_a3.packet)


def test_allocator_transform_replaces_only_a_and_rechecks_exact_p_g_source(
    sources: SyntheticA3Sources,
    acquisition: ChronologicalA3AcquisitionV1,
) -> None:
    baseline = TaskspaceSectionBundleV1(
        predictor_packet=sources.surface.predictor_state.predictor_program,
        generative_correction_packet=G_PACKET,
        coupled_preimage_packet=acquisition.pass_p0_control.packet,
        terminal_quotient_packet=TERMINAL_PACKET,
    )
    for compiled in acquisition.proposals:
        transform = compiled.allocator_proposal.transform
        assert isinstance(transform, functools.partial)
        assert set(transform.keywords or {}) == {"source_binding", "replacement_packet"}
        assert all(not isinstance(value, EncoderOnlyA3TargetV1) for value in transform.args)
        assert all(
            not isinstance(value, EncoderOnlyA3TargetV1)
            for value in (transform.keywords or {}).values()
        )
        replaced = transform(baseline)
        assert replaced.predictor_packet == baseline.predictor_packet
        assert replaced.generative_correction_packet == baseline.generative_correction_packet
        assert replaced.terminal_quotient_packet == baseline.terminal_quotient_packet
        assert replaced.coupled_preimage_packet == compiled.compiled_a3.packet
        parsed = parse_predictor_preserving_a3_packet(
            replaced.coupled_preimage_packet,
            expected_source_binding=acquisition.source_binding,
        )
        assert parsed.program == compiled.compiled_a3.program

    proposal = acquisition.proposals[0].allocator_proposal
    with pytest.raises(ChronologicalA3EncoderError, match="predictor P differs"):
        proposal.transform(
            TaskspaceSectionBundleV1(
                predictor_packet=b"tampered-p",
                generative_correction_packet=G_PACKET,
                coupled_preimage_packet=acquisition.pass_p0_control.packet,
            )
        )
    with pytest.raises(ChronologicalA3EncoderError, match="correction G differs"):
        proposal.transform(
            TaskspaceSectionBundleV1(
                predictor_packet=sources.surface.predictor_state.predictor_program,
                generative_correction_packet=b"tampered-g",
                coupled_preimage_packet=acquisition.pass_p0_control.packet,
            )
        )


def test_compile_verifies_each_packet_with_two_strict_parses_and_public_decodes(
    monkeypatch: pytest.MonkeyPatch,
    sources: SyntheticA3Sources,
) -> None:
    parse_count = 0
    decode_count = 0
    real_parse = encoder.parse_predictor_preserving_a3_packet
    real_decode = encoder.decode_predictor_preserving_a3_packet

    def counted_parse(*args: object, **kwargs: object):
        nonlocal parse_count
        parse_count += 1
        return real_parse(*args, **kwargs)

    def counted_decode(*args: object, **kwargs: object):
        nonlocal decode_count
        decode_count += 1
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(encoder, "parse_predictor_preserving_a3_packet", counted_parse)
    monkeypatch.setattr(encoder, "decode_predictor_preserving_a3_packet", counted_decode)
    compile_chronological_a3_proposal_groups(
        predictor_surface=sources.surface,
        decoded_g=sources.decoded_g,
        corrected_y1_overlay=sources.overlay,
        target=sources.target,
        plans=(
            A3PrefixPlanV1(
                ChronologicalA3Interpretation.CORRECTED_Y1_SUPPORT_COPY_V1,
                (1,),
            ),
        ),
        proposal_id_prefix="replay",
    )
    assert parse_count == 4
    assert decode_count == 4


def test_receipt_is_strict_and_refuses_authority_truth_smuggling(
    acquisition: ChronologicalA3AcquisitionV1,
) -> None:
    receipt = acquisition.proposals[0].receipt
    payload = receipt.to_receipt_bytes()
    assert parse_chronological_a3_encoder_receipt(payload) == receipt

    tampered = json.loads(payload)
    tampered["exact_score_claim"] = True
    tampered_payload = json.dumps(
        tampered,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    with pytest.raises(ChronologicalA3EncoderError, match="truth was relaxed"):
        parse_chronological_a3_encoder_receipt(tampered_payload)
    with pytest.raises(ChronologicalA3EncoderError, match="not canonical JSON"):
        parse_chronological_a3_encoder_receipt(payload + b"\n")


def test_pair_binding_and_requested_positive_prefix_fail_closed(
    sources: SyntheticA3Sources,
) -> None:
    wrong_pair_target = EncoderOnlyA3TargetV1(
        source_pair_ids=(PAIR_ID + 1,),
        target_camera_y0=sources.target.target_camera_y0,
        custody_sha256=sources.target.custody_sha256,
    )
    with pytest.raises(ChronologicalA3EncoderError, match="pair order differs"):
        compile_chronological_a3_proposal_groups(
            predictor_surface=sources.surface,
            decoded_g=sources.decoded_g,
            corrected_y1_overlay=sources.overlay,
            target=wrong_pair_target,
            plans=(
                A3PrefixPlanV1(
                    ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1,
                    (1,),
                ),
            ),
        )
    with pytest.raises(ChronologicalA3EncoderError, match="fewer than requested prefix 3"):
        compile_chronological_a3_proposal_groups(
            predictor_surface=sources.surface,
            decoded_g=sources.decoded_g,
            corrected_y1_overlay=sources.overlay,
            target=sources.target,
            plans=(
                A3PrefixPlanV1(
                    ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1,
                    (3,),
                ),
            ),
        )


def test_xip2_guidance_exposes_both_domains_from_one_strict_payload_without_wire_claim(
    sources: SyntheticA3Sources,
) -> None:
    guidance = build_xip2_guidance_targets(
        predictor_surface=sources.surface,
        decoded_g=sources.decoded_g,
        corrected_y1_overlay=sources.overlay,
        xi=np.zeros((1, 6), dtype=np.float64),
        pitch=0.0,
        q_levels=4096,
        coder=XIP2Coder.DELTA_AR,
    )
    assert guidance.xip2_payload.startswith(b"XIP2")
    assert guidance.camera_then_r is guidance.target_for_domain(XIP2WarpDomain.CAMERA_THEN_R)
    assert guidance.scorer_then_factor2 is guidance.target_for_domain(
        XIP2WarpDomain.SCORER_THEN_FACTOR2
    )
    assert np.array_equal(guidance.camera_then_r.target_camera_y0, sources.overlay.camera_y1)
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    expected_scorer = np.round(operator.apply(sources.overlay.camera_y1[0])).astype(np.uint8)
    realized_numerators, denominator = operator.apply_numerators(
        guidance.scorer_then_factor2.target_camera_y0[0]
    )
    assert np.array_equal(realized_numerators, expected_scorer.astype(np.int64) * denominator)
    assert guidance.domains_share_identical_xip2_bytes is True
    assert guidance.xip2_payload_is_not_embedded_in_current_a3_packet is True
    assert guidance.pitch_is_not_carried_by_current_xip2_container is True
    assert guidance.full_xip2_a3_receiver_implemented is False
    assert guidance.scorer_invoked is False
    assert guidance.score_claim is False
    assert guidance.research_only is True

    domain_acquisition = compile_chronological_a3_proposal_groups(
        predictor_surface=sources.surface,
        decoded_g=sources.decoded_g,
        corrected_y1_overlay=sources.overlay,
        target=guidance.target_for_domain(XIP2WarpDomain.CAMERA_THEN_R),
        plans=(
            A3PrefixPlanV1(
                ChronologicalA3Interpretation.CORRECTED_Y1_SUPPORT_COPY_V1,
                (1,),
            ),
        ),
        proposal_id_prefix="xip2guide",
    )
    proposal = domain_acquisition.proposals[0]
    assert proposal.compiled_a3.program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1
    assert guidance.xip2_payload not in proposal.compiled_a3.packet
