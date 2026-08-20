# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import ReceiverRealizationProfileV1
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_dsl.generative_taskspace_correction import DecodedGenerativeCorrectionV1
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_HEADER,
    CorrectedY1SupportCopyCellV1,
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Error,
    PredictorPreservingA3Mode,
    PredictorPreservingA3ProgramV1,
    SparseConstantRGBCellV1,
    apply_disjoint_camera_cell_reconstruction_v1,
    compile_predictor_preserving_a3,
    decode_predictor_preserving_a3_packet,
    derive_predictor_preserving_a3_source_binding,
    parse_predictor_preserving_a3_packet,
    parse_predictor_preserving_a3_receipt,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    PredictorPreservingOverlayError,
    PredictorPreservingOverlayResultV1,
    overlay_g_on_predictor_camera_y1,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2

PAIR_ID = 17
G_CHANGED_CELL = (11, 13)
RGB_A_CELL = (200, 300)


@pytest.fixture(scope="module")
def coupled_sources() -> tuple[
    PredictorCameraPairSurfaceV1,
    DecodedGenerativeCorrectionV1,
    PredictorPreservingOverlayResultV1,
]:
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
        correction_packet_sha256="a" * 64,
    )
    overlay = overlay_g_on_predictor_camera_y1(
        surface.camera_p1,
        state.labels,
        decoded_g,
    )
    return surface, decoded_g, overlay


def _pass_program() -> PredictorPreservingA3ProgramV1:
    return PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1)


def _rgb_program(rgb: tuple[int, int, int] = (101, 102, 103)) -> PredictorPreservingA3ProgramV1:
    return PredictorPreservingA3ProgramV1(
        PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
        constant_rgb_cells=(SparseConstantRGBCellV1(PAIR_ID, *RGB_A_CELL, rgb),),
    )


def _copy_program() -> PredictorPreservingA3ProgramV1:
    return PredictorPreservingA3ProgramV1(
        PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1,
        corrected_y1_copy_cells=(CorrectedY1SupportCopyCellV1(PAIR_ID, *G_CHANGED_CELL),),
    )


def _compile(
    program: PredictorPreservingA3ProgramV1,
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
):
    surface, decoded_g, overlay = coupled_sources
    return compile_predictor_preserving_a3(
        program,
        predictor_surface=surface,
        decoded_g=decoded_g,
        corrected_y1_overlay=overlay,
    )


def test_pass_p0_is_exact_zero_mutation_but_binds_corrected_y1(
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
) -> None:
    surface, _decoded_g, overlay = coupled_sources
    compiled = _compile(_pass_program(), coupled_sources)
    receipt = compiled.receipt

    assert len(compiled.packet) == PACKET_HEADER.size == 62
    assert receipt.packet_body_bytes == 0
    assert receipt.serialized_coordinate_bytes == 0
    assert receipt.serialized_rgb_value_bytes == 0
    assert receipt.control_row_count == 0
    assert receipt.owned_scorer_cells == 0
    assert receipt.owned_camera_values == 0
    assert receipt.actually_changed_camera_values == 0
    assert receipt.p0_pass_through_zero_mutation is True
    assert np.array_equal(compiled.decoded.camera_y0, surface.camera_p0)
    assert np.array_equal(compiled.decoded.camera_y1, overlay.camera_y1)
    assert np.array_equal(compiled.decoded.chronological_camera_frames[:, 0], surface.camera_p0)
    assert np.array_equal(compiled.decoded.chronological_camera_frames[:, 1], overlay.camera_y1)
    assert not np.any(compiled.decoded.owned_scorer_mask)
    assert not np.any(compiled.decoded.owned_camera_mask)
    assert receipt.corrected_camera_y1_sha256 == overlay.receipt.corrected_camera_y1_sha256
    assert receipt.output_camera_y1_sha256 == receipt.corrected_camera_y1_sha256
    assert receipt.predictor_labels_sha256 == surface.predictor_state.labels_sha256
    assert receipt.dense_predictor_frames_serialized is False
    assert receipt.scorer_invoked is False
    assert receipt.score_claim is False
    assert receipt.candidate_claim is False
    assert receipt.research_only is True


def test_sparse_rgb_cell_realizes_exact_numerator_and_preserves_all_unowned_signal(
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
) -> None:
    surface, _decoded_g, overlay = coupled_sources
    compiled = _compile(_rgb_program(), coupled_sources)
    result = compiled.decoded
    receipt = compiled.receipt
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    p0_num, denominator = operator.apply_numerators(surface.camera_p0[0])
    y0_num, output_denominator = operator.apply_numerators(result.camera_y0[0])
    scorer_owned = result.owned_scorer_mask[0]
    camera_owned = result.owned_camera_mask[0]

    assert len(compiled.packet) == PACKET_HEADER.size + 9
    assert receipt.packet_body_bytes == 9
    assert receipt.serialized_coordinate_bytes == 6
    assert receipt.serialized_rgb_value_bytes == 3
    assert receipt.control_row_count == receipt.owned_scorer_cells == 1
    assert receipt.owned_camera_values == 12
    assert 1 <= receipt.actually_changed_camera_values <= 12
    assert denominator == output_denominator == receipt.scorer_denominator
    assert np.array_equal(y0_num[RGB_A_CELL], np.asarray((101, 102, 103)) * denominator)
    assert np.array_equal(y0_num[~scorer_owned], p0_num[~scorer_owned])
    assert np.array_equal(result.camera_y0[0][~camera_owned], surface.camera_p0[0][~camera_owned])
    assert np.array_equal(result.camera_y1, overlay.camera_y1)
    assert receipt.constant_rgb_target_numerators_exact is True
    assert receipt.corrected_y1_support_numerators_exact is False
    assert receipt.outside_a_owned_p0_camera_values_preserved is True
    assert receipt.outside_a_owned_p0_scorer_numerators_preserved is True


def test_corrected_y1_copy_mode_is_coordinate_only_and_preserves_p0_outside_support(
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
) -> None:
    surface, _decoded_g, overlay = coupled_sources
    compiled = _compile(_copy_program(), coupled_sources)
    result = compiled.decoded
    receipt = compiled.receipt
    owned = result.owned_camera_mask[0]
    scorer_owned = result.owned_scorer_mask[0]
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    y0_num, denominator = operator.apply_numerators(result.camera_y0[0])
    y1_num, y1_denominator = operator.apply_numerators(overlay.camera_y1[0])

    assert len(compiled.packet) == PACKET_HEADER.size + 6
    assert receipt.packet_body_bytes == receipt.serialized_coordinate_bytes == 6
    assert receipt.serialized_rgb_value_bytes == 0
    assert np.array_equal(result.camera_y0[0][owned], overlay.camera_y1[0][owned])
    assert np.array_equal(result.camera_y0[0][~owned], surface.camera_p0[0][~owned])
    assert np.array_equal(y0_num[scorer_owned], y1_num[scorer_owned])
    assert denominator == y1_denominator
    assert receipt.corrected_y1_support_numerators_exact is True
    assert receipt.constant_rgb_target_numerators_exact is False
    assert receipt.full_a3_se3_xi_basis_universe_implemented is False
    assert receipt.se3_xi_inverse_solve_implemented is False
    assert receipt.encoder_side_row_selection_solved is False
    assert receipt.implementation_scope.endswith("not_full_a3_se3_xi_basis.v1")


def test_generic_owned_cell_kernel_is_packet_domain_independent_and_exact(
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
) -> None:
    surface, _decoded_g, _overlay = coupled_sources
    row = SparseConstantRGBCellV1(PAIR_ID, *RGB_A_CELL, (201, 202, 203))
    result = apply_disjoint_camera_cell_reconstruction_v1(
        surface.camera_p0,
        source_pair_ids=(PAIR_ID,),
        constant_rgb_cells=(row,),
    )
    expected = np.asarray(row.rgb_u8, dtype=np.int64) * result.scorer_denominator

    assert result.constant_rgb_target_numerators_exact is True
    assert result.copy_source_target_numerators_exact is False
    assert np.array_equal(result.output_scorer_numerators[0, *RGB_A_CELL], expected)
    assert np.array_equal(
        result.output_camera[~result.owned_camera_mask],
        surface.camera_p0[~result.owned_camera_mask],
    )


def test_public_decode_replays_twice_and_strictly_parses_receipt(
    monkeypatch: pytest.MonkeyPatch,
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
) -> None:
    import tac.witness_dsl.predictor_preserving_coupled_preimage as module

    compiled = _compile(_pass_program(), coupled_sources)
    surface, decoded_g, overlay = coupled_sources
    counts = {"decode": 0, "receipt_parse": 0}
    original_decode = module._decode_parsed_packet
    original_receipt_parse = module.parse_predictor_preserving_a3_receipt

    def counted_decode(*args: Any, **kwargs: Any):
        counts["decode"] += 1
        return original_decode(*args, **kwargs)

    def counted_receipt_parse(*args: Any, **kwargs: Any):
        counts["receipt_parse"] += 1
        return original_receipt_parse(*args, **kwargs)

    monkeypatch.setattr(module, "_decode_parsed_packet", counted_decode)
    monkeypatch.setattr(module, "parse_predictor_preserving_a3_receipt", counted_receipt_parse)
    decoded = decode_predictor_preserving_a3_packet(
        compiled.packet,
        predictor_surface=surface,
        decoded_g=decoded_g,
        corrected_y1_overlay=overlay,
    )

    assert decoded.receipt == compiled.receipt
    assert counts == {"decode": 2, "receipt_parse": 1}


def test_packet_parse_reencode_crc_and_source_drift_fail_closed(
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
) -> None:
    surface, decoded_g, overlay = coupled_sources
    compiled = _compile(_rgb_program(), coupled_sources)
    source = derive_predictor_preserving_a3_source_binding(surface, decoded_g, overlay)
    parsed = parse_predictor_preserving_a3_packet(
        compiled.packet,
        expected_source_binding=source,
    )
    assert parsed.program == compiled.program
    assert parsed.packet_body_bytes == 9

    corrupted = compiled.packet[:-1] + bytes([compiled.packet[-1] ^ 1])
    with pytest.raises(PredictorPreservingA3Error, match="CRC"):
        parse_predictor_preserving_a3_packet(corrupted, expected_source_binding=source)
    with pytest.raises(PredictorPreservingA3Error, match=r"length accounting|trailing"):
        parse_predictor_preserving_a3_packet(compiled.packet + b"\x00", expected_source_binding=source)

    changed_frames = surface.chronological_camera_frames.copy()
    changed_frames[0, 0, 0, 0, 0] ^= 1
    changed_surface = PredictorCameraPairSurfaceV1(
        predictor_state=surface.predictor_state,
        chronological_camera_frames=changed_frames,
    )
    with pytest.raises(PredictorPreservingA3Error, match="source binding differs"):
        decode_predictor_preserving_a3_packet(
            compiled.packet,
            predictor_surface=changed_surface,
            decoded_g=decoded_g,
            corrected_y1_overlay=overlay,
        )


def test_program_rejects_rows_in_pass_duplicates_order_aliases_and_foreign_pairs(
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
) -> None:
    row = SparseConstantRGBCellV1(PAIR_ID, *RGB_A_CELL, (1, 2, 3))
    with pytest.raises(PredictorPreservingA3Error, match="zero control rows"):
        PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.PASS_P0_V1,
            constant_rgb_cells=(row,),
        )
    with pytest.raises(PredictorPreservingA3Error, match="unique canonical"):
        PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
            constant_rgb_cells=(row, row),
        )
    later = SparseConstantRGBCellV1(PAIR_ID, 3, 3, (1, 2, 3))
    earlier = SparseConstantRGBCellV1(PAIR_ID, 2, 2, (1, 2, 3))
    with pytest.raises(PredictorPreservingA3Error, match="unique canonical"):
        PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
            constant_rgb_cells=(later, earlier),
        )

    with pytest.raises(PredictorPreservingA3Error, match="no-op alias"):
        _compile(_rgb_program((10, 20, 30)), coupled_sources)
    with pytest.raises(PredictorPreservingA3Error, match="outside the bound source window"):
        _compile(
            PredictorPreservingA3ProgramV1(
                PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
                constant_rgb_cells=(SparseConstantRGBCellV1(PAIR_ID + 1, *RGB_A_CELL, (1, 2, 3)),),
            ),
            coupled_sources,
        )


def test_receipt_parse_recomposes_source_and_rejects_numeric_truth_smuggling(
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
) -> None:
    receipt = _compile(_rgb_program(), coupled_sources).receipt
    payload = receipt.to_receipt_bytes()
    assert parse_predictor_preserving_a3_receipt(payload) == receipt

    def encoded(**updates: Any) -> bytes:
        body = json.loads(payload)
        body.update(updates)
        return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")

    with pytest.raises(PredictorPreservingA3Error, match="exact integers"):
        parse_predictor_preserving_a3_receipt(encoded(source_pair_ids=[True]))
    with pytest.raises(PredictorPreservingA3Error, match="exact integers"):
        parse_predictor_preserving_a3_receipt(encoded(source_pair_ids=[17.0]))
    with pytest.raises(PredictorPreservingA3Error, match="exact integers"):
        parse_predictor_preserving_a3_receipt(encoded(packet_bytes=float(receipt.packet_bytes)))
    with pytest.raises(PredictorPreservingA3Error, match="recompose"):
        parse_predictor_preserving_a3_receipt(encoded(predictor_labels_sha256="f" * 64))
    with pytest.raises(PredictorPreservingA3Error, match="truth labels"):
        parse_predictor_preserving_a3_receipt(encoded(score_claim=True))
    with pytest.raises(PredictorPreservingA3Error, match="not canonical JSON"):
        parse_predictor_preserving_a3_receipt(payload + b"\n")


def test_source_overlay_foreign_keys_cannot_be_substituted(
    coupled_sources: tuple[
        PredictorCameraPairSurfaceV1,
        DecodedGenerativeCorrectionV1,
        PredictorPreservingOverlayResultV1,
    ],
) -> None:
    surface, decoded_g, overlay = coupled_sources
    substituted_g = replace(decoded_g, correction_packet_sha256="b" * 64)
    with pytest.raises(PredictorPreservingA3Error, match="live P/G sources"):
        derive_predictor_preserving_a3_source_binding(surface, substituted_g, overlay)

    bad_overlay_receipt = replace(
        overlay.receipt,
        corrected_camera_y1_sha256="c" * 64,
    )
    with pytest.raises(PredictorPreservingOverlayError, match="differs from receipt"):
        PredictorPreservingOverlayResultV1(
            camera_y1=overlay.camera_y1,
            changed_label_mask=overlay.changed_label_mask,
            owned_camera_mask=overlay.owned_camera_mask,
            receipt=bad_overlay_receipt,
        )
