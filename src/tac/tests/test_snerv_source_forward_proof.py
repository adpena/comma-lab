# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.analysis.snerv_source_forward_producer import (
    build_snerv_output2_boundary_verdict,
)
from tac.analysis.snerv_source_forward_proof import (
    DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS,
    REPARAMETERIZED_RENAME_REQUIRED,
    SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA,
    SOURCE_FORWARD_SURFACES,
    SOURCE_FORWARD_TENSOR_NAMES,
    SOURCE_IDENTICAL,
    build_snerv_payload_bitflip_falsification,
    build_snerv_source_forward_proof_action_effect,
    build_snerv_source_forward_surface_provenance,
    validate_snerv_output2_boundary_verdict,
    validate_snerv_source_forward_proof_action_effect,
)

ACTION_ID = "a" * 64
ARCHIVE_SHA = "1" * 64


def _tensor_surfaces(*, output2_delta: float = 0.0) -> dict[str, dict[str, np.ndarray]]:
    base = {
        "coord_time_embedding": np.array([[0.0, 1.0]], dtype=np.float64),
        "mfu_in": np.zeros((1, 1, 2, 2), dtype=np.float64),
        "mfu_out": np.ones((1, 1, 2, 2), dtype=np.float64),
        "hfr_in": np.ones((1, 1, 2, 2), dtype=np.float64),
        "hfr_out": np.full((1, 1, 2, 2), 2.0, dtype=np.float64),
        "tub_in": np.full((1, 1, 2, 2), 3.0, dtype=np.float64),
        "tub_out": np.full((1, 1, 2, 2), 4.0, dtype=np.float64),
        "output_2": np.full((1, 1, 2, 2), 5.0, dtype=np.float64),
        "rgb_pair_float": np.zeros((1, 2, 3, 2, 2), dtype=np.float64),
        "rgb_pair_uint8": np.zeros((1, 2, 3, 2, 2), dtype=np.uint8),
        "segnet_input": np.zeros((1, 3, 2, 2), dtype=np.float64),
        "posenet_input": np.zeros((1, 6, 2, 2), dtype=np.float64),
        "segnet_logits": np.zeros((1, 4, 2, 2), dtype=np.float64),
        "segnet_argmax": np.zeros((1, 2, 2), dtype=np.int64),
        "posenet_output": np.zeros((1, 12), dtype=np.float64),
    }
    surfaces = {surface: {name: value.copy() for name, value in base.items()} for surface in SOURCE_FORWARD_SURFACES}
    if output2_delta:
        surfaces["archive_parseback"]["output_2"] = (
            surfaces["archive_parseback"]["output_2"] + output2_delta
        )
    return surfaces


def _output2_header(*, consumes: bool = True, shape_match: bool = True) -> dict:
    return {
        "tub_output2_storage": {
            "stored": True,
            "source_payload_present": True,
            "receiver_executes_output2_fusion_from_payload": consumes,
            "receiver_frame_decode_consumes_output2": consumes,
            "receiver_output2_frame_shape_match": shape_match,
            "receiver_frame_decode_binding_status": "unit_test_receiver_bound",
            "score_lagrangian_admission": False,
            "score_lagrangian_action": "unit_test_no_score_authority",
        }
    }


def _source_identical_output2_verdict() -> dict:
    return build_snerv_output2_boundary_verdict(
        tensors_by_surface=_tensor_surfaces(),
        archive_decoder_header=_output2_header(),
        tolerance=0.0,
    )


def _bitflip() -> dict:
    return build_snerv_payload_bitflip_falsification(
        bitflip_section="decoder_payload.output_2",
        baseline_section_sha256="2" * 64,
        mutated_section_sha256="3" * 64,
        proof_passed_after_bitflip=False,
        first_failed_tensor="output_2",
        first_failed_surface="archive_parseback",
        bit_offset=7,
        bit_mask=1,
    )


def _scorer_deltas() -> dict:
    return {
        "d_seg": 0.0,
        "d_pose": 0.0,
        "delta_score_nonrate": 0.0,
        "by_surface": {
            surface: {"d_seg": 0.0, "d_pose": 0.0}
            for surface in SOURCE_FORWARD_SURFACES
        },
    }


def test_output2_boundary_verdict_accepts_only_source_identical_receiver_consumed() -> None:
    verdict = _source_identical_output2_verdict()

    assert verdict["schema"] == SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA
    assert verdict["verdict"] == SOURCE_IDENTICAL
    assert verdict["passed"] is True
    assert verdict["blockers"] == []
    assert validate_snerv_output2_boundary_verdict(verdict)["passed"] is True


def test_output2_boundary_verdict_blocks_receiver_side_rename_or_drop_basis() -> None:
    rename = build_snerv_output2_boundary_verdict(
        tensors_by_surface=_tensor_surfaces(),
        archive_decoder_header=_output2_header(consumes=False),
        tolerance=0.0,
    )
    drop = build_snerv_output2_boundary_verdict(
        tensors_by_surface=_tensor_surfaces(),
        archive_decoder_header=_output2_header(shape_match=False),
        tolerance=0.0,
    )

    assert rename["verdict"] == REPARAMETERIZED_RENAME_REQUIRED
    assert "snerv_output2_tensor_present_but_not_receiver_consumed" in rename["blockers"]
    assert rename["minimal_causal_basis_recommendation"] == [
        "lf_carrier",
        "hf_carrier",
        "mfu_state",
        "hfr_state",
        "tub_temporal_state",
        "pair_adapter",
        "derive_output_2",
    ]
    assert drop["verdict"] == DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
    assert "snerv_output2_stored_but_receiver_shape_mismatch" in drop["blockers"]
    assert "snerv_output2_adapter_would_be_required" in drop["blockers"]


def test_source_forward_proof_row_requires_clearable_output2_boundary() -> None:
    blocked_boundary = build_snerv_output2_boundary_verdict(
        tensors_by_surface=_tensor_surfaces(output2_delta=1.0),
        archive_decoder_header=_output2_header(),
        tolerance=0.0,
    )

    row = build_snerv_source_forward_proof_action_effect(
        action_id=ACTION_ID,
        archive_sha256=ARCHIVE_SHA,
        archive_bytes=123,
        payload_section_hashes={"decoder_payload.output_2": "2" * 64},
        pair_ids=[0],
        tensors_by_surface=_tensor_surfaces(),
        scorer_deltas=_scorer_deltas(),
        destructive_payload_bit_flip=_bitflip(),
        output2_boundary_verdict=blocked_boundary,
        surface_provenance=build_snerv_source_forward_surface_provenance(
            pair_ids=[0],
            archive_sha256=ARCHIVE_SHA,
        ),
    )

    assert row["passed"] is False
    assert row["launch_gate_clearable"] is False
    status = validate_snerv_source_forward_proof_action_effect(row)
    assert status["passed"] is False
    assert "snerv_output2_boundary_not_source_identical:REPARAMETERIZED_RENAME_REQUIRED" in status["blockers"]
    assert "snerv_source_forward_launch_gate_clearable_false" in status["blockers"]


def test_source_forward_proof_row_clears_when_output2_boundary_is_source_identical() -> None:
    row = build_snerv_source_forward_proof_action_effect(
        action_id=ACTION_ID,
        archive_sha256=ARCHIVE_SHA,
        archive_bytes=123,
        payload_section_hashes={"decoder_payload.output_2": "2" * 64},
        pair_ids=[0],
        tensors_by_surface=_tensor_surfaces(),
        scorer_deltas=_scorer_deltas(),
        destructive_payload_bit_flip=_bitflip(),
        output2_boundary_verdict=_source_identical_output2_verdict(),
        surface_provenance=build_snerv_source_forward_surface_provenance(
            pair_ids=[0],
            archive_sha256=ARCHIVE_SHA,
        ),
    )

    assert set(row["tensor_names"]) == set(SOURCE_FORWARD_TENSOR_NAMES)
    assert row["passed"] is True
    assert row["launch_gate_clearable"] is True
    assert validate_snerv_source_forward_proof_action_effect(row)["passed"] is True
