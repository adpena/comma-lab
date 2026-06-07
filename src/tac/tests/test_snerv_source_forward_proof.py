# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.analysis.snerv_source_forward_producer import (
    SOURCE_GRAPH_UNPROVEN,
    build_snerv_official_torch_upstream_capture_manifest,
    build_snerv_output2_boundary_verdict,
    validate_snerv_official_torch_upstream_capture_manifest,
)
from tac.analysis.snerv_source_forward_proof import (
    DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS,
    REPARAMETERIZED_RENAME_REQUIRED,
    SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA,
    SOURCE_FORWARD_SURFACES,
    SOURCE_FORWARD_TENSOR_NAMES,
    SOURCE_IDENTICAL,
    build_snerv_payload_bitflip_falsification,
    build_snerv_payload_bitflip_falsification_matrix,
    build_snerv_source_forward_proof_action_effect,
    build_snerv_source_forward_surface_provenance,
    validate_snerv_output2_boundary_verdict,
    validate_snerv_payload_bitflip_falsification,
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


def _output2_header(
    *,
    consumes: bool = True,
    shape_match: bool = True,
    shape_adapter_applied: bool = False,
) -> dict:
    return {
        "tub_output2_storage": {
            "stored": True,
            "source_payload_present": True,
            "receiver_executes_output2_fusion_from_payload": consumes,
            "receiver_frame_decode_consumes_output2": consumes,
            "receiver_output2_frame_shape_match": shape_match,
            "shape_adapter_forbidden": True,
            "shape_adapter_applied": shape_adapter_applied,
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
        receiver_replay_failed=True,
        bit_offset=7,
        bit_mask=1,
    )


def _bitflip_matrix() -> dict:
    first_tensors = {
        "metadata_payload": "coord_time_embedding",
        "lf_payload": "tub_in",
        "decoder_payload": "output_2",
        "step_map_packet": "rgb_pair_uint8",
    }
    return build_snerv_payload_bitflip_falsification_matrix(
        {
            section: build_snerv_payload_bitflip_falsification(
                bitflip_section=section,
                baseline_section_sha256=f"{idx + 1:x}" * 64,
                mutated_section_sha256=f"{idx + 5:x}" * 64,
                proof_passed_after_bitflip=False,
                first_failed_tensor=first_tensor,
                first_failed_surface="archive_parseback",
                receiver_replay_failed=first_tensor != "rgb_pair_uint8",
                rgb_pair_uint8_changed=first_tensor == "rgb_pair_uint8",
                bit_offset=idx,
                bit_mask=1,
            )
            for idx, (section, first_tensor) in enumerate(first_tensors.items())
        }
    )


def test_payload_bitflip_requires_receiver_or_scorer_impact() -> None:
    row = build_snerv_payload_bitflip_falsification(
        bitflip_section="decoder_payload.output_2",
        baseline_section_sha256="2" * 64,
        mutated_section_sha256="3" * 64,
        proof_passed_after_bitflip=False,
        first_failed_tensor="output_2",
        first_failed_surface="archive_parseback",
    )

    status = validate_snerv_payload_bitflip_falsification(row)

    assert row["passed"] is False
    assert row["first_failed_authority_pair"] == "official_torch->archive_parseback"
    assert status["passed"] is False
    assert (
        "snerv_payload_bitflip_downstream_receiver_or_scorer_impact_missing"
        in status["blockers"]
    )


def test_payload_bitflip_accepts_scorer_impact_with_named_surface() -> None:
    row = build_snerv_payload_bitflip_falsification(
        bitflip_section="decoder_payload.output_2",
        baseline_section_sha256="2" * 64,
        mutated_section_sha256="3" * 64,
        proof_passed_after_bitflip=False,
        first_failed_tensor="segnet_argmax",
        first_failed_surface="official_scorer",
        segnet_argmax_changed=True,
        first_scorer_surface_changed="segnet_argmax",
    )

    assert row["first_failed_authority_pair"] == "official_torch->official_scorer"
    assert row["passed"] is True
    assert validate_snerv_payload_bitflip_falsification(row)["passed"] is True


def test_payload_bitflip_requires_named_authority_pair() -> None:
    row = build_snerv_payload_bitflip_falsification(
        bitflip_section="decoder_payload.output_2",
        baseline_section_sha256="2" * 64,
        mutated_section_sha256="3" * 64,
        proof_passed_after_bitflip=False,
        first_failed_tensor="output_2",
        first_failed_surface="archive_parseback",
        receiver_replay_failed=True,
    )
    row["first_failed_authority_pair"] = None

    status = validate_snerv_payload_bitflip_falsification(row)

    assert status["passed"] is False
    assert (
        "snerv_payload_bitflip_first_failed_authority_pair_missing"
        in status["blockers"]
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


def _official_torch_lineage(
    *,
    source_scope: str = "official_trained_checkpoint",
    capture_origin: str | None = None,
) -> dict[str, str | bool]:
    return {
        "trained_checkpoint_lineage": "official_trained_checkpoint_state_dict",
        "checkpoint_sha256": "6" * 64,
        "state_dict_sha256": "7" * 64,
        "model_source_sha256": "8" * 64,
        "source_config_lineage": "official_trained_run_config",
        "source_config_sha256": "9" * 64,
        "source_config_kind": "official_snerv_t_train_config",
        "source_config_source": "unit_test_exact_trained_config",
        "source_config_is_fixture": False,
        "source_scope": source_scope,
        "capture_origin": (
            capture_origin
            if capture_origin is not None
            else (
                "official_upstream_trained_checkpoint"
                if source_scope == "official_trained_checkpoint"
                else "official_upstream_source_fixture"
            )
        ),
    }


def _surface_provenance(
    *,
    source_scope: str = "official_trained_checkpoint",
    capture_origin: str | None = None,
) -> dict:
    return build_snerv_source_forward_surface_provenance(
        pair_ids=[0],
        archive_sha256=ARCHIVE_SHA,
        tensor_capture_authority_by_surface={
            "official_torch": "upstream_snerv_t_forward_source_graph"
        },
        extra_by_surface={
            "official_torch": _official_torch_lineage(
                source_scope=source_scope,
                capture_origin=capture_origin,
            )
        },
    )


def test_official_upstream_manifest_fixture_scope_is_not_authority() -> None:
    manifest = build_snerv_official_torch_upstream_capture_manifest(
        pair_ids=[0],
        tensor_names=SOURCE_FORWARD_TENSOR_NAMES,
        model_source_sha256="8" * 64,
        checkpoint_sha256="6" * 64,
        state_dict_sha256="7" * 64,
        source_config_lineage="official_trained_run_config",
        source_config_sha256="9" * 64,
        source_config_kind="official_snerv_t_train_config",
        source_config_source="unit_test_exact_trained_config",
        source_config_is_fixture=False,
        decoder_len=7,
        source_scope="official_source_fixture_state",
        trained_checkpoint_lineage="official_trained_checkpoint_state_dict",
        capture_origin="official_upstream_source_fixture",
    )

    assert manifest["capture_verdict"] == SOURCE_GRAPH_UNPROVEN
    assert manifest["source_graph_unproven"] is True
    assert manifest["upstream_forward_replay_verified"] is False
    assert manifest["source_forward_replay_authority"] is False
    status = validate_snerv_official_torch_upstream_capture_manifest(
        manifest,
        pair_ids=[0],
        tensor_names=SOURCE_FORWARD_TENSOR_NAMES,
    )
    assert status["passed"] is False
    assert "snerv_official_torch_source_graph_unproven" in status["blockers"]
    assert (
        "snerv_official_torch_capture_verdict_source_graph_unproven"
        in status["blockers"]
    )
    assert (
        "snerv_official_torch_trained_checkpoint_source_scope_missing"
        in status["blockers"]
    )


def test_official_upstream_manifest_partial_tensor_set_is_not_authority() -> None:
    manifest = build_snerv_official_torch_upstream_capture_manifest(
        pair_ids=[0],
        tensor_names=["output_2"],
        model_source_sha256="8" * 64,
        checkpoint_sha256="6" * 64,
        state_dict_sha256="7" * 64,
        source_config_lineage="official_trained_run_config",
        source_config_sha256="9" * 64,
        source_config_kind="official_snerv_t_train_config",
        source_config_source="unit_test_exact_trained_config",
        source_config_is_fixture=False,
        decoder_len=7,
        source_scope="official_trained_checkpoint",
        trained_checkpoint_lineage="official_trained_checkpoint_state_dict",
        capture_origin="official_upstream_trained_checkpoint",
    )

    assert manifest["capture_verdict"] == SOURCE_GRAPH_UNPROVEN
    assert manifest["source_graph_unproven"] is True
    assert manifest["upstream_forward_replay_verified"] is False
    assert manifest["source_forward_replay_authority"] is False
    assert "rgb_pair_uint8" in manifest["missing_required_tensor_names"]
    status = validate_snerv_official_torch_upstream_capture_manifest(
        manifest,
        pair_ids=[0],
        tensor_names=["output_2"],
    )
    assert status["passed"] is False
    assert any(
        blocker.startswith(
            "snerv_official_torch_manifest_missing_required_tensors:"
        )
        for blocker in status["blockers"]
    )


def test_official_upstream_manifest_fixture_config_is_not_authority() -> None:
    manifest = build_snerv_official_torch_upstream_capture_manifest(
        pair_ids=[0],
        tensor_names=SOURCE_FORWARD_TENSOR_NAMES,
        model_source_sha256="8" * 64,
        checkpoint_sha256="6" * 64,
        state_dict_sha256="7" * 64,
        source_config_lineage="official_source_fixture_config",
        source_config_sha256="9" * 64,
        source_config_kind="official_snerv_t_tub_source_fixture_config",
        source_config_source="deterministic_official_source_fixture",
        source_config_is_fixture=True,
        decoder_len=7,
        source_scope="official_trained_checkpoint",
        trained_checkpoint_lineage="official_trained_checkpoint_state_dict",
        capture_origin="official_upstream_trained_checkpoint",
    )

    assert manifest["capture_verdict"] == SOURCE_GRAPH_UNPROVEN
    assert manifest["source_forward_replay_authority"] is False
    status = validate_snerv_official_torch_upstream_capture_manifest(
        manifest,
        pair_ids=[0],
        tensor_names=SOURCE_FORWARD_TENSOR_NAMES,
    )
    assert status["passed"] is False
    assert "snerv_official_torch_trained_config_lineage_missing" in status["blockers"]
    assert "snerv_official_torch_source_config_fixture_forbidden" in status["blockers"]


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
    assert drop["required_next_step"] == (
        "drop_stored_output2_and_store_mfu_hfr_tub_lf_hf_pair_adapter_basis"
    )


def test_output2_boundary_forbids_shape_adapter_even_when_values_match() -> None:
    verdict = build_snerv_output2_boundary_verdict(
        tensors_by_surface=_tensor_surfaces(),
        archive_decoder_header=_output2_header(shape_adapter_applied=True),
        tolerance=0.0,
    )

    assert verdict["verdict"] == DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
    assert verdict["passed"] is False
    assert "snerv_output2_shape_adapter_forbidden" in verdict["blockers"]
    assert verdict["archive_tub_output2_storage"]["shape_adapter_forbidden"] is True
    assert verdict["archive_tub_output2_storage"]["shape_adapter_applied"] is True
    status = validate_snerv_output2_boundary_verdict(verdict)
    assert status["passed"] is False
    assert "snerv_output2_shape_adapter_forbidden" in status["blockers"]


def test_output2_boundary_validator_rejects_forged_source_identical_without_receiver_consumption() -> None:
    verdict = _source_identical_output2_verdict()
    verdict["archive_tub_output2_storage"]["receiver_frame_decode_consumes_output2"] = False

    status = validate_snerv_output2_boundary_verdict(verdict)

    assert status["passed"] is False
    assert "snerv_output2_not_consumed_by_receiver_frame_decode" in status["blockers"]


def test_output2_boundary_validator_rejects_nested_blockers_and_missing_surface() -> None:
    verdict = _source_identical_output2_verdict()
    verdict["blockers"] = ["snerv_output2_tensor_present_but_not_receiver_consumed"]
    verdict["has_output2_by_surface"]["numpy_receiver"] = False

    status = validate_snerv_output2_boundary_verdict(verdict)

    assert status["passed"] is False
    assert "snerv_output2_boundary_nested_blockers_present" in status["blockers"]
    assert "snerv_output2_missing_source_forward_surface:numpy_receiver" in status["blockers"]


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
        destructive_payload_bit_flip_matrix=_bitflip_matrix(),
        output2_boundary_verdict=blocked_boundary,
        surface_provenance=_surface_provenance(),
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
        destructive_payload_bit_flip_matrix=_bitflip_matrix(),
        output2_boundary_verdict=_source_identical_output2_verdict(),
        surface_provenance=_surface_provenance(),
    )

    assert set(row["tensor_names"]) == set(SOURCE_FORWARD_TENSOR_NAMES)
    assert row["passed"] is True
    assert row["launch_gate_clearable"] is True
    assert validate_snerv_source_forward_proof_action_effect(row)["passed"] is True


def test_source_forward_proof_accepts_strict_source_graph_capture_origin() -> None:
    row = build_snerv_source_forward_proof_action_effect(
        action_id=ACTION_ID,
        archive_sha256=ARCHIVE_SHA,
        archive_bytes=123,
        payload_section_hashes={"decoder_payload.output_2": "2" * 64},
        pair_ids=[0],
        tensors_by_surface=_tensor_surfaces(),
        scorer_deltas=_scorer_deltas(),
        destructive_payload_bit_flip=_bitflip(),
        destructive_payload_bit_flip_matrix=_bitflip_matrix(),
        output2_boundary_verdict=_source_identical_output2_verdict(),
        surface_provenance=_surface_provenance(
            capture_origin="official_upstream_trained_checkpoint_source_graph"
        ),
    )

    status = validate_snerv_source_forward_proof_action_effect(row)

    assert row["passed"] is True
    assert row["launch_gate_clearable"] is True
    assert status["passed"] is True
    assert (
        "snerv_source_forward_official_torch_capture_origin_missing"
        not in status["blockers"]
    )


def test_source_forward_proof_rejects_fixture_scope_official_torch_authority() -> None:
    row = build_snerv_source_forward_proof_action_effect(
        action_id=ACTION_ID,
        archive_sha256=ARCHIVE_SHA,
        archive_bytes=123,
        payload_section_hashes={"decoder_payload.output_2": "2" * 64},
        pair_ids=[0],
        tensors_by_surface=_tensor_surfaces(),
        scorer_deltas=_scorer_deltas(),
        destructive_payload_bit_flip=_bitflip(),
        destructive_payload_bit_flip_matrix=_bitflip_matrix(),
        output2_boundary_verdict=_source_identical_output2_verdict(),
        surface_provenance=_surface_provenance(source_scope="official_source_fixture_state"),
    )

    status = validate_snerv_source_forward_proof_action_effect(row)
    assert row["passed"] is False
    assert status["passed"] is False
    assert (
        "snerv_source_forward_official_torch_trained_checkpoint_source_scope_missing"
        in status["blockers"]
    )


def test_source_forward_proof_rejects_official_torch_missing_upstream_capture_authority() -> None:
    provenance = _surface_provenance()
    provenance["official_torch"]["tensor_capture_authority"] = "real_surface_forward_capture"
    row = build_snerv_source_forward_proof_action_effect(
        action_id=ACTION_ID,
        archive_sha256=ARCHIVE_SHA,
        archive_bytes=123,
        payload_section_hashes={"decoder_payload.output_2": "2" * 64},
        pair_ids=[0],
        tensors_by_surface=_tensor_surfaces(),
        scorer_deltas=_scorer_deltas(),
        destructive_payload_bit_flip=_bitflip(),
        destructive_payload_bit_flip_matrix=_bitflip_matrix(),
        output2_boundary_verdict=_source_identical_output2_verdict(),
        surface_provenance=provenance,
    )

    status = validate_snerv_source_forward_proof_action_effect(row)

    assert row["passed"] is False
    assert status["passed"] is False
    assert (
        "snerv_source_forward_official_torch_upstream_tensor_capture_authority_missing"
        in status["blockers"]
    )


def test_source_forward_proof_rejects_official_torch_missing_model_source_sha() -> None:
    provenance = _surface_provenance()
    provenance["official_torch"].pop("model_source_sha256")
    row = build_snerv_source_forward_proof_action_effect(
        action_id=ACTION_ID,
        archive_sha256=ARCHIVE_SHA,
        archive_bytes=123,
        payload_section_hashes={"decoder_payload.output_2": "2" * 64},
        pair_ids=[0],
        tensors_by_surface=_tensor_surfaces(),
        scorer_deltas=_scorer_deltas(),
        destructive_payload_bit_flip=_bitflip(),
        destructive_payload_bit_flip_matrix=_bitflip_matrix(),
        output2_boundary_verdict=_source_identical_output2_verdict(),
        surface_provenance=provenance,
    )

    status = validate_snerv_source_forward_proof_action_effect(row)

    assert row["passed"] is False
    assert status["passed"] is False
    assert "snerv_source_forward_official_torch_model_source_sha256_invalid" in status["blockers"]
