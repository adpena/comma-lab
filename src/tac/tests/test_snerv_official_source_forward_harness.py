# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.analysis.snerv_official_source_forward_harness import (
    DEFAULT_OFFICIAL_SNERV_REPO,
    OFFICIAL_SNERV_SHA,
    SCHEMA,
    TRAINED_CHECKPOINT_MAPPING_SCHEMA,
    build_snerv_official_source_forward_harness_artifact,
    build_snerv_official_trained_checkpoint_mapping_manifest,
)
from tac.analysis.snerv_official_source_parity_audit import (
    build_snerv_official_source_parity_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _official_repo() -> Path:
    if not DEFAULT_OFFICIAL_SNERV_REPO.exists():
        pytest.skip(f"official SNeRV checkout is absent: {DEFAULT_OFFICIAL_SNERV_REPO}")
    return DEFAULT_OFFICIAL_SNERV_REPO


def test_snerv_official_trained_checkpoint_mapping_manifest_proves_mfu_hfr() -> None:
    state = _minimal_official_decoder_state(decoder_len=8)

    manifest = build_snerv_official_trained_checkpoint_mapping_manifest(
        state,
        decoder_len=None,
        state_dict_kind="unit_test_official_checkpoint",
        source="unit-test",
    )

    assert manifest["schema"] == TRAINED_CHECKPOINT_MAPPING_SCHEMA
    assert manifest["official_trained_checkpoint_loaded"] is True
    assert manifest["decoder_len"] == 8
    assert manifest["decoder_len_source"] == "inferred_from_decoder_prefixes"
    assert manifest["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"] is True
    assert manifest["official_tub_temporal_encoder_weight_mapping_proven"] is False
    assert manifest["mapped_weight_key_count"] == len(state)
    receiver_keys = {row["receiver_key"] for row in manifest["weight_entries"]}
    assert "hfr.lh.conv1.weight" in receiver_keys
    assert "mfu.upsample_mid.weight" in receiver_keys
    assert "mfu.rb_high.residual_blocks.0.conv2.bias" in receiver_keys
    rows = {row["component_id"]: row for row in manifest["component_rows"]}
    assert rows["hfr"]["trained_checkpoint_weight_mapping_proven"] is True
    assert rows["mfu"]["trained_checkpoint_weight_mapping_proven"] is True
    assert rows["tub"]["trained_checkpoint_weight_mapping_proven"] is False
    assert "snerv_official_tub_encoder_decoder_weights_not_loaded" in rows["tub"][
        "blockers"
    ]
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" in manifest[
        "blockers"
    ]
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_snerv_official_trained_checkpoint_mapping_manifest_blocks_missing_hfr() -> None:
    state = {
        key: value
        for key, value in _minimal_official_decoder_state(decoder_len=8).items()
        if not key.startswith(("decoder.8.", "decoder.9.", "decoder.10."))
    }

    manifest = build_snerv_official_trained_checkpoint_mapping_manifest(
        state,
        decoder_len=8,
        state_dict_kind="unit_test_official_checkpoint",
        source="unit-test",
    )

    assert manifest["official_trained_checkpoint_loaded"] is True
    assert manifest["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"] is False
    rows = {row["component_id"]: row for row in manifest["component_rows"]}
    assert rows["mfu"]["trained_checkpoint_weight_mapping_proven"] is True
    assert rows["hfr"]["trained_checkpoint_weight_mapping_proven"] is False
    assert "snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete" in rows[
        "hfr"
    ]["blockers"]
    assert "snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete" in manifest[
        "blockers"
    ]


def test_snerv_official_trained_checkpoint_mapping_manifest_consumes_native_receiver_state() -> None:
    state = _minimal_native_receiver_state()

    manifest = build_snerv_official_trained_checkpoint_mapping_manifest(
        state,
        decoder_len=None,
        state_dict_kind="unit_test_native_mlx_receiver_state",
        source="unit-test-native",
    )

    assert manifest["official_trained_checkpoint_loaded"] is True
    assert manifest["state_dict_mapping_dialect"] == "native_mlx_receiver_state"
    assert manifest["decoder_len"] is None
    assert manifest["decoder_len_source"] == "not_applicable_native_receiver_state"
    assert manifest["official_hfr_trained_checkpoint_weight_mapping_proven"] is True
    assert manifest["official_mfu_trained_checkpoint_weight_mapping_proven"] is False
    assert manifest["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"] is False
    assert manifest["official_mfu_receiver_activation_payload_bound"] is True
    assert manifest["official_native_receiver_state_mapping_proven"] is True
    assert manifest["official_tub_temporal_encoder_weight_mapping_proven"] is False
    assert manifest["mapped_weight_key_count"] == 12
    assert manifest["mapped_activation_key_count"] == 3
    receiver_keys = {row["receiver_key"] for row in manifest["weight_entries"]}
    assert "hfr.lh.conv1.weight" in receiver_keys
    assert "hfr.hh.conv2.bias" in receiver_keys
    activation_keys = {row["receiver_key"] for row in manifest["activation_entries"]}
    assert activation_keys == {
        "inputs.mfu.low",
        "inputs.mfu.skip_high",
        "inputs.mfu.skip_mid",
    }
    rows = {row["component_id"]: row for row in manifest["component_rows"]}
    assert rows["hfr"]["trained_checkpoint_weight_mapping_proven"] is True
    assert rows["mfu"]["trained_checkpoint_weight_mapping_proven"] is False
    assert rows["mfu"]["receiver_activation_payload_bound"] is True
    assert rows["tub"]["receiver_activation_payload_bound"] is False
    assert "snerv_official_trained_checkpoint_decoder_len_not_resolved" not in manifest[
        "blockers"
    ]
    assert "snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete" not in manifest[
        "blockers"
    ]
    assert "snerv_official_mfu_hfr_tub_weight_mapping_missing" not in manifest[
        "blockers"
    ]
    assert "snerv_official_mfu_native_receiver_activation_payload_not_upstream_weight_mapping" in manifest[
        "blockers"
    ]
    assert "snerv_official_tub_encoder_decoder_weights_not_loaded" in manifest[
        "blockers"
    ]


def test_snerv_official_source_forward_harness_proves_mfu_hfr_mapping() -> None:
    artifact = build_snerv_official_source_forward_harness_artifact(
        official_repo_dir=_official_repo(),
        repo_root=REPO_ROOT,
        generated_utc="20260604T000000Z",
    )

    assert artifact["schema"] == SCHEMA
    assert artifact["score_claim"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False
    assert artifact["official_repo"]["head_sha"] == OFFICIAL_SNERV_SHA
    assert artifact["official_trained_checkpoint_loaded"] is False
    assert artifact["official_mfu_hfr_source_fixture_forward_parity_passed"] is True
    assert artifact["official_mfu_hfr_tub_forward_parity_passed"] is False
    assert artifact["full_tub_source_forward_parity_proven"] is False
    receiver_replay = artifact["receiver_payload_frame_replay"]
    assert receiver_replay["receiver_runtime_decode_proven"] is True
    assert receiver_replay["frame_producing_official_payload_replay_proven"] is True
    assert receiver_replay["decoded_frames_shape"][0] > 0
    assert receiver_replay["decoded_frames_sha256"]
    assert receiver_replay["official_tub_output2_fusion_executed"] is True
    assert receiver_replay["receiver_frame_decode_consumes_output2"] is True
    assert (
        "snerv_official_tub_output2_receiver_frame_decode_not_bound"
        not in receiver_replay["blockers"]
    )
    assert (
        "snerv_official_tub_output2_receiver_frame_decode_not_bound"
        not in artifact["blockers"]
    )
    assert receiver_replay["source_forward_replay_authority"] is False

    manifest = artifact["official_weight_manifest"]
    assert manifest["state_dict_kind"] == (
        "synthetic_dyadic_source_fixture_not_official_checkpoint"
    )
    assert manifest["official_mfu_hfr_source_fixture_weight_mapping_proven"] is True
    assert manifest["official_tub_temporal_encoder_weight_mapping_proven"] is False
    assert manifest["state_dict_key_count"] == 28
    receiver_keys = {row["receiver_key"] for row in manifest["weight_entries"]}
    assert "mfu.upsample_mid.weight" in receiver_keys
    assert "mfu.rb_mid.input_conv.weight" in receiver_keys
    assert "hfr.lh.conv1.weight" in receiver_keys
    assert "hfr.hh.conv2.bias" in receiver_keys

    rows = {row["component_id"]: row for row in artifact["component_rows"]}
    for component_id in ("mfu", "hfr"):
        row = rows[component_id]
        assert row["source_forward_parity_proven"] is True
        assert row["max_abs_error"] == 0.0
        assert row["official_output_sha256"] == row["portable_output_sha256"]
        assert row["output_hashes_bit_identical"] is True
        assert row["blockers"] == []
        assert row["score_claim"] is False

    tub = rows["tub"]
    assert tub["primitive_source_forward_parity_proven"] is True
    assert tub["portable_output2_fusion_receiver_mapping_proven"] is True
    assert tub["source_forward_parity_proven"] is False
    assert tub["max_abs_error"] == 0.0
    assert tub["graph_input_max_abs_error"] == 0.0
    assert tub["output2_fusion_max_abs_error"] == 0.0
    assert tub["official_output_sha256"] == tub["portable_output_sha256"]
    assert (
        "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing"
        in tub["closed_blockers"]
    )
    assert (
        "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing"
        in tub["closed_blockers"]
    )
    assert (
        "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing"
        in tub["blockers"]
    )
    assert (
        "snerv_official_tub_portable_output2_decoder_weight_mapping_missing"
        in tub["blockers"]
    )
    assert "snerv_official_pytorch_wavelets_runtime_dependency_missing" in tub["blockers"]

    local_gap = artifact["local_receiver_adapter_source_gap"]
    assert local_gap["receiver_safe_adapter_present"] is True
    assert local_gap["official_source_forward_markers_present"] is False
    assert local_gap["source_forward_parity_proven"] is False


def test_snerv_official_source_forward_harness_consumes_receiver_bound_export_without_source_authority() -> None:
    export_report = {
        "schema": "snerv_checkpoint_archive_export.v1",
        "_source_path": "/Volumes/VertigoDataTier/pact/unit/snerv_checkpoint_archive_export.json",
        "_source_sha256": "a" * 64,
        "checkpoint_epoch": 1299,
        "archive_bytes": 91_445,
        "archive_sha256": "b" * 64,
        "packet_bytes": 88_000,
        "packet_sha256": "c" * 64,
        "official_checkpoint_export_binding": {
            "schema": "snerv_official_checkpoint_export_binding.v1",
            "selected_packet_status": "receiver_bound_official_payload",
            "native_checkpoint_export_bound_to_official_payload": True,
            "official_receiver_payload_bound": True,
            "official_receiver_tensor_map_verified": True,
            "official_trained_checkpoint_state_dict_slice_present": False,
            "official_trained_checkpoint_state_dict_mapping_verified": False,
            "blockers": [
                "snerv_official_trained_checkpoint_state_dict_not_loaded",
                "snerv_official_trained_checkpoint_source_forward_replay_missing",
            ],
            "preserved_blockers": [
                "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
                "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
            ],
        },
    }

    artifact = build_snerv_official_source_forward_harness_artifact(
        official_repo_dir=_official_repo(),
        repo_root=REPO_ROOT,
        checkpoint_export_reports=(export_report,),
        generated_utc="20260604T000000Z",
    )

    binding = artifact["official_checkpoint_export_binding_evidence"]
    assert binding["official_export_bound"] is True
    assert binding["official_receiver_payload_bound"] is True
    assert binding["official_receiver_tensor_map_verified"] is True
    assert binding["closed_campaign_blockers"] == [
        "snerv_official_mfu_hfr_tub_export_not_bound"
    ]
    assert "snerv_official_mfu_hfr_tub_export_not_bound" not in binding["blockers"]
    assert artifact["official_export_bound"] is True
    assert artifact["official_trained_checkpoint_loaded"] is False
    assert artifact["full_tub_source_forward_parity_proven"] is False
    assert (
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority"
        in artifact["blockers"]
    )
    assert (
        "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing"
        in artifact["blockers"]
    )
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" in artifact[
        "blockers"
    ]
    assert artifact["score_claim"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False


def test_snerv_official_source_forward_harness_consumes_loaded_native_checkpoint_without_fake_mapping() -> None:
    manifest = build_snerv_official_trained_checkpoint_mapping_manifest(
        {
            "low": np.zeros((2, 3, 4, 6), dtype=np.float32),
            "skip_mid": np.zeros((2, 3, 8, 12), dtype=np.float32),
        },
        decoder_len=None,
        state_dict_kind="unit_test_native_mlx_checkpoint_state_dict",
        source="/Volumes/VertigoDataTier/pact/unit/native.state.npsd",
    )

    artifact = build_snerv_official_source_forward_harness_artifact(
        official_repo_dir=_official_repo(),
        repo_root=REPO_ROOT,
        trained_checkpoint_mapping_manifests=(manifest,),
        generated_utc="20260604T000000Z",
    )

    trained = artifact["official_trained_checkpoint_mapping_manifest"]
    assert artifact["official_trained_checkpoint_loaded"] is True
    assert trained["official_trained_checkpoint_loaded"] is True
    assert trained["state_dict_key_count"] == 2
    assert artifact["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"] is False
    assert artifact["official_tub_temporal_encoder_weight_mapping_proven"] is False
    assert artifact["official_trained_checkpoint_state_dict_mapping_verified"] is False
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" not in artifact[
        "blockers"
    ]
    tub = next(row for row in artifact["component_rows"] if row["component_id"] == "tub")
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" not in tub[
        "blockers"
    ]
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" not in tub[
        "preserved_blockers"
    ]
    assert "snerv_official_trained_checkpoint_decoder_len_not_resolved" not in artifact[
        "blockers"
    ]
    assert "snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete" in artifact[
        "blockers"
    ]
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" in artifact[
        "blockers"
    ]
    assert artifact["full_tub_source_forward_parity_proven"] is False
    assert artifact["score_claim"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False


def test_snerv_source_audit_consumes_harness_artifact_without_fake_pass(
    tmp_path: Path,
) -> None:
    artifact = build_snerv_official_source_forward_harness_artifact(
        official_repo_dir=_official_repo(),
        repo_root=REPO_ROOT,
        generated_utc="20260604T000000Z",
    )
    artifact_path = tmp_path / "snerv_forward_harness.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")

    report = build_snerv_official_source_parity_audit(
        official_repo_dir=_official_repo(),
        repo_root=REPO_ROOT,
        official_forward_parity_artifact_path=artifact_path,
        generated_utc="20260604T000000Z",
    )

    artifact_row = report["official_forward_parity_artifact_row"]
    assert artifact_row["status"] == "present"
    assert artifact_row["parity_passed"] is False
    assert artifact_row["parity_falsified"] is False
    assert "component_not_proven:tub" in artifact_row["blockers"]
    assert "component_not_proven:mfu" not in artifact_row["blockers"]
    assert "component_not_proven:hfr" not in artifact_row["blockers"]
    assert report["official_mfu_hfr_tub_parity_proven"] is False


def _minimal_official_decoder_state(decoder_len: int) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for offset in range(3):
        prefix = f"decoder.{decoder_len + offset}"
        state[f"{prefix}.conv1.weight"] = np.zeros((3, 3, 1, 1), dtype=np.float32)
        state[f"{prefix}.conv1.bias"] = np.zeros((3,), dtype=np.float32)
        state[f"{prefix}.conv2.weight"] = np.zeros((3, 3, 3, 3), dtype=np.float32)
        state[f"{prefix}.conv2.bias"] = np.zeros((3,), dtype=np.float32)
    for offset in (3, 5):
        prefix = f"decoder.{decoder_len + offset}"
        state[f"{prefix}.weight"] = np.zeros((3, 3, 2, 2), dtype=np.float32)
        state[f"{prefix}.bias"] = np.zeros((3,), dtype=np.float32)
    for offset in (4, 6):
        prefix = f"decoder.{decoder_len + offset}"
        state[f"{prefix}.main.0.weight"] = np.zeros((3, 3, 3, 3), dtype=np.float32)
        state[f"{prefix}.main.0.bias"] = np.zeros((3,), dtype=np.float32)
        state[f"{prefix}.main.1.0.conv1.weight"] = np.zeros(
            (3, 3, 3, 3),
            dtype=np.float32,
        )
        state[f"{prefix}.main.1.0.conv1.bias"] = np.zeros((3,), dtype=np.float32)
        state[f"{prefix}.main.1.0.conv2.weight"] = np.zeros(
            (3, 3, 3, 3),
            dtype=np.float32,
        )
        state[f"{prefix}.main.1.0.conv2.bias"] = np.zeros((3,), dtype=np.float32)
    return state


def _minimal_native_receiver_state() -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {
        "low": np.zeros((2, 3, 2, 2), dtype=np.float32),
        "skip_mid": np.zeros((2, 3, 4, 4), dtype=np.float32),
        "skip_high": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }
    for name in ("lh", "hl", "hh"):
        state[f"hfr_{name}_conv1_weight"] = np.zeros((3, 3, 1, 1), dtype=np.float32)
        state[f"hfr_{name}_conv1_bias"] = np.zeros((3,), dtype=np.float32)
        state[f"hfr_{name}_conv2_weight"] = np.zeros((3, 3, 3, 3), dtype=np.float32)
        state[f"hfr_{name}_conv2_bias"] = np.zeros((3,), dtype=np.float32)
    return state
