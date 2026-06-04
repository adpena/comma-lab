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
