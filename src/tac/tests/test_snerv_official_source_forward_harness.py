# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.analysis.snerv_official_source_forward_harness import (
    DEFAULT_OFFICIAL_SNERV_REPO,
    OFFICIAL_SNERV_SHA,
    SCHEMA,
    build_snerv_official_source_forward_harness_artifact,
)
from tac.analysis.snerv_official_source_parity_audit import (
    build_snerv_official_source_parity_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _official_repo() -> Path:
    if not DEFAULT_OFFICIAL_SNERV_REPO.exists():
        pytest.skip(f"official SNeRV checkout is absent: {DEFAULT_OFFICIAL_SNERV_REPO}")
    return DEFAULT_OFFICIAL_SNERV_REPO


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
    assert tub["source_forward_parity_proven"] is False
    assert tub["max_abs_error"] == 0.0
    assert tub["official_output_sha256"] == tub["portable_output_sha256"]
    assert "snerv_official_tub_encoder_decoder_weights_not_loaded" in tub["blockers"]
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
