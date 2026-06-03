# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from tac.analysis.snerv_official_primitive_replay import (
    RECEIVER_RUNTIME_DECODE_SCHEMA,
    build_snerv_official_primitive_replay_binding,
    build_snerv_official_receiver_runtime_decode_contract,
    snerv_primitive_source_replay_status,
)

REPO = Path(__file__).resolve().parents[3]


def test_snerv_official_primitive_replay_binding_splits_authority() -> None:
    payload = build_snerv_official_primitive_replay_binding(repo_root=REPO)

    assert payload["schema"] == "snerv_official_mfu_hfr_tub_primitive_replay_binding.v1"
    assert payload["all_primitive_source_replay_proven"] is True
    assert payload["full_stack_source_forward_replay_proven"] is False
    assert payload["receiver_export_bound"] is False
    assert payload["score_claim"] is False
    runtime = payload["official_receiver_runtime_decode_contract"]
    assert runtime["schema"] == RECEIVER_RUNTIME_DECODE_SCHEMA
    assert runtime["all_runtime_modules_import_safe"] is True
    assert runtime["all_numeric_source_replay_tests_hashed"] is True
    assert runtime["receiver_runtime_decode_proven"] is False
    assert "snerv_mfu_official_weight_packet_decode_missing" in runtime["blockers"]
    rows = {row["component_id"]: row for row in payload["component_rows"]}
    assert set(rows) == {"mfu", "hfr", "tub"}
    assert rows["mfu"]["feature_id"] == "official_multi_resolution_fusion_blocks"
    assert rows["mfu"]["primitive_source_replay_proven"] is True
    assert rows["mfu"]["full_stack_source_forward_replay_proven"] is False
    assert rows["mfu"]["missing_source_markers"] == []
    assert rows["mfu"]["missing_test_markers"] == []
    assert rows["mfu"]["source_sha256"]
    assert rows["mfu"]["test_sha256"]
    receiver_row = rows["mfu"]["receiver_runtime_decode_row"]
    assert receiver_row["runtime_module_import_safe"] is True
    assert receiver_row["numeric_source_replay_test_present"] is True
    assert receiver_row["receiver_runtime_decode_proven"] is False


def test_snerv_primitive_source_replay_status_is_feature_keyed() -> None:
    row = snerv_primitive_source_replay_status(
        repo_root=REPO,
        feature_id="official_high_frequency_restoration_heads",
    )

    assert row is not None
    assert row["component_id"] == "hfr"
    assert row["primitive_source_replay_proven"] is True
    assert row["status"] == "primitive_source_replay_proven_full_stack_missing"
    assert row["receiver_runtime_decode_row"]["status"] == (
        "numeric_source_replay_bound_receiver_decode_missing"
    )
    assert "snerv_hfr_official_weight_packet_decode_missing" in (
        row["receiver_runtime_decode_row"]["blockers"]
    )

    assert (
        snerv_primitive_source_replay_status(
            repo_root=REPO,
            feature_id="not_a_tracked_feature",
        )
        is None
    )


def test_snerv_receiver_runtime_decode_contract_is_hash_backed_and_fail_closed() -> None:
    payload = build_snerv_official_receiver_runtime_decode_contract(repo_root=REPO)

    assert payload["schema"] == RECEIVER_RUNTIME_DECODE_SCHEMA
    assert payload["all_runtime_modules_import_safe"] is True
    assert payload["all_numeric_source_replay_tests_hashed"] is True
    assert payload["receiver_runtime_decode_proven"] is False
    assert payload["receiver_export_bound"] is False
    assert payload["score_claim"] is False
    rows = {row["component_id"]: row for row in payload["component_rows"]}
    assert set(rows) == {"mfu", "hfr", "tub"}
    for component_id, row in rows.items():
        assert row["runtime_module_sha256"]
        assert row["numeric_test_sha256"]
        assert row["runtime_module_import_safe"] is True
        assert row["runtime_entrypoints_present"] is True
        assert row["numeric_source_replay_test_present"] is True
        assert row["receiver_runtime_decode_proven"] is False
        assert row["present_forbidden_receiver_import_markers"] == []
        assert row["status"] == "numeric_source_replay_bound_receiver_decode_missing"
        assert any(component_id in blocker for blocker in row["blockers"])
