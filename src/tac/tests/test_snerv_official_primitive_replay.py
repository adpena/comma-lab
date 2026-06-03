# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from tac.analysis.snerv_official_primitive_replay import (
    build_snerv_official_primitive_replay_binding,
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
    rows = {row["component_id"]: row for row in payload["component_rows"]}
    assert set(rows) == {"mfu", "hfr", "tub"}
    assert rows["mfu"]["feature_id"] == "official_multi_resolution_fusion_blocks"
    assert rows["mfu"]["primitive_source_replay_proven"] is True
    assert rows["mfu"]["full_stack_source_forward_replay_proven"] is False
    assert rows["mfu"]["missing_source_markers"] == []
    assert rows["mfu"]["missing_test_markers"] == []


def test_snerv_primitive_source_replay_status_is_feature_keyed() -> None:
    row = snerv_primitive_source_replay_status(
        repo_root=REPO,
        feature_id="official_high_frequency_restoration_heads",
    )

    assert row is not None
    assert row["component_id"] == "hfr"
    assert row["primitive_source_replay_proven"] is True
    assert row["status"] == "primitive_source_replay_proven_full_stack_missing"

    assert (
        snerv_primitive_source_replay_status(
            repo_root=REPO,
            feature_id="not_a_tracked_feature",
        )
        is None
    )
