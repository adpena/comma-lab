# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from comma_lab.operator_storage_waterfall import (
    DEFAULT_WORK_TIER_ORDER,
    FALSE_AUTHORITY_FIELDS,
    operator_cold_store_roots,
    operator_storage_policy_payload,
    operator_storage_tier_cli_specs,
    storage_preflight_artifact_catalog_metadata,
)


def test_operator_storage_waterfall_defaults_to_vertigo_then_apdatastore() -> None:
    payload = operator_storage_policy_payload()

    assert DEFAULT_WORK_TIER_ORDER == (
        ("vertigo", "/Volumes/VertigoDataTier/pact"),
        ("apdatastore", "/Volumes/APDataStore/pact"),
    )
    assert payload["storage_tier_cli_specs"] == [
        "vertigo=/Volumes/VertigoDataTier/pact",
        "apdatastore=/Volumes/APDataStore/pact",
    ]
    assert payload["local_disk_enabled_by_default"] is False
    assert payload["local_disk_enabled"] is False
    assert payload["local_disk_requires_explicit_opt_in"] is True
    for key, value in FALSE_AUTHORITY_FIELDS.items():
        assert payload[key] is value


def test_operator_storage_waterfall_cold_store_roots_follow_tier_order() -> None:
    assert operator_cold_store_roots() == (
        "/Volumes/VertigoDataTier/pact/cold_store",
        "/Volumes/APDataStore/pact/cold_store",
    )


def test_operator_storage_waterfall_allows_explicit_tier_overrides_without_local_default() -> None:
    specs = operator_storage_tier_cli_specs(("fast=/mnt/fast", "deep=/mnt/deep"))
    payload = operator_storage_policy_payload(
        storage_tier_overrides=("fast=/mnt/fast", "deep=/mnt/deep"),
    )

    assert specs == ("fast=/mnt/fast", "deep=/mnt/deep")
    assert payload["cold_store_roots"] == ["/mnt/fast/cold_store", "/mnt/deep/cold_store"]
    assert payload["explicit_storage_tier_override"] is True
    assert payload["local_disk_enabled"] is False


def test_operator_storage_waterfall_rejects_absolute_or_parent_cold_store_subdir() -> None:
    with pytest.raises(ValueError, match="cold_store_subdir must be relative"):
        operator_cold_store_roots(cold_store_subdir="/tmp/cold")
    with pytest.raises(ValueError, match="may not contain"):
        operator_cold_store_roots(cold_store_subdir="../cold")


def test_operator_storage_waterfall_marks_ignored_preflight_artifacts_local_only() -> None:
    metadata = storage_preflight_artifact_catalog_metadata(
        storage_plan_path=".omx/research/dqs1_local_first_storage_plan_20260524.json",
        cleanup_plan_path=".omx/research/dqs1_local_first_proactive_cleanup_20260524.json",
        journal_path=(
            ".omx/research/dqs1_local_first_proactive_cleanup_20260524.json."
            "journal.jsonl"
        ),
    )

    boundary = metadata["tracked_local_boundary"][
        ".omx/research/dqs1_local_first_storage_plan_20260524.json"
    ]
    assert boundary["local_only"] is True
    assert boundary["git_boundary"] == "gitignored_local_control_artifact"

    retention_metadata = storage_preflight_artifact_catalog_metadata(
        storage_plan_path=None,
        cleanup_plan_path=".omx/research/dqs1_proactive_artifact_retention_20260524T052547Z_round000.json",
        journal_path=".omx/research/dqs1_proactive_artifact_retention_20260524T052547Z_round000.json.journal.jsonl",
    )
    retention_boundary = retention_metadata["tracked_local_boundary"][
        ".omx/research/dqs1_proactive_artifact_retention_20260524T052547Z_round000.json"
    ]
    assert retention_boundary["local_only"] is True
    assert retention_boundary["git_boundary"] == "gitignored_local_control_artifact"
