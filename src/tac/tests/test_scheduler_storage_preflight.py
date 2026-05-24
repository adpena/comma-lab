# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from comma_lab.operator_storage_waterfall import POLICY_ID, POLICY_SCHEMA
from comma_lab.scheduler.storage_preflight import (
    build_scheduler_storage_preflight_experiment,
    validate_scheduler_storage_preflight_config,
)
from tools import compact_experiment_artifacts, plan_experiment_storage


def _flag_values(command: list[str], flag: str) -> list[str]:
    return [command[index + 1] for index, item in enumerate(command[:-1]) if item == flag]


def test_storage_preflight_defaults_to_operator_policy() -> None:
    experiment = build_scheduler_storage_preflight_experiment(
        experiment_id="scheduler_preflight",
        lane_id="lane_storage_fixture",
        tags=["scheduler-preflight", "storage", "cleanup", "no-score-authority"],
        artifact_prefix="fixture",
        date="20260524T010203Z",
        results_root="/Volumes/VertigoDataTier/pact/experiments/results/fixture",
        proactive_cleanup_execute=True,
        proactive_cleanup_action="move",
    )

    storage_step, cleanup_step = experiment["steps"]
    storage_command = storage_step["command"]
    cleanup_command = cleanup_step["command"]

    assert _flag_values(storage_command, "--storage-tier") == [
        "vertigo=/Volumes/VertigoDataTier/pact",
        "apdatastore=/Volumes/APDataStore/pact",
    ]
    assert _flag_values(cleanup_command, "--cold-store-root") == [
        "/Volumes/VertigoDataTier/pact/cold_store",
        "/Volumes/APDataStore/pact/cold_store",
    ]
    assert _flag_values(cleanup_command, "--journal-output") == [
        ".omx/research/fixture_proactive_cleanup_20260524T010203Z.json.journal.jsonl"
    ]
    metadata = experiment["metadata"]
    assert metadata["operator_storage_policy"]["policy_id"] == POLICY_ID
    assert metadata["operator_storage_policy"]["schema"] == POLICY_SCHEMA
    artifact_metadata = metadata["artifact_catalog_metadata"]
    assert artifact_metadata["policy_id"] == POLICY_ID
    assert artifact_metadata["storage_plan_path"] == (
        ".omx/research/fixture_storage_plan_20260524T010203Z.json"
    )
    assert artifact_metadata["cleanup_plan_path"] == (
        ".omx/research/fixture_proactive_cleanup_20260524T010203Z.json"
    )
    assert artifact_metadata["journal_path"] == (
        ".omx/research/fixture_proactive_cleanup_20260524T010203Z.json.journal.jsonl"
    )
    assert metadata["score_claim"] is False
    assert storage_step["telemetry"]["artifact_catalog_metadata"] == artifact_metadata
    assert cleanup_step["telemetry"]["artifact_catalog_metadata"] == artifact_metadata


def test_storage_preflight_validation_uses_policy_cold_store_defaults() -> None:
    validate_scheduler_storage_preflight_config(
        proactive_cleanup_execute=True,
        proactive_cleanup_action="move",
        proactive_cleanup_cold_store_roots=(),
    )


def test_storage_and_cleanup_tools_emit_artifact_catalog_metadata(
    tmp_path: Path,
) -> None:
    tier_root = tmp_path / "tier"
    tier_root.mkdir()
    storage_plan = ".omx/research/storage_tool_fixture.json"
    cleanup_plan = ".omx/research/cleanup_tool_fixture.json"
    journal = ".omx/research/cleanup_tool_fixture.json.journal.jsonl"

    storage_args = plan_experiment_storage.parse_args(
        [
            "--storage-tier",
            f"fixture={tier_root}",
            "--workload-subdir",
            "work",
            "--reserve-free-gb",
            "0",
            "--create",
            "--allow-local-storage-tier",
            "--storage-plan-path",
            storage_plan,
            "--cleanup-plan-path",
            cleanup_plan,
            "--journal-path",
            journal,
        ]
    )
    storage_payload, ok = plan_experiment_storage.build_payload(storage_args)

    assert ok is True
    assert storage_payload["operator_storage_policy"]["policy_id"] == POLICY_ID
    assert storage_payload["artifact_catalog_metadata"]["storage_plan_path"] == storage_plan
    assert storage_payload["artifact_catalog_metadata"]["cleanup_plan_path"] == cleanup_plan
    assert storage_payload["artifact_catalog_metadata"]["journal_path"] == journal
    assert storage_payload["artifact_catalog_metadata"]["score_claim"] is False

    cleanup_root = tmp_path / "empty_results"
    cleanup_root.mkdir()
    cleanup_output = tmp_path / "cleanup.json"
    assert compact_experiment_artifacts.main(
        [
            str(cleanup_root),
            "--repo-root",
            str(tmp_path),
            "--min-bytes",
            "1",
            "--json-output",
            str(cleanup_output),
            "--journal-output",
            str(tmp_path / "cleanup.json.journal.jsonl"),
            "--storage-plan-path",
            storage_plan,
            "--cleanup-plan-path",
            cleanup_plan,
        ]
    ) == 0
    cleanup_payload = json.loads(cleanup_output.read_text(encoding="utf-8"))
    assert cleanup_payload["operator_storage_policy"]["policy_id"] == POLICY_ID
    assert cleanup_payload["artifact_catalog_metadata"]["storage_plan_path"] == storage_plan
    assert cleanup_payload["artifact_catalog_metadata"]["cleanup_plan_path"] == cleanup_plan
    assert cleanup_payload["artifact_catalog_metadata"]["score_claim"] is False


def test_storage_tool_allows_expected_sha_overwrite(tmp_path: Path) -> None:
    tier_root = tmp_path / "tier"
    tier_root.mkdir()
    output = tmp_path / "storage.json"
    output.write_text('{"prior":true}\n', encoding="utf-8")
    expected_sha = sha256(output.read_bytes()).hexdigest()

    assert (
        plan_experiment_storage.main(
            [
                "--output",
                str(output),
                "--storage-tier",
                f"fixture={tier_root}",
                "--workload-subdir",
                "work",
                "--reserve-free-gb",
                "0",
                "--create",
                "--allow-local-storage-tier",
                "--expected-output-sha256",
                expected_sha,
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["score_claim"] is False
