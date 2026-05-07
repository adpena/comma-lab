from __future__ import annotations

from pathlib import Path

import pytest

from tools.audit_pr102_zero_byte_tuning_custody import (
    EXPECTED_PR100_SHA256,
    build_pr102_zero_byte_tuning_custody,
)

REPO = Path(__file__).resolve().parents[3]


def test_pr102_zero_byte_tuning_custody_identifies_wrong_generic_intake_archive() -> None:
    pr102_intake = REPO / "experiments/results/public_pr_intake_full/public_pr102_intake_20260505_auto"
    pr100_archive = REPO / "experiments/results/public_pr100_intake_20260504_codex/archive.zip"
    if not pr102_intake.exists() or not pr100_archive.exists():
        pytest.skip("local public PR100/PR102 intake artifacts are not present")

    payload = build_pr102_zero_byte_tuning_custody(
        pr102_intake_dir=pr102_intake,
        pr100_archive=pr100_archive,
        correct_pr102_archive=pr100_archive,
        repo_root=REPO,
    )

    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_source_schema_review"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["correct_pr102_archive"]["sha256"] == EXPECTED_PR100_SHA256
    assert payload["pr100_reference_archive"]["sha256"] == EXPECTED_PR100_SHA256
    assert payload["correct_pr102_archive"]["members"] == ["0.bin"]
    assert payload["existing_pr102_intake_archive_wrong"] is True
    assert "existing_pr102_intake_archive_is_wrong_release_asset" in payload["readiness_blockers"]
    assert payload["zero_byte_runtime_contract"]["archive_sha256_equal"] is True
    assert payload["zero_byte_runtime_contract"]["archive_byte_delta"] == 0
    assert payload["zero_byte_runtime_contract"]["delta_scale"] == 0.0095
    assert payload["zero_byte_runtime_contract"]["frame0_red_add_one"] is True
    assert payload["compress_contract"]["points_to_pr100_release"] is True
    assert "pr102_exact_cuda_replay_missing" in payload["dispatch_blockers"]
