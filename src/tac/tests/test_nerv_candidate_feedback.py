# SPDX-License-Identifier: MIT
"""Tests for harvestable NeRV candidate byte-feedback rows."""

from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.nerv_candidate_feedback import (
    build_nerv_candidate_feedback_row,
    write_nerv_candidate_feedback_files,
)


def _runner_report(tmp_path: Path) -> dict[str, object]:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"candidate")
    proof = tmp_path / "receiver_proof.json"
    proof.write_text('{"receiver_contract_satisfied": true}\n', encoding="utf-8")
    return {
        "mode": "executed_hi_nerv_mlx_scoreaware_and_exported",
        "execute_family": "hi_nerv",
        "modelsize_candidate_selection": {
            "candidate": {
                "candidate_id": "hinerv_np600_ld12_ed24_dc32_int4",
            },
            "pr95_stack_binding": {
                "schema": "pr95_stack_binding_requirements.v1",
                "satisfied_count": 5,
                "missing_count": 12,
                "complete": False,
                "blockers": [
                    "hi_nerv_real_posenet_teacher_missing",
                    "hi_nerv_archive_in_loop_byte_oracle_missing",
                ],
            },
        },
        "candidate_curriculum_plan": {
            "candidate_id": "hinerv_np600_ld12_ed24_dc32_int4",
            "candidate_conditioned": True,
            "byte_oracle_logging": {
                "candidate_num_pairs": 600,
                "measured_num_pairs": 32,
                "feedback_scope": "partial_pair_advisory",
                "scope_matches_candidate": False,
                "feedback_ready": False,
                "hard_byte_ceiling": 178_000,
                "nominal_total_payload_bytes": 160_000,
                "measured_payload_bytes": None,
                "measured_archive_bytes": 42_000,
                "measured_minus_nominal_bytes": -118_000,
            },
        },
        "archive_path": archive.as_posix(),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": "archive-sha",
        "snerv_binary_profile": {
            "profile_written": True,
            "profile_path": (tmp_path / "snerv_binary_profile.json").as_posix(),
            "verdict": "current_snerv_artifact_rate_blocked_by_explicit_lf_payload",
            "charged_archive_bytes": 10_057_021,
            "snar1_packet_bytes": 10_030_524,
            "lf_payload_bytes": 9_996_235,
            "lf_payload_fraction_of_packet": 0.9965,
            "lf_payload_bytes_per_coeff": 0.1729,
            "blockers": ["snerv_lf_payload_dominates_packet"],
        },
        "receiver_proof_report_paths": [proof.as_posix()],
        "blockers": ["partial_pair_byte_feedback_only"],
    }


def test_build_nerv_candidate_feedback_row_preserves_scope_and_false_authority(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runner_report.json"
    source.write_text('{"schema": "runner"}\n', encoding="utf-8")

    row = build_nerv_candidate_feedback_row(
        runner_report=_runner_report(tmp_path),
        source_report_path=source,
    )

    assert row["schema"] == "nerv_candidate_feedback_row.v1"
    assert row["family"] == "hi_nerv"
    assert row["candidate_id"] == "hinerv_np600_ld12_ed24_dc32_int4"
    assert row["candidate_num_pairs"] == 600
    assert row["measured_num_pairs"] == 32
    assert row["feedback_scope"] == "partial_pair_advisory"
    assert row["scope_matches_candidate"] is False
    assert row["feedback_ready"] is False
    assert row["measured_archive_bytes"] == 42_000
    assert row["archive_sha256"] == "archive-sha"
    assert row["snerv_binary_profile_written"] is True
    assert (
        row["snerv_binary_profile_verdict"]
        == "current_snerv_artifact_rate_blocked_by_explicit_lf_payload"
    )
    assert row["snerv_binary_profile_lf_payload_bytes"] == 9_996_235
    assert row["snerv_binary_profile_lf_payload_fraction_of_packet"] == 0.9965
    assert row["snerv_binary_profile_blockers"] == [
        "snerv_lf_payload_dominates_packet"
    ]
    assert row["pr95_stack_binding_schema"] == (
        "pr95_stack_binding_requirements.v1"
    )
    assert row["pr95_stack_binding_satisfied_count"] == 5
    assert row["pr95_stack_binding_missing_count"] == 12
    assert row["pr95_stack_binding_complete"] is False
    assert row["pr95_stack_binding_blockers"] == [
        "hi_nerv_real_posenet_teacher_missing",
        "hi_nerv_archive_in_loop_byte_oracle_missing",
    ]
    assert row["source_report_sha256"] is not None
    assert row["blockers"] == ["partial_pair_byte_feedback_only"]
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_write_nerv_candidate_feedback_files_writes_json_and_append_ledger(
    tmp_path: Path,
) -> None:
    output = write_nerv_candidate_feedback_files(
        runner_report=_runner_report(tmp_path),
        output_dir=tmp_path / "feedback",
    )

    row_path = Path(output["row_path"])
    ledger_path = Path(output["ledger_path"])
    assert row_path.is_file()
    assert ledger_path.is_file()

    row = json.loads(row_path.read_text())
    ledger_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert output["schema"] == "nerv_candidate_byte_feedback_ledger.v1"
    assert output["append_only"] is True
    assert row["feedback_ready"] is False
    assert ledger_rows == [row]
    assert output["score_claim"] is False
    assert output["promotion_eligible"] is False
