# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.analysis.pr95_distortion_practices_guard import (
    PAYLOAD_GUARD_SCHEMA,
    PRACTICES,
    SCHEMA,
    SOURCE_INVENTORY_SCHEMA,
    build_pr95_distortion_practices_payload_guard,
    build_pr95_distortion_practices_row_guard,
    build_pr95_distortion_source_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pr95_distortion_source_inventory_is_source_derived() -> None:
    inventory = build_pr95_distortion_source_inventory(REPO_ROOT)

    assert inventory["schema"] == SOURCE_INVENTORY_SCHEMA
    assert inventory["source_ready"] is True
    assert inventory["blockers"] == []
    check_ids = {row["check_id"] for row in inventory["source_records"]}
    assert "upstream_frame_utils_seq_len_2" in check_ids
    assert "upstream_posenet_uses_yuv6_pair" in check_ids
    assert "pr95_training_eval_roundtrip_ste" in check_ids
    assert "pr95_eight_stage_curriculum_present" in check_ids
    assert len(inventory["practice_source_rows"]) == len(PRACTICES)


def test_pr95_distortion_guard_accepts_hinerv_pr95_curriculum_row() -> None:
    guard = build_pr95_distortion_practices_row_guard(
        _hinerv_row(),
        repo_root=REPO_ROOT,
    )

    assert guard["schema"] == SCHEMA
    assert guard["family"] == "hi_nerv"
    assert guard["launch_allowed"] is True
    assert guard["blockers"] == []
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["scorer_preprocess_eval_roundtrip_yuv6"]["observed"] is True
    assert rows["dual_component_real_scorer_pressure"]["observed"] is True
    assert rows["pr95_staged_qat_coder_curriculum"]["observed"] is True
    assert guard["score_claim"] is False
    assert guard["ready_for_exact_eval_dispatch"] is False


def test_pr95_distortion_guard_blocks_snerv_without_eval_roundtrip() -> None:
    row = _snerv_row()
    command = row["command"]
    command.remove("--snerv-score-aware-long-training-eval-roundtrip-ste")

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert (
        "snerv_pr95_distortion_scorer_preprocess_eval_roundtrip_yuv6_missing"
        in guard["blockers"]
    )
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["scorer_preprocess_eval_roundtrip_yuv6"]["observed"] is False


def test_pr95_distortion_payload_guard_extracts_verdict_rows() -> None:
    payload = {"schema": "example", "selected_local_mlx_experiments": [_snerv_row()]}

    guard = build_pr95_distortion_practices_payload_guard(payload, repo_root=REPO_ROOT)

    assert guard["schema"] == PAYLOAD_GUARD_SCHEMA
    assert guard["candidate_row_count"] == 1
    assert guard["launch_allowed"] is True
    assert guard["blockers"] == []


def _base_command(family: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        family,
        "--num-pairs",
        "600",
        "--epochs",
        "16",
        "--distillation-device",
        "gpu",
        "--segnet-distillation-weight",
        "1.0",
        "--pose-distillation-weight",
        "1.0",
        "--coder-aware-qat",
        "--coder-qat-c1a-entropy-weight",
        "0.0001",
        "--mlx-prefilter-scorer-device",
        "gpu",
        "--mlx-prefilter-scorer-batch-pairs",
        "8",
        "--output-dir",
        "/Volumes/VertigoDataTier/pact/test_pr95_guard",
    ]


def _hinerv_row() -> dict:
    command = _base_command("hi_nerv")
    command.extend(
        [
            "--batch-pairs",
            "8",
            "--hi-nerv-optimizer-policy",
            "pr95_curriculum",
        ]
    )
    return {
        "id": "hi_row",
        "family": "hi_nerv",
        "command": command,
        "score_lowering_gate": {
            "schema": "nerv_long_training_score_lowering_gate.v1",
            "local_mlx_executable": True,
        },
    }


def _snerv_row() -> dict:
    command = _base_command("snerv")
    command.extend(
        [
            "--snerv-score-aware-long-training-batch-pairs",
            "8",
            "--snerv-score-aware-long-training-eval-roundtrip-ste",
            "--snerv-score-aware-long-training-pr95-faithful-curriculum",
            "--snerv-score-aware-long-training-optimizer",
            "pact_muon_adamw",
        ]
    )
    return {
        "id": "snerv_row",
        "family": "snerv",
        "command": command,
        "score_lowering_gate": {
            "schema": "nerv_long_training_score_lowering_gate.v1",
            "local_mlx_executable": True,
        },
    }
