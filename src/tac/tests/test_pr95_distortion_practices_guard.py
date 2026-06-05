# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.analysis.pr95_distortion_practices_guard import (
    PAYLOAD_GUARD_SCHEMA,
    PRACTICES,
    SCHEMA,
    SOURCE_INVENTORY_SCHEMA,
    TELEMETRY_CONTRACT_SCHEMA,
    build_pr95_distortion_practices_payload_guard,
    build_pr95_distortion_practices_row_guard,
    build_pr95_distortion_source_inventory,
    build_pr95_evaluate_scorer_domain_telemetry_contract,
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
    assert rows["official_evaluate_archive_byte_price"]["observed"] is True
    assert rows["scorer_domain_telemetry_contract"]["observed"] is True
    assert rows["pr95_staged_qat_coder_curriculum"]["observed"] is True
    assert guard["score_claim"] is False
    assert guard["ready_for_exact_eval_dispatch"] is False


def test_pr95_distortion_guard_blocks_snerv_without_eval_roundtrip() -> None:
    row = _snerv_row()
    command = row["command"]
    command.remove("--snerv-score-aware-long-training-eval-roundtrip-ste")

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert "snerv_pr95_distortion_scorer_preprocess_eval_roundtrip_yuv6_missing" in guard["blockers"]
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["scorer_preprocess_eval_roundtrip_yuv6"]["observed"] is False


def test_pr95_distortion_guard_blocks_fake_parity_without_byte_binding() -> None:
    row = _hinerv_row()
    del row["upstream_evaluate_score_binding"]

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert "hi_nerv_pr95_distortion_official_evaluate_archive_byte_price_missing" in guard["blockers"]
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["official_evaluate_archive_byte_price"]["observed"] is False


def test_pr95_distortion_guard_blocks_fake_parity_without_scorer_telemetry_contract() -> None:
    row = _hinerv_row()
    del row["pr95_evaluate_scorer_domain_telemetry_contract"]

    guard = build_pr95_distortion_practices_row_guard(row, repo_root=REPO_ROOT)

    assert guard["launch_allowed"] is False
    assert "hi_nerv_pr95_distortion_scorer_domain_telemetry_contract_missing" in guard["blockers"]
    rows = {row["practice_id"]: row for row in guard["practice_rows"]}
    assert rows["scorer_domain_telemetry_contract"]["observed"] is False


def test_pr95_distortion_telemetry_contract_names_evaluate_domains() -> None:
    contract = build_pr95_evaluate_scorer_domain_telemetry_contract("snerv")

    assert contract["schema"] == TELEMETRY_CONTRACT_SCHEMA
    assert contract["segnet_scored_frame_index"] == 1
    assert contract["posenet_scored_frame_indices"] == [0, 1]
    assert contract["argmax_occupancy_gate_required"] is True
    assert contract["fail_closed_on_missing_metrics"] is True
    assert any("snerv_segnet_last_frame_distill" in name for name in contract["segnet_last_frame_argmax_metric_names"])
    assert any("occupied_class_fraction" in name for name in contract["segnet_argmax_occupancy_metric_names"])
    assert any("posenet_yuv6_pair" in name for name in contract["posenet_yuv6_pair_metric_names"])
    assert contract["score_claim"] is False


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
        "--hard-byte-ceiling",
        "3980000",
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
        "hard_byte_ceiling": 3_980_000,
        "upstream_evaluate_score_binding": _upstream_evaluate_score_binding("hi_nerv"),
        "pr95_evaluate_scorer_domain_telemetry_contract": (
            build_pr95_evaluate_scorer_domain_telemetry_contract("hi_nerv")
        ),
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
        "hard_byte_ceiling": 3_980_000,
        "upstream_evaluate_score_binding": _upstream_evaluate_score_binding("snerv"),
        "pr95_evaluate_scorer_domain_telemetry_contract": (
            build_pr95_evaluate_scorer_domain_telemetry_contract("snerv")
        ),
        "score_lowering_gate": {
            "schema": "nerv_long_training_score_lowering_gate.v1",
            "local_mlx_executable": True,
        },
    }


def _upstream_evaluate_score_binding(family: str) -> dict:
    return {
        "schema": "nerv_row_upstream_evaluate_binding.v1",
        "family": family,
        "rate": {
            "archive_authority": "submission_dir/archive.zip.stat().st_size",
            "canonical_denominator_bytes": 37_545_489,
            "rate_price_per_archive_byte": 25 / 37_545_489,
            "raw_output_shape_bytes_are_not_rate_denominator": (1200 * 874 * 1164 * 3),
        },
    }
