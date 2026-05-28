# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler import (
    MLX_DRIFT_SCOPE_RECOMMENDATION_BATCH_SCHEMA,
    MLX_DRIFT_SCOPE_RECOMMENDATION_SCHEMA,
    build_mlx_drift_scope_recommendation,
    build_mlx_drift_scope_recommendation_batch,
)
from comma_lab.scheduler.experiment_queue import ExperimentQueueError

REPO_ROOT = Path(__file__).resolve().parents[3]


def _summary(path: Path) -> dict:
    payload = {
        "schema": "pr95_hnerv_mlx_conv2d_drift_scope_search.v1",
        "lane_id": "lane_pr95_hnerv_mlx_reproduction",
        "archive_family": "hnerv_pr95",
        "candidate_family": "hnerv_pr95_public_archive",
        "evidence_grade": "[macOS-MLX research-signal]",
        "optimization_target": {
            "profile": "contest_video_overfit",
            "target_video": "upstream/videos/0.mkv",
        },
        "candidate_count": 2,
        "minimal_no_cliff_candidate": {
            "candidate_id": "preset_blocks02_kahan_fp32",
            "kind": "preset",
            "override_count": 6,
            "max_abs": 0.000946044921875,
            "mean_abs": 0.000037276826333254576,
            "p99_abs": 0.0002,
            "p999_abs": 0.0008,
            "drift_cliff_name": None,
            "conv2d_accumulation_overrides": {
                "blocks.0.conv": "kahan_fp32",
                "blocks.0.skip_conv": "kahan_fp32",
                "blocks.1.conv": "kahan_fp32",
                "blocks.1.skip_conv": "kahan_fp32",
                "blocks.2.conv": "kahan_fp32",
                "blocks.2.skip_conv": "kahan_fp32",
            },
        },
        "best_by_delta_candidate": {
            "candidate_id": "preset_blocks_refine_kahan_fp32",
            "kind": "preset",
            "override_count": 14,
            "max_abs": 0.0008697509765625,
            "mean_abs": 0.00003437255509197712,
            "p99_abs": 0.00018,
            "p999_abs": 0.00075,
            "drift_cliff_name": None,
            "conv2d_accumulation_overrides": {"blocks.0.conv": "kahan_fp32"},
        },
        "exact_readiness_refusal": {"ready": False, "blockers": ["local_only"]},
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_mlx_drift_scope_harvest_selects_minimal_no_cliff(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = _summary(summary_path)

    recommendation = build_mlx_drift_scope_recommendation(
        summary,
        summary_path=summary_path,
        repo_root=tmp_path,
    )

    assert recommendation["schema"] == MLX_DRIFT_SCOPE_RECOMMENDATION_SCHEMA
    assert recommendation["selected_candidate_id"] == "preset_blocks02_kahan_fp32"
    assert recommendation["recommended_conv2d_override_preset"] == (
        "blocks02_kahan_fp32"
    )
    assert recommendation["recommended_override_count"] == 6
    assert recommendation["no_cliff"] is True
    assert recommendation["score_claim"] is False
    assert recommendation["adoption"]["flags"] == [
        "--mlx-gpu-drift-conv2d-override-preset",
        "blocks02_kahan_fp32",
    ]


def test_mlx_drift_scope_harvest_can_select_best_delta(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = _summary(summary_path)

    recommendation = build_mlx_drift_scope_recommendation(
        summary,
        summary_path=summary_path,
        repo_root=tmp_path,
        selection_policy="best_delta",
    )

    assert recommendation["selected_candidate_id"] == "preset_blocks_refine_kahan_fp32"
    assert recommendation["recommended_conv2d_override_preset"] == (
        "blocks_refine_kahan_fp32"
    )


def test_mlx_drift_scope_harvest_batch_ranks_primary_by_scope_then_delta(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    summary = _summary(summary_path)

    batch = build_mlx_drift_scope_recommendation_batch(
        [summary],
        summary_paths=[summary_path],
        repo_root=tmp_path,
    )

    assert batch["schema"] == MLX_DRIFT_SCOPE_RECOMMENDATION_BATCH_SCHEMA
    assert batch["primary_recommendation"]["selected_candidate_id"] == (
        "preset_blocks02_kahan_fp32"
    )
    assert batch["ready_for_exact_eval_dispatch"] is False


def test_mlx_drift_scope_harvest_rejects_truthy_authority(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = _summary(summary_path)
    summary["score_claim"] = True

    with pytest.raises(ExperimentQueueError, match="score_claim"):
        build_mlx_drift_scope_recommendation(
            summary,
            summary_path=summary_path,
            repo_root=tmp_path,
        )


def test_mlx_drift_scope_harvest_cli(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    output = tmp_path / "recommendation.json"
    _summary(summary_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "harvest_mlx_drift_scope_recommendation.py"),
            "--summary",
            str(summary_path),
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["primary_recommended_conv2d_override_preset"] == (
        "blocks02_kahan_fp32"
    )
    assert payload["primary_recommendation"]["selected_candidate_id"] == (
        "preset_blocks02_kahan_fp32"
    )
