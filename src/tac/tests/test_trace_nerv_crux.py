# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "trace_nerv_crux.py"


def _run_trace(tmp_path: Path, payload: dict[str, object], *extra: str) -> list[dict]:
    artifact = tmp_path / "training_artifact.json"
    out = tmp_path / "trace_rows.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--training-artifact",
            str(artifact),
            "--out",
            str(out),
            *extra,
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    return rows


def _by_metric(rows: list[dict]) -> dict[str, dict]:
    return {str(row["metric"]): row for row in rows}


def test_trace_nerv_crux_emits_contest_unit_rows_without_score_claim(tmp_path: Path) -> None:
    rows = _run_trace(
        tmp_path,
        {
            "final_loss_components": {
                "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass": 50.0,
                "loss_part_segnet_direct_live_argmax_disagreement": 0.25,
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
                "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.2,
                "loss_part_pose_direct_live_raw_mse": 0.00196,
                "loss_part_pose_direct_live_score_term": 0.14,
                "loss_part_pose_direct_live_score_marginal_wrt_raw_mse": 35.714285714285715,
                "loss_part_pose_direct_live_yuv6_pair_std": 0.22,
                "loss_part_pose_direct_live_yuv6_pair_temporal_delta_std": 0.08,
                "train_time_archive_bytes": 150_000,
            }
        },
    )

    metrics = _by_metric(rows)
    assert metrics["score_weighted_total_unsolved_argmax_mass"][
        "score_units"
    ] == pytest.approx(50.0)
    assert metrics["pose_direct_live_score_term"]["score_units"] == pytest.approx(0.14)
    assert metrics["pose_direct_live_score_marginal_wrt_raw_mse"][
        "value"
    ] == pytest.approx(5.0 / ((10.0 * 0.00196) ** 0.5))
    assert metrics["archive_rate_score"]["score_units"] == pytest.approx(
        25.0 * 150_000 / 37_545_489
    )
    assert {row["authority"] for row in rows} == {
        "macos_mlx_false_authority_no_score_claim"
    }
    for row in rows:
        assert "score_claim" not in row
        assert "promotion_eligible" not in row


def test_trace_nerv_crux_blocks_missing_required_direct_live_posenet(tmp_path: Path) -> None:
    rows = _run_trace(
        tmp_path,
        {"final_loss_components": {"train_time_archive_bytes": 1_000}},
        "--require-direct-live-posenet",
    )

    blockers = {row.get("blocker") for row in rows if row.get("blocker")}
    assert "missing_direct_live_posenet_raw_mse" in blockers
