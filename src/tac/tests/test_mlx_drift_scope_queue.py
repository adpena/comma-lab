# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler import (
    MLX_DRIFT_SCOPE_PLAN_SCHEMA,
    MLX_DRIFT_SCOPE_QUEUE_SCHEMA,
)
from comma_lab.scheduler import (
    build_mlx_drift_scope_search_queue as package_build_mlx_drift_scope_search_queue,
)
from comma_lab.scheduler.experiment_queue import ExperimentQueueError
from comma_lab.scheduler.mlx_drift_scope_queue import build_mlx_drift_scope_search_queue

REPO_ROOT = Path(__file__).resolve().parents[3]


def _plan(tmp_path: Path, *, device: str = "gpu") -> dict:
    output_dir = tmp_path / f"scope_{device}"
    return {
        "schema": MLX_DRIFT_SCOPE_PLAN_SCHEMA,
        "candidate_id": f"scope_{device}",
        "lane_id": "lane_pr95_hnerv_mlx_reproduction",
        "archive_family": "hnerv_pr95",
        "candidate_family": "hnerv_pr95_public_archive",
        "optimization_target": {
            "profile": "contest_video_overfit",
            "target_video": "upstream/videos/0.mkv",
            "video_scope": "contest_single_video",
            "generalizable_against_videos": False,
        },
        "recommended_execution": {
            "schema": "local_mlx_drift_scope_search_recommended_execution.v1",
            "tool": "tools/run_pr95_mlx_conv2d_drift_scope_search.py",
            "mlx_device": device,
            "device": device,
            "resource_kind": "local_mlx" if device == "gpu" else "local_cpu",
            "output_manifest": str(output_dir / "scope_search_summary.json"),
            "output_dir": str(output_dir),
            "expected_output_schema": "pr95_hnerv_mlx_conv2d_drift_scope_search.v1",
            "python_command_args": [
                ".venv/bin/python",
                "tools/run_pr95_mlx_conv2d_drift_scope_search.py",
                "--output-dir",
                str(output_dir),
                "--mlx-device",
                device,
                "--allow-existing-output-dir",
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_mlx_drift_scope_queue_is_exported_from_scheduler_package() -> None:
    assert MLX_DRIFT_SCOPE_PLAN_SCHEMA == "local_mlx_drift_scope_search_plan.v1"
    assert MLX_DRIFT_SCOPE_QUEUE_SCHEMA == "mlx_drift_scope_search_execution_queue_plan.v1"
    assert package_build_mlx_drift_scope_search_queue is build_mlx_drift_scope_search_queue


def test_mlx_drift_scope_queue_compiles_cpu_and_gpu_plans(tmp_path: Path) -> None:
    queue = build_mlx_drift_scope_search_queue(
        [_plan(tmp_path, device="cpu"), _plan(tmp_path, device="gpu")],
        queue_id="drift_scope_fixture",
        repo_root=tmp_path,
        local_cpu_concurrency=4,
        local_mlx_concurrency=2,
        timeout_seconds=1800,
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"] == {"local_cpu": 4, "local_mlx": 2}
    cpu_step = queue["experiments"][0]["steps"][0]
    gpu_step = queue["experiments"][1]["steps"][0]
    assert cpu_step["resources"]["kind"] == "local_cpu"
    assert gpu_step["resources"]["kind"] == "local_mlx"
    assert gpu_step["timeout_seconds"] == 1800
    assert queue["experiments"][1]["metadata"]["schema"] == MLX_DRIFT_SCOPE_QUEUE_SCHEMA
    assert queue["experiments"][1]["metadata"]["optimization_profile"] == (
        "contest_video_overfit"
    )
    assert queue["experiments"][1]["metadata"]["score_claim"] is False
    assert any(
        condition["type"] == "json_equals"
        and condition["key"] == "optimization_target.profile"
        and condition["equals"] == "contest_video_overfit"
        for condition in gpu_step["postconditions"]
    )
    assert any(
        condition["type"] == "json_false_authority"
        for condition in gpu_step["postconditions"]
    )


def test_mlx_drift_scope_queue_rejects_output_mismatch(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["recommended_execution"]["python_command_args"][
        plan["recommended_execution"]["python_command_args"].index("--output-dir") + 1
    ] = str(tmp_path / "other")

    with pytest.raises(ExperimentQueueError, match="output-dir"):
        build_mlx_drift_scope_search_queue(
            [plan],
            queue_id="drift_scope_fixture",
            repo_root=tmp_path,
        )


def test_pr95_scope_search_plan_only_builds_queueable_plan(tmp_path: Path) -> None:
    output_dir = tmp_path / "pr95_scope"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_conv2d_drift_scope_search.py"),
            "--output-dir",
            str(output_dir),
            "--mlx-device",
            "gpu",
            "--optimization-profile",
            "contest_video_overfit",
            "--target-video",
            "upstream/videos/0.mkv",
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    assert stdout["schema"] == "local_mlx_drift_scope_search_plan_summary.v1"
    assert plan["schema"] == MLX_DRIFT_SCOPE_PLAN_SCHEMA
    assert plan["optimization_target"]["profile"] == "contest_video_overfit"
    assert plan["optimization_target"]["contest_overfit_allowed"] is True
    assert plan["optimization_target"]["portable_to"]
    assert plan["recommended_execution"]["resource_kind"] == "local_mlx"
    assert "--allow-existing-output-dir" in plan["recommended_execution"]["python_command_args"]

    queue = build_mlx_drift_scope_search_queue(
        [plan],
        queue_id="pr95_scope_fixture",
        repo_root=REPO_ROOT,
        local_mlx_concurrency=2,
    )
    assert queue["experiments"][0]["steps"][0]["resources"]["kind"] == "local_mlx"
