# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler import (
    LOCAL_TRAINING_QUEUE_SCHEMA,
)
from comma_lab.scheduler import (
    build_local_training_execution_queue as package_build_local_training_execution_queue,
)
from comma_lab.scheduler.experiment_queue import ExperimentQueueError
from comma_lab.scheduler.local_training_queue import build_local_training_execution_queue

REPO = Path(__file__).resolve().parents[3]


def test_local_training_queue_is_exported_from_scheduler_package() -> None:
    assert LOCAL_TRAINING_QUEUE_SCHEMA == "local_training_execution_queue_plan.v1"
    assert package_build_local_training_execution_queue is build_local_training_execution_queue


def _plan(tmp_path: Path, *, backend: str = "mlx", device: str = "gpu") -> dict:
    return {
        "schema": "representation_training_probe_plan_v1",
        "candidate_id": f"boostnerv_{backend}_{device}",
        "lane_id": "lane_local_training_fixture",
        "representation_family": "boostnerv",
        "substrate_family": "nerv_family",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "recommended_execution": {
            "schema": "local_training_recommended_execution.v1",
            "tool": "tools/run_local_training_plan.py",
            "training_backend": backend,
            "device": device,
            "output_manifest": str(tmp_path / "manifest.json"),
            "representation_manifest": str(tmp_path / "representation_manifest.json"),
            "python_command_args": [
                ".venv/bin/python",
                "tools/run_local_training_plan.py",
                "--output",
                str(tmp_path / "manifest.json"),
                "--representation-manifest",
                str(tmp_path / "representation_manifest.json"),
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }


def test_local_training_queue_compiles_mlx_plan(tmp_path: Path) -> None:
    queue = build_local_training_execution_queue(
        [_plan(tmp_path)],
        queue_id="local_training_fixture",
        repo_root=tmp_path,
        local_cpu_concurrency=8,
        local_mlx_concurrency=2,
        timeout_seconds=3600,
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"]["local_mlx"] == 2
    experiment = queue["experiments"][0]
    assert experiment["metadata"]["training_backend"] == "mlx"
    assert experiment["metadata"]["score_claim"] is False
    step = experiment["steps"][0]
    assert step["resources"]["kind"] == "local_mlx"
    assert step["timeout_seconds"] == 3600
    assert any(
        condition["type"] == "json_false_authority"
        and condition["path"] == "manifest.json"
        for condition in step["postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["path"] == "representation_manifest.json"
        and condition["key"] == "schema"
        for condition in step["postconditions"]
    )


def test_local_training_queue_maps_mlx_cpu_to_local_cpu(tmp_path: Path) -> None:
    queue = build_local_training_execution_queue(
        [_plan(tmp_path, backend="mlx", device="cpu")],
        queue_id="local_training_fixture",
        repo_root=tmp_path,
    )

    assert queue["experiments"][0]["steps"][0]["resources"]["kind"] == "local_cpu"


def test_local_training_queue_maps_local_numpy_to_local_cpu(tmp_path: Path) -> None:
    queue = build_local_training_execution_queue(
        [_plan(tmp_path, backend="local_numpy", device="local_numpy")],
        queue_id="local_training_fixture",
        repo_root=tmp_path,
    )

    experiment = queue["experiments"][0]
    assert experiment["metadata"]["training_backend"] == "local_numpy"
    assert experiment["steps"][0]["resources"]["kind"] == "local_cpu"


def test_local_training_queue_honors_explicit_scheduler_resource_kind(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, backend="local_numpy", device="auto")
    plan["recommended_execution"]["scheduler_resource_kind"] = "local_cpu"
    queue = build_local_training_execution_queue(
        [plan],
        queue_id="local_training_fixture",
        repo_root=tmp_path,
    )

    assert queue["experiments"][0]["steps"][0]["resources"]["kind"] == "local_cpu"


def test_local_training_queue_rejects_truthy_authority(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["recommended_execution"]["score_claim"] = True

    with pytest.raises(ExperimentQueueError, match="score_claim"):
        build_local_training_execution_queue(
            [plan],
            queue_id="local_training_fixture",
            repo_root=tmp_path,
        )


def test_local_training_queue_rejects_command_output_manifest_mismatch(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    command = plan["recommended_execution"]["python_command_args"]
    command[command.index("--output") + 1] = str(tmp_path / "other_manifest.json")

    with pytest.raises(ExperimentQueueError, match=r"output.*does not match"):
        build_local_training_execution_queue(
            [plan],
            queue_id="local_training_fixture",
            repo_root=tmp_path,
        )


def test_local_training_queue_accepts_output_dir_for_pr95_style_manifest(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    output_dir = tmp_path / "pr95_smoke"
    plan["recommended_execution"]["output_manifest"] = str(output_dir / "manifest.json")
    plan["recommended_execution"]["representation_manifest"] = str(
        output_dir / "representation_training_manifest.json"
    )
    plan["recommended_execution"]["tool"] = "tools/run_pr95_local_training_probe.py"
    plan["recommended_execution"]["python_command_args"] = [
        ".venv/bin/python",
        "tools/run_pr95_local_training_probe.py",
        "--output-dir",
        str(output_dir),
    ]

    queue = build_local_training_execution_queue(
        [plan],
        queue_id="local_training_fixture",
        repo_root=tmp_path,
    )

    assert queue["experiments"][0]["steps"][0]["postconditions"][0]["path"] == (
        "pr95_smoke/manifest.json"
    )


def test_build_local_training_execution_queue_cli(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    queue_path = tmp_path / "queue.json"
    plan_path.write_text(
        json.dumps(_plan(tmp_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_local_training_execution_queue.py"),
            "--plan",
            str(plan_path),
            "--output",
            str(queue_path),
            "--queue-id",
            "local_training_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--local-mlx-concurrency",
            "2",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert queue["queue_id"] == "local_training_cli_fixture"
    assert queue["controls"]["max_concurrency"]["local_mlx"] == 2
    summary = json.loads(result.stdout)
    assert summary["score_claim"] is False


def test_build_local_training_execution_queue_cli_refuses_clobber(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    queue_path = tmp_path / "queue.json"
    plan_path.write_text(
        json.dumps(_plan(tmp_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    queue_path.write_text("existing\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_local_training_execution_queue.py"),
            "--plan",
            str(plan_path),
            "--output",
            str(queue_path),
            "--queue-id",
            "local_training_cli_fixture",
            "--repo-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
