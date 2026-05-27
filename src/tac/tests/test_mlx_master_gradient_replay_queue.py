# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler import (
    MLX_MASTER_GRADIENT_REPLAY_QUEUE_SCHEMA,
)
from comma_lab.scheduler import (
    build_mlx_master_gradient_replay_queue as package_build_queue,
)
from comma_lab.scheduler.experiment_queue import ExperimentQueueError
from comma_lab.scheduler.mlx_master_gradient_replay_queue import (
    build_mlx_master_gradient_replay_queue,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_bundle(path: Path, *, ready: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mlx_master_gradient_replay_bundle.v1",
        "tool": "tools/extract_master_gradient_mlx.py",
        "argv": [
            "--archive",
            "experiments/results/archive.zip",
            "--out",
            ".omx/state/original.npy",
            "--n-pairs",
            "2",
        ],
        "archive": {"sha256": "a" * 64, "bytes": 1234},
        "output": {"npy_sha256": "b" * 64, "npy_shape": [8, 2, 3]},
        "calibration_gate": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": ready,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_mlx_master_gradient_replay_queue_builds_strict_local_mlx_steps(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "bundle.replay_bundle.json")

    queue = build_mlx_master_gradient_replay_queue(
        replay_bundle_paths=[bundle],
        output_root=tmp_path / "runs",
        queue_id="mlx-replay-loop",
        repo_root=REPO_ROOT,
        timeout_seconds=120,
    )

    assert package_build_queue is build_mlx_master_gradient_replay_queue
    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"]["local_mlx"] == 1
    experiment = queue["experiments"][0]
    assert experiment["metadata"]["schema"] == MLX_MASTER_GRADIENT_REPLAY_QUEUE_SCHEMA
    assert experiment["metadata"]["score_claim"] is False
    assert experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
    step = experiment["steps"][0]
    assert step["resources"]["kind"] == "local_mlx"
    assert step["timeout_seconds"] == 120
    assert "--strict" in step["command"]
    assert "--run-id" in step["command"]
    matched_postcondition = next(
        condition
        for condition in step["postconditions"]
        if condition.get("type") == "json_equals" and condition.get("key") == "matched"
    )
    assert matched_postcondition["equals"] is True
    postcondition_types = [condition["type"] for condition in step["postconditions"]]
    assert postcondition_types.count("json_false_authority") == 2
    assert step["telemetry"]["input_artifact_paths"] == [str(bundle)]


def test_mlx_master_gradient_replay_queue_rejects_truthy_dispatch_gate(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(tmp_path / "bundle.replay_bundle.json", ready=True)

    with pytest.raises(ExperimentQueueError, match="ready_for_exact_eval_dispatch"):
        build_mlx_master_gradient_replay_queue(
            replay_bundle_paths=[bundle],
            output_root=tmp_path / "runs",
            queue_id="mlx-replay-loop",
            repo_root=REPO_ROOT,
        )


def test_build_mlx_master_gradient_replay_queue_cli_writes_queue(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle.replay_bundle.json")
    output = tmp_path / "queue.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_mlx_master_gradient_replay_queue.py"),
            "--replay-bundle",
            str(bundle),
            "--output",
            str(output),
            "--output-root",
            str(tmp_path / "runs"),
            "--queue-id",
            "mlx-replay-cli",
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    queue = json.loads(output.read_text(encoding="utf-8"))
    assert queue["experiments"][0]["metadata"]["schema"] == (
        MLX_MASTER_GRADIENT_REPLAY_QUEUE_SCHEMA
    )
    assert queue["experiments"][0]["metadata"]["promotion_eligible"] is False
