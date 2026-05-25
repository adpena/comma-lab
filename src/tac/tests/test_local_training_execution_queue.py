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
    plan = _plan(tmp_path)
    plan["candidate_params"] = {
        "optimizer_descriptor_id": "pr95_stage8_muon_adamw_mlx",
        "optimizer_config_sha256": "a" * 64,
        "parameter_group_lr_policy_id": "embedding_theta1_hidden_muon_adamw",
        "parameter_group_lr_policy_sha256": "b" * 64,
    }
    queue = build_local_training_execution_queue(
        [plan],
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
    assert experiment["metadata"]["optimizer_descriptor_id"] == (
        "pr95_stage8_muon_adamw_mlx"
    )
    assert experiment["metadata"]["parameter_group_lr_policy_id"] == (
        "embedding_theta1_hidden_muon_adamw"
    )
    assert experiment["metadata"]["score_claim"] is False
    step = experiment["steps"][0]
    assert step["resources"]["kind"] == "local_mlx"
    assert step["timeout_seconds"] == 3600
    assert any(
        condition["type"] == "json_false_authority"
        and condition["path"] == "manifest.json"
        for condition in step["postconditions"]
    )
    false_authority = next(
        condition
        for condition in step["postconditions"]
        if condition["type"] == "json_false_authority"
        and condition["path"] == "manifest.json"
    )
    assert "dispatch_packet_ready" in false_authority["false_or_missing"]
    assert any(
        condition["type"] == "json_equals"
        and condition["path"] == "representation_manifest.json"
        and condition["key"] == "schema"
        for condition in step["postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["path"] == "representation_manifest.json"
        and condition["key"] == "candidate_id"
        and condition["equals"] == "boostnerv_mlx_gpu"
        for condition in step["postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["path"] == "representation_manifest.json"
        and condition["key"] == "candidate_params.optimizer_descriptor_id"
        and condition["equals"] == "pr95_stage8_muon_adamw_mlx"
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resource_kind", "modal_gpu"),
        ("scheduler_resource_kind", "cuda_auth"),
    ],
)
def test_local_training_queue_rejects_nonlocal_resource_kind(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    plan = _plan(tmp_path, backend="local_numpy", device="auto")
    plan["recommended_execution"][field] = value

    with pytest.raises(ExperimentQueueError, match="local training resource"):
        build_local_training_execution_queue(
            [plan],
            queue_id="local_training_fixture",
            repo_root=tmp_path,
        )


def test_local_training_queue_keeps_source_sha_out_of_artifact_paths(
    tmp_path: Path,
) -> None:
    source_sha = "a" * 64
    source_dir = tmp_path / "source_tree"
    plan = _plan(tmp_path, backend="local_numpy", device="auto")
    plan["source_dir"] = str(source_dir)
    plan["source_tree_sha256"] = source_sha

    queue = build_local_training_execution_queue(
        [plan],
        queue_id="local_training_fixture",
        repo_root=tmp_path,
    )

    experiment = queue["experiments"][0]
    step = experiment["steps"][0]
    assert experiment["metadata"]["source_tree_sha256"] == source_sha
    assert step["telemetry"]["input_artifact_paths"] == [str(source_dir)]


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


def test_local_training_queue_rejects_duplicate_output_manifests(
    tmp_path: Path,
) -> None:
    plan_a = _plan(tmp_path)
    plan_b = _plan(tmp_path)
    plan_b["candidate_id"] = "other_candidate_same_output"

    with pytest.raises(ExperimentQueueError, match="duplicate output_manifest"):
        build_local_training_execution_queue(
            [plan_a, plan_b],
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


def test_local_training_queue_compiles_hinton_mlx_smoke_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    report = tmp_path / "hinton" / "executed_smoke_100ep_verdict.json"
    plan["schema"] = "hinton_mlx_long_training_smoke_verdict.v1"
    plan["candidate_id"] = "hinton_distilled_scorer_surrogate"
    plan["representation_family"] = "hinton_distilled_scorer_surrogate"
    plan["substrate_family"] = "pr95_mlx_hnerv_family"
    plan["recommended_execution"]["tool"] = "tools/run_hinton_mlx_long_training_smoke.py"
    plan["recommended_execution"]["output_manifest"] = str(report)
    plan["recommended_execution"]["representation_manifest"] = None
    plan["recommended_execution"]["python_command_args"] = [
        ".venv/bin/python",
        "tools/run_hinton_mlx_long_training_smoke.py",
        "--output-report",
        str(report),
        "--execute-smoke",
    ]

    queue = build_local_training_execution_queue(
        [plan],
        queue_id="hinton_mlx_smoke_fixture",
        repo_root=tmp_path,
    )

    experiment = queue["experiments"][0]
    assert experiment["metadata"]["source_plan_schema"] == (
        "hinton_mlx_long_training_smoke_verdict.v1"
    )
    assert experiment["steps"][0]["resources"]["kind"] == "local_mlx"
    assert experiment["steps"][0]["postconditions"][0]["path"] == (
        "hinton/executed_smoke_100ep_verdict.json"
    )
    assert experiment["metadata"]["score_claim"] is False


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
