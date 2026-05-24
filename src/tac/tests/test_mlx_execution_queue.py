# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler import (
    MLX_ACQUISITION_FOLLOWUP_SCHEMA,
    MLX_EXECUTION_QUEUE_SCHEMA,
)
from comma_lab.scheduler import (
    build_mlx_scorer_response_execution_queue as package_build_mlx_scorer_response_execution_queue,
)
from comma_lab.scheduler.experiment_queue import ExperimentQueueError
from comma_lab.scheduler.mlx_execution_queue import (
    build_mlx_scorer_response_execution_queue,
)
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_execution_plan import (
    build_mlx_scorer_response_execution_plan,
)
from tac.local_acceleration.mlx_profile_stability import build_profile_stability_manifest

REPO = Path(__file__).resolve().parents[3]


def test_mlx_execution_queue_is_exported_from_scheduler_package() -> None:
    assert MLX_EXECUTION_QUEUE_SCHEMA == "mlx_scorer_response_execution_queue_plan.v1"
    assert MLX_ACQUISITION_FOLLOWUP_SCHEMA == "mlx_scorer_response_acquisition_followup.v1"
    assert (
        package_build_mlx_scorer_response_execution_queue
        is build_mlx_scorer_response_execution_queue
    )


def _profile(device: str = "cpu") -> dict:
    return {
        "schema_version": "mlx_scorer_response_profile.v1",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "reference_cache_dir": "/tmp/reference-cache",
        "candidate_cache_dir": "/tmp/candidate-cache",
        "archive_size_bytes": 178417,
        "start_pair": 16,
        "max_pairs": 4,
        "rows": [
            {
                "device": device,
                "batch_pairs": 1,
                "n_samples": 4,
                "pair_window": [16, 20],
                "canonical_score": 0.178195,
                "avg_posenet_dist": 0.000006,
                "avg_segnet_dist": 0.000515,
                "posenet_sha256": "p" * 64,
                "segnet_sha256": "s" * 64,
                "pairs_per_second": 12.0 if device == "gpu" else 1.0,
                "start_pair": 16,
            }
        ],
    }


def _plan(tmp_path: Path, *, device: str = "cpu", name: str = "response") -> dict:
    manifest = build_profile_stability_manifest(_profile(device), baseline_device=device)
    return build_mlx_scorer_response_execution_plan(
        manifest,
        response_output=tmp_path / f"{name}.json",
        components_dir=tmp_path / f"{name}_components",
        allow_gpu_research_signal=device == "gpu",
    )


def test_mlx_execution_queue_compiles_cpu_and_gpu_plans(tmp_path: Path) -> None:
    queue = build_mlx_scorer_response_execution_queue(
        [
            _plan(tmp_path, device="cpu", name="cpu_response"),
            _plan(tmp_path, device="gpu", name="gpu_response"),
        ],
        queue_id="mlx_queue_fixture",
        repo_root=tmp_path,
        local_cpu_concurrency=4,
        local_mlx_concurrency=1,
        timeout_seconds=1800,
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"] == {"local_cpu": 4, "local_mlx": 1}
    cpu_step = queue["experiments"][0]["steps"][0]
    gpu_step = queue["experiments"][1]["steps"][0]
    assert cpu_step["resources"]["kind"] == "local_cpu"
    assert gpu_step["resources"]["kind"] == "local_mlx"
    assert cpu_step["timeout_seconds"] == 1800
    assert any(
        condition["type"] == "json_false_authority"
        and condition["axis_equals"] == EVIDENCE_TAG_MLX
        for condition in gpu_step["postconditions"]
    )
    assert gpu_step["telemetry"]["recursive"] is True
    assert queue["experiments"][1]["metadata"]["score_claim"] is False
    assert queue["experiments"][1]["metadata"]["tool"] == (
        "comma_lab.scheduler.mlx_execution_queue"
    )
    assert queue["experiments"][1]["metadata"]["plan_count"] == 2


def test_mlx_execution_queue_can_append_acquisition_followup(tmp_path: Path) -> None:
    queue = build_mlx_scorer_response_execution_queue(
        [_plan(tmp_path, device="gpu", name="gpu_response")],
        queue_id="mlx_queue_fixture",
        repo_root=tmp_path,
        lane_id="lane_mlx_batch_fixture",
        include_acquisition_followup=True,
        acquisition_baseline_response_paths=[tmp_path / "baseline_response.json"],
        acquisition_run_root=tmp_path / "acquisition_runs",
        acquisition_campaign_id="batch_fixture",
        acquisition_candidate_limit=7,
        acquisition_campaign_plan_max_k=3,
        acquisition_total_byte_budget=1234,
        acquisition_materializer_execution_limit=2,
        acquisition_max_steps=1,
        emit_acquisition_staircase_plan=True,
    )

    experiment = queue["experiments"][0]
    assert experiment["metadata"]["include_acquisition_followup"] is True
    assert experiment["metadata"]["acquisition_followup_schema"] == MLX_ACQUISITION_FOLLOWUP_SCHEMA
    assert experiment["metadata"]["acquisition_requires_window_baseline"] is True
    assert experiment["metadata"]["score_claim"] is False
    assert len(experiment["steps"]) == 3
    response_step, dataset_step, followup_step = experiment["steps"]
    assert response_step["resources"]["kind"] == "local_mlx"
    assert dataset_step["id"] == "build_mlx_window_response_dataset"
    assert dataset_step["requires"] == ["run_mlx_scorer_response"]
    assert "tools/build_mlx_window_response_dataset.py" in dataset_step["command"]
    assert "--baseline-response" in dataset_step["command"]
    assert followup_step["id"] == "build_mlx_acquisition_batch"
    assert followup_step["requires"] == ["build_mlx_window_response_dataset"]
    assert followup_step["resources"]["kind"] == "local_cpu"
    assert followup_step["telemetry"]["schema"] == MLX_ACQUISITION_FOLLOWUP_SCHEMA
    command = followup_step["command"]
    assert command[:2] == [".venv/bin/python", "tools/run_byte_shaving_materializer_campaign.py"]
    assert "--scorer-response" in command
    assert "--inverse-scorer-allow-native-mlx-window-objective" in command
    assert command[command.index("--candidate-limit") + 1] == "7"
    assert command[command.index("--campaign-plan-max-k") + 1] == "3"
    assert command[command.index("--total-byte-budget") + 1] == "1234"
    assert command[command.index("--materializer-execution-limit") + 1] == "2"
    assert "--emit-staircase-plan" in command
    assert any(
        condition["type"] == "json_equals"
        and condition["equals"] == "scorer_response_dataset.v1"
        for condition in dataset_step["postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["key"] == "schema"
        and condition["equals"] == "byte_shaving_materializer_campaign_run.v1"
        for condition in followup_step["postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["equals"] == "inverse_steganalysis_discrete_action_functional.v1"
        for condition in followup_step["postconditions"]
    )
    assert any(
        condition["type"] == "json_equals"
        and condition["equals"] == "byte_shaving_campaign_plan.v1"
        for condition in followup_step["postconditions"]
    )


def test_mlx_execution_queue_requires_baseline_for_acquisition_followup(
    tmp_path: Path,
) -> None:
    with pytest.raises(ExperimentQueueError, match="baseline"):
        build_mlx_scorer_response_execution_queue(
            [_plan(tmp_path, device="gpu", name="gpu_response")],
            queue_id="mlx_queue_fixture",
            repo_root=tmp_path,
            include_acquisition_followup=True,
        )


def test_mlx_execution_queue_rejects_placeholder_output(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["recommended_execution"]["response_output"] = None

    with pytest.raises(ExperimentQueueError, match="response_output"):
        build_mlx_scorer_response_execution_queue(
            [plan],
            queue_id="mlx_queue_fixture",
            repo_root=tmp_path,
        )


def test_mlx_execution_queue_rejects_truthy_authority(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["score_claim"] = True

    with pytest.raises(ExperimentQueueError, match="score_claim"):
        build_mlx_scorer_response_execution_queue(
            [plan],
            queue_id="mlx_queue_fixture",
            repo_root=tmp_path,
        )


def test_mlx_execution_queue_rejects_nested_truthy_authority(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["recommended_execution"]["ready_for_exact_eval_dispatch"] = True

    with pytest.raises(ExperimentQueueError, match="recommended_execution"):
        build_mlx_scorer_response_execution_queue(
            [plan],
            queue_id="mlx_queue_fixture",
            repo_root=tmp_path,
        )


def test_mlx_execution_queue_rejects_command_device_mismatch(tmp_path: Path) -> None:
    plan = _plan(tmp_path, device="cpu")
    command = plan["recommended_execution"]["python_command_args"]
    command[command.index("--device") + 1] = "gpu"

    with pytest.raises(ExperimentQueueError, match=r"--device='gpu'.*'cpu'"):
        build_mlx_scorer_response_execution_queue(
            [plan],
            queue_id="mlx_queue_fixture",
            repo_root=tmp_path,
        )


def test_mlx_execution_queue_rejects_command_output_mismatch(tmp_path: Path) -> None:
    plan = _plan(tmp_path, device="cpu")
    command = plan["recommended_execution"]["python_command_args"]
    command[command.index("--output") + 1] = str(tmp_path / "other.json")

    with pytest.raises(ExperimentQueueError, match=r"--output=.*does not match"):
        build_mlx_scorer_response_execution_queue(
            [plan],
            queue_id="mlx_queue_fixture",
            repo_root=tmp_path,
        )


def test_mlx_execution_queue_requires_gpu_research_flag_for_gpu_command(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, device="gpu")
    command = plan["recommended_execution"]["python_command_args"]
    command.remove("--allow-gpu-research-signal")

    with pytest.raises(ExperimentQueueError, match="missing --allow-gpu-research-signal"):
        build_mlx_scorer_response_execution_queue(
            [plan],
            queue_id="mlx_queue_fixture",
            repo_root=tmp_path,
        )


def test_build_mlx_execution_queue_cli(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    queue_path = tmp_path / "queue.json"
    plan_path.write_text(json.dumps(_plan(tmp_path), sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_scorer_response_execution_queue.py"),
            "--plan",
            str(plan_path),
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--local-cpu-concurrency",
            "4",
            "--timeout-seconds",
            "1800",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["queue_id"] == "mlx_cli_fixture"
    assert payload["controls"]["max_concurrency"]["local_cpu"] == 4
    summary = json.loads(result.stdout)
    assert summary["score_claim"] is False


def test_build_mlx_execution_queue_cli_with_acquisition_followup(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    queue_path = tmp_path / "queue.json"
    plan_path.write_text(json.dumps(_plan(tmp_path), sort_keys=True), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/build_mlx_scorer_response_execution_queue.py"),
            "--plan",
            str(plan_path),
            "--output",
            str(queue_path),
            "--queue-id",
            "mlx_cli_fixture",
            "--repo-root",
            str(tmp_path),
            "--include-acquisition-followup",
            "--acquisition-baseline-response",
            str(tmp_path / "baseline_response.json"),
            "--acquisition-run-root",
            str(tmp_path / "acquisition_runs"),
            "--acquisition-candidate-limit",
            "5",
            "--acquisition-max-steps",
            "1",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert len(payload["experiments"][0]["steps"]) == 3
    assert (
        payload["experiments"][0]["metadata"]["acquisition_followup_schema"]
        == MLX_ACQUISITION_FOLLOWUP_SCHEMA
    )
