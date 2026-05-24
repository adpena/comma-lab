# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.local_acceleration.pr95_hnerv_mlx_contract import (
    PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER,
    PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER,
    PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER,
    PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pr95_mlx_optimizer_matrix_cli_emits_queueable_plans(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "matrix"
    queue_path = output_root / "queue.json"
    manifest_path = output_root / "matrix_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_pr95_mlx_optimizer_matrix_queue.py"),
            "--stage",
            "1",
            "--stage",
            "8",
            "--seed",
            "23",
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--synthetic-pairs",
            "2",
            "--output-root",
            str(output_root),
            "--queue-output",
            str(queue_path),
            "--manifest-output",
            str(manifest_path),
            "--queue-id",
            "pr95_mlx_matrix_fixture",
            "--local-mlx-concurrency",
            "3",
            "--write-pr95-public-archive-export",
            "--prove-pr95-runtime-consumption",
            "--runtime-proof-max-output-bytes",
            "7000000",
            "--write-source-faithful-preprocess-smoke",
            "--source-preprocess-shape",
            "1,2,8,10,3",
            "--source-preprocess-camera-hw",
            "11,13",
            "--source-preprocess-gradient-shape",
            "1,2,8,10,3",
            "--write-source-video-preprocess-smoke",
            "--source-video-path",
            "upstream/videos/0.mkv",
            "--source-video-upstream-dir",
            "upstream",
            "--source-video-pair-index",
            "0",
            "--source-video-output-hw",
            "8,10",
            "--source-video-gradient-shape",
            "1,2,8,10,3",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))

    assert summary["schema"] == "pr95_hnerv_mlx_optimizer_matrix_queue_summary_v1"
    assert summary["plan_count"] == 2
    assert manifest["schema"] == "pr95_hnerv_mlx_optimizer_matrix_queue.v1"
    assert manifest["plan_count"] == 2
    assert manifest["refusal_count"] == 0
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["write_pr95_public_archive_export"] is True
    assert manifest["prove_pr95_runtime_consumption"] is True
    assert manifest["write_source_faithful_preprocess_smoke"] is True
    assert manifest["source_preprocess_shape"] == "1,2,8,10,3"
    assert manifest["source_preprocess_camera_hw"] == "11,13"
    assert manifest["source_preprocess_gradient_shape"] == "1,2,8,10,3"
    assert manifest["write_source_video_preprocess_smoke"] is True
    assert manifest["source_video_path"] == "upstream/videos/0.mkv"
    assert manifest["source_video_upstream_dir"] == "upstream"
    assert manifest["source_video_pair_indices"] == [0]
    assert manifest["source_video_output_hw"] == "8,10"
    assert manifest["source_video_gradient_shape"] == "1,2,8,10,3"
    assert manifest["queue_output_sha256"] == summary["queue_sha256"]
    assert {row["stage_index"] for row in manifest["plans"]} == {1, 8}
    assert all(len(row["matrix_cell_id"]) == 64 for row in manifest["plans"])
    assert {row["optimizer_descriptor_id"] for row in manifest["plans"]} == {
        "pr95_stage1_adamw_baseline_mlx",
        "pr95_stage8_muon_adamw_mlx",
    }
    for row in manifest["plans"]:
        plan = json.loads((REPO_ROOT / row["plan"]).read_text(encoding="utf-8"))
        assert plan["score_claim"] is False
        assert plan["recommended_execution"]["resource_kind"] == "local_mlx"
        assert plan["recommended_execution"]["score_claim"] is False
        assert plan["recommended_execution"]["archive_export_manifest"].endswith(
            "pr95_public_archive_export.json"
        )
        assert plan["recommended_execution"]["runtime_consumption_proof"].endswith(
            "runtime_consumption_proof.json"
        )
        assert plan["recommended_execution"][
            "source_faithful_preprocess_smoke"
        ].endswith("source_faithful_preprocess_smoke.json")
        assert "--write-source-faithful-preprocess-smoke" in plan[
            "recommended_execution"
        ]["python_command_args"]
        assert "--write-source-video-preprocess-smoke" in plan[
            "recommended_execution"
        ]["python_command_args"]
        assert "--source-preprocess-shape" in plan["recommended_execution"][
            "python_command_args"
        ]
        assert "--source-video-output-hw" in plan["recommended_execution"][
            "python_command_args"
        ]
        assert "1,2,8,10,3" in plan["recommended_execution"]["python_command_args"]
        assert "11,13" in plan["recommended_execution"]["python_command_args"]
        assert "8,10" in plan["recommended_execution"]["python_command_args"]

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["queue_id"] == "pr95_mlx_matrix_fixture"
    assert queue["controls"]["max_concurrency"]["local_mlx"] == 3
    assert len(queue["experiments"]) == 2
    assert all(
        experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
        for experiment in queue["experiments"]
    )
    assert all(
        any(
            condition["type"] == "json_equals"
            and condition["path"].endswith("runtime_consumption_proof.json")
            and condition["key"] == "runtime_consumption_proven"
            and condition["equals"] is True
            for condition in experiment["steps"][0]["postconditions"]
        )
        for experiment in queue["experiments"]
    )
    assert all(
        any(
            condition["type"] == "json_equals"
            and condition["path"].endswith("source_faithful_preprocess_smoke.json")
            and condition["key"] == "source_faithful_preprocess_ready"
            and condition["equals"] is True
            for condition in experiment["steps"][0]["postconditions"]
        )
        for experiment in queue["experiments"]
    )
    assert all(
        any(
            condition["type"] == "json_equals"
            and condition["path"].endswith("source_faithful_preprocess_smoke.json")
            and condition["key"] == "gradient_probe.gradient_reachable"
            and condition["equals"] is True
            for condition in experiment["steps"][0]["postconditions"]
        )
        for experiment in queue["experiments"]
    )
    assert all(
        any(
            condition["type"] == "json_array_contains"
            and condition["path"].endswith("source_faithful_preprocess_smoke.json")
            and condition["key"] == "exact_readiness_refusal.blockers"
            and condition["contains"]
            == PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER
            for condition in experiment["steps"][0]["postconditions"]
        )
        for experiment in queue["experiments"]
    )
    assert all(
        any(
            condition["type"] == "json_equals"
            and condition["path"].endswith("source_video_preprocess_smoke.json")
            and condition["key"] == "source_video_loader_ready"
            and condition["equals"] is True
            for condition in experiment["steps"][0]["postconditions"]
        )
        for experiment in queue["experiments"]
    )
    assert all(
        any(
            condition["type"] == "json_array_contains"
            and condition["path"].endswith("source_video_preprocess_smoke.json")
            and condition["key"] == "exact_readiness_refusal.blockers"
            and condition["contains"]
            == PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER
            for condition in experiment["steps"][0]["postconditions"]
        )
        for experiment in queue["experiments"]
    )

    validation = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert validation.returncode == 0, validation.stderr
    assert json.loads(validation.stdout)["valid"] is True


def test_pr95_mlx_optimizer_matrix_skips_non_executable_descriptor(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "matrix"
    queue_path = output_root / "queue.json"
    manifest_path = output_root / "matrix_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_pr95_mlx_optimizer_matrix_queue.py"),
            "--stage",
            "8",
            "--seed",
            "23",
            "--optimizer-descriptor-id",
            "pr95_stage8_muon_adamw_mlx",
            "--optimizer-descriptor-id",
            "pr95_muon_all_stages_descriptor_only",
            "--output-root",
            str(output_root),
            "--queue-output",
            str(queue_path),
            "--manifest-output",
            str(manifest_path),
            "--queue-id",
            "pr95_mlx_matrix_refusal_fixture",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["plan_count"] == 1
    assert manifest["refusal_count"] == 1
    assert manifest["refusals"][0]["optimizer_descriptor_id"] == (
        "pr95_muon_all_stages_descriptor_only"
    )
    assert "not executable on MLX" in manifest["refusals"][0]["reason"]
    assert manifest["refusals"][0]["queued"] is False
    assert manifest["refusals"][0]["score_claim"] is False


def test_pr95_mlx_optimizer_matrix_queue_executes_and_harvests_one_cell(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "matrix_exec"
    queue_path = output_root / "queue.json"
    manifest_path = output_root / "matrix_manifest.json"
    queue_id = "pr95_mlx_matrix_exec_fixture"
    state_path = tmp_path / "matrix_exec.sqlite"
    candidate_queue_path = output_root / "optimizer_candidate_queue.json"

    build = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_pr95_mlx_optimizer_matrix_queue.py"),
            "--stage",
            "1",
            "--seed",
            "29",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--synthetic-pairs",
            "2",
            "--output-root",
            str(output_root),
            "--queue-output",
            str(queue_path),
            "--manifest-output",
            str(manifest_path),
            "--queue-id",
            queue_id,
            "--local-mlx-concurrency",
            "1",
            "--write-source-faithful-preprocess-smoke",
            "--source-preprocess-shape",
            "1,2,8,10,3",
            "--source-preprocess-camera-hw",
            "11,13",
            "--source-preprocess-gradient-shape",
            "1,2,8,10,3",
            "--train-on-source-video-pairs",
            "--source-video-path",
            "upstream/videos/0.mkv",
            "--source-video-upstream-dir",
            "upstream",
            "--source-video-pair-index",
            "0",
            "--source-video-output-hw",
            "384,512",
            "--source-video-loss-surface",
            "rgb_yuv6_mse",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert build.returncode == 0, build.stderr

    init = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "init",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert init.returncode == 0, init.stderr

    worker = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "run-worker",
            "--execute",
            "--noncanonical-state-rationale",
            "pytest fixture state path isolated under tmp_path",
            "--max-steps",
            "1",
            "--max-parallel",
            "1",
            "--max-idle-cycles",
            "1",
            "--poll-interval-seconds",
            "0.2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert worker.returncode == 0, worker.stderr
    worker_summary = json.loads(worker.stdout)
    assert worker_summary["success_count"] == 1
    assert worker_summary["failure_count"] == 0
    assert worker_summary["step_results"][0]["failed_postconditions"] == []

    harvest = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "harvest_local_training_optimizer_candidates.py"),
            "--queue",
            str(queue_path),
            "--state",
            str(state_path),
            "--repo-root",
            str(REPO_ROOT),
            "--output",
            str(candidate_queue_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert harvest.returncode == 0, harvest.stderr
    summary = json.loads(harvest.stdout)
    assert summary["n_candidates"] == 1
    assert summary["dispatch_ready_count"] == 0
    candidate_queue = json.loads(candidate_queue_path.read_text(encoding="utf-8"))
    assert candidate_queue["score_claim"] is False
    assert candidate_queue["promotion_eligible"] is False
    assert candidate_queue["rank_or_kill_eligible"] is False
    assert candidate_queue["ready_for_exact_eval_dispatch"] is False
    candidate = candidate_queue["top_k"][0]
    assert candidate["candidate_id"].endswith(
        "_seed29_steps1_c36_source_video_rgb_yuv6"
    )
    assert candidate["candidate_params"]["optimizer_descriptor_id"] == (
        "pr95_stage1_adamw_baseline_mlx"
    )
    assert candidate["candidate_params"]["stage_index"] == 1
    assert candidate["candidate_params"]["seed"] == 29
    assert candidate["candidate_params"]["source_video_training"] is True
    assert candidate["candidate_params"]["training_loss_surface"] == "rgb_yuv6_mse"
    assert candidate["candidate_params"]["target_yuv6_shape"] == [1, 2, 192, 256, 6]
    assert candidate["candidate_params"]["target_source_kind"] == (
        "pr95_source_video_rgb_pairs"
    )
    assert candidate["candidate_params"]["source_faithful_preprocess_smoke_present"] is True
    assert candidate["candidate_params"]["source_preprocess_camera_hw"] == "11,13"
    preprocess_signal = candidate["consumer_payload"]["representation_training_probe"][
        "source_faithful_preprocess"
    ]
    assert preprocess_signal["present"] is True
    assert preprocess_signal["gradient_reachable"] is True
    assert PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER in (
        preprocess_signal["exact_readiness_blockers"]
    )
    assert PR95_SOURCE_VIDEO_RGB_NOT_FULL_SCORER_BLOCKER not in (
        candidate["dispatch_blockers"]
    )
    assert PR95_SEGNET_POSENET_LOSS_UNWIRED_BLOCKER in (
        candidate["dispatch_blockers"]
    )
    assert candidate["ready_for_exact_eval_dispatch"] is False


def test_pr95_mlx_optimizer_matrix_refuses_when_no_executable_plans(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "matrix"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_pr95_mlx_optimizer_matrix_queue.py"),
            "--stage",
            "1",
            "--optimizer-descriptor-id",
            "pr95_stage8_muon_adamw_mlx",
            "--output-root",
            str(output_root),
            "--queue-output",
            str(output_root / "queue.json"),
            "--manifest-output",
            str(output_root / "matrix_manifest.json"),
            "--queue-id",
            "pr95_mlx_matrix_all_refused_fixture",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )

    assert result.returncode != 0
    assert "no executable PR95 MLX matrix plans selected" in result.stderr
    assert not (output_root / "queue.json").exists()
