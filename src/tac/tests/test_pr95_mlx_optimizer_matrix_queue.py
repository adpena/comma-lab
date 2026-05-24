# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
    assert candidate["candidate_id"].endswith("_seed29_steps1_c36")
    assert candidate["candidate_params"]["optimizer_descriptor_id"] == (
        "pr95_stage1_adamw_baseline_mlx"
    )
    assert candidate["candidate_params"]["stage_index"] == 1
    assert candidate["candidate_params"]["seed"] == 29
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
