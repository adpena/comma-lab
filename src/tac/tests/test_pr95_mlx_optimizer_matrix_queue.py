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

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["queue_id"] == "pr95_mlx_matrix_fixture"
    assert queue["controls"]["max_concurrency"]["local_mlx"] == 3
    assert len(queue["experiments"]) == 2
    assert all(
        experiment["metadata"]["ready_for_exact_eval_dispatch"] is False
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
