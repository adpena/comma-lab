# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pr95_mlx_full_frame_parity_mode_sweep_plan_is_queueable(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pr95_parity_sweep_plan"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_full_frame_parity_mode_sweep.py"),
            "--archive-zip",
            str(tmp_path / "archive.zip"),
            "--output-dir",
            str(output_dir),
            "--mlx-device",
            "cpu",
            "--candidate",
            "fixed_fp32:none",
            "--candidate",
            "optimized:blocks02_kahan_fp32",
            "--timeout-seconds",
            "7",
            "--max-output-bytes",
            "7000000",
            "--max-mismatch-samples",
            "5",
            "--jobs",
            "3",
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))

    assert stdout["schema"] == (
        "pr95_hnerv_mlx_full_frame_parity_mode_sweep_plan_summary.v1"
    )
    assert plan["schema"] == "pr95_hnerv_mlx_full_frame_parity_mode_sweep_plan.v1"
    assert plan["lane_id"] == "lane_pr95_hnerv_mlx_reproduction"
    assert plan["jobs"] == 3
    assert plan["candidate_count"] == 2
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["rank_or_kill_eligible"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["exact_readiness_refusal"]["ready"] is False
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in plan[
        "exact_readiness_refusal"
    ]["blockers"]

    commands = plan["candidate_commands"]
    assert {row["candidate_id"] for row in commands} == {
        "fixed_fp32__none",
        "optimized__blocks02_kahan_fp32",
    }
    assert len({row["work_dir"] for row in commands}) == 2
    for row in commands:
        args = row["python_command_args"]
        assert "tools/prove_pr95_public_archive_full_frame_parity.py" in args[1]
        assert "--work-dir" in args
        assert row["work_dir"] in args
        assert "--conv2d-accumulation-mode" in args
        assert "--conv2d-override-preset" in args

    recommended = plan["recommended_execution"]
    assert recommended["tool"] == "tools/run_pr95_mlx_full_frame_parity_mode_sweep.py"
    assert "--jobs" in recommended["python_command_args"]
    assert "3" in recommended["python_command_args"]


def test_pr95_mlx_full_frame_parity_mode_sweep_rejects_gpu_fp64(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_pr95_mlx_full_frame_parity_mode_sweep.py"),
            "--archive-zip",
            str(tmp_path / "archive.zip"),
            "--output-dir",
            str(tmp_path / "out"),
            "--mlx-device",
            "gpu",
            "--candidate",
            "fixed_fp64:none",
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert result.returncode != 0
    assert "fixed_fp64 is unsupported on MLX GPU" in result.stderr
