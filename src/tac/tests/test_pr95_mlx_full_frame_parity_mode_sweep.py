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


def test_pr95_mlx_full_frame_parity_mode_sweep_can_append_scope_candidates(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "pr95_scope_plan"
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
            "optimized:none",
            "--include-scope-search-candidates",
            "--scope-block-count",
            "2",
            "--scope-no-presets",
            "--scope-include-individual-modules",
            "--scope-include-pair-blocks",
            "--jobs",
            "2",
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    commands = plan["candidate_commands"]

    assert plan["include_scope_search_candidates"] is True
    assert plan["scope_block_count"] == 2
    assert plan["scope_include_individual_modules"] is True
    assert plan["scope_include_pair_blocks"] is True
    candidate_ids = {row["candidate_id"] for row in commands}
    assert {
        "optimized__none",
        "scope__single_block__block0_kahan_fp32",
        "scope__single_block__block1_kahan_fp32",
        "scope__prefix_blocks__blocks0_1_kahan_fp32",
        "scope__individual_module__module_blocks_1_conv_kahan_fp32",
        "scope__individual_module__module_refine0_kahan_fp32",
    }.issubset(candidate_ids)
    assert len(candidate_ids) == len(commands)
    block0 = next(
        row
        for row in commands
        if row["candidate_id"] == "scope__single_block__block0_kahan_fp32"
    )
    assert block0["candidate_source"] == "canonical_scope_search"
    assert block0["scope_search_kind"] == "single_block"
    assert block0["conv2d_override_preset"] == "none"
    assert block0["conv2d_override_items"] == [
        "blocks.0.conv=kahan_fp32",
        "blocks.0.skip_conv=kahan_fp32",
    ]
    assert "--conv2d-override" in block0["python_command_args"]
