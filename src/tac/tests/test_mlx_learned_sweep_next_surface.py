# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.mlx_learned_sweep_next_surface import (
    MLXLearnedSweepNextSurfaceError,
    build_mlx_learned_sweep_next_surface_report,
    render_mlx_learned_sweep_next_surface_markdown,
)
from tac.repo_io import write_json

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _row(
    *,
    sweep_config_id: str,
    ready: bool,
    rank: int,
    exact: bool = False,
) -> dict:
    return {
        "schema": "mlx_dynamic_learned_sweep_row.v1",
        **_false_authority(),
        "dispatch_attempted": False,
        "gpu_launched": False,
        "sweep_config_id": sweep_config_id,
        "queue_candidate_id": f"candidate::{sweep_config_id}::smoke::{rank}",
        "candidate_id": "candidate",
        "optimization_pass_id": "smoke",
        "execution_layer": "claimed_exact" if exact else "local",
        "substrate": "[contest-CPU]" if exact else "[macOS-CPU advisory]",
        "rank": rank,
        "acquisition_value": 0.1 / rank,
        "expected_improvement": 0.01 / rank,
        "ready_for_local_sweep": ready,
        "exact_eval_candidate": exact,
        "allowed_use": "planning_only",
        "selected_pair_indices": [10, 11],
    }


def _plan(rows: list[dict]) -> dict:
    return {
        "schema": "mlx_dynamic_learned_sweep_plan.v1",
        **_false_authority(),
        "summary": {
            "ranked_row_count": len(rows),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "ranked_sweep_rows": rows,
    }


def test_next_surface_routes_ready_local_mlx_rows_to_autopilot_queue() -> None:
    report = build_mlx_learned_sweep_next_surface_report(
        _plan([_row(sweep_config_id="mlx_local_response", ready=True, rank=1)])
    )

    assert report["recommended_next_surface"]["id"] == "build_local_mlx_autopilot_queue"
    assert report["recommended_next_surface"]["status"] == "ready"
    assert report["blockers"] == []
    assert report["summary"]["ready_mlx_local_response_row_count"] == 1
    assert report["score_claim"] is False
    assert report["dispatch_attempted"] is False


def test_next_surface_routes_macos_cpu_advisory_to_supported_queue() -> None:
    report = build_mlx_learned_sweep_next_surface_report(
        _plan(
            [
                _row(sweep_config_id="macos_cpu_advisory", ready=True, rank=1),
                _row(
                    sweep_config_id="contest_cpu_exact_candidate",
                    ready=False,
                    rank=2,
                    exact=True,
                ),
            ]
        )
    )

    assert (
        report["recommended_next_surface"]["id"]
        == "build_macos_cpu_advisory_autopilot_queue"
    )
    assert report["recommended_next_surface"]["status"] == (
        "ready_pending_selection_artifact_validation"
    )
    assert "no_ready_mlx_local_response_rows" in report["routing_notes"]
    assert "macos_cpu_advisory_actuator_supported" in report["routing_notes"]
    assert "macos_cpu_advisory_executor_missing_for_learned_sweep" not in report["blockers"]
    assert "macos_cpu_advisory_is_not_exact_score_authority" in report["blockers"]
    assert "contest_exact_rows_not_ready_without_auth_axis_payload" in report["blockers"]
    assert report["summary"]["ready_macos_cpu_advisory_row_count"] == 1
    assert report["summary"]["contest_exact_candidate_row_count"] == 1
    assert report["top_ready_rows"][0]["sweep_config_id"] == "macos_cpu_advisory"
    assert report["top_ready_rows"][0]["ready_for_exact_eval_dispatch"] is False


def test_next_surface_rejects_truthy_nested_authority() -> None:
    plan = _plan([_row(sweep_config_id="macos_cpu_advisory", ready=True, rank=1)])
    plan["ranked_sweep_rows"][0]["solver_stack_wire_in"] = {"score_claim": True}

    with pytest.raises(MLXLearnedSweepNextSurfaceError, match="score_claim=truthy"):
        build_mlx_learned_sweep_next_surface_report(plan)


def test_next_surface_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "next_surface.json"
    markdown_path = tmp_path / "next_surface.md"
    write_json(
        plan_path,
        _plan([_row(sweep_config_id="macos_cpu_advisory", ready=True, rank=1)]),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/plan_mlx_learned_sweep_next_surface.py"),
            "--plan",
            str(plan_path),
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
            "--repo-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["recommended_next_surface"]["id"] == (
        "build_macos_cpu_advisory_autopilot_queue"
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema"] == "mlx_dynamic_learned_sweep_next_surface_report.v1"
    assert report["source_plan"]["sha256"]
    assert report["promotion_eligible"] is False
    assert "macos_cpu_advisory_actuator_supported" in report["routing_notes"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "MLX Learned Sweep Next Surface" in markdown
    assert "`build_macos_cpu_advisory_autopilot_queue`" in markdown
    assert (
        render_mlx_learned_sweep_next_surface_markdown(report)
        == markdown_path.read_text(encoding="utf-8")
    )
