# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_preflight import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_SCHEMA,
    build_l5_v2_tt5l_sideinfo_lightning_execution_preflight,
    l5_v2_tt5l_sideinfo_lightning_execution_preflight_json,
    render_l5_v2_tt5l_sideinfo_lightning_execution_preflight_markdown,
)

from .test_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import _write_manifest


def _write_plan(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan(
        manifest=manifest,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        source_commit="a" * 40,
    )
    plan_path = tmp_path / ".omx/research/lightning_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path, plan


def test_tt5l_lightning_execution_preflight_builds_per_axis_claim_templates(
    tmp_path: Path,
) -> None:
    plan_path, plan = _write_plan(tmp_path)

    preflight = build_l5_v2_tt5l_sideinfo_lightning_execution_preflight(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
        claims_text="",
        current_head_commit=str(plan["source_commit"]),
    )

    expected_count = len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS) * len(
        L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
    )
    assert preflight["schema"] == L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_SCHEMA
    assert preflight["score_claim"] is False
    assert preflight["promotion_eligible"] is False
    assert preflight["ready_for_exact_eval_dispatch"] is False
    assert preflight["dispatch_attempted"] is False
    assert preflight["source_plan_commit_matches_current_head"] is True
    assert preflight["cell_count"] == expected_count
    assert preflight["ready_cell_count"] == expected_count
    assert preflight["ready_for_operator_claiming"] is True
    assert preflight["ready_for_provider_dispatch"] is False
    assert preflight["global_blockers"] == [
        "requires_lightning_identity_and_workspace_preflight_before_submit",
        "requires_source_manifest_staged_to_lightning_workspace_before_submit",
        "requires_operator_to_submit_source_plan_spec_commands",
        "requires_harvested_contest_cpu_and_contest_cuda_cells_before_sideinfo_effect_claim",
        "score_claim_forbidden_until_effect_curve_artifact_passes",
    ]

    zero_cpu = next(
        cell
        for cell in preflight["cells"]
        if cell["variant"] == "zero" and cell["axis"] == "contest_cpu"
    )
    assert zero_cpu["axis_label"] == "[contest-CPU]"
    assert zero_cpu["lane_id"] == "lane_l5_v2_tt5l_sideinfo_effect_curve_zero_contest_cpu"
    assert "tools/claim_lane_dispatch.py claim" in zero_cpu["claim_command"]
    assert "--platform lightning" in zero_cpu["claim_command"]
    assert zero_cpu["job_name"] in zero_cpu["claim_command"]
    assert zero_cpu["pair_group_id"] in zero_cpu["claim_command"]
    assert zero_cpu["source_spec_command_is_authoritative"] is True
    assert zero_cpu["expected_result_json"].endswith("contest_auth_eval.json")
    assert "completed_lightning_exact_eval_harvested" in zero_cpu[
        "terminal_success_claim_template"
    ]
    assert "failed_lightning_exact_eval_no_score_claim" in zero_cpu[
        "terminal_failure_claim_template"
    ]
    assert "build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py" in (
        zero_cpu["harvest_probe_command_template"]
    )


def test_tt5l_lightning_execution_preflight_fails_closed_on_active_claim_conflict(
    tmp_path: Path,
) -> None:
    plan_path, plan = _write_plan(tmp_path)
    claims_text = """# Active lane dispatch claims

| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |
|---|---|---|---|---|---|---|---|
| 2026-05-17T12:00:00Z | sister | lane_l5_v2_tt5l_sideinfo_effect_curve_zero_contest_cpu | lightning | existing-job |  | active_dispatching | existing work |
"""

    preflight = build_l5_v2_tt5l_sideinfo_lightning_execution_preflight(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
        claims_text=claims_text,
        current_head_commit=str(plan["source_commit"]),
    )

    zero_cpu = next(
        cell
        for cell in preflight["cells"]
        if cell["variant"] == "zero" and cell["axis"] == "contest_cpu"
    )
    assert zero_cpu["ready_for_operator_claiming"] is False
    assert "active_lane_claim_conflict" in zero_cpu["blockers"]
    assert zero_cpu["active_claim_conflicts"][0]["instance_job_id"] == "existing-job"
    assert preflight["ready_for_operator_claiming"] is False
    assert "active_lane_claim_conflict" in preflight["blockers"]


def test_tt5l_lightning_execution_preflight_json_and_markdown_are_axis_labelled(
    tmp_path: Path,
) -> None:
    plan_path, plan = _write_plan(tmp_path)
    preflight = build_l5_v2_tt5l_sideinfo_lightning_execution_preflight(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
        current_head_commit=str(plan["source_commit"]),
    )

    decoded = json.loads(l5_v2_tt5l_sideinfo_lightning_execution_preflight_json(preflight))
    report = render_l5_v2_tt5l_sideinfo_lightning_execution_preflight_markdown(
        preflight
    )

    assert decoded["schema"] == L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_PREFLIGHT_SCHEMA
    assert "Lightning execution preflight" in report
    assert "[contest-CPU]" in report
    assert "[contest-CUDA]" in report
    assert "score_claim: `false`" in report
    assert "promotion_eligible: `false`" in report
    assert "claim_command" in report


def test_tt5l_lightning_execution_preflight_cli_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    plan_path, _plan = _write_plan(tmp_path)
    claims_path = tmp_path / "claims.md"
    output_json = tmp_path / "preflight.json"
    output_md = tmp_path / "preflight.md"
    claims_path.write_text("", encoding="utf-8")

    proc = subprocess.run(
        [
            str(root / "tools" / "build_l5_v2_tt5l_sideinfo_lightning_execution_preflight.py"),
            "--lightning-plan-json",
            str(plan_path),
            "--claims-path",
            str(claims_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--repo-root",
            str(tmp_path),
            "--current-head-commit",
            "a" * 40,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ready_for_operator_claiming"] is True
    assert payload["source_plan_commit_matches_current_head"] is True
    assert "score_claim=false" in proc.stdout
    assert output_md.is_file()


def test_tt5l_lightning_execution_preflight_blocks_stale_source_plan_commit(
    tmp_path: Path,
) -> None:
    plan_path, plan = _write_plan(tmp_path)

    preflight = build_l5_v2_tt5l_sideinfo_lightning_execution_preflight(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
        claims_text="",
        current_head_commit="b" * 40,
    )

    assert preflight["source_plan_commit_matches_current_head"] is False
    assert preflight["ready_cell_count"] == 0
    assert preflight["ready_for_operator_claiming"] is False
    assert "source_plan_commit_mismatch_current_head" in preflight["blockers"]
    assert all(
        "source_plan_commit_mismatch_current_head" in cell["blockers"]
        for cell in preflight["cells"]
    )
