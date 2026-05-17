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
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA,
    T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV,
    build_l5_v2_tt5l_sideinfo_lightning_execution_bundle,
    l5_v2_tt5l_sideinfo_lightning_execution_bundle_json,
    render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_markdown,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_preflight import (
    build_l5_v2_tt5l_sideinfo_lightning_execution_preflight,
)

from .test_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import _write_manifest


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object], dict[str, object], dict[str, object]]:
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
    preflight = build_l5_v2_tt5l_sideinfo_lightning_execution_preflight(
        plan=plan,
        plan_path=plan_path,
        repo_root=tmp_path,
        claims_text="",
        current_head_commit="b" * 40,
    )
    preflight_path = tmp_path / ".omx/research/preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dykstra_path = tmp_path / ".omx/state/dykstra_feasibility_time_traveler_l5.json"
    dykstra_path.parent.mkdir(parents=True, exist_ok=True)
    dykstra_path.write_text(
        json.dumps(
            {
                "schema": "dykstra_feasibility_verdict_v1",
                "verdict": "FEASIBLE",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, plan_path, preflight_path, manifest, plan, preflight


def test_tt5l_lightning_execution_bundle_builds_dry_run_commands(
    tmp_path: Path,
) -> None:
    manifest_path, plan_path, preflight_path, manifest, plan, preflight = _write_inputs(
        tmp_path
    )

    bundle = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle(
        preflight=preflight,
        preflight_path=preflight_path,
        lightning_plan=plan,
        lightning_plan_path=plan_path,
        variant_manifest=manifest,
        variant_manifest_path=manifest_path,
        repo_root=tmp_path,
        current_head_commit="c" * 40,
    )

    expected_count = len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS) * len(
        L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
    )
    assert bundle["schema"] == L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA
    assert bundle["planning_only"] is True
    assert bundle["score_claim"] is False
    assert bundle["score_claim_valid"] is False
    assert bundle["promotion_eligible"] is False
    assert bundle["ready_for_exact_eval_dispatch"] is False
    assert bundle["ready_for_provider_dispatch"] is False
    assert bundle["dispatch_attempted"] is False
    assert bundle["cell_count"] == expected_count
    assert bundle["ready_dry_run_cell_count"] == expected_count
    assert bundle["ready_for_dry_run_submit"] is True
    assert bundle["ready_for_non_dry_run_submit"] is False
    assert bundle["dykstra_feasibility_status"]["artifact_valid"] is True

    zero_cpu = next(
        cell
        for cell in bundle["cells"]
        if cell["variant"] == "zero" and cell["axis"] == "contest_cpu"
    )
    assert zero_cpu["axis_label"] == "[contest-CPU]"
    assert zero_cpu["eval_device"] == "cpu"
    assert "scripts/lightning_repro_workspace.py" in zero_cpu[
        "stage_source_manifest_command_template"
    ]
    dry_run = zero_cpu["dry_run_submit_command"]
    assert "scripts/launch_lightning_batch_job.py exact-eval" in dry_run
    assert "--dry-run" in dry_run
    assert "--adjudicate" in dry_run
    assert "--eval-device cpu" in dry_run
    assert f"--dispatch-lane-id {zero_cpu['lane_id']}" in dry_run
    assert "--source-manifest experiments/results/lightning_batch/" in dry_run
    assert zero_cpu["source_spec_command_sha256"] in dry_run
    for runtime_env in T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV:
        assert f"--env {runtime_env}" in dry_run
    non_dry = zero_cpu["non_dry_run_submit_command_template"]
    assert "--dry-run" not in non_dry
    assert "--studio '<lightning-studio>'" in non_dry
    assert "--remote-preflight-ssh-target '<lightning-ssh-target>'" in non_dry
    for runtime_env in T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV:
        assert f"--env {runtime_env}" in non_dry
    assert zero_cpu["ready_for_provider_dispatch"] is False
    assert zero_cpu["ready_for_non_dry_run_submit"] is False
    assert "claim_command_must_be_run_first" in zero_cpu[
        "non_dry_run_submit_blockers"
    ]


def test_tt5l_lightning_execution_bundle_records_missing_dykstra_without_blocking_dry_run(
    tmp_path: Path,
) -> None:
    manifest_path, plan_path, preflight_path, manifest, plan, preflight = _write_inputs(
        tmp_path
    )
    (tmp_path / ".omx/state/dykstra_feasibility_time_traveler_l5.json").unlink()

    bundle = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle(
        preflight=preflight,
        preflight_path=preflight_path,
        lightning_plan=plan,
        lightning_plan_path=plan_path,
        variant_manifest=manifest,
        variant_manifest_path=manifest_path,
        repo_root=tmp_path,
    )

    assert bundle["ready_for_dry_run_submit"] is True
    assert bundle["ready_for_provider_dispatch"] is False
    assert bundle["dykstra_feasibility_status"]["artifact_exists"] is False
    assert "dykstra:dykstra_feasibility_artifact_missing" in bundle["global_blockers"]


def test_tt5l_lightning_execution_bundle_fails_closed_on_command_sha_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path, plan_path, preflight_path, manifest, plan, preflight = _write_inputs(
        tmp_path
    )
    preflight["cells"][0]["source_spec_command_sha256"] = "0" * 64

    bundle = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle(
        preflight=preflight,
        preflight_path=preflight_path,
        lightning_plan=plan,
        lightning_plan_path=plan_path,
        variant_manifest=manifest,
        variant_manifest_path=manifest_path,
        repo_root=tmp_path,
    )

    assert bundle["ready_for_dry_run_submit"] is False
    assert "source_spec_command_sha256_mismatch" in bundle["blockers"]
    assert bundle["cells"][0]["ready_for_dry_run_submit"] is False


def test_tt5l_lightning_execution_bundle_json_and_markdown_are_axis_labelled(
    tmp_path: Path,
) -> None:
    manifest_path, plan_path, preflight_path, manifest, plan, preflight = _write_inputs(
        tmp_path
    )
    bundle = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle(
        preflight=preflight,
        preflight_path=preflight_path,
        lightning_plan=plan,
        lightning_plan_path=plan_path,
        variant_manifest=manifest,
        variant_manifest_path=manifest_path,
        repo_root=tmp_path,
    )

    decoded = json.loads(l5_v2_tt5l_sideinfo_lightning_execution_bundle_json(bundle))
    report = render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_markdown(bundle)

    assert decoded["schema"] == L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA
    assert "Lightning execution bundle" in report
    assert "[contest-CPU]" in report
    assert "[contest-CUDA]" in report
    assert "score_claim: `false`" in report
    assert "promotion_eligible: `false`" in report
    assert "dry_run_submit_command" in report
    assert "non_dry_run_submit_command_template" in report


def test_tt5l_lightning_execution_bundle_cli_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    manifest_path, plan_path, preflight_path, _manifest, _plan, _preflight = _write_inputs(
        tmp_path
    )
    output_json = tmp_path / "bundle.json"
    output_md = tmp_path / "bundle.md"

    proc = subprocess.run(
        [
            str(root / "tools" / "build_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py"),
            "--preflight-json",
            str(preflight_path),
            "--lightning-plan-json",
            str(plan_path),
            "--variant-manifest-json",
            str(manifest_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--repo-root",
            str(tmp_path),
            "--current-head-commit",
            "d" * 40,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["ready_for_dry_run_submit"] is True
    assert "score_claim=false" in proc.stdout
    assert output_md.is_file()
