# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_MEASUREMENT_SCHEDULE_SCHEMA,
    build_l5_v2_lattice_measurement_schedule,
)
from tac.optimization.l5_v2_paired_measurement_dispatch_plan import (
    L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_SCHEMA,
    build_l5_v2_paired_measurement_dispatch_plan,
    dispatch_plan_json,
    render_l5_v2_paired_measurement_dispatch_plan_markdown,
)

_FALSE_AUTHORITY_KEYS = {
    "planning_only": True,
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "dispatch_attempted": False,
    "adjudication_required": True,
}


def test_l5_v2_dispatch_plan_expands_active_measurements_to_paired_work_units() -> None:
    schedule = build_l5_v2_lattice_measurement_schedule()

    plan = build_l5_v2_paired_measurement_dispatch_plan(schedule=schedule)

    assert plan["schema"] == L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_SCHEMA
    for key, expected in _FALSE_AUTHORITY_KEYS.items():
        assert plan[key] is expected
    assert plan["active_measurement_ids"] == [
        "measure_c1_world_model_foveation_paired_exact",
        "measure_z5_predictive_coding_paired_exact",
        "measure_tt5l_autonomy_paired_exact",
    ]
    assert plan["required_axes"] == ["contest_cpu", "contest_cuda"]
    assert plan["work_unit_count"] == 3
    assert plan["ready_work_unit_count"] == 0
    assert plan["blockers"]
    assert {row["measurement_id"] for row in plan["work_units"]} == set(
        plan["active_measurement_ids"]
    )


def test_l5_v2_dispatch_plan_uses_only_canonical_paired_modal_dispatcher() -> None:
    schedule = build_l5_v2_lattice_measurement_schedule()

    plan = build_l5_v2_paired_measurement_dispatch_plan(schedule=schedule)

    for row in plan["work_units"]:
        command = row["dispatch_command_template"]
        assert "tools/dispatch_modal_paired_auth_eval.py" in command
        assert "experiments/modal_auth_eval.py" not in command
        assert "experiments/modal_auth_eval_cpu.py" not in command
        assert "--expected-runtime-tree-sha256 auto" in command
        assert "--skip-axis-if-promotable-anchor-exists" in command
        assert "--execute" not in row["dispatch_command"]
        assert row["dispatch_command_executable"] is False
        assert row["standalone_active_claim_command"] is None
        assert row["preclaim_forbidden"] is True
        assert "claim_lane_dispatch.py claim" not in command


def test_l5_v2_dispatch_plan_preserves_pairing_and_recovery_lifecycle() -> None:
    schedule = build_l5_v2_lattice_measurement_schedule()

    plan = build_l5_v2_paired_measurement_dispatch_plan(schedule=schedule)

    for row in plan["work_units"]:
        for key, expected in _FALSE_AUTHORITY_KEYS.items():
            assert row[key] is expected
        assert row["required_axes"] == ["contest_cpu", "contest_cuda"]
        assert row["pair_group_id"].startswith("pair_l5_v2_")
        assert row["lanes"]["contest_cpu"].endswith("_contest_cpu")
        assert row["lanes"]["contest_cuda"].endswith("_contest_cuda")
        assert row["pair_group_id"] in row["dispatch_command_template"]
        assert row["dispatch_command"][
            row["dispatch_command"].index("--pair-group-id") + 1
        ] == row["pair_group_id"]
        assert row["claim_lifecycle_owner"].startswith(
            "tools/dispatch_modal_paired_auth_eval.py"
        )
        assert set(row["harvest_commands"]) == {"contest_cpu", "contest_cuda"}
        assert all(
            command.startswith(".venv/bin/python tools/recover_modal_auth_eval.py --output-dir ")
            for command in row["harvest_commands"].values()
        )
        assert "requires_byte_closed_archive_path" in row["dispatch_blockers"]
        assert "requires_archive_sha256" in row["dispatch_blockers"]
        assert "requires_submission_dir_or_inflate_runtime" in row["dispatch_blockers"]


def test_l5_v2_dispatch_plan_top_level_blockers_include_row_and_dispatch_gaps() -> None:
    plan = build_l5_v2_paired_measurement_dispatch_plan(
        schedule=build_l5_v2_lattice_measurement_schedule()
    )

    assert "requires_byte_closed_archive_path" in plan["blockers"]
    assert "requires_archive_sha256" in plan["blockers"]
    assert "l5_v2_probe_observation_missing" in plan["blockers"]


def test_l5_v2_dispatch_plan_fails_closed_on_bad_schedule_schema() -> None:
    plan = build_l5_v2_paired_measurement_dispatch_plan(
        schedule={
            "schema": "wrong",
            "active_measurement_ids": ["missing_measurement"],
            "measurements": [],
        }
    )

    assert plan["work_unit_count"] == 0
    assert "l5_v2_measurement_schedule_schema_mismatch" in plan["blockers"]
    assert "l5_v2_active_measurement_missing:missing_measurement" in plan["blockers"]
    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False


def test_l5_v2_dispatch_plan_json_and_markdown_are_durable() -> None:
    plan = build_l5_v2_paired_measurement_dispatch_plan(
        schedule=build_l5_v2_lattice_measurement_schedule()
    )
    decoded = json.loads(dispatch_plan_json(plan))
    report = render_l5_v2_paired_measurement_dispatch_plan_markdown(plan)

    assert decoded["schema"] == L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_SCHEMA
    assert "L5 v2 paired measurement dispatch plan" in report
    assert "planning_only: `true`" in report
    assert "score_claim_valid: `false`" in report
    assert "paired_dispatch_tool" in report
    assert "tools/dispatch_modal_paired_auth_eval.py" in report


def test_l5_v2_dispatch_plan_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_dispatch_plan_{tmp_path.name}"
    )
    schedule_path = artifact_root / "schedule.json"
    output_json = artifact_root / "dispatch_plan.json"
    output_md = artifact_root / "dispatch_plan.md"
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        schedule = build_l5_v2_lattice_measurement_schedule()
        assert schedule["schema"] == L5V2_MEASUREMENT_SCHEDULE_SCHEMA
        schedule_path.write_text(json.dumps(schedule), encoding="utf-8")

        proc = subprocess.run(
            [
                "tools/build_l5_v2_paired_measurement_dispatch_plan.py",
                "--schedule-json",
                str(schedule_path.relative_to(root)),
                "--output-json",
                str(output_json.relative_to(root)),
                "--output-md",
                str(output_md.relative_to(root)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert output_json.is_file()
        assert output_md.is_file()
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["work_unit_count"] == 3
        assert payload["ready_work_unit_count"] == 0
        assert "score_claim=false" in proc.stdout
    finally:
        if artifact_root.exists():
            import shutil

            shutil.rmtree(artifact_root)
