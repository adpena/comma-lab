# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_MEASUREMENT_SCHEDULE_SCHEMA,
    build_l5_v2_lattice_measurement_schedule,
    render_l5_v2_lattice_measurement_schedule_markdown,
    schedule_json,
)


def _eligible_row(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "eligible_for_architecture_lock": True,
        "blockers": [],
    }


def test_l5_v2_schedule_fails_closed_to_probe_filling_without_intake() -> None:
    schedule = build_l5_v2_lattice_measurement_schedule()

    assert schedule["schema"] == L5V2_MEASUREMENT_SCHEDULE_SCHEMA
    assert schedule["first_match_wins"] is True
    assert schedule["active_rule_id"] == "fill_missing_c1_z5_tt5l_probe_observations"
    assert schedule["score_claim"] is False
    assert schedule["promotion_eligible"] is False
    assert schedule["ready_for_exact_eval_dispatch"] is False
    assert schedule["rank_reward_allowed"] is False
    assert set(schedule["active_measurement_ids"]) == {
        "measure_c1_world_model_foveation_paired_exact",
        "measure_z5_predictive_coding_paired_exact",
        "measure_tt5l_autonomy_paired_exact",
    }


def test_l5_v2_schedule_routes_to_sideinfo_curve_after_probe_eligibility() -> None:
    intake = {
        "verdict": {
            "evaluated_observations": [
                _eligible_row("c1_world_model_foveation"),
                _eligible_row("z5_predictive_coding_world_model"),
                _eligible_row("time_traveler_l5_autonomy"),
            ]
        }
    }

    schedule = build_l5_v2_lattice_measurement_schedule(probe_intake=intake)

    assert schedule["eligible_candidates"] == [
        "c1_world_model_foveation",
        "time_traveler_l5_autonomy",
        "z5_predictive_coding_world_model",
    ]
    assert schedule["active_rule_id"] == "measure_tt5l_sideinfo_effect_curve"
    assert schedule["active_measurement_ids"] == ["measure_tt5l_sideinfo_effect_curve"]
    sideinfo = next(
        row
        for row in schedule["measurements"]
        if row["measurement_id"] == "measure_tt5l_sideinfo_effect_curve"
    )
    assert sideinfo["score_claim"] is False
    assert sideinfo["required_axes"] == ["contest_cpu", "contest_cuda"]
    assert "consumption_proof_is_not_yet_usefulness_proof" in sideinfo["blockers"]
    assert (
        "requires_paired_cpu_cuda_sideinfo_effect_curve_before_architecture_lock"
        in sideinfo["blockers"]
    )


def test_l5_v2_schedule_json_and_markdown_are_durable() -> None:
    schedule = build_l5_v2_lattice_measurement_schedule()
    decoded = json.loads(schedule_json(schedule))
    report = render_l5_v2_lattice_measurement_schedule_markdown(schedule)

    assert decoded["schema"] == L5V2_MEASUREMENT_SCHEDULE_SCHEMA
    assert "L5 v2 lattice measurement schedule" in report
    assert "required_axes" in report
    assert "score_claim: `false`" in report


def test_l5_v2_schedule_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_measurement_schedule_{tmp_path.name}"
    )
    output_json = artifact_root / "schedule.json"
    output_md = artifact_root / "schedule.md"
    try:
        proc = subprocess.run(
            [
                "tools/build_l5_v2_lattice_measurement_schedule.py",
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
        assert payload["active_rule_id"] == "fill_missing_c1_z5_tt5l_probe_observations"
        assert "score_claim=false" in proc.stdout
    finally:
        if artifact_root.exists():
            import shutil

            shutil.rmtree(artifact_root)
