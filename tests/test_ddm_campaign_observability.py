from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import costate_digest  # noqa: E402

from tac import witness_run_artifacts as wra  # noqa: E402
from tac.canonical_equations.ddm_pose_finish_engagement_watch_20260725 import (  # noqa: E402
    derive_pose_finish_engagement_watch,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _campaign(root: Path, name: str, *, step: int, mtime_ns: int) -> Path:
    run = root / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "telemetry").mkdir()
    (run / "verdicts").mkdir()
    (run / "run.log").write_text("governed fixture\n", encoding="utf-8")
    _write_json(
        run / "run_identity.json",
        {
            "schema": "ddm_joint_descent_run_identity.v1",
            "seed": 0,
        },
    )
    _write_json(
        run / "launch_manifest.json",
        {
            "schema": "detached_local_process_launch.v1",
            "argv": ["python3", "launcher.py"],
        },
    )
    pose_state = {
        "schema": "ddm_pose_finish_exact_verdict_plateau_state.v1",
        "classification": "INSUFFICIENT_EXACT_VERDICTS",
        "exact_verdict_steps": [0, 1, step],
        "strict_seg_admission_steps": [1],
        "strict_seg_admissions": 1,
        "engaged_global_step": None,
        "latest_relative_slope": None,
        "latest_relative_slope_stderr": None,
        "exact_d_seg": [0.07, 0.069, 0.068],
    }
    for index, seconds in enumerate((300.0, 304.0), 1):
        _write_json(
            run / "telemetry" / f"step{index:06d}.json",
            {
                "schema": "ddm_joint_descent_full_run_step.v1",
                "global_step": index,
                "seconds": seconds,
                "initial": {"d_seg": 0.1, "d_pose": 2.0},
                "final": {"d_seg": 0.09, "d_pose": 1.5},
            },
        )
        (run / "checkpoints" / f"stage_accepted_global{index:06d}.npz").write_bytes(
            b"fixture"
        )
    _write_json(
        run / "telemetry" / "geometry_step000002_local_exact_gradient_shrink00_track0001_cured.json",
        {"schema": "fixture.geometry.cure"},
    )
    _write_json(
        run / "verdicts" / f"stage_step{step:06d}_n600.json",
        {
            "schema": "ddm_joint_descent_chunked_stage_verdict.v1",
            "num_pairs": 600,
            "global_step": step,
            "d_seg": 0.068,
            "d_pose": 1.25,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "pose_finish_engage_state": pose_state,
        },
    )
    _write_json(
        run / "full_run_receipt.json",
        {
            "schema": "ddm_joint_descent_full_run_receipt.v1",
            "campaign_blocker": "BLOCKED_FIXTURE",
            "banked_r1_comparator": {"score_contribution": 0.127},
            "schedule": {
                "measured_seconds_per_step": 312.0,
                "pose_finish_engage": {
                    "ema_span": 3,
                    "hysteresis": 3,
                    "settle_window": 3,
                },
                "stages": [
                    {
                        "maximum_steps": 150,
                        "verdict_interval_steps": 50,
                    },
                    {"maximum_steps": 150, "verdict_interval_steps": 50},
                    {"maximum_steps": 150, "verdict_interval_steps": 50},
                ],
            },
        },
    )
    os.utime(run, ns=(mtime_ns, mtime_ns))
    return run


def test_governed_ddm_layout_is_a_name_agnostic_canonical_run(tmp_path: Path) -> None:
    run = _campaign(tmp_path, "arbitrary_name", step=50, mtime_ns=1_000_000_000)
    assert wra.is_run_dir(run)


def test_latest_campaign_discovery_uses_wra_and_directory_mtime(tmp_path: Path) -> None:
    older = _campaign(tmp_path, "not_a_ddm_prefix_a", step=25, mtime_ns=1_000_000_000)
    newer = _campaign(tmp_path, "not_a_ddm_prefix_b", step=50, mtime_ns=2_000_000_000)
    assert wra.is_run_dir(older)
    assert wra.is_run_dir(newer)
    assert costate_digest.discover_latest_ddm_campaign_run([tmp_path]) == newer


def test_campaign_observability_keeps_exact_and_batch_local_authority_separate(
    tmp_path: Path,
) -> None:
    run = _campaign(tmp_path, "fixture", step=50, mtime_ns=2_000_000_000)
    report = costate_digest.read_ddm_campaign_observability(run)
    rows = {row["row_id"]: row for row in report["rows"]}

    assert report["status"] == "STOPPED_WITH_TYPED_BLOCKER"
    assert rows["latest_exact_n600_verdict"]["d_seg"] == pytest.approx(0.068)
    assert rows["accepted_steps_and_cadence"]["accepted_step_count"] == 2
    assert rows["accepted_steps_and_cadence"]["measured_seconds_per_step"] == pytest.approx(
        302.0
    )
    assert rows["cumulative_batch_local_trace"]["epistemic_status"] == (
        "ADVISORY_BATCH_LOCAL"
    )
    assert rows["cumulative_batch_local_trace"]["not_n600_verdict"] is True
    assert rows["cumulative_batch_local_trace"][
        "delta_d_seg_sum_final_minus_initial"
    ] == pytest.approx(-0.02)
    assert rows["geometry_cure_events"]["event_count"] == 1
    assert rows["schedule_endpoint_eta"]["counterfactual_after_governed_stop"] is True
    assert rows["pose_finish_engagement_watch"]["watch"][
        "candidate_engagement_global_step"
    ] == 250
    assert rows["pose_finish_engagement_watch"]["watch"][
        "settled_engagement_global_step"
    ] == 300

    lines, surfaced = costate_digest.section_ddm_campaign_run([tmp_path])
    assert surfaced == report
    assert any("ADVISORY_BATCH_LOCAL; NOT n600" in line for line in lines)


def test_campaign_section_handles_pretelemetry_run_without_fabricating_cadence(
    tmp_path: Path,
) -> None:
    run = _campaign(tmp_path, "fixture", step=0, mtime_ns=2_000_000_000)
    for path in (run / "telemetry").glob("step*.json"):
        path.unlink()

    lines, report = costate_digest.section_ddm_campaign_run([tmp_path])

    cadence = next(
        row
        for row in report["rows"]
        if row["row_id"] == "accepted_steps_and_cadence"
    )
    assert cadence["measured_seconds_per_step"] is None
    assert any("pending" in line for line in lines)


def test_pose_finish_watch_is_conditional_and_preserves_banked_c1_budget() -> None:
    watch = derive_pose_finish_engagement_watch(
        verdict_interval_steps=50,
        ema_span=3,
        hysteresis=3,
        settle_window=3,
        observed_exact_verdicts=3,
        fallback_score_contribution=0.127,
    )
    assert watch["classification"] == "PRE_CONDITIONAL_ENGAGEMENT_WINDOW"
    assert watch["candidate_engagement_verdict_index_one_based"] == 5
    assert watch["settled_engagement_verdict_index_one_based"] == 6
    assert watch["candidate_engagement_global_step"] == 250
    assert watch["settled_engagement_global_step"] == 300
    assert watch["conditional_on_exact_n600_seg_plateau"] is True
    assert watch["fallback"]["score_contribution"] == pytest.approx(0.127)
