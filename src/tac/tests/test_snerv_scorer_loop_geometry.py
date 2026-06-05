# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tac.analysis.nerv_rate_allocator_bridge import build_nerv_rate_allocator_bridge
from tac.analysis.nerv_rate_allocator_queue import build_nerv_rate_allocator_work_queue
from tac.analysis.snerv_scorer_loop_geometry import (
    BYTE_PRICE,
    SCHEMA,
    SnervScorerLoopGeometryError,
    build_snerv_scorer_loop_geometry_report,
    render_snerv_scorer_loop_geometry_markdown,
)
from tools import build_snerv_scorer_loop_geometry as cli


def test_snerv_scorer_loop_geometry_decomposes_contest_lagrangian(
    tmp_path: Path,
) -> None:
    path = _write_result(
        tmp_path / "snerv_result.json",
        baseline_score=1.2,
        best_score=1.1,
        baseline_seg=0.003,
        best_seg=0.00301,
        baseline_pose=0.001,
        best_pose=0.00081,
        baseline_bytes=1000,
        best_bytes=1004,
        search_mode="learned_random_subspace",
    )

    report = build_snerv_scorer_loop_geometry_report([path])

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["best_descent_score_delta_linf"] == pytest.approx(-0.1)
    assert report["lowest_local_score_linf"] == pytest.approx(1.1)
    row = report["reports"][0]
    best = row["best_contribution"]
    assert best["delta_seg_term"] == pytest.approx(0.001)
    expected_pose = math.sqrt(10.0 * 0.00081) - math.sqrt(10.0 * 0.001)
    assert best["delta_pose_term"] == pytest.approx(expected_pose)
    assert best["delta_rate_term"] == pytest.approx(4 * BYTE_PRICE)
    assert "pose_geometry_primary_current_descent" in row["geometry_verdicts"]
    assert "component_tradeoff_admitted_by_lagrangian" in row["geometry_verdicts"]
    assert report["aggregate"]["best_search_mode"] == "learned_random_subspace"
    assert report["aggregate"]["rate_is_current_descent_driver"] is False
    assert row["section_value_rows"][0]["delta_nonrate_score"] < 0.0
    assert row["byte_price_plan"]["decision_rows"][0]["economic_decision"] == "admit"
    assert row["byte_price_plan"]["decision_rows"][0]["decision"] == "demote"
    assert "advisory_or_proxy_axis_not_promotion_authority" in row[
        "byte_price_plan"
    ]["blockers"]
    assert report["allocator_units"][0]["unit_type"] == "snerv_scorer_loop_qat_result"
    assert report["allocator_units"][0]["section_value_rows"] == row[
        "section_value_rows"
    ]


def test_snerv_scorer_loop_geometry_units_feed_rate_allocator_bridge(
    tmp_path: Path,
) -> None:
    path = _write_result(tmp_path / "snerv_result.json")
    report = build_snerv_scorer_loop_geometry_report([path])

    bridge = build_nerv_rate_allocator_bridge(
        master_bridge={
            "schema": "nerv_master_consumer_bridge.v1",
            "baseline_to_beat": "pr95",
            "top_priority_carriers": ["snerv", "hi_nerv"],
            "master_consumer_units": [report["allocator_units"][0]],
            "blockers": [],
        }
    )

    assert bridge["schema"] == "nerv_rate_allocator_bridge.v1"
    assert bridge["rate_allocator_work_orders"][0]["work_order_type"] == (
        "snerv_scorer_loop_qat_full600_followup"
    )
    assert bridge["rate_allocator_work_orders"][0]["payload"]["selection_policy"] == (
        "score_primary_lagrangian_geometry"
    )
    assert bridge["rate_allocator_work_orders"][0]["payload"]["section_value_rows"]
    assert bridge["rate_allocator_work_orders"][0]["payload"]["byte_price_plan"][
        "schema"
    ] == "compact_nerv_byte_price_controller.v1"

    queue = build_nerv_rate_allocator_work_queue(rate_bridge=bridge)
    assert queue["section_admission_queue_row_count"] == len(
        report["reports"][0]["section_value_rows"]
    )
    qat_row = queue["queue_rows"][0]
    assert qat_row["planner_ingest"]["source_section_value_rows"]
    assert qat_row["planner_ingest"]["source_byte_price_plan_schema"] == (
        "compact_nerv_byte_price_controller.v1"
    )
    assert qat_row["planner_ingest"]["source_byte_price_decision_rows"]


def test_snerv_scorer_loop_geometry_surfaces_rejected_score_descent(
    tmp_path: Path,
) -> None:
    path = _write_result(tmp_path / "snerv_result.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["result"]
    baseline = dict(result["baseline"])
    rejected = dict(result["evaluations"][1])
    rejected.update(
        {
            "accepted": False,
            "score_linf": baseline["score_linf"] - 0.25,
            "archive_bytes": baseline["archive_bytes"] + 4,
            "rate_term": (baseline["archive_bytes"] + 4) * BYTE_PRICE,
            "blockers": ["byte_growth_guard_failed", "pair_pose_worsening_fraction_guard_failed"],
        }
    )
    result["best"] = baseline
    result["evaluations"][1] = rejected
    result["accepted_improvement"] = False
    result["ready_for_pose_guard_gate"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_snerv_scorer_loop_geometry_report([path])

    row = report["reports"][0]
    assert report["best_descent_score_delta_linf"] == pytest.approx(0.0)
    assert row["rejected_score_descent_count"] == 1
    assert row["best_rejected_score_descent"]["label"] == rejected["label"]
    assert row["best_rejected_score_descent"]["score_delta_linf"] == pytest.approx(
        -0.25
    )
    assert "byte_growth_guard_failed" in row["best_rejected_score_descent"]["blockers"]
    assert "snerv_rejected_scorer_descent_admission_repair_required" in report[
        "blockers"
    ]
    assert (
        report["recommended_next_actions"][0]["id"]
        == "repair_rejected_scorer_descent_admission"
    )
    assert "scale_score_primary_random_subspace_batch" not in {
        action["id"] for action in report["recommended_next_actions"]
    }
    assert report["allocator_units"][0]["best_rejected_score_descent"]["label"] == (
        rejected["label"]
    )
    bridge = build_nerv_rate_allocator_bridge(
        master_bridge={
            "schema": "nerv_master_consumer_bridge.v1",
            "baseline_to_beat": "pr95",
            "top_priority_carriers": ["snerv", "hi_nerv"],
            "master_consumer_units": [report["allocator_units"][0]],
            "blockers": [],
        }
    )
    repair_order = bridge["rate_allocator_work_orders"][0]
    assert repair_order["work_order_type"] == "snerv_scorer_loop_qat_training_repair"
    assert repair_order["payload"]["best_rejected_score_descent"]["label"] == (
        rejected["label"]
    )


def test_snerv_scorer_loop_geometry_ignores_probe_only_rejected_descent_for_repair(
    tmp_path: Path,
) -> None:
    path = _write_result(tmp_path / "snerv_result.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["result"]
    baseline = result["baseline"]
    probe = dict(result["evaluations"][2])
    probe.update(
        {
            "label": "nes_probe_001_minus_probe",
            "accepted": False,
            "score_linf": baseline["score_linf"] - 0.25,
            "archive_bytes": baseline["archive_bytes"] + 4,
            "rate_term": (baseline["archive_bytes"] + 4) * BYTE_PRICE,
            "blockers": ["nes_probe_only_not_candidate"],
        }
    )
    result["evaluations"][2] = probe
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_snerv_scorer_loop_geometry_report([path])

    assert report["reports"][0]["accepted_trial_count"] == 1
    assert report["reports"][0]["rejected_score_descent_count"] == 0
    assert report["reports"][0]["best_rejected_score_descent"] is None
    assert "snerv_rejected_scorer_descent_admission_repair_required" not in report[
        "blockers"
    ]
    action_ids = {action["id"] for action in report["recommended_next_actions"]}
    assert "repair_rejected_scorer_descent_admission" not in action_ids
    assert "scale_score_primary_random_subspace_batch" in action_ids


def test_snerv_scorer_loop_geometry_preserves_materialized_packet_for_full600(
    tmp_path: Path,
) -> None:
    path = _write_result(
        tmp_path / "snerv_result.json",
        best_packet_materialized=True,
    )

    report = build_snerv_scorer_loop_geometry_report([path])

    row = report["reports"][0]
    assert row["accepted_improvement"] is True
    assert row["best_packet_materialized"] is True
    assert row["best_packet_path"] == "/ssd/snerv/best_packet.snar"
    assert row["best_packet_bytes"] == 117736
    assert row["best_packet_sha256"] == "f" * 64
    bind_action = {
        action["id"]: action for action in report["recommended_next_actions"]
    }["bind_archive_codec_to_descent_step"]
    assert "best_decoder_packet_materialization_missing" not in bind_action[
        "blockers"
    ]

    bridge = build_nerv_rate_allocator_bridge(
        master_bridge={
            "schema": "nerv_master_consumer_bridge.v1",
            "baseline_to_beat": "pr95",
            "top_priority_carriers": ["snerv", "hi_nerv"],
            "master_consumer_units": [report["allocator_units"][0]],
            "blockers": [],
        }
    )
    scale_order = bridge["rate_allocator_work_orders"][0]
    assert scale_order["work_order_type"] == "snerv_scorer_loop_qat_full600_followup"
    assert scale_order["payload"]["best_packet_materialized"] is True
    assert scale_order["payload"]["best_packet_path"] == "/ssd/snerv/best_packet.snar"
    assert scale_order["payload"]["best_packet_sha256"] == "f" * 64
    assert "section_value_profile_missing" not in scale_order["blockers"]


def test_snerv_scorer_loop_geometry_cli_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    result = _write_result(tmp_path / "snerv_result.json")
    out_json = tmp_path / "geometry.json"
    out_md = tmp_path / "geometry.md"

    rc = cli.main(
        [
            "--result-json",
            str(result),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--label",
            "unit_test_geometry",
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["label"] == "unit_test_geometry"
    assert payload["promotion_eligible"] is False
    assert out_md.read_text(encoding="utf-8").startswith("# SNeRV scorer-loop geometry")


def test_snerv_scorer_loop_geometry_does_not_compare_absolute_scores_across_pair_counts(
    tmp_path: Path,
) -> None:
    low_absolute = _write_result(
        tmp_path / "one_pair.json",
        baseline_score=0.55,
        best_score=0.54,
        search_mode="learned_random_subspace",
        n_pairs=1,
    )
    strong_descent = _write_result(
        tmp_path / "four_pair.json",
        baseline_score=1.20,
        best_score=1.10,
        search_mode="learned_random_subspace",
        n_pairs=4,
    )

    report = build_snerv_scorer_loop_geometry_report([low_absolute, strong_descent])

    assert report["lowest_local_score_input_path"] == low_absolute.as_posix()
    assert report["best_descent_input_path"] == strong_descent.as_posix()
    assert report["best_descent_score_delta_linf"] == pytest.approx(-0.1)


def test_snerv_scorer_loop_geometry_requires_inputs() -> None:
    with pytest.raises(SnervScorerLoopGeometryError, match="at least one"):
        build_snerv_scorer_loop_geometry_report([])


def test_snerv_scorer_loop_geometry_render_includes_blockers(tmp_path: Path) -> None:
    path = _write_result(tmp_path / "snerv_result.json")
    report = build_snerv_scorer_loop_geometry_report([path])

    markdown = render_snerv_scorer_loop_geometry_markdown(report)

    assert "full600_receiver_proof_required" in markdown
    assert "paired_contest_cpu_cuda_auth_eval_missing" in markdown


def _write_result(
    path: Path,
    *,
    baseline_score: float = 1.0,
    best_score: float = 0.9,
    baseline_seg: float = 0.002,
    best_seg: float = 0.0019,
    baseline_pose: float = 0.001,
    best_pose: float = 0.0008,
    baseline_bytes: int = 1000,
    best_bytes: int = 998,
    search_mode: str = "learned_random_subspace",
    n_pairs: int = 4,
    best_packet_materialized: bool = False,
) -> Path:
    payload = {
        "schema": "snerv_scorer_loop_qat_local_trainer.v1",
        "axis_tag": "[macOS-CPU advisory]",
        "result": {
            "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
            "axis_tag": "[macOS-CPU advisory]",
            "n_pairs": n_pairs,
            "levels": 2,
            "wavelet": "db2",
            "qat_bits": 8,
            "search_mode": search_mode,
            "component_guard_mode": "score_primary",
            "scorer_loop_evaluations": 3,
            "baseline": {
                "label": "least_squares_qat_baseline",
                "accepted": True,
                "archive_bytes": baseline_bytes,
                "score_linf": baseline_score,
                "d_seg_linf": baseline_seg,
                "d_pose_linf": baseline_pose,
                "rate_term": baseline_bytes * BYTE_PRICE,
            },
            "best": {
                "label": "learned_subspace_001_plus",
                "accepted": True,
                "archive_bytes": best_bytes,
                "score_linf": best_score,
                "d_seg_linf": best_seg,
                "d_pose_linf": best_pose,
                "rate_term": best_bytes * BYTE_PRICE,
            },
            "evaluations": [
                {
                    "label": "least_squares_qat_baseline",
                    "accepted": True,
                    "archive_bytes": baseline_bytes,
                    "score_linf": baseline_score,
                    "d_seg_linf": baseline_seg,
                    "d_pose_linf": baseline_pose,
                    "rate_term": baseline_bytes * BYTE_PRICE,
                    "blockers": [],
                },
                {
                    "label": "learned_subspace_001_plus",
                    "accepted": True,
                    "archive_bytes": best_bytes,
                    "score_linf": best_score,
                    "d_seg_linf": best_seg,
                    "d_pose_linf": best_pose,
                    "rate_term": best_bytes * BYTE_PRICE,
                    "blockers": [],
                },
                {
                    "label": "learned_subspace_001_minus",
                    "accepted": False,
                    "archive_bytes": baseline_bytes,
                    "score_linf": baseline_score,
                    "d_seg_linf": baseline_seg,
                    "d_pose_linf": baseline_pose,
                    "rate_term": baseline_bytes * BYTE_PRICE,
                    "blockers": ["score_gate_failed"],
                },
            ],
            "best_pair_deltas": [],
            "accepted_improvement": True,
            "ready_for_pose_guard_gate": True,
            "receiver_contract_satisfied": True,
            "blockers": ["local_smoke_only_not_full_600_pairs"],
        },
    }
    if best_packet_materialized:
        payload.update(
            {
                "best_packet_materialized": True,
                "best_packet_path": "/ssd/snerv/best_packet.snar",
                "best_packet_bytes": 117736,
                "best_packet_sha256": "f" * 64,
                "best_packet_materialization": {
                    "schema": "snerv_scorer_loop_best_packet_materialization.v1",
                    "materialized": True,
                    "best_packet_path": "/ssd/snerv/best_packet.snar",
                    "best_packet_bytes": 117736,
                    "best_packet_sha256": "f" * 64,
                    "blockers": [],
                },
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
