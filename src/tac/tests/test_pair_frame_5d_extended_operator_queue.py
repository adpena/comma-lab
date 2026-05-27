# SPDX-License-Identifier: MIT
"""Tests for the 5D extended-operator local queue builder."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from comma_lab.scheduler.frontier_rate_attack_feedback import (
    AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA,
    AUTONOMOUS_CHAIN_OPTIMIZATION_SCHEMA,
    build_frontier_autonomous_chain_optimization_queue,
)
from comma_lab.scheduler.frontier_rate_attack_feedback_cycle import (
    write_frontier_refresh_artifacts,
)
from comma_lab.scheduler.pair_frame_5d_coverage_acquisition_queue import (
    build_pair_frame_5d_coverage_acquisition_queue,
)
from comma_lab.scheduler.pair_frame_5d_extended_operator_queue import (
    PAIR_FRAME_5D_EXTENDED_OPERATOR_QUEUE_SCHEMA,
    build_pair_frame_5d_extended_operator_queue,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CpuCudaAxis,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_coverage import (
    COVERAGE_AUDIT_SCHEMA,
    WORK_ORDER_SCHEMA,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators import (
    ExtendedOperation,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TOOL = _REPO_ROOT / "tools" / "build_5d_extended_operator_queue.py"
_FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_packet_ready": False,
    "reproduction_claim": False,
    "reproduction_equivalence": False,
}


def _cell(pair_idx: int) -> PairFrameScorerGeometryCell:
    return PairFrameScorerGeometryCell(
        pair_idx=pair_idx,
        frame_idx=2 * pair_idx,
        scorer_axis=ScorerAxis.SEGNET_5CLASS,
        receiver_runtime=ReceiverRuntime.RAW_RESIDUAL,
        cpu_cuda_axis=CpuCudaAxis.CONTEST_CPU,
        predicted_delta_score=-0.01,
        predicted_byte_cost=0,
        receiver_feasibility=True,
    )


def _write_canvas(path: pathlib.Path) -> None:
    canvas = PairFrameScorerGeometryLattice(
        archive_sha256="c" * 64,
        cells={_cell(0).coordinate: _cell(0)},
    )
    path.write_text(
        json.dumps(
            {
                "schema": "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1",
                "archive_sha256": canvas.archive_sha256,
                "cells": [cell.as_dict() for cell in canvas._cells.values()],
            }
        ),
        encoding="utf-8",
    )


def _write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _coverage_work_order(order_id: str, *, priority: int = 1) -> dict[str, object]:
    return {
        "schema": WORK_ORDER_SCHEMA,
        "id": order_id,
        "priority": priority,
        "reason": f"unit {order_id}",
        "consumer": "unit_consumer",
        "target": {"archive_sha256": "c" * 64},
        "suggested_next_tools": [],
        "allowed_use": "experiment_queue_v1_planning_input_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **_FALSE_AUTHORITY,
    }


def _coverage_audit(*work_orders: dict[str, object]) -> dict[str, object]:
    return {
        "schema": COVERAGE_AUDIT_SCHEMA,
        "archive_sha256": "c" * 64,
        "verdict": "densification_required",
        "work_order_count": len(work_orders),
        "work_orders": list(work_orders),
        "blockers": [],
        "allowed_use": "local_planning_and_experiment_queue_acquisition_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        **_FALSE_AUTHORITY,
    }


def test_build_pair_frame_5d_extended_operator_queue_shape(
    tmp_path: pathlib.Path,
) -> None:
    canvas_path = tmp_path / "canvas.json"
    _write_canvas(canvas_path)

    queue = build_pair_frame_5d_extended_operator_queue(
        repo_root=_REPO_ROOT,
        canvas_path=canvas_path,
        output_root=tmp_path / "operator_outputs",
        queue_id="unit_5d_extended_operator_queue",
        top_n=4,
        local_cpu_concurrency=3,
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"]["local_cpu"] == 3
    assert len(queue["experiments"]) == 8
    operations = {
        experiment["metadata"]["operation"] for experiment in queue["experiments"]
    }
    assert operations == {operation.value for operation in ExtendedOperation}
    for experiment in queue["experiments"]:
        assert experiment["status"] == "queued"
        assert experiment["metadata"]["schema"] == (
            PAIR_FRAME_5D_EXTENDED_OPERATOR_QUEUE_SCHEMA
        )
        assert experiment["metadata"]["score_claim"] is False
        assert experiment["metadata"]["promotable"] is False
        command = experiment["steps"][0]["command"]
        assert "tools/apply_8_extended_operators_to_5d_canvas_cli.py" in command
        assert "--output" in command
        false_authority = experiment["steps"][0]["postconditions"][1]
        assert false_authority["type"] == "json_false_authority"
        assert false_authority["required_false"] == []


def test_build_5d_extended_operator_queue_cli(tmp_path: pathlib.Path) -> None:
    canvas_path = tmp_path / "canvas.json"
    queue_path = tmp_path / "queue.json"
    _write_canvas(canvas_path)

    subprocess.run(
        [
            sys.executable,
            str(_TOOL),
            "--canvas-path",
            str(canvas_path),
            "--output-root",
            str(tmp_path / "operator_outputs"),
            "--queue-out",
            str(queue_path),
            "--queue-id",
            "unit_5d_extended_operator_queue_cli",
            "--top-n",
            "4",
            "--local-cpu-concurrency",
            "2",
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["queue_id"] == "unit_5d_extended_operator_queue_cli"
    assert payload["controls"]["max_concurrency"]["local_cpu"] == 2
    assert len(payload["experiments"]) == 8


def test_refresh_artifacts_emit_pair_frame_5d_extended_operator_queue(
    tmp_path: pathlib.Path,
) -> None:
    canvas_path = tmp_path / "populated_5d_canvas.json"
    output_dir = tmp_path / "refresh"
    _write_canvas(canvas_path)

    artifacts = write_frontier_refresh_artifacts(
        output_dir=output_dir,
        repo_root=_REPO_ROOT,
        report={
            "queue_id": "unit_frontier_refresh",
            "candidate_limit": 4,
            "results_root": str(tmp_path / "results"),
            "pair_frame_5d_canvas_paths": [str(canvas_path)],
            **_FALSE_AUTHORITY,
        },
    )

    queue_ref = artifacts["pair_frame_5d_extended_operator_queue"]
    assert queue_ref.endswith("pair_frame_5d_extended_operator_queue.json")
    audit_ref = artifacts["pair_frame_5d_canvas_coverage_audit"]
    assert audit_ref.endswith("pair_frame_5d_canvas_coverage_audit.json")
    acquisition_queue_ref = artifacts["pair_frame_5d_coverage_acquisition_queue"]
    assert acquisition_queue_ref.endswith("pair_frame_5d_coverage_acquisition_queue.json")
    queue = json.loads((_REPO_ROOT / queue_ref).read_text(encoding="utf-8"))
    audit = json.loads((_REPO_ROOT / audit_ref).read_text(encoding="utf-8"))
    acquisition_queue = json.loads(
        (_REPO_ROOT / acquisition_queue_ref).read_text(encoding="utf-8")
    )
    assert len(queue["experiments"]) == 8
    assert audit["verdict"] == "densification_required"
    assert audit["score_claim"] is False
    assert len(acquisition_queue["experiments"]) == audit["work_order_count"] + 2
    assert acquisition_queue["experiments"][-2]["id"] == "audit_blocked_followup_requests"
    assert acquisition_queue["experiments"][-2]["steps"][-1]["id"] == (
        "run_followup_execution_queue_bounded_local"
    )
    report = json.loads(
        (output_dir / "feedback_refresh_report.json").read_text(encoding="utf-8")
    )
    assert report["artifacts"]["pair_frame_5d_canvas"].endswith(
        "populated_5d_canvas.json"
    )
    assert (
        report["pair_frame_5d_extended_operator_queue_summary"]["coverage_verdict"]
        == "densification_required"
    )
    assert (
        report["pair_frame_5d_extended_operator_queue_summary"][
            "coverage_acquisition_queue"
        ]
        == acquisition_queue_ref
    )
    followup_queue_ref = report["pair_frame_5d_coverage_acquisition_queue_summary"][
        "followup_execution_queue_path"
    ]
    assert followup_queue_ref.endswith(
        "pair_frame_5d_coverage_acquisition/followup_execution_queue.json"
    )
    assert report["pair_frame_5d_coverage_acquisition_queue_summary"][
        "followup_execution_queue_planned_by_queue"
    ] is True
    assert report["pair_frame_5d_coverage_acquisition_queue_summary"][
        "followup_execution_bounded_local_run_completed"
    ] is False
    assert (
        report["pair_frame_5d_coverage_acquisition_queue_summary"][
            "coverage_verdict"
        ]
        == "densification_required"
    )
    assert (
        report["operator_commands"]["validate_pair_frame_5d_extended_operator_queue"]
        == [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            queue_ref,
            "validate",
        ]
    )
    assert report["operator_commands"][
        "validate_pair_frame_5d_followup_execution_queue_after_acquisition"
    ] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        followup_queue_ref,
        "validate",
    ]
    assert report["operator_commands"][
        "run_pair_frame_5d_followup_execution_queue_bounded_local_after_acquisition"
    ][:5] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        followup_queue_ref,
        "run-worker",
    ]


def test_autonomous_parent_queue_binds_pair_frame_5d_child_when_present(
    tmp_path: pathlib.Path,
) -> None:
    canvas_path = tmp_path / "populated_5d_canvas.json"
    _write_canvas(canvas_path)
    child_queue = build_pair_frame_5d_extended_operator_queue(
        repo_root=_REPO_ROOT,
        canvas_path=canvas_path,
        output_root=tmp_path / "outputs",
        queue_id="unit_pair_frame_5d_child",
        top_n=4,
    )
    child_queue_path = _write_json(
        tmp_path / "pair_frame_5d_child_queue.json",
        child_queue,
    )
    coverage_audit_path = _write_json(
        tmp_path / "pair_frame_5d_coverage_audit.json",
        _coverage_audit(
            _coverage_work_order("densify_pair_coverage_for_grouped_search"),
        ),
    )
    coverage_queue = build_pair_frame_5d_coverage_acquisition_queue(
        repo_root=_REPO_ROOT,
        coverage_audit_path=coverage_audit_path,
        canvas_path=canvas_path,
        output_root=tmp_path / "coverage_outputs",
        queue_id="unit_pair_frame_5d_coverage_child",
        top_n=4,
    )
    coverage_queue_path = _write_json(
        tmp_path / "pair_frame_5d_coverage_child_queue.json",
        coverage_queue,
    )
    autonomous = {
        "schema": AUTONOMOUS_CHAIN_OPTIMIZATION_SCHEMA,
        "generated_at_utc": "2026-05-27T00:00:00Z",
        "chain_count": 1,
        "top_chain_ids": ["unit_chain"],
        "target_classes": [],
        "registered_target_count": 0,
        "unregistered_target_count": 0,
        "rate_only_candidate_count": 0,
        "rate_only_saved_bytes_total": 0,
        "rows": [
            {
                "schema": AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA,
                "chain_id": "unit_chain",
                "chain_family": "unit_pair_frame_5d",
                "optimization_objective": "unit",
                "scheduler_actions": [],
                "priority_score": 1.0,
                **_FALSE_AUTHORITY,
            }
        ],
        **_FALSE_AUTHORITY,
    }
    autonomous_path = _write_json(tmp_path / "autonomous_chain.json", autonomous)

    parent_queue = build_frontier_autonomous_chain_optimization_queue(
        repo_root=_REPO_ROOT,
        autonomous_chain_optimization=autonomous,
        autonomous_chain_optimization_path=autonomous_path,
        artifact_paths_by_key={
            "pair_frame_5d_coverage_acquisition_queue": coverage_queue_path,
            "pair_frame_5d_extended_operator_queue": child_queue_path,
        },
        results_root=tmp_path / "results",
        queue_id="unit_autonomous_parent_with_5d_child",
        chain_limit=1,
    )

    assert parent_queue is not None
    experiment = parent_queue["experiments"][0]
    assert experiment["status"] == "queued"
    metadata = experiment["metadata"]
    assert metadata["queue_actuation_ready"] is True
    assert metadata["missing_queue_artifact_keys"] == []
    assert metadata["local_queue_actions"][0]["queue_artifact_key"] == (
        "pair_frame_5d_coverage_acquisition_queue"
    )
    assert metadata["local_queue_actions"][0]["max_steps"] == 16
    assert metadata["local_queue_actions"][1]["queue_artifact_key"] == (
        "pair_frame_5d_extended_operator_queue"
    )
    assert any(
        step["id"].startswith("run_")
        and any("pair_frame_5d_child_queue.json" in arg for arg in step["command"])
        for step in experiment["steps"]
    )
    assert any(
        step["id"].startswith("run_")
        and any(
            "pair_frame_5d_coverage_child_queue.json" in arg
            for arg in step["command"]
        )
        and "16" in step["command"]
        for step in experiment["steps"]
    )
