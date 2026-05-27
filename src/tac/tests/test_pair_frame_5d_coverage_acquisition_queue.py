# SPDX-License-Identifier: MIT
"""Tests for queue-owned 5D canvas coverage acquisition."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from comma_lab.scheduler.pair_frame_5d_coverage_acquisition_queue import (
    PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA,
    PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA,
    PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA,
    PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA,
    build_coverage_acquisition_plan,
    build_pair_frame_5d_coverage_acquisition_queue,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_coverage import (
    COVERAGE_AUDIT_SCHEMA,
    WORK_ORDER_SCHEMA,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_BUILD_TOOL = _REPO_ROOT / "tools" / "build_5d_canvas_coverage_acquisition_queue.py"
_EMIT_TOOL = _REPO_ROOT / "tools" / "emit_5d_canvas_coverage_acquisition_plan.py"


def _work_order(order_id: str, *, priority: int = 1) -> dict[str, object]:
    return {
        "schema": WORK_ORDER_SCHEMA,
        "id": order_id,
        "priority": priority,
        "reason": f"unit {order_id}",
        "consumer": "unit_consumer",
        "target": {"archive_sha256": "e" * 64},
        "suggested_next_tools": [],
        "allowed_use": "experiment_queue_v1_planning_input_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
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


def _audit(*work_orders: dict[str, object]) -> dict[str, object]:
    return {
        "schema": COVERAGE_AUDIT_SCHEMA,
        "archive_sha256": "e" * 64,
        "verdict": "densification_required",
        "work_order_count": len(work_orders),
        "work_orders": list(work_orders),
        "blockers": ["unit_blocker"],
        "allowed_use": "local_planning_and_experiment_queue_acquisition_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
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


def _write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_build_coverage_acquisition_plan_for_fail_closed_anchor_order() -> None:
    audit = _audit(_work_order("populate_missing_paired_cpu_cuda_axis_anchors"))

    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )

    assert plan["schema"] == PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA
    assert plan["executable_now"] is False
    assert plan["score_claim"] is False
    assert "requires_byte_closed_submission_bundle_for_paired_auth_eval" in (
        plan["blocking_conditions"]
    )
    request = plan["followup_lane_requests"][0]
    assert request["schema"] == PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA
    assert request["canonical_paired_dispatch_tool"] == (
        "tools/dispatch_modal_paired_auth_eval.py"
    )
    assert request["paired_dispatch_command_contract_blockers"] == []
    assert "--submission-bundle" not in " ".join(
        request["paired_dispatch_command_template"]
    )
    assert request["required_input_schema_version"] == "submission_bundle_v1_20260526"


def test_build_coverage_acquisition_plan_for_mlx_negative_delta_request() -> None:
    audit = _audit(_work_order("acquire_negative_delta_cells_before_operator_fanout"))

    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
        output_root="experiments/results/unit_coverage_acquisition",
    )

    assert plan["executable_now"] is False
    assert plan["preferred_resource_kind"] == "local_mlx"
    request = plan["followup_lane_requests"][0]
    assert request["schema"] == PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA
    assert request["canonical_execution_tool"] == "tools/run_mlx_scorer_response_cache.py"
    assert request["preferred_execution_queue_schema"] == (
        "mlx_scorer_response_execution_queue_plan.v1"
    )
    assert "--allow-gpu-research-signal" in request["command_template"]
    assert "--archive-size-bytes" in request["command_template"]
    assert request["score_claim"] is False


def test_emit_coverage_acquisition_plan_cli(tmp_path: pathlib.Path) -> None:
    audit_path = _write_json(
        tmp_path / "audit.json",
        _audit(_work_order("densify_pair_coverage_for_grouped_search")),
    )
    canvas_path = _write_json(
        tmp_path / "canvas.json",
        {
            "schema": "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1",
            "archive_sha256": "e" * 64,
            "cells": [],
        },
    )
    out_path = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(_EMIT_TOOL),
            "--coverage-audit",
            str(audit_path),
            "--work-order-id",
            "densify_pair_coverage_for_grouped_search",
            "--canvas-path",
            str(canvas_path),
            "--output",
            str(out_path),
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA
    assert payload["promotable"] is False


def test_build_pair_frame_5d_coverage_acquisition_queue_shape(
    tmp_path: pathlib.Path,
) -> None:
    audit_path = _write_json(
        tmp_path / "audit.json",
        _audit(
            _work_order("populate_missing_paired_cpu_cuda_axis_anchors", priority=1),
            _work_order("densify_frame_coverage_for_masked_and_feathered_search", priority=2),
        ),
    )

    queue = build_pair_frame_5d_coverage_acquisition_queue(
        repo_root=_REPO_ROOT,
        coverage_audit_path=audit_path,
        canvas_path=tmp_path / "canvas.json",
        output_root=tmp_path / "requests",
        queue_id="unit_coverage_acquisition",
        local_cpu_concurrency=2,
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"]["local_cpu"] == 2
    assert queue["metadata"]["blocked_work_order_ids"] == [
        "populate_missing_paired_cpu_cuda_axis_anchors"
    ]
    assert queue["metadata"]["executable_work_order_ids"] == [
        "densify_frame_coverage_for_masked_and_feathered_search"
    ]
    assert len(queue["experiments"]) == 3
    first = queue["experiments"][0]
    assert first["metadata"]["schema"] == PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA
    assert first["metadata"]["score_claim"] is False
    assert first["metadata"]["executable_now"] is False
    assert first["metadata"]["blocking_conditions"] == [
        "requires_byte_closed_submission_bundle_for_paired_auth_eval",
        "requires_dispatch_claim_before_paid_or_remote_exact_axis_work",
    ]
    assert first["steps"][0]["command"][1] == "tools/emit_5d_canvas_coverage_acquisition_plan.py"
    refresh = queue["experiments"][-1]
    assert refresh["id"] == "refresh_reaudit_and_refire_extended_operators"
    assert refresh["metadata"]["external_blocking_work_order_ids"] == [
        "populate_missing_paired_cpu_cuda_axis_anchors"
    ]


def test_build_coverage_acquisition_queue_cli(tmp_path: pathlib.Path) -> None:
    audit_path = _write_json(
        tmp_path / "audit.json",
        _audit(_work_order("acquire_negative_delta_cells_before_operator_fanout")),
    )
    queue_path = tmp_path / "queue.json"

    subprocess.run(
        [
            sys.executable,
            str(_BUILD_TOOL),
            "--coverage-audit",
            str(audit_path),
            "--canvas-path",
            str(tmp_path / "canvas.json"),
            "--output-root",
            str(tmp_path / "requests"),
            "--queue-out",
            str(queue_path),
            "--queue-id",
            "unit_coverage_acquisition_cli",
            "--local-cpu-concurrency",
            "3",
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["queue_id"] == "unit_coverage_acquisition_cli"
    assert payload["controls"]["max_concurrency"]["local_cpu"] == 3
    assert len(payload["experiments"]) == 2
