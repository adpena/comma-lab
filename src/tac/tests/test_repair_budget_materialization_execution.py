# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.frontier_rate_attack_feedback import (
    AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_EXECUTION_ROW_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA,
    TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA,
    build_frontier_repair_budget_materialization_execution_report,
    build_frontier_repair_budget_waterfill_queue,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _assert_false_authority(payload: dict[str, object]) -> None:
    for key, value in FALSE_AUTHORITY.items():
        assert payload.get(key) is value, key


def _materialization_plan() -> dict[str, object]:
    parent_id = "repair_rate_floor_parent_abc123"
    child_id = "repair_budget_spent_child_def456"
    return {
        "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA,
        "chain_id": "global_many_op_rate_distortion_receiver_campaign",
        "parent_candidate_chain_id": parent_id,
        "candidate_chain_row_count": 2,
        "rate_only_floor_preserved_before_repair_spend": True,
        "spent_budget_candidates_are_children_of_rate_only_floor": True,
        "rate_only_candidate_remains_valid_even_if_child_regresses": True,
        "candidate_archive_materialized": False,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
        "candidate_chain_rows": [
            {
                "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
                "candidate_kind": "rate_only_floor_parent",
                "candidate_chain_id": parent_id,
                "chain_id": "global_many_op_rate_distortion_receiver_campaign",
                "materialization_order": 1,
                "parent_candidate_chain_id": None,
                "saved_bytes_total": 160,
                "candidate_archive_materialized": False,
                "candidate_archive_path": None,
                "runtime_consumption_proof_present": False,
                "receiver_consumed": False,
                "component_response_replayed": False,
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_materializer_execution": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": ["rate_only_candidate_archive_materialization_missing"],
                **FALSE_AUTHORITY,
            },
            {
                "schema": REPAIR_BUDGET_MATERIALIZATION_PLAN_ROW_SCHEMA,
                "candidate_kind": "spent_budget_repair_child",
                "candidate_chain_id": child_id,
                "chain_id": "global_many_op_rate_distortion_receiver_campaign",
                "materialization_order": 2,
                "parent_candidate_chain_id": parent_id,
                "parent_must_be_preserved_before_child": True,
                "child_must_not_replace_parent_archive": True,
                "allocation_rank": 1,
                "allocation_candidate_id": "pairset_drop_many_fixture",
                "requested_repair_bytes": 32,
                "proposed_encoder_repair_bytes": 32,
                "candidate_archive_materialized": False,
                "candidate_archive_path": None,
                "runtime_consumption_proof_present": False,
                "receiver_consumed": False,
                "component_response_replayed": False,
                "budget_spend_allowed": False,
                "ready_for_budget_spend": False,
                "ready_for_materializer_execution": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": ["parent_rate_only_archive_materialization_required"],
                **FALSE_AUTHORITY,
            },
        ],
    }


def test_repair_budget_materialization_execution_report_refuses_until_runtime_proof(
    tmp_path: Path,
) -> None:
    plan = _materialization_plan()

    report = build_frontier_repair_budget_materialization_execution_report(
        repair_budget_materialization_plan=plan,
        repair_budget_materialization_plan_path=tmp_path / "plan.json",
    )

    assert report["schema"] == REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
    _assert_false_authority(report)
    assert report["rate_only_floor_parent_first"] is True
    assert report["candidate_archive_materialized"] is False
    assert report["runtime_consumption_proof_present"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["exact_readiness_refusal"]["ready"] is False
    assert "candidate_archives_not_materialized" in report["blockers"]
    rows = report["execution_rows"]
    assert [row["candidate_kind"] for row in rows] == [
        "rate_only_floor_parent",
        "spent_budget_repair_child",
    ]
    assert all(
        row["schema"] == REPAIR_BUDGET_MATERIALIZATION_EXECUTION_ROW_SCHEMA
        for row in rows
    )
    assert rows[1]["parent_candidate_chain_id"] == report["parent_candidate_chain_id"]
    assert "parent_rate_only_archive_materialization_required" in rows[1]["blockers"]

    plan_path = _write_json(tmp_path / "repair_budget_materialization_plan.json", plan)
    report_path = tmp_path / "repair_budget_materialization_execution_report.json"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_repair_budget_materialization_execution_report.py",
            "--materialization-plan",
            str(plan_path),
            "--execution-report-out",
            str(report_path),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["candidate_archive_materialized"] is False
    materialized = json.loads(report_path.read_text(encoding="utf-8"))
    assert materialized["schema"] == REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
    assert materialized["exact_readiness_refusal"]["ready"] is False


def test_repair_budget_waterfill_queue_emits_execution_audit_step(
    tmp_path: Path,
) -> None:
    autonomous_path = _write_json(
        tmp_path / "autonomous_chain.json",
        {
            "schema": "frontier_rate_attack_autonomous_chain_optimization.v1",
            "rows": [
                {
                    "schema": AUTONOMOUS_CHAIN_OPTIMIZATION_ROW_SCHEMA,
                    "chain_id": "global_many_op_rate_distortion_receiver_campaign",
                    "chain_family": "rate_distortion_receiver_closed_many_op_campaign",
                    "rate_budget_preservation_plan": {},
                    **FALSE_AUTHORITY,
                }
            ],
            **FALSE_AUTHORITY,
        },
    )
    harvest_path = _write_json(
        tmp_path / "harvest.json",
        {
            "schema": TARGETED_COMPONENT_CORRECTION_RESPONSE_HARVEST_SCHEMA,
            "rows": [],
            **FALSE_AUTHORITY,
        },
    )
    budget_path = _write_json(
        tmp_path / "receiver_closed_budget.json",
        {
            "schema": "frontier_rate_attack_receiver_closed_correction_budget.v1",
            "receiver_closed_saved_bytes_total": 0,
            **FALSE_AUTHORITY,
        },
    )

    queue = build_frontier_repair_budget_waterfill_queue(
        repo_root=REPO_ROOT,
        autonomous_chain_optimization=json.loads(autonomous_path.read_text()),
        autonomous_chain_optimization_path=autonomous_path,
        targeted_component_correction_response_harvest=json.loads(
            harvest_path.read_text()
        ),
        targeted_component_correction_response_harvest_path=harvest_path,
        receiver_closed_correction_budget=json.loads(budget_path.read_text()),
        receiver_closed_correction_budget_path=budget_path,
        results_root=tmp_path / "results",
        queue_id="repair_waterfill_execution_audit_unit",
        chain_limit=1,
    )

    assert queue is not None
    experiment = queue["experiments"][0]
    step_ids = [step["id"] for step in experiment["steps"]]
    assert step_ids == [
        "emit_repair_budget_waterfill_work_order",
        "emit_repair_budget_materialization_plan",
        "audit_repair_budget_materialization_execution",
    ]
    audit_step = experiment["steps"][2]
    assert audit_step["requires"] == ["emit_repair_budget_materialization_plan"]
    assert audit_step["command"][1] == (
        "tools/build_frontier_repair_budget_materialization_execution_report.py"
    )
    assert any(
        condition.get("equals") == REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA
        for condition in audit_step["postconditions"]
    )
    assert (
        experiment["metadata"]["candidate_chain_execution_report_path"].endswith(
            "repair_budget_materialization_execution_report.json"
        )
    )
    assert experiment["metadata"]["candidate_archive_materialized"] is False
    _assert_false_authority(experiment["metadata"])
