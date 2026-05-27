# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA
from comma_lab.scheduler.repair_campaign_materialization_queue import (
    REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_EXPERIMENT_METADATA_SCHEMA,
    REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_GATE_SCHEMA,
    REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_QUEUE_METADATA_SCHEMA,
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
    build_repair_campaign_byte_closed_materialization_queue,
)
from tac.optimization.repair_campaign_scorer import score_repair_campaign

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _work_order(tmp_path: Path) -> dict[str, object]:
    mlx = tmp_path / "segnet_mlx_response.json"
    ref = tmp_path / "segnet_reference_mlx_response.json"
    mlx.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    ref.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
        "chain_id": "unit_repair_chain",
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 40,
            **_false_authority(),
        },
        "typed_response_ledger": {
            "schema": "frontier_rate_attack_repair_budget_typed_response_ledger.v1",
            "available_receiver_closed_rate_credit_bytes": 40,
            "rows": [
                {
                    "schema": (
                        "frontier_rate_attack_repair_budget_typed_response_row.v1"
                    ),
                    "typed_response_id": "segnet_region_ready",
                    "candidate_id": "segnet_class_region_waterfill",
                    "correction_family": "segnet_class_region_waterfill",
                    "targeted_dimensions": ["segnet", "region"],
                    "operation_levels": ["frame", "region"],
                    "entropy_position_label": (
                        "before_entropy_coder_distribution_shaping"
                    ),
                    "requested_repair_bytes": 32,
                    "objective_delta_score_units": -0.0010,
                    "local_mlx_response_path": str(mlx),
                    "reference_local_mlx_response_path": str(ref),
                    "segnet_class_region_mask_ids": ["road_boundary"],
                    "interaction_scope": {
                        "pair_indices": [7, 9],
                        "region_ids": ["road_boundary"],
                        **_false_authority(),
                    },
                    "stacking_interaction_terms": {
                        "must_remeasure_with_parent_and_sibling_repairs": True,
                        **_false_authority(),
                    },
                    **_false_authority(),
                }
            ],
            **_false_authority(),
        },
        **_false_authority(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_byte_closed_materialization_queue_emits_archive_bound_steps(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    work_order_path = _write_json(tmp_path / "work_order.json", work_order)
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)
    report_path = _write_json(tmp_path / "score_report.json", report)

    queue = build_repair_campaign_byte_closed_materialization_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=report_path,
        work_order_path=work_order_path,
        results_root=tmp_path / "results",
        queue_id="unit_repair_materialization",
    )

    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["metadata"]["schema"] == (
        REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_QUEUE_METADATA_SCHEMA
    )
    assert queue["metadata"]["ready_experiment_count"] == 1
    experiment = queue["experiments"][0]
    assert experiment["status"] == "queued"
    assert experiment["metadata"]["schema"] == (
        REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_EXPERIMENT_METADATA_SCHEMA
    )
    assert experiment["metadata"]["local_mlx_rows_are_advisory_only"] is True
    assert experiment["metadata"][
        "exact_eval_handoff_requires_complete_archive_runtime_component_custody"
    ] is True
    assert experiment["metadata"]["repair_materialization_lineage"][
        "target_queue_artifact_key"
    ] == "repair_campaign_byte_closed_materialization_queue"
    assert experiment["metadata"]["interaction_dynamics"]["active_scales"]
    assert [step["id"] for step in experiment["steps"]] == [
        "emit_repair_budget_materialization_plan",
        "emit_repair_family_materializer_manifest",
        "emit_repair_budget_child_component_replay_manifests",
        "bind_repair_budget_materializer_execution",
        "audit_repair_budget_materialization_execution",
        "emit_selected_repair_materialization_gate",
    ]
    assert experiment["steps"][0]["command"][1] == (
        "tools/build_frontier_repair_budget_materialization_plan.py"
    )
    assert str(work_order_path) in experiment["steps"][0]["command"]
    assert experiment["metadata"]["repair_family_materializer_manifest_schema"] == (
        REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
    )
    assert "--materializer-manifest" in experiment["steps"][3]["command"]
    assert experiment["steps"][-1]["postconditions"][0]["equals"] == (
        REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_GATE_SCHEMA
    )


def test_byte_closed_materialization_queue_cli_writes_queue(tmp_path: Path) -> None:
    work_order = _work_order(tmp_path)
    work_order_path = _write_json(tmp_path / "work_order.json", work_order)
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)
    report_path = _write_json(tmp_path / "score_report.json", report)
    queue_path = tmp_path / "repair_materialization_queue.json"

    result = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "tools"
                / "build_repair_campaign_byte_closed_materialization_queue.py"
            ),
            "--score-report",
            str(report_path),
            "--work-order",
            str(work_order_path),
            "--materialization-queue-out",
            str(queue_path),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "unit_repair_materialization",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["schema"] == QUEUE_SCHEMA
    assert payload["metadata"]["ready_experiment_count"] == 1
    validate_result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate_result.returncode == 0, validate_result.stderr
    worker_result_path = tmp_path / "worker_result.json"
    worker_result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "experiment_queue.py"),
            "--queue",
            str(queue_path),
            "run-worker",
            "--execute",
            "--max-steps",
            "7",
            "--max-experiments",
            "1",
            "--max-parallel",
            "1",
            "--output",
            str(worker_result_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert worker_result.returncode == 0, worker_result.stderr
    worker_payload = json.loads(worker_result_path.read_text(encoding="utf-8"))
    assert worker_payload["failure_count"] == 0
