# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA
from comma_lab.scheduler.repair_campaign_score_queue import (
    REPAIR_CAMPAIGN_SCORE_EXPERIMENT_METADATA_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_QUEUE_METADATA_SCHEMA,
    build_repair_campaign_score_queue,
)
from tac.optimization.repair_campaign_scorer import REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA

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
                    "acquisition_id": "segnet_region_acq",
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
                    **_false_authority(),
                },
            ],
            **_false_authority(),
        },
        **_false_authority(),
    }


def _waterfill_queue(work_order_path: Path) -> dict[str, object]:
    return {
        "schema": QUEUE_SCHEMA,
        "queue_id": "repair_budget_waterfill_queue",
        "controls": {
            "mode": "running",
            "local_first": True,
            "max_concurrency": {"local_io_heavy": 1},
        },
        "metadata": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "experiments": [
            {
                "id": "waterfill_chain_a",
                "priority": 1,
                "status": "queued",
                "metadata": {
                    "chain_id": "cascade_c_chain",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "steps": [
                    {
                        "id": "emit_repair_budget_waterfill_work_order",
                        "kind": "command",
                        "command": [
                            sys.executable,
                            "tools/build_frontier_repair_budget_waterfill_work_order.py",
                            "--work-order-out",
                            str(work_order_path),
                        ],
                        "resources": {"kind": "local_io_heavy"},
                        "postconditions": [],
                    }
                ],
            }
        ],
    }


def test_build_repair_campaign_score_queue_from_waterfill_work_order(
    tmp_path: Path,
) -> None:
    work_order_path = tmp_path / "waterfill_work_order.json"
    work_order_path.write_text(
        json.dumps(_work_order(tmp_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    queue = build_repair_campaign_score_queue(
        repo_root=REPO_ROOT,
        repair_budget_waterfill_queue=_waterfill_queue(work_order_path),
        repair_budget_waterfill_queue_path=tmp_path / "repair_budget_waterfill_queue.json",
        results_root=tmp_path / "results",
        queue_id="unit_repair_campaign_score_queue",
    )

    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["metadata"]["schema"] == REPAIR_CAMPAIGN_SCORE_QUEUE_METADATA_SCHEMA
    assert queue["metadata"]["ready_experiment_count"] == 1
    assert queue["metadata"]["blocked_experiment_count"] == 0
    experiment = queue["experiments"][0]
    assert experiment["status"] == "queued"
    assert experiment["metadata"]["schema"] == (
        REPAIR_CAMPAIGN_SCORE_EXPERIMENT_METADATA_SCHEMA
    )
    assert experiment["metadata"]["queue_actuation_ready"] is True
    assert experiment["steps"][0]["id"] == (
        "assert_repair_budget_waterfill_work_order_materialized"
    )
    command = experiment["steps"][1]["command"]
    assert command[:2] == [".venv/bin/python", "tools/score_repair_campaign.py"]
    assert str(work_order_path) in command
    stackability_command = experiment["steps"][2]["command"]
    assert stackability_command[:2] == [
        ".venv/bin/python",
        "tools/build_repair_campaign_stackability_queue.py",
    ]
    assert experiment["steps"][2]["requires"] == [
        "score_repair_campaign_from_typed_ledger"
    ]
    assert experiment["metadata"]["repair_campaign_stackability_queue_path"] in (
        stackability_command
    )
    assert experiment["metadata"][
        "repair_campaign_stackability_worker_result_path"
    ].endswith("repair_campaign_stackability_worker_result.json")
    assert experiment["metadata"]["continual_learning_followup_default"] is True
    assert experiment["metadata"]["stackability_worker_limits"] == {
        "schema": "repair_campaign_stackability_worker_limits.v1",
        "max_steps": 8,
        "max_experiments": 2,
        "max_parallel": 1,
        "timeout_seconds": 900,
    }
    assert experiment["steps"][3]["id"] == (
        "validate_repair_campaign_stackability_queue"
    )
    assert experiment["steps"][3]["requires"] == [
        "build_repair_campaign_stackability_queue"
    ]
    assert experiment["steps"][4]["id"] == (
        "run_repair_campaign_stackability_queue_bounded_local"
    )
    assert experiment["steps"][4]["requires"] == [
        "validate_repair_campaign_stackability_queue"
    ]
    assert experiment["metadata"]["repair_campaign_stackability_queue_path"] in (
        experiment["steps"][4]["command"]
    )
    assert experiment["steps"][1]["requires"] == [
        "assert_repair_budget_waterfill_work_order_materialized"
    ]
    assert "--score-report-out" in command
    postconditions = experiment["steps"][1]["postconditions"]
    assert {
        "type": "json_equals",
        "path": experiment["metadata"]["repair_campaign_score_report_path"],
        "key": "schema",
        "equals": REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
    } in postconditions


def test_repair_campaign_score_queue_cli_writes_and_score_step_runs(
    tmp_path: Path,
) -> None:
    work_order_path = tmp_path / "waterfill_work_order.json"
    waterfill_queue_path = tmp_path / "repair_budget_waterfill_queue.json"
    score_queue_path = tmp_path / "repair_campaign_score_queue.json"
    work_order_path.write_text(
        json.dumps(_work_order(tmp_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    waterfill_queue_path.write_text(
        json.dumps(_waterfill_queue(work_order_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_repair_campaign_score_queue.py",
            "--repair-budget-waterfill-queue",
            str(waterfill_queue_path),
            "--score-queue-out",
            str(score_queue_path),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "unit_repair_campaign_score_queue",
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_result = json.loads(result.stdout)
    assert cli_result["ready_for_exact_eval_dispatch"] is False
    assert cli_result["ready_experiment_count"] == 1
    queue = json.loads(score_queue_path.read_text(encoding="utf-8"))
    prerequisite_command = queue["experiments"][0]["steps"][0]["command"]
    prerequisite = subprocess.run(
        [sys.executable, *prerequisite_command[1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prerequisite.returncode == 0, prerequisite.stderr
    command = queue["experiments"][0]["steps"][1]["command"]
    step_result = subprocess.run(
        [sys.executable, *command[1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert step_result.returncode == 0, step_result.stderr
    report_path = Path(
        queue["experiments"][0]["metadata"]["repair_campaign_score_report_path"]
    )
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA
    assert report["rows"][0]["typed_response_id"] == "segnet_region_ready"
    assert report["ready_for_local_mlx_advisory_execution_count"] == 1
    stackability_command = queue["experiments"][0]["steps"][2]["command"]
    stackability_result = subprocess.run(
        [sys.executable, *stackability_command[1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert stackability_result.returncode == 0, stackability_result.stderr
    stackability_queue_path = Path(
        queue["experiments"][0]["metadata"]["repair_campaign_stackability_queue_path"]
    )
    assert stackability_queue_path.is_file()
    stackability_queue = json.loads(stackability_queue_path.read_text(encoding="utf-8"))
    assert stackability_queue["schema"] == QUEUE_SCHEMA
    assert stackability_queue["metadata"]["ready_experiment_count"] == 1
    validate_command = queue["experiments"][0]["steps"][3]["command"]
    validate_result = subprocess.run(
        [sys.executable, *validate_command[1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate_result.returncode == 0, validate_result.stderr
    worker_command = queue["experiments"][0]["steps"][4]["command"]
    worker_result = subprocess.run(
        [sys.executable, *worker_command[1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert worker_result.returncode == 0, worker_result.stderr
    worker_result_path = Path(
        queue["experiments"][0]["metadata"][
            "repair_campaign_stackability_worker_result_path"
        ]
    )
    worker_payload = json.loads(worker_result_path.read_text(encoding="utf-8"))
    assert worker_payload["schema"] == "experiment_queue_worker_result.v1"
    assert worker_payload["failure_count"] == 0
    stackability_experiment = stackability_queue["experiments"][0]
    assert Path(stackability_experiment["metadata"]["probe_output_path"]).is_file()
    assert Path(stackability_experiment["metadata"]["replay_bundle_path"]).is_file()
    assert Path(stackability_experiment["metadata"]["learning_signal_path"]).is_file()
    assert Path(
        stackability_experiment["metadata"]["posterior_append_report_path"]
    ).is_file()


def test_repair_campaign_score_queue_defers_missing_work_order_to_prerequisite_step(
    tmp_path: Path,
) -> None:
    waterfill_queue = _waterfill_queue(tmp_path / "missing.json")

    queue = build_repair_campaign_score_queue(
        repo_root=REPO_ROOT,
        repair_budget_waterfill_queue=waterfill_queue,
        repair_budget_waterfill_queue_path=tmp_path / "repair_budget_waterfill_queue.json",
        results_root=tmp_path / "results",
        queue_id="unit_repair_campaign_score_queue",
    )

    experiment = queue["experiments"][0]
    assert experiment["status"] == "queued"
    assert experiment["metadata"]["queue_actuation_ready"] is True
    assert experiment["metadata"]["queue_actuation_blockers"] == []
    assert (
        experiment["metadata"]["repair_budget_waterfill_work_order_exists_at_build"]
        is False
    )
    assert experiment["steps"][0]["id"] == (
        "assert_repair_budget_waterfill_work_order_materialized"
    )
