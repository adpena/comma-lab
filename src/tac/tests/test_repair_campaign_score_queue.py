# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA
from comma_lab.scheduler.repair_campaign_score_queue import (
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    REPAIR_CAMPAIGN_SCORE_EXPERIMENT_METADATA_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_QUEUE_METADATA_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_QUEUE_SUMMARY_SCHEMA,
    REPAIR_POSTERIOR_ACQUISITION_FOLLOWUP_EXPERIMENT_METADATA_SCHEMA,
    REPAIR_POSTERIOR_ACQUISITION_FOLLOWUP_QUEUE_METADATA_SCHEMA,
    build_repair_campaign_score_queue,
    build_repair_posterior_acquisition_followup_queue,
    summarize_repair_campaign_score_queue,
    summarize_repair_posterior_acquisition_followup_queue,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
    REPAIR_CAMPAIGN_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA,
)
from tac.optimization.repair_campaign_posterior import (
    REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_POSTERIOR_ACQUISITION_FOLLOWUP_SCHEMA,
    REPAIR_CAMPAIGN_POSTERIOR_PRIOR_SUMMARY_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
)

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
                {
                    "schema": (
                        "frontier_rate_attack_repair_budget_typed_response_row.v1"
                    ),
                    "typed_response_id": "selector_missing",
                    "candidate_id": "per_region_selector_codec_missing",
                    "acquisition_id": "selector_codec_acq",
                    "correction_family": "per_region_selector_codec",
                    "targeted_dimensions": ["selector_stream", "region"],
                    "operation_levels": ["region"],
                    "entropy_position_label": "selector_codec_entropy",
                    "requested_repair_bytes": 16,
                    "objective_delta_score_units": -0.0002,
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
    blocked_signal_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "build_repair_campaign_blocked_learning_signals"
    )
    assert blocked_signal_step["requires"] == [
        "score_repair_campaign_from_typed_ledger"
    ]
    assert experiment["metadata"][
        "repair_campaign_blocked_learning_signal_report_path"
    ].endswith("repair_campaign_blocked_learning_signal_report.json")
    blocked_append_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "append_blocked_repair_campaign_learning_posterior"
    )
    assert blocked_append_step["requires"] == [
        "build_repair_campaign_blocked_learning_signals"
    ]
    assert experiment["metadata"][
        "repair_campaign_blocked_posterior_append_report_path"
    ].endswith("repair_campaign_blocked_posterior_append_report.json")
    assert experiment["metadata"]["blocked_learning_followup_default"] is True
    stackability_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "build_repair_campaign_stackability_queue"
    )
    stackability_command = stackability_step["command"]
    assert stackability_command[:2] == [
        ".venv/bin/python",
        "tools/build_repair_campaign_stackability_queue.py",
    ]
    assert stackability_step["requires"] == [
        "score_repair_campaign_from_typed_ledger"
    ]
    assert experiment["metadata"]["repair_campaign_stackability_queue_path"] in (
        stackability_command
    )
    assert experiment["metadata"][
        "repair_campaign_stackability_worker_result_path"
    ].endswith("repair_campaign_stackability_worker_result.json")
    assert experiment["metadata"][
        "repair_cascade_mlx_probe_queue_path"
    ].endswith("repair_cascade_mlx_probe_queue.json")
    assert experiment["metadata"][
        "repair_cascade_mlx_probe_worker_result_path"
    ].endswith("repair_cascade_mlx_probe_worker_result.json")
    assert experiment["metadata"]["continual_learning_followup_default"] is True
    assert experiment["metadata"]["stackability_worker_limits"] == {
        "schema": "repair_campaign_stackability_worker_limits.v1",
        "max_steps": 8,
        "max_experiments": 2,
        "max_parallel": 1,
        "timeout_seconds": 900,
    }
    assert experiment["metadata"]["cascade_mlx_probe_followup_default"] is True
    cascade_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "build_repair_cascade_mlx_probe_queue"
    )
    assert cascade_step["requires"] == ["score_repair_campaign_from_typed_ledger"]
    assert cascade_step["command"][:2] == [
        ".venv/bin/python",
        "tools/build_repair_cascade_mlx_probe_queue.py",
    ]
    assert experiment["metadata"]["repair_cascade_mlx_probe_queue_path"] in (
        cascade_step["command"]
    )
    validate_cascade_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "validate_repair_cascade_mlx_probe_queue"
    )
    assert validate_cascade_step["requires"] == [
        "build_repair_cascade_mlx_probe_queue"
    ]
    worker_cascade_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "run_repair_cascade_mlx_probe_queue_bounded_local"
    )
    assert worker_cascade_step["requires"] == [
        "validate_repair_cascade_mlx_probe_queue"
    ]
    assert (
        worker_cascade_step["command"][
            worker_cascade_step["command"].index("--max-steps") + 1
        ]
        == "5"
    )
    validate_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "validate_repair_campaign_stackability_queue"
    )
    assert validate_step["requires"] == [
        "build_repair_campaign_stackability_queue"
    ]
    worker_step = next(
        step
        for step in experiment["steps"]
        if step["id"] == "run_repair_campaign_stackability_queue_bounded_local"
    )
    assert worker_step["requires"] == [
        "validate_repair_campaign_stackability_queue"
    ]
    assert experiment["metadata"]["repair_campaign_stackability_queue_path"] in (
        worker_step["command"]
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


def test_repair_campaign_score_queue_can_bind_posterior_prior_input(
    tmp_path: Path,
) -> None:
    work_order_path = tmp_path / "waterfill_work_order.json"
    posterior_path = tmp_path / "repair_campaign_stackability_posterior.jsonl"
    posterior_path.write_text("", encoding="utf-8")
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
        posterior_path=posterior_path,
    )

    experiment = queue["experiments"][0]
    command = next(
        step["command"]
        for step in experiment["steps"]
        if step["id"] == "score_repair_campaign_from_typed_ledger"
    )
    assert "--posterior" in command
    assert str(posterior_path) in command
    assert experiment["metadata"]["campaign_scorer_uses_posterior_priors"] is True
    assert experiment["metadata"]["repair_campaign_stackability_posterior_path"] == str(
        posterior_path
    )
    assert queue["metadata"]["campaign_scorer_uses_posterior_priors"] is True
    assert (
        DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH.name
        == "repair_campaign_stackability_posterior.jsonl"
    )
    summary = summarize_repair_campaign_score_queue(
        repair_campaign_score_queue=queue,
        queue_path=tmp_path / "repair_campaign_score_queue.json",
        posterior_path=posterior_path,
    )
    assert summary["schema"] == REPAIR_CAMPAIGN_SCORE_QUEUE_SUMMARY_SCHEMA
    assert summary["campaign_scorer_uses_posterior_priors"] is True
    assert summary["posterior_prior_summary"][
        "schema"
    ] == REPAIR_CAMPAIGN_POSTERIOR_PRIOR_SUMMARY_SCHEMA


def test_repair_posterior_acquisition_followup_queue_routes_to_child_queue(
    tmp_path: Path,
) -> None:
    response_harvest_path = tmp_path / "targeted_response_harvest.json"
    targeted_queue_path = tmp_path / "targeted_queue.json"
    repair_score_queue_path = tmp_path / "repair_score_queue.json"
    response_harvest_path.write_text('{"schema":"response_harvest.v1"}\n')
    targeted_queue_path.write_text(
        json.dumps(
            {
                "schema": QUEUE_SCHEMA,
                "queue_id": "targeted_component_correction_queue",
                "controls": {"mode": "running"},
                "metadata": _false_authority(),
                "experiments": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    repair_score_queue_path.write_text(
        json.dumps(
            {
                "schema": QUEUE_SCHEMA,
                "queue_id": "repair_campaign_score_queue",
                "controls": {"mode": "running"},
                "metadata": _false_authority(),
                "experiments": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    route = {
        "schema": REPAIR_CAMPAIGN_POSTERIOR_ACQUISITION_FOLLOWUP_SCHEMA,
        "recommended_acquisition_policy": (
            "increase_priority_for_targeted_component_response_harvest"
        ),
        "observation_count": 2,
        "family_ids": ["region_boundary_repair"],
        "typed_response_ids": ["typed_001"],
        "priority_score": 100,
        "activation_action": "harvest_targeted_component_response_rows",
        "queue_artifact_key": "targeted_component_correction_response_harvest",
        "required_evidence_surface": "targeted_component_correction_response_harvest",
        "missing_artifact_total": 0,
        "blocker_total": 0,
        **_false_authority(),
    }
    mlx_custody_route = {
        **route,
        "recommended_acquisition_policy": (
            "materialize_missing_local_mlx_custody_before_stackability"
        ),
        "activation_action": "materialize_missing_local_mlx_custody",
        "queue_artifact_key": "repair_cascade_mlx_probe_queue",
        "required_evidence_surface": "local_mlx_research_signal",
        "priority_score": 70,
    }
    posterior_summary = {
        "schema": REPAIR_CAMPAIGN_POSTERIOR_PRIOR_SUMMARY_SCHEMA,
        "posterior_path": str(tmp_path / "posterior.jsonl"),
        "posterior_row_count": 2,
        "family_prior_count": 0,
        "family_priors": [],
        "acquisition_followup_route_count": 2,
        "acquisition_followup_routes": [route, mlx_custody_route],
        "allowed_use": "unit_test",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **_false_authority(),
    }

    queue = build_repair_posterior_acquisition_followup_queue(
        repo_root=tmp_path,
        posterior_prior_summary=posterior_summary,
        artifact_paths_by_key={
            "targeted_component_correction_response_harvest": response_harvest_path,
            "targeted_component_correction_queue": targeted_queue_path,
            "repair_campaign_score_queue": repair_score_queue_path,
        },
        results_root=tmp_path / "results",
        queue_id="unit_repair_posterior_followup",
    )

    assert queue["schema"] == QUEUE_SCHEMA
    assert (
        queue["metadata"]["schema"]
        == REPAIR_POSTERIOR_ACQUISITION_FOLLOWUP_QUEUE_METADATA_SCHEMA
    )
    assert queue["metadata"]["ready_experiment_count"] == 2
    experiment = queue["experiments"][0]
    assert experiment["status"] == "queued"
    assert (
        experiment["metadata"]["schema"]
        == REPAIR_POSTERIOR_ACQUISITION_FOLLOWUP_EXPERIMENT_METADATA_SCHEMA
    )
    assert experiment["metadata"]["queue_actuation_ready"] is True
    assert experiment["metadata"]["executable_queue_artifact_keys"] == [
        "targeted_component_correction_queue"
    ]
    step_commands = [step["command"] for step in experiment["steps"]]
    assert any(command[:3] == [".venv/bin/python", "-m", "json.tool"] for command in step_commands)
    assert any(
        "tools/experiment_queue.py" in command
        and "run-worker" in command
        and str(targeted_queue_path) in command
        for command in step_commands
    )
    mlx_experiment = queue["experiments"][1]
    assert mlx_experiment["status"] == "queued"
    assert mlx_experiment["metadata"]["executable_queue_artifact_keys"] == [
        "repair_campaign_score_queue"
    ]
    assert mlx_experiment["metadata"]["queue_actuation_blockers"] == []
    summary = summarize_repair_posterior_acquisition_followup_queue(
        repair_posterior_followup_queue=queue,
        queue_path=tmp_path / "repair_posterior_acquisition_followup_queue.json",
    )
    assert summary["posterior_acquisition_followup_route_count"] == 2
    assert summary["ready_experiment_count"] == 2


def test_repair_campaign_score_queue_cli_writes_and_score_step_runs(
    tmp_path: Path,
) -> None:
    work_order_path = tmp_path / "waterfill_work_order.json"
    waterfill_queue_path = tmp_path / "repair_budget_waterfill_queue.json"
    score_queue_path = tmp_path / "repair_campaign_score_queue.json"
    posterior_path = tmp_path / "repair_campaign_stackability_posterior.jsonl"
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
            "--posterior",
            str(posterior_path),
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
    assert queue["metadata"]["repair_campaign_stackability_posterior_path"] == str(
        posterior_path
    )
    assert queue["metadata"]["repair_campaign_stackability_posterior_lock_path"] == str(
        posterior_path.with_name(f".{posterior_path.name}.lock")
    )
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
    assert report["optimizer_decision"]["blocked_allocation_count"] == 1
    blocked_signal_command = next(
        step["command"]
        for step in queue["experiments"][0]["steps"]
        if step["id"] == "build_repair_campaign_blocked_learning_signals"
    )
    blocked_signal_result = subprocess.run(
        [sys.executable, *blocked_signal_command[1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked_signal_result.returncode == 0, blocked_signal_result.stderr
    blocked_signal_report_path = Path(
        queue["experiments"][0]["metadata"][
            "repair_campaign_blocked_learning_signal_report_path"
        ]
    )
    blocked_signal_report = json.loads(
        blocked_signal_report_path.read_text(encoding="utf-8")
    )
    assert blocked_signal_report["schema"] == (
        REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA
    )
    assert blocked_signal_report["blocked_signal_count"] == 1
    blocked_signal = blocked_signal_report["learning_signal_rows"][0]
    assert blocked_signal["typed_response_id"] == "selector_missing"
    assert blocked_signal["learning_signal_kind"] == (
        "blocked_repair_campaign_allocation"
    )
    assert blocked_signal["local_planning_update"][
        "local_planning_update_ready"
    ] is True
    blocked_append_command = next(
        step["command"]
        for step in queue["experiments"][0]["steps"]
        if step["id"] == "append_blocked_repair_campaign_learning_posterior"
    )
    assert "--posterior-path" in blocked_append_command
    assert str(posterior_path) in blocked_append_command
    assert "--lock-path" in blocked_append_command
    assert str(posterior_path.with_name(f".{posterior_path.name}.lock")) in (
        blocked_append_command
    )
    blocked_append_result = subprocess.run(
        [sys.executable, *blocked_append_command[1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked_append_result.returncode == 0, blocked_append_result.stderr
    blocked_append_report_path = Path(
        queue["experiments"][0]["metadata"][
            "repair_campaign_blocked_posterior_append_report_path"
        ]
    )
    blocked_append_report = json.loads(
        blocked_append_report_path.read_text(encoding="utf-8")
    )
    assert blocked_append_report["schema"] == (
        REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA
    )
    assert blocked_append_report["signal_count"] == 1
    assert blocked_append_report["posterior_path"] == str(posterior_path)
    assert posterior_path.is_file()
    stackability_command = next(
        step["command"]
        for step in queue["experiments"][0]["steps"]
        if step["id"] == "build_repair_campaign_stackability_queue"
    )
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
    validate_command = next(
        step["command"]
        for step in queue["experiments"][0]["steps"]
        if step["id"] == "validate_repair_campaign_stackability_queue"
    )
    validate_result = subprocess.run(
        [sys.executable, *validate_command[1:]],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate_result.returncode == 0, validate_result.stderr
    worker_command = next(
        step["command"]
        for step in queue["experiments"][0]["steps"]
        if step["id"] == "run_repair_campaign_stackability_queue_bounded_local"
    )
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
    assert experiment["steps"][0]["id"] == "validate_repair_budget_waterfill_source_queue"
    assert experiment["steps"][1]["id"] == "run_repair_budget_waterfill_source_queue"
    assert experiment["steps"][2]["id"] == (
        "assert_repair_budget_waterfill_work_order_materialized"
    )
    assert experiment["steps"][2]["requires"] == [
        "run_repair_budget_waterfill_source_queue"
    ]


def test_blocked_learning_signal_cli_accepts_child_queue_activation_plan(
    tmp_path: Path,
) -> None:
    activation_plan_path = tmp_path / "activation_plan.json"
    signal_report_path = tmp_path / "activation_signal_report.json"
    activation_plan = {
        "schema": REPAIR_CAMPAIGN_CHILD_QUEUE_ACTIVATION_PLAN_SCHEMA,
        "queue_id": "repair_budget_waterfill_queue",
        "queue_sha256": "a" * 64,
        "queue_path": "repair_budget_waterfill_queue.json",
        "blocked_experiment_count": 1,
        "blocked_experiments": [
            {
                "experiment_id": "repair_waterfill_chain",
                "status": "frozen",
                "tags": ["segnet-posenet-waterfill", "no-score-authority"],
                "activation_blockers": [
                    "no_targeted_component_correction_response_rows",
                    "no_receiver_closed_saved_bytes_available",
                ],
                "activation_actions": [
                    {
                        "blocker": "no_targeted_component_correction_response_rows",
                        "activation_action": (
                            "harvest_targeted_component_response_rows"
                        ),
                        "evidence_surface": (
                            "targeted_component_correction_response_harvest"
                        ),
                    },
                    {
                        "blocker": "no_receiver_closed_saved_bytes_available",
                        "activation_action": (
                            "materialize_receiver_closed_rate_budget_credit"
                        ),
                        "evidence_surface": "receiver_closed_correction_budget",
                    },
                ],
                "step_count": 1,
                "step_evidence_refs": [
                    {
                        "step_id": "inspect_blocked_repair_waterfill",
                        "telemetry_input_artifact_paths": [
                            "targeted_component_correction_response_harvest.json"
                        ],
                        "postcondition_paths": ["repair_budget_waterfill.json"],
                    }
                ],
            }
        ],
        **_false_authority(),
    }
    activation_plan_path.write_text(
        json.dumps(activation_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_repair_campaign_blocked_learning_signals.py",
            "--activation-plan",
            str(activation_plan_path),
            "--blocked-signal-report-out",
            str(signal_report_path),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_result = json.loads(result.stdout)
    assert cli_result["source_kind"] == "activation_plan"
    report = json.loads(signal_report_path.read_text(encoding="utf-8"))
    assert report["schema"] == REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA
    assert report["blocked_signal_count"] == 1
    signal = report["learning_signal_rows"][0]
    assert signal["learning_signal_kind"] == "blocked_child_queue_activation_plan"
    assert signal["family_id"] == "segnet-posenet-waterfill"
    assert signal["local_planning_update"]["recommended_acquisition_policy"] == (
        "increase_priority_for_targeted_component_response_harvest"
    )
