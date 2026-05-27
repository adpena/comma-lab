# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.repair_campaign_stackability_queue import (
    REPAIR_CAMPAIGN_STACKABILITY_QUEUE_METADATA_SCHEMA,
    build_repair_campaign_stackability_queue,
)
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
    build_repair_campaign_stackability_probe,
    score_repair_campaign,
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
                },
                {
                    "schema": (
                        "frontier_rate_attack_repair_budget_typed_response_row.v1"
                    ),
                    "typed_response_id": "selector_missing",
                    "candidate_id": "per_region_selector_codec",
                    "acquisition_id": "selector_acq",
                    "correction_family": "per_region_selector_codec",
                    "targeted_dimensions": ["selector_stream", "region"],
                    "operation_levels": ["entropy_coder"],
                    "entropy_position_label": "selector_codec_entropy",
                    "requested_repair_bytes": 8,
                    "objective_delta_score_units": -0.0008,
                    **_false_authority(),
                },
            ],
            **_false_authority(),
        },
        **_false_authority(),
    }


def test_stackability_probe_preserves_optimizer_signal_and_authority(
    tmp_path: Path,
) -> None:
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)

    probe = build_repair_campaign_stackability_probe(
        score_report=report,
        typed_response_id="segnet_region_ready",
        repo_root=tmp_path,
    )

    assert probe["schema"] == REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA
    assert probe["stackability_ready"] is True
    assert probe["component_response_axis"] == "[macOS-MLX research-signal]"
    assert probe["allocated_repair_bytes"] == 32
    assert probe["entropy_position_class"] == "pre_entropy_distribution_shaping"
    assert probe["optimizer_allocation"]["targeted_dimensions"] == [
        "segnet",
        "region",
    ]
    assert probe["source_score_row"]["interaction_scope"]["pair_indices"] == [7, 9]
    assert probe["budget_spend_allowed"] is False
    assert probe["ready_for_exact_eval_dispatch"] is False
    assert "local_mlx_probe_is_not_score_authority" in probe["blockers"]


def test_stackability_queue_emits_executable_local_probe(tmp_path: Path) -> None:
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)
    report_path = tmp_path / "repair_campaign_score_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    queue = build_repair_campaign_stackability_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=report_path,
        results_root=tmp_path / "results",
        queue_id="stackability_test",
    )

    assert queue["metadata"]["schema"] == (
        REPAIR_CAMPAIGN_STACKABILITY_QUEUE_METADATA_SCHEMA
    )
    assert queue["metadata"]["ready_experiment_count"] == 1
    assert queue["metadata"]["blocked_experiment_count"] == 0
    experiment = queue["experiments"][0]
    assert experiment["status"] == "queued"
    command = [
        sys.executable if item == ".venv/bin/python" else str(item)
        for item in experiment["steps"][0]["command"]
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads(
        Path(experiment["metadata"]["probe_output_path"]).read_text(encoding="utf-8")
    )
    assert artifact["schema"] == REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA
    assert artifact["stackability_ready"] is True
    assert artifact["budget_spend_allowed"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False


def test_stackability_queue_freezes_bad_selected_allocation_with_exact_missing_names(
    tmp_path: Path,
) -> None:
    report = score_repair_campaign(payload=_work_order(tmp_path), repo_root=tmp_path)
    selected = report["optimizer_decision"]["selected_allocation_rows"][0]
    selected["typed_response_id"] = "selector_missing"

    queue = build_repair_campaign_stackability_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=tmp_path / "repair_campaign_score_report.json",
        results_root=tmp_path / "results",
        queue_id="stackability_blocked_test",
    )

    experiment = queue["experiments"][0]
    blockers = experiment["metadata"]["queue_actuation_blockers"]
    assert experiment["status"] == "frozen"
    assert "local_mlx_response_path:missing_or_unverified" in blockers
    assert "reference_local_mlx_response_path:missing_or_unverified" in blockers
    assert "runtime_consumption_proof_path" in blockers
    assert experiment["metadata"]["budget_spend_allowed"] is False
    assert experiment["metadata"]["ready_for_exact_eval_dispatch"] is False


def test_stackability_queue_freezes_empty_optimizer_selection(
    tmp_path: Path,
) -> None:
    work_order = _work_order(tmp_path)
    work_order["receiver_closed_rate_credit"]["receiver_closed_saved_bytes_total"] = 0
    work_order["typed_response_ledger"]["available_receiver_closed_rate_credit_bytes"] = 0
    report = score_repair_campaign(payload=work_order, repo_root=tmp_path)

    queue = build_repair_campaign_stackability_queue(
        repo_root=REPO_ROOT,
        score_report=report,
        score_report_path=tmp_path / "repair_campaign_score_report.json",
        results_root=tmp_path / "results",
        queue_id="stackability_empty_test",
    )

    assert queue["metadata"]["selected_allocation_count"] == 0
    assert queue["metadata"]["ready_experiment_count"] == 0
    assert queue["metadata"]["blocked_experiment_count"] == 1
    assert "optimizer_selected_allocation_rows_empty" in queue["metadata"][
        "queue_actuation_blockers"
    ]
    assert queue["experiments"][0]["status"] == "frozen"
    assert queue["experiments"][0]["metadata"]["budget_spend_allowed"] is False
