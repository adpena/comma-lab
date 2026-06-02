# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.nerv_long_training_campaign_plan import (
    build_nerv_long_training_campaign_plan,
)
from tac.cathedral.consumer_contract import HookNumber, validate_consumer_module
from tac.cathedral_consumers import nerv_long_training_campaign_consumer as consumer
from tools import build_nerv_long_training_campaign_consumer_verdict as cli


def test_long_training_campaign_consumer_routes_local_mlx_without_exact_authority() -> None:
    plan = _campaign_plan()

    verdict = consumer.consume_candidate(plan)

    assert verdict["schema"] == "nerv_long_training_campaign_consumer_result.v1"
    assert verdict["planner_action"] == (
        "route_launchable_local_mlx_campaign_rows_without_exact_dispatch"
    )
    assert verdict["local_mlx_route_recommended"] is True
    assert verdict["ready_local_mlx_experiment_count"] == 3
    assert verdict["blocked_experiment_count"] == 0
    assert verdict["gated_experiment_count"] == 3
    assert verdict["family_summary"]["hi_nerv"]["ready_local_mlx_count"] == 2
    assert verdict["family_summary"]["hi_nerv"]["gated_count"] == 2
    assert verdict["family_summary"]["snerv"]["ready_local_mlx_count"] == 1
    assert verdict["family_summary"]["snerv"]["blocked_count"] == 0
    assert verdict["family_summary"]["snerv"]["gated_count"] == 1
    assert verdict["exact_auth_recommended"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False
    assert verdict["score_claim"] is False
    assert verdict["promotion_eligible"] is False
    assert "campaign_plan_is_not_execution" in verdict["blockers"]
    assert (
        "PR95_same_axis_control_replay_required_before_beat_claim"
        in verdict["blocked_exact_dispatch_dependencies"]
    )
    first = verdict["selected_local_mlx_experiments"][0]
    assert first["id"].startswith("hi_nerv_")
    assert "tools/run_compact_renderer_mlx_spine_runner.py" in first["command"]
    assert first["score_lowering_gate"]["receiver_proof_required"] is True
    assert first["score_lowering_gate"]["cpu_replay_ready"] is False


def test_long_training_campaign_consumer_accepts_extracted_experiment_queue() -> None:
    queue = _campaign_plan()["experiment_queue"]

    verdict = consumer.consume_candidate(queue)

    assert verdict["source_schema"] == "experiment_queue.v1"
    assert verdict["queue_id"] == "nerv_long_training_campaign_queue.v1"
    assert verdict["campaign_row_count"] == 0
    assert verdict["experiment_count"] == 3
    assert verdict["ready_local_mlx_experiment_count"] == 3
    assert verdict["local_mlx_route_recommended"] is True
    assert verdict["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_consumer_fails_closed_on_schema_and_authority() -> None:
    verdict = consumer.consume_candidate(
        {
            "schema": "wrong",
            "ready_for_exact_eval_dispatch": True,
            "score_claim": True,
        }
    )

    assert verdict["planner_action"] == "repair_campaign_plan_schema_before_cathedral_consumption"
    assert verdict["local_mlx_route_recommended"] is False
    assert "campaign_plan_schema_mismatch" in verdict["blockers"]
    assert "campaign_score_claim_overclaimed" in verdict["blockers"]
    assert "campaign_ready_for_exact_eval_dispatch_overclaimed" in verdict["blockers"]
    assert verdict["score_claim"] is False
    assert verdict["ready_for_exact_eval_dispatch"] is False


def test_long_training_campaign_consumer_contract() -> None:
    reg = validate_consumer_module(
        consumer,
        module_path="tac.cathedral_consumers.nerv_long_training_campaign_consumer",
    )

    assert reg.contract_compliant, reg.validation_errors
    assert consumer.CONSUMER_NAME == "nerv_long_training_campaign_consumer"
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
    assert consumer.update_from_anchor({"ignored": True}) is None


def test_long_training_campaign_consumer_cli_writes_verdict(tmp_path: Path) -> None:
    source = tmp_path / "campaign.json"
    out_json = tmp_path / "verdict.json"
    out_md = tmp_path / "verdict.md"
    source.write_text(json.dumps(_campaign_plan()), encoding="utf-8")

    rc = cli.main(
        [
            "--campaign-plan-or-queue",
            str(source),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert rc == 0
    verdict = json.loads(out_json.read_text(encoding="utf-8"))
    assert verdict["schema"] == "nerv_long_training_campaign_consumer_result.v1"
    assert verdict["ready_local_mlx_experiment_count"] == 3
    assert verdict["score_claim"] is False
    assert out_md.read_text(encoding="utf-8").startswith(
        "# NeRV Long-Training Campaign Consumer Verdict"
    )


def _campaign_plan() -> dict:
    return build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_hinerv_budget(),
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw", "lion"),
        epochs=16,
        batch_pairs=4,
        learning_rate=3.0e-4,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
    )


def _hinerv_budget() -> dict:
    return {
        "schema": "nerv_modelsize_budget.v1",
        "selected_candidates": [
            {
                "schema": "hinerv_modelsize_candidate.v1",
                "family": "hi_nerv",
                "candidate_id": "hinerv_tiny",
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "decoder_codec": "int4_mixed",
                "nominal_total_payload_bytes": 120_000,
                "nominal_under_ceiling": True,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_budget() -> dict:
    return {
        "schema": "snerv_modelsize_budget.v1",
        "selected_candidates": [
            {
                "schema": "snerv_modelsize_candidate.v1",
                "family": "snerv",
                "candidate_id": "snerv_tiny",
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "decoder_payload_codec": "int4_symmetric",
                "nominal_total_payload_bytes": 190_000,
                "nominal_under_ceiling": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
