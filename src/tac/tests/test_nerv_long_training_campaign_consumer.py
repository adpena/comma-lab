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
    assert verdict["ready_local_mlx_experiment_count"] == 1
    assert verdict["blocked_experiment_count"] == 2
    assert verdict["gated_experiment_count"] == 3
    assert verdict["family_summary"]["hi_nerv"]["ready_local_mlx_count"] == 0
    assert verdict["family_summary"]["hi_nerv"]["blocked_count"] == 2
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
    assert first["id"].startswith("snerv_")
    assert "tools/run_compact_renderer_mlx_spine_runner.py" in first["command"]
    assert first["score_lowering_gate"]["receiver_proof_required"] is True
    assert first["score_lowering_gate"]["cpu_replay_ready"] is False
    assert first["launch_authority_contract"] == {
        "schema": "nerv_long_training_queue_launch_authority_contract.v1",
        "queue_status_is_local_mlx_plan": True,
        "queue_status_is_receiver_proof": False,
        "queue_status_is_cpu_replay_proof": False,
        "queue_status_is_exact_eval_authority": False,
    }


def test_long_training_campaign_consumer_requires_launch_authority_contract() -> None:
    queue = json.loads(json.dumps(_campaign_plan()["experiment_queue"]))
    snerv = next(row for row in queue["experiments"] if row["family"] == "snerv")
    snerv.pop("launch_authority_contract")

    verdict = consumer.consume_candidate(queue)

    assert verdict["planner_action"] == "close_campaign_row_blockers_then_reconsume"
    assert verdict["local_mlx_route_recommended"] is False
    assert verdict["ready_local_mlx_experiment_count"] == 0
    assert any(
        blocker.endswith("_launch_authority_contract_missing")
        for blocker in verdict["blockers"]
    )


def test_long_training_campaign_consumer_preserves_hinerv_supersession_metadata(
) -> None:
    plan = _hinerv_official_supersession_campaign_plan()

    verdict = consumer.consume_candidate(plan["experiment_queue"])

    assert verdict["family_summary"]["hi_nerv"]["ready_local_mlx_count"] == 0
    assert verdict["family_summary"]["hi_nerv"]["blocked_count"] == 1
    [snerv_experiment_id] = verdict["selected_local_mlx_experiment_ids"]
    assert snerv_experiment_id.startswith("snerv_snerv_np600_haar_lv2_")
    assert snerv_experiment_id.endswith("_native_rate_aware_training")


def test_long_training_campaign_consumer_accepts_extracted_experiment_queue() -> None:
    queue = _campaign_plan()["experiment_queue"]

    verdict = consumer.consume_candidate(queue)

    assert verdict["source_schema"] == "experiment_queue.v1"
    assert verdict["queue_id"] == "nerv_long_training_campaign_queue.v1"
    assert verdict["campaign_row_count"] == 0
    assert verdict["experiment_count"] == 3
    assert verdict["ready_local_mlx_experiment_count"] == 1
    assert verdict["blocked_experiment_count"] == 2
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
    assert verdict["ready_local_mlx_experiment_count"] == 1
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


def _hinerv_official_supersession_campaign_plan() -> dict:
    hinerv_budget = _hinerv_budget()
    generic = dict(hinerv_budget["selected_candidates"][0])
    generic.update(
        {
            "candidate_id": "hinerv_np600_ld4_ed12_dc8_int8_mixed_ceil36000",
            "decoder_codec": "int8_mixed",
            "nominal_total_payload_bytes": 90_000,
            "byte_headroom": 88_000,
            "use_hierarchical_feature_grid": False,
            "use_convnext_blocks": False,
        }
    )
    official = dict(generic)
    official.update(
        {
            "candidate_id": "hinerv_np600_ld4_ed16_dc8_hfg_cnx_int2_mixed_ceil36000",
            "decoder_codec": "int2_mixed",
            "nominal_total_payload_bytes": 110_000,
            "byte_headroom": 68_000,
            "use_hierarchical_feature_grid": True,
            "use_convnext_blocks": True,
            "local_grid_levels": 2,
            "local_grid_channels": 4,
            "convnext_mlp_ratio": 2,
            "convnext_kernel_size": 3,
        }
    )
    hinerv_budget["selected_candidates"] = [generic, official]
    return build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=hinerv_budget,
        snerv_modelsize_budget=_snerv_budget(),
        optimizer_kinds=("adamw",),
        epochs=16,
        batch_pairs=4,
        learning_rate=2.7e-5,
        output_root="/Volumes/VertigoDataTier/pact/test_campaigns",
        max_candidates_per_family=1,
        candidate_feedback_sources=(
            {
                "schema": "nerv_candidate_feedback_row.v1",
                "feedback_kind": "training_telemetry",
                "family": "hi_nerv",
                "candidate_id": generic["candidate_id"],
                "candidate_num_pairs": 600,
                "measured_num_pairs": 600,
                "feedback_scope": "full600_training_telemetry",
                "scope_matches_candidate": True,
                "feedback_ready": False,
                "seg_stagnation_detected": True,
                "observed_learning_rate": 2.7e-5,
                "recommended_segnet_distillation_weight": 2.0,
                "recommended_launch_mutations": [
                    "increase_segnet_distillation_weight_from_stagnation_telemetry"
                ],
            },
        ),
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
                "candidate_id": (
                    "snerv_np600_haar_lv2_lfb1p5_stepb0p5_"
                    "fc11e2_p1_mfu1-2-4_hfr0_t0_adbase_int4_symmetric_ceil178000"
                ),
                "num_pairs": 600,
                "hard_byte_ceiling": 178_000,
                "wavelet": "haar",
                "levels": 2,
                "bits_per_coeff": 1.5,
                "step_map_bits_per_coeff": 0.5,
                "decoder_payload_codec": "int4_symmetric",
                "snerv_model_size_adapter": "snerv_fc_dim_emb_size_adapter_v1",
                "fc_dim": 11,
                "emb_size": 2,
                "patch_radius": 1,
                "mfu_scales": [1, 2, 4],
                "hfr_gain": 0.0,
                "temporal_context": 0,
                "decoder_feature_count": 16,
                "nominal_total_payload_bytes": 190_000,
                "nominal_under_ceiling": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
