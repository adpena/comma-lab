# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

from tac.optimization.cooperative_receiver_campaigns import (
    TT5L_MEASURED_TIMING_SMOKE_COMMAND,
    build_campaign_queue,
    render_markdown,
    write_campaign_queue,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate


def test_campaign_queue_contains_four_team_convergence_and_blocks_dispatch() -> None:
    manifest = build_campaign_queue()

    assert manifest["schema"] == "tac_cooperative_receiver_campaign_queue_v1"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_ready_count"] == 0
    assert {
        "1d62a114",
        "27cd8b41",
        "cbc6b48b",
        "d1fb9f6a",
        "fdfc347f",
        "local_dpw1_substrate",
        "campaign_lane_c5_20260514",
        "campaign_lane_c7_20260514",
        "campaign_lane_c2_20260514",
        "campaign_lane_c4_20260514",
        "campaign_lane_c3_20260514",
    }.issubset(set(manifest["source_commits"]))

    rows = manifest["top_k"]
    assert [row["campaign_id"] for row in rows[:7]] == [
        "darts_confirmed_time_traveler_config",
        "time_traveler_world_model_substrate",
        "sabor_boundary_only_renderer",
        "s2sbs_hf_byte_stuffing",
        "driving_prior_pretrained_renderer_2032",
        "driving_prior_world_model_substrate",
        "h15_coord_mlp_residual_sidecar_pr103_on_pr106",
    ]
    assert "tools/probe_driving_prior_readiness.py" in rows[4]["timing_smoke_command"]
    assert "driving_prior_world_model/tests" in rows[5]["timing_smoke_command"]
    assert "tools/probe_coord_mlp_residual_sidecar.py" in rows[6]["timing_smoke_command"]
    assert len(rows) == 19
    assert all(validate_proxy_candidate(row) == [] for row in rows)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in rows)
    assert all(row["score_claim"] is False for row in rows)


def test_time_traveler_campaigns_use_measured_timing_smoke_not_help() -> None:
    manifest = build_campaign_queue()
    rows = {row["campaign_id"]: row for row in manifest["top_k"]}
    time_traveler_rows = [
        rows["darts_confirmed_time_traveler_config"],
        rows["time_traveler_world_model_substrate"],
    ]

    for row in time_traveler_rows:
        command = row["timing_smoke_command"]
        assert command == TT5L_MEASURED_TIMING_SMOKE_COMMAND
        assert "smoke_time_traveler_l5_autonomy_macos_cpu.py" in command
        assert " --help" not in command
        assert "--epochs 1" in command
        assert "--batch-size 1" in command
        assert "--allow-non-darwin" in command


def test_long_term_campaign_backfill_rows_have_metadata_and_dispatch_gates() -> None:
    manifest = build_campaign_queue()
    rows = {row["campaign_id"]: row for row in manifest["top_k"]}
    expected = {
        "c5_full_cooperative_receiver_substrate_campaign_20260514": {
            "lane_id": "lane_c5_full_cooperative_receiver_substrate_campaign_20260514",
            "tier": "medium_to_long_term",
            "horizon": "4-8",
            "cost": [30.00, 50.00],
        },
        "c7_darts_supernet_architecture_search_campaign_20260514": {
            "lane_id": "lane_c7_darts_supernet_architecture_search_campaign_20260514",
            "tier": "medium_to_long_term",
            "horizon": "6-12",
            "cost": [100.00, 300.00],
        },
        "c2_z7_mature_predictive_receiver_l5_campaign_20260514": {
            "lane_id": "lane_c2_z7_mature_predictive_receiver_l5_campaign_20260514",
            "tier": "long_term",
            "horizon": "8-12",
            "cost": [50.00, 100.00],
        },
        "c4_queued_architectural_moves_campaign_20260514": {
            "lane_id": "lane_c4_queued_architectural_moves_campaign_20260514",
            "tier": "short_to_medium_term",
            "horizon": "12-24",
            "cost": [50.00, 150.00],
        },
        "c3_multi_year_zen_floor_sub_005_campaign_20260514": {
            "lane_id": "lane_c3_multi_year_zen_floor_sub_005_campaign_20260514",
            "tier": "multi_year",
            "horizon": "52-156",
            "cost": [500.00, 2000.00],
        },
    }

    assert expected.keys() <= rows.keys()
    for campaign_id, spec in expected.items():
        row = rows[campaign_id]
        assert row["lane_id"] == spec["lane_id"]
        assert row["lane_class"] == "substrate_engineering"
        assert row["campaign_tier"] == spec["tier"]
        assert row["expected_horizon_weeks"] == spec["horizon"]
        assert row["estimated_cost_usd_band"] == spec["cost"]
        assert row["cost_metadata"]["estimated_cost_usd_band"] == spec["cost"]
        assert row["timeline_metadata"]["expected_horizon_weeks"] == spec["horizon"]
        assert row["timeline_metadata"]["campaign_tier"] == spec["tier"]
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["rank_or_kill_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["dispatch_gating"]["dispatch_requires_lane_claim"] is True
        assert row["dispatch_gating"]["dispatch_requires_operator_authorization"] is True
        assert "requires_lane_dispatch_claim_before_gpu_or_remote_eval" in row["dispatch_blockers"]
        assert validate_proxy_candidate(row) == []


def test_campaign_queue_markdown_and_writer_are_deterministic(tmp_path) -> None:
    json_path = tmp_path / "campaign_queue.json"
    md_path = tmp_path / "campaign_queue.md"

    manifest = write_campaign_queue(json_path, markdown_output=md_path, top_k=2)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert loaded == manifest
    assert [row["rank_hint"] for row in loaded["top_k"]] == [1, 2]
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown == render_markdown(manifest)
    assert "time_traveler_world_model_substrate" in markdown
    assert "score_claim: `false`" in markdown
