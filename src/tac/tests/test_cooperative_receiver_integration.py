# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

from tac.optimization.cooperative_receiver_integration import (
    build_integration_manifest,
    render_markdown,
    write_integration_manifest,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tac.packet_compiler.deterministic_compiler import (
    packetir_operation_set_bridge_contract,
)


def test_integration_manifest_threads_campaigns_into_solver_surfaces() -> None:
    manifest = build_integration_manifest()

    assert manifest["schema"] == "tac_cooperative_receiver_solver_integration_v1"
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["campaign_count"] == 19

    autopilot_rows = manifest["autopilot_dispatch_hook"]["rows"]
    assert len(autopilot_rows) == 19
    assert all(validate_proxy_candidate(row) == [] for row in autopilot_rows)
    assert {
        row["source_campaign_id"]
        for row in autopilot_rows
        if row["source_campaign_id"]
        in {
            "time_traveler_world_model_substrate",
            "sabor_boundary_only_renderer",
            "s2sbs_hf_byte_stuffing",
            "h15_coord_mlp_residual_sidecar_pr103_on_pr106",
            "driving_prior_world_model_substrate",
            "a1_plus_lapose_composition",
            "a1_plus_wavelet_residual_retarget",
            "c5_full_cooperative_receiver_substrate_campaign_20260514",
            "c7_darts_supernet_architecture_search_campaign_20260514",
            "c2_z7_mature_predictive_receiver_l5_campaign_20260514",
            "c4_queued_architectural_moves_campaign_20260514",
            "c3_multi_year_zen_floor_sub_005_campaign_20260514",
        }
    } == {
        "time_traveler_world_model_substrate",
        "sabor_boundary_only_renderer",
        "s2sbs_hf_byte_stuffing",
        "h15_coord_mlp_residual_sidecar_pr103_on_pr106",
        "driving_prior_world_model_substrate",
        "a1_plus_lapose_composition",
        "a1_plus_wavelet_residual_retarget",
        "c5_full_cooperative_receiver_substrate_campaign_20260514",
        "c7_darts_supernet_architecture_search_campaign_20260514",
        "c2_z7_mature_predictive_receiver_l5_campaign_20260514",
        "c4_queued_architectural_moves_campaign_20260514",
        "c3_multi_year_zen_floor_sub_005_campaign_20260514",
    }

    meta = manifest["meta_lagrangian_hook"]
    assert len(meta["rows"]) == 19
    assert meta["score_claim"] is False
    assert all(row["proxy_row"] is True for row in meta["rows"])
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in meta["rows"])

    pareto = manifest["pareto_constraint_hook"]
    assert pareto["binding"] is False
    assert len(pareto["rows"]) == 19
    assert all(row["pareto_eligible"] is False for row in pareto["rows"])

    continual = manifest["continual_learning_hook"]
    assert continual["posterior_update_allowed"] is False
    assert "byte_closed_archive_sha256" in continual["required_before_update"]


def test_long_term_campaign_backfill_survives_autopilot_manifest_shape() -> None:
    manifest = build_integration_manifest()
    rows = {
        row["campaign_id"]: row
        for row in manifest["autopilot_dispatch_hook"]["rows"]
    }
    expected = {
        "c5_full_cooperative_receiver_substrate_campaign_20260514": ("4-8", [30.00, 50.00]),
        "c7_darts_supernet_architecture_search_campaign_20260514": ("6-12", [100.00, 300.00]),
        "c2_z7_mature_predictive_receiver_l5_campaign_20260514": ("8-12", [50.00, 100.00]),
        "c4_queued_architectural_moves_campaign_20260514": ("12-24", [50.00, 150.00]),
        "c3_multi_year_zen_floor_sub_005_campaign_20260514": ("52-156", [500.00, 2000.00]),
    }

    assert expected.keys() <= rows.keys()
    for campaign_id, (horizon, cost_band) in expected.items():
        row = rows[campaign_id]
        assert row["source_campaign_id"] == campaign_id
        assert row["lane_id"].startswith("lane_")
        assert row["lane_class"] == "substrate_engineering"
        assert row["expected_horizon_weeks"] == horizon
        assert row["estimated_cost_usd_band"] == cost_band
        assert row["cost_metadata"]["estimated_cost_usd_band"] == cost_band
        assert row["timeline_metadata"]["expected_horizon_weeks"] == horizon
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert "requires_exact_eval_readiness_gate" in row["dispatch_blockers"]
        assert "requires_lane_dispatch_claim_before_gpu_or_remote_eval" in row["dispatch_blockers"]


def test_integration_manifest_wires_xray_magic_codec_and_compiler_hooks() -> None:
    manifest = build_integration_manifest()

    xray_rows = manifest["xray_hook"]["cooperative_receiver_packet_grammars"]
    assert {
        "DFL1",
        "TT5L",
        "SBO1",
        "S2SB",
        "CMLR",
        "DPW1",
    }.issubset({row["magic_ascii"] for row in xray_rows})

    magic_codec_rows = manifest["magic_codec_hook"]["entries"]
    assert {"OWV2", "OWV3", "IMPS"}.issubset(
        {row["magic_ascii"] for row in magic_codec_rows}
    )

    compiler = manifest["packet_compiler_hook"]
    assert compiler["canonical_module"] == "tac.packet_compiler.deterministic_compiler"
    assert compiler["score_claim"] is False
    assert compiler["ready_for_exact_eval_dispatch"] is False
    contract = packetir_operation_set_bridge_contract()
    assert compiler["packetir_operation_set_bridge_contract"] == contract
    assert compiler["recommended_ir_schema"] == contract["recommended_ir_schema"]
    assert compiler["required_order"] == list(contract["required_order"])
    assert compiler["required_proofs"] == list(contract["required_proofs"])


def test_integration_writer_and_markdown_are_deterministic(tmp_path) -> None:
    json_path = tmp_path / "integration_manifest.json"
    md_path = tmp_path / "integration_manifest.md"

    manifest = write_integration_manifest(json_path, markdown_output=md_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert loaded == manifest
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown == render_markdown(manifest)
    assert "Cooperative-Receiver Solver Integration" in markdown
    assert "`TT5L`" in markdown
