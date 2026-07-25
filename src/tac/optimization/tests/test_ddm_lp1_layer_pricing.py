from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tac.optimization.ddm_lp1_layer_pricing import (
    LayerPricingError,
    build_receipt,
)

REPO = Path(__file__).resolve().parents[4]
CONFIG = REPO / ".omx/research/configs/ddm_lp1_layer_pricing_20260725.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_materialized_sources_produce_complete_corrected_budget_and_sense() -> None:
    result = build_receipt(_config(), config_path=CONFIG)
    waterfill = result["c1_corrected_waterfill"]
    assert waterfill["source_exact_control_subtotal_bytes"] == 133_941
    assert waterfill["corrected_measured_allocated_bytes"] == 134_211
    assert waterfill["source_planning_reserve_bytes"] == 65_789
    assert waterfill["unallocated_headroom_bytes"] == 65_789
    assert waterfill["g4_savings_applied_to_current_c1_bytes"] == 0
    assert len(waterfill["rows"]) == 13
    assert {row["stream"] for row in waterfill["rows"]} == {
        "v15_predictor_zip_outer_home",
        "g1_movable_worldsheet_outer_home",
        "receiver_realization_profile",
        "solved_template_outer_home",
        "manifest",
        "central_directory_and_eocd",
        "v15_exact_control_subtotal",
        "lane_program_seed",
        "contextual_bounded_collateral_shared_application_stage_reserve",
        "v18b_first_exact_pricing_rung_reserve",
        "j3_finish_and_xi_refinement_reserve",
        "final_coder_and_container_contingency",
        "hard_total",
    }
    assert result["costate_sense"]["row_count"] == 25
    assert result["costate_sense"]["boundary_rows"] == 16
    assert result["costate_sense"]["cell_rows"] == 9
    assert result["costate_sense"]["zero_allocation_rows"] == 25


def test_context_races_preserve_same_object_scope() -> None:
    result = build_receipt(_config(), config_path=CONFIG)
    rows = {row["race_id"]: row for row in result["context_keep_drop_rows"]}
    assert rows["g4_aggregate_pixel_time_order"]["savings_bytes"] == 89_161
    assert rows["g4_aggregate_pixel_time_order"]["disposition"] == "KEEP_CONTEXT"
    assert rows["g4_predictor_boundary_distance"]["savings_bytes"] == -192_417
    assert rows["g4_predictor_boundary_distance"]["disposition"] == "DROP_CONTEXT"
    assert rows["dm1_joint_shared_semantic_container"]["savings_bytes"] == 2_555
    assert result["cc2_coordination"]["lp1_selected_new_codec"] is False


def test_all_semantic_rows_are_deepest_l4_but_not_waterfilled() -> None:
    result = build_receipt(_config(), config_path=CONFIG)
    rows = result["costate_sense"]["rows"]
    assert all(row["typed_home"]["layer_home"] == "L4_scorer_feature" for row in rows)
    assert {row["typed_home"]["type"] for row in rows} == {"FIBER", "RESIDUAL"}
    assert all(row["l4_through_l3_survival"] == "MEASURED_EXACT" for row in rows)
    assert all(row["mass_pays_measured_reach"] is False for row in rows)
    assert all(row["waterfill_allocation_bytes"] == 0 for row in rows)
    assert all(row["g4_same_object_context_price_bytes"] is None for row in rows)


def test_source_hash_and_authority_drift_fail_closed() -> None:
    invalid_hash = _config()
    invalid_hash["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(LayerPricingError, match="SHA-256 differs"):
        build_receipt(invalid_hash, config_path=CONFIG)

    invalid_authority = copy.deepcopy(_config())
    invalid_authority["score_claim"] = True
    with pytest.raises(LayerPricingError, match="score_claim"):
        build_receipt(invalid_authority, config_path=CONFIG)
