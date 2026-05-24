# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES
from tac.optimization.byte_shaving_campaign import (
    COUPLED_OPERATION_SET_SCHEMA,
    INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER,
    INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY,
    INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND,
    INVERSE_ACTION_MATERIALIZATION_BRIDGE_SCHEMA,
    INVERSE_ACTION_WATER_BUCKET_PORTFOLIO_SCHEMA,
    SIGNAL_SURFACE_SCHEMA,
    ByteShavingCampaignError,
    build_byte_shaving_campaign_plan,
    build_inverse_action_materialization_bridge,
    build_signal_surface_from_candidate_queue,
    build_signal_surface_from_engineered_correction_targeting,
    build_signal_surface_from_inverse_action_functional,
    build_signal_surface_from_master_gradient_anchor,
    validate_signal_surface,
)
from tac.packet_compiler.deterministic_compiler import PACKET_IR_OPERATION_SET_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "plan_byte_shaving_campaign.py"


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _surface() -> dict[str, object]:
    return {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "post_training_byte_shave_seed0",
        "candidate_id": "boostnerv_seed0",
        "lane_id": "boostnerv_post_train_shaving",
        "combo_beam_width": 16,
        "max_combo_count": 16,
        "units": [
            {
                "unit_id": "pair0371",
                "unit_kind": "pair",
                "candidate_saved_bytes": 1000,
                "predicted_quality_score_cost": 0.00015,
                "confidence": 0.9,
                "operations": [
                    {
                        "operation_id": "drop_pair",
                        "operation_family": "drop_pair",
                        "candidate_saved_bytes": 1000,
                        "predicted_quality_score_cost": 0.00015,
                    },
                    {
                        "operation_id": "substitute_pair",
                        "operation_family": "substitute_pair",
                        "candidate_saved_bytes": 700,
                        "predicted_quality_score_cost": 0.00002,
                    },
                ],
            },
            {
                "unit_id": "byte_null_run_a",
                "unit_kind": "byte_range",
                "candidate_saved_bytes": 500,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.95,
                "operation_families": ["null_remove_or_seed"],
            },
            {
                "unit_id": "tensor_head7",
                "unit_kind": "tensor",
                "candidate_saved_bytes": 900,
                "predicted_quality_score_cost": 0.0002,
                "confidence": 0.8,
                "operation_families": ["quantize_tensor"],
            },
        ],
        "interactions": [
            {
                "interaction_id": "pair_null_synergy",
                "unit_ids": ["pair0371", "byte_null_run_a"],
                "extra_saved_bytes": 120,
                "delta_score": -0.00001,
                "rationale": "shared selector/header overhead disappears together",
            }
        ],
        "conflicts": [{"unit_ids": ["pair0371", "tensor_head7"]}],
        **_false_authority(),
    }


def _inverse_action_payload() -> dict[str, object]:
    return {
        "schema": "inverse_steganalysis_discrete_action_functional.v1",
        "tool": "tac.optimization.inverse_steganalysis_acquisition",
        "math_model": {
            "representation": "discrete_riemann_sum_with_second_order_interactions",
            "stationarity_rule": "select positive euler_lagrange_residual cells",
            "lambda_rate": 0.0000005,
        },
        "integral_totals": {"cell_count": 1, "blocked_cell_count": 0},
        "water_bucket": {
            "schema": "inverse_steganalysis_water_bucket_plan.v1",
            "selected_count": 1,
            "selected_expected_score_gain": 0.0004,
            "selected_cells": [
                {
                    "atom_id": "inverse_surface_pair0007",
                    "candidate_id": "candidate_pair0007",
                    "scope_axis": "pairs",
                    "component": "posenet",
                    "water_fill_cost_bytes": 32,
                    "expected_score_gain": 0.0004,
                    "euler_lagrange_residual": 0.00039,
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "cells": [],
        **_false_authority(),
    }


def test_plan_builds_combination_ladder_with_interactions_and_conflicts() -> None:
    surface = _surface()
    surface["inverse_scorer_surface_refs"] = [
        {
            "kind": "inverse_scorer_surface",
            "path": "experiments/results/inverse_surface.json",
            "sha256": "d" * 64,
        }
    ]
    plan = build_byte_shaving_campaign_plan(surface, max_k=3)
    combo = plan["recommended_combination"]

    assert plan["schema"] == "byte_shaving_campaign_plan.v1"
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert combo["selected_unit_ids"] == ["pair0371", "byte_null_run_a"]
    assert combo["candidate_saved_bytes"] == 1620
    assert combo["active_interactions"][0]["interaction_id"] == "pair_null_synergy"
    assert "tensor_head7" not in combo["selected_unit_ids"]
    assert combo["expected_delta_score"] == pytest.approx(-25.0 * 1620 / CONTEST_ORIGINAL_BYTES + 0.00015 - 0.00001)
    assert plan["search_space_policy"]["combination_search"] == ("bounded_beam_over_units_and_operation_alternatives")
    assert plan["search_space_policy"]["permutation_search"] == ("bounded_operation_order_permutations_for_top_combos")
    assert plan["search_space_policy"]["operation_set_search"] == (
        "durable_coupled_operation_sets_preserve_interactions_and_order_for_queueing"
    )
    assert "pair" in plan["search_space_policy"]["unit_layers"]
    assert "drop_pair" in plan["operation_order_priors"]
    assert plan["inverse_scorer_surface_refs"] == surface["inverse_scorer_surface_refs"]
    operation_set = plan["recommended_operation_set"]
    assert operation_set["schema"] == COUPLED_OPERATION_SET_SCHEMA
    assert operation_set["operation_set_id"] == f"opset_{combo['combo_id']}"
    assert operation_set["selected_unit_ids"] == combo["selected_unit_ids"]
    assert operation_set["active_interactions"] == combo["active_interactions"]
    assert operation_set["requires_atomic_materialization"] is True
    assert operation_set["chosen_operation_sequence_is_permutation"] is True
    assert len(operation_set["chosen_operation_sequence_sha256"]) == 64
    assert operation_set["partial_materialization_allowed"] is False
    assert operation_set["score_claim"] is False
    packet_ir = plan["packet_ir_operation_sets"][0]
    assert packet_ir["schema"] == PACKET_IR_OPERATION_SET_SCHEMA
    assert packet_ir["candidate_id"] == "boostnerv_seed0"
    assert packet_ir["lane_id"] == "boostnerv_post_train_shaving"
    assert packet_ir["source_operation_set_id"] == operation_set["operation_set_id"]
    assert packet_ir["compiler_contract"]["score_claim"] is False
    assert packet_ir["requires_atomic_materialization"] is True
    assert packet_ir["partial_materialization_allowed"] is False
    assert packet_ir["chosen_operation_sequence_sha256"] == operation_set[
        "chosen_operation_sequence_sha256"
    ]
    assert packet_ir["required_proofs_status"]["runtime_consumption_proof"] == (
        "missing"
    )
    assert packet_ir["operations"][0]["schema"] == "packet_ir_operation_v1"
    assert packet_ir["operations"][0]["compiler_phase"] == "pack"
    assert "packetir_operation_not_byte_closed:pair" in packet_ir["operations"][0][
        "blockers"
    ]
    assert "packetir_operation_set_requires_runtime_consumption_proof" in packet_ir[
        "blockers"
    ]
    assert packet_ir["score_claim"] is False


def test_plan_exposes_bounded_operation_permutation_ladder() -> None:
    surface = _surface()
    surface["operation_order_priors"] = {
        "null_remove_or_seed": 1,
        "drop_pair": 5,
    }
    plan = build_byte_shaving_campaign_plan(surface, max_k=3)
    permutation_row = plan["permutation_ladder"][0]
    best_order = permutation_row["permutations"][0]["operation_sequence"]

    assert permutation_row["schema"] == "byte_shaving_operation_permutation_row.v1"
    assert best_order[0]["operation_family"] == "null_remove_or_seed"
    assert best_order[1]["operation_family"] == "drop_pair"
    assert permutation_row["permutations"][0]["prior_order_inversion_count"] == 0
    assert permutation_row["score_claim"] is False
    operation_set = plan["operation_set_ladder"][0]
    assert operation_set["chosen_operation_sequence_source"] == (
        "bounded_permutation_ladder_rank_1"
    )
    assert operation_set["chosen_operation_sequence"][0]["operation_family"] == (
        "null_remove_or_seed"
    )
    assert operation_set["chosen_operation_sequence"][1]["operation_family"] == (
        "drop_pair"
    )


def test_operation_sequence_hash_includes_family_materializer_identity() -> None:
    surface = _surface()
    surface["units"][0]["operations"][0]["representation_family_class"] = (
        "hnerv_variant"
    )
    first = build_byte_shaving_campaign_plan(surface, max_k=3)
    first_set = first["operation_set_ladder"][0]

    surface["units"][0]["operations"][0]["representation_family_class"] = "non_nerv"
    second = build_byte_shaving_campaign_plan(surface, max_k=3)
    second_set = second["operation_set_ladder"][0]

    assert first_set["chosen_operation_sequence"][0][
        "representation_family_class"
    ] == "hnerv_variant"
    assert second_set["chosen_operation_sequence"][0][
        "representation_family_class"
    ] == "non_nerv"
    assert first_set["chosen_operation_sequence_sha256"] != second_set[
        "chosen_operation_sequence_sha256"
    ]


def test_prefix_ladder_marks_conflicting_prefixes_and_does_not_recommend_them() -> None:
    plan = build_byte_shaving_campaign_plan(_surface(), max_k=3)
    conflicting = next(row for row in plan["sweep_ladder"] if row["sweep_id"] == "top_0003")

    assert conflicting["conflict_violations"] == [["pair0371", "tensor_head7"]]
    assert "prefix_selection_violates_conflict_sets" in conflicting["dispatch_blockers"]
    assert plan["recommended_prefix"]["sweep_id"] != "top_0003"


def test_plan_recommends_operation_alternative_when_drop_cost_is_too_high() -> None:
    surface = _surface()
    surface["units"][0]["operations"][0]["predicted_quality_score_cost"] = 0.005
    plan = build_byte_shaving_campaign_plan(surface, max_k=3)
    pair = next(row for row in plan["ranked_units"] if row["unit_id"] == "pair0371")

    assert pair["recommended_operation_family"] == "substitute_pair"
    assert pair["candidate_saved_bytes"] == 700


def test_signal_surface_rejects_truthy_authority() -> None:
    surface = _surface()
    surface["score_claim"] = True

    with pytest.raises(ByteShavingCampaignError, match="score_claim"):
        validate_signal_surface(surface)


def test_candidate_queue_surface_preserves_calibration_and_rejects_authority() -> None:
    queue = {
        "schema": "optimizer_candidate_queue_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "top_k": [
            {
                "candidate_id": "drop_bytes",
                "source_candidate_id": "trained_seed7",
                "unit_kind": "byte_range",
                "operation_family": "entropy_recode",
                "target_kind": "byte_range_entropy_coder_v1",
                "materializer": "byte_range_entropy_coder_adapter",
                "operation_params": {"codec": "range"},
                "candidate_saved_bytes": 101,
                "predicted_quality_score_cost": 0.00001,
                "confidence": 0.7,
                "evidence_grade": "[macOS-MLX research-signal]",
                "evidence_semantics": "strict_calibrated_local_spend_triage",
                "source_paths": ["experiments/results/seed7/manifest.json"],
                "candidate_archive_sha256": "c" * 64,
                "candidate_archive_bytes": 178600,
                "local_axis": "macOS-MLX",
                "target_axis": "contest-CPU",
                "projected_contest_score": 0.19203,
                "master_gradient_provenance": {"anchor_count": 1},
                "canonical_equation_provenance": {"equation_id": "fixture_v1"},
                "atom_ids": ["atom_1"],
                "dispatch_blockers": ["needs_materializer"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
    }

    surface = build_signal_surface_from_candidate_queue(queue)
    unit = surface["units"][0]
    plan = build_byte_shaving_campaign_plan(surface)
    ranked = plan["ranked_units"][0]

    assert unit["evidence_grade"] == "[macOS-MLX research-signal]"
    assert unit["candidate_archive_sha256"] == "c" * 64
    assert ranked["source_candidate_id"] == "trained_seed7"
    assert ranked["master_gradient_signal"] == {"anchor_count": 1}
    assert ranked["canonical_equation_provenance"] == {"equation_id": "fixture_v1"}
    assert ranked["atom_ids"] == ["atom_1"]
    assert ranked["recommended_operation_family"] == "entropy_recode"
    assert ranked["recommended_operation_materializer"] == "byte_range_entropy_coder_adapter"
    assert ranked["recommended_operation_target_kind"] == "byte_range_entropy_coder_v1"
    assert ranked["recommended_operation_params"] == {"codec": "range"}
    selected = plan["recommended_prefix"]["selected_operations"][0]
    assert selected["materializer"] == "byte_range_entropy_coder_adapter"
    assert selected["target_kind"] == "byte_range_entropy_coder_v1"
    assert selected["params"] == {"codec": "range"}

    queue["top_k"][0]["score_claim"] = True
    with pytest.raises(ByteShavingCampaignError, match="score_claim"):
        build_signal_surface_from_candidate_queue(queue)


def test_inverse_action_functional_converts_to_plannable_surface() -> None:
    surface = build_signal_surface_from_inverse_action_functional(_inverse_action_payload())
    plan = build_byte_shaving_campaign_plan(surface)
    ranked = plan["ranked_units"][0]

    assert surface["units"][0]["unit_kind"] == "scorer_inverse_surface_cell"
    assert surface["units"][0]["candidate_saved_bytes"] == 0
    assert surface["water_bucket_materialization_portfolio"]["schema"] == (
        INVERSE_ACTION_WATER_BUCKET_PORTFOLIO_SCHEMA
    )
    assert surface["water_bucket_materialization_portfolio"]["rows"][0][
        "actuation_mode"
    ] == "high_level_operation_compiler_required"
    assert ranked["expected_delta_score"] == pytest.approx(-0.0004)
    assert ranked["recommended_operation_family"] == (
        INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY
    )
    assert ranked["recommended_operation_materializer"] == (
        INVERSE_ACTION_HIGH_LEVEL_MATERIALIZER
    )
    assert ranked["recommended_operation_target_kind"] == (
        INVERSE_ACTION_HIGH_LEVEL_TARGET_KIND
    )
    assert plan["inverse_action_materialization_portfolios"][0]["actuation_modes"] == [
        "high_level_operation_compiler_required"
    ]
    assert plan["recommended_prefix"]["selected_unit_ids"] == ["inverse_action_inverse_surface_pair0007"]
    assert plan["packet_ir_operation_sets"] == []
    assert plan["score_claim"] is False
    bridge = build_inverse_action_materialization_bridge(plan)
    assert bridge["schema"] == INVERSE_ACTION_MATERIALIZATION_BRIDGE_SCHEMA
    assert bridge["portfolio_count"] == 1
    assert bridge["portfolio_row_count"] == 1
    assert bridge["queue_consumable_portfolio_row_count"] == 0
    assert bridge["queue_consumable_packet_ir_operation_set_count"] == 0
    assert bridge["queue_consumable_packet_ir_operation_set_ids"] == []
    assert bridge["high_level_operation_compiler_required_count"] == 1
    assert bridge["source_provenance_operation_set_count"] == 0
    assert bridge["packet_ir_operation_set_count"] == 0
    assert bridge["portfolio_row_bridge_links"][0]["queue_consumable"] is False
    assert bridge["portfolio_row_bridge_links"][0][
        "matched_packet_ir_operation_set_ids"
    ] == []
    assert (
        "inverse_action_operation_set_compiler_required"
        in bridge["portfolio_row_bridge_links"][0]["blockers"]
    )
    assert bridge["queue_consumption"]["next_gate"] == (
        "inverse_action_operation_set_compiler"
    )
    assert (
        "inverse_action_operation_set_compiler_required_for_cells_without_source_provenance"
        in bridge["dispatch_blockers"]
    )
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False


def test_inverse_action_functional_leaf_cells_are_explicit_opt_in() -> None:
    surface = build_signal_surface_from_inverse_action_functional(
        _inverse_action_payload(),
        allow_leaf_cell_candidates=True,
    )
    plan = build_byte_shaving_campaign_plan(surface)
    ranked = plan["ranked_units"][0]

    assert surface["water_bucket_materialization_portfolio"]["rows"][0][
        "actuation_mode"
    ] == "leaf_cell_candidate_explicit_opt_in"
    assert ranked["recommended_operation_family"] == (
        "materialize_inverse_scorer_cell_candidate"
    )
    assert ranked["recommended_operation_materializer"] == (
        "inverse_scorer_cell_candidate_adapter"
    )
    assert ranked["recommended_operation_target_kind"] == (
        "inverse_scorer_cell_candidate_v1"
    )


def test_inverse_action_functional_rehydrates_family_operations_from_provenance() -> None:
    payload = _inverse_action_payload()
    payload["cells"] = [
        {
            "atom_id": "mlx_family_opset",
            "source_provenance": {
                "schema": (
                    "inverse_steganalysis_mlx_acquisition_batch_operation_set_provenance.v1"
                ),
                "operation_set_id": "mlx_family_set",
                "candidate_saved_bytes": 900,
                "source_family_classes": [
                    "hnerv_variant",
                    "boostnerv_bolton",
                    "non_nerv",
                ],
                "receiver_contract_kinds": [
                    "family_agnostic_hnerv_variant_mlx_candidate_receiver",
                    "family_agnostic_boostnerv_bolton_mlx_candidate_receiver",
                    "family_agnostic_non_nerv_mlx_candidate_receiver",
                ],
                "operation_portability": "family_agnostic",
                "selected_operations": [
                    {
                        "unit_id": "hnerv_section",
                        "unit_kind": "archive_section",
                        "operation_id": "hnerv_recode",
                        "operation_family": "section_entropy_recode",
                        "target_kind": "archive_section_entropy_recode_v1",
                        "candidate_saved_bytes": 300,
                        "predicted_quality_score_delta": -0.00005,
                        "representation_family_class": "hnerv_variant",
                        "receiver_contract_kind": (
                            "family_agnostic_hnerv_variant_mlx_candidate_receiver"
                        ),
                    },
                    {
                        "unit_id": "boost_tensor",
                        "unit_kind": "tensor",
                        "operation_id": "boost_factorize",
                        "operation_family": "factorize_tensor",
                        "target_kind": "tensor_factorize_v1",
                        "candidate_saved_bytes": 300,
                        "predicted_quality_score_delta": -0.00005,
                        "representation_family_class": "boostnerv_bolton",
                        "bolt_on_families": ["boostnerv"],
                    },
                    {
                        "unit_id": "packet_member",
                        "unit_kind": "packet_member",
                        "operation_id": "packet_recompress",
                        "operation_family": "member_recompress",
                        "target_kind": "packet_member_recompress_v1",
                        "candidate_saved_bytes": 300,
                        "predicted_quality_score_delta": -0.00005,
                        "representation_family_class": "non_nerv",
                    },
                ],
            },
        }
    ]
    payload["water_bucket"]["selected_cells"][0]["atom_id"] = "mlx_family_opset"

    surface = build_signal_surface_from_inverse_action_functional(payload)
    plan = build_byte_shaving_campaign_plan(surface, max_k=3)
    op_set = plan["operation_set_ladder"][0]
    packet_ir = plan["packet_ir_operation_sets"][0]

    assert [unit["unit_kind"] for unit in surface["units"]] == [
        "archive_section",
        "tensor",
        "packet_member",
    ]
    assert {
        operation["representation_family_class"]
        for operation in op_set["selected_operations"]
    } == {"hnerv_variant", "boostnerv_bolton", "non_nerv"}
    assert "section_entropy_recode" in op_set["operation_families"]
    assert "factorize_tensor" in op_set["operation_families"]
    assert "member_recompress" in op_set["operation_families"]
    assert op_set["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert packet_ir["schema"] == PACKET_IR_OPERATION_SET_SCHEMA
    assert packet_ir["source_portfolio_schema"] == (
        INVERSE_ACTION_WATER_BUCKET_PORTFOLIO_SCHEMA
    )
    assert packet_ir["byte_closed_operation_count"] == 3
    assert sorted(
        operation["compiler_phase"]
        for operation in packet_ir["operations"]
    ) == ["arithmetic", "arithmetic", "representation"]
    assert {
        operation["representation_family_class"]
        for operation in packet_ir["operations"]
    } == {"hnerv_variant", "boostnerv_bolton", "non_nerv"}
    assert all(
        "packetir_operation_not_byte_closed" not in "\n".join(operation["blockers"])
        for operation in packet_ir["operations"]
    )
    assert "packetir_operation_set_requires_materializer_contexts" in packet_ir[
        "blockers"
    ]
    assert packet_ir["ready_for_exact_eval_dispatch"] is False
    bridge = build_inverse_action_materialization_bridge(plan)
    assert bridge["schema"] == INVERSE_ACTION_MATERIALIZATION_BRIDGE_SCHEMA
    assert bridge["source_provenance_operation_set_count"] == 1
    assert bridge["high_level_operation_compiler_required_count"] == 0
    assert bridge["packet_ir_operation_set_count"] == len(
        plan["packet_ir_operation_sets"]
    )
    assert bridge["packet_ir_byte_closed_operation_count"] >= 3
    assert bridge["queue_consumable_portfolio_row_count"] == 1
    assert bridge["queue_consumable_packet_ir_operation_set_count"] == 1
    assert bridge["queue_consumable_packet_ir_operation_set_ids"] == [
        packet_ir["operation_set_id"]
    ]
    assert bridge["queue_consumption"]["next_gate"] == (
        "build_byte_shaving_campaign_queue_packet_ir_lowering"
    )
    assert bridge["portfolio_row_bridge_links"][0]["queue_consumable"] is True
    assert bridge["portfolio_row_bridge_links"][0][
        "matched_packet_ir_operation_set_ids"
    ] == [packet_ir["operation_set_id"]]
    assert bridge["portfolio_row_bridge_links"][0][
        "matched_source_operation_set_ids"
    ] == [op_set["operation_set_id"]]
    assert set(bridge["representation_family_classes"]) == {
        "hnerv_variant",
        "boostnerv_bolton",
        "non_nerv",
    }
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False


def test_inverse_action_compiler_hint_lowers_to_family_packet_ir() -> None:
    payload = _inverse_action_payload()
    payload["cells"] = [
        {
            "atom_id": "compiled_family_opset",
            "operation_set_compiler": {
                "schema": "inverse_action_operation_set_compiler_hint.v1",
                "operation_set_id": "compiled_family_set",
                "candidate_saved_bytes": 900,
                "operation_portability": "family_agnostic",
                "selected_operations": [
                    {
                        "unit_id": "compiled_hnerv_section",
                        "target_kind": "archive_section_entropy_recode_v1",
                        "archive_section": "decoder_blob",
                        "candidate_saved_bytes": 300,
                        "representation_family_class": "hnerv_variant",
                    },
                    {
                        "unit_id": "compiled_boost_tensor",
                        "target_kind": "tensor_factorize_v1",
                        "tensor_name": "boost.overlay",
                        "candidate_saved_bytes": 300,
                        "representation_family_class": "boostnerv_bolton",
                    },
                    {
                        "unit_id": "compiled_packet_member",
                        "target_kind": "packet_member_recompress_v1",
                        "member_name": "0.bin",
                        "candidate_saved_bytes": 300,
                        "representation_family_class": "non_nerv",
                    },
                ],
            },
        }
    ]
    payload["water_bucket"]["selected_cells"][0]["atom_id"] = "compiled_family_opset"

    surface = build_signal_surface_from_inverse_action_functional(payload)
    plan = build_byte_shaving_campaign_plan(surface, max_k=3)
    packet_ir = plan["packet_ir_operation_sets"][0]
    bridge = build_inverse_action_materialization_bridge(plan)

    assert [unit["unit_kind"] for unit in surface["units"]] == [
        "archive_section",
        "tensor",
        "packet_member",
    ]
    assert surface["water_bucket_materialization_portfolio"]["actuation_modes"] == [
        "compiled_operation_set"
    ]
    assert surface["source_signal_refs"][0]["compiled_operation_set_count"] == 1
    assert packet_ir["byte_closed_operation_count"] == 3
    assert {
        operation["target_kind"] for operation in packet_ir["operations"]
    } == {
        "archive_section_entropy_recode_v1",
        "tensor_factorize_v1",
        "packet_member_recompress_v1",
    }
    archive_operation = next(
        operation
        for operation in packet_ir["operations"]
        if operation["target_kind"] == "archive_section_entropy_recode_v1"
    )
    assert archive_operation["params"]["archive_section"] == "decoder_blob"
    assert bridge["compiled_operation_set_count"] == 1
    assert bridge["high_level_operation_compiler_required_count"] == 0
    assert bridge["queue_consumable_portfolio_row_count"] == 1
    assert bridge["queue_consumable_packet_ir_operation_set_ids"] == [
        packet_ir["operation_set_id"]
    ]
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False


def test_mlx_placeholder_provenance_defers_to_compiler_hint() -> None:
    payload = _inverse_action_payload()
    payload["cells"] = [
        {
            "atom_id": "mlx_compiler_handoff",
            "source_provenance": {
                "schema": (
                    "inverse_steganalysis_mlx_acquisition_batch_operation_set_provenance.v1"
                ),
                "operation_set_id": "mlx_placeholder_set",
                "candidate_id": "mlx_candidate",
                "candidate_saved_bytes": 0,
                "selected_operations": [
                    {
                        "unit_id": "mlx_response_row",
                        "unit_kind": "scorer_response_row",
                        "operation_family": "materialize_scorer_response_candidate",
                        "target_kind": "mlx_scorer_response_candidate_v1",
                        "candidate_saved_bytes": 0,
                    }
                ],
            },
            "operation_set_compiler": {
                "schema": "inverse_action_operation_set_compiler_hint.v1",
                "operation_set_id": "compiled_from_mlx_hint",
                "candidate_saved_bytes": 384,
                "operation_portability": "family_agnostic",
                "selected_operations": [
                    {
                        "unit_id": "compiled_decoder_blob",
                        "target_kind": "archive_section_entropy_recode_v1",
                        "archive_section": "decoder_blob",
                        "candidate_saved_bytes": 256,
                        "representation_family_class": "hnerv_variant",
                    },
                    {
                        "unit_id": "compiled_packet_member",
                        "target_kind": "packet_member_recompress_v1",
                        "member_name": "0.bin",
                        "candidate_saved_bytes": 128,
                        "representation_family_class": "non_nerv",
                    },
                ],
            },
        }
    ]
    payload["water_bucket"]["selected_cells"][0]["atom_id"] = "mlx_compiler_handoff"

    surface = build_signal_surface_from_inverse_action_functional(payload)
    plan = build_byte_shaving_campaign_plan(surface, max_k=2)
    bridge = build_inverse_action_materialization_bridge(plan)

    assert surface["water_bucket_materialization_portfolio"]["actuation_modes"] == [
        "compiled_operation_set"
    ]
    assert [unit["unit_kind"] for unit in surface["units"]] == [
        "archive_section",
        "packet_member",
    ]
    assert bridge["source_provenance_operation_set_count"] == 0
    assert bridge["compiled_operation_set_count"] == 1
    assert bridge["high_level_operation_compiler_required_count"] == 0
    assert bridge["queue_consumable_packet_ir_operation_set_count"] == 1
    assert {
        operation["target_kind"]
        for operation in plan["packet_ir_operation_sets"][0]["operations"]
    } == {"archive_section_entropy_recode_v1", "packet_member_recompress_v1"}
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False


def test_inverse_action_compiler_hint_unsupported_target_fails_closed() -> None:
    payload = _inverse_action_payload()
    payload["cells"] = [
        {
            "atom_id": "unsupported_compiler",
            "operation_set_compiler": {
                "schema": "inverse_action_operation_set_compiler_hint.v1",
                "target_kind": "unsupported_family_operation_v1",
            },
        }
    ]
    payload["water_bucket"]["selected_cells"][0]["atom_id"] = "unsupported_compiler"

    surface = build_signal_surface_from_inverse_action_functional(payload)
    plan = build_byte_shaving_campaign_plan(surface)
    bridge = build_inverse_action_materialization_bridge(plan)

    assert surface["water_bucket_materialization_portfolio"]["actuation_modes"] == [
        "high_level_operation_compiler_required"
    ]
    assert bridge["compiled_operation_set_count"] == 0
    assert bridge["high_level_operation_compiler_required_count"] == 1
    assert bridge["queue_consumption"]["next_gate"] == (
        "inverse_action_operation_set_compiler"
    )


def test_inverse_action_units_compose_with_non_inverse_combination_ladder() -> None:
    surface = _surface()
    inverse_surface = build_signal_surface_from_inverse_action_functional(
        _inverse_action_payload()
    )
    surface["units"].extend(inverse_surface["units"])
    surface["source_signal_refs"] = inverse_surface["source_signal_refs"]

    plan = build_byte_shaving_campaign_plan(surface, max_k=4)
    combo_ids = set(plan["recommended_combination"]["selected_unit_ids"])

    assert "inverse_action_inverse_surface_pair0007" in combo_ids
    assert "pair0371" in combo_ids
    assert "byte_null_run_a" in combo_ids
    assert plan["recommended_combination"]["score_claim"] is False
    assert plan["score_claim"] is False


def test_engineered_correction_targeting_subsumes_legacy_sidecar() -> None:
    sidecar = {
        "schema": "master_gradient_consumer_engineered_correction_targeting_v1",
        "consumer_id": "engineered_correction_targeting",
        "archive_sha256": "a" * 64,
        "measurement_axis": "contest_cuda",
        "measurement_hardware": "linux_x86_64_t4",
        "n_bytes": 128,
        "n_pairs": 2,
        "targets_per_pair": 1,
        "total_targets": 2,
        "top_per_pair_targets": [
            {
                "pair_index": 0,
                "byte_index": 7,
                "per_pair_distortion_magnitude": 0.25,
                "per_pair_variance_rank": 3,
            },
            {
                "pair_index": 1,
                "byte_index": 11,
                "per_pair_distortion_magnitude": 0.5,
                "per_pair_variance_rank": 1,
            },
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    surface = build_signal_surface_from_engineered_correction_targeting(
        sidecar,
        default_predicted_quality_score_delta=-0.0002,
    )
    plan = build_byte_shaving_campaign_plan(surface)
    ranked = plan["ranked_units"][0]

    assert surface["units"][0]["unit_kind"] == "correction_target"
    assert ranked["recommended_operation_family"] == "apply_engineered_correction"
    assert ranked["candidate_saved_bytes"] == 0
    assert ranked["expected_delta_score"] == pytest.approx(-0.0002)
    assert ranked["engineered_correction_signal"]["byte_index"] == 7
    assert plan["score_claim"] is False


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "surface.json"
    output = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    source.write_text(json.dumps(_surface()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source",
            str(source),
            "--output",
            str(output),
            "--md-out",
            str(md_out),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "combinations=" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["combination_ladder"]
    assert "Recommended Combination" in md_out.read_text(encoding="utf-8")


def test_cli_can_plan_from_inverse_action_functional(tmp_path: Path) -> None:
    source = tmp_path / "inverse_action.json"
    output = tmp_path / "plan.json"
    bridge_out = tmp_path / "bridge.json"
    source.write_text(json.dumps(_inverse_action_payload()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source",
            str(source),
            "--from-inverse-action-functional",
            "--output",
            str(output),
            "--inverse-action-materialization-bridge-out",
            str(bridge_out),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "byte_shaving_campaign_plan.v1"
    assert payload["ranked_units"][0]["recommended_operation_family"] == (
        INVERSE_ACTION_HIGH_LEVEL_OPERATION_FAMILY
    )
    assert payload["inverse_action_materialization_portfolios"][0][
        "actuation_modes"
    ] == ["high_level_operation_compiler_required"]
    assert payload["score_claim"] is False
    bridge = json.loads(bridge_out.read_text(encoding="utf-8"))
    assert bridge["schema"] == INVERSE_ACTION_MATERIALIZATION_BRIDGE_SCHEMA
    assert bridge["high_level_operation_compiler_required_count"] == 1
    assert bridge["packet_ir_operation_set_count"] == 0
    assert bridge["portfolio_row_bridge_links"][0]["queue_consumable"] is False
    assert bridge["score_claim"] is False


def test_master_gradient_anchor_builds_planning_only_byte_surface(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    repo = tmp_path
    state = repo / ".omx" / "state"
    state.mkdir(parents=True)
    archive_sha = "a" * 64
    gradient_path = state / "mg.npy"
    np.save(
        gradient_path,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    ledger = state / "master_gradient_anchors.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "master_gradient_anchor_v1",
                "archive_sha256": archive_sha,
                "gradient_array_path": ".omx/state/mg.npy",
                "gradient_tensor_kind": "aggregate_per_byte_v1",
                "measurement_axis": "[macOS-CPU advisory]",
                "measurement_hardware": "darwin_arm64_local_cpu_advisory",
                "measurement_call_id": "local-test",
                "measurement_utc": "2026-05-23T00:00:00Z",
                "n_bytes": 5,
                "n_pairs_used": 1,
                "n_pairs_total": 5,
                "scored_archive_sha256": archive_sha,
                "scored_archive_bytes": 123,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    surface = build_signal_surface_from_master_gradient_anchor(
        archive_sha256=archive_sha,
        repo_root=repo,
        low_sensitivity_quantile=0.8,
        max_units=8,
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=repo)

    assert surface["schema"] == SIGNAL_SURFACE_SCHEMA
    assert surface["score_claim"] is False
    assert len(surface["units"]) == 2
    assert surface["units"][0]["source_span"] == {"start": 0, "end_exclusive": 2}
    assert surface["units"][0]["master_gradient_signal"]["score_claim"] is False
    assert plan["recommended_combination"]["selected_unit_ids"] == [
        "mg_byte_span_0000000_0000002",
        "mg_byte_span_0000003_0000005",
    ]


def test_cli_can_plan_from_master_gradient_anchor(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    archive_sha = "b" * 64
    np.save(
        state / "mg.npy",
        np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
    )
    (state / "master_gradient_anchors.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "master_gradient_anchor_v1",
                "archive_sha256": archive_sha,
                "gradient_array_path": ".omx/state/mg.npy",
                "gradient_tensor_kind": "aggregate_per_byte_v1",
                "measurement_axis": "[macOS-CPU advisory]",
                "measurement_hardware": "darwin_arm64_local_cpu_advisory",
                "measurement_call_id": "local-test",
                "measurement_utc": "2026-05-23T00:00:00Z",
                "n_bytes": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--from-master-gradient-archive-sha",
            archive_sha,
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
            "--master-gradient-low-quantile",
            "0.67",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "byte_shaving_campaign_plan.v1"
    assert payload["ranked_units"]
    assert payload["score_claim"] is False
