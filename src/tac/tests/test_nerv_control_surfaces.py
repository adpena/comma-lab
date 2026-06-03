# SPDX-License-Identifier: MIT
"""Tests for NeRV modelsize/control/sweep/master-consumer surfaces."""

from __future__ import annotations

from pathlib import Path

from tac.analysis.nerv_control_inventory import (
    NERV_CONTROL_INVENTORY_SCHEMA,
    build_nerv_control_inventory,
)
from tac.analysis.nerv_implementation_design_sweep import (
    STACK_REQUIREMENTS,
    build_nerv_implementation_design_sweep,
)
from tac.analysis.nerv_master_consumer_bridge import (
    build_nerv_master_consumer_bridge,
)
from tac.analysis.nerv_modelsize_archive_curve import (
    build_modelsize_archive_curve,
    parse_byte_caps,
)
from tac.analysis.nerv_rate_allocator_bridge import build_nerv_rate_allocator_bridge
from tac.analysis.nerv_rate_allocator_queue import build_nerv_rate_allocator_work_queue
from tac.analysis.nerv_source_parity_contract import build_nerv_source_parity_contract
from tac.cathedral.consumer_contract import validate_consumer_module
from tac.cathedral_consumers import nerv_top_priority_stack_consumer


def test_modelsize_curve_is_lower_bound_and_handles_too_small_caps() -> None:
    payload = build_modelsize_archive_curve(
        byte_caps=(36_000, 178_417),
        resolution_modes={
            "scorer_internal_384x512": 384 * 512,
            "contest_output_1164x874": 1164 * 874,
        },
    )

    assert payload["schema"] == "nerv_modelsize_archive_curve.v1"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert len(payload["curve_rows"]) == 8
    by_key = {
        (
            row["resolution_mode"],
            row["solved_budget"]["family"],
            row["target_archive_byte_cap"],
        ): row["solved_budget"]
        for row in payload["curve_rows"]
    }
    small_full_hnerv = by_key[("contest_output_1164x874", "hnerv", 36_000)]
    assert "target_byte_cap_below_minimum_solved_budget" in small_full_hnerv[
        "blockers"
    ]
    assert small_full_hnerv["target_fit"]["slack_bytes"] < 0
    scorer_snerv = by_key[("scorer_internal_384x512", "snerv", 178_417)]
    assert scorer_snerv["derived"]["fc_dim"] > 0
    assert scorer_snerv["ideal_quant_payload"]["measured"] is False


def test_parse_byte_caps_supports_repeated_and_comma_separated_values() -> None:
    assert parse_byte_caps(["36_000,72000", "178417"]) == (
        36_000,
        72_000,
        178_417,
    )


def test_control_inventory_routes_existing_hooks_without_authority(tmp_path: Path) -> None:
    root = _minimal_repo_root(tmp_path)
    payload = build_nerv_control_inventory(repo_root=root)

    assert payload["schema"] == NERV_CONTROL_INVENTORY_SCHEMA
    assert payload["focus_families"] == ["hi_nerv", "snerv"]
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    control_ids = {row["control_id"] for row in payload["control_rows"]}
    assert {"snerv_frequency_split", "hi_nerv_hierarchical_capacity"} <= control_ids
    assert payload["binding_gap_rows"]
    assert "xray_and_master_gradient" in payload["local_binding_surfaces"]
    assert payload["implementation_sweep"]["score_claim"] is False


def test_implementation_design_sweep_passes_synthetic_full_stack(tmp_path: Path) -> None:
    repo_root = _minimal_repo_root(tmp_path / "repo")
    oss_root = tmp_path / "oss"
    _write_official_sources(oss_root)
    _write_local_sources(repo_root)
    _write(repo_root / ".omx/research/codex_findings_snerv_test.md", "snerv memo")
    _write(repo_root / ".omx/research/hinerv_stack_design.json", "{}")
    proof_refs = {
        stack_id: tuple(req["promotion_proofs"])
        for stack_id, req in STACK_REQUIREMENTS.items()
    }

    payload = build_nerv_implementation_design_sweep(
        repo_root=repo_root,
        oss_audit_root=oss_root,
        proof_refs=proof_refs,
        generated_utc="2026-06-02T12:00:00+00:00",
    )

    assert payload["schema"] == "nerv_implementation_design_sweep.v1"
    assert payload["production_hardened_claim"] is False
    rows = {row["stack_id"]: row for row in payload["stack_sweeps"]}
    assert rows["snerv"]["production_blockers"] == []
    assert rows["hinerv"]["production_blockers"] == []
    assert payload["related_omx_design_memo_count"] == 2
    memo_paths = {row["path"] for row in payload["related_omx_design_memo_refs"]}
    assert ".omx/research/codex_findings_snerv_test.md" in memo_paths
    assert ".omx/research/hinerv_stack_design.json" in memo_paths
    assert payload["blockers"] == [
        "PR95_same_axis_control_replay_required_before_beat_claim",
        "PR101_and_Z5_nonterminal_block_new_exact_or_full_video_cuda",
    ]


def test_implementation_design_sweep_fails_closed_on_missing_proofs(
    tmp_path: Path,
) -> None:
    repo_root = _minimal_repo_root(tmp_path / "repo")
    oss_root = tmp_path / "oss"
    _write_official_sources(oss_root)
    _write_local_sources(repo_root)

    payload = build_nerv_implementation_design_sweep(
        repo_root=repo_root,
        oss_audit_root=oss_root,
        proof_refs={},
    )

    rows = {row["stack_id"]: row for row in payload["stack_sweeps"]}
    assert rows["snerv"]["verdict"].startswith("NO_GO_PRODUCTION_HARDENED")
    assert "snerv_proof_missing:official_forward_parity" in rows["snerv"][
        "production_blockers"
    ]
    assert "hinerv_proof_missing:prune_quant_codec_roundtrip" in rows["hinerv"][
        "production_blockers"
    ]
    assert payload["promotion_eligible"] is False


def test_master_consumer_bridge_and_cathedral_consumer_are_no_authority() -> None:
    seam = {
        "schema": "nerv_top_priority_stack_seam.v1",
        "axis_tag": "[planning/control]",
        "go_no_go_verdict": "GO_LOCAL_STACK_OPTIMIZATION__NO_GO",
        "top_priority_carriers": ["snerv", "hinerv"],
        "baseline_to_beat": "pr95_hnerv_muon",
        "blockers": ["snerv_proof_missing:full600_byte_closed_receiver_proof"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    control_inventory = {
        "schema": "nerv_control_inventory.v1",
        "axis_tag": "[planning/control]",
        "verdict": "GO_LOCAL_CONTROL_BINDING__NO_GO",
        "control_rows": [
            {
                "control_id": "bitmask_and_zero_packing",
                "applies_to": "cross_stack",
                "binding_status": "wired",
                "missing_bindings": [],
            }
        ],
        "binding_gap_rows": [],
        "local_binding_surfaces": {
            "section_value_and_codebook": [
                "tools/profile_pact_nerv_selector_v3_mlx_section_value.py"
            ]
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    implementation_sweep = {
        "schema": "nerv_implementation_design_sweep.v1",
        "axis_tag": "[planning/control]",
        "verdict": "GO_IMPLEMENTATION_TRIAGE__NO_GO",
        "stack_sweeps": [
            {
                "stack_id": "snerv",
                "production_blockers": ["snerv_proof_missing:official_forward_parity"],
            }
        ],
        "blockers": ["snerv_proof_missing:official_forward_parity"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    source_parity = build_nerv_source_parity_contract(
        repo_root=Path(__file__).resolve().parents[3],
        families=("snerv",),
    )

    bridge = build_nerv_master_consumer_bridge(
        seam=seam,
        control_inventory=control_inventory,
        implementation_sweep=implementation_sweep,
        source_parity_contract=source_parity,
    )
    assert bridge["schema"] == "nerv_master_consumer_bridge.v1"
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False
    assert bridge["master_consumer_units"]
    assert any(
        unit["unit_type"] == "design_memo_anchor_index"
        for unit in bridge["master_consumer_units"]
    )
    source_units = [
        unit
        for unit in bridge["master_consumer_units"]
        if unit["unit_type"] == "source_parity_feature_route"
    ]
    assert source_units
    primitive_unit = next(
        unit
        for unit in source_units
        if unit["feature_id"] == "snerv_official_mfu_hfr_tub_numeric_primitives"
    )
    assert primitive_unit["planner_action"] == (
        "consume_bound_source_feature_before_new_glue"
    )
    assert primitive_unit["required_for_long_training"] is True
    parity_unit = next(
        unit
        for unit in source_units
        if unit["feature_id"] == "snerv_official_mfu_hfr_tub_parity"
    )
    assert parity_unit["planner_action"] == (
        "close_source_parity_gap_before_promotion_claim"
    )
    assert parity_unit["required_for_long_training"] is False
    assert "snerv_official_mfu_hfr_tub_parity_missing" in parity_unit["blockers"]

    result = nerv_top_priority_stack_consumer.consume_candidate(bridge)
    assert result["schema"] == "nerv_top_priority_stack_consumer_result.v1"
    assert result["planner_action"] == "route_to_source_faithful_snerv_hinerv_parity"
    assert result["promotable"] is False
    assert result["predicted_delta_adjustment"] == 0.0

    registration = validate_consumer_module(nerv_top_priority_stack_consumer)
    assert registration.contract_compliant, registration.validation_errors


def test_master_bridge_consumes_snerv_geometry_units_without_collapsing_runs() -> None:
    seam = {
        "schema": "nerv_top_priority_stack_seam.v1",
        "top_priority_carriers": ["snerv", "hi_nerv"],
        "baseline_to_beat": "pr95_hnerv_muon",
        "blockers": [],
    }
    control_inventory = {
        "schema": "nerv_control_inventory.v1",
        "control_rows": [],
        "binding_gap_rows": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    implementation_sweep = {
        "schema": "nerv_implementation_design_sweep.v1",
        "stack_sweeps": [],
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    geometry = {
        "schema": "snerv_scorer_loop_geometry.v1",
        "label": "two_run_geometry",
        "best_descent_score_delta_linf": -0.02,
        "lowest_local_score_linf": 0.8,
        "blockers": ["full600_receiver_proof_required"],
        "allocator_units": [
            {
                "unit_type": "snerv_scorer_loop_qat_result",
                "family": "snerv",
                "unit_id": "snerv_scorer_loop_qat:one_pair",
                "report_path": "one.json",
                "n_pairs": 1,
                "score_delta_linf": -0.02,
                "receiver_contract_satisfied": True,
            },
            {
                "unit_type": "snerv_scorer_loop_qat_result",
                "family": "snerv",
                "unit_id": "snerv_scorer_loop_qat:four_pair",
                "report_path": "four.json",
                "n_pairs": 4,
                "score_delta_linf": -0.01,
                "receiver_contract_satisfied": True,
            },
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }

    bridge = build_nerv_master_consumer_bridge(
        seam=seam,
        control_inventory=control_inventory,
        implementation_sweep=implementation_sweep,
        snerv_scorer_loop_geometry=geometry,
    )
    units = [
        unit
        for unit in bridge["master_consumer_units"]
        if unit["unit_type"] == "snerv_scorer_loop_qat_result"
    ]
    assert len(units) == 2
    assert all(unit["geometry_report_label"] == "two_run_geometry" for unit in units)

    rate_bridge = build_nerv_rate_allocator_bridge(master_bridge=bridge)
    orders = [
        row
        for row in rate_bridge["rate_allocator_work_orders"]
        if row["work_order_type"] == "snerv_scorer_loop_qat_full600_followup"
    ]
    assert len(orders) == 2
    assert {
        order["source_unit_id"] for order in orders
    } == {
        "snerv_scorer_loop_qat:one_pair",
        "snerv_scorer_loop_qat:four_pair",
    }
    assert {
        order["payload"]["n_pairs"] for order in orders
    } == {1, 4}


def test_rate_allocator_bridge_routes_units_without_authority() -> None:
    rate_bridge = _synthetic_rate_bridge()

    assert rate_bridge["schema"] == "nerv_rate_allocator_bridge.v1"
    assert rate_bridge["score_claim"] is False
    assert rate_bridge["promotion_eligible"] is False
    assert rate_bridge["ready_for_exact_eval_dispatch"] is False
    modes = {row["mode"] for row in rate_bridge["receiver_precision_mode_policy"]}
    assert {
        "fp16_protected",
        "int8_protected",
        "int4",
        "int2",
        "zero",
        "rle_only",
    } <= modes
    orders = {row["work_order_id"]: row for row in rate_bridge["rate_allocator_work_orders"]}
    assert "close_snerv_receiver_rate_promotion_gates" in orders
    assert "route_bitmask_and_zero_packing_to_rate_allocator" in orders
    assert "route_master_gradient_xray_stack_to_rate_allocator" in orders
    assert any(
        row["work_order_type"] == "measured_modelsize_budget_ladder"
        for row in orders.values()
    )
    assert "bind_hi_nerv_decoder_weight_saliency_to_waterfill" in orders
    assert "replay_hi_nerv_hi_nerv_local_tiny_decoder_weight_waterfill" in orders
    assert "score_hi_nerv_hi_nerv_local_tiny_full_video_mlx_replay" in orders
    assert "guard_hi_nerv_archive_backend_drift_local_velocity" in orders
    assert "replay_snerv_snerv_trained_ladder_row_archive_decoder_weight_waterfill" in orders
    assert "compile_snerv_snerv_local_tiny_decoder_modes_to_receiver" in orders
    assert "replay_snerv_explicit_fp163_decoder_mode_plan_pair_robust" in orders
    assert any(
        row["work_order_type"] == "snerv_scorer_loop_qat_full600_followup"
        for row in orders.values()
    )
    assert any(
        row["work_order_type"] == "snerv_lf_payload_codec_full_archive_replay"
        for row in orders.values()
    )
    zero_order = orders["route_bitmask_and_zero_packing_to_rate_allocator"]
    assert {"zero", "rle_only", "int2", "int4"} <= set(
        zero_order["receiver_precision_modes"]
    )
    mode_order = orders["compile_snerv_snerv_local_tiny_decoder_modes_to_receiver"]
    assert {"fp16_protected", "int4"} <= set(mode_order["receiver_precision_modes"])
    assert mode_order["payload"]["mode_plan_cli_arg"] == "fp16,int4,fp16"
    assert mode_order["payload"]["probe_command_argv"] == [
        ".venv/bin/python",
        "tools/probe_snerv_decoder_mode_assignments.py",
        "--mode-plan",
        "fp16,int4,fp16",
        "--receiver-packet-dir",
        "/Volumes/VertigoDataTier/pact/snerv_decoder_mode_assignment_packets/snerv_local_tiny",
    ]
    assert mode_order["payload"]["probe_receiver_packet_dir"].endswith(
        "/snerv_local_tiny"
    )
    assert "receiver_decoded_byte_accounting_required" in mode_order["blockers"]
    waterfill_order = orders["replay_hi_nerv_hi_nerv_local_tiny_decoder_weight_waterfill"]
    assert waterfill_order["work_order_type"] == "decoder_weight_waterfill_archive_replay"
    assert waterfill_order["payload"]["archive_ladder_replay_command_argv"] == [
        ".venv/bin/python",
        "tools/build_hinerv_archive_size_ladder.py",
        "--row-id",
        "hi_nerv_local_tiny",
    ]
    assert waterfill_order["payload"]["archive_ladder_replay_output_dir"].endswith(
        "/hi_nerv_local_tiny"
    )
    assert "full_video_decoder_weight_saliency_replay_missing" in waterfill_order[
        "blockers"
    ]
    full_video_order = orders["score_hi_nerv_hi_nerv_local_tiny_full_video_mlx_replay"]
    assert (
        full_video_order["work_order_type"]
        == "receiver_proven_archive_full_video_mlx_replay"
    )
    assert full_video_order["payload"]["archive_bytes"] == 134_908
    assert full_video_order["payload"]["archive_path"].endswith("/archive.zip")
    assert full_video_order["payload"]["submission_dir"].endswith("/submission")
    assert full_video_order["payload"]["receiver_proof_ready"] is True
    assert "full_video_mlx_scorer_response_missing" in full_video_order["blockers"]
    drift_order = orders["guard_hi_nerv_archive_backend_drift_local_velocity"]
    assert drift_order["work_order_type"] == "local_backend_drift_authority_guard"
    assert drift_order["payload"]["local_dev_velocity_ready"] is True
    assert drift_order["payload"]["ready_backend_for_local_iteration"] == "mlx_metal"
    assert "contest_cpu_cuda_exact_eval_not_executed" in drift_order["blockers"]
    snerv_waterfill_order = orders[
        "replay_snerv_snerv_trained_ladder_row_archive_decoder_weight_waterfill"
    ]
    assert snerv_waterfill_order["payload"]["archive_ladder_replay_command_argv"] == []
    probe_order = orders["replay_snerv_explicit_fp163_decoder_mode_plan_pair_robust"]
    assert probe_order["payload"]["candidate_count"] == 1
    assert probe_order["payload"]["receiver_archive_packet_path"] == (
        "/Volumes/VertigoDataTier/pact/snerv_decoder_mode_assignment_packets/"
        "snerv_local_tiny/0000_explicit_fp163.snar"
    )
    assert probe_order["payload"]["receiver_archive_replay_verified"] is True
    assert (
        probe_order["payload"]["receiver_archive_packet_is_contest_archive_zip"]
        is False
    )
    qat_order = next(
        row
        for row in orders.values()
        if row["work_order_type"] == "snerv_scorer_loop_qat_full600_followup"
    )
    assert qat_order["work_order_type"] == "snerv_scorer_loop_qat_full600_followup"
    assert qat_order["payload"]["n_pairs"] == 1
    assert qat_order["payload"]["history_count"] == 1
    assert qat_order["payload"]["score_delta_linf"] == -0.01
    assert qat_order["payload"]["candidate_count"] == 3
    assert qat_order["payload"]["accepted_candidate_count"] == 1
    assert qat_order["payload"]["best_pair_deltas"][0]["pair_index"] == 0
    assert qat_order["payload"]["receiver_contract_satisfied"] is True
    assert "snerv_scorer_loop_qat_full600_missing" in qat_order["blockers"]
    lf_codec_order = next(
        row
        for row in orders.values()
        if row["work_order_type"] == "snerv_lf_payload_codec_full_archive_replay"
    )
    assert lf_codec_order["payload"]["selected_mode"] == "portfolio_auto"
    assert lf_codec_order["payload"]["selected_payload_bytes"] == 72
    assert lf_codec_order["payload"]["source_artifact_path"].endswith(
        "snerv_lf_payload_codec_sweep.json"
    )
    assert lf_codec_order["payload"]["source_artifact_bytes"] == 1234
    assert lf_codec_order["payload"]["source_artifact_sha256"] == "a" * 64
    assert lf_codec_order["payload"]["section_value_row_count"] == 1
    assert lf_codec_order["payload"]["byte_price_plan"]["schema"] == (
        "compact_nerv_byte_price_controller.v1"
    )
    assert "full_archive_receiver_replay_required" in lf_codec_order["blockers"]
    for order in orders.values():
        assert order["score_claim"] is False
        assert order["score_claim_valid"] is False
        assert order["promotion_eligible"] is False
        assert order["ready_for_exact_eval_dispatch"] is False


def test_rate_allocator_queue_compiles_work_orders_without_authority() -> None:
    rate_bridge = _synthetic_rate_bridge()
    section_value = {
        "schema": "compact_renderer_mlx_section_value_profile.v1",
        "candidate_id": "compact_candidate",
        "section_value_rows": [
            {
                "row_id": "cut_selector",
                "section_id": "selectors_rc",
                "archive_sha256": "a" * 64,
                "axis_tag": "[contest-CUDA]",
                "receiver_proof_status": "satisfied",
                "full_video_coverage": True,
                "archive_bytes_removed_vs_baseline": 200_000,
                "delta_nonrate_score": 0.01,
            },
            {
                "row_id": "admit_residual",
                "section_id": "residual_rc",
                "row_kind": "new_residual_sidecar",
                "archive_sha256": "b" * 64,
                "axis_tag": "[contest-CUDA]",
                "receiver_proof_status": "satisfied",
                "full_video_coverage": True,
                "bytes": 10_000,
                "delta_nonrate_score": -0.01,
            },
            {
                "row_id": "sampled_residual",
                "section_id": "residual_sampled",
                "row_kind": "new_residual_sidecar",
                "archive_sha256": "c" * 64,
                "axis_tag": "[contest-CUDA]",
                "receiver_proof_status": "satisfied",
                "max_pairs": 600,
                "n_samples": 6,
                "bytes": 10_000,
                "delta_nonrate_score": -0.01,
            },
        ],
    }
    queue = build_nerv_rate_allocator_work_queue(
        rate_bridge=rate_bridge,
        section_value_artifacts=(section_value,),
        queue_id="test_nerv_rate_allocator_queue",
    )

    assert queue["schema"] == "nerv_rate_allocator_work_queue.v1"
    assert queue["queue_kind"] == "planner_queue_not_experiment_queue"
    assert queue["queue_row_count"] == rate_bridge["work_order_count"]
    assert queue["blocked_queue_row_count"] > 0
    assert queue["local_planning_ready_row_count"] >= 0
    assert queue["section_admission_plan_count"] == 2
    assert queue["section_admission_queue_row_count"] == 4
    assert queue["section_admission_decision_counts"]["admit"] == 1
    assert queue["section_admission_decision_counts"]["cut"] == 1
    assert queue["section_admission_decision_counts"]["demote"] == 2
    assert queue["score_claim"] is False
    assert queue["score_claim_valid"] is False
    assert queue["promotion_eligible"] is False
    assert queue["ready_for_exact_eval_dispatch"] is False
    assert queue["exact_or_full_video_cuda_allowed"] is False
    assert queue["dispatch_allowed"] is False
    assert queue["activation_policy"]["planner_rows_are_executable_experiments"] is False
    assert "final_rate_attack" in queue["target_consumer_index"]
    assert "bit_allocator" in queue["target_consumer_index"]
    assert {"fp16_protected", "int8_protected", "int4", "int2", "zero", "rle_only"} <= set(
        queue["precision_mode_index"]
    )

    rows = {row["work_order_id"]: row for row in queue["queue_rows"]}
    zero_row = rows["route_bitmask_and_zero_packing_to_rate_allocator"]
    assert zero_row["planner_ingest"]["ingest_kind"] == "reuse_existing_control_binding"
    assert {"zero", "rle_only"} <= set(zero_row["receiver_precision_modes"])
    modelsize_rows = [
        row
        for row in rows.values()
        if row["planner_ingest"]["ingest_kind"] == "measured_modelsize_ladder_work_order"
    ]
    assert modelsize_rows
    assert all(
        row["planner_ingest"]["producer_tool"]
        == "tools/emit_nerv_trained_ladder_row.py"
        for row in modelsize_rows
    )
    assert all(
        row["planner_ingest"]["existing_tool_ingress"]
        == "tools/build_nerv_receiver_closed_modelsize_ladder.py"
        for row in modelsize_rows
    )
    assert all(
        row["planner_ingest"]["intermediate_harvest_tool"]
        == "tools/harvest_nerv_receiver_closed_ladder_rows.py"
        for row in modelsize_rows
    )
    assert all(
        row["pipeline_custody"]["canonical_ingest_sequence"]
        == [
            "tools/emit_nerv_trained_ladder_row.py",
            "tools/harvest_nerv_receiver_closed_ladder_rows.py",
            "tools/build_nerv_receiver_closed_modelsize_ladder.py",
        ]
        for row in modelsize_rows
    )
    assert all(
        row["planner_ingest"]["planning_context_tool"]
        == "tools/build_nerv_modelsize_archive_curve.py"
        for row in modelsize_rows
    )
    full_video_row = rows["score_hi_nerv_hi_nerv_local_tiny_full_video_mlx_replay"]
    assert (
        full_video_row["planner_ingest"]["ingest_kind"]
        == "receiver_proven_archive_full_video_mlx_replay"
    )
    assert full_video_row["planner_ingest"][
        "local_full_video_mlx_replay_runnable_now"
    ] is True
    assert full_video_row["planner_ingest"]["runnable_now"] is False
    assert full_video_row["planner_ingest"]["archive_bytes"] == 134_908
    cache_argv = full_video_row["planner_ingest"][
        "local_full_video_cache_command_argv"
    ]
    response_argv = full_video_row["planner_ingest"][
        "local_full_video_response_command_argv"
    ]
    assert cache_argv[:3] == [
        ".venv/bin/python",
        "tools/materialize_mlx_scorer_cache_from_submission.py",
        "--archive",
    ]
    assert "--receiver-direct-cache" in cache_argv
    assert cache_argv[cache_argv.index("--archive") + 1].endswith(
        "/hi_nerv_local_tiny/archive.zip"
    )
    assert "tools/run_mlx_scorer_response_cache.py" in response_argv
    assert response_argv[response_argv.index("--archive-size-bytes") + 1] == "134908"
    assert "--allow-unaudited-candidate-cache-debug" in response_argv
    assert (
        full_video_row["planner_ingest"][
            "local_full_video_response_cache_identity_mode"
        ]
        == "receiver_direct_unaudited_debug_override"
    )
    assert (
        "experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
        in response_argv
    )
    assert (
        full_video_row["planner_ingest"][
            "local_full_video_output_is_promotion_authority"
        ]
        is False
    )
    assert "full_video_mlx_scorer_response_missing" in full_video_row["blockers"]
    drift_row = rows["guard_hi_nerv_archive_backend_drift_local_velocity"]
    assert (
        drift_row["planner_ingest"]["ingest_kind"]
        == "local_backend_drift_authority_guard"
    )
    assert drift_row["planner_ingest"]["local_dev_velocity_ready"] is True
    assert (
        drift_row["planner_ingest"]["ready_backend_for_local_iteration"]
        == "mlx_metal"
    )
    assert (
        drift_row["planner_ingest"]["local_backend_output_is_promotion_authority"]
        is False
    )
    qat_row = next(
        row
        for row in rows.values()
        if row["work_order_type"] == "snerv_scorer_loop_qat_full600_followup"
    )
    assert (
        qat_row["planner_ingest"]["ingest_kind"]
        == "snerv_scorer_loop_qat_full600_followup"
    )
    assert (
        qat_row["planner_ingest"]["producer_tool"]
        == "experiments/train_substrate_snerv_scorer_loop_local.py"
    )
    assert (
        qat_row["planner_ingest"]["local_full600_continuation_runnable_now"]
        is True
    )
    assert qat_row["planner_ingest"]["source_history_count"] == 1
    assert qat_row["planner_ingest"]["source_score_delta_linf"] == -0.01
    assert qat_row["planner_ingest"]["source_candidate_count"] == 3
    assert qat_row["planner_ingest"]["source_accepted_candidate_count"] == 1
    assert qat_row["planner_ingest"]["source_best_pair_deltas"][0]["pair_index"] == 0
    qat_command = qat_row["planner_ingest"][
        "local_full600_continuation_command_argv"
    ]
    assert qat_command[:3] == [
        ".venv/bin/python",
        "experiments/train_substrate_snerv_scorer_loop_local.py",
        "--score-loop",
    ]
    assert qat_command[qat_command.index("--n-pairs") + 1] == "600"
    assert qat_command[qat_command.index("--wavelet") + 1] == "haar"
    assert (
        qat_row["planner_ingest"][
            "local_full600_continuation_output_is_promotion_authority"
        ]
        is False
    )
    lf_codec_row = next(
        row
        for row in rows.values()
        if row["work_order_type"] == "snerv_lf_payload_codec_full_archive_replay"
    )
    assert lf_codec_row["planner_ingest"]["ingest_kind"] == (
        "snerv_lf_payload_codec_full_archive_replay"
    )
    assert lf_codec_row["planner_ingest"]["producer_tool"] == (
        "tools/recode_snerv_lf_payload_archive.py"
    )
    assert lf_codec_row["planner_ingest"]["intermediate_harvest_tool"] == (
        "tools/build_snerv_lf_payload_codec_sweep.py"
    )
    assert lf_codec_row["planner_ingest"]["source_selected_mode"] == (
        "portfolio_auto"
    )
    assert lf_codec_row["planner_ingest"]["source_selected_payload_bytes"] == 72
    assert lf_codec_row["planner_ingest"]["source_artifact_path"].endswith(
        "snerv_lf_payload_codec_sweep.json"
    )
    assert lf_codec_row["planner_ingest"]["source_artifact_bytes"] == 1234
    assert lf_codec_row["planner_ingest"]["source_artifact_sha256"] == "a" * 64
    assert lf_codec_row["planner_ingest"]["source_packet_path"].endswith(
        "source.snar"
    )
    assert lf_codec_row["planner_ingest"]["source_packet_bytes"] == 5678
    assert lf_codec_row["planner_ingest"]["source_packet_sha256"] == "b" * 64
    assert lf_codec_row["planner_ingest"]["source_byte_price_plan_schema"] == (
        "compact_nerv_byte_price_controller.v1"
    )
    assert (
        lf_codec_row["planner_ingest"][
            "local_lf_codec_packet_sweep_is_promotion_authority"
        ]
        is False
    )
    assert (
        lf_codec_row["planner_ingest"]["local_full_archive_replay_runnable_now"]
        is True
    )
    command = lf_codec_row["planner_ingest"]["local_full_archive_replay_command_argv"]
    assert command[:5] == [
        ".venv/bin/python",
        "tools/recode_snerv_lf_payload_archive.py",
        "--packet",
        "/Volumes/VertigoDataTier/pact/snerv_lf_payload/source.snar",
        "--mode",
    ]
    assert "portfolio_auto" in command
    assert (
        lf_codec_row["planner_ingest"][
            "local_full_archive_replay_output_is_promotion_authority"
        ]
        is False
    )
    assert "" not in lf_codec_row["pipeline_custody"]["canonical_ingress_paths"]
    assert "" not in lf_codec_row["pipeline_custody"]["canonical_ingest_sequence"]
    gate_row = rows["close_snerv_receiver_rate_promotion_gates"]
    assert gate_row["status"] == "blocked_until_prerequisite_evidence"
    assert gate_row["planner_ingest"]["runnable_now"] is False
    mode_row = rows["compile_snerv_snerv_local_tiny_decoder_modes_to_receiver"]
    assert mode_row["planner_ingest"]["ingest_kind"] == (
        "receiver_visible_decoder_mode_assignment"
    )
    assert mode_row["planner_ingest"]["producer_tool"] == (
        "tools/build_snerv_waterfill_mode_assignment.py"
    )
    assert mode_row["planner_ingest"]["local_advisory_probe_runnable_now"] is True
    assert mode_row["planner_ingest"]["local_advisory_probe_command_argv"] == [
        ".venv/bin/python",
        "tools/probe_snerv_decoder_mode_assignments.py",
        "--mode-plan",
        "fp16,int4,fp16",
        "--receiver-packet-dir",
        "/Volumes/VertigoDataTier/pact/snerv_decoder_mode_assignment_packets/snerv_local_tiny",
    ]
    assert (
        mode_row["planner_ingest"]["local_advisory_output_is_promotion_authority"]
        is False
    )
    probe_row = rows["replay_snerv_explicit_fp163_decoder_mode_plan_pair_robust"]
    assert probe_row["planner_ingest"]["ingest_kind"] == (
        "decoder_mode_pair_robust_probe_followup"
    )
    assert probe_row["planner_ingest"]["producer_tool"] == (
        "tools/probe_snerv_decoder_mode_assignments.py"
    )
    assert probe_row["planner_ingest"]["source_receiver_packet_path"] == (
        "/Volumes/VertigoDataTier/pact/snerv_decoder_mode_assignment_packets/"
        "snerv_local_tiny/0000_explicit_fp163.snar"
    )
    assert probe_row["planner_ingest"]["source_receiver_replay_verified"] is True
    assert (
        probe_row["planner_ingest"]["local_pair_robust_replay_runnable_now"] is True
    )
    assert (
        probe_row["planner_ingest"][
            "local_pair_robust_replay_is_promotion_authority"
        ]
        is False
    )
    saliency_row = rows["bind_hi_nerv_decoder_weight_saliency_to_waterfill"]
    assert saliency_row["planner_ingest"]["ingest_kind"] == (
        "decoder_weight_saliency_waterfill_binding"
    )
    waterfill_row = rows["replay_hi_nerv_hi_nerv_local_tiny_decoder_weight_waterfill"]
    assert waterfill_row["planner_ingest"]["ingest_kind"] == (
        "decoder_weight_waterfill_archive_replay"
    )
    assert waterfill_row["planner_ingest"]["local_replay_runnable_now"] is True
    assert waterfill_row["planner_ingest"]["local_replay_command_argv"] == [
        ".venv/bin/python",
        "tools/build_hinerv_archive_size_ladder.py",
        "--row-id",
        "hi_nerv_local_tiny",
    ]
    assert (
        waterfill_row["planner_ingest"]["local_replay_output_is_promotion_authority"]
        is False
    )
    assert waterfill_row["planner_ingest"]["runnable_now"] is False
    snerv_waterfill_row = rows[
        "replay_snerv_snerv_trained_ladder_row_archive_decoder_weight_waterfill"
    ]
    assert snerv_waterfill_row["planner_ingest"]["producer_tool"] == (
        "tools/build_snerv_trained_ladder_waterfill.py"
    )
    assert snerv_waterfill_row["planner_ingest"]["existing_tool_ingress"] == (
        "tools/prove_snerv_receiver_archive.py"
    )
    assert snerv_waterfill_row["planner_ingest"]["local_replay_runnable_now"] is False
    for row in queue["queue_rows"]:
        assert row["score_claim"] is False
        assert row["score_claim_valid"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["dispatch_allowed"] is False
        assert row["exact_or_full_video_cuda_allowed"] is False
        custody = row["pipeline_custody"]
        assert custody["schema"] == "nerv_rate_allocator_pipeline_custody.v1"
        assert custody["direct_ad_hoc_execution_allowed"] is False
        assert custody["orphan_output_allowed"] is False
        assert custody["promotion_authority_allowed"] is False
        assert custody["runnable_now_is_authority"] is False
        assert custody["output_must_reenter"] == (
            "archive_bound_contract_or_receiver_closed_ladder_or_section_value_profile"
        )
        if row["planner_ingest"].get("producer_tool") or row["planner_ingest"].get(
            "existing_tool_ingress"
        ):
            assert custody["custody_mode"] == "canonical_pipeline_ingest"
            assert custody["canonical_ingress_paths"]
    admission_rows = {
        row["row_id"]: row for row in queue["section_admission_queue_rows"]
    }
    assert admission_rows["cut_selector"]["decision"] == "cut"
    assert admission_rows["admit_residual"]["decision"] == "admit"
    assert admission_rows["sampled_residual"]["decision"] == "demote"
    assert admission_rows["snerv_lf_payload_codec_portfolio_auto"]["decision"] == (
        "demote"
    )
    assert admission_rows["snerv_lf_payload_codec_portfolio_auto"][
        "economic_decision"
    ] == "cut"
    assert "full_video_coverage_missing" in admission_rows["sampled_residual"][
        "blockers"
    ]
    assert "full_video_coverage_missing" in admission_rows[
        "snerv_lf_payload_codec_portfolio_auto"
    ]["blockers"]
    for row in queue["section_admission_queue_rows"]:
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["dispatch_allowed"] is False


def _synthetic_rate_bridge() -> dict:
    seam = {
        "schema": "nerv_top_priority_stack_seam.v1",
        "axis_tag": "[planning/control]",
        "go_no_go_verdict": "GO_LOCAL_STACK_OPTIMIZATION__NO_GO",
        "top_priority_carriers": ["snerv", "hinerv"],
        "baseline_to_beat": "pr95_hnerv_muon",
        "blockers": ["pr101_cpu_recovery_pending_blocks_new_exact_or_full_video"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    control_inventory = {
        "schema": "nerv_control_inventory.v1",
        "axis_tag": "[planning/control]",
        "control_rows": [
            {
                "control_id": "bitmask_and_zero_packing",
                "applies_to": "cross_stack",
                "binding_status": "partially_wired",
                "missing_bindings": ["receiver_zero_rle_grammar_missing"],
            },
            {
                "control_id": "master_gradient_xray_stack",
                "applies_to": "cross_stack",
                "binding_status": "partially_wired",
                "missing_bindings": ["decoder_atom_sensitivity_missing"],
            },
        ],
        "binding_gap_rows": [
            {"gap_id": "decoder_atom_sensitivity_missing"},
        ],
        "local_binding_surfaces": {},
        "decoder_weight_saliency_replays": {
            "hi_nerv": {
                "schema": "hinerv_decoder_weight_saliency_replay.v1",
                "row_count": 1,
                "full_video_coverage": False,
                "saliency_group_count": 1,
                "blockers": ["full_video_coverage_missing"],
            }
        },
        "decoder_weight_waterfill_reports": {
            "hi_nerv": {
                "schema": "hinerv_archive_ladder_waterfill.v1",
                "report_path": ".omx/research/hinerv_archive_ladder_waterfill.json",
                "row_count": 1,
                "waterfill_rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "archive_bytes": 134_938,
                        "archive_sha256": "e" * 64,
                        "state_npz_artifact_sha256": "f" * 64,
                        "waterfill_summary": {
                            "group_count": 24,
                            "total_selected_byte_delta": -307_010,
                        },
                        "archive_ladder_replay_command_axis_tag": (
                            "[planning/control:false-authority]"
                        ),
                        "archive_ladder_replay_command_argv": [
                            ".venv/bin/python",
                            "tools/build_hinerv_archive_size_ladder.py",
                            "--row-id",
                            "hi_nerv_local_tiny",
                        ],
                        "archive_ladder_replay_command_hint": (
                            ".venv/bin/python tools/build_hinerv_archive_size_ladder.py "
                            "--row-id hi_nerv_local_tiny"
                        ),
                        "archive_ladder_replay_output_dir": (
                            "/Volumes/VertigoDataTier/pact/"
                            "hinerv_archive_ladder_waterfill_replay/"
                            "hi_nerv_local_tiny"
                        ),
                        "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                    }
                ],
                "blockers": [
                    "decoder_weight_saliency_replay_required_for_authority"
                ],
            },
            "snerv": {
                "schema": "snerv_trained_ladder_waterfill.v1",
                "report_path": ".omx/research/snerv_trained_ladder_waterfill.json",
                "row_count": 1,
                "waterfill_rows": [
                    {
                        "row_id": "snerv_trained_ladder_row_archive",
                        "archive_bytes": 1_188_221,
                        "archive_sha256": "9" * 64,
                        "archive_ladder_replay_command_argv": [],
                        "archive_ladder_replay_output_dir": None,
                        "waterfill_summary": {
                            "group_count": 3,
                            "total_selected_byte_delta": 0,
                        },
                        "blockers": ["sample_pair_count_below_full600"],
                    }
                ],
                "blockers": ["sample_pair_count_below_full600"],
            },
        },
        "archive_ladder_replay_actuator_reports": {
            "hi_nerv": {
                "schema": "hinerv_archive_ladder_replay_actuator.v1",
                "report_path": ".omx/research/hinerv_archive_ladder_replay_actuator.json",
                "row_count": 1,
                "receiver_proof_ready_row_count": 1,
                "replay_rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "status": "existing_report_loaded_false_authority",
                        "archive_bytes": 134_908,
                        "archive_sha256": "a" * 64,
                        "archive_path": (
                            "/Volumes/VertigoDataTier/pact/"
                            "hinerv_archive_ladder_waterfill_replay/"
                            "hi_nerv_local_tiny/archive.zip"
                        ),
                        "submission_dir": (
                            "/Volumes/VertigoDataTier/pact/"
                            "hinerv_archive_ladder_waterfill_replay/"
                            "hi_nerv_local_tiny/submission"
                        ),
                        "spine_manifest_path": (
                            "/Volumes/VertigoDataTier/pact/"
                            "hinerv_archive_ladder_waterfill_replay/"
                            "hi_nerv_local_tiny/spine_manifest.json"
                        ),
                        "receiver_proof_path": (
                            "/Volumes/VertigoDataTier/pact/"
                            "hinerv_archive_ladder_waterfill_replay/"
                            "hi_nerv_local_tiny/receiver_proof.json"
                        ),
                        "decoder_weight_waterfill_plan_path": (
                            "/Volumes/VertigoDataTier/pact/"
                            "hinerv_archive_ladder_waterfill_replay/"
                            "hi_nerv_local_tiny/waterfill_plan.json"
                        ),
                        "replay_report_path": (
                            ".omx/research/"
                            "hinerv_archive_size_ladder_replay_tiny.json"
                        ),
                        "replay_report_sha256": "b" * 64,
                        "receiver_proof_ready": True,
                        "archive_export_backend_counts": {"mlx": 1},
                        "blockers": [
                            "hinerv_archive_size_row_has_no_nonrate_score"
                        ],
                    }
                ],
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
            }
        },
        "archive_backend_drift_reports": {
            "hi_nerv": {
                "schema": "hinerv_archive_backend_drift.v1",
                "status": "local_backend_drift_within_tolerance_false_authority",
                "report_path": ".omx/research/hinerv_archive_backend_drift.json",
                "reference_label": "pytorch_portable_fallback",
                "candidate_label": "mlx_metal",
                "row_count": 4,
                "matched_row_count": 4,
                "byte_ready_row_count": 4,
                "max_abs_byte_delta_allowed": 1024,
                "max_abs_byte_delta_observed": 81,
                "sum_byte_delta_candidate_minus_reference": -194,
                "sum_rate_score_delta_candidate_minus_reference": -0.000129185,
                "within_byte_drift_tolerance": True,
                "local_dev_velocity_ready": True,
                "ready_backend_for_local_iteration": "mlx_metal",
                "blockers": [
                    "contest_cpu_cuda_exact_eval_not_executed",
                    "hinerv_archive_backend_drift_local_dev_velocity_only",
                ],
            }
        },
        "decoder_mode_assignment_reports": {
            "snerv": {
                "schema": "snerv_waterfill_mode_assignment.v1",
                "row_count": 1,
                "assignment_rows": [
                    {
                        "row_id": "snerv_local_tiny",
                        "decoder_payload_schema": "snerv_decoder_payload.v3",
                        "mode_plan_cli_arg": "fp16,int4,fp16",
                        "mode_histogram": {"fp16": 2, "int4": 1},
                        "ready_for_local_advisory_probe": True,
                        "ready_for_receiver_mode_export": True,
                        "probe_command_axis_tag": "[macOS-CPU advisory]",
                        "probe_command_argv": [
                            ".venv/bin/python",
                            "tools/probe_snerv_decoder_mode_assignments.py",
                            "--mode-plan",
                            "fp16,int4,fp16",
                            "--receiver-packet-dir",
                            "/Volumes/VertigoDataTier/pact/snerv_decoder_mode_assignment_packets/snerv_local_tiny",
                        ],
                        "probe_command_hint": (
                            ".venv/bin/python "
                            "tools/probe_snerv_decoder_mode_assignments.py "
                            "--mode-plan fp16,int4,fp16 --receiver-packet-dir "
                            "/Volumes/VertigoDataTier/pact/"
                            "snerv_decoder_mode_assignment_packets/snerv_local_tiny"
                        ),
                        "probe_receiver_packet_dir": (
                            "/Volumes/VertigoDataTier/pact/"
                            "snerv_decoder_mode_assignment_packets/snerv_local_tiny"
                        ),
                        "blockers": [
                            "receiver_mode_export_requires_byte_accounting"
                        ],
                    }
                ],
                "blockers": [
                    "mode_assignment_is_false_authority_until_receiver_replay"
                ],
            }
        },
        "decoder_mode_probe_reports": {
            "snerv": {
                "schema": "snerv_decoder_mode_assignment_probe.v1",
                "best_plan_label": "explicit_fp163",
                "best_plan_score_linf_advisory": 3.58,
                "mode_plan_count": 1,
                "candidates": [
                    {
                        "label": "explicit_fp163",
                        "modes": ["fp16", "fp16", "fp16"],
                        "mode_histogram": {"fp16": 3},
                        "score_linf": 3.58,
                        "receiver_archive_packet_path": (
                            "/Volumes/VertigoDataTier/pact/"
                            "snerv_decoder_mode_assignment_packets/"
                            "snerv_local_tiny/0000_explicit_fp163.snar"
                        ),
                        "receiver_archive_packet_bytes": 456_578,
                        "receiver_archive_packet_sha256": "d" * 64,
                        "receiver_archive_replay_verified": True,
                        "receiver_archive_packet_is_contest_archive_zip": False,
                    }
                ],
                "blockers": ["macos_cpu_advisory_only"],
            }
        },
        "snerv_scorer_loop_qat_reports": {
            "snerv": {
                "schema": "snerv_scorer_loop_qat_local_trainer.v1",
                "status": "snerv_scorer_loop_qat_report_available_false_authority",
                "axis_tag": "[macOS-CPU advisory]",
                "report_path": (
                    ".omx/research/"
                    "snerv_scorer_loop_qat_local_trainer_smoke.json"
                ),
                "n_pairs": 1,
                "levels": 1,
                "wavelet": "haar",
                "qat_bits": 8,
                "search_mode": "random_signed",
                "scorer_loop_evaluations": 1,
                "history_count": 1,
                "selection_policy": "largest_pair_count_then_accepted_then_evaluations",
                "baseline_archive_bytes": 929_937,
                "best_archive_bytes": 929_937,
                "baseline_score_linf": 0.8545,
                "best_score_linf": 0.8445,
                "score_delta_linf": -0.01,
                "score_delta_fraction": -0.011703920421299,
                "candidate_count": 3,
                "accepted_candidate_count": 1,
                "rejected_candidate_count": 2,
                "best_pair_deltas": [
                    {
                        "pair_index": 0,
                        "score_linf_without_rate_delta": -0.02,
                        "d_seg_linf_delta": -0.001,
                        "d_pose_linf_delta": -0.0001,
                    }
                ],
                "accepted_improvement": False,
                "ready_for_pose_guard_gate": False,
                "receiver_contract_satisfied": True,
                "result_sha256": "e" * 64,
                "blockers": [
                    "local_smoke_only_not_full_600_pairs",
                    "paired_contest_cpu_cuda_pass_missing",
                ],
            }
        },
        "snerv_lf_payload_codec_sweep_reports": {
            "snerv": {
                "schema": "snerv_lf_payload_codec_sweep.v1",
                "status": "snerv_lf_payload_codec_sweep_rows_available_false_authority",
                "report_path": ".omx/research/snerv_lf_payload_codec_sweep.json",
                "source_artifact_path": ".omx/research/snerv_lf_payload_codec_sweep.json",
                "source_artifact_bytes": 1234,
                "source_artifact_sha256": "a" * 64,
                "source_kind": "snar1_packet",
                "source_packet_path": (
                    "/Volumes/VertigoDataTier/pact/snerv_lf_payload/source.snar"
                ),
                "source_packet_bytes": 5678,
                "source_packet_sha256": "b" * 64,
                "source_packet_metadata": {"n_pairs": 16},
                "selection_policy": (
                    "largest_plane_count_then_richest_section_value_then_lowest_selected_payload_bytes"
                ),
                "history_count": 1,
                "plane_count": 16,
                "raw_i64_bytes": 512,
                "baseline_mode": "int64_lzma",
                "baseline_payload_bytes": 300,
                "selected_mode": "portfolio_auto",
                "selected_payload_bytes": 72,
                "selected_packet_schema": "snerv_lf_quant_payload.v2",
                "row_count": 1,
                "section_value_row_count": 1,
                "byte_price_decision_counts": {"demote": 1},
                "codec_rows": [
                    {
                        "mode": "portfolio_auto",
                        "payload_bytes": 72,
                        "packet_schema": "snerv_lf_quant_payload.v2",
                        "blockers": [
                            "snerv_lf_payload_codec_row_has_no_scorer_replay"
                        ],
                    }
                ],
                "section_value_rows": [
                    {
                        "row_id": "snerv_lf_payload_codec_portfolio_auto",
                        "section_id": "snerv_lf_payload",
                        "family": "snerv",
                        "scope": "lf_payload_codec_replacement",
                        "row_kind": "existing_section_cut",
                        "candidate_mode": "portfolio_auto",
                        "baseline_mode": "int64_lzma",
                        "baseline_payload_bytes": 300,
                        "payload_bytes": 72,
                        "byte_delta": -228,
                        "delta_nonrate_score": 0.0,
                        "axis_tag": "[planning/control]",
                        "receiver_proof_status": (
                            "packet_exact_only_full_archive_replay_missing"
                        ),
                        "full_video_coverage": False,
                        "blockers": ["full_video_coverage_missing"],
                    }
                ],
                "byte_price_plan": {
                    "schema": "compact_nerv_byte_price_controller.v1",
                    "decision_rows": [
                        {
                            "row_id": "snerv_lf_payload_codec_portfolio_auto",
                            "section_id": "snerv_lf_payload",
                            "row_kind": "existing_section_cut",
                            "decision": "demote",
                            "economic_decision": "cut",
                            "byte_delta": -228,
                            "delta_nonrate_score": 0.0,
                            "source": {"candidate_mode": "portfolio_auto"},
                            "blockers": ["full_video_coverage_missing"],
                        }
                    ],
                },
                "blockers": [
                    "snerv_lf_payload_codec_sweep_false_authority_no_scorer_replay"
                ],
            }
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    implementation_sweep = {
        "schema": "nerv_implementation_design_sweep.v1",
        "axis_tag": "[planning/control]",
        "stack_sweeps": [
            {
                "stack_id": "snerv",
                "production_blockers": [
                    "snerv_proof_missing:mixed_precision_receiver_byte_accounting",
                    "snerv_proof_missing:full600_byte_closed_receiver_proof",
                    "snerv_proof_missing:paired_contest_CPU_CUDA_pass",
                ],
            }
        ],
        "blockers": ["snerv_proof_missing:mixed_precision_receiver_byte_accounting"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    modelsize_curve = build_modelsize_archive_curve(
        byte_caps=(178_417,),
        resolution_modes={"scorer_internal_384x512": 384 * 512},
    )
    bridge = build_nerv_master_consumer_bridge(
        seam=seam,
        control_inventory=control_inventory,
        implementation_sweep=implementation_sweep,
        modelsize_curve=modelsize_curve,
    )

    return build_nerv_rate_allocator_bridge(master_bridge=bridge)


def _minimal_repo_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".omx/state").mkdir(parents=True, exist_ok=True)
    return root


def _write_official_sources(oss_root: Path) -> None:
    snerv_root = oss_root / "repos" / "SNeRV"
    hinerv_root = oss_root / "repos" / "HiNeRV"
    for rel in STACK_REQUIREMENTS["snerv"]["official_required_files"]:
        _write(
            snerv_root / rel,
            (
                "--modelsize fc_dim np.roots pytorch_wavelets DWT haar "
                "MFU HFR SNeRV_T emb_size quant_model_bit quant_embed_bit "
                "quant_embed2_bit quant_vid.pth"
            ),
        )
    for rel in STACK_REQUIREMENTS["hinerv"]["official_required_files"]:
        _write(
            hinerv_root / rel,
            (
                "GridEncoding grid_level grid_level_scale TemporalLocalGridEncoding "
                "temp_local_grid video_to_patch patch_to_video patch_mode "
                "trilinear upsample out_patch_size --prune-ratio --quant-level "
                "--quant-noise --quant-ste compress_and_save_model"
            ),
        )


def _write_local_sources(repo_root: Path) -> None:
    for rel in STACK_REQUIREMENTS["snerv"]["local_surfaces"]:
        _write(
            repo_root / rel,
            (
                "SNERV_SPECTRA_PRESERVING_ADAPTER "
                "SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF MultiResolutionFusionUnit "
                "HighFrequencyRestorer SnervTemporalExtension receiver archive "
                "sha256 bytes int2 int4 int8 fp16 zero PoseNet pose guard "
                "scorer_loop waterfill linf allocation"
            ),
        )
    for rel in STACK_REQUIREMENTS["hinerv"]["local_surfaces"]:
        _write(
            repo_root / rel,
            (
                "sketch l0 not_source_faithful SegNet PoseNet score quant "
                "noise coder master_gradient VJP linf archive receiver bytes "
                "HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF "
                "apply_decoder_pruning apply_decoder_quant_noise "
                "measure_hi_nerv_decoder_bitstream_roundtrip "
                "select_hi_nerv_bitstream_codec_by_scorer_waterfill"
            ),
        )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
