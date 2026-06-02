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

    bridge = build_nerv_master_consumer_bridge(
        seam=seam,
        control_inventory=control_inventory,
        implementation_sweep=implementation_sweep,
    )
    assert bridge["schema"] == "nerv_master_consumer_bridge.v1"
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False
    assert bridge["master_consumer_units"]
    assert any(
        unit["unit_type"] == "design_memo_anchor_index"
        for unit in bridge["master_consumer_units"]
    )

    result = nerv_top_priority_stack_consumer.consume_candidate(bridge)
    assert result["schema"] == "nerv_top_priority_stack_consumer_result.v1"
    assert result["planner_action"] == "route_to_source_faithful_snerv_hinerv_parity"
    assert result["promotable"] is False
    assert result["predicted_delta_adjustment"] == 0.0

    registration = validate_consumer_module(nerv_top_priority_stack_consumer)
    assert registration.contract_compliant, registration.validation_errors


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
    zero_order = orders["route_bitmask_and_zero_packing_to_rate_allocator"]
    assert {"zero", "rle_only", "int2", "int4"} <= set(
        zero_order["receiver_precision_modes"]
    )
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
    assert queue["section_admission_plan_count"] == 1
    assert queue["section_admission_queue_row_count"] == 3
    assert queue["section_admission_decision_counts"]["admit"] == 1
    assert queue["section_admission_decision_counts"]["cut"] == 1
    assert queue["section_admission_decision_counts"]["demote"] == 1
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
    gate_row = rows["close_snerv_receiver_rate_promotion_gates"]
    assert gate_row["status"] == "blocked_until_prerequisite_evidence"
    assert gate_row["planner_ingest"]["runnable_now"] is False
    for row in queue["queue_rows"]:
        assert row["score_claim"] is False
        assert row["score_claim_valid"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["dispatch_allowed"] is False
        assert row["exact_or_full_video_cuda_allowed"] is False
    admission_rows = {
        row["row_id"]: row for row in queue["section_admission_queue_rows"]
    }
    assert admission_rows["cut_selector"]["decision"] == "cut"
    assert admission_rows["admit_residual"]["decision"] == "admit"
    assert admission_rows["sampled_residual"]["decision"] == "demote"
    assert "full_video_coverage_missing" in admission_rows["sampled_residual"][
        "blockers"
    ]
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
                "simplified not_source_faithful official_SNeRV receiver archive "
                "sha256 bytes int2 int4 int8 fp16 zero PoseNet pose guard "
                "scorer_loop waterfill linf allocation"
            ),
        )
    for rel in STACK_REQUIREMENTS["hinerv"]["local_surfaces"]:
        _write(
            repo_root / rel,
            (
                "sketch l0 not_source_faithful SegNet PoseNet score quant "
                "noise coder master_gradient VJP linf archive receiver bytes"
            ),
        )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
