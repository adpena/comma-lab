# SPDX-License-Identifier: MIT
from __future__ import annotations

import os

import pytest

from tac.witness_dsl.spec_g111_batch16_v9_semantic_base import (
    PROGRAM_NAME,
    TARGET_CONTRACT_SCHEMA,
    TARGET_LEVER_NAME,
    Y1_RATE_ARBITRATION_SCHEMA,
    G111Batch16V9SemanticBaseError,
    compile_g111_batch16_v9_semantic_base_launch_config,
    structural_semantic_rate_preflight,
)


def test_g111_rejects_non_sha_before_opening_target() -> None:
    with pytest.raises(G111Batch16V9SemanticBaseError, match="lowercase SHA-256"):
        compile_g111_batch16_v9_semantic_base_launch_config(
            training_target_capsule="/does/not/exist.json",
            training_target_capsule_sha256="not-a-sha",
        )


def test_g111_structural_semantic_rate_is_exact_and_not_a_score_claim() -> None:
    preflight = structural_semantic_rate_preflight()
    assert preflight["input_dim"] == 80
    assert preflight["counted_tensor_values"] == 71_159
    assert preflight["model_data_bytes"] == 72_430
    assert preflight["raw_y1_data_bytes"] == 38_400
    assert preflight["raw_semantic_packet_bytes"] == 111_840
    assert preflight["complete_archive_measured"] is False
    assert preflight["learned_entropy_predicted"] is False
    assert preflight["candidate_or_score_claim"] is False


def test_g111_real_production_capsule_compiles_cold_typed_producer() -> None:
    receipt = os.environ.get("TAC_G109_PRODUCTION_RECEIPT")
    receipt_sha = os.environ.get("TAC_G109_PRODUCTION_RECEIPT_SHA256")
    if not receipt or not receipt_sha:
        pytest.skip("set TAC_G109_PRODUCTION_RECEIPT and its external SHA for the real integration")

    config = compile_g111_batch16_v9_semantic_base_launch_config(
        training_target_capsule=receipt,
        training_target_capsule_sha256=receipt_sha,
    )
    flags = config.typed.to_program().flag_dict()
    target = config.dsl_program_manifest["training_target_contract"]

    assert config.name == PROGRAM_NAME
    assert TARGET_LEVER_NAME in config.dsl_program_manifest["expected_active_levers"]
    assert flags["--training-target-capsule"] == target["physical_receipt"]["path"]
    assert flags["--training-target-capsule-sha256"] == receipt_sha
    assert flags["--fresh-producer"] is True
    assert flags["--verdict-batch"] == 16
    assert flags["--self-orient"] is False
    assert flags["--render-aa"] == "none"
    assert flags["--mod-dim"] == 32
    assert flags["--pose-carrier-source"] == "generated_y1"
    assert flags["--out-dir"].startswith("/Volumes/VertigoDataTier/pact/")
    assert flags.get("--resume-from") is None
    assert flags.get("--warm-start-weights-only") is None
    assert target["pair_count"] == 600
    assert target["schema"] == TARGET_CONTRACT_SCHEMA
    assert target["same_forward_seg_margin_pose"] is True
    assert target["conditional_y0_source"] == "final_odd_code_y1_render"
    assert target["conditional_y0_source_boundary"] == "scorer_grid_uint8"
    assert target["conditional_y0_camera_realization"] == "tac.v10_factor2_selected_preimage.v1"
    assert target["pose_gradient_public_camera_realization_identical"] is True
    assert target["semantic_training_loss_public_wire_identical"] is False
    assert target["semantic_stage_selection_public_wire_identical"] is False
    assert target["serialized_even_code_rows_required"] is False
    assert target["post_semantic_compile_xi_refit_required"] is True
    assert target["y1_rate_arbitration"] == Y1_RATE_ARBITRATION_SCHEMA
    assert target["y1_rate_domain"] == "exact_complete_archive_zip_bytes"
    assert target["y1_wire_families"] == ["raw_i16le", "delta_rice_best_k"]
    assert target["outer_zip_methods"] == ["stored", "deflated"]
    assert target["fresh_lineage_root_seed_persisted"] is True
    assert target["fresh_lineage_root_recomputed_by_consumer"] is True
    assert target["physical_cold_full_state_checkpoint_before_first_step"] is True
    assert target["full_state_companion_required_for_own_lineage_claim"] is True
    assert target["recursive_physical_checkpoint_chain_required"] is True
    assert target["fresh_lineage_tip_schema"] == (
        "tac.fresh_producer_lineage_tip.v1"
    )
    assert target["resume_requires_external_parent_receipt_path_and_sha256"] is True
    assert target["semantic_verdict_surface"] == "parsed_G105_public_wire_v1"
    assert target["semantic_checkpoint_selection_surface"] == "parsed_G105_public_wire_v1"
    assert target["legacy_arbitrary_scale_int8_selection_allowed"] is False
    assert target["parsed_g105_wire_verdict_implemented"] is False
    assert target["frontier_launch_blocker"] == (
        "parsed_G105_wire_quantized_semantic_verdict_and_selection_not_wired"
    )
    assert target["structural_semantic_rate_preflight"] == (structural_semantic_rate_preflight())
    assert target["candidate_payload_allowed"] is False
    assert config.dsl_program_manifest["held"] is True
    assert "G105" in config.dsl_program_manifest["hold_reason"]
    assert config.dsl_program_manifest["pointer_moved"] is False
