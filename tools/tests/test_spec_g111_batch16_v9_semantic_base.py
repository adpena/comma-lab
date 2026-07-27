# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os

import pytest

from tac.witness_dsl.spec_g111_batch16_v9_semantic_base import (
    PROGRAM_NAME,
    TARGET_CONTRACT_SCHEMA,
    TARGET_LEVER_NAME,
    Y1_RATE_ARBITRATION_SCHEMA,
    G111Batch16V9SemanticBaseError,
    _find_green_dry_start_release,
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


def test_g111_release_requires_same_typed_config_and_physical_target(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    target_path = str((tmp_path / "g109.json").resolve())
    target_sha = "1" * 64
    typed_hash = "2" * 64
    report = {
        "gate": "full_config_dry_start",
        "config": PROGRAM_NAME,
        "typed_config_hash": typed_hash,
        "num_pairs": 600,
        "green": True,
        "boot_ok": True,
        "resume_round_trip_ok": True,
        "peak_rss_gib": 24.0,
        "sec_per_ep_marginal": 42.0,
        "ts": "20260727T150000Z",
    }
    argv = [
        "python",
        "trainer.py",
        "--training-target-capsule",
        target_path,
        "--training-target-capsule-sha256",
        target_sha,
        "--num-pairs",
        "600",
    ]
    launch = {
        "schema": "witness_launch_manifest.v1",
        "config_family": PROGRAM_NAME,
        "spec_id": PROGRAM_NAME,
        "dsl_compile_hash": "3" * 64,
        "resolved_launch_argv": argv,
    }
    report_path = run / "dry_start_report.json"
    launch_path = run / "launch_manifest.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    contract = {
        "physical_receipt": {"path": target_path},
        "external_receipt_sha256": target_sha,
    }

    release = _find_green_dry_start_release(
        typed_config_hash=typed_hash,
        target_contract=contract,
        search_roots=(tmp_path,),
    )
    assert release is not None
    assert release["report"]["sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert release["launch_manifest"]["sha256"] == hashlib.sha256(
        launch_path.read_bytes()
    ).hexdigest()

    report["typed_config_hash"] = "4" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert (
        _find_green_dry_start_release(
            typed_config_hash=typed_hash,
            target_contract=contract,
            search_roots=(tmp_path,),
        )
        is None
    )


def test_g111_real_production_capsule_compiles_cold_typed_producer(
    tmp_path,
) -> None:
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
    assert target["parsed_g105_wire_verdict_implemented"] is True
    assert target["external_exhaustive_stage_compiler"] == (
        "tac.g121_retained_prepose.v2"
    )
    assert target["selected_xip2_coder_archive_abi_closed"] is True
    assert target["frontier_launch_blocker"] is None
    assert target["structural_semantic_rate_preflight"] == (structural_semantic_rate_preflight())
    assert target["candidate_payload_allowed"] is False
    if config.dsl_program_manifest["launch_blockers"]:
        assert config.dsl_program_manifest["held"] is True
        assert "dry-start" in config.dsl_program_manifest["hold_reason"]
    else:
        assert config.dsl_program_manifest["held"] is False
        assert config.dsl_program_manifest["hold_reason"] is None
        assert (
            config.dsl_program_manifest["green_dry_start_release"]["boot_ok"]
            is True
        )
    assert config.dsl_program_manifest["pointer_moved"] is False

    # The governed continuation is a typed delta over this same cold producer:
    # it must preserve the G109 physical target binding while committing to the
    # resume directory and its exact fresh-lineage parent receipt.
    from tools.launch_witness_run import (
        g111_production_resume_overrides,
        with_internal_dsl_lever,
        write_dsl_bound_launch,
    )

    parent_receipt = (
        tmp_path / "fresh_lineage" / f"{'a' * 64}.receipt.json"
    )
    parent_receipt.parent.mkdir()
    parent_receipt.write_bytes(b'{"sealed":"g111-parent"}\n')
    parent_sha = hashlib.sha256(parent_receipt.read_bytes()).hexdigest()
    (tmp_path / "fresh_lineage_tip.json").write_text(
        json.dumps(
            {
                "schema": "tac.fresh_producer_lineage_tip.v1",
                "receipt_path": str(parent_receipt),
                "receipt_sha256": parent_sha,
                "receipt_bytes": parent_receipt.stat().st_size,
                "checkpoint_id_sha256": "a" * 64,
                "root_sha256": "b" * 64,
                "sequence_index": 1,
                "epoch": 1,
                "stage": "stageCE",
                "complete_trajectory_proven": True,
            },
            sort_keys=True,
        )
    )
    resume_overrides = g111_production_resume_overrides(
        PROGRAM_NAME,
        tmp_path,
        dry_start=False,
    )
    resumed = with_internal_dsl_lever(
        config,
        name="g111_production_resume_v1",
        overrides=resume_overrides,
    )
    resumed_flags = resumed.typed.to_program().flag_dict()
    assert resumed_flags["--training-target-capsule"] == (
        target["physical_receipt"]["path"]
    )
    assert resumed_flags["--training-target-capsule-sha256"] == receipt_sha
    assert resumed_flags["--resume-from"] == str(tmp_path.resolve())
    assert resumed_flags["--fresh-lineage-parent-receipt"] == str(parent_receipt)
    assert (
        resumed_flags["--fresh-lineage-parent-receipt-sha256"] == parent_sha
    )
    assert resumed.typed.typed_config_hash() != config.typed.typed_config_hash()
    launch_sh, provenance, manifest, document = write_dsl_bound_launch(
        resumed,
        tmp_path / "bound_resume_launch",
        program_name=PROGRAM_NAME,
    )
    assert launch_sh.is_file() and provenance.is_file() and manifest.is_file()
    assert len(document["dsl_compile_hash"]) == 64
    assert "--resume-from" in document["resolved_argv"]
    assert str(tmp_path.resolve()) in document["resolved_argv"]
