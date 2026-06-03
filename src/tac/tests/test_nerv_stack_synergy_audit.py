# SPDX-License-Identifier: MIT
"""Tests for the HiNeRV/SNeRV full-stack synergy audit."""

from __future__ import annotations

from pathlib import Path

from tac.analysis.nerv_stack_synergy_audit import build_nerv_stack_synergy_audit

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_nerv_stack_synergy_audit_is_false_authority_and_binds_both_stacks() -> None:
    audit = build_nerv_stack_synergy_audit(
        repo_root=REPO_ROOT,
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        memo_limit_per_stack=8,
        marker_limit_per_stack=12,
    )

    assert audit["schema"] == "nerv_stack_synergy_audit.v1"
    assert audit["score_claim"] is False
    assert audit["ready_for_exact_eval_dispatch"] is False
    stacks = {row["stack_id"]: row for row in audit["stacks"]}
    assert set(stacks) == {"hi_nerv", "snerv"}
    assert len(audit["blockers"]) == len(set(audit["blockers"]))
    for stack in stacks.values():
        assert len(stack["blockers"]) == len(set(stack["blockers"]))
    assert stacks["hi_nerv"]["modelsize_budget"]["schema"] == "nerv_modelsize_budget.v1"
    assert stacks["snerv"]["modelsize_budget"]["schema"] == "snerv_modelsize_budget.v1"
    modelsize_binding = stacks["hi_nerv"]["modelsize_budget_binding"]
    assert modelsize_binding["schema"] == "hinerv_modelsize_budget_binding.v1"
    assert modelsize_binding["bound"] is True
    assert modelsize_binding["modelsize_archive_budget_bound"] is True
    assert modelsize_binding["trained_archive_byte_oracle_bound"] is False
    assert modelsize_binding["selected_candidate_count"] > 0
    assert not modelsize_binding["blockers"]
    snerv_modelsize_binding = stacks["snerv"]["modelsize_budget_binding"]
    assert snerv_modelsize_binding["schema"] == "snerv_modelsize_budget_binding.v1"
    assert snerv_modelsize_binding["bound"] is True
    assert snerv_modelsize_binding["modelsize_archive_budget_bound"] is True
    assert snerv_modelsize_binding["real_snar1_archive_byte_oracle_bound"] is False
    assert snerv_modelsize_binding["selected_candidate_count"] > 0
    assert not snerv_modelsize_binding["blockers"]
    assert stacks["hi_nerv"]["pr95_stack_binding"]["schema"] == (
        "pr95_stack_binding_requirements.v1"
    )
    assert stacks["snerv"]["pr95_stack_binding"]["schema"] == (
        "pr95_stack_binding_requirements.v1"
    )
    assert stacks["hi_nerv"]["pr95_stack_binding"]["complete"] is False
    assert stacks["snerv"]["pr95_stack_binding"]["complete"] is False
    assert stacks["hi_nerv"]["local_status"] == (
        "mlx_train_export_adapter_present_but_not_full_upstream_control_surface"
    )
    assert stacks["snerv"]["local_status"] == (
        "receiver_bound_advisory_export_present_mlx_native_surfaces_unproven"
    )
    assert stacks["hi_nerv"]["source_faithfulness"][
        "source_faithful_upstream_hinerv"
    ] is False
    assert stacks["snerv"]["source_faithfulness"][
        "source_faithful_official_snerv"
    ] is False
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in stacks[
        "snerv"
    ]["blockers"]
    assert stacks["snerv"]["snerv_mlx_native_adapter_contract"][
        "surfaces_ready"
    ] is True
    assert (
        "hinerv_local_architecture_not_source_faithful_upstream_hinerv_feature_grid"
        in stacks["hi_nerv"]["blockers"]
    )
    assert (
        "hinerv_modelsize_candidate_consumption_requires_trained_archive_byte_oracle"
        in stacks["hi_nerv"]["blockers"]
    )
    assert "hi_nerv_modelsize_archive_budget_missing" not in stacks["hi_nerv"][
        "blockers"
    ]
    assert "hinerv_official_convnext_feature_grid_path_missing" not in stacks[
        "hi_nerv"
    ]["blockers"]
    assert "hinerv_official_patch_index_path_missing" not in stacks["hi_nerv"][
        "blockers"
    ]
    assert "hinerv_official_trilinear_feature_interpolation_path_missing" not in stacks["hi_nerv"]["blockers"]
    feature_grid_binding = stacks["hi_nerv"][
        "official_feature_grid_convnext_binding"
    ]
    assert feature_grid_binding["schema"] == (
        "hinerv_official_feature_grid_convnext_binding.v1"
    )
    assert feature_grid_binding["bound"] is True
    assert feature_grid_binding["full_upstream_source_forward_replay_proven"] is False
    assert not feature_grid_binding["blockers"]
    assert {row["source_id"] for row in feature_grid_binding["source_rows"]} == {
        "architecture",
        "mlx_renderer",
        "archive_roundtrip_tests",
    }
    assert stacks["hi_nerv"]["official_grid_trilinear_binding"]["bound"] is True
    assert stacks["hi_nerv"]["official_grid_trilinear_binding"]["score_claim"] is False
    patch_binding = stacks["hi_nerv"]["official_patch_index_binding"]
    assert patch_binding["schema"] == "hinerv_official_patch_index_binding.v1"
    assert patch_binding["bound"] is True
    assert patch_binding["full_patch_frame_equivalence_replay_proven"] is False
    assert not patch_binding["blockers"]
    official_source_binding = stacks["hi_nerv"]["official_source_audit_binding"]
    assert official_source_binding["schema"] == (
        "hinerv_official_source_audit_stack_binding.v1"
    )
    assert official_source_binding["artifact_supplied"] is False
    assert official_source_binding["official_forward_replay_ran"] is False
    assert official_source_binding["full_upstream_source_forward_replay_proven"] is False
    assert "hinerv_official_source_audit_artifact_not_supplied" in (
        official_source_binding["blockers"]
    )
    assert "hinerv_official_source_audit_artifact_not_supplied" not in stacks[
        "hi_nerv"
    ]["blockers"]
    assert {row["source_id"] for row in patch_binding["source_rows"]} == {
        "official_patch",
        "official_patch_tests",
    }
    assert (
        "hinerv_official_quantnoise_control_not_bound_to_mlx_trainer"
        not in stacks["hi_nerv"]["blockers"]
    )
    quantnoise_binding = stacks["hi_nerv"]["quantnoise_control_binding"]
    assert quantnoise_binding["schema"] == "hinerv_quantnoise_control_binding.v1"
    assert quantnoise_binding["bound"] is True
    assert quantnoise_binding["official_quant_levels_6_7_executable"] is True
    assert not quantnoise_binding["blockers"]
    assert {
        row["source_id"] for row in quantnoise_binding["source_rows"]
    } == {"bitstream", "mlx_renderer", "runner", "waterfill"}
    assert "hinerv_torchac_style_bitstream_pipeline_missing" in stacks["hi_nerv"]["blockers"]
    assert "hinerv_decoder_weight_saliency_waterfill_not_in_trainer" in stacks["hi_nerv"]["blockers"]
    assert "hinerv_receiver_load_strict_false_schema_drift_risk" not in stacks[
        "hi_nerv"
    ]["blockers"]
    strict_receiver = stacks["hi_nerv"]["strict_receiver_load_binding"]
    assert strict_receiver["schema"] == "hinerv_strict_receiver_load_binding.v1"
    assert strict_receiver["bound"] is True
    assert strict_receiver["strict_receiver_load"] is True
    assert not strict_receiver["blockers"]
    assert {row["source_id"] for row in strict_receiver["source_rows"]} == {
        "inflate",
        "receiver_tests",
    }
    archive_candidate = stacks["hi_nerv"]["archive_candidate_binding"]
    assert archive_candidate["schema"] == "hinerv_archive_candidate_binding.v1"
    assert archive_candidate["bound"] is True
    assert archive_candidate["byte_closed_archive_export_bound"] is True
    assert archive_candidate["receiver_proof_bound"] is True
    assert archive_candidate["archive_in_loop_byte_oracle_bound"] is False
    assert not archive_candidate["blockers"]
    assert {row["source_id"] for row in archive_candidate["source_rows"]} == {
        "archive_candidate",
        "archive_candidate_tests",
        "runtime_bridge",
    }
    assert "hi_nerv_byte_closed_archive_export_missing" not in stacks[
        "hi_nerv"
    ]["blockers"]
    assert "hi_nerv_receiver_proof_missing" not in stacks["hi_nerv"]["blockers"]
    assert "hi_nerv_archive_in_loop_byte_oracle_missing" in stacks["hi_nerv"][
        "blockers"
    ]
    assert "hi_nerv_real_posenet_teacher_missing" in stacks["hi_nerv"]["blockers"]
    assert "hi_nerv_qat_forward_missing" not in stacks["hi_nerv"]["blockers"]
    assert "hi_nerv_coder_aware_regularizer_missing" not in stacks["hi_nerv"]["blockers"]
    assert "snerv_pr95_staged_curriculum_missing" in stacks["snerv"]["blockers"]
    assert "snerv_modelsize_archive_budget_missing" not in stacks["snerv"][
        "blockers"
    ]
    assert (
        "hinerv_execute_family_does_not_yet_consume_modelsize_candidate_id"
        not in stacks["hi_nerv"]["blockers"]
    )
    assert (
        "snerv_local_carrier_not_source_faithful_official_snerv_multilayer_stack"
        in stacks["snerv"]["blockers"]
    )
    assert "snerv_official_mfu_source_forward_replay_missing" not in stacks[
        "snerv"
    ]["blockers"]
    assert "snerv_official_hfr_source_forward_replay_missing" not in stacks[
        "snerv"
    ]["blockers"]
    assert "snerv_official_snerv_t_full_tub_path_not_source_forward_parity" not in stacks[
        "snerv"
    ]["blockers"]
    assert (
        "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing"
        in stacks["snerv"]["blockers"]
    )
    assert "snerv_official_mfu_hfr_tub_receiver_export_not_bound" in stacks[
        "snerv"
    ]["blockers"]
    replay = stacks["snerv"]["official_mfu_hfr_tub_primitive_replay_binding"]
    assert replay["all_primitive_source_replay_proven"] is True
    assert replay["full_stack_source_forward_replay_proven"] is False
    assert replay["receiver_export_bound"] is False
    replay_rows = {row["component_id"]: row for row in replay["component_rows"]}
    assert set(replay_rows) == {"mfu", "hfr", "tub"}
    assert all(row["primitive_source_replay_proven"] for row in replay_rows.values())
    assert all(row["missing_source_markers"] == [] for row in replay_rows.values())
    assert all(row["missing_test_markers"] == [] for row in replay_rows.values())
    assert "snerv_official_snerv_t_temporal_path_missing" not in stacks["snerv"]["blockers"]
    assert "snerv_official_haar_j1_parity_missing" in stacks["snerv"]["blockers"]
    assert "snerv_l2_linf_receiver_packet_rate_accounting_not_separated" in stacks["snerv"]["blockers"]
    assert (
        "snerv_modelsize_candidate_consumption_requires_real_snar1_archive_byte_oracle"
        in stacks["snerv"]["blockers"]
    )
    assert "snerv_fc_dim_modelsize_control_not_bound_to_planner" not in stacks[
        "snerv"
    ]["blockers"]
    assert any(
        row["rel_path"] == "src/tac/analysis/nerv_modelsize_budget.py"
        and row["present"]
        for row in stacks["snerv"]["local_surface_files"]
    )
    assert any(
        row["rel_path"] == "src/tac/substrates/snerv_inverse_steg_carrier/official_mfu.py"
        and row["present"]
        for row in stacks["snerv"]["local_surface_files"]
    )
    assert any(
        row["rel_path"] == "src/tac/substrates/snerv_inverse_steg_carrier/official_hfr.py"
        and row["present"]
        for row in stacks["snerv"]["local_surface_files"]
    )
    assert any(
        row["rel_path"] == "src/tac/substrates/snerv_inverse_steg_carrier/official_tub.py"
        and row["present"]
        for row in stacks["snerv"]["local_surface_files"]
    )
    assert (
        "measured SNAR1 archive-byte curve for official modelsize candidates"
        in stacks["snerv"]["source_faithfulness"]["missing_upstream_axes"]
    )
    assert any(
        "official --modelsize/fc_dim candidates are source-bound" in item
        for item in stacks["snerv"]["planner_curriculum_links"]
    )
    assert (
        "snerv_execute_family_does_not_yet_consume_modelsize_candidate_id"
        not in stacks["snerv"]["blockers"]
    )
    assert any(row["present"] for row in stacks["hi_nerv"]["local_surface_files"])
    assert any(row["present"] for row in stacks["snerv"]["local_surface_files"])
    assert audit["planner_policy"]["planner_and_curriculum_are_coupled"] is True


def test_nerv_stack_synergy_audit_references_related_memos_and_upstream_controls() -> None:
    audit = build_nerv_stack_synergy_audit(
        repo_root=REPO_ROOT,
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        memo_limit_per_stack=20,
        marker_limit_per_stack=4,
    )

    stacks = {row["stack_id"]: row for row in audit["stacks"]}
    assert "--modelsize" in stacks["hi_nerv"]["upstream_controls"]["hnerv_flags"]
    assert "--modelsize" in stacks["snerv"]["upstream_controls"]["snerv_flags"]
    assert "--quant-level" in stacks["hi_nerv"]["upstream_controls"]["hinerv_flags"]
    assert any(
        "snerv" in row["rel_path"].lower()
        for row in stacks["snerv"]["related_memos"]
    )
    assert any(
        "hinerv" in row["rel_path"].lower() or "hnerv" in row["rel_path"].lower()
        for row in stacks["hi_nerv"]["related_memos"]
    )
    assert audit["shared_synergy_surface_count"] > 0


def test_nerv_stack_synergy_audit_consumes_hinerv_official_forward_falsification_without_promotion() -> None:
    audit = build_nerv_stack_synergy_audit(
        repo_root=REPO_ROOT,
        hard_byte_ceilings=(178_000,),
        num_pairs=600,
        memo_limit_per_stack=4,
        marker_limit_per_stack=4,
        hinerv_official_source_audit={
            "schema": "hinerv_official_source_parity_audit.v1",
            "authority": "false_authority_source_audit_no_score_claim",
            "official_forward_parity_proven": False,
            "official_forward_parity_artifact_row": {
                "status": "present",
                "path": "/Volumes/VertigoDataTier/pact/evidence/hinerv_forward.json",
                "bytes": 1234,
                "sha256": "a" * 64,
                "parity_passed": False,
                "parity_falsified": True,
                "falsification_accepted": True,
            },
            "component_state_rows": [
                {
                    "component_id": "core_hierarchical_renderer",
                    "source_forward_parity_proven": False,
                    "source_forward_parity_falsified": True,
                    "official_source_forward_replay": {
                        "backend": "official_torch_cpu_full_hinerv_forward",
                        "replay_ran": True,
                        "input_bundle_sha256": "b" * 64,
                        "official_output_sha256": "c" * 64,
                        "official_weight_sha256": "d" * 64,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                },
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    stacks = {row["stack_id"]: row for row in audit["stacks"]}
    binding = stacks["hi_nerv"]["official_source_audit_binding"]
    assert binding["artifact_supplied"] is True
    assert binding["audit_schema_valid"] is True
    assert binding["official_forward_replay_ran"] is True
    assert binding["official_forward_replay_backend"] == (
        "official_torch_cpu_full_hinerv_forward"
    )
    assert binding["official_forward_input_bundle_sha256"] == "b" * 64
    assert binding["official_forward_output_sha256"] == "c" * 64
    assert binding["official_weight_sha256"] == "d" * 64
    assert binding["official_forward_parity_proven"] is False
    assert binding["official_forward_parity_falsified"] is True
    assert binding["falsification_accepted"] is True
    assert binding["full_upstream_source_forward_replay_proven"] is False
    assert binding["blockers"] == []
    assert binding["score_claim"] is False
    assert binding["ready_for_exact_eval_dispatch"] is False
    assert stacks["hi_nerv"]["source_faithfulness"][
        "source_faithful_upstream_hinerv"
    ] is False
    assert (
        "hinerv_local_architecture_not_source_faithful_upstream_hinerv_feature_grid"
        in stacks["hi_nerv"]["blockers"]
    )
