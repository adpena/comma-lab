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
    assert "hinerv_official_convnext_feature_grid_path_missing" in stacks["hi_nerv"]["blockers"]
    assert "hinerv_official_trilinear_feature_interpolation_path_missing" not in stacks["hi_nerv"]["blockers"]
    assert stacks["hi_nerv"]["official_grid_trilinear_binding"]["bound"] is True
    assert stacks["hi_nerv"]["official_grid_trilinear_binding"]["score_claim"] is False
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
    assert "hi_nerv_real_posenet_teacher_missing" in stacks["hi_nerv"]["blockers"]
    assert "hi_nerv_qat_forward_missing" not in stacks["hi_nerv"]["blockers"]
    assert "hi_nerv_coder_aware_regularizer_missing" not in stacks["hi_nerv"]["blockers"]
    assert "snerv_pr95_staged_curriculum_missing" in stacks["snerv"]["blockers"]
    assert (
        "hinerv_execute_family_does_not_yet_consume_modelsize_candidate_id"
        not in stacks["hi_nerv"]["blockers"]
    )
    assert (
        "snerv_local_carrier_not_source_faithful_official_snerv_multilayer_stack"
        in stacks["snerv"]["blockers"]
    )
    assert "snerv_official_mfu_source_forward_replay_missing" in stacks["snerv"]["blockers"]
    assert "snerv_official_hfr_source_forward_replay_missing" in stacks["snerv"]["blockers"]
    assert "snerv_official_snerv_t_full_tub_path_not_source_forward_parity" in stacks["snerv"]["blockers"]
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
