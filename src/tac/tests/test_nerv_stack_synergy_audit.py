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
    assert stacks["hi_nerv"]["modelsize_budget"]["schema"] == "nerv_modelsize_budget.v1"
    assert stacks["snerv"]["modelsize_budget"]["schema"] == "snerv_modelsize_budget.v1"
    assert stacks["hi_nerv"]["local_status"] == (
        "mlx_train_export_adapter_present_but_not_full_upstream_control_surface"
    )
    assert stacks["snerv"]["local_status"] == (
        "receiver_bound_advisory_export_present_mlx_native_train_missing"
    )
    assert stacks["hi_nerv"]["source_faithfulness"][
        "source_faithful_upstream_hinerv"
    ] is False
    assert stacks["snerv"]["source_faithfulness"][
        "source_faithful_official_snerv"
    ] is False
    assert "snerv_mlx_native_train_export_adapter_missing" in stacks["snerv"]["blockers"]
    assert (
        "hinerv_local_architecture_not_source_faithful_upstream_hinerv_feature_grid"
        in stacks["hi_nerv"]["blockers"]
    )
    assert (
        "hinerv_modelsize_candidate_consumption_requires_trained_archive_byte_oracle"
        in stacks["hi_nerv"]["blockers"]
    )
    assert (
        "hinerv_execute_family_does_not_yet_consume_modelsize_candidate_id"
        not in stacks["hi_nerv"]["blockers"]
    )
    assert (
        "snerv_local_carrier_not_source_faithful_official_snerv_multilayer_stack"
        in stacks["snerv"]["blockers"]
    )
    assert (
        "snerv_modelsize_candidate_consumption_requires_real_snar1_archive_byte_oracle"
        in stacks["snerv"]["blockers"]
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
