# SPDX-License-Identifier: MIT
"""Tests for PR95-grade stack binding requirements."""

from __future__ import annotations

from tac.analysis.pr95_stack_binding_requirements import (
    PRELAUNCH_REQUIREMENT_IDS,
    REQUIREMENTS,
    build_pr95_long_campaign_prelaunch_gate,
    build_pr95_stack_binding_evidence,
    build_pr95_stack_binding_requirements,
)


def test_pr95_stack_binding_requirements_fail_closed_when_evidence_missing() -> None:
    audit = build_pr95_stack_binding_requirements(family="hi_nerv", evidence={})

    assert audit["schema"] == "pr95_stack_binding_requirements.v1"
    assert audit["family"] == "hi_nerv"
    assert audit["required_count"] == len(REQUIREMENTS)
    assert audit["satisfied_count"] == 0
    assert audit["missing_count"] == len(REQUIREMENTS)
    assert audit["complete"] is False
    assert "hi_nerv_real_posenet_teacher_missing" in audit["blockers"]
    assert "hi_nerv_archive_in_loop_byte_oracle_missing" in audit["blockers"]
    assert audit["score_claim"] is False
    assert audit["ready_for_exact_eval_dispatch"] is False


def test_pr95_stack_binding_requirements_record_partial_real_evidence() -> None:
    audit = build_pr95_stack_binding_requirements(
        family="snerv",
        evidence={
            "carrier_source_or_documented_adaptation": True,
            "modelsize_archive_budget": True,
            "byte_closed_archive_export": True,
            "exact_auth_gate_plan": True,
        },
    )

    assert audit["satisfied_count"] == 4
    assert audit["missing_count"] == len(REQUIREMENTS) - 4
    rows = {row["requirement_id"]: row for row in audit["rows"]}
    assert rows["modelsize_archive_budget"]["satisfied"] is True
    assert rows["byte_closed_archive_export"]["satisfied"] is True
    assert rows["real_segnet_teacher"]["satisfied"] is False
    assert "snerv_byte_closed_archive_export_missing" not in audit["blockers"]
    assert "snerv_real_segnet_teacher_missing" in audit["blockers"]


def test_pr95_stack_binding_requirements_can_reach_complete_without_authority() -> None:
    audit = build_pr95_stack_binding_requirements(
        family="control",
        evidence={requirement.evidence_key: True for requirement in REQUIREMENTS},
    )

    assert audit["complete"] is True
    assert audit["missing_count"] == 0
    assert audit["blockers"] == []
    assert audit["promotion_eligible"] is False
    assert audit["rank_or_kill_eligible"] is False


def test_pr95_long_campaign_prelaunch_gate_excludes_post_run_proofs() -> None:
    audit = build_pr95_stack_binding_requirements(
        family="hi_nerv",
        evidence=build_pr95_stack_binding_evidence(
            carrier_source_or_documented_adaptation=True,
            modelsize_archive_budget=True,
            pr95_staged_curriculum=True,
            real_segnet_teacher=True,
            real_posenet_teacher=True,
            differentiable_pose_preprocess=True,
            eval_roundtrip_ste=True,
            scorer_input_distribution_guard=True,
            ema_archive_selection=True,
            qat_forward=True,
            coder_aware_regularizer=True,
            muon_adamw_partition=True,
            receiver_proof=False,
            local_cpu_replay_gate=False,
        ),
    )
    gate = build_pr95_long_campaign_prelaunch_gate(audit)

    assert gate["schema"] == "pr95_stack_binding_long_campaign_prelaunch_gate.v1"
    assert gate["required_count"] == len(PRELAUNCH_REQUIREMENT_IDS)
    assert gate["launch_allowed"] is True
    assert gate["blockers"] == []
    assert "receiver_proof" in gate["post_run_requirements_excluded"]
    assert "local_cpu_replay_gate" in gate["post_run_requirements_excluded"]
    assert audit["complete"] is False
