# SPDX-License-Identifier: MIT
"""Smoke test for tools/operator_briefing.py orchestrator.

The orchestrator delegates to apogee_intN_pareto + score_dashboard +
predicted_vs_actual_reconciler — those have their own thorough test suites.
This file just guards against the orchestrator's subprocess wiring breaking
(wrong tool path, broken --skip flags, JSON composition bug).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BRIEFING = REPO / "tools" / "operator_briefing.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BRIEFING), *args],
        capture_output=True, text=True, check=True,
    )


def _load_briefing_module():
    spec = importlib.util.spec_from_file_location("operator_briefing_under_test", BRIEFING)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_briefing_delegates_to_repo_venv_when_available(monkeypatch):
    module = _load_briefing_module()
    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="{}\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run(BRIEFING, ["--json"])

    assert calls
    if module.REPO_VENV_PYTHON.is_file():
        assert calls[0][0] == str(module.REPO_VENV_PYTHON)


def test_briefing_runs_all_three_phases():
    proc = _run("--top", "3")
    assert "Codex inbox" in proc.stdout
    assert "open_questions:" in proc.stdout
    assert "Dispatch claim coordination" in proc.stdout
    assert "claim_lane_dispatch.py summary" in proc.stdout
    assert "Cloud provider readiness" in proc.stdout
    assert "Phase 1" in proc.stdout
    assert "Phase 2" in proc.stdout
    assert "Phase 3" in proc.stdout
    assert "Phase 1 exact-eval packets" in proc.stdout
    assert "Phase 1 exact-ready queues" in proc.stdout
    assert "Phase 1 blocked readiness artifacts" in proc.stdout
    assert "pr91_hpm1_readiness_bundle" in proc.stdout
    assert "wr01_apply_pr106x_half" in proc.stdout
    assert "pr106_q10_151byte_brotli" in proc.stdout
    assert "pr106x_lgblock16_1byte_brotli" in proc.stdout
    assert "hnerv_hlm1_fixed_latent_recode_exact_eval" in proc.stdout
    assert "hnerv_hlm1_xmember_exact_eval_20260514" in proc.stdout
    assert "Copy-safe next steps" in proc.stdout
    assert "assert_packet_ready_for_submit" in proc.stdout
    assert "refresh_with_operator_exact_cuda_approval" in proc.stdout
    assert "Phase 9 — L5-v2 TT5L-first frontier readiness" in proc.stdout
    assert "next non-PR106 L5 action:" in proc.stdout
    assert "asymptotic candidate count:" in proc.stdout
    assert "Asymptotic candidates:" in proc.stdout
    assert "paired measurement plan:" in proc.stdout
    assert "ATW v2 D4 verdict:" in proc.stdout
    assert "next exact-eval targets:" in proc.stdout


def test_briefing_skip_pareto_omits_phase1():
    proc = _run("--skip-pareto", "--top", "3")
    # Use the section-header form ("Phase 1 — ...") rather than bare "Phase 1"
    # because Phase 8 readiness summary (added 2026-05-09) mentions Phase 1
    # as a navigation breadcrumb, which is intentional.
    assert "Phase 1 — Pre-dispatch" not in proc.stdout
    assert "Phase 2" in proc.stdout
    assert "Phase 3" in proc.stdout


def test_briefing_skip_dashboard_omits_phase2():
    proc = _run("--skip-dashboard", "--top", "3")
    assert "Phase 1 — Pre-dispatch" in proc.stdout
    assert "Phase 2 — Post-dispatch" not in proc.stdout
    assert "Phase 3" in proc.stdout


def test_briefing_skip_reconciler_omits_phase3():
    proc = _run("--skip-reconciler", "--top", "3")
    assert "Phase 1 — Pre-dispatch" in proc.stdout
    assert "Phase 2" in proc.stdout
    assert "Phase 3 — Post-dispatch" not in proc.stdout


def test_briefing_json_composite_has_all_three_keys():
    proc = _run("--json", "--top", "3")
    out = json.loads(proc.stdout)
    assert out["target_score"] == 0.19
    assert "codex_inbox_summary" in out
    assert out["codex_inbox_summary"]["schema_version"] == "codex_to_claude_inbox_v1_20260518"
    assert "open_questions_count" in out["codex_inbox_summary"]
    assert out["dispatch_claim_summary"]["schema"] == "pact.dispatch_claim_summary.v1"
    assert out["dispatch_claim_historical_summary"]["schema"] == "pact.dispatch_claim_summary.v1"
    assert out["dispatch_readiness"]["schema"] == "pact.operator_dispatch_readiness.v1"
    assert out["dispatch_readiness"]["phase_1_exact_eval_packets"]["status"] in {
        "READY",
        "BLOCKED",
        "PENDING",
    }
    assert len(out["xray_tools"]) >= 5
    assert all(row["score_claim"] is False for row in out["xray_tools"])
    assert all(row["score_claim_valid"] is False for row in out["xray_tools"])
    assert all(row["promotion_eligible"] is False for row in out["xray_tools"])
    assert all(row["rank_or_kill_eligible"] is False for row in out["xray_tools"])
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in out["xray_tools"])
    assert all(row["tool_exists"] is True for row in out["xray_tools"])
    assert {
        "tools/xray_archive_section_entropy_heatmap.py",
        "tools/xray_paired_cpu_cuda_axis_delta.py",
        "tools/xray_pair_component_errors.py",
        "tools/xray_hardpair_hitlist.py",
        "tools/xray_substrate_classifier.py",
    }.issubset({row["tool"] for row in out["xray_tools"]})
    integration = out["cooperative_receiver_solver_integration"]
    assert integration["score_claim"] is False
    assert integration["score_claim_valid"] is False
    assert integration["promotion_eligible"] is False
    assert integration["rank_or_kill_eligible"] is False
    assert integration["ready_for_exact_eval_dispatch"] is False
    assert integration["campaign_count"] >= 1
    assert integration["autopilot_rows"] >= 1
    assert integration["meta_lagrangian_rows"] >= 1
    assert integration["pareto_rows"] >= 1
    assert integration["continual_learning_posterior_update_allowed"] is False
    assert integration["xray_grammars"] >= 1
    assert integration["magic_codec_entries"] >= 1
    assert integration["packet_compiler_grammars"] >= 1
    assert integration["canonical_packet_compiler"] == (
        "tac.packet_compiler.deterministic_compiler"
    )
    l5 = out["l5_v2_frontier_readiness"]
    assert l5["schema"] == "pact.l5_v2_frontier_readiness.v1"
    assert l5["score_claim"] is False
    assert l5["promotion_eligible"] is False
    assert l5["rank_or_kill_eligible"] is False
    assert l5["ready_for_exact_eval_dispatch"] is False
    atw_gate = l5["atw_v2_phase2_gate_status"]
    assert atw_gate["schema"] == "atw_codec_v2_phase2_gate_status_v1"
    assert atw_gate["d4_verdict"] == "INDEPENDENT"
    assert atw_gate["phase2_status"] == (
        "defer_measured_a1_latent_class_conditioning_surface"
    )
    assert atw_gate["next_action"] == "do_not_dispatch_atw_v2_phase2_from_this_signal"
    assert atw_gate["score_claim"] is False
    assert atw_gate["promotion_eligible"] is False
    assert atw_gate["rank_or_kill_eligible"] is False
    assert atw_gate["ready_for_exact_eval_dispatch"] is False
    assert atw_gate["dispatch_allowed"] is False
    assert atw_gate["phase2_lift_allowed"] is False
    assert l5["atw_v2_phase2_d4_verdict"] == "INDEPENDENT"
    assert l5["atw_v2_phase2_dispatch_allowed"] is False
    assert l5["atw_v2_phase2_lift_allowed"] is False
    assert l5["asymptotic_pursuit_candidate_count"] == 3
    assert len(l5["l5_v2_asymptotic_next_action_status"]) == 3
    assert {
        row["candidate_id"]
        for row in l5["asymptotic_pursuit_candidate_sample"]
    } == {
        "z6_z7_z8_predictive_coding_world_models",
        "rudin_floor_interpretable_ml_substrate",
        "tishby_ib_pure_substrate",
    }
    assert all(
        row["ready_for_exact_eval_dispatch"] is False
        for row in l5["asymptotic_pursuit_candidate_sample"]
    )
    assert all(
        "l1_scaffold_present" in row
        for row in l5["asymptotic_pursuit_candidate_sample"]
    )
    assert all(
        "expected_first_artifact_status" in row
        for row in l5["asymptotic_pursuit_candidate_sample"]
    )
    assert all(
        row["recommended_next_action_status"] == "completed_or_superseded"
        for row in l5["asymptotic_pursuit_candidate_sample"]
    )
    assert all(
        row["effective_recommended_next_action_id"].startswith(
            "completed_or_superseded:"
        )
        for row in l5["asymptotic_pursuit_candidate_sample"]
    )
    assert all(
        row["l5_v2_asymptotic_next_action_status"]["ledger_present"] is True
        for row in l5["asymptotic_pursuit_candidate_sample"]
    )
    assert all(
        row["l5_v2_asymptotic_next_action_status"][
            "next_prerequisite_status"
        ]["status"]
        == "completed_or_superseded"
        for row in l5["asymptotic_pursuit_candidate_sample"]
    )
    assert l5["measurement_schedule_score_claim"] is False
    assert l5["measurement_schedule_promotion_eligible"] is False
    assert l5["measurement_schedule_ready_for_exact_eval_dispatch"] is False
    assert l5["measurement_schedule_tool_path"].endswith(
        "tools/build_l5_v2_lattice_measurement_schedule.py"
    )
    assert l5["measurement_schedule_artifact_path"].endswith(
        ".omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json"
    )
    assert l5["paired_measurement_dispatch_plan_tool_path"].endswith(
        "tools/build_l5_v2_paired_measurement_dispatch_plan.py"
    )
    assert l5["paired_measurement_dispatch_plan_artifact_path"].endswith(
        ".omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.json"
    )
    assert l5["paired_measurement_dispatch_plan_exists"] is True
    assert l5["paired_measurement_dispatch_plan_source_schedule_stale"] is False
    assert l5["paired_measurement_dispatch_plan_source_schedule_sha256"] == l5[
        "paired_measurement_dispatch_plan_current_source_schedule_sha256"
    ]
    assert l5["paired_measurement_dispatch_plan_work_unit_count"] == 3
    assert l5["paired_measurement_dispatch_plan_ready_work_unit_count"] == 0
    assert l5["paired_measurement_dispatch_plan_score_claim"] is False
    assert l5["paired_measurement_dispatch_plan_score_claim_valid"] is False
    assert l5["paired_measurement_dispatch_plan_promotion_eligible"] is False
    assert l5["paired_measurement_dispatch_plan_ready_for_exact_eval_dispatch"] is False
    assert l5["paired_measurement_dispatch_plan_rank_or_kill_eligible"] is False
    assert l5["paired_measurement_dispatch_plan_dispatch_attempted"] is False
    assert len(l5["paired_measurement_dispatch_plan_command_sample"]) == 3
    assert all(
        "tools/dispatch_modal_paired_auth_eval.py"
        in row["dispatch_command_template"]
        for row in l5["paired_measurement_dispatch_plan_command_sample"]
    )
    assert all(
        "experiments/modal_auth_eval.py"
        not in row["dispatch_command_template"]
        for row in l5["paired_measurement_dispatch_plan_command_sample"]
    )
    assert all(
        row["dispatch_command_executable"] is False
        for row in l5["paired_measurement_dispatch_plan_command_sample"]
    )
    assert all(
        row["ready_for_operator_dispatch"] is False
        for row in l5["paired_measurement_dispatch_plan_command_sample"]
    )
    assert all(
        row["ready_for_provider_dispatch"] is False
        for row in l5["paired_measurement_dispatch_plan_command_sample"]
    )
    assert all(
        row["readiness_blockers"]
        for row in l5["paired_measurement_dispatch_plan_command_sample"]
    )
    assert all(
        "requires_byte_closed_archive_path" in row["readiness_blockers"]
        for row in l5["paired_measurement_dispatch_plan_command_sample"]
    )
    assert l5["target_rows_are_fail_fast_only"] is True
    assert isinstance(l5["canonical_sideinfo_evidence_present"], bool)
    if not l5["canonical_sideinfo_evidence_present"]:
        assert "requires_byte_closed_temporal_sideinfo_consumption_proof" in l5["blockers"]
    packetir_hash_matches = (
        l5["packetir_matrix_artifact_sha256"] == l5["packetir_matrix_expected_sha256"]
    )
    if packetir_hash_matches:
        assert "l5_v2_packetir_matrix_artifact_sha_mismatch" not in l5["blockers"]
    else:
        assert "l5_v2_packetir_matrix_artifact_sha_mismatch" in l5["blockers"]
    assert l5["packetir_section_entropy_matrix_exists"] is True
    assert l5["packetir_section_entropy_profiled_candidate_count"] >= 2
    assert l5["packetir_section_entropy_prototype_row_count"] >= 1
    assert l5["packetir_section_entropy_rate_positive_prototype_row_count"] == 0
    assert l5["packetir_section_entropy_adaptive_prototype_row_count"] >= 1
    assert l5["packetir_section_entropy_rate_positive_adaptive_prototype_row_count"] == 0
    assert (
        l5["packetir_section_entropy_derived_prefix_adaptive_prototype_row_count"]
        >= 1
    )
    assert (
        l5[
            "packetir_section_entropy_rate_positive_derived_prefix_adaptive_prototype_row_count"
        ]
        >= 1
    )
    assert (
        l5["packetir_section_entropy_best_adaptive_prototype"][
            "delta_bytes_vs_source_section"
        ]
        == 1
    )
    assert l5["packetir_section_entropy_best_rate_positive_adaptive_prototype"] is None
    assert (
        l5[
            "packetir_section_entropy_best_rate_positive_derived_prefix_adaptive_prototype"
        ]["delta_bytes_vs_source_section"]
        == -1
    )
    active_claim_count = int(out["dispatch_claim_summary"].get("active_count") or 0)
    assert l5["active_dispatch_claim_count"] == active_claim_count
    assert l5["dispatch_claim_gate_blocked"] == (active_claim_count > 0)
    if active_claim_count or not packetir_hash_matches:
        assert l5["packetir_matrix_dispatch_targets_suppressed"] is True
        assert l5["next_exact_eval_target_count"] == 0
        assert l5["next_exact_eval_targets"] == []
        assert l5["next_exact_eval_targets_sample"] == []
        if active_claim_count:
            assert (
                f"blocked_active_dispatch_claims_present:{active_claim_count}"
                in l5["blockers"]
            )
    else:
        assert l5["packetir_matrix_dispatch_targets_suppressed"] is False
        assert l5["next_exact_eval_target_count"] == 0
        assert l5["next_exact_eval_targets"] == []
        assert l5["next_exact_eval_targets_sample"] == []
    if packetir_hash_matches:
        assert l5["packetir_status_counts"]["runtime_consumption_blocked"] == 16
    else:
        assert sum(int(value) for value in l5["packetir_status_counts"].values()) >= 1
    assert l5["packetir_paired_candidate_count"] == 0
    assert l5["pr106_stack_cell_candidate_count"] == 0
    assert "l5_v2_packetir_no_runtime_bound_paired_exact_candidates" in l5[
        "blockers"
    ]
    if l5["next_exact_eval_targets_sample"]:
        assert all(
            row["ready_for_exact_eval_dispatch"] is False
            for row in l5["next_exact_eval_targets_sample"]
        )
        assert all(
            row["paired_dispatch_tool"] == "tools/dispatch_modal_paired_auth_eval.py"
            for row in l5["next_exact_eval_targets_sample"]
        )
        assert all(
            "--expected-runtime-tree-sha256 auto" in row["command_template"]
            for row in l5["next_exact_eval_targets_sample"]
        )
        assert all(
            "--skip-axis-if-promotable-anchor-exists" in row["command_template"]
            for row in l5["next_exact_eval_targets_sample"]
        )
        assert all(
            "<AXIS_SPECIFIC_MODAL_UPLOADED_RUNTIME_TREE_SHA256>"
            not in row["command_template"]
            for row in l5["next_exact_eval_targets_sample"]
        )
        assert all(
            "experiments/modal_auth_eval.py" not in row["command_template"]
            and "experiments/modal_auth_eval_cpu.py" not in row["command_template"]
            for row in l5["next_exact_eval_targets_sample"]
        )
    assert "provider_readiness" in out
    assert out["provider_readiness"].get("score_claim") is False
    assert "pareto" in out
    assert "dashboard" in out
    assert "reconciler" in out
    assert "exact_eval_packets" in out
    assert "active_supplementary_lanes" in out
    assert out["active_supplementary_lanes"] == []
    assert "exact_ready_queue_audit" in out
    assert out["exact_ready_queue_audit"]["schema"] == (
        "optimizer_exact_ready_queue_terminal_evidence_audit_v1"
    )
    assert out["exact_ready_queue_audit"].get("stale_ready_row_count") == 0
    assert out["exact_ready_queue_audit"].get("suppressed_ready_row_count", 0) >= 1
    assert str(out["exact_ready_queue_audit"].get("suppression_manifest_path", "")).endswith(
        ".omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json"
    )
    assert "non_dispatchable_readiness_artifacts" in out
    supplementary_rows = {row["lane_id"]: row for row in out["supplementary_lanes"]}
    assert all(row["score_claim"] is False for row in supplementary_rows.values())
    assert all(row["score_claim_valid"] is False for row in supplementary_rows.values())
    assert all(row["promotion_eligible"] is False for row in supplementary_rows.values())
    assert all(row["rank_or_kill_eligible"] is False for row in supplementary_rows.values())
    assert supplementary_rows["lane_pr106_latent_sidecar"]["score_target_routing"]["active"] is False
    assert supplementary_rows["lane_pr106_latent_sidecar"]["score_target_routing"]["status"] == "above_target"
    assert supplementary_rows["lane_pr106_latent_sidecar"]["dispatch_routing"]["active"] is False
    assert supplementary_rows["lane_pr106_latent_sidecar"]["dispatch_routing"]["status"] == "score_target_inactive"
    assert supplementary_rows["lane_pr106_latent_sidecar"]["ready_for_operator_dispatch"] is False
    assert supplementary_rows["lane_pr106_latent_sidecar"]["ready_for_exact_eval_dispatch"] is False
    assert out["active_gated_lanes"] == []
    assert out["active_composition_lanes"] == []
    composition_rows = {row["lane_id"]: row for row in out["composition_lanes"]}
    assert all(row["score_claim"] is False for row in composition_rows.values())
    assert all(row["score_claim_valid"] is False for row in composition_rows.values())
    assert all(row["promotion_eligible"] is False for row in composition_rows.values())
    assert all(row["rank_or_kill_eligible"] is False for row in composition_rows.values())
    stacked = composition_rows["lane_pr106_stacked"]
    assert stacked["score_target_routing"]["active"] is True
    assert stacked["dispatch_routing"]["active"] is False
    assert stacked["dispatch_routing"]["status"] == "dispatch_gate_blocked"
    assert stacked["ready_for_operator_dispatch"] is False
    assert stacked["ready_for_exact_eval_dispatch"] is False
    assert "gate_condition_not_satisfied" in stacked["dispatch_routing"]["blockers"]
    assert "operator_one_liner_has_unresolved_placeholders" in stacked["dispatch_routing"]["blockers"]
    row = out["non_dispatchable_readiness_artifacts"][0]
    assert row["kind"] == "pr91_hpm1_readiness_bundle"
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["dispatch_attempted"] is False
    assert row["promotion_eligible"] is False
    assert row["score_claim"] is False
    assert row["artifact_path"].endswith("pr91_hpm1_readiness_20260506_codex/readiness.json")
    assert row["runtime_contract_artifact_path"].endswith(
        "pr91_hpm1_runtime_contract_20260506_codex/runtime_contract.json"
    )
    assert row["readiness_artifact_hash_matches_live"] is True
    assert row["runtime_artifact_hash_matches_live"] is True
    assert row["archive_custody_matches"] is True
    assert row["hpm1_mask_custody_matches"] is True
    assert row["zip_wire_contract_passed"] is True
    assert row["ambient_device_call_count"] >= 1
    assert row["contradiction_count"] >= 1
    assert row["dispatch_blockers"]
    assert row["artifact_dispatch_blockers"]
    assert "not a score or dispatch artifact" in row["summary"]
    packet_rows = {row["lane_id"]: row for row in out["exact_eval_packets"]}
    assert set(packet_rows) >= {
        "wr01_apply_pr106x_half",
        "pr106_q10_151byte_brotli",
        "pr106x_lgblock16_1byte_brotli",
        "hnerv_hlm1_fixed_latent_recode_exact_eval",
        "hnerv_hlm1_xmember_exact_eval_20260514",
    }
    q10 = packet_rows["pr106_q10_151byte_brotli"]
    assert q10["ready_for_submit"] is False
    assert q10["repeat_dispatch_allowed"] is False
    assert q10["dispatch_action"] == "terminal_exact_eval_evidence_stop"
    assert q10["terminal_exact_eval_evidence_blockers"]
    assert any(
        blocker.startswith("same_lane_terminal_negative_for_same_archive")
        or blocker.startswith("same_lane_terminal_score_not_below_active_floor")
        for blocker in q10["terminal_exact_eval_evidence_blockers"]
    )
    assert q10["preflight_ready"] is True
    assert q10["compliance_ok"] is True
    assert q10["payload_diff_ready"] is True
    assert q10["dry_run_ready"] is True
    assert "missing_active_lane_dispatch_claim" in q10["blockers"]
    assert "missing_lightning_environment" in q10["blockers"]
    assert q10["commands"] == {}
    assert "submit" in q10["suppressed_commands"]
    assert q10["operator_next_steps"]["schema"] == "terminal_exact_eval_evidence_stop_v1"
    q10_step_ids = [step["id"] for step in q10["operator_next_steps"]["steps"]]
    assert q10_step_ids == [
        "review_terminal_cuda_result",
        "choose_byte_different_successor_candidate",
    ]
    lgblock16 = packet_rows["pr106x_lgblock16_1byte_brotli"]
    assert lgblock16["ready_for_submit"] is False
    assert lgblock16["preflight_ready"] is True
    assert lgblock16["compliance_ok"] is True
    assert lgblock16["payload_diff_ready"] is True
    assert lgblock16["dry_run_ready"] is True
    assert "missing_active_lane_dispatch_claim" in lgblock16["blockers"]
    if lgblock16["terminal_exact_eval_evidence_blockers"]:
        assert lgblock16["repeat_dispatch_allowed"] is False
        assert lgblock16["commands"] == {}
        assert "submit" in lgblock16["suppressed_commands"]
        assert lgblock16["operator_next_steps"]["schema"] == "terminal_exact_eval_evidence_stop_v1"
    else:
        assert lgblock16["operator_next_steps"]["schema"] == "hnerv_lowlevel_operator_next_steps_v1"
    hlm1 = packet_rows["hnerv_hlm1_fixed_latent_recode_exact_eval"]
    assert hlm1["archive_sha256"] == "8801845d5099b957898fb6c6e58625bfb4cc065085ed2e3154c2cbc702dc91e0"
    assert hlm1["archive_bytes"] == 186423
    packet_path = REPO / (
        "experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/"
        "hlm1_exact_eval_packet.json"
    )
    packet_text = packet_path.read_text(encoding="utf-8")
    hlm1_packet = json.loads(packet_text)
    assert "/Users/adpena" not in packet_text
    assert hlm1_packet["runtime_tree_sha256"] != hlm1_packet["local_runtime_tree_sha256"]
    modal_cpu_runtime_sha = hlm1_packet["runtime_manifest"]["modal_cpu_runtime_tree_sha256"]
    assert modal_cpu_runtime_sha != hlm1_packet["runtime_tree_sha256"]
    if hlm1_packet.get("terminal_exact_eval_evidence_blockers"):
        assert hlm1_packet["ready_for_submit"] is False
        assert hlm1_packet["commands"] == {}
        assert "submit" in hlm1_packet["suppressed_commands"]
        command_surface = hlm1_packet["suppressed_commands"]
    else:
        command_surface = hlm1_packet["commands"]
    assert "submit_contest_cpu" not in command_surface
    assert "paired_dispatch_plan" in command_surface
    assert "submit_paired_cpu_cuda" in command_surface
    assert command_surface["submit"] == command_surface["submit_paired_cpu_cuda"]
    assert "tools/dispatch_modal_paired_auth_eval.py" in command_surface["submit"]
    assert "--expected-runtime-tree-sha256 auto" in command_surface["submit"]
    assert "--skip-axis-if-promotable-anchor-exists" in command_surface["submit"]
    assert "--execute" in command_surface["submit"]
    assert "--execute" not in command_surface["paired_dispatch_plan"]
    assert "experiments/modal_auth_eval.py" not in command_surface["submit"]
    assert "experiments/modal_auth_eval_cpu.py" not in command_surface["submit"]
    refresh_cmd = hlm1_packet["operator_next_steps"]["steps"][0]["copy_safe_command"]
    assert "--operator-approved-exact-cuda" not in refresh_cmd
    assert hlm1_packet["runtime_hlm1_decode_consumption_claim"] is True
    assert hlm1_packet["runtime_hlm1_valid_mutation_changes_raw"] is True
    assert hlm1_packet["artifacts"]["pre_submission_compliance"].endswith(
        "pre_submission_compliance.static_clean.public.json"
    )
    assert hlm1["preflight_ready"] is True
    assert hlm1["compliance_ok"] is True
    assert hlm1["payload_diff_ready"] is True
    assert hlm1["dry_run_ready"] is True
    assert hlm1["score_affecting_runtime_changed"] is True
    packet_step_ids = [step["id"] for step in hlm1_packet["operator_next_steps"]["steps"]]
    if hlm1_packet.get("terminal_exact_eval_evidence_blockers"):
        assert packet_step_ids == [
            "review_terminal_cuda_result",
            "choose_byte_different_successor_candidate",
        ]
    else:
        assert packet_step_ids == [
            "refresh_static_packet_no_dispatch",
            "optional_local_cuda_exact_eval",
            "submit_modal_exact_cuda",
            "harvest_modal_exact_cuda",
            "submit_modal_exact_cpu",
            "harvest_modal_exact_cpu",
        ]
    if hlm1["terminal_exact_eval_evidence_blockers"]:
        assert hlm1["operator_next_steps"]["schema"] == "terminal_exact_eval_evidence_stop_v1"
        hlm1_step_ids = [step["id"] for step in hlm1["operator_next_steps"]["steps"]]
        assert hlm1_step_ids == [
            "review_terminal_cuda_result",
            "choose_byte_different_successor_candidate",
        ]
    else:
        assert hlm1["operator_next_steps"]["schema"] == "hnerv_hlm1_operator_next_steps_v1"

    xmember = packet_rows["hnerv_hlm1_xmember_exact_eval_20260514"]
    assert xmember["archive_sha256"] == (
        "391400008b69e66f8bd522f4eb2a53c465e58a17e536d171caf039f9e51e874f"
    )
    assert xmember["archive_bytes"] == 186415
    assert xmember["ready_for_submit"] is False
    assert xmember["repeat_dispatch_allowed"] is False
    assert xmember["dispatch_action"] == "terminal_exact_eval_evidence_stop"
    assert xmember["commands"] == {}
    assert "submit" in xmember["suppressed_commands"]
    assert xmember["terminal_exact_eval_evidence_blockers"]
    assert any(
        blocker.startswith("same_lane_terminal_score_not_below_active_floor_for_same_archive")
        or blocker.startswith("same_lane_terminal_runtime_mismatch_for_same_archive")
        for blocker in xmember["terminal_exact_eval_evidence_blockers"]
    )


def test_briefing_hides_above_target_rows_by_default_but_can_show_them():
    proc = _run("--skip-dashboard", "--skip-reconciler", "--top", "3")
    assert "lane_pr106_latent_sidecar —" not in proc.stdout
    assert "lane_pr106_yshift_sidechannel —" not in proc.stdout
    assert "hidden inactive/above target 0.1900" in proc.stdout
    assert "lane_pr106_stacked —" not in proc.stdout
    assert "lane_pr106_stacked[dispatch_gate_blocked]" in proc.stdout

    shown = _run("--skip-dashboard", "--skip-reconciler", "--show-above-target", "--top", "3")
    assert "lane_pr106_latent_sidecar —" in shown.stdout
    assert "target routing: above_target" in shown.stdout
    assert "lane_pr106_stacked —" in shown.stdout
    assert "dispatch routing: dispatch_gate_blocked" in shown.stdout


def test_phase_worklist_active_rows_require_dispatch_ready_contract():
    mod = _load_briefing_module()
    groups = [
        mod.PHASE_1_SUPPLEMENTARY_LANES,
        mod.PHASE_4_GATED_LANES,
        mod.PHASE_5_COMPOSITION_LANES,
    ]

    for lanes in groups:
        rows = mod._annotate_score_target_lanes(
            lanes,
            target_score=0.19,
            active_only=False,
        )
        active_rows = mod._annotate_score_target_lanes(
            lanes,
            target_score=0.19,
            active_only=True,
        )
        for row in rows:
            dispatch = row["dispatch_routing"]
            assert row["ready_for_operator_dispatch"] is dispatch["active"]
            if row["ready_for_exact_eval_dispatch"]:
                assert row["ready_for_operator_dispatch"] is True
            if row.get("gate_condition") and row.get("gate_ready") is not True:
                assert dispatch["active"] is False
                assert "gate_condition_not_satisfied" in dispatch["blockers"]
            if "<" in str(row.get("one_liner", "")):
                assert dispatch["active"] is False
                assert "operator_one_liner_has_unresolved_placeholders" in dispatch["blockers"]
        assert {
            row["lane_id"]
            for row in active_rows
        } == {
            row["lane_id"]
            for row in rows
            if row["dispatch_routing"]["active"]
        }


def test_l5_v2_briefing_suppresses_packetir_targets_on_matrix_sha_mismatch(
    tmp_path: Path,
) -> None:
    mod = _load_briefing_module()
    matrix_path = tmp_path / ".omx" / "research" / "matrix.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "next_exact_eval_target_count": 1,
                "next_exact_eval_targets": [
                    {
                        "candidate_id": "stale",
                        "paired_dispatch_tool": "tools/dispatch_modal_paired_auth_eval.py",
                        "command_template": (
                            ".venv/bin/python tools/dispatch_modal_paired_auth_eval.py "
                            "--expected-runtime-tree-sha256 auto "
                            "--skip-axis-if-promotable-anchor-exists"
                        ),
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    mod.REPO_ROOT = tmp_path
    mod.PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH = ".omx/research/matrix.json"
    mod.PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256 = "0" * 64
    mod.l5_v2_canonical_sideinfo_gate_evidence = lambda: None
    mod.l5_v2_dispatch_readiness = lambda gate_evidence=None: {
        "blockers": [],
        "ready_for_gate_probe_dispatch": False,
        "ready_for_score_or_rank_dispatch": False,
        "ready_for_dispatch": False,
        "tt5l_campaign_readiness": {
            "next_non_pr106_l5_action": {"action_id": "materialize_tt5l_proof"},
            "first_anchor_timing_smoke_allowed": False,
        },
        "packetir_stack_evidence": {"paired_candidate_count": 0},
        "pr106_stack_cell_candidates": {"candidate_count": 0},
    }

    l5 = mod._l5_v2_frontier_readiness(dispatch_claim_summary={"active_count": 0})

    assert "l5_v2_packetir_matrix_artifact_sha_mismatch" in l5["blockers"]
    assert l5["packetir_matrix_dispatch_targets_suppressed"] is True
    assert l5["primary_staircase"] == "tt5l_first_non_pr106_l5_v2"
    assert l5["next_non_pr106_l5_action"]["action_id"] == "materialize_tt5l_proof"
    assert l5["next_exact_eval_target_count"] == 0
    assert l5["next_exact_eval_targets"] == []


def test_l5_v2_briefing_blocks_stale_paired_measurement_plan(
    tmp_path: Path,
) -> None:
    mod = _load_briefing_module()
    schedule_path = tmp_path / ".omx" / "research" / "schedule.json"
    plan_path = tmp_path / ".omx" / "research" / "plan.json"
    schedule_path.parent.mkdir(parents=True)
    schedule_path.write_text('{"schema":"changed"}\n', encoding="utf-8")
    stale_sha = "0" * 64
    plan_path.write_text(
        json.dumps(
            {
                "schema": mod.L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_SCHEMA,
                "planning_only": True,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "dispatch_attempted": False,
                "paired_dispatch_tool": "tools/dispatch_modal_paired_auth_eval.py",
                "source_schedule_path": ".omx/research/schedule.json",
                "source_schedule_sha256": stale_sha,
                "work_units": [],
                "work_unit_count": 0,
                "ready_work_unit_count": 0,
                "blockers": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    mod.REPO_ROOT = tmp_path
    mod.L5V2_PAIRED_MEASUREMENT_DISPATCH_PLAN_ARTIFACT_PATH = (
        ".omx/research/plan.json"
    )

    plan = mod._load_l5_v2_paired_measurement_dispatch_plan()

    assert plan["source_schedule_stale"] is True
    assert plan["source_schedule_sha256"] == stale_sha
    assert plan["current_source_schedule_sha256"] != stale_sha
    assert (
        "l5_v2_paired_measurement_dispatch_plan_source_schedule_stale"
        in plan["load_blockers"]
    )


def test_l5_v2_briefing_suppresses_packetir_targets_on_active_claims(
    tmp_path: Path,
) -> None:
    mod = _load_briefing_module()
    matrix_path = tmp_path / ".omx" / "research" / "matrix.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_text = (
        json.dumps(
            {
                "candidate_count": 1,
                "next_exact_eval_target_count": 1,
                "next_exact_eval_targets": [
                    {
                        "candidate_id": "format_0x0c_exact_radix",
                        "lane_id": "pr106_packetir_format_0x0c_exact_radix_contest_cpu",
                        "paired_dispatch_tool": "tools/dispatch_modal_paired_auth_eval.py",
                        "command_template": (
                            ".venv/bin/python tools/dispatch_modal_paired_auth_eval.py "
                            "--expected-runtime-tree-sha256 auto "
                            "--skip-axis-if-promotable-anchor-exists"
                        ),
                        "dispatch_status": (
                            "requires_claim_lane_dispatch_before_provider_launch"
                        ),
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n"
    )
    matrix_path.write_text(matrix_text, encoding="utf-8")
    mod.REPO_ROOT = tmp_path
    mod.PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_PATH = ".omx/research/matrix.json"
    mod.PR106_PACKETIR_CANDIDATE_MATRIX_ARTIFACT_SHA256 = hashlib.sha256(
        matrix_text.encode("utf-8")
    ).hexdigest()
    mod.l5_v2_canonical_sideinfo_gate_evidence = lambda: None
    mod.l5_v2_dispatch_readiness = lambda gate_evidence=None: {
        "blockers": [],
        "ready_for_gate_probe_dispatch": False,
        "ready_for_score_or_rank_dispatch": False,
        "ready_for_dispatch": False,
        "tt5l_campaign_readiness": {
            "next_non_pr106_l5_action": {"action_id": "materialize_tt5l_proof"},
            "first_anchor_timing_smoke_allowed": False,
        },
        "packetir_stack_evidence": {"paired_candidate_count": 0},
        "pr106_stack_cell_candidates": {"candidate_count": 0},
    }

    l5 = mod._l5_v2_frontier_readiness(dispatch_claim_summary={"active_count": 2})

    assert l5["packetir_matrix_dispatch_targets_suppressed"] is True
    assert l5["active_dispatch_claim_count"] == 2
    assert l5["dispatch_claim_gate_blocked"] is True
    assert "blocked_active_dispatch_claims_present:2" in l5["blockers"]
    assert l5["next_exact_eval_target_count"] == 0
    assert l5["next_exact_eval_targets"] == []


def test_briefing_json_skip_pareto_still_surfaces_exact_ready_audit():
    proc = _run("--json", "--skip-pareto", "--top", "3")
    out = json.loads(proc.stdout)

    assert "pareto" not in out
    assert "exact_eval_packets" not in out
    assert "exact_ready_queue_audit" in out
    assert out["exact_ready_queue_audit"]["schema"] == (
        "optimizer_exact_ready_queue_terminal_evidence_audit_v1"
    )
    assert out["exact_ready_queue_audit"].get("stale_ready_row_count") == 0


def test_briefing_json_each_phase_has_n_total_or_n_configs():
    """Each sub-tool must emit a JSON dict with at least one count field."""
    proc = _run("--json", "--top", "3")
    out = json.loads(proc.stdout)
    # Each tool emits its own count fields — ensure at least one is present
    assert any(k in out["pareto"] for k in ("n_configs", "n_pareto_frontier"))
    assert any(k in out["dashboard"] for k in ("n_total", "n_displayed"))
    assert any(k in out["reconciler"] for k in ("n_configs", "n_landed"))
    packet_rows = {row["lane_id"]: row for row in out["exact_eval_packets"]}
    assert packet_rows["wr01_apply_pr106x_half"]["operator_next_steps"]["schema"] == "wr01_operator_next_steps_v1"
    wr01_step_ids = [
        step["id"]
        for step in packet_rows["wr01_apply_pr106x_half"]["operator_next_steps"]["steps"]
    ]
    q10_step_ids = [
        step["id"]
        for step in packet_rows["pr106_q10_151byte_brotli"]["operator_next_steps"]["steps"]
    ]
    lgblock16_step_ids = [
        step["id"]
        for step in packet_rows["pr106x_lgblock16_1byte_brotli"]["operator_next_steps"]["steps"]
    ]
    hlm1 = packet_rows["hnerv_hlm1_fixed_latent_recode_exact_eval"]
    xmember = packet_rows["hnerv_hlm1_xmember_exact_eval_20260514"]
    hlm1_step_ids = [
        step["id"]
        for step in hlm1["operator_next_steps"]["steps"]
    ]
    xmember_step_ids = [
        step["id"]
        for step in xmember["operator_next_steps"]["steps"]
    ]
    assert "assert_packet_ready_for_submit" in wr01_step_ids
    assert q10_step_ids == [
        "review_terminal_cuda_result",
        "choose_byte_different_successor_candidate",
    ]
    if packet_rows["pr106x_lgblock16_1byte_brotli"]["terminal_exact_eval_evidence_blockers"]:
        assert lgblock16_step_ids == [
            "review_terminal_cuda_result",
            "choose_byte_different_successor_candidate",
        ]
    else:
        assert "submit_exact_cuda" in lgblock16_step_ids
    if hlm1["terminal_exact_eval_evidence_blockers"]:
        assert hlm1_step_ids == [
            "review_terminal_cuda_result",
            "choose_byte_different_successor_candidate",
        ]
    else:
        assert "submit_modal_paired_cpu_cuda" in hlm1_step_ids
    assert xmember_step_ids == [
        "review_terminal_cuda_result",
        "choose_byte_different_successor_candidate",
    ]
    assert out["non_dispatchable_readiness_artifacts"][0]["score_claim"] is False


def test_pr106_latent_operator_oneliner_uses_score_table_env() -> None:
    mod = _load_briefing_module()
    rows = {
        row["lane_id"]: row
        for row in mod.PHASE_1_SUPPLEMENTARY_LANES
    }
    one_liner = rows["lane_pr106_latent_sidecar"]["one_liner"]

    assert "--env PR106_LATENT_MODE=score_table" in one_liner
    assert "--env PR106_LATENT_SCORE_TABLE_RESUME=1" in one_liner
    assert "--label lane_pr106_latent_sidecar" in one_liner
    assert rows["lane_pr106_latent_sidecar"]["kaggle_bundle_tool"] == "tools/kaggle_build_pr106_latent_score_table.py"
    assert rows["lane_pr106_latent_sidecar"]["kaggle_harvest_tool"] == "tools/harvest_kaggle_pr106_latent_score_table.py"
    assert rows["lane_pr106_latent_sidecar"]["kaggle_kernel_slug"] == "adpena/comma-lab-pr106-latent-score-table"


def test_pr106_yshift_operator_row_surfaces_kaggle_score_table_tools() -> None:
    mod = _load_briefing_module()
    rows = {
        row["lane_id"]: row
        for row in mod.PHASE_4_GATED_LANES
    }

    assert rows["lane_pr106_yshift_sidechannel"]["kaggle_bundle_tool"] == "tools/kaggle_build_pr106_yshift_score_table.py"
    assert rows["lane_pr106_yshift_sidechannel"]["kaggle_harvest_tool"] == "tools/harvest_kaggle_pr106_yshift_score_table.py"
    assert rows["lane_pr106_yshift_sidechannel"]["kaggle_kernel_slug"] == "adpena/comma-lab-pr106-yshift-score-table"


def test_pr91_readiness_row_surfaces_audit_errors(monkeypatch):
    mod = _load_briefing_module()
    manifest = {"canonical_payload_without_tool_manifest_sha256": "not-recomputed-in-this-test"}
    closed_payload = {
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "tool_run_manifest": manifest,
        "dispatch_blockers": ["blocked"],
    }

    def fake_run_json(script, extra_args=None):
        if script == mod.PR91_HPM1_READINESS:
            return {"_error": "live readiness audit failed"}
        return dict(closed_payload)

    monkeypatch.setattr(mod, "_run_json", fake_run_json)
    monkeypatch.setattr(mod, "_load_json_file", lambda path: dict(closed_payload))
    monkeypatch.setattr(mod, "_canonical_payload_hash", lambda payload: "same")
    monkeypatch.setattr(mod, "_manifest_hash_self_consistent", lambda payload: True)

    row = mod._load_pr91_hpm1_readiness_artifact()

    assert row["state"] == "AUDIT_ERROR_FAIL_CLOSED"
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_claim"] is False
    assert row["audit_errors"] == ["live readiness audit failed"]


def test_operator_briefing_surfaces_inverse_scorer_chain_readiness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    scan_root = tmp_path / "results"
    chain_dir = scan_root / "parent" / "ias1_chain"
    chain_dir.mkdir(parents=True)
    manifest = chain_dir / mod.INVERSE_SCORER_CHAIN_MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "schema": "inverse_scorer_cell_candidate_chain_v1",
                "candidate_archive_sha256": "a" * 64,
                "candidate_archive_bytes": 123,
                "candidate_archive": {"sha256": "a" * 64, "bytes": 123},
                "runtime_adapter_ready": True,
                "receiver_proof_ready": True,
                "receiver_contract_satisfied": True,
                "inflate_parity_satisfied": False,
                "candidate_runtime_adapter_blocker_cleared": True,
                "full_frame_or_shell_parity_required": True,
                "next_required_gates": ["inflate_or_full_frame_parity", "contest_auth_eval"],
                "readiness_blockers": [
                    "candidate_inflate_output_parity_missing",
                    "exact_auth_eval_required_before_score_claim",
                ],
                "dispatch_blockers": [
                    "inverse_scorer_cell_candidate_chain_is_not_dispatch_authorization",
                    "candidate_inflate_output_parity_missing",
                    "exact_auth_eval_required_before_score_claim",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "INVERSE_SCORER_CHAIN_SCAN_ROOT", scan_root)
    monkeypatch.setattr(
        mod,
        "_load_pr91_hpm1_readiness_artifact",
        lambda: {
            "kind": "pr91_hpm1_readiness_bundle",
            "name": "fixture",
            "state": "BLOCKED_FAIL_CLOSED",
            "dispatch_blockers": ["blocked"],
            "artifact_dispatch_blockers": ["blocked"],
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "promotion_eligible": False,
            "audit_errors": [],
            "artifact_path": "readiness.json",
            "runtime_contract_artifact_path": "runtime.json",
            "readiness_artifact_hash_matches_live": True,
            "runtime_artifact_hash_matches_live": True,
            "archive_custody_matches": True,
            "hpm1_mask_custody_matches": True,
            "zip_wire_contract_passed": True,
            "ambient_device_call_count": 0,
            "contradiction_count": 0,
            "summary": "fixture",
            "next_patch": "fixture",
        },
    )

    rows = mod._inverse_scorer_cell_chain_readiness_artifacts()
    text = mod._format_non_dispatchable_readiness_artifacts()

    assert rows[0]["kind"] == "inverse_scorer_cell_candidate_chain"
    assert rows[0]["receiver_contract_satisfied"] is True
    assert rows[0]["inflate_parity_satisfied"] is False
    assert rows[0]["ready_for_exact_eval_dispatch"] is False
    assert rows[0]["score_claim"] is False
    assert "candidate_inflate_output_parity_missing" in rows[0]["dispatch_blockers"]
    assert "inverse_scorer_cell_candidate_chain" in text
    assert "inflate_or_full_frame_parity" in text


def test_dispatch_claim_summary_formats_active_claim(monkeypatch):
    mod = _load_briefing_module()
    monkeypatch.setattr(
        mod,
        "_dispatch_claim_summary",
        lambda: {
            "schema": "pact.dispatch_claim_summary.v1",
            "active_count": 1,
            "stale_nonterminal_count": 0,
            "terminal_latest_count": 3,
            "unparsable_timestamp_count": 1,
            "invalid_lane_id_count": 1,
            "active": [
                {
                    # FAKE_LANE_OK: synthetic operator-briefing formatter fixture.
                    "lane_id": "lane_a1_cuda",
                    "instance_job_id": "job-123",
                    "platform": "lightning",
                    "status": "eval",
                    "agent": "codex",
                }
            ],
            "stale_nonterminal": [],
            "unparsable_timestamp": [
                {
                    "lane_id": "lane_bad_time",
                    "instance_job_id": "bad-time-job",
                    "timestamp_utc": "not-a-time",
                    "status": "failed_modal_smoke_red",
                }
            ],
            "invalid_lane_id": [
                {
                    "lane_id": "0",
                    "instance_job_id": "bad-job",
                    "status": "refused_dispatch_bad_claim",
                }
            ],
        },
    )
    monkeypatch.setattr(
        mod,
        "_dispatch_claim_historical_summary",
        lambda: {
            "schema": "pact.dispatch_claim_summary.v1",
            "invalid_lane_id_count": 4,
            "unparsable_timestamp_count": 0,
        },
    )

    text = mod._format_dispatch_claim_summary()

    assert "unparsable_timestamp: 1" in text
    assert "UNPARSABLE TIMESTAMPS" in text
    assert "lane_bad_time" in text
    assert "invalid_lane_id: 1" in text
    assert "INVALID LANE IDS" in text
    assert "All-history claim hygiene: WARNING" in text
    assert "invalid_lane_id=4" in text
    assert "lane_id=0" in text
    assert "ACTIVE CONFLICT GUARD" in text
    assert "lane_id=lane_a1_cuda" in text
    assert "job=job-123" in text
    assert "platform=lightning" in text


def test_dispatch_readiness_blocks_when_every_exact_packet_is_terminal(monkeypatch):
    mod = _load_briefing_module()
    monkeypatch.setattr(
        mod,
        "_exact_ready_queue_audit",
        lambda: {"stale_ready_row_count": 0},
    )
    monkeypatch.setattr(
        mod,
        "_exact_eval_packet_summaries",
        lambda: [
            {
                "lane_id": "terminal_packet",
                "ready_for_submit": False,
                "dispatch_action": "terminal_exact_eval_evidence_stop",
            }
        ],
    )

    readiness = mod._dispatch_readiness()
    text = mod._format_dispatch_readiness()

    assert readiness["phase_1_exact_eval_packets"]["status"] == "BLOCKED"
    assert "all exact-eval packets are blocked or terminal" in text
    assert "Phase 1 (pre-dispatch Pareto):              READY" not in text


def test_exact_eval_packet_ready_for_submit_requires_all_static_and_env_gates(
    tmp_path,
    monkeypatch,
):
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "DISPATCH_CLAIMS", tmp_path / "missing_claims.md")
    (tmp_path / "packet.json").write_text(
        json.dumps(
            {
                "ready_for_submit": True,
                "preflight_ready": False,
                "static_blockers": ["static gate not run"],
                "compliance_ok": False,
                "payload_diff_ready": True,
                "dry_run_ready": True,
                "missing_env": ["LIGHTNING_API_KEY"],
                "commands": {"submit": "do-not-copy"},
            }
        ),
        encoding="utf-8",
    )

    packet = mod._load_exact_eval_packet(
        {
            "lane_id": "static_env_fixture",
            "name": "static/env fixture",
            "packet_path": "packet.json",
        }
    )

    assert packet["ready_for_submit"] is False
    assert packet["dispatch_action"] == "blocked_static_or_env_gates"
    assert "static_preflight_not_ready" in packet["submit_gate_blockers"]
    assert "static_compliance_not_ok" in packet["submit_gate_blockers"]
    assert "missing_submit_environment" in packet["submit_gate_blockers"]
    assert packet["missing_env"] == ["LIGHTNING_API_KEY"]


def test_dispatch_readiness_does_not_mark_predicted_phase7_rollup_ready(
    tmp_path,
    monkeypatch,
):
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "_exact_ready_queue_audit",
        lambda: {"stale_ready_row_count": 0},
    )
    monkeypatch.setattr(mod, "_exact_eval_packet_summaries", lambda: [])
    rollup = (
        tmp_path
        / "experiments/results/constrained_coord_search_pr101_bias_20260509T142645Z/rollup.json"
    )
    rollup.parent.mkdir(parents=True, exist_ok=True)
    rollup.write_text(
        json.dumps(
            {
                "lane_id": "lane_pr101_bias_constrained_coord_search",
                "evidence_grade": "[predicted; constrained coord search on A1 substrate]",
                "n_variants": 64,
                "n_unique_inflates": 64,
                "dispatch_blockers": [
                    "claim lane lane_pr101_bias_constrained_coord_search before dispatch",
                    "M5 Max coarse rank should run first ($0); promote top-5 to GHA",
                ],
            }
        ),
        encoding="utf-8",
    )

    readiness = mod._dispatch_readiness()
    text = mod._format_dispatch_readiness()

    phase7 = readiness["phase_7_constrained_coord_search"]
    assert phase7["status"] == "BLOCKED"
    assert phase7["n_rollups"] == 1
    assert "latest rollup is not dispatch-ready" in phase7["reason"]
    assert "Phase 7 (constrained-coord-search):         BLOCKED" in text


def test_provider_readiness_formatter_preserves_proxy_boundary(monkeypatch):
    mod = _load_briefing_module()
    monkeypatch.setattr(
        mod,
        "_provider_readiness",
        lambda refresh=False: {
            "schema": "cloud_provider_readiness_v1",
            "generated_at_utc": "2026-05-10T00:00:00Z",
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "artifact_path": "experiments/results/cloud_provider_readiness_latest.json",
            "providers": [
                {
                    "provider": "kaggle",
                    "status": "ready_proxy",
                    "exact_cuda_evidence_allowed": False,
                    "proxy_only": True,
                    "blockers": [],
                }
            ],
        },
    )

    text = mod._format_provider_readiness()

    assert "not a dispatch or score claim" in text
    assert "score_claim:      False" in text
    assert "exact_dispatch:   False" in text
    assert "kaggle: ready_proxy" in text
    assert "proxy_only=True" in text
