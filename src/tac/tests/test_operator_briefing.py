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
import os
import subprocess
import sys
from pathlib import Path

from comma_lab.scheduler.experiment_queue import (
    connect_state,
    initialize_queue_state,
    load_queue_definition,
)

REPO = Path(__file__).resolve().parents[3]
BRIEFING = REPO / "tools" / "operator_briefing.py"
RECOVERY_TOOL = REPO / "tools" / "materialize_byte_shaving_queue_recovery.py"


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


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
    assert "Phase 1 materializer exact-ready handoffs" in proc.stdout
    assert "Phase 1 blocked readiness artifacts" in proc.stdout
    assert "Phase 6c — High-level byte-shaving acquisition queue" in proc.stdout
    assert "Phase 6e — PR95 MLX control profiles" in proc.stdout
    assert "Phase 6f — Distortion-axis scorer probe signals" in proc.stdout
    assert "Phase 6g — DQS1 drop-many greedy reducer" in proc.stdout
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
    pr95_mlx = out["pr95_mlx_control_profiles"]
    assert pr95_mlx["schema"] == "pact.pr95_mlx_control_profile_summary.v1"
    assert pr95_mlx["score_claim"] is False
    assert pr95_mlx["promotion_eligible"] is False
    assert pr95_mlx["rank_or_kill_eligible"] is False
    assert pr95_mlx["ready_for_exact_eval_dispatch"] is False
    distortion = out["distortion_axis_probe_signals"]
    assert distortion["schema"] == "pact.distortion_axis_probe_summary.v1"
    assert distortion["score_claim"] is False
    assert distortion["promotion_eligible"] is False
    assert distortion["rank_or_kill_eligible"] is False
    assert distortion["ready_for_exact_eval_dispatch"] is False
    assert out["dispatch_readiness"][
        "phase_6f_distortion_axis_probe_signals"
    ]["score_claim"] is False
    distortion_sweep = out["distortion_axis_learned_sweep_bridge"]
    assert distortion_sweep["schema"] == (
        "pact.distortion_axis_learned_sweep_bridge_summary.v1"
    )
    assert distortion_sweep["score_claim"] is False
    assert distortion_sweep["promotion_eligible"] is False
    assert distortion_sweep["rank_or_kill_eligible"] is False
    assert distortion_sweep["ready_for_exact_eval_dispatch"] is False
    assert out["dispatch_readiness"][
        "phase_6h_distortion_axis_learned_sweep_bridge"
    ]["score_claim"] is False
    assert "feedback_observation_count" in out["dispatch_readiness"][
        "phase_6h_distortion_axis_learned_sweep_bridge"
    ]
    dqs1_greedy = out["dqs1_drop_many_greedy"]
    assert dqs1_greedy["schema"] == "pact.dqs1_drop_many_greedy_summary.v1"
    assert dqs1_greedy["tool_exists"] is True
    assert dqs1_greedy["score_claim"] is False
    assert dqs1_greedy["promotion_eligible"] is False
    assert dqs1_greedy["rank_or_kill_eligible"] is False
    assert dqs1_greedy["ready_for_exact_eval_dispatch"] is False
    assert out["dispatch_readiness"]["phase_6g_dqs1_drop_many_greedy"][
        "score_claim"
    ] is False
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
    stale_ready_rows = out["exact_ready_queue_audit"].get("stale_ready_row_count")
    assert isinstance(stale_ready_rows, int)
    if stale_ready_rows:
        assert (
            out["dispatch_readiness"]["phase_1_exact_ready_queue_hygiene"]["status"]
            == "BLOCKED"
        )
    assert out["exact_ready_queue_audit"].get("suppressed_ready_row_count", 0) >= 1
    assert str(out["exact_ready_queue_audit"].get("suppression_manifest_path", "")).endswith(
        ".omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json"
    )
    assert "materializer_exact_ready_handoffs" in out
    assert out["materializer_exact_ready_handoffs"]["schema"] == (
        "pact.materializer_exact_ready_handoff_summary.v1"
    )
    assert out["materializer_exact_ready_handoffs"]["score_claim"] is False
    assert (
        out["dispatch_readiness"]["phase_1_materializer_exact_ready_handoffs"][
            "schema"
        ]
        == "pact.materializer_exact_ready_handoff_summary.v1"
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
    mod = _load_briefing_module()

    hidden = "\n\n".join(
        [
            mod._format_supplementary_lanes(),
            mod._format_gated_lanes(),
            mod._format_composition_lanes(),
        ]
    )
    assert "lane_pr106_latent_sidecar —" not in hidden
    assert "lane_pr106_yshift_sidechannel —" not in hidden
    assert "hidden inactive/above target 0.1900" in hidden
    assert "lane_pr106_stacked —" not in hidden
    assert "lane_pr106_stacked[dispatch_gate_blocked]" in hidden

    shown = "\n\n".join(
        [
            mod._format_supplementary_lanes(show_above_target=True),
            mod._format_gated_lanes(show_above_target=True),
            mod._format_composition_lanes(show_above_target=True),
        ]
    )
    assert "lane_pr106_latent_sidecar —" in shown
    assert "target routing: above_target" in shown
    assert "lane_pr106_stacked —" in shown
    assert "dispatch routing: dispatch_gate_blocked" in shown


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
    stale_ready_rows = out["exact_ready_queue_audit"].get("stale_ready_row_count")
    assert isinstance(stale_ready_rows, int)
    if stale_ready_rows:
        assert (
            out["dispatch_readiness"]["phase_1_exact_ready_queue_hygiene"]["status"]
            == "BLOCKED"
        )
    assert out["materializer_exact_ready_handoffs"]["schema"] == (
        "pact.materializer_exact_ready_handoff_summary.v1"
    )


def test_materializer_exact_ready_handoff_summary_surfaces_queue_owned_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    tool = tmp_path / "tools" / "build_materializer_exact_eval_consumer.py"
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    root = tmp_path / "experiments" / "results"
    exact_ready_path = "experiments/results/run/candidate.exact_ready_queue.json"
    _write_json(
        root / "run" / "exact_readiness_bridge_report.json",
        {
            "schema": "materializer_chain_exact_readiness_bridge_report.v1",
            "ready_candidate_count": 2,
            "blocked_candidate_count": 1,
            "rows": [
                {
                    "candidate_id": "a",
                    "exact_ready_queue_written": True,
                    "exact_ready_queue_path": exact_ready_path,
                },
                {
                    "candidate_id": "b",
                    "exact_ready_queue_written": True,
                    "exact_ready_queue_path": exact_ready_path,
                },
                {"candidate_id": "c", "exact_ready_queue_written": False},
            ],
        },
    )
    _write_json(
        root / "run" / "consumer_report.json",
        {
            "schema": "materializer_exact_eval_consumer.v1",
            "authorized_candidate_count": 1,
            "blocked_candidate_count": 1,
            "duplicate_candidate_count": 1,
            "experiment_queue_id": "consumer_queue",
            "hard_plan_blockers": [],
            "rows": [
                {
                    "candidate_id": "a",
                    "blockers": ["exact_dispatch_authority:not_authorized"],
                    "exact_ready_queue_path": exact_ready_path,
                }
            ],
        },
    )
    _write_json(
        root / "run" / "materializer_exact_eval_consumer_live.experiment_queue.json",
        {
            "schema": "experiment_queue.v1",
            "queue_id": "consumer_queue",
            "experiments": [{"id": "dry_run"}],
        },
    )
    _write_json(
        root / "run" / "dispatch_plan.json",
        {
            "schema": "materializer_exact_eval_dispatch_plan.v1",
            "authorized_candidate_count": 1,
            "blocked_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "dispatch_mode": "dry_run",
            "experiment_queue_id": "dispatch_queue",
            "plan_blockers": [],
            "hard_plan_blockers": [],
        },
    )
    monkeypatch.setattr(mod, "MATERIALIZER_HANDOFF_SCAN_ROOTS", (root,))

    summary = mod._materializer_exact_ready_handoff_summary()
    text = mod._format_materializer_exact_ready_handoffs()

    assert summary["status"] == "READY"
    assert summary["bridge_report_count"] == 1
    assert summary["bridge_ready_candidate_count"] == 2
    assert summary["consumer_authorized_candidate_count"] == 1
    assert summary["dispatch_plan_authorized_candidate_count"] == 1
    assert summary["consumer_duplicate_candidate_count"] == 1
    assert summary["consumer_experiment_queue_count"] == 1
    assert summary["consumer_tool_exists"] is True
    assert summary["discoverability_status"] == "VISIBLE"
    assert summary["top_blockers"] == ["exact_dispatch_authority:not_authorized"]
    assert set(summary["recent_consumer_output_paths"]) == {
        "experiments/results/run/materializer_exact_eval_consumer_live.experiment_queue.json",
        "experiments/results/run/consumer_report.json",
    }
    assert summary["recent_exact_ready_queue_paths"] == [exact_ready_path]
    assert "tools/build_materializer_exact_eval_consumer.py" in summary["next_command"]
    assert "--bridge-report experiments/results/run/exact_readiness_bridge_report.json" in summary["next_command"]
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert "authorized=1" in text
    assert "recent consumer outputs:" in text
    assert "first blockers:" in text
    assert exact_ready_path in text
    assert "next command:" in text


def test_materializer_exact_ready_handoff_summary_gives_next_command_without_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / "experiments" / "results"
    exact_ready_path = "experiments/results/run/candidate.exact_ready_queue.json"
    _write_json(
        root / "run" / "exact_readiness_bridge_report.json",
        {
            "schema": "materializer_chain_exact_readiness_bridge_report.v1",
            "rows": [
                {
                    "candidate_id": "a",
                    "exact_ready_queue_written": True,
                    "exact_ready_queue_path": exact_ready_path,
                }
            ],
        },
    )
    monkeypatch.setattr(mod, "MATERIALIZER_HANDOFF_SCAN_ROOTS", (root,))

    summary = mod._materializer_exact_ready_handoff_summary()

    assert summary["consumer_report_count"] == 0
    assert summary["recent_consumer_output_paths"] == []
    assert summary["recent_exact_ready_queue_paths"] == [exact_ready_path]
    assert summary["discoverability_status"] == "NEXT_COMMAND_AVAILABLE"
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["rank_or_kill_eligible"] is False
    assert (
        "--bridge-report experiments/results/run/exact_readiness_bridge_report.json"
        in summary["next_command"]
    )
    assert "materializer_exact_eval_consumer_report_${UTC}.json" in summary["next_command"]


def test_materializer_exact_ready_handoff_summary_blocks_hard_plan_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / "experiments" / "results"
    _write_json(
        root / "run" / "dispatch_plan.json",
        {
            "schema": "materializer_exact_eval_dispatch_plan.v1",
            "authorized_candidate_count": 1,
            "blocked_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "experiment_queue_id": "dispatch_queue",
            "hard_plan_blockers": ["claim_missing"],
            "rows": [],
        },
    )
    monkeypatch.setattr(mod, "MATERIALIZER_HANDOFF_SCAN_ROOTS", (root,))

    summary = mod._materializer_exact_ready_handoff_summary()

    assert summary["status"] == "BLOCKED"
    assert summary["dispatch_plan_authorized_candidate_count"] == 1
    assert summary["hard_plan_blocker_count"] == 1
    assert summary["top_blockers"] == ["claim_missing"]


def test_materializer_exact_ready_handoff_summary_supersedes_old_queue_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / "experiments" / "results"
    old_report = _write_json(
        root / "run" / "materializer_exact_eval_consumer_report_20260101T000000Z.json",
        {
            "schema": "materializer_exact_eval_consumer.v1",
            "authorized_candidate_count": 9,
            "blocked_candidate_count": 4,
            "duplicate_candidate_count": 3,
            "experiment_queue_id": "same_queue",
            "rows": [
                {
                    "candidate_id": "same_candidate",
                    "exact_ready_queue_path": "experiments/results/run/same.json",
                }
            ],
        },
    )
    new_report = _write_json(
        root / "run" / "materializer_exact_eval_consumer_report_20260101T010000Z.json",
        {
            "schema": "materializer_exact_eval_consumer.v1",
            "authorized_candidate_count": 2,
            "blocked_candidate_count": 1,
            "duplicate_candidate_count": 0,
            "experiment_queue_id": "same_queue",
            "rows": [
                {
                    "candidate_id": "same_candidate",
                    "exact_ready_queue_path": "experiments/results/run/same.json",
                }
            ],
        },
    )
    os.utime(old_report, ns=(1_000, 1_000))
    os.utime(new_report, ns=(2_000, 2_000))
    monkeypatch.setattr(mod, "MATERIALIZER_HANDOFF_SCAN_ROOTS", (root,))

    summary = mod._materializer_exact_ready_handoff_summary()

    assert summary["consumer_report_count"] == 1
    assert summary["consumer_authorized_candidate_count"] == 2
    assert summary["consumer_blocked_candidate_count"] == 1
    assert summary["consumer_duplicate_candidate_count"] == 0
    assert summary["scanned_handoff_artifact_count"] == 2
    assert summary["superseded_handoff_artifact_count"] == 1
    assert summary["recent_consumer_output_paths"] == [
        "experiments/results/run/materializer_exact_eval_consumer_report_20260101T010000Z.json"
    ]
    assert summary["recent_exact_ready_queue_paths"] == ["experiments/results/run/same.json"]


def test_byte_shaving_acquisition_summary_surfaces_latest_local_queue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / ".omx" / "research"
    campaign_dir = root / "high_level" / "campaign3"
    plan_path = campaign_dir / "byte_shaving_campaign_plan.json"
    queue_path = ".omx/research/high_level/campaign3/materializer_execution_queue.json"
    state_path = ".omx/state/experiment_queue_high_level_fixture_campaign3.sqlite"

    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "high_level_fixture_campaign3",
        "controls": {
            "mode": "running",
            "max_concurrency": {"local_mlx": 1},
        },
        "experiments": [
            {
                "id": "exp0",
                "status": "queued",
                "steps": [
                    {
                        "id": "materialize_local_proof_chain",
                        "kind": "command",
                        "command": ["python", "-c", "print('ready')"],
                        "resources": {"kind": "local_mlx"},
                    }
                ],
            }
        ],
    }
    queue = load_queue_definition(_write_json(tmp_path / queue_path, queue))
    with connect_state(tmp_path / state_path) as conn:
        initialize_queue_state(conn, queue)
    _write_json(
        plan_path,
        {
            "campaign_id": "high_level_fixture",
            "candidate_id": "inverse_steganalysis_water_bucket_plan.v1",
            "dispatch_blockers": ["requires_exact_auth_eval_before_score_claim"],
            "materialization_bridge": {
                "next_gate": "build_byte_shaving_campaign_queue_packet_ir_lowering",
                "high_level_operation_compiler_required_count": 1,
                "packet_ir_operation_set_count": 2,
                "packet_ir_byte_closed_operation_count": 3,
                "queue_consumable_packet_ir_operation_set_count": 1,
                "dispatch_blockers": [
                    "packet_ir_operation_sets_require_materializer_contexts_and_runtime_proofs"
                ],
            },
            "combination_ladder": [
                {
                    "combo_id": "combo_0001",
                    "expected_score_gain": 0.000134,
                    "unit_count": 2,
                    "operation_families": [
                        "materialize_inverse_scorer_cell_candidate"
                    ],
                    "dispatch_blockers": [
                        "requires_byte_closed_materialization_before_dispatch"
                    ],
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        campaign_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "execute": False,
            "high_level_action_source_count": 2,
            "plan": ".omx/research/high_level/campaign3/byte_shaving_campaign_plan.json",
            "queue_id": "high_level_fixture_campaign3",
            "queue_path": queue_path,
            "queue_feedback_replan_ready": True,
            "queue_feedback_replan_followup_queue_emitted": True,
            "queue_feedback_replan_followup_queue_path": (
                ".omx/research/high_level/campaign3/queue_feedback_replan_followup_queue.json"
            ),
            "queue_feedback_replan_followup_queue_blockers": [],
            "queue_feedback_replan_followup_policy": "local_autopilot_policy",
            "queue_feedback_replan_followup_policy_enabled": True,
            "queue_feedback_replan_followup_policy_blockers": [],
            "queue_feedback_replan_followup_execution_requested": True,
            "queue_feedback_replan_followup_executed": True,
            "queue_feedback_replan_followup_execution_success": True,
            "queue_feedback_replan_policy_path": (
                ".omx/research/high_level/campaign3/queue_feedback_replan_policy.json"
            ),
            "queue_feedback_replan_policy_decision": (
                "run_next_materializer_campaign_iteration"
            ),
            "queue_feedback_replan_policy_should_continue": True,
            "queue_feedback_replan_policy": {
                "schema": "queue_feedback_replan_policy.v1",
                "decision": "run_next_materializer_campaign_iteration",
                "should_continue_feedback_loop": True,
                "blockers": [],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "queue_feedback_replan_continuation_queue_path": (
                ".omx/research/high_level/campaign3/"
                "queue_feedback_replan_continuation_queue.json"
            ),
            "queue_feedback_replan_continuation_queue_emitted": True,
            "queue_feedback_replan_continuation_queue_blockers": [],
            "exact_readiness_handoff_count": 1,
            "state_path": state_path,
            "experiment_count": 3,
            "build": {
                "materializer_work_queue_executable_row_count": 2,
                "materializer_work_queue_blocked_row_count": 1,
                "blocked_row_count": 1,
                "materializer_backlog_row_count": 2,
                "local_cpu_concurrency": 18,
            },
            "worker": {
                "schema": "experiment_queue_worker_result.v1",
                "execute": False,
                "failure_count": 0,
                "success_count": 0,
                "max_parallel": 19,
                "stop_reason": "dry_run",
            },
            "observation": {
                "status_counts": {"queued": 3},
                "ready_steps": [
                    {
                        "step_id": "materialize_local_proof_chain",
                        "resource_kind": "local_mlx",
                    }
                ],
                "failed_steps": [],
                "definition_drift": {
                    "changed_step_count": 0,
                    "missing_step_count": 0,
                    "missing_hash_step_count": 0,
                },
            },
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    text = mod._format_byte_shaving_acquisition_summary()

    assert summary["status"] == "READY_LOCAL_QUEUE"
    assert summary["campaign_run_count"] == 1
    assert summary["total_experiment_count"] == 3
    assert summary["total_executable_work_count"] == 2
    assert summary["local_mlx_ready_step_count"] == 1
    assert summary["total_compiler_required_count"] == 1
    assert summary["total_packet_ir_operation_set_count"] == 2
    assert summary["total_queue_consumable_packet_ir_operation_set_count"] == 1
    assert summary["total_packet_ir_byte_closed_operation_count"] == 3
    assert summary["total_exact_readiness_handoff_count"] == 1
    assert summary["queue_feedback_ready_count"] == 1
    assert summary["queue_feedback_followup_queue_count"] == 1
    assert summary["queue_feedback_followup_policy_enabled_count"] == 1
    assert summary["queue_feedback_followup_executed_count"] == 1
    assert summary["queue_feedback_followup_execution_success_count"] == 1
    assert summary["queue_feedback_policy_continue_count"] == 1
    assert summary["queue_observation_recovery_queue_count"] == 0
    assert summary["queue_observation_recovery_executed_count"] == 0
    assert summary["queue_observation_recovery_execution_success_count"] == 0
    assert summary["queue_observation_recovery_grouped_blocker_count"] == 0
    assert summary["queue_observation_recovery_repeated_group_count"] == 0
    assert summary["queue_feedback_continuation_queue_count"] == 1
    assert summary["overall_executable_conversion_rate"] == 0.5
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert queue_path in summary["next_command"]
    assert "--state .omx/state/experiment_queue_high_level_fixture_campaign3.sqlite" in (
        summary["next_command"]
    )
    assert "--state .omx/state/experiment_queue_high_level_fixture_campaign3.sqlite" in (
        summary["observe_command"]
    )
    assert "--execute --max-parallel 0" in summary["next_command"]
    assert summary["latest_rows"][0]["plan"]["top_combo"]["expected_score_gain"] == 0.000134
    assert summary["latest_rows"][0]["compiler_required_count"] == 1
    assert summary["latest_rows"][0]["packet_ir_operation_set_count"] == 2
    assert summary["latest_rows"][0]["queue_consumable_packet_ir_operation_set_count"] == 1
    assert summary["latest_rows"][0]["exact_readiness_handoff_count"] == 1
    assert summary["latest_rows"][0]["queue_feedback_replan_ready"] is True
    assert summary["latest_rows"][0]["queue_feedback_replan_followup_queue_emitted"] is True
    assert summary["latest_rows"][0]["queue_feedback_replan_followup_policy"] == (
        "local_autopilot_policy"
    )
    assert summary["latest_rows"][0]["queue_feedback_replan_followup_policy_enabled"] is True
    assert summary["latest_rows"][0]["queue_feedback_replan_followup_policy_blocker_count"] == 0
    assert summary["latest_rows"][0]["queue_feedback_replan_followup_executed"] is True
    assert (
        summary["latest_rows"][0]["queue_feedback_replan_followup_execution_success"]
        is True
    )
    assert summary["latest_rows"][0]["queue_feedback_replan_policy_path"] == (
        ".omx/research/high_level/campaign3/queue_feedback_replan_policy.json"
    )
    assert summary["latest_rows"][0]["queue_feedback_replan_policy_decision"] == (
        "run_next_materializer_campaign_iteration"
    )
    assert summary["latest_rows"][0]["queue_feedback_replan_policy_should_continue"] is True
    assert summary["latest_rows"][0]["queue_observation_recovery_required"] is False
    assert (
        summary["latest_rows"][0]["queue_observation_maintenance_recommended"]
        is False
    )
    assert summary["latest_rows"][0]["ready_for_queue_health_recovery"] is False
    assert summary["latest_rows"][0]["queue_feedback_replan_policy_blocker_count"] == 0
    assert summary["latest_rows"][0]["queue_feedback_replan_continuation_queue_path"] == (
        ".omx/research/high_level/campaign3/"
        "queue_feedback_replan_continuation_queue.json"
    )
    assert summary["latest_rows"][0]["queue_feedback_replan_continuation_queue_emitted"] is True
    assert summary["latest_rows"][0]["queue_feedback_replan_continuation_queue_blocker_count"] == 0
    assert summary["latest_rows"][0]["queue_observation_source"] == "live"
    assert summary["latest_rows"][0]["live_queue_observation_used"] is True
    assert summary["latest_rows"][0]["live_queue_observation_healthy"] is True
    assert summary["latest_rows"][0]["live_queue_observation_mode"] == "running"
    assert summary["latest_rows"][0]["live_queue_observation_queue_sha256"]
    assert (
        summary["latest_rows"][0]["live_queue_observation_state_watermark"].get(
            "state_missing", False
        )
        is False
    )
    assert "High-level inverse-steganalysis/action-surface campaign intake" in text
    assert "status=READY_LOCAL_QUEUE" in text
    assert "conversion=50.00%" in text
    assert "compiler_gaps=1" in text
    assert "packetir_sets=2" in text
    assert "packetir_queue_ready=1" in text
    assert "exact_handoffs=1" in text
    assert "feedback_ready=True" in text
    assert "feedback_queued=True" in text
    assert "feedback_policy=local_autopilot_policy" in text
    assert "feedback_executed=True" in text
    assert "feedback_success=True" in text
    assert "feedback_continue=1" in text
    assert "queue_recovery_required=0" in text
    assert "queue_recovery_ready=0" in text
    assert "live_queue_observed=1" in text
    assert "live_queue_unhealthy=0" in text
    assert "queue_recovery_queued=0" in text
    assert "queue_recovery_executed=0" in text
    assert "queue_recovery_success=0" in text
    assert "queue_recovery_groups=0" in text
    assert "queue_recovery_repeated_groups=0" in text
    assert "queue_maintenance=0" in text
    assert "feedback_continuation_queued=1" in text
    assert "feedback_decision=run_next_materializer_campaign_iteration" in text
    assert "feedback_continue=True" in text
    assert "feedback_continuation_queued=True" in text
    assert "queue_observation_source=live" in text
    assert "live_queue_mode=running" in text
    assert "local_mlx_ready=1" in text
    assert "materialize_inverse_scorer_cell_candidate" in text
    assert "not score authority" in text


def test_byte_shaving_acquisition_summary_surfaces_feedback_candidate_widening(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / ".omx" / "research"
    plan_ref = (
        ".omx/research/high_level/campaign_dry_feedback/"
        "byte_shaving_campaign_plan.json"
    )
    run_ref = (
        ".omx/research/high_level/campaign_dry_feedback/"
        "materializer_campaign_run.json"
    )
    policy_ref = (
        ".omx/research/high_level/campaign_dry_feedback/"
        "queue_feedback_replan_policy.json"
    )

    _write_json(
        tmp_path / plan_ref,
        {
            "campaign_id": "campaign_dry_feedback",
            "candidate_id": "inverse_steganalysis_water_bucket_plan.v1",
            "dispatch_blockers": ["requires_byte_closed_materialization_before_dispatch"],
            "combination_ladder": [
                {
                    "combo_id": "combo_0001",
                    "expected_score_gain": 0.000061,
                    "unit_count": 1,
                    "operation_families": [
                        "materialize_inverse_scorer_cell_candidate"
                    ],
                    "dispatch_blockers": [
                        "requires_byte_closed_materialization_before_dispatch"
                    ],
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        tmp_path / run_ref,
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "execute": True,
            "high_level_action_source_count": 1,
            "plan": plan_ref,
            "queue_id": "campaign_dry_feedback",
            "queue_feedback_replan_ready": True,
            "queue_feedback_replan_followup_execution_requested": True,
            "queue_feedback_replan_followup_executed": True,
            "queue_feedback_replan_followup_execution_success": True,
            "queue_feedback_replan_policy_path": policy_ref,
            "queue_feedback_replan_policy_decision": (
                "widen_inverse_candidate_generation"
            ),
            "queue_feedback_replan_policy_should_continue": False,
            "queue_feedback_replan_policy": {
                "schema": "queue_feedback_replan_policy.v1",
                "decision": "widen_inverse_candidate_generation",
                "stop_reason": "feedback_action_functional_dry_no_selected_cells",
                "should_continue_feedback_loop": False,
                "ready_for_candidate_generation_widening": True,
                "feedback_action_functional_summary": {
                    "loaded": True,
                    "dry_no_selected_cells": True,
                    "cell_count": 1,
                    "selected_count": 0,
                    "blocked_cell_count": 1,
                    "materializer_archive_delta_blocked_cell_count": 1,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "queue_feedback_candidate_widening_queue_path": (
                ".omx/research/high_level/campaign_dry_feedback/"
                "queue_feedback_candidate_widening_queue.json"
            ),
            "queue_feedback_candidate_widening_queue_emitted": True,
            "queue_feedback_candidate_widening_queue_blockers": [],
            "experiment_count": 0,
            "build": {
                "materializer_work_queue_executable_row_count": 0,
                "materializer_work_queue_blocked_row_count": 1,
                "blocked_row_count": 0,
                "materializer_backlog_row_count": 0,
            },
            "worker": {
                "schema": "experiment_queue_worker_result.v1",
                "execute": True,
                "failure_count": 0,
                "success_count": 1,
                "stop_reason": "completed",
            },
            "observation": {
                "status_counts": {},
                "ready_steps": [],
                "failed_steps": [],
                "definition_drift": {
                    "changed_step_count": 0,
                    "missing_step_count": 0,
                    "missing_hash_step_count": 0,
                },
            },
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    text = mod._format_byte_shaving_acquisition_summary()

    assert summary["status"] == "NEEDS_CANDIDATE_WIDENING"
    assert summary["reason"] == (
        "latest feedback action surface has no selected materializer cells; "
        "refresh or widen inverse candidate generation"
    )
    assert summary["campaign_run_count"] == 1
    assert summary["queue_feedback_candidate_widening_ready_count"] == 1
    assert summary["queue_feedback_candidate_widening_queue_count"] == 1
    assert summary["queue_feedback_dry_no_selected_count"] == 1
    assert summary["queue_feedback_archive_delta_blocked_cell_count"] == 1
    assert summary["queue_feedback_policy_continue_count"] == 0
    assert "status: NEEDS_CANDIDATE_WIDENING" in text
    assert "feedback_widen=1" in text
    assert "feedback_dry=1" in text
    assert "feedback_archive_delta_blocked_cells=1" in text
    assert "feedback_widen_queue=1" in text
    assert summary["latest_rows"][0]["queue_feedback_replan_candidate_widening_ready"]
    assert summary["latest_rows"][0]["queue_feedback_replan_dry_no_selected_cells"]
    assert summary["latest_rows"][0]["queue_feedback_replan_feedback_cell_count"] == 1
    assert summary["latest_rows"][0]["queue_feedback_replan_feedback_selected_count"] == 0
    assert (
        summary["latest_rows"][0][
            "queue_feedback_replan_archive_delta_blocked_cell_count"
        ]
        == 1
    )


def test_byte_shaving_acquisition_summary_surfaces_feedback_actuation_queue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / ".omx" / "research"
    plan_ref = ".omx/research/high_level/campaign_actuation/byte_shaving_campaign_plan.json"
    run_ref = ".omx/research/high_level/campaign_actuation/materializer_campaign_run.json"

    _write_json(
        tmp_path / plan_ref,
        {
            "campaign_id": "campaign_actuation",
            "candidate_id": "inverse_steganalysis_water_bucket_plan.v1",
            "materialization_bridge": {
                "high_level_operation_compiler_required_count": 3,
                "packet_ir_operation_set_count": 4,
                "queue_consumable_packet_ir_operation_set_count": 0,
                "packet_ir_byte_closed_operation_count": 0,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        tmp_path / run_ref,
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "execute": True,
            "high_level_action_source_count": 1,
            "plan": plan_ref,
            "queue_id": "campaign_actuation",
            "queue_feedback_replan_ready": True,
            "queue_feedback_replan_policy_decision": (
                "widen_inverse_candidate_generation"
            ),
            "queue_feedback_replan_policy_should_continue": False,
            "queue_feedback_replan_policy": {
                "schema": "queue_feedback_replan_policy.v1",
                "decision": "widen_inverse_candidate_generation",
                "should_continue_feedback_loop": False,
                "ready_for_candidate_generation_widening": True,
                "feedback_action_functional_summary": {
                    "loaded": True,
                    "dry_no_selected_cells": False,
                    "cell_count": 3,
                    "selected_count": 3,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "queue_feedback_candidate_widening_queue_emitted": True,
            "queue_feedback_candidate_widening_queue_blockers": [],
            "queue_feedback_candidate_actuation_planning_queue_path": (
                ".omx/research/high_level/campaign_actuation/"
                "queue_feedback_candidate_actuation_planning_queue.json"
            ),
            "queue_feedback_candidate_actuation_planning_queue_emitted": True,
            "queue_feedback_candidate_actuation_planning_queue_blockers": [],
            "experiment_count": 0,
            "build": {
                "materializer_work_queue_executable_row_count": 0,
                "materializer_work_queue_blocked_row_count": 1,
                "blocked_row_count": 1,
                "materializer_backlog_row_count": 1,
            },
            "worker": {
                "schema": "experiment_queue_worker_result.v1",
                "execute": True,
                "failure_count": 0,
                "success_count": 0,
                "stop_reason": "completed",
            },
            "observation": {
                "status_counts": {},
                "ready_steps": [],
                "failed_steps": [],
                "definition_drift": {
                    "changed_step_count": 0,
                    "missing_step_count": 0,
                    "missing_hash_step_count": 0,
                },
            },
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    text = mod._format_byte_shaving_acquisition_summary()

    assert summary["status"] == "NEEDS_RECEIVER_COMPILER"
    assert summary["queue_feedback_candidate_actuation_planning_queue_count"] == 1
    assert (
        summary["latest_rows"][0][
            "queue_feedback_candidate_actuation_planning_queue_emitted"
        ]
        is True
    )
    assert "feedback_actuation_queue=1" in text


def test_byte_shaving_acquisition_summary_live_observation_wins_over_stale_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    root = tmp_path / ".omx" / "research"
    campaign_dir = root / "high_level" / "campaign_live_unhealthy"
    queue_ref = (
        ".omx/research/high_level/campaign_live_unhealthy/"
        "materializer_execution_queue.json"
    )
    state_ref = ".omx/state/experiment_queue_live_unhealthy.sqlite"
    queue = {
        "schema": "experiment_queue.v1",
        "queue_id": "live_unhealthy",
        "controls": {
            "mode": "running",
            "max_concurrency": {"local_cpu": 1},
        },
        "experiments": [
            {
                "id": "exp0",
                "status": "queued",
                "steps": [
                    {
                        "id": "materialize_local_proof_chain",
                        "kind": "command",
                        "command": ["python", "-c", "raise SystemExit(1)"],
                        "resources": {"kind": "local_cpu"},
                    }
                ],
            }
        ],
    }
    queue_path = _write_json(tmp_path / queue_ref, queue)
    queue = load_queue_definition(queue_path)
    state_path = tmp_path / state_ref
    with connect_state(state_path) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'failed',
                attempts = 1,
                updated_at_utc = '2026-05-25T00:00:00Z'
            WHERE queue_id = 'live_unhealthy'
              AND experiment_id = 'exp0'
              AND step_id = 'materialize_local_proof_chain'
            """
        )
        conn.commit()
    _write_json(
        campaign_dir / "byte_shaving_campaign_plan.json",
        {
            "combination_ladder": [],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        campaign_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "plan": (
                ".omx/research/high_level/campaign_live_unhealthy/"
                "byte_shaving_campaign_plan.json"
            ),
            "queue_id": "live_unhealthy",
            "queue_path": queue_ref,
            "state_path": state_ref,
            "experiment_count": 1,
            "high_level_action_source_count": 1,
            "build": {
                "materializer_work_queue_executable_row_count": 1,
                "materializer_work_queue_blocked_row_count": 0,
                "blocked_row_count": 0,
            },
            "worker": {
                "schema": "experiment_queue_worker_result.v1",
                "failure_count": 0,
            },
            "observation": {
                "status_counts": {"queued": 1},
                "ready_steps": [
                    {
                        "step_id": "materialize_local_proof_chain",
                        "resource_kind": "local_cpu",
                    }
                ],
                "failed_steps": [],
                "definition_drift": {
                    "changed_step_count": 0,
                    "missing_step_count": 0,
                    "missing_hash_step_count": 0,
                },
            },
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    text = mod._format_byte_shaving_acquisition_summary()
    latest = summary["latest_rows"][0]

    assert summary["status"] == "BLOCKED"
    assert summary["live_queue_observation_used_count"] == 1
    assert summary["live_queue_observation_unhealthy_count"] == 1
    assert summary["live_queue_observation_blocker_count"] == 1
    assert latest["queue_observation_source"] == "live"
    assert latest["live_queue_observation_used"] is True
    assert latest["live_queue_observation_healthy"] is False
    assert latest["live_queue_observation_mode"] == "running"
    assert latest["live_queue_observation_queue_sha256"]
    assert (
        latest["live_queue_observation_state_watermark"].get("state_missing", False)
        is False
    )
    assert latest["live_queue_observation_failed_step_count"] == 1
    assert latest["failed_step_count"] == 1
    assert latest["ready_step_count"] == 0
    assert latest["live_queue_observation_status_counts"] == {"failed": 1}
    assert latest["live_queue_observation_blockers"] == [
        "experiment_queue_observation_failed_steps:1"
    ]
    assert latest["queue_observation_recovery_required"] is True
    assert latest["ready_for_queue_health_recovery"] is True
    assert latest["operator_queue_state_mutation_required"] is True
    assert latest["queue_observation_recovery_plan_source"] == "live_queue_observation"
    assert latest["queue_observation_recovery_action_count"] == 1
    assert "experiment_queue_observation_failed_steps:1" in latest["blockers"]
    assert "1 observed queue step(s) failed" in latest["blockers"]
    assert "tools/materialize_byte_shaving_queue_recovery.py" in summary["next_command"]
    assert "--run-summary" in summary["next_command"]
    assert "--write" in summary["next_command"]
    assert "run-worker --execute" not in summary["next_command"]
    assert "live_queue_observed=1" in text
    assert "live_queue_unhealthy=1" in text
    assert "live_queue_observed=True" in text
    assert "live_queue_healthy=False" in text
    assert "live_queue_mode=running" in text
    assert "live_queue_failed_steps=1" in text
    assert "queue_recovery_required=True" in text
    assert "queue_recovery_plan_source=live_queue_observation" in text
    assert "live_queue_blockers=experiment_queue_observation_failed_steps:1" in text
    assert "recovery_materialize=.venv/bin/python tools/materialize_byte_shaving_queue_recovery.py" in text


def test_materialize_byte_shaving_queue_recovery_emits_paused_recovery_queue(
    tmp_path: Path,
) -> None:
    from comma_lab.scheduler.experiment_queue import (
        connect_state,
        initialize_queue_state,
        load_queue_definition,
    )

    run_dir = tmp_path / ".omx" / "research" / "campaign_repair"
    queue_ref = ".omx/research/campaign_repair/materializer_execution_queue.json"
    state_ref = ".omx/state/experiment_queue_repair.sqlite"
    queue = load_queue_definition(
        _write_json(
            tmp_path / queue_ref,
            {
                "schema": "experiment_queue.v1",
                "queue_id": "repair_queue",
                "controls": {"mode": "running"},
                "experiments": [
                    {
                        "id": "exp0",
                        "status": "queued",
                        "steps": [
                            {
                                "id": "materialize",
                                "kind": "command",
                                "command": ["python", "-c", "raise SystemExit(1)"],
                                "resources": {"kind": "local_cpu"},
                            }
                        ],
                    }
                ],
            },
        )
    )
    with connect_state(tmp_path / state_ref) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'failed',
                attempts = 1,
                updated_at_utc = '2026-05-25T00:00:00Z'
            WHERE queue_id = 'repair_queue'
              AND experiment_id = 'exp0'
              AND step_id = 'materialize'
            """
        )
        conn.commit()
    run_summary = _write_json(
        run_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "run_dir": ".omx/research/campaign_repair",
            "plan": ".omx/research/campaign_repair/byte_shaving_campaign_plan.json",
            "queue_id": "repair_queue",
            "queue_path": queue_ref,
            "state_path": state_ref,
            "experiment_count": 1,
            "build": {"materializer_work_queue_executable_row_count": 1},
            "worker": {"failure_count": 0},
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(RECOVERY_TOOL),
            "--repo-root",
            str(tmp_path),
            "--run-summary",
            str(run_summary),
            "--lane-id",
            "unit_live_queue_recovery",
            "--write",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    recovery_queue_path = run_dir / "queue_observation_recovery_queue.json"
    recovery_queue = json.loads(recovery_queue_path.read_text(encoding="utf-8"))

    assert payload["schema"] == (
        "byte_shaving_materializer_campaign_queue_recovery_materialization.v1"
    )
    assert payload["queue_observation_healthy"] is False
    assert payload["queue_observation_recovery_required"] is True
    assert payload["ready_for_queue_health_recovery"] is True
    assert payload["queue_observation_recovery_queue_emitted"] is True
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert (run_dir / "queue_observation.json").is_file()
    assert (run_dir / "queue_observation_recovery_plan.json").is_file()
    assert (run_dir / "queue_feedback_replan_policy.json").is_file()
    assert recovery_queue["schema"] == "experiment_queue.v1"
    assert recovery_queue["controls"]["mode"] == "paused"
    command = recovery_queue["experiments"][0]["steps"][0]["command"]
    assert "tools/experiment_queue.py" in command
    assert "rewind" in command
    assert "exp0" in command
    assert "materialize" in command
    overwrite_proc = subprocess.run(
        [
            sys.executable,
            str(RECOVERY_TOOL),
            "--repo-root",
            str(tmp_path),
            "--run-summary",
            str(run_summary),
            "--lane-id",
            "unit_live_queue_recovery",
            "--write",
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    overwrite_payload = json.loads(overwrite_proc.stdout)
    assert overwrite_payload["overwrite"] is True
    assert all(
        record["exists_before"] is True
        for record in overwrite_payload["artifact_records"]
    )


def test_materialize_byte_shaving_queue_recovery_refuses_invalid_template_rewind(
    tmp_path: Path,
) -> None:
    from comma_lab.scheduler.experiment_queue import (
        connect_state,
        initialize_queue_state,
        load_queue_definition,
    )

    run_dir = tmp_path / ".omx" / "research" / "campaign_bad_template"
    queue_ref = ".omx/research/campaign_bad_template/materializer_execution_queue.json"
    state_ref = ".omx/state/experiment_queue_bad_template.sqlite"
    template_ref = ".omx/research/campaign_bad_template/template.zip"
    action_ref = ".omx/research/campaign_bad_template/action.json"
    (tmp_path / template_ref).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / template_ref).write_text("fixture only; not a zip\n", encoding="utf-8")
    _write_json(tmp_path / action_ref, {"schema": "fixture"})
    queue = load_queue_definition(
        _write_json(
            tmp_path / queue_ref,
            {
                "schema": "experiment_queue.v1",
                "queue_id": "bad_template_queue",
                "controls": {"mode": "running"},
                "experiments": [
                    {
                        "id": "exp0",
                        "status": "queued",
                        "steps": [
                            {
                                "id": "materialize",
                                "kind": "command",
                                "command": [
                                    sys.executable,
                                    "tools/run_inverse_scorer_cell_candidate_chain.py",
                                    "--candidate-archive-template",
                                    template_ref,
                                    "--inverse-action-functional",
                                    action_ref,
                                    "--raw-contest-video-digest",
                                    "f" * 64,
                                    "--output-dir",
                                    ".omx/research/campaign_bad_template/out",
                                ],
                                "resources": {"kind": "local_mlx"},
                            }
                        ],
                    }
                ],
            },
        )
    )
    with connect_state(tmp_path / state_ref) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'failed',
                attempts = 1,
                updated_at_utc = '2026-05-25T00:00:00Z'
            WHERE queue_id = 'bad_template_queue'
              AND experiment_id = 'exp0'
              AND step_id = 'materialize'
            """
        )
        conn.commit()
    run_summary = _write_json(
        run_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "run_dir": ".omx/research/campaign_bad_template",
            "plan": ".omx/research/campaign_bad_template/byte_shaving_campaign_plan.json",
            "queue_id": "bad_template_queue",
            "queue_path": queue_ref,
            "state_path": state_ref,
            "experiment_count": 1,
            "build": {"materializer_work_queue_executable_row_count": 1},
            "worker": {"failure_count": 0},
            "commands": [
                {
                    "returncode": 0,
                    "command": [
                        sys.executable,
                        "tools/build_byte_shaving_campaign_queue.py",
                        "--plan",
                        ".omx/research/campaign_bad_template/byte_shaving_campaign_plan.json",
                        "--materializer-contexts",
                        ".omx/research/campaign_bad_template/materializer_contexts.json",
                    ],
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(RECOVERY_TOOL),
            "--repo-root",
            str(tmp_path),
            "--run-summary",
            str(run_summary),
            "--lane-id",
            "unit_live_queue_recovery",
            "--write",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    diagnostics_path = run_dir / "queue_source_failure_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    assert payload["queue_observation_recovery_required"] is True
    assert payload["queue_observation_recovery_queue_emitted"] is False
    assert payload["source_failure_diagnostic_count"] == 1
    assert payload["source_failure_non_rewindable_count"] == 1
    assert payload["source_failure_recovery_queue_execution_recommended"] is False
    assert payload["source_failure_requires_context_repair"] is True
    assert any(
        blocker.startswith("candidate_archive_template_invalid_strict_single_member_zip")
        for blocker in payload["source_failure_blockers"]
    )
    assert any(
        blocker.startswith(
            "source_failure_non_rewindable:"
            "candidate_archive_template_invalid_strict_single_member_zip"
        )
        for blocker in payload["queue_observation_recovery_queue_blockers"]
    )
    assert diagnostics["diagnostic_count"] == 1
    assert diagnostics["recovery_queue_execution_recommended"] is False
    assert diagnostics["diagnostics"][0]["rewind_likely_repeats_failure"] is True
    assert not (run_dir / "queue_observation_recovery_queue.json").exists()


def test_byte_shaving_acquisition_summary_blocks_missing_live_queue_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / ".omx" / "research"
    campaign_dir = root / "high_level" / "campaign_missing_state"
    queue_ref = (
        ".omx/research/high_level/campaign_missing_state/"
        "materializer_execution_queue.json"
    )
    state_ref = ".omx/state/experiment_queue_missing_state.sqlite"
    _write_json(
        tmp_path / queue_ref,
        {
            "schema": "experiment_queue.v1",
            "queue_id": "missing_state",
            "controls": {"mode": "running"},
            "experiments": [
                {
                    "id": "exp0",
                    "status": "queued",
                    "steps": [
                        {
                            "id": "materialize",
                            "kind": "command",
                            "command": ["python", "-c", "print('ready')"],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        campaign_dir / "byte_shaving_campaign_plan.json",
        {
            "combination_ladder": [],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        campaign_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "plan": (
                ".omx/research/high_level/campaign_missing_state/"
                "byte_shaving_campaign_plan.json"
            ),
            "queue_id": "missing_state",
            "queue_path": queue_ref,
            "state_path": state_ref,
            "experiment_count": 1,
            "high_level_action_source_count": 1,
            "build": {"materializer_work_queue_executable_row_count": 1},
            "worker": {"failure_count": 0},
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    text = mod._format_byte_shaving_acquisition_summary()
    latest = summary["latest_rows"][0]

    assert summary["status"] == "BLOCKED"
    assert latest["queue_observation_source"] == "live"
    assert latest["live_queue_observation_used"] is True
    assert latest["live_queue_observation_healthy"] is False
    assert "experiment_queue_observation_state_missing" in latest[
        "live_queue_observation_blockers"
    ]
    assert latest["queue_observation_recovery_required"] is True
    assert latest["ready_for_queue_health_recovery"] is True
    assert "experiment_queue_observation_state_missing" in latest["blockers"]
    assert "tools/materialize_byte_shaving_queue_recovery.py" in summary["next_command"]
    assert "run-worker --execute" not in summary["next_command"]
    assert "live_queue_healthy=False" in text
    assert "experiment_queue_observation_state_missing" in text


def test_byte_shaving_acquisition_summary_blocks_paused_live_queue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    root = tmp_path / ".omx" / "research"
    campaign_dir = root / "high_level" / "campaign_paused"
    queue_ref = ".omx/research/high_level/campaign_paused/materializer_execution_queue.json"
    state_ref = ".omx/state/experiment_queue_paused.sqlite"
    queue = load_queue_definition(
        _write_json(
            tmp_path / queue_ref,
            {
                "schema": "experiment_queue.v1",
                "queue_id": "paused_queue",
                "controls": {"mode": "paused"},
                "experiments": [
                    {
                        "id": "exp0",
                        "status": "queued",
                        "steps": [
                            {
                                "id": "materialize",
                                "kind": "command",
                                "command": ["python", "-c", "print('ready')"],
                                "resources": {"kind": "local_cpu"},
                            }
                        ],
                    }
                ],
            },
        )
    )
    with connect_state(tmp_path / state_ref) as conn:
        initialize_queue_state(conn, queue)
    _write_json(
        campaign_dir / "byte_shaving_campaign_plan.json",
        {
            "combination_ladder": [],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        campaign_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "plan": ".omx/research/high_level/campaign_paused/byte_shaving_campaign_plan.json",
            "queue_id": "paused_queue",
            "queue_path": queue_ref,
            "state_path": state_ref,
            "experiment_count": 1,
            "high_level_action_source_count": 1,
            "build": {"materializer_work_queue_executable_row_count": 1},
            "worker": {"failure_count": 0},
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    text = mod._format_byte_shaving_acquisition_summary()
    latest = summary["latest_rows"][0]

    assert summary["status"] == "BLOCKED"
    assert latest["queue_observation_source"] == "live"
    assert latest["live_queue_observation_used"] is True
    assert latest["live_queue_observation_healthy"] is True
    assert latest["live_queue_observation_mode"] == "paused"
    assert latest["ready_step_count"] == 0
    assert "experiment_queue_observation_mode_not_running:paused" in latest[
        "blockers"
    ]
    assert summary["next_command"].endswith(" observe --tail-lines 20")
    assert "run-worker --execute" not in summary["next_command"]
    assert "live_queue_mode=paused" in text
    assert "experiment_queue_observation_mode_not_running:paused" in text


def test_byte_shaving_acquisition_summary_surfaces_queue_recovery_signal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / ".omx" / "research"
    campaign_dir = root / "high_level" / "campaign_recovery"
    queue_path = (
        ".omx/research/high_level/campaign_recovery/materializer_execution_queue.json"
    )
    state_path = (
        ".omx/research/high_level/campaign_recovery/materializer_execution_queue.sqlite"
    )
    _write_json(
        campaign_dir / "byte_shaving_campaign_plan.json",
        {
            "combination_ladder": [],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        campaign_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "plan": (
                ".omx/research/high_level/campaign_recovery/"
                "byte_shaving_campaign_plan.json"
            ),
            "queue_id": "high_level_fixture_recovery",
            "queue_path": queue_path,
            "state_path": state_path,
            "experiment_count": 1,
            "queue_observation_path": (
                ".omx/research/high_level/campaign_recovery/queue_observation.json"
            ),
            "queue_observation_recovery_plan_path": (
                ".omx/research/high_level/campaign_recovery/"
                "queue_observation_recovery_plan.json"
            ),
            "queue_observation_recovery_queue_path": (
                ".omx/research/high_level/campaign_recovery/"
                "queue_observation_recovery_queue.json"
            ),
            "queue_observation_recovery_queue_state_path": (
                ".omx/research/high_level/campaign_recovery/"
                "queue_observation_recovery_queue.sqlite"
            ),
            "queue_observation_recovery_queue_emitted": True,
            "queue_observation_recovery_queue_blockers": [],
            "queue_observation_recovery_policy_enabled": True,
            "queue_observation_recovery_execution_requested": True,
            "queue_observation_recovery_executed": False,
            "queue_observation_recovery_execution_success": False,
            "queue_observation_recovery_policy_blockers": [
                "queue_observation_recovery_validation:source_state_watermark_drift"
            ],
            "queue_observation_recovery_execution": {
                "schema": (
                    "byte_shaving_materializer_campaign_queue_observation_recovery_"
                    "execution.v1"
                ),
                "success": False,
                "blockers": [
                    "queue_observation_recovery_validation:"
                    "source_state_watermark_drift"
                ],
                "source_observation_after": {
                    "healthy": False,
                    "blockers": ["experiment_queue_observation_failed_steps:1"],
                },
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "queue_observation_recovery_required": True,
            "queue_observation_maintenance_recommended": False,
            "queue_observation_recovery_plan": {
                "schema": "queue_observation_recovery_plan.v1",
                "recovery_required": True,
                "maintenance_recommended": False,
                "action_count": 1,
                "required_action_count": 1,
                "maintenance_action_count": 0,
                "grouped_blocker_count": 1,
                "repeated_group_count": 1,
                "grouped_blockers": [
                    {
                        "schema": "queue_observation_recovery_blocker_group.v1",
                        "blocker_family": "experiment_queue_observation_failed_steps",
                        "scope_kind": "materializer_target",
                        "scope_value": (
                            "entropy_adapter:archive_section_entropy_recode_v1"
                        ),
                        "count": 2,
                        "repeated": True,
                        "score_claim": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "actions": [
                    {
                        "action": "rewind_failed_step",
                        "required": True,
                        "requires_explicit_execution": True,
                        "score_claim": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "queue_feedback_replan_policy_path": (
                ".omx/research/high_level/campaign_recovery/"
                "queue_feedback_replan_policy.json"
            ),
            "queue_feedback_replan_policy_decision": "recover_queue_health",
            "queue_feedback_replan_policy_should_continue": False,
            "queue_feedback_replan_policy": {
                "schema": "queue_feedback_replan_policy.v1",
                "decision": "recover_queue_health",
                "should_continue_feedback_loop": False,
                "queue_observation_recovery_required": True,
                "queue_observation_maintenance_recommended": False,
                "ready_for_queue_health_recovery": True,
                "operator_queue_state_mutation_required": True,
                "auto_execute_eligible": False,
                "queue_observation_recovery_plan": {
                    "schema": "queue_observation_recovery_plan.v1",
                    "recovery_required": True,
                    "maintenance_recommended": False,
                    "action_count": 1,
                    "required_action_count": 1,
                    "maintenance_action_count": 0,
                },
                "blockers": [],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "build": {
                "materializer_work_queue_executable_row_count": 1,
                "materializer_work_queue_blocked_row_count": 0,
                "blocked_row_count": 0,
            },
            "worker": {
                "schema": "experiment_queue_worker_result.v1",
                "failure_count": 0,
            },
            "observation": {
                "status_counts": {"queued": 1},
                "failed_steps": [],
                "definition_drift": {},
            },
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    text = mod._format_byte_shaving_acquisition_summary()
    latest = summary["latest_rows"][0]

    assert summary["queue_observation_recovery_required_count"] == 1
    assert summary["queue_observation_recovery_queue_count"] == 1
    assert summary["queue_observation_recovery_executed_count"] == 0
    assert summary["queue_observation_recovery_execution_success_count"] == 0
    assert summary["queue_observation_recovery_grouped_blocker_count"] == 1
    assert summary["queue_observation_recovery_repeated_group_count"] == 1
    assert summary["ready_for_queue_health_recovery_count"] == 1
    assert summary["queue_observation_required_action_count"] == 1
    assert latest["queue_observation_recovery_required"] is True
    assert latest["ready_for_queue_health_recovery"] is True
    assert latest["operator_queue_state_mutation_required"] is True
    assert latest["queue_observation_recovery_queue_emitted"] is True
    assert latest["queue_observation_recovery_queue_blocker_count"] == 0
    assert latest["queue_observation_recovery_policy_enabled"] is True
    assert latest["queue_observation_recovery_execution_requested"] is True
    assert latest["queue_observation_recovery_executed"] is False
    assert latest["queue_observation_recovery_execution_success"] is False
    assert latest["queue_observation_recovery_grouped_blocker_count"] == 1
    assert latest["queue_observation_recovery_repeated_group_count"] == 1
    assert latest["queue_observation_recovery_top_groups"] == [
        (
            "experiment_queue_observation_failed_steps:materializer_target="
            "entropy_adapter:archive_section_entropy_recode_v1:count=2:"
            "repeated=True"
        )
    ]
    assert latest["queue_observation_recovery_execution_blockers"] == [
        "queue_observation_recovery_validation:source_state_watermark_drift"
    ]
    assert latest["queue_observation_recovery_policy_blockers"] == [
        "queue_observation_recovery_validation:source_state_watermark_drift"
    ]
    assert latest["queue_observation_recovery_source_observation_healthy"] is False
    assert latest["queue_observation_recovery_source_observation_blockers"] == [
        "experiment_queue_observation_failed_steps:1"
    ]
    assert latest["queue_feedback_replan_policy_should_continue"] is False
    assert latest["queue_observation_recovery_action_count"] == 1
    assert latest["queue_observation_required_action_count"] == 1
    assert (
        ".omx/research/high_level/campaign_recovery/"
        "queue_observation_recovery_queue.json"
    ) in summary["next_command"]
    assert (
        ".omx/research/high_level/campaign_recovery/"
        "queue_observation_recovery_queue.sqlite"
    ) in summary["next_command"]
    assert summary["next_command"].endswith(" init")
    assert (
        ".omx/research/high_level/campaign_recovery/"
        "queue_observation_recovery_queue.sqlite"
    ) in summary["observe_command"]
    assert "queue_recovery_required=1" in text
    assert "queue_recovery_ready=1" in text
    assert "queue_recovery_queued=1" in text
    assert "queue_recovery_executed=0" in text
    assert "queue_recovery_success=0" in text
    assert "queue_recovery_groups=1" in text
    assert "queue_recovery_repeated_groups=1" in text
    assert "queue_recovery_top_groups=experiment_queue_observation_failed_steps" in text
    assert "queue_recovery_execution_blockers=queue_observation_recovery_validation" in text
    assert "queue_recovery_policy_blockers=queue_observation_recovery_validation" in text
    assert "queue_recovery_source_blockers=experiment_queue_observation_failed_steps:1" in text
    assert "queue_recovery_actions=1" in text
    assert "feedback_continue=False" in text


def test_byte_shaving_acquisition_summary_gates_non_rewindable_source_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / ".omx" / "research"
    campaign_dir = root / "high_level" / "campaign_bad_template"
    queue_ref = (
        ".omx/research/high_level/campaign_bad_template/"
        "materializer_execution_queue.json"
    )
    state_ref = (
        ".omx/research/high_level/campaign_bad_template/"
        "materializer_execution_queue.sqlite"
    )
    queue = load_queue_definition(
        _write_json(
            tmp_path / queue_ref,
            {
                "schema": "experiment_queue.v1",
                "queue_id": "bad_template_recovery",
                "controls": {"mode": "running"},
                "experiments": [
                    {
                        "id": "exp0",
                        "status": "queued",
                        "steps": [
                            {
                                "id": "materialize",
                                "kind": "command",
                                "command": [
                                    sys.executable,
                                    "tools/run_inverse_scorer_cell_candidate_chain.py",
                                    "--candidate-archive-template",
                                    ".omx/research/high_level/campaign_bad_template/template.zip",
                                ],
                                "resources": {"kind": "local_mlx"},
                            }
                        ],
                    }
                ],
            },
        )
    )
    with connect_state(tmp_path / state_ref) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = 'failed',
                attempts = 1,
                updated_at_utc = '2026-05-25T00:00:00Z'
            WHERE queue_id = 'bad_template_recovery'
              AND experiment_id = 'exp0'
              AND step_id = 'materialize'
            """
        )
        conn.commit()
    _write_json(
        campaign_dir / "byte_shaving_campaign_plan.json",
        {
            "combination_ladder": [],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        campaign_dir / "queue_observation_recovery_queue.json",
        {
            "schema": "experiment_queue.v1",
            "queue_id": "stale_recovery_queue",
            "controls": {"mode": "paused"},
            "experiments": [],
        },
    )
    _write_json(
        campaign_dir / "queue_source_failure_diagnostics.json",
        {
            "schema": "byte_shaving_materializer_source_failure_diagnostics.v1",
            "diagnostics_path": (
                ".omx/research/high_level/campaign_bad_template/"
                "queue_source_failure_diagnostics.json"
            ),
            "diagnostic_count": 1,
            "non_rewindable_source_failure_count": 1,
            "recovery_queue_execution_recommended": False,
            "recovery_queue_execution_blockers": [
                (
                    "source_failure_non_rewindable:"
                    "candidate_archive_template_invalid_strict_single_member_zip:"
                    "invalid ZIP archive"
                )
            ],
            "blockers": [
                (
                    "candidate_archive_template_invalid_strict_single_member_zip:"
                    "invalid ZIP archive"
                )
            ],
            "requires_context_repair": True,
            "recommended_next_action": (
                "repair_materializer_contexts_and_rebuild_source_queue"
            ),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        campaign_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "run_dir": ".omx/research/high_level/campaign_bad_template",
            "plan": (
                ".omx/research/high_level/campaign_bad_template/"
                "byte_shaving_campaign_plan.json"
            ),
            "queue_id": "bad_template_recovery",
            "queue_path": queue_ref,
            "state_path": state_ref,
            "experiment_count": 1,
            "queue_observation_recovery_queue_path": (
                ".omx/research/high_level/campaign_bad_template/"
                "queue_observation_recovery_queue.json"
            ),
            "queue_observation_recovery_queue_state_path": (
                ".omx/research/high_level/campaign_bad_template/"
                "queue_observation_recovery_queue.sqlite"
            ),
            "queue_observation_recovery_queue_emitted": True,
            "queue_observation_recovery_execution_success": False,
            "build": {"materializer_work_queue_executable_row_count": 1},
            "worker": {"failure_count": 0},
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    text = mod._format_byte_shaving_acquisition_summary()
    latest = summary["latest_rows"][0]

    assert summary["source_failure_diagnostic_count"] == 1
    assert summary["source_failure_non_rewindable_count"] == 1
    assert summary["source_failure_recovery_gated_count"] == 1
    assert latest["queue_observation_recovery_queue_emitted"] is True
    assert latest["source_failure_diagnostic_count"] == 1
    assert latest["source_failure_non_rewindable_count"] == 1
    assert latest["source_failure_recovery_queue_execution_recommended"] is False
    assert latest["source_failure_requires_context_repair"] is True
    assert (
        ".omx/research/high_level/campaign_bad_template/"
        "materializer_execution_queue.json"
    ) in summary["next_command"]
    assert "queue_observation_recovery_queue.json" not in summary["next_command"]
    assert summary["next_command"].endswith(" observe --tail-lines 20")
    assert (
        ".omx/research/high_level/campaign_bad_template/"
        "materializer_execution_queue.sqlite"
    ) in summary["observe_command"]
    assert "source_non_rewindable=1" in text
    assert "source_recovery_recommended=False" in text
    assert "source_failure_blockers=candidate_archive_template_invalid" in text
    assert "source_failure_recovery_blockers=source_failure_non_rewindable" in text


def test_byte_shaving_acquisition_summary_surfaces_post_recovery_replan_signal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / ".omx" / "research"
    campaign_dir = root / "high_level" / "campaign_post_recovery"
    post_continuation_queue = (
        ".omx/research/high_level/campaign_post_recovery/"
        "queue_feedback_replan_continuation_queue_after_recovery.json"
    )
    post_continuation_state = (
        ".omx/research/high_level/campaign_post_recovery/"
        "queue_feedback_replan_continuation_queue_after_recovery.sqlite"
    )
    post_followup_queue = (
        ".omx/research/high_level/campaign_post_recovery/"
        "queue_feedback_replan_followup_queue_after_recovery.json"
    )
    post_followup_state = (
        ".omx/research/high_level/campaign_post_recovery/"
        "queue_feedback_replan_followup_queue.sqlite"
    )
    _write_json(
        campaign_dir / "byte_shaving_campaign_plan.json",
        {
            "combination_ladder": [],
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    _write_json(
        campaign_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "plan": (
                ".omx/research/high_level/campaign_post_recovery/"
                "byte_shaving_campaign_plan.json"
            ),
            "queue_id": "high_level_fixture_post_recovery",
            "queue_path": (
                ".omx/research/high_level/campaign_post_recovery/"
                "materializer_execution_queue.json"
            ),
            "state_path": (
                ".omx/research/high_level/campaign_post_recovery/"
                "materializer_execution_queue.sqlite"
            ),
            "experiment_count": 1,
            "queue_observation_recovery_required": True,
            "queue_observation_recovery_queue_emitted": True,
            "queue_observation_recovery_execution_requested": True,
            "queue_observation_recovery_executed": True,
            "queue_observation_recovery_execution_success": True,
            "queue_observation_recovery_execution": {
                "schema": (
                    "byte_shaving_materializer_campaign_queue_observation_recovery_"
                    "execution.v1"
                ),
                "success": True,
                "blockers": [],
                "source_observation_after": {"healthy": True, "blockers": []},
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "post_recovery_feedback_replan_triggered": True,
            "post_recovery_feedback_replan_success": True,
            "post_recovery_queue_observation_path": (
                ".omx/research/high_level/campaign_post_recovery/"
                "queue_observation_after_recovery.json"
            ),
            "post_recovery_queue_observation_recovery_plan_path": (
                ".omx/research/high_level/campaign_post_recovery/"
                "queue_observation_recovery_plan_after_recovery.json"
            ),
            "post_recovery_queue_feedback_replan_request_path": (
                ".omx/research/high_level/campaign_post_recovery/"
                "queue_feedback_replan_request_after_recovery.json"
            ),
            "post_recovery_queue_feedback_replan_policy_path": (
                ".omx/research/high_level/campaign_post_recovery/"
                "queue_feedback_replan_policy_after_recovery.json"
            ),
            "post_recovery_queue_feedback_replan_policy_decision": (
                "run_next_materializer_campaign_iteration"
            ),
            "post_recovery_queue_feedback_replan_followup_queue_path": (
                post_followup_queue
            ),
            "post_recovery_queue_feedback_replan_followup_state_path": (
                post_followup_state
            ),
            "post_recovery_queue_feedback_replan_followup_queue_emitted": True,
            "post_recovery_queue_feedback_replan_followup_executed": True,
            "post_recovery_queue_feedback_replan_followup_execution_success": True,
            "post_recovery_queue_feedback_replan_policy_should_continue": True,
            "post_recovery_queue_feedback_replan_continuation_queue_path": (
                post_continuation_queue
            ),
            "post_recovery_queue_feedback_replan_continuation_queue_state_path": (
                post_continuation_state
            ),
            "post_recovery_queue_feedback_replan_continuation_queue_emitted": True,
            "post_recovery_feedback_replan": {
                "schema": (
                    "byte_shaving_materializer_campaign_post_recovery_feedback_"
                    "replan.v1"
                ),
                "triggered": True,
                "artifacts_emitted": True,
                "success": True,
                "queue_feedback_replan_followup_queue_blockers": [],
                "queue_feedback_replan_followup_policy_blockers": [],
                "queue_feedback_replan_continuation_queue_blockers": [],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "queue_feedback_replan_policy": {
                "schema": "queue_feedback_replan_policy.v1",
                "decision": "recover_queue_health",
                "should_continue_feedback_loop": False,
                "ready_for_queue_health_recovery": True,
                "blockers": [],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "build": {
                "materializer_work_queue_executable_row_count": 1,
                "materializer_work_queue_blocked_row_count": 0,
                "blocked_row_count": 0,
            },
            "worker": {
                "schema": "experiment_queue_worker_result.v1",
                "failure_count": 0,
            },
            "observation": {"status_counts": {"queued": 1}},
            "commands": [{"returncode": 0}],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    text = mod._format_byte_shaving_acquisition_summary()
    latest = summary["latest_rows"][0]

    assert summary["queue_observation_recovery_execution_success_count"] == 1
    assert summary["post_recovery_feedback_replan_count"] == 1
    assert summary["post_recovery_feedback_replan_success_count"] == 1
    assert summary["post_recovery_feedback_followup_queue_count"] == 1
    assert summary["post_recovery_feedback_followup_executed_count"] == 1
    assert summary["post_recovery_feedback_followup_execution_success_count"] == 1
    assert summary["post_recovery_feedback_policy_continue_count"] == 1
    assert summary["post_recovery_feedback_continuation_queue_count"] == 1
    assert latest["post_recovery_feedback_replan_triggered"] is True
    assert latest["post_recovery_feedback_replan_success"] is True
    assert latest["post_recovery_queue_feedback_replan_followup_queue_path"] == (
        post_followup_queue
    )
    assert latest[
        "post_recovery_queue_feedback_replan_continuation_queue_path"
    ] == post_continuation_queue
    assert latest[
        "post_recovery_queue_feedback_replan_continuation_queue_state_path"
    ] == post_continuation_state
    assert latest["post_recovery_queue_feedback_replan_policy_decision"] == (
        "run_next_materializer_campaign_iteration"
    )
    assert post_continuation_queue in summary["next_command"]
    assert post_continuation_state in summary["next_command"]
    assert summary["next_command"].endswith(" init")
    assert post_continuation_queue in summary["observe_command"]
    assert post_continuation_state in summary["observe_command"]
    assert "post_recovery_replan=1" in text
    assert "post_recovery_replan_success=1" in text
    assert "post_recovery_feedback_queued=1" in text
    assert "post_recovery_feedback_executed=1" in text
    assert "post_recovery_feedback_success=1" in text
    assert "post_recovery_continue=1" in text
    assert "post_recovery_continuation_queued=1" in text
    assert "post_recovery_feedback_executed=True" in text
    assert "post_recovery_feedback_success=True" in text
    assert "post_recovery_continue=True" in text
    assert "post_recovery_decision=run_next_materializer_campaign_iteration" in text


def test_byte_shaving_acquisition_summary_blocks_authority_leaks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / ".omx" / "research"
    campaign_dir = root / "leaky" / "campaign"
    _write_json(
        campaign_dir / "byte_shaving_campaign_plan.json",
        {
            "combination_ladder": [],
            "dispatch_blockers": ["requires_exact_auth_eval_before_score_claim"],
        },
    )
    _write_json(
        campaign_dir / "materializer_campaign_run.json",
        {
            "schema": "byte_shaving_materializer_campaign_run.v1",
            "plan": ".omx/research/leaky/campaign/byte_shaving_campaign_plan.json",
            "queue_id": "leaky_queue",
            "queue_path": ".omx/research/leaky/campaign/materializer_execution_queue.json",
            "experiment_count": 1,
            "score_claim": True,
            "ready_for_exact_eval_dispatch": True,
            "build": {"promotion_eligible": True},
            "worker": {"failure_count": 0},
            "observation": {"definition_drift": {}, "status_counts": {}},
        },
    )
    monkeypatch.setattr(mod, "BYTE_SHAVING_ACQUISITION_SCAN_ROOTS", (root,))

    summary = mod._byte_shaving_acquisition_summary()
    latest = summary["latest_rows"][0]

    assert summary["status"] == "BLOCKED"
    assert summary["score_claim"] is False
    assert latest["score_claim"] is False
    assert latest["ready_for_exact_eval_dispatch"] is False
    assert "campaign_run_authority_field_true:score_claim" in latest["blockers"]
    assert "campaign_run_authority_field_true:ready_for_exact_eval_dispatch" in latest["blockers"]
    assert "campaign_run_authority_field_true:promotion_eligible" in latest["blockers"]


def test_materializer_exact_ready_handoff_summary_keeps_reused_queue_id_inputs_distinct(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / "experiments" / "results"
    first_report = _write_json(
        root / "run_a" / "materializer_exact_eval_consumer_report_20260101T000000Z.json",
        {
            "schema": "materializer_exact_eval_consumer.v1",
            "authorized_candidate_count": 1,
            "blocked_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "experiment_queue_id": "materializer_exact_eval_consumer_queue",
            "rows": [
                {
                    "candidate_id": "a",
                    "exact_ready_queue_path": "experiments/results/run_a/input.json",
                }
            ],
        },
    )
    second_report = _write_json(
        root / "run_b" / "materializer_exact_eval_consumer_report_20260101T010000Z.json",
        {
            "schema": "materializer_exact_eval_consumer.v1",
            "authorized_candidate_count": 1,
            "blocked_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "experiment_queue_id": "materializer_exact_eval_consumer_queue",
            "rows": [
                {
                    "candidate_id": "b",
                    "exact_ready_queue_path": "experiments/results/run_b/input.json",
                }
            ],
        },
    )
    os.utime(first_report, ns=(1_000, 1_000))
    os.utime(second_report, ns=(2_000, 2_000))
    monkeypatch.setattr(mod, "MATERIALIZER_HANDOFF_SCAN_ROOTS", (root,))

    summary = mod._materializer_exact_ready_handoff_summary()

    assert summary["consumer_report_count"] == 2
    assert summary["consumer_authorized_candidate_count"] == 2
    assert summary["superseded_handoff_artifact_count"] == 0
    assert set(summary["recent_exact_ready_queue_paths"]) == {
        "experiments/results/run_a/input.json",
        "experiments/results/run_b/input.json",
    }


def test_materializer_exact_ready_handoff_summary_supersedes_old_candidate_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / "experiments" / "results"
    old_report = _write_json(
        root / "old" / "consumer_report.json",
        {
            "schema": "materializer_exact_eval_consumer.v1",
            "authorized_candidate_count": 0,
            "blocked_candidate_count": 1,
            "duplicate_candidate_count": 0,
            "experiment_queue_id": "same_queue",
            "rows": [
                {
                    "candidate_id": "same_candidate",
                    "archive_sha256": "a" * 64,
                    "blockers": ["old_blocker"],
                    "exact_ready_queue_path": "experiments/results/old/input.json",
                }
            ],
        },
    )
    new_report = _write_json(
        root / "new" / "consumer_report.json",
        {
            "schema": "materializer_exact_eval_consumer.v1",
            "authorized_candidate_count": 1,
            "blocked_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "experiment_queue_id": "same_queue",
            "rows": [
                {
                    "candidate_id": "same_candidate",
                    "archive_sha256": "a" * 64,
                    "exact_ready_queue_path": "experiments/results/new/input.json",
                }
            ],
        },
    )
    os.utime(old_report, ns=(1_000, 1_000))
    os.utime(new_report, ns=(2_000, 2_000))
    monkeypatch.setattr(mod, "MATERIALIZER_HANDOFF_SCAN_ROOTS", (root,))

    summary = mod._materializer_exact_ready_handoff_summary()

    assert summary["consumer_report_count"] == 1
    assert summary["consumer_authorized_candidate_count"] == 1
    assert summary["consumer_blocked_candidate_count"] == 0
    assert summary["top_blockers"] == []
    assert summary["superseded_handoff_artifact_count"] == 1
    assert summary["recent_exact_ready_queue_paths"] == ["experiments/results/new/input.json"]


def test_materializer_exact_ready_handoff_summary_keeps_distinct_stable_identities(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    root = tmp_path / "experiments" / "results"
    first_report = _write_json(
        root / "run_a" / "consumer_report.json",
        {
            "schema": "materializer_exact_eval_consumer.v1",
            "authorized_candidate_count": 1,
            "blocked_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "experiment_queue_id": "same_queue",
            "rows": [
                {
                    "candidate_id": "same_candidate",
                    "archive_sha256": "a" * 64,
                    "runtime_content_tree_sha256": "1" * 64,
                    "runtime_tree_sha256": "2" * 64,
                    "score_axis": "contest_cuda",
                    "stable_identity": (
                        f"archive={'a' * 64}:runtime_content={'1' * 64}:"
                        f"runtime_tree={'2' * 64}:score_axis=contest_cuda"
                    ),
                    "exact_ready_queue_path": "experiments/results/run_a/input.json",
                }
            ],
        },
    )
    second_report = _write_json(
        root / "run_b" / "consumer_report.json",
        {
            "schema": "materializer_exact_eval_consumer.v1",
            "authorized_candidate_count": 1,
            "blocked_candidate_count": 0,
            "duplicate_candidate_count": 0,
            "experiment_queue_id": "same_queue",
            "rows": [
                {
                    "candidate_id": "same_candidate",
                    "archive_sha256": "a" * 64,
                    "runtime_content_tree_sha256": "3" * 64,
                    "runtime_tree_sha256": "4" * 64,
                    "score_axis": "contest_cuda",
                    "stable_identity": (
                        f"archive={'a' * 64}:runtime_content={'3' * 64}:"
                        f"runtime_tree={'4' * 64}:score_axis=contest_cuda"
                    ),
                    "exact_ready_queue_path": "experiments/results/run_b/input.json",
                }
            ],
        },
    )
    os.utime(first_report, ns=(1_000, 1_000))
    os.utime(second_report, ns=(2_000, 2_000))
    monkeypatch.setattr(mod, "MATERIALIZER_HANDOFF_SCAN_ROOTS", (root,))

    summary = mod._materializer_exact_ready_handoff_summary()

    assert summary["consumer_report_count"] == 2
    assert summary["consumer_authorized_candidate_count"] == 2
    assert summary["superseded_handoff_artifact_count"] == 0
    assert set(summary["recent_exact_ready_queue_paths"]) == {
        "experiments/results/run_a/input.json",
        "experiments/results/run_b/input.json",
    }


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


def test_operator_briefing_surfaces_frontier_feedback_cycle_autopolicy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    cycle_dir = tmp_path / "cycle"
    refresh_dir = cycle_dir / "initial_refresh"
    queue_path = refresh_dir / "dqs1_followup_queue.json"
    _write_json(
        refresh_dir / "feedback_refresh_report.json",
        {
            "schema": "frontier_rate_attack_feedback_refresh.v1",
            "queue_id": "frontier_feedback_unit",
            "results_root": "experiments/results/dqs1",
            "artifacts": {
                "dqs1_followup_queue": str(queue_path),
            },
            "selected_candidate_ids": [
                "pairset_drop_two_a",
                "pairset_drop_many_k004_a",
                "pairset_geometry_lowimpact_k003_a",
            ],
            "materializer_feedback_payload_count": 2,
            "dqs1_observation_count": 0,
            "local_cpu_eureka_planning": {
                "schema": "frontier_rate_attack_local_cpu_eureka_discovery.v1",
                "signal_count": 2,
                "planner_hint_count": 1,
                "planner_hints": [
                    {
                        "hint_id": "dqs1_expand_beyond_drop_two_near_boundary",
                        "pairset_acquisition_profile": {
                            "active": True,
                            "drop_many_counts": [3, 4, 6, 8],
                            "max_drop_many": 96,
                        },
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "operator_commands": {
                "run_frontier_feedback_cycle": [
                    ".venv/bin/python",
                    "tools/run_frontier_rate_attack_feedback_cycle.py",
                    "--candidate-limit",
                    "2",
                ],
            },
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        },
    )
    _write_json(
        cycle_dir / "frontier_rate_attack_feedback_cycle.json",
        {
            "schema": "frontier_rate_attack_feedback_cycle.v1",
            "initial_refresh": {
                "artifacts": {
                    "dqs1_followup_queue": str(queue_path),
                    "feedback_refresh_report": str(
                        refresh_dir / "feedback_refresh_report.json"
                    ),
                },
                "selected_candidate_ids": [
                    "pairset_drop_two_a",
                    "pairset_drop_two_b",
                ],
                "queue_validate": {"valid": True},
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "harvest_signal": {
                "harvest_path_count": 0,
                "harvest_paths": [],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )
    late_refresh_dir = tmp_path / "retention_smoke"
    _write_json(
        late_refresh_dir / "feedback_refresh_report.json",
        {
            "schema": "frontier_rate_attack_feedback_refresh.v1",
            "queue_id": "frontier_feedback_retention_smoke",
            "artifacts": {
                "dqs1_followup_queue": str(
                    late_refresh_dir / "dqs1_followup_queue.json"
                ),
            },
            "selected_candidate_ids": ["pairset_drop_two_b"],
            "materializer_feedback_payload_count": 0,
            "dqs1_observation_count": 0,
            "local_cpu_eureka_planning": {
                "schema": "frontier_rate_attack_local_cpu_eureka_discovery.v1",
                "signal_count": 0,
                "planner_hint_count": 0,
                "planner_hints": [],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        },
    )
    monkeypatch.setattr(mod, "FRONTIER_FEEDBACK_SCAN_ROOTS", (tmp_path,))

    summary = mod._frontier_feedback_cycle_summary()

    assert summary["status"] == "READY_LOCAL_EXECUTION"
    assert summary["cycle_tool_exists"] is True
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["score_claim"] is False
    assert summary["cycle_report_count"] == 1
    assert summary["refresh_report_count"] == 2
    assert "--execute-followup" in summary["next_command"]
    assert summary["latest_cycle"]["initial_selected_candidate_count"] == 2
    assert summary["latest_refresh"]["eureka_signal_count"] == 0
    assert summary["latest_eureka_refresh"]["eureka_signal_count"] == 2
    assert summary["latest_eureka_refresh"]["eureka_planner_hint_ids"] == [
        "dqs1_expand_beyond_drop_two_near_boundary"
    ]
    assert summary["latest_eureka_refresh"]["selected_drop_many_candidate_count"] == 1
    assert summary["latest_eureka_refresh"]["selected_geometry_candidate_count"] == 1
    assert summary["latest_eureka_refresh"]["eureka_pairset_profile_active"] is True
    assert summary["latest_eureka_refresh"]["eureka_drop_many_counts"] == [3, 4, 6, 8]
    text = mod._format_frontier_feedback_cycle_summary()
    assert "authority: planning/local only" in text
    assert "eureka_hints: 0" in text
    assert "latest eureka refresh:" in text
    assert "selected_drop_many: 1" in text
    assert "selected_geometry: 1" in text


def test_operator_briefing_surfaces_repair_waterfill_action_functional_queue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    refresh_dir = tmp_path / "repair_refresh"
    repair_queue = refresh_dir / "repair_budget_waterfill_queue.json"
    autonomous_queue = refresh_dir / "autonomous_chain_optimization_queue.json"
    _write_json(
        repair_queue,
        {
            "schema": "experiment_queue.v1",
            "queue_id": "repair_waterfill_unit",
            "controls": {"mode": "running", "local_first": True},
            "experiments": [
                {
                    "id": "repair_waterfill_global",
                    "status": "frozen",
                    "priority": 1,
                    "metadata": {
                        "queue_actuation_ready": False,
                        "missing_prerequisite_artifact_keys": [
                            "targeted_component_correction_response_harvest"
                        ],
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                    "steps": [
                        {
                            "id": "inspect_missing_prerequisites",
                            "kind": "command",
                            "command": [".venv/bin/python", "-m", "json.tool", "{}"],
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        autonomous_queue,
        {
            "schema": "experiment_queue.v1",
            "queue_id": "autonomous_chain_unit",
            "controls": {"mode": "running", "local_first": True},
            "experiments": [
                {
                    "id": "autonomous_chain_global",
                    "status": "frozen",
                    "priority": 1,
                    "metadata": {
                        "queue_actuation_ready": False,
                        "blocked_child_queue_artifact_keys": [
                            "repair_budget_waterfill_queue"
                        ],
                        "missing_queue_artifact_keys": [],
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                    "steps": [
                        {
                            "id": "inspect_blocked_child_queue",
                            "kind": "command",
                            "command": [".venv/bin/python", "-m", "json.tool", "{}"],
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        refresh_dir / "feedback_refresh_report.json",
        {
            "schema": "frontier_rate_attack_feedback_refresh.v1",
            "queue_id": "frontier_feedback_repair_visibility",
            "artifacts": {
                "rate_budget_preservation_plan": str(
                    refresh_dir / "rate_budget_preservation_plan.json"
                ),
                "repair_budget_waterfill_queue": str(repair_queue),
                "autonomous_chain_optimization_queue": str(autonomous_queue),
                "autonomous_chain_optimization": str(
                    refresh_dir / "autonomous_chain_optimization.json"
                ),
            },
            "rate_budget_preservation_plan": {
                "schema": "frontier_rate_attack_rate_budget_preservation_plan.v1",
                "rate_only_candidate_count": 17,
                "rate_only_saved_bytes_total": 160,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "autonomous_chain_optimization": {
                "schema": "frontier_rate_attack_autonomous_chain_optimization.v1",
                "chain_count": 1,
                "top_chain_ids": ["global_many_op_rate_distortion_receiver_campaign"],
                "target_classes": ["packet_member", "archive_section", "tensor"],
                "rate_only_candidate_count": 17,
                "rate_only_saved_bytes_total": 160,
                "rows": [
                    {
                        "scheduler_actions": [
                            {
                                "id": "fit_segnet_posenet_repair_waterfill_policy",
                                "queue_artifact_key": "repair_budget_waterfill_queue",
                                "advisory_only": False,
                                "score_claim": False,
                                "promotion_eligible": False,
                                "rank_or_kill_eligible": False,
                                "ready_for_exact_eval_dispatch": False,
                            }
                        ]
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )
    monkeypatch.setattr(mod, "FRONTIER_FEEDBACK_SCAN_ROOTS", (tmp_path,))

    summary = mod._frontier_feedback_cycle_summary()

    assert summary["status"] == "AUTONOMOUS_CHAIN_QUEUE_BLOCKED"
    assert summary["rate_budget_preservation_plan_count"] == 1
    assert summary["autonomous_chain_artifact_count"] == 1
    assert summary["repair_budget_waterfill_queue_count"] == 1
    assert summary["autonomous_chain_queue_count"] == 1
    assert summary["action_functional_queue_integrated_count"] == 1
    latest = summary["latest_refresh"]
    assert latest["rate_budget_preservation_candidate_count"] == 17
    assert latest["rate_budget_preservation_saved_bytes_total"] == 160
    assert latest["concrete_repair_waterfill_action_count"] == 1
    assert latest["status"] == "AUTONOMOUS_CHAIN_QUEUE_BLOCKED"
    assert latest["repair_budget_waterfill_queue_status"] == "FROZEN"
    assert latest["autonomous_chain_optimization_queue_status"] == "FROZEN"
    assert latest["score_claim"] is False
    assert latest["ready_for_exact_eval_dispatch"] is False
    text = mod._format_frontier_feedback_cycle_summary()
    assert "rate_budget_preservation: True candidates=17 bytes=160" in text
    assert "repair_waterfill: actions=1 concrete=1 queue=FROZEN" in text


def test_pr95_mlx_control_profile_summary_surfaces_queue_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    root = tmp_path / "profiles" / "pr95_mlx_full_source_video_runtime_profile_fixture"
    run_manifest = _write_json(
        root / "plans" / "stage1" / "manifest.json",
        {
            "schema": "pr95_hnerv_mlx_timing_smoke_manifest_v1",
            "training_loss_surface": "rgb_yuv6_mse",
            "train_seconds": 0.125,
            "runtime_consumption_proof": {
                "runtime_consumption_proven": True,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )
    queue = _write_json(
        root / "experiment_queue.json",
        {
            "schema": "experiment_queue.v1",
            "queue_id": "pr95_mlx_profile_fixture",
            "controls": {"mode": "running", "local_first": True},
            "experiments": [],
        },
    )
    _write_json(
        root / "matrix_manifest.json",
        {
            "schema": "pr95_hnerv_mlx_optimizer_matrix_queue.v1",
            "queue_id": "pr95_mlx_profile_fixture",
            "queue_output": str(queue),
            "control_profile": "full_pr95_source_video_runtime",
            "stage_indices": [1],
            "source_video_loss_surface": "rgb_yuv6_mse",
            "train_on_source_video_pairs": True,
            "prove_pr95_runtime_consumption": True,
            "plans": [
                {
                    "candidate_id": "stage1_fixture",
                    "stage_index": 1,
                    "run_manifest": str(run_manifest),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_attempted": False,
                    "gpu_launched": False,
                }
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )
    monkeypatch.setattr(mod, "PR95_MLX_CONTROL_PROFILE_SCAN_ROOTS", (tmp_path,))

    summary = mod._pr95_mlx_control_profile_summary()
    latest = summary["latest_profile"]

    assert summary["profile_count"] == 1
    assert summary["blocked_count"] == 0
    assert latest["status"] == "QUEUE_READY"
    assert latest["control_profile"] == "full_pr95_source_video_runtime"
    assert latest["runtime_consumption_proven_count"] == 1
    assert latest["source_video_loss_surface"] == "rgb_yuv6_mse"
    assert latest["score_claim"] is False
    assert latest["ready_for_exact_eval_dispatch"] is False
    assert "runtime_proven=1" in mod._format_pr95_mlx_control_profiles()


def test_distortion_axis_probe_summary_surfaces_wave2_signals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    root = tmp_path / "tier_1_distortion_axis_probes_20260521"
    _write_json(
        root / "probe_7_hinton_kl_t2_longer_temporal_context_segnet_verdict.json",
        {
            "probe_id": "probe_7_temporal",
            "probe_name": "Probe 7 temporal",
            "verdict": "POSITIVE_SIGNAL_PLATEAU",
            "axis_tag": "[macOS-CPU advisory]",
            "evidence_grade": "macOS-CPU-advisory",
            "lane_id": "lane_wave2",
            "actual_signature": {
                "per_window_results": {
                    "4": {
                        "W": 4,
                        "kl_mean_temporal": 0.011,
                        "ratio_over_ccc_static_baseline": 16.5,
                        "ratio_over_probe_6_w2": 1.46,
                        "classes_with_measurable_drift": 3,
                    },
                    "6": {
                        "W": 6,
                        "kl_mean_temporal": 0.014,
                        "ratio_over_ccc_static_baseline": 21.2,
                        "ratio_over_probe_6_w2": 1.88,
                        "classes_with_measurable_drift": 3,
                    },
                }
            },
            "next_action_on_POSITIVE_PLATEAU": "Probe 6 W=2 dispatch unchanged.",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )
    _write_json(
        root / "probe_8_uniward_per_segment_label_segnet_verdict.json",
        {
            "probe_id": "probe_8_segment",
            "probe_name": "Probe 8 segment",
            "verdict": "POSITIVE_SIGNAL_PER_SEGMENT_PARTIAL",
            "axis_tag": "[macOS-CPU advisory]",
            "evidence_grade": "macOS-CPU-advisory",
            "lane_id": "lane_wave2",
            "actual_signature": {
                "min_segment_textured_avg_weight": 0.5233,
                "spread_segment_textured_avg_weight": 0.4027,
                "valid_segment_count": 22,
            },
            "next_action_on_PARTIAL": "Combine per-instance and multi-scale UNIWARD.",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )
    _write_json(
        root / "probe_9_uniward_per_instance_multi_scale_wavelet_combined_verdict.json",
        {
            "probe_id": "probe_9_combined",
            "probe_name": "Probe 9 combined",
            "verdict": "POSITIVE_SIGNAL_BREAKS_THRESHOLD",
            "axis_tag": "[macOS-CPU advisory]",
            "evidence_grade": "macOS-CPU-advisory",
            "lane_id": "lane_wave3",
            "actual_signature": {
                "min_segment_textured_avg_weight_combined": 0.2597,
                "spread_segment_textured_avg_weight_combined": 0.3556,
                "valid_segment_count": 22,
                "any_segment_below_threshold": True,
                "wavelet_name": "db8",
                "wavelet_levels": 3,
            },
            "next_action_on_POSITIVE_BREAKS_THRESHOLD": (
                "Tier-2 paid dispatch on per-instance multi-scale UNIWARD."
            ),
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )
    monkeypatch.setattr(mod, "DISTORTION_AXIS_PROBE_SCAN_ROOTS", (root,))

    summary = mod._distortion_axis_probe_summary()

    assert summary["status"] == "ADVISORY_SIGNALS"
    assert summary["probe_count"] == 3
    assert summary["positive_signal_count"] == 3
    assert summary["partial_signal_count"] == 1
    assert summary["plateau_signal_count"] == 1
    assert summary["threshold_broken_count"] == 1
    assert summary["best_temporal_context"]["W"] == 6
    assert summary["best_temporal_context"]["ratio_over_ccc_static_baseline"] == 21.2
    assert summary["best_uniward_segment"]["segment_min_textured_avg_weight"] == 0.2597
    assert summary["best_uniward_segment"]["segment_valid_count"] == 22
    assert summary["best_uniward_segment"]["wavelet_name"] == "db8"
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    text = mod._format_distortion_axis_probe_summary()
    assert "best temporal KL: W=6 ratio=21.20x" in text
    assert "best UNIWARD segment: min=0.2597" in text
    assert "threshold_broken=1" in text
    assert "authority: macOS-CPU advisory only" in text


def test_distortion_axis_learned_sweep_bridge_summary_surfaces_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    root = tmp_path / "results"
    _write_json(
        root / "distortion_axis_probe_learned_sweep_candidates.json",
        {
            "schema": "distortion_axis_probe_learned_sweep_candidates.v1",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
            "execution_bridge_status": (
                "planning_payload_only_selection_adapter_required_before_local_actuation"
            ),
            "summary": {
                "adapted_candidate_count": 1,
                "suppressed_candidate_count": 1,
                "best_predicted_score_mean": 0.182,
                "best_non_authoritative_repair_budget_score": 0.010,
                "best_non_authoritative_repair_budget_bytes_equivalent": 15018.2,
            },
        },
    )
    _write_json(
        root / "distortion_axis_probe_learned_sweep_plan.json",
        {
            "schema": "mlx_dynamic_learned_sweep_plan.v1",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
            "summary": {
                "candidate_count": 1,
                "ranked_row_count": 2,
                "local_ready_row_count": 2,
                "suppressed_observed_row_count": 0,
            },
        },
    )
    _write_json(
        root / "distortion_axis_probe_learned_sweep_feedback_summary.json",
        {
            "schema": "distortion_axis_probe_learned_sweep_feedback.v1",
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
            "candidate_id": (
                "distortion_axis:uniward_per_instance_multi_scale_wavelet_combined_v1"
            ),
            "sweep_config_id": "macos_cpu_advisory",
            "optimization_pass_id": "smoke",
            "observed_axis": "macos_cpu_advisory",
            "observed_score_or_delta": -0.010,
            "observation_jsonl": "observations.jsonl",
            "observation_jsonl_sha256": "a" * 64,
            "replan": {
                "schema": "mlx_dynamic_learned_sweep_plan.v1",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "suppressed_observed_row_count": 1,
                "local_ready_row_count": 1,
            },
        },
    )
    monkeypatch.setattr(mod, "DISTORTION_AXIS_LEARNED_SWEEP_SCAN_ROOTS", (root,))

    summary = mod._distortion_axis_learned_sweep_summary()

    assert summary["status"] == "ADVISORY_PLAN"
    assert summary["payload_count"] == 1
    assert summary["plan_count"] == 1
    assert summary["feedback_count"] == 1
    assert summary["adapted_candidate_count"] == 1
    assert summary["suppressed_candidate_count"] == 1
    assert summary["local_ready_row_count"] == 2
    assert summary["feedback_observation_count"] == 1
    assert summary["feedback_replan_suppressed_count"] == 1
    assert summary["best_repair_budget_bytes_equivalent"] == 15018.2
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    text = mod._format_distortion_axis_learned_sweep_summary()
    assert "best non-authoritative repair budget: 15018.2" in text
    assert "feedback_observations=1" in text
    assert "replan_suppressed=1" in text
    assert "authority: learned-sweep planning only" in text


def test_dqs1_drop_many_greedy_summary_surfaces_defer_verdict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    root = tmp_path / "results"
    verdict_path = root / (
        "dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_20260525"
    ) / "verdict.json"
    _write_json(
        verdict_path,
        {
            "schema": "dqs1_drop_many_build_1c_greedy_independent_heuristic_verdict.v1",
            "lane_id": (
                "lane_dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_20260525"
            ),
            "build_1c_final_verdict": (
                "NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES"
            ),
            "build_1c_final_verdict_reason": (
                "ALL empirical K>1 sisters regress vs K=1."
            ),
            "canonical_provenance": {
                "score_claim": False,
                "score_claim_valid": False,
                "promotable": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "canonical_equation_candidate_refinement": {
                "refinement_field_proposed": {
                    "empirical_k1_best_drop_one_pair_index": 371,
                    "empirical_k1_best_drop_one_delta_vs_base": -6.658e-7,
                }
            },
            "empirical_drop_one_anchor_distribution": {
                "n_negative_delta_pairs": 1,
                "n_positive_delta_pairs": 8,
            },
            "catalog_313_probe_outcomes_row": {
                "verdict": "DEFER",
                "status": "blocking",
                "reactivation_criteria": "BUILD-1b lands new empirical K>1 anchors",
            },
            "operator_routable_next_cascade": [
                "BUILD-1b paid Modal CPU paired exact-eval on unmeasured pairs.",
            ],
        },
    )
    monkeypatch.setattr(mod, "DQS1_DROP_MANY_GREEDY_VERDICT_SCAN_ROOTS", (root,))

    summary = mod._dqs1_drop_many_greedy_summary()

    assert summary["status"] == "ADVISORY_DEFER"
    assert summary["tool_exists"] is True
    assert summary["verdict_count"] == 1
    assert summary["latest_verdict"]["k1_best_pair_index"] == 371
    assert summary["latest_verdict"]["catalog_313_verdict"] == "DEFER"
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    text = mod._format_dqs1_drop_many_greedy_summary()
    assert "k1_pair=371" in text
    assert "BUILD-1b paid Modal CPU" in text
    assert "research/planning only" in text


def test_operator_briefing_blocks_frontier_feedback_cycle_authority_leak(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    _write_json(
        tmp_path / "frontier_rate_attack_feedback_cycle.json",
        {
            "schema": "frontier_rate_attack_feedback_cycle.v1",
            "score_claim": True,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        },
    )
    monkeypatch.setattr(mod, "FRONTIER_FEEDBACK_SCAN_ROOTS", (tmp_path,))

    summary = mod._frontier_feedback_cycle_summary()

    assert summary["status"] == "BLOCKED"
    assert summary["error_count"] == 1
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["score_claim"] is False


def test_operator_briefing_blocks_nested_frontier_feedback_cycle_authority_leak(
    tmp_path: Path,
) -> None:
    mod = _load_briefing_module()
    path = _write_json(
        tmp_path / "frontier_rate_attack_feedback_cycle.json",
        {
            "schema": "frontier_rate_attack_feedback_cycle.v1",
            "initial_refresh": {
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "post_followup_eureka_planning": {
                "payload": {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": True,
                }
            },
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )

    row = mod._frontier_feedback_cycle_row(path)

    assert row["status"] == "BLOCKED_AUTHORITY_LEAK"
    assert (
        "truthy_authority:post_followup_eureka_planning.payload.ready_for_exact_eval_dispatch"
        in row["blockers"]
    )


def test_operator_briefing_surfaces_campaign_wave_frontier_feedback_queue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_briefing_module()
    cycle_dir = tmp_path / "cycle"
    post_queue = cycle_dir / "post_harvest_refresh" / "dqs1_followup_queue.json"
    campaign_queue = (
        cycle_dir / "campaign_wave_002" / "refresh" / "dqs1_followup_queue.json"
    )
    _write_json(
        cycle_dir / "frontier_rate_attack_feedback_cycle.json",
        {
            "schema": "frontier_rate_attack_feedback_cycle.v1",
            "initial_refresh": {
                "artifacts": {"dqs1_followup_queue": str(cycle_dir / "initial.json")},
                "selected_candidate_ids": ["pairset_a", "pairset_b"],
                "queue_validate": {"valid": True},
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "harvest_signal": {
                "harvest_path_count": 2,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "post_harvest_refresh": {
                "artifacts": {"dqs1_followup_queue": str(post_queue)},
                "selected_candidate_ids": ["pairset_c"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "campaign_execution": {
                "requested_wave_count": 3,
                "completed_additional_wave_count": 1,
                "stop_reason": "wave_limit_reached",
                "waves": [
                    {
                        "wave_index": 2,
                        "refresh": {
                            "artifacts": {"dqs1_followup_queue": str(campaign_queue)},
                            "selected_candidate_ids": ["pairset_d"],
                            "score_claim": False,
                            "promotion_eligible": False,
                            "rank_or_kill_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        },
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )
    monkeypatch.setattr(mod, "FRONTIER_FEEDBACK_SCAN_ROOTS", (tmp_path,))

    summary = mod._frontier_feedback_cycle_summary()

    assert summary["status"] == "CAMPAIGN_QUEUE_READY"
    assert summary["latest_cycle"]["campaign_wave_count"] == 1
    assert summary["latest_cycle"]["latest_campaign_queue_path"] == str(campaign_queue)
    assert "--campaign-waves 4" in summary["next_command"]
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["score_claim"] is False


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
