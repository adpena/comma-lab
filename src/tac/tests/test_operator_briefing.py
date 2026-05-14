"""Smoke test for tools/operator_briefing.py orchestrator.

The orchestrator delegates to apogee_intN_pareto + score_dashboard +
predicted_vs_actual_reconciler — those have their own thorough test suites.
This file just guards against the orchestrator's subprocess wiring breaking
(wrong tool path, broken --skip flags, JSON composition bug).
"""
from __future__ import annotations

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


def test_briefing_runs_all_three_phases():
    proc = _run("--top", "3")
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
    assert "Copy-safe next steps" in proc.stdout
    assert "assert_packet_ready_for_submit" in proc.stdout
    assert "refresh_with_operator_exact_cuda_approval" in proc.stdout


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
    assert (
        "--expected-runtime-tree-sha256 "
        + hlm1_packet["runtime_tree_sha256"]
        in hlm1_packet["commands"]["submit"]
    )
    modal_cpu_runtime_sha = hlm1_packet["runtime_manifest"]["modal_cpu_runtime_tree_sha256"]
    assert modal_cpu_runtime_sha != hlm1_packet["runtime_tree_sha256"]
    assert (
        "--expected-runtime-tree-sha256 "
        + modal_cpu_runtime_sha
        in hlm1_packet["commands"]["submit_contest_cpu"]
    )
    assert "experiments/modal_auth_eval_cpu.py" in hlm1_packet["commands"]["submit_contest_cpu"]
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
    hlm1_step_ids = [
        step["id"]
        for step in hlm1["operator_next_steps"]["steps"]
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
        assert "submit_modal_exact_cuda" in hlm1_step_ids
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
