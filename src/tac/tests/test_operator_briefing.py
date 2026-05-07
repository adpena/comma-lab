"""Smoke test for tools/operator_briefing.py orchestrator.

The orchestrator delegates to apogee_intN_pareto + score_dashboard +
predicted_vs_actual_reconciler — those have their own thorough test suites.
This file just guards against the orchestrator's subprocess wiring breaking
(wrong tool path, broken --skip flags, JSON composition bug).
"""
from __future__ import annotations

import json
import importlib.util
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
    assert "Phase 1" in proc.stdout
    assert "Phase 2" in proc.stdout
    assert "Phase 3" in proc.stdout
    assert "Phase 1 exact-eval packets" in proc.stdout
    assert "Phase 1 blocked readiness artifacts" in proc.stdout
    assert "pr91_hpm1_readiness_bundle" in proc.stdout
    assert "wr01_apply_pr106x_half" in proc.stdout
    assert "pr106x_lgblock16_1byte_brotli" in proc.stdout
    assert "Copy-safe next steps" in proc.stdout
    assert "assert_packet_ready_for_submit" in proc.stdout
    assert "refresh_with_operator_exact_cuda_approval" in proc.stdout


def test_briefing_skip_pareto_omits_phase1():
    proc = _run("--skip-pareto", "--top", "3")
    assert "Phase 1" not in proc.stdout
    assert "Phase 2" in proc.stdout
    assert "Phase 3" in proc.stdout


def test_briefing_skip_dashboard_omits_phase2():
    proc = _run("--skip-dashboard", "--top", "3")
    assert "Phase 1" in proc.stdout
    assert "Phase 2" not in proc.stdout
    assert "Phase 3" in proc.stdout


def test_briefing_skip_reconciler_omits_phase3():
    proc = _run("--skip-reconciler", "--top", "3")
    assert "Phase 1" in proc.stdout
    assert "Phase 2" in proc.stdout
    assert "Phase 3" not in proc.stdout


def test_briefing_json_composite_has_all_three_keys():
    proc = _run("--json", "--top", "3")
    out = json.loads(proc.stdout)
    assert "pareto" in out
    assert "dashboard" in out
    assert "reconciler" in out
    assert "exact_eval_packets" in out
    assert "non_dispatchable_readiness_artifacts" in out
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
        "pr106x_lgblock16_1byte_brotli",
    }
    lgblock16 = packet_rows["pr106x_lgblock16_1byte_brotli"]
    assert lgblock16["ready_for_submit"] is False
    assert lgblock16["preflight_ready"] is True
    assert lgblock16["compliance_ok"] is True
    assert lgblock16["payload_diff_ready"] is True
    assert lgblock16["dry_run_ready"] is True
    assert "missing_active_lane_dispatch_claim" in lgblock16["blockers"]
    assert lgblock16["operator_next_steps"]["schema"] == "hnerv_lowlevel_operator_next_steps_v1"


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
    assert (
        packet_rows["wr01_apply_pr106x_half"]["operator_next_steps"]["steps"][4]["id"]
        == "assert_packet_ready_for_submit"
    )
    assert (
        packet_rows["pr106x_lgblock16_1byte_brotli"]["operator_next_steps"]["steps"][4]["id"]
        == "submit_exact_cuda"
    )
    assert out["non_dispatchable_readiness_artifacts"][0]["score_claim"] is False


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
