"""Smoke test for tools/operator_briefing.py orchestrator.

The orchestrator delegates to apogee_intN_pareto + score_dashboard +
predicted_vs_actual_reconciler — those have their own thorough test suites.
This file just guards against the orchestrator's subprocess wiring breaking
(wrong tool path, broken --skip flags, JSON composition bug).
"""
from __future__ import annotations

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


def test_briefing_runs_all_three_phases():
    proc = _run("--top", "3")
    assert "Phase 1" in proc.stdout
    assert "Phase 2" in proc.stdout
    assert "Phase 3" in proc.stdout
    assert "Phase 1 exact-eval packets" in proc.stdout
    assert "Phase 1 blocked high-EV lanes" in proc.stdout
    assert "pr91_hpm1_runtime_contract" in proc.stdout
    assert "wr01_apply_pr106x_half" in proc.stdout


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
    assert "blocked_high_ev_lanes" in out
    assert out["blocked_high_ev_lanes"][0]["lane_id"] == "pr91_hpm1_runtime_contract"
    assert out["blocked_high_ev_lanes"][0]["ready_for_exact_eval_dispatch"] is False
    assert out["blocked_high_ev_lanes"][0]["archive_custody_matches"] is True
    assert out["blocked_high_ev_lanes"][0]["hpm1_mask_custody_matches"] is True
    assert out["blocked_high_ev_lanes"][0]["ambient_device_call_count"] >= 1
    assert out["blocked_high_ev_lanes"][0]["contradiction_count"] >= 1


def test_briefing_json_each_phase_has_n_total_or_n_configs():
    """Each sub-tool must emit a JSON dict with at least one count field."""
    proc = _run("--json", "--top", "3")
    out = json.loads(proc.stdout)
    # Each tool emits its own count fields — ensure at least one is present
    assert any(k in out["pareto"] for k in ("n_configs", "n_pareto_frontier"))
    assert any(k in out["dashboard"] for k in ("n_total", "n_displayed"))
    assert any(k in out["reconciler"] for k in ("n_configs", "n_landed"))
    assert out["exact_eval_packets"][0]["lane_id"] == "wr01_apply_pr106x_half"
    assert out["blocked_high_ev_lanes"][0]["score_claim"] is False
