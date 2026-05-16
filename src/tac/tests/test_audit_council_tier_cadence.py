# SPDX-License-Identifier: MIT
"""Tests for tools/audit_council_tier_cadence.py."""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TOOL_PATH = _REPO_ROOT / "tools" / "audit_council_tier_cadence.py"


@pytest.fixture
def cadence_module():
    name = "audit_council_tier_cadence"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOL_PATH)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod  # register before exec so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


def _write_anchor(
    posterior: Path,
    *,
    deliberation_id: str,
    tier: str,
    written_at_utc: str,
    event_type: str = "dispatched",
    predicted_mission_contribution: str | None = None,
    override_invoked: bool = False,
    override_rationale: str | None = None,
    deferred_substrate_retrospective_due_utc: str | None = None,
    deferred_substrate_id: str | None = None,
):
    """Append a synthetic anchor row matching the v1 schema.

    Mission-alignment fields (predicted_mission_contribution / override_invoked
    / etc.) are optional; legacy rows without them are accepted by the
    loader via backfill-default (apparatus_maintenance for T2+).
    """
    payload = {
        "schema": "council_deliberation_posterior_v1",
        "deliberation_id": deliberation_id,
        "topic": "synth",
        "council_tier": tier,
        "council_attendees": ["Shannon"],
        "council_quorum_met": True,
        "council_verdict": "PROCEED",
        "council_dissent": [],
        "council_assumption_adversary_verdict": [
            {"assumption": "A", "classification": "HARD-EARNED", "rationale": "R"}
        ],
        "council_decisions_recorded": [],
        "related_deliberation_ids": [],
        "event_type": event_type,
        "written_at_utc": written_at_utc,
    }
    if predicted_mission_contribution is not None:
        payload["predicted_mission_contribution"] = predicted_mission_contribution
    if override_invoked:
        payload["override_invoked"] = True
    if override_rationale is not None:
        payload["override_rationale"] = override_rationale
    if deferred_substrate_retrospective_due_utc is not None:
        payload["deferred_substrate_retrospective_due_utc"] = (
            deferred_substrate_retrospective_due_utc
        )
    if deferred_substrate_id is not None:
        payload["deferred_substrate_id"] = deferred_substrate_id
    posterior.parent.mkdir(parents=True, exist_ok=True)
    with posterior.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def test_classify_under_budget(cadence_module):
    v = cadence_module.classify_tier_count("T3", 1)
    assert v.verdict == cadence_module.WITHIN_BUDGET
    assert v.budget == 13


def test_classify_approaching_limit(cadence_module):
    # T3 budget = 13/30d; 80% threshold = 10.4
    v = cadence_module.classify_tier_count("T3", 11)
    assert v.verdict == cadence_module.APPROACHING_LIMIT


def test_classify_over_cadence(cadence_module):
    v = cadence_module.classify_tier_count("T3", 14)
    assert v.verdict == cadence_module.OVER_CADENCE
    assert "STOP AND CONSOLIDATE" in v.alert_message


def test_t1_always_unbounded(cadence_module):
    v = cadence_module.classify_tier_count("T1", 1000)
    assert v.verdict == cadence_module.UNBOUNDED


def test_audit_empty_posterior(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    report = cadence_module.audit_cadence(posterior_path=posterior)
    assert report["any_over_cadence"] is False
    counts = {e["tier"]: e["count"] for e in report["per_tier"]}
    assert counts == {"T1": 0, "T2": 0, "T3": 0, "T4": 0}


def test_audit_t3_over_cadence(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    # Write 14 T3 deliberations in the last 7 days (over T3 budget of 13/30d).
    for i in range(14):
        _write_anchor(
            posterior,
            deliberation_id=f"t3_delib_{i:03d}",
            tier="T3",
            written_at_utc=(now - _dt.timedelta(days=i * 0.5)).isoformat(timespec="seconds"),
        )
    report = cadence_module.audit_cadence(posterior_path=posterior)
    assert report["any_over_cadence"] is True
    t3 = next(e for e in report["per_tier"] if e["tier"] == "T3")
    assert t3["verdict"] == cadence_module.OVER_CADENCE
    assert t3["count"] == 14


def test_audit_t2_within_budget(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    # 30 T2 deliberations in 30 days; budget is 90 -> WITHIN_BUDGET (33%).
    for i in range(30):
        _write_anchor(
            posterior,
            deliberation_id=f"t2_delib_{i:03d}",
            tier="T2",
            written_at_utc=(now - _dt.timedelta(days=i * 0.9)).isoformat(timespec="seconds"),
        )
    report = cadence_module.audit_cadence(posterior_path=posterior)
    assert report["any_over_cadence"] is False
    t2 = next(e for e in report["per_tier"] if e["tier"] == "T2")
    assert t2["verdict"] == cadence_module.WITHIN_BUDGET


def test_audit_outcome_events_not_counted(cadence_module, tmp_path: Path):
    """Outcome rows reference a deliberation_id but don't increment the count.

    Only `event_type == 'dispatched'` rows count toward the cadence budget.
    """
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    _write_anchor(posterior, deliberation_id="t3_one", tier="T3", written_at_utc=now, event_type="dispatched")
    _write_anchor(posterior, deliberation_id="t3_one", tier="T3", written_at_utc=now, event_type="outcome")
    _write_anchor(posterior, deliberation_id="t3_one", tier="T3", written_at_utc=now, event_type="outcome")
    report = cadence_module.audit_cadence(posterior_path=posterior)
    t3 = next(e for e in report["per_tier"] if e["tier"] == "T3")
    assert t3["count"] == 1


def test_render_text_includes_per_tier_table(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    _write_anchor(
        posterior,
        deliberation_id="t1_delib",
        tier="T1",
        written_at_utc=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    )
    report = cadence_module.audit_cadence(posterior_path=posterior)
    text = cadence_module.render_text(report)
    assert "Tier" in text
    assert "T1" in text
    assert "T2" in text
    assert "T3" in text
    assert "T4" in text


def test_render_text_emits_operator_action_on_over_cadence(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    # 3 T4 deliberations -> T4 budget is 2 -> OVER_CADENCE.
    for i in range(3):
        _write_anchor(
            posterior,
            deliberation_id=f"t4_delib_{i:03d}",
            tier="T4",
            written_at_utc=(now - _dt.timedelta(days=i)).isoformat(timespec="seconds"),
        )
    report = cadence_module.audit_cadence(posterior_path=posterior)
    text = cadence_module.render_text(report)
    assert "OPERATOR ACTION REQUESTED" in text
    assert "stop-and-consolidate" in text.lower()


def test_cli_json_output_smoke(tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    _write_anchor(
        posterior,
        deliberation_id="cli_smoke",
        tier="T2",
        written_at_utc=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        # Use frontier_breaking so the rigor-dominant mission-alignment alert
        # does NOT fire on this single-row smoke fixture (would flip exit
        # code to 1 per the post-extension contract).
        predicted_mission_contribution="frontier_breaking",
    )
    result = subprocess.run(
        [sys.executable, str(_TOOL_PATH), "--posterior-path", str(posterior), "--json"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    # Schema version bumped to v2 with the mission-alignment extension.
    assert payload["schema"] == "council_tier_cadence_audit_v2"
    assert payload["any_over_cadence"] is False
    assert payload["any_mission_alignment_alert_fired"] is False


def test_cli_returns_rc_1_on_over_cadence(tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    for i in range(3):
        _write_anchor(
            posterior,
            deliberation_id=f"t4_overcadence_{i:03d}",
            tier="T4",
            written_at_utc=(now - _dt.timedelta(days=i)).isoformat(timespec="seconds"),
        )
    result = subprocess.run(
        [sys.executable, str(_TOOL_PATH), "--posterior-path", str(posterior)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 1


# ───────────────────────────────────────────────────────────────────
# Mission-alignment alert tests (operator binding directive 2026-05-16).
# Per CLAUDE.md "Mission alignment — non-negotiable" subsection.
# ───────────────────────────────────────────────────────────────────


def test_mission_contribution_distribution_alert_fires_when_rigor_dominant(
    cadence_module, tmp_path: Path,
):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    # 7 apparatus_maintenance + 3 frontier_breaking = 70% rigor → fires.
    for i in range(7):
        _write_anchor(
            posterior,
            deliberation_id=f"app_{i}",
            tier="T2",
            written_at_utc=(now - _dt.timedelta(days=i)).isoformat(timespec="seconds"),
            predicted_mission_contribution="apparatus_maintenance",
        )
    for i in range(3):
        _write_anchor(
            posterior,
            deliberation_id=f"fb_{i}",
            tier="T2",
            written_at_utc=(now - _dt.timedelta(days=i)).isoformat(timespec="seconds"),
            predicted_mission_contribution="frontier_breaking",
        )
    alert = cadence_module.compute_mission_contribution_distribution_alert(
        posterior_path=posterior, lookback_days=30, now_utc=now,
    )
    assert alert.fired is True
    assert "RIGOR-DOMINANT" in alert.summary
    assert alert.details["total_t2_plus_deliberations"] == 10
    assert alert.details["rigor_overhead_plus_apparatus_count"] == 7


def test_mission_contribution_distribution_alert_silent_when_frontier_dominant(
    cadence_module, tmp_path: Path,
):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    # 2 apparatus_maintenance + 8 frontier_breaking = 20% rigor → silent.
    for i in range(2):
        _write_anchor(
            posterior,
            deliberation_id=f"app_{i}",
            tier="T2",
            written_at_utc=(now - _dt.timedelta(days=i)).isoformat(timespec="seconds"),
            predicted_mission_contribution="apparatus_maintenance",
        )
    for i in range(8):
        _write_anchor(
            posterior,
            deliberation_id=f"fb_{i}",
            tier="T2",
            written_at_utc=(now - _dt.timedelta(days=i)).isoformat(timespec="seconds"),
            predicted_mission_contribution="frontier_breaking",
        )
    alert = cadence_module.compute_mission_contribution_distribution_alert(
        posterior_path=posterior, lookback_days=30, now_utc=now,
    )
    assert alert.fired is False
    assert "OK" in alert.summary


def test_overdue_retrospective_alert_fires(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    # Retrospective due 30 days ago.
    overdue_due = (now - _dt.timedelta(days=30)).isoformat(timespec="seconds")
    _write_anchor(
        posterior,
        deliberation_id="deferred_substrate_overdue",
        tier="T2",
        written_at_utc=now.isoformat(timespec="seconds"),
        predicted_mission_contribution="frontier_protecting",
        deferred_substrate_retrospective_due_utc=overdue_due,
        deferred_substrate_id="lane_some_substrate",
    )
    alert = cadence_module.compute_overdue_retrospective_alert(
        posterior_path=posterior, now_utc=now,
    )
    assert alert.fired is True
    assert alert.details["overdue_count"] == 1
    assert "OVERDUE" in alert.summary


def test_overdue_retrospective_alert_silent_for_future_due(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    # Retrospective due in 30 days.
    future_due = (now + _dt.timedelta(days=30)).isoformat(timespec="seconds")
    _write_anchor(
        posterior,
        deliberation_id="deferred_substrate_future",
        tier="T2",
        written_at_utc=now.isoformat(timespec="seconds"),
        predicted_mission_contribution="frontier_protecting",
        deferred_substrate_retrospective_due_utc=future_due,
        deferred_substrate_id="lane_some_substrate",
    )
    alert = cadence_module.compute_overdue_retrospective_alert(
        posterior_path=posterior, now_utc=now,
    )
    assert alert.fired is False


def test_overdue_retrospective_alert_silent_when_no_deferrals(
    cadence_module, tmp_path: Path,
):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    _write_anchor(
        posterior,
        deliberation_id="d1",
        tier="T2",
        written_at_utc=now.isoformat(timespec="seconds"),
        predicted_mission_contribution="frontier_breaking",
    )
    alert = cadence_module.compute_overdue_retrospective_alert(
        posterior_path=posterior, now_utc=now,
    )
    assert alert.fired is False


def test_annual_gate_audit_alert_fires_for_aged_gate(cadence_module, tmp_path: Path):
    # Build a tiny CLAUDE.md with a 2-year-old catalog entry.
    repo_root = tmp_path
    claude_md = repo_root / "CLAUDE.md"
    claude_md.write_text(
        "# header\n\n"
        "## Meta-bug class catalog\n\n"
        "1. `check_foo_bar` — Refused 2024-01-01 because reasons.\n"
    )
    # No audit memo exists for catalog #1.
    now = _dt.datetime(2026, 5, 16, tzinfo=_dt.UTC)
    alert = cadence_module.compute_annual_gate_audit_alert(
        repo_root=repo_root, now_utc=now,
    )
    assert alert.fired is True
    assert alert.details["overdue_count"] == 1
    assert alert.details["overdue_gates"][0]["catalog_number"] == 1


def test_annual_gate_audit_alert_silent_with_audit_memo(cadence_module, tmp_path: Path):
    # Build a tiny CLAUDE.md with a 2-year-old catalog entry AND its audit memo.
    repo_root = tmp_path
    claude_md = repo_root / "CLAUDE.md"
    claude_md.write_text(
        "# header\n\n"
        "## Meta-bug class catalog\n\n"
        "1. `check_foo_bar` — Refused 2024-01-01 because reasons.\n"
    )
    memo_dir = repo_root / ".omx" / "research"
    memo_dir.mkdir(parents=True)
    (memo_dir / "annual_gate_audit_catalog_1_2025.md").write_text(
        "audit memo body\n"
    )
    now = _dt.datetime(2026, 5, 16, tzinfo=_dt.UTC)
    alert = cadence_module.compute_annual_gate_audit_alert(
        repo_root=repo_root, now_utc=now,
    )
    assert alert.fired is False


def test_annual_gate_audit_alert_silent_for_new_gate(cadence_module, tmp_path: Path):
    repo_root = tmp_path
    claude_md = repo_root / "CLAUDE.md"
    claude_md.write_text(
        "# header\n\n"
        "## Meta-bug class catalog\n\n"
        "1. `check_foo_bar` — Refused 2026-04-01 because reasons.\n"
    )
    now = _dt.datetime(2026, 5, 16, tzinfo=_dt.UTC)
    alert = cadence_module.compute_annual_gate_audit_alert(
        repo_root=repo_root, now_utc=now,
    )
    assert alert.fired is False


def test_audit_cadence_includes_mission_alignment_alerts(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    _write_anchor(
        posterior,
        deliberation_id="d1",
        tier="T2",
        written_at_utc=now.isoformat(timespec="seconds"),
        predicted_mission_contribution="frontier_breaking",
    )
    report = cadence_module.audit_cadence(
        posterior_path=posterior, now_utc=now,
    )
    assert report["schema"] == "council_tier_cadence_audit_v2"
    assert "mission_alignment_alerts" in report
    assert len(report["mission_alignment_alerts"]) == 3
    assert "any_mission_alignment_alert_fired" in report
    alert_classes = {a["alert_class"] for a in report["mission_alignment_alerts"]}
    assert alert_classes == {
        "mission_contribution_distribution_alert",
        "overdue_retrospective_alert",
        "annual_gate_audit_alert",
    }


def test_render_text_includes_mission_alignment_section(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    _write_anchor(
        posterior,
        deliberation_id="d1",
        tier="T2",
        written_at_utc=now.isoformat(timespec="seconds"),
        predicted_mission_contribution="frontier_breaking",
    )
    report = cadence_module.audit_cadence(posterior_path=posterior, now_utc=now)
    text = cadence_module.render_text(report)
    assert "Mission-alignment alerts" in text
    assert "mission_contribution_distribution_alert" in text
    assert "overdue_retrospective_alert" in text
    assert "annual_gate_audit_alert" in text


def test_render_text_operator_action_on_mission_alert(cadence_module, tmp_path: Path):
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    # Single apparatus_maintenance row → 100% rigor-dominant → fires.
    _write_anchor(
        posterior,
        deliberation_id="d1",
        tier="T2",
        written_at_utc=now.isoformat(timespec="seconds"),
        predicted_mission_contribution="apparatus_maintenance",
    )
    report = cadence_module.audit_cadence(posterior_path=posterior, now_utc=now)
    text = cadence_module.render_text(report)
    assert "OPERATOR ACTION REQUESTED: at least one mission-alignment alert is FIRED" in text


def test_cli_returns_rc_1_on_mission_alignment_alert(tmp_path: Path):
    """CLI rc=1 when only a mission-alignment alert is fired (no over-cadence)."""
    posterior = tmp_path / "council.jsonl"
    now = _dt.datetime.now(_dt.UTC)
    _write_anchor(
        posterior,
        deliberation_id="d1",
        tier="T2",
        written_at_utc=now.isoformat(timespec="seconds"),
        predicted_mission_contribution="apparatus_maintenance",
    )
    result = subprocess.run(
        [sys.executable, str(_TOOL_PATH), "--posterior-path", str(posterior)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 1


def test_mission_alignment_alert_dataclass_pinned(cadence_module):
    """Sanity-check the dataclass shape is canonical."""
    assert hasattr(cadence_module, "MissionAlignmentAlert")
    cls = cadence_module.MissionAlignmentAlert
    instance = cls(
        alert_class="x",
        fired=False,
        summary="s",
        details={},
    )
    assert instance.alert_class == "x"
    assert instance.fired is False
