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
):
    """Append a synthetic anchor row matching the v1 schema."""
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
    )
    result = subprocess.run(
        [sys.executable, str(_TOOL_PATH), "--posterior-path", str(posterior), "--json"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema"] == "council_tier_cadence_audit_v1"
    assert payload["any_over_cadence"] is False


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
