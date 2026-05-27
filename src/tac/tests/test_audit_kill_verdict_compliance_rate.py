# SPDX-License-Identifier: MIT
"""Tests for tools/audit_kill_verdict_compliance_rate.py (ML3 apparatus-process change).

Per META-RESURRECTION-AUDIT-V2 op-routables Item #4 ML3. Uses a synthetic
probe-outcomes ledger so the test does not depend on live canonical state.
"""
from __future__ import annotations

import json

import tools.audit_kill_verdict_compliance_rate as audit


def _write_ledger(tmp_path, rows):
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    path = state_dir / "probe_outcomes.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return path


def test_classify_compliant_kill_with_reactivation():
    row = {
        "verdict": "KILL",
        "blocker_status": "blocking",
        "reactivation_criteria": "re-run cycle 0 with proper fine-tune",
    }
    assert audit._classify_compliance(row) == "COMPLIANT"


def test_classify_non_compliant_kill_missing_reactivation():
    row = {"verdict": "KILL", "blocker_status": "blocking"}
    assert audit._classify_compliance(row) == "NON_COMPLIANT_KILL"


def test_classify_defer_with_next_action_compliant():
    row = {"verdict": "DEFER", "blocker_status": "advisory", "next_action": "operator_authorize_rerun"}
    assert audit._classify_compliance(row) == "COMPLIANT"


def test_classify_defer_missing_both_indeterminate():
    row = {"verdict": "DEFER", "blocker_status": "blocking"}
    assert audit._classify_compliance(row) == "INDETERMINATE"


def test_classify_positive_verdict_always_compliant():
    for v in ("PROCEED", "PROMOTE", "PARTIAL", "INDEPENDENT"):
        assert audit._classify_compliance({"verdict": v}) == "COMPLIANT"


def test_clean_ledger_100_percent_no_alert(tmp_path):
    rows = [
        {"probe_id": "a", "substrate": "s1", "verdict": "DEFER", "blocker_status": "advisory", "next_action": "x"},
        {"probe_id": "b", "substrate": "s2", "verdict": "KILL", "blocker_status": "blocking", "reactivation_criteria": "y"},
        {"probe_id": "c", "substrate": "s3", "verdict": "PROCEED"},
    ]
    _write_ledger(tmp_path, rows)
    report = audit.audit_kill_verdict_compliance(repo_root=tmp_path)
    # PROCEED excluded from kill-compliance denominator.
    assert report["total_negative_result_verdicts"] == 2
    assert report["compliance_rate"] == 1.0
    assert report["stop_and_consolidate_alert"] is False


def test_below_threshold_fires_stop_and_consolidate(tmp_path):
    rows = [
        {"probe_id": "k1", "substrate": "s1", "verdict": "KILL", "blocker_status": "blocking"},  # non-compliant
        {"probe_id": "k2", "substrate": "s2", "verdict": "KILL", "blocker_status": "blocking"},  # non-compliant
        {"probe_id": "d1", "substrate": "s3", "verdict": "DEFER", "blocker_status": "advisory", "next_action": "x"},  # compliant
    ]
    _write_ledger(tmp_path, rows)
    report = audit.audit_kill_verdict_compliance(repo_root=tmp_path, threshold=0.90)
    assert report["total_negative_result_verdicts"] == 3
    assert report["n_non_compliant_kill"] == 2
    assert report["compliance_rate"] < 0.90
    assert report["stop_and_consolidate_alert"] is True


def test_latest_per_probe_wins(tmp_path):
    rows = [
        {"probe_id": "p", "substrate": "s", "verdict": "KILL", "blocker_status": "blocking"},  # earlier non-compliant
        {"probe_id": "p", "substrate": "s", "verdict": "KILL", "blocker_status": "blocking", "reactivation_criteria": "ratified"},  # later compliant
    ]
    _write_ledger(tmp_path, rows)
    report = audit.audit_kill_verdict_compliance(repo_root=tmp_path)
    assert report["total_negative_result_verdicts"] == 1
    assert report["compliance_rate"] == 1.0


def test_missing_ledger_is_empty_compliant(tmp_path):
    (tmp_path / ".git").mkdir(exist_ok=True)
    report = audit.audit_kill_verdict_compliance(repo_root=tmp_path)
    assert report["total_negative_result_verdicts"] == 0
    assert report["compliance_rate"] == 1.0
    assert report["stop_and_consolidate_alert"] is False


def test_main_strict_rc(tmp_path):
    rows = [{"probe_id": "k", "substrate": "s", "verdict": "KILL", "blocker_status": "blocking"}]
    _write_ledger(tmp_path, rows)
    rc = audit.main(["--strict", "--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_json_rc0_on_clean(tmp_path):
    rows = [{"probe_id": "d", "substrate": "s", "verdict": "DEFER", "next_action": "x"}]
    _write_ledger(tmp_path, rows)
    rc = audit.main(["--json", "--strict", "--repo-root", str(tmp_path)])
    assert rc == 0


def test_live_repo_audit_runs():
    # Smoke: the tool runs against the live canonical ledger without raising.
    report = audit.audit_kill_verdict_compliance()
    assert "compliance_rate" in report
    assert 0.0 <= report["compliance_rate"] <= 1.0
