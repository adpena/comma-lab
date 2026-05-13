"""Tests for tools/diagnose_lane.py — covers parsing helpers and exit logic.

Network calls (Vast.ai, SSH) are NOT exercised; instead the helpers that
parse SSH output sections, archive bytes, scores, and heartbeat ages are
tested directly.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "diagnose_lane.py"

_spec = importlib.util.spec_from_file_location("_diagnose_lane_under_test", TOOL_PATH)
diagnose_lane = importlib.util.module_from_spec(_spec)
sys.modules["_diagnose_lane_under_test"] = diagnose_lane
_spec.loader.exec_module(diagnose_lane)  # type: ignore[union-attr]


def test_heartbeat_age_minutes_returns_none_on_empty():
    assert diagnose_lane._heartbeat_age_minutes("") is None


def test_heartbeat_age_minutes_parses_recent_mtime():
    # mtime = now - 5 minutes
    mtime = datetime.now(timezone.utc).timestamp() - 5 * 60
    section = f"{mtime} ./scripts/heartbeat.log"
    age = diagnose_lane._heartbeat_age_minutes(section)
    assert age is not None
    assert 4.5 < age < 5.5  # within ~30s window


def test_heartbeat_age_minutes_handles_garbage():
    assert diagnose_lane._heartbeat_age_minutes("not_a_number\n") is None
    assert diagnose_lane._heartbeat_age_minutes("\n") is None


def test_archive_bytes_parses_first_line():
    section = "345802 ./submissions/robust_current/archive.zip\n12000 ./old.zip"
    assert diagnose_lane._archive_bytes(section) == 345802


def test_archive_bytes_returns_none_on_empty():
    assert diagnose_lane._archive_bytes("") is None
    assert diagnose_lane._archive_bytes("garbage line") is None


def test_final_score_extracts_from_run_record():
    section = json.dumps({"score": 1.05, "status": "ok"})
    assert diagnose_lane._final_score(section) == 1.05


def test_final_score_handles_alt_keys():
    assert diagnose_lane._final_score('{"auth_score": 2.29}') == 2.29
    assert diagnose_lane._final_score('{"final_score": 0.90, "x": 1}') == 0.90
    assert diagnose_lane._final_score('{"total": 0.33}') == 0.33


def test_final_score_returns_none_when_absent():
    assert diagnose_lane._final_score("") is None
    assert diagnose_lane._final_score('{"other_field": 1.0}') is None


def test_render_handles_missing_fields():
    diag = diagnose_lane.LaneDiagnosis(instance_id="12345")
    out = diagnose_lane.render(diag)
    assert "12345" in out
    assert "(not in tracker)" in out


def test_render_includes_ssh_failed_reason():
    diag = diagnose_lane.LaneDiagnosis(
        instance_id="12345",
        ssh_failed_reason="connection refused",
    )
    out = diagnose_lane.render(diag)
    assert "SSH_FAILED" in out
    assert "connection refused" in out


def test_render_includes_log_tails():
    diag = diagnose_lane.LaneDiagnosis(
        instance_id="12345",
        ssh_host="1.2.3.4",
        ssh_port=22,
        heartbeat_age_minutes=2.5,
        log_tails={"RUN_LOG": "training step 100\nloss=0.5"},
    )
    out = diagnose_lane.render(diag)
    assert "--- RUN_LOG ---" in out
    assert "training step 100" in out


def test_render_flags_stale_heartbeat():
    diag = diagnose_lane.LaneDiagnosis(
        instance_id="12345",
        heartbeat_age_minutes=120.0,
    )
    out = diagnose_lane.render(diag)
    assert "STALE" in out


def test_render_flags_fresh_heartbeat():
    diag = diagnose_lane.LaneDiagnosis(
        instance_id="12345",
        heartbeat_age_minutes=1.0,
    )
    out = diagnose_lane.render(diag)
    assert "FRESH" in out


def test_run_returns_timeout_classification(monkeypatch):
    """Verify _run distinguishes timeout from generic exception (matches verify_vast_instances)."""
    import subprocess

    def fake_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, out, err = diagnose_lane._run(["ls"])
    assert rc == -1
    assert err == "TIMEOUT"


def test_run_returns_exception_classification(monkeypatch):
    import subprocess

    def fake_run(*a, **kw):
        raise OSError("permission denied")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, out, err = diagnose_lane._run(["ls"])
    assert rc == -1
    assert err.startswith("EXC:")
    assert "permission denied" in err


def test_lane_diagnosis_field_renamed_to_accrued_upload_cost_usd():
    """R32 Finding 2: prior field `accrued_cost_usd` was misleading —
    it stored vast_info.get('inet_up_cost'), the upload-network cost,
    NOT the accrued GPU compute spend. Downstream JSON consumers got a
    near-zero number that they mistook for GPU cost. The fix renames
    the field so the name matches what Vast.ai actually returns.

    Anchor: the misleading name must be GONE; the descriptive name must
    be PRESENT.
    """
    fields = {f.name for f in diagnose_lane.LaneDiagnosis.__dataclass_fields__.values()}
    assert "accrued_cost_usd" not in fields, (
        "the misleading field name `accrued_cost_usd` must not exist — "
        "it implied GPU compute cost but stored upload-network cost"
    )
    assert "accrued_upload_cost_usd" in fields, (
        "expected renamed field `accrued_upload_cost_usd` describing what "
        "Vast.ai's `inet_up_cost` actually returns"
    )
