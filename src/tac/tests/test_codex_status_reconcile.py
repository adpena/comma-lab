"""Tests for the hardened codex delegation/notification apparatus:
- codex_status._bucket reconciliation classifier (RUNNING/NEEDS_REVIEW/REVIEWED/DIED/STALE)
- codex_delegate de-confliction preflight (refuse duplicate-live-label unless --force)

These extinct the two session-observed failure modes: (a) a landed-but-un-dispositioned
arm masquerading as "done/fine" (drift/signal-loss), and (b) over-launching an arm whose
cluster a live arm already covers. See tools/codex_status.py + tools/codex_delegate.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import codex_delegate  # noqa: E402
import codex_status  # noqa: E402


# ---- _bucket: the reconciliation classifier -------------------------------------------
def test_bucket_running_when_alive_no_marker():
    assert codex_status._bucket(alive=True, done=None, disp=None, age_h=0.1) == "RUNNING"


def test_bucket_needs_review_when_landed_without_terminal_disposition():
    # done marker present but no disposition = the drift the operator flagged
    assert codex_status._bucket(alive=False, done={"rc": "0"}, disp=None, age_h=1.0) == "NEEDS_REVIEW"
    # held_entangled is NON-terminal ⇒ still needs review
    assert codex_status._bucket(False, {"rc": "0"}, "held_entangled", 1.0) == "NEEDS_REVIEW"


def test_bucket_reviewed_only_on_terminal_disposition():
    for st in ("reviewed_committed", "respawned", "closed"):
        assert codex_status._bucket(False, {"rc": "0"}, st, 1.0) == "REVIEWED"


def test_bucket_died_recent_vs_stale_old():
    # no proc, no marker, launched recently ⇒ DIED (actionable)
    assert codex_status._bucket(False, None, None, age_h=1.0) == "DIED"
    # ancient ⇒ STALE (suppressed noise)
    assert codex_status._bucket(False, None, None, age_h=99.0) == "STALE"
    # unknown age ⇒ STALE (don't cry wolf)
    assert codex_status._bucket(False, None, None, age_h=None) == "STALE"


def test_bucket_done_dominates_liveness_race():
    # a marker present + proc briefly still alive ⇒ classify by disposition, not RUNNING
    assert codex_status._bucket(True, {"rc": "0"}, "reviewed_committed", 0.1) == "REVIEWED"


# ---- de-confliction preflight ---------------------------------------------------------
def test_delegate_refuses_duplicate_live_label(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(codex_delegate, "_live_labels", lambda: ["warmstart_organ_n1_rl"])
    pf = tmp_path / "p.txt"
    pf.write_text("hi", encoding="utf-8")
    rc = codex_delegate.main([
        "--label", "warmstart_organ_n1_rl", "--prompt-file", str(pf), "--no-launch",
    ])
    assert rc == 3
    assert "REFUSED" in capsys.readouterr().out


def test_delegate_force_overrides_duplicate(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_delegate, "_live_labels", lambda: ["dupe"])
    monkeypatch.setattr(codex_delegate, "_append_ledger", lambda row: None)
    pf = tmp_path / "p.txt"
    pf.write_text("hi", encoding="utf-8")
    rc = codex_delegate.main(["--label", "dupe", "--prompt-file", str(pf), "--no-launch", "--force"])
    assert rc == 0  # --force bypasses the refusal


def test_delegate_allows_distinct_label(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_delegate, "_live_labels", lambda: ["other_arm"])
    monkeypatch.setattr(codex_delegate, "_append_ledger", lambda row: None)
    pf = tmp_path / "p.txt"
    pf.write_text("hi", encoding="utf-8")
    rc = codex_delegate.main(["--label", "fresh_arm", "--prompt-file", str(pf), "--no-launch"])
    assert rc == 0


def test_live_labels_empty_when_no_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_delegate, "LEDGER", tmp_path / "nope.jsonl")
    assert codex_delegate._live_labels() == []
