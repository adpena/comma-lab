"""Tests for the auto-push Stop hook (tools/auto_push_main.py).

Covers the pure scan_diff() hygiene surface (the fail-safe gate that HOLDS a push
on a fleet-IP / credential hit), the marker loop-guard helpers, and an integration
smoke that proves the real hook is fail-open (exits 0, never wedges a session) and
that --dry-run reports a decision without touching the remote.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools" / "auto_push_main.py"


def _load():
    spec = importlib.util.spec_from_file_location("auto_push_main", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load()


# ------------------------------- scan_diff() -----------------------------------
def test_scan_flags_tailscale_fleet_ip():
    # 100.64.0.0/10 CGNAT range — the fleet IPs that must never reach a public remote.
    diff = "+    ssh user@100.96.14.201  # bat00\n"
    hits = A.scan_diff(diff)
    assert any(name == "tailscale_fleet_ip" for name, _ in hits)


def test_scan_flags_private_key_and_tokens():
    diff = (
        "+-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "+ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n"
        '+vast_api_key = "abcdef0123456789abcdef"\n'
    )
    names = {name for name, _ in A.scan_diff(diff)}
    assert "private_key_block" in names
    assert "github_token" in names
    assert "generic_secret_assignment" in names


def test_scan_clean_on_ordinary_code_and_public_ips():
    # A sha256, a loopback/public-DNS IP, and a normal 100.x that is NOT in CGNAT
    # range (100.200.x is public) must NOT false-positive.
    diff = (
        "+    sha = 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f70819a2b3c4d5e6'\n"
        "+    dns = '8.8.8.8'  # public\n"
        "+    x = 100.200.pi  # not an IP, not CGNAT\n"
        "+    def compute_score(seg, pose, rate):\n"
    )
    assert A.scan_diff(diff) == []


def test_scan_redacts_the_sample():
    diff = "+ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\n"
    hits = A.scan_diff(diff)
    assert hits, "token should be flagged"
    _, sample = hits[0]
    # The logged/surfaced sample is redacted (never the full secret).
    assert "…" in sample
    assert "ABCDEFGHIJKLMNOPQRST" not in sample


# ------------------------------- marker loop-guard -----------------------------
def test_marker_roundtrip(tmp_path):
    assert A._marker_head(tmp_path) == ""  # absent ⇒ empty
    A._write_marker(tmp_path, "deadbeef")
    assert A._marker_head(tmp_path) == "deadbeef"


# ------------------------------- _block_reason ---------------------------------
def test_block_reason_hold_and_failure_are_actionable():
    hold = A._block_reason({"action": "hold", "ahead": "3", "hits": "tailscale_fleet_ip(100.96…01)"})
    assert "HELD" in hold and "tailscale_fleet_ip" in hold
    fail = A._block_reason({"action": "push_failed", "rc": 1, "detail": "non-fast-forward"})
    assert "FAILED" in fail and "non-fast-forward" in fail


# ------------------------------- integration smoke -----------------------------
def test_dry_run_reports_a_decision_and_never_pushes():
    # --dry-run must exit 0 and print a verdict; it must never emit a decision
    # block (that channel is only for a live HOLD / push-failure).
    r = subprocess.run([sys.executable, str(_TOOL), "--dry-run"],
                       capture_output=True, text=True, timeout=60, input="")
    assert r.returncode == 0
    assert "[auto-push]" in r.stdout
    # dry-run stdout is NOT the Stop-hook decision protocol.
    assert '"decision"' not in r.stdout


def test_hook_is_fail_open_on_stop_payload():
    # Fed a real Stop-hook JSON payload with stop_hook_active=True, the hook must
    # exit 0 and (because we caused the stop) emit NO decision block — loop guard.
    payload = json.dumps({"stop_hook_active": True})
    r = subprocess.run([sys.executable, str(_TOOL)],
                       capture_output=True, text=True, timeout=60, input=payload)
    assert r.returncode == 0
    assert '"decision"' not in r.stdout
