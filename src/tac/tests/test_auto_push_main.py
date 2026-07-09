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


# ------------------------------- fmtools advisory helpers ----------------------
def test_added_lines_extracts_only_plus_lines():
    diff = (
        "diff --git a/x b/x\n"
        "--- a/x\n"
        "+++ b/x\n"       # header — must be dropped
        "+added one\n"
        " context\n"      # unchanged context — must be dropped
        "-removed\n"      # removal — must be dropped
        "+added two\n"
    )
    assert A._added_lines(diff) == "added one\nadded two"


def test_fm_advisory_none_on_empty_text_never_spawns():
    # Empty added-text short-circuits to None with no subprocess (fast, no FM).
    assert A.fm_secret_advisory("") is None
    assert A.fm_secret_advisory("   \n  ") is None


# ------------------------------- self-reference exclusion ----------------------
def test_scanner_excludes_its_own_source_and_tests():
    # The scanner + its tests carry secret PATTERNS (regex defs) + secret-SHAPED fixtures by design.
    # The outgoing-diff scan MUST exclude them (git pathspec) or it holds every commit touching itself —
    # it held its own bootstrap push 2026-07-09. Any new file carrying example secrets is added here.
    assert "tools/auto_push_main.py" in A._SCAN_EXCLUDE_PATHSPECS
    assert "src/tac/tests/test_auto_push_main.py" in A._SCAN_EXCLUDE_PATHSPECS
    assert ".gitleaks.toml" in A._SCAN_EXCLUDE_PATHSPECS  # documents the CGNAT range literally


# ------------------------------- gitleaks third layer --------------------------
def test_gitleaks_config_has_tailscale_rule_and_self_allowlist():
    # The gitleaks config inherits the curated ruleset, adds the Tailscale CGNAT rule the default set
    # lacks, and allowlists the self-reference files (same exclusion the regex layer uses).
    cfg = (_REPO / ".gitleaks.toml").read_text()
    assert "useDefault = true" in cfg
    assert "tailscale-cgnat-fleet-ip" in cfg
    assert "tools/auto_push_main" in cfg  # self-exclusion allowlist path


def test_gitleaks_scan_is_fail_open_when_config_absent(tmp_path):
    # No .gitleaks.toml under tmp_path ⇒ gitleaks_scan returns None (skip / fail-open), never raises.
    assert A.gitleaks_scan(tmp_path) is None


# ------------------------------- FM prompt-echo filter -------------------------
def test_fm_prompt_echo_detects_hallucinated_exemplar():
    # The FM once echoed its own few-shot exemplar as a "finding" on a diff that did not
    # contain it (2026-07-09). Reason cites the exemplar, scanned text lacks it -> echo.
    reason = "DATABASE_URL = postgres://user:S3cr3tPw9x@host/db"
    assert A._fm_prompt_echo(reason, "def compute_score(seg, pose, rate): ...") is True


def test_fm_prompt_echo_passes_when_exemplar_truly_in_diff():
    # If the diff GENUINELY contains the exemplar string, it is a real finding, not an echo.
    reason = "embedded password S3cr3tPw9x"
    scanned = "+DATABASE_URL = postgres://user:S3cr3tPw9x@host/db\n"
    assert A._fm_prompt_echo(reason, scanned) is False


def test_fm_prompt_echo_clean_reasons_untouched():
    assert A._fm_prompt_echo("private fleet IP found", "ssh 100.96.1.2") is False
    assert A._fm_prompt_echo("", "anything") is False


# ------------------------------- integration smoke -----------------------------
# All subprocess smokes pass --dry-run --no-fmtools: --dry-run guarantees NO push
# side-effect; --no-fmtools skips the on-device FM subprocess (fast + deterministic
# even when the fmtools venv is present on the host).
def test_dry_run_reports_a_decision_and_never_pushes():
    r = subprocess.run([sys.executable, str(_TOOL), "--dry-run", "--no-fmtools"],
                       capture_output=True, text=True, timeout=60, input="")
    assert r.returncode == 0
    assert "[auto-push]" in r.stdout
    # dry-run stdout is NOT the Stop-hook decision protocol.
    assert '"decision"' not in r.stdout


def test_hook_is_fail_open_on_stop_payload():
    # Fed a real Stop-hook JSON payload with stop_hook_active=True, the hook must
    # exit 0 and emit NO decision block. --dry-run keeps it push-free.
    payload = json.dumps({"stop_hook_active": True})
    r = subprocess.run([sys.executable, str(_TOOL), "--dry-run", "--no-fmtools"],
                       capture_output=True, text=True, timeout=60, input=payload)
    assert r.returncode == 0
    assert '"decision"' not in r.stdout
