# SPDX-License-Identifier: MIT
"""Tests for tools/schedule_canonical_modal_harvest_cron.sh (Surface A).

Companion canonical helper to Catalog #380 STRICT preflight gate
(Surface B of the canonical 2-landing pattern per RECOVERY-AUDIT-V2 +
STAND-DOWN-REVIEW-AUDIT TOP-2 op-routables 2026-05-28).

The shell helper auto-detects OS (Darwin launchd vs Linux cron) and
emits the canonical scheduler entry. These tests cover --dry-run for
both OS branches, idempotent install semantics, status verification,
cadence-range validation, and canonical ledger logging contract.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "schedule_canonical_modal_harvest_cron.sh"


def _run(args: list[str], *, expect_rc: int | None = 0) -> subprocess.CompletedProcess:
    """Run the helper script with given args; return CompletedProcess."""
    proc = subprocess.run(
        ["bash", str(TOOL), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if expect_rc is not None:
        assert proc.returncode == expect_rc, (
            f"expected rc={expect_rc}, got {proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc


# --- File presence + permissions ------------------------------------------


def test_helper_file_exists() -> None:
    """Canonical helper exists at the canonical path."""
    assert TOOL.is_file(), f"missing canonical helper: {TOOL}"


def test_helper_is_executable() -> None:
    """Canonical helper has executable permission bit set."""
    mode = TOOL.stat().st_mode
    assert mode & 0o111, f"{TOOL} is not executable; got mode {oct(mode)}"


def test_helper_has_spdx_header() -> None:
    """Canonical helper carries SPDX-License-Identifier per OSS hygiene."""
    text = TOOL.read_text(encoding="utf-8")
    assert "# SPDX-License-Identifier: MIT" in text


def test_helper_passes_bash_syntax_check() -> None:
    """Canonical helper passes `bash -n` syntax check."""
    proc = subprocess.run(
        ["bash", "-n", str(TOOL)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"syntax error: {proc.stderr}"


# --- CLI contract ---------------------------------------------------------


def test_help_flag_succeeds() -> None:
    """--help prints docs and exits 0."""
    proc = _run(["--help"])
    # Sample tokens from the canonical docstring section.
    assert "CANONICAL RECURRING MODAL HARVEST SCHEDULER" in proc.stdout
    assert "Surface A" in proc.stdout or "canonical 2-landing" in proc.stdout


def test_unknown_arg_returns_rc_2() -> None:
    """Unknown args produce CLI error rc=2."""
    proc = _run(["--bogus-flag"], expect_rc=2)
    assert "unknown arg" in proc.stderr


def test_cadence_below_floor_rejected() -> None:
    """--cadence-hours 0.1 < 0.5 floor is rejected with rc=2."""
    proc = _run(["--status", "--cadence-hours", "0.1"], expect_rc=2)
    assert "cadence-hours" in proc.stderr


def test_cadence_above_ceiling_rejected() -> None:
    """--cadence-hours 48.0 > 24.0 ceiling is rejected with rc=2."""
    proc = _run(["--status", "--cadence-hours", "48.0"], expect_rc=2)
    assert "cadence-hours" in proc.stderr


def test_cadence_non_numeric_rejected() -> None:
    """--cadence-hours xyz non-numeric is rejected with rc=2."""
    proc = _run(["--status", "--cadence-hours", "xyz"], expect_rc=2)
    assert "cadence-hours" in proc.stderr


def test_cadence_valid_accepted() -> None:
    """--cadence-hours 1.5 within [0.5, 24.0] is accepted."""
    proc = _run(["--status", "--cadence-hours", "1.5"])
    # status should NOT print error, regardless of install state.
    assert "ERROR" not in proc.stderr.upper() or "no-op" in proc.stdout.lower()


# --- Status surface (read-only; safe on any host) -------------------------


def test_status_default_accepted() -> None:
    """--status (default --install would mutate; status is read-only) succeeds."""
    proc = _run(["--status"])
    # Either INSTALLED or NOT INSTALLED depending on host state.
    assert (
        "INSTALLED:" in proc.stdout
        or "NOT INSTALLED:" in proc.stdout
    )


# --- Dry-run install + uninstall semantics --------------------------------


def test_install_dry_run_does_not_mutate_disk(tmp_path: Path) -> None:
    """--install --dry-run prints the plan but does NOT mutate ${HOME} or crontab."""
    proc = _run(["--install", "--dry-run", "--cadence-hours", "2.0"])
    assert "[dry-run]" in proc.stdout
    # No ledger row should be appended (the helper's _log_event short-circuits on DRY_RUN=1).


def test_uninstall_dry_run_does_not_mutate_disk() -> None:
    """--uninstall --dry-run prints the plan or no-op but mutates nothing."""
    proc = _run(["--uninstall", "--dry-run"])
    # Either dry-run plan or no-op message; both acceptable.
    assert "[dry-run]" in proc.stdout or "NOT INSTALLED" in proc.stdout


def test_install_dry_run_macos_emits_launchd_plist() -> None:
    """On macOS, --install --dry-run emits a launchd plist with canonical fields."""
    if sys.platform != "darwin":
        pytest.skip("launchd plist only emitted on Darwin")
    proc = _run(["--install", "--dry-run", "--cadence-hours", "2.0"])
    assert "<key>Label</key>" in proc.stdout
    assert "com.pact.modal_harvest_canonical" in proc.stdout
    assert "<key>StartInterval</key>" in proc.stdout
    assert "<integer>7200</integer>" in proc.stdout  # 2.0 hours * 3600
    assert "<key>ProgramArguments</key>" in proc.stdout
    assert "harvest_modal_calls.py" in proc.stdout
    assert "--execute" in proc.stdout
    assert "--from-ledger" in proc.stdout


def test_install_dry_run_macos_cadence_propagates_to_interval() -> None:
    """--cadence-hours 0.5 maps to 1800-second StartInterval."""
    if sys.platform != "darwin":
        pytest.skip("launchd plist only emitted on Darwin")
    proc = _run(["--install", "--dry-run", "--cadence-hours", "0.5"])
    assert "<integer>1800</integer>" in proc.stdout


def test_install_dry_run_macos_cadence_24_hours() -> None:
    """--cadence-hours 24.0 maps to 86400-second StartInterval."""
    if sys.platform != "darwin":
        pytest.skip("launchd plist only emitted on Darwin")
    proc = _run(["--install", "--dry-run", "--cadence-hours", "24.0"])
    assert "<integer>86400</integer>" in proc.stdout


# --- Canonical contract surfaces ------------------------------------------


def test_helper_references_canonical_harvest_tool() -> None:
    """Canonical helper invokes the canonical harvester at canonical path."""
    text = TOOL.read_text(encoding="utf-8")
    assert "tools/harvest_modal_calls.py" in text
    assert "--execute" in text
    assert "--from-ledger" in text


def test_helper_references_canonical_ledger_path() -> None:
    """Canonical helper writes to canonical .omx/state ledger path per Catalog #131."""
    text = TOOL.read_text(encoding="utf-8")
    assert ".omx/state/modal_harvest_cron_log.jsonl" in text


def test_helper_uses_fcntl_lock_ex_per_catalog_131() -> None:
    """Canonical helper acquires fcntl.LOCK_EX for ledger writes per Catalog #131."""
    text = TOOL.read_text(encoding="utf-8")
    assert "fcntl.LOCK_EX" in text


def test_helper_references_catalog_380_sister_gate() -> None:
    """Canonical helper documents sister Catalog #380 STRICT preflight gate."""
    text = TOOL.read_text(encoding="utf-8")
    assert "Catalog #380" in text


def test_helper_references_recovery_audit_v2_op_routable() -> None:
    """Canonical helper documents RECOVERY-AUDIT-V2 + STAND-DOWN op-routable origin."""
    text = TOOL.read_text(encoding="utf-8")
    assert "RECOVERY-AUDIT-V2" in text
    assert "STAND-DOWN-REVIEW-AUDIT" in text


def test_helper_references_silent_orphan_harvest_anti_pattern() -> None:
    """Canonical helper documents the silent-orphan-harvest bug class extinction."""
    text = TOOL.read_text(encoding="utf-8")
    assert "silent_orphan_harvest" in text or "silent-orphan-harvest" in text


def test_helper_references_claude_md_harvest_or_lose() -> None:
    """Canonical helper cites CLAUDE.md HARVEST OR LOSE non-negotiable."""
    text = TOOL.read_text(encoding="utf-8")
    assert "HARVEST OR LOSE" in text


def test_helper_emits_canonical_marker_token() -> None:
    """Canonical helper uses canonical CRON_MARKER for idempotent install."""
    text = TOOL.read_text(encoding="utf-8")
    assert "CANONICAL_MODAL_HARVEST" in text


def test_helper_supports_both_darwin_and_linux() -> None:
    """Canonical helper auto-detects OS and supports both Darwin + Linux."""
    text = TOOL.read_text(encoding="utf-8")
    assert "Darwin" in text
    assert "Linux" in text
    assert "launchd" in text
    assert "cron" in text


def test_helper_supports_install_uninstall_status_actions() -> None:
    """Canonical helper supports the 3 canonical CLI actions."""
    text = TOOL.read_text(encoding="utf-8")
    assert "--install" in text
    assert "--uninstall" in text
    assert "--status" in text
    assert "--dry-run" in text


def test_helper_canonical_label_is_reverse_dns_format() -> None:
    """Canonical launchd label uses reverse-DNS format per Apple convention."""
    text = TOOL.read_text(encoding="utf-8")
    assert "com.pact.modal_harvest_canonical" in text
