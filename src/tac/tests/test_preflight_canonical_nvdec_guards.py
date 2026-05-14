# SPDX-License-Identifier: MIT
"""Tests for Preflight Checks 54 & 55 — canonical NVDEC workflow guards.

- Check 54: ``check_phase2_launch_polls_setup_log`` —
  scripts/launch_lane_on_vastai.py cmd_phase2_launch must call
  _poll_setup_log_for_outcome AND honor a skip_post_verify opt-in.
- Check 55: ``check_setup_full_probe_before_dali`` —
  scripts/remote_setup_full.sh must invoke probe_nvdec.sh
  --lightweight BEFORE the nvidia-dali-cuda120 install.

Both checks make the 2-layer NVDEC fix (commits 58e55890 + 5acebb88)
structurally permanent. Reference:
feedback_canonical_nvdec_workflow_GUARD_20260428.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_phase2_launch_polls_setup_log,
    check_setup_full_probe_before_dali,
    preflight_all,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Bare repo skeleton with scripts/ dir."""
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ── Check 54: phase2-launch polls setup.log ──────────────────────────────


_LAUNCHER_HEADER = (
    '"""Launcher fixture."""\n'
    "import argparse\n"
    "import sys\n\n"
)


def _write_launcher(repo: Path, body: str) -> Path:
    p = repo / "scripts" / "launch_lane_on_vastai.py"
    p.write_text(_LAUNCHER_HEADER + body)
    return p


def test_check_phase2_launch_passes_when_poll_present(fake_repo):
    """Canonical phase2-launch with both poll AND skip opt-in → 0 violations."""
    body = (
        "def _poll_setup_log_for_outcome(host, port, instance_id, timeout_seconds=60):\n"
        "    return 'SETUP_COMPLETE'\n\n"
        "def cmd_phase2_launch(args):\n"
        "    if not getattr(args, 'skip_post_verify', False):\n"
        "        outcome = _poll_setup_log_for_outcome(args.host, args.port, args.instance_id, timeout_seconds=60)\n"
        "        if outcome == 'NVDEC_BAD':\n"
        "            return 2\n"
        "    return 0\n"
    )
    _write_launcher(fake_repo, body)
    violations = check_phase2_launch_polls_setup_log(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert violations == [], f"expected 0 violations, got: {violations}"


def test_check_phase2_launch_fails_when_poll_missing(fake_repo):
    """Phase2-launch missing the _poll_setup_log_for_outcome call → fail."""
    body = (
        "def cmd_phase2_launch(args):\n"
        "    # fire-and-forget — but no skip_post_verify gate either\n"
        "    print('launched')\n"
        "    return 0\n"
    )
    _write_launcher(fake_repo, body)
    violations = check_phase2_launch_polls_setup_log(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert len(violations) == 1, f"expected 1 violation, got: {violations}"
    assert "_poll_setup_log_for_outcome" in violations[0]
    assert "skip_post_verify" in violations[0]


def test_check_phase2_launch_fails_when_only_skip_flag_present(fake_repo):
    """Skip flag alone is not enough — the poll must also be wired."""
    body = (
        "def cmd_phase2_launch(args):\n"
        "    if not getattr(args, 'skip_post_verify', False):\n"
        "        print('would poll here but does not')\n"
        "    return 0\n"
    )
    _write_launcher(fake_repo, body)
    violations = check_phase2_launch_polls_setup_log(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "_poll_setup_log_for_outcome" in violations[0]


def test_check_phase2_launch_fails_when_only_poll_present_no_skip(fake_repo):
    """Poll without an explicit skip opt-in is also a violation —
    operators need a documented fire-and-forget path."""
    body = (
        "def _poll_setup_log_for_outcome(host, port, instance_id):\n"
        "    return 'SETUP_COMPLETE'\n\n"
        "def cmd_phase2_launch(args):\n"
        "    outcome = _poll_setup_log_for_outcome(args.host, args.port, args.instance_id)\n"
        "    return 0\n"
    )
    _write_launcher(fake_repo, body)
    violations = check_phase2_launch_polls_setup_log(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "skip_post_verify" in violations[0]


def test_check_phase2_launch_fails_when_function_missing(fake_repo):
    """Launcher without cmd_phase2_launch at all → violation."""
    body = "def cmd_phase1(args):\n    return 0\n"
    _write_launcher(fake_repo, body)
    violations = check_phase2_launch_polls_setup_log(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "cmd_phase2_launch" in violations[0]


def test_check_phase2_launch_strict_raises(fake_repo):
    body = (
        "def cmd_phase2_launch(args):\n"
        "    return 0\n"
    )
    _write_launcher(fake_repo, body)
    with pytest.raises(MetaBugViolation, match="PHASE2-LAUNCH POLL"):
        check_phase2_launch_polls_setup_log(
            repo_root=fake_repo, strict=True, verbose=False
        )


def test_check_phase2_launch_skipped_when_no_launcher(fake_repo):
    """Repo without scripts/launch_lane_on_vastai.py → 0 violations (skip)."""
    violations = check_phase2_launch_polls_setup_log(
        repo_root=fake_repo, strict=True, verbose=False
    )
    assert violations == []


# ── Check 55: setup_full probe before DALI ────────────────────────────────


def _write_setup_full(repo: Path, body: str) -> Path:
    p = repo / "scripts" / "remote_setup_full.sh"
    p.write_text(body)
    return p


def test_check_setup_full_passes_when_probe_before_dali(fake_repo):
    """Canonical ordering: lightweight probe at Stage 0.5, DALI at Stage 3."""
    body = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "echo '=== Stage 0.5 ==='\n"
        "bash $WORKSPACE/scripts/probe_nvdec.sh --lightweight || exit 2\n"
        "echo '=== Stage 1: apt deps ==='\n"
        "apt-get install -y ffmpeg\n"
        "echo '=== Stage 3: DALI ==='\n"
        "pip install nvidia-dali-cuda120\n"
    )
    _write_setup_full(fake_repo, body)
    violations = check_setup_full_probe_before_dali(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert violations == [], f"expected 0 violations, got: {violations}"


def test_check_setup_full_fails_when_probe_after_dali(fake_repo):
    """Probe AFTER DALI install → defeats savings purpose → fail."""
    body = (
        "#!/bin/bash\n"
        "echo '=== Stage 3: DALI ==='\n"
        "pip install nvidia-dali-cuda120\n"
        "echo '=== Stage 4 ==='\n"
        "bash $WORKSPACE/scripts/probe_nvdec.sh --lightweight || exit 2\n"
    )
    _write_setup_full(fake_repo, body)
    violations = check_setup_full_probe_before_dali(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert len(violations) == 1, f"expected 1 violation, got: {violations}"
    assert "AFTER" in violations[0]


def test_check_setup_full_passes_when_neither_present(fake_repo):
    """No probe AND no DALI install → opt-out → 0 violations."""
    body = (
        "#!/bin/bash\n"
        "echo '=== Stage 1 ==='\n"
        "apt-get install -y ffmpeg\n"
    )
    _write_setup_full(fake_repo, body)
    violations = check_setup_full_probe_before_dali(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_setup_full_passes_when_probe_only_no_dali(fake_repo):
    """Probe present but no DALI ⇒ nothing to defeat ⇒ 0 violations."""
    body = (
        "#!/bin/bash\n"
        "bash $WORKSPACE/scripts/probe_nvdec.sh --lightweight || exit 2\n"
        "echo 'no DALI install in this variant'\n"
    )
    _write_setup_full(fake_repo, body)
    violations = check_setup_full_probe_before_dali(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert violations == []


def test_check_setup_full_fails_when_dali_only_no_probe(fake_repo):
    """DALI install with no Stage 0.5 probe → violation."""
    body = (
        "#!/bin/bash\n"
        "echo '=== Stage 3: DALI ==='\n"
        "pip install nvidia-dali-cuda120\n"
    )
    _write_setup_full(fake_repo, body)
    violations = check_setup_full_probe_before_dali(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "no `probe_nvdec.sh --lightweight`" in violations[0]


def test_check_setup_full_ignores_probe_in_comments(fake_repo):
    """Probe appearing only in a comment header doesn't satisfy the check —
    comment lines are space-padded so they cannot count for ordering."""
    body = (
        "#!/bin/bash\n"
        "# This script uses probe_nvdec.sh --lightweight at Stage 0.5\n"
        "echo '=== Stage 3: DALI ==='\n"
        "pip install nvidia-dali-cuda120\n"
    )
    _write_setup_full(fake_repo, body)
    violations = check_setup_full_probe_before_dali(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "no `probe_nvdec.sh --lightweight`" in violations[0]


def test_check_setup_full_strict_raises(fake_repo):
    body = (
        "#!/bin/bash\n"
        "pip install nvidia-dali-cuda120\n"
        "bash probe_nvdec.sh --lightweight\n"
    )
    _write_setup_full(fake_repo, body)
    with pytest.raises(MetaBugViolation, match="SETUP_FULL NVDEC PROBE"):
        check_setup_full_probe_before_dali(
            repo_root=fake_repo, strict=True, verbose=False
        )


def test_check_setup_full_skipped_when_no_script(fake_repo):
    """Repo without scripts/remote_setup_full.sh → 0 violations (skip)."""
    violations = check_setup_full_probe_before_dali(
        repo_root=fake_repo, strict=True, verbose=False
    )
    assert violations == []


# ── Wiring: both checks must land in preflight_all() ─────────────────────


def test_both_checks_wired_into_preflight_all():
    """preflight_all() must invoke BOTH new checks, both at strict=True.

    Source-text introspection guards against future refactors that
    silently drop either wiring.
    """
    src = inspect.getsource(preflight_all)
    assert "check_phase2_launch_polls_setup_log(strict=True" in src, (
        "preflight_all must call check_phase2_launch_polls_setup_log "
        "with strict=True (Check 54)"
    )
    assert "check_setup_full_probe_before_dali(strict=True" in src, (
        "preflight_all must call check_setup_full_probe_before_dali "
        "with strict=True (Check 55)"
    )


# ── Live-codebase parity: 0 live violations after this commit ─────────────


def test_live_codebase_check_54_zero_violations():
    """The actual repo must satisfy Check 54 right now (0 live violations).

    If this test fails, the launcher regressed and phase2-launch is no
    longer polling setup.log — that's the regression class this guard
    exists to prevent. Anchor: 0 live violations at wire-in time.
    """
    violations = check_phase2_launch_polls_setup_log(
        strict=False, verbose=False
    )
    assert violations == [], (
        f"expected 0 live violations, got {len(violations)}: {violations}"
    )


def test_live_codebase_check_55_zero_violations():
    """The actual repo must satisfy Check 55 right now (0 live violations).

    If this test fails, scripts/remote_setup_full.sh regressed and the
    lightweight NVDEC probe is no longer running before the DALI install.
    Anchor: 0 live violations at wire-in time.
    """
    violations = check_setup_full_probe_before_dali(
        strict=False, verbose=False
    )
    assert violations == [], (
        f"expected 0 live violations, got {len(violations)}: {violations}"
    )
