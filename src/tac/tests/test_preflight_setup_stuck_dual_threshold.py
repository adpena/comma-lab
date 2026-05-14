# SPDX-License-Identifier: MIT
"""Tests for Preflight Check 56 — verify_vast_instances dual-threshold guard.

- Check 56: ``check_verify_vast_setup_stuck_dual_threshold`` —
  scripts/verify_vast_instances.py must define BOTH ``--stale-minutes``
  AND ``--setup-stale-minutes``, AND the ``args.auto_destroy_stale``
  branch must consult both.

Class of bug: heuristic-based health classifier with NO timeout for the
in-flight SETUP state. A TRULY hung setup_full.sh (deadlocked, never
writes heartbeat) is classified SETUP (not IDLE), so the IDLE
``--stale-minutes`` heartbeat-freshness comparison never fires — the
instance accrues GPU cost silently forever.

Reference: feedback_setup_stuck_cost_leak_FIXED_20260428.

Magnitude/value anchors (per Round 26 convention):
  - 5+ unit tests
  - 0 live violations after wire-in
  - STRICT in preflight_all() at land time
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_verify_vast_setup_stuck_dual_threshold,
    preflight_all,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Bare repo skeleton with scripts/ dir."""
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_verify(repo: Path, body: str) -> Path:
    p = repo / "scripts" / "verify_vast_instances.py"
    p.write_text(body)
    return p


# ── Canonical green path ─────────────────────────────────────────────────


_CANONICAL_BODY = '''\
"""Per-instance health verify — fixture."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--auto-destroy-stale", action="store_true")
    p.add_argument("--stale-minutes", type=float, default=30.0)
    p.add_argument("--setup-stale-minutes", type=float, default=90.0)
    args = p.parse_args()

    healths = []  # populated elsewhere

    if args.auto_destroy_stale:
        to_destroy = [h for h in healths if h.classification in ("IDLE", "CRASHED")]
        stuck_setup = [
            h for h in healths
            if h.classification == "SETUP"
            and h.setup_age_minutes is not None
            and h.setup_age_minutes > args.setup_stale_minutes
        ]
        to_destroy.extend(stuck_setup)
        for h in to_destroy:
            print(f"destroy {h.instance_id}")
'''


def test_check_passes_on_canonical_dual_threshold(fake_repo):
    """Canonical: both flags + both thresholds in auto-destroy → 0 viols."""
    _write_verify(fake_repo, _CANONICAL_BODY)
    violations = check_verify_vast_setup_stuck_dual_threshold(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert violations == [], f"expected 0 violations, got: {violations}"


# ── Negative cases (each missing element triggers a distinct violation) ──


def test_check_fails_when_setup_stale_minutes_flag_missing(fake_repo):
    """Drop --setup-stale-minutes definition → violation."""
    body = _CANONICAL_BODY.replace(
        'p.add_argument("--setup-stale-minutes", type=float, default=90.0)\n',
        "",
    )
    _write_verify(fake_repo, body)
    violations = check_verify_vast_setup_stuck_dual_threshold(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any("--setup-stale-minutes" in v for v in violations), (
        f"expected --setup-stale-minutes violation, got: {violations}"
    )


def test_check_fails_when_stale_minutes_flag_missing(fake_repo):
    """Drop --stale-minutes definition → IDLE half of dual-threshold gone."""
    body = _CANONICAL_BODY.replace(
        'p.add_argument("--stale-minutes", type=float, default=30.0)\n',
        "",
    )
    _write_verify(fake_repo, body)
    violations = check_verify_vast_setup_stuck_dual_threshold(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any(
        "--stale-minutes" in v and "--setup-stale-minutes" not in v
        for v in violations
    ), f"expected --stale-minutes violation, got: {violations}"


def test_check_fails_when_auto_destroy_drops_setup_branch(fake_repo):
    """Auto-destroy that only fires on IDLE/CRASHED (no SETUP filter)
    leaks cost on hung setup_full.sh → violation."""
    body = _CANONICAL_BODY.replace(
        '''        stuck_setup = [
            h for h in healths
            if h.classification == "SETUP"
            and h.setup_age_minutes is not None
            and h.setup_age_minutes > args.setup_stale_minutes
        ]
        to_destroy.extend(stuck_setup)
''',
        "",
    )
    _write_verify(fake_repo, body)
    violations = check_verify_vast_setup_stuck_dual_threshold(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any(
        "setup_age_minutes" in v or "setup_stale_minutes" in v
        for v in violations
    ), f"expected setup_age_minutes / setup_stale_minutes violation, got: {violations}"


def test_check_fails_when_auto_destroy_drops_idle_branch(fake_repo):
    """Auto-destroy without IDLE classification → IDLE half gone."""
    body = _CANONICAL_BODY.replace(
        'to_destroy = [h for h in healths if h.classification in ("IDLE", "CRASHED")]',
        "to_destroy = []",
    )
    _write_verify(fake_repo, body)
    violations = check_verify_vast_setup_stuck_dual_threshold(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any(
        "IDLE" in v and "auto-destroy" in v for v in violations
    ), f"expected IDLE-in-auto-destroy violation, got: {violations}"


def test_check_fails_when_auto_destroy_branch_absent(fake_repo):
    """Whole `args.auto_destroy_stale` branch removed → violation."""
    body = '''\
"""no auto-destroy variant."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stale-minutes", type=float, default=30.0)
    p.add_argument("--setup-stale-minutes", type=float, default=90.0)
    p.parse_args()
    # nothing else — no destroy logic
'''
    _write_verify(fake_repo, body)
    violations = check_verify_vast_setup_stuck_dual_threshold(
        repo_root=fake_repo, strict=False, verbose=False
    )
    assert any(
        "args.auto_destroy_stale" in v for v in violations
    ), f"expected args.auto_destroy_stale violation, got: {violations}"


def test_check_strict_raises(fake_repo):
    """strict=True raises MetaBugViolation when violations present."""
    body = _CANONICAL_BODY.replace(
        'p.add_argument("--setup-stale-minutes", type=float, default=90.0)\n',
        "",
    )
    _write_verify(fake_repo, body)
    with pytest.raises(MetaBugViolation, match="DUAL-THRESHOLD"):
        check_verify_vast_setup_stuck_dual_threshold(
            repo_root=fake_repo, strict=True, verbose=False
        )


def test_check_skipped_when_no_script(fake_repo):
    """Repo without scripts/verify_vast_instances.py → 0 violations (skip)."""
    violations = check_verify_vast_setup_stuck_dual_threshold(
        repo_root=fake_repo, strict=True, verbose=False
    )
    assert violations == []


# ── Wiring + live-codebase parity ────────────────────────────────────────


def test_check_56_wired_into_preflight_all():
    """preflight_all() must invoke Check 56 at strict=True.

    Source-text introspection guards against future refactors that
    silently drop the wiring.
    """
    src = inspect.getsource(preflight_all)
    assert "check_verify_vast_setup_stuck_dual_threshold(" in src, (
        "preflight_all must call check_verify_vast_setup_stuck_dual_threshold "
        "(Check 56)"
    )
    assert "strict=True" in src.split(
        "check_verify_vast_setup_stuck_dual_threshold("
    )[1].split(")")[0] + ")", (
        "Check 56 must be invoked with strict=True (the wire-in pattern is "
        "0 live violations → straight to STRICT, per the Lane A pattern)"
    )


def test_live_codebase_check_56_zero_violations():
    """The actual repo must satisfy Check 56 right now (0 live violations).

    Magnitude anchor: 0 live violations after the R31 SETUP-stuck fix
    landed. If this test fails, the verify script regressed — either a
    flag was dropped or the auto-destroy path no longer consults both
    thresholds. That's the regression class this guard exists to prevent.
    """
    violations = check_verify_vast_setup_stuck_dual_threshold(
        strict=False, verbose=False
    )
    assert violations == [], (
        f"expected 0 live violations, got {len(violations)}: {violations}"
    )
