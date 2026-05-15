# SPDX-License-Identifier: MIT
"""Dedicated tests for Catalog #279 (codex bklem3v5j F1 self-protection).

The gate refuses any state of ``tools/local_pre_deploy_check.py`` where the
8th harness check ``check_dispatch_optimization_protocol`` returns vacuous-
PASS on ImportError of the canonical Catalog #270 helper. Codex review
bklem3v5j HIGH (2026-05-15) recommended explicit fail-closed-on-import +
regression test that strict harness exits nonzero when helper missing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_local_pre_deploy_helper_import_failure_fails_closed,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET = REPO_ROOT / "tools" / "local_pre_deploy_check.py"


# -----------------------------------------------------------------------------
# Live-repo regression guard (the canonical post-fix state)
# -----------------------------------------------------------------------------


def test_live_repo_clean_after_f1_fix() -> None:
    """The fix landed in the same commit batch ⇒ live count = 0."""
    v = check_local_pre_deploy_helper_import_failure_fails_closed(
        repo_root=REPO_ROOT, strict=False
    )
    assert v == [], (
        "Live repo expected clean post Catalog #279 F1 fix; "
        f"got {len(v)} violation(s):\n" + "\n".join(v[:5])
    )


def test_live_repo_strict_clean() -> None:
    """STRICT mode on the live repo MUST NOT raise."""
    check_local_pre_deploy_helper_import_failure_fails_closed(
        repo_root=REPO_ROOT, strict=True
    )


# -----------------------------------------------------------------------------
# Positive tests (gate FLAGS the F1 regression)
# -----------------------------------------------------------------------------


def _write_target(tmp_root: Path, body: str) -> None:
    target = tmp_root / "tools" / "local_pre_deploy_check.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def test_flags_vacuous_pass_regression(tmp_path: Path) -> None:
    """If the F1 vacuous-pass returns, the gate flags it."""
    body = textwrap.dedent(
        '''
        def check_dispatch_optimization_protocol(trainer, recipe):
            try:
                import canonical_dispatch_optimization_protocol as proto
            except ImportError as exc:
                return True, f"PASS-VACUOUS: protocol helper missing ({exc})"
            return True, "PASS"
        '''
    )
    _write_target(tmp_path, body)
    v = check_local_pre_deploy_helper_import_failure_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert any("vacuous-pass token" in s for s in v), v


def test_flags_missing_required_tokens(tmp_path: Path) -> None:
    """If the fix-closed token set is missing, the gate flags it."""
    body = textwrap.dedent(
        '''
        def check_dispatch_optimization_protocol(trainer, recipe):
            try:
                import canonical_dispatch_optimization_protocol as proto
            except ImportError:
                pass  # no return; not the canonical fail-closed pattern
            return True, "PASS"
        '''
    )
    _write_target(tmp_path, body)
    v = check_local_pre_deploy_helper_import_failure_fails_closed(
        repo_root=tmp_path, strict=False
    )
    # Every required token absent -> multiple violations.
    assert len(v) >= 2, v


def test_strict_mode_raises_with_catalog_279(tmp_path: Path) -> None:
    """STRICT mode raises PreflightError citing Catalog #279."""
    body = 'def check_dispatch_optimization_protocol(t, r):\n    return True, "PASS-VACUOUS: protocol helper missing (x)"\n'
    _write_target(tmp_path, body)
    with pytest.raises(PreflightError) as exc_info:
        check_local_pre_deploy_helper_import_failure_fails_closed(
            repo_root=tmp_path, strict=True
        )
    msg = str(exc_info.value)
    assert "Catalog #279" in msg
    assert "fail-closed-on-import" in msg


def test_flags_missing_function(tmp_path: Path) -> None:
    """If the 8th harness check is removed entirely, gate flags regression."""
    body = "def some_other_check():\n    return True, 'unrelated'\n"
    _write_target(tmp_path, body)
    v = check_local_pre_deploy_helper_import_failure_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert any("missing" in s for s in v), v


# -----------------------------------------------------------------------------
# Negative tests (gate ACCEPTS the canonical fix)
# -----------------------------------------------------------------------------


def test_accepts_canonical_fail_closed_pattern(tmp_path: Path) -> None:
    """The canonical fix landed in the same commit batch passes the gate."""
    body = textwrap.dedent(
        '''
        def check_dispatch_optimization_protocol(trainer, recipe):
            """8th harness check (Catalog #270)."""
            sys.path.insert(0, "tools")
            try:
                import canonical_dispatch_optimization_protocol as proto
            except ImportError as exc:
                # F1 fix per Catalog #279 fail-closed-on-import discipline.
                return False, (
                    f"FAIL: Catalog #270 protocol helper unavailable "
                    f"({exc}); fail-closed-on-import per Catalog #279."
                )
            return True, "PASS"
        '''
    )
    _write_target(tmp_path, body)
    v = check_local_pre_deploy_helper_import_failure_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert v == [], v


# -----------------------------------------------------------------------------
# Waiver tests
# -----------------------------------------------------------------------------


def test_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    """File-level waiver with non-placeholder rationale exempts the file."""
    body = textwrap.dedent(
        '''
        # DISPATCH_PROTOCOL_IMPORT_FAILURE_ALLOW_VACUOUS_OK: reviewed-fixture-for-unit-test
        def check_dispatch_optimization_protocol(t, r):
            return True, "PASS-VACUOUS: protocol helper missing (x)"
        '''
    )
    _write_target(tmp_path, body)
    v = check_local_pre_deploy_helper_import_failure_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert v == [], v


def test_waiver_placeholder_rejected(tmp_path: Path) -> None:
    """Placeholder ``<rationale>`` literal does NOT honor the waiver."""
    body = textwrap.dedent(
        '''
        # DISPATCH_PROTOCOL_IMPORT_FAILURE_ALLOW_VACUOUS_OK: <rationale>
        def check_dispatch_optimization_protocol(t, r):
            return True, "PASS-VACUOUS: protocol helper missing (x)"
        '''
    )
    _write_target(tmp_path, body)
    v = check_local_pre_deploy_helper_import_failure_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert any("vacuous-pass token" in s for s in v), v


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_missing_target_file(tmp_path: Path) -> None:
    """No tools/local_pre_deploy_check.py ⇒ gate is silent."""
    v = check_local_pre_deploy_helper_import_failure_fails_closed(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_orchestrator_strict_wireup() -> None:
    """preflight_all() wires the gate at strict=True (regression guard)."""
    text = (REPO_ROOT / "src" / "tac" / "preflight.py").read_text(encoding="utf-8")
    # Find the orchestrator call and verify strict=True.
    idx = text.find("check_local_pre_deploy_helper_import_failure_fails_closed(")
    assert idx != -1, "orchestrator callsite missing"
    block = text[idx : idx + 200]
    assert "strict=True" in block, f"strict=True expected; got: {block!r}"


# -----------------------------------------------------------------------------
# Regression test per codex's explicit recommendation:
# "strict local pre-deploy exits nonzero when the helper cannot be imported"
# -----------------------------------------------------------------------------


def test_strict_harness_exits_nonzero_when_helper_missing(tmp_path: Path) -> None:
    """End-to-end: strict harness MUST exit rc=1 when helper missing.

    Per codex review bklem3v5j HIGH F1 explicit recommendation:
    *"Add regression test that strict local pre-deploy exits nonzero
    when the helper cannot be imported."*
    """
    # Build a minimal repro repo with the canonical harness but NO helper.
    # We don't need a real trainer — the harness exits before it reaches
    # any trainer-content check on the protocol-helper-import failure.
    repo = tmp_path
    (repo / "tools").mkdir()
    # Copy the canonical harness verbatim
    canonical = TARGET.read_text(encoding="utf-8")
    (repo / "tools" / "local_pre_deploy_check.py").write_text(canonical, encoding="utf-8")
    # Provide a trivial trainer so the harness reaches the 8th check
    (repo / "experiments").mkdir()
    trainer = repo / "experiments" / "train_substrate_test_fixture.py"
    trainer.write_text(
        textwrap.dedent('''
            """Test fixture trainer (Catalog #279 regression)."""
            def _full_main():
                pass
            def main():
                pass
        '''),
        encoding="utf-8",
    )
    # Run the canonical harness in strict mode from the helper-less repo
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "local_pre_deploy_check.py"),
            "--trainer",
            str(trainer),
            "--strict",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    # The harness MUST exit nonzero — per codex's explicit recommendation
    assert result.returncode != 0, (
        f"Strict harness exited rc=0 when helper missing — F1 regression!\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # And the FAIL message MUST cite Catalog #279
    assert "Catalog #279" in result.stdout or "Catalog #279" in result.stderr, (
        f"Expected Catalog #279 in output; got:\n{result.stdout}\n{result.stderr}"
    )
