# SPDX-License-Identifier: MIT
"""Catalog #339 — SILENT-NO-SPAWN-STRUCTURAL-EXTINCTION self-protection.

Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" + "Bugs must be permanently
fixed AND self-protected against" non-negotiables. Anchor: 3 consecutive
silent-no-spawn incidents 2026-05-19 (Z6 Wave 2 4c / STC v2 / STC sister).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from tac import preflight
from tac.preflight import (
    PreflightError,
    check_modal_dispatcher_registers_call_id_before_successful_exit,
)


def _write_target(tmp_path: Path, body: str) -> Path:
    target_dir = tmp_path / "experiments"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "modal_train_lane.py"
    target.write_text(body, encoding="utf-8")
    return target


# ----------------------------------------------------------------------------
# Live-repo regression guard
# ----------------------------------------------------------------------------


def test_check_339_live_repo_zero_violations() -> None:
    """The live repo's experiments/modal_train_lane.py MUST be clean."""
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        strict=False, verbose=False
    )
    assert violations == [], (
        f"Live repo regression: {len(violations)} Catalog #339 violation(s); "
        f"first 3: {violations[:3]}"
    )


# ----------------------------------------------------------------------------
# Positive cases (regression flagged)
# ----------------------------------------------------------------------------


def test_check_339_silent_swallow_pattern_flagged(tmp_path: Path) -> None:
    """The exact 3-anchor bug pattern (silent except Exception: print) flagged."""
    _write_target(
        tmp_path,
        """
import sys

def main():
    call_id = "fc-test"
    try:
        from tac.deploy.modal.call_id_ledger import register_dispatched_call_id
        register_dispatched_call_id(
            call_id=call_id,
            lane_id="lane_x",
            label="label_x",
        )
    except Exception as exc:
        print(f"WARNING: ledger failed: {exc}", file=sys.stderr)
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "Catalog #339" in violations[0]
    assert "silent-swallow" in violations[0]


def test_check_339_bare_except_pattern_flagged(tmp_path: Path) -> None:
    """A bare `except:` swallow of register_dispatched_call_id is flagged."""
    _write_target(
        tmp_path,
        """
def main():
    try:
        register_dispatched_call_id(call_id="x", lane_id="y", label="z")
    except:
        pass
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


def test_check_339_baseexception_swallow_flagged(tmp_path: Path) -> None:
    """A `except BaseException: print` swallow is flagged."""
    _write_target(
        tmp_path,
        """
def main():
    try:
        register_dispatched_call_id_fail_closed(
            call_id="x", lane_id="y", label="z"
        )
    except BaseException as exc:
        print(exc)
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


def test_check_339_no_registration_call_at_all_flagged(tmp_path: Path) -> None:
    """A modal_train_lane.py with NO registration call is flagged."""
    _write_target(
        tmp_path,
        """
def main():
    print("dispatched but never registered")
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "no register_dispatched_call_id" in violations[0]


# ----------------------------------------------------------------------------
# Negative cases (canonical pattern accepted)
# ----------------------------------------------------------------------------


def test_check_339_fail_closed_helper_no_try_accepted(tmp_path: Path) -> None:
    """A bare fail-closed call (no try/except) is accepted — it propagates."""
    _write_target(
        tmp_path,
        """
def main():
    register_dispatched_call_id_fail_closed(
        call_id="x", lane_id="y", label="z"
    )
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_339_try_with_sys_exit_handler_accepted(tmp_path: Path) -> None:
    """try/except with sys.exit in handler IS the canonical Layer 2 fix."""
    _write_target(
        tmp_path,
        """
import sys

def main():
    try:
        register_dispatched_call_id_fail_closed(
            call_id="x", lane_id="y", label="z"
        )
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(13)
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_339_try_with_raise_handler_accepted(tmp_path: Path) -> None:
    """try/except with `raise` re-raise is accepted."""
    _write_target(
        tmp_path,
        """
def main():
    try:
        register_dispatched_call_id_fail_closed(
            call_id="x", lane_id="y", label="z"
        )
    except Exception as exc:
        print(f"diagnostic: {exc}")
        raise
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_339_narrow_except_accepted(tmp_path: Path) -> None:
    """A narrow `except RuntimeError` is accepted (not a broad swallow)."""
    _write_target(
        tmp_path,
        """
def main():
    try:
        register_dispatched_call_id_fail_closed(
            call_id="x", lane_id="y", label="z"
        )
    except RuntimeError as exc:
        # narrow handler: not flagged as silent swallow
        print(exc)
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


# ----------------------------------------------------------------------------
# Waiver semantics
# ----------------------------------------------------------------------------


def test_check_339_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        """
# SILENT_NO_SPAWN_LEDGER_SWALLOW_OK: deliberate diagnostic-only wrapper for forensics
def main():
    try:
        register_dispatched_call_id(call_id="x", lane_id="y", label="z")
    except Exception:
        pass
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_339_placeholder_rationale_rejected(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        """
# SILENT_NO_SPAWN_LEDGER_SWALLOW_OK: <rationale>
def main():
    try:
        register_dispatched_call_id(call_id="x", lane_id="y", label="z")
    except Exception:
        pass
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


def test_check_339_reason_placeholder_rejected(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        """
# SILENT_NO_SPAWN_LEDGER_SWALLOW_OK: <reason>
def main():
    try:
        register_dispatched_call_id(call_id="x", lane_id="y", label="z")
    except Exception:
        pass
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


def test_check_339_short_rationale_rejected(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        """
# SILENT_NO_SPAWN_LEDGER_SWALLOW_OK: ok
def main():
    try:
        register_dispatched_call_id(call_id="x", lane_id="y", label="z")
    except Exception:
        pass
""",
    )
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1


# ----------------------------------------------------------------------------
# Strict-mode behavior
# ----------------------------------------------------------------------------


def test_check_339_strict_raises_with_catalog_339_message(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        """
def main():
    try:
        register_dispatched_call_id(call_id="x", lane_id="y", label="z")
    except Exception:
        pass
""",
    )
    with pytest.raises(PreflightError) as excinfo:
        check_modal_dispatcher_registers_call_id_before_successful_exit(
            repo_root=tmp_path, strict=True
        )
    assert "Catalog #339" in str(excinfo.value)
    assert "silent-no-spawn" in str(excinfo.value).lower()


def test_check_339_strict_silent_on_clean(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        """
def main():
    register_dispatched_call_id_fail_closed(
        call_id="x", lane_id="y", label="z"
    )
""",
    )
    # Should NOT raise.
    result = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=True
    )
    assert result == []


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


def test_check_339_missing_target_file_silent(tmp_path: Path) -> None:
    """If experiments/modal_train_lane.py does not exist, gate is silent."""
    # tmp_path has no experiments/ dir.
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=True
    )
    assert violations == []


def test_check_339_string_repo_root_accepted(tmp_path: Path) -> None:
    """repo_root accepts a string path (preflight convention)."""
    _write_target(
        tmp_path,
        """
def main():
    register_dispatched_call_id_fail_closed(call_id="x", lane_id="y", label="z")
""",
    )
    # str (not Path)
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=str(tmp_path), strict=False
    )
    assert violations == []


def test_check_339_syntax_error_flagged_but_does_not_crash(tmp_path: Path) -> None:
    _write_target(tmp_path, "def main(::: invalid syntax\n")
    violations = check_modal_dispatcher_registers_call_id_before_successful_exit(
        repo_root=tmp_path, strict=False
    )
    assert any("SyntaxError" in v for v in violations)


# ----------------------------------------------------------------------------
# Orchestrator wire-in regression guard (gate must be STRICT in preflight_all)
# ----------------------------------------------------------------------------


def test_check_339_orchestrator_wire_in_strict_true() -> None:
    """preflight_all() MUST call check_339 with strict=True."""
    source = inspect.getsource(preflight.preflight_all)
    # Must be present at all.
    assert (
        "check_modal_dispatcher_registers_call_id_before_successful_exit"
        in source
    ), "Catalog #339 gate not wired into preflight_all()"
    # Must be strict=True (after the call name, the closest `strict=` kwarg
    # should be True).
    match = re.search(
        r"check_modal_dispatcher_registers_call_id_before_successful_exit\s*\("
        r"[^)]*?strict\s*=\s*(True|False)",
        source,
        re.DOTALL,
    )
    assert match is not None, "Gate callsite missing strict= kwarg"
    assert match.group(1) == "True", (
        f"Gate must be STRICT-from-byte-one per Catalog #339 atomicity rule; "
        f"found strict={match.group(1)}"
    )


def test_check_339_catalog_185_sister_callable() -> None:
    """Gate function must be importable via tac.preflight globals (Catalog #185)."""
    fn = getattr(preflight, "check_modal_dispatcher_registers_call_id_before_successful_exit", None)
    assert fn is not None, "Gate function missing from tac.preflight namespace"
    assert callable(fn)
