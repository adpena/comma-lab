# SPDX-License-Identifier: MIT
"""Catalog #360 — PRE-SPAWN-FATAL-OBSERVABILITY-EXTINCTION self-protection.

Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" + "Bugs must be permanently
fixed AND self-protected against" non-negotiables. Anchor: OVERNIGHT-J STC v2
5th consecutive silent-no-spawn 2026-05-21 (per
`.omx/research/stc_v2_ratify_or_defer_path_b_dispatch_landed_20260521.md`).

Sister of Catalog #339 (post-spawn registration fail-closed). Where #339
catches silent-swallow registration failures AFTER fn.spawn() returns,
#360 catches the upstream silent-no-spawn pattern: sys.exit() FATAL paths
inside `@app.local_entrypoint()` main() that fire BEFORE fn.spawn() leave
Modal in "stopped/0 tasks" with NO canonical-ledger row and NO recovery dump.

The fix routes every sys.exit upstream of fn.spawn through the canonical
helper `tac.deploy.modal.call_id_ledger.register_pre_spawn_fatal`, which
writes a `pre_spawn_fatal` event to the canonical ledger BEFORE the
caller's sys.exit fires.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tac import preflight
from tac.preflight import (
    PreflightError,
    check_modal_dispatcher_pre_spawn_fatal_observability,
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


def test_check_360_live_repo_zero_violations() -> None:
    """The live repo's experiments/modal_train_lane.py MUST be clean.

    Every sys.exit upstream of fn.spawn must route through
    _pre_spawn_fatal / register_pre_spawn_fatal.
    """
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        strict=False, verbose=False
    )
    assert violations == [], (
        f"Live repo regression: {len(violations)} Catalog #360 violation(s); "
        f"first 3: {violations[:3]}"
    )


# ----------------------------------------------------------------------------
# Positive cases (regression flagged)
# ----------------------------------------------------------------------------


def test_check_360_bare_sys_exit_upstream_of_spawn_flagged(tmp_path: Path) -> None:
    """A bare sys.exit(2) upstream of fn.spawn must be flagged."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str):
    if not lane_script:
        print("FATAL: missing lane_script")
        sys.exit(2)
    fn_call = fn.spawn(lane_script)
"""
    _write_target(tmp_path, body)
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "sys.exit upstream of fn.spawn" in violations[0]
    assert "PRE_SPAWN_FATAL_OBSERVABILITY_OK" in violations[0]


def test_check_360_multiple_unwrapped_sys_exits_flagged(tmp_path: Path) -> None:
    """Multiple unwrapped sys.exits in the same main produce multiple violations."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str, gpu: str):
    if not lane_script:
        print("FATAL: missing lane_script")
        sys.exit(2)
    if gpu not in ("CPU", "T4"):
        print("FATAL: bad gpu")
        sys.exit(2)
    if not (gpu == "T4"):
        print("FATAL: wrong gpu")
        sys.exit(2)
    fn_call = fn.spawn(lane_script)
"""
    _write_target(tmp_path, body)
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 3


# ----------------------------------------------------------------------------
# Negative cases (accepted)
# ----------------------------------------------------------------------------


def test_check_360_register_pre_spawn_fatal_call_accepted(tmp_path: Path) -> None:
    """sys.exit preceded by register_pre_spawn_fatal accepted."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str):
    if not lane_script:
        from tac.deploy.modal.call_id_ledger import register_pre_spawn_fatal
        print("FATAL: missing lane_script")
        register_pre_spawn_fatal(label="x", lane_id="y", fatal_reason="missing")
        sys.exit(2)
    fn_call = fn.spawn(lane_script)
"""
    _write_target(tmp_path, body)
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_360_inline_helper_accepted(tmp_path: Path) -> None:
    """sys.exit preceded by _pre_spawn_fatal inline helper accepted."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str):
    def _pre_spawn_fatal(reason, line_no=None, helper_source=None):
        pass
    if not lane_script:
        reason = "FATAL: missing"
        print(reason)
        _pre_spawn_fatal(reason, line_no=42, helper_source="lane_script_check")
        sys.exit(2)
    fn_call = fn.spawn(lane_script)
"""
    _write_target(tmp_path, body)
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_360_sys_exit_after_spawn_not_flagged(tmp_path: Path) -> None:
    """sys.exit AFTER fn.spawn is NOT a silent-no-spawn — covered by Catalog #339."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str):
    fn_call = fn.spawn(lane_script)
    # sys.exit AFTER spawn is OK per Catalog #360 scope; Catalog #339 covers it
    if fn_call is None:
        sys.exit(13)
"""
    _write_target(tmp_path, body)
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_360_no_local_entrypoint_no_op(tmp_path: Path) -> None:
    """File without @app.local_entrypoint() main is N/A — no violations."""
    body = """
import sys

def main():
    sys.exit(2)
"""
    _write_target(tmp_path, body)
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_360_missing_file_silent(tmp_path: Path) -> None:
    """Missing target file produces no violations."""
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


# ----------------------------------------------------------------------------
# Waiver semantics
# ----------------------------------------------------------------------------


def test_check_360_same_line_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    """Same-line PRE_SPAWN_FATAL_OBSERVABILITY_OK waiver with rationale accepted."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str):
    if not lane_script:
        print("FATAL: missing")
        sys.exit(2)  # PRE_SPAWN_FATAL_OBSERVABILITY_OK:fail_fast_pre_observability_init
    fn_call = fn.spawn(lane_script)
"""
    _write_target(tmp_path, body)
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_360_placeholder_rationale_rejected(tmp_path: Path) -> None:
    """Placeholder rationale `<rationale>` is rejected."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str):
    if not lane_script:
        print("FATAL: missing")
        sys.exit(2)  # PRE_SPAWN_FATAL_OBSERVABILITY_OK:<rationale>
    fn_call = fn.spawn(lane_script)
"""
    _write_target(tmp_path, body)
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check_360_short_rationale_rejected(tmp_path: Path) -> None:
    """Short rationale (<4 chars) is rejected."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str):
    if not lane_script:
        print("FATAL: missing")
        sys.exit(2)  # PRE_SPAWN_FATAL_OBSERVABILITY_OK:ok
    fn_call = fn.spawn(lane_script)
"""
    _write_target(tmp_path, body)
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


# ----------------------------------------------------------------------------
# Strict-mode behavior
# ----------------------------------------------------------------------------


def test_check_360_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """strict=True must raise PreflightError on violations."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str):
    if not lane_script:
        print("FATAL: missing")
        sys.exit(2)
    fn_call = fn.spawn(lane_script)
"""
    _write_target(tmp_path, body)
    with pytest.raises(PreflightError) as exc_info:
        check_modal_dispatcher_pre_spawn_fatal_observability(
            repo_root=tmp_path, strict=True, verbose=False
        )
    assert "Catalog #360" in str(exc_info.value)
    assert "PRE-SPAWN-FATAL-OBSERVABILITY-EXTINCTION" in str(exc_info.value)


def test_check_360_strict_silent_on_clean(tmp_path: Path) -> None:
    """strict=True must NOT raise when no violations."""
    body = """
import sys
import modal

app = modal.App("test")
fn = None

@app.local_entrypoint()
def main(lane_script: str):
    fn_call = fn.spawn(lane_script)
"""
    _write_target(tmp_path, body)
    # Should not raise
    violations = check_modal_dispatcher_pre_spawn_fatal_observability(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert violations == []


# ----------------------------------------------------------------------------
# Orchestrator wire-in regression guard (Catalog #176 sister discipline)
# ----------------------------------------------------------------------------


def test_check_360_wired_strict_true_in_orchestrator() -> None:
    """preflight_all() MUST call check_modal_dispatcher_pre_spawn_fatal_observability(strict=True).

    Per Catalog #176 sister discipline: every STRICT preflight gate must
    appear in the orchestrator callsite list. Per Catalog #185 META-meta:
    Live count: 0 claims are verified empirically.
    """
    source = inspect.getsource(preflight.preflight_all)
    assert "check_modal_dispatcher_pre_spawn_fatal_observability" in source
    # Must be wired strict=True (not warn-only)
    pre_spawn_idx = source.index("check_modal_dispatcher_pre_spawn_fatal_observability")
    # Look at next ~120 chars for `strict=True`
    nearby = source[pre_spawn_idx : pre_spawn_idx + 200]
    assert "strict=True" in nearby, (
        "Catalog #360 must be wired strict=True per CLAUDE.md "
        "'Strict-flip atomicity rule' (live count: 0 verified)."
    )


def test_check_360_callable_via_globals() -> None:
    """Catalog #185 sister regression: function callable via module globals."""
    func = getattr(preflight, "check_modal_dispatcher_pre_spawn_fatal_observability", None)
    assert func is not None
    assert callable(func)


# ----------------------------------------------------------------------------
# Canonical helper register_pre_spawn_fatal regression (call_id_ledger)
# ----------------------------------------------------------------------------


def test_register_pre_spawn_fatal_helper_exists_and_callable() -> None:
    """The canonical helper register_pre_spawn_fatal must be exported."""
    from tac.deploy.modal.call_id_ledger import (
        register_pre_spawn_fatal,
        EVENT_PRE_SPAWN_FATAL,
        STATUS_PRE_SPAWN_FATAL,
        VALID_EVENT_TYPES,
        VALID_STATUSES,
        TERMINAL_STATUSES,
    )
    assert callable(register_pre_spawn_fatal)
    assert EVENT_PRE_SPAWN_FATAL == "pre_spawn_fatal"
    assert STATUS_PRE_SPAWN_FATAL == "pre_spawn_fatal"
    assert EVENT_PRE_SPAWN_FATAL in VALID_EVENT_TYPES
    assert STATUS_PRE_SPAWN_FATAL in VALID_STATUSES
    assert STATUS_PRE_SPAWN_FATAL in TERMINAL_STATUSES


def test_register_pre_spawn_fatal_signature() -> None:
    """register_pre_spawn_fatal must accept the canonical kwarg contract."""
    from tac.deploy.modal.call_id_ledger import register_pre_spawn_fatal
    sig = inspect.signature(register_pre_spawn_fatal)
    params = sig.parameters
    # Required keyword-only
    for required in ("label", "lane_id", "fatal_reason"):
        assert required in params, f"register_pre_spawn_fatal missing required kwarg {required!r}"
        assert params[required].kind == inspect.Parameter.KEYWORD_ONLY
    # Optional with defaults
    for optional in ("exit_code", "sys_exit_line_number", "sys_exit_helper_source", "write_last_resort_dump"):
        assert optional in params, f"register_pre_spawn_fatal missing optional kwarg {optional!r}"
