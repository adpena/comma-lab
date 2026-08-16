"""Guard: no NEW hand-rolled process-liveness copy may re-enter the tree.

WHY (task #1080, 2026-08-16).  ``_pid_alive`` had been hand-rolled 11 times and
the copies DISAGREED -- a process that exists but cannot be signalled read
ALIVE at some sites and DEAD at others, and a zombie read ALIVE forever at
most of them.  Those copies now delegate to ``tac.process_liveness``.

Migrating without a guard only resets the clock: the twelfth copy lands next
week and the drift starts over.  Per the repo law that every fix ships with the
gate that refuses its re-introduction, this test pins BOTH shapes the migration
had to chase:

* a named ``def _pid_alive`` / ``_pid_is_alive`` / ``is_process_alive``, and
* a bare inline ``os.kill(<pid>, 0)`` existence probe -- which the ``def`` grep
  does NOT see, and which is how two extra sites were found mid-migration.

The allowlists below are the DELIBERATE remainder, each with its reason.  A new
entry is not forbidden, but it must be added here consciously, which is the
whole point.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SCANNED_DIRS = ("src/tac", "tools")

# Named liveness helpers that legitimately remain.
_ALLOWED_DEFS = {
    # The canonical surface itself.
    "src/tac/process_liveness.py": "the canonical implementation",
    # Thin bool wrappers that DELEGATE to the canonical surface.  They keep
    # their own name so call sites stay unchanged; the body is one call.
    "tools/run_quality_poller.py": "delegates: process_liveness.pid_file_state",
    "tools/dashboard_supervisor.py": "delegates: process_liveness.pid_state",
    "tools/dashboard_server.py": "delegates: process_liveness.pid_state",
    "tools/snapshot_run_checkpoint_at_stage_boundary.py": "delegates: pid_state",
    "tools/supervise_ddm_b4s_burn4.py": "delegates: process_liveness.pid_state",
    "tools/supervise_ddm_r1c_rung1.py": "delegates: process_liveness.pid_state",
    "tools/run_liveness_watcher.py": "delegates: process_liveness.pid_state",
    "tools/spawn_durable_daemon.py": "delegates: pid_state(reap_own_child=True)",
    "tools/witness_chain_watchdog.py": "delegates: pid_state after int() coercion",
    "tools/run_compact_renderer_mlx_spine_runner.py": "delegates: pid_state",
    # A DIFFERENT question: pgrep cmdline match by tag, not "is pid P alive".
    # Zero callers repo-wide as of 2026-08-16.
    "tools/experiment_runner.py": "tag-based pgrep, not a pid liveness read",
}

# Bare inline ``os.kill(x, 0)`` probes not yet migrated.  Each is a real
# instance of the same class; they are OUT of task #1080's enumerated scope and
# each needs its own behaviour adjudication before it moves.
_ALLOWED_INLINE = {
    "src/tac/process_liveness.py": "the canonical implementation",
    "tools/dashboard_up.py": "unmigrated: superseded by dashboard_supervisor",
    "tools/launch_detached_process.py": "unmigrated: needs adjudication",
    "tools/memory_blackbox.py": "unmigrated: needs adjudication",
    "tools/launch_witness_run.py": "unmigrated: two probes, need adjudication",
    "tools/dashboard_reload.py": "unmigrated: needs adjudication",
    "tools/relaunch_macos_cpu_canvas_sweep_safe.py": "unmigrated: needs adjudication",
    "tools/spawn_durable_daemon.py": "_pgid_alive: killpg GROUP probe, not a pid read",
}

_DEF_NAMES = frozenset(
    {"pid_alive", "_pid_alive", "pid_is_alive", "_pid_is_alive",
     "is_process_alive", "_is_process_alive"}
)


def _parse(path: Path) -> ast.Module | None:
    """AST, not regex.

    Scanning source TEXT for these patterns produces false positives on the
    very docstrings that EXPLAIN them -- this guard flagged its own prose
    ``os.kill(0, 0)`` on first run.  The AST sees calls and definitions only,
    so documentation can describe the bug class freely.
    """
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None


def _defines_liveness_helper(tree: ast.Module) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _DEF_NAMES:
            return f"def {node.name}"
    return None


def _has_inline_kill_zero(tree: ast.Module) -> str | None:
    """An ``os.kill(<anything>, 0)`` call -- signal 0 is the existence probe."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 2:
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "kill"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "os"):
            continue
        sig = node.args[1]
        if isinstance(sig, ast.Constant) and sig.value == 0:
            return f"os.kill(..., 0) at line {node.lineno}"
    return None


def _python_files() -> list[Path]:
    out: list[Path] = []
    for rel in _SCANNED_DIRS:
        root = _REPO / rel
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            # Test fixtures spawn real children and probe them; that is the
            # test's own business, not production liveness drift.
            if "/tests/" in path.as_posix():
                continue
            out.append(path)
    return out


def _offenders(detect, allowed: dict[str, str]) -> list[str]:
    found: list[str] = []
    for path in _python_files():
        rel = path.relative_to(_REPO).as_posix()
        if rel in allowed:
            continue
        tree = _parse(path)
        if tree is None:
            continue
        hit = detect(tree)
        if hit:
            found.append(f"{rel}: {hit}")
    return sorted(found)


def test_no_new_hand_rolled_pid_alive_definition() -> None:
    """A new named liveness helper must delegate, or be allowlisted with a reason."""
    offenders = _offenders(_defines_liveness_helper, _ALLOWED_DEFS)
    assert not offenders, (
        "New hand-rolled process-liveness helper(s) found. Call "
        "tac.process_liveness.pid_state / pid_file_state instead of re-deriving "
        "the PermissionError and zombie legs (they disagreed across 11 copies "
        "before task #1080). If the site genuinely asks a different question, "
        "add it to _ALLOWED_DEFS with the reason.\n  " + "\n  ".join(offenders)
    )


def test_no_new_inline_kill_zero_probe() -> None:
    """A bare ``os.kill(pid, 0)`` is the same bug class the def-grep misses."""
    offenders = _offenders(_has_inline_kill_zero, _ALLOWED_INLINE)
    assert not offenders, (
        "New inline `os.kill(pid, 0)` liveness probe(s) found. This is the "
        "shape that hides from a `def _pid_alive` grep; it reads a zombie as "
        "alive forever and (bare `except OSError`) a live-but-unsignallable "
        "process as dead. Use tac.process_liveness.pid_state.\n  "
        + "\n  ".join(offenders)
    )


def test_allowlisted_delegating_wrappers_actually_delegate() -> None:
    """The allowlist must not become a place to park a NON-delegating copy.

    Every file claiming "delegates" in ``_ALLOWED_DEFS`` has to really import
    the canonical module -- otherwise the allowlist would launder exactly the
    drift this guard exists to stop.
    """
    missing: list[str] = []
    for rel, reason in _ALLOWED_DEFS.items():
        if not reason.startswith("delegates"):
            continue
        text = (_REPO / rel).read_text(encoding="utf-8", errors="replace")
        if "process_liveness" not in text:
            missing.append(rel)
    assert not missing, (
        "Allowlisted as delegating but does not import tac.process_liveness: "
        + ", ".join(missing)
    )
