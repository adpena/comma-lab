"""Controls for the canonical process-liveness surface.

These pin the two semantics the 11 hand-rolled ``_pid_alive`` copies DISAGREED
on (measured 2026-08-16), so the drift cannot silently return:

* ``PermissionError`` -> ALIVE  (3 sites said alive, 2 said dead)
* a zombie            -> DEAD   (2 sites checked, 3 did not)

plus the third state no copy had: UNREADABLE, whose absence was the #1064
defect -- a blind watcher reading as a dead child.
"""

from __future__ import annotations

import os
from pathlib import Path

from tac import process_liveness as pl


def _reaped_pid() -> int:
    """A pid that is readable-but-gone: fork, exit, reap."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


# --- the divergence the copies encoded two different ways ---------------------


def test_permission_error_is_alive_not_dead() -> None:
    """pid 1 exists but cannot be signalled by a normal user: that is ALIVE.

    Two copies (`dashboard_supervisor`, `snapshot_run_checkpoint_at_stage_boundary`)
    swallowed this with a broad ``except`` and reported DEAD -- a live process
    read as exited.
    """
    try:
        os.kill(1, 0)
    except PermissionError:
        pass
    except ProcessLookupError:  # pragma: no cover - no init visible
        return
    assert pl.pid_state(1) == pl.ALIVE


def test_zombie_is_dead_not_alive() -> None:
    """``kill(pid, 0)`` succeeds on a zombie forever; liveness must say DEAD."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns
        os._exit(0)
    try:
        os.waitpid(pid, os.WNOHANG)  # let it become a zombie, do not reap
        os.kill(pid, 0)  # signal-probe still succeeds -- the trap
        assert pl.pid_state(pid, zombie_is_dead=True) == pl.DEAD
        assert pl.pid_state(pid, zombie_is_dead=False) == pl.ALIVE
    except ProcessLookupError:  # pragma: no cover - already reaped by the runner
        pass
    finally:
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass


# --- the third state, which is the whole point --------------------------------


def test_invalid_pids_are_unreadable_never_dead() -> None:
    for bad in (0, -1, -999):
        assert pl.pid_state(bad) == pl.UNREADABLE, bad


def test_live_pid_is_alive() -> None:
    assert pl.pid_state(os.getpid()) == pl.ALIVE


def test_reaped_child_is_dead() -> None:
    assert pl.pid_state(_reaped_pid()) == pl.DEAD


def test_pid_file_states(tmp_path: Path) -> None:
    missing = tmp_path / "absent.pid"
    garbage = tmp_path / "garbage.pid"
    garbage.write_text("not-a-pid", encoding="utf-8")
    empty = tmp_path / "empty.pid"
    empty.write_text("", encoding="utf-8")
    zero = tmp_path / "zero.pid"
    zero.write_text("0", encoding="utf-8")
    live = tmp_path / "live.pid"
    live.write_text(f"{os.getpid()}\n", encoding="utf-8")
    dead = tmp_path / "dead.pid"
    dead.write_text(str(_reaped_pid()), encoding="utf-8")

    assert pl.pid_file_state(missing) == pl.UNREADABLE
    assert pl.pid_file_state(garbage) == pl.UNREADABLE
    assert pl.pid_file_state(empty) == pl.UNREADABLE
    assert pl.pid_file_state(zero) == pl.UNREADABLE
    assert pl.pid_file_state(live) == pl.ALIVE
    assert pl.pid_file_state(dead) == pl.DEAD


def test_read_pid_file_returns_none_not_a_sentinel_int(tmp_path: Path) -> None:
    """None, never 0/-1: a sentinel int would be arithmetically usable as a pid."""
    bad = tmp_path / "bad.pid"
    bad.write_text("nope", encoding="utf-8")
    assert pl.read_pid_file(bad) is None
    good = tmp_path / "good.pid"
    good.write_text(str(os.getpid()), encoding="utf-8")
    assert pl.read_pid_file(good) == os.getpid()


def test_bool_true_is_not_a_pid() -> None:
    """``True`` is an int in Python; it must not read as pid 1."""
    assert pl.pid_state(True) == pl.UNREADABLE


def test_pid_alive_shim_matches_state() -> None:
    assert pl.pid_alive(os.getpid()) is True
    assert pl.pid_alive(_reaped_pid()) is False
    assert pl.pid_alive(0) is False
