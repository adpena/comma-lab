"""Canonical process-liveness read — ONE surface, tri-state, zombie-aware.

WHY THIS EXISTS (measured 2026-08-16): ``_pid_alive`` was hand-rolled **11
times** across ``tools/`` and ``src/tac/``, and the copies DISAGREE on the same
question:

===========================================  ===================  ==========
site                                         ``PermissionError``  zombie
===========================================  ===================  ==========
``run_liveness_watcher``                     alive                checked
``spawn_durable_daemon``                     alive                checked+reap
``witness_chain_watchdog``                   alive                no
``dashboard_supervisor``                     **dead**             no
``snapshot_run_checkpoint_at_stage_boundary``  **dead**           no
===========================================  ===================  ==========

A process that exists but cannot be signalled (owned by another user) reads
ALIVE at three sites and DEAD at two.  That is drift, not a design choice --
none of the divergent sites documents an intent to differ.

The semantics below are the UNION of the strongest existing copies, not a new
invention:

* ``ESRCH`` -> ``DEAD``; ``EPERM`` -> ``ALIVE`` (POSIX: the pid exists, we just
  cannot signal it).  Three of five copies already agree.
* A **zombie is DEAD**.  ``kill(pid, 0)`` succeeds on a reaped-but-unwaited
  child forever, so signal-liveness alone reports an exited process alive
  indefinitely (``run_liveness_watcher._pid_is_zombie``'s rationale, preserved).
* ``pid <= 0`` is ``UNREADABLE`` -- 0 means "my process group" and negatives are
  group targets; neither is a liveness question.

The THIRD state (``UNREADABLE``) is what no copy had, and its absence was a
measured defect (#1064): a watcher whose pid file is missing or garbage cannot
observe anything, yet a bool collapsed that into "the child died" -- so the
watcher exited quietly and watched nothing for an entire run.  Blindness and
death demand opposite responses, so they must be distinguishable at the source.

Sisters: ``[[the-control-plane-fails-silently-make-the-silence-loud-at-launch-20260816]]``
(the genus) and the repo law that a bug class fixed at one surface stays live at
the other six.
"""

from __future__ import annotations

import errno
import os
import subprocess
from pathlib import Path

__all__ = [
    "ALIVE",
    "DEAD",
    "UNREADABLE",
    "pid_alive",
    "pid_file_state",
    "pid_state",
    "read_pid_file",
]

ALIVE = "alive"
DEAD = "dead"
UNREADABLE = "unreadable"

_ZOMBIE_PROBE_TIMEOUT_S = 10.0


def _is_zombie(pid: int) -> bool:
    """True iff ``ps`` reports state ``Z``.  Fail-open: unknown -> not zombie.

    Failing open matters: if ``ps`` is missing or times out we must not
    manufacture a DEAD verdict from an absent measurement.
    """
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=_ZOMBIE_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip().startswith("Z")


def _try_reap(pid: int) -> bool:
    """Non-blocking reap IF ``pid`` is our own child.  True iff it was reaped.

    Opt-in only (see ``pid_state``): reaping is a real side effect that would
    steal a ``waitpid`` result from a caller that owns the child.
    """
    try:
        reaped, _status = os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        return False
    return reaped == pid


def pid_state(
    pid: int, *, zombie_is_dead: bool = True, reap_own_child: bool = False
) -> str:
    """Tri-state liveness of ``pid``: ``ALIVE`` / ``DEAD`` / ``UNREADABLE``.

    ``UNREADABLE`` means the question could not be asked (invalid pid), never
    that the answer is "no".
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return UNREADABLE
    if reap_own_child and _try_reap(pid):
        return DEAD
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return DEAD
    except PermissionError:
        return ALIVE  # exists; owned by another user
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return DEAD
        if exc.errno == errno.EPERM:
            return ALIVE
        return UNREADABLE
    if zombie_is_dead and _is_zombie(pid):
        return DEAD
    return ALIVE


def read_pid_file(path: Path | str) -> int | None:
    """Parse a pid file.  ``None`` iff unreadable/garbage/non-positive."""
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def pid_file_state(
    path: Path | str, *, zombie_is_dead: bool = True, reap_own_child: bool = False
) -> str:
    """Tri-state liveness from a pid FILE.

    A missing / empty / garbage / non-positive file is ``UNREADABLE`` -- the
    watcher is blind, which is not the same fact as "the child exited".
    """
    pid = read_pid_file(path)
    if pid is None:
        return UNREADABLE
    return pid_state(pid, zombie_is_dead=zombie_is_dead, reap_own_child=reap_own_child)


def pid_alive(pid: int) -> bool:
    """Bool compat shim.  Prefer ``pid_state`` -- this cannot express blindness."""
    return pid_state(pid) == ALIVE
