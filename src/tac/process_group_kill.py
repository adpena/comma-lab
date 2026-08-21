# SPDX-License-Identifier: MIT
"""Canonical process-GROUP kill — so a timeout reaches the whole tree, not just the shell.

WHY THIS EXISTS (measured). ``subprocess.run(cmd, timeout=N)`` kills the DIRECT CHILD
only. When that child is a shell, a wrapper script, or another python that spawns, the
real worker is a GRANDCHILD, and it survives the timeout. Two independent measured
consequences, both from this repo:

* **The work keeps running after the harness gave up.** ``ddm_cpu1``, 2026-08-20:
  ``src/tac/submission_chain.py::run_inflate`` ran ``["bash", "inflate.sh", ...]`` under
  ``timeout=1800``. ``TimeoutExpired`` fired at 1799.99997045 s and killed ``bash``. The
  decoder underneath ran on to completion and wrote its own report — the full
  3,662,409,600 B field, ``pair_count 600``, at **4,369.600210089 s**. A run the harness
  recorded as FAILED had a complete report on disk, 2,570 s past the wall the harness
  believed it had enforced. In a paid container that is 2,570 s of metered orphan.

* **The absence is then misread as "it never started."** ``ddm_lc2``, 2026-08-10:
  ``tools/launch_detached_process.py`` returned the WRAPPER pid 21915; the dispatcher was
  grandchild 21916, re-parented to PID 1 by ``start_new_session``. ``kill 21915`` did not
  stop it, the dispatcher went on to build the CUDA image and spawn both axes, and the
  reasoning "I killed it" → "it never spawned" was written into a custody record as
  proof. ``ddm_cd1`` §7 records the same inference a second time: an rc=144 reaper killed
  the shell around ``fire_modal_auth_eval.py``, the python child kept walking toward a
  PAID dispatch, and the empty ledger/claims/output dir seconds later read as "it did
  nothing". Two firers then briefly raced the same lane.

The cure is POSIX and old: put the child in its OWN session/group at spawn
(``start_new_session=True``) and signal the GROUP (``os.killpg``) on timeout. Then the
grandchild inherits the group and the kill reaches it.

WHY A NEW MODULE AND NOT ANOTHER PRIVATE COPY. Seven private twins already exist
(``tools/memory_guard.py::_kill_pgrp``, ``tools/safe_run.py::_kill_group``,
``src/tac/witness_control/verdict_reclaim.py::_kill_group``,
``src/comma_lab/scheduler/experiment_queue.py::_signal_process_group``,
``tools/dashboard_supervisor.py::_killpg``, ``scripts/launch_lane_with_retry.py``,
``tools/dashboard_ctl.py``), none importable, and they disagree on grace, on escalation,
and on whether to verify the group actually died. This module is the sibling of
``tac.process_liveness``, which consolidated ``_pid_alive`` after the same drift was
measured across 11 copies — same problem, same cure, same anti-twin gate.

WHAT THIS MODULE DOES NOT FIX. Killing the tree is half the defect; the other half is the
INFERENCE from the resulting absence. ``ddm_cd1`` names it: "absence of a downstream
artifact seconds after an rc=144 is indistinguishable from a process that never
started." ``group_survivors_after_kill`` on the raised timeout is the observable this
module contributes toward that half — a caller can state whether the tree is provably
down rather than assuming it.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_TERM_GRACE_S",
    "GroupKillReceipt",
    "ProcessGroupTimeout",
    "group_alive",
    "kill_process_group",
    "run_in_process_group",
]

# SIGTERM first so a worker can flush; SIGKILL after. 5 s matches the longest-standing
# private twin (memory_guard's DEFAULT_SIGKILL_GRACE_SEC) rather than inventing a value.
DEFAULT_TERM_GRACE_S = 5.0
_POLL_S = 0.1


@dataclass(frozen=True)
class GroupKillReceipt:
    """What the kill actually did — never inferred, always observed."""

    pgid: int
    term_delivered: bool
    kill_delivered: bool
    survivors_after: bool
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pgid": self.pgid,
            "term_delivered": self.term_delivered,
            "kill_delivered": self.kill_delivered,
            "survivors_after": self.survivors_after,
            "elapsed_s": round(self.elapsed_s, 4),
        }


class ProcessGroupTimeout(subprocess.TimeoutExpired):
    """`TimeoutExpired` that also reports whether the TREE went down.

    Subclasses `TimeoutExpired` on purpose: every existing `except
    subprocess.TimeoutExpired` keeps working, and callers that care can read
    `group_survivors_after_kill` instead of assuming the absence they observe means
    the work stopped.
    """

    def __init__(
        self,
        cmd: Any,
        timeout: float,
        *,
        output: Any = None,
        stderr: Any = None,
        receipt: GroupKillReceipt | None = None,
    ) -> None:
        super().__init__(cmd, timeout, output=output, stderr=stderr)
        self.kill_receipt = receipt
        self.group_survivors_after_kill = bool(receipt.survivors_after) if receipt else True


def group_alive(pgid: int) -> bool:
    """True iff at least one process remains in group ``pgid``.

    ``killpg(pgid, 0)`` is the POSIX existence probe. ``PermissionError`` means the group
    EXISTS but is not ours to signal — reported ALIVE, the fail-safe direction (we could
    not have killed it either). A non-positive pgid is never probed: ``killpg`` with a
    non-positive argument has broadcast semantics.

    ZOMBIE CAVEAT, measured on macOS 2026-08-21 by this module's own grandchild test: a
    group whose ONLY remaining member is an unreaped zombie answers ``EPERM`` here, so it
    reads ALIVE even though every process in it is dead. That is why
    :func:`kill_process_group` takes a ``reap`` callback — the caller's own exited child
    must be reaped before "did the tree go down?" can be answered honestly. Without it
    the receipt reported ``survivors_after=True`` for a tree that was fully dead, which
    is the same false-inference family this module exists to end.
    """

    if pgid <= 0:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_process_group(
    pgid: int,
    *,
    term_grace_s: float = DEFAULT_TERM_GRACE_S,
    poll_s: float = _POLL_S,
    reap: Callable[[], None] | None = None,
) -> GroupKillReceipt:
    """SIGTERM the group, wait out ``term_grace_s``, then SIGKILL. Verify, never assume.

    Refuses to signal a non-positive pgid (broadcast) or THIS process's own group
    (suicide-plus-parent). Returns a receipt whose ``survivors_after`` was OBSERVED after
    the escalation, so a caller can say "the tree is down" only when it is.

    ``reap`` is called before every liveness read. A caller that is the PARENT of a
    process in this group must pass one (typically ``proc.poll``): an exited-but-unreaped
    child is a zombie, and a zombie-only group still reads alive (see
    :func:`group_alive`). Reaping is the difference between measuring the tree and
    measuring our own bookkeeping.
    """

    started = time.monotonic()
    if pgid <= 0:
        raise ValueError(f"refusing to signal non-positive pgid {pgid} (broadcast semantics)")
    if pgid == os.getpgrp():
        raise ValueError(
            f"refusing to signal our OWN process group {pgid} — the child was not "
            "spawned with start_new_session=True, so a group kill would take us with it"
        )

    def _alive() -> bool:
        if reap is not None:
            reap()
        return group_alive(pgid)

    term = _signal_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(term_grace_s, 0.0)
    while time.monotonic() < deadline and _alive():
        time.sleep(poll_s)

    kill = False
    if _alive():
        kill = _signal_group(pgid, signal.SIGKILL)
        # SIGKILL is not instantaneous; give the kernel a bounded moment to reap.
        hard_deadline = time.monotonic() + 2.0
        while time.monotonic() < hard_deadline and _alive():
            time.sleep(poll_s)

    return GroupKillReceipt(
        pgid=pgid,
        term_delivered=term,
        kill_delivered=kill,
        survivors_after=_alive(),
        elapsed_s=time.monotonic() - started,
    )


def _signal_group(pgid: int, sig: int) -> bool:
    """Deliver ``sig`` to the group; False iff the group was already gone/not ours."""

    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def run_in_process_group(
    cmd: list[str],
    *,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    text: bool = False,
    check: bool = False,
    stdout: Any = None,
    stderr: Any = None,
    term_grace_s: float = DEFAULT_TERM_GRACE_S,
) -> subprocess.CompletedProcess:
    """`subprocess.run` whose timeout reaches the whole tree.

    Drop-in for the `subprocess.run(cmd, cwd=, env=, capture_output=, text=, timeout=,
    check=, stdout=, stderr=)` shape. The one behavioural difference is the one that
    matters: the child leads its own session, so on timeout the SIGTERM/SIGKILL
    escalation lands on the GROUP and takes every grandchild with it, instead of
    orphaning the real worker to PID 1 (the ddm_cpu1 4,369 s decoder).

    ``stdout``/``stderr`` accept the same values as `subprocess.run` — most usefully an
    open file object, so a long run streams straight to a log fd instead of buffering in
    the parent. That shape is NOT a nicety: `modal_train_lane` runs a trainer for up to
    14 h under `stdout=logf`, and forcing it through `capture_output` would hold the
    whole run in parent RSS and destroy log liveness under a crash. Passing either
    together with ``capture_output=True`` raises `ValueError`, mirroring
    `subprocess.run`.

    Raises `ProcessGroupTimeout` (a `TimeoutExpired`) carrying the kill receipt.
    """

    if capture_output:
        if stdout is not None or stderr is not None:
            raise ValueError(
                "capture_output=True is mutually exclusive with stdout/stderr "
                "(same contract as subprocess.run)"
            )
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    # argv list, never shell=True — the group discipline below assumes an exec'd argv.
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=text,
        start_new_session=True,
    )
    # start_new_session makes the child a session AND group leader, so pgid == pid.
    # Read it from the OS anyway rather than assuming; a race that loses the child
    # here must not turn into a signal aimed at the wrong group.
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:  # already exited
        pgid = proc.pid

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # `proc.poll()` is a non-blocking reap: once the shell exits it clears the
        # zombie that would otherwise keep the group reading alive.
        receipt = kill_process_group(pgid, term_grace_s=term_grace_s, reap=lambda: proc.poll())
        # Reap the direct child so it does not linger as a zombie, and collect
        # whatever it had already written.
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - group is dead by here
            proc.kill()
            out, err = proc.communicate()
        raise ProcessGroupTimeout(
            cmd, float(timeout or 0.0), output=out, stderr=err, receipt=receipt
        ) from None
    except BaseException:
        # `start_new_session` also detaches the child from OUR terminal's signal
        # delivery, so a Ctrl-C that used to reach it no longer does. Without this the
        # cure would have INTRODUCED the orphan it exists to prevent, on the
        # interactive path. Take the tree down, then let the exception continue.
        kill_process_group(pgid, term_grace_s=term_grace_s, reap=lambda: proc.poll())
        raise

    completed = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=out, stderr=err)
    return completed
