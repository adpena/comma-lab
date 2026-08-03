#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Single-writer lock for resumable append-only job directories.

WHY THIS EXISTS (ddm_sm1, 2026-08-03 — measured, not hypothetical). A resumable job
whose state is an append-only JSONL is safe against a crash and unsafe against a
*second copy of itself*. Two writers each hold their own in-memory replay of the store,
each appends rows the other never saw, and the result is a file with duplicate keys and
a broken state chain -- silently, because every individual row is well-formed.

The trigger was an instrument defect, not carelessness: the agent harness reported a
detached background job as ``failed exit code 144`` (SIGURG). The signal killed the
harness's WRAPPER; the Python child kept running. Relaunching on that notification --
the obviously correct response to "your job died" -- produced a second writer. Damage:
26 rows / 23 unique, three duplicated instances, one broken pair-state chain, in a store
that had already been read and reasoned over.

THE LAW: a nonzero exit from a launcher is not evidence that the job died. Liveness is
a property of the process table, never of an exit code. Because a human (or an agent)
cannot be relied on to check, the store defends itself.

This is the ``.omx/state/.commit-lock`` discipline (CLAUDE.md "Subagent commits MUST use
serializer") applied to job output directories rather than to git: an advisory
``fcntl.LOCK_EX | LOCK_NB`` whose holder identifies itself, so the second writer refuses
loudly instead of corrupting quietly.

Fail-closed by construction: acquisition is non-blocking and raises. A job that cannot
prove it is the only writer does not run.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["SingleWriterLockError", "read_lock_holder", "single_writer_lock"]

LOCK_BASENAME = ".writer.lock"


class SingleWriterLockError(RuntimeError):
    """Raised when another live process already owns this output directory."""


def read_lock_holder(out_dir: Path | str) -> dict | None:
    """Best-effort read of the holder record; ``None`` if absent or unreadable.

    Diagnostic only -- never a substitute for the lock itself. A stale record can
    outlive its process (kill -9 leaves the file); the AUTHORITY is whether
    ``flock`` succeeds, not what this file says.
    """
    path = Path(out_dir) / LOCK_BASENAME
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@contextmanager
def single_writer_lock(
    out_dir: Path | str,
    *,
    label: str,
    skip: bool = False,
) -> Iterator[Path | None]:
    """Own ``out_dir`` exclusively for the duration of the block.

    Args:
        out_dir: the job's output directory; created if absent.
        label: human-readable job identity recorded for the next contender to read.
        skip: bypass entirely (single-process test harnesses only). Named ``skip``
            rather than defaulting to on, because a lock you can forget to take is
            the thing this module exists to prevent.

    Raises:
        SingleWriterLockError: another process holds the lock. The message names the
            recorded holder so the operator can decide between waiting and killing,
            instead of guessing.
    """
    if skip:
        yield None
        return

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / LOCK_BASENAME
    fh = path.open("a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            holder = read_lock_holder(d)
            who = (
                f"pid {holder.get('pid')} on {holder.get('host')} "
                f"({holder.get('label')}, started {holder.get('started_utc')})"
                if holder
                else "an unidentified process"
            )
            raise SingleWriterLockError(
                f"[refuse] {d} is already owned by {who}. A SECOND writer on a "
                f"resumable append-only store produces duplicate keys and a broken "
                f"state chain, silently. If you believe the holder is dead, CHECK THE "
                f"PROCESS TABLE (`ps -axo pid,command | grep {holder.get('pid') if holder else '<pid>'}`) "
                f"-- a launcher's nonzero exit is not evidence the job died."
            ) from exc

        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps({
            "label": label,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }))
        fh.flush()
        os.fsync(fh.fileno())
        yield path
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
