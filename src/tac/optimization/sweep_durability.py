"""Two recurring failure classes, made structural instead of remembered.

Both of these are written down in CLAUDE.md / MEMORY and both recurred anyway inside
`ddm_rs2` on 2026-08-03.  Written down is not enforced; a callable is.

CLASS 1 — LOOP-END-ONLY SAVING (CLAUDE.md forbids it by name)
    A sweep that writes its result once, after the loop, loses everything to any kill.
    `ddm_rs2`'s v1 drive sweep ran 24 of 36 n600 groups over 1,901 s and left no artifact.
    `resumable_units()` turns "re-run" into "resume": a unit whose receipt is already on
    disk is skipped, so a kill costs at most one unit and the resume path is the ONLY path
    (there is no separate --resume flag to forget to pass).

CLASS 2 — LIVENESS PROBED FROM THE PROCESS TABLE
    In one session the process probe was wrong THREE times in BOTH directions:
    `pgrep -f <script>.py` matched the watcher shells' own command lines and reported ALIVE
    for minutes after a real death; then a lagging log tail plus `ps` made the same operator
    declare two deaths for a job that had already finished n600.  A pattern probe matches
    anything that MENTIONS the pattern -- most insidiously your own watchers -- and from
    inside you cannot tell which case you are in.
    `job_state()` reads the RECEIPT instead: state is a function of what the job has
    WRITTEN, which cannot self-match and cannot lag behind the job's own progress.

Neither function knows anything about this codec; both are generic sweep hygiene.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

#: A job whose newest receipt is older than this is STALLED, not RUNNING.
DEFAULT_STALL_AFTER_S = 300.0

RUNNING = "RUNNING"
DONE = "DONE"
STALLED = "STALLED"
NOT_STARTED = "NOT_STARTED"


def unit_receipt_path(out_dir: Path | str, unit: Any, suffix: str = ".npz") -> Path:
    """Deterministic receipt path for one sweep unit."""
    key = "_".join(str(p) for p in (unit if isinstance(unit, tuple | list) else (unit,)))
    if not key or "/" in key or ".." in key:
        raise ValueError(f"unit key {key!r} is empty or unsafe as a filename")
    return Path(out_dir) / f"u_{key}{suffix}"


def resumable_units(
    out_dir: Path | str,
    units: Iterable[Any],
    *,
    suffix: str = ".npz",
) -> tuple[list[Any], list[Any]]:
    """Split `units` into (todo, already_done) by which receipts exist on disk.

    The resume path is the only path: callers iterate `todo` and write one receipt per
    unit as it completes, so a kill costs at most one unit and re-running IS the resume.
    """
    todo, done = [], []
    for u in units:
        (done if unit_receipt_path(out_dir, u, suffix).exists() else todo).append(u)
    return todo, done


def job_state(
    receipt_dir: Path | str,
    *,
    expected_units: int | None = None,
    stall_after_s: float = DEFAULT_STALL_AFTER_S,
    suffix: str = ".npz",
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Read a sweep's state from its RECEIPTS. Never from the process table.

    Returns ``state`` in {NOT_STARTED, RUNNING, STALLED, DONE} plus the counts that
    justify it, so a caller can report the DENOMINATOR rather than a bare verdict --
    an empty receipt directory is NOT_STARTED, never a vacuous pass.
    """
    d = Path(receipt_dir)
    receipts = sorted(d.glob(f"u_*{suffix}")) if d.is_dir() else []
    n = len(receipts)
    newest = max((p.stat().st_mtime for p in receipts), default=None)
    age = None if newest is None else max(0.0, now() - newest)
    if n == 0:
        state = NOT_STARTED
    elif expected_units is not None and n >= expected_units:
        state = DONE
    elif age is not None and age > stall_after_s:
        state = STALLED
    else:
        state = RUNNING
    return {
        "state": state,
        "units_done": n,
        "units_expected": expected_units,
        "newest_receipt_age_s": age,
        "stall_after_s": stall_after_s,
        "receipt_dir": str(d),
        "evidence": "receipt mtimes only; the process table was NOT consulted",
    }


def missing_units(
    receipt_dir: Path | str,
    units: Sequence[Any],
    *,
    suffix: str = ".npz",
) -> list[Any]:
    """The units a finished-looking sweep still owes. Empty scope is reported, not passed."""
    todo, _ = resumable_units(receipt_dir, units, suffix=suffix)
    return todo


__all__ = [
    "DEFAULT_STALL_AFTER_S",
    "DONE",
    "NOT_STARTED",
    "RUNNING",
    "STALLED",
    "job_state",
    "missing_units",
    "resumable_units",
    "unit_receipt_path",
]
