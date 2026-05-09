"""Shared `.omx/state/lightning_active_jobs.json` writer with fcntl serialization.

Per CLAUDE.md non-negotiable + catalog #131 (proactive custody+concurrency
audit, 2026-05-09): every shared-state write must serialize on a sibling
fcntl lockfile and use a unique tmp suffix to prevent concurrent dispatchers
from silently dropping each other's updates.

Before this module landed, four separate experiment scripts maintained their
own bare load→mutate→write cycles against
``.omx/state/lightning_active_jobs.json``:

- ``experiments/arch_shrink_x0.4_lightning_full.py`` (dispatcher)
- ``experiments/arch_shrink_x0.4_lightning_harvest.py`` (harvester)
- ``experiments/lossy_coarsening_lightning_cuda_test.py`` (dispatcher)
- ``experiments/lossy_coarsening_lightning_harvest.py`` (harvester)

Each one had:

  data = json.loads(PATH.read_text())  # racey read
  data.append(record)                   # in-process mutation
  PATH.write_text(json.dumps(data))     # racey write — last-writer-wins

Two parallel runs (e.g., sister harvester + sister dispatcher fired by
``cron`` at the same minute) silently dropped each other's updates because
the load happened before the lock and the write replaced the file unconditionally.

This module exposes the canonical transactional API: ``register_job`` /
``upsert_job`` / ``mark_job_terminal`` all do load→mutate→save inside an
exclusive ``fcntl.flock(LOCK_EX)`` so concurrent updates of distinct rows
all survive.

MEDIUM 2 (codex round-3, 2026-05-09): the previous ``load_active_jobs``
returned ``[]`` on corrupt JSON or non-list state. ``update_active_jobs_locked``
then wrote the empty snapshot back, silently overwriting the malformed file
with a fresh one — DROPPING all active and terminal rows. Harvesters that
re-read after the dispatcher write would see ``[]`` and skip every still-
running job.

The fix: a strict load path (``load_active_jobs_strict``) that raises
``ActiveJobsCorruptError`` on corrupt or non-list state. The mutating
helpers (``update_active_jobs_locked`` / ``register_job`` / ``upsert_job`` /
``mark_job_terminal``) now use the strict path INSIDE the lock and
quarantine the corrupt file (rename to ``.corrupt.<utc>``) before
refusing the dispatch. Read-only callers (e.g., harvesters that already
fail-closed on empty state) keep the lenient ``load_active_jobs`` path
for backward compatibility.

Sister of:

- ``tac.continual_learning.posterior_update_locked`` (catalog #128)
- ``tac.vastai_tracker.register_instance`` (sibling fcntl pattern)
- ``tac.deploy.lightning.lightning_dispatch._lightning_state_lock`` (sibling)

Memory: feedback_codex_round3_findings_fix_landed_20260509.md.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[4]
ACTIVE_JOBS_PATH = REPO_ROOT / ".omx" / "state" / "lightning_active_jobs.json"
ACTIVE_JOBS_LOCK = ACTIVE_JOBS_PATH.with_suffix(ACTIVE_JOBS_PATH.suffix + ".lock")


class ActiveJobsCorruptError(RuntimeError):
    """Raised when the active-jobs file is corrupt or non-list and cannot be
    safely mutated.

    The mutating helpers (``update_active_jobs_locked``, ``register_job``,
    etc.) raise this rather than silently overwriting the bad file with
    an empty list — which would drop every active and terminal row and
    leave dispatchers / harvesters out of sync. The corrupt file is
    QUARANTINED (renamed to ``.corrupt.<utc>``) so the operator can
    investigate; the canonical path is then absent on subsequent calls
    (the next register_job will create it fresh).

    Repair recipe:
      1. Inspect ``.omx/state/lightning_active_jobs.json.corrupt.<utc>``
      2. If it's recoverable (e.g. truncated tail), hand-edit and rename
         back to ``lightning_active_jobs.json``.
      3. If unrecoverable, leave the quarantine file as forensic evidence;
         the next register_job will start with an empty list.

    Memory: feedback_codex_round3_findings_fix_landed_20260509.md.
    """


@contextlib.contextmanager
def _active_jobs_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the active-jobs lock file.

    Lock is process-advisory (``fcntl.flock`` ``LOCK_EX``); multiple
    processes contending serialize on the lock file.
    """
    p = lock_path or ACTIVE_JOBS_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load_active_jobs(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the active-jobs file or return empty list (lenient).

    Safe to call without holding the lock — readers see a stable snapshot
    because writers use ``os.replace`` for the final commit. Concurrent
    readers may race against writers, but each individual read sees either
    the pre-write state or the post-write state, never a partial write.

    LENIENT semantics: corrupt JSON or non-list state returns ``[]``.
    This is appropriate for read-only callers (harvesters that already
    fail-closed when no active jobs are seen) but UNSAFE for mutating
    callers — see ``load_active_jobs_strict`` and the MEDIUM 2 note in
    the module docstring.
    """
    p = path or ACTIVE_JOBS_PATH
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    return rows


def _quarantine_corrupt_file(path: Path) -> Path:
    """Move ``path`` to ``path.corrupt.<utc>`` for forensic inspection.

    Idempotent: if ``path`` does not exist, this is a no-op and returns
    ``path``. Otherwise, returns the new quarantine path. Uses ``utcnow``
    in ISO basic format (``YYYYMMDDTHHMMSSZ``) so successive corruption
    events produce distinct file names. Per CLAUDE.md "Forbidden /tmp
    paths in any persisted artifact", the quarantine path stays a sibling
    of the canonical path under ``.omx/state/`` (NOT in ``/tmp``).
    """
    if not path.exists():
        return path
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}")
    # Bump suffix if a same-second collision happens (unlikely but possible).
    counter = 0
    while quarantine.exists():
        counter += 1
        quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}.{counter}")
    os.rename(path, quarantine)
    return quarantine


def load_active_jobs_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """Strict load for mutating callers — raises ActiveJobsCorruptError on corrupt state.

    MUST be called from inside ``_active_jobs_lock`` by mutating callers.
    The mutating helpers (``register_job`` / ``upsert_job`` /
    ``mark_job_terminal`` / ``update_active_jobs_locked``) use this so a
    malformed ``.omx/state/lightning_active_jobs.json`` is NEVER silently
    overwritten with a fresh empty list (which would drop every active
    and terminal row).

    Returns ``[]`` if the path simply does not exist (the empty-tracker
    bootstrap case is normal and not corruption).

    Raises:
        ActiveJobsCorruptError: when the file exists and is either
            invalid JSON or not a list.
    """
    p = path or ACTIVE_JOBS_PATH
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActiveJobsCorruptError(
            f"active-jobs file at {p} contains invalid JSON: {exc}. "
            "Mutating writes are refused to avoid dropping active and "
            "terminal rows. Operator: inspect the file, then either fix "
            "it in place OR move it aside; the next register_job will "
            "create a fresh empty file."
        ) from exc
    if not isinstance(rows, list):
        raise ActiveJobsCorruptError(
            f"active-jobs file at {p} has non-list root (type={type(rows).__name__}); "
            "mutating writes are refused. Expected a list of job records."
        )
    return rows


def _save_active_jobs(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomic write — unique tmp + fsync + os.replace.

    MUST be called inside ``_active_jobs_lock``. The CALLER is responsible
    for the lock; this method enforces only the unique-tmp + fsync invariants.
    """
    p = path or ACTIVE_JOBS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, indent=2) + "\n"
    tmp = p.with_suffix(p.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def update_active_jobs_locked(
    mutate_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    quarantine_on_corrupt: bool = True,
) -> list[dict[str, Any]]:
    """Locked transactional update of the active-jobs file.

    Acquires exclusive fcntl lock on ``lock_path``, then INSIDE the lock:
      1. STRICT-reload state from disk (raises ``ActiveJobsCorruptError``
         on corrupt JSON or non-list root)
      2. apply ``mutate_fn(rows) -> rows`` to produce the new state
      3. write back atomically via ``_save_active_jobs``

    Returns the new state. Multiple parallel dispatchers serialize on the
    lock so updates of distinct rows all survive. Without this, two
    dispatchers loading the same stale state + each appending their own row
    + each replacing → ONE row's update silently dropped.

    MEDIUM 2 (codex round-3, 2026-05-09): on corrupt state the previous
    behaviour was to silently treat the file as empty and overwrite it,
    DROPPING every active and terminal row. The new contract:

      - quarantine_on_corrupt=True (default): rename the bad file to
        ``.corrupt.<utc>`` so the operator can investigate, then RAISE
        ``ActiveJobsCorruptError``. Dispatch is refused; the next call
        will create a fresh empty file. The mutate is NOT applied.
      - quarantine_on_corrupt=False: do not rename; simply raise
        ``ActiveJobsCorruptError``. Useful for tests that don't want a
        side-effect on the canonical state path.

    Example:

        def append_record(rows):
            rows.append({"job_name": ..., "lane_id": ...})
            return rows

        update_active_jobs_locked(append_record)
    """
    p_path = path or ACTIVE_JOBS_PATH
    l_path = lock_path or ACTIVE_JOBS_LOCK
    with _active_jobs_lock(l_path):
        try:
            rows = load_active_jobs_strict(p_path)
        except ActiveJobsCorruptError:
            if quarantine_on_corrupt:
                quarantine_path = _quarantine_corrupt_file(p_path)
                raise ActiveJobsCorruptError(
                    f"active-jobs file at {p_path} was corrupt; "
                    f"quarantined to {quarantine_path}. Mutate refused; "
                    "operator must repair (see ActiveJobsCorruptError docstring)."
                )
            raise
        new_rows = mutate_fn(rows)
        _save_active_jobs(new_rows, p_path)
        return new_rows


def register_job(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Append ``record`` to the active-jobs file under fcntl lock.

    Idempotency check is the CALLER's responsibility (job_name uniqueness
    is enforced by the dispatcher, not by this helper).
    """

    def _append(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows.append(record)
        return rows

    return update_active_jobs_locked(_append, path=path, lock_path=lock_path)


def upsert_job(
    record: dict[str, Any],
    *,
    key: str = "job_name",
    path: Path | None = None,
    lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Append ``record`` if no row with the same ``record[key]`` exists,
    otherwise replace the existing row in-place under fcntl lock.
    """
    target_key = record.get(key)
    if target_key is None:
        raise ValueError(f"upsert_job: record missing key {key!r}")

    def _upsert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        new_rows = [r for r in rows if r.get(key) != target_key]
        new_rows.append(record)
        return new_rows

    return update_active_jobs_locked(_upsert, path=path, lock_path=lock_path)


def mark_job_terminal(
    job_name: str,
    *,
    terminal_status: str,
    extra_fields: dict[str, Any] | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Mark the row with ``job_name`` as terminal under fcntl lock."""

    def _mark(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            if row.get("job_name") == job_name:
                row["terminal_status"] = terminal_status
                if extra_fields:
                    row.update(extra_fields)
                break
        return rows

    return update_active_jobs_locked(_mark, path=path, lock_path=lock_path)


__all__ = [
    "ACTIVE_JOBS_PATH",
    "ACTIVE_JOBS_LOCK",
    "ActiveJobsCorruptError",
    "load_active_jobs",
    "load_active_jobs_strict",
    "update_active_jobs_locked",
    "register_job",
    "upsert_job",
    "mark_job_terminal",
]
