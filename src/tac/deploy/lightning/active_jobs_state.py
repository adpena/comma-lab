# SPDX-License-Identifier: MIT
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
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


# Codex round 5 HIGH 2 fix sister landing (catalog #140): in-process
# lock-held depth counter. Pairs with ``_active_jobs_lock_held()`` so
# ``_save_active_jobs`` can runtime-assert "caller holds the lock"
# instead of relying on a comment-only contract.
#
# OP-7 fix (codex chunk 5, 2026-05-15): the depth counter MUST be
# thread-local. Pre-fix, the counter was a module-level int shared across
# threads. Two threads in the same process could observe depth>0
# simultaneously and BOTH skip the fcntl acquire (lines 132-135 in the
# original `_active_jobs_lock` body), since that block only runs on the
# 0->1 transition. Thread A enters with depth=0, acquires fcntl,
# increments to 1; thread B then enters, sees depth==1 (treats it as
# "re-entrant within this process is safe"), increments to 2, and runs
# the critical section concurrently with thread A — silently dropping
# concurrent register/upsert/mark calls. Sister of Catalog #131
# (`check_no_bare_writes_to_shared_state`) + Catalog #140
# (`check_state_writers_own_their_lock_end_to_end`): #131 + #140 enforce
# the lock-held contract; OP-7 makes the depth counter PER-THREAD so the
# contract is honored across threaded dispatch (e.g. `concurrent.futures.
# ThreadPoolExecutor` over `tools/parallel_dispatch_top_k.py`).
_active_jobs_lock_depth_tls = threading.local()


def _get_active_jobs_lock_depth() -> int:
    """Return THIS thread's local depth counter (default 0)."""

    return int(getattr(_active_jobs_lock_depth_tls, "depth", 0))


def _set_active_jobs_lock_depth(value: int) -> None:
    """Set THIS thread's local depth counter."""

    _active_jobs_lock_depth_tls.depth = int(value)


def _active_jobs_lock_held() -> bool:
    """Return True if THIS THREAD is currently inside ``_active_jobs_lock``.

    OP-7 fix (2026-05-15): thread-local query so ``_save_active_jobs``
    refuses writes from sibling threads that did NOT acquire the lock,
    even when another thread in the same process holds it.
    """
    return _get_active_jobs_lock_depth() > 0


@contextlib.contextmanager
def _active_jobs_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the active-jobs lock file.

    Lock is process-advisory (``fcntl.flock`` ``LOCK_EX``); multiple
    processes contending serialize on the lock file.

    Codex round 5 HIGH 2 fix sister landing (catalog #140): tracks an
    in-process depth counter so ``_save_active_jobs`` can refuse writes
    that bypass the canonical locked path. Re-entry within the same
    THREAD is counted (depth > 1); fcntl is only re-acquired on the
    0->1 transition to avoid same-process deadlock.

    OP-7 fix (codex chunk 5, 2026-05-15): the depth counter is now
    PER-THREAD. A different thread in the same process that enters this
    context manager will see thread-local depth=0 and proceed to
    fcntl-acquire; ``fcntl.flock`` blocks (LOCK_EX) until the holding
    thread releases. This serializes inter-thread access correctly.
    Re-entry within the SAME thread still short-circuits (no deadlock).
    """

    p = lock_path or ACTIVE_JOBS_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    if _get_active_jobs_lock_depth() > 0:
        _set_active_jobs_lock_depth(_get_active_jobs_lock_depth() + 1)
        try:
            yield None
        finally:
            _set_active_jobs_lock_depth(_get_active_jobs_lock_depth() - 1)
        return
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _set_active_jobs_lock_depth(_get_active_jobs_lock_depth() + 1)
        try:
            yield fd
        finally:
            _set_active_jobs_lock_depth(_get_active_jobs_lock_depth() - 1)
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
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
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

    Codex round 5 HIGH 2 fix sister landing (catalog #140): runtime-asserts
    the caller holds ``_active_jobs_lock``. Pre-fix this method documented
    "MUST be called inside the lock" but did not enforce it; comment-only
    contracts are FORBIDDEN per CLAUDE.md.
    """
    if not _active_jobs_lock_held():
        raise RuntimeError(
            "_save_active_jobs called WITHOUT holding _active_jobs_lock. "
            "This is a CONCURRENCY BUG: concurrent register/upsert/mark "
            "calls can silently drop rows. Use update_active_jobs_locked / "
            "register_job / upsert_job / mark_job_terminal which own the "
            "full lock-load-mutate-save cycle. See codex round 5 HIGH 2 "
            "fix (catalog #140)."
        )
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
        except ActiveJobsCorruptError as exc:
            if quarantine_on_corrupt:
                quarantine_path = _quarantine_corrupt_file(p_path)
                raise ActiveJobsCorruptError(
                    f"active-jobs file at {p_path} was corrupt; "
                    f"quarantined to {quarantine_path}. Mutate refused; "
                    "operator must repair (see ActiveJobsCorruptError docstring)."
                ) from exc
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


# ─────────────────────────────────────────────────────────────────────────
# Codex round 6 HIGH 2 fix (catalog #143): create-pending-row-before-submit
# ─────────────────────────────────────────────────────────────────────────
#
# Lightning dispatchers previously called ``Job.run(...)`` (which CREATES
# the paid job + bills the operator) THEN persisted the active-jobs row.
# With round-3's strict ``register_job``, a corrupt active-jobs file →
# paid job created but tracker write fails → invisible orphan paid job
# (the harvester never knows).
#
# The canonical fix is the create-pending-row-before-submit pattern:
#
#   1. ``register_pending_job_locked(metadata)`` — writes a "pending" row
#      to the tracker BEFORE submit. If the tracker is corrupt, dispatcher
#      refuses to even attempt the paid submit.
#   2. ``Job.run(...)`` — actually submit; bill starts here.
#   3. ``update_pending_to_active_locked(job_name, ...)`` — promote the
#      pending row to active with the submit result.
#   4. On submit failure: ``cancel_pending_job_locked(job_name)`` to drop
#      the pending row.
#
# A "pending" row is identical schema to an "active" row but carries
# ``status="pending"`` and a sentinel ``submit_result={"status":"pending"}``
# so the harvester can distinguish pending-but-never-submitted from
# truly active jobs.

PENDING_STATUS_TOKEN = "pending"
ACTIVE_STATUS_TOKEN = "active"


class PendingJobNotFoundError(RuntimeError):
    """Raised when ``update_pending_to_active_locked`` or
    ``cancel_pending_job_locked`` is asked to operate on a job_name with
    no matching ``status=pending`` row. Defensive: catches a refactor
    that drops the pending-row precondition.
    """


def register_pending_job_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Append a ``status=pending`` row BEFORE the paid submit.

    Use this immediately before any paid ``Job.run(...)``. If the
    active-jobs tracker is corrupt, this call raises
    ``ActiveJobsCorruptError`` and the dispatcher refuses to attempt
    the submit at all. If it succeeds, the pending row is durable
    (fsync + atomic replace) so even a process crash mid-submit
    leaves a forensic record the operator can inspect.

    The caller MUST set a unique ``record["job_name"]`` (the
    dispatcher's job_name); the row is keyed by it for the
    ``update_pending_to_active_locked`` / ``cancel_pending_job_locked``
    follow-ups.

    Idempotency: if a row with the same ``job_name`` AND
    ``status=pending`` already exists, this is a no-op (returns the
    current state). If a row with the same ``job_name`` exists but
    is NOT pending, this raises ``ValueError`` to refuse silently
    overwriting a real active row.
    """
    job_name = record.get("job_name")
    if not job_name:
        raise ValueError("register_pending_job_locked: record must include 'job_name'")
    pending_record = {**record}
    pending_record.setdefault("status", PENDING_STATUS_TOKEN)
    pending_record["submit_result"] = {"status": PENDING_STATUS_TOKEN}

    def _append_pending(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            if r.get("job_name") == job_name:
                if r.get("status") == PENDING_STATUS_TOKEN:
                    return rows  # idempotent
                raise ValueError(
                    f"register_pending_job_locked: refusing to overwrite "
                    f"existing non-pending row for job_name={job_name!r} "
                    f"(status={r.get('status')!r}). The dispatcher must "
                    "use a unique job_name per submit attempt."
                )
        rows.append(pending_record)
        return rows

    return update_active_jobs_locked(_append_pending, path=path, lock_path=lock_path)


def update_pending_to_active_locked(
    job_name: str,
    *,
    submit_result: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Promote a pending row to active after successful submit.

    Replaces the pending row (matched by ``job_name``) with an active
    row that records the actual submit_result. Raises
    ``PendingJobNotFoundError`` if no pending row exists for ``job_name``
    (which would indicate a refactor regression — every submit must
    have a pending precursor).
    """

    def _promote(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        promoted = False
        for row in rows:
            if (
                row.get("job_name") == job_name
                and row.get("status") == PENDING_STATUS_TOKEN
            ):
                row["status"] = ACTIVE_STATUS_TOKEN
                row["submit_result"] = submit_result
                if extra_fields:
                    row.update(extra_fields)
                promoted = True
                break
        if not promoted:
            raise PendingJobNotFoundError(
                f"update_pending_to_active_locked: no pending row for "
                f"job_name={job_name!r}. Dispatcher MUST call "
                "`register_pending_job_locked(...)` BEFORE any paid "
                "submit (catalog #143). If you reach here from a "
                "refactor, the create-pending-row-before-submit pattern "
                "was bypassed."
            )
        return rows

    return update_active_jobs_locked(_promote, path=path, lock_path=lock_path)


FAILED_UNKNOWN_BILLING_STATUS_TOKEN = "failed_unknown_billing"


def mark_pending_failed_unknown_billing_locked(
    job_name: str,
    *,
    failure_reason: str,
    submit_partial_result: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Codex round 7+8 HIGH 1 (Catalog #147) — mark pending row as
    ``status=failed_unknown_billing`` when submit raised AT/AFTER the network
    call.

    The previous Lightning-dispatcher pattern wrapped ``submit_lightning_job``
    in ``except BaseException`` and unconditionally called
    ``cancel_pending_job_locked``. That assumed every exception happened
    BEFORE billing — which is FALSE for the entire ``Job.run(...)`` window.
    SDK timeouts, post-create API errors, property-access failures, or
    ``KeyboardInterrupt`` mid-submit can leave a real paid Lightning job
    while the previous code silently deleted the only harvester-visible
    breadcrumb.

    The fix splits submit into:

    1. ``cancel_pending_job_locked(...)`` — for KNOWN pre-network failures
       (import error, missing config, invalid args). The dispatcher catches
       these from the pre-network preparation step.
    2. ``mark_pending_failed_unknown_billing_locked(...)`` — for ambiguous
       failures inside the ``Job.run(...)`` window. The pending row is
       PRESERVED and re-tagged so the harvester can see it and the operator
       can manually reconcile against the Lightning dashboard.

    The status token ``failed_unknown_billing`` is intentionally distinct
    from ``active`` (real billing job) and ``pending`` (pre-submit) so the
    harvester can surface these for manual review without flipping them
    into the spend-rollup or the success/failure ledger automatically.

    Raises ``PendingJobNotFoundError`` if no pending row exists.
    """
    if not failure_reason:
        raise ValueError(
            "mark_pending_failed_unknown_billing_locked: failure_reason "
            "must be a non-empty string for forensic recovery"
        )

    def _mark(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        marked = False
        for row in rows:
            if (
                row.get("job_name") == job_name
                and row.get("status") == PENDING_STATUS_TOKEN
            ):
                row["status"] = FAILED_UNKNOWN_BILLING_STATUS_TOKEN
                row["failure_reason"] = failure_reason
                row["submit_status_unknown_at_utc"] = _dt.datetime.now(
                    _dt.UTC,
                ).isoformat(timespec="seconds")
                if submit_partial_result is not None:
                    row["submit_partial_result"] = submit_partial_result
                if extra_fields:
                    row.update(extra_fields)
                marked = True
                break
        if not marked:
            raise PendingJobNotFoundError(
                f"mark_pending_failed_unknown_billing_locked: no pending row "
                f"for job_name={job_name!r}; nothing to mark."
            )
        return rows

    return update_active_jobs_locked(_mark, path=path, lock_path=lock_path)


def cancel_pending_job_locked(
    job_name: str,
    *,
    failure_reason: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Drop a pending row when submit fails BEFORE billing starts.

    Raises ``PendingJobNotFoundError`` if no pending row exists for
    ``job_name`` (defensive: a cancel without a matching pending is a
    refactor regression).

    NOTE: if submit failed but we cannot tell whether billing started
    (e.g., timeout mid-submit), the safer choice is to leave the
    pending row in place AND mark it as ``status=failed_unknown_billing``
    via :func:`mark_pending_failed_unknown_billing_locked` (Codex round
    7+8 HIGH 1, Catalog #147). Cancellation is for the unambiguous
    "submit raised before any network call to Lightning" case ONLY.
    """

    def _cancel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        new_rows: list[dict[str, Any]] = []
        cancelled = False
        for row in rows:
            if (
                row.get("job_name") == job_name
                and row.get("status") == PENDING_STATUS_TOKEN
            ):
                cancelled = True
                continue  # drop
            new_rows.append(row)
        if not cancelled:
            raise PendingJobNotFoundError(
                f"cancel_pending_job_locked: no pending row for "
                f"job_name={job_name!r}; nothing to cancel."
            )
        return new_rows

    return update_active_jobs_locked(_cancel, path=path, lock_path=lock_path)


__all__ = [
    "ACTIVE_JOBS_LOCK",
    "ACTIVE_JOBS_PATH",
    "ACTIVE_STATUS_TOKEN",
    "FAILED_UNKNOWN_BILLING_STATUS_TOKEN",  # Catalog #147
    "PENDING_STATUS_TOKEN",
    "ActiveJobsCorruptError",
    "PendingJobNotFoundError",
    "cancel_pending_job_locked",
    "load_active_jobs",
    "load_active_jobs_strict",
    "mark_job_terminal",
    "mark_pending_failed_unknown_billing_locked",  # Catalog #147
    "register_job",
    "register_pending_job_locked",
    "update_active_jobs_locked",
    "update_pending_to_active_locked",
    "upsert_job",
]
