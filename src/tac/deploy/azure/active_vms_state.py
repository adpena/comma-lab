# SPDX-License-Identifier: MIT
"""Shared `.omx/state/azure_active_vms.json` writer with fcntl serialization.

Per CLAUDE.md non-negotiable + catalog #131 (proactive custody+concurrency
audit, 2026-05-09): every shared-state write must serialize on a sibling
fcntl lockfile and use a unique tmp suffix to prevent concurrent dispatchers
from silently dropping each other's updates.

Before this module landed, ``src/tac/deploy/azure/azure_dispatch.py`` did

    rows = _load_active_vms()       # racey read OUTSIDE any lock
    rows.append(record)              # in-process mutation
    _save_active_vms(rows)           # racey write — write_text directly

Two concurrent ``provision_spot_vm`` calls (e.g., parent dispatcher + sister
recovery script firing the same minute) silently dropped each other's
updates because the load happened before any lock and the write replaced
the file unconditionally.

The previous module was incorrectly listed in catalog #131's
``_BARE_WRITE_CANONICAL_HELPERS`` exempt list as "already locked" (codex
round 4 HIGH 1 finding, 2026-05-09). The exemption masked the fact that
no fcntl lock existed; the META gate reported false-green while concurrent
Azure provisions could still drop VM rows.

This module exposes the canonical transactional API: ``register_active_vm`` /
``unregister_active_vm`` both do load→mutate→save inside an exclusive
``fcntl.flock(LOCK_EX)`` so concurrent updates of distinct rows all survive.

Sister of:

- ``tac.continual_learning.posterior_update_locked`` (catalog #128)
- ``tac.deploy.lightning.active_jobs_state.update_active_jobs_locked``
  (catalog #131 sister)
- ``tac.vastai_tracker.register_instance`` (sibling fcntl pattern)
- ``scripts/verify_vast_instances.py::_save_setup_first_seen``
  (catalog #132 sister, transactional REPLACE form)

Memory: feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md.
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
ACTIVE_VMS_PATH = REPO_ROOT / ".omx" / "state" / "azure_active_vms.json"
ACTIVE_VMS_LOCK = ACTIVE_VMS_PATH.with_suffix(ACTIVE_VMS_PATH.suffix + ".lock")


class ActiveVMsCorruptError(RuntimeError):
    """Raised when the active-VMs file is corrupt or non-list and cannot be
    safely mutated.

    The mutating helpers (``update_active_vms_locked``, ``register_active_vm``,
    ``unregister_active_vm``) raise this rather than silently overwriting the
    bad file with an empty list — which would drop every active VM row and
    leave cleanup tooling unable to detect orphans. The corrupt file is
    QUARANTINED (renamed to ``.corrupt.<utc>``) so the operator can
    investigate; the canonical path is then absent on subsequent calls
    (the next register_active_vm will create it fresh).

    Repair recipe:
      1. Inspect ``.omx/state/azure_active_vms.json.corrupt.<utc>``
      2. If it's recoverable (e.g. truncated tail), hand-edit and rename
         back to ``azure_active_vms.json``.
      3. If unrecoverable, leave the quarantine file as forensic evidence;
         the next register_active_vm will start with an empty list.

    Memory: feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md.
    """


# Codex round 5 HIGH 2 fix sister landing (catalog #140): in-process
# lock-held depth counter. Pairs with ``_active_vms_lock_held()`` so
# ``_save_active_vms_atomic`` can runtime-assert "caller holds the lock".
_active_vms_lock_depth: int = 0


def _active_vms_lock_held() -> bool:
    """Return True if THIS process is currently inside ``_active_vms_lock``."""
    return _active_vms_lock_depth > 0


@contextlib.contextmanager
def _active_vms_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the active-VMs lock file.

    Lock is process-advisory (``fcntl.flock`` ``LOCK_EX``); multiple
    processes contending serialize on the lock file.

    Codex round 5 HIGH 2 fix sister landing (catalog #140): tracks an
    in-process depth counter so ``_save_active_vms_atomic`` can refuse
    writes that bypass the canonical locked path.
    """
    global _active_vms_lock_depth

    p = lock_path or ACTIVE_VMS_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    if _active_vms_lock_depth > 0:
        _active_vms_lock_depth += 1
        try:
            yield None
        finally:
            _active_vms_lock_depth -= 1
        return
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _active_vms_lock_depth += 1
        try:
            yield fd
        finally:
            _active_vms_lock_depth -= 1
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load_active_vms(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the active-VMs file or return empty list (lenient).

    Safe to call without holding the lock — readers see a stable snapshot
    because writers use ``os.replace`` for the final commit. LENIENT
    semantics: corrupt JSON or non-list state returns ``[]``. UNSAFE for
    mutating callers — see ``load_active_vms_strict`` and
    ``update_active_vms_locked``.
    """
    p = path or ACTIVE_VMS_PATH
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
    ``path``. Otherwise, returns the new quarantine path. Per CLAUDE.md
    "Forbidden /tmp paths in any persisted artifact", the quarantine
    path stays a sibling of the canonical path under ``.omx/state/``.
    """
    if not path.exists():
        return path
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}")
    counter = 0
    while quarantine.exists():
        counter += 1
        quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}.{counter}")
    os.rename(path, quarantine)
    return quarantine


def load_active_vms_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """Strict load for mutating callers — raises ActiveVMsCorruptError on corrupt state.

    MUST be called from inside ``_active_vms_lock`` by mutating callers.
    The mutating helpers (``update_active_vms_locked`` / ``register_active_vm`` /
    ``unregister_active_vm``) use this so a malformed
    ``.omx/state/azure_active_vms.json`` is NEVER silently overwritten with
    a fresh empty list (which would drop every active VM row and prevent
    cleanup tooling from finding orphans).

    Returns ``[]`` if the path simply does not exist (the empty-tracker
    bootstrap case is normal and not corruption).

    Raises:
        ActiveVMsCorruptError: when the file exists and is either invalid
            JSON or not a list.
    """
    p = path or ACTIVE_VMS_PATH
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActiveVMsCorruptError(
            f"active-VMs file at {p} contains invalid JSON: {exc}. "
            "Mutating writes are refused to avoid dropping active VM rows. "
            "Operator: inspect the file, then either fix it in place OR "
            "move it aside; the next register_active_vm will create a "
            "fresh empty file."
        ) from exc
    if not isinstance(rows, list):
        raise ActiveVMsCorruptError(
            f"active-VMs file at {p} has non-list root (type={type(rows).__name__}); "
            "mutating writes are refused. Expected a list of VM records."
        )
    return rows


def _save_active_vms_atomic(
    rows: list[dict[str, Any]], path: Path | None = None
) -> None:
    """Atomic write — unique tmp + fsync + os.replace.

    Codex round 5 HIGH 2 fix sister landing (catalog #140): runtime-asserts
    the caller holds ``_active_vms_lock``. Pre-fix this method documented
    "MUST be called inside the lock" but did not enforce it; comment-only
    contracts are FORBIDDEN per CLAUDE.md.
    """
    if not _active_vms_lock_held():
        raise RuntimeError(
            "_save_active_vms_atomic called WITHOUT holding _active_vms_lock. "
            "This is a CONCURRENCY BUG: concurrent register/unregister "
            "calls can silently drop rows. Use update_active_vms_locked / "
            "register_active_vm_record / unregister_active_vm_by_name "
            "which own the full lock-load-mutate-save cycle. See codex "
            "round 5 HIGH 2 fix (catalog #140)."
        )
    p = path or ACTIVE_VMS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, indent=2, sort_keys=True) + "\n"
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


def update_active_vms_locked(
    mutate_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    quarantine_on_corrupt: bool = True,
) -> list[dict[str, Any]]:
    """Locked transactional update of the active-VMs file.

    Acquires exclusive fcntl lock on ``lock_path``, then INSIDE the lock:
      1. STRICT-reload state from disk (raises ``ActiveVMsCorruptError`` on
         corrupt JSON or non-list root)
      2. apply ``mutate_fn(rows) -> rows`` to produce the new state
      3. write back atomically via ``_save_active_vms_atomic``

    Returns the new state. Multiple parallel dispatchers serialize on the
    lock so updates of distinct rows all survive. Without this, two
    dispatchers loading the same stale state + each appending their own row
    + each replacing → ONE row's update silently dropped.

    Sister contract to
    ``tac.deploy.lightning.active_jobs_state.update_active_jobs_locked``.

    Example:

        def append_record(rows):
            rows.append({"vm_name": ..., "label": ...})
            return rows

        update_active_vms_locked(append_record)
    """
    p_path = path or ACTIVE_VMS_PATH
    l_path = lock_path or ACTIVE_VMS_LOCK
    with _active_vms_lock(l_path):
        try:
            rows = load_active_vms_strict(p_path)
        except ActiveVMsCorruptError:
            if quarantine_on_corrupt:
                quarantine_path = _quarantine_corrupt_file(p_path)
                raise ActiveVMsCorruptError(
                    f"active-VMs file at {p_path} was corrupt; "
                    f"quarantined to {quarantine_path}. Mutate refused; "
                    "operator must repair (see ActiveVMsCorruptError docstring)."
                )
            raise
        new_rows = mutate_fn(rows)
        _save_active_vms_atomic(new_rows, p_path)
        return new_rows


def register_active_vm_record(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Append ``record`` to the active-VMs file under fcntl lock."""

    def _append(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows.append(record)
        return rows

    return update_active_vms_locked(_append, path=path, lock_path=lock_path)


def unregister_active_vm_by_name(
    vm_name: str,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Remove the row whose ``vm_name`` matches under fcntl lock."""

    def _remove(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if r.get("vm_name") != vm_name]

    return update_active_vms_locked(_remove, path=path, lock_path=lock_path)


__all__ = [
    "ACTIVE_VMS_PATH",
    "ACTIVE_VMS_LOCK",
    "ActiveVMsCorruptError",
    "load_active_vms",
    "load_active_vms_strict",
    "update_active_vms_locked",
    "register_active_vm_record",
    "unregister_active_vm_by_name",
]
