# SPDX-License-Identifier: MIT
"""Centralized Vast.ai instance tracker.

Per CLAUDE.md non-negotiable + memory `feedback_oneshot_vastai_subagent_failure_pattern`:
every Vast.ai launch (any `vastai create instance` invocation) MUST register
the instance ID in a tracker file so a separate cleanup script can detect
orphans (instances that were created but never destroyed).

This module is the canonical entrypoint. It writes
`.omx/state/vastai_active_instances.json` (a JSON array of records) using a
file lock so concurrent launches do not corrupt the file.

Usage:

    from tac.vastai_tracker import register_instance
    register_instance(
        instance_id="123456",
        label="pact-example-20260427",
        metadata={"dph": 0.25, "experiment": "example"},
    )

The companion CLI tool `tools/vastai_orphan_cleanup.py` reads this file,
queries `vastai show instances`, and identifies orphans.

Codex round 7 HIGH 2 (2026-05-09, Catalog #148):
the previous ``_load_records`` helper silently returned ``[]`` for malformed
tracker JSON. ``register_instance`` then wrote the new single record over the
corrupt file, dropping every previously-tracked active instance. The fix mirrors
the canonical pattern from
``tac.deploy.lightning.active_jobs_state.load_active_jobs_strict``: a strict
loader raises ``VastaiTrackerCorruptError``; the locked writers quarantine the
corrupt state to ``<path>.corrupt.<utc>`` and refuse to overwrite.
``list_instances`` retains the lossy-on-purpose ``_load_records`` for read-only
consumers; mutating callers (``register_instance`` / ``remove_instance``) MUST
use the strict path.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Canonical path. The preflight scanner
# (`check_vastai_create_writes_tracker`) literally greps for
# "vastai_active_instances" within 30 lines after a `vastai create instance`
# call. The string below — and any caller of this module — therefore
# satisfies the meta-bug check.
_TRACKER_REL = ".omx/state/vastai_active_instances.json"


class VastaiTrackerCorruptError(RuntimeError):
    """Raised when the on-disk vastai_active_instances tracker is malformed.

    Codex round 7 HIGH 2 fix (Catalog #148): mutating callers MUST refuse to
    proceed when the tracker JSON is malformed. The previous behaviour
    silently returned ``[]`` and overwrote the file with a single new record,
    dropping every other tracked instance. Read-only consumers (``list_instances``,
    cleanup tools) may continue to use the lossy ``_load_records`` path.

    The raiser is responsible for quarantining the corrupt file via
    :func:`_quarantine_corrupt_tracker` so that the next call starts fresh.
    """


def _repo_root() -> Path:
    """Walk up from this file until we find the repo root (has .omx/)."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / ".omx").is_dir():
            return parent
    # Fallback: assume src/tac/<file>, repo root is two levels up.
    return here.parents[2]


def tracker_path(repo_root: Path | None = None) -> Path:
    """Absolute path to vastai_active_instances.json."""
    root = repo_root if repo_root is not None else _repo_root()
    return root / _TRACKER_REL


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Read the tracker JSON; return [] if missing or malformed.

    READ-ONLY consumers ONLY. Mutating writers (``register_instance``,
    ``remove_instance``) MUST use :func:`load_active_instances_strict`
    instead so a corrupt tracker fails closed rather than silently
    dropping every active instance on the next write (codex round 7 HIGH 2,
    Catalog #148).
    """
    if not path.exists():
        return []
    try:
        text = path.read_text()
        if not text.strip():
            return []
        data = json.loads(text)
        if isinstance(data, list):
            return data
        # Tolerate legacy {"instances": [...]} shape.
        if isinstance(data, dict) and isinstance(data.get("instances"), list):
            return data["instances"]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def load_active_instances_strict(path: Path) -> list[dict[str, Any]]:
    """Read the tracker JSON, raising ``VastaiTrackerCorruptError`` on failure.

    Codex round 7 HIGH 2 (Catalog #148) — mirrors
    ``tac.deploy.lightning.active_jobs_state.load_active_jobs_strict``.

    - missing file → returns ``[]`` (treated as empty, NOT corrupt — first
      registration is allowed to create the file)
    - empty file → returns ``[]``
    - JSON decode error → raises ``VastaiTrackerCorruptError``
    - JSON is dict-without-"instances" / scalar / nested-list → raises
      ``VastaiTrackerCorruptError``
    - JSON is the legacy ``{"instances": [...]}`` shape → returns the list
    - JSON is the canonical ``[...]`` shape → returns the list
    """
    if not path.exists():
        return []
    try:
        text = path.read_text()
    except OSError as exc:
        raise VastaiTrackerCorruptError(
            f"vastai tracker {path} unreadable: {exc!r}"
        ) from exc
    if not text.strip():
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VastaiTrackerCorruptError(
            f"vastai tracker {path} contains malformed JSON: {exc!r}"
        ) from exc
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("instances"), list):
        return data["instances"]
    raise VastaiTrackerCorruptError(
        f"vastai tracker {path} has unrecognised JSON shape "
        f"(top-level type={type(data).__name__})."
    )


def _quarantine_corrupt_tracker(path: Path) -> Path | None:
    """Move a corrupt tracker file aside so the next call can start fresh.

    Returns the quarantined path, or ``None`` if the source did not exist.
    The quarantine name is ``<path>.corrupt.<utc>`` (mirror of the canonical
    pattern in ``tac.deploy.lightning.active_jobs_state``).
    """
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = path.with_suffix(path.suffix + f".corrupt.{stamp}")
    try:
        path.rename(dest)
        print(
            f"[vastai-tracker] quarantined corrupt tracker → {dest}",
            file=sys.stderr,
        )
        return dest
    except OSError:
        # Best-effort: if rename fails (permission/cross-device), leave the
        # file in place. The strict loader will keep raising on re-read.
        return None


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def register_instance(
    instance_id: str,
    label: str,
    metadata: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Append an instance record to the canonical tracker file.

    The write is performed under an `fcntl.flock(LOCK_EX)` on the parent
    directory so concurrent launches (rare but possible from the deploy
    pipeline) do not race.

    Returns the record that was written.
    """
    if not instance_id:
        raise ValueError("instance_id must be a non-empty string")
    if not label:
        raise ValueError("label must be a non-empty string")
    record: dict[str, Any] = {
        "instance_id": str(instance_id),
        "label": str(label),
        "registered_at_utc": _now_iso(),
        "registered_by_pid": os.getpid(),
        "registered_by_host": socket.gethostname(),
    }
    if metadata:
        # Coerce every value to JSON-safe; refuse non-serialisable structures
        # so a typo here doesn't silently lose the metadata.
        try:
            json.dumps(metadata)
            record["metadata"] = metadata
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"register_instance: metadata is not JSON-serialisable: {e!r}"
            ) from e

    path = tracker_path(repo_root=repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a sibling lockfile for fcntl (Path.write_text doesn't expose fd).
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lockfd:
        fcntl.flock(lockfd.fileno(), fcntl.LOCK_EX)
        try:
            # Codex round 7 HIGH 2 (Catalog #148): use strict loader inside
            # the lock so a corrupt tracker fails closed — never silently
            # overwrite N tracked instances with a single new record.
            try:
                records = load_active_instances_strict(path)
            except VastaiTrackerCorruptError:
                _quarantine_corrupt_tracker(path)
                # Re-raise so the caller surfaces the corrupt-tracker event
                # to the operator (loud orphan-recovery banner) instead of
                # silently treating it as empty.
                raise
            records.append(record)
            _write_records(path, records)
        finally:
            fcntl.flock(lockfd.fileno(), fcntl.LOCK_UN)
    return record


def list_instances(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Return every record in the tracker (in registration order).

    Read-only path: tolerates malformed JSON and returns ``[]`` so cleanup
    tools can continue. Mutating callers MUST use the strict loader inside
    a fcntl-locked register/remove call (see :func:`register_instance`).
    """
    return _load_records(tracker_path(repo_root=repo_root))


def remove_instance(
    instance_id: str,
    repo_root: Path | None = None,
) -> bool:
    """Drop the record for `instance_id` (after destroy). Returns True if found.

    Codex round 7 HIGH 2 (Catalog #148): uses the strict loader inside the
    lock and quarantines the tracker on corruption so we never overwrite
    a corrupt file with a single removal record (which would drop every
    other tracked instance).
    """
    path = tracker_path(repo_root=repo_root)
    if not path.exists():
        return False
    lock_path = path.with_suffix(path.suffix + ".lock")
    found = False
    with open(lock_path, "w") as lockfd:
        fcntl.flock(lockfd.fileno(), fcntl.LOCK_EX)
        try:
            try:
                records = load_active_instances_strict(path)
            except VastaiTrackerCorruptError:
                _quarantine_corrupt_tracker(path)
                raise
            new_records = [
                r for r in records if str(r.get("instance_id")) != str(instance_id)
            ]
            found = len(new_records) != len(records)
            if found:
                _write_records(path, new_records)
        finally:
            fcntl.flock(lockfd.fileno(), fcntl.LOCK_UN)
    return found


__all__ = [
    "VastaiTrackerCorruptError",
    "tracker_path",
    "load_active_instances_strict",
    "list_instances",
    "register_instance",
    "remove_instance",
]
