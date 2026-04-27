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
        label="pact-lane-a-20260427",
        metadata={"dph": 0.25, "experiment": "lane_a"},
    )

The companion CLI tool `tools/vastai_orphan_cleanup.py` reads this file,
queries `vastai show instances`, and identifies orphans.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Canonical path. The preflight scanner
# (`check_vastai_create_writes_tracker`) literally greps for
# "vastai_active_instances" within 30 lines after a `vastai create instance`
# call. The string below — and any caller of this module — therefore
# satisfies the meta-bug check.
_TRACKER_REL = ".omx/state/vastai_active_instances.json"


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
    """Read the tracker JSON; return [] if missing or malformed."""
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
            records = _load_records(path)
            records.append(record)
            _write_records(path, records)
        finally:
            fcntl.flock(lockfd.fileno(), fcntl.LOCK_UN)
    return record


def list_instances(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Return every record in the tracker (in registration order)."""
    return _load_records(tracker_path(repo_root=repo_root))


def remove_instance(
    instance_id: str,
    repo_root: Path | None = None,
) -> bool:
    """Drop the record for `instance_id` (after destroy). Returns True if found."""
    path = tracker_path(repo_root=repo_root)
    if not path.exists():
        return False
    lock_path = path.with_suffix(path.suffix + ".lock")
    found = False
    with open(lock_path, "w") as lockfd:
        fcntl.flock(lockfd.fileno(), fcntl.LOCK_EX)
        try:
            records = _load_records(path)
            new_records = [
                r for r in records if str(r.get("instance_id")) != str(instance_id)
            ]
            found = len(new_records) != len(records)
            if found:
                _write_records(path, new_records)
        finally:
            fcntl.flock(lockfd.fileno(), fcntl.LOCK_UN)
    return found
