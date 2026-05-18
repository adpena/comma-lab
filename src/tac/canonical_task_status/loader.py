# SPDX-License-Identifier: MIT
"""Strict loader for the canonical task-status ledger."""

from __future__ import annotations

import datetime as _dt
import json
import os
from collections.abc import Iterable
from pathlib import Path

from .contract import (
    VALID_TRANSITIONS,
    CanonicalTaskStatusCorruptError,
    CanonicalTaskStatusRow,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LEDGER_RELATIVE_PATH = Path(".omx/state/canonical_task_status.jsonl")
LOCK_RELATIVE_PATH = Path(".omx/state/.canonical_task_status.lock")


def ledger_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return root / LEDGER_RELATIVE_PATH


def lock_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    return root / LOCK_RELATIVE_PATH


def _quarantine_corrupt_file(path: Path) -> Path:
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}")
    counter = 0
    while quarantine.exists():
        counter += 1
        quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}.{counter}")
    os.rename(path, quarantine)
    return quarantine


def _validate_history(rows: Iterable[CanonicalTaskStatusRow]) -> None:
    latest: dict[str, CanonicalTaskStatusRow] = {}
    seen_registration: set[str] = set()
    for row in rows:
        prev = latest.get(row.task_id)
        if row.event_type == "registered":
            if row.task_id in seen_registration:
                raise CanonicalTaskStatusCorruptError(
                    f"duplicate registration for task_id {row.task_id!r}"
                )
            if row.status != "pending":
                raise CanonicalTaskStatusCorruptError(
                    f"registration for {row.task_id!r} must have pending status"
                )
            seen_registration.add(row.task_id)
        elif prev is None:
            raise CanonicalTaskStatusCorruptError(
                f"non-registration event for unknown task_id {row.task_id!r}"
            )
        elif row.event_type != "note":
            allowed = VALID_TRANSITIONS.get(prev.status, frozenset())
            if row.status not in allowed:
                raise CanonicalTaskStatusCorruptError(
                    f"invalid transition for {row.task_id!r}: {prev.status} -> {row.status}"
                )
        elif row.status != prev.status:
            raise CanonicalTaskStatusCorruptError(
                f"note event for {row.task_id!r} changed status {prev.status} -> {row.status}"
            )
        latest[row.task_id] = row


def load_canonical_task_status_strict(
    repo_root: str | Path | None = None,
) -> list[CanonicalTaskStatusRow]:
    """Load and validate every append-only task-status row.

    Corrupt ledgers are quarantined rather than silently overwritten. Missing
    ledgers are valid and load as an empty list.
    """

    path = ledger_path(repo_root)
    if not path.exists():
        return []
    rows: list[CanonicalTaskStatusRow] = []
    try:
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError("row is not an object")
                rows.append(CanonicalTaskStatusRow.from_json_obj(obj))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                quarantine = _quarantine_corrupt_file(path)
                raise CanonicalTaskStatusCorruptError(
                    f"corrupt canonical task-status row {line_number}; quarantined to {quarantine}"
                ) from exc
        _validate_history(rows)
    except CanonicalTaskStatusCorruptError:
        raise
    return rows


def latest_status_by_task_id(
    task_id: str,
    repo_root: str | Path | None = None,
) -> CanonicalTaskStatusRow | None:
    latest = None
    for row in load_canonical_task_status_strict(repo_root):
        if row.task_id == task_id:
            latest = row
    return latest


def latest_statuses(repo_root: str | Path | None = None) -> dict[str, CanonicalTaskStatusRow]:
    out: dict[str, CanonicalTaskStatusRow] = {}
    for row in load_canonical_task_status_strict(repo_root):
        out[row.task_id] = row
    return out

