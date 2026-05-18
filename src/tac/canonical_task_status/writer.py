# SPDX-License-Identifier: MIT
"""Fcntl-locked append helpers for canonical task status."""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Literal

from .contract import (
    VALID_TRANSITIONS,
    CanonicalTaskStatusInvalidTransitionError,
    CanonicalTaskStatusRow,
)
from .loader import (
    latest_status_by_task_id,
    ledger_path,
    load_canonical_task_status_strict,
    lock_path,
)

LOCK_TIMEOUT_SECONDS = 30.0


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_iso(value: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(_dt.UTC)


def _monotonic_event_timestamp(prev: CanonicalTaskStatusRow | None = None) -> str:
    now = _dt.datetime.now(_dt.UTC)
    if prev is not None:
        prev_ts = _parse_iso(prev.event_timestamp_utc)
        if now <= prev_ts:
            now = prev_ts + _dt.timedelta(microseconds=1)
    return now.isoformat(timespec="microseconds").replace("+00:00", "Z")


@contextlib.contextmanager
def _ledger_lock(repo_root: str | Path | None = None):
    path = lock_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"could not acquire {path} within {LOCK_TIMEOUT_SECONDS}s") from None
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _append_row_locked(row: CanonicalTaskStatusRow, repo_root: str | Path | None = None) -> None:
    path = ledger_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    payload = json.dumps(row.to_json_obj(), sort_keys=True, separators=(",", ":"))
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex[:12]}")
    tmp.write_text(existing + payload + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _base_event_fields(actor: str, session_id: str) -> dict[str, object]:
    now = _monotonic_event_timestamp()
    return {
        "event_timestamp_utc": now,
        "event_actor": actor,
        "written_at_utc": now,
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
        "session_id": session_id,
    }


def _assert_transition(prev: CanonicalTaskStatusRow, new_status: str) -> None:
    allowed = VALID_TRANSITIONS.get(prev.status, frozenset())
    if new_status not in allowed:
        raise CanonicalTaskStatusInvalidTransitionError(
            f"invalid canonical task-status transition for {prev.task_id}: "
            f"{prev.status} -> {new_status}; allowed={sorted(allowed)}"
        )


def register_task(
    task_id: str,
    source_design_memo: str | Path,
    title: str,
    owner: str,
    predicted_cost_usd: float | None = None,
    predicted_delta_s_band: tuple[float, float] | None = None,
    *,
    actor: str,
    session_id: str,
    repo_root: str | Path | None = None,
    notes: str = "",
) -> CanonicalTaskStatusRow:
    """Register a pending task idempotently."""

    with _ledger_lock(repo_root):
        existing = latest_status_by_task_id(task_id, repo_root)
        if existing is not None:
            return existing
        row = CanonicalTaskStatusRow(
            task_id=task_id,
            source_design_memo=str(source_design_memo),
            title=title,
            status="pending",
            owner=owner,
            predicted_cost_usd=predicted_cost_usd,
            predicted_delta_s_band=predicted_delta_s_band,
            test_status="pending",
            event_type="registered",
            event_notes=notes,
            **_base_event_fields(actor, session_id),
        )
        _append_row_locked(row, repo_root)
        return row


def update_status(
    task_id: str,
    new_status: Literal["pending", "in_progress", "completed", "blocked", "deferred", "cancelled"],
    *,
    actor: str,
    session_id: str,
    notes: str = "",
    commit_shas: tuple[str, ...] = (),
    test_status: Literal["green", "red", "n_a", "pending"] = "pending",
    blockers: tuple[str, ...] = (),
    actual_delta_s: float | None = None,
    repo_root: str | Path | None = None,
) -> CanonicalTaskStatusRow:
    """Append a status-change row after validating the state machine."""

    with _ledger_lock(repo_root):
        prev = latest_status_by_task_id(task_id, repo_root)
        if prev is None:
            raise KeyError(f"unknown canonical task_id: {task_id}")
        _assert_transition(prev, new_status)
        now = _monotonic_event_timestamp(prev)
        event_type = {
            "completed": "completion",
            "blocked": "blocked",
            "cancelled": "cancelled",
        }.get(new_status, "status_change")
        started_at = prev.started_at_utc
        if new_status == "in_progress" and started_at is None:
            started_at = now
        completed_at = now if new_status == "completed" else prev.completed_at_utc
        row = CanonicalTaskStatusRow(
            task_id=prev.task_id,
            source_design_memo=prev.source_design_memo,
            title=prev.title,
            status=new_status,
            owner=prev.owner,
            predicted_cost_usd=prev.predicted_cost_usd,
            predicted_delta_s_band=prev.predicted_delta_s_band,
            actual_delta_s=actual_delta_s,
            commit_shas=tuple(commit_shas) or prev.commit_shas,
            test_status=test_status,
            blockers=tuple(blockers),
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            event_type=event_type,
            event_notes=notes,
            event_timestamp_utc=now,
            event_actor=actor,
            session_id=session_id,
            written_at_utc=now,
            written_pid=os.getpid(),
            written_host=socket.gethostname(),
        )
        _append_row_locked(row, repo_root)
        return row


def append_note(
    task_id: str,
    notes: str,
    *,
    actor: str,
    session_id: str,
    repo_root: str | Path | None = None,
) -> CanonicalTaskStatusRow:
    """Append an audit note without changing current status."""

    with _ledger_lock(repo_root):
        prev = latest_status_by_task_id(task_id, repo_root)
        if prev is None:
            raise KeyError(f"unknown canonical task_id: {task_id}")
        event_notes = notes
        if prev.actual_delta_s is not None and "[empirical:" not in event_notes:
            event_notes = f"{notes}; carried_forward_empirical_evidence: {prev.event_notes}"
        event_ts = _monotonic_event_timestamp(prev)
        row = CanonicalTaskStatusRow(
            task_id=prev.task_id,
            source_design_memo=prev.source_design_memo,
            title=prev.title,
            status=prev.status,
            owner=prev.owner,
            predicted_cost_usd=prev.predicted_cost_usd,
            predicted_delta_s_band=prev.predicted_delta_s_band,
            actual_delta_s=prev.actual_delta_s,
            commit_shas=prev.commit_shas,
            test_status=prev.test_status,
            blockers=prev.blockers,
            started_at_utc=prev.started_at_utc,
            completed_at_utc=prev.completed_at_utc,
            event_type="note",
            event_notes=event_notes,
            event_timestamp_utc=event_ts,
            event_actor=actor,
            written_at_utc=event_ts,
            written_pid=os.getpid(),
            written_host=socket.gethostname(),
            session_id=session_id,
        )
        _append_row_locked(row, repo_root)
        return row


def validate_ledger(repo_root: str | Path | None = None) -> int:
    """Strictly validate the ledger and return row count."""

    return len(load_canonical_task_status_strict(repo_root))
