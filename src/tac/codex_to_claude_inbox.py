# SPDX-License-Identifier: MIT
"""Canonical Codex to Claude inbox ledger.

The inbox is an append-only JSONL channel for Codex questions, relays, Claude
answers, operator-default invocations, acknowledgements, and withdrawals. It is
intentionally small and boring: one fcntl-locked writer, strict JSONL loading,
latest-row-wins status derivation, and no in-place mutation.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
INBOX_SCHEMA_VERSION = "codex_to_claude_inbox_v1_20260518"
INBOX_PATH = REPO_ROOT / ".omx" / "state" / "codex_to_claude_inbox.jsonl"
INBOX_LOCK = INBOX_PATH.with_suffix(INBOX_PATH.suffix + ".lock")

LOCK_TIMEOUT_SECONDS = 30

EVENT_QUESTION = "question"
EVENT_RELAY = "relay"
EVENT_ANSWER = "answer"
EVENT_ACK = "ack"
EVENT_OPERATOR_DEFAULT_INVOKED = "operator_default_invoked"
EVENT_WITHDRAW = "withdraw"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_QUESTION,
        EVENT_RELAY,
        EVENT_ANSWER,
        EVENT_ACK,
        EVENT_OPERATOR_DEFAULT_INVOKED,
        EVENT_WITHDRAW,
    }
)

STATUS_OPEN = "open"
STATUS_ANSWERED = "answered"
STATUS_OPERATOR_DEFAULT_INVOKED = "operator_default_invoked"
STATUS_STALE = "stale"
STATUS_WITHDRAWN = "withdrawn"

VALID_STATUSES = frozenset(
    {
        STATUS_OPEN,
        STATUS_ANSWERED,
        STATUS_OPERATOR_DEFAULT_INVOKED,
        STATUS_STALE,
        STATUS_WITHDRAWN,
    }
)
TERMINAL_STATUSES = frozenset(
    {
        STATUS_ANSWERED,
        STATUS_OPERATOR_DEFAULT_INVOKED,
        STATUS_WITHDRAWN,
        STATUS_STALE,
    }
)


class InboxRowValidationError(ValueError):
    """Raised when an inbox row violates the canonical schema."""


class InboxRowCorruptError(RuntimeError):
    """Raised when the inbox JSONL file cannot be parsed safely."""


_lock_depth_tls = threading.local()


def _get_lock_depth() -> int:
    return int(getattr(_lock_depth_tls, "depth", 0))


def _set_lock_depth(value: int) -> None:
    _lock_depth_tls.depth = int(value)


def _inbox_lock_held() -> bool:
    return _get_lock_depth() > 0


def _resolve_lock_path(path: Path | None, lock_path: Path | None) -> Path:
    if lock_path is not None:
        return Path(lock_path)
    if path is not None:
        inbox_path = Path(path)
        return inbox_path.with_suffix(inbox_path.suffix + ".lock")
    return INBOX_LOCK


@contextlib.contextmanager
def _inbox_lock(lock_path: Path | None = None):
    path = Path(lock_path or INBOX_LOCK)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")  # noqa: SIM115 - caller owns lock lifetime.
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                _set_lock_depth(_get_lock_depth() + 1)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Could not acquire {path} within {LOCK_TIMEOUT_SECONDS}s") from None
                time.sleep(0.05)
        yield
    finally:
        if _get_lock_depth() > 0:
            _set_lock_depth(_get_lock_depth() - 1)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise InboxRowValidationError("timestamp must be a non-empty string")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise InboxRowValidationError(f"invalid UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise InboxRowValidationError(f"timestamp must include timezone: {value!r}")
    return parsed.astimezone(dt.UTC)


def _new_event_id(prefix: str) -> str:
    return f"{prefix}_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"


def _validate_token(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InboxRowValidationError(f"{name} must be a non-empty string")
    text = value.strip()
    if "\n" in text or "\r" in text:
        raise InboxRowValidationError(f"{name} cannot contain newlines")
    return text


def _validate_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InboxRowValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _normalize_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _target_event_id(row: dict[str, Any]) -> str:
    response_to = row.get("response_to_event_id")
    if isinstance(response_to, str) and response_to.strip():
        return response_to.strip()
    return str(row.get("event_id", "")).strip()


def _reserved_fields() -> set[str]:
    return {
        "schema_version",
        "event_id",
        "event_type",
        "status",
        "asked_at_utc",
        "relay_at_utc",
        "answered_at_utc",
        "ack_at_utc",
        "withdrawn_at_utc",
        "default_invoked_at_utc",
        "blocking_task_id",
        "question_text",
        "relay_text",
        "answer_text",
        "ack_text",
        "withdraw_reason",
        "default_used",
        "context_pointers",
        "suggested_options",
        "codex_default_if_no_response",
        "response_deadline_utc",
        "response_to_event_id",
        "answer_memo_path",
        "agent",
        "subagent_id",
        "session_id",
        "written_at_utc",
        "written_pid",
        "written_host",
    }


def _base_row(
    *,
    event_type: str,
    status: str,
    event_id: str | None = None,
    agent: str,
    subagent_id: str | None,
    session_id: str | None,
    prefix: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    if event_type not in VALID_EVENT_TYPES:
        raise InboxRowValidationError(f"invalid event_type: {event_type!r}")
    if status not in VALID_STATUSES:
        raise InboxRowValidationError(f"invalid status: {status!r}")
    collisions = sorted(_reserved_fields().intersection(extra))
    if collisions:
        raise InboxRowValidationError(f"extra field(s) collide with reserved schema fields: {collisions}")
    row = {
        "schema_version": INBOX_SCHEMA_VERSION,
        "event_id": _validate_token("event_id", _new_event_id(prefix) if event_id is None else event_id),
        "event_type": event_type,
        "status": status,
        "agent": _validate_token("agent", agent),
        "subagent_id": subagent_id,
        "session_id": session_id,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    row.update(extra)
    return row


def _validate_row(row: dict[str, Any]) -> None:
    if not isinstance(row, dict):
        raise InboxRowValidationError("row must be a JSON object")
    if row.get("schema_version") != INBOX_SCHEMA_VERSION:
        raise InboxRowValidationError("schema_version mismatch")
    _validate_token("event_id", str(row.get("event_id", "")))
    event_type = row.get("event_type")
    if event_type not in VALID_EVENT_TYPES:
        raise InboxRowValidationError(f"invalid event_type: {event_type!r}")
    status = row.get("status")
    if status not in VALID_STATUSES:
        raise InboxRowValidationError(f"invalid status: {status!r}")
    _parse_utc(str(row.get("written_at_utc", "")))
    if event_type == EVENT_QUESTION:
        _validate_text("question_text", str(row.get("question_text", "")))
        if row.get("response_deadline_utc"):
            _parse_utc(str(row["response_deadline_utc"]))
    if event_type == EVENT_RELAY:
        _validate_text("relay_text", str(row.get("relay_text", "")))
    if event_type == EVENT_ANSWER:
        _validate_token("response_to_event_id", str(row.get("response_to_event_id", "")))
        _validate_text("answer_text", str(row.get("answer_text", "")))
    if event_type == EVENT_OPERATOR_DEFAULT_INVOKED:
        _validate_token("response_to_event_id", str(row.get("response_to_event_id", "")))
        _validate_text("default_used", str(row.get("default_used", "")))
    if event_type == EVENT_ACK:
        _validate_token("response_to_event_id", str(row.get("response_to_event_id", "")))
        _validate_text("ack_text", str(row.get("ack_text", "")))
    if event_type == EVENT_WITHDRAW:
        _validate_token("response_to_event_id", str(row.get("response_to_event_id", "")))
        _validate_text("withdraw_reason", str(row.get("withdraw_reason", "")))


def _json_line(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def load_inbox_strict(path: Path | None = None) -> list[dict[str, Any]]:
    inbox_path = Path(path or INBOX_PATH)
    if not inbox_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    current_lineno: int | str = "?"
    try:
        lines = inbox_path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            current_lineno = lineno
            if not line.strip():
                continue
            row = json.loads(line)
            _validate_row(row)
            rows.append(row)
    except (json.JSONDecodeError, InboxRowValidationError) as exc:
        corrupt_path = inbox_path.with_name(
            f"{inbox_path.name}.corrupt.{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"
        )
        try:
            shutil.move(str(inbox_path), str(corrupt_path))
        except OSError:
            corrupt_path = inbox_path
        raise InboxRowCorruptError(
            f"inbox JSONL at {inbox_path} is corrupt near line {current_lineno}; "
            f"quarantined to {corrupt_path}: {exc}"
        ) from exc
    return rows


def load_inbox(path: Path | None = None) -> list[dict[str, Any]]:
    return load_inbox_strict(path)


def _append_event_locked(
    row: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    inbox_path = Path(path or INBOX_PATH)
    lock = _resolve_lock_path(path, lock_path)
    _validate_row(row)
    if _inbox_lock_held():
        return _append_event_unlocked(row, path=inbox_path)
    with _inbox_lock(lock):
        return _append_event_unlocked(row, path=inbox_path)


def _append_event_unlocked(row: dict[str, Any], *, path: Path) -> dict[str, Any]:
    inbox_path = Path(path)
    existing = inbox_path.read_text(encoding="utf-8") if inbox_path.exists() else ""
    inbox_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = inbox_path.with_suffix(inbox_path.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        payload = existing
        if payload and not payload.endswith("\n"):
            payload += "\n"
        payload += _json_line(row)
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, inbox_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return row


def latest_status_by_event_id(*, path: Path | None = None) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for row in load_inbox_strict(path):
        statuses[_target_event_id(row)] = str(row["status"])
    return statuses


def query_by_event_id(event_id: str, *, path: Path | None = None) -> list[dict[str, Any]]:
    target = _validate_token("event_id", event_id)
    return [
        row
        for row in load_inbox_strict(path)
        if row.get("event_id") == target or row.get("response_to_event_id") == target
    ]


def _latest_rows_by_target(*, path: Path | None = None) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_inbox_strict(path):
        latest[_target_event_id(row)] = row
    return latest


def query_open_questions(*, asked_by: str = "codex", path: Path | None = None) -> list[dict[str, Any]]:
    rows = load_inbox_strict(path)
    latest = latest_status_by_event_id(path=path)
    questions = [
        row
        for row in rows
        if row.get("event_type") == EVENT_QUESTION
        and row.get("agent") == asked_by
        and latest.get(str(row["event_id"])) == STATUS_OPEN
    ]
    return questions


def query_open_questions_for_claude(*, path: Path | None = None) -> list[dict[str, Any]]:
    return query_open_questions(asked_by="codex", path=path)


def query_unread_relays(*, since_utc: str, path: Path | None = None) -> list[dict[str, Any]]:
    cutoff = _parse_utc(since_utc)
    rows = load_inbox_strict(path)
    acked = {
        str(row.get("response_to_event_id"))
        for row in rows
        if row.get("event_type") == EVENT_ACK and row.get("response_to_event_id")
    }
    relays = []
    for row in rows:
        if row.get("event_type") != EVENT_RELAY:
            continue
        written = _parse_utc(str(row["written_at_utc"]))
        if written >= cutoff and row.get("event_id") not in acked:
            relays.append(row)
    return relays


def append_inbox_question(
    *,
    question_text: str,
    blocking_task_id: str | None = None,
    context_pointers: tuple[str, ...] | list[str] = (),
    suggested_options: tuple[str, ...] | list[str] = (),
    codex_default_if_no_response: str | None = None,
    response_deadline_utc: str | None = None,
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    event_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if response_deadline_utc:
        _parse_utc(response_deadline_utc)
    row = _base_row(
        event_type=EVENT_QUESTION,
        status=STATUS_OPEN,
        event_id=event_id,
        agent=agent,
        subagent_id=subagent_id,
        session_id=session_id,
        prefix="question",
        extra=extra,
    )
    row.update(
        {
            "asked_at_utc": row["written_at_utc"],
            "blocking_task_id": blocking_task_id,
            "question_text": _validate_text("question_text", question_text),
            "context_pointers": list(_normalize_tuple(context_pointers)),
            "suggested_options": list(_normalize_tuple(suggested_options)),
            "codex_default_if_no_response": (
                _validate_text("codex_default_if_no_response", codex_default_if_no_response)
                if codex_default_if_no_response is not None
                else None
            ),
            "response_deadline_utc": response_deadline_utc,
        }
    )
    return _append_event_locked(row, path=path, lock_path=lock_path)


def append_inbox_relay(
    *,
    relay_text: str,
    context_pointers: tuple[str, ...] | list[str] = (),
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    event_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    row = _base_row(
        event_type=EVENT_RELAY,
        status=STATUS_ANSWERED,
        event_id=event_id,
        agent=agent,
        subagent_id=subagent_id,
        session_id=session_id,
        prefix="relay",
        extra=extra,
    )
    row.update(
        {
            "relay_at_utc": row["written_at_utc"],
            "relay_text": _validate_text("relay_text", relay_text),
            "context_pointers": list(_normalize_tuple(context_pointers)),
        }
    )
    return _append_event_locked(row, path=path, lock_path=lock_path)


def _require_open_question(response_to_event_id: str, *, path: Path | None = None) -> dict[str, Any]:
    target = _validate_token("response_to_event_id", response_to_event_id)
    rows = query_by_event_id(target, path=path)
    question = next((row for row in rows if row.get("event_type") == EVENT_QUESTION), None)
    if question is None:
        raise InboxRowValidationError(f"question not found: {target}")
    if latest_status_by_event_id(path=path).get(target) != STATUS_OPEN:
        raise InboxRowValidationError(f"question is not open: {target}")
    return question


def append_inbox_answer(
    *,
    response_to_event_id: str,
    answer_text: str,
    answer_memo_path: str | None = None,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    event_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    with _inbox_lock(_resolve_lock_path(path, lock_path)):
        _require_open_question(response_to_event_id, path=path)
        row = _base_row(
            event_type=EVENT_ANSWER,
            status=STATUS_ANSWERED,
            event_id=event_id,
            agent=agent,
            subagent_id=subagent_id,
            session_id=session_id,
            prefix="answer",
            extra=extra,
        )
        row.update(
            {
                "answered_at_utc": row["written_at_utc"],
                "response_to_event_id": _validate_token("response_to_event_id", response_to_event_id),
                "answer_text": _validate_text("answer_text", answer_text),
                "answer_memo_path": answer_memo_path,
            }
        )
        return _append_event_locked(row, path=path, lock_path=lock_path)


def append_inbox_operator_default_invoked(
    *,
    response_to_event_id: str,
    default_used: str,
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    event_id: str | None = None,
    now_utc: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    with _inbox_lock(_resolve_lock_path(path, lock_path)):
        question = _require_open_question(response_to_event_id, path=path)
        declared_default = str(question.get("codex_default_if_no_response") or "").strip()
        if not declared_default:
            raise InboxRowValidationError("cannot invoke default without codex_default_if_no_response")
        used = _validate_text("default_used", default_used)
        if used != declared_default:
            raise InboxRowValidationError("default_used must match codex_default_if_no_response")
        deadline_raw = question.get("response_deadline_utc")
        if not deadline_raw:
            raise InboxRowValidationError("cannot invoke default without response_deadline_utc")
        deadline = _parse_utc(str(deadline_raw))
        now = _parse_utc(now_utc) if now_utc else _utc_now()
        if now < deadline:
            raise InboxRowValidationError("cannot invoke operator default before response deadline")
        row = _base_row(
            event_type=EVENT_OPERATOR_DEFAULT_INVOKED,
            status=STATUS_OPERATOR_DEFAULT_INVOKED,
            event_id=event_id,
            agent=agent,
            subagent_id=subagent_id,
            session_id=session_id,
            prefix="default",
            extra=extra,
        )
        row.update(
            {
                "default_invoked_at_utc": row["written_at_utc"],
                "response_to_event_id": _validate_token("response_to_event_id", response_to_event_id),
                "default_used": used,
            }
        )
        return _append_event_locked(row, path=path, lock_path=lock_path)


def append_inbox_ack(
    *,
    response_to_event_id: str,
    ack_text: str,
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    event_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    target = _validate_token("response_to_event_id", response_to_event_id)
    with _inbox_lock(_resolve_lock_path(path, lock_path)):
        current = latest_status_by_event_id(path=path).get(target)
        if current is None:
            raise InboxRowValidationError(f"event not found: {target}")
        row = _base_row(
            event_type=EVENT_ACK,
            status=current,
            event_id=event_id,
            agent=agent,
            subagent_id=subagent_id,
            session_id=session_id,
            prefix="ack",
            extra=extra,
        )
        row.update(
            {
                "ack_at_utc": row["written_at_utc"],
                "response_to_event_id": target,
                "ack_text": _validate_text("ack_text", ack_text),
            }
        )
        return _append_event_locked(row, path=path, lock_path=lock_path)


def append_inbox_withdraw(
    *,
    response_to_event_id: str,
    reason: str,
    agent: str = "codex",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    event_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    with _inbox_lock(_resolve_lock_path(path, lock_path)):
        _require_open_question(response_to_event_id, path=path)
        row = _base_row(
            event_type=EVENT_WITHDRAW,
            status=STATUS_WITHDRAWN,
            event_id=event_id,
            agent=agent,
            subagent_id=subagent_id,
            session_id=session_id,
            prefix="withdraw",
            extra=extra,
        )
        row.update(
            {
                "withdrawn_at_utc": row["written_at_utc"],
                "response_to_event_id": _validate_token("response_to_event_id", response_to_event_id),
                "withdraw_reason": _validate_text("withdraw_reason", reason),
            }
        )
        return _append_event_locked(row, path=path, lock_path=lock_path)


def inbox_summary(*, path: Path | None = None, since_utc: str | None = None) -> dict[str, Any]:
    rows = load_inbox_strict(path)
    statuses = latest_status_by_event_id(path=path)
    open_questions = query_open_questions_for_claude(path=path)
    now = _utc_now()
    ages: list[float] = []
    expired_open = []
    for row in open_questions:
        asked = _parse_utc(str(row["asked_at_utc"]))
        ages.append((now - asked).total_seconds() / 3600.0)
        deadline = row.get("response_deadline_utc")
        if deadline and now >= _parse_utc(str(deadline)):
            expired_open.append(row["event_id"])
    relays = query_unread_relays(since_utc=since_utc, path=path) if since_utc else [
        row for row in rows if row.get("event_type") == EVENT_RELAY
    ]
    return {
        "schema_version": INBOX_SCHEMA_VERSION,
        "path": str(path or INBOX_PATH),
        "row_count": len(rows),
        "event_count": len(statuses),
        "open_questions_count": len(open_questions),
        "open_questions_oldest_age_hours": round(max(ages), 4) if ages else 0.0,
        "expired_open_questions_count": len(expired_open),
        "expired_open_question_event_ids": expired_open,
        "relays_count": len(relays),
        "latest_status_by_event_id": statuses,
    }


__all__ = [
    "EVENT_ACK",
    "EVENT_ANSWER",
    "EVENT_OPERATOR_DEFAULT_INVOKED",
    "EVENT_QUESTION",
    "EVENT_RELAY",
    "EVENT_WITHDRAW",
    "INBOX_LOCK",
    "INBOX_PATH",
    "INBOX_SCHEMA_VERSION",
    "STATUS_ANSWERED",
    "STATUS_OPEN",
    "STATUS_OPERATOR_DEFAULT_INVOKED",
    "STATUS_STALE",
    "STATUS_WITHDRAWN",
    "TERMINAL_STATUSES",
    "VALID_EVENT_TYPES",
    "VALID_STATUSES",
    "InboxRowCorruptError",
    "InboxRowValidationError",
    "append_inbox_ack",
    "append_inbox_answer",
    "append_inbox_operator_default_invoked",
    "append_inbox_question",
    "append_inbox_relay",
    "append_inbox_withdraw",
    "inbox_summary",
    "latest_status_by_event_id",
    "load_inbox",
    "load_inbox_strict",
    "query_by_event_id",
    "query_open_questions",
    "query_open_questions_for_claude",
    "query_unread_relays",
]
