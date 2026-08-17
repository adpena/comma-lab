# SPDX-License-Identifier: MIT
"""Strict loader for the canonical task-status ledger."""

from __future__ import annotations

import datetime as _dt
import json
import os
import warnings
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


def _validate_history(
    rows: Iterable[CanonicalTaskStatusRow],
) -> dict[str, str]:
    """Return ``{task_id: reason}`` for every task whose history is UNREADABLE.

    PER-TASK ISOLATION, not file-level death (measured cure, 2026-08-16).  One
    orphan row -- a lone ``completion`` for a task never registered -- took the
    ENTIRE 548-row ledger down: ``load_canonical_task_status_strict`` and
    ``canonical_task_status_violations`` both refused, so every writer in the
    fleet was blocked for ~7h and a live arm could not file its rows at all.
    A single unreadable task had been rendered as total death.

    Strictness is PRESERVED, not traded away: an unreadable task's rows are
    EXCLUDED from the served set (no consumer can act on a broken history) and
    the reason is reported.  What changes is blast radius -- the other 547 rows
    are fine and must stay usable.  Unparseable JSON is a different failure and
    still raises + quarantines in the caller: that is file-level corruption,
    where partial service really would be unsafe.

    Third instance of one law today (see ``tac.process_liveness``): a state that
    could not be READ must not be encoded as a negative RESULT, and here it must
    not be encoded as everyone else's outage either.
    """

    latest: dict[str, CanonicalTaskStatusRow] = {}
    seen_registration: set[str] = set()
    unreadable: dict[str, str] = {}
    for row in rows:
        if row.task_id in unreadable:
            continue  # history already broken; later rows cannot repair it
        prev = latest.get(row.task_id)
        reason: str | None = None
        if row.event_type == "registered":
            if row.task_id in seen_registration:
                reason = "duplicate registration"
            elif row.status != "pending":
                reason = f"registration must have pending status, got {row.status!r}"
            else:
                seen_registration.add(row.task_id)
        elif prev is None:
            reason = "non-registration event for unknown task_id"
        elif row.event_type != "note":
            allowed = VALID_TRANSITIONS.get(prev.status, frozenset())
            if row.status not in allowed:
                reason = f"invalid transition: {prev.status} -> {row.status}"
        elif row.status != prev.status:
            reason = f"note event changed status {prev.status} -> {row.status}"
        if reason is not None:
            unreadable[row.task_id] = reason
            latest.pop(row.task_id, None)
            continue
        latest[row.task_id] = row
    return unreadable


def _parse_rows(
    path: Path,
) -> tuple[list[CanonicalTaskStatusRow], tuple[int, Exception] | None, dict[str, str]]:
    """Parse every row, SPLITTING file corruption from row-contract violations.

    Returns ``(rows, file_corruption, contract_violations)`` where
    ``file_corruption`` is ``(line_number, exception)`` or ``None``, and
    ``contract_violations`` maps ``task_id -> reason``.

    THE TWO FAILURE CLASSES THIS FUNCTION USED TO CONFLATE, and what it cost
    (measured 2026-08-17, by a live arm mid-run):

    A row carrying ``actual_delta_s`` without its ``[empirical:<path>]`` note
    raises ``ValueError`` from ``contract.py:280``.  The old code returned that
    as "first corrupt line", the caller read it as file corruption, and
    QUARANTINED -- i.e. MOVED AWAY -- the entire shared ledger.  An arm found the
    file simply gone and had to restore it by hand.  ONE arm's contract
    violation deleted shared state for the WHOLE FLEET.

    They are not the same failure and must not share a response:

    * FILE corruption -- the line is not JSON, or is not an object.  The byte
      stream is untrustworthy; serving a truncated prefix would be worse than
      refusing.  Quarantine stays.
    * ROW-CONTRACT violation -- the line IS a well-formed JSON object and one
      field RELATIONSHIP the schema polices does not hold.  Every other row is
      exactly as valid as it was a moment ago.  Isolating the offending TASK is
      proportionate; moving the file is not.

    Sister of the per-task isolation cure one layer up in ``_validate_history``
    (2026-08-16, same file).  That fix bounded the blast radius of a broken
    HISTORY and left a broken ROW at "the whole file" -- a defect cured at one
    site and left uncured at the second, which is its own recurring genus here.

    A contract violation marks the whole TASK unreadable rather than dropping
    just the one row: dropping a single row can make the rest of that task's
    history incoherent (drop a ``registered`` row and every later row looks
    orphaned), which would manufacture a second, fictional defect.

    The offending exception is RETURNED, not swallowed, so callers can chain it
    with ``from``: the reason a line failed is the whole diagnostic, and
    dropping it would make the report a record of its own blindness.
    """

    rows: list[CanonicalTaskStatusRow] = []
    contract_violations: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            return rows, (line_number, exc), contract_violations
        if not isinstance(obj, dict):
            return rows, (line_number, ValueError("row is not an object")), contract_violations
        try:
            rows.append(CanonicalTaskStatusRow.from_json_obj(obj))
        except (ValueError, TypeError) as exc:
            # A well-formed object that fails the row contract. Key by task_id so
            # the offender is NAMED; fall back to the line when the id itself is
            # unusable, so the row is still reported rather than silently dropped.
            raw_id = obj.get("task_id")
            key = raw_id if isinstance(raw_id, str) and raw_id.strip() else f"<line {line_number}>"
            contract_violations.setdefault(key, f"row contract violated at line {line_number}: {exc}")
    return rows, None, contract_violations


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
    rows, file_corruption, contract_violations = _parse_rows(path)
    if file_corruption is not None:
        corrupt_line, corrupt_exc = file_corruption
        quarantine = _quarantine_corrupt_file(path)
        raise CanonicalTaskStatusCorruptError(
            f"corrupt canonical task-status row {corrupt_line}; quarantined to {quarantine}"
        ) from corrupt_exc
    # History errors first, then contract violations OVERWRITE them: a row that
    # failed its contract was never appended, so any history gap it leaves behind
    # is a CONSEQUENCE, and reporting the consequence would hide the cause.
    unreadable = _validate_history(rows)
    unreadable.update(contract_violations)
    if unreadable:
        # LOUD, never silent: a skipped task is debt somebody must clear, and a
        # quiet skip is how "the ledger looks fine" becomes a lie (vacuity==pass).
        detail = "; ".join(f"{tid}: {why}" for tid, why in sorted(unreadable.items()))
        warnings.warn(
            "canonical_task_status: "
            f"{len(unreadable)} task(s) UNREADABLE and EXCLUDED from the served "
            f"rows -- their history is broken and cannot be served, but the "
            f"remaining {len(rows) - sum(1 for r in rows if r.task_id in unreadable)} "
            f"rows load and writes proceed. Repair by appending a corrected "
            f"lifecycle (the ledger is append-only). {detail}",
            stacklevel=2,
        )
        rows = [row for row in rows if row.task_id not in unreadable]
    return rows


def unreadable_task_ids(repo_root: str | Path | None = None) -> dict[str, str]:
    """``{task_id: reason}`` for tasks excluded by the loader.

    The queryable half of the isolation cure: callers that need to know WHICH
    task is broken (a repair tool, an audit) ask here instead of parsing a
    warning string.

    Raises ``CanonicalTaskStatusCorruptError`` on an unparseable line WITHOUT
    quarantining -- a read-only audit must not rename the file it audits, and
    it must not answer from a truncated read either.  Skipping the bad line
    would be actively misleading: dropping a ``registered`` row makes every
    later row for that task look orphaned, so a swallow here would INVENT
    unreadable tasks.  Refusing is the honest answer (vacuity==pass).
    """

    path = ledger_path(repo_root)
    if not path.exists():
        return {}
    rows, file_corruption, contract_violations = _parse_rows(path)
    if file_corruption is not None:
        corrupt_line, corrupt_exc = file_corruption
        raise CanonicalTaskStatusCorruptError(
            f"corrupt canonical task-status row {corrupt_line}; cannot report "
            f"unreadable tasks from a truncated read (not quarantined -- this "
            f"is a read-only audit; call the strict loader to quarantine)"
        ) from corrupt_exc
    unreadable = _validate_history(rows)
    unreadable.update(contract_violations)
    return unreadable


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

