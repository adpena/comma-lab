# SPDX-License-Identifier: MIT
"""Canonical Modal call_id ledger — fcntl-locked JSONL append-only audit trail.

Operator directive 2026-05-15: *"perhaps canonicalize and record the call_id
in a log or archive or in review_tracker duckdb or similar"* — close the
orphan-state on the per-dispatch ``modal_call_id.txt`` / ``modal_metadata.json``
sentinels by adding a SINGLE QUERYABLE ledger that captures every Modal
``.spawn()`` call_id at the moment of dispatch and every subsequent
harvest/failure/staleness event as APPEND-ONLY rows.

Why a new canonical helper?
───────────────────────────
Pre-landing, Modal call_ids were captured at SCATTERED surfaces:

- ``experiments/results/lane_*_modal/modal_call_id.txt`` (per-dispatch text)
- ``experiments/results/lane_*_modal/modal_metadata.json`` (per-dispatch JSON)
- ``.omx/state/active_lane_dispatch_claims.md`` (markdown ledger w/ embedded ids)
- ``tools/harvest_modal_calls.py`` (crawler that re-discovers them)

There was NO single source-of-truth for "show me every call_id we have ever
dispatched in chronological order, including outcome". Per CLAUDE.md "Modal
``.spawn()`` HARVEST OR LOSE" non-negotiable: every dispatch via
``modal_train_lane.py`` MUST be followed by a scheduled harvest within 24h.
The harvester relies on per-dispatch metadata files whose discovery cost is
O(N_dispatches) ``glob`` calls — a fragile primary index for an audit-grade
ledger.

This module is the canonical primary index. It mirrors:

- ``tac.deploy.lightning.active_jobs_state`` — fcntl-locked JSONL state helper
- ``tac.deploy.azure.active_vms_state`` — sister pattern for Azure
- ``tac.continual_learning.posterior_update_locked`` — Catalog #128
- ``tools/subagent_checkpoint.py`` — JSONL append-only with fcntl LOCK_EX
  (Catalog #206 anchor)

Schema (one event per JSONL row)
────────────────────────────────
Every row is one event in a call_id's lifecycle. The ledger is APPEND-ONLY
per CLAUDE.md "HISTORICAL_PROVENANCE" classification + Catalog #110 / #113 /
#132 (locked writes preserve deletions) — rows are NEVER mutated; new events
become NEW rows referencing the same ``call_id``. The full lifecycle of a
call_id is reconstructable by querying ``query_by_call_id(call_id)`` which
returns chronological rows.

Required event_types::

    - "dispatched"          — written immediately after fn.spawn() returns
    - "harvested"           — written when harvest succeeds (rc=0 + result)
    - "failed"              — harvest succeeded but call returned non-zero rc
    - "stale"               — call_id expired the 24h Modal result cache TTL
    - "manually_terminated" — operator killed via dashboard / modal app stop

Schema fields per row::

    {
        "schema_version": 1,
        "event_type": "dispatched" | "harvested" | "failed" | "stale" | ...,
        "call_id": "fc-01KRN...",
        "lane_id": "lane_d4_wyner_ziv_frame_0_substrate_20260514",
        "label": "substrate_d4_..._modal_t4_dispatch_20260515T135915Z__smoke",
        "platform": "modal",
        "gpu": "T4",
        "expected_cost_usd": 0.30,
        "expected_axis": "cuda",
        "recipe": "substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch",
        "dispatched_at_utc": "2026-05-15T13:59:15Z",
        "harvested_at_utc": null,
        "status": "dispatched",
        "rc": null,
        "elapsed_seconds": null,
        "cost_actual_usd": null,
        "score": null,
        "score_axis": null,
        "archive_sha256": null,
        "archive_bytes": null,
        "evidence_grade": null,
        "max_seconds": 5400,
        "mounted_code_git_head": "fe3faa9b1d8408055928b695deddfb9527e5fcf8",
        "agent": "claude" | "codex" | "operator",
        "subagent_id": null,
        "session_id": null,
        "written_at_utc": "2026-05-15T13:59:15.123Z",
        "written_pid": 12345,
        "written_host": "macbook-air",
    }

Path discipline
───────────────
- ``MODAL_CALL_ID_LEDGER_PATH`` = ``.omx/state/modal_call_id_ledger.jsonl``
  COMMITTED per HISTORICAL_PROVENANCE classification (Catalog #113).
- The lock file ``.lock`` is gitignored (LIVE_STATE).
- ``.tmp.<uuid12>`` files are gitignored (LIVE_STATE).
- Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact (the
  transient-evidence trap)" — the ledger lives at ``.omx/state/`` under the
  repo root.

Bare writes are FORBIDDEN
─────────────────────────
Per CLAUDE.md Catalog #131 (``check_no_bare_writes_to_shared_state``) every
write to ``MODAL_CALL_ID_LEDGER_PATH`` MUST acquire ``fcntl.flock(LOCK_EX)``
on the lock file + use a unique ``.tmp.<uuid12>`` + ``os.replace``. The
public API (``register_dispatched_call_id`` / ``update_call_id_outcome``)
does this; direct ``open(...).write(...)`` outside the canonical helper is
refused by Catalog #131 sister gate (this module's path is registered there).

Memory: feedback_modal_call_id_ledger_canonical_landed_20260515.md.
"""

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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
MODAL_CALL_ID_LEDGER_PATH = REPO_ROOT / ".omx" / "state" / "modal_call_id_ledger.jsonl"
MODAL_CALL_ID_LEDGER_LOCK = MODAL_CALL_ID_LEDGER_PATH.with_suffix(MODAL_CALL_ID_LEDGER_PATH.suffix + ".lock")

# Schema version pinned for forward compatibility.
SCHEMA_VERSION = 1

# Lock acquisition timeout (seconds). Single-row appends are <10ms; 30s is
# generous even under heavy fan-out contention from sibling subagents.
LOCK_TIMEOUT_SECONDS = 30

# Canonical event taxonomy. A call_id may have many events; the latest in
# chronological order is the current state.
EVENT_DISPATCHED = "dispatched"
EVENT_HARVESTED = "harvested"
EVENT_FAILED = "failed"
EVENT_STALE = "stale"
EVENT_MANUALLY_TERMINATED = "manually_terminated"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_DISPATCHED,
        EVENT_HARVESTED,
        EVENT_FAILED,
        EVENT_STALE,
        EVENT_MANUALLY_TERMINATED,
    }
)

# Status tokens. ``status`` echoes the latest event_type for convenience.
STATUS_DISPATCHED = EVENT_DISPATCHED
STATUS_HARVESTED = EVENT_HARVESTED
STATUS_FAILED = EVENT_FAILED
STATUS_STALE = EVENT_STALE
STATUS_MANUALLY_TERMINATED = EVENT_MANUALLY_TERMINATED

VALID_STATUSES = VALID_EVENT_TYPES

# Terminal statuses (no further events expected). A call_id whose latest row
# carries one of these is "harvested or otherwise resolved" for the purpose of
# ``query_unharvested``.
TERMINAL_STATUSES = frozenset({STATUS_HARVESTED, STATUS_FAILED, STATUS_STALE, STATUS_MANUALLY_TERMINATED})


class CallIdLedgerCorruptError(RuntimeError):
    """Raised when the call_id ledger file is corrupt and cannot be safely
    appended to.

    Sister of ``ActiveJobsCorruptError`` (Catalog #138 strict-load discipline).
    The append helpers raise this rather than silently overwriting the bad
    file, which would erase the historical audit trail of every dispatched
    call_id. The corrupt file is QUARANTINED to ``.corrupt.<utc>`` so the
    operator can inspect; the next ``register_dispatched_call_id`` will create
    a fresh empty ledger.

    Repair recipe:
      1. Inspect ``.omx/state/modal_call_id_ledger.jsonl.corrupt.<utc>``
      2. If recoverable (truncated tail, bad row), hand-edit and rename back
      3. If unrecoverable, leave the quarantine file as forensic evidence;
         the next ``register_dispatched_call_id`` will start with an empty file.
    """


# In-process lock-held depth (Catalog #140 sister pattern). Pairs with
# ``_ledger_lock_held()`` so any direct-write helper can refuse calls that
# bypass the canonical locked path.
_ledger_lock_depth: int = 0


def _ledger_lock_held() -> bool:
    """Return True if THIS process is currently inside ``_ledger_lock``."""
    return _ledger_lock_depth > 0


@contextlib.contextmanager
def _ledger_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the ledger lock file.

    Lock is process-advisory (``fcntl.flock`` ``LOCK_EX``); multiple
    processes contending serialize on the lock file. Re-entry within the same
    process is counted (depth > 1); fcntl is only re-acquired on the 0->1
    transition to avoid same-process deadlock.
    """
    global _ledger_lock_depth

    p = lock_path or MODAL_CALL_ID_LEDGER_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    if _ledger_lock_depth > 0:
        _ledger_lock_depth += 1
        try:
            yield None
        finally:
            _ledger_lock_depth -= 1
        return
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"could not acquire {p} within {LOCK_TIMEOUT_SECONDS}s") from None
                time.sleep(0.05)
        _ledger_lock_depth += 1
        try:
            yield fd
        finally:
            _ledger_lock_depth -= 1
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format with seconds precision."""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _quarantine_corrupt_file(path: Path) -> Path:
    """Move ``path`` to ``path.corrupt.<utc>`` for forensic inspection.

    Idempotent: if ``path`` does not exist, this is a no-op and returns
    ``path``. Otherwise, returns the new quarantine path.
    """
    if not path.exists():
        return path
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}")
    counter = 0
    while quarantine.exists():
        counter += 1
        quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}.{counter}")
    os.rename(path, quarantine)
    return quarantine


def load_call_ids(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the call_id ledger or return empty list (LENIENT loader).

    Safe to call without holding the lock — readers see a stable snapshot
    because writers append under the lock + each write is atomic at the
    line level. Concurrent readers may observe a partial trailing line if
    a write is in flight; this loader silently skips malformed lines for
    backward compatibility with consumers that just want a best-effort view.

    LENIENT semantics: malformed JSON lines are SKIPPED with no error. This
    is appropriate for read-only callers (dashboards, query helpers) but
    UNSAFE for mutating callers — see ``load_call_ids_strict`` per Catalog
    #138 strict-load discipline.
    """
    p = path or MODAL_CALL_ID_LEDGER_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_call_ids_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """Strict load for mutating callers — raises CallIdLedgerCorruptError on
    corrupt state.

    MUST be called from inside ``_ledger_lock`` by mutating callers. The
    mutating helpers (``register_dispatched_call_id`` / ``update_call_id_outcome``)
    use this so a malformed ledger is NEVER silently appended to (which would
    corrupt the audit trail). Returns ``[]`` if the path does not exist (the
    empty-ledger bootstrap case is normal and not corruption).

    Raises:
        CallIdLedgerCorruptError: when the file exists and contains any
            malformed JSON line OR any row whose root is not a dict.
    """
    p = path or MODAL_CALL_ID_LEDGER_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise CallIdLedgerCorruptError(f"call_id ledger at {p} could not be read: {exc}") from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CallIdLedgerCorruptError(
                f"call_id ledger at {p} line {lineno} contains invalid JSON: "
                f"{exc}. Mutating writes are refused to preserve the audit "
                "trail. Operator: inspect the file, then either fix it in "
                "place OR move it aside; the next register_dispatched_call_id "
                "will create a fresh empty file."
            ) from exc
        if not isinstance(row, dict):
            raise CallIdLedgerCorruptError(
                f"call_id ledger at {p} line {lineno} has non-dict root "
                f"(type={type(row).__name__}); expected JSON object."
            )
        rows.append(row)
    return rows


def _validate_event_record(record: dict[str, Any]) -> None:
    """Sanity-check a record before append. Raises ValueError on bad input."""
    call_id = record.get("call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("call_id must be a non-empty string")
    if any(c in call_id for c in ("\n", "\t", "\x1f")):
        raise ValueError("call_id must not contain newlines/tabs/0x1f")

    event_type = record.get("event_type")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}, got {event_type!r}")

    status = record.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)!r}, got {status!r}")

    schema_version = record.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {schema_version!r}")


def _append_event_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    quarantine_on_corrupt: bool = True,
) -> dict[str, Any]:
    """Append a single event record under fcntl lock.

    Acquires exclusive fcntl lock on ``lock_path``, then INSIDE the lock:
      1. STRICT-reload ledger from disk (raises ``CallIdLedgerCorruptError``
         on any malformed row — append is refused).
      2. Append ``record`` line to a unique ``.tmp.<uuid12>`` file containing
         all existing rows + the new row.
      3. ``fsync`` + ``os.replace`` for atomic commit.

    Per CLAUDE.md "Locked writes preserve deletions" (Catalog #132): the
    ledger is APPEND-ONLY — existing rows are preserved verbatim; only new
    rows are added.
    """
    _validate_event_record(record)
    p_path = path or MODAL_CALL_ID_LEDGER_PATH
    l_path = lock_path or MODAL_CALL_ID_LEDGER_LOCK

    with _ledger_lock(l_path):
        try:
            rows = load_call_ids_strict(p_path)
        except CallIdLedgerCorruptError as exc:
            if quarantine_on_corrupt:
                quarantine_path = _quarantine_corrupt_file(p_path)
                raise CallIdLedgerCorruptError(
                    f"call_id ledger at {p_path} was corrupt; quarantined to "
                    f"{quarantine_path}. Append refused; operator must repair "
                    "(see CallIdLedgerCorruptError docstring)."
                ) from exc
            raise

        new_rows = [*rows, record]
        _save_ledger(new_rows, p_path)
        return record


def _save_ledger(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomic write — unique tmp + fsync + os.replace.

    Runtime-asserts the caller holds ``_ledger_lock``. Pre-fix this method
    documented "MUST be called inside the lock" but did not enforce it;
    comment-only contracts are FORBIDDEN per CLAUDE.md.
    """
    if not _ledger_lock_held():
        raise RuntimeError(
            "_save_ledger called WITHOUT holding _ledger_lock. This is a "
            "CONCURRENCY BUG: concurrent appends can silently drop rows. "
            "Use _append_event_locked / register_dispatched_call_id / "
            "update_call_id_outcome which own the full lock-load-append-save "
            "cycle."
        )
    p = path or MODAL_CALL_ID_LEDGER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
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


# ─────────────────────────────────────────────────────────────────────────
# Public API — register_dispatched_call_id + update_call_id_outcome
# ─────────────────────────────────────────────────────────────────────────


def register_dispatched_call_id(
    *,
    call_id: str,
    lane_id: str,
    label: str,
    dispatched_at_utc: str | None = None,
    platform: str = "modal",
    gpu: str = "T4",
    expected_cost_usd: float | None = None,
    expected_axis: str = "cuda",
    recipe: str | None = None,
    max_seconds: int | None = None,
    mounted_code_git_head: str | None = None,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append a ``dispatched`` event row immediately after fn.spawn() returns.

    Caller MUST invoke this BEFORE the local entrypoint exits so the canonical
    ledger captures the call_id even if the per-dispatch sentinel files
    (``modal_call_id.txt`` / ``modal_metadata.json``) are not written
    (concurrent crash, sister-subagent edit, etc.).

    Returns the appended record (including server-side fields like
    ``written_at_utc`` / ``written_pid`` / ``written_host``).
    """
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("call_id must be a non-empty string")
    if not isinstance(lane_id, str) or not lane_id.strip():
        raise ValueError("lane_id must be a non-empty string")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_DISPATCHED,
        "call_id": call_id,
        "lane_id": lane_id,
        "label": label,
        "platform": platform,
        "gpu": gpu,
        "expected_cost_usd": expected_cost_usd,
        "expected_axis": expected_axis,
        "recipe": recipe,
        "dispatched_at_utc": dispatched_at_utc or _now_iso(),
        "harvested_at_utc": None,
        "status": STATUS_DISPATCHED,
        "rc": None,
        "elapsed_seconds": None,
        "cost_actual_usd": None,
        "score": None,
        "score_axis": None,
        "archive_sha256": None,
        "archive_bytes": None,
        "evidence_grade": None,
        "max_seconds": max_seconds,
        "mounted_code_git_head": mounted_code_git_head,
        "agent": agent,
        "subagent_id": subagent_id,
        "session_id": session_id,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    # Allow callers to attach extra metadata (e.g. operator notes, parent
    # session anchor) without having to extend the schema. Reserved fields
    # cannot be overwritten via extra to preserve audit-trail integrity.
    reserved = set(record.keys())
    for k, v in extra.items():
        if k in reserved:
            raise ValueError(f"extra kwarg {k!r} collides with a reserved schema field")
        record[k] = v
    return _append_event_locked(record, path=path, lock_path=lock_path)


def update_call_id_outcome(
    *,
    call_id: str,
    status: str,
    event_type: str | None = None,
    harvest_result: dict[str, Any] | None = None,
    harvested_at_utc: str | None = None,
    rc: int | None = None,
    elapsed_seconds: float | None = None,
    score: float | None = None,
    score_axis: str | None = None,
    archive_sha256: str | None = None,
    archive_bytes: int | None = None,
    evidence_grade: str | None = None,
    cost_actual_usd: float | None = None,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append an outcome event row referencing an existing call_id.

    Per CLAUDE.md "HISTORICAL_PROVENANCE" + Catalog #110 / #132 — outcomes
    are NEW rows referencing the same ``call_id``, NEVER mutations of the
    original ``dispatched`` row. The full lifecycle of a call_id is the
    chronological sequence of all rows with that ``call_id`` value.

    The ``event_type`` defaults to the value of ``status``; pass it
    explicitly to disambiguate (e.g. status="failed" event_type="stale" if
    the caller wants to record an outcome class distinct from the status
    token).

    Returns the appended record.
    """
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("call_id must be a non-empty string")
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)!r}, got {status!r}")
    resolved_event_type = event_type or status
    if resolved_event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}, got {resolved_event_type!r}")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": resolved_event_type,
        "call_id": call_id,
        # The fields below are denormalized from the dispatched row when known.
        # Caller may pass them via **extra to pin specific values; otherwise
        # consumers reconstruct via query_by_call_id().
        "lane_id": extra.pop("lane_id", None),
        "label": extra.pop("label", None),
        "platform": extra.pop("platform", "modal"),
        "gpu": extra.pop("gpu", None),
        "expected_cost_usd": extra.pop("expected_cost_usd", None),
        "expected_axis": extra.pop("expected_axis", None),
        "recipe": extra.pop("recipe", None),
        "dispatched_at_utc": extra.pop("dispatched_at_utc", None),
        "harvested_at_utc": harvested_at_utc or _now_iso(),
        "status": status,
        "rc": rc,
        "elapsed_seconds": elapsed_seconds,
        "cost_actual_usd": cost_actual_usd,
        "score": score,
        "score_axis": score_axis,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "evidence_grade": evidence_grade,
        "max_seconds": extra.pop("max_seconds", None),
        "mounted_code_git_head": extra.pop("mounted_code_git_head", None),
        "agent": agent,
        "subagent_id": subagent_id,
        "session_id": session_id,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    if harvest_result is not None:
        record["harvest_result"] = harvest_result

    # Allow extra metadata (preserving audit-trail integrity).
    reserved = set(record.keys())
    for k, v in extra.items():
        if k in reserved:
            raise ValueError(f"extra kwarg {k!r} collides with a reserved schema field")
        record[k] = v

    return _append_event_locked(record, path=path, lock_path=lock_path)


# ─────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────


def query_by_call_id(
    call_id: str,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all events for ``call_id`` in chronological JSONL append order.

    Empty list if no rows exist for the id.
    """
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("call_id must be a non-empty string")
    return [r for r in load_call_ids(path) if r.get("call_id") == call_id]


def query_by_lane(
    lane_id: str,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all events for ``lane_id`` in chronological JSONL append order."""
    if not isinstance(lane_id, str) or not lane_id.strip():
        raise ValueError("lane_id must be a non-empty string")
    return [r for r in load_call_ids(path) if r.get("lane_id") == lane_id]


def query_all_post_utc(
    utc_str: str,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all events with ``written_at_utc >= utc_str`` (ISO-8601 string).

    String comparison works because ISO-8601 with Z suffix is sortable.
    """
    if not isinstance(utc_str, str) or not utc_str.strip():
        raise ValueError("utc_str must be a non-empty ISO-8601 string")
    return [
        r for r in load_call_ids(path) if isinstance(r.get("written_at_utc"), str) and r["written_at_utc"] >= utc_str
    ]


def query_unharvested(
    *,
    older_than_seconds: float = 0,
    path: Path | None = None,
    now_utc: _dt.datetime | None = None,
) -> list[dict[str, Any]]:
    """Return the latest event row per call_id whose status is non-terminal.

    "Non-terminal" means the latest row's status is NOT in ``TERMINAL_STATUSES``
    (i.e. still ``dispatched``). This is the harvester's primary index — every
    call_id needing a ``FunctionCall.from_id(...).get(...)`` poll.

    ``older_than_seconds``: optional age filter — only return call_ids whose
    ``dispatched_at_utc`` is at least this many seconds ago. Useful for
    skipping just-spawned calls that probably aren't done yet.
    """
    rows = load_call_ids(path)
    # Reduce to latest event per call_id.
    latest_by_call_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = row.get("call_id")
        if not isinstance(cid, str):
            continue
        # JSONL append order is chronological; later overwrites earlier.
        latest_by_call_id[cid] = row

    unharvested = [row for row in latest_by_call_id.values() if row.get("status") not in TERMINAL_STATUSES]
    if older_than_seconds > 0:
        cutoff = (now_utc or _dt.datetime.now(_dt.UTC)) - _dt.timedelta(seconds=older_than_seconds)
        cutoff_iso = cutoff.isoformat(timespec="microseconds").replace("+00:00", "Z")
        unharvested = [
            r
            for r in unharvested
            if isinstance(r.get("dispatched_at_utc"), str) and r["dispatched_at_utc"] <= cutoff_iso
        ]
    return unharvested


def latest_status_by_call_id(
    *,
    path: Path | None = None,
) -> dict[str, str]:
    """Return ``{call_id: latest_status}`` dict for every call_id in the ledger.

    Convenience for dashboards / status reporters that just want the current
    state per id without scanning every event.
    """
    rows = load_call_ids(path)
    out: dict[str, str] = {}
    for row in rows:
        cid = row.get("call_id")
        status = row.get("status")
        if isinstance(cid, str) and isinstance(status, str):
            out[cid] = status  # later overwrites earlier
    return out


__all__ = [
    "EVENT_DISPATCHED",
    "EVENT_FAILED",
    "EVENT_HARVESTED",
    "EVENT_MANUALLY_TERMINATED",
    "EVENT_STALE",
    "MODAL_CALL_ID_LEDGER_LOCK",
    "MODAL_CALL_ID_LEDGER_PATH",
    "SCHEMA_VERSION",
    "STATUS_DISPATCHED",
    "STATUS_FAILED",
    "STATUS_HARVESTED",
    "STATUS_MANUALLY_TERMINATED",
    "STATUS_STALE",
    "TERMINAL_STATUSES",
    "VALID_EVENT_TYPES",
    "VALID_STATUSES",
    "CallIdLedgerCorruptError",
    "latest_status_by_call_id",
    "load_call_ids",
    "load_call_ids_strict",
    "query_all_post_utc",
    "query_by_call_id",
    "query_by_lane",
    "query_unharvested",
    "register_dispatched_call_id",
    "update_call_id_outcome",
]
