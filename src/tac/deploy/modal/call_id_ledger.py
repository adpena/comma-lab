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
import threading
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
MODAL_CALL_ID_LEDGER_PATH = REPO_ROOT / ".omx" / "state" / "modal_call_id_ledger.jsonl"
MODAL_CALL_ID_LEDGER_LOCK = MODAL_CALL_ID_LEDGER_PATH.with_suffix(MODAL_CALL_ID_LEDGER_PATH.suffix + ".lock")
# Sidecar index — LIVE_STATE per Catalog #113 (gitignored). Lazily rebuilt
# from the canonical JSONL; corruption / loss has zero forensic cost since
# the canonical ledger remains the source of truth. Per OP-10 read-amortized
# query helpers (Catalog #245 sister-class fix per codex chunk 5 Finding #2).
MODAL_CALL_ID_LEDGER_INDEX_PATH = REPO_ROOT / ".omx" / "state" / "modal_call_id_ledger_index.json"

# Schema version pinned for forward compatibility.
SCHEMA_VERSION = 1
# Sidecar index schema version. Bumped on incompatible index-format changes.
# Mismatch → forced rebuild from canonical JSONL (no risk; just re-scan).
INDEX_SCHEMA_VERSION = 1

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
# Catalog #360 sister of #339: a pre-spawn FATAL in `main()` that calls
# `sys.exit()` BEFORE `fn.spawn()` produces a silent-no-spawn pattern from
# Modal's perspective (app initialized, no task fired, app stopped). The
# canonical helper `register_pre_spawn_fatal` writes this event so the
# harvester + operator dashboard can distinguish silent-no-spawn from
# successful dispatch. Per CLAUDE.md "Bugs must be permanently fixed AND
# self-protected against" + the OVERNIGHT-J STC v2 5th consecutive silent-
# no-spawn anchor. The synthetic call_id format is
# `pre_spawn_fatal_<label>_<utc_compact>` since no Modal call_id exists yet.
EVENT_PRE_SPAWN_FATAL = "pre_spawn_fatal"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_DISPATCHED,
        EVENT_HARVESTED,
        EVENT_FAILED,
        EVENT_STALE,
        EVENT_MANUALLY_TERMINATED,
        EVENT_PRE_SPAWN_FATAL,
    }
)

# Status tokens. ``status`` echoes the latest event_type for convenience.
STATUS_DISPATCHED = EVENT_DISPATCHED
STATUS_HARVESTED = EVENT_HARVESTED
STATUS_FAILED = EVENT_FAILED
STATUS_STALE = EVENT_STALE
STATUS_MANUALLY_TERMINATED = EVENT_MANUALLY_TERMINATED
STATUS_PRE_SPAWN_FATAL = EVENT_PRE_SPAWN_FATAL

VALID_STATUSES = VALID_EVENT_TYPES

# Terminal statuses (no further events expected). A call_id whose latest row
# carries one of these is "harvested or otherwise resolved" for the purpose of
# ``query_unharvested``. Pre-spawn-fatal is terminal: no spawn ever happened,
# no further events to expect.
TERMINAL_STATUSES = frozenset({STATUS_HARVESTED, STATUS_FAILED, STATUS_STALE, STATUS_MANUALLY_TERMINATED, STATUS_PRE_SPAWN_FATAL})


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


# Thread-local lock-held depth (Catalog #140 sister pattern). Pairs with
# ``_ledger_lock_held()`` so any direct-write helper can refuse calls that
# bypass the canonical locked path. This must be thread-local: a process-global
# depth counter lets sibling threads skip the fcntl acquire while another
# thread is inside the critical section.
_ledger_lock_depth_tls = threading.local()


def _get_ledger_lock_depth() -> int:
    """Return this thread's local ledger-lock re-entry depth."""

    return int(getattr(_ledger_lock_depth_tls, "depth", 0))


def _set_ledger_lock_depth(value: int) -> None:
    """Set this thread's local ledger-lock re-entry depth."""

    _ledger_lock_depth_tls.depth = int(value)


def _ledger_lock_held() -> bool:
    """Return True if THIS thread is currently inside ``_ledger_lock``."""
    return _get_ledger_lock_depth() > 0


@contextlib.contextmanager
def _ledger_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the ledger lock file.

    Lock is process-advisory (``fcntl.flock`` ``LOCK_EX``); multiple
    processes contending serialize on the lock file. Re-entry within the same
    process is counted (depth > 1); fcntl is only re-acquired on the 0->1
    transition to avoid same-process deadlock.
    """
    p = lock_path or MODAL_CALL_ID_LEDGER_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    depth = _get_ledger_lock_depth()
    if depth > 0:
        _set_ledger_lock_depth(depth + 1)
        try:
            yield None
        finally:
            _set_ledger_lock_depth(_get_ledger_lock_depth() - 1)
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
        _set_ledger_lock_depth(_get_ledger_lock_depth() + 1)
        try:
            yield fd
        finally:
            _set_ledger_lock_depth(_get_ledger_lock_depth() - 1)
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
    use_truly_appending: bool = True,
) -> dict[str, Any]:
    """Append a single event record under fcntl lock.

    Per OP-10 (codex chunk 5 Finding #2 H-18): defaults to the O(1) amortized
    write path (POSIX O_APPEND under fcntl lock + sidecar index update in
    process). The legacy O(N) full-rewrite path is preserved behind
    ``use_truly_appending=False`` for callers that need the strict-reload
    semantics (e.g. corrupt-recovery flows that explicitly want a full
    serialization to validate every existing row).

    Default O(1) path:
      1. Acquire fcntl LOCK_EX
      2. ``_validate_ledger_tail`` — cheap O(1) sanity check on trailing line
      3. POSIX ``os.write`` with ``O_APPEND`` semantics (atomic at the kernel
         level for payloads <= PIPE_BUF; fcntl additionally serializes us
         against other process appenders)
      4. fsync the fd
      5. Update sidecar index with the new (offset, total_size) tuple

    Legacy O(N) path (``use_truly_appending=False``):
      1. Acquire fcntl LOCK_EX
      2. STRICT-reload ledger from disk (raises ``CallIdLedgerCorruptError``
         on ANY malformed row — append is refused)
      3. Build full payload from rows + record, write to unique ``.tmp.<uuid12>``
      4. ``fsync`` + ``os.replace`` for atomic commit

    Per CLAUDE.md "Locked writes preserve deletions" (Catalog #132): both
    paths are APPEND-ONLY — existing rows are preserved verbatim; only new
    rows are added.
    """
    if use_truly_appending:
        return _append_event_truly_appending_locked(
            record,
            path=path,
            lock_path=lock_path,
        )

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

    NOTE (OP-10 / codex chunk 5 Finding #2): this O(N) full rewrite path is
    PRESERVED for backward compatibility (corrupt-recovery paths, full ledger
    serialization), but new appends route through ``_append_event_truly_appending_locked``
    which uses POSIX ``O_APPEND`` for O(1) writes. See module docstring +
    ``feedback_op10_modal_ledger_append_amortization_landed_20260515.md``.
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
# OP-10: O(1) append + sidecar index amortization
# ─────────────────────────────────────────────────────────────────────────
#
# Per codex chunk 5 Finding #2 (`.omx/research/codex_chunked_full_codebase_review_20260515.md`
# H-18): the original ``_append_event_locked`` path does a FULL READ + FULL
# REWRITE of the JSONL file per single dispatch event. At 10000 entries this
# is ~14 MB of read+write under the exclusive fcntl lock, which can exceed
# the 30s ``LOCK_TIMEOUT_SECONDS`` ceiling under fan-out and recreates the
# orphan-call_id failure mode the canonical ledger exists to eliminate.
#
# OP-10 read-amortization preserves HISTORICAL_PROVENANCE per CLAUDE.md
# Catalog #110 / #113 / #132 (locked writes preserve deletions): the canonical
# JSONL file remains APPEND-ONLY; existing rows are NEVER mutated; new events
# become NEW rows referencing the same call_id. The amortization is purely
# the WRITE path (POSIX O_APPEND under fcntl lock instead of full rewrite)
# plus a sidecar INDEX file for O(1) reads.
#
# - WRITE path (truly-appending):
#     * Acquire fcntl LOCK_EX
#     * STRICT-validate the trailing region is well-formed (no torn write)
#     * Open with ``os.O_APPEND | os.O_WRONLY | os.O_CREAT``; POSIX guarantees
#       atomic append for write() at <= PIPE_BUF bytes (typically 4 KB)
#     * write() the new row terminated with \n; record file_size_after
#     * fsync
#     * Update sidecar index in-process (no second disk write inside the lock
#       — sidecar may be flushed lazily on read or on every Nth append)
#
# - SIDECAR INDEX (LIVE_STATE — gitignored):
#     * Path: MODAL_CALL_ID_LEDGER_INDEX_PATH
#     * Schema: {"schema_version": 1, "index_built_at_utc": "...",
#                "ledger_path": "...", "ledger_size_at_build": N,
#                "last_indexed_byte": M,
#                "call_id_to_byte_offsets": {"fc-...": [0, 1234], ...},
#                "lane_id_to_byte_offsets": {"lane_...": [0, 1234], ...}}
#     * Lazily REBUILT on read if (a) sidecar missing, (b) sidecar's
#       ``ledger_path`` differs, (c) sidecar's ``last_indexed_byte`` >
#       current ledger size (ledger truncated/rotated), (d)
#       ``schema_version`` mismatch.
#     * Lazily EXTENDED on read if ``ledger_size_at_build`` < current size
#       (incremental scan from ``last_indexed_byte``).
#
# - QUERY helpers:
#     * ``query_by_call_id_indexed(cid)``: O(1 + k) where k = events for cid
#     * ``query_by_lane_indexed(lane_id)``: O(1 + k)
#     * Original ``query_by_call_id`` / ``query_by_lane`` continue to work
#       (still O(N) scans for backward compat); they are NOT removed.
#
# - HISTORICAL_PROVENANCE proof:
#     * The canonical JSONL file is NEVER rewritten by the new write path.
#     * Each appended row is byte-identical to what _save_ledger would have
#       written (same json.dumps(sort_keys=True) + "\n" payload).
#     * The sidecar index is LIVE_STATE; deleting it forces rebuild but
#       does not lose any data.


def _truly_appending_write_locked(
    record: dict[str, Any],
    *,
    path: Path,
) -> tuple[int, int]:
    """Append a single row using POSIX O_APPEND under fcntl lock.

    Returns ``(byte_offset_of_new_row, total_file_size_after)`` so the index
    cache can be updated in-process without a second read of the file.

    Runtime-asserts the caller holds ``_ledger_lock``. Per CLAUDE.md
    "Comment-only contracts — FORBIDDEN".

    The line payload is byte-identical to what ``_save_ledger`` would have
    written (``json.dumps(record, sort_keys=True) + "\\n"``) so existing
    consumers of the canonical JSONL see no behavioral change.
    """
    if not _ledger_lock_held():
        raise RuntimeError(
            "_truly_appending_write_locked called WITHOUT holding _ledger_lock. "
            "Per CLAUDE.md Catalog #131 (no bare writes to shared state) + "
            "#140 (state writers own their lock end-to-end), append must "
            "happen inside the canonical fcntl-locked context manager."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    # Open with O_APPEND | O_WRONLY | O_CREAT. POSIX atomic-append guarantee:
    # writes <= PIPE_BUF (typically 4096 bytes; per-row dispatched/harvested
    # records run ~600-1100 bytes, well under the limit) are atomic with
    # respect to other O_APPEND writers. The fcntl lock additionally serializes
    # the read-validate + write window so concurrent appenders cannot interleave.
    fd = os.open(str(path), os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        # Snapshot pre-write size to compute the new row's byte offset.
        pre_size = os.fstat(fd).st_size
        n = os.write(fd, payload)
        if n != len(payload):
            # Partial write — should be impossible for a payload < PIPE_BUF
            # under POSIX O_APPEND, but we surface fail-loud per CLAUDE.md
            # "Forbidden silent-skip cascades (the bootstrap trap)".
            raise OSError(
                f"_truly_appending_write_locked: short write ({n} of "
                f"{len(payload)} bytes) to {path}; partial-write recovery "
                "is NOT supported because it would corrupt the audit trail."
            )
        os.fsync(fd)
        post_size = os.fstat(fd).st_size
    finally:
        os.close(fd)
    return pre_size, post_size


def _validate_ledger_tail(path: Path) -> None:
    """Verify the trailing line of the ledger is well-formed JSONL or empty.

    Cheap O(1) sanity check that runs INSIDE the lock before each append.
    A torn trailing line would indicate the previous append was interrupted
    mid-write (which O_APPEND + fcntl lock should prevent, but we verify
    fail-loud per CLAUDE.md "Bugs must be permanently fixed AND
    self-protected against").

    Raises ``CallIdLedgerCorruptError`` on a torn / non-JSON trailing line
    so the caller can quarantine. An empty trailing newline is treated as
    well-formed (json.loads of "" is invalid; we strip first).
    """
    if not path.exists():
        return  # empty ledger == well-formed
    size = path.stat().st_size
    if size == 0:
        return
    # Read up to last 64 KB to bound work even on huge ledgers; this is
    # always enough for one well-formed event row (max ~2 KB).
    READ_TAIL = min(size, 65536)
    with open(path, "rb") as f:
        f.seek(size - READ_TAIL)
        tail = f.read(READ_TAIL)
    text = tail.decode("utf-8", errors="replace")
    # Find last well-formed line break.
    if not text.endswith("\n"):
        # Trailing region missing newline → torn write.
        raise CallIdLedgerCorruptError(
            f"call_id ledger at {path} trailing region (last {READ_TAIL} "
            f"bytes) does not end with newline; possible torn write. "
            "Append refused; operator must inspect."
        )
    # Parse the last line for JSON validity.
    lines = text.splitlines()
    if not lines:
        return
    last = lines[-1].strip()
    if not last:
        return
    try:
        row = json.loads(last)
    except json.JSONDecodeError as exc:
        raise CallIdLedgerCorruptError(
            f"call_id ledger at {path} trailing line is invalid JSON: "
            f"{exc}. Append refused; operator must inspect/repair."
        ) from exc
    if not isinstance(row, dict):
        raise CallIdLedgerCorruptError(
            f"call_id ledger at {path} trailing line has non-dict root "
            f"(type={type(row).__name__}); expected JSON object."
        )


def _append_event_truly_appending_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    update_index: bool = True,
) -> dict[str, Any]:
    """O(1) append path — POSIX O_APPEND under fcntl lock.

    Per codex chunk 5 Finding #2 + OP-10 fix. This is the new canonical
    write path. The original ``_append_event_locked`` (full rewrite) is
    PRESERVED for callers that explicitly need it (corrupt-recovery, full
    serialization, schema migration); it is NO LONGER the default.

    HISTORICAL_PROVENANCE invariants (Catalog #110 / #113 / #132):
      * Canonical JSONL file is APPEND-ONLY (existing bytes never modified)
      * New row's byte payload is identical to what ``_save_ledger`` would
        have produced (same ``json.dumps(record, sort_keys=True) + "\\n"``)
      * Quarantine path on tail-corruption is the same as the full-rewrite
        path

    Returns the appended record (with server-side fields stamped).
    """
    _validate_event_record(record)
    p_path = path or MODAL_CALL_ID_LEDGER_PATH
    l_path = lock_path or MODAL_CALL_ID_LEDGER_LOCK

    with _ledger_lock(l_path):
        # Tail validation — refuse to append onto torn / non-JSON trailing
        # bytes. Quarantine the file so the canonical path becomes empty
        # for the next caller (consistent with the full-rewrite path's
        # quarantine semantics).
        try:
            _validate_ledger_tail(p_path)
        except CallIdLedgerCorruptError as exc:
            quarantine_path = _quarantine_corrupt_file(p_path)
            raise CallIdLedgerCorruptError(
                f"call_id ledger at {p_path} tail was corrupt; quarantined "
                f"to {quarantine_path}. Append refused; operator must "
                "repair (see CallIdLedgerCorruptError docstring)."
            ) from exc

        byte_offset, total_size = _truly_appending_write_locked(record, path=p_path)

        # Update sidecar index in-process. The sidecar is LIVE_STATE so a
        # write failure here is non-fatal (next read will rebuild from the
        # canonical JSONL).
        if update_index:
            try:
                _index_record_event_locked(
                    record=record,
                    byte_offset=byte_offset,
                    total_size=total_size,
                    ledger_path=p_path,
                )
            except Exception:
                # Sidecar update failures must NOT block the write since the
                # canonical JSONL is the source of truth. Log-via-fail-soft
                # per CLAUDE.md "Forbidden silent-skip cascades" exception:
                # the canonical artifact is already written and intact.
                pass

        return record


# ─────────────────────────────────────────────────────────────────────────
# Sidecar index — LIVE_STATE, lazily rebuilt from canonical JSONL
# ─────────────────────────────────────────────────────────────────────────


def _empty_index() -> dict[str, Any]:
    """Return a fresh empty sidecar-index payload."""
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "index_built_at_utc": _now_iso(),
        "ledger_path": str(MODAL_CALL_ID_LEDGER_PATH),
        "ledger_size_at_build": 0,
        "last_indexed_byte": 0,
        "call_id_to_byte_offsets": {},
        "lane_id_to_byte_offsets": {},
    }


def _load_index(
    *,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Load sidecar index file or return empty index.

    LENIENT: missing file or JSON parse failure → empty index (caller
    rebuilds). The sidecar is LIVE_STATE; corruption has zero forensic
    cost since the canonical JSONL is the source of truth.
    """
    p = index_path or MODAL_CALL_ID_LEDGER_INDEX_PATH
    if not p.exists():
        return _empty_index()
    try:
        text = p.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            return _empty_index()
        if data.get("schema_version") != INDEX_SCHEMA_VERSION:
            return _empty_index()
        # Defensive defaults for missing keys.
        data.setdefault("ledger_path", str(MODAL_CALL_ID_LEDGER_PATH))
        data.setdefault("ledger_size_at_build", 0)
        data.setdefault("last_indexed_byte", 0)
        data.setdefault("call_id_to_byte_offsets", {})
        data.setdefault("lane_id_to_byte_offsets", {})
        return data
    except (OSError, json.JSONDecodeError):
        return _empty_index()


def _save_index(
    index: dict[str, Any],
    *,
    index_path: Path | None = None,
) -> None:
    """Write sidecar index atomically (tmp + os.replace).

    No fcntl lock required because the sidecar is LIVE_STATE and last-writer-
    wins is acceptable: the next read will rebuild from the canonical JSONL
    if the sidecar is stale or torn. Atomic-replace prevents partial-read
    races for concurrent readers.
    """
    p = index_path or MODAL_CALL_ID_LEDGER_INDEX_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        tmp.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _index_record_event_locked(
    *,
    record: dict[str, Any],
    byte_offset: int,
    total_size: int,
    ledger_path: Path,
    index_path: Path | None = None,
) -> None:
    """Update sidecar index with a single new event row.

    MUST be called inside ``_ledger_lock`` (caller already holds it).
    Updates ``call_id_to_byte_offsets`` and ``lane_id_to_byte_offsets``
    plus ``last_indexed_byte`` to track incremental progress.
    """
    if not _ledger_lock_held():
        raise RuntimeError(
            "_index_record_event_locked called WITHOUT holding _ledger_lock; "
            "concurrent index writes can produce inconsistent offset maps."
        )
    index = _load_index(index_path=index_path)
    # If sidecar's ledger_path mismatches OR the cached size > the byte_offset
    # of the new row, the index is stale; force a rebuild.
    if index.get("ledger_path") != str(ledger_path) or index.get("last_indexed_byte", 0) > byte_offset:
        index = _empty_index()
        index["ledger_path"] = str(ledger_path)
    cid = record.get("call_id")
    lid = record.get("lane_id")
    if isinstance(cid, str):
        offsets = index["call_id_to_byte_offsets"].setdefault(cid, [])
        offsets.append(byte_offset)
    if isinstance(lid, str):
        offsets = index["lane_id_to_byte_offsets"].setdefault(lid, [])
        offsets.append(byte_offset)
    index["last_indexed_byte"] = total_size
    index["ledger_size_at_build"] = total_size
    _save_index(index, index_path=index_path)


def _rebuild_index_from_ledger(
    *,
    ledger_path: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Full rescan of canonical JSONL → fresh sidecar index.

    Used when the sidecar is missing, corrupt, schema-mismatched, or its
    ``ledger_path`` does not match the current canonical path. O(N) one-time
    cost; subsequent appends update the sidecar incrementally.

    Runs WITHOUT holding ``_ledger_lock`` because (a) the canonical JSONL is
    APPEND-ONLY so a partial trailing line at most reflects an in-flight
    append (which the canonical helpers serialize via fcntl), and (b) the
    sidecar is LIVE_STATE so concurrent rebuilders converge to the same
    answer (last-writer-wins via atomic-replace).
    """
    p = ledger_path or MODAL_CALL_ID_LEDGER_PATH
    index = _empty_index()
    index["ledger_path"] = str(p)
    if not p.exists():
        _save_index(index, index_path=index_path)
        return index
    # Stream-parse with byte offsets.
    with open(p, "rb") as f:
        offset = 0
        for line_bytes in f:
            stripped = line_bytes.strip()
            if stripped:
                try:
                    row = json.loads(stripped.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Skip torn/non-JSON line during rebuild; lenient by
                    # design since the rebuild is a sidecar-only operation
                    # and the canonical JSONL remains the source of truth.
                    offset += len(line_bytes)
                    continue
                if isinstance(row, dict):
                    cid = row.get("call_id")
                    lid = row.get("lane_id")
                    if isinstance(cid, str):
                        index["call_id_to_byte_offsets"].setdefault(cid, []).append(offset)
                    if isinstance(lid, str):
                        index["lane_id_to_byte_offsets"].setdefault(lid, []).append(offset)
            offset += len(line_bytes)
    index["ledger_size_at_build"] = offset
    index["last_indexed_byte"] = offset
    _save_index(index, index_path=index_path)
    return index


def _ensure_index_fresh(
    *,
    ledger_path: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Return the sidecar index, rebuilding/extending if stale.

    Three cases:
      1. Sidecar missing or corrupt → full rebuild
      2. Sidecar's ``last_indexed_byte`` > current ledger size → ledger
         truncated/rotated; full rebuild
      3. Sidecar's ``last_indexed_byte`` < current ledger size →
         INCREMENTAL extend (read only the new bytes)

    Case 3 is the hot path under load and is what makes the sidecar
    valuable — index never re-scans bytes already indexed.
    """
    p = ledger_path or MODAL_CALL_ID_LEDGER_PATH
    index = _load_index(index_path=index_path)

    if index.get("ledger_path") != str(p):
        return _rebuild_index_from_ledger(ledger_path=p, index_path=index_path)

    if not p.exists():
        # Index might be non-empty but ledger gone → rebuild (empty result)
        return _rebuild_index_from_ledger(ledger_path=p, index_path=index_path)

    cur_size = p.stat().st_size
    last_indexed = int(index.get("last_indexed_byte", 0))

    if last_indexed > cur_size:
        # Ledger truncated/rotated → rebuild
        return _rebuild_index_from_ledger(ledger_path=p, index_path=index_path)

    if last_indexed == cur_size:
        return index  # already fresh

    # Incremental extend: read [last_indexed, cur_size) and append to index
    with open(p, "rb") as f:
        f.seek(last_indexed)
        offset = last_indexed
        for line_bytes in f:
            stripped = line_bytes.strip()
            if stripped:
                try:
                    row = json.loads(stripped.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    offset += len(line_bytes)
                    continue
                if isinstance(row, dict):
                    cid = row.get("call_id")
                    lid = row.get("lane_id")
                    if isinstance(cid, str):
                        index["call_id_to_byte_offsets"].setdefault(cid, []).append(offset)
                    if isinstance(lid, str):
                        index["lane_id_to_byte_offsets"].setdefault(lid, []).append(offset)
            offset += len(line_bytes)
    index["last_indexed_byte"] = offset
    index["ledger_size_at_build"] = offset
    _save_index(index, index_path=index_path)
    return index


def _read_rows_at_offsets(
    offsets: list[int],
    *,
    ledger_path: Path,
) -> list[dict[str, Any]]:
    """Read JSONL rows at specific byte offsets in the canonical ledger.

    O(k) for k rows requested vs O(N) for full file scan. Lines with
    decode/parse errors are skipped (sidecar may have stale offsets after
    quarantine; failing-soft preserves the read path).
    """
    if not offsets:
        return []
    rows: list[dict[str, Any]] = []
    with open(ledger_path, "rb") as f:
        for off in sorted(offsets):
            try:
                f.seek(off)
                line_bytes = f.readline()
            except OSError:
                continue
            stripped = line_bytes.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def query_by_call_id_indexed(
    call_id: str,
    *,
    ledger_path: Path | None = None,
    index_path: Path | None = None,
) -> list[dict[str, Any]]:
    """O(1 + k) lookup of all events for ``call_id`` via sidecar index.

    Read-amortized version of ``query_by_call_id``. Sidecar is lazily
    refreshed on call (incremental scan from ``last_indexed_byte``).

    Returns rows in chronological JSONL append order.
    """
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("call_id must be a non-empty string")
    p = ledger_path or MODAL_CALL_ID_LEDGER_PATH
    if not p.exists():
        return []
    index = _ensure_index_fresh(ledger_path=p, index_path=index_path)
    offsets = index.get("call_id_to_byte_offsets", {}).get(call_id, [])
    return _read_rows_at_offsets(offsets, ledger_path=p)


def query_by_lane_indexed(
    lane_id: str,
    *,
    ledger_path: Path | None = None,
    index_path: Path | None = None,
) -> list[dict[str, Any]]:
    """O(1 + k) lookup of all events for ``lane_id`` via sidecar index."""
    if not isinstance(lane_id, str) or not lane_id.strip():
        raise ValueError("lane_id must be a non-empty string")
    p = ledger_path or MODAL_CALL_ID_LEDGER_PATH
    if not p.exists():
        return []
    index = _ensure_index_fresh(ledger_path=p, index_path=index_path)
    offsets = index.get("lane_id_to_byte_offsets", {}).get(lane_id, [])
    return _read_rows_at_offsets(offsets, ledger_path=p)


def rebuild_sidecar_index(
    *,
    ledger_path: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    """Operator-callable: force a full rebuild of the sidecar index.

    Useful after manual ledger edits, quarantine recovery, or schema
    upgrades. Equivalent to deleting the sidecar file and calling any
    indexed query helper.
    """
    return _rebuild_index_from_ledger(ledger_path=ledger_path, index_path=index_path)


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
    upstream_snapshot_sha256: str | None = None,
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
        # 12-month premortem item #2 (2026-05-16): persist the upstream/
        # snapshot SHA-256 at dispatch time so a harvested call_id can be
        # cross-checked against the snapshot it was scored on. Legacy rows
        # pre-this-field land as ``None`` (snapshot unknown).
        "upstream_snapshot_sha256": upstream_snapshot_sha256,
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

    appended = _append_event_locked(record, path=path, lock_path=lock_path)

    # Catalog #343 DX auto-update wire-in: every harvested outcome with a
    # numeric score auto-refreshes the canonical frontier pointer so
    # operator-facing surfaces never drift from canonical state. Fail-quiet
    # per CLAUDE.md "Subagent coherence-by-default" maximum-signal-preservation:
    # pointer refresh failure must not propagate (ledger write already
    # succeeded; pointer is a downstream observability surface).
    try:
        from tac.canonical_frontier_pointer import (
            auto_refresh_canonical_frontier_after_dispatch_outcome,
        )

        auto_refresh_canonical_frontier_after_dispatch_outcome(
            status=status,
            score=score,
            score_axis=score_axis,
            archive_sha256=archive_sha256,
        )
    except Exception:  # noqa: BLE001 — fail-quiet per the contract
        pass

    return appended


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


# ---------------------------------------------------------------------------
# Catalog #339 — SILENT-NO-SPAWN-STRUCTURAL-EXTINCTION (2026-05-19)
# ---------------------------------------------------------------------------
# Bug class anchor: today's 3 consecutive incidents (Z6 Wave 2 4c / STC v2 /
# STC sister) all exhibited the silent-no-spawn pattern — `.spawn()` returned
# successfully but the existing `try/except Exception: print WARNING + continue`
# at `experiments/modal_train_lane.py:1921-1960` silently swallowed any
# `register_dispatched_call_id` failure. The Modal dispatch already happened
# (paid GPU meter started), but the ledger had no row → harvester invisible
# → operator effort wasted.
#
# Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable + "Bugs
# must be permanently fixed AND self-protected against": every dispatch MUST
# be followed by a scheduled harvest within 24h. The harvester reads the
# canonical ledger. No ledger row = no harvest = invisible orphan dispatch.
#
# Fix: `register_dispatched_call_id_fail_closed` REFUSES to swallow ANY
# failure. On exception, it (a) prints a fail-closed diagnostic referencing
# Catalog #339, (b) attempts a last-resort tmp-file dump so the harvester
# can recover via `tools/harvest_modal_calls.py --recover-from-tmp`, and
# (c) re-raises so the caller's `sys.exit` path becomes non-zero.
# ---------------------------------------------------------------------------


class LedgerRegistrationFailedError(RuntimeError):
    """Raised when ``register_dispatched_call_id_fail_closed`` cannot
    persist a dispatched-event row to the canonical ledger.

    Catalog #339 (SILENT-NO-SPAWN-STRUCTURAL-EXTINCTION 2026-05-19). The
    canonical helper used to ``try/except Exception: print WARNING`` so any
    fcntl-lock contention / corrupt-state / disk-full / sister-subagent
    edit silently broke the harvester invariant per CLAUDE.md
    "Modal `.spawn()` HARVEST OR LOSE" non-negotiable.

    Callers (e.g. ``experiments/modal_train_lane.py::main``) MUST either
    let this propagate (so ``sys.exit`` becomes non-zero) OR wrap it AND
    invoke ``_write_last_resort_tmp_ledger_dump`` so the harvester can
    recover via ``tools/harvest_modal_calls.py --recover-from-tmp``.

    Sister of ``CallIdLedgerCorruptError`` (Catalog #138 strict-load
    discipline) and ``LightningStateLostUpdateError`` (Catalog #131/#138
    sister fcntl-locked state class).
    """


_LAST_RESORT_TMP_DUMP_DIR = REPO_ROOT / ".omx" / "state" / "modal_call_id_ledger_recovery_tmp"


def _write_last_resort_tmp_ledger_dump(record: dict[str, Any]) -> Path:
    """Atomically dump a dispatched-event record to a recovery directory.

    Used by ``register_dispatched_call_id_fail_closed`` when the canonical
    fcntl-locked append fails. The harvester scans this directory and
    re-attempts the canonical append for each surviving row.

    Returns the path of the written tmp file. Best-effort: if THIS write
    also fails, ``LedgerRegistrationFailedError`` is still raised by the
    caller so the operator sees the dispatch failed structurally.
    """

    _LAST_RESORT_TMP_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_name = f"recover_{record.get('call_id', 'unknown')}_{uuid.uuid4().hex[:12]}.json"
    tmp_path = _LAST_RESORT_TMP_DUMP_DIR / tmp_name
    payload = json.dumps(record, sort_keys=True) + "\n"
    # Atomic write per Catalog #131 atomic-replace discipline so the harvester
    # never reads a half-written row.
    staging = tmp_path.with_suffix(tmp_path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
    staging.write_text(payload, encoding="utf-8")
    os.replace(staging, tmp_path)
    return tmp_path


def register_dispatched_call_id_fail_closed(
    *,
    call_id: str,
    lane_id: str,
    label: str,
    write_last_resort_dump: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fail-closed wrapper for :func:`register_dispatched_call_id`.

    Catalog #339 — SILENT-NO-SPAWN-STRUCTURAL-EXTINCTION (2026-05-19).

    Unlike the legacy pattern in ``experiments/modal_train_lane.py``
    (``try: register_dispatched_call_id(...) except Exception: print``),
    this wrapper REFUSES to swallow failures. On exception it:

    1. Constructs the dispatched-event record using the SAME schema as
       the canonical helper (so the recovery surface matches).
    2. Best-effort dumps the record to
       ``.omx/state/modal_call_id_ledger_recovery_tmp/<call_id>.json``
       so ``tools/harvest_modal_calls.py --recover-from-tmp`` can re-attempt
       the canonical append later (preserving the harvester invariant
       per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE").
    3. Raises ``LedgerRegistrationFailedError`` so the caller's exit path
       becomes non-zero, surfacing the failure to the operator instead of
       silently producing an invisible orphan Modal dispatch.

    Returns the appended record on success (identical to the canonical
    helper).
    """

    try:
        return register_dispatched_call_id(
            call_id=call_id,
            lane_id=lane_id,
            label=label,
            **kwargs,
        )
    except Exception as exc:
        last_resort_path: Path | None = None
        if write_last_resort_dump:
            record = {
                "schema_version": SCHEMA_VERSION,
                "event_type": EVENT_DISPATCHED,
                "call_id": call_id,
                "lane_id": lane_id,
                "label": label,
                "status": STATUS_DISPATCHED,
                "dispatched_at_utc": kwargs.get("dispatched_at_utc") or _now_iso(),
                "platform": kwargs.get("platform", "modal"),
                "gpu": kwargs.get("gpu", "T4"),
                "max_seconds": kwargs.get("max_seconds"),
                "mounted_code_git_head": kwargs.get("mounted_code_git_head"),
                "agent": kwargs.get("agent", "claude"),
                "upstream_snapshot_sha256": kwargs.get("upstream_snapshot_sha256"),
                "_recovery_reason": (
                    f"canonical_append_failed:{type(exc).__name__}:{str(exc)[:200]}"
                ),
            }
            try:
                last_resort_path = _write_last_resort_tmp_ledger_dump(record)
            except Exception:  # pragma: no cover -- truly catastrophic
                last_resort_path = None
        msg = (
            f"Catalog #339 — register_dispatched_call_id failed for "
            f"call_id={call_id!r} lane={lane_id!r}: "
            f"{type(exc).__name__}: {exc}. "
        )
        if last_resort_path is not None:
            msg += (
                f"Last-resort dump written to {last_resort_path}; "
                "harvester recovery via "
                "`tools/harvest_modal_calls.py --recover-from-tmp`."
            )
        else:
            msg += (
                "Last-resort dump ALSO failed; the Modal dispatch is "
                "INVISIBLE to the harvester per CLAUDE.md "
                "'Modal .spawn() HARVEST OR LOSE' non-negotiable. "
                "Operator must manually look up the call_id via the "
                "Modal dashboard."
            )
        raise LedgerRegistrationFailedError(msg) from exc


def register_pre_spawn_fatal(
    *,
    label: str,
    lane_id: str,
    fatal_reason: str,
    exit_code: int = 2,
    sys_exit_line_number: int | None = None,
    sys_exit_helper_source: str | None = None,
    write_last_resort_dump: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Catalog #360 sister of #339 — pre-spawn FATAL observability helper.

    The OVERNIGHT-J STC v2 5th consecutive silent-no-spawn anchor (per
    `.omx/research/stc_v2_ratify_or_defer_path_b_dispatch_landed_20260521.md`)
    proved that the upstream-of-``fn.spawn()`` ``sys.exit(2)`` paths in
    ``experiments/modal_train_lane.py::main`` produce a silent-no-spawn
    pattern: Modal app initializes (creates mounts + functions), the
    ``@app.local_entrypoint()`` ``main()`` then calls ``sys.exit(2)``
    BEFORE ``fn.spawn()`` queues a task, and Modal sees a clean exit with
    zero tasks. The app goes to "stopped" state with no row in the
    canonical ledger and no recovery dump.

    This helper extincts that structural bug class by writing a
    ``pre_spawn_fatal`` event to the canonical ledger BEFORE the caller
    calls ``sys.exit(exit_code)``. The harvester + operator dashboard +
    cost-band reconciliation surface can then distinguish:

    - silent-no-spawn from cleanly-dispatched-but-failed (Catalog #339)
    - operational/environmental fatal from substrate paradigm falsification
    - which specific ``sys.exit(2)`` path fired (line + helper source)

    Per Carmack MVP-first 5-step + CLAUDE.md "Bugs must be permanently
    fixed AND self-protected against" + "Modal `.spawn()` HARVEST OR LOSE"
    non-negotiables. Sister of Catalog #339 (post-spawn registration
    fail-closed); together they extinct the silent-no-spawn bug class at
    BOTH the pre-spawn AND post-spawn surfaces.

    Args:
        label: same label argument the caller would have passed to fn.spawn().
            Used to build the synthetic call_id (since no Modal call_id
            exists yet).
        lane_id: dispatch lane id per Catalog #126.
        fatal_reason: human-readable FATAL message (typically the same string
            the caller prints to stderr before sys.exit).
        exit_code: the sys.exit code the caller will use (default 2 matches
            modal_train_lane.py FATAL paths).
        sys_exit_line_number: optional source line number where the
            sys.exit(...) is about to fire. Helps the harvester correlate
            failure pattern to specific upstream check.
        sys_exit_helper_source: optional helper module/path that triggered
            the FATAL (e.g. "_collect_trainer_extra_mount_payload").
        write_last_resort_dump: best-effort tmp-dump fallback if canonical
            append fails. Mirrors Catalog #339's last-resort dump pattern.
        **kwargs: additional metadata fields (gpu, mounted_code_git_head,
            agent, etc.) forwarded to the canonical ledger row.

    Returns:
        The appended record (or the last-resort-dump record on canonical
        append failure). The caller should ALWAYS proceed to sys.exit()
        after invoking this helper.

    Raises:
        Nothing — this helper is intentionally fail-OPEN (best-effort
        observability) so it cannot block the FATAL exit it's instrumenting.
        Failure modes are surfaced via last-resort tmp-dump + a printed
        warning to stderr (NOT a raise) so the caller's sys.exit() flow
        is preserved.
    """
    import sys as _sys

    # Synthetic call_id: no Modal call_id exists yet (fn.spawn never ran).
    # Format: `pre_spawn_fatal_<label_first_64>_<utc_compact>` so the row
    # is uniquely identifiable across multiple pre-spawn fatals on the
    # same label (e.g. STC v2's 5 consecutive fatals would produce 5 rows).
    utc_compact = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    label_token = label[:64].replace("/", "_").replace(" ", "_")
    synthetic_call_id = f"pre_spawn_fatal_{label_token}_{utc_compact}"

    record_extras: dict[str, Any] = {
        "fatal_reason": fatal_reason[:1000],
        "exit_code": int(exit_code),
        "sys_exit_line_number": sys_exit_line_number,
        "sys_exit_helper_source": sys_exit_helper_source,
        # Mark the row as NEVER-SPAWNED so the harvester does not attempt
        # to poll Modal for a result. Per CLAUDE.md "Forbidden empirical-
        # claim-without-evidence-tag" — score_claim=False structurally.
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "modal_call_id_actually_exists": False,
        "evidence_grade": "[pre_spawn_fatal_diagnostic]",
    }
    # Allow caller to forward additional context fields (gpu, head, etc.).
    record_extras.update(kwargs)

    # Construct the record directly with EVENT_PRE_SPAWN_FATAL semantics.
    # We do NOT route through register_dispatched_call_id() because that
    # helper hardcodes event_type=EVENT_DISPATCHED + status=STATUS_DISPATCHED
    # and reserves those keys (they cannot be overridden via extras).
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_PRE_SPAWN_FATAL,
        "call_id": synthetic_call_id,
        "lane_id": lane_id,
        "label": label,
        "platform": kwargs.get("platform", "modal"),
        "gpu": kwargs.get("gpu", "T4"),
        "expected_cost_usd": kwargs.get("expected_cost_usd"),
        "expected_axis": kwargs.get("expected_axis", "cuda"),
        "recipe": kwargs.get("recipe"),
        "dispatched_at_utc": _now_iso(),
        "harvested_at_utc": None,
        "status": STATUS_PRE_SPAWN_FATAL,
        "rc": int(exit_code),
        "elapsed_seconds": None,
        "cost_actual_usd": 0.0,  # pre-spawn = zero paid cost
        "score": None,
        "score_axis": None,
        "archive_sha256": None,
        "archive_bytes": None,
        "max_seconds": kwargs.get("max_seconds"),
        "mounted_code_git_head": kwargs.get("mounted_code_git_head"),
        "agent": kwargs.get("agent", "claude"),
        "subagent_id": kwargs.get("subagent_id"),
        "session_id": kwargs.get("session_id"),
        "upstream_snapshot_sha256": kwargs.get("upstream_snapshot_sha256"),
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
        # Catalog #360-specific observability fields. Keys do not collide
        # with the canonical schema because they are pre-spawn-fatal sub-
        # surface only; harvester + dashboard consumers detect via
        # event_type=EVENT_PRE_SPAWN_FATAL.
        "fatal_reason": record_extras["fatal_reason"],
        "exit_code": record_extras["exit_code"],
        "sys_exit_line_number": record_extras["sys_exit_line_number"],
        "sys_exit_helper_source": record_extras["sys_exit_helper_source"],
        "evidence_grade": record_extras["evidence_grade"],
        "modal_call_id_actually_exists": record_extras["modal_call_id_actually_exists"],
        "score_claim": record_extras["score_claim"],
        "promotion_eligible": record_extras["promotion_eligible"],
        "rank_or_kill_eligible": record_extras["rank_or_kill_eligible"],
    }
    try:
        return _append_event_locked(record)
    except Exception as exc:
        # Fail-OPEN: best-effort tmp-dump, but never raise. The caller's
        # sys.exit() flow must not be blocked by observability machinery.
        last_resort_path: Path | None = None
        if write_last_resort_dump:
            record = {
                "schema_version": SCHEMA_VERSION,
                "event_type": EVENT_PRE_SPAWN_FATAL,
                "call_id": synthetic_call_id,
                "lane_id": lane_id,
                "label": label,
                "status": STATUS_PRE_SPAWN_FATAL,
                "dispatched_at_utc": _now_iso(),
                "_recovery_reason": (
                    f"pre_spawn_fatal_canonical_append_failed:{type(exc).__name__}:{str(exc)[:200]}"
                ),
                **record_extras,
            }
            try:
                last_resort_path = _write_last_resort_tmp_ledger_dump(record)
            except Exception:  # pragma: no cover -- truly catastrophic
                last_resort_path = None
        print(
            f"[Catalog #360] pre-spawn-fatal observability write failed: "
            f"{type(exc).__name__}: {exc}. "
            + (
                f"Last-resort dump at {last_resort_path}."
                if last_resort_path is not None
                else "Last-resort dump ALSO failed. The pre-spawn fatal will be "
                "INVISIBLE to the harvester. Operator must manually correlate "
                "the Modal app id (printed before sys.exit) to this dispatch."
            ),
            file=_sys.stderr,
        )
        # Return synthetic record so callers can still log/print it locally.
        return {
            "schema_version": SCHEMA_VERSION,
            "event_type": EVENT_PRE_SPAWN_FATAL,
            "call_id": synthetic_call_id,
            "label": label,
            "lane_id": lane_id,
            **record_extras,
        }


def poll_ledger_for_call_id(
    call_id: str,
    *,
    timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.5,
    path: Path | None = None,
) -> bool:
    """Poll the canonical ledger for ``call_id`` presence within ``timeout``.

    Catalog #339 sister mitigation for the operator-authorize caller.
    Returns ``True`` once a row with ``call_id`` appears in the canonical
    ledger (any event_type); ``False`` if the timeout elapses.

    Used by ``tools/operator_authorize.py::_dispatch_modal`` AFTER
    ``modal_output_indicates_spawned_call(...)`` confirms ``.spawn()``
    happened, to verify the ledger row actually landed BEFORE returning
    rc=0 to the operator. If the spawn happened but the ledger row never
    appeared, the harvester is blind — caller must raise SystemExit
    instead of silently reporting success.
    """

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            rows = load_call_ids(path)
        except Exception:
            # Lenient-load failures here are NOT fatal — the harvester
            # uses the strict-load path which raises CallIdLedgerCorruptError;
            # we just want presence, not strict correctness, so retry.
            rows = []
        for row in rows:
            if row.get("call_id") == call_id:
                return True
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_seconds)
    return False


__all__ = [
    "EVENT_DISPATCHED",
    "EVENT_FAILED",
    "EVENT_HARVESTED",
    "EVENT_MANUALLY_TERMINATED",
    "EVENT_STALE",
    "INDEX_SCHEMA_VERSION",
    "MODAL_CALL_ID_LEDGER_INDEX_PATH",
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
    "LedgerRegistrationFailedError",
    "latest_status_by_call_id",
    "load_call_ids",
    "load_call_ids_strict",
    "poll_ledger_for_call_id",
    "query_all_post_utc",
    "query_by_call_id",
    "query_by_call_id_indexed",
    "query_by_lane",
    "query_by_lane_indexed",
    "query_unharvested",
    "rebuild_sidecar_index",
    "register_dispatched_call_id",
    "register_dispatched_call_id_fail_closed",
    "update_call_id_outcome",
]
