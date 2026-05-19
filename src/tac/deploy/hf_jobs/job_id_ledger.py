# SPDX-License-Identifier: MIT
"""Canonical HF Jobs dispatch ledger (Catalog #342 sister of Catalog #245).

Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable applied to
HF Jobs: every ``huggingface_hub.HfApi().run_uv_job(...)`` dispatch puts
the job id in a transient state surface (HF Hub job dashboard). The ledger
is the durable, queryable, fcntl-locked JSONL append-only HISTORICAL_PROVENANCE
record per Catalog #110 / #113 / #128 / #131 / #138 / #245.

Schema (mirrors ``tac.deploy.modal.call_id_ledger`` per the canonical
4-layer pattern):

- ``schema_version``: pinned at MODULE-level constant.
- ``event_type``: one of {``dispatched``, ``harvested``, ``failed``, ``stale``}.
- ``hf_jobs_id``: HF Jobs job id returned by ``run_uv_job(...).id``.
- ``lane_id``: canonical lane id (e.g.
  ``lane_hf_jobs_segnet_surrogate_distillation_20260519``).
- ``label``: human-readable dispatch label.
- ``platform``: always ``"hf_jobs"`` for this ledger.
- ``flavor``: HF Jobs hardware flavor (e.g. ``"t4-small"`` / ``"a10g-small"``).
- ``expected_cost_usd``: dispatch-time cost estimate (per HF Jobs pricing).
- ``expected_axis``: target evaluator axis (``cuda`` / ``cpu`` / ``advisory``).
- ``recipe``: operator-authorize recipe path.
- ``dispatched_at_utc`` / ``harvested_at_utc`` / ``written_at_utc``:
  ISO-8601 UTC timestamps.
- ``status``: monotonic state machine (``dispatched`` → ``harvested`` /
  ``failed`` / ``stale``).
- ``rc`` / ``elapsed_seconds`` / ``cost_actual_usd``: harvest outcome.
- ``score`` / ``score_axis`` / ``archive_sha256`` / ``archive_bytes`` /
  ``evidence_grade``: per-axis custody fields (CLAUDE.md "Apples-to-apples
  evidence discipline" + Catalog #127 fail-closed custody).
- ``hub_model_repo``: HF Hub destination repo id (e.g.
  ``adpena/segnet-image-level-surrogate-mobilenet-v3-small-200ep``).
- ``hub_model_sha``: HF Hub destination commit sha when known.
- ``hub_dataset_repo``: HF Hub source dataset repo id (e.g.
  ``adpena/comma-video-segnet-image-level-600pairs``).
- ``hub_dataset_sha``: HF Hub source dataset commit sha (provenance pin).
- ``mounted_code_git_head``: local repo HEAD sha at dispatch time.
- ``agent`` / ``subagent_id`` / ``session_id``: provenance.
- ``written_pid`` / ``written_host``: writer identity (forensic).

Per CLAUDE.md HISTORICAL_PROVENANCE: outcomes are NEW rows referencing the
same ``hf_jobs_id``, NEVER mutations of the original ``dispatched`` row.
The full lifecycle of an hf_jobs_id is the chronological sequence of all
rows with that id.

Catalog #131 (``check_no_bare_writes_to_shared_state``) is honored via
fcntl-locked atomic append (``_append_event_locked``). Catalog #138
(``check_state_writers_strict_load_for_mutating_path``) is honored via
``load_hf_jobs_strict`` which raises ``HFJobsLedgerCorruptError`` on
malformed JSONL.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Schema / constants
# --------------------------------------------------------------------------

SCHEMA_VERSION = "hf_jobs_ledger_v1_catalog342_20260519"

EVENT_DISPATCHED = "dispatched"
EVENT_HARVESTED = "harvested"
EVENT_FAILED = "failed"
EVENT_STALE = "stale"

VALID_EVENT_TYPES: frozenset[str] = frozenset({
    EVENT_DISPATCHED,
    EVENT_HARVESTED,
    EVENT_FAILED,
    EVENT_STALE,
})

# Status taxonomy — mirrors event_type for downstream consumers that only
# look at status (the dispatched event has status="dispatched"; harvested
# has status="harvested"; failed has status="failed"; stale has
# status="stale").
STATUS_DISPATCHED = "dispatched"
TERMINAL_STATUSES: frozenset[str] = frozenset({
    EVENT_HARVESTED,
    EVENT_FAILED,
    EVENT_STALE,
})


def _repo_root() -> Path:
    """Locate the repository root by walking up from this module."""

    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() and (parent / ".omx").exists():
            return parent
    raise RuntimeError(
        "Could not locate repository root from "
        f"{here!s}; expected pyproject.toml + .omx/ sibling."
    )


HF_JOBS_CALL_ID_LEDGER_PATH = _repo_root() / ".omx" / "state" / "hf_jobs_call_id_ledger.jsonl"
"""Canonical fcntl-locked JSONL append-only ledger path.

Registered in :data:`tac.preflight._SHARED_STATE_PATH_MARKERS` so Catalog
#131 (``check_no_bare_writes_to_shared_state``) refuses direct writes
outside the canonical helper.
"""

_HF_JOBS_CALL_ID_LEDGER_LOCK = HF_JOBS_CALL_ID_LEDGER_PATH.with_suffix(".lock")


class HFJobsLedgerCorruptError(RuntimeError):
    """Raised by ``load_hf_jobs_strict`` when the ledger contains malformed JSONL.

    Per Catalog #138 strict-load fail-closed discipline: a corrupt ledger
    line means a downstream consumer that uses the lenient ``load_hf_jobs``
    loader may silently drop rows. The strict loader is the canonical
    surface for any mutating helper that must see EVERY historical row
    (e.g., ``register_dispatched_hf_jobs_id`` which validates schema
    invariants against the existing chronology).
    """


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 with timezone suffix."""

    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# fcntl-locked append (Catalog #131 sister discipline)
# --------------------------------------------------------------------------


def _ensure_dirs(path: Path) -> None:
    """Create parent directories for the ledger path (idempotent)."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _serialize_row(record: dict[str, Any]) -> str:
    """Return canonical JSON serialization with deterministic key order."""

    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _append_event_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Append a row to the ledger under an exclusive fcntl lock.

    Per Catalog #131 + #128 + #245 sister canonical pattern: writes go
    through a unique ``.tmp.<uuid12>`` file then ``os.replace`` is NOT
    used here because the ledger is append-only (we append to the live
    file inside the lock). The lock serializes concurrent writers across
    processes; the append is atomic per POSIX line-buffered append
    semantics for files opened with ``O_APPEND``.
    """

    target = Path(path) if path is not None else HF_JOBS_CALL_ID_LEDGER_PATH
    lock_target = Path(lock_path) if lock_path is not None else _HF_JOBS_CALL_ID_LEDGER_LOCK
    _ensure_dirs(target)
    _ensure_dirs(lock_target)

    # Validate schema version + event_type at append time so a corrupt
    # writer cannot leak silently.
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: got {record.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r} (canonical Catalog #342 ledger)"
        )
    event_type = record.get("event_type")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type {event_type!r} not in {sorted(VALID_EVENT_TYPES)} "
            f"(canonical Catalog #342 ledger)"
        )

    serialized = _serialize_row(record) + "\n"

    # Touch lock file (separate from ledger) so the lock survives ledger
    # rotation / archival without losing exclusive serialization.
    with open(lock_target, "a") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with open(target, "a") as fh:
                fh.write(serialized)
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    return record


# --------------------------------------------------------------------------
# Loaders (lenient + strict per Catalog #138)
# --------------------------------------------------------------------------


def load_hf_jobs(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Lenient loader — skips malformed lines.

    Use for diagnostic / dashboard surfaces that can tolerate dropped rows.
    DO NOT use from a mutating helper (register / update) — those must use
    ``load_hf_jobs_strict`` per Catalog #138 fail-closed discipline.
    """

    target = Path(path) if path is not None else HF_JOBS_CALL_ID_LEDGER_PATH
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(target) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_hf_jobs_strict(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Strict loader — raises ``HFJobsLedgerCorruptError`` on any malformed row.

    Per Catalog #138 sister discipline. Use from any mutating helper that
    must see the full chronology (e.g., outcome registration that needs to
    verify the dispatched row exists; query helpers that promise
    completeness; backfill / quarantine flows).
    """

    target = Path(path) if path is not None else HF_JOBS_CALL_ID_LEDGER_PATH
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(target) as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HFJobsLedgerCorruptError(
                    f"malformed JSON at {target!s}:{lineno}: {exc!s}; "
                    f"quarantine via `mv {target!s} {target!s}.corrupt.{_now_iso()}` "
                    f"then re-run the canonical helper (Catalog #138 sister)."
                ) from exc
            if not isinstance(obj, dict):
                raise HFJobsLedgerCorruptError(
                    f"non-dict root at {target!s}:{lineno}: {type(obj).__name__}"
                )
            rows.append(obj)
    return rows


# --------------------------------------------------------------------------
# Public API — register_dispatched_hf_jobs_id + update_hf_jobs_outcome
# --------------------------------------------------------------------------


def register_dispatched_hf_jobs_id(
    *,
    hf_jobs_id: str,
    lane_id: str,
    label: str,
    dispatched_at_utc: str | None = None,
    flavor: str = "t4-small",
    expected_cost_usd: float | None = None,
    expected_axis: str = "cuda",
    recipe: str | None = None,
    max_seconds: int | None = None,
    mounted_code_git_head: str | None = None,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    hub_model_repo: str | None = None,
    hub_dataset_repo: str | None = None,
    hub_dataset_sha: str | None = None,
    upstream_snapshot_sha256: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append a ``dispatched`` event row immediately after ``run_uv_job(...)`` returns.

    Caller MUST invoke this BEFORE the local entrypoint exits so the
    canonical ledger captures the hf_jobs_id even if the per-dispatch
    sentinel files (job dashboard / local metadata.json) are not written
    (concurrent crash, sister-subagent edit, etc.).

    Returns the appended record (including server-side fields like
    ``written_at_utc`` / ``written_pid`` / ``written_host``).

    Sister of :func:`tac.deploy.modal.call_id_ledger.register_dispatched_call_id`.
    """

    if not isinstance(hf_jobs_id, str) or not hf_jobs_id.strip():
        raise ValueError("hf_jobs_id must be a non-empty string")
    if not isinstance(lane_id, str) or not lane_id.strip():
        raise ValueError("lane_id must be a non-empty string")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_DISPATCHED,
        "hf_jobs_id": hf_jobs_id,
        "lane_id": lane_id,
        "label": label,
        "platform": "hf_jobs",
        "flavor": flavor,
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
        "hub_model_repo": hub_model_repo,
        "hub_model_sha": None,
        "hub_dataset_repo": hub_dataset_repo,
        "hub_dataset_sha": hub_dataset_sha,
        "upstream_snapshot_sha256": upstream_snapshot_sha256,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    reserved = set(record.keys())
    for k, v in extra.items():
        if k in reserved:
            raise ValueError(
                f"extra kwarg {k!r} collides with a reserved schema field"
            )
        record[k] = v
    return _append_event_locked(record, path=path, lock_path=lock_path)


def update_hf_jobs_outcome(
    *,
    hf_jobs_id: str,
    status: str,
    event_type: str | None = None,
    harvested_at_utc: str | None = None,
    rc: int | None = None,
    elapsed_seconds: float | None = None,
    score: float | None = None,
    score_axis: str | None = None,
    archive_sha256: str | None = None,
    archive_bytes: int | None = None,
    evidence_grade: str | None = None,
    cost_actual_usd: float | None = None,
    hub_model_sha: str | None = None,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append an outcome event row referencing an existing hf_jobs_id.

    Per CLAUDE.md HISTORICAL_PROVENANCE + Catalog #110 / #132 — outcomes
    are NEW rows referencing the same ``hf_jobs_id``, NEVER mutations of
    the original ``dispatched`` row. The full lifecycle of an hf_jobs_id
    is the chronological sequence of all rows with that id.

    The ``event_type`` defaults to the value of ``status``; pass it
    explicitly to disambiguate (e.g. status="failed" event_type="stale" if
    the caller wants to record an outcome class distinct from the status
    token).

    Returns the appended record.

    Sister of :func:`tac.deploy.modal.call_id_ledger.update_call_id_outcome`.
    """

    if not isinstance(hf_jobs_id, str) or not hf_jobs_id.strip():
        raise ValueError("hf_jobs_id must be a non-empty string")
    if status not in TERMINAL_STATUSES:
        raise ValueError(
            f"status {status!r} not in {sorted(TERMINAL_STATUSES)} "
            f"(canonical Catalog #342 ledger terminal vocabulary)"
        )
    resolved_event_type = event_type if event_type is not None else status
    if resolved_event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type {resolved_event_type!r} not in {sorted(VALID_EVENT_TYPES)} "
            f"(canonical Catalog #342 ledger)"
        )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": resolved_event_type,
        "hf_jobs_id": hf_jobs_id,
        "status": status,
        "harvested_at_utc": harvested_at_utc or _now_iso(),
        "rc": rc,
        "elapsed_seconds": elapsed_seconds,
        "cost_actual_usd": cost_actual_usd,
        "score": score,
        "score_axis": score_axis,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "evidence_grade": evidence_grade,
        "hub_model_sha": hub_model_sha,
        "agent": agent,
        "subagent_id": subagent_id,
        "session_id": session_id,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    reserved = set(record.keys())
    for k, v in extra.items():
        if k in reserved:
            raise ValueError(
                f"extra kwarg {k!r} collides with a reserved schema field"
            )
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


# --------------------------------------------------------------------------
# Query helpers
# --------------------------------------------------------------------------


def query_by_hf_jobs_id(
    hf_jobs_id: str,
    *,
    path: Path | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Return all rows for the given hf_jobs_id in chronological order."""

    rows = load_hf_jobs_strict(path=path) if strict else load_hf_jobs(path=path)
    return [r for r in rows if r.get("hf_jobs_id") == hf_jobs_id]


def query_by_lane(
    lane_id: str,
    *,
    path: Path | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Return all rows for the given lane_id in chronological order."""

    rows = load_hf_jobs_strict(path=path) if strict else load_hf_jobs(path=path)
    return [r for r in rows if r.get("lane_id") == lane_id]


def latest_status_by_hf_jobs_id(
    hf_jobs_id: str,
    *,
    path: Path | None = None,
) -> str | None:
    """Return the most-recent status for the given hf_jobs_id, or None if absent."""

    rows = query_by_hf_jobs_id(hf_jobs_id, path=path)
    if not rows:
        return None
    return rows[-1].get("status")


# --------------------------------------------------------------------------
# Catalog #339 sister: fail-closed registration + poll-for-presence
# --------------------------------------------------------------------------
#
# Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable applied to
# HF Jobs: every ``HfApi().run_uv_job(...)`` dispatch MUST be followed by a
# scheduled harvest within the canonical TTL window. The harvester reads the
# canonical ledger. No ledger row = no harvest = invisible orphan dispatch.
#
# The Modal sister at ``tac.deploy.modal.call_id_ledger`` landed
# ``register_dispatched_call_id_fail_closed`` (Catalog #339) +
# ``poll_ledger_for_call_id`` (sister mitigation) 2026-05-19. This block
# mirrors that pattern for HF Jobs so the operator-authorize wrapper
# (``tools/operator_authorize.py::_dispatch_hf_jobs``) gets the same
# protection.


class HFJobsLedgerRegistrationFailedError(RuntimeError):
    """Raised when ``register_dispatched_hf_jobs_id_fail_closed`` cannot
    persist a dispatched-event row to the canonical HF Jobs ledger.

    Catalog #339 sister for HF Jobs. The canonical ``register_dispatched_hf_jobs_id``
    helper may raise ``ValueError`` (schema validation), ``OSError`` (disk
    full / permission denied), ``BlockingIOError`` (fcntl-lock contention),
    or sister errors from the underlying ``_append_event_locked``. Without
    fail-closed wrapping, the caller's ``try/except Exception: print``
    pattern silently breaks the harvester invariant per CLAUDE.md
    "Modal `.spawn()` HARVEST OR LOSE" applied to HF Jobs.

    Callers (e.g. ``tools/operator_authorize.py::_dispatch_hf_jobs``) MUST
    either let this propagate (so ``sys.exit`` becomes non-zero) OR wrap it
    AND invoke ``_write_last_resort_tmp_hf_jobs_ledger_dump`` so the
    harvester can recover via a future
    ``tools/harvest_hf_jobs_calls.py --recover-from-tmp`` sister CLI.

    Sister of ``HFJobsLedgerCorruptError`` (Catalog #138 strict-load) and
    ``tac.deploy.modal.call_id_ledger.LedgerRegistrationFailedError``.
    """


def _hf_jobs_ledger_recovery_tmp_dir() -> Path:
    """Return canonical recovery-tmp directory for hf_jobs ledger dumps."""

    return _repo_root() / ".omx" / "state" / "hf_jobs_call_id_ledger_recovery_tmp"


def _write_last_resort_tmp_hf_jobs_ledger_dump(record: dict[str, Any]) -> Path:
    """Atomically dump a dispatched-event record to a recovery directory.

    Used by ``register_dispatched_hf_jobs_id_fail_closed`` when the canonical
    fcntl-locked append fails. A future ``tools/harvest_hf_jobs_calls.py``
    sister can scan this directory and re-attempt the canonical append for
    each surviving row.

    Returns the path of the written tmp file. Best-effort: if THIS write
    also fails, ``HFJobsLedgerRegistrationFailedError`` is still raised by
    the caller so the operator sees the dispatch failed structurally.

    Sister of
    ``tac.deploy.modal.call_id_ledger._write_last_resort_tmp_ledger_dump``.
    """

    tmp_dir = _hf_jobs_ledger_recovery_tmp_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_name = f"recover_{record.get('hf_jobs_id', 'unknown')}_{uuid.uuid4().hex[:12]}.json"
    tmp_path = tmp_dir / tmp_name
    payload = json.dumps(record, sort_keys=True) + "\n"
    # Atomic write per Catalog #131 atomic-replace discipline so the
    # harvester never reads a half-written row.
    staging = tmp_path.with_suffix(tmp_path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
    staging.write_text(payload, encoding="utf-8")
    os.replace(staging, tmp_path)
    return tmp_path


def register_dispatched_hf_jobs_id_fail_closed(
    *,
    hf_jobs_id: str,
    lane_id: str,
    label: str,
    write_last_resort_dump: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fail-closed wrapper for :func:`register_dispatched_hf_jobs_id`.

    Catalog #339 sister for HF Jobs. Mirror of
    :func:`tac.deploy.modal.call_id_ledger.register_dispatched_call_id_fail_closed`.

    Unlike a legacy ``try: register_dispatched_hf_jobs_id(...) except
    Exception: print`` pattern, this wrapper REFUSES to swallow failures.
    On exception it:

    1. Constructs the dispatched-event record using the SAME schema as
       :func:`register_dispatched_hf_jobs_id` (so the recovery surface
       matches the canonical append).
    2. Best-effort dumps the record to
       ``.omx/state/hf_jobs_call_id_ledger_recovery_tmp/<hf_jobs_id>.json``
       so a future
       ``tools/harvest_hf_jobs_calls.py --recover-from-tmp`` sister can
       re-attempt the canonical append later (preserving the harvester
       invariant per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" applied
       to HF Jobs).
    3. Raises ``HFJobsLedgerRegistrationFailedError`` so the caller's exit
       path becomes non-zero, surfacing the failure to the operator instead
       of silently producing an invisible orphan HF Jobs dispatch.

    Returns the appended record on success (identical to the canonical
    helper).
    """

    try:
        return register_dispatched_hf_jobs_id(
            hf_jobs_id=hf_jobs_id,
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
                "hf_jobs_id": hf_jobs_id,
                "lane_id": lane_id,
                "label": label,
                "status": STATUS_DISPATCHED,
                "dispatched_at_utc": kwargs.get("dispatched_at_utc") or _now_iso(),
                "platform": "hf_jobs",
                "flavor": kwargs.get("flavor", "t4-small"),
                "expected_cost_usd": kwargs.get("expected_cost_usd"),
                "expected_axis": kwargs.get("expected_axis", "cuda"),
                "recipe": kwargs.get("recipe"),
                "max_seconds": kwargs.get("max_seconds"),
                "mounted_code_git_head": kwargs.get("mounted_code_git_head"),
                "agent": kwargs.get("agent", "claude"),
                "hub_model_repo": kwargs.get("hub_model_repo"),
                "hub_dataset_repo": kwargs.get("hub_dataset_repo"),
                "hub_dataset_sha": kwargs.get("hub_dataset_sha"),
                "upstream_snapshot_sha256": kwargs.get("upstream_snapshot_sha256"),
                "_recovery_reason": (
                    f"canonical_append_failed:{type(exc).__name__}:{str(exc)[:200]}"
                ),
            }
            try:
                last_resort_path = _write_last_resort_tmp_hf_jobs_ledger_dump(record)
            except Exception:  # pragma: no cover -- truly catastrophic
                last_resort_path = None
        msg = (
            f"Catalog #339 sister (HF Jobs) — register_dispatched_hf_jobs_id "
            f"failed for hf_jobs_id={hf_jobs_id!r} lane={lane_id!r}: "
            f"{type(exc).__name__}: {exc}. "
        )
        if last_resort_path is not None:
            msg += (
                f"Last-resort dump written to {last_resort_path}; harvester "
                "recovery via future `tools/harvest_hf_jobs_calls.py "
                "--recover-from-tmp` sister CLI."
            )
        else:
            msg += (
                "Last-resort dump ALSO failed; the HF Jobs dispatch is "
                "INVISIBLE to the harvester per CLAUDE.md 'Modal .spawn() "
                "HARVEST OR LOSE' non-negotiable applied to HF Jobs. "
                "Operator must manually look up the hf_jobs_id via the HF "
                "Hub job dashboard."
            )
        raise HFJobsLedgerRegistrationFailedError(msg) from exc


def poll_ledger_for_hf_jobs_id(
    hf_jobs_id: str,
    *,
    timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.5,
    path: Path | None = None,
) -> bool:
    """Poll the canonical ledger for ``hf_jobs_id`` presence within ``timeout``.

    Catalog #339 sister mitigation for the operator-authorize caller.
    Returns ``True`` once a row with ``hf_jobs_id`` appears in the canonical
    ledger (any event_type); ``False`` if the timeout elapses.

    Used by ``tools/operator_authorize.py::_dispatch_hf_jobs`` AFTER
    ``run_uv_job(...)`` returns AND the wrapper has invoked
    ``register_dispatched_hf_jobs_id_fail_closed``, to verify the ledger
    row actually landed BEFORE returning rc=0 to the operator. If the
    dispatch happened but the ledger row never appeared, the harvester is
    blind — caller must raise SystemExit instead of silently reporting
    success.

    Sister of :func:`tac.deploy.modal.call_id_ledger.poll_ledger_for_call_id`.
    """

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            rows = load_hf_jobs(path)
        except Exception:
            # Lenient-load failures here are NOT fatal — the harvester
            # uses the strict-load path which raises HFJobsLedgerCorruptError;
            # we just want presence, not strict correctness, so retry.
            rows = []
        for row in rows:
            if row.get("hf_jobs_id") == hf_jobs_id:
                return True
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_seconds)
    return False


__all__ = [
    "EVENT_DISPATCHED",
    "EVENT_FAILED",
    "EVENT_HARVESTED",
    "EVENT_STALE",
    "HF_JOBS_CALL_ID_LEDGER_PATH",
    "HFJobsLedgerCorruptError",
    "HFJobsLedgerRegistrationFailedError",
    "SCHEMA_VERSION",
    "STATUS_DISPATCHED",
    "TERMINAL_STATUSES",
    "VALID_EVENT_TYPES",
    "latest_status_by_hf_jobs_id",
    "load_hf_jobs",
    "load_hf_jobs_strict",
    "poll_ledger_for_hf_jobs_id",
    "query_by_hf_jobs_id",
    "query_by_lane",
    "register_dispatched_hf_jobs_id",
    "register_dispatched_hf_jobs_id_fail_closed",
    "update_hf_jobs_outcome",
]
