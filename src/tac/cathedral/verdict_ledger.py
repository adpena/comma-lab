# SPDX-License-Identifier: MIT
"""Cathedral autopilot consumer verdict ledger — fcntl-locked JSONL append-only.

Per T3 council prioritization 2026-05-19 rank #4 + Assumption-Adversary
"Cathedral autopilot activation is gated on Phase-2 evidence" CARGO-CULTED
verdict (commit ``79bd5695d``): the autopilot IS already runtime-activated
per Catalog #336/#337 (32 contract-compliant consumers fire per iteration
when ``invoke_cathedral_consumers_on_candidates`` is invoked from main()).
The orphan-signal failure mode that remained was that verdicts were emitted
to stdout only and lost when the process exited — no operator-facing
historical surface to query.

This module closes that surface per the canonical 4-layer pattern
(Catalog #245 Modal call_id ledger / Catalog #313 probe-outcomes ledger /
Catalog #333 codex-to-claude inbox / Catalog #344 canonical equations
registry):

  Layer 1 (THIS module): canonical fcntl-locked JSONL ledger helper.
  Layer 2 (sister): operator-facing CLI ``tools/cathedral_autopilot_activation_summary.py``.
  Layer 3 (sister): STRICT preflight gate (deferred per Catalog #299 quota
                    brake; the cathedral consumer contract Catalog #335
                    already provides the structural protection).
  Layer 4: wire-in inside ``tools/cathedral_autopilot_autonomous_loop.py::main``
           so every autopilot run that fires consumers persists the verdicts.

Schema (one event per JSONL row):

    {
        "schema_version": "cathedral_consumer_verdict_v1_20260519",
        "event_type": "consumer_invocation_batch",
        "session_id": "<utc-stamped uuid12 per invocation batch>",
        "panel_axis": "contest_cpu" | "contest_cuda",
        "rank_axis": "eig_per_dollar" | "predicted_score_delta",
        "consumer_count": 32,
        "consumer_names": [...],
        "candidates_invoked": 8,
        "candidate_ids": [...],
        "n_invocations": 256,
        "n_non_vacuous": 256,
        "n_errors": 0,
        "master_gradient_annotation_count": 8,
        "top_candidates_summary": [...],
        "invocations_summary_path": null OR "experiments/results/<...>/full.json",
        "written_at_utc": "...",
        "written_pid": 12345,
        "written_host": "...",
        "agent": "claude" | "codex" | "operator",
        "subagent_id": "...",
        "notes": "free-form context",
    }

The PER-INVOCATION detail (256 rows = consumer × candidate cross-product)
is NOT persisted to the ledger directly — that would inflate the ledger
quickly. Instead, the ledger row is a SUMMARY with optional ``invocations_summary_path``
pointing to a detail JSON under ``experiments/results/cathedral_autopilot_*/``.
The summary path follows the canonical Catalog #154 GC discipline (under
``experiments/results/`` with a tool-owned manifest).

Path discipline:
  * Ledger: ``.omx/state/cathedral_autopilot_consumer_verdicts.jsonl`` (committed
    per HISTORICAL_PROVENANCE per Catalog #110/#113).
  * Lock file: ``.omx/state/cathedral_autopilot_consumer_verdicts.jsonl.lock``
    (gitignored LIVE_STATE).
  * Bare writes to the path are refused by Catalog #131 sister gate
    (path is registered in ``_SHARED_STATE_PATH_MARKERS``).

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
every persisted row carries `score_claim=False` + `promotion_eligible=False`
+ `axis_tag="[predicted]"` so the ledger cannot leak into score/promotion
authority surfaces.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]

CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH = (
    REPO_ROOT / ".omx" / "state" / "cathedral_autopilot_consumer_verdicts.jsonl"
)
CATHEDRAL_CONSUMER_VERDICT_LEDGER_LOCK = (
    CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH.with_suffix(
        CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH.suffix + ".lock"
    )
)

# Lock acquisition timeout (seconds). Mirrors Catalog #245/#313/#344 default.
LOCK_TIMEOUT_SECONDS = 30

CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION = "cathedral_consumer_verdict_v1_20260519"

# Canonical event taxonomy.
EVENT_CONSUMER_INVOCATION_BATCH = "consumer_invocation_batch"
EVENT_OPERATOR_REVIEW = "operator_review"
EVENT_RECTIFIED_VERDICT = "rectified_verdict"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_CONSUMER_INVOCATION_BATCH,
        EVENT_OPERATOR_REVIEW,
        EVENT_RECTIFIED_VERDICT,
    }
)


class CathedralConsumerVerdictLedgerCorruptError(RuntimeError):
    """Raised when the verdict ledger is corrupt; mirrors Catalog #245 sister."""


_ledger_lock_depth_tls = threading.local()


def _get_lock_depth() -> int:
    return int(getattr(_ledger_lock_depth_tls, "depth", 0))


def _set_lock_depth(value: int) -> None:
    _ledger_lock_depth_tls.depth = int(value)


def _ledger_lock_held() -> bool:
    return _get_lock_depth() > 0


@contextlib.contextmanager
def _ledger_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the verdict ledger lock file.

    Re-entry is counted (depth > 1); fcntl is only re-acquired on 0->1.
    Mirrors Catalog #344 ``_registry_lock`` pattern.
    """
    p = lock_path or CATHEDRAL_CONSUMER_VERDICT_LEDGER_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    depth = _get_lock_depth()
    if depth > 0:
        _set_lock_depth(depth + 1)
        try:
            yield None
        finally:
            _set_lock_depth(_get_lock_depth() - 1)
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
                    raise TimeoutError(
                        f"could not acquire {p} within {LOCK_TIMEOUT_SECONDS}s"
                    ) from None
                time.sleep(0.05)
        _set_lock_depth(_get_lock_depth() + 1)
        try:
            yield fd
        finally:
            _set_lock_depth(_get_lock_depth() - 1)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _quarantine_corrupt_file(path: Path) -> Path:
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


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_verdict_events_lenient(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read all verdict events; skip malformed lines silently (read-only callers)."""
    p = path or CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


def load_verdict_ledger_strict(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Strict load for mutating callers; raises on corrupt state (Catalog #138)."""
    p = path or CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise CathedralConsumerVerdictLedgerCorruptError(
            f"cathedral verdict ledger at {p} could not be read: {exc}"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError as exc:
            raise CathedralConsumerVerdictLedgerCorruptError(
                f"cathedral verdict ledger at {p} line {lineno}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(r, dict):
            raise CathedralConsumerVerdictLedgerCorruptError(
                f"cathedral verdict ledger at {p} line {lineno}: non-dict root "
                f"(type={type(r).__name__})"
            )
        rows.append(r)
    return rows


def _validate_event_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION!r}"
        )
    if record.get("event_type") not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}"
        )
    session_id = record.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    # Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323
    # every verdict ledger row MUST be non-promotable by construction.
    for field_name in ("score_claim", "promotion_eligible"):
        if record.get(field_name) not in (False, None):
            raise ValueError(
                f"{field_name} must be False per Catalog #287/#323 non-promotable invariant"
            )


def _save_ledger(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomic write under lock — tmp + fsync + os.replace."""
    if not _ledger_lock_held():
        raise RuntimeError(
            "_save_ledger called WITHOUT holding _ledger_lock; this is a "
            "CONCURRENCY BUG per Catalog #140 (state writers own their lock end-to-end)."
        )
    p = path or CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH
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


def _append_event_locked(
    record: dict[str, Any],
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Append a single ledger event under fcntl lock."""
    _validate_event_record(record)
    p_path = path or CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH
    l_path = lock_path or CATHEDRAL_CONSUMER_VERDICT_LEDGER_LOCK

    with _ledger_lock(l_path):
        try:
            existing = load_verdict_ledger_strict(p_path)
        except CathedralConsumerVerdictLedgerCorruptError:
            _quarantine_corrupt_file(p_path)
            existing = []
        existing.append(record)
        _save_ledger(existing, p_path)
    return record


def append_consumer_invocation_batch(
    invocation_payload: Mapping[str, Any],
    *,
    panel_axis: str,
    rank_axis: str,
    candidate_ids: Iterable[str],
    top_candidates_summary: Iterable[Mapping[str, Any]] | None = None,
    invocations_summary_path: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
    notes: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Append a 'consumer_invocation_batch' event from an autopilot run.

    ``invocation_payload`` is the dict returned by
    ``tools/cathedral_autopilot_autonomous_loop.py::invoke_cathedral_consumers_on_candidates``
    (schema ``cathedral_consumer_invocation_v1_20260519``).

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323
    the persisted row carries ``score_claim=False`` + ``promotion_eligible=False``
    so the ledger cannot leak into score/promotion authority surfaces.
    """
    if not isinstance(invocation_payload, Mapping):
        raise TypeError(
            f"invocation_payload must be Mapping, got {type(invocation_payload).__name__}"
        )
    sid = session_id or f"cathedral_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"

    invocations = list(invocation_payload.get("invocations", []))
    n_invocations = len(invocations)
    n_errors = sum(1 for inv in invocations if "error" in inv)
    n_non_vacuous = sum(
        1 for inv in invocations
        if (
            "error" not in inv
            and (
                abs(float(inv.get("predicted_delta_adjustment", 0.0))) > 0.0
                or str(inv.get("rationale", "")).strip()
            )
        )
    )

    mg_annotations = list(invocation_payload.get("master_gradient_annotations", []))

    candidate_id_list = list(candidate_ids)

    top_summary = list(top_candidates_summary) if top_candidates_summary else []

    record = {
        "schema_version": CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION,
        "event_type": EVENT_CONSUMER_INVOCATION_BATCH,
        "session_id": sid,
        "panel_axis": panel_axis,
        "rank_axis": rank_axis,
        "consumer_count": int(invocation_payload.get("consumer_count", 0)),
        "consumer_names": list(invocation_payload.get("consumer_names", [])),
        "candidates_invoked": int(invocation_payload.get("candidates_invoked", 0)),
        "candidate_ids": candidate_id_list,
        "n_invocations": n_invocations,
        "n_non_vacuous": n_non_vacuous,
        "n_errors": n_errors,
        "master_gradient_annotation_count": len(mg_annotations),
        "top_candidates_summary": top_summary,
        "invocations_summary_path": invocations_summary_path,
        "written_at_utc": _utc_now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
        "agent": agent or "claude",
        "subagent_id": subagent_id,
        "notes": notes,
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted, cathedral consumer invocation]",
    }
    return _append_event_locked(record, path=path, lock_path=lock_path)


def query_sessions(
    *,
    since_utc: str | None = None,
    until_utc: str | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all ledger rows whose ``written_at_utc`` falls in the range.

    ``since_utc`` / ``until_utc`` are ISO strings (``YYYY-MM-DDTHH:MM:SSZ``);
    None means unbounded.
    """
    rows = load_verdict_events_lenient(path)
    if since_utc is None and until_utc is None:
        return rows
    out: list[dict[str, Any]] = []
    for r in rows:
        ts = r.get("written_at_utc")
        if not isinstance(ts, str):
            continue
        if since_utc and ts < since_utc:
            continue
        if until_utc and ts > until_utc:
            continue
        out.append(r)
    return out


def query_latest_session(
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Return the most-recent invocation batch row (None if ledger empty)."""
    rows = load_verdict_events_lenient(path)
    batches = [r for r in rows if r.get("event_type") == EVENT_CONSUMER_INVOCATION_BATCH]
    if not batches:
        return None
    return sorted(batches, key=lambda r: r.get("written_at_utc", ""))[-1]


def query_consumer_activity_summary(
    *,
    since_utc: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate per-consumer activity counts across recent sessions.

    Returns a dict mapping ``consumer_name`` to
    ``{session_count, candidate_count_total, last_seen_utc}``.
    """
    rows = query_sessions(since_utc=since_utc, path=path)
    consumer_stats: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("event_type") != EVENT_CONSUMER_INVOCATION_BATCH:
            continue
        ts = r.get("written_at_utc", "")
        candidates_invoked = int(r.get("candidates_invoked", 0))
        for name in r.get("consumer_names", []):
            entry = consumer_stats.setdefault(
                name,
                {"session_count": 0, "candidate_count_total": 0, "last_seen_utc": ""},
            )
            entry["session_count"] += 1
            entry["candidate_count_total"] += candidates_invoked
            if ts > entry["last_seen_utc"]:
                entry["last_seen_utc"] = ts
    return consumer_stats


__all__ = [
    "CATHEDRAL_CONSUMER_VERDICT_LEDGER_PATH",
    "CATHEDRAL_CONSUMER_VERDICT_LEDGER_LOCK",
    "CATHEDRAL_CONSUMER_VERDICT_SCHEMA_VERSION",
    "EVENT_CONSUMER_INVOCATION_BATCH",
    "EVENT_OPERATOR_REVIEW",
    "EVENT_RECTIFIED_VERDICT",
    "VALID_EVENT_TYPES",
    "CathedralConsumerVerdictLedgerCorruptError",
    "load_verdict_events_lenient",
    "load_verdict_ledger_strict",
    "append_consumer_invocation_batch",
    "query_sessions",
    "query_latest_session",
    "query_consumer_activity_summary",
]
