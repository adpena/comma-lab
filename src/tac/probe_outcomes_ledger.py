# SPDX-License-Identifier: MIT
"""Canonical probe-outcomes ledger — fcntl-locked JSONL append-only audit trail.

Operator directive 2026-05-16 (PROBE-OUTCOMES-BAKE-IN subagent):
*"bake in the FULL 4-layer canonical pattern per the Catalog #245 Modal call_id
ledger exemplar so probe-disambiguator verdicts are queryable across sessions
and gating dispatch BEFORE we re-run something an existing adjudicated probe
already settled."*

Why a new canonical helper?
───────────────────────────
Pre-landing, probe-disambiguator outcomes (per CLAUDE.md "Meta-Lagrangian /
Pareto solver" non-negotiable + "Subagent coherence-by-default" probe-disambiguator
hook + Catalog #292 per-deliberation assumption surfacing) were CAPTURED but
SCATTERED across many surfaces with no single index:

- ``.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md`` (per-probe markdown)
- ``.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.json`` (per-probe JSON)
- ``.omx/research/*disambig*.md`` (sister disambiguation memos)
- ``.omx/research/*probe*.md`` (60+ probe outcomes across the contest)
- ``.omx/research/l5_v2_probe_*`` (Time-Traveler L5 v2 probe corpus)

There was NO single queryable source-of-truth answering "has this substrate
already been adjudicated by a probe within the last 30 days, and if so what
was the verdict?" Subagent dispatchers can therefore re-fire dispatch
recommendations on substrates where an INDEPENDENT / DEFER / KILL verdict
already settled the question — burning paid GPU re-measuring an answer the
apparatus already has.

This module is the canonical primary index. It mirrors:

- ``tac.deploy.modal.call_id_ledger`` — the Catalog #245 exemplar this gate
  is the per-method bake-in of (canonical-helper-share when serves; per
  CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" we adopt the
  canonical fcntl-locked JSONL pattern because the bug class is identical:
  scattered evidence with no queryable index).
- ``tac.deploy.lightning.active_jobs_state`` — sister fcntl-locked JSONL helper
- ``tac.continual_learning.posterior_update_locked`` — Catalog #128
- ``tac.council_continual_learning`` — Catalog #300 council deliberation ledger
- ``tools/subagent_checkpoint.py`` — Catalog #206 JSONL append-only with fcntl

Schema (one event per JSONL row)
────────────────────────────────
Every row is one event in a probe outcome's lifecycle. The ledger is
APPEND-ONLY per CLAUDE.md "HISTORICAL_PROVENANCE" classification + Catalog
#110 / #113 / #132 (locked writes preserve deletions) — rows are NEVER
mutated; new verdicts become NEW rows referencing the same ``probe_id``. The
full lifecycle is reconstructable by querying ``query_by_probe_id(probe_id)``
which returns chronological rows.

Required event_types::

    - "adjudicated"     — initial verdict from probe-disambiguator run
    - "ratified"        — operator/council confirmed verdict, blocker persists
    - "superseded"      — new evidence supersedes prior verdict (sister probe re-ran)
    - "expired"         — outcome aged past 30-day staleness window (advisory)
    - "operator_override" — operator explicitly cleared a blocking outcome

Required verdict tokens (verdict semantics independent of event_type)::

    - "INDEPENDENT"  — probe metric indicates no signal in conditioning axis (BLOCKS dispatch)
    - "KILL"         — probe shows substrate fundamentally falsified (BLOCKS dispatch)
    - "DEFER"        — research-pending; alternative reducers warrant trial (BLOCKS dispatch)
    - "PROMOTE"      — probe shows clear positive signal; PROCEED authorized
    - "PROCEED"      — probe shows necessary conditioning; PROCEED authorized
    - "PARTIAL"      — partial-signal regime; operator/council adjudication required
    - "OPERATOR_REVIEW_REQUIRED" — probe-outcome ambiguous (default for unclear backfill)

Required blocker_status tokens::

    - "blocking"  — outcome ACTIVELY BLOCKS dispatch (gate fires at preflight + operator-authorize)
    - "advisory"  — outcome recorded for posterity; informational only
    - "expired"   — outcome aged past staleness window (auto-transition; non-blocking)

Schema fields per row::

    {
        "schema_version": 1,
        "event_type": "adjudicated" | ...,
        "probe_id": "atw_v2_d4_h_latent_given_scorer_class_20260516",
        "substrate": "atw_codec_v2",
        "recipe_path": ".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml",
        "probe_kind": "h_latent_given_scorer_class",
        "verdict": "INDEPENDENT",
        "metric_name": "mutual_information_bits_per_symbol",
        "metric_value": 0.006385,
        "threshold": 0.5,
        "threshold_token": "MEANINGFUL_CONDITIONING",
        "evidence_path": ".omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md",
        "next_action": "do_not_dispatch_atw_v2_phase2_from_this_signal",
        "blocker_status": "blocking",
        "dispatched_at_utc": "2026-05-16T22:47:41Z",
        "adjudicated_at_utc": "2026-05-16T22:47:41Z",
        "expires_at_utc": "2026-06-15T22:47:41Z",  # adjudicated + 30 days
        "written_at_utc": "2026-05-16T22:47:42.123Z",
        "written_pid": 12345,
        "written_host": "macbook-air",
        "agent": "claude" | "codex" | "operator",
        "subagent_id": "atw_v2_d4_probe_20260516",
        "session_id": null,
        "notes": "free-form operator/agent context",
    }

Path discipline
───────────────
- ``PROBE_OUTCOMES_LEDGER_PATH`` = ``.omx/state/probe_outcomes.jsonl``
  COMMITTED per HISTORICAL_PROVENANCE classification (Catalog #113).
- The lock file ``.lock`` is gitignored (LIVE_STATE).
- ``.tmp.<uuid12>`` files are gitignored (LIVE_STATE).
- Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" — the ledger
  lives at ``.omx/state/`` under the repo root.

Bare writes are FORBIDDEN
─────────────────────────
Per CLAUDE.md Catalog #131 (``check_no_bare_writes_to_shared_state``) every
write to ``PROBE_OUTCOMES_LEDGER_PATH`` MUST acquire ``fcntl.flock(LOCK_EX)``
on the lock file + use a unique ``.tmp.<uuid12>`` + ``os.replace``. The
public API (``register_probe_outcome`` / ``update_probe_outcome``) does this;
direct ``open(...).write(...)`` outside the canonical helper is refused by
Catalog #131 sister gate (this module's path is registered there).

Catalog #313 wires the canonical STRICT preflight gate
``check_dispatch_target_has_no_predecessor_adjudicated_outcome`` over this
module's ``latest_blocking_outcome_by_recipe`` query helper — refusing
operator-authorize dispatch on a recipe whose substrate has a recent blocking
adjudicated verdict in ``{INDEPENDENT, KILL, DEFER}``.

Memory: feedback_probe_outcomes_canonical_ledger_landed_20260516.md.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_OUTCOMES_LEDGER_PATH = REPO_ROOT / ".omx" / "state" / "probe_outcomes.jsonl"
PROBE_OUTCOMES_LEDGER_LOCK = PROBE_OUTCOMES_LEDGER_PATH.with_suffix(
    PROBE_OUTCOMES_LEDGER_PATH.suffix + ".lock"
)

# Schema version pinned for forward compatibility.
SCHEMA_VERSION = 1

# Lock acquisition timeout (seconds). Single-row appends are <10ms; 30s is
# generous even under heavy fan-out contention from sibling subagents.
LOCK_TIMEOUT_SECONDS = 30

# Staleness window — outcomes older than this default age are considered
# expired (advisory only); gate-time queries can override the window.
DEFAULT_STALENESS_WINDOW_DAYS = 30

# Canonical event taxonomy. A probe_id may have many events; the latest in
# chronological order is the current state.
EVENT_ADJUDICATED = "adjudicated"
EVENT_RATIFIED = "ratified"
EVENT_SUPERSEDED = "superseded"
EVENT_EXPIRED = "expired"
EVENT_OPERATOR_OVERRIDE = "operator_override"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_ADJUDICATED,
        EVENT_RATIFIED,
        EVENT_SUPERSEDED,
        EVENT_EXPIRED,
        EVENT_OPERATOR_OVERRIDE,
    }
)

# Canonical verdict taxonomy.
VERDICT_INDEPENDENT = "INDEPENDENT"
VERDICT_KILL = "KILL"
VERDICT_DEFER = "DEFER"
VERDICT_PROMOTE = "PROMOTE"
VERDICT_PROCEED = "PROCEED"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_OPERATOR_REVIEW_REQUIRED = "OPERATOR_REVIEW_REQUIRED"
# Slot A NEGATIVE-RESULTS-AUDIT-V2 FIX O3 (2026-05-28): infrastructure-failure
# verdict semantically distinct from INDEPENDENT (paradigm-disambiguation empirical
# null) per the F19 segfault empirical anchor + canonical
# `paradigm_vs_implementation_falsification_distinct_from_infrastructure_failure_v1`
# anti-pattern lineage. An INDEPENDENT verdict says "the probe ran cleanly and
# the metric indicates no signal in the conditioning axis"; an
# INFRASTRUCTURE_FAILURE verdict says "the probe could not be measured because
# the code/runtime/dispatch infrastructure crashed (segfault, OOM, malformed
# input, NaN propagation, etc.)" — fundamentally different signal classes that
# the prior conflation poisoned downstream paradigm-vs-implementation
# classification per Catalog #307. Per CLAUDE.md "Apples-to-apples evidence
# discipline" non-negotiable: the verdict taxonomy must distinguish them so the
# operator + autopilot ranker know to re-probe with corrected infrastructure
# vs to accept paradigm-disambiguation null. F19 anchor:
# ``v4_hand_rolled_faiss_ivf_pq_m2_ksub128_topk3_600pair_segfault_20260518``
# SEGFAULTS at both 100-pair smoke AND 600-pair full on REAL A1 softmax data —
# the verdict is NOT paradigm-INDEPENDENT (the V1/V2 sister Faiss codecs DID
# disambiguate Shannon transition zones at MI=2.46) but rather a hand-rolled-V4
# infrastructure-implementation failure. The canonical extinction sister to
# this verdict class is the F19-style probe being re-routed to a non-segfaulting
# Faiss variant (V5/V6/V7/V8) rather than treated as a paradigm-KILL.
VERDICT_INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"

VALID_VERDICTS = frozenset(
    {
        VERDICT_INDEPENDENT,
        VERDICT_KILL,
        VERDICT_DEFER,
        VERDICT_PROMOTE,
        VERDICT_PROCEED,
        VERDICT_PARTIAL,
        VERDICT_OPERATOR_REVIEW_REQUIRED,
        VERDICT_INFRASTRUCTURE_FAILURE,
    }
)

# Verdicts that BLOCK dispatch when blocker_status is "blocking" and the
# outcome is within the staleness window. Per CLAUDE.md "Forbidden premature
# KILL without research exhaustion": these verdicts are research-deferrals not
# kills; the gate REFUSES re-dispatch but does NOT mark the lane killed.
# INFRASTRUCTURE_FAILURE is BLOCKING because re-running the exact same
# infrastructure-broken probe again would just re-crash and waste paid GPU
# spend; resolution requires either (a) sister probe with corrected
# infrastructure, OR (b) operator override per Catalog #313 paired-env bypass.
BLOCKING_VERDICTS = frozenset({VERDICT_INDEPENDENT, VERDICT_KILL, VERDICT_DEFER, VERDICT_INFRASTRUCTURE_FAILURE})

# Blocker status taxonomy.
BLOCKER_STATUS_BLOCKING = "blocking"
BLOCKER_STATUS_ADVISORY = "advisory"
BLOCKER_STATUS_EXPIRED = "expired"

VALID_BLOCKER_STATUSES = frozenset(
    {BLOCKER_STATUS_BLOCKING, BLOCKER_STATUS_ADVISORY, BLOCKER_STATUS_EXPIRED}
)


class ProbeOutcomesLedgerCorruptError(RuntimeError):
    """Raised when the probe-outcomes ledger file is corrupt and cannot be
    safely appended to.

    Sister of ``CallIdLedgerCorruptError`` (Catalog #245 strict-load discipline)
    + ``ActiveJobsCorruptError`` (Catalog #138). The append helpers raise this
    rather than silently overwriting the bad file, which would erase the
    historical audit trail of every adjudicated probe outcome. The corrupt
    file is QUARANTINED to ``.corrupt.<utc>`` so the operator can inspect;
    the next ``register_probe_outcome`` creates a fresh empty ledger.
    """


# Thread-local lock-held depth (Catalog #140 sister pattern). Pairs with
# ``_ledger_lock_held()`` so any direct-write helper can refuse calls that
# bypass the canonical locked path.
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
    p = lock_path or PROBE_OUTCOMES_LEDGER_LOCK
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
                    raise TimeoutError(
                        f"could not acquire {p} within {LOCK_TIMEOUT_SECONDS}s"
                    ) from None
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
    """Return UTC timestamp in ISO-8601 format with microsecond precision."""
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


def load_outcomes(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the probe-outcomes ledger or return empty list (LENIENT loader).

    Safe to call without holding the lock — readers see a stable snapshot
    because writers append under the lock + each write is atomic at the
    line level. Concurrent readers may observe a partial trailing line if
    a write is in flight; this loader silently skips malformed lines for
    backward compatibility with consumers that just want a best-effort view.

    LENIENT semantics: malformed JSON lines are SKIPPED with no error. This
    is appropriate for read-only callers (dashboards, query helpers) but
    UNSAFE for mutating callers — see ``load_outcomes_strict`` per Catalog
    #138 strict-load discipline.
    """
    p = path or PROBE_OUTCOMES_LEDGER_PATH
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


def load_outcomes_strict(path: Path | None = None) -> list[dict[str, Any]]:
    """Strict load for mutating callers — raises ProbeOutcomesLedgerCorruptError
    on corrupt state.

    MUST be called from inside ``_ledger_lock`` by mutating callers. The
    mutating helpers use this so a malformed ledger is NEVER silently appended
    to (which would corrupt the audit trail). Returns ``[]`` if the path does
    not exist (the empty-ledger bootstrap case is normal and not corruption).

    Raises:
        ProbeOutcomesLedgerCorruptError: when the file exists and contains any
            malformed JSON line OR any row whose root is not a dict.
    """
    p = path or PROBE_OUTCOMES_LEDGER_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProbeOutcomesLedgerCorruptError(
            f"probe-outcomes ledger at {p} could not be read: {exc}"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ProbeOutcomesLedgerCorruptError(
                f"probe-outcomes ledger at {p} line {lineno} contains invalid "
                f"JSON: {exc}. Mutating writes are refused to preserve the "
                "audit trail. Operator: inspect the file, then either fix it "
                "in place OR move it aside; the next register_probe_outcome "
                "will create a fresh empty file."
            ) from exc
        if not isinstance(row, dict):
            raise ProbeOutcomesLedgerCorruptError(
                f"probe-outcomes ledger at {p} line {lineno} has non-dict root "
                f"(type={type(row).__name__}); expected JSON object."
            )
        rows.append(row)
    return rows


def _validate_event_record(record: dict[str, Any]) -> None:
    """Sanity-check a record before append. Raises ValueError on bad input."""
    probe_id = record.get("probe_id")
    if not isinstance(probe_id, str) or not probe_id.strip():
        raise ValueError("probe_id must be a non-empty string")
    if any(c in probe_id for c in ("\n", "\t", "\x1f")):
        raise ValueError("probe_id must not contain newlines/tabs/0x1f")

    substrate = record.get("substrate")
    if not isinstance(substrate, str) or not substrate.strip():
        raise ValueError("substrate must be a non-empty string")

    event_type = record.get("event_type")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}, got {event_type!r}"
        )

    verdict = record.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"verdict must be one of {sorted(VALID_VERDICTS)!r}, got {verdict!r}"
        )

    blocker_status = record.get("blocker_status")
    if blocker_status not in VALID_BLOCKER_STATUSES:
        raise ValueError(
            f"blocker_status must be one of {sorted(VALID_BLOCKER_STATUSES)!r}, "
            f"got {blocker_status!r}"
        )

    schema_version = record.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {schema_version!r}"
        )


def _save_ledger(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomic write — unique tmp + fsync + os.replace.

    Runtime-asserts the caller holds ``_ledger_lock``. Per CLAUDE.md
    "Comment-only contracts are FORBIDDEN" + Catalog #140
    (state writers own their lock end-to-end).
    """
    if not _ledger_lock_held():
        raise RuntimeError(
            "_save_ledger called WITHOUT holding _ledger_lock. This is a "
            "CONCURRENCY BUG: concurrent appends can silently drop rows. "
            "Use _append_event_locked / register_probe_outcome / "
            "update_probe_outcome which own the full lock-load-append-save cycle."
        )
    p = path or PROBE_OUTCOMES_LEDGER_PATH
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
    quarantine_on_corrupt: bool = True,
) -> dict[str, Any]:
    """Append a single event record under fcntl lock.

    Mirrors the Catalog #245 ``_append_event_locked`` pattern (full
    load-validate-append-save). Per CLAUDE.md "Locked writes preserve
    deletions" (Catalog #132): APPEND-ONLY — existing rows preserved verbatim;
    only new rows are added.
    """
    _validate_event_record(record)
    p_path = path or PROBE_OUTCOMES_LEDGER_PATH
    l_path = lock_path or PROBE_OUTCOMES_LEDGER_LOCK

    with _ledger_lock(l_path):
        try:
            rows = load_outcomes_strict(p_path)
        except ProbeOutcomesLedgerCorruptError as exc:
            if quarantine_on_corrupt:
                quarantine_path = _quarantine_corrupt_file(p_path)
                raise ProbeOutcomesLedgerCorruptError(
                    f"probe-outcomes ledger at {p_path} was corrupt; "
                    f"quarantined to {quarantine_path}. Append refused; "
                    "operator must repair (see ProbeOutcomesLedgerCorruptError "
                    "docstring)."
                ) from exc
            raise

        new_rows = [*rows, record]
        _save_ledger(new_rows, p_path)
        return record


def _compute_expires_at_utc(
    adjudicated_at_utc: str,
    *,
    staleness_window_days: int = DEFAULT_STALENESS_WINDOW_DAYS,
) -> str:
    """Compute the expires_at_utc timestamp = adjudicated + staleness window.

    The gate uses this field to auto-transition outcomes to ``expired`` when
    the current time is past expires_at_utc. Per CLAUDE.md "Substrate
    retirement discipline" the 30-day default aligns with the L1 SCAFFOLD
    staleness window (Catalog #298).
    """
    try:
        adjudicated = _dt.datetime.fromisoformat(adjudicated_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"adjudicated_at_utc must be ISO-8601: {adjudicated_at_utc!r} ({exc})"
        ) from exc
    expires = adjudicated + _dt.timedelta(days=staleness_window_days)
    return expires.isoformat(timespec="microseconds").replace("+00:00", "Z")


# ─────────────────────────────────────────────────────────────────────────
# Public API — register_probe_outcome + update_probe_outcome
# ─────────────────────────────────────────────────────────────────────────


def register_probe_outcome(
    *,
    probe_id: str,
    substrate: str,
    recipe_path: str | None,
    probe_kind: str,
    verdict: str,
    metric_name: str,
    metric_value: float,
    threshold: float | None = None,
    threshold_token: str | None = None,
    evidence_path: str | None = None,
    next_action: str | None = None,
    blocker_status: str | None = None,
    dispatched_at_utc: str | None = None,
    adjudicated_at_utc: str | None = None,
    staleness_window_days: int = DEFAULT_STALENESS_WINDOW_DAYS,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    notes: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append an ``adjudicated`` event row immediately after a probe-disambiguator
    run produces a verdict.

    The blocker_status defaults to ``blocking`` when ``verdict`` is in
    ``BLOCKING_VERDICTS`` (INDEPENDENT / KILL / DEFER), else ``advisory``.
    The expires_at_utc is auto-computed = adjudicated + staleness_window_days.

    Returns the appended record (including server-side fields like
    ``written_at_utc`` / ``expires_at_utc`` / ``written_pid`` / ``written_host``).
    """
    if not isinstance(probe_id, str) or not probe_id.strip():
        raise ValueError("probe_id must be a non-empty string")
    if not isinstance(substrate, str) or not substrate.strip():
        raise ValueError("substrate must be a non-empty string")
    if not isinstance(probe_kind, str) or not probe_kind.strip():
        raise ValueError("probe_kind must be a non-empty string")
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"verdict must be one of {sorted(VALID_VERDICTS)!r}, got {verdict!r}"
        )

    resolved_adjudicated_at = adjudicated_at_utc or _now_iso()
    resolved_blocker_status = blocker_status or (
        BLOCKER_STATUS_BLOCKING if verdict in BLOCKING_VERDICTS else BLOCKER_STATUS_ADVISORY
    )
    if resolved_blocker_status not in VALID_BLOCKER_STATUSES:
        raise ValueError(
            f"blocker_status must be one of {sorted(VALID_BLOCKER_STATUSES)!r}, "
            f"got {resolved_blocker_status!r}"
        )

    expires_at_utc = _compute_expires_at_utc(
        resolved_adjudicated_at,
        staleness_window_days=staleness_window_days,
    )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EVENT_ADJUDICATED,
        "probe_id": probe_id,
        "substrate": substrate,
        "recipe_path": recipe_path,
        "probe_kind": probe_kind,
        "verdict": verdict,
        "metric_name": metric_name,
        "metric_value": float(metric_value) if metric_value is not None else None,
        "threshold": float(threshold) if threshold is not None else None,
        "threshold_token": threshold_token,
        "evidence_path": evidence_path,
        "next_action": next_action,
        "blocker_status": resolved_blocker_status,
        "dispatched_at_utc": dispatched_at_utc,
        "adjudicated_at_utc": resolved_adjudicated_at,
        "expires_at_utc": expires_at_utc,
        "staleness_window_days": staleness_window_days,
        "agent": agent,
        "subagent_id": subagent_id,
        "session_id": session_id,
        "notes": notes,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    # Reserved-field collision check (preserve audit-trail integrity).
    reserved = set(record.keys())
    for k, v in extra.items():
        if k in reserved:
            raise ValueError(f"extra kwarg {k!r} collides with a reserved schema field")
        record[k] = v
    return _append_event_locked(record, path=path, lock_path=lock_path)


def update_probe_outcome(
    *,
    probe_id: str,
    event_type: str,
    verdict: str | None = None,
    blocker_status: str | None = None,
    notes: str | None = None,
    agent: str = "claude",
    subagent_id: str | None = None,
    session_id: str | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append an outcome-update event row referencing an existing probe_id.

    Per CLAUDE.md "HISTORICAL_PROVENANCE" + Catalog #110 / #132 — updates are
    NEW rows referencing the same ``probe_id``, NEVER mutations of the original
    ``adjudicated`` row. The full lifecycle is the chronological sequence of
    all rows with that ``probe_id`` value.

    Use cases:
      - ratified: operator/council confirmed the verdict
      - superseded: a sister probe ran and superseded this outcome
      - expired: outcome aged past staleness window (auto-transition)
      - operator_override: operator explicitly cleared a blocking outcome
        (must pair with explicit rationale in ``notes``)
    """
    if not isinstance(probe_id, str) or not probe_id.strip():
        raise ValueError("probe_id must be a non-empty string")
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}, got {event_type!r}"
        )

    # Look up the latest row for the probe_id to inherit fields not explicitly
    # overridden by this update. Provides denormalization so consumers can see
    # the complete state from any row.
    existing = latest_outcome_by_probe_id(probe_id, path=path)
    if existing is None:
        raise ValueError(
            f"probe_id {probe_id!r} has no prior adjudicated event; "
            "call register_probe_outcome() first"
        )

    resolved_verdict = verdict if verdict is not None else existing.get("verdict")
    if resolved_verdict not in VALID_VERDICTS:
        raise ValueError(
            f"verdict must be one of {sorted(VALID_VERDICTS)!r}, got {resolved_verdict!r}"
        )

    # Auto-transition blocker_status on event_type=expired and operator_override
    # unless caller explicitly passes a value.
    if blocker_status is None:
        if event_type == EVENT_EXPIRED:
            resolved_blocker_status = BLOCKER_STATUS_EXPIRED
        elif event_type == EVENT_OPERATOR_OVERRIDE:
            resolved_blocker_status = BLOCKER_STATUS_ADVISORY
        else:
            resolved_blocker_status = existing.get("blocker_status", BLOCKER_STATUS_ADVISORY)
    else:
        resolved_blocker_status = blocker_status
    if resolved_blocker_status not in VALID_BLOCKER_STATUSES:
        raise ValueError(
            f"blocker_status must be one of {sorted(VALID_BLOCKER_STATUSES)!r}, "
            f"got {resolved_blocker_status!r}"
        )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_type": event_type,
        "probe_id": probe_id,
        "substrate": existing.get("substrate"),
        "recipe_path": existing.get("recipe_path"),
        "probe_kind": existing.get("probe_kind"),
        "verdict": resolved_verdict,
        "metric_name": existing.get("metric_name"),
        "metric_value": existing.get("metric_value"),
        "threshold": existing.get("threshold"),
        "threshold_token": existing.get("threshold_token"),
        "evidence_path": existing.get("evidence_path"),
        "next_action": existing.get("next_action"),
        "blocker_status": resolved_blocker_status,
        "dispatched_at_utc": existing.get("dispatched_at_utc"),
        "adjudicated_at_utc": existing.get("adjudicated_at_utc"),
        "expires_at_utc": existing.get("expires_at_utc"),
        "staleness_window_days": existing.get("staleness_window_days"),
        "agent": agent,
        "subagent_id": subagent_id,
        "session_id": session_id,
        "notes": notes,
        "written_at_utc": _now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    reserved = set(record.keys())
    for k, v in extra.items():
        if k in reserved:
            raise ValueError(f"extra kwarg {k!r} collides with a reserved schema field")
        record[k] = v

    return _append_event_locked(record, path=path, lock_path=lock_path)


# ─────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeOutcomeView:
    """Read-only view over a probe-outcome's latest event.

    Returned by ``latest_blocking_outcome_by_recipe`` /
    ``latest_blocking_outcome_by_substrate`` to give callers a typed handle on
    the verdict + evidence_path + blocker_status + expires_at_utc for the
    operator-authorize fatal-error message.
    """

    probe_id: str
    substrate: str
    recipe_path: str | None
    probe_kind: str
    verdict: str
    metric_name: str
    metric_value: float | None
    threshold: float | None
    threshold_token: str | None
    evidence_path: str | None
    next_action: str | None
    blocker_status: str
    adjudicated_at_utc: str
    expires_at_utc: str | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ProbeOutcomeView:
        return cls(
            probe_id=str(row.get("probe_id", "")),
            substrate=str(row.get("substrate", "")),
            recipe_path=row.get("recipe_path"),
            probe_kind=str(row.get("probe_kind", "")),
            verdict=str(row.get("verdict", "")),
            metric_name=str(row.get("metric_name", "")),
            metric_value=row.get("metric_value"),
            threshold=row.get("threshold"),
            threshold_token=row.get("threshold_token"),
            evidence_path=row.get("evidence_path"),
            next_action=row.get("next_action"),
            blocker_status=str(row.get("blocker_status", "")),
            adjudicated_at_utc=str(row.get("adjudicated_at_utc", "")),
            expires_at_utc=row.get("expires_at_utc"),
        )


def query_by_probe_id(
    probe_id: str,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all events for ``probe_id`` in chronological JSONL append order."""
    if not isinstance(probe_id, str) or not probe_id.strip():
        raise ValueError("probe_id must be a non-empty string")
    return [r for r in load_outcomes(path) if r.get("probe_id") == probe_id]


def query_by_substrate(
    substrate: str,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all events for ``substrate`` in chronological JSONL append order."""
    if not isinstance(substrate, str) or not substrate.strip():
        raise ValueError("substrate must be a non-empty string")
    return [r for r in load_outcomes(path) if r.get("substrate") == substrate]


def query_by_recipe(
    recipe_path: str | Path,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return all events whose ``recipe_path`` matches the given recipe.

    Recipe paths are compared as repository-relative strings; absolute paths
    are normalized to relative form by stripping the repo root prefix.
    """
    if not isinstance(recipe_path, (str, Path)):
        raise ValueError("recipe_path must be a string or Path")
    needle = _normalize_recipe_path(str(recipe_path))
    matches: list[dict[str, Any]] = []
    for row in load_outcomes(path):
        candidate = row.get("recipe_path")
        if not isinstance(candidate, str):
            continue
        if _normalize_recipe_path(candidate) == needle:
            matches.append(row)
    return matches


def _normalize_recipe_path(p: str) -> str:
    """Normalize a recipe path string to a stable repo-relative form."""
    p = p.strip()
    # Strip the repo root prefix if present so absolute paths and relative
    # paths produce the same identity.
    repo_str = str(REPO_ROOT).rstrip("/")
    if p.startswith(repo_str + "/"):
        return p[len(repo_str) + 1 :]
    return p


def latest_outcome_by_probe_id(
    probe_id: str,
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Return the latest event row for ``probe_id`` or None.

    "Latest" is the last row in JSONL append order with this probe_id.
    """
    rows = query_by_probe_id(probe_id, path=path)
    return rows[-1] if rows else None


def latest_outcome_by_substrate(
    substrate: str,
    *,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Return the latest event row for ``substrate`` (latest by chronological
    append order across all probe_ids targeting this substrate).
    """
    rows = query_by_substrate(substrate, path=path)
    return rows[-1] if rows else None


def query_blocking_outcomes(
    *,
    now_utc: _dt.datetime | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return the LATEST event-per-probe_id rows whose effective blocker_status
    is ``blocking`` AND whose expires_at_utc is in the future.

    Effective blocker_status is derived from the latest event for each probe_id:
    - blocker_status == "blocking" → blocking
    - blocker_status == "expired" → not blocking
    - blocker_status == "advisory" → not blocking
    - additionally: if expires_at_utc <= now_utc, treat as expired (not blocking)
    """
    rows = load_outcomes(path)
    latest_by_probe_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = row.get("probe_id")
        if isinstance(pid, str):
            latest_by_probe_id[pid] = row  # JSONL append order; later wins

    now = now_utc or _dt.datetime.now(_dt.UTC)
    now_iso = now.isoformat(timespec="microseconds").replace("+00:00", "Z")

    blocking: list[dict[str, Any]] = []
    for row in latest_by_probe_id.values():
        if row.get("blocker_status") != BLOCKER_STATUS_BLOCKING:
            continue
        expires_at = row.get("expires_at_utc")
        if isinstance(expires_at, str) and expires_at <= now_iso:
            continue  # outcome has aged past the staleness window
        blocking.append(row)
    return blocking


def latest_blocking_outcome_by_recipe(
    recipe_path: str | Path,
    *,
    now_utc: _dt.datetime | None = None,
    path: Path | None = None,
) -> ProbeOutcomeView | None:
    """Return the most-recent blocking outcome for the recipe, or None.

    Operator-authorize uses this to decide whether to refuse a dispatch.
    Per CLAUDE.md "Forbidden premature KILL": a blocking outcome does NOT
    mean the lane is killed — it means the apparatus has already adjudicated
    this probe within the staleness window and a fresh dispatch would re-do
    work the system already settled. Bypass via paired-env operator override
    (see ``tools/operator_authorize.py::_check_predecessor_probe_outcome``).
    """
    blockers = query_blocking_outcomes(now_utc=now_utc, path=path)
    target = _normalize_recipe_path(str(recipe_path))
    matches: list[dict[str, Any]] = []
    for row in blockers:
        candidate = row.get("recipe_path")
        if not isinstance(candidate, str):
            continue
        if _normalize_recipe_path(candidate) == target:
            matches.append(row)
    if not matches:
        return None
    # Latest by adjudicated_at_utc (string compare; ISO-8601 sortable).
    matches.sort(
        key=lambda r: str(r.get("adjudicated_at_utc", "")), reverse=True
    )
    return ProbeOutcomeView.from_row(matches[0])


def latest_blocking_outcome_by_substrate(
    substrate: str,
    *,
    now_utc: _dt.datetime | None = None,
    path: Path | None = None,
) -> ProbeOutcomeView | None:
    """Return the most-recent blocking outcome for the substrate, or None.

    Used as a fallback by ``_check_predecessor_probe_outcome`` when the
    operator-authorize callsite cannot resolve recipe_path to a registered
    outcome (e.g., recipe renamed; substrate string is the durable key).
    """
    blockers = query_blocking_outcomes(now_utc=now_utc, path=path)
    matches = [r for r in blockers if r.get("substrate") == substrate]
    if not matches:
        return None
    matches.sort(
        key=lambda r: str(r.get("adjudicated_at_utc", "")), reverse=True
    )
    return ProbeOutcomeView.from_row(matches[0])


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
        r for r in load_outcomes(path)
        if isinstance(r.get("written_at_utc"), str) and r["written_at_utc"] >= utc_str
    ]


__all__ = [
    "BLOCKER_STATUS_ADVISORY",
    "BLOCKER_STATUS_BLOCKING",
    "BLOCKER_STATUS_EXPIRED",
    "BLOCKING_VERDICTS",
    "DEFAULT_STALENESS_WINDOW_DAYS",
    "EVENT_ADJUDICATED",
    "EVENT_EXPIRED",
    "EVENT_OPERATOR_OVERRIDE",
    "EVENT_RATIFIED",
    "EVENT_SUPERSEDED",
    "LOCK_TIMEOUT_SECONDS",
    "PROBE_OUTCOMES_LEDGER_LOCK",
    "PROBE_OUTCOMES_LEDGER_PATH",
    "SCHEMA_VERSION",
    "VALID_BLOCKER_STATUSES",
    "VALID_EVENT_TYPES",
    "VALID_VERDICTS",
    "VERDICT_DEFER",
    "VERDICT_INDEPENDENT",
    "VERDICT_INFRASTRUCTURE_FAILURE",
    "VERDICT_KILL",
    "VERDICT_OPERATOR_REVIEW_REQUIRED",
    "VERDICT_PARTIAL",
    "VERDICT_PROCEED",
    "VERDICT_PROMOTE",
    "ProbeOutcomeView",
    "ProbeOutcomesLedgerCorruptError",
    "latest_blocking_outcome_by_recipe",
    "latest_blocking_outcome_by_substrate",
    "latest_outcome_by_probe_id",
    "latest_outcome_by_substrate",
    "load_outcomes",
    "load_outcomes_strict",
    "query_all_post_utc",
    "query_blocking_outcomes",
    "query_by_probe_id",
    "query_by_recipe",
    "query_by_substrate",
    "register_probe_outcome",
    "update_probe_outcome",
]
