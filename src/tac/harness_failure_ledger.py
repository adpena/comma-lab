"""Canonical harness FAILURE LEDGER — Weng "weakness mining" formalized as a queryable surface.

Source: the 2026-07-07 harvest of Lilian Weng's "Harness Engineering for Self-Improvement"
(``docs/harvest_weng_harness_20260707.md``). The post's Self-Harness loop starts from "rich
failure records" that (a) separate the terminal verifier-level cause from the causal mechanism,
(b) keep the CAUSAL STATUS honest (a hypothesized diagnosis is not a measured one — and a
falsified diagnosis stays recorded, per this repo's daemon-death saga where FOUR successive
theories each fit all evidence available at their time), and (c) prefer "recurrent error
patterns that are addressable … and can be resolved by narrow changes" when proposing harness
edits. This module is that record store.

BOUNDARY vs sibling surfaces (checked before building — this is NOT a parallel registry):

* ``tools/memory_blackbox.py`` records the SYSTEM MEMORY trajectory (samples), not failures.
* ``.omx/state/subagent_progress.jsonl`` records live subagent checkpoints (resume state).
* ``tac.council_continual_learning`` records DELIBERATION verdicts (decisions, not incidents).
* ``tac.probe_outcomes_ledger`` records probe VERDICTS (measurement outcomes on levers).
* Memory files record the LESSON prose; this ledger records the structured, countable,
  rankable incident state (recurrence_count, causal-status history, resolution) that a
  controller can QUERY. Rows reference the memory/DAG prose via ``related_ref``.

Store: ``.omx/state/harness_failure_ledger.jsonl`` — APPEND-ONLY event rows under fcntl
LOCK_EX (the ``tac.council_continual_learning.append_council_anchor`` pattern; Catalog
#128/#131/#245 sister discipline). Current state per ``failure_id`` is DERIVED from the event
history (latest-event-wins per field; diagnoses accumulate). Nothing is ever mutated: a wrong
diagnosis is superseded by a new ``diagnosis`` event, never rewritten.

Costate-controller SENSE wiring: ``rank_open_failures()`` / ``sense_rows()`` are the query
surface the #247 shadow controller reads via ``tac.witness_control.producer_bridge``
(harness-failure producer signal). Ranking = unresolved first, then recurrence_count
descending — the post's "preference for recurrent, addressable patterns".
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CAUSAL_STATUSES",
    "CLASS_ALIASES",
    "DEFAULT_LEDGER_PATH",
    "EVENT_KINDS",
    "EVENT_KINDS_V2",
    "FAILURE_SHAPES",
    "HANDOFF_STAGES",
    "RESOLUTIONS",
    "RESOLUTION_STATES",
    "RESOLVED_STATES",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V2",
    "SURFACES",
    "FailureEvent",
    "FailureEventV2",
    "FailureState",
    "FailureStateV2",
    "append_failure_event",
    "append_failure_event_v2",
    "failure_states",
    "failure_states_v2",
    "legacy_row_class_key",
    "legacy_row_resolution_state",
    "legacy_row_ts",
    "load_failure_events",
    "load_failure_events_v2",
    "load_raw_rows",
    "normalize_class_id",
    "project_legacy_rows_to_v2",
    "rank_open_failures",
    "record_diagnosis",
    "record_failure",
    "record_recurrence",
    "record_resolution",
    "sense_rows",
    "summarize_v2",
]

SCHEMA_VERSION = "harness_failure.v1"
SCHEMA_VERSION_V2 = "harness_failure.v2"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = _REPO_ROOT / ".omx" / "state" / "harness_failure_ledger.jsonl"

#: Where the failure happened (the post's editable-surface taxonomy, ours).
SURFACES = ("daemon", "subagent", "gate", "tool", "trainer-launch")
#: Honesty ladder for a diagnosis (a falsified one STAYS in the history).
CAUSAL_STATUSES = ("hypothesized", "measured", "falsified")
#: Lifecycle of the failure class itself.
RESOLUTIONS = ("open", "worked-around", "class-fixed", "gate-landed")
EVENT_KINDS = ("opened", "diagnosis", "recurrence", "resolution")


class FailureLedgerError(ValueError):
    """Invalid event field (fail-closed at the writer, per Catalog #138 discipline)."""


@dataclass(frozen=True)
class FailureEvent:
    """One append-only event row. ``diagnosis``/``causal_status`` describe the diagnosis
    carried BY THIS EVENT; the per-failure history is the sequence of such events."""

    failure_id: str
    event: str  # one of EVENT_KINDS
    ts: str
    surface: str = ""
    terminal_cause: str = ""
    diagnosis: str = ""
    causal_status: str = ""
    mechanism_exposed: str = ""
    related_ref: str = ""
    resolution: str = ""
    note: str = ""
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate(ev: FailureEvent) -> None:
    if not ev.failure_id or not ev.failure_id.strip():
        raise FailureLedgerError("failure_id is required")
    if ev.event not in EVENT_KINDS:
        raise FailureLedgerError(f"event must be one of {EVENT_KINDS}, got {ev.event!r}")
    if ev.event == "opened":
        if ev.surface not in SURFACES:
            raise FailureLedgerError(f"surface must be one of {SURFACES}, got {ev.surface!r}")
        if not ev.terminal_cause.strip():
            raise FailureLedgerError("opened event requires terminal_cause")
    if ev.causal_status and ev.causal_status not in CAUSAL_STATUSES:
        raise FailureLedgerError(
            f"causal_status must be one of {CAUSAL_STATUSES}, got {ev.causal_status!r}"
        )
    if ev.event == "diagnosis" and not ev.causal_status:
        raise FailureLedgerError("diagnosis event requires causal_status")
    if ev.resolution and ev.resolution not in RESOLUTIONS:
        raise FailureLedgerError(
            f"resolution must be one of {RESOLUTIONS}, got {ev.resolution!r}"
        )
    if ev.event == "resolution" and not ev.resolution:
        raise FailureLedgerError("resolution event requires resolution")


def append_failure_event(
    ev: FailureEvent,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """Append one event row under fcntl LOCK_EX. APPEND-ONLY: rows are never mutated;
    supersession = a later event for the same ``failure_id``."""
    _validate(ev)
    ledger = path or DEFAULT_LEDGER_PATH
    lock = lock_path or ledger.with_name("." + ledger.name + ".lock")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(ev.to_dict(), sort_keys=True, allow_nan=False)
    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)


def record_failure(
    failure_id: str,
    *,
    surface: str,
    terminal_cause: str,
    diagnosis: str = "",
    causal_status: str = "hypothesized",
    mechanism_exposed: str = "",
    related_ref: str = "",
    resolution: str = "open",
    ts: str | None = None,
    path: Path | None = None,
) -> FailureEvent:
    """Open a failure record (the "rich failure record" seed)."""
    ev = FailureEvent(
        failure_id=failure_id, event="opened", ts=ts or _utc_now_iso(),
        surface=surface, terminal_cause=terminal_cause, diagnosis=diagnosis,
        causal_status=causal_status if diagnosis else "",
        mechanism_exposed=mechanism_exposed, related_ref=related_ref,
        resolution=resolution,
    )
    append_failure_event(ev, path=path)
    return ev


def record_diagnosis(
    failure_id: str,
    *,
    diagnosis: str,
    causal_status: str,
    mechanism_exposed: str = "",
    note: str = "",
    ts: str | None = None,
    path: Path | None = None,
) -> FailureEvent:
    """Append a diagnosis event. A later ``falsified`` event supersedes but does NOT erase
    an earlier ``hypothesized``/``measured`` one — wrong diagnoses stay recorded."""
    ev = FailureEvent(
        failure_id=failure_id, event="diagnosis", ts=ts or _utc_now_iso(),
        diagnosis=diagnosis, causal_status=causal_status,
        mechanism_exposed=mechanism_exposed, note=note,
    )
    append_failure_event(ev, path=path)
    return ev


def record_recurrence(
    failure_id: str, *, note: str = "", ts: str | None = None, path: Path | None = None
) -> FailureEvent:
    """The same failure class fired again (recurrence_count is derived from these)."""
    ev = FailureEvent(
        failure_id=failure_id, event="recurrence", ts=ts or _utc_now_iso(), note=note
    )
    append_failure_event(ev, path=path)
    return ev


def record_resolution(
    failure_id: str, *, resolution: str, note: str = "",
    ts: str | None = None, path: Path | None = None,
) -> FailureEvent:
    """Move the failure's lifecycle state (open → worked-around → class-fixed → gate-landed)."""
    ev = FailureEvent(
        failure_id=failure_id, event="resolution", ts=ts or _utc_now_iso(),
        resolution=resolution, note=note,
    )
    append_failure_event(ev, path=path)
    return ev


def load_failure_events(path: Path | None = None) -> list[FailureEvent]:
    """Lenient loader (skips malformed / wrong-schema lines), mirroring
    ``tac.council_continual_learning.load_council_anchors`` semantics."""
    ledger = path or DEFAULT_LEDGER_PATH
    if not ledger.exists():
        return []
    out: list[FailureEvent] = []
    for raw in ledger.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_VERSION:
            continue
        try:
            ev = FailureEvent(**{k: payload.get(k, "") for k in FailureEvent.__dataclass_fields__})
            _validate(ev)
        except (TypeError, FailureLedgerError):
            continue
        out.append(ev)
    return out


@dataclass
class FailureState:
    """Derived current state of one failure_id (latest-event-wins per field;
    the diagnosis HISTORY is preserved in order, falsified entries included)."""

    failure_id: str
    surface: str = ""
    terminal_cause: str = ""
    mechanism_exposed: str = ""
    related_ref: str = ""
    resolution: str = "open"
    recurrence_count: int = 1  # the opening incident counts as the first occurrence
    first_ts: str = ""
    last_ts: str = ""
    diagnosis_history: list[dict] = field(default_factory=list)

    @property
    def current_causal_status(self) -> str:
        """Status of the latest still-live diagnosis, else 'falsified' (all theories dead),
        else '' (no diagnosis yet). A ``falsified`` event kills EVERY earlier entry with the
        same diagnosis text (a dead theory's earlier 'hypothesized' row must not resurrect)."""
        dead = {r["diagnosis"] for r in self.diagnosis_history
                if r["causal_status"] == "falsified"}
        for row in reversed(self.diagnosis_history):
            if row["causal_status"] != "falsified" and row["diagnosis"] not in dead:
                return row["causal_status"]
        return "falsified" if self.diagnosis_history else ""

    def to_dict(self) -> dict:
        return {
            "failure_id": self.failure_id, "surface": self.surface,
            "terminal_cause": self.terminal_cause,
            "mechanism_exposed": self.mechanism_exposed,
            "related_ref": self.related_ref, "resolution": self.resolution,
            "recurrence_count": self.recurrence_count,
            "causal_status": self.current_causal_status,
            "first_ts": self.first_ts, "last_ts": self.last_ts,
            "diagnosis_history": list(self.diagnosis_history),
        }


def failure_states(path: Path | None = None) -> dict[str, FailureState]:
    """Fold the event log into per-failure current state (pure derivation, no mutation)."""
    states: dict[str, FailureState] = {}
    for ev in load_failure_events(path):
        st = states.setdefault(ev.failure_id, FailureState(failure_id=ev.failure_id))
        if not st.first_ts:
            st.first_ts = ev.ts
        st.last_ts = ev.ts
        if ev.surface:
            st.surface = ev.surface
        if ev.terminal_cause:
            st.terminal_cause = ev.terminal_cause
        if ev.mechanism_exposed:
            st.mechanism_exposed = ev.mechanism_exposed
        if ev.related_ref:
            st.related_ref = ev.related_ref
        if ev.resolution:
            st.resolution = ev.resolution
        if ev.event == "recurrence":
            st.recurrence_count += 1
        if ev.diagnosis or (ev.event == "diagnosis" and ev.causal_status):
            st.diagnosis_history.append(
                {"ts": ev.ts, "diagnosis": ev.diagnosis,
                 "causal_status": ev.causal_status, "note": ev.note}
            )
    return states


def rank_open_failures(path: Path | None = None) -> list[FailureState]:
    """The SENSE ranking (the post's "preference for recurrent, addressable patterns"):
    unresolved classes first (open > worked-around), recurrence_count descending, then
    most-recent first. gate-landed/class-fixed entries sink (still queryable as history)."""
    order = {"open": 0, "worked-around": 1, "class-fixed": 2, "gate-landed": 3}
    states = list(failure_states(path).values())
    # two-pass stable sort: most-recent first within ties, then the primary ranking
    states.sort(key=lambda s: s.last_ts, reverse=True)
    states.sort(key=lambda s: (order.get(s.resolution, 0), -s.recurrence_count))
    return states


def sense_rows(path: Path | None = None, *, limit: int = 10) -> list[dict]:
    """Plain-dict rows for the costate controller SENSE (JSONL-persistable). Read-only,
    fail-safe by construction (missing ledger → empty list, never fabricated)."""
    return [s.to_dict() for s in rank_open_failures(path)[:limit]]


# ======================================================================================
# FailureEventV2 lifecycle — the A1 typed migration (harness_engineering crosswalk,
# .omx/research/harness_engineering_crosswalk_20260719_codex.md, ADOPT-A1).
#
# WHY (measured): the live ledger accumulated 68 rows across ≥3 incompatible writer
# generations — the canonical ``harness_failure.v1`` (57 rows: ``failure_id`` + ``event``),
# plus 11 schemaless ad-hoc rows keyed by ``failure_class`` (``ts_utc``/``utc``) or
# ``class_id`` (``first_seen_utc``/``written_at_utc``). A reader that coalesces heterogeneous
# key names (``tools/costate_digest.py::_LEDGER_CLASS_KEYS``) is a band-aid: newer rows still
# depend on the reader remembering every alias, prose in a ``resolution`` field can be mistaken
# for a closure event, and two class aliases double-count. V2 makes the SCHEMA canonical:
# ONE ``class_id`` key, TYPED status transitions, ``resolution_state`` NEVER inferred from
# prose, and recurrence rows referencing the parent class. Migration is APPEND-ONLY: the
# migration script projects legacy rows into canonical V2 rows that SUPERSEDE them; the
# originals are preserved (Catalog #110/#113 HISTORICAL_PROVENANCE discipline).
# ======================================================================================

#: Typed lifecycle events (supersede the v1 ``EVENT_KINDS`` prose-adjacent set).
EVENT_KINDS_V2 = (
    "OBSERVED",       # a rich failure record is opened
    "RECURRENCE",     # the same class fired again (references parent via class_id)
    "FIX_LANDED",     # the immediate code fix landed (one of the two landings)
    "GATE_LANDED",    # a STRICT/warn gate against re-introduction landed
    "VERIFIED_CLOSED",# fix + gate + a verifying observation — the class is closed
    "REOPENED",       # a prior closure was falsified; the class is open again
    "SUPERSEDED",     # this class_id is an alias of / replaced by another (see parent_class_id)
)

#: Typed resolution states. NEVER inferred from free prose — set explicitly on an event or
#: derived from the (typed) event_kind. FIX_ONLY / GATE_ONLY are the memo's key distinction:
#: a half-landing is NOT closed (the two-landing rule is owed the other half).
RESOLUTION_STATES = (
    "OPEN",           # no landing yet
    "FIX_ONLY",       # immediate fix landed; gate still owed
    "GATE_ONLY",      # gate landed; a durable code fix still owed
    "VERIFY_PENDING", # closure asserted (often in legacy prose) but not typed-verified
    "CLOSED",         # fix + gate + verified
    "SUPERSEDED",     # folded into another class_id
)

#: A resolution_state counts as "resolved" (not owed further work) only for these.
RESOLVED_STATES = frozenset({"CLOSED", "SUPERSEDED"})

#: Upstream causal vocabulary (crosswalk "earliest failed handoff" + "failure shape").
HANDOFF_STAGES = (
    "availability", "retrieval", "invocation", "relevance",
    "execution", "proof", "consumption", "lifecycle",
)
FAILURE_SHAPES = (
    "capability_gap", "context_gap", "authority_gap", "semantic_drift",
    "custody_gap", "liveness_misclassification", "migration_gap", "convergence_gap",
)

#: The two known class aliases (measured 2026-07-19: 22 raw ids → 20 semantic classes).
#: Each maps a non-canonical raw id to its canonical class_id. Applied at migration AND read.
CLASS_ALIASES = {
    "codex_probe_token_limit_death_incomplete_wip":
        "codex_probe_token_limit_death_incomplete_wip_20260712",
    "dashboard_false_FAIL_at_init":
        "dashboard_hardcoded_gate_boundary_false_fail_at_init",
}

#: Legacy ``resolution`` ENUM (the v1 RESOLUTIONS) → typed V2 resolution_state. Prose values
#: that are NOT one of these enum members are NEVER mapped from here (see the precedence in
#: ``legacy_row_resolution_state``).
_LEGACY_RESOLUTION_MAP = {
    "open": "OPEN",
    "worked-around": "FIX_ONLY",
    "class-fixed": "FIX_ONLY",
    "gate-landed": "CLOSED",
}
#: event_kind → default resolution_state when an event omits an explicit one.
_EVENT_KIND_DEFAULT_STATE = {
    "OBSERVED": "OPEN",
    "FIX_LANDED": "FIX_ONLY",
    "GATE_LANDED": "GATE_ONLY",
    "VERIFIED_CLOSED": "CLOSED",
    "REOPENED": "OPEN",
    "SUPERSEDED": "SUPERSEDED",
    # RECURRENCE intentionally absent: it never changes resolution_state.
}
#: resolution_state → the event_kind a migration projection should carry for it.
_STATE_TO_EVENT_KIND = {
    "OPEN": "OBSERVED",
    "FIX_ONLY": "FIX_LANDED",
    "GATE_ONLY": "GATE_LANDED",
    "VERIFY_PENDING": "OBSERVED",
    "CLOSED": "VERIFIED_CLOSED",
    "SUPERSEDED": "SUPERSEDED",
}
#: legacy timestamp field aliases, in preference order.
_LEGACY_TS_KEYS = ("ts", "ts_utc", "utc", "written_at_utc", "first_seen_utc")


class FailureLedgerV2Error(ValueError):
    """Invalid V2 event field (fail-closed at the writer)."""


@dataclass(frozen=True)
class FailureEventV2:
    """One canonical, append-only V2 event. ``class_id`` is the single stable identity;
    ``event_kind`` and ``resolution_state`` are TYPED (never prose). Optional fields carry
    the crosswalk vocabulary (earliest_failed_handoff / failure_shape / claim_boundary /
    worker_epoch / owner / verdict_scope / evidence_refs / next_trigger)."""

    class_id: str
    event_kind: str
    ts: str
    resolution_state: str = ""
    parent_class_id: str = ""       # for RECURRENCE / SUPERSEDED references
    recurrence_count: int = 0       # optional convenience on a projection row
    surface: str = ""
    terminal_cause: str = ""
    diagnosis: str = ""
    mechanism_exposed: str = ""
    earliest_failed_handoff: str = ""
    failure_shape: str = ""
    claim_boundary: str = ""
    worker_epoch: str = ""
    intervention_id: str = ""
    ablation_id: str = ""
    evidence_refs: str = ""
    next_trigger: str = ""
    owner: str = ""
    verdict_scope: str = ""
    related_ref: str = ""
    legacy_alias: str = ""          # the raw class-key(s) this canonical id folded
    migrated_from: str = ""         # provenance: legacy rows this projection supersedes
    note: str = ""
    schema: str = SCHEMA_VERSION_V2

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _resolved_state_for(event_kind: str, explicit: str) -> str:
    if explicit:
        return explicit
    return _EVENT_KIND_DEFAULT_STATE.get(event_kind, "")


def _validate_v2(ev: FailureEventV2) -> None:
    if not ev.class_id or not ev.class_id.strip():
        raise FailureLedgerV2Error("class_id is required")
    if ev.event_kind not in EVENT_KINDS_V2:
        raise FailureLedgerV2Error(
            f"event_kind must be one of {EVENT_KINDS_V2}, got {ev.event_kind!r}"
        )
    if ev.resolution_state and ev.resolution_state not in RESOLUTION_STATES:
        raise FailureLedgerV2Error(
            f"resolution_state must be one of {RESOLUTION_STATES}, got {ev.resolution_state!r}"
        )
    if ev.earliest_failed_handoff and ev.earliest_failed_handoff not in HANDOFF_STAGES:
        raise FailureLedgerV2Error(
            f"earliest_failed_handoff must be one of {HANDOFF_STAGES}, "
            f"got {ev.earliest_failed_handoff!r}"
        )
    if ev.failure_shape and ev.failure_shape not in FAILURE_SHAPES:
        raise FailureLedgerV2Error(
            f"failure_shape must be one of {FAILURE_SHAPES}, got {ev.failure_shape!r}"
        )
    if ev.event_kind in ("RECURRENCE", "SUPERSEDED") and not ev.parent_class_id.strip():
        # a recurrence/supersession must name the class it refers to (may equal class_id
        # for a self-recurrence; a supersession must name the canonical successor).
        raise FailureLedgerV2Error(
            f"{ev.event_kind} event requires parent_class_id"
        )


def append_failure_event_v2(
    ev: FailureEventV2,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """Append one V2 event under fcntl LOCK_EX (mirrors ``append_failure_event``)."""
    _validate_v2(ev)
    ledger = path or DEFAULT_LEDGER_PATH
    lock = lock_path or ledger.with_name("." + ledger.name + ".lock")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(ev.to_dict(), sort_keys=True, allow_nan=False)
    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)


def normalize_class_id(raw: str) -> str:
    """Apply the canonical CLASS_ALIASES so aliased ids collapse to one semantic class."""
    return CLASS_ALIASES.get(raw, raw)


def load_raw_rows(path: Path | None = None) -> list[dict]:
    """Every JSON-object line (any schema generation), lenient — the shared substrate for
    migration + the schema-tolerant reader. Malformed lines are skipped, never fabricated."""
    ledger = path or DEFAULT_LEDGER_PATH
    if not ledger.exists():
        return []
    out: list[dict] = []
    for raw in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def legacy_row_class_key(row: dict) -> str:
    """First populated class-identifier across writer generations (canonicalized), else ''.
    V2 rows return their ``class_id``; legacy rows fall back across the alias key names."""
    for k in ("class_id", "failure_id", "failure_class", "class", "bug_class"):
        v = row.get(k)
        if v:
            return normalize_class_id(str(v))
    return ""


def legacy_row_ts(row: dict) -> str:
    """Coalesce the timestamp across writer generations (empty string sorts first)."""
    for k in _LEGACY_TS_KEYS:
        v = row.get(k)
        if v:
            return str(v)
    return ""


def legacy_row_resolution_state(row: dict) -> str:
    """Map a legacy row's STRUCTURED markers → typed resolution_state.

    NEVER infers closure from free prose (the memo's binding rule). Precedence:
      1. ``status`` token (structured tokens like ``open_prevention_owed`` /
         ``resolved_with_prevention_owed`` / ``recurrence_resolved_cure_confirmed``);
      2. ``resolution`` field IFF it is a known v1 enum member;
      3. ``resolved`` boolean → CLOSED when true;
      4. a NON-ENUM non-empty ``resolution`` prose value → VERIFY_PENDING
         (closure asserted-in-prose but not typed — distinct from OPEN);
      5. else OPEN.
    """
    status = str(row.get("status") or "").strip().lower()
    if status:
        if status.startswith("open"):
            return "OPEN"
        if "prevention_owed" in status:  # e.g. resolved_with_prevention_owed
            return "FIX_ONLY"
        if any(tok in status for tok in ("resolved", "closed", "fixed", "cure_confirmed")):
            return "CLOSED"
    res = row.get("resolution")
    if isinstance(res, str):
        enum = res.strip().lower()
        if enum in _LEGACY_RESOLUTION_MAP:
            return _LEGACY_RESOLUTION_MAP[enum]
    if row.get("resolved") is True:
        return "CLOSED"
    if isinstance(res, str) and res.strip():
        # non-enum prose in the resolution field: a closure was ASSERTED but is not typed.
        return "VERIFY_PENDING"
    return "OPEN"


def _legacy_row_is_recurrence(row: dict) -> bool:
    return str(row.get("event") or "").strip().lower() == "recurrence"


def project_legacy_rows_to_v2(rows: list[dict]) -> list[FailureEventV2]:
    """Project non-V2 rows into ONE canonical V2 projection event per semantic class.

    Pure function (no I/O). Folds all legacy rows for a class by timestamp; the LATEST row
    supplies the typed resolution_state (structured-only). Recurrence count = number of
    legacy recurrence events. Provenance (folded raw ids + aliases) is preserved on the row.
    V2 rows already present are IGNORED here (idempotent projection)."""
    already_v2 = {
        normalize_class_id(str(row["class_id"]))
        for row in rows
        if row.get("schema") == SCHEMA_VERSION_V2 and row.get("class_id")
    }
    by_class: dict[str, list[dict]] = {}
    alias_seen: dict[str, set[str]] = {}
    for row in rows:
        if row.get("schema") == SCHEMA_VERSION_V2:
            continue  # already migrated
        raw_key = None
        for k in ("class_id", "failure_id", "failure_class", "class", "bug_class"):
            if row.get(k):
                raw_key = str(row[k])
                break
        if not raw_key:
            continue
        cid = normalize_class_id(raw_key)
        by_class.setdefault(cid, []).append(row)
        alias_seen.setdefault(cid, set()).add(raw_key)

    projections: list[FailureEventV2] = []
    for cid, class_rows in sorted(by_class.items()):
        if cid in already_v2:
            continue  # idempotent: this class already has a canonical V2 row
        ordered = sorted(class_rows, key=legacy_row_ts)
        latest = ordered[-1]
        state = legacy_row_resolution_state(latest)
        event_kind = _STATE_TO_EVENT_KIND.get(state, "OBSERVED")
        rec = sum(1 for r in ordered if _legacy_row_is_recurrence(r))
        first_ts = legacy_row_ts(ordered[0]) or _utc_now_iso()
        last_ts = legacy_row_ts(latest) or first_ts
        aliases = sorted(a for a in alias_seen[cid] if a != cid)
        surface = ""
        for r in ordered:
            if r.get("surface"):
                surface = str(r["surface"])
                break
        projections.append(FailureEventV2(
            class_id=cid,
            event_kind=event_kind,
            ts=_utc_now_iso(),
            resolution_state=state,
            recurrence_count=rec,
            surface=surface,
            legacy_alias=", ".join(aliases),
            migrated_from=(
                f"{len(ordered)} legacy row(s) [{first_ts}..{last_ts}] "
                f"schema-generations folded"
            ),
            note="canonical V2 projection (migration); legacy rows preserved as provenance",
        ))
    return projections


def load_failure_events_v2(path: Path | None = None) -> list[FailureEventV2]:
    """Load only the canonical V2 rows (schema == harness_failure.v2), lenient."""
    out: list[FailureEventV2] = []
    for payload in load_raw_rows(path):
        if payload.get("schema") != SCHEMA_VERSION_V2:
            continue
        try:
            ev = FailureEventV2(**{
                k: payload.get(k, FailureEventV2.__dataclass_fields__[k].default)
                for k in FailureEventV2.__dataclass_fields__
            })
            _validate_v2(ev)
        except (TypeError, FailureLedgerV2Error):
            continue
        out.append(ev)
    return out


@dataclass
class FailureStateV2:
    """Derived current state of one class_id from its V2 event history (typed only)."""

    class_id: str
    resolution_state: str = "OPEN"
    recurrence_count: int = 0
    first_ts: str = ""
    last_ts: str = ""
    event_history: list[dict] = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.resolution_state in RESOLVED_STATES

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "resolution_state": self.resolution_state,
            "recurrence_count": self.recurrence_count,
            "is_resolved": self.is_resolved,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
        }


def failure_states_v2(path: Path | None = None) -> dict[str, FailureStateV2]:
    """Fold V2 events into per-class state (latest resolution-affecting event wins).

    Folds in APPEND ORDER (the file order), NOT by ``ts``: an append-only log serialized under
    fcntl LOCK_EX makes append-order = commit-order the authority, and it is robust to a
    backdated / hand-written ``ts`` (a later-appended event always supersedes). This matches
    the V1 ``failure_states`` folder and the digest's file-order reader."""
    states: dict[str, FailureStateV2] = {}
    for ev in load_failure_events_v2(path):
        st = states.setdefault(ev.class_id, FailureStateV2(class_id=ev.class_id))
        if not st.first_ts:
            st.first_ts = ev.ts
        st.last_ts = ev.ts
        st.event_history.append({"event_kind": ev.event_kind, "ts": ev.ts,
                                 "resolution_state": ev.resolution_state})
        if ev.event_kind == "RECURRENCE":
            st.recurrence_count += 1
            continue
        # a projection row may carry an explicit recurrence_count from folded legacy rows.
        if ev.recurrence_count:
            st.recurrence_count = max(st.recurrence_count, ev.recurrence_count)
        new_state = _resolved_state_for(ev.event_kind, ev.resolution_state)
        if new_state:
            st.resolution_state = new_state
    return states


def summarize_v2(path: Path | None = None) -> dict:
    """Canonical V2 summary consumed by the digest. ``unresolved`` = states not yet CLOSED/
    SUPERSEDED that are still OPEN; ``not_closed`` = the wider owed set (OPEN/FIX_ONLY/
    GATE_ONLY/VERIFY_PENDING). No phantom '?' class is possible (class_id is required)."""
    states = failure_states_v2(path)
    open_ids = sorted(k for k, s in states.items() if s.resolution_state == "OPEN")
    not_closed = sorted(k for k, s in states.items() if not s.is_resolved)
    recurrent = sorted(k for k, s in states.items() if s.recurrence_count >= 1)
    return {
        "classes": len(states),
        "unresolved": open_ids,
        "not_closed": not_closed,
        "recurrent": recurrent,
        "states": {k: s.resolution_state for k, s in sorted(states.items())},
    }
