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
    "DEFAULT_LEDGER_PATH",
    "EVENT_KINDS",
    "FailureEvent",
    "FailureState",
    "RESOLUTIONS",
    "SURFACES",
    "append_failure_event",
    "failure_states",
    "load_failure_events",
    "rank_open_failures",
    "record_diagnosis",
    "record_failure",
    "record_recurrence",
    "record_resolution",
    "sense_rows",
]

SCHEMA_VERSION = "harness_failure.v1"

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
        return {k: v for k, v in self.__dict__.items()}


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
