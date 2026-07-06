"""Lever ACTIVATION ledger — the "off is a tracked queue, never a forgotten default" apparatus.

Per the CLAUDE.md NON-NEGOTIABLE "'Off' is a tracked queue" (operator 2026-07-06 "Default off is a
source of orphaned signal ... never orphaned or rediscovery or counting on me to remember"): a
score-affecting lever may default OFF (safety) ONLY if "off" is a TRACKED queue state the controller
drains — never a silent default that relies on operator memory.

``lever_registry`` already answers COVERAGE ("is this trainer flag held by a DSL ``Lever`` factory?").
This module adds the missing ACTIVATION dimension: "has this DSL lever ever FIRED (been launched as an
A/B arm) and been MEASURED?" A lever that is held-by-the-DSL yet NEVER-FIRED is still orphaned signal
until this ledger tracks it.

State machine per lever:  never-fired  ->fired->  fired-unmeasured  ->measured->  measured
                          (and ->retired, with a recorded reason, from any state; terminal).

The ledger is a canonical, APPEND-ONLY, fcntl-locked JSONL at ``.omx/state/lever_activation_ledger.jsonl``
(mirrors the other ``.omx/state/*.jsonl`` canonical stores). It is POPULATED by REAL events only
(NO-FAKE): a "fired" event is written by ``tools/launch_witness_run.py`` when a run launches with
``--dsl-lever NAME`` (the canonical DSL-launcher path), a "measured" event when that run's byte-closed
verdict lands. It starts EMPTY — so ``never_fired()`` honestly returns EVERY DSL lever until one is
actually launched through the DSL path. (A raw-flag launch that bypasses the DSL launcher is NOT
recorded — which is itself the config-orphan the DSL exists to extinct; such launches SHOULD go through
the DSL so the campaign can track them.)

The #247 costate SENSE layer consumes ``duty_to_measure()`` to rank never-fired high-value levers into
its DECIDE queue: the CONTROLLER remembers and surfaces; the operator never has to.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tac.witness_dsl.lever_registry import lever_factories

_REPO_ROOT = Path(__file__).resolve().parents[3]
LEDGER_PATH = _REPO_ROOT / ".omx" / "state" / "lever_activation_ledger.jsonl"

# canonical event vocabulary
EVENT_FIRED = "fired"        # a run launched with this lever (an A/B arm was actually run)
EVENT_MEASURED = "measured"  # a byte-closed verdict landed for a run using this lever
EVENT_RETIRED = "retired"    # the lever was retired with a recorded reason (dormant-with-reactivation)
VALID_EVENTS = frozenset({EVENT_FIRED, EVENT_MEASURED, EVENT_RETIRED})

# canonical states (latest-event-wins per lever, with fired/measured monotonic)
STATE_NEVER_FIRED = "never-fired"
STATE_FIRED_UNMEASURED = "fired-unmeasured"
STATE_MEASURED = "measured"
STATE_RETIRED = "retired"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_activation(
    lever: str,
    event: str,
    *,
    run_ref: str | None = None,
    verdict_ref: str | None = None,
    reason: str = "",
    agent: str | None = None,
    path: Path | None = None,
) -> dict:
    """Append ONE activation event (fcntl-locked, APPEND-ONLY). Returns the written row.

    ``event`` must be one of :data:`VALID_EVENTS`. ``run_ref`` is the run out-dir / id; ``verdict_ref``
    the byte-closed verdict artifact (required-ish for a "measured" event to be trustworthy, but not
    hard-enforced here — the reader flags a measured-without-verdict as weak). NEVER a score claim.
    """
    if event not in VALID_EVENTS:
        raise ValueError(f"invalid activation event {event!r}; must be one of {sorted(VALID_EVENTS)}")
    if not lever or not isinstance(lever, str):
        raise ValueError(f"lever must be a non-empty str, got {lever!r}")
    row = {
        "lever": lever,
        "event": event,
        "run_ref": run_ref,
        "verdict_ref": verdict_ref,
        "reason": reason,
        "agent": agent,
        "ts": _utc(),
    }
    p = Path(path) if path is not None else LEDGER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    # fcntl-locked append (canonical .omx/state pattern). Best-effort on platforms without fcntl.
    try:
        import fcntl
        with open(p, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - non-POSIX fallback
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
    return row


def _read_events(path: Path | None = None) -> list[dict]:
    """Read all events, lenient (skips corrupt lines rather than failing the whole read)."""
    p = Path(path) if path is not None else LEDGER_PATH
    if not p.exists():
        return []
    out: list[dict] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(row, dict) and row.get("lever") and row.get("event") in VALID_EVENTS:
            out.append(row)
    return out


@dataclass(frozen=True)
class ActivationStatus:
    lever: str
    ever_fired: bool
    ever_measured: bool
    retired: bool
    state: str
    last_event: str | None
    last_ts: str | None
    n_fired: int
    n_measured: int


def activation_status(lever: str, path: Path | None = None) -> ActivationStatus:
    """The activation state of ONE lever (from its recorded events)."""
    evs = [e for e in _read_events(path) if e.get("lever") == lever]
    fired = [e for e in evs if e["event"] == EVENT_FIRED]
    measured = [e for e in evs if e["event"] == EVENT_MEASURED]
    retired = any(e["event"] == EVENT_RETIRED for e in evs)
    last = evs[-1] if evs else None
    # retired is terminal; else measured > fired > never.
    if retired:
        state = STATE_RETIRED
    elif measured:
        state = STATE_MEASURED
    elif fired:
        state = STATE_FIRED_UNMEASURED
    else:
        state = STATE_NEVER_FIRED
    return ActivationStatus(
        lever=lever,
        ever_fired=bool(fired),
        ever_measured=bool(measured),
        retired=retired,
        state=state,
        last_event=last["event"] if last else None,
        last_ts=last.get("ts") if last else None,
        n_fired=len(fired),
        n_measured=len(measured),
    )


def known_levers() -> tuple[str, ...]:
    """The canonical set of DSL ``Lever`` factory names (from ``lever_registry.lever_factories``)."""
    return tuple(sorted(lever_factories().keys()))


def never_fired(known: tuple[str, ...] | None = None, path: Path | None = None) -> tuple[str, ...]:
    """DSL levers that have NEVER fired (no ``fired`` event) and are not retired — the orphan surface.

    An empty ledger honestly returns every DSL lever: none has been launched through the DSL path yet.
    """
    names = known if known is not None else known_levers()
    out = []
    for name in names:
        st = activation_status(name, path)
        if not st.ever_fired and not st.retired:
            out.append(name)
    return tuple(out)


def duty_to_measure(known: tuple[str, ...] | None = None, path: Path | None = None) -> tuple[str, ...]:
    """Levers OWED a measurement: never-fired OR fired-but-never-measured (not retired). The costate
    SENSE ranks these into its DECIDE queue (the duty-to-measure the discipline mandates)."""
    names = known if known is not None else known_levers()
    out = []
    for name in names:
        st = activation_status(name, path)
        if st.retired:
            continue
        if not st.ever_fired or not st.ever_measured:
            out.append(name)
    return tuple(out)


def activation_report(known: tuple[str, ...] | None = None, path: Path | None = None) -> list[dict]:
    """The operator-facing 'what is registered but never fired?' surface — one row per DSL lever
    (default OFF by construction for score-affecting levers), with its activation state + reason-to-be
    (that the controller, not the operator, holds the queue)."""
    names = known if known is not None else known_levers()
    rows = []
    for name in names:
        st = activation_status(name, path)
        rows.append({
            "lever": name,
            "default": "off",  # score-affecting levers default off by construction (safety)
            "state": st.state,
            "ever_fired": st.ever_fired,
            "ever_measured": st.ever_measured,
            "n_fired": st.n_fired,
            "n_measured": st.n_measured,
            "last_event": st.last_event,
            "last_ts": st.last_ts,
        })
    # surface never-fired / fired-unmeasured first (the queue the controller drains)
    _order = {STATE_NEVER_FIRED: 0, STATE_FIRED_UNMEASURED: 1, STATE_MEASURED: 2, STATE_RETIRED: 3}
    rows.sort(key=lambda r: (_order.get(r["state"], 9), r["lever"]))
    return rows


__all__ = [
    "ActivationStatus",
    "EVENT_FIRED",
    "EVENT_MEASURED",
    "EVENT_RETIRED",
    "LEDGER_PATH",
    "STATE_FIRED_UNMEASURED",
    "STATE_MEASURED",
    "STATE_NEVER_FIRED",
    "STATE_RETIRED",
    "VALID_EVENTS",
    "activation_report",
    "activation_status",
    "duty_to_measure",
    "known_levers",
    "never_fired",
    "record_activation",
]
