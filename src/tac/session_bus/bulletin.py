# SPDX-License-Identifier: MIT
"""Session-events bulletin — a fcntl-locked append-only cross-agent broadcast surface.

Background — the coordination gap this closes
──────────────────────────────────────────────
2026-07-09 (task #388): during the P6 seal, review round 1 found SYNTHESIS_v3's ``b_c``
section stale because a SIBLING agent landed the #386 gate MID-ROUND. Long-running agents
have no way to learn "a gate just RULED / a verdict just LANDED / a spec was just EDITED"
without re-grepping the whole repo every few minutes. The bulletin is the missing broadcast
channel: producers ``post_event`` a tiny structured row the moment something lands; consumers
(a seal round, a synthesis writer, a controller) call ``staleness_check`` to ask "did anything
about my subjects land since I started?" without polling files.

BOUNDARY vs sibling surfaces (checked before building — this is NOT a parallel store):

* ``.omx/state/subagent_progress.jsonl`` (``tac.session_bus.recovery_manifest`` / the
  ``tools/subagent_checkpoint.py`` tool) records LIVE SUBAGENT CHECKPOINTS (resume state) —
  a different fact (where an agent IS) than a bulletin event (what just HAPPENED, broadcast).
* ``tac.review_counter`` records the clean-pass COUNTER (derived durable state). It is a
  PRODUCER here (``record_round`` posts a ``verdict_landed`` event) but its ledger is the
  authority; the bulletin is a fire-and-forget notification.
* ``tac.council_continual_learning`` records DELIBERATION verdicts (decisions).
* ``tac.harness_failure_ledger`` records structured FAILURE incidents.
  The bulletin holds none of those facts durably — it is a transient, session-scoped event
  feed. Rows are LIVE_STATE (gitignored under ``.omx/state/*``), not HISTORICAL_PROVENANCE.

Store: ``.omx/state/session_events.jsonl`` — append-only JSONL under ``fcntl.flock(LOCK_EX)``
(Catalog #128/#131 pattern; mirrors ``tac.review_counter.record_round``). Each row is exactly
one physical line (``json.dumps`` escapes embedded newlines), so a process killed mid-write
leaves only a torn LAST line that the lenient reader skips — no valid row is ever corrupted.

Determinism: plain UTC ISO-8601 timestamps. File order == append order (guaranteed under the
lock), so integer-offset resume (``events_since(offset)``) is stable across readers.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "BULLETIN_DISABLE_ENV",
    "BULLETIN_LOCK_ENV",
    "BULLETIN_PATH_ENV",
    "DEFAULT_BULLETIN_LOCK_PATH",
    "DEFAULT_BULLETIN_PATH",
    "EVENT_AGENT_COMPLETED",
    "EVENT_AGENT_DIED",
    "EVENT_AGENT_SPAWNED",
    "EVENT_GATE_RULED",
    "EVENT_KINDS",
    "EVENT_MEMO_LANDED",
    "EVENT_SPEC_EDITED",
    "EVENT_VERDICT_LANDED",
    "SCHEMA_VERSION",
    "SessionBusError",
    "StalenessReport",
    "event_count",
    "events_since",
    "post_event",
    "read_events",
    "staleness_check",
]

SCHEMA_VERSION = "session_bus_event_v1"

# ─── Event-kind taxonomy (closed enum; post_event refuses anything else) ───
EVENT_GATE_RULED = "gate_ruled"
EVENT_VERDICT_LANDED = "verdict_landed"
EVENT_SPEC_EDITED = "spec_edited"
EVENT_MEMO_LANDED = "memo_landed"
EVENT_AGENT_SPAWNED = "agent_spawned"
EVENT_AGENT_COMPLETED = "agent_completed"
EVENT_AGENT_DIED = "agent_died"

EVENT_KINDS = frozenset(
    {
        EVENT_GATE_RULED,
        EVENT_VERDICT_LANDED,
        EVENT_SPEC_EDITED,
        EVENT_MEMO_LANDED,
        EVENT_AGENT_SPAWNED,
        EVENT_AGENT_COMPLETED,
        EVENT_AGENT_DIED,
    }
)

# This module sits at ``src/tac/session_bus/bulletin.py`` → repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BULLETIN_PATH = _REPO_ROOT / ".omx" / "state" / "session_events.jsonl"
DEFAULT_BULLETIN_LOCK_PATH = _REPO_ROOT / ".omx" / "state" / ".session_events.jsonl.lock"

# ─── Environment overrides (inherited by SPAWNED subprocess children) ───
# A monkeypatch of ``DEFAULT_BULLETIN_PATH`` only isolates the CURRENT process; a
# ``multiprocessing`` / ``subprocess`` child re-imports this module fresh and would write
# to the real live store (the #389 round-1 residual: 12 events leaked from the review_counter
# multiprocessing test's children). These env vars ARE inherited by children, so a parent
# that sets them (a test fixture, or a dispatched trainer that must not emit) redirects or
# mutes EVERY descendant process. Precedence: explicit call arg > env override > module default.
#   * ``TAC_SESSION_BULLETIN_PATH`` / ``TAC_SESSION_BULLETIN_LOCK_PATH`` — redirect the default
#     store/lock (used to isolate to tmp while still VERIFYING the emitted rows);
#   * ``TAC_SESSION_BULLETIN_DISABLE`` (truthy) — hard-mute default-path writes in this process
#     and all children (an explicit ``bulletin_path=`` call still writes — the caller chose a
#     destination on purpose).
BULLETIN_PATH_ENV = "TAC_SESSION_BULLETIN_PATH"
BULLETIN_LOCK_ENV = "TAC_SESSION_BULLETIN_LOCK_PATH"
BULLETIN_DISABLE_ENV = "TAC_SESSION_BULLETIN_DISABLE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class SessionBusError(ValueError):
    """A bulletin event failed validation and was refused before persistence."""


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _bulletin_disabled() -> bool:
    """True iff ``TAC_SESSION_BULLETIN_DISABLE`` is truthy (env-inherited by children)."""
    return os.environ.get(BULLETIN_DISABLE_ENV, "").strip().lower() in _TRUTHY


def _resolve_bulletin_path(explicit: Path | None) -> Path:
    """explicit arg > ``TAC_SESSION_BULLETIN_PATH`` env > module ``DEFAULT_BULLETIN_PATH``."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(BULLETIN_PATH_ENV, "").strip()
    return Path(env) if env else DEFAULT_BULLETIN_PATH


def _resolve_lock_path(explicit: Path | None) -> Path:
    """explicit arg > ``TAC_SESSION_BULLETIN_LOCK_PATH`` env > ``DEFAULT_BULLETIN_LOCK_PATH``."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(BULLETIN_LOCK_ENV, "").strip()
    return Path(env) if env else DEFAULT_BULLETIN_LOCK_PATH


def _validate(kind: str, subject: str, payload: dict, agent_label: str) -> None:
    if kind not in EVENT_KINDS:
        raise SessionBusError(
            f"kind must be one of {sorted(EVENT_KINDS)}, got {kind!r}"
        )
    if not isinstance(subject, str) or not subject.strip():
        raise SessionBusError("subject must be a non-empty string")
    if not isinstance(agent_label, str) or not agent_label.strip():
        raise SessionBusError("agent_label must be a non-empty string")
    if not isinstance(payload, dict):
        raise SessionBusError(
            f"payload must be a dict, got {type(payload).__name__}"
        )
    # Reject payloads that would produce a multi-line or non-serializable row.
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SessionBusError(f"payload must be JSON-serializable: {exc}") from exc


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _needs_newline_separator(path: Path) -> bool:
    """True iff ``path`` exists, is non-empty, and its last byte is not ``\\n``.

    Such a torn tail is left when a process is killed mid-write (the row is a single
    ``line + "\\n"`` write, so a crash can truncate before the newline). Appending the
    next row directly would MERGE it onto the torn fragment, producing one corrupt
    physical line that :func:`read_events`' tolerant reader silently drops — losing
    BOTH the torn tail and the (valid) new row. Callers write a leading ``\\n`` when
    this returns True so the torn tail becomes its own (skipped) line and the new row
    lands clean. No-op on the normal path (file already ends in ``\\n``) -> the append
    is byte-identical there. Called under the same fcntl lock as the append, so no
    concurrent writer can change the tail between the check and the write.
    """
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return False
    if size == 0:
        return False
    with path.open("rb") as rf:
        rf.seek(-1, os.SEEK_END)
        return rf.read(1) != b"\n"


def post_event(
    kind: str,
    subject: str,
    payload: dict | None = None,
    agent_label: str = "",
    *,
    bulletin_path: Path | None = None,
    lock_path: Path | None = None,
) -> dict:
    """Append one bulletin event under fcntl LOCK_EX; return the row as written.

    ``kind`` must be one of :data:`EVENT_KINDS`. ``subject`` is the thing the event is
    ABOUT (e.g. ``"#386"``, ``"b_c"``, a surface id) — it is what :func:`staleness_check`
    matches against. ``payload`` is a freeform JSON-serializable dict of detail; an optional
    ``payload["subjects"]`` list of additional strings is also matched by staleness_check.
    """
    payload = {} if payload is None else payload
    _validate(kind, subject, payload, agent_label)
    record = {
        "schema": SCHEMA_VERSION,
        "kind": kind,
        "subject": subject,
        "payload": payload,
        "agent_label": agent_label,
        "written_at_utc": _utcnow_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
    }
    # Hard-mute (env-inherited by children): validate as usual, but skip the write when
    # falling back to the default/env path. An explicit ``bulletin_path=`` is honored even
    # under DISABLE — the caller deliberately chose a destination.
    if bulletin_path is None and _bulletin_disabled():
        return record
    path = _resolve_bulletin_path(bulletin_path)
    lock = _resolve_lock_path(lock_path)
    _ensure_parent(path)
    _ensure_parent(lock)
    # json.dumps escapes embedded newlines, so the row is exactly one physical line.
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            sep = "\n" if _needs_newline_separator(path) else ""
            with path.open("a", encoding="utf-8") as bf:
                bf.write(sep + line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)
    return record


def read_events(*, bulletin_path: Path | None = None) -> list[dict]:
    """Read all valid events in append order; skip malformed/torn lines silently.

    Lenient by construction (mirrors ``tac.review_counter.load_rounds``): a process killed
    mid-append leaves a partial last line whose ``json.loads`` fails and is skipped, so a
    reader never crashes on a torn write and never returns a corrupt row.

    Path resolution is SYMMETRIC with :func:`post_event`: explicit ``bulletin_path`` arg >
    ``TAC_SESSION_BULLETIN_PATH`` env > module default. Without this, an env-redirected
    writer (the documented "isolate to tmp while VERIFYING the emitted rows" use, inherited
    by children) would post to the tmp store but a same-process ``read_events()`` /
    ``event_count()`` / ``staleness_check()`` with no explicit path would silently read the
    REAL default store — a write/read asymmetry (canon-wave #389 R13). The DISABLE env is a
    WRITE-only mute and is deliberately NOT consulted here (reading a store is never muted).
    """
    path = _resolve_bulletin_path(bulletin_path)
    if not path.exists():
        return []
    out: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA_VERSION:
            continue
        out.append(rec)
    return out


def event_count(*, bulletin_path: Path | None = None) -> int:
    """Total number of valid events (the current offset high-water mark)."""
    return len(read_events(bulletin_path=bulletin_path))


def events_since(
    marker: int | str | None = None,
    kinds: list[str] | tuple[str, ...] | None = None,
    *,
    bulletin_path: Path | None = None,
) -> list[dict]:
    """Return events after ``marker``, optionally filtered to ``kinds``.

    ``marker`` semantics:
      * ``None`` → all events;
      * ``int``  → offset into append order; returns ``events[marker:]`` (the resume-exact
        path — record ``event_count()`` when you start, pass it back later);
      * ``str``  → ISO-8601 timestamp; returns events whose ``written_at_utc`` is STRICTLY
        greater (lexicographic compare is correct for same-format UTC ISO strings). Ties in
        the same microsecond are excluded — prefer the integer offset for exact resume.
    """
    rows = read_events(bulletin_path=bulletin_path)
    if isinstance(marker, bool):  # guard: bool is an int subclass
        raise SessionBusError("marker must be int offset, ISO str, or None (not bool)")
    if marker is None:
        selected = rows
    elif isinstance(marker, int):
        if marker < 0:
            raise SessionBusError("integer marker (offset) must be >= 0")
        selected = rows[marker:]
    elif isinstance(marker, str):
        selected = [r for r in rows if str(r.get("written_at_utc", "")) > marker]
    else:
        raise SessionBusError(
            f"marker must be int offset, ISO str, or None; got {type(marker).__name__}"
        )
    if kinds is not None:
        kind_set = set(kinds)
        selected = [r for r in selected if r.get("kind") in kind_set]
    return selected


@dataclass(frozen=True)
class StalenessReport:
    """Result of a :func:`staleness_check`.

    ``stale`` is True iff any event touched one of the checked subjects since ``since``.
    ``events`` is the matching events (append order). ``matched_subjects`` is the subset of
    ``checked_subjects`` that had at least one hit.
    """

    stale: bool
    events: tuple[dict, ...]
    checked_subjects: tuple[str, ...]
    since: int | str | None
    matched_subjects: tuple[str, ...] = field(default=())

    def describe(self) -> str:
        if not self.stale:
            return (
                f"FRESH: no events touching {list(self.checked_subjects)} since "
                f"{self.since!r}"
            )
        return (
            f"STALE: {len(self.events)} event(s) touching "
            f"{list(self.matched_subjects)} landed since {self.since!r} — re-read before "
            f"sealing"
        )


def _event_subject_strings(rec: dict) -> list[str]:
    """All subject-ish strings for an event: its ``subject`` plus payload['subjects']."""
    strings: list[str] = []
    subj = rec.get("subject")
    if isinstance(subj, str) and subj:
        strings.append(subj)
    payload = rec.get("payload")
    if isinstance(payload, dict):
        extra = payload.get("subjects")
        if isinstance(extra, list):
            strings.extend(s for s in extra if isinstance(s, str) and s)
    return strings


def _subject_matches(query: str, event_subject_strings: list[str]) -> bool:
    """Case-insensitive both-direction substring match (robust to id vs. label drift)."""
    q = query.strip().lower()
    if not q:
        return False
    for es in event_subject_strings:
        esl = es.lower()
        if q in esl or esl in q:
            return True
    return False


def staleness_check(
    subjects: list[str] | tuple[str, ...],
    since: int | str | None = None,
    kinds: list[str] | tuple[str, ...] | None = None,
    *,
    bulletin_path: Path | None = None,
) -> StalenessReport:
    """Did anything about ``subjects`` land since ``since``? (the seal-round use case).

    ``subjects`` is the list of things the caller cares about (e.g. ``["b_c", "flip",
    "#386"]``). Matching is case-insensitive both-direction substring against each event's
    ``subject`` and any ``payload['subjects']``. ``since`` uses the same marker semantics as
    :func:`events_since` (int offset recommended for exact resume). ``kinds`` optionally
    restricts to specific event kinds.
    """
    if not isinstance(subjects, (list, tuple)) or not subjects:
        raise SessionBusError("subjects must be a non-empty list of strings")
    if not all(isinstance(s, str) for s in subjects):
        raise SessionBusError("subjects must all be strings")
    candidates = events_since(since, kinds, bulletin_path=bulletin_path)
    hits: list[dict] = []
    matched: set[str] = set()
    for rec in candidates:
        ev_subjects = _event_subject_strings(rec)
        for q in subjects:
            if _subject_matches(q, ev_subjects):
                hits.append(rec)
                matched.add(q)
                break
    return StalenessReport(
        stale=bool(hits),
        events=tuple(hits),
        checked_subjects=tuple(subjects),
        since=since,
        matched_subjects=tuple(s for s in subjects if s in matched),
    )
