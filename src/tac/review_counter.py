"""Canonical review-counter ledger — the clean-pass counter as durable, derived state.

Per the operating manual (``docs/operating_manual_craft_handoff.md`` §6.3 + §8.8) and the
CLAUDE.md "Recursive adversarial review protocol": the clean-pass counter is *consecutive
CLEAN review rounds since the last finding*; it **resets on ANY finding** (fixes made in
response to review are unreviewed new code), and a surface **SEALS at 3** consecutive CLEAN
rounds. Before this module the counter lived in chat prose — exactly the "track the counter
explicitly; never declare SEAL from fatigue" failure the manual names (§8.8).

This module makes the counter DERIVED, never hand-set: writers append immutable round
records via :func:`record_round`; :func:`current_state` recomputes the counter from the
ledger every time. There is no API to set the counter, so it cannot be quietly advanced.

**Honest boundary (anti-gaming):** the ledger records *claims* that review rounds happened;
no gate here verifies a review actually occurred. :func:`record_round` requires a non-empty
``reviewed_commits`` list (a "review" of nothing is refused) and refuses inconsistent rows
(``CLEAN`` with findings, ``NOT_CLEAN`` without), but a dispatcher that records a CLEAN
round without running a real review is lying to the ledger, not beating it — the same trust
model as the lane registry and the council posterior. The structural protections are the
serializer/commit trail (``reviewed_commits`` are auditable shas) and the review dispatch
contract (``tac.subagent_contract.review_contract``), not this module.

Ledger: ``.omx/state/review_counter.jsonl`` — fcntl-locked append-only JSONL per the
Catalog #128 / #131 / #245 sister discipline (mirrors
``tac.council_continual_learning.append_council_anchor``). APPEND-ONLY per Catalog
#110/#113: to supersede a round, append a new row with the same ``(surface_id, round_n)``
— latest-row-wins at read time; rows are never mutated in place.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "DEFAULT_REVIEW_COUNTER_PATH",
    "DEFAULT_REVIEW_COUNTER_LOCK_PATH",
    "SEAL_THRESHOLD",
    "VERDICT_CLEAN",
    "VERDICT_NOT_CLEAN",
    "ReviewCounterState",
    "ReviewRound",
    "ReviewRoundValidationError",
    "current_state",
    "load_rounds",
    "record_round",
]

SCHEMA_VERSION = "review_counter_v1"

#: Consecutive CLEAN rounds required for SEAL (CLAUDE.md "Recursive adversarial review
#: protocol" gate 4: "3 consecutive clean passes required").
SEAL_THRESHOLD = 3

VERDICT_CLEAN = "CLEAN"
VERDICT_NOT_CLEAN = "NOT_CLEAN"
_VALID_VERDICTS = frozenset({VERDICT_CLEAN, VERDICT_NOT_CLEAN})

# Repo root resolution mirrors the rest of tac.* (this module sits in ``src/tac/``).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_REVIEW_COUNTER_PATH = _REPO_ROOT / ".omx" / "state" / "review_counter.jsonl"
DEFAULT_REVIEW_COUNTER_LOCK_PATH = (
    _REPO_ROOT / ".omx" / "state" / ".review_counter.jsonl.lock"
)


class ReviewRoundValidationError(ValueError):
    """A review-round record failed validation and was refused before persistence."""


@dataclass(frozen=True)
class ReviewRound:
    """One immutable review-round record (a claim that a review round happened)."""

    surface_id: str
    round_n: int
    findings_count: int
    verdict: str
    reviewed_commits: tuple[str, ...]
    notes: str = ""
    written_at_utc: str = ""
    written_pid: int | None = None
    written_host: str | None = None
    schema: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.surface_id, str) or not self.surface_id.strip():
            raise ReviewRoundValidationError("surface_id must be a non-empty string")
        if not isinstance(self.round_n, int) or isinstance(self.round_n, bool) or (
            self.round_n < 1
        ):
            raise ReviewRoundValidationError("round_n must be an int >= 1")
        if not isinstance(self.findings_count, int) or isinstance(
            self.findings_count, bool
        ) or self.findings_count < 0:
            raise ReviewRoundValidationError("findings_count must be an int >= 0")
        if self.verdict not in _VALID_VERDICTS:
            raise ReviewRoundValidationError(
                f"verdict must be one of {sorted(_VALID_VERDICTS)}, got {self.verdict!r}"
            )
        # Verdict/findings consistency — the counter is DERIVED, so a row that lies
        # about its own consistency is refused at the door.
        if self.verdict == VERDICT_CLEAN and self.findings_count != 0:
            raise ReviewRoundValidationError(
                "CLEAN verdict requires findings_count == 0 "
                f"(got {self.findings_count}); a round with findings is NOT_CLEAN"
            )
        if self.verdict == VERDICT_NOT_CLEAN and self.findings_count < 1:
            raise ReviewRoundValidationError(
                "NOT_CLEAN verdict requires findings_count >= 1"
            )
        # Anti-gaming floor: a review round must name WHAT it reviewed. This does not
        # prove the review happened (see module docstring boundary), but it refuses the
        # cheapest gaming path: recording a CLEAN round that reviewed nothing.
        commits = tuple(self.reviewed_commits)
        if not commits or any(
            not isinstance(c, str) or not c.strip() for c in commits
        ):
            raise ReviewRoundValidationError(
                "reviewed_commits must be a non-empty list of non-empty commit "
                "identifiers — a round that reviewed nothing is not a review round"
            )
        object.__setattr__(self, "reviewed_commits", commits)


@dataclass(frozen=True)
class ReviewCounterState:
    """Derived counter state for one surface (computed, never stored)."""

    surface_id: str
    rounds_recorded: int
    latest_round_n: int
    consecutive_clean: int
    sealed: bool
    last_verdict: str | None
    rounds: tuple[ReviewRound, ...] = field(repr=False, default=())

    def describe(self) -> str:
        """One-line human/prompt-facing summary (feeds review_contract counter_context)."""
        if self.rounds_recorded == 0:
            return f"surface {self.surface_id}: no review rounds recorded; counter 0/{SEAL_THRESHOLD}"
        seal = "SEALED" if self.sealed else f"counter {self.consecutive_clean}/{SEAL_THRESHOLD}"
        return (
            f"surface {self.surface_id}: {self.rounds_recorded} round(s) recorded, "
            f"latest round {self.latest_round_n} verdict {self.last_verdict}; {seal}"
        )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_to_dict(r: ReviewRound) -> dict:
    return {
        "schema": r.schema,
        "surface_id": r.surface_id,
        "round_n": r.round_n,
        "findings_count": r.findings_count,
        "verdict": r.verdict,
        "reviewed_commits": list(r.reviewed_commits),
        "notes": r.notes,
        "written_at_utc": r.written_at_utc,
        "written_pid": r.written_pid,
        "written_host": r.written_host,
    }


def _round_from_dict(payload: dict) -> ReviewRound:
    return ReviewRound(
        surface_id=str(payload["surface_id"]),
        round_n=int(payload["round_n"]),
        findings_count=int(payload["findings_count"]),
        verdict=str(payload["verdict"]),
        reviewed_commits=tuple(str(c) for c in payload.get("reviewed_commits", ())),
        notes=str(payload.get("notes", "")),
        written_at_utc=str(payload.get("written_at_utc", "")),
        written_pid=(
            int(payload["written_pid"]) if payload.get("written_pid") is not None else None
        ),
        written_host=payload.get("written_host"),
        schema=str(payload.get("schema", SCHEMA_VERSION)),
    )


def _ensure_state_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def record_round(
    surface_id: str,
    round_n: int,
    findings_count: int,
    verdict: str,
    notes: str = "",
    reviewed_commits: list[str] | tuple[str, ...] = (),
    *,
    ledger_path: Path | None = None,
    lock_path: Path | None = None,
) -> ReviewRound:
    """Validate + append one review-round record under fcntl LOCK_EX; return the record.

    APPEND-ONLY: to supersede a round, call again with the same ``(surface_id,
    round_n)`` — :func:`current_state` applies latest-row-wins. The counter itself is
    never written; it is derived at read time.
    """
    record = ReviewRound(
        surface_id=surface_id,
        round_n=round_n,
        findings_count=findings_count,
        verdict=verdict,
        reviewed_commits=tuple(reviewed_commits),
        notes=notes,
        written_at_utc=_utcnow_iso(),
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )
    ledger = ledger_path or DEFAULT_REVIEW_COUNTER_PATH
    lock = lock_path or DEFAULT_REVIEW_COUNTER_LOCK_PATH
    _ensure_state_dir(ledger)
    _ensure_state_dir(lock)
    line = json.dumps(_round_to_dict(record), sort_keys=True, allow_nan=False)
    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            with ledger.open("a", encoding="utf-8") as lf:
                lf.write(line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)
    return record


def load_rounds(
    surface_id: str | None = None,
    *,
    ledger_path: Path | None = None,
) -> list[ReviewRound]:
    """Lenient loader (skips malformed/wrong-schema/invalid lines), file order preserved.

    Mirrors ``tac.council_continual_learning.load_council_anchors`` semantics: a
    corrupted line degrades to a skipped line, never to a crashed reader.
    """
    ledger = ledger_path or DEFAULT_REVIEW_COUNTER_PATH
    if not ledger.exists():
        return []
    out: list[ReviewRound] = []
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
            record = _round_from_dict(payload)
        except (ReviewRoundValidationError, KeyError, TypeError, ValueError):
            continue
        if surface_id is None or record.surface_id == surface_id:
            out.append(record)
    return out


def current_state(
    surface_id: str,
    *,
    ledger_path: Path | None = None,
) -> ReviewCounterState:
    """Derive the clean-pass counter for ``surface_id`` from the ledger.

    Semantics (manual §6.3 / §8.8 + CLAUDE.md recursive-review protocol):

    * latest-row-wins per ``(surface_id, round_n)`` (append-only supersession);
    * rounds ordered by ``round_n``;
    * ``consecutive_clean`` = CLEAN rounds counted back from the latest round until
      the first NOT_CLEAN (any finding resets the streak to whatever follows it);
    * ``sealed`` when ``consecutive_clean >= SEAL_THRESHOLD`` (3).
    """
    rows = load_rounds(surface_id, ledger_path=ledger_path)
    latest_by_round: dict[int, ReviewRound] = {}
    for row in rows:  # file order == append order; later rows supersede
        latest_by_round[row.round_n] = row
    ordered = tuple(latest_by_round[n] for n in sorted(latest_by_round))
    consecutive = 0
    for row in reversed(ordered):
        if row.verdict != VERDICT_CLEAN:
            break
        consecutive += 1
    return ReviewCounterState(
        surface_id=surface_id,
        rounds_recorded=len(ordered),
        latest_round_n=ordered[-1].round_n if ordered else 0,
        consecutive_clean=consecutive,
        sealed=consecutive >= SEAL_THRESHOLD,
        last_verdict=ordered[-1].verdict if ordered else None,
        rounds=ordered,
    )
