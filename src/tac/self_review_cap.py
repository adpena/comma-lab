"""Hard self-review round cap — A3 of the harness-engineering crosswalk
(``.omx/research/harness_engineering_crosswalk_20260719_codex.md``).

CLOSES ledger class ``arm_review_spiral_unbounded_seal_loop`` (row 64, prevention owed:
"delegate contract should cap self-review rounds (e.g. 5) with escalate-to-MAIN on
non-convergence instead of unbounded respin"). Empirical anchor: an arm produced FIX21 +
15 review_fix spec memos + a 200 MB log before death because the CLEAN-x3 seal counter never
converged (the R12-D lens-coverage expansion pattern at arm scale).

TWO COUNTERS, kept SEPARATE by design (the crosswalk's binding distinction):

  * ``rounds_completed`` — a HARD, monotonic cap. NEVER reset by a finding. When it reaches
    ``SELF_REVIEW_ROUND_CAP`` (5) without a seal, the next round is REFUSED and the verdict
    is ``ESCALATE_MAIN``. This is the anti-spiral brake.
  * ``clean_pass_streak`` — the canonical 3-clean-pass seal counter. RESETS on ANY finding
    (per CLAUDE.md "Recursive adversarial review protocol"). A finding in round 5 resets
    ONLY this streak, NOT the hard round cap — so a late finding cannot buy unbounded
    additional rounds.

State store: ``.omx/state/self_review_rounds.jsonl`` (APPEND-ONLY, fcntl-locked; the
``tac.harness_failure_ledger`` discipline). Current state per ``arm_id`` is DERIVED from the
event history. This module is queryable so the delegate wrapper / a preflight gate can refuse
a sixth self-review start and emit an escalation receipt.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CLEAN_PASSES_TO_SEAL",
    "DEFAULT_STATE_PATH",
    "SELF_REVIEW_ROUND_CAP",
    "VERDICTS",
    "ReviewRoundState",
    "load_round_events",
    "may_start_round",
    "record_round",
    "review_state",
    "self_review_verdict",
]

#: The hard cap on self-review rounds before mandatory escalation (crosswalk A3: "e.g. 5").
SELF_REVIEW_ROUND_CAP = 5
#: Consecutive clean rounds required to SEAL (CLAUDE.md 3-clean-pass discipline).
CLEAN_PASSES_TO_SEAL = 3
#: Terminal verdicts.
VERDICTS = ("PROCEED", "SEALED", "ESCALATE_MAIN")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = _REPO_ROOT / ".omx" / "state" / "self_review_rounds.jsonl"


class SelfReviewCapError(ValueError):
    """Invalid round event (fail-closed at the writer)."""


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class _RoundEvent:
    arm_id: str
    ts: str
    clean: bool
    note: str = ""
    schema: str = "self_review_round.v1"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _append_event(ev: _RoundEvent, *, path: Path | None = None) -> None:
    store = path or DEFAULT_STATE_PATH
    store.parent.mkdir(parents=True, exist_ok=True)
    lock = store.with_name("." + store.name + ".lock")
    line = json.dumps(ev.to_dict(), sort_keys=True, allow_nan=False)
    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            with store.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)


def load_round_events(path: Path | None = None) -> list[_RoundEvent]:
    store = path or DEFAULT_STATE_PATH
    if not store.exists():
        return []
    out: list[_RoundEvent] = []
    for raw in store.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("schema") != "self_review_round.v1":
            continue
        if not payload.get("arm_id"):
            continue
        out.append(_RoundEvent(
            arm_id=str(payload["arm_id"]), ts=str(payload.get("ts", "")),
            clean=bool(payload.get("clean", False)), note=str(payload.get("note", "")),
        ))
    return out


@dataclass
class ReviewRoundState:
    """Derived per-arm review state. ``rounds_completed`` is the HARD cap counter (never reset
    by findings); ``clean_pass_streak`` resets on any finding."""

    arm_id: str
    rounds_completed: int = 0
    clean_pass_streak: int = 0
    last_ts: str = ""
    history: list[dict] = field(default_factory=list)

    @property
    def sealed(self) -> bool:
        return self.clean_pass_streak >= CLEAN_PASSES_TO_SEAL

    @property
    def cap_reached(self) -> bool:
        return self.rounds_completed >= SELF_REVIEW_ROUND_CAP

    @property
    def verdict(self) -> str:
        if self.sealed:
            return "SEALED"
        if self.cap_reached:
            return "ESCALATE_MAIN"
        return "PROCEED"

    def to_dict(self) -> dict:
        return {
            "arm_id": self.arm_id,
            "rounds_completed": self.rounds_completed,
            "clean_pass_streak": self.clean_pass_streak,
            "sealed": self.sealed,
            "cap_reached": self.cap_reached,
            "verdict": self.verdict,
            "cap": SELF_REVIEW_ROUND_CAP,
            "last_ts": self.last_ts,
        }


def review_state(arm_id: str, *, path: Path | None = None) -> ReviewRoundState:
    """Fold the round events for ``arm_id`` into current state (pure derivation)."""
    st = ReviewRoundState(arm_id=arm_id)
    for ev in load_round_events(path):
        if ev.arm_id != arm_id:
            continue
        st.rounds_completed += 1  # HARD counter: monotonic, never reset by a finding
        if ev.clean:
            st.clean_pass_streak += 1
        else:
            st.clean_pass_streak = 0  # a finding resets ONLY the seal streak
        st.last_ts = ev.ts
        st.history.append({"ts": ev.ts, "clean": ev.clean})
    return st


def may_start_round(arm_id: str, *, path: Path | None = None) -> tuple[bool, str]:
    """May the arm START another self-review round? Returns (allowed, verdict).

    Refuses (False, "ESCALATE_MAIN") once ``SELF_REVIEW_ROUND_CAP`` rounds have completed
    without a seal. An already-sealed arm returns (False, "SEALED") — no further rounds
    needed. Otherwise (True, "PROCEED")."""
    st = review_state(arm_id, path=path)
    if st.sealed:
        return False, "SEALED"
    if st.cap_reached:
        return False, "ESCALATE_MAIN"
    return True, "PROCEED"


def record_round(
    arm_id: str, *, clean: bool, note: str = "",
    ts: str | None = None, path: Path | None = None,
) -> ReviewRoundState:
    """Record a COMPLETED self-review round and return the new derived state.

    Records unconditionally (the round happened); the HARD cap is enforced at the NEXT
    ``may_start_round`` call, so the cap can never be silently bypassed by recording."""
    if not arm_id or not arm_id.strip():
        raise SelfReviewCapError("arm_id is required")
    _append_event(_RoundEvent(arm_id=arm_id, ts=ts or _utc_now_iso(),
                              clean=bool(clean), note=note), path=path)
    return review_state(arm_id, path=path)


def self_review_verdict(arm_id: str, *, path: Path | None = None) -> str:
    """Terminal verdict for the arm: PROCEED / SEALED / ESCALATE_MAIN."""
    return review_state(arm_id, path=path).verdict
