"""Tests for tac.review_counter — the derived clean-pass counter ledger.

Transitions under test: advance (CLEAN streak grows), reset (any NOT_CLEAN zeroes the
streak), seal (3 consecutive CLEAN), latest-row-wins supersession, validation refusals
(the anti-gaming floor), and concurrent-append safety.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from tac import review_counter as rc


@pytest.fixture(autouse=True)
def _isolate_bulletin(tmp_path, monkeypatch):
    """record_round posts a fail-open verdict_landed bulletin (task #388). Redirect the
    bulletin store to tmp so these tests never pollute the live
    ``.omx/state/session_events.jsonl`` feed a concurrent seal agent reads (#389 hygiene).

    #389 round-2 fix: ALSO set the ``TAC_SESSION_BULLETIN_*`` env vars. A ``monkeypatch``
    of the module default only isolates THIS process — a ``multiprocessing`` child (the
    ``test_concurrent_appends_all_land_intact`` workers) re-imports ``bulletin`` fresh and
    ignores the setattr, leaking events to the LIVE store (the documented round-1 residual).
    The env vars are inherited by spawned children, so they redirect the children too."""
    from tac.session_bus import bulletin as _bull

    ev = tmp_path / "ev.jsonl"
    lock = tmp_path / ".ev.lock"
    monkeypatch.setattr(_bull, "DEFAULT_BULLETIN_PATH", ev)
    monkeypatch.setattr(_bull, "DEFAULT_BULLETIN_LOCK_PATH", lock)
    monkeypatch.setenv(_bull.BULLETIN_PATH_ENV, str(ev))
    monkeypatch.setenv(_bull.BULLETIN_LOCK_ENV, str(lock))


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "review_counter.jsonl"


def _rec(ledger: Path, round_n: int, verdict: str, findings: int, **kw):
    return rc.record_round(
        "surf",
        round_n,
        findings,
        verdict,
        notes=kw.pop("notes", ""),
        reviewed_commits=kw.pop("reviewed_commits", ["abc123"]),
        ledger_path=ledger,
        lock_path=ledger.with_suffix(".lock"),
        **kw,
    )


# --- crash-safe append (canon-wave #389 R5) --------------------------------------


def test_torn_tail_then_append_does_not_lose_new_round(ledger: Path) -> None:
    """A crash-mid-write torn tail (no trailing newline) must NOT swallow the next
    recorded round by concatenation — sibling of the bulletin crash-safe-append fix."""
    _rec(ledger, 1, rc.VERDICT_CLEAN, 0)
    # simulate a process killed mid-write: partial JSON line, NO trailing newline
    with ledger.open("a", encoding="utf-8") as f:
        f.write('{"surface_id": "surf", "round_n": 2, "torn')
    _rec(ledger, 3, rc.VERDICT_CLEAN, 0)
    valid = 0
    for raw in ledger.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            json.loads(raw)
            valid += 1
        except json.JSONDecodeError:
            pass
    assert valid == 2  # round 1 + round 3 both intact; torn fragment is its own skipped line
    # current_state still resolves (round 3 is the latest CLEAN)
    state = rc.current_state("surf", ledger_path=ledger)
    assert state.last_verdict == rc.VERDICT_CLEAN


def test_normal_append_no_spurious_blank_line(ledger: Path) -> None:
    """The newline-guard is a no-op on the healthy path (file already ends in ``\\n``)."""
    _rec(ledger, 1, rc.VERDICT_CLEAN, 0)
    _rec(ledger, 2, rc.VERDICT_CLEAN, 0)
    raw = ledger.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\n\n" not in raw


# --- transitions -----------------------------------------------------------------


def test_empty_ledger_state_is_zero(ledger: Path) -> None:
    state = rc.current_state("surf", ledger_path=ledger)
    assert state.rounds_recorded == 0
    assert state.consecutive_clean == 0
    assert not state.sealed
    assert state.last_verdict is None
    assert "counter 0/3" in state.describe()


def test_clean_rounds_advance_counter(ledger: Path) -> None:
    _rec(ledger, 1, rc.VERDICT_CLEAN, 0)
    _rec(ledger, 2, rc.VERDICT_CLEAN, 0)
    state = rc.current_state("surf", ledger_path=ledger)
    assert state.consecutive_clean == 2
    assert not state.sealed


def test_finding_resets_counter(ledger: Path) -> None:
    _rec(ledger, 1, rc.VERDICT_CLEAN, 0)
    _rec(ledger, 2, rc.VERDICT_CLEAN, 0)
    _rec(ledger, 3, rc.VERDICT_NOT_CLEAN, 2)
    state = rc.current_state("surf", ledger_path=ledger)
    assert state.consecutive_clean == 0
    assert state.last_verdict == rc.VERDICT_NOT_CLEAN
    assert not state.sealed


def test_seal_at_three_consecutive_clean(ledger: Path) -> None:
    _rec(ledger, 1, rc.VERDICT_NOT_CLEAN, 1)
    for n in (2, 3, 4):
        _rec(ledger, n, rc.VERDICT_CLEAN, 0)
    state = rc.current_state("surf", ledger_path=ledger)
    assert state.consecutive_clean == 3
    assert state.sealed
    assert "SEALED" in state.describe()


def test_streak_counts_only_after_last_finding(ledger: Path) -> None:
    _rec(ledger, 1, rc.VERDICT_CLEAN, 0)
    _rec(ledger, 2, rc.VERDICT_NOT_CLEAN, 1)
    _rec(ledger, 3, rc.VERDICT_CLEAN, 0)
    state = rc.current_state("surf", ledger_path=ledger)
    assert state.consecutive_clean == 1  # round 1's CLEAN does not survive the reset


def test_latest_row_wins_supersession(ledger: Path) -> None:
    _rec(ledger, 1, rc.VERDICT_CLEAN, 0)
    # A later append supersedes round 1 (e.g. the round is retracted as NOT_CLEAN).
    _rec(ledger, 1, rc.VERDICT_NOT_CLEAN, 1, notes="retraction")
    state = rc.current_state("surf", ledger_path=ledger)
    assert state.rounds_recorded == 1
    assert state.consecutive_clean == 0
    assert state.rounds[0].notes == "retraction"


def test_counter_is_derived_not_stored(ledger: Path) -> None:
    _rec(ledger, 1, rc.VERDICT_CLEAN, 0)
    raw = ledger.read_text(encoding="utf-8")
    assert "consecutive_clean" not in raw
    assert "sealed" not in raw


def test_surfaces_are_independent(ledger: Path) -> None:
    _rec(ledger, 1, rc.VERDICT_CLEAN, 0)
    rc.record_round(
        "other", 1, 1, rc.VERDICT_NOT_CLEAN,
        reviewed_commits=["def456"],
        ledger_path=ledger, lock_path=ledger.with_suffix(".lock"),
    )
    assert rc.current_state("surf", ledger_path=ledger).consecutive_clean == 1
    assert rc.current_state("other", ledger_path=ledger).consecutive_clean == 0


# --- validation refusals (the anti-gaming floor) -----------------------------------


def test_clean_with_findings_refused(ledger: Path) -> None:
    with pytest.raises(rc.ReviewRoundValidationError, match="CLEAN verdict requires"):
        _rec(ledger, 1, rc.VERDICT_CLEAN, 2)


def test_not_clean_without_findings_refused(ledger: Path) -> None:
    with pytest.raises(rc.ReviewRoundValidationError, match="NOT_CLEAN"):
        _rec(ledger, 1, rc.VERDICT_NOT_CLEAN, 0)


def test_empty_reviewed_commits_refused(ledger: Path) -> None:
    with pytest.raises(rc.ReviewRoundValidationError, match="reviewed nothing"):
        _rec(ledger, 1, rc.VERDICT_CLEAN, 0, reviewed_commits=[])
    with pytest.raises(rc.ReviewRoundValidationError, match="reviewed nothing"):
        _rec(ledger, 1, rc.VERDICT_CLEAN, 0, reviewed_commits=["  "])
    assert not ledger.exists()  # refused BEFORE persistence


def test_invalid_verdict_and_round_refused(ledger: Path) -> None:
    with pytest.raises(rc.ReviewRoundValidationError, match="verdict"):
        _rec(ledger, 1, "MOSTLY_CLEAN", 0)
    with pytest.raises(rc.ReviewRoundValidationError, match="round_n"):
        _rec(ledger, 0, rc.VERDICT_CLEAN, 0)


def test_loader_skips_malformed_lines(ledger: Path) -> None:
    _rec(ledger, 1, rc.VERDICT_CLEAN, 0)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write("not json\n")
        fh.write(json.dumps({"schema": "other_v9", "surface_id": "surf"}) + "\n")
        fh.write(json.dumps({"schema": rc.SCHEMA_VERSION}) + "\n")  # missing fields
    state = rc.current_state("surf", ledger_path=ledger)
    assert state.rounds_recorded == 1
    assert state.consecutive_clean == 1


# --- concurrent-append safety -------------------------------------------------------


def _append_worker(args) -> None:
    ledger_str, n = args
    from tac import review_counter as rcw

    rcw.record_round(
        "concurrent", n, 0, rcw.VERDICT_CLEAN,
        reviewed_commits=[f"sha{n}"],
        ledger_path=Path(ledger_str),
        lock_path=Path(ledger_str + ".lock"),
    )


def test_concurrent_appends_all_land_intact(ledger: Path, tmp_path: Path) -> None:
    rounds = list(range(1, 13))
    with multiprocessing.Pool(4) as pool:
        pool.map(_append_worker, [(str(ledger), n) for n in rounds])
    lines = [ln for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(rounds)
    for ln in lines:
        json.loads(ln)  # every line intact (no interleaved partial writes)
    state = rc.current_state("concurrent", ledger_path=ledger)
    assert state.rounds_recorded == len(rounds)
    assert state.sealed


def test_subprocess_children_honor_bulletin_env_no_live_leak(
    ledger: Path, tmp_path: Path
) -> None:
    """The round-1 residual, now fixed: multiprocessing children's fail-open bulletin posts
    must NOT reach the live store. The autouse ``_isolate_bulletin`` fixture sets the
    ``TAC_SESSION_BULLETIN_*`` env vars (inherited by spawned children); this asserts every
    child's ``verdict_landed`` event landed in the env-redirected tmp store (which is NOT the
    live default), proving zero leak."""
    from tac.session_bus import bulletin as _bull

    # The env var (set by the autouse fixture) points here; a spawned child re-imports
    # ``bulletin`` fresh, so its ``DEFAULT_BULLETIN_PATH`` is the REAL live store — the ONLY
    # way a child's event lands in this tmp file is by honoring the inherited env var. If the
    # fix were absent, this file would have 0 events and the live store would grow.
    redirected = tmp_path / "ev.jsonl"
    rounds = list(range(1, 13))
    with multiprocessing.Pool(4) as pool:
        pool.map(_append_worker, [(str(ledger), n) for n in rounds])
    events = _bull.read_events(bulletin_path=redirected)
    verdicts = [e for e in events if e["kind"] == _bull.EVENT_VERDICT_LANDED]
    assert len(verdicts) == len(rounds)  # every child's post landed in tmp, none in live
    assert {e["subject"] for e in verdicts} == {"concurrent"}
