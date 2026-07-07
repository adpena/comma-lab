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


def test_concurrent_appends_all_land_intact(ledger: Path) -> None:
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
