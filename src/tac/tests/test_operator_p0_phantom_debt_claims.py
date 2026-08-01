"""Controls for the phantom-debt cross-check in tools/operator_p0_digest.py.

WHY THIS EXISTS (MEASURED, 2026-08-01): MAIN picked up two open rows and began
triage. Both had been fixed hours earlier — `b02b99cecb` is titled
"ddm_tr6 (#851): triage the 6 CI-blind reds", `57d4747e60` "ddm_rt1 #845: ...
test was STALE". Nothing said resolved. A sweep found 13 of 54 open rows carrying
a commit that NAMES them.

The check is a PROMPT, never a verdict — so the tests below pin BOTH directions:
it must fire on a real claim, and it must never render an un-run scan as "none".
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import operator_p0_digest as m  # noqa: E402


# --------------------------------------------------------------------------
# row_task_numbers — both sources, because the ledger populates them unevenly.
# --------------------------------------------------------------------------


def test_numbers_come_from_task_ids_and_from_the_id_itself() -> None:
    """Six live rows carry their number ONLY in the p0_id; reading one source under-scopes."""
    assert m.row_task_numbers({"p0_id": "p0_366_joint_pose_finishing"}) == {366}
    assert m.row_task_numbers(
        {"p0_id": "p0_ema_calibration_20260717", "task_ids": ["408", "404"]}
    ) == {404, 408}


def test_multi_task_rows_keep_every_number() -> None:
    """MEASURED live: p0_408 tracks {408,404}; p0_497 tracks {497,502}."""
    row = {"p0_id": "p0_497_basis_cure_decisive_ab", "task_ids": ["497", "502"]}
    assert m.row_task_numbers(row) == {497, 502}


def test_dates_in_ids_are_not_mistaken_for_task_numbers() -> None:
    """`…_20260717` must not register as task 2026 / 0717."""
    assert m.row_task_numbers({"p0_id": "p0_lane_three_cruxes_20260717"}) == set()


def test_scalar_task_ids_field_is_accepted() -> None:
    assert m.row_task_numbers({"p0_id": "p0_x", "task_ids": "851"}) == {851}


# --------------------------------------------------------------------------
# The matcher: fires on a real claim, and does NOT fire on a near-miss.
# --------------------------------------------------------------------------


def _fake_subjects(monkeypatch, subjects):
    monkeypatch.setattr(m, "_git_subjects", lambda root, lookback: subjects)


def test_positive_control_reproduces_the_851_incident(monkeypatch) -> None:
    """The exact 2026-08-01 shape: an open row whose fix commit names it."""
    _fake_subjects(monkeypatch, [
        ("b02b99cecb", "ddm_tr6 (#851): triage the 6 CI-blind reds rt1 surfaced"),
        ("deadbeef00", "unrelated work with no task reference"),
    ])
    rows = [{"p0_id": "p0_851_ci_blind", "task_ids": ["851"]}]
    rep = m.claiming_commits(".", rows)
    assert rep["status"] == "COMPLETE"
    assert rep["scanned_commits"] == 2
    assert rep["examined_rows"] == 1
    assert rep["rows_with_numbers"] == 1
    assert list(rep["claims"]) == ["p0_851_ci_blind"]
    assert rep["claims"]["p0_851_ci_blind"][0][2] == 851, "matched number must be carried"


def test_the_matched_number_is_rendered(monkeypatch) -> None:
    """The first live run needed a SECOND grep to confirm a match was legitimate.

    A reader who cannot see which number matched cannot verify the row without
    redoing the work — so the number is part of the contract, not decoration.
    """
    _fake_subjects(monkeypatch, [
        ("15aad5a28b", "tr1 v9 telemetry port (#804): also closes #404 and #304"),
    ])
    rows = [{"p0_id": "p0_408_telemetry", "task_ids": ["408", "404"]}]
    out = m.format_claims(m.claiming_commits(".", rows))
    assert "matched #404" in out, "must show WHICH number matched, not just the sha"


def test_longer_numbers_do_not_match_a_shorter_task(monkeypatch) -> None:
    """`#8510` must NOT satisfy task 851 — a substring match is a false claim."""
    _fake_subjects(monkeypatch, [("cafe000000", "work on #8510 and #1851")])
    rows = [{"p0_id": "p0_851_x", "task_ids": ["851"]}]
    assert m.claiming_commits(".", rows)["claims"] == {}


def test_unnumbered_rows_are_excluded_from_the_denominator(monkeypatch) -> None:
    """16 of 22 live rows carry no number; counting them would understate coverage."""
    _fake_subjects(monkeypatch, [("aaa0000000", "something about #366")])
    rows = [
        {"p0_id": "p0_366_joint_pose_finishing"},
        {"p0_id": "p0_lane_three_cruxes_20260717"},
    ]
    rep = m.claiming_commits(".", rows)
    assert rep["examined_rows"] == 2
    assert rep["rows_with_numbers"] == 1, "the date-only row has no number to check"


# --------------------------------------------------------------------------
# VACUITY — an un-run scan must never render as "nothing claimed".
# --------------------------------------------------------------------------


def test_git_unavailable_is_VACUOUS_not_clean(monkeypatch) -> None:
    """The day's genus: empty scope must not emit the clean-pass symbol."""
    monkeypatch.setattr(m, "_git_subjects", lambda root, lookback: None)
    rep = m.claiming_commits(".", [{"p0_id": "p0_851_x", "task_ids": ["851"]}])
    assert rep["status"] == "VACUOUS_NO_GIT"
    assert rep["scanned_commits"] is None, "None != 0; 'could not run' != 'found none'"
    out = m.format_claims(rep)
    assert "VACUOUS" in out and "DID NOT RUN" in out
    assert "none" not in out.lower().replace("nothing", ""), "must not read as a clean pass"


def test_clean_result_still_states_its_denominator(monkeypatch) -> None:
    """A real 'none' is fine — but only when it says how much it looked at."""
    _fake_subjects(monkeypatch, [("aaa0000000", "no task refs here")])
    out = m.format_claims(m.claiming_commits(".", [{"p0_id": "p0_1_x", "task_ids": ["851"]}]))
    assert "1 examined" in out and "last 1" in out and "none" in out


def test_no_rows_is_not_reported_as_a_pass(monkeypatch) -> None:
    _fake_subjects(monkeypatch, [("aaa0000000", "subject")])
    assert m.claiming_commits(".", [])["status"] == "VACUOUS_NO_ROWS"


# --------------------------------------------------------------------------
# The hook must stay FAIL-OPEN. A memory surface that wedges a session is worse
# than the forgetting it prevents.
# --------------------------------------------------------------------------


def test_session_start_hook_still_exits_zero_with_the_claims_line() -> None:
    assert m.main(["--session-start"]) == 0


def test_session_start_hook_exits_zero_even_if_the_scan_explodes(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("git exploded")
    monkeypatch.setattr(m, "claiming_commits", boom)
    assert m.main(["--session-start"]) == 0


def test_claims_cli_runs_against_the_live_ledger() -> None:
    assert m.main(["--claims"]) == 0


def test_explicit_task_ids_are_never_plausibility_filtered() -> None:
    """`task_ids` is authoritative — a human wrote it; it takes NO year guard.

    Review-pass finding on this module: an earlier draft applied a `< 2000`
    bound to EVERY source, which would silently drop explicit rows once task
    ids pass 2000 — the exact silent-under-scoping bug the check exists to cure.
    """
    assert m.row_task_numbers({"p0_id": "p0_x", "task_ids": ["2026"]}) == {2026}


def test_inferred_id_numbers_still_take_the_year_guard() -> None:
    """A bare `_2026_` SEGMENT of an id is a year, not a task."""
    assert m.row_task_numbers({"p0_id": "p0_something_2026_x"}) == set()
    assert m.row_task_numbers({"p0_id": "p0_366_x_2026_y"}) == {366}


def test_zero_commits_is_VACUOUS_even_though_git_worked(monkeypatch) -> None:
    """Review pass 2: git SUCCEEDING with an empty history is still empty scope.

    It would have rendered "0 of N ... in the last 0 — none" — denominator
    honest, but still carrying the clean-pass word. Empty scope gets its own
    symbol rather than trusting a reader to notice a zero.
    """
    _fake_subjects(monkeypatch, [])
    rep = m.claiming_commits(".", [{"p0_id": "p0_851_x", "task_ids": ["851"]}])
    assert rep["status"] == "VACUOUS_NO_COMMITS"
    assert rep["scanned_commits"] == 0, "0 != None; git ran, history was empty"
    out = m.format_claims(rep)
    assert "VACUOUS" in out and "EMPTY history" in out
