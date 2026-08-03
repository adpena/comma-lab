"""Behaviour tests for the sweep-durability helpers.

Each test would FAIL if the function body were replaced by a canonical-looking constant:
every assertion depends on files actually on disk and on the arguments passed.
"""

from __future__ import annotations

import pytest

from tac.optimization.sweep_durability import (
    DONE,
    NOT_STARTED,
    RUNNING,
    STALLED,
    job_state,
    missing_units,
    resumable_units,
    unit_receipt_path,
)


def _touch(d, unit, suffix=".npz"):
    p = unit_receipt_path(d, unit, suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def test_receipt_path_is_deterministic_and_tuple_aware(tmp_path) -> None:
    assert unit_receipt_path(tmp_path, (3, 5)).name == "u_3_5.npz"
    assert unit_receipt_path(tmp_path, 7).name == "u_7.npz"
    assert unit_receipt_path(tmp_path, (3, 5)) == unit_receipt_path(tmp_path, (3, 5))


@pytest.mark.parametrize("bad", ["", "a/b", ".."])
def test_receipt_path_refuses_unsafe_keys(tmp_path, bad: str) -> None:
    with pytest.raises(ValueError):
        unit_receipt_path(tmp_path, bad)


def test_resumable_units_returns_everything_when_nothing_ran(tmp_path) -> None:
    todo, done = resumable_units(tmp_path, [(0, 0), (0, 1), (1, 0)])
    assert todo == [(0, 0), (0, 1), (1, 0)]
    assert done == []


def test_resumable_units_skips_exactly_the_units_on_disk(tmp_path) -> None:
    _touch(tmp_path, (0, 1))
    todo, done = resumable_units(tmp_path, [(0, 0), (0, 1), (1, 0)])
    assert todo == [(0, 0), (1, 0)]
    assert done == [(0, 1)]


def test_resumable_units_is_empty_once_all_are_present(tmp_path) -> None:
    units = [(0, 0), (0, 1)]
    for u in units:
        _touch(tmp_path, u)
    todo, done = resumable_units(tmp_path, units)
    assert todo == []
    assert done == units


def test_resumable_units_respects_the_suffix(tmp_path) -> None:
    _touch(tmp_path, (0, 0), ".json")
    todo, _ = resumable_units(tmp_path, [(0, 0)], suffix=".npz")
    assert todo == [(0, 0)]
    todo2, _ = resumable_units(tmp_path, [(0, 0)], suffix=".json")
    assert todo2 == []


def test_missing_units_is_the_todo_half(tmp_path) -> None:
    _touch(tmp_path, 1)
    assert missing_units(tmp_path, [1, 2, 3]) == [2, 3]


# ------------------------------------------------------------ job_state


def test_empty_dir_is_NOT_STARTED_not_a_vacuous_pass(tmp_path) -> None:
    st = job_state(tmp_path, expected_units=5)
    assert st["state"] == NOT_STARTED
    assert st["units_done"] == 0
    assert st["units_expected"] == 5


def test_missing_dir_is_NOT_STARTED(tmp_path) -> None:
    assert job_state(tmp_path / "nope")["state"] == NOT_STARTED


def test_partial_and_fresh_is_RUNNING(tmp_path) -> None:
    _touch(tmp_path, 0)
    st = job_state(tmp_path, expected_units=3)
    assert st["state"] == RUNNING
    assert st["units_done"] == 1
    assert st["newest_receipt_age_s"] <= st["stall_after_s"]


def test_partial_and_stale_is_STALLED_not_RUNNING(tmp_path) -> None:
    _touch(tmp_path, 0)
    st = job_state(tmp_path, expected_units=3, now=lambda: 10**12)
    assert st["state"] == STALLED
    assert st["newest_receipt_age_s"] > st["stall_after_s"]


def test_complete_is_DONE_even_when_stale(tmp_path) -> None:
    for u in range(3):
        _touch(tmp_path, u)
    st = job_state(tmp_path, expected_units=3, now=lambda: 10**12)
    assert st["state"] == DONE


def test_done_beats_stalled_precedence(tmp_path) -> None:
    for u in range(4):
        _touch(tmp_path, u)
    assert job_state(tmp_path, expected_units=3, now=lambda: 10**12)["state"] == DONE


def test_unknown_expected_count_never_reports_DONE(tmp_path) -> None:
    _touch(tmp_path, 0)
    assert job_state(tmp_path)["state"] in {RUNNING, STALLED}


def test_state_reports_its_denominator_and_its_evidence(tmp_path) -> None:
    _touch(tmp_path, 0)
    st = job_state(tmp_path, expected_units=9)
    assert st["units_expected"] == 9
    assert "process table was NOT consulted" in st["evidence"]


def test_stall_threshold_is_honoured_from_the_argument(tmp_path) -> None:
    _touch(tmp_path, 0)
    late = 10**12
    assert job_state(tmp_path, stall_after_s=10**13, now=lambda: late)["state"] == RUNNING
    assert job_state(tmp_path, stall_after_s=1.0, now=lambda: late)["state"] == STALLED
