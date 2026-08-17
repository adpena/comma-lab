# SPDX-License-Identifier: MIT
"""Controls for PER-TASK isolation in the canonical task-status ledger.

The 2026-08-16 outage verbatim: ONE orphan row -- a lone ``completion`` for a
task never registered, appended as line 548 of 548 -- made
``load_canonical_task_status_strict`` and ``canonical_task_status_violations``
BOTH refuse.  Every writer in the fleet was blocked for ~7h and a live arm
could not file its rows at all.  A single unreadable task had been rendered as
total death.

These tests pin the cure and its boundaries.  They live in their own module
rather than in ``test_canonical_task_status.py`` because that file carries a
sister agent's in-flight work; editing it would absorb their uncommitted lines
into my commit (Catalog #314/#340, task #911).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from tac.canonical_task_status import (
    CanonicalTaskStatusCorruptError,
    latest_status_by_task_id,
    load_canonical_task_status_strict,
    register_task,
    unreadable_task_ids,
    update_status,
)

LEDGER = Path(".omx/state/canonical_task_status.jsonl")


def _append_orphan_completion(repo_root: Path, task_id: str) -> None:
    """Append a lone ``completion`` for a task that was never registered."""

    ledger = repo_root / LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": "canonical_task_status_v1_20260518",
                    "task_id": task_id,
                    "source_design_memo": ".omx/research/memo.md",
                    "title": "Orphan",
                    "status": "completed",
                    "owner": "codex",
                    "predicted_cost_usd": None,
                    "predicted_delta_s_band": None,
                    "actual_delta_s": None,
                    "commit_shas": [],
                    "test_status": "pending",
                    "blockers": [],
                    "started_at_utc": None,
                    "completed_at_utc": "2026-08-16T00:00:00Z",
                    "event_type": "completion",
                    "event_timestamp_utc": "2026-08-16T00:00:00Z",
                    "event_actor": "codex_test",
                    "event_notes": "orphan",
                    "session_id": "s1",
                    "written_at_utc": "2026-08-16T00:00:00Z",
                    "written_pid": 1,
                    "written_host": "test",
                }
            )
            + "\n"
        )


def _register(repo_root: Path, task_id: str, title: str) -> None:
    register_task(
        task_id,
        ".omx/research/memo.md",
        title,
        "codex",
        actor="codex_test",
        session_id="s1",
        repo_root=repo_root,
    )


def test_orphan_isolates_to_its_task_and_the_rest_still_load(tmp_path: Path) -> None:
    """One broken history must not be a fleet-wide outage."""
    _register(tmp_path, "memo::GOOD", "Healthy")
    _append_orphan_completion(tmp_path, "memo::ORPHAN")

    with pytest.warns(UserWarning, match="UNREADABLE"):
        rows = load_canonical_task_status_strict(tmp_path)
    assert [row.task_id for row in rows] == ["memo::GOOD"]

    broken = unreadable_task_ids(tmp_path)
    assert set(broken) == {"memo::ORPHAN"}
    assert "unknown task_id" in broken["memo::ORPHAN"]


def test_writes_still_proceed_with_an_orphan_present(tmp_path: Path) -> None:
    """The outage's actual cost was BLOCKED WRITERS -- pin that they work."""
    _register(tmp_path, "memo::GOOD", "Healthy")
    _append_orphan_completion(tmp_path, "memo::ORPHAN")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        update_status(
            "memo::GOOD",
            "in_progress",
            actor="codex_test",
            session_id="s1",
            repo_root=tmp_path,
        )
        latest = latest_status_by_task_id("memo::GOOD", tmp_path)
    assert latest is not None
    assert latest.status == "in_progress"


def test_exclusion_is_loud_never_silent(tmp_path: Path) -> None:
    """A quiet skip is how 'the ledger looks fine' becomes a lie (vacuity==pass)."""
    _register(tmp_path, "memo::GOOD", "Healthy")
    _append_orphan_completion(tmp_path, "memo::ORPHAN")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_canonical_task_status_strict(tmp_path)
    messages = [str(w.message) for w in caught]
    assert any("memo::ORPHAN" in m for m in messages), messages
    assert any("UNREADABLE" in m for m in messages), messages


def test_clean_ledger_is_silent_and_reports_nothing_unreadable(tmp_path: Path) -> None:
    _register(tmp_path, "memo::CLEAN", "Clean")
    assert unreadable_task_ids(tmp_path) == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a clean ledger must NOT warn
        assert len(load_canonical_task_status_strict(tmp_path)) == 1


def test_a_later_row_cannot_repair_a_broken_history(tmp_path: Path) -> None:
    """Append-only means the break stands until a corrected lifecycle is added."""
    _append_orphan_completion(tmp_path, "memo::ORPHAN")
    _register(tmp_path, "memo::LATER", "Registered after the orphan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rows = load_canonical_task_status_strict(tmp_path)
    assert [row.task_id for row in rows] == ["memo::LATER"]
    assert set(unreadable_task_ids(tmp_path)) == {"memo::ORPHAN"}


def test_unreadable_task_ids_refuses_on_corrupt_line_without_quarantining(
    tmp_path: Path,
) -> None:
    """A read-only audit must not rename the file, nor answer from a partial read.

    Swallowing the bad line would be actively misleading: dropping a
    ``registered`` row makes every later row for that task look orphaned, so a
    swallow here would INVENT unreadable tasks.
    """
    ledger = tmp_path / LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(CanonicalTaskStatusCorruptError):
        unreadable_task_ids(tmp_path)
    assert ledger.exists()  # NOT quarantined by the audit path
    assert list(ledger.parent.glob("*.corrupt.*")) == []


def test_missing_ledger_is_valid_and_empty(tmp_path: Path) -> None:
    assert unreadable_task_ids(tmp_path) == {}
    assert load_canonical_task_status_strict(tmp_path) == []


def _append_contract_violating_completion(repo_root: Path, task_id: str) -> None:
    """A well-formed JSON object that fails ONE field relationship.

    ``contract.py:280`` requires an ``[empirical:<path>]`` note on any row
    carrying ``actual_delta_s``.  This row parses perfectly as JSON and violates
    exactly that relationship -- the shape of the 2026-08-17 incident.
    """

    ledger = repo_root / LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": "canonical_task_status_v1_20260518",
                    "task_id": task_id,
                    "source_design_memo": ".omx/research/memo.md",
                    "title": "Carries a delta",
                    "status": "completed",
                    "owner": "codex",
                    "predicted_cost_usd": None,
                    "predicted_delta_s_band": None,
                    "actual_delta_s": -0.0983195,
                    "commit_shas": [],
                    "test_status": "pending",
                    "blockers": [],
                    "started_at_utc": None,
                    "completed_at_utc": "2026-08-17T00:00:00Z",
                    "event_type": "completion",
                    "event_timestamp_utc": "2026-08-17T00:00:00Z",
                    "event_actor": "codex_test",
                    "event_notes": "no empirical tag here",
                    "session_id": "s1",
                    "written_at_utc": "2026-08-17T00:00:00Z",
                    "written_pid": 1,
                    "written_host": "test",
                }
            )
            + "\n"
        )


def test_row_contract_violation_does_not_move_the_ledger(tmp_path: Path) -> None:
    """THE 2026-08-17 INCIDENT: one arm's bad row must not delete shared state.

    A row failing the ``[empirical:]`` contract used to be read as file
    corruption, and the caller QUARANTINED -- moved -- the whole ledger. A live
    arm found the file gone mid-run. The ledger must stay exactly where it is.
    """
    _register(tmp_path, "memo::GOOD", "Healthy")
    _register(tmp_path, "memo::BAD", "Violator")
    _append_contract_violating_completion(tmp_path, "memo::BAD")
    ledger = tmp_path / LEDGER
    before = ledger.read_bytes()

    with pytest.warns(UserWarning, match="UNREADABLE"):
        rows = load_canonical_task_status_strict(tmp_path)

    assert ledger.exists(), "the shared ledger was MOVED by one bad row"
    assert ledger.read_bytes() == before, "the ledger was rewritten, not just read"
    assert list(ledger.parent.glob("*.corrupt.*")) == [], "quarantined a contract violation"
    assert [row.task_id for row in rows] == ["memo::GOOD"]

    broken = unreadable_task_ids(tmp_path)
    assert set(broken) == {"memo::BAD"}
    assert "row contract violated" in broken["memo::BAD"]
    assert "empirical" in broken["memo::BAD"], "the offender's reason must name the cause"


def test_file_corruption_still_quarantines(tmp_path: Path) -> None:
    """The path I must NOT have broken: unparseable bytes are a different class.

    A line that is not JSON makes later offsets untrustworthy, so refusing and
    quarantining stays correct. Splitting the two classes must not soften this.
    """
    _register(tmp_path, "memo::GOOD", "Healthy")
    ledger = tmp_path / LEDGER
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")

    with pytest.raises(CanonicalTaskStatusCorruptError):
        load_canonical_task_status_strict(tmp_path)
    assert not ledger.exists(), "file corruption must still quarantine"
    assert len(list(ledger.parent.glob("*.corrupt.*"))) == 1


def test_a_non_object_row_is_file_corruption_not_a_contract_violation(tmp_path: Path) -> None:
    """A bare JSON scalar parses, but there is no row to hold to a contract."""
    ledger = tmp_path / LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text("42\n", encoding="utf-8")
    with pytest.raises(CanonicalTaskStatusCorruptError):
        unreadable_task_ids(tmp_path)
    assert ledger.exists(), "the read-only audit must not quarantine"
