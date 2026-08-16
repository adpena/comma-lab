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
