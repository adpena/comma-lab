# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_duckdb import refresh_table
from tac.canonical_task_status import (
    CanonicalTaskStatusCorruptError,
    CanonicalTaskStatusInvalidTransitionError,
    append_note,
    check_canonical_task_status_no_dangling_transitions,
    latest_status_by_task_id,
    load_canonical_task_status_strict,
    query_task_history,
    query_tasks_by_status,
    register_task,
    update_status,
)


def test_register_update_query_and_history_are_append_only(tmp_path: Path) -> None:
    (tmp_path / ".omx/research").mkdir(parents=True)
    (tmp_path / ".omx/research/memo.md").write_text("# Memo\n")
    row = register_task(
        "memo::ITEM_1",
        ".omx/research/memo.md",
        "Build canonical helper",
        "codex",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
        predicted_delta_s_band=(-0.2, -0.1),
    )
    duplicate = register_task(
        "memo::ITEM_1",
        ".omx/research/memo.md",
        "Different title ignored by idempotence",
        "codex",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    assert duplicate == row

    in_progress = update_status(
        "memo::ITEM_1",
        "in_progress",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    append_note(
        "memo::ITEM_1",
        "still running",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    completed = update_status(
        "memo::ITEM_1",
        "completed",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
        test_status="green",
        commit_shas=("abc123",),
    )

    assert in_progress.started_at_utc is not None
    assert completed.status == "completed"
    assert latest_status_by_task_id("memo::ITEM_1", tmp_path).status == "completed"  # type: ignore[union-attr]
    assert query_tasks_by_status("pending", owner="codex", repo_root=tmp_path) == []
    assert len(query_task_history("memo::ITEM_1", repo_root=tmp_path)) == 4
    assert len({
        row.event_timestamp_utc
        for row in query_task_history("memo::ITEM_1", repo_root=tmp_path)
    }) == 4
    ledger = tmp_path / ".omx/state/canonical_task_status.jsonl"
    assert len(ledger.read_text().splitlines()) == 4
    assert check_canonical_task_status_no_dangling_transitions(repo_root=tmp_path) == []


def test_invalid_state_machine_transition_refuses_write(tmp_path: Path) -> None:
    register_task(
        "memo::ITEM_2",
        ".omx/research/memo.md",
        "Bad transition",
        "codex",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    with pytest.raises(CanonicalTaskStatusInvalidTransitionError):
        update_status(
            "memo::ITEM_2",
            "completed",
            actor="codex_test",
            session_id="s1",
            repo_root=tmp_path,
        )
    assert len(load_canonical_task_status_strict(tmp_path)) == 1


def test_actual_delta_requires_empirical_note(tmp_path: Path) -> None:
    register_task(
        "memo::ITEM_3",
        ".omx/research/memo.md",
        "Empirical discipline",
        "codex",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    update_status(
        "memo::ITEM_3",
        "in_progress",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    with pytest.raises(ValueError, match="empirical"):
        update_status(
            "memo::ITEM_3",
            "completed",
            actor="codex_test",
            session_id="s1",
            repo_root=tmp_path,
            actual_delta_s=-0.01,
        )


def test_note_after_empirical_completion_carries_evidence_tag(tmp_path: Path) -> None:
    register_task(
        "memo::ITEM_5",
        ".omx/research/memo.md",
        "Empirical note",
        "codex",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    update_status(
        "memo::ITEM_5",
        "in_progress",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    update_status(
        "memo::ITEM_5",
        "completed",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
        actual_delta_s=-0.01,
        notes="[empirical:experiments/results/example.json] measured anchor",
    )
    note = append_note(
        "memo::ITEM_5",
        "operator reviewed",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    assert note.actual_delta_s == pytest.approx(-0.01)
    assert "[empirical:experiments/results/example.json]" in note.event_notes


def test_strict_loader_rejects_missing_audit_fields(tmp_path: Path) -> None:
    ledger = tmp_path / ".omx/state/canonical_task_status.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "canonical_task_status_v1_20260518",
                "task_id": "memo::ITEM_BAD",
                "source_design_memo": ".omx/research/memo.md",
                "title": "Bad",
                "status": "pending",
                "owner": "codex",
                "predicted_cost_usd": None,
                "predicted_delta_s_band": None,
                "actual_delta_s": None,
                "commit_shas": [],
                "test_status": "pending",
                "blockers": [],
                "started_at_utc": None,
                "completed_at_utc": None,
                "event_type": "registered",
                "event_timestamp_utc": "",
                "event_actor": "codex_test",
                "event_notes": "",
                "session_id": "",
                "written_at_utc": "",
                "written_pid": 0,
                "written_host": "",
            }
        )
        + "\n"
    )
    with pytest.raises(CanonicalTaskStatusCorruptError):
        load_canonical_task_status_strict(tmp_path)


def test_duckdb_refresh_exposes_latest_view(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    assert duckdb is not None
    register_task(
        "memo::ITEM_4",
        ".omx/research/memo.md",
        "DuckDB view",
        "codex",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    update_status(
        "memo::ITEM_4",
        "blocked",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
        blockers=("waiting_for_probe",),
    )
    result = refresh_table(
        "canonical_task_status",
        tmp_path,
        db_path=tmp_path / ".omx/state/canonical.duckdb",
    )
    assert result["row_count"] == 2
    con = duckdb.connect(str(tmp_path / ".omx/state/canonical.duckdb"), read_only=True)
    try:
        rows = con.execute(
            "SELECT task_id, status, blockers FROM canonical_task_status_latest"
        ).fetchall()
    finally:
        con.close()
    assert rows == [("memo::ITEM_4", "blocked", json.dumps(["waiting_for_probe"]))]


def test_duckdb_refresh_preserves_last_good_table_on_corrupt_ledger(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    assert duckdb is not None
    register_task(
        "memo::ITEM_6",
        ".omx/research/memo.md",
        "Last good",
        "codex",
        actor="codex_test",
        session_id="s1",
        repo_root=tmp_path,
    )
    db_path = tmp_path / ".omx/state/canonical.duckdb"
    refresh_table("canonical_task_status", tmp_path, db_path=db_path)
    (tmp_path / ".omx/state/canonical_task_status.jsonl").write_text("{bad json\n")
    with pytest.raises(CanonicalTaskStatusCorruptError):
        refresh_table("canonical_task_status", tmp_path, db_path=db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("SELECT task_id FROM canonical_task_status_latest").fetchall()
    finally:
        con.close()
    assert rows == [("memo::ITEM_6",)]
