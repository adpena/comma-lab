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
    unreadable_task_ids,
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
    """The ROW is rejected; the FILE is not moved (blast radius, 2026-08-17).

    Originally this asserted ``CanonicalTaskStatusCorruptError``, which also
    QUARANTINED -- moved away -- the whole ledger.  A live outage measured what
    that costs: one arm appended a row missing its ``[empirical:]`` custody note
    and the shared ledger vanished mid-run for the entire fleet.

    Missing audit fields are the SAME class: a well-formed JSON object failing a
    field relationship.  The test for quarantine is whether the remaining bytes
    can still be parsed -- for unparseable JSON they cannot (that path still
    raises, pinned in ``test_canonical_task_status_isolation.py``); for a bad
    field they can.  So rejection is preserved and its blast radius is bounded:
    the row is excluded, the task is named, the file stays put.
    """
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
    before = ledger.read_bytes()

    with pytest.warns(UserWarning, match="UNREADABLE"):
        rows = load_canonical_task_status_strict(tmp_path)
    assert rows == [], "a row missing its audit fields must never be served"

    broken = unreadable_task_ids(tmp_path)
    assert set(broken) == {"memo::ITEM_BAD"}
    assert "row contract violated" in broken["memo::ITEM_BAD"]

    assert ledger.exists(), "one bad row must not move the shared ledger"
    assert ledger.read_bytes() == before
    assert list(ledger.parent.glob("*.corrupt.*")) == []


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


# ── SPECIFICATION WITHOUT REGISTRATION (ddm_rg5 #825) ────────────────────────────────
def _focus_repo(tmp_path: Path, focus_text: str, *, register: list[str] = ()) -> Path:
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / ".omx" / "research").mkdir(parents=True)
    memo = tmp_path / ".omx" / "research" / "m_20260731.md"
    memo.write_text("# memo\n", encoding="utf-8")
    for tid in register:
        register_task(tid, ".omx/research/m_20260731.md", f"t{tid}", "owner",
                      actor="t", session_id="s", repo_root=tmp_path)
    (tmp_path / ".omx" / "state" / "current_focus.md").write_text(
        focus_text, encoding="utf-8")
    return tmp_path


def test_unregistered_task_ref_in_current_focus_is_surfaced(tmp_path: Path) -> None:
    """Prose is not existence.

    The receipt (R1-C, 2026-07-31): #815 was skipped in the ledger, #822 likewise, and MAIN then
    asserted a BLOCKING RELATION between #815 and #820 — neither of which had a row at all.
    Every downstream reader took the focus doc's ``#NNN`` as evidence the task existed.
    """
    from tac.canonical_task_status.checks import unregistered_task_refs_in_current_focus

    root = _focus_repo(tmp_path, "burn #815 blocked by #820; #816 done\n", register=["816"])
    v = unregistered_task_refs_in_current_focus(root)
    assert len(v) == 2
    assert any("#815" in x for x in v) and any("#820" in x for x in v)
    assert all("#816" not in x for x in v)


def test_registration_check_ignores_ids_below_the_baseline(tmp_path: Path) -> None:
    """A bare ``#NNN`` cannot distinguish a TASK from a CATALOG number.

    ``#396`` and ``#316`` appear in the live focus doc as catalog GATES, and the JSONL holds
    historical rows while the live TaskList is SoT for #200+. Firing on those would make the gate
    noise, and a noisy gate is an ignored gate — the failure this whole task exists to break.
    """
    from tac.canonical_task_status.checks import unregistered_task_refs_in_current_focus

    root = _focus_repo(tmp_path, "Catalog #396 warn-only; #316 strict; task #825 open\n")
    v = unregistered_task_refs_in_current_focus(root)
    assert len(v) == 1 and "#825" in v[0]


def test_registration_violations_flow_into_the_canonical_gate(tmp_path: Path) -> None:
    from tac.canonical_task_status import canonical_task_status_violations

    root = _focus_repo(tmp_path, "#899 is the next thing\n")
    assert any("#899" in x for x in canonical_task_status_violations(root))


def test_the_backfilled_orphans_are_registered() -> None:
    """The eleven ids ddm_rg5 (#825) backfilled must stay registered.

    Deliberately NOT ``unregistered_task_refs_in_current_focus() == []``: ``current_focus.md`` is
    edited continuously by concurrent arms — three NEW unregistered ids (#826/#827/#828) appeared
    in it while this landing was in flight — so a live-count-0 unit test would flake on other
    arms' prose rather than on a defect. The live ratchet belongs in the preflight gate, which
    runs against the tree as it is at commit time; the durable, non-racy property a test can own
    is that the known orphans did not silently disappear again.
    """
    from tac.canonical_task_status import latest_status_by_task_id

    for tid in ("809", "815", "819", "820", "821", "822", "824", "825", "826", "827", "828"):
        row = latest_status_by_task_id(tid)
        assert row is not None, f"#{tid} lost its registration"
        assert row.title.strip(), f"#{tid} registered with an empty title"
