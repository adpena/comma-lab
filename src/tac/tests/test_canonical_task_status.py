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
    # FULL ΔS custody supplied, empirical tag deliberately withheld: the ORIGINAL
    # invariant must still be the one that fires. Pinned this way so the ddm_op3 custody
    # clause cannot mask the older evidence clause by raising first on an unrelated
    # defect -- two independent gates, each provably reachable.
    with pytest.raises(ValueError, match="empirical"):
        update_status(
            "memo::ITEM_3",
            "completed",
            actor="codex_test",
            session_id="s1",
            repo_root=tmp_path,
            actual_delta_s=-0.01,
            notes="[baseline:eval_root/submissions/v4d_ms8/report.txt=0.8984335] [n600]",
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
        notes=(
            "[empirical:experiments/results/example.json] measured anchor "
            "[baseline:eval_root/submissions/v4d_ms8/report.txt=0.8984335] [n600]"
        ),
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


# ---------------------------------------------------------------------------
# ddm_op3 (2026-08-03) -- ΔS CUSTODY CLAUSES.
#
# The two rows that actually misdirected readers this week are REPLAYED here verbatim.
# That is the point: a gate justified by incidents it cannot reproduce is a story, and
# the obvious design (one more clause on `actual_delta_s`) would have PASSED both of
# them, because both carry actual_delta_s = None and state their number in `title`.
#
# MEASURED live counts over the 417-row ledger at landing:
#   8 rows carry a typed actual_delta_s      (0 of them name a baseline)
#   8 rows assert a ΔS only in free text     (1 overlaps the above)
#   -> 15 distinct rows assert a ΔS; the pre-existing invariant could see 1.
# New rows: live count 0 by construction, because the writer refuses at mint time and
# the reader only warns. No backfill is owed before this binds.
# ---------------------------------------------------------------------------

from tac.canonical_task_status.contract import (
    CanonicalTaskStatusRow,
    delta_s_custody_findings,
)
from tac.canonical_task_status.writer import DeltaSCustodyError

# VERBATIM from .omx/state/canonical_task_status.jsonl, task 826 and task 827.
#
# The 826 string below is byte-exact and it CORRECTS this test's own first draft, which
# rendered the delta as "= -0.0983195 S". The real row writes it as a bare parenthesised
# number with no S label anywhere. A control that replays an edited row is not a control;
# it certifies the detector against a case chosen to suit it. Fixing the fabrication is
# what surfaced the sharper rule: 826 DID name a reference ("vs ref 0.7685479"), so the
# defect was never a missing baseline -- it was an UN-RE-DERIVABLE one.
_REAL_826_TITLE = (
    "FIRE-ORDER-0: gr1_cell_drop50_archive.zip (359,221 B, sha256 a6398e44...) "
    "byte-closed at seg_plus_rate 0.6702284 vs ref 0.7685479 (-0.0983195), "
    "never through exact eval"
)
_REAL_827_TITLE = (
    "THE COMPOSITION ROW: ep854 seg base composed onto the v4d pose payload, "
    "byte-closed at 285,529 B (seg+rate -0.0866789 S = 11.178% of gap)"
)


def _findings(title: str, notes: str = "", delta: float | None = None) -> tuple[str, ...]:
    return delta_s_custody_findings(actual_delta_s=delta, event_notes=notes, title=title)


def test_POSITIVE_CONTROL_the_real_826_row_is_caught():
    """KNOWN-BAD, replayed VERBATIM. Its -0.0983195 was measured against a v4d-era
    reference and re-prices to +0.0034632 against the live best -- an INVERSION. It
    carries actual_delta_s=None AND no S label, so it is reachable only through the
    reference-comparison clause."""
    found = _findings(_REAL_826_TITLE)
    assert any(f.startswith("MISSING_BASELINE") for f in found)
    assert any(f.startswith("UNDECLARED_PARTIAL_COMPOSITE") for f in found)
    # The specific diagnosis, not just "something is missing".
    assert any("bare reference NUMBER" in f for f in found)


def test_MEASURED_LIMIT_an_unlabelled_bare_delta_is_still_invisible():
    """The honest boundary of this instrument, pinned so it cannot be overstated.

    Detection keys on CLAIM LANGUAGE -- an S label, or reference-comparison prose. A
    delta written as a naked parenthesised number with neither would pass, and widening
    to 'any signed decimal' would fire on coordinates, byte counts and ratios across the
    whole ledger. This test exists so the next reader learns that limit from the suite
    rather than from a missed row."""
    assert _findings("some row (-0.0983195), never through exact eval") == ()


def test_POSITIVE_CONTROL_the_real_827_row_is_caught_as_a_partial_composite():
    """KNOWN-BAD, replayed verbatim. The sharpest case: a delta over a SUBSET of the S
    terms is not a ΔS. Its omitted pose term measured +19.302316 -- 234.7x the prize it
    was advertising -- so the real composed row is +19.22, not -0.0866."""
    found = _findings(_REAL_827_TITLE)
    assert any(f.startswith("UNDECLARED_PARTIAL_COMPOSITE") for f in found)
    assert any(f.startswith("MISSING_BASELINE") for f in found)


def test_POSITIVE_CONTROL_a_fully_custodied_row_passes():
    """KNOWN-GOOD. Without this the clause could be satisfied by refusing everything."""
    assert _findings(
        "gr1_cell_drop50 re-priced",
        notes=(
            "[empirical:eval_root/submissions/v4d_cx1_pj2ix2/report.txt] "
            "[baseline:eval_root/submissions/v4d_cx1_pj2ix2/report.txt=0.8264972] "
            "[n600] re-priced to +0.0034632 S"
        ),
        delta=+0.0034632,
    ) == ()


def test_POSITIVE_CONTROL_rows_that_assert_no_delta_are_not_touched():
    """The clause must be SILENT on the 402 of 417 rows that assert nothing. A gate that
    fires everywhere is noise, and noise is how a real finding gets ignored."""
    assert _findings("plain infrastructure task", notes="landed a loader; no score claim") == ()
    assert _findings("archive is 353,808 bytes at 0.00431179 d_seg") == ()


def test_partial_composite_may_not_occupy_the_typed_full_S_field():
    found = _findings(
        "seg+rate leg",
        notes="[empirical:x] [baseline:y=0.8264972] [n600] [partial:seg+rate]",
        delta=-0.0822362,
    )
    assert any(f.startswith("PARTIAL_IN_TYPED_FIELD") for f in found)
    # ... and with the typed field left None, the declared partial is accepted.
    assert _findings(
        "seg+rate leg",
        notes="[empirical:x] [baseline:y=0.8264972] [n600] [partial:seg+rate] -0.0822362 S",
    ) == ()


def test_population_coordinate_is_required():
    """n=73 read -0.122 WIN where its own n600 read +0.152 LOSS."""
    assert any(
        f.startswith("MISSING_POPULATION")
        for f in _findings("x", notes="[empirical:x] [baseline:y=0.82] delta -0.0100 S")
    )
    assert not any(
        f.startswith("MISSING_POPULATION")
        for f in _findings("x", notes="[empirical:x] [baseline:y=0.82] [n=73] -0.0100 S")
    )


def test_reader_WARNS_and_stays_total_over_history(tmp_path: Path) -> None:
    """The append-only ledger has 15 historical rows that predate this rule. A reader
    that RAISED on them would break campaign-wide recall -- strictly worse than the
    defect it reports. So: warn, and keep returning the row."""
    row = CanonicalTaskStatusRow(
        task_id="826",
        source_design_memo=".omx/research/memo.md",
        title=_REAL_826_TITLE,
        status="pending",
        owner="claude",
        event_type="registered",
        event_timestamp_utc="2026-07-31T20:30:18.741140Z",
        event_actor="claude",
        written_at_utc="2026-07-31T20:30:18.741140Z",
        written_pid=1,
        written_host="h",
        session_id="s",
    )
    with pytest.warns(UserWarning, match="without full custody"):
        rebuilt = CanonicalTaskStatusRow.from_json_obj(row.to_json_obj())
    assert rebuilt.title == _REAL_826_TITLE  # total: the row still comes back


def test_writer_REFUSES_a_new_uncustodied_delta(tmp_path: Path) -> None:
    """The other half of the split: no NEW uncustodied ΔS can be minted."""
    register_task(
        "memo::ITEM_OP3",
        ".omx/research/memo.md",
        "custody gate",
        "claude",
        actor="t",
        session_id="s1",
        repo_root=tmp_path,
    )
    update_status(
        "memo::ITEM_OP3", "in_progress", actor="t", session_id="s1", repo_root=tmp_path
    )
    with pytest.raises(DeltaSCustodyError, match="MISSING_BASELINE"):
        update_status(
            "memo::ITEM_OP3",
            "completed",
            actor="t",
            session_id="s1",
            repo_root=tmp_path,
            actual_delta_s=-0.0675451,
            notes="[empirical:eval_root/submissions/v4d_pj2/report.txt] pose win",
        )


def test_writer_REFUSES_a_free_text_delta_at_REGISTRATION(tmp_path: Path) -> None:
    """Registration is where the class actually entered: both real rows stated their ΔS
    in the title at register time with actual_delta_s=None."""
    with pytest.raises(DeltaSCustodyError):
        register_task(
            "memo::ITEM_OP3B",
            ".omx/research/memo.md",
            _REAL_826_TITLE,
            "claude",
            actor="t",
            session_id="s1",
            repo_root=tmp_path,
        )


def test_writer_ACCEPTS_a_custodied_delta(tmp_path: Path) -> None:
    """KNOWN-GOOD end-to-end: the gate is passable, so it is a gate and not a wall."""
    register_task(
        "memo::ITEM_OP3C",
        ".omx/research/memo.md",
        "custodied win",
        "claude",
        actor="t",
        session_id="s1",
        repo_root=tmp_path,
    )
    update_status(
        "memo::ITEM_OP3C", "in_progress", actor="t", session_id="s1", repo_root=tmp_path
    )
    row = update_status(
        "memo::ITEM_OP3C",
        "completed",
        actor="t",
        session_id="s1",
        repo_root=tmp_path,
        actual_delta_s=-0.0675451,
        notes=(
            "[empirical:eval_root/submissions/v4d_pj2/report.txt] "
            "[baseline:eval_root/submissions/v4d_ms8/report.txt=0.8984335] [n600]"
        ),
    )
    assert row.actual_delta_s == pytest.approx(-0.0675451)


def test_append_note_carries_forward_without_re_demanding_custody(tmp_path: Path) -> None:
    """You must custody what you ASSERT; you may carry forward what someone else did.
    Gating the carry-forward path would make historical tasks un-annotatable."""
    register_task(
        "memo::ITEM_OP3D", ".omx/research/memo.md", "carry", "claude",
        actor="t", session_id="s1", repo_root=tmp_path,
    )
    update_status("memo::ITEM_OP3D", "in_progress", actor="t", session_id="s1", repo_root=tmp_path)
    update_status(
        "memo::ITEM_OP3D", "completed", actor="t", session_id="s1", repo_root=tmp_path,
        actual_delta_s=-0.01,
        notes=(
            "[empirical:x.json] [baseline:eval_root/submissions/v4d_ms8/report.txt=0.8984335] [n600]"
        ),
    )
    note = append_note(
        "memo::ITEM_OP3D", "operator reviewed", actor="t", session_id="s1", repo_root=tmp_path
    )
    assert note.actual_delta_s == pytest.approx(-0.01)
