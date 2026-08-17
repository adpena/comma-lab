# SPDX-License-Identifier: MIT
"""Controls for PER-ROW isolation in the codex→claude inbox.

Second site of the 2026-08-17 blast-radius class (the first was the canonical
task-status ledger).  ``_validate_row`` polices FIELD RELATIONSHIPS -- unknown
``event_type``, bad ``status``, unparseable timestamp -- and the old loader
caught those in the same handler as ``json.JSONDecodeError``, then
``shutil.move``d the whole shared inbox.  One arm's malformed row deleted state
for the whole fleet.

Blast radius MEASURED, with its level stated (an earlier note said "7 read
consumers" without saying 7 of WHAT): **7 in-module read paths** reach the
spine (``load_inbox`` :423 plus :467/:476/:483/:489/:507/:768), and exactly
**1 external production call site** (``preflight.py``:30607).  Two of the seven
are reached from WRITE paths (:607, :710), so an append destroyed the file too.

The population was MEASURED before this fix, not assumed: of four modules that
quarantine a ledger, three (``lattice_state_ledger``, ``recursive_adversarial_review``,
``master_gradient``) raise only on unparseable bytes or a non-dict root and are
CLEAN.  This module was the only one that conflated the classes.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from tac.codex_to_claude_inbox import (
    INBOX_SCHEMA_VERSION,
    InboxRowCorruptError,
    append_inbox_answer,
    load_inbox_strict,
    unreadable_inbox_rows,
)


def _good_row(event_id: str) -> dict:
    return {
        "schema_version": INBOX_SCHEMA_VERSION,
        "event_id": event_id,
        "event_type": "question",
        "status": "open",
        "agent": "codex",
        "subagent_id": None,
        "session_id": "s1",
        "written_at_utc": "2026-08-17T00:00:00Z",
        "written_pid": 1,
        "written_host": "test",
        "question_text": "is the ledger readable?",
    }


def _write(path: Path, *rows: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_row_contract_violation_does_not_move_the_inbox(tmp_path: Path) -> None:
    """THE INCIDENT: one bad row must not delete the fleet's shared channel."""
    inbox = tmp_path / "codex_to_claude_inbox.jsonl"
    bad = _good_row("evt::BAD")
    bad["status"] = "not_a_real_status"  # a field RELATIONSHIP, not bad bytes
    _write(inbox, _good_row("evt::GOOD"), bad)
    before = inbox.read_bytes()

    with pytest.warns(UserWarning, match="UNREADABLE"):
        rows = load_inbox_strict(inbox)

    assert inbox.exists(), "the shared inbox was MOVED by one bad row"
    assert inbox.read_bytes() == before, "the inbox was rewritten, not just read"
    assert list(inbox.parent.glob("*.corrupt.*")) == [], "quarantined a contract violation"
    assert [r["event_id"] for r in rows] == ["evt::GOOD"]

    broken = unreadable_inbox_rows(inbox)
    assert set(broken) == {"evt::BAD"}
    assert "row contract violated" in broken["evt::BAD"]
    assert "status" in broken["evt::BAD"], "the reason must name the failing field"


def test_schema_version_bump_does_not_destroy_the_inbox(tmp_path: Path) -> None:
    """The sharpest live case: a routine migration must not read as data loss.

    ``_validate_row`` demands an EXACT ``schema_version``, so bumping it fails
    EVERY existing row at once.  Under the old handler that moved the whole
    file on the first read.
    """
    inbox = tmp_path / "codex_to_claude_inbox.jsonl"
    old_a, old_b = _good_row("evt::OLD_A"), _good_row("evt::OLD_B")
    old_a["schema_version"] = old_b["schema_version"] = "codex_to_claude_inbox_v0_ANCIENT"
    _write(inbox, old_a, old_b)

    with pytest.warns(UserWarning, match="UNREADABLE"):
        rows = load_inbox_strict(inbox)
    assert rows == []  # nothing servable...
    assert inbox.exists(), "a schema bump MOVED the inbox"  # ...but nothing destroyed
    assert set(unreadable_inbox_rows(inbox)) == {"evt::OLD_A", "evt::OLD_B"}


def test_file_corruption_still_quarantines(tmp_path: Path) -> None:
    """The path I must NOT have broken: unparseable bytes are a different class."""
    inbox = tmp_path / "codex_to_claude_inbox.jsonl"
    _write(inbox, _good_row("evt::GOOD"), "{not json at all")

    with pytest.raises(InboxRowCorruptError):
        load_inbox_strict(inbox)
    assert not inbox.exists(), "file corruption must still quarantine"
    assert len(list(inbox.parent.glob("*.corrupt.*"))) == 1


def test_non_object_row_is_file_corruption_not_a_contract_violation(tmp_path: Path) -> None:
    """A bare JSON scalar parses, but there is no row to hold to a contract."""
    inbox = tmp_path / "codex_to_claude_inbox.jsonl"
    _write(inbox, "42")
    with pytest.raises(InboxRowCorruptError):
        unreadable_inbox_rows(inbox)
    assert inbox.exists(), "the read-only audit must not quarantine"


def test_clean_inbox_is_silent(tmp_path: Path) -> None:
    inbox = tmp_path / "codex_to_claude_inbox.jsonl"
    _write(inbox, _good_row("evt::A"), _good_row("evt::B"))
    assert unreadable_inbox_rows(inbox) == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a clean inbox must NOT warn
        assert len(load_inbox_strict(inbox)) == 2


def test_append_survives_a_stale_contract_violating_row(tmp_path: Path) -> None:
    """WRITES read the same spine -- so the old handler destroyed on APPEND too.

    ``append_inbox_answer`` -> ``_require_open_question`` ->
    ``latest_status_by_event_id`` -> ``load_inbox_strict``.  Measured, not
    assumed: 7 in-module read paths reach that spine, two of them
    (:607, :710) from write paths.  So an arm merely FILING a question while a
    stale bad row sat in the log used to move the fleet's shared channel.
    """
    inbox = tmp_path / "codex_to_claude_inbox.jsonl"
    bad = _good_row("evt::STALE_BAD")
    bad["status"] = "not_a_real_status"
    _write(inbox, _good_row("evt::OPEN_Q"), bad)

    # append_inbox_answer -> _require_open_question -> latest_status_by_event_id
    with pytest.warns(UserWarning, match="UNREADABLE"):
        append_inbox_answer(
            response_to_event_id="evt::OPEN_Q",
            answer_text="yes -- the bad row is excluded, not fatal",
            session_id="s1",
            path=inbox,
        )

    assert inbox.exists(), "an APPEND moved the inbox"
    assert list(inbox.parent.glob("*.corrupt.*")) == []
    with pytest.warns(UserWarning, match="UNREADABLE"):
        rows = load_inbox_strict(inbox)
    assert [r["event_type"] for r in rows] == ["question", "answer"]
    assert set(unreadable_inbox_rows(inbox)) == {"evt::STALE_BAD"}


def test_missing_inbox_is_valid_and_empty(tmp_path: Path) -> None:
    inbox = tmp_path / "codex_to_claude_inbox.jsonl"
    assert load_inbox_strict(inbox) == []
    assert unreadable_inbox_rows(inbox) == {}
