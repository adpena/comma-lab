"""Tests for tools/costate_digest.py failure-ledger reader — schema-tolerance.

The harness failure ledger accumulated rows across several writer generations:
  - canonical ``harness_failure.v1``: ``failure_id`` + ``event`` (opened/diagnosis/resolution)
  - legacy shapes: a bare ``class``/``failure_class`` key + terminal markers like
    ``event='self_protected'``, ``status='resolved'``, or a populated ``resolution`` field.

The prior reader keyed strictly on ``failure_id`` and recognised only ``event=='resolution'``,
so every legacy row collapsed into one phantom ``'?'`` class (unresolvable, since a None key
never resolves) AND hid genuinely-open legacy items behind that phantom (the digest showed
``open: ?`` while a real "cron the watchdog" item was open). These tests pin the schema-tolerant
reader so no writer generation re-creates the phantom, and — critically — so ``event`` stays
AUTHORITATIVE for v1 rows (an ``event='opened'`` row carrying a planned-fix ``resolution`` field
must NOT be marked resolved).

Read-only + score-neutral: the reader never mutates the ledger.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import costate_digest as cd  # noqa: E402


# --- _ledger_class_key: fallback chain across writer generations ------------------------------

def test_class_key_prefers_failure_id():
    assert cd._ledger_class_key({"failure_id": "A", "failure_class": "B", "class": "C"}) == "A"


def test_class_key_falls_back_to_failure_class_then_class():
    assert cd._ledger_class_key({"failure_class": "B", "class": "C"}) == "B"
    assert cd._ledger_class_key({"class": "C"}) == "C"
    assert cd._ledger_class_key({"bug_class": "D"}) == "D"


def test_class_key_truly_keyless_row_is_question_mark():
    # Only a row with NO class identifier at all falls to '?'.
    assert cd._ledger_class_key({"note": "orphan", "ts": "x"}) == "?"


# --- _ledger_row_is_resolution: terminal detection across generations -------------------------

def test_v1_resolution_event_is_terminal():
    assert cd._ledger_row_is_resolution({"event": "resolution", "failure_id": "X"}) is True


def test_legacy_self_protected_event_is_terminal():
    assert cd._ledger_row_is_resolution({"event": "self_protected", "class": "X"}) is True


def test_legacy_status_resolved_is_terminal_when_no_event():
    assert cd._ledger_row_is_resolution({"failure_class": "X", "status": "resolved"}) is True


def test_legacy_bare_resolution_field_is_terminal_when_no_event():
    assert cd._ledger_row_is_resolution({"class": "X", "resolution": "fixed it in abc123"}) is True


def test_event_is_authoritative_opened_with_resolution_field_is_NOT_terminal():
    # THE regression: a v1 lifecycle row using event='opened' may carry a resolution field
    # describing the PLANNED fix. event must win — this row is still OPEN.
    row = {"schema": "harness_failure.v1", "event": "opened", "failure_id": "X",
           "resolution": "planned: add a retry loop"}
    assert cd._ledger_row_is_resolution(row) is False


def test_diagnosis_event_is_not_terminal():
    assert cd._ledger_row_is_resolution({"event": "diagnosis", "failure_id": "X"}) is False


def test_open_legacy_row_with_only_owed_fix_is_not_terminal():
    row = {"failure_class": "phantom", "owed_fix": "cron the watchdog", "ts_utc": "2026-07-16"}
    assert cd._ledger_row_is_resolution(row) is False


# --- _summarize_failure_ledger: end-to-end over a mixed-schema ledger --------------------------

def _mixed_ledger() -> list[dict]:
    """Encodes the exact shapes observed in the real ledger (no live-data coupling)."""
    return [
        # v1 class, opened -> ... -> resolution (latest terminal): RESOLVED
        {"schema": "harness_failure.v1", "failure_id": "cls_v1", "event": "opened",
         "resolution": "planned fix text"},
        {"schema": "harness_failure.v1", "failure_id": "cls_v1", "event": "diagnosis"},
        {"schema": "harness_failure.v1", "failure_id": "cls_v1", "event": "resolution"},
        # legacy self_protected: RESOLVED, keyed by bare 'class'
        {"class": "cls_self_protected", "event": "self_protected",
         "resolution": "auto-retry landed"},
        # legacy open->resolved pair keyed by 'failure_class'
        {"failure_class": "cls_pair", "event": None, "note": "opened"},
        {"failure_class": "cls_pair", "status": "resolved", "resolution": "permanent fix landed"},
        # legacy OPEN item (owed fix, no terminal marker): the one genuinely-unresolved class
        {"failure_class": "cls_open", "owed_fix": "cron the watchdog"},
    ]


def test_summary_no_phantom_question_mark_class():
    summary = cd._summarize_failure_ledger(_mixed_ledger())
    # Four distinct real classes, ZERO phantom '?' bucket.
    assert summary["classes"] == 4
    assert "?" not in summary["unresolved"]


def test_summary_unresolved_is_exactly_the_open_class():
    summary = cd._summarize_failure_ledger(_mixed_ledger())
    assert summary["unresolved"] == ["cls_open"]


def test_summary_recurrent_counts_nonresolution_rows():
    # cls_v1 has 2 non-resolution rows (opened, diagnosis) -> recurrent.
    summary = cd._summarize_failure_ledger(_mixed_ledger())
    assert "cls_v1" in summary["recurrent"]


def test_behavior_not_constants_legacy_open_surfaces_by_name_not_phantom():
    # BEHAVIOR guard: the OLD strict reader (failure_id-only + event=='resolution') would have
    # keyed all four legacy rows under '?' and reported open:'?'. This asserts the real class
    # NAME surfaces — it would FAIL if the schema-tolerant body were reverted to strict logic.
    summary = cd._summarize_failure_ledger(_mixed_ledger())
    assert summary["unresolved"] == ["cls_open"]
    assert summary["classes"] == 4  # not 1-phantom + N-v1


def test_empty_ledger_is_zero_not_crash():
    summary = cd._summarize_failure_ledger([])
    assert summary == {"classes": 0, "unresolved": [], "recurrent": []}


def test_non_dict_rows_are_ignored():
    summary = cd._summarize_failure_ledger([{"failure_id": "x", "event": "resolution"}, "junk", 42, None])
    assert summary["classes"] == 1
