"""Tests for tac.harness_failure_ledger — the Weng weakness-mining failure ledger."""
from __future__ import annotations

import json

import pytest

from tac.harness_failure_ledger import (
    DEFAULT_LEDGER_PATH,
    FailureEvent,
    FailureLedgerError,
    append_failure_event,
    failure_states,
    load_failure_events,
    rank_open_failures,
    record_diagnosis,
    record_failure,
    record_recurrence,
    record_resolution,
    sense_rows,
)


@pytest.fixture()
def ledger(tmp_path):
    return tmp_path / "harness_failure_ledger.jsonl"


def test_append_and_load_roundtrip(ledger):
    record_failure("f1", surface="daemon", terminal_cause="died at ~5min", path=ledger)
    events = load_failure_events(ledger)
    assert len(events) == 1
    assert events[0].failure_id == "f1"
    assert events[0].surface == "daemon"
    assert events[0].resolution == "open"


def test_append_only_two_events_two_lines(ledger):
    record_failure("f1", surface="tool", terminal_cause="x", path=ledger)
    record_recurrence("f1", path=ledger)
    assert len(ledger.read_text().splitlines()) == 2


def test_opened_requires_valid_surface(ledger):
    with pytest.raises(FailureLedgerError):
        record_failure("f1", surface="kitchen-sink", terminal_cause="x", path=ledger)


def test_opened_requires_terminal_cause(ledger):
    with pytest.raises(FailureLedgerError):
        record_failure("f1", surface="gate", terminal_cause="  ", path=ledger)


def test_bad_causal_status_refused(ledger):
    with pytest.raises(FailureLedgerError):
        record_diagnosis("f1", diagnosis="d", causal_status="vibes", path=ledger)


def test_bad_resolution_refused(ledger):
    with pytest.raises(FailureLedgerError):
        record_resolution("f1", resolution="fixed-forever", path=ledger)


def test_diagnosis_history_preserves_falsified(ledger):
    """The daemon-saga property: four wrong theories STAY recorded after falsification."""
    record_failure("f1", surface="daemon", terminal_cause="silent death",
                   diagnosis="sandbox teardown", causal_status="hypothesized", path=ledger)
    record_diagnosis("f1", diagnosis="sandbox teardown", causal_status="falsified",
                     note="unsandboxed gen died too", path=ledger)
    record_diagnosis("f1", diagnosis="harness long-call sweep", causal_status="measured",
                     path=ledger)
    st = failure_states(ledger)["f1"]
    assert len(st.diagnosis_history) == 3
    assert st.diagnosis_history[1]["causal_status"] == "falsified"
    assert st.current_causal_status == "measured"


def test_current_causal_status_all_falsified(ledger):
    record_failure("f1", surface="daemon", terminal_cause="x",
                   diagnosis="theory A", causal_status="hypothesized", path=ledger)
    record_diagnosis("f1", diagnosis="theory A", causal_status="falsified", path=ledger)
    st = failure_states(ledger)["f1"]
    # the falsified event kills the earlier 'hypothesized' row of the SAME theory text —
    # a dead theory must not resurrect through its earlier hypothesized entry.
    assert st.current_causal_status == "falsified"


def test_recurrence_count_increments(ledger):
    record_failure("f1", surface="subagent", terminal_cause="x", path=ledger)
    record_recurrence("f1", path=ledger)
    record_recurrence("f1", path=ledger)
    assert failure_states(ledger)["f1"].recurrence_count == 3


def test_resolution_lifecycle(ledger):
    record_failure("f1", surface="gate", terminal_cause="x", path=ledger)
    record_resolution("f1", resolution="class-fixed", note="gate + fix landed", path=ledger)
    assert failure_states(ledger)["f1"].resolution == "class-fixed"


def test_rank_open_failures_ordering(ledger):
    record_failure("resolved", surface="tool", terminal_cause="x", path=ledger)
    record_resolution("resolved", resolution="gate-landed", path=ledger)
    record_failure("once", surface="tool", terminal_cause="x", path=ledger)
    record_failure("thrice", surface="daemon", terminal_cause="x", path=ledger)
    record_recurrence("thrice", path=ledger)
    record_recurrence("thrice", path=ledger)
    ranked = [s.failure_id for s in rank_open_failures(ledger)]
    assert ranked == ["thrice", "once", "resolved"]


def test_lenient_loader_skips_malformed(ledger):
    record_failure("f1", surface="tool", terminal_cause="x", path=ledger)
    with ledger.open("a") as fh:
        fh.write("not json\n")
        fh.write(json.dumps({"schema": "other.v9", "failure_id": "f2"}) + "\n")
    assert len(load_failure_events(ledger)) == 1


def test_sense_rows_shape_and_limit(ledger):
    for i in range(4):
        record_failure(f"f{i}", surface="tool", terminal_cause="x", path=ledger)
    rows = sense_rows(ledger, limit=2)
    assert len(rows) == 2
    for key in ("failure_id", "surface", "terminal_cause", "recurrence_count",
                "causal_status", "resolution", "diagnosis_history"):
        assert key in rows[0]


def test_sense_rows_missing_ledger_empty(tmp_path):
    assert sense_rows(tmp_path / "nope.jsonl") == []


def test_default_path_is_omx_state():
    assert DEFAULT_LEDGER_PATH.name == "harness_failure_ledger.jsonl"
    assert DEFAULT_LEDGER_PATH.parent.name == "state"
    assert DEFAULT_LEDGER_PATH.parent.parent.name == ".omx"


def test_validate_event_kind(ledger):
    with pytest.raises(FailureLedgerError):
        append_failure_event(
            FailureEvent(failure_id="f1", event="exploded", ts="2026-07-07T00:00:00Z"),
            path=ledger,
        )
