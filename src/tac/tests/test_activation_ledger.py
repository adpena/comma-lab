"""Tests for the lever ACTIVATION ledger (tac.witness_dsl.activation_ledger) — the "'off' is a tracked
queue" apparatus (#247 SENSE / CLAUDE.md orphaned-signal non-negotiable)."""
from __future__ import annotations

import pytest

from tac.witness_dsl import activation_ledger as al


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "lever_activation_ledger.jsonl"


# --- record / read ---------------------------------------------------------
def test_record_and_read_roundtrip(ledger):
    row = al.record_activation("EikonalViscosity", al.EVENT_FIRED, run_ref="run_x", path=ledger)
    assert row["lever"] == "EikonalViscosity" and row["event"] == "fired" and row["ts"]
    evs = al._read_events(ledger)
    assert len(evs) == 1 and evs[0]["run_ref"] == "run_x"


def test_invalid_event_raises(ledger):
    with pytest.raises(ValueError):
        al.record_activation("X", "bogus", path=ledger)


def test_empty_lever_raises(ledger):
    with pytest.raises(ValueError):
        al.record_activation("", al.EVENT_FIRED, path=ledger)


def test_read_missing_file_is_empty(tmp_path):
    assert al._read_events(tmp_path / "nope.jsonl") == []


def test_read_skips_corrupt_lines(ledger):
    al.record_activation("A", al.EVENT_FIRED, path=ledger)
    with open(ledger, "a") as f:
        f.write("{not json\n\n")
    al.record_activation("B", al.EVENT_FIRED, path=ledger)
    assert {e["lever"] for e in al._read_events(ledger)} == {"A", "B"}


# --- state machine ---------------------------------------------------------
def test_state_never_fired_for_unknown(ledger):
    st = al.activation_status("SegFocalGamma", ledger)
    assert st.state == al.STATE_NEVER_FIRED and not st.ever_fired and not st.ever_measured


def test_state_fired_unmeasured(ledger):
    al.record_activation("SegFocalGamma", al.EVENT_FIRED, path=ledger)
    st = al.activation_status("SegFocalGamma", ledger)
    assert st.state == al.STATE_FIRED_UNMEASURED and st.ever_fired and not st.ever_measured
    assert st.n_fired == 1


def test_state_measured(ledger):
    al.record_activation("SegFocalGamma", al.EVENT_FIRED, path=ledger)
    al.record_activation("SegFocalGamma", al.EVENT_MEASURED, verdict_ref="v1", path=ledger)
    st = al.activation_status("SegFocalGamma", ledger)
    assert st.state == al.STATE_MEASURED and st.ever_measured and st.n_measured == 1


def test_retired_is_terminal(ledger):
    al.record_activation("SegFocalGamma", al.EVENT_FIRED, path=ledger)
    al.record_activation("SegFocalGamma", al.EVENT_MEASURED, path=ledger)
    al.record_activation("SegFocalGamma", al.EVENT_RETIRED, reason="dominated", path=ledger)
    st = al.activation_status("SegFocalGamma", ledger)
    assert st.state == al.STATE_RETIRED and st.retired


# --- known_levers / never_fired / duty_to_measure --------------------------
def test_known_levers_are_the_dsl_factories():
    known = al.known_levers()
    assert "EikonalViscosity" in known and "SegFocalGamma" in known and "BoundaryDistance" in known
    assert known == tuple(sorted(known))  # sorted, deterministic


def test_empty_ledger_all_known_are_never_fired(ledger):
    known = al.known_levers()
    nf = al.never_fired(known, path=ledger)
    assert set(nf) == set(known)  # honest: nothing fired via the DSL path yet


def test_never_fired_drops_a_fired_lever(ledger):
    known = ("A", "B", "C")
    al.record_activation("B", al.EVENT_FIRED, path=ledger)
    assert set(al.never_fired(known, path=ledger)) == {"A", "C"}


def test_never_fired_excludes_retired(ledger):
    known = ("A", "B")
    al.record_activation("A", al.EVENT_RETIRED, reason="x", path=ledger)
    assert set(al.never_fired(known, path=ledger)) == {"B"}


def test_duty_to_measure_includes_fired_unmeasured(ledger):
    known = ("A", "B", "C")
    al.record_activation("A", al.EVENT_FIRED, path=ledger)                 # fired, unmeasured -> owed
    al.record_activation("B", al.EVENT_FIRED, path=ledger)
    al.record_activation("B", al.EVENT_MEASURED, path=ledger)             # measured -> not owed
    owed = set(al.duty_to_measure(known, path=ledger))
    assert owed == {"A", "C"} and "B" not in owed


# --- report ordering -------------------------------------------------------
def test_activation_report_surfaces_never_fired_first(ledger):
    known = ("A", "B", "C")
    al.record_activation("A", al.EVENT_FIRED, path=ledger)
    al.record_activation("A", al.EVENT_MEASURED, path=ledger)   # A measured
    al.record_activation("B", al.EVENT_FIRED, path=ledger)      # B fired-unmeasured
    # C never fired
    rows = al.activation_report(known, path=ledger)
    assert [r["lever"] for r in rows] == ["C", "B", "A"]  # never < fired-unmeasured < measured
    assert all(r["default"] == "off" for r in rows)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
