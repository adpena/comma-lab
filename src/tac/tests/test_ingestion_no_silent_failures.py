# SPDX-License-Identifier: MIT
"""Ingestion boundaries must be TOTAL *and* LOUD (operator binding 2026-08-01:
"No silent failures anywhere ever").

MEASURED INCIDENT. ``event_notes`` was a list in 1 of 395 rows of
``.omx/state/canonical_task_status.jsonl`` (task 793, written 2026-07-31T09:20:59Z).
Two distinct failures rode that one row:

  1. ``graph_memory.build._clean`` raised ``TypeError`` inside ``re.sub`` -> the WHOLE
     graph build died -> every ``recall_fused``/graph query returned nothing -> the
     reader fell back to hand-grepping for a full session. 100% of recall lost to
     0.25% bad data.
  2. ``canonical_task_status.contract.from_json_obj`` did ``str(obj.get(...))``, turning
     the list into the literal ``"['FINAL race verdict...']"`` -- brackets and quotes --
     and NEVER raised. That is worse than the crash: a crash announces itself.

The cure is deliberately ASYMMETRIC, matching the append-only ledger discipline
(Catalog #110/#113):
  * WRITE side fails CLOSED -- no new malformed row can be born.
  * READ side stays TOTAL but LOUD -- historical rows remain readable, never silently
    mangled, and every coercion is recorded so the row gets FIXED.

These tests assert BEHAVIOUR at both boundaries. A fix that restored availability by
swallowing the anomaly would pass a naive "does recall work" test and still be wrong;
that is precisely what the loudness assertions below forbid.
"""

from __future__ import annotations

import warnings

import pytest

from tac.canonical_task_status.contract import (
    SCHEMA_VERSION,
    CanonicalTaskStatusRow,
    _coerce_event_notes,
)
from tac.graph_memory.build import _clean, ingestion_anomalies

# --------------------------------------------------------------------------------
# graph_memory._clean -- TOTAL (recall survives) *and* LOUD (the row is surfaced)
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("plain string", "plain string"),
        ("  collapsed \n  whitespace ", "collapsed whitespace"),
        (None, ""),
        (["a", "b"], "a b"),
        (("x", "y"), "x y"),
        ([], ""),
        (42, "42"),
        (3.5, "3.5"),
        ({"k": "v"}, "{'k': 'v'}"),
    ],
)
def test_clean_is_total_over_every_json_type(value, expected):
    """No JSON-derived value may raise. The graph IS the recall surface."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _clean(value, 400) == expected


def test_clean_reproduces_the_exact_measured_incident():
    """The literal shape that killed recall: a 1-element list of a verdict string."""
    row = ["FINAL race verdict [macOS-CPU advisory]: hull_moved_s=FALSE everywhere."]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _clean(row, 400) == row[0]  # not "['FINAL race...']"


def test_clean_still_caps_and_normalises_after_coercion():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _clean(["x" * 50, "y" * 50], 20) == "x" * 20


def test_clean_WARNS_on_coercion_and_records_it():
    """LOUDNESS. A coercion that repairs availability by hiding the schema violation
    just converts a crash into permanent invisible corpus debt."""
    before = len(ingestion_anomalies())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _clean(["bad"], 400, field="event_notes", source="ledger.jsonl")
    assert len(caught) == 1
    msg = str(caught[0].message)
    assert "event_notes" in msg and "ledger.jsonl" in msg and "list" in msg
    recorded = ingestion_anomalies()
    assert len(recorded) == before + 1
    assert recorded[-1]["field"] == "event_notes"
    assert recorded[-1]["type"] == "list"


def test_clean_does_NOT_warn_on_well_formed_strings():
    """The alarm must stay meaningful: no warning on the 99.75% healthy path."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(50):
            _clean("a normal summary", 400, field="title")
    assert caught == []


# --------------------------------------------------------------------------------
# canonical_task_status -- WRITE fails closed, READ is loud-but-total
# --------------------------------------------------------------------------------

_BASE_ROW = dict(
    schema_version=SCHEMA_VERSION,
    task_id="t1",
    source_design_memo="memo.md",
    title="title",
    status="pending",
    owner="claude",
    predicted_cost_usd=None,
    predicted_delta_s_band=None,
    actual_delta_s=None,
    commit_shas=(),
    test_status="pending",
    blockers=(),
    started_at_utc=None,
    completed_at_utc=None,
    event_type="registered",
    event_timestamp_utc="2026-08-01T00:00:00Z",
    event_actor="claude",
    event_notes="ok",
    session_id="s",
    written_at_utc="2026-08-01T00:00:00Z",
    written_pid=1,
    written_host="h",
)


@pytest.mark.parametrize("field", ["task_id", "title", "status", "owner", "event_notes"])
def test_write_side_REFUSES_non_string_free_text(field):
    """FAIL CLOSED. The dataclass annotation is a hint; this is the enforcement."""
    # `status` must also be a VALID status, so use a list to trip the type check first.
    with pytest.raises(TypeError, match=f"{field} must be str"):
        CanonicalTaskStatusRow(**{**_BASE_ROW, field: ["a list"]})


def test_write_side_accepts_well_formed_row():
    row = CanonicalTaskStatusRow(**_BASE_ROW)
    assert row.event_notes == "ok"


def test_read_side_joins_SANELY_not_via_str_of_list():
    """The old ``str(value)`` produced "['a', 'b']". Joining is the honest reading."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _coerce_event_notes(["a", "b"]) == "a; b"
        assert "[" not in _coerce_event_notes(["a"])


def test_read_side_WARNS_so_the_source_row_gets_corrected():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _coerce_event_notes(["x"])
    assert len(caught) == 1
    assert "append" in str(caught[0].message)  # names the append-only remedy


def test_read_side_is_silent_and_identity_on_strings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _coerce_event_notes("already fine") == "already fine"
        assert _coerce_event_notes(None) == ""
    assert caught == []


def test_read_side_stays_TOTAL_so_historical_rows_remain_loadable():
    """Append-only means the bad row cannot be rewritten; the reader must not die on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = CanonicalTaskStatusRow.from_json_obj(
            {**_BASE_ROW, "event_notes": ["historical", "list"]}
        )
    assert row.event_notes == "historical; list"


def test_the_live_ledger_loads_and_any_anomaly_is_visible():
    """Regression on the REAL file: it must parse, and if a malformed row is still
    present the parse must WARN rather than pass silently."""
    from pathlib import Path

    ledger = Path(__file__).resolve().parents[3] / ".omx" / "state" / "canonical_task_status.jsonl"
    if not ledger.exists():
        pytest.skip("no canonical task-status ledger in this checkout")
    import json

    malformed = 0
    total_rows = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total_rows += 1
            obj = json.loads(line)
            if not isinstance(obj.get("event_notes", ""), str):
                malformed += 1
            CanonicalTaskStatusRow.from_json_obj(obj)  # must never raise
    # Count BY CLASS. The original `len(caught) == malformed` was broader than its own
    # intent: it asserted that the event_notes coercion is the ONLY thing the reader can
    # ever warn about, so any second independent warning class would fail it. ddm_op3
    # added one (ΔS custody). The invariant that actually matters -- one warning per
    # malformed row, and never a silent pass -- is preserved by filtering.
    coercion = [w for w in caught if "not str" in str(w.message)]
    custody = [w for w in caught if "without full custody" in str(w.message)]
    assert len(coercion) == malformed, "every malformed row must produce exactly one warning"
    assert len(coercion) + len(custody) == len(caught), (
        f"unclassified reader warning(s) over {total_rows} rows: "
        f"{[str(w.message)[:120] for w in caught if w not in coercion and w not in custody]}"
    )
    # Scope denominator, so a future run that examines ZERO rows cannot read as a pass.
    assert total_rows > 0, "empty ledger scope is VACUOUS, never a pass"
