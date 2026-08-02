# SPDX-License-Identifier: MIT
"""BEHAVIOUR tests for the rivalry field-set on race receipts (ddm_wt1, task #868).

ddm_gd5 measured that the "measured-better successor" relation has no representation in code, so
no static analyser can find it. This module is the recording side of that cure. Every test below
asserts a PROPERTY of the join or of a refusal on real inputs; none asserts that a constant equals
itself. The controls are stated explicitly: a challenger that WINS and is unadopted must surface
(positive), and one that LOSES must not (negative), because a join that fires on everything is
worth exactly as much as one that fires on nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.race_receipt import (
    RivalryRow,
    read_rivalry_rows,
    rivalry_rows_from_arms,
    unadopted_better_challengers,
)

_REPO = Path(__file__).resolve().parents[3]
_GD1_RECEIPT = _REPO / ".omx/research/ddm_gd1_hilbert_order_race_receipt_20260731.json"


def _row(**over) -> RivalryRow:
    base = {
        "role": "token_scan_order",
        "incumbent": "raster",
        "challenger": "hilbert",
        "metric": "lossless coded bytes",
        "axis": "[macOS-CPU advisory]",
        "unit": "B",
        "incumbent_value": 100.0,
        "challenger_value": 90.0,
        "adopted": False,
        "not_adopted_reason": "queued behind the burn",
    }
    base.update(over)
    return RivalryRow(**base)


# ── the join: both controls ──────────────────────────────────────────────────────────
def test_join_fires_on_an_unadopted_winner() -> None:
    """POSITIVE control: better + unadopted is exactly the harm the #864 P0 names."""
    rows = [_row()]
    hits = unadopted_better_challengers(rows)
    assert [r.challenger for r in hits] == ["hilbert"]
    assert hits[0].delta == -10.0


def test_join_is_silent_on_a_loser_and_on_an_adopted_winner() -> None:
    """NEGATIVE controls. A loser carries no harm; an adopted winner is already wired."""
    loser = _row(challenger="serpentine", challenger_value=110.0)
    adopted = _row(challenger="gilbert", challenger_value=80.0, adopted=True,
                   not_adopted_reason="")
    assert unadopted_better_challengers([loser, adopted]) == []


def test_join_respects_direction_instead_of_guessing_it() -> None:
    """"Better" is not derivable from a number: bytes fall, throughput rises.

    The same pair of values must produce OPPOSITE verdicts under opposite direction flags — a gate
    that guessed would invert half of them silently.
    """
    falling = _row(incumbent_value=100.0, challenger_value=90.0, lower_is_better=True)
    rising = _row(incumbent_value=100.0, challenger_value=90.0, lower_is_better=False)
    assert falling.challenger_wins
    assert not rising.challenger_wins
    assert len(unadopted_better_challengers([falling, rising])) == 1


def test_a_tie_is_not_a_win() -> None:
    """Parity must not be reported as a superseding successor — the exact error ddm_pw1 corrected
    on the pose Jacobian pair (RACED-PARITY read as RACED-SUPERSEDED)."""
    assert not _row(challenger_value=100.0).challenger_wins
    assert unadopted_better_challengers([_row(challenger_value=100.0)]) == []


def test_join_ranks_by_measured_margin_not_by_declaration_order() -> None:
    """The queue's whole advantage over an orphan registry is that harm is QUANTIFIED."""
    small = _row(challenger="a", challenger_value=99.0)
    large = _row(challenger="b", challenger_value=10.0)
    assert [r.challenger for r in unadopted_better_challengers([small, large])] == ["b", "a"]


# ── refusals ─────────────────────────────────────────────────────────────────────────
def test_unadopted_row_refuses_a_placeholder_reason() -> None:
    """An unadopted winner with no reason IS the silently-orphaned successor."""
    for bad in ("", "   ", "TBD", "<reason>", "n/a"):
        with pytest.raises(ValueError, match="not_adopted_reason"):
            _row(not_adopted_reason=bad)


def test_adopted_row_needs_no_reason() -> None:
    """The refusal is scoped: adoption is self-explanatory."""
    assert _row(adopted=True, not_adopted_reason="").adopted


def test_row_refuses_self_comparison_and_empty_identifiers() -> None:
    with pytest.raises(ValueError, match="same identifier"):
        _row(challenger="raster")
    for empty in ("role", "incumbent", "challenger", "metric", "axis", "unit"):
        with pytest.raises(ValueError, match=empty):
            _row(**{empty: "  "})


def test_row_refuses_non_numeric_values() -> None:
    for bad in ("100", None, True):
        with pytest.raises(ValueError, match="real number"):
            _row(challenger_value=bad)


# ── derivation + round-trip ──────────────────────────────────────────────────────────
def test_delta_is_derived_not_stored_and_reader_recomputes_it() -> None:
    """A stored derived value that disagrees with its inputs is a stale-artifact confound."""
    payload = _row().to_dict()
    payload["delta"] = 12345.0           # a hand-edited (or stale) receipt
    payload["challenger_wins"] = False
    rebuilt = RivalryRow.from_dict(payload)
    assert rebuilt.delta == -10.0
    assert rebuilt.challenger_wins


def test_arms_builder_refuses_an_incumbent_that_was_not_raced() -> None:
    """Without the incumbent in the race there is no shared control, so no comparison."""
    with pytest.raises(ValueError, match="not among the raced arms"):
        rivalry_rows_from_arms(
            role="r", incumbent_arm="absent", arms={"a": {"bytes": 1}}, value_key="bytes",
            metric="m", axis="x", unit="B", not_adopted_reason="lost")


def test_arms_builder_emits_one_row_per_challenger_and_marks_the_adopted_one() -> None:
    rows = rivalry_rows_from_arms(
        role="token_scan_order", incumbent_arm="base",
        arms={"base": {"bytes": 100}, "win": {"bytes": 90}, "lose": {"bytes": 110}},
        value_key="bytes", metric="bytes", axis="[advisory]", unit="B",
        not_adopted_reason="lost the race", adopted_arm="win")
    assert {r.challenger for r in rows} == {"win", "lose"}
    assert all(r.incumbent == "base" for r in rows)
    assert next(r for r in rows if r.challenger == "win").adopted
    assert unadopted_better_challengers(rows) == []  # the winner WAS adopted


# ── the wire-in, on the real landed receipt ──────────────────────────────────────────
def test_reader_tolerates_receipts_that_predate_the_schema(tmp_path) -> None:
    """Most receipts have no rivalry block; treating that as an error makes the join unrunnable."""
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"schema": "something.v1", "token_arms": {}}), encoding="utf-8")
    assert read_rivalry_rows(p) == []
    assert read_rivalry_rows(tmp_path / "missing.json") == []
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="unreadable"):
        assert read_rivalry_rows(tmp_path / "bad.json") == []


def test_one_malformed_row_does_not_take_down_the_whole_join(tmp_path) -> None:
    """The named recurrence: one row of 395 violating its field contract raised inside the
    graph-memory reader and killed ALL fused-recall queries campaign-wide — 100% of recall lost
    to 0.25% bad data. A join that dies on one bad row is a join nobody can rely on.

    Total-but-LOUD: the good rows survive AND the skip warns. Silent skipping would be the
    opposite error — an all-malformed receipt would read as "no rivalry", indistinguishable from
    "no race was ever run".
    """
    good = _row().to_dict()
    bad = _row(challenger="other").to_dict()
    del bad["unit"]
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps({"rivalry": [good, bad]}), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="malformed"):
        rows = read_rivalry_rows(p)
    assert [r.challenger for r in rows] == ["hilbert"], "the good row must survive"
    assert unadopted_better_challengers(rows), "the join must still run"


@pytest.mark.skipif(not _GD1_RECEIPT.exists(), reason="gd1 race receipt not present")
def test_gd1_receipt_carries_rivalry_rows_and_the_join_runs_on_them() -> None:
    """THE PILOT WIRE-IN, on the real landed receipt rather than a fixture.

    gd1 raced scan orders and coders against the live raster/SMEVR path. The incumbent won every
    arm, so the honest expectation is a NON-EMPTY rivalry block and an EMPTY join — rivalry was
    recorded, and no unadopted successor beats what is live. Both halves are asserted: a receipt
    with no rows would make the join vacuous, which is the failure this test exists to catch.
    """
    rows = read_rivalry_rows(_GD1_RECEIPT)
    assert rows, "the pilot receipt must carry rivalry rows or the wire-in is not landed"
    assert {r.role for r in rows} >= {"token_scan_order", "token_entropy_coder"}
    assert all(r.incumbent != r.challenger for r in rows)
    assert all(r.not_adopted_reason for r in rows if not r.adopted)
    hits = unadopted_better_challengers(rows)
    assert hits == [], (
        "the live incumbent won every gd1 arm; a hit here means the receipt or the direction "
        f"flag drifted: {[(h.role, h.challenger, h.delta) for h in hits]}")


@pytest.mark.skipif(not _GD1_RECEIPT.exists(), reason="gd1 race receipt not present")
def test_gd1_rivalry_rows_never_vary_two_factors_at_once() -> None:
    """A scan-order row compared against a coder swap answers neither question.

    The two-factor arms are excluded BY NAME in the receipt so the exclusion is auditable rather
    than implicit in a filter.
    """
    payload = json.loads(_GD1_RECEIPT.read_text(encoding="utf-8"))
    excluded = payload.get("rivalry_excluded_arms", {})
    assert set(excluded) == {"b1_brotli11_hilbert", "c1_lzma1_hilbert"}
    assert all(v.strip() for v in excluded.values()), "each exclusion must carry its reason"
    challengers = {r.challenger for r in read_rivalry_rows(_GD1_RECEIPT)}
    assert not (challengers & set(excluded))
