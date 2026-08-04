"""magnitude_kind + the F2 consumption gate (cg1 / xa2 F2) on fl2's force ledger.

fl2's `magnitude_kind` shipped with zero test coverage; this file closes that gap
and covers the `oracle_assisted` third kind plus the fail-closed consumption API.

The bug class under test is `bz1`'s MIRAGE LAW at the table layer: a consumer that
reads `magnitude_s` without reading `magnitude_kind` banks a description ceiling
or an oracle number as if it were a realized-through-R price.
"""

from __future__ import annotations

import pytest

from tac.witness_control.force_class_edge_ledger import (
    BANKABLE_MAGNITUDE_KINDS,
    LEDGER,
    MAGNITUDE_KINDS,
    ORACLE_ASSISTED_KINDS,
    ForceLedgerRow,
    MagnitudeKindError,
    composed_candidate_total,
    consume_magnitude,
    magnitude_kind_census,
)


def _row(**over) -> ForceLedgerRow:
    base = {
        "row_id": "test.row",
        "scope_kind": "CLASS",
        "scope": "Lane",
        "force": "test_force",
        "force_kind": "OBJECTIVE",
        "verb": "ERODE",
        "verdict": "IMPROVES",
        "measured": True,
        "evidence": "x",
        "evidence_source": "y",
        "verdict_scope": "INSTANCE",
        "protection": "N_A",
    }
    base.update(over)
    return ForceLedgerRow(**base)


# ---- the schema ------------------------------------------------------------


def test_oracle_assisted_is_a_first_class_kind():
    assert "oracle_assisted" in MAGNITUDE_KINDS


def test_oracle_assisted_is_neither_a_description_gap_nor_bankable():
    """It is a real measurement made with an instrument the decoder lacks --
    so demoting it to description_gap understates it and promoting it to
    realized_through_R banks it. It needs its own kind."""
    assert "oracle_assisted" not in BANKABLE_MAGNITUDE_KINDS
    assert frozenset({"oracle_assisted"}) == ORACLE_ASSISTED_KINDS
    assert "description_gap" not in ORACLE_ASSISTED_KINDS


def test_only_realized_through_R_is_bankable_by_default():
    assert frozenset({"realized_through_R"}) == BANKABLE_MAGNITUDE_KINDS


def test_bad_magnitude_kind_is_refused_at_construction():
    with pytest.raises(ValueError, match="bad magnitude_kind"):
        _row(magnitude_s=0.1, magnitude_kind="vibes")


def test_magnitude_without_a_kind_is_refused_at_construction():
    with pytest.raises(ValueError, match="magnitude_kind is N_A"):
        _row(magnitude_s=0.1)


def test_oracle_assisted_row_constructs():
    r = _row(magnitude_s=0.5, magnitude_kind="oracle_assisted")
    assert r.magnitude_kind == "oracle_assisted"


# ---- the consumption gate --------------------------------------------------


def test_realized_through_R_consumes_by_default():
    r = _row(magnitude_s=0.25, magnitude_kind="realized_through_R")
    assert consume_magnitude(r) == 0.25


def test_description_gap_is_refused_by_default():
    r = _row(magnitude_s=0.9, magnitude_kind="description_gap")
    with pytest.raises(MagnitudeKindError, match="not in the accepted set"):
        consume_magnitude(r)


def test_oracle_assisted_is_refused_by_default():
    r = _row(magnitude_s=0.9, magnitude_kind="oracle_assisted")
    with pytest.raises(MagnitudeKindError, match="oracle_assisted"):
        consume_magnitude(r)


def test_widening_accept_kinds_is_explicit_and_works():
    r = _row(magnitude_s=0.9, magnitude_kind="oracle_assisted")
    assert consume_magnitude(r, accept_kinds={"oracle_assisted"}) == 0.9


def test_missing_magnitude_is_refused_not_treated_as_zero():
    """Silently returning 0.0 would make an absent price look like a measured no-op."""
    with pytest.raises(MagnitudeKindError, match="no magnitude_s"):
        consume_magnitude(_row())


def test_typo_in_accept_kinds_is_refused_rather_than_silently_matching_nothing():
    r = _row(magnitude_s=0.25, magnitude_kind="realized_through_R")
    with pytest.raises(MagnitudeKindError, match="unknown magnitude kinds"):
        consume_magnitude(r, accept_kinds={"realised_through_R"})  # British spelling


# ---- composed candidates ---------------------------------------------------


def test_composed_total_sums_bankable_rows():
    rows = [
        _row(row_id="a", magnitude_s=0.1, magnitude_kind="realized_through_R"),
        _row(row_id="b", magnitude_s=0.2, magnitude_kind="realized_through_R"),
    ]
    assert composed_candidate_total(rows) == pytest.approx(0.3)


def test_composed_total_refuses_a_mixed_sum_rather_than_skipping_the_bad_part():
    """A silently-skipped part is how a composed total stops meaning its name."""
    rows = [
        _row(row_id="a", magnitude_s=0.1, magnitude_kind="realized_through_R"),
        _row(row_id="b", magnitude_s=9.9, magnitude_kind="description_gap"),
    ]
    with pytest.raises(MagnitudeKindError):
        composed_candidate_total(rows)


def test_composed_total_refuses_an_oracle_part():
    rows = [
        _row(row_id="a", magnitude_s=0.1, magnitude_kind="realized_through_R"),
        _row(row_id="b", magnitude_s=0.4, magnitude_kind="oracle_assisted"),
    ]
    with pytest.raises(MagnitudeKindError):
        composed_candidate_total(rows)


# ---- census reports the denominator ---------------------------------------


def test_census_reports_denominator_not_just_a_count():
    c = magnitude_kind_census()
    assert c["rows_total"] == len(LEDGER)
    assert c["rows_with_magnitude"] <= c["rows_total"]
    assert sum(c["by_kind"].values()) == c["rows_with_magnitude"]


def test_census_empty_oracle_set_is_measured_empty_not_unlooked():
    c = magnitude_kind_census()
    assert c["oracle_assisted_count"] == len(c["oracle_assisted_row_ids"])
    assert c["rows_with_magnitude"] > 0, "an empty numerator over an empty denominator is vacuous"


def test_census_counts_oracle_rows_when_present():
    rows = (
        _row(row_id="a", magnitude_s=0.1, magnitude_kind="realized_through_R"),
        _row(row_id="b", magnitude_s=0.4, magnitude_kind="oracle_assisted"),
    )
    c = magnitude_kind_census(rows)
    assert c["oracle_assisted_row_ids"] == ["b"]
    assert c["bankable_rows"] == 1


def test_live_ledger_bankable_share_is_the_minority_and_is_reported():
    """13 of 17 live magnitudes are description ceilings. A consumer that ignored
    the kind field would sum ~4x the recoverable total."""
    c = magnitude_kind_census()
    assert c["bankable_rows"] < c["rows_with_magnitude"]


def test_every_live_magnitude_row_declares_a_non_NA_kind():
    for r in LEDGER:
        if r.magnitude_s is not None:
            assert r.magnitude_kind in MAGNITUDE_KINDS and r.magnitude_kind != "N_A"


def test_empty_composition_refuses_rather_than_reporting_a_clean_zero():
    """m50 at the composition layer: a filter that dropped every part would
    otherwise read as a measured zero-delta candidate."""
    with pytest.raises(MagnitudeKindError, match="no parts"):
        composed_candidate_total([])


def test_accept_kinds_accepts_a_tuple_not_just_a_set():
    r = _row(magnitude_s=0.9, magnitude_kind="oracle_assisted")
    assert consume_magnitude(r, accept_kinds=("oracle_assisted",)) == 0.9
