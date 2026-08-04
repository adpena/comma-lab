"""Tests for the ddm_cg1 actuator-granularity annotation over fl2's force ledger.

Behavioural, not constant-checking: each test would fail if the module were
replaced by a stub returning canonical markers (NO-FAKE forbidden class #2).
"""

from __future__ import annotations

from tac.force_actuator_granularity import (
    AGGREGATE_GRANULARITIES,
    GOVERNING_LAW,
    GRANULARITY,
    PER_ELEMENT_GRANULARITIES,
    coverage,
    granularity_of,
    predicted_dead,
    unclassified_forces,
)
from tac.witness_control.force_class_edge_ledger import LEDGER

# ---- this module must ANNOTATE fl2, never duplicate it ---------------------


def test_every_classified_force_actually_exists_in_fl2s_ledger():
    """A classification for a force fl2 does not have is drift, not coverage."""
    fl2_forces = {r.force for r in LEDGER}
    orphans = sorted(set(GRANULARITY) - fl2_forces)
    assert not orphans, f"classifications with no fl2 force: {orphans}"


def test_no_fl2_force_is_left_unclassified():
    assert unclassified_forces() == []


def test_this_module_does_not_redefine_fl2s_axes():
    import tac.force_actuator_granularity as m

    for owned_by_fl2 in ("VERBS", "VERDICTS", "PROTECTIONS", "CLASSES", "EDGES"):
        assert not hasattr(m, owned_by_fl2), (
            f"{owned_by_fl2} belongs to fl2's ledger; redefining it here forks the schema"
        )


# ---- the axis fl2 cannot express -------------------------------------------


def test_grow_lane_is_an_aggregate_actuator_though_its_verb_is_not_AGGREGATE():
    """The exact hole this module fills.

    fl2 records `as1.grow_lane.harms` with verb="TRANSFER" and a measured
    +0.2459 S harm, but cannot say WHY it failed, because its only marker for
    'aggregate' is a sentinel inside the verb axis.
    """
    row = next(r for r in LEDGER if r.row_id == "as1.grow_lane.harms")
    assert row.verb == "TRANSFER"  # so fl2's AGGREGATE sentinel is unavailable
    assert granularity_of(row.force) in AGGREGATE_GRANULARITIES


def test_class_prior_logit_shift_is_also_aggregate_with_a_real_verb():
    row = next(r for r in LEDGER if r.row_id == "as1.class_prior_logit_shift")
    assert row.verb != "AGGREGATE"
    assert granularity_of(row.force) in AGGREGATE_GRANULARITIES


def test_granularity_is_orthogonal_to_verb():
    """Both aggregate and per-element forces occur under the same verb."""
    by_verb: dict[str, set[str]] = {}
    for r in LEDGER:
        g = granularity_of(r.force)
        if g in AGGREGATE_GRANULARITIES or g in PER_ELEMENT_GRANULARITIES:
            by_verb.setdefault(r.verb, set()).add(
                "AGG" if g in AGGREGATE_GRANULARITIES else "ELEM"
            )
    assert any(v == {"AGG", "ELEM"} for v in by_verb.values()), (
        "if no verb hosts both granularities, the axis would be redundant with verb"
    )


# ---- the law and its falsifier ---------------------------------------------


def test_the_law_scoreboard_is_clean_zero_aggregate_actuators_improved():
    """The law's whole empirical content. If this ever fails, the law is broken
    and that is a MAJOR finding -- do not 'fix' the test, adjudicate the row."""
    assert coverage()["measured_aggregate_rows_that_improved"] == 0


def test_the_law_has_been_tested_against_real_measured_rows():
    """Guards against a vacuous pass: zero-improved is only meaningful if
    aggregate actuators were actually measured (m50)."""
    assert coverage()["measured_aggregate_rows"] >= 8


def test_per_edge_tie_calibration_is_per_tile_not_aggregate():
    """The adjudicated near-counterexample: named per-edge, actuated per-cell.

    ru1 measured '+yield in 17/18 cells' and 'required RGB direction differs per
    edge' -- edge supplies the sign (a prior), cells are the actuator.
    """
    assert granularity_of("per_edge_tie_calibration") == "PER_TILE"
    row = next(r for r in LEDGER if r.row_id == "pc2.tie_calibration")
    assert row.verdict == "IMPROVES" and row.magnitude_kind == "realized_through_R"


def test_predicted_dead_excludes_already_measured_rows():
    """A prediction and a confirmation must not be conflated into one count."""
    for r in predicted_dead():
        assert r.verdict not in ("HARMS", "NEUTRAL", "INERT")


def test_predicted_dead_is_nonempty_so_the_prediction_is_live():
    assert len(predicted_dead()) >= 5


def test_predicted_dead_rows_are_all_aggregate():
    for r in predicted_dead():
        assert granularity_of(r.force) in AGGREGATE_GRANULARITIES


def test_road_undriv_bulk_field_is_predicted_dead():
    """BUILT_UNFIRED per-side scalar on the SAME edge where the per-cell version
    already works -- the sharpest live instance of the prediction."""
    assert "as1.road_undriv_bulk_field" in {r.row_id for r in predicted_dead()}


# ---- coverage reports the denominator --------------------------------------


def test_coverage_reports_denominator_not_just_a_count():
    c = coverage()
    assert c["forces_total"] == len({r.force for r in LEDGER})
    assert c["forces_classified"] + c["forces_unclassified"] == c["forces_total"]


def test_coverage_partitions_forces_without_double_counting():
    c = coverage()
    assert sum(c["by_granularity"].values()) == c["forces_total"]


def test_unknown_force_is_unclassified_not_silently_benign():
    assert granularity_of("a_force_that_does_not_exist") == "UNCLASSIFIED"
    assert "UNCLASSIFIED" not in AGGREGATE_GRANULARITIES
    assert "UNCLASSIFIED" not in PER_ELEMENT_GRANULARITIES


def test_coverage_accepts_an_explicit_row_subset():
    subset = [r for r in LEDGER if r.scope == "Lane"]
    assert coverage(subset)["forces_total"] == len({r.force for r in subset})
    assert coverage(subset)["forces_total"] < coverage()["forces_total"]


def test_aggregate_and_per_element_sets_are_disjoint():
    assert not (AGGREGATE_GRANULARITIES & PER_ELEMENT_GRANULARITIES)


def test_governing_law_states_prior_not_actuator_and_names_its_falsifier():
    assert "PRIOR" in GOVERNING_LAW and "never the actuator" in GOVERNING_LAW
    assert "PREDICTED DEAD" in GOVERNING_LAW
