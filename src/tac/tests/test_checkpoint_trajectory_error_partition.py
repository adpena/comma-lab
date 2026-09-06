"""Tests for ``checkpoint_trajectory_error_partition_v1`` (ddm_md1, 2026-09-04).

The law's whole value is that its calibration gate is an INTEGER identity, not a tolerance, and
that its reachability read is a CEILING on credit rather than a prediction.  These tests pin both,
plus the honesty fields (axis, non-promotable, cadence-conditional exclusion) that stop the number
being transferred out of its scope.
"""

from __future__ import annotations

import pytest

from tac.canonical_equations.checkpoint_trajectory_error_partition_20260904 import (
    DEFAULT_CHURN_FLIPS,
    DEFAULT_PERSISTENT_FRACTION,
    ERROR_CLASSES,
    MD2_STEP_ZERO_WRONG_POOL_SITES,
    TRAJECTORY_CLASSES,
    build_checkpoint_trajectory_error_partition_v1,
    floor_clears_target,
    partition_is_exact,
    reachability_floor,
)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def test_partition_is_exact_accepts_an_exact_integer_partition() -> None:
    assert partition_is_exact({"a": [3, 4, 5], "b": [1, 1, 0]}, [4, 5, 5]) is True


def test_partition_is_exact_rejects_a_dropped_class() -> None:
    # dropping HEALED from a four-class reading is exactly the defect this refuses.
    assert partition_is_exact({"a": [3, 4, 5]}, [4, 5, 5]) is False


def test_partition_is_exact_rejects_a_double_counted_site() -> None:
    assert partition_is_exact({"a": [4, 5, 5], "b": [4, 5, 5]}, [4, 5, 5]) is False


def test_partition_is_exact_rejects_a_single_bad_checkpoint() -> None:
    # the gate is over EVERY checkpoint; agreeing at the endpoints is not enough.
    assert partition_is_exact({"a": [4, 9, 5], "b": [0, 0, 0]}, [4, 5, 5]) is False


def test_partition_is_exact_refuses_a_ragged_series() -> None:
    with pytest.raises(ValueError):
        partition_is_exact({"a": [1, 2]}, [1, 2, 3])


def test_partition_is_exact_refuses_an_empty_class_map_rather_than_passing_vacuously() -> None:
    # a gate with no classes in it would return True over any total; a vacuous pass is exactly
    # how a broken decomposition ships with a green light.
    with pytest.raises(ValueError):
        partition_is_exact({}, [])
    with pytest.raises(ValueError):
        partition_is_exact({}, [4, 5])


def test_partition_is_exact_is_true_on_a_zero_checkpoint_series_with_classes() -> None:
    assert partition_is_exact({"a": []}, []) is True


# ---------------------------------------------------------------------------
# the reachability read
# ---------------------------------------------------------------------------
def test_reachability_floor_splits_the_terminal_numerator_exactly() -> None:
    out = reachability_floor(terminal_numerator=1000, persistent_numerator=600, denominator=1.0e6)
    assert out["terminal_distortion"] == pytest.approx(1.0e-3)
    assert out["persistent_floor"] == pytest.approx(6.0e-4)
    assert out["optimizer_reachable_distortion"] == pytest.approx(4.0e-4)
    assert out["persistent_share"] == pytest.approx(0.6)
    assert out["optimizer_reachable_share"] == pytest.approx(0.4)
    assert out["persistent_share"] + out["optimizer_reachable_share"] == pytest.approx(1.0)


def test_reachability_floor_is_all_reachable_when_nothing_persists() -> None:
    out = reachability_floor(terminal_numerator=500, persistent_numerator=0, denominator=1000.0)
    assert out["persistent_share"] == 0.0
    assert out["optimizer_reachable_share"] == 1.0


def test_reachability_floor_is_zero_reachable_when_everything_persists() -> None:
    out = reachability_floor(terminal_numerator=500, persistent_numerator=500, denominator=1000.0)
    assert out["persistent_share"] == 1.0
    assert out["optimizer_reachable_share"] == 0.0
    assert out["optimizer_reachable_distortion"] == 0.0


def test_reachability_floor_handles_a_zero_error_run_without_dividing_by_zero() -> None:
    out = reachability_floor(terminal_numerator=0, persistent_numerator=0, denominator=1000.0)
    assert out["persistent_share"] == 0.0
    assert out["optimizer_reachable_share"] == 0.0


def test_reachability_floor_refuses_a_persistent_numerator_above_the_total() -> None:
    with pytest.raises(ValueError):
        reachability_floor(terminal_numerator=10, persistent_numerator=11, denominator=1.0)


def test_reachability_floor_refuses_a_nonpositive_denominator() -> None:
    with pytest.raises(ValueError):
        reachability_floor(terminal_numerator=10, persistent_numerator=1, denominator=0.0)


def test_reachability_floor_refuses_negative_numerators() -> None:
    with pytest.raises(ValueError):
        reachability_floor(terminal_numerator=-1, persistent_numerator=0, denominator=1.0)


# ---------------------------------------------------------------------------
# the verdict helper
# ---------------------------------------------------------------------------
def test_floor_clears_target_is_strict_and_directional() -> None:
    assert floor_clears_target(persistent_floor=1.0e-4, target_distortion=1.3646784205e-4) is True
    assert floor_clears_target(persistent_floor=2.0e-3, target_distortion=1.3646784205e-4) is False
    # a floor exactly AT the target does not clear it: the target is what the score must beat
    assert floor_clears_target(persistent_floor=1.0e-4, target_distortion=1.0e-4) is False


def test_floor_clears_target_refuses_a_nonpositive_target() -> None:
    with pytest.raises(ValueError):
        floor_clears_target(persistent_floor=1.0, target_distortion=0.0)


# ---------------------------------------------------------------------------
# the equation record
# ---------------------------------------------------------------------------
def test_equation_builds_with_five_anchors_and_nonnegative_residuals() -> None:
    eq = build_checkpoint_trajectory_error_partition_v1()
    assert eq.equation_id == "checkpoint_trajectory_error_partition_v1"
    assert len(eq.empirical_anchors) == 5
    assert all(a.residual >= 0.0 for a in eq.empirical_anchors)
    ids = [a.anchor_id for a in eq.empirical_anchors]
    assert any("cold_control" in i for i in ids)
    assert any("warm_transition" in i for i in ids)
    assert any("ng5_tau_band" in i for i in ids)
    assert any("data_order_control" in i for i in ids)
    assert any("different_start" in i for i in ids)
    # every anchor is addressable in the residual map, so no anchor can be added without one
    assert set(eq.predicted_vs_empirical_residual) == set(ids)


def test_ng5_anchor_records_a_missed_prediction_and_the_within_pool_null() -> None:
    """ddm_md2: the burn default RAISED the share, and the overlap carries its honest null."""

    eq = build_checkpoint_trajectory_error_partition_v1()
    ng5 = next(a for a in eq.empirical_anchors if "ng5_tau_band" in a.anchor_id)
    out = ng5.empirical_output
    # the charter predicted 40-55%; it measured higher than the cold control's 62.011%
    assert out["prediction_holds"] is False
    assert out["falsifier_fired"] is False
    assert out["persistent_terminal_share"] > 0.60
    # the burn default repairs more than it creates where the cold control did the reverse
    assert out["created_over_repaired_ratio"] < 1.0
    assert out["cold_created_over_repaired_ratio"] > 2.0
    # ... and still leaves the floor an order of magnitude above the corner
    assert out["persistent_floor_over_target"] > 10.0
    assert out["persistent_floor_over_target"] < out["cold_instance_persistent_floor_over_target"]
    # the site overlap must never be quotable without the null it is measured against
    assert (
        out["persistent_site_overlap_within_pool_chance_jaccard"]
        < out["persistent_site_overlap_with_cold_control_jaccard"]
        < out["persistent_site_overlap_attainable_max_jaccard"]
    )


def test_known_boundary_records_the_shared_root_init_seed_limit() -> None:
    """Every cell descends from one root init seed, so init-invariance is NOT measured -- say so.

    ddm_md4 burned a cell from a different STARTING POINT, which is why the boundary no longer
    claims the starts are shared; what it must still refuse to claim is ROOT-SEED invariance.
    """

    eq = build_checkpoint_trajectory_error_partition_v1()
    boundary = eq.domain_of_validity["known_boundary"].lower()
    assert "one root init seed" in boundary
    assert "init-invariance" in boundary
    assert "unmeasured" in boundary
    ng5 = next(a for a in eq.empirical_anchors if "ng5_tau_band" in a.anchor_id)
    # build_provenance_for_research_sidecar carries reactivation_criteria in rejection_reason
    assert "different initialisation" in ng5.provenance.rejection_reason.lower()


def test_the_two_instances_agree_on_the_share_and_disagree_on_the_sites() -> None:
    eq = build_checkpoint_trajectory_error_partition_v1()
    warm = next(a for a in eq.empirical_anchors if "warm_transition" in a.anchor_id).empirical_output
    # the SHARE transfers across the one optimizer lever ...
    assert abs(warm["two_instance_spread_pp"]) < 5.0
    # ... while the SITE SETS do not, at every measured step
    assert min(warm["warm_born_absent_from_cold_same_step_range"]) > 0.30
    assert warm["warm_born_absent_from_cold_own_peak"] > 0.30


def test_equation_axis_is_advisory_and_non_promotable() -> None:
    eq = build_checkpoint_trajectory_error_partition_v1()
    axis = eq.domain_of_validity["measurement_axis"][0]
    assert "advisory" in axis
    assert "NON-PROMOTABLE" in axis
    assert "NON-PROMOTABLE" in eq.domain_of_validity["result_type"]


def test_equation_excludes_transferring_the_share_across_cadences() -> None:
    # the classes are cadence-conditional by construction; the exclusion must say so, or the
    # number will be lifted into a charter that sampled differently.
    eq = build_checkpoint_trajectory_error_partition_v1()
    excluded = " ".join(eq.domain_of_validity["excluded"])
    assert "cadence" in excluded
    assert "prediction that any lever reaches it" in excluded
    assert "float-tolerance" in excluded
    assert "non-integral sample weights" in excluded.lower()


def test_equation_names_its_producer_and_the_callable_it_exports() -> None:
    eq = build_checkpoint_trajectory_error_partition_v1()
    assert "experiments/ddm_md1_micro_to_macro.py" in eq.canonical_producers
    assert eq.python_callable_module_path.endswith(":reachability_floor")


def test_equation_records_the_preregistered_prediction_and_falsifier() -> None:
    eq = build_checkpoint_trajectory_error_partition_v1()
    predicted = eq.empirical_anchors[0].predicted_output
    assert predicted["preregistered_prediction"] == 0.60
    assert predicted["preregistered_falsifier"] == 0.40
    assert predicted["preregistration"] == ".omx/research/ddm_md1_prereg_20260904.md"


def test_equation_anchor_reports_an_exact_integer_calibration_gate() -> None:
    eq = build_checkpoint_trajectory_error_partition_v1()
    empirical = eq.empirical_anchors[0].empirical_output
    assert empirical["calibration_gate_integer_residual"] == 0


def test_class_vocabulary_is_the_six_way_partition_with_healed_named() -> None:
    assert TRAJECTORY_CLASSES[0] == "ALWAYS_CORRECT"
    assert set(ERROR_CLASSES) == set(TRAJECTORY_CLASSES) - {"ALWAYS_CORRECT"}
    assert "HEALED" in ERROR_CLASSES
    assert len(TRAJECTORY_CLASSES) == 6


def test_default_thresholds_match_the_instrument() -> None:
    md1 = pytest.importorskip("experiments.ddm_md1_micro_to_macro")
    assert DEFAULT_CHURN_FLIPS == md1.DEFAULT_CHURN_FLIPS
    assert DEFAULT_PERSISTENT_FRACTION == md1.PERSISTENT_FRACTION
    assert list(TRAJECTORY_CLASSES) == list(md1.SITE_CLASSES)


def test_equation_is_exported_from_the_registry_package() -> None:
    import tac.canonical_equations as ce

    assert ce.build_checkpoint_trajectory_error_partition_v1().equation_id == (
        "checkpoint_trajectory_error_partition_v1"
    )
    assert ce.reachability_floor is reachability_floor


def test_anchor_carries_the_measured_md1_instance_and_its_reading_dependence() -> None:
    eq = build_checkpoint_trajectory_error_partition_v1()
    out = eq.empirical_anchors[0].empirical_output
    assert out["terminal_weighted_wrong_site_numerator"] == 331_080
    assert out["persistent_weighted_wrong_site_numerator"] == 205_305
    assert out["denominator"] == 117_964_800.0
    assert out["calibration_gate_integer_residual"] == 0
    # the reachability helper must reproduce the anchor's own share from its own numerators
    derived = reachability_floor(
        terminal_numerator=out["terminal_weighted_wrong_site_numerator"],
        persistent_numerator=out["persistent_weighted_wrong_site_numerator"],
        denominator=out["denominator"],
    )
    assert derived["persistent_share"] == pytest.approx(out["persistent_terminal_share"])
    assert derived["persistent_floor"] == pytest.approx(out["persistent_floor_d_seg_hat"])
    # the shadow reading holds the prediction; the live reading would fire the falsifier, and the
    # anchor must carry BOTH so the number is never quoted without its reading.
    assert out["prediction_holds_on_the_shadow_forward"] is True
    assert out["falsifier_fired_on_the_shadow_forward"] is False
    assert out["live_forward_persistent_share"] < 0.40
    assert "SHADOW" in out["live_forward_reading_note"]


def test_anchor_created_over_repaired_ratio_is_consistent_with_its_own_parts() -> None:
    out = build_checkpoint_trajectory_error_partition_v1().empirical_anchors[0].empirical_output
    assert out["created_over_repaired_ratio"] == pytest.approx(
        out["created_over_the_run"] / out["repaired_over_the_run"]
    )
    # net change must reconcile the start and terminal numerators exactly
    assert out["start_numerator"] + out["created_over_the_run"] - out["repaired_over_the_run"] == (
        out["terminal_weighted_wrong_site_numerator"]
    )


def test_floor_does_not_clear_the_sub_012_target_on_the_measured_instance() -> None:
    out = build_checkpoint_trajectory_error_partition_v1().empirical_anchors[0].empirical_output
    assert not floor_clears_target(
        persistent_floor=out["persistent_floor_d_seg_hat"],
        target_distortion=out["sub_012_target_d_seg"],
    )
    assert out["persistent_floor_over_target"] > 12.0


# ---------------------------------------------------------------------------
# ddm_md4: the burned different-START cell
# ---------------------------------------------------------------------------
def _md4_anchor():
    eq = build_checkpoint_trajectory_error_partition_v1()
    return next(a for a in eq.empirical_anchors if "different_start" in a.anchor_id)


def test_md4_anchor_records_the_falsifier_as_FIRED_above_its_pre_registered_bar() -> None:
    """The whole point of the cell: the 0.70 gate had to be able to fire, and it did."""

    out = _md4_anchor().empirical_output
    assert out["falsifier_fired"] is True
    assert out["persistent_site_overlap_with_cold_control_jaccard"] >= 0.70


def test_md4_jaccard_sits_strictly_between_its_null_and_its_attainable_maximum() -> None:
    """A set-overlap number is meaningless without both bounds; a gate pinned at either is fake."""

    out = _md4_anchor().empirical_output
    assert (
        out["persistent_site_overlap_within_pool_chance_jaccard"]
        < out["persistent_site_overlap_with_cold_control_jaccard"]
        < out["persistent_site_overlap_attainable_max_jaccard"]
    )


def test_md4_is_the_first_anchor_whose_step_zero_pool_differs_from_the_cold_control() -> None:
    """Earlier cells shared the pool by construction; this one does not, and that is the lever."""

    out = _md4_anchor().empirical_output
    assert out["step_zero_pool_jaccard_with_cold_control"] < 0.70
    assert out["step_zero_pool_sites"] != MD2_STEP_ZERO_WRONG_POOL_SITES


def test_md4_floor_moves_far_less_than_the_reachable_error_it_shares_a_cell_with() -> None:
    """The share rose because the REACHABLE part shrank, not because the floor grew."""

    md4 = _md4_anchor().empirical_output
    ng5 = next(
        a
        for a in build_checkpoint_trajectory_error_partition_v1().empirical_anchors
        if "ng5_tau_band" in a.anchor_id
    ).empirical_output
    floor_move = abs(md4["persistent_floor_d_seg_hat"] / ng5["persistent_floor_d_seg_hat"] - 1.0)
    md4_reach = md4["terminal_d_seg_hat"] - md4["persistent_floor_d_seg_hat"]
    ng5_reach = ng5["terminal_d_seg_hat"] - ng5["persistent_floor_d_seg_hat"]
    reach_move = abs(md4_reach / ng5_reach - 1.0)
    assert floor_move < 0.01
    assert reach_move > 0.05
    assert reach_move > 10 * floor_move
    # and the share still went UP, which is only coherent with the two facts above
    assert md4["persistent_terminal_share"] > ng5["persistent_terminal_share"]


def test_md4_records_the_step_zero_to_trajectory_transfer_rate() -> None:
    """md3 section 6 named the gap: step-0-wrong is not step-0-UNREACHABLE. This closes it."""

    out = _md4_anchor().empirical_output
    assert 0.85 < out["step_zero_to_trajectory_transfer"] < 0.95
