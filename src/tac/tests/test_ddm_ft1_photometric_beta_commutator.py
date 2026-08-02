# SPDX-License-Identifier: MIT
"""Tests for ddm_ft1 — the photometric<->beta commutator channel decomposition.

Every test below verifies BEHAVIOUR on constructed inputs whose correct answer
is known independently of the implementation (a hand-placed saturated pixel, a
hand-placed sub-quantum move, a quadratic whose stationary point is known in
closed form).  None of them assert a constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.ddm_ft1_photometric_beta_commutator_20260802 import (
    FIT_MENU_MAGNITUDES,
    FT1_N600_CENSUS,
    FT1_N600_FULL_DISPLACEMENT_CENSUS,
    OPERATOR_COMMUTATION_CONTROL,
    clamp_channel_is_inert,
    commutator_channels,
    fiber_transport_delta_ab,
    staleness_priority,
    transport_admissible,
)


# --------------------------------------------------------------------------- #
# commutator_channels
# --------------------------------------------------------------------------- #
def test_identical_frames_report_every_channel_zero_and_beta_inert():
    y = np.array([[10.0, 20.0], [30.0, 40.0]])
    ch = commutator_channels(y_fit=y, y_ship=y.copy())
    assert ch["clip_sym_diff"] == 0.0
    assert ch["round_only_frac"] == 0.0
    assert ch["u8_diff_frac"] == 0.0
    assert ch["pre_rms"] == 0.0
    assert ch["beta_inert"] is True


def test_shape_mismatch_raises_rather_than_broadcasting():
    with pytest.raises(ValueError, match="shape mismatch"):
        commutator_channels(y_fit=np.zeros((2, 2)), y_ship=np.zeros((2, 3)))


def test_clamp_channel_counts_only_saturation_status_flips():
    # one pixel crosses 255.5 (saturates), one moves a lot but stays in range.
    y_fit = np.array([255.0, 10.0, 100.0])
    y_ship = np.array([300.0, 60.0, 100.0])
    ch = commutator_channels(y_fit=y_fit, y_ship=y_ship)
    assert ch["clip_sym_diff"] == pytest.approx(1.0 / 3.0)
    assert ch["clip_frac_fit"] == 0.0
    assert ch["clip_frac_ship"] == pytest.approx(1.0 / 3.0)


def test_clamp_channel_zero_when_both_geometries_saturate_the_same_pixel():
    # Saturation present but UNCHANGED -> the operators still commuted here.
    y_fit = np.array([300.0, 10.0])
    y_ship = np.array([400.0, 10.0])
    ch = commutator_channels(y_fit=y_fit, y_ship=y_ship)
    assert ch["clip_frac_fit"] == pytest.approx(0.5)
    assert ch["clip_sym_diff"] == 0.0
    assert ch["u8_diff_frac"] == 0.0  # both clamp to 255


def test_clip_mass_measures_signed_excess_on_both_sides():
    ch = commutator_channels(y_fit=np.array([-4.0, 259.0]),
                             y_ship=np.array([0.0, 255.0]))
    assert ch["clip_mass_fit"] == pytest.approx((4.0 + 4.0) / 2.0)
    assert ch["clip_mass_ship"] == 0.0


def test_round_channel_catches_sub_quantum_moves_that_change_uint8():
    # 0.4 < 0.5 quantum, but it steps across the rounding boundary at 10.5.
    ch = commutator_channels(y_fit=np.array([10.3]), y_ship=np.array([10.7]))
    assert ch["u8_diff_frac"] == 1.0
    assert ch["round_only_frac"] == 1.0
    assert ch["pre_max_abs"] == pytest.approx(0.4)


def test_sub_quantum_move_that_does_not_cross_a_boundary_is_not_counted():
    ch = commutator_channels(y_fit=np.array([10.1]), y_ship=np.array([10.4]))
    assert ch["u8_diff_frac"] == 0.0
    assert ch["round_only_frac"] == 0.0
    assert ch["beta_inert"] is True


def test_geometry_channel_reports_pre_quantization_magnitude():
    y_fit = np.array([0.0, 0.0, 0.0, 0.0])
    y_ship = np.array([3.0, 4.0, 0.0, 0.0])
    ch = commutator_channels(y_fit=y_fit, y_ship=y_ship)
    assert ch["pre_rms"] == pytest.approx(np.sqrt(25.0 / 4.0))
    assert ch["pre_max_abs"] == pytest.approx(4.0)
    assert ch["u8_diff_frac"] == pytest.approx(0.5)


def test_large_move_is_not_attributed_to_the_round_channel():
    ch = commutator_channels(y_fit=np.array([10.0]), y_ship=np.array([40.0]))
    assert ch["u8_diff_frac"] == 1.0
    assert ch["round_only_frac"] == 0.0


def test_custom_range_bounds_are_honoured():
    ch = commutator_channels(y_fit=np.array([5.0]), y_ship=np.array([20.0]),
                             lo=0.0, hi=15.0)
    assert ch["clip_frac_ship"] == 1.0
    assert ch["clip_sym_diff"] == 1.0


def test_clamp_inert_predicate_tracks_the_clamp_channel_only():
    inert = commutator_channels(y_fit=np.array([10.0]), y_ship=np.array([40.0]))
    assert clamp_channel_is_inert(inert) is True   # big move, no saturation
    flipped = commutator_channels(y_fit=np.array([10.0]),
                                  y_ship=np.array([900.0]))
    assert clamp_channel_is_inert(flipped) is False


# --------------------------------------------------------------------------- #
# staleness_priority — the ranking that survived the falsification
# --------------------------------------------------------------------------- #
def test_priority_key_is_geometry_not_clipping():
    big_move_no_clip = commutator_channels(y_fit=np.array([10.0, 10.0]),
                                           y_ship=np.array([40.0, 40.0]))
    tiny_move_with_clip = commutator_channels(y_fit=np.array([254.0, 10.0]),
                                              y_ship=np.array([256.0, 10.0]))
    hi = staleness_priority(big_move_no_clip)
    lo = staleness_priority(tiny_move_with_clip)
    # the clipping pair has the larger clamp channel...
    assert lo["clamp_channel_bound"] > hi["clamp_channel_bound"]
    # ...and the SMALLER priority, which is the whole point of the falsification
    assert hi["key"] > lo["key"]
    assert hi["key_name"] == "pre_rms"


def test_priority_marks_only_a_genuinely_inert_partner_retirable():
    inert = commutator_channels(y_fit=np.array([10.1]), y_ship=np.array([10.4]))
    assert staleness_priority(inert)["retirable"] is True
    active = commutator_channels(y_fit=np.array([10.0]), y_ship=np.array([40.0]))
    assert staleness_priority(active)["retirable"] is False


def test_priority_names_the_forbidden_key_so_it_cannot_be_rediscovered():
    p = staleness_priority(commutator_channels(y_fit=np.array([1.0]),
                                               y_ship=np.array([2.0])))
    assert "FALSIFIED" in p["forbidden_key"]
    assert "clip" in p["forbidden_key"]


# --------------------------------------------------------------------------- #
# transport_admissible — the hull boundary
# --------------------------------------------------------------------------- #
def test_transport_admissible_inside_the_fitted_hull():
    v = transport_admissible(delta_partner=0.5)
    assert v["admissible"] is True
    assert v["required_action"] == "linear_fiber_transport"


def test_transport_refused_past_the_menu_maximum():
    v = transport_admissible(delta_partner=7.5)
    assert v["admissible"] is False
    assert v["outside_magnitude"] is True
    assert v["required_action"] == "joint_gauss_newton_resolve"


def test_transport_refused_for_a_sign_the_fit_could_not_express():
    v = transport_admissible(delta_partner=0.5, opposes_fit_sign=True)
    assert v["admissible"] is False
    assert v["outside_sign"] is True
    assert v["outside_magnitude"] is False


def test_hull_boundary_value_is_inside_not_outside():
    assert transport_admissible(delta_partner=1.0)["admissible"] is True
    assert transport_admissible(delta_partner=1.0001)["admissible"] is False


# --------------------------------------------------------------------------- #
# fiber_transport_delta_ab — the implicit-function-theorem step
# --------------------------------------------------------------------------- #
def test_transport_recovers_the_exact_stationary_shift_of_a_quadratic():
    """L(z,b) = 0.5 (z - m b)^T H (z - m b): argmin is z*(b) = m b exactly.

    So the true shift over delta_b is m*delta_b, and the mixed partial that
    feeds the transport is d/db grad_z L = -H m.
    """
    h = np.array([[4.0, 1.0], [1.0, 3.0]])
    m = np.array([0.7, -0.2])
    mixed = -h @ m
    got = fiber_transport_delta_ab(hessian_ab=h, mixed_partial=mixed,
                                   delta_partner=0.25)
    np.testing.assert_allclose(got, m * 0.25, rtol=1e-12)


def test_transport_is_linear_in_the_partner_step():
    h = np.eye(2) * 2.0
    m = np.array([1.0, -3.0])
    one = fiber_transport_delta_ab(hessian_ab=h, mixed_partial=m,
                                   delta_partner=1.0)
    two = fiber_transport_delta_ab(hessian_ab=h, mixed_partial=m,
                                   delta_partner=2.0)
    np.testing.assert_allclose(two, 2.0 * one, rtol=1e-12)


def test_zero_partner_step_transports_nothing():
    got = fiber_transport_delta_ab(hessian_ab=np.eye(2),
                                   mixed_partial=np.array([5.0, 5.0]),
                                   delta_partner=0.0)
    np.testing.assert_allclose(got, np.zeros(2))


def test_singular_hessian_is_refused_not_pseudo_inverted():
    with pytest.raises(ValueError, match="singular"):
        fiber_transport_delta_ab(hessian_ab=np.array([[1.0, 0.0], [0.0, 0.0]]),
                                 mixed_partial=np.array([1.0, 1.0]),
                                 delta_partner=1.0)


def test_indefinite_hessian_is_refused():
    with pytest.raises(ValueError, match=r"singular|indefinite"):
        fiber_transport_delta_ab(hessian_ab=np.array([[1.0, 0.0], [0.0, -2.0]]),
                                 mixed_partial=np.array([1.0, 1.0]),
                                 delta_partner=1.0)


def test_non_finite_inputs_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        fiber_transport_delta_ab(hessian_ab=np.eye(2),
                                 mixed_partial=np.array([np.nan, 0.0]),
                                 delta_partner=1.0)


# --------------------------------------------------------------------------- #
# recorded measurements
# --------------------------------------------------------------------------- #
def test_fit_menu_is_the_v4d_seed_menu():
    assert FIT_MENU_MAGNITUDES == (0.0, 0.5, 1.0)


def test_operator_commutation_control_is_confirmed_at_roundoff():
    assert OPERATOR_COMMUTATION_CONTROL["verdict"] == "CONFIRMED_AT_FLOAT_ROUNDOFF"
    assert OPERATOR_COMMUTATION_CONTROL["max_rel_residual"] < 1e-12


def test_census_records_a_scoped_negative_and_a_denominator():
    assert FT1_N600_CENSUS["verdict_scope"] == "formulation"
    assert FT1_N600_CENSUS["pairs"] == 600
    # the population the verdict is about must be non-empty (vacuity rule)
    assert FT1_N600_CENSUS["stale"] > 0
    assert FT1_N600_CENSUS["score_claim"] is False


def test_census_hull_escape_union_is_at_least_each_part():
    c = FT1_N600_CENSUS
    assert c["outside_fitted_set_union"] >= c["extrapolated_magnitude"]
    assert c["outside_fitted_set_union"] >= c["opposing_sign"]
    assert c["outside_fitted_set_union"] <= (c["extrapolated_magnitude"]
                                             + c["opposing_sign"])
    assert c["stale_and_outside_fitted_set"] <= c["stale"]


def test_census_carries_the_reductio_that_kills_the_clip_criterion():
    c = FT1_N600_CENSUS
    # the worst-drift pair in the population has ZERO clamp channel, so the
    # clip criterion retires it -- that is the falsification, in the data.
    assert c["worst_drift_pair"]["clip_sym_diff"] == 0.0
    assert c["worst_drift_pair"]["pre_rms"] == c["pre_rms_max"]
    assert c["clip_criterion_would_retire"] > c["geometry_criterion_would_retire"]


def test_census_predictor_correlations_are_negligible_and_ranker_is_not():
    c = FT1_N600_CENSUS
    assert abs(c["pearson_clip_frac_vs_abs_delta_beta"]) < 0.1
    assert abs(c["pearson_clip_sym_diff_vs_abs_delta_beta"]) < 0.1
    assert c["pearson_pre_rms_vs_abs_delta_beta"] > 0.5


def test_round_channel_dominates_the_clamp_channel_as_measured():
    c = FT1_N600_CENSUS
    assert c["round_channel_share_of_u8_change"] > 0.5
    assert c["clamp_share_of_total_change_mean"] < 1e-3


def test_full_displacement_census_widens_rather_than_contradicts_the_beta_one():
    full, beta = FT1_N600_FULL_DISPLACEMENT_CENSUS, FT1_N600_CENSUS
    assert full["pairs"] == beta["pairs"] == 600
    # widening the partner set can only ADD stale rows, never remove them
    assert full["stale_any_partner"] >= beta["stale"]
    assert full["stale_via_beta_only"] + full["stale_via_both"] == full["stale_any_partner"]


def test_full_displacement_census_records_that_no_stage_resolved_ab():
    # this is the whole defect: later stages moved partners and left (a,b) put.
    assert FT1_N600_FULL_DISPLACEMENT_CENSUS["ab_resolved_by_later_stages"] == 0
    assert FT1_N600_FULL_DISPLACEMENT_CENSUS["pose_dims_drifted"] > 0


def test_no_pair_has_an_inert_partner_so_nothing_retires_for_free():
    full = FT1_N600_FULL_DISPLACEMENT_CENSUS
    assert full["pairs_with_inert_partner"] == 0
    # ...yet the falsified criterion would still have retired 199 of them
    assert full["clip_criterion_would_retire"] > 0
    assert full["clip_criterion_retired_max_pre_rms"] > 1.0
