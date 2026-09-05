"""Re-derivation guards for the ddm_hc2 flip-location address-floor equation.

The law was MEASURED by the ddm_hc2 arm; its memo is the primary artifact.  These tests do not
restate the memo -- they pin the arithmetic the headline constants are built from and the three
decisions the equation exists to make:

  * the LOSSLESS SPLIT: hc1's indicator/conditional decomposition of the shipped stream is exact,
    and its branches sum to the measured indicator;
  * the ADDRESS-COUNT RATIO: components / sites is the whole structural budget of a clustering
    representation, and on this body it leaves too little headroom to clear the bar;
  * the BAND COROLLARY: the boundary-offset premise is TRUE and still refused, because the
    incumbent already spends almost nothing outside the derivable band.
"""

from __future__ import annotations

import math

from tac.canonical_equations.flip_location_component_address_floor_20260905 import (
    BAND_FLIPS,
    BAND_OUT_OF_BAND_INDICATOR_BYTES,
    BAND_POSITIONS,
    BITS_PER_FLIP_NO_BRANCH,
    COMPONENT_SIZE_BUCKETS_8CONN,
    COMPONENTS_4CONN,
    COMPONENTS_8CONN,
    CONDITIONAL_BYTES,
    EQUATION_ID,
    FLIPS,
    FLIPS_AT_TOKEN_CLASS_DISTANCE_1,
    GE16_BYTES,
    GE16_COMPONENT_SHARE,
    GE16_COMPONENTS,
    GE16_SITES,
    INDICATOR_BYTES,
    LIVE_POSITIONS,
    NO_BRANCH_BYTES,
    POSITIONS,
    REFUSE_BELOW_BYTES,
    SATURATED_FLIPS,
    STREAM_BYTES,
    STREAM_BYTES_IDEAL,
    YES_BRANCH_BYTES,
    address_count_ratio,
    boundary_band_is_already_localised,
    build_flip_location_component_address_floor_v1,
    ceiling_refused,
    clustering_headroom_fraction,
    component_representation_can_pay,
    component_representation_headroom_bytes,
)


def test_hc1_split_is_exact_and_reconstructs_the_shipped_stream():
    """indicator = yes-branch + no-branch, and indicator + conditional = the stream's ideal length."""
    assert math.isclose(YES_BRANCH_BYTES + NO_BRANCH_BYTES, INDICATOR_BYTES, abs_tol=1e-6)
    assert math.isclose(
        INDICATOR_BYTES + CONDITIONAL_BYTES, STREAM_BYTES_IDEAL, abs_tol=1e-6
    )
    # the ideal length rounds up into the shipped stream's byte count
    assert 0.0 < STREAM_BYTES - STREAM_BYTES_IDEAL < 1.0
    # hc1's headline: the indicator is ~97.8 % of the stream
    assert 0.977 < INDICATOR_BYTES / STREAM_BYTES_IDEAL < 0.979


def test_bits_per_flip_is_the_no_branch_sum_over_the_flip_count():
    assert math.isclose(
        BITS_PER_FLIP_NO_BRANCH, NO_BRANCH_BYTES * 8.0 / FLIPS, rel_tol=1e-9
    )
    # hc1 measured 2.6917 bits/flip on the dx2 body; the fs2 body agrees to 0.4 %
    assert abs(BITS_PER_FLIP_NO_BRANCH - 2.6917) < 0.02


def test_saturated_positions_hold_no_flips_and_live_count_matches_the_sister_arms():
    assert SATURATED_FLIPS == 0
    assert LIVE_POSITIONS == 50_009_121  # mi1 / mc1 re-derived independently
    assert LIVE_POSITIONS < POSITIONS


def test_component_buckets_account_for_every_flip_site_and_every_no_branch_byte():
    comps = sum(c for c, _s, _b in COMPONENT_SIZE_BUCKETS_8CONN.values())
    sites = sum(s for _c, s, _b in COMPONENT_SIZE_BUCKETS_8CONN.values())
    byts = sum(b for _c, _s, b in COMPONENT_SIZE_BUCKETS_8CONN.values())
    assert comps == COMPONENTS_8CONN
    assert sites == FLIPS
    assert math.isclose(byts, NO_BRANCH_BYTES, rel_tol=2e-5)


def test_address_count_ratio_is_the_clustering_budget():
    rho = address_count_ratio()
    assert math.isclose(rho, COMPONENTS_8CONN / FLIPS, rel_tol=1e-12)
    assert 0.75 < rho < 0.76
    assert math.isclose(clustering_headroom_fraction(), 1.0 - rho, rel_tol=1e-12)
    # 4-connectivity clusters even less than 8-connectivity
    assert COMPONENTS_4CONN > COMPONENTS_8CONN
    assert address_count_ratio(COMPONENTS_4CONN, FLIPS) > rho


def test_the_prior_law_premise_about_large_components_is_falsified_by_the_geometry():
    """The charter predicted the saving would sit in the <= 20 % of components with >= 16 sites."""
    assert math.isclose(GE16_COMPONENT_SHARE, GE16_COMPONENTS / COMPONENTS_8CONN, rel_tol=1e-3)
    assert GE16_COMPONENT_SHARE < 1e-4  # measured 0.0052 %, not 20 %
    assert GE16_SITES / FLIPS < 1e-3
    assert GE16_BYTES / NO_BRANCH_BYTES < 1e-3
    # singletons dominate: bucket "1" holds most of the components and most of the bytes
    singleton_comps, singleton_sites, singleton_bytes = COMPONENT_SIZE_BUCKETS_8CONN["1"]
    assert singleton_comps == singleton_sites
    assert singleton_comps / COMPONENTS_8CONN > 0.80
    assert singleton_bytes / NO_BRANCH_BYTES > 0.60


def test_component_gate_is_only_necessary_and_this_body_PASSES_it():
    """The honest boundary: the address-count arithmetic alone did NOT refuse this arm.

    Absorbed sites 227,555 - 172,193 = 55,362, priced at the incumbent's mean 0.335 B/site, is
    18,554 B of optimistic headroom -- above the 5,000 B bar.  The refusal came from the MEASURED
    shape code, not from the geometry.  Anyone quoting the ratio as the closure is overstating it.
    """
    headroom = component_representation_headroom_bytes(
        components=COMPONENTS_8CONN, sites=FLIPS, incumbent_bytes=NO_BRANCH_BYTES
    )
    assert math.isclose(
        headroom, (FLIPS - COMPONENTS_8CONN) * NO_BRANCH_BYTES / FLIPS, rel_tol=1e-12
    )
    assert 18_000 < headroom < 19_000
    assert component_representation_can_pay(
        components=COMPONENTS_8CONN,
        sites=FLIPS,
        incumbent_bytes=NO_BRANCH_BYTES,
        required_saving_bytes=REFUSE_BELOW_BYTES,
    )
    # a set with no absorbed sites at all IS hopeless a priori -- that is all the gate settles
    assert not component_representation_can_pay(
        components=FLIPS,
        sites=FLIPS,
        incumbent_bytes=NO_BRANCH_BYTES,
    )
    assert (
        component_representation_headroom_bytes(
            components=FLIPS, sites=FLIPS, incumbent_bytes=NO_BRANCH_BYTES
        )
        == 0.0
    )


def test_boundary_offset_premise_is_true_but_the_band_is_already_priced():
    assert FLIPS_AT_TOKEN_CLASS_DISTANCE_1 / FLIPS > 0.99  # flips ARE one-pixel boundary moves
    assert BAND_FLIPS[1] / FLIPS > 0.994  # and they sit in the D=1 band
    assert BAND_POSITIONS[1] / POSITIONS < 0.03  # which is under 3 % of the field
    # yet the incumbent already spends under 5 % of its indicator outside that band
    assert not boundary_band_is_already_localised(
        out_of_band_indicator_bytes=BAND_OUT_OF_BAND_INDICATOR_BYTES[1],
        indicator_bytes=INDICATOR_BYTES,
    )
    assert BAND_OUT_OF_BAND_INDICATOR_BYTES[1] / INDICATOR_BYTES < 0.05
    # widening the band only shrinks what is left outside it
    assert (
        BAND_OUT_OF_BAND_INDICATOR_BYTES[3]
        < BAND_OUT_OF_BAND_INDICATOR_BYTES[2]
        < BAND_OUT_OF_BAND_INDICATOR_BYTES[1]
    )
    # a coder that had NOT localised the boundary would pass the same check
    assert boundary_band_is_already_localised(
        out_of_band_indicator_bytes=0.4 * INDICATOR_BYTES, indicator_bytes=INDICATOR_BYTES
    )


def test_refusal_bar_is_the_charter_bar():
    assert REFUSE_BELOW_BYTES == 5_000.0
    assert ceiling_refused(4_999.9)
    assert not ceiling_refused(5_000.0)


def test_equation_builds_with_three_verified_anchors_and_a_refusal_verdict():
    eq = build_flip_location_component_address_floor_v1()
    assert eq.equation_id == EQUATION_ID
    assert len(eq.empirical_anchors) == 3
    assert len(eq.predicted_vs_empirical_residual) == 3
    ceiling = next(a for a in eq.empirical_anchors if "no_location_set_representation" in a.anchor_id)
    assert ceiling.empirical_output["typed_verdict"] == "CEILING-REFUSED"
    assert "REFUSAL-ONLY" in " ".join(eq.domain_of_validity["measurement_axis"])
    assert "NON-PROMOTABLE" in eq.domain_of_validity["result_type"]
