# SPDX-License-Identifier: MIT
"""Tests for the ddm_mq1 format-vs-search attribution law.

These verify BEHAVIOUR, not constants: every test below would FAIL if the
classifier, either admissibility gate, or the vacuity handling were broken.
"""

from __future__ import annotations

import math

import pytest

from tac.canonical_equations.ddm_mq1_format_vs_search_attribution_20260801 import (
    ARGMIN_AGREEMENT_MEASURED,
    ARGMIN_IDENTIFIED_MIN_AGREEMENT,
    EQUATION_ID,
    LATTICE_PCT_TOTAL,
    RATE_BINDS_MIN_FRACTION,
    SEARCH_DOMINANCE_THRESHOLD,
    SEARCH_PCT_TOTAL,
    build_ddm_mq1_format_vs_search_attribution_v1,
    format_vs_search,
)

BYTE_TO_S = 25.0 / 37_545_489.0


# --------------------------------------------------------------------------- #
# the attribution itself
# --------------------------------------------------------------------------- #
def test_search_dominant_is_classified_search_limited() -> None:
    out = format_vs_search(1.0, 40.0)
    assert out["verdict"] == "SEARCH_LIMITED"
    assert out["search_to_lattice_ratio"] == pytest.approx(40.0)
    assert out["search_share"] == pytest.approx(40.0 / 41.0)


def test_format_dominant_is_classified_format_limited() -> None:
    out = format_vs_search(40.0, 1.0)
    assert out["verdict"] == "FORMAT_LIMITED"


def test_comparable_gaps_refuse_to_pick_a_side() -> None:
    out = format_vs_search(1.0, 1.0)
    assert out["verdict"] == "COMPARABLE_MEASURE_BOTH"


def test_threshold_boundary_is_inclusive_on_the_search_side() -> None:
    at = format_vs_search(1.0, SEARCH_DOMINANCE_THRESHOLD)
    just_below = format_vs_search(1.0, SEARCH_DOMINANCE_THRESHOLD - 1e-9)
    assert at["verdict"] == "SEARCH_LIMITED"
    assert just_below["verdict"] == "COMPARABLE_MEASURE_BOTH"


def test_zero_lattice_gap_gives_infinite_ratio_not_a_crash() -> None:
    out = format_vs_search(0.0, 1.0)
    assert math.isinf(out["search_to_lattice_ratio"])
    assert out["verdict"] == "SEARCH_LIMITED"


def test_negative_gap_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        format_vs_search(-1.0, 1.0)


# --------------------------------------------------------------------------- #
# VACUITY: an unrun gate is never a pass (the empty-scope confound)
# --------------------------------------------------------------------------- #
def test_both_gaps_zero_is_undetermined_not_closed() -> None:
    out = format_vs_search(0.0, 0.0)
    assert out["verdict"] == "UNDETERMINED_EMPTY"
    assert out["codebook_admissible"] is None


def test_unrun_gates_leave_admissibility_undetermined() -> None:
    out = format_vs_search(1.0, 40.0)
    assert out["codebook_admissible"] is None
    assert "gate_not_run" in out["codebook_gate_reason"]


def test_one_gate_alone_is_still_undetermined() -> None:
    only_rate = format_vs_search(1.0, 40.0, addressable_bytes=10**7,
                                 distortion_at_stake_S=0.1)
    only_argmin = format_vs_search(1.0, 40.0, argmin_agreement=1.0)
    assert only_rate["codebook_admissible"] is None
    assert only_argmin["codebook_admissible"] is None


# --------------------------------------------------------------------------- #
# REFUSAL 1 — rate must bind
# --------------------------------------------------------------------------- #
def test_degenerate_rate_refuses_the_codebook() -> None:
    out = format_vs_search(1.0, 40.0, addressable_bytes=123,
                           distortion_at_stake_S=0.2765034, argmin_agreement=1.0)
    assert out["rate_binds"] is False
    assert out["codebook_admissible"] is False
    assert out["codebook_gate_reason"] == "REFUSED_rate_does_not_bind_lambda_degenerate"


def test_binding_rate_passes_gate_one() -> None:
    # bytes chosen so the rate fraction clears the threshold outright
    need = RATE_BINDS_MIN_FRACTION * 0.1 / BYTE_TO_S
    out = format_vs_search(1.0, 40.0, addressable_bytes=int(need * 2),
                           distortion_at_stake_S=0.1, argmin_agreement=1.0)
    assert out["rate_binds"] is True
    assert out["codebook_admissible"] is True
    assert "OBJECTIVE_weighted" in out["codebook_gate_reason"]


def test_zero_distortion_at_stake_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        format_vs_search(1.0, 40.0, addressable_bytes=1, distortion_at_stake_S=0.0)


# --------------------------------------------------------------------------- #
# REFUSAL 2 — the argmin must be identified
# --------------------------------------------------------------------------- #
def test_unidentified_argmin_refuses_even_when_rate_binds() -> None:
    need = RATE_BINDS_MIN_FRACTION * 0.1 / BYTE_TO_S
    out = format_vs_search(1.0, 40.0, addressable_bytes=int(need * 2),
                           distortion_at_stake_S=0.1,
                           argmin_agreement=ARGMIN_AGREEMENT_MEASURED)
    assert out["argmin_identified"] is False
    assert out["codebook_admissible"] is False
    assert out["codebook_gate_reason"] == "REFUSED_argmin_unidentified_no_density_to_fit"


def test_argmin_agreement_outside_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match=r"fraction in \[0,1\]"):
        format_vs_search(1.0, 40.0, argmin_agreement=1.5)


def test_argmin_gate_boundary_is_inclusive() -> None:
    need = int(RATE_BINDS_MIN_FRACTION * 0.1 / BYTE_TO_S) * 2
    at = format_vs_search(1.0, 40.0, addressable_bytes=need,
                          distortion_at_stake_S=0.1,
                          argmin_agreement=ARGMIN_IDENTIFIED_MIN_AGREEMENT)
    assert at["argmin_identified"] is True


# --------------------------------------------------------------------------- #
# the measured v4d anchor reproduces through the callable
# --------------------------------------------------------------------------- #
def test_measured_v4d_pose_payload_is_search_limited_and_refuses_codebook() -> None:
    out = format_vs_search(
        LATTICE_PCT_TOTAL, SEARCH_PCT_TOTAL,
        addressable_bytes=123, distortion_at_stake_S=0.2765034,
        argmin_agreement=ARGMIN_AGREEMENT_MEASURED,
    )
    assert out["verdict"] == "SEARCH_LIMITED"
    # the headline 33x separation, recomputed rather than asserted as a literal
    assert out["search_to_lattice_ratio"] == pytest.approx(
        SEARCH_PCT_TOTAL / LATTICE_PCT_TOTAL)
    assert out["search_to_lattice_ratio"] > 30.0
    assert out["codebook_admissible"] is False


def test_negative_control_column_still_reads_search_limited() -> None:
    """p0 is the WEAKEST coordinate; even it clears the threshold (6.6x)."""
    out = format_vs_search(0.0213, 0.1412)
    assert out["verdict"] == "SEARCH_LIMITED"
    assert out["search_to_lattice_ratio"] > SEARCH_DOMINANCE_THRESHOLD


# --------------------------------------------------------------------------- #
# equation object contract
# --------------------------------------------------------------------------- #
def test_equation_builds_with_two_independent_anchors() -> None:
    eq = build_ddm_mq1_format_vs_search_attribution_v1()
    assert eq.equation_id == EQUATION_ID
    assert len(eq.empirical_anchors) == 2
    ids = {a.anchor_id for a in eq.empirical_anchors}
    assert any("format_vs_search" in i for i in ids)
    assert any("two_refusals" in i for i in ids)
    for a in eq.empirical_anchors:
        assert a.empirical_verification_status == "VERIFIED_VIA_EMPIRICAL_ANCHOR"


def test_equation_declares_producers_and_consumers() -> None:
    eq = build_ddm_mq1_format_vs_search_attribution_v1()
    assert eq.canonical_producers, "orphan equation: no producers"
    assert eq.canonical_consumers, "orphan equation: no consumers"


def test_equation_excludes_promotion_use_and_additive_gaps() -> None:
    eq = build_ddm_mq1_format_vs_search_attribution_v1()
    excluded = " ".join(eq.domain_of_validity["excluded"]).lower()
    assert "additive" in excluded
    assert "promotion" in excluded


def test_registered_evaluator_rejects_off_contract_inputs() -> None:
    from tac.canonical_equations.evaluators import get_evaluator

    ev = get_evaluator(EQUATION_ID)
    assert ev({"gap_lattice": 1.0, "gap_search": 40.0})["verdict"] == "SEARCH_LIMITED"
    with pytest.raises(ValueError, match="canonical callable contract"):
        ev({"gap_lattice": 1.0})
    with pytest.raises(ValueError, match="canonical callable contract"):
        ev({"gap_lattice": 1.0, "gap_search": 40.0, "bogus": 1})
