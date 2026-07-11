# SPDX-License-Identifier: MIT
"""Behavior tests for the four Einstein-pass covariance laws (2026-07-10)."""
from __future__ import annotations

import math

import pytest

from tac.canonical_equations.einstein_pass_covariance_laws_20260710 import (
    ALL_EINSTEIN_PASS_BUILDERS,
    COVARIANCE_CLASSES,
    COVARIANT_LAW,
    GAUGE_MEASUREMENT,
    SCORER_FRAME_STRUCTURAL,
    allowed_island_count_delta,
    dseg_covariant_residual,
    flip_byte_exchange_rate,
    implied_flip_quanta,
    score_delta_per_archive_byte,
    score_delta_per_pixel_pair_flip,
    smooth_witness_dseg_floor,
    smooth_witness_triple_costs,
    stored_correction_is_score_positive,
)


def test_all_four_equations_build_and_validate() -> None:
    eqs = [b() for b in ALL_EINSTEIN_PASS_BUILDERS]
    assert len({e.equation_id for e in eqs}) == 4
    for e in eqs:
        assert e.empirical_anchors, e.equation_id
        assert e.canonical_producers and e.canonical_consumers, e.equation_id


def test_every_equation_carries_covariance_class_metadata() -> None:
    for build in ALL_EINSTEIN_PASS_BUILDERS:
        eq = build()
        cc = eq.domain_of_validity["covariance_class"]
        if isinstance(cc, dict):
            # law-vs-constant split (the spike-floor case): every value in vocabulary
            assert cc["law"] == COVARIANT_LAW
            assert GAUGE_MEASUREMENT in cc.values()
            for v in cc.values():
                assert v in COVARIANCE_CLASSES
        else:
            assert cc in COVARIANCE_CLASSES


# ---- Law 1: atomic units + exchange rate (exact arithmetic, not constants-only) ----
def test_flip_quantum_exact_arithmetic() -> None:
    # recompute from independent arithmetic, not from the module constants
    assert score_delta_per_pixel_pair_flip() == pytest.approx(
        100.0 / (600 * 384 * 512), rel=0, abs=0
    )
    assert score_delta_per_archive_byte() == pytest.approx(25.0 / 37_545_489, rel=0, abs=0)


def test_exchange_rate_value_and_parameter_dependence() -> None:
    rate = flip_byte_exchange_rate()
    assert rate == pytest.approx(1.27311, rel=1e-4)
    # behavior, not constant: halving the pair count doubles the flip quantum
    assert flip_byte_exchange_rate(pairs=300) == pytest.approx(2 * rate, rel=1e-12)
    # and a bigger uncompressed denominator cheapens bytes -> more bytes per flip
    assert flip_byte_exchange_rate(n_uncompressed=2 * 37_545_489) == pytest.approx(
        2 * rate, rel=1e-12
    )


def test_stored_correction_break_even_predicate() -> None:
    # 10 flips buy strictly fewer than 12.73 bytes
    assert stored_correction_is_score_positive(12.7, 10.0)
    assert not stored_correction_is_score_positive(12.8, 10.0)
    assert not stored_correction_is_score_positive(0.0, 0.0)  # strict inequality


def test_pointer_move_is_twenty_flip_quanta_within_rounding() -> None:
    quanta = implied_flip_quanta(-1.6999999999989246e-05)
    assert quanta == pytest.approx(20.0, abs=0.1)
    # residual vs the integer is far below the reported-component rounding (~1e-6 S)
    assert abs(20 * score_delta_per_pixel_pair_flip() - 1.7e-05) < 1e-6


# ---- Law 2: temporal-majority floor ----
def test_majority_is_optimal_over_the_spike_triple() -> None:
    costs = smooth_witness_triple_costs()
    assert costs["majority"] == 1 == min(costs.values())
    assert costs["spike"] == 2 and costs["other"] == 3


def test_smooth_witness_floor_equals_spike_rate_and_rejects_nonminority() -> None:
    assert smooth_witness_dseg_floor(0.005318) == pytest.approx(0.005318)
    assert smooth_witness_dseg_floor(0.0) == 0.0
    with pytest.raises(ValueError):
        smooth_witness_dseg_floor(0.6)
    with pytest.raises(ValueError):
        smooth_witness_dseg_floor(-0.01)


def test_floor_law_scope_declares_family_level_and_gauge_constant() -> None:
    from tac.canonical_equations.einstein_pass_covariance_laws_20260710 import (
        build_gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1 as b,
    )

    eq = b()
    assert "FAMILY-level" in eq.domain_of_validity["scope"]
    a = eq.empirical_anchors[0]
    assert a.inputs["gt_spike_rate_n600"] == 0.005318
    assert a.inputs["repairable_fraction"] == 0.977
    # the need band is 4.5-7x below the floor (the strategic content)
    lo, hi = a.inputs["sub015_need_band"]
    assert 4.0 < 0.005318 / hi < 0.005318 / lo < 7.5


# ---- Law 3: covariant/gauge decomposition ----
def test_covariant_residual_clamps_and_subtracts() -> None:
    assert dseg_covariant_residual(0.0052, 0.005318) == 0.0  # converged witness: d_cov ~ 0
    assert dseg_covariant_residual(0.008, 0.0053) == pytest.approx(0.0027)
    assert dseg_covariant_residual(0.0, 0.0053) == 0.0


def test_decomposition_law_forces_T1_and_ground_frame() -> None:
    from tac.canonical_equations.einstein_pass_covariance_laws_20260710 import (
        build_dseg_covariant_gauge_decomposition_v1 as b,
    )

    eq = b()
    forces = " ".join(eq.domain_of_validity["forces"])
    assert "phase_advection_consistency" in forces
    assert "ground-frame" in forces
    assert eq.domain_of_validity["covariance_class"] == COVARIANT_LAW


# ---- Law 4: island charge conservation ----
def test_island_delta_bounded_by_events() -> None:
    assert allowed_island_count_delta(0) == 0
    assert allowed_island_count_delta(3) == 3
    with pytest.raises(ValueError):
        allowed_island_count_delta(-1)


def test_island_law_consequences_forbid_capacity_chasing() -> None:
    from tac.canonical_equations.einstein_pass_covariance_laws_20260710 import (
        build_island_topological_charge_conservation_v1 as b,
    )

    eq = b()
    cons = " ".join(eq.domain_of_validity["consequences"])
    assert "capacity" in cons and "source" in cons
    assert eq.domain_of_validity["covariance_class"] == COVARIANT_LAW


def test_flip_exchange_is_scorer_frame_structural_not_covariant() -> None:
    from tac.canonical_equations.einstein_pass_covariance_laws_20260710 import (
        build_score_atomic_flip_byte_exchange_v1 as b,
    )

    eq = b()
    # the exchange rate depends on grid/pairs/N: apparatus-frame, NOT frame-free
    assert eq.domain_of_validity["covariance_class"] == SCORER_FRAME_STRUCTURAL
    assert math.isfinite(flip_byte_exchange_rate())
