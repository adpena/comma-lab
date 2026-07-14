from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from tac.local_acceleration.throughput_frontier_math import (
    PrecisionLayer,
    PrecisionOption,
    SupportCost,
    cadence_compute_fraction,
    certify_argmax_intervals,
    continuous_margin_waterfill,
    covariant_pair_reuse_break_even,
    crt_reduction_certificate,
    fixed_width_reduction_certificate,
    float32_ordered_sum,
    integer_ordered_sum,
    max_plus_ordered_reduce,
    minimum_signed_accumulator_bits,
    multiphase_surface_tension_metric_certificate,
    number_system_disposition,
    ordinal_top1_concordance_diagnostic,
    solve_discrete_precision_waterfill,
    support_closure_flop_accounting,
    tropical_argmax_ordered_reduce,
    winner_rival_margin_hinge,
)


def test_minimum_accumulator_width_is_tight() -> None:
    assert minimum_signed_accumulator_bits(max_abs_term=0, fan_in=10) == 1
    assert minimum_signed_accumulator_bits(max_abs_term=127, fan_in=10) == 12
    cert = fixed_width_reduction_certificate(
        max_abs_term=127,
        fan_in=10,
        accumulator_bits=12,
    )
    assert cert["sum_abs_bound"] == 1270
    assert cert["no_overflow"] is True
    assert fixed_width_reduction_certificate(
        max_abs_term=127,
        fan_in=10,
        accumulator_bits=11,
    )["no_overflow"] is False


def test_integer_and_max_plus_are_permutation_invariant_but_fp32_is_not() -> None:
    float_values = np.asarray([2**24, 1.0, -(2**24)], dtype=np.float32)
    orders = tuple(itertools.permutations(range(3)))
    float_results = {float(float32_ordered_sum(float_values, order)) for order in orders}
    assert float_results == {0.0, 1.0}

    integer_values = (2**24, 1, -(2**24))
    assert {integer_ordered_sum(integer_values, order) for order in orders} == {1}

    left = np.asarray([1.0, 7.0, -4.0])
    right = np.asarray([3.0, -2.0, 8.0])
    tropical = {max_plus_ordered_reduce(left, right, order) for order in orders}
    assert tropical == {5.0}

    tied = (7.0, 7.0, 3.0)
    tied_results = {
        tropical_argmax_ordered_reduce(tied, order) for order in orders
    }
    assert tied_results == {(7.0, 0)}


def test_crt_requires_coprime_moduli_and_strict_dynamic_range() -> None:
    good = crt_reduction_certificate(
        max_abs_term=100,
        fan_in=10,
        moduli=(43, 47),
    )
    assert good["modulus_product"] == 2021
    assert good["symmetric_reconstruction_injective"] is True

    too_small = crt_reduction_certificate(
        max_abs_term=100,
        fan_in=10,
        moduli=(31, 37),
    )
    assert too_small["symmetric_reconstruction_injective"] is False
    non_coprime = crt_reduction_certificate(
        max_abs_term=10,
        fan_in=2,
        moduli=(6, 9),
    )
    assert non_coprime["pairwise_coprime"] is False
    assert non_coprime["symmetric_reconstruction_injective"] is False


def test_number_system_table_refuses_compensation_as_l70_proof() -> None:
    rows = {
        row["candidate"]: row
        for row in number_system_disposition(
            max_abs_term=127,
            fan_in=10,
            crt_moduli=(43, 47),
        )
    }
    assert rows["bounded_fixed_point_integer"]["order_invariant"] is True
    assert rows["kahan_neumaier_or_naive_eft"]["order_invariant"] is False
    assert rows["posit_without_quire"]["order_invariant"] is False
    assert rows["max_plus_tropical"]["pact_disposition"].startswith("DECISION_HEAD_ONLY")


def test_interval_argmax_certificate_is_strict_and_classwise() -> None:
    logits = np.asarray(
        [
            [4.0, 3.0, 0.0],
            [1.0, 1.0, 0.0],
            [5.0, 4.5, -1.0],
        ],
        dtype=np.float64,
    )
    errors = np.asarray(
        [
            [0.4, 0.4, 0.4],
            [0.0, 0.0, 0.0],
            [0.1, 0.39, 0.1],
        ]
    )
    cert = certify_argmax_intervals(logits, errors)
    assert cert.reference_winner.tolist() == [0, 0, 0]
    assert cert.reference_margin.tolist() == [1.0, 0.0, 0.5]
    assert cert.certified.tolist() == [True, False, True]
    assert cert.certified_count == 2
    assert cert.certified_fraction == pytest.approx(2.0 / 3.0)

    boundary = certify_argmax_intervals(np.asarray([[1.0, 0.0]]), 0.5)
    assert boundary.robust_margin[0] == 0.0
    assert boundary.certified[0] == np.bool_(False)


def test_full_ordinal_concordance_is_stronger_than_top1_identity() -> None:
    reference = np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    candidate = np.asarray([[3.0, 1.0, 2.0], [4.0, 2.0, 1.0]])
    diagnostic = ordinal_top1_concordance_diagnostic(reference, candidate)
    assert diagnostic.top1_preserved.tolist() == [True, True]
    assert diagnostic.full_ordinal_concordance.tolist() == [False, True]
    assert diagnostic.pairwise_order_concordance_fraction.tolist() == [
        pytest.approx(2.0 / 3.0),
        1.0,
    ]

    hinge = winner_rival_margin_hinge(
        np.asarray([[2.0, 3.0, 0.0], [4.0, 2.0, 1.0]]),
        np.asarray([0, 0]),
        margin=0.0,
    )
    assert hinge.tolist() == [1.0, 0.0]


def test_surface_tension_metric_closure_exposes_wetting_violation() -> None:
    sigma = np.asarray(
        [
            [1.0, 0.7, 1.8],
            [0.7, 1.0, 0.8],
            [1.8, 0.8, 1.0],
        ]
    )
    certificate = multiphase_surface_tension_metric_certificate(sigma)
    assert certificate.gamma_limit_metric_admissible is False
    assert certificate.spatial_orientation_anisotropy_certified is False
    assert certificate.metric_closure[0, 2] == pytest.approx(1.5)
    assert certificate.triangle_violations == (
        {
            "left": 0,
            "middle": 1,
            "right": 2,
            "direct": 1.8,
            "via": 1.5,
            "excess": pytest.approx(0.3),
        },
    )


def test_discrete_waterfill_finds_exact_pareto_minimum() -> None:
    layers = (
        PrecisionLayer(
            "early",
            (
                PrecisionOption(bits=4, error_bound=0.40, measured_cost=1.0),
                PrecisionOption(bits=8, error_bound=0.05, measured_cost=3.0),
            ),
        ),
        PrecisionLayer(
            "head",
            (
                PrecisionOption(bits=4, error_bound=0.30, measured_cost=1.0),
                PrecisionOption(bits=8, error_bound=0.02, measured_cost=4.0),
            ),
        ),
    )
    allocation = solve_discrete_precision_waterfill(layers, error_budget=0.35)
    assert [choice.bits for choice in allocation.choices] == [8, 4]
    assert allocation.total_error_bound == pytest.approx(0.35)
    assert allocation.total_measured_cost == 4.0

    with pytest.raises(ValueError, match="no precision allocation"):
        solve_discrete_precision_waterfill(layers, error_budget=0.01)


def test_discrete_waterfill_preserves_one_ulp_certificate_advantage() -> None:
    lower = math.nextafter(0.5, 0.0)
    layers = (
        PrecisionLayer(
            "tight",
            (
                PrecisionOption(bits=4, error_bound=0.5, measured_cost=1.0),
                PrecisionOption(bits=5, error_bound=lower, measured_cost=1.0),
            ),
        ),
    )
    allocation = solve_discrete_precision_waterfill(layers, error_budget=lower)
    assert allocation.choices[0].bits == 5


def test_continuous_waterfill_hits_budget_and_respects_box() -> None:
    result = continuous_margin_waterfill(
        sensitivity_coefficients=np.asarray([8.0, 2.0]),
        cost_per_bit=np.asarray([1.0, 1.0]),
        error_budget=0.1,
        min_bits=np.asarray([1.0, 1.0]),
        max_bits=np.asarray([16.0, 16.0]),
    )
    bits = np.asarray(result["bits"])
    assert np.all(bits >= 1.0)
    assert np.all(bits <= 16.0)
    assert float(result["total_error_bound"]) == pytest.approx(0.1, rel=1e-10)
    assert bits[0] - bits[1] == pytest.approx(2.0, rel=1e-10)


def test_continuous_waterfill_extreme_valid_box_stays_certified() -> None:
    budget = math.ldexp(1.0, -500)
    result = continuous_margin_waterfill(
        sensitivity_coefficients=np.asarray([1.0]),
        cost_per_bit=np.asarray([1.0]),
        error_budget=budget,
        min_bits=np.asarray([0.0]),
        max_bits=np.asarray([1000.0]),
    )
    assert result["total_error_bound"] <= budget
    assert np.asarray(result["bits"])[0] == pytest.approx(500.0)


def test_continuous_waterfill_avoids_finite_ratio_overflow() -> None:
    result = continuous_margin_waterfill(
        sensitivity_coefficients=np.asarray([1e308]),
        cost_per_bit=np.asarray([1e-308]),
        error_budget=1.0,
        min_bits=np.asarray([0.0]),
        max_bits=np.asarray([2000.0]),
    )
    assert np.all(np.isfinite(np.asarray(result["bits"])))
    assert math.isfinite(float(result["total_error_bound"]))
    assert math.isfinite(float(result["total_cost"]))
    assert result["total_error_bound"] <= 1.0


def test_dependency_closure_charges_global_teacher_support() -> None:
    accounting = support_closure_flop_accounting(
        (
            SupportCost(
                name="local_head",
                dense_flops=20.0,
                requested_active_fraction=0.05,
                closed_active_fraction=0.25,
            ),
            SupportCost(
                name="global_se_encoder",
                dense_flops=80.0,
                requested_active_fraction=0.05,
                closed_active_fraction=1.0,
                global_dependency=True,
            ),
        )
    )
    assert accounting["naive_mask_speedup_upper_bound"] == 20.0
    assert accounting["dependency_closed_flops"] == 85.0
    assert accounting["dependency_closed_speedup_upper_bound"] == pytest.approx(100 / 85)

    with pytest.raises(ValueError, match="global dependency"):
        SupportCost(
            name="bad_global",
            dense_flops=1.0,
            requested_active_fraction=0.1,
            closed_active_fraction=0.9,
            global_dependency=True,
        )

    empty = support_closure_flop_accounting(
        (
            SupportCost(
                name="empty_local_demand",
                dense_flops=1.0,
                requested_active_fraction=0.0,
                closed_active_fraction=0.0,
            ),
        )
    )
    assert math.isinf(empty["dependency_closed_speedup_upper_bound"])


def test_cadence_requires_disjoint_areas_and_reuse_has_break_even() -> None:
    assert cadence_compute_fraction(
        disjoint_area_fractions=np.asarray([0.2, 0.1]),
        refresh_cadences=np.asarray([2, 10]),
        remainder_cadence=1,
    ) == pytest.approx(0.81)
    with pytest.raises(ValueError, match="disjoint"):
        cadence_compute_fraction(
            disjoint_area_fractions=np.asarray([0.8, 0.4]),
            refresh_cadences=np.asarray([1, 1]),
        )
    with pytest.raises(ValueError, match="positive integers"):
        cadence_compute_fraction(
            disjoint_area_fractions=np.asarray([0.2]),
            refresh_cadences=np.asarray([2]),
            remainder_cadence=1.5,
        )

    assert covariant_pair_reuse_break_even(
        exact_teacher_cost=10.0,
        warp_cost=2.0,
        refresh_cost=16.0,
    ) == 2.0
    assert math.isinf(
        covariant_pair_reuse_break_even(
            exact_teacher_cost=2.0,
            warp_cost=2.0,
            refresh_cost=1.0,
        )
    )
