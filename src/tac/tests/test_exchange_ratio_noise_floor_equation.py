"""Tests for ``tac.canonical_equations.exchange_ratio_noise_floor_20260903``.

The equation module states the law a second time, in pure form, so the registry's
EQUATIONS leg does not merely point at a script.  Two statements of one law drift.
These tests are the DRIFT GUARD: the equation's bootstrap, its calibration
constant, its score arithmetic, and its bootstrap spec must agree EXACTLY with the
producer ``experiments/ddm_xr1_exchange_ratio_noise_floor.py`` on the same inputs.

They also pin the acceptance rule -- admissible iff the 95% UPPER edge is below
zero, never the point estimate -- because that rule is the whole reason the
equation exists.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_equations.exchange_ratio_noise_floor_20260903 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EQUATION_ID,
    PAIR_COUNT,
    RATE_DENOMINATOR_BYTES,
    RATE_NUMERATOR,
    SIGMA_B_BYTES,
    bootstrap_delta_bytes,
    bootstrap_mean,
    build_exchange_ratio_noise_floor_v1,
    calibration_constant_bytes,
    delta_s_from_components,
    draw_pair_indices,
    exchange_ratio_is_defined,
    near_win_is_admissible,
    percentile_interval_95,
)

REPO = Path(__file__).resolve().parents[3]
PRODUCER_PATH = REPO / "experiments" / "ddm_xr1_exchange_ratio_noise_floor.py"


def _producer():
    spec = importlib.util.spec_from_file_location("_xr1_producer", PRODUCER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_xr1_producer"] = module
    spec.loader.exec_module(module)
    return module


xr1 = _producer()


# --------------------------------------------------------------------------
# drift guard: one law, two statements, identical results
# --------------------------------------------------------------------------


def test_bootstrap_spec_matches_the_producer_exactly():
    assert BOOTSTRAP_SEED == xr1.BOOTSTRAP_SEED
    assert BOOTSTRAP_RESAMPLES == xr1.BOOTSTRAP_RESAMPLES
    assert PAIR_COUNT == xr1.PAIR_COUNT
    assert RATE_NUMERATOR == xr1.RATE_NUMERATOR
    assert RATE_DENOMINATOR_BYTES == xr1.RATE_DENOMINATOR_BYTES


def test_draw_matrix_matches_the_producers_seeded_draw():
    """The producer and the equation must resample the SAME pairs from the same seed."""
    rng = np.random.default_rng(xr1.BOOTSTRAP_SEED)
    producer_draws = rng.integers(
        0, xr1.PAIR_COUNT, size=(xr1.BOOTSTRAP_RESAMPLES, xr1.PAIR_COUNT), dtype=np.uint16
    )
    np.testing.assert_array_equal(draw_pair_indices(), producer_draws)


def test_byte_bootstrap_matches_the_producer_on_the_same_inputs():
    rng = np.random.default_rng(11)
    base = rng.normal(1_500.0, 300.0, size=PAIR_COUNT)
    candidate = base + rng.normal(-8.0, 30.0, size=PAIR_COUNT)
    draws = draw_pair_indices(resamples=25)
    pair_delta, producer_samples, producer_fixed = xr1.exact_total_calibrated_bootstrap(
        base, candidate, draws, exact_delta_bytes=-2_950
    )
    equation_samples = bootstrap_delta_bytes(pair_delta, draws, exact_delta_bytes=-2_950)
    np.testing.assert_allclose(equation_samples, producer_samples, rtol=0, atol=1e-9)
    assert calibration_constant_bytes(pair_delta, -2_950) == pytest.approx(producer_fixed)


def test_mean_bootstrap_matches_the_producer_on_the_same_inputs():
    rng = np.random.default_rng(13)
    values = rng.uniform(1e-4, 1e-3, size=PAIR_COUNT)
    draws = draw_pair_indices(resamples=25)
    producer_samples, _ = xr1.exact_mean_calibrated_bootstrap(
        values, draws, exact_mean=0.000512
    )
    np.testing.assert_allclose(
        bootstrap_mean(values, draws, exact_mean=0.000512),
        producer_samples,
        rtol=0,
        atol=1e-18,
    )


def test_score_arithmetic_matches_the_producer_term_for_term():
    producer = xr1.score_delta(
        base_d_seg=0.0003474002587608993,
        candidate_d_seg=0.0003874630492646247,
        base_d_pose=0.0001470109127694741,
        candidate_d_pose=0.00014620431466028094,
        delta_bytes=-2_940,
    )
    equation = float(
        delta_s_from_components(
            base_d_seg=0.0003474002587608993,
            candidate_d_seg=0.0003874630492646247,
            base_d_pose=0.0001470109127694741,
            candidate_d_pose=0.00014620431466028094,
            delta_bytes=-2_940,
        )
    )
    assert equation == pytest.approx(producer["delta_s"], rel=0, abs=1e-18)


def test_interval_matches_the_producers_percentile_convention():
    rng = np.random.default_rng(17)
    values = rng.normal(0.0, 1.0, size=500)
    producer = xr1.percentile_interval(values)
    low, high = percentile_interval_95(values)
    assert low == pytest.approx(producer["low"])
    assert high == pytest.approx(producer["high"])


# --------------------------------------------------------------------------
# the acceptance rule
# --------------------------------------------------------------------------


def test_admissible_requires_the_upper_edge_below_zero_not_the_point():
    """A negative POINT with an interval reaching zero is a coin flip, not a win."""
    straddling = np.linspace(-0.004, 0.001, 400)  # mean negative, upper edge positive
    assert float(straddling.mean()) < 0.0
    assert near_win_is_admissible(straddling) is False
    clearly_negative = np.linspace(-0.004, -0.001, 400)
    assert near_win_is_admissible(clearly_negative) is True


def test_exchange_ratio_is_undefined_when_the_denominator_changes_sign():
    assert exchange_ratio_is_defined(np.linspace(0.001, 0.004, 50)) is True
    assert exchange_ratio_is_defined(np.linspace(-0.004, -0.001, 50)) is True
    assert exchange_ratio_is_defined(np.linspace(-0.002, 0.002, 50)) is False


def test_identity_draw_reproduces_the_retained_exact_total():
    pair_delta = np.full(PAIR_COUNT, -5.0)
    identity = np.arange(PAIR_COUNT, dtype=np.uint16)[None, :]
    samples = bootstrap_delta_bytes(pair_delta, identity, exact_delta_bytes=-2_998)
    assert samples[0] == pytest.approx(-2_998.0)


def test_bootstrap_refuses_a_population_that_is_not_n600():
    with pytest.raises(ValueError, match="n600"):
        bootstrap_delta_bytes(np.zeros(96), draw_pair_indices(resamples=2), exact_delta_bytes=0)
    with pytest.raises(ValueError, match="n600"):
        bootstrap_mean(np.zeros(96), draw_pair_indices(resamples=2), exact_mean=0.0)


# --------------------------------------------------------------------------
# the registered equation
# --------------------------------------------------------------------------


def test_equation_builds_with_all_three_measured_anchors():
    eq = build_exchange_ratio_noise_floor_v1()
    assert eq.equation_id == EQUATION_ID
    anchor_ids = {anchor.anchor_id for anchor in eq.empirical_anchors}
    assert anchor_ids == {
        "physical_null_reencode_x3_sigma_b_20260903",
        "jbp1_row_a_pair_bootstrap_delta_bytes_20260903",
        "fcd3_pair_bootstrap_delta_s_and_exchange_ratio_20260903",
    }


def test_equation_declares_its_producer_and_is_not_orphaned():
    eq = build_exchange_ratio_noise_floor_v1()
    assert eq.canonical_producers == ("experiments/ddm_xr1_exchange_ratio_noise_floor.py",)
    assert eq.canonical_consumers


def test_equation_callable_path_resolves_to_the_acceptance_rule():
    eq = build_exchange_ratio_noise_floor_v1()
    module_path, _, attribute = eq.python_callable_module_path.partition(":")
    module = importlib.import_module(module_path)
    assert getattr(module, attribute) is near_win_is_admissible


def test_fcd3_anchor_records_that_the_interval_excludes_zero():
    """The charter's falsifier: an interval including zero would reopen the win-win cone."""
    eq = build_exchange_ratio_noise_floor_v1()
    anchor = next(
        a for a in eq.empirical_anchors if a.anchor_id.startswith("fcd3_pair_bootstrap")
    )
    low, high = anchor.empirical_output["delta_s_interval_95"]
    assert low > 0.0 and high > 0.0
    assert anchor.empirical_output["interval_excludes_zero"] is True
    assert anchor.empirical_output["falsifier_fired"] is False
    assert anchor.empirical_output["admissible"] is False
    # the point estimate must lie inside its own interval
    assert low <= anchor.predicted_output["delta_s_point"] <= high


def test_jbp1_anchor_interval_is_negative_and_inside_the_charter_bound():
    eq = build_exchange_ratio_noise_floor_v1()
    anchor = next(a for a in eq.empirical_anchors if a.anchor_id.startswith("jbp1_row_a"))
    low, high = anchor.empirical_output["interval_95_bytes"]
    assert high < 0.0
    assert low <= anchor.inputs["exact_delta_bytes"] <= high
    assert anchor.empirical_output["half_width_bytes"] < 600.0
    assert anchor.empirical_output["prediction_held"] is True


def test_physical_anchor_states_a_zero_sigma_and_the_consequence():
    eq = build_exchange_ratio_noise_floor_v1()
    anchor = next(a for a in eq.empirical_anchors if a.anchor_id.startswith("physical_null"))
    assert anchor.empirical_output["sigma_b_bytes"] == SIGMA_B_BYTES == 0.0
    assert anchor.inputs["scorer_runs"] == 0
    assert anchor.inputs["physical_repeats"] == 3


def test_domain_of_validity_forbids_site_and_prefix_resampling():
    eq = build_exchange_ratio_noise_floor_v1()
    forbidden = " ".join(eq.domain_of_validity["forbidden_resampling"])
    assert "site-level" in forbidden
    assert "prefix" in forbidden


def test_domain_of_validity_binds_the_transfer_refusal_and_the_apparatus_label():
    eq = build_exchange_ratio_noise_floor_v1()
    assert "UNGRADABLE" in eq.domain_of_validity["transfer_rule"]
    assert "same-object" in eq.domain_of_validity["transfer_rule"]
    result_type = eq.domain_of_validity["result_type"]
    assert "APPARATUS" in result_type
    assert "NOT a d_seg" in result_type
