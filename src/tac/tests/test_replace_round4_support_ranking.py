# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.scorer_surrogate.replace_round4_support_ranking import (
    BLOCK_FEATURE_COUNT,
    GLOBAL_FEATURE_COUNT,
    ORDERED_PAIR_COUNT,
    aggregate_quadratic_statistics,
    block_scores,
    calibrated_block_scores,
    calibration_diagnostics,
    deterministic_topk_mask,
    exact_support_target,
    fit_block_calibrators,
    fit_exact_quadratic,
    fit_isotonic_calibrator,
    global_scores,
    ordered_class_pair_ids,
    pair_id_to_classes,
    pairwise_rank_block_statistics,
    support_feature_matrices,
    weighted_topk_block_statistics,
    weighted_topk_statistics,
)
from tac.witness_dsl.replace_round4_support_ranking_policy import (
    RETAINED_MASS_BAR,
    ReplaceRound4SupportRankingPolicy,
)


def test_policy_seals_area_bar_ladder_and_conditional_economics() -> None:
    policy = ReplaceRound4SupportRankingPolicy()
    contract = policy.compile_measurement_contract()
    assert contract["selected_prefix_cells"] == 2311
    assert contract["realized_area_fraction"] == pytest.approx(0.047017415364583336)
    assert policy.retained_mass_bar == RETAINED_MASS_BAR == 0.47
    assert policy.conditional_composed_label_coefficient == pytest.approx(
        0.05246287035291876
    )
    assert policy.conditional_variable_cost_reduction_x == pytest.approx(19.061099655298698)
    assert contract["economics"]["wall_clock_claim"] is False
    with pytest.raises(ValueError, match="preregistered"):
        ReplaceRound4SupportRankingPolicy(calibration_bin_count=32)


def test_ordered_pair_chart_has_no_diagonal_and_round_trips() -> None:
    labels = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    logits = np.zeros((1, 5, 1, 5), dtype=np.float32)
    competitors = [4, 0, 3, 2, 1]
    for column, competitor in enumerate(competitors):
        logits[0, competitor, 0, column] = 2.0
    pair = ordered_class_pair_ids(labels, logits)
    assert pair.shape == labels.shape
    assert len(set(pair.reshape(-1).tolist())) == 5
    for source, pair_id in enumerate(pair.reshape(-1)):
        got_source, got_competitor = pair_id_to_classes(int(pair_id))
        assert (got_source, got_competitor) == (source, competitors[source])
        assert got_source != got_competitor


def test_feature_charts_make_pair_and_sensitivity_channels_explicit() -> None:
    generator = np.random.default_rng(455)
    prefix = generator.standard_normal((1, 32, 4, 5), dtype=np.float32)
    labels = generator.integers(0, 5, size=(8, 10), dtype=np.int64)
    margins = generator.standard_normal((8, 10), dtype=np.float32)
    logits = generator.standard_normal((1, 5, 8, 10), dtype=np.float32)
    pair = ordered_class_pair_ids(labels, logits)
    global_x, block_x, pair_rows = support_feature_matrices(
        prefix,
        labels,
        margins,
        pair,
        checkpoint_index=1,
        checkpoint_count=3,
        stride=1,
    )
    assert global_x.shape == (20, GLOBAL_FEATURE_COUNT)
    assert block_x.shape == (20, BLOCK_FEATURE_COUNT)
    assert pair_rows.shape == (20,)
    assert np.all(global_x[np.arange(20), 42 + pair_rows] == 1.0)

    sampled_pair = pair[::2, ::2][::2, ::2]
    sampled_global, sampled_block, sampled_rows = support_feature_matrices(
        prefix,
        labels,
        margins,
        sampled_pair,
        checkpoint_index=1,
        checkpoint_count=3,
        stride=2,
    )
    assert sampled_global.shape == (6, GLOBAL_FEATURE_COUNT)
    assert sampled_block.shape == (6, BLOCK_FEATURE_COUNT)
    assert np.array_equal(sampled_rows, sampled_pair.reshape(-1))


def test_exact_support_and_tie_break_select_exact_count() -> None:
    costate = np.zeros((1, 3, 4, 4), dtype=np.float32)
    costate[:, :, :2, :2] = 4.0
    mass, support, count = exact_support_target(costate, area_fraction=0.24)
    assert count == 1
    assert support.sum() == 1
    assert np.unravel_index(np.argmax(mass), mass.shape) == (0, 0)
    tied = deterministic_topk_mask(np.ones(4), count=2)
    assert tied.tolist() == [True, True, False, False]


def test_weighted_topk_solver_is_exact_normal_equation_optimum() -> None:
    generator = np.random.default_rng(455)
    records = []
    for _ in range(5):
        x = generator.standard_normal((80, 7))
        latent = x[:, 1] - 0.5 * x[:, 4]
        y = latent >= np.quantile(latent, 0.8)
        records.append(weighted_topk_statistics(x, y))
    fit = fit_exact_quadratic(aggregate_quadratic_statistics(records))
    assert fit.weights.shape == (7,)
    assert fit.certificate["normal_equation_optimum_certified"] is True
    assert fit.certificate["numerical_rank"] == 7
    assert np.corrcoef(global_scores(x, fit.weights), latent)[0, 1] > 0.8


def test_exact_solver_certifies_the_preregistered_retained_eigenspace() -> None:
    # A finite-accumulation rhs component in a declared numerical nullspace is
    # diagnostic, not a first-order failure of the rank-truncated MP problem.
    gram = np.diag([4.0, 0.0])
    rhs = np.array([8.0, 1.0e-7])
    from tac.scorer_surrogate.replace_round4_support_ranking import QuadraticStatistics

    fit = fit_exact_quadratic(
        QuadraticStatistics(
            gram=gram,
            rhs=rhs,
            target_square=16.0,
            row_count=2,
            state_count=1,
        )
    )
    assert fit.weights.tolist() == pytest.approx([2.0, 0.0])
    assert fit.certificate["normal_equation_gradient_inf"] == pytest.approx(1.0e-7)
    assert fit.certificate["retained_space_normal_equation_gradient_inf"] == pytest.approx(0.0)
    assert fit.certificate["discarded_space_rhs_l2"] == pytest.approx(1.0e-7)
    assert fit.certificate["normal_equation_optimum_certified"] is True


def test_pair_block_and_implicit_pairwise_statistics_are_finite() -> None:
    generator = np.random.default_rng(455)
    x = generator.standard_normal((240, 6))
    pair = np.tile(np.arange(ORDERED_PAIR_COUNT), 12)
    y = (x[:, 0] + 0.25 * pair) >= np.quantile(x[:, 0] + 0.25 * pair, 0.8)
    weighted = weighted_topk_block_statistics(x, pair, y)
    pairwise = pairwise_rank_block_statistics(x, pair, y)
    weighted_heads = np.stack([fit_exact_quadratic(row).weights for row in weighted])
    pairwise_heads = np.stack([fit_exact_quadratic(row).weights for row in pairwise])
    assert weighted_heads.shape == pairwise_heads.shape == (ORDERED_PAIR_COUNT, 6)
    assert np.isfinite(block_scores(x, pair, weighted_heads)).all()


def test_train_only_isotonic_calibration_and_block_fallback_are_explicit() -> None:
    raw = np.linspace(-3.0, 3.0, 400)
    labels = raw > 0.5
    calibrator = fit_isotonic_calibrator(raw, labels, bin_count=16)
    probability = calibrator.predict(raw)
    assert calibrator.valid
    assert np.all(np.diff(probability) >= -1e-15)
    diagnostic = calibration_diagnostics(probability, labels)
    assert diagnostic["expected_calibration_error_10bin"] < 0.2

    pair = np.zeros(raw.size, dtype=np.int64)
    pair[-1] = 1  # block 1 is invalid and must use the global fit.
    global_cal, block_cal = fit_block_calibrators(raw, labels, pair, bin_count=16)
    calibrated, fallback = calibrated_block_scores(
        raw,
        pair,
        global_calibrator=global_cal,
        block_calibrators=block_cal,
    )
    assert np.isfinite(calibrated).all()
    assert fallback[-1]
    assert not fallback[:-1].any()
