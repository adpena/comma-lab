# SPDX-License-Identifier: MIT
"""Deterministic unit tests for the frozen-replay convex-head formulation.

These tests use small synthetic arrays solely as implementation oracles.  They
do not constitute the mission's real-n600 empirical evidence.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from tac.scorer_surrogate.frozen_replay_convex_head import (
    FAIL_CLOSED_COSINE_BAR,
    FEATURE_NAMES,
    MIN_TEACHER_AMORTIZATION,
    FrozenReplayError,
    aggregate_sufficient_statistics,
    cache_exact_label_sufficient_statistics,
    derive_contraction_certificate,
    derive_mission_verdict,
    deterministic_replay_assignments,
    fit_cached_convex_head,
    frozen_feature_matrix,
    predict_costate,
    sampled_costate_rows,
    teacher_call_accounting,
    vector_fidelity,
)
from tac.witness_dsl.frozen_replay_convex_head_policy import FrozenReplayConvexHeadPolicy


def _feature_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 4, 6
    frame = np.arange(3 * height * width, dtype=np.float32).reshape(3, height, width)
    labels = (np.arange(height * width).reshape(height, width) % 5).astype(np.int64)
    margins = np.linspace(-1.0, 1.0, height * width, dtype=np.float32).reshape(height, width)
    return frame, labels, margins


def _ridge_fixture() -> tuple[object, np.ndarray]:
    true_weights = np.asarray(
        [
            [1.0, -0.5, 0.25],
            [0.2, 0.8, -0.3],
            [-0.7, 0.1, 0.5],
            [0.4, -0.2, 0.9],
        ],
        dtype=np.float32,
    )
    records = []
    feature_blocks = []
    for seed in (7, 8, 9):
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(32, 4)).astype(np.float32)
        targets = (
            features @ true_weights
            + np.float32(0.01) * rng.normal(size=(32, 3)).astype(np.float32)
        ).astype(np.float32)
        records.append(cache_exact_label_sufficient_statistics(features, targets))
        feature_blocks.append(features)
    return aggregate_sufficient_statistics(records), np.concatenate(feature_blocks, axis=0)


def test_n600_assignment_is_unique_balanced_and_deterministic() -> None:
    checkpoint_names = ("ce299", "muon_start726", "terminal")
    rows = deterministic_replay_assignments(
        n_pairs=600,
        checkpoint_names=checkpoint_names,
        holdout_period=5,
        seed=455,
    )

    assert len(rows) == 600
    assert {row.pair_index for row in rows} == set(range(600))
    assert Counter(row.split for row in rows) == {"train": 480, "heldout": 120}
    assert Counter((row.checkpoint_name, row.split) for row in rows) == {
        (checkpoint_name, split): count
        for checkpoint_name in checkpoint_names
        for split, count in (("train", 160), ("heldout", 40))
    }
    assert all(row.checkpoint_name == checkpoint_names[row.checkpoint_index] for row in rows)
    assert rows == deterministic_replay_assignments(
        n_pairs=600,
        checkpoint_names=checkpoint_names,
        holdout_period=5,
        seed=455,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_pairs": False}, "n_pairs"),
        ({"n_pairs": 10.5}, "n_pairs"),
        ({"checkpoint_names": ()}, "checkpoint_names"),
        ({"holdout_period": 1}, "holdout_period"),
        ({"holdout_period": 2.5}, "holdout_period"),
        ({"seed": True}, "seed"),
    ],
)
def test_replay_assignment_invalid_contracts_fail_closed(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "n_pairs": 10,
        "checkpoint_names": ("a", "b", "c"),
        "holdout_period": 5,
        "seed": 455,
    }
    values.update(kwargs)
    with pytest.raises(FrozenReplayError, match=message):
        deterministic_replay_assignments(**values)  # type: ignore[arg-type]


def test_fixed_features_and_sampled_targets_have_registered_fp32_shapes() -> None:
    frame, labels, margins = _feature_inputs()
    features = frozen_feature_matrix(
        frame,
        labels,
        margins,
        checkpoint_index=1,
        checkpoint_count=3,
        stride=2,
    )
    targets = sampled_costate_rows(np.ones((1, 3, 4, 6), dtype=np.float32), stride=2)

    assert features.shape == (6, len(FEATURE_NAMES)) == (6, 31)
    assert targets.shape == (6, 3)
    assert features.dtype == targets.dtype == np.float32
    assert features.flags.c_contiguous and targets.flags.c_contiguous
    np.testing.assert_array_equal(features[:, -3:], np.asarray([[0.0, 1.0, 0.0]] * 6))

    full_features = frozen_feature_matrix(
        frame,
        labels,
        margins,
        checkpoint_index=1,
        checkpoint_count=3,
        stride=1,
    )
    weights = np.arange(len(FEATURE_NAMES) * 3, dtype=np.float32).reshape(len(FEATURE_NAMES), 3)
    prediction = predict_costate(full_features, weights, height=4, width=6)
    assert prediction.shape == (1, 3, 4, 6)
    assert prediction.dtype == np.float32
    np.testing.assert_array_equal(
        prediction.reshape(3, -1).T,
        np.asarray(full_features @ weights, dtype=np.float32),
    )


def test_feature_target_and_prediction_invalids_fail_closed() -> None:
    frame, labels, margins = _feature_inputs()
    with pytest.raises(FrozenReplayError, match="frame_nchw"):
        frozen_feature_matrix(
            frame[:2], labels, margins, checkpoint_index=0, checkpoint_count=3, stride=1
        )
    with pytest.raises(FrozenReplayError, match="integer class ids"):
        frozen_feature_matrix(
            frame,
            labels.astype(np.float32),
            margins,
            checkpoint_index=0,
            checkpoint_count=3,
            stride=1,
        )
    with pytest.raises(FrozenReplayError, match="nonfinite"):
        bad_margins = margins.copy()
        bad_margins[0, 0] = np.nan
        frozen_feature_matrix(
            frame, labels, bad_margins, checkpoint_index=0, checkpoint_count=3, stride=1
        )
    with pytest.raises(FrozenReplayError, match="stride"):
        sampled_costate_rows(np.zeros((3, 4, 6), dtype=np.float32), stride=0)
    with pytest.raises(FrozenReplayError, match="stride"):
        sampled_costate_rows(np.zeros((3, 4, 6), dtype=np.float32), stride=1.5)  # type: ignore[arg-type]
    with pytest.raises(FrozenReplayError, match="stride"):
        frozen_feature_matrix(
            frame,
            labels,
            margins,
            checkpoint_index=0,
            checkpoint_count=3,
            stride=1.5,  # type: ignore[arg-type]
        )
    with pytest.raises(FrozenReplayError, match="shape"):
        sampled_costate_rows(np.zeros((2, 4, 6), dtype=np.float32), stride=1)
    with pytest.raises(FrozenReplayError, match="aligned shapes"):
        cache_exact_label_sufficient_statistics(
            np.zeros((4, 2), dtype=np.float32), np.zeros((3, 3), dtype=np.float32)
        )
    with pytest.raises(FrozenReplayError, match=r"height\*width"):
        predict_costate(
            np.zeros((4, 2), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            height=1,
            width=3,
        )
    with pytest.raises(FrozenReplayError, match="predicted costate contains nonfinite"):
        predict_costate(
            np.full((1, 2), np.finfo(np.float32).max, dtype=np.float32),
            np.full((2, 3), np.finfo(np.float32).max, dtype=np.float32),
            height=1,
            width=1,
        )


def test_exact_label_cache_and_aggregation_reconcile_objective_statistics() -> None:
    x0 = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y0 = np.asarray([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    x1 = np.asarray([[0.5, -1.0], [2.0, 0.25], [-0.5, 3.0]], dtype=np.float32)
    y1 = np.asarray([[0.0, 1.0, 2.0], [-1.0, 0.5, 1.0], [3.0, -2.0, 0.0]], dtype=np.float32)
    records = (
        cache_exact_label_sufficient_statistics(x0, y0),
        cache_exact_label_sufficient_statistics(x1, y1),
    )
    stats = aggregate_sufficient_statistics(records)
    all_x = np.concatenate((x0, x1), axis=0)
    all_y = np.concatenate((y0, y1), axis=0)

    assert stats.state_count == 2
    assert stats.row_count == 5
    np.testing.assert_array_equal(stats.gram, np.asarray(all_x.T @ all_x, dtype=np.float32))
    np.testing.assert_array_equal(stats.rhs, np.asarray(all_x.T @ all_y, dtype=np.float32))
    assert stats.target_square_sum == pytest.approx(
        float(np.sum(np.square(all_y.astype(np.float64)))), rel=0.0, abs=0.0
    )
    np.testing.assert_array_equal(stats.per_state_rows, np.asarray([2, 3], dtype=np.int64))
    assert all(len(record.feature_sha256) == 64 for record in records)
    assert all(len(record.target_sha256) == 64 for record in records)

    incompatible = cache_exact_label_sufficient_statistics(
        np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)
    )
    with pytest.raises(FrozenReplayError, match="different feature charts"):
        aggregate_sufficient_statistics((records[0], incompatible))


def test_spectral_scale_ridge_certificate_proves_one_third_contraction() -> None:
    stats, _ = _ridge_fixture()
    hessian, rhs_mean, certificate = derive_contraction_certificate(stats)
    gram_mean = np.asarray(stats.gram / np.float32(stats.row_count), dtype=np.float32)
    data_eigenvalues = np.linalg.eigvalsh(gram_mean.astype(np.float64))
    data_lmin, data_lmax = float(data_eigenvalues[0]), float(data_eigenvalues[-1])
    expected_ridge = np.float32(data_lmax)
    if float(expected_ridge) < data_lmax:
        expected_ridge = np.nextafter(expected_ridge, np.float32(np.inf))

    assert certificate.norm == "Euclidean parameter norm / Frobenius head norm"
    assert certificate.data_curvature_min == pytest.approx(data_lmin, rel=0.0, abs=0.0)
    assert certificate.data_curvature_max == pytest.approx(data_lmax, rel=0.0, abs=0.0)
    assert certificate.ridge_lambda == float(expected_ridge)
    assert certificate.ridge_lambda >= certificate.data_curvature_max
    np.testing.assert_array_equal(
        hessian,
        np.asarray(
            gram_mean
            + np.float32(certificate.ridge_lambda)
            * np.eye(gram_mean.shape[0], dtype=np.float32),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        rhs_mean, np.asarray(stats.rhs / np.float32(stats.row_count), dtype=np.float32)
    )
    assert certificate.mu > 0.0
    assert certificate.smoothness_L >= certificate.mu
    assert certificate.ideal_step_size_eta == pytest.approx(
        2.0 / (certificate.mu + certificate.smoothness_L), rel=0.0, abs=0.0
    )
    assert certificate.step_size_eta == float(np.float32(certificate.ideal_step_size_eta))
    assert certificate.ideal_contraction_gamma <= certificate.ideal_gamma_upper_bound + 1e-7
    assert certificate.ideal_gamma_upper_bound == pytest.approx(1.0 / 3.0)
    assert certificate.fp32_step_rounding_slack == pytest.approx(
        max(0.0, certificate.contraction_gamma - certificate.ideal_contraction_gamma),
        rel=0.0,
        abs=0.0,
    )

    executed_eta = float(np.float32(certificate.step_size_eta))
    iteration = np.eye(hessian.shape[0]) - executed_eta * hessian.astype(np.float64)
    observed_operator_norm = float(np.linalg.norm(iteration, ord=2))
    assert observed_operator_norm == pytest.approx(certificate.contraction_gamma, abs=1e-12)
    assert 0.0 <= observed_operator_norm < 1.0


def test_fit_obeys_contraction_above_fp32_floor_and_residual_bounds() -> None:
    stats, _ = _ridge_fixture()
    fit = fit_cached_convex_head(stats, epochs=8)
    numeric_floor = 128.0 * np.finfo(np.float32).eps
    observed_ratios = [
        float(row["parameter_contraction_ratio"])
        for row in fit.trace
        if row["parameter_contraction_ratio"] is not None
    ]

    assert observed_ratios
    assert all(0.0 <= ratio <= fit.certificate.contraction_gamma + 1e-3 for ratio in observed_ratios)
    for index, row in enumerate(fit.trace[1:], start=1):
        previous_error = float(fit.trace[index - 1]["parameter_error_norm"])
        if previous_error <= numeric_floor:
            assert row["parameter_contraction_ratio"] is None
        else:
            assert row["parameter_contraction_ratio"] is not None

    weight_error = fit.weights.astype(np.float64) - fit.optimum_weights
    actual_parameter_error = float(np.linalg.norm(weight_error))
    gram_mean = np.asarray(
        stats.gram / np.float32(stats.row_count), dtype=np.float32
    ).astype(np.float64)
    actual_prediction_rmse = float(
        np.sqrt(max(0.0, float(np.sum(weight_error * (gram_mean @ weight_error)))))
    )
    hessian, _, _ = derive_contraction_certificate(stats)
    actual_objective_gap = max(
        0.0,
        0.5 * float(np.sum(weight_error * (hessian.astype(np.float64) @ weight_error))),
    )
    assert fit.actual_parameter_residual == pytest.approx(actual_parameter_error, abs=1e-15)
    assert fit.actual_prediction_rmse_residual == pytest.approx(actual_prediction_rmse, abs=1e-15)
    assert fit.actual_objective_gap == pytest.approx(actual_objective_gap, abs=1e-15)
    assert fit.actual_parameter_residual <= fit.residual_parameter_bound + 1e-12
    assert fit.actual_prediction_rmse_residual <= fit.residual_prediction_rmse_bound + 1e-12
    assert fit.actual_objective_gap <= fit.objective_gap_bound + 1e-12
    assert fit.residual_bounds_validated is True
    assert float(fit.trace[-1]["objective_gap"]) == pytest.approx(fit.actual_objective_gap, abs=1e-15)
    assert fit.terminal_gradient_norm >= 0.0
    assert fit.per_state_gradient_variance >= 0.0
    assert fit.per_state_gradient_second_moment >= fit.per_state_gradient_variance

    with pytest.raises(FrozenReplayError, match="epochs"):
        fit_cached_convex_head(stats, epochs=1.5)  # type: ignore[arg-type]


def test_fit_records_scale_relative_contraction_below_absolute_fp32_epsilon() -> None:
    """A tiny valid head must not lose its contraction witness to an absolute floor."""

    rng = np.random.default_rng(462)
    features = rng.normal(size=(64, 4)).astype(np.float32)
    tiny_weights = np.float32(1e-8) * np.asarray(
        [
            [1.0, -0.5, 0.25],
            [0.2, 0.8, -0.3],
            [-0.7, 0.1, 0.5],
            [0.4, -0.2, 0.9],
        ],
        dtype=np.float32,
    )
    targets = np.asarray(features @ tiny_weights, dtype=np.float32)
    stats = aggregate_sufficient_statistics(
        (cache_exact_label_sufficient_statistics(features, targets),)
    )

    fit = fit_cached_convex_head(stats, epochs=3)
    initial_distance = float(np.linalg.norm(fit.optimum_weights.astype(np.float64)))
    first_distance = float(fit.trace[0]["parameter_error_norm"])
    raw_first_ratio = first_distance / initial_distance
    absolute_fp32_floor = 128.0 * np.finfo(np.float32).eps
    scale_relative_floor = absolute_fp32_floor * initial_distance

    assert scale_relative_floor < initial_distance < absolute_fp32_floor
    assert 0.0 <= raw_first_ratio < 1.0
    assert fit.trace[0]["parameter_contraction_ratio"] == pytest.approx(raw_first_ratio)


def test_teacher_call_law_generic_cache_example_measures_sixteen_x() -> None:
    accounting = teacher_call_accounting(
        naive_teacher_calls=9_600,
        fresh_anchor_samples=600,
        paired_difference_samples=0,
        exact_labels_per_difference=2,
        observed_teacher_forwards=600,
    )

    assert accounting["law"] == "C_teacher = A + c_label * D"
    assert accounting["derived_C_teacher"] == accounting["observed_teacher_forwards"] == 600
    assert accounting["teacher_calls_per_effective_training_step"] == pytest.approx(1.0 / 16.0)
    assert accounting["teacher_call_amortization_x"] == pytest.approx(16.0)
    assert accounting["saving_calls"] == 9_000
    assert accounting["reconciliation"] == "PASS"

    with pytest.raises(FrozenReplayError, match="do not reconcile"):
        teacher_call_accounting(
            naive_teacher_calls=9_600,
            fresh_anchor_samples=600,
            paired_difference_samples=0,
            exact_labels_per_difference=2,
            observed_teacher_forwards=599,
        )


def test_dsl_seals_per_state_teacher_batch_and_effective_state_steps() -> None:
    policy = FrozenReplayConvexHeadPolicy()
    contract = policy.compile_measurement_contract()

    assert policy.fit_epochs == 15
    assert policy.teacher_batch_size == 1
    assert policy.effective_training_state_steps == 7_200
    assert contract["fit_epochs"] == 15
    assert contract["teacher_batch_size"] == 1
    assert contract["effective_training_state_steps"] == 7_200
    assert str(contract["constant_provenance"]["fit_epochs"]).startswith("DERIVED")

    accounting = teacher_call_accounting(
        naive_teacher_calls=policy.effective_training_state_steps,
        fresh_anchor_samples=600,
        paired_difference_samples=0,
        exact_labels_per_difference=2,
        observed_teacher_forwards=600,
    )
    assert accounting["teacher_calls_per_effective_training_step"] == pytest.approx(1.0 / 12.0)
    assert accounting["teacher_call_amortization_x"] == pytest.approx(12.0)
    with pytest.raises(ValueError, match="mean cross-entropy"):
        FrozenReplayConvexHeadPolicy(teacher_batch_size=4)


def test_verdict_is_fail_closed_on_either_gate_and_invalid_metrics() -> None:
    go = derive_mission_verdict(
        heldout_costate_cosine=FAIL_CLOSED_COSINE_BAR,
        teacher_call_amortization_x=MIN_TEACHER_AMORTIZATION,
    )
    assert go["verdict"] == "GO"
    assert go["cosine_gate_pass"] and go["amortization_gate_pass"]
    assert go["score_claim"] is False and go["promotion_eligible"] is False
    assert "FORMULATION x INSTANCE" in go["verdict_scope"]

    bad_cosine = derive_mission_verdict(
        heldout_costate_cosine=FAIL_CLOSED_COSINE_BAR - 1e-6,
        teacher_call_amortization_x=MIN_TEACHER_AMORTIZATION,
    )
    bad_calls = derive_mission_verdict(
        heldout_costate_cosine=FAIL_CLOSED_COSINE_BAR,
        teacher_call_amortization_x=MIN_TEACHER_AMORTIZATION - 1e-6,
    )
    assert bad_cosine["verdict"] == "NO-GO" and not bad_cosine["cosine_gate_pass"]
    assert bad_calls["verdict"] == "NO-GO" and not bad_calls["amortization_gate_pass"]
    with pytest.raises(FrozenReplayError, match="finite"):
        derive_mission_verdict(
            heldout_costate_cosine=float("nan"),
            teacher_call_amortization_x=MIN_TEACHER_AMORTIZATION,
        )
    with pytest.raises(FrozenReplayError, match="finite"):
        derive_mission_verdict(
            heldout_costate_cosine=FAIL_CLOSED_COSINE_BAR,
            teacher_call_amortization_x=float("inf"),
        )


def test_vector_fidelity_reports_dot_cosine_relative_error_and_invalids() -> None:
    reference = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    same = vector_fidelity(reference, reference.copy())
    opposite = vector_fidelity(reference, -reference)
    zero = vector_fidelity(reference, np.zeros_like(reference))

    assert same["compared_elements"] == 3
    assert same["cosine_similarity"] == pytest.approx(1.0)
    assert same["relative_l2_error"] == pytest.approx(0.0)
    assert same["dot"] == pytest.approx(float(np.dot(reference, reference)))
    assert opposite["cosine_similarity"] == pytest.approx(-1.0)
    assert opposite["relative_l2_error"] == pytest.approx(2.0)
    assert zero["cosine_similarity"] is None
    assert zero["relative_l2_error"] == pytest.approx(1.0)
    with pytest.raises(FrozenReplayError, match="different shapes"):
        vector_fidelity(np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32))
    with pytest.raises(FrozenReplayError, match="nonfinite"):
        vector_fidelity(np.asarray([np.nan], dtype=np.float32), np.zeros(1, dtype=np.float32))
