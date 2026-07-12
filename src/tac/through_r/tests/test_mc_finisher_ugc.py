# SPDX-License-Identifier: MIT
"""UGC / DisARM / RLOO tests for the direction-pinned #396/#400 mask finisher."""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from tac.through_r.mc_finisher import (
    DirectionPinnedMaskFinisher,
    DirectionPinnedPairLocalObjective,
    MCFinisherError,
    exact_bernoulli_estimator_moments,
    exact_bernoulli_logit_gradient,
    measure_estimator_variance,
    sample_bernoulli_gradient,
    ugc_boundary_threshold,
)


def _interaction_objective(mask: np.ndarray) -> float:
    b = np.asarray(mask, dtype=np.float64)
    return float(
        1.3 * b[0]
        - 0.7 * b[1]
        + 0.4 * b[2]
        + 2.0 * b[0] * b[2]
        - 1.1 * b[1] * b[2]
    )


def _six_bit_objective(mask: np.ndarray) -> float:
    b = np.asarray(mask, dtype=np.float64)
    linear = np.array([-6.0, -4.0, -2.0, 1.0, 3.0, 5.0])
    return float(10.0 + linear @ b + 0.75 * b[0] * b[1] + 1.5 * b[2] * b[4])


def test_ugc_threshold_and_strict_boundary_switch() -> None:
    assert ugc_boundary_threshold(5) == pytest.approx(0.1)
    p = np.array([0.099, 0.101, 0.5, 0.899, 0.901])
    est = sample_bernoulli_gradient(
        "ugc", lambda mask: float(np.asarray(mask).sum()), p, np.random.default_rng(3)
    )
    assert est.boundary_coordinates.tolist() == [True, False, False, False, True]
    with pytest.raises(MCFinisherError, match="positive"):
        ugc_boundary_threshold(0)


def test_ugc_switches_coordinatewise_and_keeps_disarm_interior() -> None:
    p = np.array([0.02, 0.5, 0.8])
    ugc = sample_bernoulli_gradient("ugc", _interaction_objective, p, np.random.default_rng(7))
    disarm = sample_bernoulli_gradient(
        "disarm", _interaction_objective, p, np.random.default_rng(7)
    )
    assert ugc.n_function_evals == 3
    assert disarm.n_function_evals == 2
    assert ugc.boundary_coordinates.tolist() == [True, False, False]
    assert np.array_equal(ugc.masks[0], disarm.masks[0])
    assert np.array_equal(ugc.masks[1], disarm.masks[1])
    assert np.array_equal(ugc.gradient[1:], disarm.gradient[1:])
    if ugc.selected_bitflip_coordinate != 0:
        assert ugc.gradient[0] == 0.0


@pytest.mark.parametrize("estimator", ["ugc", "disarm", "rloo"])
def test_unbiased_gradient_mean_matches_bruteforce(estimator: str) -> None:
    """Required UGC unbiasedness check, also guarding both comparison formulas."""

    p = np.array([0.02, 0.5, 0.8])
    exact = exact_bernoulli_logit_gradient(_interaction_objective, p)
    rng = np.random.default_rng(123)
    draws = np.stack(
        [
            sample_bernoulli_gradient(estimator, _interaction_objective, p, rng).gradient
            for _ in range(50_000)
        ]
    )
    # Interacting, mixed boundary/interior objective.  The tolerance is substantially
    # tighter than the gradient scale while remaining robust to Monte Carlo sampling.
    assert np.allclose(draws.mean(axis=0), exact, atol=6e-3, rtol=0.0)


def test_exact_gradient_matches_finite_difference_in_logits() -> None:
    phi = np.array([-1.1, 0.3, 1.7])
    p = 1.0 / (1.0 + np.exp(-phi))
    exact = exact_bernoulli_logit_gradient(_interaction_objective, p)

    def expected(logits: np.ndarray) -> float:
        probs = 1.0 / (1.0 + np.exp(-logits))
        total = 0.0
        for index in range(8):
            b = np.array([(index >> j) & 1 for j in range(3)], dtype=np.int8)
            mass = float(np.prod(np.where(b == 1, probs, 1.0 - probs)))
            total += mass * _interaction_objective(b)
        return total

    eps = 1e-5
    finite_difference = np.empty(3)
    for j in range(3):
        step = np.zeros(3)
        step[j] = eps
        finite_difference[j] = (expected(phi + step) - expected(phi - step)) / (2.0 * eps)
    assert np.allclose(exact, finite_difference, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize(
    ("estimator", "samples", "padding"),
    [("ugc", 21, 1), ("disarm", 32, 0), ("rloo", 32, 0)],
)
def test_variance_receipt_uses_exact_matched_budget(
    estimator: str, samples: int, padding: int
) -> None:
    receipt = measure_estimator_variance(
        estimator,
        _six_bit_objective,
        [0.02, 0.02, 0.02, 0.5, 0.5, 0.5],
        eval_budget=64,
        seed=4,
    )
    assert receipt.function_evals == 64
    assert receipt.n_samples == samples
    assert receipt.budget_padding_evals == padding
    assert receipt.trace_variance is not None and receipt.trace_variance >= 0.0
    assert receipt.mean_gradient is not None and receipt.mean_gradient.shape == (6,)


def test_es_variance_is_not_laundered_into_gradient_variance() -> None:
    actual_calls = 0

    def counted_objective(mask: np.ndarray) -> float:
        nonlocal actual_calls
        actual_calls += 1
        return _six_bit_objective(mask)

    receipt = measure_estimator_variance(
        "one_plus_one_es", counted_objective, np.full(6, 0.02), eval_budget=64, seed=2
    )
    assert receipt.function_evals == 64
    assert actual_calls == 64
    assert receipt.n_samples == 63
    assert receipt.trace_variance is None
    assert receipt.mean_gradient is None
    assert receipt.proposal_gain_variance is not None
    assert receipt.proposal_gain_variance >= 0.0


def test_exact_enumeration_variance_is_zero_and_budget_checked() -> None:
    receipt = measure_estimator_variance(
        "exact_enumeration",
        _six_bit_objective,
        np.full(6, 0.5),
        eval_budget=64,
        seed=0,
    )
    assert receipt.function_evals == 64
    assert receipt.trace_variance == 0.0
    assert receipt.budget_padding_evals == 0
    assert receipt.mean_gradient is not None
    with pytest.raises(MCFinisherError, match="requires at least"):
        measure_estimator_variance(
            "exact_enumeration",
            _six_bit_objective,
            np.full(6, 0.5),
            eval_budget=63,
            seed=0,
        )


def test_exact_estimator_moments_remove_short_run_sampling_noise() -> None:
    p = np.array([1.0 / 24.0] * 3 + [0.5] * 3)
    moments = exact_bernoulli_estimator_moments(_six_bit_objective, p)
    exact_gradient = exact_bernoulli_logit_gradient(_six_bit_objective, p)
    assert set(moments) == {"ugc", "disarm", "rloo"}
    for receipt in moments.values():
        assert receipt.probability_mass == pytest.approx(1.0)
        assert receipt.objective_states == 64
        assert np.allclose(receipt.mean_gradient, exact_gradient, atol=1e-12, rtol=0.0)
        assert receipt.trace_variance == pytest.approx(receipt.coordinate_variance.sum())
    assert moments["ugc"].trace_variance < moments["disarm"].trace_variance


def test_exact_estimator_moments_refuse_explosive_support() -> None:
    with pytest.raises(MCFinisherError, match="K<=10"):
        exact_bernoulli_estimator_moments(
            lambda mask: float(mask.sum()), np.full(11, 0.5)
        )


@pytest.mark.parametrize("estimator", ["ugc", "disarm", "rloo"])
def test_gradient_finishers_preserve_strict_exact_ratchet_and_budget(estimator: str) -> None:
    probabilities = np.array([0.02, 0.02, 0.02, 0.5, 0.5, 0.5])
    start = _six_bit_objective(np.zeros(6, dtype=np.int8))
    finisher = DirectionPinnedMaskFinisher(
        _six_bit_objective,
        probabilities,
        estimator=estimator,
        initial_value=start,
        seed=8,
    )
    result = finisher.run(eval_budget=64)
    accepted_values = [out.candidate_value for out in result.outcomes if out.accepted]
    trajectory = [start, *accepted_values]
    assert all(b < a for a, b in pairwise(trajectory))
    assert result.best_s <= result.start_s
    assert result.function_evals == result.eval_budget == 64


def test_exact_enumeration_finds_global_optimum_with_same_ratchet() -> None:
    start_mask = np.zeros(6, dtype=np.int8)
    finisher = DirectionPinnedMaskFinisher(
        _six_bit_objective,
        np.full(6, 0.5),
        estimator="exact_enumeration",
        initial_mask=start_mask,
        initial_value=_six_bit_objective(start_mask),
        seed=0,
    )
    result = finisher.run(eval_budget=64)
    brute = min(
        (_six_bit_objective(np.array([(i >> j) & 1 for j in range(6)])), i)
        for i in range(64)
    )
    assert result.best_s == pytest.approx(brute[0])
    assert result.function_evals == 64
    assert result.delta_s <= 0.0


def test_one_plus_one_es_is_deterministic_and_monotone() -> None:
    probabilities = np.full(6, 0.02)
    initial_value = _six_bit_objective(np.zeros(6, dtype=np.int8))
    a = DirectionPinnedMaskFinisher(
        _six_bit_objective,
        probabilities,
        estimator="one_plus_one_es",
        initial_value=initial_value,
        seed=11,
    ).run(eval_budget=64)
    b = DirectionPinnedMaskFinisher(
        _six_bit_objective,
        probabilities,
        estimator="one_plus_one_es",
        initial_value=initial_value,
        seed=11,
    ).run(eval_budget=64)
    assert np.array_equal(a.best_mask, b.best_mask)
    assert a.best_s == b.best_s
    assert [o.candidate_mask.tolist() for o in a.outcomes] == [
        o.candidate_mask.tolist() for o in b.outcomes
    ]
    assert a.best_s <= a.start_s


def test_resume_restores_rng_mask_and_eval_position(tmp_path) -> None:
    snapshot = tmp_path / "ugc_snapshot.json"
    probabilities = np.array([0.02, 0.02, 0.02, 0.5, 0.5, 0.5])
    start = _six_bit_objective(np.zeros(6, dtype=np.int8))
    uninterrupted = DirectionPinnedMaskFinisher(
        _six_bit_objective,
        probabilities,
        estimator="ugc",
        initial_value=start,
        seed=14,
    ).run(eval_budget=64)

    partial_finisher = DirectionPinnedMaskFinisher(
        _six_bit_objective,
        probabilities,
        estimator="ugc",
        initial_value=start,
        seed=14,
    )
    partial = partial_finisher.run(eval_budget=32, snapshot_path=snapshot)
    assert partial.function_evals == 32
    resumed = DirectionPinnedMaskFinisher.resume_from(snapshot, _six_bit_objective)
    completed = resumed.run(eval_budget=64, snapshot_path=snapshot)
    assert completed.function_evals == 64
    assert completed.best_s == uninterrupted.best_s
    assert np.array_equal(completed.best_mask, uninterrupted.best_mask)
    assert not snapshot.with_name(snapshot.name + ".tmp").exists()


def test_jsonl_log_carries_exact_acceptance_custody(tmp_path) -> None:
    log = tmp_path / "outcomes.jsonl"
    finisher = DirectionPinnedMaskFinisher(
        _six_bit_objective,
        np.full(6, 0.02),
        estimator="one_plus_one_es",
        initial_value=_six_bit_objective(np.zeros(6, dtype=np.int8)),
        seed=3,
    )
    result = finisher.run(eval_budget=8, log_path=log)
    rows = log.read_text().splitlines()
    assert len(rows) == len(result.outcomes) == 8
    assert all("function_evals_after" in row and "candidate_value" in row for row in rows)


def test_invalid_inputs_and_nonfinite_objective_fail_closed() -> None:
    with pytest.raises(MCFinisherError, match="probabilities"):
        DirectionPinnedMaskFinisher(_six_bit_objective, [0.2, 1.2], estimator="ugc")
    with pytest.raises(MCFinisherError, match="unknown estimator"):
        DirectionPinnedMaskFinisher(_six_bit_objective, [0.2, 0.3], estimator="fake")
    with pytest.raises(MCFinisherError, match="non-finite"):
        DirectionPinnedMaskFinisher(
            lambda _mask: float("nan"), [0.2, 0.3], estimator="disarm"
        )


def test_pair_local_objective_uses_nonlinear_pose_mean_and_real_joint_bytes() -> None:
    base_table = np.zeros((4, 2), dtype=np.uint8)

    def archive_bytes(table: np.ndarray) -> int:
        # Deliberately interacting joint byte law: two active rows cost less together.
        active = int(np.count_nonzero(table))
        return 100 + 7 * active - (5 if active == 2 else 0)

    objective = DirectionPinnedPairLocalObjective(
        base_dseg=np.array([0.1, 0.2, 0.3, 0.4]),
        base_dpose=np.array([0.01, 0.04, 0.09, 0.16]),
        base_table=base_table,
        candidate_pairs=np.array([1, 3]),
        candidate_columns=np.array([0, 1]),
        candidate_values=np.array([2, 3]),
        edited_dseg=np.array([0.05, 0.25]),
        edited_dpose=np.array([0.02, 0.08]),
        archive_bytes_fn=archive_bytes,
    )
    both = objective.components(np.array([1, 1]))
    assert both.d_seg == pytest.approx(np.mean([0.1, 0.05, 0.3, 0.25]))
    assert both.d_pose == pytest.approx(np.mean([0.01, 0.02, 0.09, 0.08]))
    assert both.archive_bytes == 109  # 100 + 14 - joint interaction 5
    assert objective.table_for_mask([1, 1])[1, 0] == 2
    assert objective.table_for_mask([1, 1])[3, 1] == 3
    # The full canonical score, including sqrt(mean pose), is the callable value.
    assert objective(np.array([1, 1])) == both.s


def test_pair_local_objective_refuses_duplicate_pairs_and_bad_bytes() -> None:
    with pytest.raises(MCFinisherError, match="distinct"):
        DirectionPinnedPairLocalObjective(
            base_dseg=np.zeros(4),
            base_dpose=np.zeros(4),
            base_table=np.zeros((4, 2), dtype=np.uint8),
            candidate_pairs=np.array([1, 1]),
            candidate_columns=np.array([0, 1]),
            candidate_values=np.array([1, 1]),
            edited_dseg=np.zeros(2),
            edited_dpose=np.zeros(2),
            archive_bytes_fn=lambda _table: 10,
        )
    objective = DirectionPinnedPairLocalObjective(
        base_dseg=np.zeros(4),
        base_dpose=np.zeros(4),
        base_table=np.zeros((4, 2), dtype=np.uint8),
        candidate_pairs=np.array([1, 2]),
        candidate_columns=np.array([0, 1]),
        candidate_values=np.array([1, 1]),
        edited_dseg=np.zeros(2),
        edited_dpose=np.zeros(2),
        archive_bytes_fn=lambda _table: -1,
    )
    with pytest.raises(MCFinisherError, match="negative"):
        objective(np.zeros(2, dtype=np.int8))
