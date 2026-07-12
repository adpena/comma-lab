from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.micro_batch_functional_parity_20260712 import (
    EQUATION_ID,
    build_micro_batch_functional_parity_training_admission_v1,
    per_pair_weighted_batch_mean,
    training_admission_predicate,
)


def test_per_pair_normalization_is_not_global_batch_denominator() -> None:
    numerator = np.asarray([[10.0, 0.0], [10.0, 0.0]])
    weight = np.asarray([[1.0, 0.0], [10.0, 0.0]])
    got = per_pair_weighted_batch_mean(numerator, weight, eps=1e-12)
    global_ratio = float(numerator.sum() / (weight.sum() + 1e-12))
    assert got == pytest.approx((10.0 + 1.0) / 2.0)
    assert got != pytest.approx(global_ratio)


def test_per_pair_normalization_validates_domain() -> None:
    with pytest.raises(ValueError, match="share shape"):
        per_pair_weighted_batch_mean(np.ones((2, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        per_pair_weighted_batch_mean(np.ones((1, 2)), np.asarray([[1.0, -1.0]]))
    with pytest.raises(ValueError, match="eps"):
        per_pair_weighted_batch_mean(np.ones((1, 2)), np.ones((1, 2)), eps=0.0)


def test_training_admission_is_training_only_and_requires_measured_speedup() -> None:
    base = {
        "loss_delta": 1e-6,
        "loss_tolerance": 1e-5,
        "gradient_delta": 2e-5,
        "gradient_tolerance": 1e-4,
        "measured_speedup": 1.25,
    }
    assert training_admission_predicate(**base)
    assert not training_admission_predicate(**{**base, "measured_speedup": 1.0})
    assert not training_admission_predicate(**{**base, "gradient_delta": 2e-4})
    assert not training_admission_predicate(**base, scope="score")
    assert not training_admission_predicate(**base, requests_score_authority=True)


def test_equation_preserves_drift_and_denies_score_authority() -> None:
    eq = build_micro_batch_functional_parity_training_admission_v1()
    assert eq.equation_id == EQUATION_ID
    assert "never global denominator" in eq.domain_of_validity["normalization"]
    assert eq.domain_of_validity["score_authority"].startswith("none")
    assert eq.domain_of_validity["frontier_authority"].startswith("reports/latest.md")
    assert eq.empirical_anchors[0].empirical_output["speed_receipt"].startswith("OWED")
