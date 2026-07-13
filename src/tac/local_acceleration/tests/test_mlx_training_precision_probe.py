from __future__ import annotations

import numpy as np
import pytest

from tac.local_acceleration.mlx_training_precision_probe import (
    PrecisionGoBars,
    aggregate_pair_gradient_metrics,
    evaluate_precision_gate,
    gradient_metrics,
)


def test_gradient_metrics_exact_and_orthogonal() -> None:
    exact = gradient_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert exact["cosine"] == pytest.approx(1.0)
    assert exact["relative_l2"] == 0.0
    orth = gradient_metrics(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert orth["cosine"] == pytest.approx(0.0)


def test_pair_aggregate_and_gate_require_n600() -> None:
    agg = aggregate_pair_gradient_metrics(
        [{"cosine": 0.999, "relative_l2": 0.01}] * 600
    )
    assert agg["n_pairs"] == 600
    gate = evaluate_precision_gate(
        fp32_seconds=2.0,
        candidate_seconds=1.0,
        global_cosine=0.999,
        pair_cosine_min=0.999,
        quality_pairs=600,
    )
    assert gate["verdict"] == "GO"
    blocked = evaluate_precision_gate(
        fp32_seconds=2.0,
        candidate_seconds=1.0,
        global_cosine=0.999,
        pair_cosine_min=0.999,
        quality_pairs=599,
    )
    assert blocked["verdict"] == "NO_GO"


def test_gate_thresholds_are_preregistered() -> None:
    assert PrecisionGoBars().minimum_speedup == 1.5
    assert PrecisionGoBars().minimum_global_gradient_cosine == 0.99
    assert PrecisionGoBars().minimum_pair_gradient_cosine == 0.99
