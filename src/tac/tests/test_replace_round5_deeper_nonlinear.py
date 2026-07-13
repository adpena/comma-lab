# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.scorer_surrogate.replace_round5_deeper_nonlinear import (
    DEEP_FEATURE_COUNT,
    PairGatedMLPWeights,
    deeper_pair_block_features,
    disagreement_query_audit,
    pair_gated_logits_numpy,
    resize_bilinear_align_corners_false,
)


def test_numpy_bilinear_matches_torch_align_corners_false() -> None:
    torch = pytest.importorskip("torch")
    value = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 7.0
    observed = resize_bilinear_align_corners_false(value, 6, 8)
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(value[None]), size=(6, 8), mode="bilinear", align_corners=False
    )[0].numpy()
    assert np.allclose(observed, expected, rtol=0.0, atol=5e-7)


def test_deeper_chart_has_sealed_width_and_pair_rows() -> None:
    prefix = np.arange(32 * 8 * 12, dtype=np.float32).reshape(1, 32, 8, 12) / 1000.0
    block2 = np.arange(24 * 4 * 6, dtype=np.float32).reshape(1, 24, 4, 6) / 100.0
    block3 = np.arange(48 * 2 * 3, dtype=np.float32).reshape(1, 48, 2, 3) / 10.0
    labels = np.indices((16, 24)).sum(axis=0).astype(np.int64) % 5
    margins = np.linspace(-2.0, 2.0, 16 * 24, dtype=np.float32).reshape(16, 24)
    pair_ids = np.arange(8 * 12, dtype=np.int16).reshape(8, 12) % 20
    rows, pairs = deeper_pair_block_features(
        prefix,
        block2,
        block3,
        labels,
        margins,
        pair_ids,
        checkpoint_index=1,
        checkpoint_count=3,
        stride=2,
    )
    assert rows.shape == (24, DEEP_FEATURE_COUNT)
    assert pairs.shape == (24,)
    assert np.array_equal(pairs, pair_ids[::2, ::2].reshape(-1))


def test_pair_gated_numpy_selects_only_requested_head() -> None:
    features = np.ones((4, DEEP_FEATURE_COUNT), dtype=np.float32)
    input_weight = np.zeros((3, DEEP_FEATURE_COUNT), dtype=np.float32)
    input_bias = np.array((1.0, 2.0, 3.0), dtype=np.float32)
    output_weight = np.zeros((20, 3), dtype=np.float32)
    output_weight[2] = (1.0, 0.0, 0.0)
    output_weight[7] = (0.0, 1.0, 0.0)
    output_bias = np.arange(20, dtype=np.float32)
    weights = PairGatedMLPWeights(input_weight, input_bias, output_weight, output_bias)
    observed = pair_gated_logits_numpy(features, np.array((2, 7, 2, 7)), weights)
    assert np.array_equal(observed, np.array((3.0, 9.0, 3.0, 9.0), dtype=np.float32))


def test_disagreement_query_audit_is_seeded_and_positive_propensity() -> None:
    probabilities = np.array(
        [
            np.linspace(0.0, 1.0, 100, dtype=np.float32),
            np.linspace(0.1, 0.9, 100, dtype=np.float32),
            np.linspace(0.2, 0.8, 100, dtype=np.float32),
        ]
    )
    support = np.arange(100) >= 80
    first = disagreement_query_audit(probabilities, support, seed=455)
    second = disagreement_query_audit(probabilities, support, seed=455)
    assert first == second
    assert first["targeted_count"] == 4
    assert first["random_audit_count"] == 1
    assert first["random_audit_positive_propensity"] > 0.0

