from __future__ import annotations

import numpy as np

from experiments.precompute_gradient_corrections import (
    estimated_sparse_bytes,
    greedy_waterfill_correction_map,
    sparsify_and_quantize,
)


def test_greedy_always_picks_highest_ratio_first() -> None:
    gradients = np.array([[[[0.2], [5.0], [1.0], [3.0]]]], dtype=np.float32)

    corrections, meta = greedy_waterfill_correction_map(
        gradients,
        rate_cap_bytes=102,
        return_metadata=True,
    )

    assert meta["selected_indices"] == [1, 3]
    assert corrections.reshape(-1).tolist() == [0, 127, 0, 127]


def test_greedy_stops_at_rate_cap() -> None:
    gradients = np.array([[[[4.0], [3.0], [2.0]]]], dtype=np.float32)

    corrections, meta = greedy_waterfill_correction_map(
        gradients,
        rate_cap_bytes=101,
        return_metadata=True,
    )

    assert meta["estimated_bytes"] == 101
    assert meta["selected_indices"] == [0]
    assert np.count_nonzero(corrections) == 1


def test_fixed_budget_mode_matches_v1_topk_behavior() -> None:
    gradients = np.array(
        [[[[0.1, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]],
        dtype=np.float32,
    )

    sparse = sparsify_and_quantize(
        gradients,
        top_k_pct=50.0,
        quantize_bits=8,
        allocation_strategy="fixed-budget",
    )

    assert sparse["n_kept"] == 2
    assert sparse["indices"].tolist() == [1, 3]
    assert sparse["values"].dtype == np.int8


def test_empty_candidate_set_returns_identity() -> None:
    gradients = np.zeros((1, 2, 2, 3), dtype=np.float32)

    corrections, meta = greedy_waterfill_correction_map(
        gradients,
        rate_cap_bytes=500,
        return_metadata=True,
    )

    assert np.array_equal(corrections, np.zeros_like(corrections))
    assert meta["selected_indices"] == []
    assert meta["estimated_bytes"] == 100


def test_tie_breaking_is_deterministic_by_flat_index() -> None:
    gradients = np.ones((1, 1, 4, 1), dtype=np.float32)

    _, first = greedy_waterfill_correction_map(
        gradients,
        rate_cap_bytes=102,
        return_metadata=True,
    )
    _, second = greedy_waterfill_correction_map(
        gradients,
        rate_cap_bytes=102,
        return_metadata=True,
    )

    assert first["selected_indices"] == [0, 1]
    assert second["selected_indices"] == [0, 1]


def test_greedy_uses_fewer_bytes_than_fixed_budget_for_same_gain() -> None:
    gradients = np.array(
        [[[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
        dtype=np.float32,
    )

    _, greedy_meta = greedy_waterfill_correction_map(
        gradients,
        rate_cap_bytes=101,
        return_metadata=True,
    )
    fixed = sparsify_and_quantize(
        gradients,
        top_k_pct=50.0,
        quantize_bits=8,
        allocation_strategy="fixed-budget",
    )

    assert greedy_meta["total_gain"] == 10.0
    assert np.isclose(fixed["total_gain"], greedy_meta["total_gain"])
    assert greedy_meta["estimated_bytes"] < estimated_sparse_bytes(fixed)
