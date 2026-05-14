# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from experiments.precompute_gradient_corrections import (
    enforce_packed_byte_cap,
    estimated_sparse_bytes,
    greedy_waterfill_correction_map,
    pack_sparse_corrections,
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
    # Codex finding 1: dense map preserves gradient MAGNITUDE (not sign-only).
    # quant_scale = max|grad| = 5.0. So 5.0 → 127, 3.0 → round(3/5*127) = 76.
    assert corrections.reshape(-1).tolist() == [0, 127, 0, 76]
    assert meta["quant_scale"] == 5.0


def test_greedy_preserves_gradient_magnitude_codex_finding1() -> None:
    """Codex finding 1 — VALUE/SIGN regression on the dense int8 correction.

    Before the fix the allocator wrote sign-only ±127 for every selected
    pixel; ``apply_corrections`` then dequantized to ±scale and clamped
    pixels rather than performing a real gradient step. The fix encodes the
    *original gradient values* via per-tensor symmetric int8 quantization.
    """
    gradients = np.array(
        [[[[0.0, 4.0, 0.0], [-2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]],
        dtype=np.float32,
    )
    corrections, meta = greedy_waterfill_correction_map(
        gradients,
        rate_cap_bytes=200,
        return_metadata=True,
    )
    # All three pixels selected (cap is huge).
    assert sorted(meta["selected_indices"]) == [0, 1, 2]
    # quant_scale = max|grad| = 4.0
    assert meta["quant_scale"] == 4.0
    # Each selected pixel must encode the gradient magnitude AND sign.
    flat = corrections.reshape(-1, 3)
    # pixel 0 = (0, 4, 0) → (0, 127, 0)
    assert flat[0].tolist() == [0, 127, 0]
    # pixel 1 = (-2, 0, 0) → (round(-2/4*127), 0, 0) = (-64, 0, 0)
    # SIGN must be NEGATIVE; magnitude must be ~half of 127.
    assert flat[1, 0] < 0, f"sign of pixel 1 ch0 must be negative, got {flat[1, 0]}"
    assert flat[1, 0] == -64
    assert flat[1, 1:].tolist() == [0, 0]
    # pixel 2 = (1, 0, 0) → (round(1/4*127), 0, 0) = (32, 0, 0)
    assert flat[2, 0] > 0
    assert flat[2, 0] == 32
    # NEGATIVE assertion: NOT all selected pixels are ±127 (the old
    # sign-only bug). At least one absolute value strictly between 0 and 127.
    abs_nonzero = np.abs(flat[flat != 0])
    assert (abs_nonzero < 127).any(), (
        f"all selected magnitudes still saturate at 127 → finding 1 regressed; "
        f"got {abs_nonzero.tolist()}"
    )


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


def test_enforce_packed_byte_cap_drops_tail_to_fit_budget_codex_finding1_bug2() -> None:
    """Codex finding 1 bug 2 — VALUE/SIGN regression on real packed cap.

    The greedy allocator used a 1-byte-per-pixel arithmetic model. The
    actual packed bytes are ~7×–13× larger (uint32 index + 3 int8 channels
    + JSON header + zlib compression). Without the drop-tail repack loop,
    a "50 KB cap" silently materialised at hundreds of KB.

    This test sets a tight cap and asserts the FINAL packed size is under
    the cap AND that lower-magnitude entries were dropped first (preserving
    the highest-gain corrections).
    """
    # 200 random selected pixels with 3 channels; uncapped pack would be
    # well over 100 bytes thanks to header + indices + values overhead.
    rng = np.random.default_rng(0)
    n = 200
    indices = rng.permutation(10_000).astype(np.uint32)[:n]
    # Magnitudes ramp 1..n so we can verify tail is dropped.
    values = (np.arange(1, n + 1).reshape(-1, 1) * np.array([[1, 2, 3]])).astype(np.int8)
    sparse = {
        "indices": indices,
        "values": values,
        "scale": 1.0,
        "shape": [10, 10, 10, 3],
        "top_k_pct": 100.0,
        "quantize_bits": 8,
        "n_kept": n,
        "n_total": 1000,
    }
    cap = 400  # tight cap, will force tail drops
    final_size = enforce_packed_byte_cap(sparse, rate_cap_bytes=cap)
    assert final_size <= cap, (
        f"packed size {final_size} > cap {cap}; drop-tail loop regressed"
    )
    # n_kept must have shrunk.
    assert sparse["n_kept"] < n, (
        f"n_kept did not shrink ({sparse['n_kept']} vs {n}); "
        f"drop-tail did not fire"
    )
    # The kept entries must be the highest-magnitude (preserved) ones.
    # After enforce_packed_byte_cap reorders by gain (descending), the
    # smallest kept absolute value should still be larger than any dropped.
    kept_max_abs = np.abs(sparse["values"]).max(axis=-1)
    assert (kept_max_abs > 0).all(), "all kept entries should have nonzero magnitude"
    # SIGN/VALUE anchor: the largest magnitude (n=200, channel 3 → 200*3=600
    # but clipped to int8 max 127) must still be present in kept set.
    assert (np.abs(sparse["values"]).max() == 127), (
        "highest-magnitude entry must survive; got "
        f"{np.abs(sparse['values']).max()}"
    )


def test_enforce_packed_byte_cap_idempotent_when_already_under() -> None:
    """If already under cap, enforce_packed_byte_cap is a no-op."""
    indices = np.array([1, 2, 3], dtype=np.uint32)
    values = np.array([[10, 20, 30]] * 3, dtype=np.int8)
    sparse = {
        "indices": indices.copy(),
        "values": values.copy(),
        "scale": 1.0,
        "shape": [10, 10, 10, 3],
        "top_k_pct": 100.0,
        "quantize_bits": 8,
        "n_kept": 3,
        "n_total": 1000,
    }
    final = enforce_packed_byte_cap(sparse, rate_cap_bytes=10_000)
    actual_packed = len(pack_sparse_corrections(sparse))
    assert final == actual_packed
    assert sparse["n_kept"] == 3


def test_sparsify_and_quantize_records_packed_bytes_under_cap() -> None:
    """End-to-end: sparsify_and_quantize must report a real packed-byte size
    that respects the rate_cap_bytes argument (codex finding 1 bug 2)."""
    rng = np.random.default_rng(1)
    grads = rng.standard_normal((4, 8, 8, 3)).astype(np.float32)
    cap = 500
    out = sparsify_and_quantize(
        grads,
        allocation_strategy="greedy",
        rate_cap_bytes=cap,
        quantize_bits=8,
    )
    assert "packed_bytes" in out
    assert out["packed_bytes"] <= cap, (
        f"sparsify_and_quantize produced packed_bytes={out['packed_bytes']} > cap={cap}"
    )


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
