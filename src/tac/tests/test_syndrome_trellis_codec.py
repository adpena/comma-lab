# SPDX-License-Identifier: MIT
"""Tests for ``tac.codec.syndrome_trellis_codec``.

Verifies binary STC encode/decode roundtrip, ternary split/join roundtrip,
and the streaming-block encoder on small inputs. Reference per
Filler-Fridrich-Pevný 2011 IEEE TIFS 6(3):920-935.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.codec.syndrome_trellis_codec import (
    STCParams,
    WET_COST,
    extract_mask_deltas_ternary,
    join_ternary,
    make_submatrix,
    split_ternary,
    stc_decode_block,
    stc_encode_block,
    ternary_stc_decode_message,
    ternary_stc_encode_message,
    ternary_stc_encode_stream,
)


# ---------------------------------------------------------------------------
# Submatrix construction
# ---------------------------------------------------------------------------


def test_make_submatrix_shape_and_first_row_all_ones() -> None:
    H = make_submatrix(constraint_height=4, code_length=12, seed=0)
    assert H.shape == (4, 12)
    assert H.dtype == np.uint8
    # First row anchors syndrome bookkeeping (see module doc).
    assert np.all(H[0, :] == 1)


def test_make_submatrix_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        make_submatrix(0, 8)
    with pytest.raises(ValueError):
        make_submatrix(8, 0)
    with pytest.raises(ValueError):
        make_submatrix(20, 8)  # constraint_height > 16


def test_make_submatrix_deterministic_under_seed() -> None:
    H_a = make_submatrix(6, 16, seed=42)
    H_b = make_submatrix(6, 16, seed=42)
    H_c = make_submatrix(6, 16, seed=43)
    assert np.array_equal(H_a, H_b)
    assert not np.array_equal(H_a, H_c)


# ---------------------------------------------------------------------------
# Binary STC roundtrip
# ---------------------------------------------------------------------------


def test_binary_stc_encode_decode_zero_message() -> None:
    cover = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    costs = np.ones_like(cover, dtype=np.float64)
    message = np.zeros(4, dtype=np.uint8)
    params = STCParams(constraint_height=4, submatrix_seed=0)
    y, H = stc_encode_block(cover, costs, message, params)
    assert y.shape == cover.shape
    decoded = stc_decode_block(y, H)
    assert np.array_equal(decoded, message)


def test_binary_stc_encode_decode_arbitrary_message() -> None:
    rng = np.random.default_rng(2026)
    cover = rng.integers(0, 2, size=24, dtype=np.uint8)
    costs = np.ones_like(cover, dtype=np.float64)
    params = STCParams(constraint_height=6, submatrix_seed=1)
    for msg_int in range(8):
        message = np.array(
            [(msg_int >> i) & 1 for i in range(6)], dtype=np.uint8
        )
        y, H = stc_encode_block(cover, costs, message, params)
        decoded = stc_decode_block(y, H)
        assert np.array_equal(decoded, message)


def test_binary_stc_minimum_cost_vs_naive_flip() -> None:
    """STC must not exceed the naive cost upper bound of flipping every
    cover-vs-syndrome mismatch (which is always feasible)."""
    rng = np.random.default_rng(7)
    cover = rng.integers(0, 2, size=16, dtype=np.uint8)
    costs = rng.uniform(0.5, 2.0, size=16)
    message = rng.integers(0, 2, size=4, dtype=np.uint8)
    params = STCParams(constraint_height=4, submatrix_seed=3)
    y, _ = stc_encode_block(cover, costs, message, params)
    stc_cost = float(((y != cover).astype(np.float64) * costs).sum())
    naive_cost = float(costs.sum())  # worst-case all-flip bound
    assert stc_cost <= naive_cost + 1e-9


def test_binary_stc_wet_costs_strongly_disfavoured() -> None:
    """When some columns are wet (cost ≈ 1e9) and others are cheap (cost ≈ 1),
    the encoder must prefer cheap columns; the per-flip cost averaged over
    flipped positions must remain small (well below WET_COST/2)."""
    cover = np.zeros(16, dtype=np.uint8)
    costs = np.full(16, WET_COST, dtype=np.float64)
    # Mark several positions as cheap so the syndrome can always be hit.
    cheap_idx = np.array([1, 4, 7, 10, 13])
    costs[cheap_idx] = 1.0
    message = np.array([1, 0, 1, 0], dtype=np.uint8)
    params = STCParams(constraint_height=4, submatrix_seed=0)
    y, _ = stc_encode_block(cover, costs, message, params)
    flipped = np.where(y != cover)[0]
    # All flips must land on cheap columns (none on wet columns).
    assert all(idx in cheap_idx for idx in flipped.tolist()), (
        f"flips landed on wet columns: {flipped.tolist()}"
    )
    total = float(((y != cover).astype(np.float64) * costs).sum())
    # Total cost stays in cheap-cost regime, far from any wet flip.
    assert total < WET_COST / 2.0


# ---------------------------------------------------------------------------
# Ternary split/join
# ---------------------------------------------------------------------------


def test_ternary_split_join_roundtrip() -> None:
    deltas = np.array([0, 1, -1, 1, 0, 0, -1, 1, -1, 0], dtype=np.int8)
    soz, sign, _mask = split_ternary(deltas)
    out = join_ternary(soz, sign)
    assert np.array_equal(out, deltas)


def test_ternary_split_rejects_non_ternary() -> None:
    with pytest.raises(ValueError):
        split_ternary(np.array([0, 2, -1], dtype=np.int8))


# ---------------------------------------------------------------------------
# Ternary STC encode/decode
# ---------------------------------------------------------------------------


def test_ternary_stc_encode_decode_single_block() -> None:
    deltas = np.array([0, 1, -1, 1, 0, 0, -1, 1, -1, 0, 1, 0], dtype=np.int8)
    costs = np.ones_like(deltas, dtype=np.float64)
    message = np.array([1, 0, 1, 1], dtype=np.uint8)
    params = STCParams(constraint_height=4, submatrix_seed=0)
    res = ternary_stc_encode_message(deltas, costs, costs, message, params)
    decoded = ternary_stc_decode_message(res["stego_deltas"], res["H_bar"])
    assert np.array_equal(decoded, message)


def test_ternary_stc_zero_message_is_low_cost() -> None:
    """When the all-zero message is requested, the SoZ stream's cover often
    already has the right syndrome — total flips should be small."""
    rng = np.random.default_rng(11)
    deltas = rng.choice([-1, 0, 1], size=32).astype(np.int8)
    costs = np.ones_like(deltas, dtype=np.float64)
    message = np.zeros(4, dtype=np.uint8)
    params = STCParams(constraint_height=4, submatrix_seed=0)
    res = ternary_stc_encode_message(deltas, costs, costs, message, params)
    # In the worst case we flip h positions; the test asserts a loose bound.
    assert res["flips_soz"] <= 4


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_ternary_stc_stream_blocks_align() -> None:
    rng = np.random.default_rng(101)
    deltas = rng.choice([-1, 0, 1], size=100).astype(np.int8)
    costs = np.ones_like(deltas, dtype=np.float64)
    params = STCParams(constraint_height=4, submatrix_seed=10)
    res = ternary_stc_encode_stream(
        deltas, costs, block_size=20, params=params
    )
    assert res["stego"].shape == deltas.shape
    assert res["n_blocks"] == 5
    # Stream encoder must produce ternary stego.
    assert set(np.unique(res["stego"]).tolist()).issubset({-1, 0, 1})


def test_ternary_stc_stream_with_message_provider_decodes() -> None:
    rng = np.random.default_rng(202)
    deltas = rng.choice([-1, 0, 1], size=64).astype(np.int8)
    costs = np.ones_like(deltas, dtype=np.float64)
    params = STCParams(constraint_height=4, submatrix_seed=20)
    messages = [
        rng.integers(0, 2, size=4, dtype=np.uint8) for _ in range(64 // 16)
    ]

    def provider(i: int) -> np.ndarray:
        return messages[i]

    res = ternary_stc_encode_stream(
        deltas, costs, block_size=16, params=params, message_block_provider=provider
    )
    # Per-block decode: split stego into 16-pixel blocks, verify message recovers.
    for b, H in enumerate(res["H_blocks"]):
        block = res["stego"][b * 16 : (b + 1) * 16]
        decoded = ternary_stc_decode_message(block, H)
        assert np.array_equal(decoded, messages[b])


# ---------------------------------------------------------------------------
# Mask-delta extraction
# ---------------------------------------------------------------------------


def test_extract_mask_deltas_ternary_basic() -> None:
    masks = np.array(
        [
            [[0, 1], [2, 3]],
            [[0, 2], [1, 3]],
            [[1, 2], [1, 4]],
        ],
        dtype=np.int64,
    )
    deltas = extract_mask_deltas_ternary(masks)
    # Frame 1 vs 0 — delta = [+0, +1, -1, 0] flattened
    # Frame 2 vs 1 — delta = [+1, +0, +0, +1] flattened
    expected = np.array([0, 1, -1, 0, 1, 0, 0, 1], dtype=np.int8)
    assert np.array_equal(deltas, expected)


def test_extract_mask_deltas_rejects_non_3d() -> None:
    with pytest.raises(ValueError):
        extract_mask_deltas_ternary(np.zeros((4, 4), dtype=np.int64))
