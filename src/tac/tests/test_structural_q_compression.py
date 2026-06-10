# SPDX-License-Identifier: MIT
"""Behavior tests for ``tac.optimization.structural_q_compression`` (#71).

These tests enforce the NO-FAKE contract: the transforms REALLY transform the
weights/codes (a no-op is caught), the byte encoder REALLY round-trips the codec
byte maps, factorization reduces rank + reconstructs within tolerance, and
score-aware pruning beats random pruning on a planted-importance fixture. The
exact-scorer Q*-membership measurement lives in the driver (heavy render deps);
here we test the byte-cost surface + transform fidelity that the driver consumes.
"""
from __future__ import annotations

import brotli
import numpy as np
import pytest

from tac.optimization import structural_q_compression as SQ


# ---------------------------------------------------------------------------
# byte-map encode/decode round-trip (the encoder is the inverse of codec.py)
# ---------------------------------------------------------------------------
def _decode_mapped_u8(arr_u8, byte_map):
    """Reference decode (copied from codec.py) to assert encode is its inverse."""
    arr = arr_u8.astype(np.int32)
    if byte_map == "zig":
        return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)
    if byte_map == "negzig":
        zz = np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)
        return (-zz.astype(np.int16)).astype(np.int8)
    if byte_map == "off":
        return (arr.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return arr_u8.view(np.int8)
    raise ValueError(byte_map)


@pytest.mark.parametrize("byte_map", ["zig", "negzig", "off", "twos"])
def test_byte_map_encode_is_exact_inverse_of_codec_decode(byte_map):
    codes = np.arange(-127, 128, dtype=np.int8)
    u8 = SQ.encode_mapped_i8(codes, byte_map)
    back = _decode_mapped_u8(u8, byte_map)
    assert np.array_equal(back, codes), f"byte_map {byte_map} not a clean inverse"


def test_zigzag_encode_decode_roundtrip():
    codes = np.array([-5, -1, 0, 1, 5, 127, -127], dtype=np.int8)
    u8 = SQ.zigzag_encode_i8(codes)
    dec = np.where(u8.astype(np.int32) % 2 == 0, u8.astype(np.int32) // 2,
                   -(u8.astype(np.int32) // 2) - 1).astype(np.int8)
    assert np.array_equal(dec, codes)


# ---------------------------------------------------------------------------
# int8 code recovery: dequantized W == codes * scale (exactly when W is int8*scale)
# ---------------------------------------------------------------------------
def test_recover_tensor_code_roundtrips_exact_int8_scale():
    rng = np.random.default_rng(0)
    scale = float(np.float16(0.0047454833984375))
    codes_true = rng.integers(-100, 101, size=(8, 4, 3, 3)).astype(np.int8)
    weight = codes_true.astype(np.float64) * scale
    tc = SQ.recover_tensor_code("blocks.0.weight", 2, weight)
    deq = tc.codes_i8.astype(np.float64) * tc.scale
    assert np.max(np.abs(deq - weight)) < 1e-6
    assert np.array_equal(tc.codes_i8, codes_true)


def test_recover_tensor_code_handles_all_zero():
    weight = np.zeros((4, 4), dtype=np.float64)
    tc = SQ.recover_tensor_code("z", 0, weight)
    assert tc.scale == 0.0
    assert np.all(tc.codes_i8 == 0)


def test_recover_tensor_code_half_step_scale():
    # stem/rgb case: smallest nonzero magnitude == 2 * step.
    step = float(np.float16(0.012))
    codes_true = np.array([[0, 2, -2, 4], [-4, 6, 0, -8]], dtype=np.int8)
    weight = codes_true.astype(np.float64) * step
    tc = SQ.recover_tensor_code("stem.weight", 27, weight)
    deq = tc.codes_i8.astype(np.float64) * tc.scale
    assert np.max(np.abs(deq - weight)) < 1e-5


# ---------------------------------------------------------------------------
# byte encoder: replays the 7-stream grammar; bytes are the real brotli size
# ---------------------------------------------------------------------------
def _synthetic_codes(rng):
    """Build a full idx->TensorCode set with the canonical probe shapes.

    Shapes by idx are the EXACT ``HNeRVDecoder().state_dict()`` order (the order
    ``codec.decode_decoder_compact`` iterates). The 4D-weight indices are exactly
    the ``CONV4_STORAGE_PERMS`` keys; everything else is 1D bias or the 2D stem.
    """
    shapes = {
        0: (1728, 28), 1: (1728,), 2: (144, 36, 3, 3), 3: (144,),
        4: (144, 36, 3, 3), 5: (144,), 6: (108, 36, 3, 3), 7: (108,),
        8: (80, 27, 3, 3), 9: (80,), 10: (72, 20, 3, 3), 11: (72,),
        12: (72, 18, 3, 3), 13: (72,), 14: (27, 36, 1, 1), 15: (27,),
        16: (20, 27, 1, 1), 17: (20,), 18: (18, 20, 1, 1), 19: (18,),
        20: (9, 18, 3, 3), 21: (9,), 22: (18, 9, 3, 3), 23: (18,),
        24: (3, 18, 3, 3), 25: (3,), 26: (3, 18, 3, 3), 27: (3,),
    }
    codes = {}
    for idx, sh in shapes.items():
        c = rng.integers(-50, 51, size=sh).astype(np.int8)
        codes[idx] = SQ.TensorCode(f"t{idx}", idx, sh, c, 0.005)
    return codes


def test_encode_decoder_blob_bytes_is_positive_and_deterministic():
    rng = np.random.default_rng(1)
    codes = _synthetic_codes(rng)
    b1 = SQ.encode_decoder_blob_bytes(codes)
    b2 = SQ.encode_decoder_blob_bytes(codes)
    assert b1 == b2 and b1 > 0


def test_encode_blob_shrinks_when_codes_become_zeros():
    rng = np.random.default_rng(2)
    codes = _synthetic_codes(rng)
    dense_bytes = SQ.encode_decoder_blob_bytes(codes)
    # zero the two big block weights -> brotli must compress the zeros far smaller.
    zeroed = dict(codes)
    for idx in (2, 4):  # blocks.0 / blocks.1 (big 4D weights)
        tc = codes[idx]
        zeroed[idx] = SQ.TensorCode(tc.name, idx, tc.shape,
                                    np.zeros_like(tc.codes_i8), tc.scale)
    sparse_bytes = SQ.encode_decoder_blob_bytes(zeroed)
    assert sparse_bytes < dense_bytes, "zeroing dense codes must reduce brotli bytes"


def test_encode_blob_missing_index_raises():
    rng = np.random.default_rng(3)
    codes = _synthetic_codes(rng)
    del codes[2]  # drop a big 4D weight in the storage order
    with pytest.raises(ValueError):
        SQ.encode_decoder_blob_bytes(codes)


# ---------------------------------------------------------------------------
# low-rank: reduces effective rank, reconstructs within tolerance, NOT a no-op
# ---------------------------------------------------------------------------
def test_low_rank_truncate_reduces_rank():
    rng = np.random.default_rng(4)
    # construct a genuinely rank-3 matrix, add tiny noise so full rank > 3.
    base = rng.standard_normal((20, 3)) @ rng.standard_normal((3, 15))
    w = base + 1e-3 * rng.standard_normal((20, 15))
    wr = SQ.low_rank_truncate_weight(w, 3)
    assert np.linalg.matrix_rank(wr, tol=1e-6) <= 3
    # reconstruction is close because the matrix is essentially rank-3.
    assert np.linalg.norm(wr - w) / np.linalg.norm(w) < 0.05


def test_low_rank_full_rank_is_near_identity():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((10, 8)).astype(np.float32)
    wr = SQ.low_rank_truncate_weight(w, 8)
    assert np.allclose(wr, w, atol=1e-4)


def test_low_rank_rank1_is_not_a_noop_on_dense_matrix():
    rng = np.random.default_rng(6)
    w = rng.standard_normal((12, 9)).astype(np.float32)  # full rank, dense
    wr = SQ.low_rank_truncate_weight(w, 1)
    # rank-1 of a dense full-rank matrix must DIFFER materially (catches no-op).
    assert np.linalg.norm(wr - w) / np.linalg.norm(w) > 0.1
    assert np.linalg.matrix_rank(wr, tol=1e-6) == 1


def test_low_rank_4d_conv_preserves_shape():
    rng = np.random.default_rng(7)
    w = rng.standard_normal((16, 8, 3, 3)).astype(np.float32)
    wr = SQ.low_rank_truncate_weight(w, 5)
    assert wr.shape == w.shape


def test_factored_param_count_break_even():
    # dense (144, 324) ; factored cheaper iff rank < 144*324/(144+324) ~= 99.7
    dense, fact_low = SQ.factored_param_count((144, 36, 3, 3), 50)
    _, fact_high = SQ.factored_param_count((144, 36, 3, 3), 120)
    assert fact_low < dense  # rank 50 below break-even -> saves
    assert fact_high > dense  # rank 120 above break-even -> costs


# ---------------------------------------------------------------------------
# score-aware pruning: beats random + magnitude on a planted-importance fixture
# ---------------------------------------------------------------------------
def test_score_aware_prune_keeps_high_sensitivity_entries():
    codes = np.array([10, -20, 3, -1, 7, -50], dtype=np.int8)
    sens = np.array([0.1, 0.9, 0.2, 0.05, 0.3, 0.95], dtype=np.float64)
    pruned = SQ.score_aware_prune_codes(codes, sens, keep_fraction=0.5)
    # the 3 highest-sensitivity entries (idx 1, 4, 5) survive; the rest are zero.
    assert pruned[1] == -20 and pruned[5] == -50 and pruned[4] == 7
    assert pruned[0] == 0 and pruned[3] == 0


def test_score_aware_prune_beats_random_on_planted_importance():
    rng = np.random.default_rng(8)
    n = 400
    codes = rng.integers(-60, 61, size=n).astype(np.int8)
    # planted: the first 100 weights carry all the "score" mass.
    sens = np.zeros(n)
    sens[:100] = np.abs(codes[:100]) + 1.0
    keep = 0.25  # keep ~100
    aware = SQ.score_aware_prune_codes(codes, sens, keep)
    # random keep: pick 100 random indices.
    random_keep = np.zeros_like(codes)
    idx = rng.choice(n, 100, replace=False)
    random_keep[idx] = codes[idx]
    # importance retained = sum of sensitivity of surviving entries.
    aware_retained = sens[aware != 0].sum()
    random_retained = sens[random_keep != 0].sum()
    assert aware_retained > random_retained
    # aware keeps essentially ALL the planted mass.
    assert aware_retained >= 0.95 * sens.sum()


def test_score_aware_prune_keep_one_is_noop():
    codes = np.array([1, 2, 3], dtype=np.int8)
    sens = np.array([1.0, 2.0, 3.0])
    out = SQ.score_aware_prune_codes(codes, sens, keep_fraction=1.0)
    assert np.array_equal(out, codes)


def test_score_aware_prune_actually_zeros_entries():
    rng = np.random.default_rng(9)
    codes = rng.integers(-40, 41, size=(8, 8)).astype(np.int8)
    codes[codes == 0] = 1  # ensure no pre-existing zeros
    sens = np.abs(codes).astype(np.float64)
    pruned = SQ.score_aware_prune_codes(codes, sens, keep_fraction=0.5)
    n_zero = int((pruned == 0).sum())
    assert n_zero >= codes.size // 2 - 1  # roughly half zeroed
    # the pruned tensor must DIFFER from input (catches no-op masquerade).
    assert not np.array_equal(pruned, codes)


def test_score_aware_prune_rejects_bad_keep_fraction():
    codes = np.array([1, 2], dtype=np.int8)
    sens = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        SQ.score_aware_prune_codes(codes, sens, keep_fraction=0.0)
    with pytest.raises(ValueError):
        SQ.score_aware_prune_codes(codes, sens, keep_fraction=1.5)


def test_score_aware_prune_shape_mismatch_raises():
    codes = np.array([1, 2, 3], dtype=np.int8)
    sens = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        SQ.score_aware_prune_codes(codes, sens, keep_fraction=0.5)


def test_magnitude_sensitivity_is_abs():
    codes = np.array([-3, 0, 5], dtype=np.int8)
    s = SQ.magnitude_sensitivity(codes)
    assert np.array_equal(s, np.array([3.0, 0.0, 5.0]))


# ---------------------------------------------------------------------------
# rate-only ΔS sign + magnitude
# ---------------------------------------------------------------------------
def test_rate_only_delta_score_negative_when_bytes_drop():
    ds = SQ.rate_only_delta_score(141_422, 162_127)  # keep=0.7 measured byte drop
    assert ds < 0
    expected = 25.0 * (141_422 - 162_127) / SQ.RATE_DENOM
    assert abs(ds - expected) < 1e-12


def test_rate_only_delta_score_zero_when_unchanged():
    assert SQ.rate_only_delta_score(162_127, 162_127) == 0.0


def test_brotli_zeros_are_cheap_sanity():
    # the mechanism behind the pruning win: zeros compress far below random int8.
    rng = np.random.default_rng(10)
    dense = rng.integers(-60, 61, size=20_000).astype(np.int8).tobytes()
    sparse = np.zeros(20_000, dtype=np.int8).tobytes()
    assert len(brotli.compress(sparse, quality=11)) < len(brotli.compress(dense, quality=11)) // 10
