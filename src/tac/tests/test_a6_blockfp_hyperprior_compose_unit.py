"""Unit tests for ``tac.codec.a6_selfcomp_blockfp_hyperprior_compose``.

Tests are organised by the contract sections in the module docstring:

1. Roundtrip — encode then decode recovers the input bit-for-bit on
   synthetic symbol streams across multiple seeds.
2. Block-size invariance — smaller block size → finer per-block scale →
   lower payload bits per symbol at the cost of more side-info bytes.
3. Edge cases — 1-element, all-zero, all-equal, empty.
4. Hyperprior degenerate cases — narrow sigma and wide sigma.
5. Byte-budget identity — total bytes = header + side-info + payload, and
   the side-info budget matches ``num_blocks * scale_bytes_per_block``.
6. Composability — compose of decompose-of-compose is byte-identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.codec.a6_selfcomp_blockfp_hyperprior_compose import (
    ALPHA_DEFAULT,
    SCALE_QUANT_FP16,
    SCALE_QUANT_FP32,
    SCALE_QUANT_UINT8,
    SIGMA_FLOOR,
    compose_blockfp_with_hyperprior,
    decompose_blockfp_with_hyperprior,
    encode_blockfp_only,
    encode_hyperprior_only,
    hyperprior_sigma_from_scale,
    split_into_blockfp,
)


# ── 1. Roundtrip on synthetic streams ─────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2026])
def test_roundtrip_random_synthetic_stream(seed: int):
    rng = np.random.default_rng(seed)
    n = 4096
    # Spread of scales: values up to 100 in magnitude (within int8 range).
    symbols = rng.integers(-100, 101, size=n, dtype=np.int8)
    encoded, ledger = compose_blockfp_with_hyperprior(
        symbols, block_size=64, scale_quant=SCALE_QUANT_FP16
    )
    recovered = decompose_blockfp_with_hyperprior(encoded)
    assert recovered.shape == (n,)
    assert recovered.dtype == np.int8
    np.testing.assert_array_equal(recovered, symbols)
    # Ledger sanity: parts add up to total.
    assert (
        ledger["header_bytes"]
        + ledger["side_info_bytes"]
        + ledger["payload_bytes"]
        == ledger["total_bytes"]
    )


@pytest.mark.parametrize("scale_quant", [SCALE_QUANT_FP16, SCALE_QUANT_FP32, SCALE_QUANT_UINT8])
def test_roundtrip_each_scale_quant_mode(scale_quant: int):
    rng = np.random.default_rng(13)
    symbols = rng.integers(-50, 51, size=512, dtype=np.int8)
    encoded, _ = compose_blockfp_with_hyperprior(
        symbols, block_size=32, scale_quant=scale_quant
    )
    recovered = decompose_blockfp_with_hyperprior(encoded)
    np.testing.assert_array_equal(recovered, symbols)


# ── 2. Block-size invariance / monotonicity ───────────────────────────────


def test_smaller_block_means_smaller_payload_with_more_side_info():
    """Smaller B ⇒ finer per-block scale ⇒ payload should not be larger.

    The point of compose is that scale-conditional PMFs charge fewer bits
    per symbol when the block is uniformly small-magnitude (or uniformly
    large-magnitude), so payload monotonically improves as B shrinks. The
    side-info cost grows linearly in num_blocks. Both halves of the trade
    are visible in the ledger.
    """
    rng = np.random.default_rng(99)
    # A heteroscedastic stream: alternating large-mag and small-mag chunks
    # so per-block scale differs strongly across blocks.
    chunks = []
    for i in range(16):
        if i % 2 == 0:
            chunks.append(rng.integers(-5, 6, size=128, dtype=np.int8))
        else:
            chunks.append(rng.integers(-100, 101, size=128, dtype=np.int8))
    symbols = np.concatenate(chunks).astype(np.int8)

    _, ledger_big = compose_blockfp_with_hyperprior(
        symbols, block_size=512, scale_quant=SCALE_QUANT_FP16
    )
    _, ledger_small = compose_blockfp_with_hyperprior(
        symbols, block_size=64, scale_quant=SCALE_QUANT_FP16
    )

    # Smaller B → strictly more side-info.
    assert ledger_small["side_info_bytes"] > ledger_big["side_info_bytes"]
    # Smaller B → payload no larger (scale-conditional PMFs are tighter).
    assert ledger_small["payload_bytes"] <= ledger_big["payload_bytes"]


# ── 3. Edge cases ─────────────────────────────────────────────────────────


def test_one_element_tensor_roundtrips():
    symbols = np.array([42], dtype=np.int8)
    encoded, ledger = compose_blockfp_with_hyperprior(
        symbols, block_size=16, scale_quant=SCALE_QUANT_FP16
    )
    assert ledger["n_total"] == 1
    assert ledger["n_blocks"] == 1
    recovered = decompose_blockfp_with_hyperprior(encoded)
    np.testing.assert_array_equal(recovered, symbols)


def test_all_zero_tensor_roundtrips():
    symbols = np.zeros(256, dtype=np.int8)
    encoded, ledger = compose_blockfp_with_hyperprior(
        symbols, block_size=32, scale_quant=SCALE_QUANT_FP16
    )
    recovered = decompose_blockfp_with_hyperprior(encoded)
    np.testing.assert_array_equal(recovered, symbols)
    # All-zero blocks: scale = 0 for every block, sigma collapses to
    # sigma_floor. All symbols are 0; arithmetic coding charges very few
    # bits. Payload (after header) should be very small.
    assert ledger["payload_bytes"] < 50  # tight bound; near-empty stream


def test_all_equal_tensor_roundtrips():
    symbols = np.full(256, 7, dtype=np.int8)
    encoded, _ = compose_blockfp_with_hyperprior(
        symbols, block_size=32, scale_quant=SCALE_QUANT_FP16
    )
    recovered = decompose_blockfp_with_hyperprior(encoded)
    np.testing.assert_array_equal(recovered, symbols)


def test_empty_tensor_roundtrips():
    symbols = np.zeros((0,), dtype=np.int8)
    encoded, ledger = compose_blockfp_with_hyperprior(
        symbols, block_size=16, scale_quant=SCALE_QUANT_FP16, verify_roundtrip=False
    )
    assert ledger["n_total"] == 0
    assert ledger["n_blocks"] == 0
    recovered = decompose_blockfp_with_hyperprior(encoded)
    assert recovered.shape == (0,)


# ── 4. Hyperprior degenerate cases ────────────────────────────────────────


def test_hyperprior_narrow_sigma_zero_scale():
    """All-zero block: scale=0, sigma collapses to sigma_floor.

    sigma_floor=0.5 means a tight Gaussian centered at zero. The
    discretized PMF at sigma=0.5 puts >0.99 probability on symbol 0,
    so encoding 32 zeros should cost ~ -32 * log2(0.99) ≈ 0.5 bits =
    1 byte after the ChARM header overhead.
    """
    sigma = hyperprior_sigma_from_scale(0.0)
    assert sigma == SIGMA_FLOOR

    symbols = np.zeros(32, dtype=np.int8)
    encoded, ledger = compose_blockfp_with_hyperprior(
        symbols, block_size=32, scale_quant=SCALE_QUANT_FP16
    )
    # The payload (range-coded body) should be very small. Total payload
    # bytes counts the ChARM header (~9), range-coder finalisation bytes
    # (~10), and the chunk-section header (12 bytes for 1 chunk: 4-byte
    # n_chunks + 8-byte per-chunk header). Empirical: 33 bytes; bound 50.
    assert ledger["payload_bytes"] < 50


def test_hyperprior_wide_sigma_large_scale():
    """Block with the full int8 dynamic range: scale=127, sigma is wide.

    sigma_floor + alpha*127 = 0.5 + 0.55*127 ≈ 70. A Gaussian PMF with
    sigma=70 over a 256-symbol alphabet is nearly uniform — so the rate
    per symbol approaches log2(256) = 8 bits. We assert the per-symbol
    rate is in [6, 9] bits (uniform ± slack).
    """
    sigma = hyperprior_sigma_from_scale(127.0)
    assert sigma > 50.0

    rng = np.random.default_rng(2026)
    # Symbols spread across the full range so every class is hit.
    symbols = rng.integers(-127, 128, size=2048, dtype=np.int8)
    encoded, ledger = compose_blockfp_with_hyperprior(
        symbols, block_size=2048, scale_quant=SCALE_QUANT_FP16
    )
    bits_per_symbol = 8.0 * ledger["payload_bytes"] / ledger["n_total"]
    # Wide-sigma encoding ≈ uniform ⇒ near 8 bits/symbol.
    assert 6.0 <= bits_per_symbol <= 9.0


# ── 5. Byte-budget identity ───────────────────────────────────────────────


def test_byte_budget_identity_holds_at_every_block_size():
    """total_bytes = header + side_info + payload, and side_info matches
    num_blocks * scale_bytes_per_block exactly.
    """
    rng = np.random.default_rng(3)
    symbols = rng.integers(-30, 31, size=1024, dtype=np.int8)
    for block_size in [8, 16, 32, 64, 128, 256]:
        for scale_quant, expected_per_block in [
            (SCALE_QUANT_FP16, 2),
            (SCALE_QUANT_FP32, 4),
            (SCALE_QUANT_UINT8, 1),
        ]:
            _, ledger = compose_blockfp_with_hyperprior(
                symbols, block_size=block_size, scale_quant=scale_quant
            )
            assert (
                ledger["header_bytes"]
                + ledger["side_info_bytes"]
                + ledger["payload_bytes"]
                == ledger["total_bytes"]
            )
            assert (
                ledger["side_info_bytes"]
                == ledger["n_blocks"] * expected_per_block
            )
            assert ledger["scale_bytes_per_block"] == expected_per_block


# ── 6. Composability ──────────────────────────────────────────────────────


def test_compose_decompose_compose_byte_identical():
    """compose(decompose(compose(W))) == compose(W) at the byte level.

    This is the canonical "encoder is bit-deterministic" guarantee.
    """
    rng = np.random.default_rng(57)
    symbols = rng.integers(-80, 81, size=1500, dtype=np.int8)
    encoded_a, _ = compose_blockfp_with_hyperprior(
        symbols, block_size=48, scale_quant=SCALE_QUANT_FP16
    )
    recovered = decompose_blockfp_with_hyperprior(encoded_a)
    encoded_b, _ = compose_blockfp_with_hyperprior(
        recovered, block_size=48, scale_quant=SCALE_QUANT_FP16
    )
    assert encoded_a == encoded_b


# ── 7. Standalone-baseline byte ledger sanity ─────────────────────────────


def test_blockfp_only_byte_ledger():
    """Block-FP standalone uses raw int8 residuals — payload == n_total bytes."""
    rng = np.random.default_rng(11)
    symbols = rng.integers(-60, 61, size=1024, dtype=np.int8)
    _, ledger = encode_blockfp_only(
        symbols, block_size=32, scale_quant=SCALE_QUANT_FP16
    )
    # 1 byte per symbol for the payload (no entropy coding).
    assert ledger["payload_bytes"] == 1024
    # Side-info: 32 blocks × 2 bytes/block = 64.
    assert ledger["n_blocks"] == 32
    assert ledger["side_info_bytes"] == 64


def test_hyperprior_only_byte_ledger_smaller_for_skewed_streams():
    """Hyperprior standalone with global sigma — should beat raw int8 on a
    near-zero-mean stream because the PMF concentrates near zero.
    """
    rng = np.random.default_rng(23)
    # Tightly-distributed symbols → entropy ≪ 8 bits, so range-coded payload
    # should be smaller than the raw int8 byte count.
    symbols = rng.integers(-3, 4, size=1024, dtype=np.int8)
    _, ledger_h = encode_hyperprior_only(symbols)
    _, ledger_b = encode_blockfp_only(
        symbols, block_size=32, scale_quant=SCALE_QUANT_FP16
    )
    assert ledger_h["payload_bytes"] < ledger_b["payload_bytes"]


# ── 8. Wire-format guards ─────────────────────────────────────────────────


def test_decompose_rejects_bad_magic():
    bad = b"BAD!" + b"\x00" * 32
    with pytest.raises(ValueError, match="bad magic"):
        decompose_blockfp_with_hyperprior(bad)


def test_decompose_rejects_truncated_blob():
    with pytest.raises(ValueError, match="too short"):
        decompose_blockfp_with_hyperprior(b"A6BF")


def test_compose_rejects_negative_block_size():
    symbols = np.zeros(4, dtype=np.int8)
    with pytest.raises(ValueError, match="block_size"):
        compose_blockfp_with_hyperprior(symbols, block_size=0)


def test_split_into_blockfp_handles_partial_last_block():
    symbols = np.arange(50, dtype=np.int8)  # 50 elements, block_size=16 → 4 blocks
    split = split_into_blockfp(symbols, block_size=16)
    assert split.n_blocks == 4
    assert split.blocks[0].size == 16
    assert split.blocks[1].size == 16
    assert split.blocks[2].size == 16
    assert split.blocks[3].size == 2  # 50 - 3*16 = 2
    # Per-block scales are max abs.
    assert split.scales[0] == 15.0  # arange(0,16) → max=15
    assert split.scales[3] == 49.0  # arange(48,50) → max=49


# ── 9. Hyperprior sigma map sanity ────────────────────────────────────────


def test_hyperprior_sigma_is_monotone_in_scale():
    sigmas = [hyperprior_sigma_from_scale(s) for s in [0, 1, 5, 25, 50, 100, 127]]
    assert all(sigmas[i] < sigmas[i + 1] for i in range(len(sigmas) - 1))


def test_hyperprior_sigma_floor_clamp():
    """Even at scale=0, sigma >= sigma_floor (so PMF is positive)."""
    sigma = hyperprior_sigma_from_scale(0.0, sigma_floor=0.5, alpha=0.0)
    assert sigma == 0.5


def test_hyperprior_rejects_negative_scale():
    with pytest.raises(ValueError, match="scale"):
        hyperprior_sigma_from_scale(-1.0)


def test_split_into_blockfp_handles_int8_min_without_overflow():
    symbols = np.array([-128, -2, 3], dtype=np.int8)
    split = split_into_blockfp(symbols, block_size=3)
    assert split.scales.tolist() == [128.0]


def test_compose_rejects_non_default_wire_hyperparams():
    symbols = np.array([0, 1, -1], dtype=np.int8)
    with pytest.raises(ValueError, match="does not serialize alpha"):
        compose_blockfp_with_hyperprior(symbols, block_size=3, alpha=0.10)
    with pytest.raises(ValueError, match="does not serialize sigma_floor"):
        compose_blockfp_with_hyperprior(symbols, block_size=3, sigma_floor=0.25)


def test_decompose_rejects_trailing_bytes_non_empty():
    symbols = np.array([1, -2, 3, -4], dtype=np.int8)
    encoded, _ = compose_blockfp_with_hyperprior(symbols, block_size=2)
    with pytest.raises(ValueError, match="trailing bytes"):
        decompose_blockfp_with_hyperprior(encoded + b"UNCONSUMED")


def test_decompose_rejects_trailing_bytes_empty():
    symbols = np.zeros((0,), dtype=np.int8)
    encoded, _ = compose_blockfp_with_hyperprior(
        symbols, block_size=4, verify_roundtrip=False
    )
    with pytest.raises(ValueError, match="trailing bytes"):
        decompose_blockfp_with_hyperprior(encoded + b"UNCONSUMED")


def test_decompose_rejects_tampered_zero_block_size():
    symbols = np.array([1, 2, 3], dtype=np.int8)
    encoded, _ = compose_blockfp_with_hyperprior(symbols, block_size=3)
    tampered = bytearray(encoded)
    tampered[6:8] = b"\x00\x00"
    with pytest.raises(ValueError, match="block_size"):
        decompose_blockfp_with_hyperprior(bytes(tampered))


def test_baseline_helpers_return_bytes_matching_ledger_total():
    symbols = np.array([-128, -10, 0, 10, 127], dtype=np.int8)
    block_blob, block_ledger = encode_blockfp_only(
        symbols, block_size=2, scale_quant=SCALE_QUANT_FP16
    )
    hyper_blob, hyper_ledger = encode_hyperprior_only(symbols)
    assert len(block_blob) == block_ledger["total_bytes"]
    assert len(hyper_blob) == hyper_ledger["total_bytes"]
