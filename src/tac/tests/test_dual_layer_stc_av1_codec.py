"""Tests for ``tac.codec.dual_layer_stc_av1_codec``.

Verifies lossless encode/decode roundtrip on:
  * synthetic 5-class delta streams,
  * pure-zero deltas,
  * monotone deltas (all sign, no zeros),
  * extracted-from-masks streams,
and asserts compression-ratio sanity against direct brotli on int8 deltas.

Reference per Filler & Pevný 2010 IEEE TIFS 5(2):388-393 (dual-layer STC).
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.codec.dual_layer_stc_av1_codec import (
    DualLayerStats,
    HEADER_STRUCT,
    LAYER2_LEN_STRUCT,
    MAGIC,
    VERSION,
    decode_dual_layer,
    encode_dual_layer,
    extract_full_mask_deltas,
)


# ---------------------------------------------------------------------------
# Roundtrip on synthetic streams
# ---------------------------------------------------------------------------


def test_roundtrip_5class_synthetic() -> None:
    rng = np.random.default_rng(2026)
    deltas = rng.integers(-4, 5, size=4096, dtype=np.int16)
    blob = encode_dual_layer(deltas)
    out = decode_dual_layer(blob)
    assert out.shape == deltas.shape
    assert np.array_equal(out, deltas.astype(np.int16))


def test_roundtrip_all_zero_stream() -> None:
    deltas = np.zeros(1024, dtype=np.int16)
    blob = encode_dual_layer(deltas)
    out = decode_dual_layer(blob)
    assert np.array_equal(out, deltas)
    # All-zero stream must trip the empty-magnitude flag (no Layer 2 payload).
    layer2_len = LAYER2_LEN_STRUCT.unpack_from(
        blob, HEADER_STRUCT.size + _layer1_len(blob)
    )[0]
    assert layer2_len == 0


def test_roundtrip_no_zeros_stream() -> None:
    """Stream with no zero deltas — sign and magnitude both fully populated."""
    rng = np.random.default_rng(7)
    deltas = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4], size=512).astype(np.int16)
    assert (deltas != 0).all()
    blob = encode_dual_layer(deltas)
    out = decode_dual_layer(blob)
    assert np.array_equal(out, deltas)


def test_roundtrip_ternary_only_stream() -> None:
    """Stream that's already ternary (magnitudes all 1) — Layer 2 should be tiny."""
    rng = np.random.default_rng(11)
    deltas = rng.choice([-1, 0, 1], size=2048).astype(np.int16)
    blob = encode_dual_layer(deltas)
    out = decode_dual_layer(blob)
    assert np.array_equal(out, deltas)


def test_roundtrip_extreme_int8_range() -> None:
    """Boundary check at int8 limits."""
    deltas = np.array([-127, -64, -1, 0, 1, 64, 127], dtype=np.int16)
    blob = encode_dual_layer(deltas)
    out = decode_dual_layer(blob)
    assert np.array_equal(out, deltas)


def test_encode_rejects_non_1d() -> None:
    with pytest.raises(ValueError):
        encode_dual_layer(np.zeros((4, 4), dtype=np.int16))


def test_encode_rejects_empty() -> None:
    with pytest.raises(ValueError):
        encode_dual_layer(np.zeros(0, dtype=np.int16))


def test_encode_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        encode_dual_layer(np.array([0, 200, -1], dtype=np.int16))


def test_encode_rejects_oversized_magnitude() -> None:
    """Magnitude must fit in uint8 (>=0, <=127 because we still use int8 sign)."""
    with pytest.raises(ValueError):
        encode_dual_layer(np.array([-128, 0, 1], dtype=np.int16))


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def _layer1_len(blob: bytes) -> int:
    _, _, _, _, _, layer1_len = HEADER_STRUCT.unpack_from(blob, 0)
    return int(layer1_len)


def test_wire_format_magic_and_version() -> None:
    deltas = np.array([0, 1, -1, 2], dtype=np.int16)
    blob = encode_dual_layer(deltas)
    magic, version, _flags, n_sym, n_nz, _layer1_len = HEADER_STRUCT.unpack_from(blob, 0)
    assert magic == MAGIC
    assert version == VERSION
    assert n_sym == 4
    assert n_nz == 3


def test_decode_rejects_bad_magic() -> None:
    deltas = np.array([0, 1, -1], dtype=np.int16)
    blob = bytearray(encode_dual_layer(deltas))
    blob[0:4] = b"XXXX"
    with pytest.raises(ValueError):
        decode_dual_layer(bytes(blob))


def test_decode_rejects_bad_version() -> None:
    deltas = np.array([0, 1, -1], dtype=np.int16)
    blob = bytearray(encode_dual_layer(deltas))
    blob[4] = 99
    with pytest.raises(ValueError):
        decode_dual_layer(bytes(blob))


def test_decode_rejects_truncated_blob() -> None:
    deltas = np.array([0, 1, -1, 0], dtype=np.int16)
    blob = encode_dual_layer(deltas)
    with pytest.raises(ValueError):
        decode_dual_layer(blob[: len(blob) // 2])


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_record_stc_stats_returns_dataclass() -> None:
    rng = np.random.default_rng(3)
    deltas = rng.integers(-4, 5, size=256, dtype=np.int16)
    blob, stats = encode_dual_layer(
        deltas, constraint_height=4, block_size=16, record_stc_stats=True
    )
    assert isinstance(stats, DualLayerStats)
    assert stats.n_symbols == 256
    assert stats.n_nonzero == int((deltas != 0).sum())
    assert stats.layer1_bytes > 0
    assert stats.layer2_bytes > 0 or stats.n_nonzero == 0
    assert stats.total_bytes == len(blob)
    assert stats.stc_uniform_n_blocks == (256 + 15) // 16
    assert stats.stc_uniform_constraint_height == 4
    # Verify roundtrip still holds.
    assert np.array_equal(decode_dual_layer(blob), deltas)


def test_stats_nonzero_fraction_matches_input() -> None:
    deltas = np.array([0, 0, 0, 1, -2, 3], dtype=np.int16)
    blob, stats = encode_dual_layer(
        deltas, constraint_height=3, block_size=6, record_stc_stats=True
    )
    assert stats.nonzero_fraction == pytest.approx(3 / 6)


# ---------------------------------------------------------------------------
# Compression-ratio sanity
# ---------------------------------------------------------------------------


def test_dual_layer_beats_uncompressed_int8_baseline_on_sparse_stream() -> None:
    """On a sparse stream (mostly zeros), dual-layer must beat raw int8 size."""
    rng = np.random.default_rng(17)
    deltas = np.zeros(8192, dtype=np.int16)
    nz_idx = rng.choice(8192, size=200, replace=False)
    deltas[nz_idx] = rng.choice([-2, -1, 1, 2], size=200)
    blob = encode_dual_layer(deltas)
    raw_int8_size = 8192  # one byte per symbol
    assert len(blob) < raw_int8_size, (
        f"dual-layer {len(blob)} B failed to beat raw int8 {raw_int8_size} B "
        "on a 2.4%-density stream"
    )


def test_dual_layer_beats_uncompressed_int8_baseline_on_5class_uniform() -> None:
    """Even on uniform 5-class deltas the dual-layer should not blow up."""
    rng = np.random.default_rng(23)
    deltas = rng.integers(-4, 5, size=4096, dtype=np.int16)
    blob = encode_dual_layer(deltas)
    raw_int8_size = 4096
    # Brotli on uniform random ints might NOT beat the raw bytes; we only
    # require the codec stays within 2× of raw size on this adversarial input
    # (real mask streams are far more skewed, so the realistic ratio is
    # measured in tools/pr_alpha_mask_dual_layer_stc_empirical.py).
    assert len(blob) <= 2 * raw_int8_size, (
        f"dual-layer {len(blob)} B exceeded 2× raw int8 {raw_int8_size} B"
    )


# ---------------------------------------------------------------------------
# Helper: extract_full_mask_deltas
# ---------------------------------------------------------------------------


def test_extract_full_mask_deltas_basic() -> None:
    masks = np.array(
        [
            [[0, 1], [2, 3]],
            [[0, 2], [1, 3]],
            [[1, 2], [1, 4]],
        ],
        dtype=np.int64,
    )
    deltas = extract_full_mask_deltas(masks)
    # Frame 1 vs 0: [+0, +1, -1, +0]; Frame 2 vs 1: [+1, +0, +0, +1]
    expected = np.array([0, 1, -1, 0, 1, 0, 0, 1], dtype=np.int16)
    assert np.array_equal(deltas, expected)


def test_extract_full_mask_deltas_preserves_magnitude() -> None:
    """Unlike the ternary extractor, this must preserve full magnitude."""
    masks = np.array(
        [
            [[0]],
            [[4]],
            [[1]],
        ],
        dtype=np.int64,
    )
    deltas = extract_full_mask_deltas(masks)
    expected = np.array([4, -3], dtype=np.int16)
    assert np.array_equal(deltas, expected)


def test_extract_full_mask_deltas_rejects_non_3d() -> None:
    with pytest.raises(ValueError):
        extract_full_mask_deltas(np.zeros((4, 4), dtype=np.int64))


# ---------------------------------------------------------------------------
# End-to-end with real-shape extraction
# ---------------------------------------------------------------------------


def test_e2e_synthetic_mask_stream_roundtrip() -> None:
    """Build a mask stream, extract full deltas, encode, decode, verify."""
    rng = np.random.default_rng(2026)
    masks = rng.integers(0, 5, size=(8, 6, 8), dtype=np.int64)
    deltas = extract_full_mask_deltas(masks)
    blob = encode_dual_layer(deltas)
    out = decode_dual_layer(blob)
    assert np.array_equal(out, deltas)
