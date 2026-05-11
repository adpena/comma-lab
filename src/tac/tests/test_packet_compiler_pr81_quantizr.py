"""Tests for the PR81 ``qzs3`` Quantizr packet-compiler primitives.

Covers FP4 codebook round-trip, nibble packing, ROUTER_ACTION packing,
golden-vector pinning, and representative failure modes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    PR81_POS_LEVELS,
    FP4Codebook,
    decode_router_actions,
    encode_router_actions,
    pack_nibbles,
    unpack_nibbles,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── FP4Codebook construction ────────────────────────────────────────────────


def test_pr81_default_pos_levels_match_quantizr() -> None:
    assert PR81_POS_LEVELS == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    cb = FP4Codebook()
    assert cb.pos_levels == PR81_POS_LEVELS
    np.testing.assert_array_equal(
        cb.levels_array(),
        np.array(PR81_POS_LEVELS, dtype=np.float32),
    )


def test_pr81_codebook_accepts_custom_levels() -> None:
    custom = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    cb = FP4Codebook(pos_levels=custom)
    assert cb.pos_levels == custom


def test_pr81_codebook_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 8"):
        FP4Codebook(pos_levels=(0.0, 1.0, 2.0))


def test_pr81_codebook_rejects_negative_levels() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FP4Codebook(pos_levels=(0.0, 0.5, -1.0, 1.5, 2.0, 3.0, 4.0, 6.0))


def test_pr81_codebook_rejects_non_monotone_levels() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        FP4Codebook(pos_levels=(0.0, 2.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))


# ── Quantise / dequantise round-trip ────────────────────────────────────────


def test_pr81_quantize_dequantize_round_trips_at_canonical_levels() -> None:
    cb = FP4Codebook()
    # Build values whose magnitudes land exactly on codebook levels.
    levels = np.array(PR81_POS_LEVELS, dtype=np.float32)
    signs = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    block_values = (levels[None, :] * signs[:, None]).reshape(-1)  # 32 values
    scales = np.ones(1, dtype=np.float32)
    nibbles = cb.quantize(block_values, scales=scales, block_size=32)
    decoded = cb.dequantize_from_nibbles(
        nibbles, scales=scales, block_size=32, n_values=block_values.size
    )
    np.testing.assert_array_equal(decoded, block_values)


def test_pr81_quantize_with_multiple_blocks_works() -> None:
    cb = FP4Codebook()
    rng = np.random.default_rng(seed=42)
    values = rng.uniform(-3.0, 3.0, size=64).astype(np.float32)
    scales = np.array([2.0, 1.0], dtype=np.float32)
    nibbles = cb.quantize(values, scales=scales, block_size=32)
    assert nibbles.shape == (64,)
    decoded = cb.dequantize_from_nibbles(
        nibbles, scales=scales, block_size=32, n_values=values.size
    )
    # Quantisation error bounded by max codebook step times scale.
    np.testing.assert_allclose(decoded, values, atol=2.0)


def test_pr81_quantize_pads_non_divisible_length() -> None:
    cb = FP4Codebook()
    values = np.arange(35, dtype=np.float32) / 10.0
    scales = np.array([0.1, 0.1], dtype=np.float32)
    nibbles = cb.quantize(values, scales=scales, block_size=32)
    assert nibbles.size == 64  # padded to multiple of block_size
    decoded = cb.dequantize_from_nibbles(
        nibbles, scales=scales, block_size=32, n_values=values.size
    )
    assert decoded.size == 35


def test_pr81_quantize_rejects_zero_or_negative_block_size() -> None:
    cb = FP4Codebook()
    with pytest.raises(ValueError, match="block_size"):
        cb.quantize(np.zeros(32, dtype=np.float32), scales=np.ones(1), block_size=0)


def test_pr81_quantize_rejects_mismatched_scales_count() -> None:
    cb = FP4Codebook()
    with pytest.raises(ValueError, match="scales has"):
        cb.quantize(
            np.zeros(64, dtype=np.float32),
            scales=np.ones(1, dtype=np.float32),
            block_size=32,
        )


def test_pr81_quantize_rejects_zero_scale() -> None:
    cb = FP4Codebook()
    with pytest.raises(ValueError, match="strictly positive"):
        cb.quantize(
            np.zeros(32, dtype=np.float32),
            scales=np.array([0.0]),
            block_size=32,
        )


def test_pr81_dequantize_rejects_negative_n_values() -> None:
    cb = FP4Codebook()
    with pytest.raises(ValueError, match="n_values"):
        cb.dequantize_from_nibbles(
            np.zeros(32, dtype=np.uint8),
            scales=np.ones(1, dtype=np.float32),
            block_size=32,
            n_values=-1,
        )


def test_pr81_dequantize_rejects_non_nibble_values() -> None:
    cb = FP4Codebook()
    with pytest.raises(ValueError, match="4 bits"):
        cb.dequantize_from_nibbles(
            np.array([0, 16] + [0] * 30),
            scales=np.ones(1, dtype=np.float32),
            block_size=32,
        )


# ── Nibble packing helpers ──────────────────────────────────────────────────


def test_pr81_pack_unpack_nibbles_round_trip() -> None:
    nibbles = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint8
    )
    packed = pack_nibbles(nibbles)
    recovered = unpack_nibbles(packed, count=nibbles.size)
    np.testing.assert_array_equal(recovered, nibbles)
    assert len(packed) == nibbles.size // 2


def test_pr81_pack_nibbles_rejects_odd_count() -> None:
    nibbles = np.array([0, 1, 2], dtype=np.uint8)
    with pytest.raises(ValueError, match="even"):
        pack_nibbles(nibbles)


def test_pr81_pack_nibbles_rejects_value_overflow() -> None:
    nibbles = np.array([0, 16], dtype=np.uint8)
    with pytest.raises(ValueError, match="4 bits"):
        pack_nibbles(nibbles)


def test_pr81_pack_nibbles_rejects_pre_cast_wraparound() -> None:
    nibbles = np.array([0, 256], dtype=np.int64)
    with pytest.raises(ValueError, match="4 bits"):
        pack_nibbles(nibbles)


def test_pr81_unpack_nibbles_truncates_to_count() -> None:
    nibbles = np.arange(8, dtype=np.uint8)
    packed = pack_nibbles(nibbles)
    recovered = unpack_nibbles(packed, count=5)
    np.testing.assert_array_equal(recovered, nibbles[:5])


def test_pr81_unpack_nibbles_rejects_count_over_capacity() -> None:
    packed = b"\x00\x00"
    with pytest.raises(ValueError, match="exceeds available"):
        unpack_nibbles(packed, count=5)


def test_pr81_unpack_nibbles_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="count must be >= 0"):
        unpack_nibbles(b"\x00", count=-1)


# ── ROUTER_ACTION packing ───────────────────────────────────────────────────


def test_pr81_router_action_round_trip_default_bits() -> None:
    rng = np.random.default_rng(seed=20260511)
    actions = rng.integers(0, 8, size=600, dtype=np.uint8)
    packed = encode_router_actions(actions, bits=3)
    assert len(packed) == 225  # 600 * 3 / 8 = 225
    recovered = decode_router_actions(packed, count=600, bits=3)
    np.testing.assert_array_equal(recovered, actions)


def test_pr81_router_action_round_trip_other_bit_widths() -> None:
    rng = np.random.default_rng(seed=20260511)
    for bits in (1, 2, 4, 5, 7, 8):
        max_val = (1 << bits) - 1
        actions = rng.integers(0, max_val + 1, size=120, dtype=np.uint8)
        packed = encode_router_actions(actions, bits=bits)
        expected_len = (120 * bits + 7) // 8
        assert len(packed) == expected_len
        recovered = decode_router_actions(packed, count=120, bits=bits)
        np.testing.assert_array_equal(recovered, actions)


def test_pr81_router_action_round_trip_empty_stream() -> None:
    actions = np.zeros(0, dtype=np.uint8)
    packed = encode_router_actions(actions, bits=3)
    assert packed == b""
    recovered = decode_router_actions(packed, count=0, bits=3)
    assert recovered.size == 0


def test_pr81_router_action_rejects_out_of_range_values() -> None:
    actions = np.array([0, 1, 8], dtype=np.uint8)
    with pytest.raises(ValueError, match="out of range"):
        encode_router_actions(actions, bits=3)


def test_pr81_router_action_rejects_invalid_bit_width() -> None:
    with pytest.raises(ValueError, match="bits must be"):
        encode_router_actions(np.array([0], dtype=np.uint8), bits=0)
    with pytest.raises(ValueError, match="bits must be"):
        encode_router_actions(np.array([0], dtype=np.uint8), bits=9)
    with pytest.raises(ValueError, match="bits must be"):
        decode_router_actions(b"", count=0, bits=0)


def test_pr81_router_action_decode_rejects_insufficient_payload() -> None:
    with pytest.raises(ValueError, match="requires"):
        decode_router_actions(b"\x00", count=10, bits=3)


def test_pr81_router_action_decode_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="count must be"):
        decode_router_actions(b"", count=-1, bits=3)


# ── Frozen dataclass invariants ─────────────────────────────────────────────


def test_pr81_fp4_codebook_is_frozen() -> None:
    cb = FP4Codebook()
    with pytest.raises((AttributeError, TypeError)):
        cb.pos_levels = (1.0,) * 8  # type: ignore[misc]


# ── Golden vectors ──────────────────────────────────────────────────────────


class TestPR81GoldenVectors:
    def test_fp4_codebook_golden_vector(self) -> None:
        cb = FP4Codebook()
        rng = np.random.default_rng(seed=20260511)
        values = rng.uniform(-3.0, 3.0, size=64).astype(np.float32)
        scales = np.array([1.5, 0.75], dtype=np.float32)
        nibbles = cb.quantize(values, scales=scales, block_size=32)
        packed = pack_nibbles(nibbles)
        digest = hashlib.sha256(packed).hexdigest()
        golden = GOLDEN_DIR / "pr81_fp4_codebook_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR81 FP4 codebook nibble stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "block_size": 32,
                        "n_blocks": int(scales.size),
                        "n_values": int(values.size),
                        "packed_len": len(packed),
                        "pos_levels": list(PR81_POS_LEVELS),
                        "schema": "pr81_fp4_codebook.v1",
                        "scales": [float(s) for s in scales.tolist()],
                        "seed": 20260511,
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def test_router_action_golden_vector(self) -> None:
        rng = np.random.default_rng(seed=20260511)
        actions = rng.integers(0, 8, size=600, dtype=np.uint8)
        packed = encode_router_actions(actions, bits=3)
        digest = hashlib.sha256(packed).hexdigest()
        golden = GOLDEN_DIR / "pr81_router_action_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR81 ROUTER_ACTION byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "bits": 3,
                        "count": 600,
                        "packed_len": len(packed),
                        "schema": "pr81_router_action.v1",
                        "seed": 20260511,
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
