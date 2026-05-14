# SPDX-License-Identifier: MIT
"""Tests for the PR64 ``unified_brotli`` pose-velocity-only codec.

Covers round-trip identity, golden-vector pinning, PR64 source-recipe parity,
and representative failure modes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    decode_unified_brotli_pose_velocity,
    encode_unified_brotli_pose_velocity,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Frozen dataclass + defaults ─────────────────────────────────────────────


def test_pr64_stream_is_frozen_dataclass() -> None:
    v = np.array([20.0, 20.5], dtype=np.float32)
    s = encode_unified_brotli_pose_velocity(v)
    with pytest.raises((AttributeError, TypeError)):
        s.n_frames = 999  # type: ignore[misc]


def test_pr64_defaults_match_pr64_source() -> None:
    """PR64 hardcodes scale=1/512 and bias=20.0."""
    v = np.array([20.0, 20.5], dtype=np.float32)
    s = encode_unified_brotli_pose_velocity(v)
    assert s.scale == pytest.approx(1.0 / 512.0)
    assert s.bias == pytest.approx(20.0)


# ── Round-trip identity ─────────────────────────────────────────────────────


def test_pr64_round_trips_single_frame() -> None:
    v = np.array([20.0], dtype=np.float32)
    s = encode_unified_brotli_pose_velocity(v)
    assert s.n_frames == 1
    assert len(s.payload) == 2  # uint16 vel0 only
    r = decode_unified_brotli_pose_velocity(s.payload, n_frames=1)
    assert r.shape == (1,)
    np.testing.assert_allclose(r, v, atol=1.5 / 512.0)


def test_pr64_round_trips_typical_velocity_stream() -> None:
    """Realistic 600-frame straight-line driving velocity."""
    rng = np.random.default_rng(20260512)
    v = (20.0 + rng.uniform(0.0, 10.0, 600)).astype(np.float32)
    s = encode_unified_brotli_pose_velocity(v)
    r = decode_unified_brotli_pose_velocity(s.payload, n_frames=600)
    np.testing.assert_allclose(r, v, atol=1.5 / 512.0)


def test_pr64_payload_layout_matches_pr64_source() -> None:
    """Wire format: ``uint16 vel0 || int16[n-1] deltas`` little-endian."""
    v = np.array([20.0, 20.5, 19.8, 20.1], dtype=np.float32)
    s = encode_unified_brotli_pose_velocity(v)
    # Expect 2 bytes for vel0 + 6 bytes for 3 int16 deltas = 8 bytes.
    assert len(s.payload) == 2 + 3 * 2

    import struct as _st

    (vel0,) = _st.unpack_from("<H", s.payload, 0)
    # vel0 = round((20.0 - 20.0) / (1/512)) = 0
    assert vel0 == 0


def test_pr64_round_trip_zero_only_stream() -> None:
    v = np.full(8, 20.0, dtype=np.float32)
    s = encode_unified_brotli_pose_velocity(v)
    r = decode_unified_brotli_pose_velocity(s.payload, n_frames=8)
    np.testing.assert_array_equal(r, v)


def test_pr64_negative_deltas_round_trip() -> None:
    """Velocities decreasing below `bias` are supported because PR64 uses
    int32 cumsum for reconstruction."""
    v = np.array([20.5, 20.4, 20.1, 19.9, 19.6], dtype=np.float32)
    s = encode_unified_brotli_pose_velocity(v)
    r = decode_unified_brotli_pose_velocity(s.payload, n_frames=5)
    np.testing.assert_allclose(r, v, atol=1.5 / 512.0)


# ── Encoder failure modes ───────────────────────────────────────────────────


def test_pr64_encode_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        encode_unified_brotli_pose_velocity(np.array([], dtype=np.float32))


def test_pr64_encode_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="1D"):
        encode_unified_brotli_pose_velocity(np.zeros((4, 2), dtype=np.float32))


def test_pr64_encode_rejects_non_positive_scale() -> None:
    v = np.array([20.0], dtype=np.float32)
    with pytest.raises(ValueError, match="scale must be > 0"):
        encode_unified_brotli_pose_velocity(v, scale=0.0)
    with pytest.raises(ValueError, match="scale must be > 0"):
        encode_unified_brotli_pose_velocity(v, scale=-1.0)


def test_pr64_encode_rejects_first_frame_below_bias() -> None:
    """PR64 requires v[0] >= bias because vel0 is uint16."""
    v = np.array([19.5, 20.0], dtype=np.float32)
    with pytest.raises(ValueError, match="first-frame velocity quantum"):
        encode_unified_brotli_pose_velocity(v)


def test_pr64_encode_rejects_first_frame_above_uint16_range() -> None:
    """vel0 must fit uint16 [0, 65535]."""
    v = np.array([2000.0, 20.0], dtype=np.float32)
    with pytest.raises(ValueError, match="first-frame velocity quantum"):
        encode_unified_brotli_pose_velocity(v)


def test_pr64_encode_rejects_delta_out_of_int16_range() -> None:
    """A velocity jump > ~128 m/s (huge!) trips the int16 delta ceiling."""
    v = np.array([20.0, 100.0], dtype=np.float32)
    with pytest.raises(ValueError, match="velocity deltas out of int16"):
        encode_unified_brotli_pose_velocity(v)


# ── Decoder failure modes ───────────────────────────────────────────────────


def test_pr64_decode_rejects_bytes_only() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        decode_unified_brotli_pose_velocity("not bytes", n_frames=1)  # type: ignore[arg-type]


def test_pr64_decode_rejects_zero_or_negative_n_frames() -> None:
    with pytest.raises(ValueError, match="n_frames must be > 0"):
        decode_unified_brotli_pose_velocity(b"\x00\x00", n_frames=0)
    with pytest.raises(ValueError, match="n_frames must be > 0"):
        decode_unified_brotli_pose_velocity(b"\x00\x00", n_frames=-1)


def test_pr64_decode_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be > 0"):
        decode_unified_brotli_pose_velocity(b"\x00\x00", n_frames=1, scale=0.0)


def test_pr64_decode_rejects_truncated_payload() -> None:
    """Expected length for n_frames=5 is 2 + 4*2 = 10 bytes."""
    with pytest.raises(ValueError, match="payload length"):
        decode_unified_brotli_pose_velocity(b"\x00" * 4, n_frames=5)


def test_pr64_decode_rejects_oversized_payload() -> None:
    """A trailing byte raises (silent corruption is impossible)."""
    with pytest.raises(ValueError, match="payload length"):
        decode_unified_brotli_pose_velocity(b"\x00" * 12, n_frames=5)


# ── Round-trip with custom scale/bias ───────────────────────────────────────


def test_pr64_round_trips_custom_scale_and_bias() -> None:
    """Caller may override (scale, bias) for non-PR64-default semantics."""
    rng = np.random.default_rng(seed=20260512)
    v = (100.0 + rng.uniform(0.0, 50.0, 100)).astype(np.float32)
    s = encode_unified_brotli_pose_velocity(v, scale=1.0 / 256.0, bias=100.0)
    r = decode_unified_brotli_pose_velocity(
        s.payload, n_frames=100, scale=1.0 / 256.0, bias=100.0
    )
    np.testing.assert_allclose(r, v, atol=1.5 / 256.0)


# ── Golden vector ───────────────────────────────────────────────────────────


class TestPR64GoldenVector:
    def test_unified_brotli_pose_velocity_golden_vector(self) -> None:
        rng = np.random.default_rng(seed=20260512)
        n_frames = 600
        v = (20.0 + rng.uniform(0.0, 10.0, n_frames)).astype(np.float32)
        stream = encode_unified_brotli_pose_velocity(v)
        digest = hashlib.sha256(stream.payload).hexdigest()
        golden = GOLDEN_DIR / "pr64_unified_brotli_pose_velocity_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR64 unified-brotli pose-velocity byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "bias": 20.0,
                        "n_frames": int(n_frames),
                        "payload_len": len(stream.payload),
                        "scale": 1.0 / 512.0,
                        "schema": "pr64_unified_brotli_pose_velocity.v1",
                        "seed": 20260512,
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            # Also write the input fixture (fp32 little-endian) so Rust ports can read.
            (
                GOLDEN_DIR / "pr64_unified_brotli_pose_velocity_v1_velocities.bin"
            ).write_bytes(v.astype(np.float32).tobytes())
