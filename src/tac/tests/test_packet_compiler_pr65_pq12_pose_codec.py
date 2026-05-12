"""Tests for the PR65 ``henosis`` PQ12 pose codec.

Covers the 12-bit / 3-byte / 2-value packing grammar from PR65.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    MAGIC_PQ12,
    PQ12_MAX_QUANTUM,
    PQ12PoseStream,
    decode_pq12_pose,
    encode_pq12_pose,
)
from tac.packet_compiler.pr65_pq12_pose_codec import (
    _pack_12bit_pairs,
    _unpack_12bit_pairs,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Magic constants pinned to PR65 source ───────────────────────────────────


def test_pr65_magic_constants_match_source() -> None:
    assert MAGIC_PQ12 == b"PQ12"
    assert len(MAGIC_PQ12) == 4
    assert PQ12_MAX_QUANTUM == 0xFFF
    assert PQ12_MAX_QUANTUM == 4095


# ── 12-bit pair pack/unpack helpers ─────────────────────────────────────────


def test_pr65_12bit_pack_unpack_even_count_round_trips() -> None:
    """Packing 4 values into 6 bytes, unpacking back, must be identity."""
    values = np.array([0, 1, 4095, 2048], dtype=np.uint16)
    packed = _pack_12bit_pairs(values)
    assert len(packed) == 2 * 3  # 2 pairs * 3 bytes
    unpacked = _unpack_12bit_pairs(packed, expected_count=4)
    np.testing.assert_array_equal(unpacked, values)


def test_pr65_12bit_pack_odd_count_pads_with_zero() -> None:
    """An odd-count input rounds up; trailing slot is zero-padded."""
    values = np.array([5, 17, 33], dtype=np.uint16)
    packed = _pack_12bit_pairs(values)
    assert len(packed) == 2 * 3  # ceil(3/2) = 2 pairs
    unpacked = _unpack_12bit_pairs(packed, expected_count=3)
    np.testing.assert_array_equal(unpacked, values)


def test_pr65_12bit_pack_rejects_oversized_value() -> None:
    values = np.array([4096], dtype=np.uint16)
    with pytest.raises(ValueError, match="exceeds 12-bit"):
        _pack_12bit_pairs(values)


def test_pr65_12bit_pack_rejects_non_uint16_input() -> None:
    values = np.array([5], dtype=np.int32)
    with pytest.raises(TypeError, match="uint16"):
        _pack_12bit_pairs(values)


def test_pr65_12bit_unpack_rejects_non_multiple_of_3() -> None:
    with pytest.raises(ValueError, match="multiple of 3"):
        _unpack_12bit_pairs(b"\x00\x00", expected_count=1)


def test_pr65_12bit_unpack_rejects_oversized_expected_count() -> None:
    with pytest.raises(ValueError, match="only fits"):
        _unpack_12bit_pairs(b"\x00\x00\x00", expected_count=3)


def test_pr65_12bit_pack_empty_returns_empty() -> None:
    assert _pack_12bit_pairs(np.array([], dtype=np.uint16)) == b""


# ── Round-trip identity ─────────────────────────────────────────────────────


def test_pr65_stream_is_frozen_dataclass() -> None:
    poses = np.full((4, 6), 20.0, dtype=np.float32)
    s = encode_pq12_pose(poses)
    with pytest.raises((AttributeError, TypeError)):
        s.n_frames = 999  # type: ignore[misc]


def test_pr65_round_trips_typical_pose() -> None:
    rng = np.random.default_rng(20260512)
    n_frames, n_dims = 100, 6
    poses = np.zeros((n_frames, n_dims), dtype=np.float32)
    poses[:, 0] = 20.0 + rng.uniform(0.0, 10.0, n_frames)
    poses[:, 1:] = rng.uniform(-1.0, 1.0, (n_frames, 5))
    s = encode_pq12_pose(poses)
    r = decode_pq12_pose(s.payload)
    assert r.shape == poses.shape
    # 12-bit quantum has range/4095 resolution per column; tolerate ~1.5 quanta.
    per_col_max_quantum = (poses.max(axis=0) - poses.min(axis=0)) / 4095.0
    np.testing.assert_allclose(r, poses, atol=1.5 * per_col_max_quantum.max())


def test_pr65_payload_starts_with_magic() -> None:
    poses = np.full((4, 6), 20.0, dtype=np.float32)
    s = encode_pq12_pose(poses)
    assert s.payload[:4] == MAGIC_PQ12


def test_pr65_payload_layout_matches_pr65_grammar() -> None:
    """Wire format: magic[4] + n_frames:u16 + n_dims:u16 + mn[fp32*n_dims] +
    scale[fp32*n_dims] + packed[ceil(n_frames*n_dims/2)*3]."""
    n_frames, n_dims = 10, 6
    poses = np.full((n_frames, n_dims), 20.0, dtype=np.float32)
    s = encode_pq12_pose(poses)
    total_values = n_frames * n_dims
    expected_packed_len = ((total_values + 1) // 2) * 3
    expected_total = 4 + 4 + 4 * n_dims + 4 * n_dims + expected_packed_len
    assert len(s.payload) == expected_total
    # Verify n_frames/n_dims header.
    nf, nd = struct.unpack_from("<HH", s.payload, 4)
    assert nf == n_frames
    assert nd == n_dims


def test_pr65_round_trips_odd_total_values() -> None:
    """600 frames * 7 dims = 4200 (even); 11 frames * 5 dims = 55 (odd).
    The odd-total case stresses the 12-bit pair padding boundary."""
    rng = np.random.default_rng(20260513)
    n_frames, n_dims = 11, 5
    poses = rng.uniform(0.0, 1.0, (n_frames, n_dims)).astype(np.float32)
    s = encode_pq12_pose(poses)
    r = decode_pq12_pose(s.payload)
    assert r.shape == (n_frames, n_dims)
    # Should match within the auto-derived quantum.
    np.testing.assert_allclose(r, poses, atol=1.5 / 4095.0)


def test_pr65_round_trips_pr65_default_shape() -> None:
    """PR65 production shape: (600, 6)."""
    rng = np.random.default_rng(20260514)
    poses = np.zeros((600, 6), dtype=np.float32)
    poses[:, 0] = 20.0 + rng.uniform(0.0, 10.0, 600)
    poses[:, 1:] = rng.uniform(-1.0, 1.0, (600, 5))
    s = encode_pq12_pose(poses)
    r = decode_pq12_pose(s.payload)
    assert s.n_frames == 600
    assert s.n_dims == 6
    np.testing.assert_allclose(r, poses, atol=1.5 / 4095.0 * 10.0)


# ── Encoder failure modes ───────────────────────────────────────────────────


def test_pr65_encode_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="2D"):
        encode_pq12_pose(np.zeros(6, dtype=np.float32))


def test_pr65_encode_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        encode_pq12_pose(np.zeros((0, 6), dtype=np.float32))


def test_pr65_encode_rejects_oversized_n_frames() -> None:
    poses = np.zeros((65536, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="n_frames"):
        encode_pq12_pose(poses)


def test_pr65_encode_rejects_oversized_n_dims() -> None:
    poses = np.zeros((1, 65536), dtype=np.float32)
    with pytest.raises(ValueError, match="n_dims"):
        encode_pq12_pose(poses)


def test_pr65_encode_rejects_explicit_mn_wrong_shape() -> None:
    poses = np.zeros((4, 6), dtype=np.float32)
    bad_mn = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="mn shape"):
        encode_pq12_pose(poses, mn=bad_mn)


def test_pr65_encode_rejects_explicit_scale_wrong_shape() -> None:
    poses = np.zeros((4, 6), dtype=np.float32)
    bad_scale = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="scale shape"):
        encode_pq12_pose(poses, scale=bad_scale)


def test_pr65_encode_rejects_non_positive_scale() -> None:
    poses = np.zeros((4, 6), dtype=np.float32)
    bad_scale = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError, match="strictly positive"):
        encode_pq12_pose(poses, scale=bad_scale)


def test_pr65_encode_rejects_out_of_12bit_quantisation() -> None:
    """With small scale=1e-6, values >> 4095 quanta overflow 12-bit."""
    poses = np.full((4, 6), 1.0, dtype=np.float32)
    bad_scale = np.full(6, 1e-6, dtype=np.float32)
    bad_mn = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError, match="out of 12-bit"):
        encode_pq12_pose(poses, mn=bad_mn, scale=bad_scale)


# ── Decoder failure modes ───────────────────────────────────────────────────


def test_pr65_decode_rejects_bytes_type() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        decode_pq12_pose("not bytes")  # type: ignore[arg-type]


def test_pr65_decode_rejects_wrong_magic() -> None:
    with pytest.raises(ValueError, match="missing PQ12 magic"):
        decode_pq12_pose(b"NOTQ" + b"\x00" * 16)


def test_pr65_decode_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="shape header"):
        decode_pq12_pose(MAGIC_PQ12 + b"\x00")


def test_pr65_decode_rejects_zero_dimensions() -> None:
    payload = MAGIC_PQ12 + struct.pack("<HH", 0, 1)
    with pytest.raises(ValueError, match="must be > 0"):
        decode_pq12_pose(payload)


def test_pr65_decode_rejects_truncated_mn() -> None:
    payload = MAGIC_PQ12 + struct.pack("<HH", 1, 2) + b"\x00" * 4
    with pytest.raises(ValueError, match="mn vector"):
        decode_pq12_pose(payload)


def test_pr65_decode_rejects_truncated_scale() -> None:
    payload = (
        MAGIC_PQ12
        + struct.pack("<HH", 1, 2)
        + np.zeros(2, dtype=np.float32).tobytes()
        + b"\x00" * 4
    )
    with pytest.raises(ValueError, match="scale vector"):
        decode_pq12_pose(payload)


def test_pr65_decode_rejects_non_positive_decoded_scale() -> None:
    payload = (
        MAGIC_PQ12
        + struct.pack("<HH", 1, 1)
        + np.zeros(1, dtype=np.float32).tobytes()
        + np.zeros(1, dtype=np.float32).tobytes()  # scale = 0
        + b"\x00\x00\x00"
    )
    with pytest.raises(ValueError, match="strictly positive"):
        decode_pq12_pose(payload)


def test_pr65_decode_rejects_trailing_bytes() -> None:
    poses = np.zeros((1, 1), dtype=np.float32)
    s = encode_pq12_pose(
        poses,
        mn=np.zeros(1, dtype=np.float32),
        scale=np.ones(1, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_pq12_pose(s.payload + b"\xff\xff\xff")


# ── Golden vector ───────────────────────────────────────────────────────────


class TestPR65GoldenVector:
    def test_pq12_pose_golden_vector(self) -> None:
        rng = np.random.default_rng(seed=20260512)
        n_frames, n_dims = 600, 6
        poses = np.zeros((n_frames, n_dims), dtype=np.float32)
        poses[:, 0] = 20.0 + rng.uniform(0.0, 10.0, n_frames)
        poses[:, 1:] = rng.uniform(-1.0, 1.0, (n_frames, 5))
        stream = encode_pq12_pose(poses)
        digest = hashlib.sha256(stream.payload).hexdigest()
        golden = GOLDEN_DIR / "pr65_pq12_pose_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR65 PQ12 pose byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "n_dims": int(n_dims),
                        "n_frames": int(n_frames),
                        "payload_len": len(stream.payload),
                        "schema": "pr65_pq12_pose.v1",
                        "seed": 20260512,
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (
                GOLDEN_DIR / "pr65_pq12_pose_v1_poses.bin"
            ).write_bytes(poses.astype(np.float32).tobytes())
