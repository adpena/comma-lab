"""Tests for the PR93 ``flatpup`` pose-codec packet-compiler primitives.

Covers round-trip identity, golden-vector pinning, magic constants, and
representative failure modes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    MAGIC_MODEL_COMPACT,
    MAGIC_POSE_DV,
    decode_delta_varint_pose,
    encode_delta_varint_pose,
    pack_qzmb1_block,
    unpack_qzmb1_block,
)
from tac.packet_compiler.pr93_pose_codec import (
    _decode_unsigned_varint,
    _encode_unsigned_varint,
    _zigzag_decode_u32,
    _zigzag_encode_i32,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Magic constants ─────────────────────────────────────────────────────────


def test_pr93_magic_constants_match_source_pinning() -> None:
    assert MAGIC_POSE_DV == b"QZPDV1\x00\x00"
    assert MAGIC_MODEL_COMPACT == b"QZMB1\x00\x00\x00"
    assert len(MAGIC_POSE_DV) == 8
    assert len(MAGIC_MODEL_COMPACT) == 8


# ── Zigzag + varint helpers ─────────────────────────────────────────────────


def test_pr93_zigzag_roundtrips_signed_ints() -> None:
    for value in (
        -(1 << 30),
        -1,
        0,
        1,
        17,
        -17,
        (1 << 30) - 1,
        (1 << 31) - 1,
    ):
        u = _zigzag_encode_i32(int(value))
        assert u >= 0
        recovered = _zigzag_decode_u32(u)
        assert recovered == value


def test_pr93_zigzag_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="int32 range"):
        _zigzag_encode_i32(1 << 31)
    with pytest.raises(ValueError, match="int32 range"):
        _zigzag_encode_i32(-(1 << 31) - 1)


def test_pr93_varint_roundtrips_zero_and_large() -> None:
    for value in (0, 1, 127, 128, 16384, (1 << 30) - 1, (1 << 32) - 1):
        encoded = _encode_unsigned_varint(value)
        recovered, off = _decode_unsigned_varint(encoded, 0)
        assert recovered == value
        assert off == len(encoded)


def test_pr93_varint_rejects_negative() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        _encode_unsigned_varint(-1)


def test_pr93_varint_rejects_truncated_input() -> None:
    truncated = b"\x80\x80"  # missing terminator
    with pytest.raises(ValueError, match="truncated varint"):
        _decode_unsigned_varint(truncated, 0)


# ── Delta-varint pose round-trip ────────────────────────────────────────────


def test_pr93_delta_varint_round_trips_typical_pose() -> None:
    rng = np.random.default_rng(seed=20260511)
    n_rows, n_dims = 64, 6
    poses = rng.uniform(-0.5, 0.5, size=(n_rows, n_dims)).astype(np.float32)
    stream = encode_delta_varint_pose(poses)
    recovered = decode_delta_varint_pose(stream.payload)
    assert recovered.shape == poses.shape
    # Allow up to 1 quantum per value at the chosen scale (default 1/255).
    np.testing.assert_allclose(recovered, poses, atol=stream.scale.max() * 1.5)


def test_pr93_delta_varint_explicit_lo_scale_round_trips_exactly() -> None:
    n_rows, n_dims = 10, 3
    poses = np.arange(n_rows * n_dims, dtype=np.float32).reshape(n_rows, n_dims)
    lo = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    stream = encode_delta_varint_pose(poses, lo=lo, scale=scale, bits=8)
    recovered = decode_delta_varint_pose(stream.payload)
    np.testing.assert_array_equal(recovered, poses)


def test_pr93_delta_varint_payload_starts_with_magic() -> None:
    poses = np.zeros((4, 2), dtype=np.float32)
    stream = encode_delta_varint_pose(
        poses,
        lo=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        bits=8,
    )
    assert stream.payload[:8] == MAGIC_POSE_DV


def test_pr93_delta_varint_chooses_uint16_when_first_row_overflows_uint8() -> None:
    poses = np.zeros((3, 1), dtype=np.float32)
    poses[0, 0] = 0.0
    poses[1, 0] = 300.0
    poses[2, 0] = 600.0
    stream = encode_delta_varint_pose(
        poses,
        lo=np.zeros(1, dtype=np.float32),
        scale=np.ones(1, dtype=np.float32),
    )
    # bits should be 16 because first-row max (600 → q=600) > 255
    assert stream.bits == 16
    recovered = decode_delta_varint_pose(stream.payload)
    np.testing.assert_allclose(recovered, poses, atol=1.0)


def test_pr93_delta_varint_uses_uint8_for_small_values() -> None:
    poses = np.zeros((3, 1), dtype=np.float32)
    poses[0, 0] = 5.0
    poses[1, 0] = 10.0
    poses[2, 0] = 20.0
    stream = encode_delta_varint_pose(
        poses,
        lo=np.zeros(1, dtype=np.float32),
        scale=np.ones(1, dtype=np.float32),
    )
    assert stream.bits == 8


def test_pr93_delta_varint_single_row_works() -> None:
    poses = np.array([[1.0, 2.0]], dtype=np.float32)
    stream = encode_delta_varint_pose(
        poses,
        lo=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        bits=8,
    )
    recovered = decode_delta_varint_pose(stream.payload)
    np.testing.assert_array_equal(recovered, poses)


def test_pr93_delta_varint_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="2D"):
        encode_delta_varint_pose(np.zeros(5, dtype=np.float32))


def test_pr93_delta_varint_rejects_negative_q() -> None:
    poses = np.array([[-1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="quantised value < 0"):
        encode_delta_varint_pose(
            poses,
            lo=np.zeros(1, dtype=np.float32),
            scale=np.ones(1, dtype=np.float32),
        )


def test_pr93_delta_varint_rejects_zero_scale() -> None:
    poses = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="strictly positive"):
        encode_delta_varint_pose(
            poses,
            lo=np.zeros(1, dtype=np.float32),
            scale=np.zeros(1, dtype=np.float32),
        )


def test_pr93_delta_varint_decoder_rejects_wrong_magic() -> None:
    with pytest.raises(ValueError, match="QZPDV1 magic"):
        decode_delta_varint_pose(b"NOTQZPDV" + b"\x00" * 20)


def test_pr93_delta_varint_decoder_rejects_trailing_bytes() -> None:
    poses = np.zeros((2, 1), dtype=np.float32)
    stream = encode_delta_varint_pose(
        poses,
        lo=np.zeros(1, dtype=np.float32),
        scale=np.ones(1, dtype=np.float32),
        bits=8,
    )
    padded = stream.payload + b"\x00\x00"
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_delta_varint_pose(padded)


def test_pr93_delta_varint_decoder_rejects_unsupported_bits() -> None:
    # Hand-craft a payload with bits=12 (invalid).
    import struct as _st

    payload = MAGIC_POSE_DV + _st.pack("<III", 1, 1, 12)
    with pytest.raises(ValueError, match="unsupported bits"):
        decode_delta_varint_pose(payload)


def test_pr93_delta_varint_decoder_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="shape header"):
        decode_delta_varint_pose(MAGIC_POSE_DV + b"\x00\x00")


def test_pr93_delta_varint_decoder_rejects_zero_rows_or_dims() -> None:
    import struct as _st

    with pytest.raises(ValueError, match="n_rows"):
        decode_delta_varint_pose(MAGIC_POSE_DV + _st.pack("<III", 0, 1, 8))
    with pytest.raises(ValueError, match="n_dims"):
        decode_delta_varint_pose(MAGIC_POSE_DV + _st.pack("<III", 1, 0, 8))


def test_pr93_delta_varint_decoder_rejects_truncated_vectors() -> None:
    import struct as _st

    base = MAGIC_POSE_DV + _st.pack("<III", 1, 2, 8)
    with pytest.raises(ValueError, match="lo vector"):
        decode_delta_varint_pose(base + b"\x00" * 4)
    lo = np.zeros(2, dtype=np.float32).tobytes()
    with pytest.raises(ValueError, match="scale vector"):
        decode_delta_varint_pose(base + lo + b"\x00" * 4)
    scale = np.ones(2, dtype=np.float32).tobytes()
    with pytest.raises(ValueError, match="first-row vector"):
        decode_delta_varint_pose(base + lo + scale + b"\x00")


def test_pr93_delta_varint_decoder_rejects_invalid_scale_and_negative_q() -> None:
    import struct as _st

    lo = np.zeros(1, dtype=np.float32).tobytes()
    bad_scale = np.zeros(1, dtype=np.float32).tobytes()
    first = np.array([0], dtype=np.uint8).tobytes()
    payload = MAGIC_POSE_DV + _st.pack("<III", 1, 1, 8) + lo + bad_scale + first
    with pytest.raises(ValueError, match="strictly positive"):
        decode_delta_varint_pose(payload)

    scale = np.ones(1, dtype=np.float32).tobytes()
    # n_rows=2, first=0, one delta=-1 encoded as zigzag unsigned 1.
    payload = MAGIC_POSE_DV + _st.pack("<III", 2, 1, 8) + lo + scale + first + b"\x01"
    with pytest.raises(ValueError, match="quantised pose value < 0"):
        decode_delta_varint_pose(payload)


# ── QZMB1 block grammar ─────────────────────────────────────────────────────


def test_pr93_qzmb1_round_trips_typical_block() -> None:
    arch_json = b'{"hidden": 64, "blocks": 3}'
    body = b"\x00\x01\x02\x03\xff"
    block = pack_qzmb1_block(block_size=32, arch_config_json=arch_json, body=body)
    parsed = unpack_qzmb1_block(block.payload)
    assert parsed.block_size == 32
    assert parsed.arch_config_json == arch_json
    assert parsed.body == body


def test_pr93_qzmb1_payload_starts_with_magic() -> None:
    block = pack_qzmb1_block(block_size=32, arch_config_json=b"{}")
    assert block.payload[:8] == MAGIC_MODEL_COMPACT


def test_pr93_qzmb1_round_trips_empty_body() -> None:
    block = pack_qzmb1_block(block_size=16, arch_config_json=b"{}", body=b"")
    parsed = unpack_qzmb1_block(block.payload)
    assert parsed.body == b""


def test_pr93_qzmb1_rejects_invalid_block_size() -> None:
    with pytest.raises(ValueError, match="block_size"):
        pack_qzmb1_block(block_size=0, arch_config_json=b"{}")
    with pytest.raises(ValueError, match="block_size"):
        pack_qzmb1_block(block_size=70000, arch_config_json=b"{}")


def test_pr93_qzmb1_rejects_oversized_arch_config() -> None:
    with pytest.raises(ValueError, match="too long"):
        pack_qzmb1_block(block_size=32, arch_config_json=b"x" * 65536)


def test_pr93_qzmb1_decoder_rejects_wrong_magic() -> None:
    with pytest.raises(ValueError, match="QZMB1 magic"):
        unpack_qzmb1_block(b"NOTQZMB1" + b"\x00" * 8)


def test_pr93_qzmb1_decoder_rejects_truncated_arch_config() -> None:
    import struct as _st

    payload = MAGIC_MODEL_COMPACT + _st.pack("<HH", 32, 100) + b"abc"
    with pytest.raises(ValueError, match="arch_config truncated"):
        unpack_qzmb1_block(payload)


# ── Frozen dataclass invariants ─────────────────────────────────────────────


def test_pr93_streams_are_frozen_dataclasses() -> None:
    stream = encode_delta_varint_pose(
        np.zeros((2, 1), dtype=np.float32),
        lo=np.zeros(1, dtype=np.float32),
        scale=np.ones(1, dtype=np.float32),
        bits=8,
    )
    with pytest.raises((AttributeError, TypeError)):
        stream.n_rows = 999  # type: ignore[misc]

    block = pack_qzmb1_block(block_size=32, arch_config_json=b"{}")
    with pytest.raises((AttributeError, TypeError)):
        block.block_size = 999  # type: ignore[misc]


# ── Golden vectors ──────────────────────────────────────────────────────────


class TestPR93GoldenVectors:
    def test_delta_varint_pose_golden_vector(self) -> None:
        rng = np.random.default_rng(seed=20260511)
        n_rows, n_dims = 16, 4
        poses = rng.uniform(0.0, 1.0, size=(n_rows, n_dims)).astype(np.float32)
        lo = np.full(n_dims, 0.0, dtype=np.float32)
        scale = np.full(n_dims, 1.0 / 255.0, dtype=np.float32)
        stream = encode_delta_varint_pose(poses, lo=lo, scale=scale, bits=8)
        digest = hashlib.sha256(stream.payload).hexdigest()
        golden = GOLDEN_DIR / "pr93_delta_varint_pose_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR93 delta-varint pose byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "bits": 8,
                        "n_dims": int(n_dims),
                        "n_rows": int(n_rows),
                        "payload_len": len(stream.payload),
                        "schema": "pr93_delta_varint_pose.v1",
                        "seed": 20260511,
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def test_qzmb1_golden_vector(self) -> None:
        arch_json = b'{"hidden": 64, "blocks": 3, "input_dim": 5}'
        body = bytes(range(64))  # 64 deterministic body bytes
        block = pack_qzmb1_block(block_size=32, arch_config_json=arch_json, body=body)
        digest = hashlib.sha256(block.payload).hexdigest()
        golden = GOLDEN_DIR / "pr93_qzmb1_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR93 QZMB1 byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "arch_config_json_len": len(arch_json),
                        "block_size": 32,
                        "body_len": len(body),
                        "payload_len": len(block.payload),
                        "schema": "pr93_qzmb1.v1",
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
