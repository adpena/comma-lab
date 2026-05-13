"""Tests for the PR63 ``qpose14`` codec primitives.

Covers the two PR63 primitives: the uint16-view int16 pose codec and the
packed-payload single-zip-member grammar.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    PR63_QPOSE14_N_DIMS,
    PR63_SIGNED_SCALE,
    PR63_VEL_BIAS,
    PR63_VEL_SCALE,
    QPose14PackedPayload,
    decode_qpose14_uint16_view_int16,
    encode_qpose14_uint16_view_int16,
    pack_qpose14_packed_payload,
    unpack_qpose14_packed_payload,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Constants pinned to PR63 source ─────────────────────────────────────────


def test_pr63_constants_match_source() -> None:
    """PR63 source-pinned constants: n_dims=6, vel scale=1/512, signed scale=1/2048."""
    assert PR63_QPOSE14_N_DIMS == 6
    assert PR63_VEL_SCALE == pytest.approx(1.0 / 512.0)
    assert PR63_VEL_BIAS == pytest.approx(20.0)
    assert PR63_SIGNED_SCALE == pytest.approx(1.0 / 2048.0)


# ── qpose14 uint16-view int16 codec ─────────────────────────────────────────


def test_pr63_codec_stream_is_frozen_dataclass() -> None:
    poses = np.full((2, 6), 20.0, dtype=np.float32)
    poses[:, 1:] = 0.0
    s = encode_qpose14_uint16_view_int16(poses)
    with pytest.raises((AttributeError, TypeError)):
        s.n_frames = 999  # type: ignore[misc]


def test_pr63_codec_round_trips_typical_pose() -> None:
    """Realistic forward-velocity + small lateral/yaw poses."""
    rng = np.random.default_rng(20260512)
    n_frames = 100
    poses = np.zeros((n_frames, 6), dtype=np.float32)
    poses[:, 0] = 20.0 + rng.uniform(0.0, 10.0, n_frames)
    poses[:, 1:] = rng.uniform(-1.0, 1.0, (n_frames, 5))
    s = encode_qpose14_uint16_view_int16(poses)
    r = decode_qpose14_uint16_view_int16(s.payload, n_frames=n_frames)
    # Tolerance: col-0 quantum is 1/512 = 0.002; cols 1-5 quantum is 1/2048 = 0.0005.
    np.testing.assert_allclose(r[:, 0], poses[:, 0], atol=1.5 / 512.0)
    np.testing.assert_allclose(r[:, 1:], poses[:, 1:], atol=1.5 / 2048.0)


def test_pr63_codec_payload_length_matches_pr63_layout() -> None:
    """PR63 wire format: flat uint16[n_frames * 6] = 12 bytes/frame."""
    poses = np.full((50, 6), 20.0, dtype=np.float32)
    poses[:, 1:] = 0.0
    s = encode_qpose14_uint16_view_int16(poses)
    assert len(s.payload) == 50 * 6 * 2


def test_pr63_codec_round_trip_negative_signed_columns() -> None:
    """Cols 1-5 may be negative (stored as int16 view in uint16 slot)."""
    poses = np.array(
        [
            [20.0, -0.5, 0.5, -0.25, 0.25, 0.0],
            [25.0, 0.5, -0.5, 0.25, -0.25, 1.0],
        ],
        dtype=np.float32,
    )
    s = encode_qpose14_uint16_view_int16(poses)
    r = decode_qpose14_uint16_view_int16(s.payload, n_frames=2)
    np.testing.assert_allclose(r[:, 0], poses[:, 0], atol=1.5 / 512.0)
    np.testing.assert_allclose(r[:, 1:], poses[:, 1:], atol=1.5 / 2048.0)


def test_pr63_codec_encode_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        encode_qpose14_uint16_view_int16(np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="must have 6 dims"):
        encode_qpose14_uint16_view_int16(np.zeros((4, 5), dtype=np.float32))


def test_pr63_codec_encode_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        encode_qpose14_uint16_view_int16(np.zeros((0, 6), dtype=np.float32))


def test_pr63_codec_encode_rejects_col0_below_bias() -> None:
    """col-0 < 20.0 makes q[:, 0] negative which doesn't fit uint16."""
    poses = np.array([[19.5, 0, 0, 0, 0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="col-0 velocity"):
        encode_qpose14_uint16_view_int16(poses)


def test_pr63_codec_encode_rejects_signed_col_out_of_int16() -> None:
    """Signed-col values > 16.0 (= 2048 * 0.0005 * 16) exceed int16."""
    poses = np.array([[20.0, 100.0, 0, 0, 0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="signed-col"):
        encode_qpose14_uint16_view_int16(poses)


def test_pr63_codec_decode_rejects_bytes_type() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        decode_qpose14_uint16_view_int16("not bytes", n_frames=1)  # type: ignore[arg-type]


def test_pr63_codec_decode_rejects_zero_n_frames() -> None:
    with pytest.raises(ValueError, match="n_frames must be > 0"):
        decode_qpose14_uint16_view_int16(b"\x00" * 12, n_frames=0)


def test_pr63_codec_decode_rejects_truncated_payload() -> None:
    with pytest.raises(ValueError, match="payload length"):
        decode_qpose14_uint16_view_int16(b"\x00" * 6, n_frames=2)


def test_pr63_codec_decode_rejects_oversized_payload() -> None:
    with pytest.raises(ValueError, match="payload length"):
        decode_qpose14_uint16_view_int16(b"\x00" * (4 * 6 * 2), n_frames=2)


# ── PR63 packed-payload single-zip-member grammar ───────────────────────────


def test_pr63_packed_payload_round_trips_three_sections() -> None:
    """PR63 default: 3 sections (mask/model/pose_q)."""
    secs = [b"mask_data_aaa", b"model_data_bbbb", b"pose_data_ccccc"]
    framed = pack_qpose14_packed_payload(secs)
    parsed = unpack_qpose14_packed_payload(framed_bytes(framed, secs))
    assert parsed.n_members == 3
    assert list(parsed.sections) == secs


def framed_bytes(payload_obj: QPose14PackedPayload, sections: list[bytes]) -> bytes:
    """Reconstruct the wire bytes from the encoder-output dataclass."""
    out = bytearray()
    out += struct.pack("<I", payload_obj.n_members)
    for s in sections:
        out += struct.pack("<I", len(s))
        out += s
    return bytes(out)


def test_pr63_packed_payload_total_bytes_matches_layout() -> None:
    secs = [b"a", b"bb", b"ccc"]
    framed = pack_qpose14_packed_payload(secs)
    # Header (4) + 3 * (length-prefix 4 + body) = 4 + (4+1) + (4+2) + (4+3) = 22.
    assert framed.total_bytes == 4 + (4 + 1) + (4 + 2) + (4 + 3)


def test_pr63_packed_payload_supports_custom_n_members() -> None:
    """The primitive supports n_members != 3 (PR63 default)."""
    secs = [b"a", b"b"]
    framed = pack_qpose14_packed_payload(secs, expected_n_members=2)
    raw = framed_bytes(framed, secs)
    parsed = unpack_qpose14_packed_payload(raw, expected_n_members=2)
    assert parsed.n_members == 2


def test_pr63_packed_payload_rejects_mismatched_n_members() -> None:
    with pytest.raises(ValueError, match="section count"):
        pack_qpose14_packed_payload([b"a", b"b"], expected_n_members=3)


def test_pr63_packed_payload_rejects_invalid_n_members() -> None:
    with pytest.raises(ValueError, match="expected_n_members must be > 0"):
        pack_qpose14_packed_payload([b"a"], expected_n_members=0)


def test_pr63_packed_payload_rejects_non_bytes_section() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        pack_qpose14_packed_payload(["str-is-not-bytes", b"b", b"c"])  # type: ignore[list-item]


def test_pr63_packed_payload_decode_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="truncated header"):
        unpack_qpose14_packed_payload(b"\x00\x00")


def test_pr63_packed_payload_decode_rejects_section_count_mismatch() -> None:
    secs = [b"a", b"b"]
    framed = pack_qpose14_packed_payload(secs, expected_n_members=2)
    raw = framed_bytes(framed, secs)
    with pytest.raises(ValueError, match="n_members 2 != expected_n_members 3"):
        unpack_qpose14_packed_payload(raw, expected_n_members=3)


def test_pr63_packed_payload_decode_rejects_truncated_body() -> None:
    raw = struct.pack("<I", 1) + struct.pack("<I", 100) + b"abc"
    with pytest.raises(ValueError, match="truncated section"):
        unpack_qpose14_packed_payload(raw, expected_n_members=1)


def test_pr63_packed_payload_decode_rejects_trailing_bytes() -> None:
    secs = [b"a"]
    framed = pack_qpose14_packed_payload(secs, expected_n_members=1)
    raw = framed_bytes(framed, secs) + b"\xff"
    with pytest.raises(ValueError, match="trailing bytes"):
        unpack_qpose14_packed_payload(raw, expected_n_members=1)


def test_pr63_packed_payload_dataclass_is_frozen() -> None:
    framed = pack_qpose14_packed_payload([b"a"], expected_n_members=1)
    with pytest.raises((AttributeError, TypeError)):
        framed.n_members = 999  # type: ignore[misc]


# ── Golden vectors ──────────────────────────────────────────────────────────


class TestPR63GoldenVectors:
    def test_qpose14_uint16_int16_golden_vector(self) -> None:
        rng = np.random.default_rng(seed=20260512)
        n_frames = 600
        poses = np.zeros((n_frames, 6), dtype=np.float32)
        poses[:, 0] = 20.0 + rng.uniform(0.0, 10.0, n_frames)
        poses[:, 1:] = rng.uniform(-1.0, 1.0, (n_frames, 5))
        stream = encode_qpose14_uint16_view_int16(poses)
        digest = hashlib.sha256(stream.payload).hexdigest()
        golden = GOLDEN_DIR / "pr63_qpose14_uint16_int16_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR63 qpose14 byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "n_dims": 6,
                        "n_frames": int(n_frames),
                        "payload_len": len(stream.payload),
                        "schema": "pr63_qpose14_uint16_int16.v1",
                        "seed": 20260512,
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            # Also write the input fixture (fp32 little-endian).
            (
                GOLDEN_DIR / "pr63_qpose14_uint16_int16_v1_poses.bin"
            ).write_bytes(poses.astype(np.float32).tobytes())

    def test_qpose14_single_zip_member_golden_vector(self) -> None:
        # Deterministic byte sections — sha-pinned so any wire-format
        # change requires intentional regeneration.
        secs = [
            bytes(range(40)),
            bytes(range(40, 80)),
            bytes(range(80, 120)),
        ]
        framed = pack_qpose14_packed_payload(secs, expected_n_members=3)
        raw = framed_bytes(framed, secs)
        digest = hashlib.sha256(raw).hexdigest()
        golden = GOLDEN_DIR / "pr63_qpose14_single_zip_member_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR63 qpose14 single-zip-member byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "n_members": 3,
                        "payload_len": len(raw),
                        "schema": "pr63_qpose14_single_zip_member.v1",
                        "section_lens": [len(s) for s in secs],
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
