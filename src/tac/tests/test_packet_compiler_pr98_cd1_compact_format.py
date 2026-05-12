"""Tests for the PR98 CD1 compact-format primitive (schema-elision V1)."""

from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    CD1CompactFormat,
    CD1_MAGIC,
    SUPPORTED_SCALE_BITS,
    decode_cd1_compact,
    encode_cd1_compact,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Round-trip / behavior ───────────────────────────────────────────────────


def test_cd1_constants_match_pr98_source() -> None:
    """PR98 source line 84: ``if magic != b'CD1'``."""
    assert CD1_MAGIC == b"CD1"
    # PR98 line 96-100: scale_bits in {16, 32}.
    assert SUPPORTED_SCALE_BITS == (16, 32)


def test_cd1_roundtrip_minimal_two_tensor_archive_fp16() -> None:
    schema = [("a", (4,)), ("b", (3,))]
    tensors = [
        (np.array([-1, 0, 1, 2], dtype=np.int8), 0.5),
        (np.array([10, 20, 30], dtype=np.int8), 1.5),
    ]
    enc = encode_cd1_compact(tensors, scale_bits=16)
    dec = decode_cd1_compact(enc, schema)
    assert set(dec.keys()) == {"a", "b"}
    q_a, scale_a, shape_a = dec["a"]
    q_b, scale_b, shape_b = dec["b"]
    assert q_a.tolist() == [-1, 0, 1, 2]
    assert scale_a == 0.5
    assert shape_a == (4,)
    assert q_b.tolist() == [10, 20, 30]
    assert scale_b == 1.5
    assert shape_b == (3,)


def test_cd1_roundtrip_with_fp32_scales() -> None:
    schema = [("a", (4,))]
    tensors = [(np.array([-1, 0, 1, 2], dtype=np.int8), 0.123456789)]
    enc = encode_cd1_compact(tensors, scale_bits=32)
    dec = decode_cd1_compact(enc, schema)
    q, scale, _ = dec["a"]
    # fp32 should preserve the value more accurately than fp16.
    assert abs(scale - 0.123456789) < 1e-6
    assert q.tolist() == [-1, 0, 1, 2]


def test_cd1_roundtrip_multidim_shapes() -> None:
    schema = [("w", (3, 2, 4)), ("b", (3,))]
    rng = np.random.default_rng(42)
    tensors = [
        (rng.integers(-127, 128, size=24, dtype=np.int8), 0.05),
        (rng.integers(-127, 128, size=3, dtype=np.int8), 0.1),
    ]
    enc = encode_cd1_compact(tensors)
    dec = decode_cd1_compact(enc, schema)
    # Bodies decoded back match input (after flatten round-trip).
    assert dec["w"][0].tolist() == tensors[0][0].tolist()
    assert dec["b"][0].tolist() == tensors[1][0].tolist()


def test_cd1_header_layout_matches_pr98_source() -> None:
    """PR98 source layout: b'CD1' + sb(u8) + n_t(u32_le) + per-tensor scale+body."""
    schema = [("x", (2,))]
    tensors = [(np.array([5, -5], dtype=np.int8), 1.0)]
    enc = encode_cd1_compact(tensors, scale_bits=16)
    # First 3 bytes = CD1 magic.
    assert enc[0:3] == b"CD1"
    # Byte 3 = scale_bits.
    assert struct.unpack("<B", enc[3:4])[0] == 16
    # Bytes 4-7 = n_tensors (u32 LE).
    assert struct.unpack("<I", enc[4:8])[0] == 1
    # Bytes 8-9 = fp16 scale.
    fp16_scale = np.frombuffer(enc[8:10], dtype=np.float16)[0]
    assert fp16_scale == 1.0
    # Bytes 10-11 = zigzag body for [5, -5].
    # zigzag(5)=10, zigzag(-5)=9.
    assert enc[10:12] == bytes([10, 9])


def test_cd1_n_tensors_in_header_equals_input_count() -> None:
    schema = [("a", (1,)), ("b", (1,)), ("c", (1,))]
    tensors = [(np.array([i], dtype=np.int8), 1.0) for i in range(3)]
    enc = encode_cd1_compact(tensors)
    n_t = struct.unpack("<I", enc[4:8])[0]
    assert n_t == 3
    dec = decode_cd1_compact(enc, schema)
    assert set(dec.keys()) == {"a", "b", "c"}


def test_cd1_zigzag_body_is_decoded_correctly() -> None:
    """PR98 zigzag: x>=0 -> 2x, x<0 -> -2x-1. Verify decode inverse."""
    schema = [("z", (5,))]
    tensors = [(np.array([0, 1, -1, 2, -2], dtype=np.int8), 1.0)]
    enc = encode_cd1_compact(tensors)
    dec = decode_cd1_compact(enc, schema)
    q, _, _ = dec["z"]
    assert q.tolist() == [0, 1, -1, 2, -2]


def test_cd1_empty_archive_zero_tensors() -> None:
    """An empty archive (n_t=0) is valid — magic + scale_bits + n_t=0."""
    enc = encode_cd1_compact([])
    assert enc[0:3] == b"CD1"
    assert struct.unpack("<I", enc[4:8])[0] == 0
    dec = decode_cd1_compact(enc, [])
    assert dec == {}


def test_cd1_preserves_schema_iteration_order() -> None:
    """The decoder must consume tensors in the order the schema declares."""
    schema = [("z", (2,)), ("a", (2,)), ("m", (2,))]
    tensors = [
        (np.array([1, 2], dtype=np.int8), 1.0),
        (np.array([3, 4], dtype=np.int8), 2.0),
        (np.array([5, 6], dtype=np.int8), 3.0),
    ]
    enc = encode_cd1_compact(tensors)
    dec = decode_cd1_compact(enc, schema)
    assert dec["z"][0].tolist() == [1, 2]
    assert dec["a"][0].tolist() == [3, 4]
    assert dec["m"][0].tolist() == [5, 6]


def test_cd1_iter_is_consumed_exactly_once_by_encoder() -> None:
    """The encoder must materialise the iterable so length is known up front."""
    schema = [("a", (1,)), ("b", (1,))]
    tensors_iter = iter(
        [
            (np.array([1], dtype=np.int8), 1.0),
            (np.array([2], dtype=np.int8), 2.0),
        ]
    )
    enc = encode_cd1_compact(tensors_iter)
    dec = decode_cd1_compact(enc, schema)
    assert dec["a"][0].tolist() == [1]
    assert dec["b"][0].tolist() == [2]


# ── Failure modes ───────────────────────────────────────────────────────────


def test_cd1_rejects_unsupported_scale_bits() -> None:
    with pytest.raises(ValueError, match="scale_bits must be one of"):
        encode_cd1_compact([], scale_bits=8)  # type: ignore[arg-type]


def test_cd1_rejects_non_int8_body() -> None:
    with pytest.raises(ValueError, match="dtype=int8"):
        encode_cd1_compact([(np.array([0], dtype=np.uint8), 1.0)])


def test_cd1_rejects_non_array_body() -> None:
    with pytest.raises(ValueError, match="np.ndarray"):
        encode_cd1_compact([([0, 1], 1.0)])  # type: ignore[list-item]


def test_cd1_rejects_non_finite_scale() -> None:
    with pytest.raises(ValueError, match="finite"):
        encode_cd1_compact([(np.array([0], dtype=np.int8), float("nan"))])
    with pytest.raises(ValueError, match="finite"):
        encode_cd1_compact([(np.array([0], dtype=np.int8), float("inf"))])


def test_cd1_decoder_rejects_bad_magic() -> None:
    bad_bytes = b"XYZ" + b"\x10\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="bad CD1 magic"):
        decode_cd1_compact(bad_bytes, [])


def test_cd1_decoder_rejects_non_bytes_input() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        decode_cd1_compact("garbage", [])  # type: ignore[arg-type]


def test_cd1_decoder_rejects_unsupported_scale_bits_in_header() -> None:
    # Construct a malformed CD1 header with scale_bits=24.
    buf = io.BytesIO()
    buf.write(CD1_MAGIC)
    buf.write(struct.pack("<B", 24))
    buf.write(struct.pack("<I", 0))
    with pytest.raises(ValueError, match="unsupported CD1 scale_bits"):
        decode_cd1_compact(buf.getvalue(), [])


def test_cd1_decoder_rejects_tensor_count_mismatch() -> None:
    """If the schema length doesn't match the encoded n_tensors, raise."""
    schema = [("a", (1,)), ("b", (1,))]  # 2 tensors
    enc = encode_cd1_compact(
        [(np.array([0], dtype=np.int8), 1.0)]
    )  # 1 tensor
    with pytest.raises(ValueError, match="tensor count mismatch"):
        decode_cd1_compact(enc, schema)


def test_cd1_decoder_rejects_truncated_body() -> None:
    """A short payload (missing body bytes) must raise."""
    schema = [("a", (4,))]
    # Encode tensor of size 4, then truncate.
    enc = encode_cd1_compact([(np.array([1, 2, 3, 4], dtype=np.int8), 1.0)])
    truncated = enc[:-2]
    with pytest.raises(ValueError, match="truncated"):
        decode_cd1_compact(truncated, schema)


def test_cd1_decoder_rejects_truncated_scale() -> None:
    """Header parsed but body section can't even read a scale."""
    schema = [("a", (4,))]
    enc = encode_cd1_compact(
        [(np.array([1, 2, 3, 4], dtype=np.int8), 1.0)], scale_bits=32
    )
    # Strip everything after the n_tensors field.
    truncated = enc[:8] + b"\x00"  # only 1 byte where 4 are needed for fp32 scale
    with pytest.raises(ValueError, match="truncated"):
        decode_cd1_compact(truncated, schema)


# ── Golden vector ───────────────────────────────────────────────────────────


class TestPR98CD1CompactFormatGoldenVector:
    """The CD1 grammar produces deterministic bytes — pin the SHA against
    a representative fixture so future refactors stay byte-faithful."""

    def test_cd1_compact_format_golden_vector(self) -> None:
        rng = np.random.default_rng(42)
        # Fixture: representative HNeRV-layout sub-archive.
        schema = [
            ("blocks.0.weight", (4, 3, 3, 3)),
            ("blocks.0.bias", (4,)),
            ("rgb.weight", (3, 4, 1, 1)),
            ("rgb.bias", (3,)),
        ]
        tensors = []
        for name, shape in schema:
            n_el = int(np.prod(shape))
            arr = rng.integers(-127, 128, size=n_el, dtype=np.int8)
            scale = float(rng.uniform(0.001, 0.1))
            tensors.append((arr, scale))
        enc = encode_cd1_compact(tensors, scale_bits=16)
        digest = hashlib.sha256(enc).hexdigest()
        manifest = {
            "schema": "pr98_cd1_compact_format.v1",
            "scale_bits": 16,
            "n_tensors": len(schema),
            "encoded_length_bytes": len(enc),
            "sha256": digest,
        }
        golden = GOLDEN_DIR / "pr98_cd1_compact_format_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR98 CD1 compact-format SHA changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


# ── Registry token ─────────────────────────────────────────────────────────


def test_pr98_token_registered_in_phase1_packet_compiler() -> None:
    from tac.phase1_packet_compiler import PACKET_COMPILER_TRANSFORMS

    assert (
        "pr98_cd1_compact_architecture_ordered_decoder_format"
        in PACKET_COMPILER_TRANSFORMS
    )
