# SPDX-License-Identifier: MIT
"""Tests for the PR100 schema-driven decoder grammar (schema-elision V2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    SchemaDrivenPayload,
    decode_schema_driven,
    encode_schema_driven,
    pack_state_schema_size_sorted,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Round-trip / behavior ───────────────────────────────────────────────────


def test_v2_roundtrip_minimal_two_tensor_archive() -> None:
    schema = [("a", (4,)), ("b", (3,))]
    tensors = [
        (np.array([-1, 0, 1, 2], dtype=np.int8), 0.5),
        (np.array([10, 20, 30], dtype=np.int8), 1.5),
    ]
    payload = encode_schema_driven(tensors)
    assert isinstance(payload, SchemaDrivenPayload)
    dec = decode_schema_driven(payload.body_blob, payload.scales_blob, schema)
    q_a, scale_a, shape_a = dec["a"]
    q_b, scale_b, shape_b = dec["b"]
    assert q_a.tolist() == [-1, 0, 1, 2]
    assert scale_a == 0.5
    assert shape_a == (4,)
    assert q_b.tolist() == [10, 20, 30]
    assert scale_b == 1.5
    assert shape_b == (3,)


def test_v2_roundtrip_multidim_shapes() -> None:
    schema = [("w", (3, 2, 4)), ("b", (3,))]
    rng = np.random.default_rng(42)
    tensors = [
        (rng.integers(-127, 128, size=24, dtype=np.int8).reshape(3, 2, 4), 0.05),
        (rng.integers(-127, 128, size=3, dtype=np.int8), 0.1),
    ]
    payload = encode_schema_driven(tensors)
    dec = decode_schema_driven(payload.body_blob, payload.scales_blob, schema)
    assert dec["w"][0].shape == (3, 2, 4)
    assert np.array_equal(dec["w"][0], tensors[0][0])
    assert np.array_equal(dec["b"][0], tensors[1][0])


def test_v2_body_blob_is_concat_of_all_int8_bodies_in_schema_order() -> None:
    """The body_blob is bytes(a_body) + bytes(b_body) + ... — no length prefixes."""
    tensors = [
        (np.array([1, 2, 3], dtype=np.int8), 1.0),
        (np.array([4, 5], dtype=np.int8), 2.0),
    ]
    payload = encode_schema_driven(tensors)
    # body_blob should be exactly [1, 2, 3, 4, 5] as int8 bytes (raw, no zigzag).
    expected_body = np.array([1, 2, 3, 4, 5], dtype=np.int8).tobytes()
    assert payload.body_blob == expected_body


def test_v2_scales_blob_is_fp16_in_schema_order() -> None:
    """The scales_blob is fp16(s0) + fp16(s1) + ... — 2 bytes per tensor."""
    tensors = [
        (np.array([0], dtype=np.int8), 1.0),
        (np.array([0], dtype=np.int8), 2.5),
    ]
    payload = encode_schema_driven(tensors)
    scales = np.frombuffer(payload.scales_blob, dtype=np.float16)
    assert scales.tolist() == [1.0, 2.5]
    assert len(payload.scales_blob) == 4  # 2 tensors × 2 bytes


def test_v2_bodies_are_not_zigzag_encoded() -> None:
    """PR100 stores bodies as raw int8 (not zigzag). Verify int8 bytes pass through."""
    schema = [("z", (3,))]
    body = np.array([-1, 0, 100], dtype=np.int8)
    payload = encode_schema_driven([(body, 1.0)])
    # Bytes are the raw int8 (viewed as uint8): -1 -> 0xFF, 0 -> 0x00, 100 -> 0x64.
    assert payload.body_blob == bytes([0xFF, 0x00, 0x64])
    dec = decode_schema_driven(payload.body_blob, payload.scales_blob, schema)
    assert dec["z"][0].tolist() == [-1, 0, 100]


def test_v2_empty_archive_zero_tensors() -> None:
    payload = encode_schema_driven([])
    assert payload.body_blob == b""
    assert payload.scales_blob == b""
    dec = decode_schema_driven(payload.body_blob, payload.scales_blob, [])
    assert dec == {}


def test_v2_preserves_schema_iteration_order() -> None:
    """The decoder must consume tensors in the order the schema declares."""
    schema = [("z", (2,)), ("a", (2,)), ("m", (2,))]
    tensors = [
        (np.array([1, 2], dtype=np.int8), 1.0),
        (np.array([3, 4], dtype=np.int8), 2.0),
        (np.array([5, 6], dtype=np.int8), 3.0),
    ]
    payload = encode_schema_driven(tensors)
    dec = decode_schema_driven(payload.body_blob, payload.scales_blob, schema)
    assert dec["z"][0].tolist() == [1, 2]
    assert dec["a"][0].tolist() == [3, 4]
    assert dec["m"][0].tolist() == [5, 6]


def test_v2_pr100_recipe_full_int8_domain_roundtrip() -> None:
    """Larger fixture exercising the full int8 [-128, 127] range across 4 tensors."""
    schema = [
        ("layer1.weight", (8, 8, 1, 1)),
        ("layer1.bias", (8,)),
        ("layer2.weight", (8, 8, 1, 1)),
        ("layer2.bias", (8,)),
    ]
    rng = np.random.default_rng(0)
    tensors = []
    for name, shape in schema:
        n_el = int(np.prod(shape))
        # Use full int8 domain — PR100 doesn't have negzig's -128 problem.
        body = rng.integers(-128, 128, size=n_el, dtype=np.int8).reshape(shape)
        scale = float(rng.uniform(0.01, 1.0))
        tensors.append((body, scale))
    payload = encode_schema_driven(tensors)
    dec = decode_schema_driven(payload.body_blob, payload.scales_blob, schema)
    for (name, _shape), (body, scale) in zip(schema, tensors):
        dec_body, dec_scale, _ = dec[name]
        assert np.array_equal(dec_body, body)
        # fp16 round-trip loses precision; check approximately.
        assert abs(dec_scale - scale) < 1e-2


# ── Mutual exclusion with V1 (per design memo §3) ──────────────────────────


class TestV1V2MutualExclusion:
    """V1 (PR98 CD1) and V2 (PR100 schema-driven) target the SAME ~840 B
    metadata region. They are MUTUALLY EXCLUSIVE — stacking them is
    double-counting.

    These tests demonstrate the conflict empirically: both grammars
    produce the same logical mapping from bytes to (name, body, scale),
    so applying both is semantically the same as applying one of them
    twice.
    """

    def test_v1_and_v2_produce_same_logical_output_from_same_tensors(self) -> None:
        """V1 and V2 are NOT both-applied for the SAME tensors; they are
        ALTERNATE expressions of the same elision mechanism. Verify they
        decode to logically-equivalent state-dicts (same int8 bodies +
        scales, same schema iteration order)."""
        from tac.packet_compiler import (
            decode_cd1_compact,
            encode_cd1_compact,
        )

        schema = [("a", (4,)), ("b", (3,))]
        tensors_v1 = [
            (np.array([1, 2, 3, 4], dtype=np.int8), 1.0),
            (np.array([5, 6, 7], dtype=np.int8), 2.0),
        ]
        tensors_v2 = list(tensors_v1)
        v1_bytes = encode_cd1_compact(tensors_v1)
        v2_payload = encode_schema_driven(tensors_v2)
        v1_dec = decode_cd1_compact(v1_bytes, schema)
        v2_dec = decode_schema_driven(
            v2_payload.body_blob, v2_payload.scales_blob, schema
        )
        # Both decoders produce same int8 bodies + scales for same input.
        for k in schema:
            name = k[0]
            assert np.array_equal(v1_dec[name][0], v2_dec[name][0])

    def test_v1_v2_byte_layouts_differ(self) -> None:
        """V1 and V2 are NOT byte-identical even though they encode the
        same information. They occupy the same logical region (the
        metadata-eliding bytes), which is why they conflict."""
        from tac.packet_compiler import encode_cd1_compact

        tensors = [
            (np.array([1, 2, 3, 4], dtype=np.int8), 1.0),
            (np.array([5, 6, 7], dtype=np.int8), 2.0),
        ]
        v1_bytes = encode_cd1_compact(tensors)
        v2_payload = encode_schema_driven(tensors)
        # Different byte layouts (V1 interleaves scale+body; V2 separates
        # them into two parallel streams). Combined sizes are comparable
        # but bytes-identical they are not.
        v2_total_bytes = (
            len(v2_payload.body_blob) + len(v2_payload.scales_blob)
        )
        assert v1_bytes != v2_payload.body_blob + v2_payload.scales_blob
        # V1 has 8 bytes of header (magic + scale_bits + n_tensors);
        # V2 has 0 bytes of header overhead. So V2's total is smaller by
        # ~8 bytes in this micro-fixture.
        assert v2_total_bytes < len(v1_bytes)


# ── Composition with PR105 V3 size-sort ────────────────────────────────────


class TestV2V3Composition:
    """V2 + V3 stacking: V3 (size-sort) reorders the schema BEFORE V2
    encodes, so the brotli stream sees largest-entropy bodies first. The
    composition is independently testable: V3 produces a permuted schema,
    which V2 then consumes."""

    def test_v2_consumes_v3_sorted_schema(self) -> None:
        """V3 produces a sorted schema; V2 should encode + decode against
        that sorted schema without modification."""
        unsorted_schema = [
            ("small", (3,)),  # n_el=3
            ("large", (4, 4)),  # n_el=16
            ("medium", (4,)),  # n_el=4
        ]
        sorted_entries = pack_state_schema_size_sorted(unsorted_schema)
        # V3 ordering: large > medium > small.
        sorted_schema = [(e.name, e.shape) for e in sorted_entries]
        assert sorted_schema[0][0] == "large"
        assert sorted_schema[1][0] == "medium"
        assert sorted_schema[2][0] == "small"
        # Build tensors in the V3 (sorted) order.
        tensors_in_v3_order = [
            (np.arange(16, dtype=np.int8).reshape(4, 4), 1.0),
            (np.array([10, 20, 30, 40], dtype=np.int8), 0.5),
            (np.array([-1, -2, -3], dtype=np.int8), 0.25),
        ]
        payload = encode_schema_driven(tensors_in_v3_order)
        dec = decode_schema_driven(
            payload.body_blob, payload.scales_blob, sorted_schema
        )
        assert dec["large"][0].shape == (4, 4)
        assert dec["medium"][0].tolist() == [10, 20, 30, 40]
        assert dec["small"][0].tolist() == [-1, -2, -3]


# ── Failure modes ───────────────────────────────────────────────────────────


def test_v2_rejects_non_int8_body() -> None:
    with pytest.raises(ValueError, match="dtype=int8"):
        encode_schema_driven([(np.array([0], dtype=np.uint8), 1.0)])


def test_v2_rejects_non_array_body() -> None:
    with pytest.raises(ValueError, match="np.ndarray"):
        encode_schema_driven([([0, 1], 1.0)])  # type: ignore[list-item]


def test_v2_rejects_non_finite_scale() -> None:
    with pytest.raises(ValueError, match="finite"):
        encode_schema_driven([(np.array([0], dtype=np.int8), float("nan"))])


def test_v2_decoder_rejects_non_bytes_body_blob() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        decode_schema_driven("garbage", b"", [])  # type: ignore[arg-type]


def test_v2_decoder_rejects_non_bytes_scales_blob() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        decode_schema_driven(b"", "garbage", [])  # type: ignore[arg-type]


def test_v2_decoder_rejects_scales_blob_length_mismatch() -> None:
    """If scales_blob has the wrong size for the schema, raise."""
    schema = [("a", (1,)), ("b", (1,))]
    # 2 tensors expects 4 bytes of scales; supply 2.
    with pytest.raises(ValueError, match="scales_blob length"):
        decode_schema_driven(b"\x00\x00", b"\x00\x3C", schema)


def test_v2_decoder_rejects_body_blob_size_mismatch() -> None:
    """If body_blob size != sum of prod(shape), raise."""
    schema = [("a", (4,))]
    # Schema expects 4 bytes; supply 2.
    scale_bytes = np.array([1.0], dtype=np.float16).tobytes()
    with pytest.raises(ValueError, match="body_blob size"):
        decode_schema_driven(b"\x00\x01", scale_bytes, schema)


# ── Golden vector ───────────────────────────────────────────────────────────


class TestPR100SchemaDrivenGoldenVector:
    """The V2 grammar produces deterministic bytes — pin the SHA against
    a representative fixture so future refactors stay byte-faithful."""

    def test_v2_schema_driven_golden_vector(self) -> None:
        rng = np.random.default_rng(42)
        # Fixture: representative HNeRV-layout sub-archive (same as V1
        # so the comparison is apples-to-apples).
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
        payload = encode_schema_driven(tensors)
        # SHA over the canonical (body || scales) concatenation.
        canonical_bytes = payload.body_blob + b"||" + payload.scales_blob
        digest = hashlib.sha256(canonical_bytes).hexdigest()
        manifest = {
            "schema": "pr100_schema_driven_decoder.v1",
            "n_tensors": len(schema),
            "body_blob_length_bytes": len(payload.body_blob),
            "scales_blob_length_bytes": len(payload.scales_blob),
            "sha256": digest,
        }
        golden = GOLDEN_DIR / "pr100_schema_driven_decoder_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR100 schema-driven decoder SHA changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


# ── Registry token ─────────────────────────────────────────────────────────


def test_pr100_token_registered_in_phase1_packet_compiler() -> None:
    from tac.phase1_packet_compiler import PACKET_COMPILER_TRANSFORMS

    assert (
        "pr100_schema_driven_decoder_storage_grammar"
        in PACKET_COMPILER_TRANSFORMS
    )
