"""Tests for the 3 PR101 GOLD primitives ported to ``tac.packet_compiler``.

Covers:

* ``pr101_decoder_storage_order`` — DECODER_STORAGE_ORDER permutation +
  DECODER_STREAM_ENDS split-brotli boundaries.
* ``pr101_conv4_storage_perms`` — CONV4_STORAGE_PERMS per-tensor 4D-axis
  permutation table + auto-computed inverses.
* ``pr101_decoder_byte_maps`` — DECODER_BYTE_MAPS per-tensor sign-encoding
  strategy selector (negzig / zig / twos / off).

Each primitive has 3+ tests covering encode/decode round-trip,
PR101-source-byte-faithful match, schema validation, and golden-vector
SHA parity.

Source: ``src/tac/packet_compiler/pr101_decoder_storage_order.py`` +
``pr101_conv4_storage_perms.py`` + ``pr101_decoder_byte_maps.py``.

[empirical:src/tac/packet_compiler/golden_vectors/pr101_decoder_storage_order_v1.json]
[empirical:src/tac/packet_compiler/golden_vectors/pr101_conv4_storage_perms_v1.json]
[empirical:src/tac/packet_compiler/golden_vectors/pr101_decoder_byte_maps_v1.json]
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    ByteMapStrategy,
    Conv4StoragePermSchema,
    DecoderByteMapsSchema,
    DecoderStorageOrderSchema,
    PR101_CONV4_STORAGE_PERMS,
    PR101_DECODER_BYTE_MAPS,
    PR101_DECODER_STORAGE_ORDER,
    PR101_DECODER_STREAM_ENDS,
    VALID_BYTE_MAP_STRATEGIES,
    apply_inverse_perm,
    apply_storage_perm,
    compute_inverse_perm,
    compute_inverse_perms,
    decode_byte_map,
    encode_byte_map,
    partition_buffer_by_stream_ends,
    reorder_tensors_for_storage,
    restore_tensor_order_from_storage,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ═══════════════════════════════════════════════════════════════════════════
# Primitive 1: DECODER_STORAGE_ORDER
# ═══════════════════════════════════════════════════════════════════════════


class TestDecoderStorageOrder:
    """Tests for the PR101 GOLD decoder-storage-order primitive."""

    # ── Schema construction + validation ─────────────────────────────────

    def test_pr101_anchor_table_constructs_clean(self) -> None:
        """The exact PR101 anchor table must construct without error."""
        schema = DecoderStorageOrderSchema(
            storage_order=PR101_DECODER_STORAGE_ORDER,
            stream_ends=PR101_DECODER_STREAM_ENDS,
            n_tensors=28,
        )
        assert schema.n_tensors == 28
        assert len(schema.storage_order) == 28
        assert len(schema.stream_ends) == 7

    def test_anchor_table_is_a_valid_permutation(self) -> None:
        """PR101's storage_order must be a permutation of range(28)."""
        assert sorted(PR101_DECODER_STORAGE_ORDER) == list(range(28))

    def test_anchor_stream_ends_are_strictly_increasing(self) -> None:
        """PR101's stream_ends: (1, 2, 22, 23, 26, 27, 28) — strictly increasing."""
        for i in range(1, len(PR101_DECODER_STREAM_ENDS)):
            assert (
                PR101_DECODER_STREAM_ENDS[i] > PR101_DECODER_STREAM_ENDS[i - 1]
            )
        assert PR101_DECODER_STREAM_ENDS[-1] == 28

    def test_rejects_non_permutation_storage_order(self) -> None:
        with pytest.raises(ValueError, match="permutation"):
            DecoderStorageOrderSchema(
                storage_order=(0, 0, 2, 3),
                stream_ends=(4,),
                n_tensors=4,
            )

    def test_rejects_storage_order_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length"):
            DecoderStorageOrderSchema(
                storage_order=(0, 1, 2),
                stream_ends=(4,),
                n_tensors=4,
            )

    def test_rejects_non_increasing_stream_ends(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            DecoderStorageOrderSchema(
                storage_order=(0, 1, 2, 3),
                stream_ends=(2, 2, 4),
                n_tensors=4,
            )

    def test_rejects_stream_ends_not_terminating_at_n(self) -> None:
        with pytest.raises(ValueError, match="must equal n_tensors"):
            DecoderStorageOrderSchema(
                storage_order=(0, 1, 2, 3),
                stream_ends=(1, 3),
                n_tensors=4,
            )

    # ── Reorder / restore round-trip ────────────────────────────────────

    def test_reorder_then_restore_is_identity(self) -> None:
        """The encoder/decoder pair must round-trip on any tensor list."""
        schema = DecoderStorageOrderSchema(
            storage_order=(2, 0, 3, 1),
            stream_ends=(2, 4),
            n_tensors=4,
        )
        tensors = (b"AAAA", b"BBB", b"CC", b"D")

        reordered_buffer = reorder_tensors_for_storage(list(tensors), schema)
        # The buffer must equal tensors[2] || tensors[0] || tensors[3] || tensors[1].
        assert reordered_buffer == b"CC" + b"AAAA" + b"D" + b"BBB"

        # Restore: build per-storage-position list, then invert.
        per_storage_tensors = [b"CC", b"AAAA", b"D", b"BBB"]
        restored = restore_tensor_order_from_storage(
            per_storage_tensors, schema
        )
        assert restored == tensors

    def test_reorder_with_pr101_anchor_table_yields_expected_concatenation(
        self,
    ) -> None:
        """Concatenation with PR101 anchor table is correct for a synthetic
        28-tensor fixture."""
        schema = DecoderStorageOrderSchema(
            storage_order=PR101_DECODER_STORAGE_ORDER,
            stream_ends=PR101_DECODER_STREAM_ENDS,
            n_tensors=28,
        )
        # 28 tensors with byte content [b"00", b"01", ..., b"27"]
        tensors = [f"{i:02d}".encode("ascii") for i in range(28)]
        buffer = reorder_tensors_for_storage(tensors, schema)

        # Expected: PR101_DECODER_STORAGE_ORDER applied as index list.
        expected = b"".join(
            tensors[idx] for idx in PR101_DECODER_STORAGE_ORDER
        )
        assert buffer == expected
        assert len(buffer) == 28 * 2

    def test_reorder_rejects_length_mismatch(self) -> None:
        schema = DecoderStorageOrderSchema(
            storage_order=(0, 1, 2),
            stream_ends=(3,),
            n_tensors=3,
        )
        with pytest.raises(ValueError, match="length"):
            reorder_tensors_for_storage([b"a", b"b"], schema)

    def test_restore_rejects_length_mismatch(self) -> None:
        schema = DecoderStorageOrderSchema(
            storage_order=(0, 1, 2),
            stream_ends=(3,),
            n_tensors=3,
        )
        with pytest.raises(ValueError, match="length"):
            restore_tensor_order_from_storage([b"a"], schema)

    # ── Partitioning ────────────────────────────────────────────────────

    def test_partition_matches_pr101_stream_boundary_semantics(self) -> None:
        """Partition: PR101 source contract — 7 streams over the 28-tensor buffer."""
        schema = DecoderStorageOrderSchema(
            storage_order=PR101_DECODER_STORAGE_ORDER,
            stream_ends=PR101_DECODER_STREAM_ENDS,
            n_tensors=28,
        )
        # Each tensor 4 bytes -> 28*4 = 112 bytes total.
        sizes = [4] * 28
        buffer = bytes(range(112))
        segments = partition_buffer_by_stream_ends(buffer, schema, sizes)
        assert len(segments) == 7
        # Boundaries are: 0..1, 1..2, 2..22, 22..23, 23..26, 26..27, 27..28
        # Multiplied by 4 bytes/tensor:
        expected_lens = [1 * 4, 1 * 4, 20 * 4, 1 * 4, 3 * 4, 1 * 4, 1 * 4]
        assert [len(s) for s in segments] == expected_lens
        # Reassembly must equal the input buffer.
        assert b"".join(segments) == buffer

    def test_partition_rejects_sum_mismatch(self) -> None:
        schema = DecoderStorageOrderSchema(
            storage_order=(0, 1),
            stream_ends=(2,),
            n_tensors=2,
        )
        with pytest.raises(ValueError, match="sum"):
            partition_buffer_by_stream_ends(b"AAAA", schema, [4, 4])

    def test_partition_rejects_per_tensor_sizes_length_mismatch(self) -> None:
        schema = DecoderStorageOrderSchema(
            storage_order=(0, 1, 2),
            stream_ends=(3,),
            n_tensors=3,
        )
        with pytest.raises(ValueError, match="length"):
            partition_buffer_by_stream_ends(b"AAAA", schema, [1, 1])

    # ── Golden vector ────────────────────────────────────────────────────

    def test_pr101_decoder_storage_order_golden_vector(self) -> None:
        """Pin the PR101 anchor table + a synthetic 28-tensor concatenation
        SHA-256 so future refactors stay byte-faithful."""
        schema = DecoderStorageOrderSchema(
            storage_order=PR101_DECODER_STORAGE_ORDER,
            stream_ends=PR101_DECODER_STREAM_ENDS,
            n_tensors=28,
        )
        # Deterministic 28-tensor fixture: tensor i = bytes of length (i+1)*2
        # filled with byte value i. Total = sum(2*(i+1) for i in 0..27)
        # = 2 * 28 * 29 / 2 = 812 bytes.
        tensors = [bytes([i]) * (2 * (i + 1)) for i in range(28)]
        buffer = reorder_tensors_for_storage(tensors, schema)
        # Partition along stream_ends boundaries.
        sizes = [len(tensors[idx]) for idx in PR101_DECODER_STORAGE_ORDER]
        segments = partition_buffer_by_stream_ends(buffer, schema, sizes)
        digest = hashlib.sha256(buffer).hexdigest()

        # Persist as golden vector.
        manifest = {
            "schema": "pr101_decoder_storage_order.v1",
            "n_tensors": 28,
            "n_streams": len(PR101_DECODER_STREAM_ENDS),
            "buffer_length": len(buffer),
            "segment_lengths": [len(s) for s in segments],
            "sha256": digest,
            "storage_order": list(PR101_DECODER_STORAGE_ORDER),
            "stream_ends": list(PR101_DECODER_STREAM_ENDS),
        }
        golden = GOLDEN_DIR / "pr101_decoder_storage_order_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR101 decoder-storage-order buffer SHA changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Primitive 2: CONV4_STORAGE_PERMS
# ═══════════════════════════════════════════════════════════════════════════


class TestConv4StoragePerms:
    """Tests for the PR101 GOLD 4D-axis permutation primitive."""

    # ── Schema construction + validation ─────────────────────────────────

    def test_pr101_anchor_table_constructs_clean(self) -> None:
        schema = Conv4StoragePermSchema.from_perms(PR101_CONV4_STORAGE_PERMS)
        assert len(schema.perms) == 13
        assert len(schema.inverse_perms) == 13

    def test_anchor_table_inverses_via_argsort(self) -> None:
        """Auto-computed inverses match np.argsort exactly per PR101 source."""
        schema = Conv4StoragePermSchema.from_perms(PR101_CONV4_STORAGE_PERMS)
        for idx, perm in PR101_CONV4_STORAGE_PERMS.items():
            expected = tuple(int(x) for x in np.argsort(perm))
            assert schema.inverse_perms[idx] == expected, (
                f"inverse mismatch at idx={idx}: {schema.inverse_perms[idx]} "
                f"vs {expected}"
            )

    def test_rejects_non_4tuple_permutation(self) -> None:
        with pytest.raises(ValueError, match="4-tuple"):
            Conv4StoragePermSchema.from_perms({0: (0, 1, 2)})

    def test_rejects_non_permutation(self) -> None:
        with pytest.raises(ValueError, match="permutation"):
            Conv4StoragePermSchema.from_perms({0: (0, 0, 1, 2)})

    def test_rejects_negative_key(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            Conv4StoragePermSchema.from_perms({-1: (0, 1, 2, 3)})

    def test_rejects_non_mapping_input(self) -> None:
        with pytest.raises(TypeError, match="Mapping"):
            Conv4StoragePermSchema.from_perms([(0, (0, 1, 2, 3))])  # type: ignore[arg-type]

    # ── compute_inverse_perm semantics ──────────────────────────────────

    def test_compute_inverse_perm_matches_numpy_argsort(self) -> None:
        """The pure-Python inverse implementation matches np.argsort."""
        for perm in [
            (0, 1, 2, 3),
            (3, 2, 1, 0),
            (3, 0, 2, 1),  # PR101 source
            (0, 1, 3, 2),  # PR101 source
            (1, 0, 2, 3),  # PR101 source
        ]:
            our = compute_inverse_perm(perm)
            np_ref = tuple(int(x) for x in np.argsort(perm))
            assert our == np_ref, f"inverse({perm}) = {our} vs {np_ref}"

    def test_compute_inverse_perm_rejects_non_permutation(self) -> None:
        with pytest.raises(ValueError, match="permutation"):
            compute_inverse_perm((0, 0, 1, 2))

    def test_compute_inverse_perms_returns_frozen_mapping(self) -> None:
        result = compute_inverse_perms({0: (0, 1, 2, 3), 4: (3, 2, 1, 0)})
        # MappingProxyType raises TypeError on mutation.
        with pytest.raises(TypeError):
            result[99] = (0, 1, 2, 3)  # type: ignore[index]

    # ── apply_storage_perm / apply_inverse_perm round-trip ──────────────

    def test_apply_storage_perm_matches_numpy_transpose(self) -> None:
        """Our 4D permutation matches np.transpose(reshape, perm)."""
        shape = (4, 3, 2, 2)
        arr = np.arange(48, dtype=np.uint8)
        for perm in PR101_CONV4_STORAGE_PERMS.values():
            expected = np.transpose(arr.reshape(shape), perm).copy().tobytes()
            actual = apply_storage_perm(arr.tobytes(), shape, perm)
            assert expected == actual, f"perm {perm} disagrees with numpy"

    def test_apply_storage_then_inverse_is_identity(self) -> None:
        """Round-trip on every PR101 permutation."""
        shape = (4, 3, 2, 2)
        arr = bytes(range(48))
        for perm in PR101_CONV4_STORAGE_PERMS.values():
            inv = compute_inverse_perm(perm)
            stored_shape = tuple(shape[i] for i in perm)
            permuted = apply_storage_perm(arr, shape, perm)
            restored = apply_inverse_perm(permuted, stored_shape, inv)
            assert restored == arr, f"roundtrip failed at perm={perm}"

    def test_apply_storage_perm_matches_pr101_decoder_decode_pattern(self) -> None:
        """Verify the (apply -> apply_inverse) pair is byte-equivalent to
        PR101 source decoder: ``q.reshape(stored_shape); np.transpose(q,
        inverse_perm).copy()`` (codec.py line 282-285)."""
        shape = (8, 6, 3, 3)
        arr = np.arange(8 * 6 * 3 * 3, dtype=np.uint8)
        perm = (1, 0, 2, 3)  # PR101 tensor 14 perm
        inv = compute_inverse_perm(perm)
        stored_shape = tuple(shape[i] for i in perm)

        # Encoder: apply_storage_perm flattens to bytes (matches numpy.transpose).
        permuted_bytes = apply_storage_perm(arr.tobytes(), shape, perm)

        # PR101 source decode pattern (codec.py line 282-285):
        #   q = decode_mapped_u8(zz, byte_map)
        #   q = q.reshape(stored_shape)
        #   q = np.transpose(q, inverse_perm).copy()
        pr101_decoded = (
            np.frombuffer(permuted_bytes, dtype=np.uint8)
            .reshape(stored_shape)
        )
        pr101_decoded = np.transpose(pr101_decoded, inv).copy()
        assert pr101_decoded.tobytes() == arr.tobytes()

        # Our apply_inverse_perm with same inputs:
        ours_restored = apply_inverse_perm(permuted_bytes, stored_shape, inv)
        assert ours_restored == arr.tobytes()

    def test_apply_storage_perm_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="prod"):
            apply_storage_perm(b"\x00" * 10, (2, 2, 2, 2), (0, 1, 2, 3))

    # ── Golden vector ────────────────────────────────────────────────────

    def test_pr101_conv4_storage_perms_golden_vector(self) -> None:
        """Pin the PR101 anchor table SHA-256 over its canonical JSON
        serialisation so future refactors stay byte-faithful."""
        # Canonical serialisation: sorted by key, each entry as (perm, inverse).
        schema = Conv4StoragePermSchema.from_perms(PR101_CONV4_STORAGE_PERMS)
        canonical = sorted(
            (k, list(v), list(schema.inverse_perms[k]))
            for k, v in schema.perms.items()
        )
        blob = json.dumps(canonical, sort_keys=False).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()

        manifest = {
            "schema": "pr101_conv4_storage_perms.v1",
            "n_entries": len(schema.perms),
            "indices": sorted(int(k) for k in schema.perms.keys()),
            "sha256": digest,
        }
        golden = GOLDEN_DIR / "pr101_conv4_storage_perms_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR101 conv4-storage-perms canonical SHA changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Primitive 3: DECODER_BYTE_MAPS
# ═══════════════════════════════════════════════════════════════════════════


class TestDecoderByteMaps:
    """Tests for the PR101 GOLD sign-encoding strategy selector primitive."""

    # ── Schema construction + validation ─────────────────────────────────

    def test_pr101_anchor_table_constructs_clean(self) -> None:
        schema = DecoderByteMapsSchema.from_table(PR101_DECODER_BYTE_MAPS)
        assert len(schema.byte_maps) == 4
        assert schema.default_strategy == "zig"

    def test_strategy_for_uses_default_for_unlisted_tensor(self) -> None:
        """Mirrors PR101 source line 279:
        ``DECODER_BYTE_MAPS.get(idx, 'zig')``."""
        schema = DecoderByteMapsSchema.from_table(PR101_DECODER_BYTE_MAPS)
        assert schema.strategy_for(0) == "zig"  # default
        assert schema.strategy_for(5) == "zig"  # default
        assert schema.strategy_for(9) == "negzig"  # PR101 anchor
        assert schema.strategy_for(14) == "negzig"  # PR101 anchor
        assert schema.strategy_for(20) == "twos"  # PR101 anchor
        assert schema.strategy_for(27) == "off"  # PR101 anchor

    def test_strategy_for_rejects_non_int_input(self) -> None:
        schema = DecoderByteMapsSchema.from_table(PR101_DECODER_BYTE_MAPS)
        with pytest.raises(TypeError, match="int"):
            schema.strategy_for("9")  # type: ignore[arg-type]

    def test_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="not in"):
            DecoderByteMapsSchema.from_table({0: "unknown_strategy"})

    def test_rejects_bad_default(self) -> None:
        with pytest.raises(ValueError, match="default_strategy"):
            DecoderByteMapsSchema.from_table(
                {0: "zig"}, default_strategy="garbage"
            )

    def test_rejects_negative_key(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            DecoderByteMapsSchema.from_table({-1: "zig"})

    def test_rejects_non_string_strategy(self) -> None:
        with pytest.raises(ValueError, match="must be str"):
            DecoderByteMapsSchema.from_table({0: 5})  # type: ignore[dict-item]

    def test_valid_strategies_set_is_frozen(self) -> None:
        """The 4 valid strategies must be exactly {zig, negzig, twos, off}."""
        assert VALID_BYTE_MAP_STRATEGIES == frozenset(
            {"zig", "negzig", "twos", "off"}
        )

    # ── encode/decode round-trip ────────────────────────────────────────

    def test_all_4_strategies_round_trip_on_bijective_domain(self) -> None:
        """Every value in the per-strategy bijective domain survives
        encode -> decode.

        Note: ``negzig`` is NOT a bijection over the full int8 range
        because ``-(-128)`` overflows int8 and wraps to ``-128``, so
        both ``-128`` and ``0`` would need to map to the same encoded
        byte ``0`` to round-trip — they cannot. PR101's training
        bounded INT8 quantisation to ``[-127, 127]`` precisely because
        of this edge case (see PR101 source line 234 which casts via
        int16 then back to int8). We test the bijective domain only.
        """
        # `zig` / `twos` / `off` are bijective over full int8 range.
        full_arr = np.arange(-128, 128, dtype=np.int8)
        for strat in ("zig", "twos", "off"):
            enc = encode_byte_map(full_arr, strat)
            assert len(enc) == 256, f"strategy {strat} produced {len(enc)} bytes"
            dec = decode_byte_map(enc, strat)
            assert np.array_equal(full_arr, dec), (
                f"round-trip failed for strategy {strat}"
            )

        # `negzig` is bijective over [-127, 127] only.
        negzig_domain = np.arange(-127, 128, dtype=np.int8)
        enc = encode_byte_map(negzig_domain, "negzig")
        assert len(enc) == 255
        dec = decode_byte_map(enc, "negzig")
        assert np.array_equal(negzig_domain, dec), (
            "negzig round-trip failed on [-127, 127]"
        )

    def test_zig_strategy_matches_pr101_zigzag_semantics(self) -> None:
        """PR101 source line 226-227: zigzag_decode_u8 maps even u8 to +x/2
        and odd u8 to -(x//2)-1."""
        # Standard zigzag: 0 -> 0, 1 -> -1, 2 -> 1, 3 -> -2, 4 -> 2, ...
        arr_u8 = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        decoded = decode_byte_map(arr_u8.tobytes(), "zig")
        assert decoded.tolist() == [0, -1, 1, -2, 2]

    def test_off_strategy_matches_pr101_offset_128_semantics(self) -> None:
        """PR101 source line 236: ``(u8 - 128).astype(int8)`` — 0 -> -128, 128 -> 0, 255 -> 127."""
        arr_u8 = np.array([0, 128, 255], dtype=np.uint8)
        decoded = decode_byte_map(arr_u8.tobytes(), "off")
        assert decoded.tolist() == [-128, 0, 127]

    def test_twos_strategy_matches_pr101_view_semantics(self) -> None:
        """PR101 source line 238: ``arr.view(int8)`` — raw bit reinterpret.
        0 -> 0, 127 -> 127, 128 -> -128, 255 -> -1."""
        arr_u8 = np.array([0, 127, 128, 255], dtype=np.uint8)
        decoded = decode_byte_map(arr_u8.tobytes(), "twos")
        assert decoded.tolist() == [0, 127, -128, -1]

    def test_negzig_strategy_matches_pr101_neg_zigzag_semantics(self) -> None:
        """PR101 source line 234:
        ``(-zigzag_decode_u8(arr).astype(int16)).astype(int8)``.
        u8 input -> zig decode -> negate. So:
        zig(0) = 0 -> -0 = 0
        zig(1) = -1 -> -(-1) = 1
        zig(2) = 1 -> -1
        zig(3) = -2 -> 2
        zig(255) = -128 -> -(-128) overflows int8 -> hits int16 -> casts back.
        PR101 explicitly casts via int16 first so -(-128) = 128 -> 128 cast
        to int8 wraps to -128. We must match that exactly."""
        arr_u8 = np.array([0, 1, 2, 3, 255], dtype=np.uint8)
        decoded = decode_byte_map(arr_u8.tobytes(), "negzig")
        assert decoded.tolist() == [0, 1, -1, 2, -128]

    def test_decode_byte_map_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            decode_byte_map(b"\x00", "garbage")  # type: ignore[arg-type]

    def test_encode_byte_map_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            encode_byte_map(
                np.array([0], dtype=np.int8), "garbage"  # type: ignore[arg-type]
            )

    def test_encode_byte_map_rejects_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="int8"):
            encode_byte_map(np.array([0], dtype=np.int16), "zig")

    def test_decode_byte_map_rejects_non_bytes(self) -> None:
        with pytest.raises(TypeError, match="bytes"):
            decode_byte_map([0, 1, 2], "zig")  # type: ignore[arg-type]

    # ── Golden vector ────────────────────────────────────────────────────

    def test_pr101_decoder_byte_maps_golden_vector(self) -> None:
        """Pin the PR101 anchor table + the 4-strategy decode output over
        a deterministic uint8 fixture so future refactors stay byte-faithful."""
        # Deterministic input: every u8 value 0..255.
        arr_u8 = np.arange(256, dtype=np.uint8)

        # Decode under each strategy; canonical serialisation = sorted by name.
        decoded_per_strategy = {
            strat: list(int(x) for x in decode_byte_map(arr_u8.tobytes(), strat))
            for strat in sorted(VALID_BYTE_MAP_STRATEGIES)
        }

        # Canonical PR101 anchor table representation.
        anchor = sorted(
            (int(k), str(v)) for k, v in PR101_DECODER_BYTE_MAPS.items()
        )
        blob = json.dumps(
            {
                "anchor_table": anchor,
                "decoded_per_strategy": decoded_per_strategy,
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()

        manifest = {
            "schema": "pr101_decoder_byte_maps.v1",
            "n_anchor_entries": len(anchor),
            "n_strategies": len(VALID_BYTE_MAP_STRATEGIES),
            "input_length": 256,
            "sha256": digest,
        }
        golden = GOLDEN_DIR / "pr101_decoder_byte_maps_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR101 decoder-byte-maps canonical SHA changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Integration: PR101 anchor data export hygiene
# ═══════════════════════════════════════════════════════════════════════════


class TestPR101AnchorDataHygiene:
    """Tests that the 3 anchor tables match the PR101 source bytes exactly."""

    def test_pr101_storage_order_anchor_matches_pr101_source(self) -> None:
        """PR101 codec.py:32-35 literal."""
        expected = (
            14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
            18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
        )
        assert PR101_DECODER_STORAGE_ORDER == expected

    def test_pr101_stream_ends_anchor_matches_pr101_source(self) -> None:
        """PR101 codec.py:36 literal."""
        assert PR101_DECODER_STREAM_ENDS == (1, 2, 22, 23, 26, 27, 28)

    def test_pr101_conv4_perms_anchor_matches_pr101_source(self) -> None:
        """PR101 codec.py:38-52 dict literal."""
        expected = {
            2:  (3, 0, 2, 1),
            4:  (3, 0, 2, 1),
            6:  (0, 1, 2, 3),
            8:  (3, 0, 1, 2),
            10: (3, 0, 2, 1),
            12: (3, 0, 1, 2),
            14: (1, 0, 2, 3),
            16: (3, 0, 2, 1),
            18: (1, 0, 2, 3),
            20: (0, 3, 2, 1),
            22: (0, 3, 2, 1),
            24: (0, 2, 3, 1),
            26: (0, 1, 3, 2),
        }
        assert dict(PR101_CONV4_STORAGE_PERMS) == expected

    def test_pr101_byte_maps_anchor_matches_pr101_source(self) -> None:
        """PR101 codec.py:57-62 dict literal."""
        expected = {
            9:  "negzig",
            14: "negzig",
            20: "twos",
            27: "off",
        }
        assert dict(PR101_DECODER_BYTE_MAPS) == expected

    def test_byte_map_strategy_type_alias_is_str(self) -> None:
        """ByteMapStrategy is a Literal[str] for type-check ergonomics."""
        # At runtime the typing literal evaluates to str-ish; we just
        # confirm the imported symbol is usable as a type-hint placeholder
        # (Python does not enforce the value set at runtime).
        sample: ByteMapStrategy = "zig"
        assert sample == "zig"
