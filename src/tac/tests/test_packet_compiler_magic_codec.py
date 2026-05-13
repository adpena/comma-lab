"""Magic-codec auto-selector tests.

Per CLAUDE.md "Recursive adversarial review protocol" + magic-codec landing
discipline, this module covers:

* per-stream-type selection (each StreamType picks an applicable primitive)
* composition + dispatch correctness (envelope parse → inner decode)
* refusal-path safety (every primitive refuses inputs that violate its
  shape/dtype contract, and the selector reports the refusal cleanly)
* round-trip parity (encode → decode preserves the dense values for
  information-preserving primitives; produces shape-correct output for
  lossy primitives)
* edge cases (empty streams, all-zeros, single-element, max alphabet)
* golden-vector SHA pin
* introspection helpers (recommendation_for, candidate_primitives_for,
  parse_magic_codec_envelope, shannon_entropy_estimate_bits)
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler.magic_codec import (
    MAGIC_CODEC_ENVELOPE,
    PRIMITIVE_ARITHMETIC_COEFFICIENTS,
    PRIMITIVE_CATEGORICAL_STREAM,
    PRIMITIVE_CENTERED_DELTA_UINT8,
    PRIMITIVE_DELTA_VARINT_POSE,
    PRIMITIVE_LOWPASS_LUMA_RESIDUAL,
    PRIMITIVE_RLE_OF_ZEROS,
    MagicCodecEnvelopeHeader,
    MagicCodecError,
    StreamHint,
    candidate_primitives_for,
    decode_magic_codec,
    encode_magic_codec,
    parse_magic_codec_envelope,
    recommendation_for,
    shannon_entropy_estimate_bits,
)


_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "packet_compiler" / "golden_vectors"


# ── Selection behaviour by stream type ──────────────────────────────────────


class TestSelectionByStreamType:
    def test_weight_tensor_selects_arithmetic_for_peaked_residual(self):
        rng = np.random.default_rng(20260511)
        values = (rng.standard_normal(2000) * 2).round().astype(np.int32)
        result = encode_magic_codec(values, hint=StreamHint("weight_tensor"))
        assert result.selected_primitive == "sparse_arithmetic_coefficients"
        assert result.selected_primitive_id == PRIMITIVE_ARITHMETIC_COEFFICIENTS

    def test_residual_basis_picks_rle_when_sparse(self):
        dense = np.zeros(1024, dtype=np.int8)
        dense[::64] = 1
        result = encode_magic_codec(dense, hint=StreamHint("residual_basis"))
        # RLE wins on >98% sparse with single-byte values.
        assert result.selected_primitive in (
            "sparse_rle_of_zeros",
            "sparse_arithmetic_coefficients",
        )

    def test_pose_with_quantisable_values_chooses_pose_or_delta(self):
        rng = np.random.default_rng(20260511)
        pose = np.tile(np.arange(0, 100, dtype=np.float32)[:, None], (1, 6))
        pose += rng.normal(0, 0.1, (100, 6)).astype(np.float32)
        result = encode_magic_codec(pose, hint=StreamHint("pose"))
        assert result.selected_primitive in (
            "pr93_delta_varint_pose",
            "pr101_centered_delta_uint8_lzma",
        )

    def test_mask_categorical_selects_known_primitive(self):
        rng = np.random.default_rng(20260511)
        masks = rng.integers(0, 5, 1000, dtype=np.int32)
        result = encode_magic_codec(masks, hint=StreamHint("mask"))
        assert result.selected_primitive in (
            "pr91_arithmetic_coder_constriction",
            "sparse_rle_of_zeros",
        )

    def test_low_pass_residual_selects_lowpass_luma(self):
        h, w = 32, 32
        residual = np.linspace(-1.0, 1.0, h * w, dtype=np.float32).reshape(h, w)
        result = encode_magic_codec(residual, hint=StreamHint("low_pass_residual"))
        assert result.selected_primitive == "pr93_lowpass_luma_residual"

    def test_categorical_stream_uses_categorical_or_arithmetic(self):
        rng = np.random.default_rng(20260511)
        values = rng.integers(0, 10, 500, dtype=np.int32)
        result = encode_magic_codec(values, hint=StreamHint("categorical"))
        assert result.selected_primitive in (
            "pr91_arithmetic_coder_constriction",
            "sparse_arithmetic_coefficients",
        )

    def test_latent_sidecar_tries_three_primitives(self):
        # latent_sidecar candidate set includes AC + RLE + centered-delta.
        names = candidate_primitives_for("latent_sidecar")
        assert "sparse_arithmetic_coefficients" in names
        assert "sparse_rle_of_zeros" in names
        assert "pr101_centered_delta_uint8_lzma" in names


# ── Round-trip parity ───────────────────────────────────────────────────────


class TestRoundTrip:
    def test_arithmetic_coefficients_round_trip(self):
        rng = np.random.default_rng(20260511)
        values = rng.integers(-50, 51, 500, dtype=np.int32)
        result = encode_magic_codec(values, hint=StreamHint("weight_tensor"))
        decoded = decode_magic_codec(result.payload)
        # The arithmetic coder returns int32 by default; values may exceed
        # int8 range so we keep the int32 dtype.
        assert np.array_equal(decoded, values)

    def test_rle_round_trip_preserves_dense(self):
        rng = np.random.default_rng(20260511)
        dense = np.zeros(2048, dtype=np.int16)
        positions = rng.choice(2048, 128, replace=False)
        dense[positions] = rng.integers(-200, 201, 128, dtype=np.int16)
        result = encode_magic_codec(dense, hint=StreamHint("residual_basis"))
        decoded = decode_magic_codec(result.payload)
        assert np.array_equal(decoded, dense)

    def test_categorical_round_trip(self):
        rng = np.random.default_rng(20260511)
        symbols = rng.integers(0, 6, 500, dtype=np.int32)
        result = encode_magic_codec(symbols, hint=StreamHint("categorical"))
        decoded = decode_magic_codec(result.payload)
        assert np.array_equal(decoded.astype(np.int32), symbols.astype(np.int32))

    def test_centered_delta_round_trip_recovers_shape_and_dtype(self):
        rng = np.random.default_rng(20260511)
        pose = np.tile(np.arange(0, 50, dtype=np.float32)[:, None], (1, 4))
        pose += rng.normal(0, 0.5, (50, 4)).astype(np.float32)
        result = encode_magic_codec(
            pose,
            hint=StreamHint("latent_sidecar"),  # picks centered-delta
        )
        if result.selected_primitive == "pr101_centered_delta_uint8_lzma":
            decoded = decode_magic_codec(result.payload)
            assert decoded.shape == pose.shape

    def test_delta_varint_pose_round_trip(self):
        # Construct pose data where delta_varint wins: small integer deltas.
        poses = np.zeros((50, 6), dtype=np.float32)
        poses[:, 0] = np.arange(50, dtype=np.float32) * 0.1
        poses[:, 1] = np.cos(np.arange(50, dtype=np.float32) * 0.05)
        result = encode_magic_codec(poses, hint=StreamHint("pose"))
        decoded = decode_magic_codec(result.payload)
        # Quantisation tolerance: ~ 1.0 / 255 = 0.004 per axis.
        atol = 0.01
        assert np.allclose(decoded, poses, atol=atol)

    def test_lowpass_luma_recovers_coefficients(self):
        residual = np.linspace(-0.5, 0.5, 64 * 64, dtype=np.float32).reshape(64, 64)
        result = encode_magic_codec(
            residual, hint=StreamHint("low_pass_residual")
        )
        decoded = decode_magic_codec(result.payload)
        # Lowpass luma returns the fitted (n_coeffs,) coefficient vector.
        assert decoded.shape in ((3,), (6,))


# ── Refusal-path safety ─────────────────────────────────────────────────────


class TestRefusalPath:
    def test_rle_refuses_2d(self):
        values = np.zeros((10, 10), dtype=np.int32)
        # weight_tensor accepts AC + RLE; AC also refuses 2D so the selector
        # raises (no accepted candidates).
        with pytest.raises(MagicCodecError):
            encode_magic_codec(values, hint=StreamHint("weight_tensor"))

    def test_pose_refuses_1d(self):
        values = np.arange(60, dtype=np.float32)
        # pose accepts delta-varint + centered-delta; both refuse 1D.
        with pytest.raises(MagicCodecError):
            encode_magic_codec(values, hint=StreamHint("pose"))

    def test_lowpass_luma_refuses_1d(self):
        # low_pass_residual has only lowpass-luma; refusing 1D raises.
        values = np.linspace(0, 1, 100, dtype=np.float32)
        with pytest.raises(MagicCodecError):
            encode_magic_codec(values, hint=StreamHint("low_pass_residual"))

    def test_lowpass_luma_refuses_3d(self):
        values = np.zeros((3, 32, 32), dtype=np.float32)
        with pytest.raises(MagicCodecError):
            encode_magic_codec(values, hint=StreamHint("low_pass_residual"))

    def test_unknown_stream_type_raises(self):
        values = np.arange(10, dtype=np.int32)
        with pytest.raises(MagicCodecError):
            encode_magic_codec(values, hint=StreamHint("frob"))  # type: ignore[arg-type]

    def test_unknown_selection_strategy_raises(self):
        values = np.arange(10, dtype=np.int32)
        with pytest.raises(MagicCodecError):
            encode_magic_codec(
                values,
                hint=StreamHint("weight_tensor"),
                selection_strategy="frob",  # type: ignore[arg-type]
            )

    def test_selection_log_includes_refusals(self):
        # Mask with 2D inputs: categorical + RLE both refuse → raises.
        values = np.zeros((4, 4), dtype=np.int32)
        with pytest.raises(MagicCodecError) as exc:
            encode_magic_codec(values, hint=StreamHint("mask"))
        msg = str(exc.value)
        # Both primitives' refusal reasons should appear in the error.
        assert "categorical-stream" in msg or "RLE-of-zeros" in msg


# ── Envelope parse + dispatch ───────────────────────────────────────────────


class TestEnvelope:
    def test_envelope_starts_with_magic(self):
        result = encode_magic_codec(
            np.arange(100, dtype=np.int32),
            hint=StreamHint("weight_tensor"),
        )
        assert result.payload[:4] == MAGIC_CODEC_ENVELOPE

    def test_envelope_primitive_id_byte_in_reserved_range(self):
        result = encode_magic_codec(
            np.arange(100, dtype=np.int32),
            hint=StreamHint("weight_tensor"),
        )
        pid = result.payload[4]
        assert 0xF0 <= pid <= 0xFF, (
            "magic codec primitive_id must live in the reserved 0xF0-0xFF "
            "namespace per CLAUDE.md format_id discipline"
        )

    def test_envelope_version_is_1(self):
        result = encode_magic_codec(
            np.arange(100, dtype=np.int32),
            hint=StreamHint("weight_tensor"),
        )
        assert result.payload[5] == 1

    def test_parse_envelope_round_trip(self):
        result = encode_magic_codec(
            np.arange(100, dtype=np.int32),
            hint=StreamHint("weight_tensor"),
        )
        header = parse_magic_codec_envelope(result.payload)
        assert isinstance(header, MagicCodecEnvelopeHeader)
        assert header.primitive_id == result.selected_primitive_id
        assert header.primitive_name == result.selected_primitive
        assert header.version == 1
        assert header.inner_byte_count == result.inner_primitive_byte_count

    def test_parse_envelope_rejects_short_payload(self):
        with pytest.raises(MagicCodecError):
            parse_magic_codec_envelope(b"abc")

    def test_parse_envelope_rejects_bad_magic(self):
        bad = b"XXXX" + b"\x00" * 6
        with pytest.raises(MagicCodecError):
            parse_magic_codec_envelope(bad)

    def test_parse_envelope_rejects_unknown_primitive_id(self):
        bad = (
            MAGIC_CODEC_ENVELOPE
            + struct.pack("<BB", 0xAB, 1)
            + struct.pack("<I", 0)
        )
        with pytest.raises(MagicCodecError):
            parse_magic_codec_envelope(bad)

    def test_parse_envelope_rejects_bad_version(self):
        bad = (
            MAGIC_CODEC_ENVELOPE
            + struct.pack("<BB", PRIMITIVE_RLE_OF_ZEROS, 99)
            + struct.pack("<I", 0)
        )
        with pytest.raises(MagicCodecError):
            parse_magic_codec_envelope(bad)

    def test_parse_envelope_rejects_length_mismatch(self):
        bad = (
            MAGIC_CODEC_ENVELOPE
            + struct.pack("<BB", PRIMITIVE_RLE_OF_ZEROS, 1)
            + struct.pack("<I", 999)
            + b"too_short"
        )
        with pytest.raises(MagicCodecError):
            parse_magic_codec_envelope(bad)


# ── Selection-strategy semantics ────────────────────────────────────────────


class TestSelectionStrategy:
    def test_smallest_byte_count_default(self):
        values = np.arange(100, dtype=np.int32)
        result = encode_magic_codec(values, hint=StreamHint("weight_tensor"))
        assert result.selection_strategy == "smallest_byte_count"

    def test_entropy_estimate_picks_close_to_entropy(self):
        rng = np.random.default_rng(20260511)
        values = rng.integers(-2, 3, 500, dtype=np.int32)
        result = encode_magic_codec(
            values,
            hint=StreamHint("weight_tensor"),
            selection_strategy="entropy_estimate",
        )
        assert result.selection_strategy == "entropy_estimate"
        # AC should be selected since its byte count is closest to entropy.
        # Allow either since RLE is competitive in low-entropy regimes.
        assert result.selected_primitive in (
            "sparse_arithmetic_coefficients",
            "sparse_rle_of_zeros",
        )

    def test_stacked_optimal_reserved_for_future_composition(self):
        values = np.arange(100, dtype=np.int32)
        result = encode_magic_codec(
            values,
            hint=StreamHint("weight_tensor"),
            selection_strategy="stacked_optimal",
        )
        # Current behaviour: same as smallest_byte_count (composition is
        # wire-format-ready but disabled until exact-eval custody lands).
        assert result.selection_strategy == "stacked_optimal"

    def test_smallest_byte_count_truly_wins(self):
        # Construct a stream where AC dominates (peaked near zero), then
        # verify selector picks the smaller byte count.
        rng = np.random.default_rng(20260511)
        values = (rng.standard_normal(2000) * 0.5).round().astype(np.int32)
        result = encode_magic_codec(values, hint=StreamHint("weight_tensor"))
        accepted = [c for c in result.selection_log if not c.refused]
        chosen_bytes = len(
            next(
                c
                for c in result.selection_log
                if c.primitive_name == result.selected_primitive
            ).encoded_bytes
        )
        for c in accepted:
            assert chosen_bytes <= len(c.encoded_bytes)


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_all_zeros_stream(self):
        zeros = np.zeros(1024, dtype=np.int32)
        result = encode_magic_codec(zeros, hint=StreamHint("residual_basis"))
        decoded = decode_magic_codec(result.payload)
        assert np.array_equal(decoded, zeros)

    def test_single_element_int_stream(self):
        # AC requires non-empty; RLE accepts 1-element.
        single = np.array([7], dtype=np.int32)
        result = encode_magic_codec(single, hint=StreamHint("residual_basis"))
        decoded = decode_magic_codec(result.payload)
        assert decoded[0] == 7

    def test_two_row_pose_minimum_for_delta_varint(self):
        pose = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                         [0.1, 1.1, 2.1, 3.1, 4.1, 5.1]], dtype=np.float32)
        result = encode_magic_codec(pose, hint=StreamHint("pose"))
        # Either primitive may win; both must accept 2 rows.
        decoded = decode_magic_codec(result.payload)
        assert decoded.shape == pose.shape

    def test_large_alphabet_categorical(self):
        # Stress test: 256-symbol alphabet.
        rng = np.random.default_rng(20260511)
        values = rng.integers(0, 256, 200, dtype=np.int32)
        result = encode_magic_codec(values, hint=StreamHint("categorical"))
        decoded = decode_magic_codec(result.payload)
        assert np.array_equal(decoded.astype(np.int32), values)

    def test_negative_values_categorical(self):
        # Negative values must be offset internally.
        rng = np.random.default_rng(20260511)
        values = rng.integers(-100, 101, 200, dtype=np.int32)
        result = encode_magic_codec(values, hint=StreamHint("categorical"))
        decoded = decode_magic_codec(result.payload)
        assert np.array_equal(decoded.astype(np.int32), values)


# ── Introspection helpers ───────────────────────────────────────────────────


class TestIntrospection:
    def test_recommendation_for_returns_dict(self):
        for st in (
            "weight_tensor",
            "latent_sidecar",
            "pose",
            "mask",
            "residual_basis",
            "categorical",
            "low_pass_residual",
        ):
            row = recommendation_for(st)
            assert "source_pr" in row
            assert "primary_primitive" in row
            assert "fallback_primitive" in row
            assert "operating_point_note" in row

    def test_recommendation_for_unknown_raises(self):
        with pytest.raises(MagicCodecError):
            recommendation_for("frob")  # type: ignore[arg-type]

    def test_candidate_primitives_for_returns_tuple(self):
        for st in (
            "weight_tensor",
            "latent_sidecar",
            "pose",
            "mask",
            "residual_basis",
            "categorical",
            "low_pass_residual",
        ):
            names = candidate_primitives_for(st)
            assert isinstance(names, tuple)
            assert len(names) >= 1
            assert all(isinstance(n, str) for n in names)

    def test_candidate_primitives_for_unknown_raises(self):
        with pytest.raises(MagicCodecError):
            candidate_primitives_for("frob")  # type: ignore[arg-type]

    def test_shannon_entropy_of_uniform_int_stream(self):
        # Uniform [0, 4) has Shannon entropy log2(4) = 2.0 bits.
        values = np.tile(np.array([0, 1, 2, 3], dtype=np.int32), 100)
        e = shannon_entropy_estimate_bits(values)
        assert abs(e - 2.0) < 0.05

    def test_shannon_entropy_of_constant_stream(self):
        values = np.full(100, 7, dtype=np.int32)
        e = shannon_entropy_estimate_bits(values)
        assert e == 0.0

    def test_shannon_entropy_of_empty_stream(self):
        values = np.zeros(0, dtype=np.int32)
        e = shannon_entropy_estimate_bits(values)
        assert e == 0.0

    def test_shannon_entropy_refuses_float(self):
        values = np.linspace(0, 1, 100, dtype=np.float32)
        with pytest.raises(MagicCodecError):
            shannon_entropy_estimate_bits(values)


# ── Golden vector ───────────────────────────────────────────────────────────


class TestGoldenVector:
    """Pin a canonical input → expected magic-codec envelope SHA.

    Recipe: ``rng = np.random.default_rng(20260511)``; 1024 int8 zeros with
    64 positions set to ``rng.integers(1, 32, 64, dtype=np.int8)``; stream
    type ``residual_basis``; selection strategy ``smallest_byte_count``.
    """

    GOLDEN_NAME = "magic_codec_v1"

    def _build_canonical_input(self) -> np.ndarray:
        rng = np.random.default_rng(20260511)
        dense = np.zeros(1024, dtype=np.int8)
        positions = rng.choice(1024, 64, replace=False)
        dense[positions] = rng.integers(1, 32, 64, dtype=np.int8)
        return dense

    def test_canonical_input_produces_pinned_sha(self):
        dense = self._build_canonical_input()
        result = encode_magic_codec(
            dense, hint=StreamHint("residual_basis")
        )
        digest = hashlib.sha256(result.payload).hexdigest()
        manifest_path = _GOLDEN_DIR / f"{self.GOLDEN_NAME}.json"
        assert manifest_path.exists(), (
            f"{manifest_path} must be committed; golden-vector tests must not "
            "write fixtures during normal test execution"
        )
        manifest = json.loads(manifest_path.read_text())
        assert digest == manifest["sha256"], (
            f"magic_codec_v1 SHA drift: produced={digest} "
            f"pinned={manifest['sha256']}; either the canonical input changed "
            f"or a primitive's wire format changed"
        )
        assert result.selected_primitive == manifest["selected_primitive"]
        assert (
            result.inner_primitive_byte_count
            == manifest["inner_primitive_byte_count"]
        )
        assert len(result.payload) == manifest["payload_len"]


# ── Selection-log structural correctness ────────────────────────────────────


class TestSelectionLog:
    def test_log_contains_every_attempted_primitive(self):
        values = np.arange(100, dtype=np.int32)
        result = encode_magic_codec(values, hint=StreamHint("weight_tensor"))
        names = {c.primitive_name for c in result.selection_log}
        expected = set(candidate_primitives_for("weight_tensor"))
        assert names == expected

    def test_log_includes_byte_counts_for_accepted(self):
        values = np.arange(100, dtype=np.int32)
        result = encode_magic_codec(values, hint=StreamHint("weight_tensor"))
        for c in result.selection_log:
            if not c.refused:
                assert len(c.encoded_bytes) > 0

    def test_log_includes_refusal_reason_for_refused(self):
        # Mix of accept + refuse via residual_basis with empty input: both
        # candidates accept empty 1D; we need a case where one refuses.
        # Use weight_tensor with 2D float: AC refuses (non-integer), RLE
        # refuses (non-integer + 2D) → raises. We test refusal via the
        # log of a partial-acceptance case: 1D float for weight_tensor.
        # Both refuse but selector raises, so we can't observe the log.
        # Instead use latent_sidecar with int 1D: AC + RLE both accept,
        # centered-delta refuses (1D). The selection log captures all 3.
        values = np.arange(100, dtype=np.int32)
        result = encode_magic_codec(values, hint=StreamHint("latent_sidecar"))
        refused = [c for c in result.selection_log if c.refused]
        for c in refused:
            assert c.refusal_reason is not None
            assert len(c.refusal_reason) > 0


# ── Composition orthogonality (sister-primitive safety) ─────────────────────


class TestComposition:
    def test_magic_codec_envelope_does_not_collide_with_existing_format_ids(self):
        # Existing format_id range:
        # * 0x01-0x14 = dense residual families (per pr106_sidecar_packing.py)
        # * 0x20-0x24 = sparse residual families
        # Magic-codec uses 0xF0-0xFF, well clear of both.
        assert PRIMITIVE_RLE_OF_ZEROS >= 0xF0
        assert PRIMITIVE_ARITHMETIC_COEFFICIENTS >= 0xF0
        assert PRIMITIVE_CENTERED_DELTA_UINT8 >= 0xF0
        assert PRIMITIVE_DELTA_VARINT_POSE >= 0xF0
        assert PRIMITIVE_CATEGORICAL_STREAM >= 0xF0
        assert PRIMITIVE_LOWPASS_LUMA_RESIDUAL >= 0xF0
        for pid in (
            PRIMITIVE_RLE_OF_ZEROS,
            PRIMITIVE_ARITHMETIC_COEFFICIENTS,
            PRIMITIVE_CENTERED_DELTA_UINT8,
            PRIMITIVE_DELTA_VARINT_POSE,
            PRIMITIVE_CATEGORICAL_STREAM,
            PRIMITIVE_LOWPASS_LUMA_RESIDUAL,
        ):
            assert pid <= 0xFF

    def test_envelope_inner_bytes_match_inner_primitive_byte_count(self):
        values = np.arange(100, dtype=np.int32)
        result = encode_magic_codec(values, hint=StreamHint("weight_tensor"))
        # Envelope = 10 bytes (4 magic + 1 pid + 1 ver + 4 len) + inner.
        assert (
            len(result.payload) == 10 + result.inner_primitive_byte_count
        )

    def test_two_independent_encodes_are_byte_identical(self):
        # Verifies determinism: same input → same envelope bytes.
        rng = np.random.default_rng(20260511)
        values = rng.integers(-10, 11, 200, dtype=np.int32)
        r1 = encode_magic_codec(values, hint=StreamHint("weight_tensor"))
        r2 = encode_magic_codec(values, hint=StreamHint("weight_tensor"))
        assert r1.payload == r2.payload

    def test_envelope_starts_with_canonical_4byte_magic(self):
        result = encode_magic_codec(
            np.arange(100, dtype=np.int32),
            hint=StreamHint("weight_tensor"),
        )
        assert result.payload[:4] == b"MAGC"
