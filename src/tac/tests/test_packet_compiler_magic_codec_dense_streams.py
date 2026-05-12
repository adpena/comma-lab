"""Tests for ``tac.packet_compiler.magic_codec_dense_streams``.

Per CLAUDE.md "Recursive adversarial review protocol" + Catalog #91 (encoder/
decoder dequantisation roundtrip tested) this module covers:

* round-trip parity for each codec choice (brotli / lzma / magic_codec_classic);
* per-stream codec selection (smallest-byte-count default + forced overrides);
* refusal paths (empty bundle, duplicate names, non-ASCII names, > 255 streams,
  no codec accepts the stream);
* envelope corruption detection (truncation, bad magic, unknown codec_id,
  trailing bytes);
* deterministic-bytes guarantee (same input → byte-identical envelope);
* introspection helpers + permanent-False score-claim discipline.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tac.packet_compiler.magic_codec import StreamHint
from tac.packet_compiler.magic_codec_dense_streams import (
    CODEC_BROTLI,
    CODEC_LZMA,
    CODEC_MAGIC_CLASSIC,
    DENSE_STREAMS_VERSION,
    DenseStreamInput,
    DenseStreamsResult,
    MAGIC_DENSE_STREAMS,
    MagicCodecDenseStreamsError,
    decode_magic_codec_dense_streams,
    encode_magic_codec_dense_streams,
    parse_magic_codec_dense_streams_envelope,
)


# ── Round-trip parity ───────────────────────────────────────────────────────


class TestRoundTrip:
    def test_single_brotli_friendly_stream(self):
        # Repeating-pattern int data — brotli should win and round-trip exact.
        values = np.tile(np.array([1, 2, 3, 4], dtype=np.int32), 256)
        streams = [DenseStreamInput("repeat", values, hint=None)]
        result = encode_magic_codec_dense_streams(streams)
        decoded = decode_magic_codec_dense_streams(result.payload)
        assert list(decoded.keys()) == ["repeat"]
        np.testing.assert_array_equal(decoded["repeat"], values)

    def test_three_stream_bundle_residual_latent_hyperprior(self):
        residual = np.array([0, 0, 5, 0, 0, 3, 0, 0, 0, 1] * 50, dtype=np.int32)
        rng = np.random.RandomState(0)
        latent = rng.randn(32, 8).astype(np.float32)
        hyperprior = np.array([0, 0, 1, 0, 0, 2] * 30, dtype=np.int32)
        streams = [
            DenseStreamInput(
                "residual", residual, hint=StreamHint(stream_type="weight_tensor")
            ),
            DenseStreamInput("latent", latent, hint=None),
            DenseStreamInput(
                "hyperprior",
                hyperprior,
                hint=StreamHint(stream_type="weight_tensor"),
            ),
        ]
        result = encode_magic_codec_dense_streams(streams)
        decoded = decode_magic_codec_dense_streams(result.payload)
        assert list(decoded.keys()) == ["residual", "latent", "hyperprior"]
        np.testing.assert_array_equal(decoded["residual"], residual)
        np.testing.assert_array_equal(decoded["latent"], latent)
        np.testing.assert_array_equal(decoded["hyperprior"], hyperprior)

    def test_brotli_only_strategy_round_trips(self):
        rng = np.random.RandomState(1)
        values = rng.randint(-128, 127, size=512, dtype=np.int32)
        streams = [DenseStreamInput("rand", values, hint=None)]
        result = encode_magic_codec_dense_streams(
            streams, selection_strategy="brotli_only"
        )
        for s in result.selections:
            assert s.selected_codec_name == "brotli"
        decoded = decode_magic_codec_dense_streams(result.payload)
        np.testing.assert_array_equal(decoded["rand"], values)

    def test_lzma_only_strategy_round_trips(self):
        values = np.zeros(1024, dtype=np.int32)
        values[::8] = 7
        streams = [DenseStreamInput("zeros_with_spikes", values, hint=None)]
        result = encode_magic_codec_dense_streams(
            streams, selection_strategy="lzma_only"
        )
        for s in result.selections:
            assert s.selected_codec_name == "lzma"
        decoded = decode_magic_codec_dense_streams(result.payload)
        np.testing.assert_array_equal(decoded["zeros_with_spikes"], values)

    def test_magic_classic_only_strategy_round_trips(self):
        values = np.array([0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0], dtype=np.int32)
        hint = StreamHint(stream_type="weight_tensor")
        streams = [DenseStreamInput("sparse", values, hint=hint)]
        result = encode_magic_codec_dense_streams(
            streams, selection_strategy="magic_classic_only"
        )
        assert result.selections[0].selected_codec_name == "magic_codec_classic"
        decoded = decode_magic_codec_dense_streams(result.payload)
        np.testing.assert_array_equal(decoded["sparse"], values)

    def test_float32_2d_array_round_trips(self):
        rng = np.random.RandomState(2)
        latent = rng.randn(16, 32).astype(np.float32)
        streams = [DenseStreamInput("latent_2d", latent, hint=None)]
        result = encode_magic_codec_dense_streams(streams)
        decoded = decode_magic_codec_dense_streams(result.payload)
        np.testing.assert_array_equal(decoded["latent_2d"], latent)
        assert decoded["latent_2d"].dtype == np.float32
        assert decoded["latent_2d"].shape == (16, 32)

    def test_uint8_array_round_trips(self):
        values = np.arange(256, dtype=np.uint8)
        streams = [DenseStreamInput("u8_range", values, hint=None)]
        result = encode_magic_codec_dense_streams(streams)
        decoded = decode_magic_codec_dense_streams(result.payload)
        np.testing.assert_array_equal(decoded["u8_range"], values)
        assert decoded["u8_range"].dtype == np.uint8

    def test_empty_array_round_trips(self):
        values = np.array([], dtype=np.int32)
        streams = [DenseStreamInput("empty", values, hint=None)]
        result = encode_magic_codec_dense_streams(streams)
        decoded = decode_magic_codec_dense_streams(result.payload)
        assert decoded["empty"].shape == (0,)
        assert decoded["empty"].dtype == np.int32


# ── Per-stream codec selection ──────────────────────────────────────────────


class TestSelection:
    def test_zeros_stream_chooses_smallest(self):
        # All-zero stream of int32 — both brotli and lzma should be tiny.
        values = np.zeros(1024, dtype=np.int32)
        streams = [DenseStreamInput("z", values, hint=None)]
        result = encode_magic_codec_dense_streams(streams)
        sel = result.selections[0]
        # Smallest-byte-count chooses brotli or lzma (not magic_classic since
        # hint=None makes magic_classic refused).
        assert sel.selected_codec_id in (CODEC_BROTLI, CODEC_LZMA)
        assert sel.selected_byte_count > 0
        # The selection log records all 3 candidates.
        assert len(sel.candidates) == 3
        magic_classic_candidate = [
            c for c in sel.candidates if c.codec_name == "magic_codec_classic"
        ]
        assert len(magic_classic_candidate) == 1
        assert magic_classic_candidate[0].refused is True

    def test_sparse_stream_with_hint_includes_magic_classic_candidate(self):
        values = np.zeros(1024, dtype=np.int32)
        values[::64] = 7
        hint = StreamHint(stream_type="weight_tensor", is_sparse=True)
        streams = [DenseStreamInput("sparse", values, hint=hint)]
        result = encode_magic_codec_dense_streams(streams)
        sel = result.selections[0]
        magic_classic_candidate = [
            c for c in sel.candidates if c.codec_name == "magic_codec_classic"
        ]
        assert len(magic_classic_candidate) == 1
        # With the hint, magic_classic should NOT be refused on this shape.
        assert magic_classic_candidate[0].refused is False

    def test_brotli_only_omits_other_candidates(self):
        values = np.arange(100, dtype=np.int32)
        streams = [DenseStreamInput("a", values, hint=None)]
        result = encode_magic_codec_dense_streams(
            streams, selection_strategy="brotli_only"
        )
        assert len(result.selections[0].candidates) == 1
        assert result.selections[0].candidates[0].codec_name == "brotli"

    def test_magic_classic_only_refuses_without_hint(self):
        values = np.arange(100, dtype=np.int32)
        streams = [DenseStreamInput("a", values, hint=None)]
        with pytest.raises(
            MagicCodecDenseStreamsError, match="no codec accepted stream"
        ):
            encode_magic_codec_dense_streams(
                streams, selection_strategy="magic_classic_only"
            )


# ── Refusal paths ───────────────────────────────────────────────────────────


class TestRefusalPaths:
    def test_empty_bundle_rejected(self):
        with pytest.raises(
            MagicCodecDenseStreamsError, match=">= 1 stream"
        ):
            encode_magic_codec_dense_streams([])

    def test_duplicate_stream_names_rejected(self):
        values = np.array([1, 2, 3], dtype=np.int32)
        streams = [
            DenseStreamInput("a", values, hint=None),
            DenseStreamInput("a", values, hint=None),
        ]
        with pytest.raises(
            MagicCodecDenseStreamsError, match="duplicate stream name"
        ):
            encode_magic_codec_dense_streams(streams)

    def test_non_ascii_name_rejected(self):
        values = np.array([1, 2, 3], dtype=np.int32)
        streams = [DenseStreamInput("résidu", values, hint=None)]
        with pytest.raises(
            MagicCodecDenseStreamsError, match="ASCII-encodable"
        ):
            encode_magic_codec_dense_streams(streams)

    def test_empty_name_rejected(self):
        values = np.array([1, 2, 3], dtype=np.int32)
        streams = [DenseStreamInput("", values, hint=None)]
        with pytest.raises(
            MagicCodecDenseStreamsError, match="non-empty"
        ):
            encode_magic_codec_dense_streams(streams)

    def test_name_too_long_rejected(self):
        values = np.array([1, 2, 3], dtype=np.int32)
        streams = [DenseStreamInput("x" * 256, values, hint=None)]
        with pytest.raises(
            MagicCodecDenseStreamsError, match="255 ASCII bytes"
        ):
            encode_magic_codec_dense_streams(streams)

    def test_too_many_streams_rejected(self):
        values = np.array([1], dtype=np.int32)
        streams = [DenseStreamInput(f"s{i}", values, hint=None) for i in range(256)]
        with pytest.raises(
            MagicCodecDenseStreamsError, match="<= 255 streams"
        ):
            encode_magic_codec_dense_streams(streams)

    def test_unknown_selection_strategy_rejected(self):
        values = np.array([1, 2, 3], dtype=np.int32)
        streams = [DenseStreamInput("a", values, hint=None)]
        with pytest.raises(
            MagicCodecDenseStreamsError, match="unknown selection_strategy"
        ):
            encode_magic_codec_dense_streams(streams, selection_strategy="boop")


# ── Envelope corruption ─────────────────────────────────────────────────────


class TestEnvelopeCorruption:
    def _ok_payload(self) -> bytes:
        values = np.array([1, 2, 3], dtype=np.int32)
        streams = [DenseStreamInput("a", values, hint=None)]
        return encode_magic_codec_dense_streams(streams).payload

    def test_short_payload_rejected(self):
        with pytest.raises(MagicCodecDenseStreamsError, match="too short"):
            parse_magic_codec_dense_streams_envelope(b"MDS")

    def test_bad_magic_rejected(self):
        with pytest.raises(MagicCodecDenseStreamsError, match="envelope magic"):
            parse_magic_codec_dense_streams_envelope(b"XXXX\x01\x01")

    def test_bad_version_rejected(self):
        bad = b"MDS1" + bytes([99, 0])
        with pytest.raises(
            MagicCodecDenseStreamsError, match="unsupported dense-streams version"
        ):
            parse_magic_codec_dense_streams_envelope(bad)

    def test_truncated_at_inner_bytes_rejected(self):
        payload = self._ok_payload()
        # Remove the last 4 bytes — should land in inner bytes for the only
        # stream.
        truncated = payload[:-4]
        with pytest.raises(MagicCodecDenseStreamsError, match="truncated"):
            decode_magic_codec_dense_streams(truncated)

    def test_unknown_codec_id_rejected(self):
        payload = self._ok_payload()
        # Find the codec_id byte: header(6) + name_len(1) + name("a"=1) = 8.
        forged = bytearray(payload)
        forged[8] = 0xAB  # unknown codec_id
        with pytest.raises(
            MagicCodecDenseStreamsError, match="unknown codec_id"
        ):
            decode_magic_codec_dense_streams(bytes(forged))

    def test_trailing_bytes_rejected(self):
        payload = self._ok_payload()
        with pytest.raises(MagicCodecDenseStreamsError, match="trailing bytes"):
            decode_magic_codec_dense_streams(payload + b"\x00\x00\x00")

    def test_corrupted_brotli_inner_rejected(self):
        # Force a brotli stream then corrupt the inner bytes.
        values = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        streams = [DenseStreamInput("a", values, hint=None)]
        result = encode_magic_codec_dense_streams(
            streams, selection_strategy="brotli_only"
        )
        forged = bytearray(result.payload)
        # Flip a byte deep inside the brotli body (after name + codec + len).
        forged[-1] ^= 0xFF
        with pytest.raises(
            MagicCodecDenseStreamsError, match="brotli decompress failed"
        ):
            decode_magic_codec_dense_streams(bytes(forged))


# ── Determinism ─────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_input_yields_byte_identical_envelope(self):
        rng_state = np.random.RandomState(42).get_state()
        rng = np.random.RandomState()
        rng.set_state(rng_state)
        latent = rng.randn(32, 4).astype(np.float32)
        rng.set_state(rng_state)
        latent_again = rng.randn(32, 4).astype(np.float32)
        np.testing.assert_array_equal(latent, latent_again)
        streams_1 = [DenseStreamInput("l", latent, hint=None)]
        streams_2 = [DenseStreamInput("l", latent_again, hint=None)]
        r1 = encode_magic_codec_dense_streams(streams_1)
        r2 = encode_magic_codec_dense_streams(streams_2)
        assert r1.payload == r2.payload

    def test_score_claims_permanently_false(self):
        values = np.array([1, 2, 3], dtype=np.int32)
        streams = [DenseStreamInput("a", values, hint=None)]
        result = encode_magic_codec_dense_streams(streams)
        assert result.score_claim is False
        assert result.promotion_eligible is False
        assert result.ready_for_exact_eval_dispatch is False
        assert (
            result.target_substrate_hint
            == "any_packetized_archive_with_dense_residual"
        )


# ── Introspection ───────────────────────────────────────────────────────────


class TestIntrospection:
    def test_parse_header_reports_n_streams(self):
        values = np.array([1, 2, 3], dtype=np.int32)
        streams = [
            DenseStreamInput("a", values, hint=None),
            DenseStreamInput("b", values, hint=None),
            DenseStreamInput("c", values, hint=None),
        ]
        result = encode_magic_codec_dense_streams(streams)
        header = parse_magic_codec_dense_streams_envelope(result.payload)
        assert header.version == DENSE_STREAMS_VERSION
        assert header.n_streams == 3

    def test_selection_log_records_every_candidate(self):
        values = np.array([0, 0, 1, 0], dtype=np.int32)
        streams = [
            DenseStreamInput(
                "a", values, hint=StreamHint(stream_type="weight_tensor")
            )
        ]
        result = encode_magic_codec_dense_streams(streams)
        candidates = result.selections[0].candidates
        names = {c.codec_name for c in candidates}
        assert names == {"brotli", "lzma", "magic_codec_classic"}

    def test_envelope_constant_matches_wire(self):
        assert MAGIC_DENSE_STREAMS == b"MDS1"
        assert len(MAGIC_DENSE_STREAMS) == 4

    def test_codec_ids_in_0x60_range(self):
        # CLAUDE.md format_id namespace allocation — dense-streams codecs
        # live in 0x60-0x6F (no collision with magic_codec inner primitives
        # at 0xF0-0xFF or PR106 sidecar 0xFE).
        assert CODEC_BROTLI == 0x60
        assert CODEC_LZMA == 0x61
        assert CODEC_MAGIC_CLASSIC == 0x62

    def test_envelope_header_layout(self):
        values = np.array([1], dtype=np.int32)
        streams = [DenseStreamInput("x", values, hint=None)]
        result = encode_magic_codec_dense_streams(streams)
        assert result.payload[:4] == MAGIC_DENSE_STREAMS
        version, n_streams = struct.unpack_from("<BB", result.payload, 4)
        assert version == DENSE_STREAMS_VERSION
        assert n_streams == 1
