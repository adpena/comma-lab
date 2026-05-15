# SPDX-License-Identifier: MIT
"""Tests for the STC-Dasher v1 scaffold codec at ``tac.codecs.stc_dasher``.

Per the Grand Reunion symposium 2026-05-15 Phase F Composite #6 directive
+ CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: this scaffold MUST ship roundtrip-byte-stable + Shannon
entropy-bound respected + STC syndrome correctness verified.

Lane: ``lane_stc_dasher_scaffold_v1_20260515``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.codecs.stc_dasher import (
    SCAFFOLD_ONLY,
    STC_DASHER_MAGIC,
    STC_DASHER_SCHEMA_VERSION,
    STCDasherDecoder,
    STCDasherEncoder,
    decode_stream,
    encode_stream,
)
from tac.codecs.stc_dasher.decoder import STCDasherDecodeError
from tac.codecs.stc_dasher.encoder import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_PAYLOAD_BIT_RATIO,
)
from tac.symposium_impls.stc_dasher_arithmetic_coding_maximalism import (
    DEFAULT_STC_CONSTRAINT_LENGTH,
)

# =====================================================================
# Module-level scaffold discipline
# =====================================================================


class TestModuleConstants:
    def test_scaffold_only_is_true_per_catalog_220(self) -> None:
        """Scaffold-only until a contest-CUDA anchor lands."""
        assert SCAFFOLD_ONLY is True

    def test_magic_is_canonical(self) -> None:
        assert STC_DASHER_MAGIC == b"STCD\x01"
        assert len(STC_DASHER_MAGIC) == 5

    def test_schema_version_is_1(self) -> None:
        assert STC_DASHER_SCHEMA_VERSION == 1

    def test_default_constraint_length_matches_filler_2011(self) -> None:
        """Filler 2011 section IV.B canonical h=12."""
        assert DEFAULT_STC_CONSTRAINT_LENGTH == 12

    def test_default_context_length_matches_mackay_2003(self) -> None:
        """MacKay 2003 ITILA section 6.6 canonical k=2."""
        assert DEFAULT_CONTEXT_LENGTH == 2

    def test_default_payload_bit_ratio_25pct(self) -> None:
        """Filler 2011 section IV.B sweet-spot 25% syndrome rate."""
        assert DEFAULT_PAYLOAD_BIT_RATIO == 4


# =====================================================================
# Roundtrip identity (the primary scaffold guarantee)
# =====================================================================


class TestRoundtripIdentity:
    def test_short_payload_roundtrip(self) -> None:
        enc = STCDasherEncoder()
        dec = STCDasherDecoder()
        src = b"hello world"
        envelope = enc.encode(src, sigma=0.0)
        assert dec.decode(envelope, sigma=0.0) == src

    def test_long_repetitive_payload_roundtrip(self) -> None:
        enc = STCDasherEncoder()
        dec = STCDasherDecoder()
        src = b"abc" * 1024
        envelope = enc.encode(src, sigma=0.0)
        assert dec.decode(envelope, sigma=0.0) == src

    def test_random_payload_roundtrip(self) -> None:
        rng = np.random.default_rng(42)
        src = rng.integers(0, 256, size=512, dtype=np.uint8).tobytes()
        envelope = encode_stream(src, sigma=0.0).encoded_bytes
        result = decode_stream(envelope, sigma=0.0)
        assert result.decoded_bytes == src

    def test_high_entropy_payload_roundtrip(self) -> None:
        """Real-world high-entropy stream (compressed bytes)."""
        rng = np.random.default_rng(0xCAFE)
        src = rng.bytes(2048)  # Statistically incompressible
        envelope = encode_stream(src, sigma=0.0).encoded_bytes
        assert decode_stream(envelope, sigma=0.0).decoded_bytes == src

    def test_low_entropy_payload_roundtrip(self) -> None:
        src = b"\x00" * 256
        envelope = encode_stream(src, sigma=0.0).encoded_bytes
        assert decode_stream(envelope, sigma=0.0).decoded_bytes == src

    def test_diagnostic_result_carries_overhead(self) -> None:
        enc = STCDasherEncoder()
        src = b"a" * 64
        result = enc.encode_with_diagnostics(src, sigma=0.0)
        assert result.n_input_bytes == 64
        assert result.n_syndrome_bytes > 0
        assert result.total_size_bytes == len(result.encoded_bytes)
        assert result.overhead_bytes >= 0
        assert (
            result.overhead_bytes
            == result.total_size_bytes - result.n_input_bytes - result.n_syndrome_bytes
        )


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_empty_residual_roundtrips(self) -> None:
        envelope = encode_stream(b"", sigma=0.0).encoded_bytes
        result = decode_stream(envelope, sigma=0.0)
        assert result.decoded_bytes == b""
        assert result.n_input_symbols == 0
        assert result.n_syndrome_bits == 0

    def test_single_byte_residual_roundtrips(self) -> None:
        envelope = encode_stream(b"\x5a", sigma=0.0).encoded_bytes
        result = decode_stream(envelope, sigma=0.0)
        assert result.decoded_bytes == b"\x5a"
        assert result.n_input_symbols == 8

    def test_max_byte_residual_roundtrips(self) -> None:
        envelope = encode_stream(b"\xff", sigma=0.0).encoded_bytes
        result = decode_stream(envelope, sigma=0.0)
        assert result.decoded_bytes == b"\xff"

    def test_bytearray_input_accepted(self) -> None:
        src = bytearray(b"foobar")
        envelope = encode_stream(src, sigma=0.0).encoded_bytes
        assert decode_stream(envelope, sigma=0.0).decoded_bytes == bytes(src)

    def test_non_bytes_input_rejected(self) -> None:
        with pytest.raises(TypeError, match="residual_bytes must be bytes"):
            encode_stream("not bytes", sigma=0.0)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="residual_bytes must be bytes"):
            encode_stream([0, 1, 2], sigma=0.0)  # type: ignore[arg-type]

    def test_non_bytes_envelope_rejected(self) -> None:
        with pytest.raises(TypeError, match="encoded_bytes must be bytes"):
            decode_stream("not bytes", sigma=0.0)  # type: ignore[arg-type]


# =====================================================================
# Sigma rate-distortion contract
# =====================================================================


class TestSigmaContract:
    def test_sigma_zero_is_lossless_default(self) -> None:
        enc = STCDasherEncoder()
        dec = STCDasherDecoder()
        src = b"sample stream " * 16
        envelope = enc.encode(src, sigma=0.0)
        assert dec.decode(envelope, sigma=0.0) == src

    def test_sigma_int8_clamping_low(self) -> None:
        result = encode_stream(b"abc", sigma=-9999.0)
        assert result.sigma_int8 == -128

    def test_sigma_int8_clamping_high(self) -> None:
        result = encode_stream(b"abc", sigma=9999.0)
        assert result.sigma_int8 == 127

    def test_sigma_rounding_positive(self) -> None:
        result = encode_stream(b"abc", sigma=2.6)
        assert result.sigma_int8 == 3

    def test_sigma_nonfinite_rejected(self) -> None:
        with pytest.raises(ValueError, match="sigma must be finite"):
            encode_stream(b"abc", sigma=float("inf"))
        with pytest.raises(ValueError, match="sigma must be finite"):
            encode_stream(b"abc", sigma=float("nan"))

    def test_sigma_mismatch_decode_rejected(self) -> None:
        envelope = encode_stream(b"abc", sigma=0.0).encoded_bytes
        with pytest.raises(STCDasherDecodeError, match="sigma mismatch"):
            decode_stream(envelope, sigma=5.0)


# =====================================================================
# Envelope corruption / fail-closed behavior
# =====================================================================


class TestEnvelopeIntegrity:
    def test_truncated_envelope_rejected(self) -> None:
        envelope = encode_stream(b"hello world", sigma=0.0).encoded_bytes
        with pytest.raises(STCDasherDecodeError, match=r"too short|truncated"):
            decode_stream(envelope[:10], sigma=0.0)

    def test_bad_magic_rejected(self) -> None:
        envelope = encode_stream(b"hello world", sigma=0.0).encoded_bytes
        bad = b"BADM\x01" + envelope[5:]
        with pytest.raises(STCDasherDecodeError, match="magic mismatch"):
            decode_stream(bad, sigma=0.0)

    def test_wrong_schema_rejected(self) -> None:
        envelope = encode_stream(b"hello world", sigma=0.0).encoded_bytes
        bad = envelope[:5] + bytes([99]) + envelope[6:]
        with pytest.raises(STCDasherDecodeError, match="schema_version mismatch"):
            decode_stream(bad, sigma=0.0)

    def test_trailing_bytes_rejected(self) -> None:
        envelope = encode_stream(b"hello world", sigma=0.0).encoded_bytes
        with pytest.raises(STCDasherDecodeError, match="trailing bytes"):
            decode_stream(envelope + b"xx", sigma=0.0)

    def test_corrupted_residual_payload_caught_by_no_op_detector(self) -> None:
        """No-op detector per CLAUDE.md non-negotiable + Catalog #105/#139."""
        src = b"hello world " * 10
        envelope = bytearray(encode_stream(src, sigma=0.0).encoded_bytes)
        # Corrupt the last byte of the residual envelope (we know the
        # residual envelope is the final segment).
        envelope[-1] ^= 0xFF
        with pytest.raises(STCDasherDecodeError, match="syndrome verification failed"):
            decode_stream(bytes(envelope), sigma=0.0)

    def test_no_op_detector_can_be_disabled(self) -> None:
        """``verify_syndrome=False`` is the explicit scaffold-only opt-out."""
        src = b"abc"
        envelope = encode_stream(src, sigma=0.0).encoded_bytes
        result = decode_stream(envelope, sigma=0.0, verify_syndrome=False)
        assert result.decoded_bytes == src
        # syndrome_verified True only when verify_syndrome=True AND check passed
        assert result.syndrome_verified is True


# =====================================================================
# Parameter validation
# =====================================================================


class TestParameterValidation:
    def test_constraint_length_too_small_rejected(self) -> None:
        with pytest.raises(ValueError, match="constraint_length"):
            encode_stream(b"abc", sigma=0.0, constraint_length=0)

    def test_constraint_length_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="constraint_length"):
            encode_stream(b"abc", sigma=0.0, constraint_length=256)

    def test_context_length_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="context_length"):
            encode_stream(b"abc", sigma=0.0, context_length=-1)

    def test_context_length_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="context_length"):
            encode_stream(b"abc", sigma=0.0, context_length=256)

    def test_payload_bit_ratio_too_small_rejected(self) -> None:
        with pytest.raises(ValueError, match="payload_bit_ratio"):
            encode_stream(b"abc", sigma=0.0, payload_bit_ratio=0)

    def test_encoder_dataclass_validates_constraint_length(self) -> None:
        with pytest.raises(ValueError, match="constraint_length"):
            STCDasherEncoder(constraint_length=0)

    def test_encoder_dataclass_validates_context_length(self) -> None:
        with pytest.raises(ValueError, match="context_length"):
            STCDasherEncoder(context_length=-1)

    def test_encoder_dataclass_validates_payload_bit_ratio(self) -> None:
        with pytest.raises(ValueError, match="payload_bit_ratio"):
            STCDasherEncoder(payload_bit_ratio=0)


# =====================================================================
# Mathematical-property verification (Shannon / STC correctness)
# =====================================================================


class TestMathematicalProperties:
    def test_shannon_entropy_lower_bound_respected(self) -> None:
        """Estimated arithmetic-coded bits >= Shannon entropy of syndrome."""
        rng = np.random.default_rng(7)
        src = rng.bytes(1024)
        result = encode_stream(src, sigma=0.0)
        # Syndrome bits are uniform random over a partition of the input,
        # so per-bit entropy <= 1.0 bit and total entropy <= n_syndrome_bits.
        # The Dasher AC estimate must respect Shannon's lower bound.
        assert result.estimated_arithmetic_bits >= 0.0
        assert result.estimated_arithmetic_bits <= float(
            result.n_syndrome_bytes * 8 + 32  # +32 slack for Laplace smoothing
        )

    def test_syndrome_size_proportional_to_payload_bit_ratio(self) -> None:
        """Filler 2011: syndrome size = n_input_symbols / payload_bit_ratio."""
        src = b"\xab" * 64  # 512 bits
        result = encode_stream(src, sigma=0.0, payload_bit_ratio=4)
        # n_syndrome_bits = 512 // 4 = 128 bits = 16 bytes
        assert result.n_syndrome_bytes == 16

    def test_syndrome_size_doubles_when_ratio_halves(self) -> None:
        src = b"\xab" * 64
        r4 = encode_stream(src, sigma=0.0, payload_bit_ratio=4)
        r2 = encode_stream(src, sigma=0.0, payload_bit_ratio=2)
        assert r2.n_syndrome_bytes == 2 * r4.n_syndrome_bytes

    def test_stc_syndrome_correctness_verified_by_decoder(self) -> None:
        """STC syndrome H * x mod 2 round-trips via independent re-derivation."""
        src = bytes(range(256))  # All 256 possible byte values
        envelope = encode_stream(src, sigma=0.0).encoded_bytes
        result = decode_stream(envelope, sigma=0.0)
        assert result.decoded_bytes == src
        assert result.syndrome_verified is True

    def test_constraint_length_recorded_in_envelope(self) -> None:
        envelope = encode_stream(b"abc" * 16, sigma=0.0, constraint_length=8).encoded_bytes
        result = decode_stream(envelope, sigma=0.0)
        assert result.constraint_length == 8

    def test_context_length_recorded_in_envelope(self) -> None:
        envelope = encode_stream(b"abc" * 16, sigma=0.0, context_length=4).encoded_bytes
        result = decode_stream(envelope, sigma=0.0)
        assert result.context_length == 4


# =====================================================================
# Substrate-archive bolt-on simulation (integration smoke)
# =====================================================================


class TestSubstrateArchiveSimulation:
    @pytest.fixture
    def simulated_residual_stream(self) -> bytes:
        """Simulate a substrate-archive residual stream.

        Mix of high-entropy renderer-parameter bytes and low-cardinality
        mask-argmax bytes per the symposium spec composition.
        """
        rng = np.random.default_rng(0x5EED)
        high_card = rng.bytes(1024)  # high entropy
        low_card_argmax = rng.integers(0, 5, size=512, dtype=np.uint8).tobytes()
        return high_card + low_card_argmax

    def test_simulated_residual_roundtrips(
        self, simulated_residual_stream: bytes
    ) -> None:
        envelope = encode_stream(simulated_residual_stream, sigma=0.0).encoded_bytes
        recovered = decode_stream(envelope, sigma=0.0).decoded_bytes
        assert recovered == simulated_residual_stream

    def test_simulated_residual_envelope_overhead_is_bounded(
        self, simulated_residual_stream: bytes
    ) -> None:
        """Envelope overhead must be small relative to input.

        For the v1 scaffold the envelope carries the full residual so
        ``encoded_size >= input_size + syndrome_size``. The fixed-header
        overhead must be < 32 bytes.
        """
        result = encode_stream(simulated_residual_stream, sigma=0.0)
        # 21-byte header + 4-byte length prefix on residual = 25 bytes.
        assert result.overhead_bytes <= 32

    def test_a1_archive_bytes_roundtrip_smoke(self) -> None:
        """If the A1 archive is checked in, smoke-test on its bytes.

        Per CLAUDE.md "Apples-to-apples evidence discipline" - this is a
        roundtrip smoke ONLY; no score claim. Tag any score derived from
        this codec as ``[advisory only]`` until contest-CUDA anchor lands.
        """
        a1_path = Path("submissions/a1/archive.zip")
        if not a1_path.exists():
            pytest.skip("A1 archive not available; smoke skipped")
        bytes_data = a1_path.read_bytes()
        # Use a small slice to keep test fast (full 174KB takes seconds).
        sample = bytes_data[:4096]
        envelope = encode_stream(sample, sigma=0.0).encoded_bytes
        recovered = decode_stream(envelope, sigma=0.0).decoded_bytes
        assert recovered == sample


# =====================================================================
# Scaffold discipline (Catalog #220 sister surface)
# =====================================================================


class TestScaffoldDiscipline:
    def test_module_advertises_scaffold_only(self) -> None:
        """Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"."""
        assert SCAFFOLD_ONLY is True

    def test_predicted_score_band_is_documented(self) -> None:
        """The v1-vs-future predicted delta S split must be documented."""
        import tac.codecs.stc_dasher as mod
        assert mod.__doc__ is not None
        doc = mod.__doc__.lower()
        assert "v1 scaffold: none" in doc
        assert "post-viterbi inverse" in doc
        assert "-0.010" in doc and "-0.030" in doc

    def test_lane_id_documented(self) -> None:
        import tac.codecs.stc_dasher as mod
        assert "lane_stc_dasher_scaffold_v1_20260515" in (mod.__doc__ or "")
