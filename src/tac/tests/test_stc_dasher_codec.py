# SPDX-License-Identifier: MIT
"""Tests for the STC-Dasher substrate-agnostic codec scaffold."""

from __future__ import annotations

import pytest

from tac.codecs.stc_dasher import (
    SCAFFOLD_ONLY,
    STC_DASHER_MAGIC,
    STC_DASHER_SCHEMA_VERSION,
    STCDasherDecodeError,
    STCDasherDecoder,
    STCDasherEncoder,
    decode_stream,
    encode_stream,
)


def test_module_exports_scaffold_contract() -> None:
    assert SCAFFOLD_ONLY is True
    assert STC_DASHER_MAGIC == b"STCD\x01"
    assert STC_DASHER_SCHEMA_VERSION == 1


def test_encode_decode_stream_roundtrip_verifies_syndrome() -> None:
    payload = (b"hard-pair-residual:" + bytes(range(32))) * 3
    encoded = encode_stream(payload, sigma=0)

    assert encoded.encoded_bytes.startswith(STC_DASHER_MAGIC)
    assert encoded.n_input_bytes == len(payload)
    assert encoded.n_syndrome_bytes > 0
    assert encoded.estimated_arithmetic_bits >= 0.0

    decoded = decode_stream(encoded.encoded_bytes, sigma=0)
    assert decoded.decoded_bytes == payload
    assert decoded.syndrome_verified is True
    assert decoded.n_input_symbols == len(payload) * 8


def test_encode_rejects_non_default_payload_bit_ratio_until_schema_bump() -> None:
    with pytest.raises(ValueError, match="payload_bit_ratio is fixed at 4"):
        encode_stream(b"payload", sigma=0, payload_bit_ratio=3)


def test_stateful_encoder_decoder_roundtrip() -> None:
    payload = b"lane_stc_dasher_scaffold_v1" * 5

    encoded = STCDasherEncoder().encode(payload, sigma=0)
    decoded = STCDasherDecoder().decode(encoded, sigma=0)

    assert decoded == payload


def test_decode_rejects_sigma_mismatch() -> None:
    encoded = encode_stream(b"payload", sigma=0).encoded_bytes

    with pytest.raises(STCDasherDecodeError, match="sigma mismatch"):
        decode_stream(encoded, sigma=1)


def test_decode_rejects_bad_magic() -> None:
    encoded = bytearray(encode_stream(b"payload", sigma=0).encoded_bytes)
    encoded[0:5] = b"BAD!!"

    with pytest.raises(STCDasherDecodeError, match="magic mismatch"):
        decode_stream(bytes(encoded), sigma=0)
