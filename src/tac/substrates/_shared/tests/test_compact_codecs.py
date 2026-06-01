# SPDX-License-Identifier: MIT
"""Tests for shared compact decoder and integer-stream codecs."""

from __future__ import annotations

import numpy as np
import torch

from tac.substrates._shared.decoder_state_codec import (
    _decode_int8_record,
    _decode_nbit_record,
    _encode_int8_record,
    _encode_nbit_record,
    decoder_state_codec_stats,
    deserialize_decoder_state_dict,
    serialize_decoder_state_dict,
)
from tac.substrates._shared.int_stream_codec import (
    decode_uint_stream,
    encode_uint_stream,
    int_stream_codec_stats,
    pack_fixed_width_uints,
    unpack_fixed_width_uints,
)


def test_decoder_state_codec_int8_roundtrips_shapes_and_metadata() -> None:
    torch.manual_seed(0)
    state = {
        "conv.weight": torch.randn(6, 3, 3, 3) * 0.02,
        "conv.bias": torch.randn(6) * 0.01,
    }
    blob = serialize_decoder_state_dict(state, codec="int8_mixed")
    stats = decoder_state_codec_stats(blob)
    decoded = deserialize_decoder_state_dict(blob)

    assert stats.codec == "int8_mixed"
    assert stats.envelope_bytes == len(blob)
    assert set(decoded) == set(state)
    assert decoded["conv.weight"].shape == state["conv.weight"].shape
    assert decoded["conv.bias"].shape == state["conv.bias"].shape
    assert torch.isfinite(decoded["conv.weight"]).all()


def test_decoder_state_codec_int4_and_int2_bitpacked_roundtrip_shapes() -> None:
    torch.manual_seed(1)
    state = {"w": torch.randn(5, 4) * 0.03}
    for codec in ("int4_mixed", "int2_mixed"):
        blob = serialize_decoder_state_dict(state, codec=codec)
        stats = decoder_state_codec_stats(blob)
        decoded = deserialize_decoder_state_dict(blob)
        assert stats.codec == codec
        assert decoded["w"].shape == state["w"].shape
        assert torch.isfinite(decoded["w"]).all()


def test_decoder_state_codec_scale_bundled_roundtrips_shapes_and_metadata() -> None:
    torch.manual_seed(2)
    state = {
        "conv.weight": torch.randn(8, 3, 3, 3)
        * torch.tensor([0.001, 0.2, 0.003, 0.1, 0.002, 0.4, 0.004, 0.05]).reshape(
            8, 1, 1, 1
        ),
        "conv.bias": torch.randn(8) * 0.01,
    }
    for codec in ("int8_scale_bundled", "int4_scale_bundled", "int2_scale_bundled"):
        blob = serialize_decoder_state_dict(state, codec=codec)
        stats = decoder_state_codec_stats(blob)
        decoded = deserialize_decoder_state_dict(blob)
        assert stats.codec == codec
        assert stats.envelope_bytes == len(blob)
        assert decoded["conv.weight"].shape == state["conv.weight"].shape
        assert decoded["conv.bias"].shape == state["conv.bias"].shape
        assert torch.isfinite(decoded["conv.weight"]).all()


def test_decoder_state_codec_portfolio_auto_beats_or_matches_int8_modes() -> None:
    torch.manual_seed(3)
    state = {
        "conv.weight": torch.randn(16, 3, 3, 3)
        * torch.linspace(0.001, 0.4, steps=16).reshape(16, 1, 1, 1),
        "conv.bias": torch.randn(16) * 0.01,
    }

    auto = serialize_decoder_state_dict(state, codec="portfolio_auto")
    int8_plain = serialize_decoder_state_dict(state, codec="int8_mixed")
    int8_bundled = serialize_decoder_state_dict(state, codec="int8_scale_bundled")
    decoded = deserialize_decoder_state_dict(auto)

    assert len(auto) <= min(len(int8_plain), len(int8_bundled))
    assert decoder_state_codec_stats(auto).codec in {
        "int8_mixed",
        "int8_scale_bundled",
        "fp16_enveloped",
    }
    assert decoded["conv.weight"].shape == state["conv.weight"].shape


def test_scale_bundled_records_store_nontrivial_order_and_restore_axis0() -> None:
    tensor = torch.tensor(
        [
            [[1.0, -1.0]],
            [[0.001, -0.001]],
            [[0.1, -0.1]],
            [[0.01, -0.01]],
        ],
        dtype=torch.float32,
    )

    int8_record = _encode_int8_record(tensor, scale_bundled=True)
    assert int8_record["kind"] == "int8_per_channel_axis0_fp16_scale_bundled"
    assert list(int8_record["axis0_order"]) != list(range(tensor.shape[0]))
    assert _decode_int8_record(int8_record).shape == tensor.shape

    for bits in (2, 4):
        nbit_record = _encode_nbit_record(tensor, bits=bits, scale_bundled=True)
        assert nbit_record["kind"] == (
            f"int{bits}_per_channel_axis0_fp16_scale_bundled_bitpacked"
        )
        assert list(nbit_record["axis0_order"]) != list(range(tensor.shape[0]))
        assert _decode_nbit_record(nbit_record, bits=bits).shape == tensor.shape


def test_int_stream_auto_uses_compact_envelope_and_roundtrips() -> None:
    values = np.array([0, 0, 0, 5, 5, 5, 5, 7, 7, 0, 0, 1], dtype=np.int64)
    blob = encode_uint_stream(values, mode="auto", max_value=7)
    stats = int_stream_codec_stats(blob)
    decoded = decode_uint_stream(blob, count=len(values), max_value=7)
    legacy_bytes = values.astype(np.uint16).tobytes()

    assert stats.count == len(values)
    assert stats.envelope_bytes == len(blob)
    assert len(blob) < len(legacy_bytes) + 256
    assert np.array_equal(decoded, values)


def test_fixed_width_uint_packing_supports_int2_and_int4_surfaces() -> None:
    two_bit = np.array([0, 1, 2, 3, 0, 3, 2, 1, 1], dtype=np.int64)
    packed_2 = pack_fixed_width_uints(two_bit, bits=2)
    assert np.array_equal(
        unpack_fixed_width_uints(packed_2, bits=2, count=len(two_bit)),
        two_bit,
    )

    four_bit = np.arange(16, dtype=np.int64)
    packed_4 = pack_fixed_width_uints(four_bit, bits=4)
    assert np.array_equal(
        unpack_fixed_width_uints(packed_4, bits=4, count=len(four_bit)),
        four_bit,
    )
