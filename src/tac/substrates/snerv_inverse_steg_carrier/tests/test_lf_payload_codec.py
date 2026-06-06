# SPDX-License-Identifier: MIT
"""Tests for receiver-visible SNeRV LF payload v2 codec."""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_lf_quant_payload,
    encode_lf_quant_payload,
    inspect_lf_quant_payload_header,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec import (
    SNERV_LF_QUANT_V2_MAGIC,
    SnervLfPayloadCodecError,
    decode_lf_quant_payload_v2,
    encode_lf_quant_payload_v2_with_report,
    inspect_lf_quant_payload_v2,
    selected_lf_payload_codec_label,
)


def test_lf_payload_v2_raw_storage_roundtrips_without_mutation() -> None:
    plane = np.array([[0, -1, 2], [-3, 4, -5]], dtype=np.int64)

    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="raw_i64",
        wrapper="none",
    )
    decoded = decode_lf_quant_payload_v2(packet)

    assert packet.startswith(SNERV_LF_QUANT_V2_MAGIC)
    np.testing.assert_array_equal(decoded[0], plane)
    assert report.mode_histogram == {"raw_i64": 1}
    assert report.wrapper_histogram == {"none": 1}
    assert report.score_claim is False
    assert report.ready_for_exact_eval_dispatch is False


def test_lf_payload_v2_zero_heavy_portfolio_beats_raw_i64() -> None:
    plane = np.zeros((64, 64), dtype=np.int64)
    plane[3, 7] = -2
    plane[17, 4] = 1
    plane[45, 51] = -1

    packet, report = encode_lf_quant_payload_v2_with_report([plane])
    decoded = decode_lf_quant_payload_v2(packet)

    np.testing.assert_array_equal(decoded[0], plane)
    assert report.packet_bytes < plane.size * np.dtype("<i8").itemsize
    assert set(report.mode_histogram).issubset(
        {
            "raw_i64",
            "sparse_signed_varint",
            "sparse_unsigned_varint",
            "zigzag_delta_varint",
            "zero_run_varint",
            "unsigned_int2_bitpack",
            "unsigned_int4_bitpack",
            "unsigned_int8_bitpack",
            "unsigned_int2_escape_varint",
            "unsigned_int4_escape_varint",
            "unsigned_int8_escape_varint",
            "signed_int2_bitpack",
            "signed_int4_bitpack",
            "signed_int8_bitpack",
        }
    )


def test_lf_payload_v2_signed_bitpack_exactness_and_range_refusal() -> None:
    plane = np.array([[-2, -1, 0, 1], [1, 0, -1, -2]], dtype=np.int64)
    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="int2",
        wrapper="none",
    )

    np.testing.assert_array_equal(decode_lf_quant_payload_v2(packet)[0], plane)
    assert report.mode_histogram == {"signed_int2_bitpack": 1}

    with pytest.raises(SnervLfPayloadCodecError, match="signed_int2_bitpack"):
        encode_lf_quant_payload_v2_with_report(
            [np.array([[2]], dtype=np.int64)],
            mode="int2",
            wrapper="none",
        )


def test_lf_payload_v2_unsigned_bitpack_exactness_and_range_refusal() -> None:
    plane = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="uint2",
        wrapper="none",
    )

    np.testing.assert_array_equal(decode_lf_quant_payload_v2(packet)[0], plane)
    assert report.mode_histogram == {"unsigned_int2_bitpack": 1}

    with pytest.raises(SnervLfPayloadCodecError, match="unsigned_int2_bitpack"):
        encode_lf_quant_payload_v2_with_report(
            [np.array([[-1]], dtype=np.int64)],
            mode="uint2",
            wrapper="none",
        )
    with pytest.raises(SnervLfPayloadCodecError, match="unsigned_int2_bitpack"):
        encode_lf_quant_payload_v2_with_report(
            [np.array([[4]], dtype=np.int64)],
            mode="uint2",
            wrapper="none",
        )


def test_lf_payload_v2_unsigned_escape_varint_roundtrips_sparse_high_tail() -> None:
    plane = np.zeros((16, 16), dtype=np.int64)
    plane[0, 1] = 1
    plane[2, 3] = 2
    plane[4, 5] = 3
    plane[9, 4] = 111
    plane[15, 0] = 999

    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="uint2_escape",
        wrapper="none",
    )
    decoded = decode_lf_quant_payload_v2(packet)

    np.testing.assert_array_equal(decoded[0], plane)
    assert report.mode_histogram == {"unsigned_int2_escape_varint": 1}
    assert report.packet_bytes < plane.size * np.dtype("<i8").itemsize

    with pytest.raises(SnervLfPayloadCodecError, match="non-negative"):
        encode_lf_quant_payload_v2_with_report(
            [np.array([[-1, 0, 1]], dtype=np.int64)],
            mode="uint2_escape",
            wrapper="none",
        )


def test_lf_payload_v2_unsigned_escape_beats_signed_escape_for_nonnegative_plane() -> None:
    plane = (np.arange(32 * 32, dtype=np.int64).reshape(32, 32) % 4).astype(np.int64)
    plane[9, 4] = 111
    plane[15, 0] = 999

    unsigned_packet, unsigned_report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="uint2_escape",
        wrapper="none",
    )
    signed_packet, signed_report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="int2_escape",
        wrapper="none",
    )

    np.testing.assert_array_equal(decode_lf_quant_payload_v2(unsigned_packet)[0], plane)
    np.testing.assert_array_equal(decode_lf_quant_payload_v2(signed_packet)[0], plane)
    assert unsigned_report.packet_bytes < signed_report.packet_bytes


def test_lf_payload_v2_sparse_signed_varint_roundtrips_scattered_nonzeros() -> None:
    plane = np.zeros((32, 32), dtype=np.int64)
    coords = [(0, 3, -1), (3, 17, 2), (11, 8, -31), (24, 5, 7), (31, 29, -2)]
    for y, x, value in coords:
        plane[y, x] = value

    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="sparse_signed",
        wrapper="none",
    )

    np.testing.assert_array_equal(decode_lf_quant_payload_v2(packet)[0], plane)
    assert report.mode_histogram == {"sparse_signed_varint": 1}
    assert report.packet_bytes < plane.size


def test_lf_payload_v2_sparse_unsigned_varint_roundtrips_and_refuses_negative() -> None:
    plane = np.zeros((32, 32), dtype=np.int64)
    plane[2, 5] = 1
    plane[13, 21] = 18
    plane[28, 7] = 255

    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="sparse_unsigned",
        wrapper="none",
    )

    np.testing.assert_array_equal(decode_lf_quant_payload_v2(packet)[0], plane)
    assert report.mode_histogram == {"sparse_unsigned_varint": 1}
    assert report.packet_bytes < plane.size
    with pytest.raises(SnervLfPayloadCodecError, match="non-negative"):
        encode_lf_quant_payload_v2_with_report(
            [np.array([[0, -1, 2]], dtype=np.int64)],
            mode="sparse_unsigned",
            wrapper="none",
        )


def test_lf_payload_v2_portfolio_can_pick_sparse_bitmask_for_scattered_tiny_support() -> None:
    rng = np.random.default_rng(123)
    plane = np.zeros((128, 128), dtype=np.int64)
    support = rng.choice(plane.size, size=2_000, replace=False)
    values = rng.choice(
        np.array([-3, -2, -1, 1, 2, 3, 17, -33], dtype=np.int64),
        size=support.size,
    )
    plane.reshape(-1)[support] = values

    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        wrapper="none",
    )

    np.testing.assert_array_equal(decode_lf_quant_payload_v2(packet)[0], plane)
    assert report.mode_histogram == {"sparse_signed_varint": 1}
    assert (
        selected_lf_payload_codec_label(
            report.as_jsonable(),
            requested_codec="portfolio_auto",
        )
        == "v2:sparse_signed_varint:none"
    )


def test_selected_lf_payload_codec_label_reports_mixed_receiver_grammar() -> None:
    label = selected_lf_payload_codec_label(
        {
            "schema": "snerv_lf_quant_payload.v2",
            "mode_histogram": {
                "zero_run_varint": 2,
                "sparse_signed_varint": 1,
                "ignored": 0,
            },
            "wrapper_histogram": {"brotli_q9": 1, "none": 2},
        },
        requested_codec="auto",
    )

    assert label == "v2:portfolio:sparse_signed_varint+zero_run_varint:brotli_q9+none"


def test_selected_lf_payload_codec_label_falls_back_for_legacy_report() -> None:
    assert (
        selected_lf_payload_codec_label(
            {"schema": "snerv_lf_quant_payload.v1", "codec": "legacy_lzma"},
            requested_codec="auto",
        )
        == "legacy_lzma"
    )
    assert (
        selected_lf_payload_codec_label(
            {"schema": "snerv_lf_quant_payload.v1"},
            requested_codec="auto",
        )
        == "auto"
    )


def test_lf_payload_v2_portfolio_can_pick_unsigned_escape_for_nonnegative_tail() -> None:
    plane = (np.arange(32 * 32, dtype=np.int64).reshape(32, 32) % 4).astype(np.int64)
    plane[4, 6] = 127
    plane[10, 9] = 255

    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        wrapper="none",
    )
    decoded = decode_lf_quant_payload_v2(packet)

    np.testing.assert_array_equal(decoded[0], plane)
    assert any(mode.startswith("unsigned_int") for mode in report.mode_histogram)


def test_lf_payload_v2_signed_escape_varint_roundtrips_sparse_outliers() -> None:
    plane = np.zeros((16, 16), dtype=np.int64)
    plane[0, 0] = -2
    plane[0, 1] = 1
    plane[3, 5] = 97
    plane[9, 4] = -111

    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="int2_escape",
        wrapper="none",
    )
    decoded = decode_lf_quant_payload_v2(packet)

    np.testing.assert_array_equal(decoded[0], plane)
    assert report.mode_histogram == {"signed_int2_escape_varint": 1}
    assert report.packet_bytes < plane.size * np.dtype("<i8").itemsize


def test_lf_payload_v2_portfolio_can_pick_escape_mode_for_sparse_tail() -> None:
    plane = (np.arange(32 * 32, dtype=np.int64).reshape(32, 32) % 4) - 2
    plane[4, 6] = 127
    plane[10, 9] = -126

    packet, report = encode_lf_quant_payload_v2_with_report(
        [plane],
        wrapper="none",
    )
    decoded = decode_lf_quant_payload_v2(packet)

    np.testing.assert_array_equal(decoded[0], plane)
    assert "signed_int2_escape_varint" in report.mode_histogram


def test_lf_payload_v2_metadata_is_deterministic_and_inspectable() -> None:
    planes = [
        np.arange(-8, 8, dtype=np.int64).reshape(4, 4),
        np.zeros((4, 4), dtype=np.int64),
    ]

    packet_a, report_a = encode_lf_quant_payload_v2_with_report(planes)
    packet_b, _report_b = encode_lf_quant_payload_v2_with_report(planes)
    inspected = inspect_lf_quant_payload_v2(packet_a)

    assert packet_a == packet_b
    assert inspected.schema == report_a.schema
    assert inspected.plane_count == 2
    assert inspected.packet_bytes == len(packet_a)
    assert inspected.promotion_eligible is False
    assert inspected.ready_for_exact_eval_dispatch is False


def test_lf_payload_v2_binary_header_replaces_json_header_without_signal_loss() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec as mod

    planes = [
        (np.arange(48, dtype=np.int64).reshape(6, 8) % 4) - 2,
        np.zeros((6, 8), dtype=np.int64),
    ]
    binary_packet, binary_report = encode_lf_quant_payload_v2_with_report(
        planes,
        mode="portfolio_auto",
        wrapper="none",
        header_format="binary",
    )
    json_packet, json_report = encode_lf_quant_payload_v2_with_report(
        planes,
        mode="portfolio_auto",
        wrapper="none",
        header_format="json",
        allow_legacy_json_header=True,
    )

    assert binary_packet.startswith(SNERV_LF_QUANT_V2_MAGIC)
    assert json_packet.startswith(SNERV_LF_QUANT_V2_MAGIC)
    assert binary_packet[len(SNERV_LF_QUANT_V2_MAGIC)] == mod._BINARY_HEADER_VERSION
    assert json_packet[len(SNERV_LF_QUANT_V2_MAGIC)] == mod._JSON_HEADER_VERSION
    assert binary_report.header_format == "binary"
    assert json_report.header_format == "json"
    assert binary_report.header_bytes < json_report.header_bytes
    assert binary_report.packet_bytes < json_report.packet_bytes
    assert binary_report.mode_histogram == json_report.mode_histogram
    assert binary_report.wrapper_histogram == json_report.wrapper_histogram
    binary_header_len = mod._HEADER.unpack(binary_packet[: mod._HEADER.size])[2]
    binary_header = binary_packet[
        mod._HEADER.size : mod._HEADER.size + binary_header_len
    ]
    assert b"schema" not in binary_header
    assert b"decoded_sha256" not in binary_header

    for ref, got in zip(planes, decode_lf_quant_payload_v2(binary_packet), strict=True):
        np.testing.assert_array_equal(got, ref)
    for ref, got in zip(planes, decode_lf_quant_payload_v2(json_packet), strict=True):
        np.testing.assert_array_equal(got, ref)
    inspected = inspect_lf_quant_payload_v2(binary_packet)
    assert inspected.header_format == "binary"
    assert inspected.header_bytes == binary_report.header_bytes


def test_lf_payload_v2_compact_binary_omits_per_plane_hash_overhead() -> None:
    planes = [
        np.array([[idx % 4]], dtype=np.int64)
        for idx in range(64)
    ]

    compact_packet, compact_report = encode_lf_quant_payload_v2_with_report(
        planes,
        mode="uint2",
        wrapper="none",
    )
    legacy_packet, legacy_report = encode_lf_quant_payload_v2_with_report(
        planes,
        mode="uint2",
        wrapper="none",
        header_format="legacy_binary",
    )

    assert compact_packet[len(SNERV_LF_QUANT_V2_MAGIC)] == 3
    assert legacy_packet[len(SNERV_LF_QUANT_V2_MAGIC)] == 2
    assert compact_report.payload_bytes == legacy_report.payload_bytes
    assert (
        legacy_report.header_bytes - compact_report.header_bytes
        == 32 * (len(planes) - 1)
    )
    assert compact_report.packet_bytes < legacy_report.packet_bytes
    for ref, got in zip(planes, decode_lf_quant_payload_v2(compact_packet), strict=True):
        np.testing.assert_array_equal(got, ref)
    inspected = inspect_lf_quant_payload_v2(compact_packet)
    assert inspected.plane_rows[0].decoded_sha256
    assert inspected.header_bytes == compact_report.header_bytes


def test_lf_payload_v2_compact_binary_aggregate_hash_catches_header_drift() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec as mod

    plane = np.array([[0, 1, 2, 3]], dtype=np.int64)
    packet, _report = encode_lf_quant_payload_v2_with_report(
        [plane],
        mode="uint2",
        wrapper="none",
    )
    mutated = bytearray(packet)
    pos = mod._HEADER.size + len(mod._BINARY_HEADER_MAGIC)
    _plane_count, pos = mod.decode_varint(mutated, pos)
    _payload_bytes, pos = mod.decode_varint(mutated, pos)
    pos += 64
    _height, pos = mod.decode_varint(mutated, pos)
    _width, pos = mod.decode_varint(mutated, pos)
    mutated[pos] = mod._MODE_TO_CODE["signed_int2_bitpack"]

    with pytest.raises(SnervLfPayloadCodecError, match="aggregate sha256 mismatch"):
        decode_lf_quant_payload_v2(bytes(mutated))


def test_lf_payload_v2_json_header_encoding_is_guarded_but_decodable() -> None:
    planes = [
        np.array([[-2, -1], [0, 1]], dtype=np.int64),
        np.array([[1, 0], [-1, -2]], dtype=np.int64),
    ]

    with pytest.raises(SnervLfPayloadCodecError, match="JSON header encoding is blocked"):
        encode_lf_quant_payload_v2_with_report(
            planes,
            mode="int2",
            wrapper="none",
            header_format="json",
        )

    packet, report = encode_lf_quant_payload_v2_with_report(
        planes,
        mode="int2",
        wrapper="none",
        header_format="json",
        allow_legacy_json_header=True,
    )
    decoded = decode_lf_quant_payload_v2(packet)
    inspected = inspect_lf_quant_payload_v2(packet)

    assert packet.startswith(SNERV_LF_QUANT_V2_MAGIC)
    assert packet[len(SNERV_LF_QUANT_V2_MAGIC)] == 1
    assert inspected.packet_bytes == report.packet_bytes
    assert inspected.plane_count == len(planes)
    for ref, got in zip(planes, decoded, strict=True):
        np.testing.assert_array_equal(got, ref)


def test_lf_payload_v2_binary_header_cuts_many_tiny_plane_json_overhead() -> None:
    values = np.array([-2, -1, 0, 1], dtype=np.int64)
    planes = [
        np.array([[values[idx % values.size]]], dtype=np.int64)
        for idx in range(128)
    ]

    binary_packet, binary_report = encode_lf_quant_payload_v2_with_report(
        planes,
        mode="int2",
        wrapper="none",
    )
    json_packet, json_report = encode_lf_quant_payload_v2_with_report(
        planes,
        mode="int2",
        wrapper="none",
        header_format="json",
        allow_legacy_json_header=True,
    )
    decoded = decode_lf_quant_payload_v2(binary_packet)
    inspected = inspect_lf_quant_payload_v2(binary_packet)

    assert binary_packet[len(SNERV_LF_QUANT_V2_MAGIC)] == 3
    assert binary_report.payload_bytes == json_report.payload_bytes
    binary_overhead = binary_report.packet_bytes - binary_report.payload_bytes
    json_overhead = json_report.packet_bytes - json_report.payload_bytes
    assert binary_overhead < json_overhead
    assert binary_overhead < json_overhead // 4
    assert json_report.packet_bytes - binary_report.packet_bytes > 15_000
    assert inspected.packet_bytes == binary_report.packet_bytes
    assert inspected.plane_count == len(planes)
    for ref, got in zip(planes, decoded, strict=True):
        np.testing.assert_array_equal(got, ref)


def test_lf_payload_portfolio_auto_uses_bounded_wrappers_by_default() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec as mod

    wrappers = mod._normalize_wrapper("portfolio_auto")

    assert wrappers == ("none", "brotli_auto", "lzma_auto")
    assert "brotli" not in wrappers
    assert "brotli_q11" not in wrappers
    assert "lzma" not in wrappers
    assert "lzma_extreme" not in wrappers
    assert mod._brotli_quality_for_wrapper("brotli_auto", payload_bytes=0) == 6
    assert mod._brotli_quality_for_wrapper("brotli_auto", payload_bytes=128) == 6
    assert (
        mod._brotli_quality_for_wrapper(
            "brotli_auto",
            payload_bytes=mod.SNERV_LF_BROTLI_AUTO_Q11_MAX_INPUT_BYTES + 1,
        )
        == 6
    )
    assert mod._lzma_preset_for_wrapper("lzma_auto", payload_bytes=0) == 6
    assert mod._lzma_preset_for_wrapper("lzma_auto", payload_bytes=128) == 6
    assert (
        mod._lzma_preset_for_wrapper(
            "lzma_auto",
            payload_bytes=mod.SNERV_LF_LZMA_AUTO_EXTREME_MAX_INPUT_BYTES + 1,
        )
        == 6
    )
    assert mod._brotli_quality_for_wrapper("brotli", payload_bytes=10_000_000) == 11
    assert mod._lzma_preset_for_wrapper(
        "lzma_extreme",
        payload_bytes=10_000_000,
    ) == (9 | mod.lzma.PRESET_EXTREME)


def test_archive_lf_payload_codec_v2_is_receiver_decoded() -> None:
    planes = [
        np.array([[0, -1, 0], [0, 0, 1]], dtype=np.int64),
        np.arange(12, dtype=np.int64).reshape(3, 4) - 6,
    ]

    payload = encode_lf_quant_payload(planes, codec="portfolio_auto")
    decoded = decode_lf_quant_payload(payload)
    report = inspect_lf_quant_payload_header(payload)

    assert payload.startswith(SNERV_LF_QUANT_V2_MAGIC)
    assert report["header_format"] == "binary"
    for ref, got in zip(planes, decoded, strict=True):
        np.testing.assert_array_equal(got, ref)


def test_archive_lf_payload_auto_considers_v2_integer_portfolio() -> None:
    rng = np.random.default_rng(7)
    plane = rng.integers(0, 4, size=(128, 128), dtype=np.int64)

    auto_payload = encode_lf_quant_payload([plane], codec="auto")
    portfolio_payload = encode_lf_quant_payload([plane], codec="portfolio_auto")
    legacy_payload = encode_lf_quant_payload([plane], codec="int64_lzma")
    spatial_payload = encode_lf_quant_payload(
        [plane],
        codec="spatial_delta_zigzag_leb128_lzma",
    )
    decoded = decode_lf_quant_payload(auto_payload)

    np.testing.assert_array_equal(decoded[0], plane)
    assert auto_payload.startswith(SNERV_LF_QUANT_V2_MAGIC)
    assert len(auto_payload) == min(
        len(portfolio_payload),
        len(legacy_payload),
        len(spatial_payload),
    )
    assert len(auto_payload) < len(legacy_payload)


def test_archive_lf_payload_selected_label_is_reusable_codec_control() -> None:
    plane = np.array([[-2, -1, 0, 1], [1, 0, -1, -2]], dtype=np.int64)

    payload = encode_lf_quant_payload(
        [plane],
        codec="v2:signed_int2_bitpack:none",
    )
    report = inspect_lf_quant_payload_v2(payload)

    np.testing.assert_array_equal(decode_lf_quant_payload(payload)[0], plane)
    assert report.mode_histogram == {"signed_int2_bitpack": 1}
    assert report.wrapper_histogram == {"none": 1}
    assert (
        selected_lf_payload_codec_label(
            report.as_jsonable(),
            requested_codec="v2:signed_int2_bitpack:none",
        )
        == "v2:signed_int2_bitpack:none"
    )


def test_spatial_delta_lf_payload_reports_actual_codec_not_unknown_v2() -> None:
    plane = np.arange(16, dtype=np.int64).reshape(4, 4)

    payload = encode_lf_quant_payload(
        [plane],
        codec="spatial_delta_zigzag_leb128_lzma",
    )
    report = inspect_lf_quant_payload_header(payload)

    np.testing.assert_array_equal(decode_lf_quant_payload(payload)[0], plane)
    assert report["codec"] == "spatial_delta_zigzag_leb128_lzma"
    assert (
        selected_lf_payload_codec_label(
            report,
            requested_codec="spatial_delta_zigzag_leb128_lzma",
        )
        == "spatial_delta_zigzag_leb128_lzma"
    )


def test_archive_lf_payload_codec_v1_default_stays_legacy() -> None:
    plane = np.array([[1, -2], [3, -4]], dtype=np.int64)

    payload = encode_lf_quant_payload([plane])

    assert not payload.startswith(SNERV_LF_QUANT_V2_MAGIC)
    np.testing.assert_array_equal(decode_lf_quant_payload(payload)[0], plane)
