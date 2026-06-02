# SPDX-License-Identifier: MIT
"""Tests for receiver-visible SNeRV LF payload v2 codec."""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_lf_quant_payload,
    encode_lf_quant_payload,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec import (
    SNERV_LF_QUANT_V2_MAGIC,
    SnervLfPayloadCodecError,
    decode_lf_quant_payload_v2,
    encode_lf_quant_payload_v2_with_report,
    inspect_lf_quant_payload_v2,
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
            "zigzag_delta_varint",
            "zero_run_varint",
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


def test_archive_lf_payload_codec_v2_is_receiver_decoded() -> None:
    planes = [
        np.array([[0, -1, 0], [0, 0, 1]], dtype=np.int64),
        np.arange(12, dtype=np.int64).reshape(3, 4) - 6,
    ]

    payload = encode_lf_quant_payload(planes, codec="portfolio_auto")
    decoded = decode_lf_quant_payload(payload)

    assert payload.startswith(SNERV_LF_QUANT_V2_MAGIC)
    for ref, got in zip(planes, decoded, strict=True):
        np.testing.assert_array_equal(got, ref)


def test_archive_lf_payload_codec_v1_default_stays_legacy() -> None:
    plane = np.array([[1, -2], [3, -4]], dtype=np.int64)

    payload = encode_lf_quant_payload([plane])

    assert not payload.startswith(SNERV_LF_QUANT_V2_MAGIC)
    np.testing.assert_array_equal(decode_lf_quant_payload(payload)[0], plane)
