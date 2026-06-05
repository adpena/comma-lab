# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV receiver semantic section neutralization."""

from __future__ import annotations

import hashlib
import lzma

import numpy as np
import pytest

from tac.analysis.snerv_step_map_coder import encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    DECODER_PAYLOAD_V1_SCHEMA,
    DECODER_SUBBANDS,
    SNERV_DECODER_MAGIC,
    _pack_subpacket,
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
    pack_snerv_archive_snar2,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import HfGenerationDecoder
from tac.substrates.snerv_inverse_steg_carrier.section_value import (
    RECEIVER_DECODE_ONLY_STATUS,
    SNERV_SNAR1_SECTION_VALUE_SCHEMA,
    SnervSectionValueError,
    neutralize_snerv_section,
    neutralize_snerv_sections,
)


def _packet(*, wire_format: str = "snar1") -> bytes:
    step_maps = [
        np.exp2(
            np.linspace(0.0, 1.0, 9, dtype=np.float32).reshape(3, 3) + idx * 0.1
        )
        for idx in range(6)
    ]
    decoder = HfGenerationDecoder.zeros(levels=1)
    decoder.kernels[0]["LH"][:] = 0.125
    decoder.kernels[0]["HL"][:] = -0.25
    decoder.kernels[0]["HH"][:] = 0.5
    packer = pack_snerv_archive_snar2 if wire_format == "snar2" else pack_snerv_archive
    archive = packer(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[0.0] * 6),
        lf_payload=encode_lf_quant_payload(
            [
                (np.arange(9, dtype=np.int64).reshape(3, 3) + idx)
                for idx in range(6)
            ],
            codec="portfolio_auto",
        ),
        decoder_payload=encode_decoder_payload(decoder),
        step_map_packet=encode_step_maps(step_maps, bins=8).packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 3,
            "carrier_hw": [6, 6],
            "orig_hw": [6, 6],
            "lf_plane_count": 6,
            "levels": 1,
            "wavelet": "haar",
        },
    )
    return archive.packet


def _packet_with_legacy_decoder_output_affine_tail() -> bytes:
    raw_values = np.arange(29, dtype="<f4")
    raw = raw_values.tobytes()
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    decoder = HfGenerationDecoder.zeros(levels=1)
    decoder_payload = _pack_subpacket(
        SNERV_DECODER_MAGIC,
        {
            "schema": DECODER_PAYLOAD_V1_SCHEMA,
            "levels": 1,
            "subbands": list(DECODER_SUBBANDS),
            "kernel_shape": [3, 3],
            "feature_count": 9,
            "model_size_config": decoder.model_size.as_jsonable(),
            "dtype": "float32_le",
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_bytes": len(raw),
            "compressed_bytes": len(compressed),
            "output_affine": {
                "mode": "scalar",
                "count": 1,
                "scale": float(raw_values[-2]),
                "bias": float(raw_values[-1]),
                "dtype": "float32_le",
            },
        },
        compressed,
    )
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[0.0] * 6),
        lf_payload=encode_lf_quant_payload(
            [np.zeros((3, 3), dtype=np.int64) for _ in range(6)],
            codec="portfolio_auto",
        ),
        decoder_payload=decoder_payload,
        step_map_packet=encode_step_maps(
            [np.ones((3, 3), dtype=np.float32) for _ in range(6)],
            bins=8,
        ).packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 3,
            "carrier_hw": [6, 6],
            "orig_hw": [6, 6],
            "lf_plane_count": 6,
            "levels": 1,
            "wavelet": "haar",
        },
    )
    return archive.packet


def test_snerv_section_value_neutralizes_decoder_with_receiver_decode() -> None:
    packet = _packet()

    report = neutralize_snerv_section(packet, "decoder_payload")

    assert report["schema"] == SNERV_SNAR1_SECTION_VALUE_SCHEMA
    assert report["packet_wire_format"] == "snar1"
    assert report["section"] == "decoder_payload"
    assert report["section_changed"] is True
    assert report["receiver_decode_status"] == "receiver_decode_succeeded"
    assert report["score_claim"] is False
    decoded = unpack_snerv_archive(report["packet"])
    assert decoded.decode_decoder().levels == 1
    assert np.count_nonzero(decoded.decode_decoder().kernels[0]["LH"]) == 0
    assert decode_snerv_archive_frames(report["packet"]).shape == (1, 2, 3, 6, 6)
    row = report["section_value_row"]
    assert row["byte_delta"] == -report["baseline_section_bytes"]
    assert row["receiver_decode_status"] == "receiver_decode_succeeded"
    assert row["receiver_decode_passed"] is True
    assert row["receiver_proof_status"] == RECEIVER_DECODE_ONLY_STATUS
    assert row["runtime_consumption_proof_passed"] is False
    assert "delta_nonrate_score_missing" in row["blockers"]
    assert "runtime_consumption_proof_not_executed_for_neutralized_packet" in row[
        "blockers"
    ]


def test_snerv_section_value_preserves_snar2_wire_format() -> None:
    packet = _packet(wire_format="snar2")

    report = neutralize_snerv_section(packet, "step_map_packet")

    assert report["schema"] == SNERV_SNAR1_SECTION_VALUE_SCHEMA
    assert report["packet_wire_format"] == "snar2"
    assert report["baseline_packet_schema"] == "snerv_inverse_steg_archive.snar2.v1"
    assert report["neutralized_packet_schema"] == "snerv_inverse_steg_archive.snar2.v1"
    assert report["packet"].startswith(b"SNAR2")
    assert report["section_value_row"]["row_id"].startswith("snerv_snar2_")
    assert report["section_value_row"]["packet_wire_format"] == "snar2"
    assert decode_snerv_archive_frames(report["packet"]).shape == (1, 2, 3, 6, 6)


def test_snerv_section_value_neutralizes_step_maps_and_rejects_required_lf() -> None:
    packet = _packet()

    step = neutralize_snerv_section(packet, "step_map_packet")
    assert step["neutralization_method"] == "per_map_constant_step_map"
    assert step["receiver_decode_status"] == "receiver_decode_succeeded"
    assert step["section_changed"] is True

    combined = neutralize_snerv_sections(packet)
    assert combined["variant_count"] == 2
    assert len(combined["section_value_rows"]) == 2
    assert combined["score_claim"] is False

    with pytest.raises(SnervSectionValueError, match="not neutralizable"):
        neutralize_snerv_section(packet, "lf_payload")


def test_snerv_section_value_decodes_legacy_decoder_output_affine_tail() -> None:
    packet = _packet_with_legacy_decoder_output_affine_tail()

    decoded = unpack_snerv_archive(packet).decode_decoder()
    assert decoded.levels == 1
    assert decoded.kernels[0]["LH"].shape == (3, 3)

    step = neutralize_snerv_section(packet, "step_map_packet")
    assert step["receiver_decode_status"] == "receiver_decode_succeeded"
    assert decode_snerv_archive_frames(step["packet"]).shape == (1, 2, 3, 6, 6)
