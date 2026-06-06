# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.spectral_band_allocator import (
    HEADER_SIZE,
    build_score_tethered_spectral_band_allocator_receiver_proof,
    decode_score_tethered_spectral_band_allocator_payload,
    encode_score_tethered_spectral_band_allocator_payload,
    inspect_score_tethered_spectral_band_allocator_payload,
)


def test_spectral_band_allocator_binary_payload_roundtrips_without_json_labels() -> None:
    frames = np.arange(2 * 2 * 3 * 8 * 8, dtype=np.float32).reshape(2, 2, 3, 8, 8)

    packet = encode_score_tethered_spectral_band_allocator_payload(
        frames,
        pair_indices=(3, 7),
        lf_downsample=4,
        budget_units=127,
    )
    header, table = inspect_score_tethered_spectral_band_allocator_payload(
        packet.packet
    )
    decoded = decode_score_tethered_spectral_band_allocator_payload(packet.packet)

    assert packet.packet[:1] != b"{"
    assert packet.payload_bytes == len(packet.packet)
    assert packet.header["human_readable_payload_labels"] is False
    assert packet.header["header_bytes"] == HEADER_SIZE
    assert header["allocation_table_raw_bytes"] == len(table)
    assert decoded.shape == (2, 3, 4)
    np.testing.assert_array_equal(decoded.sum(axis=-1), np.full((2, 3), 127))
    assert packet.as_jsonable()["score_claim"] is False


def test_spectral_band_allocator_receiver_proof_closes_allocator_blockers() -> None:
    frames = np.linspace(0.0, 255.0, 1 * 2 * 3 * 8 * 8, dtype=np.float32).reshape(
        1,
        2,
        3,
        8,
        8,
    )

    proof, payload = build_score_tethered_spectral_band_allocator_receiver_proof(
        frames,
        pair_indices=(0,),
        packet_path="/Volumes/VertigoDataTier/pact/unit/snerv_packet.bin",
        source_packet_sha256="a" * 64,
        lf_downsample=4,
        budget_units=255,
        payload_path="/Volumes/VertigoDataTier/pact/unit/spectral.ssba",
    )

    assert payload
    assert proof["schema"] == (
        "snerv_score_tethered_spectral_band_allocator_receiver_proof.v1"
    )
    assert proof["receiver_payload_implemented"] is True
    assert proof["receiver_decode_proven"] is True
    assert proof["numpy_receiver_decode"] is True
    assert proof["score_tethered_allocation_implemented"] is True
    assert proof["section_native_byte_telemetry_present"] is True
    assert proof["human_readable_payload_labels"] is False
    assert proof["payload_bytes"] == len(payload)
    assert proof["closed_campaign_blockers"] == [
        "snerv_score_tethered_lf_hf_band_allocator_not_implemented",
        "snerv_mfu_hfr_section_native_byte_telemetry_missing",
    ]
    assert proof["blockers"] == [
        "snerv_score_tethered_spectral_band_allocator_false_authority"
    ]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
