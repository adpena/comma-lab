# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.joint_lf_hf_codebook import (
    build_joint_lf_hf_factorized_codebook_receiver_proof,
    decode_joint_lf_hf_factorized_codebook_payload,
    encode_joint_lf_hf_factorized_codebook_payload,
    inspect_joint_lf_hf_factorized_codebook_payload,
)


def test_joint_lf_hf_codebook_packet_roundtrips_and_reports_bytes() -> None:
    frames = np.arange(2 * 2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 2, 3, 4, 4)

    packet = encode_joint_lf_hf_factorized_codebook_payload(
        frames,
        pair_indices=(4, 9),
        block_hw=(2, 2),
        codebook_size=64,
        quant_step=1.0,
    )
    header, compressed = inspect_joint_lf_hf_factorized_codebook_payload(
        packet.packet
    )
    decoded = decode_joint_lf_hf_factorized_codebook_payload(packet.packet)

    assert packet.payload_bytes == len(packet.packet)
    assert packet.header["packet_bytes"] == len(packet.packet)
    assert header["pair_indices"] == [4, 9]
    assert header["section_native_byte_telemetry_present"] is True
    assert header["codebook_raw_bytes"] > 0
    assert header["index_raw_bytes"] > 0
    assert len(compressed) == header["compressed_bytes"]
    np.testing.assert_array_equal(decoded, frames)
    assert packet.as_jsonable()["score_claim"] is False


def test_joint_lf_hf_codebook_receiver_proof_closes_implementation_blockers() -> None:
    frames = np.linspace(0.0, 80.0, 2 * 2 * 1 * 4 * 4, dtype=np.float32).reshape(
        2,
        2,
        1,
        4,
        4,
    )

    proof, payload = build_joint_lf_hf_factorized_codebook_receiver_proof(
        frames,
        pair_indices=(0, 1),
        packet_path="/Volumes/VertigoDataTier/pact/unit/snerv_packet.bin",
        source_packet_sha256="a" * 64,
        block_hw=(2, 2),
        codebook_size=64,
        quant_step=0.25,
        payload_path="/Volumes/VertigoDataTier/pact/unit/joint_codebook.sjlc",
    )

    assert payload
    assert proof["schema"] == (
        "snerv_joint_lf_hf_factorized_codebook_receiver_proof.v1"
    )
    assert proof["receiver_payload_implemented"] is True
    assert proof["receiver_decode_proven"] is True
    assert proof["numpy_receiver_decode"] is True
    assert proof["section_native_byte_telemetry_present"] is True
    assert proof["payload_bytes"] == len(payload)
    assert proof["closed_campaign_blockers"] == [
        "snerv_joint_lf_hf_factorized_codebook_not_implemented",
        "snerv_joint_lf_hf_codebook_numpy_receiver_missing",
        "snerv_joint_lf_hf_codebook_section_byte_telemetry_missing",
    ]
    assert proof["blockers"] == [
        "snerv_joint_lf_hf_factorized_codebook_false_authority"
    ]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
