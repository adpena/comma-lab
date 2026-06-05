# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.lf_conditioned_hf_residual import (
    build_lf_conditioned_hf_residual_receiver_proof,
    decode_lf_conditioned_hf_residual_payload,
    encode_lf_conditioned_hf_residual_payload,
    inspect_lf_conditioned_hf_residual_payload,
)


def test_lf_conditioned_hf_residual_packet_roundtrips_and_reports_bytes() -> None:
    frames = np.arange(2 * 2 * 3 * 4 * 6, dtype=np.float32).reshape(2, 2, 3, 4, 6)

    packet = encode_lf_conditioned_hf_residual_payload(
        frames,
        pair_indices=(7, 9),
        anchor_downsample=2,
        residual_quant_step=0.25,
    )
    header, compressed = inspect_lf_conditioned_hf_residual_payload(packet.packet)
    decoded = decode_lf_conditioned_hf_residual_payload(packet.packet)

    assert packet.payload_bytes == len(packet.packet)
    assert packet.header["packet_bytes"] == len(packet.packet)
    assert header["pair_indices"] == [7, 9]
    assert header["section_native_byte_telemetry_present"] is True
    assert len(compressed) == header["compressed_bytes"]
    np.testing.assert_allclose(decoded, frames, atol=0.125)
    assert packet.as_jsonable()["score_claim"] is False


def test_lf_conditioned_hf_residual_receiver_proof_closes_payload_blocker() -> None:
    frames = np.linspace(0.0, 255.0, 2 * 2 * 1 * 4 * 4, dtype=np.float32).reshape(
        2,
        2,
        1,
        4,
        4,
    )

    proof, payload = build_lf_conditioned_hf_residual_receiver_proof(
        frames,
        pair_indices=(0, 1),
        packet_path="/Volumes/VertigoDataTier/pact/unit/snerv_packet.bin",
        source_packet_sha256="a" * 64,
        residual_quant_step=0.5,
        payload_path="/Volumes/VertigoDataTier/pact/unit/residual.slhr",
    )

    assert payload
    assert proof["schema"] == "snerv_lf_conditioned_hf_residual_receiver_proof.v1"
    assert proof["receiver_payload_implemented"] is True
    assert proof["receiver_decode_proven"] is True
    assert proof["section_native_byte_telemetry_present"] is True
    assert proof["payload_bytes"] == len(payload)
    assert proof["closed_campaign_blockers"] == [
        "snerv_hf_residual_generator_receiver_payload_not_implemented"
    ]
    assert proof["blockers"] == [
        "snerv_lf_conditioned_hf_residual_payload_false_authority"
    ]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
