# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.lf_latent_hyperprior import (
    HEADER_SIZE,
    build_lf_latent_hyperprior_receiver_proof,
    decode_lf_latent_hyperprior_payload,
    encode_lf_latent_hyperprior_payload,
    inspect_lf_latent_hyperprior_payload,
)


def test_lf_latent_hyperprior_binary_payload_replays_lf_latents() -> None:
    frames = np.linspace(0.0, 96.0, 2 * 2 * 2 * 8 * 8, dtype=np.float32).reshape(
        2,
        2,
        2,
        8,
        8,
    )

    packet = encode_lf_latent_hyperprior_payload(
        frames,
        pair_indices=(2, 5),
        lf_downsample=4,
        quant_step=0.5,
    )
    header, mean_raw, scale_raw, compressed = inspect_lf_latent_hyperprior_payload(
        packet.packet
    )
    decoded = decode_lf_latent_hyperprior_payload(packet.packet)

    assert packet.packet[:1] != b"{"
    assert packet.payload_bytes == len(packet.packet)
    assert packet.header["human_readable_payload_labels"] is False
    assert packet.header["header_bytes"] == HEADER_SIZE
    assert header["mean_raw_bytes"] == len(mean_raw)
    assert header["scale_raw_bytes"] == len(scale_raw)
    assert header["latent_symbol_compressed_bytes"] == len(compressed)
    assert decoded.shape == (2, 2, 2, 2, 2)
    assert packet.as_jsonable()["score_claim"] is False


def test_lf_latent_hyperprior_receiver_proof_closes_decoder_blockers() -> None:
    frames = np.linspace(10.0, 110.0, 1 * 2 * 3 * 8 * 8, dtype=np.float32).reshape(
        1,
        2,
        3,
        8,
        8,
    )

    proof, payload = build_lf_latent_hyperprior_receiver_proof(
        frames,
        pair_indices=(0,),
        packet_path="/Volumes/VertigoDataTier/pact/unit/snerv_packet.bin",
        source_packet_sha256="b" * 64,
        lf_downsample=4,
        quant_step=1.0,
        payload_path="/Volumes/VertigoDataTier/pact/unit/hyperprior.slhp",
    )

    assert payload
    assert proof["schema"] == "snerv_lf_latent_hyperprior_receiver_proof.v1"
    assert proof["receiver_payload_implemented"] is True
    assert proof["receiver_decode_proven"] is True
    assert proof["numpy_receiver_decode"] is True
    assert proof["entropy_model_implemented"] is True
    assert proof["hyperprior_scale_present"] is True
    assert proof["receiver_replay_proven"] is True
    assert proof["section_native_byte_telemetry_present"] is True
    assert proof["human_readable_payload_labels"] is False
    assert proof["payload_bytes"] == len(payload)
    assert proof["closed_campaign_blockers"] == [
        "snerv_lf_latent_hyperprior_not_implemented",
        "snerv_lf_latent_hyperprior_numpy_decoder_missing",
        "snerv_lf_latent_hyperprior_receiver_replay_missing",
    ]
    assert proof["blockers"] == ["snerv_lf_latent_hyperprior_payload_false_authority"]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False
