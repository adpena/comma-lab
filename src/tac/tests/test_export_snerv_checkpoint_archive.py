# SPDX-License-Identifier: MIT
"""Tests for direct SNeRV checkpoint archive export."""

from __future__ import annotations

import numpy as np

from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_snerv_archive_frames,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import SnervModelSizeConfig
from tools.export_snerv_checkpoint_archive import build_snerv_checkpoint_packet


def test_snerv_checkpoint_packet_uses_state_lf_and_decoder_directly() -> None:
    model_size = SnervModelSizeConfig(fc_dim=9, emb_size=0, temporal_context=0)
    lf = np.zeros((2, 2, 3, 8, 8), dtype=np.float32)
    for pair_idx in range(2):
        for frame_idx in range(2):
            for channel_idx in range(3):
                lf[pair_idx, frame_idx, channel_idx] = (
                    32.0
                    + 3.0 * pair_idx
                    + 5.0 * frame_idx
                    + 7.0 * channel_idx
                )
    state: dict[str, np.ndarray] = {"latents_lf_planes": lf}
    for subband in ("LH", "HL", "HH"):
        state[f"decoder_kernels.0.{subband}"] = np.zeros(
            (model_size.feature_count,),
            dtype=np.float32,
        )

    packet = build_snerv_checkpoint_packet(
        state,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="portfolio_auto",
        model_size=model_size,
        metadata_extra={"unit_test_marker": "direct_checkpoint_packet"},
    )
    decoded = unpack_snerv_archive(packet.packet)
    frames = decode_snerv_archive_frames(packet.packet)

    assert decoded.metadata.get("checkpoint_export_schema") is None
    assert decoded.metadata["unit_test_marker"] == "direct_checkpoint_packet"
    assert decoded.metadata["hf_decoder_fit_mode"] == "trained_mlx_checkpoint_decoder_kernels"
    assert decoded.metadata["native_mlx_training_executed"] is True
    assert decoded.metadata["score_claim"] is False
    assert packet.total_bytes == len(packet.packet)
    assert frames.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(frames).all()
