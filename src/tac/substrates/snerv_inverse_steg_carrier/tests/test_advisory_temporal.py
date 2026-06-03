# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.analysis.snerv_step_map_coder import decode_step_maps, encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.advisory import (
    _decode_receiver_codes_into_pairs,
    _LfRecord,
    resolve_snerv_modelsize_control,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_decoder_payload,
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SnervCarrierError,
    SnervFrameCode,
    SnervModelSizeConfig,
    decode_frame,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    quantize_lf,
)


def test_advisory_temporal_decode_matches_receiver_archive_replay() -> None:
    """NO-FAKE: temporal SNeRV advisory decode consumes LF timelines."""

    n_pairs = 2
    h, w = 24, 32
    frames = _temporal_rgb_pairs(n_pairs=n_pairs, hw=(h, w))
    pyrs = [
        encode_frame_lf(frames[pair, frame, channel], levels=1, wavelet="haar")
        for pair in range(n_pairs)
        for frame in range(2)
        for channel in range(3)
    ]
    model_size = SnervModelSizeConfig(
        fc_dim=9,
        emb_size=0,
        temporal_context=1,
        temporal_mode="official_haar_dwt1d_lowpass",
    )
    decoder = fit_hf_decoder_least_squares(
        pyrs,
        levels=1,
        model_size=model_size,
        temporal_group_count=3,
    )
    decoder_payload = encode_decoder_payload(decoder)
    receiver_decoder = decode_decoder_payload(decoder_payload)
    step_maps = [np.full(pyr.lf.shape, 1.0, dtype=np.float32) for pyr in pyrs]
    step_packet = encode_step_maps(step_maps, bins=8)
    receiver_steps = decode_step_maps(step_packet.packet)
    q_planes = []
    zeros = []
    codes = []
    records = []
    flat = 0
    for pair in range(n_pairs):
        for frame in range(2):
            for channel in range(3):
                pyr = pyrs[flat]
                q, scale, zero = quantize_lf(
                    pyr.lf,
                    per_element_steps=receiver_steps[flat],
                )
                receiver_zero = float(np.asarray(zero, dtype="<f4"))
                q_planes.append(q)
                zeros.append(receiver_zero)
                records.append(
                    _LfRecord(
                        pair_index=pair,
                        frame_index=frame,
                        channel_index=channel,
                        lf=pyr.lf,
                    )
                )
                codes.append(
                    SnervFrameCode(
                        lf_quant=q,
                        lf_scale=scale,
                        lf_zero=receiver_zero,
                        lf_shape=q.shape,
                        levels=1,
                        wavelet="haar",
                        orig_hw=(h, w),
                        per_element_steps=receiver_steps[flat],
                    )
                )
                flat += 1

    with pytest.raises(SnervCarrierError, match="lf_sequence"):
        decode_frame(codes[0], receiver_decoder)

    reference_pairs = torch.from_numpy(frames.astype(np.float32))
    advisory_recon = _decode_receiver_codes_into_pairs(
        records=records,
        codes=codes,
        decoder=receiver_decoder,
        reference_pairs=reference_pairs,
    )
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=zeros),
        lf_payload=encode_lf_quant_payload(q_planes),
        decoder_payload=decoder_payload,
        step_map_packet=step_packet.packet,
        metadata={
            "n_pairs": n_pairs,
            "frames_per_pair": 2,
            "channels": 3,
            "lf_plane_count": len(q_planes),
            "levels": 1,
            "wavelet": "haar",
            "carrier_hw": [h, w],
        },
    )
    receiver_recon = decode_snerv_archive_frames(archive.packet)

    assert receiver_decoder.model_size.temporal_context == 1
    np.testing.assert_array_equal(advisory_recon.cpu().numpy(), receiver_recon)


def test_advisory_resolves_official_modelsize_into_receiver_config() -> None:
    resolution = resolve_snerv_modelsize_control(
        full_data_length=100,
        final_size=4096,
        snerv_fc_dim=9,
        snerv_fc_dim_explicit=False,
        snerv_emb_size=2,
        snerv_official_modelsize_mparams=1.0,
        snerv_official_enc_strds=(2, 2),
        snerv_official_dec_strds=(2, 2),
    )

    assert resolution.capacity_source == "official_snerv_modelsize"
    assert resolution.model_size.fc_dim > 9
    assert resolution.official_modelsize_solution is not None
    assert resolution.official_modelsize_solution["modelsize_mparams"] == 1.0
    assert resolution.official_modelsize_solution["fc_dim"] == (
        resolution.model_size.fc_dim
    )
    metadata = resolution.metadata()
    assert metadata["model_size"]["fc_dim"] == resolution.model_size.fc_dim
    assert metadata["score_claim"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False


def test_advisory_rejects_conflicting_manual_fc_dim_and_official_modelsize() -> None:
    with pytest.raises(SnervCarrierError, match="conflicts"):
        resolve_snerv_modelsize_control(
            full_data_length=100,
            final_size=4096,
            snerv_fc_dim=3,
            snerv_fc_dim_explicit=True,
            snerv_emb_size=2,
            snerv_official_modelsize_mparams=1.0,
            snerv_official_enc_strds=(2, 2),
            snerv_official_dec_strds=(2, 2),
        )


def _temporal_rgb_pairs(*, n_pairs: int, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.empty((n_pairs, 2, 3, h, w), dtype=np.float32)
    for pair in range(n_pairs):
        for frame in range(2):
            t = pair * 2 + frame
            for channel in range(3):
                out[pair, frame, channel] = np.clip(
                    122.0
                    + 17.0 * np.sin((xx - 1.5 * t) / (4.5 + channel))
                    + 9.0 * np.cos((yy + 0.5 * t) / (3.5 + channel))
                    + 4.0 * channel,
                    0.0,
                    255.0,
                )
    return out
