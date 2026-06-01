# SPDX-License-Identifier: MIT
"""NO-FAKE tests for advisory receiver-archive replay verification."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tac.analysis.snerv_step_map_coder import decode_step_maps, encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.advisory import (
    _verify_receiver_archive_full_frame_replay,
    _verify_receiver_archive_roundtrip,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    quantize_lf,
)


def test_advisory_receiver_archive_roundtrip_verifies_live_packet_state() -> None:
    archive, q, zero, receiver_steps, decoder = _tiny_receiver_archive()

    ok, error = _verify_receiver_archive_roundtrip(
        archive,
        lf_quant_planes=[q],
        lf_zero_points=[zero],
        step_maps=receiver_steps,
        decoder=decoder,
    )

    assert ok is True
    assert error == ""


def test_advisory_receiver_archive_roundtrip_rejects_mutated_packet() -> None:
    archive, q, zero, receiver_steps, decoder = _tiny_receiver_archive()
    packet = bytearray(archive.packet)
    packet[-1] ^= 0x01

    ok, error = _verify_receiver_archive_roundtrip(
        replace(archive, packet=bytes(packet)),
        lf_quant_planes=[q],
        lf_zero_points=[zero],
        step_maps=receiver_steps,
        decoder=decoder,
    )

    assert ok is False
    assert "SnervArchiveError" in error


def test_advisory_full_frame_replay_verifies_archive_visible_tensor() -> None:
    archive = _tiny_full_frame_receiver_archive()
    receiver_frames = decode_snerv_archive_frames(archive.packet)

    ok, error, replayed = _verify_receiver_archive_full_frame_replay(
        archive,
        reference_frames=_torch_tensor(receiver_frames),
    )
    bad_ok, bad_error, _ = _verify_receiver_archive_full_frame_replay(
        archive,
        reference_frames=_torch_tensor(receiver_frames + 1.0),
    )

    assert ok is True
    assert error == ""
    assert replayed is not None
    np.testing.assert_array_equal(replayed, receiver_frames)
    assert bad_ok is False
    assert "receiver_frame_replay_mismatch" in bad_error


def _tiny_receiver_archive():
    yy, xx = np.mgrid[0:32, 0:48].astype(np.float64)
    frames = [
        np.clip(
            100.0 + 30.0 * np.sin(xx / 6.0 + i * 0.1) + 5.0 * np.cos(yy / 4.0),
            0.0,
            255.0,
        )
        for i in range(3)
    ]
    pyramids = [encode_frame_lf(frame, levels=2) for frame in frames]
    decoder = fit_hf_decoder_least_squares(pyramids, levels=2)
    lf = pyramids[0].lf
    steps = np.full(lf.shape, 2.0, dtype=np.float32)
    steps[0, 0] = 4.0
    step_packet = encode_step_maps([steps], bins=4)
    receiver_steps = decode_step_maps(step_packet.packet)
    q, _, zero = quantize_lf(lf, per_element_steps=receiver_steps[0])
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[zero]),
        lf_payload=encode_lf_quant_payload([q]),
        decoder_payload=encode_decoder_payload(decoder),
        step_map_packet=step_packet.packet,
        metadata={
            "lf_plane_count": 1,
            "levels": 2,
            "wavelet": pyramids[0].wavelet,
            "orig_hw": [32, 48],
        },
    )
    return archive, q, zero, receiver_steps, decoder


def _tiny_full_frame_receiver_archive():
    yy, xx = np.mgrid[0:32, 0:48].astype(np.float64)
    frames = []
    for frame_index in range(2):
        for channel_index in range(3):
            frames.append(
                np.clip(
                    95.0
                    + 20.0 * np.sin(xx / 7.0 + frame_index * 0.2)
                    + 8.0 * np.cos(yy / 5.0 + channel_index * 0.1),
                    0.0,
                    255.0,
                )
            )
    pyramids = [encode_frame_lf(frame, levels=2) for frame in frames]
    decoder = fit_hf_decoder_least_squares(pyramids, levels=2)
    q_planes = []
    zeros = []
    steps = []
    for idx, pyramid in enumerate(pyramids):
        step = np.full(pyramid.lf.shape, 2.0 + idx * 0.05, dtype=np.float32)
        packet = encode_step_maps([step], bins=4)
        receiver_step = decode_step_maps(packet.packet)[0]
        q, _, zero = quantize_lf(pyramid.lf, per_element_steps=receiver_step)
        q_planes.append(q)
        zeros.append(zero)
        steps.append(receiver_step)
    step_packet = encode_step_maps(steps, bins=4)
    return pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=zeros),
        lf_payload=encode_lf_quant_payload(q_planes),
        decoder_payload=encode_decoder_payload(decoder),
        step_map_packet=step_packet.packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 3,
            "lf_plane_count": 6,
            "levels": 2,
            "wavelet": pyramids[0].wavelet,
            "carrier_hw": [32, 48],
        },
    )


def _torch_tensor(array: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(array, dtype=np.float32))
