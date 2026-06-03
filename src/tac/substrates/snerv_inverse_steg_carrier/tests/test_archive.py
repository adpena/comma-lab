# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV receiver archive packet bundling."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from tac.analysis.snerv_step_map_coder import decode_step_maps, encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    HEADER_LEN_FMT,
    SECTION_ORDER,
    SNERV_ARCHIVE_MAGIC,
    SnervArchiveError,
    decode_decoder_payload,
    decode_lf_metadata_payload,
    decode_lf_quant_payload,
    decode_snerv_archive_frame_planes,
    decode_snerv_archive_frames,
    decode_snerv_archive_step_maps,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    HfGenerationDecoder,
    SnervFrameCode,
    SnervModelSizeConfig,
    decode_frame,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    quantize_lf,
)


def _step_maps() -> list[np.ndarray]:
    yy, xx = np.mgrid[0:8, 0:12].astype(np.float32)
    return [
        np.exp2(0.5 + 0.05 * np.sin(xx / 3.0) + i * 0.01).astype(np.float32)
        for i in range(3)
    ]


def test_archive_bundles_sections_and_decodes_receiver_step_maps() -> None:
    step_packet = encode_step_maps(_step_maps(), bins=16)
    metadata_payload = encode_lf_metadata_payload(lf_zero_points=[0.25, 0.5, 0.75])
    lf_planes = [
        np.arange(12, dtype=np.int64).reshape(3, 4),
        -np.arange(12, dtype=np.int64).reshape(3, 4),
        np.ones((3, 4), dtype=np.int64) * 7,
    ]
    decoder = HfGenerationDecoder.zeros(levels=2)
    lf_payload = encode_lf_quant_payload(lf_planes)
    decoder_payload = encode_decoder_payload(decoder)
    archive = pack_snerv_archive(
        metadata_payload=metadata_payload,
        lf_payload=lf_payload,
        decoder_payload=decoder_payload,
        step_map_packet=step_packet.packet,
        metadata={"lf_plane_count": 3, "levels": 4, "wavelet": "db2"},
    )
    decoded = unpack_snerv_archive(archive.packet)

    assert archive.packet.startswith(SNERV_ARCHIVE_MAGIC)
    assert archive.section_order == SECTION_ORDER
    assert decoded.section_order == SECTION_ORDER
    for ref, got in zip(lf_planes, decoded.decode_lf_quant_planes(), strict=True):
        np.testing.assert_array_equal(got, ref)
    decoded_decoder = decoded.decode_decoder()
    assert decoded_decoder.levels == decoder.levels
    for lvl in range(decoder.levels):
        for subband in ("LH", "HL", "HH"):
            np.testing.assert_allclose(
                decoded_decoder.kernels[lvl][subband],
                decoder.kernels[lvl][subband],
            )
    np.testing.assert_allclose(decoded.decode_lf_zero_points(), [0.25, 0.5, 0.75])
    assert len(decoded.decode_step_maps()) == 3
    assert len(decode_snerv_archive_step_maps(archive.packet)) == 3
    assert archive.score_claim is False
    assert archive.ready_for_exact_eval_dispatch is False


def test_archive_is_deterministic_and_hash_checked() -> None:
    step_packet = encode_step_maps(_step_maps(), bins=4).packet
    metadata_payload = encode_lf_metadata_payload(lf_zero_points=[1.0, 2.0, 3.0])
    kwargs = {
        "metadata_payload": metadata_payload,
        "lf_payload": b"lf",
        "decoder_payload": b"decoder",
        "step_map_packet": step_packet,
        "metadata": {"lf_plane_count": 3},
    }
    a = pack_snerv_archive(**kwargs)
    b = pack_snerv_archive(**kwargs)

    assert a.packet == b.packet
    mutated = bytearray(a.packet)
    mutated[-1] ^= 0x01
    with pytest.raises(SnervArchiveError, match="sha256 mismatch"):
        unpack_snerv_archive(bytes(mutated))


def test_archive_rejects_trailing_payload_and_noncontiguous_offsets() -> None:
    step_packet = encode_step_maps(_step_maps(), bins=4).packet
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[1.0, 2.0, 3.0]),
        lf_payload=b"lf",
        decoder_payload=b"decoder",
        step_map_packet=step_packet,
        metadata={"lf_plane_count": 3},
    )

    with pytest.raises(SnervArchiveError, match="trailing payload"):
        unpack_snerv_archive(archive.packet + b"x")

    broken = _rewrite_header(
        archive.packet,
        lambda header: header["sections"][1].update({"offset": 999}),
    )
    with pytest.raises(SnervArchiveError, match="expected"):
        unpack_snerv_archive(broken)


def test_lf_metadata_payload_rejects_bad_counts_and_alignment() -> None:
    payload = encode_lf_metadata_payload(lf_zero_points=[1.0, 2.0])

    with pytest.raises(SnervArchiveError, match="expected 3"):
        decode_lf_metadata_payload(payload, expected_count=3)
    with pytest.raises(SnervArchiveError, match="float32-aligned"):
        decode_lf_metadata_payload(payload + b"x")
    with pytest.raises(SnervArchiveError, match="non-empty"):
        encode_lf_metadata_payload(lf_zero_points=[])


def test_lf_quant_and_decoder_payloads_roundtrip_independently() -> None:
    planes = [np.array([[1, -2], [3, -4]], dtype=np.int32)]
    decoded_planes = decode_lf_quant_payload(encode_lf_quant_payload(planes))
    np.testing.assert_array_equal(decoded_planes[0], planes[0])

    decoder = HfGenerationDecoder.zeros(levels=1)
    decoder.kernels[0]["LH"][0, 0] = 0.125
    decoded_decoder = decode_decoder_payload(encode_decoder_payload(decoder))
    assert decoded_decoder.levels == 1
    assert decoded_decoder.kernels[0]["LH"][0, 0] == pytest.approx(0.125)

    with pytest.raises(SnervArchiveError, match="integers"):
        encode_lf_quant_payload([np.array([1.25], dtype=np.float32)])


def test_quantized_decoder_payload_codecs_roundtrip_receiver_values() -> None:
    decoder = HfGenerationDecoder.zeros(levels=2)
    for lvl in range(decoder.levels):
        for idx, subband in enumerate(("LH", "HL", "HH"), start=1):
            decoder.kernels[lvl][subband] = (
                np.arange(9, dtype=np.float64).reshape(3, 3) - 4.0
            ) * (0.0125 * idx * (lvl + 1))

    legacy_payload = encode_decoder_payload(decoder)
    payload_sizes = {}
    for codec, tolerance in (
        ("int8_symmetric", 1.5e-3),
        ("int4_symmetric", 4.5e-2),
        ("int2_symmetric", 3.1e-1),
    ):
        payload = encode_decoder_payload(decoder, codec=codec)
        header = _read_subpacket_header(payload)
        decoded = decode_decoder_payload(payload)
        payload_sizes[codec] = len(payload)

        assert header["schema"] == "snerv_decoder_payload.v2"
        assert header["codec"] == codec
        assert header["bits_per_weight"] in {8, 4, 2}
        assert header["quantizer"] == "symmetric_per_kernel_fp16_scale"
        for lvl in range(decoder.levels):
            for subband in ("LH", "HL", "HH"):
                np.testing.assert_allclose(
                    decoded.kernels[lvl][subband],
                    decoder.kernels[lvl][subband],
                    atol=tolerance,
                    rtol=0.0,
                )

    assert payload_sizes["int2_symmetric"] <= payload_sizes["int4_symmetric"]
    assert payload_sizes["int4_symmetric"] <= payload_sizes["int8_symmetric"]
    assert decode_decoder_payload(legacy_payload).levels == decoder.levels


def test_decoder_payload_roundtrips_nondefault_model_size_controls() -> None:
    model_size = SnervModelSizeConfig(fc_dim=10, emb_size=2, patch_radius=1)
    decoder = HfGenerationDecoder.zeros(levels=1, model_size=model_size)
    pattern = np.linspace(-0.06, 0.06, model_size.feature_count, dtype=np.float64)
    decoder.kernels[0]["LH"] = pattern
    decoder.kernels[0]["HL"] = pattern * 0.5
    decoder.kernels[0]["HH"] = -pattern * 0.25

    for codec, schema, tolerance in (
        ("float32_lzma", "snerv_decoder_payload.v1", 2.0e-9),
        ("int8_symmetric", "snerv_decoder_payload.v2", 6e-4),
        ("int4_symmetric", "snerv_decoder_payload.v2", 1.0e-2),
        ("int2_symmetric", "snerv_decoder_payload.v2", 4.5e-2),
        ("mixed_magnitude_symmetric", "snerv_decoder_payload.v3", 1.0e-2),
    ):
        payload = encode_decoder_payload(decoder, codec=codec)
        header = _read_subpacket_header(payload)
        decoded = decode_decoder_payload(payload)

        assert header["schema"] == schema
        assert header["feature_count"] == 12
        assert header["kernel_shape"] == [12]
        assert header["model_size_config"] == model_size.as_jsonable()
        assert decoded.model_size == model_size
        assert decoded.kernels[0]["LH"].shape == (12,)
        np.testing.assert_allclose(
            decoded.kernels[0]["LH"],
            decoder.kernels[0]["LH"],
            atol=tolerance,
            rtol=0.0,
        )


def test_mixed_decoder_payload_uses_per_kernel_modes_and_roundtrip_values() -> None:
    decoder = HfGenerationDecoder.zeros(levels=2)
    magnitudes = [0.0, 0.008, 0.025, 0.08, 0.2, 0.012]
    groups = [
        (lvl, subband)
        for lvl in range(decoder.levels)
        for subband in ("LH", "HL", "HH")
    ]
    pattern = np.linspace(-1.0, 1.0, 9, dtype=np.float64).reshape(3, 3)
    for magnitude, (lvl, subband) in zip(magnitudes, groups, strict=True):
        decoder.kernels[lvl][subband] = pattern * magnitude

    payload = encode_decoder_payload(decoder, codec="mixed_magnitude_symmetric")
    header = _read_subpacket_header(payload)
    decoded = decode_decoder_payload(payload)

    assert header["schema"] == "snerv_decoder_payload.v3"
    assert header["codec"] == "mixed_magnitude_symmetric"
    assert header["quantizer"] == "mixed_per_kernel_zero_int2_int4_int8_fp16"
    assert header["mode_code_bits"] == 3
    assert header["mode_histogram"] == {
        "zero": 1,
        "int2": 2,
        "int4": 1,
        "int8": 1,
        "fp16": 1,
    }
    assert header["mode_code_bytes"] == 3
    assert header["scale_count"] == 4
    assert header["fp16_value_bytes"] == 18
    for lvl, subband in groups:
        np.testing.assert_allclose(
            decoded.kernels[lvl][subband],
            decoder.kernels[lvl][subband],
            atol=0.02,
            rtol=0.0,
        )


def test_mixed_decoder_payload_accepts_explicit_mode_assignments() -> None:
    decoder = HfGenerationDecoder.zeros(levels=2)
    groups = [
        (lvl, subband)
        for lvl in range(decoder.levels)
        for subband in ("LH", "HL", "HH")
    ]
    pattern = np.linspace(-1.0, 1.0, 9, dtype=np.float64).reshape(3, 3)
    magnitudes = [0.0, 0.008, 0.025, 0.08, 0.2, 0.0]
    modes = ("zero", "int2", "int4", "int8", "fp16", "zero")
    for magnitude, (lvl, subband) in zip(magnitudes, groups, strict=True):
        decoder.kernels[lvl][subband] = pattern * magnitude

    payload = encode_decoder_payload(
        decoder,
        codec="mixed_magnitude_symmetric",
        mixed_modes=modes,
    )
    header = _read_subpacket_header(payload)
    decoded = decode_decoder_payload(payload)

    assert header["schema"] == "snerv_decoder_payload.v3"
    assert header["mode_assignment_source"] == "explicit"
    assert header["mode_histogram"] == {
        "zero": 2,
        "int2": 1,
        "int4": 1,
        "int8": 1,
        "fp16": 1,
    }
    for (lvl, subband), mode in zip(groups, modes, strict=True):
        expected = decoder.kernels[lvl][subband]
        actual = decoded.kernels[lvl][subband]
        if mode == "zero":
            np.testing.assert_array_equal(actual, np.zeros((3, 3)))
        else:
            np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0.0)

    with pytest.raises(SnervArchiveError, match="mode count"):
        encode_decoder_payload(
            decoder,
            codec="mixed_magnitude_symmetric",
            mixed_modes=modes[:-1],
        )
    with pytest.raises(SnervArchiveError, match="unsupported decoder mixed mode"):
        encode_decoder_payload(
            decoder,
            codec="mixed_magnitude_symmetric",
            mixed_modes=("zero", "banana", "int4", "int8", "fp16", "zero"),
        )
    with pytest.raises(SnervArchiveError, match="require mixed codec"):
        encode_decoder_payload(decoder, codec="int8_symmetric", mixed_modes=modes)


def test_quantized_decoder_payload_rejects_corrupt_payload_bytes() -> None:
    decoder = HfGenerationDecoder.zeros(levels=1)
    payload = bytearray(encode_decoder_payload(decoder, codec="int4_symmetric"))
    payload[-1] ^= 0x01

    with pytest.raises(SnervArchiveError, match="sha256 mismatch"):
        decode_decoder_payload(bytes(payload))


def test_archive_decoded_sections_reconstruct_receiver_frame() -> None:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:32, 0:48].astype(np.float64)
    frames = [
        np.clip(
            120.0
            + 25.0 * np.sin(xx / 7.0 + i * 0.15)
            + 15.0 * np.cos(yy / 5.0)
            + rng.standard_normal(xx.shape) * 0.5,
            0.0,
            255.0,
        )
        for i in range(4)
    ]
    pyrs = [encode_frame_lf(frame, levels=2) for frame in frames]
    decoder = fit_hf_decoder_least_squares(pyrs, levels=2)
    lf = pyrs[0].lf
    steps = np.full(lf.shape, 2.0, dtype=np.float32)
    steps[0, 0] = 8.0
    q, scale, zero = quantize_lf(lf, per_element_steps=steps)
    step_packet = encode_step_maps([steps], bins=16)
    decoder_payload = encode_decoder_payload(decoder)
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[zero]),
        lf_payload=encode_lf_quant_payload([q]),
        decoder_payload=decoder_payload,
        step_map_packet=step_packet.packet,
        metadata={
            "lf_plane_count": 1,
            "levels": 2,
            "wavelet": pyrs[0].wavelet,
            "orig_hw": list(frames[0].shape),
        },
    )
    decoded = unpack_snerv_archive(archive.packet)

    receiver_q = decoded.decode_lf_quant_planes()[0]
    receiver_steps = decoded.decode_step_maps()[0]
    receiver_zero = float(decoded.decode_lf_zero_points()[0])
    receiver_code = SnervFrameCode(
        lf_quant=receiver_q,
        lf_scale=scale,
        lf_zero=receiver_zero,
        lf_shape=receiver_q.shape,
        levels=2,
        wavelet=pyrs[0].wavelet,
        orig_hw=frames[0].shape,
        per_element_steps=receiver_steps,
    )
    direct_code = SnervFrameCode(
        lf_quant=q,
        lf_scale=scale,
        lf_zero=zero,
        lf_shape=q.shape,
        levels=2,
        wavelet=pyrs[0].wavelet,
        orig_hw=frames[0].shape,
        per_element_steps=decode_snerv_archive_step_maps(archive.packet)[0],
    )

    np.testing.assert_array_equal(receiver_q, q)
    np.testing.assert_allclose(
        decode_frame(receiver_code, decoded.decode_decoder()),
        decode_frame(direct_code, decode_decoder_payload(decoder_payload)),
    )


def test_archive_full_frame_replay_reconstructs_ordered_pair_tensor() -> None:
    rng = np.random.default_rng(11)
    h, w = 32, 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    frames = []
    for frame_index in range(2):
        channels = []
        for channel_index in range(3):
            channels.append(
                np.clip(
                    110.0
                    + 18.0 * np.sin(xx / (5.0 + channel_index))
                    + 12.0 * np.cos(yy / (4.0 + frame_index))
                    + 4.0 * channel_index
                    + rng.standard_normal(xx.shape) * 0.2,
                    0.0,
                    255.0,
                )
            )
        frames.append(channels)
    pyrs = [
        encode_frame_lf(channel, levels=2)
        for frame in frames
        for channel in frame
    ]
    decoder = fit_hf_decoder_least_squares(pyrs, levels=2)
    decoder_payload = encode_decoder_payload(decoder)
    q_planes = []
    zero_points = []
    step_maps = []
    for idx, pyr in enumerate(pyrs):
        steps = np.full(pyr.lf.shape, 1.5 + 0.1 * idx, dtype=np.float32)
        steps[0, 0] = 4.0 + idx
        step_maps.append(steps)
    step_packet = encode_step_maps(step_maps, bins=16)
    receiver_step_maps = decode_step_maps(step_packet.packet)
    for pyr, receiver_steps in zip(pyrs, receiver_step_maps, strict=True):
        q, _, zero = quantize_lf(pyr.lf, per_element_steps=receiver_steps)
        q_planes.append(q)
        zero_points.append(zero)
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=zero_points),
        lf_payload=encode_lf_quant_payload(q_planes),
        decoder_payload=decoder_payload,
        step_map_packet=step_packet.packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 3,
            "lf_plane_count": 6,
            "levels": 2,
            "wavelet": pyrs[0].wavelet,
            "carrier_hw": [h, w],
        },
    )

    decoded = unpack_snerv_archive(archive.packet)
    receiver_decoder = decoded.decode_decoder()
    expected_planes = []
    for q, zero, receiver_steps in zip(
        decoded.decode_lf_quant_planes(),
        decoded.decode_lf_zero_points(),
        decoded.decode_step_maps(),
        strict=True,
    ):
        code = SnervFrameCode(
            lf_quant=q,
            lf_scale=1.0,
            lf_zero=float(zero),
            lf_shape=q.shape,
            levels=2,
            wavelet=pyrs[0].wavelet,
            orig_hw=(h, w),
            per_element_steps=receiver_steps,
        )
        expected_planes.append(
            np.clip(decode_frame(code, receiver_decoder), 0.0, 255.0).astype(np.float32)
        )
    flat = decode_snerv_archive_frame_planes(archive.packet)
    replayed = decode_snerv_archive_frames(archive.packet)

    assert replayed.shape == (1, 2, 3, h, w)
    assert len(flat) == 6
    for expected, got in zip(expected_planes, flat, strict=True):
        np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(replayed.reshape(6, h, w), np.stack(expected_planes))


def test_archive_receiver_replay_consumes_temporal_context_from_lf_sequence() -> None:
    h, w = 32, 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    frames = []
    for frame_index in range(2):
        channels = []
        for channel_index in range(3):
            channels.append(
                np.clip(
                    120.0
                    + 20.0 * np.sin((xx - 2 * frame_index) / (5.0 + channel_index))
                    + 10.0 * np.cos((yy + frame_index) / 4.0)
                    + 3.0 * channel_index,
                    0.0,
                    255.0,
                )
            )
        frames.append(channels)
    pyrs = [
        encode_frame_lf(channel, levels=2, wavelet="haar")
        for frame in frames
        for channel in frame
    ]
    model_size = SnervModelSizeConfig(fc_dim=9, emb_size=0, temporal_context=1)
    decoder = fit_hf_decoder_least_squares(
        pyrs,
        levels=2,
        model_size=model_size,
        temporal_group_count=3,
    )
    decoder_payload = encode_decoder_payload(decoder)
    step_maps = [np.full(pyr.lf.shape, 1.0, dtype=np.float32) for pyr in pyrs]
    step_packet = encode_step_maps(step_maps, bins=8)
    receiver_step_maps = decode_step_maps(step_packet.packet)
    q_planes = []
    zero_points = []
    for pyr, receiver_steps in zip(pyrs, receiver_step_maps, strict=True):
        q, _scale, zero = quantize_lf(pyr.lf, per_element_steps=receiver_steps)
        q_planes.append(q)
        zero_points.append(zero)
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=zero_points),
        lf_payload=encode_lf_quant_payload(q_planes),
        decoder_payload=decoder_payload,
        step_map_packet=step_packet.packet,
        metadata={
            "n_pairs": 1,
            "frames_per_pair": 2,
            "channels": 3,
            "lf_plane_count": 6,
            "levels": 2,
            "wavelet": "haar",
            "carrier_hw": [h, w],
        },
    )

    decoded = unpack_snerv_archive(archive.packet)
    q_recv = decoded.decode_lf_quant_planes()
    zeros_recv = decoded.decode_lf_zero_points()
    steps_recv = decoded.decode_step_maps()
    lf_sequence_all = [
        q.astype(np.float64) * steps + float(zero)
        for q, zero, steps in zip(q_recv, zeros_recv, steps_recv, strict=True)
    ]
    expected_planes = []
    for idx, (q, zero, steps) in enumerate(
        zip(q_recv, zeros_recv, steps_recv, strict=True)
    ):
        code = SnervFrameCode(
            lf_quant=q,
            lf_scale=1.0,
            lf_zero=float(zero),
            lf_shape=q.shape,
            levels=2,
            wavelet="haar",
            orig_hw=(h, w),
            per_element_steps=steps,
        )
        group = idx % 3
        expected_planes.append(
            np.clip(
                decode_frame(
                    code,
                    decoded.decode_decoder(),
                    lf_sequence=lf_sequence_all[group::3],
                    sequence_index=idx // 3,
                ),
                0.0,
                255.0,
            ).astype(np.float32)
        )

    replayed = decode_snerv_archive_frames(archive.packet)

    assert decoded.decode_decoder().model_size.temporal_context == 1
    assert replayed.shape == (1, 2, 3, h, w)
    np.testing.assert_array_equal(replayed.reshape(6, h, w), np.stack(expected_planes))


def test_archive_full_frame_replay_requires_pair_grouping_metadata() -> None:
    step_packet = encode_step_maps(_step_maps()[:1], bins=4)
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=[0.0]),
        lf_payload=encode_lf_quant_payload([np.zeros((8, 12), dtype=np.int64)]),
        decoder_payload=encode_decoder_payload(HfGenerationDecoder.zeros(levels=2)),
        step_map_packet=step_packet.packet,
        metadata={"lf_plane_count": 1, "levels": 2, "wavelet": "db2", "orig_hw": [32, 48]},
    )

    with pytest.raises(SnervArchiveError, match="metadata missing 'n_pairs'"):
        decode_snerv_archive_frames(archive.packet)


def test_archive_receiver_module_imports_no_torch_or_scorer() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.archive as archive_mod

    with open(archive_mod.__file__) as f:
        src = f.read()
    assert "import torch" not in src
    assert "load_score_exact_scorers" not in src


def _rewrite_header(packet: bytes, mutator) -> bytes:
    offset = len(SNERV_ARCHIVE_MAGIC)
    (header_len,) = struct.unpack(
        HEADER_LEN_FMT,
        packet[offset : offset + struct.calcsize(HEADER_LEN_FMT)],
    )
    offset += struct.calcsize(HEADER_LEN_FMT)
    header_end = offset + header_len
    header = json.loads(packet[offset:header_end].decode("utf-8"))
    mutator(header)
    header_bytes = json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return (
        SNERV_ARCHIVE_MAGIC
        + struct.pack(HEADER_LEN_FMT, len(header_bytes))
        + header_bytes
        + packet[header_end:]
    )


def _read_subpacket_header(packet: bytes) -> dict[str, object]:
    offset = 5
    (header_len,) = struct.unpack(
        HEADER_LEN_FMT,
        packet[offset : offset + struct.calcsize(HEADER_LEN_FMT)],
    )
    header_start = offset + struct.calcsize(HEADER_LEN_FMT)
    header_end = header_start + int(header_len)
    return json.loads(packet[header_start:header_end].decode("utf-8"))
