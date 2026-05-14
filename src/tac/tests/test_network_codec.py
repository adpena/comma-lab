# SPDX-License-Identifier: MIT
"""Roundtrip tests for ``tac.network_codec``.

Check 46 contract: every public encoder must demonstrate
``decode(encode(x)) ≈ x`` to a known tolerance.

``network_codec.py`` exposes:

- ``export_network_archive`` / ``inflate_network_codec`` — SIREN codec
  archive (FP16 weight storage). Inflate runs the network forward,
  so the natural roundtrip is "frames generated from the loaded
  codec match frames generated from the source codec to within FP16
  weight quantization noise".

- ``export_mask_siren_archive`` / ``inflate_mask_siren_archive`` —
  same pattern but with mask conditioning.

We use small toy dimensions so tests run in <2s.
"""

from __future__ import annotations

import torch

from tac.network_codec import (
    SIRENVideoCodec,
    export_network_archive,
    inflate_network_codec,
    load_network_codec,
)


def test_network_codec_roundtrip_within_fp16_tolerance() -> None:
    """``inflate_network_codec(export_network_archive(c))`` matches the source's
    own ``generate_all_frames()`` output to within FP16 quantization noise.
    """
    torch.manual_seed(0)
    T, H, W = 2, 8, 8
    codec = SIRENVideoCodec(
        hidden=8, layers=2, omega_0=15.0,
        num_frames=T, frame_h=H, frame_w=W,
        pos_encoding_freqs=2,
    )
    codec.eval()

    archive = export_network_archive(codec, use_fp16=True)
    inflated = inflate_network_codec(archive)
    orig = codec.generate_all_frames()

    assert inflated.shape == (T, H, W, 3) == orig.shape
    assert inflated.dtype == orig.dtype == torch.uint8
    # FP16 weight quantization on a 2-layer SIREN: a few ULPs in pixel
    # space. The smoke test asserts max diff < 2.0 — we mirror that.
    diff = (inflated.float() - orig.float()).abs().max().item()
    assert diff < 2.0, (
        f"export/inflate roundtrip diff {diff:.3f} exceeds FP16 tolerance"
    )


def test_network_codec_state_roundtrip_within_fp16_tolerance() -> None:
    """Loading the model back via ``load_network_codec`` recovers a state dict
    that matches the source within FP16 weight quantization."""
    torch.manual_seed(1)
    codec = SIRENVideoCodec(
        hidden=8, layers=2, omega_0=15.0,
        num_frames=2, frame_h=8, frame_w=8,
        pos_encoding_freqs=2,
    )
    archive = export_network_archive(codec, use_fp16=True)
    loaded = load_network_codec(archive)

    src_state = codec.state_dict()
    loaded_state = loaded.state_dict()
    assert set(src_state.keys()) == set(loaded_state.keys())
    for key in src_state:
        src = src_state[key].float()
        back = loaded_state[key].float()
        assert src.shape == back.shape
        if src.is_floating_point():
            err = (src - back).abs().max().item()
            # FP16 ulp on weights initialized U[-c, c] for SIREN ≈ 1e-3.
            assert err < 5e-3, (
                f"state-roundtrip err {err:.6f} exceeds FP16 tolerance for {key}"
            )
        else:
            assert torch.equal(src, back)


def test_network_codec_archive_metadata_preserved() -> None:
    """Metadata (hidden/layers/num_frames/...) survives export→inflate."""
    codec = SIRENVideoCodec(
        hidden=12, layers=3, omega_0=20.0,
        num_frames=2, frame_h=8, frame_w=8,
        pos_encoding_freqs=4,
    )
    archive = export_network_archive(codec)
    loaded = load_network_codec(archive)
    assert loaded.hidden == codec.hidden
    assert loaded.num_layers == codec.num_layers
    assert loaded.omega_0 == codec.omega_0
    assert loaded.num_frames == codec.num_frames
    assert loaded.frame_h == codec.frame_h
    assert loaded.frame_w == codec.frame_w
    assert loaded.pos_encoding_freqs == codec.pos_encoding_freqs
