# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for ff_nerv.

Proves the encode/decode contract of the FFV1 monolithic 0.bin grammar: a
trained state_dict + latents + meta -> archive bytes -> parsed back to
(state_dict, latents, meta) — and forward on the reconstructed model
produces the same outputs (within fp16 + int16-quant rounding) as the
original.
"""

from __future__ import annotations

import torch

from tac.substrates.ff_nerv.architecture import FfnervConfig, FfnervSubstrate
from tac.substrates.ff_nerv.archive import (
    FFV1_HEADER_SIZE,
    FFV1_MAGIC,
    FFV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> FfnervConfig:
    """Tiny config for fast CPU tests."""
    return FfnervConfig(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=2,
        initial_grid_w=2,
        decoder_channels=(16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=2,
        num_pairs=3,
        output_height=16,
        output_width=24,
        freq_grid_h=8,
        freq_grid_w=8,
    )


def _smoke_meta(cfg: FfnervConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = FfnervSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()

    blob = pack_archive(
        decoder_sd,
        latents,
        _smoke_meta(cfg),
        freq_grid_h=cfg.freq_grid_h,
        freq_grid_w=cfg.freq_grid_w,
    )
    arc = parse_archive(blob)

    assert arc.schema_version == FFV1_SCHEMA_VERSION
    assert blob[:4] == FFV1_MAGIC
    assert arc.freq_grid_h == cfg.freq_grid_h
    assert arc.freq_grid_w == cfg.freq_grid_w

    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())
    for k, v in decoder_sd.items():
        rec = arc.decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    assert arc.latents.shape == latents.shape
    quant_range = max(float(latents.max() - latents.min()), 1e-12)
    step = quant_range / 65534.0
    assert torch.allclose(arc.latents, latents, atol=step * 2.0)


def test_header_size_invariant_is_25_bytes():
    assert FFV1_HEADER_SIZE == 25


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00" * 10)
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = FfnervSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    blob = bytearray(
        pack_archive(
            decoder_sd,
            latents,
            _smoke_meta(cfg),
            freq_grid_h=cfg.freq_grid_h,
            freq_grid_w=cfg.freq_grid_w,
        )
    )
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_forward_pass_after_roundtrip_matches_original_within_tolerance():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = FfnervSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()
    blob = pack_archive(
        decoder_sd,
        latents,
        _smoke_meta(cfg),
        freq_grid_h=cfg.freq_grid_h,
        freq_grid_w=cfg.freq_grid_w,
    )
    arc = parse_archive(blob)

    rebuilt = FfnervSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents.copy_(arc.latents.to(rebuilt.latents.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    # fp16 + int16-quant roundtrip: tolerate ~5e-2 after sigmoid
    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_inflate_output_no_op_proof():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = FfnervSubstrate(cfg).eval()
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()

    blob_a = pack_archive(
        decoder_sd,
        latents,
        _smoke_meta(cfg),
        freq_grid_h=cfg.freq_grid_h,
        freq_grid_w=cfg.freq_grid_w,
    )

    mutated = latents.clone()
    mutated[0, 0] = mutated[0, 0] + 1.0
    blob_b = pack_archive(
        decoder_sd,
        mutated,
        _smoke_meta(cfg),
        freq_grid_h=cfg.freq_grid_h,
        freq_grid_w=cfg.freq_grid_w,
    )

    assert blob_a != blob_b, "no_op_proof: mutating latents must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)


def test_forward_pass_produces_unit_interval_rgb():
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = FfnervSubstrate(cfg).eval()
    idx = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_param_count_is_finite_and_positive():
    cfg = _smoke_cfg()
    model = FfnervSubstrate(cfg)
    n = model.num_parameters()
    assert n > 0, f"param count must be positive; got {n}"
