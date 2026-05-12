"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for hi_nerv.

Proves the encode/decode contract of the HIV1 monolithic 0.bin grammar and
the substrate's forward-pass parity under fp16 + per-scale int16-quant
roundtrip across the 3-scale latent pyramid.
"""

from __future__ import annotations

import torch

from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate
from tac.substrates.hi_nerv.archive import (
    HIV1_HEADER_SIZE,
    HIV1_MAGIC,
    HIV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> HinervConfig:
    return HinervConfig(
        latent_dim_coarse=4,
        latent_dim_mid=6,
        latent_dim_fine=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: HinervConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "mid_injection_block_index": cfg.mid_injection_block_index,
        "fine_injection_block_index": cfg.fine_injection_block_index,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = HinervSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    lc = sd["latents_coarse"].clone()
    lm = sd["latents_mid"].clone()
    lf = sd["latents_fine"].clone()

    blob = pack_archive(decoder_sd, lc, lm, lf, _smoke_meta(cfg))
    arc = parse_archive(blob)

    assert arc.schema_version == HIV1_SCHEMA_VERSION
    assert blob[:4] == HIV1_MAGIC
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())

    assert arc.latents_coarse.shape == lc.shape
    assert arc.latents_mid.shape == lm.shape
    assert arc.latents_fine.shape == lf.shape

    for lat, ref in (
        (arc.latents_coarse, lc),
        (arc.latents_mid, lm),
        (arc.latents_fine, lf),
    ):
        quant_range = max(float(ref.max() - ref.min()), 1e-12)
        step = quant_range / 65534.0
        assert torch.allclose(lat, ref, atol=step * 2.0)


def test_header_size_invariant_is_33_bytes():
    assert HIV1_HEADER_SIZE == 33


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00" * 8)
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = HinervSubstrate(cfg)
    decoder_sd = {
        k: v for k, v in model.state_dict().items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    sd = model.state_dict()
    blob = bytearray(
        pack_archive(
            decoder_sd,
            sd["latents_coarse"].clone(),
            sd["latents_mid"].clone(),
            sd["latents_fine"].clone(),
            _smoke_meta(cfg),
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
    model = HinervSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    blob = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
    )
    arc = parse_archive(blob)

    rebuilt = HinervSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents_coarse.copy_(arc.latents_coarse.to(rebuilt.latents_coarse.dtype))
        rebuilt.latents_mid.copy_(arc.latents_mid.to(rebuilt.latents_mid.dtype))
        rebuilt.latents_fine.copy_(arc.latents_fine.to(rebuilt.latents_fine.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_inflate_output_no_op_proof():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = HinervSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }

    blob_a = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
        _smoke_meta(cfg),
    )

    mutated_fine = sd["latents_fine"].clone()
    mutated_fine[0, 0] = mutated_fine[0, 0] + 1.0
    blob_b = pack_archive(
        decoder_sd,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        mutated_fine,
        _smoke_meta(cfg),
    )

    assert blob_a != blob_b, "no_op_proof: mutating fine latents must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents_fine[0, 0], arc_b.latents_fine[0, 0], atol=1e-6)
    # Coarse + mid latents are unchanged
    assert torch.allclose(arc_a.latents_coarse, arc_b.latents_coarse, atol=1e-6)


def test_forward_pass_produces_unit_interval_rgb():
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = HinervSubstrate(cfg).eval()
    idx = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_three_scale_latent_pyramid_is_distinct():
    """Distinctive design check: hi_nerv has 3 separate latent tensors."""
    cfg = _smoke_cfg()
    model = HinervSubstrate(cfg)
    assert hasattr(model, "latents_coarse")
    assert hasattr(model, "latents_mid")
    assert hasattr(model, "latents_fine")
    assert model.latents_coarse.shape == (cfg.num_pairs, cfg.latent_dim_coarse)
    assert model.latents_mid.shape == (cfg.num_pairs, cfg.latent_dim_mid)
    assert model.latents_fine.shape == (cfg.num_pairs, cfg.latent_dim_fine)
    # Three distinct dims (or at least different parameters)
    n_latent_params = (
        model.latents_coarse.numel()
        + model.latents_mid.numel()
        + model.latents_fine.numel()
    )
    assert n_latent_params == cfg.num_pairs * (
        cfg.latent_dim_coarse + cfg.latent_dim_mid + cfg.latent_dim_fine
    )
