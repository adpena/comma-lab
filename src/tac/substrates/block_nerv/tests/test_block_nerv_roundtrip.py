# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for block_nerv.

Mirrors ``sane_hnerv/tests/test_sane_hnerv_roundtrip.py`` adapted for the
BNV1 grammar (3 quantized per-pair sections: latents int16, lora_bias int16,
lora_gain int8).
"""

from __future__ import annotations

import torch

from tac.substrates.block_nerv.archive import (
    BNV1_HEADER_SIZE,
    BNV1_MAGIC,
    BNV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from tac.substrates.block_nerv.architecture import (
    BlockNervConfig,
    BlockNervSubstrate,
)
from tac.substrates.block_nerv.score_aware_loss import (
    BlockNervScoreAwareLoss,
    BlockScoreAwareLossWeights,
)


def _smoke_cfg() -> BlockNervConfig:
    """Tiny config so tests run fast on CPU."""
    return BlockNervConfig(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4),
        sin_frequency=30.0,
        num_pairs=4,
        output_height=24,
        output_width=32,
        num_upsample_blocks=3,
    )


def _meta_from_cfg(cfg: BlockNervConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
    }


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BlockNervSubstrate(cfg)
    sd = model.state_dict()
    skip = {"latents", "lora_latent_bias", "lora_embed_gain"}
    base_sd = {k: v for k, v in sd.items() if k not in skip}
    latents = sd["latents"].clone()
    bias = sd["lora_latent_bias"].clone() + 0.01  # nudge off zero
    gain = sd["lora_embed_gain"].clone() + 0.02
    meta = _meta_from_cfg(cfg)

    blob = pack_archive(base_sd, latents, bias, gain, meta)
    arc = parse_archive(blob)

    assert arc.schema_version == BNV1_SCHEMA_VERSION
    assert blob[:4] == BNV1_MAGIC

    assert set(arc.base_decoder_state_dict.keys()) == set(base_sd.keys())
    for k, v in base_sd.items():
        rec = arc.base_decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    assert arc.latents.shape == latents.shape
    lat_range = max(float(latents.max() - latents.min()), 1e-12)
    lat_step = lat_range / 65534.0
    assert torch.allclose(arc.latents, latents, atol=lat_step * 2.0)

    assert arc.lora_latent_bias.shape == bias.shape
    bias_range = max(float(bias.max() - bias.min()), 1e-12)
    bias_step = bias_range / 65534.0
    assert torch.allclose(arc.lora_latent_bias, bias, atol=bias_step * 2.0)

    assert arc.lora_embed_gain.shape == gain.shape
    gain_range = max(float(gain.max() - gain.min()), 1e-12)
    gain_step = gain_range / 254.0
    assert torch.allclose(arc.lora_embed_gain, gain, atol=gain_step * 2.0)


def test_header_size_invariant_is_31_bytes():
    assert BNV1_HEADER_SIZE == 31


def test_parse_archive_rejects_short_blob():
    try:
        parse_archive(b"\x00")
    except ValueError as exc:
        assert "too short" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on short blob")


def test_parse_archive_rejects_wrong_magic():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BlockNervSubstrate(cfg)
    sd = model.state_dict()
    skip = {"latents", "lora_latent_bias", "lora_embed_gain"}
    base_sd = {k: v for k, v in sd.items() if k not in skip}
    latents = sd["latents"].clone()
    bias = sd["lora_latent_bias"].clone() + 0.01
    gain = sd["lora_embed_gain"].clone() + 0.02
    meta = _meta_from_cfg(cfg)
    blob = bytearray(pack_archive(base_sd, latents, bias, gain, meta))
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_pack_rejects_bias_shape_mismatch():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BlockNervSubstrate(cfg)
    sd = model.state_dict()
    skip = {"latents", "lora_latent_bias", "lora_embed_gain"}
    base_sd = {k: v for k, v in sd.items() if k not in skip}
    latents = sd["latents"].clone()
    bad_bias = torch.zeros(cfg.num_pairs + 1, cfg.latent_dim)
    gain = sd["lora_embed_gain"].clone() + 0.02
    meta = _meta_from_cfg(cfg)
    try:
        pack_archive(base_sd, latents, bad_bias, gain, meta)
    except ValueError as exc:
        assert "lora_latent_bias shape" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bias shape mismatch")


def test_pack_rejects_gain_num_pairs_mismatch():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BlockNervSubstrate(cfg)
    sd = model.state_dict()
    skip = {"latents", "lora_latent_bias", "lora_embed_gain"}
    base_sd = {k: v for k, v in sd.items() if k not in skip}
    latents = sd["latents"].clone()
    bias = sd["lora_latent_bias"].clone() + 0.01
    bad_gain = torch.zeros(cfg.num_pairs + 2, cfg.embed_dim)
    meta = _meta_from_cfg(cfg)
    try:
        pack_archive(base_sd, latents, bias, bad_gain, meta)
    except ValueError as exc:
        assert "lora_embed_gain num_pairs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on gain num_pairs mismatch")


def test_forward_pass_after_roundtrip_matches_original_within_fp16_tolerance():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = BlockNervSubstrate(cfg).eval()
    # Set per-pair LoRA tensors away from defaults so they have to roundtrip
    with torch.no_grad():
        model.lora_latent_bias.add_(0.05)
        model.lora_embed_gain.add_(0.1)

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    skip = {"latents", "lora_latent_bias", "lora_embed_gain"}
    base_sd = {k: v for k, v in sd.items() if k not in skip}
    latents = sd["latents"].clone()
    bias = sd["lora_latent_bias"].clone()
    gain = sd["lora_embed_gain"].clone()
    meta = _meta_from_cfg(cfg)
    blob = pack_archive(base_sd, latents, bias, gain, meta)
    arc = parse_archive(blob)

    rebuilt = BlockNervSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.base_decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents.copy_(arc.latents.to(rebuilt.latents.dtype))
        rebuilt.lora_latent_bias.copy_(
            arc.lora_latent_bias.to(rebuilt.lora_latent_bias.dtype)
        )
        rebuilt.lora_embed_gain.copy_(
            arc.lora_embed_gain.to(rebuilt.lora_embed_gain.dtype)
        )
        rgb_0_b, rgb_1_b = rebuilt(idx)

    # int8 gain has coarser quantization; relax tolerance vs sane_hnerv (5e-2 -> 7e-2)
    assert torch.allclose(rgb_0_a, rgb_0_b, atol=7e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=7e-2)


# Catalog #139 byte-mutation no_op_proof
def test_byte_mutation_changes_inflate_output_no_op_proof():
    """Mutating ANY of {latents, lora_latent_bias, lora_embed_gain} must
    produce different archive bytes AND a different parsed archive."""
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = BlockNervSubstrate(cfg).eval()
    sd = model.state_dict()
    skip = {"latents", "lora_latent_bias", "lora_embed_gain"}
    base_sd = {k: v for k, v in sd.items() if k not in skip}
    latents = sd["latents"].clone()
    bias = sd["lora_latent_bias"].clone() + 0.01
    gain = sd["lora_embed_gain"].clone() + 0.02
    meta = _meta_from_cfg(cfg)

    blob_a = pack_archive(base_sd, latents, bias, gain, meta)

    # Mutate latents
    mut_lat = latents.clone()
    mut_lat[0, 0] = mut_lat[0, 0] + 1.0
    blob_b = pack_archive(base_sd, mut_lat, bias, gain, meta)
    assert blob_a != blob_b, "no_op_proof: latent mutation must change bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)

    # Mutate LoRA bias
    mut_bias = bias.clone()
    mut_bias[0, 0] = mut_bias[0, 0] + 0.5
    blob_c = pack_archive(base_sd, latents, mut_bias, gain, meta)
    assert blob_a != blob_c, "no_op_proof: lora_bias mutation must change bytes"
    arc_c = parse_archive(blob_c)
    assert not torch.allclose(
        arc_a.lora_latent_bias[0, 0], arc_c.lora_latent_bias[0, 0], atol=1e-6
    )

    # Mutate LoRA gain
    mut_gain = gain.clone()
    mut_gain[0, 0] = mut_gain[0, 0] + 0.5
    blob_d = pack_archive(base_sd, latents, bias, mut_gain, meta)
    assert blob_a != blob_d, "no_op_proof: lora_gain mutation must change bytes"
    arc_d = parse_archive(blob_d)
    assert not torch.allclose(
        arc_a.lora_embed_gain[0, 0], arc_d.lora_embed_gain[0, 0], atol=1e-3
    )


def test_score_aware_loss_smoke():
    """L0 SKETCH smoke for the Lagrangian. Identity-stub scorers."""
    torch.manual_seed(31)

    class _SegStub(torch.nn.Module):
        """Identity-ish stub: takes (B, T=1, C, H, W) and returns (B, 5, h, w)
        as a linear-projection-of-input so grad flows through.
        """

        def forward(self, x):
            # x: (B, T=1, C, H, W) -> reshape (B, C, H, W) then project to 5 channels.
            x = x[:, 0]  # (B, C, H, W)
            # 5-channel projection by repeating C=3 -> 5 via padding (preserves grad)
            pad = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])
            return torch.cat([x, pad], dim=1)

    class _PoseStub(torch.nn.Module):
        """Identity-ish stub: returns (B, 12) projection of input mean."""

        def forward(self, x):
            # x: (B, 6, H, W) -> per-channel mean -> (B, 6), pad to (B, 12)
            m = x.mean(dim=(-2, -1))  # (B, 6)
            pad = torch.zeros(m.shape[0], 6)
            return torch.cat([m, pad], dim=1)

    seg = _SegStub()
    pose = _PoseStub()
    weights = BlockScoreAwareLossWeights()
    loss_fn = BlockNervScoreAwareLoss(seg, pose, weights)

    rgb_0 = torch.rand(2, 3, 12, 16, requires_grad=True)
    rgb_1 = torch.rand(2, 3, 12, 16, requires_grad=True)
    gt_0 = torch.rand(2, 3, 12, 16)
    gt_1 = torch.rand(2, 3, 12, 16)
    archive_bytes = torch.tensor(120_000.0)

    loss, parts = loss_fn(rgb_0, rgb_1, gt_0, gt_1, archive_bytes)
    assert "rate_term" in parts
    assert "seg_term" in parts
    assert "pose_term" in parts
    assert torch.isfinite(loss)
    loss.backward()
    assert rgb_0.grad is not None
    assert rgb_1.grad is not None
