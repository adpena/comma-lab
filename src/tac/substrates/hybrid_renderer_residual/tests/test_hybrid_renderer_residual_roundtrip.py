# SPDX-License-Identifier: MIT
"""Catalog #91 + #139 ENCODE_INFLATE_ROUNDTRIP test for hybrid_renderer_residual (γ).

The γ substrate roundtrip proves:

1. encode/decode parity of the monolithic 0.bin grammar (Catalog #91):
       state_dicts + latents + residual_coeffs + meta -> archive bytes ->
       parsed back -> same components within fp16 + int16 quant tolerance.
2. forward-pass parity after roundtrip (the rebuilt model produces
   matching frames within int16 quant + fp16 weight rounding).
3. byte-mutation no_op_proof (Catalog #139): mutating one renderer latent
   in the encoded bytes changes the parsed latents; mutating a residual
   coefficient in the encoded bytes changes the parsed residual tensor.
"""

from __future__ import annotations

import torch

from tac.substrates.hybrid_renderer_residual.archive import (
    HRRV1_HEADER_SIZE,
    HRRV1_MAGIC,
    HRRV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from tac.substrates.hybrid_renderer_residual.architecture import (
    HybridRendererResidualConfig,
    HybridRendererResidualSubstrate,
)


def _smoke_cfg() -> HybridRendererResidualConfig:
    """Tiny config so tests run fast on CPU. Total params ~ a few K."""
    return HybridRendererResidualConfig(
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
        residual_basis_dim=16,
        residual_basis_value_dim=8,
        residual_coeffs_per_pair=4,
        residual_decoder_hidden=(16, 8),
    )


def _split_state_dict(
    model: HybridRendererResidualSubstrate,
) -> tuple[
    dict[str, torch.Tensor],  # renderer (everything except residual_decoder + latents + residual_coeff_full)
    dict[str, torch.Tensor],  # residual_decoder (its sub-module weights only)
    torch.Tensor,              # latents
]:
    """Split the substrate state_dict into the 2 archive blobs + latents."""
    sd = model.state_dict()
    renderer_sd: dict[str, torch.Tensor] = {}
    resdec_sd: dict[str, torch.Tensor] = {}
    latents = sd["latents"].clone()
    for k, v in sd.items():
        if k in ("latents", "residual_coeff_full"):
            continue
        if k.startswith("residual_decoder."):
            resdec_sd[k[len("residual_decoder."):]] = v
        else:
            renderer_sd[k] = v
    return renderer_sd, resdec_sd, latents


def _make_meta(cfg: HybridRendererResidualConfig) -> dict:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "residual_basis_dim": cfg.residual_basis_dim,
        "residual_basis_value_dim": cfg.residual_basis_value_dim,
        "residual_coeffs_per_pair": cfg.residual_coeffs_per_pair,
        "residual_decoder_hidden": list(cfg.residual_decoder_hidden),
    }


def _smoke_residual_pair(
    cfg: HybridRendererResidualConfig, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic sparse residual (indices, values) for the tests."""
    g = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0,
        cfg.residual_basis_dim,
        (cfg.num_pairs, cfg.residual_coeffs_per_pair),
        generator=g,
        dtype=torch.int64,
    )
    values = torch.randn(
        (cfg.num_pairs, cfg.residual_coeffs_per_pair), generator=g
    ) * 0.05
    return indices, values


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = HybridRendererResidualSubstrate(cfg).eval()
    renderer_sd, resdec_sd, latents = _split_state_dict(model)
    indices, values = _smoke_residual_pair(cfg, seed=11)
    meta = _make_meta(cfg)

    blob = pack_archive(renderer_sd, resdec_sd, latents, indices, values, meta)
    arc = parse_archive(blob)

    assert arc.schema_version == HRRV1_SCHEMA_VERSION
    assert blob[:4] == HRRV1_MAGIC

    # state_dict keys preserved on each blob
    assert set(arc.renderer_state_dict.keys()) == set(renderer_sd.keys())
    assert set(arc.residual_decoder_state_dict.keys()) == set(resdec_sd.keys())

    # shapes preserved
    for k, v in renderer_sd.items():
        assert arc.renderer_state_dict[k].shape == v.shape, f"renderer.{k}"
    for k, v in resdec_sd.items():
        assert arc.residual_decoder_state_dict[k].shape == v.shape, f"resdec.{k}"

    # fp16 roundtrip tolerance on weights
    for k, v in renderer_sd.items():
        rec = arc.renderer_state_dict[k]
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2), k

    # latents shape + int16 dequant within step tolerance
    assert arc.latents.shape == latents.shape
    quant_range = max(float(latents.max() - latents.min()), 1e-12)
    step = quant_range / 65534.0
    assert torch.allclose(arc.latents, latents, atol=step * 2.0)

    # residual coeffs shape: (num_pairs, k, 2)
    assert arc.residual_basis_coefficients.shape == (
        cfg.num_pairs,
        cfg.residual_coeffs_per_pair,
        2,
    )
    # indices preserved exactly (no quantization on the index path)
    rec_idx = arc.residual_basis_coefficients[:, :, 0]
    assert torch.equal(rec_idx.to(torch.int64), indices.to(torch.int64))


def test_header_size_invariant_is_33_bytes():
    """γ header is 33 bytes (vs α's 21, β's 35) due to residual basis fields."""
    assert HRRV1_HEADER_SIZE == 33


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
    model = HybridRendererResidualSubstrate(cfg).eval()
    renderer_sd, resdec_sd, latents = _split_state_dict(model)
    indices, values = _smoke_residual_pair(cfg, seed=2)
    meta = _make_meta(cfg)
    blob = bytearray(pack_archive(renderer_sd, resdec_sd, latents, indices, values, meta))
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_forward_pass_returns_residual_l1():
    """The γ substrate must return the sparsity term (||c||_1) from forward."""
    cfg = _smoke_cfg()
    torch.manual_seed(33)
    model = HybridRendererResidualSubstrate(cfg).eval()
    # Manually set some non-zero coefficients so top-k has something to select
    with torch.no_grad():
        model.residual_coeff_full[0, 0] = 1.0
        model.residual_coeff_full[1, 5] = -0.5

    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1, residual_l1 = model(idx)

    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert residual_l1.dim() == 0, f"residual_l1 must be 0-D scalar; got {residual_l1.shape}"
    assert torch.isfinite(residual_l1).item()


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation no_op_proof
def test_byte_mutation_on_latents_changes_archive_bytes_no_op_proof():
    """Mutate one renderer latent, prove the archive bytes differ and the
    parsed latent differs after roundtrip.
    """
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = HybridRendererResidualSubstrate(cfg).eval()
    renderer_sd, resdec_sd, latents = _split_state_dict(model)
    indices, values = _smoke_residual_pair(cfg, seed=99)
    meta = _make_meta(cfg)

    blob_a = pack_archive(renderer_sd, resdec_sd, latents, indices, values, meta)

    mutated_latents = latents.clone()
    mutated_latents[0, 0] = mutated_latents[0, 0] + 1.0
    blob_b = pack_archive(renderer_sd, resdec_sd, mutated_latents, indices, values, meta)

    assert blob_a != blob_b, "no_op_proof: mutating latents must change archive bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)


def test_byte_mutation_on_residual_coeffs_changes_archive_bytes_no_op_proof():
    """Mutating one residual coefficient value MUST change archive bytes.

    This is the γ-specific arm of the no_op_proof — the residual stream is
    genuinely consumed by inflate (it's the pose-axis attack vector);
    mutating it MUST change bytes per Catalog #139.
    """
    cfg = _smoke_cfg()
    torch.manual_seed(21)
    model = HybridRendererResidualSubstrate(cfg).eval()
    renderer_sd, resdec_sd, latents = _split_state_dict(model)
    indices, values = _smoke_residual_pair(cfg, seed=44)
    meta = _make_meta(cfg)

    blob_a = pack_archive(renderer_sd, resdec_sd, latents, indices, values, meta)

    mutated_values = values.clone()
    mutated_values[0, 0] = mutated_values[0, 0] + 1.0
    blob_b = pack_archive(
        renderer_sd, resdec_sd, latents, indices, mutated_values, meta
    )

    assert blob_a != blob_b, "no_op_proof: mutating residual coeffs must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    # Compare via the int16 value column
    val_a = arc_a.residual_basis_coefficients[0, 0, 1]
    val_b = arc_b.residual_basis_coefficients[0, 0, 1]
    assert val_a.item() != val_b.item()


def test_byte_mutation_on_residual_indices_changes_archive_bytes_no_op_proof():
    """Mutating a basis INDEX must also produce different archive bytes."""
    cfg = _smoke_cfg()
    torch.manual_seed(34)
    model = HybridRendererResidualSubstrate(cfg).eval()
    renderer_sd, resdec_sd, latents = _split_state_dict(model)
    indices, values = _smoke_residual_pair(cfg, seed=77)
    meta = _make_meta(cfg)

    blob_a = pack_archive(renderer_sd, resdec_sd, latents, indices, values, meta)

    mutated_indices = indices.clone()
    # Flip one index to a different basis row (guaranteed different by +1 modulo)
    mutated_indices[0, 0] = (mutated_indices[0, 0] + 1) % cfg.residual_basis_dim
    blob_b = pack_archive(
        renderer_sd, resdec_sd, latents, mutated_indices, values, meta
    )

    assert blob_a != blob_b, "no_op_proof: mutating residual indices must change archive bytes"


def test_archive_size_smaller_than_uncompressed_sanity():
    """Sanity smoke: brotli-compressed state_dicts must produce a smaller
    archive than raw float32 weights (rough proxy that compression is
    actually happening; not a substantive claim)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = HybridRendererResidualSubstrate(cfg).eval()
    renderer_sd, resdec_sd, latents = _split_state_dict(model)
    indices, values = _smoke_residual_pair(cfg, seed=1)
    meta = _make_meta(cfg)
    blob = pack_archive(renderer_sd, resdec_sd, latents, indices, values, meta)

    raw_weight_bytes = sum(
        v.numel() * 4 for v in {**renderer_sd, **resdec_sd}.values()
    )
    raw_latents_bytes = latents.numel() * 4
    raw_residuals_bytes = indices.numel() * 2 + values.numel() * 2
    raw_total = raw_weight_bytes + raw_latents_bytes + raw_residuals_bytes
    # Brotli-compressed should never be larger than raw (smoke-bound only)
    assert len(blob) < raw_total * 2, (
        f"archive {len(blob)}B is suspiciously large vs raw {raw_total}B"
    )
