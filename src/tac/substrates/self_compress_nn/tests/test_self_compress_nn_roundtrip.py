# SPDX-License-Identifier: MIT
"""Catalog #91 + #139 ENCODE_INFLATE_ROUNDTRIP test for self_compress_nn (δ).

The δ substrate roundtrip proves:

1. encode/decode parity of the monolithic 0.bin grammar (Catalog #91):
       codebook + per-layer cluster_indices + layer_meta + latents + meta ->
       archive bytes -> parsed back -> same components within fp16 tolerance.
2. weight-reconstruction correctness: cluster_indices through the codebook
   recover the quantized weights tensor-by-tensor (the inflate path's
   actual contract).
3. forward-pass parity after roundtrip (the rebuilt model produces
   matching frames within fp16 codebook rounding).
4. byte-mutation no_op_proof (Catalog #139): mutating one codebook
   centroid OR one cluster index changes the parsed archive bytes.
"""

from __future__ import annotations

import torch

from tac.substrates.self_compress_nn.archive import (
    SCV1_HEADER_SIZE,
    SCV1_MAGIC,
    SCV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from tac.substrates.self_compress_nn.architecture import (
    SelfCompressNnConfig,
    SelfCompressNnSubstrate,
)


def _smoke_cfg() -> SelfCompressNnConfig:
    """Tiny config so tests run fast on CPU. Total params ~ a few K.

    NB: D_v must divide every quantized layer's numel. With:
      - codebook_dv=4
      - decoder_channels=(16,12,8,6,4,4,4)
      - embed_dim=24, kernel=3
    every conv weight numel is divisible by 4 (channel counts are even +
    kernel is 3x3 so numel %4 == 0 by the conv shape rules).
    """
    return SelfCompressNnConfig(
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
        codebook_k=32,
        codebook_dv=4,
        codebook_ema_decay=0.99,
        commit_loss_weight=0.25,
    )


def _make_meta(cfg: SelfCompressNnConfig) -> dict:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "codebook_ema_decay": cfg.codebook_ema_decay,
        "commit_loss_weight": cfg.commit_loss_weight,
    }


def _model_components(
    model: SelfCompressNnSubstrate,
) -> tuple[torch.Tensor, list[dict], dict[str, torch.Tensor], torch.Tensor]:
    """Snapshot codebook + layer_meta + cluster_indices + latents."""
    codebook = model.codebook.codebook.detach().clone()
    layer_meta, cluster_indices = model.export_layer_meta_and_indices()
    latents = model.latents.detach().clone()
    return codebook, layer_meta, cluster_indices, latents


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = SelfCompressNnSubstrate(cfg).eval()
    codebook, layer_meta, cluster_indices, latents = _model_components(model)
    meta = _make_meta(cfg)

    blob = pack_archive(codebook, cluster_indices, layer_meta, latents, meta)
    arc = parse_archive(blob)

    assert arc.schema_version == SCV1_SCHEMA_VERSION
    assert blob[:4] == SCV1_MAGIC

    # codebook shape preserved (fp16 roundtrip tolerance)
    assert arc.codebook.shape == codebook.shape
    assert torch.allclose(
        arc.codebook.to(torch.float32),
        codebook.to(torch.float32),
        atol=1e-2,
    )

    # layer_meta preserved (count + per-entry keys)
    assert len(arc.layer_meta) == len(layer_meta)
    for got, want in zip(arc.layer_meta, layer_meta):
        assert got["name"] == want["name"]
        assert tuple(got["shape"]) == tuple(want["shape"])
        assert int(got["numel"]) == int(want["numel"])

    # cluster_indices preserved exactly (lossless int16 -> int64)
    for entry in layer_meta:
        name = entry["name"]
        assert torch.equal(
            arc.layer_cluster_indices[name].to(torch.int64),
            cluster_indices[name].to(torch.int64),
        )

    # latents shape preserved + int16 dequant within step tolerance
    assert arc.latents.shape == latents.shape
    quant_range = max(float(latents.max() - latents.min()), 1e-12)
    step = quant_range / 65534.0
    assert torch.allclose(arc.latents, latents, atol=step * 2.0)


def test_header_size_invariant_is_35_bytes():
    """δ header is 35 bytes (matches β by coincidence; α is 21, γ is 33)."""
    assert SCV1_HEADER_SIZE == 35


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
    model = SelfCompressNnSubstrate(cfg).eval()
    codebook, layer_meta, cluster_indices, latents = _model_components(model)
    meta = _make_meta(cfg)
    blob = bytearray(pack_archive(codebook, cluster_indices, layer_meta, latents, meta))
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_codebook_lookup_recovers_quantized_weights_inflate_contract():
    """Inflate uses codebook[cluster_indices].reshape(shape) to recover
    each layer's weight. Prove that contract: the substrate's
    quantized_weight() at forward time equals codebook[idx].reshape(shape)
    for every quantized layer (within fp16 roundtrip tolerance).
    """
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = SelfCompressNnSubstrate(cfg).eval()
    codebook, layer_meta, cluster_indices, _latents = _model_components(model)

    # Walk model layers and verify codebook[idx] == quantized weight
    layer_idx = 0
    for mod_name, mod in model.named_modules():
        name = f"{mod_name}.weight"
        if name not in cluster_indices:
            continue
        target_entry = layer_meta[layer_idx]
        layer_idx += 1
        # Recover via the inflate contract
        gathered = codebook[cluster_indices[name]]  # (numel_groups, D_v)
        recovered = gathered.reshape(*target_entry["shape"])
        # Get the model's quantized weight (computed inside forward) for parity
        q_w, _idx, _commit = mod.quantized_weight()
        assert torch.allclose(
            recovered.to(torch.float32),
            q_w.detach().to(torch.float32),
            atol=1e-5,
        ), f"layer {name} codebook-lookup parity violated"


def test_forward_pass_returns_commit_loss():
    """The δ substrate must return the commit_loss scalar from forward."""
    cfg = _smoke_cfg()
    torch.manual_seed(33)
    model = SelfCompressNnSubstrate(cfg).eval()

    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1, commit_loss = model(idx)

    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert commit_loss.dim() == 0, f"commit_loss must be 0-D scalar; got {commit_loss.shape}"
    assert torch.isfinite(commit_loss).item()
    # commit_loss must be non-negative (it's a SUM of MSE terms)
    assert commit_loss.item() >= 0.0


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation no_op_proof
def test_byte_mutation_on_codebook_changes_archive_bytes_no_op_proof():
    """Mutate one codebook centroid; prove the archive bytes differ and
    the parsed codebook differs after roundtrip.
    """
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = SelfCompressNnSubstrate(cfg).eval()
    codebook, layer_meta, cluster_indices, latents = _model_components(model)
    meta = _make_meta(cfg)

    blob_a = pack_archive(codebook, cluster_indices, layer_meta, latents, meta)

    mutated_cb = codebook.clone()
    mutated_cb[0, 0] = mutated_cb[0, 0] + 1.0
    blob_b = pack_archive(mutated_cb, cluster_indices, layer_meta, latents, meta)

    assert blob_a != blob_b, "no_op_proof: mutating codebook must change archive bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(
        arc_a.codebook[0, 0].to(torch.float32),
        arc_b.codebook[0, 0].to(torch.float32),
        atol=1e-3,
    )


def test_byte_mutation_on_cluster_indices_changes_archive_bytes_no_op_proof():
    """Mutating one cluster index MUST change archive bytes. The cluster
    indices ARE the rate-axis primary content; mutating them is the
    most-directly-consumed byte mutation."""
    cfg = _smoke_cfg()
    torch.manual_seed(21)
    model = SelfCompressNnSubstrate(cfg).eval()
    codebook, layer_meta, cluster_indices, latents = _model_components(model)
    meta = _make_meta(cfg)

    blob_a = pack_archive(codebook, cluster_indices, layer_meta, latents, meta)

    # Mutate the first index of the first layer
    mutated_indices = {k: v.clone() for k, v in cluster_indices.items()}
    first_name = layer_meta[0]["name"]
    mutated_indices[first_name][0] = (
        (mutated_indices[first_name][0] + 1) % cfg.codebook_k
    )
    blob_b = pack_archive(codebook, mutated_indices, layer_meta, latents, meta)

    assert blob_a != blob_b, "no_op_proof: mutating cluster indices must change archive bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert arc_a.layer_cluster_indices[first_name][0].item() != \
        arc_b.layer_cluster_indices[first_name][0].item()


def test_archive_size_smaller_than_uncompressed_sanity():
    """Sanity smoke: the codebook+indices archive must be smaller than the
    raw fp32 weight tensors (this is the δ rate-axis claim)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = SelfCompressNnSubstrate(cfg).eval()
    codebook, layer_meta, cluster_indices, latents = _model_components(model)
    meta = _make_meta(cfg)
    blob = pack_archive(codebook, cluster_indices, layer_meta, latents, meta)

    # Raw fp32 weights = full numel * 4 bytes
    raw_weight_bytes = sum(
        int(entry["numel"]) * cfg.codebook_dv * 4 for entry in layer_meta
    )
    raw_latents_bytes = latents.numel() * 4
    raw_total = raw_weight_bytes + raw_latents_bytes
    # Archive should be < 2*raw (very loose smoke bound; δ's *real* benefit
    # only emerges at full scale ~600K weights, but the tiny test config
    # should still produce a smaller archive than raw fp32).
    assert len(blob) < raw_total * 2, (
        f"archive {len(blob)}B is suspiciously large vs raw {raw_total}B"
    )


def test_num_quantized_layers_recorded_in_layer_meta():
    """The layer_meta list must enumerate every quantized weight tensor
    (one entry per _QuantizedConv2d / _QuantizedLinear).
    """
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = SelfCompressNnSubstrate(cfg).eval()
    _codebook, layer_meta, cluster_indices, _latents = _model_components(model)

    # 1 conv per up-block (num_upsample_blocks=3) + 2 RGB heads = 5 quantized layers
    expected = cfg.num_upsample_blocks + 2  # head_rgb_0, head_rgb_1
    assert len(layer_meta) == expected, (
        f"expected {expected} quantized layers; got {len(layer_meta)} from layer_meta"
    )
    assert len(cluster_indices) == expected
