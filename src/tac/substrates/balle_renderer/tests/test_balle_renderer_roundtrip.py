# SPDX-License-Identifier: MIT
"""Catalog #91 + #139 ENCODE_INFLATE_ROUNDTRIP test for balle_renderer (β).

The β substrate roundtrip proves:

1. encode/decode parity of the monolithic 0.bin grammar (Catalog #91):
       state_dicts + latents + scales + meta -> archive bytes ->
       parsed back -> same components within fp16 + int16 quant tolerance.
2. forward-pass parity after roundtrip (the rebuilt model produces
   matching frames within int16 quant + fp16 weight rounding).
3. byte-mutation no_op_proof (Catalog #139): mutating one latent in
   the encoded bytes changes the parsed latents.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tac.substrates._shared.inflate_runtime import CAMERA_HW
from tac.substrates.balle_renderer.architecture import (
    BalleRendererConfig,
    BalleRendererSubstrate,
)
from tac.substrates.balle_renderer.archive import (
    BRV1_HEADER_SIZE,
    BRV1_MAGIC,
    BRV1_RENDER_SILENT_SIDEINFO_BLOCKER,
    BRV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from tac.substrates.balle_renderer.brv2_sideinfo import (
    BRV2_CONSUMED_SIDEINFO_CONTRACT,
    BRV2_SIDEINFO_GAIN_META_KEY,
    brv2_section_offsets,
    decode_brv2_consumed_latents,
    encode_brv2_latent_residuals,
    pack_brv2_archive,
    parse_brv2_archive,
)
from tac.substrates.balle_renderer.inflate import inflate_one_video


def _smoke_cfg() -> BalleRendererConfig:
    """Tiny config so tests run fast on CPU. Total params ~ a few K."""
    return BalleRendererConfig(
        latent_dim=8,
        hyper_latent_dim=4,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4),
        hyper_mlp_channels=(8, 8),
        sin_frequency=30.0,
        num_pairs=4,
        output_height=24,
        output_width=32,
        num_upsample_blocks=3,
    )


def _split_state_dict(
    model: BalleRendererSubstrate,
) -> tuple[
    dict[str, torch.Tensor],  # encoder (hyper_analysis.*)
    dict[str, torch.Tensor],  # decoder (latent_embed + blocks + head_rgb_*)
    dict[str, torch.Tensor],  # hyperprior (hyper_synthesis + w_prior_*)
    torch.Tensor,              # latents
]:
    """Split the substrate state_dict into the 3 archive blobs + latents."""
    sd = model.state_dict()
    enc_sd: dict[str, torch.Tensor] = {}
    dec_sd: dict[str, torch.Tensor] = {}
    hp_sd: dict[str, torch.Tensor] = {}
    latents = sd["latents"].clone()
    for k, v in sd.items():
        if k == "latents":
            continue
        if k.startswith("hyper_analysis."):
            enc_sd[k[len("hyper_analysis."):]] = v
        elif k.startswith("hyper_synthesis.") or k.startswith("w_prior_"):
            hp_sd[k] = v
        else:
            dec_sd[k] = v
    return enc_sd, dec_sd, hp_sd, latents


def _make_meta(cfg: BalleRendererConfig, *, smoke: bool = True) -> dict:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "hyper_mlp_channels": list(cfg.hyper_mlp_channels),
        "sin_frequency": cfg.sin_frequency,
        "gdn_eps": cfg.gdn_eps,
        "quantize_noise_std": cfg.quantize_noise_std,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "smoke": bool(smoke),
        "sideinfo_consumption_contract": (
            "brv1_smoke_closure_check_only"
            if smoke
            else "brv2_consumed_sideinfo_required"
        ),
    }


def _make_brv2_meta(cfg: BalleRendererConfig) -> dict:
    meta = _make_meta(cfg, smoke=False)
    meta["sideinfo_consumption_contract"] = BRV2_CONSUMED_SIDEINFO_CONTRACT
    meta[BRV2_SIDEINFO_GAIN_META_KEY] = 16.0
    return meta


def _make_brv2_blob(
    cfg: BalleRendererConfig,
    *,
    seed: int,
) -> tuple[bytes, BalleRendererSubstrate, torch.Tensor]:
    torch.manual_seed(seed)
    model = BalleRendererSubstrate(cfg).eval()
    with torch.no_grad():
        base = torch.linspace(0.75, 1.25, steps=cfg.latent_dim)
        model.latents.copy_(base.view(1, -1).repeat(cfg.num_pairs, 1))
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        hyper_latents = model.hyper_analysis(latents)
    meta = _make_brv2_meta(cfg)
    residuals = encode_brv2_latent_residuals(
        model,
        latents,
        hyper_latents,
        meta,
    )
    blob = pack_brv2_archive(
        enc_sd,
        dec_sd,
        hp_sd,
        hyper_latents,
        residuals,
        meta,
    )
    return blob, model, latents


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BalleRendererSubstrate(cfg).eval()
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    # Hyper-latent scales: synthesize via the hyper_analysis pass
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    meta = _make_meta(cfg)

    blob = pack_archive(enc_sd, dec_sd, hp_sd, latents, scales, meta)
    arc = parse_archive(blob)

    assert arc.schema_version == BRV1_SCHEMA_VERSION
    assert blob[:4] == BRV1_MAGIC

    # state_dict keys preserved on each blob
    assert set(arc.encoder_state_dict.keys()) == set(enc_sd.keys())
    assert set(arc.decoder_state_dict.keys()) == set(dec_sd.keys())
    assert set(arc.hyperprior_state_dict.keys()) == set(hp_sd.keys())

    # shapes preserved
    for k, v in enc_sd.items():
        assert arc.encoder_state_dict[k].shape == v.shape, f"enc.{k}"
    for k, v in dec_sd.items():
        assert arc.decoder_state_dict[k].shape == v.shape, f"dec.{k}"
    for k, v in hp_sd.items():
        assert arc.hyperprior_state_dict[k].shape == v.shape, f"hp.{k}"

    # fp16 roundtrip tolerance on weights
    for k, v in dec_sd.items():
        rec = arc.decoder_state_dict[k]
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2), k

    # latents shape preserved + int16 dequant within step tolerance
    assert arc.latents.shape == latents.shape
    quant_range = max(float(latents.max() - latents.min()), 1e-12)
    step = quant_range / 65534.0
    assert torch.allclose(arc.latents, latents, atol=step * 2.0)

    # scales shape preserved + int16 dequant within step tolerance
    assert arc.scales.shape == scales.shape
    quant_range_s = max(float(scales.max() - scales.min()), 1e-12)
    step_s = quant_range_s / 65534.0
    assert torch.allclose(arc.scales, scales, atol=step_s * 2.0)

    # Quantization metadata stays in the parser contract so inflate can set
    # byte-closure tolerances without reparsing private side channels.
    for key in (
        "_lat_quant_scale",
        "_lat_quant_zero_point",
        "_sca_quant_scale",
        "_sca_quant_zero_point",
    ):
        assert key in arc.meta


def test_header_size_invariant_is_35_bytes():
    """β header is 35 bytes (vs α's 21) due to hyper_dim + scales/enc/hp section lengths."""
    assert BRV1_HEADER_SIZE == 35


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
    model = BalleRendererSubstrate(cfg).eval()
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    meta = _make_meta(cfg)
    blob = bytearray(pack_archive(enc_sd, dec_sd, hp_sd, latents, scales, meta))
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_forward_pass_after_roundtrip_matches_original_within_fp16_tolerance():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = BalleRendererSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a, _rate_a = model(idx)

    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    meta = _make_meta(cfg)
    blob = pack_archive(enc_sd, dec_sd, hp_sd, latents, scales, meta)
    arc = parse_archive(blob)

    rebuilt = BalleRendererSubstrate(cfg).eval()
    # Re-prefix the encoder keys back to "hyper_analysis." and load all blobs
    merged: dict[str, torch.Tensor] = {}
    merged.update({"hyper_analysis." + k: v for k, v in arc.encoder_state_dict.items()})
    merged.update(arc.decoder_state_dict)
    merged.update(arc.hyperprior_state_dict)
    rebuilt.load_state_dict(merged, strict=False)
    with torch.no_grad():
        rebuilt.latents.copy_(arc.latents.to(rebuilt.latents.dtype))
        rgb_0_b, rgb_1_b, _rate_b = rebuilt(idx)

    # fp16 roundtrip on state_dicts + int16 quant on latents: tolerate ~5e-2
    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation no_op_proof
def test_byte_mutation_changes_inflate_output_no_op_proof():
    """Mutate one latent, prove the archive bytes differ and parsed
    latent differs after roundtrip. The full inflate-output diff is
    smoke-level for unit tests (no PIL.Image dependency in the assertion
    path); this proves the encoder's bytes-change-bytes property which
    is the no_op_proof contract.
    """
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = BalleRendererSubstrate(cfg).eval()
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    meta = _make_meta(cfg)

    blob_a = pack_archive(enc_sd, dec_sd, hp_sd, latents, scales, meta)

    mutated = latents.clone()
    mutated[0, 0] = mutated[0, 0] + 1.0  # large delta so int16 quant catches it
    blob_b = pack_archive(enc_sd, dec_sd, hp_sd, mutated, scales, meta)

    assert blob_a != blob_b, "no_op_proof: mutating latents must change archive bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)


def test_byte_mutation_on_scales_changes_archive_bytes():
    """Sister test: mutating scales must also produce different archive bytes.

    This is the hyperprior-specific arm of the no_op_proof. The β substrate's
    scales blob is now fail-closed by inflate, and mutating it MUST change
    archive bytes even before the arithmetic coder replaces the raw int16
    latent streams.
    """
    cfg = _smoke_cfg()
    torch.manual_seed(21)
    model = BalleRendererSubstrate(cfg).eval()
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    meta = _make_meta(cfg)

    blob_a = pack_archive(enc_sd, dec_sd, hp_sd, latents, scales, meta)

    mutated_scales = scales.clone()
    mutated_scales[0, 0] = mutated_scales[0, 0] + 1.0
    blob_b = pack_archive(enc_sd, dec_sd, hp_sd, latents, mutated_scales, meta)

    assert blob_a != blob_b, "no_op_proof: mutating scales must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.scales[0, 0], arc_b.scales[0, 0], atol=1e-6)


def test_inflate_accepts_closed_scales_stream(tmp_path):
    """A valid BRV1 packet should pass the scales-closure check and render."""
    cfg = BalleRendererConfig(
        latent_dim=8,
        hyper_latent_dim=4,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4),
        hyper_mlp_channels=(8, 8),
        sin_frequency=30.0,
        num_pairs=1,
        output_height=24,
        output_width=32,
        num_upsample_blocks=3,
    )
    torch.manual_seed(20)
    model = BalleRendererSubstrate(cfg).eval()
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    blob = pack_archive(enc_sd, dec_sd, hp_sd, latents, scales, _make_meta(cfg))

    out_raw = tmp_path / "valid.raw"
    frames = inflate_one_video(blob, out_raw, device="cpu")
    assert frames == 2
    assert out_raw.stat().st_size == 2 * CAMERA_HW[0] * CAMERA_HW[1] * 3


def test_inflate_refuses_non_smoke_brv1_render_silent_sideinfo(tmp_path):
    """BRV1 is not allowed for the next non-smoke monolithic archive.

    The current raw-int16 grammar stores render latents directly. Its
    hyper-latents are closure-checked, but they do not drive entropy decode or
    pixel reconstruction. Until BRV2 lands that consumed-sideinfo path, inflate
    must fail closed for non-smoke packets instead of producing a misleading
    score-bearing archive.
    """
    cfg = BalleRendererConfig(
        latent_dim=8,
        hyper_latent_dim=4,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4),
        hyper_mlp_channels=(8, 8),
        sin_frequency=30.0,
        num_pairs=1,
        output_height=24,
        output_width=32,
        num_upsample_blocks=3,
    )
    torch.manual_seed(23)
    model = BalleRendererSubstrate(cfg).eval()
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    blob = pack_archive(
        enc_sd,
        dec_sd,
        hp_sd,
        latents,
        scales,
        _make_meta(cfg, smoke=False),
    )

    out_raw = tmp_path / "non_smoke_brv1.raw"
    with pytest.raises(
        RuntimeError,
        match="BRV1 raw-int16 side-info is render-silent",
    ):
        inflate_one_video(blob, out_raw, device="cpu")
    assert not out_raw.exists()
    assert "BRV2" in BRV1_RENDER_SILENT_SIDEINFO_BLOCKER


def test_inflate_refuses_large_smoke_tagged_brv1(tmp_path):
    """A score-shaped packet cannot bypass BRV2 by setting ``smoke=true``."""
    cfg = BalleRendererConfig(
        latent_dim=8,
        hyper_latent_dim=4,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4),
        hyper_mlp_channels=(8, 8),
        sin_frequency=30.0,
        num_pairs=17,
        output_height=24,
        output_width=32,
        num_upsample_blocks=3,
    )
    torch.manual_seed(24)
    model = BalleRendererSubstrate(cfg).eval()
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    blob = pack_archive(enc_sd, dec_sd, hp_sd, latents, scales, _make_meta(cfg))

    with pytest.raises(
        RuntimeError,
        match="BRV1 raw-int16 side-info is render-silent",
    ):
        inflate_one_video(blob, tmp_path / "oversized_smoke.raw", device="cpu")


def test_inflate_rejects_mutated_scales_stream(tmp_path):
    """The hyper-latent scales section must not be dead payload.

    A scale-byte mutation changes archive bytes but should also be consumed by
    inflate. The renderer does not use scales to draw pixels directly, so the
    runtime binds them by checking they match the packaged hyper-analysis path
    and fails closed on mismatch.
    """
    cfg = _smoke_cfg()
    torch.manual_seed(22)
    model = BalleRendererSubstrate(cfg).eval()
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    meta = _make_meta(cfg)

    mutated_scales = scales.clone()
    mutated_scales[0, 0] = mutated_scales[0, 0] + 1.0
    blob = pack_archive(enc_sd, dec_sd, hp_sd, latents, mutated_scales, meta)

    out_raw = tmp_path / "mutated.raw"
    with pytest.raises(RuntimeError, match="scales stream failed closure check"):
        inflate_one_video(blob, out_raw, device="cpu")
    assert not out_raw.exists()


def test_brv2_roundtrip_reconstructs_latents_through_consumed_sideinfo():
    """BRV2 parser must not return direct main-latent authority.

    The inflate/runtime contract reconstructs main latents from
    hyper-latents plus residuals, so a caller cannot bypass the consumed
    sideinfo path by copying ``arc.latents`` like BRV1 did.
    """
    cfg = _smoke_cfg()
    blob, model, target_latents = _make_brv2_blob(cfg, seed=41)
    arc = parse_brv2_archive(blob)

    assert not hasattr(arc, "latents")
    assert arc.hyper_latents.shape == (cfg.num_pairs, cfg.hyper_latent_dim)
    assert arc.latent_residuals.shape == (cfg.num_pairs, cfg.latent_dim)

    decoded = decode_brv2_consumed_latents(model, arc).cpu()
    residual_step = float(arc.meta["_brv2_residual_quant_scale"])
    hyper_step = float(arc.meta["_brv2_hyper_quant_scale"])
    assert torch.allclose(
        decoded,
        target_latents,
        atol=max(6e-2, 4.0 * (residual_step + hyper_step)),
    )


def test_brv2_inflate_consumes_hyper_sideinfo_mutation_changes_output(tmp_path):
    """Mutating BRV2 hyper-latent bytes must change rendered output bytes."""
    cfg = BalleRendererConfig(
        latent_dim=8,
        hyper_latent_dim=4,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4),
        hyper_mlp_channels=(8, 8),
        sin_frequency=30.0,
        num_pairs=1,
        output_height=24,
        output_width=32,
        num_upsample_blocks=3,
    )
    blob, _model, _target_latents = _make_brv2_blob(cfg, seed=42)

    out_a = tmp_path / "brv2_a.raw"
    frames = inflate_one_video(blob, out_a, device="cpu")
    assert frames == 2

    offsets = brv2_section_offsets(blob)
    hyper_start, hyper_end = offsets["hyper_latents"]
    assert hyper_end - hyper_start >= 2
    mutated = bytearray(blob)
    mutated[hyper_start + 1] ^= 0x7F

    out_b = tmp_path / "brv2_b.raw"
    frames_b = inflate_one_video(bytes(mutated), out_b, device="cpu")
    assert frames_b == 2
    assert out_a.read_bytes() != out_b.read_bytes()


def test_brv2_decode_requires_declared_consumed_sideinfo_contract():
    """BRV2 decode fails closed if the explicit sideinfo contract is absent."""
    cfg = _smoke_cfg()
    blob, model, _target_latents = _make_brv2_blob(cfg, seed=43)
    arc = parse_brv2_archive(blob)
    bad_arc = replace(
        arc,
        meta={
            **arc.meta,
            "sideinfo_consumption_contract": "brv1_smoke_closure_check_only",
        },
    )

    with pytest.raises(RuntimeError, match="missing consumed-sideinfo contract"):
        decode_brv2_consumed_latents(model, bad_arc)


def test_forward_pass_returns_rate_components():
    """The β substrate must return the Ballé rate components from forward."""
    cfg = _smoke_cfg()
    torch.manual_seed(33)
    model = BalleRendererSubstrate(cfg).eval()

    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1, rate_components = model(idx)

    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    for key in ("hyper_rate", "main_rate", "total_rate"):
        assert key in rate_components, f"missing rate component {key!r}"
        # Each rate term must be a finite scalar tensor (mean nats per element)
        v = rate_components[key]
        assert v.dim() == 0, f"{key} must be 0-D scalar; got {v.shape}"
        assert torch.isfinite(v).item(), f"{key} must be finite; got {v.item()}"


def test_archive_size_smaller_than_uncompressed_sanity():
    """Sanity smoke: brotli-compressed state_dicts must produce a smaller
    archive than raw float32 weights (~rough proxy that compression is
    actually happening; not a substantive claim)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = BalleRendererSubstrate(cfg).eval()
    enc_sd, dec_sd, hp_sd, latents = _split_state_dict(model)
    with torch.no_grad():
        scales = model.hyper_analysis(latents)
    meta = _make_meta(cfg)
    blob = pack_archive(enc_sd, dec_sd, hp_sd, latents, scales, meta)

    raw_weight_bytes = sum(
        v.numel() * 4 for v in {**enc_sd, **dec_sd, **hp_sd}.values()
    )
    raw_latents_bytes = latents.numel() * 4 + scales.numel() * 4
    raw_total = raw_weight_bytes + raw_latents_bytes
    # Brotli-compressed should never be larger than raw (smoke-bound only)
    assert len(blob) < raw_total * 2, (
        f"archive {len(blob)}B is suspiciously large vs raw {raw_total}B"
    )
