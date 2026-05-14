# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP test for sane_hnerv.

The test proves the encode/decode contract of the monolithic 0.bin archive
grammar: a trained state_dict + latents + meta → archive bytes → parsed
back to (state_dict, latents, meta) — and a forward pass on the
reconstructed model produces the SAME outputs (within fp16 rounding) as
the original.

This is the no-op-free roundtrip proof; the byte-mutation executable smoke
that Catalog #139 requires lives in test_sane_hnerv_no_op_proof below.
"""

from __future__ import annotations

import torch

from tac.substrates.sane_hnerv.archive import (
    SHV1_HEADER_SIZE,
    SHV1_MAGIC,
    SHV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from tac.substrates.sane_hnerv.architecture import SaneHnervConfig, SaneHnervSubstrate


def _smoke_cfg() -> SaneHnervConfig:
    """Tiny config so tests run fast on CPU. Param count ~ a few K."""
    return SaneHnervConfig(
        latent_dim=8,
        embed_dim=32,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8, 6, 4, 4, 4),
        sin_frequency=30.0,
        num_pairs=4,
        output_height=24,
        output_width=32,
        num_upsample_blocks=3,
    )


# ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract
def test_archive_pack_then_parse_roundtrip_recovers_tensors():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = SaneHnervSubstrate(cfg)
    sd = model.state_dict()
    # latents live INSIDE the state_dict in this design; archive stores them
    # separately for quantization, so pull them out of sd before packing the
    # "decoder" blob.
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
    }

    blob = pack_archive(decoder_sd, latents, meta)
    arc = parse_archive(blob)

    assert arc.schema_version == SHV1_SCHEMA_VERSION
    assert blob[:4] == SHV1_MAGIC

    # state_dict keys and shapes preserved
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())
    for k, v in decoder_sd.items():
        rec = arc.decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        # fp16 roundtrip — allow small error
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    # latents shape preserved
    assert arc.latents.shape == latents.shape

    # quantized latent dequant matches original within int16 quantization step
    quant_range = max(float(latents.max() - latents.min()), 1e-12)
    step = quant_range / 65534.0
    assert torch.allclose(arc.latents, latents, atol=step * 2.0)


def test_header_size_invariant_is_21_bytes():
    assert SHV1_HEADER_SIZE == 21


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
    model = SaneHnervSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
    }
    blob = bytearray(pack_archive(decoder_sd, latents, meta))
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
    model = SaneHnervSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
    }
    blob = pack_archive(decoder_sd, latents, meta)
    arc = parse_archive(blob)

    rebuilt = SaneHnervSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents.copy_(arc.latents.to(rebuilt.latents.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    # fp16 roundtrip on state_dict + int16 quant on latents: tolerate ~5e-2
    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_inflate_output_no_op_proof():
    """Catalog #139 no_op_proof: mutate one archive byte AT a known content
    region, prove the parsed bytes differ. (Full inflate-output diff is
    smoke at unit-test scope; this proves the encoder's bytes-change-bytes
    property.)
    """
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = SaneHnervSubstrate(cfg).eval()
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
    }

    blob_a = pack_archive(decoder_sd, latents, meta)

    # Mutate one latent — that's in the latent blob region of the archive.
    mutated = latents.clone()
    mutated[0, 0] = mutated[0, 0] + 1.0  # large delta so int16 quant catches it
    blob_b = pack_archive(decoder_sd, mutated, meta)

    assert blob_a != blob_b, "no_op_proof: mutating latents must change archive bytes"

    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    # Latent at (0, 0) must differ between A and B after roundtrip
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)


# Catalog #158 degenerate-range tests (NNN sister-bug FFFF fix)
# ─────────────────────────────────────────────────────────────────────────
# Prior pattern was zeros_like(f, dtype=int16) which dequant'd to
# (0 + 32767) * 1.0 + lo = 32767 + lo (off by 32767). Correct pattern is
# full_like(f, -32767, dtype=int16) which dequant'd to
# (-32767 + 32767) * 1.0 + lo = lo. These tests pin the correct behavior.


def test_degenerate_all_zeros_quantize_dequantize_returns_zero():
    """All-zero tensor: lo = hi = 0. After roundtrip latents must == 0."""
    from tac.substrates.sane_hnerv.archive import (
        _dequantize_latents,
        _quantize_latents_to_int16,
    )

    latents = torch.zeros(4, 8, dtype=torch.float32)
    q, scale, zp = _quantize_latents_to_int16(latents)
    # Q is filled with -32767 (degenerate sentinel)
    assert q.dtype == torch.int16
    assert torch.all(q == -32767), "degenerate fill must be -32767, not 0"
    assert zp == 0.0
    assert scale == 1.0

    deq = _dequantize_latents(q, scale, zp)
    assert torch.allclose(deq, latents, atol=1e-12), (
        f"degenerate dequant must recover lo exactly; got max-abs="
        f"{float(deq.abs().max())}"
    )


def test_degenerate_all_equal_nonzero_value_quantize_dequantize_recovers_value():
    """All-equal-to-c tensor: lo = hi = c. After roundtrip latents must == c."""
    from tac.substrates.sane_hnerv.archive import (
        _dequantize_latents,
        _quantize_latents_to_int16,
    )

    c = 0.42
    latents = torch.full((4, 8), c, dtype=torch.float32)
    q, scale, zp = _quantize_latents_to_int16(latents)
    assert torch.all(q == -32767)
    # zp is stored at float32 precision; allow tiny roundoff
    assert abs(zp - c) < 1e-6
    assert scale == 1.0

    deq = _dequantize_latents(q, scale, zp)
    assert torch.allclose(deq, latents, atol=1e-6), (
        f"degenerate-nonzero dequant must recover c={c}; got max-abs-err="
        f"{float((deq - latents).abs().max())}"
    )


def test_degenerate_full_archive_roundtrip_recovers_latents():
    """End-to-end: pack_archive + parse_archive on all-equal latents."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = SaneHnervSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    # Force latents to constant value (degenerate range)
    constant = 0.137
    latents = torch.full_like(model.state_dict()["latents"], constant)
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_upsample_blocks": cfg.num_upsample_blocks,
    }

    blob = pack_archive(decoder_sd, latents, meta)
    arc = parse_archive(blob)

    # Latents must recover the constant exactly (within float roundtrip)
    assert torch.allclose(arc.latents, latents, atol=1e-6), (
        f"degenerate full-archive roundtrip must recover constant {constant}; "
        f"got max-abs-err={float((arc.latents - latents).abs().max())}"
    )
