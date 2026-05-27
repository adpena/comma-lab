# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for nirvana.

Proves the encode/decode contract of the NRV1 monolithic 0.bin grammar +
patch assembly forward-pass parity under fp16 + int16-quant roundtrip.
Plus a smoke-level test that the trainer's _full_main raises
NotImplementedError per the L0 SCAFFOLD posture (Catalog #240).
"""

from __future__ import annotations

import torch

from tac.substrates.nirvana.architecture import (
    NirvanaConfig,
    NirvanaSubstrate,
)
from tac.substrates.nirvana.archive import (
    NRV1_HEADER_SIZE,
    NRV1_MAGIC,
    NRV1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> NirvanaConfig:
    return NirvanaConfig(
        latent_dim=8,
        patch_embed_dim=4,
        patch_grid_h=2,
        patch_grid_w=2,
        embed_dim=24,
        initial_patch_grid_h=3,
        initial_patch_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: NirvanaConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_patch_grid_h": cfg.initial_patch_grid_h,
        "initial_patch_grid_w": cfg.initial_patch_grid_w,
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
    model = NirvanaSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()

    blob = pack_archive(
        decoder_sd, latents, _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )
    arc = parse_archive(blob)

    assert arc.schema_version == NRV1_SCHEMA_VERSION
    assert blob[:4] == NRV1_MAGIC
    assert arc.patch_grid_h == cfg.patch_grid_h
    assert arc.patch_grid_w == cfg.patch_grid_w
    assert arc.patch_embed_dim == cfg.patch_embed_dim
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())
    for k, v in decoder_sd.items():
        rec = arc.decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    assert arc.latents.shape == latents.shape


def test_header_size_invariant_is_25_bytes():
    assert NRV1_HEADER_SIZE == 25


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
    model = NirvanaSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    blob = bytearray(
        pack_archive(
            decoder_sd, latents, _smoke_meta(cfg),
            patch_grid_h=cfg.patch_grid_h,
            patch_grid_w=cfg.patch_grid_w,
            patch_embed_dim=cfg.patch_embed_dim,
        )
    )
    blob[:4] = b"XXXX"
    try:
        parse_archive(bytes(blob))
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on bad magic")


def test_pack_archive_rejects_oversize_patch_grid():
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = NirvanaSubstrate(cfg)
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()
    try:
        pack_archive(
            decoder_sd, latents, _smoke_meta(cfg),
            patch_grid_h=256, patch_grid_w=cfg.patch_grid_w,
            patch_embed_dim=cfg.patch_embed_dim,
        )
    except ValueError as exc:
        assert "patch_grid_h" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on out-of-u8 patch_grid_h")


def test_forward_pass_after_roundtrip_matches_original_within_tolerance():
    cfg = _smoke_cfg()
    torch.manual_seed(7)
    model = NirvanaSubstrate(cfg).eval()

    idx = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        rgb_0_a, rgb_1_a = model(idx)

    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()
    blob = pack_archive(
        decoder_sd, latents, _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )
    arc = parse_archive(blob)

    rebuilt = NirvanaSubstrate(cfg).eval()
    rebuilt.load_state_dict(arc.decoder_state_dict, strict=False)
    with torch.no_grad():
        rebuilt.latents.copy_(arc.latents.to(rebuilt.latents.dtype))
        rgb_0_b, rgb_1_b = rebuilt(idx)

    assert torch.allclose(rgb_0_a, rgb_0_b, atol=5e-2)
    assert torch.allclose(rgb_1_a, rgb_1_b, atol=5e-2)


# ENCODE_INFLATE_ROUNDTRIP — Catalog #139 byte-mutation smoke
def test_byte_mutation_changes_inflate_output_no_op_proof():
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = NirvanaSubstrate(cfg).eval()
    decoder_sd = {k: v for k, v in model.state_dict().items() if k != "latents"}
    latents = model.state_dict()["latents"].clone()

    blob_a = pack_archive(
        decoder_sd, latents, _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )
    mutated = latents.clone()
    mutated[0, 0] = mutated[0, 0] + 1.0
    blob_b = pack_archive(
        decoder_sd, mutated, _smoke_meta(cfg),
        patch_grid_h=cfg.patch_grid_h,
        patch_grid_w=cfg.patch_grid_w,
        patch_embed_dim=cfg.patch_embed_dim,
    )

    assert blob_a != blob_b, "no_op_proof: mutating latents must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)


def test_forward_pass_produces_unit_interval_rgb():
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = NirvanaSubstrate(cfg).eval()
    idx = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (1, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_patch_grid_carries_correct_number_of_embeddings():
    """Distinctive design check: nirvana must have patch_grid_h * patch_grid_w embeddings."""
    cfg = _smoke_cfg()
    model = NirvanaSubstrate(cfg)
    expected = cfg.patch_grid_h * cfg.patch_grid_w
    assert model.patch_embeddings.shape[0] == expected
    assert model.patch_embeddings.shape[1] == cfg.patch_embed_dim


def test_patch_grid_dimensions_divide_output_evenly():
    """L0 SCAFFOLD invariant: output H/W must be divisible by patch_grid_h/w."""
    cfg_bad = NirvanaConfig(
        latent_dim=8, patch_embed_dim=4,
        patch_grid_h=3,  # 24 is not divisible by 3 in some configs; force a mismatch
        patch_grid_w=2,
        embed_dim=24, initial_patch_grid_h=3, initial_patch_grid_w=4,
        decoder_channels=(20, 16, 12), sin_frequency=30.0,
        num_upsample_blocks=3, num_pairs=3,
        output_height=25,  # NOT divisible by patch_grid_h=3
        output_width=32,
    )
    try:
        NirvanaSubstrate(cfg_bad)
    except ValueError as exc:
        assert "not divisible" in str(exc) or "output_height" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on patch-grid-output mismatch")


def test_full_main_implemented_and_cuda_gated(tmp_path):
    """CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: _full_main IMPLEMENTED + CUDA-gated.

    The L0 SCAFFOLD NotImplementedError is extinguished: ``_full_main`` routes
    the canonical score-aware training loop through
    ``run_pact_nerv_score_aware_training``. Per CLAUDE.md "MPS auth eval is
    NOISE" + Catalog #1, the full (non-smoke) path is CUDA-required; invoking
    it with ``--device cpu`` refuses via ``device_or_die`` (SystemExit). PAID
    DISPATCH stays gated by ``dispatch_enabled: false`` + ``research_only:
    true`` on the recipe per Catalog #325 (code complete, trigger gated).
    """
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_nirvana")
    src = inspect.getsource(trainer._full_main)
    assert "raise NotImplementedError" not in src, (
        "_full_main NotImplementedError must be extinguished per "
        "CLASS-SHIFT-FULL-MAIN-CLUSTER"
    )
    assert "run_pact_nerv_score_aware_training" in src, (
        "_full_main must route through the canonical shared training loop"
    )
    args = trainer._build_parser().parse_args(
        ["--output-dir", str(tmp_path / "out"), "--device", "cpu", "--epochs", "1"]
    )
    with pytest.raises(SystemExit):
        trainer._full_main(args)
