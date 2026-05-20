# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + #139 no_op_proof + L0 SCAFFOLD contract for CROSS-CODEC-A."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_cross_codec_a.architecture import (
    Fec6BaseCodecPlaceholder,
    PactNervCrossCodecAConfig,
    PactNervCrossCodecASubstrate,
)
from tac.substrates.pact_nerv_cross_codec_a.archive import (
    CC_A_HEADER_SIZE,
    CC_A_MAGIC,
    CC_A_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervCrossCodecAConfig:
    return PactNervCrossCodecAConfig(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_pairs=3,
        output_height=24,
        output_width=32,
        fec6_palette_size=16,
        composition_alpha=0.1,
    )


def _smoke_meta(cfg: PactNervCrossCodecAConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "composition_alpha": cfg.composition_alpha,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_cross_codec_a as m
    for name in (
        "PactNervCrossCodecAConfig",
        "PactNervCrossCodecASubstrate",
        "Fec6BaseCodecPlaceholder",
        "pack_archive",
        "parse_archive",
        "PactNervCrossCodecAScoreAwareLoss",
        "PactNervCrossCodecAArchive",
    ):
        assert hasattr(m, name), f"missing canonical symbol: {name}"


def test_fec6_base_codec_placeholder_renders_per_pair_color() -> None:
    base = Fec6BaseCodecPlaceholder(palette_size=16, num_pairs=3, output_h=24, output_w=32)
    sel = torch.tensor([0, 5, 15], dtype=torch.long)
    rgb_0, rgb_1 = base.render(sel)
    assert rgb_0.shape == (3, 3, 24, 32)
    assert rgb_1.shape == (3, 3, 24, 32)
    # Different selectors should produce different colors
    assert not torch.allclose(rgb_0[0], rgb_0[1])
    assert not torch.allclose(rgb_0[1], rgb_0[2])
    # Bounds
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_fec6_base_codec_rejects_out_of_palette_selector() -> None:
    base = Fec6BaseCodecPlaceholder(palette_size=16, num_pairs=3, output_h=24, output_w=32)
    try:
        base.render(torch.tensor([16], dtype=torch.long))
    except ValueError as exc:
        assert "out of palette" in str(exc)
    else:
        raise AssertionError("expected ValueError for selector >= palette")


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervCrossCodecASubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_cross_codec_composition_changes_with_alpha() -> None:
    """Composition alpha modulates the side-info residual contribution.

    Verifies the cross-codec contract: alpha=0 reduces to pure base codec render;
    alpha>0 mixes in Pact-NeRV side-info residual. Distinguishing alpha=0 vs
    alpha=0.5 with FORCED residual heads is the canonical disambiguation per
    Catalog #220 operational mechanism declaration.
    """
    cfg_zero = PactNervCrossCodecAConfig(
        latent_dim=8, embed_dim=24,
        initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(20, 16, 12), sin_frequency=30.0,
        num_upsample_blocks=3, num_pairs=3,
        output_height=24, output_width=32, fec6_palette_size=16,
        composition_alpha=0.0,  # zero alpha = pure base render
    )
    cfg_high = PactNervCrossCodecAConfig(
        latent_dim=8, embed_dim=24,
        initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(20, 16, 12), sin_frequency=30.0,
        num_upsample_blocks=3, num_pairs=3,
        output_height=24, output_width=32, fec6_palette_size=16,
        composition_alpha=0.5,
    )
    torch.manual_seed(42)
    m_zero = PactNervCrossCodecASubstrate(cfg_zero).eval()
    torch.manual_seed(42)
    m_high = PactNervCrossCodecASubstrate(cfg_high).eval()
    # Force non-trivial residual head magnitude (small SIREN init produces near-zero)
    with torch.no_grad():
        for m in (m_zero, m_high):
            m.head_res_0.weight.fill_(0.5)
            m.head_res_0.bias.fill_(0.3)
            m.head_res_1.weight.fill_(0.5)
            m.head_res_1.bias.fill_(0.3)
            m.latents.normal_(std=1.0)
        m_high.latents.copy_(m_zero.latents)
    idx = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        rgb_0_zero, _ = m_zero(idx)
        rgb_0_high, _ = m_high(idx)
    # Different alphas should produce different compositions (canonical contract)
    assert not torch.allclose(rgb_0_zero, rgb_0_high, atol=1e-3)


def test_archive_pack_then_parse_roundtrip() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervCrossCodecASubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents", "selectors") and not k.startswith("base_codec.")
    }
    latents = sd["latents"].clone()
    fec6_base = b"\x00\x01\x02\x03"  # placeholder fec6 bytes
    selector_bytes = b"\x00\x01\x02"
    blob = pack_archive(
        fec6_base, decoder_sd, latents, selector_bytes,
        _smoke_meta(cfg), palette_size=16,
    )
    arc = parse_archive(blob)
    assert arc.schema_version == CC_A_SCHEMA_VERSION
    assert blob[:4] == CC_A_MAGIC
    assert arc.palette_size == 16
    assert arc.selector_bytes == selector_bytes
    assert arc.fec6_base_bytes == fec6_base


def test_archive_header_size_invariant_is_30_bytes() -> None:
    assert CC_A_HEADER_SIZE == 30


def test_byte_mutation_changes_archive_no_op_proof_fec6_base() -> None:
    """Catalog #139 no-op-detector: mutating fec6_base_bytes changes archive."""
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervCrossCodecASubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents", "selectors") and not k.startswith("base_codec.")
    }
    latents = sd["latents"].clone()
    blob_a = pack_archive(b"\x00\x01", decoder_sd, latents, b"\x00\x01\x02",
                          _smoke_meta(cfg), palette_size=16)
    blob_b = pack_archive(b"\xff\x01", decoder_sd, latents, b"\x00\x01\x02",
                          _smoke_meta(cfg), palette_size=16)
    assert blob_a != blob_b


def test_byte_mutation_changes_archive_no_op_proof_selector() -> None:
    """Catalog #139 no-op-detector: mutating selector_bytes changes archive."""
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervCrossCodecASubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents", "selectors") and not k.startswith("base_codec.")
    }
    latents = sd["latents"].clone()
    blob_a = pack_archive(b"\x00", decoder_sd, latents, b"\x00\x01",
                          _smoke_meta(cfg), palette_size=16)
    blob_b = pack_archive(b"\x00", decoder_sd, latents, b"\xff\x01",
                          _smoke_meta(cfg), palette_size=16)
    assert blob_a != blob_b


def test_trainer_full_main_raises_not_implemented_at_l0_scaffold() -> None:
    import argparse
    import importlib
    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_cross_codec_a")
    ns = argparse.Namespace(output_dir=None, epochs=1, smoke=False, device="cpu")
    try:
        trainer._full_main(ns)
    except NotImplementedError as exc:
        assert "OPERATOR-GATED" in str(exc) or "L0 SCAFFOLD" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


def test_trainer_routes_through_canonical_scorer_loss_helper() -> None:
    import inspect
    from tac.substrates.pact_nerv_cross_codec_a import score_aware_loss as sal
    src = inspect.getsource(sal)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect
    import experiments.train_substrate_pact_nerv_cross_codec_a as trainer_module
    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path
    import yaml  # type: ignore[import-untyped]
    recipe = yaml.safe_load(
        (Path(__file__).resolve().parents[5]
         / ".omx/operator_authorize_recipes/substrate_pact_nerv_cross_codec_a_modal_t4_dispatch.yaml"
        ).read_text(encoding="utf-8")
    )
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path
    txt = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_cross_codec_a.sh"
    ).read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in txt
    assert "CUBLAS_WORKSPACE_CONFIG" in txt
    assert "PYTORCH_CUDA_ALLOC_CONF" in txt


def test_inflate_py_loc_under_200() -> None:
    from pathlib import Path
    loc = len(
        (Path(__file__).resolve().parents[1] / "inflate.py")
        .read_text(encoding="utf-8").splitlines()
    )
    assert loc <= 200
