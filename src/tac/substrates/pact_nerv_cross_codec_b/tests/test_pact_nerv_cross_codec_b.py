# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + #139 no_op_proof + L0 SCAFFOLD contract for CROSS-CODEC-B."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_cross_codec_b.architecture import (
    PactNervCrossCodecBConfig,
    PactNervCrossCodecBSubstrate,
    Pr106BaseCodecPlaceholder,
)
from tac.substrates.pact_nerv_cross_codec_b.archive import (
    CC_B_HEADER_SIZE,
    CC_B_MAGIC,
    CC_B_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervCrossCodecBConfig:
    return PactNervCrossCodecBConfig(
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
        pr106_score_table_size=64,
        pose_dim=6,
        ia3_init_delta_std=0.01,
        composition_alpha=0.1,
    )


def _smoke_meta(cfg: PactNervCrossCodecBConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "ia3_init_delta_std": cfg.ia3_init_delta_std,
        "composition_alpha": cfg.composition_alpha,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_cross_codec_b as m
    for name in (
        "PactNervCrossCodecBConfig",
        "PactNervCrossCodecBSubstrate",
        "Pr106BaseCodecPlaceholder",
        "pack_archive",
        "parse_archive",
        "PactNervCrossCodecBScoreAwareLoss",
        "PactNervCrossCodecBArchive",
    ):
        assert hasattr(m, name), f"missing canonical symbol: {name}"


def test_pr106_base_codec_placeholder_renders_per_pair_color() -> None:
    base = Pr106BaseCodecPlaceholder(
        score_table_size=64, num_pairs=3, output_h=24, output_w=32,
    )
    sel = torch.tensor([0, 30, 63], dtype=torch.long)
    rgb_0, rgb_1 = base.render(sel)
    assert rgb_0.shape == (3, 3, 24, 32)
    assert rgb_1.shape == (3, 3, 24, 32)
    # Different indices should produce different colors
    assert not torch.allclose(rgb_0[0], rgb_0[1])
    assert not torch.allclose(rgb_0[1], rgb_0[2])
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_pr106_base_codec_rejects_out_of_table_index() -> None:
    base = Pr106BaseCodecPlaceholder(
        score_table_size=64, num_pairs=3, output_h=24, output_w=32,
    )
    try:
        base.render(torch.tensor([64], dtype=torch.long))
    except ValueError as exc:
        assert "out of table" in str(exc)
    else:
        raise AssertionError("expected ValueError for index >= score_table_size")


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervCrossCodecBSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_ia3_gamma_residual_init_close_to_unity() -> None:
    """IA3 spec: γ_init = 1.0 + Δ where Δ ~ N(0, ia3_init_delta_std^2)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervCrossCodecBSubstrate(cfg)
    # Each γ_proj should start near zero weights + zero bias => γ ≈ 1.0
    for block in model.blocks:
        # gamma_proj.bias initialized to zero
        assert torch.allclose(
            block.gamma_proj.bias, torch.zeros_like(block.gamma_proj.bias),
            atol=1e-6,
        )
        # gamma_proj.weight stddev within 3-sigma of ia3_init_delta_std
        observed_std = block.gamma_proj.weight.std().item()
        assert observed_std < cfg.ia3_init_delta_std * 5, (
            f"IA3 init stddev too large: {observed_std} vs cap "
            f"{cfg.ia3_init_delta_std * 5}"
        )


def test_cross_codec_composition_changes_with_alpha() -> None:
    """Composition alpha modulates the side-info residual contribution."""
    cfg_zero = PactNervCrossCodecBConfig(
        latent_dim=8, embed_dim=24,
        initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(20, 16, 12), sin_frequency=30.0,
        num_upsample_blocks=3, num_pairs=3,
        output_height=24, output_width=32, pr106_score_table_size=64,
        pose_dim=6, ia3_init_delta_std=0.01, composition_alpha=0.0,
    )
    cfg_high = PactNervCrossCodecBConfig(
        latent_dim=8, embed_dim=24,
        initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(20, 16, 12), sin_frequency=30.0,
        num_upsample_blocks=3, num_pairs=3,
        output_height=24, output_width=32, pr106_score_table_size=64,
        pose_dim=6, ia3_init_delta_std=0.01, composition_alpha=0.5,
    )
    torch.manual_seed(42)
    m_zero = PactNervCrossCodecBSubstrate(cfg_zero).eval()
    torch.manual_seed(42)
    m_high = PactNervCrossCodecBSubstrate(cfg_high).eval()
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
    assert not torch.allclose(rgb_0_zero, rgb_0_high, atol=1e-3)


def test_archive_pack_then_parse_roundtrip() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervCrossCodecBSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("ego_poses", "score_indices") and not k.startswith("base_codec.")
    }
    latents = sd["latents"].clone()
    ego_poses = torch.zeros(cfg.num_pairs, cfg.pose_dim, dtype=torch.float32)
    pr106_base = b"\x00\x01\x02\x03"
    score_index_bytes = b"\x00\x10\x20"
    blob = pack_archive(
        pr106_base, decoder_sd, latents, ego_poses, score_index_bytes,
        _smoke_meta(cfg), score_table_size=64, pose_dim=6,
    )
    arc = parse_archive(blob)
    assert arc.schema_version == CC_B_SCHEMA_VERSION
    assert blob[:4] == CC_B_MAGIC
    assert arc.score_table_size == 64
    assert arc.pose_dim == 6
    assert arc.score_index_bytes == score_index_bytes
    assert arc.pr106_base_bytes == pr106_base


def test_archive_header_size_invariant_is_35_bytes() -> None:
    assert CC_B_HEADER_SIZE == 35


def test_byte_mutation_changes_archive_no_op_proof_pr106_base() -> None:
    """Catalog #139 no-op-detector: mutating pr106_base_bytes changes archive."""
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervCrossCodecBSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("ego_poses", "score_indices") and not k.startswith("base_codec.")
    }
    latents = sd["latents"].clone()
    ego_poses = torch.zeros(cfg.num_pairs, cfg.pose_dim, dtype=torch.float32)
    blob_a = pack_archive(
        b"\x00\x01", decoder_sd, latents, ego_poses, b"\x00\x01\x02",
        _smoke_meta(cfg), score_table_size=64, pose_dim=6,
    )
    blob_b = pack_archive(
        b"\xff\x01", decoder_sd, latents, ego_poses, b"\x00\x01\x02",
        _smoke_meta(cfg), score_table_size=64, pose_dim=6,
    )
    assert blob_a != blob_b


def test_byte_mutation_changes_archive_no_op_proof_score_index() -> None:
    """Catalog #139 no-op-detector: mutating score_index_bytes changes archive."""
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervCrossCodecBSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("ego_poses", "score_indices") and not k.startswith("base_codec.")
    }
    latents = sd["latents"].clone()
    ego_poses = torch.zeros(cfg.num_pairs, cfg.pose_dim, dtype=torch.float32)
    blob_a = pack_archive(
        b"\x00", decoder_sd, latents, ego_poses, b"\x00\x01\x02",
        _smoke_meta(cfg), score_table_size=64, pose_dim=6,
    )
    blob_b = pack_archive(
        b"\x00", decoder_sd, latents, ego_poses, b"\xff\x01\x02",
        _smoke_meta(cfg), score_table_size=64, pose_dim=6,
    )
    assert blob_a != blob_b


def test_trainer_full_main_implemented_and_cuda_gated(tmp_path) -> None:
    """PACT-NERV-FULL-MAIN-CLUSTER-2 2026-05-27: _full_main IMPLEMENTED + CUDA-gated."""
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_cross_codec_b")
    src = inspect.getsource(trainer._full_main)
    assert "raise NotImplementedError" not in src
    assert "run_pact_nerv_score_aware_training" in src
    args = trainer._build_parser().parse_args(
        ["--output-dir", str(tmp_path / "out"), "--device", "cpu"]
    )
    with pytest.raises(SystemExit):
        trainer._full_main(args)


def test_trainer_routes_through_canonical_scorer_loss_helper() -> None:
    import inspect

    from tac.substrates.pact_nerv_cross_codec_b import score_aware_loss as sal
    src = inspect.getsource(sal)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect

    import experiments.train_substrate_pact_nerv_cross_codec_b as trainer_module
    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]
    recipe = yaml.safe_load(
        (Path(__file__).resolve().parents[5]
         / ".omx/operator_authorize_recipes/substrate_pact_nerv_cross_codec_b_modal_t4_dispatch.yaml"
        ).read_text(encoding="utf-8")
    )
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path
    txt = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_cross_codec_b.sh"
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
