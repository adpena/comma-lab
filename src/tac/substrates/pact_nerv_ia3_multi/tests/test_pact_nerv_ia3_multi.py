# SPDX-License-Identifier: MIT
"""Catalog #91 + #139 + L0 SCAFFOLD contract for IA3-Multi (multi-layer IA3 + difficulty)."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_ia3_multi.architecture import (
    IA3MultiGammaOnlyModulation,
    PactNervIa3MultiConfig,
    PactNervIa3MultiSubstrate,
)
from tac.substrates.pact_nerv_ia3_multi.archive import (
    PIM1_HEADER_SIZE,
    PIM1_MAGIC,
    PIM1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervIa3MultiConfig:
    return PactNervIa3MultiConfig(
        latent_dim=8, embed_dim=24, initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(20, 16, 12), sin_frequency=30.0,
        num_upsample_blocks=3, pose_dim=6, difficulty_dim=1,
        ia3_init_delta_std=0.01, num_pairs=3,
        output_height=24, output_width=32,
    )


def _smoke_meta(cfg: PactNervIa3MultiConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim, "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "ia3_init_delta_std": cfg.ia3_init_delta_std,
        "output_height": cfg.output_height, "output_width": cfg.output_width,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_ia3_multi as m
    for name in (
        "PactNervIa3MultiConfig", "PactNervIa3MultiSubstrate",
        "IA3MultiGammaOnlyModulation", "pack_archive", "parse_archive",
        "PactNervIa3MultiScoreAwareLoss", "PactNervIa3MultiArchive",
    ):
        assert hasattr(m, name)


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervIa3MultiSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_ia3_multi_no_beta_projection_invariant() -> None:
    """Sister Pact-NeRV-IA3 invariant: NO β bias projection."""
    torch.manual_seed(0)
    ia3 = IA3MultiGammaOnlyModulation(
        num_features=8, pose_dim=6, difficulty_dim=1, init_delta_std=0.0
    )
    assert not hasattr(ia3, "beta_proj"), (
        "IA3-Multi MUST NOT have β bias projection per IA3 §3.2 invariant"
    )


def test_ia3_multi_gamma_init_identity_when_delta_zero() -> None:
    """γ_init = 1.0 + Δ=0 produces identity transformation."""
    torch.manual_seed(0)
    ia3 = IA3MultiGammaOnlyModulation(
        num_features=8, pose_dim=6, difficulty_dim=1, init_delta_std=0.0
    )
    x = torch.randn(2, 8, 4, 4)
    pose = torch.randn(2, 6)
    diff = torch.randn(2, 1)
    with torch.no_grad():
        y = ia3(x, pose, diff)
    assert torch.allclose(y, x, atol=1e-6), (
        "γ_init=1.0+Δ=0 must produce identity per IA3 §3.2 residual form"
    )


def test_ia3_multi_difficulty_required_when_dim_positive() -> None:
    ia3 = IA3MultiGammaOnlyModulation(
        num_features=8, pose_dim=6, difficulty_dim=1
    )
    x = torch.randn(1, 8, 4, 4)
    pose = torch.randn(1, 6)
    try:
        ia3(x, pose, None)
    except ValueError as exc:
        assert "difficulty must be provided" in str(exc)
    else:
        raise AssertionError("expected ValueError when difficulty missing")


def test_substrate_has_multi_block_ia3_modules() -> None:
    """Sister of pact_nerv_ia3: multi-block invariant — IA3 at EVERY upsample block."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervIa3MultiSubstrate(cfg)
    assert len(model.ia3_mods) == cfg.num_upsample_blocks, (
        f"multi-block IA3: {len(model.ia3_mods)} mods != "
        f"{cfg.num_upsample_blocks} upsample blocks"
    )


def test_archive_pack_then_parse_roundtrip() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervIa3MultiSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents", "ego_poses", "difficulties")
    }
    latents = sd["latents"].clone()
    ego_poses = sd["ego_poses"].clone()
    difficulties = sd["difficulties"].clone()
    blob = pack_archive(
        decoder_sd, latents, ego_poses, difficulties, _smoke_meta(cfg),
        pose_dim=cfg.pose_dim, difficulty_dim=cfg.difficulty_dim,
    )
    arc = parse_archive(blob)
    assert arc.schema_version == PIM1_SCHEMA_VERSION
    assert blob[:4] == PIM1_MAGIC
    assert arc.pose_dim == cfg.pose_dim
    assert arc.difficulty_dim == cfg.difficulty_dim


def test_archive_header_size_invariant_is_31_bytes() -> None:
    """PIM1 header carries pose_dim + difficulty_dim u8 distinctive fields."""
    assert PIM1_HEADER_SIZE == 31


def test_byte_mutation_changes_archive_no_op_proof() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervIa3MultiSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents", "ego_poses", "difficulties")
    }
    latents = sd["latents"].clone()
    ego_poses = sd["ego_poses"].clone()
    difficulties = sd["difficulties"].clone()
    blob_a = pack_archive(
        decoder_sd, latents, ego_poses, difficulties, _smoke_meta(cfg),
        pose_dim=cfg.pose_dim, difficulty_dim=cfg.difficulty_dim,
    )
    mutated_diff = difficulties.clone()
    mutated_diff[0, 0] += 1.0
    blob_b = pack_archive(
        decoder_sd, latents, ego_poses, mutated_diff, _smoke_meta(cfg),
        pose_dim=cfg.pose_dim, difficulty_dim=cfg.difficulty_dim,
    )
    assert blob_a != blob_b, (
        "no_op_proof: mutating per-pair difficulty must change archive bytes"
    )


def test_trainer_full_main_implemented_and_cuda_gated(tmp_path) -> None:
    """PACT-NERV-FULL-MAIN-CLUSTER-2 2026-05-27: _full_main IMPLEMENTED + CUDA-gated."""
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_ia3_multi")
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

    from tac.substrates.pact_nerv_ia3_multi import score_aware_loss as sal
    src = inspect.getsource(sal)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect

    import experiments.train_substrate_pact_nerv_ia3_multi as trainer_module
    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]
    recipe = yaml.safe_load(
        (Path(__file__).resolve().parents[5]
         / ".omx/operator_authorize_recipes/substrate_pact_nerv_ia3_multi_modal_t4_dispatch.yaml"
        ).read_text(encoding="utf-8")
    )
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path
    txt = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_ia3_multi.sh"
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
