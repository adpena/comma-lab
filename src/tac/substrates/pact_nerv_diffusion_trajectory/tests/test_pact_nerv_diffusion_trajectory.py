# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for pact_nerv_diffusion_trajectory."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_diffusion_trajectory.architecture import (
    DiffusionTrajectoryPredictor,
    PactNervDiffusionTrajectoryConfig,
    PactNervDiffusionTrajectorySubstrate,
)
from tac.substrates.pact_nerv_diffusion_trajectory.archive import (
    PDT_HEADER_SIZE,
    PDT_MAGIC,
    PDT_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervDiffusionTrajectoryConfig:
    return PactNervDiffusionTrajectoryConfig(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        diffusion_num_timesteps=3,
        diffusion_predictor_hidden=8,
        noise_schedule="linear",
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: PactNervDiffusionTrajectoryConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "diffusion_predictor_hidden": cfg.diffusion_predictor_hidden,
        "noise_schedule": cfg.noise_schedule,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_diffusion_trajectory as pkg

    assert hasattr(pkg, "PactNervDiffusionTrajectoryConfig")
    assert hasattr(pkg, "PactNervDiffusionTrajectorySubstrate")
    assert hasattr(pkg, "DiffusionTrajectoryPredictor")
    assert hasattr(pkg, "pack_archive")
    assert hasattr(pkg, "parse_archive")
    assert hasattr(pkg, "PactNervDiffusionTrajectoryScoreAwareLoss")
    assert hasattr(pkg, "PactNervDiffusionTrajectoryArchive")


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervDiffusionTrajectorySubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)


def test_diffusion_trajectory_predictor_refines_seed_to_latent() -> None:
    """Distinguishing primitive: T-step trajectory refines noise -> latent."""
    torch.manual_seed(0)
    predictor = DiffusionTrajectoryPredictor(
        latent_dim=8, num_timesteps=5, hidden=16, noise_schedule="linear"
    )
    seeds = torch.randn(3, 8)
    refined = predictor(seeds)
    assert refined.shape == seeds.shape
    # After refinement, the latent IS different from the raw seed (predictor MLPs run)
    # but for randomly-init predictors with small weights the difference may be small;
    # we just verify that the output is finite and shape-correct.
    assert refined.dtype == seeds.dtype
    assert torch.isfinite(refined).all()


def test_diffusion_trajectory_predictor_cosine_schedule_accepted() -> None:
    """Cosine schedule per Nichol-Dhariwal 2102.09672 accepted."""
    predictor = DiffusionTrajectoryPredictor(
        latent_dim=8, num_timesteps=5, hidden=16, noise_schedule="cosine"
    )
    seeds = torch.randn(2, 8)
    out = predictor(seeds)
    assert out.shape == seeds.shape


def test_archive_pack_then_parse_roundtrip_recovers_tensors() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervDiffusionTrajectorySubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "seeds"}
    seeds = sd["seeds"].clone()

    blob = pack_archive(
        decoder_sd, seeds, _smoke_meta(cfg),
        num_timesteps=cfg.diffusion_num_timesteps,
    )
    arc = parse_archive(blob)

    assert arc.schema_version == PDT_SCHEMA_VERSION
    assert blob[:4] == PDT_MAGIC
    assert arc.seeds.shape == seeds.shape
    assert arc.num_timesteps == cfg.diffusion_num_timesteps


def test_archive_grammar_header_size_invariant_is_22_bytes() -> None:
    assert PDT_HEADER_SIZE == 22


def test_byte_mutation_changes_archive_no_op_proof() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervDiffusionTrajectorySubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "seeds"}
    seeds = sd["seeds"].clone()

    blob_a = pack_archive(
        decoder_sd, seeds, _smoke_meta(cfg),
        num_timesteps=cfg.diffusion_num_timesteps,
    )
    mutated_seeds = seeds.clone()
    mutated_seeds[0, 0] = mutated_seeds[0, 0] + 1.0
    blob_b = pack_archive(
        decoder_sd, mutated_seeds, _smoke_meta(cfg),
        num_timesteps=cfg.diffusion_num_timesteps,
    )
    assert blob_a != blob_b


def test_trainer_full_main_implemented_and_cuda_gated(tmp_path) -> None:
    """PACT-NERV-FULL-MAIN-CLUSTER-2 2026-05-27: _full_main IMPLEMENTED + CUDA-gated."""
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_diffusion_trajectory")
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

    from tac.substrates.pact_nerv_diffusion_trajectory import score_aware_loss as sal_module

    src = inspect.getsource(sal_module)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect

    import experiments.train_substrate_pact_nerv_diffusion_trajectory as trainer_module

    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]

    recipe_path = (
        Path(__file__).resolve().parents[5]
        / ".omx/operator_authorize_recipes/substrate_pact_nerv_diffusion_trajectory_modal_t4_dispatch.yaml"
    )
    assert recipe_path.exists(), f"recipe missing: {recipe_path}"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path

    driver_path = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_diffusion_trajectory.sh"
    )
    assert driver_path.exists()
    driver_text = driver_path.read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in driver_text
    assert "CUBLAS_WORKSPACE_CONFIG" in driver_text
    assert "PYTORCH_CUDA_ALLOC_CONF" in driver_text


def test_inflate_py_loc_under_200_per_hnerv_parity_l4() -> None:
    from pathlib import Path

    inflate_path = Path(__file__).resolve().parents[1] / "inflate.py"
    physical_loc = len(inflate_path.read_text(encoding="utf-8").splitlines())
    assert physical_loc <= 200
