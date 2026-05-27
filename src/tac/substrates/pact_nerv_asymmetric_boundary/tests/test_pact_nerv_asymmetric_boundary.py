# SPDX-License-Identifier: MIT
"""Catalog #91 + #139 + L0 SCAFFOLD contract for AsymmetricBoundary (per-class FiLM)."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_asymmetric_boundary.architecture import (
    AsymmetricBoundaryFilm,
    PactNervAsymmetricBoundaryConfig,
    PactNervAsymmetricBoundarySubstrate,
)
from tac.substrates.pact_nerv_asymmetric_boundary.archive import (
    PAB1_HEADER_SIZE,
    PAB1_MAGIC,
    PAB1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervAsymmetricBoundaryConfig:
    return PactNervAsymmetricBoundaryConfig(
        latent_dim=8, embed_dim=24, initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(20, 16, 12), sin_frequency=30.0,
        num_upsample_blocks=3, num_segnet_classes=5,
        boundary_signal_dim=5, film_init_delta_std=0.01,
        num_pairs=3, output_height=24, output_width=32,
    )


def _smoke_meta(cfg: PactNervAsymmetricBoundaryConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim, "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "num_segnet_classes": cfg.num_segnet_classes,
        "film_init_delta_std": cfg.film_init_delta_std,
        "output_height": cfg.output_height, "output_width": cfg.output_width,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_asymmetric_boundary as m
    for name in (
        "PactNervAsymmetricBoundaryConfig", "PactNervAsymmetricBoundarySubstrate",
        "AsymmetricBoundaryFilm", "pack_archive", "parse_archive",
        "PactNervAsymmetricBoundaryScoreAwareLoss",
        "PactNervAsymmetricBoundaryArchive",
    ):
        assert hasattr(m, name)


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervAsymmetricBoundarySubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_boundary_film_has_both_gamma_and_beta_projections() -> None:
    """Asymmetric boundary FiLM uses BOTH γ + β (unlike sister IA3 γ-only)."""
    torch.manual_seed(0)
    film = AsymmetricBoundaryFilm(
        num_features=8, boundary_signal_dim=5, init_delta_std=0.0
    )
    assert hasattr(film, "gamma_proj"), "FiLM MUST have γ projection"
    assert hasattr(film, "beta_proj"), (
        "Asymmetric boundary FiLM MUST have β projection per FiLM γ+β contract"
    )


def test_boundary_film_init_identity_when_delta_zero() -> None:
    """γ_init=1.0+Δ=0, β_init=0+Δ=0 produces identity at delta_std=0."""
    torch.manual_seed(0)
    film = AsymmetricBoundaryFilm(
        num_features=8, boundary_signal_dim=5, init_delta_std=0.0
    )
    x = torch.randn(2, 8, 4, 4)
    b = torch.randn(2, 5)
    with torch.no_grad():
        y = film(x, b)
    assert torch.allclose(y, x, atol=1e-6), (
        "FiLM γ_init=1, β_init=0 must produce identity"
    )


def test_boundary_signal_dim_matches_segnet_classes() -> None:
    """5-class boundary signal matches upstream SegNet 5-class output (HARD-EARNED)."""
    cfg = _smoke_cfg()
    assert cfg.boundary_signal_dim == cfg.num_segnet_classes == 5


def test_substrate_has_boundary_film_module() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervAsymmetricBoundarySubstrate(cfg)
    assert hasattr(model, "boundary_film")
    assert isinstance(model.boundary_film, AsymmetricBoundaryFilm)


def test_archive_pack_then_parse_roundtrip() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervAsymmetricBoundarySubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items() if k not in ("latents", "boundary_signals")
    }
    latents = sd["latents"].clone()
    boundary_signals = sd["boundary_signals"].clone()
    blob = pack_archive(
        decoder_sd, latents, boundary_signals, _smoke_meta(cfg),
        boundary_signal_dim=cfg.boundary_signal_dim,
    )
    arc = parse_archive(blob)
    assert arc.schema_version == PAB1_SCHEMA_VERSION
    assert blob[:4] == PAB1_MAGIC
    assert arc.boundary_signal_dim == cfg.boundary_signal_dim


def test_archive_header_size_invariant_is_26_bytes() -> None:
    assert PAB1_HEADER_SIZE == 26


def test_byte_mutation_changes_archive_no_op_proof() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervAsymmetricBoundarySubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items() if k not in ("latents", "boundary_signals")
    }
    latents = sd["latents"].clone()
    boundary_signals = sd["boundary_signals"].clone()
    blob_a = pack_archive(
        decoder_sd, latents, boundary_signals, _smoke_meta(cfg),
        boundary_signal_dim=cfg.boundary_signal_dim,
    )
    mutated_b = boundary_signals.clone()
    mutated_b[0, 0] += 1.0
    blob_b = pack_archive(
        decoder_sd, latents, mutated_b, _smoke_meta(cfg),
        boundary_signal_dim=cfg.boundary_signal_dim,
    )
    assert blob_a != blob_b


def test_trainer_full_main_implemented_and_cuda_gated(tmp_path) -> None:
    """PACT-NERV-FULL-MAIN-CLUSTER-2 2026-05-27: _full_main IMPLEMENTED + CUDA-gated."""
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_asymmetric_boundary")
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

    from tac.substrates.pact_nerv_asymmetric_boundary import score_aware_loss as sal
    src = inspect.getsource(sal)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect

    import experiments.train_substrate_pact_nerv_asymmetric_boundary as trainer_module
    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]
    recipe = yaml.safe_load(
        (Path(__file__).resolve().parents[5]
         / ".omx/operator_authorize_recipes/substrate_pact_nerv_asymmetric_boundary_modal_t4_dispatch.yaml"
        ).read_text(encoding="utf-8")
    )
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path
    txt = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_asymmetric_boundary.sh"
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
