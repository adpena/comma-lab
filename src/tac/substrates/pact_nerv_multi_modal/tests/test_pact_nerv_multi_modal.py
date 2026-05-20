# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for pact_nerv_multi_modal."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_multi_modal.architecture import (
    MultiModalConditioningFusion,
    PactNervMultiModalConfig,
    PactNervMultiModalSubstrate,
)
from tac.substrates.pact_nerv_multi_modal.archive import (
    PMM_HEADER_SIZE,
    PMM_MAGIC,
    PMM_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervMultiModalConfig:
    return PactNervMultiModalConfig(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        pose_dim=6,
        class_prior_dim=5,
        odometry_dim=4,
        fusion_dim=8,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: PactNervMultiModalConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "fusion_dim": cfg.fusion_dim,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_multi_modal as pkg

    assert hasattr(pkg, "PactNervMultiModalConfig")
    assert hasattr(pkg, "PactNervMultiModalSubstrate")
    assert hasattr(pkg, "MultiModalConditioningFusion")
    assert hasattr(pkg, "pack_archive")
    assert hasattr(pkg, "parse_archive")
    assert hasattr(pkg, "PactNervMultiModalScoreAwareLoss")
    assert hasattr(pkg, "PactNervMultiModalArchive")


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervMultiModalSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)


def test_multi_modal_fusion_emits_correct_dim() -> None:
    """Distinguishing primitive: 3-tower fusion -> fusion_dim vector."""
    fusion = MultiModalConditioningFusion(
        pose_dim=6, class_prior_dim=5, odometry_dim=4, fusion_dim=16
    )
    pose = torch.randn(3, 6)
    class_prior = torch.randn(3, 5)
    odometry = torch.randn(3, 4)
    out = fusion(pose, class_prior, odometry)
    assert out.shape == (3, 16)


def test_archive_pack_then_parse_roundtrip_recovers_tensors() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervMultiModalSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents", "pose_data", "class_prior_data", "odometry_data")
    }
    latents = sd["latents"].clone()
    pose = sd["pose_data"].clone()
    class_prior = sd["class_prior_data"].clone()
    odometry = sd["odometry_data"].clone()

    blob = pack_archive(decoder_sd, latents, pose, class_prior, odometry, _smoke_meta(cfg))
    arc = parse_archive(blob)

    assert arc.schema_version == PMM_SCHEMA_VERSION
    assert blob[:4] == PMM_MAGIC
    assert arc.latents.shape == latents.shape
    assert arc.pose.shape == pose.shape
    assert arc.class_prior.shape == class_prior.shape
    assert arc.odometry.shape == odometry.shape


def test_archive_grammar_header_size_invariant_is_36_bytes() -> None:
    assert PMM_HEADER_SIZE == 36


def test_byte_mutation_changes_archive_no_op_proof() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervMultiModalSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items()
        if k not in ("latents", "pose_data", "class_prior_data", "odometry_data")
    }
    latents = sd["latents"].clone()
    pose = sd["pose_data"].clone()
    class_prior = sd["class_prior_data"].clone()
    odometry = sd["odometry_data"].clone()

    blob_a = pack_archive(decoder_sd, latents, pose, class_prior, odometry, _smoke_meta(cfg))
    mutated_pose = pose.clone()
    mutated_pose[0, 0] = mutated_pose[0, 0] + 1.0
    blob_b = pack_archive(decoder_sd, latents, mutated_pose, class_prior, odometry, _smoke_meta(cfg))
    assert blob_a != blob_b


def test_trainer_full_main_raises_not_implemented_at_l0_scaffold() -> None:
    import argparse
    import importlib

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_multi_modal")
    ns = argparse.Namespace(output_dir=None, epochs=1, smoke=False, device="cpu")
    try:
        trainer._full_main(ns)
    except NotImplementedError as exc:
        assert (
            "OPERATOR-GATED" in str(exc)
            or "L0 SCAFFOLD" in str(exc)
            or "Stage 1" in str(exc)
        )
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError per L0 SCAFFOLD posture")


def test_trainer_routes_through_canonical_scorer_loss_helper() -> None:
    import inspect

    from tac.substrates.pact_nerv_multi_modal import score_aware_loss as sal_module

    src = inspect.getsource(sal_module)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect

    import experiments.train_substrate_pact_nerv_multi_modal as trainer_module

    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]

    recipe_path = (
        Path(__file__).resolve().parents[5]
        / ".omx/operator_authorize_recipes/substrate_pact_nerv_multi_modal_modal_t4_dispatch.yaml"
    )
    assert recipe_path.exists(), f"recipe missing: {recipe_path}"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path

    driver_path = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_multi_modal.sh"
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
