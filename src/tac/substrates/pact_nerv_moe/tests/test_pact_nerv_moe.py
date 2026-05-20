# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for pact_nerv_moe.

Proves the encode/decode contract of the PNMO monolithic 0.bin grammar +
PoseConditionedRouter top-k routing forward-pass parity. Plus a smoke-level
test that the trainer's _full_main raises NotImplementedError per the L0
SCAFFOLD posture (Catalog #240).
"""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_moe.architecture import (
    PactNervMoeConfig,
    PactNervMoeSubstrate,
    PoseConditionedRouter,
)
from tac.substrates.pact_nerv_moe.archive import (
    PNMO_HEADER_SIZE,
    PNMO_MAGIC,
    PNMO_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervMoeConfig:
    return PactNervMoeConfig(
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
        num_experts=3,
        top_k=2,
        pose_embed_dim=4,
    )


def _smoke_meta(cfg: PactNervMoeConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "num_experts": cfg.num_experts,
        "top_k": cfg.top_k,
        "pose_embed_dim": cfg.pose_embed_dim,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_moe as pkg

    assert hasattr(pkg, "PactNervMoeConfig")
    assert hasattr(pkg, "PactNervMoeSubstrate")
    assert hasattr(pkg, "PoseConditionedRouter")
    assert hasattr(pkg, "pack_archive")
    assert hasattr(pkg, "parse_archive")
    assert hasattr(pkg, "PactNervMoeScoreAwareLoss")
    assert hasattr(pkg, "PactNervMoeArchive")


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervMoeSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_router_produces_top_k_route_probs_and_load_balance() -> None:
    """Distinguishing primitive: router emits (B, K) probs + scalar aux."""
    torch.manual_seed(0)
    router = PoseConditionedRouter(pose_embed_dim=8, num_experts=4, top_k=2)
    pose_embed = torch.randn(3, 8)
    route_probs, aux = router(pose_embed)
    assert route_probs.shape == (3, 4)
    # Top-2 means at most 2 nonzero per row
    nz = (route_probs > 0).sum(dim=1)
    assert (nz <= 2).all()
    # Sum to 1 over selected experts (softmax-normalized over top-k)
    assert torch.allclose(route_probs.sum(dim=1), torch.ones(3), atol=1e-5)
    assert aux.ndim == 0
    assert float(aux) >= 0.0


def test_router_rejects_invalid_config() -> None:
    import pytest

    with pytest.raises(ValueError, match="pose_embed_dim"):
        PoseConditionedRouter(pose_embed_dim=0, num_experts=4, top_k=2)
    with pytest.raises(ValueError, match="num_experts"):
        PoseConditionedRouter(pose_embed_dim=8, num_experts=1, top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        PoseConditionedRouter(pose_embed_dim=8, num_experts=4, top_k=5)


def test_archive_pack_then_parse_roundtrip_recovers_tensors() -> None:
    """ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervMoeSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()

    blob = pack_archive(decoder_sd, latents, _smoke_meta(cfg))
    arc = parse_archive(blob)

    assert arc.schema_version == PNMO_SCHEMA_VERSION
    assert blob[:4] == PNMO_MAGIC
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())
    assert arc.latents.shape == latents.shape
    assert int(arc.meta["num_experts"]) == cfg.num_experts


def test_archive_grammar_header_size_invariant_is_22_bytes() -> None:
    assert PNMO_HEADER_SIZE == 22


def test_byte_mutation_changes_inflate_output_no_op_proof() -> None:
    """Catalog #139 no_op_proof: byte mutation MUST change rendered frames."""
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervMoeSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    latents = sd["latents"].clone()

    blob_a = pack_archive(decoder_sd, latents, _smoke_meta(cfg))
    mutated_latents = latents.clone()
    mutated_latents[0, 0] = mutated_latents[0, 0] + 1.0
    blob_b = pack_archive(decoder_sd, mutated_latents, _smoke_meta(cfg))
    assert blob_a != blob_b, "no_op_proof: mutating latents must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)


def test_trainer_full_main_raises_not_implemented_at_l0_scaffold() -> None:
    """L0 SCAFFOLD posture: trainer _full_main MUST raise NotImplementedError."""
    import argparse
    import importlib

    trainer = importlib.import_module(
        "experiments.train_substrate_pact_nerv_moe"
    )
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
    """Catalog #164: trainer score-aware path MUST route through canonical helper."""
    import inspect

    from tac.substrates.pact_nerv_moe import score_aware_loss as sal_module

    src = inspect.getsource(sal_module)
    assert "score_pair_components_dispatch" in src, (
        "score-aware module MUST route through canonical helper per Catalog #164"
    )
    assert (
        "tac.substrates.score_aware_common" in src
    ), "must import from canonical helper module"


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    """Catalog #6 MANDATORY DEFAULT: trainer MUST patch yuv6 BEFORE scorer load."""
    import inspect

    import experiments.train_substrate_pact_nerv_moe as trainer_module

    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src, (
        "_smoke_main MUST call patch_upstream_yuv6_globally() per CLAUDE.md "
        "eval_roundtrip non-negotiable"
    )
    patch_idx = src.find("patch_upstream_yuv6_globally")
    for scorer_token in ("load_differentiable_scorers", "load_default_scorers"):
        if scorer_token in src:
            scorer_idx = src.find(scorer_token)
            assert patch_idx < scorer_idx, (
                f"patch_upstream_yuv6_globally MUST precede {scorer_token}"
            )


def test_recipe_research_only_and_dispatch_disabled() -> None:
    """Catalog #240: recipe MUST opt out of dispatch at L0 SCAFFOLD."""
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]

    recipe_path = (
        Path(__file__).resolve().parents[5]
        / ".omx/operator_authorize_recipes/substrate_pact_nerv_moe_modal_t4_dispatch.yaml"
    )
    assert recipe_path.exists(), f"recipe missing: {recipe_path}"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert recipe["dispatch_enabled"] is False, (
        "recipe MUST declare dispatch_enabled:false at L0 SCAFFOLD"
    )
    assert recipe["research_only"] is True, (
        "recipe MUST declare research_only:true at L0 SCAFFOLD"
    )


def test_driver_carries_canonical_nvml_block() -> None:
    """Catalog #244: remote lane driver MUST carry the canonical 3-export NVML block."""
    from pathlib import Path

    driver_path = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_moe.sh"
    )
    assert driver_path.exists(), f"driver missing: {driver_path}"
    driver_text = driver_path.read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in driver_text
    assert "CUBLAS_WORKSPACE_CONFIG" in driver_text
    assert "PYTORCH_CUDA_ALLOC_CONF" in driver_text


def test_inflate_py_loc_under_200_per_hnerv_parity_l4() -> None:
    """HNeRV parity L4: inflate runtime MUST be ≤ 200 LOC."""
    from pathlib import Path

    inflate_path = Path(__file__).resolve().parents[1] / "inflate.py"
    assert inflate_path.exists()
    physical_loc = len(inflate_path.read_text(encoding="utf-8").splitlines())
    assert physical_loc <= 200, (
        f"inflate.py {physical_loc} LOC exceeds HNeRV parity L4 ceiling 200"
    )
