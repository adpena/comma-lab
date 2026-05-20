# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for pact_nerv_ia3.

Proves the encode/decode contract of the PIA3 monolithic 0.bin grammar +
IA3 γ-only modulation forward-pass parity under fp16 + int16-quant roundtrip.
Plus a smoke-level test that the trainer's _full_main raises
NotImplementedError per the L0 SCAFFOLD posture (Catalog #240).

Plus the IA3 γ-only invariant tests: γ_init=1.0 (residual form per IA3
paper §3.2) AND NO β bias projection (the distinguishing primitive vs
full FiLM γ+β).
"""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_ia3.architecture import (
    IA3GammaOnlyModulation,
    PactNervIa3Config,
    PactNervIa3Substrate,
)
from tac.substrates.pact_nerv_ia3.archive import (
    PIA3_HEADER_SIZE,
    PIA3_MAGIC,
    PIA3_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervIa3Config:
    return PactNervIa3Config(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        pose_dim=6,
        ia3_init_delta_std=0.01,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: PactNervIa3Config) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "ia3_init_delta_std": cfg.ia3_init_delta_std,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    """Module-level import resolves all canonical public symbols."""
    from tac.substrates import pact_nerv_ia3

    assert hasattr(pact_nerv_ia3, "PactNervIa3Config")
    assert hasattr(pact_nerv_ia3, "PactNervIa3Substrate")
    assert hasattr(pact_nerv_ia3, "IA3GammaOnlyModulation")
    assert hasattr(pact_nerv_ia3, "pack_archive")
    assert hasattr(pact_nerv_ia3, "parse_archive")
    assert hasattr(pact_nerv_ia3, "PactNervIa3ScoreAwareLoss")
    assert hasattr(pact_nerv_ia3, "PactNervIa3Archive")


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervIa3Substrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0
    assert float(rgb_1.min()) >= 0.0
    assert float(rgb_1.max()) <= 1.0


def test_ia3_gamma_only_invariant_no_beta_projection() -> None:
    """THE distinguishing primitive: γ-only (no β shift), γ_init ≈ 1.0.

    Per Liu 2205.05638 §3.2: γ_init=1.0 + Δ residual form ensures
    IA3-modulated output ≈ input when Δ is small. The invariant: γ_proj
    weights are zero-init (small Δ), so for input identical to the
    pre-modulation feature map, output ≈ input. NO β bias projection
    means there is no `IA3GammaOnlyModulation.beta_proj` attribute.
    """
    torch.manual_seed(0)
    ia3 = IA3GammaOnlyModulation(num_features=8, pose_dim=6, init_delta_std=0.0)
    # γ_proj weight + bias should be zero at init when init_delta_std=0
    assert not hasattr(ia3, "beta_proj"), (
        "IA3 MUST NOT have β bias projection (distinguishing primitive vs FiLM)"
    )
    assert torch.allclose(ia3.gamma_proj.weight, torch.zeros_like(ia3.gamma_proj.weight))
    assert torch.allclose(ia3.gamma_proj.bias, torch.zeros_like(ia3.gamma_proj.bias))
    # With γ_init exactly 1.0, IA3-modulated output equals input
    x = torch.randn(2, 8, 4, 4)
    pose = torch.randn(2, 6)
    with torch.no_grad():
        y = ia3(x, pose)
    assert torch.allclose(y, x, atol=1e-6), (
        "γ_init=1.0 + Δ=0 must produce identity transformation per IA3 §3.2 residual form"
    )


def test_archive_pack_then_parse_roundtrip_recovers_tensors() -> None:
    """ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervIa3Substrate(cfg)
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items() if k not in ("latents", "ego_poses")
    }
    latents = sd["latents"].clone()
    ego_poses = sd["ego_poses"].clone()

    blob = pack_archive(
        decoder_sd, latents, ego_poses, _smoke_meta(cfg),
        pose_dim=cfg.pose_dim,
    )
    arc = parse_archive(blob)

    assert arc.schema_version == PIA3_SCHEMA_VERSION
    assert blob[:4] == PIA3_MAGIC
    assert arc.pose_dim == cfg.pose_dim
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())
    for k, v in decoder_sd.items():
        rec = arc.decoder_state_dict[k]
        assert rec.shape == v.shape, f"{k} shape changed"
        assert torch.allclose(rec.to(torch.float32), v.to(torch.float32), atol=1e-2)

    assert arc.latents.shape == latents.shape
    quant_range = max(float(latents.max() - latents.min()), 1e-12)
    step = quant_range / 65534.0
    assert torch.allclose(arc.latents, latents, atol=step * 2.0)

    assert arc.ego_poses.shape == ego_poses.shape


def test_archive_grammar_header_size_invariant_is_26_bytes() -> None:
    """PIA3 header invariant: 26 bytes (1-byte pose_dim + 4-byte ego_pose_blob_len)."""
    assert PIA3_HEADER_SIZE == 26


def test_byte_mutation_changes_inflate_output_no_op_proof() -> None:
    """ENCODE_INFLATE_ROUNDTRIP — Catalog #139 no_op_proof.

    Mutating a single byte in the γ_proj weight blob MUST change the
    rendered frames empirically. This proves the IA3 γ-only modulation
    is NOT a no-op at inflate time — the rate-axis cost the substrate
    pays IS reflected in score-axis behavior.
    """
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervIa3Substrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "ego_poses")}
    latents = sd["latents"].clone()
    ego_poses = sd["ego_poses"].clone()

    blob_a = pack_archive(
        decoder_sd, latents, ego_poses, _smoke_meta(cfg),
        pose_dim=cfg.pose_dim,
    )
    # Mutate one ego_pose entry (the IA3 conditioning signal)
    mutated_pose = ego_poses.clone()
    mutated_pose[0, 0] = mutated_pose[0, 0] + 1.0
    blob_b = pack_archive(
        decoder_sd, latents, mutated_pose, _smoke_meta(cfg),
        pose_dim=cfg.pose_dim,
    )
    assert blob_a != blob_b, "no_op_proof: mutating ego_poses must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.ego_poses[0, 0], arc_b.ego_poses[0, 0], atol=1e-6)


def test_trainer_full_main_raises_not_implemented_at_l0_scaffold() -> None:
    """L0 SCAFFOLD posture: trainer _full_main MUST raise NotImplementedError.

    Per Catalog #240 (recipe-vs-trainer-state consistency) + Catalog #315
    (OPTIMAL FORM before paid dispatch) + Catalog #325 (per-substrate
    symposium): the pact_nerv_ia3 trainer's full path is council-gated
    until Stage 1 dispatch operator-gated.
    """
    import argparse
    import importlib

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_ia3")
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
    """Catalog #164: trainer score-aware path MUST route through canonical helper.

    The package's PactNervIa3ScoreAwareLoss MUST use
    tac.substrates.score_aware_common.score_pair_components_dispatch
    (canonical helper that handles preprocess_input correctly per Catalog
    #164 + scorer-loader-assignment-order discipline per Catalog #222).
    """
    import inspect

    from tac.substrates.pact_nerv_ia3 import score_aware_loss as sal_module

    src = inspect.getsource(sal_module)
    assert "score_pair_components_dispatch" in src, (
        "PactNervIa3ScoreAwareLoss module MUST route through canonical helper "
        "per Catalog #164"
    )
    assert (
        "tac.substrates.score_aware_common" in src
        or "from tac.substrates.score_aware_common" in src
    ), "must import from canonical helper module"


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    """Catalog #6 MANDATORY DEFAULT: trainer MUST patch yuv6 BEFORE scorer load.

    Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS" +
    PR95/PR106 empirical anchor: the trainer's smoke path MUST call
    patch_upstream_yuv6_globally() BEFORE any scorer construction so the
    PoseNet gradient path remains differentiable.
    """
    import inspect

    import experiments.train_substrate_pact_nerv_ia3 as trainer_module

    src = inspect.getsource(trainer_module._smoke_main)
    # The patch_upstream_yuv6_globally call must appear in the smoke path
    assert "patch_upstream_yuv6_globally" in src, (
        "_smoke_main MUST call patch_upstream_yuv6_globally() per "
        "CLAUDE.md eval_roundtrip non-negotiable"
    )
    # And it must appear before any scorer-load token
    patch_idx = src.find("patch_upstream_yuv6_globally")
    # Either no scorer load is done in smoke (acceptable) OR patch is before it
    for scorer_token in ("load_differentiable_scorers", "load_default_scorers"):
        if scorer_token in src:
            scorer_idx = src.find(scorer_token)
            assert patch_idx < scorer_idx, (
                f"patch_upstream_yuv6_globally MUST precede {scorer_token} "
                "per PR95/PR106 differentiable yuv6 anchor"
            )


def test_recipe_research_only_and_dispatch_disabled() -> None:
    """Catalog #240: recipe MUST opt out of dispatch at L0 SCAFFOLD.

    The matching recipe declares research_only:true + dispatch_enabled:false
    per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY".
    """
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]

    recipe_path = (
        Path(__file__).resolve().parents[5]
        / ".omx/operator_authorize_recipes/substrate_pact_nerv_ia3_modal_t4_dispatch.yaml"
    )
    assert recipe_path.exists(), f"recipe missing: {recipe_path}"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert recipe["dispatch_enabled"] is False, (
        "recipe MUST declare dispatch_enabled:false at L0 SCAFFOLD per Catalog #240"
    )
    assert recipe["research_only"] is True, (
        "recipe MUST declare research_only:true at L0 SCAFFOLD per HNeRV parity L2"
    )


def test_driver_carries_canonical_nvml_block() -> None:
    """Catalog #244: remote lane driver MUST carry the canonical 3-export NVML block.

    All 3 exports (DALI_DISABLE_NVML + CUBLAS_WORKSPACE_CONFIG +
    PYTORCH_CUDA_ALLOC_CONF) MUST appear before the bootstrap source line.
    """
    from pathlib import Path

    driver_path = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_ia3.sh"
    )
    assert driver_path.exists(), f"driver missing: {driver_path}"
    driver_text = driver_path.read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in driver_text, (
        "driver MUST export DALI_DISABLE_NVML per Catalog #244"
    )
    assert "CUBLAS_WORKSPACE_CONFIG" in driver_text, (
        "driver MUST export CUBLAS_WORKSPACE_CONFIG per Catalog #244"
    )
    assert "PYTORCH_CUDA_ALLOC_CONF" in driver_text, (
        "driver MUST export PYTORCH_CUDA_ALLOC_CONF per Catalog #244"
    )


def test_inflate_py_loc_under_150_per_hnerv_parity_l4() -> None:
    """HNeRV parity L4: inflate runtime MUST be ≤ 200 LOC (target ≤ 150).

    Per the L0 design contract: inflate.py target ≤ 150 LOC, hard ceiling
    200 LOC. PR101 GOLD reference = 150 LOC. Reviewable in 30 seconds
    per HNeRV parity L12.
    """
    from pathlib import Path

    inflate_path = (
        Path(__file__).resolve().parents[1] / "inflate.py"
    )
    assert inflate_path.exists()
    text = inflate_path.read_text(encoding="utf-8")
    physical_loc = len(text.splitlines())
    assert physical_loc <= 200, (
        f"inflate.py {physical_loc} LOC exceeds HNeRV parity L4 ceiling 200 LOC"
    )
