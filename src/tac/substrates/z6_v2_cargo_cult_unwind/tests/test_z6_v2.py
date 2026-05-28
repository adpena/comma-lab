# SPDX-License-Identifier: MIT
"""Z6-v2 cargo-cult-unwind canonical L1 LONG-RUN MLX-LOCAL test suite.

Covers (per Catalog #229 PV + Catalog #168 AST discipline + Catalog #294 9-dim
checklist evidence): architecture forward shape + param count + 2-level
Rao-Ballard hierarchy boundary + FoE ego conditioning + Atick-Redlich coop
receiver loss term + Z6V2CU1 archive round-trip + numpy-portable inflate
contract + canonical contract registration.
"""

from __future__ import annotations

import torch

from tac.substrates.z6_v2_cargo_cult_unwind import (
    LANE_ID,
    OBSERVABILITY_SURFACE,
    SUBSTRATE_ID,
    Z6_V2_CONTRACT,
    Z6V2Archive,
    Z6V2Config,
    Z6V2ScoreAwareLoss,
    Z6V2ScoreAwareLossWeights,
    Z6V2Substrate,
    Z6V2_HEADER_SIZE,
    Z6V2_MAGIC,
    Z6V2_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


# Canonical config invariants


def test_canonical_substrate_id():
    assert SUBSTRATE_ID == "z6_v2_cargo_cult_unwind"


def test_canonical_lane_id():
    assert LANE_ID == "lane_z6_v2_cargo_cult_unwind_l0_scaffold_20260527"


def test_canonical_contract_id_matches():
    assert Z6_V2_CONTRACT.id == SUBSTRATE_ID
    assert Z6_V2_CONTRACT.lane_id == LANE_ID


def test_canonical_contract_operational_at_l1():
    # L1 promotion 2026-05-28: OPERATIONAL + runtime_overlay_consumed=True
    # per Catalog #220 invariant.
    assert Z6_V2_CONTRACT.score_improvement_mechanism_status == "OPERATIONAL"
    assert Z6_V2_CONTRACT.runtime_overlay_consumed is True


def test_canonical_observability_surface_6_facets():
    expected_facets = {
        "inspectable_per_layer",
        "decomposable_per_signal",
        "diff_able_across_runs",
        "queryable_post_hoc",
        "cite_able",
        "counterfactual_able",
    }
    assert set(OBSERVABILITY_SURFACE.keys()) == expected_facets


# Architecture forward + param-count invariants


def test_z6_v2_config_canonical_defaults():
    cfg = Z6V2Config(num_pairs=32)
    assert cfg.latent_dim == 24
    assert cfg.ego_dim == 6
    assert cfg.embed_dim == 64
    assert cfg.num_upsample_blocks == 7
    assert cfg.rao_ballard_level_boundary == 3
    assert cfg.film_generator_depth == 3
    assert cfg.film_hidden_width == 80
    assert cfg.output_height == 384
    assert cfg.output_width == 512


def test_z6_v2_substrate_param_count_near_300k():
    """Per design memo Candidate 1 ~300K params; empirical 307K @ defaults."""
    cfg = Z6V2Config(num_pairs=32)
    n = Z6V2Substrate(cfg).num_parameters()
    # ±15% tolerance per ~300K target.
    assert 250_000 <= n <= 350_000, f"got {n} params; target ~300K"


def test_z6_v2_substrate_forward_shape():
    cfg = Z6V2Config(num_pairs=8)
    model = Z6V2Substrate(cfg)
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (4, 3, 384, 512)
    assert rgb_1.shape == (4, 3, 384, 512)
    assert rgb_0.min().item() >= 0.0
    assert rgb_0.max().item() <= 1.0


def test_z6_v2_substrate_gradient_flow():
    cfg = Z6V2Config(num_pairs=8)
    model = Z6V2Substrate(cfg)
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    rgb_0, rgb_1 = model(idx)
    torch.manual_seed(42)
    target = torch.rand_like(rgb_0)
    loss = ((rgb_0 - target) ** 2).mean()
    loss.backward()
    # latents + ego_vecs both receive gradient (FoE ego conditioning OK).
    assert model.latents.grad is not None
    assert model.ego_vecs.grad is not None
    # At least one of them should have non-trivial gradient norm.
    lat_norm = model.latents.grad.norm().item()
    ego_norm = model.ego_vecs.grad.norm().item()
    assert (lat_norm > 0.0) or (ego_norm > 0.0), (
        f"both gradient norms are zero: lat={lat_norm} ego={ego_norm}"
    )


def test_z6_v2_layerwise_inspector_2_level_hierarchy():
    cfg = Z6V2Config(num_pairs=8)
    model = Z6V2Substrate(cfg)
    info = model.layerwise_inspector()
    # 7 blocks: first 3 = level_0_micro; remaining 4 = level_1_meso.
    assert len(info) == 7
    micro_blocks = [k for k in info if "level_0_micro" in k]
    meso_blocks = [k for k in info if "level_1_meso" in k]
    assert len(micro_blocks) == 3, f"expected 3 micro blocks; got {len(micro_blocks)}"
    assert len(meso_blocks) == 4, f"expected 4 meso blocks; got {len(meso_blocks)}"


# Archive round-trip invariants


def test_z6v2_archive_header_constants():
    assert Z6V2_MAGIC == b"Z6V2"
    assert Z6V2_HEADER_SIZE == 28
    assert Z6V2_SCHEMA_VERSION == 1


def test_z6v2_archive_pack_parse_roundtrip():
    cfg = Z6V2Config(num_pairs=8)
    model = Z6V2Substrate(cfg)
    sd = {
        k: v
        for k, v in model.state_dict().items()
        if k not in ("latents", "ego_vecs")
    }
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "rao_ballard_level_boundary": cfg.rao_ballard_level_boundary,
        "film_generator_depth": cfg.film_generator_depth,
        "film_hidden_width": cfg.film_hidden_width,
        "cooperative_receiver_beta": cfg.cooperative_receiver_beta,
    }
    blob = pack_archive(sd, model.latents.data, model.ego_vecs.data, meta)
    arc = parse_archive(blob)
    assert isinstance(arc, Z6V2Archive)
    assert arc.schema_version == Z6V2_SCHEMA_VERSION
    assert arc.latents.shape == (8, 24)
    assert arc.ego_vecs.shape == (8, 6)
    # Decoder state dict keys match.
    assert set(arc.decoder_state_dict.keys()) == set(sd.keys())
    # Meta keys preserved minus internal _* quant scales.
    assert "rao_ballard_level_boundary" in arc.meta
    assert "film_generator_depth" in arc.meta
    assert "cooperative_receiver_beta" in arc.meta


def test_z6v2_archive_size_under_600kb_at_8pair():
    """Z6-v2 archive sanity: ≤600KB at 8-pair config (sister NeRV-class size)."""
    cfg = Z6V2Config(num_pairs=8)
    model = Z6V2Substrate(cfg)
    sd = {
        k: v
        for k, v in model.state_dict().items()
        if k not in ("latents", "ego_vecs")
    }
    meta = {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "rao_ballard_level_boundary": cfg.rao_ballard_level_boundary,
        "film_generator_depth": cfg.film_generator_depth,
        "film_hidden_width": cfg.film_hidden_width,
        "cooperative_receiver_beta": cfg.cooperative_receiver_beta,
    }
    blob = pack_archive(sd, model.latents.data, model.ego_vecs.data, meta)
    assert len(blob) < 600_000


# Inflate runtime invariants


def test_inflate_module_under_200_loc_per_hnerv_l4_waiver():
    """HNeRV parity L4 waiver: inflate ≤200 LOC reviewable in 30 seconds."""
    from pathlib import Path
    inflate_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    lines = inflate_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 200, (
        f"inflate.py is {len(lines)} LOC; HNeRV L4 waiver budget is 200"
    )


def test_inflate_main_cli_signature_3arg_per_catalog_146():
    """Catalog #146 contest-compliant 3-arg runtime contract."""
    from pathlib import Path
    inflate_path = (
        Path(__file__).resolve().parent.parent / "inflate.py"
    )
    body = inflate_path.read_text(encoding="utf-8")
    # Must accept 3 positional args: archive_dir, output_dir, file_list.
    assert "archive_dir = Path(sys.argv[1])" in body
    assert "output_dir = Path(sys.argv[2])" in body
    assert "file_list_path = Path(sys.argv[3])" in body


# Score-aware loss invariants


def test_score_aware_loss_weights_canonical_constants():
    weights = Z6V2ScoreAwareLossWeights()
    assert weights.alpha_rate == 25.0
    assert weights.beta_seg == 100.0
    assert weights.contest_normalizer == 37_545_489.0
    # Cooperative-receiver primitive per Catalog #311.
    assert weights.delta_coop_receiver > 0.0
    assert weights.beta_atick_redlich > 0.0


def test_score_aware_loss_class_exists_and_torch_module():
    """Z6V2ScoreAwareLoss is a torch.nn.Module per the canonical pattern."""
    assert issubclass(Z6V2ScoreAwareLoss, torch.nn.Module)
