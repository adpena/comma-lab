# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for pact_nerv_vq."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_vq.architecture import (
    PactNervVqConfig,
    PactNervVqSubstrate,
    VectorQuantizerEMA,
)
from tac.substrates.pact_nerv_vq.archive import (
    PVQ_HEADER_SIZE,
    PVQ_MAGIC,
    PVQ_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervVqConfig:
    return PactNervVqConfig(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        codebook_size=16,
        codebook_decay=0.99,
        commitment_weight=0.25,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: PactNervVqConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "codebook_decay": cfg.codebook_decay,
        "commitment_weight": cfg.commitment_weight,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_vq as pkg

    assert hasattr(pkg, "PactNervVqConfig")
    assert hasattr(pkg, "PactNervVqSubstrate")
    assert hasattr(pkg, "VectorQuantizerEMA")
    assert hasattr(pkg, "pack_archive")
    assert hasattr(pkg, "parse_archive")
    assert hasattr(pkg, "PactNervVqScoreAwareLoss")
    assert hasattr(pkg, "PactNervVqArchive")


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervVqSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_vector_quantizer_ema_straight_through_gradient_flows() -> None:
    """Distinguishing primitive: STE means gradients flow through z_e."""
    torch.manual_seed(0)
    vq = VectorQuantizerEMA(codebook_size=16, latent_dim=8, decay=0.99)
    z_e = torch.randn(4, 8, requires_grad=True)
    z_q_st, indices, commitment = vq(z_e)
    assert z_q_st.shape == (4, 8)
    assert indices.shape == (4,)
    assert indices.dtype == torch.long
    assert commitment.requires_grad
    # Gradient flow check: backward through z_q_st should reach z_e.
    z_q_st.sum().backward()
    assert z_e.grad is not None
    assert z_e.grad.shape == z_e.shape


def test_archive_pack_then_parse_roundtrip_recovers_tensors() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervVqSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    codebook = model.quantizer.codebook.clone()
    indices = torch.tensor([0, 5, 7], dtype=torch.long)

    blob = pack_archive(decoder_sd, codebook, indices, _smoke_meta(cfg))
    arc = parse_archive(blob)

    assert arc.schema_version == PVQ_SCHEMA_VERSION
    assert blob[:4] == PVQ_MAGIC
    assert arc.codebook.shape == codebook.shape
    assert arc.indices.shape == indices.shape
    assert torch.equal(arc.indices, indices)


def test_archive_grammar_header_size_invariant_is_27_bytes() -> None:
    assert PVQ_HEADER_SIZE == 27


def test_byte_mutation_changes_archive_no_op_proof() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervVqSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
    codebook = model.quantizer.codebook.clone()
    indices_a = torch.tensor([0, 5, 7], dtype=torch.long)
    indices_b = torch.tensor([1, 5, 7], dtype=torch.long)

    blob_a = pack_archive(decoder_sd, codebook, indices_a, _smoke_meta(cfg))
    blob_b = pack_archive(decoder_sd, codebook, indices_b, _smoke_meta(cfg))
    assert blob_a != blob_b, "no_op_proof: mutating indices must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert int(arc_a.indices[0].item()) != int(arc_b.indices[0].item())


def test_trainer_full_main_raises_not_implemented_at_l0_scaffold() -> None:
    import argparse
    import importlib

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_vq")
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

    from tac.substrates.pact_nerv_vq import score_aware_loss as sal_module

    src = inspect.getsource(sal_module)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect

    import experiments.train_substrate_pact_nerv_vq as trainer_module

    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]

    recipe_path = (
        Path(__file__).resolve().parents[5]
        / ".omx/operator_authorize_recipes/substrate_pact_nerv_vq_modal_t4_dispatch.yaml"
    )
    assert recipe_path.exists(), f"recipe missing: {recipe_path}"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path

    driver_path = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_vq.sh"
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
    assert physical_loc <= 200, (
        f"inflate.py {physical_loc} LOC exceeds HNeRV parity L4 ceiling 200"
    )
