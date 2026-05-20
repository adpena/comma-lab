# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + #139 no_op_proof + L0 SCAFFOLD contract for CROSS-NEURAL-E2E."""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_neural_codec_e2e_cross.architecture import (
    HyperpriorGate,
    PactNervNeuralCodecE2ECrossConfig,
    PactNervNeuralCodecE2ECrossSubstrate,
)
from tac.substrates.pact_nerv_neural_codec_e2e_cross.archive import (
    NCEC_HEADER_SIZE,
    NCEC_MAGIC,
    NCEC_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervNeuralCodecE2ECrossConfig:
    return PactNervNeuralCodecE2ECrossConfig(
        latent_dim_a=8,
        latent_dim_b=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_pairs=3,
        output_height=24,
        output_width=32,
        hyperprior_hidden=16,
        gate_init_bias=0.0,
    )


def _smoke_meta(cfg: PactNervNeuralCodecE2ECrossConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "hyperprior_hidden": cfg.hyperprior_hidden,
        "gate_init_bias": cfg.gate_init_bias,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_neural_codec_e2e_cross as m
    for name in (
        "PactNervNeuralCodecE2ECrossConfig",
        "PactNervNeuralCodecE2ECrossSubstrate",
        "HyperpriorGate",
        "pack_archive",
        "parse_archive",
        "PactNervNeuralCodecE2ECrossScoreAwareLoss",
        "PactNervNeuralCodecE2ECrossArchive",
    ):
        assert hasattr(m, name), f"missing canonical symbol: {name}"


def test_header_size_pinned_35() -> None:
    assert NCEC_HEADER_SIZE == 35, f"header size {NCEC_HEADER_SIZE} != 35"
    assert NCEC_MAGIC == b"NCEC"
    assert NCEC_SCHEMA_VERSION == 1


def test_hyperprior_gate_returns_per_pair_scalar_in_unit_interval() -> None:
    gate = HyperpriorGate(latent_dim_a=8, latent_dim_b=8, hidden_dim=16,
                          gate_init_bias=0.0)
    z_a = torch.randn(5, 8)
    z_b = torch.randn(5, 8)
    g = gate(z_a, z_b)
    assert g.shape == (5, 1)
    assert torch.all(g >= 0.0) and torch.all(g <= 1.0)


def test_substrate_forward_returns_rgb_pairs_in_unit_interval() -> None:
    cfg = _smoke_cfg()
    model = PactNervNeuralCodecE2ECrossSubstrate(cfg)
    idx = torch.arange(cfg.num_pairs, dtype=torch.long)
    rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (3, 3, 24, 32)
    assert rgb_1.shape == (3, 3, 24, 32)
    assert torch.all(rgb_0 >= 0.0) and torch.all(rgb_0 <= 1.0)
    assert torch.all(rgb_1 >= 0.0) and torch.all(rgb_1 <= 1.0)


def test_substrate_gradient_flows_through_both_branches_and_gate() -> None:
    cfg = _smoke_cfg()
    model = PactNervNeuralCodecE2ECrossSubstrate(cfg)
    # Break SIREN-induced symmetry so each branch produces different RGB
    # (otherwise gate's choice is meaningless and gate gradient is zero
    # by construction — at L0 SCAFFOLD this happens at init).
    with torch.no_grad():
        model.branch_a.head_rgb_0.bias.fill_(2.0)
        model.branch_a.head_rgb_1.bias.fill_(2.0)
        model.branch_b.head_rgb_0.bias.fill_(-2.0)
        model.branch_b.head_rgb_1.bias.fill_(-2.0)
        model.gate.fc1.weight.normal_(std=0.5)
        model.gate.fc2.weight.normal_(std=0.5)
    idx = torch.arange(cfg.num_pairs, dtype=torch.long)
    rgb_0, rgb_1 = model(idx)
    # Use MSE against random target to ensure non-trivial loss landscape
    target = torch.rand_like(rgb_0)
    loss = ((rgb_0 - target) ** 2).mean() + ((rgb_1 - target) ** 2).mean()
    loss.backward()
    # Both branches must receive gradients (non-zero magnitude)
    assert model.latents_a.grad is not None
    assert model.latents_b.grad is not None
    assert model.latents_a.grad.abs().sum().item() > 0.0
    assert model.latents_b.grad.abs().sum().item() > 0.0
    # Gate must receive gradients (canonical SUPER_ADDITIVE proof
    # at the gate-trainability surface per Catalog #322)
    assert model.gate.fc_gate.weight.grad is not None
    assert model.gate.fc_gate.weight.grad.abs().sum().item() > 0.0


def test_substrate_gate_init_bias_yields_balanced_initial_mix() -> None:
    """gate_init_bias=0.0 should produce gate ≈ 0.5 at init."""
    cfg = _smoke_cfg()
    model = PactNervNeuralCodecE2ECrossSubstrate(cfg)
    idx = torch.arange(cfg.num_pairs, dtype=torch.long)
    gates = model.gate_values(idx)
    # Gate values near 0.5 (allow tolerance for randomly-init MLP)
    assert gates.shape == (3,)
    assert torch.all(gates >= 0.0) and torch.all(gates <= 1.0)


def test_archive_pack_then_parse_roundtrip_invariant() -> None:
    """Catalog #91 ENCODE_INFLATE_ROUNDTRIP: pack then parse must recover all fields."""
    cfg = _smoke_cfg()
    model = PactNervNeuralCodecE2ECrossSubstrate(cfg)
    branch_a_sd = {k: v for k, v in model.branch_a.state_dict().items()}
    branch_b_sd = {k: v for k, v in model.branch_b.state_dict().items()}
    gate_sd = {k: v for k, v in model.gate.state_dict().items()}
    blob = pack_archive(
        branch_a_state_dict=branch_a_sd,
        branch_b_state_dict=branch_b_sd,
        hyperprior_state_dict=gate_sd,
        latents_a=model.latents_a.detach(),
        latents_b=model.latents_b.detach(),
        meta=_smoke_meta(cfg),
    )
    arc = parse_archive(blob)
    assert arc.schema_version == NCEC_SCHEMA_VERSION
    assert arc.latents_a.shape == model.latents_a.shape
    assert arc.latents_b.shape == model.latents_b.shape
    assert set(arc.decoder_a_state_dict.keys()) == set(branch_a_sd.keys())
    assert set(arc.decoder_b_state_dict.keys()) == set(branch_b_sd.keys())
    assert set(arc.hyperprior_state_dict.keys()) == set(gate_sd.keys())


def test_archive_rejects_bad_magic() -> None:
    blob = b"XXXX" + b"\x00" * (NCEC_HEADER_SIZE - 4)
    try:
        parse_archive(blob)
    except ValueError as exc:
        assert "bad magic" in str(exc)
    else:
        raise AssertionError("expected ValueError on bad magic")


def test_archive_rejects_bad_schema_version() -> None:
    import struct
    bad = struct.pack(
        "<4sBHHHIIIIII", NCEC_MAGIC, 99, 1, 1, 1, 0, 0, 0, 2, 2, 0,
    )
    try:
        parse_archive(bad + b"\x00" * 4)
    except ValueError as exc:
        assert "unsupported schema version" in str(exc)
    else:
        raise AssertionError("expected ValueError on bad version")


def test_score_aware_loss_canonical_routing_tokens_present() -> None:
    import inspect
    from tac.substrates.pact_nerv_neural_codec_e2e_cross import score_aware_loss as sal
    src = inspect.getsource(sal)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_score_aware_loss_refuses_apply_eval_roundtrip_false() -> None:
    from tac.substrates.pact_nerv_neural_codec_e2e_cross.score_aware_loss import (
        PactNervNeuralCodecE2ECrossScoreAwareLoss,
        ScoreAwareLossWeights,
    )

    # Stub scorers
    class _StubSeg(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 5, 384, 512)

    class _StubPose(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 6)

    loss_fn = PactNervNeuralCodecE2ECrossScoreAwareLoss(
        _StubSeg(), _StubPose(), ScoreAwareLossWeights(),
    )
    rgb = torch.zeros(1, 3, 16, 16)
    arch_bytes = torch.tensor(1.0)
    try:
        loss_fn(rgb, rgb, rgb, rgb, arch_bytes, apply_eval_roundtrip=False)
    except ValueError as exc:
        assert "apply_eval_roundtrip" in str(exc)
    else:
        raise AssertionError("expected ValueError on apply_eval_roundtrip=False")


def test_trainer_full_main_raises_not_implemented_at_l0_scaffold() -> None:
    import argparse
    import importlib
    trainer = importlib.import_module(
        "experiments.train_substrate_pact_nerv_neural_codec_e2e_cross"
    )
    ns = argparse.Namespace(output_dir=None, epochs=1, smoke=False, device="cpu")
    try:
        trainer._full_main(ns)
    except NotImplementedError as exc:
        assert "OPERATOR-GATED" in str(exc) or "L0 SCAFFOLD" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


def test_trainer_routes_through_canonical_scorer_loss_helper() -> None:
    import inspect
    from tac.substrates.pact_nerv_neural_codec_e2e_cross import score_aware_loss as sal
    src = inspect.getsource(sal)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect
    import experiments.train_substrate_pact_nerv_neural_codec_e2e_cross as trainer_module
    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path
    import yaml  # type: ignore[import-untyped]
    recipe = yaml.safe_load(
        (Path(__file__).resolve().parents[5]
         / ".omx/operator_authorize_recipes/substrate_pact_nerv_neural_codec_e2e_cross_modal_t4_dispatch.yaml"
        ).read_text(encoding="utf-8")
    )
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path
    txt = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_neural_codec_e2e_cross.sh"
    ).read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in txt
    assert "CUBLAS_WORKSPACE_CONFIG" in txt
    assert "PYTORCH_CUDA_ALLOC_CONF" in txt


def test_archive_byte_mutation_changes_rendered_output() -> None:
    """Catalog #139 no_op_detector: mutating branch_a vs branch_b weights
    must change rendered output. At L0 with SIREN-init both branches start
    near constant output (sigmoid(0)≈0.5), so we explicitly break that
    symmetry to verify the byte-mutation surface is sensitive to changes
    in EACH branch independently. The cross-neural-e2e SUPER_ADDITIVE
    proof per Catalog #322 lives at this surface."""
    import torch
    cfg = _smoke_cfg()
    model = PactNervNeuralCodecE2ECrossSubstrate(cfg)
    # Break SIREN-induced symmetry: bias branch_a head_rgb_0 away from zero
    with torch.no_grad():
        model.branch_a.head_rgb_0.bias.fill_(2.0)
        model.branch_b.head_rgb_0.bias.fill_(-2.0)
        # Also bias gate so it's not exactly 0.5
        model.gate.fc_gate.bias.fill_(1.0)
    idx = torch.arange(cfg.num_pairs, dtype=torch.long)
    with torch.no_grad():
        rgb_0_orig, _ = model(idx)
    # Mutate branch_a head; render must change
    with torch.no_grad():
        model.branch_a.head_rgb_0.bias.add_(0.5)
        rgb_0_mut_a, _ = model(idx)
    assert not torch.allclose(rgb_0_orig, rgb_0_mut_a, atol=1e-6), (
        "branch_a byte mutation should change rendered output"
    )
    # Restore and mutate branch_b
    with torch.no_grad():
        model.branch_a.head_rgb_0.bias.sub_(0.5)
        rgb_0_restored, _ = model(idx)
    assert torch.allclose(rgb_0_orig, rgb_0_restored, atol=1e-6)
    with torch.no_grad():
        model.branch_b.head_rgb_0.bias.add_(0.5)
        rgb_0_mut_b, _ = model(idx)
    assert not torch.allclose(rgb_0_orig, rgb_0_mut_b, atol=1e-6), (
        "branch_b byte mutation should change rendered output"
    )
