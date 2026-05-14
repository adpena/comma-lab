# SPDX-License-Identifier: MIT
"""Architecture tests for the PR95 LoRA/DoRA substrate.

Covers LoRAAdapter init, DoRAAdapter init, PR95LoRADoRADecoder integration,
forward correctness at init (LoRA B=0 -> base unchanged; DoRA m=||W||_col ->
base unchanged), parameter freezing, and tier assignment.
"""

from __future__ import annotations

import pytest
import torch

from tac.substrates.pr95_lora_dora import (
    DEFAULT_TIER_A_TARGETS,
    DEFAULT_TIER_B_TARGETS,
    DEFAULT_TIER_C_TARGETS,
    AdapterConfig,
    DoRAAdapter,
    LoRAAdapter,
    PR95LoRADoRADecoder,
)
from tac.substrates.pr95_lora_dora.pr95_base import HNeRVDecoder

# ---------------------------------------------------------------------------
# HNeRVDecoder byte-faithful port
# ---------------------------------------------------------------------------


def test_base_decoder_parameter_count_matches_pr95() -> None:
    dec = HNeRVDecoder()
    n = sum(p.numel() for p in dec.parameters())
    assert n == 228_958, f"PR95 byte-faithful port has {n} params, expected 228,958"


def test_base_decoder_forward_shape() -> None:
    dec = HNeRVDecoder()
    z = torch.randn(3, 28)
    out = dec(z)
    assert tuple(out.shape) == (3, 2, 3, 384, 512)


def test_base_decoder_output_range() -> None:
    dec = HNeRVDecoder()
    z = torch.randn(2, 28)
    out = dec(z)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 256.0  # sigmoid * 255


# ---------------------------------------------------------------------------
# AdapterConfig validation
# ---------------------------------------------------------------------------


def test_adapter_config_kind_validation() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        AdapterConfig(name="x", kind="not_a_kind")


def test_adapter_config_rank_validation() -> None:
    with pytest.raises(ValueError, match="rank must be"):
        AdapterConfig(name="x", rank=0)


def test_adapter_config_init_validation() -> None:
    with pytest.raises(ValueError, match="init must be"):
        AdapterConfig(name="x", init="random")


def test_adapter_config_alpha_defaults_to_rank() -> None:
    cfg = AdapterConfig(name="x", rank=16)
    assert cfg.alpha == 16.0


def test_adapter_config_alpha_override() -> None:
    cfg = AdapterConfig(name="x", rank=8, alpha=32.0)
    assert cfg.alpha == 32.0


# ---------------------------------------------------------------------------
# LoRAAdapter math
# ---------------------------------------------------------------------------


def test_lora_adapter_zero_b_init_delta_is_zero() -> None:
    W = torch.randn(64, 32, 3, 3)
    cfg = AdapterConfig(name="x", rank=4, init="zero_b")
    adapter = LoRAAdapter(W, cfg)
    delta = adapter.delta()
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_lora_adapter_delta_shape_matches_frozen() -> None:
    W = torch.randn(72, 18, 3, 3)
    cfg = AdapterConfig(name="x", rank=8)
    adapter = LoRAAdapter(W, cfg)
    delta = adapter.delta()
    assert delta.shape == W.shape


def test_lora_adapter_2d_weight() -> None:
    W = torch.randn(100, 50)
    cfg = AdapterConfig(name="x", rank=4)
    adapter = LoRAAdapter(W, cfg)
    delta = adapter.delta()
    assert delta.shape == W.shape


def test_lora_adapter_pissa_init_approximates_W() -> None:
    """PiSSA init: delta = scale * B@A should equal the top-r SVD reconstruction
    of W (divided by sqrt(scale) factor in init, then scaled back at forward
    -> net scale * sqrt(scale)^-2 = 1, so total magnitude ~ top-r SVD)."""
    torch.manual_seed(42)
    W = torch.randn(50, 30)
    cfg = AdapterConfig(name="x", rank=10, init="pissa", alpha=10.0)
    adapter = LoRAAdapter(W, cfg)
    delta = adapter.delta()
    # delta = scale * B @ A where scale = 10/10 = 1.0
    # PiSSA init scales A, B by 1/sqrt(scale) = 1/1 = 1, so delta is exactly top-10 SVD.
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    expected = U[:, :10] @ torch.diag(S[:10]) @ Vh[:10, :]
    assert torch.allclose(delta, expected, atol=1e-4)


def test_lora_adapter_scaling_factor() -> None:
    W = torch.randn(32, 16, 3, 3)
    cfg = AdapterConfig(name="x", rank=8, alpha=16.0)
    adapter = LoRAAdapter(W, cfg)
    assert adapter.scale == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# DoRAAdapter math
# ---------------------------------------------------------------------------


def test_dora_adapter_init_magnitude_matches_column_norm() -> None:
    W = torch.randn(64, 32, 3, 3)
    cfg = AdapterConfig(name="x", kind="dora", rank=4)
    adapter = DoRAAdapter(W, cfg)
    flat = W.reshape(64, -1)
    expected_mag = torch.linalg.norm(flat, dim=1)
    assert torch.allclose(adapter.magnitude, expected_mag)


def test_dora_adapter_forward_at_init_equals_frozen_weight() -> None:
    """At init, DoRA delta is zero (LoRA part has B=0) and m=||W||_col, so
    W_eff = m * W / ||W||_col = W."""
    W = torch.randn(48, 24, 3, 3)
    cfg = AdapterConfig(name="x", kind="dora", rank=4)
    adapter = DoRAAdapter(W, cfg)
    flat = W.reshape(48, -1)
    eff_flat = adapter.apply_to(flat)
    assert torch.allclose(eff_flat, flat, atol=1e-5)


def test_dora_adapter_kind_required() -> None:
    cfg = AdapterConfig(name="x", kind="lora", rank=4)
    with pytest.raises(ValueError, match="DoRAAdapter requires"):
        DoRAAdapter(torch.randn(8, 4), cfg)


# ---------------------------------------------------------------------------
# PR95LoRADoRADecoder integration
# ---------------------------------------------------------------------------


def _make_synthetic_pr95_state() -> tuple[dict, dict]:
    """Build a synthetic PR95 state_dict + meta (no archive needed)."""
    dec = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(384, 512))
    sd = {k: v.clone() for k, v in dec.state_dict().items()}
    meta = {"latent_dim": 28, "base_channels": 36, "eval_size": [384, 512], "n_pairs": 600}
    return sd, meta


def test_decoder_freezes_base_when_no_b_or_a_targets() -> None:
    sd, meta = _make_synthetic_pr95_state()
    configs = [AdapterConfig(name="blocks.0", rank=4)]
    wrapped = PR95LoRADoRADecoder(sd, meta, configs, tier_b_targets=(), tier_a_targets=())
    base_trainable = sum(p.numel() for n, p in wrapped.base.named_parameters() if p.requires_grad)
    assert base_trainable == 0, "Base must be frozen when no tier B/A targets"


def test_decoder_unfreezes_tier_b_a() -> None:
    sd, meta = _make_synthetic_pr95_state()
    configs: list = []
    wrapped = PR95LoRADoRADecoder(
        sd, meta, configs,
        tier_b_targets=("refine.0",), tier_a_targets=("rgb_0",),
    )
    refine_trainable = sum(p.numel() for p in wrapped.base.refine[0].parameters() if p.requires_grad)
    rgb_trainable = sum(p.numel() for p in wrapped.base.rgb_0.parameters() if p.requires_grad)
    assert refine_trainable > 0
    assert rgb_trainable > 0


def test_decoder_forward_at_init_equals_base() -> None:
    """All Tier C LoRA adapters at zero_b init -> wrapped == base."""
    torch.manual_seed(0)
    sd, meta = _make_synthetic_pr95_state()
    configs = [AdapterConfig(name=t, rank=8) for t in DEFAULT_TIER_C_TARGETS]
    wrapped = PR95LoRADoRADecoder(sd, meta, configs, tier_b_targets=(), tier_a_targets=())
    wrapped.eval()
    base = HNeRVDecoder()
    base.load_state_dict(sd)
    base.eval()
    z = torch.randn(2, 28)
    with torch.no_grad():
        out_w = wrapped(z)
        out_b = base(z)
    assert torch.allclose(out_w, out_b, atol=1e-4)


def test_decoder_full_tier_c_lora_trainable_count() -> None:
    sd, meta = _make_synthetic_pr95_state()
    configs = [AdapterConfig(name=t, rank=8) for t in DEFAULT_TIER_C_TARGETS]
    wrapped = PR95LoRADoRADecoder(
        sd, meta, configs,
        tier_b_targets=DEFAULT_TIER_B_TARGETS,
        tier_a_targets=DEFAULT_TIER_A_TARGETS,
    )
    trainable = wrapped.trainable_param_count()
    # Per deconstruction memo: 17,416 LoRA + 4,365 Tier B + 978 Tier A = 22,759
    # The number includes both A/B per LoRA = 17,416 LoRA params + 4,365 tier B + 978 tier A
    # Allow a small tolerance since the actual count includes Tier B + Tier A weights+biases.
    assert 22_000 < trainable < 24_500, f"Expected ~22,759 trainable, got {trainable}"


def test_decoder_dora_forward_at_init_close_to_base() -> None:
    """DoRA initialization should preserve W_eff = W_frozen at step 0
    (numerical-precision dependent)."""
    torch.manual_seed(0)
    sd, meta = _make_synthetic_pr95_state()
    configs = [AdapterConfig(name="blocks.0", kind="dora", rank=4)]
    wrapped = PR95LoRADoRADecoder(sd, meta, configs, tier_b_targets=(), tier_a_targets=())
    wrapped.eval()
    base = HNeRVDecoder()
    base.load_state_dict(sd)
    base.eval()
    z = torch.randn(2, 28)
    with torch.no_grad():
        out_w = wrapped(z)
        out_b = base(z)
    diff = (out_w - out_b).abs().max().item()
    assert diff < 1e-2, f"DoRA init should be near-identity, got {diff}"


def test_decoder_gradient_flows_through_lora_only() -> None:
    """LoRA adapter params should have non-zero grads after backward;
    frozen base params should have None grads (or zero if requires_grad=False)."""
    torch.manual_seed(0)
    sd, meta = _make_synthetic_pr95_state()
    configs = [AdapterConfig(name="blocks.0", rank=4)]
    wrapped = PR95LoRADoRADecoder(sd, meta, configs, tier_b_targets=(), tier_a_targets=())
    wrapped.train()
    z = torch.randn(2, 28, requires_grad=False)
    out = wrapped(z)
    # Force a nonzero grad path through LoRA: use a loss that depends on the
    # adapter via the conv weight. Since B=0 at init, we set A and B to small
    # random values first.
    with torch.no_grad():
        adapter = wrapped.adapters["blocks__0"]
        adapter.B.copy_(torch.randn_like(adapter.B) * 1e-3)
    out = wrapped(z)
    loss = out.mean()
    loss.backward()
    assert wrapped.adapters["blocks__0"].A.grad is not None
    assert wrapped.adapters["blocks__0"].B.grad is not None
    assert wrapped.adapters["blocks__0"].A.grad.abs().sum().item() > 0
    # Frozen base weight should NOT have its requires_grad set
    assert wrapped.base.blocks[0].weight.requires_grad is False
