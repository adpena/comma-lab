# SPDX-License-Identifier: MIT
"""Tests for Tropical-Decomposition LoRA adapter."""

from __future__ import annotations

import pytest
import torch

from tac.composition.td_lora import (
    TD_LORA_MAGIC,
    TD_LORA_MAX_BRANCHES,
    TD_LORA_SCHEMA_VERSION,
    TropicalLoRAAdapter,
    TropicalLoRAError,
    TropicalLoRASpec,
    estimate_param_bytes,
    tropical_max_plus,
)


def _seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


def test_spec_validates_positive_dims() -> None:
    with pytest.raises(TropicalLoRAError):
        TropicalLoRASpec(in_features=0, out_features=4, rank=4)
    with pytest.raises(TropicalLoRAError):
        TropicalLoRASpec(in_features=4, out_features=-1, rank=4)


def test_spec_validates_rank_nonnegative() -> None:
    with pytest.raises(TropicalLoRAError):
        TropicalLoRASpec(in_features=4, out_features=4, rank=-1)


def test_spec_validates_branch_count() -> None:
    with pytest.raises(TropicalLoRAError):
        TropicalLoRASpec(in_features=4, out_features=4, rank=4, num_branches=0)
    with pytest.raises(TropicalLoRAError):
        TropicalLoRASpec(
            in_features=4,
            out_features=4,
            rank=4,
            num_branches=TD_LORA_MAX_BRANCHES + 1,
        )


def test_spec_requires_rank_divides_branches() -> None:
    with pytest.raises(TropicalLoRAError):
        TropicalLoRASpec(in_features=4, out_features=4, rank=3, num_branches=2)


def test_spec_zero_rank_legal_ablation() -> None:
    spec = TropicalLoRASpec(
        in_features=4, out_features=4, rank=0, num_branches=2
    )
    assert spec.per_branch_rank == 0


def test_spec_effective_alpha_defaults_to_per_branch_rank() -> None:
    spec = TropicalLoRASpec(
        in_features=4, out_features=4, rank=8, num_branches=2
    )
    assert spec.effective_alpha == 4.0


def test_spec_explicit_alpha_honoured() -> None:
    spec = TropicalLoRASpec(
        in_features=4, out_features=4, rank=8, num_branches=2, alpha=2.5
    )
    assert spec.effective_alpha == pytest.approx(2.5)


def test_spec_rejects_nonfinite_alpha() -> None:
    with pytest.raises(TropicalLoRAError, match="alpha must be finite"):
        TropicalLoRASpec(
            in_features=4,
            out_features=4,
            rank=8,
            num_branches=2,
            alpha=float("nan"),
        )


def test_forward_at_init_equals_base_when_include_base_true() -> None:
    _seed()
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    x = torch.randn(3, 6)
    base_out = module.base(x)
    out = module(x)
    assert torch.allclose(out, base_out, atol=1e-6)


def test_forward_shape_preserved_under_batch_axes() -> None:
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    x = torch.randn(2, 3, 6)
    out = module(x)
    assert out.shape == (2, 3, 4)


def test_forward_rejects_wrong_input_dim() -> None:
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    with pytest.raises(TropicalLoRAError):
        module(torch.randn(3, 5))


def test_base_params_are_frozen() -> None:
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    for p in module.base.parameters():
        assert not p.requires_grad
    for A in module.adapters_A:
        assert A.requires_grad
    for B in module.adapters_B:
        assert B.requires_grad


def test_gradient_flows_through_adapters_not_base() -> None:
    _seed()
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    # Make adapter branches dominate so the tropical max chooses them.
    with torch.no_grad():
        for B in module.adapters_B:
            B.fill_(1.0)
    x = torch.randn(3, 6)
    out = module(x).sum()
    out.backward()
    # Base weights either have None grad (no gradient computed) or zero.
    base_w_grad = module.base.weight.grad
    assert base_w_grad is None or torch.allclose(
        base_w_grad, torch.zeros_like(base_w_grad)
    )
    # At least one adapter should have a non-zero gradient.
    assert any(
        (B.grad is not None and B.grad.abs().sum() > 0)
        for B in module.adapters_B
    )


def test_noop_init_still_seeds_adapter_gradients() -> None:
    _seed()
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    x = torch.linspace(-1.0, 1.0, steps=18, dtype=torch.float32).reshape(3, 6)
    base_out = module.base(x)
    out = module(x)
    assert torch.allclose(out, base_out, atol=1e-6)

    out.sum().backward()
    assert module.base.weight.grad is None
    assert all(
        bias.grad is not None and bias.grad.abs().sum() > 0
        for bias in module.adapters_bias
    )
    assert any(
        B.grad is not None and B.grad.abs().sum() > 0
        for B in module.adapters_B
    )


def test_no_base_noop_init_seeds_all_tied_adapter_branches() -> None:
    _seed()
    spec = TropicalLoRASpec(
        in_features=6,
        out_features=4,
        rank=4,
        num_branches=2,
        include_base=False,
    )
    module = TropicalLoRAAdapter(spec)
    x = torch.linspace(-1.0, 1.0, steps=18, dtype=torch.float32).reshape(3, 6)
    out = module(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    out.sum().backward()
    assert all(
        bias.grad is not None and bias.grad.abs().sum() > 0
        for bias in module.adapters_bias
    )
    assert all(
        B.grad is not None and B.grad.abs().sum() > 0
        for B in module.adapters_B
    )


def test_serialize_deserialize_roundtrip() -> None:
    _seed()
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    with torch.no_grad():
        for A in module.adapters_A:
            A.normal_()
        for B in module.adapters_B:
            B.normal_()
        for bias in module.adapters_bias:
            bias.normal_()
    payload = module.serialize_state()
    assert payload[:4] == TD_LORA_MAGIC
    assert payload[4] == TD_LORA_SCHEMA_VERSION
    restored = TropicalLoRAAdapter.deserialize_state(payload)
    for orig, new in zip(module.adapters_A, restored.adapters_A, strict=True):
        assert torch.allclose(orig, new, atol=1e-6)
    for orig, new in zip(module.adapters_B, restored.adapters_B, strict=True):
        assert torch.allclose(orig, new, atol=1e-6)
    for orig, new in zip(
        module.adapters_bias,
        restored.adapters_bias,
        strict=True,
    ):
        assert torch.allclose(orig, new, atol=1e-6)


def test_serialize_deserialize_forward_matches_with_serialized_base() -> None:
    _seed()
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    with torch.no_grad():
        module.base.weight.normal_()
        module.base.bias.normal_()
        for A in module.adapters_A:
            A.normal_()
        for B in module.adapters_B:
            B.normal_()
        for bias in module.adapters_bias:
            bias.normal_()
    x = torch.randn(5, 6)
    restored = TropicalLoRAAdapter.deserialize_state(module.serialize_state())
    assert torch.allclose(module(x), restored(x), atol=0.0, rtol=0.0)


def test_deserialize_without_base_weights_fails_closed_or_accepts_supplied_base() -> None:
    _seed()
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    with torch.no_grad():
        for B in module.adapters_B:
            B.normal_()
    payload = module.serialize_state(include_base_weights=False)
    with pytest.raises(TropicalLoRAError, match="omits base weights"):
        TropicalLoRAAdapter.deserialize_state(payload)

    restored = TropicalLoRAAdapter.deserialize_state(
        payload,
        base_linear=module.base,
    )
    x = torch.randn(3, 6)
    assert torch.allclose(module(x), restored(x), atol=0.0, rtol=0.0)


def test_deserialize_rejects_wrong_magic() -> None:
    bad = b"XXXX" + b"\x00" * 32
    with pytest.raises(TropicalLoRAError):
        TropicalLoRAAdapter.deserialize_state(bad)


def test_deserialize_rejects_short_payload() -> None:
    with pytest.raises(TropicalLoRAError):
        TropicalLoRAAdapter.deserialize_state(b"abc")


def test_include_base_false_skips_base_branch() -> None:
    _seed()
    spec = TropicalLoRASpec(
        in_features=6,
        out_features=4,
        rank=4,
        num_branches=2,
        include_base=False,
    )
    module = TropicalLoRAAdapter(spec)
    with torch.no_grad():
        for B in module.adapters_B:
            B.normal_()
    x = torch.randn(3, 6)
    # All branches are adapters; with B initialised non-zero,
    # output should not match the base linear's forward.
    out = module(x)
    base_out = module.base(x)
    assert not torch.allclose(out, base_out)


def test_estimate_param_bytes_matches_actual_count() -> None:
    spec = TropicalLoRASpec(
        in_features=8, out_features=6, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    actual = sum(p.numel() for p in module.parameters() if p.requires_grad) * 4
    assert estimate_param_bytes(spec) == actual


def test_tropical_max_plus_helper() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.0, 1.0, 2.0])
    out = tropical_max_plus([a, b])
    assert torch.allclose(out, torch.tensor([3.0, 2.0, 3.0]))


def test_external_base_linear_reused() -> None:
    base = torch.nn.Linear(6, 4)
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    module = TropicalLoRAAdapter(spec, base_linear=base)
    assert module.base is base


def test_external_base_linear_shape_validated() -> None:
    base = torch.nn.Linear(6, 5)
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=4, num_branches=2
    )
    with pytest.raises(TropicalLoRAError):
        TropicalLoRAAdapter(spec, base_linear=base)


def test_rank_zero_module_is_just_base() -> None:
    _seed()
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=0, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    x = torch.randn(3, 6)
    out = module(x)
    base_out = module.base(x)
    assert torch.allclose(out, base_out)


def test_serialization_with_zero_rank() -> None:
    spec = TropicalLoRASpec(
        in_features=6, out_features=4, rank=0, num_branches=2
    )
    module = TropicalLoRAAdapter(spec)
    payload = module.serialize_state()
    restored = TropicalLoRAAdapter.deserialize_state(payload)
    assert restored.spec.rank == 0
    assert restored.spec.num_branches == 2
