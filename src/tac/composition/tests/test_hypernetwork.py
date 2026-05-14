# SPDX-License-Identifier: MIT
"""Tests for tac.composition.hypernetwork — Hypernetwork composer."""

from __future__ import annotations

import pytest
import torch

from tac.composition.hypernetwork import (
    HYPER_MAGIC,
    HYPER_SCHEMA_VERSION,
    Hypernetwork,
    HypernetworkError,
    HypernetworkSpec,
)

# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_spec_defaults_are_sane() -> None:
    s = HypernetworkSpec()
    assert s.latent_dim > 0
    assert s.hidden_dim > 0
    assert s.output_dim > 0
    assert s.num_codes > 0
    assert s.activation == "gelu"


def test_spec_rejects_zero_latent_dim() -> None:
    with pytest.raises(HypernetworkError, match="latent_dim must be positive"):
        HypernetworkSpec(latent_dim=0)


def test_spec_rejects_zero_hidden_dim() -> None:
    with pytest.raises(HypernetworkError, match="hidden_dim must be positive"):
        HypernetworkSpec(hidden_dim=0)


def test_spec_rejects_zero_output_dim() -> None:
    with pytest.raises(HypernetworkError, match="output_dim must be in"):
        HypernetworkSpec(output_dim=0)


def test_spec_rejects_huge_output_dim() -> None:
    with pytest.raises(HypernetworkError, match="output_dim must be in"):
        HypernetworkSpec(output_dim=1 << 30)


def test_spec_rejects_zero_num_codes() -> None:
    with pytest.raises(HypernetworkError, match="num_codes must be positive"):
        HypernetworkSpec(num_codes=0)


def test_spec_rejects_unknown_activation() -> None:
    with pytest.raises(HypernetworkError, match="activation must be"):
        HypernetworkSpec(activation="banana")


def test_spec_rejects_negative_init_scale() -> None:
    with pytest.raises(HypernetworkError, match="init_scale must be positive"):
        HypernetworkSpec(init_scale=-1.0)


def test_spec_rejects_nonfinite_init_scale() -> None:
    with pytest.raises(HypernetworkError, match="init_scale must be positive"):
        HypernetworkSpec(init_scale=float("inf"))


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_forward_returns_correct_shape() -> None:
    spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=16, num_codes=3)
    h = Hypernetwork(spec)
    theta = h(torch.tensor([0, 1, 2]))
    assert theta.shape == (3, 16)


def test_forward_single_code_index() -> None:
    spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=8, num_codes=5)
    h = Hypernetwork(spec)
    theta = h(torch.tensor([2]))
    assert theta.shape == (1, 8)


def test_forward_rejects_2d_index() -> None:
    spec = HypernetworkSpec()
    h = Hypernetwork(spec)
    with pytest.raises(HypernetworkError, match="must be 1-D"):
        h(torch.zeros((2, 2), dtype=torch.long))


def test_forward_rejects_out_of_range_index() -> None:
    spec = HypernetworkSpec(num_codes=3)
    h = Hypernetwork(spec)
    with pytest.raises(HypernetworkError, match="out of range"):
        h(torch.tensor([5]))


def test_forward_rejects_float_index() -> None:
    spec = HypernetworkSpec()
    h = Hypernetwork(spec)
    with pytest.raises(HypernetworkError, match="int32/int64"):
        h(torch.tensor([0.0]))


def test_forward_grad_flows_to_codes() -> None:
    spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=8, num_codes=2)
    h = Hypernetwork(spec)
    theta = h(torch.tensor([0, 1]))
    theta.sum().backward()
    assert h.codes.grad is not None
    assert h.fc1.weight.grad is not None
    assert h.fc2.weight.grad is not None


def test_generate_from_latent_continuous() -> None:
    spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=8)
    h = Hypernetwork(spec)
    z = torch.randn(5, 2)
    out = h.generate_from_latent(z)
    assert out.shape == (5, 8)


def test_generate_from_latent_rejects_bad_dim() -> None:
    spec = HypernetworkSpec(latent_dim=2)
    h = Hypernetwork(spec)
    with pytest.raises(HypernetworkError, match="latent_dim"):
        h.generate_from_latent(torch.zeros(5, 3))


def test_generate_from_latent_rejects_nonfinite_values() -> None:
    spec = HypernetworkSpec(latent_dim=2)
    h = Hypernetwork(spec)
    with pytest.raises(HypernetworkError, match="finite"):
        h.generate_from_latent(torch.tensor([[0.0, float("nan")]]))


# ---------------------------------------------------------------------------
# Different activations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("act_name", ["gelu", "relu", "tanh"])
def test_activations_all_work(act_name: str) -> None:
    spec = HypernetworkSpec(
        latent_dim=2, hidden_dim=4, output_dim=8, activation=act_name
    )
    h = Hypernetwork(spec)
    theta = h(torch.tensor([0]))
    assert theta.shape == (1, 8)
    assert torch.all(torch.isfinite(theta))


# ---------------------------------------------------------------------------
# Param-bytes accounting
# ---------------------------------------------------------------------------


def test_param_bytes_includes_all_components() -> None:
    spec = HypernetworkSpec(
        latent_dim=4, hidden_dim=8, output_dim=16, num_codes=10
    )
    h = Hypernetwork(spec)
    expected = (
        4 * 8 + 8 + 8 * 16 + 16 + 10 * 4  # 32 + 8 + 128 + 16 + 40 = 224
    ) * 4
    assert h.estimate_total_param_bytes() == expected


def test_param_bytes_dtype_scaling() -> None:
    spec = HypernetworkSpec(
        latent_dim=2, hidden_dim=4, output_dim=8, num_codes=2
    )
    h = Hypernetwork(spec)
    assert h.estimate_total_param_bytes(dtype_bytes=4) == 2 * h.estimate_total_param_bytes(
        dtype_bytes=2
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_serialize_starts_with_magic() -> None:
    h = Hypernetwork(HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=4))
    blob = h.serialize_state()
    assert blob[:4] == HYPER_MAGIC


def test_serialize_deserialize_roundtrip_recovers_spec() -> None:
    spec = HypernetworkSpec(
        latent_dim=3, hidden_dim=5, output_dim=11, num_codes=7, activation="tanh"
    )
    h = Hypernetwork(spec)
    blob = h.serialize_state()
    h2 = Hypernetwork.deserialize_state(blob)
    assert h2.spec == spec


def test_serialize_deserialize_roundtrip_recovers_weights() -> None:
    spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=4, num_codes=3)
    h = Hypernetwork(spec)
    # Mutate weights to non-init values.
    with torch.no_grad():
        h.fc1.weight.fill_(0.25)
        h.fc2.bias.fill_(-0.5)
        h.codes.fill_(0.7)
    blob = h.serialize_state()
    h2 = Hypernetwork.deserialize_state(blob)
    assert torch.allclose(h2.fc1.weight, h.fc1.weight)
    assert torch.allclose(h2.fc2.bias, h.fc2.bias)
    assert torch.allclose(h2.codes, h.codes)


def test_serialize_deserialize_forward_matches() -> None:
    spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=8, num_codes=3)
    h = Hypernetwork(spec)
    blob = h.serialize_state()
    h2 = Hypernetwork.deserialize_state(blob)
    idx = torch.tensor([0, 1, 2])
    out1 = h(idx)
    out2 = h2(idx)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_deserialize_rejects_bad_magic() -> None:
    with pytest.raises(HypernetworkError, match="bad magic"):
        Hypernetwork.deserialize_state(b"XXXX" + b"\x00" * 40)


def test_deserialize_rejects_unknown_version() -> None:
    bad = (
        HYPER_MAGIC
        + (HYPER_SCHEMA_VERSION + 99).to_bytes(2, "little")
        + b"\x00" * 100
    )
    with pytest.raises(HypernetworkError, match="unsupported schema"):
        Hypernetwork.deserialize_state(bad)


def test_codes_are_trainable() -> None:
    spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=4, num_codes=3)
    h = Hypernetwork(spec)
    assert h.codes.requires_grad


def test_compactness_vs_direct_storage() -> None:
    # 600 pairs of 1024-dim target weights via 4-dim latent + small MLP.
    spec = HypernetworkSpec(
        latent_dim=4, hidden_dim=8, output_dim=1024, num_codes=600
    )
    h = Hypernetwork(spec)
    hyper_bytes = h.estimate_total_param_bytes()
    direct_bytes = 600 * 1024 * 4
    assert hyper_bytes < direct_bytes
