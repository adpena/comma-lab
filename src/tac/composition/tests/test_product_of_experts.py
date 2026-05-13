"""Tests for tac.composition.product_of_experts — PoE composer."""

from __future__ import annotations

import pytest
import torch

from tac.composition.product_of_experts import (
    POE_MAGIC,
    POE_SCHEMA_VERSION,
    ProductOfExpertsComposer,
    ProductOfExpertsError,
    ProductOfExpertsSpec,
)

# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_spec_defaults() -> None:
    s = ProductOfExpertsSpec()
    assert s.num_experts == 2
    assert s.temperature == 1.0
    assert s.mode == "log_density_sum"


def test_spec_rejects_zero_experts() -> None:
    with pytest.raises(ProductOfExpertsError, match="num_experts must"):
        ProductOfExpertsSpec(num_experts=0)


def test_spec_rejects_alpha_length_mismatch() -> None:
    with pytest.raises(ProductOfExpertsError, match="per_expert_alpha length"):
        ProductOfExpertsSpec(num_experts=2, per_expert_alpha=(1.0,))


def test_spec_rejects_negative_alpha() -> None:
    with pytest.raises(ProductOfExpertsError, match="non-negative"):
        ProductOfExpertsSpec(num_experts=2, per_expert_alpha=(1.0, -0.5))


def test_spec_rejects_nonfinite_alpha() -> None:
    with pytest.raises(ProductOfExpertsError, match="finite"):
        ProductOfExpertsSpec(num_experts=2, per_expert_alpha=(1.0, float("nan")))


def test_spec_rejects_non_positive_temperature() -> None:
    with pytest.raises(ProductOfExpertsError, match="temperature must"):
        ProductOfExpertsSpec(temperature=0.0)


def test_spec_rejects_nonfinite_temperature() -> None:
    with pytest.raises(ProductOfExpertsError, match="temperature must"):
        ProductOfExpertsSpec(temperature=float("inf"))


def test_spec_rejects_unknown_mode() -> None:
    with pytest.raises(ProductOfExpertsError, match="unknown mode"):
        ProductOfExpertsSpec(mode="banana")


# ---------------------------------------------------------------------------
# combine_log_densities
# ---------------------------------------------------------------------------


def test_combine_log_densities_uniform_alpha_is_sum() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    lp1 = torch.tensor([0.0, -1.0])
    lp2 = torch.tensor([-2.0, 0.0])
    result = composer.combine_log_densities([lp1, lp2])
    assert torch.allclose(result, torch.tensor([-2.0, -1.0]))


def test_combine_log_densities_weighted_alpha() -> None:
    composer = ProductOfExpertsComposer(
        ProductOfExpertsSpec(num_experts=2, per_expert_alpha=(2.0, 0.5))
    )
    lp1 = torch.tensor([1.0])
    lp2 = torch.tensor([4.0])
    result = composer.combine_log_densities([lp1, lp2])
    # 2.0 * 1.0 + 0.5 * 4.0 = 2.0 + 2.0 = 4.0
    assert torch.allclose(result, torch.tensor([4.0]))


def test_combine_log_densities_wrong_count_raises() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=3))
    with pytest.raises(ProductOfExpertsError, match="expected 3 log_densities"):
        composer.combine_log_densities([torch.zeros(2), torch.zeros(2)])


def test_combine_log_densities_shape_mismatch_raises() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    with pytest.raises(ProductOfExpertsError, match="shape"):
        composer.combine_log_densities([torch.zeros(2), torch.zeros(3)])


# ---------------------------------------------------------------------------
# soft_gate
# ---------------------------------------------------------------------------


def test_soft_gate_recovers_expert_when_one_dominates() -> None:
    composer = ProductOfExpertsComposer(
        ProductOfExpertsSpec(num_experts=2, temperature=0.01)
    )
    lp1 = torch.tensor([100.0])  # very high
    lp2 = torch.tensor([0.0])
    out1 = torch.tensor([[1.0]])
    out2 = torch.tensor([[5.0]])
    result = composer.soft_gate([lp1, lp2], [out1, out2])
    assert torch.allclose(result, out1, atol=1e-3)


def test_soft_gate_balances_when_equal() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    lp1 = torch.tensor([0.0])
    lp2 = torch.tensor([0.0])
    out1 = torch.tensor([[2.0]])
    out2 = torch.tensor([[4.0]])
    result = composer.soft_gate([lp1, lp2], [out1, out2])
    # Equal log-likelihoods → 50/50 mix = 3.0
    assert torch.allclose(result, torch.tensor([[3.0]]), atol=1e-5)


def test_soft_gate_wrong_logp_count_raises() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    with pytest.raises(ProductOfExpertsError, match="log-likelihoods"):
        composer.soft_gate(
            [torch.zeros(1)],
            [torch.zeros(1, 1), torch.zeros(1, 1)],
        )


def test_soft_gate_wrong_outputs_count_raises() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    with pytest.raises(ProductOfExpertsError, match="outputs"):
        composer.soft_gate(
            [torch.zeros(1), torch.zeros(1)],
            [torch.zeros(1, 1)],
        )


def test_soft_gate_validates_output_shape_against_log_likelihood_shape() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    with pytest.raises(ProductOfExpertsError, match="leading dims"):
        composer.soft_gate(
            [torch.zeros(2), torch.zeros(2)],
            [torch.zeros(3, 1), torch.zeros(3, 1)],
        )
    with pytest.raises(ProductOfExpertsError, match="log_likelihood_shape"):
        composer.soft_gate(
            [torch.zeros(2), torch.zeros(2)],
            [torch.zeros(2), torch.zeros(2)],
        )


def test_soft_gate_rejects_nonfinite_log_likelihoods() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    with pytest.raises(ProductOfExpertsError, match="finite"):
        composer.soft_gate(
            [torch.tensor([float("nan")]), torch.zeros(1)],
            [torch.zeros(1, 1), torch.zeros(1, 1)],
        )


def test_soft_gate_grad_flows() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    lp1 = torch.tensor([0.0], requires_grad=True)
    lp2 = torch.tensor([0.0], requires_grad=True)
    out1 = torch.tensor([[1.0]], requires_grad=True)
    out2 = torch.tensor([[2.0]], requires_grad=True)
    result = composer.soft_gate([lp1, lp2], [out1, out2])
    result.sum().backward()
    assert out1.grad is not None
    assert out2.grad is not None


# ---------------------------------------------------------------------------
# hard_gate
# ---------------------------------------------------------------------------


def test_hard_gate_picks_argmax_expert() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    lp1 = torch.tensor([5.0, -1.0])
    lp2 = torch.tensor([-1.0, 3.0])
    out1 = torch.tensor([[1.0], [1.0]])
    out2 = torch.tensor([[2.0], [2.0]])
    result = composer.hard_gate([lp1, lp2], [out1, out2])
    assert result.shape == (2, 1)
    # Position 0 → expert 0 (lp1 wins); position 1 → expert 1 (lp2 wins).
    assert torch.allclose(result, torch.tensor([[1.0], [2.0]]))


def test_hard_gate_wrong_counts_raise() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    with pytest.raises(ProductOfExpertsError):
        composer.hard_gate([torch.zeros(1)], [torch.zeros(1, 1), torch.zeros(1, 1)])
    with pytest.raises(ProductOfExpertsError):
        composer.hard_gate(
            [torch.zeros(1), torch.zeros(1)],
            [torch.zeros(1, 1)],
        )


def test_hard_gate_validates_output_shape() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    with pytest.raises(ProductOfExpertsError, match="leading dims"):
        composer.hard_gate(
            [torch.zeros(2), torch.zeros(2)],
            [torch.zeros(3, 1), torch.zeros(3, 1)],
        )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_serialize_starts_with_magic() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec())
    blob = composer.serialize_state()
    assert blob[:4] == POE_MAGIC


def test_serialize_deserialize_roundtrip_uniform_alpha() -> None:
    spec = ProductOfExpertsSpec(num_experts=3, temperature=0.5, mode="soft_gate")
    composer = ProductOfExpertsComposer(spec)
    blob = composer.serialize_state()
    restored = ProductOfExpertsComposer.deserialize_state(blob)
    assert restored.spec == spec


def test_serialize_deserialize_roundtrip_explicit_alpha() -> None:
    spec = ProductOfExpertsSpec(
        num_experts=2,
        per_expert_alpha=(1.5, 0.5),
        temperature=2.0,
        mode="hard_gate",
    )
    composer = ProductOfExpertsComposer(spec)
    blob = composer.serialize_state()
    restored = ProductOfExpertsComposer.deserialize_state(blob)
    assert restored.spec == spec


def test_deserialize_rejects_bad_magic() -> None:
    with pytest.raises(ProductOfExpertsError, match="bad magic"):
        ProductOfExpertsComposer.deserialize_state(b"XXXX" + b"\x00" * 40)


def test_deserialize_rejects_unknown_version() -> None:
    bad = (
        POE_MAGIC
        + (POE_SCHEMA_VERSION + 99).to_bytes(2, "little")
        + b"\x00" * 100
    )
    with pytest.raises(ProductOfExpertsError, match="unsupported schema"):
        ProductOfExpertsComposer.deserialize_state(bad)


def test_combine_dtype_preservation() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=2))
    lp1 = torch.tensor([0.0], dtype=torch.float64)
    lp2 = torch.tensor([0.0], dtype=torch.float64)
    out = composer.combine_log_densities([lp1, lp2])
    assert out.dtype == torch.float64


def test_single_expert_passthrough() -> None:
    composer = ProductOfExpertsComposer(ProductOfExpertsSpec(num_experts=1))
    lp = torch.tensor([1.5, -0.5])
    result = composer.combine_log_densities([lp])
    assert torch.allclose(result, lp)
