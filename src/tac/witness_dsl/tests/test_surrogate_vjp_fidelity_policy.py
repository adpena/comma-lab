from __future__ import annotations

import pytest

from tac.witness_dsl.surrogate_vjp_fidelity_policy import (
    SurrogateVJPFidelityPolicy,
    SurrogateVJPFidelityPolicyError,
    validate_surrogate_vjp_fidelity_binding,
)


def test_explicit_null_measurement_receipt_is_valid_while_default_off() -> None:
    binding = SurrogateVJPFidelityPolicy().to_binding()
    assert "measurement_receipt" in binding
    assert binding["measurement_receipt"] is None
    policy = validate_surrogate_vjp_fidelity_binding(binding)
    assert policy.measurement_receipt is None
    assert policy.to_trainer_overrides() == {}


def test_absent_measurement_receipt_is_not_conflated_with_explicit_null() -> None:
    binding = SurrogateVJPFidelityPolicy().to_binding()
    del binding["measurement_receipt"]
    binding.pop("binding_sha256")
    with pytest.raises(
        SurrogateVJPFidelityPolicyError,
        match=r"missing required keys.*measurement_receipt",
    ):
        validate_surrogate_vjp_fidelity_binding(binding)


def test_enabled_policy_requires_all_three_sealed_receipts() -> None:
    with pytest.raises(SurrogateVJPFidelityPolicyError, match="requires all sealed receipts"):
        SurrogateVJPFidelityPolicy(enabled=True).to_binding()


def test_binding_hash_is_verified() -> None:
    binding = SurrogateVJPFidelityPolicy().to_binding()
    binding["anchor_k"] = 121
    with pytest.raises(SurrogateVJPFidelityPolicyError, match="hash mismatch"):
        validate_surrogate_vjp_fidelity_binding(binding)
