# SPDX-License-Identifier: MIT
"""Tests for tac.composition.sinkhorn_ot_mixing — entropic OT mixing."""

from __future__ import annotations

import pytest
import torch

from tac.composition.frontier_primitives import sinkhorn_transport_plan
from tac.composition.sinkhorn_ot_mixing import (
    SINKHORN_MAGIC,
    SINKHORN_SCHEMA_VERSION,
    SinkhornError,
    SinkhornOTMixer,
    SinkhornOTMixerSpec,
    estimate_param_bytes,
    sinkhorn_solve,
)

# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_spec_defaults_are_sane() -> None:
    s = SinkhornOTMixerSpec()
    assert s.reg > 0
    assert s.max_iters > 0
    assert s.log_domain is True


def test_spec_rejects_non_positive_reg() -> None:
    with pytest.raises(SinkhornError, match="reg must be positive"):
        SinkhornOTMixerSpec(reg=0.0)


def test_spec_rejects_non_positive_max_iters() -> None:
    with pytest.raises(SinkhornError, match="max_iters must be positive"):
        SinkhornOTMixerSpec(max_iters=0)


def test_spec_rejects_non_positive_tol() -> None:
    with pytest.raises(SinkhornError, match="tol must be positive"):
        SinkhornOTMixerSpec(tol=-1e-5)


def test_spec_rejects_non_positive_eps() -> None:
    with pytest.raises(SinkhornError, match="eps must be positive"):
        SinkhornOTMixerSpec(eps=-1e-12)


def test_spec_rejects_nonfinite_public_floats() -> None:
    with pytest.raises(SinkhornError, match="reg must be positive"):
        SinkhornOTMixerSpec(reg=float("nan"))
    with pytest.raises(SinkhornError, match="tol must be positive"):
        SinkhornOTMixerSpec(tol=float("inf"))
    with pytest.raises(SinkhornError, match="eps must be positive"):
        SinkhornOTMixerSpec(eps=float("nan"))


# ---------------------------------------------------------------------------
# sinkhorn_solve direct
# ---------------------------------------------------------------------------


def test_solver_recovers_uniform_plan_for_zero_cost() -> None:
    n = 3
    a = torch.ones(n) / n
    b = torch.ones(n) / n
    cost = torch.zeros(n, n)
    plan, c_val, _ = sinkhorn_solve(a, b, cost, reg=0.1)
    # Zero-cost → uniform plan minimising entropy = uniform 1/n²
    assert torch.allclose(plan, torch.full_like(plan, 1.0 / (n * n)), atol=1e-5)
    assert torch.allclose(c_val, torch.tensor(0.0), atol=1e-6)


def test_solver_marginals_match_input() -> None:
    a = torch.tensor([0.3, 0.7])
    b = torch.tensor([0.4, 0.6])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    plan, _, _ = sinkhorn_solve(a, b, cost, reg=0.05)
    assert torch.allclose(plan.sum(dim=1), a, atol=1e-3)
    assert torch.allclose(plan.sum(dim=0), b, atol=1e-3)


def test_solver_cost_value_nonneg_for_nonneg_cost() -> None:
    a = torch.tensor([0.5, 0.5])
    b = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    _, c_val, _ = sinkhorn_solve(a, b, cost, reg=0.1)
    assert float(c_val) >= 0


def test_solver_converges_within_max_iters() -> None:
    a = torch.tensor([0.5, 0.5])
    b = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    _, _, iters = sinkhorn_solve(a, b, cost, reg=0.1, max_iters=500, tol=1e-8)
    assert iters <= 500


def test_solver_log_domain_matches_standard_mode_at_moderate_reg() -> None:
    a = torch.tensor([0.5, 0.5])
    b = torch.tensor([0.5, 0.5])
    cost = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    p_log, _, _ = sinkhorn_solve(a, b, cost, reg=0.5, log_domain=True)
    p_std, _, _ = sinkhorn_solve(a, b, cost, reg=0.5, log_domain=False)
    assert torch.allclose(p_log, p_std, atol=1e-4)


def test_solver_standard_mode_conforms_to_frontier_sinkhorn_plan() -> None:
    a = torch.tensor([0.3, 0.7])
    b = torch.tensor([0.4, 0.6])
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    plan, cost_value, iters = sinkhorn_solve(
        a,
        b,
        cost,
        reg=0.25,
        max_iters=300,
        tol=1e-7,
        log_domain=False,
    )
    canonical = sinkhorn_transport_plan(
        a,
        b,
        cost,
        epsilon=0.25,
        max_iters=300,
        tol=1e-7,
    )
    assert torch.allclose(plan, canonical.plan, atol=1e-6)
    assert torch.allclose(cost_value, (canonical.plan * cost).sum())
    assert iters == canonical.iterations


def test_solver_rejects_1d_cost() -> None:
    with pytest.raises(SinkhornError, match="cost must be a 2-D"):
        sinkhorn_solve(torch.ones(2), torch.ones(2), torch.zeros(4))


def test_solver_rejects_2d_a() -> None:
    with pytest.raises(SinkhornError, match="a and b must be 1-D"):
        sinkhorn_solve(torch.ones(2, 2), torch.ones(2), torch.zeros(2, 2))


def test_solver_rejects_negative_marginals() -> None:
    a = torch.tensor([-0.1, 1.1])
    b = torch.tensor([0.5, 0.5])
    with pytest.raises(SinkhornError, match="non-negative"):
        sinkhorn_solve(a, b, torch.zeros(2, 2))


def test_solver_rejects_zero_mass() -> None:
    a = torch.zeros(2)
    b = torch.tensor([0.5, 0.5])
    with pytest.raises(SinkhornError, match="positive total mass"):
        sinkhorn_solve(a, b, torch.zeros(2, 2))


def test_solver_cost_shape_mismatch_raises() -> None:
    with pytest.raises(SinkhornError, match="cost shape"):
        sinkhorn_solve(torch.ones(2), torch.ones(3), torch.zeros(2, 2))


def test_solver_rejects_nonfinite_inputs() -> None:
    with pytest.raises(SinkhornError, match="finite"):
        sinkhorn_solve(
            torch.tensor([0.5, float("nan")]),
            torch.ones(2),
            torch.zeros(2, 2),
        )


# ---------------------------------------------------------------------------
# SinkhornOTMixer class
# ---------------------------------------------------------------------------


def test_mixer_transport_returns_correct_shape() -> None:
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.1))
    src = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    tgt = torch.tensor([[0.5, 0.5], [1.5, 1.5]])
    z = mixer.transport(src, tgt)
    assert z.shape == (2, 2)


def test_mixer_transport_recovers_source_for_identity_anchors() -> None:
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.001))
    src = torch.tensor([[0.0], [1.0]])
    tgt = torch.tensor([[0.0], [1.0]])
    z = mixer.transport(src, tgt)
    # With tiny reg + cost minimised at identity matching, projection ≈ source.
    assert torch.allclose(z, src, atol=0.1)


def test_mixer_transport_with_weights() -> None:
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.1))
    src = torch.tensor([[0.0], [1.0]])
    tgt = torch.tensor([[0.5]])
    sw = torch.tensor([0.7, 0.3])
    tw = torch.tensor([1.0])
    z = mixer.transport(src, tgt, source_weights=sw, target_weights=tw)
    assert z.shape == (1, 1)
    # Weighted mix toward anchor 0.5: closer to src[0]=0.0 since sw[0]=0.7.
    assert float(z[0, 0]) < 0.5


def test_mixer_cost_value_is_scalar_tensor() -> None:
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.1))
    src = torch.tensor([[0.0], [1.0]])
    tgt = torch.tensor([[0.5]])
    c = mixer.cost(src, tgt)
    assert c.dim() == 0


def test_mixer_rejects_1d_source() -> None:
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec())
    with pytest.raises(SinkhornError, match="2-D"):
        mixer.transport(torch.zeros(3), torch.zeros(2, 1))


def test_mixer_rejects_mismatched_feature_dim() -> None:
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec())
    src = torch.zeros(2, 3)
    tgt = torch.zeros(2, 4)
    with pytest.raises(SinkhornError, match="feature dim"):
        mixer.transport(src, tgt)


def test_mixer_rejects_wrong_source_weight_shape() -> None:
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec())
    src = torch.zeros(2, 1)
    tgt = torch.zeros(2, 1)
    with pytest.raises(SinkhornError, match="source_weights shape"):
        mixer.transport(src, tgt, source_weights=torch.tensor([0.5]))


def test_mixer_rejects_wrong_target_weight_shape() -> None:
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec())
    src = torch.zeros(2, 1)
    tgt = torch.zeros(2, 1)
    with pytest.raises(SinkhornError, match="target_weights shape"):
        mixer.transport(src, tgt, target_weights=torch.tensor([1.0]))


def test_serialize_deserialize_roundtrip() -> None:
    spec = SinkhornOTMixerSpec(reg=0.05, max_iters=100, tol=1e-7, log_domain=False)
    mixer = SinkhornOTMixer(spec)
    blob = mixer.serialize_state()
    assert blob[:4] == SINKHORN_MAGIC
    restored = SinkhornOTMixer.deserialize_state(blob)
    assert restored.spec.reg == pytest.approx(0.05)
    assert restored.spec.max_iters == 100
    assert restored.spec.tol == pytest.approx(1e-7)
    assert restored.spec.log_domain is False


def test_deserialize_rejects_bad_magic() -> None:
    with pytest.raises(SinkhornError, match="bad magic"):
        SinkhornOTMixer.deserialize_state(b"XXXX" + b"\x00" * 40)


def test_deserialize_rejects_unknown_version() -> None:
    bad = (
        SINKHORN_MAGIC
        + (SINKHORN_SCHEMA_VERSION + 99).to_bytes(2, "little")
        + b"\x00" * 40
    )
    with pytest.raises(SinkhornError, match="unsupported schema"):
        SinkhornOTMixer.deserialize_state(bad)


def test_estimate_param_bytes_constant() -> None:
    # Spec layout is fixed-size.
    assert estimate_param_bytes(SinkhornOTMixerSpec()) == 35


def test_grad_flows_through_transport() -> None:
    src = torch.tensor([[0.0], [1.0]], requires_grad=True)
    tgt = torch.tensor([[0.5]])
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.1))
    z = mixer.transport(src, tgt)
    z.sum().backward()
    assert src.grad is not None
    assert torch.all(torch.isfinite(src.grad))


def test_grad_flows_through_cost() -> None:
    src = torch.tensor([[0.0], [1.0]], requires_grad=True)
    tgt = torch.tensor([[0.5]])
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.1))
    c = mixer.cost(src, tgt)
    c.backward()
    assert src.grad is not None
    assert torch.all(torch.isfinite(src.grad))


def test_mixer_dtype_preservation() -> None:
    src = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    tgt = torch.tensor([[0.5]], dtype=torch.float64)
    mixer = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.1))
    z = mixer.transport(src, tgt)
    assert z.dtype == torch.float64
