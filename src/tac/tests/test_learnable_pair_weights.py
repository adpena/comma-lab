"""Tests for src/tac/learnable_pair_weights.py — Lane W-V2 LearnablePairWeights.

Pins:
  1. raw is a learnable Parameter with grad enabled.
  2. softplus keeps weights >= 0 even with very negative raw.
  3. uniform init: weights ≈ 1.0 everywhere.
  4. warm_start: weights ≈ warm_start (within softplus rounding).
  5. gradient flows through self.raw when (weights * losses).sum() is backproped.
  6. Lagrangian rate penalty drives sum(weights) toward target_sum.
  7. save/load round-trip preserves raw and weights.
  8. Corrupt snapshots fail loud (schema_version + module ID checks).
  9. weight_for_pair returns the same value as full forward.
 10. Determinism: same warm_start → same raw initial value.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.learnable_pair_weights import (
    LearnablePairWeights,
    compute_pair_weight_rate_penalty,
    load_learnable_pair_weights,
    save_learnable_pair_weights,
    _inverse_softplus,
)


# ── Basic shape + parameter properties ───────────────────────────────────


def test_raw_is_learnable_parameter():
    pw = LearnablePairWeights(600)
    assert isinstance(pw.raw, nn.Parameter)
    assert pw.raw.requires_grad
    assert pw.raw.shape == (600,)


def test_uniform_init_gives_unit_weights():
    pw = LearnablePairWeights(64)
    w = pw()
    assert torch.allclose(w, torch.ones(64), atol=1e-4)


def test_softplus_keeps_weights_nonneg_even_with_negative_raw():
    pw = LearnablePairWeights(8)
    with torch.no_grad():
        pw.raw.fill_(-100.0)
    w = pw()
    assert (w >= 0).all()
    # softplus(-100) ≈ 0 within float32 precision
    assert w.max().item() < 1e-3


def test_warm_start_preserves_values():
    target = torch.tensor([0.5, 1.0, 5.0, 2.5, 0.1, 3.3])
    pw = LearnablePairWeights(6, warm_start=target)
    w = pw()
    diff = (w - target).abs()
    assert diff.max().item() < 1e-3, (
        f"warm_start mismatch: max diff {diff.max().item()}"
    )


def test_warm_start_shape_mismatch_raises():
    bad = torch.zeros(7)
    with pytest.raises(ValueError, match="warm_start shape"):
        LearnablePairWeights(6, warm_start=bad)


def test_warm_start_negative_raises():
    bad = torch.tensor([1.0, -0.5, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        LearnablePairWeights(3, warm_start=bad)


def test_warm_start_wrong_type_raises():
    with pytest.raises(TypeError, match="warm_start must be"):
        LearnablePairWeights(3, warm_start=[1.0, 2.0, 3.0])


def test_n_pairs_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        LearnablePairWeights(0)


# ── Gradient flow ────────────────────────────────────────────────────────


def test_gradient_flows_through_raw():
    pw = LearnablePairWeights(8)
    pair_losses = torch.linspace(0.1, 1.0, 8).requires_grad_(False)
    weighted = (pw() * pair_losses).sum()
    weighted.backward()
    assert pw.raw.grad is not None
    assert torch.isfinite(pw.raw.grad).all()
    # Larger losses → larger gradient on raw at uniform init
    # (∂(softplus(r) * loss)/∂r = sigmoid(r) * loss; same sigmoid factor
    # means the ranking of |grad| matches the ranking of |loss|).
    grad_order = pw.raw.grad.argsort()
    loss_order = pair_losses.argsort()
    assert torch.equal(grad_order, loss_order)


def test_gradient_zero_when_loss_zero():
    pw = LearnablePairWeights(4)
    z = (pw() * torch.zeros(4)).sum()
    z.backward()
    assert torch.allclose(pw.raw.grad, torch.zeros(4))


# ── Lagrangian rate penalty ──────────────────────────────────────────────


def test_rate_penalty_zero_at_target_sum():
    """At uniform init, sum(weights) = n_pairs ⇒ penalty = 0."""
    pw = LearnablePairWeights(10)
    pen = compute_pair_weight_rate_penalty(pw, target_sum=10.0, lambda_rate=1.0)
    assert pen.item() == pytest.approx(0.0, abs=1e-5)


def test_rate_penalty_positive_when_sum_above_target():
    pw = LearnablePairWeights(10, warm_start=torch.full((10,), 2.0))
    # sum = 20, target = 10 ⇒ penalty = 1 * (20 - 10)² = 100
    pen = compute_pair_weight_rate_penalty(pw, target_sum=10.0, lambda_rate=1.0)
    assert pen.item() == pytest.approx(100.0, rel=1e-3)


def test_rate_penalty_drives_sum_toward_target():
    """SGD on the penalty alone should move sum(weights) toward target_sum."""
    pw = LearnablePairWeights(8, warm_start=torch.full((8,), 3.0))
    optim = torch.optim.SGD([pw.raw], lr=0.05)
    initial_sum = pw().sum().item()
    target = 8.0
    for _ in range(200):
        optim.zero_grad()
        pen = compute_pair_weight_rate_penalty(pw, target_sum=target, lambda_rate=0.1)
        pen.backward()
        optim.step()
    final_sum = pw().sum().item()
    # The constraint should pull final_sum toward target, away from initial.
    assert abs(final_sum - target) < abs(initial_sum - target)


def test_rate_penalty_default_target_is_n_pairs():
    pw = LearnablePairWeights(7)
    pen = compute_pair_weight_rate_penalty(pw, lambda_rate=1.0)
    # sum(softplus(0.5413)) = 7 * 1.0 ≈ 7 = target ⇒ penalty ≈ 0
    assert pen.item() == pytest.approx(0.0, abs=1e-3)


# ── weight_for_pair ──────────────────────────────────────────────────────


def test_weight_for_pair_matches_full_forward():
    target = torch.tensor([0.5, 2.0, 3.0, 1.0])
    pw = LearnablePairWeights(4, warm_start=target)
    full = pw()
    for i in range(4):
        assert pw.weight_for_pair(i).item() == pytest.approx(full[i].item(), rel=1e-5)


def test_weight_for_pair_with_tensor_index():
    pw = LearnablePairWeights(8)
    idx = torch.tensor(3)
    w = pw.weight_for_pair(idx)
    assert torch.allclose(w, torch.tensor(1.0), atol=1e-4)


# ── Save / load round-trip ───────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path):
    target = torch.tensor([0.1, 5.0, 2.5, 1.0, 3.3])
    pw = LearnablePairWeights(5, warm_start=target)
    out = tmp_path / "pw.pt"
    save_learnable_pair_weights(pw, out)
    assert out.exists()
    pw2 = load_learnable_pair_weights(out)
    assert pw2.n_pairs == 5
    assert torch.allclose(pw.raw, pw2.raw, atol=1e-6)
    # Forward must match
    assert torch.allclose(pw(), pw2(), atol=1e-5)


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_learnable_pair_weights(tmp_path / "nonexistent.pt")


def test_load_wrong_module_raises(tmp_path: Path):
    """A snapshot with the wrong module ID must fail loud."""
    bad = {
        "schema_version": 1,
        "module": "some.other.Module",
        "n_pairs": 5,
        "raw": torch.zeros(5),
    }
    p = tmp_path / "bad.pt"
    torch.save(bad, p)
    with pytest.raises(ValueError, match="module="):
        load_learnable_pair_weights(p)


def test_load_wrong_schema_version_raises(tmp_path: Path):
    bad = {
        "schema_version": 99,
        "module": "tac.learnable_pair_weights.LearnablePairWeights",
        "n_pairs": 5,
        "raw": torch.zeros(5),
    }
    p = tmp_path / "bad.pt"
    torch.save(bad, p)
    with pytest.raises(ValueError, match="schema_version"):
        load_learnable_pair_weights(p)


def test_load_non_dict_raises(tmp_path: Path):
    p = tmp_path / "bad.pt"
    torch.save(torch.zeros(5), p)
    with pytest.raises(TypeError, match="snapshot"):
        load_learnable_pair_weights(p)


# ── Inverse-softplus helper ──────────────────────────────────────────────


def test_inverse_softplus_round_trip():
    y = torch.tensor([0.1, 1.0, 5.0, 10.0, 25.0])
    raw = _inverse_softplus(y)
    y_back = torch.nn.functional.softplus(raw)
    diff = (y - y_back).abs()
    assert diff.max().item() < 1e-3, (
        f"inverse_softplus round-trip max diff {diff.max().item()}"
    )


# ── Determinism ──────────────────────────────────────────────────────────


def test_deterministic_init_for_warm_start():
    target = torch.tensor([1.0, 5.0, 2.0])
    pw1 = LearnablePairWeights(3, warm_start=target)
    pw2 = LearnablePairWeights(3, warm_start=target)
    assert torch.equal(pw1.raw, pw2.raw)
