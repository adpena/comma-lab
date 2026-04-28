"""Tests for src/tac/learnable_pair_weights.py — Lane W-V2.

Round 10 (2026-04-27) replaced the softplus(raw)+rate-Lagrangian
parameterisation with projected dual-ascent on a buffer ``lambda_pair``.
There is no learnable Parameter; mutation happens only via
:meth:`LearnablePairWeights.dual_update`. Round 11 Finding 2 (2026-04-28)
wires explicit ``dual_update`` calls in the training loop and removes the
empty optimizer-param-group code path. These tests pin the Round 10/11
contract.

Pins:
  1. lambda_pair is a buffer (no Parameter).
  2. Uniform init: weights ≈ 1.0 everywhere.
  3. warm_start: nonzero λ floor applied for entries above 1.0.
  4. Dual update grows λ for above-target pair losses, shrinks for below-target.
  5. Vector dual update centres on the mean (zero-sum residual ⇒ λ-mass conserved).
  6. Scalar streaming dual update with running-mean target (no-target-loss path).
  7. compute_pair_weighted_primal_loss applies (1+λ)/N weighting.
  8. save/load round-trip preserves lambda_pair.
  9. Schema/module guards reject corrupt snapshots.
 10. Compatibility shim ``compute_pair_weight_rate_penalty`` returns 0 (retired).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.learnable_pair_weights import (
    LearnablePairWeights,
    compute_pair_weight_dual_update,
    compute_pair_weighted_primal_loss,
    compute_pair_weight_rate_penalty,
    load_learnable_pair_weights,
    save_learnable_pair_weights,
    _inverse_softplus,
)


# ── Module basics (Round 10 buffer-only API) ──────────────────────────────


def test_lambda_pair_is_buffer_no_parameters():
    pw = LearnablePairWeights(600)
    # No nn.Parameter — Round 10 dropped the softplus(raw) Parameter.
    assert list(pw.parameters()) == []
    # lambda_pair is a registered buffer with requires_grad=False.
    assert "lambda_pair" in dict(pw.named_buffers())
    assert not pw.lambda_pair.requires_grad
    assert pw.lambda_pair.shape == (600,)
    # Round 11 Finding 2: also assert dual-ascent infra buffers exist.
    assert "running_target_loss" in dict(pw.named_buffers())
    assert "dual_step" in dict(pw.named_buffers())


def test_uniform_init_gives_unit_weights():
    pw = LearnablePairWeights(64)
    w = pw()
    assert torch.allclose(w, torch.ones(64), atol=1e-6)


def test_warm_start_above_one_initialises_lambda_floor():
    """Warm-start values >1 map to λ = warm-1 (zero-floored) per Round 10."""
    target = torch.tensor([0.5, 1.0, 5.0, 2.5, 0.1, 3.3])
    pw = LearnablePairWeights(6, warm_start=target)
    expected_lambda = (target - 1.0).clamp_min(0.0)
    assert torch.allclose(pw.lambda_pair, expected_lambda, atol=1e-6)
    assert torch.allclose(pw(), 1.0 + expected_lambda, atol=1e-6)


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


# ── Dual-ascent semantics ────────────────────────────────────────────────


def test_vector_dual_update_grows_above_target_shrinks_below_target():
    """λ_p ← max(0, λ_p + η * (loss_p − target)) with target = mean(loss)."""
    pw = LearnablePairWeights(3)
    with torch.no_grad():
        pw.lambda_pair.copy_(torch.tensor([0.2, 0.2, 0.2]))
    # losses [0.1, 0.5, 0.9] with explicit target=0.5, eta=0.5
    compute_pair_weight_dual_update(
        pw,
        torch.tensor([0.1, 0.5, 0.9]),
        eta=0.5,
        target_loss=torch.tensor(0.5),
    )
    # λ_0 = max(0, 0.2 + 0.5 * (0.1-0.5)) = 0.0
    # λ_1 = max(0, 0.2 + 0.5 * (0.5-0.5)) = 0.2
    # λ_2 = max(0, 0.2 + 0.5 * (0.9-0.5)) = 0.4
    expected = torch.tensor([0.0, 0.2, 0.4])
    assert torch.allclose(pw.lambdas(), expected, atol=1e-7)


def test_dual_ascent_grows_only_hard_pair_lambda():
    """One above-mean pair eventually has the strictly-largest λ."""
    pw = LearnablePairWeights(10)
    losses = torch.tensor([0.1] * 9 + [1.0])
    for _ in range(100):
        compute_pair_weight_dual_update(pw, losses, eta=0.1)
    lam = pw.lambdas().detach()
    weights = pw().detach()
    assert lam[-1].item() > lam[:-1].max().item()
    assert weights[-1].item() > weights[:-1].max().item()


def test_scalar_streaming_update_running_mean_target():
    """When target_loss=None, scalar pair_idx updates use a running-mean
    so a one-shot easy pair doesn't permanently shift the comparison."""
    pw = LearnablePairWeights(5)
    # Single observation seeds running mean to that value (so residual=0)
    compute_pair_weight_dual_update(
        pw, torch.tensor([0.3]), eta=0.5, pair_idx=2,
    )
    assert pw.lambda_pair[2].item() == pytest.approx(0.0, abs=1e-7)
    # A sequence of HIGH observations on idx 0 should grow lambda[0].
    for _ in range(50):
        compute_pair_weight_dual_update(
            pw, torch.tensor([2.0]), eta=0.05, pair_idx=0,
        )
    assert pw.lambda_pair[0].item() > 0.0
    # Other pairs untouched.
    assert pw.lambda_pair[1].item() == 0.0
    assert pw.lambda_pair[3].item() == 0.0


def test_eta_must_be_positive():
    pw = LearnablePairWeights(4)
    with pytest.raises(ValueError, match="eta must be > 0"):
        compute_pair_weight_dual_update(
            pw, torch.zeros(4), eta=0.0,
        )


def test_dual_update_rejects_nan_pair_loss():
    pw = LearnablePairWeights(3)
    bad = torch.tensor([0.1, float("nan"), 0.1])
    with pytest.raises(ValueError, match="NaN"):
        compute_pair_weight_dual_update(pw, bad, eta=0.1)


# ── Primal loss helper ───────────────────────────────────────────────────


def test_pair_primal_loss_uses_one_plus_lambda_over_n():
    pw = LearnablePairWeights(3)
    with torch.no_grad():
        pw.lambda_pair.copy_(torch.tensor([0.0, 1.0, 3.0]))
    pair_losses = torch.tensor([1.0, 2.0, 3.0])
    primal = compute_pair_weighted_primal_loss(pw, pair_losses)
    # (1*1 + 2*2 + 4*3) / 3 = (1+4+12)/3 = 17/3
    expected = (pw().detach() * pair_losses).sum() / 3.0
    assert primal.item() == pytest.approx(expected.item(), rel=1e-6)


def test_pair_primal_loss_propagates_grad_into_pair_losses_only():
    pw = LearnablePairWeights(4)
    with torch.no_grad():
        pw.lambda_pair.copy_(torch.tensor([0.5, 0.0, 1.0, 0.0]))
    pair_losses = torch.linspace(0.1, 0.4, 4).requires_grad_(True)
    primal = compute_pair_weighted_primal_loss(pw, pair_losses)
    primal.backward()
    assert pair_losses.grad is not None
    assert torch.isfinite(pair_losses.grad).all()


# ── weight_for_pair ──────────────────────────────────────────────────────


def test_weight_for_pair_matches_full_forward():
    target = torch.tensor([0.5, 2.0, 3.0, 1.0])
    pw = LearnablePairWeights(4, warm_start=target)
    full = pw()
    for i in range(4):
        assert pw.weight_for_pair(i).item() == pytest.approx(full[i].item(), rel=1e-6)


def test_weight_for_pair_with_tensor_index():
    pw = LearnablePairWeights(8)
    idx = torch.tensor(3)
    w = pw.weight_for_pair(idx)
    assert torch.allclose(w, torch.tensor(1.0), atol=1e-6)


# ── Save / load round-trip ───────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path):
    target = torch.tensor([0.1, 5.0, 2.5, 1.0, 3.3])
    pw = LearnablePairWeights(5, warm_start=target)
    with torch.no_grad():
        pw.lambda_pair.copy_(torch.tensor([0.2, 0.0, 0.0, 0.4, 0.0]))
    out = tmp_path / "pw.pt"
    save_learnable_pair_weights(pw, out)
    assert out.exists()
    pw2 = load_learnable_pair_weights(out)
    assert pw2.n_pairs == 5
    assert torch.allclose(pw.lambda_pair, pw2.lambda_pair, atol=1e-6)
    assert torch.allclose(pw(), pw2(), atol=1e-6)


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_learnable_pair_weights(tmp_path / "nonexistent.pt")


def test_load_wrong_module_raises(tmp_path: Path):
    bad = {
        "schema_version": 2,
        "module": "some.other.Module",
        "n_pairs": 5,
        "lambda_pair": torch.zeros(5),
        "weights": torch.ones(5),
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
        "lambda_pair": torch.zeros(5),
        "weights": torch.ones(5),
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


# ── Inverse-softplus helper (kept for v1 snapshot back-compat) ───────────


def test_inverse_softplus_round_trip():
    y = torch.tensor([0.1, 1.0, 5.0, 10.0, 25.0])
    raw = _inverse_softplus(y)
    y_back = torch.nn.functional.softplus(raw)
    diff = (y - y_back).abs()
    assert diff.max().item() < 1e-3


# ── Retired-shim contracts (Round 10) ────────────────────────────────────


def test_rate_penalty_is_zero_valued_shim():
    """Round 10 retired the softplus rate Lagrangian; the shim returns
    a zero tensor so legacy imports compile but the term adds no signal
    to the loss. (Round 11 Finding 2 also dropped its call site from
    train_renderer.py — it's now genuinely dead.)"""
    pw = LearnablePairWeights(8, warm_start=torch.full((8,), 3.0))
    pen = compute_pair_weight_rate_penalty(pw, target_sum=8.0, lambda_rate=1.0)
    assert pen.item() == 0.0
    assert pen.requires_grad is False


# ── Round 13 (R12-CC-2) — CUDA integration test ─────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_r13_cc2_cuda_pair_weights_full_chain_on_cuda():
    """End-to-end CUDA chain:
    (LearnablePairWeights → dual_update → compute_pair_weighted_primal_loss
    → backward). Verifies:
      1. ``lambda_pair`` buffer stays on CUDA after ``.to('cuda')``.
      2. ``dual_update`` keeps ``lambda_pair`` on CUDA.
      3. ``compute_pair_weighted_primal_loss`` on CUDA pair_losses
         produces a CUDA loss tensor.
      4. ``loss.backward()`` propagates to CUDA pair_losses (the
         multipliers are detached buffers so they don't get grad).
    This is the smallest end-to-end test that catches the device-mismatch
    bug class (covered for the bit-quant module by the C-2 fixes).
    """
    n_pairs = 8
    pw = LearnablePairWeights(n_pairs).to("cuda")
    assert pw.lambda_pair.device.type == "cuda", (
        f".to('cuda') must move lambda_pair to CUDA; got {pw.lambda_pair.device}"
    )

    # Drive a single dual update and confirm the buffer stays on CUDA.
    distortion = torch.linspace(0.1, 0.9, n_pairs, device="cuda")
    pw.dual_update(distortion, eta=0.5)
    assert pw.lambda_pair.device.type == "cuda", (
        f"dual_update must leave lambda_pair on CUDA; got {pw.lambda_pair.device}"
    )
    assert pw.lambda_pair.sum().item() > 0.0, (
        "expected at least one above-mean pair to have non-zero λ"
    )

    # Forward a CUDA pair_losses tensor through the primal loss.
    pair_losses = torch.linspace(
        0.1, 0.9, n_pairs, device="cuda", requires_grad=True,
    )
    loss = compute_pair_weighted_primal_loss(pw, pair_losses)
    assert loss.device.type == "cuda"

    # Backward must produce a CUDA gradient on pair_losses (no
    # device-mismatch crash). The multipliers are detached buffers so
    # they receive no grad.
    loss.backward()
    assert pair_losses.grad is not None
    assert pair_losses.grad.device.type == "cuda"
    assert torch.isfinite(pair_losses.grad).all()
