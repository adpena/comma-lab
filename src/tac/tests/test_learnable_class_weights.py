"""Tests for src/tac/learnable_class_weights.py — Lane PS-V2.

Round 10 (2026-04-27) replaced the softmax(raw_logits)+variance penalty
parameterisation with projected dual-ascent on a buffer ``lambda_class``
multiplied by frozen ``base_weights``. Round 11 Finding 2 (2026-04-28)
wires the explicit ``dual_update`` calls in train_renderer.py and threads
``module.weights()`` into the SegNet loss path. These tests pin the
Round 10/11 contract.

Pins:
  1. lambda_class is a buffer (no Parameter).
  2. base_weights is a buffer (frozen warm-start distribution).
  3. Uniform init: weights = 1/num_classes everywhere; sum = 1.
  4. Warm-start [1,5,5,1,1] normalises to base_weights = ws/sum(ws).
  5. Dual update grows λ for above-target classes, shrinks for below.
  6. Bottleneck class with persistently high distortion eventually has
     the largest λ AND the largest emitted weight.
  7. csv() renders 5-comma values (legacy CSV emitter).
  8. save/load round-trip preserves base_weights + lambda_class.
  9. Schema/module guards reject corrupt snapshots.
 10. Compatibility shim ``compute_class_weight_equalisation_penalty`` returns 0.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.learnable_class_weights import (
    LearnableClassWeights,
    compute_class_weight_dual_update,
    compute_class_weight_equalisation_penalty,
    compute_class_weighted_primal_loss,
    load_learnable_class_weights,
    save_learnable_class_weights,
)


# ── Module basics (Round 10 buffer-only API) ─────────────────────────────


def test_lambda_class_is_buffer_no_parameters():
    cw = LearnableClassWeights(5)
    # No nn.Parameter — Round 10 dropped the softmax(raw) Parameter.
    assert list(cw.parameters()) == []
    bufs = dict(cw.named_buffers())
    assert "base_weights" in bufs
    assert "lambda_class" in bufs
    assert "running_target_distortion" in bufs
    assert "dual_step" in bufs
    assert not cw.lambda_class.requires_grad
    assert not cw.base_weights.requires_grad


def test_uniform_init_returns_uniform_base_weights():
    cw = LearnableClassWeights(5)
    w = cw()
    assert torch.allclose(w, torch.full((5,), 0.2), atol=1e-6)
    assert w.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_warm_start_15511_reproduces_distribution():
    """[1,5,5,1,1] / 13 normalises to the base_weights vector."""
    target = torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0])
    cw = LearnableClassWeights(5, warm_start=target)
    w = cw()
    expected = target / target.sum()
    assert torch.allclose(w, expected, atol=1e-6)


def test_warm_start_shape_mismatch_raises():
    bad = torch.zeros(4)
    with pytest.raises(ValueError, match="warm_start shape"):
        LearnableClassWeights(5, warm_start=bad)


def test_warm_start_negative_raises():
    bad = torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        LearnableClassWeights(5, warm_start=bad)


def test_warm_start_wrong_type_raises():
    with pytest.raises(TypeError, match="warm_start must be"):
        LearnableClassWeights(5, warm_start=[1, 1, 1, 1, 1])


def test_num_classes_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        LearnableClassWeights(0)


# ── Dual-ascent semantics ────────────────────────────────────────────────


def test_dual_update_direction_matches_distortion_residual():
    """λ_c ← max(0, λ_c + η * (distortion_c − target))."""
    cw = LearnableClassWeights(3)
    with torch.no_grad():
        cw.lambda_class.copy_(torch.tensor([0.3, 0.3, 0.3]))
    compute_class_weight_dual_update(
        cw,
        torch.tensor([0.1, 0.5, 0.9]),
        target=torch.tensor(0.5),
        eta=0.5,
    )
    # λ_0 = max(0, 0.3 + 0.5 * (0.1-0.5)) = 0.1
    # λ_1 = max(0, 0.3 + 0.5 * (0.5-0.5)) = 0.3
    # λ_2 = max(0, 0.3 + 0.5 * (0.9-0.5)) = 0.5
    expected = torch.tensor([0.1, 0.3, 0.5])
    assert torch.allclose(cw.lambdas(), expected, atol=1e-7)


def test_class_dual_grows_only_worst_class_lambda():
    """A persistently above-mean class should accumulate the largest λ."""
    cw = LearnableClassWeights(5)
    distortion = torch.tensor([0.5, 0.01, 0.01, 0.01, 0.01])
    for _ in range(100):
        compute_class_weight_dual_update(cw, distortion, eta=0.1)
    lam = cw.lambdas().detach()
    weights = cw().detach()
    assert lam[0].item() > lam[1:].max().item()
    assert weights[0].item() > weights[1:].max().item()


def test_eta_must_be_positive():
    cw = LearnableClassWeights(5)
    with pytest.raises(ValueError, match="eta must be > 0"):
        compute_class_weight_dual_update(cw, torch.zeros(5), eta=0.0)


def test_dual_update_rejects_nan_distortion():
    cw = LearnableClassWeights(5)
    bad = torch.tensor([0.1, float("nan"), 0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="NaN"):
        compute_class_weight_dual_update(cw, bad, eta=0.1)


def test_dual_update_distortion_shape_mismatch_raises():
    cw = LearnableClassWeights(5)
    bad = torch.zeros(4)
    with pytest.raises(ValueError, match="per_class_distortion shape"):
        compute_class_weight_dual_update(cw, bad, eta=0.1)


# ── Primal loss helper ───────────────────────────────────────────────────


def test_class_primal_loss_uses_one_plus_lambda():
    """primal = sum_c (base_c * (1 + λ_c) * loss_c)."""
    cw = LearnableClassWeights(3, warm_start=torch.tensor([1.0, 2.0, 3.0]))
    with torch.no_grad():
        cw.lambda_class.copy_(torch.tensor([0.0, 1.0, 0.0]))
    per_class_loss = torch.tensor([1.0, 2.0, 3.0])
    primal = compute_class_weighted_primal_loss(cw, per_class_loss)
    expected = (cw().detach() * per_class_loss).sum()
    assert primal.item() == pytest.approx(expected.item(), rel=1e-6)


def test_class_primal_loss_propagates_grad_into_per_class_loss_only():
    cw = LearnableClassWeights(3, warm_start=torch.tensor([1.0, 2.0, 3.0]))
    per_class_loss = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    primal = compute_class_weighted_primal_loss(cw, per_class_loss)
    primal.backward()
    assert per_class_loss.grad is not None
    assert torch.isfinite(per_class_loss.grad).all()
    # Round 29 advisory fix: magnitude + sign anchor (matches Round 26 pattern).
    # Trivial-gradient-today is fine, but if compute_class_weighted_primal_loss
    # is ever refactored with a non-trivial path, this test would silently
    # pass a sign-inverted implementation. The grad is `multiplier.detach()`
    # which is `1 + lambda_class >= 1 > 0` per softmax warm-start; pin both.
    assert per_class_loss.grad.abs().max().item() > 0, (
        "per-class loss gradient is identically zero — primal-loss path not flowing"
    )
    assert (per_class_loss.grad > 0).all(), (
        "per-class loss gradient must be non-negative (multiplier = 1+lambda_class >= 1)"
    )


# ── csv() helper ─────────────────────────────────────────────────────────


def test_csv_returns_5_comma_separated_floats():
    cw = LearnableClassWeights(5)
    s = cw.csv()
    parts = s.split(",")
    assert len(parts) == 5
    vals = [float(p) for p in parts]
    # Uniform init at λ=0 ⇒ weights = 0.2 each ⇒ sum = 1.0.
    assert sum(vals) == pytest.approx(1.0, abs=1e-3)


def test_class_weights_csv_sums_to_one_after_dual_update():
    """Round 13 (I-1): csv() must L1-renormalise to maintain the
    long-standing 'sums to 1' invariant after dual updates have driven
    λ_c above zero. Pre-fix, csv() returned base * (1 + λ) which sums
    to 1 + sum(base * λ) > 1 — breaking downstream consumers that
    read the CSV as a probability distribution."""
    cw = LearnableClassWeights(5)
    # Drive lambda above zero on a single bottleneck class.
    distortion = torch.tensor([0.5, 0.01, 0.01, 0.01, 0.01])
    for _ in range(50):
        compute_class_weight_dual_update(cw, distortion, eta=0.5)
    # Confirm the dual update actually moved lambda — otherwise the test
    # would trivially pass (λ=0 ⇒ weights = base which already sums to 1).
    assert cw.lambdas().sum().item() > 0.0
    # Now csv() must still sum to 1 within float rounding.
    parts = cw.csv().split(",")
    vals = [float(p) for p in parts]
    assert sum(vals) == pytest.approx(1.0, abs=1e-5), (
        f"csv() must L1-renormalise after dual update; got sum={sum(vals)}"
    )
    # Sanity: the raw weights() (UN-renormalised) DO exceed 1 after the
    # dual update; this proves the csv() renormalisation is load-bearing.
    raw = cw.weights().detach()
    assert raw.sum().item() > 1.0


def test_class_weights_csv_non_negative_after_dual_update():
    """The renormalisation must keep all entries >= 0 (probabilities).
    Combined with the sum=1 invariant this is the standard simplex
    contract."""
    cw = LearnableClassWeights(5, warm_start=torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0]))
    distortion = torch.tensor([0.05, 0.5, 0.05, 0.05, 0.05])
    for _ in range(20):
        compute_class_weight_dual_update(cw, distortion, eta=0.3)
    parts = cw.csv().split(",")
    vals = [float(p) for p in parts]
    for v in vals:
        assert v >= 0.0, f"csv() entry must be non-negative; got {v}"


# ── Save / load round-trip ───────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path):
    target = torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0])
    cw = LearnableClassWeights(5, warm_start=target)
    with torch.no_grad():
        cw.lambda_class.copy_(torch.tensor([0.2, 0.0, 0.0, 0.4, 0.0]))
    out = tmp_path / "cw.pt"
    save_learnable_class_weights(cw, out)
    assert out.exists()
    cw2 = load_learnable_class_weights(out)
    assert cw2.num_classes == 5
    assert torch.allclose(cw.base_weights, cw2.base_weights, atol=1e-6)
    assert torch.allclose(cw.lambda_class, cw2.lambda_class, atol=1e-6)
    assert torch.allclose(cw(), cw2(), atol=1e-6)


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_learnable_class_weights(tmp_path / "nonexistent.pt")


def test_load_wrong_module_raises(tmp_path: Path):
    bad = {
        "schema_version": 2,
        "module": "some.other.Module",
        "num_classes": 5,
        "base_weights": torch.full((5,), 0.2),
        "lambda_class": torch.zeros(5),
        "weights": torch.full((5,), 0.2),
    }
    p = tmp_path / "bad.pt"
    torch.save(bad, p)
    with pytest.raises(ValueError, match="module="):
        load_learnable_class_weights(p)


def test_load_wrong_schema_version_raises(tmp_path: Path):
    bad = {
        "schema_version": 99,
        "module": "tac.learnable_class_weights.LearnableClassWeights",
        "num_classes": 5,
        "base_weights": torch.full((5,), 0.2),
        "lambda_class": torch.zeros(5),
        "weights": torch.full((5,), 0.2),
    }
    p = tmp_path / "bad.pt"
    torch.save(bad, p)
    with pytest.raises(ValueError, match="schema_version"):
        load_learnable_class_weights(p)


def test_load_non_dict_raises(tmp_path: Path):
    p = tmp_path / "bad.pt"
    torch.save(torch.zeros(5), p)
    with pytest.raises(TypeError, match="snapshot"):
        load_learnable_class_weights(p)


# ── Retired-shim contracts (Round 10) ────────────────────────────────────


def test_equalisation_penalty_is_zero_valued_shim():
    """Round 10 retired the variance-equalisation penalty; the shim
    validates inputs and returns a zero tensor."""
    cw = LearnableClassWeights(5)
    distortion = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0])
    pen = compute_class_weight_equalisation_penalty(cw, distortion, lambda_var=1.0)
    assert pen.item() == 0.0


def test_equalisation_penalty_validates_shape():
    cw = LearnableClassWeights(5)
    with pytest.raises(ValueError, match="per_class_distortion shape"):
        compute_class_weight_equalisation_penalty(cw, torch.zeros(4))


def test_equalisation_penalty_rejects_nan():
    cw = LearnableClassWeights(5)
    bad = torch.tensor([0.1, float("nan"), 0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="NaN"):
        compute_class_weight_equalisation_penalty(cw, bad)
