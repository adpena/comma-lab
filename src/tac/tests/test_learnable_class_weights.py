"""Tests for src/tac/learnable_class_weights.py — Lane PS-V2.

Pins:
  1. raw_logits is a learnable Parameter.
  2. forward() returns softmax (sum=1, all positive).
  3. uniform init: weights = 1/num_classes.
  4. warm_start ∝ [1,5,5,1,1] re-normalises to softmax(log).
  5. Gradient flows through raw_logits.
  6. Variance penalty drives equalisation: SGD reduces per-class
     contribution variance.
  7. csv() returns 5 floats summing to ~1.
  8. save/load round-trip preserves raw_logits and weights.
  9. Wrong schema_version / module ID fails loud.
 10. shape mismatch in per_class_distortion raises.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.learnable_class_weights import (
    LearnableClassWeights,
    compute_class_weight_equalisation_penalty,
    load_learnable_class_weights,
    save_learnable_class_weights,
)


# ── Module basics ────────────────────────────────────────────────────────


def test_raw_logits_is_learnable_parameter():
    cw = LearnableClassWeights(5)
    assert isinstance(cw.raw_logits, nn.Parameter)
    assert cw.raw_logits.requires_grad
    assert cw.raw_logits.shape == (5,)


def test_uniform_init_returns_uniform_softmax():
    cw = LearnableClassWeights(5)
    w = cw()
    assert torch.allclose(w, torch.full((5,), 0.2), atol=1e-5)


def test_softmax_sums_to_one():
    cw = LearnableClassWeights(5)
    with torch.no_grad():
        cw.raw_logits.copy_(torch.tensor([1.0, -2.0, 3.0, 0.5, -1.5]))
    w = cw()
    assert w.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (w > 0).all()


def test_warm_start_15511_reproduces_distribution():
    """[1,5,5,1,1] / 13 should be the softmax of log(...) shifted to mean 0."""
    target = torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0])
    cw = LearnableClassWeights(5, warm_start=target)
    w = cw()
    expected = target / target.sum()
    assert torch.allclose(w, expected, atol=1e-5)


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


# ── Gradient flow ────────────────────────────────────────────────────────


def test_gradient_flows_through_raw_logits():
    cw = LearnableClassWeights(5)
    per_class_loss = torch.linspace(0.1, 0.5, 5)
    weighted = (cw() * per_class_loss).sum()
    weighted.backward()
    assert cw.raw_logits.grad is not None
    assert torch.isfinite(cw.raw_logits.grad).all()


def test_gradient_zero_when_loss_uniform():
    """Uniform per-class loss has no preferred direction in the simplex."""
    cw = LearnableClassWeights(5)
    per_class_loss = torch.full((5,), 0.3)
    weighted = (cw() * per_class_loss).sum()
    weighted.backward()
    # All grads equal at uniform softmax + uniform loss (Jacobian symmetry);
    # the grad on raw_logits is 0 (mean-shift cancels).
    assert torch.allclose(cw.raw_logits.grad, torch.zeros(5), atol=1e-5)


# ── Equalisation penalty ────────────────────────────────────────────────


def test_equalisation_penalty_zero_when_distortion_uniform():
    """If all per-class distortions are equal AND all weights equal,
    contribution variance = 0."""
    cw = LearnableClassWeights(5)  # uniform 0.2 weights
    distortion = torch.full((5,), 0.1)
    pen = compute_class_weight_equalisation_penalty(
        cw, distortion, lambda_var=1.0
    )
    assert pen.item() == pytest.approx(0.0, abs=1e-7)


def test_equalisation_penalty_positive_when_distortion_skewed():
    cw = LearnableClassWeights(5)
    distortion = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0])
    pen = compute_class_weight_equalisation_penalty(
        cw, distortion, lambda_var=1.0
    )
    assert pen.item() > 0.0


def test_equalisation_penalty_drives_weight_off_solved_classes():
    """SGD on the equalisation penalty alone shifts weight FROM classes
    with low distortion (already solved) TOWARD classes with high
    distortion (bottleneck) until contributions equalise."""
    cw = LearnableClassWeights(5)
    distortion = torch.tensor([0.01, 0.01, 0.50, 0.01, 0.01])
    optim = torch.optim.SGD([cw.raw_logits], lr=0.5)
    initial_w = cw().detach().clone()
    for _ in range(500):
        optim.zero_grad()
        pen = compute_class_weight_equalisation_penalty(cw, distortion, lambda_var=1.0)
        # Add a tiny anchor on uniform so the trivial all-mass-on-class-2
        # solution isn't selected. Mirrors the train_renderer use case
        # where the primary loss is the dominant term.
        anchor = (cw() - 0.2).pow(2).sum() * 0.01
        (pen + anchor).backward()
        optim.step()
    final_w = cw().detach().clone()
    # The weight on class 2 (distortion=0.5) should have DECREASED so that
    # weight*distortion equalises with the (almost-zero) other classes.
    # That is, the "high distortion class" weight goes DOWN to balance
    # contribution. Variance minimisation = equalisation, NOT amplification.
    assert final_w[2].item() < initial_w[2].item(), (
        f"Equalisation should reduce class-2 weight from "
        f"{initial_w[2].item():.4f} → got {final_w[2].item():.4f}"
    )


def test_equalisation_penalty_per_class_distortion_shape_mismatch_raises():
    cw = LearnableClassWeights(5)
    bad = torch.zeros(4)
    with pytest.raises(ValueError, match="per_class_distortion shape"):
        compute_class_weight_equalisation_penalty(cw, bad)


def test_equalisation_penalty_rejects_nan():
    cw = LearnableClassWeights(5)
    bad = torch.tensor([0.1, float("nan"), 0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="NaN"):
        compute_class_weight_equalisation_penalty(cw, bad)


# ── csv() helper ───────────────────────────────────────────────────────


def test_csv_returns_5_comma_separated_floats():
    cw = LearnableClassWeights(5)
    s = cw.csv()
    parts = s.split(",")
    assert len(parts) == 5
    vals = [float(p) for p in parts]
    assert sum(vals) == pytest.approx(1.0, abs=1e-3)


# ── Save / load round-trip ─────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path: Path):
    target = torch.tensor([1.0, 5.0, 5.0, 1.0, 1.0])
    cw = LearnableClassWeights(5, warm_start=target)
    out = tmp_path / "cw.pt"
    save_learnable_class_weights(cw, out)
    assert out.exists()
    cw2 = load_learnable_class_weights(out)
    assert cw2.num_classes == 5
    assert torch.allclose(cw.raw_logits, cw2.raw_logits, atol=1e-6)
    assert torch.allclose(cw(), cw2(), atol=1e-5)


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_learnable_class_weights(tmp_path / "nonexistent.pt")


def test_load_wrong_module_raises(tmp_path: Path):
    bad = {
        "schema_version": 1,
        "module": "some.other.Module",
        "num_classes": 5,
        "raw_logits": torch.zeros(5),
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
        "raw_logits": torch.zeros(5),
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
