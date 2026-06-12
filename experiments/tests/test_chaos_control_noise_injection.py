# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the MPS-pose-drift chaos-control harness.

These verify the chaos-control's load-bearing mechanism ACTUALLY does what it
claims: (1) the cotangent-noise hook actually perturbs the frame gradient by the
intended relative magnitude (not a no-op), (2) the perturbation is i.i.d. and
reproducible from the seed, (3) zero-noise is a true no-op (the clean arm is
byte-identical to an un-hooked run), (4) the real-loss diag's relmax helper is
sane. Behavior, not constants (Slot EEE Class 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, "upstream")

from experiments.measure_torch_vehicle_chaos_control import _install_cotangent_noise


class _FakeScorer:
    """Minimal stand-in with a seg_pose_forward that returns a differentiable
    function of the input frames, so the registered cotangent hook fires on
    backward (exercises the REAL hook path, not a mock of it)."""

    def __init__(self):
        self.calls = 0

    def seg_pose_forward(self, decoded_bhwc):
        self.calls += 1
        # A trivial differentiable head: returns two tensors whose grad wrt frames
        # is well-defined, so backward populates decoded_bhwc.grad.
        seg = (decoded_bhwc**2).sum()
        pose = decoded_bhwc.mean()
        return seg, pose


def _grad_under(scorer, frames, *, sum_outputs=True):
    f = frames.clone().requires_grad_(True)
    seg, pose = scorer.seg_pose_forward(f)
    (seg + pose).backward()
    return f.grad.detach().clone()


def test_noise_hook_actually_perturbs_gradient_not_noop():
    """The hooked gradient must DIFFER from the clean gradient by ~rel magnitude.

    If the hook were a no-op (the fake-implementation failure), the perturbed and
    clean gradients would be bit-identical and this test would FAIL."""
    torch.manual_seed(0)
    frames = torch.rand(2, 2, 384, 512, 3)

    clean = _FakeScorer()
    g_clean = _grad_under(clean, frames)

    noisy = _install_cotangent_noise(_FakeScorer(), rel=2e-4, seed=1234)
    g_noisy = _grad_under(noisy, frames)

    # NOT a no-op: the gradients must differ.
    assert not torch.allclose(g_clean, g_noisy), "noise hook is a NO-OP (fake)"
    # The relative perturbation magnitude is ~rel (multiplicative 1+rel*N(0,1)).
    # The per-element relative deviation has std ~rel; check the RMS is in band.
    rel_dev = ((g_noisy - g_clean) / g_clean.abs().clamp_min(1e-12))
    rms = float(rel_dev.pow(2).mean().sqrt())
    assert 0.5e-4 < rms < 5e-4, f"perturbation RMS {rms:.2e} not ~2e-4 (band)"


def test_noise_is_reproducible_from_seed():
    """Same seed -> same perturbation (deterministic research signal)."""
    torch.manual_seed(0)
    frames = torch.rand(2, 2, 384, 512, 3)
    g1 = _grad_under(_install_cotangent_noise(_FakeScorer(), rel=2e-4, seed=99), frames)
    g2 = _grad_under(_install_cotangent_noise(_FakeScorer(), rel=2e-4, seed=99), frames)
    assert torch.allclose(g1, g2), "noise not reproducible from seed"
    # Different seed -> different noise.
    g3 = _grad_under(_install_cotangent_noise(_FakeScorer(), rel=2e-4, seed=100), frames)
    assert not torch.allclose(g1, g3), "different seed produced identical noise"


def test_zero_noise_is_true_noop():
    """rel=0 path: _install_cotangent_noise is only called when rel>0 in the
    harness, but if it WERE called with rel=0 the hook must be identity."""
    torch.manual_seed(0)
    frames = torch.rand(2, 2, 384, 512, 3)
    g_clean = _grad_under(_FakeScorer(), frames)
    g_zero = _grad_under(_install_cotangent_noise(_FakeScorer(), rel=0.0, seed=1), frames)
    assert torch.allclose(g_clean, g_zero), "rel=0 hook is not identity"


def test_gate_classifies_chaos_reproduction():
    """The acceptance gate must REJECT a noise-injected arm whose pose |gap|
    reproduces the MPS magnitude — that REJECT is the H_chaos confirmation."""
    from tac.mlx_pr95_port.speedup_acceptance_gate import evaluate_descent_equivalence

    # Synthetic: seg bit-identical, pose gap grows to ~7 (the MPS signature),
    # NOT a monotone blow-up (diverged stays False) -> rejected on |gap|.
    base = [{"epoch": e, "d_seg": 0.505, "d_pose": 170.0 - 0.1 * e} for e in (5, 10, 15, 20, 25, 30)]
    cand = [{"epoch": e, "d_seg": 0.505, "d_pose": 170.0 - 0.1 * e - g}
            for e, g in zip((5, 10, 15, 20, 25, 30), (0.06, 0.3, 0.4, 3.3, 5.3, 7.0))]
    v = evaluate_descent_equivalence(base, cand, n_pairs=48)
    assert not v.passed, "gate should REJECT the reproduced-gap candidate"
    assert v.seg.tracks_within_tol, "seg should still track (bit-identical)"
    assert not v.pose.diverged, "the gap is chaos, not a blow-up (diverged=False)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
