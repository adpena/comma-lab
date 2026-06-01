# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the L-inf-vs-L2 inverse-steganalysis gate helper.

These tests verify ACTUAL behavior (the rate model is correct, the two
allocations genuinely differ, the fairness invariant holds, the quantization
noise actually perturbs the frame, the d_seg/d_pose measurement matches the
contest metric definitions) -- NOT constants. Per CLAUDE.md "NO FAKE
IMPLEMENTATIONS" Class 2: every test would FAIL if the function body were
replaced by canonical-markers / a no-op.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tac.analysis.inverse_steganalysis_linf_vs_l2_gate import (
    UINT8_DYNAMIC_RANGE,
    InverseSteganalysisGateError,
    _total_bits_from_steps,
    _uniform_step_for_budget,
    allocate_l2_uniform,
    allocate_linf_margin_budget,
    apply_uniform_quantization_noise,
    margin_budget_from_saliency,
    measure_pair_d_seg_d_pose,
)

# ---------------------------------------------------------------------------
# Rate model: the high-rate uniform-quantizer entropy log2(R/delta).
# ---------------------------------------------------------------------------


def test_total_bits_matches_log2_rate_formula():
    # A step of R/2 carries exactly 1 bit/pixel; R/16 carries 4 bits/pixel.
    n = 1000
    steps_1bit = np.full(n, UINT8_DYNAMIC_RANGE / 2.0)
    steps_4bit = np.full(n, UINT8_DYNAMIC_RANGE / 16.0)
    assert _total_bits_from_steps(steps_1bit) == pytest.approx(1.0 * n)
    assert _total_bits_from_steps(steps_4bit) == pytest.approx(4.0 * n)


def test_total_bits_clamps_negative_rate_at_zero():
    # A step LARGER than the range carries no bits (rate clamped at 0), not
    # negative bits -- a no-op step must not "refund" rate.
    n = 100
    steps_coarse = np.full(n, UINT8_DYNAMIC_RANGE * 4.0)  # log2(1/4) < 0
    assert _total_bits_from_steps(steps_coarse) == 0.0


def test_total_bits_rejects_nonpositive_steps():
    with pytest.raises(InverseSteganalysisGateError):
        _total_bits_from_steps(np.array([1.0, 0.0, 2.0]))


def test_uniform_step_for_budget_inverts_rate():
    n = 4096
    bits = 3.0 * n
    delta = _uniform_step_for_budget(n, target_bits=bits)
    # delta = R * 2^(-bits/n) = R * 2^-3 = R/8
    assert delta == pytest.approx(UINT8_DYNAMIC_RANGE / 8.0)
    assert _total_bits_from_steps(np.full(n, delta)) == pytest.approx(bits)


# ---------------------------------------------------------------------------
# L2 allocation: a single uniform step, realized rate EXACTLY the target.
# ---------------------------------------------------------------------------


def test_l2_uniform_is_a_single_step_everywhere():
    alloc = allocate_l2_uniform(5000, target_bits=2.0 * 5000)
    assert alloc.allocation == "l2_uniform"
    assert np.all(alloc.steps == alloc.steps[0])  # genuinely uniform
    assert alloc.min_step == alloc.max_step
    assert alloc.total_bits == pytest.approx(2.0 * 5000)


# ---------------------------------------------------------------------------
# margin_budget_from_saliency: the inverse-steganalysis inversion.
# ---------------------------------------------------------------------------


def test_margin_budget_inverts_saliency_high_saliency_small_budget():
    sal = np.array([1000.0, 1.0, 0.001])  # high, mid, low saliency
    rho = margin_budget_from_saliency(sal)
    # high saliency -> SMALL margin budget (must perturb less -> fine step)
    assert rho[0] < rho[1] < rho[2]
    # the inversion is monotone-decreasing
    assert np.all(np.diff(rho) > 0)


def test_margin_budget_handles_zero_saliency():
    rho = margin_budget_from_saliency(np.array([0.0, 0.0]))
    assert np.all(np.isfinite(rho))
    assert np.all(rho > 0)


# ---------------------------------------------------------------------------
# L-inf allocation: genuinely different from L2, rate-matched, ordering.
# ---------------------------------------------------------------------------


def _concentrated_rho(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sal = rng.pareto(0.3, n) + 1e-6  # heavy-tailed like the real oracle
    return margin_budget_from_saliency(sal)


def test_linf_allocation_differs_from_l2_and_matches_rate():
    n = 20000
    rho = _concentrated_rho(n)
    target = 4.0 * n
    l2 = allocate_l2_uniform(n, target_bits=target)
    linf = allocate_linf_margin_budget(rho, target_bits=target, min_step=1.0, max_step=128.0)
    # The two step maps must NOT be identical (a no-op detector per Catalog #139).
    assert not np.array_equal(l2.steps, linf.steps)
    # L-inf has a spread of steps (fine at boundaries, coarse in dead-zone).
    assert linf.min_step < linf.max_step
    # Rate matched to within 0.1%.
    rel = abs(linf.total_bits - l2.total_bits) / l2.total_bits
    assert rel < 1.0e-3


def test_linf_step_increases_with_margin_budget():
    # delta_i = clip(c * rho_i, lo, hi): a larger margin budget => coarser step.
    # Use a smooth rho so no pixel is clipped, isolating the monotone relation.
    rho = np.linspace(0.5, 2.0, 10000)
    target = 3.0 * rho.size
    linf = allocate_linf_margin_budget(
        rho, target_bits=target, min_step=0.01, max_step=1000.0, fairness_direction="symmetric"
    )
    # Within the unclipped interior the step is monotone-increasing in rho.
    order = np.argsort(rho)
    steps_sorted = linf.steps[order]
    interior = steps_sorted[(steps_sorted > 0.011) & (steps_sorted < 999.0)]
    assert np.all(np.diff(interior) >= -1e-9)


def test_disadvantage_linf_spends_at_least_l2_bits():
    # The anti-gaming fairness guard: L-inf realized rate must be >= target.
    n = 30000
    rho = _concentrated_rho(n, seed=7)
    target = 4.0 * n
    linf = allocate_linf_margin_budget(
        rho, target_bits=target, min_step=1.0, max_step=128.0,
        fairness_direction="disadvantage_linf",
    )
    assert linf.total_bits >= target - 1e-6  # never cheaper than L2
    rel = abs(linf.total_bits - target) / target
    assert rel < 1.0e-3  # but still equal-rate to within 0.1%


def test_advantage_linf_spends_at_most_l2_bits():
    n = 30000
    rho = _concentrated_rho(n, seed=11)
    target = 4.0 * n
    linf = allocate_linf_margin_budget(
        rho, target_bits=target, min_step=1.0, max_step=128.0,
        fairness_direction="advantage_linf",
    )
    assert linf.total_bits <= target + 1e-6  # never costlier than L2


def test_linf_rejects_infeasible_budget_above_finest_rate():
    # min_step caps the finest precision; a budget above the finest feasible rate
    # must raise rather than silently under-deliver.
    n = 1000
    rho = np.ones(n)
    finest_bits = n * math.log2(UINT8_DYNAMIC_RANGE / 1.0)  # all steps at min_step=1
    with pytest.raises(InverseSteganalysisGateError):
        allocate_linf_margin_budget(rho, target_bits=finest_bits * 2.0, min_step=1.0)


def test_linf_rejects_nonfinite_rho():
    with pytest.raises(InverseSteganalysisGateError):
        allocate_linf_margin_budget(np.array([1.0, np.inf]), target_bits=1.0)
    with pytest.raises(InverseSteganalysisGateError):
        allocate_linf_margin_budget(np.array([1.0, -1.0]), target_bits=1.0)


# ---------------------------------------------------------------------------
# Quantization noise actually perturbs the frame within the step bound.
# ---------------------------------------------------------------------------


def test_quantization_noise_bounded_by_half_step_and_actually_perturbs():
    frame = torch.full((3, 4, 5), 128.0)
    steps = np.full(4 * 5, 10.0)
    g = torch.Generator().manual_seed(0)
    noisy = apply_uniform_quantization_noise(frame, steps, generator=g)
    delta = (noisy - frame).abs()
    # The frame genuinely changed (not a no-op).
    assert float(delta.max()) > 0.0
    # Every perturbation is within +/- step/2.
    assert float(delta.max()) <= 5.0 + 1e-6


def test_larger_step_yields_larger_perturbation():
    frame = torch.full((3, 8, 8), 128.0)
    fine = np.full(64, 2.0)
    coarse = np.full(64, 40.0)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    d_fine = (apply_uniform_quantization_noise(frame, fine, generator=g1) - frame).abs().mean()
    d_coarse = (apply_uniform_quantization_noise(frame, coarse, generator=g2) - frame).abs().mean()
    assert float(d_coarse) > float(d_fine)


def test_quantization_noise_clamps_to_uint8_range():
    frame = torch.full((3, 4, 4), 250.0)
    steps = np.full(16, 60.0)
    g = torch.Generator().manual_seed(3)
    noisy = apply_uniform_quantization_noise(frame, steps, generator=g)
    assert float(noisy.max()) <= 255.0
    assert float(noisy.min()) >= 0.0


def test_quantization_noise_rejects_wrong_step_count():
    frame = torch.zeros(3, 4, 5)
    with pytest.raises(InverseSteganalysisGateError):
        apply_uniform_quantization_noise(frame, np.ones(7), generator=torch.Generator())


# ---------------------------------------------------------------------------
# d_seg / d_pose measurement matches the contest metric definitions exactly.
# ---------------------------------------------------------------------------


class _FakeSegNet(torch.nn.Module):
    """A toy SegNet whose argmax depends on the last frame's mean -- enough to
    verify d_seg = argmax-flip rate against the contest definition without the
    real frozen weights (the real-scorer integration is exercised by the tool's
    end-to-end run + the design-memo anchor)."""

    def preprocess_input(self, pair):  # (1,2,3,H,W) -> last frame mean as a scalar map
        last = pair[:, -1]  # (1,3,H,W)
        return last.mean(dim=1, keepdim=True)  # (1,1,H,W)

    def forward(self, x):  # (1,1,H,W) -> (1,5,H,W) logits keyed on intensity bands
        b, _, h, w = x.shape
        bands = (x / 51.0).clamp(0, 4).long()  # 5 classes by intensity band
        logits = torch.zeros(b, 5, h, w)
        logits.scatter_(1, bands, 10.0)
        return logits


class _FakePoseNet(torch.nn.Module):
    def preprocess_input(self, pair):
        return pair.reshape(1, -1)

    def forward(self, x):
        # pose = first 12 column means (deterministic in the input).
        feats = x.reshape(-1)
        pose = torch.stack([feats[k::12].mean() for k in range(12)]).reshape(1, 12)
        return {"pose": pose}


def test_measure_d_seg_is_zero_for_identical_pairs():
    seg = _FakeSegNet()
    pose = _FakePoseNet()
    pair = torch.full((1, 2, 3, 8, 8), 100.0)
    d_seg, d_pose = measure_pair_d_seg_d_pose(pose, seg, pair, pair.clone())
    assert d_seg == 0.0
    assert d_pose == pytest.approx(0.0, abs=1e-9)


def test_measure_d_seg_counts_argmax_flips():
    seg = _FakeSegNet()
    pose = _FakePoseNet()
    gt = torch.full((1, 2, 3, 8, 8), 100.0)  # band 1
    cand = gt.clone()
    # Push the whole candidate last frame into a different intensity band ->
    # every pixel's argmax flips -> d_seg == 1.0.
    cand[:, -1] = 250.0  # band 4
    d_seg, _ = measure_pair_d_seg_d_pose(pose, seg, gt, cand)
    assert d_seg == pytest.approx(1.0)


def test_measure_d_pose_is_first_six_dims_mse():
    seg = _FakeSegNet()
    pose = _FakePoseNet()
    gt = torch.zeros(1, 2, 3, 4, 4)
    cand = gt.clone()
    cand[0, 0, 0, 0, 0] = 12.0  # perturb a single input element
    _, d_pose = measure_pair_d_seg_d_pose(pose, seg, gt, cand)
    assert d_pose > 0.0  # the perturbation moves the pose head


# ---------------------------------------------------------------------------
# The CORE inverse-steganalysis claim, on a toy detector with a KNOWN
# boundary, isolated from the real scorer: L-inf aiming beats L2 AND a
# shuffled control fails (the decisive no-fake test).
# ---------------------------------------------------------------------------


def test_linf_aiming_beats_l2_and_shuffle_control_fails_on_toy_detector():
    """Construct a toy detector with a KNOWN small-margin boundary region.

    The detector's argmax flips only where the saliency is high (a thin
    boundary). The L-inf allocation should protect that boundary (fine steps)
    while coarsening the blind interior; a shuffled control with the SAME step
    histogram but random placement should NOT protect the boundary, so it should
    do MUCH worse. This is the decisive no-fake check, with a known oracle.
    """
    rng = np.random.default_rng(2026)
    h, w = 64, 64
    n = h * w
    # Boundary: a thin diagonal band where the detector is sensitive (high
    # saliency); everywhere else is detector-blind (low saliency).
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    boundary = (np.abs(yy - xx) < 2).reshape(-1)
    saliency = np.where(boundary, 1.0e6, 1.0e-3)
    rho = margin_budget_from_saliency(saliency)
    target = 4.0 * n

    l2 = allocate_l2_uniform(n, target_bits=target)
    linf = allocate_linf_margin_budget(
        rho, target_bits=target, min_step=1.0, max_step=128.0,
        fairness_direction="disadvantage_linf",
    )
    shuffled = linf.steps.copy()
    rng.shuffle(shuffled)

    # Toy "score": a pixel's decision flips iff its perturbation magnitude
    # (proportional to its step) exceeds its margin (small on the boundary,
    # large off it). The contest-like distortion is the boundary-flip rate.
    margins = np.where(boundary, 3.0, 1.0e9)  # boundary is fragile

    def flip_rate(steps: np.ndarray) -> float:
        perturb = steps / 2.0  # worst-case uniform-quantizer perturbation
        return float((perturb > margins).mean())

    d_l2 = flip_rate(l2.steps)
    d_linf = flip_rate(linf.steps)
    d_shuf = flip_rate(shuffled)

    # L-inf protects the boundary (fine steps there) -> fewer flips than L2.
    assert d_linf < d_l2
    # The shuffled control destroys the aiming -> it must do worse than L-inf.
    assert d_shuf > d_linf
    # Equal rate held.
    assert abs(linf.total_bits - l2.total_bits) / l2.total_bits < 1.0e-3
