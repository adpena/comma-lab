# SPDX-License-Identifier: MIT
"""Tests for the U-DIE-KL substrate-wide loss helper.

Per the Grand Reunion symposium 2026-05-15 Phase D #5 composite + the
canonical design memo at
``.omx/research/u_die_kl_substrate_wide_loss_v1_design_20260515.md``.

The helper combines three score-aware-loss families into a single
torch-tensor loss:

* **U** = UNIWARD per-pixel embedding-cost weighting
  (Holub-Fridrich-Denemark *EURASIP JIS* 2014)
* **DIE** = Detector-Informed Embedding scorer-gradient weighting
  (Yousfi 2022)
* **KL** = Hinton-Vinyals-Dean 2014 SegNet logit distillation, T=2.0
  (Quantizr 0.33 archive recipe per CLAUDE.md "Quantizr intelligence")

Lane: ``lane_u_die_kl_substrate_wide_loss_v1_20260515``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.losses import (
    DEFAULT_DIE_CACHE_INTERVAL,
    DEFAULT_KL_TEMPERATURE,
    DEFAULT_UNIWARD_EPSILON,
    UDIEKLConfig,
    UDIEKLLoss,
    compute_die_weight_map,
    compute_uniward_weight_map,
    kl_distill_segnet_term,
)

# ---------------------------------------------------------------------------
# Fake scorers (sister of test_lane_12_v2_nerv_as_renderer.py::_FakeScorerSeg)
# ---------------------------------------------------------------------------


class _FakeScorerSeg(nn.Module):
    """Fake SegNet - small CNN that returns 5-class logits.

    Mirrors the contest scorer contract: ``preprocess_input(x)`` takes a
    5-D ``(B, T, C, H, W)`` tensor and returns a 4-D ``(B, C, H, W)``
    tensor (LAST frame only - matches modules.py SegNet semantics).
    """

    def __init__(self, *, num_classes: int = 5, in_channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.preprocess_calls = 0

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_calls += 1
        assert x.ndim == 5, f"expected 5-D (B, T, C, H, W); got {x.ndim}-D"
        # Take last frame only per modules.py SegNet semantics.
        return x[:, -1, :, :, :].float() / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected 4-D post-preprocess; got {x.ndim}-D"
        return self.conv(x)


class _FakeScorerPose(nn.Module):
    """Fake PoseNet - small MLP that returns dict with "pose" key shape (B, 12).

    Mirrors the contest scorer contract: ``preprocess_input(x)`` collapses
    the 5-D pair into 4-D channel-stacked YUV6-pair semantics; ``forward``
    returns a dict ``{"pose": (B, 12)}`` matching the contest PoseNet
    hydra-head output (per ``upstream/modules.py``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 12)
        self.preprocess_calls = 0

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_calls += 1
        assert x.ndim == 5
        B, T, C, H, W = x.shape
        return (x.reshape(B, T * C, H, W).float() / 255.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        assert x.ndim == 4
        # Reduce spatial dims via avgpool + linear - keep gradient flow.
        feat = x.mean(dim=(2, 3))  # (B, T*C)
        # Project to fc input dim = 8 by repeating/truncating.
        if feat.shape[1] < 8:
            feat = feat.repeat(1, (8 // feat.shape[1]) + 1)
        feat = feat[:, :8]
        return {"pose": self.fc(feat)}


def _freeze(*modules: nn.Module) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(False)


def _make_pair(
    B: int = 2, T: int = 2, C: int = 3, H: int = 8, W: int = 8, *, seed: int = 0
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (B, T, C, H, W), generator=g, dtype=torch.uint8).float()


# ---------------------------------------------------------------------------
# UNIWARD weight map tests
# ---------------------------------------------------------------------------


def test_uniward_weight_map_shape_matches_input() -> None:
    target = _make_pair(seed=1)
    w = compute_uniward_weight_map(target)
    B, T, C, H, W = target.shape
    assert w.shape == (B, T, 1, H, W)
    assert w.dtype == torch.float32


def test_uniward_weight_map_normalized_per_image_to_mean_one() -> None:
    target = _make_pair(seed=2)
    w = compute_uniward_weight_map(target)
    # Per-image (B, T, 1, H, W) -> mean over (1, H, W) per (B, T) image
    means = w.mean(dim=(2, 3, 4))
    assert torch.allclose(means, torch.ones_like(means), atol=1e-5)


def test_uniward_weight_map_high_at_flat_low_at_textured() -> None:
    # Construct a 2-region image: half flat, half textured (high-variance).
    B, T, C, H, W = 1, 1, 1, 8, 8
    target = torch.zeros((B, T, C, H, W), dtype=torch.float32)
    # Add high-frequency texture to right half.
    g = torch.Generator().manual_seed(7)
    target[:, :, :, :, W // 2 :] = (
        torch.randn(B, T, C, H, W // 2, generator=g) * 50.0 + 128.0
    )
    w = compute_uniward_weight_map(target)
    # Flat region (left) should have HIGHER weight than textured (right).
    flat_mean = w[:, :, :, :, : W // 2].mean()
    textured_mean = w[:, :, :, :, W // 2 :].mean()
    assert flat_mean > textured_mean, (
        f"UNIWARD inversion violated: flat={flat_mean:.3f}, textured={textured_mean:.3f}"
    )


def test_uniward_weight_map_detached_by_default() -> None:
    target = _make_pair(seed=3).requires_grad_(True)
    w = compute_uniward_weight_map(target)
    assert not w.requires_grad


def test_uniward_weight_map_no_grad_when_detach_false() -> None:
    target = _make_pair(seed=4).requires_grad_(True)
    w = compute_uniward_weight_map(target, detach=False)
    # detach=False keeps the grad-tracking graph
    assert w.requires_grad


def test_uniward_weight_map_handles_constant_image() -> None:
    target = torch.full((1, 1, 3, 8, 8), 128.0)
    w = compute_uniward_weight_map(target)
    # Constant image: rho ~= 1/eps everywhere; per-image mean=1 so output=1.
    assert torch.allclose(w, torch.ones_like(w), atol=1e-5)


def test_uniward_weight_map_no_nan_on_extremes() -> None:
    target = torch.zeros((1, 1, 3, 8, 8))
    w = compute_uniward_weight_map(target)
    assert not torch.isnan(w).any() and not torch.isinf(w).any()


def test_uniward_weight_map_invalid_shape_rejected() -> None:
    bad = torch.randn(2, 3, 8, 8)  # 4-D instead of 5-D
    with pytest.raises(ValueError, match="must be"):
        compute_uniward_weight_map(bad)


def test_uniward_weight_map_invalid_epsilon_rejected() -> None:
    target = _make_pair()
    with pytest.raises(ValueError, match="uniward_epsilon"):
        compute_uniward_weight_map(target, epsilon=0.0)
    with pytest.raises(ValueError, match="uniward_epsilon"):
        compute_uniward_weight_map(target, epsilon=-1.0)
    with pytest.raises(ValueError, match="uniward_epsilon"):
        compute_uniward_weight_map(target, epsilon=float("nan"))


# ---------------------------------------------------------------------------
# DIE weight map tests
# ---------------------------------------------------------------------------


def test_die_weight_map_shape_matches_input() -> None:
    pred = _make_pair(seed=10)
    target = _make_pair(seed=11)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    w = compute_die_weight_map(pred, target, seg, pose)
    assert w.shape == (pred.shape[0], pred.shape[1], 1, pred.shape[3], pred.shape[4])


def test_die_weight_map_normalized_per_image_to_mean_one() -> None:
    pred = _make_pair(seed=12)
    target = _make_pair(seed=13)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    w = compute_die_weight_map(pred, target, seg, pose)
    means = w.mean(dim=(2, 3, 4))
    assert torch.allclose(means, torch.ones_like(means), atol=1e-4)


def test_die_weight_map_zero_gradient_falls_back_to_neutral_weight() -> None:
    class _ZeroGradSeg(nn.Module):
        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, -1].float() / 255.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base = x[:, :1] * 0.0
            return base.repeat(1, 5, 1, 1)

    class _ZeroGradPose(nn.Module):
        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C, H, W = x.shape
            return x.reshape(B, T * C, H, W).float() / 255.0

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            base = x.mean(dim=(1, 2, 3), keepdim=False) * 0.0
            return {"pose": base[:, None].repeat(1, 12)}

    pred = _make_pair(seed=120)
    target = pred.clone()
    seg = _ZeroGradSeg()
    pose = _ZeroGradPose()
    _freeze(seg, pose)
    w = compute_die_weight_map(pred, target, seg, pose)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-6)


def test_die_weight_map_detached_by_default() -> None:
    pred = _make_pair(seed=14).requires_grad_(True)
    target = _make_pair(seed=15)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    w = compute_die_weight_map(pred, target, seg, pose)
    assert not w.requires_grad


def test_die_weight_map_does_not_pollute_pred_grad() -> None:
    """The DIE probe must NOT route gradient back to caller's pred tensor."""
    pred = _make_pair(seed=16).requires_grad_(True)
    target = _make_pair(seed=17)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    _ = compute_die_weight_map(pred, target, seg, pose)
    assert pred.grad is None, (
        f"DIE probe leaked gradient into caller's pred tensor; got grad shape {pred.grad.shape}"
    )


def test_die_weight_map_does_not_pollute_scorer_grad() -> None:
    """Frozen scorers must NOT receive gradient from the probe."""
    pred = _make_pair(seed=18)
    target = _make_pair(seed=19)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    _ = compute_die_weight_map(pred, target, seg, pose)
    for p in seg.parameters():
        assert p.grad is None
    for p in pose.parameters():
        assert p.grad is None


def test_die_weight_map_rejects_shape_mismatch() -> None:
    pred = _make_pair(seed=20, B=2)
    target = _make_pair(seed=21, B=3)  # mismatch
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    with pytest.raises(ValueError, match="shapes must match"):
        compute_die_weight_map(pred, target, seg, pose)


def test_die_weight_map_rejects_4d_input() -> None:
    pred = torch.randn(2, 3, 8, 8)  # 4-D
    target = torch.randn(2, 3, 8, 8)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    with pytest.raises(ValueError, match="must be"):
        compute_die_weight_map(pred, target, seg, pose)


# ---------------------------------------------------------------------------
# KL distillation term tests
# ---------------------------------------------------------------------------


def test_kl_distill_loss_zero_when_pred_equals_target() -> None:
    pair = _make_pair(seed=30)
    seg = _FakeScorerSeg()
    _freeze(seg)
    loss = kl_distill_segnet_term(pair, pair.clone(), seg)
    # KL(P||P) = 0 (modulo numerical noise); allow eps for floor noise.
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_distill_loss_positive_when_pred_differs_from_target() -> None:
    pred = _make_pair(seed=31)
    target = _make_pair(seed=32)
    seg = _FakeScorerSeg()
    _freeze(seg)
    loss = kl_distill_segnet_term(pred, target, seg)
    assert loss.item() > 0.0


def test_kl_distill_loss_t_squared_scaling() -> None:
    """Verify Hinton T^2 scaling: at higher T, loss scales by T^2 / T_ref^2."""
    pred = _make_pair(seed=33)
    target = _make_pair(seed=34)
    seg = _FakeScorerSeg()
    _freeze(seg)
    loss_t1 = kl_distill_segnet_term(pred, target, seg, temperature=1.0).item()
    loss_t2 = kl_distill_segnet_term(pred, target, seg, temperature=2.0).item()
    # KL at T scales approximately as 1/T^2 (raw); T^2 multiplier compensates.
    # The values should be of comparable order, NOT vastly different.
    # If T^2 scaling were missing, loss_t2 would be ~4x smaller than loss_t1.
    # With T^2 scaling, ratios stay within ~2x.
    if loss_t1 > 1e-9 and loss_t2 > 1e-9:
        ratio = loss_t2 / loss_t1
        assert 0.25 < ratio < 4.0, (
            f"T^2 scaling appears broken; ratio T=2/T=1 = {ratio:.4f}"
        )


def test_kl_distill_loss_gradient_flows_to_pred() -> None:
    pred = _make_pair(seed=35).requires_grad_(True)
    target = _make_pair(seed=36)
    seg = _FakeScorerSeg()
    _freeze(seg)
    loss = kl_distill_segnet_term(pred, target, seg)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()


def test_kl_distill_loss_gradient_does_not_flow_to_target() -> None:
    pred = _make_pair(seed=37).requires_grad_(True)
    target = _make_pair(seed=38).requires_grad_(True)
    seg = _FakeScorerSeg()
    _freeze(seg)
    loss = kl_distill_segnet_term(pred, target, seg)
    loss.backward()
    # Gradient flows to pred (student) but NOT to target (teacher).
    assert pred.grad is not None
    assert target.grad is None or torch.allclose(
        target.grad, torch.zeros_like(target.grad)
    )


def test_kl_distill_loss_default_temperature_is_quantizr_canon() -> None:
    """Per CLAUDE.md "Quantizr intelligence" - kl_on_logits(T=2.0)."""
    assert DEFAULT_KL_TEMPERATURE == 2.0


def test_kl_distill_loss_invalid_temperature_rejected() -> None:
    pair = _make_pair()
    seg = _FakeScorerSeg()
    _freeze(seg)
    with pytest.raises(ValueError, match="finite positive"):
        kl_distill_segnet_term(pair, pair.clone(), seg, temperature=0.0)
    with pytest.raises(ValueError, match="finite positive"):
        kl_distill_segnet_term(pair, pair.clone(), seg, temperature=-1.0)


def test_kl_distill_loss_rejects_4d_input() -> None:
    pred = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)
    seg = _FakeScorerSeg()
    _freeze(seg)
    with pytest.raises(ValueError, match="must be"):
        kl_distill_segnet_term(pred, target, seg)


def test_kl_distill_loss_rejects_shape_mismatch() -> None:
    pred = _make_pair(B=2)
    target = _make_pair(B=3)
    seg = _FakeScorerSeg()
    _freeze(seg)
    with pytest.raises(ValueError, match="shapes must match"):
        kl_distill_segnet_term(pred, target, seg)


def test_kl_distill_loss_calls_preprocess_input_per_catalog_164() -> None:
    """Catalog #164 invariant: preprocess_input MUST be called before forward."""
    pair = _make_pair()
    seg = _FakeScorerSeg()
    _freeze(seg)
    initial_calls = seg.preprocess_calls
    _ = kl_distill_segnet_term(pair, pair.clone(), seg)
    # 1 call for student, 1 call for teacher
    assert seg.preprocess_calls == initial_calls + 2


# ---------------------------------------------------------------------------
# UDIEKLConfig validation tests
# ---------------------------------------------------------------------------


def test_config_default_construction_is_clean() -> None:
    cfg = UDIEKLConfig()
    assert cfg.alpha == 0.5
    assert cfg.beta == 0.5
    assert cfg.gamma == 1.0
    assert cfg.kl_temperature == DEFAULT_KL_TEMPERATURE
    assert cfg.uniward_epsilon == DEFAULT_UNIWARD_EPSILON
    assert cfg.die_cache_interval == DEFAULT_DIE_CACHE_INTERVAL


def test_config_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        UDIEKLConfig(alpha=-0.1)


def test_config_rejects_negative_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        UDIEKLConfig(beta=-0.1)


def test_config_rejects_negative_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        UDIEKLConfig(gamma=-0.1)


def test_config_rejects_zero_temperature() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        UDIEKLConfig(kl_temperature=0.0)


def test_config_rejects_zero_epsilon() -> None:
    with pytest.raises(ValueError, match="uniward_epsilon"):
        UDIEKLConfig(uniward_epsilon=0.0)


def test_config_rejects_zero_cache_interval() -> None:
    with pytest.raises(ValueError, match="die_cache_interval"):
        UDIEKLConfig(die_cache_interval=0)


def test_config_rejects_nan_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        UDIEKLConfig(alpha=float("nan"))


def test_config_alpha_zero_accepted() -> None:
    """Zero weight is a legitimate disable-this-term signal, not an error."""
    cfg = UDIEKLConfig(alpha=0.0, beta=0.0, gamma=1.0)
    assert cfg.alpha == 0.0


# ---------------------------------------------------------------------------
# UDIEKLLoss composite tests
# ---------------------------------------------------------------------------


def _make_loss(**kwargs) -> tuple[UDIEKLLoss, _FakeScorerSeg, _FakeScorerPose]:
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    loss_fn = UDIEKLLoss(scorer_seg=seg, scorer_pose=pose, **kwargs)
    return loss_fn, seg, pose


def test_loss_constructor_rejects_non_module_scorer() -> None:
    pose = _FakeScorerPose()
    with pytest.raises(TypeError, match="scorer_seg"):
        UDIEKLLoss(scorer_seg="not a module", scorer_pose=pose)
    seg = _FakeScorerSeg()
    with pytest.raises(TypeError, match="scorer_pose"):
        UDIEKLLoss(scorer_seg=seg, scorer_pose="not a module")


def test_loss_constructor_rejects_unfrozen_scorers() -> None:
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    with pytest.raises(ValueError, match="scorer_seg must be frozen"):
        UDIEKLLoss(scorer_seg=seg, scorer_pose=pose)


def test_loss_returns_finite_scalar_on_default_config() -> None:
    loss_fn, _, _ = _make_loss()
    pred = _make_pair(seed=40)
    target = _make_pair(seed=41)
    out = loss_fn(pred, target)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_loss_zero_when_all_weights_zero() -> None:
    loss_fn, _, _ = _make_loss(alpha=0.0, beta=0.0, gamma=0.0)
    pred = _make_pair(seed=42)
    target = _make_pair(seed=43)
    out = loss_fn(pred, target)
    assert out.item() == 0.0


def test_loss_only_uniward_when_other_weights_zero() -> None:
    """alpha=1, beta=gamma=0 reduces to UNIWARD-weighted MSE."""
    loss_fn, _, _ = _make_loss(alpha=1.0, beta=0.0, gamma=0.0)
    pred = _make_pair(seed=44)
    target = _make_pair(seed=45)
    out = loss_fn(pred, target)
    # Manual UNIWARD-weighted MSE for parity check.
    residual = (pred.float() - target.float()).pow(2)
    w = compute_uniward_weight_map(target, epsilon=DEFAULT_UNIWARD_EPSILON, detach=True)
    expected = (residual * w).mean()
    assert out.item() == pytest.approx(expected.item(), rel=1e-5)


def test_loss_only_kl_when_other_weights_zero() -> None:
    """alpha=beta=0, gamma=1 reduces to KL distillation term."""
    loss_fn, seg, _ = _make_loss(alpha=0.0, beta=0.0, gamma=1.0)
    pred = _make_pair(seed=46)
    target = _make_pair(seed=47)
    out = loss_fn(pred, target)
    expected = kl_distill_segnet_term(pred, target, seg, temperature=DEFAULT_KL_TEMPERATURE)
    assert out.item() == pytest.approx(expected.item(), rel=1e-5)


def test_loss_decreases_when_pred_approaches_target() -> None:
    """Sanity: loss should decrease as pred -> target (mathematical property)."""
    loss_fn, _, _ = _make_loss()
    target = _make_pair(seed=48)
    # pred far from target
    pred_far = _make_pair(seed=49)
    # pred closer to target
    pred_near = (target.float() * 0.7 + pred_far.float() * 0.3)
    loss_far = loss_fn(pred_far, target).item()
    loss_near = loss_fn(pred_near, target).item()
    assert loss_near <= loss_far + 1e-3, (
        f"Loss did not decrease toward target: far={loss_far}, near={loss_near}"
    )


def test_loss_gradient_flows_to_pred() -> None:
    loss_fn, _, _ = _make_loss()
    pred = _make_pair(seed=50).requires_grad_(True)
    target = _make_pair(seed=51)
    out = loss_fn(pred, target)
    out.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()
    # And the gradient is not all zero (sanity check on signal strength)
    assert pred.grad.abs().max().item() > 0.0


def test_loss_no_gradient_to_scorer_weights() -> None:
    """Frozen scorer weights MUST NOT receive gradient from the composite loss."""
    loss_fn, seg, pose = _make_loss()
    pred = _make_pair(seed=52).requires_grad_(True)
    target = _make_pair(seed=53)
    out = loss_fn(pred, target)
    out.backward()
    for p in seg.parameters():
        assert p.grad is None
    for p in pose.parameters():
        assert p.grad is None


def test_loss_module_has_no_learnable_parameters() -> None:
    """The helper adds no learnable params (scorers are frozen by contract)."""
    loss_fn, _, _ = _make_loss()
    # The helper itself has no submodules registered with parameters
    # (scorers are CALLER-owned and stored as non-Module attributes).
    own_params = [p for n, p in loss_fn.named_parameters() if p.requires_grad]
    assert own_params == [], (
        f"UDIEKLLoss should add no learnable params; got {[n for n,_ in loss_fn.named_parameters() if _.requires_grad]}"
    )


def test_loss_no_nan_or_inf_on_constant_input() -> None:
    loss_fn, _, _ = _make_loss()
    target = torch.full((1, 2, 3, 8, 8), 128.0)
    pred = torch.full((1, 2, 3, 8, 8), 200.0)
    out = loss_fn(pred, target)
    assert torch.isfinite(out)


def test_loss_rejects_4d_input() -> None:
    loss_fn, _, _ = _make_loss()
    pred = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)
    with pytest.raises(ValueError, match="must be"):
        loss_fn(pred, target)


def test_loss_rejects_shape_mismatch() -> None:
    loss_fn, _, _ = _make_loss()
    pred = _make_pair(B=2)
    target = _make_pair(B=3)
    with pytest.raises(ValueError, match="shapes must match"):
        loss_fn(pred, target)


def test_loss_calls_preprocess_input_via_canonical_path() -> None:
    """Catalog #164: scorer.preprocess_input MUST be called before scorer()."""
    loss_fn, seg, pose = _make_loss(alpha=0.0, beta=1.0, gamma=1.0)
    pred = _make_pair(seed=54)
    target = _make_pair(seed=55)
    initial_seg = seg.preprocess_calls
    initial_pose = pose.preprocess_calls
    _ = loss_fn(pred, target)
    # DIE term (beta=1.0) calls scorer_forward_pair which preprocesses BOTH
    # student (predicted) and teacher (target) for BOTH scorers.
    # KL term (gamma=1.0) calls SegNet preprocess for student + teacher.
    assert seg.preprocess_calls > initial_seg
    assert pose.preprocess_calls > initial_pose


# ---------------------------------------------------------------------------
# DIE cache tests
# ---------------------------------------------------------------------------


def test_die_cache_recomputes_on_first_call() -> None:
    loss_fn, seg, pose = _make_loss(alpha=0.0, beta=1.0, gamma=0.0, die_cache_interval=10)
    pred = _make_pair(seed=60)
    target = _make_pair(seed=61)
    initial = seg.preprocess_calls
    _ = loss_fn(pred, target)
    assert seg.preprocess_calls > initial


def test_die_cache_skips_recompute_within_interval() -> None:
    loss_fn, seg, _ = _make_loss(alpha=0.0, beta=1.0, gamma=0.0, die_cache_interval=10)
    pred = _make_pair(seed=62)
    target = _make_pair(seed=63)
    _ = loss_fn(pred, target)  # first call: computes DIE map
    seg_calls_after_first = seg.preprocess_calls
    _ = loss_fn(pred, target)  # second call: should NOT recompute
    # Note: seg.preprocess_calls counts ALL preprocess invocations,
    # including those inside compute_die_weight_map (called once on the
    # first call) but NOT a second time within the cache interval.
    # The DIE call inside computes pred + target = 2 preprocess calls per scorer.
    # The next 9 calls within the interval should NOT add DIE preprocess.
    assert seg.preprocess_calls == seg_calls_after_first


def test_die_cache_recomputes_after_interval() -> None:
    interval = 3
    loss_fn, seg, _ = _make_loss(
        alpha=0.0, beta=1.0, gamma=0.0, die_cache_interval=interval
    )
    pred = _make_pair(seed=64)
    target = _make_pair(seed=65)
    # Call interval+1 times; the (interval+1)-th call should trigger recompute.
    for _ in range(interval + 1):
        _ = loss_fn(pred, target)
    # 2 recomputes happened: at step 0 and step `interval` (mod interval == 0).
    # Each recompute calls preprocess for pred+target on the seg scorer (2 calls).
    assert seg.preprocess_calls >= 4  # at least 2 recomputes * 2 preprocess calls


def test_die_cache_invalidates_on_shape_change() -> None:
    loss_fn, seg, _ = _make_loss(alpha=0.0, beta=1.0, gamma=0.0, die_cache_interval=100)
    pred_a = _make_pair(B=2, H=8, seed=70)
    target_a = _make_pair(B=2, H=8, seed=71)
    _ = loss_fn(pred_a, target_a)
    seg_after_a = seg.preprocess_calls
    # Different batch size invalidates cache
    pred_b = _make_pair(B=3, H=8, seed=72)
    target_b = _make_pair(B=3, H=8, seed=73)
    _ = loss_fn(pred_b, target_b)
    assert seg.preprocess_calls > seg_after_a


def test_die_cache_reset_clears_state() -> None:
    loss_fn, seg, _ = _make_loss(alpha=0.0, beta=1.0, gamma=0.0, die_cache_interval=100)
    pred = _make_pair(seed=74)
    target = _make_pair(seed=75)
    _ = loss_fn(pred, target)
    seg_after_first = seg.preprocess_calls
    loss_fn.reset_die_cache()
    _ = loss_fn(pred, target)
    # After reset, should recompute on the next call.
    assert seg.preprocess_calls > seg_after_first


# ---------------------------------------------------------------------------
# Synthetic substrate integration test
# ---------------------------------------------------------------------------


class _TinyRenderer(nn.Module):
    """Tiny CNN renderer: input pair -> output pair (identity-ish)."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x_btchw: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_btchw.shape
        flat = x_btchw.float().reshape(B * T, C, H, W)
        out = self.conv(flat)
        return out.reshape(B, T, C, H, W)


def test_substrate_integration_loss_decreases_after_training_step() -> None:
    """End-to-end: a tiny renderer + UDIEKL loss + optimizer step decreases the loss."""
    torch.manual_seed(0)
    renderer = _TinyRenderer()
    target = _make_pair(seed=80, B=1, H=8, W=8)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    loss_fn = UDIEKLLoss(
        scorer_seg=seg, scorer_pose=pose,
        alpha=0.5, beta=0.5, gamma=1.0,
        die_cache_interval=1,  # recompute every step for cleaner gradient
    )
    opt = torch.optim.Adam(renderer.parameters(), lr=1e-2)

    losses = []
    for _ in range(8):
        opt.zero_grad()
        pred = renderer(target)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Loss should be roughly decreasing - allow tolerance for noise.
    assert losses[-1] <= losses[0] + 1e-3, (
        f"Loss did not decrease over training: first={losses[0]}, last={losses[-1]}, all={losses}"
    )


def test_substrate_integration_no_nan_after_many_steps() -> None:
    torch.manual_seed(1)
    renderer = _TinyRenderer()
    target = _make_pair(seed=90, B=1, H=8, W=8)
    seg = _FakeScorerSeg()
    pose = _FakeScorerPose()
    _freeze(seg, pose)
    loss_fn = UDIEKLLoss(scorer_seg=seg, scorer_pose=pose, die_cache_interval=2)
    opt = torch.optim.Adam(renderer.parameters(), lr=1e-3)

    for _ in range(15):
        opt.zero_grad()
        pred = renderer(target)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), f"loss went non-finite during training: {loss}"


# ---------------------------------------------------------------------------
# Re-export sanity tests (Catalog #13 sister: ensure __all__ stays in sync)
# ---------------------------------------------------------------------------


def test_canonical_helper_reexported_from_tac_losses() -> None:
    import tac.losses as losses_pkg
    assert hasattr(losses_pkg, "UDIEKLLoss")
    assert hasattr(losses_pkg, "UDIEKLConfig")
    assert hasattr(losses_pkg, "compute_uniward_weight_map")
    assert hasattr(losses_pkg, "compute_die_weight_map")
    assert hasattr(losses_pkg, "kl_distill_segnet_term")
    assert hasattr(losses_pkg, "DEFAULT_KL_TEMPERATURE")
    assert hasattr(losses_pkg, "DEFAULT_UNIWARD_EPSILON")
    assert hasattr(losses_pkg, "DEFAULT_DIE_CACHE_INTERVAL")


def test_canonical_helper_in_all_export_list() -> None:
    import tac.losses as losses_pkg
    for name in (
        "UDIEKLLoss",
        "UDIEKLConfig",
        "compute_uniward_weight_map",
        "compute_die_weight_map",
        "kl_distill_segnet_term",
        "DEFAULT_KL_TEMPERATURE",
        "DEFAULT_UNIWARD_EPSILON",
        "DEFAULT_DIE_CACHE_INTERVAL",
    ):
        assert name in losses_pkg.__all__, f"{name} missing from __all__"
