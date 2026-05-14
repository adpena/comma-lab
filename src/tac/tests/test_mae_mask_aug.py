# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch

from tac.mae_mask_aug import MAEMaskAugConfig, MAEMaskAugmenter


def _make(B: int = 2, H: int = 64, W: int = 96, num_classes: int = 5,
          mask_ratio: float = 0.25, patch_size: int = 16,
          seed: int = 0) -> tuple[MAEMaskAugmenter, torch.Tensor]:
    cfg = MAEMaskAugConfig(
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        num_classes=num_classes,
    )
    gen = torch.Generator().manual_seed(seed)
    aug = MAEMaskAugmenter(cfg, generator=gen)
    aug.train()
    mask_gen = torch.Generator().manual_seed(seed + 1)
    masks = torch.randint(
        0, num_classes, (B, H, W), dtype=torch.long, generator=mask_gen
    )
    return aug, masks


def test_shape_and_dtype_preserved() -> None:
    aug, masks = _make(B=3, H=64, W=96)
    out, patch_mask = aug(masks)

    assert out.shape == masks.shape
    assert out.dtype == torch.long
    assert patch_mask.dtype == torch.bool
    assert patch_mask.shape == (3, 64 // 16, 96 // 16)


def test_mask_ratio_within_tolerance() -> None:
    """Empirical patch-mask rate should match config mask_ratio (±2%)."""
    aug, masks = _make(B=64, H=128, W=128, mask_ratio=0.25, seed=42)
    out, patch_mask = aug(masks)

    empirical = patch_mask.float().mean().item()
    assert abs(empirical - 0.25) < 0.02, (
        f"empirical mask rate {empirical:.4f} != target 0.25"
    )

    # Verify all unmasked patches are unchanged in the output.
    p = aug.config.patch_size
    pixel_active = patch_mask.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)
    assert torch.equal(out[~pixel_active], masks[~pixel_active])


def test_gradient_flows_through_token_logits() -> None:
    """Backprop through the loss-bridge populates token_logits.grad."""
    aug, masks = _make(seed=7)
    out, _ = aug(masks)
    assert torch.is_floating_point(out) is False  # output is long

    bridge = aug.token_loss_bridge()
    bridge.backward()

    assert aug.token_logits.grad is not None
    assert torch.isfinite(aug.token_logits.grad).all()
    assert aug.token_logits.grad.abs().sum() > 0


def test_deterministic_with_same_seed() -> None:
    aug_a, masks_a = _make(seed=2026)
    aug_b, masks_b = _make(seed=2026)
    out_a, pm_a = aug_a(masks_a)
    out_b, pm_b = aug_b(masks_b)

    assert torch.equal(out_a, out_b)
    assert torch.equal(pm_a, pm_b)


def test_eval_mode_passthrough() -> None:
    """Eval mode must return masks unchanged so contest distribution holds."""
    aug, masks = _make(seed=99)
    aug.eval()
    out, patch_mask = aug(masks)

    assert torch.equal(out, masks)
    assert patch_mask.any().item() is False


def test_disabled_config_passthrough() -> None:
    cfg = MAEMaskAugConfig(mask_ratio=0.25, enabled=False)
    aug = MAEMaskAugmenter(cfg, generator=torch.Generator().manual_seed(1))
    aug.train()
    masks = torch.randint(0, 5, (2, 32, 32), dtype=torch.long)

    out, patch_mask = aug(masks)
    assert torch.equal(out, masks)
    assert patch_mask.any().item() is False


def test_learnable_token_is_parameter() -> None:
    """token_logits must be an nn.Parameter so optimizer picks it up."""
    aug, _ = _make()
    params = list(aug.parameters())
    assert any(p is aug.token_logits for p in params)
    assert aug.token_logits.requires_grad is True
    assert aug.token_logits.shape == (5,)
