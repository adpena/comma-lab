# SPDX-License-Identifier: MIT
"""Lane MAE-V integration tests for train_renderer wiring.

Pins the contract that:
1. The mae_v_dilated_h64 profile resolves through parse_args correctly.
2. The CLI flags --use-mae-mask-aug / --mae-mask-ratio / --mae-patch-size
   are wired (anti-dead-flag per CLAUDE.md / feedback_dead_flag_wiring_pattern).
3. A 5-step micro-training loop with the augmenter active produces finite,
   decreasing losses (anti-arbitrariness magnitude anchor — Round 26 convention).
4. Eval-mode passthrough is enforced even when CLI says enabled.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from tac.experiments.train_renderer import parse_args
from tac.mae_mask_aug import MAEMaskAugConfig, MAEMaskAugmenter


def test_profile_resolves_mae_v_flags() -> None:
    """Profile mae_v_dilated_h64 must wire the three MAE knobs into args."""
    args = parse_args([
        "--profile", "mae_v_dilated_h64",
        "--tag", "unit",
        "--no-auth-eval-on-best",
    ])
    assert args.use_mae_mask_aug is True
    assert math.isclose(args.mae_mask_ratio, 0.25)
    assert args.mae_patch_size == 16


def test_cli_overrides_take_precedence_over_profile() -> None:
    """CLI flags override profile values (canonical _resolve() semantics)."""
    args = parse_args([
        "--profile", "mae_v_dilated_h64",
        "--use-mae-mask-aug",
        "--mae-mask-ratio", "0.40",
        "--mae-patch-size", "8",
        "--tag", "unit",
        "--no-auth-eval-on-best",
    ])
    assert args.use_mae_mask_aug is True
    assert math.isclose(args.mae_mask_ratio, 0.40)
    assert args.mae_patch_size == 8


def test_default_off_when_profile_does_not_set() -> None:
    """Without the MAE-V profile or CLI flags, the augmenter is OFF."""
    args = parse_args([
        "--profile", "proven_baseline",
        "--tag", "unit",
        "--no-auth-eval-on-best",
    ])
    assert args.use_mae_mask_aug is False


def test_5step_train_loop_loss_finite_and_decreases() -> None:
    """5-step micro-training with augmenter active: loss must be finite + drop.

    Anti-arbitrariness magnitude anchor (Round 26): the loss must DECREASE
    monotonically (or at minimum: final < initial) in this idealised toy
    setup so we know the augmenter does not break gradient flow.
    """
    args = parse_args([
        "--profile", "mae_v_dilated_h64",
        "--tag", "unit",
        "--no-auth-eval-on-best",
    ])
    assert args.use_mae_mask_aug is True

    torch.manual_seed(int(args.seed))

    # Toy renderer: 5-class long mask -> embedding -> conv -> 3ch RGB.
    class TinyRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(5, 4)
            self.conv = nn.Conv2d(4, 3, kernel_size=3, padding=1)

        def forward(self, masks: torch.Tensor) -> torch.Tensor:
            # masks: (B, H, W) long -> (B, 4, H, W) float
            e = self.embed(masks).permute(0, 3, 1, 2).contiguous()
            return self.conv(e)

    model = TinyRenderer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    cfg = MAEMaskAugConfig(
        mask_ratio=float(args.mae_mask_ratio),
        patch_size=int(args.mae_patch_size),
        num_classes=5,
        enabled=True,
    )
    gen = torch.Generator().manual_seed(int(args.seed) + 30_028)
    augmenter = MAEMaskAugmenter(cfg, generator=gen)
    augmenter.train()

    # Fixed input/target for repeatable loss-curve check.
    mask_gen = torch.Generator().manual_seed(int(args.seed) + 1)
    masks = torch.randint(0, 5, (2, 32, 48), dtype=torch.long, generator=mask_gen)
    target = torch.zeros(2, 3, 32, 48)

    losses: list[float] = []
    for _ in range(5):
        aug_masks, _patch_mask = augmenter(masks)
        rendered = model(aug_masks)
        loss = (rendered - target).pow(2).mean()
        assert math.isfinite(loss.item()), (
            f"loss diverged: {loss.item()} (augmenter broke gradient flow?)"
        )
        losses.append(float(loss.item()))
        opt.zero_grad()
        loss.backward()
        # Verify model gradients are finite (augmenter must not produce NaN).
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "non-finite model grad"
        opt.step()

    # Magnitude anchor: final loss must be strictly less than initial.
    assert losses[-1] < losses[0], (
        f"loss did not decrease across 5 steps: {losses} "
        "(toy model with augmenter should still converge)"
    )


def test_eval_mode_strictly_passes_through() -> None:
    """Even with --use-mae-mask-aug, eval mode must NOT perturb masks.

    This is the contest-distribution invariant: at inference time the
    scorer sees raw masks. Anything else is a measurement bug.
    """
    args = parse_args([
        "--profile", "mae_v_dilated_h64",
        "--tag", "unit",
        "--no-auth-eval-on-best",
    ])
    cfg = MAEMaskAugConfig(
        mask_ratio=float(args.mae_mask_ratio),
        patch_size=int(args.mae_patch_size),
        num_classes=5,
        enabled=True,
    )
    augmenter = MAEMaskAugmenter(cfg, generator=torch.Generator().manual_seed(0))
    augmenter.eval()

    masks = torch.randint(0, 5, (3, 64, 96), dtype=torch.long)
    out, patch_mask = augmenter(masks)

    assert torch.equal(out, masks), "eval mode altered masks (contest-eval breakage!)"
    assert not patch_mask.any().item(), "eval mode produced non-empty patch_mask"
