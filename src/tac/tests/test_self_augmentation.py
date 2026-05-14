# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn


def test_perturb_poses_is_differentiable() -> None:
    from tac.self_augmentation import SelfAugmentationConfig, SelfAugmenter

    cfg = SelfAugmentationConfig(sigma_pose_init=0.005, sigma_pose_final=0.005)
    augmenter = SelfAugmenter(cfg).double()
    poses = torch.randn(3, 6, dtype=torch.double, requires_grad=True)

    def fn(x: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(1234)
        return augmenter.perturb_poses(x, epoch=0, total_epochs=1)

    assert torch.autograd.gradcheck(fn, (poses,), eps=1e-6, atol=1e-4)


def test_perturb_poses_distribution_matches_sigma() -> None:
    from tac.self_augmentation import SelfAugmentationConfig, SelfAugmenter

    sigma = 0.02
    cfg = SelfAugmentationConfig(sigma_pose_init=sigma, sigma_pose_final=sigma)
    augmenter = SelfAugmenter(cfg)
    poses = torch.zeros(20_000, 6)

    torch.manual_seed(123)
    perturbed = augmenter.perturb_poses(poses, epoch=0, total_epochs=1)
    empirical = (perturbed - poses).std(unbiased=True).item()

    assert empirical == pytest.approx(sigma, rel=0.15)


def test_schedule_cosine_anneals_correctly() -> None:
    from tac.self_augmentation import cosine_schedule

    init = 0.005
    final = 0.001
    total_epochs = 10
    values = [
        cosine_schedule(init, final, epoch=e, total_epochs=total_epochs)
        for e in range(total_epochs)
    ]

    assert values[0] == pytest.approx(init)
    assert values[-1] == pytest.approx(final)
    assert all(a > b for a, b in zip(values, values[1:]))


def test_perturb_masks_preserves_total_pixels() -> None:
    from tac.self_augmentation import SelfAugmentationConfig, SelfAugmenter

    cfg = SelfAugmentationConfig(
        mask_flip_init=0.2,
        mask_flip_final=0.2,
        mask_num_classes=5,
    )
    augmenter = SelfAugmenter(cfg)
    masks = torch.randint(0, 5, (2, 12, 16), dtype=torch.long)

    torch.manual_seed(123)
    perturbed = augmenter.perturb_masks(masks, epoch=0, total_epochs=1)

    assert perturbed.dtype == torch.long
    assert perturbed.shape == masks.shape
    assert perturbed.numel() == masks.numel()
    assert int(perturbed.numel()) == int(masks.numel())


def test_disabled_returns_input_unchanged() -> None:
    from tac.self_augmentation import SelfAugmentationConfig, SelfAugmenter

    cfg = SelfAugmentationConfig(enabled=False)
    augmenter = SelfAugmenter(cfg)
    poses = torch.randn(4, 6)
    masks = torch.randint(0, 5, (2, 8, 8), dtype=torch.long)

    assert augmenter.perturb_poses(poses, epoch=3, total_epochs=10) is poses
    assert augmenter.perturb_masks(masks, epoch=3, total_epochs=10) is masks


def test_fit_from_proxy_auth_log(tmp_path: Path) -> None:
    from tac.self_augmentation import SelfAugmentationConfig, SelfAugmenter

    log_path = tmp_path / "proxy_auth.jsonl"
    rows = [
        {"pose_per_dim": [0.01, -0.02, 0.03, -0.04, 0.05, -0.06]},
        {"pose_per_dim": [0.02, -0.01, 0.04, -0.03, 0.06, -0.05]},
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    augmenter = SelfAugmenter(SelfAugmentationConfig())
    fitted = augmenter.fit_from_proxy_auth_log(log_path)

    assert fitted is augmenter
    assert augmenter.pose_sigma_per_dim is not None
    assert torch.all(augmenter.pose_sigma_per_dim > 0)


def test_train_epoch_integration() -> None:
    from tac.self_augmentation import SelfAugmentationConfig, SelfAugmenter

    class TinyPoseRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(5, 4)
            self.pose_proj = nn.Linear(6, 4)
            self.head = nn.Conv2d(4, 3, kernel_size=1)

        def forward(
            self,
            mask_t: torch.Tensor,
            mask_t1: torch.Tensor,
            *,
            pose: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del mask_t
            x = self.embedding(mask_t1).permute(0, 3, 1, 2).contiguous()
            if pose is not None:
                x = x + self.pose_proj(pose).view(pose.shape[0], 4, 1, 1)
            frame_t1 = self.head(x)
            frame_t = frame_t1 + 0.1
            return torch.stack([frame_t, frame_t1], dim=1).permute(0, 1, 3, 4, 2)

    torch.manual_seed(1234)
    augmenter = SelfAugmenter(SelfAugmentationConfig())
    model = TinyPoseRenderer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    mask_t = torch.randint(0, 5, (2, 8, 8), dtype=torch.long)
    mask_t1 = torch.randint(0, 5, (2, 8, 8), dtype=torch.long)
    poses = torch.randn(2, 6)
    target = torch.zeros(2, 2, 8, 8, 3)

    noisy_pose = augmenter.perturb_poses(poses, epoch=0, total_epochs=5)
    rendered = model(mask_t, mask_t1, pose=noisy_pose)
    loss = (rendered - target).pow(2).mean()
    assert math.isfinite(loss.item())

    loss.backward()
    optimizer.step()

    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
