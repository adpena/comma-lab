# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch

from tac.dinov3_cooperative_receiver_anchor import (
    CANONICAL_DINOV3_MODEL_NAME,
    DINOV3_INPUT_SIZE,
    DinoV3Features,
    _normalize_for_dinov3,
    cooperative_receiver_dinov3_kl_loss,
    dinov3_pair_features,
)


class _TinyDino(torch.nn.Module):
    """Small differentiable stand-in for a timm ViT forward_features model."""

    def __init__(self, hidden_dim: int = 8, register_tokens: int = 0) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, hidden_dim, bias=False)
        self.num_prefix_tokens = 1 + register_tokens
        self.register_tokens = register_tokens

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        cls_rgb = x.mean(dim=(2, 3))
        cls = self.proj(cls_rgb)
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
        patches_rgb = pooled.permute(0, 2, 3, 1).reshape(x.shape[0], 4, 3)
        patches = self.proj(patches_rgb)
        if self.register_tokens:
            regs = torch.zeros(
                x.shape[0],
                self.register_tokens,
                patches.shape[-1],
                dtype=patches.dtype,
                device=patches.device,
            )
            return torch.cat([cls[:, None, :], regs, patches], dim=1)
        return torch.cat([cls[:, None, :], patches], dim=1)


def test_dinov3_features_validate_shapes_and_pair_metadata() -> None:
    model = _TinyDino()
    pair = torch.rand(2, 2, 3, 12, 16) * 255.0

    features = dinov3_pair_features(pair, model=model)

    assert features.frame_0.model_name == CANONICAL_DINOV3_MODEL_NAME
    assert features.frame_0.cls_token.shape == (2, 8)
    assert features.frame_1.patch_tokens.shape == (2, 4, 8)
    assert features.frame_0.frame_idx == 0
    assert features.frame_1.frame_idx == 1


def test_dinov3_features_strip_register_tokens_from_patch_grid() -> None:
    model = _TinyDino(register_tokens=4)
    pair = torch.rand(1, 2, 3, 12, 16) * 255.0

    features = dinov3_pair_features(pair, model=model)

    assert features.frame_0.patch_tokens.shape == (1, 4, 8)
    assert features.frame_1.patch_tokens.shape == (1, 4, 8)


def test_dinov3_normalization_resizes_rgb255_inputs() -> None:
    rgb = torch.full((1, 3, 18, 20), 127.5)

    normalized = _normalize_for_dinov3(rgb)

    assert normalized.shape == (1, 3, DINOV3_INPUT_SIZE, DINOV3_INPUT_SIZE)
    assert torch.isfinite(normalized).all()


def test_dinov3_kl_loss_backprops_to_predicted_pair_only() -> None:
    model = _TinyDino()
    predicted = (torch.rand(1, 2, 3, 12, 16) * 255.0).requires_grad_()
    ground_truth = torch.rand(1, 2, 3, 12, 16) * 255.0

    loss = cooperative_receiver_dinov3_kl_loss(
        predicted,
        ground_truth,
        model=model,
        temperature=2.0,
        use_cls_token=True,
        use_patch_tokens=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert predicted.grad.abs().sum() > 0
    assert ground_truth.grad is None


def test_dinov3_kl_loss_refuses_shape_and_term_misconfiguration() -> None:
    model = _TinyDino()
    predicted = torch.rand(1, 2, 3, 12, 16)
    ground_truth = torch.rand(1, 2, 3, 12, 16)

    with pytest.raises(ValueError, match="at least one"):
        cooperative_receiver_dinov3_kl_loss(
            predicted,
            ground_truth,
            model=model,
            use_cls_token=False,
            use_patch_tokens=False,
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        cooperative_receiver_dinov3_kl_loss(
            predicted,
            ground_truth[:, :, :, :8, :8],
            model=model,
        )
    with pytest.raises(ValueError, match="frame_idx"):
        DinoV3Features(
            cls_token=torch.rand(1, 8),
            patch_tokens=torch.rand(1, 4, 8),
            frame_idx=2,
        )
