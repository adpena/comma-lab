# SPDX-License-Identifier: MIT
"""Score-aware loss for the Z7 GRU predictive-coding prebuild.

This module is a compress-time training surface only. It routes reconstructed
frame pairs through the canonical scorer contract and keeps inflate/runtime
code scorer-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components_dispatch,
)

_RGB_255_DOMAIN_EPS = 1e-3


def _validate_rgb_255_domain(name: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach()
    if not torch.isfinite(detached).all():
        raise ValueError(f"{name} must contain finite RGB values")
    min_value = float(detached.min().item())
    max_value = float(detached.max().item())
    if min_value < -_RGB_255_DOMAIN_EPS or max_value > 255.0 + _RGB_255_DOMAIN_EPS:
        raise ValueError(
            f"{name} must be in [0, 255]; got min={min_value} max={max_value}"
        )
    if max_value > 0.0 and max_value <= 1.0:
        raise ValueError(
            f"{name} appears to be unit-domain RGB; expected [0, 255] scorer input"
        )


@dataclass(frozen=True)
class Z7PredictiveCodingLossWeights:
    """Contest-shaped Z7 score-aware predictive-coding weights."""

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    beta_ib: float = 1.0
    ib_scale: float = 1e-3
    contest_normalizer: float = 37_545_489.0


class Z7GruPredictiveCodingScoreAwareLoss(torch.nn.Module):
    """Z7 score-aware recurrent predictive-coding Lagrangian.

    The scorer term uses ``score_pair_components_dispatch`` so both PoseNet and
    SegNet see the same ``preprocess_input`` pathway as other full-renderer
    substrates. The beta-IB term is the existing Z7 local residual/smoothness
    pressure, now composed with scorer-domain seg/pose terms.
    """

    def __init__(
        self,
        *,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: Z7PredictiveCodingLossWeights,
    ) -> None:
        super().__init__()
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights

    def forward(
        self,
        *,
        reconstructed_rgb_0: torch.Tensor,
        reconstructed_rgb_1: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        archive_bytes_proxy: torch.Tensor,
        residuals: torch.Tensor,
        latents: torch.Tensor,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden for Z7 score-aware loss"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")
        if residuals.dim() != 2:
            raise ValueError(
                f"residuals must be 2-D (num_pairs, latent_dim); got "
                f"{tuple(residuals.shape)}"
            )
        if latents.dim() != 2:
            raise ValueError(
                f"latents must be 2-D (num_pairs, latent_dim); got "
                f"{tuple(latents.shape)}"
            )

        for name, tensor in (
            ("reconstructed_rgb_0", reconstructed_rgb_0),
            ("reconstructed_rgb_1", reconstructed_rgb_1),
            ("gt_rgb_0", gt_rgb_0),
            ("gt_rgb_1", gt_rgb_1),
        ):
            _validate_rgb_255_domain(name, tensor)

        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        rgb_0 = reconstructed_rgb_0
        rgb_1 = reconstructed_rgb_1
        if self.training and noise_std > 0.0:
            rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std)
            rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std)

        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)
        seg_term, pose_term = score_pair_components_dispatch(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )
        pose_sqrt = torch.sqrt(pose_term.clamp(min=1e-12))
        residual_norm = residuals.pow(2).mean()
        latent_smoothness = (
            (latents[1:] - latents[:-1]).pow(2).mean()
            if latents.shape[0] > 1
            else latents.new_tensor(0.0)
        )
        ib_term = self.weights.beta_ib * self.weights.ib_scale * (
            residual_norm + latent_smoothness
        )
        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * pose_sqrt
            + ib_term
        )
        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "residual_norm": residual_norm.detach(),
            "latent_smoothness": latent_smoothness.detach(),
            "ib_term": ib_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "Z7GruPredictiveCodingScoreAwareLoss",
    "Z7PredictiveCodingLossWeights",
]
