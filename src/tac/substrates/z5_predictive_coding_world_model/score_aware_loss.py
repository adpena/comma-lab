# SPDX-License-Identifier: MIT
"""Score-aware predictive-coding world-model Lagrangian for the Z5 substrate.

Per Rao-Ballard 1999 predictive-coding hierarchy + Atick-Redlich 1990
cooperative-receiver, the Z5 Lagrangian extends Z4 by adding a
residual-entropy term:

    L = α · B/N + β_seg · d_seg + γ_pose · sqrt(d_pose)
        + λ_residual · ||residuals||²_2 / num_pairs

where:

* ``B`` is the archive byte count (carries no gradient).
* ``N`` is the contest normalizer (37,545,489).
* ``d_seg``, ``d_pose`` are canonical scorer distortions through
  ``score_pair_components`` (Catalog #164) on the reconstructed pair vs GT.
* ``λ_residual`` is the predictive-coding bit-allocation Lagrangian. Higher
  values push the predictor to learn a better forecast (smaller residuals);
  lower values let residuals carry more independent information.

The residual-entropy term IS the Rao-Ballard predictive-coding signal: by
penalizing the L2 norm of the per-pair residuals, the predictor is forced
to forecast the next-pair latent. For stationary-ergodic driving video,
this drives the predictor toward learning an ego-motion-conditioned
world-model.

Mathematical contract:

    L_train = α · B/N
              + β_seg · d_seg
              + γ_pose · sqrt(d_pose)
              + λ_residual · ||residuals||²_2 / num_pairs

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian.
Per CLAUDE.md "eval_roundtrip — non-negotiable": ``apply_eval_roundtrip=True``
is the only acceptable training mode.
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally.
Per Catalog #164: the loss MUST route through ``score_pair_components``.
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
    """Fail closed when callers pass nonzero unit-domain RGB by mistake."""
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
class PredictiveCodingLossWeights:
    """Z5 score-domain predictive-coding Lagrangian weights.

    Defaults match the contest formula. ``lambda_residual_entropy`` controls
    how aggressively the predictor is forced to forecast next-pair latents.
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    lambda_residual_entropy: float = 1.0
    """Residual-entropy weight (Rao-Ballard 1999); 0 = no PC; 1 = canonical."""
    contest_normalizer: float = 37_545_489.0


class PredictiveCodingScoreAwareLoss(torch.nn.Module):
    """Score-aware Z5 predictive-coding Lagrangian as a torch Module.

    Routes scorer calls through ``score_pair_components`` per Catalog #164.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: PredictiveCodingLossWeights,
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
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
        gt_pose_batch: torch.Tensor | None = None,
        gt_seg_batch: torch.Tensor | None = None,
        gt_seg_already_probs: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the Z5 predictive-coding score-domain Lagrangian.

        Args:
            reconstructed_rgb_0: predicted frame_0 (B, 3, H, W) in [0, 255].
            reconstructed_rgb_1: predicted frame_1 (B, 3, H, W) in [0, 255].
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors in [0, 255].
            archive_bytes_proxy: scalar tensor with archive byte count.
            residuals: ``(num_pairs, latent_dim)`` per-pair residuals from
                the substrate; the L2 norm is the predictive-coding term.
            apply_eval_roundtrip: must be True; ValueError on False.
            noise_std: per-pixel additive noise during training.

        Returns:
            ``(loss, parts_dict)``.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")
        if self.weights.lambda_residual_entropy < 0.0:
            raise ValueError(
                f"lambda_residual_entropy must be >= 0; got "
                f"{self.weights.lambda_residual_entropy}"
            )
        if residuals.dim() != 2:
            raise ValueError(
                f"residuals must be 2-D (num_pairs, latent_dim); got "
                f"{tuple(residuals.shape)}"
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
            gt_pose_batch=gt_pose_batch,
            gt_seg_batch=gt_seg_batch,
            gt_seg_already_probs=gt_seg_already_probs,
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )
        pose_sqrt = torch.sqrt(pose_term.clamp(min=1e-12))

        # Predictive-coding residual term (Rao-Ballard 1999)
        # residuals shape: (num_pairs, latent_dim)
        residual_norm = residuals.pow(2).sum(dim=-1).mean()  # mean L2² per pair
        pc_term = self.weights.lambda_residual_entropy * residual_norm

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * self.weights.pose_weight_scale * pose_sqrt
            + pc_term
        )

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "residual_norm": residual_norm.detach(),
            "pc_term": pc_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "PredictiveCodingLossWeights",
    "PredictiveCodingScoreAwareLoss",
]
