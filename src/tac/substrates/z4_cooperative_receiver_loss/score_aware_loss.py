"""Score-aware cooperative-receiver Lagrangian for the Z4 substrate.

Per Atick & Redlich 1990 cooperative-receiver theory, the optimal compressor
for a known decoder-coupled receiver does NOT minimize H(X) (pixel-MSE
proxy) but rather H(X | f_R(X)) where f_R is the scorer's reachable
statistics. For the contest scorer R = SegNet + PoseNet, the Lagrangian is:

    L = α · B/N + β_seg · d_seg + γ_pose · sqrt(d_pose) + λ_pixel · MSE_pixel

where:

* ``B`` is the archive byte count (carries no gradient).
* ``N`` is the contest normalizer (37,545,489).
* ``d_seg``, ``d_pose`` are canonical scorer distortions through
  ``score_pair_components`` (Catalog #164) on the reconstructed pair vs GT.
* ``λ_pixel`` is a small pixel-MSE residual weight (default 0.0 = pure
  cooperative-receiver). Setting ``λ_pixel = 1.0`` recovers the Z3
  baseline (pixel-MSE-only training).

The cooperative-receiver hypothesis (Atick-Redlich 1990): training with
``λ_pixel = 0`` and the scorer distortions reaches a lower auth-eval score
than training with ``λ_pixel = 1, β_seg = γ_pose = 0`` because the gradient
flow is aligned with what the contest measures, not with what humans
perceive in pixel space.

The probe-disambiguator sweeps ``λ_pixel`` ∈ [0.0, 0.5, 1.0] at fixed
``β_seg + γ_pose`` to compute the regime-conditional verdict.

Mathematical contract:

    L_train = α · B/N + β_seg · d_seg + γ_pose · sqrt(d_pose) + λ_pixel · MSE_pixel

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
    score_pair_components,
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
class CooperativeReceiverLossWeights:
    """Z4 score-domain cooperative-receiver Lagrangian weights.

    Defaults match the contest formula. ``lambda_pixel`` is the residual
    pixel-MSE term weight; default 0.0 = pure cooperative-receiver.
    Set to 1.0 to recover Z3 baseline (pixel-MSE-only) for ablation.
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula."""
    lambda_pixel: float = 0.0
    """Residual pixel-MSE weight; 0 = pure cooperative-receiver, 1 = pixel-MSE-only."""
    contest_normalizer: float = 37_545_489.0


class CooperativeReceiverScoreAwareLoss(torch.nn.Module):
    """Score-aware Z4 cooperative-receiver Lagrangian as a torch Module.

    Routes scorer calls through ``score_pair_components`` per Catalog #164.

    The encoder + decoder + per-pair latent ALL carry gradients flowing
    through the canonical scorer distortion terms (no IB regularizer; the
    Z4 intervention is loss-only on the contest objective).
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: CooperativeReceiverLossWeights,
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
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the Z4 cooperative-receiver score-domain Lagrangian.

        Args:
            reconstructed_rgb_0: predicted frame_0 (B, 3, H, W) in [0, 255].
            reconstructed_rgb_1: predicted frame_1 (B, 3, H, W) in [0, 255].
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors in [0, 255].
            archive_bytes_proxy: scalar tensor with archive byte count.
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
        if self.weights.lambda_pixel < 0.0:
            raise ValueError(
                f"lambda_pixel must be >= 0; got {self.weights.lambda_pixel}"
            )
        for name, tensor in (
            ("reconstructed_rgb_0", reconstructed_rgb_0),
            ("reconstructed_rgb_1", reconstructed_rgb_1),
            ("gt_rgb_0", gt_rgb_0),
            ("gt_rgb_1", gt_rgb_1),
        ):
            _validate_rgb_255_domain(name, tensor)

        # Lazy import per canonical pattern (matches sister substrates)
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

        seg_term, pose_term = score_pair_components(
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

        # Optional pixel-MSE residual (default 0 = pure cooperative-receiver)
        if self.weights.lambda_pixel > 0.0:
            pixel_mse_0 = (rgb_0_rt - gt_rgb_0).pow(2).mean()
            pixel_mse_1 = (rgb_1_rt - gt_rgb_1).pow(2).mean()
            pixel_mse = 0.5 * (pixel_mse_0 + pixel_mse_1)
            pixel_term = self.weights.lambda_pixel * pixel_mse
        else:
            # Detached zero so gradients are not accumulated into the no-pixel path.
            pixel_term = torch.zeros((), device=rgb_0_rt.device, dtype=rgb_0_rt.dtype)

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * self.weights.pose_weight_scale * pose_sqrt
            + pixel_term
        )

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "pixel_term": pixel_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = ["CooperativeReceiverLossWeights", "CooperativeReceiverScoreAwareLoss"]
