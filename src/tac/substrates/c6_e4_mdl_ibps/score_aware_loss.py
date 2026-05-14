"""Score-aware Lagrangian for the C6 MDL-IBPS substrate.

L = α · B/N + β_seg · d_seg + γ_pose · sqrt(d_pose) + β_ib · KL(q(z|frames) || p(z))

where:

* ``B`` is the archive byte count (encoder + decoder + latent + meta + header).
  Pass as ``archive_bytes_proxy`` (carries no gradient).
* ``N`` is the contest normalizer (37,545,489).
* ``d_seg``, ``d_pose`` are canonical scorer distortions through
  ``score_pair_components`` (Catalog #164) on the reconstructed pair vs GT.
* ``KL(q(z|frames) || p(z))`` is the variational IB upper bound on I(z; frames)
  (Tishby-Zaslavsky 2017 / Alemi et al. 2017 VIB).

Mathematical contract:

    L_train = α · B/N + β_seg · d_seg + γ_pose · sqrt(d_pose) + β_ib · KL

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian.
Per CLAUDE.md "eval_roundtrip — non-negotiable": ``apply_eval_roundtrip=True``
is the only acceptable training mode.
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally.
Per Catalog #164: the loss MUST route through ``score_pair_components``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.c6_e4_mdl_ibps.mdl_loss import kl_gaussian_to_standard_normal
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
class MDLIBPSLossWeights:
    """C6 score-domain Lagrangian weights.

    Defaults match the contest formula. ``beta_ib`` is the IB Lagrangian; the
    canonical sweep range is [0.001, 1.0] (operator-tunable).
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula."""
    beta_ib: float = 0.01
    """IB Lagrangian; controls bit budget for I(z; frames). Default 0.01."""
    contest_normalizer: float = 37_545_489.0


class MDLIBPSScoreAwareLoss(torch.nn.Module):
    """Score-aware C6 Lagrangian as a torch Module.

    Routes scorer calls through ``score_pair_components`` per Catalog #164.

    The encoder + decoder + per-pair latent ALL carry gradients. The IB
    regularizer flows through the encoder's (μ, log_σ²) output.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: MDLIBPSLossWeights,
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
        encoder_mu: torch.Tensor,
        encoder_logvar: torch.Tensor,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the C6 score-domain Lagrangian.

        Args:
            reconstructed_rgb_0: predicted frame_0 (B, 3, H, W) in [0, 255].
            reconstructed_rgb_1: predicted frame_1 (B, 3, H, W) in [0, 255].
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors in [0, 255].
            archive_bytes_proxy: scalar tensor with archive byte count.
            encoder_mu: encoder posterior mean (B, latent_dim).
            encoder_logvar: encoder posterior log-variance (B, latent_dim).
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

        # IB regularizer
        kl_per_sample = kl_gaussian_to_standard_normal(encoder_mu, encoder_logvar)
        kl_mean = kl_per_sample.mean()
        ib_term = self.weights.beta_ib * kl_mean

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * self.weights.pose_weight_scale * pose_sqrt
            + ib_term
        )

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "kl_mean": kl_mean.detach(),
            "ib_term": ib_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = ["MDLIBPSLossWeights", "MDLIBPSScoreAwareLoss"]
