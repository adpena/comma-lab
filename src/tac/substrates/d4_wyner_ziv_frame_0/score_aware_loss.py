"""Score-aware Lagrangian for the D4 Wyner-Ziv frame-0 substrate.

L = alpha * B/N + beta * d_seg + gamma * sqrt(d_pose) + lambda_res * R_res

where:

* ``B`` is the COMBINED archive byte count (base substrate + D4 sidecar). The
  trainer is responsible for accounting both contributions; the loss term
  receives a single scalar ``archive_bytes_proxy``.
* ``N`` is the contest normalizer (37,545,489).
* ``d_seg``, ``d_pose`` are the canonical scorer distortions through
  ``score_pair_components`` (Catalog #164) on the RECONSTRUCTED pair
  ``(frame_0_pred, frame_1_unchanged)`` vs ``(gt_frame_0, gt_frame_1)``.
* ``R_res`` is the residual-magnitude penalty ``||residual_coarse||²``. This
  is a per-byte-budget regularizer that, paired with the rate term,
  encourages the motion model to do as much work as possible (because a
  better motion fit means smaller residual → smaller residual blob after
  brotli → fewer bytes charged in B).

Mathematical contract:

    L_train = alpha * B/N + beta * d_seg + gamma * sqrt(d_pose) + lambda_res * R_res

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian.
Per CLAUDE.md "eval_roundtrip — non-negotiable": ``apply_eval_roundtrip=True``
is the only acceptable training mode.
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally
(this module is the score-domain loss only).
Per Catalog #164: the loss MUST route through ``score_pair_components`` so
preprocess_input is the only path scorers see.
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
class WynerZivFrame0LossWeights:
    """The score-domain Lagrangian weights.

    Defaults match the contest formula. ``lambda_residual`` controls the
    residual-magnitude regularizer. Default 0.1 is small enough that the
    scorer-axis terms dominate but large enough that the motion model is
    encouraged to do the heavy lifting (smaller residual blob → smaller
    archive).
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula. At PR101 frontier the
    pose marginal is roughly 2.71x SegNet's (CLAUDE.md "SegNet vs PoseNet"
    section), but that is an experiment knob, not a default."""
    lambda_residual: float = 0.1
    """Weight on the per-pair residual-magnitude regularizer ||residual||²."""
    contest_normalizer: float = 37_545_489.0


class WynerZivFrame0ScoreAwareLoss(torch.nn.Module):
    """Score-aware D4 Lagrangian as a torch Module.

    The motion params (SE3 or optical-flow) AND the per-pair residual
    coarse-grid tensor BOTH carry gradient. The encoder-side residual
    quantization + brotli are non-differentiable and happen at archive build
    time (the loss uses the continuous residual-magnitude penalty as the
    differentiable proxy for the bit budget).
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: WynerZivFrame0LossWeights,
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
        residual_coarse: torch.Tensor,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the D4 score-domain Lagrangian.

        Args:
            reconstructed_rgb_0: predicted frame_0 (B, 3, H, W) in [0, 255]
                from ``synthesize_frame_0(...)``.
            reconstructed_rgb_1: frame_1 (B, 3, H, W) in [0, 255] — by D4
                construction this equals the BASE substrate's frame_1
                reconstruction (which at training time is the ground truth
                frame_1 from the video).
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors (B, 3, H, W)
                in [0, 255].
            archive_bytes_proxy: scalar tensor with the COMBINED archive byte
                count (base substrate + D4 sidecar). Carries no gradient.
            residual_coarse: per-pair residual coarse-grid parameter
                ``(N, 3, h, w)`` — the differentiable proxy for the
                residual byte budget. The L² penalty encourages smaller
                residuals, which compress to fewer bytes.
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

        # Lazy import per the canonical pattern (matches sister substrates).
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

        residual_l2 = residual_coarse.pow(2).mean()

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * self.weights.pose_weight_scale * pose_sqrt
            + self.weights.lambda_residual * residual_l2
        )

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "residual_l2": residual_l2.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "WynerZivFrame0LossWeights",
    "WynerZivFrame0ScoreAwareLoss",
]
