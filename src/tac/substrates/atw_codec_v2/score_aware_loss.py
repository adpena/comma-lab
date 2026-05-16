# SPDX-License-Identifier: MIT
"""ATW codec V2 score-aware Lagrangian — Atick-Redlich (canonical) + WZ + optional IB/pixel.

Per the 2026-05-16 V2 design memo §9. V2 differs from V1 in TWO substantive ways:

1. **Atick-Redlich terms route through the CANONICAL primitive**
   ``tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss``
   per Catalog #164 + Wunderkind E1 substitution recommendation. V1 hand-
   rolled the canonical scorer-preprocess routing; V2 binds the canonical
   primitive directly to extinct the bug-class breeding pattern Wunderkind
   E1 named.

2. **G1 distill supervision term** is added: ``L_distill = cross_entropy(
   distill_head(z), segnet_argmax(decoded))`` so the G1 head learns to
   predict the scorer's class assignment from the decoded latent. This is
   the operational mechanism that lets the inflate path select B3's CDF
   table conditional on the G1-predicted class WITHOUT loading SegNet.

The V2 Lagrangian (Variant B, single-knob WZ-only DEFAULT):

::

    L_ATW_v2_B = alpha * B(theta)/N           (rate from archive bytes)
               + cooperative_receiver_loss     (canonical Atick-Redlich primitive;
                                                = beta_seg * d_seg + gamma_pose * sqrt(d_pose))
               + lambda_WZ * R_WZ_residual(z)  (Wyner-Ziv residual term)
               + lambda_distill * L_distill    (G1 distill supervision)

Variant A additionally carries ``kappa_IB * I(T; Y_predicted)`` and
``lambda_pixel * MSE(decoded, GT)`` terms; the four-corner regime sweep
``(kappa_IB, lambda_WZ, lambda_pixel) in {(0,0,0), (0,1,0), (0.1,0,0), (0,0,1)}``
recovers Atick-only / ATW canonical / Tishby IB pure / Z3 baseline.

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian.
Per CLAUDE.md "eval_roundtrip — non-negotiable": apply_eval_roundtrip=True is
the only acceptable training mode.
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally.
Per Catalog #164: the loss MUST route through scorer-preprocess input
(SCORER_PREPROCESS_HANDLED_OK via canonical Atick-Redlich primitive).
"""

# SCORER_PREPROCESS_HANDLED_OK:atw-v2-routes-cooperative-receiver-loss-through-canonical-primitive-tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss-which-internally-calls-score_pair_components-per-Catalog-164
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tac.codec.cooperative_receiver.atick_redlich import (
    AtickRedlichWeights,
    CooperativeReceiverOutput,
    cooperative_receiver_loss,
)
from tac.substrates.score_aware_common import CONTEST_POSE_SQRT_WEIGHT

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
class ATWv2LossWeights:
    """ATW V2 score-domain Lagrangian weights.

    Defaults match the contest formula for (alpha, beta_seg, gamma_pose) and
    the ATW canonical Variant B mode for (kappa_IB=0, lambda_WZ=1,
    lambda_pixel=0, lambda_distill=0.1). Setting kappa_IB > 0 + lambda_pixel
    > 0 promotes to Variant A.
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula."""
    kappa_ib: float = 0.0
    """Tishby IB regularizer weight (Variant A); 0 = no IB."""
    lambda_wz: float = 1.0
    """Wyner-Ziv residual term weight; 0 = no WZ (recovers Z4 baseline)."""
    lambda_pixel: float = 0.0
    """Pixel-MSE residual weight (Variant A); 0 = pure ATW; 1 = Z3 baseline."""
    lambda_distill: float = 0.1
    """G1 distill-head cross-entropy weight; 0 = G1 supervision disabled."""
    contest_normalizer: float = 37_545_489.0


@dataclass(frozen=True)
class ATWv2LossOutput:
    """Decomposed loss + per-term diagnostics for observability surface."""

    loss: torch.Tensor
    parts: dict[str, torch.Tensor] = field(default_factory=dict)


class ATWv2ScoreAwareLoss(torch.nn.Module):
    """ATW V2 score-aware Lagrangian as a torch Module.

    Routes Atick-Redlich terms through the canonical primitive
    ``tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss``
    per Catalog #164 + Wunderkind E1.

    The encoder + decoder + per-pair latent + WZ side-info head + G1 distill
    head ALL carry gradients flowing through:

    1. The canonical cooperative-receiver scorer terms (Atick-Redlich form).
    2. The IB regularizer on ``z_residual`` (when kappa_ib > 0; Variant A).
    3. The WZ residual term penalizing ``||z_residual||^2`` (when lambda_wz > 0).
    4. The pixel-MSE residual (when lambda_pixel > 0; Variant A; Z3 baseline mode).
    5. The G1 distill cross-entropy (when lambda_distill > 0; Variant B default).
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: ATWv2LossWeights,
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
        z_residual: torch.Tensor | None = None,
        z_predicted: torch.Tensor | None = None,
        distill_class_logits: torch.Tensor | None = None,
        distill_class_targets: torch.Tensor | None = None,
        apply_eval_roundtrip: bool = True,
    ) -> ATWv2LossOutput:
        """Compute the ATW V2 score-domain Lagrangian.

        Args:
            reconstructed_rgb_0, reconstructed_rgb_1: predicted RGB pair
                tensors (B, 3, H, W) in [0, 255]. Gradient flows.
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors in [0, 255];
                NO gradient flow.
            archive_bytes_proxy: scalar tensor with archive byte count.
            z_residual: (B, latent_dim) Wyner-Ziv residual; required when
                lambda_wz > 0 OR kappa_ib > 0.
            z_predicted: (B, latent_dim) WZ prediction; diagnostic only.
            distill_class_logits: (B, num_classes) G1 head output; required
                when lambda_distill > 0.
            distill_class_targets: (B,) long class targets for G1 cross-entropy
                (typically argmax over SegNet on the rendered output).
            apply_eval_roundtrip: must be True per CLAUDE.md non-negotiable.

        Returns:
            ATWv2LossOutput carrying scalar loss + per-term diagnostics dict.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )
        if self.weights.lambda_pixel < 0.0:
            raise ValueError(
                f"lambda_pixel must be >= 0; got {self.weights.lambda_pixel}"
            )
        if self.weights.lambda_wz < 0.0:
            raise ValueError(
                f"lambda_wz must be >= 0; got {self.weights.lambda_wz}"
            )
        if self.weights.kappa_ib < 0.0:
            raise ValueError(
                f"kappa_ib must be >= 0; got {self.weights.kappa_ib}"
            )
        if self.weights.lambda_distill < 0.0:
            raise ValueError(
                f"lambda_distill must be >= 0; got {self.weights.lambda_distill}"
            )

        wz_or_ib_active = (
            self.weights.lambda_wz > 0.0 or self.weights.kappa_ib > 0.0
        )
        if wz_or_ib_active and z_residual is None:
            raise ValueError(
                "z_residual is required when lambda_wz > 0 or kappa_ib > 0; "
                "got z_residual=None"
            )
        if self.weights.lambda_distill > 0.0 and (
            distill_class_logits is None or distill_class_targets is None
        ):
            raise ValueError(
                "distill_class_logits + distill_class_targets are required "
                "when lambda_distill > 0; received "
                f"logits={distill_class_logits is not None}, "
                f"targets={distill_class_targets is not None}"
            )

        for name, tensor in (
            ("reconstructed_rgb_0", reconstructed_rgb_0),
            ("reconstructed_rgb_1", reconstructed_rgb_1),
            ("gt_rgb_0", gt_rgb_0),
            ("gt_rgb_1", gt_rgb_1),
        ):
            _validate_rgb_255_domain(name, tensor)

        # Atick-Redlich cooperative-receiver loss via canonical primitive
        # (Catalog #164 + Wunderkind E1). Internally routes through
        # score_pair_components + apply_eval_roundtrip_during_training.
        ar_weights = AtickRedlichWeights(
            beta_seg=self.weights.beta_seg,
            gamma_pose=self.weights.gamma_pose,
            pose_weight_scale=self.weights.pose_weight_scale,
        )
        ar_out: CooperativeReceiverOutput = cooperative_receiver_loss(
            reconstructed_rgb_0,
            reconstructed_rgb_1,
            gt_rgb_0,
            gt_rgb_1,
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            weights=ar_weights,
            apply_eval_roundtrip=apply_eval_roundtrip,
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )

        device_for_zero = ar_out.cooperative_loss.device
        dtype_for_zero = ar_out.cooperative_loss.dtype

        # Tishby IB regularizer (Variant A only; tractable second-order proxy).
        if self.weights.kappa_ib > 0.0 and z_residual is not None:
            ib_proxy = 0.5 * z_residual.pow(2).mean()
            ib_term = self.weights.kappa_ib * ib_proxy
        else:
            ib_term = torch.zeros((), device=device_for_zero, dtype=dtype_for_zero)

        # Wyner-Ziv residual term: penalize ||z_residual||^2.
        if self.weights.lambda_wz > 0.0 and z_residual is not None:
            wz_residual_energy = z_residual.pow(2).mean()
            wz_term = self.weights.lambda_wz * wz_residual_energy
        else:
            wz_term = torch.zeros((), device=device_for_zero, dtype=dtype_for_zero)

        # Optional pixel-MSE residual (Variant A; Z3 baseline mode).
        if self.weights.lambda_pixel > 0.0:
            # Use rendered rgb directly (already in [0, 255]); roundtripped
            # tensors used by the cooperative-receiver primitive are not
            # available here without re-routing the primitive return value.
            pixel_mse_0 = (reconstructed_rgb_0 - gt_rgb_0).pow(2).mean()
            pixel_mse_1 = (reconstructed_rgb_1 - gt_rgb_1).pow(2).mean()
            pixel_mse = 0.5 * (pixel_mse_0 + pixel_mse_1)
            pixel_term = self.weights.lambda_pixel * pixel_mse
        else:
            pixel_term = torch.zeros((), device=device_for_zero, dtype=dtype_for_zero)

        # G1 distill supervision (Variant B DEFAULT; trains scorer-class
        # distill head against SegNet's argmax on the rendered output).
        if self.weights.lambda_distill > 0.0 and distill_class_logits is not None:
            distill_ce = torch.nn.functional.cross_entropy(
                distill_class_logits, distill_class_targets
            )
            distill_term = self.weights.lambda_distill * distill_ce
        else:
            distill_term = torch.zeros((), device=device_for_zero, dtype=dtype_for_zero)

        loss = (
            rate_term
            + ar_out.cooperative_loss
            + ib_term
            + wz_term
            + pixel_term
            + distill_term
        )

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": ar_out.seg_term.detach(),
            "pose_term": ar_out.pose_term.detach(),
            "pose_sqrt": ar_out.pose_sqrt.detach(),
            "cooperative_loss": ar_out.cooperative_loss.detach(),
            "ib_term": ib_term.detach(),
            "wz_term": wz_term.detach(),
            "pixel_term": pixel_term.detach(),
            "distill_term": distill_term.detach(),
            "loss_total": loss.detach(),
        }
        return ATWv2LossOutput(loss=loss, parts=parts)


__all__ = ["ATWv2LossOutput", "ATWv2LossWeights", "ATWv2ScoreAwareLoss"]
