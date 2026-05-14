"""Score-aware Lagrangian for the Wyner-Ziv cooperative-receiver substrate.

L = alpha * B / N + beta * d_seg + gamma * sqrt(d_pose) + delta * R_WZ

where:

* ``B`` is the archive byte count (rate term — generic Shannon rate).
* ``N`` is the contest normalizer (37,545,489 frames-equivalent).
* ``d_seg``, ``d_pose`` are the canonical scorer distortions through the
  ``score_pair_components`` pipeline (Catalog #164).
* ``R_WZ`` is the Wyner-Ziv conditional rate term: the L2 distance between
  the renderer output X and the side-info predictor Y.  Per Wyner-Ziv 1976
  ``R_WZ(D) = inf I(X; U) - I(Y; U)`` — when X and Y are close, the
  conditional bits ``H(X | Y)`` shrink and the encoder transmits less.
  Penalizing the X-Y distance during training drives the renderer toward
  a representation Y can predict from the small per-pair pose code, which
  IS the Slepian-Wolf coset binning gain.

Mathematical contract:

    L_train = alpha * B/N + beta * d_seg + gamma * sqrt(d_pose) + delta * ||X - Y||_2^2

The cooperative-receiver hook: ``score_pair_components`` already runs the
eval-roundtrip-corrected RGB through the FIXED contest scorer (SegNet +
PoseNet).  This is the SAME cooperative-receiver objective as
Atick-Redlich (sister substrate ``time_traveler_l5_autonomy``); the
DIFFERENCE is the additional ``delta * R_WZ`` term + the archive grammar
that emits a coset stream instead of a raw side-info residual.

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain
Lagrangian (NOT weight-domain proxies like rel_err²).
Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE": ``apply_eval_roundtrip=True``
is the only acceptable training mode.
Per CLAUDE.md "EMA — non-negotiable": the trainer applies EMA externally
(this module is the score-domain loss only).
Per Catalog #164: the loss MUST route through ``score_pair_components``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components,
)


def _validate_rgb_255_domain(name: str, tensor: torch.Tensor) -> None:
    """Fail closed when callers pass nonzero unit-domain RGB by mistake."""

    detached = tensor.detach()
    if not torch.isfinite(detached).all():
        raise ValueError(f"{name} must contain finite RGB values")
    min_value = float(detached.min().item())
    max_value = float(detached.max().item())
    if min_value < 0.0 or max_value > 255.0:
        raise ValueError(
            f"{name} must be in [0, 255]; got min={min_value} max={max_value}"
        )
    if max_value > 0.0 and max_value <= 1.0:
        raise ValueError(
            f"{name} appears to be unit-domain RGB; expected [0, 255] scorer input"
        )


@dataclass(frozen=True)
class WynerZivLossWeights:
    """The score-domain Lagrangian weights.

    Defaults match the contest formula. ``delta_wyner_ziv`` controls the
    Wyner-Ziv conditional-rate term; default 0.5 is the design-memo
    estimate for "X-Y distance dominates rate tradeoff at this operating
    point" (PR101 frontier band).
    """

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    """Operating-point tilt — 1.0 = contest formula. PR106 r2 has 2.71x
    pose marginal, but that is an experiment knob not a default."""
    delta_wyner_ziv: float = 0.5
    """Weight on the Wyner-Ziv conditional-rate (X-Y distance) term."""
    contest_normalizer: float = 37_545_489.0


class WynerZivCooperativeReceiverLoss(torch.nn.Module):
    """Score-aware Wyner-Ziv Lagrangian as a torch Module.

    The renderer + side-info predictor + per-pair pose codes all carry
    gradient. The encoder-side coset binning is non-differentiable and
    happens at archive build time (the loss instead uses the X-Y distance
    as the differentiable proxy for the conditional-rate budget).
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: WynerZivLossWeights,
    ) -> None:
        super().__init__()
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights

    def forward(
        self,
        rgb_0: torch.Tensor,
        rgb_1: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        archive_bytes_proxy: torch.Tensor,
        *,
        side_info_y_0: torch.Tensor | None = None,
        side_info_y_1: torch.Tensor | None = None,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Wyner-Ziv Lagrangian.

        Args:
            rgb_0, rgb_1: predicted RGB pair tensors (B, 3, H, W) in [0, 255].
            gt_rgb_0, gt_rgb_1: ground-truth RGB pair tensors (B, 3, H, W) in [0, 255].
            archive_bytes_proxy: scalar tensor with the archive byte count
                (carries no gradient — rate term is non-differentiable; the
                gradient flows through seg + pose + wyner_ziv only).
            side_info_y_0, side_info_y_1: optional ``(B, 3, H, W)`` tensors
                from the side-info predictor.  When provided, the L2
                distance ``||X - Y||_2^2`` is added to the loss weighted by
                ``delta_wyner_ziv``.  Penalizing this term IS the
                Wyner-Ziv conditional-rate proxy.
            apply_eval_roundtrip: must be True per CLAUDE.md eval_roundtrip
                non-negotiable; ValueError on False.
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
            ("rgb_0", rgb_0),
            ("rgb_1", rgb_1),
            ("gt_rgb_0", gt_rgb_0),
            ("gt_rgb_1", gt_rgb_1),
        ):
            _validate_rgb_255_domain(name, tensor)

        # Lazy import per the canonical pattern.
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

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

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * self.weights.pose_weight_scale * pose_sqrt
        )

        wz_term_value: torch.Tensor | float = 0.0
        if (
            side_info_y_0 is not None
            and side_info_y_1 is not None
            and self.weights.delta_wyner_ziv > 0.0
        ):
            # Wyner-Ziv conditional-rate proxy: X-Y distance.
            # Y is in [0, 1] (sigmoid output); X (rgb_*) is in [0, 255].
            # Normalize X to [0, 1] for the distance.
            x_0 = (rgb_0 / 255.0).clamp(0.0, 1.0)
            x_1 = (rgb_1 / 255.0).clamp(0.0, 1.0)
            wz_term = (x_0 - side_info_y_0).pow(2).mean() + (
                x_1 - side_info_y_1
            ).pow(2).mean()
            loss = loss + self.weights.delta_wyner_ziv * wz_term
            wz_term_value = wz_term.detach()

        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "wyner_ziv_term": (
                wz_term_value
                if isinstance(wz_term_value, torch.Tensor)
                else torch.tensor(float(wz_term_value))
            ),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "WynerZivCooperativeReceiverLoss",
    "WynerZivLossWeights",
]
