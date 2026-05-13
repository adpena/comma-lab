"""SABOR score-aware Lagrangian.

L = alpha * B / N + beta * d_seg + gamma * sqrt(d_pose) + delta * boundary_consistency

where:

* ``B`` is the SABOR archive byte-count proxy (closed-form upper bound from
  the boundary pixel count + class_means + decoder state_dict bytes).
* ``d_seg``, ``d_pose`` are the contest SegNet/PoseNet distortion terms
  computed via the canonical ``score_pair_components`` (Catalog #164) helper,
  which routes through ``preprocess_input`` correctly per CLAUDE.md scorer
  contract.
* ``boundary_consistency`` is an auxiliary regularizer measuring how much
  the rendered boundary RGB matches the stored boundary-RGB (since the
  boundary RGB is stored verbatim, but the refinement decoder may attenuate
  it via the residual sigmoid path — we want the boundary fidelity to
  be high). Small ``delta`` (default 0.1) makes this a soft regularizer.

Per CLAUDE.md "HNeRV parity discipline" lesson L6: score-domain Lagrangian
(NOT weight-domain proxies like rel_err²).
Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE": ``apply_eval_roundtrip=True``
is the only acceptable training mode.

PR106 r2 operating point analysis (CLAUDE.md "SegNet vs PoseNet importance"):
at frontier operating points (pose_avg ≤ 3.4e-5), the pose marginal value
is 2.71x SegNet's. The default ``pose_weight_scale=1.0`` preserves the
contest formula; the trainer can ramp it up if empirical pose_avg drops
below the crossover threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components,
)


@dataclass(frozen=True)
class SaborBoundaryOnlyLossWeights:
    """The (alpha, beta, gamma, delta) of the score-domain Lagrangian."""

    alpha_rate: float = 25.0
    """Rate term weight. Contest score = (alpha_rate * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet term weight."""

    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    """PoseNet term weight. Contest default is sqrt(10) times sqrt(d_pose)."""

    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""

    delta_boundary_consistency: float = 0.1
    """Boundary-consistency regularizer weight. Small by default.

    Encourages the refinement decoder to PRESERVE the high-fidelity boundary
    RGB rather than attenuate it. Without this term the refinement decoder
    can saturate the boundary toward its initialization rather than the
    stored boundary RGB.
    """

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py."""


class SaborBoundaryOnlyScoreAwareLoss(torch.nn.Module):
    """Lagrangian as a torch Module for SABOR substrate.

    Mirrors the canonical substrate-loss contract (per
    ``tac.substrates.score_aware_common.score_pair_components``) with an
    extra boundary-consistency regularizer.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: SaborBoundaryOnlyLossWeights,
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
        boundary_mask: torch.Tensor,
        boundary_rgb_target: torch.Tensor,
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian.

        Args:
            rgb_0, rgb_1: rendered frame pair, ``(B, 3, H, W)``, in ``[0, 1]``.
            gt_rgb_0, gt_rgb_1: GT frame pair in ``[0, 1]`` (the
                ``score_pair_components`` helper handles the scale internally
                via ``preprocess_input``).
            archive_bytes_proxy: scalar tensor — closed-form upper bound on
                the archive byte count.
            boundary_mask: ``(B, 2, H, W)`` boolean — True where boundary.
            boundary_rgb_target: ``(B, 2, 3, H, W)`` float in ``[0, 1]`` — the
                target boundary RGB (used for the consistency term only).
            apply_eval_roundtrip: must be True per CLAUDE.md.
            noise_std: per-batch noise STD (reserved; default 0.5).

        Returns:
            ``(loss_total, parts)``.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        # Lazy import per the canonical pattern.
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        if self.training and noise_std > 0.0:
            rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std / 255.0)
            rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std / 255.0)

        # The scorer contract expects RGB in [0, 255]; multiply by 255 for the
        # roundtrip + score path. The canonical helper handles preprocess.
        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0 * 255.0)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1 * 255.0)
        gt_rgb_0_scaled = gt_rgb_0 * 255.0
        gt_rgb_1_scaled = gt_rgb_1 * 255.0

        seg_term, pose_term = score_pair_components(
            seg_scorer=self.seg_scorer,
            pose_scorer=self.pose_scorer,
            rgb_0_rt=rgb_0_rt,
            rgb_1_rt=rgb_1_rt,
            gt_rgb_0=gt_rgb_0_scaled,
            gt_rgb_1=gt_rgb_1_scaled,
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )

        # Boundary consistency: encourage rendered RGB to match stored
        # boundary RGB at boundary pixels.
        # rgb_0/rgb_1 are (B, 3, H, W); stack to (B, 2, 3, H, W) to align.
        rendered_pair = torch.stack([rgb_0, rgb_1], dim=1)
        mask_chw = boundary_mask.unsqueeze(2).expand_as(rendered_pair)
        boundary_diff = (rendered_pair - boundary_rgb_target).abs() * mask_chw.float()
        denom = mask_chw.float().sum().clamp(min=1.0)
        boundary_term = boundary_diff.sum() / denom

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + self.weights.delta_boundary_consistency * boundary_term
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "boundary_term": boundary_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "SaborBoundaryOnlyLossWeights",
    "SaborBoundaryOnlyScoreAwareLoss",
]
