"""cool_chic score-aware Lagrangian — L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose).

Per CLAUDE.md HNeRV parity discipline L6 + L8: the score-domain Lagrangian must
backprop through SegNet/PoseNet via the differentiable eval-roundtrip + the
patched yuv6 (PR #95/#106 monkey-patch contract). Rate term combines (a) an
arithmetic-coder upper-bound proxy on the synthesis + AR prior state-dicts (not
implemented in this L0 SKETCH — anchored to a closed-form Ballé R(D) bound) plus
(b) the AR-prior log-density of the per-frame latents, which IS differentiable
and IS the substrate's distinguishing rate primitive.

The PR106 r2 operating point (pose_avg ~ 3.4e-5) flips marginal-value to 2.71x
pose-vs-seg; the loss exposes ``pose_weight_scale`` for the trainer to ramp up
when empirical pose_avg drops below 2.5e-4.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std reserved for forward-compat (current STE is deterministic)
- No silent CUDA fallback (caller supplies device + scorers)
- No /tmp paths
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components,
)


@dataclass(frozen=True)
class ScoreAwareLossWeights:
    """The (alpha, beta, gamma) of the score-domain Lagrangian."""

    alpha_rate: float = 25.0
    """Rate term weight. Contest score = (alpha_rate * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet term weight."""

    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    """PoseNet term weight. Contest default is sqrt(10) times sqrt(d_pose)."""

    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py."""

    ar_rate_weight: float = 1.0
    """Weight on the AR-prior negative-log-likelihood term (bits proxy)."""


class CoolChicScoreAwareLoss(torch.nn.Module):
    """Lagrangian as a torch Module for cool_chic substrate.

    Mirror of sane_hnerv's SaneHnervScoreAwareLoss with the addition of the
    AR-prior rate term consumed via ``ar_log_prob`` (returned from
    ``CoolChicSubstrate.compute_ar_log_prob()``).
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: ScoreAwareLossWeights,
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
        ar_log_prob: torch.Tensor,
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian.

        Args:
            rgb_0, rgb_1: rendered frame pair, ``(B, 3, H, W)``, in [0, 1].
            gt_rgb_0, gt_rgb_1: ground-truth frame pair (already roundtripped
                upstream by the trainer if needed).
            archive_bytes_proxy: scalar tensor — closed-form upper bound on the
                archive byte count (Ballé R(D) bound). Non-differentiable
                post-export bytes are NOT used as the proxy.
            ar_log_prob: scalar tensor = sum log p(z_t | z_{t-1}) — the AR-prior
                negative-log-likelihood is added to rate (smaller log_p = more bits).

        Returns:
            ``(loss_total, parts)`` where parts has the named components for logging.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

        del noise_std  # reserved for forward-compat
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

        # Base archive-bytes rate term (proxy)
        rate_term_archive = (
            self.weights.alpha_rate * archive_bytes_proxy / self.weights.contest_normalizer
        )
        # AR-prior NLL contributes bits to rate (sign: -log p, in nats)
        # Convert to bits proxy by dividing by ln(2). The trainer can scale via ar_rate_weight.
        import math
        ar_term = (-ar_log_prob / math.log(2.0)) * self.weights.ar_rate_weight
        rate_term = rate_term_archive + ar_term / self.weights.contest_normalizer

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
        )

        parts = {
            "rate_term": rate_term.detach(),
            "rate_term_archive": rate_term_archive.detach(),
            "ar_term": ar_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts
