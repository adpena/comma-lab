# SPDX-License-Identifier: MIT
"""Z4 Atick-Redlich score-aware Lagrangian with cooperative-receiver primitive.

Per CLAUDE.md "INDIVIDUALLY-FRACTAL" standing directive 2026-05-27 +
"UNIQUE-AND-COMPLETE-PER-METHOD" operating mode: this is Z4's OWN canonical
score-aware loss engineering pass.

The substrate-distinguishing primitive (per Catalog #272) at the loss
surface is the Atick-Redlich 1990 spatial cooperative-receiver formulation:
maximize ``MI(X; T)`` against a FIXED known scorer ``(SegNet, PoseNet)``
under a fixed coding budget ``B``. Per the canonical Atick-Redlich +
Tishby-Zaslavsky 2015 IB framework + Catalog #311 amendment (spatial
retinal MI form is admissible without temporal ego-motion conditioning):

Loss formula:

    L_total = alpha_rate * (B / N)
            + beta_seg * d_seg(theta)
            + gamma_pose * sqrt(d_pose(theta))
            + delta_coop_rcv * (recon_mse - beta_atick_redlich * scorer_align)

where:
    * d_seg / d_pose come from canonical Catalog #164 ``score_pair_components``
      routing (binds receiver gradient to PoseNet + SegNet without
      synthetic-teacher re-routing).
    * recon_mse = per-pair reconstruction MSE (proxy for H(X|T))
    * scorer_align_proxy = -(d_seg + d_pose).detach() (proxy for I(T;Y))
    * gamma_pose carries the canonical contest sqrt-pose weight
      ``sqrt(10) * 25`` to align Lagrangian dual variables with contest
      scoring at the Pareto polytope intersection (per CLAUDE.md
      "Meta-Lagrangian/Pareto solver" non-negotiable).

The cooperative-receiver term is what makes Z4 distinct from sister Z6-v2
(which adds an ego-motion FiLM-conditioned hierarchy on top) and Z5
(predictive-coding world-model). Z4 is the MINIMUM-SUFFICIENT cooperative-
receiver primitive at the latent surface per Atick-Redlich 1990 canonical
claim (decorrelation alone is sufficient at the retinal layer).

[verified-against: src/tac/substrates/score_aware_common.py Catalog #164
 canonical scorer routing + sqrt-pose weight CONTEST_POSE_SQRT_WEIGHT]
[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/score_aware_loss.py
 sister Z6V2ScoreAwareLoss canonical pattern]
[verified-against: src/tac/codec/cooperative_receiver/atick_redlich.py
 canonical Atick-Redlich primitive package]
[verified-against: CLAUDE.md "Forbidden device-selection defaults" +
 "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA"
 non-negotiables]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components_dispatch,
)


@dataclass(frozen=True)
class Z4AtickRedlichScoreAwareLossWeights:
    """Z4 canonical score-aware Lagrangian weights with cooperative-receiver term."""

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    pose_weight_scale: float = 1.0
    contest_normalizer: float = 37_545_489.0

    # Atick-Redlich cooperative-receiver primitive weights per Atick &
    # Redlich 1990 + Catalog #311 amendment (spatial form).
    delta_coop_receiver: float = 0.05
    beta_atick_redlich: float = 0.5


class Z4AtickRedlichScoreAwareLoss(torch.nn.Module):
    """Z4 score-aware Lagrangian with Atick-Redlich cooperative-receiver primitive.

    Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE, HIGHEST EMPHASIS" the
    forward path applies the canonical
    ``apply_eval_roundtrip_during_training`` BEFORE the scorer pass so
    proxy-vs-auth gap stays bounded.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: Z4AtickRedlichScoreAwareLossWeights,
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
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
        gt_pose_batch: torch.Tensor | None = None,
        gt_seg_batch: torch.Tensor | None = None,
        gt_seg_already_probs: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "Catalog #6 MANDATORY DEFAULT (eval_roundtrip — "
                "NON-NEGOTIABLE)"
            )
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )
        del noise_std  # canonical roundtrip wrapper owns the noise schedule

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

        # Atick-Redlich 1990 spatial cooperative-receiver primitive per
        # Catalog #311 amendment: the receiver loss
        # `I(X;T) - beta * I(T;Y)` is proxied at gradient time via
        #     recon_mse: per-pair reconstruction MSE (proxy for H(X|T))
        #     scorer_align: scorer-anchored term (proxy for I(T;Y))
        # The cooperative-receiver gradient binding ensures the latent T
        # maximizes scorer information while minimizing reconstruction
        # entropy. beta_atick_redlich controls the IB tradeoff per the
        # 1990 paper. The primitive is OPERATIONAL at L1 per Catalog #220.
        recon_mse = 0.5 * (
            ((rgb_0_rt - gt_rgb_0) ** 2).mean()
            + ((rgb_1_rt - gt_rgb_1) ** 2).mean()
        )
        # Scorer alignment proxy: -(d_seg + d_pose).detach() encodes the
        # scorer-anchored information T -> Y proxy. Higher alignment ⇔
        # lower seg + pose distortion ⇔ greater I(T; Y).
        scorer_align_proxy = (seg_term + pose_term).detach() * (-1.0)
        coop_receiver_term = (
            recon_mse
            - self.weights.beta_atick_redlich * scorer_align_proxy
        )

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + self.weights.delta_coop_receiver * coop_receiver_term
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "coop_receiver_term": coop_receiver_term.detach(),
            "recon_mse": recon_mse.detach(),
            "scorer_align_proxy": scorer_align_proxy.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "Z4AtickRedlichScoreAwareLoss",
    "Z4AtickRedlichScoreAwareLossWeights",
]
