# SPDX-License-Identifier: MIT
"""NSCS03 score-aware Lagrangian — end-to-end joint codec loss.

The CRUX of NSCS03: the joint loss is END-TO-END differentiable through
quantization (via Ballé 2017 noise relaxation) AND through the entropy
bottleneck (via the conditional-Gaussian + factorized-prior closed-form
rate). Gradient flows from SegNet/PoseNet back through the synthesis
transform g_s, through the quantized main latent y_hat, through the
analysis transform g_a, all the way to the input pair embedding.

::

    L_NSCS03 = α · B_proxy(θ)/N
             + β_seg · d_seg(decoded(quantized(g_a(x))))
             + γ_pose · sqrt(d_pose(decoded(quantized(g_a(x)))))
             + λ_R · (R_main(y_hat | σ) + R_hyper(z_hat))

where:

* B_proxy(θ)/N = differentiable rate proxy (Ballé closed-form bits/elem
  scaled to byte-budget; the NON-differentiable post-export archive bytes
  are tracked separately for logging)
* d_seg / d_pose are the contest scorer distortions, gradient-reachable
  via tac.differentiable_eval_roundtrip + patch_upstream_yuv6_globally
* R_main = -log2 N(y_hat; 0, σ²) per Ballé 2018 scale hyperprior
* R_hyper = -log2 p_z(z_hat) per the EntropyBottleneck factorized prior

UNIQUE-AND-COMPLETE design decisions vs `balle_renderer/score_aware_loss.py`:

1. The SUBSTRATE consumes (B, 6, H, W) PIXELS, not pair_indices. Score-aware
   loss must therefore stack frame_0+frame_1 into a single (B, 6, H, W)
   input tensor BEFORE calling the substrate forward.

2. The rate term R is END-TO-END differentiable through the bottleneck.
   At the joint loss level, the rate steers θ toward latent distributions
   that ACTUALLY arithmetic-code well — NOT toward post-hoc archive
   compression of arbitrary weights.

3. We must FORK the canonical scorer-preprocess (Catalog #164 + #226 +
   #228) for ONE specific reason: the joint codec has the gradient PATH
   from scorer back through the bottleneck back through the encoder. The
   canonical `score_pair_components_dispatch` works correctly here (it
   already handles preprocess_input on the scorer); we use it as-is.
   What we DON'T do is short-circuit the scorer when y_hat is rounded
   non-differentiably — that would defeat the whole point of NSCS03.

4. The conditional-Gaussian rate term σ (from h_s(z_hat)) MUST flow
   gradient back to h_s; the entropy-bottleneck z rate term MUST flow
   gradient back to h_a + g_a. The substrate's forward already returns
   `main_rate` and `hyper_rate` as scalar tensors with attached graph;
   this loss simply weights and sums them into L.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- patch_upstream_yuv6_globally must be called BEFORE scorer construction
  (caller's responsibility; documented in trainer)
- No silent CUDA fallback (caller supplies device + scorers)
- No /tmp paths
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components_dispatch,
)


@dataclass(frozen=True)
class NSCS03ScoreAwareLossWeights:
    """The (α, β, γ, λ_R) of the NSCS03 score-domain Lagrangian.

    Defaults per the assumptions-challenge-audit + Ballé 2018 reference:
    λ_R = 0.5 is the canonical "balance rate vs distortion at our operating
    point" anchor. Trainer sweeps [0.1, 1.0] in follow-up wave per the
    audit memo NSCS03 reactivation_criteria.
    """

    alpha_rate: float = 25.0
    """Rate term weight. Contest score = (alpha_rate * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet term weight."""

    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    """PoseNet term weight. Contest default is sqrt(10) times sqrt(d_pose)."""

    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""

    lambda_R: float = 0.5
    """Weight on the differentiable rate term (R_main + R_hyper).

    The audit memo NSCS03 hypothesizes that lambda_R = 0.5 balances the
    rate-axis pressure with the score-axis pressure such that the joint
    codec converges to a representation that is BOTH compressible AND
    score-preserving. lambda_R << 0.1 collapses rate (uniform latents
    that code well but render nothing); lambda_R >> 1.0 over-penalizes
    rate (substrate becomes near-lossless but archive size explodes)."""

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py (sum of uncompressed video bytes for 0.mkv)."""


class NSCS03JointScoreAwareLoss(torch.nn.Module):
    """The NSCS03 score-domain Lagrangian as a torch Module.

    Trainer usage (post-warmup, follow-up subagent)::

        # PATCH yuv6 BEFORE scorer construction per CLAUDE.md eval_roundtrip:
        from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
        patch_upstream_yuv6_globally()

        loss_fn = NSCS03JointScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet,
            weights=NSCS03ScoreAwareLossWeights(...),
        )
        ...
        x_pair = NSCS03JointCodecSubstrate.stack_frames_into_pair(gt_rgb_0, gt_rgb_1)
        recon, rate_components = model(x_pair)
        rgb_0_hat, rgb_1_hat = model.split_recon_into_frames(recon)
        loss, parts = loss_fn(
            rgb_0_hat, rgb_1_hat, gt_rgb_0, gt_rgb_1,
            archive_bytes_proxy=archive_bytes_estimate,
            rate_components=rate_components,
        )
        loss.backward()

    ``rate_components`` is the dict returned by
    ``NSCS03JointCodecSubstrate.forward`` containing ``main_rate``,
    ``hyper_rate``, ``total_rate``. These tensors carry the gradient
    that closes the END-TO-END loop: scorer → decoder → bottleneck →
    encoder.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: NSCS03ScoreAwareLossWeights,
    ) -> None:
        super().__init__()
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights

    def forward(
        self,
        rgb_0_hat: torch.Tensor,
        rgb_1_hat: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        archive_bytes_proxy: torch.Tensor,
        rate_components: dict[str, torch.Tensor],
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
        gt_pose_batch: torch.Tensor | None = None,
        gt_seg_batch: torch.Tensor | None = None,
        gt_seg_already_probs: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the NSCS03 score-domain Lagrangian.

        Returns:
            ``(loss_total, parts)`` where parts is a dict with named
            components for logging (``rate_term`` / ``seg_term`` /
            ``pose_term`` / ``rate_proxy_term`` / ``main_rate`` /
            ``hyper_rate`` / ``loss_total``).
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )
        for key in ("main_rate", "hyper_rate", "total_rate"):
            if key not in rate_components:
                raise ValueError(
                    f"rate_components missing required key {key!r}; "
                    "ensure substrate forward returned the dict"
                )
        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")

        # Lazy import to keep this module's import-time cheap
        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        if self.training and noise_std > 0.0:
            rgb_0_hat = rgb_0_hat + (torch.rand_like(rgb_0_hat) - 0.5) * (
                2.0 * noise_std
            )
            rgb_1_hat = rgb_1_hat + (torch.rand_like(rgb_1_hat) - 0.5) * (
                2.0 * noise_std
            )

        rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0_hat)
        rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1_hat)

        # Canonical scorer-preprocess routing per Catalog #164.
        # IMPORTANT (per docstring): we deliberately use the canonical helper
        # because it correctly handles SegNet's last-frame slicing AND
        # PoseNet's 6-channel YUV6 stacking AND scorer preprocess_input.
        # The end-to-end gradient path through the bottleneck is preserved
        # because rgb_0_rt and rgb_1_rt still carry their graph attachment
        # back to the substrate's decoder output. The score-aware-common
        # helper does NOT detach.
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

        # Rate term from POST-EXPORT archive bytes (NON-differentiable; just
        # a scalar logging anchor). The DIFFERENTIABLE rate pressure comes
        # from the rate_components dict below.
        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )

        # The differentiable rate proxy that ACTUALLY steers the codec.
        # main_rate + hyper_rate are bits/elem; sum gives total bits/elem.
        rate_proxy_term = (
            self.weights.lambda_R * rate_components["total_rate"]
        )

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + rate_proxy_term
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "rate_proxy_term": rate_proxy_term.detach(),
            "main_rate": rate_components["main_rate"].detach(),
            "hyper_rate": rate_components["hyper_rate"].detach(),
            "total_rate": rate_components["total_rate"].detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "NSCS03JointScoreAwareLoss",
    "NSCS03ScoreAwareLossWeights",
]
