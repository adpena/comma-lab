"""self_compress_nn score-aware Lagrangian (δ).

    L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ)) + λ_mdl·commit_loss(θ)

The Fields-medal grand council 2026-05-12 score-domain Lagrangian per
CLAUDE.md HNeRV parity discipline L6, extended for δ with the MDL/codebook
commit term:

* `d_seg(θ)`: contest SegNet distortion, gradient-reachable via
  ``tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training``
  + ``patch_upstream_yuv6_globally`` (PR #95/#106 monkey-patch contract).
* `d_pose(θ)`: contest PoseNet distortion via the same wiring.
* `B(θ)`: post-export archive size. For δ this is ``CODEBOOK_BYTES +
  num_quantized_weights * log2(K) / 8`` plus latents — significantly smaller
  than the α counterpart because we never store the full weight tensors.
* `commit_loss`: the SUM of VQ-VAE commit terms across all quantized
  layers (returned by ``SelfCompressNnSubstrate.forward()``). Each commit
  term has the canonical form ``||sg[q(x)] - x||_2^2 + β·||sg[x] - q(x)||_2^2``
  per van den Oord 2017. As ``commit_loss → 0``, the un-clustered weights
  approach the codebook centroids — the score-aware-trained MDL minimum.

The PR106 r2 operating point (pose_avg ~ 3.4e-5) flips marginal-value to
2.71× pose-vs-seg; the loss exposes ``pose_weight_scale`` like α/β/γ.

CLAUDE.md compliance:
- eval_roundtrip=True (NON-NEGOTIABLE)
- noise_std=0.5 (Hotz STE fix)
- No silent CUDA fallback (caller supplies device + scorers)
- No /tmp paths
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import score_pair_components


@dataclass(frozen=True)
class SelfCompressNnScoreAwareLossWeights:
    """The (α, β, γ, λ_mdl) of the δ score-domain Lagrangian.

    Defaults per the Phase 2 council SKETCH; tunable via the follow-up
    subagent's trainer CLI.
    """

    alpha_rate: float = 25.0
    """Rate term weight. Contest score = (α_rate * archive_bytes) / N."""

    beta_seg: float = 100.0
    """SegNet term weight."""

    gamma_pose: float = 1.0
    """PoseNet term weight (multiplied by sqrt(d_pose))."""

    pose_weight_scale: float = 1.0
    """At PR106-r2 operating point (pose_avg ~ 3.4e-5), set to 2.71."""

    lambda_mdl: float = 0.25
    """Weight on the codebook MDL/commit term.

    Balances "drive weights toward codebook centroids (better
    quantization, cheaper bytes)" vs "let the renderer keep enough
    expressivity to score." 0.25 mirrors VQ-VAE 2017's β. The
    follow-up subagent should sweep [0.05, 1.0]."""

    contest_normalizer: float = 37_545_489.0
    """N from contest evaluate.py."""


class SelfCompressNnScoreAwareLoss(torch.nn.Module):
    """The δ Lagrangian as a torch Module.

    Trainer usage (post-SC++-Stage-1-anchor, follow-up subagent)::

        # PATCH yuv6 BEFORE scorer construction per CLAUDE.md eval_roundtrip
        # rule:
        # tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()

        loss_fn = SelfCompressNnScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet,
            weights=SelfCompressNnScoreAwareLossWeights(...),
        )
        ...
        rgb_0, rgb_1, commit_loss = model(pair_indices)
        loss, parts = loss_fn(
            rgb_0, rgb_1, gt_rgb_0, gt_rgb_1, archive_bytes_proxy, commit_loss,
        )
        loss.backward()
        # AFTER optimizer.step(), call ``model.codebook.ema_step(grouped_w, idx)``
        # to update the codebook centroids via persistent EMA per VQ-VAE 2017.
    """

    def __init__(
        self,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: SelfCompressNnScoreAwareLossWeights,
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
        commit_loss: torch.Tensor,
        *,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the score-domain Lagrangian for the δ substrate.

        Returns:
            ``(loss_total, parts)`` where parts is a dict with named
            components for logging (``rate_term``, ``seg_term``,
            ``pose_term``, ``mdl_term``, ``loss_total``).
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden per CLAUDE.md "
                "'eval_roundtrip — non-negotiable'"
            )

        # Lazy import to keep this module's import-time cheap
        from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

        # ``noise_std`` is reserved for a future variant of the eval-roundtrip
        # (currently the canonical ``apply_eval_roundtrip_during_training``
        # uses ``Uint8STE`` deterministically — no additive noise). Kept as a
        # kwarg for forward-compat so the trainer's CLI surface remains stable.
        del noise_std  # unused at this revision; STE is deterministic
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
            self.weights.alpha_rate * archive_bytes_proxy / self.weights.contest_normalizer
        )

        mdl_term = self.weights.lambda_mdl * commit_loss

        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose
            * self.weights.pose_weight_scale
            * torch.sqrt(pose_term.clamp(min=1e-12))
            + mdl_term
        )

        parts = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "mdl_term": mdl_term.detach(),
            "commit_loss_raw": commit_loss.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts
