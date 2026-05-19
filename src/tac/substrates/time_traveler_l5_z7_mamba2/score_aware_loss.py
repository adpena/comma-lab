# SPDX-License-Identifier: MIT
"""Score-aware loss for the Z7-Mamba-2 substrate.

Compress-time training surface only. Routes reconstructed frame pairs
through the canonical scorer contract (CLAUDE.md Catalog #164
``score_pair_components_dispatch``) and keeps inflate/runtime
code scorer-free per CLAUDE.md "Strict scorer rule".

Sister of ``Z7GruPredictiveCodingScoreAwareLoss``; identical loss
formulation. The Mamba-2 vs GRU/LSTM swap is at the predictor level
(architecture.py), not the loss level — paired-comparison cleanliness
per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" canonical-vs-unique
decision per layer (loss is CANONICAL-ADOPT).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components_dispatch,
)

_RGB_255_DOMAIN_EPS = 1e-3


def _validate_rgb_255_domain(name: str, tensor: torch.Tensor) -> None:
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
class Z7Mamba2PredictiveCodingLossWeights:
    """Contest-shaped Z7-Mamba-2 score-aware predictive-coding weights."""

    alpha_rate: float = 25.0
    beta_seg: float = 100.0
    gamma_pose: float = CONTEST_POSE_SQRT_WEIGHT
    beta_ib: float = 1.0
    ib_scale: float = 1e-3
    contest_normalizer: float = 37_545_489.0


class Z7Mamba2PredictiveCodingScoreAwareLoss(torch.nn.Module):
    """Z7-Mamba-2 score-aware recurrent predictive-coding Lagrangian.

    Identical loss formulation to Z7-GRU/LSTM sister; predictor swap is
    at architecture level. The scorer term uses
    ``score_pair_components_dispatch`` so both PoseNet and SegNet see
    the same ``preprocess_input`` pathway as all other full-renderer
    substrates per CLAUDE.md Catalog #164 + #190.
    """

    def __init__(
        self,
        *,
        seg_scorer: torch.nn.Module,
        pose_scorer: torch.nn.Module,
        weights: Z7Mamba2PredictiveCodingLossWeights,
    ) -> None:
        super().__init__()
        self.seg_scorer = seg_scorer
        self.pose_scorer = pose_scorer
        self.weights = weights

    def train(self, mode: bool = True) -> Z7Mamba2PredictiveCodingScoreAwareLoss:
        """Toggle loss-mode while keeping frozen contest scorers in eval mode."""
        super().train(mode)
        self.seg_scorer.eval()
        self.pose_scorer.eval()
        return self

    @staticmethod
    def _normalize_scorer_chunk_size(
        scorer_chunk_size: int | None,
        *,
        batch_size: int,
    ) -> int:
        if scorer_chunk_size is None:
            return int(batch_size)
        chunk_size = int(scorer_chunk_size)
        if chunk_size <= 0:
            return int(batch_size)
        return min(chunk_size, int(batch_size))

    def score_terms(
        self,
        *,
        reconstructed_rgb_0: torch.Tensor,
        reconstructed_rgb_1: torch.Tensor,
        gt_rgb_0: torch.Tensor | None = None,
        gt_rgb_1: torch.Tensor | None = None,
        gt_pose_batch: torch.Tensor | None = None,
        gt_seg_batch: torch.Tensor | None = None,
        gt_seg_already_probs: bool | None = None,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.0,
        scorer_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return weighted ``(seg_term, pose_term)`` with optional scorer chunking.

        The chunked path is mathematically equivalent to the unchunked batch path
        for the canonical mean reductions used by the contest-shaped scorer
        proxies, but bounds peak scorer activation memory for 600-pair runs.
        """
        if not apply_eval_roundtrip:
            raise ValueError(
                "apply_eval_roundtrip=False is forbidden for Z7-Mamba-2 score-aware "
                "loss per CLAUDE.md eval_roundtrip non-negotiable"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0; got {noise_std}")

        cache_provided = (
            gt_pose_batch is not None
            and gt_seg_batch is not None
            and gt_seg_already_probs is not None
        )
        cache_partially = (
            gt_pose_batch is not None
            or gt_seg_batch is not None
            or gt_seg_already_probs is not None
        )
        if cache_partially and not cache_provided:
            raise ValueError(
                "score_terms received partial GT cache kwargs; supply "
                "gt_pose_batch, gt_seg_batch, and gt_seg_already_probs together"
            )
        if not cache_provided and (gt_rgb_0 is None or gt_rgb_1 is None):
            raise ValueError(
                "score_terms requires either raw GT RGB tensors or a complete "
                "GT scorer cache batch"
            )

        for name, tensor in (
            ("reconstructed_rgb_0", reconstructed_rgb_0),
            ("reconstructed_rgb_1", reconstructed_rgb_1),
        ):
            _validate_rgb_255_domain(name, tensor)
        if gt_rgb_0 is not None:
            _validate_rgb_255_domain("gt_rgb_0", gt_rgb_0)
        if gt_rgb_1 is not None:
            _validate_rgb_255_domain("gt_rgb_1", gt_rgb_1)
        if reconstructed_rgb_0.shape != reconstructed_rgb_1.shape:
            raise ValueError("reconstructed_rgb_0 and reconstructed_rgb_1 shapes differ")
        if gt_rgb_0 is not None and gt_rgb_1 is not None:
            if gt_rgb_0.shape != gt_rgb_1.shape:
                raise ValueError("gt_rgb_0 and gt_rgb_1 shapes differ")
            if reconstructed_rgb_0.shape[0] != gt_rgb_0.shape[0]:
                raise ValueError("reconstructed and GT batch sizes differ")
        if cache_provided:
            assert gt_pose_batch is not None
            assert gt_seg_batch is not None
            if reconstructed_rgb_0.shape[0] != gt_pose_batch.shape[0]:
                raise ValueError("reconstructed and cached GT pose batch sizes differ")
            if reconstructed_rgb_0.shape[0] != gt_seg_batch.shape[0]:
                raise ValueError("reconstructed and cached GT seg batch sizes differ")

        from tac.differentiable_eval_roundtrip import (
            apply_eval_roundtrip_during_training,
        )

        batch_size = int(reconstructed_rgb_0.shape[0])
        chunk_size = self._normalize_scorer_chunk_size(
            scorer_chunk_size,
            batch_size=batch_size,
        )
        seg_acc = reconstructed_rgb_0.new_tensor(0.0)
        pose_acc = reconstructed_rgb_0.new_tensor(0.0)
        for start in range(0, batch_size, chunk_size):
            end = min(batch_size, start + chunk_size)
            rgb_0 = reconstructed_rgb_0[start:end]
            rgb_1 = reconstructed_rgb_1[start:end]
            if self.training and noise_std > 0.0:
                rgb_0 = rgb_0 + (torch.rand_like(rgb_0) - 0.5) * (2.0 * noise_std)
                rgb_1 = rgb_1 + (torch.rand_like(rgb_1) - 0.5) * (2.0 * noise_std)
            rgb_0_rt = apply_eval_roundtrip_during_training(rgb_0)
            rgb_1_rt = apply_eval_roundtrip_during_training(rgb_1)
            seg_chunk, pose_chunk = score_pair_components_dispatch(
                seg_scorer=self.seg_scorer,
                pose_scorer=self.pose_scorer,
                rgb_0_rt=rgb_0_rt,
                rgb_1_rt=rgb_1_rt,
                gt_rgb_0=None if gt_rgb_0 is None else gt_rgb_0[start:end],
                gt_rgb_1=None if gt_rgb_1 is None else gt_rgb_1[start:end],
                gt_pose_batch=None if gt_pose_batch is None else gt_pose_batch[start:end],
                gt_seg_batch=None if gt_seg_batch is None else gt_seg_batch[start:end],
                gt_seg_already_probs=gt_seg_already_probs,
            )
            weight = float(end - start) / float(batch_size)
            seg_acc = seg_acc + seg_chunk * weight
            pose_acc = pose_acc + pose_chunk * weight
        return seg_acc, pose_acc

    def forward(
        self,
        *,
        reconstructed_rgb_0: torch.Tensor,
        reconstructed_rgb_1: torch.Tensor,
        gt_rgb_0: torch.Tensor,
        gt_rgb_1: torch.Tensor,
        archive_bytes_proxy: torch.Tensor,
        residuals: torch.Tensor,
        latents: torch.Tensor,
        apply_eval_roundtrip: bool = True,
        noise_std: float = 0.0,
        scorer_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if residuals.dim() != 2:
            raise ValueError(
                f"residuals must be 2-D (num_pairs, latent_dim); got "
                f"{tuple(residuals.shape)}"
            )
        if latents.dim() != 2:
            raise ValueError(
                f"latents must be 2-D (num_pairs, latent_dim); got "
                f"{tuple(latents.shape)}"
            )

        seg_term, pose_term = self.score_terms(
            reconstructed_rgb_0=reconstructed_rgb_0,
            reconstructed_rgb_1=reconstructed_rgb_1,
            gt_rgb_0=gt_rgb_0,
            gt_rgb_1=gt_rgb_1,
            apply_eval_roundtrip=apply_eval_roundtrip,
            noise_std=noise_std,
            scorer_chunk_size=scorer_chunk_size,
        )

        rate_term = (
            self.weights.alpha_rate
            * archive_bytes_proxy
            / self.weights.contest_normalizer
        )
        pose_sqrt = torch.sqrt(pose_term.clamp(min=1e-12))
        residual_norm = residuals.pow(2).mean()
        latent_smoothness = (
            (latents[1:] - latents[:-1]).pow(2).mean()
            if latents.shape[0] > 1
            else latents.new_tensor(0.0)
        )
        ib_term = self.weights.beta_ib * self.weights.ib_scale * (
            residual_norm + latent_smoothness
        )
        loss = (
            rate_term
            + self.weights.beta_seg * seg_term
            + self.weights.gamma_pose * pose_sqrt
            + ib_term
        )
        parts: dict[str, torch.Tensor] = {
            "rate_term": rate_term.detach(),
            "seg_term": seg_term.detach(),
            "pose_term": pose_term.detach(),
            "pose_sqrt": pose_sqrt.detach(),
            "residual_norm": residual_norm.detach(),
            "latent_smoothness": latent_smoothness.detach(),
            "ib_term": ib_term.detach(),
            "loss_total": loss.detach(),
        }
        return loss, parts


__all__ = [
    "Z7Mamba2PredictiveCodingLossWeights",
    "Z7Mamba2PredictiveCodingScoreAwareLoss",
]
