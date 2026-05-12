"""Canonical scorer-input contract for substrate score-aware losses.

Every full-renderer substrate loss must compare rendered frame pairs through
the same upstream scorer pathway:

1. apply the eval-roundtrip to generated RGB in the caller;
2. stage prediction and target as ``(B, T=2, C=3, H, W)`` pairs;
3. call each scorer's ``preprocess_input`` before ``forward``.

This module exists because duplicated substrate losses regressed to passing 5D
RGB directly into SegNet and 6-channel concatenated RGB directly into PoseNet,
which crashed Modal A100 dispatches and made sibling lanes incomparable.
"""

from __future__ import annotations

import math

import torch

from tac.losses import (
    DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
    SEGMENTATION_SURROGATE_SOFT_COSINE,
    scorer_loss_terms_btchw,
)

CONTEST_RATE_WEIGHT = 25.0
CONTEST_SEG_WEIGHT = 100.0
CONTEST_POSE_SQRT_WEIGHT = math.sqrt(10.0)


class ScoreAwareScorerContractError(ValueError):
    """Raised when a scorer does not expose the contest preprocessing contract."""


def stage_frame_pair(rgb_0: torch.Tensor, rgb_1: torch.Tensor) -> torch.Tensor:
    """Return ``(B, T=2, C, H, W)`` pair tensor for upstream scorer preprocess."""

    if rgb_0.dim() != 4 or rgb_1.dim() != 4:
        raise ScoreAwareScorerContractError(
            "score-aware losses expect 4D RGB tensors (B, C, H, W) before "
            f"pair staging; got {tuple(rgb_0.shape)} and {tuple(rgb_1.shape)}"
        )
    if rgb_0.shape != rgb_1.shape:
        raise ScoreAwareScorerContractError(
            "score-aware frame pair tensors must have identical shapes; got "
            f"{tuple(rgb_0.shape)} and {tuple(rgb_1.shape)}"
        )
    if rgb_0.shape[1] != 3:
        raise ScoreAwareScorerContractError(
            "score-aware frame pair tensors must be RGB with C=3; got "
            f"{rgb_0.shape[1]} channel(s)"
        )
    return torch.stack([rgb_0, rgb_1], dim=1)


def score_pair_components(
    *,
    seg_scorer: torch.nn.Module,
    pose_scorer: torch.nn.Module,
    rgb_0_rt: torch.Tensor,
    rgb_1_rt: torch.Tensor,
    gt_rgb_0: torch.Tensor,
    gt_rgb_1: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    segmentation_surrogate: str = SEGMENTATION_SURROGATE_SOFT_COSINE,
    segmentation_temperature: float = 1.0,
    fisher_rao_eps: float = 1e-6,
    sinkhorn_max_positions_per_chunk: int | None = DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(seg_dist, pose_dist)`` through the canonical scorer loss path."""

    _require_preprocess(seg_scorer, scorer_name="SegNet")
    _require_preprocess(pose_scorer, scorer_name="PoseNet")
    pair_pred = stage_frame_pair(rgb_0_rt, rgb_1_rt)
    pair_gt = stage_frame_pair(gt_rgb_0, gt_rgb_1)
    _, pose_term, seg_term = scorer_loss_terms_btchw(
        pair_pred,
        pair_gt,
        pose_scorer,
        seg_scorer,
        class_weights=class_weights,
        segmentation_surrogate=segmentation_surrogate,
        segmentation_temperature=segmentation_temperature,
        fisher_rao_eps=fisher_rao_eps,
        sinkhorn_max_positions_per_chunk=sinkhorn_max_positions_per_chunk,
    )
    return seg_term, pose_term


def _require_preprocess(scorer: torch.nn.Module, *, scorer_name: str) -> None:
    if not callable(getattr(scorer, "preprocess_input", None)):
        raise ScoreAwareScorerContractError(
            f"{scorer_name} scorer must expose preprocess_input(pair_btchw); "
            "direct scorer calls bypass the contest contract"
        )


__all__ = [
    "CONTEST_POSE_SQRT_WEIGHT",
    "CONTEST_RATE_WEIGHT",
    "CONTEST_SEG_WEIGHT",
    "ScoreAwareScorerContractError",
    "score_pair_components",
    "stage_frame_pair",
]
