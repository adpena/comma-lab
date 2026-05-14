"""Canonical scorer-input contract for substrate score-aware losses.

Every full-renderer substrate loss must compare rendered frame pairs through
the same upstream scorer pathway:

1. apply the eval-roundtrip to generated RGB in the caller;
2. stage prediction and target as ``(B, T=2, C=3, H, W)`` pairs;
3. call each scorer's ``preprocess_input`` before ``forward``.

This module exists because duplicated substrate losses regressed to passing 5D
RGB directly into SegNet and 6-channel concatenated RGB directly into PoseNet,
which crashed Modal A100 dispatches and made sibling lanes incomparable.

Optimization-aware variant
--------------------------

:func:`score_pair_components_with_cache` is the canonical cache-aware
sister. It accepts precomputed GT scorer outputs (built once at trainer
init via :func:`tac.training_optimization.build_gt_scorer_cache`) and
skips the GT scorer forward in the hot loop. Mathematically identical to
:func:`score_pair_components` (the cache stores the same tensors a
direct GT forward would produce; the target video + scorer weights are
frozen). Per the optimization-opportunities audit 2026-05-14 §3.1 this
is the single largest substrate-trainer speedup.
"""

from __future__ import annotations

import math

import torch

from tac.losses import (
    DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
    SEGMENTATION_SURROGATE_SOFT_COSINE,
    scorer_loss_terms_btchw,
)
from tac.losses.core import scorer_loss_terms_cached_btchw

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


def score_pair_components_with_cache(
    *,
    seg_scorer: torch.nn.Module,
    pose_scorer: torch.nn.Module,
    rgb_0_rt: torch.Tensor,
    rgb_1_rt: torch.Tensor,
    gt_pose_batch: torch.Tensor,
    gt_seg_batch: torch.Tensor,
    gt_seg_already_probs: bool,
    class_weights: torch.Tensor | None = None,
    segmentation_surrogate: str = SEGMENTATION_SURROGATE_SOFT_COSINE,
    segmentation_temperature: float = 1.0,
    fisher_rao_eps: float = 1e-6,
    sinkhorn_max_positions_per_chunk: int | None = DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache-aware sister of :func:`score_pair_components` (O1 from audit).

    Same scorer pathway, same mathematical loss. The difference is the
    GT scorer forward is REPLACED by an indexed cache lookup against
    precomputed PoseNet + SegNet outputs. Per the optimization audit
    2026-05-14 §3.1 this saves ~50% per-step scorer compute for substrates
    whose own forward is small (NeRV / HNeRV / SIREN / Cool-Chic).

    Canonical usage::

        from tac.training_optimization import build_gt_scorer_cache
        from tac.substrates.score_aware_common import (
            score_pair_components_with_cache,
        )

        # at trainer init:
        cache = build_gt_scorer_cache(
            target_pixels=target_pairs,
            posenet=pose_scorer,
            segnet=seg_scorer,
            device=device,
            segmentation_temperature=1.0,
        )
        print(cache.summary_line())

        # in hot loop, per batch:
        gt_pose_batch, gt_seg_batch = cache.lookup(idx, device=device)
        seg_term, pose_term = score_pair_components_with_cache(
            seg_scorer=seg_scorer,
            pose_scorer=pose_scorer,
            rgb_0_rt=rendered_frame_0,
            rgb_1_rt=rendered_frame_1,
            gt_pose_batch=gt_pose_batch,
            gt_seg_batch=gt_seg_batch,
            gt_seg_already_probs=cache.seg_already_probs,
            segmentation_temperature=cache.segmentation_temperature,
        )

    Args:
        seg_scorer: SegNet module (must expose ``preprocess_input``).
        pose_scorer: PoseNet module (must expose ``preprocess_input``).
        rgb_0_rt: predicted frame 0, ``(B, 3, H, W)``, eval-roundtrip
            already applied by caller.
        rgb_1_rt: predicted frame 1, ``(B, 3, H, W)``, eval-roundtrip
            already applied.
        gt_pose_batch: cached PoseNet output for the matching target
            pairs from :meth:`tac.training_optimization.GTScorerCache.lookup`.
            Shape ``(B, T=2, P)`` where P>=6.
        gt_seg_batch: cached SegNet output for the matching target
            pairs. Shape ``(B, K, H, W)``.
        gt_seg_already_probs: True if ``gt_seg_batch`` holds softmax
            probabilities (canonical when ``segmentation_temperature`` ==
            1.0 to skip a redundant softmax); False if it holds raw
            logits. MUST match the cache's
            :attr:`tac.training_optimization.GTScorerCache.seg_already_probs`
            attribute or the loss will be inconsistent.
        class_weights / segmentation_surrogate / segmentation_temperature
        / fisher_rao_eps / sinkhorn_max_positions_per_chunk: forwarded
            to :func:`scorer_loss_terms_cached_btchw` unchanged.

    Returns:
        ``(seg_term, pose_term)`` tensors matching
        :func:`score_pair_components` (gradient-bearing on the predicted
        path).

    Raises:
        ScoreAwareScorerContractError: If either scorer is missing the
            ``preprocess_input`` contract method.
    """

    _require_preprocess(seg_scorer, scorer_name="SegNet")
    _require_preprocess(pose_scorer, scorer_name="PoseNet")
    pair_pred = stage_frame_pair(rgb_0_rt, rgb_1_rt)
    _, pose_term, seg_term = scorer_loss_terms_cached_btchw(
        pair_pred,
        gt_pose_batch,
        gt_seg_batch,
        pose_scorer,
        seg_scorer,
        class_weights=class_weights,
        segmentation_surrogate=segmentation_surrogate,
        segmentation_temperature=segmentation_temperature,
        fisher_rao_eps=fisher_rao_eps,
        sinkhorn_max_positions_per_chunk=sinkhorn_max_positions_per_chunk,
        gt_seg_already_probs=gt_seg_already_probs,
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
    "score_pair_components_with_cache",
    "stage_frame_pair",
]
