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

from collections.abc import Mapping

import torch


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(seg_term, pose_term)`` through contest scorer preprocessing."""

    pair_pred = stage_frame_pair(rgb_0_rt, rgb_1_rt)
    pair_gt = stage_frame_pair(gt_rgb_0, gt_rgb_1)

    seg_in_pred = _preprocess(seg_scorer, pair_pred, scorer_name="SegNet")
    seg_in_gt = _preprocess(seg_scorer, pair_gt, scorer_name="SegNet")
    seg_term = seg_distortion_proxy(seg_scorer(seg_in_pred), seg_scorer(seg_in_gt))

    pose_in = _preprocess(pose_scorer, pair_pred, scorer_name="PoseNet")
    pose_target_in = _preprocess(pose_scorer, pair_gt, scorer_name="PoseNet")
    pose_out = _pose_tensor(pose_scorer(pose_in), scorer_name="PoseNet")
    pose_target = _pose_tensor(pose_scorer(pose_target_in), scorer_name="PoseNet")
    if pose_out.dim() != 2 or pose_target.dim() != 2:
        raise ScoreAwareScorerContractError(
            "PoseNet outputs must be 2D tensors shaped (B, >=6); got "
            f"{tuple(pose_out.shape)} and {tuple(pose_target.shape)}"
        )
    if pose_out.shape[1] < 6 or pose_target.shape[1] < 6:
        raise ScoreAwareScorerContractError(
            "PoseNet outputs must include at least six pose coordinates; got "
            f"{pose_out.shape[1]} and {pose_target.shape[1]}"
        )
    pose_term = ((pose_out[:, :6] - pose_target[:, :6]) ** 2).mean()
    return seg_term, pose_term


def seg_distortion_proxy(
    seg_logits_pred: torch.Tensor, seg_logits_gt: torch.Tensor
) -> torch.Tensor:
    """Soft cross-entropy between predicted and target SegNet logits."""

    log_p = torch.log_softmax(seg_logits_pred, dim=1)
    q = torch.softmax(seg_logits_gt, dim=1)
    return -(q * log_p).sum(dim=1).mean()


def _preprocess(
    scorer: torch.nn.Module,
    pair: torch.Tensor,
    *,
    scorer_name: str,
) -> torch.Tensor:
    preprocess = getattr(scorer, "preprocess_input", None)
    if not callable(preprocess):
        raise ScoreAwareScorerContractError(
            f"{scorer_name} scorer must expose preprocess_input(pair_btchw); "
            "direct scorer calls bypass the contest contract"
        )
    return preprocess(pair)


def _pose_tensor(output: object, *, scorer_name: str) -> torch.Tensor:
    if isinstance(output, Mapping):
        if "pose" not in output:
            raise ScoreAwareScorerContractError(
                f"{scorer_name} mapping output must contain a 'pose' tensor"
            )
        output = output["pose"]
    if not isinstance(output, torch.Tensor):
        raise ScoreAwareScorerContractError(
            f"{scorer_name} output must be a Tensor or mapping with 'pose'; "
            f"got {type(output).__name__}"
        )
    return output


__all__ = [
    "ScoreAwareScorerContractError",
    "score_pair_components",
    "seg_distortion_proxy",
    "stage_frame_pair",
]
