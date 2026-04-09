"""Scorer interface for task-aware codec post-filters.

A Scorer wraps one or more frozen perception networks and provides:
  - Forward pass: compute distortion between filtered and ground-truth frames
  - Saliency: compute per-pixel gradient magnitude for loss weighting
  - Score formula: combine distortion terms into the final competition metric

The Scorer is frozen — its parameters are never updated. The post-filter
learns to minimize the Scorer's output.

Example::

    scorer = Scorer.from_comma_challenge(
        posenet_path="models/posenet.safetensors",
        segnet_path="models/segnet.safetensors",
    )
    pose_dist, seg_dist = scorer.distortion(filtered_pair, gt_pair)
    score = scorer.score(pose_dist, seg_dist, rate)
    saliency = scorer.posenet_saliency(gt_frames)
"""
from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn as nn


class Scorer(Protocol):
    """Protocol for a frozen scorer that evaluates frame quality."""

    def distortion(
        self,
        filtered: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute (pose_distortion, seg_distortion) for a frame pair.

        Args:
            filtered: (1, 2, H, W, 3) uint8 or float filtered pair
            ground_truth: (1, 2, H, W, 3) uint8 or float GT pair

        Returns:
            (pose_dist, seg_dist) as floats
        """
        ...

    def score(
        self,
        pose_dist: float,
        seg_dist: float,
        rate: float,
    ) -> float:
        """Compute the competition score from distortion + rate.

        Default formula: 100 * seg_dist + sqrt(10 * pose_dist) + 25 * rate
        """
        ...

    def posenet_saliency(
        self,
        gt_frames: list[torch.Tensor],
        device: str = "cpu",
    ) -> torch.Tensor:
        """Compute per-pixel PoseNet gradient saliency on GT frames.

        Returns: (N, H, W) float tensor of gradient magnitudes.
        """
        ...


def comma_score(pose_dist: float, seg_dist: float, rate: float) -> float:
    """Comma.ai video compression challenge score formula.

    score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate

    Lower is better.
    """
    return 100.0 * seg_dist + math.sqrt(10.0 * pose_dist) + 25.0 * rate


def score_sensitivity(pose_dist: float) -> dict[str, float]:
    """Compute marginal sensitivities at the current operating point.

    Returns dict with d(score)/d(seg), d(score)/d(pose), d(score)/d(rate),
    and the leverage ratio seg/pose.
    """
    d_seg = 100.0
    d_pose = math.sqrt(10.0) / (2.0 * math.sqrt(pose_dist)) if pose_dist > 0 else float("inf")
    d_rate = 25.0  # actually 1.0 since rate = 25*raw_rate in the formula
    return {
        "d_score_d_seg": d_seg,
        "d_score_d_pose": d_pose,
        "d_score_d_rate": d_rate,
        "seg_pose_leverage": d_seg / d_pose if d_pose > 0 else float("inf"),
    }
