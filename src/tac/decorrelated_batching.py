"""Technique 12: ACE Decorrelated Gradient Batching.

Instead of processing frame pairs sequentially, batch random spatial crops
from different frames. This decorrelates gradients and enables higher
learning rates, accelerating convergence for per-scene overfitting.

Niantic showed this approach (from their ACE paper) significantly
accelerates convergence when overfitting to a single scene — which is
exactly our use case (overfitting to one 20-second video).

Key insight: sequential frame pairs have highly correlated gradients
(adjacent frames look almost identical). Mixing crops from distant frames
provides more diverse gradient signals per optimization step.

Usage:
    from tac.decorrelated_batching import DecorrelatedBatchSampler, crop_and_batch

    sampler = DecorrelatedBatchSampler(
        num_frames=1200,
        crop_size=(192, 256),
        crops_per_batch=8,
    )
    for batch_indices, crop_params in sampler:
        crops = crop_and_batch(frames, batch_indices, crop_params)
        # ... compute loss on crops
"""

from __future__ import annotations

import random
from typing import Iterator

import torch


class DecorrelatedBatchSampler:
    """Yields batches of random crops from random frames.

    Each batch contains `crops_per_batch` crops from randomly selected
    (non-adjacent) frames. This decorrelates gradients compared to
    sequential frame-pair processing.

    Args:
        num_pairs: total number of frame pairs available
        crop_size: (H, W) crop dimensions (must be <= frame size)
        crops_per_batch: number of crops per batch
        frame_size: (H, W) full frame dimensions
        min_frame_distance: minimum index distance between sampled frames
            to ensure decorrelation (default 10)
    """

    def __init__(
        self,
        num_pairs: int,
        crop_size: tuple[int, int] = (192, 256),
        crops_per_batch: int = 8,
        frame_size: tuple[int, int] = (384, 512),
        min_frame_distance: int = 10,
    ):
        self.num_pairs = num_pairs
        self.crop_h, self.crop_w = crop_size
        self.crops_per_batch = crops_per_batch
        self.frame_h, self.frame_w = frame_size
        self.min_frame_distance = min_frame_distance

        if self.crop_h > self.frame_h or self.crop_w > self.frame_w:
            raise ValueError(
                f"crop_size ({crop_size}) must be <= frame_size ({frame_size})"
            )

    def _sample_decorrelated_indices(self) -> list[int]:
        """Sample frame indices that are far apart in time.

        Uses a set for O(min_dist) removal instead of O(n) list filtering.
        """
        n = self.num_pairs
        min_dist = self.min_frame_distance
        indices = []
        available = set(range(n))

        for _ in range(self.crops_per_batch):
            if not available:
                available = set(range(n))

            idx = random.choice(list(available))
            indices.append(idx)

            # Remove nearby indices to enforce decorrelation — O(min_dist)
            available -= set(range(max(0, idx - min_dist), min(n, idx + min_dist + 1)))

        return indices

    def _sample_crop_params(self) -> list[tuple[int, int]]:
        """Sample random crop positions for each item in batch."""
        crops = []
        max_y = self.frame_h - self.crop_h
        max_x = self.frame_w - self.crop_w
        for _ in range(self.crops_per_batch):
            y = random.randint(0, max_y) if max_y > 0 else 0
            x = random.randint(0, max_x) if max_x > 0 else 0
            crops.append((y, x))
        return crops

    def sample(self) -> tuple[list[int], list[tuple[int, int]]]:
        """Generate one batch of decorrelated crop specifications.

        Returns:
            (indices, crop_params) where indices are frame pair indices
            and crop_params are (y, x) top-left corners of crops.
        """
        indices = self._sample_decorrelated_indices()
        crops = self._sample_crop_params()
        return indices, crops

    def __iter__(self) -> Iterator[tuple[list[int], list[tuple[int, int]]]]:
        """Infinite iterator over decorrelated batches."""
        while True:
            yield self.sample()


def crop_and_batch(
    frames: torch.Tensor,
    indices: list[int],
    crop_params: list[tuple[int, int]],
    crop_size: tuple[int, int] = (192, 256),
) -> torch.Tensor:
    """Extract and stack crops from different frames into a batch.

    Args:
        frames: (N, C, H, W) or (N, H, W, C) all frames
        indices: frame indices to crop from
        crop_params: (y, x) top-left corners for each crop
        crop_size: (crop_h, crop_w)

    Returns:
        (B, C, crop_h, crop_w) batch of crops
    """
    crop_h, crop_w = crop_size
    crops = []
    for idx, (y, x) in zip(indices, crop_params):
        frame = frames[idx]
        if frame.ndim == 3:
            # CHW heuristic: first dim is small (channels) AND last dim is
            # large (width). The old check `shape[0] in (1,3,6,12)` broke
            # when H happened to be 1, 3, 6, or 12.
            if frame.shape[0] <= 12 and frame.shape[2] > 12:  # CHW format
                crop = frame[:, y:y + crop_h, x:x + crop_w]
            else:  # HWC format
                crop = frame[y:y + crop_h, x:x + crop_w, :].permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        crops.append(crop)
    return torch.stack(crops, dim=0)


def crop_pair_and_batch(
    comp_pairs: torch.Tensor,
    gt_pairs: torch.Tensor,
    indices: list[int],
    crop_params: list[tuple[int, int]],
    crop_size: tuple[int, int] = (192, 256),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and stack crop pairs from different frame pairs.

    Works with the (N, 2, H, W, 3) HWC pair format used by tac training.

    Args:
        comp_pairs: (N, 2, H, W, 3) compressed frame pairs
        gt_pairs: (N, 2, H, W, 3) ground truth frame pairs
        indices: pair indices to crop from
        crop_params: (y, x) top-left corners for each crop
        crop_size: (crop_h, crop_w)

    Returns:
        (comp_crops, gt_crops) each (B, 2, crop_h, crop_w, 3)
    """
    crop_h, crop_w = crop_size
    comp_crops = []
    gt_crops = []

    for idx, (y, x) in zip(indices, crop_params):
        # HWC format: (2, H, W, 3)
        comp_crop = comp_pairs[idx, :, y:y + crop_h, x:x + crop_w, :]
        gt_crop = gt_pairs[idx, :, y:y + crop_h, x:x + crop_w, :]
        comp_crops.append(comp_crop)
        gt_crops.append(gt_crop)

    return torch.stack(comp_crops, dim=0), torch.stack(gt_crops, dim=0)


class DecorrelatedTrainingMixin:
    """Mixin for Trainer that adds decorrelated gradient batching.

    When enabled, each training step uses random crops from distant frames
    instead of full-frame sequential pairs. This decorrelates gradients
    and can enable higher learning rates.

    Usage: set `use_decorrelated_batching=True` and `crop_size=(192, 256)`
    in the training config (or monkey-patch the Trainer).

    Integration point: replace the pair selection in the training loop's
    inner loop with `self._get_decorrelated_batch()`.
    """

    def _init_decorrelated(
        self,
        num_pairs: int,
        crop_size: tuple[int, int] = (192, 256),
        crops_per_batch: int = 8,
        frame_size: tuple[int, int] = (384, 512),
    ):
        """Initialize the decorrelated batch sampler.

        Call this in Trainer.__init__ when decorrelated batching is enabled.
        """
        self._decorrelated_sampler = DecorrelatedBatchSampler(
            num_pairs=num_pairs,
            crop_size=crop_size,
            crops_per_batch=crops_per_batch,
            frame_size=frame_size,
        )
        self._crop_size = crop_size
        print(
            f"[trainer] Decorrelated batching enabled: "
            f"{crops_per_batch} crops of {crop_size} from {num_pairs} pairs"
        )

    def _get_decorrelated_batch(
        self,
        comp_pairs: torch.Tensor,
        gt_pairs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a decorrelated batch of cropped pairs.

        Args:
            comp_pairs: (N, 2, H, W, 3) all compressed pairs
            gt_pairs: (N, 2, H, W, 3) all GT pairs

        Returns:
            (comp_batch, gt_batch) each (B, 2, crop_h, crop_w, 3)
        """
        indices, crop_params = self._decorrelated_sampler.sample()
        return crop_pair_and_batch(
            comp_pairs, gt_pairs, indices, crop_params, self._crop_size
        )
