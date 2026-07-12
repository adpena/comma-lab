# SPDX-License-Identifier: MIT
"""Static-address Torch targets for the V9 island birth stack.

The labels and RGB seed are frozen scorer/GT-derived constants.  Ladder rung
changes rebuild their eased support in NumPy, then copy into the same device
storage so a compiled or captured training step does not acquire new tensor
addresses.  No deploy payload is owned here: the protected seed remains a
training-only parameter and is exported only through its dedicated checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from tac.boundary_math.island_protection import (
    build_island_masks,
    build_island_seed,
    eased_island_masks,
    island_persistence_weight,
)
from tac.cuda_v9_controller_runtime import TorchProtectedIslandSeed


@dataclass(frozen=True)
class TorchIslandClassGeometry:
    lane_cls: int
    movable_cls: int


class TorchIslandTargetRuntime:
    """Own device-resident amplify targets and build the protected RGB seed."""

    def __init__(
        self,
        labels_phw: np.ndarray,
        *,
        lane_cls: int,
        movable_cls: int,
        flags: Mapping[str, Any],
        device: Any,
    ) -> None:
        import torch

        labels = np.asarray(labels_phw, dtype=np.int64)
        if labels.ndim != 3:
            raise ValueError(f"labels must be (P,H,W), got {labels.shape}")
        if int(lane_cls) == int(movable_cls):
            raise ValueError("lane and movable classes must be distinct")
        self.labels = labels
        self.flags = dict(flags)
        self.geometry = TorchIslandClassGeometry(int(lane_cls), int(movable_cls))
        self.device = torch.device(device)
        self.persist_kind = str(self.flags["--amplify-persist"])
        self._ladder_on = bool(self.flags.get("--ladder-island-homotopy", False))
        self._eased_on = bool(self.flags.get("--seed-island-eased", False)) or self._ladder_on
        initial_px = int(self.flags["--island-dilate-px"])
        weight, lane, movable = self._build_amplify_arrays(initial_px, initial_px)
        self.weight = torch.as_tensor(weight, device=self.device)
        self.lane_mask = torch.as_tensor(lane, device=self.device)
        self.movable_mask = torch.as_tensor(movable, device=self.device)
        self.lane_px = initial_px
        self.movable_px = initial_px

    def _build_amplify_arrays(
        self, lane_px: int, movable_px: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights: list[np.ndarray] = []
        lanes: list[np.ndarray] = []
        movables: list[np.ndarray] = []
        for labels in self.labels:
            if self._eased_on:
                masks = eased_island_masks(
                    labels,
                    self.geometry.lane_cls,
                    self.geometry.movable_cls,
                    dilate_px=int(self.flags["--island-dilate-px"]),
                    lane_px=int(lane_px),
                    movable_px=int(movable_px),
                )
            else:
                masks = build_island_masks(
                    labels,
                    self.geometry.lane_cls,
                    self.geometry.movable_cls,
                    dilate_px=int(self.flags["--island-dilate-px"]),
                )
            lane = (
                np.asarray(masks.lane_mask, dtype=bool)
                if masks.lane_mask is not None
                else np.zeros(labels.shape, dtype=bool)
            )
            movable = (
                np.asarray(masks.movable_mask, dtype=bool)
                if masks.movable_mask is not None
                else np.zeros(labels.shape, dtype=bool)
            )
            movable &= ~lane
            weight = island_persistence_weight(masks.any_mask, kind=self.persist_kind)
            support = np.asarray(weight) > 0.0
            if not np.array_equal(lane | movable, support):
                raise ValueError("lane/movable masks do not partition amplification support")
            weights.append(np.asarray(weight, dtype=np.float32))
            lanes.append(lane.astype(np.float32))
            movables.append(movable.astype(np.float32))
        return np.stack(weights), np.stack(lanes), np.stack(movables)

    def refresh_amplify_(self, *, lane_px: int, movable_px: int) -> bool:
        """Copy a changed ladder rung into stable-address device tensors."""
        import torch

        lane_px = int(lane_px)
        movable_px = int(movable_px)
        if lane_px == self.lane_px and movable_px == self.movable_px:
            return False
        weight, lane, movable = self._build_amplify_arrays(lane_px, movable_px)
        with torch.no_grad():
            self.weight.copy_(torch.as_tensor(weight, device=self.device))
            self.lane_mask.copy_(torch.as_tensor(lane, device=self.device))
            self.movable_mask.copy_(torch.as_tensor(movable, device=self.device))
        self.lane_px = lane_px
        self.movable_px = movable_px
        return True

    def build_protected_seed(self, gt_frame1_phwc: np.ndarray) -> TorchProtectedIslandSeed:
        """Build the MLX-authority training-only GT-appearance residual seed."""
        import torch
        import torch.nn.functional as F

        gt = np.asarray(gt_frame1_phwc, dtype=np.float32)
        if gt.ndim != 4 or gt.shape[0] != self.labels.shape[0] or gt.shape[-1] != 3:
            raise ValueError(
                "GT seed frames must be (P,H,W,3) with the same pair count as labels, "
                f"got {gt.shape} for labels {self.labels.shape}"
            )
        target_hw = tuple(int(x) for x in self.labels.shape[1:])
        residuals: list[np.ndarray] = []
        masks_out: list[np.ndarray] = []
        seed_eased = bool(self.flags.get("--seed-island-eased", False))
        for labels, frame in zip(self.labels, gt, strict=True):
            if tuple(frame.shape[:2]) != target_hw:
                frame = (
                    F.interpolate(
                        torch.from_numpy(frame).permute(2, 0, 1)[None],
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    .permute(1, 2, 0)
                    .numpy()
                )
            builder = eased_island_masks if seed_eased else build_island_masks
            masks = builder(
                labels,
                self.geometry.lane_cls,
                self.geometry.movable_cls,
                dilate_px=int(self.flags["--island-dilate-px"]),
            )
            seed = build_island_seed(
                frame,
                masks,
                base_render_segres=None,
                blend=float(self.flags["--seed-blend"]),
            )
            residuals.append(seed.residual)
            masks_out.append(seed.mask)
        return TorchProtectedIslandSeed(
            np.stack(residuals),
            np.stack(masks_out),
            mode=str(self.flags["--containment-mode"]),
            damp=float(self.flags["--containment-damp"]),
        ).to(self.device)


def birth_scaled_logit_offsets(
    base_offsets: Any,
    birth_multipliers: Mapping[int, float],
) -> Any:
    """Scale only watched class offsets after a birth-completion latch."""
    import torch

    multipliers = torch.ones_like(base_offsets)
    for cls, value in birth_multipliers.items():
        multipliers[int(cls)] = float(value)
    return base_offsets * multipliers


def seed_compose_weight_at_epoch(anneal_epochs: int, shape: str, epoch: int) -> float:
    """Pure MLX-authority seed transfer schedule, full weight to zero."""
    anneal = int(anneal_epochs)
    epoch = int(epoch)
    if anneal <= 0 or epoch <= 1:
        return 1.0
    if epoch >= anneal:
        return 0.0
    fraction = min(max((float(epoch) - 1.0) / (float(anneal) - 1.0), 0.0), 1.0)
    if str(shape) == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * fraction))
    return 1.0 - fraction


__all__ = [
    "TorchIslandClassGeometry",
    "TorchIslandTargetRuntime",
    "birth_scaled_logit_offsets",
    "seed_compose_weight_at_epoch",
]
