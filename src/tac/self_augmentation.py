# SPDX-License-Identifier: MIT
"""Lane SAUG (Lyra-style self-augmentation) — pose and mask perturbation.

This module implements the V1 self-augmentation lane: Gaussian pose noise
plus categorical mask flip noise on a cosine-anneal schedule from
``init -> final`` over the training run. The V1 API is distinct from
``tac.self_augmentation_v2`` (Cosmos HighSigmaStrategy), which targets
input-tensor sigma noise rather than pose/mask perturbation. The two
lanes are orthogonal and are intended to compose.

Design contract (tested by ``test_self_augmentation.py``):

* ``cosine_schedule(init, final, epoch, total_epochs)`` returns the value
  for ``epoch`` along a half-cosine that starts at ``init`` and ends at
  ``final`` (strictly monotonic for ``init != final``).
* ``SelfAugmenter.perturb_poses`` is differentiable wrt the input pose
  tensor — the random noise is an additive constant for the backward
  pass, matching ``tac.self_augmentation_v2.apply_sigma_noise_to_input``.
* ``SelfAugmenter.perturb_masks`` preserves ``shape``, ``dtype=torch.long``
  and total pixel count; it flips a random subset of pixels to
  uniformly-sampled class labels in ``[0, mask_num_classes)``.
* ``enabled=False`` is the identity function for both helpers (returns
  the same Python object so callers can rely on ``out is x``).
* ``fit_from_proxy_auth_log(path)`` ingests JSONL rows containing
  ``pose_per_dim`` arrays and sets ``self.pose_sigma_per_dim`` to the
  per-dim std of the absolute residuals (clamped strictly positive).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SelfAugmentationConfig:
    """Cosine-annealed pose+mask self-augmentation schedule.

    All sigmas are in raw pose units (radians / metres / etc) — the same
    units as the pose tensor handed to ``perturb_poses``. Mask flip
    fractions are in [0, 1].
    """

    sigma_pose_init: float = 0.01
    sigma_pose_final: float = 0.001
    mask_flip_init: float = 0.05
    mask_flip_final: float = 0.005
    mask_num_classes: int = 5
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            "sigma_pose_init",
            "sigma_pose_final",
            "mask_flip_init",
            "mask_flip_final",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative float, got {value}")
        for name in ("mask_flip_init", "mask_flip_final"):
            value = getattr(self, name)
            if value > 1.0:
                raise ValueError(f"{name} must be <= 1.0, got {value}")
        if self.mask_num_classes < 1:
            raise ValueError(
                f"mask_num_classes must be >= 1, got {self.mask_num_classes}"
            )


def cosine_schedule(
    init: float,
    final: float,
    epoch: int,
    total_epochs: int,
) -> float:
    """Return the half-cosine annealed value for ``epoch``.

    At ``epoch=0`` the value is exactly ``init``; at
    ``epoch=total_epochs - 1`` it is exactly ``final``. Intermediate
    values follow ``final + 0.5 * (init - final) * (1 + cos(pi * t))``
    where ``t`` is the normalized progress in ``[0, 1]``.
    """
    if total_epochs < 1:
        raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch}")
    if total_epochs == 1:
        return float(init)
    t = min(max(epoch / (total_epochs - 1), 0.0), 1.0)
    return float(final + 0.5 * (init - final) * (1.0 + math.cos(math.pi * t)))


class SelfAugmenter(nn.Module):
    """Apply pose Gaussian noise and mask flip noise on a cosine schedule."""

    pose_sigma_per_dim: torch.Tensor | None

    def __init__(self, config: SelfAugmentationConfig) -> None:
        super().__init__()
        self.config = config
        # Optional per-dim pose sigma loaded from a proxy/auth log. When set,
        # ``perturb_poses`` multiplies the schedule sigma by this tensor
        # broadcast against the pose dim — letting the empirical residual
        # spectrum shape the per-dim noise.
        self.pose_sigma_per_dim = None

    # ------------------------------------------------------------------
    # Pose perturbation
    # ------------------------------------------------------------------
    def perturb_poses(
        self,
        poses: torch.Tensor,
        *,
        epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        """Add per-sample Gaussian noise to a pose tensor.

        Differentiable wrt ``poses`` — the noise tensor is a constant
        for the backward pass. Returns ``poses`` unchanged (same object)
        when the augmenter is disabled.
        """
        if not self.config.enabled:
            return poses
        if poses.ndim < 1:
            raise ValueError("poses must have at least one dimension")
        if not torch.is_floating_point(poses):
            raise TypeError("perturb_poses expects a floating point tensor")

        sigma = cosine_schedule(
            self.config.sigma_pose_init,
            self.config.sigma_pose_final,
            epoch,
            total_epochs,
        )
        noise = torch.randn_like(poses) * sigma
        if self.pose_sigma_per_dim is not None:
            per_dim = self.pose_sigma_per_dim.to(
                device=poses.device, dtype=poses.dtype
            )
            if per_dim.shape != poses.shape[-1:]:
                raise ValueError(
                    "pose_sigma_per_dim must broadcast against the last "
                    f"pose dim {poses.shape[-1:]}, got {tuple(per_dim.shape)}"
                )
            noise = noise * per_dim
        return poses + noise

    # ------------------------------------------------------------------
    # Mask perturbation
    # ------------------------------------------------------------------
    def perturb_masks(
        self,
        masks: torch.Tensor,
        *,
        epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        """Flip a random subset of mask pixels to uniform class labels.

        Total pixel count, dtype (``torch.long``) and shape are
        preserved. Returns ``masks`` unchanged (same object) when the
        augmenter is disabled.
        """
        if not self.config.enabled:
            return masks
        if masks.dtype != torch.long:
            raise TypeError(
                f"perturb_masks expects dtype=torch.long, got {masks.dtype}"
            )

        flip_p = cosine_schedule(
            self.config.mask_flip_init,
            self.config.mask_flip_final,
            epoch,
            total_epochs,
        )
        if flip_p <= 0.0:
            return masks.clone()

        flip_mask = torch.rand(masks.shape, device=masks.device) < flip_p
        replacement = torch.randint(
            0,
            self.config.mask_num_classes,
            masks.shape,
            device=masks.device,
            dtype=torch.long,
        )
        return torch.where(flip_mask, replacement, masks)

    # ------------------------------------------------------------------
    # Proxy/auth log fitting
    # ------------------------------------------------------------------
    def fit_from_proxy_auth_log(self, path: str | Path) -> "SelfAugmenter":
        """Set ``pose_sigma_per_dim`` from a proxy/auth JSONL log.

        Each row in the JSONL must contain ``pose_per_dim`` (an iterable
        of floats). The fitted sigma is the per-dim std of the absolute
        residual stack, clamped to a small positive floor so downstream
        noise generation never collapses to zero.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

        rows: list[list[float]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                pose = payload.get("pose_per_dim")
                if pose is None:
                    continue
                rows.append([float(v) for v in pose])
        if not rows:
            raise ValueError(f"no pose_per_dim rows in {path}")

        residuals = torch.tensor(rows, dtype=torch.float32).abs()
        per_dim = residuals.std(dim=0, unbiased=False)
        per_dim = per_dim.clamp_min(1e-6)
        self.pose_sigma_per_dim = per_dim
        return self
