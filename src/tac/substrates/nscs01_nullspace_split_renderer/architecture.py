# SPDX-License-Identifier: MIT
"""NSCS01 Nullspace Split Renderer architecture.

Per ASSUMPTIONS-CHALLENGE-AUDIT 2026-05-15 entry SA02_segnet_only_last_frame:
SegNet at upstream/modules.py:108 slices ``x[:, -1, ...]`` and so frame[0] is
in SegNet's structural nullspace. NSCS01 exploits this with a TWO-HEAD
renderer architecture:

* ``frame_0_head``: small (~30K params), coarser RGB output, optimized ONLY
  for PoseNet (frame[0] is in SegNet's nullspace, so SegNet quality of
  frame[0] is irrelevant to score).
* ``frame_1_head``: full (~150K params), high-detail RGB, optimized for
  BOTH SegNet (last-frame slice) AND PoseNet (frame-1 contribution).

Joint forward takes a shared per-pair latent ``z[i]`` and emits both
``frame_0[i]`` and ``frame_1[i]``.

Per HNeRV parity discipline lesson L12 (single-LOC-per-LOC review): the
architecture is INTENTIONALLY simple — the score gain comes from the split
(different bit-widths + different gradient routing), not from architectural
novelty.

Catalog #124 archive-grammar 8 fields are declared inline in __init__.py at
module level so the AST walker observes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

NUM_PAIRS: int = 600
"""Contest pair count."""

CAMERA_H: int = 384
"""Renderer output height (matches SegNet preprocess + scorer expected H)."""

CAMERA_W: int = 512
"""Renderer output width."""

# Archive byte targets (substrate-engineering scope; no external base substrate).
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 60_000
TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 120_000


@dataclass(frozen=True)
class NullspaceSplitConfig:
    """Static design-time parameters for the NSCS01 split renderer.

    Args:
        latent_dim: per-pair latent vector dimension (default 16).
        head0_bits: bit-width for frame_0_head weight quantization (4/6/8).
            Default 4 — frame_0 is PoseNet-only and tolerates lower fidelity.
        head1_bits: bit-width for frame_1_head weight quantization (6/8).
            Default 8 — frame_1 must be SegNet-argmax-stable.
        latent_bits: per-latent quantization (8/12). Default 12.
        head0_base_channels: base channel count for frame_0_head (~30K params).
        head1_base_channels: base channel count for frame_1_head (~150K params).
        num_pairs: contest pair count (default 600).
    """

    latent_dim: int = 16
    head0_bits: int = 4
    head1_bits: int = 8
    latent_bits: int = 12
    head0_base_channels: int = 16
    head1_base_channels: int = 48
    num_pairs: int = NUM_PAIRS

    def __post_init__(self) -> None:
        if self.head0_bits not in (4, 6, 8):
            raise ValueError(
                f"head0_bits must be 4/6/8; got {self.head0_bits}"
            )
        if self.head1_bits not in (6, 8):
            raise ValueError(
                f"head1_bits must be 6/8; got {self.head1_bits}"
            )
        if self.latent_bits not in (8, 12):
            raise ValueError(
                f"latent_bits must be 8/12; got {self.latent_bits}"
            )
        if self.latent_dim < 1 or self.latent_dim > 256:
            raise ValueError(
                f"latent_dim must be 1..256; got {self.latent_dim}"
            )
        if self.num_pairs < 1:
            raise ValueError(f"num_pairs must be >= 1; got {self.num_pairs}")


class _SmallRenderHead(nn.Module):
    """Small render head for frame_0 (PoseNet-only target).

    Architecture: latent → linear → reshape → pixel-shuffle upsample x3 →
    RGB. Total params target: ~30K at base_channels=16.
    """

    def __init__(self, latent_dim: int, base_channels: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        # Output spatial: 384x512. Init grid 12x16 = 1/32 → 3 stages of x2 → 96x128
        # then bilinear up to 384x512 (no extra params; cheap).
        self._init_h = 12
        self._init_w = 16
        c0 = base_channels * 4  # 64
        c1 = base_channels * 2  # 32
        c2 = base_channels      # 16
        self.stem = nn.Linear(latent_dim, c0 * self._init_h * self._init_w)
        # Each stage: Conv2d(c_in, c_out * 4, 3, padding=1) + PixelShuffle(2)
        self.up0 = nn.Conv2d(c0, c1 * 4, 3, padding=1)
        self.up1 = nn.Conv2d(c1, c2 * 4, 3, padding=1)
        self.up2 = nn.Conv2d(c2, c2 * 4, 3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.rgb = nn.Conv2d(c2, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, 3, 384, 512) in [0, 255]."""
        b = z.shape[0]
        x = self.stem(z).view(b, -1, self._init_h, self._init_w)
        x = torch.relu(self.ps(self.up0(x)))
        x = torch.relu(self.ps(self.up1(x)))
        x = torch.relu(self.ps(self.up2(x)))
        # x is now (B, base_channels, 96, 128); upsample to camera res via
        # bilinear (no params).
        x = nn.functional.interpolate(x, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
        rgb = torch.sigmoid(self.rgb(x)) * 255.0
        return rgb


class _LargeRenderHead(nn.Module):
    """Larger render head for frame_1 (SegNet + PoseNet target).

    Architecture: latent → linear → reshape → pixel-shuffle upsample x3 →
    refinement → RGB. Total params target: ~150K at base_channels=48.
    """

    def __init__(self, latent_dim: int, base_channels: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self._init_h = 12
        self._init_w = 16
        c0 = base_channels * 4  # 192
        c1 = base_channels * 2  # 96
        c2 = base_channels      # 48
        self.stem = nn.Linear(latent_dim, c0 * self._init_h * self._init_w)
        self.up0 = nn.Conv2d(c0, c1 * 4, 3, padding=1)
        self.up1 = nn.Conv2d(c1, c2 * 4, 3, padding=1)
        self.up2 = nn.Conv2d(c2, c2 * 4, 3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.refine = nn.Conv2d(c2, c2, 3, padding=1)
        self.rgb = nn.Conv2d(c2, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, 3, 384, 512) in [0, 255]."""
        b = z.shape[0]
        x = self.stem(z).view(b, -1, self._init_h, self._init_w)
        x = torch.relu(self.ps(self.up0(x)))
        x = torch.relu(self.ps(self.up1(x)))
        x = torch.relu(self.ps(self.up2(x)))
        x = nn.functional.interpolate(x, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
        x = torch.relu(self.refine(x))
        rgb = torch.sigmoid(self.rgb(x)) * 255.0
        return rgb


class NullspaceSplitRenderer(nn.Module):
    """The NSCS01 substrate: two render heads sharing per-pair latents.

    Forward: ``forward(pair_indices) → (frame_0, frame_1)`` where each is
    ``(B, 3, CAMERA_H, CAMERA_W)`` in [0, 255] RGB domain.

    The per-pair latent ``z[i]`` is shared between the two heads; each head
    has its own learned mapping from latent to RGB. This is the structural
    expression of "frame[0] is in SegNet's nullspace": the head that emits
    frame[0] is free to be smaller/coarser; only PoseNet sees its output.

    OOM safety: callers MUST mini-batch via ``pair_indices`` for the full
    600-pair contest dataset — full forward at 384x512 + batch 600 exceeds
    T4 (14.56 GB) per the D4 OOM anchor (Catalog #218).
    """

    def __init__(self, cfg: NullspaceSplitConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_0_head = _SmallRenderHead(
            latent_dim=cfg.latent_dim,
            base_channels=cfg.head0_base_channels,
        )
        self.frame_1_head = _LargeRenderHead(
            latent_dim=cfg.latent_dim,
            base_channels=cfg.head1_base_channels,
        )
        # Per-pair latents (the "video memory"); init small.
        self.latents = nn.Parameter(
            torch.randn(cfg.num_pairs, cfg.latent_dim) * 0.1
        )

    def reconstruct_pair(
        self,
        pair_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render (frame_0, frame_1) for the requested pair indices.

        Args:
            pair_indices: optional ``(B,)`` long tensor in
                ``[0, cfg.num_pairs)``. If None, full ``cfg.num_pairs`` are
                rendered (forbidden on T4 for cfg.num_pairs >= 256 per
                Catalog #218 OOM gate; mini-batch instead).

        Returns:
            ``(frame_0, frame_1)`` each shape ``(B, 3, CAMERA_H, CAMERA_W)``
            in [0, 255] RGB domain.
        """
        if pair_indices is None:
            z = self.latents
        else:
            if pair_indices.dim() != 1:
                raise ValueError(
                    f"pair_indices must be 1-D; got shape "
                    f"{tuple(pair_indices.shape)}"
                )
            if pair_indices.numel() == 0:
                raise ValueError("pair_indices must be non-empty")
            min_idx = int(pair_indices.min().item())
            max_idx = int(pair_indices.max().item())
            if min_idx < 0 or max_idx >= self.cfg.num_pairs:
                raise ValueError(
                    f"pair_indices range [{min_idx}, {max_idx}] outside "
                    f"[0, {self.cfg.num_pairs})"
                )
            # index_select preserves gradients into the selected rows.
            z = self.latents.index_select(0, pair_indices)
        frame_0 = self.frame_0_head(z)
        frame_1 = self.frame_1_head(z)
        return frame_0, frame_1


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "NullspaceSplitConfig",
    "NullspaceSplitRenderer",
    "_LargeRenderHead",
    "_SmallRenderHead",
]
