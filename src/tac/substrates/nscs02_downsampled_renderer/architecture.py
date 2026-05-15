# SPDX-License-Identifier: MIT
"""NSCS02 downsampled HNeRV-style decoder (renders at 192x256).

A1 baseline architecture (`submissions/a1/src/model.py`):
  6 upsample stages from (6, 8) -> (384, 512); factor 64 = 2**6.
  Channel taper [C, C, C, 0.75C, 0.58C, 0.5C, 0.5C].
  ~229K params; renders at scorer-native (384, 512).

NSCS02 architecture (this file):
  5 upsample stages from (6, 8) -> (192, 256); factor 32 = 2**5.
  Channel taper [C, C, 0.75C, 0.58C, 0.5C, 0.5C] — drops the
  intermediate full-channel stage that produces the (192, 256) -> (384, 512)
  jump in A1.
  Predicted ~165-180K params (29-39% smaller than A1).

Per ASSUMPTIONS-CHALLENGE-AUDIT NSCS02 entry:
  Both contest scorers internally interpolate to (384, 512). The
  (1164, 874) intermediate is lossy compute. SegNet stride-2 stem
  already discards half resolution. At (192, 256), 16x pixel
  reduction; if SegNet 5-class argmax + PoseNet 6-dim pose preserved,
  ZERO d_seg / d_pose cost.

Per the standing directive
``feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md``
this file is intentionally NOT a parameter-fork of A1's HNeRVDecoder
class — it is a UNIQUE-AND-COMPLETE 5-stage variant. The two could
share base via hardcoded ``output_size`` kwarg; that path was
explicitly REJECTED to honor the standing directive. The 5-stage form
admits its own optimal channel taper choice that can be tuned WITHOUT
co-tuning A1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import (
    NSCS02_BASE_CHANNELS,
    NSCS02_LATENT_DIM,
    NSCS02_RENDER_HW,
)


class NSCS02DownsampledDecoder(nn.Module):
    """5-stage HNeRV-style decoder rendering at 192x256.

    Per-frame-pair latent (28-d) -> stem -> 5 PixelShuffle(2) stages ->
    192x256 RGB pair. Mirrors A1's structural pattern (PixelShuffle +
    bilinear-skip + sin activation) but at HALF the spatial resolution
    and ONE FEWER stage.

    The decoder produces a (B, 2, 3, 192, 256) RGB pair in float32
    in [0, 255] range. The inflate runtime is responsible for the
    final upsample to scorer-native (384, 512) AND camera-native
    (874, 1164). The training proxy uses the same upsample to keep
    the train-time scorer loss aligned with inflate-time semantics.
    """

    def __init__(
        self,
        latent_dim: int = NSCS02_LATENT_DIM,
        base_channels: int = NSCS02_BASE_CHANNELS,
        render_hw: tuple[int, int] = NSCS02_RENDER_HW,
    ) -> None:
        super().__init__()
        self.render_hw = render_hw
        self.base_h, self.base_w = 6, 8
        C = base_channels

        # 5 stages from (6, 8) to (192, 256); factor 32 = 2**5.
        # Channel taper drops A1's intermediate full-channel stage.
        self.channels = [C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
        n_stages = 5
        assert len(self.channels) == n_stages + 1, (
            f"NSCS02 channel taper must be {n_stages + 1} entries; "
            f"got {len(self.channels)} from base_channels={base_channels}"
        )

        # Stem: latent -> (C, base_h, base_w)
        self.stem = nn.Linear(latent_dim, self.channels[0] * self.base_h * self.base_w)

        # 5 PixelShuffle stages
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(n_stages):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
        self.ps = nn.PixelShuffle(2)

        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        # Separate frame-0 / frame-1 RGB heads (matches A1 pattern).
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Render a frame-pair from a per-pair latent.

        Args:
            z: per-pair latent of shape (B, latent_dim).

        Returns:
            RGB tensor of shape (B, 2, 3, 192, 256) in [0, 255] range.
        """
        B = z.shape[0]
        x = self.stem(z).view(B, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips, strict=False):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        # Output shape: (B, 2, 3, render_h, render_w) — by construction (192, 256)
        return torch.stack([f0, f1], dim=1)

    def render_then_upsample_to_scorer(
        self,
        z: torch.Tensor,
        scorer_hw: tuple[int, int] = (384, 512),
        mode: str = "bicubic",
    ) -> torch.Tensor:
        """Render at downsampled resolution then upsample for scorer comparison.

        This is the canonical NSCS02 train-time forward: render at
        (192, 256), upsample to scorer-native (384, 512). Used by
        ``score_aware_loss.compute_loss`` so the proxy loss matches
        the inflate-time semantics exactly (CLAUDE.md eval_roundtrip
        non-negotiable).
        """
        rendered = self.forward(z)
        if rendered.shape[-2:] == scorer_hw:
            return rendered
        B, T, C, _, _ = rendered.shape
        flat = rendered.reshape(B * T, C, *rendered.shape[-2:])
        upsampled = F.interpolate(flat, size=scorer_hw, mode=mode, align_corners=False)
        return upsampled.reshape(B, T, C, *scorer_hw)

    def parameter_count(self) -> int:
        """Total trainable parameter count (for archive size estimation)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["NSCS02DownsampledDecoder"]
