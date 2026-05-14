# SPDX-License-Identifier: MIT
"""pr101_lc_v2_clone architecture — byte-faithful PR101 HNeRV-LC-v2 decoder clone.

Mirrors ``submissions/hnerv_ft_microcodec/src/model.py`` from PR101's public
GOLD source EXACTLY (28-tensor state_dict, 6 upsample stages 6x8 -> 384x512,
dilated-conv refine residual, sigmoid * 255 RGB heads).

The clone exists so the 3 PR101 GOLD primitives Subagent C ported
(DECODER_STORAGE_ORDER + CONV4_STORAGE_PERMS + DECODER_BYTE_MAPS) can be
exercised end-to-end against an architecture whose state_dict iteration
order matches the indices the primitives reference. Other tac substrates
(sane_hnerv, balle_renderer) have different layouts and CANNOT consume the
PR101 tables directly.

PR101 architecture invariants (verified against intake clone):

* ``latent_dim = 28``
* ``base_channels = 36``
* ``eval_size = (384, 512)``
* ``N_PAIRS = 600``
* Stem: ``Linear(28, 36 * 6 * 8)``
* 6 upsample stages: ``base_h, base_w = 6, 8`` -> ``384, 512`` (2^6 = 64)
* Channels: ``[36, 36, 36, 27, 20, 18, 18]`` (taper int(C*0.75), int(C*0.58),
  int(C*0.5), int(C*0.5))
* Per stage: Conv2d(in, out*4, 3x3, padding=1) + skip (Conv 1x1 if in != out
  else Identity) + PixelShuffle(2) + ``sin(x + identity)`` bilinear-skip
* Refine: Conv2d(final, final//2, 3x3, padding=2, dilation=2) +
  Conv2d(final//2, final, 3x3, padding=1) with ``0.1 * sin`` residual
* RGB heads: ``sigmoid(rgb_*) * 255.0`` (per-frame separate)

The clone forward returns ``(B, 2, 3, 384, 512)`` in [0, 255] like PR101's
``HNeRVDecoder.forward``; the score-aware loss module casts to [0, 1] before
feeding scorers.

CLAUDE.md compliance:
* No silent device defaults (caller passes device explicitly)
* No scorer loading inside this module (score-aware loss is separate)
* No /tmp paths
* Reviewable in 30 seconds per L12 (each block <= 20 LOC)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2  # 600 pairs


@dataclass(frozen=True)
class Pr101LcV2CloneConfig:
    """Static design-time parameters mirroring PR101's HNeRVDecoder.

    All fields default to PR101 GOLD anchor values; downstream consumers
    may shrink ``base_channels`` for smoke tests but the storage-order /
    conv4-perm / byte-map primitives ASSUME PR101 layout.
    """

    latent_dim: int = 28
    """Per-pair latent dimensionality (PR101 anchor: 28)."""

    base_channels: int = 36
    """Channel taper base (PR101 anchor: 36 -> param count ~229K)."""

    base_h: int = 6
    """Stem spatial-grid height (PR101 anchor: 6)."""

    base_w: int = 8
    """Stem spatial-grid width (PR101 anchor: 8)."""

    num_upsample_blocks: int = 6
    """Number of PixelShuffle(2) stages (PR101 anchor: 6 -> 64x upsample)."""

    num_pairs: int = _PAIRS
    """Number of per-pair latents (600 for the 1200-frame contest video)."""

    output_height: int = _CONTEST_H
    """Final RGB output height."""

    output_width: int = _CONTEST_W
    """Final RGB output width."""


def _channel_taper(base_channels: int) -> tuple[int, ...]:
    """PR101's exact channel-taper sequence: [C, C, C, .75C, .58C, .5C, .5C].

    Matches PR101 source line 21 byte-for-byte:
        ``self.channels = [C, C, C, int(C * 0.75), int(C * 0.58),
                           int(C * 0.5), int(C * 0.5)]``
    """
    C = int(base_channels)
    return (
        C,
        C,
        C,
        int(C * 0.75),
        int(C * 0.58),
        int(C * 0.5),
        int(C * 0.5),
    )


class Pr101LcV2CloneSubstrate(nn.Module):
    """The byte-faithful PR101 HNeRV-LC-v2 decoder clone.

    Input: ``(B, latent_dim)`` per-pair latent tensor (float).
    Output: ``(B, 2, 3, H, W)`` RGB pair in [0, 255].

    The state_dict iteration order matches PR101's HNeRVDecoder.state_dict()
    bit-for-bit (verified against intake clone:
    ``submissions/hnerv_ft_microcodec/src/model.py``). This is the contract
    the 3 GOLD primitives rely on.
    """

    def __init__(self, cfg: Pr101LcV2CloneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.eval_size = (cfg.output_height, cfg.output_width)
        self.base_h, self.base_w = cfg.base_h, cfg.base_w
        self.channels = _channel_taper(cfg.base_channels)

        # PR101 source line 23: stem Linear(latent_dim, C * 6 * 8)
        self.stem = nn.Linear(
            cfg.latent_dim, self.channels[0] * self.base_h * self.base_w
        )

        # PR101 source lines 25-31: 6 upsample blocks with skip
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(cfg.num_upsample_blocks):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            self.blocks.append(
                nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
            )
            if in_ch != out_ch:
                self.skips.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))
            else:
                self.skips.append(nn.Identity())
        self.ps = nn.PixelShuffle(2)

        # PR101 source lines 34-38: dilated refine residual
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, kernel_size=3, padding=1),
        )

        # PR101 source lines 39-40: separate RGB heads
        self.rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Render frame-pairs from per-pair latents.

        Args:
            z: ``(B, latent_dim)`` float latents.

        Returns:
            ``(B, 2, 3, H, W)`` RGB pair in [0, 255], matching PR101 contract.
        """
        if z.dim() != 2:
            raise ValueError(f"z must be 2-D (B, latent_dim); got {tuple(z.shape)}")
        if z.shape[1] != self.cfg.latent_dim:
            raise ValueError(
                f"z latent_dim {z.shape[1]} != cfg.latent_dim {self.cfg.latent_dim}"
            )
        B = z.shape[0]

        x = self.stem(z).view(B, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips, strict=True):
            identity = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)

    def num_parameters(self) -> int:
        """Total trainable parameter count.

        Target: ~229K with PR101 anchor (base_channels=36, latent_dim=28).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
