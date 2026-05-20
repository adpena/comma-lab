# SPDX-License-Identifier: MIT
"""nirvana architecture — NIRVANA (L0 SKETCH).

Per-patch implicit renderer with PATCH_GRID_H x PATCH_GRID_W tiles. Operator
5-tier fit-ranking MODERATE-HIGH FIT ⭐⭐⭐⭐: patch-wise specialization stacks
orthogonally with global NeRV substrates; the adaptive scheduler (training-
time only) gives more compute to high-error patches.

Literature anchor: Maiya et al. CVPR 2024 NIRVANA (paper-ID literature
reference per BUILD task #1090). The patch-wise paradigm draws on patch-CNN
literature; the adaptive scheduling component is the novel CVPR 2024
contribution.

Architecture (council-approved SKETCH 2026-05-20):

    Per-pair latent z in R^16
       |
       +-- Patch grid: 4x4 = 16 patches
       |
       v
    For each patch p in [0, NUM_PATCHES):
        h_p = PatchDecoder([z; patch_embedding[p]])
              (shared decoder weights; patch_embedding distinguishes slots)
       |
       v
    Assemble: rgb_full = stitch(rgb_patches, grid_shape)
       |
       v
    Head rgb_0 / rgb_1: 1x1 Conv on rgb_full

The L0 SCAFFOLD uses SHARED decoder weights across patches (each patch
slot is distinguished via a small learned patch_embedding that's
concatenated to z before decoding). This keeps the rate term tight; a
future L1+ variant with per-patch decoders would inflate the archive
~NUM_PATCHES x but might enable per-patch specialization.

The adaptive scheduler is NOT in the L0 substrate forward path — it lives
in the training loop only. The smoke trainer uses uniform patch sampling.

CLAUDE.md compliance:
- No silent device defaults
- No scorer load
- No /tmp paths
- Reviewable in 30 seconds per L12
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2


@dataclass(frozen=True)
class NirvanaConfig:
    """Static design-time parameters for nirvana."""

    latent_dim: int = 16
    """Per-pair latent dimensionality (shared across all patches)."""

    patch_embed_dim: int = 8
    """Per-patch learned embedding dim (distinguishes patch slots)."""

    patch_grid_h: int = 4
    """Number of patches vertically."""

    patch_grid_w: int = 4
    """Number of patches horizontally."""

    embed_dim: int = 48
    """Channels of the initial spatial-grid embedding (per-patch)."""

    decoder_channels: tuple[int, ...] = (40, 32, 24, 16, 12)
    """Per-block output channels for the per-patch decoder."""

    sin_frequency: float = 30.0

    num_upsample_blocks: int = 5
    """Number of PixelShuffle(2) blocks per patch; 5 -> 3x4 patch grid -> 96x128 patch -> assemble to 384x512."""

    initial_patch_grid_h: int = 3
    """Initial spatial-grid height per patch decoder."""

    initial_patch_grid_w: int = 4

    num_pairs: int = _PAIRS

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W


class _SinAct(nn.Module):
    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _DepthSepConv(nn.Module):
    """Depthwise-3x3 + pointwise-1x1, SIREN-friendly (mirrors ds_nerv)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _DsUpBlock(nn.Module):
    """DepthSep -> sin -> PixelShuffle(2)."""

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.dsc(x)))


class NirvanaSubstrate(nn.Module):
    """NIRVANA renderer (L0 SKETCH).

    Input: pair index ``i in [0, num_pairs)``
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1].

    The forward path:
    1. Look up per-pair latent z.
    2. For each patch p in [0, num_patches): decode (z, patch_embed[p]) into
       a per-patch RGB.
    3. Stitch patches into the full (H, W) frame.
    4. Final 1x1 conv heads produce rgb_0 / rgb_1.
    """

    def __init__(self, cfg: NirvanaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_patches = cfg.patch_grid_h * cfg.patch_grid_w
        if cfg.output_height % cfg.patch_grid_h != 0:
            raise ValueError(
                f"output_height {cfg.output_height} not divisible by "
                f"patch_grid_h {cfg.patch_grid_h}"
            )
        if cfg.output_width % cfg.patch_grid_w != 0:
            raise ValueError(
                f"output_width {cfg.output_width} not divisible by "
                f"patch_grid_w {cfg.patch_grid_w}"
            )
        self.patch_h = cfg.output_height // cfg.patch_grid_h
        self.patch_w = cfg.output_width // cfg.patch_grid_w

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

        # Per-patch learned embeddings (distinguish patch slots in the shared
        # decoder; the shared decoder + patch embedding is the rate-saving
        # CARGO-CULTED choice vs per-patch independent decoders).
        self.patch_embeddings = nn.Parameter(
            torch.empty(self.num_patches, cfg.patch_embed_dim).normal_(std=0.02)
        )

        # Shared per-patch decoder: input dim = latent_dim + patch_embed_dim
        self.combined_embed = nn.Linear(
            cfg.latent_dim + cfg.patch_embed_dim,
            cfg.embed_dim * cfg.initial_patch_grid_h * cfg.initial_patch_grid_w,
        )

        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[cfg.num_upsample_blocks]
        # Per-patch RGB heads (output 3 channels per frame_0 / frame_1)
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)

        self._siren_init()

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    if m.groups > 1:
                        fan_in = m.kernel_size[0] * m.kernel_size[1]
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def _decode_patches(self, z: torch.Tensor) -> torch.Tensor:
        """Decode all patches for a batch of latents.

        Args:
            z: (B, latent_dim)

        Returns:
            (B, num_patches, final_ch, patch_h_native, patch_w_native)
        """
        B = z.shape[0]
        # Concatenate each (z_b, patch_embed_p) for every batch + patch.
        # Result shape: (B * num_patches, latent_dim + patch_embed_dim)
        z_expanded = z.unsqueeze(1).expand(-1, self.num_patches, -1)
        patch_emb_expanded = self.patch_embeddings.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([z_expanded, patch_emb_expanded], dim=-1)
        combined_flat = combined.reshape(B * self.num_patches, -1)

        h = self.combined_embed(combined_flat)
        h = h.view(
            B * self.num_patches,
            self.cfg.embed_dim,
            self.cfg.initial_patch_grid_h,
            self.cfg.initial_patch_grid_w,
        )
        for block in self.blocks:
            h = block(h)
        # h shape: (B * num_patches, final_ch, h_native, w_native)
        return h

    def _stitch_patches(
        self, patch_rgb: torch.Tensor, B: int
    ) -> torch.Tensor:
        """Stitch (B*num_patches, 3, patch_h, patch_w) -> (B, 3, H, W).

        patch_rgb is interpolated to (patch_h, patch_w) if its native
        shape differs (e.g. PixelShuffle blocks produce one of several
        sizes per config).
        """
        # Resize each patch to the canonical patch_h x patch_w.
        if patch_rgb.shape[-2:] != (self.patch_h, self.patch_w):
            patch_rgb = F.interpolate(
                patch_rgb,
                size=(self.patch_h, self.patch_w),
                mode="bilinear",
                align_corners=False,
            )
        # Reshape (B*num_patches, 3, patch_h, patch_w) -> (B, num_patches, 3, patch_h, patch_w)
        # -> (B, 3, patch_grid_h, patch_grid_w, patch_h, patch_w)
        # -> (B, 3, H, W)
        gh = self.cfg.patch_grid_h
        gw = self.cfg.patch_grid_w
        ph = self.patch_h
        pw = self.patch_w
        patch_rgb = patch_rgb.view(B, gh, gw, 3, ph, pw)
        # Reorder to (B, 3, gh, ph, gw, pw) then flatten spatial:
        patch_rgb = patch_rgb.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patch_rgb.view(B, 3, gh * ph, gw * pw)

    def forward(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        B = pair_indices.shape[0]
        z = self.latents[pair_indices]
        h_patches = self._decode_patches(z)
        # h_patches shape: (B * num_patches, final_ch, h_native, w_native)

        rgb_0_patches = torch.sigmoid(self.head_rgb_0(h_patches))
        rgb_1_patches = torch.sigmoid(self.head_rgb_1(h_patches))

        rgb_0 = self._stitch_patches(rgb_0_patches, B)
        rgb_1 = self._stitch_patches(rgb_1_patches, B)
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        """Total trainable parameter count (target ~180K with shared decoder)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
