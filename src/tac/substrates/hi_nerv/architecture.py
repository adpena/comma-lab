"""hi_nerv architecture — Hierarchical NeRV with multi-scale latents (L0 SKETCH).

Per-frame implicit renderer with a 3-scale latent pyramid. The decoder runs
coarse-to-fine: the coarse latent seeds the initial spatial grid; the mid
latent is added at the mid-decode stage; the fine latent is added near
the final RGB heads. The substrate's distinctive prior is multi-resolution
representation.

Architecture (council-approved SKETCH 2026-05-12):

    Per-pair coarse latent z_c in R^16
    Per-pair mid    latent z_m in R^20
    Per-pair fine   latent z_f in R^24
                |
                v
    Linear z_c -> 384                          # initial freq embed
                |
                v
    Reshape (1, 384, 3, 4)
                |
                v
    Block 0..2: Conv -> sin -> PixelShuffle(2) # 6x8 -> 12x16 -> 24x32 (coarse path)
                |
                + z_m via Linear -> spatial broadcast (mid-injection)
                |
                v
    Block 3..4: Conv -> sin -> PixelShuffle(2) # 48x64 -> 96x128 (mid path)
                |
                + z_f via Linear -> spatial broadcast (fine-injection)
                |
                v
    Block 5..6: Conv -> sin -> PixelShuffle(2) # 192x256 -> 384x512 (fine path)
                |
                v
    Head rgb_0 / rgb_1: Conv 3 channels each

~240K params target (3 latent scales x ~80K decoder each).

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
class HinervConfig:
    """Static design-time parameters for hi_nerv with a 3-scale latent pyramid."""

    latent_dim_coarse: int = 16
    """Coarse-scale latent dimensionality."""

    latent_dim_mid: int = 20
    """Mid-scale latent dimensionality."""

    latent_dim_fine: int = 24
    """Fine-scale latent dimensionality."""

    embed_dim: int = 64

    initial_grid_h: int = 3
    initial_grid_w: int = 4

    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)

    sin_frequency: float = 30.0

    num_upsample_blocks: int = 7

    mid_injection_block_index: int = 2
    """After which decoder block to inject z_m (0-indexed; 2 means after block 2)."""

    fine_injection_block_index: int = 4
    """After which decoder block to inject z_f."""

    num_pairs: int = _PAIRS

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W


class _SinAct(nn.Module):
    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.conv(x)))


class _LatentInjector(nn.Module):
    """Project a per-pair latent into a (B, C, H, W) spatial-additive tensor."""

    def __init__(self, latent_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, channels)

    def forward(
        self, latent: torch.Tensor, spatial_shape: tuple[int, int]
    ) -> torch.Tensor:
        # latent: (B, latent_dim) -> (B, channels) -> (B, channels, 1, 1) -> broadcast
        v = self.proj(latent)
        h, w = spatial_shape
        return v.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)


class HinervSubstrate(nn.Module):
    """Hierarchical NeRV with 3-scale latent pyramid (L0 SKETCH)."""

    def __init__(self, cfg: HinervConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if not 0 <= cfg.mid_injection_block_index < cfg.num_upsample_blocks:
            raise ValueError(
                f"mid_injection_block_index {cfg.mid_injection_block_index} "
                f"out of range [0, {cfg.num_upsample_blocks})"
            )
        if not 0 <= cfg.fine_injection_block_index < cfg.num_upsample_blocks:
            raise ValueError(
                f"fine_injection_block_index {cfg.fine_injection_block_index} "
                f"out of range [0, {cfg.num_upsample_blocks})"
            )
        if cfg.fine_injection_block_index <= cfg.mid_injection_block_index:
            raise ValueError(
                "fine_injection_block_index must be > mid_injection_block_index"
            )

        # Per-pair learned latents at 3 scales
        self.latents_coarse = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim_coarse).normal_(std=0.02)
        )
        self.latents_mid = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim_mid).normal_(std=0.02)
        )
        self.latents_fine = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim_fine).normal_(std=0.02)
        )

        # Coarse latent embedded into the initial spatial grid
        self.latent_embed = nn.Linear(
            cfg.latent_dim_coarse,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_UpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

        # Latent injectors: project to the channel count at the injection point
        mid_inject_channels = channels[cfg.mid_injection_block_index + 1]
        fine_inject_channels = channels[cfg.fine_injection_block_index + 1]
        self.mid_injector = _LatentInjector(cfg.latent_dim_mid, mid_inject_channels)
        self.fine_injector = _LatentInjector(cfg.latent_dim_fine, fine_inject_channels)

        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)

        self._siren_init()

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
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

    def forward(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        z_c = self.latents_coarse[pair_indices]
        z_m = self.latents_mid[pair_indices]
        z_f = self.latents_fine[pair_indices]

        h = self.latent_embed(z_c).view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )

        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == self.cfg.mid_injection_block_index:
                h = h + self.mid_injector(z_m, (h.shape[-2], h.shape[-1]))
            if i == self.cfg.fine_injection_block_index:
                h = h + self.fine_injector(z_f, (h.shape[-2], h.shape[-1]))

        if h.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            h = F.interpolate(
                h,
                size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear",
                align_corners=False,
            )

        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        """Total trainable parameter count (target ~240K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
