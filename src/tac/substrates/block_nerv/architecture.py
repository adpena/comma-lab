"""block_nerv architecture — Per-Pair Block-Decoder NeRV substrate (L0 SKETCH).

Shared SIREN+PixelShuffle base + per-pair low-rank LoRA deltas applied to
the latent embedding. The shared base mirrors ``sane_hnerv`` with slightly
smaller channels (the per-pair LoRA budget eats ~70K params); the LoRA
table is a single quantized section in the BNV1 archive grammar.

LoRA application:

    z[i] in R^latent_dim                     # per-pair latent
    h_embed[i] = W_embed @ z[i]               # shared embedding
    h_lora[i] = U[i] @ V[i] @ z[i]            # per-pair low-rank residual
                                              # U[i] in R^(embed_grid x rank)
                                              # V[i] in R^(rank x latent_dim)
    h[i] = h_embed[i] + h_lora[i]             # combined
    -> reshape -> decoder blocks -> rgb_0, rgb_1

The LoRA is applied only at the embedding stage (not inside the decoder
blocks themselves) to keep inflate.py within the 100-LOC budget.

Param-count budget:
    latents:             600 * 28 = 16_800
    shared base:         ~ 130K (SIREN+PixelShuffle smaller than sane_hnerv)
    per-pair LoRA U:     600 * (embed_grid * rank) = 600 * (36*3*4 * 2) ~ 518K
                          (TOO BIG; rank=2 LoRA on full embed-grid is wrong)

    -> Correct LoRA design at L0 SKETCH:
       per-pair scalar bias on each embed-grid position
       600 * embed_grid = 600 * (36*3*4) ~ 259K   (STILL too big)

    Final L0 SKETCH design (council pre-set 2026-05-12):
       per-pair latent-space bias only: 600 * latent_dim = 16_800
       per-pair scalar gain on each embed channel: 600 * embed_dim = 21_600
       ── per-pair budget: ~ 38_400

       total: 16_800 + 130_000 + 38_400 = ~ 185K   (under target; L1 may bump)

L0 SKETCH disclaimers per CLAUDE.md "Lane maturity registry":
- research_only=true
- DEFERRED-pending-alpha-anchor
- score_claim/promotion_eligible/ready_for_exact_eval_dispatch all False

CLAUDE.md compliance:
- No silent device defaults
- No scorer loading inside this module
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
_PAIRS = _NUM_FRAMES // 2  # 600 pairs


@dataclass(frozen=True)
class BlockNervConfig:
    """Static design-time parameters for block_nerv (L0 SKETCH)."""

    latent_dim: int = 28
    embed_dim: int = 36
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (32, 26, 22, 18, 14, 10, 8)
    sin_frequency: float = 30.0
    num_pairs: int = _PAIRS
    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W
    num_upsample_blocks: int = 7


class _SinAct(nn.Module):
    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(self.w * x)


class _UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        sin_freq: float,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch * 4, kernel_size, padding=kernel_size // 2
        )
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.shuffle(self.act(self.conv(x)))


class BlockNervSubstrate(nn.Module):
    """Block-Decoder NeRV renderer (L0 SKETCH).

    Per-pair LoRA-style residuals on the latent embedding:
      * ``self.latents``: shared per-pair latent (num_pairs, latent_dim)
      * ``self.lora_latent_bias``: per-pair latent-space bias (num_pairs, latent_dim)
      * ``self.lora_embed_gain``: per-pair channel gain (num_pairs, embed_dim)

    Forward:
      z = latents[i] + lora_latent_bias[i]
      h = W_embed @ z + bias
      h = h.view(B, embed_dim, gh, gw)
      h = h * lora_embed_gain[i].view(B, embed_dim, 1, 1)
      ... decoder blocks ...
    """

    def __init__(self, cfg: BlockNervConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        # Per-pair low-rank residuals (block contribution)
        self.lora_latent_bias = nn.Parameter(
            torch.zeros(cfg.num_pairs, cfg.latent_dim)
        )
        self.lora_embed_gain = nn.Parameter(
            torch.ones(cfg.num_pairs, cfg.embed_dim)
        )

        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        channels = [cfg.embed_dim] + list(cfg.decoder_channels)
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have "
                f"at least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_UpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

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

    def forward(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Render frame-pairs with per-pair block residuals."""
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        z_base = self.latents[pair_indices]            # (B, latent_dim)
        z_bias = self.lora_latent_bias[pair_indices]   # (B, latent_dim)
        z = z_base + z_bias

        h = self.latent_embed(z)
        h = h.view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )

        # Per-pair embed-channel gain
        gain = self.lora_embed_gain[pair_indices]      # (B, embed_dim)
        h = h * gain.view(-1, self.cfg.embed_dim, 1, 1)

        for block in self.blocks:
            h = block(h)

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
        """Total trainable parameter count (L0 target ~220K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
