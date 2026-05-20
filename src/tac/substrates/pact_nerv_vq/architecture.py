# SPDX-License-Identifier: MIT
"""pact_nerv_vq architecture — Pact-NeRV-VQ (L0 SKETCH).

HNeRV-class implicit renderer with vector-quantized per-pair latents per
VQ-VAE (van den Oord 1711.00937 §3.1-3.2). The distinguishing primitive:
each per-pair latent z_e is replaced by its nearest codebook entry z_q via
straight-through estimator (Bengio 2013), the codebook updates via EMA
(van den Oord §3.2), and the commitment loss ||z_e - sg(z_q)||^2 keeps
encoder outputs close to codebook entries.

CLAUDE.md compliance: no MPS fallback, no inflate-time scorer load, no /tmp
paths, reviewable in 30 seconds per L12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_PAIRS = 600


@dataclass(frozen=True)
class PactNervVqConfig:
    """Static design-time parameters for Pact-NeRV-VQ."""

    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    codebook_size: int = 512
    """Number of discrete tokens in the codebook (van den Oord canonical: 512)."""
    codebook_decay: float = 0.99
    """EMA decay for codebook update (van den Oord canonical: 0.99)."""
    commitment_weight: float = 0.25
    """Commitment loss weight (van den Oord §3.1 canonical: 0.25)."""
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
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _DsUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.dsc(x)))


class VectorQuantizerEMA(nn.Module):
    """VQ codebook with EMA update + straight-through estimator.

    Per van den Oord 1711.00937 §3.1-3.2: nearest codebook lookup +
    straight-through estimator for gradient + EMA codebook update.

    The commitment loss term `||z_e - sg(z_q)||^2` is computed and returned
    so the trainer can add it to the score-aware Lagrangian per Catalog #6
    eval_roundtrip + score-domain canonical form.
    """

    def __init__(
        self,
        *,
        codebook_size: int,
        latent_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if codebook_size <= 0 or codebook_size > 65535:
            raise ValueError(f"codebook_size {codebook_size} out of uint16 range")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive; got {latent_dim}")
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.decay = decay
        self.epsilon = epsilon

        codebook = torch.randn(codebook_size, latent_dim) * 0.02
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", codebook.clone())

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize z_e via nearest-codebook lookup.

        Args:
            z_e: (B, latent_dim) encoder outputs.

        Returns:
            z_q: (B, latent_dim) quantized vectors (straight-through estimator).
            indices: (B,) long tensor of nearest-codebook indices.
            commitment_loss: scalar ||z_e - sg(z_q)||^2.
        """
        if z_e.dim() != 2 or z_e.shape[1] != self.latent_dim:
            raise ValueError(
                f"z_e must be (B, {self.latent_dim}); got shape {tuple(z_e.shape)}"
            )

        # Distances: (B, codebook_size)
        dists = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_e @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = dists.argmin(dim=1)
        z_q = self.codebook[indices]

        commitment_loss = F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator: gradient flows around the quantization step.
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices, commitment_loss

    @torch.no_grad()
    def ema_update(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook via EMA per van den Oord §3.2.

        Called by the trainer once per batch in training mode (no-op at eval).
        """
        encodings = F.one_hot(indices, self.codebook_size).type(z_e.dtype)
        cluster_size = encodings.sum(dim=0)
        ema_cluster = self.ema_cluster_size * self.decay + cluster_size * (1 - self.decay)
        # Laplace smoothing
        n = ema_cluster.sum()
        ema_cluster = ((ema_cluster + self.epsilon) / (n + self.codebook_size * self.epsilon)) * n
        self.ema_cluster_size = ema_cluster

        dw = encodings.t() @ z_e
        self.ema_w = self.ema_w * self.decay + dw * (1 - self.decay)
        self.codebook = self.ema_w / ema_cluster.unsqueeze(1)


class PactNervVqSubstrate(nn.Module):
    """Pact-NeRV-VQ renderer (L0 SKETCH).

    Forward:
    1. Per-pair latent z_e -> VQ codebook lookup -> z_q (straight-through).
    2. z_q -> latent_embed -> initial spatial grid.
    3. Decode through upsample chain.
    4. Final 1x1 conv heads produce rgb_0 / rgb_1.

    Per HNeRV parity L5: outputs RGB at contest camera resolution.
    """

    def __init__(self, cfg: PactNervVqConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

        self.quantizer = VectorQuantizerEMA(
            codebook_size=cfg.codebook_size,
            latent_dim=cfg.latent_dim,
            decay=cfg.codebook_decay,
        )

        self.latent_embed = nn.Linear(
            cfg.latent_dim, cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w
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
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)

        self._last_commitment_loss = torch.tensor(0.0)
        self._last_indices = torch.empty(0, dtype=torch.long)

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

    def forward(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        z_e = self.latents[pair_indices]
        z_q, indices, commitment_loss = self.quantizer(z_e)
        self._last_commitment_loss = commitment_loss
        self._last_indices = indices

        h = self.latent_embed(z_q)
        h = h.view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )

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

    @property
    def last_commitment_loss(self) -> torch.Tensor:
        """Commitment loss from the most recent forward pass (for the trainer)."""
        return self._last_commitment_loss

    @property
    def last_indices(self) -> torch.Tensor:
        """Codebook indices from the most recent forward pass."""
        return self._last_indices

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
