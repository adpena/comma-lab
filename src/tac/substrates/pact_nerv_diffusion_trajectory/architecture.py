# SPDX-License-Identifier: MIT
"""pact_nerv_diffusion_trajectory architecture — Pact-NeRV-DiffusionTrajectory (L0 SKETCH).

HNeRV-class implicit renderer with a per-pair latent-diffusion-trajectory
predictor (Rombach et al. 2022 LDM arXiv:2112.10752 + Blattmann et al. 2023
arXiv:2304.08818 video latent diffusion). Lightweight implementation per the
PACT-NERV-ULTIMATE LOC budget: 5-step depth-2 MLP predictor refines stored
Gaussian-noise seeds into per-pair latents.

CLAUDE.md compliance: no MPS fallback, no inflate-time scorer load, no /tmp.
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
class PactNervDiffusionTrajectoryConfig:
    """Static design-time parameters for Pact-NeRV-DiffusionTrajectory."""

    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    diffusion_num_timesteps: int = 5
    """Number of diffusion refinement steps (CARGO-CULTED at L0; Stage 1 sweep)."""
    diffusion_predictor_hidden: int = 32
    """Hidden dim of the per-step predictor MLP."""
    noise_schedule: str = "linear"
    """Noise schedule: 'linear' or 'cosine' (Nichol-Dhariwal 2102.09672)."""
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


class DiffusionTrajectoryPredictor(nn.Module):
    """Lightweight per-step diffusion-trajectory predictor.

    Per Rombach 2112.10752 §3.1 latent diffusion: ship a Gaussian seed +
    iteratively denoise to the target latent. The full UNet predictor is
    LOC-prohibitive at L0; this lightweight per-step depth-2 MLP suffices
    for the L0 SCAFFOLD demonstration of the canonical pattern.

    Each step t in [0, T) applies:
        delta_t = MLP_t(z_t, time_embedding(t))
        z_{t+1} = sqrt(alpha_t) * z_t + sqrt(1 - alpha_t) * delta_t

    where alpha_t comes from the noise schedule (linear or cosine).
    """

    def __init__(
        self,
        *,
        latent_dim: int = 24,
        num_timesteps: int = 5,
        hidden: int = 32,
        time_embed_dim: int = 8,
        noise_schedule: str = "linear",
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive; got {latent_dim}")
        if num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive; got {num_timesteps}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive; got {hidden}")
        if noise_schedule not in ("linear", "cosine"):
            raise ValueError(
                f"noise_schedule must be 'linear' or 'cosine'; got {noise_schedule}"
            )
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.hidden = hidden
        self.time_embed_dim = time_embed_dim
        self.noise_schedule = noise_schedule

        # Time embedding: scalar t -> time_embed_dim vector
        self.time_embed = nn.Linear(1, time_embed_dim)

        # Per-step depth-2 MLP predictors (parameter-efficient at L0)
        self.predictors = nn.ModuleList()
        for _ in range(num_timesteps):
            self.predictors.append(
                nn.Sequential(
                    nn.Linear(latent_dim + time_embed_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, latent_dim),
                )
            )

        # Pre-compute noise schedule alphas
        if noise_schedule == "linear":
            alphas = torch.linspace(0.99, 0.5, num_timesteps)
        else:  # cosine per Nichol-Dhariwal 2102.09672
            t = torch.arange(num_timesteps, dtype=torch.float32) / num_timesteps
            alphas = torch.cos(t * math.pi / 2.0) ** 2
            alphas = alphas / alphas[0]
            alphas = alphas.clamp(min=0.01, max=0.99)
        self.register_buffer("alphas", alphas)

    def forward(self, seeds: torch.Tensor) -> torch.Tensor:
        """Refine Gaussian seeds into latent vectors via T-step trajectory.

        Args:
            seeds: (B, latent_dim) Gaussian noise seeds.

        Returns:
            (B, latent_dim) refined latents.
        """
        if seeds.dim() != 2 or seeds.shape[1] != self.latent_dim:
            raise ValueError(
                f"seeds must be (B, {self.latent_dim}); got shape {tuple(seeds.shape)}"
            )

        z = seeds
        batch_size = seeds.shape[0]
        for t in range(self.num_timesteps):
            t_scalar = torch.full(
                (batch_size, 1), float(t) / max(self.num_timesteps - 1, 1),
                device=seeds.device, dtype=seeds.dtype
            )
            t_embed = self.time_embed(t_scalar)
            mlp_in = torch.cat([z, t_embed], dim=1)
            delta = self.predictors[t](mlp_in)
            alpha_t = float(self.alphas[t].item())
            z = math.sqrt(alpha_t) * z + math.sqrt(1.0 - alpha_t) * delta
        return z


class PactNervDiffusionTrajectorySubstrate(nn.Module):
    """Pact-NeRV-DiffusionTrajectory renderer (L0 SKETCH).

    Forward:
    1. Per-pair Gaussian noise seed -> diffusion-trajectory refinement -> latent.
    2. Latent -> latent_embed -> initial spatial grid.
    3. Decode through upsample chain.
    4. Final 1x1 conv heads produce rgb_0 / rgb_1.

    Per HNeRV parity L5: outputs RGB at contest camera resolution.
    """

    def __init__(self, cfg: PactNervDiffusionTrajectoryConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Per-pair noise SEEDS (stored; the trajectory predictor refines them).
        self.seeds = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=1.0)
        )

        self.predictor = DiffusionTrajectoryPredictor(
            latent_dim=cfg.latent_dim,
            num_timesteps=cfg.diffusion_num_timesteps,
            hidden=cfg.diffusion_predictor_hidden,
            noise_schedule=cfg.noise_schedule,
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

        seeds = self.seeds[pair_indices]
        z = self.predictor(seeds)

        h = self.latent_embed(z)
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

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_predictor_parameters(self) -> int:
        """Diffusion trajectory predictor parameter count (the bolt-on cost)."""
        return sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
