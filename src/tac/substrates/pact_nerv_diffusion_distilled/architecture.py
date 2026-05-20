# SPDX-License-Identifier: MIT
"""pact_nerv_diffusion_distilled architecture — Pact-NeRV-DIFFUSION-DISTILLED (L0 SKETCH).

HNeRV-class implicit renderer whose decoder is a 1-step student distilled
from a T-step diffusion teacher (Song-Dhariwal-Chen-Sutskever 2023
arXiv:2303.01469 "Consistency Models" + Yin et al. 2024 DMD 2311.18828).

The teacher network exists ONLY at compress time and is NEVER shipped in the
archive (per CLAUDE.md "Strict scorer rule" + HNeRV parity L4 ≤200 LOC
inflate budget). The student is the substrate's inflate-time renderer.

At L0 SCAFFOLD the student is a simple HNeRV-style decoder; Stage 1 dispatch
lands the real consistency-model distillation pass with epsilon-prediction
denoising + noise-conditioning + T-step teacher.

CLAUDE.md compliance: no MPS fallback, no inflate-time scorer load, no
/tmp paths, reviewable in 30 seconds per HNeRV parity L12.
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
class PactNervDiffusionDistilledConfig:
    """Static design-time parameters for Pact-NeRV-DIFFUSION-DISTILLED."""

    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    num_pairs: int = _PAIRS
    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W

    teacher_num_timesteps: int = 4
    """Compress-time T-step teacher (Stage 1 default 4; L1 sweep {1,2,4,8,16})."""
    noise_conditioning_dim: int = 16
    """Sinusoidal noise-level embedding dimension fed to student (consistency model §3)."""


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


def _noise_level_embedding(noise_level: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal noise-level embedding per consistency-model canonical pattern.

    noise_level: (B,) in [0, 1]
    Returns:    (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=noise_level.device) / half
    )  # (half,)
    args = noise_level.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


class DiffusionStudentDecoder(nn.Module):
    """1-step diffusion student decoder (the distinguishing primitive).

    At L0 SCAFFOLD: a noise-conditioned HNeRV-style decoder. The conditioning
    is a sinusoidal embedding of the noise level (consistency-model canonical
    timestep-conditioning).

    Stage 1 dispatch: the student is co-trained via consistency-model
    distillation from a T-step teacher (Song 2303.01469 §3): the student
    learns to map ANY noise level on the diffusion trajectory directly to
    x_0 in one step.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        embed_dim: int,
        initial_grid_h: int,
        initial_grid_w: int,
        decoder_channels: tuple[int, ...],
        sin_frequency: float,
        num_upsample_blocks: int,
        noise_conditioning_dim: int,
    ) -> None:
        super().__init__()
        if noise_conditioning_dim <= 0 or noise_conditioning_dim > 256:
            raise ValueError(
                f"noise_conditioning_dim must be in [1, 256]; got {noise_conditioning_dim}"
            )
        self.embed_dim = embed_dim
        self.initial_grid_h = initial_grid_h
        self.initial_grid_w = initial_grid_w
        self.noise_conditioning_dim = noise_conditioning_dim
        # Inject noise embedding into latent BEFORE spatial grid embedding.
        self.latent_proj = nn.Linear(latent_dim + noise_conditioning_dim, embed_dim * initial_grid_h * initial_grid_w)
        channels = [embed_dim, *list(decoder_channels)]
        if len(channels) <= num_upsample_blocks:
            raise ValueError(
                f"decoder_channels too short for num_upsample_blocks={num_upsample_blocks}"
            )
        blocks: list[nn.Module] = []
        for i in range(num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], sin_frequency))
        self.blocks = nn.ModuleList(blocks)
        self.final_ch = channels[num_upsample_blocks]

    def forward(
        self,
        z: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"z must be (B, latent_dim); got {tuple(z.shape)}")
        if noise_level.dim() != 1 or noise_level.shape[0] != z.shape[0]:
            raise ValueError(
                f"noise_level must be (B,) matching z; got {tuple(noise_level.shape)}"
            )
        noise_emb = _noise_level_embedding(noise_level, self.noise_conditioning_dim)
        z_cat = torch.cat([z, noise_emb], dim=1)
        h = self.latent_proj(z_cat)
        h = h.view(-1, self.embed_dim, self.initial_grid_h, self.initial_grid_w)
        for block in self.blocks:
            h = block(h)
        return h


class PactNervDiffusionDistilledSubstrate(nn.Module):
    """Pact-NeRV-DIFFUSION-DISTILLED renderer (L0 SKETCH; 1-step student)."""

    def __init__(self, cfg: PactNervDiffusionDistilledConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        self.student = DiffusionStudentDecoder(
            latent_dim=cfg.latent_dim,
            embed_dim=cfg.embed_dim,
            initial_grid_h=cfg.initial_grid_h,
            initial_grid_w=cfg.initial_grid_w,
            decoder_channels=cfg.decoder_channels,
            sin_frequency=cfg.sin_frequency,
            num_upsample_blocks=cfg.num_upsample_blocks,
            noise_conditioning_dim=cfg.noise_conditioning_dim,
        )
        self.head_rgb_0 = nn.Conv2d(self.student.final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(self.student.final_ch, 3, kernel_size=1)
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
        self,
        pair_indices: torch.Tensor,
        *,
        noise_level: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(f"pair_indices out of range [0, {self.cfg.num_pairs})")
        z = self.latents[pair_indices]
        if noise_level is None:
            # Inference-time: 1-step student always uses noise_level=0 (clean)
            noise_level = torch.zeros(z.shape[0], device=z.device)
        h = self.student(z, noise_level)
        if h.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            h = F.interpolate(
                h, size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear", align_corners=False,
            )
        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
