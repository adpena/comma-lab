# SPDX-License-Identifier: MIT
"""pact_nerv_neural_codec_e2e architecture — ULTIMATE-PAPER (L0 SKETCH).

End-to-end neural codec FUSING Ballé 2018 scale hyperprior + HNeRV decoder.
The hyperprior network produces per-latent scale parameters; the entropy
bottleneck computes a differentiable rate proxy that is co-optimized with the
HNeRV decoder under the score-aware Lagrangian.

Per Ballé-Minnen-Singh-Hwang-Johnston 2018 arXiv:1802.01436 §3: the
hyperprior h_i parameterizes a conditional Gaussian over z_i, and the
rate loss is -log2(N(z_i | 0, sigma(h_i))). At L0 SCAFFOLD the rate proxy
is a simple Gaussian likelihood; Stage 1 dispatch lands the entropy
bottleneck with proper noise-during-training + round-during-inference.

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
class PactNervNeuralCodecE2eConfig:
    """Static design-time parameters for Pact-NeRV-NEURAL-CODEC-E2E."""

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

    hyperprior_dim: int = 8
    """Ballé 2018 §3 hyperprior auxiliary latent dimension. L0 default 8;
    L1 sweep over {4, 8, 16, 32}."""
    hyperprior_hidden: int = 24
    """Hyperprior encoder hidden width."""
    min_scale: float = 0.1
    """Minimum hyperprior-emitted scale (Ballé canonical floor for stability)."""


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


class HyperpriorEncoder(nn.Module):
    """Ballé 2018 §3 scale hyperprior encoder (the distinguishing primitive).

    Maps per-pair latent z_i -> hyperprior h_i in R^hyperprior_dim, then a
    second projection produces per-latent SCALE parameters sigma_i for the
    conditional Gaussian prior p(z_i | sigma_i) = N(0, sigma_i^2).

    The rate proxy is -log2(N(z_i | 0, sigma_i)). At L0 SCAFFOLD the
    hyperprior is co-optimized with the decoder under the score-aware
    Lagrangian; Stage 1 dispatch lands the full Ballé entropy bottleneck
    with proper additive-uniform-noise during training + round-during-
    inference (Ballé 2018 §3.1).
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hyperprior_dim: int,
        hidden: int = 24,
        min_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1; got {latent_dim}")
        if hyperprior_dim < 1 or hyperprior_dim > 64:
            raise ValueError(f"hyperprior_dim must be in [1, 64]; got {hyperprior_dim}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1; got {hidden}")
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive; got {min_scale}")
        self.latent_dim = latent_dim
        self.hyperprior_dim = hyperprior_dim
        self.hidden = hidden
        self.min_scale = float(min_scale)
        # Hyperprior encoder: z -> h
        self.encode_hyper = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hyperprior_dim),
        )
        # Hyperprior decoder: h -> per-latent scale (log-scale for stability)
        self.decode_scales = nn.Sequential(
            nn.Linear(hyperprior_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (hyperprior, per-latent scales).

        scales >= min_scale via softplus + floor per Ballé canonical.
        """
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z must be (B, latent_dim={self.latent_dim}); got {tuple(z.shape)}"
            )
        h = self.encode_hyper(z)
        log_scale_raw = self.decode_scales(h)
        scales = F.softplus(log_scale_raw) + self.min_scale
        return h, scales

    def rate_proxy(self, z: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Ballé 2018 §3 rate proxy: -log2(N(z | 0, sigma^2)).

        At L0 SCAFFOLD this is a simple per-element gaussian likelihood;
        Stage 1 dispatch replaces with the full entropy bottleneck +
        additive-uniform-noise during training.

        Returns scalar in bits-per-pair.
        """
        # Gaussian log-likelihood: -log2(N(z | 0, sigma)) = 0.5 * log2(2*pi*sigma^2) + z^2 / (2 * sigma^2 * ln(2))
        log_two = math.log(2.0)
        gaussian_const = 0.5 * math.log2(2.0 * math.pi)
        log2_sigma = torch.log2(scales)
        rate_bits = (
            gaussian_const
            + log2_sigma
            + 0.5 * z.pow(2) / (scales.pow(2) * log_two)
        )
        return rate_bits.sum(dim=1).mean()  # mean over batch, sum over latent dims


class PactNervNeuralCodecE2eSubstrate(nn.Module):
    """Pact-NeRV-NEURAL-CODEC-E2E renderer (L0 SKETCH; Ballé hyperprior + HNeRV decoder)."""

    def __init__(self, cfg: PactNervNeuralCodecE2eConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        self.hyperprior = HyperpriorEncoder(
            latent_dim=cfg.latent_dim,
            hyperprior_dim=cfg.hyperprior_dim,
            hidden=cfg.hyperprior_hidden,
            min_scale=cfg.min_scale,
        )
        self.latent_embed = nn.Linear(
            cfg.latent_dim, cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w
        )
        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError("decoder_channels too short for num_upsample_blocks")
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)
        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self._last_rate_bits: torch.Tensor | None = None
        self._last_scales: torch.Tensor | None = None
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

    def forward(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(f"pair_indices out of range [0, {self.cfg.num_pairs})")
        z = self.latents[pair_indices]  # (B, latent_dim)
        _h, scales = self.hyperprior(z)  # (B, hyperprior_dim), (B, latent_dim)
        rate_bits = self.hyperprior.rate_proxy(z, scales)
        self._last_rate_bits = rate_bits
        self._last_scales = scales

        h = self.latent_embed(z)
        h = h.view(-1, self.cfg.embed_dim, self.cfg.initial_grid_h, self.cfg.initial_grid_w)
        for block in self.blocks:
            h = block(h)
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

    def num_hyperprior_parameters(self) -> int:
        """Hyperprior network parameter count (the codec cost)."""
        return sum(p.numel() for p in self.hyperprior.parameters() if p.requires_grad)
