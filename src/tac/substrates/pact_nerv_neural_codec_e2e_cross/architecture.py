# SPDX-License-Identifier: MIT
"""pact_nerv_neural_codec_e2e_cross architecture — end-to-end neural codec composition (L0 SKETCH).

CROSS-NEURAL-E2E composition variant per PACT-NERV-ULTIMATE Variant #18.

Both codec branches are HNeRV-class neural networks; a Ballé-style
hyperprior gate g(z) ∈ [0, 1] routes per-pair bits between them. At L0
SCAFFOLD the gate is a small MLP over concatenated branch latents.
L1 will replace with a learned per-region gate per Atick-Redlich 1990
cooperative-receiver + Ballé 2018 hyperprior.

Composition at inflate time:
    g = hyperprior(z_a, z_b)  # per-pair scalar in [0, 1]
    rgb = clamp(g * branch_a(z_a) + (1 - g) * branch_b(z_b), 0, 1)
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
class PactNervNeuralCodecE2ECrossConfig:
    latent_dim_a: int = 16
    latent_dim_b: int = 16
    embed_dim: int = 48
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (36, 32, 28, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    num_pairs: int = _PAIRS
    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W
    hyperprior_hidden: int = 32
    """Hyperprior gate MLP hidden dim."""
    gate_init_bias: float = 0.0
    """Initial gate bias: 0.0 = sigmoid(0) = 0.5 (50/50 branch mix)."""


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


class _HnervBranch(nn.Module):
    """Pact-NeRV-class HNeRV branch (depthwise-separable + SIREN + PixelShuffle)."""

    def __init__(
        self, latent_dim: int, embed_dim: int, initial_grid_h: int,
        initial_grid_w: int, decoder_channels: tuple[int, ...],
        sin_frequency: float, num_upsample_blocks: int,
        output_h: int, output_w: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.initial_grid_h = initial_grid_h
        self.initial_grid_w = initial_grid_w
        self.output_h = output_h
        self.output_w = output_w
        self.latent_embed = nn.Linear(
            latent_dim, embed_dim * initial_grid_h * initial_grid_w
        )
        channels = [embed_dim, *list(decoder_channels)]
        if len(channels) <= num_upsample_blocks:
            raise ValueError("decoder_channels too short for num_upsample_blocks")
        self.blocks = nn.ModuleList(
            [_DsUpBlock(channels[i], channels[i + 1], sin_frequency)
             for i in range(num_upsample_blocks)]
        )
        final_ch = channels[num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.latent_embed(z)
        h = h.view(-1, self.embed_dim, self.initial_grid_h, self.initial_grid_w)
        for block in self.blocks:
            h = block(h)
        if h.shape[-2:] != (self.output_h, self.output_w):
            h = F.interpolate(
                h, size=(self.output_h, self.output_w),
                mode="bilinear", align_corners=False,
            )
        # Both branches emit RGB in [0, 1] via sigmoid
        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1


class HyperpriorGate(nn.Module):
    """Ballé 2018 hyperprior-style gate g(z_a, z_b) ∈ [0, 1] (per-pair scalar).

    L0 SCAFFOLD: small MLP over concatenated latents. L1 will replace
    with per-region gate per Ballé 2018 §3.3 (autoregressive hyperprior).
    """

    def __init__(
        self, latent_dim_a: int, latent_dim_b: int, hidden_dim: int,
        gate_init_bias: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim_a + latent_dim_b, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_gate = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            # Init gate bias so sigmoid(bias) = 0.5 (balanced branch mix)
            self.fc_gate.bias.fill_(gate_init_bias)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Return per-pair gate scalar in [0, 1]."""
        z = torch.cat([z_a, z_b], dim=-1)
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        gate_logit = self.fc_gate(h)
        return torch.sigmoid(gate_logit)


class PactNervNeuralCodecE2ECrossSubstrate(nn.Module):
    """Pact-NeRV-NEURAL-CODEC-E2E-CROSS renderer (L0 SKETCH).

    End-to-end neural-codec composition: two HNeRV branches A,B + hyperprior
    gate g(z_a, z_b) ∈ [0, 1] routes per-pair bits between them.
    """

    def __init__(self, cfg: PactNervNeuralCodecE2ECrossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Per-pair latents for each branch
        self.latents_a = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim_a).normal_(std=0.02)
        )
        self.latents_b = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim_b).normal_(std=0.02)
        )
        # Two HNeRV branches
        self.branch_a = _HnervBranch(
            cfg.latent_dim_a, cfg.embed_dim, cfg.initial_grid_h,
            cfg.initial_grid_w, cfg.decoder_channels, cfg.sin_frequency,
            cfg.num_upsample_blocks, cfg.output_height, cfg.output_width,
        )
        self.branch_b = _HnervBranch(
            cfg.latent_dim_b, cfg.embed_dim, cfg.initial_grid_h,
            cfg.initial_grid_w, cfg.decoder_channels, cfg.sin_frequency,
            cfg.num_upsample_blocks, cfg.output_height, cfg.output_width,
        )
        # Hyperprior gate
        self.gate = HyperpriorGate(
            cfg.latent_dim_a, cfg.latent_dim_b, cfg.hyperprior_hidden,
            cfg.gate_init_bias,
        )
        self._siren_init()

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, HyperpriorGate):
                    # Skip hyperprior init; preserve gate_init_bias
                    continue
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    if m.groups > 1:
                        fan_in = m.kernel_size[0] * m.kernel_size[1]
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.Linear):
                    # Skip hyperprior internals
                    is_hyperprior = any(
                        m is sub for sub in (
                            self.gate.fc1, self.gate.fc2, self.gate.fc_gate,
                        )
                    )
                    if is_hyperprior:
                        continue
                    fan_in = m.in_features
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """End-to-end neural-codec composition via hyperprior gate."""
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if (
            pair_indices.min().item() < 0
            or pair_indices.max().item() >= self.cfg.num_pairs
        ):
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )
        z_a = self.latents_a[pair_indices]
        z_b = self.latents_b[pair_indices]
        # Two branches forward
        rgb_a_0, rgb_a_1 = self.branch_a(z_a)
        rgb_b_0, rgb_b_1 = self.branch_b(z_b)
        # Hyperprior gate (per-pair scalar in [0, 1])
        gate = self.gate(z_a, z_b)  # (B, 1)
        gate = gate.view(-1, 1, 1, 1)  # broadcast to (B, 1, H, W)
        # End-to-end composition
        rgb_0 = torch.clamp(gate * rgb_a_0 + (1.0 - gate) * rgb_b_0, 0.0, 1.0)
        rgb_1 = torch.clamp(gate * rgb_a_1 + (1.0 - gate) * rgb_b_1, 0.0, 1.0)
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gate_values(self, pair_indices: torch.Tensor) -> torch.Tensor:
        """Return per-pair gate values for observability."""
        z_a = self.latents_a[pair_indices]
        z_b = self.latents_b[pair_indices]
        return self.gate(z_a, z_b).squeeze(-1)  # (B,)
