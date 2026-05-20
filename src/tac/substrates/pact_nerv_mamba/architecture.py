# SPDX-License-Identifier: MIT
"""pact_nerv_mamba architecture - Mamba-2 state-space backbone (L0 SKETCH).

L0 SCAFFOLD uses a lightweight LINEAR-RECURRENCE proxy for the Mamba-2
selective-scan SSM. The Stage 1 dispatch lands the real
``mamba_ssm.modules.mamba2.Mamba2`` block (state-spaces/mamba install).
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
class PactNervMambaConfig:
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

    ssm_state_dim: int = 16
    """Mamba-2 state-space dimension (Gu-Dao 2023). L1 budget: 64."""

    ssm_conv_width: int = 4
    """Local 1D conv kernel before the SSM scan (Mamba canonical = 4)."""


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


class StateSpaceRecurrenceBlock(nn.Module):
    """Linear-recurrence proxy for Mamba-2 selective-scan SSM (L0 SCAFFOLD).

    Implements h_t = A * h_{t-1} + B * x_t per the canonical SSM
    formulation (Gu-Dao 2023 §2.1). The L0 stand-in uses a fixed-A diagonal
    state matrix + input-driven B; the L1 dispatch replaces with the
    selective (input-dependent) A, B per ``mamba_ssm.modules.mamba2.Mamba2``.

    Input:   x of shape (N, L, D)  (N=batch, L=sequence, D=latent_dim)
    Output:  y of shape (N, L, D)

    The recurrence is run via sequential scan at L0 (O(L)); the real Mamba-2
    block uses the SSD chunked-scan algorithm achieving O(L) work + O(log L)
    depth via the structured-state-space duality (Dao-Gu 2024 §3).
    """

    def __init__(self, latent_dim: int, state_dim: int = 16, conv_width: int = 4) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1; got {latent_dim}")
        if state_dim < 1 or state_dim > 256:
            raise ValueError(f"state_dim must be in [1, 256]; got {state_dim}")
        if conv_width < 1 or conv_width > 16:
            raise ValueError(f"conv_width must be in [1, 16]; got {conv_width}")
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.conv_width = conv_width
        # A: diagonal state matrix; A_log stores log(|A|) for positive-decay
        # parameterization (Mamba canonical: A = -exp(A_log) so A is always
        # negative-real and stable). Init |A| in [0.1, 1.0] then take log.
        self.A_log = nn.Parameter(torch.linspace(0.1, 1.0, state_dim).log())
        # B: input projection
        self.B_proj = nn.Linear(latent_dim, state_dim)
        # C: output projection
        self.C_proj = nn.Linear(state_dim, latent_dim)
        # Local 1D conv before SSM (Mamba canonical pattern)
        self.local_conv = nn.Conv1d(
            latent_dim, latent_dim, kernel_size=conv_width,
            padding=conv_width - 1, groups=latent_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (N, L, D); got {tuple(x.shape)}")
        n, l, d = x.shape
        if d != self.latent_dim:
            raise ValueError(f"latent_dim mismatch: {d} != {self.latent_dim}")
        # Local conv: (N, D, L)
        x_c = self.local_conv(x.transpose(1, 2))[:, :, :l].transpose(1, 2)
        x_c = F.silu(x_c)
        # SSM scan
        A = -torch.exp(self.A_log)  # negative real (stable)
        B = self.B_proj(x_c)  # (N, L, state_dim)
        h = torch.zeros(n, self.state_dim, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(l):
            h = h * torch.exp(A) + B[:, t]
            outs.append(h)
        H = torch.stack(outs, dim=1)  # (N, L, state_dim)
        y = self.C_proj(H)  # (N, L, D)
        return y + x


class PactNervMambaSubstrate(nn.Module):
    """Pact-NeRV-MAMBA renderer (L0 SKETCH; Mamba-2 backbone stand-in)."""

    def __init__(self, cfg: PactNervMambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        self.ssm = StateSpaceRecurrenceBlock(
            latent_dim=cfg.latent_dim,
            state_dim=cfg.ssm_state_dim,
            conv_width=cfg.ssm_conv_width,
        )
        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
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
        # Build sequence from per-pair latents (treat the full set as one "sequence")
        all_z = self.latents.unsqueeze(0)  # (1, num_pairs, latent_dim)
        ssm_out = self.ssm(all_z)  # (1, num_pairs, latent_dim)
        z = ssm_out[0, pair_indices]  # (B, latent_dim)
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
