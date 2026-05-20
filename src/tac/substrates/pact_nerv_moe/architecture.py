# SPDX-License-Identifier: MIT
"""pact_nerv_moe architecture — Pact-NeRV-MOE (L0 SKETCH).

HNeRV-class implicit renderer with K=4 expert decoders + pose-embedding-
conditioned top-k=2 router (Shazeer-Mirhoseini-Maziarz-Davis-Le-Hinton-Dean
2017 arXiv:1701.06538). The distinguishing primitive vs IA3/distilled: a
PoseConditionedRouter routes per-pair compute to top-2 of K experts via
softmax over a small pose embedding, with load-balancing auxiliary loss.

Per Atick-Redlich 1990 cooperative-receiver gate (sister of Z4): the receiver
(scorer) drives per-input compute dispatch. At L0 SCAFFOLD the router is
randomly initialized; Stage 1 dispatch lands the score-driven routing.

CLAUDE.md compliance: no MPS fallback, no inflate-time scorer load (router
features are upstream of any scorer; experts are pure HNeRV decoders), no
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
class PactNervMoeConfig:
    """Static design-time parameters for Pact-NeRV-MOE."""

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

    num_experts: int = 4
    """K experts (L0 default 4; Stage 1 sweep K in {2, 4, 8})."""
    top_k: int = 2
    """Top-k routing per Shazeer canonical + Mixtral OSS."""
    pose_embed_dim: int = 16
    """Pose-embedding dimension for router conditioning."""


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


class PoseConditionedRouter(nn.Module):
    """Pose-embedding-conditioned top-k MoE router (the distinguishing primitive).

    Per Shazeer 1701.06538 §3: route x in R^d to top-k of K experts via
    softmax(W * x). The pose embedding e_i is the routing signal — different
    poses (highway / urban / parking) route to different specialist experts.

    Returns:
        route_probs: (B, K) softmax over K experts (zeros for non-top-k)
        load_balance_aux: scalar auxiliary loss penalizing uneven expert usage
                          per Shazeer §4 (mean fraction routed * mean prob)
    """

    def __init__(self, *, pose_embed_dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        if pose_embed_dim <= 0:
            raise ValueError(f"pose_embed_dim must be positive; got {pose_embed_dim}")
        if num_experts < 2:
            raise ValueError(f"num_experts must be >= 2; got {num_experts}")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts]; got {top_k}")
        self.pose_embed_dim = pose_embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(pose_embed_dim, num_experts, bias=False)

    def forward(self, pose_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pose_embed.dim() != 2 or pose_embed.shape[1] != self.pose_embed_dim:
            raise ValueError(
                f"pose_embed must be (B, pose_embed_dim={self.pose_embed_dim}); "
                f"got {tuple(pose_embed.shape)}"
            )
        logits = self.gate(pose_embed)  # (B, K)
        # Top-k mask
        topk_vals, topk_idx = torch.topk(logits, k=self.top_k, dim=-1)
        mask = torch.zeros_like(logits)
        mask.scatter_(1, topk_idx, 1.0)
        # Softmax-normalize over top-k entries
        masked_logits = logits.masked_fill(mask == 0, float("-inf"))
        route_probs = F.softmax(masked_logits, dim=-1)
        # Replace NaN (from all-masked rows; shouldn't happen but defense-in-depth)
        route_probs = torch.nan_to_num(route_probs, nan=0.0)

        # Load-balancing aux (Shazeer §4: importance + load = balanced when product min)
        full_probs = F.softmax(logits, dim=-1)  # over ALL experts
        frac_routed = mask.mean(dim=0)  # (K,) per-expert routing fraction
        mean_prob = full_probs.mean(dim=0)  # (K,) per-expert mean prob
        load_balance_aux = (
            self.num_experts * (frac_routed * mean_prob).sum()
        )

        return route_probs, load_balance_aux


class _ExpertDecoder(nn.Module):
    """One HNeRV-class expert decoder (K of these in the MoE substrate)."""

    def __init__(
        self,
        *,
        embed_dim: int,
        decoder_channels: tuple[int, ...],
        sin_frequency: float,
        num_upsample_blocks: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        channels = [embed_dim, *list(decoder_channels)]
        if len(channels) <= num_upsample_blocks:
            raise ValueError(
                f"decoder_channels too short for num_upsample_blocks={num_upsample_blocks}"
            )
        blocks: list[nn.Module] = []
        for i in range(num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], sin_frequency))
        self.blocks = nn.ModuleList(blocks)
        final_ch = channels[num_upsample_blocks]
        self.head = nn.Conv2d(final_ch, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class PactNervMoeSubstrate(nn.Module):
    """Pact-NeRV-MOE renderer (L0 SKETCH; K=4 experts + pose-conditioned router)."""

    def __init__(self, cfg: PactNervMoeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        self.latent_embed = nn.Linear(
            cfg.latent_dim, cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w
        )
        self.pose_embed = nn.Linear(cfg.latent_dim, cfg.pose_embed_dim)
        self.router = PoseConditionedRouter(
            pose_embed_dim=cfg.pose_embed_dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
        )
        # One head emits 6 channels (rgb_0 + rgb_1) per expert; weighted-sum
        # by route_probs[k] gives the final mixed output.
        self.experts = nn.ModuleList(
            [
                _ExpertDecoder(
                    embed_dim=cfg.embed_dim,
                    decoder_channels=cfg.decoder_channels,
                    sin_frequency=cfg.sin_frequency,
                    num_upsample_blocks=cfg.num_upsample_blocks,
                    output_channels=6,
                )
                for _ in range(cfg.num_experts)
            ]
        )
        self._last_load_balance_aux: torch.Tensor | None = None
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
        pose_e = self.pose_embed(z)  # (B, pose_embed_dim)
        route_probs, load_balance_aux = self.router(pose_e)  # (B, K), scalar
        self._last_load_balance_aux = load_balance_aux

        h = self.latent_embed(z)
        h = h.view(
            -1, self.cfg.embed_dim, self.cfg.initial_grid_h, self.cfg.initial_grid_w
        )

        # Mix expert outputs by route probs
        mixed_output: torch.Tensor | None = None
        for k, expert in enumerate(self.experts):
            ek_out = expert(h)  # (B, 6, H, W)
            # Resize to contest output if needed
            if ek_out.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
                ek_out = F.interpolate(
                    ek_out,
                    size=(self.cfg.output_height, self.cfg.output_width),
                    mode="bilinear",
                    align_corners=False,
                )
            w_k = route_probs[:, k].view(-1, 1, 1, 1)  # (B, 1, 1, 1)
            weighted = w_k * ek_out
            mixed_output = weighted if mixed_output is None else mixed_output + weighted

        assert mixed_output is not None
        rgb_0 = torch.sigmoid(mixed_output[:, 0:3])
        rgb_1 = torch.sigmoid(mixed_output[:, 3:6])
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_router_parameters(self) -> int:
        """Pose-conditioned router parameter count (the bolt-on cost)."""
        return sum(p.numel() for p in self.router.parameters() if p.requires_grad)
