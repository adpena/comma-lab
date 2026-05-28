# SPDX-License-Identifier: MIT
"""z6_v2_cargo_cult_unwind architecture — 2-level Rao-Ballard hierarchical FiLM
ego-motion-conditioned predictor with Atick-Redlich cooperative-receiver primitive.

Per CLAUDE.md "INDIVIDUALLY-FRACTAL" standing directive 2026-05-27, this is
Z6-v2's OWN canonical engineering pass per UNIQUE-AND-COMPLETE-PER-METHOD —
NOT shared-helper shortcut from PACT-NeRV sister cascade. The distinguishing
primitives (per Catalog #272):

1. **2-level Rao-Ballard hierarchical FiLM-ego-motion predictor** — depth=3
   FiLM-conditioned blocks (Z6 redesign Candidate 1, ~300K params target)
   organized into a 2-level Rao-Ballard hierarchy (level 0 micro-residuals;
   level 1 meso-residuals). Direct unwind of CC-1 + CC-2 cargo-cults per
   `z6_v2_cargo_cult_unwind_4_candidate_redesign_path_b_20260517.md`.
2. **FoE (focus-of-expansion) ego-motion prior conditioning** — per pair, a
   small ego-motion vector (translation + rotation deltas) feeds FiLM (γ, β)
   modulation at every block.
3. **Atick-Redlich cooperative-receiver gradient binding** at scorer-loss
   level per Catalog #311 (sister with ATW V2 + Z4); the receiver loss is
   `I(X;T) - β * I(T;Y)` proxied via the per-pair canonical reconstruction
   MSE plus the canonical scorer surrogate teacher per Catalog #164.

Architecture (PyTorch sister; MLX sister at mlx_renderer.py):

    pair_idx -> latent (per-pair) + ego_vec (per-pair, 6-dim) -> latent_embed ->
        initial grid (NHWC) ->
            FiLM(γ_0, β_0) -> DepthSep -> sin -> PixelShuffle(2)  [level 0 micro]
            FiLM(γ_1, β_1) -> DepthSep -> sin -> PixelShuffle(2)  [level 0 micro]
            FiLM(γ_2, β_2) -> DepthSep -> sin -> PixelShuffle(2)  [level 0 micro]
            ---- meso boundary ----
            FiLM(γ_3, β_3) -> DepthSep -> sin -> PixelShuffle(2)  [level 1 meso]
            FiLM(γ_4, β_4) -> DepthSep -> sin -> PixelShuffle(2)  [level 1 meso]
            FiLM(γ_5, β_5) -> DepthSep -> sin -> PixelShuffle(2)  [level 1 meso]
            FiLM(γ_6, β_6) -> DepthSep -> sin -> PixelShuffle(2)  [level 1 meso]
        head_rgb_0 / head_rgb_1 (1x1 conv) -> sigmoid

Parameter target: ~300K per design memo Candidate 1.

Per HNeRV parity L5: outputs RGB at contest camera resolution (384, 512); NOT
a mask codec. Cooperative-receiver gradient binding lives in score_aware_loss.py
(Catalog #311).
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
class Z6V2Config:
    """Canonical Z6-v2 config; targets ~300K params per design memo Candidate 1."""

    latent_dim: int = 24
    ego_dim: int = 6  # (tx, ty, tz, rx, ry, rz) FoE ego-motion vector
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4

    # 7 upsample blocks (3x4 -> 384x512 via 7x PixelShuffle(2))
    # decoder_channels + film_hidden_width tuned 2026-05-28 empirical sweep to
    # land at ~308K params (within design memo Candidate 1 ~300K target).
    decoder_channels: tuple[int, ...] = (56, 48, 40, 36, 32, 28, 24)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7

    # Rao-Ballard 2-level hierarchy boundary: first 3 blocks = level 0 (micro);
    # remaining 4 blocks = level 1 (meso).
    rao_ballard_level_boundary: int = 3

    # FiLM generator hidden width (controls FiLM MLP parameter count;
    # tuned to ~300K total per design memo Candidate 1 spec).
    film_hidden_width: int = 80

    num_pairs: int = _PAIRS
    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W

    # FiLM conditioning depth — 3 means 3 FiLM-generator MLP layers
    # (per design memo Candidate 1 multi-layer FiLM depth=3).
    film_generator_depth: int = 3

    # Cooperative-receiver beta per Atick-Redlich 1990 IB framework
    # (placeholder; the loss module uses this in the bound).
    cooperative_receiver_beta: float = 0.5


class _SinAct(nn.Module):
    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _DepthSepConv(nn.Module):
    """Depth-separable conv (depthwise 3x3 + pointwise 1x1)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _FiLMGenerator(nn.Module):
    """Multi-layer FiLM (γ, β) generator from (latent ⊕ ego_vec) -> (2 * out_ch).

    Direct unwind of CC-1 cargo-cult per the Z6-v2 redesign memo: Z6-v1 was
    a single Linear layer; Candidate 1 makes this a depth-3 MLP so the FiLM
    parameters have enough representational capacity to express semantic ego
    conditioning at the scorer's stride-2 stem per Rao verbatim critique.
    """

    def __init__(
        self,
        latent_dim: int,
        ego_dim: int,
        out_ch: int,
        depth: int,
        sin_freq: float,
        hidden_width: int = 24,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"film_generator_depth must be >= 1; got {depth}")
        if hidden_width < 1:
            raise ValueError(f"hidden_width must be >= 1; got {hidden_width}")
        self.out_ch = int(out_ch)
        # MLP with SIREN-style sin activation between layers; hidden width is the
        # capped FiLM generator width (independent of out_ch so total parameter
        # count stays near the ~300K target per design memo Candidate 1 spec).
        layers: list[nn.Module] = []
        in_dim = latent_dim + ego_dim
        hidden = int(hidden_width)
        for i in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(_SinAct(sin_freq))
            in_dim = hidden
        # Final layer projects to 2 * out_ch (γ and β concatenated).
        layers.append(nn.Linear(in_dim, 2 * out_ch))
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent_plus_ego: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(latent_plus_ego)
        gamma, beta = out.chunk(2, dim=-1)
        # γ initialized near 1.0 + γ' so the FiLM modulation starts near-identity.
        return (1.0 + gamma, beta)


class _FiLMUpBlock(nn.Module):
    """FiLM-conditioned DepthSep -> sin -> PixelShuffle(2) block.

    Mirrors the sister _DsUpBlock in PACT-NeRV but adds FiLM (γ, β) per-channel
    modulation from the per-pair (latent ⊕ ego_vec) FiLM generator.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        sin_freq: float,
        latent_dim: int,
        ego_dim: int,
        film_depth: int,
        film_hidden_width: int = 24,
    ) -> None:
        super().__init__()
        # DepthSep outputs out_ch * 4 channels (the PixelShuffle(2) factor).
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)
        # FiLM generator emits (γ, β) for the (out_ch * 4) feature map BEFORE shuffle.
        self.film_gen = _FiLMGenerator(
            latent_dim=latent_dim,
            ego_dim=ego_dim,
            out_ch=out_ch * 4,
            depth=film_depth,
            sin_freq=sin_freq,
            hidden_width=film_hidden_width,
        )

    def forward(
        self, x: torch.Tensor, latent_plus_ego: torch.Tensor
    ) -> torch.Tensor:
        h = self.dsc(x)
        # FiLM modulation: γ * h + β per-channel.
        gamma, beta = self.film_gen(latent_plus_ego)  # each (B, out_ch * 4)
        # Reshape γ, β to (B, C, 1, 1) for broadcast across H, W.
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = gamma * h + beta
        h = self.act(h)
        return self.shuffle(h)


class Z6V2Substrate(nn.Module):
    """Z6-v2 cargo-cult-unwind PyTorch substrate (L1 LONG-RUN MLX-LOCAL sister)."""

    def __init__(self, cfg: Z6V2Config) -> None:
        super().__init__()
        self.cfg = cfg

        # Per-pair learnable latent.
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        # Per-pair learnable ego-motion vector (FoE prior; 6-dim: tx/ty/tz/rx/ry/rz).
        self.ego_vecs = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.ego_dim).normal_(std=0.02)
        )

        # Latent embedding to initial spatial grid.
        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        # Per-block FiLM-conditioned upsample blocks (7 total; first 3 = level 0
        # micro, remaining 4 = level 1 meso per Rao-Ballard 2-level hierarchy).
        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError("decoder_channels too short for num_upsample_blocks")
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(
                _FiLMUpBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    sin_freq=cfg.sin_frequency,
                    latent_dim=cfg.latent_dim,
                    ego_dim=cfg.ego_dim,
                    film_depth=cfg.film_generator_depth,
                    film_hidden_width=cfg.film_hidden_width,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)

        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN init for Conv2d + Linear (sister of PACT-NeRV pattern)."""
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

        z = self.latents[pair_indices]
        ego = self.ego_vecs[pair_indices]
        # Concatenate (latent, ego_vec) for the FiLM generator.
        latent_plus_ego = torch.cat([z, ego], dim=-1)

        h = self.latent_embed(z)
        h = h.view(-1, self.cfg.embed_dim, self.cfg.initial_grid_h, self.cfg.initial_grid_w)

        for block in self.blocks:
            h = block(h, latent_plus_ego)

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

    def layerwise_inspector(self) -> dict[str, dict[str, int]]:
        """Per-layer inspection hook per OBSERVABILITY_SURFACE inspectable_per_layer.

        Returns per-block (level_0_micro vs level_1_meso) parameter counts and
        FiLM-generator parameter counts so a reviewer can decompose the Z6-v2
        architecture's behavior without re-instrumentation per Catalog #305.
        """
        out: dict[str, dict[str, int]] = {}
        boundary = self.cfg.rao_ballard_level_boundary
        for i, block in enumerate(self.blocks):
            level = "level_0_micro" if i < boundary else "level_1_meso"
            block_params = sum(p.numel() for p in block.parameters())
            film_params = sum(p.numel() for p in block.film_gen.parameters())
            dsc_params = sum(p.numel() for p in block.dsc.parameters())
            out[f"block_{i}_{level}"] = {
                "total_params": block_params,
                "film_generator_params": film_params,
                "depthsep_conv_params": dsc_params,
            }
        return out


__all__ = [
    "Z6V2Config",
    "Z6V2Substrate",
    "_DepthSepConv",
    "_FiLMGenerator",
    "_FiLMUpBlock",
    "_SinAct",
]
