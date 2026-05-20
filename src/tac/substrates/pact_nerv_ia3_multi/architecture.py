# SPDX-License-Identifier: MIT
"""pact_nerv_ia3_multi architecture - multi-layer IA3 γ-only + per-pair difficulty.

Extends sister pact_nerv_ia3 (commit 9cf9bdb16) with:
- Multi-block IA3 modulation at EVERY upsample block (vs single-block)
- Per-pair difficulty signal fused with ego-pose conditioning

HARD-EARNED-EMPIRICALLY-SUPERIOR per FILM-FAMILY-RESEARCH Section 8.6.
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
_POSE_DIM = 6
_DIFFICULTY_DIM = 1


@dataclass(frozen=True)
class PactNervIa3MultiConfig:
    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    pose_dim: int = _POSE_DIM
    difficulty_dim: int = _DIFFICULTY_DIM
    ia3_init_delta_std: float = 0.01
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


class IA3MultiGammaOnlyModulation(nn.Module):
    """Multi-block IA3 γ-only with per-pair difficulty + pose conditioning.

    Per IA3 paper §3.2 + FILM-FAMILY-RESEARCH §8.6: γ_init=1.0 + Δ residual
    form. Multi-layer is HARD-EARNED-EMPIRICALLY-SUPERIOR for video-temporal
    conditioning vs single-layer (TeNeRV + HNeRV ablation).

    Distinguishing primitive vs sister pact_nerv_ia3: this class fuses
    BOTH ego-pose AND per-pair difficulty into the γ projection.
    """

    def __init__(
        self,
        num_features: int,
        pose_dim: int = _POSE_DIM,
        difficulty_dim: int = _DIFFICULTY_DIM,
        init_delta_std: float = 0.01,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive; got {num_features}")
        if pose_dim <= 0:
            raise ValueError(f"pose_dim must be positive; got {pose_dim}")
        if difficulty_dim < 0:
            raise ValueError(f"difficulty_dim must be non-negative; got {difficulty_dim}")
        if init_delta_std < 0:
            raise ValueError(f"init_delta_std must be non-negative; got {init_delta_std}")
        self.num_features = num_features
        self.pose_dim = pose_dim
        self.difficulty_dim = difficulty_dim
        cond_dim = pose_dim + difficulty_dim
        self.gamma_proj = nn.Linear(cond_dim, num_features)
        # NO β projection - this is THE IA3 distinguishing primitive vs FiLM
        with torch.no_grad():
            self.gamma_proj.weight.normal_(mean=0.0, std=init_delta_std)
            self.gamma_proj.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        pose: torch.Tensor,
        difficulty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must be (B, C, H, W); got {tuple(x.shape)}")
        if pose.dim() != 2:
            raise ValueError(f"pose must be (B, pose_dim); got {tuple(pose.shape)}")
        if pose.shape[1] != self.pose_dim:
            raise ValueError(f"pose dim {pose.shape[1]} != {self.pose_dim}")
        if x.shape[1] != self.num_features:
            raise ValueError(f"x channels {x.shape[1]} != {self.num_features}")
        if self.difficulty_dim == 0:
            cond = pose
        else:
            if difficulty is None:
                raise ValueError(
                    f"difficulty must be provided when difficulty_dim={self.difficulty_dim}"
                )
            if difficulty.dim() != 2 or difficulty.shape[1] != self.difficulty_dim:
                raise ValueError(
                    f"difficulty must be (B, {self.difficulty_dim}); got "
                    f"{tuple(difficulty.shape)}"
                )
            cond = torch.cat([pose, difficulty], dim=1)
        gamma = 1.0 + self.gamma_proj(cond)
        return x * gamma.view(-1, self.num_features, 1, 1)


class PactNervIa3MultiSubstrate(nn.Module):
    """Pact-NeRV-IA3-Multi renderer (L0 SKETCH; multi-block IA3 + difficulty)."""

    def __init__(self, cfg: PactNervIa3MultiConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        self.ego_poses = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.pose_dim).normal_(std=0.02)
        )
        # Per-pair difficulty signal (HARD-EARNED-CANONICAL-EQUATION;
        # initialized to mean-ish via small normal, learned at train time).
        self.difficulties = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.difficulty_dim).normal_(std=0.05)
        )

        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )
        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError("decoder_channels too short for num_upsample_blocks")
        blocks: list[nn.Module] = []
        ia3_mods: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
            # Multi-block: IA3 modulation at EVERY upsample block (vs single block).
            ia3_mods.append(
                IA3MultiGammaOnlyModulation(
                    num_features=channels[i + 1],
                    pose_dim=cfg.pose_dim,
                    difficulty_dim=cfg.difficulty_dim,
                    init_delta_std=cfg.ia3_init_delta_std,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.ia3_mods = nn.ModuleList(ia3_mods)

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
                    # Skip IA3 γ_proj (already zero-init per IA3 §3.2)
                    if any(m is mod.gamma_proj for mod in self.ia3_mods):
                        continue
                    fan_in = m.in_features
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(
        self,
        pair_indices: torch.Tensor,
        ego_pose: torch.Tensor | None = None,
        difficulty: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(f"pair_indices out of range [0, {self.cfg.num_pairs})")

        z = self.latents[pair_indices]
        pose = self.ego_poses[pair_indices] if ego_pose is None else ego_pose
        diff = self.difficulties[pair_indices] if difficulty is None else difficulty

        h = self.latent_embed(z)
        h = h.view(
            -1, self.cfg.embed_dim, self.cfg.initial_grid_h, self.cfg.initial_grid_w
        )
        # Per-block forward + multi-block IA3 γ-only modulation
        for block, ia3 in zip(self.blocks, self.ia3_mods):
            h = block(h)
            h = ia3(h, pose, diff)

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

    def num_ia3_modulation_parameters(self) -> int:
        return sum(
            p.numel() for ia3 in self.ia3_mods for p in ia3.parameters()
            if p.requires_grad
        )
