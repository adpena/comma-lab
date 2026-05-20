# SPDX-License-Identifier: MIT
"""pact_nerv_distilled_scorer architecture — Pact-NeRV-DistilledScorer (L0 SKETCH).

HNeRV-class implicit renderer with a Hinton-distilled internal scorer surrogate
(Hinton-Vinyals-Dean 2015 arXiv:1503.02531) conditioning the decoder feature
maps. The distinguishing primitive vs IA3 / FiLM: a small Conv2d-based
DistilledScorerSurrogate is co-trained via KL-T=2.0 to mimic frozen SegNet
and PoseNet logits, then the surrogate's intermediate features condition
the HNeRV upsample chain (cross-attention-light: per-block channel-bias
projection from surrogate-feature globally-pooled vector).

The surrogate IS the substrate's distinguishing structural element. At L0
SCAFFOLD the surrogate is randomly initialized; Stage 1 dispatch lands the
actual KL-T=2.0 distillation pass from frozen SegNet + PoseNet.

CLAUDE.md compliance: no MPS fallback, no inflate-time scorer load (the
surrogate IS the inflate-time conditioner — distinct from upstream SegNet/
PoseNet), no /tmp paths, reviewable in 30 seconds per L12.
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
_PAIRS = _NUM_FRAMES // 2


@dataclass(frozen=True)
class PactNervDistilledScorerConfig:
    """Static design-time parameters for Pact-NeRV-DistilledScorer."""

    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    surrogate_hidden: int = 32
    """Channel width of the distilled scorer surrogate's hidden Conv layers."""
    surrogate_feature_dim: int = 16
    """Globally-pooled feature dim the surrogate emits to condition the decoder."""
    distill_temperature: float = 2.0
    """Hinton-Vinyals-Dean 2015 §3 KL distillation temperature (T=2.0 canonical)."""
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


class DistilledScorerSurrogate(nn.Module):
    """Hinton-distilled internal scorer surrogate (the distinguishing primitive).

    Two-tower light: takes a per-pair RGB-like input (B, 3, H, W) at smoke
    resolution and emits a globally-pooled feature vector (B, feature_dim).
    At Stage 1 dispatch, this surrogate is co-trained via KL-T=2.0 distillation
    from frozen upstream SegNet + PoseNet logits per Hinton 1503.02531 §3.

    Distinguishing primitive: the surrogate features condition the HNeRV
    decoder via per-block channel-bias projection. Cross-block re-use of
    the SAME surrogate-feature vector (no per-block recomputation) keeps
    LOC bounded and aligns with the apparatus PR101 GOLD reviewability target.
    """

    def __init__(
        self,
        *,
        hidden: int = 32,
        feature_dim: int = 16,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be positive; got {hidden}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive; got {feature_dim}")
        self.hidden = hidden
        self.feature_dim = feature_dim
        # Lightweight 3-layer Conv encoder; stride 2 each layer reduces HxW 8x.
        self.conv1 = nn.Conv2d(3, hidden, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden, feature_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(
                f"x must be (B, 3, H, W); got shape {tuple(x.shape)}"
            )
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        # Global average pool -> (B, feature_dim)
        return h.mean(dim=[2, 3])


class PactNervDistilledScorerSubstrate(nn.Module):
    """Pact-NeRV-DistilledScorer renderer (L0 SKETCH).

    Forward:
    1. Latent embedding -> initial spatial grid.
    2. For each upsample block: upsample + add per-block channel bias derived
       from the distilled scorer surrogate's globally-pooled feature vector.
    3. Final 1x1 conv heads produce rgb_0 / rgb_1.

    Per HNeRV parity L5: outputs RGB at contest camera resolution.
    """

    def __init__(self, cfg: PactNervDistilledScorerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        self.surrogate_init_input = nn.Parameter(
            torch.empty(cfg.num_pairs, 3, 32, 32).normal_(std=0.02)
        )

        self.latent_embed = nn.Linear(
            cfg.latent_dim, cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w
        )

        self.surrogate = DistilledScorerSurrogate(
            hidden=cfg.surrogate_hidden, feature_dim=cfg.surrogate_feature_dim
        )

        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        bias_projs: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
            bias_projs.append(nn.Linear(cfg.surrogate_feature_dim, channels[i + 1]))
        self.blocks = nn.ModuleList(blocks)
        self.bias_projs = nn.ModuleList(bias_projs)

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

        z = self.latents[pair_indices]
        surr_in = self.surrogate_init_input[pair_indices]
        surr_feat = self.surrogate(surr_in)

        h = self.latent_embed(z)
        h = h.view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )

        for block, bias_proj in zip(self.blocks, self.bias_projs):
            h = block(h)
            bias = bias_proj(surr_feat)  # (B, C_b)
            h = h + bias.view(-1, bias.shape[1], 1, 1)

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

    def num_surrogate_parameters(self) -> int:
        """Distilled-scorer surrogate parameter count (the bolt-on cost)."""
        return sum(
            p.numel() for p in self.surrogate.parameters() if p.requires_grad
        )
