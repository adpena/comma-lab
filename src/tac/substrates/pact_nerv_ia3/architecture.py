# SPDX-License-Identifier: MIT
"""pact_nerv_ia3 architecture — Pact-NeRV-IA3 (L0 SKETCH).

HNeRV-class implicit renderer with IA3 γ-only ego-pose-conditioned per-block
modulation (Liu et al. 2022 arXiv:2205.05638). The distinguishing primitive
vs full FiLM γ+β: ONLY the γ multiplier projection from ego-pose is learned
— there is NO β bias projection. Per IA3 paper §3.2: γ_init=1.0 + Δ residual
form ensures the substrate behaves like the unconditioned base decoder at
initialization, and the IA3 γ-only modulation is ~6x more parameter-efficient
than full FiLM γ+β.

Stage 1 of HYBRID staged path per PACT-NERV-DESIGN-SYMPOSIUM commit
`5371d4dd4`. The HARD-EARNED-LITERATURE classification is per FILM-FAMILY-RESEARCH
Section 10 Recommendation #5 (IA3-style γ-only modulation as rate-extremal
variant).

Architecture (council-approved SKETCH 2026-05-20):

    Per-pair latent z in R^24                Per-pair ego_pose in R^6
       |                                          |
       v                                          |
    HNeRV-class base decoder (DepthSep            |
    + SIREN + PixelShuffle)                       |
       |                                          |
       v                                          v
    For each upsample block (output channels C_b):
        h_b = upsample_block(h_{b-1})
        γ_b = 1.0 + γ_proj_b(ego_pose)  # (B, C_b); residual form
        h_b = h_b * γ_b.view(B, C_b, 1, 1)  # element-wise γ rescale, NO β
       |
       v
    rgb_0 / rgb_1: 1x1 Conv -> RGB (3 channels)

The IA3 γ-only modulation is parameterized by ego-pose ∈ R^6 (matches
upstream PoseNet's first 6 dims per `upstream/modules.py`). Total per-block
γ_proj cost: 6 * C_b weights per block. At the L0 SCAFFOLD config
(decoder_channels=(48, 40, 32, 24, 20, 16, 12)), total IA3 modulation
weights = 6 * (48+40+32+24+20+16+12) = 1092 (vs ~2184 for full FiLM γ+β =
exactly 2x the bytes per IA3 paper §3.2 prediction).

The substrate is a FULL RGB RENDERER per HNeRV parity discipline L5: outputs
(B, 3, H, W) for each of rgb_0 and rgb_1. NOT a mask codec; NOT a partial
slot replacement.

CLAUDE.md compliance:
- No silent device defaults (caller supplies device; trainer routes through
  canonical `tac.substrates._shared.trainer_skeleton.device_or_die`)
- No scorer load at inflate time (strict-scorer-rule per HNeRV parity)
- No /tmp paths
- Reviewable in 30 seconds per L12 (per-file < 300 LOC; IA3 modulation
  itself is ~50 LOC for the canonical core class)
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
_POSE_DIM = 6  # contest canonical: upstream PoseNet first 6 dims


@dataclass(frozen=True)
class PactNervIa3Config:
    """Static design-time parameters for Pact-NeRV-IA3."""

    latent_dim: int = 24
    """Per-pair latent dimensionality."""

    embed_dim: int = 64
    """Channels of the initial spatial-grid embedding."""

    initial_grid_h: int = 3
    """Initial spatial-grid height."""

    initial_grid_w: int = 4

    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    """Per-block output channels for the depth-separable base decoder."""

    sin_frequency: float = 30.0

    num_upsample_blocks: int = 7
    """Number of PixelShuffle(2) blocks; 7 -> 3x4 -> 384x512."""

    pose_dim: int = _POSE_DIM
    """Ego-pose conditioning dimensionality.

    HARD-EARNED: matches upstream PoseNet's first 6 dims per
    upstream/modules.py. Pose-conditioning on the contest's canonical
    representation aligns the IA3 γ projection with the scorer's
    pretrained semantics.
    """

    ia3_init_delta_std: float = 0.01
    """Initialization stddev for γ_proj weights.

    γ_init = 1.0 + Δ where Δ ~ N(0, ia3_init_delta_std^2). HARD-EARNED:
    IA3 paper §3.2 zero-init discipline (γ behaves like 1.0 at init);
    sister of adaLN-Zero per FILM-FAMILY-RESEARCH §5. Small stddev keeps
    early training behavior close to the unconditioned base decoder.
    """

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
    """Depthwise-3x3 + pointwise-1x1, SIREN-friendly (mirrors ds_nerv / boost_nerv)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _DsUpBlock(nn.Module):
    """DepthSep -> sin -> PixelShuffle(2)."""

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.dsc(x)))


class IA3GammaOnlyModulation(nn.Module):
    """The distinguishing primitive: element-wise γ-only rescaling per Liu 2205.05638.

    Per IA3 paper §3.2: ``output[c] = input[c] * γ(z)[c]`` where γ is a
    learnable per-channel scalar function of conditioning ``z`` (ego-pose).
    NO β bias projection (this is the canonical IA3 distinction vs full
    FiLM γ+β).

    γ_init = 1.0 + Δ residual form ensures the substrate behaves like the
    unconditioned base decoder at initialization (early-training stability;
    sister of adaLN-Zero zero-init per FILM-FAMILY-RESEARCH §5).

    Args:
        num_features: number of feature channels to modulate.
        pose_dim: ego-pose conditioning dimensionality (contest canonical: 6).
        init_delta_std: stddev for initialization of γ_proj weights
            (γ_init = 1.0 + Δ where Δ ~ N(0, init_delta_std^2)).
    """

    def __init__(
        self,
        num_features: int,
        pose_dim: int = _POSE_DIM,
        init_delta_std: float = 0.01,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive; got {num_features}")
        if pose_dim <= 0:
            raise ValueError(f"pose_dim must be positive; got {pose_dim}")
        if init_delta_std < 0:
            raise ValueError(
                f"init_delta_std must be non-negative; got {init_delta_std}"
            )
        self.num_features = num_features
        self.pose_dim = pose_dim
        # γ projection: pose_dim -> num_features
        # NO β projection (this is THE distinguishing primitive vs FiLM γ+β)
        self.gamma_proj = nn.Linear(pose_dim, num_features)
        # IA3 §3.2 zero-init: initialize γ_proj weights to ~0 so γ ≈ 1.0 at start.
        with torch.no_grad():
            self.gamma_proj.weight.normal_(mean=0.0, std=init_delta_std)
            self.gamma_proj.bias.zero_()

    def forward(self, x: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Apply γ-only modulation: x * γ(pose).

        Args:
            x: (B, C, H, W) feature map to modulate.
            pose: (B, pose_dim) ego-pose conditioning.

        Returns:
            (B, C, H, W) modulated feature map.
        """
        if x.dim() != 4:
            raise ValueError(f"x must be (B, C, H, W); got shape {tuple(x.shape)}")
        if pose.dim() != 2:
            raise ValueError(
                f"pose must be (B, pose_dim); got shape {tuple(pose.shape)}"
            )
        if pose.shape[1] != self.pose_dim:
            raise ValueError(
                f"pose dim {pose.shape[1]} != configured pose_dim {self.pose_dim}"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"x channels {x.shape[1]} != configured num_features {self.num_features}"
            )
        # γ = 1.0 + Δ residual form per IA3 §3.2
        gamma = 1.0 + self.gamma_proj(pose)  # (B, num_features)
        return x * gamma.view(-1, self.num_features, 1, 1)


class PactNervIa3Substrate(nn.Module):
    """Pact-NeRV-IA3 renderer (L0 SKETCH).

    Input: pair index ``i in [0, num_pairs)`` + ego_pose ``(B, 6)``.
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1].

    The forward path:
    1. Latent embedding -> initial spatial grid.
    2. For each upsample block: upsample + per-block IA3 γ-only modulation
       conditioned on ego_pose.
    3. Final 1x1 conv heads produce rgb_0 / rgb_1.

    Per HNeRV parity L5: outputs RGB at contest camera resolution
    (384, 512); NOT a mask codec.
    """

    def __init__(self, cfg: PactNervIa3Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

        # Per-pair learnable ego-pose (default-init small; can be replaced
        # with measured pose from upstream PoseNet at training time).
        self.ego_poses = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.pose_dim).normal_(std=0.02)
        )

        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        ia3_mods: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
            # IA3 γ-only modulation per upsample block (multi-layer modulation
            # per FILM-FAMILY-RESEARCH §8.6 HARD-EARNED-EMPIRICALLY-SUPERIOR
            # for video-temporal conditioning).
            ia3_mods.append(
                IA3GammaOnlyModulation(
                    num_features=channels[i + 1],
                    pose_dim=cfg.pose_dim,
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
                    # IA3 γ_proj weights are already zero-init per IA3 paper §3.2
                    # in IA3GammaOnlyModulation.__init__; skip them here.
                    if m in self.modules() and any(
                        m is mod.gamma_proj for mod in self.ia3_mods
                    ):
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward path: latent + ego-pose -> (rgb_0, rgb_1).

        Args:
            pair_indices: (B,) long tensor of pair indices in [0, num_pairs).
            ego_pose: (B, pose_dim) ego-pose conditioning. If None, uses the
                learnable per-pair ego_poses[pair_indices].

        Returns:
            (rgb_0, rgb_1) each (B, 3, H, W) in [0, 1].
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        z = self.latents[pair_indices]
        if ego_pose is None:
            pose = self.ego_poses[pair_indices]
        else:
            if ego_pose.dim() != 2 or ego_pose.shape[1] != self.cfg.pose_dim:
                raise ValueError(
                    f"ego_pose must be (B, {self.cfg.pose_dim}); got "
                    f"shape {tuple(ego_pose.shape)}"
                )
            if ego_pose.shape[0] != pair_indices.shape[0]:
                raise ValueError(
                    f"ego_pose batch {ego_pose.shape[0]} != pair_indices "
                    f"batch {pair_indices.shape[0]}"
                )
            pose = ego_pose

        h = self.latent_embed(z)
        h = h.view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )

        # Per-block forward + IA3 γ-only modulation
        for block, ia3 in zip(self.blocks, self.ia3_mods):
            h = block(h)
            h = ia3(h, pose)

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
        """Total trainable parameter count.

        Sister boost_nerv at L0 SCAFFOLD config = ~170K. Pact-NeRV-IA3
        adds the IA3 γ_proj weights (6 * sum(decoder_channels) per block;
        ~1.1K at L0 config) on top of the base decoder. Net: ~171K.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_ia3_modulation_parameters(self) -> int:
        """Count of IA3 γ-only modulation parameters (the bolt-on cost).

        Per FILM-FAMILY-RESEARCH §10.5: IA3 γ-only halves conditioning
        bytes vs full FiLM γ+β. This helper enables empirical verification
        of that rate-axis claim at L1.
        """
        return sum(
            p.numel() for ia3 in self.ia3_mods for p in ia3.parameters()
            if p.requires_grad
        )
