# SPDX-License-Identifier: MIT
"""hi_nerv architecture — Hierarchical NeRV with multi-scale latents (L0 SKETCH).

Per-frame implicit renderer with a 3-scale latent pyramid. The decoder runs
coarse-to-fine: the coarse latent seeds the initial spatial grid; the mid
latent is added at the mid-decode stage; the fine latent is added near
the final RGB heads. The substrate's distinctive prior is multi-resolution
representation.

Architecture (council-approved SKETCH 2026-05-12):

    Per-pair coarse latent z_c in R^16
    Per-pair mid    latent z_m in R^20
    Per-pair fine   latent z_f in R^24
                |
                v
    Linear z_c -> 384                          # initial freq embed
                |
                v
    Reshape (1, 384, 3, 4)
                |
                v
    Block 0..2: Conv -> sin -> PixelShuffle(2) # 6x8 -> 12x16 -> 24x32 (coarse path)
                |
                + z_m via Linear -> spatial broadcast (mid-injection)
                |
                v
    Block 3..4: Conv -> sin -> PixelShuffle(2) # 48x64 -> 96x128 (mid path)
                |
                + z_f via Linear -> spatial broadcast (fine-injection)
                |
                v
    Block 5..6: Conv -> sin -> PixelShuffle(2) # 192x256 -> 384x512 (fine path)
                |
                v
    Head rgb_0 / rgb_1: Conv 3 channels each

~240K params target (3 latent scales x ~80K decoder each).

CLAUDE.md compliance:
- No silent device defaults
- No scorer load
- No /tmp paths
- Reviewable in 30 seconds per L12
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2
LATENT_STATE_KEYS: tuple[str, str, str] = (
    "latents_coarse",
    "latents_mid",
    "latents_fine",
)
HINERV_OFFICIAL_FEATURE_GRID_CONVNEXT_PROOF: str = (
    "receiver_visible_hierarchical_feature_grid_convnext_trilinear_v1"
)
HINERV_LOCAL_FEATURE_GRID_CONVNEXT_RECEIVER_PROOF: str = (
    "receiver_visible_local_hierarchical_feature_grid_convnext_modulo_sampler_v1"
)
HINERV_OFFICIAL_HIERARCHICAL_RENDERER_SOURCE_FORWARD_PROOF: str = (
    "official_hinerv_core_forward_replay_maps_or_rejects_local_portable_v1"
)
HINERV_FEATURE_GRID_CONVNEXT_SOURCE_FORWARD_BLOCKERS: tuple[str, ...] = (
    "hinerv_official_feature_grid_source_forward_parity_missing",
    "hinerv_core_hierarchical_renderer_source_forward_replay_missing",
    "hinerv_local_feature_grid_sampler_differs_from_official_grid_trilinear3d",
)


def hi_nerv_feature_grid_convnext_authority_status() -> dict[str, object]:
    """Return the false-authority split for the local feature-grid path."""

    return {
        "schema": "hinerv_feature_grid_convnext_authority_status.v1",
        "local_receiver_visible": True,
        "local_receiver_proof": HINERV_LOCAL_FEATURE_GRID_CONVNEXT_RECEIVER_PROOF,
        "legacy_receiver_proof_alias": HINERV_OFFICIAL_FEATURE_GRID_CONVNEXT_PROOF,
        "official_grid_trilinear3d_temporal_only_bound": True,
        "local_runtime_uses_modulo_sampler": True,
        "official_core_forward_parity_proven": False,
        "blockers": HINERV_FEATURE_GRID_CONVNEXT_SOURCE_FORWARD_BLOCKERS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


@dataclass(frozen=True)
class HinervConfig:
    """Static design-time parameters for hi_nerv with a 3-scale latent pyramid."""

    latent_dim_coarse: int = 16
    """Coarse-scale latent dimensionality."""

    latent_dim_mid: int = 20
    """Mid-scale latent dimensionality."""

    latent_dim_fine: int = 24
    """Fine-scale latent dimensionality."""

    embed_dim: int = 64

    initial_grid_h: int = 3
    initial_grid_w: int = 4

    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)

    sin_frequency: float = 30.0

    num_upsample_blocks: int = 7

    mid_injection_block_index: int = 2
    """After which decoder block to inject z_m (0-indexed; 2 means after block 2)."""

    fine_injection_block_index: int = 4
    """After which decoder block to inject z_f."""

    num_pairs: int = _PAIRS

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W

    use_hierarchical_feature_grid: bool = False
    """Enable receiver-visible HiNeRV temporal-local feature grids."""

    use_convnext_blocks: bool = False
    """Enable ConvNeXt-style refinement after each upsample/encoding step."""

    local_grid_levels: int = 2
    """Number of temporal-local feature-grid levels per upsample block."""

    local_grid_channels: int = 4
    """Base channels per feature-grid level before the 1x1 projection."""

    convnext_mlp_ratio: int = 2
    """Expansion ratio for the ConvNeXt-style pointwise MLP."""

    convnext_kernel_size: int = 7
    """Depthwise kernel size for the ConvNeXt-style block."""

    use_bilinear_skip: bool = False
    """PR95 HF-residual paths (F1; deep_hinerv_snerv_fidelity_review_20260609 H1).

    When True each ``_UpBlock`` becomes ``sin(w*(PixelShuffle(conv(x)) +
    skip(bilinear_2x(x))))`` (the per-block bilinear residual that carries the
    coarse signal forward so the conv branch only learns the HF correction) AND
    the renderer applies a terminal ``h += refine_residual_scale*sin(refine(h))``
    HF residual before the RGB heads. Default OFF preserves the historical
    skip-free NeRV carrier byte-for-byte (no new params when False). The PR95
    reference (model.py:46-51) is the canonical source; verified MISSING in our
    carrier (mlx_renderer.py:561-577) and the direct mechanical cause of the
    diverse-but-blurry mean-field that collapses SegNet argmax to d_seg~0.50."""

    refine_residual_scale: float = 0.1
    """Scale on the terminal ``sin(refine(h))`` HF residual (PR95 uses 0.1)."""

    init_seed: int = 0
    """Deterministic MLX/Torch initialization seed for train-time reproducibility."""


def trilinear_upsample(
    grid: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    num_pairs: int,
    target_h: int,
    target_w: int,
    local_scale: int,
) -> torch.Tensor:
    """Sample a temporal-local grid into ``(B,H,W,C)`` features.

    HiNeRV's local encoding is indexed by video time and by local coordinates
    inside the current upsample cell.  The spatial local coordinates are exact
    modulo indices; the temporal axis is linearly interpolated so pair indices
    can address coarser temporal grids without hidden side channels.
    """

    if grid.dim() != 4:
        raise ValueError(f"grid must be (T,S,S,C), got {tuple(grid.shape)}")
    time_bins, scale_h, scale_w, channels = (int(v) for v in grid.shape)
    if scale_h != int(local_scale) or scale_w != int(local_scale):
        raise ValueError(
            f"grid local scale {(scale_h, scale_w)} != {int(local_scale)}"
        )
    if time_bins <= 0 or channels <= 0:
        raise ValueError("grid must have positive temporal bins and channels")
    if pair_indices.dtype != torch.long:
        raise ValueError("pair_indices must be torch.long")
    if pair_indices.dim() != 1:
        raise ValueError("pair_indices must be 1-D")

    device = grid.device
    dtype = grid.dtype
    denom = max(int(num_pairs) - 1, 1)
    t = pair_indices.to(device=device, dtype=torch.float32) * (
        float(time_bins - 1) / float(denom)
    )
    t0 = torch.floor(t).to(dtype=torch.long).clamp(0, time_bins - 1)
    t1 = torch.clamp(t0 + 1, max=time_bins - 1)
    alpha = (t - t0.to(dtype=torch.float32)).to(dtype=dtype).view(-1, 1, 1, 1)
    temporal = grid[t0] * (1.0 - alpha) + grid[t1] * alpha

    ys = torch.arange(int(target_h), device=device, dtype=torch.long) % int(local_scale)
    xs = torch.arange(int(target_w), device=device, dtype=torch.long) % int(local_scale)
    flat_local = (ys[:, None] * int(local_scale) + xs[None, :]).reshape(-1)
    flat = temporal.reshape(int(pair_indices.numel()), int(local_scale) ** 2, channels)
    sampled = flat[:, flat_local, :]
    return sampled.reshape(int(pair_indices.numel()), int(target_h), int(target_w), channels)


class HierarchicalFeatureGrid(nn.Module):
    """Receiver-visible temporal-local feature grid from official HiNeRV."""

    def __init__(
        self,
        *,
        num_pairs: int,
        local_scale: int,
        base_channels: int,
        levels: int,
        out_channels: int,
        reduction: int,
    ) -> None:
        super().__init__()
        if levels <= 0:
            raise ValueError("local_grid_levels must be positive")
        if local_scale <= 0:
            raise ValueError("local_scale must be positive")
        self.num_pairs = int(num_pairs)
        self.local_scale = int(local_scale)
        self.levels = int(levels)
        self.base_channels = int(base_channels)
        self.level_channels: list[int] = []
        grids: list[nn.Parameter] = []
        for level in range(self.levels):
            time_bins = max(2, math.ceil(self.num_pairs / float(2**level)))
            channels = max(1, (self.base_channels * (2**level)) // max(1, reduction))
            self.level_channels.append(int(channels))
            grid = torch.empty(
                time_bins,
                self.local_scale,
                self.local_scale,
                channels,
            )
            grids.append(nn.Parameter(grid.normal_(std=0.02)))
        self.grids = nn.ParameterList(grids)
        self.proj = nn.Conv2d(sum(self.level_channels), int(out_channels), kernel_size=1)

    def forward(
        self,
        pair_indices: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        sampled = [
            trilinear_upsample(
                grid,
                pair_indices,
                num_pairs=self.num_pairs,
                target_h=h,
                target_w=w,
                local_scale=self.local_scale,
            )
            for grid in self.grids
        ]
        enc = torch.cat(sampled, dim=-1).permute(0, 3, 1, 2).to(dtype=dtype)
        return self.proj(enc)


class _LayerNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(channels)))
        self.bias = nn.Parameter(torch.zeros(int(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_nhwc = x.permute(0, 2, 3, 1)
        y = F.layer_norm(x_nhwc, (int(x.shape[1]),), self.weight, self.bias)
        return y.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """Small ConvNeXt-style depthwise block used after HiNeRV local encoding."""

    def __init__(self, channels: int, *, mlp_ratio: int, kernel_size: int) -> None:
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("convnext_kernel_size must be positive and odd")
        hidden = max(channels, channels * max(1, int(mlp_ratio)))
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = _LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), 1.0e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = F.gelu(self.pwconv1(x))
        x = self.pwconv2(x)
        return residual + self.gamma * x


class _SinAct(nn.Module):
    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.conv(x)))


class _LatentInjector(nn.Module):
    """Project a per-pair latent into a (B, C, H, W) spatial-additive tensor."""

    def __init__(self, latent_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, channels)

    def forward(
        self, latent: torch.Tensor, spatial_shape: tuple[int, int]
    ) -> torch.Tensor:
        # latent: (B, latent_dim) -> (B, channels) -> (B, channels, 1, 1) -> broadcast
        v = self.proj(latent)
        h, w = spatial_shape
        return v.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)


class HinervSubstrate(nn.Module):
    """Hierarchical NeRV with 3-scale latent pyramid.

    Default mode preserves the historical local HIV1 archive grammar.  Enabling
    ``use_hierarchical_feature_grid`` and ``use_convnext_blocks`` adds the
    official HiNeRV-style temporal-local grid encoding and lightweight
    ConvNeXt-style refinement as receiver-visible decoder parameters.
    """

    def __init__(self, cfg: HinervConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if not 0 <= cfg.mid_injection_block_index < cfg.num_upsample_blocks:
            raise ValueError(
                f"mid_injection_block_index {cfg.mid_injection_block_index} "
                f"out of range [0, {cfg.num_upsample_blocks})"
            )
        if not 0 <= cfg.fine_injection_block_index < cfg.num_upsample_blocks:
            raise ValueError(
                f"fine_injection_block_index {cfg.fine_injection_block_index} "
                f"out of range [0, {cfg.num_upsample_blocks})"
            )
        if cfg.fine_injection_block_index <= cfg.mid_injection_block_index:
            raise ValueError(
                "fine_injection_block_index must be > mid_injection_block_index"
            )
        if cfg.use_hierarchical_feature_grid and cfg.local_grid_levels <= 0:
            raise ValueError("local_grid_levels must be positive")
        if cfg.use_hierarchical_feature_grid and cfg.local_grid_channels <= 0:
            raise ValueError("local_grid_channels must be positive")
        if cfg.use_convnext_blocks and cfg.convnext_mlp_ratio <= 0:
            raise ValueError("convnext_mlp_ratio must be positive")
        if cfg.use_convnext_blocks and (
            cfg.convnext_kernel_size <= 0 or cfg.convnext_kernel_size % 2 == 0
        ):
            raise ValueError("convnext_kernel_size must be positive and odd")

        # Per-pair learned latents at 3 scales
        self.latents_coarse = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim_coarse).normal_(std=0.02)
        )
        self.latents_mid = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim_mid).normal_(std=0.02)
        )
        self.latents_fine = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim_fine).normal_(std=0.02)
        )

        # Coarse latent embedded into the initial spatial grid
        self.latent_embed = nn.Linear(
            cfg.latent_dim_coarse,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_UpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)
        feature_grids: list[nn.Module] = []
        convnext_blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            out_ch = channels[i + 1]
            if cfg.use_hierarchical_feature_grid:
                feature_grids.append(
                    HierarchicalFeatureGrid(
                        num_pairs=cfg.num_pairs,
                        local_scale=2,
                        base_channels=cfg.local_grid_channels,
                        levels=cfg.local_grid_levels,
                        out_channels=out_ch,
                        reduction=max(1, i + 1),
                    )
                )
            else:
                feature_grids.append(nn.Identity())
            if cfg.use_convnext_blocks:
                convnext_blocks.append(
                    ConvNeXtBlock(
                        out_ch,
                        mlp_ratio=cfg.convnext_mlp_ratio,
                        kernel_size=cfg.convnext_kernel_size,
                    )
                )
            else:
                convnext_blocks.append(nn.Identity())
        self.feature_grids = nn.ModuleList(feature_grids)
        self.convnext_blocks = nn.ModuleList(convnext_blocks)

        # Latent injectors: project to the channel count at the injection point
        mid_inject_channels = channels[cfg.mid_injection_block_index + 1]
        fine_inject_channels = channels[cfg.fine_injection_block_index + 1]
        self.mid_injector = _LatentInjector(cfg.latent_dim_mid, mid_inject_channels)
        self.fine_injector = _LatentInjector(cfg.latent_dim_fine, fine_inject_channels)

        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)

        self._siren_init()

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
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

        z_c = self.latents_coarse[pair_indices]
        z_m = self.latents_mid[pair_indices]
        z_f = self.latents_fine[pair_indices]

        h = self.latent_embed(z_c).view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )

        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.cfg.use_hierarchical_feature_grid:
                h = h + self.feature_grids[i](
                    pair_indices,
                    spatial_shape=(h.shape[-2], h.shape[-1]),
                    dtype=h.dtype,
                )
            if self.cfg.use_convnext_blocks:
                h = self.convnext_blocks[i](h)
            if i == self.cfg.mid_injection_block_index:
                h = h + self.mid_injector(z_m, (h.shape[-2], h.shape[-1]))
            if i == self.cfg.fine_injection_block_index:
                h = h + self.fine_injector(z_f, (h.shape[-2], h.shape[-1]))

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
        """Total trainable parameter count (target ~240K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def expected_decoder_state_shapes(cfg: HinervConfig) -> dict[str, tuple[int, ...]]:
    """Return required non-latent decoder state keys and tensor shapes."""

    model = HinervSubstrate(cfg)
    return {
        key: tuple(int(dim) for dim in tensor.shape)
        for key, tensor in model.state_dict().items()
        if key not in LATENT_STATE_KEYS
    }


def validate_decoder_state_dict(
    decoder_state_dict: Mapping[str, torch.Tensor],
    cfg: HinervConfig,
    *,
    context: str = "hi_nerv_decoder_state",
) -> None:
    """Fail closed if decoder state is incomplete, extra, or shape-corrupt."""

    expected = expected_decoder_state_shapes(cfg)
    given = {
        str(key): tuple(int(dim) for dim in tensor.shape)
        for key, tensor in decoder_state_dict.items()
    }
    missing = tuple(sorted(set(expected) - set(given)))
    unexpected = tuple(sorted(set(given) - set(expected)))
    shape_mismatch = tuple(
        sorted(
            key
            for key in expected.keys() & given.keys()
            if expected[key] != given[key]
        )
    )
    if missing or unexpected or shape_mismatch:
        detail = {
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatch": {
                key: {"expected": expected[key], "actual": given[key]}
                for key in shape_mismatch
            },
        }
        raise ValueError(f"{context} invalid: {detail}")
