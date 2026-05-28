# SPDX-License-Identifier: MIT
"""Z5 Rao-Ballard 2-level hierarchical predictive coding substrate (PyTorch).

Per T4 SYMPOSIUM Wave N+13 verdict + Catalog #311 + Catalog #312 hierarchical
predictive coding canonical quadruple + Rao-Ballard 1999 bidirectional
inference framework.

Architecture (1:1 mirrored by ``mlx_renderer.Z5RaoBallardSubstrateMLX``):

    Per pair index t in [0, T):
        z_low_t      = self.low_latents[t]            (per-pair learnable)
        z_high_t     = self.high_latents[t]           (per-pair learnable)
        ego_motion_t = self.ego_vecs[t]               (per-pair learnable)
        z_low_pred   = predictor(z_high_t, ego_motion_t)
        residual_t   = z_low_t - z_low_pred           (training: minimized)
        rgb_0, rgb_1 = decoder(z_low_t)               (Z6-style FiLM + PixelShuffle)

The training Lagrangian (Catalog #311 cooperative-receiver + Catalog #312
hierarchical quadruple):

    L = lambda_recon * MSE(rgb_pred, rgb_target)
      + lambda_residual * ||residual_t||_2^2
      + lambda_scorer * score_aware_loss(rgb_pred, rgb_target)  [Catalog #164]

At inflate time the residual is reconstructed from the stored z_low + z_high
+ ego_motion (since we store z_low directly, the residual is implicit; at the
SCAFFOLD level we ship the low_latents directly per HNeRV parity L4 inflate
budget; the Phase 2 entropy-coding pass would store residuals instead — that
is the canonical bit-savings mechanism per Catalog #344 equation).

[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/architecture.py]
[verified-against: src/tac/substrates/z5_predictive_coding_world_model/architecture.py]
[verified-against: Rao+Ballard 1999 hierarchical predictive coding]
"""
# AUTOCAST_FP16_WAIVED:substrate_engineering_scaffold_no_pytorch_cuda_autocast_per_mlx_first_canonical_doctrine
# TF32_WAIVED:substrate_engineering_scaffold_no_pytorch_cuda_tf32_per_mlx_first_canonical_doctrine
# TORCH_COMPILE_WAIVED:substrate_engineering_scaffold_research_only_no_torch_compile
# NO_GRAD_WAIVED:substrate_engineering_uses_pytorch_autograd_for_pytorch_path_only_mlx_uses_value_and_grad
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width)."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair)."""

Z5_TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 90_000
"""Predicted minimum (decoder + low_latents + high_latents + ego + predictor FP4)."""

Z5_TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 240_000
"""Predicted maximum (FP16 weights + per-pair latents at 600 pairs)."""


@dataclass(frozen=True)
class Z5RaoBallardConfig:
    """Z5 Rao-Ballard 2-level hierarchical config (PyTorch + MLX parity)."""

    low_latent_dim: int = 24
    high_latent_dim: int = 16
    ego_dim: int = 6
    embed_dim: int = 32
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (24, 20, 16, 12, 8, 6, 4)
    num_upsample_blocks: int = 7
    sin_frequency: float = 30.0
    film_generator_depth: int = 3
    film_hidden_width: int = 24
    num_pairs: int = NUM_PAIRS
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    predictor_hidden_dim: int = 48
    predictor_num_layers: int = 2
    lambda_residual: float = 1.0
    cooperative_receiver_beta: float = 0.10  # Catalog #311 Atick-Redlich

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)


class _Z5HierarchicalPredictor(nn.Module):
    """2-level Rao-Ballard predictor: ``(z_high, ego_motion) -> z_low_pred``.

    Per Rao+Ballard 1999, the high-level prediction must condition on BOTH
    the higher-level latent AND an ego-motion proxy (focus-of-expansion;
    FoE) per Catalog #311. The default 2-layer config maps to:

        h = tanh(W_high @ z_high + W_ego @ ego_motion)
        h = GELU(W_hidden @ h)
        z_low_pred = W_out @ h
    """

    def __init__(
        self,
        *,
        high_latent_dim: int,
        low_latent_dim: int,
        ego_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers not in (2, 3):
            raise ValueError(f"predictor_num_layers must be 2 or 3; got {num_layers}")
        self.high_latent_dim = int(high_latent_dim)
        self.low_latent_dim = int(low_latent_dim)
        self.ego_dim = int(ego_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.high_to_hidden = nn.Linear(high_latent_dim, hidden_dim)
        self.ego_to_hidden = nn.Linear(ego_dim, hidden_dim)
        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.hidden_to_low = nn.Linear(hidden_dim, low_latent_dim)

    def forward(
        self, z_high: torch.Tensor, ego_motion: torch.Tensor
    ) -> torch.Tensor:
        h = torch.tanh(self.high_to_hidden(z_high) + self.ego_to_hidden(ego_motion))
        h = self.hidden_layers(h)
        return self.hidden_to_low(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _Z5Decoder(nn.Module):
    """Z6-style PixelShuffle decoder reading z_low (low-level latent)."""

    def __init__(self, cfg: Z5RaoBallardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.initial_proj = nn.Linear(
            cfg.low_latent_dim, cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w
        )
        in_ch = cfg.embed_dim
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            out_ch = cfg.decoder_channels[i]
            blocks.append(nn.Conv2d(in_ch, 4 * out_ch, kernel_size=3, padding=1))
            blocks.append(nn.PixelShuffle(2))
            blocks.append(nn.GELU())
            in_ch = out_ch
        blocks.append(nn.Conv2d(in_ch, 6, kernel_size=3, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z_low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z_low.dim() != 2 or z_low.shape[1] != self.cfg.low_latent_dim:
            raise ValueError(
                f"decoder expects (B, low_latent_dim={self.cfg.low_latent_dim}); "
                f"got {tuple(z_low.shape)}"
            )
        batch = z_low.shape[0]
        flat = self.initial_proj(z_low)
        grid = flat.view(
            batch, self.cfg.embed_dim, self.cfg.initial_grid_h, self.cfg.initial_grid_w
        )
        out = self.blocks(grid)
        if (
            out.shape[-2] != self.cfg.output_height
            or out.shape[-1] != self.cfg.output_width
        ):
            out = torch.nn.functional.interpolate(
                out,
                size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear",
                align_corners=False,
            )
        out = torch.sigmoid(out)
        rgb_0 = out[:, :3, :, :]
        rgb_1 = out[:, 3:, :, :]
        return rgb_0, rgb_1


class Z5RaoBallardSubstrate(nn.Module):
    """Z5 PyTorch substrate (sister of MLX renderer; PyTorch-side mirror).

    Per Rao+Ballard 1999 hierarchical predictive coding: 2-level latent
    structure with z_high predicting z_low conditional on ego-motion.
    """

    def __init__(self, cfg: Z5RaoBallardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.low_latents = nn.Parameter(
            torch.randn(cfg.num_pairs, cfg.low_latent_dim) * 0.02
        )
        self.high_latents = nn.Parameter(
            torch.randn(cfg.num_pairs, cfg.high_latent_dim) * 0.02
        )
        self.ego_vecs = nn.Parameter(
            torch.randn(cfg.num_pairs, cfg.ego_dim) * 0.02
        )
        self.predictor = _Z5HierarchicalPredictor(
            high_latent_dim=cfg.high_latent_dim,
            low_latent_dim=cfg.low_latent_dim,
            ego_dim=cfg.ego_dim,
            hidden_dim=cfg.predictor_hidden_dim,
            num_layers=cfg.predictor_num_layers,
        )
        self.decoder = _Z5Decoder(cfg)

    def reconstruct_pair(self, pair_indices: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Return (rgb_0, rgb_1, residual) for the given pair indices."""
        z_low = self.low_latents[pair_indices]
        z_high = self.high_latents[pair_indices]
        ego = self.ego_vecs[pair_indices]
        z_low_pred = self.predictor(z_high, ego)
        residual = z_low - z_low_pred
        rgb_0, rgb_1 = self.decoder(z_low)
        return rgb_0, rgb_1, residual

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "EVAL_HW",
    "NUM_PAIRS",
    "Z5RaoBallardConfig",
    "Z5RaoBallardSubstrate",
    "Z5_TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "Z5_TOTAL_ARCHIVE_TARGET_BYTES_MIN",
]
