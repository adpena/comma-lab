# SPDX-License-Identifier: MIT
"""boost_nerv_pr110_residual architecture — MLX residual learner (L0 SCAFFOLD).

Per binding 2026-05-26 reframing: MLX-first per Catalog #1265. The residual
learner is a TINY MLP-on-RGB conditioned on PR110-extracted per-pair latent.
PyTorch state_dict export (Catalog #1251 sister bridge) is produced at
archive-build time; MLX trainer is the canonical training loop.

Design memo: .omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md
"""

from __future__ import annotations

from dataclasses import dataclass

_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2


@dataclass(frozen=True)
class BoostNervPr110ResidualConfig:
    """Static design-time parameters for boost_nerv_pr110_residual.

    Per Catalog #290 canonical-vs-unique decision per layer (see design memo):
    these defaults are CARGO-CULTED at L0 and require empirical sweep at L1.
    """

    pr110_latent_dim: int = 24
    """Dimensionality of the per-pair latent EXTRACTED FROM PR110's archive.

    PR110 archive carries a per-frame latent code (HNeRV-family); we project
    its frame_0 latent to this dim and use it as conditioning input to the
    residual learner. NOT a fresh latent.
    """

    residual_hidden_dim: int = 12
    """TinyConv hidden channel count for the residual head.

    Mirrors sister `boost_nerv/_BoostingHead` shape (canonical residual MLP
    primitive across NeRV family). At L0 the cost is ~3K params per head.
    """

    num_boosting_rounds: int = 1
    """Number of iterative residual-refinement rounds applied on top of
    PR110-base reconstructions.

    L0 default = 1; L1 sweep = 1/2/3 rounds (CARGO-CULTED per design memo).
    """

    boosting_gain_clamp: float = 0.05
    """Per-round residual clamp magnitude (tighter than sister boost_nerv's
    0.10 because residual is on top of contest-grade base; less room to
    perturb without destroying PR110's 0.193 frontier baseline)."""

    residual_spatial_h: int = 96
    """Internal residual spatial-grid height before bilinear upsample to
    (384, 512) contest output. Downsampling is the rate-vs-fidelity tradeoff
    primitive (CARGO-CULTED at L0; sweep 48/96/192 at L1)."""

    residual_spatial_w: int = 128

    residual_quant_bits: int = 8
    """Per-pixel residual quantization bit-depth (int8 default; sweep int4/int8/FP4 at L1)."""

    num_pairs: int = _PAIRS

    output_height: int = _CONTEST_H

    output_width: int = _CONTEST_W


def _import_mlx_nn():
    """Lazy MLX import — keeps shape tests cheap on machines without mlx installed."""
    import mlx.nn as nn

    return nn


def _import_mlx_core():
    """Lazy MLX core import."""
    import mlx.core as mx

    return mx


class ResidualHeadMLX:
    """Tiny MLX residual head: TinyConv(rgb_pr110_base, z_pr110_proj) -> residual.

    Per Catalog #290 canonical-vs-unique decision: this MLP shape MIRRORS
    sister `boost_nerv/_BoostingHead` (canonical residual-head primitive
    across NeRV family) but is implemented in MLX rather than PyTorch.

    Conditioning input is the PR110-extracted per-pair latent z_pr110
    (NOT a fresh latent), keeping the residual codec's contribution to
    rate term ≤ residual blob bytes (no per-pair latent table in BPR1).
    """

    def __init__(self, cfg: BoostNervPr110ResidualConfig) -> None:
        self.cfg = cfg
        nn = _import_mlx_nn()
        self.z_proj = nn.Linear(cfg.pr110_latent_dim, cfg.residual_hidden_dim)
        self.conv1 = nn.Conv2d(
            in_channels=3 + cfg.residual_hidden_dim,
            out_channels=cfg.residual_hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=cfg.residual_hidden_dim,
            out_channels=3,
            kernel_size=1,
        )

    def forward(self, rgb_pr110_base, z_pr110):
        """Forward pass.

        Inputs (MLX-native NHWC convention; sister Catalog #1265 cascade
        uses the same NHWC convention for MLX-side scorer preprocess):
            rgb_pr110_base: (B, H_residual, W_residual, 3) MLX array — PR110
                base reconstruction downsampled to internal residual grid.
            z_pr110: (B, pr110_latent_dim) MLX array — per-pair latent
                extracted from PR110 archive.

        Output:
            residual: (B, H_residual, W_residual, 3) MLX array in [-1, 1]
                (tanh-bounded; subsequent code clamps to ±boosting_gain_clamp).
        """
        mx = _import_mlx_core()
        nn = _import_mlx_nn()

        z_emb = self.z_proj(z_pr110)  # (B, hidden_dim)
        # Broadcast z_emb across spatial grid (NHWC: channel dim last).
        b, c = z_emb.shape
        h_grid, w_grid = rgb_pr110_base.shape[1], rgb_pr110_base.shape[2]
        z_grid = mx.broadcast_to(
            z_emb[:, None, None, :], (b, h_grid, w_grid, c)
        )

        h = mx.concatenate([rgb_pr110_base, z_grid], axis=-1)
        h = nn.relu(self.conv1(h))
        residual = mx.tanh(self.conv2(h))
        return residual


def compose_pr110_base_plus_residual(rgb_pr110_base, residual, gain_clamp: float):
    """Canonical composition: rgb_composed = clamp(rgb_pr110_base + clamp(residual, ±gain), 0, 1).

    This is the ONE-LINE math at the heart of "boosting against PR110": the
    frozen base learner output is preserved; the residual learner ADDS a
    bounded correction. Both inputs are MLX arrays in [0, 1] / [-1, 1]
    respectively; output is MLX in [0, 1].
    """
    mx = _import_mlx_core()
    residual_clamped = mx.clip(residual, -gain_clamp, gain_clamp)
    composed = mx.clip(rgb_pr110_base + residual_clamped, 0.0, 1.0)
    return composed


def num_residual_parameters(cfg: BoostNervPr110ResidualConfig) -> int:
    """Theoretical parameter count for the residual learner only.

    PR110 base is FROZEN and EXTERNAL — its parameters do NOT count toward
    the substrate's rate term (those bytes are PR110's; we preserve them
    unchanged). The boosting substrate's parameter count is the residual
    learner alone.

    Per-round residual head:
        z_proj: pr110_latent_dim × hidden_dim + hidden_dim (Linear bias)
        conv1: (3 + hidden_dim) × hidden_dim × 3 × 3 + hidden_dim
        conv2: hidden_dim × 3 × 1 × 1 + 3
    """
    h = cfg.residual_hidden_dim
    z_proj = cfg.pr110_latent_dim * h + h
    conv1 = (3 + h) * h * 3 * 3 + h
    conv2 = h * 3 * 1 * 1 + 3
    per_round = z_proj + conv1 + conv2
    return per_round * cfg.num_boosting_rounds
