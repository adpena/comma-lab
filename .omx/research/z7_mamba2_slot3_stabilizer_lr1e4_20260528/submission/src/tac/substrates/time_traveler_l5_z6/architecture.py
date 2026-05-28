# SPDX-License-Identifier: MIT
"""Z6 Time-Traveler L5 F-asymptote node — FiLM-conditioned next-frame predictor architecture.

Per the Z6/Z7/Z8 design memo Section 4.1, Z6 implements the SIMPLEST viable
predictive-coding architecture (single-layer FiLM-conditioned next-frame
predictor). Architecture forward pass per pair t:

    z_0           ←   self.latent_init (stored as latent_init_blob)
    for t in 1..T:
        z_t_pred  =   predictor(z_{t-1}, ego_motion[t])
        z_t       =   z_t_pred + residual[t]      (residual stored in archive)
        rgb_0, rgb_1 = decoder(z_t)

For training, ``z_t`` is the auto-decoder parameter (the trained tensor); the
FiLM predictor learns to minimize the L2 norm of residual[t] across t.

Distinction from sister Z5 architecture:
- Z5 uses a 2-3 layer hierarchical recurrent predictor (~150-200 LOC, ~120K
  params); Z6 uses a SINGLE FiLM-conditioned conv block (~75K params).
- FiLM modulation (Perez 2017) maps ego_motion -> (scale, shift) per channel
  on the predictor's conv output; simpler than recurrent state.
- The simplification trades a small amount of predictive power for
  substantially lower engineering risk (sister Z5 pattern was sister-tested
  but still pends Phase 2 council approval; Z6 explicitly opts for the
  MINIMUM viable predictive-coding architecture as the FIRST F-asymptote-
  class-shift empirical anchor).

Catalog #124 archive-grammar 8 fields are declared inline in __init__.py at
module level so the AST walker observes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width)."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair)."""

# Archive byte targets per design memo Section 4.1 + Section 10.
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 80_000
"""Predicted minimum archive bytes (target ~97 KB design memo Section 10)."""

TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 250_000
"""Predicted maximum archive bytes (sister Z5 ceiling for safety)."""


@dataclass(frozen=True)
class Z6PredictiveCodingConfig:
    """Static design-time parameters for the Z6 substrate.

    Args:
        latent_dim: per-pair latent dimensionality (default 24).
        encoder_input_channels: encoder input channels (default 3 = RGB).
        encoder_hidden_dim: encoder hidden state dimension (default 64).
        decoder_embed_dim: decoder initial-grid channel count (default 32).
        decoder_initial_grid_h: decoder initial grid height (default 3).
        decoder_initial_grid_w: decoder initial grid width (default 4).
        decoder_channels: per-block output channels.
        decoder_num_upsample_blocks: number of PixelShuffle(2) blocks (default 6).
        num_pairs: contest pair count (default 600).
        output_height: scorer-resolution height (default 384).
        output_width: scorer-resolution width (default 512).
        predictor_hidden_dim: FiLM predictor conv hidden dim (default 64).
        predictor_film_mlp_hidden_dim: FiLM MLP hidden dim (default 32).
        predictor_ego_motion_dim: ego-motion projection dim (default 8).
        predictor_kernel_size: FiLM predictor conv kernel size (default 3).
        identity_predictor: when True, predictor is identity (no learning;
            ablation probe-disambiguator regime).
        lambda_residual_entropy: weight on residual L2 / entropy proxy in
            the training Lagrangian (default 1.0).
        latent_init_std: stddev for z_0 initialization (default 0.02).
    """

    latent_dim: int = 24
    encoder_input_channels: int = 3
    encoder_hidden_dim: int = 64
    decoder_embed_dim: int = 32
    decoder_initial_grid_h: int = 3
    decoder_initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (24, 20, 16, 12, 8, 6)
    decoder_num_upsample_blocks: int = 6
    num_pairs: int = NUM_PAIRS
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    predictor_hidden_dim: int = 64
    predictor_film_mlp_hidden_dim: int = 32
    predictor_ego_motion_dim: int = 8
    predictor_kernel_size: int = 3
    identity_predictor: bool = False
    lambda_residual_entropy: float = 1.0
    latent_init_std: float = 0.02
    # Z6-v2 Candidate 1 extension per Phase 3 council §9. depth=1 preserves
    # Z6-v1 backward compatibility (FilmConditionedNextFramePredictor); depth>=2
    # uses MultiLayerFilmPredictor per the Path B BUILD design memo §4.1.
    predictor_depth: int = 1

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)


class FilmConditionedNextFramePredictor(nn.Module):
    """Z6 substrate-distinguishing primitive: FiLM-conditioned next-frame predictor.

    Per design memo Section 4.1: single-layer FiLM-conditioned conv that
    predicts ``z_t`` from ``z_{t-1} + ego_motion[t]``. The FiLM MLP maps the
    ego-motion vector to per-channel (scale, shift) modulation parameters.

    Architecture:

        ego_motion (B, ego_motion_dim)
                |
                v [FiLM MLP: Linear -> SiLU -> Linear]
                |
        (scale, shift)  each (B, latent_dim)
                |
        z_prev (B, latent_dim)
                |
                v [project latent -> 1x1 spatial via reshape]
                |
        z_prev_spatial (B, latent_dim, 1, 1)
                |
                v [Conv2d kernel_size=k stride=1 padding=k//2]
                |
        h (B, hidden_dim, 1, 1)
                |
                v [FiLM modulation: h * scale + shift via broadcast]
                |
        modulated (B, hidden_dim, 1, 1)
                |
                v [Project hidden -> latent via 1x1 conv]
                |
        z_pred (B, latent_dim)

    Mode 'identity' (config.identity_predictor=True): predict z_t = z_{t-1}.
    This is the ablation control for the probe-disambiguator (Catalog #125
    hook #6): if the full-predictor variant does not beat the identity
    variant by ΔS > 0.005, the predictive-coding hypothesis is refuted.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        film_mlp_hidden_dim: int,
        ego_motion_dim: int,
        kernel_size: int = 3,
        identity_predictor: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size not in (1, 3, 5):
            raise ValueError(
                f"predictor_kernel_size must be 1, 3, or 5; got {kernel_size}"
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                f"predictor_kernel_size must be odd for symmetric padding; "
                f"got {kernel_size}"
            )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.film_mlp_hidden_dim = film_mlp_hidden_dim
        self.ego_motion_dim = ego_motion_dim
        self.kernel_size = kernel_size
        self.identity_predictor = identity_predictor

        if not identity_predictor:
            # FiLM modulation network: ego_motion -> (scale, shift)
            self.film_mlp = nn.Sequential(
                nn.Linear(ego_motion_dim, film_mlp_hidden_dim),
                nn.SiLU(),
                nn.Linear(film_mlp_hidden_dim, hidden_dim * 2),
            )
            # Latent -> hidden conv (single-layer per design memo Section 4.1)
            self.input_conv = nn.Conv2d(
                latent_dim, hidden_dim, kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            # Hidden -> latent projection (1x1 conv)
            self.output_conv = nn.Conv2d(
                hidden_dim, latent_dim, kernel_size=1, padding=0,
            )
        # When identity_predictor=True we have no trainable parameters

    def forward(
        self,
        z_prev: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> torch.Tensor:
        """Predict z_t from z_{t-1} + ego_motion.

        Args:
            z_prev: ``(B, latent_dim)``.
            ego_motion: ``(B, ego_motion_dim)``.

        Returns:
            ``(B, latent_dim)`` predicted z_t.
        """
        if z_prev.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z_prev last dim {z_prev.shape[-1]} != latent_dim {self.latent_dim}"
            )
        if ego_motion.shape[-1] != self.ego_motion_dim:
            raise ValueError(
                f"ego_motion last dim {ego_motion.shape[-1]} != ego_motion_dim "
                f"{self.ego_motion_dim}"
            )
        if self.identity_predictor:
            return z_prev

        batch = z_prev.shape[0]
        # FiLM modulation parameters
        film_params = self.film_mlp(ego_motion)  # (B, 2*hidden_dim)
        scale, shift = film_params.chunk(2, dim=-1)  # each (B, hidden_dim)
        scale = scale.view(batch, self.hidden_dim, 1, 1)
        shift = shift.view(batch, self.hidden_dim, 1, 1)

        # Project latent -> spatial for conv
        z_prev_spatial = z_prev.view(batch, self.latent_dim, 1, 1)
        # Single-layer conv (the predictor backbone per design memo Section 4.1)
        h = self.input_conv(z_prev_spatial)  # (B, hidden_dim, 1, 1)
        # FiLM modulation: h * scale + shift (broadcast)
        h = h * scale + shift
        h = torch.tanh(h)
        # Project hidden -> latent
        z_pred_spatial = self.output_conv(h)  # (B, latent_dim, 1, 1)
        return z_pred_spatial.view(batch, self.latent_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiLayerFilmPredictor(nn.Module):
    """Z6-v2 Candidate 1 — Depth-3 multi-layer FiLM stack predictor.

    Per Z6 Path B BUILD design memo §4.1 + Phase 3 council §9 Wave 2 spec:
    extends ``FilmConditionedNextFramePredictor`` from depth=1 (~75K params
    total substrate including encoder/decoder/latent_init) to depth=3 with
    per-layer ego-conditioning. Target predictor block param count ~236K so
    the full substrate weighs ~300K total per the council's binding ceiling.

    Architecture per design memo §4.1::

        ego_motion (B, ego_motion_dim)
                |
        z_prev (B, latent_dim)
                |
                v [project latent -> spatial (B, latent_dim, 1, 1)]
                |
                v [Layer 1: Conv k=3 -> FiLM(γ_1, β_1 from MLP_ego_1) -> ReLU]
                |
                v [Layer 2: Conv k=3 -> FiLM(γ_2, β_2 from MLP_ego_2) -> ReLU]
                |
                v [Layer 3: Conv k=3 -> FiLM(γ_3, β_3 from MLP_ego_3)]
                |
                v [Project hidden -> latent via 1x1 conv]
                |
        z_pred (B, latent_dim)

    Per Rao-Ballard 1999 hierarchical predictive coding: depth=3 single-
    spatial-resolution FiLM approximates 3-level hierarchy where Layer 1
    captures coarse motion (large-scale ego-aligned transformations); Layer
    2 captures mid-scale structure (per-region object motion); Layer 3
    captures fine-scale residuals (per-pixel adjustments). Each layer's FiLM
    gate is independently ego-modulated so the predictor can express
    conditional motion-magnitude scaling at each scale.

    Per Tishby IB framework: I(T;Y) increases with predictor capacity until
    saturation; Z6-v1's 75K-param single-layer FiLM likely fell below the
    saturation point on the contest scorer; ~300K params sits at the typical
    saturation range for similar conditional-image-prediction tasks per the
    Perez 2017 FiLM ablations on CLEVR.

    Atick critique (per Phase 3 council Revision #6): this depth=3 stack
    PRESERVES Z6-v1's PoseNet ego sidecar; it does NOT enrich the side-
    information channel. If Wave 2 ΔS < 0.005 / identity-WIN, Candidate 4c
    (scorer-logit conditioning) per the design memo §4.4c becomes the next
    pivot. The capacity unwind addresses CC-1 + CC-2 (single FiLM stem) per
    the design memo §3 cargo-cult audit; the side-info enrichment is a
    distinct CC-5 unwind addressed by sister Candidate 4c if Wave 2 DEFERs.

    Mode ``identity_predictor=True``: returns ``z_prev`` unchanged with no
    trainable parameters. Used by the identity-predictor disambiguator probe
    per Council Revision #2 (SAME-archive-bytes comparison test).
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        film_mlp_hidden_dim: int,
        ego_motion_dim: int,
        kernel_size: int = 3,
        depth: int = 3,
        identity_predictor: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size not in (1, 3, 5):
            raise ValueError(
                f"predictor_kernel_size must be 1, 3, or 5; got {kernel_size}"
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                f"predictor_kernel_size must be odd for symmetric padding; "
                f"got {kernel_size}"
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.film_mlp_hidden_dim = film_mlp_hidden_dim
        self.ego_motion_dim = ego_motion_dim
        self.kernel_size = kernel_size
        self.depth = depth
        self.identity_predictor = identity_predictor

        if not identity_predictor:
            # Per-layer FiLM modulation MLPs: ego_motion -> (scale, shift)
            # independent per depth-level so each layer has its own ego
            # conditioning per Rao-Ballard 1999 hierarchical predictive coding.
            self.film_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(ego_motion_dim, film_mlp_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(film_mlp_hidden_dim, hidden_dim * 2),
                )
                for _ in range(depth)
            ])
            # Per-layer conv blocks. Layer 0 is the input projection from
            # latent_dim to hidden_dim; subsequent layers are hidden->hidden.
            self.convs = nn.ModuleList()
            for layer_idx in range(depth):
                in_channels = latent_dim if layer_idx == 0 else hidden_dim
                self.convs.append(
                    nn.Conv2d(
                        in_channels, hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
            # Final projection hidden_dim -> latent_dim (1x1 conv per Z6-v1)
            self.output_conv = nn.Conv2d(
                hidden_dim, latent_dim, kernel_size=1, padding=0,
            )

    def forward(
        self,
        z_prev: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> torch.Tensor:
        """Predict z_t from z_{t-1} + ego_motion via depth-N FiLM stack.

        Args:
            z_prev: ``(B, latent_dim)``.
            ego_motion: ``(B, ego_motion_dim)``.

        Returns:
            ``(B, latent_dim)`` predicted z_t.
        """
        if z_prev.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z_prev last dim {z_prev.shape[-1]} != latent_dim {self.latent_dim}"
            )
        if ego_motion.shape[-1] != self.ego_motion_dim:
            raise ValueError(
                f"ego_motion last dim {ego_motion.shape[-1]} != ego_motion_dim "
                f"{self.ego_motion_dim}"
            )
        if self.identity_predictor:
            return z_prev

        batch = z_prev.shape[0]
        # Project latent -> spatial for conv stack
        h = z_prev.view(batch, self.latent_dim, 1, 1)

        for layer_idx in range(self.depth):
            # FiLM modulation parameters for this layer
            film_params = self.film_mlps[layer_idx](ego_motion)  # (B, 2*hidden_dim)
            scale, shift = film_params.chunk(2, dim=-1)
            scale = scale.view(batch, self.hidden_dim, 1, 1)
            shift = shift.view(batch, self.hidden_dim, 1, 1)
            # Per-layer conv
            h = self.convs[layer_idx](h)
            # FiLM modulation: h * scale + shift (broadcast)
            h = h * scale + shift
            # Per-layer non-linearity. Last layer uses tanh consistent with
            # the Z6-v1 sister; intermediate layers use ReLU per the design
            # memo §4.1 ASCII diagram.
            if layer_idx == self.depth - 1:
                h = torch.tanh(h)
            else:
                h = torch.relu(h)

        # Project hidden -> latent via 1x1 conv
        z_pred_spatial = self.output_conv(h)  # (B, latent_dim, 1, 1)
        return z_pred_spatial.view(batch, self.latent_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _Z6Encoder(nn.Module):
    """Encoder reused from Z5 pattern (small CNN -> latent_dim projection)."""

    def __init__(self, *, input_channels: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.stem = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.head_mu = nn.Linear(hidden_dim, latent_dim)
        self.head_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.dim() != 4:
            raise ValueError(
                f"encoder expects (B, C, H, W); got shape {tuple(frames.shape)}"
            )
        feats = self.stem(frames)
        pooled = feats.mean(dim=(2, 3))
        return self.head_mu(pooled), self.head_logvar(pooled)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _Z6Decoder(nn.Module):
    """Decoder reused from Z5 PixelShuffle NeRV pattern."""

    def __init__(
        self,
        *,
        latent_dim: int,
        embed_dim: int,
        initial_grid_h: int,
        initial_grid_w: int,
        decoder_channels: tuple[int, ...],
        num_upsample_blocks: int,
        output_height: int,
        output_width: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.initial_grid_h = initial_grid_h
        self.initial_grid_w = initial_grid_w
        self.decoder_channels = tuple(int(c) for c in decoder_channels)
        self.num_upsample_blocks = int(num_upsample_blocks)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        if len(self.decoder_channels) < self.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels must have >= num_upsample_blocks entries; "
                f"got {len(self.decoder_channels)} for {self.num_upsample_blocks} blocks"
            )
        self.initial_proj = nn.Linear(
            latent_dim, embed_dim * initial_grid_h * initial_grid_w
        )
        blocks: list[nn.Module] = []
        in_ch = embed_dim
        for i in range(self.num_upsample_blocks):
            out_ch = self.decoder_channels[i]
            blocks.append(nn.Conv2d(in_ch, 4 * out_ch, kernel_size=3, padding=1))
            blocks.append(nn.PixelShuffle(2))
            blocks.append(nn.ReLU(inplace=False))
            in_ch = out_ch
        blocks.append(nn.Conv2d(in_ch, 6, kernel_size=3, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"decoder expects (B, latent_dim={self.latent_dim}); got {tuple(z.shape)}"
            )
        batch = z.shape[0]
        flat = self.initial_proj(z)
        grid = flat.view(batch, self.embed_dim, self.initial_grid_h, self.initial_grid_w)
        out = self.blocks(grid)
        if out.shape[-2] != self.output_height or out.shape[-1] != self.output_width:
            out = torch.nn.functional.interpolate(
                out,
                size=(self.output_height, self.output_width),
                mode="bilinear",
                align_corners=False,
            )
        out = torch.sigmoid(out)
        rgb_0 = out[:, :3, :, :]
        rgb_1 = out[:, 3:, :, :]
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Z6PredictiveCodingSubstrate(nn.Module):
    """Z6 FiLM-conditioned predictive-coding substrate.

    Composition (per design memo Section 4.1):
    - encoder (small CNN)
    - predictor (single FiLM-conditioned conv block)
    - decoder (PixelShuffle NeRV-style)
    - latent_init (stored z_0)
    - residuals (per-pair stored r_t)
    - ego_motion_buffer (per-pair stored ego-motion vector)

    Training mode forward (per pair):
        For pair index t:
          1. If t == 0: z_t = self.latent_init
          2. If t > 0:  z_t_pred = predictor(z_{t-1}, ego_motion[t])
                       z_t      = z_t_pred + self.residuals[t]
          3. rgb_0, rgb_1 = decoder(z_t)
        Loss = score_aware(rgb) + lambda_residual * |residuals|²

    Inflate mode: same recurrence; predictor weights loaded from
    predictor_blob; residuals loaded from residuals_blob; ego_motion
    loaded from ego_motion_blob.
    """

    def __init__(self, cfg: Z6PredictiveCodingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _Z6Encoder(
            input_channels=cfg.encoder_input_channels,
            hidden_dim=cfg.encoder_hidden_dim,
            latent_dim=cfg.latent_dim,
        )
        self.decoder = _Z6Decoder(
            latent_dim=cfg.latent_dim,
            embed_dim=cfg.decoder_embed_dim,
            initial_grid_h=cfg.decoder_initial_grid_h,
            initial_grid_w=cfg.decoder_initial_grid_w,
            decoder_channels=cfg.decoder_channels,
            num_upsample_blocks=cfg.decoder_num_upsample_blocks,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
        )
        # Initialize shared trainable state before the mode-specific predictor.
        # This keeps full-FiLM vs identity-predictor controls apples-to-apples:
        # with the same seed, shared decoder/latent/residual tensors match even
        # though the identity predictor has no trainable parameters.
        self.latent_init = nn.Parameter(
            torch.empty(cfg.latent_dim).normal_(std=cfg.latent_init_std)
        )
        # Per-pair residuals; residuals[0] is unused (z_0 := latent_init)
        self.residuals = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=cfg.latent_init_std)
        )
        # Per-pair ego-motion proxy. Sourced from the trainer (e.g.
        # projected PoseNet output) and stored as a fixed buffer at inflate
        # time. At training time the trainer pins the ego_motion buffer.
        self.register_buffer(
            "ego_motion_buffer",
            torch.zeros(cfg.num_pairs, cfg.predictor_ego_motion_dim),
            persistent=True,
        )
        # Z6-v2 Candidate 1 dispatch: pick depth=1 (Z6-v1 backward-compat) or
        # depth>=2 (multi-layer FiLM per Path B BUILD design memo §4.1).
        if cfg.predictor_depth <= 1:
            self.predictor = FilmConditionedNextFramePredictor(
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.predictor_hidden_dim,
                film_mlp_hidden_dim=cfg.predictor_film_mlp_hidden_dim,
                ego_motion_dim=cfg.predictor_ego_motion_dim,
                kernel_size=cfg.predictor_kernel_size,
                identity_predictor=cfg.identity_predictor,
            )
        else:
            self.predictor = MultiLayerFilmPredictor(
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.predictor_hidden_dim,
                film_mlp_hidden_dim=cfg.predictor_film_mlp_hidden_dim,
                ego_motion_dim=cfg.predictor_ego_motion_dim,
                kernel_size=cfg.predictor_kernel_size,
                depth=cfg.predictor_depth,
                identity_predictor=cfg.identity_predictor,
            )

    def reconstruct_pair(
        self,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct (rgb_0, rgb_1, z_t) for the requested pair indices.

        Per-pair-batch reconstruction (Catalog #218). The predictor is
        applied autoregressively from pair 0 up to max(pair_indices), then
        z_t is selected at the requested indices and decoded.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.

        Returns:
            (rgb_0, rgb_1, z_at_indices) each on the substrate's device.
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs}); "
                f"got [{pair_indices.min().item()}, {pair_indices.max().item()}]"
            )

        max_idx = int(pair_indices.max().item())
        # Roll forward the autoregressive predictor from 0 to max_idx
        z_history: list[torch.Tensor] = [self.latent_init.unsqueeze(0)]  # (1, latent_dim)
        for t in range(1, max_idx + 1):
            z_prev = z_history[-1]  # (1, latent_dim)
            ego_t = self.ego_motion_buffer[t].unsqueeze(0)  # (1, ego_motion_dim)
            z_pred = self.predictor(z_prev, ego_t)
            z_t = z_pred + self.residuals[t].unsqueeze(0)
            z_history.append(z_t)

        # Select z at requested indices
        z_stacked = torch.cat(z_history, dim=0)
        z_at_indices = z_stacked[pair_indices]  # (B, latent_dim)

        rgb_0, rgb_1 = self.decoder(z_at_indices)
        return rgb_0, rgb_1, z_at_indices

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters_breakdown(self) -> dict[str, int]:
        """Encoder / decoder / predictor / latent_init / residuals counts."""
        return {
            "encoder": self.encoder.num_parameters(),
            "decoder": self.decoder.num_parameters(),
            "predictor": self.predictor.num_parameters(),
            "latent_init": self.latent_init.numel(),
            "residuals": self.residuals.numel(),
            "total": self.num_parameters(),
        }


__all__ = [
    "EVAL_HW",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "FilmConditionedNextFramePredictor",
    "MultiLayerFilmPredictor",
    "Z6PredictiveCodingConfig",
    "Z6PredictiveCodingSubstrate",
]
