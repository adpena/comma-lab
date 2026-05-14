"""Z5 predictive-coding world-model substrate composition.

Per the Time-Traveler L5 staircase Step 3 design, Z5 builds on Z4's
encoder+decoder+per-pair latent grammar by inserting a recurrent
hierarchical predictor between successive per-pair latents. The encoded
representation per pair becomes:

    z_t = predictor(z_{t-1}, ego_motion_t) + residual_t

At inflate time, the predictor regenerates the predicted ``z_t`` from
``z_{t-1}`` plus ego-motion, then ADDS the stored ``residual_t`` to
recover the actual ``z_t``. Decoding proceeds as in Z4.

The predictor itself is a small 2-3 layer recurrent network (GRU-like
with explicit ego-motion conditioning). Its parameters are stored ONCE
in the archive (~50KB FP4 quantized) and provide compounding savings
when the residual entropy << absolute latent entropy.

Architecture overview:

    z_0           ←   self.latent_init (stored as latent_init_blob)
    for t in 1..T:
        z_t_pred  =   predictor(z_{t-1}, ego_motion[t])
        z_t       =   z_t_pred + residual[t]      (residual stored in archive)
        rgb_0, rgb_1 = decoder(z_t)

For training, ``z_t`` is the auto-decoder parameter (the trained tensor).
The predictor learns to minimize the L2 norm of residual[t] across t.

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

# Archive byte targets.
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 80_000
"""Predicted minimum (Z4 baseline + predictor params - residual savings)."""

TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 250_000
"""Predicted maximum (predictor + larger residual encoding)."""


@dataclass(frozen=True)
class PredictiveCodingConfig:
    """Static design-time parameters for the Z5 substrate.

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
        predictor_hidden_dim: hierarchical predictor hidden dim (default 64).
        predictor_num_layers: predictor depth — 2 or 3 (default 2).
        predictor_ego_motion_dim: ego-motion projection dim (default 8).
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
    predictor_num_layers: int = 2
    predictor_ego_motion_dim: int = 8
    identity_predictor: bool = False
    lambda_residual_entropy: float = 1.0
    latent_init_std: float = 0.02

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)


class HierarchicalPredictor(nn.Module):
    """2-3 layer recurrent predictor: ``predict z_t from z_{t-1} + ego_motion``.

    Per Rao-Ballard 1999, hierarchical predictive-coding requires the
    predictor depth > 1 for non-trivial scene dynamics. The default 2-layer
    config maps to (state-from-prev-z + state-from-ego-motion) →
    fused-state → next-z-prediction.

    Mode 'identity' (config.identity_predictor=True): predict z_t = z_{t-1}.
    This is the ablation control for the probe-disambiguator: if the
    full-predictor variant does not beat the identity variant by ΔS > 0.005,
    the predictive-coding hypothesis is refuted.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        ego_motion_dim: int,
        identity_predictor: bool = False,
    ) -> None:
        super().__init__()
        if num_layers not in (2, 3):
            raise ValueError(
                f"predictor_num_layers must be 2 or 3; got {num_layers}"
            )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ego_motion_dim = ego_motion_dim
        self.identity_predictor = identity_predictor

        if not identity_predictor:
            # Latent → hidden projection
            self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)
            # Ego-motion → hidden projection
            self.ego_to_hidden = nn.Linear(ego_motion_dim, hidden_dim)
            # Fused-state hidden layer(s)
            layers: list[nn.Module] = []
            in_dim = hidden_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.GELU())
                in_dim = hidden_dim
            self.fused_layers = nn.Sequential(*layers)
            # Final projection to next-z
            self.hidden_to_z = nn.Linear(hidden_dim, latent_dim)
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
        z_hidden = self.z_to_hidden(z_prev)
        ego_hidden = self.ego_to_hidden(ego_motion)
        fused = torch.tanh(z_hidden + ego_hidden)
        fused = self.fused_layers(fused)
        return self.hidden_to_z(fused)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _Z5Encoder(nn.Module):
    """Encoder reused from Z4 pattern."""

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


class _Z5Decoder(nn.Module):
    """Decoder reused from Z4 pattern."""

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


class PredictiveCodingSubstrate(nn.Module):
    """Z5 predictive-coding substrate.

    Composition:
    - encoder (small CNN)
    - predictor (hierarchical 2-3 layer recurrent net)
    - decoder (PixelShuffle NeRV-style)
    - latent_init (stored z_0)
    - residuals (per-pair stored r_t)

    Training mode forward (per pair):
        For pair index t:
          1. If t == 0: z_t = self.latent_init
          2. If t > 0:  z_t_pred = predictor(z_{t-1}, ego_motion[t])
                       z_t      = z_t_pred + self.residuals[t]
          3. rgb_0, rgb_1 = decoder(z_t)
        Loss = score_aware(rgb) + lambda_residual * |residuals|²

    Eval mode forward:
        Same — predictor must be applied at inflate time (residuals
        are stored in the archive; predictor is stored in the archive).
    """

    def __init__(self, cfg: PredictiveCodingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _Z5Encoder(
            input_channels=cfg.encoder_input_channels,
            hidden_dim=cfg.encoder_hidden_dim,
            latent_dim=cfg.latent_dim,
        )
        self.decoder = _Z5Decoder(
            latent_dim=cfg.latent_dim,
            embed_dim=cfg.decoder_embed_dim,
            initial_grid_h=cfg.decoder_initial_grid_h,
            initial_grid_w=cfg.decoder_initial_grid_w,
            decoder_channels=cfg.decoder_channels,
            num_upsample_blocks=cfg.decoder_num_upsample_blocks,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
        )
        self.predictor = HierarchicalPredictor(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.predictor_hidden_dim,
            num_layers=cfg.predictor_num_layers,
            ego_motion_dim=cfg.predictor_ego_motion_dim,
            identity_predictor=cfg.identity_predictor,
        )
        # z_0 — the initial state
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

    def reconstruct_pair(
        self,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct (rgb_0, rgb_1, z_t) for the requested pair indices.

        Per-pair-batch reconstruction (Catalog #218). The predictor is
        applied autoregressively from pair 0 up to max(pair_indices),
        then z_t is selected at the requested indices and decoded.

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
        # z_history is a list of (1, latent_dim) tensors; stack to (max_idx+1, latent_dim)
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
    "HierarchicalPredictor",
    "NUM_PAIRS",
    "PredictiveCodingConfig",
    "PredictiveCodingSubstrate",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
]
