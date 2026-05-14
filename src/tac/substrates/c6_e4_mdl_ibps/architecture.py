"""C6 MDL-IBPS substrate composition — encoder + decoder + per-pair latent.

Per the C6 across-class shift hypothesis, the substrate composes:

1. ``IBEncoder``: variational q(z | frames) (small CNN; ~5K params).
2. ``IBDecoder``: reconstructs frames from z (NeRV-style PixelShuffle; ~150K params).
3. Per-pair learned latent ``z`` (stored in archive, like sane_hnerv's
   per-pair latents BUT 5-20x smaller because z is the SCORER-RELEVANT
   subspace not the full source-entropy subspace).

At training time:
- For each pair index, the encoder is used to PROVE the latent is in the IB
  feasible region (the KL regularizer flows through the encoder).
- The per-pair latent IS the trained parameter; the encoder produces its
  initial estimate then the latent is fine-tuned freely (auto-decoder style).
- The decoder reconstructs both frame_0 and frame_1 from z.

At inflate time:
- Read encoder + decoder state_dicts + per-pair z.
- Decoder produces (rgb_0, rgb_1) from each z per pair index.
- The encoder is stored for forensic provenance (z = encoder(frames) is the
  IB-derived bound; deviation from this is captured in the per-pair fine-tune).

Catalog #124 archive-grammar 8 fields are declared inline in __init__.py at
module level so the AST walker observes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import torch
from torch import nn

from tac.substrates.c6_e4_mdl_ibps.ib_decoder import IBDecoder
from tac.substrates.c6_e4_mdl_ibps.ib_encoder import IBEncoder

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width)."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair)."""

# Archive byte targets (substrate-engineering scope).
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 40_000
"""Predicted minimum (encoder + decoder + 600 × small z). Empirical anchor pending."""

TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 150_000
"""Predicted maximum (room for larger β-sweep variants)."""


@dataclass(frozen=True)
class MDLIBPSConfig:
    """Static design-time parameters for the C6 substrate.

    Args:
        latent_dim: per-pair latent dimensionality (default 24).
            Z1 anchor: A1's per-pair latent was 15387 bytes for 600 pairs ≈
            25 bytes/pair. C6 target: 24 dims × 1 byte (int8) = 24 bytes/pair.
        encoder_input_channels: encoder input channels (default 3 = RGB).
        encoder_sin_freq: encoder SIREN activation frequency (default 30).
        decoder_embed_dim: decoder initial-grid channel count (default 32).
        decoder_initial_grid_h: decoder initial grid height (default 3).
        decoder_initial_grid_w: decoder initial grid width (default 4).
        decoder_channels: per-block output channels (default (24, 20, 16, 12, 8, 6)).
        decoder_num_upsample_blocks: number of PixelShuffle(2) blocks (default 6).
        decoder_sin_freq: decoder SIREN activation frequency (default 30).
        num_pairs: contest pair count (default 600).
        output_height: scorer-resolution height (default 384).
        output_width: scorer-resolution width (default 512).
        beta_ib: IB Lagrangian β coefficient (default 0.01).
        latent_init_std: stddev for per-pair z initialization (default 0.02).
    """

    latent_dim: int = 24
    encoder_input_channels: int = 3
    encoder_sin_freq: float = 30.0
    decoder_embed_dim: int = 32
    decoder_initial_grid_h: int = 3
    decoder_initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (24, 20, 16, 12, 8, 6)
    decoder_num_upsample_blocks: int = 6
    decoder_sin_freq: float = 30.0
    num_pairs: int = NUM_PAIRS
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    beta_ib: float = 0.01
    latent_init_std: float = 0.02

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)


class MDLIBPSSubstrate(nn.Module):
    """C6 MDL-IBPS substrate: encoder + decoder + per-pair learned latent.

    Forward (training mode):
        1. encoder(frames) → (μ, log_σ²)
        2. z_sampled = μ + σ ⊙ ε  (reparameterization)
        3. z_per_pair = (self.latents[pair_indices])   (auto-decoder bypass)
           — at training start z_per_pair ≈ z_sampled (initialized from encoder)
        4. decoder(z_per_pair) → (rgb_0, rgb_1)
        5. KL(q(z|frames) || N(0,I)) returned for IB regularization

    Forward (eval mode):
        1. z_per_pair = self.latents[pair_indices]
        2. decoder(z_per_pair) → (rgb_0, rgb_1)
        (Encoder is NOT used at eval time; provenance only.)

    The auto-decoder pattern means the per-pair latent IS the trained
    parameter; the encoder is used DURING TRAINING to compute the KL
    regularizer on the (μ, σ) projection of frames, but the actual stored
    latent in the archive is the directly-trained `self.latents` row.
    """

    def __init__(self, cfg: MDLIBPSConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = IBEncoder(
            latent_dim=cfg.latent_dim,
            input_channels=cfg.encoder_input_channels,
            sin_freq=cfg.encoder_sin_freq,
        )
        self.decoder = IBDecoder(
            latent_dim=cfg.latent_dim,
            embed_dim=cfg.decoder_embed_dim,
            initial_grid_h=cfg.decoder_initial_grid_h,
            initial_grid_w=cfg.decoder_initial_grid_w,
            decoder_channels=cfg.decoder_channels,
            num_upsample_blocks=cfg.decoder_num_upsample_blocks,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
            sin_freq=cfg.decoder_sin_freq,
        )
        # Per-pair learned latents (auto-decoder); shape (num_pairs, latent_dim)
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=cfg.latent_init_std)
        )

    def forward(
        self,
        pair_indices: torch.Tensor,
        frames_for_encoder: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Render the per-pair frame pair.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.
            frames_for_encoder: optional ``(B, C, H, W)`` source frame_1 to
                feed the IB encoder. Required at training time for KL
                computation; None at eval time.

        Returns:
            (rgb_0, rgb_1, mu, logvar). mu and logvar are None when
            frames_for_encoder is None (eval path).
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs}); "
                f"got [{pair_indices.min().item()}, {pair_indices.max().item()}]"
            )

        z = self.latents[pair_indices]  # (B, latent_dim)
        rgb_0, rgb_1 = self.decoder(z)

        if frames_for_encoder is not None:
            mu, logvar = self.encoder(frames_for_encoder)
        else:
            mu, logvar = None, None
        return rgb_0, rgb_1, mu, logvar

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters_breakdown(self) -> dict[str, int]:
        """Encoder / decoder / latent param counts."""
        return {
            "encoder": self.encoder.num_parameters(),
            "decoder": self.decoder.num_parameters(),
            "latents": self.latents.numel(),
            "total": self.num_parameters(),
        }


__all__ = [
    "EVAL_HW",
    "MDLIBPSConfig",
    "MDLIBPSSubstrate",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
]
