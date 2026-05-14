"""Z4 cooperative-receiver substrate composition — A1+Z3 substrate, loss-only intervention.

Per the Time-Traveler L5 staircase Step 2 design, Z4 is a LOSS-ONLY
intervention on top of the Z3 (Balle hyperprior) substrate. The architecture
is identical to Z3: encoder + decoder + per-pair latent. The intervention
happens in ``score_aware_loss.CooperativeReceiverScoreAwareLoss`` which
replaces pixel-MSE with the canonical scorer distortions through
``score_pair_components`` (Catalog #164).

Because the architecture is byte-identical to Z3, this module re-exports
the Z3 substrate primitives via a thin wrapper that adds the
``cooperative_receiver_meta`` tag to the archive meta dict for forensic
distinction at audit time. At inflate time the wrapper is transparent
(``CooperativeReceiverSubstrate`` IS a renamed ``CooperativeReceiverConfig``
+ encoder/decoder/latents).

Catalog #124 archive-grammar 8 fields are declared inline in __init__.py at
module level so the AST walker observes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import torch
from torch import nn

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width)."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair)."""

# Archive byte targets (substrate-engineering scope; inherits Z3 byte band).
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 50_000
"""Predicted minimum: ~Z3 byte band + cooperative-receiver tag (~50 bytes)."""

TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 200_000
"""Predicted maximum (inherits Z3's hyperprior overhead)."""


@dataclass(frozen=True)
class CooperativeReceiverConfig:
    """Static design-time parameters for the Z4 substrate (mirrors Z3 + cooperative-receiver tag).

    The architecture is byte-identical to Z3; this config carries the
    cooperative-receiver provenance fields used at training/audit time.

    Args:
        latent_dim: per-pair latent dimensionality (default 24; matches Z3).
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
        cooperative_receiver_lambda_pixel: weight on the residual pixel-MSE
            term in the loss. Default 0.0 (pure cooperative-receiver). 1.0
            recovers pixel-MSE training (the Z3 baseline objective).
        cooperative_receiver_atick_redlich_form: when True, weights match
            Atick-Redlich H(X|W+A+P) form (default True).
        latent_init_std: stddev for per-pair z initialization (default 0.02).
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
    cooperative_receiver_lambda_pixel: float = 0.0
    cooperative_receiver_atick_redlich_form: bool = True
    latent_init_std: float = 0.02

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)


class _Z4Encoder(nn.Module):
    """Small encoder producing per-pair-z initialization.

    Architecturally trivial (small MLP over a global-average-pooled frame
    feature) — the design intent is that the cooperative-receiver LOSS
    dominates score-improvement, not the encoder's capacity.
    """

    def __init__(self, *, input_channels: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # Channel-mixer + global avg pool + small MLP head
        self.stem = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.head_mu = nn.Linear(hidden_dim, latent_dim)
        self.head_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) of shape (B, latent_dim)."""
        if frames.dim() != 4:
            raise ValueError(
                f"encoder expects (B, C, H, W); got shape {tuple(frames.shape)}"
            )
        feats = self.stem(frames)  # (B, hidden, H, W)
        pooled = feats.mean(dim=(2, 3))  # (B, hidden)
        mu = self.head_mu(pooled)
        logvar = self.head_logvar(pooled)
        return mu, logvar

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _Z4Decoder(nn.Module):
    """Small NeRV-style decoder: latent → reconstructed frame pair.

    Uses PixelShuffle(2) upsampling blocks to scale a small grid to
    (output_height, output_width). Output is a (2, 3, H, W) frame pair.
    """

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
            # PixelShuffle(2) needs 4 * out_ch input channels
            blocks.append(nn.Conv2d(in_ch, 4 * out_ch, kernel_size=3, padding=1))
            blocks.append(nn.PixelShuffle(2))
            blocks.append(nn.ReLU(inplace=False))
            in_ch = out_ch
        # Project to 2 frames × 3 channels = 6 channels
        blocks.append(nn.Conv2d(in_ch, 6, kernel_size=3, padding=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (rgb_0, rgb_1) each of shape (B, 3, H, W) in unit-range [0, 1]."""
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"decoder expects (B, latent_dim={self.latent_dim}); got {tuple(z.shape)}"
            )
        batch = z.shape[0]
        # Project to initial grid: (B, embed_dim, h, w)
        flat = self.initial_proj(z)
        grid = flat.view(batch, self.embed_dim, self.initial_grid_h, self.initial_grid_w)
        out = self.blocks(grid)
        # out shape: (B, 6, H', W'); resize to (output_height, output_width)
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


class CooperativeReceiverSubstrate(nn.Module):
    """Z4 cooperative-receiver substrate: byte-identical to Z3, loss-only intervention.

    Forward (training mode):
        1. encoder(frames) → (μ, log_σ²)  (provenance / KL-init only)
        2. z_per_pair = self.latents[pair_indices]   (auto-decoder)
        3. decoder(z_per_pair) → (rgb_0, rgb_1)  ∈ [0, 1] unit range
        4. (μ, log_σ²) returned for forensic provenance (not for IB regularizer
           — the Z4 intervention is loss-only on scorer distortions, not IB).

    Forward (eval mode):
        1. z_per_pair = self.latents[pair_indices]
        2. decoder(z_per_pair) → (rgb_0, rgb_1)
        (Encoder is not used at eval time; provenance only.)
    """

    def __init__(self, cfg: CooperativeReceiverConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = _Z4Encoder(
            input_channels=cfg.encoder_input_channels,
            hidden_dim=cfg.encoder_hidden_dim,
            latent_dim=cfg.latent_dim,
        )
        self.decoder = _Z4Decoder(
            latent_dim=cfg.latent_dim,
            embed_dim=cfg.decoder_embed_dim,
            initial_grid_h=cfg.decoder_initial_grid_h,
            initial_grid_w=cfg.decoder_initial_grid_w,
            decoder_channels=cfg.decoder_channels,
            num_upsample_blocks=cfg.decoder_num_upsample_blocks,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
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
            frames_for_encoder: optional ``(B, C, H, W)`` source frame to
                feed the encoder. Required at training time for forensic
                provenance; None at eval time.

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
    "CooperativeReceiverConfig",
    "CooperativeReceiverSubstrate",
    "EVAL_HW",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
]
