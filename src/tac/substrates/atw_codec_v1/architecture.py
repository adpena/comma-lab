# SPDX-License-Identifier: MIT
"""ATW codec V1 architecture — Atick-Tishby-Wyner cooperative-receiver substrate.

Per the 2026-05-15 grand reunion symposium Composite #1, the ATW codec
extends the A1+Z4 encoder/decoder/per-pair-latent base with TWO new
structural surfaces:

1. **Wyner-Ziv side-info head** — a tiny MLP ``predict(t | scorer_class_prior)``
   that predicts the per-pair latent from the scorer class prior. The
   archive ships ``z_residual = z - z_predicted_from_head`` instead of ``z``
   directly. At inflate time, the decoder reconstructs
   ``z = z_residual + side_info_head(scorer_class_prior_table[pair_index])``.
   The side-info head + class-prior precomputed table are bytewise-tiny
   (~1 KB total) but unlock ~30-50% latent rate savings (Wyner-Ziv 1976).

2. **Tishby IB regularizer hook** — a hyperparameter ``kappa_ib`` that
   weights an additional ``I(T; Y_predicted)`` term in the loss (computed
   via cheap variational posterior approximation). Default 0.0 = no IB
   regularization (Atick-Redlich + WZ pure mode); 0.05-0.1 enables IB.

The encoder + decoder modules are intentionally **architecturally identical
to Z4**: small convolutional encoder + PixelShuffle decoder. The ATW
intervention is structural (WZ side-info head + IB hook) AND in the loss
function, not in the encoder/decoder capacity.

Catalog #124 archive-grammar 8 fields are declared inline in __init__.py at
module level so the AST walker observes them.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" the
``ATWCodec`` is an L1 SCAFFOLD — the operational mechanism (WZ side-info
head consumption at inflate) is wired and verifiable but the ``_full_main``
trainer raises ``NotImplementedError`` per Phase 2 council gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width)."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair)."""

# Archive byte targets — ATW canonical mode predicts a 30-50% reduction vs A1 [prediction]
# in the latent_blob; total archive byte band conservatively narrower than Z4.
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 40_000
"""Predicted minimum: A1 baseline -50% latent + ATW1 magic + tiny WZ head."""

TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 200_000
"""Predicted maximum: A1 baseline + ATW1 magic + tiny WZ head."""


# ---------------------------------------------------------------------------
# Scorer class prior dimensionality.
#
# The Wyner-Ziv side-info head consumes the scorer's class-prior summary
# (a low-dimensional projection of the published SegNet+PoseNet outputs on
# a small per-pair anchor frame) and predicts the per-pair latent.
# Dimensionality 16 is a design-time choice: SegNet has 5 classes; PoseNet
# emits 6 pose deltas; 16 is the union with headroom for class-prior
# embedding. Configurable via ``ATWCodecConfig.scorer_class_prior_dim``.
# ---------------------------------------------------------------------------
DEFAULT_SCORER_CLASS_PRIOR_DIM: int = 16


@dataclass(frozen=True)
class ATWCodecConfig:
    """Static design-time parameters for the ATW codec V1 substrate.

    Encoder / decoder dimensions inherit Z4 defaults verbatim — the ATW
    intervention is structural (WZ side-info head + IB hook) AND in the
    loss function, not in the encoder/decoder capacity.

    Args:
        latent_dim: per-pair latent dimensionality (default 24; matches Z4).
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
        scorer_class_prior_dim: dimensionality of the scorer class prior
            consumed by the WZ side-info head (default 16; design-time
            choice combining 5 SegNet classes + 6 PoseNet pose deltas
            with headroom for class-prior embedding).
        wz_head_hidden_dim: hidden dimensionality of the WZ side-info head
            MLP (default 32; tiny so total head bytes < 1 KB).
        wz_head_enabled: when False, the WZ head is structurally a no-op
            (predicts zeros), and the archive carries the latent verbatim.
            Default True. Setting to False recovers the Z4 baseline.
        ib_kappa_default: default IB regularizer weight (0.0 = no IB).
        wz_lambda_default: default WZ residual weight (1.0 = WZ active).
        pixel_lambda_default: default pixel-MSE residual weight (0.0 = pure ATW).
        atw_atick_redlich_form: when True, β·d_seg + γ·sqrt(d_pose) form
            (default True; matches Atick-Redlich 1990 + contest formula).
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
    scorer_class_prior_dim: int = DEFAULT_SCORER_CLASS_PRIOR_DIM
    wz_head_hidden_dim: int = 32
    wz_head_enabled: bool = True
    ib_kappa_default: float = 0.0
    wz_lambda_default: float = 1.0
    pixel_lambda_default: float = 0.0
    atw_atick_redlich_form: bool = True
    latent_init_std: float = 0.02

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)


class _ATWEncoder(nn.Module):
    """Small encoder producing per-pair-z initialization (Z4-equivalent).

    The encoder is intentionally low-capacity — the design intent is that
    the ATW LOSS + WZ side-info head dominate score-improvement, not the
    encoder's representation capacity.
    """

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
        mu = self.head_mu(pooled)
        logvar = self.head_logvar(pooled)
        return mu, logvar

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _ATWDecoder(nn.Module):
    """Small NeRV-style decoder: latent → reconstructed (rgb_0, rgb_1) frame pair.

    Architecturally identical to Z4 decoder. PixelShuffle(2) upsampling
    blocks scale a small grid to ``(output_height, output_width)``. Output
    is a (2, 3, H, W) frame pair in ``[0, 1]`` unit range.
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


class _WZSideInfoHead(nn.Module):
    """Wyner-Ziv side-info head: predicts per-pair latent from scorer class prior.

    The WZ-encoded archive ships ``z_residual = z - head(scorer_class_prior)``
    rather than ``z`` directly. ~30-50% rate savings come from
    ``z_residual`` having lower entropy than ``z`` when the head's
    prediction quality is good.

    The head is intentionally TINY (a single hidden-layer MLP, ~1 KB total
    parameters) so the side-info head bytes ship in the archive WITHOUT
    swamping the rate budget. The class-prior precomputed table is
    additionally tiny (NUM_PAIRS × scorer_class_prior_dim float16 ≈ 19 KB).

    When ``wz_head_enabled=False`` the head returns zeros (structural no-op);
    the archive carries the latent verbatim and ATW reduces to Z4 baseline.
    """

    def __init__(
        self,
        *,
        scorer_class_prior_dim: int,
        latent_dim: int,
        hidden_dim: int,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.scorer_class_prior_dim = scorer_class_prior_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.enabled = bool(enabled)

        if self.enabled:
            self.fc1 = nn.Linear(scorer_class_prior_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, latent_dim)
        else:
            # Zero-parameter no-op when WZ head disabled.
            self.fc1 = None
            self.fc2 = None

    def forward(self, scorer_class_prior: torch.Tensor) -> torch.Tensor:
        """Predict ``z_predicted`` of shape ``(B, latent_dim)`` from class prior.

        Args:
            scorer_class_prior: ``(B, scorer_class_prior_dim)`` float32 tensor;
                each row is the per-pair scorer class prior summary
                (precomputed at compress-time from the published scorer).

        Returns:
            ``(B, latent_dim)`` predicted latent. Zero tensor when
            ``enabled=False``.
        """
        if scorer_class_prior.dim() != 2:
            raise ValueError(
                f"scorer_class_prior must be 2-D (B, dim); got "
                f"{tuple(scorer_class_prior.shape)}"
            )
        if scorer_class_prior.shape[1] != self.scorer_class_prior_dim:
            raise ValueError(
                f"scorer_class_prior dim {scorer_class_prior.shape[1]} != "
                f"expected {self.scorer_class_prior_dim}"
            )
        batch = scorer_class_prior.shape[0]
        if not self.enabled or self.fc1 is None or self.fc2 is None:
            return torch.zeros(
                batch,
                self.latent_dim,
                device=scorer_class_prior.device,
                dtype=scorer_class_prior.dtype,
            )
        h = torch.relu(self.fc1(scorer_class_prior))
        return self.fc2(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ATWCodec(nn.Module):
    """ATW codec V1 — Atick-Tishby-Wyner cooperative-receiver substrate.

    Forward (training mode):
        1. encoder(frames) → (μ, log_σ²) (provenance / IB-init only)
        2. z_per_pair = self.latents[pair_indices]  (auto-decoder)
        3. z_predicted = wz_side_info_head(scorer_class_prior_per_pair)
        4. z_residual = z_per_pair - z_predicted
        5. decoder(z_per_pair) → (rgb_0, rgb_1) ∈ [0, 1] unit range
        6. (μ, log_σ², z_residual, z_predicted) returned for loss assembly

    Forward (eval / inflate mode):
        1. z_residual = self.latents[pair_indices]   (loaded from archive)
        2. z_predicted = wz_side_info_head(scorer_class_prior_table[pair_indices])
        3. z_per_pair = z_residual + z_predicted
        4. decoder(z_per_pair) → (rgb_0, rgb_1)

    Note: at inflate time, the substrate stores ``z_residual`` in
    ``self.latents`` (loaded from archive) and reconstructs ``z`` via the
    WZ side-info head. The class-prior precomputed table is loaded from
    archive into ``self.scorer_class_prior_table``.

    Catalog #220 OPERATIONAL contract: the WZ side-info head IS the operational
    score-improvement mechanism. ``z_residual + z_predicted_table[pair]`` produces
    a different reconstructed RGB pair than ``z`` alone would (verified by
    no-op detector in ``test_atw_codec_v1_scaffold.py``).
    """

    def __init__(self, cfg: ATWCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = _ATWEncoder(
            input_channels=cfg.encoder_input_channels,
            hidden_dim=cfg.encoder_hidden_dim,
            latent_dim=cfg.latent_dim,
        )
        self.decoder = _ATWDecoder(
            latent_dim=cfg.latent_dim,
            embed_dim=cfg.decoder_embed_dim,
            initial_grid_h=cfg.decoder_initial_grid_h,
            initial_grid_w=cfg.decoder_initial_grid_w,
            decoder_channels=cfg.decoder_channels,
            num_upsample_blocks=cfg.decoder_num_upsample_blocks,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
        )
        self.wz_side_info_head = _WZSideInfoHead(
            scorer_class_prior_dim=cfg.scorer_class_prior_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.wz_head_hidden_dim,
            enabled=cfg.wz_head_enabled,
        )
        # Per-pair learned latents (auto-decoder); shape (num_pairs, latent_dim).
        # At training time these are the FULL z_per_pair; at inflate time
        # these are loaded as z_residual from archive.
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=cfg.latent_init_std)
        )
        # Per-pair scorer class prior precomputed table; loaded from archive
        # at inflate time. At training time, populated externally from the
        # scorer roundtrip of the per-pair anchor frames. Initialized to zeros
        # so a fresh substrate can forward without the table loaded.
        self.register_buffer(
            "scorer_class_prior_table",
            torch.zeros(cfg.num_pairs, cfg.scorer_class_prior_dim),
        )

    def forward(
        self,
        pair_indices: torch.Tensor,
        frames_for_encoder: torch.Tensor | None = None,
        *,
        compute_wz_residual: bool = False,
        decode_mode: str = "full_latent",
    ) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor | None, torch.Tensor | None,
        torch.Tensor | None, torch.Tensor | None,
    ]:
        """Render the per-pair frame pair with optional WZ residual computation.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.
            frames_for_encoder: optional ``(B, C, H, W)`` source frame to
                feed the encoder. Required at training time for forensic
                provenance; None at eval time.
            compute_wz_residual: when True, also return ``(z_residual,
                z_predicted)`` for loss assembly. False at eval time.
            decode_mode: ``"full_latent"`` decodes ``self.latents`` directly
                (training/default). ``"wz_residual"`` treats ``self.latents``
                as archived residuals and decodes
                ``self.latents + WZ_head(class_prior)`` (inflate/eval path).

        Returns:
            ``(rgb_0, rgb_1, mu, logvar, z_residual, z_predicted)``.
            ``mu``/``logvar`` are None when ``frames_for_encoder is None``.
            ``z_residual``/``z_predicted`` are None when
            ``compute_wz_residual=False``.
        """
        if decode_mode not in {"full_latent", "wz_residual"}:
            raise ValueError(
                f"decode_mode must be 'full_latent' or 'wz_residual'; got {decode_mode!r}"
            )
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.numel() == 0:
            raise ValueError("pair_indices must be non-empty")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs}); "
                f"got [{pair_indices.min().item()}, {pair_indices.max().item()}]"
            )

        z_stored = self.latents[pair_indices]  # (B, latent_dim)
        class_prior = self.scorer_class_prior_table[pair_indices]
        z_predicted_for_decode = self.wz_side_info_head(class_prior)
        z = (
            z_stored + z_predicted_for_decode
            if decode_mode == "wz_residual"
            else z_stored
        )
        rgb_0, rgb_1 = self.decoder(z)

        if frames_for_encoder is not None:
            mu, logvar = self.encoder(frames_for_encoder)
        else:
            mu, logvar = None, None

        z_residual: torch.Tensor | None = None
        z_predicted: torch.Tensor | None = None
        if compute_wz_residual:
            z_predicted = z_predicted_for_decode
            z_residual = z_stored if decode_mode == "wz_residual" else z_stored - z_predicted

        return rgb_0, rgb_1, mu, logvar, z_residual, z_predicted

    def reconstruct_from_wz_residual(
        self,
        pair_indices: torch.Tensor,
        z_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inflate-time reconstruction: ``z = z_residual + side_info_head(class_prior)``.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.
            z_residual: ``(B, latent_dim)`` archived residual from ATW1 archive.

        Returns:
            ``(rgb_0, rgb_1)`` reconstructed frame pair.
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if z_residual.dim() != 2 or z_residual.shape[1] != self.cfg.latent_dim:
            raise ValueError(
                f"z_residual must be (B, latent_dim={self.cfg.latent_dim}); "
                f"got {tuple(z_residual.shape)}"
            )
        if pair_indices.shape[0] != z_residual.shape[0]:
            raise ValueError(
                f"pair_indices and z_residual batch sizes mismatch: "
                f"{pair_indices.shape[0]} vs {z_residual.shape[0]}"
            )
        class_prior = self.scorer_class_prior_table[pair_indices]
        z_predicted = self.wz_side_info_head(class_prior)
        z_full = z_residual + z_predicted
        return self.decoder(z_full)

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters_breakdown(self) -> dict[str, int]:
        """Encoder / decoder / WZ-head / latent param counts."""
        return {
            "encoder": self.encoder.num_parameters(),
            "decoder": self.decoder.num_parameters(),
            "wz_side_info_head": self.wz_side_info_head.num_parameters(),
            "latents": self.latents.numel(),
            "total": self.num_parameters(),
        }


__all__ = [
    "DEFAULT_SCORER_CLASS_PRIOR_DIM",
    "EVAL_HW",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "ATWCodec",
    "ATWCodecConfig",
]
