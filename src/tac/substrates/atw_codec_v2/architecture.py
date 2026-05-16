# SPDX-License-Identifier: MIT
"""ATW codec V2 architecture — Atick-Tishby-Wyner full-stack cooperative-receiver substrate.

Per the 2026-05-16 V2 design memo §4 + §7. V2 extends the V1 encoder/decoder
base with TWO new structural surfaces beyond V1's WZ side-info head:

1. **G1 scorer-class distill head** — small MLP
   ``g(decoded_latent_per_pixel) -> 5-way SegNet softmax``. Replaces the
   Ballé-style 50KB hyperprior with a 1KB distilled head. At compress
   time the head is trained against SegNet's argmax on the rendered output
   (NOT GT — matches inflate-time signal). At inflate time
   ``distill_head(decoded_latent)`` provides hyperprior gating for the
   range-coder WITHOUT loading SegNet weights.

2. **B3 scorer-conditional CDF table** — empirical histogram of
   ``(z_residual_quantized, class_index)`` pairs precomputed at compress;
   shipped ~2KB side-info. At inflate ``cdf_table[class_index, :]`` selects
   the conditional CDF for range-decoder. Reduces per-latent rate via
   class-conditional entropy coding.

The encoder + decoder + WZ side-info head modules are architecturally
inherited from V1 verbatim. The ATW V2 intervention is structural
(G1 + B3) AND in the score-aware loss (canonical Atick-Redlich primitive
routing per Wunderkind E1 / Catalog #164).

Two variants per design memo §4:

* **Variant A** (three-knob κ_IB / λ_WZ / λ_pixel) preserves V1 form
* **Variant B** (single-knob WZ-only) is the UNIQUE-AND-COMPLETE default

Catalog #124 archive-grammar 8 fields are declared inline in __init__.py at
module level so the AST walker observes them.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
ATW V2 ships with ``_full_main`` IMPLEMENTED but the recipe carries
``research_only=true`` until the D4 probe gates pass. The operational
mechanism (G1 + B3 + WZ side-info head consumption at inflate) is wired
and verifiable; the byte-mutation smoke per Catalog #220 + #272 confirms
non-trivial distinguishing-feature contribution at inflate time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn

EVAL_HW: tuple[int, int] = (384, 512)
"""Contest scorer-resolution (height, width)."""

NUM_PAIRS: int = 600
"""Contest pair count (1200 frames / 2 frames per pair)."""

# Archive byte targets per design memo §7: predicted 80-120KB total vs A1's 179KB
# (40-55% reduction; consistent with V1 design memo's 30-50% latent savings
# estimate per CC-3 reactivation criterion).
TOTAL_ARCHIVE_TARGET_BYTES_MIN: int = 60_000
"""Predicted minimum (Variant B + tight G1/B3): WZ residual + small heads."""

TOTAL_ARCHIVE_TARGET_BYTES_MAX: int = 180_000
"""Predicted maximum (Variant A + larger encoder/decoder): conservative upper."""

DEFAULT_SCORER_CLASS_PRIOR_DIM: int = 16
"""Wyner-Ziv side-info head input dimensionality.

SegNet has 5 classes; PoseNet emits 6 pose deltas; 16 is the union with
headroom for class-prior embedding. Configurable via
``ATWv2CodecConfig.scorer_class_prior_dim``.
"""

NUM_SEGNET_CLASSES: int = 5
"""Upstream SegNet output classes (per upstream/modules.py)."""

CDF_TABLE_NUM_SYMBOLS: int = 256
"""Per-class CDF table symbol count for B3 range coder (int8 latent symbols)."""


class ATWv2Variant(StrEnum):
    """ATW V2 variant selector — Variant A (three-knob) or B (single-knob WZ-only)."""

    A_THREE_KNOB = "A"
    """V1-inherited three-knob form preserving probe-disambiguator regime sweep."""

    B_WZ_ONLY = "B"
    """UNIQUE-AND-COMPLETE single-knob WZ-only form; DEFAULT per design memo §4.3."""


@dataclass(frozen=True)
class ATWv2CodecConfig:
    """Static design-time parameters for the ATW codec V2 substrate.

    Encoder / decoder dimensions inherit V1 (which inherited Z4) defaults
    verbatim — the V2 intervention is structural (G1 + B3 + closed-form WZ)
    AND in the loss function routing (canonical Atick-Redlich primitive),
    not in the encoder/decoder capacity.

    Args:
        variant: ATWv2Variant.A_THREE_KNOB or B_WZ_ONLY. Default B per §4.3.
        latent_dim: per-pair latent dimensionality (default 24; matches V1/Z4).
        encoder_input_channels: encoder input channels (default 3 = RGB).
        encoder_hidden_dim: encoder hidden state dimension (default 64).
        decoder_embed_dim: decoder initial-grid channel count (default 32).
        decoder_initial_grid_h: decoder initial grid height (default 3).
        decoder_initial_grid_w: decoder initial grid width (default 4).
        decoder_channels: per-block output channels.
        decoder_num_upsample_blocks: number of PixelShuffle(2) blocks (default 6).
        num_pairs: contest pair count (default 600).
        output_height, output_width: scorer-resolution (default 384x512).
        scorer_class_prior_dim: WZ side-info head input dim (default 16).
        wz_head_hidden_dim: WZ MLP hidden dim (default 32; ~1KB total).
        wz_head_enabled: when False the head returns zeros (debug toggle only).
        g1_distill_hidden_dim: G1 distill head hidden dim (default 32; ~1KB).
        g1_distill_enabled: when False G1 head not constructed (Variant A regime
            sweep corner OR debug; default True for full V2 ship).
        b3_cdf_enabled: when False B3 CDF table is uniform (Variant A regime
            sweep corner OR debug; default True for full V2 ship).
        ib_kappa_default: Tishby IB regularizer (Variant A only; default 0.0).
        wz_lambda_default: Wyner-Ziv residual weight (default 1.0).
        pixel_lambda_default: Z3 pixel-MSE residual weight (Variant A only;
            default 0.0).
        latent_init_std: stddev for per-pair z initialization (default 0.02).
    """

    variant: ATWv2Variant = ATWv2Variant.B_WZ_ONLY
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
    g1_distill_hidden_dim: int = 32
    g1_distill_enabled: bool = True
    b3_cdf_enabled: bool = True
    ib_kappa_default: float = 0.0
    wz_lambda_default: float = 1.0
    pixel_lambda_default: float = 0.0
    latent_init_std: float = 0.02

    @property
    def output_hw(self) -> tuple[int, int]:
        return (self.output_height, self.output_width)


class _ATWv2Encoder(nn.Module):
    """Small encoder producing per-pair-z initialization (V1-equivalent).

    Encoder capacity is intentionally low — the design intent is that the
    ATW V2 LOSS + WZ + G1 + B3 dominate score-improvement, not encoder
    representation capacity.
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


class _ATWv2Decoder(nn.Module):
    """Small NeRV-style decoder: latent -> reconstructed (rgb_0, rgb_1) pair.

    Architecturally inherited from V1. PixelShuffle(2) upsampling blocks
    scale a small grid to ``(output_height, output_width)``. Output is a
    ``(B, 6, H, W)`` tensor decomposed into a ``(2, 3, H, W)`` frame pair
    in ``[0, 1]`` unit range.
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

    Inherited verbatim from V1 architecture. The WZ-encoded archive ships
    ``z_residual = z - head(scorer_class_prior)`` rather than ``z`` directly.
    The head is intentionally TINY (single hidden-layer MLP, ~1 KB total
    parameters) so the side-info head bytes ship in the archive WITHOUT
    swamping the rate budget.
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
            self.fc1 = None
            self.fc2 = None

    def forward(self, scorer_class_prior: torch.Tensor) -> torch.Tensor:
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


class _G1DistillHead(nn.Module):
    """G1 scorer-class distill head — replaces Ballé-style hyperprior.

    Small MLP ``g(per_pair_latent) -> 5-way SegNet class softmax``.

    Trained at compress time against SegNet's argmax on the rendered output
    (NOT GT — matches what the decoder sees at inflate). At inflate time the
    head's softmax output provides the conditioning index for B3's CDF table
    selection, replacing the need to load SegNet weights at inflate.

    The head is intentionally TINY (~256 params; ~1KB after fp16). Per the
    Wunderkind G1 substitution candidate
    (``feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md``).
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int = NUM_SEGNET_CLASSES,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.enabled = bool(enabled)
        if self.enabled:
            self.fc1 = nn.Linear(latent_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc1 = None
            self.fc2 = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict per-pair scorer class logits ``(B, num_classes)``.

        When ``enabled=False`` returns a uniform-logits tensor (Variant A
        regime-sweep G1-disabled corner).
        """
        if z.dim() != 2:
            raise ValueError(
                f"G1 distill head expects (B, latent_dim); got {tuple(z.shape)}"
            )
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"G1 latent_dim mismatch: got {z.shape[1]}, expected {self.latent_dim}"
            )
        batch = z.shape[0]
        if not self.enabled or self.fc1 is None or self.fc2 is None:
            return torch.zeros(
                batch, self.num_classes, device=z.device, dtype=z.dtype
            )
        h = torch.relu(self.fc1(z))
        return self.fc2(h)

    def predict_class(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``(B,)`` long class index = argmax(softmax(forward(z)))."""
        logits = self.forward(z)
        return torch.argmax(logits, dim=1).to(torch.long)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ATWv2Codec(nn.Module):
    """ATW codec V2 — Atick-Tishby-Wyner full-stack cooperative-receiver substrate.

    Forward (training mode):
        1. encoder(frames) -> (mu, log_sigma^2) (provenance / IB-init only)
        2. z_per_pair = self.latents[pair_indices]  (auto-decoder)
        3. z_predicted = wz_side_info_head(scorer_class_prior_per_pair)
        4. z_residual = z_per_pair - z_predicted
        5. distill_class_logits = g1_distill_head(z_per_pair)   (G1 supervision)
        6. decoder(z_per_pair) -> (rgb_0, rgb_1) in [0, 1] unit range
        7. (rgb_0, rgb_1, mu, logvar, z_residual, z_predicted, distill_class_logits)

    Forward (eval / inflate mode):
        1. z_residual = self.latents[pair_indices]  (loaded from archive)
        2. z_predicted = wz_side_info_head(scorer_class_prior_table[pair_indices])
        3. z_per_pair = z_residual + z_predicted   <-- WZ reconstruction
        4. decoder(z_per_pair) -> (rgb_0, rgb_1)

    Catalog #220 OPERATIONAL contract: the WZ side-info head + G1 distill
    head + B3 scorer-conditional CDF table ARE the operational
    score-improvement mechanism. ``z_residual + wz_head(class_prior[i])``
    produces a different reconstructed RGB pair than ``z`` alone would
    (verified by Catalog #272 byte-mutation smoke).
    """

    def __init__(self, cfg: ATWv2CodecConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = _ATWv2Encoder(
            input_channels=cfg.encoder_input_channels,
            hidden_dim=cfg.encoder_hidden_dim,
            latent_dim=cfg.latent_dim,
        )
        self.decoder = _ATWv2Decoder(
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
        self.g1_distill_head = _G1DistillHead(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.g1_distill_hidden_dim,
            num_classes=NUM_SEGNET_CLASSES,
            enabled=cfg.g1_distill_enabled,
        )

        # Per-pair learned latents (auto-decoder); shape (num_pairs, latent_dim).
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=cfg.latent_init_std)
        )
        # Per-pair scorer class prior precomputed table; loaded from archive at
        # inflate time. Initialized to zeros so a fresh substrate forwards.
        self.register_buffer(
            "scorer_class_prior_table",
            torch.zeros(cfg.num_pairs, cfg.scorer_class_prior_dim),
        )
        # B3 scorer-conditional CDF table: (num_classes, num_symbols) fp16
        # at archive ship-time; fp32 in-memory for arithmetic stability.
        # Initialized to uniform (1.0/num_symbols) so a fresh substrate
        # produces a non-degenerate CDF.
        self.register_buffer(
            "cdf_table",
            torch.full(
                (NUM_SEGNET_CLASSES, CDF_TABLE_NUM_SYMBOLS),
                1.0 / float(CDF_TABLE_NUM_SYMBOLS),
            ),
        )

    def forward(
        self,
        pair_indices: torch.Tensor,
        frames_for_encoder: torch.Tensor | None = None,
        *,
        compute_wz_residual: bool = False,
        compute_g1_logits: bool = False,
        decode_mode: str = "full_latent",
    ) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor | None, torch.Tensor | None,
        torch.Tensor | None, torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Render per-pair frame pair with optional WZ + G1 outputs.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.
            frames_for_encoder: optional ``(B, C, H, W)`` source frame for
                encoder. Required at training for forensic provenance; None
                at eval.
            compute_wz_residual: when True, also return ``(z_residual,
                z_predicted)`` for loss assembly.
            compute_g1_logits: when True, also return G1 distill_class_logits.
            decode_mode: ``"full_latent"`` decodes ``self.latents`` directly
                (training/default). ``"wz_residual"`` treats ``self.latents``
                as archived residuals and decodes
                ``self.latents + WZ_head(class_prior)`` (inflate path).

        Returns:
            ``(rgb_0, rgb_1, mu, logvar, z_residual, z_predicted,
              distill_class_logits)``.
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

        distill_class_logits: torch.Tensor | None = None
        if compute_g1_logits and self.cfg.g1_distill_enabled:
            distill_class_logits = self.g1_distill_head(z)

        return rgb_0, rgb_1, mu, logvar, z_residual, z_predicted, distill_class_logits

    def reconstruct_from_wz_residual(
        self,
        pair_indices: torch.Tensor,
        z_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inflate-time reconstruction: ``z = z_residual + WZ_head(class_prior)``.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.
            z_residual: ``(B, latent_dim)`` archived residual from ATW2 archive.

        Returns:
            ``(rgb_0, rgb_1)`` reconstructed frame pair in [0, 1] unit range.
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
        """Encoder / decoder / WZ-head / G1-head / latent param counts."""
        return {
            "encoder": self.encoder.num_parameters(),
            "decoder": self.decoder.num_parameters(),
            "wz_side_info_head": self.wz_side_info_head.num_parameters(),
            "g1_distill_head": self.g1_distill_head.num_parameters(),
            "latents": self.latents.numel(),
            "total": self.num_parameters(),
        }


__all__ = [
    "CDF_TABLE_NUM_SYMBOLS",
    "DEFAULT_SCORER_CLASS_PRIOR_DIM",
    "EVAL_HW",
    "NUM_PAIRS",
    "NUM_SEGNET_CLASSES",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "ATWv2Codec",
    "ATWv2CodecConfig",
    "ATWv2Variant",
]
