"""vq_vae architecture — discrete codebook + tiny encoder/decoder.

L0 SKETCH scaffold per operator approval 2026-05-12. The VQ-VAE (van den Oord
2017) replaces continuous latents with a **discrete codebook** of K embeddings.
The encoder produces a per-cell spatial grid (H/8 x W/8 x D); each cell is
quantized via nearest-neighbor lookup against the codebook; the decoder reads
the index grid.

Architecture (council-sketch 2026-05-12; not yet empirical-anchored):

    Per-pair learned per-cell features e_{i,j} in R^D  (one frame per side)
       |
       v
    Encoder: tiny Conv-GELU stack (~10K params) maps frame-conditioned features
    to spatial grid (H/8 x W/8 x D) where D = codebook embedding dim
       |
       v
    Quantizer: nearest-neighbor lookup vs codebook (K x D); index out
       |
       v
    Straight-through estimator: gradient flows through codebook lookup
       |
       v
    Decoder: tiny PixelShuffle-style upsample (~10K params) -> RGB
       |
       v
    Per-pair output: (rgb_0, rgb_1)

Council notes:
- Total param target: ~180K (~10K encoder + ~10K decoder + ~4K codebook K=512,D=8
  + ~156K per-pair indices stored as part of the archive, not as model params)
- Codebook adaptation: STE-gradient only (Bengio 2013) at L0 SKETCH; van den Oord
  persistent N_c/m_c EMA buffer form (decay=0.99) is DEFERRED-pending-substrate-
  engineering. The ``codebook_ema_decay=0.99`` config field is reserved for the
  future EMA-buffer implementation (R4 finding Z-8.1, 2026-05-13). Promotion
  past L0 SKETCH requires either implementing the persistent N_c/m_c buffers
  (register_buffer("ema_cluster_size", torch.zeros(K)) +
  register_buffer("ema_w", torch.zeros(K, D)) updated BEFORE quantize per
  training step) OR explicit operator decision to ship the STE-gradient-only
  variant. Per CLAUDE.md "Comment-only contracts are FORBIDDEN" — see van den
  Oord council seat verdict in
  feedback_review_zeta_r4_LANDED_20260513.md Finding Z-8.1.
- Encoder/decoder weight EMA 0.997 per CLAUDE.md "EMA — non-negotiable".

CLAUDE.md compliance:
- No silent device defaults (caller passes device)
- No scorer loading inside this module
- No /tmp paths
- Reviewable in 30 seconds per L12
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2  # 600 pairs


@dataclass(frozen=True)
class VqVaeConfig:
    """Static design-time parameters for vq_vae (L0 SKETCH)."""

    codebook_size: int = 512
    """K: number of codebook entries. Rate = log2(K) bits per cell."""

    embedding_dim: int = 8
    """D: per-entry embedding dimension. Codebook params = K*D."""

    encoder_hidden: int = 24
    """Hidden channels of the tiny encoder."""

    decoder_hidden: int = 24
    """Hidden channels of the tiny decoder."""

    grid_downsample: int = 8
    """Spatial-grid downsample factor (H/8 x W/8 grid by default)."""

    num_pairs: int = _PAIRS
    """Number of latent rows (600 for 1200-frame contest video)."""

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W

    codebook_ema_decay: float = 0.99
    """Reserved for van den Oord persistent N_c/m_c codebook EMA decay (NOT
    weight EMA 0.997). DEFERRED-pending-substrate-engineering at L0 SKETCH —
    the architecture currently updates the codebook via STE gradient only (no
    register_buffer for ema_cluster_size / ema_w). Per R4 finding Z-8.1
    (2026-05-13): this field is honest config-reservation for the future EMA
    buffer implementation; current trainer does NOT consume it. See
    feedback_review_zeta_r4_LANDED_20260513.md Finding Z-8.1 for the van den
    Oord + Tao + Filler verdict."""

    commitment_cost: float = 0.25
    """Commitment loss weight (encoder pulled towards chosen codebook entry)."""

    def __post_init__(self) -> None:
        if self.codebook_size <= 0 or self.codebook_size > 65536:
            raise ValueError("codebook_size must be in (0, 65536]")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.grid_downsample <= 0:
            raise ValueError("grid_downsample must be positive")
        if self.grid_downsample & (self.grid_downsample - 1):
            raise ValueError("grid_downsample must be a power of two")
        if self.output_height % self.grid_downsample or self.output_width % self.grid_downsample:
            raise ValueError("output dimensions must be divisible by grid_downsample")
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if not (0.0 <= self.codebook_ema_decay < 1.0):
            raise ValueError("codebook_ema_decay must be in [0, 1)")
        if self.commitment_cost < 0.0:
            raise ValueError("commitment_cost must be non-negative")


class _Encoder(nn.Module):
    """Tiny encoder: per-frame conv stack -> spatial grid of embeddings."""

    def __init__(self, in_ch: int, hidden: int, out_dim: int, downsample: int) -> None:
        super().__init__()
        # Stride-2 stack: downsample=8 means 3 stride-2 convs
        layers: list[nn.Module] = [nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1)]
        cur = hidden
        n_stride2 = downsample.bit_length() - 1
        for _ in range(n_stride2):
            layers.append(nn.Conv2d(cur, hidden, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            cur = hidden
        layers.append(nn.Conv2d(cur, out_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    """Tiny decoder: index grid -> per-pixel RGB via bilinear upsample + MLP."""

    def __init__(self, in_dim: int, hidden: int, output_h: int, output_w: int) -> None:
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: (B, D, H/8, W/8) — upsample to (B, D, H, W) then synth
        up = F.interpolate(
            grid,
            size=(self.output_h, self.output_w),
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(self.net(up))


class VqVaeSubstrate(nn.Module):
    """VQ-VAE substrate: per-pair feature grid + codebook + tiny decoder.

    Forward signature mirrors sane_hnerv for trainer interop:
        forward(pair_indices) -> (rgb_0, rgb_1), each (B, 3, H, W).

    The codebook is a trainable ``nn.Parameter`` updated via the
    straight-through estimator (Bengio 2013) at L0 SKETCH. The van den Oord
    persistent N_c/m_c EMA buffer form (decay=0.99) is DEFERRED-pending-
    substrate-engineering per R4 finding Z-8.1 (2026-05-13): the
    ``cfg.codebook_ema_decay`` config field is reserved for the future buffer
    implementation but the current architecture does NOT register the
    ``ema_cluster_size`` / ``ema_w`` buffers nor consume the decay during
    training. Per CLAUDE.md "Comment-only contracts are FORBIDDEN", this
    docstring is the authoritative description.
    """

    def __init__(self, cfg: VqVaeConfig) -> None:
        super().__init__()
        self.cfg = cfg

        h_grid = cfg.output_height // cfg.grid_downsample
        w_grid = cfg.output_width // cfg.grid_downsample

        # Per-pair feature grid (the trainable analog of cool_chic's continuous latents)
        # Shape: (num_pairs, 2, D, h_grid, w_grid) — 2 = (frame_0, frame_1)
        self.per_pair_features = nn.Parameter(
            torch.empty(cfg.num_pairs, 2, cfg.embedding_dim, h_grid, w_grid).normal_(std=0.02)
        )

        # Codebook: K x D
        self.codebook = nn.Parameter(
            torch.empty(cfg.codebook_size, cfg.embedding_dim).uniform_(-1.0 / cfg.codebook_size, 1.0 / cfg.codebook_size)
        )

        # Encoder: a tiny refinement net applied to the per-pair feature before quantization
        # In a true VQ-VAE the encoder maps a frame -> grid; here we apply it as a
        # per-cell refinement on top of the per-pair learned feature for simplicity.
        # NOTE: a future production VQ-VAE would feed raw RGB through _Encoder; this
        # L0 SKETCH keeps the param count low by treating per_pair_features as the
        # encoder output directly.
        self.encoder_refine = _Encoder(
            in_ch=cfg.embedding_dim,
            hidden=cfg.encoder_hidden,
            out_dim=cfg.embedding_dim,
            downsample=1,  # no further downsample; the per-pair features already at grid resolution
        )

        # Decoder
        self.decoder = _Decoder(
            in_dim=cfg.embedding_dim,
            hidden=cfg.decoder_hidden,
            output_h=cfg.output_height,
            output_w=cfg.output_width,
        )

    def _quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nearest-neighbor codebook lookup with straight-through gradient.

        Args:
            z_e: ``(B, D, H, W)`` encoder output.

        Returns:
            (z_q, indices, commitment_loss):
                z_q: ``(B, D, H, W)`` quantized output with STE gradient.
                indices: ``(B, H, W)`` long-tensor of selected codebook indices.
                commitment_loss: scalar — encoder pulled towards chosen entry.
        """
        b, d, h, w = z_e.shape
        # Flatten spatial: (B*H*W, D)
        flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, d)
        # Compute squared distances to each codebook entry
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        a_sq = (flat * flat).sum(dim=1, keepdim=True)  # (N, 1)
        b_sq = (self.codebook * self.codebook).sum(dim=1).unsqueeze(0)  # (1, K)
        cross = flat @ self.codebook.t()  # (N, K)
        dists = a_sq + b_sq - 2.0 * cross
        indices = dists.argmin(dim=1)  # (N,)
        # Lookup: gather codebook by indices
        z_q_flat = self.codebook[indices]  # (N, D)
        # Reshape back: (B, H, W, D) -> (B, D, H, W)
        z_q = z_q_flat.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        # Straight-through estimator: forward returns z_q, backward bypasses to z_e
        z_q_ste = z_e + (z_q - z_e).detach()
        # Commitment loss: encoder pulled towards quantized
        commitment = ((z_e - z_q.detach()) ** 2).mean()
        return z_q_ste, indices.view(b, h, w), commitment

    def forward(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Render frame-pairs at the given pair indices.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.

        Returns:
            ``(rgb_0, rgb_1)``, each ``(B, 3, H, W)`` in ``[0, 1]``.
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(f"pair_indices out of range [0, {self.cfg.num_pairs})")

        # (B, 2, D, h_grid, w_grid)
        per_pair = self.per_pair_features[pair_indices]
        feat_0 = per_pair[:, 0]  # (B, D, h_grid, w_grid)
        feat_1 = per_pair[:, 1]

        # Encoder refine (acts on (B, D, h, w))
        z_e_0 = self.encoder_refine(feat_0)
        z_e_1 = self.encoder_refine(feat_1)

        # Quantize each frame independently
        z_q_0, _idx_0, _commit_0 = self._quantize(z_e_0)
        z_q_1, _idx_1, _commit_1 = self._quantize(z_e_1)

        rgb_0 = self.decoder(z_q_0)
        rgb_1 = self.decoder(z_q_1)
        return rgb_0, rgb_1

    def compute_commitment_loss(self, pair_indices: torch.Tensor) -> torch.Tensor:
        """Compute the VQ-VAE commitment loss for the given pair indices.

        Trainer calls this and adds ``cfg.commitment_cost * commitment_loss``
        to the score-aware Lagrangian. Returns a scalar tensor.
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        per_pair = self.per_pair_features[pair_indices]
        feat_0 = per_pair[:, 0]
        feat_1 = per_pair[:, 1]
        z_e_0 = self.encoder_refine(feat_0)
        z_e_1 = self.encoder_refine(feat_1)
        _zq0, _idx0, c0 = self._quantize(z_e_0)
        _zq1, _idx1, c1 = self._quantize(z_e_1)
        return 0.5 * (c0 + c1)

    def encode_indices_for_archive(self) -> torch.Tensor:
        """Compute all codebook indices for all pairs (offline; trainer calls
        this at export time to feed pack_archive).

        Returns:
            ``(num_pairs, 2, h_grid, w_grid)`` int64 tensor of codebook indices.
        """
        out = []
        with torch.no_grad():
            for i in range(self.cfg.num_pairs):
                feat_0 = self.encoder_refine(self.per_pair_features[i : i + 1, 0])
                feat_1 = self.encoder_refine(self.per_pair_features[i : i + 1, 1])
                _zq0, idx0, _c0 = self._quantize(feat_0)
                _zq1, idx1, _c1 = self._quantize(feat_1)
                out.append(torch.stack([idx0[0], idx1[0]], dim=0))  # (2, h_grid, w_grid)
        return torch.stack(out, dim=0).to(torch.int64)  # (num_pairs, 2, h, w)

    def runtime_state_dict_for_archive(self) -> dict[str, torch.Tensor]:
        """Return only inflate-time tensors for the VQV1 archive.

        The encoder and per-pair feature grid are training-only: once the
        codebook indices have been exported, inflate reconstructs quantized
        grids from ``codebook[indices]`` and then runs the decoder. Carrying
        encoder or per-pair-feature bytes in the archive is therefore a pure
        rate leak.
        """

        return {
            name: tensor.detach().clone()
            for name, tensor in self.state_dict().items()
            if name == "codebook" or name.startswith("decoder.")
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
