"""grayscale_lut architecture — analog grayscale stream + FiLM-conditioned RGB decoder.

L0 SKETCH scaffold per operator approval 2026-05-12. The grayscale-LUT family
(Selfcomp / szabolcs-cs's PR #56 paradigm) factors per-pair information into:

    1. A per-pair 1-channel grayscale field at low spatial resolution
       (H/grayscale_downsample x W/grayscale_downsample), quantized to int8.
       This is the analog-signal rate term (highly compressible).
    2. A tiny FiLM-conditioned RGB decoder (~94K params per Selfcomp anchor)
       that maps (grayscale + per-pair embedding) -> RGB.

Architecture (council-sketch 2026-05-12; not yet empirical-anchored):

    Per-pair grayscale field G_t in R^(num_pairs, H/D, W/D)  (D=grayscale_downsample)
       |
       v
    Bilinear upsample G_t to (H, W) -> g_full in R^(B, 1, H, W)
       |
       v
    Per-pair embedding e_t in R^(num_pairs, embedding_dim)
       (FiLM generator: e_t -> (gamma_t, beta_t) for each decoder layer)
       |
       v
    Decoder: Conv-stack with FiLM modulation:
        h = conv(g_full)
        for each block:
            h = act(film(conv(h), gamma_t, beta_t))
        rgb_0_logits, rgb_1_logits = head_0(h), head_1(h)
       |
       v
    Per-pair output: (sigmoid(rgb_0_logits), sigmoid(rgb_1_logits))

Council notes:
- Total param target: ~94K decoder + per-pair grayscale fields (the dominant
  rate term, NOT counted as model params)
- Per-pair embedding dim 16; FiLM generator ~5K params
- EMA decay 0.997 per CLAUDE.md "EMA — non-negotiable" (applied externally)

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
class GrayscaleLutConfig:
    """Static design-time parameters for grayscale_lut (L0 SKETCH)."""

    grayscale_downsample: int = 4
    """Spatial downsample factor for the analog grayscale stream (H/4 x W/4)."""

    decoder_hidden: int = 48
    """Hidden channels of the colorization decoder."""

    decoder_blocks: int = 4
    """Number of FiLM-conditioned decoder blocks."""

    embedding_dim: int = 16
    """Per-pair embedding dimensionality for FiLM conditioning."""

    num_pairs: int = _PAIRS
    """Number of latent rows (600 for 1200-frame contest video)."""

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W

    def __post_init__(self) -> None:
        if self.grayscale_downsample <= 0:
            raise ValueError("grayscale_downsample must be positive")
        if self.decoder_hidden <= 0:
            raise ValueError("decoder_hidden must be positive")
        if self.decoder_blocks <= 0:
            raise ValueError("decoder_blocks must be positive")
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if self.output_height % self.grayscale_downsample or self.output_width % self.grayscale_downsample:
            raise ValueError("output dimensions must be divisible by grayscale_downsample")


class _FiLMBlock(nn.Module):
    """One Conv + GELU block with FiLM (gamma, beta) modulation."""

    def __init__(self, hidden: int, embedding_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.film_gen = nn.Linear(embedding_dim, 2 * hidden)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        gamma_beta = self.film_gen(emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # Broadcast: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + gamma) + beta
        return self.act(h)


class GrayscaleLutSubstrate(nn.Module):
    """Grayscale-LUT substrate: analog grayscale stream + FiLM-conditioned RGB decoder.

    Forward signature mirrors sane_hnerv for trainer interop:
        forward(pair_indices) -> (rgb_0, rgb_1), each (B, 3, H, W).

    The grayscale stream is stored as a learnable parameter at train time
    (fp32) and quantized to int8 at archive export time.
    """

    def __init__(self, cfg: GrayscaleLutConfig) -> None:
        super().__init__()
        self.cfg = cfg

        h_g = cfg.output_height // cfg.grayscale_downsample
        w_g = cfg.output_width // cfg.grayscale_downsample

        # Per-pair grayscale field (the analog-signal rate term)
        # In an actual training run this is initialized from the GT grayscale
        # of the contest video; here we initialize to mid-gray.
        self.grayscale = nn.Parameter(
            torch.full((cfg.num_pairs, 1, h_g, w_g), 0.5, dtype=torch.float32)
        )

        # Per-pair embedding for FiLM conditioning
        self.pair_embedding = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.embedding_dim).normal_(std=0.02)
        )

        # Decoder stem
        self.stem = nn.Conv2d(1, cfg.decoder_hidden, kernel_size=3, padding=1)

        # Decoder FiLM blocks
        self.blocks = nn.ModuleList(
            [_FiLMBlock(cfg.decoder_hidden, cfg.embedding_dim) for _ in range(cfg.decoder_blocks)]
        )

        # Two RGB heads (frame 0 and frame 1)
        self.head_rgb_0 = nn.Conv2d(cfg.decoder_hidden, 3, kernel_size=3, padding=1)
        self.head_rgb_1 = nn.Conv2d(cfg.decoder_hidden, 3, kernel_size=3, padding=1)

    def _upsample_grayscale(self, gs: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            gs,
            size=(self.cfg.output_height, self.cfg.output_width),
            mode="bilinear",
            align_corners=False,
        )

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

        gs = self.grayscale[pair_indices]  # (B, 1, H/D, W/D)
        emb = self.pair_embedding[pair_indices]  # (B, embedding_dim)

        gs_up = self._upsample_grayscale(gs)  # (B, 1, H, W)
        h = self.stem(gs_up)  # (B, hidden, H, W)
        for blk in self.blocks:
            h = blk(h, emb)

        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    def quantize_grayscale_for_archive(self) -> torch.Tensor:
        """Quantize grayscale parameter to uint8 for archive export.

        Returns:
            ``(num_pairs, 1, H/D, W/D)`` uint8 tensor of quantized grayscale.
        """
        # Clamp to [0, 1] and scale to [0, 255]
        with torch.no_grad():
            q = (self.grayscale.detach().clamp(0.0, 1.0) * 255.0).round()
        return q.to(torch.uint8)

    def runtime_state_dict_for_archive(self) -> dict[str, torch.Tensor]:
        """Return only inflate-time tensors for the GLV1 archive.

        The grayscale field is exported separately (as uint8) into its own
        archive section. The pair_embedding is required at inflate time to
        re-instantiate the FiLM conditioning. The decoder weights (stem +
        blocks + heads) are required at inflate time.
        """
        return {
            name: tensor.detach().clone()
            for name, tensor in self.state_dict().items()
            if name != "grayscale"
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
