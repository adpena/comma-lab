"""cool_chic architecture — per-frame latent grids + shared synthesis MLP + AR prior.

L0 SKETCH scaffold per operator approval 2026-05-12. The Cool-Chic family
(Ladune et al., 2023) flips the NeRV recipe: most parameters live in **per-frame
latents**, the renderer is a TINY shared synthesis MLP, and rate is paid via an
AR (autoregressive) density estimate over the latents.

Architecture (council-sketch 2026-05-12; not yet empirical-anchored):

    Per-pair latent grid (multi-scale spatial):
        L_coarse: R^(num_pairs, C_coarse, H/16, W/16)
        L_fine:   R^(num_pairs, C_fine,   H/8,  W/8)
       |
       v
    AR prior P(L_t | L_{t-1}) — small conditional density network estimates
    p(z) for rate term; consumes previous-frame latent z_{t-1} as context.
       |
       v
    Synthesis: bilinear-upsample latents to (H, W), concat across scales,
    small MLP -> RGB. Tiny (~10K params shared).
       |
       v
    Per-pair output: (rgb_0, rgb_1) — same synthesis used for both frames of
    the pair, but each frame has its own latent row.

Council notes:
- Total param target: ~200K (~10K shared synthesis + ~10K AR prior + ~180K latents)
- Latents are the rate budget; synthesis + AR prior fit inside the decoder blob.
- EMA decay 0.997 per CLAUDE.md "EMA — non-negotiable" (applied externally).

CLAUDE.md compliance:
- No silent device defaults (caller passes device)
- No scorer loading inside this module
- No /tmp paths
- Reviewable in 30 seconds per L12
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2  # 600 pairs


@dataclass(frozen=True)
class CoolChicConfig:
    """Static design-time parameters for cool_chic (L0 SKETCH).

    All fields keyword-overridable; no silent magic.
    """

    latent_channels_coarse: int = 4
    """Channels of the coarse-scale latent grid (per-pair)."""

    latent_channels_fine: int = 4
    """Channels of the fine-scale latent grid (per-pair)."""

    coarse_scale_factor: int = 16
    """Coarse-scale spatial downsample factor (H/16, W/16)."""

    fine_scale_factor: int = 8
    """Fine-scale spatial downsample factor (H/8, W/8)."""

    synthesis_hidden: int = 32
    """Hidden size of the shared synthesis MLP."""

    synthesis_layers: int = 3
    """Layers of the synthesis MLP (incl. output)."""

    ar_prior_hidden: int = 24
    """Hidden size of the AR prior conditional density network."""

    num_pairs: int = _PAIRS
    """Number of latent rows (600 for 1200-frame contest video)."""

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W


class _SynthesisMLP(nn.Module):
    """Tiny shared synthesis MLP: per-pixel feature -> RGB."""

    def __init__(self, in_ch: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_ch
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(prev, hidden, kernel_size=1))
            layers.append(nn.GELU())
            prev = hidden
        layers.append(nn.Conv2d(prev, 3, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sigmoid(self.net(x))


class _ARPriorNet(nn.Module):
    """Conditional AR prior: predicts (mean, log_scale) for z_t given z_{t-1}.

    This is a tiny conv-net mapping a previous-frame latent to per-element
    Gaussian parameters; the rate term consumes log p(z_t | z_{t-1}).
    """

    def __init__(self, in_ch: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, in_ch * 2, kernel_size=3, padding=1),
        )

    def forward(self, prev_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_scale), each same shape as prev_latent."""
        out = self.net(prev_latent)
        mean, log_scale = out.chunk(2, dim=1)
        # Clamp log_scale for numerical stability
        log_scale = log_scale.clamp(min=-7.0, max=7.0)
        return mean, log_scale


class CoolChicSubstrate(nn.Module):
    """Cool-Chic substrate: per-frame latents + shared synthesis + AR prior.

    Forward signature mirrors sane_hnerv for trainer interop:
        forward(pair_indices) -> (rgb_0, rgb_1), each (B, 3, H, W).

    The AR rate term is exposed separately via ``compute_ar_log_prob`` so the
    score-aware loss can include it.
    """

    def __init__(self, cfg: CoolChicConfig) -> None:
        super().__init__()
        self.cfg = cfg

        h_coarse = cfg.output_height // cfg.coarse_scale_factor
        w_coarse = cfg.output_width // cfg.coarse_scale_factor
        h_fine = cfg.output_height // cfg.fine_scale_factor
        w_fine = cfg.output_width // cfg.fine_scale_factor

        # Per-pair latent grids
        self.latents_coarse = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_channels_coarse, h_coarse, w_coarse).normal_(std=0.02)
        )
        self.latents_fine = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_channels_fine, h_fine, w_fine).normal_(std=0.02)
        )

        # Shared synthesis: takes concatenated coarse+fine latents (upsampled)
        synthesis_in = cfg.latent_channels_coarse + cfg.latent_channels_fine
        self.synthesis = _SynthesisMLP(synthesis_in, cfg.synthesis_hidden, cfg.synthesis_layers)

        # Per-frame heads (2 frames per pair) - we re-use synthesis but apply
        # a small per-frame offset embedding to break symmetry between frame 0
        # and frame 1.
        self.frame_offset = nn.Parameter(torch.zeros(2, synthesis_in, 1, 1))

        # AR prior: one for coarse, one for fine
        self.ar_prior_coarse = _ARPriorNet(cfg.latent_channels_coarse, cfg.ar_prior_hidden)
        self.ar_prior_fine = _ARPriorNet(cfg.latent_channels_fine, cfg.ar_prior_hidden)

    def _upsample_to_output(self, latent: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            latent,
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

        z_coarse = self.latents_coarse[pair_indices]  # (B, C_c, H/16, W/16)
        z_fine = self.latents_fine[pair_indices]  # (B, C_f, H/8, W/8)

        z_coarse_up = self._upsample_to_output(z_coarse)
        z_fine_up = self._upsample_to_output(z_fine)

        # (B, C_c + C_f, H, W)
        feat = torch.cat([z_coarse_up, z_fine_up], dim=1)

        # Per-frame offset breaks frame-0/frame-1 symmetry
        feat_0 = feat + self.frame_offset[0]
        feat_1 = feat + self.frame_offset[1]

        rgb_0 = self.synthesis(feat_0)
        rgb_1 = self.synthesis(feat_1)
        return rgb_0, rgb_1

    def compute_ar_log_prob(self) -> torch.Tensor:
        """Compute the AR-prior log-probability across all pair latents.

        Returns a scalar tensor = sum over pairs of log p(z_t | z_{t-1}).
        First pair uses a zero prior context. This is the rate proxy for the
        score-aware loss.
        """
        # Coarse axis
        coarse_log_p = self._ar_log_prob_chain(self.latents_coarse, self.ar_prior_coarse)
        fine_log_p = self._ar_log_prob_chain(self.latents_fine, self.ar_prior_fine)
        return coarse_log_p + fine_log_p

    @staticmethod
    def _ar_log_prob_chain(latents: torch.Tensor, prior_net: _ARPriorNet) -> torch.Tensor:
        """log-prob chain over pairs using AR prior conditioned on z_{t-1}.

        For t=0 we use a zero-context prediction; for t>=1 we condition on
        latents[t-1].
        """
        num_pairs = latents.shape[0]
        prev = torch.zeros_like(latents[0:1])  # (1, C, h, w)
        # Stack previous latents: [zero, latents[0], latents[1], ..., latents[N-2]]
        prev_stack = torch.cat([prev, latents[: num_pairs - 1]], dim=0)
        mean, log_scale = prior_net(prev_stack)
        # Gaussian log-prob
        # log p = -0.5 * ((z - mu) / sigma)^2 - log_scale - 0.5*log(2pi)
        diff = (latents - mean) / torch.exp(log_scale)
        log_two_pi = math.log(2.0 * math.pi)
        log_p = -0.5 * diff * diff - log_scale - 0.5 * log_two_pi
        return log_p.sum()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
