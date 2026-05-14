# SPDX-License-Identifier: MIT
"""128K-parameter Quantizr-class FiLM-conditioned decoder for PARADIGM-δεζ T1.

Architecture
------------

Quantizr (PR101 silver, 0.33 archive) used a FiLM-conditioned depthwise-
separable CNN at ~88K params + ~64KB FP4. Track 1 targets a slightly larger
**128K-parameter budget** (~80KB FP4) to absorb the latent + hyperprior
side-information without losing reconstruction quality. The architecture
keeps Quantizr's depthwise-separable pattern so quantisation behaves
predictably:

  Stem    : Linear(latent_dim → C0 * H0 * W0)             — base feature map
  Stage k : DepthwiseConv(Ck) + FiLM(Ck, latent) + PixelShuffle(2)
            for k in [1..6]; channel taper Ck = base * shrink^k
  Refine  : 1x1 Conv -> dilated DepthwiseConv -> 1x1 Conv  — residual head
  Heads   : Conv2d(C_final, 3) for frame 0 and frame 1 (separate)

The base channel count + shrink factor is tuned to hit ~128K params at
``latent_dim=28`` (matching the canonical A1 latents). The defaults are
verified by ``decoder_128k_param_count`` to land in [120K, 136K]; an
``RuntimeError`` is raised if a config drifts outside that band so future
edits cannot silently change the param budget.

CLAUDE.md compliance
--------------------

- Pure ``torch.nn`` module; no MPS / CUDA defaults baked in (caller chooses
  device).
- ``forward()`` is deterministic given inputs (no dropout, no random
  augmentation in the decoder body).
- The output range is ``[0, 255]`` via sigmoid * 255 to match the contest
  ``upstream/evaluate.py`` expectation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

DECODER_128K_PARAM_BUDGET = 128_000
"""Nominal target param count. Empirical band [120K, 136K] is enforced."""

DECODER_128K_PARAM_BAND = (120_000, 136_000)
"""(low, high) inclusive band the constructed decoder must land within.

Defaults below were tuned against this band; a CI/preflight regression test
should re-verify after any config change.
"""


@dataclass(frozen=True)
class Decoder128KConfig:
    """Static configuration for the 128K decoder.

    Attributes
    ----------
    latent_dim : int
        Per-pair latent dimensionality. Defaults to 28 (A1 canonical).
    base_channels : int
        Channel count at the base 6×8 feature map. Defaults to 32.
    base_h, base_w : int
        Initial spatial size before the 6 PixelShuffle×2 stages
        (final output is base_h * 64 × base_w * 64 = 384×512 with defaults).
    eval_h, eval_w : int
        Output spatial size; the 6 stages must produce this.
    channel_shrink : tuple[float, ...]
        Per-stage shrink factor applied to base_channels (k=0..5). Defaults
        match Quantizr taper.
    film_hidden : int
        Hidden width of the FiLM conditioning MLP that maps latent → per-stage
        γ/β.
    """

    latent_dim: int = 28
    base_channels: int = 46
    base_h: int = 6
    base_w: int = 8
    eval_h: int = 384
    eval_w: int = 512
    channel_shrink: tuple[float, ...] = (1.0, 1.0, 0.875, 0.75, 0.625, 0.5)
    film_hidden: int = 24

    def stage_channels(self) -> list[int]:
        chans = [self.base_channels]
        for s in self.channel_shrink:
            chans.append(max(8, int(round(self.base_channels * s))))
        return chans

    def __post_init__(self) -> None:
        # Validate output size matches 6 PixelShuffle stages.
        expected_h = self.base_h * (2 ** 6)
        expected_w = self.base_w * (2 ** 6)
        if expected_h != self.eval_h or expected_w != self.eval_w:
            raise ValueError(
                f"6 PixelShuffle stages from ({self.base_h}, {self.base_w}) "
                f"produce ({expected_h}, {expected_w}); config requested "
                f"({self.eval_h}, {self.eval_w}). Adjust base_h/base_w."
            )
        if len(self.channel_shrink) != 6:
            raise ValueError(
                "channel_shrink must have exactly 6 entries (one per "
                f"PixelShuffle stage); got {len(self.channel_shrink)}"
            )


class _FiLMConditioner(nn.Module):
    """Per-stage FiLM conditioner: latent → (γ, β)."""

    def __init__(self, latent_dim: int, hidden: int, channels: int):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden)
        self.linear2 = nn.Linear(hidden, channels * 2)
        self.channels = channels

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim); x: (B, C, H, W)
        h = F.relu(self.linear1(z))
        gamma_beta = self.linear2(h)  # (B, 2C)
        gamma = gamma_beta[:, : self.channels].view(-1, self.channels, 1, 1)
        beta = gamma_beta[:, self.channels:].view(-1, self.channels, 1, 1)
        return x * (1.0 + gamma) + beta


class _DepthwiseSeparableUp(nn.Module):
    """One up-stage: depthwise → pointwise(out*4) → PixelShuffle(2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.skip_proj = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        identity = self.skip_proj(identity)
        x = self.ps(self.pw(self.dw(x)))
        return x + identity


class Decoder128K(nn.Module):
    """128K-parameter FiLM-conditioned decoder for Track 1.

    See module docstring for architecture. The class is intentionally
    lightweight (no learnable embedding tables, no attention) so that
    FP4 PTQ + LSQ QAT stays well-conditioned.
    """

    def __init__(self, config: Decoder128KConfig | None = None):
        super().__init__()
        self.config = config or Decoder128KConfig()
        cfg = self.config

        chans = cfg.stage_channels()
        self.stage_channels_list = chans

        self.stem = nn.Linear(cfg.latent_dim, chans[0] * cfg.base_h * cfg.base_w)

        self.stages = nn.ModuleList()
        self.films = nn.ModuleList()
        for k in range(6):
            self.stages.append(_DepthwiseSeparableUp(chans[k], chans[k + 1]))
            self.films.append(
                _FiLMConditioner(cfg.latent_dim, cfg.film_hidden, chans[k + 1])
            )

        final_ch = chans[-1]
        # Refine: 1x1 -> dilated dw -> 1x1
        refine_mid = max(8, final_ch // 2)
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, refine_mid, 1),
            nn.Conv2d(refine_mid, refine_mid, 3, padding=2, dilation=2, groups=refine_mid),
            nn.Conv2d(refine_mid, final_ch, 1),
        )

        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a (B, latent_dim) latent batch into (B, 2, 3, H, W) RGB pairs.

        The output tensor matches the A1 decoder's contract so the trainer
        can swap one for the other without changing the loss path.
        """
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"expected (B, {self.config.latent_dim}) latent, got {tuple(z.shape)}"
            )
        cfg = self.config
        B = z.shape[0]
        x = self.stem(z).view(B, self.stage_channels_list[0], cfg.base_h, cfg.base_w)
        x = torch.sin(x)
        for stage, film in zip(self.stages, self.films):
            x = stage(x)
            x = film(z, x)
            x = torch.sin(x)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)


def decoder_128k_param_count(config: Decoder128KConfig | None = None) -> int:
    """Construct a Decoder128K and return its trainable param count."""
    decoder = Decoder128K(config or Decoder128KConfig())
    return int(sum(p.numel() for p in decoder.parameters() if p.requires_grad))


def build_decoder_128k(
    config: Decoder128KConfig | None = None,
    *,
    enforce_param_band: bool = True,
) -> Decoder128K:
    """Build a Decoder128K and verify it lands inside :data:`DECODER_128K_PARAM_BAND`.

    Parameters
    ----------
    config : Decoder128KConfig
        Config (defaults: latent_dim=28, base_channels=32, ...)
    enforce_param_band : bool
        If True, raise RuntimeError when param count is outside the band
        ``DECODER_128K_PARAM_BAND``. Disable only for explicit ablation
        studies that intentionally probe a different param budget.

    Returns
    -------
    Decoder128K
    """
    cfg = config or Decoder128KConfig()
    decoder = Decoder128K(cfg)
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    low, high = DECODER_128K_PARAM_BAND
    if enforce_param_band and not (low <= n_params <= high):
        raise RuntimeError(
            f"Decoder128K with config {cfg} has {n_params} params; "
            f"expected band [{low}, {high}]. Tune base_channels/film_hidden "
            "or pass enforce_param_band=False for ablation."
        )
    return decoder
