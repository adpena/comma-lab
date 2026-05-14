"""IBEncoder — variational q(z | frames) per Tishby-Zaslavsky 2015 IB.

Per the C6 across-class shift hypothesis: the encoder maps source video frames
to a LOW-DIM latent ``z`` that preserves only scorer-relevant information.
The IB Lagrangian (in mdl_loss.py) trades reconstruction error against
mutual-information ``I(z; frames)`` via the β coefficient.

Architecture (Hotz Carmack-style minimal CNN; reviewable in 30s per L12):

    Input  : per-pair RGB frame_1 (B, 3, H, W) in [0, 1]
             (frame_0 is reconstructed from z + decoder; SegNet sees only frame_1)
       ↓
    Conv 3 → 16 (stride=4, kernel=5)                   → (B, 16, H/4, W/4)
       ↓
    sin activation (SIREN-style; freq=30)
       ↓
    Conv 16 → 32 (stride=4, kernel=5)                   → (B, 32, H/16, W/16)
       ↓
    sin activation
       ↓
    Conv 32 → 64 (stride=4, kernel=3)                   → (B, 64, H/64, W/64)
       ↓
    AdaptiveAvgPool2d(1) → (B, 64, 1, 1) → flatten     → (B, 64)
       ↓
    Linear 64 → 2*d_z                                  → (B, 2*d_z)
       ↓
    chunk into (μ, log_σ²)                             → q(z | frames) = N(μ, σ²)

Sampling at training (reparameterization trick):
    z ~ N(μ, σ²)  via  z = μ + σ ⊙ ε,  ε ~ N(0, I)

At inflate time: z is stored quantized + dequantized; encoder is also stored
(needed if we want to re-derive z from frames; for the contest packet z is
fixed per-pair so encoder runtime is not strictly required at inflate, but
storing it enables verifiability + future probe-disambiguator).

Per CLAUDE.md FORBIDDEN_PATTERNS:
- NO scorer load in this module (score-aware loss is separate).
- NO silent device defaults (caller passes device).
- NO /tmp paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SinAct(nn.Module):
    """Sine activation (SIREN/NeRF-style); ω=30 per NeRF default."""

    def __init__(self, w: float = 30.0) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(self.w * x)


class IBEncoder(nn.Module):
    """Variational q(z | frames) — outputs (μ, log_σ²) for a Gaussian posterior.

    Args:
        latent_dim: dimensionality of the per-pair latent z (default 24).
        input_channels: input channel count (default 3 = RGB).
        sin_freq: SIREN activation frequency (default 30.0).
    """

    def __init__(
        self,
        latent_dim: int = 24,
        input_channels: int = 3,
        sin_freq: float = 30.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or latent_dim > 256:
            raise ValueError(f"latent_dim must be in (0, 256]; got {latent_dim}")

        self.latent_dim = int(latent_dim)
        self.sin_freq = float(sin_freq)

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=4, padding=2)
        self.act1 = _SinAct(sin_freq)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2)
        self.act2 = _SinAct(sin_freq)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1)
        self.act3 = _SinAct(sin_freq)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Output 2*d_z (μ and log_σ²)
        self.head = nn.Linear(64, 2 * latent_dim)
        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN initialization: weights ~ Uniform(-c/fan_in, c/fan_in)."""
        w = self.sin_freq
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode frames → (μ, log_σ²) of q(z | frames).

        Args:
            frames: input RGB tensor ``(B, C, H, W)`` in unit range [0, 1].

        Returns:
            (mu, logvar): each ``(B, latent_dim)`` float32.
        """
        if frames.dim() != 4:
            raise ValueError(f"frames must be 4D (B, C, H, W); got {tuple(frames.shape)}")

        h = self.act1(self.conv1(frames))
        h = self.act2(self.conv2(h))
        h = self.act3(self.conv3(h))
        h = self.pool(h).flatten(1)  # (B, 64)
        out = self.head(h)
        mu, logvar = out.chunk(2, dim=-1)
        # Clamp logvar to a reasonable range to avoid numerical blowup
        logvar = logvar.clamp(min=-10.0, max=10.0)
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(μ, σ²) via the reparameterization trick.

        Args:
            mu: posterior mean ``(B, latent_dim)``.
            logvar: posterior log-variance ``(B, latent_dim)``.

        Returns:
            z: sampled latent ``(B, latent_dim)``.
        """
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # At eval: use the posterior mean (deterministic)
            return mu

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["IBEncoder"]
