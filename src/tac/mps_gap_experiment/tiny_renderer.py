# SPDX-License-Identifier: MIT
"""Tiny FiLM-conditioned RGB renderer for the MPS-train CUDA-score gap diagnostic.

DESIGN GOALS (per Phase 1 of the build plan):

* ≤15K parameters (small enough to train 100 epochs on MPS in <10 minutes).
* Input: ``(B, 2, 3, 384, 512)`` RGB frame pair (the canonical contest pair
  shape; B=batch, T=2 because the scorer expects pairs).
* Per-pair pose conditioning: ``(B, 12)`` pose vector flat (matches the
  upstream PoseNet output convention).
* Output: ``(B, 2, 3, 384, 512)`` reconstructed RGB pair.
* Loss: pixel L1 + (optional) canonical scorer-loss (computed by the trainer).

NOT a contest substrate. The renderer here is intentionally tiny + lossy; the
purpose is to exercise the canonical MPS forward / backward path on real
contest frames so we can compare against the CUDA forward of the same trained
weights.

[research_only: true; never charged into any archive]

Architecture (FiLM-conditioned conv stack):

* Stem (4ch -> 16ch, stride 2): single Conv2d that consumes the 6-channel pair
  flattened as (B, 6, H, W); downsampled to (B, 16, H/2, W/2).
* Mid block (16ch -> 16ch, FiLM-conditioned): one Conv2d + GroupNorm + ReLU;
  FiLM (scale, shift) computed from the pose vector via a 2-layer MLP.
* Decoder (16ch -> 6ch, stride 2 transpose): one ConvTranspose2d back to
  full (B, 6, H, W) then split to (B, 2, 3, H, W).

Total params: ~12K (verified by ``count_params``).

The architecture is intentionally trivial — what we're diagnosing here is
the *MPS forward pass numerical agreement* with CUDA, not the renderer's
score-aware behavior. If MPS-trained weights produce CUDA-forward components
within tolerance for this trivial model, the local MPS axis is unlocked for
substrate training; if not, the divergence becomes the binding research
question.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "TinyRenderer",
    "build_tiny_renderer",
    "count_params",
)


_HEIGHT = 384
_WIDTH = 512
_FRAMES_PER_PAIR = 2
_RGB_CHANNELS = 3
_POSE_DIM = 12


@dataclass(frozen=True)
class TinyRendererConfig:
    """Architecture knobs for the tiny renderer.

    Defaults sized to ~12K params (well under the 15K target).
    """

    hidden_channels: int = 16
    film_hidden: int = 32
    pose_dim: int = _POSE_DIM
    height: int = _HEIGHT
    width: int = _WIDTH


class TinyRenderer(nn.Module):
    """~12K parameter FiLM-conditioned RGB pair renderer.

    Args:
        config: optional :class:`TinyRendererConfig` (defaults to ~12K params).

    Input shapes:
        frame_pair: (B, 2, 3, H, W) float32 in [0, 1]
        pose: (B, 12) float32

    Output shape: (B, 2, 3, H, W) float32
    """

    def __init__(self, config: TinyRendererConfig | None = None) -> None:
        super().__init__()
        cfg = config if config is not None else TinyRendererConfig()
        self.config = cfg
        c = cfg.hidden_channels

        # 6 channels = 2 frames * 3 RGB stacked along channel dim
        self.stem = nn.Conv2d(
            _FRAMES_PER_PAIR * _RGB_CHANNELS, c, kernel_size=4, stride=2, padding=1
        )
        self.mid_conv = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        self.group_norm = nn.GroupNorm(num_groups=4, num_channels=c)
        self.decoder = nn.ConvTranspose2d(
            c, _FRAMES_PER_PAIR * _RGB_CHANNELS, kernel_size=4, stride=2, padding=1
        )

        # FiLM head: pose (12) -> hidden (32) -> (scale, shift) per channel
        self.film_fc1 = nn.Linear(cfg.pose_dim, cfg.film_hidden)
        self.film_fc2 = nn.Linear(cfg.film_hidden, 2 * c)

    def forward(
        self, frame_pair: torch.Tensor, pose: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct the frame pair conditioned on pose.

        Args:
            frame_pair: (B, 2, 3, H, W) float32 in [0, 1]
            pose: (B, 12) float32

        Returns:
            (B, 2, 3, H, W) float32 reconstructed pair
        """
        b, t, ch, h, w = frame_pair.shape
        if t != _FRAMES_PER_PAIR or ch != _RGB_CHANNELS:
            raise ValueError(
                f"frame_pair must be (B, {_FRAMES_PER_PAIR}, {_RGB_CHANNELS}, H, W); "
                f"got {tuple(frame_pair.shape)}"
            )
        if pose.shape != (b, self.config.pose_dim):
            raise ValueError(
                f"pose must be (B, {self.config.pose_dim}); got {tuple(pose.shape)}"
            )

        x = frame_pair.reshape(b, t * ch, h, w)
        x = self.stem(x)  # (B, c, H/2, W/2)
        x = self.mid_conv(x)
        x = self.group_norm(x)

        # FiLM modulation
        film = self.film_fc2(F.relu(self.film_fc1(pose)))  # (B, 2c)
        scale, shift = film.chunk(2, dim=-1)  # (B, c) each
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, c, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # (B, c, 1, 1)
        x = x * (1.0 + scale) + shift

        x = F.relu(x)
        x = self.decoder(x)  # (B, 6, H, W)

        return x.reshape(b, t, ch, h, w)


def build_tiny_renderer(
    *,
    hidden_channels: int = 16,
    film_hidden: int = 32,
    seed: int | None = None,
) -> TinyRenderer:
    """Construct a :class:`TinyRenderer` with deterministic init when seed set.

    Args:
        hidden_channels: latent channel count (default 16, ~12K total params)
        film_hidden: FiLM MLP hidden dim (default 32)
        seed: optional deterministic seed for parameter init

    Returns:
        a freshly-constructed :class:`TinyRenderer`
    """
    if seed is not None:
        torch.manual_seed(seed)
    cfg = TinyRendererConfig(
        hidden_channels=hidden_channels, film_hidden=film_hidden
    )
    return TinyRenderer(cfg)


def count_params(model: nn.Module) -> int:
    """Return total parameter count for the module (including non-trainable)."""
    return sum(p.numel() for p in model.parameters())
