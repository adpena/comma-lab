"""NeRV pose-conditioning bolt-on — FiLM-style modulation across substrates.

Per operator directive 2026-05-11 (NeRV-family expansion bolt-on #2) +
CLAUDE.md HNeRV parity discipline. This bolt-on adds **per-frame pose
conditioning** to ANY NeRV-family substrate via Feature-wise Linear
Modulation (FiLM; Perez 2018). It is a generic, substrate-agnostic adapter
that ingests a per-pair pose vector and produces (scale, shift) modulation
parameters that any decoder can consume.

Why FiLM
--------
- **Lightweight**: per-frame pose → (scale, shift) of size (latent_dim,)
  each. The modulator adds tens-of-bytes per pair (compress time);
  inflate-time cost is one elementwise op per latent.
- **Compositional**: FiLM is a generic mechanism. The modulator output can
  be applied:
  - Pre-decoder: `z_modulated = scale * z + shift`, then forward through any
    NeRV substrate's decoder unchanged.
  - Mid-decoder: as feature-map modulation at any decoder stage (Phase B).
- **Sister to Phase 2 T15 time-varying FiLM**: T15 has its own pre-design
  memo and dispatch budget. THIS bolt-on is the GENERIC mechanism; T15 is a
  specific instantiation with its own training contract. Phase A bolt-on
  scope: pre-decoder modulation only; T15 covers mid-decoder.

Architecture (default config)
-----------------------------
- `FiLMModulator`: small MLP that takes pose (B, 6) → (scale (B, latent_dim),
  shift (B, latent_dim)). 2-layer MLP with hidden_dim=64.
- `apply_film`: helper that does `scale * z + shift` with optional clamp.
- `compute_pose_input_for_pair`: extracts a per-pair pose vector from the
  full pose stream (mean of the 2 frames' poses, padded to 6 dims).

CLAUDE.md compliance
--------------------
- L4 (inflate ≤ 200 LOC): the FiLM modulator's INFERENCE-TIME work is
  negligible (one MLP forward + elementwise op). The modulator is shipped
  in the archive (small).
- L5 (full RGB renderer): the bolt-on is a pre-decoder adapter; the
  underlying renderer remains a full RGB renderer.
- L8 (eval-roundtrip): no impact on the eval-roundtrip loop (modulation
  happens BEFORE the decoder's forward).
- L13 (KILL is last resort): N/A.

This bolt-on intentionally does NOT define an archive grammar — its weights
are shipped as a small additional section in the host substrate's archive,
under the substrate's existing schema_keys_in_order pinning.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FiLMModulatorConfig:
    """Frozen config for the FiLM pose-conditioning bolt-on.

    Attributes
    ----------
    pose_dim
        Input pose dimensionality. Default 6 (matches contest PoseNet output).
    latent_dim
        Output modulation dim (must match host substrate's latent_dim).
    hidden_dim
        Modulator MLP hidden width. Default 64.
    n_hidden_layers
        Number of hidden layers. Default 1 (so total MLP depth = 2).
    init_scale_to_one
        If True (default), bias-initialize the scale head so initial output
        is identity (scale=1, shift=0). This is the canonical FiLM init that
        ensures training starts equivalent to "no modulation".
    clamp_scale_min, clamp_scale_max
        Clamp range for the learned scale (numerical stability).
    """

    pose_dim: int = 6
    latent_dim: int = 16
    hidden_dim: int = 64
    n_hidden_layers: int = 1
    init_scale_to_one: bool = True
    clamp_scale_min: float = 0.1
    clamp_scale_max: float = 10.0

    def __post_init__(self) -> None:
        if self.pose_dim <= 0:
            raise ValueError(f"pose_dim must be positive, got {self.pose_dim}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.n_hidden_layers < 0:
            raise ValueError(f"n_hidden_layers must be non-negative, got {self.n_hidden_layers}")
        if self.clamp_scale_min <= 0 or self.clamp_scale_max <= self.clamp_scale_min:
            raise ValueError("clamp_scale_min must be positive and < clamp_scale_max")


# ── FiLMModulator ────────────────────────────────────────────────────────


class FiLMModulator(nn.Module):
    """Per-frame pose → (scale, shift) modulation params.

    Shapes:
        Input  pose: (B, pose_dim)
        Output scale: (B, latent_dim), shift: (B, latent_dim)

    The scale is post-clamped to [clamp_scale_min, clamp_scale_max] for
    numerical stability. With `init_scale_to_one=True`, the initial output is
    (scale ≈ 1, shift ≈ 0) so the modulator starts as identity.
    """

    def __init__(self, config: FiLMModulatorConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_dim = config.pose_dim
        for _ in range(config.n_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU(inplace=False))
            in_dim = config.hidden_dim
        self.body = nn.Sequential(*layers)
        # Two heads: scale + shift, each (latent_dim,)
        self.scale_head = nn.Linear(in_dim, config.latent_dim)
        self.shift_head = nn.Linear(in_dim, config.latent_dim)
        if config.init_scale_to_one:
            # Init scale_head so output starts at 1 (identity FiLM).
            nn.init.zeros_(self.scale_head.weight)
            nn.init.ones_(self.scale_head.bias)
            nn.init.zeros_(self.shift_head.weight)
            nn.init.zeros_(self.shift_head.bias)

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``pose (B, pose_dim) → (scale (B, latent_dim), shift (B, latent_dim))``."""
        if pose.dim() != 2 or pose.shape[1] != self.config.pose_dim:
            raise ValueError(
                f"FiLMModulator expected (B, {self.config.pose_dim}), got {tuple(pose.shape)}"
            )
        h = self.body(pose) if len(self.body) > 0 else pose
        scale = self.scale_head(h).clamp(
            min=self.config.clamp_scale_min, max=self.config.clamp_scale_max
        )
        shift = self.shift_head(h)
        return scale, shift


# ── apply_film helper ───────────────────────────────────────────────────


def apply_film(
    z: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """``z_out = scale * z + shift`` with shape checks."""
    if z.shape != scale.shape:
        raise ValueError(f"z shape {z.shape} != scale shape {scale.shape}")
    if z.shape != shift.shape:
        raise ValueError(f"z shape {z.shape} != shift shape {shift.shape}")
    return scale * z + shift


# ── pose extraction helper ──────────────────────────────────────────────


def compute_pose_input_for_pair(
    pose_stream: torch.Tensor,
    pair_indices: torch.Tensor,
    pose_dim: int = 6,
) -> torch.Tensor:
    """Extract per-pair pose input from the full pose stream.

    Phase A: takes the mean of the 2 frames' poses for each pair, then pads
    to ``pose_dim`` if necessary. This is a coarse summary; Phase B may use
    relative pose (frame[i+1] - frame[i]) or richer features.

    Parameters
    ----------
    pose_stream
        Full pose stream of shape (n_frames, P) where P >= pose_dim. The
        contest PoseNet emits 12-dim pose; the first 6 are used by the
        scorer. We slice to ``pose_dim`` then mean over the 2 frames in each
        pair.
    pair_indices
        Long tensor (B,) of pair indices into the pair sequence (NOT frame
        indices; frame_index = pair_index * 2).
    pose_dim
        Output pose vector dimensionality. Default 6.

    Returns
    -------
    Tensor of shape (B, pose_dim).
    """
    if pose_stream.dim() != 2:
        raise ValueError(f"pose_stream must be 2-D, got shape {tuple(pose_stream.shape)}")
    if pose_stream.shape[1] < pose_dim:
        raise ValueError(
            f"pose_stream has {pose_stream.shape[1]} dims, need >= {pose_dim}"
        )
    if pair_indices.dim() != 1:
        raise ValueError(f"pair_indices must be 1-D, got shape {tuple(pair_indices.shape)}")
    n_frames = pose_stream.shape[0]
    if (pair_indices.max() * 2 + 1) >= n_frames:
        raise ValueError(
            f"pair_indices max {pair_indices.max().item()} requires frame "
            f"{pair_indices.max() * 2 + 1} but pose_stream has {n_frames}"
        )
    pose_truncated = pose_stream[:, :pose_dim]
    out = []
    for idx in pair_indices.tolist():
        f0 = pose_truncated[2 * idx]
        f1 = pose_truncated[2 * idx + 1]
        out.append(0.5 * (f0 + f1))
    return torch.stack(out, dim=0)


# ── Compose helper: pose-modulated latent ───────────────────────────────


def modulate_latent(
    *,
    z: torch.Tensor,
    pose: torch.Tensor,
    modulator: FiLMModulator,
) -> torch.Tensor:
    """End-to-end: apply FiLM modulation to ``z`` conditioned on ``pose``.

    Returns the modulated latent ready for any NeRV-family decoder. Use this
    helper when wiring the bolt-on into a substrate's training loop.
    """
    scale, shift = modulator(pose)
    return apply_film(z, scale, shift)
