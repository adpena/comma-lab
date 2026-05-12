"""NeRV-Enc/Dec separated bolt-on — explicit encoder/decoder split for any NeRV.

Per operator directive 2026-05-11 (NeRV-family expansion bolt-on #1) +
CLAUDE.md HNeRV parity discipline. This bolt-on lets ANY NeRV-family
substrate (NeRV / BlockNeRV / FFNeRV / DSNeRV / HiNeRV / TCNeRV / MNeRV)
run with an EXPLICIT compress-time encoder + inflate-time decoder split,
where the encoder learns to embed input frames into latents and the decoder
reconstructs.

Why an explicit Enc/Dec split
-----------------------------
- **Faster fitting**: Lane 12-v2 / PR100 substrates train per-pair latents
  by SGD on the latent table itself (one row per pair). The Enc/Dec split
  amortizes that over an encoder network so new pairs can be embedded with
  ONE forward pass (≈seconds) instead of needing per-pair gradient descent
  (≈minutes-hours).
- **Stronger generalization signal**: the encoder must learn to compress
  arbitrary frames, which forces the latent space to be smooth and
  semantically meaningful, not just memorized per-pair.
- **Compositional**: this module exposes a generic `NeRVEncoder` interface
  that takes (B, 2, 3, H, W) → (B, latent_dim). Pair it with ANY NeRV decoder
  by passing the encoder's output to the decoder's `forward`.

Architecture (default config)
-----------------------------
- Encoder: a small CNN (4 stride-2 stages) + global-avg-pool + linear head
  that maps (B, 2, 3, 384, 512) → (B, latent_dim).
- Decoder: provided by the caller (any NeRV-family substrate).
- Training mode: encoder + decoder trained jointly via the same score-aware
  Lagrangian as Lane 12-v2.
- Inflate-time: ONLY the decoder is shipped. The encoded latents (one per
  pair) are pre-computed at compress time and packed in the latent_blob,
  exactly as in Lane 12-v2 (so the encoder is a TRAINING tool, not an
  inflate-time component).

CLAUDE.md compliance
--------------------
- L4 (inflate ≤ 200 LOC): the encoder is COMPRESS-TIME ONLY. Nothing about
  this bolt-on changes the inflate runtime budget — the decoder still
  reconstructs per-pair from latents.
- L5 (full RGB renderer): the decoder this bolt-on couples with must be a
  full RGB renderer. The encoder is symmetric in C+H+W.
- L8 (eval-roundtrip): the trainer using this bolt-on still routes through
  `eval_roundtrip_uint8_clamp` for the decoder output (rendered side).
- L13 (KILL is last resort): N/A.

This module deliberately does NOT define an archive grammar — the encoder
is NOT shipped. The caller's substrate-specific archive grammar is the only
on-disk format.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NeRVEncoderConfig:
    """Frozen config for the NeRV-Enc/Dec bolt-on.

    Attributes
    ----------
    latent_dim
        Output latent dimensionality. Must match the paired NeRV decoder.
    base_channels
        Encoder base channel width.
    n_stages
        Number of stride-2 stages. Default 4 → 16x downsample.
    eval_size
        Input frame size (H, W). Default (384, 512).
    frames_per_pair
        Frames per encoded pair (always 2 for contest pair).
    """

    latent_dim: int = 16
    base_channels: int = 16
    n_stages: int = 4
    eval_size: tuple[int, int] = (384, 512)
    frames_per_pair: int = 2

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.n_stages < 1:
            raise ValueError(f"n_stages must be ≥1, got {self.n_stages}")
        if self.frames_per_pair != 2:
            raise ValueError("Phase A pinned at frames_per_pair=2")


# ── Encoder ──────────────────────────────────────────────────────────────


class NeRVEncoder(nn.Module):
    """Compress-time encoder: (B, 2, 3, H, W) → (B, latent_dim).

    Uses a small CNN over the concatenated frame pair (channel-stacked) +
    global-avg-pool + linear head. The output is the per-pair latent that
    feeds the paired decoder.
    """

    def __init__(self, config: NeRVEncoderConfig) -> None:
        super().__init__()
        self.config = config
        in_ch = 3 * config.frames_per_pair  # channel-stacked pair
        layers: list[nn.Module] = []
        current_ch = in_ch
        for i in range(config.n_stages):
            out_ch = config.base_channels * (2 ** i)
            layers.append(nn.Conv2d(current_ch, out_ch, 3, stride=2, padding=1))
            layers.append(nn.GroupNorm(min(8, out_ch), out_ch))
            layers.append(nn.ReLU(inplace=False))
            current_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.final_channels = current_ch
        self.head = nn.Linear(current_ch, config.latent_dim)

    def forward(self, frame_pair: torch.Tensor) -> torch.Tensor:
        """``frame_pair (B, 2, 3, H, W) → (B, latent_dim)``."""
        if frame_pair.dim() != 5:
            raise ValueError(
                f"NeRVEncoder expected 5-D (B, 2, 3, H, W), got {tuple(frame_pair.shape)}"
            )
        if frame_pair.shape[1] != self.config.frames_per_pair:
            raise ValueError(
                f"NeRVEncoder expected frames_per_pair={self.config.frames_per_pair}, "
                f"got {frame_pair.shape[1]}"
            )
        if frame_pair.shape[2] != 3:
            raise ValueError(
                f"NeRVEncoder expected 3 channels, got {frame_pair.shape[2]}"
            )
        B, F_pp, C, H, W = frame_pair.shape
        # Channel-stack frames into (B, F_pp*C, H, W) and normalize to [0, 1].
        x = frame_pair.float().reshape(B, F_pp * C, H, W) / 255.0
        x = self.body(x)
        # Global avg pool → (B, final_channels)
        x = x.mean(dim=(2, 3))
        return self.head(x)


# ── Compress-time orchestrator ───────────────────────────────────────────


def encode_pair_batch(
    *,
    encoder: NeRVEncoder,
    frame_pairs_uint8: torch.Tensor,
) -> torch.Tensor:
    """Encode a batch of frame pairs to latents at compress time.

    Inputs at camera resolution (B, 2, 3, 874, 1164) are downsampled to the
    encoder's expected `eval_size` before forward. Returns (B, latent_dim).

    This helper is the TRAINING-time + COMPRESS-TIME orchestrator. At
    inflate-time, the encoder is NOT used — the encoded latents are
    pre-computed at compress time and packed in the archive.
    """
    if frame_pairs_uint8.dim() != 5:
        raise ValueError(
            f"encode_pair_batch expected 5-D (B, 2, 3, H, W), got {tuple(frame_pairs_uint8.shape)}"
        )
    B, F_pp, C, H, W = frame_pairs_uint8.shape
    target_h, target_w = encoder.config.eval_size
    if (H, W) != (target_h, target_w):
        flat = frame_pairs_uint8.float().reshape(B * F_pp, C, H, W)
        resized = F.interpolate(
            flat, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        frame_pairs_input = resized.reshape(B, F_pp, C, target_h, target_w)
    else:
        frame_pairs_input = frame_pairs_uint8.float()
    return encoder(frame_pairs_input)


# ── Joint enc-dec training loss helper ───────────────────────────────────


def joint_train_step_with_decoder(
    *,
    encoder: NeRVEncoder,
    decoder: nn.Module,
    frame_pairs_uint8: torch.Tensor,
    decoder_train_step_fn,
    pair_indices: torch.Tensor,
    extra_kwargs: dict | None = None,
) -> dict:
    """Joint encoder+decoder training step.

    The encoder maps frame pairs → latents; those latents are passed to the
    decoder's `train_step` via the caller-supplied `decoder_train_step_fn`.
    The caller's `decoder_train_step_fn` is expected to accept latents
    (or a latent_table whose `forward(pair_indices)` returns latents) and
    a frame_pairs_uint8 + scorer arguments.

    Phase A: this is a LATENT-INJECTION wrapper — it builds an in-memory
    `_OnTheFlyLatentTable` whose `forward(pair_indices)` returns the encoder's
    latents at the requested indices, so the substrate's existing train_step
    contract is preserved. Substrate-specific train_step is provided by the
    caller (so this bolt-on stays decoder-agnostic).

    Returns the dict produced by the substrate's train_step (typically with
    keys `loss`, `loss_seg`, `loss_pose`, ...).
    """
    if extra_kwargs is None:
        extra_kwargs = {}
    latents = encode_pair_batch(encoder=encoder, frame_pairs_uint8=frame_pairs_uint8)

    # Lookup-table style: when decoder_train_step_fn calls latent_table(pair_indices),
    # we need to return the encoder's latents for those indices. Since the encoder
    # produces latents in the SAME order as the input batch, we map pair_indices →
    # batch positions via a contiguous lookup.
    if pair_indices.shape[0] != latents.shape[0]:
        raise ValueError(
            f"pair_indices size {pair_indices.shape[0]} != encoder batch {latents.shape[0]}"
        )

    class _OnTheFlyLatentTable(nn.Module):
        def __init__(self, batch_latents: torch.Tensor, batch_pair_indices: torch.Tensor) -> None:
            super().__init__()
            self.batch_latents = batch_latents
            self.batch_pair_indices = batch_pair_indices
            # Pre-compute index map: batch_pair_indices[i] → row i in batch_latents.
            self._index_map = {
                int(idx.item()): i for i, idx in enumerate(batch_pair_indices)
            }

        def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
            rows = []
            for idx in pair_indices.tolist():
                if idx not in self._index_map:
                    raise ValueError(
                        f"pair_index {idx} not in encoder batch (have {sorted(self._index_map)})"
                    )
                rows.append(self.batch_latents[self._index_map[idx]])
            return torch.stack(rows, dim=0)

    on_the_fly = _OnTheFlyLatentTable(latents, pair_indices)
    return decoder_train_step_fn(
        latent_table=on_the_fly,
        pair_indices=pair_indices,
        gt_pairs_uint8=frame_pairs_uint8,
        **extra_kwargs,
    )
