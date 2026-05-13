"""Deterministic scorer-free renderer for the DPW1 placeholder archive."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .archive import (
    DrivingPriorWorldModelError,
    expected_prior_weight_bytes,
    expected_residual_bytes,
)

if TYPE_CHECKING:
    from .config import DrivingPriorWorldModelConfig


def render_prior_world_model(
    config: DrivingPriorWorldModelConfig,
    prior_weights: bytes,
    residual_bytes: bytes,
    pair_indices: Iterable[int],
):
    """Render RGB frame pairs from charged prior/codebook and residual bytes.

    Returns a ``numpy.uint8`` array with shape ``(B, 2, H, W, 3)``. The
    implementation is intentionally deterministic and scorer-free; it consumes
    both charged byte sections directly.
    """

    import numpy as np

    prior = bytes(prior_weights)
    residual = bytes(residual_bytes)
    if len(prior) != expected_prior_weight_bytes(config):
        raise DrivingPriorWorldModelError("prior_weights length does not match config")
    if len(residual) != expected_residual_bytes(config):
        raise DrivingPriorWorldModelError("residual_bytes length does not match config")

    pairs = tuple(int(idx) for idx in pair_indices)
    for idx in pairs:
        if idx < 0 or idx >= config.num_pairs:
            raise DrivingPriorWorldModelError(f"pair index {idx} out of range")

    codebook = np.frombuffer(prior, dtype=np.uint8).reshape(config.codebook_entries, 3)
    residual_grid = np.frombuffer(residual, dtype=np.int8).reshape(
        config.num_pairs,
        2,
        config.residual_grid_height,
        config.residual_grid_width,
        3,
    )
    y = np.arange(config.output_height, dtype=np.int32)[:, None]
    x = np.arange(config.output_width, dtype=np.int32)[None, :]
    residual_y = np.minimum(
        (np.arange(config.output_height) * config.residual_grid_height)
        // config.output_height,
        config.residual_grid_height - 1,
    )
    residual_x = np.minimum(
        (np.arange(config.output_width) * config.residual_grid_width)
        // config.output_width,
        config.residual_grid_width - 1,
    )

    out = np.empty(
        (len(pairs), 2, config.output_height, config.output_width, 3),
        dtype=np.uint8,
    )
    for batch_idx, pair_idx in enumerate(pairs):
        for frame_in_pair in (0, 1):
            codebook_index = (
                x * 31 + y * 17 + pair_idx * 13 + frame_in_pair * 7
            ) % config.codebook_entries
            base = codebook[codebook_index].astype(np.int16)
            residual_block = residual_grid[pair_idx, frame_in_pair][
                residual_y[:, None],
                residual_x[None, :],
            ].astype(np.int16)
            out[batch_idx, frame_in_pair] = np.clip(base + residual_block, 0, 255).astype(
                np.uint8
            )
    return out
