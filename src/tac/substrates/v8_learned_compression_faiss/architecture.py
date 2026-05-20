# SPDX-License-Identifier: MIT
"""Tiny deterministic V8 categorical-posterior helpers.

The real V8 design calls for a learned Gumbel-Softmax categorical posterior
plus scale hyperprior. This module implements the cheap local smoke contract:
deterministic categorical codewords and inspectable hyperprior statistics that
can be packed into a byte-closed archive. It does not claim trained score
movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class V8CategoricalPosteriorConfig:
    """Local V8 categorical posterior configuration."""

    categorical_groups: int = 16
    codebook_size: int = 128
    seed: int = 20260520

    def validate(self) -> None:
        if self.categorical_groups <= 0 or self.categorical_groups > 255:
            raise ValueError("categorical_groups must be in [1, 255]")
        if self.codebook_size <= 1 or self.codebook_size > 256:
            raise ValueError("codebook_size must be in [2, 256]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def deterministic_categorical_codewords(
    num_frames: int,
    config: V8CategoricalPosteriorConfig,
) -> bytes:
    """Return one deterministic categorical codeword per output frame.

    This is a local substitute for the trained straight-through
    Gumbel-Softmax sampler. It is deterministic, bounded by the configured
    codebook, and produces a concrete codeword stream for byte-closed inflate.
    """

    config.validate()
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    state = (int(config.seed) ^ 0x9E3779B9) & 0xFFFFFFFF
    out = bytearray()
    for frame_idx in range(num_frames):
        group = frame_idx % config.categorical_groups
        state = (1664525 * state + 1013904223 + group + frame_idx) & 0xFFFFFFFF
        out.append(state % config.codebook_size)
    return bytes(out)


def deterministic_rgb_codebook(codebook_size: int) -> tuple[tuple[int, int, int], ...]:
    """Build a compact deterministic RGB codebook for fixture inflation."""

    if codebook_size <= 1 or codebook_size > 256:
        raise ValueError("codebook_size must be in [2, 256]")
    return tuple(
        (
            (idx * 37 + 11) % 256,
            (idx * 67 + 23) % 256,
            (idx * 97 + 41) % 256,
        )
        for idx in range(codebook_size)
    )


def build_scale_hyperprior_from_codewords(
    codewords: bytes,
    *,
    categorical_groups: int,
) -> tuple[tuple[float, float], ...]:
    """Return per-group ``(mean_code, scale_code)`` hyperprior stats."""

    if not codewords:
        raise ValueError("codewords must be non-empty")
    if categorical_groups <= 0 or categorical_groups > 255:
        raise ValueError("categorical_groups must be in [1, 255]")
    groups: list[list[int]] = [[] for _ in range(categorical_groups)]
    for idx, code in enumerate(codewords):
        groups[idx % categorical_groups].append(int(code))
    stats: list[tuple[float, float]] = []
    for values in groups:
        if not values:
            stats.append((0.0, 1.0))
            continue
        mean = sum(values) / float(len(values))
        var = sum((value - mean) ** 2 for value in values) / float(len(values))
        stats.append((mean, max(math.sqrt(var), 1.0)))
    return tuple(stats)
