# SPDX-License-Identifier: MIT
"""Scorer-free DPW1 inflate/apply helpers."""

from __future__ import annotations

from collections.abc import Iterable

from .archive import (
    DrivingPriorWorldModelArchive,
    DrivingPriorWorldModelError,
    parse_archive,
)
from .renderer import render_prior_world_model


def apply_world_model_archive(
    archive: DrivingPriorWorldModelArchive | bytes,
    pair_indices: Iterable[int] | None = None,
    *,
    require_nonzero: bool = True,
):
    """Apply a parsed DPW1 archive and return generated RGB frame pairs.

    This helper is safe for inflate-time use: it imports no contest scorers,
    reads no hidden sidecars, and treats structural no-op packets as blockers
    by default.
    """

    parsed = parse_archive(archive) if isinstance(archive, (bytes, bytearray)) else archive
    if require_nonzero and parsed.structural_noop:
        raise DrivingPriorWorldModelError(
            "driving-prior world-model archive is structural no-op"
        )
    pairs = tuple(range(parsed.config.num_pairs)) if pair_indices is None else tuple(pair_indices)
    return render_prior_world_model(
        parsed.config,
        parsed.prior_weights,
        parsed.residual_bytes,
        pairs,
    )


def inflate_world_model_archive(
    archive: DrivingPriorWorldModelArchive | bytes,
    pair_indices: Iterable[int] | None = None,
    *,
    require_nonzero: bool = True,
):
    """Alias for callers that treat the scaffold as an inflate primitive."""

    return apply_world_model_archive(
        archive,
        pair_indices=pair_indices,
        require_nonzero=require_nonzero,
    )
