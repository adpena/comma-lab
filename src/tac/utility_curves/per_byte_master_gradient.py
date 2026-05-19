# SPDX-License-Identifier: MIT
"""Per-byte master-gradient utility (∂score/∂byte from `tac.master_gradient`).

The canonical per-byte master gradient lives in
``tac.master_gradient_consumers``: per-pair ``∂score/∂byte`` estimates
indexed by archive byte position. This module exposes ONE utility
callable that maps an archive bytes tensor + master-gradient tensor to a
per-byte sensitivity scalar — high in score-relevant bytes, low in
score-orthogonal bytes — that the unified action consumes as a
sensitivity-aware rate-axis contribution.

Per the synthesis memo §"OTHER APPLICATIONS" #24: per-byte master-gradient
sensitivity — bit-allocator on per-byte sensitivity. This utility IS the
per-byte sensitivity term in that pairing.

Catalog #125 hook 3 (bit-allocator): ACTIVE — directly consumable by
``tac.optimization.bit_allocator_end_to_end.allocate_per_pair_bits``.
Catalog #125 hook 4 (cathedral autopilot dispatch): ACTIVE — the autopilot
ranker consumes this utility when sweeping per-byte rate budgets via
``evaluate_with_water_filling``.
"""
from __future__ import annotations

import math
from typing import Any

import torch

_EPS = 1e-12


def per_byte_master_gradient_utility(
    archive_bytes: torch.Tensor,
    master_gradient: torch.Tensor,
    duals: Any = None,
) -> torch.Tensor:
    """Sum of per-byte master-gradient utility ``|∂score/∂byte|``.

    Catalog #125 hook 3 (bit-allocator): ACTIVE — directly consumable
    by ``tac.optimization.bit_allocator_end_to_end``.
    Catalog #125 hook 4 (cathedral autopilot dispatch): ACTIVE.

    The utility is the L2 norm of the per-byte master gradient across the
    pair/axis dimensions; high values correspond to "sensitive bytes"
    (sister Catalog #324 framing) and low values correspond to bytes the
    solver can rate-collapse with negligible score impact.

    Args:
        archive_bytes: A 1-D ``torch.Tensor`` ``(n_bytes,)`` carrying
          the archive byte values (typically uint8 cast to float for
          gradient purposes). The function uses only the shape, not the
          values; this keeps the utility byte-content-independent which
          is the canonical Catalog #318 raw-byte-API discipline (the
          master gradient is the authority surface, not the bytes).
        master_gradient: A 2- or 3-D ``torch.Tensor`` carrying per-byte
          gradients. Shape ``(n_bytes, n_axes)`` or
          ``(n_bytes, n_pairs, n_axes)`` per the canonical
          ``MasterGradientBoundarySummary`` format.
        duals: Optional ``DualVariables``. Reads ``duals.lambda_rate``
          as the canonical per-byte weight (defaults to 1.0 when absent).

    Returns:
        A 0-D ``torch.Tensor`` carrying the sum of per-byte L2-norm
        sensitivity values. Has autograd ``grad_fn`` when
        ``master_gradient.requires_grad``.
    """
    if archive_bytes.ndim != 1:
        raise ValueError(
            f"per_byte_master_gradient_utility: archive_bytes must be 1-D; "
            f"got ndim={archive_bytes.ndim} shape={tuple(archive_bytes.shape)}"
        )
    if master_gradient.ndim not in (2, 3):
        raise ValueError(
            f"per_byte_master_gradient_utility: master_gradient must be 2-D "
            f"(n_bytes, n_axes) or 3-D (n_bytes, n_pairs, n_axes); got "
            f"ndim={master_gradient.ndim} shape={tuple(master_gradient.shape)}"
        )
    if master_gradient.shape[0] != archive_bytes.shape[0]:
        raise ValueError(
            f"per_byte_master_gradient_utility: archive_bytes and "
            f"master_gradient must share n_bytes; got "
            f"archive_bytes={archive_bytes.shape[0]} vs "
            f"master_gradient={master_gradient.shape[0]}"
        )

    if master_gradient.ndim == 3:
        n_bytes = master_gradient.shape[0]
        flat = master_gradient.reshape(n_bytes, -1)
    else:
        flat = master_gradient
    # Per-byte L2 norm across remaining dims, then sum.
    per_byte_norms = torch.linalg.vector_norm(flat, dim=1)

    weight = 1.0
    if duals is not None and hasattr(duals, "lambda_rate"):
        weight = float(duals.lambda_rate)
        if not math.isfinite(weight):
            raise ValueError(
                f"per_byte_master_gradient_utility: duals.lambda_rate must "
                f"be finite; got {weight!r}"
            )
    return weight * per_byte_norms.sum()


__all__ = ["per_byte_master_gradient_utility"]
