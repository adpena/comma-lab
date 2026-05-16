# SPDX-License-Identifier: MIT
"""STC v2 codec — thin substrate-scoped wrapper around the canonical STC kernel.

Per the UNIQUE-AND-COMPLETE-PER-METHOD operating mode the STC v2 substrate
needs its OWN named encode/decode entry points so the substrate package is
self-contained for vendoring per Catalog #295 (the inflate.py runtime cannot
depend on ``tac.*`` imports). This module re-exports the canonical
``tac.stc_boundary_codec`` API surface under stc_v2-scoped names so the
inflate path can vendor a thin self-contained codec module if the operator
elects to seal a submission packet.

Canonical-vs-unique decision (per the 2026-05-16 design memo):

  * The Filler & Pevny 2010 syndrome-trellis coding kernel is UNIQUE FORK at
    the substrate level (STC IS the substrate-class-shift; no canonical
    equivalent in the contest stack).
  * The KERNEL implementation lives at ``tac.stc_boundary_codec`` and is
    shared with other lanes that have historically used STC (PR-alpha mask
    empirical, yucr STC payload). The stc_v2 substrate ADOPTS the kernel
    because re-implementing it would just be duplication.
  * The substrate-scoped wrappers carry stc_v2-specific behavior: tighter
    contracts (always verify roundtrip), explicit return types, and a
    consistent error vocabulary used by the trainer + inflate path.

Strict-scorer-rule compliance: this module imports NO scorer code. All scorer
calls happen in the trainer's compress-time path.
"""
from __future__ import annotations

from pathlib import Path

import torch

from tac.stc_boundary_codec import (
    _STCB_MAGIC as STCB_MAGIC,
    decode_mask_video_stc as _canonical_decode,
    encode_mask_video_stc as _canonical_encode,
)


def encode_stc_v2_masks(
    masks: torch.Tensor,
    output_path: str | Path,
    *,
    boundary_fraction: float = 0.05,
) -> int:
    """Encode a class-id mask tensor as an STCB v1 file (stc_v2-scoped wrapper).

    Args:
        masks: ``(N, H, W)`` integer tensor of class IDs in ``[0, NUM_CLASSES)``.
        output_path: where to write the ``.stcb`` file.
        boundary_fraction: target fraction of pixels per frame marked as
            boundary. Default 0.05 per the 2026-05-16 design memo
            Section 2.2.3.

    Returns:
        Number of bytes written.

    Raises:
        ValueError: shape / dtype / class-range violations.
        RuntimeError: roundtrip mismatch (encoder verifies before returning).
    """
    return _canonical_encode(
        masks,
        Path(output_path),
        boundary_fraction=boundary_fraction,
        per_frame_threshold=True,
        verify_roundtrip=True,
    )


def decode_stc_v2_masks(stcb_path: str | Path) -> torch.Tensor:
    """Decode an STCB v1 file back to ``(N, H, W)`` int64 class IDs."""
    return _canonical_decode(Path(stcb_path))


__all__ = [
    "STCB_MAGIC",
    "decode_stc_v2_masks",
    "encode_stc_v2_masks",
]
