"""Residual basis primitives for non-HNeRV score-table generation.

This package hosts SCAFFOLD-level (Level 0 SKETCH) residual-basis lanes that
operate on PR106-family decoded outputs (or any contest-shaped (T, H, W, 3) uint8
RGB stream). The lanes are research-signal-only by construction: no scoring,
no exact eval, no dispatch, no archive bytes touched.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13 lessons
and "Forbidden representation-without-archive-grammar (the 'research-substrate
trap')", a residual-basis lane is admissible at L0 if and only if every public
API:

  1. emits `score_claim=False`, `promotion_eligible=False`,
     `ready_for_exact_eval_dispatch=False` in its result manifest,
  2. tags `evidence_grade` as research-signal (NOT `[contest-CUDA]` /
     `[contest-CPU]`),
  3. does NOT load PoseNet/SegNet/scorer weights,
  4. does NOT modify or repack any archive bytes,
  5. carries an EXPLICIT path to L1+ promotion (archive grammar + parser
     manifest + inflate runtime + score-aware loss + no-op proof).

Current public API
------------------

`wavelet_residual_pr106` is the only exported module. It is a research-signal
scaffold, not a score lane: it computes DWT sparsity/entropy statistics over
decoded PR106-family RGB frames and freezes all result manifests to
`score_claim=False`.

See also
--------

`feedback_wavelet_residual_basis_pr106_scaffold_landed_20260511.md` —
the canonical landing memo with the 6-hook wire-in declaration and the
operator-gated path to L1+ promotion.
"""

from __future__ import annotations

from tac.residual_basis.wavelet_residual_pr106 import (
    BandStats,
    WaveletResidualError,
    WaveletResidualResult,
    compute_wavelet_residual_stats,
    decompose_frame_to_bands,
    load_decoded_raw_frames,
    reconstruct_frame_from_bands,
)

__all__ = [
    "BandStats",
    "WaveletResidualError",
    "WaveletResidualResult",
    "compute_wavelet_residual_stats",
    "decompose_frame_to_bands",
    "load_decoded_raw_frames",
    "reconstruct_frame_from_bands",
]
