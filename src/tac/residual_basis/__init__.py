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

* `wavelet_residual_pr106` — DWT sparsity/entropy stats (Mallat 1989).
* `numpy_inverse_dwt` — numpy-only Haar inverse-DWT (≤80 LOC; clears the
  wavelet L1 inflate-runtime-dep blocker per
  `feedback_numpy_inverse_dwt_landed_20260511.md`).
* `cool_chic_residual` — hierarchical pyramid signal (Ladune et al. 2023).
* `c3_residual` — conditional residual signal (Kim et al. 2024).
* `siren_residual` — frequency-domain signature for sinusoidal coordinate
  MLPs (Sitzmann et al. 2020).
* `coordinate_mlp_residual` — family-agnostic Laplacian smoothness prior
  (Tancik et al. 2020).

All six modules emit research-signal artifacts with promotion-status frozen
to False; no callsite can promote them past L0 without satisfying the 8
archive-grammar fields per HNeRV parity discipline.

See also
--------

* `feedback_wavelet_residual_basis_pr106_scaffold_landed_20260511.md`
* `feedback_numpy_inverse_dwt_landed_20260511.md`
* `feedback_nonhnerv_residual_basis_scaffolds_landed_20260511.md`
"""

from __future__ import annotations

from tac.residual_basis.c3_residual import (
    C3ConditionalStats,
    C3ResidualError,
    C3ResidualResult,
    compute_c3_residual_stats,
    compute_conditional_residual,
)
from tac.residual_basis.cool_chic_residual import (
    CoolChicPyramidLevelStats,
    CoolChicResidualError,
    CoolChicResidualResult,
    compute_cool_chic_residual_stats,
    compute_pyramid_residual,
)
from tac.residual_basis.coordinate_mlp_residual import (
    CoordinateMlpResidualError,
    CoordinateMlpResidualResult,
    CoordinateMlpSmoothnessStats,
    compute_coordinate_mlp_residual_stats,
    compute_finite_difference_laplacian,
)
from tac.residual_basis.numpy_inverse_dwt import (
    NumpyInverseDWTError,
    haar_inverse_2d_multi_level,
    haar_inverse_2d_single_level,
)
from tac.residual_basis.pr106_sidecar_packing import (
    PR106_RESIDUAL_FORMAT_IDS,
    PR106_RESIDUAL_FORMAT_NAMES,
    PR106_RESIDUAL_MAGIC,
    BuildResidualArchiveResult,
    ParsedResidualArchive,
    ResidualArchiveError,
    build_archive,
    expect_format_id,
    parse_archive,
)
from tac.residual_basis.siren_residual import (
    SirenFrequencyBandStats,
    SirenResidualError,
    SirenResidualResult,
    compute_radial_frequency_buckets,
    compute_siren_residual_stats,
)
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
    "BuildResidualArchiveResult",
    "C3ConditionalStats",
    "C3ResidualError",
    "C3ResidualResult",
    "CoolChicPyramidLevelStats",
    "CoolChicResidualError",
    "CoolChicResidualResult",
    "CoordinateMlpResidualError",
    "CoordinateMlpResidualResult",
    "CoordinateMlpSmoothnessStats",
    "NumpyInverseDWTError",
    "PR106_RESIDUAL_FORMAT_IDS",
    "PR106_RESIDUAL_FORMAT_NAMES",
    "PR106_RESIDUAL_MAGIC",
    "ParsedResidualArchive",
    "ResidualArchiveError",
    "SirenFrequencyBandStats",
    "SirenResidualError",
    "SirenResidualResult",
    "WaveletResidualError",
    "WaveletResidualResult",
    "build_archive",
    "compute_c3_residual_stats",
    "compute_conditional_residual",
    "compute_cool_chic_residual_stats",
    "compute_coordinate_mlp_residual_stats",
    "compute_finite_difference_laplacian",
    "compute_pyramid_residual",
    "compute_radial_frequency_buckets",
    "compute_siren_residual_stats",
    "compute_wavelet_residual_stats",
    "decompose_frame_to_bands",
    "expect_format_id",
    "haar_inverse_2d_multi_level",
    "haar_inverse_2d_single_level",
    "load_decoded_raw_frames",
    "parse_archive",
    "reconstruct_frame_from_bands",
]
