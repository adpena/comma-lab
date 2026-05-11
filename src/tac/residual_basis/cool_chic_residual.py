"""Cool-Chic residual basis SCAFFOLD over PR106 r2 decoded RGB.

Cool-Chic (Ladune et al., 2023, "Cool-Chic: Coordinate-based Low Complexity
Hierarchical Image Codec") encodes an image as a SMALL hierarchical pyramid
of latent grids that a tiny coordinate-MLP decoder reads. The whole pipeline
is tiny (~10K-100K parameters) and emits a single image as a compressed
artifact.

This module is a research-signal SCAFFOLD per CLAUDE.md HNeRV parity
discipline. It is admissible at L0 if and only if every public API:

  1. emits `score_claim=False`, `promotion_eligible=False`,
     `ready_for_exact_eval_dispatch=False`,
  2. tags `evidence_grade="research_signal"` (NOT contest-CUDA/CPU),
  3. does NOT load PoseNet/SegNet/scorer weights,
  4. does NOT modify or repack any archive bytes,
  5. carries an explicit path to L1+ promotion.

It DOES expose
--------------

* a typed `CoolChicResidualResult` dataclass with promotion-status invariants
  frozen to research-only,
* a `compute_cool_chic_residual_stats()` entry point that accepts a decoded
  RGB array (T, H, W, 3) uint8 and returns per-level pyramid sparsity +
  entropy estimates,
* a `compute_pyramid_residual()` helper that produces a 3-level hierarchical
  residual pyramid (downsample-by-2 at each level via box-mean pooling) over
  the input frames.

It does NOT
-----------

* implement Cool-Chic's coordinate-MLP decoder (research-only; the decoder
  needs an export-first archive grammar + ≤100 LOC inflate budget before
  it can enter the contest packet),
* claim Cool-Chic codec parity (this is a pyramid-residual signal generator,
  not a full Cool-Chic implementation).

Path to L1+ promotion (per HNeRV parity discipline)
---------------------------------------------------

Same 8-archive-grammar-field requirement as the wavelet scaffold:
`archive_grammar`, `parser_section_manifest`, `inflate_runtime_loc_budget`
(≤100 LOC ≤2 ext deps), `runtime_dep_closure`, `export_format`,
`score_aware_loss`, `bolt_on_loc_budget`, `no_op_detector_planned`.

Cool-Chic specifically needs an INT8/FP4-quantized coordinate-MLP weight
packing + an arithmetic-coded latent stream per pyramid level. The natural
binding-time decision is to use PR101's centered-delta-uint8 weight packing +
PR103 arithmetic coding for the latent stream (both already in
`tac.packet_compiler`) as the export contract.

References
----------

Ladune, T., Philippe, P., Hamidouche, W., Henry, F., & Deforges, O. (2023).
"Cool-Chic: Coordinate-based Low Complexity Hierarchical Image Codec." ICCV.

Handoff P3 "Cool-Chic/C3/VQ/SIREN/coordinate MLP" + operator directive
2026-05-11.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np

CAMERA_H: Final[int] = 874
CAMERA_W: Final[int] = 1164
RGB_CHANNELS: Final[int] = 3
DEFAULT_PYRAMID_LEVELS: Final[int] = 3
DEFAULT_SPARSITY_EPSILON: Final[float] = 1e-6
DEFAULT_HIST_BINS: Final[int] = 257
RESEARCH_SIGNAL_EVIDENCE_GRADE: Final[str] = "research_signal"


class CoolChicResidualError(ValueError):
    """Raised on shape / dtype contract violations."""


@dataclass(frozen=True)
class CoolChicPyramidLevelStats:
    """Per-level pyramid sparsity + entropy summary."""

    level: int
    height: int
    width: int
    n_coefficients: int
    abs_mean: float
    abs_std: float
    abs_max: float
    sparsity_fraction: float
    entropy_bits: float

    def assert_invariants(self) -> None:
        if self.n_coefficients <= 0:
            raise CoolChicResidualError(
                f"non-positive n_coefficients={self.n_coefficients}"
            )
        for name, value in (
            ("abs_mean", self.abs_mean),
            ("abs_std", self.abs_std),
            ("abs_max", self.abs_max),
            ("sparsity_fraction", self.sparsity_fraction),
            ("entropy_bits", self.entropy_bits),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise CoolChicResidualError(
                    f"non-finite-or-negative {name}={value} at level {self.level}"
                )
        if self.sparsity_fraction > 1.0:
            raise CoolChicResidualError(
                f"sparsity_fraction={self.sparsity_fraction} > 1.0"
            )


@dataclass(frozen=True)
class CoolChicResidualResult:
    """Result of `compute_cool_chic_residual_stats()`.

    Promotion-status fields are frozen to research-only by construction; no
    callsite may pass `score_claim=True` etc. into this dataclass.
    """

    pyramid_levels: int
    n_frames: int
    n_channels: int
    height: int
    width: int
    per_level_stats: tuple[CoolChicPyramidLevelStats, ...]
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    evidence_grade: str = field(default=RESEARCH_SIGNAL_EVIDENCE_GRADE, init=False)
    schema: str = field(default="cool_chic_residual_pr106_scaffold_v1", init=False)

    def assert_invariants(self) -> None:
        if self.pyramid_levels < 1:
            raise CoolChicResidualError(
                f"pyramid_levels={self.pyramid_levels} must be >= 1"
            )
        if self.n_frames <= 0 or self.n_channels <= 0:
            raise CoolChicResidualError(
                f"non-positive n_frames={self.n_frames} or n_channels={self.n_channels}"
            )
        if not self.per_level_stats:
            raise CoolChicResidualError("per_level_stats must be non-empty")
        for stats in self.per_level_stats:
            stats.assert_invariants()
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise CoolChicResidualError(
                "promotion-status fields must remain False (scaffold-only)"
            )
        if self.evidence_grade != RESEARCH_SIGNAL_EVIDENCE_GRADE:
            raise CoolChicResidualError(
                f"evidence_grade must be {RESEARCH_SIGNAL_EVIDENCE_GRADE!r}"
            )


def _validate_rgb_array(frames: np.ndarray) -> None:
    if frames.ndim != 4:
        raise CoolChicResidualError(f"expected (T, H, W, 3); got ndim={frames.ndim}")
    if frames.shape[3] != RGB_CHANNELS:
        raise CoolChicResidualError(
            f"expected last-axis size 3 (RGB); got {frames.shape[3]}"
        )
    if frames.shape[0] == 0:
        raise CoolChicResidualError("expected n_frames >= 1; got 0")
    if frames.dtype not in (np.uint8, np.float32, np.float64):
        raise CoolChicResidualError(
            f"expected dtype uint8 / float32 / float64; got {frames.dtype}"
        )


def _box_mean_downsample(arr: np.ndarray) -> np.ndarray:
    """2x box-mean downsample on (..., H, W). H, W must be even (crop if not)."""
    h = (arr.shape[-2] // 2) * 2
    w = (arr.shape[-1] // 2) * 2
    cropped = arr[..., :h, :w]
    reshaped = cropped.reshape(*cropped.shape[:-2], h // 2, 2, w // 2, 2)
    return reshaped.mean(axis=(-3, -1))


def compute_pyramid_residual(
    frame: np.ndarray, *, levels: int = DEFAULT_PYRAMID_LEVELS
) -> list[np.ndarray]:
    """Hierarchical box-mean pyramid for a single (H, W, 3) frame.

    Returns a list `[level_0, level_1, ..., level_{L-1}]` where `level_0`
    is the input itself and successive levels are 2x downsamples. This is a
    research-signal-only smoothing pyramid; Cool-Chic's actual coordinate-MLP
    decoder is NOT implemented here.
    """

    if levels < 1:
        raise CoolChicResidualError(f"levels={levels} must be >= 1")
    if frame.ndim != 3 or frame.shape[2] != RGB_CHANNELS:
        raise CoolChicResidualError(f"expected (H, W, 3); got {frame.shape}")
    chw = np.transpose(frame.astype(np.float64, copy=False), (2, 0, 1))
    out: list[np.ndarray] = [chw]
    cur = chw
    for _ in range(levels - 1):
        cur = _box_mean_downsample(cur)
        out.append(cur)
    return out


def _entropy_bits(coefficients: np.ndarray, *, hist_bins: int) -> float:
    """Shannon entropy estimate via histogram over int8-clamped centered coeffs."""
    clamped = np.clip(np.round(coefficients), -128, 128).astype(np.int32)
    counts = np.bincount(clamped.flatten() + 128, minlength=hist_bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(probs * np.log2(probs)))


def _sparsity_fraction(arr: np.ndarray, *, epsilon: float) -> float:
    return float(np.mean(np.abs(arr) < epsilon))


def compute_cool_chic_residual_stats(
    frames: np.ndarray,
    *,
    pyramid_levels: int = DEFAULT_PYRAMID_LEVELS,
    sparsity_epsilon: float = DEFAULT_SPARSITY_EPSILON,
    hist_bins: int = DEFAULT_HIST_BINS,
) -> CoolChicResidualResult:
    """Compute per-level pyramid sparsity + entropy over RGB frames.

    `frames` shape (T, H, W, 3) uint8 / float. Returns a typed
    `CoolChicResidualResult` with promotion-status invariants frozen to
    research-only.
    """

    _validate_rgb_array(frames)
    if pyramid_levels < 1:
        raise CoolChicResidualError(f"pyramid_levels={pyramid_levels} must be >= 1")
    n_frames, h, w, _ = frames.shape

    # Stack per-level coefficient statistics across all frames.
    level_coefficient_buckets: list[list[np.ndarray]] = [[] for _ in range(pyramid_levels)]
    for t in range(n_frames):
        pyramid = compute_pyramid_residual(frames[t], levels=pyramid_levels)
        # For each level beyond 0, the residual = (this level - upsample of next level).
        # For level 0 (full resolution) we keep the frame itself centered around its mean
        # so the entropy reflects detail not absolute pixel values.
        for level_idx, arr in enumerate(pyramid):
            if level_idx == 0:
                centered = arr - arr.mean()
                level_coefficient_buckets[0].append(centered)
            else:
                level_coefficient_buckets[level_idx].append(arr - arr.mean())

    per_level_stats: list[CoolChicPyramidLevelStats] = []
    for level_idx in range(pyramid_levels):
        if not level_coefficient_buckets[level_idx]:
            continue
        stacked = np.stack(level_coefficient_buckets[level_idx])
        abs_vals = np.abs(stacked)
        per_level_stats.append(
            CoolChicPyramidLevelStats(
                level=level_idx,
                height=int(stacked.shape[-2]),
                width=int(stacked.shape[-1]),
                n_coefficients=int(stacked.size),
                abs_mean=float(abs_vals.mean()),
                abs_std=float(abs_vals.std()),
                abs_max=float(abs_vals.max()),
                sparsity_fraction=_sparsity_fraction(
                    stacked, epsilon=sparsity_epsilon
                ),
                entropy_bits=_entropy_bits(stacked, hist_bins=hist_bins),
            )
        )

    result = CoolChicResidualResult(
        pyramid_levels=pyramid_levels,
        n_frames=int(n_frames),
        n_channels=int(RGB_CHANNELS),
        height=int(h),
        width=int(w),
        per_level_stats=tuple(per_level_stats),
    )
    result.assert_invariants()
    return result
