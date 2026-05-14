# SPDX-License-Identifier: MIT
"""C3 (Compressed Conditional Content) residual basis SCAFFOLD over PR106 r2 decoded RGB.

C3 (Kim et al., 2024, "C3: High-performance and low-complexity neural compression
from a single image") generalizes Cool-Chic by adding a CONDITIONING input
(per-image hyperprior) plus a learned context model for the latent grid bits.
The conditioning is what lets a tiny coordinate-MLP decoder produce
sharp images from very small latent grids.

This module is a research-signal SCAFFOLD per CLAUDE.md HNeRV parity discipline.
It is admissible at L0 if and only if every public API:

  1. emits `score_claim=False`, `promotion_eligible=False`,
     `ready_for_exact_eval_dispatch=False`,
  2. tags `evidence_grade="research_signal"`,
  3. does NOT load PoseNet/SegNet/scorer weights,
  4. does NOT modify or repack any archive bytes,
  5. carries an explicit path to L1+ promotion.

It DOES expose
--------------

* a typed `C3ResidualResult` dataclass with promotion-status invariants
  frozen to research-only,
* a `compute_c3_residual_stats()` entry point that accepts a decoded
  RGB array (T, H, W, 3) and an OPTIONAL per-frame conditioning signal
  (per-pair pose offset, per-pair flow field summary, etc.) and returns
  conditional sparsity + entropy estimates,
* a `compute_conditional_residual()` helper that estimates per-frame
  residual conditioned on the mean-frame pair (frame[t] - frame[t-1]) —
  the simplest possible conditioning signal.

It does NOT
-----------

* implement C3's actual coordinate-MLP decoder + context model (research-only),
* claim C3 codec parity.

Path to L1+ promotion
---------------------

Same 8-archive-grammar-field requirement. C3 specifically needs an
arithmetic-coded latent stream + a small context-model NN. The natural binding
is to use PR91's HPACMini-style context model + PR103 arithmetic coding for
the latent stream.

References
----------

Kim, H., Bauer, M., Theis, L., Schwarz, J. R., & Dupont, E. (2024). "C3: High-
performance and low-complexity neural compression from a single image." CVPR.

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
DEFAULT_SPARSITY_EPSILON: Final[float] = 1e-6
DEFAULT_HIST_BINS: Final[int] = 257
RESEARCH_SIGNAL_EVIDENCE_GRADE: Final[str] = "research_signal"


class C3ResidualError(ValueError):
    """Raised on shape / dtype contract violations."""


@dataclass(frozen=True)
class C3ConditionalStats:
    """Per-frame conditional residual sparsity + entropy summary."""

    conditioning_mode: str  # "frame_delta" | "mean_baseline"
    n_coefficients: int
    abs_mean: float
    abs_std: float
    abs_max: float
    sparsity_fraction: float
    entropy_bits: float

    def assert_invariants(self) -> None:
        if self.n_coefficients <= 0:
            raise C3ResidualError(f"non-positive n_coefficients={self.n_coefficients}")
        if self.conditioning_mode not in ("frame_delta", "mean_baseline"):
            raise C3ResidualError(
                f"unknown conditioning_mode={self.conditioning_mode!r}"
            )
        for name, value in (
            ("abs_mean", self.abs_mean),
            ("abs_std", self.abs_std),
            ("abs_max", self.abs_max),
            ("sparsity_fraction", self.sparsity_fraction),
            ("entropy_bits", self.entropy_bits),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise C3ResidualError(f"non-finite-or-negative {name}={value}")
        if self.sparsity_fraction > 1.0:
            raise C3ResidualError(
                f"sparsity_fraction={self.sparsity_fraction} > 1.0"
            )


@dataclass(frozen=True)
class C3ResidualResult:
    """Result of `compute_c3_residual_stats()`. Promotion-status frozen."""

    n_frames: int
    n_channels: int
    height: int
    width: int
    stats: C3ConditionalStats
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    evidence_grade: str = field(default=RESEARCH_SIGNAL_EVIDENCE_GRADE, init=False)
    schema: str = field(default="c3_residual_pr106_scaffold_v1", init=False)

    def assert_invariants(self) -> None:
        if self.n_frames <= 0 or self.n_channels <= 0:
            raise C3ResidualError(
                f"non-positive n_frames={self.n_frames} or n_channels={self.n_channels}"
            )
        if self.height <= 0 or self.width <= 0:
            raise C3ResidualError(
                f"non-positive shape h={self.height} w={self.width}"
            )
        self.stats.assert_invariants()
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise C3ResidualError(
                "promotion-status fields must remain False (scaffold-only)"
            )
        if self.evidence_grade != RESEARCH_SIGNAL_EVIDENCE_GRADE:
            raise C3ResidualError(
                f"evidence_grade must be {RESEARCH_SIGNAL_EVIDENCE_GRADE!r}"
            )


def _validate_rgb_array(frames: np.ndarray) -> None:
    if frames.ndim != 4:
        raise C3ResidualError(f"expected (T, H, W, 3); got ndim={frames.ndim}")
    if frames.shape[3] != RGB_CHANNELS:
        raise C3ResidualError(
            f"expected last-axis size 3 (RGB); got {frames.shape[3]}"
        )
    if frames.shape[0] == 0:
        raise C3ResidualError("expected n_frames >= 1; got 0")
    if frames.dtype not in (np.uint8, np.float32, np.float64):
        raise C3ResidualError(
            f"expected dtype uint8 / float32 / float64; got {frames.dtype}"
        )


def compute_conditional_residual(
    frames: np.ndarray, *, conditioning_mode: str = "frame_delta"
) -> np.ndarray:
    """Compute per-frame residual conditioned on a baseline.

    `conditioning_mode="frame_delta"` returns `frames[1:] - frames[:-1]`
    (the per-pair frame delta — the simplest conditioning signal that
    captures motion content).

    `conditioning_mode="mean_baseline"` returns `frames - frames.mean(axis=0)`
    (each frame minus the temporal-mean frame).

    Both modes preserve the float scale of the original frames; downstream
    quantization/entropy is the consumer's responsibility.
    """

    _validate_rgb_array(frames)
    arr = frames.astype(np.float64, copy=False)
    if conditioning_mode == "frame_delta":
        if arr.shape[0] < 2:
            raise C3ResidualError("frame_delta requires >= 2 frames")
        return arr[1:] - arr[:-1]
    if conditioning_mode == "mean_baseline":
        return arr - arr.mean(axis=0, keepdims=True)
    raise C3ResidualError(f"unknown conditioning_mode={conditioning_mode!r}")


def _entropy_bits(arr: np.ndarray, *, hist_bins: int) -> float:
    clamped = np.clip(np.round(arr), -128, 128).astype(np.int32)
    counts = np.bincount(clamped.flatten() + 128, minlength=hist_bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(probs * np.log2(probs)))


def _sparsity_fraction(arr: np.ndarray, *, epsilon: float) -> float:
    return float(np.mean(np.abs(arr) < epsilon))


def compute_c3_residual_stats(
    frames: np.ndarray,
    *,
    conditioning_mode: str = "frame_delta",
    sparsity_epsilon: float = DEFAULT_SPARSITY_EPSILON,
    hist_bins: int = DEFAULT_HIST_BINS,
) -> C3ResidualResult:
    """Compute conditional residual stats over RGB frames.

    `frames` shape (T, H, W, 3). Returns a typed `C3ResidualResult` with
    promotion-status invariants frozen to research-only.
    """

    _validate_rgb_array(frames)
    n_frames, h, w, _ = frames.shape
    residual = compute_conditional_residual(frames, conditioning_mode=conditioning_mode)
    abs_vals = np.abs(residual)
    stats = C3ConditionalStats(
        conditioning_mode=conditioning_mode,
        n_coefficients=int(residual.size),
        abs_mean=float(abs_vals.mean()),
        abs_std=float(abs_vals.std()),
        abs_max=float(abs_vals.max()),
        sparsity_fraction=_sparsity_fraction(residual, epsilon=sparsity_epsilon),
        entropy_bits=_entropy_bits(residual, hist_bins=hist_bins),
    )
    result = C3ResidualResult(
        n_frames=int(n_frames),
        n_channels=int(RGB_CHANNELS),
        height=int(h),
        width=int(w),
        stats=stats,
    )
    result.assert_invariants()
    return result
