"""Generic coordinate-MLP residual basis SCAFFOLD over PR106 r2 decoded RGB.

A coordinate-MLP is the GENERIC pattern that subsumes SIREN, NeRV, HNeRV,
Cool-Chic, C3, INR-family compression: each pixel is rendered as `MLP(c)`
where `c` is a coordinate vector (x, y, t) and the MLP weights are the
compressed representation.

This module is a research-signal SCAFFOLD per CLAUDE.md HNeRV parity discipline.
The specific MLP architecture (depth, width, activation) is a downstream
choice — this scaffold provides the FAMILY-AGNOSTIC sparsity / smoothness
prior that ALL coordinate-MLP families share: the SECOND-ORDER FINITE
DIFFERENCE (Laplacian-like) magnitude over each frame.

It is admissible at L0 if and only if every public API:

  1. emits `score_claim=False`, `promotion_eligible=False`,
     `ready_for_exact_eval_dispatch=False`,
  2. tags `evidence_grade="research_signal"`,
  3. does NOT load PoseNet/SegNet/scorer weights,
  4. does NOT modify or repack any archive bytes,
  5. carries an explicit path to L1+ promotion.

It DOES expose
--------------

* a typed `CoordinateMlpResidualResult` dataclass with promotion-status
  invariants frozen to research-only,
* a `compute_coordinate_mlp_residual_stats()` entry point that accepts a
  decoded RGB array (T, H, W, 3) and returns:
    - second-order finite-difference (Laplacian) magnitude statistics
    - per-channel smoothness fraction (low-Laplacian fraction)
* a `compute_finite_difference_laplacian()` helper.

It does NOT
-----------

* implement a specific coordinate-MLP architecture (research-only),
* claim parity with SIREN / NeRV / HNeRV / Cool-Chic / C3 / any specific
  coordinate-MLP family (this is the FAMILY-AGNOSTIC scaffold).

Path to L1+ promotion
---------------------

Each specific coordinate-MLP family (SIREN, NeRV, HNeRV, Cool-Chic, C3, …)
needs its own L1+ scaffold with its specific architecture + archive grammar.
This module is the SHARED prior, not a single family's codec.

References
----------

Tancik, M., Srinivasan, P. P., Mildenhall, B., et al. (2020). "Fourier Features
Let Networks Learn High Frequency Functions in Low Dimensional Domains." NeurIPS.

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
DEFAULT_SMOOTHNESS_EPSILON: Final[float] = 1.0  # uint8 scale: 1 LSB
RESEARCH_SIGNAL_EVIDENCE_GRADE: Final[str] = "research_signal"


class CoordinateMlpResidualError(ValueError):
    """Raised on shape / dtype contract violations."""


@dataclass(frozen=True)
class CoordinateMlpSmoothnessStats:
    """Per-frame Laplacian-magnitude smoothness summary."""

    n_coefficients: int
    abs_mean: float
    abs_std: float
    abs_max: float
    smoothness_fraction: float  # |Laplacian| < epsilon
    energy_log_mean: float  # mean log(|Laplacian|^2 + 1)

    def assert_invariants(self) -> None:
        if self.n_coefficients <= 0:
            raise CoordinateMlpResidualError(
                f"non-positive n_coefficients={self.n_coefficients}"
            )
        for name, value in (
            ("abs_mean", self.abs_mean),
            ("abs_std", self.abs_std),
            ("abs_max", self.abs_max),
            ("smoothness_fraction", self.smoothness_fraction),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise CoordinateMlpResidualError(
                    f"non-finite-or-negative {name}={value}"
                )
        if self.smoothness_fraction > 1.0:
            raise CoordinateMlpResidualError(
                f"smoothness_fraction={self.smoothness_fraction} > 1.0"
            )
        if not math.isfinite(self.energy_log_mean):
            raise CoordinateMlpResidualError("non-finite energy_log_mean")


@dataclass(frozen=True)
class CoordinateMlpResidualResult:
    """Result of `compute_coordinate_mlp_residual_stats()`. Promotion frozen."""

    n_frames: int
    n_channels: int
    height: int
    width: int
    stats: CoordinateMlpSmoothnessStats
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    evidence_grade: str = field(default=RESEARCH_SIGNAL_EVIDENCE_GRADE, init=False)
    schema: str = field(default="coordinate_mlp_residual_pr106_scaffold_v1", init=False)

    def assert_invariants(self) -> None:
        if self.n_frames <= 0 or self.n_channels <= 0:
            raise CoordinateMlpResidualError(
                f"non-positive n_frames={self.n_frames} or n_channels={self.n_channels}"
            )
        if self.height <= 0 or self.width <= 0:
            raise CoordinateMlpResidualError(
                f"non-positive h={self.height} w={self.width}"
            )
        self.stats.assert_invariants()
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise CoordinateMlpResidualError(
                "promotion-status fields must remain False (scaffold-only)"
            )
        if self.evidence_grade != RESEARCH_SIGNAL_EVIDENCE_GRADE:
            raise CoordinateMlpResidualError(
                f"evidence_grade must be {RESEARCH_SIGNAL_EVIDENCE_GRADE!r}"
            )


def _validate_rgb_array(frames: np.ndarray) -> None:
    if frames.ndim != 4:
        raise CoordinateMlpResidualError(
            f"expected (T, H, W, 3); got ndim={frames.ndim}"
        )
    if frames.shape[3] != RGB_CHANNELS:
        raise CoordinateMlpResidualError(
            f"expected last-axis size 3 (RGB); got {frames.shape[3]}"
        )
    if frames.shape[0] == 0:
        raise CoordinateMlpResidualError("expected n_frames >= 1; got 0")
    if frames.dtype not in (np.uint8, np.float32, np.float64):
        raise CoordinateMlpResidualError(
            f"expected dtype uint8 / float32 / float64; got {frames.dtype}"
        )


def compute_finite_difference_laplacian(frame: np.ndarray) -> np.ndarray:
    """Compute the 4-neighbor finite-difference Laplacian of a (H, W, 3) frame.

    Returns a (3, H, W) float64 array. The interior pixels carry the standard
    5-point stencil `L = f(x-1, y) + f(x+1, y) + f(x, y-1) + f(x, y+1) - 4*f(x, y)`;
    the boundary is replicated (Neumann boundary).
    """

    if frame.ndim != 3 or frame.shape[2] != RGB_CHANNELS:
        raise CoordinateMlpResidualError(f"expected (H, W, 3); got {frame.shape}")
    f = frame.astype(np.float64, copy=False)
    chw = np.transpose(f, (2, 0, 1))  # (3, H, W)
    lap = np.zeros_like(chw)
    # Interior: 5-point stencil.
    lap[:, 1:-1, 1:-1] = (
        chw[:, :-2, 1:-1]
        + chw[:, 2:, 1:-1]
        + chw[:, 1:-1, :-2]
        + chw[:, 1:-1, 2:]
        - 4.0 * chw[:, 1:-1, 1:-1]
    )
    # Edges: forward/backward differences.
    lap[:, 0, :] = lap[:, 1, :]
    lap[:, -1, :] = lap[:, -2, :]
    lap[:, :, 0] = lap[:, :, 1]
    lap[:, :, -1] = lap[:, :, -2]
    return lap


def compute_coordinate_mlp_residual_stats(
    frames: np.ndarray, *, smoothness_epsilon: float = DEFAULT_SMOOTHNESS_EPSILON
) -> CoordinateMlpResidualResult:
    """Compute Laplacian-magnitude smoothness statistics over RGB frames.

    `frames` shape (T, H, W, 3). Returns a typed `CoordinateMlpResidualResult`
    with promotion-status invariants frozen to research-only.
    """

    _validate_rgb_array(frames)
    n_frames, h, w, _ = frames.shape

    all_laplacians: list[np.ndarray] = []
    for t in range(n_frames):
        lap = compute_finite_difference_laplacian(frames[t])
        all_laplacians.append(lap)
    stacked = np.stack(all_laplacians)  # (T, 3, H, W)
    abs_vals = np.abs(stacked)
    smoothness_fraction = float(np.mean(abs_vals < smoothness_epsilon))
    energy_log_mean = float(np.mean(np.log(stacked**2 + 1.0)))

    stats = CoordinateMlpSmoothnessStats(
        n_coefficients=int(stacked.size),
        abs_mean=float(abs_vals.mean()),
        abs_std=float(abs_vals.std()),
        abs_max=float(abs_vals.max()),
        smoothness_fraction=smoothness_fraction,
        energy_log_mean=energy_log_mean,
    )
    result = CoordinateMlpResidualResult(
        n_frames=int(n_frames),
        n_channels=int(RGB_CHANNELS),
        height=int(h),
        width=int(w),
        stats=stats,
    )
    result.assert_invariants()
    return result
