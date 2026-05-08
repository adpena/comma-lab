"""Frame-conditional bit-budget allocator (Track 1 council prescription, Decision 5).

Idea
────
Different frames in the contest video have DIFFERENT contest-distortion
sensitivity. High-motion / textured / occluded frames yield more contest-score
improvement per allocated bit than low-motion / low-texture frames. This
module exposes pure-CPU allocators that distribute a fixed total bit budget
across N frames proportionally to a per-frame complexity proxy.

Two-step pipeline
─────────────────
1. ``compute_per_frame_complexity(video_path, n_frames)`` reads frames from a
   contest-shape mkv via pyav and returns a 1-D numpy array of complexity
   scalars (edge density × pixel variance × frame-to-frame difference). All
   scalars are non-negative. Pure CPU + numpy + pyav; no torch, no GPU.

2. ``allocate_per_frame_bits(complexities, total_bit_budget, eta=1.0,
   floor=0.5, cap=2.0)`` returns the same-shape budget vector that sums to
   ``total_bit_budget`` (within rounding tolerance) where each entry is
   proportional to ``complexities[i]**eta`` clamped between
   ``floor*avg`` and ``cap*avg``.

Mathematics
───────────
Let ``c_i = complexities[i]``, ``w_i = c_i**eta``. Without floor/cap the
proportional allocation is::

    b_i = total * w_i / sum(w_j)

The floor/cap clamps after normalisation, then re-distributes the residual
mass evenly across the *unclamped* frames. We iterate up to a small fixed
number of times so that all clamps are honoured AND ``sum(b_i) == total``
within ±0.5 bits.

Edge cases
──────────
* ``eta=0`` → all weights equal → uniform allocation regardless of
  complexity.
* ``n_frames=1`` → single-element vector containing ``total``.
* identical complexities → uniform regardless of ``eta``.
* zero complexity total (every frame zero) → uniform fallback (prevents
  divide-by-zero; honours floor by symmetry).

CLAUDE.md compliance
────────────────────
* No torch import → no MPS / CUDA path.
* No scorer load → CPU-prep only.
* Output is a budget vector. Score impact requires per-frame score-marginal
  evidence which this module does NOT supply (dispatch_blocker:
  ``awaiting_per_frame_score_marginal``).

Cross-references
────────────────
* ``tac.score_geometry``: closed-form contest score formula. The frame-axis
  marginal is *one of three* (seg / pose / rate); this allocator targets
  the rate axis (charged bytes per frame).
* Memory ``feedback_pr101_frame_conditional_bit_budget_*_20260508.md``: the
  empirical anchor on PR101's monolithic latent stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "ComplexityComponents",
    "compute_per_frame_complexity",
    "allocate_per_frame_bits",
]


@dataclass(frozen=True)
class ComplexityComponents:
    """The three multiplicative factors of the per-frame complexity proxy.

    Each array has length ``n_frames``. The top-level complexity is
    ``edge_density * pixel_variance * frame_difference`` element-wise.
    """

    edge_density: np.ndarray
    pixel_variance: np.ndarray
    frame_difference: np.ndarray

    @property
    def complexity(self) -> np.ndarray:
        return self.edge_density * self.pixel_variance * self.frame_difference


# ─────────────────────────────────────────────────────────────────────────
# Per-frame complexity proxy
# ─────────────────────────────────────────────────────────────────────────


def _luma_from_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """Return BT.601 luma as float32 in [0, 255]."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB array, got shape {rgb.shape}")
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _edge_density(luma: np.ndarray) -> float:
    """Mean absolute gradient magnitude — cheap Sobel-like proxy.

    Uses simple finite differences (no scipy dependency) so this stays in
    pure numpy. The mean absolute gradient is a non-negative scalar; it is
    bounded above by 255 / sqrt(2).
    """
    dx = np.abs(np.diff(luma, axis=1))
    dy = np.abs(np.diff(luma, axis=0))
    return float(0.5 * (dx.mean() + dy.mean()))


def _pixel_variance(luma: np.ndarray) -> float:
    """Per-frame luma variance (single non-negative scalar)."""
    return float(luma.var())


def _frame_difference(luma_curr: np.ndarray, luma_prev: np.ndarray | None) -> float:
    """Mean absolute difference vs previous luma. First frame: 0.0."""
    if luma_prev is None:
        return 0.0
    return float(np.abs(luma_curr - luma_prev).mean())


def compute_per_frame_complexity(
    video_path: str | Path,
    n_frames: int,
    *,
    return_components: bool = False,
) -> np.ndarray | ComplexityComponents:
    """Read up to ``n_frames`` frames and compute a per-frame complexity.

    Parameters
    ----------
    video_path : Path-like
        Path to a contest-shape video readable by pyav (e.g. ``upstream/videos/0.mkv``).
    n_frames : int
        Number of frames to read from the start of the stream.
    return_components : bool
        If True, return the underlying :class:`ComplexityComponents`. Default
        returns the multiplied 1-D complexity array.

    Returns
    -------
    np.ndarray or ComplexityComponents
        Length-``n_frames`` non-negative float64 array, OR the components
        dataclass if ``return_components=True``.

    Notes
    -----
    The first frame's frame-difference is 0.0 (no predecessor). To avoid
    that frame collapsing to zero complexity (and thus being clamped to the
    floor), we replace the first-frame difference with the median of the
    remaining differences before multiplying. This is a conventional
    convention for first-frame motion proxies.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    import av  # local import; keeps this module importable for unit tests

    edge = np.zeros(n_frames, dtype=np.float64)
    var = np.zeros(n_frames, dtype=np.float64)
    diff = np.zeros(n_frames, dtype=np.float64)

    prev_luma: np.ndarray | None = None
    captured = 0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if captured >= n_frames:
                break
            rgb = frame.to_ndarray(format="rgb24")
            luma = _luma_from_rgb_uint8(rgb)
            edge[captured] = _edge_density(luma)
            var[captured] = _pixel_variance(luma)
            diff[captured] = _frame_difference(luma, prev_luma)
            prev_luma = luma
            captured += 1

    if captured < n_frames:
        raise ValueError(
            f"video {video_path} yielded only {captured} frames, requested {n_frames}"
        )

    # Convention: first-frame difference replaced with median of rest so it
    # is not artificially zero (which would multiply the whole row to zero).
    if n_frames > 1:
        diff[0] = float(np.median(diff[1:]))

    components = ComplexityComponents(
        edge_density=edge, pixel_variance=var, frame_difference=diff
    )
    if return_components:
        return components
    return components.complexity


# ─────────────────────────────────────────────────────────────────────────
# Bit budget allocator
# ─────────────────────────────────────────────────────────────────────────


def _clamp_and_redistribute(
    weights: np.ndarray,
    total: float,
    floor: float,
    cap: float,
    *,
    max_iter: int = 20,
    tol: float = 1e-9,
) -> np.ndarray:
    """Distribute ``total`` proportional to ``weights`` then iteratively clamp.

    Algorithm
    ---------
    1. Initial proportional allocation: ``b_i = total * w_i / sum(w_j)``.
    2. Compute uniform mean ``mu = total / n``.
    3. Define ``lo = floor * mu`` and ``hi = cap * mu``.
    4. Clamp every entry to ``[lo, hi]``; track which entries were clamped.
    5. The clamped entries consume a fixed amount; the residual must be
       redistributed across the unclamped entries proportional to their
       (un-normalised) weights.
    6. Repeat until no entry is clamped in an iteration OR ``max_iter`` hit.

    The ``floor`` and ``cap`` are expressed as multiples of the uniform
    average (e.g. ``floor=0.5`` means every frame gets ≥ 50 % of the average
    bit count; ``cap=2.0`` means no frame gets > 200 % of the average).
    """
    n = weights.size
    mu = total / n
    lo = floor * mu
    hi = cap * mu

    if lo > mu or hi < mu:
        # Floor/cap interval excludes the mean — the constraint is
        # infeasible (every frame must equal mu in the limit). Snap to mu.
        return np.full(n, mu, dtype=np.float64)

    # Pre-conditions: floor ≤ 1 ≤ cap and total bits must satisfy
    # n * lo ≤ total ≤ n * hi (always true by construction since
    # n * lo = n * floor * mu = floor * total ≤ total and same for cap).

    # Indices that are NOT yet locked at floor/cap.
    locked_low = np.zeros(n, dtype=bool)
    locked_high = np.zeros(n, dtype=bool)
    allocation = np.zeros(n, dtype=np.float64)

    for _ in range(max_iter):
        active = ~(locked_low | locked_high)
        if not active.any():
            break

        active_weights = weights[active]
        active_weight_sum = float(active_weights.sum())

        # Bits already consumed by locked entries.
        consumed = float(allocation[locked_low].sum() + allocation[locked_high].sum())
        remaining = total - consumed

        if active_weight_sum <= 0.0:
            # Active entries have zero weight: split remaining evenly.
            allocation[active] = remaining / active.sum() if active.sum() else 0.0
        else:
            allocation[active] = remaining * active_weights / active_weight_sum

        # Lock at most ONE side per iteration so the residual mass on the
        # unlocked side can flow to the other side. Locking both sides in
        # one step can shrink the unlocked pool to empty before the
        # residual is redistributed (causing sum != total).
        new_lock_high = (~locked_low) & (~locked_high) & (allocation > hi + tol)
        if new_lock_high.any():
            allocation[new_lock_high] = hi
            locked_high |= new_lock_high
            continue

        new_lock_low = (~locked_low) & (~locked_high) & (allocation < lo - tol)
        if new_lock_low.any():
            allocation[new_lock_low] = lo
            locked_low |= new_lock_low
            continue

        break

    return allocation


def allocate_per_frame_bits(
    complexities: Sequence[float] | np.ndarray,
    total_bit_budget: float,
    *,
    eta: float = 1.0,
    floor: float = 0.5,
    cap: float = 2.0,
) -> np.ndarray:
    """Allocate ``total_bit_budget`` across frames proportional to complexity.

    Parameters
    ----------
    complexities : 1-D array-like of non-negative floats
        Per-frame complexity proxy (e.g. the output of
        :func:`compute_per_frame_complexity`).
    total_bit_budget : float
        Total number of bits to distribute. Must be ≥ 0.
    eta : float, default 1.0
        Concentration exponent. ``eta=0`` collapses to uniform allocation;
        ``eta=1`` is purely proportional; ``eta>1`` concentrates more bits
        on high-complexity frames; ``eta<0`` is permitted but unusual
        (allocates more bits to low-complexity frames). Must be finite.
    floor : float, default 0.5
        Minimum per-frame budget as a multiple of the uniform mean
        ``mu = total_bit_budget / n``. Must satisfy ``0 ≤ floor ≤ 1``.
    cap : float, default 2.0
        Maximum per-frame budget as a multiple of the uniform mean. Must
        satisfy ``cap ≥ 1``.

    Returns
    -------
    np.ndarray, shape (n,), float64
        Per-frame budget. The vector sums to ``total_bit_budget`` within
        ±0.5 (rounding tolerance).
    """
    arr = np.asarray(complexities, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"complexities must be 1-D, got shape {arr.shape}")
    n = arr.size
    if n == 0:
        raise ValueError("complexities must be non-empty")
    if total_bit_budget < 0:
        raise ValueError(f"total_bit_budget must be non-negative, got {total_bit_budget}")
    if (arr < 0).any():
        raise ValueError("complexities must be non-negative")
    if not np.isfinite(eta):
        raise ValueError(f"eta must be finite, got {eta}")
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"floor must be in [0, 1], got {floor}")
    if cap < 1.0:
        raise ValueError(f"cap must be >= 1, got {cap}")

    if n == 1:
        return np.array([float(total_bit_budget)], dtype=np.float64)

    if total_bit_budget == 0.0:
        return np.zeros(n, dtype=np.float64)

    # Compute weights = complexity^eta. eta=0 → uniform regardless of input.
    if eta == 0.0:
        weights = np.ones(n, dtype=np.float64)
    elif arr.sum() == 0.0:
        # All zero complexities → uniform fallback (still honours floor by
        # symmetry — every entry equals mu, which is ≥ lo by definition).
        weights = np.ones(n, dtype=np.float64)
    else:
        # Avoid 0**negative_eta by treating exact zeros as a tiny epsilon
        # relative to the smallest positive weight.
        if eta < 0:
            positive_min = float(arr[arr > 0].min()) if (arr > 0).any() else 1.0
            safe = np.where(arr > 0, arr, positive_min * 1e-6)
            weights = safe ** eta
        else:
            weights = arr ** eta

    return _clamp_and_redistribute(weights, float(total_bit_budget), floor, cap)
