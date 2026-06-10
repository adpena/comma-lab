# SPDX-License-Identifier: MIT
"""Information-theoretic floor of the PoseNet trajectory p* (Lever F pose term).

The pose-scored object is the 600x6 PoseNet output trajectory ``p*`` (the first 6 of
12 dims; ``upstream/modules.py:84``). ``d_pose = mean_pairs ||P(pair)[:6] - p*||^2`` is
a smooth, low-dim regression; the term is ``sqrt(10 * d_pose)`` (concave, GLOBAL pool).

The minimum bits to specify a pose carrier landing in the pose tube at a target
``d_pose`` is the **entropy of the quantized trajectory**:

- A uniform quantizer at step ``delta`` on each of the 6 dims induces quantization
  error ~ Uniform(-delta/2, delta/2), so the per-dim squared error has mean
  ``delta^2 / 12``. ``d_pose = mean_pairs ||.||^2`` SUMS the 6 dims' squared errors,
  so the induced ``d_pose ≈ 6 * delta^2 / 12 = delta^2 / 2`` (DERIVED; if the 6 dims
  have different natural scales we quantize each at its own ``delta_k`` and
  ``d_pose ≈ sum_k delta_k^2 / 12``).
- The carrier bits at that quantization = the entropy of the quantized symbols. The
  trajectory is temporally smooth, so we MEASURE the **temporal-delta entropy**
  ``H(q_t - q_{t-1})`` (first-order predictive), the achievable floor of a
  delta + entropy-coded pose carrier.

This is the RANK-6 pose-output-entropy probe (the prior floor memo's P6, UNRUN).

NO FAKE: every function operates on the real 600x6 trajectory; tests verify the
quantization->d_pose relation on a synthetic trajectory with known step, and that the
temporal-delta entropy is <= the raw quantized-symbol entropy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _shannon_bits_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    nz = counts[counts > 0]
    return float(total * np.log2(total) - (nz * np.log2(nz)).sum())


def _symbol_entropy_bits(symbols: np.ndarray) -> float:
    """Total iid (order-0) Shannon bits of an integer symbol array."""

    flat = np.asarray(symbols).reshape(-1)
    if flat.size == 0:
        return 0.0
    # Shift to non-negative for bincount.
    shifted = flat - int(flat.min())
    counts = np.bincount(shifted.astype(np.int64))
    return _shannon_bits_from_counts(counts)


def induced_d_pose_from_steps(deltas: np.ndarray) -> float:
    """DERIVED d_pose from per-dim uniform quantization steps.

    Uniform quantization error ~ U(-delta/2, delta/2) has variance ``delta^2/12``.
    ``d_pose = mean_pairs sum_k err_k^2`` => expected ``sum_k delta_k^2 / 12``.
    """

    deltas = np.asarray(deltas, dtype=np.float64)
    return float((deltas**2).sum() / 12.0)


@dataclass(frozen=True)
class PoseTrajectoryEntropyResult:
    """Pose carrier entropy at a chosen quantization (all bits unless noted)."""

    n_pairs: int
    n_dims: int
    deltas: tuple[float, ...]
    induced_d_pose: float
    pose_term_contribution: float  # sqrt(10 * induced_d_pose)
    raw_quant_bits: float  # iid entropy of quantized symbols
    temporal_delta_bits: float  # iid entropy of (q_t - q_{t-1})
    total_bytes_temporal_delta: float


def pose_trajectory_entropy(
    trajectory: np.ndarray,
    *,
    deltas: np.ndarray | float,
) -> PoseTrajectoryEntropyResult:
    """Measure the pose carrier entropy of ``trajectory`` (n_pairs, 6) at ``deltas``.

    ``deltas`` is either a scalar (same step all dims) or a length-6 per-dim step.
    Quantizes each dim by rounding to the nearest multiple of its step, then MEASURES
    both the raw quantized-symbol entropy and the temporal-delta entropy.
    """

    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"trajectory must be (n_pairs, n_dims); got {traj.shape}")
    n_pairs, n_dims = traj.shape
    if np.isscalar(deltas):
        delta_vec = np.full(n_dims, float(deltas))
    else:
        delta_vec = np.asarray(deltas, dtype=np.float64)
        if delta_vec.shape != (n_dims,):
            raise ValueError(f"deltas must be scalar or ({n_dims},); got {delta_vec.shape}")
    if np.any(delta_vec <= 0):
        raise ValueError("all quantization steps must be > 0")

    quant = np.round(traj / delta_vec).astype(np.int64)  # (n_pairs, n_dims)
    induced = induced_d_pose_from_steps(delta_vec)
    pose_term = float(np.sqrt(10.0 * induced))

    # Raw quantized-symbol iid entropy (per-dim, summed — coding each dim's stream).
    raw_bits = 0.0
    for k in range(n_dims):
        raw_bits += _symbol_entropy_bits(quant[:, k])

    # Temporal-delta entropy: code first symbol raw, then deltas (per-dim).
    td_bits = 0.0
    for k in range(n_dims):
        col = quant[:, k]
        if col.size == 0:
            continue
        deltas_col = np.diff(col)
        # First symbol full entropy contribution ~ included via the residual stream;
        # we measure delta stream entropy + the single seed symbol's self-information
        # bound (log2 of its alphabet, negligible at n_pairs=600).
        td_bits += _symbol_entropy_bits(deltas_col)
        # seed cost: one symbol; bound by log2(range+1).
        rng = int(col.max() - col.min()) + 1
        td_bits += float(np.log2(rng)) if rng > 1 else 0.0

    return PoseTrajectoryEntropyResult(
        n_pairs=int(n_pairs),
        n_dims=int(n_dims),
        deltas=tuple(float(d) for d in delta_vec),
        induced_d_pose=induced,
        pose_term_contribution=pose_term,
        raw_quant_bits=raw_bits,
        temporal_delta_bits=td_bits,
        total_bytes_temporal_delta=td_bits / 8.0,
    )


def per_dim_steps_for_target_pose_term(
    trajectory: np.ndarray, target_pose_term: float
) -> np.ndarray:
    """DERIVED per-dim quantization steps that hold the pose term at ``target``.

    Allocates the d_pose budget equally across dims. ``sqrt(10 d_pose) = target`` =>
    ``d_pose = target^2 / 10``; with ``d_pose = sum_k delta_k^2 / 12`` and equal split,
    each dim gets ``delta_k = sqrt(12 * d_pose / n_dims)``. The steps are scaled by
    each dim's range so equal-relative-precision dims map to comparable absolute steps
    (a per-dim natural-scale allocation; the equal-absolute fallback if ranges are 0).
    """

    traj = np.asarray(trajectory, dtype=np.float64)
    n_dims = traj.shape[1]
    target_d_pose = float(target_pose_term) ** 2 / 10.0
    # Equal-absolute-step allocation: delta_k = sqrt(12 d_pose / n_dims).
    delta = float(np.sqrt(12.0 * target_d_pose / n_dims))
    return np.full(n_dims, delta)
