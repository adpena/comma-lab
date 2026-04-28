"""Lane EC-V2: Greedy Pareto-frontier correction selector.

Lane EC-V2 (Pareto-dominant variant) replaces the V1 fixed-budget top-K
percentile policy with a greedy water-fill that ranks (frame, pixel)
candidates by ``score_gain / byte_cost`` and keeps adding deltas until
the byte budget is hit. Predicted band [contest-CUDA]: [0.75, 1.00].

Design (council Yousfi+Quantizr+Hotz, 2026-04-28):

  - Each candidate cell is a single (flat-index) location whose absolute
    gradient magnitude approximates the score gain from flipping it
    ±127 (int8 quant). Byte cost is ``byte_per_pixel`` (default 1)
    plus a small fixed overhead amortized across the archive.
  - Selection is greedy by descending ``gain/byte``. Ties broken by
    flat-index (deterministic — see test_tie_breaking_is_deterministic_by_flat_index).
  - The byte cap is enforced inclusive: a candidate that would push us
    PAST the cap is rejected; the loop terminates as soon as the next
    candidate violates the bound.

The heavy implementation lives in
``experiments/precompute_gradient_corrections.greedy_waterfill_correction_map``
(called by both EC and EC-V2). This module exposes the V2 selection
helpers in a tac-side API so the orchestration code (deploy script,
trick_stack, V2 ablations) can import them via::

    from tac.engineered_corrections_v2 import (
        rank_correction_candidates,
        select_top_k_per_frame,
    )

The V2 greedy regression suite (src/tac/tests/test_lane_ec_v2_greedy.py)
runs against the canonical impl. This module is a thin wrapper that
adds the candidate-ranking and per-frame-budget helpers Lane EC-V2 ships.
"""
from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from experiments.precompute_gradient_corrections import (  # noqa: E402
    estimated_sparse_bytes,
    greedy_waterfill_correction_map,
    sparsify_and_quantize,
)

__all__ = [
    "rank_correction_candidates",
    "select_top_k_per_frame",
    "select_pareto_frontier",
    # Re-exports kept here so callers don't need a second import line.
    "greedy_waterfill_correction_map",
    "sparsify_and_quantize",
    "estimated_sparse_bytes",
]


def rank_correction_candidates(
    gradients: np.ndarray,
    *,
    score_sensitivity: np.ndarray | None = None,
    byte_per_pixel: float = 1.0,
) -> list[tuple[int, int, float]]:
    """Rank (frame, flat_pixel_index) candidates by descending
    ``gain/byte`` ratio.

    Args:
        gradients: ``(N_frames, H, W, C)`` float32 gradient tensor. Per
            cell the gain is ``sqrt(sum(grad**2))`` (L2 magnitude).
        score_sensitivity: optional ``(N_frames, H, W)`` per-cell weight.
            When supplied, the gain is ``abs(score_sensitivity)`` instead
            of the gradient L2 — useful for SegNet which uses
            argmax-margin deltas rather than raw gradients.
        byte_per_pixel: bytes consumed per kept cell (V2 packing model).

    Returns:
        Sorted list of ``(frame_idx, flat_pixel_idx_in_frame, ratio)``
        tuples, descending by ratio. Tie-broken deterministically by
        ``(frame_idx, flat_pixel_idx_in_frame)``.
    """
    grads = np.asarray(gradients)
    if grads.ndim != 4:
        raise ValueError(f"gradients must have shape (N,H,W,C), got {grads.shape}")
    if byte_per_pixel <= 0:
        raise ValueError("byte_per_pixel must be positive")

    N, H, W, C = grads.shape
    pixels_per_frame = H * W

    if score_sensitivity is None:
        gains = np.sqrt((grads.astype(np.float32) ** 2).sum(axis=-1))
    else:
        gains = np.abs(np.asarray(score_sensitivity, dtype=np.float32))
        if gains.shape != (N, H, W):
            raise ValueError(
                f"score_sensitivity must have shape ({N},{H},{W}), got {gains.shape}"
            )

    # heapq is a min-heap; negate ratio for max-heap behaviour. Include
    # (frame_idx, pixel_idx) as the tie-breaker so identical-ratio cells
    # still yield a deterministic order across reruns.
    heap: list[tuple[float, int, int, float]] = []
    flat_gains = gains.reshape(N, pixels_per_frame)
    for frame_idx in range(N):
        for pix_idx in range(pixels_per_frame):
            gain = float(flat_gains[frame_idx, pix_idx])
            if not math.isfinite(gain) or gain <= 0.0:
                continue
            ratio = gain / float(byte_per_pixel)
            heapq.heappush(heap, (-ratio, frame_idx, pix_idx, gain))

    ranked: list[tuple[int, int, float]] = []
    while heap:
        neg_ratio, frame_idx, pix_idx, _gain = heapq.heappop(heap)
        ranked.append((frame_idx, pix_idx, -neg_ratio))
    return ranked


def select_top_k_per_frame(
    ranked: list[tuple[int, int, float]],
    byte_budget: int,
    *,
    byte_per_pixel: float = 1.0,
    byte_overhead: int = 100,
    max_per_frame: int | None = None,
) -> dict[int, list[int]]:
    """Select the subset of ranked candidates that fits in
    ``byte_budget`` total bytes, optionally enforcing a per-frame cap.

    The cap defends against pathological gradients where a single frame
    consumes the entire budget. When ``max_per_frame`` is None there is
    no per-frame limit (pure global greedy).

    Args:
        ranked: Output of :func:`rank_correction_candidates`.
        byte_budget: Total byte cap (matches the deploy script's
            ``--max-artifact-bytes``).
        byte_per_pixel: Byte cost per kept cell.
        byte_overhead: Fixed overhead deducted before greedy selection
            starts (matches the V2 packing model).
        max_per_frame: Optional per-frame cell cap.

    Returns:
        Mapping ``frame_idx -> [flat_pixel_idx, ...]`` of selected cells.
    """
    if byte_per_pixel <= 0:
        raise ValueError("byte_per_pixel must be positive")
    if byte_budget < 0:
        raise ValueError("byte_budget must be non-negative")

    used_bytes = int(byte_overhead)
    selected: dict[int, list[int]] = {}
    per_frame_count: dict[int, int] = {}

    for frame_idx, pix_idx, _ratio in ranked:
        if max_per_frame is not None and per_frame_count.get(frame_idx, 0) >= max_per_frame:
            continue
        next_bytes = used_bytes + int(byte_per_pixel)
        if next_bytes > byte_budget:
            break
        used_bytes = next_bytes
        selected.setdefault(frame_idx, []).append(pix_idx)
        per_frame_count[frame_idx] = per_frame_count.get(frame_idx, 0) + 1
    return selected


def select_pareto_frontier(
    gradients: np.ndarray,
    *,
    rate_cap_bytes: int,
    margin_deltas: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the Pareto-frontier sparse correction map for Lane EC-V2.

    This is a one-call orchestrator: rank candidates, select within the
    budget, and return the same sparse-correction dict
    :func:`tac.engineered_corrections.compute_per_frame_corrections`
    returns. Callers get an end-to-end EC-V2 pipeline without having to
    stitch the helpers together.

    The implementation delegates to
    :func:`greedy_waterfill_correction_map` so V1 vs V2 stay byte-equal
    (the V2 greedy IS the canonical fast-arithmetic-model selection).
    """
    return sparsify_and_quantize(
        np.asarray(gradients),
        allocation_strategy="greedy",
        rate_cap_bytes=rate_cap_bytes,
        # The greedy path ignores top_k_pct + quantize_bits is held at 8
        # to match the deploy script's default.
        top_k_pct=0.0,
        quantize_bits=8,
    )
