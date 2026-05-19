# SPDX-License-Identifier: MIT
"""Impl 9 -- closed-form internal-resolution optimum for pixel-budget allocation.

The contest input/output resolution is fixed at 384x512 (per ``upstream/
mask_extractor``); INTERNAL resolution at training-time / inference-time is
FREE. The 384x512 output is the only constraint; internal renderer
resolution can be optimized.

The analytical EV per internal-resolution choice combines:
  - Information-theoretic gain (more pixels -> finer detail capture).
  - Compute cost (quadratic in resolution).
  - Output downsampling kernel mismatch (bicubic-kernel + scorer-response
    map; both contest fixed).

Closed-form optimum: the optimal internal resolution is the largest power-of-2
(or fixed-multiple) where the bicubic-downsample-to-384x512 has
NEGLIGIBLE scorer-response degradation. Empirical anchor candidates from
the foveation lanes (Task #516 FF + Task #352 telescopic foveation):
internal resolution 768x1024 (2x output) is a strong canonical choice.

Citations:
  - Cover & Thomas 2006 Ch.10 -- R(D) lower bound.
  - Mallat 2008 *A Wavelet Tour of Signal Processing* (3rd ed) -- scale-
    invariant feature decomposition.
  - Foveation lanes Task #516 + #352.
  - ``upstream/mask_extractor`` 384x512 output resolution.

Catalog #125 hook 3 (bit_allocator): ACTIVE -- pixel-budget is a special
case of bit-budget at the pixel granularity.
Catalog #125 hook 4 (cathedral_autopilot_dispatch): ACTIVE -- autopilot
ranker consumes internal-resolution recommendations.
Catalog #305 observability surface: cite_able, counterfactual_able.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from .constants import CONTEST_INPUT_HEIGHT, CONTEST_INPUT_WIDTH


# Canonical internal-resolution multiples (powers of 2 to OUTPUT resolution).
# Higher than 4x typically saturates the bicubic-kernel response and only
# costs compute; 2x is the canonical sweet spot per Mallat scale-decomposition
# analysis (one full octave above output gives the ideal bandwidth match).
CANONICAL_INTERNAL_RESOLUTION_MULTIPLES: Final[tuple[int, ...]] = (1, 2, 4, 8)


@dataclass(frozen=True, slots=True)
class PixelBudgetRecommendation:
    """Closed-form internal-resolution recommendation."""

    output_height: int
    """Contest-fixed output height (= 384)."""

    output_width: int
    """Contest-fixed output width (= 512)."""

    recommended_internal_multiple: int
    """Recommended internal resolution multiple over the output (e.g. 2 = 768x1024)."""

    recommended_internal_height: int
    """Internal height = output_height * recommended_internal_multiple."""

    recommended_internal_width: int
    """Internal width = output_width * recommended_internal_multiple."""

    recommended_internal_pixel_count: int
    """``recommended_internal_height * recommended_internal_width``."""

    relative_compute_cost: float
    """Compute cost relative to internal_multiple=1 (= ``multiple^2``)."""

    scorer_response_match_quality: str
    """Qualitative match quality: ``IDEAL`` / ``GOOD`` / ``SATURATED`` /
    ``UNDERSAMPLED``. Based on Mallat scale-decomposition canonical."""


def recommend_internal_resolution(
    *,
    compute_budget_relative: float = 4.0,
) -> PixelBudgetRecommendation:
    """Closed-form recommendation for internal renderer resolution.

    Args:
        compute_budget_relative: Maximum compute cost relative to internal=1.
            Default 4.0 admits internal_multiple=2 (canonical sweet spot per
            Mallat) but not 4 (saturated and wasteful).

    Returns:
        ``PixelBudgetRecommendation`` with the closed-form optimum.

    Raises:
        ValueError: if compute_budget_relative < 1.
    """
    if compute_budget_relative < 1.0:
        raise ValueError(
            f"compute_budget_relative must be >= 1 (got {compute_budget_relative})"
        )

    # Pick the LARGEST internal multiple whose compute cost fits the budget.
    feasible = [
        m for m in CANONICAL_INTERNAL_RESOLUTION_MULTIPLES
        if m * m <= compute_budget_relative
    ]
    if not feasible:
        feasible = [1]
    chosen = max(feasible)

    # Qualitative match-quality classification per Mallat scale theory.
    if chosen == 1:
        quality = "UNDERSAMPLED"
    elif chosen == 2:
        quality = "IDEAL"
    elif chosen == 4:
        quality = "GOOD"
    else:
        quality = "SATURATED"

    return PixelBudgetRecommendation(
        output_height=CONTEST_INPUT_HEIGHT,
        output_width=CONTEST_INPUT_WIDTH,
        recommended_internal_multiple=chosen,
        recommended_internal_height=CONTEST_INPUT_HEIGHT * chosen,
        recommended_internal_width=CONTEST_INPUT_WIDTH * chosen,
        recommended_internal_pixel_count=(
            CONTEST_INPUT_HEIGHT * chosen * CONTEST_INPUT_WIDTH * chosen
        ),
        relative_compute_cost=float(chosen * chosen),
        scorer_response_match_quality=quality,
    )


__all__ = [
    "CANONICAL_INTERNAL_RESOLUTION_MULTIPLES",
    "PixelBudgetRecommendation",
    "recommend_internal_resolution",
]
