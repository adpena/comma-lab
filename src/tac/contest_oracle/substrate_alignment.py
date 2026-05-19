# SPDX-License-Identifier: MIT
"""Impl 12 -- substrate-shape contest-alignment criterion (NEW; genuinely missing).

Derives a closed-form alignment score for each canonical substrate
architecture vs the contest's scoring shape. Per the design memo:
"NeRV mis-aligned; HNeRV partial; per-pair+per-class+per-pixel fully aligned".

The alignment criterion captures HOW WELL a substrate's output structure
matches the contest scoring scaffold (5-class SegNet + 6-dim PoseNet +
600-pair + 384x512 spatial). A fully-aligned substrate has minimal
representational waste; a mis-aligned substrate trains parameters that
the contest scorer never sees (CLAUDE.md ``HNeRV / leaderboard-implementation
parity discipline`` L5).

Citations:
  - CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L5 --
    architecture must be the FULL renderer (RGB out), not single-component slot.
  - CLAUDE.md "Forbidden representation-without-archive-grammar" 8th forbidden
    pattern -- mis-aligned substrates are the research-substrate trap.
  - PR101 GOLD canonical -- per-class FiLM + 600-pair structure = fully aligned.

Catalog #125 hook 4 (cathedral_autopilot_dispatch): ACTIVE -- alignment
score IS a canonical re-ranking signal for autopilot.
Catalog #305 observability surface: cite_able, decomposable_per_signal,
counterfactual_able.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .constants import (
    CONTEST_INPUT_HEIGHT,
    CONTEST_INPUT_WIDTH,
    CONTEST_NUM_PAIRS,
    SEGNET_NUM_CLASSES,
)


class AlignmentFacet(StrEnum):
    """Canonical alignment facets vs the contest scoring scaffold."""

    FULL_RENDERER_RGB_OUT = "full_renderer_rgb_out"
    """Substrate outputs full RGB frames (vs single-component slot like masks-only)."""

    PER_PAIR_STRUCTURE = "per_pair_structure"
    """Substrate respects 600-pair decomposition (vs whole-video monolithic)."""

    PER_CLASS_STRUCTURE = "per_class_structure"
    """Substrate respects 5-class SegNet decomposition (vs class-agnostic)."""

    SPATIAL_RESOLUTION_MATCH = "spatial_resolution_match"
    """Substrate output is 384x512 (or canonical multiple thereof)."""

    POSE_AXIS_AWARE = "pose_axis_aware"
    """Substrate's training objective routes through PoseNet gradients
    (vs pose-axis blind)."""

    EVAL_ROUNDTRIP_AWARE = "eval_roundtrip_aware"
    """Substrate trains under eval_roundtrip=True per CLAUDE.md non-negotiable."""

    BYTE_DETERMINISTIC = "byte_deterministic"
    """Substrate produces byte-stable archive bytes given seed + config."""

    SCORER_PREPROCESS_DIFFERENTIABLE = "scorer_preprocess_differentiable"
    """Substrate uses differentiable rgb_to_yuv6 + scorer preprocess (Catalog #164)."""


@dataclass(frozen=True, slots=True)
class SubstrateAlignmentScore:
    """Substrate alignment audit result."""

    substrate_id: str
    """Canonical substrate identifier."""

    aligned_facets: tuple[AlignmentFacet, ...]
    """Tuple of facets the substrate satisfies."""

    misaligned_facets: tuple[AlignmentFacet, ...]
    """Tuple of facets the substrate does NOT satisfy."""

    alignment_score: float
    """``aligned / total`` in [0, 1]. 1.0 = fully aligned."""

    verdict: str
    """Qualitative verdict: ``FULLY_ALIGNED`` / ``MOSTLY_ALIGNED`` /
    ``PARTIALLY_ALIGNED`` / ``MIS_ALIGNED``."""


# Canonical complete facet set
_ALL_FACETS: Final[tuple[AlignmentFacet, ...]] = tuple(AlignmentFacet)


def score_substrate_alignment(
    *,
    substrate_id: str,
    aligned_facets: Sequence[AlignmentFacet],
) -> SubstrateAlignmentScore:
    """Closed-form substrate alignment audit.

    Args:
        substrate_id: Canonical substrate identifier (e.g. "pr101_lc_v2",
            "nscs01_nullspace_split_renderer").
        aligned_facets: Sequence of facets the substrate satisfies.

    Returns:
        ``SubstrateAlignmentScore`` with the full per-facet decomposition.

    Raises:
        ValueError: if substrate_id is empty.
    """
    if not substrate_id:
        raise ValueError("substrate_id must be non-empty")

    aligned_set = set(aligned_facets)
    aligned_tuple = tuple(f for f in _ALL_FACETS if f in aligned_set)
    misaligned_tuple = tuple(f for f in _ALL_FACETS if f not in aligned_set)

    alignment_score = len(aligned_tuple) / len(_ALL_FACETS)

    if alignment_score >= 0.875:  # >=7 of 8
        verdict = "FULLY_ALIGNED"
    elif alignment_score >= 0.625:  # >=5 of 8
        verdict = "MOSTLY_ALIGNED"
    elif alignment_score >= 0.375:  # >=3 of 8
        verdict = "PARTIALLY_ALIGNED"
    else:
        verdict = "MIS_ALIGNED"

    return SubstrateAlignmentScore(
        substrate_id=substrate_id,
        aligned_facets=aligned_tuple,
        misaligned_facets=misaligned_tuple,
        alignment_score=float(alignment_score),
        verdict=verdict,
    )


# Canonical PR101 GOLD reference: 7 of 8 facets aligned (per design memo).
PR101_GOLD_ALIGNED_FACETS: Final[tuple[AlignmentFacet, ...]] = (
    AlignmentFacet.FULL_RENDERER_RGB_OUT,
    AlignmentFacet.PER_PAIR_STRUCTURE,
    AlignmentFacet.PER_CLASS_STRUCTURE,
    AlignmentFacet.SPATIAL_RESOLUTION_MATCH,
    AlignmentFacet.POSE_AXIS_AWARE,
    AlignmentFacet.EVAL_ROUNDTRIP_AWARE,
    AlignmentFacet.BYTE_DETERMINISTIC,
    # PR101 lacks SCORER_PREPROCESS_DIFFERENTIABLE per the original PR101
    # implementation; later PR #95/#106 added the differentiable rgb_to_yuv6.
)


def pr101_gold_reference() -> SubstrateAlignmentScore:
    """Convenience: PR101 GOLD canonical reference alignment score (= 0.875)."""
    return score_substrate_alignment(
        substrate_id="pr101_gold_canonical_reference",
        aligned_facets=PR101_GOLD_ALIGNED_FACETS,
    )


__all__ = [
    "AlignmentFacet",
    "PR101_GOLD_ALIGNED_FACETS",
    "SubstrateAlignmentScore",
    "pr101_gold_reference",
    "score_substrate_alignment",
]
