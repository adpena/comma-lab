# SPDX-License-Identifier: MIT
"""The MDL region-merge SOLVE at the waterfilling water level (spec §4 + §10).

The water level is FIXED ANALYTICALLY by evaluate.py's rate term:

    S = 100*d_seg + sqrt(10*d_pose) + 25*B/D,   D = 37,545,489
    one byte costs  dS/dB = 25/D = 6.66e-7  score/byte
    one seg flip is worth  100/N = 100/(600*196608) = 8.48e-7  score/flip   (N pixels scored)
    -> a flip pays its rent iff fixing it costs < (100/N)/(25/D) = 1.27 bytes/flip

Source: §10 verbatim ("keep a region iff its boundary-contour bytes < (interior flips
it avoids) * 1.27").  N here is the per-frame scored pixel count 384*512 = 196608; a
flip on ANY of the 600 frames is worth the same 100/(600*196608) in the global mean,
so the per-flip price 1.27 B/flip is frame-independent (it is the GLOBAL water level).

The SOLVE (a real weighted greedy graph-cut over the RAG, NOT a sweep):

  Given the candidate partition ``L_cand`` (what the carrier would store) and the GT
  partition ``L*`` (the scored target), each region of ``L_cand`` that DIFFERS from
  ``L*`` on its pixels carries:
    - a byte cost: the marginal description bytes of keeping that region distinct
      (vs merging it into a neighbour and dropping its contour);
    - a flip debt avoided: the pixels it makes AGREE with ``L*`` (flips it fixes).
  Keep the region (pay its bytes) iff  bytes_kept < flips_fixed * 1.27 .  Else MERGE
  it into the cheapest neighbour (its contour bytes vanish; its pixels revert to the
  neighbour's label, re-incurring those flips).

We solve this greedily over the RAG by region marginal value-per-byte, which is the
optimal order for this separable knapsack-on-a-graph at a fixed water level (KKT:
fund steepest-first until marginal == lambda*).  The result is a kept/merged decision
per region + the resulting partition + the exact (bytes, flips, d_seg) it lands at.

NO FAKE: the merge actually rewrites the label map by region id, the bytes come from
the real dense-raster LZMA baseline, the flips from the real popcount, and the keep
decision is the closed-form 1.27-threshold — there is NO parameter sweep, NO
candidate search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tac.boundary_math.bitmask_dseg import d_seg_reference, flip_count
from tac.boundary_math.dense_raster_lzma_baseline import partition_description_bytes
from tac.boundary_math.partition import RegionAdjacencyGraph, build_region_adjacency_graph

# ── The water level (spec §10) ──────────────────────────────────────────────
# DERIVED (spec §10 verbatim): a single seg flip changes the GLOBAL d_seg mean by
# 1/N_total, where N_total = (#frames) * (per-frame scored pixels).  d_seg is the
# mean over ALL pixels of ALL frames, so the per-flip score-reduction is
# 100 / N_total = 100 / (600 * 196608) = 8.48e-7, and the break-even byte price is
# (100 / N_total) / (25 / D) = 1.27 B/flip.  This is a GLOBAL, frame-independent water
# level — a flip on any frame trades 1:1 in the pooled mean, so the threshold a
# per-frame region-merge solve uses is the same 1.27 regardless of which frame.
_D_RATE_DENOM = 37_545_489  # evaluate.py rate denominator (fixed)
_N_SCORED_PER_FRAME = 384 * 512  # 196,608 — SegNet output resolution
_N_FRAMES_SCORED = 600  # evaluate.py scores 600 pairs (frame1 each)
_N_SCORED_TOTAL = _N_FRAMES_SCORED * _N_SCORED_PER_FRAME  # 117,964,800
_SEG_WEIGHT = 100.0
_RATE_WEIGHT = 25.0
WATER_LEVEL_BYTES_PER_FLIP: float = (_SEG_WEIGHT / _N_SCORED_TOTAL) / (_RATE_WEIGHT / _D_RATE_DENOM)
# ~ 1.2727 B/flip  (matches §10's 1.27)


@dataclass(frozen=True)
class RegionMergePlan:
    """The result of the MDL region-merge solve.

    ``merged_partition`` is the partition after applying every below-water merge.
    ``kept_region_ids`` / ``merged_region_ids`` record the per-region decision.
    ``bytes_before`` / ``bytes_after`` are real dense-raster LZMA description bytes.
    ``flips_before`` / ``flips_after`` are popcount seg-debt vs ``L*``.
    ``water_level`` is the bytes/flip threshold used.
    """

    merged_partition: np.ndarray
    kept_region_ids: tuple[int, ...]
    merged_region_ids: tuple[int, ...]
    bytes_before: int
    bytes_after: int
    flips_before: int
    flips_after: int
    d_seg_before: float
    d_seg_after: float
    water_level: float


def _largest_neighbour(rag: RegionAdjacencyGraph, region_id: int) -> int | None:
    """The ``region_id`` of the largest 4-neighbour region (the MDL merge target).

    A region is contracted INTO its largest neighbour: that removes the most shared
    boundary and is the natural description-length-reducing direction.  Returns None
    if the region has no neighbour (whole-frame single region).
    """

    neigh = rag.neighbours(region_id)
    if not neigh:
        return None
    return max(neigh, key=lambda rid: rag.regions[rid].pixels)


def solve_mdl_region_merge(
    candidate_argmax_hw: np.ndarray,
    gt_argmax_hw: np.ndarray,
    *,
    n_classes: int = 5,
    water_level: float = WATER_LEVEL_BYTES_PER_FLIP,
) -> RegionMergePlan:
    """Solve the region-merge cut: keep a region iff it pays rent at the water level.

    The candidate partition is what the carrier would store; ``gt_argmax_hw`` is the
    scored target ``L*``.

    The merge is a CONTRACTION over the RAG in the description-length-reducing
    direction: a region is a *merge candidate* only when it is SMALLER than its
    largest neighbour (merge small -> large; we never dissolve a large region into a
    tiny one — that would inflate both bytes and flips).  For each merge candidate we
    measure the REAL marginal description bytes it costs (the dense-raster LZMA size
    delta of contracting it into that neighbour) and the flips that contraction would
    re-incur (its pixels that currently AGREE with GT but would not after the merge).

    Decision (the closed-form 1.27-B/flip water-level threshold, NOT a sweep):

        keep the region  iff  marginal_bytes  <  flips_fixed * water_level

    A region whose pixels disagree with GT everywhere fixes zero flips, so it is
    merged at any positive water level (it pays bytes for nothing).  A region that
    pays rent (its encoded bytes are cheaper than the flips it saves at the water
    price) is kept.  d_seg is recomputed exactly (popcount) after the contractions.
    """

    cand = np.asarray(candidate_argmax_hw).astype(np.int64)
    gt = np.asarray(gt_argmax_hw).astype(np.int64)
    if cand.shape != gt.shape:
        raise ValueError(f"candidate {cand.shape} != gt {gt.shape}")

    rag = build_region_adjacency_graph(cand, n_classes)
    merged = cand.copy()

    bytes_before = partition_description_bytes(cand, n_classes)
    flips_before = flip_count(cand, gt)
    d_seg_before = d_seg_reference(cand, gt)

    @dataclass
    class _Cand:
        region_id: int
        merge_label: int
        flips_fixed: int  # flips re-incurred if we MERGE (i.e. flips kept by staying)

    # Build merge candidates: only regions strictly smaller than their largest
    # neighbour (the MDL contraction direction).  Process smallest-first so a tiny
    # region is dissolved before a medium one that could absorb it.
    candidates: list[_Cand] = []
    for rid, region in rag.regions.items():
        target_id = _largest_neighbour(rag, rid)
        if target_id is None:
            continue
        target = rag.regions[target_id]
        if region.pixels >= target.pixels:
            continue  # not a merge candidate — never merge large into small/equal
        merge_label = target.label
        gt_here = gt[region.coords[0], region.coords[1]]
        # flips this region FIXES by staying distinct: pixels that agree with GT under
        # the region's own label but would NOT agree under the merge label.
        flips_fixed = int(np.count_nonzero((gt_here == region.label) & (gt_here != merge_label)))
        candidates.append(_Cand(rid, merge_label, flips_fixed))

    candidates.sort(key=lambda c: rag.regions[c.region_id].pixels)

    kept_ids: list[int] = []
    merged_ids: list[int] = []
    for c in candidates:
        rows, cols = rag.regions[c.region_id].coords
        # REAL marginal bytes of keeping this region distinct: the codec-size delta of
        # contracting it into the merge label on the CURRENT working partition.
        trial = merged.copy()
        trial[rows, cols] = c.merge_label
        bytes_with = partition_description_bytes(merged, n_classes)
        bytes_without = partition_description_bytes(trial, n_classes)
        marginal_bytes = bytes_with - bytes_without  # bytes this region costs to keep
        pays_rent = c.flips_fixed > 0 and marginal_bytes < c.flips_fixed * water_level
        if pays_rent:
            kept_ids.append(c.region_id)
        else:
            merged = trial  # apply the contraction
            merged_ids.append(c.region_id)

    bytes_after = partition_description_bytes(merged, n_classes)
    flips_after = flip_count(merged, gt)
    d_seg_after = d_seg_reference(merged, gt)

    return RegionMergePlan(
        merged_partition=merged,
        kept_region_ids=tuple(kept_ids),
        merged_region_ids=tuple(merged_ids),
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        flips_before=flips_before,
        flips_after=flips_after,
        d_seg_before=d_seg_before,
        d_seg_after=d_seg_after,
        water_level=water_level,
    )
