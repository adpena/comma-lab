# SPDX-License-Identifier: MIT
"""Region adjacency graph + connected components of the SegNet argmax partition ``L*``.

Source spec §3: the RIGHT data structure for the frame1 seg target is the
**region adjacency graph** of ``L*`` (connected components = contiguity) instead of
a dense 384x512x3 RGB array.  Each connected component of constant label is a
"region"; two regions are adjacent iff some pixel of one is 4-neighbour to a pixel
of the other.

Connected components are computed via ``scipy.ndimage.label`` per class (4-connectivity,
matching the contour codec's chain-code neighbourhood).  This is a real labelling of
the array, NOT a placeholder — tests verify component counts + adjacency on known
partitions.

Reuse note (NO FAKE, only what the code honors): exact contest-unit region PRICING
(margin stats, size class) is delegated to ``tac.analysis.hinerv_hard_region_miner``
(``region_margin_stats`` / ``size_class_for_pixels``); this module supplies the
graph/contiguity structure those prices attach to.  We do NOT re-derive pricing here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# 4-connectivity structuring element (matches chain-code boundary neighbourhood).
_CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass(frozen=True)
class Region:
    """A connected component of constant label in the partition.

    ``region_id`` is a stable per-partition integer id.  ``label`` is the SegNet
    class id of every pixel in the region.  ``pixels`` is the count.  ``coords`` is
    a ``(2, n)`` int array of (row, col) pixel coordinates (kept for rasterization +
    contour extraction; small for the boundary regions that matter).
    """

    region_id: int
    label: int
    pixels: int
    coords: np.ndarray  # (2, n) int64 (rows, cols)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Render this region as a boolean (H, W) mask."""

        m = np.zeros(shape, dtype=bool)
        m[self.coords[0], self.coords[1]] = True
        return m


@dataclass
class RegionAdjacencyGraph:
    """The RAG: regions (nodes) + adjacency (edges) over a fixed-shape partition.

    ``adjacency[region_id]`` is the set of neighbouring region ids.  ``region_of``
    is a dense ``(H, W)`` int array mapping each pixel to its ``region_id`` (so a
    merge can rewrite labels by region id directly).
    """

    shape: tuple[int, int]
    regions: dict[int, Region]
    adjacency: dict[int, set[int]]
    region_of: np.ndarray  # (H, W) int64 region id per pixel
    labels: np.ndarray  # (H, W) int64 class id per pixel (the source L*)

    def neighbours(self, region_id: int) -> set[int]:
        return self.adjacency.get(region_id, set())

    def n_regions(self) -> int:
        return len(self.regions)


def connected_components(
    argmax_hw: np.ndarray, n_classes: int = 5
) -> tuple[np.ndarray, dict[int, Region]]:
    """Label connected components of constant class via 4-connectivity.

    Returns ``(region_of, regions)`` where ``region_of`` is a dense ``(H, W)`` int
    array of per-pixel region ids (>= 0, contiguous) and ``regions`` maps each id to
    its :class:`Region`.  Components never cross a class boundary (each component is
    a maximal 4-connected run of a single class).
    """

    a = np.asarray(argmax_hw)
    if a.ndim != 2:
        raise ValueError(f"argmax must be (H, W); got {a.shape}")
    region_of = np.full(a.shape, -1, dtype=np.int64)
    regions: dict[int, Region] = {}
    next_id = 0
    for c in range(n_classes):
        class_mask = a == c
        if not class_mask.any():
            continue
        labelled, n = ndimage.label(class_mask, structure=_CONN4)
        for comp in range(1, n + 1):
            comp_mask = labelled == comp
            rows, cols = np.nonzero(comp_mask)
            region_of[rows, cols] = next_id
            regions[next_id] = Region(
                region_id=next_id,
                label=c,
                pixels=int(rows.size),
                coords=np.stack([rows, cols], axis=0).astype(np.int64),
            )
            next_id += 1
    if (region_of < 0).any():
        # Should be impossible if classes cover [0, n_classes); fail closed.
        missing = int(np.count_nonzero(region_of < 0))
        raise ValueError(f"{missing} pixels not assigned to any region (class id out of range?)")
    return region_of, regions


def build_region_adjacency_graph(
    argmax_hw: np.ndarray, n_classes: int = 5
) -> RegionAdjacencyGraph:
    """Build the full RAG (components + 4-neighbour adjacency edges)."""

    a = np.asarray(argmax_hw).astype(np.int64)
    region_of, regions = connected_components(a, n_classes)
    adjacency: dict[int, set[int]] = {rid: set() for rid in regions}

    # Horizontal + vertical adjacency: pixels differing in region id across a
    # 4-neighbour edge connect those two regions.
    left = region_of[:, :-1]
    right = region_of[:, 1:]
    h_mask = left != right
    for ra, rb in zip(left[h_mask].tolist(), right[h_mask].tolist(), strict=True):
        adjacency[ra].add(rb)
        adjacency[rb].add(ra)

    top = region_of[:-1, :]
    bot = region_of[1:, :]
    v_mask = top != bot
    for ra, rb in zip(top[v_mask].tolist(), bot[v_mask].tolist(), strict=True):
        adjacency[ra].add(rb)
        adjacency[rb].add(ra)

    return RegionAdjacencyGraph(
        shape=(int(a.shape[0]), int(a.shape[1])),
        regions=regions,
        adjacency=adjacency,
        region_of=region_of,
        labels=a,
    )
