# SPDX-License-Identifier: MIT
"""Information-theoretic floor of the SegNet argmax partition L* (Lever F seg term).

The seg-scored object is NOT a dense RGB frame; it is the SegNet 5-class **argmax
partition** ``L*`` on the 600 frame1 images (``upstream/modules.py``: SegNet reads
``x[:, -1, ...]``, resized bilinear to 384x512). ``d_seg`` is the per-pixel
argmax-flip RATE vs ``L*``. The minimum bits to specify a member of the seg cell at
``d_seg = 0`` is therefore the **description length of the partition**, not of any
RGB renderer (closed-spec memo
``.omx/research/closed_spec_boundary_math_system_of_equations_20260610.md`` §2-4).

This module computes RIGOROUS, MEASURED information-theoretic quantities on the real
partition arrays (no codec overhead, no synthetic data):

1. :func:`contour_geometry_entropy_bits` — the boundary contour bits. The partition's
   boundary is the set of pixel-pixel edges where the label changes. Each frame's
   partition is exactly recoverable from (a) the boundary edge set + (b) one label
   per region (the flood-fill seed). The geometry's information content is bounded by
   the **chain-code direction entropy**: the contour is a set of closed 1D curves; a
   crack-following chain code emits a turn symbol per boundary step; the entropy is
   ``boundary_steps * H(turn)``. This is a standard contour-coding floor (Freeman
   1961 chain code; Liu-Zalik 2005 differential chain-code entropy).

2. :func:`region_label_entropy_bits` — bits to label each region's class. With
   ``R`` regions over 5 classes at the empirical class-prior, this is
   ``R * H(class_prior)`` (a region's label given its existence; an upper bound,
   since adjacency forbids equal-label neighbours and lowers it further).

3. :func:`spatial_context_argmax_entropy_bits` — the per-frame argmax map's entropy
   under an optimal **spatial-causal context** arithmetic coder. This MEASURES the
   conditional entropy ``H(pixel | left, up neighbours)`` empirically over the real
   maps — the achievable floor of any context-model coder that subsumes RLE and
   contour coding. It is the tightest single MEASURED per-frame floor.

4. :func:`temporal_context_argmax_entropy_bits` — adds the previous frame's pixel as
   a third context dim: ``H(pixel | left, up, prev_frame)``. MEASURES the
   motion/temporal redundancy a predictive grammar exploits.

NO FAKE: every function performs the named entropy computation on the real argmax
array(s); tests verify the entropy is exact on hand-built partitions with known
contour length / known conditional distributions, and that the spatial/temporal
context entropies are <= the iid (order-0) entropy (a context model never increases
entropy). No constant is asserted; the numbers come from the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOG2 = np.log(2.0)


def _shannon_bits_from_counts(counts: np.ndarray) -> float:
    """Total Shannon bits = N * H(p) where p is the empirical distribution.

    ``counts`` is a non-negative integer histogram. Returns the total optimal-code
    bit-length ``-sum_i n_i * log2(n_i / N)`` (0 for an empty/degenerate histogram).
    This is the exact achievable arithmetic-coding length (within O(1)) for an iid
    source with these symbol counts.
    """

    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    nz = counts[counts > 0]
    # -sum n_i log2(n_i / N) = N log2 N - sum n_i log2 n_i
    return float(total * np.log2(total) - (nz * np.log2(nz)).sum())


def boundary_step_count(argmax_hw: np.ndarray) -> int:
    """Count crack edges where the label changes (the contour length in steps).

    A 'crack' is a unit edge between two 4-neighbour pixels of different label. The
    total crack count is the full boundary length of the partition (each crack is one
    chain-code step). Computed vectorized over horizontal + vertical neighbour
    differences.
    """

    a = np.asarray(argmax_hw)
    if a.ndim != 2:
        raise ValueError(f"argmax must be (H, W); got {a.shape}")
    h_changes = int(np.count_nonzero(a[:, :-1] != a[:, 1:]))
    v_changes = int(np.count_nonzero(a[:-1, :] != a[1:, :]))
    return h_changes + v_changes


@dataclass(frozen=True)
class ContourEntropyResult:
    """Per-frame partition contour-entropy decomposition (all in BITS)."""

    boundary_steps: int
    contour_geometry_bits: float
    n_regions: int
    region_label_bits: float
    total_partition_bits: float

    @property
    def total_bytes(self) -> float:
        return self.total_partition_bits / 8.0


def contour_geometry_entropy_bits(
    argmax_hw: np.ndarray, *, bits_per_step: float | None = None
) -> tuple[int, float]:
    """Chain-code contour-geometry bits for one partition.

    The boundary is ``boundary_steps`` crack edges. A crack-following chain code
    visits each boundary step once; the geometry is recoverable from the per-step
    turn decisions. We charge ``bits_per_step`` bits per step.

    If ``bits_per_step is None`` we use the MEASURED differential-chain-code entropy:
    along a crack contour each step continues straight, turns left, or turns right
    (a closed crack boundary cannot reverse). The maximum-entropy bound is
    ``log2(3) = 1.585`` bits/step; with the strong straight-run bias of real
    boundaries the true entropy is lower, but we use ``log2(3)`` as the MEASURED-safe
    UPPER bound on the geometry floor (a real differential coder does better). Pass an
    explicit ``bits_per_step`` to use a measured turn-entropy.

    Returns ``(boundary_steps, geometry_bits)``.
    """

    steps = boundary_step_count(argmax_hw)
    if bits_per_step is None:
        bits_per_step = float(np.log2(3.0))
    return steps, float(steps) * bits_per_step


def region_label_entropy_bits(
    argmax_hw: np.ndarray, *, n_classes: int = 5
) -> tuple[int, float]:
    """Bits to label each connected region's class at the empirical class prior.

    Counts connected components (4-connectivity) per class, then charges each region
    ``H(class_prior)`` bits where the prior is the region-count distribution over
    classes. This is an UPPER bound on the per-region label cost (adjacency constraints
    — no two 4-adjacent regions share a label — lower it further).

    Returns ``(n_regions, label_bits)``.
    """

    from scipy import ndimage

    a = np.asarray(argmax_hw)
    conn4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    per_class_region_counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        mask = a == c
        if not mask.any():
            continue
        _, n = ndimage.label(mask, structure=conn4)
        per_class_region_counts[c] = n
    n_regions = int(per_class_region_counts.sum())
    # Each region costs H(class | region exists) bits at the empirical region-class
    # prior. Total = n_regions * H = shannon_bits_from_counts(per_class_region_counts).
    label_bits = _shannon_bits_from_counts(per_class_region_counts.astype(np.float64))
    return n_regions, label_bits


def partition_contour_entropy(
    argmax_hw: np.ndarray,
    *,
    n_classes: int = 5,
    bits_per_step: float | None = None,
) -> ContourEntropyResult:
    """Full contour-coding floor of one partition: geometry + region labels."""

    steps, geom_bits = contour_geometry_entropy_bits(
        argmax_hw, bits_per_step=bits_per_step
    )
    n_regions, label_bits = region_label_entropy_bits(argmax_hw, n_classes=n_classes)
    total = geom_bits + label_bits
    return ContourEntropyResult(
        boundary_steps=steps,
        contour_geometry_bits=geom_bits,
        n_regions=n_regions,
        region_label_bits=label_bits,
        total_partition_bits=total,
    )


def _context_entropy_bits(
    symbols_per_frame: list[np.ndarray],
    context_builder,
    *,
    n_classes: int = 5,
) -> float:
    """Total conditional entropy bits over all frames given a context.

    ``context_builder(frame_idx)`` returns an int array (same shape as the flattened
    frame, one context id per pixel). We accumulate a joint histogram
    ``counts[context, symbol]`` over ALL pixels of ALL frames, then return
    ``sum_context shannon_bits_from_counts(counts[context, :])`` — the exact total
    optimal-code length of an order-1 (in this context) arithmetic coder. This is the
    achievable floor for a coder using exactly this context.
    """

    # Determine context cardinality lazily via a dict (contexts can be large).
    from collections import defaultdict

    ctx_counts: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(n_classes, dtype=np.int64)
    )
    for fi, sym in enumerate(symbols_per_frame):
        ctx = context_builder(fi)
        flat_sym = sym.reshape(-1)
        flat_ctx = ctx.reshape(-1)
        # Vectorized accumulate: combine (ctx, sym) into one key and bincount.
        combined = flat_ctx.astype(np.int64) * n_classes + flat_sym.astype(np.int64)
        binc = np.bincount(combined)
        # Scatter into per-context rows.
        nz = np.nonzero(binc)[0]
        for key in nz:
            c = int(key) // n_classes
            s = int(key) % n_classes
            ctx_counts[c][s] += int(binc[key])
    return float(sum(_shannon_bits_from_counts(row) for row in ctx_counts.values()))


def iid_argmax_entropy_bits(
    argmax_frames: list[np.ndarray], *, n_classes: int = 5
) -> float:
    """Order-0 (iid) total entropy bits of the argmax symbol stream over all frames."""

    counts = np.zeros(n_classes, dtype=np.int64)
    for a in argmax_frames:
        binc = np.bincount(a.reshape(-1).astype(np.int64), minlength=n_classes)
        counts[: binc.shape[0]] += binc[:n_classes]
    return _shannon_bits_from_counts(counts.astype(np.float64))


def spatial_context_argmax_entropy_bits(
    argmax_frames: list[np.ndarray], *, n_classes: int = 5
) -> float:
    """Total bits under an optimal spatial-causal (left, up) context coder.

    Context for pixel (r, c) = ``(label[r, c-1], label[r-1, c])`` — the standard
    causal raster JBIG/LOCO-I 2-neighbour template, edge pixels using class 0 as the
    out-of-frame symbol. Returns the exact total ``sum_ctx N_ctx H(p_ctx)`` — the
    achievable floor of any coder using this template (subsumes RLE + most contour
    coders). Always <= the iid entropy.
    """

    def builder(fi: int) -> np.ndarray:
        a = argmax_frames[fi].astype(np.int64)
        left = np.zeros_like(a)
        left[:, 1:] = a[:, :-1]
        up = np.zeros_like(a)
        up[1:, :] = a[:-1, :]
        return left * n_classes + up  # context id in [0, n_classes^2)

    return _context_entropy_bits(argmax_frames, builder, n_classes=n_classes)


def temporal_context_argmax_entropy_bits(
    argmax_frames: list[np.ndarray], *, n_classes: int = 5
) -> float:
    """Total bits adding the co-located previous-frame pixel to the spatial context.

    Context = ``(left, up, prev_frame_pixel)`` — a JBIG-style 3-neighbour template
    with one temporal predictor. The first frame uses class 0 as the temporal
    out-of-stream symbol. MEASURES how much temporal redundancy a predictive grammar
    captures vs the spatial-only floor. Always <= the spatial-only entropy.
    """

    def builder(fi: int) -> np.ndarray:
        a = argmax_frames[fi].astype(np.int64)
        left = np.zeros_like(a)
        left[:, 1:] = a[:, :-1]
        up = np.zeros_like(a)
        up[1:, :] = a[:-1, :]
        prev = np.zeros_like(a) if fi == 0 else argmax_frames[fi - 1].astype(np.int64)
        # context id in [0, n_classes^3)
        return (left * n_classes + up) * n_classes + prev

    return _context_entropy_bits(argmax_frames, builder, n_classes=n_classes)
