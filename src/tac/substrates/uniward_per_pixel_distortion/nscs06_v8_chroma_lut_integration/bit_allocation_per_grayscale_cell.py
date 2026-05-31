# SPDX-License-Identifier: MIT
"""UNIWARD cost-map -> ``tac.bit_allocator.per_byte`` -> NSCS06 v8 GRAYSCALE
stream per-cell bit allocation (WAVE-5B; retargets the orphan-loop #1570 wire
from the 240-byte chroma LUT to the DOMINANT grayscale byte surface).

WHY THIS MODULE (the #1570 landing memo's own highest-EV next step)
------------------------------------------------------------------
The sister ``bit_allocation_per_lut_byte.py`` closed orphan-loop #1570 by wiring
``compute_uniward_cost_map -> allocate_per_byte -> NSCS06 v8 chroma LUT`` but the
HONEST VERDICT (Catalog #307) was: on the 240-byte chroma LUT the byte-savings
surface is too small (max ΔS ≈ −2.8e-5, two orders of magnitude below the
contest noise floor). The memo's reactivation path (b) is verbatim:

    "apply the UNIWARD bit-allocation to a LARGER entropy-coded surface than the
     240-byte chroma LUT (e.g. the grayscale stream which dominates the v8
     archive bytes)."

The CH08 GRAYSCALE_STREAM is ``num_pairs * grayscale_h * grayscale_w`` raw uint8
bytes (one byte per low-res luma cell), e.g. ``600 * 96 * 128 = 7,372,800`` bytes
vs the 240-byte LUT — a ``30,720x`` larger surface. ONLY the target surface
changes; the canonical allocator + UNIWARD cost map + no-op proof are REUSED.

THE SPATIAL ALLOCATION INSIGHT (distinct from the LUT)
-----------------------------------------------------
Unlike the LUT (binned by ``(level, class)``), the grayscale stream is SPATIAL:
byte ``(p, gy, gx)`` is the luma of a specific low-res cell. The UNIWARD cost
map is ALSO spatial ``(N, H, W)``. So we aggregate the per-pixel UNIWARD cost
into a per-(gy, gx) low-res sensitivity map (averaged across pairs), allocate a
bit budget across the ``grayscale_h * grayscale_w`` CELLS via the canonical
``tac.bit_allocator.per_byte`` allocator, then BROADCAST each cell's bit-depth
across all ``num_pairs`` bytes at that spatial position. This is:

  - **performant** — the allocator runs over ``gh*gw`` (~12K) cells, not over
    ``num_pairs*gh*gw`` (~7M) individual bytes (the dict-based ``allocate_per_byte``
    would be intractable at 7M bytes);
  - **spatially coherent** — a textured spatial region (HIGH UNIWARD cost =
    scorer-BLIND = SAFE to perturb) is coarsened at EVERY pair, while a smooth
    region (LOW cost = scorer-SENSITIVE) keeps full luma precision everywhere;
  - **faithful** — the cell bit-depth is applied to the REAL grayscale stream
    bytes the v8 inflate consumes, so coarsening DOES change rendered output.

THE END-TO-END WIRE (the #1570 retarget):

    real frames (upstream/videos/0.mkv)
      -> tac.uniward_delta.compute_uniward_cost_map  (N,3,H,W)->(N,H,W)
         [HIGH cost = textured = scorer-BLIND = SAFE = LOW scorer sensitivity]
      -> aggregate_uniward_cost_into_grayscale_cells  (gh,gw) per-cell weight
         [downsample the per-pixel cost to the low-res cell grid, avg over pairs]
      -> per_cell_sensitivity = 1 / (eps + per_cell_weight)  (gh*gw,)  [INVERSE]
      -> tac.bit_allocator.per_byte.allocate_per_byte(budget, sensitivity)
         [TOP_K_BY_SENSITIVITY vs UNIFORM_BASELINE over gh*gw cells]
      -> quantize_grayscale_stream_by_cell_allocation(stream, bits_per_cell)
         [each cell's luma rounded to its allocated bit-depth across ALL pairs]
      -> NSCS06 v8 CH08 archive with the quantized grayscale stream
         [inflate-time bilinear-upsample + lookup_rgb_via_chroma_lut consumes
          EVERY grayscale byte => coarsening DOES change render => not a no-op
          per Catalog #105/#139/#220]

THE FALSIFIABLE CLAIM (Catalog #307):

    On the DOMINANT grayscale surface, UNIWARD-weighted per-cell allocation
    saves archive bytes ABOVE the contest noise floor (``|ΔS_rate| >> 1e-5``)
    at MATCHED advisory rendered-RGB fidelity (vs the uniform baseline). If it
    does NOT Pareto-dominate uniform above the noise floor, the paradigm is
    IMPLEMENTATION-LEVEL falsified at this surface (honest negative per CLAUDE.md
    "Forbidden premature KILL"); the canonical equation stays FORMALIZATION_PENDING.

NON-FAKE PROOF (Catalog #105/#139/#220):

    The cost-map MUST change the allocation. ``allocation_diff_from_uniform_cells``
    returns the cell offsets whose bit-depth differs between the UNIWARD and the
    uniform allocation; a non-empty diff is the structural proof the cost-map is
    actually consumed (a no-op wire would produce an identical allocation
    regardless of the cost-map). The smoke gate refuses to claim a wire-in
    unless the diff is non-empty.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317/#341:
every result is non-promotable ``[macOS-MLX research-signal]`` /
``[macOS-CPU advisory]``; the ~$0.06 paired-CUDA anchor is a separate
operator-funded step. Per Catalog #323 the comparison manifest carries canonical
Provenance markers.

Per Catalog #230 sister-disjoint: NSCS06 v8 substrate is READ-ONLY
consumer-imported; this module does NOT modify the v8 substrate. It SHADOWS the
grayscale stream bytes with a UNIWARD-bit-allocated variant the v8 archive
builder accepts unchanged (same ``(num_pairs * gh * gw,)`` uint8 length).

Canonical equation anchor (proposed; FORMALIZATION_PENDING per Catalog #344):
``uniward_grayscale_stream_bit_allocation_savings_v1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tac.bit_allocator.per_byte import (
    PerByteAllocationMethod,
    PerByteAllocationPlan,
    allocate_per_byte,
)

__all__ = [
    "GRAYSCALE_STREAM_INTEGRATION_NAME",
    "GRAYSCALE_STREAM_INTEGRATION_VERSION",
    "PER_CELL_SENSITIVITY_EPS",
    "PerGrayscaleCellBitAllocationResult",
    "aggregate_uniward_cost_into_grayscale_cells",
    "allocate_grayscale_cells_uniform_baseline",
    "allocate_grayscale_cells_uniward_weighted",
    "allocation_diff_from_uniform_cells",
    "build_canonical_provenance_for_grayscale_bit_allocation",
    "build_uniward_bit_allocated_grayscale_stream",
    "compute_uniward_cost_map_for_frames",
    "per_cell_sensitivity_from_uniward_cell_weights",
    "quantize_grayscale_stream_by_cell_allocation",
]

GRAYSCALE_STREAM_INTEGRATION_NAME = (
    "uniward_cost_map_bit_allocation_per_grayscale_cell_into_nscs06_v8"
)
GRAYSCALE_STREAM_INTEGRATION_VERSION = (
    "v1_2026-05-31_wave5b_grayscale_stream_retarget_of_orphan_loop_1570"
)

# Inverse-sensitivity stabilizer. The UNIWARD per-cell cost is HIGH where the
# scorer is BLIND (safe to perturb); the allocator wants HIGH sensitivity where
# the scorer is SENSITIVE. We invert: sensitivity = 1 / (eps + cost). eps keeps
# the inversion finite for zero-cost cells.
PER_CELL_SENSITIVITY_EPS = 1e-6

# Canonical proposed equation id per Catalog #344 (FORMALIZATION_PENDING until a
# paired-CUDA empirical anchor lands per CLAUDE.md "Submission auth eval - BOTH
# CPU AND CUDA").
CANONICAL_EQUATION_ID_PROPOSED = (
    "uniward_grayscale_stream_bit_allocation_savings_v1"
)


@dataclass(frozen=True)
class PerGrayscaleCellBitAllocationResult:
    """Result of UNIWARD-weighted per-grayscale-CELL bit allocation.

    Attributes
    ----------
    grayscale_quantized : np.ndarray, shape (num_pairs, gh, gw), dtype=uint8
        The grayscale stream after per-cell bit-depth quantization. Same shape +
        dtype as the input so the v8 archive builder is agnostic.
    bits_per_cell : np.ndarray, shape (gh*gw,), dtype=int64
        Allocated bit-depth for each spatial CELL (in canonical flat order
        ``gy * gw + gx``). Broadcast across all num_pairs bytes at that cell.
    plan : PerByteAllocationPlan
        The canonical allocator plan (carries Catalog #323 Provenance).
    per_cell_sensitivity : np.ndarray, shape (gh*gw,)
        The inverse-UNIWARD-cost sensitivity fed to the allocator.
    method : str
        ``"top_k_by_sensitivity"`` (UNIWARD-weighted) or ``"uniform_baseline"``.
    num_pairs : int
    grayscale_h : int
    grayscale_w : int
    """

    grayscale_quantized: np.ndarray
    bits_per_cell: np.ndarray
    plan: PerByteAllocationPlan
    per_cell_sensitivity: np.ndarray
    method: str
    num_pairs: int
    grayscale_h: int
    grayscale_w: int

    @property
    def n_cells(self) -> int:
        return int(self.grayscale_h * self.grayscale_w)

    @property
    def n_stream_bytes(self) -> int:
        return int(self.num_pairs * self.grayscale_h * self.grayscale_w)

    @property
    def n_cells_at_full_precision(self) -> int:
        """Count of CELLS that kept the full 8-bit cap (high-sensitivity)."""
        return int((self.bits_per_cell >= 8).sum())

    @property
    def n_cells_coarsened(self) -> int:
        """Count of CELLS coarsened below 8 bits (low-sensitivity / blind)."""
        return int((self.bits_per_cell < 8).sum())


def compute_uniward_cost_map_for_frames(
    rgb_pairs: np.ndarray,
    *,
    sigma: float = 1e-4,
) -> np.ndarray:
    """Compute the S-UNIWARD per-pixel cost map for compress-time frames.

    Thin numpy bridge around ``tac.uniward_delta.compute_uniward_cost_map``
    (torch-based) so callers stay numpy-native. HIGH cost = textured =
    scorer-BLIND = SAFE to perturb = LOW scorer sensitivity.

    Parameters
    ----------
    rgb_pairs : np.ndarray, shape (N, 3, H, W), dtype=uint8
        Compress-time GT RGB frames at OUTPUT resolution.
    sigma : float
        UNIWARD low-energy stabilizer (default matches ``fridrich``).

    Returns
    -------
    np.ndarray, shape (N, H, W), dtype=float32
        Unnormalized per-pixel UNIWARD cost (HIGH = textured = blind = safe).
    """
    if rgb_pairs.dtype != np.uint8:
        raise ValueError(f"rgb_pairs must be uint8; got {rgb_pairs.dtype}")
    if rgb_pairs.ndim != 4 or rgb_pairs.shape[1] != 3:
        raise ValueError(f"rgb_pairs must be (N, 3, H, W); got {rgb_pairs.shape}")
    import torch

    from tac.uniward_delta import compute_uniward_cost_map

    frames_t = torch.from_numpy(rgb_pairs.astype(np.float32))
    cost = compute_uniward_cost_map(frames_t, sigma=sigma)
    return cost.detach().cpu().numpy().astype(np.float32, copy=False)


def aggregate_uniward_cost_into_grayscale_cells(
    per_pixel_uniward_cost: np.ndarray,
    *,
    grayscale_h: int,
    grayscale_w: int,
) -> np.ndarray:
    """Aggregate the per-pixel UNIWARD cost into per-(gy, gx) low-res cells.

    The per-pixel UNIWARD cost ``(N, H, W)`` is at OUTPUT resolution; the
    grayscale stream is at low-res ``(N, gh, gw)``. We block-average the
    per-pixel cost into the ``gh * gw`` cell grid (each cell = the mean cost
    over its ``(H/gh) x (W/gw)`` output-pixel block), then average ACROSS PAIRS
    so the per-cell sensitivity is shared by every pair's byte at that cell
    (the bit-depth is broadcast across pairs at quantize time).

    Block-averaging (not strided subsampling) keeps the full per-pixel cost
    signal: a textured spatial region's high cost propagates to its cell even
    if the cell's center pixel happens to be smooth.

    Parameters
    ----------
    per_pixel_uniward_cost : np.ndarray, shape (N, H, W), dtype float
        From ``compute_uniward_cost_map_for_frames``.
    grayscale_h, grayscale_w : int
        Low-res cell-grid dimensions (must divide H, W respectively when the
        output resolution is an integer multiple; otherwise PIL-style bilinear
        downsample is used as a fallback).

    Returns
    -------
    np.ndarray, shape (gh, gw), dtype float64
        Per-cell mean UNIWARD cost (non-negative, finite), averaged over pairs.
    """
    cost = np.asarray(per_pixel_uniward_cost)
    if cost.ndim != 3:
        raise ValueError(
            f"per_pixel_uniward_cost must be (N, H, W); got {cost.shape}"
        )
    if int(grayscale_h) <= 0 or int(grayscale_w) <= 0:
        raise ValueError(
            f"grayscale_h/grayscale_w must be positive; got "
            f"{grayscale_h}x{grayscale_w}"
        )
    n, h, w = cost.shape
    gh, gw = int(grayscale_h), int(grayscale_w)
    cost_f = cost.astype(np.float64)

    if h % gh == 0 and w % gw == 0:
        # Exact block-average (faithful: every output pixel contributes once).
        bh, bw = h // gh, w // gw
        # (N, gh, bh, gw, bw) -> mean over (bh, bw)
        blocks = cost_f.reshape(n, gh, bh, gw, bw)
        per_cell_per_pair = blocks.mean(axis=(2, 4))  # (N, gh, gw)
    else:
        # Fallback: PIL bilinear downsample per pair (non-integer ratio).
        from PIL import Image

        per_cell_per_pair = np.stack(
            [
                np.asarray(
                    Image.fromarray(cost_f[p].astype(np.float32)).resize(
                        (gw, gh), Image.BILINEAR
                    ),
                    dtype=np.float64,
                )
                for p in range(n)
            ]
        )  # (N, gh, gw)
    # Average across pairs: the cell sensitivity is shared by every pair.
    return per_cell_per_pair.mean(axis=0)  # (gh, gw)


def per_cell_sensitivity_from_uniward_cell_weights(
    per_cell_cost: np.ndarray,
    *,
    eps: float = PER_CELL_SENSITIVITY_EPS,
) -> np.ndarray:
    """Convert per-cell UNIWARD cost into per-cell allocator sensitivity.

    The UNIWARD cost is HIGH where the scorer is BLIND (safe to perturb). The
    allocator wants HIGH sensitivity where the scorer is SENSITIVE, so we
    INVERT: ``sensitivity = 1 / (eps + cost)``.

    Parameters
    ----------
    per_cell_cost : np.ndarray, shape (gh, gw)
        From ``aggregate_uniward_cost_into_grayscale_cells``.
    eps : float
        Inversion stabilizer; keeps zero-cost cells finite.

    Returns
    -------
    np.ndarray, shape (gh * gw,), dtype float64
        Per-cell sensitivity in canonical flat order ``gy * gw + gx``.
        Non-negative and finite.
    """
    cost = np.asarray(per_cell_cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"per_cell_cost must be (gh, gw); got {cost.shape}")
    inv = 1.0 / (float(eps) + cost)
    return inv.reshape(-1)


def allocate_grayscale_cells_uniward_weighted(
    per_cell_sensitivity: np.ndarray,
    *,
    total_budget_bits: int,
    top_k: int,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> PerByteAllocationPlan:
    """Allocate the grayscale CELL bit budget via UNIWARD TOP_K_BY_SENSITIVITY.

    Spends the per-cell 8-bit cap on the ``top_k`` most-sensitive CELLS (where
    the scorer is SENSITIVE = smooth regions), then distributes residual bits to
    the next-most-sensitive cells. The scorer-blind (textured) cells coarsen.

    Routes through the canonical ``tac.bit_allocator.per_byte.allocate_per_byte``
    so the plan carries Catalog #323 Provenance + the canonical
    ``per_byte_leverage_uniformly_distributed_v1`` equation id.
    """
    sensitivity_per_cell = {
        i: float(per_cell_sensitivity[i])
        for i in range(int(per_cell_sensitivity.shape[0]))
    }
    return allocate_per_byte(
        total_budget_bits=int(total_budget_bits),
        sensitivity_per_byte=sensitivity_per_cell,
        method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
        top_k=int(top_k),
        per_byte_bit_cap=8,
        archive_sha256=archive_sha256,
        captured_at_utc=captured_at_utc,
    )


def allocate_grayscale_cells_uniform_baseline(
    n_cells: int,
    *,
    total_budget_bits: int,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> PerByteAllocationPlan:
    """Allocate the grayscale CELL bit budget via the canonical UNIFORM_BASELINE.

    Every cell receives ``total_budget_bits // n_cells`` bits (the null
    hypothesis of equation ``per_byte_leverage_uniformly_distributed_v1``). This
    is the FALSIFIABLE comparison baseline for the UNIWARD-weighted allocation.
    """
    sensitivity_per_cell = dict.fromkeys(range(int(n_cells)), 1.0)
    return allocate_per_byte(
        total_budget_bits=int(total_budget_bits),
        sensitivity_per_byte=sensitivity_per_cell,
        method=PerByteAllocationMethod.UNIFORM_BASELINE,
        per_byte_bit_cap=8,
        archive_sha256=archive_sha256,
        captured_at_utc=captured_at_utc,
    )


def _bits_per_cell_array(
    plan: PerByteAllocationPlan,
    n_cells: int,
    *,
    min_bits_per_cell: int = 0,
) -> np.ndarray:
    """Materialize a dense (n_cells,) int64 bit-depth array from a plan.

    The plan maps only the cell offsets it allocated to; cells NOT in the plan
    (below the top-K cutoff with zero residual) receive ``min_bits_per_cell``.

    The ``min_bits_per_cell`` floor (rate-allocation-with-floor pattern) means
    UNIWARD spends EXTRA precision on high-sensitivity cells while every cell
    retains a minimum precision — the scorer-blind cells are GRACEFULLY
    COARSENED rather than DESTROYED. The plan's own allocation is never reduced
    below the floor.
    """
    bits = np.full((int(n_cells),), int(min_bits_per_cell), dtype=np.int64)
    for offset, b in plan.bits_per_byte.items():
        if 0 <= int(offset) < int(n_cells):
            bits[int(offset)] = max(int(b), int(min_bits_per_cell))
    return bits


def quantize_grayscale_stream_by_cell_allocation(
    grayscale_stream: np.ndarray,
    bits_per_cell: np.ndarray,
) -> np.ndarray:
    """Quantize each grayscale CELL's luma to its allocated bit-depth.

    A cell allocated ``b`` bits keeps only ``2**b`` distinct levels: the
    canonical uniform mid-rise quantizer maps the original ``[0, 255]`` luma to
    its nearest representable level at bit-depth ``b``, then expands back to the
    ``[0, 255]`` uint8 range (so the v8 bilinear-upsample + LUT lookup pipeline
    is agnostic to the bit-depth — only the *entropy* of the stream changes).

    The same bit-depth is applied to the cell's byte at EVERY pair (the bit
    budget is allocated per spatial cell and broadcast across pairs). A cell
    allocated 8 bits is unchanged (full precision); a cell allocated <= 0 bits
    collapses to 128 (mid-grey). Fewer bits -> fewer distinct values -> lower
    entropy -> smaller compressed grayscale section.

    Parameters
    ----------
    grayscale_stream : np.ndarray, shape (num_pairs, gh, gw), dtype=uint8
        The raw grayscale stream to coarsen.
    bits_per_cell : np.ndarray, shape (gh*gw,), dtype int
        Per-CELL allocated bit-depth (canonical flat order ``gy * gw + gx``).

    Returns
    -------
    np.ndarray, shape (num_pairs, gh, gw), dtype=uint8
        The quantized grayscale stream.
    """
    if grayscale_stream.dtype != np.uint8:
        raise ValueError(
            f"grayscale_stream must be uint8; got {grayscale_stream.dtype}"
        )
    if grayscale_stream.ndim != 3:
        raise ValueError(
            f"grayscale_stream must be (num_pairs, gh, gw); got "
            f"{grayscale_stream.shape}"
        )
    num_pairs, gh, gw = grayscale_stream.shape
    n_cells = gh * gw
    bits = np.asarray(bits_per_cell).reshape(-1)
    if bits.shape[0] != n_cells:
        raise ValueError(
            f"bits_per_cell len {bits.shape[0]} != n_cells {n_cells}"
        )

    # Build a per-cell quantization LUT for each distinct bit-depth so the
    # quantization is a vectorized table lookup (fast at num_pairs*gh*gw scale).
    # quant_lut_by_bits[b] maps [0,255] -> the b-bit representable value.
    quant_lut_by_bits: dict[int, np.ndarray] = {}
    distinct_bits = {int(b) for b in bits.tolist()}
    values = np.arange(256, dtype=np.float64)
    for b in distinct_bits:
        if b >= 8:
            quant_lut_by_bits[b] = values.astype(np.uint8)  # unchanged
        elif b <= 0:
            quant_lut_by_bits[b] = np.full(256, 128, dtype=np.uint8)
        else:
            levels = (1 << b) - 1  # e.g. b=4 -> 15 steps in [0,255]
            step = 255.0 / levels
            q = np.round(np.round(values / step) * step)
            quant_lut_by_bits[b] = np.clip(q, 0, 255).astype(np.uint8)

    # Per-cell quantization: for each cell, look up its luma values through the
    # cell's bit-depth quant LUT. Vectorized over (num_pairs, cell).
    flat = grayscale_stream.reshape(num_pairs, n_cells).astype(np.int64)
    out = np.empty_like(flat, dtype=np.uint8)
    bits_arr = bits.astype(np.int64)
    for cell in range(n_cells):
        out[:, cell] = quant_lut_by_bits[int(bits_arr[cell])][flat[:, cell]]
    return out.reshape(num_pairs, gh, gw)


def allocation_diff_from_uniform_cells(
    uniward_bits: np.ndarray,
    uniform_bits: np.ndarray,
) -> np.ndarray:
    """Return the grayscale CELL offsets whose bit-depth differs (NON-FAKE proof).

    Per Catalog #105/#139/#220 no-op detection: a non-empty diff is the
    structural proof that the UNIWARD cost-map actually changed the bit
    allocation (vs the uniform baseline). The smoke gate refuses to claim a
    wire-in unless this diff is non-empty.

    Returns
    -------
    np.ndarray, shape (n_changed,), dtype=int64
        Sorted cell offsets where ``uniward_bits != uniform_bits``.
    """
    u = np.asarray(uniward_bits).reshape(-1)
    f = np.asarray(uniform_bits).reshape(-1)
    if u.shape != f.shape:
        raise ValueError(
            f"shape mismatch: uniward {u.shape} vs uniform {f.shape}"
        )
    return np.nonzero(u != f)[0].astype(np.int64)


def build_uniward_bit_allocated_grayscale_stream(
    *,
    grayscale_stream: np.ndarray,
    rgb_pairs: np.ndarray,
    total_budget_bits: int,
    top_k: int,
    uniward_cost_map: np.ndarray | None = None,
    min_bits_per_cell: int = 2,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> PerGrayscaleCellBitAllocationResult:
    """End-to-end: UNIWARD cost -> allocate_per_byte -> quantized v8 grayscale stream.

    This is the canonical WAVE-5B wire: it computes (or consumes) the UNIWARD
    cost map on real frames, aggregates it into the ``(gh, gw)`` low-res cell
    grid, inverts to per-cell sensitivity, allocates the bit budget via the
    canonical ``tac.bit_allocator.per_byte`` TOP_K_BY_SENSITIVITY allocator, and
    quantizes the grayscale stream to the per-cell bit-depths (broadcast across
    pairs).

    Parameters
    ----------
    grayscale_stream : np.ndarray, shape (num_pairs, gh, gw), dtype=uint8
        The raw low-res grayscale stream (luma) the v8 archive carries.
    rgb_pairs : np.ndarray, shape (N, 3, H, W), dtype=uint8
        Compress-time GT RGB frames at OUTPUT resolution. ``N`` (cost-map pairs)
        may equal ``num_pairs`` or a smaller representative subset; the per-cell
        cost is averaged over the provided pairs.
    total_budget_bits : int
        Total bit budget across all CELLS. The full-precision budget is
        ``gh * gw * 8``; a budget below that forces coarsening.
    top_k : int
        Number of most-sensitive CELLS to keep at the 8-bit cap.
    uniward_cost_map : np.ndarray | None
        Precomputed per-pixel UNIWARD cost (N, H, W). If None, computed via
        ``compute_uniward_cost_map_for_frames``.
    min_bits_per_cell : int
        Floor bit-depth for EVERY cell (default 2 = 4-level luma). UNIWARD spends
        the premium budget above this floor on high-sensitivity cells; the
        scorer-blind cells are gracefully coarsened to the floor rather than
        destroyed.
    archive_sha256, captured_at_utc : forwarded to the allocator for Provenance.

    Returns
    -------
    PerGrayscaleCellBitAllocationResult
    """
    if grayscale_stream.dtype != np.uint8 or grayscale_stream.ndim != 3:
        raise ValueError(
            f"grayscale_stream must be (num_pairs, gh, gw) uint8; got "
            f"{grayscale_stream.shape} {grayscale_stream.dtype}"
        )
    num_pairs, gh, gw = grayscale_stream.shape
    n_cells = int(gh * gw)

    if uniward_cost_map is None:
        uniward_cost_map = compute_uniward_cost_map_for_frames(rgb_pairs)

    per_cell_cost = aggregate_uniward_cost_into_grayscale_cells(
        uniward_cost_map.astype(np.float32),
        grayscale_h=gh,
        grayscale_w=gw,
    )
    sensitivity = per_cell_sensitivity_from_uniward_cell_weights(per_cell_cost)

    plan = allocate_grayscale_cells_uniward_weighted(
        sensitivity,
        total_budget_bits=int(total_budget_bits),
        top_k=int(top_k),
        archive_sha256=archive_sha256,
        captured_at_utc=captured_at_utc,
    )
    bits = _bits_per_cell_array(
        plan, n_cells, min_bits_per_cell=int(min_bits_per_cell)
    )
    stream_q = quantize_grayscale_stream_by_cell_allocation(grayscale_stream, bits)

    return PerGrayscaleCellBitAllocationResult(
        grayscale_quantized=stream_q,
        bits_per_cell=bits,
        plan=plan,
        per_cell_sensitivity=sensitivity,
        method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY.value,
        num_pairs=int(num_pairs),
        grayscale_h=int(gh),
        grayscale_w=int(gw),
    )


def build_canonical_provenance_for_grayscale_bit_allocation(
    *,
    result: PerGrayscaleCellBitAllocationResult,
    total_budget_bits: int,
    top_k: int,
) -> dict:
    """Build canonical Provenance per Catalog #323 + Catalog #341 markers.

    Every per-grayscale-cell bit-allocation result carries non-promotable
    markers so the surface cannot leak into score/promotion signals. Sister of
    ``bit_allocation_per_lut_byte.build_canonical_provenance_for_bit_allocation``.
    """
    return {
        "integration_id": GRAYSCALE_STREAM_INTEGRATION_NAME,
        "integration_version": GRAYSCALE_STREAM_INTEGRATION_VERSION,
        "consumed_substrate_id": "nscs06_v8_chroma_lut",
        "consumed_substrate_scope": "read_only_consumer_import",
        "consumed_byte_surface": "grayscale_stream",
        "canonical_equation_id_proposed": CANONICAL_EQUATION_ID_PROPOSED,
        "n_cells": result.n_cells,
        "n_stream_bytes": result.n_stream_bytes,
        "total_budget_bits": int(total_budget_bits),
        "top_k": int(top_k),
        "n_cells_at_full_precision": result.n_cells_at_full_precision,
        "n_cells_coarsened": result.n_cells_coarsened,
        "allocation_method": result.method,
        # Canonical Provenance non-promotable markers per Catalog #341.
        "evidence_grade": "macOS-MLX research-signal",
        "score_claim": False,
        "promotable": False,
        "axis_tag": "[predicted]",
        "hardware_substrate_recommendation": "darwin_arm64_m5_max_mlx_local",
        "measurement_axis": "[macOS-MLX research-signal]",
        # Sister hooks per Catalog #125 (bit-allocator PRIMARY hook #3).
        "hook_numbers_fired": [1, 2, 3, 5, 6],
        "entropy_position": "P3_entropy_coded_dominant_grayscale_stream_per_cell_bit_allocation",
        # Sister-disjoint discipline per Catalog #230.
        "nscs06_v8_substrate_modification_scope": "none_read_only_consumer_import",
    }
