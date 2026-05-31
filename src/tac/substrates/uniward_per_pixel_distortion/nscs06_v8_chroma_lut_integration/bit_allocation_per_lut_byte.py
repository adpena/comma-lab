# SPDX-License-Identifier: MIT
"""UNIWARD cost-map -> ``tac.bit_allocator.per_byte`` -> NSCS06 v8 chroma-LUT
per-byte bit allocation (the long-orphaned inverse-steganalysis loop, #1570).

This module is the CONNECTIVE TISSUE the orphan-loop #1570 was missing. The
sister ``lut_derivation_uniward_weighted.py`` already weights *which RGB value*
each (level, class) bin picks (the canonical UNIWARD-weighted median). This
module is orthogonal and distinct: it allocates a *bit budget* across the 240
LUT bytes by UNIWARD sensitivity, so the scorer-BLIND bins are coarsened to
fewer bits (lower entropy -> smaller compressed LUT section) while the
scorer-SENSITIVE bins keep full 8-bit precision exactly where it matters.

THE END-TO-END WIRE (the orphan-loop #1570 closure):

    real frames (upstream/videos/0.mkv)
      -> tac.uniward_delta.compute_uniward_cost_map  (B,3,H,W)->(B,H,W)
         [HIGH cost = textured = scorer-BLIND = SAFE = low sensitivity]
      -> aggregate_per_pixel_uniward_weights_into_lut_bins  (16,5) per-bin weight
      -> per_lut_byte_sensitivity = 1 / (eps + per_bin_weight)  (240,)
         [INVERSE: sensitive bins (low UNIWARD weight) -> HIGH allocator sensitivity]
      -> tac.bit_allocator.per_byte.allocate_per_byte(total_budget_bits, sensitivity)
         [TOP_K_BY_SENSITIVITY: spend the 8-bit cap on the K most-sensitive LUT
          bytes; coarsen the rest to their allocated bits]
      -> quantize_lut_by_allocation(lut, bits_per_byte)
         [each LUT byte rounded to its allocated bit-depth: a 4-bit byte keeps
          only 16 distinct levels instead of 256 -> lower entropy]
      -> NSCS06 v8 archive (CH08 grammar) with the quantized LUT
         [the inflate-time lookup_rgb_via_chroma_lut consumes EVERY LUT byte,
          so coarsening DOES change rendered output -> not a no-op per
          Catalog #105/#139/#220]

THE FALSIFIABLE CLAIM (Catalog #307):

    UNIWARD-weighted allocation (TOP_K_BY_SENSITIVITY) produces a smaller
    compressed LUT section OR a smaller advisory reconstruction error at
    matched total bit budget than the UNIFORM_BASELINE allocation.

If the UNIWARD-weighted quantized LUT compresses identically to the uniform
quantized LUT AND yields identical advisory reconstruction error, the
paradigm is IMPLEMENTATION-LEVEL falsified at this surface (the cost-map's
sensitivity ranking does not buy anything the uniform allocation does not).
That is an honest negative per CLAUDE.md "Forbidden premature KILL".

NON-FAKE PROOF (Catalog #105/#139/#220):

    The cost-map MUST change the allocation. ``allocation_diff_from_uniform``
    returns the set of LUT bytes whose bit-depth differs between the UNIWARD
    allocation and the uniform allocation; a non-empty diff is the structural
    proof the cost-map is actually consumed (a wire that produced an identical
    allocation regardless of the cost-map would be a no-op). The smoke gate
    refuses to claim a wire-in unless the diff is non-empty.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317/#341:
every result is non-promotable ``[macOS-MLX research-signal]`` /
``[macOS-CPU advisory]``; the ~$0.06 paired-CUDA anchor is a separate
operator-funded step. Per Catalog #323 the comparison manifest carries
canonical Provenance markers.

Per Catalog #230 sister-disjoint: NSCS06 v8 substrate is READ-ONLY
consumer-imported; this module does NOT modify the v8 substrate. It SHADOWS
the chroma-LUT bytes with a UNIWARD-bit-allocated variant the v8 archive
builder accepts unchanged (same ``(16,5,3)`` uint8 shape).

Canonical equation anchor (proposed; FORMALIZATION_PENDING per Catalog #344):
``uniward_cost_map_bit_allocation_per_lut_byte_savings_v1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tac.bit_allocator.per_byte import (
    PerByteAllocationMethod,
    PerByteAllocationPlan,
    allocate_per_byte,
)

from .weight_map_per_lut_index import (
    GRAYSCALE_LEVELS_DEFAULT,
    NUM_SEGNET_CLASSES,
    PerLutIndexUniwardWeights,
    aggregate_per_pixel_uniward_weights_into_lut_bins,
)

__all__ = [
    "BIT_ALLOCATION_INTEGRATION_NAME",
    "BIT_ALLOCATION_INTEGRATION_VERSION",
    "PER_LUT_BYTE_SENSITIVITY_EPS",
    "PerLutByteBitAllocationResult",
    "allocate_lut_bits_uniform_baseline",
    "allocate_lut_bits_uniward_weighted",
    "allocation_diff_from_uniform",
    "build_canonical_provenance_for_bit_allocation",
    "build_uniward_bit_allocated_chroma_lut",
    "compute_uniward_cost_map_for_frames",
    "per_lut_byte_sensitivity_from_uniward_weights",
    "quantize_lut_by_allocation",
]

BIT_ALLOCATION_INTEGRATION_NAME = (
    "uniward_cost_map_bit_allocation_per_lut_byte_into_nscs06_v8_chroma_lut"
)
BIT_ALLOCATION_INTEGRATION_VERSION = "v1_2026-05-31_orphan_loop_1570_closure"

# Inverse-sensitivity stabilizer. The UNIWARD per-bin weight is HIGH where the
# scorer is BLIND (safe to perturb); the allocator wants HIGH sensitivity where
# the scorer is SENSITIVE. We invert: sensitivity = 1 / (eps + weight). eps
# keeps the inversion finite for zero-weight (empty) bins.
PER_LUT_BYTE_SENSITIVITY_EPS = 1e-6

# Canonical proposed equation id per Catalog #344 (FORMALIZATION_PENDING until
# the paired-CUDA empirical anchor lands per CLAUDE.md "Submission auth eval -
# BOTH CPU AND CUDA").
CANONICAL_EQUATION_ID_PROPOSED = (
    "uniward_cost_map_bit_allocation_per_lut_byte_savings_v1"
)


@dataclass(frozen=True)
class PerLutByteBitAllocationResult:
    """Result of UNIWARD-weighted per-LUT-byte bit allocation.

    Attributes
    ----------
    lut_quantized : np.ndarray, shape (levels, classes, 3), dtype=uint8
        The chroma LUT after per-byte bit-depth quantization. Same shape +
        dtype as the input LUT so the v8 archive builder is agnostic.
    bits_per_lut_byte : np.ndarray, shape (levels*classes*3,), dtype=int64
        Allocated bit-depth for each LUT byte (in canonical flat order
        ``level * classes * 3 + class * 3 + channel``).
    plan : PerByteAllocationPlan
        The canonical allocator plan (carries Catalog #323 Provenance).
    per_lut_byte_sensitivity : np.ndarray, shape (levels*classes*3,)
        The inverse-UNIWARD-weight sensitivity fed to the allocator.
    method : str
        ``"top_k_by_sensitivity"`` (UNIWARD-weighted) or ``"uniform_baseline"``.
    grayscale_levels : int
    num_segnet_classes : int
    """

    lut_quantized: np.ndarray
    bits_per_lut_byte: np.ndarray
    plan: PerByteAllocationPlan
    per_lut_byte_sensitivity: np.ndarray
    method: str
    grayscale_levels: int
    num_segnet_classes: int

    @property
    def n_lut_bytes(self) -> int:
        return int(self.grayscale_levels * self.num_segnet_classes * 3)

    @property
    def n_bytes_at_full_precision(self) -> int:
        """Count of LUT bytes that kept the full 8-bit cap (high-sensitivity)."""
        return int((self.bits_per_lut_byte >= 8).sum())

    @property
    def n_bytes_coarsened(self) -> int:
        """Count of LUT bytes coarsened below 8 bits (low-sensitivity / blind)."""
        return int((self.bits_per_lut_byte < 8).sum())


def compute_uniward_cost_map_for_frames(
    rgb_pairs: np.ndarray,
    *,
    sigma: float = 1e-4,
) -> np.ndarray:
    """Compute the S-UNIWARD per-pixel cost map for compress-time frames.

    Thin numpy bridge around ``tac.uniward_delta.compute_uniward_cost_map``
    (which is torch-based) so callers can stay numpy-native. HIGH cost =
    textured = scorer-BLIND = SAFE to perturb = LOW scorer sensitivity.

    Parameters
    ----------
    rgb_pairs : np.ndarray, shape (N, 3, H, W), dtype=uint8
        Compress-time GT RGB frames (the same input the v8 LUT derivation
        consumes).
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
    # Lazy import torch so numpy-only callers that pass a precomputed cost map
    # to the downstream helpers do not pay the torch import cost.
    import torch

    from tac.uniward_delta import compute_uniward_cost_map

    frames_t = torch.from_numpy(rgb_pairs.astype(np.float32))
    cost = compute_uniward_cost_map(frames_t, sigma=sigma)
    return cost.detach().cpu().numpy().astype(np.float32, copy=False)


def per_lut_byte_sensitivity_from_uniward_weights(
    per_bin_weights: PerLutIndexUniwardWeights,
    *,
    eps: float = PER_LUT_BYTE_SENSITIVITY_EPS,
) -> np.ndarray:
    """Convert per-(level, class) UNIWARD weights into per-LUT-byte sensitivity.

    The UNIWARD weight is HIGH where the scorer is BLIND (safe to perturb).
    The allocator wants HIGH sensitivity where the scorer is SENSITIVE, so we
    INVERT: ``sensitivity = 1 / (eps + weight)``. Each (level, class) bin's
    inverse weight is broadcast across its 3 RGB channel-bytes (the 3 bytes
    of a LUT entry share the bin's scorer sensitivity).

    Parameters
    ----------
    per_bin_weights : PerLutIndexUniwardWeights
        From ``aggregate_per_pixel_uniward_weights_into_lut_bins``.
    eps : float
        Inversion stabilizer; keeps empty (zero-weight) bins finite.

    Returns
    -------
    np.ndarray, shape (levels * classes * 3,), dtype=float64
        Per-LUT-byte sensitivity in canonical flat order
        ``level * classes * 3 + class * 3 + channel``. Non-negative and finite.
    """
    weight_per_bin = per_bin_weights.weight_per_bin  # (levels, classes)
    levels, classes = weight_per_bin.shape
    # INVERSE: sensitive bins (low UNIWARD weight) -> high allocator sensitivity.
    inv = 1.0 / (float(eps) + weight_per_bin.astype(np.float64))
    # Broadcast each (level, class) inverse-weight across its 3 RGB bytes.
    sensitivity = np.repeat(inv.reshape(levels, classes, 1), 3, axis=2)
    return sensitivity.reshape(-1)


def allocate_lut_bits_uniward_weighted(
    per_lut_byte_sensitivity: np.ndarray,
    *,
    total_budget_bits: int,
    top_k: int,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> PerByteAllocationPlan:
    """Allocate the LUT bit budget via UNIWARD TOP_K_BY_SENSITIVITY.

    Spends the per-byte 8-bit cap on the ``top_k`` most-sensitive LUT bytes
    (where the scorer is SENSITIVE), then distributes residual bits to the
    next-most-sensitive bytes. The scorer-blind bytes get coarsened.

    Routes through the canonical ``tac.bit_allocator.per_byte.allocate_per_byte``
    so the plan carries Catalog #323 Provenance + the canonical
    ``per_byte_leverage_uniformly_distributed_v1`` equation id.
    """
    sensitivity_per_byte = {
        i: float(per_lut_byte_sensitivity[i])
        for i in range(int(per_lut_byte_sensitivity.shape[0]))
    }
    return allocate_per_byte(
        total_budget_bits=int(total_budget_bits),
        sensitivity_per_byte=sensitivity_per_byte,
        method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
        top_k=int(top_k),
        per_byte_bit_cap=8,
        archive_sha256=archive_sha256,
        captured_at_utc=captured_at_utc,
    )


def allocate_lut_bits_uniform_baseline(
    n_lut_bytes: int,
    *,
    total_budget_bits: int,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> PerByteAllocationPlan:
    """Allocate the LUT bit budget via the canonical UNIFORM_BASELINE.

    Every LUT byte receives ``total_budget_bits // n_lut_bytes`` bits (the
    null hypothesis of equation ``per_byte_leverage_uniformly_distributed_v1``).
    This is the FALSIFIABLE comparison baseline for the UNIWARD-weighted
    allocation.
    """
    # The uniform allocator does not use sensitivity; pass a flat unit prior.
    sensitivity_per_byte = dict.fromkeys(range(int(n_lut_bytes)), 1.0)
    return allocate_per_byte(
        total_budget_bits=int(total_budget_bits),
        sensitivity_per_byte=sensitivity_per_byte,
        method=PerByteAllocationMethod.UNIFORM_BASELINE,
        per_byte_bit_cap=8,
        archive_sha256=archive_sha256,
        captured_at_utc=captured_at_utc,
    )


def _bits_per_byte_array(
    plan: PerByteAllocationPlan,
    n_lut_bytes: int,
    *,
    min_bits_per_byte: int = 0,
) -> np.ndarray:
    """Materialize a dense (n_lut_bytes,) int64 bit-depth array from a plan.

    The plan's ``bits_per_byte`` maps only the byte offsets it allocated to.
    Bytes NOT in the plan (below the top-K cutoff with zero residual) receive
    ``min_bits_per_byte`` bits.

    The ``min_bits_per_byte`` floor (rate-allocation-with-floor pattern) means
    UNIWARD spends EXTRA precision on high-sensitivity bytes while every byte
    retains a minimum precision — the scorer-blind bytes are GRACEFULLY
    COARSENED (e.g. to 2-bit / 4-level chroma) rather than DESTROYED (collapsed
    to a single value). The 0-floor variant (full destruction of low-sensitivity
    bytes) is empirically catastrophic for reconstruction; see the smoke memo.
    The plan's own allocation is never reduced below the floor.
    """
    bits = np.full((int(n_lut_bytes),), int(min_bits_per_byte), dtype=np.int64)
    for offset, b in plan.bits_per_byte.items():
        if 0 <= int(offset) < int(n_lut_bytes):
            bits[int(offset)] = max(int(b), int(min_bits_per_byte))
    return bits


def quantize_lut_by_allocation(
    lut: np.ndarray,
    bits_per_lut_byte: np.ndarray,
) -> np.ndarray:
    """Quantize each LUT byte to its allocated bit-depth (the coarsening).

    A LUT byte allocated ``b`` bits keeps only ``2**b`` distinct levels: the
    canonical uniform mid-rise quantizer maps the original ``[0, 255]`` value
    to its nearest representable level at bit-depth ``b``, then expands back to
    the ``[0, 255]`` uint8 range (so the v8 lookup + downstream pipeline is
    agnostic to the bit-depth — only the *entropy* of the bytes changes).

    A byte allocated 8 bits is unchanged (full precision). A byte allocated 0
    bits collapses to a single representable value (128, the mid-grey
    canonical fallback). Fewer bits -> fewer distinct values -> lower entropy
    -> smaller compressed LUT section.

    Parameters
    ----------
    lut : np.ndarray, shape (levels, classes, 3), dtype=uint8
        The chroma LUT to coarsen.
    bits_per_lut_byte : np.ndarray, shape (levels*classes*3,), dtype int
        Per-byte allocated bit-depth (canonical flat order).

    Returns
    -------
    np.ndarray, shape (levels, classes, 3), dtype=uint8
        The quantized LUT.
    """
    if lut.dtype != np.uint8:
        raise ValueError(f"lut must be uint8; got {lut.dtype}")
    if lut.ndim != 3 or lut.shape[2] != 3:
        raise ValueError(f"lut must be (levels, classes, 3); got {lut.shape}")
    flat = lut.reshape(-1).astype(np.float64)
    bits = np.asarray(bits_per_lut_byte).reshape(-1)
    if bits.shape[0] != flat.shape[0]:
        raise ValueError(
            f"bits_per_lut_byte len {bits.shape[0]} != n_lut_bytes {flat.shape[0]}"
        )
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        b = int(bits[i])
        v = flat[i]
        if b >= 8:
            out[i] = v  # full precision, unchanged
        elif b <= 0:
            out[i] = 128.0  # 0-bit byte carries no info -> mid-grey fallback
        else:
            levels = (1 << b) - 1  # e.g. b=4 -> 15 representable steps in [0,255]
            step = 255.0 / levels
            out[i] = np.round(np.round(v / step) * step)
    return np.clip(out, 0, 255).astype(np.uint8).reshape(lut.shape)


def allocation_diff_from_uniform(
    uniward_bits: np.ndarray,
    uniform_bits: np.ndarray,
) -> np.ndarray:
    """Return the LUT byte offsets whose bit-depth differs (NON-FAKE proof).

    Per Catalog #105/#139/#220 no-op detection: a non-empty diff is the
    structural proof that the UNIWARD cost-map actually changed the bit
    allocation (vs the uniform baseline). The smoke gate refuses to claim a
    wire-in unless this diff is non-empty.

    Returns
    -------
    np.ndarray, shape (n_changed,), dtype=int64
        Sorted byte offsets where ``uniward_bits != uniform_bits``.
    """
    u = np.asarray(uniward_bits).reshape(-1)
    f = np.asarray(uniform_bits).reshape(-1)
    if u.shape != f.shape:
        raise ValueError(f"shape mismatch: uniward {u.shape} vs uniform {f.shape}")
    return np.nonzero(u != f)[0].astype(np.int64)


def build_uniward_bit_allocated_chroma_lut(
    *,
    rgb_pairs: np.ndarray,
    class_labels: np.ndarray,
    base_lut: np.ndarray,
    total_budget_bits: int,
    top_k: int,
    uniward_cost_map: np.ndarray | None = None,
    min_bits_per_byte: int = 2,
    grayscale_levels: int = GRAYSCALE_LEVELS_DEFAULT,
    num_segnet_classes: int = NUM_SEGNET_CLASSES,
    archive_sha256: str | None = None,
    captured_at_utc: str | None = None,
) -> PerLutByteBitAllocationResult:
    """End-to-end: UNIWARD cost -> allocate_per_byte -> quantized v8 LUT.

    This is the canonical orphan-loop #1570 wire: it computes (or consumes) the
    UNIWARD cost map on real frames, aggregates it into the v8 (level, class)
    bins, inverts to per-LUT-byte sensitivity, allocates the bit budget via the
    canonical ``tac.bit_allocator.per_byte`` TOP_K_BY_SENSITIVITY allocator, and
    quantizes the base LUT to the allocated bit-depths.

    Parameters
    ----------
    rgb_pairs : np.ndarray, shape (N, 3, H, W), dtype=uint8
        Compress-time GT RGB frames.
    class_labels : np.ndarray, shape (N, H, W), dtype=uint8
        SegNet argmax labels per pixel.
    base_lut : np.ndarray, shape (levels, classes, 3), dtype=uint8
        The canonical (unweighted-median) v8 LUT to coarsen. Built via
        ``tac.substrates.nscs06_v8_chroma_lut.architecture.build_chroma_lut_from_ground_truth``.
    total_budget_bits : int
        Total bit budget across all LUT bytes. The full-precision budget is
        ``n_lut_bytes * 8``; a budget below that forces coarsening.
    top_k : int
        Number of most-sensitive LUT bytes to keep at the 8-bit cap.
    uniward_cost_map : np.ndarray | None
        Precomputed per-pixel UNIWARD cost (N, H, W). If None, computed via
        ``compute_uniward_cost_map_for_frames``.
    min_bits_per_byte : int
        Floor bit-depth for EVERY LUT byte (default 2 = 4-level chroma). UNIWARD
        spends the premium budget above this floor on high-sensitivity bytes;
        scorer-blind bytes are gracefully coarsened to the floor rather than
        destroyed. A 0 floor (full destruction of low-sensitivity bytes) is
        empirically catastrophic for reconstruction — see the smoke memo.
    archive_sha256, captured_at_utc : forwarded to the allocator for Provenance.

    Returns
    -------
    PerLutByteBitAllocationResult
        Carries the quantized LUT, the per-byte bit array, the canonical plan,
        and the per-byte sensitivity.
    """
    if base_lut.shape != (grayscale_levels, num_segnet_classes, 3):
        raise ValueError(
            f"base_lut shape {base_lut.shape} != "
            f"({grayscale_levels}, {num_segnet_classes}, 3)"
        )
    n_lut_bytes = int(grayscale_levels * num_segnet_classes * 3)

    if uniward_cost_map is None:
        uniward_cost_map = compute_uniward_cost_map_for_frames(rgb_pairs)

    per_bin = aggregate_per_pixel_uniward_weights_into_lut_bins(
        rgb_pairs=rgb_pairs,
        class_labels=class_labels,
        per_pixel_uniward_weight=uniward_cost_map.astype(np.float32),
        grayscale_levels=grayscale_levels,
        num_segnet_classes=num_segnet_classes,
    )
    sensitivity = per_lut_byte_sensitivity_from_uniward_weights(per_bin)

    plan = allocate_lut_bits_uniward_weighted(
        sensitivity,
        total_budget_bits=int(total_budget_bits),
        top_k=int(top_k),
        archive_sha256=archive_sha256,
        captured_at_utc=captured_at_utc,
    )
    bits = _bits_per_byte_array(
        plan, n_lut_bytes, min_bits_per_byte=int(min_bits_per_byte)
    )
    lut_q = quantize_lut_by_allocation(base_lut, bits)

    return PerLutByteBitAllocationResult(
        lut_quantized=lut_q,
        bits_per_lut_byte=bits,
        plan=plan,
        per_lut_byte_sensitivity=sensitivity,
        method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY.value,
        grayscale_levels=int(grayscale_levels),
        num_segnet_classes=int(num_segnet_classes),
    )


def build_canonical_provenance_for_bit_allocation(
    *,
    result: PerLutByteBitAllocationResult,
    total_budget_bits: int,
    top_k: int,
) -> dict:
    """Build canonical Provenance per Catalog #323 + Catalog #341 markers.

    Every per-LUT-byte bit-allocation result carries non-promotable markers so
    the surface cannot leak into score/promotion signals. Sister of
    ``weight_map_per_lut_index.build_canonical_provenance_for_per_lut_index_aggregation``.
    """
    return {
        "integration_id": BIT_ALLOCATION_INTEGRATION_NAME,
        "integration_version": BIT_ALLOCATION_INTEGRATION_VERSION,
        "consumed_substrate_id": "nscs06_v8_chroma_lut",
        "consumed_substrate_scope": "read_only_consumer_import",
        "canonical_equation_id_proposed": CANONICAL_EQUATION_ID_PROPOSED,
        "n_lut_bytes": result.n_lut_bytes,
        "total_budget_bits": int(total_budget_bits),
        "top_k": int(top_k),
        "n_bytes_at_full_precision": result.n_bytes_at_full_precision,
        "n_bytes_coarsened": result.n_bytes_coarsened,
        "allocation_method": result.method,
        # Canonical Provenance non-promotable markers per Catalog #341.
        "evidence_grade": "macOS-MLX research-signal",
        "score_claim": False,
        "promotable": False,
        "axis_tag": "[predicted]",
        "hardware_substrate_recommendation": "darwin_arm64_m5_max_mlx_local",
        "measurement_axis": "[macOS-MLX research-signal]",
        # Sister hooks per Catalog #125 (bit-allocator PRIMARY hook #3).
        "hook_numbers_fired": [1, 3, 5],
        "entropy_position": "P3_entropy_coded_sidecar_per_lut_byte_bit_allocation",
        # Sister-disjoint discipline per Catalog #230.
        "nscs06_v8_substrate_modification_scope": "none_read_only_consumer_import",
    }
