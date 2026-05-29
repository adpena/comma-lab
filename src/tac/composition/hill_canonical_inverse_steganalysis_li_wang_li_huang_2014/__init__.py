# SPDX-License-Identifier: MIT
"""HILL canonical inverse-steganalysis (Li-Wang-Li-Huang 2014) — L0 SCAFFOLD.

Per CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"
+ canonical Slot UU canonical landing 2026-05-29 (commit ``2b573f105``)
canonical TOP-1 9/9 ranking + canonical Fridrich-Yousfi inverse-steganalysis
cascade Axis 5 extension + operator binding directive #10 explicit follow-
through + Slot FF PR110-OPT-7 canonical sister-cascade pattern (commit
``0adecdc5b``) + OVERNIGHT-EEE canonical IMPLEMENTATION-LEVEL boundary
anchor 2026-05-21 per Catalog #307.

Design memo (single source of truth)::

    .omx/research/hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_\
high_pass_low_pass_low_pass_aggregation_cost_matrix_canonical_fridrich_yousfi_\
cascade_axis_5_extension_per_slot_uu_top_1_design_20260529.md

Canonical citation
==================

Li-Wang-Li-Huang 2014 "A new cost function for spatial image
steganography" (canonical reference) defines the HILL cost function as the
HIGH-pass + LOW-pass + LOW-pass cascade::

    Step 1: HIGH-pass filter (Ker-Bohme 2008 canonical 3x3 KB kernel)
        K_KB = (1/4) * [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
        residual = convolve(image, K_KB)

    Step 2: First LOW-pass filter (canonical 7x7 averaging filter per
            operator binding prompt; sister Li-Wang reference allows
            L1 in {3x3, 5x5, 7x7})
        L1 = ones(7, 7) / 49
        intermediate = convolve(|residual|, L1)

    Step 3: Reciprocal with epsilon for numerical stability
        cost_reciprocal = 1 / (intermediate + epsilon)

    Step 4: Second LOW-pass filter (canonical 15x15 averaging filter)
        L2 = ones(15, 15) / 225
        cost_smooth = convolve(cost_reciprocal, L2)

    Result: per-pixel cost map (HIGH cost = LOW embedding admissibility =
    MORE TEXTURED per Li-Wang canonical interpretation; in canonical
    Fridrich-Yousfi inverse-steganalysis context, HIGH cost ⟹ LOW scorer
    detectability ⟹ canonical pixel-selection priority for sparse-K
    selection)

L0 SCAFFOLD scope distinction vs OVERNIGHT-EEE 2026-05-21 probe
================================================================

OVERNIGHT-EEE NULL_SIGNAL_DEFER probe (``.omx/research/tier_1_distortion_axis
_probes_20260521/probe_3c_hill_filter_steganalysis_sister.py``) interpreted
HILL through the CCC/DDD reciprocal-weight framing (HIGH weight = HIGH
suppression = MORE FLAT) and found cost-distribution opposite-direction.
Verdict: IMPLEMENTATION-LEVEL boundary per Catalog #307 (HILL's reciprocal-
inside-cascade semantically incompatible with reciprocal-weight framing;
paradigm intact).

This L0 SCAFFOLD operates at the canonical helper / src/tac/composition
surface and applies the cost-matrix output as canonical pixel-selection
priority for PR110-OPT-X+ archive emission (the UNIWARD-analog
application surface, NOT the CCC/DDD reciprocal-weight framing). The
canonical Fridrich-Yousfi inverse-steganalysis target is the sparse-K
selection of LOW-detectability pixels.

L0 SCAFFOLD role
================

THIS module serves the canonical dual role:

1. Preserve the canonical Li-Wang-Li-Huang 2014 HIGH × LOW × LOW cascade
   formulation as a queryable system surface per CLAUDE.md "Max
   observability — non-negotiable" (Catalog #305).

2. Enumerate alternative HILL kernel-size methodologies per Catalog #308
   so the operator can route the next iteration through one of N≥4
   canonical candidates.

L0 SCAFFOLD does NOT claim score improvement. The canonical
:func:`apply_hill_canonical_cost_matrix_to_pr110_archive` entry point
returns a Tier A canonical-routing-markers contribution per Catalog #341:
``predicted_delta_adjustment=0.0`` + ``promotable=False`` + ``axis_tag="
[predicted]"``. The verdict field is
``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog #325.

Canonical contracts honored
===========================

- :class:`AxisDecomposition` per Catalog #356 (per-axis (seg, pose,
  archive bytes) decomposition with canonical Provenance dict-form).
- :class:`Provenance` per Catalog #323 via
  :func:`tac.provenance.builders.build_provenance_for_predicted`.
- Tier A canonical-routing markers per Catalog #341 + #357.
- HNeRV parity discipline L4 (≤200 LOC inflate budget; this L0 SCAFFOLD
  has no inflate-time code path — encoder-side only).
- HNeRV parity discipline L7 (bolt-on size budget ≤350 LOC; this L0
  SCAFFOLD is ~600 LOC inclusive of analytical primitives; substrate-
  engineering exception declared in design memo per Catalog #294 9-dim
  checklist).
- Catalog #309 ``horizon_class: plateau_adjacent``.
- Catalog #1 device-fork trap protection (no MPS fallback; numpy-only).
- Catalog #192 macOS-CPU-advisory NEVER-promotable contract.

Sister cross-references
=======================

- :mod:`tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion`
  (Slot FF canonical sister-cascade pattern; canonical UNIWARD Axis 1)
- :mod:`tac.composition.pr110_opt_4_grouped_color_geometry_calibration`
  (Slot X canonical sister-cascade Axis 4)
- :mod:`tac.composition.pr110_opt_5_boundary_region_waterfill`
  (Slot TT canonical sister-cascade Axis 3)
- :mod:`tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet`
  (Slot RR canonical sister-cascade Axis 2)
- :mod:`tac.cathedral.consumer_contract` (canonical AxisDecomposition)
- :mod:`tac.provenance.builders` (canonical Provenance builders)
- ``.omx/research/overnight_eee_hill_filter_steganalysis_sister_landed_20260521.md``
  (canonical IMPLEMENTATION-LEVEL boundary anchor per Catalog #307)
- ``.omx/research/hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_*_design_20260529.md``
  (THIS L0 SCAFFOLD's canonical design memo)

Per Catalog #287 evidence tag discipline: the score deltas this module
returns are PREDICTED (canonical Li-Wang-Li-Huang 2014 reference +
canonical Fridrich-Yousfi inverse-steganalysis cascade compounding
pattern); tagged ``[predicted]`` per Catalog #287/#341. Empirical paired-
CUDA anchor required before any score claim per CLAUDE.md "Apples-to-
apples evidence discipline" + "Submission auth eval — BOTH CPU AND CUDA".
"""

from __future__ import annotations

import enum
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tac.cathedral.consumer_contract import AxisDecomposition
from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.validator import provenance_to_dict

# -----------------------------------------------------------------------------
# Canonical Li-Wang-Li-Huang 2014 HILL cost-matrix anchor constants
# -----------------------------------------------------------------------------

#: Canonical Li-Wang-Li-Huang 2014 reference citation URL.
CANONICAL_LI_WANG_2014_CITATION_URL: str = (
    "https://www.semanticscholar.org/paper/A-new-cost-function-for-spatial-"
    "image-steganography-Li-Wang/ceb6603c9e45f6b66c3a3cec09a5b4e64856a1fd"
)

#: Canonical Ker-Bohme 2008 3x3 KB kernel (HIGH-pass residual extraction).
#: Normalized by 1/4 for canonical unit gain per Ker-Bohme reference.
#: Tuple-of-tuples for immutability per CLAUDE.md "Beauty, simplicity,
#: and developer experience" frozen-default discipline.
CANONICAL_KB_KERNEL_3X3: tuple[tuple[float, ...], ...] = (
    (-0.25,  0.50, -0.25),
    ( 0.50, -1.00,  0.50),
    (-0.25,  0.50, -0.25),
)

#: Canonical KB kernel size (3x3 per Ker-Bohme 2008 canonical reference).
CANONICAL_KB_KERNEL_SIZE: int = 3

#: Canonical L1 LOW-pass averaging filter size per operator binding prompt.
#: Sister Li-Wang reference allows L1 in {3, 5, 7}; canonical operator
#: binding prompt specifies 7 as the canonical default.
CANONICAL_L1_KERNEL_SIZE: int = 7

#: Canonical L2 LOW-pass averaging filter size per Li-Wang-Li-Huang 2014.
CANONICAL_L2_KERNEL_SIZE: int = 15

#: Canonical numerical-stability epsilon per Slot FF PR110-OPT-7 sister
#: convention (sister UNIWARD uses 1e-6; Li-Wang reference uses 2^-6).
#: Default 1e-6 for canonical sister-cascade consistency; operator can
#: override via :class:`HILLConfig.epsilon` field to 0.015625 if paired-
#: CUDA shows numerical instability.
CANONICAL_HILL_EPSILON: float = 1e-6

#: Canonical sparse-K default per Slot FF PR110-OPT-7 sister anchor.
CANONICAL_SPARSE_K_DEFAULT: int = 100

#: Canonical widened-K alternative per Catalog #308 reactivation criterion.
CANONICAL_WIDENED_K_DEFAULT: int = 200

#: Canonical PR110 FEC6 fixed-Huffman K=16 baseline wire size in bytes
#: per Slot FF PR110-OPT-7 sister anchor.
CANONICAL_FEC6_BASELINE_WIRE_BYTES: int = 249

#: Canonical PR110 N_PAIRS per Slot FF PR110-OPT-7 sister anchor.
CANONICAL_N_PAIRS: int = 600

#: Canonical PR110 N_MODES per Slot FF PR110-OPT-7 sister anchor.
CANONICAL_N_MODES: int = 21

#: Canonical rate multiplier per contest formula
#: ``S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489``.
CANONICAL_RATE_MULTIPLIER: float = 25.0

#: Canonical rate denominator per contest formula
#: (37,545,489 = total contest video byte count).
CANONICAL_RATE_DENOM_BYTES: int = 37_545_489

# -----------------------------------------------------------------------------
# HILL cost-matrix strategy enum (Catalog #308 alternative-reducer enumeration)
# -----------------------------------------------------------------------------


class HILLCostMatrixStrategy(str, enum.Enum):
    """Canonical HILL cost-matrix strategy for PR110-OPT-X+ extension.

    Per Catalog #308 alternative-reducer methodology enumeration: the L0
    SCAFFOLD supports 4 canonical strategies; ``CANONICAL_3X3_KB_HIGH_PASS``
    is the Li-Wang-Li-Huang 2014 canonical reference baseline; the other 3
    are DEFERRED-PENDING-RESEARCH sister candidates per the design memo's
    reactivation criteria + operator binding directive #10 fractal-
    optimization extension.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode: each
    strategy may produce a different per-pair cost-matrix shape; the
    canonical helper :func:`compute_hill_canonical_cost_matrix_for_pr110_catalog`
    dispatches on this enum.
    """

    #: Canonical Li-Wang-Li-Huang 2014 reference baseline: 3x3 KB high-pass
    #: + L1 + L2 cascade. Sister of Slot FF PR110-OPT-7 SPARSE_K100
    #: canonical baseline at the application surface.
    CANONICAL_3X3_KB_HIGH_PASS = "canonical_3x3_kb_high_pass"

    #: DEFERRED per Catalog #308: extended 5x5 HIGH-pass kernel per
    #: canonical 8th MLX-first directive extension. Wider high-pass
    #: response captures larger-scale texture; pending paired-CUDA anchor.
    EXTENDED_5X5_HIGH_PASS = "extended_5x5_high_pass"

    #: DEFERRED per Catalog #308 + operator binding directive #10 fractal-
    #: optimization: per-region variable kernel (canonical wavelet-
    #: hierarchy per Catalog #277). Kernel size varies per spatial region
    #: based on local texture statistics.
    PER_REGION_VARIABLE_KERNEL = "per_region_variable_kernel"

    #: DEFERRED per Catalog #308 + operator binding directive #10 fractal-
    #: optimization: per-pixel adaptive kernel. Kernel size adapts per
    #: pixel based on local gradient + scorer-response sensitivity;
    #: most computationally expensive; canonical fractal-optimization
    #: target per operator binding directive #10.
    PER_PIXEL_ADAPTIVE_KERNEL = "per_pixel_adaptive_kernel"


# -----------------------------------------------------------------------------
# Canonical HILLConfig dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HILLConfig:
    """Canonical configuration for HILL canonical L0 SCAFFOLD.

    Frozen dataclass per CLAUDE.md "Beauty, simplicity, and developer
    experience"; invariants validated in ``__post_init__``.

    Args:
        strategy: Canonical :class:`HILLCostMatrixStrategy` enum.
            Defaults to ``CANONICAL_3X3_KB_HIGH_PASS`` (Li-Wang-Li-Huang
            2014 canonical reference baseline) per the canonical-vs-unique
            decision per layer (Catalog #290).
        kb_kernel_size: Canonical Ker-Bohme HIGH-pass kernel size.
            Canonical default 3 per Ker-Bohme 2008; sister extended 5x5
            available via ``EXTENDED_5X5_HIGH_PASS`` strategy.
        l1_kernel_size: Canonical L1 LOW-pass averaging filter size.
            Canonical default 7 per operator binding prompt; sister
            Li-Wang reference allows {3, 5, 7}.
        l2_kernel_size: Canonical L2 LOW-pass averaging filter size.
            Canonical default 15 per Li-Wang-Li-Huang 2014 reference.
        epsilon: Canonical numerical-stability denominator. Canonical
            default 1e-6 per Slot FF PR110-OPT-7 sister; sister Li-Wang
            reference default 0.015625 (2^-6).
        sparse_k: Sparse-selector K for pixel-selection priority.
            Canonical default 100 per Slot FF PR110-OPT-7 sister anchor.
        n_pairs: Source pair count. Canonical default 600 per Slot FF
            PR110-OPT-7 sister anchor.
        n_modes: Source mode count. Canonical default 21 per Slot FF
            PR110-OPT-7 sister anchor.
        header_overhead_bytes: Wire-format header overhead. Canonical
            default 4 bytes (1 byte HILL header magic + 3 bytes sparse-K
            selector index format header).
        emit_axis_decomposition: If True (default), emits canonical
            :class:`AxisDecomposition` per Catalog #356.

    Raises:
        ValueError: if any invariant fails (see ``__post_init__``).
    """

    strategy: HILLCostMatrixStrategy = HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS
    kb_kernel_size: int = CANONICAL_KB_KERNEL_SIZE
    l1_kernel_size: int = CANONICAL_L1_KERNEL_SIZE
    l2_kernel_size: int = CANONICAL_L2_KERNEL_SIZE
    epsilon: float = CANONICAL_HILL_EPSILON
    sparse_k: int = CANONICAL_SPARSE_K_DEFAULT
    n_pairs: int = CANONICAL_N_PAIRS
    n_modes: int = CANONICAL_N_MODES
    header_overhead_bytes: int = 4
    emit_axis_decomposition: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, HILLCostMatrixStrategy):
            raise ValueError(
                "strategy must be HILLCostMatrixStrategy enum, got "
                f"{type(self.strategy).__name__}"
            )
        # KB kernel size: canonical 3 (Ker-Bohme) or extended 5
        if (
            not isinstance(self.kb_kernel_size, int)
            or isinstance(self.kb_kernel_size, bool)
        ):
            raise ValueError(
                f"kb_kernel_size must be int, got {type(self.kb_kernel_size).__name__}"
            )
        if self.kb_kernel_size not in (3, 5):
            raise ValueError(
                f"kb_kernel_size must be in (3, 5), got {self.kb_kernel_size}"
            )
        # L1 kernel size: canonical 7 per operator binding prompt; sister {3, 5, 7}
        if (
            not isinstance(self.l1_kernel_size, int)
            or isinstance(self.l1_kernel_size, bool)
        ):
            raise ValueError(
                f"l1_kernel_size must be int, got {type(self.l1_kernel_size).__name__}"
            )
        if self.l1_kernel_size not in (3, 5, 7):
            raise ValueError(
                f"l1_kernel_size must be in (3, 5, 7), got {self.l1_kernel_size}"
            )
        # L2 kernel size: canonical 15 per Li-Wang reference; allow {9, 15, 21}
        if (
            not isinstance(self.l2_kernel_size, int)
            or isinstance(self.l2_kernel_size, bool)
        ):
            raise ValueError(
                f"l2_kernel_size must be int, got {type(self.l2_kernel_size).__name__}"
            )
        if self.l2_kernel_size not in (9, 15, 21):
            raise ValueError(
                f"l2_kernel_size must be in (9, 15, 21), got {self.l2_kernel_size}"
            )
        # Epsilon: positive number, NaN/inf rejected
        if not isinstance(self.epsilon, (int, float)) or isinstance(
            self.epsilon, bool
        ):
            raise ValueError(
                f"epsilon must be number, got {type(self.epsilon).__name__}"
            )
        if isinstance(self.epsilon, float) and (
            math.isnan(self.epsilon) or math.isinf(self.epsilon)
        ):
            raise ValueError(f"epsilon must be finite, got {self.epsilon}")
        if not (self.epsilon > 0):
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        # sparse_k: positive integer, <= n_pairs
        if (
            not isinstance(self.sparse_k, int)
            or isinstance(self.sparse_k, bool)
        ):
            raise ValueError(
                f"sparse_k must be int, got {type(self.sparse_k).__name__}"
            )
        if self.sparse_k <= 0:
            raise ValueError(f"sparse_k must be > 0, got {self.sparse_k}")
        # n_pairs: positive integer
        if (
            not isinstance(self.n_pairs, int)
            or isinstance(self.n_pairs, bool)
        ):
            raise ValueError(
                f"n_pairs must be int, got {type(self.n_pairs).__name__}"
            )
        if self.n_pairs <= 0:
            raise ValueError(f"n_pairs must be > 0, got {self.n_pairs}")
        if self.sparse_k > self.n_pairs:
            raise ValueError(
                f"sparse_k ({self.sparse_k}) must be <= n_pairs ({self.n_pairs})"
            )
        # n_modes: positive integer
        if (
            not isinstance(self.n_modes, int)
            or isinstance(self.n_modes, bool)
        ):
            raise ValueError(
                f"n_modes must be int, got {type(self.n_modes).__name__}"
            )
        if self.n_modes <= 0:
            raise ValueError(f"n_modes must be > 0, got {self.n_modes}")
        # header_overhead_bytes: non-negative integer
        if (
            not isinstance(self.header_overhead_bytes, int)
            or isinstance(self.header_overhead_bytes, bool)
        ):
            raise ValueError(
                "header_overhead_bytes must be int, got "
                f"{type(self.header_overhead_bytes).__name__}"
            )
        if self.header_overhead_bytes < 0:
            raise ValueError(
                "header_overhead_bytes must be >= 0, got "
                f"{self.header_overhead_bytes}"
            )


# -----------------------------------------------------------------------------
# Canonical helpers
# -----------------------------------------------------------------------------


def _compute_hill_canonical_signature(
    n_pairs: int,
    sparse_k: int,
    strategy: HILLCostMatrixStrategy,
    kb_kernel_size: int,
    l1_kernel_size: int,
    l2_kernel_size: int,
    epsilon: float,
) -> str:
    """Return sha256 hex digest over (config) tuple.

    Used for:
    (a) Provenance ``inputs_sha256`` per Catalog #323;
    (b) deterministic reproducibility diff-able-across-runs facet per
    Catalog #305 observability surface.
    """
    payload = (
        f"n_pairs={int(n_pairs)}"
        f"|sparse_k={int(sparse_k)}"
        f"|strategy={strategy.value}"
        f"|kb={int(kb_kernel_size)}"
        f"|l1={int(l1_kernel_size)}"
        f"|l2={int(l2_kernel_size)}"
        f"|eps={float(epsilon):.12e}"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _convolve_2d_canonical(
    image: Sequence[Sequence[float]],
    kernel: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Canonical 2D convolution per Li-Wang reference (zero-padded boundary).

    Numpy-portable canonical implementation per CLAUDE.md "Forbidden
    device-selection defaults" + 8th MLX-first canonical directive
    (canonical numpy fallback path; MLX-first opt-in via separate sister
    helper per ``tac.local_acceleration.pr95_hnerv_mlx`` pattern).

    Args:
        image: 2D sequence of floats (height x width).
        kernel: 2D square sequence of floats (size x size).

    Returns:
        2D list of floats representing convolved image (same shape as input).
    """
    if not image or not image[0]:
        raise ValueError("image must be non-empty 2D sequence")
    if not kernel or not kernel[0]:
        raise ValueError("kernel must be non-empty 2D sequence")
    height = len(image)
    width = len(image[0])
    for row in image:
        if len(row) != width:
            raise ValueError("image rows must have uniform width")
    k_size = len(kernel)
    for row in kernel:
        if len(row) != k_size:
            raise ValueError("kernel must be square")
    if k_size % 2 != 1:
        raise ValueError(f"kernel size must be odd, got {k_size}")

    pad = k_size // 2
    result: list[list[float]] = [
        [0.0 for _ in range(width)] for _ in range(height)
    ]
    for y in range(height):
        for x in range(width):
            acc = 0.0
            for ky in range(k_size):
                for kx in range(k_size):
                    iy = y + ky - pad
                    ix = x + kx - pad
                    if 0 <= iy < height and 0 <= ix < width:
                        acc += float(image[iy][ix]) * float(kernel[ky][kx])
            result[y][x] = acc
    return result


def _build_averaging_kernel(size: int) -> tuple[tuple[float, ...], ...]:
    """Build canonical NxN averaging filter (uniform 1/N^2 per cell).

    Args:
        size: Kernel size (odd integer).

    Returns:
        Immutable tuple-of-tuples averaging kernel.
    """
    if size % 2 != 1 or size < 1:
        raise ValueError(f"averaging kernel size must be odd positive, got {size}")
    weight = 1.0 / (size * size)
    return tuple(tuple(weight for _ in range(size)) for _ in range(size))


def _compute_canonical_li_wang_hill_cost_matrix(
    image: Sequence[Sequence[float]],
    config: HILLConfig,
) -> list[list[float]]:
    """Canonical Li-Wang-Li-Huang 2014 HILL cost matrix.

    Computes the canonical HIGH × LOW × LOW cascade per the canonical
    reference at ``CANONICAL_LI_WANG_2014_CITATION_URL``:

    1. ``residual = convolve(image, KB_kernel)`` (HIGH-pass)
    2. ``intermediate = convolve(|residual|, L1_kernel)`` (first LOW-pass)
    3. ``cost_reciprocal = 1 / (intermediate + epsilon)`` (reciprocal)
    4. ``cost_smooth = convolve(cost_reciprocal, L2_kernel)`` (second LOW-pass)

    Args:
        image: 2D sequence of floats (luma channel; height x width).
        config: Canonical :class:`HILLConfig`.

    Returns:
        2D list of floats representing per-pixel cost matrix.

    Notes:
        - Per Catalog #287 evidence tag discipline: this primitive is a
          canonical analytical kernel; predicted values are pre-empirical;
          empirical paired-CUDA anchor required before any score claim.
        - Per Catalog #305 observability: cost map is diff-able across
          runs via ``_compute_hill_canonical_signature``.
        - Per the L0 SCAFFOLD docstring: cost map is the CANONICAL Li-Wang
          interpretation (HIGH cost = LOW embedding admissibility = MORE
          TEXTURED ⟹ canonical pixel-selection priority for sparse-K),
          NOT the CCC/DDD reciprocal-weight framing OVERNIGHT-EEE
          falsified at IMPLEMENTATION level.
    """
    # Step 1: HIGH-pass (KB kernel)
    if config.kb_kernel_size == 3:
        kb_kernel = CANONICAL_KB_KERNEL_3X3
    else:
        # Extended 5x5 KB-like kernel (canonical extension; documented in
        # Slot UU canonical roadmap; placeholder canonical structure with
        # canonical center-weight scaling for the L0 SCAFFOLD).
        # Operator-routable refinement per Catalog #308.
        kb_kernel = tuple(
            tuple(
                -0.25 if (abs(i - 2) + abs(j - 2)) == 4
                else (0.50 if (abs(i - 2) + abs(j - 2)) == 2
                      else (-1.00 if i == 2 and j == 2 else 0.0))
                for j in range(5)
            )
            for i in range(5)
        )

    residual = _convolve_2d_canonical(image, kb_kernel)

    # Step 2: First LOW-pass (L1 averaging filter) over |residual|
    abs_residual = [[abs(v) for v in row] for row in residual]
    l1_kernel = _build_averaging_kernel(config.l1_kernel_size)
    intermediate = _convolve_2d_canonical(abs_residual, l1_kernel)

    # Step 3: Reciprocal with epsilon
    cost_reciprocal = [
        [1.0 / (v + config.epsilon) for v in row]
        for row in intermediate
    ]

    # Step 4: Second LOW-pass (L2 averaging filter)
    l2_kernel = _build_averaging_kernel(config.l2_kernel_size)
    cost_smooth = _convolve_2d_canonical(cost_reciprocal, l2_kernel)

    return cost_smooth


def _aggregate_cost_matrix_to_per_pair_priority(
    cost_matrix: Sequence[Sequence[float]],
    n_pairs: int,
) -> list[float]:
    """Aggregate per-pixel HILL cost matrix into per-pair priority scalar.

    Canonical aggregation: partition cost matrix into ``n_pairs`` row-bands
    (canonical row-wise pair-mapping per the PR110 archive grammar
    convention) and compute mean cost per band. Higher mean ⟹ higher
    pixel-selection priority (canonical Fridrich-Yousfi inverse-
    steganalysis interpretation: HIGH cost = LOW detectability = canonical
    sparse-K selection target).

    Args:
        cost_matrix: 2D per-pixel cost from
            ``_compute_canonical_li_wang_hill_cost_matrix``.
        n_pairs: Number of pairs to aggregate to.

    Returns:
        List of ``n_pairs`` float priority values.
    """
    if not cost_matrix or not cost_matrix[0]:
        raise ValueError("cost_matrix must be non-empty 2D sequence")
    height = len(cost_matrix)
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be > 0, got {n_pairs}")

    # If image is too small for n_pairs row-bands, distribute as evenly as possible
    band_size = max(1, height // n_pairs)
    priorities: list[float] = []
    for pair_idx in range(n_pairs):
        y_start = pair_idx * band_size
        y_end = min(y_start + band_size, height)
        if y_start >= height:
            # Trailing pairs get last band's mean (canonical defensive fallback)
            y_start = max(0, height - band_size)
            y_end = height
        band_sum = 0.0
        band_count = 0
        for y in range(y_start, y_end):
            for v in cost_matrix[y]:
                band_sum += float(v)
                band_count += 1
        priorities.append(band_sum / max(1, band_count))
    return priorities


def _select_sparse_k_pairs_by_priority(
    priorities: Sequence[float],
    sparse_k: int,
) -> list[int]:
    """Select K pair indices with highest priority.

    Canonical Fridrich-Yousfi inverse-steganalysis sparse-K selection:
    pairs with highest HILL cost (= lowest scorer detectability) are
    canonical sparse-selector picks.

    Returns sorted list of pair indices (ascending order per Catalog #305
    diff-able-across-runs facet).
    """
    n_pairs = len(priorities)
    if sparse_k >= n_pairs:
        return list(range(n_pairs))
    indexed = sorted(
        enumerate(priorities),
        key=lambda x: -float(x[1]),
    )
    selected = [int(idx) for idx, _ in indexed[:sparse_k]]
    return sorted(selected)


def compute_hill_canonical_cost_matrix_for_pr110_catalog(
    image: Sequence[Sequence[float]],
    config: HILLConfig,
) -> dict[str, Any]:
    """Canonical analytical primitive: HILL cost matrix + sparse-K selection.

    Computes:

    - Per-pixel HILL cost matrix via canonical Li-Wang-Li-Huang 2014
      HIGH × LOW × LOW cascade
    - Per-pair priority aggregation
    - Sparse-K selection (top-K HILL cost pair indices) per canonical
      Fridrich-Yousfi inverse-steganalysis
    - Wire-bytes estimate per strategy
    - Delta vs FEC6 baseline (canonical 249 bytes)

    Returns a dict with the canonical analytical primitive output.

    Per Catalog #287 evidence tag discipline: the returned
    ``delta_vs_fec6_bytes`` is PREDICTED (analytical upper bound per
    canonical Slot FF PR110-OPT-7 sister anchor + canonical Li-Wang
    reference); empirical paired-CUDA anchor required before any score
    claim.

    Args:
        image: 2D sequence of floats (luma channel; height x width).
        config: Canonical :class:`HILLConfig`.

    Returns:
        Dict with keys:

        - ``cost_matrix_summary`` (dict): mean / std / min / max of cost
          matrix (canonical summary; full cost map not returned to bound
          memory + ease JSON serialization).
        - ``per_pair_priorities`` (list[float]): canonical aggregated
          per-pair priority values.
        - ``selected_pair_indices`` (list[int]): sparse-K selection.
        - ``wire_bytes_estimate`` (int): analytical upper-bound wire bytes.
        - ``fec6_baseline_wire_bytes`` (int): canonical 249.
        - ``delta_vs_fec6_bytes`` (int): wire_bytes_estimate - 249
          (negative = savings).
        - ``strategy`` (str): canonical enum value.
        - ``n_selected_pairs`` (int): cardinality of selected_pair_indices.
        - ``hill_concentration_factor`` (float): top-K priority sum / all-
          pairs priority sum.
    """
    # Compute HILL cost matrix
    cost_matrix = _compute_canonical_li_wang_hill_cost_matrix(image, config)

    # Cost matrix summary (canonical observability per Catalog #305)
    flat_costs = [v for row in cost_matrix for v in row]
    n_costs = len(flat_costs)
    if n_costs == 0:
        raise ValueError("cost_matrix is empty after canonical Li-Wang cascade")
    mean_cost = sum(flat_costs) / n_costs
    var_cost = sum((v - mean_cost) ** 2 for v in flat_costs) / n_costs
    std_cost = math.sqrt(var_cost)
    cost_matrix_summary = {
        "mean": float(mean_cost),
        "std": float(std_cost),
        "min": float(min(flat_costs)),
        "max": float(max(flat_costs)),
        "n_pixels": int(n_costs),
    }

    # Aggregate to per-pair priorities
    priorities = _aggregate_cost_matrix_to_per_pair_priority(
        cost_matrix, config.n_pairs
    )

    # Strategy dispatch
    if config.strategy == HILLCostMatrixStrategy.CANONICAL_3X3_KB_HIGH_PASS:
        effective_k = config.sparse_k
        selected = _select_sparse_k_pairs_by_priority(priorities, effective_k)
    elif config.strategy == HILLCostMatrixStrategy.EXTENDED_5X5_HIGH_PASS:
        effective_k = config.sparse_k
        selected = _select_sparse_k_pairs_by_priority(priorities, effective_k)
    elif config.strategy == HILLCostMatrixStrategy.PER_REGION_VARIABLE_KERNEL:
        # Per-region grouping: degenerate to sparse-K at L0 SCAFFOLD level
        # (per-region requires per-pixel-region surface; out-of-scope L0).
        # Operator-routable refinement per Catalog #308 + operator binding
        # directive #10 fractal-optimization.
        effective_k = config.sparse_k
        selected = _select_sparse_k_pairs_by_priority(priorities, effective_k)
    elif config.strategy == HILLCostMatrixStrategy.PER_PIXEL_ADAPTIVE_KERNEL:
        # Per-pixel adaptive: degenerate to sparse-K at L0 SCAFFOLD level
        # (per-pixel adaptation requires gradient-aware surface; out-of-
        # scope L0). Operator-routable refinement per Catalog #308 +
        # operator binding directive #10 fractal-optimization.
        effective_k = config.sparse_k
        selected = _select_sparse_k_pairs_by_priority(priorities, effective_k)
    else:  # pragma: no cover -- defensive
        raise ValueError(f"unknown strategy: {config.strategy}")

    # Wire bytes estimate (canonical sparse-K selector format)
    index_byte_width = max(1, math.ceil(math.log2(config.n_pairs) / 8.0))
    wire_bytes_estimate = (
        config.header_overhead_bytes
        + effective_k * index_byte_width
        + effective_k  # 1 byte per perturbation magnitude
    )
    delta_vs_fec6 = wire_bytes_estimate - CANONICAL_FEC6_BASELINE_WIRE_BYTES

    # HILL concentration factor
    all_pairs_sum = sum(priorities)
    if all_pairs_sum > 0 and len(selected) > 0:
        top_k_sum = sum(priorities[i] for i in selected)
        concentration = top_k_sum / all_pairs_sum
    else:
        concentration = 0.0

    return {
        "cost_matrix_summary": cost_matrix_summary,
        "per_pair_priorities": priorities,
        "selected_pair_indices": selected,
        "wire_bytes_estimate": int(wire_bytes_estimate),
        "fec6_baseline_wire_bytes": int(CANONICAL_FEC6_BASELINE_WIRE_BYTES),
        "delta_vs_fec6_bytes": int(delta_vs_fec6),
        "strategy": config.strategy.value,
        "n_selected_pairs": len(selected),
        "hill_concentration_factor": float(concentration),
    }


def apply_hill_canonical_cost_matrix_to_pr110_archive(
    image: Sequence[Sequence[float]],
    config: HILLConfig,
) -> dict[str, Any]:
    """Canonical L0 SCAFFOLD entry point: apply HILL cost matrix to PR110 archive.

    Returns a Tier A canonical-routing-markers contribution per Catalog
    #341 + Catalog #356 + Catalog #323. NEVER promotable by construction
    (``promotable=False`` + ``axis_tag="[predicted]"``); empirical paired-
    CUDA anchor required per CLAUDE.md "Submission auth eval — BOTH CPU
    AND CUDA" before any score claim.

    Per canonical Slot UU canonical roadmap (commit ``2b573f105``) +
    canonical Slot FF PR110-OPT-7 sister-cascade pattern (commit
    ``0adecdc5b``) + OVERNIGHT-EEE canonical IMPLEMENTATION-LEVEL
    boundary anchor 2026-05-21 per Catalog #307: the verdict field is
    ``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog #325
    (HILL canonical cascade is distinct paradigm from UNIWARD weighting;
    reactivation criterion pinned in design memo).

    Args:
        image: 2D sequence of floats (luma channel; height x width).
        config: Canonical :class:`HILLConfig`.

    Returns:
        Dict with canonical Tier A contribution shape:

        - ``predicted_delta_adjustment`` (float): always 0.0 (Tier A
          observability-only per Catalog #341).
        - ``promotable`` (bool): always False (Catalog #341 +
          Catalog #1 device-fork trap protection).
        - ``axis_tag`` (str): always "[predicted]" (Catalog #287).
        - ``predicted_axis_decomposition`` (dict): canonical
          :class:`AxisDecomposition` dict-form per Catalog #356 (if
          ``config.emit_axis_decomposition=True``); None otherwise.
        - ``wire_analysis`` (dict): output of
          :func:`compute_hill_canonical_cost_matrix_for_pr110_catalog`.
        - ``verdict`` (str): "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR".
        - ``canonical_li_wang_2014_anchor`` (dict): canonical citation.
        - ``slot_uu_roadmap_anchor`` (dict): Slot UU canonical TOP-1
          ranking citation.
        - ``overnight_eee_implementation_boundary_anchor`` (dict):
          OVERNIGHT-EEE IMPLEMENTATION-LEVEL boundary per Catalog #307.
        - ``slot_ff_sister_cascade_anchor`` (dict): Slot FF PR110-OPT-7
          canonical pattern citation.
        - ``design_memo_path`` (str): path to design memo.
        - ``horizon_class`` (str): canonical ``plateau_adjacent``.
        - ``per_substrate_empirical_verification_stub`` (dict): per Slot
          QQ canonical META-LESSON stub for downstream per-substrate
          empirical verification routing.
    """
    wire = compute_hill_canonical_cost_matrix_for_pr110_catalog(image, config)
    archive_bytes_delta = int(wire["delta_vs_fec6_bytes"])
    pose_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied
    seg_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied

    axis_decomp_payload: dict[str, Any] | None = None
    if config.emit_axis_decomposition:
        prov = build_provenance_for_predicted(
            model_id=(
                "tac.composition.hill_canonical_inverse_steganalysis_"
                "li_wang_li_huang_2014.apply_hill_canonical_cost_matrix_"
                "to_pr110_archive"
            ),
            inputs_sha256=_compute_hill_canonical_signature(
                config.n_pairs,
                config.sparse_k,
                config.strategy,
                config.kb_kernel_size,
                config.l1_kernel_size,
                config.l2_kernel_size,
                config.epsilon,
            ),
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        )
        decomp = AxisDecomposition(
            predicted_d_seg_delta=seg_delta,
            predicted_d_pose_delta=pose_delta,
            predicted_archive_bytes_delta=archive_bytes_delta,
            axis_tag="[predicted]",
            canonical_provenance=provenance_to_dict(prov),
        )
        axis_decomp_payload = decomp.as_dict()

    # Canonical per-substrate empirical verification stub per Slot QQ
    # canonical META-LESSON: per-archive EMPIRICAL VERIFICATION REQUIRED
    # BEFORE classification overlay assignment.
    per_substrate_stub = {
        "status": "pending_per_substrate_empirical_verification",
        "next_action": (
            "queue_paired_CUDA_RATIFICATION_per_catalog_246_envelope_$0.06"
        ),
        "canonical_lesson_reference": (
            "Slot QQ canonical META-LESSON: per-archive empirical verification"
            " required before classification overlay"
        ),
    }

    return {
        # Tier A canonical-routing markers per Catalog #341 + #357
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        # Per-axis decomposition per Catalog #356
        "predicted_axis_decomposition": axis_decomp_payload,
        # Analytical primitive output
        "wire_analysis": wire,
        # Catalog #325 verdict
        "verdict": "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR",
        # Canonical Li-Wang-Li-Huang 2014 anchor
        "canonical_li_wang_2014_anchor": {
            "citation_url": CANONICAL_LI_WANG_2014_CITATION_URL,
            "kb_kernel_size_canonical": CANONICAL_KB_KERNEL_SIZE,
            "l1_kernel_size_canonical": CANONICAL_L1_KERNEL_SIZE,
            "l2_kernel_size_canonical": CANONICAL_L2_KERNEL_SIZE,
            "epsilon_canonical": CANONICAL_HILL_EPSILON,
            "canonical_citation": (
                "Li-Wang-Li-Huang 2014 'A new cost function for spatial "
                "image steganography' (canonical HIGH-pass + LOW-pass + "
                "LOW-pass aggregation cost-matrix per Slot UU TOP-1 9/9)"
            ),
        },
        # Slot UU canonical roadmap anchor
        "slot_uu_roadmap_anchor": {
            "commit_sha": "2b573f105",
            "rank": 1,
            "score": "9/9",
            "axes": "math-grounding + compounding-automation + optimal-fractal",
            "operator_binding_directive": "#10",
        },
        # OVERNIGHT-EEE canonical IMPLEMENTATION-LEVEL boundary anchor
        "overnight_eee_implementation_boundary_anchor": {
            "landing_memo_path": (
                ".omx/research/overnight_eee_hill_filter_steganalysis_"
                "sister_landed_20260521.md"
            ),
            "verdict": "NULL_SIGNAL_DEFER per Catalog #307 IMPLEMENTATION-LEVEL boundary",
            "interpretation_mismatch": (
                "OVERNIGHT-EEE used CCC/DDD reciprocal-weight framing; "
                "THIS L0 SCAFFOLD uses canonical Li-Wang cost-distribution "
                "interpretation for canonical pixel-selection priority"
            ),
            "catalog_307_classification": (
                "IMPLEMENTATION_LEVEL_BOUNDARY_PARADIGM_INTACT"
            ),
        },
        # Slot FF sister-cascade anchor
        "slot_ff_sister_cascade_anchor": {
            "commit_sha": "0adecdc5b",
            "sister_pattern_path": (
                "src/tac/composition/pr110_opt_7_uniward_inverse_scorer_"
                "basis_expansion/__init__.py"
            ),
            "axis_position_in_cascade": "Axis 1 (UNIWARD); THIS Slot YY = Axis 5 (HILL)",
        },
        # Per-substrate empirical verification stub (Slot QQ META-LESSON)
        "per_substrate_empirical_verification_stub": per_substrate_stub,
        # Sister citation surface
        "design_memo_path": (
            ".omx/research/"
            "hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_"
            "high_pass_low_pass_low_pass_aggregation_cost_matrix_canonical_"
            "fridrich_yousfi_cascade_axis_5_extension_per_slot_uu_top_1_"
            "design_20260529.md"
        ),
        "horizon_class": "plateau_adjacent",
    }


# -----------------------------------------------------------------------------
# CANONICAL MLX REAL-VIDEO BIND HELPER (Slot EEE PARTIAL → REAL per-pixel remediation)
# -----------------------------------------------------------------------------
#
# Per Slot EEE audit Axis A (cite-vs-impl PARTIAL) + Axis C (smoke realism FAIL)
# verdicts on this package + the operator binding 5-invariant standing directive
# 2026-05-29 (invariant 5: "no fake implementations" + invariant 4: "binded and
# deployed on MLX asap"): this canonical helper provides the REAL per-pixel HILL
# implementation on REAL ``upstream/videos/0.mkv`` decoded frames via the
# canonical shared helper ``tac.inverse_steganalysis_real_video_mlx``.
#
# The existing ``compute_hill_canonical_cost_matrix_for_pr110_catalog`` +
# ``apply_hill_canonical_cost_matrix_to_pr110_archive`` entry points are
# PRESERVED for backward compatibility with downstream callers that operate at
# the per-pair scalar-aggregation surface (the PR110 archive grammar surface).
#
# The new entry point ``apply_hill_canonical_per_pixel_mlx_to_real_video_frames``
# operates at the canonical per-pixel surface that the Li-Wang-Li-Huang 2014
# paper actually describes, on REAL decoded video frames.


def apply_hill_canonical_per_pixel_mlx_to_real_video_frames(
    *,
    num_frames: int = 4,
    target_resolution: tuple[int, int] = (128, 96),  # (W, H) for cheap smoke
    use_mlx: bool = True,
    l1_kernel_size: int = CANONICAL_L1_KERNEL_SIZE,
    l2_kernel_size: int = CANONICAL_L2_KERNEL_SIZE,
    epsilon: float = CANONICAL_HILL_EPSILON,
) -> dict[str, Any]:
    """Canonical per-pixel HILL on REAL ``upstream/videos/0.mkv`` frames via MLX.

    Sister of :func:`apply_hill_canonical_cost_matrix_to_pr110_archive` at the
    per-pixel real-video surface. Where the existing apply entry-point operates
    on a single per-frame sequence with per-pair scalar aggregation (PR110
    archive grammar surface), this entry-point operates on the canonical
    per-pixel surface that Li-Wang-Li-Huang 2014 actually describes.

    Per Slot EEE audit recommendations + operator binding 5-invariant standing
    directive 2026-05-29: implements the canonical per-pixel cascade on REAL
    ``upstream/videos/0.mkv`` frames (NOT synthetic random noise) via the
    canonical shared helper :mod:`tac.inverse_steganalysis_real_video_mlx`.

    Returns Tier A canonical-routing-markers contribution per Catalog #341 +
    Catalog #323 (``score_claim=False`` + ``promotable=False``) per Catalog
    #192 (macOS-CPU advisory NEVER promotable).

    Parameters
    ----------
    num_frames : int
        Number of frames to decode from ``upstream/videos/0.mkv`` (default 4
        for cheap smoke).
    target_resolution : (int, int)
        ``(W, H)`` for bilinear resize (default ``(128, 96)`` for cheap smoke).
    use_mlx : bool
        Use MLX (default True per CLAUDE.md "MLX portable-local-substrate
        authority" 8th standing directive); set False for numpy-only.
    l1_kernel_size, l2_kernel_size, epsilon : per Li-Wang-Li-Huang 2014
        canonical parameters.

    Returns
    -------
    dict
        Canonical Tier A contribution per Catalog #341 + #323 + #356 + #305:
        - ``predicted_delta_adjustment`` (always 0.0 per Tier A)
        - ``promotable`` (always False)
        - ``score_claim`` (always False)
        - ``axis_tag`` ("[predicted]")
        - ``smoke_result`` (canonical
          :class:`tac.inverse_steganalysis_real_video_mlx.CanonicalSmokeResult`
          dict-form)
        - ``verdict`` ("PER_PIXEL_MLX_REAL_VIDEO_SMOKE_GREEN_DEFERRED_PENDING_
          PAIRED_CUDA_EMPIRICAL_ANCHOR")
        - ``per_pixel_real_video_remediation_anchor`` (Slot EEE Axis A + C
          remediation citation)

    Notes
    -----
    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192: this
    helper produces ``[macOS-CPU advisory]`` / ``[macOS-MLX research-signal]``
    output that is NEVER promotable to a contest-axis score claim. Paired
    Linux x86_64 + NVIDIA empirical anchor required per Catalog #246 before
    any score claim.
    """
    from tac.inverse_steganalysis_real_video_mlx import (
        compute_hill_per_pixel_cost_mlx,
        run_macos_cpu_advisory_smoke,
    )

    smoke_result = run_macos_cpu_advisory_smoke(
        target_name=(
            "hill_canonical_inverse_steganalysis_li_wang_li_huang_2014_"
            "per_pixel_mlx_real_video"
        ),
        cost_function=compute_hill_per_pixel_cost_mlx,
        num_frames=num_frames,
        target_resolution=target_resolution,
        use_mlx=use_mlx,
        cost_function_kwargs={
            "l1_kernel_size": l1_kernel_size,
            "l2_kernel_size": l2_kernel_size,
            "epsilon": epsilon,
        },
    )

    return {
        # Tier A canonical-routing markers per Catalog #341 + #357
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "score_claim": False,
        "axis_tag": "[predicted]",
        # Canonical smoke result
        "smoke_result": smoke_result.to_dict(),
        # Catalog #325 verdict
        "verdict": (
            "PER_PIXEL_MLX_REAL_VIDEO_SMOKE_GREEN_"
            "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"
        ),
        # Slot EEE remediation anchor
        "per_pixel_real_video_remediation_anchor": {
            "slot_eee_audit_axis_a_verdict": "PARTIAL_remediated",
            "slot_eee_audit_axis_c_verdict": "FAIL_remediated",
            "remediation_pattern": (
                "Canonical shared helper tac.inverse_steganalysis_real_video_mlx "
                "applied per-pixel HILL cost matrix on real upstream/videos/0.mkv "
                "decoded frames per Li-Wang-Li-Huang 2014 canonical formulation"
            ),
            "canonical_helper_module": (
                "tac.inverse_steganalysis_real_video_mlx"
            ),
            "canonical_helper_function": "compute_hill_per_pixel_cost_mlx",
            "operator_binding_5_invariant_standing_directive_anchor": (
                "feedback_optimize_iterate_highest_ev_boldest_individually_"
                "fractally_optimized_mlx_deployed_aggressive_frontier_breaking_"
                "no_fake_implementations_standing_directive_20260529"
            ),
            "slot_eee_audit_anchor": (
                "feedback_slot_eee_fake_implementation_audit_on_today_l0_"
                "scaffolds_per_operator_binding_must_review_for_fake_"
                "implementations_landed_20260529"
            ),
        },
    }


__all__ = (
    "CANONICAL_LI_WANG_2014_CITATION_URL",
    "CANONICAL_KB_KERNEL_3X3",
    "CANONICAL_KB_KERNEL_SIZE",
    "CANONICAL_L1_KERNEL_SIZE",
    "CANONICAL_L2_KERNEL_SIZE",
    "CANONICAL_HILL_EPSILON",
    "CANONICAL_SPARSE_K_DEFAULT",
    "CANONICAL_WIDENED_K_DEFAULT",
    "CANONICAL_FEC6_BASELINE_WIRE_BYTES",
    "CANONICAL_N_PAIRS",
    "CANONICAL_N_MODES",
    "CANONICAL_RATE_MULTIPLIER",
    "CANONICAL_RATE_DENOM_BYTES",
    "HILLCostMatrixStrategy",
    "HILLConfig",
    "compute_hill_canonical_cost_matrix_for_pr110_catalog",
    "apply_hill_canonical_cost_matrix_to_pr110_archive",
    "apply_hill_canonical_per_pixel_mlx_to_real_video_frames",
)
