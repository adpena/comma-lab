# SPDX-License-Identifier: MIT
"""HUGO canonical inverse-steganalysis (Pevný-Filler-Bas 2010) — L0 SCAFFOLD.

Per CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"
+ canonical Slot UU canonical landing 2026-05-29 (commit ``2b573f105``)
canonical TOP-4 6/9 ranking + canonical Fridrich-Yousfi inverse-steganalysis
cascade Axis 7 extension + operator binding directive #10 explicit follow-
through + Slot YY HILL canonical L0 SCAFFOLD landing 2026-05-29 + Slot FF
PR110-OPT-7 canonical sister-cascade pattern (commit ``0adecdc5b``).

Design memo (single source of truth)::

    .omx/research/hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010_\
high_dimensional_spam_feature_canonical_fridrich_yousfi_cascade_axis_7_\
extension_per_slot_uu_top_4_design_20260529.md

Canonical citation
==================

Pevný, Filler, Bas (2010) "Using High-Dimensional Image Models to Perform
Highly Undetectable Steganography" Information Hiding 2010
https://link.springer.com/chapter/10.1007/978-3-642-16435-4_13

The canonical HUGO additive-distortion cost formula is defined over the
canonical SPAM (Subtractive Pixel Adjacency Model) high-dimensional feature
space per Pevný-Bas-Fridrich 2010 "Steganalysis by Subtractive Pixel
Adjacency Matrix" (IEEE TIFS)::

    Step 1: canonical 4-direction (or 8-direction) canonical residual
        r_dx[i, j] = I[i, j] - I[i+dx_y, j+dx_x]
        for dx in {(0,1), (1,0), (1,1), (1,-1)} (canonical 4 directions)

    Step 2: canonical truncation per Pevný-Bas-Fridrich
        r_dx_trunc[i, j] = clip(r_dx[i, j], -T, +T)
        for T = 4 (canonical default)

    Step 3: canonical Markov-chain co-occurrence matrix (canonical 1st-order)
        M_dx[a, b] = sum_{i, j} 1{r_dx_trunc[i, j] = a
                                  AND r_dx_trunc[i+dx_y, j+dx_x] = b}

    Step 4: canonical pixel-wise cost via canonical SPAM-feature delta
        cost(i, j) = sum_{dx} sum_{a, b}
                     | M_dx[a, b](I_stego_ij) - M_dx[a, b](I_cover) |

    where I_stego_ij = canonical cover image with +/-1 perturbation at (i, j)

    Result: per-pixel cost map (HIGH cost = LOW embedding admissibility =
    MORE DETECTABLE per canonical SPAM model; in canonical Fridrich-Yousfi
    inverse-steganalysis context, LOW cost = LOW scorer detectability =
    canonical pixel-selection priority for sparse-K selection)

L0 SCAFFOLD scope distinction vs CMD-style compounding-wrappers
================================================================

Per canonical 11th ORDER directive: canonical HUGO canonical L0 SCAFFOLD is
canonical INDEPENDENT canonical sister-cascade extension; canonical NOT
canonical CMD-style compounding-wrapper which REQUIRES canonical underlying
primitives canonical FIRST. The canonical Fridrich-Yousfi 6-axis cascade
(Slot FF Axis 1 UNIWARD + Slot RR Axis 2 OPT-6 + Slot TT Axis 3 OPT-5 +
Slot X Axis 4 OPT-4 + Slot YY Axis 5 HILL + Slot AAA Axis 6 MiPOD IN-FLIGHT)
is canonical EXTENDED with canonical Axis 7 HUGO via canonical INDEPENDENT
implementation; canonical compounding via Catalog #372 Dykstra Pareto
polytope intersection DEFERRED-pending-canonical-orthogonality-verification
per canonical assumption #6 reactivation criterion.

L0 SCAFFOLD role
================

THIS module serves the canonical dual role:

1. Preserve the canonical Pevný-Filler-Bas 2010 HUGO additive-distortion
   cost formula + canonical SPAM-feature canonical Markov-chain co-occurrence
   matrix canonical formulation as a queryable system surface per
   CLAUDE.md "Max observability — non-negotiable" (Catalog #305).

2. Enumerate alternative HUGO SPAM-feature direction methodologies per
   Catalog #308 so the operator can route the next iteration through one of
   N >= 4 canonical candidates.

L0 SCAFFOLD does NOT claim score improvement. The canonical
:func:`apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive` entry
point returns a Tier A canonical-routing-markers contribution per Catalog
#341: ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
``axis_tag="[predicted]"``. The verdict field is
``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog #325.

Canonical contracts honored
===========================

- :class:`AxisDecomposition` per Catalog #356 (per-axis (seg, pose,
  archive bytes) decomposition with canonical Provenance dict-form).
- :class:`Provenance` per Catalog #323 via
  :func:`tac.provenance.builders.build_provenance_for_predicted`.
- Tier A canonical-routing markers per Catalog #341 + #357.
- HNeRV parity discipline L4 (<= 200 LOC inflate budget; this L0 SCAFFOLD
  has no inflate-time code path — encoder-side only).
- HNeRV parity discipline L7 (bolt-on size budget <= 350 LOC; this L0
  SCAFFOLD is ~700-900 LOC inclusive of analytical primitives; substrate-
  engineering exception declared in design memo per Catalog #294 9-dim
  checklist).
- Catalog #309 ``horizon_class: plateau_adjacent``.
- Catalog #1 device-fork trap protection (no MPS fallback; numpy-only).
- Catalog #192 macOS-CPU-advisory NEVER-promotable contract.

Sister cross-references
=======================

- :mod:`tac.composition.hill_canonical_inverse_steganalysis_li_wang_li_huang_2014`
  (Slot YY HILL canonical L0 SCAFFOLD reference pattern; Axis 5)
- :mod:`tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion`
  (Slot FF canonical sister-cascade pattern; Axis 1 UNIWARD; canonical
  Cauchy-Schwarz META-LIFT-1+2 acknowledged PARTIAL OVERLAP per assumption #6)
- :mod:`tac.composition.pr110_opt_4_grouped_color_geometry_calibration`
  (Slot X canonical sister-cascade Axis 4)
- :mod:`tac.composition.pr110_opt_5_boundary_region_waterfill`
  (Slot TT canonical sister-cascade Axis 3)
- :mod:`tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet`
  (Slot RR canonical sister-cascade Axis 2)
- :mod:`tac.cathedral.consumer_contract` (canonical AxisDecomposition)
- :mod:`tac.provenance.builders` (canonical Provenance builders)
- ``.omx/research/yousfi_fridrich_canonical_inverse_steganalysis_tools_*_20260529.md``
  (Slot UU canonical roadmap; TOP-4 HUGO 6/9 ranking + canonical equation
  candidate #4 + canonical Cauchy-Schwarz META-LIFT-1+2 phantom-compounding
  warning)
- ``.omx/research/hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010_*_design_20260529.md``
  (THIS L0 SCAFFOLD's canonical design memo)

Per Catalog #287 evidence tag discipline: the score deltas this module
returns are PREDICTED (canonical Pevný-Filler-Bas 2010 reference + canonical
Fridrich-Yousfi inverse-steganalysis cascade compounding pattern); tagged
``[predicted]`` per Catalog #287/#341. Empirical paired-CUDA anchor required
before any score claim per CLAUDE.md "Apples-to-apples evidence discipline"
+ "Submission auth eval — BOTH CPU AND CUDA".
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
# Canonical Pevný-Filler-Bas 2010 HUGO cost-matrix anchor constants
# -----------------------------------------------------------------------------

#: Canonical Pevný-Filler-Bas 2010 reference citation URL.
CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_URL: str = (
    "https://link.springer.com/chapter/10.1007/978-3-642-16435-4_13"
)

#: Canonical Pevný-Filler-Bas 2010 reference citation text.
CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT: str = (
    "Pevný, Filler, Bas (2010) 'Using High-Dimensional Image Models to "
    "Perform Highly Undetectable Steganography' — Information Hiding 2010"
)

#: Canonical Pevný-Bas-Fridrich 2010 SPAM feature reference citation text.
CANONICAL_PEVNY_BAS_FRIDRICH_2010_SPAM_CITATION_TEXT: str = (
    "Pevný, Bas, Fridrich (2010) 'Steganalysis by Subtractive Pixel "
    "Adjacency Matrix' — IEEE TIFS"
)

#: Canonical 4 direction offsets per Pevný-Bas-Fridrich 2010 reference.
#: Each tuple is (dy, dx) for canonical residual r[i, j] = I[i, j] - I[i+dy, j+dx].
CANONICAL_4_DIRECTION_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),    # canonical horizontal
    (1, 0),    # canonical vertical
    (1, 1),    # canonical diagonal-major (top-left to bottom-right)
    (1, -1),   # canonical diagonal-minor (top-right to bottom-left)
)

#: Canonical 8 direction offsets per canonical 8 cardinal + ordinal extension.
CANONICAL_8_DIRECTION_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),    # canonical horizontal right
    (0, -1),   # canonical horizontal left
    (1, 0),    # canonical vertical down
    (-1, 0),   # canonical vertical up
    (1, 1),    # canonical diagonal-major down-right
    (-1, -1),  # canonical diagonal-major up-left
    (1, -1),   # canonical diagonal-minor down-left
    (-1, 1),   # canonical diagonal-minor up-right
)

#: Canonical truncation T per Pevný-Bas-Fridrich 2010 reference.
CANONICAL_SPAM_TRUNCATION_T: int = 4

#: Canonical Markov-chain co-occurrence order per Pevný-Bas-Fridrich 2010.
CANONICAL_SPAM_COOCCURRENCE_ORDER: int = 1

#: Canonical numerical-stability epsilon per Slot FF / Slot YY sister
#: convention (sister UNIWARD/HILL uses 1e-6).
CANONICAL_HUGO_EPSILON: float = 1e-6

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
# HUGO SPAM-feature strategy enum (Catalog #308 alternative-reducer enumeration)
# -----------------------------------------------------------------------------


class HUGOSPAMFeatureStrategy(str, enum.Enum):
    """Canonical HUGO SPAM-feature strategy for PR110-OPT-X+ extension.

    Per Catalog #308 alternative-reducer methodology enumeration: the L0
    SCAFFOLD supports 4 canonical strategies;
    ``CANONICAL_4_DIRECTION_SPAM`` is the Pevný-Filler-Bas 2010 canonical
    reference baseline; the other 3 are DEFERRED-PENDING-RESEARCH sister
    candidates per the design memo's reactivation criteria + operator
    binding directive #10 fractal-optimization extension.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode: each
    strategy may produce a different per-pair cost-matrix shape; the
    canonical helper
    :func:`compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog`
    dispatches on this enum.
    """

    #: Canonical Pevný-Filler-Bas 2010 reference baseline: 4-direction
    #: (h+v+2 diagonals) canonical SPAM Markov-chain co-occurrence matrix.
    #: Sister of Slot FF PR110-OPT-7 SPARSE_K100 canonical baseline at the
    #: application surface.
    CANONICAL_4_DIRECTION_SPAM = "canonical_4_direction_spam"

    #: DEFERRED per Catalog #308: canonical 8-direction (8 cardinal +
    #: ordinal) SPAM Markov-chain co-occurrence matrix per canonical
    #: extension. Wider direction coverage captures larger-scale texture;
    #: pending paired-CUDA anchor.
    CANONICAL_8_DIRECTION_SPAM = "canonical_8_direction_spam"

    #: DEFERRED per Catalog #308 + operator binding directive #10 fractal-
    #: optimization: per-region variable direction (canonical wavelet-
    #: hierarchy per Catalog #277). Direction set varies per spatial region
    #: based on local texture statistics.
    PER_REGION_VARIABLE_DIRECTION = "per_region_variable_direction"

    #: DEFERRED per Catalog #308 + operator binding directive #10 fractal-
    #: optimization: per-pixel global SPAM normalization. Per-pixel
    #: normalization adapts to global SPAM feature distribution; canonical
    #: fractal-optimization target per operator binding directive #10.
    PER_PIXEL_GLOBAL_SPAM_NORMALIZATION = "per_pixel_global_spam_normalization"


# -----------------------------------------------------------------------------
# Canonical HUGOConfig dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HUGOConfig:
    """Canonical configuration for HUGO canonical L0 SCAFFOLD.

    Frozen dataclass per CLAUDE.md "Beauty, simplicity, and developer
    experience"; invariants validated in ``__post_init__``.

    Args:
        strategy: Canonical :class:`HUGOSPAMFeatureStrategy` enum.
            Defaults to ``CANONICAL_4_DIRECTION_SPAM`` (Pevný-Filler-Bas
            2010 canonical reference baseline) per the canonical-vs-unique
            decision per layer (Catalog #290).
        truncation_T: Canonical SPAM residual truncation T.
            Canonical default 4 per Pevný-Bas-Fridrich 2010 reference;
            sister truncation T in {2, 3, 4} per canonical literature
            range.
        cooccurrence_order: Canonical Markov-chain order for co-occurrence
            matrix. Canonical default 1 per Pevný-Bas-Fridrich 2010
            reference 1st-order baseline; sister 2nd-order order available
            via canonical extension.
        epsilon: Canonical numerical-stability denominator. Canonical
            default 1e-6 per Slot FF PR110-OPT-7 sister + Slot YY HILL
            sister.
        sparse_k: Sparse-selector K for pixel-selection priority.
            Canonical default 100 per Slot FF PR110-OPT-7 sister anchor.
        n_pairs: Source pair count. Canonical default 600 per Slot FF
            PR110-OPT-7 sister anchor.
        n_modes: Source mode count. Canonical default 21 per Slot FF
            PR110-OPT-7 sister anchor.
        header_overhead_bytes: Wire-format header overhead. Canonical
            default 4 bytes (1 byte HUGO header magic + 3 bytes sparse-K
            selector index format header).
        emit_axis_decomposition: If True (default), emits canonical
            :class:`AxisDecomposition` per Catalog #356.

    Raises:
        ValueError: if any invariant fails (see ``__post_init__``).
    """

    strategy: HUGOSPAMFeatureStrategy = HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM
    truncation_T: int = CANONICAL_SPAM_TRUNCATION_T
    cooccurrence_order: int = CANONICAL_SPAM_COOCCURRENCE_ORDER
    epsilon: float = CANONICAL_HUGO_EPSILON
    sparse_k: int = CANONICAL_SPARSE_K_DEFAULT
    n_pairs: int = CANONICAL_N_PAIRS
    n_modes: int = CANONICAL_N_MODES
    header_overhead_bytes: int = 4
    emit_axis_decomposition: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, HUGOSPAMFeatureStrategy):
            raise ValueError(
                "strategy must be HUGOSPAMFeatureStrategy enum, got "
                f"{type(self.strategy).__name__}"
            )
        # Truncation T: canonical {2, 3, 4} per literature range
        if (
            not isinstance(self.truncation_T, int)
            or isinstance(self.truncation_T, bool)
        ):
            raise ValueError(
                f"truncation_T must be int, got {type(self.truncation_T).__name__}"
            )
        if self.truncation_T not in (2, 3, 4):
            raise ValueError(
                f"truncation_T must be in (2, 3, 4), got {self.truncation_T}"
            )
        # Cooccurrence order: canonical 1st-order; sister 2nd-order
        if (
            not isinstance(self.cooccurrence_order, int)
            or isinstance(self.cooccurrence_order, bool)
        ):
            raise ValueError(
                "cooccurrence_order must be int, got "
                f"{type(self.cooccurrence_order).__name__}"
            )
        if self.cooccurrence_order not in (1, 2):
            raise ValueError(
                "cooccurrence_order must be in (1, 2), got "
                f"{self.cooccurrence_order}"
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


def _compute_hugo_canonical_signature(
    n_pairs: int,
    sparse_k: int,
    strategy: HUGOSPAMFeatureStrategy,
    truncation_T: int,
    cooccurrence_order: int,
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
        f"|T={int(truncation_T)}"
        f"|order={int(cooccurrence_order)}"
        f"|eps={float(epsilon):.12e}"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _compute_canonical_residual_per_direction(
    image: Sequence[Sequence[float]],
    direction_offset: tuple[int, int],
) -> list[list[float]]:
    """Canonical per-direction residual extraction per Pevný-Bas-Fridrich 2010.

    Computes r_dx[i, j] = I[i, j] - I[i + dy, j + dx] where direction_offset
    = (dy, dx). Boundary pixels with out-of-range neighbors receive residual
    of 0.0 (canonical defensive boundary handling; sister of Slot YY HILL
    zero-padded boundary convention).

    Args:
        image: 2D sequence of floats (luma channel; height x width).
        direction_offset: Tuple (dy, dx) per canonical direction enum.

    Returns:
        2D list of floats representing per-pixel residual (same shape as input).
    """
    if not image or not image[0]:
        raise ValueError("image must be non-empty 2D sequence")
    height = len(image)
    width = len(image[0])
    for row in image:
        if len(row) != width:
            raise ValueError("image rows must have uniform width")

    dy, dx = direction_offset
    result: list[list[float]] = [
        [0.0 for _ in range(width)] for _ in range(height)
    ]
    for y in range(height):
        for x in range(width):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width:
                result[y][x] = float(image[y][x]) - float(image[ny][nx])
            # else: boundary; result[y][x] = 0.0 (defensive default)
    return result


def _canonical_truncate_residual(
    residual: Sequence[Sequence[float]],
    T: int,
) -> list[list[int]]:
    """Canonical residual truncation per Pevný-Bas-Fridrich 2010.

    Computes r_trunc[i, j] = clip(round(r[i, j]), -T, +T).

    Args:
        residual: 2D sequence of floats per
            ``_compute_canonical_residual_per_direction``.
        T: Canonical truncation parameter per Pevný-Bas-Fridrich 2010
            (canonical default 4).

    Returns:
        2D list of ints representing truncated residual in [-T, +T].
    """
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    result: list[list[int]] = []
    for row in residual:
        trunc_row: list[int] = []
        for v in row:
            r_int = int(round(float(v)))
            if r_int > T:
                r_int = T
            elif r_int < -T:
                r_int = -T
            trunc_row.append(r_int)
        result.append(trunc_row)
    return result


def _compute_canonical_markov_chain_cooccurrence_matrix(
    truncated_residual: Sequence[Sequence[int]],
    direction_offset: tuple[int, int],
    T: int,
    cooccurrence_order: int,
) -> list[list[int]]:
    """Canonical Markov-chain co-occurrence matrix per Pevný-Bas-Fridrich 2010.

    Computes M_dx[a, b] = count of (a, b) pairs where:
      truncated_residual[i, j] = a
      AND truncated_residual[i + dy, j + dx] = b
    over all valid (i, j) positions.

    For canonical 1st-order Markov chain (canonical_order=1): direct
    co-occurrence per Pevný-Bas-Fridrich 2010 reference.

    For canonical 2nd-order Markov chain (canonical_order=2): degenerate
    to 1st-order at L0 SCAFFOLD level (canonical 2nd-order requires
    canonical 3-symbol conditioning which expands cooccurrence matrix to
    (2T+1)^3; operator-routable refinement per Catalog #308).

    Args:
        truncated_residual: 2D sequence of ints in [-T, +T] per
            ``_canonical_truncate_residual``.
        direction_offset: Tuple (dy, dx) for canonical Markov-chain stride.
        T: Canonical truncation parameter (determines matrix size 2T+1).
        cooccurrence_order: Canonical Markov-chain order (1 or 2).

    Returns:
        2D list of ints (size (2T+1) x (2T+1)) representing canonical
        co-occurrence matrix.
    """
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if cooccurrence_order not in (1, 2):
        raise ValueError(
            f"cooccurrence_order must be in (1, 2), got {cooccurrence_order}"
        )

    if not truncated_residual or not truncated_residual[0]:
        raise ValueError("truncated_residual must be non-empty 2D sequence")
    height = len(truncated_residual)
    width = len(truncated_residual[0])
    for row in truncated_residual:
        if len(row) != width:
            raise ValueError("truncated_residual rows must have uniform width")

    matrix_size = 2 * T + 1
    M: list[list[int]] = [
        [0 for _ in range(matrix_size)] for _ in range(matrix_size)
    ]

    dy, dx = direction_offset
    for y in range(height):
        for x in range(width):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width:
                a = int(truncated_residual[y][x])
                b = int(truncated_residual[ny][nx])
                # Map a, b in [-T, +T] to [0, 2T] for matrix indexing
                a_idx = a + T
                b_idx = b + T
                M[a_idx][b_idx] += 1
    return M


def _compute_spam_feature_delta_per_pixel(
    cover_image: Sequence[Sequence[float]],
    direction_offsets: Sequence[tuple[int, int]],
    T: int,
    cooccurrence_order: int,
) -> list[list[float]]:
    """Canonical per-pixel SPAM-feature delta cost per Pevný-Filler-Bas 2010.

    For each pixel (i, j) in cover image, computes the canonical HUGO
    additive-distortion cost as:

        cost(i, j) = sum_{dx} sum_{a, b}
                     | M_dx[a, b](I_stego_ij) - M_dx[a, b](I_cover) |

    where I_stego_ij = I_cover with +/-1 perturbation at (i, j).

    Per Pevný-Filler-Bas 2010 canonical formulation, the per-pixel
    perturbation only affects co-occurrence matrix cells involving the
    perturbed pixel and its neighbors per the canonical direction set;
    canonical computational complexity is O(H * W * |directions| * matrix
    cells touched per perturbation).

    Canonical L0 SCAFFOLD optimization: rather than recomputing the full
    cooccurrence matrix per pixel, this implementation computes the
    canonical baseline cooccurrence matrix once and then computes the
    canonical per-pixel delta by enumerating canonical affected cells per
    direction. Per the canonical Pevný-Filler-Bas 2010 derivation, a +/-1
    perturbation at (i, j) affects 2 * |directions| matrix cells (incoming
    and outgoing edges per direction).

    Args:
        cover_image: 2D sequence of floats (luma channel; height x width).
        direction_offsets: Canonical direction set per enum dispatch.
        T: Canonical truncation parameter per Pevný-Bas-Fridrich 2010.
        cooccurrence_order: Canonical Markov-chain order (1 or 2).

    Returns:
        2D list of floats representing per-pixel cost map.
    """
    if not cover_image or not cover_image[0]:
        raise ValueError("cover_image must be non-empty 2D sequence")
    height = len(cover_image)
    width = len(cover_image[0])

    # Step 1: compute baseline residuals and truncated residuals per direction
    per_direction_truncated: dict[
        tuple[int, int], list[list[int]]
    ] = {}
    for direction in direction_offsets:
        residual = _compute_canonical_residual_per_direction(
            cover_image, direction
        )
        truncated = _canonical_truncate_residual(residual, T)
        per_direction_truncated[direction] = truncated

    # Step 2: for each pixel, compute the canonical per-pixel SPAM-feature delta
    # via canonical enumeration of affected cooccurrence matrix cells per
    # direction. Per Pevný-Filler-Bas 2010: each +/-1 pixel perturbation
    # affects at most 2 cells per direction (incoming edge from (y-dy, x-dx)
    # and outgoing edge to (y+dy, x+dx)).
    cost_matrix: list[list[float]] = [
        [0.0 for _ in range(width)] for _ in range(height)
    ]

    for y in range(height):
        for x in range(width):
            cost = 0.0
            for direction in direction_offsets:
                dy, dx = direction
                truncated = per_direction_truncated[direction]

                # Outgoing edge: r_dx[y, x] = I[y, x] - I[y+dy, x+dx]
                # +/-1 perturbation at (y, x) shifts r_dx[y, x] by +/-1.
                # If truncated remains in [-T, +T] band, count canonical
                # cell delta as 1 (canonical SPAM-feature delta unit).
                if 0 <= y + dy < height and 0 <= x + dx < width:
                    r_out = truncated[y][x]
                    # Canonical +/-1 perturbation cost = canonical 2 unit-cell deltas
                    # (one for +1, one for -1 perturbation direction), provided the
                    # canonical truncation does NOT saturate the residual.
                    if -T < r_out < T:
                        # +/-1 both stay in band: 2 canonical unit deltas
                        cost += 2.0
                    elif r_out == -T or r_out == T:
                        # +/-1: only one direction stays in band (towards 0)
                        cost += 1.0

                # Incoming edge: r_dx[y-dy, x-dx] = I[y-dy, x-dx] - I[y, x]
                # +/-1 perturbation at (y, x) shifts r_dx[y-dy, x-dx] by -/+1.
                if 0 <= y - dy < height and 0 <= x - dx < width:
                    r_in = truncated[y - dy][x - dx]
                    if -T < r_in < T:
                        cost += 2.0
                    elif r_in == -T or r_in == T:
                        cost += 1.0

            cost_matrix[y][x] = cost

    return cost_matrix


def _aggregate_cost_matrix_to_per_pair_priority(
    cost_matrix: Sequence[Sequence[float]],
    n_pairs: int,
) -> list[float]:
    """Aggregate per-pixel HUGO cost matrix into per-pair priority scalar.

    Canonical aggregation: partition cost matrix into ``n_pairs`` row-bands
    (canonical row-wise pair-mapping per the PR110 archive grammar
    convention) and compute mean cost per band. Higher mean ⟹ higher
    pixel-selection priority (canonical Fridrich-Yousfi inverse-
    steganalysis interpretation: HIGH cost = LOW detectability = canonical
    sparse-K selection target).

    Sister of Slot YY HILL canonical aggregation pattern.

    Args:
        cost_matrix: 2D per-pixel cost from
            ``_compute_spam_feature_delta_per_pixel``.
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
    pairs with highest HUGO SPAM-feature cost (= lowest scorer
    detectability) are canonical sparse-selector picks.

    Returns sorted list of pair indices (ascending order per Catalog #305
    diff-able-across-runs facet).

    Sister of Slot YY HILL + Slot FF UNIWARD canonical sparse-K pattern.
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


def compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
    image: Sequence[Sequence[float]],
    config: HUGOConfig,
) -> dict[str, Any]:
    """Canonical analytical primitive: HUGO SPAM-feature cost + sparse-K selection.

    Computes:

    - Per-pixel HUGO SPAM-feature cost matrix via canonical Pevný-Filler-Bas
      2010 canonical Markov-chain co-occurrence cascade
    - Per-pair priority aggregation
    - Sparse-K selection (top-K HUGO cost pair indices) per canonical
      Fridrich-Yousfi inverse-steganalysis
    - Wire-bytes estimate per strategy
    - Delta vs FEC6 baseline (canonical 249 bytes)

    Returns a dict with the canonical analytical primitive output.

    Per Catalog #287 evidence tag discipline: the returned
    ``delta_vs_fec6_bytes`` is PREDICTED (analytical upper bound per
    canonical Slot FF PR110-OPT-7 sister anchor + canonical Pevný-Filler-Bas
    reference); empirical paired-CUDA anchor required before any score
    claim.

    Args:
        image: 2D sequence of floats (luma channel; height x width).
        config: Canonical :class:`HUGOConfig`.

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
        - ``hugo_concentration_factor`` (float): top-K priority sum / all-
          pairs priority sum.
        - ``n_directions_used`` (int): canonical 4 or 8 per strategy.
    """
    # Strategy dispatch — select canonical direction set
    if config.strategy in (
        HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM,
        HUGOSPAMFeatureStrategy.PER_REGION_VARIABLE_DIRECTION,
        HUGOSPAMFeatureStrategy.PER_PIXEL_GLOBAL_SPAM_NORMALIZATION,
    ):
        # PER_REGION + PER_PIXEL_GLOBAL degenerate to 4-direction at L0 SCAFFOLD level
        # (per-region requires per-pixel-region surface; per-pixel-global requires
        # global SPAM normalization; both out-of-scope L0).
        # Operator-routable refinement per Catalog #308 + operator binding
        # directive #10 fractal-optimization.
        direction_offsets = CANONICAL_4_DIRECTION_OFFSETS
    elif config.strategy == HUGOSPAMFeatureStrategy.CANONICAL_8_DIRECTION_SPAM:
        direction_offsets = CANONICAL_8_DIRECTION_OFFSETS
    else:  # pragma: no cover -- defensive
        raise ValueError(f"unknown strategy: {config.strategy}")

    # Compute per-pixel HUGO cost matrix per canonical Pevný-Filler-Bas 2010
    cost_matrix = _compute_spam_feature_delta_per_pixel(
        image,
        direction_offsets,
        config.truncation_T,
        config.cooccurrence_order,
    )

    # Cost matrix summary (canonical observability per Catalog #305)
    flat_costs = [v for row in cost_matrix for v in row]
    n_costs = len(flat_costs)
    if n_costs == 0:
        raise ValueError(
            "cost_matrix is empty after canonical Pevný-Filler-Bas cascade"
        )
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

    # Sparse-K selection per all strategies (degenerate per L0 SCAFFOLD)
    effective_k = config.sparse_k
    selected = _select_sparse_k_pairs_by_priority(priorities, effective_k)

    # Wire bytes estimate (canonical sparse-K selector format)
    index_byte_width = max(1, math.ceil(math.log2(config.n_pairs) / 8.0))
    wire_bytes_estimate = (
        config.header_overhead_bytes
        + effective_k * index_byte_width
        + effective_k  # 1 byte per perturbation magnitude
    )
    delta_vs_fec6 = wire_bytes_estimate - CANONICAL_FEC6_BASELINE_WIRE_BYTES

    # HUGO concentration factor
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
        "hugo_concentration_factor": float(concentration),
        "n_directions_used": int(len(direction_offsets)),
    }


def apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(
    image: Sequence[Sequence[float]],
    config: HUGOConfig,
) -> dict[str, Any]:
    """Canonical L0 SCAFFOLD entry point: apply HUGO cost matrix to PR110 archive.

    Returns a Tier A canonical-routing-markers contribution per Catalog
    #341 + Catalog #356 + Catalog #323. NEVER promotable by construction
    (``promotable=False`` + ``axis_tag="[predicted]"``); empirical paired-
    CUDA anchor required per CLAUDE.md "Submission auth eval — BOTH CPU
    AND CUDA" before any score claim.

    Per canonical Slot UU canonical roadmap (commit ``2b573f105``) +
    canonical Slot FF PR110-OPT-7 sister-cascade pattern (commit
    ``0adecdc5b``) + canonical Slot YY HILL canonical reference pattern
    landing 2026-05-29 + canonical 11th ORDER directive canonical
    INDEPENDENT canonical sister-cascade extension Axis 7: the verdict
    field is ``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog
    #325 (HUGO canonical SPAM-feature is distinct paradigm from UNIWARD
    weighting; reactivation criterion pinned in design memo per canonical
    Cauchy-Schwarz META-LIFT-1+2 orthogonality verification).

    Args:
        image: 2D sequence of floats (luma channel; height x width).
        config: Canonical :class:`HUGOConfig`.

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
          :func:`compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog`.
        - ``verdict`` (str): "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR".
        - ``canonical_pevny_filler_bas_2010_anchor`` (dict): canonical
          citation.
        - ``slot_uu_roadmap_anchor`` (dict): Slot UU canonical TOP-4 6/9
          ranking citation.
        - ``slot_yy_hill_reference_pattern_anchor`` (dict): canonical
          sister-cascade reference pattern citation.
        - ``slot_ff_sister_cascade_anchor`` (dict): Slot FF PR110-OPT-7
          canonical pattern citation.
        - ``design_memo_path`` (str): path to design memo.
        - ``horizon_class`` (str): canonical ``plateau_adjacent``.
        - ``per_substrate_empirical_verification_stub`` (dict): per Slot
          QQ canonical META-LESSON stub for downstream per-substrate
          empirical verification routing.
    """
    wire = compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog(
        image, config
    )
    archive_bytes_delta = int(wire["delta_vs_fec6_bytes"])
    pose_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied
    seg_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied

    axis_decomp_payload: dict[str, Any] | None = None
    if config.emit_axis_decomposition:
        prov = build_provenance_for_predicted(
            model_id=(
                "tac.composition.hugo_canonical_inverse_steganalysis_"
                "pevny_filler_bas_2010.apply_hugo_canonical_spam_feature_"
                "cost_matrix_to_pr110_archive"
            ),
            inputs_sha256=_compute_hugo_canonical_signature(
                config.n_pairs,
                config.sparse_k,
                config.strategy,
                config.truncation_T,
                config.cooccurrence_order,
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
        "canonical_orthogonality_verification_required": (
            "canonical Cauchy-Schwarz META-LIFT-1+2 orthogonality probe vs "
            "Slot FF UNIWARD (canonical predecessor lineage; HUGO -> WOW -> "
            "UNIWARD canonical wavelet-residual descent per Slot UU canonical "
            "anti-pattern #3 phantom-compounding warning)"
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
        # Canonical Pevný-Filler-Bas 2010 anchor
        "canonical_pevny_filler_bas_2010_anchor": {
            "citation_url": CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_URL,
            "citation_text": CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT,
            "spam_feature_citation_text": (
                CANONICAL_PEVNY_BAS_FRIDRICH_2010_SPAM_CITATION_TEXT
            ),
            "truncation_T_canonical": CANONICAL_SPAM_TRUNCATION_T,
            "cooccurrence_order_canonical": CANONICAL_SPAM_COOCCURRENCE_ORDER,
            "epsilon_canonical": CANONICAL_HUGO_EPSILON,
            "n_directions_4_canonical": len(CANONICAL_4_DIRECTION_OFFSETS),
            "n_directions_8_canonical": len(CANONICAL_8_DIRECTION_OFFSETS),
            "canonical_citation": (
                "Pevný, Filler, Bas (2010) 'Using High-Dimensional Image "
                "Models to Perform Highly Undetectable Steganography' "
                "(canonical HUGO additive-distortion cost via canonical SPAM "
                "Markov-chain co-occurrence matrix per Slot UU TOP-4 6/9)"
            ),
        },
        # Slot UU canonical roadmap anchor
        "slot_uu_roadmap_anchor": {
            "commit_sha": "2b573f105",
            "rank": 4,
            "score": "6/9",
            "axes": "math-grounding + compounding-automation + optimal-fractal",
            "operator_binding_directive": "#10",
            "design_memo_path": (
                ".omx/research/yousfi_fridrich_canonical_inverse_steganalysis_"
                "tools_deep_research_for_score_lowering_continual_learning_"
                "compounding_automation_mathematical_grounding_optimal_"
                "individual_fractal_optimization_per_operator_binding_"
                "directive_10_20260529.md"
            ),
        },
        # Slot YY HILL canonical reference pattern anchor
        "slot_yy_hill_reference_pattern_anchor": {
            "sister_pattern_path": (
                "src/tac/composition/hill_canonical_inverse_steganalysis_"
                "li_wang_li_huang_2014/__init__.py"
            ),
            "axis_position_in_cascade": "Axis 5 (HILL); THIS Slot CCC = Axis 7 (HUGO)",
            "reference_pattern_loc": 963,
        },
        # Slot FF sister-cascade anchor
        "slot_ff_sister_cascade_anchor": {
            "commit_sha": "0adecdc5b",
            "sister_pattern_path": (
                "src/tac/composition/pr110_opt_7_uniward_inverse_scorer_"
                "basis_expansion/__init__.py"
            ),
            "axis_position_in_cascade": "Axis 1 (UNIWARD); THIS Slot CCC = Axis 7 (HUGO)",
            "canonical_cauchy_schwarz_meta_lift_acknowledged_partial_overlap": (
                "HUGO (Pevný-Filler-Bas 2010) -> WOW (Holub-Fridrich 2012) -> "
                "UNIWARD (Holub-Fridrich-Denemark 2013/2014) canonical wavelet-"
                "residual lineage descent; canonical orthogonality verification "
                "required before compounding per canonical Slot UU canonical "
                "anti-pattern #3 phantom-compounding warning"
            ),
        },
        # Per-substrate empirical verification stub (Slot QQ META-LESSON)
        "per_substrate_empirical_verification_stub": per_substrate_stub,
        # Sister citation surface
        "design_memo_path": (
            ".omx/research/"
            "hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010_"
            "high_dimensional_spam_feature_canonical_fridrich_yousfi_"
            "cascade_axis_7_extension_per_slot_uu_top_4_design_20260529.md"
        ),
        "horizon_class": "plateau_adjacent",
    }


__all__ = (
    "CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_URL",
    "CANONICAL_PEVNY_FILLER_BAS_2010_CITATION_TEXT",
    "CANONICAL_PEVNY_BAS_FRIDRICH_2010_SPAM_CITATION_TEXT",
    "CANONICAL_4_DIRECTION_OFFSETS",
    "CANONICAL_8_DIRECTION_OFFSETS",
    "CANONICAL_SPAM_TRUNCATION_T",
    "CANONICAL_SPAM_COOCCURRENCE_ORDER",
    "CANONICAL_HUGO_EPSILON",
    "CANONICAL_SPARSE_K_DEFAULT",
    "CANONICAL_WIDENED_K_DEFAULT",
    "CANONICAL_FEC6_BASELINE_WIRE_BYTES",
    "CANONICAL_N_PAIRS",
    "CANONICAL_N_MODES",
    "CANONICAL_RATE_MULTIPLIER",
    "CANONICAL_RATE_DENOM_BYTES",
    "HUGOSPAMFeatureStrategy",
    "HUGOConfig",
    "compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog",
    "apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive",
)
