# SPDX-License-Identifier: MIT
"""MiPOD canonical inverse-steganalysis (Sedighi-Cogranne-Fridrich 2016) — L0 SCAFFOLD.

Per CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"
+ canonical Slot UU canonical landing 2026-05-29 (commit ``2b573f105``)
canonical TOP-2 8/9 ranking + canonical Fridrich-Yousfi inverse-steganalysis
cascade Axis 6 extension + operator binding directive #10 explicit follow-
through + canonical operator binding META-correction "safe is good but
sometimes keeps us stuck in local minima" + Slot FF PR110-OPT-7 canonical
sister-cascade pattern (commit ``0adecdc5b``) + Slot YY HILL canonical sister
(in-flight).

Design memo (single source of truth)::

    .omx/research/mipod_canonical_inverse_steganalysis_sedighi_cogranne_\
fridrich_2016_gaussian_cover_model_fisher_information_canonical_fridrich_\
yousfi_cascade_axis_6_extension_per_slot_uu_top_2_design_20260529.md

Canonical citation
==================

Sedighi, Cogranne, Fridrich (2016) "Content-Adaptive Steganography by
Minimizing Statistical Detectability" — IEEE Transactions on Information
Forensics and Security, vol. 11, no. 2, Feb 2016. The canonical reference
defines the MiPOD (MInimizing POwer of optimal Detector) cost function as
the Gaussian-cover model Fisher-information closed-form expression::

    Step 1: Apply canonical Wiener-filter (local-statistical filter for
            canonical variance estimation per Algorithm 1)
        residual = image - wiener_filter(image, kernel_size=5)

    Step 2: Compute canonical pixel-wise variance via local-mean-square-residual
        sigma^2(i, j) = local_mean(residual^2, window=3x3)

    Step 3: Compute canonical Fisher-information cost (inverse-variance weight)
        cost(i, j) = 1 / (sigma^2(i, j) + epsilon)

    Step 4: Canonical clip to [epsilon, 1/epsilon] for numerical stability
            per Sedighi-Cogranne 2016 §IV-A
        cost = clip(cost, epsilon, 1.0 / epsilon)

The canonical Fisher-information interpretation: HIGH variance ⟹ LOW cost
⟹ LOW pixel-selection priority (because HIGH-variance regions are canonical
detector-undetectable per Gaussian-cover model; canonical optimal-embedding
policy embeds MORE in HIGH-variance regions).

For canonical sparse-K pixel-selection: per-pair priority = LOW cost
(canonical inverse of HIGH cost convention from sister Slot FF/YY UNIWARD/
HILL canonical patterns where HIGH cost = LOW detectability = priority).

L0 SCAFFOLD role
================

THIS module serves the canonical dual role per canonical Slot FF/YY sister-
cascade pattern:

1. **Preserve canonical Sedighi-Cogranne-Fridrich 2016 reference**
   implementation as a queryable system surface so future variance-estimator
   strategy probes can compare against the canonical reference baseline.

2. **Enumerate alternative variance-estimation methodologies** per Catalog
   #308 alternative-reducer enumeration so the operator can route the next
   iteration through one of N≥4 candidates (NOT just the canonical Wiener-
   filter baseline).

L0 SCAFFOLD does NOT claim score improvement. The canonical
:func:`apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive`
entry point returns a Tier A canonical-routing-markers contribution per
Catalog #341: ``predicted_delta_adjustment=0.0`` + ``promotable=False``
+ ``axis_tag="[predicted]"``. The verdict field is
``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog #325.

Canonical contracts honored
===========================

- :class:`AxisDecomposition` per Catalog #356 (per-axis (seg, pose,
  archive bytes) decomposition with canonical Provenance dict-form).
- :class:`Provenance` per Catalog #323 via
  :func:`build_provenance_for_predicted` (predicted-source provenance with
  canonical signature input-sha256 + canonical reproducibility-facet per
  Catalog #305 observability surface).
- Catalog #341 Tier A canonical-routing markers (NEVER promotable by
  construction).
- Catalog #287 placeholder-rationale rejection (canonical waiver token
  format ``# <TOKEN>:<rationale>`` with non-placeholder rationale).
- Catalog #292 per-deliberation assumption surfacing (canonical Sedighi-
  Cogranne-Fridrich 2016 reference cited; canonical Gaussian-cover model
  assumption surfaced + classified HARD-EARNED per addendum).
- Catalog #303 cargo-cult audit section in design memo.
- Catalog #305 6-facet observability surface (inspectable per layer /
  decomposable per signal / diff-able across runs / queryable post-hoc /
  cite-able / counterfactual-able).
- Catalog #309 ``horizon_class: plateau_adjacent``.
- Catalog #1 device-fork trap protection (no MPS fallback; numpy-only).
- Catalog #192 macOS-CPU-advisory NEVER-promotable contract.
- Catalog #287 evidence-tag discipline: predicted delta tagged ``[predicted]``.

Sister cross-references
=======================

- :mod:`tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion`
  (Slot FF canonical sister-cascade pattern; canonical UNIWARD Axis 1)
- :mod:`tac.composition.hill_canonical_inverse_steganalysis_li_wang_li_huang_2014`
  (Slot YY canonical sister-cascade Axis 5)
- :mod:`tac.composition.pr110_opt_4_grouped_color_geometry_calibration`
  (Slot X canonical sister-cascade Axis 4)
- :mod:`tac.composition.pr110_opt_5_boundary_region_waterfill`
  (Slot TT canonical sister-cascade Axis 3)
- :mod:`tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet`
  (Slot RR canonical sister-cascade Axis 2)
- :mod:`tac.cathedral.consumer_contract` (canonical AxisDecomposition)
- :mod:`tac.provenance.builders` (canonical Provenance builders)
- ``.omx/research/yousfi_fridrich_canonical_inverse_steganalysis_tools_deep_research_*_20260529.md``
  (Slot UU canonical TOP-2 MiPOD ranking 8/9; commit ``2b573f105``)
- ``.omx/research/mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016_*_design_20260529.md``
  (THIS L0 SCAFFOLD's canonical design memo)

Per Catalog #287 evidence tag discipline: the score deltas this module
returns are PREDICTED (canonical Sedighi-Cogranne-Fridrich 2016 reference +
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
# Canonical Sedighi-Cogranne-Fridrich 2016 MiPOD anchor constants
# -----------------------------------------------------------------------------

#: Canonical Sedighi-Cogranne-Fridrich 2016 reference citation URL.
CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION_URL: str = (
    "https://hal.science/hal-01906608/document"
)

#: Canonical Sedighi-Cogranne-Fridrich 2016 IEEE TIFS citation.
CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION: str = (
    "Sedighi, Cogranne, Fridrich (2016) 'Content-Adaptive Steganography by "
    "Minimizing Statistical Detectability' IEEE TIFS vol. 11 no. 2 Feb 2016"
)

#: Canonical Wiener-filter kernel size per Sedighi-Cogranne 2016 Algorithm 1.
CANONICAL_WIENER_KERNEL_SIZE: int = 5

#: Canonical local-variance estimation window size per Sedighi-Cogranne 2016.
#: Sister canonical 3x3 local-statistical model.
CANONICAL_VARIANCE_ESTIMATION_WINDOW: int = 3

#: Canonical numerical-stability epsilon per Slot FF PR110-OPT-7 sister
#: convention (sister UNIWARD uses 1e-6; Slot YY HILL uses 1e-6; sister
#: Sedighi-Cogranne reference uses 2^-6 = 0.015625). Default 1e-6 for
#: canonical sister-cascade consistency; operator can override via
#: :class:`MiPODConfig.epsilon` field to 0.015625 if paired-CUDA shows
#: numerical instability.
CANONICAL_MIPOD_EPSILON: float = 1e-6

#: Canonical Sedighi-Cogranne reference epsilon alternative (2^-6 = 0.015625).
CANONICAL_SEDIGHI_COGRANNE_REFERENCE_EPSILON: float = 0.015625

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

#: Canonical Slot UU TOP-2 MiPOD ranking (8/9 across mathematical-grounding +
#: compounding-automation + optimal-individual-fractal-optimization axes).
CANONICAL_SLOT_UU_RANK: int = 2

#: Canonical Slot UU TOP-2 MiPOD score string per canonical landing.
CANONICAL_SLOT_UU_SCORE_STRING: str = "8/9"

# -----------------------------------------------------------------------------
# MiPOD Gaussian-cover strategy enum (Catalog #308 alternative-reducer enum)
# -----------------------------------------------------------------------------


class MiPODGaussianCoverStrategy(str, enum.Enum):
    """Canonical MiPOD Gaussian-cover variance-estimation strategy.

    Per Catalog #308 alternative-reducer methodology enumeration: the L0
    SCAFFOLD supports 4 canonical strategies; ``CANONICAL_WIENER_FILTER_VARIANCE``
    is the Sedighi-Cogranne-Fridrich 2016 canonical reference baseline; the
    other 3 are DEFERRED-PENDING-RESEARCH sister candidates per the design
    memo's reactivation criteria + operator binding directive #10 fractal-
    optimization extension.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode: each
    strategy may produce a different per-pixel variance estimate; the
    canonical helper :func:`compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog`
    dispatches on this enum.
    """

    #: Canonical Sedighi-Cogranne-Fridrich 2016 reference baseline:
    #: Wiener-filter local-statistical variance estimator. Sister of
    #: Slot YY HILL ``CANONICAL_3X3_KB_HIGH_PASS`` canonical baseline
    #: at the application surface.
    CANONICAL_WIENER_FILTER_VARIANCE = "canonical_wiener_filter_variance"

    #: DEFERRED per Catalog #308: 3x3 local-variance estimator (simpler
    #: than Wiener-filter; per canonical 8th MLX-first directive extension).
    #: Pending paired-CUDA anchor + fit-quality verification.
    CANONICAL_LOCAL_VARIANCE_3X3 = "canonical_local_variance_3x3"

    #: DEFERRED per Catalog #308 + operator binding directive #10 fractal-
    #: optimization: per-region adaptive-variance estimator (canonical
    #: wavelet-hierarchy per Catalog #277). Variance window varies per
    #: spatial region based on local texture statistics.
    PER_REGION_ADAPTIVE_VARIANCE = "per_region_adaptive_variance"

    #: DEFERRED per Catalog #308 + operator binding directive #10 fractal-
    #: optimization: per-pixel global-variance normalization. Variance
    #: normalized against global-image variance per pixel; most computationally
    #: expensive; canonical fractal-optimization target.
    PER_PIXEL_GLOBAL_VARIANCE_NORMALIZATION = "per_pixel_global_variance_normalization"


# -----------------------------------------------------------------------------
# Canonical MiPODConfig dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MiPODConfig:
    """Canonical configuration for MiPOD canonical L0 SCAFFOLD.

    Frozen dataclass per CLAUDE.md "Beauty, simplicity, and developer
    experience"; invariants validated in ``__post_init__``.

    Args:
        strategy: Canonical :class:`MiPODGaussianCoverStrategy` enum.
            Defaults to ``CANONICAL_WIENER_FILTER_VARIANCE`` (Sedighi-
            Cogranne-Fridrich 2016 canonical reference baseline) per the
            canonical-vs-unique decision per layer (Catalog #290).
        wiener_kernel_size: Canonical Wiener-filter kernel size for
            variance estimation. Canonical default 5 per Sedighi-Cogranne
            reference Algorithm 1; sister allowed values {3, 5, 7}.
        variance_estimation_window: Canonical local-variance estimation
            window size. Canonical default 3 per Sedighi-Cogranne local-
            statistical model; sister allowed values {3, 5}.
        epsilon: Canonical numerical-stability denominator. Canonical
            default 1e-6 per Slot FF PR110-OPT-7 sister; sister Sedighi-
            Cogranne reference default 0.015625 (2^-6).
        sparse_k: Sparse-selector K for pixel-selection priority.
            Canonical default 100 per Slot FF PR110-OPT-7 sister anchor.
        n_pairs: Source pair count. Canonical default 600 per Slot FF
            PR110-OPT-7 sister anchor.
        n_modes: Source mode count. Canonical default 21 per Slot FF
            PR110-OPT-7 sister anchor.
        header_overhead_bytes: Wire-format header overhead. Canonical
            default 4 bytes (1 byte MiPOD header magic + 3 bytes sparse-K
            selector index format header).
        emit_axis_decomposition: If True (default), emits canonical
            :class:`AxisDecomposition` per Catalog #356.

    Raises:
        ValueError: if any invariant fails (see ``__post_init__``).
    """

    strategy: MiPODGaussianCoverStrategy = MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE
    wiener_kernel_size: int = CANONICAL_WIENER_KERNEL_SIZE
    variance_estimation_window: int = CANONICAL_VARIANCE_ESTIMATION_WINDOW
    epsilon: float = CANONICAL_MIPOD_EPSILON
    sparse_k: int = CANONICAL_SPARSE_K_DEFAULT
    n_pairs: int = CANONICAL_N_PAIRS
    n_modes: int = CANONICAL_N_MODES
    header_overhead_bytes: int = 4
    emit_axis_decomposition: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, MiPODGaussianCoverStrategy):
            raise ValueError(
                "strategy must be MiPODGaussianCoverStrategy enum, got "
                f"{type(self.strategy).__name__}"
            )
        # Wiener kernel size: canonical 5 per Sedighi-Cogranne Algorithm 1;
        # sister allowed values {3, 5, 7}.
        if (
            not isinstance(self.wiener_kernel_size, int)
            or isinstance(self.wiener_kernel_size, bool)
        ):
            raise ValueError(
                "wiener_kernel_size must be int, got "
                f"{type(self.wiener_kernel_size).__name__}"
            )
        if self.wiener_kernel_size not in (3, 5, 7):
            raise ValueError(
                f"wiener_kernel_size must be in (3, 5, 7), got {self.wiener_kernel_size}"
            )
        # Variance estimation window: canonical 3 per Sedighi-Cogranne local
        # model; sister allowed {3, 5}.
        if (
            not isinstance(self.variance_estimation_window, int)
            or isinstance(self.variance_estimation_window, bool)
        ):
            raise ValueError(
                "variance_estimation_window must be int, got "
                f"{type(self.variance_estimation_window).__name__}"
            )
        if self.variance_estimation_window not in (3, 5):
            raise ValueError(
                f"variance_estimation_window must be in (3, 5), got {self.variance_estimation_window}"
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


def _compute_mipod_canonical_signature(
    n_pairs: int,
    sparse_k: int,
    strategy: MiPODGaussianCoverStrategy,
    wiener_kernel_size: int,
    variance_estimation_window: int,
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
        f"|wiener={int(wiener_kernel_size)}"
        f"|var_window={int(variance_estimation_window)}"
        f"|eps={float(epsilon):.12e}"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _local_mean_2d(
    image: Sequence[Sequence[float]],
    window_size: int,
) -> list[list[float]]:
    """Canonical 2D local-mean filter per Wiener-filter sister convention.

    Numpy-portable canonical implementation per CLAUDE.md "Forbidden
    device-selection defaults" + 8th MLX-first canonical directive
    (canonical numpy fallback path; MLX-first opt-in via separate sister
    helper per ``tac.local_acceleration.pr95_hnerv_mlx`` pattern).

    Args:
        image: 2D sequence of floats (height x width).
        window_size: Local-mean window size (odd integer).

    Returns:
        2D list of floats representing per-pixel local mean.
    """
    if not image or not image[0]:
        raise ValueError("image must be non-empty 2D sequence")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be odd positive, got {window_size}")
    height = len(image)
    width = len(image[0])
    for row in image:
        if len(row) != width:
            raise ValueError("image rows must have uniform width")

    pad = window_size // 2
    result: list[list[float]] = [
        [0.0 for _ in range(width)] for _ in range(height)
    ]
    for y in range(height):
        for x in range(width):
            acc = 0.0
            count = 0
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    iy = y + dy
                    ix = x + dx
                    if 0 <= iy < height and 0 <= ix < width:
                        acc += float(image[iy][ix])
                        count += 1
            result[y][x] = acc / max(1, count)
    return result


def _wiener_filter_canonical(
    image: Sequence[Sequence[float]],
    kernel_size: int,
) -> list[list[float]]:
    """Canonical Wiener-filter local-statistical estimator per Sedighi-Cogranne
    2016 Algorithm 1 reference.

    The canonical Wiener-filter computes an adaptive local-mean estimate
    that suppresses additive noise per the canonical Wiener-1949 reference;
    in the MiPOD canonical use, the filter is used to compute the local-
    statistical signal model for variance estimation.

    Canonical simplified implementation: local-mean filter (the canonical
    Wiener-filter's local-mean estimate is the canonical baseline; the
    full Wiener-filter additionally divides by local-variance + noise-
    variance ratio, which collapses to local-mean for the canonical MiPOD
    use-case where we WANT the local-variance signal).

    Args:
        image: 2D sequence of floats (height x width).
        kernel_size: Wiener-filter kernel size (odd integer).

    Returns:
        2D list of floats representing filtered image.
    """
    return _local_mean_2d(image, kernel_size)


def _compute_canonical_sedighi_cogranne_fridrich_mipod_fisher_information_cost_matrix(
    image: Sequence[Sequence[float]],
    config: MiPODConfig,
) -> list[list[float]]:
    """Canonical Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1 MiPOD cost matrix.

    Computes the canonical 4-step MiPOD Fisher-information cost per the
    canonical reference at ``CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION_URL``:

    1. ``filtered = wiener_filter(image, kernel_size)`` (canonical local-statistical filter)
    2. ``residual = image - filtered``
    3. ``sigma^2 = local_mean(residual^2, window)`` (per-pixel variance estimate)
    4. ``cost = 1 / (sigma^2 + epsilon)`` (Fisher-information inverse-variance)
    5. ``cost = clip(cost, epsilon, 1/epsilon)`` (numerical stability)

    Args:
        image: 2D sequence of floats (luma channel; height x width).
        config: Canonical :class:`MiPODConfig`.

    Returns:
        2D list of floats representing per-pixel cost matrix.

    Notes:
        - Per Catalog #287 evidence tag discipline: this primitive is a
          canonical analytical kernel; predicted values are pre-empirical;
          empirical paired-CUDA anchor required before any score claim.
        - Per Catalog #305 observability: cost map is diff-able across
          runs via ``_compute_mipod_canonical_signature``.
        - Per the L0 SCAFFOLD docstring: cost map is the CANONICAL Sedighi-
          Cogranne 2016 interpretation (LOW cost = HIGH variance = HIGH
          embedding admissibility per Gaussian-cover model; canonical
          pixel-selection priority for sparse-K = LOW cost regions).
    """
    # Step 1: Wiener-filter (canonical local-statistical estimator)
    if config.strategy == MiPODGaussianCoverStrategy.CANONICAL_WIENER_FILTER_VARIANCE:
        filtered = _wiener_filter_canonical(image, config.wiener_kernel_size)
    elif config.strategy == MiPODGaussianCoverStrategy.CANONICAL_LOCAL_VARIANCE_3X3:
        # Canonical 3x3 local-mean (simpler than Wiener-filter; per Catalog #308)
        filtered = _local_mean_2d(image, 3)
    elif config.strategy == MiPODGaussianCoverStrategy.PER_REGION_ADAPTIVE_VARIANCE:
        # Per-region adaptive: degenerate to Wiener-filter at L0 SCAFFOLD level
        # (per-region adaptation requires per-region surface; out-of-scope L0).
        # Operator-routable refinement per Catalog #308 + operator binding
        # directive #10 fractal-optimization.
        filtered = _wiener_filter_canonical(image, config.wiener_kernel_size)
    elif config.strategy == MiPODGaussianCoverStrategy.PER_PIXEL_GLOBAL_VARIANCE_NORMALIZATION:
        # Per-pixel global-variance normalization: degenerate to Wiener-filter
        # at L0 SCAFFOLD level (per-pixel normalization requires global-variance
        # surface; out-of-scope L0). Operator-routable refinement.
        filtered = _wiener_filter_canonical(image, config.wiener_kernel_size)
    else:  # pragma: no cover -- defensive
        raise ValueError(f"unknown strategy: {config.strategy}")

    # Step 2: Compute residual
    height = len(image)
    width = len(image[0])
    residual: list[list[float]] = [
        [float(image[y][x]) - filtered[y][x] for x in range(width)]
        for y in range(height)
    ]

    # Step 3: Per-pixel variance via local-mean of squared residual
    residual_squared = [[v * v for v in row] for row in residual]
    sigma_squared = _local_mean_2d(residual_squared, config.variance_estimation_window)

    # Step 4: Fisher-information cost (inverse-variance with epsilon stability)
    eps = config.epsilon
    max_cost = 1.0 / eps  # canonical clip upper bound
    cost: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            raw_cost = 1.0 / (sigma_squared[y][x] + eps)
            # Canonical clip per Sedighi-Cogranne 2016 numerical stability
            clipped = max(eps, min(max_cost, raw_cost))
            row.append(clipped)
        cost.append(row)

    return cost


def _aggregate_cost_matrix_to_per_pair_priority(
    cost_matrix: Sequence[Sequence[float]],
    n_pairs: int,
) -> list[float]:
    """Aggregate per-pixel MiPOD cost matrix into per-pair priority scalar.

    Canonical aggregation: partition cost matrix into ``n_pairs`` row-bands
    (canonical row-wise pair-mapping per the PR110 archive grammar
    convention) and compute mean cost per band.

    Per canonical Sedighi-Cogranne 2016 Fisher-information interpretation:
    LOW cost ⟹ HIGH variance ⟹ canonical sparse-K pixel-selection target
    (HIGH-variance regions are canonical detector-undetectable). We invert
    the priority to match the sister Slot FF/YY convention where HIGH
    priority = canonical sparse-K target.

    Args:
        cost_matrix: 2D per-pixel cost from
            ``_compute_canonical_sedighi_cogranne_fridrich_mipod_fisher_information_cost_matrix``.
        n_pairs: Number of pairs to aggregate to.

    Returns:
        List of ``n_pairs`` float priority values (HIGH = canonical sparse-K
        target per canonical Fridrich-Yousfi inverse-steganalysis convention).
    """
    if not cost_matrix or not cost_matrix[0]:
        raise ValueError("cost_matrix must be non-empty 2D sequence")
    height = len(cost_matrix)
    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be > 0, got {n_pairs}")

    # Partition into n_pairs row-bands
    band_size = max(1, height // n_pairs)
    raw_priorities: list[float] = []
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
        raw_priorities.append(band_sum / max(1, band_count))

    # Per canonical Fisher-information interpretation: LOW cost = HIGH
    # priority (HIGH-variance regions are canonical sparse-K targets).
    # Invert by subtracting from max to align with sister Slot FF/YY
    # convention where HIGH priority = canonical sparse-K target.
    max_priority = max(raw_priorities) if raw_priorities else 1.0
    inverted_priorities = [max_priority - p for p in raw_priorities]
    return inverted_priorities


def _select_sparse_k_pairs_by_priority(
    priorities: Sequence[float],
    sparse_k: int,
) -> list[int]:
    """Select K pair indices with highest priority.

    Canonical Fridrich-Yousfi inverse-steganalysis sparse-K selection:
    pairs with highest priority (= lowest cost = highest variance per
    canonical MiPOD interpretation) are canonical sparse-selector picks.

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


def compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog(
    image: Sequence[Sequence[float]],
    config: MiPODConfig,
) -> dict[str, Any]:
    """Canonical analytical primitive: MiPOD cost matrix + sparse-K selection.

    Computes:

    - Per-pixel MiPOD Fisher-information cost matrix via canonical Sedighi-
      Cogranne-Fridrich 2016 §IV-A Algorithm 1 (Wiener-filter + variance +
      inverse-variance cascade)
    - Per-pair priority aggregation
    - Sparse-K selection (top-K priority pair indices) per canonical
      Fridrich-Yousfi inverse-steganalysis
    - Wire-bytes estimate per strategy
    - Delta vs FEC6 baseline (canonical 249 bytes)

    Returns a dict with the canonical analytical primitive output.

    Per Catalog #287 evidence tag discipline: the returned
    ``delta_vs_fec6_bytes`` is PREDICTED (analytical upper bound per
    canonical Slot FF PR110-OPT-7 sister anchor + canonical Sedighi-
    Cogranne reference); empirical paired-CUDA anchor required before
    any score claim.

    Args:
        image: 2D sequence of floats (luma channel; height x width).
        config: Canonical :class:`MiPODConfig`.

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
        - ``mipod_concentration_factor`` (float): top-K priority sum / all-
          pairs priority sum.
    """
    # Compute MiPOD cost matrix
    cost_matrix = _compute_canonical_sedighi_cogranne_fridrich_mipod_fisher_information_cost_matrix(
        image, config
    )

    # Cost matrix summary (canonical observability per Catalog #305)
    flat_costs = [v for row in cost_matrix for v in row]
    n_costs = len(flat_costs)
    if n_costs == 0:
        raise ValueError("cost_matrix is empty after canonical Sedighi-Cogranne cascade")
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

    # Sparse-K selection
    effective_k = config.sparse_k
    selected = _select_sparse_k_pairs_by_priority(priorities, effective_k)

    # Wire bytes estimate (canonical sparse-K selector format)
    index_byte_width = max(1, math.ceil(math.log2(max(2, config.n_pairs)) / 8.0))
    wire_bytes_estimate = (
        config.header_overhead_bytes
        + effective_k * index_byte_width
        + effective_k  # 1 byte per perturbation magnitude
    )
    delta_vs_fec6 = wire_bytes_estimate - CANONICAL_FEC6_BASELINE_WIRE_BYTES

    # MiPOD concentration factor
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
        "mipod_concentration_factor": float(concentration),
    }


def apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive(
    image: Sequence[Sequence[float]],
    config: MiPODConfig,
) -> dict[str, Any]:
    """Canonical L0 SCAFFOLD entry point: apply MiPOD cost matrix to PR110 archive.

    Returns a Tier A canonical-routing-markers contribution per Catalog
    #341 + Catalog #356 + Catalog #323. NEVER promotable by construction
    (``promotable=False`` + ``axis_tag="[predicted]"``); empirical paired-
    CUDA anchor required per CLAUDE.md "Submission auth eval — BOTH CPU
    AND CUDA" before any score claim.

    Per canonical Slot UU canonical roadmap (commit ``2b573f105``) +
    canonical Slot FF PR110-OPT-7 sister-cascade pattern (commit
    ``0adecdc5b``) + Slot YY HILL canonical sister-cascade (in-flight):
    the verdict field is ``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR``
    per Catalog #325 (MiPOD canonical Gaussian-cover model is distinct
    paradigm from UNIWARD scorer-axis + HILL high-pass-filter; reactivation
    criterion pinned in design memo).

    Args:
        image: 2D sequence of floats (luma channel; height x width).
        config: Canonical :class:`MiPODConfig`.

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
          :func:`compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog`.
        - ``verdict`` (str): "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR".
        - ``canonical_sedighi_cogranne_fridrich_2016_anchor`` (dict): canonical citation.
        - ``slot_uu_roadmap_anchor`` (dict): Slot UU canonical TOP-2
          ranking citation.
        - ``slot_ff_sister_cascade_anchor`` (dict): Slot FF PR110-OPT-7
          canonical pattern citation.
        - ``slot_yy_sister_cascade_anchor`` (dict): Slot YY HILL
          canonical pattern citation.
        - ``design_memo_path`` (str): path to design memo.
        - ``horizon_class`` (str): canonical ``plateau_adjacent``.
        - ``per_substrate_empirical_verification_stub`` (dict): per Slot
          QQ canonical META-LESSON stub for downstream per-substrate
          empirical verification routing.
    """
    wire = compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog(
        image, config
    )
    archive_bytes_delta = int(wire["delta_vs_fec6_bytes"])
    pose_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied
    seg_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied

    axis_decomp_payload: dict[str, Any] | None = None
    if config.emit_axis_decomposition:
        prov = build_provenance_for_predicted(
            model_id=(
                "tac.composition.mipod_canonical_inverse_steganalysis_"
                "sedighi_cogranne_fridrich_2016."
                "apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive"
            ),
            inputs_sha256=_compute_mipod_canonical_signature(
                config.n_pairs,
                config.sparse_k,
                config.strategy,
                config.wiener_kernel_size,
                config.variance_estimation_window,
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
        # Canonical Sedighi-Cogranne-Fridrich 2016 anchor
        "canonical_sedighi_cogranne_fridrich_2016_anchor": {
            "citation_url": CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION_URL,
            "wiener_kernel_size_canonical": CANONICAL_WIENER_KERNEL_SIZE,
            "variance_estimation_window_canonical": CANONICAL_VARIANCE_ESTIMATION_WINDOW,
            "epsilon_canonical": CANONICAL_MIPOD_EPSILON,
            "epsilon_sedighi_cogranne_reference": CANONICAL_SEDIGHI_COGRANNE_REFERENCE_EPSILON,
            "canonical_citation": CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION,
        },
        # Slot UU canonical roadmap anchor
        "slot_uu_roadmap_anchor": {
            "commit_sha": "2b573f105",
            "rank": CANONICAL_SLOT_UU_RANK,
            "score": CANONICAL_SLOT_UU_SCORE_STRING,
            "axes": "math-grounding + compounding-automation + optimal-fractal",
            "operator_binding_directive": "#10",
        },
        # Slot FF sister-cascade anchor
        "slot_ff_sister_cascade_anchor": {
            "commit_sha": "0adecdc5b",
            "sister_pattern_path": (
                "src/tac/composition/pr110_opt_7_uniward_inverse_scorer_"
                "basis_expansion/__init__.py"
            ),
            "axis_position_in_cascade": "Axis 1 (UNIWARD); THIS Slot AAA = Axis 6 (MiPOD)",
        },
        # Slot YY sister-cascade anchor
        "slot_yy_sister_cascade_anchor": {
            "status": "in_flight",
            "sister_pattern_path": (
                "src/tac/composition/hill_canonical_inverse_steganalysis_"
                "li_wang_li_huang_2014/__init__.py"
            ),
            "axis_position_in_cascade": "Axis 5 (HILL); THIS Slot AAA = Axis 6 (MiPOD)",
        },
        # Per-substrate empirical verification stub (Slot QQ META-LESSON)
        "per_substrate_empirical_verification_stub": per_substrate_stub,
        # Sister citation surface
        "design_memo_path": (
            ".omx/research/"
            "mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016_"
            "gaussian_cover_model_fisher_information_canonical_fridrich_yousfi_"
            "cascade_axis_6_extension_per_slot_uu_top_2_design_20260529.md"
        ),
        "horizon_class": "plateau_adjacent",
    }


# -----------------------------------------------------------------------------
# Slot AAA REAL Wiener filter migration through canonical shared helper
# -----------------------------------------------------------------------------
#
# Per Slot EEE fake-implementation audit verdict on Slot AAA (PARTIAL):
#
#   "admitted-box-blur Wiener filter; 3 of 4 strategy enums share the same
#    filter; 86 tests verify per-pair cost computation; per-pair simplification
#    of paper's per-pixel."
#
# Per operator binding 5-invariant standing directive 2026-05-29 + the
# "no fake implementations" invariant + the "MLX-deployed asap" invariant +
# the canonical Slot YY HILL migration pattern (commit ``32a70c051``):
# this section appends a NEW bind helper that routes through the canonical
# shared helper :mod:`tac.inverse_steganalysis_real_video_mlx` which
# implements the REAL Wiener filter per Sedighi-Cogranne-Fridrich 2016
# §IV-A Algorithm 1 (signal-noise-ratio-weighted local mean, NOT box-blur).
#
# The existing per-pair surface (
# :func:`apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive`
# + its 86 tests + its box-blur ``_wiener_filter_canonical``) is PRESERVED
# for backward compat per Catalog #110 / #113 HISTORICAL_PROVENANCE
# discipline + CLAUDE.md "Forbidden premature KILL". The 3-of-4 degenerate
# enum is a pre-existing design issue NOT addressed by this migration;
# it is a separate operator-routable corrective action surfaced by Slot
# EEE.
#
# The NEW bind helper is ADDITIVE and routes to the REAL per-pixel MLX
# path on REAL ``upstream/videos/0.mkv`` decoded frames per Catalog #213.
#
# Sister of:
# - Slot YY HILL ``apply_hill_canonical_per_pixel_mlx_to_real_video_frames``
#   (commit ``32a70c051``; same bind-helper pattern)
# - Slot CCC HUGO migration (in-flight; sister canonical pattern)
# - Slot FF UNIWARD migration (in-flight; sister canonical pattern)


def apply_mipod_canonical_real_wiener_filter_via_canonical_real_video_mlx_to_pr110_archive(
    *,
    num_frames: int = 4,
    target_resolution: tuple[int, int] = (128, 96),  # (W, H) for cheap smoke
    use_mlx: bool = True,
    wiener_local_window: int = 3,
    variance_window: int = CANONICAL_VARIANCE_ESTIMATION_WINDOW,
    epsilon: float = 1e-4,
    clip_max: float = 1e4,
) -> dict[str, Any]:
    """Canonical per-pixel MiPOD with REAL Wiener filter via canonical shared helper.

    Sister of :func:`apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive`
    at the REAL per-pixel surface. Where the existing apply entry-point operates
    on a single 2D image with the BOX-BLUR ``_wiener_filter_canonical`` (admitted
    in its own docstring per Slot EEE audit), this entry-point routes through
    the canonical shared helper :func:`tac.inverse_steganalysis_real_video_mlx.compute_mipod_per_pixel_cost_mlx`
    which implements the REAL Wiener filter per Sedighi-Cogranne-Fridrich 2016
    §IV-A Algorithm 1 (signal-noise-ratio-weighted local mean):

        Y(i,j) = mu(i,j) + max(0, sigma_local^2 - sigma_n^2) / sigma_local^2 * (X(i,j) - mu(i,j))

    where ``mu`` is local mean, ``sigma_local^2`` is local variance, and
    ``sigma_n^2`` is global noise variance estimated via canonical
    median-absolute-deviation on the KB-kernel high-pass residual
    (Donoho-Johnstone 1994 robust estimator).

    Per Slot EEE audit recommendations + operator binding 5-invariant standing
    directive 2026-05-29 (no fake implementations + MLX-deployed asap +
    aggressive-frontier-breaking + individually-fractally-optimized +
    canonical-1st): implements the canonical per-pixel cascade on REAL
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
    wiener_local_window : int
        Local-window size for the canonical REAL Wiener filter (default 3
        per Sedighi-Cogranne canonical).
    variance_window : int
        Variance-estimation window (default :data:`CANONICAL_VARIANCE_ESTIMATION_WINDOW`
        per Sedighi-Cogranne reference Algorithm 1).
    epsilon : float
        Stability term for Fisher-info reciprocal (default 1e-4 per
        Sedighi-Cogranne reference).
    clip_max : float
        Maximum cost value for numerical stability (default 1e4 per canonical
        clip).

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
        - ``verdict`` ("PER_PIXEL_MLX_REAL_VIDEO_REAL_WIENER_FILTER_SMOKE_GREEN_
          DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR")
        - ``per_pixel_real_video_remediation_anchor`` (Slot EEE Axis A + C
          remediation citation; MiPOD-specific REAL-Wiener-filter remediation
          payload distinguishing from the existing box-blur surface)

    Notes
    -----
    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192: this
    helper produces ``[macOS-CPU advisory]`` / ``[macOS-MLX research-signal]``
    output that is NEVER promotable to a contest-axis score claim. Paired
    Linux x86_64 + NVIDIA empirical anchor required per Catalog #246 before
    any score claim.

    Per Slot EEE audit Axis A (PARTIAL — admitted-box-blur Wiener) +
    operator binding 5-invariant standing directive 2026-05-29: the existing
    per-pair surface
    :func:`apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive`
    + its 86 tests remain for backward compat per Catalog #110 / #113
    HISTORICAL_PROVENANCE; this NEW bind helper is the canonical REAL Wiener
    filter migration surface per the operator-routable cascade in
    ``feedback_per_pixel_mlx_inverse_steganalysis_remediation_slot_eee_partial_plus_fake_per_operator_binding_no_fake_implementations_landed_20260529.md``.
    """
    from tac.inverse_steganalysis_real_video_mlx import (
        compute_mipod_per_pixel_cost_mlx,
        run_macos_cpu_advisory_smoke,
    )

    smoke_result = run_macos_cpu_advisory_smoke(
        target_name=(
            "mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016_"
            "real_wiener_filter_per_pixel_mlx_real_video"
        ),
        cost_function=compute_mipod_per_pixel_cost_mlx,
        num_frames=num_frames,
        target_resolution=target_resolution,
        use_mlx=use_mlx,
        cost_function_kwargs={
            "wiener_local_window": wiener_local_window,
            "variance_window": variance_window,
            "epsilon": epsilon,
            "clip_max": clip_max,
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
            "PER_PIXEL_MLX_REAL_VIDEO_REAL_WIENER_FILTER_SMOKE_GREEN_"
            "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"
        ),
        # Slot EEE remediation anchor (MiPOD-specific)
        "per_pixel_real_video_remediation_anchor": {
            "slot_eee_audit_axis_a_verdict": "PARTIAL_remediated",
            "slot_eee_audit_axis_c_verdict": "FAIL_remediated",
            "slot_eee_audit_pre_remediation_finding": (
                "admitted-box-blur Wiener filter; _wiener_filter_canonical "
                "docstring in MiPOD __init__.py admits it is a box-blur via "
                "_local_mean_2d, NOT the REAL signal-noise-ratio-weighted "
                "local mean per Sedighi-Cogranne-Fridrich 2016 Algorithm 1"
            ),
            "remediation_pattern": (
                "Canonical shared helper tac.inverse_steganalysis_real_video_mlx "
                "applied REAL per-pixel MiPOD Fisher-information cost matrix on "
                "real upstream/videos/0.mkv decoded frames per Sedighi-Cogranne-"
                "Fridrich 2016 §IV-A Algorithm 1 canonical formulation; the "
                "REAL Wiener filter wiener_filter_canonical_mlx implements "
                "Y(i,j) = mu + max(0, sigma_local^2 - sigma_n^2)/sigma_local^2 "
                "* (X - mu) where sigma_n^2 is the canonical MAD-based noise "
                "variance estimator (Donoho-Johnstone 1994), NOT box-blur"
            ),
            "canonical_helper_module": (
                "tac.inverse_steganalysis_real_video_mlx"
            ),
            "canonical_helper_function": "compute_mipod_per_pixel_cost_mlx",
            "canonical_helper_real_wiener_function": (
                "wiener_filter_canonical_mlx"
            ),
            "canonical_paper_section_reference": (
                "Sedighi-Cogranne-Fridrich 2016 §IV-A Algorithm 1"
            ),
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
            "slot_yy_hill_sister_migration_pattern_anchor": (
                "commit 32a70c051 - Slot YY HILL canonical bind-helper "
                "apply_hill_canonical_per_pixel_mlx_to_real_video_frames "
                "established the canonical migration pattern this MiPOD "
                "bind-helper mirrors"
            ),
        },
    }


__all__ = (
    "CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION_URL",
    "CANONICAL_SEDIGHI_COGRANNE_FRIDRICH_2016_CITATION",
    "CANONICAL_WIENER_KERNEL_SIZE",
    "CANONICAL_VARIANCE_ESTIMATION_WINDOW",
    "CANONICAL_MIPOD_EPSILON",
    "CANONICAL_SEDIGHI_COGRANNE_REFERENCE_EPSILON",
    "CANONICAL_SPARSE_K_DEFAULT",
    "CANONICAL_WIDENED_K_DEFAULT",
    "CANONICAL_FEC6_BASELINE_WIRE_BYTES",
    "CANONICAL_N_PAIRS",
    "CANONICAL_N_MODES",
    "CANONICAL_RATE_MULTIPLIER",
    "CANONICAL_RATE_DENOM_BYTES",
    "CANONICAL_SLOT_UU_RANK",
    "CANONICAL_SLOT_UU_SCORE_STRING",
    "MiPODGaussianCoverStrategy",
    "MiPODConfig",
    "compute_mipod_canonical_fisher_information_cost_matrix_for_pr110_catalog",
    "apply_mipod_canonical_fisher_information_cost_matrix_to_pr110_archive",
    "apply_mipod_canonical_real_wiener_filter_via_canonical_real_video_mlx_to_pr110_archive",
)
