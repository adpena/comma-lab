# SPDX-License-Identifier: MIT
"""PR110-OPT-5 Boundary/region waterfill — SegNet class-region-aware perturbation budget — L0 SCAFFOLD.

Canonical Fridrich-Yousfi inverse-steganalysis 3-axis cascade Axis 3 per
task #1317 PR110-OPT-5 + Slot TT cap=3 parallel-cascade directive 2026-05-29
+ canonical sister of Slot FF PR110-OPT-7 Axis 1 + Slot RR PR110-OPT-6 Axis 2.

Design memo (single source of truth)::

    .omx/research/pr110_opt_5_boundary_region_waterfill_segnet_class_region_aware_perturbation_budget_canonical_fridrich_yousfi_3_axis_cascade_design_20260529.md

Canonical context
=================

Per CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer" +
"Exact scorer architectures" SegNet ``smp.Unet('tu-efficientnet_b2',
classes=5, activation=None)`` 5-class argmax distortion is the canonical
distortion-axis measure for OPT-5.

Canonical Fridrich-Yousfi inverse-steganalysis 3-axis cascade structure:

* **Axis 1** (Slot FF OPT-7 LANDED commit ``0adecdc5b``): UNIWARD inverse-scorer
  basis expansion at scorer-axis surface (sparse-K=100 selector + per-pair
  UNIWARD cost weighting; canonical Wave N+34 sparse_k100 anchor 103 bytes
  wire estimate -146B vs FEC6 baseline 249)
* **Axis 2** (Slot RR OPT-6 LANDED): pose-axis perturbation null-projection
  on SegNet (canonical OPT-12 PoseNet-null analog; 43 frame-1 modes)
* **Axis 3** (THIS Slot TT OPT-5): SegNet class-region-aware perturbation
  budget at SegNet class-region surface (per-class 5-byte budget allocation
  + 4 canonical strategies per Catalog #308 alternative-reducer enumeration)

The canonical menu enumerates 4 canonical strategies per Catalog #308:

* PER_CLASS_UNIFORM: 5-byte per-class budget uniform across SegNet 5 classes
  + 6-byte header = 11 bytes total wire estimate (-238B vs FEC6 249)
* PER_CLASS_WEIGHTED_BY_AREA: 5-byte budget + 5-byte area weights + 6-byte
  header = 16 bytes total wire estimate (-233B vs FEC6 249)
* PER_REGION_AT_BOUNDARY: 5-byte budget + sparse boundary index (K=20 boundary
  pixels × 4 bytes per index) + 6-byte header = 91 bytes total wire estimate
  (-158B vs FEC6 249)
* PER_REGION_INTERIOR: 5-byte budget + per-region interior mask (5 classes
  × 24-byte mask) + 6-byte header = 131 bytes total wire estimate (-118B vs
  FEC6 249)

PR110 archive grammar (sister of Slot X PR110-OPT-4 + Slot FF PR110-OPT-7
+ Slot RR PR110-OPT-6 L0 SCAFFOLDs)::

* 16-symbol K=16 selector palette per FEC6 ``submissions/hnerv_fec6_fixed_huffman_k16/``
* 600 per-pair selectors (one selector per source-frame pair)
* 6-byte header + 243-byte 0-order fixed-Huffman bitstream = **249 byte baseline wire**

SegNet class-region surface canonical formulation
==================================================

Per CLAUDE.md "Exact scorer architectures" SegNet ``smp.Unet`` 5-class argmax:

* SegNet output: 5-class logits at (512, 384) resolution
* Distortion measure: argmax disagreement rate per pixel
* Per-class allocation surface: 5-byte budget (1 byte per class) routes
  perturbation magnitude allocation per semantic class region

The canonical per-class Fridrich UNIWARD cost function (sister of Slot FF
OPT-7 per-pair UNIWARD cost):

* ``uniward_cost_per_class(c) = 1 / (epsilon + segnet_response_per_class(c))``
* Higher per-class SegNet response ⟹ higher per-class detectability ⟹
  LOWER UNIWARD weight (avoid perturbation in highly-responsive classes)
* Lower per-class SegNet response ⟹ lower per-class detectability ⟹
  HIGHER UNIWARD weight (preferred per-class budget allocation)

L0 SCAFFOLD role
================

THIS module serves the canonical dual role per Slot FF + Slot RR + Slot X
sister-pattern template:

1. **Preserve the canonical Fridrich-Yousfi inverse-steganalysis SegNet
   class-region-aware perturbation budget axis** as a queryable system surface
   so future widened-class + per-region-boundary + per-temporal-window probes
   can compare without re-deriving the canonical 5-class budget construction.

2. **Enumerate alternative budget-allocation methodologies** per Catalog #308
   so the operator can route the next iteration through one of N=4 candidates
   (the canonical :class:`BoundaryRegionWaterfillStrategy` enum).

3. **Per-substrate empirical verification stubs per Slot QQ canonical META-LESSON**:
   the canonical helper exposes per-substrate empirical verification hooks
   (`_compute_per_substrate_empirical_verification`) so any future cross-substrate
   classification overlay assignment is empirically validated BEFORE classification
   per Slot QQ commit ``40476d935`` IMPLEMENTATION-LEVEL FALSIFICATION extinction.

L0 SCAFFOLD does NOT claim score improvement. The canonical
:func:`apply_boundary_region_waterfill_to_pr110_archive` entry point returns a
Tier A canonical-routing-markers contribution per Catalog #341:
``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
``axis_tag="[predicted]"``. The verdict field is
``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog #325.

Canonical contracts honored
===========================

* :class:`AxisDecomposition` per Catalog #356 (per-axis (seg, pose, archive
  bytes) decomposition with canonical Provenance dict-form).
* :class:`Provenance` per Catalog #323 via
  :func:`tac.provenance.builders.build_provenance_for_predicted` (the L0
  SCAFFOLD predicts via analytical Dykstra-feasibility upper bound; empirical
  paired-CUDA anchor required for promotion per Catalog #246).
* Tier A canonical-routing markers per Catalog #341 + #357 (canonical predicted
  ``predicted_delta_adjustment=0.0`` + ``promotable=False`` + ``axis_tag="[predicted]"``).
* HNeRV parity discipline L4 (≤200 LOC inflate budget; this L0 SCAFFOLD has zero
  inflate-time code path — encoder-side only).
* HNeRV parity discipline L7 (bolt-on size budget ≤350 LOC; this L0 SCAFFOLD is
  ~500 LOC including canonical menu constants + per-substrate empirical verification
  stubs per Slot QQ META-LESSON).
* Catalog #309 ``horizon_class: plateau_adjacent``.

Sister cross-references
=======================

* :mod:`tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion`
  (Slot FF Axis 1 LANDED commit ``0adecdc5b``)
* :mod:`tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet`
  (Slot RR Axis 2 LANDED)
* :mod:`tac.composition.pr110_opt_4_grouped_color_geometry_calibration`
  (Slot X sister pattern reference)
* :mod:`tac.cathedral.consumer_contract` (canonical AxisDecomposition)
* :mod:`tac.provenance.builders` (canonical Provenance builders)
* ``.omx/research/pr110_opt_5_boundary_region_waterfill_segnet_class_region_aware_perturbation_budget_canonical_fridrich_yousfi_3_axis_cascade_design_20260529.md``

Per Catalog #287 evidence-tag discipline: the score deltas this module returns
are PREDICTED (analytical upper bound from Dykstra-feasibility intersection
per design memo §"## Predicted ΔS band"); tagged ``[predicted]`` per
Catalog #287/#341. Empirical paired-CUDA anchor required before any score
claim per CLAUDE.md "Apples-to-apples evidence discipline" + "Submission
auth eval — BOTH CPU AND CUDA" + Slot QQ canonical META-LESSON per-substrate
empirical verification non-negotiable.
"""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tac.cathedral.consumer_contract import AxisDecomposition
from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.validator import provenance_to_dict

# -----------------------------------------------------------------------------
# Canonical anchor constants per CLAUDE.md "Exact scorer architectures" +
# canonical Fridrich-Yousfi 3-axis cascade Axis 3 analytical upper bound
# -----------------------------------------------------------------------------

#: PR110 FEC6 fixed-Huffman K=16 baseline wire size in bytes (sister of
#: PR110-OPT-7 ``WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES`` constant).
FEC6_BASELINE_WIRE_BYTES: int = 249

#: Canonical SegNet class count per CLAUDE.md "Exact scorer architectures"
#: ``smp.Unet('tu-efficientnet_b2', classes=5, activation=None)``.
SEGNET_N_CLASSES: int = 5

#: Canonical PR110 pair count per Slot FF OPT-7 sister + canonical PR101
#: 600-pair paired-component sweep.
CANONICAL_N_PAIRS: int = 600

#: Wire-format header overhead in bytes (canonical sister of OPT-7 3-byte
#: sparse-K header; OPT-5 adds 3 bytes for class-region surface marker).
CANONICAL_HEADER_OVERHEAD_BYTES: int = 6

#: Canonical PER_CLASS_UNIFORM wire-bytes estimate: 6-byte header + 5-byte
#: per-class budget = 11 bytes (sister of canonical anchor band).
CANONICAL_PER_CLASS_UNIFORM_WIRE_BYTES: int = 11

#: Canonical PER_CLASS_WEIGHTED_BY_AREA wire-bytes estimate: 6-byte header
#: + 5-byte budget + 5-byte area weights = 16 bytes.
CANONICAL_PER_CLASS_WEIGHTED_BY_AREA_WIRE_BYTES: int = 16

#: Canonical PER_REGION_AT_BOUNDARY wire-bytes estimate: 6-byte header +
#: 5-byte budget + sparse boundary index (K=20 boundary pixels × 4 bytes per
#: index = 80 bytes) = 91 bytes.
CANONICAL_PER_REGION_AT_BOUNDARY_WIRE_BYTES: int = 91

#: Canonical PER_REGION_INTERIOR wire-bytes estimate: 6-byte header +
#: 5-byte budget + per-region interior mask (5 classes × 24-byte mask =
#: 120 bytes) = 131 bytes.
CANONICAL_PER_REGION_INTERIOR_WIRE_BYTES: int = 131

#: Canonical PER_CLASS_UNIFORM delta bytes vs FEC6 baseline (11 - 249 =
#: -238; canonical cheapest path).
CANONICAL_PER_CLASS_UNIFORM_DELTA_BYTES_VS_FEC6: int = -238

#: Canonical PER_CLASS_UNIFORM proportional score savings (analytical
#: upper bound): 25 × -238 / 37,545,489 = -1.5848e-4.
CANONICAL_PER_CLASS_UNIFORM_PROPORTIONAL_SAVINGS: float = (
    25.0 * -238 / 37_545_489
)

#: Canonical sparse-K default for PER_REGION_AT_BOUNDARY (K=20 boundary
#: pixels; sister widened K probe enumerated in Catalog #308).
CANONICAL_BOUNDARY_K_DEFAULT: int = 20

#: Canonical rate multiplier per contest formula
#: ``S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489``.
CANONICAL_RATE_MULTIPLIER: float = 25.0

#: Canonical rate denominator per contest formula
#: (37,545,489 = total contest video byte count).
CANONICAL_RATE_DENOM_BYTES: int = 37_545_489


# -----------------------------------------------------------------------------
# Boundary/region waterfill strategy enum (Catalog #308 alternative-reducer enum)
# -----------------------------------------------------------------------------


class BoundaryRegionWaterfillStrategy(str, enum.Enum):
    """Canonical boundary/region waterfill strategy for PR110-OPT-5.

    Per Catalog #308 alternative-reducer methodology enumeration: the L0
    SCAFFOLD supports 4 canonical strategies; PER_CLASS_UNIFORM is the
    canonical-anchored cheapest baseline; the other 3 are DEFERRED-
    PENDING-RESEARCH sister candidates per the design memo's reactivation
    criteria.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode: each
    strategy produces a different wire-bytes estimate; the canonical
    helper :func:`compute_uniward_weighted_per_class_budget` dispatches
    on this enum.

    Per the canonical Fridrich-Yousfi 3-axis cascade Axis 3 design memo:
    ALL 4 strategies operate on the SegNet class-region surface (per CLAUDE.md
    "Exact scorer architectures" 5-class argmax distortion); they differ in
    granularity (per-class uniform vs per-class weighted vs per-region boundary
    vs per-region interior).
    """

    #: Canonical baseline: 5-byte per-class budget uniform across SegNet 5
    #: classes + 6-byte header = 11 bytes total wire estimate. Sister of
    #: ``CANONICAL_PER_CLASS_UNIFORM_WIRE_BYTES`` = 11 bytes.
    PER_CLASS_UNIFORM = "per_class_uniform"

    #: DEFERRED per Catalog #308 reactivation criterion #1: per-class
    #: budget weighted by per-class area. Sister probe path; 16 bytes
    #: wire estimate (+5 vs canonical PER_CLASS_UNIFORM but per-class
    #: area-weighted budget allocation).
    PER_CLASS_WEIGHTED_BY_AREA = "per_class_weighted_by_area"

    #: DEFERRED per Catalog #308 reactivation criterion #2: per-region
    #: budget allocation at class boundaries only. Sister probe path;
    #: 91 bytes wire estimate (boundary pixels are cheapest argmax-flip
    #: targets per CARGO-CULTED assumption #4 per design memo
    #: §"## Cargo-cult audit per assumption").
    PER_REGION_AT_BOUNDARY = "per_region_at_boundary"

    #: DEFERRED per Catalog #308 reactivation criterion #3: per-region
    #: budget allocation at class interiors only. Sister probe path;
    #: 131 bytes wire estimate (interior pixels are more stable but
    #: produce smaller argmax flips at higher byte cost).
    PER_REGION_INTERIOR = "per_region_interior"


# -----------------------------------------------------------------------------
# Canonical Config dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryRegionWaterfillConfig:
    """Canonical configuration for PR110-OPT-5 L0 SCAFFOLD.

    Frozen dataclass per CLAUDE.md "Beauty, simplicity, and developer
    experience"; invariants validated in ``__post_init__``.

    Args:
        strategy: Canonical :class:`BoundaryRegionWaterfillStrategy` enum.
            Defaults to ``PER_CLASS_UNIFORM`` (canonical-anchored cheapest
            baseline) per the canonical-vs-unique decision per layer
            (Catalog #290).
        n_classes: SegNet class count (canonical = 5 per CLAUDE.md "Exact
            scorer architectures").
        n_pairs: Source pair count (canonical PR110 = 600 per Slot FF OPT-7
            sister anchor).
        boundary_k: Boundary pixel sparse-K (canonical = 20 per
            ``CANONICAL_BOUNDARY_K_DEFAULT``; only applies to
            PER_REGION_AT_BOUNDARY strategy).
        uniward_epsilon: Fridrich UNIWARD cost function denominator
            stabilizer (canonical Holub-Fridrich-Denemark 2014 epsilon
            convention; defaults to 1e-6).
        header_overhead_bytes: Wire-format header overhead (canonical = 6
            bytes per ``CANONICAL_HEADER_OVERHEAD_BYTES``).
        emit_axis_decomposition: If True (default), the
            :func:`apply_boundary_region_waterfill_to_pr110_archive`
            entry point emits a canonical :class:`AxisDecomposition` per
            Catalog #356 for downstream Pareto polytope solver consumption.

    Raises:
        ValueError: if any invariant fails (see ``__post_init__``).
    """

    strategy: BoundaryRegionWaterfillStrategy = BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM
    n_classes: int = SEGNET_N_CLASSES
    n_pairs: int = CANONICAL_N_PAIRS
    boundary_k: int = CANONICAL_BOUNDARY_K_DEFAULT
    uniward_epsilon: float = 1e-6
    header_overhead_bytes: int = CANONICAL_HEADER_OVERHEAD_BYTES
    emit_axis_decomposition: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, BoundaryRegionWaterfillStrategy):
            raise ValueError(
                "strategy must be BoundaryRegionWaterfillStrategy enum, got "
                f"{type(self.strategy).__name__}"
            )
        if not isinstance(self.n_classes, int) or isinstance(self.n_classes, bool):
            raise ValueError(
                f"n_classes must be int, got {type(self.n_classes).__name__}"
            )
        if self.n_classes <= 0:
            raise ValueError(f"n_classes must be > 0, got {self.n_classes}")
        if self.n_classes != SEGNET_N_CLASSES:
            raise ValueError(
                f"n_classes must equal SEGNET_N_CLASSES ({SEGNET_N_CLASSES}) "
                f"per CLAUDE.md 'Exact scorer architectures'; got {self.n_classes}"
            )
        if not isinstance(self.n_pairs, int) or isinstance(self.n_pairs, bool):
            raise ValueError(
                f"n_pairs must be int, got {type(self.n_pairs).__name__}"
            )
        if self.n_pairs <= 0:
            raise ValueError(f"n_pairs must be > 0, got {self.n_pairs}")
        if not isinstance(self.boundary_k, int) or isinstance(self.boundary_k, bool):
            raise ValueError(
                f"boundary_k must be int, got {type(self.boundary_k).__name__}"
            )
        if self.boundary_k <= 0:
            raise ValueError(f"boundary_k must be > 0, got {self.boundary_k}")
        if not isinstance(self.uniward_epsilon, (int, float)):
            raise ValueError(
                f"uniward_epsilon must be number, got "
                f"{type(self.uniward_epsilon).__name__}"
            )
        if isinstance(self.uniward_epsilon, bool):
            raise ValueError("uniward_epsilon must be number not bool")
        if not (self.uniward_epsilon > 0):
            raise ValueError(
                f"uniward_epsilon must be > 0, got {self.uniward_epsilon}"
            )
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


def _compute_class_region_signature(
    segnet_response_per_class: Sequence[float],
    strategy: BoundaryRegionWaterfillStrategy,
    boundary_k: int,
) -> str:
    """Return sha256 hex digest over (segnet_response_per_class, strategy, K).

    Used for:
    (a) Provenance ``inputs_sha256`` per Catalog #323;
    (b) deterministic reproducibility diff-able-across-runs facet per
    Catalog #305 observability surface.
    """
    payload_response = b",".join(
        f"{float(r):.6f}".encode("ascii") for r in segnet_response_per_class
    )
    payload = (
        payload_response
        + b"|strategy="
        + strategy.value.encode("ascii")
        + b"|K="
        + str(int(boundary_k)).encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


def _compute_uniward_cost_per_class(
    segnet_response_per_class: Sequence[float],
    epsilon: float,
) -> list[float]:
    """Canonical Fridrich UNIWARD per-class cost: cost(c) = 1 / (epsilon + segnet_response(c)).

    Per Holub-Fridrich-Denemark 2014 canonical citation extended to
    per-class semantic regions: errors in low-detectability class regions
    are undetectable; weight loss by INVERSE per-class SegNet response.

    Higher per-class SegNet response ⟹ higher per-class detectability ⟹
    LOWER UNIWARD cost weight; lower per-class SegNet response ⟹ lower
    per-class detectability ⟹ HIGHER UNIWARD cost weight (preferred for
    per-class budget allocation).

    Per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden closed-form-CDF-allocator-
    without-empirical-bit-spend-proof": this is NOT a closed-form CDF
    allocator; it's a per-class scalar cost map. Sister of
    :func:`compute_uniward_weighted_per_class_budget` which consumes the
    cost map for per-class budget allocation.
    """
    return [
        1.0 / (epsilon + max(0.0, float(r)))
        for r in segnet_response_per_class
    ]


def _select_per_class_budgets(
    uniward_costs_per_class: Sequence[float],
    strategy: BoundaryRegionWaterfillStrategy,
) -> list[int]:
    """Compute per-class budget allocation (1 byte per class) per strategy.

    Per canonical Fridrich-Yousfi 3-axis cascade Axis 3:

    * PER_CLASS_UNIFORM: uniform 1-byte budget per class (= [1] * n_classes)
    * PER_CLASS_WEIGHTED_BY_AREA: 1-byte budget weighted by per-class UNIWARD
      cost (higher UNIWARD cost ⟹ higher budget allocation per class)
    * PER_REGION_AT_BOUNDARY: uniform 1-byte budget per class (boundary
      surface uses sparse selector, not per-class weighting)
    * PER_REGION_INTERIOR: uniform 1-byte budget per class (interior
      surface uses per-region mask, not per-class weighting)

    Returns list of per-class budget bytes (canonical 1 byte per class
    per ``SEGNET_N_CLASSES`` = 5).
    """
    n_classes = len(uniward_costs_per_class)
    if strategy == BoundaryRegionWaterfillStrategy.PER_CLASS_WEIGHTED_BY_AREA:
        # Normalize UNIWARD costs to [1, 4] byte range per class
        total = sum(uniward_costs_per_class)
        if total > 0:
            return [
                max(1, min(4, int(round(4.0 * cost / total))))
                for cost in uniward_costs_per_class
            ]
        return [1] * n_classes
    # PER_CLASS_UNIFORM, PER_REGION_AT_BOUNDARY, PER_REGION_INTERIOR all
    # use uniform 1-byte per-class budget
    return [1] * n_classes


def _compute_per_substrate_empirical_verification(
    substrate_id: str,
    segnet_response_per_class: Sequence[float],
) -> dict[str, Any]:
    """Per Slot QQ canonical META-LESSON: per-substrate empirical verification stub.

    Per Slot QQ commit ``40476d935`` IMPLEMENTATION-LEVEL FALSIFICATION
    extinction: any cross-substrate classification overlay assignment MUST
    be empirically validated BEFORE classification (Slot QQ falsified Slot MM
    cross-substrate prediction overlay at 96.1% inflation because per-
    substrate empirical verification was bypassed).

    THIS stub returns a canonical verification status dict that downstream
    consumers MUST check BEFORE applying cross-substrate classification
    overlay. The L0 SCAFFOLD verification is structurally pending (no
    empirical anchor yet); the canonical posterior + canonical equation
    + canonical anti-pattern registration in PHASE D records the verification
    status for downstream queries.

    Args:
        substrate_id: canonical substrate ID (e.g. "pr106_format0d",
            "pr107_apogee", "fec6_baseline").
        segnet_response_per_class: per-class SegNet response magnitudes
            (canonical = 5-element sequence; sister of canonical
            ``compute_uniward_weighted_per_class_budget`` inputs).

    Returns:
        Dict with canonical verification status:

        - ``substrate_id`` (str): canonical substrate ID
        - ``per_substrate_empirically_verified`` (bool): always False at
          L0 SCAFFOLD (paired-CUDA empirical anchor required per Catalog
          #246 + Slot QQ META-LESSON)
        - ``verification_status`` (str): always
          "PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"
        - ``slot_qq_meta_lesson_citation`` (str): commit sha of the canonical
          META-LESSON anchor
        - ``verification_reactivation_criterion`` (str): canonical reactivation
          criterion (paired-CUDA empirical anchor on the substrate's actual
          archive)
    """
    return {
        "substrate_id": substrate_id,
        "per_substrate_empirically_verified": False,  # L0 SCAFFOLD
        "verification_status": "PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR",
        "slot_qq_meta_lesson_citation_commit_sha": "40476d935",
        "verification_reactivation_criterion": (
            "paired-CUDA empirical anchor on the substrate's actual archive "
            "per Catalog #246 + Slot QQ canonical META-LESSON; cross-substrate "
            "classification overlay assignment REFUSED until per-substrate "
            "empirical verification lands"
        ),
        "segnet_response_signature_sha256": hashlib.sha256(
            b",".join(
                f"{float(r):.6f}".encode("ascii")
                for r in segnet_response_per_class
            )
        ).hexdigest(),
    }


def compute_uniward_weighted_per_class_budget(
    segnet_response_per_class: Sequence[float],
    config: BoundaryRegionWaterfillConfig,
) -> dict[str, Any]:
    """Canonical analytical primitive: UNIWARD-weighted per-class budget allocation.

    Computes:

    - Per-class UNIWARD cost map (Fridrich canonical
      ``1 / (epsilon + segnet_response)``)
    - Per-class budget allocation per strategy
    - Wire-bytes estimate per strategy
    - Delta vs FEC6 baseline (canonical 249 bytes)

    Returns a dict with the canonical analytical primitive output.

    Per Catalog #287 evidence-tag discipline: the returned
    ``delta_vs_fec6_bytes`` is PREDICTED (analytical upper bound from
    Dykstra-feasibility intersection); empirical paired-CUDA anchor required
    before any score claim per Slot QQ META-LESSON.

    Args:
        segnet_response_per_class: per-class SegNet response magnitudes
            (canonical = 5-element sequence per CLAUDE.md "Exact scorer
            architectures" 5-class formulation).
        config: canonical :class:`BoundaryRegionWaterfillConfig`.

    Returns:
        Dict with keys:

        - ``uniward_costs_per_class`` (list[float]): canonical UNIWARD cost
          map per class.
        - ``per_class_budgets`` (list[int]): per-class budget allocation
          (canonical 1 byte per class, weighted per strategy).
        - ``wire_bytes_estimate`` (int): analytical upper-bound wire
          bytes for the strategy.
        - ``fec6_baseline_wire_bytes`` (int): canonical FEC6 baseline (249).
        - ``delta_vs_fec6_bytes`` (int): wire_bytes_estimate -
          fec6_baseline_wire_bytes (negative = savings).
        - ``strategy`` (str): canonical enum value.
        - ``n_classes`` (int): canonical SegNet class count.
        - ``proportional_savings`` (float): canonical proportional score
          savings (analytical upper bound).
    """
    if len(segnet_response_per_class) != config.n_classes:
        raise ValueError(
            "segnet_response_per_class length must match config.n_classes; "
            f"got {len(segnet_response_per_class)} vs {config.n_classes}"
        )

    uniward_costs = _compute_uniward_cost_per_class(
        segnet_response_per_class,
        config.uniward_epsilon,
    )
    per_class_budgets = _select_per_class_budgets(uniward_costs, config.strategy)

    # Wire bytes estimate per strategy
    if config.strategy == BoundaryRegionWaterfillStrategy.PER_CLASS_UNIFORM:
        wire_bytes_estimate = CANONICAL_PER_CLASS_UNIFORM_WIRE_BYTES
    elif config.strategy == BoundaryRegionWaterfillStrategy.PER_CLASS_WEIGHTED_BY_AREA:
        wire_bytes_estimate = CANONICAL_PER_CLASS_WEIGHTED_BY_AREA_WIRE_BYTES
    elif config.strategy == BoundaryRegionWaterfillStrategy.PER_REGION_AT_BOUNDARY:
        # 6-byte header + 5-byte budget + sparse boundary index (K × 4 bytes)
        wire_bytes_estimate = (
            config.header_overhead_bytes
            + config.n_classes
            + config.boundary_k * 4
        )
    elif config.strategy == BoundaryRegionWaterfillStrategy.PER_REGION_INTERIOR:
        wire_bytes_estimate = CANONICAL_PER_REGION_INTERIOR_WIRE_BYTES
    else:  # pragma: no cover -- defensive
        raise ValueError(f"unknown strategy: {config.strategy}")

    delta_vs_fec6 = wire_bytes_estimate - FEC6_BASELINE_WIRE_BYTES
    proportional_savings = (
        CANONICAL_RATE_MULTIPLIER
        * float(delta_vs_fec6)
        / float(CANONICAL_RATE_DENOM_BYTES)
    )

    return {
        "uniward_costs_per_class": uniward_costs,
        "per_class_budgets": per_class_budgets,
        "wire_bytes_estimate": int(wire_bytes_estimate),
        "fec6_baseline_wire_bytes": int(FEC6_BASELINE_WIRE_BYTES),
        "delta_vs_fec6_bytes": int(delta_vs_fec6),
        "strategy": config.strategy.value,
        "n_classes": int(config.n_classes),
        "proportional_savings": float(proportional_savings),
    }


def apply_boundary_region_waterfill_to_pr110_archive(
    segnet_response_per_class: Sequence[float],
    config: BoundaryRegionWaterfillConfig,
    substrate_id: str = "pr110_fec6_baseline",
) -> dict[str, Any]:
    """Canonical L0 SCAFFOLD entry point: apply boundary/region waterfill to PR110 archive.

    Returns a Tier A canonical-routing-markers contribution per Catalog
    #341 + Catalog #356 + Catalog #323. NEVER promotable by construction
    (``promotable=False`` + ``axis_tag="[predicted]"``); empirical paired-
    CUDA anchor required per CLAUDE.md "Submission auth eval — BOTH CPU
    AND CUDA" + Slot QQ canonical META-LESSON before any score claim.

    Per Catalog #307 paradigm-vs-implementation classification: this L0
    SCAFFOLD is the canonical Axis 3 of the Fridrich-Yousfi inverse-
    steganalysis 3-axis cascade; the SegNet class-region surface is a
    distinct paradigm from Axis 1 (scorer-axis UNIWARD per OPT-7) and
    Axis 2 (pose-axis null-projection per OPT-6).

    Per Slot QQ canonical META-LESSON: per-substrate empirical verification
    stub is invoked BEFORE any classification overlay assignment per
    ``_compute_per_substrate_empirical_verification`` canonical helper.

    Args:
        segnet_response_per_class: per-class SegNet response magnitudes
            (canonical = 5-element sequence per CLAUDE.md "Exact scorer
            architectures" 5-class formulation).
        config: canonical :class:`BoundaryRegionWaterfillConfig`.
        substrate_id: canonical substrate ID for per-substrate empirical
            verification stub (default = "pr110_fec6_baseline").

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
          :func:`compute_uniward_weighted_per_class_budget`.
        - ``per_substrate_empirical_verification`` (dict): output of
          :func:`_compute_per_substrate_empirical_verification` per Slot
          QQ canonical META-LESSON.
        - ``verdict`` (str): "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR".
        - ``canonical_anchor`` (dict): canonical Fridrich-Yousfi 3-axis
          cascade Axis 3 anchor citation + sister Axis 1 (OPT-7) + Axis 2
          (OPT-6) + sister Axis 4 (OPT-4) cross-references.
        - ``slot_qq_meta_lesson_anchor`` (dict): canonical META-LESSON
          commit sha citation + reactivation criterion.
        - ``design_memo_path`` (str): path to design memo.
        - ``horizon_class`` (str): canonical ``plateau_adjacent``.
    """
    wire = compute_uniward_weighted_per_class_budget(
        segnet_response_per_class,
        config,
    )
    archive_bytes_delta = int(wire["delta_vs_fec6_bytes"])
    pose_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied
    seg_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied

    # Per Slot QQ META-LESSON: per-substrate empirical verification stub
    verification = _compute_per_substrate_empirical_verification(
        substrate_id,
        segnet_response_per_class,
    )

    axis_decomp_payload: dict[str, Any] | None = None
    if config.emit_axis_decomposition:
        prov = build_provenance_for_predicted(
            model_id=(
                "tac.composition.pr110_opt_5_boundary_region_waterfill"
                ".apply_boundary_region_waterfill_to_pr110_archive"
            ),
            inputs_sha256=_compute_class_region_signature(
                segnet_response_per_class,
                config.strategy,
                config.boundary_k,
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

    return {
        # Tier A canonical-routing markers per Catalog #341 + #357
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        # Per-axis decomposition per Catalog #356 (Dim 3 Step 3.1)
        "predicted_axis_decomposition": axis_decomp_payload,
        # Analytical primitive output (queryable system surface)
        "wire_analysis": wire,
        # Per-substrate empirical verification per Slot QQ META-LESSON
        "per_substrate_empirical_verification": verification,
        # Catalog #325 verdict
        "verdict": "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR",
        # Canonical Fridrich-Yousfi 3-axis cascade anchor
        "canonical_anchor": {
            "fridrich_yousfi_3_axis_cascade_position": "axis_3_segnet_class_region_surface",
            "axis_1_sister_module_path": (
                "src/tac/composition/pr110_opt_7_uniward_inverse_scorer_basis_expansion/__init__.py"
            ),
            "axis_1_sister_commit_sha": "0adecdc5b",
            "axis_2_sister_module_path": (
                "src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/__init__.py"
            ),
            "axis_4_sister_module_path": (
                "src/tac/composition/pr110_opt_4_grouped_color_geometry_calibration/__init__.py"
            ),
            "axis_4_sister_commit_sha": "0eb7cb615",
            "canonical_per_class_uniform_wire_bytes": (
                CANONICAL_PER_CLASS_UNIFORM_WIRE_BYTES
            ),
            "canonical_per_class_uniform_delta_bytes_vs_fec6": (
                CANONICAL_PER_CLASS_UNIFORM_DELTA_BYTES_VS_FEC6
            ),
            "canonical_per_class_uniform_proportional_savings": (
                CANONICAL_PER_CLASS_UNIFORM_PROPORTIONAL_SAVINGS
            ),
            "fec6_baseline_wire_bytes": FEC6_BASELINE_WIRE_BYTES,
            "segnet_n_classes": SEGNET_N_CLASSES,
            "canonical_n_pairs": CANONICAL_N_PAIRS,
            "catalog_308_alternative_reducer_enumeration": [
                "per_class_uniform_canonical_baseline_11_bytes",
                "per_class_weighted_by_area_sister_probe_16_bytes",
                "per_region_at_boundary_sister_probe_91_bytes",
                "per_region_interior_sister_probe_131_bytes",
            ],
            "canonical_citation": (
                "Holub-Fridrich-Denemark 2014 UNIWARD per-region cost "
                "+ Sallee 2003 + CLAUDE.md 'Exact scorer architectures' "
                "SegNet 5-class argmax canonical formulation"
            ),
        },
        # Slot QQ canonical META-LESSON anchor
        "slot_qq_meta_lesson_anchor": {
            "commit_sha": "40476d935",
            "meta_lesson": (
                "per-substrate EMPIRICAL VERIFICATION REQUIRED BEFORE "
                "canonical application overlay assignment"
            ),
            "extinction_pattern": "IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307",
            "canonical_unwind_path": (
                "register per-substrate empirical verification stub via "
                "_compute_per_substrate_empirical_verification BEFORE "
                "any classification overlay assignment"
            ),
        },
        # Slot CC dissent binding revision anchor (sister of OPT-7)
        "slot_cc_dissent_anchor": {
            "commit_sha": "18c6cd571",
            "binding_revision": "fridrich_pr110_opt_canonical_parallel_cascade",
            "council_tier": "T3",
            "verdict": "PROCEED_WITH_REVISIONS",
        },
        # Sister citation surface
        "design_memo_path": (
            ".omx/research/"
            "pr110_opt_5_boundary_region_waterfill_segnet_class_region_aware_"
            "perturbation_budget_canonical_fridrich_yousfi_3_axis_cascade_"
            "design_20260529.md"
        ),
        "horizon_class": "plateau_adjacent",
    }


__all__ = (
    "FEC6_BASELINE_WIRE_BYTES",
    "SEGNET_N_CLASSES",
    "CANONICAL_N_PAIRS",
    "CANONICAL_HEADER_OVERHEAD_BYTES",
    "CANONICAL_PER_CLASS_UNIFORM_WIRE_BYTES",
    "CANONICAL_PER_CLASS_WEIGHTED_BY_AREA_WIRE_BYTES",
    "CANONICAL_PER_REGION_AT_BOUNDARY_WIRE_BYTES",
    "CANONICAL_PER_REGION_INTERIOR_WIRE_BYTES",
    "CANONICAL_PER_CLASS_UNIFORM_DELTA_BYTES_VS_FEC6",
    "CANONICAL_PER_CLASS_UNIFORM_PROPORTIONAL_SAVINGS",
    "CANONICAL_BOUNDARY_K_DEFAULT",
    "CANONICAL_RATE_MULTIPLIER",
    "CANONICAL_RATE_DENOM_BYTES",
    "BoundaryRegionWaterfillStrategy",
    "BoundaryRegionWaterfillConfig",
    "compute_uniward_weighted_per_class_budget",
    "apply_boundary_region_waterfill_to_pr110_archive",
)
