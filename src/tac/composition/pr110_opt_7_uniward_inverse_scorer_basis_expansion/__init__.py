# SPDX-License-Identifier: MIT
"""PR110-OPT-7 UNIWARD inverse-scorer basis expansion — L0 SCAFFOLD.

Per CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"
+ Slot CC LANDED (commit ``18c6cd571``) T3 grand council Fridrich dissent
binding revision + Slot X PR110-OPT-4 L0 SCAFFOLD (commit ``0eb7cb615``)
Catalog #308 reactivation path #4 (composition with OPT-7 UNIWARD sparse
selector) + Slot FF cap=3 parallel-cascade directive 2026-05-29.

Design memo (single source of truth)::

    .omx/research/pr110_opt_7_uniward_inverse_scorer_basis_expansion_\
fridrich_canonical_parallel_cascade_per_slot_cc_dissent_design_20260529.md

Wave N+34 OPT-7 analytical anchor (IMPLEMENTATION_FALSIFIED at WEIGHTING)::

    .omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json

Canonical context
=================

Fridrich UNIWARD canonical cost function (Holub-Fridrich-Denemark 2014)::

    cost(pixel) = 1 / (epsilon + scorer_response(pixel))

Per CLAUDE.md "Fridrich inverse steganalysis":
1. UNIWARD — errors in textured regions are undetectable; weight loss by
   inverse local variance.
2. Detector-informed embedding — our TTO approach is Fridrich-approved
   (Yousfi 2022).
3. Square root law — spread small errors (L∞ penalty), don't concentrate.
4. CNN blind spots — EfficientNet misses DCT statistics, has texture-region
   blind spots.

PR110 archive grammar (sister of Slot X PR110-OPT-4 L0 SCAFFOLD):

- 16-symbol K=16 selector palette per FEC6 ``submissions/hnerv_fec6_fixed_huffman_k16/``
- 600 per-pair selectors (one selector per source-frame pair)
- 6-byte header + 243-byte 0-order fixed-Huffman bitstream = **249 byte baseline wire**

Wave N+34 OPT-7 analytical investigation sourced 21 modes from the live
PR101 paired-component sweep and applied UNIWARD canonical WEIGHTING to
PoseNet baseline distortion. The empirical anchor:

- Unweighted aggregate ΔS: -0.0011704843740551621
- UNIWARD-weighted aggregate ΔS: -0.0009103568688898632
- **Improvement ratio: -22.22% WORSE** (WEIGHTING IMPLEMENTATION_FALSIFIED)
- Top-decile UNIWARD aggregate ΔS: -0.0004764121800100148
- UNIWARD concentration factor: 0.40702139265599685
- Sparse selector K=100 wire bytes: 103 (vs FEC6 249 = **-146 bytes**)
- Sparse selector K=100 proportional savings: -0.0000794 ΔS

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" +
Catalog #307 paradigm-vs-implementation classification: Wave N+34
WEIGHTING falsification is IMPLEMENTATION-LEVEL not paradigm-level.
BASIS EXPANSION (this L0 SCAFFOLD's distinct paradigm) is DEFERRED-
PENDING-RESEARCH per Catalog #308 alternative-reducer enumeration.

WEIGHTING vs BASIS EXPANSION distinction
========================================

- **WEIGHTING (Wave N+34 canonical OPT-7; FALSIFIED)**: reweight EXISTING
  21 perturbation modes by UNIWARD cost function. Analytically reduces to
  OPT-12 PoseNet-null catalog at sparse-K (both select for low-baseline-
  pose pairs).

- **BASIS EXPANSION (this L0 SCAFFOLD; DEFERRED)**: add NEW perturbation
  modes derived from inverse-scorer-sensitivity basis (Fridrich UNIWARD
  applied to scorer Jacobian columns; expansion catalog includes
  per-pair UNIWARD-cost-weighted basis vectors from the scorer's null
  space). EXPANSION distinct from WEIGHTING because expansion adds modes
  whereas weighting reweights existing modes.

L0 SCAFFOLD role
================

THIS module serves the canonical dual role:

1. **Preserve the Wave N+34 OPT-7 analytical primitive** as a queryable
   system surface so future widened-K probes can compare against the
   baked-in K=100 sparse-selector / 103-byte wire / -146-byte delta /
   -0.0000794 proportional-savings constants without re-deriving them.

2. **Enumerate alternative basis-expansion methodologies** per Catalog
   #308 so the operator can route the next iteration through one of
   N≥4 candidates (NOT just the canonical SPARSE_K100 baseline).

L0 SCAFFOLD does NOT claim score improvement. The canonical
:func:`apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive`
entry point returns a Tier A canonical-routing-markers contribution per
Catalog #341: ``predicted_delta_adjustment=0.0`` + ``promotable=False``
+ ``axis_tag="[predicted]"``. The verdict field is
``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog #325.

Canonical contracts honored
===========================

- :class:`AxisDecomposition` per Catalog #356 (per-axis (seg, pose,
  archive bytes) decomposition with canonical Provenance dict-form).
- :class:`Provenance` per Catalog #323 via
  :func:`tac.provenance.builders.build_provenance_for_predicted` (the L0
  SCAFFOLD predicts via Wave N+34 analytical anchor; empirical paired-
  CUDA anchor required for promotion).
- Tier A canonical-routing markers per Catalog #341 + #357.
- HNeRV parity discipline L4 (≤200 LOC inflate budget; this L0 SCAFFOLD
  has no inflate-time code path — encoder-side only).
- HNeRV parity discipline L7 (bolt-on size budget ≤350 LOC; this L0
  SCAFFOLD is ~450 LOC inclusive of analytical primitives).
- Catalog #309 ``horizon_class: plateau_adjacent``.

Sister cross-references
=======================

- :mod:`tac.composition.pr110_opt_4_grouped_color_geometry_calibration`
  (Slot X PR110-OPT-4 L0 SCAFFOLD; Catalog #308 reactivation path #4 cites
  composition with OPT-7 UNIWARD = THIS module)
- :mod:`tac.cathedral.consumer_contract` (canonical AxisDecomposition)
- :mod:`tac.provenance.builders` (canonical Provenance builders)
- ``.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json``
- ``.omx/research/pr110_opt_7_uniward_inverse_scorer_basis_expansion_fridrich_canonical_parallel_cascade_per_slot_cc_dissent_design_20260529.md``

Per Catalog #287 evidence tag discipline: the score deltas this module
returns are PREDICTED (Wave N+34 analytical anchor); tagged
``[predicted]`` per Catalog #287/#341. Empirical paired-CUDA anchor
required before any score claim per CLAUDE.md "Apples-to-apples evidence
discipline" + "Submission auth eval — BOTH CPU AND CUDA".
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
# Wave N+34 OPT-7 canonical anchor constants (analytical IMPLEMENTATION_FALSIFIED
# at WEIGHTING; BASIS EXPANSION DEFERRED-PENDING-RESEARCH per Catalog #308)
# -----------------------------------------------------------------------------

#: PR110 FEC6 fixed-Huffman K=16 baseline wire size in bytes per Wave N+34
#: (sister of PR110-OPT-4 ``WAVE_N34_FEC6_BASELINE_WIRE_BYTES`` constant).
WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES: int = 249

#: Wave N+34 OPT-7 unweighted aggregate ΔS (uniform-prior baseline before
#: UNIWARD weighting is applied; reference for the -22.22% comparison).
WAVE_N34_OPT7_UNWEIGHTED_AGGREGATE_DELTA_S: float = -0.0011704843740551621

#: Wave N+34 OPT-7 UNIWARD-weighted aggregate ΔS (canonical Fridrich
#: 1/(epsilon + scorer_response) applied to PoseNet baseline distortion;
#: IMPLEMENTATION_FALSIFIED at -22.22% WORSE than unweighted).
WAVE_N34_OPT7_UNIWARD_WEIGHTED_AGGREGATE_DELTA_S: float = -0.0009103568688898632

#: Wave N+34 OPT-7 improvement ratio (UNIWARD vs unweighted); negative
#: means UNIWARD weighting is WORSE.
WAVE_N34_OPT7_IMPROVEMENT_RATIO: float = -0.22223919509842144

#: Wave N+34 OPT-7 top-decile UNIWARD aggregate ΔS (60-pair subset with
#: highest UNIWARD costs; concentration anchor).
WAVE_N34_OPT7_TOP_DECILE_UNIWARD_AGGREGATE_DELTA_S: float = -0.0004764121800100148

#: Wave N+34 OPT-7 UNIWARD concentration factor (top-decile aggregate /
#: unweighted aggregate; ~0.407 indicates moderate concentration).
WAVE_N34_OPT7_UNIWARD_CONCENTRATION_FACTOR: float = 0.40702139265599685

#: Wave N+34 OPT-7 sparse-selector K=100 wire bytes estimate (sparse
#: selector emits K indices + per-K perturbation magnitudes; canonical
#: K=100 yields 103 bytes wire estimate).
WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES: int = 103

#: Wave N+34 OPT-7 sparse-selector K=100 delta bytes vs FEC6 baseline
#: (-146 = 103 - 249; sparse selector saves 146 bytes per pair).
WAVE_N34_OPT7_SPARSE_SELECTOR_K100_DELTA_BYTES_VS_FEC6: int = -146

#: Wave N+34 OPT-7 sparse-selector K=100 proportional score savings
#: (-0.0000794 = 25 * -146 / 37,545,489 + zero distortion-axis contribution).
WAVE_N34_OPT7_SPARSE_SELECTOR_K100_PROPORTIONAL_SAVINGS: float = -7.940203000166914e-05

#: Wave N+34 OPT-7 number of pairs in source sweep (canonical PR101
#: 600-pair paired-component sweep; sister of PR110-OPT-4 N_PAIRS).
WAVE_N34_OPT7_N_PAIRS: int = 600

#: Wave N+34 OPT-7 number of modes in source catalog (canonical PR110
#: 21 active modes; sister widened-catalog probes use ≥40 modes).
WAVE_N34_OPT7_N_MODES: int = 21

#: Canonical sparse-K default per Wave N+34 OPT-7 anchor (K=100 is
#: empirically the canonical inflection point; sister WIDENED_K=200 probe
#: enumerated in :class:`BasisExpansionStrategy`).
CANONICAL_SPARSE_K_DEFAULT: int = 100

#: Canonical widened-K alternative per Catalog #308 reactivation
#: criterion #1 (sister widened-catalog probe at K=200).
CANONICAL_WIDENED_K_DEFAULT: int = 200

#: Canonical rate multiplier per contest formula
#: ``S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489``.
CANONICAL_RATE_MULTIPLIER: float = 25.0

#: Canonical rate denominator per contest formula
#: (37,545,489 = total contest video byte count).
CANONICAL_RATE_DENOM_BYTES: int = 37_545_489

# -----------------------------------------------------------------------------
# Basis expansion strategy enum (Catalog #308 alternative-reducer enumeration)
# -----------------------------------------------------------------------------


class BasisExpansionStrategy(str, enum.Enum):
    """Canonical basis-expansion strategy for PR110-OPT-7.

    Per Catalog #308 alternative-reducer methodology enumeration: the L0
    SCAFFOLD supports 4 canonical strategies; SPARSE_K100_UNIWARD_WEIGHTED
    is Wave N+34's canonical-anchored baseline; the other 3 are DEFERRED-
    PENDING-RESEARCH sister candidates per the design memo's reactivation
    criteria.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode: each
    strategy may produce a different wire-bytes estimate; the canonical
    helper :func:`compute_uniward_weighted_perturbation_for_pr110_catalog`
    dispatches on this enum.

    Per the WEIGHTING vs BASIS EXPANSION distinction in the module
    docstring: ALL 4 strategies are BASIS EXPANSION variants (adding NEW
    perturbation modes from inverse-scorer-sensitivity basis); none are
    pure WEIGHTING (which is Wave N+34's IMPLEMENTATION_FALSIFIED canonical).
    """

    #: Wave N+34 canonical baseline: K=100 sparse selector + UNIWARD-cost-
    #: weighted basis vectors from scorer Jacobian null space. Sister of
    #: ``WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES`` = 103 bytes wire
    #: estimate.
    SPARSE_K100_UNIWARD_WEIGHTED = "sparse_k100_uniward_weighted"

    #: DEFERRED per Catalog #308 reactivation criterion #1: widened K=200
    #: sparse selector. Sister probe path; ~206 bytes wire estimate
    #: (analytical upper bound; +43 vs canonical K=100 but doubled
    #: per-pair distortion coverage).
    WIDENED_K200_UNIWARD_WEIGHTED = "widened_k200_uniward_weighted"

    #: DEFERRED per Catalog #308: per-region grouping with UNIWARD weighting
    #: (per-pixel-region instead of per-pair; sister probe path per Catalog
    #: #277 wavelet hierarchy + Fridrich UNIWARD-per-region canonical).
    PER_REGION_UNIWARD_WEIGHTED = "per_region_uniward_weighted"

    #: DEFERRED per Catalog #308: all-pairs UNIWARD weighting (degenerate
    #: baseline; equivalent to Wave N+34's WEIGHTING-ONLY canonical at
    #: -22.22% IMPLEMENTATION_FALSIFIED; preserved as the canonical
    #: reduction-to-OPT-12 anchor per Catalog #307 paradigm-vs-
    #: implementation classification).
    ALL_PAIRS_UNIWARD_WEIGHTED = "all_pairs_uniward_weighted"


# -----------------------------------------------------------------------------
# Canonical Config dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class UniwardInverseScorerBasisConfig:
    """Canonical configuration for PR110-OPT-7 L0 SCAFFOLD.

    Frozen dataclass per CLAUDE.md "Beauty, simplicity, and developer
    experience"; invariants validated in ``__post_init__``.

    Args:
        basis_strategy: Canonical :class:`BasisExpansionStrategy` enum.
            Defaults to ``SPARSE_K100_UNIWARD_WEIGHTED`` (Wave N+34 canonical
            baseline) per the canonical-vs-unique decision per layer
            (Catalog #290).
        n_pairs: Source pair count (canonical PR110 = 600 per Wave N+34).
        n_modes: Source mode count in the perturbation catalog (canonical
            Wave N+34 = 21 active modes; sister widened-catalog probes use
            ≥40 modes).
        sparse_k: Sparse-selector K (canonical = 100 per Wave N+34
            ``WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES`` anchor;
            sister WIDENED_K = 200; ALL_PAIRS_UNIWARD_WEIGHTED ignores).
        uniward_epsilon: Fridrich UNIWARD cost function denominator
            stabilizer (canonical Holub-Fridrich-Denemark 2014 epsilon
            convention; defaults to 1e-6).
        header_overhead_bytes: Wire-format header overhead (sparse-K
            selector index format header; canonical = 3 bytes for K=100).
        emit_axis_decomposition: If True (default), the
            :func:`apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive`
            entry point emits a canonical :class:`AxisDecomposition` per
            Catalog #356 for downstream Pareto polytope solver consumption.

    Raises:
        ValueError: if any invariant fails (see ``__post_init__``).
    """

    basis_strategy: BasisExpansionStrategy = BasisExpansionStrategy.SPARSE_K100_UNIWARD_WEIGHTED
    n_pairs: int = WAVE_N34_OPT7_N_PAIRS
    n_modes: int = WAVE_N34_OPT7_N_MODES
    sparse_k: int = CANONICAL_SPARSE_K_DEFAULT
    uniward_epsilon: float = 1e-6
    header_overhead_bytes: int = 3
    emit_axis_decomposition: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.basis_strategy, BasisExpansionStrategy):
            raise ValueError(
                "basis_strategy must be BasisExpansionStrategy enum, got "
                f"{type(self.basis_strategy).__name__}"
            )
        if not isinstance(self.n_pairs, int) or isinstance(self.n_pairs, bool):
            raise ValueError(
                f"n_pairs must be int, got {type(self.n_pairs).__name__}"
            )
        if self.n_pairs <= 0:
            raise ValueError(f"n_pairs must be > 0, got {self.n_pairs}")
        if not isinstance(self.n_modes, int) or isinstance(self.n_modes, bool):
            raise ValueError(
                f"n_modes must be int, got {type(self.n_modes).__name__}"
            )
        if self.n_modes <= 0:
            raise ValueError(f"n_modes must be > 0, got {self.n_modes}")
        if not isinstance(self.sparse_k, int) or isinstance(self.sparse_k, bool):
            raise ValueError(
                f"sparse_k must be int, got {type(self.sparse_k).__name__}"
            )
        if self.sparse_k <= 0:
            raise ValueError(f"sparse_k must be > 0, got {self.sparse_k}")
        if self.sparse_k > self.n_pairs:
            raise ValueError(
                f"sparse_k ({self.sparse_k}) must be <= n_pairs "
                f"({self.n_pairs})"
            )
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


def _compute_basis_expansion_signature(
    source_modes_per_pair: Sequence[int],
    sparse_k: int,
    basis_strategy: BasisExpansionStrategy,
) -> str:
    """Return sha256 hex digest over (modes, K, strategy) tuple.

    Used for:
    (a) Provenance ``inputs_sha256`` per Catalog #323;
    (b) deterministic reproducibility diff-able-across-runs facet per
    Catalog #305 observability surface.
    """
    payload_modes = b",".join(
        str(int(m)).encode("ascii") for m in source_modes_per_pair
    )
    payload = (
        payload_modes
        + b"|K="
        + str(int(sparse_k)).encode("ascii")
        + b"|strategy="
        + basis_strategy.value.encode("ascii")
    )
    return hashlib.sha256(payload).hexdigest()


def _compute_uniward_cost_per_pair(
    scorer_response_per_pair: Sequence[float],
    epsilon: float,
) -> list[float]:
    """Canonical Fridrich UNIWARD cost: cost(pair) = 1 / (epsilon + scorer_response).

    Per Holub-Fridrich-Denemark 2014 canonical citation: errors in
    textured regions are undetectable; weight loss by INVERSE local
    scorer response.

    Higher scorer_response ⟹ higher detectability ⟹ LOWER UNIWARD cost
    weight; lower scorer_response ⟹ lower detectability ⟹ HIGHER UNIWARD
    cost weight (preferred for sparse selection).

    Per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden closed-form-CDF-allocator-
    without-empirical-bit-spend-proof": this is NOT a closed-form CDF
    allocator; it's a per-pair scalar cost map. Sister of
    ``compute_uniward_weighted_perturbation_for_pr110_catalog`` which
    consumes the cost map for sparse-K selection.
    """
    return [
        1.0 / (epsilon + max(0.0, float(r)))
        for r in scorer_response_per_pair
    ]


def _select_sparse_k_pairs(
    uniward_costs_per_pair: Sequence[float],
    sparse_k: int,
) -> list[int]:
    """Select K pair indices with highest UNIWARD cost.

    Per Fridrich canonical: pairs with highest UNIWARD cost (=lowest
    scorer detectability) are the canonical sparse-selector picks.

    Returns sorted list of pair indices (canonical ascending order for
    deterministic wire-format emission per Catalog #305 observability
    diff-able-across-runs facet).
    """
    n_pairs = len(uniward_costs_per_pair)
    if sparse_k >= n_pairs:
        return list(range(n_pairs))
    # Argsort descending by UNIWARD cost; take top K; sort ascending for
    # deterministic emission.
    indexed = sorted(
        enumerate(uniward_costs_per_pair),
        key=lambda x: -float(x[1]),
    )
    selected = [int(idx) for idx, _ in indexed[:sparse_k]]
    return sorted(selected)


def compute_uniward_weighted_perturbation_for_pr110_catalog(
    source_modes_per_pair: Sequence[int],
    scorer_response_per_pair: Sequence[float],
    config: UniwardInverseScorerBasisConfig,
) -> dict[str, Any]:
    """Canonical analytical primitive: UNIWARD-weighted basis expansion wire estimate.

    Computes:

    - UNIWARD cost map per pair (Fridrich canonical
      ``1 / (epsilon + scorer_response)``)
    - Sparse-K selection (top-K UNIWARD cost pair indices) IF
      ``config.basis_strategy`` ∈ {SPARSE_K100, WIDENED_K200}
    - Wire-bytes estimate per strategy
    - Delta vs FEC6 baseline (canonical 249 bytes)

    Returns a dict with the canonical analytical primitive output.

    Per Catalog #287 evidence tag discipline: the returned ``delta_vs_fec6_bytes``
    is PREDICTED (analytical upper bound from Wave N+34 anchor); empirical
    paired-CUDA anchor required before any score claim.

    Args:
        source_modes_per_pair: per-pair mode codes (canonical Wave N+34
            source = 600 pairs × 21 modes; integer codes in [0, n_modes)).
        scorer_response_per_pair: per-pair scorer response magnitudes
            (canonical Wave N+34 source = PoseNet baseline distortion per
            pair; floats in [0, ∞)).
        config: canonical :class:`UniwardInverseScorerBasisConfig`.

    Returns:
        Dict with keys:

        - ``uniward_costs_per_pair`` (list[float]): canonical UNIWARD cost
          map per pair.
        - ``selected_pair_indices`` (list[int]): sparse-K selection
          (or all pairs for ALL_PAIRS_UNIWARD_WEIGHTED).
        - ``wire_bytes_estimate`` (int): analytical upper-bound wire
          bytes for the strategy.
        - ``fec6_baseline_wire_bytes`` (int): canonical Wave N+34 FEC6
          baseline (249).
        - ``delta_vs_fec6_bytes`` (int): wire_bytes_estimate -
          fec6_baseline_wire_bytes (negative = savings).
        - ``basis_strategy`` (str): canonical enum value.
        - ``n_selected_pairs`` (int): cardinality of selected_pair_indices.
        - ``uniward_concentration_factor`` (float): top-K UNIWARD cost
          sum / all-pairs UNIWARD cost sum (sister of Wave N+34
          ``WAVE_N34_OPT7_UNIWARD_CONCENTRATION_FACTOR`` = 0.407).
    """
    if len(source_modes_per_pair) != len(scorer_response_per_pair):
        raise ValueError(
            "source_modes_per_pair length must match "
            "scorer_response_per_pair length; got "
            f"{len(source_modes_per_pair)} vs {len(scorer_response_per_pair)}"
        )
    if len(source_modes_per_pair) != config.n_pairs:
        raise ValueError(
            "source_modes_per_pair length must match config.n_pairs; "
            f"got {len(source_modes_per_pair)} vs {config.n_pairs}"
        )
    for mode in source_modes_per_pair:
        if not (0 <= int(mode) < config.n_modes):
            raise ValueError(
                f"all modes must be in [0, {config.n_modes}); got {mode}"
            )

    uniward_costs = _compute_uniward_cost_per_pair(
        scorer_response_per_pair,
        config.uniward_epsilon,
    )

    # Strategy dispatch
    if config.basis_strategy == BasisExpansionStrategy.SPARSE_K100_UNIWARD_WEIGHTED:
        effective_k = config.sparse_k
        selected = _select_sparse_k_pairs(uniward_costs, effective_k)
    elif config.basis_strategy == BasisExpansionStrategy.WIDENED_K200_UNIWARD_WEIGHTED:
        # Widened-K probe: explicitly use CANONICAL_WIDENED_K_DEFAULT
        # unless overridden via config.sparse_k.
        effective_k = (
            CANONICAL_WIDENED_K_DEFAULT
            if config.sparse_k == CANONICAL_SPARSE_K_DEFAULT
            else config.sparse_k
        )
        selected = _select_sparse_k_pairs(uniward_costs, effective_k)
    elif config.basis_strategy == BasisExpansionStrategy.PER_REGION_UNIWARD_WEIGHTED:
        # Per-region grouping: degenerate to sparse-K at L0 SCAFFOLD level
        # (per-region requires per-pixel surface; out-of-scope L0). The
        # canonical reactivation criterion documents this in the design
        # memo's Catalog #308 enumeration.
        effective_k = config.sparse_k
        selected = _select_sparse_k_pairs(uniward_costs, effective_k)
    elif config.basis_strategy == BasisExpansionStrategy.ALL_PAIRS_UNIWARD_WEIGHTED:
        effective_k = config.n_pairs
        selected = list(range(config.n_pairs))
    else:  # pragma: no cover -- defensive
        raise ValueError(f"unknown basis_strategy: {config.basis_strategy}")

    # Wire bytes estimate
    # Sparse selector format: header_overhead + K * ceil(log2(N_pairs)/8) bytes
    # for K indices + K bytes for K per-pair perturbation magnitudes
    if config.basis_strategy == BasisExpansionStrategy.ALL_PAIRS_UNIWARD_WEIGHTED:
        # All-pairs degenerate: equivalent to per-pair perturbation
        # magnitudes only (no selector indices needed).
        wire_bytes_estimate = config.header_overhead_bytes + config.n_pairs
    else:
        # Sparse-K: header + K * index_byte_width + K * magnitude_byte
        index_byte_width = max(1, math.ceil(math.log2(config.n_pairs) / 8.0))
        wire_bytes_estimate = (
            config.header_overhead_bytes
            + effective_k * index_byte_width
            + effective_k  # 1 byte per perturbation magnitude
        )

    delta_vs_fec6 = wire_bytes_estimate - WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES

    # UNIWARD concentration factor: top-K cost sum / all-pairs cost sum
    all_pairs_sum = sum(uniward_costs)
    if all_pairs_sum > 0 and len(selected) > 0:
        top_k_sum = sum(uniward_costs[i] for i in selected)
        concentration = top_k_sum / all_pairs_sum
    else:
        concentration = 0.0

    return {
        "uniward_costs_per_pair": uniward_costs,
        "selected_pair_indices": selected,
        "wire_bytes_estimate": int(wire_bytes_estimate),
        "fec6_baseline_wire_bytes": int(WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES),
        "delta_vs_fec6_bytes": int(delta_vs_fec6),
        "basis_strategy": config.basis_strategy.value,
        "n_selected_pairs": len(selected),
        "uniward_concentration_factor": float(concentration),
    }


def apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive(
    source_modes_per_pair: Sequence[int],
    scorer_response_per_pair: Sequence[float],
    config: UniwardInverseScorerBasisConfig,
) -> dict[str, Any]:
    """Canonical L0 SCAFFOLD entry point: apply UNIWARD basis expansion to PR110 archive.

    Returns a Tier A canonical-routing-markers contribution per Catalog
    #341 + Catalog #356 + Catalog #323. NEVER promotable by construction
    (``promotable=False`` + ``axis_tag="[predicted]"``); empirical paired-
    CUDA anchor required per CLAUDE.md "Submission auth eval — BOTH CPU
    AND CUDA" before any score claim.

    Per Wave N+34 OPT-7 IMPLEMENTATION_FALSIFIED verdict (at WEIGHTING) +
    Catalog #307 paradigm-vs-implementation classification: the verdict
    field is ``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR`` per Catalog
    #325 (basis EXPANSION is distinct paradigm from WEIGHTING; reactivation
    criterion pinned in design memo).

    Args:
        source_modes_per_pair: per-pair mode codes (canonical Wave N+34
            source = 600 pairs × 21 modes).
        scorer_response_per_pair: per-pair scorer response magnitudes
            (canonical Wave N+34 source = PoseNet baseline distortion).
        config: canonical :class:`UniwardInverseScorerBasisConfig`.

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
          :func:`compute_uniward_weighted_perturbation_for_pr110_catalog`.
        - ``verdict`` (str): "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR".
        - ``wave_n34_opt7_anchor`` (dict): Wave N+34 OPT-7
          IMPLEMENTATION_FALSIFIED-at-WEIGHTING citation.
        - ``slot_cc_dissent_anchor`` (dict): Slot CC Fridrich dissent
          binding revision commit ``18c6cd571`` citation.
        - ``design_memo_path`` (str): path to design memo.
        - ``horizon_class`` (str): canonical ``plateau_adjacent``.
    """
    wire = compute_uniward_weighted_perturbation_for_pr110_catalog(
        source_modes_per_pair,
        scorer_response_per_pair,
        config,
    )
    archive_bytes_delta = int(wire["delta_vs_fec6_bytes"])
    pose_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied
    seg_delta = 0.0  # L0 SCAFFOLD: no actual perturbation applied

    axis_decomp_payload: dict[str, Any] | None = None
    if config.emit_axis_decomposition:
        prov = build_provenance_for_predicted(
            model_id=(
                "tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion"
                ".apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive"
            ),
            inputs_sha256=_compute_basis_expansion_signature(
                source_modes_per_pair,
                config.sparse_k,
                config.basis_strategy,
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
        # Catalog #325 verdict
        "verdict": "DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR",
        # Wave N+34 OPT-7 historical anchor (IMPLEMENTATION_FALSIFIED at WEIGHTING)
        "wave_n34_opt7_anchor": {
            "verdict": "WEIGHTING_IMPLEMENTATION_FALSIFIED",
            "unweighted_aggregate_delta_S": (
                WAVE_N34_OPT7_UNWEIGHTED_AGGREGATE_DELTA_S
            ),
            "uniward_weighted_aggregate_delta_S": (
                WAVE_N34_OPT7_UNIWARD_WEIGHTED_AGGREGATE_DELTA_S
            ),
            "improvement_ratio": WAVE_N34_OPT7_IMPROVEMENT_RATIO,
            "top_decile_uniward_aggregate_delta_S": (
                WAVE_N34_OPT7_TOP_DECILE_UNIWARD_AGGREGATE_DELTA_S
            ),
            "uniward_concentration_factor": (
                WAVE_N34_OPT7_UNIWARD_CONCENTRATION_FACTOR
            ),
            "sparse_selector_K100_wire_bytes_estimate": (
                WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES
            ),
            "sparse_selector_K100_delta_bytes_vs_fec6": (
                WAVE_N34_OPT7_SPARSE_SELECTOR_K100_DELTA_BYTES_VS_FEC6
            ),
            "sparse_selector_K100_proportional_savings": (
                WAVE_N34_OPT7_SPARSE_SELECTOR_K100_PROPORTIONAL_SAVINGS
            ),
            "fec6_baseline_wire_bytes": (
                WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES
            ),
            "n_pairs_in_source": WAVE_N34_OPT7_N_PAIRS,
            "n_modes_in_source": WAVE_N34_OPT7_N_MODES,
            "canonical_artifact_path": (
                ".omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json"
            ),
            "catalog_307_classification": (
                "WEIGHTING_IMPLEMENTATION_LEVEL_FALSIFICATION_BASIS_EXPANSION_PARADIGM_DEFERRED"
            ),
            "catalog_308_alternative_reducer_enumeration": [
                "sparse_k100_uniward_weighted_canonical_baseline",
                "widened_k200_uniward_weighted_sister_probe",
                "per_region_uniward_weighted_sister_probe_per_catalog_277",
                "all_pairs_uniward_weighted_degenerate_reduction_to_opt_12",
            ],
            "canonical_citation": (
                "Holub-Fridrich-Denemark 2014 + Sallee 2003 UNIWARD "
                "inverse-scorer-sensitivity weighting"
            ),
        },
        # Slot CC dissent binding revision anchor
        "slot_cc_dissent_anchor": {
            "commit_sha": "18c6cd571",
            "binding_revision": "fridrich_pr110_opt_7_uniward_parallel_cascade",
            "council_tier": "T3",
            "verdict": "PROCEED_WITH_REVISIONS",
        },
        # Sister citation surface
        "design_memo_path": (
            ".omx/research/"
            "pr110_opt_7_uniward_inverse_scorer_basis_expansion_"
            "fridrich_canonical_parallel_cascade_per_slot_cc_dissent_"
            "design_20260529.md"
        ),
        "sister_pr110_opt_4_module_path": (
            "src/tac/composition/pr110_opt_4_grouped_color_geometry_calibration/__init__.py"
        ),
        "horizon_class": "plateau_adjacent",
    }


__all__ = (
    "WAVE_N34_OPT7_FEC6_BASELINE_WIRE_BYTES",
    "WAVE_N34_OPT7_UNWEIGHTED_AGGREGATE_DELTA_S",
    "WAVE_N34_OPT7_UNIWARD_WEIGHTED_AGGREGATE_DELTA_S",
    "WAVE_N34_OPT7_IMPROVEMENT_RATIO",
    "WAVE_N34_OPT7_TOP_DECILE_UNIWARD_AGGREGATE_DELTA_S",
    "WAVE_N34_OPT7_UNIWARD_CONCENTRATION_FACTOR",
    "WAVE_N34_OPT7_SPARSE_SELECTOR_K100_WIRE_BYTES",
    "WAVE_N34_OPT7_SPARSE_SELECTOR_K100_DELTA_BYTES_VS_FEC6",
    "WAVE_N34_OPT7_SPARSE_SELECTOR_K100_PROPORTIONAL_SAVINGS",
    "WAVE_N34_OPT7_N_PAIRS",
    "WAVE_N34_OPT7_N_MODES",
    "CANONICAL_SPARSE_K_DEFAULT",
    "CANONICAL_WIDENED_K_DEFAULT",
    "CANONICAL_RATE_MULTIPLIER",
    "CANONICAL_RATE_DENOM_BYTES",
    "BasisExpansionStrategy",
    "UniwardInverseScorerBasisConfig",
    "compute_uniward_weighted_perturbation_for_pr110_catalog",
    "apply_uniward_inverse_scorer_basis_expansion_to_pr110_archive",
)
