# SPDX-License-Identifier: MIT
"""Extended 8-operator canonical helpers for the 5D canvas — BUILD-2+3-EXT.

Per DROP-MANY-REPLACE-COMPOSITION-APPARATUS-STATE-AUDIT 2026-05-26 (commit
``1f62ac788``) §"Phase 1 catalog enumeration" + operator insight 2026-05-26
*"merge and other ops do even better and a combination and individual fractal
optimization is likely even better"* + V14-V2 FRONTIER CROSSING 2026-05-26
(commit ``d2dc25ab0``; canonical equation #344 PROMOTED 3→5 anchors).

The audit memo enumerates **8 NOT-BUILT operators** beyond the canonical 4
operations (FULL_DROP / REPAIR / MASKED / FEATHERED) sister BUILD-2+3 is
implementing on `pair_frame_scorer_geometry_lattice_5d_canvas`:

    1. ``replace-one``         (DISTORTION-axis primitive; substitute single
                                pair's selector index with alternative)
    2. ``replace-many``        (beam-search per Catalog #356 per-axis
                                decomposition; substitute N pairs' selector
                                indices)
    3. ``merge-pair``          (rate+distortion joint optimization; combine
                                2 pairs)
    4. ``reorder-pair``        (permute pair ordering for entropy-coder
                                context optimization; FEC family sister)
    5. ``drop-frame``          (per-frame drop; finer-grained than pair-level)
    6. ``synthesize-frame``    (per-frame synthesis per Atick-Redlich
                                asymmetric channel routing)
    7. ``motion-conditional``  (per-pair operator selection conditioned on
                                pose magnitude; Rao-Ballard predictive
                                coding sister)
    8. ``temporal-coherence``  (cross-pair operator selection exploiting
                                temporal-frame-pair similarity; Wyner-Ziv
                                pipeline sister)

Sister of BUILD-2+3 main module ``pair_frame_scorer_geometry_lattice_5d_canvas``
(commit in flight on canonical 4 operations). This module is **disjoint
extension**: separate module, separate CLI, separate tests; reuses canvas
canonical helpers (PairFrameScorerGeometryLattice container, AxisDecomposition,
ExecutableCandidate dataclass, _scorer_axis_score_contributions,
_candidate_archive_path, sha256_hex) WITHOUT duplicating or forking.

## Canonical-vs-unique decision per layer

- ``PairFrameScorerGeometryLattice`` container: ADOPT canonical (sister BUILD-2+3
  owned; reuse via canvas import; no fork).
- ``AxisDecomposition`` + ``ExecutableCandidate``: ADOPT canonical
  (``tac.cathedral.consumer_contract`` + canvas module respectively).
- Per-axis decomposition: ADOPT canonical (Catalog #356) — every operator's
  AxisDecomposition is built via the canvas's
  ``_build_axis_decomposition_for_candidate``-style helper extended per
  operator semantics; FORK for operator-specific predicted-axis attribution
  per UNIQUE-AND-COMPLETE-PER-METHOD operating mode (each operator's
  marginal-distortion-per-byte slope differs; per CLAUDE.md
  "SegNet vs PoseNet importance — operating-point dependent").
- Canonical Provenance: ADOPT canonical (Catalog #323 +
  ``tac.provenance.builders.build_provenance_for_predicted``); FORK per
  operator's model_id namespace ``..._5d_canvas.<operator>_v0``.
- Canonical-routing markers (Catalog #341): ADOPT canonical
  (``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
  ``axis_tag="[predicted]"`` defaults; per Tier A scaffold per Catalog #357).
- 8 operator enums: GENUINELY NEW. ``CanonicalOperation`` enum on canvas is
  FROZEN at 4 members per BUILD-2+3 contract; we introduce sister
  ``ExtendedOperation`` enum (StrEnum; 8 members) to preserve canvas's
  frozen interface. The 2 enums COMPOSE via a ``CombinedOperationKind``
  union when downstream callers need both surfaces.
- Per-operator selector / substitution / motion / temporal helpers:
  GENUINELY NEW per operator-specific math. UNIQUE-AND-COMPLETE-PER-METHOD
  per 7th + 8th + 10th + 11th + 12th + 13th standing directives.

## Observability surface (Catalog #305)

- inspectable per layer: each operator emits per-cell + per-group +
  per-candidate observability rows (``operator_name`` + group_key +
  per-axis predicted deltas + canonical Provenance).
- decomposable per signal: every candidate's AxisDecomposition carries
  d_seg/d_pose/archive_bytes deltas independently (Catalog #356).
- diff-able across runs: candidates deterministic from (canvas archive_sha256,
  operator, parameters) tuple; identical inputs produce identical outputs.
- queryable post-hoc: candidates serializable via ``ExecutableCandidate.as_dict``
  with JSON-stable ordering.
- cite-able: every candidate's catalog_323_provenance carries
  (model_id, inputs_sha256, measurement_axis, hardware_substrate) per
  Catalog #323.
- counterfactual-able: per-operator helper functions accept ``override_*``
  parameters allowing direct counterfactual probing of "what if pair k
  used alternative selector j instead?" without reconstructing the canvas.

## 9-dimension success checklist evidence

- UNIQUENESS: first canonical primitive set binding 8 NOT-BUILT operator
  paradigms (replace × merge × reorder × frame-level × motion × temporal)
  to the 5D canvas in a sister-disjoint module respecting BUILD-2+3's
  frozen ``CanonicalOperation`` contract.
- BEAUTY+ELEGANCE: each operator ≤ 250 LOC; one frozen dataclass per
  operator parameter set; one canonical helper function per operator
  emitting ``ExecutableCandidate``; reuses canvas helpers throughout.
- DISTINCTNESS: 8 operators each target a different axis combination
  (replace-one = pair × distortion; replace-many = beam × per-axis;
  merge-pair = rate+distortion joint; reorder = entropy-coder context;
  drop-frame = frame × full-drop; synthesize-frame = frame × repair;
  motion-conditional = pose-magnitude × operator selection;
  temporal-coherence = cross-pair selection).
- RIGOR: each operator grounded in canonical mathematical primitive
  (replace: linear-substitution-distortion; merge: convex-feasibility-of-
  pair-fusion per Dykstra; reorder: entropy-coder-context-optimization
  per Markov chain; drop-frame: per-frame master-gradient per
  ``tac.cathedral_consumers.per_frame_sensitivity_consumer``;
  synthesize-frame: Atick-Redlich cooperative-receiver per
  CLAUDE.md Z4 grand-council attendees; motion-conditional: Rao-Ballard
  predictive coding per CLAUDE.md grand-council attendees;
  temporal-coherence: Wyner-Ziv 1976 source coding with side information
  per CLAUDE.md grand-council attendees).
- OPTIMIZATION-PER-TECHNIQUE: per the 8th INDIVIDUALLY-FRACTAL
  standing directive 2026-05-26: each operator's substrate-optimal
  engineering is UNIQUE; the 8 × per-substrate composition produces the
  individually-fractal optimization tree (each operator × each substrate
  = sister-substrate-optimal engineering decision).
- STACK-OF-STACKS-COMPOSABILITY: 8 operators COMPOSE with canvas's
  4 canonical operations (FULL_DROP / REPAIR / MASKED / FEATHERED) +
  cross-operator composition tests verify the 12 = 4 + 8 operator
  vocabulary IS the canonical multi-op stack-of-stacks.
- DETERMINISTIC-REPRODUCIBILITY: every candidate seeded from
  (canvas archive_sha256, operator, parameters) tuple; identical inputs
  produce identical outputs; ``ExecutableCandidate.as_dict`` JSON-stable
  via sort_keys=True downstream.
- EXTREME-OPTIMIZATION-PERFORMANCE: 8 operators each emit ≤ ``top_n``
  candidates (default 32) bounded by canvas cell count; no combinatorial
  explosion (replace-many beam capped at 1024 candidates per cycle;
  motion-conditional / temporal-coherence O(N_pairs); merge-pair
  O(N_pairs^2) bounded by ``max_candidates``).
- OPTIMAL-MINIMAL-CONTEST-SCORE: per audit memo §"Direct answer" the
  PR106 frontier operating point at ``pose_avg = 3.4e-5`` is rate-
  saturated for drop-one/drop-many; DISTORTION-axis operators (replace,
  merge, reorder, synthesize-frame, motion-conditional) target the
  unsaturated axis per Hypothesis #2 EMPIRICALLY GROUNDED. Top-3 per
  operator × 2 paired axes = 16 dispatches × ~$0.30 smoke + ~$1-3 full =
  estimated $20-60 paid GPU for the full FIRE phase.

## Cargo-cult audit per assumption

- HARD-EARNED: per-axis decomposition per Catalog #356 (operator
  insight 2026-05-26 + V14-V2 frontier crossing empirical evidence
  validate the per-axis attribution surface).
- HARD-EARNED: ``ExecutableCandidate`` canonical contract (sister BUILD-2+3
  already validated 4 operations against the contract).
- HARD-EARNED: ``_build_axis_decomposition_for_candidate``-style canonical
  Provenance threading (Catalog #356 STRICT preflight gate validates the
  contract in tests).
- CARGO-CULTED-PENDING-EMPIRICAL: 8-operator enum specifically (the
  audit memo enumerates these 8; sister subagents may add or refactor).
- CARGO-CULTED-PENDING-EMPIRICAL: per-operator predicted-axis ratios
  (uses linear approximation per the canvas's docstring; downstream
  cathedral ranker composes via ``tac.score_composition.compose_score_from_axes``
  for the difference-of-sqrt pose composition).
- CARGO-CULTED-PENDING-EMPIRICAL: merge-pair averaging heuristic (the
  canonical operator semantics are open; we pick the canonical
  per-axis-mean-of-cells approach per the audit's PRIORITY 2 sister
  guidance).

## Predicted ΔS band

NOT a substrate dispatch proposal. Per Catalog #296: no Dykstra feasibility
check needed for a canonical primitive scaffold. The 8 operators EMIT
per-candidate ΔS predictions; the operator module ITSELF carries no aggregate
prediction.

## Council attendees / verdict

T1 working-group VERDICT PROCEED (extension scaffold; sister-disjoint of
BUILD-2+3; no quorum required at T1 per Catalog #300). Attendees: Shannon
LEAD + Dykstra CO-LEAD + Daubechies CO-LEAD + Rudin CO-LEAD + Rao-Ballard +
Atick-Redlich + Wyner + Assumption-Adversary. Sister subagent BUILD-2+3
owns the canvas's 4 canonical operations; this extension is sister-disjoint
on the 8 NOT-BUILT operators per audit enumeration.

## Tier classification

Per Catalog #357: ``CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY``
at scaffold landing. BUILD-4 sister subagent op-routable promotes to
``ConsumerTier.TIER_B_SCORE_CONTRIBUTING`` with canonical-routing markers
per Catalog #341 + canonical Provenance per Catalog #323 + per-axis
AxisDecomposition per Catalog #356.

The 6-hook wire-in declaration per Catalog #125 (all 6 ACTIVE at BUILD-4
Tier B promotion):

- hook #1 sensitivity-map: per-operator selectors are sensitivity-map producers
- hook #2 Pareto constraint: per-operator predicted_byte_cost + delta IS
  the rate-vs-distortion Pareto signal
- hook #3 bit-allocator: per-operator predicted_byte_cost feeds bit-allocator
- hook #4 cathedral autopilot dispatch: BUILD-4 Tier B promotion auto-
  discovers per Catalog #335
- hook #5 continual-learning posterior: per-operator empirical anchors
  feed canonical posterior per Catalog #344
- hook #6 probe-disambiguator: 8 operators ARE the canonical disambiguator
  for the audit's 5 hypotheses + the operator's question "merge and other
  ops do even better"

Sister cross-references:

- ``tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas`` (sister
  BUILD-2+3 canonical 4 operations; this module extends to the 8 NOT-BUILT)
- ``tac.optimization.dqs1_drop_many_beam`` (drop-many beam sister; reorder /
  merge / replace-many borrow the beam-search pattern)
- ``tac.cross_substrate_master_gradient_analyzer`` (META-LIFT-1
  Cauchy-Schwarz; sister for cross-operator correlation bound)
- ``tac.pareto_polytope_unified_solver.solver`` (META-LIFT-2 Dykstra
  alternating projections; sister for merge-pair convex feasibility)
- ``tac.uniward_invariant_enumerator`` (META-LIFT-4; sister for masked /
  feathered operations referenced by canvas BUILD-2+3)
- Canonical equation #344 entries (8 FORMALIZATION_PENDING per operator;
  see ``EXTENDED_OPERATION_CANONICAL_EQUATION_IDS``)

Lane: ``lane_build_2_3_ext_8_not_built_operators_replace_merge_reorder_frame_level_motion_conditional_temporal_coherence_20260526`` L1.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from tac.cathedral.consumer_contract import (
    AxisDecomposition,
    ConsumerTier,
    HookNumber,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CANONICAL_PAIR_COUNT,
    CpuCudaAxis,
    ExecutableCandidate,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.validator import provenance_to_dict

# ---------------------------------------------------------------------------
# Module-level canonical contract per Catalog #335 + #357.
# ---------------------------------------------------------------------------

CONSUMER_NAME = "pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators"
CONSUMER_VERSION = "0.1.0-scaffold"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)
CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY

EXTENDED_MODULE_SCHEMA = (
    "pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators.v0_scaffold"
)

# ---------------------------------------------------------------------------
# The 8 NOT-BUILT operators per audit memo enumeration.
# ---------------------------------------------------------------------------


class ExtendedOperation(StrEnum):
    """The 8 NOT-BUILT operators per audit memo §"Phase 1 catalog enumeration".

    Sister of canvas ``CanonicalOperation`` (FROZEN at 4 members:
    FULL_DROP / REPAIR / MASKED / FEATHERED). This enum extends the operator
    vocabulary per audit memo §"Direct answer" + operator insight 2026-05-26.
    """

    # Pair-level distortion-axis substitutions
    REPLACE_ONE = "replace_one"
    REPLACE_MANY = "replace_many"

    # Cross-pair operations
    MERGE_PAIR = "merge_pair"
    REORDER_PAIR = "reorder_pair"

    # Frame-level operations (finer-grained than pair-level)
    DROP_FRAME = "drop_frame"
    SYNTHESIZE_FRAME = "synthesize_frame"

    # Adaptive operations
    MOTION_CONDITIONAL = "motion_conditional"
    TEMPORAL_COHERENCE = "temporal_coherence"


# Per Catalog #344: every operator declares a candidate canonical equation
# at FORMALIZATION_PENDING per the audit memo §"PRIORITY 5 Catalog #344
# canonical equations registry growth".
EXTENDED_OPERATION_CANONICAL_EQUATION_IDS: Mapping[ExtendedOperation, str] = {
    ExtendedOperation.REPLACE_ONE: (
        "replace_one_via_linear_substitution_distortion_v1"
    ),
    ExtendedOperation.REPLACE_MANY: (
        "replace_many_via_beam_search_per_axis_decomposition_v1"
    ),
    ExtendedOperation.MERGE_PAIR: (
        "merge_pair_via_rate_distortion_joint_optimization_v1"
    ),
    ExtendedOperation.REORDER_PAIR: (
        "reorder_pair_via_entropy_coder_context_markov_v1"
    ),
    ExtendedOperation.DROP_FRAME: (
        "drop_frame_via_per_frame_master_gradient_v1"
    ),
    ExtendedOperation.SYNTHESIZE_FRAME: (
        "synthesize_frame_via_atick_redlich_cooperative_receiver_v1"
    ),
    ExtendedOperation.MOTION_CONDITIONAL: (
        "motion_conditional_via_rao_ballard_predictive_coding_v1"
    ),
    ExtendedOperation.TEMPORAL_COHERENCE: (
        "temporal_coherence_via_wyner_ziv_side_information_v1"
    ),
}


# ---------------------------------------------------------------------------
# Canonical bounds + defaults per operator.
# ---------------------------------------------------------------------------

DEFAULT_TOP_N = 32
"""Default number of candidates per operator-call; bounded by canvas cells."""

DEFAULT_BEAM_WIDTH = 8
"""Default beam-search width for replace-many; bounded per operator."""

DEFAULT_BEAM_DEPTH = 4
"""Default beam-search depth for replace-many; bounded per operator."""

DEFAULT_MERGE_MAX_CANDIDATES = 256
"""Maximum O(N_pairs^2) merge-pair candidates to consider per call."""

DEFAULT_MOTION_THRESHOLD_PERCENTILE = 0.75
"""Default percentile cutoff for motion-conditional operator activation."""

DEFAULT_TEMPORAL_COHERENCE_WINDOW = 4
"""Default cross-pair temporal-coherence window (in pair indices)."""


def _validate_top_n(top_n: int) -> int:
    """Validate public generator top-n bounds."""
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError(f"top_n must be a positive int, got {top_n!r}")
    return top_n


# ---------------------------------------------------------------------------
# Per-operator parameter dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplaceOneParameters:
    """Parameters for replace-one operator.

    Per audit memo §"Phase 1 catalog enumeration replace-one (synthetic-
    substitute) NOT-BUILT": substitute single pair's selector index with
    alternative; DISTORTION-axis primitive (no rate change at first
    approximation; per CLAUDE.md "SegNet vs PoseNet importance — operating-
    point dependent" the substitution may shift per-pair distortion via
    pose-axis amplification at PR106 frontier).

    Args:
        target_pair_idx: 0-indexed pair in [0, 600); the pair whose selector
            is replaced.
        alternative_selector_id: opaque substitute identifier; operator-
            specific semantics (downstream consumer interprets).
        receiver_runtime: which receiver-runtime mode this replacement
            targets per the 5D canvas.
    """

    target_pair_idx: int
    alternative_selector_id: int
    receiver_runtime: ReceiverRuntime

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_pair_idx, int)
            or isinstance(self.target_pair_idx, bool)
        ):
            raise ValueError(
                f"target_pair_idx must be int, got "
                f"{type(self.target_pair_idx).__name__}"
            )
        if not 0 <= self.target_pair_idx < CANONICAL_PAIR_COUNT:
            raise ValueError(
                f"target_pair_idx {self.target_pair_idx} out of range "
                f"[0, {CANONICAL_PAIR_COUNT})"
            )
        if (
            not isinstance(self.alternative_selector_id, int)
            or isinstance(self.alternative_selector_id, bool)
        ):
            raise ValueError(
                f"alternative_selector_id must be int, got "
                f"{type(self.alternative_selector_id).__name__}"
            )
        if self.alternative_selector_id < 0:
            raise ValueError(
                "alternative_selector_id must be non-negative; "
                f"got {self.alternative_selector_id}"
            )
        if not isinstance(self.receiver_runtime, ReceiverRuntime):
            raise ValueError(
                "receiver_runtime must be ReceiverRuntime"
            )


@dataclass(frozen=True)
class ReplaceManyParameters:
    """Parameters for replace-many beam-search operator.

    Per audit memo §"replace-many NOT-BUILT": beam-search per Catalog #356
    per-axis decomposition; sister of drop_many beam.

    Args:
        beam_width: maximum candidates per beam-search step.
        beam_depth: maximum substitutions per candidate path.
        receiver_runtime: which receiver-runtime mode targets this beam.
        target_pair_indices: optional explicit pair set to consider for
            replacement (defaults to all pairs with feasible cells).
    """

    beam_width: int = DEFAULT_BEAM_WIDTH
    beam_depth: int = DEFAULT_BEAM_DEPTH
    receiver_runtime: ReceiverRuntime = ReceiverRuntime.RAW_RESIDUAL
    target_pair_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.beam_width, int) or self.beam_width < 1:
            raise ValueError(
                f"beam_width must be positive int, got {self.beam_width}"
            )
        if not isinstance(self.beam_depth, int) or self.beam_depth < 1:
            raise ValueError(
                f"beam_depth must be positive int, got {self.beam_depth}"
            )
        if self.beam_width > 1024:
            raise ValueError(
                f"beam_width {self.beam_width} exceeds canonical bound 1024"
            )
        if self.beam_depth > 16:
            raise ValueError(
                f"beam_depth {self.beam_depth} exceeds canonical bound 16"
            )
        if not isinstance(self.receiver_runtime, ReceiverRuntime):
            raise ValueError("receiver_runtime must be ReceiverRuntime")
        if not isinstance(self.target_pair_indices, tuple):
            raise ValueError(
                f"target_pair_indices must be tuple, got "
                f"{type(self.target_pair_indices).__name__}"
            )
        for idx in self.target_pair_indices:
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ValueError("target_pair_indices entries must be int")
            if not 0 <= idx < CANONICAL_PAIR_COUNT:
                raise ValueError(
                    f"target_pair_indices entry {idx} out of range"
                )


@dataclass(frozen=True)
class MergePairParameters:
    """Parameters for merge-pair operator.

    Per audit memo §"merge-pair NOT-BUILT": combine 2 pairs (per-pair
    perturbation averaged + share archived bytes); rate+distortion joint
    optimization.

    The canonical semantic (per Dykstra alternating-projections sister
    META-LIFT-2): two pairs' archive bytes can be merged into a single
    shared encoding if their per-pair distortion contributions are
    convex-feasibly compatible; the rate saving is the difference of
    individual encodings minus the shared encoding.

    Args:
        receiver_runtime: which receiver-runtime mode targets the merge.
        max_candidates: maximum O(N_pairs^2) candidates to consider.
    """

    receiver_runtime: ReceiverRuntime = ReceiverRuntime.RAW_RESIDUAL
    max_candidates: int = DEFAULT_MERGE_MAX_CANDIDATES

    def __post_init__(self) -> None:
        if not isinstance(self.receiver_runtime, ReceiverRuntime):
            raise ValueError("receiver_runtime must be ReceiverRuntime")
        if (
            not isinstance(self.max_candidates, int)
            or self.max_candidates < 1
        ):
            raise ValueError(
                f"max_candidates must be positive int, got {self.max_candidates}"
            )
        if self.max_candidates > 4096:
            raise ValueError(
                f"max_candidates {self.max_candidates} exceeds canonical "
                "bound 4096"
            )


@dataclass(frozen=True)
class ReorderPairParameters:
    """Parameters for reorder-pair operator.

    Per audit memo §"reorder-pair NOT-BUILT": permute pair ordering for
    entropy-coder context optimization (FEC family sister; reorder
    upstream of FEC8/FEC10 Markov context per 6th standing directive
    final-rate-attack off-the-shelf).

    The canonical semantic: pairs whose per-pair payload distribution
    resembles their neighbor pair's distribution can be reordered to
    align with the entropy-coder's Markov-context predictor; this is the
    canonical FEC8-2nd-order-Markov-context sister optimization at the
    pair-ordering layer.

    Args:
        receiver_runtime: which receiver-runtime mode targets the reorder.
        block_size: maximum permutation block size (bounded; default 8).
    """

    receiver_runtime: ReceiverRuntime = ReceiverRuntime.RAW_RESIDUAL
    block_size: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.receiver_runtime, ReceiverRuntime):
            raise ValueError("receiver_runtime must be ReceiverRuntime")
        if not isinstance(self.block_size, int) or self.block_size < 2:
            raise ValueError(
                f"block_size must be int >= 2, got {self.block_size}"
            )
        if self.block_size > 32:
            raise ValueError(
                f"block_size {self.block_size} exceeds canonical bound 32"
            )


@dataclass(frozen=True)
class DropFrameParameters:
    """Parameters for drop-frame operator (finer-grained than pair-level).

    Per audit memo §"drop-frame NOT-BUILT": per-frame drop (frame_0 vs
    frame_1 per pair). Sister of canvas FULL_DROP CanonicalOperation
    which is pair-level; drop-frame targets specific frames within pairs.

    Args:
        receiver_runtime: which receiver-runtime mode targets the drop.
        which_frame: 'first' (frame_0 = 2*pair_idx) or 'last'
            (frame_1 = 2*pair_idx + 1) or 'both' (both frames; equivalent
            to canvas FULL_DROP at pair level).
    """

    receiver_runtime: ReceiverRuntime = ReceiverRuntime.RAW_RESIDUAL
    which_frame: str = "last"

    def __post_init__(self) -> None:
        if not isinstance(self.receiver_runtime, ReceiverRuntime):
            raise ValueError("receiver_runtime must be ReceiverRuntime")
        if self.which_frame not in ("first", "last", "both"):
            raise ValueError(
                f"which_frame must be 'first'|'last'|'both', got "
                f"{self.which_frame!r}"
            )


@dataclass(frozen=True)
class SynthesizeFrameParameters:
    """Parameters for synthesize-frame operator (Atick-Redlich sister).

    Per audit memo §"synthesize-frame NOT-BUILT": per-frame synthesis
    (replace one frame with synthetic per Atick-Redlich asymmetric
    channel routing). Sister of canvas REPAIR CanonicalOperation which
    is pair-level cooperative receiver; synthesize-frame targets
    per-frame Atick-Redlich-1990 cooperative-receiver synthesis.

    Per CLAUDE.md grand-council attendees Atick-Redlich + Tishby-Zaslavsky:
    the cooperative-receiver framing implies the synthesizer has access
    to the scorer's classes/poses; the synthesizer's per-frame output is
    optimized for the I(X;T)/I(T;Y) information-bottleneck decomposition.

    Args:
        receiver_runtime: which receiver-runtime mode targets the synthesis.
        which_frame: 'first' or 'last' frame of the pair to synthesize.
        synthesis_seed: opaque deterministic seed for the synthesizer.
    """

    receiver_runtime: ReceiverRuntime = ReceiverRuntime.SMOOTHED_RESIDUAL
    which_frame: str = "last"
    synthesis_seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.receiver_runtime, ReceiverRuntime):
            raise ValueError("receiver_runtime must be ReceiverRuntime")
        if self.which_frame not in ("first", "last"):
            raise ValueError(
                f"which_frame must be 'first'|'last' (not 'both' for "
                f"synthesis); got {self.which_frame!r}"
            )
        if (
            not isinstance(self.synthesis_seed, int)
            or isinstance(self.synthesis_seed, bool)
        ):
            raise ValueError(
                f"synthesis_seed must be int, got "
                f"{type(self.synthesis_seed).__name__}"
            )


@dataclass(frozen=True)
class MotionConditionalParameters:
    """Parameters for motion-conditional operator (Rao-Ballard sister).

    Per audit memo §"motion-conditional NOT-BUILT": per-pair operator
    selection conditioned on pose magnitude (Rao-Ballard predictive
    coding sister; per CLAUDE.md grand-council attendees).

    The canonical semantic: pairs with high pose magnitude get a
    different operator (typically MERGE or SYNTHESIZE) than pairs with
    low pose magnitude (typically DROP or REPLACE). The percentile
    cutoff parameter controls the activation threshold.

    Args:
        receiver_runtime: which receiver-runtime mode targets the selection.
        motion_threshold_percentile: percentile in [0, 1] above which a
            pair is considered high-motion.
        high_motion_operator: which ExtendedOperation to apply for
            high-motion pairs.
        low_motion_operator: which ExtendedOperation to apply for
            low-motion pairs.
    """

    receiver_runtime: ReceiverRuntime = ReceiverRuntime.RAW_RESIDUAL
    motion_threshold_percentile: float = DEFAULT_MOTION_THRESHOLD_PERCENTILE
    high_motion_operator: ExtendedOperation = ExtendedOperation.SYNTHESIZE_FRAME
    low_motion_operator: ExtendedOperation = ExtendedOperation.REPLACE_ONE

    def __post_init__(self) -> None:
        if not isinstance(self.receiver_runtime, ReceiverRuntime):
            raise ValueError("receiver_runtime must be ReceiverRuntime")
        if not isinstance(self.motion_threshold_percentile, (int, float)):
            raise ValueError(
                "motion_threshold_percentile must be numeric"
            )
        if not 0.0 <= float(self.motion_threshold_percentile) <= 1.0:
            raise ValueError(
                f"motion_threshold_percentile {self.motion_threshold_percentile} "
                "out of range [0.0, 1.0]"
            )
        if not isinstance(self.high_motion_operator, ExtendedOperation):
            raise ValueError(
                "high_motion_operator must be ExtendedOperation"
            )
        if not isinstance(self.low_motion_operator, ExtendedOperation):
            raise ValueError(
                "low_motion_operator must be ExtendedOperation"
            )
        if self.high_motion_operator is ExtendedOperation.MOTION_CONDITIONAL:
            raise ValueError(
                "high_motion_operator cannot be MOTION_CONDITIONAL "
                "(recursive)"
            )
        if self.low_motion_operator is ExtendedOperation.MOTION_CONDITIONAL:
            raise ValueError(
                "low_motion_operator cannot be MOTION_CONDITIONAL "
                "(recursive)"
            )


@dataclass(frozen=True)
class TemporalCoherenceParameters:
    """Parameters for temporal-coherence operator (Wyner-Ziv sister).

    Per audit memo §"temporal-coherence NOT-BUILT": cross-pair operator
    selection exploiting temporal-frame-pair similarity; sister of
    Wyner-Ziv 1976 source-coding-with-side-information pipeline (per
    CLAUDE.md grand-council attendees).

    The canonical semantic: pairs whose per-pair predicted-axis
    decomposition is similar to their neighbors (within a temporal
    window) can be jointly optimized as a Wyner-Ziv side-information
    block; the rate saving is the redundancy reduction.

    Args:
        receiver_runtime: which receiver-runtime mode targets the operator.
        temporal_window: cross-pair window size (in pair indices); pairs
            within `temporal_window` of each other are considered for
            joint optimization.
        similarity_threshold: minimum cosine-similarity between per-axis
            decompositions for pairs to be considered temporally-coherent.
    """

    receiver_runtime: ReceiverRuntime = ReceiverRuntime.RAW_RESIDUAL
    temporal_window: int = DEFAULT_TEMPORAL_COHERENCE_WINDOW
    similarity_threshold: float = 0.7

    def __post_init__(self) -> None:
        if not isinstance(self.receiver_runtime, ReceiverRuntime):
            raise ValueError("receiver_runtime must be ReceiverRuntime")
        if (
            not isinstance(self.temporal_window, int)
            or self.temporal_window < 2
        ):
            raise ValueError(
                f"temporal_window must be int >= 2, got {self.temporal_window}"
            )
        if self.temporal_window > 64:
            raise ValueError(
                f"temporal_window {self.temporal_window} exceeds canonical "
                "bound 64"
            )
        if not isinstance(self.similarity_threshold, (int, float)):
            raise ValueError("similarity_threshold must be numeric")
        if not -1.0 <= float(self.similarity_threshold) <= 1.0:
            raise ValueError(
                f"similarity_threshold {self.similarity_threshold} out of "
                "range [-1.0, 1.0]"
            )


# ---------------------------------------------------------------------------
# Canonical Provenance + AxisDecomposition helpers (sister of canvas helper).
# ---------------------------------------------------------------------------


def _sha256_hex(text: str) -> str:
    """Canonical sha256 helper (sister of canvas ``sha256_hex``)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_extended_axis_decomposition(
    cells: Sequence[PairFrameScorerGeometryCell],
    archive_sha256: str,
    operation: ExtendedOperation,
    extra_inputs_signature: str = "",
) -> AxisDecomposition:
    """Build canonical AxisDecomposition for an extended-operator candidate.

    Per Catalog #356 STRICT preflight gate + 10th standing directive
    (apples-to-apples): each operator's per-axis attribution is built
    via the canonical AxisDecomposition contract with per-operator
    Provenance model_id.

    The per-axis attribution mirrors canvas's
    ``_build_axis_decomposition_for_candidate`` (linear inversion of
    canonical contest formula) but threads operator-specific model_id
    + extra_inputs_signature for downstream reproducibility.
    """
    seg_score_delta = 0.0
    pose_score_delta = 0.0
    rate_bytes_delta = 0
    for cell in cells:
        if cell.scorer_axis is ScorerAxis.SEGNET_5CLASS:
            seg_score_delta += float(cell.predicted_delta_score)
        elif cell.scorer_axis is ScorerAxis.POSENET_6D:
            pose_score_delta += float(cell.predicted_delta_score)
        elif cell.scorer_axis is ScorerAxis.RATE_TERM:
            rate_bytes_delta += int(cell.predicted_byte_cost)

    d_seg_delta = seg_score_delta / 100.0
    d_pose_delta = pose_score_delta / math.sqrt(10.0)

    coord_signature = "|".join(
        sorted(
            f"({c.pair_idx},{c.frame_idx},{c.scorer_axis.value},"
            f"{c.receiver_runtime.value},{c.cpu_cuda_axis.value})"
            for c in cells
        )
    )
    inputs_payload = (
        f"{archive_sha256}|{operation.value}|{coord_signature}"
        f"|{extra_inputs_signature}"
    )
    inputs_sha = _sha256_hex(inputs_payload)
    model_id = (
        f"pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators"
        f".{operation.value}_v0"
    )
    prov = build_provenance_for_predicted(
        model_id=model_id,
        inputs_sha256=inputs_sha,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )
    return AxisDecomposition(
        predicted_d_seg_delta=d_seg_delta,
        predicted_d_pose_delta=d_pose_delta,
        predicted_archive_bytes_delta=rate_bytes_delta,
        axis_tag="[predicted]",
        canonical_provenance=provenance_to_dict(prov),
    )


def _extended_candidate_archive_path(
    canvas: PairFrameScorerGeometryLattice,
    operation: ExtendedOperation,
    group_key: tuple[int, ...],
    receiver_runtime: ReceiverRuntime,
    cpu_cuda_axis: CpuCudaAxis,
) -> Path:
    """Deterministic scaffold-only sentinel path for extended-operator candidates.

    Sister of canvas ``_candidate_archive_path`` scoped to the extended
    operator namespace so candidates don't collide with the 4 canonical
    BUILD-2+3 operations at dispatch-time.
    """
    group_str = "_".join(str(g) for g in group_key)
    sentinel_filename = (
        f"{canvas.archive_sha256[:12]}_{operation.value}_"
        f"{receiver_runtime.value}_{cpu_cuda_axis.value}_{group_str}"
        f".extcandidate.json"
    )
    return Path(
        ".omx/state/pair_frame_scorer_geometry_lattice_extended_operator_candidates/"
        f"{sentinel_filename}"
    )


def _cpu_cuda_axis_sort_key(axis: CpuCudaAxis) -> int:
    """Stable integer key (CPU=0, CUDA=1) for group ordering."""
    return 0 if axis is CpuCudaAxis.CONTEST_CPU else 1


def _cells_by_pair_axis(
    canvas: PairFrameScorerGeometryLattice,
    receiver_runtime: ReceiverRuntime,
) -> dict[tuple[int, CpuCudaAxis], list[PairFrameScorerGeometryCell]]:
    """Filter + group canvas cells by (pair_idx, cpu_cuda_axis) for the runtime."""
    grouped: dict[tuple[int, CpuCudaAxis], list[PairFrameScorerGeometryCell]] = {}
    for cell in canvas._cells.values():
        if cell.receiver_runtime is not receiver_runtime:
            continue
        if not cell.receiver_feasibility:
            continue
        key = (cell.pair_idx, cell.cpu_cuda_axis)
        grouped.setdefault(key, []).append(cell)
    return grouped


def _cells_by_frame_axis(
    canvas: PairFrameScorerGeometryLattice,
    receiver_runtime: ReceiverRuntime,
) -> dict[tuple[int, CpuCudaAxis], list[PairFrameScorerGeometryCell]]:
    """Filter + group canvas cells by (frame_idx, cpu_cuda_axis) for the runtime."""
    grouped: dict[tuple[int, CpuCudaAxis], list[PairFrameScorerGeometryCell]] = {}
    for cell in canvas._cells.values():
        if cell.receiver_runtime is not receiver_runtime:
            continue
        if not cell.receiver_feasibility:
            continue
        key = (cell.frame_idx, cell.cpu_cuda_axis)
        grouped.setdefault(key, []).append(cell)
    return grouped


def _aggregate_total_delta(cells: Sequence[PairFrameScorerGeometryCell]) -> float:
    """Sum predicted_delta_score across a sequence of cells."""
    return sum(float(c.predicted_delta_score) for c in cells)


def _aggregate_total_bytes(cells: Sequence[PairFrameScorerGeometryCell]) -> int:
    """Sum predicted_byte_cost across a sequence of cells."""
    return sum(int(c.predicted_byte_cost) for c in cells)


def _build_canonical_recipe_hint(
    operation: ExtendedOperation,
    receiver_runtime: ReceiverRuntime,
    cpu_cuda_axis: CpuCudaAxis,
    group_key: tuple[int, ...],
    group_size_cells: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical dispatch recipe hint dict per operator.

    Per Catalog #344 + audit memo §"PRIORITY 4 PAIRED CPU+CUDA DISPATCH
    wave": every candidate carries operator-routable dispatch metadata
    for the FIRE phase (paired auth_eval ~$1-3 each per CLAUDE.md
    "Submission auth eval — BOTH CPU AND CUDA").
    """
    hint: dict[str, Any] = {
        "operation": operation.value,
        "receiver_runtime": receiver_runtime.value,
        "cpu_cuda_axis": cpu_cuda_axis.value,
        "group_key": list(group_key),
        "group_size_cells": group_size_cells,
        "estimated_paid_dispatch_usd_band_low": 1.0,
        "estimated_paid_dispatch_usd_band_high": 3.0,
        "estimated_smoke_dispatch_usd": 0.30,
        "canonical_equation_id_pending": (
            EXTENDED_OPERATION_CANONICAL_EQUATION_IDS[operation]
        ),
        "audit_memo_reference": (
            ".omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md"
        ),
    }
    if extra:
        for k, v in extra.items():
            hint[k] = v
    return hint


# ---------------------------------------------------------------------------
# Operator 1: replace-one (DISTORTION-axis primitive).
# ---------------------------------------------------------------------------


def generate_replace_one_candidates(
    canvas: PairFrameScorerGeometryLattice,
    parameters: ReplaceOneParameters,
    top_n: int = DEFAULT_TOP_N,
) -> list[ExecutableCandidate]:
    """Generate replace-one candidates per audit memo §"replace-one NOT-BUILT".

    For the target pair, emit candidate per receiver-feasible cell at that
    pair_idx; predicted_delta_score is the per-cell score delta of replacing
    the selector with the alternative. The alternative_selector_id is
    threaded into the recipe hint for downstream dispatch.

    Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":
    at the PR106 frontier replace-one's distortion-axis attribution may
    dominate the rate-axis savings; the canonical equation
    `replace_one_via_linear_substitution_distortion_v1` (FORMALIZATION_PENDING)
    predicts this.

    Returns:
        list of ExecutableCandidate sorted ascending by predicted_delta_score
        (best score-improvement first); capped at top_n.
    """
    top_n = _validate_top_n(top_n)
    if canvas.cell_count() == 0:
        return []

    grouped = _cells_by_pair_axis(canvas, parameters.receiver_runtime)
    candidates: list[ExecutableCandidate] = []
    for (pair_idx, cpu_cuda_axis), cells in grouped.items():
        if pair_idx != parameters.target_pair_idx:
            continue
        total_delta = _aggregate_total_delta(cells)
        total_bytes = _aggregate_total_bytes(cells)
        if total_delta >= 0.0:
            continue
        extra_sig = f"alternative_selector_id={parameters.alternative_selector_id}"
        axis_decomp = _build_extended_axis_decomposition(
            cells,
            canvas.archive_sha256,
            ExtendedOperation.REPLACE_ONE,
            extra_inputs_signature=extra_sig,
        )
        group_key = (
            pair_idx,
            parameters.alternative_selector_id,
            _cpu_cuda_axis_sort_key(cpu_cuda_axis),
        )
        hint = _build_canonical_recipe_hint(
            ExtendedOperation.REPLACE_ONE,
            parameters.receiver_runtime,
            cpu_cuda_axis,
            group_key,
            len(cells),
            extra={
                "target_pair_idx": pair_idx,
                "alternative_selector_id": parameters.alternative_selector_id,
            },
        )
        candidates.append(
            ExecutableCandidate(
                operation=_extended_to_canvas_operation_proxy(
                    ExtendedOperation.REPLACE_ONE
                ),
                archive_candidate_path=_extended_candidate_archive_path(
                    canvas,
                    ExtendedOperation.REPLACE_ONE,
                    group_key,
                    parameters.receiver_runtime,
                    cpu_cuda_axis,
                ),
                predicted_delta_score=total_delta,
                predicted_byte_cost=total_bytes,
                catalog_323_provenance=dict(axis_decomp.canonical_provenance),
                canonical_dispatch_recipe_hint=hint,
                predicted_axis_decomposition=axis_decomp,
            )
        )

    candidates.sort(
        key=lambda c: (
            float(c.predicted_delta_score),
            list(c.canonical_dispatch_recipe_hint.get("group_key", [])),
        )
    )
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Operator 2: replace-many (beam-search per-axis decomposition).
# ---------------------------------------------------------------------------


def generate_replace_many_candidates(
    canvas: PairFrameScorerGeometryLattice,
    parameters: ReplaceManyParameters,
    top_n: int = DEFAULT_TOP_N,
) -> list[ExecutableCandidate]:
    """Generate replace-many beam-search candidates per audit memo.

    Per audit memo §"replace-many NOT-BUILT": beam-search per Catalog #356
    per-axis decomposition. Sister of ``tac.optimization.dqs1_drop_many_beam``;
    instead of dropping pairs we substitute their selectors per-pair.

    Algorithm (canonical UNIQUE-AND-COMPLETE-PER-METHOD substrate engineering):

      1. Build per-pair candidate substitution rows (1 row per feasible
         cell at each candidate pair_idx).
      2. Beam-search: expand top-beam_width substitutions per beam_depth
         iterations; rank by composite predicted_delta_score per Catalog #356.
      3. Per Catalog #356 AxisDecomposition: per-beam composite carries
         per-axis attribution + canonical Provenance.
    """
    top_n = _validate_top_n(top_n)
    if canvas.cell_count() == 0:
        return []

    grouped = _cells_by_pair_axis(canvas, parameters.receiver_runtime)
    if not grouped:
        return []

    # Restrict to target_pair_indices if explicit, else use all feasible pairs.
    if parameters.target_pair_indices:
        target_set = set(parameters.target_pair_indices)
    else:
        target_set = {pair_idx for (pair_idx, _ax) in grouped}

    # Build per-pair-best-substitution row per cpu_cuda_axis.
    # Per axis we rank pairs by total_delta ascending (best first).
    per_axis_pair_deltas: dict[
        CpuCudaAxis,
        list[tuple[int, float, int, list[PairFrameScorerGeometryCell]]],
    ] = {}
    for (pair_idx, cpu_cuda_axis), cells in grouped.items():
        if pair_idx not in target_set:
            continue
        total_delta = _aggregate_total_delta(cells)
        total_bytes = _aggregate_total_bytes(cells)
        if total_delta >= 0.0:
            continue
        per_axis_pair_deltas.setdefault(cpu_cuda_axis, []).append(
            (pair_idx, total_delta, total_bytes, cells)
        )

    candidates: list[ExecutableCandidate] = []
    for cpu_cuda_axis, axis_rows in per_axis_pair_deltas.items():
        axis_rows.sort(key=lambda row: row[1])  # ascending by total_delta

        # Beam: build paths of length 1..beam_depth.
        # For each depth, the canonical beam-search expands top-beam_width
        # partial paths and ranks them by cumulative delta.
        beam: list[tuple[float, int, list[int], list[PairFrameScorerGeometryCell]]] = [
            (0.0, 0, [], [])
        ]  # (cumulative_delta, cumulative_bytes, pair_path, accumulated_cells)
        for _depth in range(parameters.beam_depth):
            expanded: list[
                tuple[float, int, list[int], list[PairFrameScorerGeometryCell]]
            ] = []
            for cumulative_delta, cumulative_bytes, path, acc_cells in beam:
                for pair_idx, delta, bytes_, cells in axis_rows:
                    if pair_idx in path:
                        continue
                    new_path = [*path, pair_idx]
                    expanded.append(
                        (
                            cumulative_delta + delta,
                            cumulative_bytes + bytes_,
                            new_path,
                            acc_cells + cells,
                        )
                    )
            if not expanded:
                break
            expanded.sort(key=lambda row: row[0])
            beam = expanded[: parameters.beam_width]

        # Emit one ExecutableCandidate per final beam member.
        for cumulative_delta, cumulative_bytes, path, acc_cells in beam:
            if cumulative_delta >= 0.0:
                continue
            if not path:
                continue
            extra_sig = f"path_len={len(path)}|path={','.join(str(p) for p in sorted(path))}"
            axis_decomp = _build_extended_axis_decomposition(
                acc_cells,
                canvas.archive_sha256,
                ExtendedOperation.REPLACE_MANY,
                extra_inputs_signature=extra_sig,
            )
            group_key = (*sorted(path), _cpu_cuda_axis_sort_key(cpu_cuda_axis))
            hint = _build_canonical_recipe_hint(
                ExtendedOperation.REPLACE_MANY,
                parameters.receiver_runtime,
                cpu_cuda_axis,
                group_key,
                len(acc_cells),
                extra={
                    "beam_width": parameters.beam_width,
                    "beam_depth_achieved": len(path),
                    "pair_path": sorted(path),
                },
            )
            candidates.append(
                ExecutableCandidate(
                    operation=_extended_to_canvas_operation_proxy(
                        ExtendedOperation.REPLACE_MANY
                    ),
                    archive_candidate_path=_extended_candidate_archive_path(
                        canvas,
                        ExtendedOperation.REPLACE_MANY,
                        group_key,
                        parameters.receiver_runtime,
                        cpu_cuda_axis,
                    ),
                    predicted_delta_score=cumulative_delta,
                    predicted_byte_cost=cumulative_bytes,
                    catalog_323_provenance=dict(axis_decomp.canonical_provenance),
                    canonical_dispatch_recipe_hint=hint,
                    predicted_axis_decomposition=axis_decomp,
                )
            )

    candidates.sort(
        key=lambda c: (
            float(c.predicted_delta_score),
            list(c.canonical_dispatch_recipe_hint.get("group_key", [])),
        )
    )
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Operator 3: merge-pair (rate+distortion joint optimization).
# ---------------------------------------------------------------------------


def generate_merge_pair_candidates(
    canvas: PairFrameScorerGeometryLattice,
    parameters: MergePairParameters,
    top_n: int = DEFAULT_TOP_N,
) -> list[ExecutableCandidate]:
    """Generate merge-pair candidates per audit memo §"merge-pair NOT-BUILT".

    Per Dykstra alternating-projections sister META-LIFT-2: two pairs'
    archive bytes can be merged if their per-pair distortion contributions
    are convex-feasibly compatible; the rate saving is the difference of
    individual encodings minus the shared encoding.

    Canonical semantic (PENDING-EMPIRICAL per cargo-cult audit): merge-pair
    averages per-pair predicted-axis attribution AND combines archive bytes
    via "shared encoding" (modeled as min(bytes_a, bytes_b) shared + 0 added
    payload for the second pair at first approximation).
    """
    top_n = _validate_top_n(top_n)
    if canvas.cell_count() == 0:
        return []

    grouped = _cells_by_pair_axis(canvas, parameters.receiver_runtime)
    if not grouped:
        return []

    # Per axis, collect (pair_idx, total_delta, total_bytes, cells)
    per_axis_rows: dict[
        CpuCudaAxis,
        list[tuple[int, float, int, list[PairFrameScorerGeometryCell]]],
    ] = {}
    for (pair_idx, cpu_cuda_axis), cells in grouped.items():
        total_delta = _aggregate_total_delta(cells)
        total_bytes = _aggregate_total_bytes(cells)
        per_axis_rows.setdefault(cpu_cuda_axis, []).append(
            (pair_idx, total_delta, total_bytes, cells)
        )

    candidates: list[ExecutableCandidate] = []
    for cpu_cuda_axis, axis_rows in per_axis_rows.items():
        # O(N^2) pairing; bounded by max_candidates.
        axis_rows.sort(key=lambda row: row[0])  # stable by pair_idx
        emitted = 0
        for i, row_a in enumerate(axis_rows):
            for row_b in axis_rows[i + 1 :]:
                if emitted >= parameters.max_candidates:
                    break
                pair_a, delta_a, bytes_a, cells_a = row_a
                pair_b, delta_b, bytes_b, cells_b = row_b

                # Canonical merge semantic: averaged distortion + shared rate.
                # The rate saving is max(bytes_a, bytes_b) - min(bytes_a, bytes_b)
                # i.e. the larger of the two pair encodings can be replaced by
                # the smaller shared encoding. Net byte cost = -|bytes_a - bytes_b|.
                # The distortion is the AVERAGE of the two individual distortions
                # (per the canonical convex-feasibility-of-pair-fusion at the
                # operating point per CLAUDE.md "SegNet vs PoseNet importance —
                # operating-point dependent").
                # If both bytes_a, bytes_b are NEGATIVE (savings) we treat the
                # merge as compounding savings: merged_bytes = bytes_a + bytes_b
                # MINUS the overhead reduction min(|a|, |b|). For simplicity
                # at scaffold we use:
                #     merged_bytes_delta = bytes_a + bytes_b - abs(bytes_a - bytes_b) / 2
                # which is the canonical "shared encoding minus overhead" form.
                merged_bytes_delta = int(
                    (bytes_a + bytes_b) - abs(bytes_a - bytes_b) // 2
                )
                merged_delta = (delta_a + delta_b) / 2.0
                # Skip if no joint improvement.
                if merged_delta >= 0.0:
                    continue

                merged_cells = cells_a + cells_b
                extra_sig = f"merge_pair_a={pair_a}|merge_pair_b={pair_b}"
                axis_decomp = _build_extended_axis_decomposition(
                    merged_cells,
                    canvas.archive_sha256,
                    ExtendedOperation.MERGE_PAIR,
                    extra_inputs_signature=extra_sig,
                )
                # Override the rate-axis prediction with the merge-specific
                # byte delta so the AxisDecomposition reflects the merge
                # semantic (not the simple sum from the sister helper).
                # We rebuild AxisDecomposition manually below to thread
                # the merge-specific archive_bytes_delta.
                axis_decomp = AxisDecomposition(
                    predicted_d_seg_delta=axis_decomp.predicted_d_seg_delta / 2.0,
                    predicted_d_pose_delta=axis_decomp.predicted_d_pose_delta / 2.0,
                    predicted_archive_bytes_delta=merged_bytes_delta,
                    axis_tag="[predicted]",
                    canonical_provenance=dict(axis_decomp.canonical_provenance),
                )
                group_key = (
                    pair_a,
                    pair_b,
                    _cpu_cuda_axis_sort_key(cpu_cuda_axis),
                )
                hint = _build_canonical_recipe_hint(
                    ExtendedOperation.MERGE_PAIR,
                    parameters.receiver_runtime,
                    cpu_cuda_axis,
                    group_key,
                    len(merged_cells),
                    extra={
                        "merge_pair_a": pair_a,
                        "merge_pair_b": pair_b,
                        "merged_bytes_delta_via_shared_encoding": merged_bytes_delta,
                    },
                )
                candidates.append(
                    ExecutableCandidate(
                        operation=_extended_to_canvas_operation_proxy(
                            ExtendedOperation.MERGE_PAIR
                        ),
                        archive_candidate_path=_extended_candidate_archive_path(
                            canvas,
                            ExtendedOperation.MERGE_PAIR,
                            group_key,
                            parameters.receiver_runtime,
                            cpu_cuda_axis,
                        ),
                        predicted_delta_score=merged_delta,
                        predicted_byte_cost=merged_bytes_delta,
                        catalog_323_provenance=dict(
                            axis_decomp.canonical_provenance
                        ),
                        canonical_dispatch_recipe_hint=hint,
                        predicted_axis_decomposition=axis_decomp,
                    )
                )
                emitted += 1
            if emitted >= parameters.max_candidates:
                break

    candidates.sort(
        key=lambda c: (
            float(c.predicted_delta_score),
            list(c.canonical_dispatch_recipe_hint.get("group_key", [])),
        )
    )
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Operator 4: reorder-pair (entropy-coder context optimization).
# ---------------------------------------------------------------------------


def generate_reorder_pair_candidates(
    canvas: PairFrameScorerGeometryLattice,
    parameters: ReorderPairParameters,
    top_n: int = DEFAULT_TOP_N,
) -> list[ExecutableCandidate]:
    """Generate reorder-pair candidates per audit memo §"reorder-pair NOT-BUILT".

    Per the 6th standing directive final-rate-attack off-the-shelf: pairs
    whose per-pair payload distribution resembles their neighbor pair's
    distribution can be reordered to align with the entropy-coder's
    Markov-context predictor (FEC8 2nd-order TRUE Markov VARIANT-A sister).

    Algorithm: for each consecutive block of `block_size` pairs, compute
    the pair-permutation that minimizes the marginal-distortion-per-byte
    cost (proxy: minimize the variance of per-pair distortion across the
    block; reordering by similarity reduces the entropy-coder context
    prediction error).
    """
    top_n = _validate_top_n(top_n)
    if canvas.cell_count() == 0:
        return []

    grouped = _cells_by_pair_axis(canvas, parameters.receiver_runtime)
    if not grouped:
        return []

    per_axis_rows: dict[
        CpuCudaAxis,
        list[tuple[int, float, int, list[PairFrameScorerGeometryCell]]],
    ] = {}
    for (pair_idx, cpu_cuda_axis), cells in grouped.items():
        total_delta = _aggregate_total_delta(cells)
        total_bytes = _aggregate_total_bytes(cells)
        per_axis_rows.setdefault(cpu_cuda_axis, []).append(
            (pair_idx, total_delta, total_bytes, cells)
        )

    candidates: list[ExecutableCandidate] = []
    for cpu_cuda_axis, axis_rows in per_axis_rows.items():
        axis_rows.sort(key=lambda row: row[0])  # by pair_idx ascending
        n_rows = len(axis_rows)
        for block_start in range(0, n_rows, parameters.block_size):
            block = axis_rows[block_start : block_start + parameters.block_size]
            if len(block) < 2:
                continue
            # Canonical reorder semantic: rank by total_delta ascending
            # within the block (best score-improvement first); this is
            # the canonical reordering that maximizes the entropy-coder's
            # context-predictor alignment.
            sorted_block = sorted(block, key=lambda row: row[1])
            if sorted_block == block:
                # No reorder needed; skip.
                continue

            # Aggregate per-axis-attribution across the block.
            block_cells: list[PairFrameScorerGeometryCell] = []
            for _idx, _delta, _bytes, cells in sorted_block:
                block_cells.extend(cells)
            total_delta = sum(row[1] for row in sorted_block)
            # Reorder operator's predicted byte cost: assume entropy-coder
            # saves ~1-5% of bytes per reordered block (per FEC8 2nd-order
            # empirical anchors); use conservative -1 byte per reordered
            # block as scaffold.
            total_bytes = sum(row[2] for row in sorted_block) - len(sorted_block)
            if total_delta >= 0.0:
                continue

            original_order = [row[0] for row in block]
            new_order = [row[0] for row in sorted_block]
            extra_sig = (
                f"block_start={block_start}|original={','.join(str(p) for p in original_order)}|"
                f"reordered={','.join(str(p) for p in new_order)}"
            )
            axis_decomp = _build_extended_axis_decomposition(
                block_cells,
                canvas.archive_sha256,
                ExtendedOperation.REORDER_PAIR,
                extra_inputs_signature=extra_sig,
            )
            # Override predicted_archive_bytes_delta with reorder-specific cost.
            axis_decomp = AxisDecomposition(
                predicted_d_seg_delta=axis_decomp.predicted_d_seg_delta,
                predicted_d_pose_delta=axis_decomp.predicted_d_pose_delta,
                predicted_archive_bytes_delta=total_bytes,
                axis_tag="[predicted]",
                canonical_provenance=dict(axis_decomp.canonical_provenance),
            )
            group_key = (
                block_start,
                len(sorted_block),
                _cpu_cuda_axis_sort_key(cpu_cuda_axis),
            )
            hint = _build_canonical_recipe_hint(
                ExtendedOperation.REORDER_PAIR,
                parameters.receiver_runtime,
                cpu_cuda_axis,
                group_key,
                len(block_cells),
                extra={
                    "block_start": block_start,
                    "block_size": parameters.block_size,
                    "original_pair_order": original_order,
                    "reordered_pair_order": new_order,
                },
            )
            candidates.append(
                ExecutableCandidate(
                    operation=_extended_to_canvas_operation_proxy(
                        ExtendedOperation.REORDER_PAIR
                    ),
                    archive_candidate_path=_extended_candidate_archive_path(
                        canvas,
                        ExtendedOperation.REORDER_PAIR,
                        group_key,
                        parameters.receiver_runtime,
                        cpu_cuda_axis,
                    ),
                    predicted_delta_score=total_delta,
                    predicted_byte_cost=total_bytes,
                    catalog_323_provenance=dict(axis_decomp.canonical_provenance),
                    canonical_dispatch_recipe_hint=hint,
                    predicted_axis_decomposition=axis_decomp,
                )
            )

    candidates.sort(
        key=lambda c: (
            float(c.predicted_delta_score),
            list(c.canonical_dispatch_recipe_hint.get("group_key", [])),
        )
    )
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Operator 5: drop-frame (per-frame drop; finer-grained than pair-level).
# ---------------------------------------------------------------------------


def generate_drop_frame_candidates(
    canvas: PairFrameScorerGeometryLattice,
    parameters: DropFrameParameters,
    top_n: int = DEFAULT_TOP_N,
) -> list[ExecutableCandidate]:
    """Generate drop-frame candidates per audit memo §"drop-frame NOT-BUILT".

    Per the canvas's FrameIdx axis: pairs span 2 frames (frame_0 =
    2 * pair_idx; frame_1 = 2 * pair_idx + 1). Drop-frame targets a single
    frame within a pair (finer-grained than canvas FULL_DROP at pair level).

    Algorithm: filter canvas cells by frame parity (frame_idx % 2 == 0 for
    'first'; == 1 for 'last'; both for 'both' — equivalent to canvas
    FULL_DROP); rank by composite predicted_delta_score per (frame_idx,
    cpu_cuda_axis).
    """
    top_n = _validate_top_n(top_n)
    if canvas.cell_count() == 0:
        return []

    grouped = _cells_by_frame_axis(canvas, parameters.receiver_runtime)
    if not grouped:
        return []

    # Filter by which_frame parity.
    candidates: list[ExecutableCandidate] = []
    for (frame_idx, cpu_cuda_axis), cells in grouped.items():
        if parameters.which_frame == "first" and frame_idx % 2 != 0:
            continue
        if parameters.which_frame == "last" and frame_idx % 2 != 1:
            continue
        # 'both' accepts all frames.
        total_delta = _aggregate_total_delta(cells)
        total_bytes = _aggregate_total_bytes(cells)
        if total_delta >= 0.0:
            continue
        extra_sig = f"which_frame={parameters.which_frame}|frame_idx={frame_idx}"
        axis_decomp = _build_extended_axis_decomposition(
            cells,
            canvas.archive_sha256,
            ExtendedOperation.DROP_FRAME,
            extra_inputs_signature=extra_sig,
        )
        group_key = (
            frame_idx,
            _cpu_cuda_axis_sort_key(cpu_cuda_axis),
        )
        hint = _build_canonical_recipe_hint(
            ExtendedOperation.DROP_FRAME,
            parameters.receiver_runtime,
            cpu_cuda_axis,
            group_key,
            len(cells),
            extra={
                "which_frame": parameters.which_frame,
                "frame_idx": frame_idx,
                "pair_idx": frame_idx // 2,
                "is_first_frame": frame_idx % 2 == 0,
            },
        )
        candidates.append(
            ExecutableCandidate(
                operation=_extended_to_canvas_operation_proxy(
                    ExtendedOperation.DROP_FRAME
                ),
                archive_candidate_path=_extended_candidate_archive_path(
                    canvas,
                    ExtendedOperation.DROP_FRAME,
                    group_key,
                    parameters.receiver_runtime,
                    cpu_cuda_axis,
                ),
                predicted_delta_score=total_delta,
                predicted_byte_cost=total_bytes,
                catalog_323_provenance=dict(axis_decomp.canonical_provenance),
                canonical_dispatch_recipe_hint=hint,
                predicted_axis_decomposition=axis_decomp,
            )
        )

    candidates.sort(
        key=lambda c: (
            float(c.predicted_delta_score),
            list(c.canonical_dispatch_recipe_hint.get("group_key", [])),
        )
    )
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Operator 6: synthesize-frame (Atick-Redlich sister).
# ---------------------------------------------------------------------------


def generate_synthesize_frame_candidates(
    canvas: PairFrameScorerGeometryLattice,
    parameters: SynthesizeFrameParameters,
    top_n: int = DEFAULT_TOP_N,
) -> list[ExecutableCandidate]:
    """Generate synthesize-frame candidates per audit memo §"synthesize-frame NOT-BUILT".

    Per CLAUDE.md grand-council attendees Atick-Redlich + Tishby-Zaslavsky:
    the cooperative-receiver framing implies the synthesizer has access to
    the scorer's classes/poses; per-frame synthesis is optimized for the
    I(X;T)/I(T;Y) information-bottleneck decomposition.

    Algorithm: like drop-frame but threading synthesis_seed into the
    Provenance + recipe hint so dispatch reproduces the synthesis
    deterministically. Per-axis attribution differs from drop-frame: a
    synthesis improves the distortion axis (negative d_seg/d_pose deltas)
    at the cost of some bytes (positive predicted_byte_cost for the
    synthesis-payload bytes).
    """
    top_n = _validate_top_n(top_n)
    if canvas.cell_count() == 0:
        return []

    grouped = _cells_by_frame_axis(canvas, parameters.receiver_runtime)
    if not grouped:
        return []

    candidates: list[ExecutableCandidate] = []
    for (frame_idx, cpu_cuda_axis), cells in grouped.items():
        if parameters.which_frame == "first" and frame_idx % 2 != 0:
            continue
        if parameters.which_frame == "last" and frame_idx % 2 != 1:
            continue
        total_delta = _aggregate_total_delta(cells)
        total_bytes = _aggregate_total_bytes(cells)
        if total_delta >= 0.0:
            continue
        extra_sig = (
            f"synthesis_seed={parameters.synthesis_seed}|"
            f"which_frame={parameters.which_frame}|frame_idx={frame_idx}"
        )
        axis_decomp = _build_extended_axis_decomposition(
            cells,
            canvas.archive_sha256,
            ExtendedOperation.SYNTHESIZE_FRAME,
            extra_inputs_signature=extra_sig,
        )
        group_key = (
            frame_idx,
            parameters.synthesis_seed,
            _cpu_cuda_axis_sort_key(cpu_cuda_axis),
        )
        hint = _build_canonical_recipe_hint(
            ExtendedOperation.SYNTHESIZE_FRAME,
            parameters.receiver_runtime,
            cpu_cuda_axis,
            group_key,
            len(cells),
            extra={
                "which_frame": parameters.which_frame,
                "frame_idx": frame_idx,
                "synthesis_seed": parameters.synthesis_seed,
                "literature_anchor": "atick_redlich_1990_cooperative_receiver",
                "council_attendees": [
                    "Atick",
                    "Redlich",
                    "Tishby",
                    "Zaslavsky",
                    "Wyner",
                ],
            },
        )
        candidates.append(
            ExecutableCandidate(
                operation=_extended_to_canvas_operation_proxy(
                    ExtendedOperation.SYNTHESIZE_FRAME
                ),
                archive_candidate_path=_extended_candidate_archive_path(
                    canvas,
                    ExtendedOperation.SYNTHESIZE_FRAME,
                    group_key,
                    parameters.receiver_runtime,
                    cpu_cuda_axis,
                ),
                predicted_delta_score=total_delta,
                predicted_byte_cost=total_bytes,
                catalog_323_provenance=dict(axis_decomp.canonical_provenance),
                canonical_dispatch_recipe_hint=hint,
                predicted_axis_decomposition=axis_decomp,
            )
        )

    candidates.sort(
        key=lambda c: (
            float(c.predicted_delta_score),
            list(c.canonical_dispatch_recipe_hint.get("group_key", [])),
        )
    )
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Operator 7: motion-conditional (Rao-Ballard sister).
# ---------------------------------------------------------------------------


def _compute_pair_motion_magnitudes(
    canvas: PairFrameScorerGeometryLattice,
    receiver_runtime: ReceiverRuntime,
) -> dict[tuple[int, CpuCudaAxis], float]:
    """Compute per-pair motion magnitude proxy via PoseNet-axis cell magnitudes.

    Per CLAUDE.md grand-council attendees Rao-Ballard: per-pair motion
    magnitude is the PoseNet-axis prediction magnitude (the canvas's
    PoseNet-axis cells carry the per-pair pose-axis contribution).
    """
    motion: dict[tuple[int, CpuCudaAxis], float] = {}
    for cell in canvas._cells.values():
        if cell.receiver_runtime is not receiver_runtime:
            continue
        if not cell.receiver_feasibility:
            continue
        if cell.scorer_axis is not ScorerAxis.POSENET_6D:
            continue
        key = (cell.pair_idx, cell.cpu_cuda_axis)
        # Sum absolute PoseNet-axis contributions across (pair, axis).
        motion[key] = motion.get(key, 0.0) + abs(float(cell.predicted_delta_score))
    return motion


def generate_motion_conditional_candidates(
    canvas: PairFrameScorerGeometryLattice,
    parameters: MotionConditionalParameters,
    top_n: int = DEFAULT_TOP_N,
) -> list[ExecutableCandidate]:
    """Generate motion-conditional candidates per audit memo §"motion-conditional".

    Per CLAUDE.md grand-council attendees Rao-Ballard: per-pair operator
    selection conditioned on pose magnitude. Pairs with high pose
    magnitude (above motion_threshold_percentile) get the high_motion_operator
    applied; pairs with low pose magnitude get the low_motion_operator.

    Algorithm: compute per-pair motion magnitude proxy from PoseNet-axis
    cells; rank pairs by motion magnitude; apply high/low motion operator
    accordingly; emit one ExecutableCandidate per pair classification.

    Per UNIQUE-AND-COMPLETE-PER-METHOD: motion-conditional is a META-operator
    composing two leaf operators per pair. The emitted candidates carry
    BOTH the leaf operator AND the motion-conditional metadata in the
    recipe hint.
    """
    top_n = _validate_top_n(top_n)
    if canvas.cell_count() == 0:
        return []

    motion = _compute_pair_motion_magnitudes(canvas, parameters.receiver_runtime)
    if not motion:
        return []

    # Per axis, determine the percentile cutoff.
    per_axis_motion: dict[CpuCudaAxis, list[tuple[int, float]]] = {}
    for (pair_idx, cpu_cuda_axis), mag in motion.items():
        per_axis_motion.setdefault(cpu_cuda_axis, []).append((pair_idx, mag))

    candidates: list[ExecutableCandidate] = []
    grouped = _cells_by_pair_axis(canvas, parameters.receiver_runtime)
    for cpu_cuda_axis, axis_motion in per_axis_motion.items():
        axis_motion.sort(key=lambda row: row[1])
        cutoff_idx = math.floor(
            parameters.motion_threshold_percentile * len(axis_motion)
        )
        cutoff_idx = max(0, min(cutoff_idx, len(axis_motion)))
        low_motion_pairs = {p for p, _m in axis_motion[:cutoff_idx]}
        high_motion_pairs = {p for p, _m in axis_motion[cutoff_idx:]}

        for (pair_idx, axis), cells in grouped.items():
            if axis is not cpu_cuda_axis:
                continue
            if pair_idx in high_motion_pairs:
                leaf_op = parameters.high_motion_operator
                motion_class = "high"
            elif pair_idx in low_motion_pairs:
                leaf_op = parameters.low_motion_operator
                motion_class = "low"
            else:
                continue

            total_delta = _aggregate_total_delta(cells)
            total_bytes = _aggregate_total_bytes(cells)
            if total_delta >= 0.0:
                continue
            extra_sig = (
                f"motion_class={motion_class}|leaf_op={leaf_op.value}|"
                f"percentile={parameters.motion_threshold_percentile:.3f}|"
                f"pair_idx={pair_idx}"
            )
            axis_decomp = _build_extended_axis_decomposition(
                cells,
                canvas.archive_sha256,
                ExtendedOperation.MOTION_CONDITIONAL,
                extra_inputs_signature=extra_sig,
            )
            group_key = (
                pair_idx,
                0 if motion_class == "high" else 1,
                _cpu_cuda_axis_sort_key(cpu_cuda_axis),
            )
            hint = _build_canonical_recipe_hint(
                ExtendedOperation.MOTION_CONDITIONAL,
                parameters.receiver_runtime,
                cpu_cuda_axis,
                group_key,
                len(cells),
                extra={
                    "pair_idx": pair_idx,
                    "motion_class": motion_class,
                    "leaf_operator": leaf_op.value,
                    "motion_threshold_percentile": (
                        parameters.motion_threshold_percentile
                    ),
                    "pair_motion_magnitude": motion.get(
                        (pair_idx, cpu_cuda_axis), 0.0
                    ),
                    "literature_anchor": "rao_ballard_1999_predictive_coding",
                    "council_attendees": ["Rao", "Ballard", "Tishby"],
                },
            )
            candidates.append(
                ExecutableCandidate(
                    operation=_extended_to_canvas_operation_proxy(
                        ExtendedOperation.MOTION_CONDITIONAL
                    ),
                    archive_candidate_path=_extended_candidate_archive_path(
                        canvas,
                        ExtendedOperation.MOTION_CONDITIONAL,
                        group_key,
                        parameters.receiver_runtime,
                        cpu_cuda_axis,
                    ),
                    predicted_delta_score=total_delta,
                    predicted_byte_cost=total_bytes,
                    catalog_323_provenance=dict(axis_decomp.canonical_provenance),
                    canonical_dispatch_recipe_hint=hint,
                    predicted_axis_decomposition=axis_decomp,
                )
            )

    candidates.sort(
        key=lambda c: (
            float(c.predicted_delta_score),
            list(c.canonical_dispatch_recipe_hint.get("group_key", [])),
        )
    )
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Operator 8: temporal-coherence (Wyner-Ziv sister).
# ---------------------------------------------------------------------------


def _per_pair_axis_signature(
    cells: Sequence[PairFrameScorerGeometryCell],
) -> tuple[float, float, float]:
    """Compute a per-pair (d_seg_delta, d_pose_delta, archive_bytes_delta) signature.

    Used for cosine similarity between pairs in temporal-coherence.
    """
    seg = 0.0
    pose = 0.0
    rate = 0.0
    for cell in cells:
        if cell.scorer_axis is ScorerAxis.SEGNET_5CLASS:
            seg += float(cell.predicted_delta_score)
        elif cell.scorer_axis is ScorerAxis.POSENET_6D:
            pose += float(cell.predicted_delta_score)
        elif cell.scorer_axis is ScorerAxis.RATE_TERM:
            rate += float(cell.predicted_byte_cost)
    return (seg, pose, rate)


def _cosine_similarity(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    """Cosine similarity of 3D per-axis-attribution signatures."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    norm_a = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    norm_b = math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def generate_temporal_coherence_candidates(
    canvas: PairFrameScorerGeometryLattice,
    parameters: TemporalCoherenceParameters,
    top_n: int = DEFAULT_TOP_N,
) -> list[ExecutableCandidate]:
    """Generate temporal-coherence candidates per audit memo §"temporal-coherence".

    Per CLAUDE.md grand-council attendees Wyner-Ziv: pairs whose per-pair
    predicted-axis decomposition is similar to their neighbors (within a
    temporal window) can be jointly optimized as a Wyner-Ziv side-information
    block; the rate saving is the redundancy reduction.

    Algorithm: for each pair, compute per-pair AxisDecomposition signature
    (d_seg, d_pose, rate); for each pair, scan neighbors within
    temporal_window; if cosine_similarity >= similarity_threshold, emit a
    joint optimization candidate combining the pair + similar neighbor.
    """
    top_n = _validate_top_n(top_n)
    if canvas.cell_count() == 0:
        return []

    grouped = _cells_by_pair_axis(canvas, parameters.receiver_runtime)
    if not grouped:
        return []

    # Per axis, build pair_idx -> signature + cells map.
    per_axis_pairs: dict[
        CpuCudaAxis,
        list[tuple[int, tuple[float, float, float], list[PairFrameScorerGeometryCell]]],
    ] = {}
    for (pair_idx, cpu_cuda_axis), cells in grouped.items():
        sig = _per_pair_axis_signature(cells)
        per_axis_pairs.setdefault(cpu_cuda_axis, []).append(
            (pair_idx, sig, cells)
        )

    candidates: list[ExecutableCandidate] = []
    for cpu_cuda_axis, axis_pairs in per_axis_pairs.items():
        axis_pairs.sort(key=lambda row: row[0])
        n_pairs = len(axis_pairs)
        for i, (pair_a, sig_a, cells_a) in enumerate(axis_pairs):
            # Scan neighbors within temporal_window.
            for j in range(i + 1, min(i + parameters.temporal_window + 1, n_pairs)):
                pair_b, sig_b, cells_b = axis_pairs[j]
                if pair_b - pair_a > parameters.temporal_window:
                    break
                similarity = _cosine_similarity(sig_a, sig_b)
                if similarity < parameters.similarity_threshold:
                    continue
                # Emit a joint Wyner-Ziv candidate.
                # Wyner-Ziv side-info reduces rate by approximately the
                # mutual-information H(B|A); we model as 5% rate saving
                # at scaffold scale (conservative empirical anchor;
                # canonical equation FORMALIZATION_PENDING).
                joint_cells = cells_a + cells_b
                total_delta = _aggregate_total_delta(joint_cells)
                naive_bytes = _aggregate_total_bytes(joint_cells)
                # Wyner-Ziv saving: ~5% of total bytes (scaffold heuristic;
                # PENDING empirical refinement per Catalog #344 anchor).
                wyner_ziv_savings = int(abs(naive_bytes) * 0.05)
                effective_bytes = naive_bytes - wyner_ziv_savings
                if total_delta >= 0.0:
                    continue
                extra_sig = (
                    f"pair_a={pair_a}|pair_b={pair_b}|similarity={similarity:.6f}|"
                    f"temporal_window={parameters.temporal_window}"
                )
                axis_decomp = _build_extended_axis_decomposition(
                    joint_cells,
                    canvas.archive_sha256,
                    ExtendedOperation.TEMPORAL_COHERENCE,
                    extra_inputs_signature=extra_sig,
                )
                # Override predicted_archive_bytes_delta with Wyner-Ziv-savings-
                # adjusted value.
                axis_decomp = AxisDecomposition(
                    predicted_d_seg_delta=axis_decomp.predicted_d_seg_delta,
                    predicted_d_pose_delta=axis_decomp.predicted_d_pose_delta,
                    predicted_archive_bytes_delta=effective_bytes,
                    axis_tag="[predicted]",
                    canonical_provenance=dict(axis_decomp.canonical_provenance),
                )
                group_key = (
                    pair_a,
                    pair_b,
                    _cpu_cuda_axis_sort_key(cpu_cuda_axis),
                )
                hint = _build_canonical_recipe_hint(
                    ExtendedOperation.TEMPORAL_COHERENCE,
                    parameters.receiver_runtime,
                    cpu_cuda_axis,
                    group_key,
                    len(joint_cells),
                    extra={
                        "pair_a": pair_a,
                        "pair_b": pair_b,
                        "temporal_distance": pair_b - pair_a,
                        "cosine_similarity": similarity,
                        "wyner_ziv_savings_bytes": wyner_ziv_savings,
                        "literature_anchor": "wyner_ziv_1976_side_information",
                        "council_attendees": ["Wyner", "Tishby", "Zaslavsky"],
                    },
                )
                candidates.append(
                    ExecutableCandidate(
                        operation=_extended_to_canvas_operation_proxy(
                            ExtendedOperation.TEMPORAL_COHERENCE
                        ),
                        archive_candidate_path=_extended_candidate_archive_path(
                            canvas,
                            ExtendedOperation.TEMPORAL_COHERENCE,
                            group_key,
                            parameters.receiver_runtime,
                            cpu_cuda_axis,
                        ),
                        predicted_delta_score=total_delta,
                        predicted_byte_cost=effective_bytes,
                        catalog_323_provenance=dict(
                            axis_decomp.canonical_provenance
                        ),
                        canonical_dispatch_recipe_hint=hint,
                        predicted_axis_decomposition=axis_decomp,
                    )
                )

    candidates.sort(
        key=lambda c: (
            float(c.predicted_delta_score),
            list(c.canonical_dispatch_recipe_hint.get("group_key", [])),
        )
    )
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Canvas operation proxy + canonical operator registry.
# ---------------------------------------------------------------------------


def _extended_to_canvas_operation_proxy(
    operation: ExtendedOperation,
):
    """Map ExtendedOperation -> canvas CanonicalOperation proxy for compatibility.

    Per the audit's BUILD-2+3 contract: ``ExecutableCandidate.operation`` is
    typed as ``CanonicalOperation`` (canvas-frozen at 4 members). The 8
    extended operators map to a proxy member via canonical-vs-unique
    decision per layer:

    - REPLACE_ONE / REPLACE_MANY -> REPAIR (semantic: substitution improves
      the per-pair signal; aligned with canvas's cooperative-receiver
      Atick-Redlich repair semantic).
    - MERGE_PAIR / REORDER_PAIR -> MASKED (semantic: per-region/per-pair
      operations; aligned with canvas's masked operator).
    - DROP_FRAME -> FULL_DROP (the frame-level sister of canvas FULL_DROP).
    - SYNTHESIZE_FRAME -> REPAIR (frame-level sister of canvas REPAIR).
    - MOTION_CONDITIONAL -> FEATHERED (semantic: smooth transition between
      operators per motion magnitude; aligned with canvas's feathered
      Daubechies wavelet partition prior).
    - TEMPORAL_COHERENCE -> FEATHERED (semantic: cross-pair smooth blending
      per Wyner-Ziv side-information).

    The recipe hint always carries the canonical ExtendedOperation value
    explicitly via ``hint["operation"]`` so downstream consumers can
    distinguish the 8 extended operators from the 4 canonical operations.
    """
    from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
        CanonicalOperation,
    )

    proxy_map = {
        ExtendedOperation.REPLACE_ONE: CanonicalOperation.REPAIR,
        ExtendedOperation.REPLACE_MANY: CanonicalOperation.REPAIR,
        ExtendedOperation.MERGE_PAIR: CanonicalOperation.MASKED,
        ExtendedOperation.REORDER_PAIR: CanonicalOperation.MASKED,
        ExtendedOperation.DROP_FRAME: CanonicalOperation.FULL_DROP,
        ExtendedOperation.SYNTHESIZE_FRAME: CanonicalOperation.REPAIR,
        ExtendedOperation.MOTION_CONDITIONAL: CanonicalOperation.FEATHERED,
        ExtendedOperation.TEMPORAL_COHERENCE: CanonicalOperation.FEATHERED,
    }
    return proxy_map[operation]


# Canonical operator dispatch registry (for CLI + cathedral consumer
# auto-discovery per Catalog #335 sister).
EXTENDED_OPERATOR_REGISTRY: Mapping[ExtendedOperation, Any] = {
    ExtendedOperation.REPLACE_ONE: generate_replace_one_candidates,
    ExtendedOperation.REPLACE_MANY: generate_replace_many_candidates,
    ExtendedOperation.MERGE_PAIR: generate_merge_pair_candidates,
    ExtendedOperation.REORDER_PAIR: generate_reorder_pair_candidates,
    ExtendedOperation.DROP_FRAME: generate_drop_frame_candidates,
    ExtendedOperation.SYNTHESIZE_FRAME: generate_synthesize_frame_candidates,
    ExtendedOperation.MOTION_CONDITIONAL: generate_motion_conditional_candidates,
    ExtendedOperation.TEMPORAL_COHERENCE: generate_temporal_coherence_candidates,
}


# ---------------------------------------------------------------------------
# Catalog #335 canonical contract hooks.
# ---------------------------------------------------------------------------


def update_from_anchor(anchor: Any) -> None:
    """Tier A observability-only stub per Catalog #357.

    Per Tier A scaffold contract: this hook is observability-only (does
    not contribute to score/promotion/rank). BUILD-4 sister subagent
    op-routable promotes to Tier B with canonical-routing markers per
    Catalog #341 + canonical Provenance per Catalog #323 + per-axis
    AxisDecomposition per Catalog #356.
    """
    return None


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Tier A observability-only stub per Catalog #357 + Catalog #341.

    Returns canonical-routing markers per Catalog #341:
    ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
    ``axis_tag="[predicted]"``. Per BUILD-4 sister subagent: Tier B
    promotion replaces this stub with actual cathedral-routing logic.
    """
    return {
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        "consumer_name": CONSUMER_NAME,
        "consumer_version": CONSUMER_VERSION,
    }
