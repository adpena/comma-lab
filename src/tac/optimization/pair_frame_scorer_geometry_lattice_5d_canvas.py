# SPDX-License-Identifier: MIT
"""Pair-frame scorer-geometry 5D canvas — canonical binding scaffold.

Per PAIR-FRAME-SCORER-GEOMETRY-LATTICE design memo
(`.omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md`)
+ codex eureka residual-gap call-out
(`codex_findings_eureka_drop_many_rate_distortion_budget_20260525T143351Z_codex.md`
§"Residual Gap") verbatim:

> "bind pair component xray rows, frame-axis master-gradient decomposition,
> SegNet/PoseNet score geometry, CPU/CUDA axis labels, and receiver feasibility
> into a pair-frame scorer-geometry lattice that can generate queue-executable
> full-drop, repair, masked, and feathered starts without false authority"

This module is the **SCAFFOLD SKELETON** of the canonical 5D canvas binding.
Sibling of the existing codex v1 row-based module
``tac.optimization.pair_frame_scorer_geometry_lattice`` (commit `4ed9eb905`)
which is DQS1-acquisition-row-bound. The two modules are complementary:

- **codex v1 row-based** (existing): `pair_frame_scorer_geometry_lattice.py` —
  row-based pair-frame geometry binding for DQS1 acquisition + queue
  executable drop requests. Pair-only axis exposure.
- **5D canvas** (this scaffold): `pair_frame_scorer_geometry_lattice_5d_canvas.py` —
  next-level CANVAS binding the 5 axes codex named (pair × frame ×
  scorer_axis × receiver_runtime × cpu_cuda_axis). The 4 canonical
  operations (full-drop / repair / masked / feathered) operate on this
  canvas.

The empirical population + executable operation generators are sister-
subagent deliverables BUILD-1 through BUILD-4 per design memo DELIVERABLE 3.
BUILD-1 explicitly composes with the codex v1 row-based reader to populate
empirical cells.

## Canonical-vs-unique decision per layer

- AxisDecomposition: ADOPT canonical
  ``tac.cathedral.consumer_contract.AxisDecomposition`` (Catalog #356).
- HookNumber + ConsumerTier: ADOPT canonical
  ``tac.cathedral.consumer_contract`` (Catalog #335 + #357).
- Score composition: ADOPT canonical
  ``tac.score_composition.compose_score_from_axes`` (Catalog #356).
- Provenance: ADOPT canonical Provenance per Catalog #323; threaded through
  every cell's ``catalog_323_provenance`` field.
- Codex v1 row-based reader: ADOPT (BUILD-1 sister subagent reads codex v1
  output to populate empirical cells; no fork).
- 5D canvas schema: GENUINELY NEW. No existing canonical helper binds
  (pair × frame × scorer_axis × receiver_runtime × cpu_cuda_axis); the bind
  contract IS the design contribution per the design memo.

## Observability surface (Catalog #305)

Per design memo §"Observability surface" — each lattice axis queryable
independently via canonical primitives; per-cell measurement fields
decomposable; lattice deterministic from archive sha256; per-cell
``as_dict`` JSON serialization round-trip; cite-able via
``catalog_323_provenance``; counterfactual-able via per-cell
``query_receiver_runtime_feasibility``.

## 9-dimension success checklist evidence

Per design memo §"9-dimension success checklist evidence" — UNIQUENESS
(first canonical primitive binding the 5 axes codex named at canvas
granularity) + BEAUTY (ONE frozen dataclass + ONE container + 4 operation
generators; ~300-400 LOC budget) + DISTINCTNESS (canvas vs codex v1
row-based vs sister DROP-MANY-BEAM-DESIGN) + RIGOR (grounded in 7
canonical mathematical primitives) + OPTIMIZATION-PER-TECHNIQUE
(per-receiver-runtime substrate-optimal engineering) + STACK-OF-STACKS-
COMPOSABILITY (canvas IS the composability primitive) + DETERMINISTIC-
REPRODUCIBILITY (per-cell deterministic from archive sha) + EXTREME-
OPTIMIZATION-PERFORMANCE (sparse representation bounded) + OPTIMAL-
MINIMAL-CONTEST-SCORE (unblocks 4 simultaneous operator-explicit requests).

## Cargo-cult audit per assumption

Per design memo §"Cargo-cult audit per assumption" — 3 HARD-EARNED
(pairset_component_marginal + per_pair_master_gradient_taylor + Catalog
#356 AxisDecomposition) + 3 CARGO-CULTED-PENDING-EMPIRICAL (5-mode
receiver_runtime enum + 5D coordinate granularity + static lattice
assumption).

## Predicted ΔS band

NOT a substrate dispatch proposal. Per Catalog #296: no Dykstra
feasibility check needed for a canonical primitive scaffold. The canvas
EMITS per-cell ΔS predictions; the canvas ITSELF carries no aggregate
prediction.

## Council attendees / verdict

T1 working-group VERDICT PROCEED (design memo + scaffold skeleton; no
quorum required at T1 per Catalog #300). Attendees: Shannon LEAD + Dykstra
CO-LEAD + Daubechies CO-LEAD + Rudin CO-LEAD + Atick + Carmack + Assumption-
Adversary. Sister subagent BUILD-1 picks up the empirical 5D canvas
population.

## Scope of this skeleton

The skeleton exports the canonical interface contract + frozen dataclasses
+ container class structure. The empirical canvas population, the 4
operation generators, and the canonical CLI are sister-subagent
deliverables per design memo DELIVERABLE 3 (BUILD-1 through BUILD-4).

Until BUILD-1 lands, the container's ``build_lattice`` raises
``NotImplementedError`` with the canonical sentinel referencing BUILD-1
sister subagent op-routable; the 4 operation generators raise the same
sentinel.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the
``NotImplementedError`` is DEFER-pending-BUILD-1, NOT a paradigm refusal.

## Tier classification

Per Catalog #357: ``CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY``
at scaffold landing. BUILD-4 sister subagent op-routable promotes to
``ConsumerTier.TIER_B_SCORE_CONTRIBUTING`` with canonical-routing markers
per Catalog #341 + canonical Provenance per Catalog #323 + per-axis
AxisDecomposition per Catalog #356.

The 6-hook wire-in declaration per Catalog #125 (all 6 ACTIVE at BUILD-4
Tier B promotion):

- hook #1 sensitivity-map: per-cell bind_pair_component_xray +
  decompose_frame_axis_master_gradient are sensitivity-map producers
- hook #2 Pareto constraint: per-cell query_receiver_runtime_feasibility
  IS Dykstra alternating-projection feasibility
- hook #3 bit-allocator: per-cell predicted_byte_cost IS the bit-allocator
  primary signal
- hook #4 cathedral autopilot dispatch: BUILD-4 Tier B promotion auto-
  discovers per Catalog #335
- hook #5 continual-learning posterior: per-cell measurement updates flow
  into canonical posterior per Catalog #344
- hook #6 probe-disambiguator: the 5D canvas IS the canonical
  disambiguator across operations + receiver_runtime modes + CPU/CUDA axes

Sister cross-references:

- ``tac.optimization.pair_frame_scorer_geometry_lattice`` (codex v1
  row-based sibling; BUILD-1 composes with this reader)
- ``tac.optimization.decoder_q_pairset_acquisition`` (sister DQS1
  acquisition; canvas-vs-algorithm distinction)
- ``tac.master_gradient_comparison.multi_granularity``
  (sister Cable D master-gradient producers)
- ``tac.cathedral_consumers.per_pair_gradient_clustering_consumer``
  (sister consumer interface pattern)
- ``tac.cathedral_consumers.per_frame_sensitivity_consumer``
  (sister per-frame consumer)
- Canonical equation #36 ``pairset_component_marginal_score_decomposition_v1``
  (HARD-EARNED sister)
- Canonical equation #18 ``pose_axis_cuda_amplification_v1``
  (HARD-EARNED sister for cpu_cuda_axis_amplification factor)

Lane: ``lane_pair_frame_scorer_geometry_lattice_design_memo_20260525`` L1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from tac.cathedral.consumer_contract import (
    AxisDecomposition,
    ConsumerTier,
    HookNumber,
)

# ---------------------------------------------------------------------------
# Module-level canonical contract per Catalog #335 + #357.
# ---------------------------------------------------------------------------

CONSUMER_NAME = "pair_frame_scorer_geometry_lattice_5d_canvas"
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

CANVAS_SCHEMA = "pair_frame_scorer_geometry_lattice_5d_canvas.v0_scaffold"
CELL_SCHEMA = "pair_frame_scorer_geometry_cell.v0_scaffold"


# ---------------------------------------------------------------------------
# Canonical 5D lattice axes.
# ---------------------------------------------------------------------------


class ScorerAxis(StrEnum):
    """Canonical 3-axis contest decomposition per CLAUDE.md formula.

    S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37_545_489
    """

    SEGNET_5CLASS = "segnet_5class"
    POSENET_6D = "posenet_6d"
    RATE_TERM = "rate_term"


class ReceiverRuntime(StrEnum):
    """Canonical 5 receiver-runtime modes.

    Per design memo cargo-cult audit: the 5-mode enum is CARGO-CULTED-
    PENDING-EMPIRICAL; sister subagent may register alternative receiver
    modes via Catalog #335 auto-discovery extension.

    - RAW_RESIDUAL: identity / no compensation
    - SMOOTHED_RESIDUAL: low-pass filter compensation
    - MASKED: per-region SegNet-class-aware byte mask (UNIWARD/HILL sister)
    - FEATHERED: smooth-transition Daubechies wavelet partition prior
    - FULL_DROP: byte-removal at archive grammar boundary
    """

    RAW_RESIDUAL = "raw_residual"
    SMOOTHED_RESIDUAL = "smoothed_residual"
    MASKED = "masked"
    FEATHERED = "feathered"
    FULL_DROP = "full_drop"


class CpuCudaAxis(StrEnum):
    """Canonical 1:1 hardware-compliant axes per CLAUDE.md "Submission auth eval".

    - CONTEST_CPU: Linux x86_64 (GHA ubuntu-latest runner)
    - CONTEST_CUDA_T4: NVIDIA T4 / A100 / 4090 / equivalent
    """

    CONTEST_CPU = "contest_cpu"
    CONTEST_CUDA_T4 = "contest_cuda_t4"


class CanonicalOperation(StrEnum):
    """The 4 canonical operations the canvas unblocks per design memo §DELIVERABLE 1.

    - FULL_DROP: drop entire pair OR frame; rate saving = byte_cost; scorer
      penalty depends on receiver_runtime compensation
    - REPAIR: drop + add per-pair/per-frame repair signal (Atick-Redlich
      cooperative-receiver paradigm)
    - MASKED: per-region SegNet-class-aware byte mask (UNIWARD region weighting)
    - FEATHERED: smooth-transition variant of masked (Daubechies multi-scale
      wavelet partition prior)
    """

    FULL_DROP = "full_drop"
    REPAIR = "repair"
    MASKED = "masked"
    FEATHERED = "feathered"


# ---------------------------------------------------------------------------
# Canonical canvas constants.
# ---------------------------------------------------------------------------

CANONICAL_PAIR_COUNT = 600
"""Per CLAUDE.md canonical contest formula: 600 non-overlapping pairs
(seq_len=2 per upstream/evaluate.py) across the canonical 1200-frame
contest video ``upstream/videos/0.mkv``."""

CANONICAL_FRAME_COUNT = 1200
"""Canonical contest frame count per upstream/videos/0.mkv."""


# ---------------------------------------------------------------------------
# Per-cell frozen dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairFrameScorerGeometryCell:
    """Single cell in the canonical 5D canvas.

    Per design memo §DELIVERABLE 1 mathematical formulation:

        L: PairIdx × FrameIdx × ScorerAxis × ReceiverRuntime × CpuCudaAxis
             -> MeasurementTuple

    Each cell binds the 5D coordinate to a 4-tuple measurement: predicted
    score delta, predicted byte cost, receiver feasibility, and canonical
    Provenance per Catalog #323.

    Per Catalog #356 + Catalog #357 + Catalog #323: every measurement
    field is observability-only at Tier A; BUILD-4 sister subagent
    promotes to Tier B with canonical-routing markers per Catalog #341.

    Args:
        pair_idx: 0-indexed pair index in [0, CANONICAL_PAIR_COUNT).
        frame_idx: 0-indexed frame index in [0, CANONICAL_FRAME_COUNT).
            For pair-only operations, derived as ``2 * pair_idx`` (first
            frame of pair) or ``2 * pair_idx + 1`` (last frame of pair).
        scorer_axis: canonical 3-axis decomposition member.
        receiver_runtime: canonical 5-mode receiver member.
        cpu_cuda_axis: canonical 1:1 hardware-compliant axis member.
        predicted_delta_score: per-cell ΔS prediction composed via canonical
            contest formula; signed (negative = score improvement).
        predicted_byte_cost: per-cell archive byte cost; signed (negative
            = bytes removed).
        receiver_feasibility: per-cell receiver-runtime feasibility (True
            if the receiver runtime is structurally compatible with the
            scorer axis's signal preservation requirements per Dykstra
            alternating-projection sister analysis).
        catalog_323_provenance: dict-form Provenance per Catalog #323;
            produced via ``tac.provenance.builders.build_provenance_for_predicted``
            (or sister builder) and ``tac.provenance.validator.provenance_to_dict``.
    """

    pair_idx: int
    frame_idx: int
    scorer_axis: ScorerAxis
    receiver_runtime: ReceiverRuntime
    cpu_cuda_axis: CpuCudaAxis
    predicted_delta_score: float
    predicted_byte_cost: int
    receiver_feasibility: bool
    catalog_323_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pair_idx, int) or isinstance(self.pair_idx, bool):
            raise ValueError(
                f"pair_idx must be int, got {type(self.pair_idx).__name__}"
            )
        if not 0 <= self.pair_idx < CANONICAL_PAIR_COUNT:
            raise ValueError(
                f"pair_idx {self.pair_idx} out of range "
                f"[0, {CANONICAL_PAIR_COUNT})"
            )
        if not isinstance(self.frame_idx, int) or isinstance(self.frame_idx, bool):
            raise ValueError(
                f"frame_idx must be int, got {type(self.frame_idx).__name__}"
            )
        if not 0 <= self.frame_idx < CANONICAL_FRAME_COUNT:
            raise ValueError(
                f"frame_idx {self.frame_idx} out of range "
                f"[0, {CANONICAL_FRAME_COUNT})"
            )
        if not isinstance(self.scorer_axis, ScorerAxis):
            raise ValueError(
                f"scorer_axis must be ScorerAxis, got "
                f"{type(self.scorer_axis).__name__}"
            )
        if not isinstance(self.receiver_runtime, ReceiverRuntime):
            raise ValueError(
                f"receiver_runtime must be ReceiverRuntime, got "
                f"{type(self.receiver_runtime).__name__}"
            )
        if not isinstance(self.cpu_cuda_axis, CpuCudaAxis):
            raise ValueError(
                f"cpu_cuda_axis must be CpuCudaAxis, got "
                f"{type(self.cpu_cuda_axis).__name__}"
            )
        if not isinstance(self.predicted_delta_score, (int, float)):
            raise ValueError(
                f"predicted_delta_score must be numeric, got "
                f"{type(self.predicted_delta_score).__name__}"
            )
        object.__setattr__(
            self, "predicted_delta_score", float(self.predicted_delta_score)
        )
        if not isinstance(self.predicted_byte_cost, int) or isinstance(
            self.predicted_byte_cost, bool
        ):
            raise ValueError(
                f"predicted_byte_cost must be int (signed), got "
                f"{type(self.predicted_byte_cost).__name__}"
            )
        if not isinstance(self.receiver_feasibility, bool):
            raise ValueError(
                f"receiver_feasibility must be bool, got "
                f"{type(self.receiver_feasibility).__name__}"
            )
        if not isinstance(self.catalog_323_provenance, Mapping):
            raise ValueError(
                "catalog_323_provenance must be a Mapping "
                f"(Catalog #323 dict-form Provenance), got "
                f"{type(self.catalog_323_provenance).__name__}"
            )

    @property
    def coordinate(self) -> tuple[int, int, ScorerAxis, ReceiverRuntime, CpuCudaAxis]:
        """Canonical 5D coordinate tuple for canvas keying."""
        return (
            self.pair_idx,
            self.frame_idx,
            self.scorer_axis,
            self.receiver_runtime,
            self.cpu_cuda_axis,
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization (byte-stable via sort_keys=True at writer)."""
        return {
            "schema": CELL_SCHEMA,
            "pair_idx": int(self.pair_idx),
            "frame_idx": int(self.frame_idx),
            "scorer_axis": self.scorer_axis.value,
            "receiver_runtime": self.receiver_runtime.value,
            "cpu_cuda_axis": self.cpu_cuda_axis.value,
            "predicted_delta_score": float(self.predicted_delta_score),
            "predicted_byte_cost": int(self.predicted_byte_cost),
            "receiver_feasibility": bool(self.receiver_feasibility),
            "catalog_323_provenance": dict(self.catalog_323_provenance),
        }


# ---------------------------------------------------------------------------
# Executable candidate emitted by the 4 operation generators.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutableCandidate:
    """Canonical archive candidate emitted by a canvas operation generator.

    Per design memo §DELIVERABLE 1 canonical primitive operations:

        generate_queue_executable_start(...) -> ExecutableCandidate

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
    every ExecutableCandidate carries archive_candidate_path + predicted
    measurements + canonical Provenance + canonical-routing markers per
    Catalog #341 (defaults to non-promotable per Tier A scaffold; BUILD-4
    Tier B promotion sets ``promotable=False`` per Tier B contract +
    empirically-grounded ``axis_tag``).

    Args:
        operation: which of the 4 canonical operations produced this
            candidate (FULL_DROP / REPAIR / MASKED / FEATHERED).
        archive_candidate_path: path to the executable archive bytes
            (or scaffold-only sentinel until BUILD-2 lands).
        predicted_delta_score: composed via canonical contest formula
            from per-cell deltas across the affected (pair, frame)
            coordinates.
        predicted_byte_cost: archive byte delta (signed; negative =
            archive smaller).
        catalog_323_provenance: dict-form Provenance per Catalog #323.
        canonical_routing_markers: dict carrying Tier A canonical markers
            per Catalog #341 (default ``predicted_delta_adjustment=0.0`` +
            ``promotable=False`` + ``axis_tag="[predicted]"``).
        canonical_dispatch_recipe_hint: operator-routable hint for the
            DISPATCH wave (e.g. estimated cost + Modal recipe identifier +
            sister dispatch claim TTL).
        predicted_axis_decomposition: optional AxisDecomposition per
            Catalog #356 (BUILD-3 sister subagent populates).
    """

    operation: CanonicalOperation
    archive_candidate_path: Path
    predicted_delta_score: float
    predicted_byte_cost: int
    catalog_323_provenance: Mapping[str, Any] = field(default_factory=dict)
    canonical_routing_markers: Mapping[str, Any] = field(
        default_factory=lambda: {
            "predicted_delta_adjustment": 0.0,
            "promotable": False,
            "axis_tag": "[predicted]",
        }
    )
    canonical_dispatch_recipe_hint: Mapping[str, Any] = field(default_factory=dict)
    predicted_axis_decomposition: AxisDecomposition | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, CanonicalOperation):
            raise ValueError(
                f"operation must be CanonicalOperation, got "
                f"{type(self.operation).__name__}"
            )
        if not isinstance(self.archive_candidate_path, Path):
            raise ValueError(
                f"archive_candidate_path must be Path, got "
                f"{type(self.archive_candidate_path).__name__}"
            )
        if not isinstance(self.predicted_delta_score, (int, float)):
            raise ValueError(
                f"predicted_delta_score must be numeric, got "
                f"{type(self.predicted_delta_score).__name__}"
            )
        object.__setattr__(
            self, "predicted_delta_score", float(self.predicted_delta_score)
        )
        if not isinstance(self.predicted_byte_cost, int) or isinstance(
            self.predicted_byte_cost, bool
        ):
            raise ValueError(
                f"predicted_byte_cost must be int (signed), got "
                f"{type(self.predicted_byte_cost).__name__}"
            )
        for fname, expected in (
            ("catalog_323_provenance", Mapping),
            ("canonical_routing_markers", Mapping),
            ("canonical_dispatch_recipe_hint", Mapping),
        ):
            value = getattr(self, fname)
            if not isinstance(value, expected):
                raise ValueError(
                    f"{fname} must be a Mapping, got {type(value).__name__}"
                )
        if self.predicted_axis_decomposition is not None and not isinstance(
            self.predicted_axis_decomposition, AxisDecomposition
        ):
            raise ValueError(
                "predicted_axis_decomposition must be AxisDecomposition or "
                f"None, got {type(self.predicted_axis_decomposition).__name__}"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization."""
        payload: dict[str, Any] = {
            "operation": self.operation.value,
            "archive_candidate_path": str(self.archive_candidate_path),
            "predicted_delta_score": float(self.predicted_delta_score),
            "predicted_byte_cost": int(self.predicted_byte_cost),
            "catalog_323_provenance": dict(self.catalog_323_provenance),
            "canonical_routing_markers": dict(self.canonical_routing_markers),
            "canonical_dispatch_recipe_hint": dict(self.canonical_dispatch_recipe_hint),
        }
        if self.predicted_axis_decomposition is not None:
            payload["predicted_axis_decomposition"] = (
                self.predicted_axis_decomposition.as_dict()
            )
        else:
            payload["predicted_axis_decomposition"] = None
        return payload


# ---------------------------------------------------------------------------
# Canonical sentinel for BUILD-1 / BUILD-2 deferral.
# ---------------------------------------------------------------------------


_BUILD_1_DEFER_SENTINEL = (
    "Scaffold-only per PAIR-FRAME-SCORER-GEOMETRY-LATTICE design memo; "
    "BUILD-1 sister subagent op-routable populates empirical 5D canvas "
    "from .omx/state/master_gradient_anchors.jsonl + per-pair component "
    "xray + per-frame master-gradient consumer rows + codex v1 row-based "
    "reader at tac.optimization.pair_frame_scorer_geometry_lattice. See "
    ".omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md "
    "§DELIVERABLE 3 BUILD-1."
)

_BUILD_2_DEFER_SENTINEL = (
    "Scaffold-only per PAIR-FRAME-SCORER-GEOMETRY-LATTICE design memo; "
    "BUILD-2 sister subagent op-routable implements the 4 canonical "
    "operation generators. See "
    ".omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md "
    "§DELIVERABLE 3 BUILD-2."
)


# ---------------------------------------------------------------------------
# Canonical primitive operations (scaffold; defer to BUILD-1 / BUILD-2).
# ---------------------------------------------------------------------------


def bind_pair_component_xray(
    pair_idx: int,
    archive_path: Path,
) -> dict[ScorerAxis, Any]:
    """Extract per-pair component-level scorer response per Cable D master-gradient.

    Returns dict mapping ``scorer_axis`` -> per-frame array of component
    magnitudes. SCAFFOLD: BUILD-1 sister subagent implements via
    ``tac.master_gradient_comparison.multi_granularity`` + per-pair
    gradient clustering producer surface.

    Raises:
        NotImplementedError: pending BUILD-1.
    """
    if not isinstance(pair_idx, int) or isinstance(pair_idx, bool):
        raise ValueError(f"pair_idx must be int, got {type(pair_idx).__name__}")
    if not 0 <= pair_idx < CANONICAL_PAIR_COUNT:
        raise ValueError(
            f"pair_idx {pair_idx} out of range [0, {CANONICAL_PAIR_COUNT})"
        )
    if not isinstance(archive_path, Path):
        raise ValueError(
            f"archive_path must be Path, got {type(archive_path).__name__}"
        )
    raise NotImplementedError(_BUILD_1_DEFER_SENTINEL)


def decompose_frame_axis_master_gradient(
    frame_idx: int,
    archive_path: Path,
) -> dict[ScorerAxis, Any]:
    """Per-frame master gradient per per_frame_master_gradient_consumer (codex sister).

    Returns dict mapping ``scorer_axis`` -> per-pair array of gradient
    magnitudes. SCAFFOLD: BUILD-1 sister subagent implements via
    ``tac.cathedral_consumers.per_frame_sensitivity_consumer`` or sister
    per-frame gradient consumer.

    Raises:
        NotImplementedError: pending BUILD-1.
    """
    if not isinstance(frame_idx, int) or isinstance(frame_idx, bool):
        raise ValueError(f"frame_idx must be int, got {type(frame_idx).__name__}")
    if not 0 <= frame_idx < CANONICAL_FRAME_COUNT:
        raise ValueError(
            f"frame_idx {frame_idx} out of range [0, {CANONICAL_FRAME_COUNT})"
        )
    if not isinstance(archive_path, Path):
        raise ValueError(
            f"archive_path must be Path, got {type(archive_path).__name__}"
        )
    raise NotImplementedError(_BUILD_1_DEFER_SENTINEL)


def compute_segnet_posenet_score_geometry(
    pair_idx: int,
    frame_idx: int,
    archive_path: Path,
) -> dict[CpuCudaAxis, dict[ScorerAxis, float]]:
    """Per-pair × per-frame × per-axis × CPU/CUDA score-component decomposition.

    Returns nested dict ``{cpu_cuda_axis: {scorer_axis: scalar}}``.

    Per canonical equation #18 ``pose_axis_cuda_amplification_v1`` (in
    registry; well-calibrated): CUDA-axis pose drift gap is ~5x CPU for
    HNeRV-class archives at PR106-frontier operating point.

    SCAFFOLD: BUILD-1 sister subagent implements via paired CPU+CUDA
    component xray rows + sister ``cpu_cuda_score_gap_v1`` (equation #17)
    per-archive calibration.

    Raises:
        NotImplementedError: pending BUILD-1.
    """
    if not isinstance(pair_idx, int) or isinstance(pair_idx, bool):
        raise ValueError(f"pair_idx must be int, got {type(pair_idx).__name__}")
    if not isinstance(frame_idx, int) or isinstance(frame_idx, bool):
        raise ValueError(f"frame_idx must be int, got {type(frame_idx).__name__}")
    raise NotImplementedError(_BUILD_1_DEFER_SENTINEL)


def query_receiver_runtime_feasibility(
    pair_idx: int,
    frame_idx: int,
    receiver_runtime: ReceiverRuntime,
    archive_path: Path,
) -> dict[ScorerAxis, bool]:
    """Per-receiver-mode feasibility check via Dykstra alternating-projection sister.

    Returns dict mapping ``scorer_axis`` -> bool feasibility (True if the
    receiver runtime is structurally compatible with the scorer axis's
    signal preservation requirements).

    Per CLAUDE.md "Council conduct" Dykstra co-lead: alternating-projection
    feasibility IS the arbiter of multi-constraint composition achievability.

    SCAFFOLD: BUILD-1 sister subagent implements via per-substrate archive
    grammar feasibility lookup table.

    Raises:
        NotImplementedError: pending BUILD-1.
    """
    if not isinstance(receiver_runtime, ReceiverRuntime):
        raise ValueError(
            f"receiver_runtime must be ReceiverRuntime, got "
            f"{type(receiver_runtime).__name__}"
        )
    raise NotImplementedError(_BUILD_1_DEFER_SENTINEL)


def generate_queue_executable_start(
    operation: CanonicalOperation,
    pair_idxs: Sequence[int],
    frame_idxs: Sequence[int],
    receiver_runtime: ReceiverRuntime,
    cpu_cuda_axis: CpuCudaAxis,
    canvas: PairFrameScorerGeometryLattice,
) -> ExecutableCandidate:
    """Emit canonical archive candidate per design memo §DELIVERABLE 1.

    Returns ExecutableCandidate carrying archive_candidate_path + predicted
    measurements + canonical Provenance + canonical-routing markers
    (Tier A non-promotable defaults per Catalog #341) + canonical dispatch
    recipe hint.

    Per codex's explicit naming: emits "without false authority" — the
    candidate's canonical_routing_markers default to
    ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
    ``axis_tag="[predicted]"`` per Catalog #341.

    SCAFFOLD: BUILD-2 sister subagent implements the 4 operation generator
    bodies; this stub validates inputs and raises BUILD-2 deferral.

    Raises:
        NotImplementedError: pending BUILD-2.
    """
    if not isinstance(operation, CanonicalOperation):
        raise ValueError(
            f"operation must be CanonicalOperation, got "
            f"{type(operation).__name__}"
        )
    if not isinstance(receiver_runtime, ReceiverRuntime):
        raise ValueError(
            f"receiver_runtime must be ReceiverRuntime, got "
            f"{type(receiver_runtime).__name__}"
        )
    if not isinstance(cpu_cuda_axis, CpuCudaAxis):
        raise ValueError(
            f"cpu_cuda_axis must be CpuCudaAxis, got "
            f"{type(cpu_cuda_axis).__name__}"
        )
    raise NotImplementedError(_BUILD_2_DEFER_SENTINEL)


# ---------------------------------------------------------------------------
# Canonical container class.
# ---------------------------------------------------------------------------


class PairFrameScorerGeometryLattice:
    """Canonical 5D canvas binding pair × frame × scorer_axis × receiver_runtime × cpu_cuda_axis.

    Per design memo §DELIVERABLE 1: the canvas is the canonical binding
    surface; the 4 canonical operations (full-drop / repair / masked /
    feathered) operate on this canvas; downstream consumers (drop-many
    beam search, inverse-scorer null-direction sweep, global low-impact
    frame-drop probe, CUDA-axis DQS1 variant) consume from the canvas
    without bespoke per-family wiring.

    Designed per (per design memo §"Cargo-cult audit per assumption" 3
    HARD-EARNED priors):

    - Daubechies multi-scale wavelet partition prior (receiver_runtime
      hierarchy; feathered = smooth-transition wavelet partition)
    - Dykstra alternating-projection feasibility (multi-dim convex
      intersection; ``query_receiver_runtime_feasibility``)
    - Atick-Redlich cooperative-receiver (receiver_runtime semantics;
      repair operation)
    - Wyner-Ziv side-information (receiver-aware encoding)
    - Cable D master-gradient consumers (per-pair + per-frame
      decomposition producer surface)
    - Catalog #356 per-axis AxisDecomposition (scorer_axis structural
      binding)
    - Catalog #357 Tier B canonical contract (per Catalog #341 canonical-
      routing markers for BUILD-4 promotion)

    Scope of this scaffold: canonical interface contract + per-cell
    lookup + canonical operation generator stubs. The empirical canvas
    population + 4 operation generator bodies + canonical CLI are sister-
    subagent deliverables BUILD-1 through BUILD-4.
    """

    def __init__(
        self,
        archive_sha256: str,
        cells: Mapping[tuple, PairFrameScorerGeometryCell] | None = None,
    ) -> None:
        """Initialize canvas for a given archive.

        Args:
            archive_sha256: canonical archive sha256 the canvas was
                populated against (per design memo deterministic
                reproducibility invariant).
            cells: optional dict mapping per-cell coordinate tuple to
                ``PairFrameScorerGeometryCell``. Default empty; BUILD-1
                populates via ``build_lattice`` classmethod.
        """
        if not isinstance(archive_sha256, str) or not archive_sha256:
            raise ValueError(
                "archive_sha256 must be a non-empty string (canonical "
                "archive sha per design memo deterministic reproducibility)"
            )
        self.archive_sha256 = archive_sha256
        self._cells: dict[tuple, PairFrameScorerGeometryCell] = dict(cells or {})

    @classmethod
    def build_lattice(
        cls, archive_path: Path
    ) -> PairFrameScorerGeometryLattice:
        """Build full 5D canvas for given archive.

        Sparse representation: only cells where receiver_feasibility=True
        OR predicted_delta_score is finite are populated.

        Expected sparse canvas size ~10^4-10^5 cells per current DQS1
        cascade (~581 candidates explored; sparse canvas extension is
        canvas-bounded).

        SCAFFOLD: BUILD-1 sister subagent implements via reads of
        ``.omx/state/master_gradient_anchors.jsonl`` + per-pair component
        xray rows + per-frame master-gradient consumer rows + sister
        codex v1 row-based reader at
        ``tac.optimization.pair_frame_scorer_geometry_lattice.build_pair_frame_scorer_geometry_lattice``.

        Raises:
            NotImplementedError: pending BUILD-1.
        """
        if not isinstance(archive_path, Path):
            raise ValueError(
                f"archive_path must be Path, got {type(archive_path).__name__}"
            )
        raise NotImplementedError(_BUILD_1_DEFER_SENTINEL)

    @classmethod
    def load_empirical_lattice(
        cls,
        archive_sha256: str,
        repo_root: Path,
    ) -> PairFrameScorerGeometryLattice:
        """Load canonical empirical canvas JSON from .omx/state/.

        Expected JSON path:
        ``.omx/state/pair_frame_scorer_geometry_lattice/<archive_sha[:12]>_<utc>.json``

        SCAFFOLD: BUILD-1 sister subagent implements canonical JSON
        schema + fcntl-locked write path + this reader.

        Raises:
            NotImplementedError: pending BUILD-1.
        """
        if not isinstance(archive_sha256, str) or not archive_sha256:
            raise ValueError("archive_sha256 must be a non-empty string")
        if not isinstance(repo_root, Path):
            raise ValueError(
                f"repo_root must be Path, got {type(repo_root).__name__}"
            )
        raise NotImplementedError(_BUILD_1_DEFER_SENTINEL)

    def query_cell(
        self,
        pair_idx: int,
        frame_idx: int,
        scorer_axis: ScorerAxis,
        receiver_runtime: ReceiverRuntime,
        cpu_cuda_axis: CpuCudaAxis,
    ) -> PairFrameScorerGeometryCell | None:
        """Per-cell lookup by 5D coordinate.

        Returns the cell if populated; None if the coordinate is not in
        the sparse canvas (the canvas does not enumerate all 21.6M
        cells; sparse cells correspond to empirical anchors or
        feasibility-positive coordinates).
        """
        key = (pair_idx, frame_idx, scorer_axis, receiver_runtime, cpu_cuda_axis)
        return self._cells.get(key)

    def cell_count(self) -> int:
        """Number of populated cells in this canvas (sparse representation)."""
        return len(self._cells)

    def generate_full_drop_starts(self, top_n: int = 32) -> list[ExecutableCandidate]:
        """Codex's queued blocked family: global low-impact full-pair/frame-drop probe.

        Emits up to ``top_n`` ExecutableCandidates ranked by per-cell
        predicted ΔS via the canvas's canonical per-axis × receiver_runtime
        × cpu_cuda_axis decomposition. Each candidate is a FULL_DROP
        operation across a sequence of (pair, frame) coordinates.

        SCAFFOLD: BUILD-2 sister subagent implements the ranking + emission.

        Raises:
            NotImplementedError: pending BUILD-2.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n must be positive int, got {top_n}")
        raise NotImplementedError(_BUILD_2_DEFER_SENTINEL)

    def generate_repair_starts(self, top_n: int = 32) -> list[ExecutableCandidate]:
        """Per Atick-Redlich cooperative-receiver: drop + repair signal.

        Emits up to ``top_n`` ExecutableCandidates for REPAIR operations
        across (pair, frame) coordinates. Repair requires non-trivial
        receiver_runtime (MASKED / FEATHERED / additive); RAW_RESIDUAL is
        OUT-OF-SCOPE for repair.

        SCAFFOLD: BUILD-2 sister subagent implements via canonical Atick-
        Redlich cooperative-receiver loss decomposition.

        Raises:
            NotImplementedError: pending BUILD-2.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n must be positive int, got {top_n}")
        raise NotImplementedError(_BUILD_2_DEFER_SENTINEL)

    def generate_masked_starts(self, top_n: int = 32) -> list[ExecutableCandidate]:
        """Per UNIWARD/HILL/J-UNIWARD: per-region SegNet-class-aware byte mask.

        Emits up to ``top_n`` ExecutableCandidates for MASKED operations.
        The MASKED receiver_runtime IS the canonical mode; sister modes
        FEATHERED + RAW_RESIDUAL provide structural variation.

        SCAFFOLD: BUILD-2 sister subagent implements via canonical UNIWARD
        steganalysis-inverse + per_segnet_class_chroma_consumer sister.

        Raises:
            NotImplementedError: pending BUILD-2.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n must be positive int, got {top_n}")
        raise NotImplementedError(_BUILD_2_DEFER_SENTINEL)

    def generate_feathered_starts(self, top_n: int = 32) -> list[ExecutableCandidate]:
        """Per Daubechies multi-scale wavelet partition prior: smooth-transition mask.

        Emits up to ``top_n`` ExecutableCandidates for FEATHERED operations.
        The FEATHERED receiver_runtime IS the canonical mode; sister modes
        MASKED + SMOOTHED_RESIDUAL provide structural variation.

        SCAFFOLD: BUILD-2 sister subagent implements via canonical Daubechies
        wavelet codec + multi-scale partition prior.

        Raises:
            NotImplementedError: pending BUILD-2.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n must be positive int, got {top_n}")
        raise NotImplementedError(_BUILD_2_DEFER_SENTINEL)


# ---------------------------------------------------------------------------
# Canonical contract callable surfaces per Catalog #335 (Tier A scaffold).
# ---------------------------------------------------------------------------


def update_from_anchor(anchor: Any) -> None:
    """Continual-learning posterior hook (Catalog #125 hook #5) — Tier A no-op.

    Per Catalog #335 canonical Protocol: every cathedral consumer exposes
    ``update_from_anchor``. At Tier A scaffold landing this is a no-op;
    BUILD-4 sister subagent op-routable populates the canonical posterior
    update via ``tac.canonical_equations.update_equation_with_empirical_anchor``
    for the QUEUED ``pair_frame_scorer_geometry_lattice_4d_binding_canonical_v1``
    canonical equation candidate.
    """
    # Tier A scaffold: posterior updates are sister-subagent BUILD-4
    # responsibility. Anchor accepted + ignored per canonical contract.
    _ = anchor
    return None


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Cathedral autopilot dispatch hook (Catalog #125 hook #4) — Tier A markers.

    Per Catalog #335 + Catalog #341: returns canonical non-promotable
    markers at Tier A scaffold landing. BUILD-4 sister subagent op-routable
    promotes to Tier B with empirically-grounded ``axis_tag`` +
    ``predicted_delta_adjustment != 0.0`` via canvas query.
    """
    if not isinstance(candidate, Mapping):
        raise ValueError(
            f"candidate must be a Mapping, got {type(candidate).__name__}"
        )
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "pair_frame_scorer_geometry_lattice_5d_canvas Tier A scaffold; "
            "BUILD-4 sister subagent op-routable promotes to Tier B "
            "with empirically-grounded axis_tag + non-zero "
            "predicted_delta_adjustment via canvas query. "
            "Per design memo .omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
    }


__all__ = [
    "CANONICAL_FRAME_COUNT",
    "CANONICAL_PAIR_COUNT",
    "CANVAS_SCHEMA",
    "CELL_SCHEMA",
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_TIER",
    "CONSUMER_VERSION",
    "CanonicalOperation",
    "CpuCudaAxis",
    "ExecutableCandidate",
    "PairFrameScorerGeometryCell",
    "PairFrameScorerGeometryLattice",
    "ReceiverRuntime",
    "ScorerAxis",
    "bind_pair_component_xray",
    "compute_segnet_posenet_score_geometry",
    "consume_candidate",
    "decompose_frame_axis_master_gradient",
    "generate_queue_executable_start",
    "query_receiver_runtime_feasibility",
    "update_from_anchor",
]
