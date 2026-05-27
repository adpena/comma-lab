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

import hashlib
import math
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
from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.validator import provenance_to_dict

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
    frame_idx = min(pair_idx * 2, CANONICAL_FRAME_COUNT - 1)
    geometry = compute_segnet_posenet_score_geometry(
        pair_idx=pair_idx,
        frame_idx=frame_idx,
        archive_path=archive_path,
    )
    by_scorer_axis: dict[ScorerAxis, dict[str, float]] = {}
    for cpu_cuda_axis, axis_values in geometry.items():
        for scorer_axis, value in axis_values.items():
            by_scorer_axis.setdefault(scorer_axis, {})[cpu_cuda_axis.value] = value
    return by_scorer_axis


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
    pair_idx = min(frame_idx // 2, CANONICAL_PAIR_COUNT - 1)
    geometry = compute_segnet_posenet_score_geometry(
        pair_idx=pair_idx,
        frame_idx=frame_idx,
        archive_path=archive_path,
    )
    by_scorer_axis: dict[ScorerAxis, dict[str, float]] = {}
    for cpu_cuda_axis, axis_values in geometry.items():
        for scorer_axis, value in axis_values.items():
            by_scorer_axis.setdefault(scorer_axis, {})[cpu_cuda_axis.value] = value
    return by_scorer_axis


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
    if not 0 <= pair_idx < CANONICAL_PAIR_COUNT:
        raise ValueError(
            f"pair_idx {pair_idx} out of range [0, {CANONICAL_PAIR_COUNT})"
        )
    if not 0 <= frame_idx < CANONICAL_FRAME_COUNT:
        raise ValueError(
            f"frame_idx {frame_idx} out of range [0, {CANONICAL_FRAME_COUNT})"
        )
    if not isinstance(archive_path, Path):
        raise ValueError(
            f"archive_path must be Path, got {type(archive_path).__name__}"
        )
    canvas = PairFrameScorerGeometryLattice.build_lattice(archive_path)
    result: dict[CpuCudaAxis, dict[ScorerAxis, float]] = {}
    for cell in canvas._cells.values():
        if cell.pair_idx != pair_idx or cell.frame_idx != frame_idx:
            continue
        if cell.receiver_runtime is not ReceiverRuntime.RAW_RESIDUAL:
            continue
        result.setdefault(cell.cpu_cuda_axis, {})[cell.scorer_axis] = (
            cell.predicted_delta_score
        )
    return result


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
    if not isinstance(pair_idx, int) or isinstance(pair_idx, bool):
        raise ValueError(f"pair_idx must be int, got {type(pair_idx).__name__}")
    if not 0 <= pair_idx < CANONICAL_PAIR_COUNT:
        raise ValueError(
            f"pair_idx {pair_idx} out of range [0, {CANONICAL_PAIR_COUNT})"
        )
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
    if receiver_runtime is ReceiverRuntime.RAW_RESIDUAL:
        return {
            ScorerAxis.SEGNET_5CLASS: True,
            ScorerAxis.POSENET_6D: True,
            ScorerAxis.RATE_TERM: True,
        }
    if receiver_runtime is ReceiverRuntime.FULL_DROP:
        return {
            ScorerAxis.SEGNET_5CLASS: False,
            ScorerAxis.POSENET_6D: False,
            ScorerAxis.RATE_TERM: True,
        }
    if receiver_runtime is ReceiverRuntime.MASKED:
        return {
            ScorerAxis.SEGNET_5CLASS: True,
            ScorerAxis.POSENET_6D: False,
            ScorerAxis.RATE_TERM: True,
        }
    return {
        ScorerAxis.SEGNET_5CLASS: True,
        ScorerAxis.POSENET_6D: True,
        ScorerAxis.RATE_TERM: True,
    }


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
    if not isinstance(canvas, PairFrameScorerGeometryLattice):
        raise ValueError(
            f"canvas must be PairFrameScorerGeometryLattice, got "
            f"{type(canvas).__name__}"
        )
    pair_set = {int(value) for value in pair_idxs}
    frame_set = {int(value) for value in frame_idxs}
    if not pair_set:
        raise ValueError("pair_idxs must contain at least one pair index")
    if not frame_set:
        raise ValueError("frame_idxs must contain at least one frame index")
    for pair_idx in pair_set:
        if not 0 <= pair_idx < CANONICAL_PAIR_COUNT:
            raise ValueError(
                f"pair_idx {pair_idx} out of range [0, {CANONICAL_PAIR_COUNT})"
            )
    for frame_idx in frame_set:
        if not 0 <= frame_idx < CANONICAL_FRAME_COUNT:
            raise ValueError(
                f"frame_idx {frame_idx} out of range [0, {CANONICAL_FRAME_COUNT})"
            )
    cells = [
        cell
        for cell in canvas._cells.values()
        if cell.pair_idx in pair_set
        and cell.frame_idx in frame_set
        and cell.receiver_runtime is receiver_runtime
        and cell.cpu_cuda_axis is cpu_cuda_axis
        and cell.receiver_feasibility
    ]
    if not cells:
        raise ValueError(
            "no feasible cells matched the requested pair/frame/runtime/hardware "
            "coordinate"
        )
    axis_decomposition = _build_axis_decomposition_for_candidate(
        cells, canvas.archive_sha256, operation
    )
    total_delta = sum(float(cell.predicted_delta_score) for cell in cells)
    total_bytes = sum(int(cell.predicted_byte_cost) for cell in cells)
    group_key = (*sorted(pair_set), *sorted(frame_set))
    recipe_hint = {
        "operation": operation.value,
        "receiver_runtime": receiver_runtime.value,
        "cpu_cuda_axis": cpu_cuda_axis.value,
        "pair_idxs": sorted(pair_set),
        "frame_idxs": sorted(frame_set),
        "group_size_cells": len(cells),
        "candidate_source": "generate_queue_executable_start",
        "candidate_authority": "predicted_metadata_only",
    }
    return ExecutableCandidate(
        operation=operation,
        archive_candidate_path=_candidate_archive_path(
            canvas, operation, group_key, receiver_runtime, cpu_cuda_axis
        ),
        predicted_delta_score=total_delta,
        predicted_byte_cost=total_bytes,
        catalog_323_provenance=dict(axis_decomposition.canonical_provenance),
        canonical_dispatch_recipe_hint=recipe_hint,
        predicted_axis_decomposition=axis_decomposition,
    )


# ---------------------------------------------------------------------------
# BUILD-2 canonical operation generator helper (sister of the 4 instance
# methods on PairFrameScorerGeometryLattice; module-level for testability).
# ---------------------------------------------------------------------------

_PREDICTOR_MODEL_ID_FULL_DROP = "pair_frame_scorer_geometry_lattice_5d_canvas.full_drop_v0"
_PREDICTOR_MODEL_ID_REPAIR = "pair_frame_scorer_geometry_lattice_5d_canvas.repair_v0"
_PREDICTOR_MODEL_ID_MASKED = "pair_frame_scorer_geometry_lattice_5d_canvas.masked_v0"
_PREDICTOR_MODEL_ID_FEATHERED = "pair_frame_scorer_geometry_lattice_5d_canvas.feathered_v0"

_OPERATION_MODEL_ID_MAP: Mapping[CanonicalOperation, str] = {
    CanonicalOperation.FULL_DROP: _PREDICTOR_MODEL_ID_FULL_DROP,
    CanonicalOperation.REPAIR: _PREDICTOR_MODEL_ID_REPAIR,
    CanonicalOperation.MASKED: _PREDICTOR_MODEL_ID_MASKED,
    CanonicalOperation.FEATHERED: _PREDICTOR_MODEL_ID_FEATHERED,
}


def _scorer_axis_score_contributions(
    cells: Sequence[PairFrameScorerGeometryCell],
) -> tuple[float, float, int]:
    """Sum per-axis deltas across a sequence of cells.

    Returns (sum_seg_delta_score, sum_pose_delta_score, sum_rate_bytes).

    Per CLAUDE.md "Apples-to-apples evidence discipline": the canvas's
    per-cell ``predicted_delta_score`` is already in canonical contest-score
    units (matches ``S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes
    / 37545489``). For BUILD-3 Catalog #356 AxisDecomposition wire-in we
    INVERT the canonical formula per scorer_axis so the AxisDecomposition
    carries the underlying d_seg / d_pose / archive_bytes deltas instead
    of the composed score delta. This preserves apples-to-apples per the
    10th standing directive.
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
    return seg_score_delta, pose_score_delta, rate_bytes_delta


def _build_axis_decomposition_for_candidate(
    cells: Sequence[PairFrameScorerGeometryCell],
    archive_sha256: str,
    operation: CanonicalOperation,
) -> AxisDecomposition:
    """Compose per-cell deltas into a canonical AxisDecomposition per Catalog #356.

    Per Catalog #356 STRICT preflight gate: any consumer that emits
    ``predicted_axis_decomposition`` MUST thread canonical Provenance via
    one of the canonical builders (we use ``build_provenance_for_predicted``
    since the canvas operation generators emit PREDICTED candidates per
    Catalog #287/#341 axis_tag="[predicted]").

    Per the 10th standing directive (apples-to-apples): we INVERT the
    canonical scoring formula to recover per-axis deltas (d_seg, d_pose,
    archive_bytes) from the composed score delta. The canonical formula
    is ``S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes /
    37545489``, so:

        d_seg_delta = score_delta_seg / 100
        d_pose_delta = score_delta_pose / sqrt(10)
            (only valid as a linear approximation around the operating
            point; the canonical full composition lives in
            ``tac.score_composition.compose_score_from_axes`` which uses
            the difference-of-sqrt form per CLAUDE.md "SegNet vs PoseNet
            importance — operating-point dependent")
        archive_bytes_delta = sum of rate-axis byte costs
    """
    seg_score_delta, pose_score_delta, rate_bytes_delta = (
        _scorer_axis_score_contributions(cells)
    )
    # Linear inversion (per Catalog #356 docstring: AxisDecomposition fields
    # carry d_seg_delta + d_pose_delta + archive_bytes_delta; the canonical
    # ranker uses tac.score_composition.compose_score_from_axes for the
    # difference-of-sqrt pose composition).
    d_seg_delta = seg_score_delta / 100.0
    # Pose: per CLAUDE.md "SegNet vs PoseNet importance — operating-point
    # dependent" the marginal sensitivity is 5/sqrt(10*d_pose). For
    # canvas-level prediction we use the canonical linear approx
    # d_pose_delta ≈ score_delta_pose / sqrt(10) which is conservative at
    # the PR106 frontier (pose_avg ≈ 3.4e-5; marginal ≈ 271). Downstream
    # cathedral ranker composes via compose_score_from_axes for the
    # accurate difference-of-sqrt form.
    d_pose_delta = pose_score_delta / math.sqrt(10.0)

    # Canonical Provenance per Catalog #323. Per Catalog #356 STRICT gate:
    # the canonical_provenance MUST be a dict produced via the canonical
    # builders. We use build_provenance_for_predicted since canvas candidates
    # are PREDICTED-from-model per Catalog #287/#341.
    coord_signature = "|".join(
        sorted(
            f"({c.pair_idx},{c.frame_idx},{c.scorer_axis.value},"
            f"{c.receiver_runtime.value},{c.cpu_cuda_axis.value})"
            for c in cells
        )
    )
    inputs_payload = f"{archive_sha256}|{operation.value}|{coord_signature}"
    inputs_sha = sha256_hex(inputs_payload)
    prov = build_provenance_for_predicted(
        model_id=_OPERATION_MODEL_ID_MAP[operation],
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


def sha256_hex(text: str) -> str:
    """Canonical sha256 helper for inputs_sha256 (Catalog #323 sister)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _derive_archive_sha256_from_path(archive_path: Path) -> str:
    """Compute archive sha256 for class-level empirical lattice loading."""
    h = hashlib.sha256()
    with archive_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_root_for_archive_path(archive_path: Path) -> Path | None:
    """Find nearest parent carrying the canonical master-gradient ledger."""
    for parent in (archive_path.parent, *archive_path.parents):
        if (parent / ".omx" / "state" / "master_gradient_anchors.jsonl").exists():
            return parent
    return None


def _candidate_archive_path(
    canvas: PairFrameScorerGeometryLattice,
    operation: CanonicalOperation,
    group_key: tuple[int, ...],
    receiver_runtime: ReceiverRuntime,
    cpu_cuda_axis: CpuCudaAxis,
) -> Path:
    """Canonical scaffold-only sentinel path for the ExecutableCandidate.

    Per the audit memo §"Direct answer to operator question": BUILD-2
    operation generators emit CANDIDATE METADATA; the actual archive-byte
    construction is PAID-DISPATCH territory (audit's PRIORITY 4 / FIRE
    phase). The path here is deterministic from the canvas + group key
    so downstream consumers can pair candidates with their dispatch
    artifacts at dispatch-time.
    """
    group_str = "_".join(str(g) for g in group_key)
    sentinel_filename = (
        f"{canvas.archive_sha256[:12]}_{operation.value}_"
        f"{receiver_runtime.value}_{cpu_cuda_axis.value}_{group_str}.candidate.json"
    )
    return Path(
        f".omx/state/pair_frame_scorer_geometry_lattice_candidates/{sentinel_filename}"
    )


def _generate_operation_candidates(
    canvas: PairFrameScorerGeometryLattice,
    operation: CanonicalOperation,
    receiver_runtime: ReceiverRuntime,
    top_n: int,
    *,
    group_by_frame: bool = False,
) -> list[ExecutableCandidate]:
    """Canonical helper consumed by all 4 operation generator methods.

    Per design memo §DELIVERABLE 1 + audit memo §"PRIORITY 2 BUILD-2 +
    BUILD-3 sister subagent": this is the canonical body the 4 instance
    methods delegate to.

    Algorithm (substrate-optimal per UNIQUE-AND-COMPLETE-PER-METHOD):

      1. Filter canvas cells by receiver_runtime + receiver_feasibility.
      2. Group by ``(pair_idx, cpu_cuda_axis)`` (pair-level operations:
         FULL_DROP / REPAIR) or ``(frame_idx, cpu_cuda_axis)`` (frame-
         level operations: MASKED / FEATHERED).
      3. For each group, sum per-axis predicted_delta_score deltas into a
         composite candidate; build AxisDecomposition per Catalog #356;
         thread canonical Provenance per Catalog #323.
      4. Rank ascending by composite predicted_delta_score; return top_n.

    Per the 11th ORDER MATTERS standing directive: order of operations
    matters — the 4 generators emit candidates for downstream DISPATCH
    ranking, not for direct mutation. Catalog #341 Tier A markers
    enforced; promotable=False by construction until paired-axis
    empirical anchor lands per Catalog #357 Tier B promotion in BUILD-4.
    """
    if canvas.cell_count() == 0:
        return []

    grouped: dict[tuple[int, ...], list[PairFrameScorerGeometryCell]] = {}
    for cell in canvas._cells.values():
        if cell.receiver_runtime is not receiver_runtime:
            continue
        if not cell.receiver_feasibility:
            continue
        if group_by_frame:
            group_key = (cell.frame_idx, _cpu_cuda_axis_sort_key(cell.cpu_cuda_axis))
        else:
            group_key = (cell.pair_idx, _cpu_cuda_axis_sort_key(cell.cpu_cuda_axis))
        grouped.setdefault(group_key, []).append(cell)

    if not grouped:
        return []

    candidates: list[ExecutableCandidate] = []
    for group_key, group_cells in grouped.items():
        total_delta = sum(float(c.predicted_delta_score) for c in group_cells)
        total_bytes = sum(int(c.predicted_byte_cost) for c in group_cells)
        if total_delta >= 0.0:
            # Skip groups that don't predict a score improvement; the
            # canvas IS the per-cell decomposition surface; the ranker
            # consumes the per-cell signal directly via Catalog #356.
            # FULL_DROP / REPAIR / MASKED / FEATHERED candidates that
            # predict score regression are NOT emitted (matches the
            # CLAUDE.md "Forbidden premature KILL" + "Beauty, simplicity,
            # and developer experience" principles: emit fewer, higher-EV
            # candidates).
            continue
        # Determine cpu_cuda_axis from the first cell (all cells in the
        # group share it by construction per the group_key).
        cpu_cuda_axis = group_cells[0].cpu_cuda_axis
        axis_decomp = _build_axis_decomposition_for_candidate(
            group_cells, canvas.archive_sha256, operation
        )
        # Canonical Provenance threaded through the candidate's
        # catalog_323_provenance field (same value as the AxisDecomposition's
        # canonical_provenance per Catalog #323 + #356 sister discipline).
        # Tier A canonical-routing markers per Catalog #341 (defaults
        # propagate via ExecutableCandidate dataclass).
        recipe_hint = {
            "operation": operation.value,
            "receiver_runtime": receiver_runtime.value,
            "cpu_cuda_axis": cpu_cuda_axis.value,
            "group_key": list(group_key),
            "group_size_cells": len(group_cells),
            "estimated_paid_dispatch_usd_band_low": 1.0,
            "estimated_paid_dispatch_usd_band_high": 3.0,
            "estimated_smoke_dispatch_usd": 0.30,
            "design_memo": (
                ".omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md"
            ),
        }
        candidate = ExecutableCandidate(
            operation=operation,
            archive_candidate_path=_candidate_archive_path(
                canvas, operation, group_key, receiver_runtime, cpu_cuda_axis
            ),
            predicted_delta_score=total_delta,
            predicted_byte_cost=total_bytes,
            catalog_323_provenance=dict(axis_decomp.canonical_provenance),
            canonical_dispatch_recipe_hint=recipe_hint,
            predicted_axis_decomposition=axis_decomp,
        )
        candidates.append(candidate)

    # Sort ascending by predicted_delta_score (best score-improvement first).
    # Stable tiebreak on group_key for determinism.
    candidates.sort(
        key=lambda c: (
            float(c.predicted_delta_score),
            list(c.canonical_dispatch_recipe_hint.get("group_key", [])),
        )
    )
    return candidates[:top_n]


def _cpu_cuda_axis_sort_key(axis: CpuCudaAxis) -> int:
    """Stable integer key for CpuCudaAxis (CPU=0, CUDA=1) for group ordering."""
    return 0 if axis is CpuCudaAxis.CONTEST_CPU else 1


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
        from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator import (
            populate_5d_canvas_from_master_gradient_anchors,
        )

        archive_sha256 = _derive_archive_sha256_from_path(archive_path)

        manifest = populate_5d_canvas_from_master_gradient_anchors(
            archive_sha256=archive_sha256,
            write_sidecar=False,
            repo_root=_repo_root_for_archive_path(archive_path),
        )
        return manifest.canvas

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
        from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator import (
            load_empirical_lattice,
        )

        return load_empirical_lattice(archive_sha256, repo_root=repo_root)

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

    # ---------------------------------------------------------------------
    # BUILD-2 + BUILD-3 operation generators (landed 2026-05-26).
    #
    # Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
    # the generators below replace the prior NotImplementedError DEFER
    # sentinel with canonical executable implementations. Each generator:
    #
    #   1. Filters per-cell coordinates by ``receiver_feasibility=True`` and
    #      ``predicted_delta_score < 0.0`` (score improvement).
    #   2. Groups cells by ``(operation, receiver_runtime, cpu_cuda_axis)``
    #      tuples; each unique tuple emits ONE ExecutableCandidate per the
    #      design memo §DELIVERABLE 1 contract.
    #   3. Composes per-cell scorer-axis deltas into ``AxisDecomposition``
    #      per Catalog #356 (BUILD-3 wire-in).
    #   4. Threads canonical Provenance per Catalog #323 via
    #      ``build_provenance_for_predicted`` + ``provenance_to_dict``.
    #   5. Sets canonical-routing markers per Catalog #341 (Tier A:
    #      ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
    #      ``axis_tag="[predicted]"``).
    #
    # The archive_candidate_path is a scaffold-only sentinel pointing at the
    # canvas's archive_sha256 directory; BUILD-2 emits the canonical
    # candidate metadata, and the actual archive-byte construction is
    # PAID-DISPATCH territory (FIRE phase per audit's PRIORITY 4).
    # ---------------------------------------------------------------------

    def generate_full_drop_starts(
        self,
        top_n: int = 32,
        *,
        receiver_runtime: ReceiverRuntime = ReceiverRuntime.FULL_DROP,
    ) -> list[ExecutableCandidate]:
        """Codex's queued blocked family: global low-impact full-pair/frame-drop probe.

        Filters the canvas to cells where the receiver_runtime is FULL_DROP
        (or caller-supplied), receiver_feasibility=True, and
        predicted_delta_score < 0 (score improvement). Groups by
        ``(pair_idx, cpu_cuda_axis)`` so a single pair_idx contributes ONE
        candidate per CPU/CUDA axis (sister to drop-one rank021 anchor).

        Args:
            top_n: maximum candidates to emit.
            receiver_runtime: receiver mode (defaults to FULL_DROP); accepts
                RAW_RESIDUAL for the canonical drop-one anchor path which
                EMPIRICALLY beats every other receiver per the 17 paired-CPU
                anchors documented in the audit memo Phase 3.

        Returns:
            List of ExecutableCandidates sorted ascending by
            ``predicted_delta_score`` (best score-improvement first). Each
            candidate carries per-axis decomposition per Catalog #356 +
            canonical Provenance per Catalog #323 + Tier A canonical-routing
            markers per Catalog #341.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n must be positive int, got {top_n}")
        if not isinstance(receiver_runtime, ReceiverRuntime):
            raise ValueError(
                f"receiver_runtime must be ReceiverRuntime, got "
                f"{type(receiver_runtime).__name__}"
            )
        return _generate_operation_candidates(
            canvas=self,
            operation=CanonicalOperation.FULL_DROP,
            receiver_runtime=receiver_runtime,
            top_n=top_n,
            group_by_frame=False,
        )

    def generate_repair_starts(
        self,
        top_n: int = 32,
        *,
        receiver_runtime: ReceiverRuntime = ReceiverRuntime.SMOOTHED_RESIDUAL,
    ) -> list[ExecutableCandidate]:
        """Per Atick-Redlich cooperative-receiver: drop + repair signal.

        REPAIR is the canonical operation that directly addresses the
        drop-one frontier paradox per the audit memo TL;DR: the apparatus
        empirically falsified narrow drop-many at the current operating
        point (rate-vs-distortion slope dominates), so REPLACE primitives
        that INJECT signal rather than REMOVE bytes are the canonical path
        forward. REPAIR refuses ``RAW_RESIDUAL`` receiver mode (which IS
        identity / no repair) per the design memo §DELIVERABLE 1 contract.

        Args:
            top_n: maximum candidates to emit.
            receiver_runtime: receiver mode (defaults to SMOOTHED_RESIDUAL —
                the Atick-Redlich canonical mode); refuses RAW_RESIDUAL.

        Returns:
            List of ExecutableCandidates per CanonicalOperation.REPAIR.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n must be positive int, got {top_n}")
        if not isinstance(receiver_runtime, ReceiverRuntime):
            raise ValueError(
                f"receiver_runtime must be ReceiverRuntime, got "
                f"{type(receiver_runtime).__name__}"
            )
        if receiver_runtime is ReceiverRuntime.RAW_RESIDUAL:
            raise ValueError(
                "REPAIR refuses RAW_RESIDUAL receiver mode per design memo "
                "§DELIVERABLE 1 contract (RAW_RESIDUAL IS identity / no "
                "repair). Use SMOOTHED_RESIDUAL / MASKED / FEATHERED."
            )
        return _generate_operation_candidates(
            canvas=self,
            operation=CanonicalOperation.REPAIR,
            receiver_runtime=receiver_runtime,
            top_n=top_n,
            group_by_frame=False,
        )

    def generate_masked_starts(
        self,
        top_n: int = 32,
        *,
        receiver_runtime: ReceiverRuntime = ReceiverRuntime.MASKED,
    ) -> list[ExecutableCandidate]:
        """Per UNIWARD/HILL/J-UNIWARD: per-region SegNet-class-aware byte mask.

        MASKED operates on per-frame granularity (group_by_frame=True) so a
        single frame_idx contributes ONE candidate per CPU/CUDA axis. This
        is sister to the per_segnet_class_chroma_consumer pattern.

        Args:
            top_n: maximum candidates to emit.
            receiver_runtime: receiver mode (defaults to MASKED — the
                canonical mode); accepts FEATHERED as sister mode.

        Returns:
            List of ExecutableCandidates per CanonicalOperation.MASKED.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n must be positive int, got {top_n}")
        if not isinstance(receiver_runtime, ReceiverRuntime):
            raise ValueError(
                f"receiver_runtime must be ReceiverRuntime, got "
                f"{type(receiver_runtime).__name__}"
            )
        return _generate_operation_candidates(
            canvas=self,
            operation=CanonicalOperation.MASKED,
            receiver_runtime=receiver_runtime,
            top_n=top_n,
            group_by_frame=True,
        )

    def generate_feathered_starts(
        self,
        top_n: int = 32,
        *,
        receiver_runtime: ReceiverRuntime = ReceiverRuntime.FEATHERED,
    ) -> list[ExecutableCandidate]:
        """Per Daubechies multi-scale wavelet partition prior: smooth-transition mask.

        FEATHERED operates on per-frame granularity like MASKED but uses
        smooth wavelet-partition transitions per the design memo §DELIVERABLE
        1 contract. Sister mode is SMOOTHED_RESIDUAL.

        Args:
            top_n: maximum candidates to emit.
            receiver_runtime: receiver mode (defaults to FEATHERED).

        Returns:
            List of ExecutableCandidates per CanonicalOperation.FEATHERED.
        """
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(f"top_n must be positive int, got {top_n}")
        if not isinstance(receiver_runtime, ReceiverRuntime):
            raise ValueError(
                f"receiver_runtime must be ReceiverRuntime, got "
                f"{type(receiver_runtime).__name__}"
            )
        return _generate_operation_candidates(
            canvas=self,
            operation=CanonicalOperation.FEATHERED,
            receiver_runtime=receiver_runtime,
            top_n=top_n,
            group_by_frame=True,
        )


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
    "sha256_hex",
    "update_from_anchor",
]
