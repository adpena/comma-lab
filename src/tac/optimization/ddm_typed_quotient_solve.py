# SPDX-License-Identifier: MIT
"""Strict scorer-metric primitives for the DDM MS2 typed quotient solve.

This module deliberately does not discover scorer geometry.  It consumes a
SHA-bound, measured geometry and refuses identity/Euclidean controls.  That
separation prevents an absent Fisher/Pose/composite-R Hessian from silently
becoming ``I`` inside KKT, lattice, trust-region, or dictionary updates.

The public surface covers the reusable mathematical core of MS2:

* visible quotient coordinates only; gauge coordinates are not variables;
* exact second-order active-set KKT with per-dimension effective quanta;
* a bounded exhaustive metric lattice sieve before any lattice-family negative;
* generalized/oblique SVD in the measured scorer metric;
* strict typed-atlas, alternation, and tolerance-ladder validation.

Every result remains advisory until an external receiver callback remeasures
the candidate through uint8, the exact R operator, and both frozen scorers.
"""

from __future__ import annotations

import importlib
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from tac.optimization.coupled_margin_levelset import (
    CouplingOperator,
    KKTStep,
    solve_active_set_kkt,
)

N_CLASSES: Final = 5
CLASS_PAIRS: Final = tuple(itertools.combinations(range(N_CLASSES), 2))
RATE_SCORE_PER_BYTE: Final = 25.0 / 37_545_489.0
METRIC_COORDINATE_SYSTEM: Final = "seg_rank4_winner_rival_hyperplanes_plus_pose6"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
STREAM_TYPE_CONTRACT_MODULE: Final = "tac.optimization.ddm_min_description_contract"
_STREAM_TYPE_VALUES: Final = frozenset(
    {"SKELETON", "CONNECTION", "FIBER", "GAUGE", "RESIDUAL"}
)


class TypedQuotientSolveError(ValueError):
    """Malformed or epistemically inadmissible MS2 solve input."""


class EvaluationRecursionLevel(StrEnum):
    """The three evaluate.py inversion levels from FEED-603."""

    LEVEL0_SCORE_SIGNATURE = "L0_SCORE_DISCRETE_QUOTIENT_X_CONTINUOUS_QUADRATIC"
    LEVEL1_SCORER_INTERNALS = "L1_ARGMAX_CELL_MARGIN_FIBER_POSE6"
    LEVEL2_PAIR_TRAJECTORY = "L2_PAIR_INDEPENDENCE_X_TEMPORAL_EVENT_FLOW"


class ScorerVisibility(StrEnum):
    SEG = "SEG_VISIBLE"
    POSE = "POSE_VISIBLE"
    JOINT = "SEG_POSE_VISIBLE"
    INVISIBLE_GAUGE = "SCORER_INVISIBLE_GAUGE"


class G4TemporalClass(StrEnum):
    STATIC_IN_IMAGE = "STATIC_IN_IMAGE"
    STATIC_IN_XI_PROXY = "STATIC_IN_XI_PROXY"
    TRANSIENT = "TRANSIENT"


class AlternationStage(StrEnum):
    ARGMAX_CELL = "ARGMAX_CELL_SELECTION"
    WITHIN_CELL_LATTICE = "WITHIN_CELL_CONTINUOUS_LATTICE"
    REAL_CODER_PRICE = "REAL_CODER_PRICE"


def _sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypedQuotientSolveError(f"{field} must be a lowercase SHA-256")
    return value


def canonical_stream_type_value(value: object, field: str = "stream_type") -> str:
    """Normalize against the canonical minimum-description stream contract.

    The isolated base predates the concurrently owned ``StreamType`` landing.
    If that class is present, it is the validator.  Until MAIN composes that
    landing, only its five announced wire values are accepted; this module
    deliberately defines no parallel enum or typed-tag schema.
    """

    raw = value.value if isinstance(getattr(value, "value", None), str) else value
    if not isinstance(raw, str):
        raise TypedQuotientSolveError(f"{field} must be a canonical stream-type value")
    contract = importlib.import_module(STREAM_TYPE_CONTRACT_MODULE)
    canonical_type = getattr(contract, "StreamType", None)
    if canonical_type is not None:
        try:
            normalized = canonical_type(raw)
        except (TypeError, ValueError) as exc:
            raise TypedQuotientSolveError(
                f"{field} is not admitted by {STREAM_TYPE_CONTRACT_MODULE}.StreamType"
            ) from exc
        return str(normalized.value if hasattr(normalized, "value") else normalized)
    if raw not in _STREAM_TYPE_VALUES:
        raise TypedQuotientSolveError(
            f"{field} is not one of the canonical stream wire values"
        )
    return raw


def _finite_matrix(value: np.ndarray, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)
    ):
        raise TypedQuotientSolveError(f"{field} must be a finite symmetric square matrix")
    return 0.5 * (matrix + matrix.T)


def _is_scalar_identity(matrix: np.ndarray) -> bool:
    scale = float(np.trace(matrix) / matrix.shape[0])
    return scale > 0.0 and np.allclose(
        matrix,
        scale * np.eye(matrix.shape[0], dtype=np.float64),
        rtol=1e-8,
        atol=1e-10,
    )


def _immutable(value: np.ndarray, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MeasuredScorerGeometry:
    """SHA-bound scorer coordinates, metric Gram, and exact second order.

    ``metric_gram`` governs CVP, proposal ranking, basis reduction, and
    generalized dictionary updates.  ``composite_hessian`` is the measured
    composite-R/scorer second-order term.  Their sum is the KKT metric.
    Neither matrix may be a scalar identity control.
    """

    metric_id: str
    coordinate_system: str
    metric_gram: np.ndarray
    composite_hessian: np.ndarray
    seg_head_rank: int
    pose_rank: int
    evidence_axis: str
    geometry_receipt_sha256: str
    composite_r_adjoint_sha256: str
    inner_jacobian_sha256: str
    pose_quadratic_sha256: str
    dual_metric_readback_active: bool
    bregman_binding_active: bool

    def __post_init__(self) -> None:
        if not isinstance(self.metric_id, str) or not self.metric_id.strip():
            raise TypedQuotientSolveError("metric_id must be nonempty")
        if "euclid" in self.metric_id.lower() or "identity" in self.metric_id.lower():
            raise TypedQuotientSolveError("identity/Euclidean metric IDs are control-only")
        if self.coordinate_system != METRIC_COORDINATE_SYSTEM:
            raise TypedQuotientSolveError(f"coordinate_system must be scorer-native {METRIC_COORDINATE_SYSTEM!r}")
        if self.evidence_axis != EVIDENCE_AXIS:
            raise TypedQuotientSolveError(f"evidence_axis must be {EVIDENCE_AXIS!r}")
        if isinstance(self.seg_head_rank, bool) or self.seg_head_rank != 4:
            raise TypedQuotientSolveError("Seg geometry must carry the exact rank-4 head")
        if isinstance(self.pose_rank, bool) or not isinstance(self.pose_rank, int) or not 1 <= self.pose_rank <= 6:
            raise TypedQuotientSolveError("Pose quadratic rank must be in [1, 6]")
        if self.dual_metric_readback_active is not True:
            raise TypedQuotientSolveError("dual-metric readback is required")
        if self.bregman_binding_active is not True:
            raise TypedQuotientSolveError("Bregman/Fisher binding is required")
        for value, field in (
            (self.geometry_receipt_sha256, "geometry_receipt_sha256"),
            (self.composite_r_adjoint_sha256, "composite_r_adjoint_sha256"),
            (self.inner_jacobian_sha256, "inner_jacobian_sha256"),
            (self.pose_quadratic_sha256, "pose_quadratic_sha256"),
        ):
            _sha256(value, field)

        gram = _finite_matrix(self.metric_gram, "metric_gram")
        hessian = _finite_matrix(self.composite_hessian, "composite_hessian")
        if gram.shape != hessian.shape:
            raise TypedQuotientSolveError("metric Gram and composite Hessian dimensions differ")
        if float(np.linalg.eigvalsh(gram).min()) < -1e-10:
            raise TypedQuotientSolveError("metric_gram must be positive semidefinite")
        if float(np.linalg.eigvalsh(hessian).min()) < -1e-10:
            raise TypedQuotientSolveError("composite_hessian must be positive semidefinite")
        combined = gram + hessian
        if _is_scalar_identity(gram) or _is_scalar_identity(combined):
            raise TypedQuotientSolveError(
                "identity/Euclidean geometry is a non-verdict control"
            )
        if float(np.linalg.eigvalsh(combined).min()) <= 1e-12:
            raise TypedQuotientSolveError("combined measured metric must be positive definite without identity damping")
        object.__setattr__(self, "metric_gram", _immutable(gram))
        object.__setattr__(self, "composite_hessian", _immutable(hessian))

    @property
    def dimension(self) -> int:
        return int(self.metric_gram.shape[0])

    @property
    def second_order_metric(self) -> np.ndarray:
        """Return the exact scorer metric plus composite second-order term."""

        return _immutable(self.metric_gram + self.composite_hessian)


@dataclass(frozen=True, slots=True)
class EffectiveQuantum:
    """Per-visible-dimension quantum: uint8 step times scorer sensitivity."""

    dof_label: str
    uint8_step: float
    scorer_sensitivity: float
    tolerance_rungs: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dof_label, str) or not self.dof_label.strip():
            raise TypedQuotientSolveError("quantum dof_label must be nonempty")
        step = float(self.uint8_step)
        sensitivity = float(self.scorer_sensitivity)
        if not math.isfinite(step) or step <= 0.0:
            raise TypedQuotientSolveError("uint8_step must be finite and positive")
        if not math.isfinite(sensitivity) or sensitivity <= 0.0:
            raise TypedQuotientSolveError("scorer_sensitivity must be measured, finite, and positive")
        rungs = tuple(float(value) for value in self.tolerance_rungs)
        if (
            not rungs
            or any(not math.isfinite(value) or value <= 0.0 for value in rungs)
            or any(right <= left for left, right in itertools.pairwise(rungs))
        ):
            raise TypedQuotientSolveError("tolerance_rungs must be finite, positive, and strictly increasing")
        object.__setattr__(self, "tolerance_rungs", rungs)

    @property
    def effective_quantum(self) -> float:
        return float(self.uint8_step * self.scorer_sensitivity)


@dataclass(frozen=True, slots=True)
class TypedBlock:
    """One PF2-reconciled typed block with explicit byte semantics."""

    block_id: str
    stratum: str
    scorer_visibility: ScorerVisibility
    temporal_class: G4TemporalClass
    class_pair: tuple[int, int]
    representation_type: str
    recursion_level: EvaluationRecursionLevel
    measured_flip_mass: int
    counted_bytes: int
    parameter_bytes: int
    exception_bytes: int
    connection_operator_code_bytes: int
    amortization_factor: int
    coder_race_winner: str
    pose_serving: bool
    atlas_receipt_sha256: str
    coder_race_receipt_sha256: str
    physical_bev_custody_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise TypedQuotientSolveError("block_id must be nonempty")
        if not isinstance(self.stratum, str) or not self.stratum.strip():
            raise TypedQuotientSolveError("block stratum must be nonempty")
        if (
            not isinstance(self.class_pair, tuple)
            or len(self.class_pair) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.class_pair
            )
            or self.class_pair not in CLASS_PAIRS
        ):
            raise TypedQuotientSolveError("class_pair must be one of the ten unordered pairs")
        try:
            scorer_visibility = ScorerVisibility(self.scorer_visibility)
            temporal_class = G4TemporalClass(self.temporal_class)
            recursion_level = EvaluationRecursionLevel(self.recursion_level)
        except (TypeError, ValueError) as exc:
            raise TypedQuotientSolveError(
                "scorer visibility, temporal class, and recursion level must be canonical"
            ) from exc
        object.__setattr__(self, "scorer_visibility", scorer_visibility)
        object.__setattr__(self, "temporal_class", temporal_class)
        object.__setattr__(self, "recursion_level", recursion_level)
        for value, field in (
            (self.measured_flip_mass, "measured_flip_mass"),
            (self.counted_bytes, "counted_bytes"),
            (self.parameter_bytes, "parameter_bytes"),
            (self.exception_bytes, "exception_bytes"),
            (self.connection_operator_code_bytes, "connection_operator_code_bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypedQuotientSolveError(f"{field} must be a nonnegative integer")
        if (
            isinstance(self.amortization_factor, bool)
            or not isinstance(self.amortization_factor, int)
            or self.amortization_factor < 1
        ):
            raise TypedQuotientSolveError("amortization_factor must be a positive integer")
        _sha256(self.atlas_receipt_sha256, "atlas_receipt_sha256")
        _sha256(self.coder_race_receipt_sha256, "coder_race_receipt_sha256")
        representation_type = canonical_stream_type_value(
            self.representation_type,
            "representation_type",
        )
        coder_race_winner = canonical_stream_type_value(
            self.coder_race_winner,
            "coder_race_winner",
        )
        object.__setattr__(self, "representation_type", representation_type)
        object.__setattr__(self, "coder_race_winner", coder_race_winner)
        if not isinstance(self.pose_serving, bool):
            raise TypedQuotientSolveError("pose_serving must be an exact boolean")
        if coder_race_winner not in {"SKELETON", "FIBER"}:
            raise TypedQuotientSolveError("coder race must measure SKELETON versus FIBER before distillation")
        if self.pose_serving and (
            representation_type != "FIBER"
            or scorer_visibility not in {ScorerVisibility.POSE, ScorerVisibility.JOINT}
        ):
            raise TypedQuotientSolveError(
                "pose-serving WS1 content must be FIBER priced in Pose-visible coordinates"
            )
        if representation_type == "GAUGE" and (
            scorer_visibility is not ScorerVisibility.INVISIBLE_GAUGE
            or any(
                (
                    self.counted_bytes,
                    self.parameter_bytes,
                    self.exception_bytes,
                    self.connection_operator_code_bytes,
                )
            )
        ):
            raise TypedQuotientSolveError("GAUGE must be scorer-invisible and exactly zero bytes")
        if representation_type == "CONNECTION":
            if self.connection_operator_code_bytes != 0:
                raise TypedQuotientSolveError("CONNECTION operator code is FREE")
            if self.counted_bytes != self.parameter_bytes + self.exception_bytes:
                raise TypedQuotientSolveError("CONNECTION counted bytes must equal parameters plus exceptions")
        if self.physical_bev_custody_sha256 is None:
            if temporal_class is G4TemporalClass.STATIC_IN_XI_PROXY and self.amortization_factor != 1:
                raise TypedQuotientSolveError("G4 xi-proxy is not physical BEV and cannot claim BEV amortization")
        else:
            _sha256(self.physical_bev_custody_sha256, "physical_bev_custody_sha256")


@dataclass(frozen=True, slots=True)
class TypedAtlasValidation:
    valid: bool
    block_count: int
    class_pair_count: int
    total_measured_flip_mass: int
    atlas_receipt_sha256: str
    coder_race_receipt_sha256: str


def validate_typed_atlas(
    blocks: Sequence[TypedBlock],
    *,
    measured_flip_mass_by_pair: Mapping[tuple[int, int], int],
) -> TypedAtlasValidation:
    """Require PF2 custody and measured coverage of all ten class boundaries."""

    rows = tuple(blocks)
    if not rows:
        raise TypedQuotientSolveError("typed atlas is empty")
    if len({row.block_id for row in rows}) != len(rows):
        raise TypedQuotientSolveError("typed atlas block IDs must be unique")
    if set(measured_flip_mass_by_pair) != set(CLASS_PAIRS):
        raise TypedQuotientSolveError("measured flip-mass map must cover all ten class pairs")
    masses: dict[tuple[int, int], int] = {}
    for pair in CLASS_PAIRS:
        value = measured_flip_mass_by_pair[pair]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypedQuotientSolveError("flip masses must be measured nonnegative integers")
        masses[pair] = value
    if sum(masses.values()) <= 0:
        raise TypedQuotientSolveError("measured class-pair flip mass is empty")
    covered = {row.class_pair for row in rows}
    if covered != set(CLASS_PAIRS):
        raise TypedQuotientSolveError("typed blocks do not cover all ten class pairs")
    observed_masses = dict.fromkeys(CLASS_PAIRS, 0)
    for row in rows:
        observed_masses[row.class_pair] += row.measured_flip_mass
    if observed_masses != masses:
        raise TypedQuotientSolveError(
            "typed block flip masses do not reconcile to measured class-pair totals"
        )
    atlas_hashes = {row.atlas_receipt_sha256 for row in rows}
    race_hashes = {row.coder_race_receipt_sha256 for row in rows}
    if len(atlas_hashes) != 1 or len(race_hashes) != 1:
        raise TypedQuotientSolveError("typed atlas/race rows must share one custodied source")
    return TypedAtlasValidation(
        valid=True,
        block_count=len(rows),
        class_pair_count=len(covered),
        total_measured_flip_mass=sum(masses.values()),
        atlas_receipt_sha256=next(iter(atlas_hashes)),
        coder_race_receipt_sha256=next(iter(race_hashes)),
    )


def validate_alternation_trace(
    stages: Sequence[AlternationStage | str],
    *,
    pose_tube_active_each_iteration: bool,
    real_coder_price_inside_objective: bool,
) -> int:
    """Validate complete cell -> lattice -> real-coder cycles."""

    trace = tuple(AlternationStage(value) for value in stages)
    expected = (
        AlternationStage.ARGMAX_CELL,
        AlternationStage.WITHIN_CELL_LATTICE,
        AlternationStage.REAL_CODER_PRICE,
    )
    if not trace or len(trace) % len(expected) != 0:
        raise TypedQuotientSolveError("alternation trace must contain complete three-stage cycles")
    for offset in range(0, len(trace), len(expected)):
        if trace[offset : offset + len(expected)] != expected:
            raise TypedQuotientSolveError("typed subproblems are out of canonical order")
    if pose_tube_active_each_iteration is not True:
        raise TypedQuotientSolveError("Pose tube must be active inside every member-selection cycle")
    if real_coder_price_inside_objective is not True:
        raise TypedQuotientSolveError("real coder price must be inside the solve objective")
    return len(trace) // len(expected)


def validate_geometry_ladder(geometry_ids: Sequence[str]) -> tuple[str, ...]:
    """Require the most-optimal measured geometry first; controls come later."""

    ladder = tuple(str(value) for value in geometry_ids)
    if not ladder or ladder[0] != "MEASURED_SCORER_SECOND_ORDER":
        raise TypedQuotientSolveError("geometry ladder must begin at MEASURED_SCORER_SECOND_ORDER")
    if "IDENTITY_EUCLIDEAN_CONTROL" in ladder and ladder.index("IDENTITY_EUCLIDEAN_CONTROL") == 0:
        raise TypedQuotientSolveError("identity Euclidean may only be a later control")
    return ladder


def solve_metric_active_block(
    operator: CouplingOperator,
    *,
    geometry: MeasuredScorerGeometry,
    quanta: Sequence[EffectiveQuantum],
    maximum_integer_steps: int = 8,
) -> KKTStep:
    """Solve one visible block in the measured second-order scorer metric."""

    if geometry.dimension != operator.matrix.shape[1]:
        raise TypedQuotientSolveError("scorer geometry and quotient dimensions differ")
    quantum_rows = tuple(quanta)
    if tuple(row.dof_label for row in quantum_rows) != operator.dof_labels:
        raise TypedQuotientSolveError("effective quanta must match visible DOF labels in order")
    if (
        isinstance(maximum_integer_steps, bool)
        or not isinstance(maximum_integer_steps, int)
        or maximum_integer_steps < 1
    ):
        raise TypedQuotientSolveError("maximum_integer_steps must be a positive integer")
    trust_radius = np.asarray(
        [row.effective_quantum * maximum_integer_steps for row in quantum_rows],
        dtype=np.float64,
    )
    # The measured metric is already second-order.  Damping is exactly zero:
    # identity damping would silently reintroduce Euclidean geometry.
    return solve_active_set_kkt(
        operator,
        description_metric=geometry.second_order_metric,
        damping=0.0,
        trust_radius=trust_radius,
        use_gauss_newton=False,
    )


@dataclass(frozen=True, slots=True)
class ExactMetricSieveResult:
    integer_coefficients: np.ndarray | None
    realized_step: np.ndarray | None
    metric_objective: float | None
    counted_bytes: int | None
    evaluated_candidates: int
    feasible_candidates: int
    search_complete: bool
    node_limit: int
    method: str = "BOUNDED_EXHAUSTIVE_METRIC_SIEVE"

    def __post_init__(self) -> None:
        if self.integer_coefficients is not None:
            object.__setattr__(
                self,
                "integer_coefficients",
                _immutable(self.integer_coefficients, dtype=np.int64),
            )
        if self.realized_step is not None:
            object.__setattr__(self, "realized_step", _immutable(self.realized_step))


def bounded_exact_metric_sieve(
    target_step: np.ndarray,
    *,
    geometry: MeasuredScorerGeometry,
    quanta: Sequence[EffectiveQuantum],
    integer_radius: int,
    node_limit: int,
    feasible: Callable[[np.ndarray], bool],
    real_coder_bytes: Callable[[np.ndarray], int],
) -> ExactMetricSieveResult:
    """Exhaust one finite lattice box using metric error plus exact byte price.

    ``feasible`` must enforce both argmax-cell membership and the Pose tube.
    ``real_coder_bytes`` must invoke a real parse-back-checked coder.  A
    truncated search is useful as a first rung but can never support a
    lattice-family negative because ``search_complete`` is false.
    """

    target = np.asarray(target_step, dtype=np.float64)
    quantum_rows = tuple(quanta)
    if target.shape != (geometry.dimension,) or not np.isfinite(target).all():
        raise TypedQuotientSolveError("target_step must be a finite quotient vector")
    if len(quantum_rows) != target.size:
        raise TypedQuotientSolveError("one effective quantum is required per quotient dimension")
    if isinstance(integer_radius, bool) or not isinstance(integer_radius, int) or integer_radius < 0:
        raise TypedQuotientSolveError("integer_radius must be a nonnegative integer")
    if isinstance(node_limit, bool) or not isinstance(node_limit, int) or node_limit < 1:
        raise TypedQuotientSolveError("node_limit must be a positive integer")
    quantum = np.asarray([row.effective_quantum for row in quantum_rows], dtype=np.float64)
    center = np.rint(target / quantum).astype(np.int64)
    ranges = tuple(range(int(value) - integer_radius, int(value) + integer_radius + 1) for value in center)
    total_nodes = math.prod(len(values) for values in ranges)
    evaluated = 0
    feasible_count = 0
    best_key: tuple[float, int, tuple[int, ...]] | None = None
    best_coefficients: np.ndarray | None = None
    best_step: np.ndarray | None = None
    best_bytes: int | None = None
    metric = geometry.second_order_metric
    for coefficients_tuple in itertools.product(*ranges):
        if evaluated >= node_limit:
            break
        evaluated += 1
        coefficients = np.asarray(coefficients_tuple, dtype=np.int64)
        candidate = coefficients.astype(np.float64) * quantum
        if feasible(candidate) is not True:
            continue
        feasible_count += 1
        counted = real_coder_bytes(candidate)
        if isinstance(counted, bool) or not isinstance(counted, int) or counted < 0:
            raise TypedQuotientSolveError("real_coder_bytes must return an exact nonnegative integer")
        error = candidate - target
        objective = float(
            0.5 * error @ metric @ error + RATE_SCORE_PER_BYTE * counted
        )
        key = (objective, counted, tuple(int(value) for value in coefficients))
        if best_key is None or key < best_key:
            best_key = key
            best_coefficients = coefficients
            best_step = candidate
            best_bytes = counted
    return ExactMetricSieveResult(
        integer_coefficients=best_coefficients,
        realized_step=best_step,
        metric_objective=(best_key[0] if best_key is not None else None),
        counted_bytes=best_bytes,
        evaluated_candidates=evaluated,
        feasible_candidates=feasible_count,
        search_complete=evaluated == total_nodes,
        node_limit=node_limit,
    )


@dataclass(frozen=True, slots=True)
class GeneralizedDictionaryUpdate:
    coefficients: np.ndarray
    basis_rows: np.ndarray
    reconstruction: np.ndarray
    weighted_residual_squared: float
    metric_orthonormality_error: float
    rank: int
    method: str = "MEASURED_METRIC_GENERALIZED_SVD_LS"

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", _immutable(self.coefficients))
        object.__setattr__(self, "basis_rows", _immutable(self.basis_rows))
        object.__setattr__(self, "reconstruction", _immutable(self.reconstruction))


def generalized_metric_dictionary_update(
    samples: np.ndarray,
    *,
    geometry: MeasuredScorerGeometry,
    rank: int,
) -> GeneralizedDictionaryUpdate:
    """Return the deterministic rank-r LS update in the scorer metric.

    If ``G=L L^T``, SVD is applied to ``X L`` and mapped back with
    ``B = V_r^T L^{-1}``.  Thus ``B G B^T = I`` and the update is oblique in
    raw coordinates rather than a plain Euclidean SVD.
    """

    values = np.asarray(samples, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != geometry.dimension
        or not np.isfinite(values).all()
    ):
        raise TypedQuotientSolveError("samples must be finite NxD scorer-coordinate rows")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= min(values.shape):
        raise TypedQuotientSolveError("dictionary rank is outside the sample matrix range")
    metric = geometry.second_order_metric
    try:
        factor = np.linalg.cholesky(metric)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - constructor already proves this.
        raise TypedQuotientSolveError("measured scorer metric is not positive definite") from exc
    whitened = values @ factor
    u, singular, vt = np.linalg.svd(whitened, full_matrices=False)
    coefficients = u[:, :rank] * singular[:rank]
    basis_rows = np.linalg.solve(factor.T, vt[:rank].T).T

    # Fix SVD sign ambiguity so checkpoint/resume is byte deterministic.
    for index in range(rank):
        pivot = int(np.argmax(np.abs(basis_rows[index])))
        if basis_rows[index, pivot] < 0.0:
            basis_rows[index] *= -1.0
            coefficients[:, index] *= -1.0
    reconstruction = coefficients @ basis_rows
    residual = values - reconstruction
    weighted_residual = float(np.einsum("ni,ij,nj->", residual, metric, residual))
    orthonormality = basis_rows @ metric @ basis_rows.T
    orthonormality_error = float(np.max(np.abs(orthonormality - np.eye(rank)), initial=0.0))
    return GeneralizedDictionaryUpdate(
        coefficients=coefficients,
        basis_rows=basis_rows,
        reconstruction=reconstruction,
        weighted_residual_squared=weighted_residual,
        metric_orthonormality_error=orthonormality_error,
        rank=rank,
    )


__all__ = [
    "CLASS_PAIRS",
    "EVIDENCE_AXIS",
    "METRIC_COORDINATE_SYSTEM",
    "N_CLASSES",
    "RATE_SCORE_PER_BYTE",
    "STREAM_TYPE_CONTRACT_MODULE",
    "AlternationStage",
    "EffectiveQuantum",
    "EvaluationRecursionLevel",
    "ExactMetricSieveResult",
    "G4TemporalClass",
    "GeneralizedDictionaryUpdate",
    "MeasuredScorerGeometry",
    "ScorerVisibility",
    "TypedAtlasValidation",
    "TypedBlock",
    "TypedQuotientSolveError",
    "bounded_exact_metric_sieve",
    "canonical_stream_type_value",
    "generalized_metric_dictionary_update",
    "solve_metric_active_block",
    "validate_alternation_trace",
    "validate_geometry_ladder",
    "validate_typed_atlas",
]
