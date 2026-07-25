# SPDX-License-Identifier: MIT
"""Fisher-metric multiple shooting for DDM feature-relay solves.

The solver in this module is deliberately separated from evaluator
acceptance.  It solves a local, station-wise quadratic model with explicit
continuity constraints.  Predicted station reductions are diagnostics only;
the only admission helper in this module requires an end-to-end realized
receiver/scorer row.

The intended station chain is

``#580 range(A) input -> block2 PRE-SE -> block3 PRE-SE -> rank-4 head``.

No Euclidean station objective is accepted.  Euclidean norms are emitted only
as labeled controls beside the primary categorical margin-Fisher geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

SCHEMA: Final = "ddm_feature_relay_multiple_shooting.v1"
METRIC_KIND: Final = "categorical_margin_fisher_gram"
DIRECT_METHOD: Final = "single_composed_final_station_linearization"
RELAY_METHOD: Final = "stationwise_kkt_multiple_shooting_exact_continuity"
ACCEPTANCE_AUTHORITY: Final = (
    "END_ONLY_REALIZED_RECEIVER_PARSEBACK_UINT8_R_FROZEN_SCORERS"
)


class FeatureRelayError(ValueError):
    """Fail-closed malformed feature-relay geometry or evidence."""


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise FeatureRelayError(f"{name} must be a nonempty finite vector")
    return np.ascontiguousarray(vector)


def _finite_matrix(
    value: Any,
    *,
    name: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.size == 0
        or not np.all(np.isfinite(matrix))
        or (shape is not None and matrix.shape != shape)
    ):
        expected = "" if shape is None else f" with shape {shape}"
        raise FeatureRelayError(f"{name} must be a nonempty finite matrix{expected}")
    return np.ascontiguousarray(matrix)


def _psd_matrix(value: Any, *, name: str, dimension: int) -> np.ndarray:
    matrix = _finite_matrix(value, name=name, shape=(dimension, dimension))
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
    tolerance = np.finfo(np.float64).eps * dimension * scale * 64.0
    if (
        not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance)
        or float(np.linalg.eigvalsh((matrix + matrix.T) * 0.5).min())
        < -tolerance
    ):
        raise FeatureRelayError(f"{name} must be symmetric positive semidefinite")
    return np.ascontiguousarray((matrix + matrix.T) * 0.5)


@dataclass(frozen=True, slots=True)
class RelayStationV1:
    """One measured feature station and its primary Fisher geometry."""

    station_id: str
    layer_path: str
    target_delta: np.ndarray
    metric_gram: np.ndarray
    metric_kind: str
    evidence_sha256: str
    measurement_status: str = "MEASURED"

    def __post_init__(self) -> None:
        target = _finite_vector(self.target_delta, name=f"{self.station_id}.target_delta")
        metric = _psd_matrix(
            self.metric_gram,
            name=f"{self.station_id}.metric_gram",
            dimension=target.size,
        )
        if (
            not self.station_id
            or not self.layer_path
            or self.metric_kind != METRIC_KIND
            or self.measurement_status != "MEASURED"
            or len(self.evidence_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.evidence_sha256)
            or float(np.linalg.norm(metric, ord="fro")) <= 0.0
        ):
            raise FeatureRelayError(f"{self.station_id or 'station'} custody differs")
        object.__setattr__(self, "target_delta", target)
        object.__setattr__(self, "metric_gram", metric)

    @property
    def dimension(self) -> int:
        return int(self.target_delta.size)


@dataclass(frozen=True, slots=True)
class RelaySegmentV1:
    """One measured local segment Jacobian in the relay chain."""

    segment_id: str
    source_id: str
    target_id: str
    jacobian: np.ndarray
    evidence_sha256: str
    measurement_status: str = "MEASURED"

    def __post_init__(self) -> None:
        jacobian = _finite_matrix(self.jacobian, name=f"{self.segment_id}.jacobian")
        if (
            not self.segment_id
            or not self.source_id
            or not self.target_id
            or self.source_id == self.target_id
            or self.measurement_status != "MEASURED"
            or len(self.evidence_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.evidence_sha256)
        ):
            raise FeatureRelayError(f"{self.segment_id or 'segment'} custody differs")
        object.__setattr__(self, "jacobian", jacobian)


@dataclass(frozen=True, slots=True)
class RelayProblemV1:
    """A three-station relay problem with one #580-projected actuator."""

    stations: tuple[RelayStationV1, ...]
    segments: tuple[RelaySegmentV1, ...]
    actuator_dimension: int
    actuator_metric: np.ndarray
    input_station_id: str = "range_a_input"

    def __post_init__(self) -> None:
        if (
            len(self.stations) not in {2, 3}
            or len(self.segments) != len(self.stations)
            or isinstance(self.actuator_dimension, bool)
            or self.actuator_dimension <= 0
        ):
            raise FeatureRelayError("relay requires two or three stations and one segment per station")
        actuator_metric = _psd_matrix(
            self.actuator_metric,
            name="actuator_metric",
            dimension=self.actuator_dimension,
        )
        station_ids = tuple(station.station_id for station in self.stations)
        if len(set(station_ids)) != len(station_ids):
            raise FeatureRelayError("relay station IDs must be unique")
        expected_source = self.input_station_id
        expected_source_dimension = self.actuator_dimension
        for station, segment in zip(self.stations, self.segments, strict=True):
            if (
                segment.source_id != expected_source
                or segment.target_id != station.station_id
                or segment.jacobian.shape
                != (station.dimension, expected_source_dimension)
            ):
                raise FeatureRelayError(
                    f"segment chain differs at {segment.segment_id}: "
                    f"{segment.jacobian.shape} does not map "
                    f"{expected_source}->{station.station_id}"
                )
            expected_source = station.station_id
            expected_source_dimension = station.dimension
        object.__setattr__(self, "actuator_metric", actuator_metric)


def _block_offsets(problem: RelayProblemV1) -> tuple[int, tuple[int, ...]]:
    actuator_end = problem.actuator_dimension
    offsets: list[int] = []
    cursor = actuator_end
    for station in problem.stations:
        offsets.append(cursor)
        cursor += station.dimension
    return cursor, tuple(offsets)


def _continuity_matrix(problem: RelayProblemV1) -> np.ndarray:
    variable_count, offsets = _block_offsets(problem)
    row_count = sum(station.dimension for station in problem.stations)
    constraints = np.zeros((row_count, variable_count), dtype=np.float64)
    row = 0
    source_offset = 0
    source_dimension = problem.actuator_dimension
    for station, segment, target_offset in zip(
        problem.stations,
        problem.segments,
        offsets,
        strict=True,
    ):
        stop = row + station.dimension
        constraints[row:stop, target_offset : target_offset + station.dimension] = np.eye(
            station.dimension,
            dtype=np.float64,
        )
        constraints[row:stop, source_offset : source_offset + source_dimension] = (
            -segment.jacobian
        )
        row = stop
        source_offset = target_offset
        source_dimension = station.dimension
    return constraints


def _objective_terms(problem: RelayProblemV1) -> tuple[np.ndarray, np.ndarray]:
    variable_count, offsets = _block_offsets(problem)
    quadratic = np.zeros((variable_count, variable_count), dtype=np.float64)
    linear = np.zeros(variable_count, dtype=np.float64)
    quadratic[: problem.actuator_dimension, : problem.actuator_dimension] = (
        problem.actuator_metric
    )
    for station, offset in zip(problem.stations, offsets, strict=True):
        stop = offset + station.dimension
        quadratic[offset:stop, offset:stop] = station.metric_gram
        linear[offset:stop] = -(station.metric_gram @ station.target_delta)
    return quadratic, linear


def _minimum_norm_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    cutoff = np.finfo(np.float64).eps * max(matrix.shape)
    solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=cutoff)
    return np.ascontiguousarray(solution)


def _metric_energy(vector: np.ndarray, metric: np.ndarray) -> float:
    return float(0.5 * vector @ metric @ vector)


def solve_multiple_shooting(problem: RelayProblemV1) -> dict[str, Any]:
    """Solve the station-wise KKT system with exact linear continuity.

    Singular Fisher directions are handled by the machine-epsilon
    minimum-norm pseudoinverse.  No damping, learning rate, shrink factor, or
    Euclidean station objective is introduced.
    """

    quadratic, linear = _objective_terms(problem)
    continuity = _continuity_matrix(problem)
    variable_count = quadratic.shape[0]
    constraint_count = continuity.shape[0]
    kkt = np.block(
        [
            [quadratic, continuity.T],
            [continuity, np.zeros((constraint_count, constraint_count), dtype=np.float64)],
        ]
    )
    rhs = np.concatenate((-linear, np.zeros(constraint_count, dtype=np.float64)))
    solution = _minimum_norm_solve(kkt, rhs)
    primal = solution[:variable_count]
    residual = continuity @ primal
    scale = max(1.0, float(np.linalg.norm(continuity) * np.linalg.norm(primal)))
    tolerance = (
        np.finfo(np.float64).eps
        * max(kkt.shape)
        * scale
        * 512.0
    )
    if float(np.linalg.norm(residual)) > tolerance:
        raise FeatureRelayError(
            "multiple-shooting continuity residual exceeds fp64 derived tolerance"
        )

    _, offsets = _block_offsets(problem)
    actuator = np.ascontiguousarray(primal[: problem.actuator_dimension])
    station_rows: list[dict[str, Any]] = []
    for station, offset in zip(problem.stations, offsets, strict=True):
        state = np.ascontiguousarray(primal[offset : offset + station.dimension])
        before = _metric_energy(station.target_delta, station.metric_gram)
        remaining = station.target_delta - state
        after = _metric_energy(remaining, station.metric_gram)
        station_rows.append(
            {
                "station_id": station.station_id,
                "layer_path": station.layer_path,
                "metric_kind": station.metric_kind,
                "evidence_sha256": station.evidence_sha256,
                "predicted_fisher_debt_before": before,
                "predicted_fisher_debt_after": after,
                "predicted_fisher_reduction": before - after,
                "euclidean_control_only": {
                    "state_l2": float(np.linalg.norm(state)),
                    "target_delta_l2": float(np.linalg.norm(station.target_delta)),
                    "remaining_l2": float(np.linalg.norm(remaining)),
                },
                "state_delta": state.tolist(),
            }
        )
    return {
        "schema": SCHEMA,
        "method": RELAY_METHOD,
        "metric_primary": METRIC_KIND,
        "euclidean_authority": False,
        "acceptance_authority": ACCEPTANCE_AUTHORITY,
        "actuator_delta": actuator.tolist(),
        "actuator_fisher_cost": _metric_energy(actuator, problem.actuator_metric),
        "continuity_residual_l2": float(np.linalg.norm(residual)),
        "continuity_tolerance_fp64_derived": tolerance,
        "station_rows": station_rows,
        "predicted_only": True,
        "realized_acceptance": None,
    }


def solve_direct_final_station(problem: RelayProblemV1) -> dict[str, Any]:
    """Solve the one-shot composed final-station control for comparison."""

    composed = problem.segments[0].jacobian
    for segment in problem.segments[1:]:
        composed = segment.jacobian @ composed
    final = problem.stations[-1]
    normal = problem.actuator_metric + composed.T @ final.metric_gram @ composed
    rhs = composed.T @ final.metric_gram @ final.target_delta
    actuator = _minimum_norm_solve(normal, rhs)
    predicted = composed @ actuator
    before = _metric_energy(final.target_delta, final.metric_gram)
    after = _metric_energy(final.target_delta - predicted, final.metric_gram)
    return {
        "schema": SCHEMA,
        "method": DIRECT_METHOD,
        "metric_primary": METRIC_KIND,
        "euclidean_authority": False,
        "acceptance_authority": ACCEPTANCE_AUTHORITY,
        "actuator_delta": actuator.tolist(),
        "final_station_id": final.station_id,
        "predicted_final_fisher_debt_before": before,
        "predicted_final_fisher_debt_after": after,
        "predicted_final_fisher_reduction": before - after,
        "euclidean_control_only": {
            "actuator_l2": float(np.linalg.norm(actuator)),
            "predicted_final_delta_l2": float(np.linalg.norm(predicted)),
        },
        "predicted_only": True,
        "realized_acceptance": None,
    }


def realized_joint_delta(
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float | str]:
    """Return the exact contest-formula component delta for one realized row."""

    required = ("d_seg", "d_pose", "archive_bytes")
    for name, row in (("reference", reference), ("candidate", candidate)):
        for key in required:
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise FeatureRelayError(f"{name}.{key} is not a realized nonnegative scalar")
        if (
            isinstance(row.get("archive_bytes"), bool)
            or not isinstance(row.get("archive_bytes"), int)
        ):
            raise FeatureRelayError(f"{name}.archive_bytes must be an exact integer")
        if row.get("receiver_parseback_exact") is not True:
            raise FeatureRelayError(f"{name} lacks exact receiver parse-back")
        if (
            row.get("realized_through_r_uint8") is not True
            or row.get("frozen_scorers") is not True
        ):
            raise FeatureRelayError(f"{name} lacks realized-through-R scorer custody")
        if row.get("num_pairs") != 600 or row.get("score_claim") is not False:
            raise FeatureRelayError(f"{name} is not an advisory n600 realized row")
    seg = 100.0 * (float(candidate["d_seg"]) - float(reference["d_seg"]))
    pose = math.sqrt(10.0 * float(candidate["d_pose"])) - math.sqrt(
        10.0 * float(reference["d_pose"])
    )
    rate = (
        25.0
        * (float(candidate["archive_bytes"]) - float(reference["archive_bytes"]))
        / 37_545_489.0
    )
    return {
        "acceptance_authority": ACCEPTANCE_AUTHORITY,
        "seg_term": seg,
        "pose_term": pose,
        "rate_term": rate,
        "joint_delta": seg + pose + rate,
    }


def admit_realized_endpoint(
    *,
    method: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit only a strict negative end-to-end realized joint delta."""

    if method not in {DIRECT_METHOD, RELAY_METHOD}:
        raise FeatureRelayError("realized method identity differs")
    delta = realized_joint_delta(reference=reference, candidate=candidate)
    accepted = float(delta["joint_delta"]) < 0.0
    return {
        "schema": SCHEMA,
        "method": method,
        "accepted": accepted,
        "decision": "ACCEPT_REALIZED_END" if accepted else "REJECT_REALIZED_END",
        "delta": delta,
        "intermediate_prediction_used_for_acceptance": False,
        "acceptance_authority": ACCEPTANCE_AUTHORITY,
    }


def compare_realized_radius(
    *,
    direct_rows: Sequence[Mapping[str, Any]],
    relay_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare measured accepted radii at an equal end-verdict budget.

    Each row must be a fully realized endpoint admission carrying an integer
    ``radius_quanta``.  Predicted station rows cannot enter this comparison.
    """

    if not direct_rows or len(direct_rows) != len(relay_rows):
        raise FeatureRelayError("radius comparison requires equal nonempty verdict budgets")

    def _validated_rows(
        rows: Sequence[Mapping[str, Any]],
        method: str,
    ) -> dict[int, bool]:
        decisions: dict[int, bool] = {}
        for row in rows:
            radius = row.get("radius_quanta")
            if (
                row.get("schema") != SCHEMA
                or row.get("method") != method
                or row.get("acceptance_authority") != ACCEPTANCE_AUTHORITY
                or row.get("intermediate_prediction_used_for_acceptance") is not False
                or isinstance(radius, bool)
                or not isinstance(radius, int)
                or radius <= 0
                or radius in decisions
                or not isinstance(row.get("accepted"), bool)
            ):
                raise FeatureRelayError("radius row lacks realized endpoint custody")
            decisions[radius] = bool(row["accepted"])
        return decisions

    direct_decisions = _validated_rows(direct_rows, DIRECT_METHOD)
    relay_decisions = _validated_rows(relay_rows, RELAY_METHOD)
    if set(direct_decisions) != set(relay_decisions):
        raise FeatureRelayError("radius comparison requires the same realized radius ladder")

    def _accepted_prefix_radius(decisions: Mapping[int, bool]) -> int:
        accepted_radius = 0
        for radius in sorted(decisions):
            if not decisions[radius]:
                break
            accepted_radius = radius
        return accepted_radius

    direct_radius = _accepted_prefix_radius(direct_decisions)
    relay_radius = _accepted_prefix_radius(relay_decisions)
    return {
        "schema": SCHEMA,
        "equal_realized_verdict_budget": len(direct_rows),
        "direct_radius_quanta": direct_radius,
        "relay_radius_quanta": relay_radius,
        "relay_minus_direct_radius_quanta": relay_radius - direct_radius,
        "relay_beats_direct": relay_radius > direct_radius,
        "verdict": (
            "RELAY_RADIUS_GT_DIRECT"
            if relay_radius > direct_radius
            else "RELAY_RADIUS_LE_DIRECT_FORMULATION_SCOPED"
        ),
        "acceptance_authority": ACCEPTANCE_AUTHORITY,
    }


__all__ = [
    "ACCEPTANCE_AUTHORITY",
    "DIRECT_METHOD",
    "METRIC_KIND",
    "RELAY_METHOD",
    "FeatureRelayError",
    "RelayProblemV1",
    "RelaySegmentV1",
    "RelayStationV1",
    "admit_realized_endpoint",
    "compare_realized_radius",
    "realized_joint_delta",
    "solve_direct_final_station",
    "solve_multiple_shooting",
]
