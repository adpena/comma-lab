# SPDX-License-Identifier: MIT
"""Pure custody primitives for measured receiver-closed rank-4 corrections.

This module never loads or invokes a scorer.  Measurement runners supply fresh
candidate-state first-order and finite-secant rows.  The code here validates
those rows, keeps trust regions isolated by target class and pre-step margin
bucket, and solves a deterministic minimum-norm convex inequality problem in a
chart of dimension at most four.  A solver status is never an admission: the
calling runner must round-trip the actual packet and rerun its hard oracle.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np

RECEIPT_SCHEMA: Final = "realized_secant_custody_receipt.v1"
PACKET_MAGIC: Final = b"G2ES1"
PACKET_HEADER: Final = struct.Struct(">5sB")


class RealizedSecantCustodyError(ValueError):
    """Refuse malformed, pooled, nonfinite, or under-custodied evidence."""


class QPStatus(StrEnum):
    SOLVED = "SOLVED"
    INFEASIBLE = "INFEASIBLE"


class PairSolveStatus(StrEnum):
    """Terminal receiver-closed disposition for one measured pair."""

    TRUST_REGION_REFUSED = "TRUST_REGION_REFUSED"
    QP_INFEASIBLE = "QP_INFEASIBLE"
    NEGATIVE_REALIZED_HARD_ORACLE_REFUSED = "NEGATIVE_REALIZED_HARD_ORACLE_REFUSED"
    RATE_BREAK_EVEN_REFUSED = "RATE_BREAK_EVEN_REFUSED"
    KKT_RESIDUAL_REFUSED = "KKT_RESIDUAL_REFUSED"
    DOUBLE_DECODE_REFUSED = "DOUBLE_DECODE_REFUSED"
    ADMITTED_RECEIVER_CLOSED = "ADMITTED_RECEIVER_CLOSED"


TERMINAL_PAIR_STATUSES: Final = frozenset(status.value for status in PairSolveStatus)


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RealizedSecantCustodyError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise RealizedSecantCustodyError(f"{label} must be finite")
    return result


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RealizedSecantCustodyError(f"{label} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise RealizedSecantCustodyError(f"{label} must be >= {minimum}")
    return result


def _finite_vector(value: Any, label: str, *, size: int | None = None) -> tuple[float, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise RealizedSecantCustodyError(f"{label} must be a one-dimensional real vector")
    vector = array.astype(np.float64, copy=False)
    if size is not None and vector.size != size:
        raise RealizedSecantCustodyError(f"{label} must have exactly {size} values")
    if not np.isfinite(vector).all():
        raise RealizedSecantCustodyError(f"{label} must be finite")
    return tuple(float(item) for item in vector)


@dataclass(frozen=True)
class WriteSecantObservation:
    """One declared-write response inside a pair/column finite-secant row."""

    ordinal: int
    target_class: int
    current_class: int
    pre_margin: float
    margin_bucket: str
    expected_sign: int
    feature_displacement: tuple[float, ...]
    predicted_margin_delta: float
    realized_margin_delta: float
    secant_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordinal", _exact_int(self.ordinal, "ordinal"))
        object.__setattr__(self, "target_class", _exact_int(self.target_class, "target_class"))
        object.__setattr__(self, "current_class", _exact_int(self.current_class, "current_class"))
        object.__setattr__(self, "pre_margin", _finite_scalar(self.pre_margin, "pre_margin"))
        if not isinstance(self.margin_bucket, str) or not self.margin_bucket:
            raise RealizedSecantCustodyError("margin_bucket must be nonempty")
        if isinstance(self.expected_sign, bool) or self.expected_sign not in (-1, 1):
            raise RealizedSecantCustodyError("expected_sign must be exactly -1 or +1")
        object.__setattr__(
            self,
            "feature_displacement",
            _finite_vector(self.feature_displacement, "feature_displacement", size=144),
        )
        for field in ("predicted_margin_delta", "realized_margin_delta", "secant_ratio"):
            object.__setattr__(self, field, _finite_scalar(getattr(self, field), field))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WriteSecantObservation:
        try:
            return cls(
                ordinal=value["ordinal"],
                target_class=value["target_class"],
                current_class=value["current_class"],
                pre_margin=value["pre_margin"],
                margin_bucket=value["margin_bucket"],
                expected_sign=value["expected_sign"],
                feature_displacement=tuple(value["feature_displacement"]),
                predicted_margin_delta=value["predicted_margin_delta"],
                realized_margin_delta=value["realized_margin_delta"],
                secant_ratio=value["secant_ratio"],
            )
        except (KeyError, TypeError) as exc:
            raise RealizedSecantCustodyError("malformed declared-write secant row") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "target_class": self.target_class,
            "current_class": self.current_class,
            "pre_margin": self.pre_margin,
            "margin_bucket": self.margin_bucket,
            "expected_sign": self.expected_sign,
            "feature_displacement": list(self.feature_displacement),
            "predicted_margin_delta": self.predicted_margin_delta,
            "realized_margin_delta": self.realized_margin_delta,
            "secant_ratio": self.secant_ratio,
        }


@dataclass(frozen=True)
class SecantObservation:
    """Exactly one independent receiver observation for a pair/chart column."""

    pair_index: int
    column_index: int
    signed_amplitude: float
    applied_rgb_l2: float
    applied_rgb_linf: float
    uint8_saturation_count: int
    writes: tuple[WriteSecantObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_index", _exact_int(self.pair_index, "pair_index"))
        object.__setattr__(self, "column_index", _exact_int(self.column_index, "column_index"))
        amplitude = _finite_scalar(self.signed_amplitude, "signed_amplitude")
        if amplitude == 0.0:
            raise RealizedSecantCustodyError("signed_amplitude must be nonzero")
        object.__setattr__(self, "signed_amplitude", amplitude)
        for field in ("applied_rgb_l2", "applied_rgb_linf"):
            value = _finite_scalar(getattr(self, field), field)
            if value < 0.0:
                raise RealizedSecantCustodyError(f"{field} must be nonnegative")
            object.__setattr__(self, field, value)
        object.__setattr__(
            self,
            "uint8_saturation_count",
            _exact_int(self.uint8_saturation_count, "uint8_saturation_count"),
        )
        if not isinstance(self.writes, tuple) or not self.writes:
            raise RealizedSecantCustodyError("a secant row must contain declared writes")
        if any(not isinstance(row, WriteSecantObservation) for row in self.writes):
            raise RealizedSecantCustodyError("writes must contain typed observations")
        ordinals = [row.ordinal for row in self.writes]
        if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
            raise RealizedSecantCustodyError("declared-write ordinals must be unique and sorted")
        for row in self.writes:
            expected_ratio = row.realized_margin_delta / amplitude
            if not math.isclose(row.secant_ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-12):
                raise RealizedSecantCustodyError("secant_ratio does not match realized delta/amplitude")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SecantObservation:
        try:
            return cls(
                pair_index=value["pair_index"],
                column_index=value["column_index"],
                signed_amplitude=value["signed_amplitude"],
                applied_rgb_l2=value["applied_rgb_l2"],
                applied_rgb_linf=value["applied_rgb_linf"],
                uint8_saturation_count=value["uint8_saturation_count"],
                writes=tuple(WriteSecantObservation.from_dict(row) for row in value["writes"]),
            )
        except (KeyError, TypeError) as exc:
            raise RealizedSecantCustodyError("malformed pair/column secant row") from exc

    def as_dict(self) -> dict[str, Any]:
        value = {
            "pair_index": self.pair_index,
            "column_index": self.column_index,
            "signed_amplitude": self.signed_amplitude,
            "applied_rgb_l2": self.applied_rgb_l2,
            "applied_rgb_linf": self.applied_rgb_linf,
            "uint8_saturation_count": self.uint8_saturation_count,
            "writes": [row.as_dict() for row in self.writes],
        }
        value["row_sha256"] = canonical_sha256(value)
        return value


@dataclass(frozen=True)
class TrustRegion:
    target_class: int
    margin_bucket: str
    observation_count: int
    max_relative_residual: float
    min_abs_signed_response: float
    usable: bool
    refusal_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_class": self.target_class,
            "margin_bucket": self.margin_bucket,
            "observation_count": self.observation_count,
            "max_relative_residual": self.max_relative_residual,
            "min_abs_signed_response": self.min_abs_signed_response,
            "usable": self.usable,
            "refusal_reasons": list(self.refusal_reasons),
        }


def build_trust_regions(
    observations: Sequence[SecantObservation],
    *,
    relative_residual_tolerance: float,
    response_epsilon: float = 1e-12,
) -> tuple[TrustRegion, ...]:
    """Validate isolated class/bucket regions without cross-group pooling."""

    tolerance = _finite_scalar(relative_residual_tolerance, "relative_residual_tolerance")
    epsilon = _finite_scalar(response_epsilon, "response_epsilon")
    if tolerance < 0.0 or epsilon <= 0.0:
        raise RealizedSecantCustodyError("trust tolerances must be nonnegative/positive")
    grouped: dict[tuple[int, str], list[WriteSecantObservation]] = defaultdict(list)
    for observation in observations:
        if not isinstance(observation, SecantObservation):
            raise RealizedSecantCustodyError("trust input must contain typed secant observations")
        for row in observation.writes:
            grouped[(row.target_class, row.margin_bucket)].append(row)
    if not grouped:
        raise RealizedSecantCustodyError("trust construction requires observations")

    result: list[TrustRegion] = []
    for (target_class, bucket), rows in sorted(grouped.items()):
        reasons: set[str] = set()
        residuals: list[float] = []
        signed_responses: list[float] = []
        for row in rows:
            predicted_signed = row.expected_sign * row.predicted_margin_delta
            realized_signed = row.expected_sign * row.realized_margin_delta
            signed_responses.append(realized_signed)
            if predicted_signed <= epsilon:
                reasons.add("FIRST_ORDER_SIGN_OR_ZERO")
            if realized_signed <= epsilon:
                reasons.add("REALIZED_SIGN_OR_ZERO")
            denominator = max(abs(row.predicted_margin_delta), abs(row.realized_margin_delta), epsilon)
            residual = abs(row.realized_margin_delta - row.predicted_margin_delta) / denominator
            residuals.append(residual)
            if residual > tolerance:
                reasons.add("RELATIVE_SECANT_RESIDUAL")
        result.append(
            TrustRegion(
                target_class=target_class,
                margin_bucket=bucket,
                observation_count=len(rows),
                max_relative_residual=max(residuals),
                min_abs_signed_response=min(signed_responses),
                usable=not reasons,
                refusal_reasons=tuple(sorted(reasons)),
            )
        )
    return tuple(result)


def build_pair_trust_region_custody(
    observations: Sequence[SecantObservation],
    *,
    pair_count: int,
    relative_residual_tolerance: float,
) -> tuple[dict[str, Any], ...]:
    """Build canonical hashed trust rows without pooling across measured pairs."""

    pairs = _exact_int(pair_count, "pair_count", minimum=1)
    grouped: dict[int, list[SecantObservation]] = defaultdict(list)
    for observation in observations:
        if not isinstance(observation, SecantObservation):
            raise RealizedSecantCustodyError("trust custody requires typed secant observations")
        if observation.pair_index >= pairs:
            raise RealizedSecantCustodyError("trust custody observation pair is out of range")
        grouped[observation.pair_index].append(observation)
    if sorted(grouped) != list(range(pairs)):
        raise RealizedSecantCustodyError("trust custody requires observations for every pair")

    custody: list[dict[str, Any]] = []
    for pair_index in range(pairs):
        regions = build_trust_regions(
            grouped[pair_index],
            relative_residual_tolerance=relative_residual_tolerance,
        )
        for region in regions:
            row = {"pair_index": pair_index, **region.as_dict()}
            row["row_sha256"] = canonical_sha256(row)
            custody.append(row)
    return tuple(custody)


@dataclass(frozen=True)
class MinimalNormSolve:
    coefficients: tuple[float, ...]
    status: QPStatus
    active_rows: tuple[int, ...]
    max_primal_violation: float | None
    min_active_multiplier: float | None
    stationarity_residual: float | None
    objective: float | None
    candidate_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "coefficients": list(self.coefficients),
            "status": self.status.value,
            "active_rows": list(self.active_rows),
            "max_primal_violation": self.max_primal_violation,
            "min_active_multiplier": self.min_active_multiplier,
            "stationarity_residual": self.stationarity_residual,
            "objective": self.objective,
            "candidate_count": self.candidate_count,
        }


def solve_minimal_norm_inequalities(
    margin_jacobian: np.ndarray,
    required_margin_delta: np.ndarray,
    rgb_direction_matrix: np.ndarray,
    baseline_rgb: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> MinimalNormSolve:
    """Solve ``min 0.5||alpha||^2`` with margins and exact RGB box bounds.

    The chart dimension is bounded by four, so lexicographically enumerating
    all linearly independent active sets of size at most the chart dimension is
    deterministic and complete for this convex projection problem.  RGB box
    inequalities are included in the same KKT system; they are not post-hoc
    clipping constraints.
    """

    jacobian = np.asarray(margin_jacobian, dtype=np.float64)
    debt = np.asarray(required_margin_delta, dtype=np.float64)
    directions = np.asarray(rgb_direction_matrix, dtype=np.float64)
    baseline = np.asarray(baseline_rgb, dtype=np.float64)
    tol = _finite_scalar(tolerance, "tolerance")
    if jacobian.ndim != 2 or jacobian.shape[0] == 0 or not 1 <= jacobian.shape[1] <= 4:
        raise RealizedSecantCustodyError("margin Jacobian must have nonempty MxD shape with D<=4")
    dimension = jacobian.shape[1]
    if debt.shape != (jacobian.shape[0],):
        raise RealizedSecantCustodyError("required margin debt does not match Jacobian")
    if directions.ndim != 2 or directions.shape[1] != dimension or directions.shape[0] == 0:
        raise RealizedSecantCustodyError("RGB direction matrix must have shape PxD")
    if baseline.shape != (directions.shape[0],):
        raise RealizedSecantCustodyError("baseline RGB vector does not match direction rows")
    if not all(np.isfinite(value).all() for value in (jacobian, debt, directions, baseline)):
        raise RealizedSecantCustodyError("QP arrays must be finite")
    if np.any((baseline < 0.0) | (baseline > 255.0)) or tol <= 0.0:
        raise RealizedSecantCustodyError("baseline RGB must lie in [0,255] and tolerance must be positive")

    # G alpha >= h.  Rows are margin, lower RGB, then upper RGB constraints.
    matrix = np.concatenate((jacobian, directions, -directions), axis=0)
    rhs = np.concatenate((debt, -baseline, baseline - 255.0), axis=0)
    if np.all(rhs <= tol):
        coefficients = np.zeros(dimension, dtype=np.float64)
        return MinimalNormSolve(tuple(coefficients), QPStatus.SOLVED, (), 0.0, None, 0.0, 0.0, 1)

    candidates: list[tuple[float, tuple[float, ...], tuple[int, ...], np.ndarray, np.ndarray]] = []
    indices = range(matrix.shape[0])
    for size in range(1, dimension + 1):
        for active in itertools.combinations(indices, size):
            active_matrix = matrix[np.asarray(active)]
            if np.linalg.matrix_rank(active_matrix, tol=tol) != size:
                continue
            gram = active_matrix @ active_matrix.T
            try:
                multipliers = np.linalg.solve(gram, rhs[np.asarray(active)])
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(multipliers).all() or np.any(multipliers < -tol):
                continue
            multipliers = np.maximum(multipliers, 0.0)
            coefficients = active_matrix.T @ multipliers
            violation = rhs - matrix @ coefficients
            if float(np.max(violation)) > tol:
                continue
            objective = 0.5 * float(coefficients @ coefficients)
            candidates.append(
                (objective, tuple(float(value) for value in coefficients), active, coefficients, multipliers)
            )

    if not candidates:
        return MinimalNormSolve(
            tuple(0.0 for _ in range(dimension)),
            QPStatus.INFEASIBLE,
            (),
            None,
            None,
            None,
            None,
            0,
        )

    objective, _, active, coefficients, multipliers = min(candidates, key=lambda row: (row[0], row[1], row[2]))
    max_violation = max(0.0, float(np.max(rhs - matrix @ coefficients)))
    stationarity = coefficients - matrix[np.asarray(active)].T @ multipliers
    return MinimalNormSolve(
        tuple(float(value) for value in coefficients),
        QPStatus.SOLVED,
        tuple(active),
        max_violation,
        float(np.min(multipliers)),
        float(np.max(np.abs(stationarity), initial=0.0)),
        objective,
        len(candidates),
    )


def encode_coefficient_packet(coefficients: Sequence[float]) -> bytes:
    """Encode one canonical little-endian float64 chart packet with CRC32."""

    values = np.asarray(coefficients)
    if values.ndim != 1 or not 1 <= values.size <= 4 or values.dtype.kind not in "iuf":
        raise RealizedSecantCustodyError("coefficient packet requires one to four real values")
    vector = values.astype("<f8", copy=False)
    if not np.isfinite(vector).all():
        raise RealizedSecantCustodyError("coefficient packet values must be finite")
    body = PACKET_HEADER.pack(PACKET_MAGIC, vector.size) + vector.tobytes(order="C")
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def decode_coefficient_packet(payload: bytes) -> tuple[float, ...]:
    """Strictly decode a canonical chart packet."""

    if not isinstance(payload, bytes) or len(payload) < PACKET_HEADER.size + 8 + 4:
        raise RealizedSecantCustodyError("coefficient packet is truncated")
    magic, count = PACKET_HEADER.unpack(payload[: PACKET_HEADER.size])
    expected = PACKET_HEADER.size + count * 8 + 4
    if magic != PACKET_MAGIC or not 1 <= count <= 4 or len(payload) != expected:
        raise RealizedSecantCustodyError("coefficient packet header/length mismatch")
    body, checksum = payload[:-4], payload[-4:]
    if zlib.crc32(body) & 0xFFFFFFFF != struct.unpack(">I", checksum)[0]:
        raise RealizedSecantCustodyError("coefficient packet checksum mismatch")
    values = np.frombuffer(body[PACKET_HEADER.size :], dtype="<f8")
    if values.size != count or not np.isfinite(values).all():
        raise RealizedSecantCustodyError("coefficient packet contains invalid values")
    if encode_coefficient_packet(values) != payload:
        raise RealizedSecantCustodyError("coefficient packet is not canonical")
    return tuple(float(value) for value in values)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise RealizedSecantCustodyError("custody value is not canonical finite JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def validate_receipt(receipt: Mapping[str, Any], *, expected_pair_count: int) -> str:
    """Validate complete per-pair/per-column custody and return its receipt hash."""

    pairs = _exact_int(expected_pair_count, "expected_pair_count", minimum=1)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise RealizedSecantCustodyError("receipt schema mismatch")
    completed_prefix = _exact_int(receipt.get("completed_prefix"), "completed_prefix", minimum=1)
    if completed_prefix != pairs:
        raise RealizedSecantCustodyError("receipt completed_prefix does not match expected_pair_count")
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        raise RealizedSecantCustodyError("receipt config must be an object")
    relative_residual_tolerance = _finite_scalar(
        config.get("relative_secant_residual_tolerance"),
        "config.relative_secant_residual_tolerance",
    )
    if relative_residual_tolerance < 0.0:
        raise RealizedSecantCustodyError("config relative secant residual tolerance must be nonnegative")
    raw_columns = receipt.get("column_indices")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise RealizedSecantCustodyError("receipt must declare nonempty column_indices")
    columns = [_exact_int(value, "column_index") for value in raw_columns]
    if columns != sorted(columns) or len(columns) != len(set(columns)) or len(columns) > 4:
        raise RealizedSecantCustodyError("receipt columns must be sorted unique rank-at-most-4")
    raw_rows = receipt.get("secant_observations")
    if not isinstance(raw_rows, list):
        raise RealizedSecantCustodyError("receipt secant_observations must be a list")
    if len(raw_rows) != pairs * len(columns):
        raise RealizedSecantCustodyError("receipt lacks exactly one observation per pair per column")
    seen: set[tuple[int, int]] = set()
    observations: list[SecantObservation] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise RealizedSecantCustodyError("receipt observation must be an object")
        row_payload = {key: value for key, value in raw.items() if key != "row_sha256"}
        if raw.get("row_sha256") != canonical_sha256(row_payload):
            raise RealizedSecantCustodyError("secant row hash mismatch")
        row = SecantObservation.from_dict(raw)
        observations.append(row)
        key = (row.pair_index, row.column_index)
        if row.pair_index >= pairs or row.column_index not in columns or key in seen:
            raise RealizedSecantCustodyError("secant pair/column coverage is invalid")
        seen.add(key)
    expected = {(pair, column) for pair in range(pairs) for column in columns}
    if seen != expected:
        raise RealizedSecantCustodyError("secant pair/column coverage is incomplete")
    expected_trust_regions = list(
        build_pair_trust_region_custody(
            observations,
            pair_count=pairs,
            relative_residual_tolerance=relative_residual_tolerance,
        )
    )
    if receipt.get("pair_trust_regions") != expected_trust_regions:
        raise RealizedSecantCustodyError("per-pair trust-region custody mismatch")

    pair_solves = receipt.get("pair_solves")
    if not isinstance(pair_solves, list) or len(pair_solves) != pairs:
        raise RealizedSecantCustodyError("receipt must preserve one explicit solve/refusal per pair")
    solve_indices: list[int] = []
    solve_statuses: list[str] = []
    solve_admitted: list[bool] = []
    for row in pair_solves:
        if not isinstance(row, Mapping):
            raise RealizedSecantCustodyError("pair solve/refusal row must be an object")
        solve_indices.append(_exact_int(row.get("pair_index"), "pair_solve.pair_index"))
        status = row.get("status")
        if not isinstance(status, str) or not status or status not in TERMINAL_PAIR_STATUSES:
            raise RealizedSecantCustodyError("pair solve status is not a recognized nonempty terminal status")
        admitted = row.get("admitted")
        if type(admitted) is not bool:
            raise RealizedSecantCustodyError("pair solve admitted must be an exact bool")
        if admitted != (status == PairSolveStatus.ADMITTED_RECEIVER_CLOSED.value):
            raise RealizedSecantCustodyError("pair solve status/admitted consistency mismatch")
        solve_statuses.append(status)
        solve_admitted.append(admitted)
    if solve_indices != list(range(pairs)):
        raise RealizedSecantCustodyError("pair solve/refusal rows are not contiguous")
    unusable_pairs = {
        _exact_int(row["pair_index"], "pair_trust_region.pair_index")
        for row in expected_trust_regions
        if row["usable"] is False
    }
    for pair_index in unusable_pairs:
        if solve_statuses[pair_index] != PairSolveStatus.TRUST_REGION_REFUSED.value or solve_admitted[pair_index]:
            raise RealizedSecantCustodyError("unusable trust region must be refused and not admitted")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt_hash = canonical_sha256(unsigned)
    if receipt.get("receipt_sha256") != receipt_hash:
        raise RealizedSecantCustodyError("receipt hash mismatch")
    return receipt_hash


__all__ = [
    "RECEIPT_SCHEMA",
    "MinimalNormSolve",
    "PairSolveStatus",
    "QPStatus",
    "RealizedSecantCustodyError",
    "SecantObservation",
    "TrustRegion",
    "WriteSecantObservation",
    "build_pair_trust_region_custody",
    "build_trust_regions",
    "canonical_sha256",
    "decode_coefficient_packet",
    "encode_coefficient_packet",
    "solve_minimal_norm_inequalities",
    "validate_receipt",
]
