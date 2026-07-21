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
BIDIRECTIONAL_RECEIPT_SCHEMA: Final = "bidirectional_amplitude_ladder_receipt.v1"
CHART_BIDIRECTIONAL_RECEIPT_SCHEMA: Final = "chart_bidirectional_amplitude_ladder_receipt.v1"
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


@dataclass(frozen=True)
class BidirectionalWriteObservation:
    """Odd/even decomposition for one declared write at one amplitude rung."""

    ordinal: int
    target_class: int
    current_class: int
    stratum: str
    pre_margin: float
    margin_bucket: str
    positive_predicted_margin_delta: float
    positive_realized_margin_delta: float
    negative_predicted_margin_delta: float
    negative_realized_margin_delta: float
    odd_predicted_margin_delta: float
    odd_realized_margin_delta: float
    even_predicted_margin_delta: float
    even_realized_margin_delta: float
    odd_predicted_secant: float
    odd_realized_secant: float
    even_predicted_secant: float
    even_realized_secant: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordinal", _exact_int(self.ordinal, "ordinal"))
        object.__setattr__(self, "target_class", _exact_int(self.target_class, "target_class"))
        object.__setattr__(self, "current_class", _exact_int(self.current_class, "current_class"))
        for field in ("stratum", "margin_bucket"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise RealizedSecantCustodyError(f"{field} must be nonempty")
        for field in (
            "pre_margin",
            "positive_predicted_margin_delta",
            "positive_realized_margin_delta",
            "negative_predicted_margin_delta",
            "negative_realized_margin_delta",
            "odd_predicted_margin_delta",
            "odd_realized_margin_delta",
            "even_predicted_margin_delta",
            "even_realized_margin_delta",
            "odd_predicted_secant",
            "odd_realized_secant",
            "even_predicted_secant",
            "even_realized_secant",
        ):
            object.__setattr__(self, field, _finite_scalar(getattr(self, field), field))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BidirectionalWriteObservation:
        try:
            return cls(**{field: value[field] for field in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise RealizedSecantCustodyError("malformed bidirectional write row") from exc

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class BidirectionalRungObservation:
    """One paired ``-a,+a`` receiver response for a chart direction."""

    pair_index: int
    direction_index: int
    rung_index: int
    amplitude: float
    positive_source: str
    negative_source: str
    positive: SecantObservation
    negative: SecantObservation
    positive_applied_rgb_delta: tuple[float, ...]
    negative_applied_rgb_delta: tuple[float, ...]
    writes: tuple[BidirectionalWriteObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_index", _exact_int(self.pair_index, "pair_index"))
        object.__setattr__(self, "direction_index", _exact_int(self.direction_index, "direction_index"))
        object.__setattr__(self, "rung_index", _exact_int(self.rung_index, "rung_index"))
        amplitude = _finite_scalar(self.amplitude, "amplitude")
        if amplitude <= 0.0:
            raise RealizedSecantCustodyError("bidirectional amplitude must be positive")
        object.__setattr__(self, "amplitude", amplitude)
        for field in ("positive_source", "negative_source"):
            if getattr(self, field) not in {"MEASURED_G2F", "REUSED_G2E_RUNG0_PRIOR"}:
                raise RealizedSecantCustodyError(f"{field} has unknown custody")
        if not isinstance(self.positive, SecantObservation) or not isinstance(self.negative, SecantObservation):
            raise RealizedSecantCustodyError("bidirectional branches must be typed secant observations")
        for branch, sign in ((self.positive, 1.0), (self.negative, -1.0)):
            if branch.pair_index != self.pair_index or branch.column_index != self.direction_index:
                raise RealizedSecantCustodyError("bidirectional branch identity mismatch")
            if not math.isclose(branch.signed_amplitude, sign * amplitude, rel_tol=0.0, abs_tol=1e-12):
                raise RealizedSecantCustodyError("bidirectional branch amplitude mismatch")
        positive_delta = _finite_vector(self.positive_applied_rgb_delta, "positive_applied_rgb_delta")
        negative_delta = _finite_vector(self.negative_applied_rgb_delta, "negative_applied_rgb_delta")
        expected_size = 3 * len(self.positive.writes)
        if len(self.positive.writes) != len(self.negative.writes) or len(positive_delta) != expected_size:
            raise RealizedSecantCustodyError("bidirectional write/delta geometry mismatch")
        if len(negative_delta) != expected_size:
            raise RealizedSecantCustodyError("bidirectional negative delta geometry mismatch")
        object.__setattr__(self, "positive_applied_rgb_delta", positive_delta)
        object.__setattr__(self, "negative_applied_rgb_delta", negative_delta)
        for delta, branch, label in (
            (positive_delta, self.positive, "positive"),
            (negative_delta, self.negative, "negative"),
        ):
            vector = np.asarray(delta, dtype=np.float64)
            if not math.isclose(float(np.linalg.norm(vector)), branch.applied_rgb_l2, rel_tol=1e-9, abs_tol=1e-12):
                raise RealizedSecantCustodyError(f"{label} applied RGB L2 custody mismatch")
            if not math.isclose(
                float(np.max(np.abs(vector), initial=0.0)),
                branch.applied_rgb_linf,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise RealizedSecantCustodyError(f"{label} applied RGB Linf custody mismatch")
        if not isinstance(self.writes, tuple) or len(self.writes) != len(self.positive.writes):
            raise RealizedSecantCustodyError("bidirectional derived writes are incomplete")
        if any(not isinstance(row, BidirectionalWriteObservation) for row in self.writes):
            raise RealizedSecantCustodyError("bidirectional writes must be typed")
        if [row.ordinal for row in self.writes] != list(range(len(self.writes))):
            raise RealizedSecantCustodyError("bidirectional write ordinals must be contiguous")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BidirectionalRungObservation:
        try:
            positive_raw = value["positive"]
            negative_raw = value["negative"]
            if not isinstance(positive_raw, Mapping) or not isinstance(negative_raw, Mapping):
                raise TypeError
            for raw in (positive_raw, negative_raw):
                payload = {key: item for key, item in raw.items() if key != "row_sha256"}
                if raw.get("row_sha256") != canonical_sha256(payload):
                    raise RealizedSecantCustodyError("nested secant branch hash mismatch")
            row = cls(
                pair_index=value["pair_index"],
                direction_index=value["direction_index"],
                rung_index=value["rung_index"],
                amplitude=value["amplitude"],
                positive_source=value["positive_source"],
                negative_source=value["negative_source"],
                positive=SecantObservation.from_dict(positive_raw),
                negative=SecantObservation.from_dict(negative_raw),
                positive_applied_rgb_delta=tuple(value["positive_applied_rgb_delta"]),
                negative_applied_rgb_delta=tuple(value["negative_applied_rgb_delta"]),
                writes=tuple(BidirectionalWriteObservation.from_dict(item) for item in value["writes"]),
            )
        except (KeyError, TypeError) as exc:
            raise RealizedSecantCustodyError("malformed bidirectional rung row") from exc
        expected = build_bidirectional_rung_observation(
            positive=row.positive,
            negative=row.negative,
            rung_index=row.rung_index,
            strata=tuple(write.stratum for write in row.writes),
            positive_source=row.positive_source,
            negative_source=row.negative_source,
            positive_applied_rgb_delta=row.positive_applied_rgb_delta,
            negative_applied_rgb_delta=row.negative_applied_rgb_delta,
        )
        if row != expected:
            raise RealizedSecantCustodyError("bidirectional derived write custody mismatch")
        return row

    def as_dict(self) -> dict[str, Any]:
        value = {
            "pair_index": self.pair_index,
            "direction_index": self.direction_index,
            "rung_index": self.rung_index,
            "amplitude": self.amplitude,
            "positive_source": self.positive_source,
            "negative_source": self.negative_source,
            "positive": self.positive.as_dict(),
            "negative": self.negative.as_dict(),
            "positive_applied_rgb_delta": list(self.positive_applied_rgb_delta),
            "negative_applied_rgb_delta": list(self.negative_applied_rgb_delta),
            "writes": [row.as_dict() for row in self.writes],
        }
        value["row_sha256"] = canonical_sha256(value)
        return value


@dataclass(frozen=True)
class ChartBranchCustody:
    """Full-plane chart geometry and applied-RGB custody for one signed branch."""

    coefficient_delta: float
    max_centerline_displacement_pixels: float
    coverage_sha256: str
    applied_rgb_delta_sha256: str
    changed_pixel_count: int
    changed_rgb_value_count: int

    def __post_init__(self) -> None:
        for field in ("coefficient_delta", "max_centerline_displacement_pixels"):
            object.__setattr__(self, field, _finite_scalar(getattr(self, field), field))
        if self.coefficient_delta == 0.0 or self.max_centerline_displacement_pixels <= 0.0:
            raise RealizedSecantCustodyError("chart branch displacement must be nonzero and positive")
        for field in ("coverage_sha256", "applied_rgb_delta_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise RealizedSecantCustodyError(f"{field} must be a lowercase SHA-256")
        for field in ("changed_pixel_count", "changed_rgb_value_count"):
            object.__setattr__(self, field, _exact_int(getattr(self, field), field))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChartBranchCustody:
        try:
            return cls(**{field: value[field] for field in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise RealizedSecantCustodyError("malformed chart branch custody") from exc

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class ChartBidirectionalRungObservation:
    """One paired receiver response to a coherent polynomial-chart move."""

    pair_index: int
    direction_index: int
    rung_index: int
    amplitude: float
    amplitude_unit: str
    line_index: int
    coefficient_name: str
    coefficient_index: int
    coefficient_gain_pixels_per_unit: float
    baseline_coverage_sha256: str
    rgb_delta_encoding: str
    positive_chart: ChartBranchCustody
    negative_chart: ChartBranchCustody
    positive: SecantObservation
    negative: SecantObservation
    writes: tuple[BidirectionalWriteObservation, ...]

    def __post_init__(self) -> None:
        for field in ("pair_index", "direction_index", "rung_index", "line_index", "coefficient_index"):
            object.__setattr__(self, field, _exact_int(getattr(self, field), field))
        amplitude = _finite_scalar(self.amplitude, "amplitude")
        gain = _finite_scalar(self.coefficient_gain_pixels_per_unit, "coefficient_gain_pixels_per_unit")
        if amplitude <= 0.0 or gain <= 0.0:
            raise RealizedSecantCustodyError("chart amplitude and coefficient gain must be positive")
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "coefficient_gain_pixels_per_unit", gain)
        if self.amplitude_unit != "native_scorer_centerline_pixels":
            raise RealizedSecantCustodyError("chart amplitude unit is not canonical")
        if self.coefficient_name != "centerline_intercept":
            raise RealizedSecantCustodyError("chart coefficient name is not canonical")
        if self.rgb_delta_encoding != "int16_le_hwc_384x512x3":
            raise RealizedSecantCustodyError("chart RGB delta encoding is not canonical")
        if (
            not isinstance(self.baseline_coverage_sha256, str)
            or len(self.baseline_coverage_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.baseline_coverage_sha256)
        ):
            raise RealizedSecantCustodyError("baseline coverage hash must be a lowercase SHA-256")
        if not isinstance(self.positive_chart, ChartBranchCustody) or not isinstance(
            self.negative_chart, ChartBranchCustody
        ):
            raise RealizedSecantCustodyError("chart branches must carry typed geometry custody")
        expected_delta = amplitude / gain
        if not math.isclose(self.positive_chart.coefficient_delta, expected_delta, rel_tol=1e-12, abs_tol=1e-15):
            raise RealizedSecantCustodyError("positive chart coefficient normalization mismatch")
        if not math.isclose(self.negative_chart.coefficient_delta, -expected_delta, rel_tol=1e-12, abs_tol=1e-15):
            raise RealizedSecantCustodyError("negative chart coefficient normalization mismatch")
        for branch in (self.positive_chart, self.negative_chart):
            if not math.isclose(
                branch.max_centerline_displacement_pixels,
                amplitude,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RealizedSecantCustodyError("chart screen displacement/amplitude mismatch")
        if not isinstance(self.positive, SecantObservation) or not isinstance(self.negative, SecantObservation):
            raise RealizedSecantCustodyError("chart receiver branches must be typed secant observations")
        for branch, sign in ((self.positive, 1.0), (self.negative, -1.0)):
            if branch.pair_index != self.pair_index or branch.column_index != self.direction_index:
                raise RealizedSecantCustodyError("chart branch identity mismatch")
            if not math.isclose(branch.signed_amplitude, sign * amplitude, rel_tol=0.0, abs_tol=1e-12):
                raise RealizedSecantCustodyError("chart branch amplitude mismatch")
        if not isinstance(self.writes, tuple) or len(self.writes) != len(self.positive.writes):
            raise RealizedSecantCustodyError("chart derived writes are incomplete")
        if any(not isinstance(row, BidirectionalWriteObservation) for row in self.writes):
            raise RealizedSecantCustodyError("chart derived writes must be typed")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChartBidirectionalRungObservation:
        try:
            positive_raw = value["positive"]
            negative_raw = value["negative"]
            if not isinstance(positive_raw, Mapping) or not isinstance(negative_raw, Mapping):
                raise TypeError
            for raw in (positive_raw, negative_raw):
                payload = {key: item for key, item in raw.items() if key != "row_sha256"}
                if raw.get("row_sha256") != canonical_sha256(payload):
                    raise RealizedSecantCustodyError("nested chart secant branch hash mismatch")
            row = cls(
                pair_index=value["pair_index"],
                direction_index=value["direction_index"],
                rung_index=value["rung_index"],
                amplitude=value["amplitude"],
                amplitude_unit=value["amplitude_unit"],
                line_index=value["line_index"],
                coefficient_name=value["coefficient_name"],
                coefficient_index=value["coefficient_index"],
                coefficient_gain_pixels_per_unit=value["coefficient_gain_pixels_per_unit"],
                baseline_coverage_sha256=value["baseline_coverage_sha256"],
                rgb_delta_encoding=value["rgb_delta_encoding"],
                positive_chart=ChartBranchCustody.from_dict(value["positive_chart"]),
                negative_chart=ChartBranchCustody.from_dict(value["negative_chart"]),
                positive=SecantObservation.from_dict(positive_raw),
                negative=SecantObservation.from_dict(negative_raw),
                writes=tuple(BidirectionalWriteObservation.from_dict(item) for item in value["writes"]),
            )
        except (KeyError, TypeError) as exc:
            raise RealizedSecantCustodyError("malformed chart bidirectional rung row") from exc
        expected_writes = _build_bidirectional_writes(
            positive=row.positive,
            negative=row.negative,
            strata=tuple(write.stratum for write in row.writes),
        )
        if row.writes != expected_writes:
            raise RealizedSecantCustodyError("chart bidirectional derived write custody mismatch")
        return row

    def as_dict(self) -> dict[str, Any]:
        value = {
            "pair_index": self.pair_index,
            "direction_index": self.direction_index,
            "rung_index": self.rung_index,
            "amplitude": self.amplitude,
            "amplitude_unit": self.amplitude_unit,
            "line_index": self.line_index,
            "coefficient_name": self.coefficient_name,
            "coefficient_index": self.coefficient_index,
            "coefficient_gain_pixels_per_unit": self.coefficient_gain_pixels_per_unit,
            "baseline_coverage_sha256": self.baseline_coverage_sha256,
            "rgb_delta_encoding": self.rgb_delta_encoding,
            "positive_chart": self.positive_chart.as_dict(),
            "negative_chart": self.negative_chart.as_dict(),
            "positive": self.positive.as_dict(),
            "negative": self.negative.as_dict(),
            "writes": [row.as_dict() for row in self.writes],
        }
        value["row_sha256"] = canonical_sha256(value)
        return value


def _build_bidirectional_writes(
    *,
    positive: SecantObservation,
    negative: SecantObservation,
    strata: Sequence[str],
) -> tuple[BidirectionalWriteObservation, ...]:
    """Rederive odd/even write rows shared by local-pixel and chart ladders."""

    if positive.signed_amplitude <= 0.0 or negative.signed_amplitude >= 0.0:
        raise RealizedSecantCustodyError("paired branches must be ordered positive then negative")
    amplitude = positive.signed_amplitude
    if not math.isclose(-negative.signed_amplitude, amplitude, rel_tol=0.0, abs_tol=1e-12):
        raise RealizedSecantCustodyError("paired branches must use equal absolute amplitude")
    if len(strata) != len(positive.writes) or len(negative.writes) != len(positive.writes):
        raise RealizedSecantCustodyError("paired branch write/stratum coverage mismatch")
    writes: list[BidirectionalWriteObservation] = []
    for positive_write, negative_write, stratum in zip(positive.writes, negative.writes, strata, strict=True):
        identities = (
            positive_write.ordinal,
            positive_write.target_class,
            positive_write.current_class,
            positive_write.pre_margin,
            positive_write.margin_bucket,
        )
        negative_identities = (
            negative_write.ordinal,
            negative_write.target_class,
            negative_write.current_class,
            negative_write.pre_margin,
            negative_write.margin_bucket,
        )
        if identities != negative_identities:
            raise RealizedSecantCustodyError("paired branch declared-write identities differ")
        odd_predicted = 0.5 * (positive_write.predicted_margin_delta - negative_write.predicted_margin_delta)
        odd_realized = 0.5 * (positive_write.realized_margin_delta - negative_write.realized_margin_delta)
        even_predicted = 0.5 * (positive_write.predicted_margin_delta + negative_write.predicted_margin_delta)
        even_realized = 0.5 * (positive_write.realized_margin_delta + negative_write.realized_margin_delta)
        writes.append(
            BidirectionalWriteObservation(
                ordinal=positive_write.ordinal,
                target_class=positive_write.target_class,
                current_class=positive_write.current_class,
                stratum=stratum,
                pre_margin=positive_write.pre_margin,
                margin_bucket=positive_write.margin_bucket,
                positive_predicted_margin_delta=positive_write.predicted_margin_delta,
                positive_realized_margin_delta=positive_write.realized_margin_delta,
                negative_predicted_margin_delta=negative_write.predicted_margin_delta,
                negative_realized_margin_delta=negative_write.realized_margin_delta,
                odd_predicted_margin_delta=odd_predicted,
                odd_realized_margin_delta=odd_realized,
                even_predicted_margin_delta=even_predicted,
                even_realized_margin_delta=even_realized,
                odd_predicted_secant=odd_predicted / amplitude,
                odd_realized_secant=odd_realized / amplitude,
                even_predicted_secant=even_predicted / amplitude,
                even_realized_secant=even_realized / amplitude,
            )
        )
    return tuple(writes)


def build_bidirectional_rung_observation(
    *,
    positive: SecantObservation,
    negative: SecantObservation,
    rung_index: int,
    strata: Sequence[str],
    positive_source: str,
    negative_source: str,
    positive_applied_rgb_delta: Sequence[float],
    negative_applied_rgb_delta: Sequence[float],
) -> BidirectionalRungObservation:
    """Construct and rederive one paired rung without trusting serialized summaries."""

    amplitude = positive.signed_amplitude
    writes = _build_bidirectional_writes(positive=positive, negative=negative, strata=strata)
    return BidirectionalRungObservation(
        pair_index=positive.pair_index,
        direction_index=positive.column_index,
        rung_index=rung_index,
        amplitude=amplitude,
        positive_source=positive_source,
        negative_source=negative_source,
        positive=positive,
        negative=negative,
        positive_applied_rgb_delta=tuple(positive_applied_rgb_delta),
        negative_applied_rgb_delta=tuple(negative_applied_rgb_delta),
        writes=writes,
    )


def build_chart_bidirectional_rung_observation(
    *,
    positive: SecantObservation,
    negative: SecantObservation,
    rung_index: int,
    strata: Sequence[str],
    line_index: int,
    coefficient_index: int,
    coefficient_gain_pixels_per_unit: float,
    baseline_coverage_sha256: str,
    positive_chart: ChartBranchCustody,
    negative_chart: ChartBranchCustody,
) -> ChartBidirectionalRungObservation:
    """Construct a coherent chart rung while rederiving all odd/even write rows."""

    writes = _build_bidirectional_writes(positive=positive, negative=negative, strata=strata)
    return ChartBidirectionalRungObservation(
        pair_index=positive.pair_index,
        direction_index=positive.column_index,
        rung_index=rung_index,
        amplitude=positive.signed_amplitude,
        amplitude_unit="native_scorer_centerline_pixels",
        line_index=line_index,
        coefficient_name="centerline_intercept",
        coefficient_index=coefficient_index,
        coefficient_gain_pixels_per_unit=coefficient_gain_pixels_per_unit,
        baseline_coverage_sha256=baseline_coverage_sha256,
        rgb_delta_encoding="int16_le_hwc_384x512x3",
        positive_chart=positive_chart,
        negative_chart=negative_chart,
        positive=positive,
        negative=negative,
        writes=writes,
    )


def build_bidirectional_trust_region_custody(
    observations: Sequence[BidirectionalRungObservation | ChartBidirectionalRungObservation],
    *,
    relative_residual_tolerance: float,
    response_epsilon: float = 1e-12,
) -> tuple[dict[str, Any], ...]:
    """Build pair/direction/rung/class/bucket trust rows from central secants."""

    tolerance = _finite_scalar(relative_residual_tolerance, "relative_residual_tolerance")
    epsilon = _finite_scalar(response_epsilon, "response_epsilon")
    if tolerance < 0.0 or epsilon <= 0.0:
        raise RealizedSecantCustodyError("bidirectional trust tolerances are invalid")
    if not observations or any(
        not isinstance(row, (BidirectionalRungObservation, ChartBidirectionalRungObservation)) for row in observations
    ):
        raise RealizedSecantCustodyError("bidirectional trust requires typed observations")
    result: list[dict[str, Any]] = []
    for observation in sorted(
        observations,
        key=lambda row: (row.pair_index, row.direction_index, row.rung_index),
    ):
        grouped: dict[tuple[int, str], list[BidirectionalWriteObservation]] = defaultdict(list)
        for write in observation.writes:
            grouped[(write.target_class, write.margin_bucket)].append(write)
        for (target_class, bucket), writes in sorted(grouped.items()):
            residuals: list[float] = []
            even_ratios: list[float] = []
            reasons: set[str] = set()
            sign_consistent_count = 0
            for write in writes:
                denominator = max(
                    abs(write.odd_predicted_secant),
                    abs(write.odd_realized_secant),
                    epsilon,
                )
                residual = abs(write.odd_realized_secant - write.odd_predicted_secant) / denominator
                residuals.append(residual)
                even_ratios.append(abs(write.even_realized_secant) / max(abs(write.odd_realized_secant), epsilon))
                branch_signs = tuple(
                    abs(predicted) > epsilon and abs(realized) > epsilon and predicted * realized > 0.0
                    for predicted, realized in (
                        (
                            write.positive_predicted_margin_delta,
                            write.positive_realized_margin_delta,
                        ),
                        (
                            write.negative_predicted_margin_delta,
                            write.negative_realized_margin_delta,
                        ),
                        (write.odd_predicted_secant, write.odd_realized_secant),
                    )
                )
                if all(branch_signs):
                    sign_consistent_count += 1
                else:
                    reasons.add("BIDIRECTIONAL_SIGN_OR_ZERO")
                if residual > tolerance:
                    reasons.add("RELATIVE_SECANT_RESIDUAL")
            if observation.positive.applied_rgb_linf <= 0.0 or observation.negative.applied_rgb_linf <= 0.0:
                reasons.add("ZERO_APPLIED_RGB_BRANCH")
            row: dict[str, Any] = {
                "pair_index": observation.pair_index,
                "direction_index": observation.direction_index,
                "rung_index": observation.rung_index,
                "amplitude": observation.amplitude,
                "target_class": target_class,
                "margin_bucket": bucket,
                "observation_count": len(writes),
                "sign_consistent_count": sign_consistent_count,
                "max_relative_residual": max(residuals),
                "max_even_to_odd_ratio": max(even_ratios),
                "min_abs_odd_realized_secant": min(abs(write.odd_realized_secant) for write in writes),
                "positive_uint8_saturation_count": observation.positive.uint8_saturation_count,
                "negative_uint8_saturation_count": observation.negative.uint8_saturation_count,
                "saturation_associated": bool(
                    observation.positive.uint8_saturation_count or observation.negative.uint8_saturation_count
                ),
                "usable": not reasons,
                "refusal_reasons": sorted(reasons),
            }
            row["row_sha256"] = canonical_sha256(row)
            result.append(row)
    return tuple(result)


def select_best_bidirectional_rungs(
    trust_regions: Sequence[Mapping[str, Any]],
    *,
    effective_direction_count_by_pair: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    """Select the lowest-residual usable rung for every effective direction."""

    if not effective_direction_count_by_pair:
        raise RealizedSecantCustodyError("effective direction counts must be nonempty")
    counts = [_exact_int(value, "effective_direction_count", minimum=1) for value in effective_direction_count_by_pair]
    if any(value > 4 for value in counts):
        raise RealizedSecantCustodyError("effective direction count exceeds rank four")
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trust_regions:
        pair = _exact_int(row.get("pair_index"), "trust.pair_index")
        direction = _exact_int(row.get("direction_index"), "trust.direction_index")
        rung = _exact_int(row.get("rung_index"), "trust.rung_index")
        if pair >= len(counts) or direction >= counts[pair]:
            raise RealizedSecantCustodyError("trust region references a non-effective direction")
        grouped[(pair, direction, rung)].append(row)
    selections: list[dict[str, Any]] = []
    for pair, count in enumerate(counts):
        for direction in range(count):
            candidates: list[tuple[float, float, float, int, list[Mapping[str, Any]]]] = []
            refusal_reasons: set[str] = set()
            for (candidate_pair, candidate_direction, rung), rows in sorted(grouped.items()):
                if (candidate_pair, candidate_direction) != (pair, direction):
                    continue
                if all(row.get("usable") is True for row in rows):
                    candidates.append(
                        (
                            max(float(row["max_relative_residual"]) for row in rows),
                            max(float(row["max_even_to_odd_ratio"]) for row in rows),
                            float(rows[0]["amplitude"]),
                            rung,
                            rows,
                        )
                    )
                else:
                    refusal_reasons.update(reason for row in rows for reason in row.get("refusal_reasons", ()))
            selected = min(candidates, key=lambda item: item[:4]) if candidates else None
            row = {
                "pair_index": pair,
                "direction_index": direction,
                "selected": selected is not None,
                "selected_rung_index": selected[3] if selected is not None else None,
                "selected_amplitude": selected[2] if selected is not None else None,
                "max_relative_residual": selected[0] if selected is not None else None,
                "max_even_to_odd_ratio": selected[1] if selected is not None else None,
                "refusal_reasons": [] if selected is not None else sorted(refusal_reasons),
            }
            row["row_sha256"] = canonical_sha256(row)
            selections.append(row)
    return tuple(selections)


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


def validate_bidirectional_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_pair_count: int,
) -> str:
    """Rebuild the G2f paired-rung trust and selection tables from raw rows."""

    pairs = _exact_int(expected_pair_count, "expected_pair_count", minimum=1)
    if receipt.get("schema") != BIDIRECTIONAL_RECEIPT_SCHEMA:
        raise RealizedSecantCustodyError("bidirectional receipt schema mismatch")
    if _exact_int(receipt.get("completed_prefix"), "completed_prefix", minimum=1) != pairs:
        raise RealizedSecantCustodyError("bidirectional receipt completed_prefix mismatch")
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        raise RealizedSecantCustodyError("bidirectional receipt config must be an object")
    tolerance = _finite_scalar(
        config.get("relative_secant_residual_tolerance"),
        "config.relative_secant_residual_tolerance",
    )
    if tolerance < 0.0:
        raise RealizedSecantCustodyError("bidirectional residual tolerance must be nonnegative")
    prior = config.get("g2e_rung0_prior")
    prior_hashes = receipt.get("g2e_rung0_prior_observation_hashes")
    if (
        not isinstance(prior, Mapping)
        or prior.get("secant_observation_count") != 64
        or prior.get("remeasured") is not False
        or not isinstance(prior_hashes, list)
        or len(prior_hashes) != 64
        or any(not isinstance(value, str) or len(value) != 64 for value in prior_hashes)
    ):
        raise RealizedSecantCustodyError("bidirectional G2e rung-0 prior custody mismatch")
    amplitude_ladder = config.get("amplitude_ladder")
    if not isinstance(amplitude_ladder, list) or not amplitude_ladder:
        raise RealizedSecantCustodyError("bidirectional amplitude ladder must be nonempty")
    amplitudes = [_finite_scalar(value, "amplitude_ladder") for value in amplitude_ladder]
    if any(value <= 0.0 for value in amplitudes) or amplitudes != sorted(set(amplitudes)):
        raise RealizedSecantCustodyError("bidirectional amplitudes must be positive sorted unique")
    counts_raw = receipt.get("effective_direction_count_by_pair")
    if not isinstance(counts_raw, list) or len(counts_raw) != pairs:
        raise RealizedSecantCustodyError("bidirectional effective-direction coverage mismatch")
    counts = [_exact_int(value, "effective_direction_count", minimum=1) for value in counts_raw]
    if any(value > 4 for value in counts):
        raise RealizedSecantCustodyError("bidirectional effective direction exceeds rank four")

    raw_rows = receipt.get("bidirectional_observations")
    if not isinstance(raw_rows, list):
        raise RealizedSecantCustodyError("bidirectional observations must be a list")
    expected_count = sum(counts) * len(amplitudes)
    if len(raw_rows) != expected_count:
        raise RealizedSecantCustodyError("bidirectional observation coverage count mismatch")
    observations: list[BidirectionalRungObservation] = []
    seen: set[tuple[int, int, int]] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise RealizedSecantCustodyError("bidirectional observation must be an object")
        payload = {key: value for key, value in raw.items() if key != "row_sha256"}
        if raw.get("row_sha256") != canonical_sha256(payload):
            raise RealizedSecantCustodyError("bidirectional observation row hash mismatch")
        observation = BidirectionalRungObservation.from_dict(raw)
        key = (observation.pair_index, observation.direction_index, observation.rung_index)
        if (
            observation.pair_index >= pairs
            or observation.direction_index >= counts[observation.pair_index]
            or observation.rung_index >= len(amplitudes)
            or key in seen
        ):
            raise RealizedSecantCustodyError("bidirectional observation identity is invalid")
        if not math.isclose(
            observation.amplitude,
            amplitudes[observation.rung_index],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RealizedSecantCustodyError("bidirectional rung amplitude/config mismatch")
        seen.add(key)
        observations.append(observation)
    expected = {
        (pair, direction, rung)
        for pair, count in enumerate(counts)
        for direction in range(count)
        for rung in range(len(amplitudes))
    }
    if seen != expected:
        raise RealizedSecantCustodyError("bidirectional observation coverage is incomplete")

    trust_regions = list(
        build_bidirectional_trust_region_custody(
            observations,
            relative_residual_tolerance=tolerance,
        )
    )
    if receipt.get("pair_direction_rung_trust_regions") != trust_regions:
        raise RealizedSecantCustodyError("bidirectional trust-region custody mismatch")
    selections = list(
        select_best_bidirectional_rungs(
            trust_regions,
            effective_direction_count_by_pair=counts,
        )
    )
    if receipt.get("selected_rungs") != selections:
        raise RealizedSecantCustodyError("bidirectional selected-rung custody mismatch")

    pair_solves = receipt.get("pair_solves")
    if not isinstance(pair_solves, list) or len(pair_solves) != pairs:
        raise RealizedSecantCustodyError("bidirectional receipt must preserve one pair solve")
    selected_by_pair = defaultdict(list)
    for row in selections:
        selected_by_pair[int(row["pair_index"])].append(row["selected"] is True)
    for pair, solve in enumerate(pair_solves):
        if not isinstance(solve, Mapping) or solve.get("pair_index") != pair:
            raise RealizedSecantCustodyError("bidirectional pair solves are not contiguous")
        status = solve.get("status")
        admitted = solve.get("admitted")
        if status not in TERMINAL_PAIR_STATUSES or type(admitted) is not bool:
            raise RealizedSecantCustodyError("bidirectional pair solve terminal custody is invalid")
        if admitted != (status == PairSolveStatus.ADMITTED_RECEIVER_CLOSED.value):
            raise RealizedSecantCustodyError("bidirectional pair solve status/admitted mismatch")
        if not all(selected_by_pair[pair]) and (status != PairSolveStatus.TRUST_REGION_REFUSED.value or admitted):
            raise RealizedSecantCustodyError("missing selected direction must fail closed before QP")

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt_hash = canonical_sha256(unsigned)
    if receipt.get("receipt_sha256") != receipt_hash:
        raise RealizedSecantCustodyError("bidirectional receipt hash mismatch")
    return receipt_hash


def validate_chart_bidirectional_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_pair_count: int,
) -> str:
    """Rebuild coherent chart-rung trust, selections, and pixel-level comparison."""

    pairs = _exact_int(expected_pair_count, "expected_pair_count", minimum=1)
    if receipt.get("schema") != CHART_BIDIRECTIONAL_RECEIPT_SCHEMA:
        raise RealizedSecantCustodyError("chart bidirectional receipt schema mismatch")
    if _exact_int(receipt.get("completed_prefix"), "completed_prefix", minimum=1) != pairs:
        raise RealizedSecantCustodyError("chart bidirectional receipt completed_prefix mismatch")
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        raise RealizedSecantCustodyError("chart bidirectional receipt config must be an object")
    tolerance = _finite_scalar(
        config.get("relative_secant_residual_tolerance"),
        "config.relative_secant_residual_tolerance",
    )
    amplitude_ladder = config.get("amplitude_ladder")
    comparator = config.get("pixel_level_comparator")
    if tolerance < 0.0 or not isinstance(amplitude_ladder, list) or not amplitude_ladder:
        raise RealizedSecantCustodyError("chart trust tolerance/amplitude ladder is invalid")
    amplitudes = [_finite_scalar(value, "amplitude_ladder") for value in amplitude_ladder]
    if any(value <= 0.0 for value in amplitudes) or amplitudes != sorted(set(amplitudes)):
        raise RealizedSecantCustodyError("chart amplitudes must be positive sorted unique")
    if not isinstance(comparator, Mapping):
        raise RealizedSecantCustodyError("chart receipt lacks pixel-level comparator custody")
    for field in ("receipt_file_sha256", "canonical_receipt_sha256"):
        value = comparator.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise RealizedSecantCustodyError("chart pixel comparator hash custody is invalid")
    if _exact_int(comparator.get("completed_prefix"), "pixel_level_comparator.completed_prefix", minimum=1) < pairs:
        raise RealizedSecantCustodyError("chart pixel comparator does not cover the measured prefix")

    raw_rows = receipt.get("chart_bidirectional_observations")
    if not isinstance(raw_rows, list) or len(raw_rows) != pairs * len(amplitudes):
        raise RealizedSecantCustodyError("chart bidirectional observation coverage count mismatch")
    observations: list[ChartBidirectionalRungObservation] = []
    seen: set[tuple[int, int]] = set()
    pair_geometry: dict[int, tuple[Any, ...]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise RealizedSecantCustodyError("chart bidirectional observation must be an object")
        payload = {key: value for key, value in raw.items() if key != "row_sha256"}
        if raw.get("row_sha256") != canonical_sha256(payload):
            raise RealizedSecantCustodyError("chart bidirectional observation row hash mismatch")
        observation = ChartBidirectionalRungObservation.from_dict(raw)
        key = (observation.pair_index, observation.rung_index)
        if (
            observation.pair_index >= pairs
            or observation.direction_index != 0
            or observation.rung_index >= len(amplitudes)
            or key in seen
        ):
            raise RealizedSecantCustodyError("chart bidirectional observation identity is invalid")
        if not math.isclose(
            observation.amplitude,
            amplitudes[observation.rung_index],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RealizedSecantCustodyError("chart rung amplitude/config mismatch")
        geometry = (
            observation.line_index,
            observation.coefficient_index,
            observation.coefficient_gain_pixels_per_unit,
            observation.baseline_coverage_sha256,
        )
        if observation.pair_index in pair_geometry and pair_geometry[observation.pair_index] != geometry:
            raise RealizedSecantCustodyError("chart pair geometry drifted across amplitude rungs")
        pair_geometry[observation.pair_index] = geometry
        seen.add(key)
        observations.append(observation)
    if seen != {(pair, rung) for pair in range(pairs) for rung in range(len(amplitudes))}:
        raise RealizedSecantCustodyError("chart bidirectional observation coverage is incomplete")

    trust_regions = list(
        build_bidirectional_trust_region_custody(
            observations,
            relative_residual_tolerance=tolerance,
        )
    )
    if receipt.get("pair_direction_rung_trust_regions") != trust_regions:
        raise RealizedSecantCustodyError("chart bidirectional trust-region custody mismatch")
    selections = list(
        select_best_bidirectional_rungs(
            trust_regions,
            effective_direction_count_by_pair=[1] * pairs,
        )
    )
    if receipt.get("selected_rungs") != selections:
        raise RealizedSecantCustodyError("chart bidirectional selected-rung custody mismatch")

    comparison = receipt.get("level_comparison")
    if not isinstance(comparison, Mapping):
        raise RealizedSecantCustodyError("chart receipt lacks level comparison")
    pixel_selected = comparison.get("pixel_pair_selected")
    if (
        not isinstance(pixel_selected, list)
        or len(pixel_selected) != pairs
        or any(type(value) is not bool for value in pixel_selected)
    ):
        raise RealizedSecantCustodyError("chart pixel selected vector is invalid")
    chart_selected = [bool(row["selected"]) for row in selections]
    if comparison.get("chart_pair_selected") != chart_selected:
        raise RealizedSecantCustodyError("chart selected vector does not rederive")
    chart_only = [
        index
        for index, (chart, pixel) in enumerate(zip(chart_selected, pixel_selected, strict=True))
        if chart and not pixel
    ]
    pixel_only = [
        index
        for index, (chart, pixel) in enumerate(zip(chart_selected, pixel_selected, strict=True))
        if pixel and not chart
    ]
    both = [
        index
        for index, (chart, pixel) in enumerate(zip(chart_selected, pixel_selected, strict=True))
        if chart and pixel
    ]
    neither = [
        index
        for index, (chart, pixel) in enumerate(zip(chart_selected, pixel_selected, strict=True))
        if not chart and not pixel
    ]
    expected_comparison = {
        "pixel_pair_selected": pixel_selected,
        "chart_pair_selected": chart_selected,
        "pixel_selected_pair_count": sum(pixel_selected),
        "chart_selected_pair_count": sum(chart_selected),
        "chart_only_rescued_pair_indices": chart_only,
        "chart_only_rescued_pair_count": len(chart_only),
        "pixel_only_pair_indices": pixel_only,
        "pixel_only_pair_count": len(pixel_only),
        "both_selected_pair_indices": both,
        "both_selected_pair_count": len(both),
        "neither_selected_pair_indices": neither,
        "neither_selected_pair_count": len(neither),
    }
    if comparison != expected_comparison:
        raise RealizedSecantCustodyError("chart level comparison does not rederive")

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt_hash = canonical_sha256(unsigned)
    if receipt.get("receipt_sha256") != receipt_hash:
        raise RealizedSecantCustodyError("chart bidirectional receipt hash mismatch")
    return receipt_hash


__all__ = [
    "BIDIRECTIONAL_RECEIPT_SCHEMA",
    "CHART_BIDIRECTIONAL_RECEIPT_SCHEMA",
    "RECEIPT_SCHEMA",
    "BidirectionalRungObservation",
    "BidirectionalWriteObservation",
    "ChartBidirectionalRungObservation",
    "ChartBranchCustody",
    "MinimalNormSolve",
    "PairSolveStatus",
    "QPStatus",
    "RealizedSecantCustodyError",
    "SecantObservation",
    "TrustRegion",
    "WriteSecantObservation",
    "build_bidirectional_rung_observation",
    "build_bidirectional_trust_region_custody",
    "build_chart_bidirectional_rung_observation",
    "build_pair_trust_region_custody",
    "build_trust_regions",
    "canonical_sha256",
    "decode_coefficient_packet",
    "encode_coefficient_packet",
    "select_best_bidirectional_rungs",
    "solve_minimal_norm_inequalities",
    "validate_bidirectional_receipt",
    "validate_chart_bidirectional_receipt",
    "validate_receipt",
]
