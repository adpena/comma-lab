# SPDX-License-Identifier: MIT
"""Typed sided-tolerance addendum for SDWL1 and asymmetric DDM pricing.

The base SDWL1 packet remains byte-identical.  This additive document gives
e1 a pair-normal renderer bound and e2 a direction-specific reduced cost.
Every off-diagonal class orientation is a distinct row; reverse directions
are cross-linked but never averaged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

from tac.optimization.ddm_dv2_sdwl1 import CLASS_NAMES, canonical_json_bytes

HEADER_SCHEMA: Final = "sdwl1.sided_tolerance.header.v1"
ROW_SCHEMA: Final = "sdwl1.sided_tolerance.row.v1"
E1_SCHEMA: Final = "ddm.e1.sided_normal_bounds.v1"
E2_SCHEMA: Final = "ddm.e2.sided_reduced_cost.v1"
CLASS_COUNT: Final = len(CLASS_NAMES)
ORDERED_PAIR_COUNT: Final = CLASS_COUNT * (CLASS_COUNT - 1)
TEMPORAL_STRATA: Final = ("n600_full", "n600_first64_tail", "n600_last64_tail")
_ROW_KEYS: Final = {
    "schema",
    "winner_id",
    "winner",
    "rival_id",
    "rival",
    "orientation",
    "reverse_orientation",
    "temporal_stratum",
    "boundary_pixel_count",
    "reverse_boundary_pixel_count",
    "head_normal_l2",
    "margin_quantiles",
    "d2_quantiles",
    "inner_tolerance_d2",
    "outer_tolerance_d2",
    "inner_to_outer_ratio",
    "asymmetry_delta_d2",
    "first_rung",
    "verdict_scope",
}
_HEADER_KEYS: Final = {
    "schema",
    "axis",
    "score_claim",
    "promotion_eligible",
    "class_names",
    "ordered_pair_count",
    "temporal_strata",
    "source_video_sha256",
    "segnet_weights_sha256",
    "upstream_modules_sha256",
    "telemetry_sha256",
}
_QUANTILE_KEYS: Final = ("min", "q01", "q10", "median", "q90", "q99", "max")


class SidedToleranceError(ValueError):
    """Raised on malformed, symmetric-by-accident, or stale sided custody."""


def orientation(winner: int, rival: int) -> str:
    if winner == rival or not (0 <= winner < CLASS_COUNT and 0 <= rival < CLASS_COUNT):
        raise SidedToleranceError("ordered-pair ids must be distinct valid classes")
    return f"{CLASS_NAMES[winner]}->{CLASS_NAMES[rival]}"


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise SidedToleranceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SidedToleranceError(f"{name} must be finite and nonnegative")
    return result


def _quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return dict.fromkeys(_QUANTILE_KEYS)
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise SidedToleranceError("margin/D2 samples must be finite and nonnegative")
    measured = np.quantile(array, [0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0])
    return {
        name: float(value)
        for name, value in zip(_QUANTILE_KEYS, measured, strict=True)
    }


def _validate_quantiles(value: Any, *, name: str) -> dict[str, float | None]:
    if not isinstance(value, dict) or set(value) != set(_QUANTILE_KEYS):
        raise SidedToleranceError(f"{name} has noncanonical quantile keys")
    previous: float | None = None
    validated: dict[str, float | None] = {}
    for key in _QUANTILE_KEYS:
        raw = value[key]
        if raw is None:
            if any(item is not None for item in value.values()):
                raise SidedToleranceError(f"{name} mixes null and measured quantiles")
            return dict.fromkeys(_QUANTILE_KEYS)
        current = _finite_nonnegative(raw, name=f"{name}.{key}")
        if previous is not None and current < previous:
            raise SidedToleranceError(f"{name} quantiles are not monotone")
        validated[key] = current
        previous = current
    return validated


@dataclass(frozen=True, slots=True)
class SidedToleranceHeader:
    schema: str
    axis: str
    score_claim: bool
    promotion_eligible: bool
    class_names: list[str]
    ordered_pair_count: int
    temporal_strata: list[str]
    source_video_sha256: str
    segnet_weights_sha256: str
    upstream_modules_sha256: str
    telemetry_sha256: str


@dataclass(frozen=True, slots=True)
class SidedToleranceRow:
    schema: str
    winner_id: int
    winner: str
    rival_id: int
    rival: str
    orientation: str
    reverse_orientation: str
    temporal_stratum: str
    boundary_pixel_count: int
    reverse_boundary_pixel_count: int
    head_normal_l2: float
    margin_quantiles: dict[str, float | None]
    d2_quantiles: dict[str, float | None]
    inner_tolerance_d2: float | None
    outer_tolerance_d2: float | None
    inner_to_outer_ratio: float | None
    asymmetry_delta_d2: float | None
    first_rung: str
    verdict_scope: str


def build_header(
    *,
    source_video_sha256: str,
    segnet_weights_sha256: str,
    upstream_modules_sha256: str,
    telemetry_sha256: str,
) -> SidedToleranceHeader:
    digests = {
        "source_video_sha256": source_video_sha256,
        "segnet_weights_sha256": segnet_weights_sha256,
        "upstream_modules_sha256": upstream_modules_sha256,
        "telemetry_sha256": telemetry_sha256,
    }
    for name, value in digests.items():
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise SidedToleranceError(f"{name} must be a lowercase SHA-256")
    return SidedToleranceHeader(
        schema=HEADER_SCHEMA,
        axis="[macOS-CPU frozen-SegNet advisory]",
        score_claim=False,
        promotion_eligible=False,
        class_names=list(CLASS_NAMES),
        ordered_pair_count=ORDERED_PAIR_COUNT,
        temporal_strata=list(TEMPORAL_STRATA),
        **digests,
    )


def build_sided_rows(
    *,
    temporal_stratum: str,
    margins_by_orientation: Mapping[str, Sequence[float] | np.ndarray],
    pair_norms_by_orientation: Mapping[str, float],
) -> list[SidedToleranceRow]:
    """Build all 20 directional rows from measured boundary-margin samples."""

    if temporal_stratum not in TEMPORAL_STRATA:
        raise SidedToleranceError(f"unknown temporal stratum: {temporal_stratum}")
    expected = {
        orientation(winner, rival)
        for winner in range(CLASS_COUNT)
        for rival in range(CLASS_COUNT)
        if winner != rival
    }
    if set(margins_by_orientation) != expected or set(pair_norms_by_orientation) != expected:
        raise SidedToleranceError("sided samples and head norms must cover exactly all 20 orientations")
    rows: list[SidedToleranceRow] = []
    for winner in range(CLASS_COUNT):
        for rival in range(CLASS_COUNT):
            if winner == rival:
                continue
            key = orientation(winner, rival)
            reverse = orientation(rival, winner)
            norm = _finite_nonnegative(pair_norms_by_orientation[key], name=f"{key}.head_normal_l2")
            reverse_norm = _finite_nonnegative(
                pair_norms_by_orientation[reverse],
                name=f"{reverse}.head_normal_l2",
            )
            if norm <= 0.0 or not math.isclose(norm, reverse_norm, rel_tol=1e-9, abs_tol=1e-12):
                raise SidedToleranceError("reverse orientations must share one positive head-normal norm")
            margins = np.asarray(margins_by_orientation[key], dtype=np.float64).reshape(-1)
            reverse_margins = np.asarray(
                margins_by_orientation[reverse],
                dtype=np.float64,
            ).reshape(-1)
            if np.any(margins < 0.0) or np.any(reverse_margins < 0.0):
                raise SidedToleranceError("winner-side margins must be nonnegative")
            distances = margins / norm
            reverse_distances = reverse_margins / norm
            own_quantiles = _quantiles(distances)
            margin_quantiles = _quantiles(margins)
            inner = own_quantiles["q10"]
            reverse_quantiles = _quantiles(reverse_distances)
            outer = reverse_quantiles["q10"]
            ratio = (
                float(inner / outer)
                if inner is not None and outer is not None and outer > 0.0
                else None
            )
            delta = (
                float(inner - outer)
                if inner is not None and outer is not None
                else None
            )
            rows.append(
                SidedToleranceRow(
                    schema=ROW_SCHEMA,
                    winner_id=winner,
                    winner=CLASS_NAMES[winner],
                    rival_id=rival,
                    rival=CLASS_NAMES[rival],
                    orientation=key,
                    reverse_orientation=reverse,
                    temporal_stratum=temporal_stratum,
                    boundary_pixel_count=int(margins.size),
                    reverse_boundary_pixel_count=int(reverse_margins.size),
                    head_normal_l2=norm,
                    margin_quantiles=margin_quantiles,
                    d2_quantiles=own_quantiles,
                    inner_tolerance_d2=inner,
                    outer_tolerance_d2=outer,
                    inner_to_outer_ratio=ratio,
                    asymmetry_delta_d2=delta,
                    first_rung=(
                        "inverse-solve one receiver-realizable segment on each side through "
                        "the exact R path and frozen SegNet before e2 spends bytes"
                    ),
                    verdict_scope=(
                        "frozen SegNet head-space boundary samples on the declared temporal "
                        "stratum; pixel-space realization and Pose collateral remain unproven"
                    ),
                )
            )
    return rows


def validate_row(value: Mapping[str, Any]) -> SidedToleranceRow:
    if set(value) != _ROW_KEYS or value.get("schema") != ROW_SCHEMA:
        raise SidedToleranceError("sided-tolerance row schema/keys are malformed")
    winner = int(value["winner_id"])
    rival = int(value["rival_id"])
    key = orientation(winner, rival)
    reverse = orientation(rival, winner)
    if (
        value["winner"] != CLASS_NAMES[winner]
        or value["rival"] != CLASS_NAMES[rival]
        or value["orientation"] != key
        or value["reverse_orientation"] != reverse
        or value["temporal_stratum"] not in TEMPORAL_STRATA
    ):
        raise SidedToleranceError("sided-tolerance row identity is inconsistent")
    counts = (value["boundary_pixel_count"], value["reverse_boundary_pixel_count"])
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
        raise SidedToleranceError("boundary pixel counts must be nonnegative integers")
    norm = _finite_nonnegative(value["head_normal_l2"], name="head_normal_l2")
    if norm <= 0.0:
        raise SidedToleranceError("head normal must be positive")
    margin_quantiles = _validate_quantiles(value["margin_quantiles"], name="margin_quantiles")
    d2_quantiles = _validate_quantiles(value["d2_quantiles"], name="d2_quantiles")
    inner = value["inner_tolerance_d2"]
    outer = value["outer_tolerance_d2"]
    ratio = value["inner_to_outer_ratio"]
    delta = value["asymmetry_delta_d2"]
    if inner is None or outer is None:
        if any(item is not None for item in (inner, outer, ratio, delta)):
            raise SidedToleranceError("empty-sided rows must have all-null tolerance/asymmetry")
    else:
        inner = _finite_nonnegative(inner, name="inner_tolerance_d2")
        outer = _finite_nonnegative(outer, name="outer_tolerance_d2")
        if ratio is not None:
            ratio = _finite_nonnegative(ratio, name="inner_to_outer_ratio")
        delta = float(delta)
        if not math.isfinite(delta):
            raise SidedToleranceError("asymmetry_delta_d2 must be finite")
        if d2_quantiles["q10"] != inner:
            raise SidedToleranceError("inner tolerance must equal the row q10 D2")
    if not isinstance(value["first_rung"], str) or not value["first_rung"].strip():
        raise SidedToleranceError("positive rows require a first rung")
    if not isinstance(value["verdict_scope"], str) or not value["verdict_scope"].strip():
        raise SidedToleranceError("rows require verdict_scope")
    return SidedToleranceRow(
        **{
            **dict(value),
            "head_normal_l2": norm,
            "margin_quantiles": margin_quantiles,
            "d2_quantiles": d2_quantiles,
            "inner_tolerance_d2": inner,
            "outer_tolerance_d2": outer,
            "inner_to_outer_ratio": ratio,
            "asymmetry_delta_d2": delta,
        }
    )


def export_jsonl(
    header: SidedToleranceHeader,
    rows: Sequence[SidedToleranceRow],
) -> bytes:
    """Return canonical JSONL bytes with strict complete 20xstrata coverage."""

    expected_count = ORDERED_PAIR_COUNT * len(TEMPORAL_STRATA)
    if len(rows) != expected_count:
        raise SidedToleranceError(f"expected {expected_count} rows, got {len(rows)}")
    identities = {
        (row.temporal_stratum, row.orientation)
        for row in rows
    }
    expected = {
        (stratum, orientation(winner, rival))
        for stratum in TEMPORAL_STRATA
        for winner in range(CLASS_COUNT)
        for rival in range(CLASS_COUNT)
        if winner != rival
    }
    if identities != expected:
        raise SidedToleranceError("JSONL rows do not cover the full 5x5 off-diagonal matrix")
    lines = [canonical_json_bytes(asdict(header)).rstrip(b"\n")]
    lines.extend(
        canonical_json_bytes(asdict(validate_row(asdict(row)))).rstrip(b"\n")
        for row in sorted(rows, key=lambda item: (item.temporal_stratum, item.winner_id, item.rival_id))
    )
    return b"\n".join(lines) + b"\n"


def parse_jsonl(payload: bytes) -> tuple[SidedToleranceHeader, list[SidedToleranceRow]]:
    try:
        values = [json.loads(line) for line in payload.splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SidedToleranceError("invalid sided-tolerance JSONL") from exc
    if not values or not isinstance(values[0], dict) or set(values[0]) != _HEADER_KEYS:
        raise SidedToleranceError("sided-tolerance header keys are malformed")
    header_value = values[0]
    if (
        header_value.get("schema") != HEADER_SCHEMA
        or header_value.get("axis") != "[macOS-CPU frozen-SegNet advisory]"
        or header_value.get("score_claim") is not False
        or header_value.get("promotion_eligible") is not False
        or header_value.get("class_names") != list(CLASS_NAMES)
        or header_value.get("ordered_pair_count") != ORDERED_PAIR_COUNT
        or header_value.get("temporal_strata") != list(TEMPORAL_STRATA)
    ):
        raise SidedToleranceError("sided-tolerance header authority is malformed")
    header = build_header(
        source_video_sha256=header_value["source_video_sha256"],
        segnet_weights_sha256=header_value["segnet_weights_sha256"],
        upstream_modules_sha256=header_value["upstream_modules_sha256"],
        telemetry_sha256=header_value["telemetry_sha256"],
    )
    rows = [validate_row(value) for value in values[1:]]
    if export_jsonl(header, rows) != payload:
        raise SidedToleranceError("sided-tolerance JSONL is noncanonical")
    return header, rows


def export_e1_bounds(row: SidedToleranceRow) -> dict[str, Any]:
    """Translate one measured row into signed renderer-realizable normal bounds."""

    validated = validate_row(asdict(row))
    if validated.inner_tolerance_d2 is None or validated.outer_tolerance_d2 is None:
        raise SidedToleranceError("e1 bounds require measurements on both boundary sides")
    return {
        "schema": E1_SCHEMA,
        "orientation": validated.orientation,
        "reverse_orientation": validated.reverse_orientation,
        "normal_coordinate": "frozen_head_D2",
        "inner_signed_bound": -validated.inner_tolerance_d2,
        "outer_signed_bound": validated.outer_tolerance_d2,
        "renderer_contract": (
            "e1 must realize both signed bounds through camera uint8, R, and the "
            "frozen SegNet before admission"
        ),
        "score_claim": False,
    }


def price_e2_sided_update(
    row: SidedToleranceRow,
    *,
    realized_inner_excess_d2: float,
    realized_outer_excess_d2: float,
    lambda_inner_seg: float,
    lambda_outer_seg: float,
    pose_objective_delta: float,
    lambda_pose: float,
    delta_archive_bytes: int,
    lambda_byte: float,
) -> dict[str, Any]:
    """Price inner and outer boundary debt with independent e2 multipliers."""

    validated = validate_row(asdict(row))
    values = {
        "realized_inner_excess_d2": realized_inner_excess_d2,
        "realized_outer_excess_d2": realized_outer_excess_d2,
        "lambda_inner_seg": lambda_inner_seg,
        "lambda_outer_seg": lambda_outer_seg,
        "pose_objective_delta": pose_objective_delta,
        "lambda_pose": lambda_pose,
        "lambda_byte": lambda_byte,
    }
    normalized = {
        name: _finite_nonnegative(value, name=name) for name, value in values.items()
    }
    if isinstance(delta_archive_bytes, bool) or not isinstance(delta_archive_bytes, int):
        raise SidedToleranceError("delta_archive_bytes must be an integer")
    inner_price = normalized["lambda_inner_seg"] * normalized["realized_inner_excess_d2"]
    outer_price = normalized["lambda_outer_seg"] * normalized["realized_outer_excess_d2"]
    pose_price = normalized["lambda_pose"] * normalized["pose_objective_delta"]
    byte_price = normalized["lambda_byte"] * delta_archive_bytes
    return {
        "schema": E2_SCHEMA,
        "orientation": validated.orientation,
        "inner_price": inner_price,
        "outer_price": outer_price,
        "pose_price": pose_price,
        "byte_price": byte_price,
        "reduced_cost": inner_price + outer_price + pose_price + byte_price,
        "asymmetric_seg_prices": normalized["lambda_inner_seg"] != normalized["lambda_outer_seg"],
        "score_claim": False,
    }


__all__ = [
    "E1_SCHEMA",
    "E2_SCHEMA",
    "HEADER_SCHEMA",
    "ORDERED_PAIR_COUNT",
    "ROW_SCHEMA",
    "TEMPORAL_STRATA",
    "SidedToleranceError",
    "SidedToleranceHeader",
    "SidedToleranceRow",
    "build_header",
    "build_sided_rows",
    "export_e1_bounds",
    "export_jsonl",
    "orientation",
    "parse_jsonl",
    "price_e2_sided_update",
    "validate_row",
]
