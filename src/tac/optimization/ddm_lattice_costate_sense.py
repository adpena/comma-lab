# SPDX-License-Identifier: MIT
"""Typed solver-member telemetry for the live DDM costate SENSE surface.

The producer is advisory and read-only.  It records where exact member
selection is constrained, what the real residual coder paid, and which local
degrees of freedom remain.  Missing optimizer duals are represented as
unavailable with an explicit reason; they are never imputed from margins.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.mdl_polytope_member import modular_uint8_residual

PAIR_SCHEMA = "ddm_min_description_lattice_sense_pair.v1"
FACTOR_SCHEMA = "ddm_min_description_lattice_sense_factorization.v1"
PRODUCER_ID = "ddm_ms1_min_description_lattice_solve"
POOL_ID = "solver_member_selection"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"


class LatticeSenseError(ValueError):
    """Malformed solver telemetry or non-reproducible factorization."""


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise LatticeSenseError(f"{field} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise LatticeSenseError(f"{field} must be finite and nonnegative")
    return result


def _strata(labels: np.ndarray) -> dict[str, np.ndarray]:
    value = np.asarray(labels)
    if value.ndim != 2 or value.dtype.kind not in "iu" or value.size == 0:
        raise LatticeSenseError("labels must be a nonempty integer HxW field")
    edge = np.zeros(value.shape, dtype=bool)
    vertical = value[1:] != value[:-1]
    horizontal = value[:, 1:] != value[:, :-1]
    edge[1:] |= vertical
    edge[:-1] |= vertical
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    saddle = np.zeros(value.shape, dtype=bool)
    if min(value.shape) >= 2:
        a = value[:-1, :-1]
        b = value[1:, :-1]
        c = value[:-1, 1:]
        d = value[1:, 1:]
        three_class = (
            ((a != b) & (a != c) & (b != c))
            | ((a != b) & (a != d) & (b != d))
            | ((a != c) & (a != d) & (c != d))
            | ((b != c) & (b != d) & (c != d))
        )
        saddle[:-1, :-1] |= three_class
        saddle[1:, :-1] |= three_class
        saddle[:-1, 1:] |= three_class
        saddle[1:, 1:] |= three_class
    return {"cell": ~edge, "edge": edge & ~saddle, "saddle": saddle}


@dataclass(frozen=True)
class LatticeSensePair:
    pair_id: int
    selected_sha256: str
    origin_sha256: str
    residual_sha256: str
    canonical_member_bytes: int
    selected_residual_bytes: int
    active_constraint_count: int
    constraint_count: int
    active_by_class: Mapping[str, int]
    active_by_stratum: Mapping[str, int]
    shadow_prices: Mapping[str, Any]
    degeneracy: Mapping[str, Any]
    residual: Mapping[str, Any]
    basis: Mapping[str, Any]
    dimension_typing: Mapping[str, Any]
    tie_break: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIR_SCHEMA,
            "producer_id": PRODUCER_ID,
            "pair_id": self.pair_id,
            "pool_id": POOL_ID,
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "selected_sha256": self.selected_sha256,
            "origin_sha256": self.origin_sha256,
            "residual_sha256": self.residual_sha256,
            "rate": {
                "canonical_member_bytes": self.canonical_member_bytes,
                "selected_residual_bytes": self.selected_residual_bytes,
                "delta_bytes": self.selected_residual_bytes
                - self.canonical_member_bytes,
                "coder": "zlib-9 exact modular uint8 residual",
                "partition": {
                    "FREE": "generic modular add and integer basis construction",
                    "NULL": "pinned invisible coordinates; zero payload credit",
                    "COUNTED": self.selected_residual_bytes,
                },
            },
            "active_set": {
                "active_constraint_count": self.active_constraint_count,
                "constraint_count": self.constraint_count,
                "active_fraction": self.active_constraint_count
                / self.constraint_count,
                "per_class": dict(self.active_by_class),
                "per_stratum": dict(self.active_by_stratum),
                "frame_scope": {
                    "frame_0": "seg_free_pose_coupled",
                    "frame_1": "seg_argmax_and_pose_coupled",
                },
            },
            "shadow_prices": dict(self.shadow_prices),
            "degeneracy": dict(self.degeneracy),
            "residual": dict(self.residual),
            "basis": dict(self.basis),
            "dimension_typing": dict(self.dimension_typing),
            "tie_break": self.tie_break,
        }


def build_lattice_sense_pair(
    *,
    pair_id: int,
    selected: np.ndarray,
    origin: np.ndarray,
    labels: np.ndarray,
    winner_rival_margins: np.ndarray,
    canonical_member_bytes: int,
    selected_residual_bytes: int,
    active_tolerance: float,
    basis_norms: Sequence[float] | Mapping[str, Any],
    local_facet_dimensions: np.ndarray | None = None,
    duals: np.ndarray | None = None,
) -> LatticeSensePair:
    """Build one exact, costate-consumable pair row."""

    if isinstance(pair_id, bool) or not isinstance(pair_id, int) or pair_id < 0:
        raise LatticeSenseError("pair_id must be a nonnegative exact integer")
    pair = np.asarray(selected)
    base = np.asarray(origin)
    if (
        pair.dtype != np.uint8
        or base.dtype != np.uint8
        or pair.shape != base.shape
        or pair.ndim != 4
        or pair.shape[0] != 2
        or pair.shape[-1] != 3
    ):
        raise LatticeSenseError("selected/origin must be same-shape uint8 (2,H,W,3)")
    target = np.asarray(labels)
    margins = np.asarray(winner_rival_margins, dtype=np.float64)
    if target.shape != margins.shape or target.ndim != 2:
        raise LatticeSenseError("labels and margins must share scorer HxW geometry")
    if target.dtype.kind not in "iu" or not np.isfinite(margins).all() or np.any(margins < 0):
        raise LatticeSenseError("labels/margins leave their valid domains")
    tolerance = _finite_nonnegative(active_tolerance, "active_tolerance")
    canonical_bytes = int(canonical_member_bytes)
    residual_bytes = int(selected_residual_bytes)
    if (
        isinstance(canonical_member_bytes, bool)
        or isinstance(selected_residual_bytes, bool)
        or canonical_bytes <= 0
        or residual_bytes <= 0
    ):
        raise LatticeSenseError("coder byte counts must be positive exact integers")
    active = margins <= tolerance
    strata = _strata(target)
    active_by_class = {
        str(class_id): int(np.count_nonzero(active & (target == class_id)))
        for class_id in sorted(int(value) for value in np.unique(target))
    }
    active_by_stratum = {
        name: int(np.count_nonzero(active & mask))
        for name, mask in strata.items()
    }
    if isinstance(basis_norms, Mapping):
        try:
            basis = {
                "reduction": "saturated_integer_kernel_plus_exact_size_reduction",
                "count": int(basis_norms["count"]),
                "norm_min": float(basis_norms["norm_min"]),
                "norm_p50": float(basis_norms["norm_p50"]),
                "norm_p95": float(basis_norms["norm_p95"]),
                "norm_max": float(basis_norms["norm_max"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise LatticeSenseError("basis summary is malformed") from exc
        ordered = (
            basis["norm_min"],
            basis["norm_p50"],
            basis["norm_p95"],
            basis["norm_max"],
        )
        if (
            basis["count"] <= 0
            or not all(math.isfinite(value) and value > 0 for value in ordered)
            or tuple(sorted(ordered)) != ordered
        ):
            raise LatticeSenseError("basis summary must be positive and ordered")
    else:
        norms = np.asarray(tuple(basis_norms), dtype=np.float64)
        if (
            norms.ndim != 1
            or norms.size == 0
            or not np.isfinite(norms).all()
            or np.any(norms <= 0)
        ):
            raise LatticeSenseError("basis_norms must be finite positive values")
        basis = {
            "reduction": "saturated_integer_kernel_plus_exact_size_reduction",
            "count": int(norms.size),
            "norm_min": float(norms.min()),
            "norm_p50": float(np.quantile(norms, 0.5)),
            "norm_p95": float(np.quantile(norms, 0.95)),
            "norm_max": float(norms.max()),
        }
    if duals is None:
        shadow_prices: dict[str, Any] = {
            "available": False,
            "scope": "per_typed_dimension_bucket",
            "pooling": "FORBIDDEN",
            "reason": (
                "bounded integer projection does not expose KKT multipliers; "
                "no pooled lambda or margin proxy imputed"
            ),
            "values": None,
        }
    else:
        raise LatticeSenseError(
            "pooled dual arrays are forbidden; emit duals per "
            "(stratum x scorer-visibility x g4 temporal class) bucket"
        )
    if local_facet_dimensions is None:
        degeneracy: dict[str, Any] = {
            "available": False,
            "reason": "local facet rank was not emitted by this solve form",
            "deterministic_tie_break": "coded_bytes_then_member_sha256",
        }
    else:
        dimensions = np.asarray(local_facet_dimensions)
        if (
            dimensions.dtype.kind not in "iu"
            or dimensions.size == 0
            or np.any(dimensions < 0)
            or np.any(dimensions > 3)
        ):
            raise LatticeSenseError("local facet dimensions must be integer values in [0,3]")
        degeneracy = {
            "available": True,
            "minimum_local_facet_dimension": int(dimensions.min()),
            "maximum_local_facet_dimension": int(dimensions.max()),
            "mean_local_facet_dimension": float(dimensions.mean()),
            "histogram": {
                str(value): int(np.count_nonzero(dimensions == value))
                for value in range(4)
            },
            "sha256": _sha256_array(dimensions),
            "deterministic_tie_break": "coded_bytes_then_member_sha256",
        }
    residual = modular_uint8_residual(pair, base)
    signed = pair.astype(np.int16) - base.astype(np.int16)
    return LatticeSensePair(
        pair_id=pair_id,
        selected_sha256=_sha256_array(pair),
        origin_sha256=_sha256_array(base),
        residual_sha256=_sha256_array(residual),
        canonical_member_bytes=canonical_bytes,
        selected_residual_bytes=residual_bytes,
        active_constraint_count=int(np.count_nonzero(active)),
        constraint_count=int(active.size),
        active_by_class=active_by_class,
        active_by_stratum=active_by_stratum,
        shadow_prices=shadow_prices,
        degeneracy=degeneracy,
        residual={
            "nonzero_values": int(np.count_nonzero(residual)),
            "signed_l1": int(np.abs(signed.astype(np.int32)).sum()),
            "signed_l2": float(np.linalg.norm(signed.astype(np.float64).ravel())),
        },
        basis={
            "metric": "identity_euclidean_diagnostic_only",
            "headline_eligible": False,
            "successor_metric": (
                "seg_rank4_head_x_margin_fisher_plus_pose_low_rank_quadratic"
            ),
            **basis,
        },
        dimension_typing={
            "status": "BLOCKED_TYPED_ATLAS_NOT_BOUND",
            "required_bucket_axes": [
                "stratum",
                "scorer_visibility",
                "g4_temporal_class",
            ],
            "available_axes": ["stratum"],
            "missing_axes": ["scorer_visibility", "g4_temporal_class"],
            "scorer_visibility_values": [
                "ker(A)-invisible",
                "seg-visible",
                "pose-visible",
            ],
            "effective_quantum": (
                "BLOCKED; requires uint8-step x per-dimension scorer sensitivity"
            ),
            "reconciliation": (
                "consume ddm_pf2_dimension_conditioned_two_type typed atlas "
                "when its custodied checkpoint lands"
            ),
        },
        tie_break="minimum_real_coder_bytes_then_member_sha256",
    )


def factorize_lattice_sense(
    rows: Sequence[Mapping[str, Any]],
    *,
    coder_noise_floor_bytes: int,
    maximum_factors: int = 8,
    per_factor_coder_race: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """SVD-factorize telemetry and admit vocabulary roles only by coder race.

    A numerical factor above the coder noise floor is not yet a distilled
    vocabulary column.  Its per-stratum SKELETON-versus-FIBER role is admitted
    only when a measured coder race has a strict winner; ties and missing races
    remain explicitly unrouted.
    """

    values = list(rows)
    if not values:
        raise LatticeSenseError("factorization requires at least one pair row")
    if (
        isinstance(coder_noise_floor_bytes, bool)
        or not isinstance(coder_noise_floor_bytes, int)
        or coder_noise_floor_bytes < 0
    ):
        raise LatticeSenseError("coder_noise_floor_bytes must be a nonnegative exact integer")
    if not isinstance(maximum_factors, int) or not 1 <= maximum_factors <= 64:
        raise LatticeSenseError("maximum_factors must lie in [1,64]")
    class_ids = sorted(
        {
            class_id
            for row in values
            for class_id in row["active_set"]["per_class"]
        }
    )
    feature_names = [
        "rate_delta_bytes",
        "residual_nonzero_values",
        "residual_signed_l1",
        *(f"active_class_{class_id}" for class_id in class_ids),
        "active_cell",
        "active_edge",
        "active_saddle",
    ]
    matrix_rows: list[list[float]] = []
    pair_ids: list[int] = []
    for row in values:
        if row.get("schema") != PAIR_SCHEMA:
            raise LatticeSenseError("factorization input schema drift")
        pair_ids.append(int(row["pair_id"]))
        matrix_rows.append(
            [
                float(row["rate"]["delta_bytes"]),
                float(row["residual"]["nonzero_values"]),
                float(row["residual"]["signed_l1"]),
                *(
                    float(row["active_set"]["per_class"].get(class_id, 0))
                    for class_id in class_ids
                ),
                *(
                    float(row["active_set"]["per_stratum"].get(name, 0))
                    for name in ("cell", "edge", "saddle")
                ),
            ]
        )
    if len(set(pair_ids)) != len(pair_ids):
        raise LatticeSenseError("factorization pair IDs must be unique")
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0] = 1.0
    normalized = (matrix - means) / scales
    left, singular, right = np.linalg.svd(normalized, full_matrices=False)
    count = min(maximum_factors, len(singular))
    byte_scale = scales[0]
    factors: list[dict[str, Any]] = []
    for index in range(count):
        byte_amplitude = abs(float(singular[index] * right[index, 0] * byte_scale))
        above_noise = byte_amplitude >= coder_noise_floor_bytes
        race = (
            None
            if per_factor_coder_race is None
            else per_factor_coder_race.get(index)
        )
        representation: dict[str, Any]
        if race is None:
            representation = {
                "distilled": False,
                "status": "BLOCKED_NO_PER_STRATUM_CODER_RACE",
                "tag": None,
                "stratum": None,
            }
        else:
            try:
                stratum = str(race["stratum"])
                skeleton_bytes = int(race["skeleton_bytes"])
                fiber_bytes = int(race["fiber_bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LatticeSenseError(
                    f"factor {index} coder race is malformed"
                ) from exc
            if (
                not stratum
                or isinstance(race.get("skeleton_bytes"), bool)
                or isinstance(race.get("fiber_bytes"), bool)
                or skeleton_bytes < 0
                or fiber_bytes < 0
            ):
                raise LatticeSenseError(
                    f"factor {index} coder race leaves its valid domain"
                )
            if skeleton_bytes == fiber_bytes:
                representation = {
                    "distilled": False,
                    "status": "BLOCKED_CODER_RACE_TIE",
                    "tag": None,
                    "stratum": stratum,
                    "skeleton_bytes": skeleton_bytes,
                    "fiber_bytes": fiber_bytes,
                }
            else:
                tag = (
                    "SKELETON"
                    if skeleton_bytes < fiber_bytes
                    else "FIBER"
                )
                representation = {
                    "distilled": above_noise,
                    "status": (
                        "DISTILLED_BY_MEASURED_CODER_RACE"
                        if above_noise
                        else "BLOCKED_BELOW_REAL_CODER_NOISE"
                    ),
                    "tag": tag if above_noise else None,
                    "stratum": stratum,
                    "skeleton_bytes": skeleton_bytes,
                    "fiber_bytes": fiber_bytes,
                    "route": (
                        ["pf1_token_coder", "g1_skeleton_coder"]
                        if tag == "SKELETON"
                        else [
                            "transform_quantize_entropy",
                            "stratum_amplitude_law_coder",
                        ]
                    )
                    if above_noise
                    else [],
                }
        factors.append(
            {
                "factor_index": index,
                "singular_value": float(singular[index]),
                "byte_amplitude": byte_amplitude,
                "above_real_coder_noise_floor": above_noise,
                "representation": representation,
                "loadings": {
                    name: float(value)
                    for name, value in zip(
                        feature_names, right[index], strict=True
                    )
                },
                "pair_scores_sha256": _sha256_array(left[:, index]),
            }
        )
    return {
        "schema": FACTOR_SCHEMA,
        "producer_id": PRODUCER_ID,
        "pool_id": POOL_ID,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "pair_count": len(values),
        "feature_names": feature_names,
        "coder_noise_floor_bytes": coder_noise_floor_bytes,
        "factors": factors,
        "admitted_factor_count": sum(
            row["above_real_coder_noise_floor"] for row in factors
        ),
        "distilled_factor_count": sum(
            row["representation"]["distilled"] for row in factors
        ),
        "distillation_rule": (
            "a factor must clear real coder noise and win a measured per-stratum "
            "SKELETON-versus-FIBER coder race; no default representation role"
        ),
        "routes": [
            "v18_column_generation",
            "dv1_dv2_grammar",
            "menu1_nonadditive_pool",
            "rule118_free_basis_adjudication",
        ],
        "matrix_sha256": _sha256_array(matrix),
    }


def write_sense_jsonl_atomic(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Publish a canonical JSONL SENSE artifact atomically."""

    values = list(rows)
    if not values or any(row.get("schema") != PAIR_SCHEMA for row in values):
        raise LatticeSenseError("JSONL publication requires typed pair rows")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
        for row in values
    )
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "EVIDENCE_AXIS",
    "FACTOR_SCHEMA",
    "PAIR_SCHEMA",
    "POOL_ID",
    "PRODUCER_ID",
    "LatticeSenseError",
    "LatticeSensePair",
    "build_lattice_sense_pair",
    "factorize_lattice_sense",
    "write_sense_jsonl_atomic",
]
