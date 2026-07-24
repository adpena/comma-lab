# SPDX-License-Identifier: MIT
"""Fail-closed custody contract for DDM scorer-metric measurements.

This module does not estimate, solve, price, or adjudicate.  It authenticates
four independently produced measurement components and exposes a single bundle
gate to MS2, PF2R, and RD1.  Every file is rehashed at consumption time.

``COMPLETE`` is intentionally expensive to claim:

* Seg: all 1,200 PF2 buckets carry a measured rank-4 margin-Fisher Gram and
  lambda range over the same SHA-bound atlas.
* Pose: all 600 pairs carry the exact batch-32 Pose6 quadratic/tube and an
  explicit convergence flag.
* composite-R: all 1,200 buckets carry the exact model second order and paired
  realized secants side by side.
* dual readback: all 1,200 buckets carry Fisher-vs-Euclidean cosine and
  relative norm, with Euclidean labeled control-only.

Partial receipts remain useful system intelligence, but can never activate the
minimum-description headline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    TypedStreamTag,
)
from tac.repo_io import sha256_file

BUNDLE_SCHEMA: Final = "ddm_metric_custody_bundle.v1"
COMPONENT_SCHEMA: Final = "ddm_metric_custody_component_receipt.v1"
ARTIFACT_SCHEMA: Final = "ddm_metric_custody_artifact.v1"
PF2_ATLAS_SCHEMA: Final = "ddm_pf2_dimension_conditioned_two_type_measurement.v1"
PF2_BUCKET_SCHEMA: Final = "ddm_pf2_typed_split_atlas_bucket.v2"
G3_REGISTRY_SCHEMA: Final = "ddm_g3_hard_pair_registry.v1"
SEG_DATA_SCHEMA: Final = "ddm_seg_metric_custody.v1"
POSE_DATA_SCHEMA: Final = "ddm_pose_metric_custody.v1"
COMPOSITE_R_DATA_SCHEMA: Final = "ddm_composite_r_second_order_custody.v1"
DUAL_DATA_SCHEMA: Final = "ddm_dual_metric_diagnostics.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
POINTER: Final = "0.1910828242 [contest-CPU]"
PAIR_COUNT: Final = 600
SCORER_BATCH_SIZE: Final = 32
PF2_BUCKET_COUNT: Final = 1200
SEG_HEAD_RANK: Final = 4
POSE_OUTPUT_DIMENSION: Final = 6
HARD_PAIR_ORDER: Final = (
    "top24",
    "top64",
    "stratified_control24",
    "full_n600",
)


class MetricCustodyError(ValueError):
    """Malformed, stale, incomplete, or false-authority metric custody."""


class ComponentId(StrEnum):
    SEG_METRIC = "SEG_METRIC"
    POSE_METRIC = "POSE_METRIC"
    COMPOSITE_R_SECOND_ORDER = "COMPOSITE_R_SECOND_ORDER"
    DUAL_METRIC_DIAGNOSTICS = "DUAL_METRIC_DIAGNOSTICS"


class CustodyStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


COMPONENT_IDS: Final = tuple(ComponentId)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MetricCustodyError(f"{field} must be an exact integer >= {minimum}")
    return value


def _finite(value: object, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricCustodyError(f"{field} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise MetricCustodyError(f"{field} must be finite and >= {minimum}")
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MetricCustodyError(f"cannot read JSON custody artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise MetricCustodyError(f"JSON custody artifact must be an object: {path}")
    return value


def _resolve_path(path: str, *, repository_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.resolve(strict=True)
        if candidate.is_absolute()
        else (repository_root / candidate).resolve(strict=True)
    )


@dataclass(frozen=True, slots=True)
class ArtifactCustody:
    """One immutable input or output file, revalidated on every load."""

    path: str
    bytes: int
    sha256: str
    role: str
    content_schema: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactCustody:
        required = {"schema", "path", "bytes", "sha256", "role", "content_schema"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise MetricCustodyError("artifact custody keys differ from sealed schema")
        if value["schema"] != ARTIFACT_SCHEMA:
            raise MetricCustodyError("artifact custody schema differs")
        path = value["path"]
        role = value["role"]
        content_schema = value["content_schema"]
        if not isinstance(path, str) or not path:
            raise MetricCustodyError("artifact path must be nonempty")
        if not isinstance(role, str) or not role:
            raise MetricCustodyError("artifact role must be nonempty")
        if not isinstance(content_schema, str) or not content_schema:
            raise MetricCustodyError("artifact content_schema must be nonempty")
        if not _is_sha256(value["sha256"]):
            raise MetricCustodyError("artifact sha256 must be lowercase SHA-256")
        return cls(
            path=path,
            bytes=_exact_int(value["bytes"], "artifact bytes"),
            sha256=str(value["sha256"]),
            role=role,
            content_schema=content_schema,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ARTIFACT_SCHEMA,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "role": self.role,
            "content_schema": self.content_schema,
        }

    def revalidate(self, *, repository_root: Path) -> Path:
        try:
            resolved = _resolve_path(self.path, repository_root=repository_root)
        except FileNotFoundError as exc:
            raise MetricCustodyError(f"custody artifact is missing: {self.path}") from exc
        if not resolved.is_file():
            raise MetricCustodyError(f"custody artifact is not a file: {self.path}")
        if resolved.stat().st_size != self.bytes:
            raise MetricCustodyError(f"custody artifact byte drift: {self.path}")
        if sha256_file(resolved) != self.sha256:
            raise MetricCustodyError(f"custody artifact SHA-256 drift: {self.path}")
        value = _read_json(resolved)
        if value.get("schema") != self.content_schema:
            raise MetricCustodyError(f"custody artifact content schema drift: {self.path}")
        return resolved


@dataclass(frozen=True, slots=True)
class ComponentReceipt:
    component_id: ComponentId
    status: CustodyStatus
    sample_count: int
    scorer_batch_size: int
    input_lineage: tuple[ArtifactCustody, ...]
    data_artifact: ArtifactCustody | None
    blockers: tuple[str, ...]
    next_measurement: str
    typed_stream_tags: tuple[TypedStreamTag, ...]

    @property
    def complete(self) -> bool:
        return self.status is CustodyStatus.COMPLETE


@dataclass(frozen=True, slots=True)
class MetricCustodyBundle:
    path: Path
    bundle_id: str
    status: CustodyStatus
    atlas: ArtifactCustody
    hard_pair_registry: ArtifactCustody
    components: Mapping[ComponentId, ComponentReceipt]
    blockers: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.status is CustodyStatus.COMPLETE

    def headline_flags(self) -> dict[str, bool]:
        """Return only the flags this custody bundle has authority to set."""

        return {
            "scorer_metric_active": self.complete,
            "pose_tube_active": (self.components[ComponentId.POSE_METRIC].complete and self.complete),
        }


def artifact_custody(
    path: Path,
    *,
    repository_root: Path,
    role: str,
    content_schema: str,
) -> ArtifactCustody:
    """Build current-file custody without granting scientific authority."""

    resolved = path.expanduser().resolve(strict=True)
    try:
        display = str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        display = str(resolved)
    return ArtifactCustody(
        path=display,
        bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved),
        role=role,
        content_schema=content_schema,
    )


def _matrix(
    value: object,
    *,
    field: str,
    dimension: int,
    positive_semidefinite: bool,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (dimension, dimension)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-11)
    ):
        raise MetricCustodyError(f"{field} must be a finite symmetric {dimension}x{dimension} matrix")
    matrix = 0.5 * (matrix + matrix.T)
    if positive_semidefinite and float(np.linalg.eigvalsh(matrix).min()) < -1e-9:
        raise MetricCustodyError(f"{field} must be positive semidefinite")
    return matrix


def _validate_typed_tag(value: object) -> TypedStreamTag:
    if not isinstance(value, Mapping):
        raise MetricCustodyError("metric table row must carry a TypedStreamTag mapping")
    try:
        tag = TypedStreamTag.from_dict(value)
    except ValueError as exc:
        raise MetricCustodyError("metric table row has invalid TypedStreamTag") from exc
    if tag.layer_home not in {LayerHome.L4_SCORER_FEATURE, LayerHome.L5_VERDICT}:
        raise MetricCustodyError("metric custody rows must live at L4 or L5")
    return tag


def _pf2_rows(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if value.get("schema") != PF2_ATLAS_SCHEMA:
        raise MetricCustodyError("PF2 receipt schema mismatch")
    atlas = value.get("typed_split_atlas")
    if not isinstance(atlas, Mapping):
        raise MetricCustodyError("PF2 typed_split_atlas is missing")
    rows = atlas.get("rows")
    if atlas.get("bucket_count") != PF2_BUCKET_COUNT or not isinstance(rows, list) or len(rows) != PF2_BUCKET_COUNT:
        raise MetricCustodyError("PF2 atlas must contain exactly 1,200 rows")
    result: list[Mapping[str, Any]] = []
    bucket_ids: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("schema") != PF2_BUCKET_SCHEMA
            or not isinstance(row.get("bucket_id"), str)
            or not row["bucket_id"]
            or row["bucket_id"] in bucket_ids
        ):
            raise MetricCustodyError("PF2 atlas rows must be typed with unique bucket IDs")
        bucket_ids.add(row["bucket_id"])
        result.append(row)
    dimensions = atlas.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise MetricCustodyError("PF2 atlas dimensions are missing")
    expected_counts = {
        "class_pairs": 10,
        "class_stratum": 2,
        "visibility": 4,
        "g4_temporal_class": 3,
        "representation_type": 5,
    }
    for field, count in expected_counts.items():
        if not isinstance(dimensions.get(field), list) or len(dimensions[field]) != count:
            raise MetricCustodyError(f"PF2 atlas dimension {field} is incomplete")
    return tuple(result)


def _validate_hard_pair_registry(value: Mapping[str, Any]) -> None:
    if value.get("schema") != G3_REGISTRY_SCHEMA:
        raise MetricCustodyError("G3 hard-pair registry schema mismatch")
    top24 = value.get("top24")
    top64 = value.get("top64")
    control = value.get("stratified_control24")
    if (
        not isinstance(top24, list)
        or not isinstance(top64, list)
        or not isinstance(control, list)
        or len(top24) != 24
        or len(top64) != 64
        or len(control) != 24
        or top64[:24] != top24
    ):
        raise MetricCustodyError("G3 hard-pair subsets are malformed")
    for field, rows in (("top24", top24), ("top64", top64), ("control", control)):
        if len(set(rows)) != len(rows) or any(
            isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < PAIR_COUNT for row in rows
        ):
            raise MetricCustodyError(f"G3 {field} pair IDs are malformed")
    correlations = value.get("correlation_receipt")
    if (
        not isinstance(correlations, Mapping)
        or correlations.get("epistemic_status") != "MEASURED_REPLAY"
        or correlations.get("n_measured_proposals") != 338
    ):
        raise MetricCustodyError("G3 subset-to-full validity receipt is missing")


def _validate_seg_data(
    value: Mapping[str, Any],
    *,
    atlas_sha256: str,
    hard_pair_registry_sha256: str,
    atlas_rows: Sequence[Mapping[str, Any]],
) -> None:
    if (
        value.get("schema") != SEG_DATA_SCHEMA
        or value.get("pf2_atlas_sha256") != atlas_sha256
        or value.get("g3_hard_pair_registry_sha256") != hard_pair_registry_sha256
        or value.get("measurement_schedule") != list(HARD_PAIR_ORDER)
        or value.get("pair_count") != PAIR_COUNT
        or value.get("scorer_batch_size") != SCORER_BATCH_SIZE
        or value.get("head_rank") != SEG_HEAD_RANK
    ):
        raise MetricCustodyError("Seg metric data header mismatch")
    metric_id = value.get("metric_id")
    if (
        not isinstance(metric_id, str)
        or not metric_id
        or "euclid" in metric_id.lower()
        or "identity" in metric_id.lower()
    ):
        raise MetricCustodyError("Seg primary metric must be named non-Euclidean Fisher geometry")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != PF2_BUCKET_COUNT:
        raise MetricCustodyError("Seg metric must cover all 1,200 PF2 buckets")
    atlas_by_id = {str(row["bucket_id"]): row for row in atlas_rows}
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise MetricCustodyError("Seg metric row must be an object")
        bucket_id = row.get("bucket_id")
        if bucket_id not in atlas_by_id or bucket_id in observed:
            raise MetricCustodyError("Seg metric bucket identity differs from PF2 atlas")
        observed.add(str(bucket_id))
        source = atlas_by_id[str(bucket_id)]
        for field in ("class_pair", "class_stratum", "visibility", "g4_temporal_class", "representation_type"):
            if row.get(field) != source.get(field):
                raise MetricCustodyError(f"Seg metric bucket {bucket_id} changed PF2 key {field}")
        gram = _matrix(
            row.get("margin_fisher_gram"),
            field=f"{bucket_id}.margin_fisher_gram",
            dimension=SEG_HEAD_RANK,
            positive_semidefinite=True,
        )
        spectrum = np.asarray(row.get("eigenvalues_ascending"), dtype=np.float64)
        if (
            spectrum.shape != (SEG_HEAD_RANK,)
            or not np.isfinite(spectrum).all()
            or not np.allclose(spectrum, np.linalg.eigvalsh(gram), rtol=1e-7, atol=1e-9)
        ):
            raise MetricCustodyError(f"{bucket_id}: Seg eigenspectrum mismatch")
        lambda_range = row.get("lambda_range")
        if (
            not isinstance(lambda_range, list)
            or len(lambda_range) != 2
            or _finite(lambda_range[0], "lambda minimum", minimum=0.0)
            > _finite(lambda_range[1], "lambda maximum", minimum=0.0)
        ):
            raise MetricCustodyError(f"{bucket_id}: invalid lambda range")
        if row.get("sample_count") != PAIR_COUNT:
            raise MetricCustodyError(f"{bucket_id}: Seg row is not full n600")
        _validate_typed_tag(row.get("typed_stream_tag"))


def _validate_pose_data(
    value: Mapping[str, Any],
    *,
    atlas_sha256: str,
    hard_pair_registry_sha256: str,
    atlas_rows: Sequence[Mapping[str, Any]],
) -> None:
    del atlas_rows
    if (
        value.get("schema") != POSE_DATA_SCHEMA
        or value.get("pf2_atlas_sha256") != atlas_sha256
        or value.get("g3_hard_pair_registry_sha256") != hard_pair_registry_sha256
        or value.get("measurement_schedule") != list(HARD_PAIR_ORDER)
        or value.get("pair_count") != PAIR_COUNT
        or value.get("scorer_batch_size") != SCORER_BATCH_SIZE
        or value.get("output_dimension") != POSE_OUTPUT_DIMENSION
        or value.get("metric_surface") != "EXACT_POSENET_OUTPUT_MSE_QUADRATIC"
    ):
        raise MetricCustodyError("Pose metric data header mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != PAIR_COUNT:
        raise MetricCustodyError("Pose metric must cover exact pair IDs 0..599")
    for pair_id, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("pair_id") != pair_id:
            raise MetricCustodyError("Pose rows must be ordered exact pair IDs 0..599")
        center = np.asarray(row.get("center"), dtype=np.float64)
        factors = np.asarray(row.get("low_rank_factors"), dtype=np.float64)
        rank = row.get("rank")
        if (
            center.shape != (POSE_OUTPUT_DIMENSION,)
            or not np.isfinite(center).all()
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 1 <= rank <= POSE_OUTPUT_DIMENSION
            or factors.shape != (POSE_OUTPUT_DIMENSION, rank)
            or not np.isfinite(factors).all()
        ):
            raise MetricCustodyError(f"Pose pair {pair_id} quadratic is malformed")
        if _finite(row.get("tube_radius"), "Pose tube radius", minimum=0.0) <= 0.0:
            raise MetricCustodyError(f"Pose pair {pair_id} tube radius must be positive")
        converged = row.get("converged")
        convergence_status = row.get("convergence_status")
        if not isinstance(converged, bool):
            raise MetricCustodyError(f"Pose pair {pair_id} convergence flag must be boolean")
        if converged and convergence_status != "CONVERGED":
            raise MetricCustodyError(f"Pose pair {pair_id} converged status mismatch")
        if not converged and (
            not isinstance(convergence_status, str) or not convergence_status.startswith("NON_CONVERGED_")
        ):
            raise MetricCustodyError(f"Pose pair {pair_id} non-convergence must be explicit")
        _validate_typed_tag(row.get("typed_stream_tag"))


def _validate_composite_r_data(
    value: Mapping[str, Any],
    *,
    atlas_sha256: str,
    hard_pair_registry_sha256: str,
    atlas_rows: Sequence[Mapping[str, Any]],
) -> None:
    if (
        value.get("schema") != COMPOSITE_R_DATA_SCHEMA
        or value.get("pf2_atlas_sha256") != atlas_sha256
        or value.get("g3_hard_pair_registry_sha256") != hard_pair_registry_sha256
        or value.get("measurement_schedule") != list(HARD_PAIR_ORDER)
        or value.get("pair_count") != PAIR_COUNT
        or value.get("scorer_batch_size") != SCORER_BATCH_SIZE
        or value.get("kernel_binding") != "separable_resize_full_kernel_direct_sum_v1"
        or value.get("paired_secant_pattern") != "g2f_plus_minus_equal_amplitude"
    ):
        raise MetricCustodyError("composite-R data header mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != PF2_BUCKET_COUNT:
        raise MetricCustodyError("composite-R custody must cover all 1,200 PF2 buckets")
    expected = {str(row["bucket_id"]) for row in atlas_rows}
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("bucket_id") not in expected:
            raise MetricCustodyError("composite-R row has unknown PF2 bucket")
        bucket_id = str(row["bucket_id"])
        if bucket_id in observed:
            raise MetricCustodyError("composite-R bucket IDs must be unique")
        observed.add(bucket_id)
        dimension = _exact_int(row.get("dimension"), "composite-R dimension", minimum=1)
        _matrix(
            row.get("model_hessian"),
            field=f"{bucket_id}.model_hessian",
            dimension=dimension,
            positive_semidefinite=True,
        )
        adjoint = np.asarray(row.get("adjoint_readback"), dtype=np.float64)
        positive = np.asarray(row.get("realized_secant_positive"), dtype=np.float64)
        negative = np.asarray(row.get("realized_secant_negative"), dtype=np.float64)
        if (
            adjoint.shape != (dimension,)
            or positive.shape != (dimension,)
            or negative.shape != (dimension,)
            or not np.isfinite(adjoint).all()
            or not np.isfinite(positive).all()
            or not np.isfinite(negative).all()
        ):
            raise MetricCustodyError(f"{bucket_id}: composite-R vectors are malformed")
        amplitude = _finite(row.get("secant_amplitude"), "secant amplitude", minimum=0.0)
        if amplitude <= 0.0:
            raise MetricCustodyError(f"{bucket_id}: secant amplitude must be positive")
        _validate_typed_tag(row.get("typed_stream_tag"))


def _validate_dual_data(
    value: Mapping[str, Any],
    *,
    atlas_sha256: str,
    hard_pair_registry_sha256: str,
    atlas_rows: Sequence[Mapping[str, Any]],
) -> None:
    if (
        value.get("schema") != DUAL_DATA_SCHEMA
        or value.get("pf2_atlas_sha256") != atlas_sha256
        or value.get("g3_hard_pair_registry_sha256") != hard_pair_registry_sha256
        or value.get("measurement_schedule") != list(HARD_PAIR_ORDER)
        or value.get("pair_count") != PAIR_COUNT
        or value.get("primary_metric") != "MARGIN_FISHER"
        or value.get("control_metric") != "EUCLIDEAN_CONTROL_ONLY"
    ):
        raise MetricCustodyError("dual-metric data header mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != PF2_BUCKET_COUNT:
        raise MetricCustodyError("dual-metric diagnostics must cover all 1,200 PF2 buckets")
    expected = {str(row["bucket_id"]) for row in atlas_rows}
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("bucket_id") not in expected:
            raise MetricCustodyError("dual-metric row has unknown PF2 bucket")
        bucket_id = str(row["bucket_id"])
        if bucket_id in observed:
            raise MetricCustodyError("dual-metric bucket IDs must be unique")
        observed.add(bucket_id)
        cosine = _finite(row.get("fisher_euclidean_cosine"), "dual-metric cosine")
        rel_norm = _finite(
            row.get("fisher_to_euclidean_rel_norm"),
            "dual-metric relative norm",
            minimum=0.0,
        )
        if not -1.0 <= cosine <= 1.0 or rel_norm <= 0.0:
            raise MetricCustodyError(f"{bucket_id}: dual-metric diagnostic is outside range")
        if row.get("euclidean_role") != "LABELED_CONTROL_ONLY":
            raise MetricCustodyError(f"{bucket_id}: Euclidean must be control-only")
        _validate_typed_tag(row.get("typed_stream_tag"))


_DATA_VALIDATORS = {
    ComponentId.SEG_METRIC: _validate_seg_data,
    ComponentId.POSE_METRIC: _validate_pose_data,
    ComponentId.COMPOSITE_R_SECOND_ORDER: _validate_composite_r_data,
    ComponentId.DUAL_METRIC_DIAGNOSTICS: _validate_dual_data,
}


def load_component_receipt(
    path: Path,
    *,
    repository_root: Path,
    atlas_sha256: str,
    hard_pair_registry_sha256: str,
    atlas_rows: Sequence[Mapping[str, Any]],
) -> ComponentReceipt:
    """Load, rehash, and scientifically validate one component receipt."""

    value = _read_json(path)
    required = {
        "schema",
        "component_id",
        "status",
        "evidence_axis",
        "score_claim",
        "research_only",
        "sample_count",
        "scorer_batch_size",
        "input_lineage",
        "data_artifact",
        "blockers",
        "next_measurement",
        "typed_stream_tags",
        "main_landing_review_required",
    }
    if set(value) != required or value.get("schema") != COMPONENT_SCHEMA:
        raise MetricCustodyError("component receipt keys/schema differ")
    try:
        component_id = ComponentId(value["component_id"])
        status = CustodyStatus(value["status"])
    except (TypeError, ValueError) as exc:
        raise MetricCustodyError("component ID/status differs from sealed vocabulary") from exc
    if (
        value["evidence_axis"] != EVIDENCE_AXIS
        or value["score_claim"] is not False
        or value["research_only"] is not True
        or value["main_landing_review_required"] is not True
    ):
        raise MetricCustodyError("component false-authority guard differs")
    lineage_raw = value["input_lineage"]
    if not isinstance(lineage_raw, list) or not lineage_raw:
        raise MetricCustodyError("component input lineage must be nonempty")
    lineage = tuple(ArtifactCustody.from_dict(row) for row in lineage_raw)
    for artifact in lineage:
        artifact.revalidate(repository_root=repository_root)
    lineage_hashes = {artifact.sha256 for artifact in lineage}
    if status is CustodyStatus.COMPLETE and not {
        atlas_sha256,
        hard_pair_registry_sha256,
    }.issubset(lineage_hashes):
        raise MetricCustodyError("COMPLETE component lineage must include exact PF2 and G3 inputs")
    blockers_raw = value["blockers"]
    if (
        not isinstance(blockers_raw, list)
        or any(not isinstance(row, str) or not row for row in blockers_raw)
        or len(blockers_raw) != len(set(blockers_raw))
    ):
        raise MetricCustodyError("component blockers must be unique nonempty strings")
    next_measurement = value["next_measurement"]
    if not isinstance(next_measurement, str) or len(next_measurement.strip()) < 16:
        raise MetricCustodyError("component next_measurement must be substantive")
    tags_raw = value["typed_stream_tags"]
    if not isinstance(tags_raw, list) or not tags_raw:
        raise MetricCustodyError("component typed_stream_tags must be nonempty")
    tags = tuple(_validate_typed_tag(row) for row in tags_raw)
    data_raw = value["data_artifact"]
    data_artifact = None if data_raw is None else ArtifactCustody.from_dict(data_raw)
    sample_count = _exact_int(value["sample_count"], "component sample_count")
    batch_size = _exact_int(value["scorer_batch_size"], "component scorer_batch_size", minimum=1)
    if status is CustodyStatus.COMPLETE:
        if blockers_raw or data_artifact is None:
            raise MetricCustodyError("COMPLETE component cannot carry blockers or omit data")
        if sample_count != PAIR_COUNT or batch_size != SCORER_BATCH_SIZE:
            raise MetricCustodyError("COMPLETE component must be n600 batch32")
        data_path = data_artifact.revalidate(repository_root=repository_root)
        data = _read_json(data_path)
        validator = _DATA_VALIDATORS[component_id]
        validator(
            data,
            atlas_sha256=atlas_sha256,
            hard_pair_registry_sha256=hard_pair_registry_sha256,
            atlas_rows=atlas_rows,
        )
    else:
        if not blockers_raw:
            raise MetricCustodyError("PARTIAL/BLOCKED component must name blockers")
        if data_artifact is not None:
            data_artifact.revalidate(repository_root=repository_root)
    return ComponentReceipt(
        component_id=component_id,
        status=status,
        sample_count=sample_count,
        scorer_batch_size=batch_size,
        input_lineage=lineage,
        data_artifact=data_artifact,
        blockers=tuple(blockers_raw),
        next_measurement=next_measurement,
        typed_stream_tags=tags,
    )


def load_metric_custody_bundle(
    path: str | Path,
    *,
    repository_root: str | Path,
    require_complete: bool = False,
) -> MetricCustodyBundle:
    """Load a bundle, rehash every edge, and optionally require COMPLETE."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path = manifest_path.resolve(strict=True)
    value = _read_json(manifest_path)
    required = {
        "schema",
        "bundle_id",
        "status",
        "evidence_axis",
        "score_claim",
        "research_only",
        "pointer",
        "pointer_moved",
        "pf2_atlas",
        "g3_hard_pair_registry",
        "component_receipts",
        "hard_pair_order",
        "consumers",
        "blockers",
        "headline_admissibility",
        "main_landing_review_required",
    }
    if set(value) != required or value.get("schema") != BUNDLE_SCHEMA:
        raise MetricCustodyError("bundle manifest keys/schema differ")
    bundle_id = value["bundle_id"]
    if not isinstance(bundle_id, str) or not bundle_id:
        raise MetricCustodyError("bundle_id must be nonempty")
    try:
        status = CustodyStatus(value["status"])
    except (TypeError, ValueError) as exc:
        raise MetricCustodyError("bundle status differs from sealed vocabulary") from exc
    if (
        value["evidence_axis"] != EVIDENCE_AXIS
        or value["score_claim"] is not False
        or value["research_only"] is not True
        or value["pointer"] != POINTER
        or value["pointer_moved"] is not False
        or value["main_landing_review_required"] is not True
    ):
        raise MetricCustodyError("bundle false-authority guard differs")

    atlas = ArtifactCustody.from_dict(value["pf2_atlas"])
    atlas_path = atlas.revalidate(repository_root=root)
    atlas_value = _read_json(atlas_path)
    atlas_rows = _pf2_rows(atlas_value)
    hard_registry = ArtifactCustody.from_dict(value["g3_hard_pair_registry"])
    hard_path = hard_registry.revalidate(repository_root=root)
    hard_value = _read_json(hard_path)
    _validate_hard_pair_registry(hard_value)
    if value["hard_pair_order"] != list(HARD_PAIR_ORDER):
        raise MetricCustodyError("bundle does not start from G3 hard-pair order")

    component_refs = value["component_receipts"]
    if not isinstance(component_refs, Mapping) or set(component_refs) != {
        component.value for component in COMPONENT_IDS
    }:
        raise MetricCustodyError("bundle must reference exactly four components")
    components: dict[ComponentId, ComponentReceipt] = {}
    for component_id in COMPONENT_IDS:
        receipt_artifact = ArtifactCustody.from_dict(component_refs[component_id.value])
        receipt_path = receipt_artifact.revalidate(repository_root=root)
        receipt = load_component_receipt(
            receipt_path,
            repository_root=root,
            atlas_sha256=atlas.sha256,
            hard_pair_registry_sha256=hard_registry.sha256,
            atlas_rows=atlas_rows,
        )
        if receipt.component_id is not component_id:
            raise MetricCustodyError("component receipt stored under wrong manifest key")
        components[component_id] = receipt

    blockers_raw = value["blockers"]
    if (
        not isinstance(blockers_raw, list)
        or any(not isinstance(row, str) or not row for row in blockers_raw)
        or len(blockers_raw) != len(set(blockers_raw))
    ):
        raise MetricCustodyError("bundle blockers must be unique nonempty strings")
    component_blockers = list(
        dict.fromkeys(blocker for component_id in COMPONENT_IDS for blocker in components[component_id].blockers)
    )
    if blockers_raw != component_blockers:
        raise MetricCustodyError("bundle blockers must exactly preserve component blocker order")
    all_complete = all(receipt.complete for receipt in components.values())
    expected_status = CustodyStatus.COMPLETE if all_complete else CustodyStatus.PARTIAL
    if status is not expected_status:
        raise MetricCustodyError("bundle status disagrees with component completeness")
    if all_complete and blockers_raw:
        raise MetricCustodyError("COMPLETE bundle cannot carry blockers")
    if not all_complete and not blockers_raw:
        raise MetricCustodyError("PARTIAL bundle must carry blockers")
    headline = value["headline_admissibility"]
    if not isinstance(headline, Mapping) or headline != {
        "bundle_complete": all_complete,
        "scorer_metric_active": all_complete,
        "pose_tube_active": all_complete,
        "score_claim": False,
    }:
        raise MetricCustodyError("headline admissibility does not match bundle state")
    consumers = value["consumers"]
    if consumers != [
        "ms2_typed_quotient_solve",
        "pf2r_metric_active_three_formulation",
        "rd1_dimension_duals",
    ]:
        raise MetricCustodyError("bundle consumer set differs from delegated contract")
    bundle = MetricCustodyBundle(
        path=manifest_path,
        bundle_id=bundle_id,
        status=status,
        atlas=atlas,
        hard_pair_registry=hard_registry,
        components=components,
        blockers=tuple(blockers_raw),
    )
    if require_complete and not bundle.complete:
        raise MetricCustodyError("metric custody bundle is PARTIAL: " + ", ".join(bundle.blockers))
    return bundle


__all__ = [
    "ARTIFACT_SCHEMA",
    "BUNDLE_SCHEMA",
    "COMPONENT_SCHEMA",
    "COMPOSITE_R_DATA_SCHEMA",
    "DUAL_DATA_SCHEMA",
    "EVIDENCE_AXIS",
    "G3_REGISTRY_SCHEMA",
    "HARD_PAIR_ORDER",
    "PAIR_COUNT",
    "PF2_BUCKET_COUNT",
    "POSE_DATA_SCHEMA",
    "SCORER_BATCH_SIZE",
    "SEG_DATA_SCHEMA",
    "ArtifactCustody",
    "ComponentId",
    "ComponentReceipt",
    "CustodyStatus",
    "MetricCustodyBundle",
    "MetricCustodyError",
    "artifact_custody",
    "load_component_receipt",
    "load_metric_custody_bundle",
]
