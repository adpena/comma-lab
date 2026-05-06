#!/usr/bin/env python3
"""Plan Alpha lossy-geometry plus sparse-repair byte budgets.

This empirical planner consumes two existing non-promotable Alpha artifacts:
an ``alpha_mask_codec_candidate_matrix_v1`` manifest and an
``alpha_mask_primitive_component_response_plan_v1`` plan.  It does not decode
masks, does not run scorer networks, does not launch jobs, and does not build
archives.  Its output is a deterministic set of byte-budget estimates and
archive-build specs to revisit after official CUDA component-response evidence
returns.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "experiments/alpha_lossy_repair_budget_planner.py"
SCHEMA = "alpha_lossy_repair_budget_planner_v1"
SPEC_SCHEMA = "alpha_lossy_sparse_repair_archive_build_spec_v1"
REPORT_NAME = "alpha_lossy_repair_budget_plan.json"
SPECS_DIR_NAME = "candidate_archive_specs"
EVIDENCE_GRADE = "empirical"
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this planner. All byte budgets are empirical "
    "planning estimates. A concrete archive, payload closure, and exact CUDA "
    "auth eval are required before ranking, promotion, retirement, or any "
    "score claim."
)

DEFAULT_MATRIX_MANIFEST = (
    REPO_ROOT
    / "experiments/results/alpha_mask_codec_candidate_matrix_pfp16_20260501_full/"
    / "alpha_mask_codec_candidate_matrix.json"
)
DEFAULT_PRIMITIVE_PLAN = (
    REPO_ROOT
    / "experiments/results/alpha_mask_primitive_response_plan_pfp16_20260501_r1/"
    / "alpha_mask_primitive_response_plan.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/alpha_lossy_repair_budget_planner"
PROTECTED_CLASS_IDS = (1, 2)


class AlphaLossyRepairPlannerError(ValueError):
    """Raised when planner inputs are unsafe or scientifically over-claimed."""


@dataclass(frozen=True)
class BudgetConfig:
    """Byte-model knobs for deterministic planning estimates.

    These values are not learned scorer signal.  They define a transparent
    sparse-patch container model so candidate repair budgets can be compared
    before CUDA component-response data is available.
    """

    max_primitive_points: int = 64
    max_specs: int = 12
    max_policy_points: int = 8
    target_mask_member_bytes: int | None = None
    repair_global_header_bytes: int = 32
    repair_point_header_bytes: int = 24
    repair_primitive_ref_bytes: int = 2
    repair_run_record_bytes: int = 6
    repair_pixels_per_run: int = 64
    repair_bits_per_pixel: float = 3.0
    repair_compression_ratio: float = 0.70
    lossy_base_bytes: tuple[int, ...] = ()


@dataclass(frozen=True)
class PrimitivePoint:
    index: int
    primitive_id: str
    kind: str
    operation: str
    frame_index: int | None
    source_class: int | None
    target_class: int | None
    selection_weight: float
    changed_pixels: int
    selected_pixels_before_cap: int | None
    selected_pixels_after_cap: int | None
    mask_member_size_bytes: int
    archive_bytes: int
    archive_sha256: str
    archive_path: str


@dataclass(frozen=True)
class LossyBaseCandidate:
    base_id: str
    source: str
    estimated_mask_member_bytes: int
    details: Mapping[str, Any]


@dataclass(frozen=True)
class RepairPolicy:
    policy_id: str
    policy_kind: str
    selected_points: tuple[PrimitivePoint, ...]
    details: Mapping[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AlphaLossyRepairPlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AlphaLossyRepairPlannerError(f"{path} must contain a JSON object")
    return payload


def _round_float(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlphaLossyRepairPlannerError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise AlphaLossyRepairPlannerError(f"{field} must be finite")
    return out


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlphaLossyRepairPlannerError(f"{field} must be an integer")
    return int(value)


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlphaLossyRepairPlannerError(f"{field} must be an integer")
    if value < 0:
        raise AlphaLossyRepairPlannerError(f"{field} must be nonnegative")
    return int(value)


def _positive_int(value: Any, *, field: str) -> int:
    out = _nonnegative_int(value, field=field)
    if out <= 0:
        raise AlphaLossyRepairPlannerError(f"{field} must be positive")
    return out


def _require_false(payload: Mapping[str, Any], key: str, *, context: str) -> None:
    if payload.get(key) is not False:
        raise AlphaLossyRepairPlannerError(f"{context}.{key} must be false")


def _require_non_promotable(payload: Mapping[str, Any], *, context: str) -> None:
    _require_false(payload, "score_claim", context=context)
    _require_false(payload, "promotion_eligible", context=context)


def _require_exact_cuda_warning(payload: Mapping[str, Any], *, field: str, context: str) -> None:
    if "contest_auth_eval.py --device cuda" not in str(payload.get(field, "")):
        raise AlphaLossyRepairPlannerError(f"{context}.{field} must require exact CUDA auth eval")


def _validate_config(config: BudgetConfig) -> None:
    _positive_int(config.max_primitive_points, field="max_primitive_points")
    _positive_int(config.max_specs, field="max_specs")
    _positive_int(config.max_policy_points, field="max_policy_points")
    if config.target_mask_member_bytes is not None:
        _positive_int(config.target_mask_member_bytes, field="target_mask_member_bytes")
    _positive_int(config.repair_global_header_bytes, field="repair_global_header_bytes")
    _positive_int(config.repair_point_header_bytes, field="repair_point_header_bytes")
    _positive_int(config.repair_primitive_ref_bytes, field="repair_primitive_ref_bytes")
    _positive_int(config.repair_run_record_bytes, field="repair_run_record_bytes")
    _positive_int(config.repair_pixels_per_run, field="repair_pixels_per_run")
    if config.repair_bits_per_pixel <= 0.0:
        raise AlphaLossyRepairPlannerError("repair_bits_per_pixel must be positive")
    if not (0.0 < config.repair_compression_ratio <= 4.0):
        raise AlphaLossyRepairPlannerError("repair_compression_ratio must be in (0, 4]")
    for index, value in enumerate(config.lossy_base_bytes):
        _positive_int(value, field=f"lossy_base_bytes[{index}]")


def _safe_relative_path(raw: str, *, field: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise AlphaLossyRepairPlannerError(f"{field} is an unsafe empty/NUL/backslash path")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise AlphaLossyRepairPlannerError(f"{field} must be relative, got {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise AlphaLossyRepairPlannerError(f"{field} must be relative, got {raw!r}")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise AlphaLossyRepairPlannerError(f"{field} must not contain traversal: {raw!r}")
    if any(part.startswith(".") or part.startswith("._") for part in parts):
        raise AlphaLossyRepairPlannerError(f"{field} hidden/system path rejected: {raw!r}")
    if any(part == "__MACOSX" for part in parts):
        raise AlphaLossyRepairPlannerError(f"{field} hidden/system path rejected: {raw!r}")
    return path


def _resolve_reference_path(raw: str, *, manifest_dir: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise AlphaLossyRepairPlannerError(f"{field} is an unsafe empty/NUL path")
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"{field} does not exist: {resolved}")
        return resolved

    _safe_relative_path(raw, field=field)
    candidates = [manifest_dir / raw, REPO_ROOT / raw, Path.cwd() / raw]
    for item in candidates:
        resolved = item.resolve()
        if resolved.exists():
            return resolved
    searched = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"{field} path {raw!r} does not exist; searched {searched}")


def _resolve_plan_relative_path(raw: str, *, plan_dir: Path, field: str) -> Path:
    rel = _safe_relative_path(raw, field=field)
    resolved = (plan_dir / Path(*rel.parts)).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    return resolved


def _path_hint(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _file_meta(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if root is not None:
        try:
            path_hint = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            path_hint = _path_hint(resolved)
    else:
        path_hint = _path_hint(resolved)
    return {
        "path": path_hint,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _source_file_meta(path: Path) -> dict[str, Any]:
    if path.exists():
        return _file_meta(path)
    return {
        "path": _path_hint(path),
        "exists": False,
    }


def _validate_file_record(
    path: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    size_keys: tuple[str, ...] = ("size_bytes", "bytes"),
    sha_keys: tuple[str, ...] = ("sha256",),
) -> dict[str, Any]:
    actual = _file_meta(path)
    expected_size: Any = None
    for key in size_keys:
        if key in record:
            expected_size = record[key]
            break
    if expected_size is not None and int(expected_size) != actual["size_bytes"]:
        raise AlphaLossyRepairPlannerError(
            f"{label} size mismatch: manifest={expected_size} actual={actual['size_bytes']}"
        )
    expected_sha: Any = None
    for key in sha_keys:
        if key in record:
            expected_sha = record[key]
            break
    if expected_sha is not None and str(expected_sha) != actual["sha256"]:
        raise AlphaLossyRepairPlannerError(
            f"{label} sha256 mismatch: manifest={expected_sha} actual={actual['sha256']}"
        )
    actual["validated_against_manifest"] = True
    return actual


def _source_mask_member_size(source: Mapping[str, Any]) -> int:
    mask_member = source.get("mask_member")
    if not isinstance(mask_member, Mapping):
        raise AlphaLossyRepairPlannerError("matrix source.mask_member missing")
    raw = mask_member.get("size_bytes", mask_member.get("raw_bytes"))
    return _positive_int(raw, field="source.mask_member.size_bytes")


def _load_matrix_context(matrix_manifest: Path) -> dict[str, Any]:
    manifest_path = matrix_manifest.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "alpha_mask_codec_candidate_matrix_v1":
        raise AlphaLossyRepairPlannerError(f"unsupported matrix schema {manifest.get('schema')!r}")
    _require_non_promotable(manifest, context="matrix")
    _require_false(manifest, "scorer_network_loaded", context="matrix")
    _require_exact_cuda_warning(manifest, field="canonical_score_source_required", context="matrix")
    if manifest.get("evidence_grade") != EVIDENCE_GRADE:
        raise AlphaLossyRepairPlannerError("matrix evidence_grade must be empirical")

    manifest_dir = manifest_path.parent
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise AlphaLossyRepairPlannerError("matrix source object missing")

    source_archive_meta: dict[str, Any] | None = None
    if source.get("archive_path"):
        archive_path = _resolve_reference_path(
            str(source["archive_path"]),
            manifest_dir=manifest_dir,
            field="matrix.source.archive_path",
        )
        source_archive_meta = _validate_file_record(
            archive_path,
            {
                "size_bytes": source.get("archive_size_bytes"),
                "sha256": source.get("archive_sha256"),
            },
            label="matrix source archive",
        )

    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(manifest.get("candidates", [])):
        if not isinstance(candidate, Mapping):
            raise AlphaLossyRepairPlannerError(f"matrix candidates[{index}] must be an object")
        _require_non_promotable(candidate, context=f"matrix.candidates[{index}]")
        if candidate.get("skipped") is True:
            continue
        artifact = candidate.get("artifact")
        if not isinstance(artifact, Mapping):
            raise AlphaLossyRepairPlannerError(f"matrix candidates[{index}].artifact missing")
        artifact_path = _resolve_reference_path(
            str(artifact.get("path", "")),
            manifest_dir=manifest_dir,
            field=f"matrix.candidates[{index}].artifact.path",
        )
        artifact_meta = _validate_file_record(
            artifact_path,
            artifact,
            label=f"matrix candidate artifact {candidate.get('family', index)!r}",
        )
        agreement = candidate.get("agreement")
        if not isinstance(agreement, Mapping):
            raise AlphaLossyRepairPlannerError(f"matrix candidates[{index}].agreement missing")
        candidates.append(
            {
                "name": str(candidate.get("name", "")),
                "family": str(candidate.get("family", "")),
                "payload_format": str(candidate.get("payload_format", "")),
                "exact_reconstruction": bool(candidate.get("exact_reconstruction")),
                "diagnostic_reference": bool(candidate.get("diagnostic_reference")),
                "argmax_agreement": _round_float(float(agreement.get("argmax_agreement", 0.0))),
                "different_pixels": int(agreement.get("different_pixels", 0)),
                "size_bytes": int(artifact_meta["size_bytes"]),
                "sha256": artifact_meta["sha256"],
                "path": artifact_meta["path"],
                "candidate_archive_member": artifact.get("candidate_archive_member"),
                "runtime_decoder_integration_required": bool(
                    candidate.get("runtime_decoder_integration_required", True)
                ),
            }
        )

    source_mask_bytes = _source_mask_member_size(source)
    exact_candidates = [
        item
        for item in candidates
        if item["exact_reconstruction"] is True and item["diagnostic_reference"] is False
    ]
    best_exact = min(exact_candidates, key=lambda item: (item["size_bytes"], item["family"]), default=None)
    return {
        "manifest": manifest,
        "manifest_file": _file_meta(manifest_path),
        "source_archive": source_archive_meta,
        "source_mask_member_size_bytes": int(source_mask_bytes),
        "source_decoded_mask_sha256": (
            source.get("decoded_masks", {}).get("class_id_u8_sha256")
            if isinstance(source.get("decoded_masks"), Mapping)
            else None
        ),
        "source_mask_member": dict(source.get("mask_member", {})),
        "candidates": candidates,
        "best_exact_candidate": best_exact,
    }


def _load_primitive_context(primitive_plan: Path, *, config: BudgetConfig) -> dict[str, Any]:
    plan_path = primitive_plan.resolve()
    plan = _read_json(plan_path)
    if plan.get("format") != "official_component_response_plan_v1":
        raise AlphaLossyRepairPlannerError(f"unsupported primitive plan format {plan.get('format')!r}")
    if plan.get("alpha_plan_format") != "alpha_mask_primitive_component_response_plan_v1":
        raise AlphaLossyRepairPlannerError(
            f"unsupported alpha primitive plan format {plan.get('alpha_plan_format')!r}"
        )
    _require_non_promotable(plan, context="primitive_plan")
    _require_false(plan, "official_component_response", context="primitive_plan")
    _require_false(plan, "scorer_network_loaded", context="primitive_plan")
    _require_exact_cuda_warning(plan, field="canonical_score_source_required", context="primitive_plan")
    if plan.get("evidence_grade") != EVIDENCE_GRADE:
        raise AlphaLossyRepairPlannerError("primitive plan evidence_grade must be empirical")

    plan_dir = plan_path.parent
    perturbation = plan.get("perturbation")
    if not isinstance(perturbation, Mapping):
        raise AlphaLossyRepairPlannerError("primitive plan perturbation object missing")
    variants_rel = str(perturbation.get("archive_variants_manifest", ""))
    variants_path = _resolve_plan_relative_path(
        variants_rel,
        plan_dir=plan_dir,
        field="primitive_plan.perturbation.archive_variants_manifest",
    )
    variants_meta = _validate_file_record(
        variants_path,
        {"sha256": perturbation.get("archive_variants_manifest_sha256")},
        label="primitive archive variants manifest",
        size_keys=(),
    )
    variants = _read_json(variants_path)
    if variants.get("format") != "alpha_mask_primitive_archive_variants_v1":
        raise AlphaLossyRepairPlannerError(f"unsupported variants format {variants.get('format')!r}")
    _require_non_promotable(variants, context="primitive_variants")
    _require_false(variants, "official_component_response", context="primitive_variants")
    if variants.get("evidence_grade") != EVIDENCE_GRADE:
        raise AlphaLossyRepairPlannerError("primitive variants evidence_grade must be empirical")

    baseline_archive = plan.get("baseline_archive")
    if not isinstance(baseline_archive, Mapping):
        raise AlphaLossyRepairPlannerError("primitive plan baseline_archive missing")
    baseline_archive_meta: dict[str, Any] | None = None
    if baseline_archive.get("path_hint"):
        baseline_path = _resolve_reference_path(
            str(baseline_archive["path_hint"]),
            manifest_dir=plan_dir,
            field="primitive_plan.baseline_archive.path_hint",
        )
        baseline_archive_meta = _validate_file_record(
            baseline_path,
            {
                "size_bytes": baseline_archive.get("bytes"),
                "sha256": baseline_archive.get("sha256"),
            },
            label="primitive baseline archive",
        )

    points: list[PrimitivePoint] = []
    raw_points = plan.get("points")
    if not isinstance(raw_points, list):
        raise AlphaLossyRepairPlannerError("primitive plan points must be a list")
    for offset, point in enumerate(raw_points):
        if not isinstance(point, Mapping):
            raise AlphaLossyRepairPlannerError(f"primitive plan points[{offset}] must be an object")
        _require_non_promotable(point, context=f"primitive_plan.points[{offset}]")
        if point.get("official_component_response") is not False:
            raise AlphaLossyRepairPlannerError(f"primitive_plan.points[{offset}].official_component_response must be false")
        if str(point.get("role", "")) == "baseline" or int(point.get("index", offset)) == 0:
            continue
        primitive = point.get("primitive")
        mask_delta = point.get("mask_delta")
        mask_member = point.get("mask_member")
        if not isinstance(primitive, Mapping):
            raise AlphaLossyRepairPlannerError(f"primitive plan points[{offset}].primitive missing")
        if not isinstance(mask_delta, Mapping):
            raise AlphaLossyRepairPlannerError(f"primitive plan points[{offset}].mask_delta missing")
        if not isinstance(mask_member, Mapping):
            raise AlphaLossyRepairPlannerError(f"primitive plan points[{offset}].mask_member missing")
        archive_rel = str(point.get("archive", ""))
        archive_path = _resolve_plan_relative_path(
            archive_rel,
            plan_dir=plan_dir,
            field=f"primitive_plan.points[{offset}].archive",
        )
        archive_meta = _validate_file_record(
            archive_path,
            {
                "size_bytes": point.get("archive_bytes"),
                "sha256": point.get("archive_sha256"),
            },
            label=f"primitive point archive {point.get('index', offset)}",
        )
        points.append(
            PrimitivePoint(
                index=int(point.get("index", offset)),
                primitive_id=str(point.get("primitive_id", primitive.get("primitive_id", ""))),
                kind=str(primitive.get("kind", "")),
                operation=str(primitive.get("operation", "")),
                frame_index=_optional_int(primitive.get("frame_index"), field=f"points[{offset}].frame_index"),
                source_class=_optional_int(primitive.get("source_class"), field=f"points[{offset}].source_class"),
                target_class=_optional_int(primitive.get("target_class"), field=f"points[{offset}].target_class"),
                selection_weight=_finite_float(
                    point.get("selection_weight", primitive.get("selection_weight", 0.0)),
                    field=f"points[{offset}].selection_weight",
                ),
                changed_pixels=_nonnegative_int(mask_delta.get("changed_pixels"), field=f"points[{offset}].changed_pixels"),
                selected_pixels_before_cap=_optional_int(
                    mask_delta.get("selected_pixels_before_cap"),
                    field=f"points[{offset}].selected_pixels_before_cap",
                ),
                selected_pixels_after_cap=_optional_int(
                    mask_delta.get("selected_pixels_after_cap"),
                    field=f"points[{offset}].selected_pixels_after_cap",
                ),
                mask_member_size_bytes=_positive_int(mask_member.get("size_bytes"), field=f"points[{offset}].mask_member.size_bytes"),
                archive_bytes=int(archive_meta["size_bytes"]),
                archive_sha256=str(archive_meta["sha256"]),
                archive_path=archive_rel,
            )
        )
        if len(points) >= config.max_primitive_points:
            break

    if not points:
        raise AlphaLossyRepairPlannerError("primitive plan contains no non-baseline points")

    return {
        "plan": plan,
        "variants": variants,
        "plan_file": _file_meta(plan_path),
        "variants_file": variants_meta,
        "baseline_archive": baseline_archive_meta,
        "source": plan.get("source", {}),
        "points": sorted(points, key=lambda item: item.index),
        "primitive_points_available": max(0, len(raw_points) - 1),
        "primitive_points_used": len(points),
    }


def _lossy_base_candidates(
    *,
    primitive: Mapping[str, Any],
    matrix: Mapping[str, Any],
    config: BudgetConfig,
) -> list[LossyBaseCandidate]:
    points: list[PrimitivePoint] = list(primitive["points"])
    by_size = sorted(points, key=lambda point: (point.mask_member_size_bytes, point.index))
    source_mask_bytes = int(matrix["source_mask_member_size_bytes"])
    bases: list[LossyBaseCandidate] = []

    min_point = by_size[0]
    bases.append(
        LossyBaseCandidate(
            base_id="primitive_plan_min_crf_mask_member",
            source="primitive_plan_point_mask_member_size",
            estimated_mask_member_bytes=int(min_point.mask_member_size_bytes),
            details={
                "representative_point_index": int(min_point.index),
                "representative_primitive_id": min_point.primitive_id,
                "assumption": (
                    "smallest primitive-plan masks.mkv point approximates a lossy geometry base "
                    "at the same CRF/fps; it is not score evidence"
                ),
            },
        )
    )

    median_point = by_size[len(by_size) // 2]
    if median_point.index != min_point.index:
        bases.append(
            LossyBaseCandidate(
                base_id="primitive_plan_median_crf_mask_member",
                source="primitive_plan_point_mask_member_size",
                estimated_mask_member_bytes=int(median_point.mask_member_size_bytes),
                details={
                    "representative_point_index": int(median_point.index),
                    "representative_primitive_id": median_point.primitive_id,
                    "assumption": (
                        "median primitive-plan masks.mkv point approximates a less optimistic lossy base "
                        "at the same CRF/fps; it is not score evidence"
                    ),
                },
            )
        )

    for index, bytes_value in enumerate(config.lossy_base_bytes):
        bases.append(
            LossyBaseCandidate(
                base_id=f"operator_lossy_base_estimate_{index:02d}",
                source="operator_cli_lossy_base_bytes",
                estimated_mask_member_bytes=int(bytes_value),
                details={
                    "operator_supplied_bytes": int(bytes_value),
                    "assumption": "explicit operator-provided lossy-base byte hypothesis; not measured score evidence",
                },
            )
        )

    bases.append(
        LossyBaseCandidate(
            base_id="current_mask_member_reference",
            source="matrix_source_mask_member",
            estimated_mask_member_bytes=source_mask_bytes,
            details={
                "assumption": "current masks.mkv member reference; included for budget comparison, not a lossy base",
            },
        )
    )

    deduped: list[LossyBaseCandidate] = []
    seen: set[tuple[str, int]] = set()
    for base in bases:
        key = (base.base_id, base.estimated_mask_member_bytes)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(base)
    return deduped


def _bounded_counts(total: int, *, max_count: int) -> list[int]:
    total = max(0, int(total))
    max_count = max(0, int(max_count))
    if total == 0 or max_count == 0:
        return [0]
    counts = {0, 1, total, min(total, max_count)}
    step = 2
    while step <= total and step <= max_count:
        counts.add(step)
        step *= 2
    return sorted(count for count in counts if count <= total and count <= max_count)


def _protected(point: PrimitivePoint) -> bool:
    return point.source_class in PROTECTED_CLASS_IDS or point.target_class in PROTECTED_CLASS_IDS


def _policy_orderings(points: list[PrimitivePoint]) -> list[tuple[str, str, list[PrimitivePoint], dict[str, Any]]]:
    return [
        (
            "response_pending_proxy_selection_weight",
            "component_response_top_k_pending_proxy",
            sorted(points, key=lambda point: (-point.selection_weight, point.index)),
            {
                "final_selection_source": "official_cuda_component_response_pending",
                "proxy_ordering": "primitive selection_weight descending",
            },
        ),
        (
            "protected_class_first_proxy",
            "protected_class_proxy_top_k",
            sorted(points, key=lambda point: (not _protected(point), -point.selection_weight, point.index)),
            {
                "protected_class_ids": list(PROTECTED_CLASS_IDS),
                "proxy_ordering": "protected class involvement, then selection_weight descending",
            },
        ),
        (
            "changed_pixels_desc_proxy",
            "large_sparse_patch_proxy_top_k",
            sorted(points, key=lambda point: (-point.changed_pixels, point.index)),
            {
                "proxy_ordering": "changed_pixels descending",
            },
        ),
    ]


def _repair_policies(points: list[PrimitivePoint], *, config: BudgetConfig) -> list[RepairPolicy]:
    counts = _bounded_counts(len(points), max_count=config.max_policy_points)
    policies: list[RepairPolicy] = [
        RepairPolicy(
            policy_id="no_sparse_repair",
            policy_kind="no_repair_reference",
            selected_points=(),
            details={"description": "lossy base only; exact CUDA eval still required before any score use"},
        )
    ]
    for ordering_id, kind, ordered_points, details in _policy_orderings(points):
        for count in counts:
            if count == 0:
                continue
            selected = tuple(ordered_points[:count])
            policies.append(
                RepairPolicy(
                    policy_id=f"{ordering_id}_k{count:02d}",
                    policy_kind=kind,
                    selected_points=selected,
                    details={
                        **details,
                        "selected_count_for_estimate": int(count),
                        "component_response_required_before_final_selection": True,
                        "proxy_ordering_only": True,
                    },
                )
            )

    deduped: list[RepairPolicy] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for policy in policies:
        key = (policy.policy_kind, tuple(point.index for point in policy.selected_points))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(policy)
    return deduped


def _estimate_repair_payload(policy: RepairPolicy, *, config: BudgetConfig) -> dict[str, Any]:
    point_records: list[dict[str, Any]] = []
    raw_total = 0
    run_total = 0
    changed_total = 0
    if policy.selected_points:
        raw_total += int(config.repair_global_header_bytes)
    for point in policy.selected_points:
        runs = int(math.ceil(point.changed_pixels / config.repair_pixels_per_run)) if point.changed_pixels else 0
        pixel_bits_bytes = int(math.ceil(point.changed_pixels * config.repair_bits_per_pixel / 8.0))
        raw_bytes = (
            int(config.repair_point_header_bytes)
            + int(config.repair_primitive_ref_bytes)
            + runs * int(config.repair_run_record_bytes)
            + pixel_bits_bytes
        )
        raw_total += raw_bytes
        run_total += runs
        changed_total += int(point.changed_pixels)
        point_records.append(
            {
                "index": int(point.index),
                "primitive_id": point.primitive_id,
                "kind": point.kind,
                "operation": point.operation,
                "frame_index": point.frame_index,
                "source_class": point.source_class,
                "target_class": point.target_class,
                "selection_weight": _round_float(point.selection_weight),
                "changed_pixels": int(point.changed_pixels),
                "estimated_runs": int(runs),
                "estimated_raw_bytes": int(raw_bytes),
            }
        )

    encoded = int(math.ceil(raw_total * config.repair_compression_ratio))
    if not policy.selected_points:
        encoded = 0
    return {
        "selected_point_count": int(len(policy.selected_points)),
        "selected_changed_pixels": int(changed_total),
        "estimated_runs": int(run_total),
        "estimated_raw_bytes": int(raw_total),
        "estimated_encoded_bytes": int(encoded),
        "point_estimates": point_records,
    }


def _candidate_budget_records(
    *,
    bases: list[LossyBaseCandidate],
    policies: list[RepairPolicy],
    matrix: Mapping[str, Any],
    target_mask_member_bytes: int,
    config: BudgetConfig,
) -> list[dict[str, Any]]:
    source_mask_bytes = int(matrix["source_mask_member_size_bytes"])
    best_exact = matrix.get("best_exact_candidate")
    best_exact_bytes = int(best_exact["size_bytes"]) if isinstance(best_exact, Mapping) else None
    records: list[dict[str, Any]] = []
    for base in bases:
        for policy in policies:
            repair_estimate = _estimate_repair_payload(policy, config=config)
            total_bytes = int(base.estimated_mask_member_bytes) + int(repair_estimate["estimated_encoded_bytes"])
            remaining_vs_source = source_mask_bytes - int(base.estimated_mask_member_bytes)
            record = {
                "spec_id": f"{base.base_id}__{policy.policy_id}",
                "score_claim": False,
                "promotion_eligible": False,
                "evidence_grade": EVIDENCE_GRADE,
                "candidate_archive_built": False,
                "requires_exact_cuda_auth_eval": True,
                "requires_component_response_before_build": bool(policy.selected_points),
                "lossy_base": {
                    "base_id": base.base_id,
                    "source": base.source,
                    "estimated_mask_member_bytes": int(base.estimated_mask_member_bytes),
                    "details": dict(base.details),
                },
                "repair_policy": {
                    "policy_id": policy.policy_id,
                    "policy_kind": policy.policy_kind,
                    "selected_point_indices_for_proxy_estimate": [
                        int(point.index) for point in policy.selected_points
                    ],
                    "details": dict(policy.details),
                },
                "repair_estimate": repair_estimate,
                "estimated_mask_stream_bytes": int(total_bytes),
                "target_mask_member_bytes": int(target_mask_member_bytes),
                "target_mask_member_pass": bool(total_bytes <= target_mask_member_bytes),
                "current_mask_member_size_bytes": int(source_mask_bytes),
                "delta_vs_current_mask_member_bytes": int(total_bytes - source_mask_bytes),
                "under_current_mask_member": bool(total_bytes <= source_mask_bytes),
                "repair_budget_remaining_vs_current_mask_member_bytes": int(remaining_vs_source),
                "repair_bytes_fit_remaining_current_mask_budget": bool(
                    int(repair_estimate["estimated_encoded_bytes"]) <= remaining_vs_source
                ),
            }
            if best_exact_bytes is not None:
                record["best_exact_matrix_candidate_size_bytes"] = int(best_exact_bytes)
                record["delta_vs_best_exact_matrix_candidate_bytes"] = int(total_bytes - best_exact_bytes)
            records.append(record)
    return sorted(
        records,
        key=lambda item: (
            int(item["estimated_mask_stream_bytes"]),
            -int(item["repair_estimate"]["selected_changed_pixels"]),
            str(item["spec_id"]),
        ),
    )


def _assumptions(config: BudgetConfig) -> dict[str, Any]:
    return {
        "score_claim": False,
        "promotion_eligible": False,
        "component_response_status": "pending",
        "component_response_use": (
            "repair point ordering is a proxy until official CUDA component-response "
            "returns; selected point ids in this report are byte-model estimates only"
        ),
        "lossy_base_bytes_source": (
            "primitive plan point masks.mkv sizes at the plan CRF/fps plus optional "
            "operator-provided byte hypotheses"
        ),
        "repair_payload_model": {
            "global_header_bytes": int(config.repair_global_header_bytes),
            "point_header_bytes": int(config.repair_point_header_bytes),
            "primitive_ref_bytes": int(config.repair_primitive_ref_bytes),
            "run_record_bytes": int(config.repair_run_record_bytes),
            "pixels_per_run": int(config.repair_pixels_per_run),
            "bits_per_pixel": float(config.repair_bits_per_pixel),
            "compression_ratio": float(config.repair_compression_ratio),
        },
        "non_claim_boundary": (
            "These estimates do not imply scorer distortion, score deltas, method ranking, "
            "promotion, or retirement. They only size charged payload hypotheses."
        ),
    }


def _config_record(config: BudgetConfig) -> dict[str, Any]:
    return json.loads(json.dumps(dataclasses.asdict(config), allow_nan=False))


def _provenance(command: list[str] | None) -> dict[str, Any]:
    return {
        "tool": PRODUCER,
        "command": list(command) if command is not None else list(sys.argv),
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "selected_environment": {
            key: os.environ[key]
            for key in (
                "PYTHONHASHSEED",
                "UV_PROJECT_ENVIRONMENT",
                "TAC_FFMPEG",
                "TAC_FFPROBE",
                "TAC_UPSTREAM_DIR",
            )
            if key in os.environ
        },
        "source_files": [
            _source_file_meta(REPO_ROOT / PRODUCER),
            _source_file_meta(REPO_ROOT / "experiments" / "alpha_mask_codec_candidate_matrix.py"),
            _source_file_meta(REPO_ROOT / "experiments" / "build_alpha_mask_primitive_response_plan.py"),
        ],
    }


def _spec_for_record(
    *,
    record: Mapping[str, Any],
    report_path: Path,
    matrix: Mapping[str, Any],
    primitive: Mapping[str, Any],
    config: BudgetConfig,
) -> dict[str, Any]:
    return {
        "schema": SPEC_SCHEMA,
        "producer": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "candidate_archive_built": False,
        "archive_path": None,
        "requires_exact_cuda_auth_eval": True,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "parent_report": {
            "path": _path_hint(report_path),
            "note": "parent report is written after spec generation and records this spec's SHA-256",
        },
        "source_hashes": {
            "matrix_manifest": matrix["manifest_file"],
            "primitive_plan": primitive["plan_file"],
            "primitive_variants_manifest": primitive["variants_file"],
        },
        "component_response_handoff": {
            "status": "pending",
            "required_before_build": bool(record["requires_component_response_before_build"]),
            "allowed_sources": [
                "official CUDA component-response curves with archive.zip -> inflate.sh -> upstream/evaluate.py custody"
            ],
            "selection_rule_after_response": (
                "replace proxy selected point ids with response-safe sparse repair atoms "
                "that satisfy PoseNet/SegNet gates under the recorded byte budget"
            ),
        },
        "estimated_payload": {
            "lossy_base": record["lossy_base"],
            "repair_policy": record["repair_policy"],
            "repair_estimate": record["repair_estimate"],
            "estimated_mask_stream_bytes": int(record["estimated_mask_stream_bytes"]),
            "current_mask_member_size_bytes": int(record["current_mask_member_size_bytes"]),
            "delta_vs_current_mask_member_bytes": int(record["delta_vs_current_mask_member_bytes"]),
            "target_mask_member_bytes": int(record["target_mask_member_bytes"]),
            "target_mask_member_pass": bool(record["target_mask_member_pass"]),
        },
        "build_recipe_after_component_response": [
            "choose final sparse repair atoms from official CUDA component-response evidence",
            "encode a charged lossy geometry base plus sparse repair payload inside archive.zip",
            "record deterministic archive manifest, payload hashes, decoder contract, and source manifest",
            "run exact CUDA auth eval before any score, rank, promotion, or retirement claim",
        ],
        "prohibited_uses": [
            "score_claim",
            "promotion",
            "ranking",
            "method_retirement",
            "paper empirical claim without later exact CUDA archive evidence",
        ],
        "config": _config_record(config),
    }


def _write_candidate_specs(
    *,
    output_dir: Path,
    report_path: Path,
    records: list[dict[str, Any]],
    matrix: Mapping[str, Any],
    primitive: Mapping[str, Any],
    config: BudgetConfig,
) -> list[dict[str, Any]]:
    specs_dir = output_dir / SPECS_DIR_NAME
    specs_dir.mkdir(parents=True, exist_ok=True)
    selected = records[: config.max_specs]
    specs: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        spec_path = specs_dir / f"{index:03d}_{record['spec_id']}.json"
        spec = _spec_for_record(
            record=record,
            report_path=report_path,
            matrix=matrix,
            primitive=primitive,
            config=config,
        )
        _assert_non_promotable(spec, context=f"spec {spec_path}")
        _write_json(spec_path, spec)
        specs.append(
            {
                "spec_id": record["spec_id"],
                "path": _path_hint(spec_path),
                "size_bytes": int(spec_path.stat().st_size),
                "sha256": _sha256_file(spec_path),
                "estimated_mask_stream_bytes": int(record["estimated_mask_stream_bytes"]),
                "requires_component_response_before_build": bool(
                    record["requires_component_response_before_build"]
                ),
                "candidate_archive_built": False,
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
    return specs


def _assert_non_promotable(payload: Mapping[str, Any], *, context: str) -> None:
    if payload.get("score_claim") is not False:
        raise AssertionError(f"{context} score_claim must be false")
    if payload.get("promotion_eligible") is not False:
        raise AssertionError(f"{context} promotion_eligible must be false")


def _assert_report(report: Mapping[str, Any]) -> None:
    _assert_non_promotable(report, context="report")
    if report.get("scorer_network_loaded") is not False:
        raise AssertionError("report scorer_network_loaded must be false")
    if report.get("evidence_grade") != EVIDENCE_GRADE:
        raise AssertionError("report evidence_grade must be empirical")
    for record in report.get("budget_records", []):
        _assert_non_promotable(record, context=f"budget record {record.get('spec_id')}")
        if record.get("candidate_archive_built") is not False:
            raise AssertionError(f"budget record {record.get('spec_id')} must not build an archive")
    for spec in report.get("candidate_next_archive_specs", []):
        _assert_non_promotable(spec, context=f"spec record {spec.get('spec_id')}")
        if spec.get("candidate_archive_built") is not False:
            raise AssertionError(f"spec record {spec.get('spec_id')} must not build an archive")


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_NAME
    if report_path.exists() and not force:
        raise FileExistsError(f"{report_path} already exists; use --force to overwrite")


def plan_lossy_repair_budgets(
    *,
    primitive_plan: Path,
    matrix_manifest: Path,
    output_dir: Path,
    config: BudgetConfig,
    command: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a deterministic non-promotable Alpha lossy repair budget report."""

    _validate_config(config)
    _prepare_output_dir(output_dir, force=force)
    matrix = _load_matrix_context(matrix_manifest)
    primitive = _load_primitive_context(primitive_plan, config=config)
    target_mask_member_bytes = int(config.target_mask_member_bytes or matrix["source_mask_member_size_bytes"])

    bases = _lossy_base_candidates(primitive=primitive, matrix=matrix, config=config)
    policies = _repair_policies(list(primitive["points"]), config=config)
    budget_records = _candidate_budget_records(
        bases=bases,
        policies=policies,
        matrix=matrix,
        target_mask_member_bytes=target_mask_member_bytes,
        config=config,
    )

    report_path = output_dir / REPORT_NAME
    candidate_specs = _write_candidate_specs(
        output_dir=output_dir,
        report_path=report_path,
        records=budget_records,
        matrix=matrix,
        primitive=primitive,
        config=config,
    )

    best_under_current = [
        record
        for record in budget_records
        if record["under_current_mask_member"] is True and record["target_mask_member_pass"] is True
    ][: config.max_specs]
    exact_candidates = [
        candidate
        for candidate in matrix["candidates"]
        if candidate["exact_reconstruction"] is True and candidate["diagnostic_reference"] is False
    ]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "local_empirical_planner_only": True,
        "scorer_network_loaded": False,
        "remote_jobs_launched": False,
        "archives_built": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "purpose": (
            "size Alpha lossy-geometry plus sparse-repair candidates under explicit byte-model "
            "assumptions, then emit archive-build specs to revisit after CUDA component-response"
        ),
        "assumptions": _assumptions(config),
        "config": _config_record(config),
        "custody": {
            "matrix_manifest": matrix["manifest_file"],
            "primitive_plan": primitive["plan_file"],
            "primitive_variants_manifest": primitive["variants_file"],
            "matrix_source_archive": matrix["source_archive"],
            "primitive_baseline_archive": primitive["baseline_archive"],
            "source_decoded_mask_sha256": {
                "matrix": matrix.get("source_decoded_mask_sha256"),
                "primitive": (
                    primitive.get("source", {}).get("decoded_masks", {}).get("class_id_u8_sha256")
                    if isinstance(primitive.get("source"), Mapping)
                    and isinstance(primitive.get("source", {}).get("decoded_masks"), Mapping)
                    else None
                ),
            },
        },
        "matrix_context": {
            "source_mask_member_size_bytes": int(matrix["source_mask_member_size_bytes"]),
            "source_mask_member": matrix["source_mask_member"],
            "exact_candidate_count": len(exact_candidates),
            "best_exact_candidate": matrix["best_exact_candidate"],
            "exact_candidates_by_bytes": sorted(
                exact_candidates,
                key=lambda item: (int(item["size_bytes"]), str(item["family"])),
            ),
        },
        "primitive_context": {
            "primitive_points_available": int(primitive["primitive_points_available"]),
            "primitive_points_used": int(primitive["primitive_points_used"]),
            "point_summary": [
                {
                    "index": point.index,
                    "primitive_id": point.primitive_id,
                    "kind": point.kind,
                    "operation": point.operation,
                    "frame_index": point.frame_index,
                    "source_class": point.source_class,
                    "target_class": point.target_class,
                    "selection_weight": _round_float(point.selection_weight),
                    "changed_pixels": point.changed_pixels,
                    "mask_member_size_bytes": point.mask_member_size_bytes,
                    "archive_bytes": point.archive_bytes,
                    "archive_sha256": point.archive_sha256,
                }
                for point in primitive["points"]
            ],
        },
        "lossy_base_candidates": [
            {
                "base_id": base.base_id,
                "source": base.source,
                "estimated_mask_member_bytes": int(base.estimated_mask_member_bytes),
                "details": dict(base.details),
            }
            for base in bases
        ],
        "planner_summary": {
            "budget_record_count": int(len(budget_records)),
            "candidate_spec_count": int(len(candidate_specs)),
            "target_mask_member_bytes": int(target_mask_member_bytes),
            "current_mask_member_size_bytes": int(matrix["source_mask_member_size_bytes"]),
            "best_under_current_mask_member": best_under_current,
            "next_step": (
                "wait for official CUDA component-response, then instantiate only specs whose "
                "selected repair atoms pass component gates and rerun exact CUDA auth eval"
            ),
        },
        "candidate_next_archive_specs": candidate_specs,
        "budget_records": budget_records,
        "provenance": _provenance(command),
    }
    _assert_report(report)
    _write_json(report_path, report)
    return report


def _parse_lossy_base_bytes(value: str | None) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    out: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid lossy base byte value {raw!r}") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError("lossy base bytes must be positive")
        out.append(parsed)
    return tuple(out)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitive-plan", type=Path, default=DEFAULT_PRIMITIVE_PLAN)
    parser.add_argument("--matrix-manifest", type=Path, default=DEFAULT_MATRIX_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-primitive-points", type=int, default=BudgetConfig.max_primitive_points)
    parser.add_argument("--max-specs", type=int, default=BudgetConfig.max_specs)
    parser.add_argument("--max-policy-points", type=int, default=BudgetConfig.max_policy_points)
    parser.add_argument("--target-mask-member-bytes", type=int, default=None)
    parser.add_argument("--repair-global-header-bytes", type=int, default=BudgetConfig.repair_global_header_bytes)
    parser.add_argument("--repair-point-header-bytes", type=int, default=BudgetConfig.repair_point_header_bytes)
    parser.add_argument("--repair-primitive-ref-bytes", type=int, default=BudgetConfig.repair_primitive_ref_bytes)
    parser.add_argument("--repair-run-record-bytes", type=int, default=BudgetConfig.repair_run_record_bytes)
    parser.add_argument("--repair-pixels-per-run", type=int, default=BudgetConfig.repair_pixels_per_run)
    parser.add_argument("--repair-bits-per-pixel", type=float, default=BudgetConfig.repair_bits_per_pixel)
    parser.add_argument("--repair-compression-ratio", type=float, default=BudgetConfig.repair_compression_ratio)
    parser.add_argument(
        "--lossy-base-bytes",
        type=_parse_lossy_base_bytes,
        default=(),
        help="Optional comma-separated lossy-base byte hypotheses to include as planning assumptions.",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing planner report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = BudgetConfig(
        max_primitive_points=args.max_primitive_points,
        max_specs=args.max_specs,
        max_policy_points=args.max_policy_points,
        target_mask_member_bytes=args.target_mask_member_bytes,
        repair_global_header_bytes=args.repair_global_header_bytes,
        repair_point_header_bytes=args.repair_point_header_bytes,
        repair_primitive_ref_bytes=args.repair_primitive_ref_bytes,
        repair_run_record_bytes=args.repair_run_record_bytes,
        repair_pixels_per_run=args.repair_pixels_per_run,
        repair_bits_per_pixel=args.repair_bits_per_pixel,
        repair_compression_ratio=args.repair_compression_ratio,
        lossy_base_bytes=args.lossy_base_bytes,
    )
    report = plan_lossy_repair_budgets(
        primitive_plan=args.primitive_plan,
        matrix_manifest=args.matrix_manifest,
        output_dir=args.output_dir,
        config=config,
        command=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
        force=args.force,
    )
    print(
        f"[empirical:{args.output_dir / REPORT_NAME}] Alpha lossy repair planner wrote "
        f"{report['planner_summary']['candidate_spec_count']} candidate specs from "
        f"{report['planner_summary']['budget_record_count']} budget records. No score claim; "
        "component-response plus exact CUDA auth eval are required before build/use.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
