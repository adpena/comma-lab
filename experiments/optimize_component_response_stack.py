#!/usr/bin/env python3
"""Optimize offline stack candidates from official component-response evidence.

This is a deterministic planning tool. It does not run scorers, build
archives, queue remote work, or claim that response deltas compose. Its output
is always non-promotable until a selected stack is built as its own archive and
evaluated through the canonical CUDA auth-eval path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORMAT = "component_response_stack_optimizer_v1"
PRODUCER = "experiments/optimize_component_response_stack.py"
SUMMARY_FORMAT = "official_component_response_summary_v1"
CURVE_FORMAT = "official_component_response_curves_v1"
PLAN_FORMAT = "official_component_response_plan_v1"
PREDICTION_DELTAS_FORMAT = "official_component_response_prediction_deltas_v1"
CANONICAL_COMPONENT_RESPONSE_PATH = "archive_zip_inflate_sh_upstream_evaluate_py"
CANONICAL_RESPONSE_EVAL_PATH = "archive.zip -> inflate.sh -> upstream/evaluate.py"
ORIGINAL_VIDEO_BYTES = 37_545_489
CONTEST_SAMPLE_COUNT = 600
COMPONENTS = ("posenet", "segnet")
CURVE_COMPONENTS = ("posenet", "segnet", "combined")
SCORE_EPS = 1e-12
INVALID_PROJECTED_SCORE = 1.0e300
RATE_SLOPE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES


class ComponentResponseStackOptimizerError(ValueError):
    """Raised when response evidence cannot support offline stack planning."""


@dataclass(frozen=True)
class BaselineState:
    archive: Mapping[str, Any]
    values: Mapping[str, float]
    score: Mapping[str, float]


@dataclass(frozen=True)
class ResponseAction:
    action_id: str
    source_id: str
    source_path: str
    evidence_mode: str
    epsilon: float
    archive: Mapping[str, Any]
    contest_auth_eval_json: Mapping[str, Any] | None
    values: Mapping[str, float]
    deltas: Mapping[str, float | int]
    score_terms: Mapping[str, float]
    score_deltas: Mapping[str, float]
    point_metadata: Mapping[str, Any]
    custody: Mapping[str, Any]


@dataclass(frozen=True)
class ResponseSource:
    source_id: str
    input_path: Path
    input_format: str
    evidence_mode: str
    promotion_eligible_input: bool
    baseline: BaselineState
    actions: tuple[ResponseAction, ...]
    source_files: tuple[Mapping[str, Any], ...]
    validation: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ComponentResponseStackOptimizerError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ComponentResponseStackOptimizerError(f"{label} must be a JSON object: {path}")
    return payload


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _maybe_file_meta(path: Path) -> dict[str, Any]:
    meta = {"path": str(path.resolve()), "exists": bool(path.is_file())}
    if path.is_file():
        meta.update({"bytes": int(path.stat().st_size), "sha256": _sha256_file(path)})
    return meta


def _finite_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComponentResponseStackOptimizerError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ComponentResponseStackOptimizerError(f"{field} must be finite")
    if minimum is not None and out < minimum:
        raise ComponentResponseStackOptimizerError(f"{field} must be >= {minimum}")
    return out


def _require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComponentResponseStackOptimizerError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ComponentResponseStackOptimizerError(f"{field} must be >= {minimum}")
    return int(value)


def _epsilon_key(value: Any) -> str:
    return f"{_finite_float(value, field='epsilon'):.17g}"


def _epsilon_label(value: float) -> str:
    text = f"{float(value):.12g}"
    text = text.replace("-", "m").replace("+", "p").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return slug or "input"


def _combined_no_rate(*, pose: float, seg: float) -> float:
    return 100.0 * seg + math.sqrt(10.0 * pose)


def _score_breakdown(*, pose: float, seg: float, archive_bytes: int) -> dict[str, float]:
    pose = _finite_float(pose, field="posenet_dist", minimum=0.0)
    seg = _finite_float(seg, field="segnet_dist", minimum=0.0)
    archive_bytes = _require_int(archive_bytes, field="archive_bytes", minimum=0)
    score_seg = 100.0 * seg
    score_pose = math.sqrt(10.0 * pose)
    score_rate = RATE_SLOPE_SCORE_PER_BYTE * archive_bytes
    total = score_seg + score_pose + score_rate
    return {
        "archive_bytes": int(archive_bytes),
        "combined_no_rate": score_seg + score_pose,
        "posenet_dist": pose,
        "posenet_score_term": score_pose,
        "rate_score_term": score_rate,
        "score": total,
        "segnet_dist": seg,
        "segnet_score_term": score_seg,
    }


def _score_delta(
    baseline: Mapping[str, float],
    projected: Mapping[str, float],
) -> dict[str, float]:
    return {
        "archive_bytes_delta": float(projected["archive_bytes"] - baseline["archive_bytes"]),
        "combined_no_rate_delta": projected["combined_no_rate"] - baseline["combined_no_rate"],
        "posenet_dist_delta": projected["posenet_dist"] - baseline["posenet_dist"],
        "posenet_score_delta": projected["posenet_score_term"] - baseline["posenet_score_term"],
        "rate_score_delta": projected["rate_score_term"] - baseline["rate_score_term"],
        "score_delta": projected["score"] - baseline["score"],
        "segnet_dist_delta": projected["segnet_dist"] - baseline["segnet_dist"],
        "segnet_score_delta": projected["segnet_score_term"] - baseline["segnet_score_term"],
    }


def _values_from_pose_seg(*, pose: float, seg: float) -> dict[str, float]:
    pose = _finite_float(pose, field="posenet", minimum=0.0)
    seg = _finite_float(seg, field="segnet", minimum=0.0)
    return {
        "combined": _combined_no_rate(pose=pose, seg=seg),
        "posenet": pose,
        "segnet": seg,
    }


def _values_from_mapping(payload: Mapping[str, Any], *, field: str) -> dict[str, float]:
    return _values_from_pose_seg(
        pose=_finite_float(payload.get("posenet"), field=f"{field}.posenet", minimum=0.0),
        seg=_finite_float(payload.get("segnet"), field=f"{field}.segnet", minimum=0.0),
    )


def _archive_meta_from_mapping(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    archive = payload.get("archive", payload)
    if isinstance(archive, str):
        return {"path": archive}
    if not isinstance(archive, Mapping):
        raise ComponentResponseStackOptimizerError(f"{field}.archive must be an object or path")
    out: dict[str, Any] = {}
    if "path" in archive:
        out["path"] = str(archive["path"])
    if "bytes" in archive:
        out["bytes"] = _require_int(archive["bytes"], field=f"{field}.archive.bytes", minimum=0)
    elif "archive_bytes" in payload:
        out["bytes"] = _require_int(
            payload["archive_bytes"],
            field=f"{field}.archive_bytes",
            minimum=0,
        )
    if "sha256" in archive:
        out["sha256"] = str(archive["sha256"])
    elif "archive_sha256" in payload:
        out["sha256"] = str(payload["archive_sha256"])
    return out


def _contest_eval_meta_from_mapping(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = payload.get("contest_auth_eval_json")
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"path": raw}
    if not isinstance(raw, Mapping):
        raise ComponentResponseStackOptimizerError("contest_auth_eval_json must be an object or path")
    out = dict(raw)
    if "path" in out:
        out["path"] = str(out["path"])
    return out


def _resolve_artifact_path(value: Any, *, root: Path, default_name: str | None = None) -> Path:
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or not value:
        if default_name is None:
            raise ComponentResponseStackOptimizerError(f"invalid artifact path: {value!r}")
        value = default_name
    path = Path(value)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        candidates.append(root / path.name)
    else:
        candidates.append(root / path)
        candidates.append(root / path.name)
    if default_name is not None:
        candidates.append(root / default_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _promotion_blocker_codes(payload: Mapping[str, Any]) -> list[str]:
    blockers = payload.get("promotion_blockers")
    if not isinstance(blockers, list):
        return []
    out: list[str] = []
    for item in blockers:
        if isinstance(item, Mapping) and isinstance(item.get("code"), str):
            out.append(str(item["code"]))
        elif isinstance(item, str):
            out.append(item)
    return out


def _validate_curve_payload(
    curve: Mapping[str, Any],
    *,
    path: Path,
    component: str | None,
    allow_calibration_inputs: bool,
) -> dict[str, Any]:
    if curve.get("format") != CURVE_FORMAT:
        raise ComponentResponseStackOptimizerError(
            f"{path}: expected format {CURVE_FORMAT!r}"
        )
    observed_component = curve.get("component")
    if component is not None and observed_component != component:
        raise ComponentResponseStackOptimizerError(
            f"{path}: component={observed_component!r}, expected {component!r}"
        )
    device = curve.get("device")
    if device != "cuda":
        raise ComponentResponseStackOptimizerError(
            f"{path}: non-CUDA component response rejected: device={device!r}"
        )
    if curve.get("official_component_response") is not True:
        raise ComponentResponseStackOptimizerError(
            f"{path}: official_component_response must be true"
        )
    if curve.get("canonical_scorer_path") is not True:
        raise ComponentResponseStackOptimizerError(
            f"{path}: canonical_scorer_path must be true"
        )
    if curve.get("component_response_path") != CANONICAL_COMPONENT_RESPONSE_PATH:
        raise ComponentResponseStackOptimizerError(
            f"{path}: component_response_path must be {CANONICAL_COMPONENT_RESPONSE_PATH!r}"
        )
    gates = curve.get("gate_results")
    if not isinstance(gates, Mapping):
        raise ComponentResponseStackOptimizerError(f"{path}: gate_results is required")
    for gate in ("finite_values", "coverage_passed", "zero_repro"):
        if gates.get(gate) is not True:
            raise ComponentResponseStackOptimizerError(
                f"{path}: gate_results.{gate} must be true for offline planning"
            )
    promotion_eligible = curve.get("promotion_eligible") is True and curve.get("passed") is True
    if not allow_calibration_inputs:
        required_true = {
            "promotion_eligible": curve.get("promotion_eligible"),
            "passed": curve.get("passed"),
            "gate_results.promotion_gate_passed": gates.get("promotion_gate_passed"),
            "gate_results.prediction_error_passed": gates.get("prediction_error_passed"),
            "gate_results.signal_present": gates.get("signal_present"),
        }
        false_fields = [field for field, value in required_true.items() if value is not True]
        blockers = _promotion_blocker_codes(curve)
        if false_fields or blockers:
            raise ComponentResponseStackOptimizerError(
                f"{path}: promotable recommendations require passed official CUDA response "
                f"curves; failed fields={false_fields}, blockers={blockers}"
            )
    return {
        "component": observed_component,
        "evidence_mode": (
            "promotable_cuda_official_component_response"
            if promotion_eligible
            else "calibration_cuda_official_component_response"
        ),
        "gate_results": dict(gates),
        "path": str(path.resolve()),
        "promotion_blocker_codes": _promotion_blocker_codes(curve),
        "promotion_eligible": bool(promotion_eligible),
    }


def _load_curve_file(
    path: Path,
    *,
    component: str | None,
    allow_calibration_inputs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json_object(path, label="official component-response curve")
    validation = _validate_curve_payload(
        payload,
        path=path,
        component=component,
        allow_calibration_inputs=allow_calibration_inputs,
    )
    return payload, validation


def _points_by_epsilon(points: Any, *, field: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(points, list) or not points:
        raise ComponentResponseStackOptimizerError(f"{field} must be a non-empty list")
    out: dict[str, Mapping[str, Any]] = {}
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise ComponentResponseStackOptimizerError(f"{field}[{index}] must be an object")
        key = _epsilon_key(point.get("epsilon"))
        if key in out:
            raise ComponentResponseStackOptimizerError(f"{field} has duplicate epsilon {key}")
        out[key] = point
    return out


def _zero_point(points: Iterable[Mapping[str, Any]], *, field: str) -> Mapping[str, Any] | None:
    matches = [
        point
        for point in points
        if abs(_finite_float(point.get("epsilon"), field=f"{field}.epsilon")) <= SCORE_EPS
    ]
    if len(matches) > 1:
        raise ComponentResponseStackOptimizerError(f"{field} has multiple eps=0 points")
    return matches[0] if matches else None


def _validate_archive_identity_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for key in ("bytes", "sha256"):
        if key in left and key in right and left.get(key) != right.get(key):
            raise ComponentResponseStackOptimizerError(
                f"{context}: archive {key} mismatch: {left.get(key)!r} != {right.get(key)!r}"
            )


def _source_file_records(files: Iterable[Path]) -> tuple[Mapping[str, Any], ...]:
    records = [_file_meta(path) for path in sorted({path.resolve() for path in files})]
    return tuple(records)


def _make_action(
    *,
    source_id: str,
    source_path: Path,
    evidence_mode: str,
    epsilon: float,
    archive: Mapping[str, Any],
    contest_auth_eval_json: Mapping[str, Any] | None,
    values: Mapping[str, float],
    baseline: BaselineState,
    point_metadata: Mapping[str, Any],
    custody_extra: Mapping[str, Any] | None = None,
) -> ResponseAction:
    archive_bytes = _require_int(archive.get("bytes"), field="point.archive.bytes", minimum=0)
    score_terms = _score_breakdown(
        pose=float(values["posenet"]),
        seg=float(values["segnet"]),
        archive_bytes=archive_bytes,
    )
    deltas: dict[str, float | int] = {
        "archive_bytes_delta": int(archive_bytes - int(baseline.archive["bytes"])),
        "combined_delta": float(values["combined"] - baseline.values["combined"]),
        "posenet_dist_delta": float(values["posenet"] - baseline.values["posenet"]),
        "segnet_dist_delta": float(values["segnet"] - baseline.values["segnet"]),
    }
    score_deltas = _score_delta(baseline.score, score_terms)
    archive_sha = str(archive.get("sha256", "no_sha"))
    action_id = (
        f"{source_id}:eps_{_epsilon_label(epsilon)}:"
        f"{archive_sha[:12]}:{_canonical_hash([epsilon, archive, values])[:12]}"
    )
    custody = {
        "archive_bytes_recorded": "bytes" in archive,
        "archive_sha256_recorded": "sha256" in archive,
        "contest_auth_eval_json_recorded": contest_auth_eval_json is not None,
        "source_path": str(source_path.resolve()),
    }
    if custody_extra:
        custody.update(dict(custody_extra))
    return ResponseAction(
        action_id=action_id,
        source_id=source_id,
        source_path=str(source_path.resolve()),
        evidence_mode=evidence_mode,
        epsilon=float(epsilon),
        archive=dict(archive),
        contest_auth_eval_json=dict(contest_auth_eval_json) if contest_auth_eval_json else None,
        values=dict(values),
        deltas=deltas,
        score_terms=score_terms,
        score_deltas=score_deltas,
        point_metadata=dict(point_metadata),
        custody=custody,
    )


def _source_from_summary(
    summary_path: Path,
    *,
    source_id: str,
    allow_calibration_inputs: bool,
) -> ResponseSource:
    summary = _load_json_object(summary_path, label="official component-response summary")
    if summary.get("format") != SUMMARY_FORMAT:
        raise ComponentResponseStackOptimizerError(
            f"{summary_path}: expected format {SUMMARY_FORMAT!r}"
        )
    if summary.get("device") != "cuda":
        raise ComponentResponseStackOptimizerError(
            f"{summary_path}: non-CUDA summary rejected: device={summary.get('device')!r}"
        )
    if not allow_calibration_inputs and summary.get("promotion_eligible") is not True:
        raise ComponentResponseStackOptimizerError(
            f"{summary_path}: promotable recommendations require summary promotion_eligible=true"
        )

    curve_paths = summary.get("response_curve_paths")
    if not isinstance(curve_paths, Mapping):
        raise ComponentResponseStackOptimizerError(
            f"{summary_path}: response_curve_paths are required to verify canonical metadata"
        )
    curves: dict[str, dict[str, Any]] = {}
    curve_validations: dict[str, dict[str, Any]] = {}
    loaded_files = [summary_path]
    for component in CURVE_COMPONENTS:
        curve_path = _resolve_artifact_path(
            curve_paths.get(component),
            root=summary_path.parent,
            default_name=f"{component}_official_response_curve.json",
        )
        if not curve_path.is_file():
            raise ComponentResponseStackOptimizerError(
                f"{summary_path}: response curve for {component} not found: {curve_path}"
            )
        curve, validation = _load_curve_file(
            curve_path,
            component=component,
            allow_calibration_inputs=allow_calibration_inputs,
        )
        curves[component] = curve
        curve_validations[component] = validation
        loaded_files.append(curve_path)

    points = _points_by_epsilon(summary.get("points"), field=f"{summary_path}.points")
    zero = _zero_point(points.values(), field=f"{summary_path}.points")
    if zero is None:
        raise ComponentResponseStackOptimizerError(f"{summary_path}: summary requires eps=0 point")
    baseline_values = _values_from_mapping(zero.get("values", {}), field="summary.zero.values")
    baseline_archive = _archive_meta_from_mapping(zero, field="summary.zero")
    if "bytes" not in baseline_archive:
        raise ComponentResponseStackOptimizerError(f"{summary_path}: zero point archive bytes missing")
    baseline_score = _score_breakdown(
        pose=baseline_values["posenet"],
        seg=baseline_values["segnet"],
        archive_bytes=int(baseline_archive["bytes"]),
    )
    baseline = BaselineState(
        archive=baseline_archive,
        values=baseline_values,
        score=baseline_score,
    )

    for component, curve in curves.items():
        curve_baseline = curve.get("baseline")
        if not isinstance(curve_baseline, Mapping):
            raise ComponentResponseStackOptimizerError(
                f"{summary_path}: {component} curve missing baseline"
            )
        curve_baseline_archive = _archive_meta_from_mapping(
            curve_baseline,
            field=f"{component}.baseline",
        )
        _validate_archive_identity_match(
            baseline_archive,
            curve_baseline_archive,
            context=f"{summary_path}: {component} baseline",
        )
        curve_values = _values_from_mapping(
            curve_baseline.get("values", {}),
            field=f"{component}.baseline.values",
        )
        for key in COMPONENTS:
            if abs(curve_values[key] - baseline_values[key]) > 1e-10:
                raise ComponentResponseStackOptimizerError(
                    f"{summary_path}: {component} baseline {key} mismatch"
                )

    input_promotable = (
        summary.get("promotion_eligible") is True
        and all(validation["promotion_eligible"] for validation in curve_validations.values())
    )
    evidence_mode = (
        "promotable_cuda_official_component_response"
        if input_promotable
        else "calibration_cuda_official_component_response"
    )
    actions: list[ResponseAction] = []
    for point in sorted(
        points.values(),
        key=lambda item: _finite_float(item.get("epsilon"), field="summary.point.epsilon"),
    ):
        epsilon = _finite_float(point.get("epsilon"), field="summary.point.epsilon")
        if abs(epsilon) <= SCORE_EPS:
            continue
        values = _values_from_mapping(point.get("values", {}), field="summary.point.values")
        archive = _archive_meta_from_mapping(point, field="summary.point")
        contest_meta = _contest_eval_meta_from_mapping(point)
        if "bytes" not in archive or "sha256" not in archive:
            raise ComponentResponseStackOptimizerError(
                f"{summary_path}: point epsilon={epsilon} missing archive bytes/sha256"
            )
        point_metadata = point.get("point_metadata")
        if not isinstance(point_metadata, Mapping):
            point_metadata = {}
        actions.append(
            _make_action(
                source_id=source_id,
                source_path=summary_path,
                evidence_mode=evidence_mode,
                epsilon=epsilon,
                archive=archive,
                contest_auth_eval_json=contest_meta,
                values=values,
                baseline=baseline,
                point_metadata=point_metadata,
            )
        )

    validation = {
        "allow_calibration_inputs": bool(allow_calibration_inputs),
        "curve_validations": curve_validations,
        "summary_promotion_eligible": summary.get("promotion_eligible"),
    }
    return ResponseSource(
        source_id=source_id,
        input_path=summary_path.resolve(),
        input_format=SUMMARY_FORMAT,
        evidence_mode=evidence_mode,
        promotion_eligible_input=bool(input_promotable),
        baseline=baseline,
        actions=tuple(actions),
        source_files=_source_file_records(loaded_files),
        validation=validation,
    )


def _source_from_curves(
    curve_paths: Sequence[Path],
    *,
    source_id: str,
    input_path: Path,
    allow_calibration_inputs: bool,
) -> ResponseSource:
    curves: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    for path in sorted(curve_paths, key=lambda item: item.name):
        payload = _load_json_object(path, label="official component-response curve")
        component = payload.get("component")
        if component not in CURVE_COMPONENTS:
            raise ComponentResponseStackOptimizerError(
                f"{path}: component must be one of {CURVE_COMPONENTS}"
            )
        curve, validation = _load_curve_file(
            path,
            component=str(component),
            allow_calibration_inputs=allow_calibration_inputs,
        )
        curves[str(component)] = curve
        validations[str(component)] = validation
    if not curves:
        raise ComponentResponseStackOptimizerError(f"{input_path}: no response curves found")

    reference_component = "combined" if "combined" in curves else sorted(curves)[0]
    reference = curves[reference_component]
    baseline_payload = reference.get("baseline")
    if not isinstance(baseline_payload, Mapping):
        raise ComponentResponseStackOptimizerError(f"{input_path}: curve baseline is required")
    baseline_archive = _archive_meta_from_mapping(
        baseline_payload,
        field=f"{reference_component}.baseline",
    )
    baseline_values = _values_from_mapping(
        baseline_payload.get("values", {}),
        field=f"{reference_component}.baseline.values",
    )
    if "bytes" not in baseline_archive:
        raise ComponentResponseStackOptimizerError(f"{input_path}: baseline archive bytes missing")
    baseline = BaselineState(
        archive=baseline_archive,
        values=baseline_values,
        score=_score_breakdown(
            pose=baseline_values["posenet"],
            seg=baseline_values["segnet"],
            archive_bytes=int(baseline_archive["bytes"]),
        ),
    )

    point_maps = {
        component: _points_by_epsilon(curve.get("points"), field=f"{component}.points")
        for component, curve in curves.items()
    }
    reference_keys = set(point_maps[reference_component])
    for component, point_map in point_maps.items():
        if set(point_map) != reference_keys:
            raise ComponentResponseStackOptimizerError(
                f"{input_path}: {component} curve epsilon ladder differs"
            )
        curve_baseline = curves[component].get("baseline")
        if isinstance(curve_baseline, Mapping):
            _validate_archive_identity_match(
                baseline_archive,
                _archive_meta_from_mapping(curve_baseline, field=f"{component}.baseline"),
                context=f"{input_path}: {component} baseline",
            )

    input_promotable = all(validation["promotion_eligible"] for validation in validations.values())
    evidence_mode = (
        "promotable_cuda_official_component_response"
        if input_promotable
        else "calibration_cuda_official_component_response"
    )
    actions: list[ResponseAction] = []
    for key in sorted(reference_keys, key=lambda raw: float(raw)):
        ref_point = point_maps[reference_component][key]
        epsilon = _finite_float(ref_point.get("epsilon"), field="curve.point.epsilon")
        if abs(epsilon) <= SCORE_EPS:
            continue
        values: dict[str, float] = {}
        for component in COMPONENTS:
            if component in point_maps:
                component_point = point_maps[component][key]
                values[component] = _finite_float(
                    component_point.get("value"),
                    field=f"{component}.point.value",
                    minimum=0.0,
                )
            else:
                all_components = ref_point.get("all_components")
                if not isinstance(all_components, Mapping):
                    raise ComponentResponseStackOptimizerError(
                        f"{input_path}: single-curve input needs all_components for {component}"
                    )
                values[component] = _finite_float(
                    all_components.get(component),
                    field=f"{reference_component}.all_components.{component}",
                    minimum=0.0,
                )
        values["combined"] = _combined_no_rate(pose=values["posenet"], seg=values["segnet"])
        archive = _archive_meta_from_mapping(ref_point, field="curve.point")
        for component, point_map in point_maps.items():
            _validate_archive_identity_match(
                archive,
                _archive_meta_from_mapping(point_map[key], field=f"{component}.point"),
                context=f"{input_path}: epsilon={epsilon} {component}",
            )
        if "bytes" not in archive or "sha256" not in archive:
            raise ComponentResponseStackOptimizerError(
                f"{input_path}: point epsilon={epsilon} missing archive bytes/sha256"
            )
        point_metadata = ref_point.get("point_metadata")
        if not isinstance(point_metadata, Mapping):
            point_metadata = {}
        actions.append(
            _make_action(
                source_id=source_id,
                source_path=input_path,
                evidence_mode=evidence_mode,
                epsilon=epsilon,
                archive=archive,
                contest_auth_eval_json=_contest_eval_meta_from_mapping(ref_point),
                values=values,
                baseline=baseline,
                point_metadata=point_metadata,
            )
        )

    return ResponseSource(
        source_id=source_id,
        input_path=input_path.resolve(),
        input_format=CURVE_FORMAT,
        evidence_mode=evidence_mode,
        promotion_eligible_input=bool(input_promotable),
        baseline=baseline,
        actions=tuple(actions),
        source_files=_source_file_records(curve_paths),
        validation={
            "allow_calibration_inputs": bool(allow_calibration_inputs),
            "curve_validations": validations,
        },
    )


def _archive_meta_for_plan_point(
    point: Mapping[str, Any],
    *,
    root: Path,
    field: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_archive = point.get("archive")
    if raw_archive is None:
        raise ComponentResponseStackOptimizerError(f"{field}.archive is required")
    archive_path: Path | None = None
    if isinstance(raw_archive, str):
        archive_path = _resolve_artifact_path(raw_archive, root=root)
        archive = {"path": str(archive_path)}
    elif isinstance(raw_archive, Mapping):
        archive = _archive_meta_from_mapping(point, field=field)
        if isinstance(raw_archive.get("path"), str):
            archive_path = _resolve_artifact_path(raw_archive["path"], root=root)
            archive["path"] = str(archive_path)
    else:
        raise ComponentResponseStackOptimizerError(f"{field}.archive must be a path or object")
    if "archive_bytes" in point and "bytes" not in archive:
        archive["bytes"] = _require_int(point["archive_bytes"], field=f"{field}.archive_bytes", minimum=0)
    if "archive_sha256" in point and "sha256" not in archive:
        archive["sha256"] = str(point["archive_sha256"])
    custody: dict[str, Any] = {"archive_file_present": False}
    if archive_path is not None and archive_path.is_file():
        actual = _file_meta(archive_path)
        custody["archive_file_present"] = True
        custody["archive_file"] = actual
        archive.setdefault("bytes", actual["bytes"])
        archive.setdefault("sha256", actual["sha256"])
        if archive.get("bytes") != actual["bytes"] or archive.get("sha256") != actual["sha256"]:
            raise ComponentResponseStackOptimizerError(
                f"{field}: archive path does not match declared bytes/sha256"
            )
    if "bytes" not in archive or "sha256" not in archive:
        raise ComponentResponseStackOptimizerError(
            f"{field}: exact point archive requires recorded archive bytes and sha256"
        )
    return archive, custody


def _validate_contest_eval_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    n_samples = payload.get("n_samples")
    if n_samples != CONTEST_SAMPLE_COUNT:
        raise ComponentResponseStackOptimizerError(
            f"{path}: n_samples={n_samples!r}, expected {CONTEST_SAMPLE_COUNT}"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ComponentResponseStackOptimizerError(f"{path}: provenance object is required")
    if provenance.get("device") != "cuda":
        raise ComponentResponseStackOptimizerError(
            f"{path}: non-CUDA contest eval rejected: provenance.device={provenance.get('device')!r}"
        )
    tool = provenance.get("tool")
    if tool is not None and not str(tool).endswith("experiments/contest_auth_eval.py"):
        raise ComponentResponseStackOptimizerError(
            f"{path}: provenance.tool must be experiments/contest_auth_eval.py"
        )
    archive_bytes = _require_int(
        payload.get("archive_size_bytes"),
        field=f"{path}.archive_size_bytes",
        minimum=0,
    )
    if archive_bytes != int(archive["bytes"]):
        raise ComponentResponseStackOptimizerError(
            f"{path}: archive_size_bytes={archive_bytes!r} != declared {archive['bytes']!r}"
        )
    archive_sha = provenance.get("archive_sha256") or payload.get("archive_sha256")
    if archive_sha != archive.get("sha256"):
        raise ComponentResponseStackOptimizerError(
            f"{path}: archive sha256={archive_sha!r} != declared {archive.get('sha256')!r}"
        )
    values = _values_from_pose_seg(
        pose=_finite_float(payload.get("avg_posenet_dist"), field=f"{path}.avg_posenet_dist", minimum=0.0),
        seg=_finite_float(payload.get("avg_segnet_dist"), field=f"{path}.avg_segnet_dist", minimum=0.0),
    )
    expected = _score_breakdown(
        pose=values["posenet"],
        seg=values["segnet"],
        archive_bytes=archive_bytes,
    )
    recorded = payload.get("score_recomputed_from_components", payload.get("final_score"))
    recorded_score = _finite_float(recorded, field=f"{path}.score_recomputed_from_components")
    if abs(recorded_score - expected["score"]) > 1e-8:
        raise ComponentResponseStackOptimizerError(
            f"{path}: score_recomputed_from_components does not match contest formula"
        )
    return values


def _source_from_plan(
    plan_path: Path,
    *,
    source_id: str,
    allow_calibration_inputs: bool,
) -> ResponseSource:
    plan = _load_json_object(plan_path, label="official component-response plan")
    if plan.get("format") == PREDICTION_DELTAS_FORMAT:
        raise ComponentResponseStackOptimizerError(
            f"{plan_path}: prediction-only deltas cannot drive stack recommendations"
        )
    if plan.get("format") != PLAN_FORMAT:
        raise ComponentResponseStackOptimizerError(f"{plan_path}: unsupported format {plan.get('format')!r}")
    if not allow_calibration_inputs:
        raise ComponentResponseStackOptimizerError(
            f"{plan_path}: plan manifests do not carry promotable response gates; "
            "use --allow-calibration-inputs only for exact-point calibration"
        )
    perturbation = plan.get("perturbation")
    if isinstance(perturbation, Mapping):
        if perturbation.get("auth_eval_required") not in (None, "cuda"):
            raise ComponentResponseStackOptimizerError(
                f"{plan_path}: perturbation.auth_eval_required must be cuda"
            )
        canonical = perturbation.get("canonical_response_eval_path")
        if canonical not in (None, CANONICAL_RESPONSE_EVAL_PATH):
            raise ComponentResponseStackOptimizerError(
                f"{plan_path}: perturbation canonical path is not canonical"
            )

    raw_points = plan.get("points")
    points = _points_by_epsilon(raw_points, field=f"{plan_path}.points")
    if any(_contest_eval_meta_from_mapping(point) is None for point in points.values()):
        raise ComponentResponseStackOptimizerError(
            f"{plan_path}: prediction-only plan points rejected; every point needs "
            "contest_auth_eval_json exact CUDA evidence"
        )
    zero = _zero_point(points.values(), field=f"{plan_path}.points")
    if zero is None:
        raise ComponentResponseStackOptimizerError(
            f"{plan_path}: exact-point plan input requires an eps=0 baseline point"
        )

    point_records: dict[str, dict[str, Any]] = {}
    loaded_files = [plan_path]
    for key, point in points.items():
        archive, custody = _archive_meta_for_plan_point(
            point,
            root=plan_path.parent,
            field=f"{plan_path}.points[{key}]",
        )
        contest_meta = _contest_eval_meta_from_mapping(point)
        assert contest_meta is not None
        eval_path = _resolve_artifact_path(contest_meta, root=plan_path.parent)
        if not eval_path.is_file():
            raise ComponentResponseStackOptimizerError(
                f"{plan_path}: contest_auth_eval_json not found for epsilon {key}: {eval_path}"
            )
        loaded_files.append(eval_path)
        eval_payload = _load_json_object(eval_path, label="contest_auth_eval.json")
        values = _validate_contest_eval_payload(
            eval_payload,
            path=eval_path,
            archive=archive,
        )
        eval_meta = _file_meta(eval_path)
        contest_meta = {**contest_meta, **eval_meta}
        point_records[key] = {
            "archive": archive,
            "contest_auth_eval_json": contest_meta,
            "custody": custody,
            "epsilon": _finite_float(point.get("epsilon"), field="plan.point.epsilon"),
            "point_metadata": point.get("point_metadata") if isinstance(point.get("point_metadata"), Mapping) else {},
            "values": values,
        }

    zero_record = point_records[_epsilon_key(zero.get("epsilon"))]
    baseline_archive = zero_record["archive"]
    baseline_values = zero_record["values"]
    baseline = BaselineState(
        archive=baseline_archive,
        values=baseline_values,
        score=_score_breakdown(
            pose=baseline_values["posenet"],
            seg=baseline_values["segnet"],
            archive_bytes=int(baseline_archive["bytes"]),
        ),
    )
    actions: list[ResponseAction] = []
    for key in sorted(point_records, key=lambda raw: float(raw)):
        record = point_records[key]
        epsilon = float(record["epsilon"])
        if abs(epsilon) <= SCORE_EPS:
            continue
        actions.append(
            _make_action(
                source_id=source_id,
                source_path=plan_path,
                evidence_mode="calibration_cuda_exact_point_archives",
                epsilon=epsilon,
                archive=record["archive"],
                contest_auth_eval_json=record["contest_auth_eval_json"],
                values=record["values"],
                baseline=baseline,
                point_metadata=record["point_metadata"],
                custody_extra=record["custody"],
            )
        )

    return ResponseSource(
        source_id=source_id,
        input_path=plan_path.resolve(),
        input_format=PLAN_FORMAT,
        evidence_mode="calibration_cuda_exact_point_archives",
        promotion_eligible_input=False,
        baseline=baseline,
        actions=tuple(actions),
        source_files=_source_file_records(loaded_files),
        validation={
            "allow_calibration_inputs": True,
            "plan_exact_point_evidence": True,
            "promotable_response_gates_present": False,
        },
    )


def _load_source(
    path: Path,
    *,
    source_id: str,
    allow_calibration_inputs: bool,
) -> ResponseSource:
    path = path.resolve()
    if path.is_dir():
        summary = path / "official_component_response_summary.json"
        plan = path / "official_component_response_plan.json"
        curve_paths = sorted(path.glob("*_official_response_curve.json"))
        if summary.is_file():
            return _source_from_summary(
                summary,
                source_id=source_id,
                allow_calibration_inputs=allow_calibration_inputs,
            )
        if curve_paths:
            return _source_from_curves(
                curve_paths,
                source_id=source_id,
                input_path=path,
                allow_calibration_inputs=allow_calibration_inputs,
            )
        if plan.is_file():
            return _source_from_plan(
                plan,
                source_id=source_id,
                allow_calibration_inputs=allow_calibration_inputs,
            )
        raise ComponentResponseStackOptimizerError(f"{path}: no supported response JSON found")
    payload = _load_json_object(path, label="component-response input")
    fmt = payload.get("format")
    if fmt == SUMMARY_FORMAT:
        return _source_from_summary(
            path,
            source_id=source_id,
            allow_calibration_inputs=allow_calibration_inputs,
        )
    if fmt == CURVE_FORMAT:
        return _source_from_curves(
            [path],
            source_id=source_id,
            input_path=path,
            allow_calibration_inputs=allow_calibration_inputs,
        )
    if fmt in {PLAN_FORMAT, PREDICTION_DELTAS_FORMAT}:
        return _source_from_plan(
            path,
            source_id=source_id,
            allow_calibration_inputs=allow_calibration_inputs,
        )
    raise ComponentResponseStackOptimizerError(f"{path}: unsupported format {fmt!r}")


def _validate_shared_baseline(sources: Sequence[ResponseSource]) -> BaselineState:
    if not sources:
        raise ComponentResponseStackOptimizerError("at least one response input is required")
    baseline = sources[0].baseline
    for source in sources[1:]:
        _validate_archive_identity_match(
            baseline.archive,
            source.baseline.archive,
            context=f"{source.input_path}: shared baseline",
        )
        if int(source.baseline.archive.get("bytes", -1)) != int(baseline.archive["bytes"]):
            raise ComponentResponseStackOptimizerError(
                f"{source.input_path}: baseline archive bytes differ from first input"
            )
        for component in COMPONENTS:
            if abs(source.baseline.values[component] - baseline.values[component]) > 1e-10:
                raise ComponentResponseStackOptimizerError(
                    f"{source.input_path}: baseline {component} differs from first input"
                )
    return baseline


def _action_to_json(action: ResponseAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "archive": dict(action.archive),
        "contest_auth_eval_json": (
            dict(action.contest_auth_eval_json) if action.contest_auth_eval_json else None
        ),
        "custody": dict(action.custody),
        "deltas": dict(action.deltas),
        "epsilon": action.epsilon,
        "evidence_mode": action.evidence_mode,
        "point_metadata": dict(action.point_metadata),
        "score_deltas": dict(action.score_deltas),
        "score_terms": dict(action.score_terms),
        "source_id": action.source_id,
        "source_path": action.source_path,
        "values": dict(action.values),
    }


def _source_to_json(source: ResponseSource) -> dict[str, Any]:
    return {
        "action_count": len(source.actions),
        "baseline": {
            "archive": dict(source.baseline.archive),
            "score_terms": dict(source.baseline.score),
            "values": dict(source.baseline.values),
        },
        "evidence_mode": source.evidence_mode,
        "input_format": source.input_format,
        "input_path": str(source.input_path),
        "promotion_eligible_input": source.promotion_eligible_input,
        "source_files": [dict(item) for item in source.source_files],
        "source_id": source.source_id,
        "validation": dict(source.validation),
    }


def _metadata_int(
    metadata: Mapping[str, Any],
    keys: Sequence[str],
    *,
    minimum: int = 1,
) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return max(minimum, int(value))
        if isinstance(value, float) and value.is_integer():
            return max(minimum, int(value))
    return None


def _infer_atom_family(action: ResponseAction) -> str:
    metadata = action.point_metadata
    for key in ("atom_family", "family", "component_family"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    text = " ".join(
        [
            action.action_id,
            action.source_id,
            str(metadata.get("atom_id", "")),
            str(metadata.get("atom_kind", "")),
        ]
    ).lower()
    if "pose" in text or "qpose" in text or "qp1" in text:
        return "pose"
    if "mask" in text or "seg" in text:
        return "mask"
    if "post" in text or "residual" in text:
        return "postprocess"
    return "component_response"


def _action_interaction_risk(action: ResponseAction) -> tuple[str, float, list[str]]:
    pose_score_delta = float(action.score_deltas["posenet_score_delta"])
    seg_score_delta = float(action.score_deltas["segnet_score_delta"])
    component_score_delta = float(action.score_deltas["combined_no_rate_delta"])
    reasons: list[str] = []
    if component_score_delta >= 0.0:
        reasons.append("no_component_score_improvement")
    if pose_score_delta > 0.0:
        reasons.append("posenet_regresses")
    if seg_score_delta > 0.0:
        reasons.append("segnet_regresses")
    if pose_score_delta * seg_score_delta < 0.0:
        reasons.append("component_tradeoff")
    if not reasons:
        return "low", 0.0, []
    if "no_component_score_improvement" in reasons or len(reasons) >= 2:
        return "high", 0.35, reasons
    return "medium", 0.15, reasons


def _default_stack_interaction_flags(family: str) -> dict[str, list[str]]:
    if family == "pose":
        return {
            "synergizes_with": [
                "renderer_pose_conditioning",
                "hard_pair_trace_selection",
                "qpose_trained_decoder_contract",
            ],
            "antagonizes_with": [
                "untrained_renderer_pose_contract",
                "mask_geometry_without_pose_regeneration",
            ],
        }
    if family == "mask":
        return {
            "synergizes_with": [
                "mask_grammar_atoms",
                "segnet_hard_pair_selection",
                "renderer_conditioned_mask_decoder",
            ],
            "antagonizes_with": [
                "stale_pose_stream",
                "pose_sensitive_boundary_shift",
            ],
        }
    if family == "postprocess":
        return {
            "synergizes_with": [
                "component_trace_hard_pair_selection",
                "charged_residual_payload",
                "multipass_selector",
            ],
            "antagonizes_with": [
                "archive_byte_budget",
                "runtime_budget_if_decoder_branching_grows",
            ],
        }
    if family == "renderer_quantization":
        return {
            "synergizes_with": [
                "qzs3_grouped_quantization",
                "archive_overhead_single_member_pack",
                "scorer_path_qat",
            ],
            "antagonizes_with": [
                "segnet_texture_collapse",
                "posenet_geometry_collapse",
            ],
        }
    return {
        "synergizes_with": ["exact_trace_feedback", "offline_waterfill_selection"],
        "antagonizes_with": ["unvalidated_additive_delta_assumption"],
    }


def _metadata_string_list(metadata: Mapping[str, Any], key: str) -> list[str] | None:
    value = metadata.get(key)
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [str(item) for item in value]
    return None


def _exact_eval_stack_gate_recommendations() -> list[dict[str, Any]]:
    return [
        {
            "gate": "closed_archive_payload",
            "recommendation": (
                "Every selected atom must be charged inside archive.zip or fixed contest code; "
                "no scorer, network, or host-local sidecars."
            ),
        },
        {
            "gate": "canonical_cuda_auth_eval",
            "recommendation": (
                "Build the selected stack as its own archive and run archive.zip -> "
                "inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py "
                "--device cuda."
            ),
        },
        {
            "gate": "component_antagonism_review",
            "recommendation": (
                "Reject or split stacks whose exact eval shows PoseNet/SegNet regressions "
                "that exceed the selected atoms' expected water-fill utility."
            ),
        },
        {
            "gate": "t4_promotion_confirmation",
            "recommendation": (
                "Treat fast-chip diagnostics as planning evidence; promote only identical "
                "archive bytes with T4/equivalent custody and full component gates."
            ),
        },
    ]


def _allocation_atom_from_action(action: ResponseAction) -> dict[str, Any]:
    metadata = action.point_metadata
    family = _infer_atom_family(action)
    charged_bytes = _metadata_int(
        metadata,
        ("charged_bytes", "estimated_charged_bytes", "atom_bytes"),
    )
    charged_bytes_source = "point_metadata"
    if charged_bytes is None:
        charged_bytes = max(1, abs(int(action.deltas["archive_bytes_delta"])))
        charged_bytes_source = "abs(archive_bytes_delta)"
    expected_component_score_saved = max(
        0.0,
        -float(action.score_deltas["combined_no_rate_delta"]),
    )
    rate_cost = RATE_SLOPE_SCORE_PER_BYTE * charged_bytes
    risk_level, risk_factor, risk_reasons = _action_interaction_risk(action)
    risk_penalty = expected_component_score_saved * risk_factor
    utility = expected_component_score_saved - rate_cost - risk_penalty
    atom_id = str(metadata.get("atom_id") or action.action_id)
    flags = _default_stack_interaction_flags(family)
    return {
        "atom_id": atom_id,
        "action_id": action.action_id,
        "source_id": action.source_id,
        "family": family,
        "charged_bytes": charged_bytes,
        "charged_bytes_source": charged_bytes_source,
        "expected_component_score_saved": expected_component_score_saved,
        "expected_component_score_saved_per_byte": (
            expected_component_score_saved / charged_bytes
        ),
        "rate_score_cost": rate_cost,
        "rate_slope_score_per_byte": RATE_SLOPE_SCORE_PER_BYTE,
        "interaction_risk": risk_level,
        "interaction_risk_reasons": risk_reasons,
        "synergizes_with": _metadata_string_list(metadata, "synergizes_with")
        or flags["synergizes_with"],
        "antagonizes_with": _metadata_string_list(metadata, "antagonizes_with")
        or flags["antagonizes_with"],
        "risk_penalty_score": risk_penalty,
        "waterfill_utility_score": utility,
        "risk_adjusted_utility_per_byte": utility / charged_bytes,
        "waterfill_positive_ev": utility > 0.0,
        "component_score_deltas": {
            "combined_no_rate_delta": float(action.score_deltas["combined_no_rate_delta"]),
            "posenet_score_delta": float(action.score_deltas["posenet_score_delta"]),
            "segnet_score_delta": float(action.score_deltas["segnet_score_delta"]),
        },
        "archive_bytes_delta": int(action.deltas["archive_bytes_delta"]),
        "evidence_mode": action.evidence_mode,
        "score_claim": False,
        "requires_stacked_exact_eval": True,
    }


def _allocation_atom_sort_key(atom: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(atom["risk_adjusted_utility_per_byte"]),
        -float(atom["expected_component_score_saved_per_byte"]),
        {"low": 0, "medium": 1, "high": 2}.get(str(atom["interaction_risk"]), 9),
        str(atom["atom_id"]),
    )


def _atom_allocation_table(sources: Sequence[ResponseSource]) -> dict[str, Any]:
    atoms = [
        _allocation_atom_from_action(action)
        for source in sources
        for action in source.actions
    ]
    atoms = sorted(atoms, key=_allocation_atom_sort_key)
    for rank, atom in enumerate(atoms, start=1):
        atom["rank"] = rank
    family_summary: dict[str, Any] = {}
    for atom in atoms:
        family = str(atom["family"])
        summary = family_summary.setdefault(
            family,
            {"atom_count": 0, "positive_ev_count": 0, "top_atom_ids": []},
        )
        summary["atom_count"] += 1
        if atom["waterfill_positive_ev"]:
            summary["positive_ev_count"] += 1
        if len(summary["top_atom_ids"]) < 10:
            summary["top_atom_ids"].append(atom["atom_id"])
    return {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "derived_component_response_waterfill",
        "method": (
            "expected component-score reduction per charged byte minus rate "
            "cost and deterministic component interaction risk"
        ),
        "rate_slope_score_per_byte": RATE_SLOPE_SCORE_PER_BYTE,
        "exact_eval_stack_gate_recommendations": _exact_eval_stack_gate_recommendations(),
        "family_summary": family_summary,
        "ranked_atoms": atoms,
    }


def _combo_constraints(
    *,
    projected: Mapping[str, float],
    constraints: Mapping[str, float | int],
    actions: Sequence[ResponseAction],
) -> dict[str, Any]:
    archive_passed = projected["archive_bytes"] <= constraints["archive_bytes_budget"]
    pose_passed = projected["posenet_dist"] <= constraints["max_posenet_dist"]
    seg_passed = projected["segnet_dist"] <= constraints["max_segnet_dist"]
    formula_passed = all(math.isfinite(float(projected[key])) for key in projected)
    custody_passed = all(
        action.custody.get("archive_bytes_recorded") is True
        and action.custody.get("archive_sha256_recorded") is True
        and action.custody.get("contest_auth_eval_json_recorded") is True
        for action in actions
    )
    return {
        "archive_bytes_budget": {
            "limit": int(constraints["archive_bytes_budget"]),
            "margin": int(constraints["archive_bytes_budget"] - projected["archive_bytes"]),
            "passed": bool(archive_passed),
            "value": int(projected["archive_bytes"]),
        },
        "posenet_gate": {
            "limit": float(constraints["max_posenet_dist"]),
            "margin": float(constraints["max_posenet_dist"] - projected["posenet_dist"]),
            "passed": bool(pose_passed),
            "value": float(projected["posenet_dist"]),
        },
        "reproducibility_custody": {
            "passed": bool(custody_passed),
            "required_fields": [
                "archive.bytes",
                "archive.sha256",
                "contest_auth_eval_json",
                "source JSON sha256",
            ],
        },
        "score_formula": {
            "formula": "100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37545489",
            "passed": bool(formula_passed),
            "score": float(projected["score"]),
        },
        "segnet_gate": {
            "limit": float(constraints["max_segnet_dist"]),
            "margin": float(constraints["max_segnet_dist"] - projected["segnet_dist"]),
            "passed": bool(seg_passed),
            "value": float(projected["segnet_dist"]),
        },
    }


def _candidate_from_actions(
    actions: Sequence[ResponseAction],
    *,
    baseline: BaselineState,
    constraints: Mapping[str, float | int],
) -> dict[str, Any]:
    pose = float(baseline.values["posenet"]) + sum(
        float(action.deltas["posenet_dist_delta"]) for action in actions
    )
    seg = float(baseline.values["segnet"]) + sum(
        float(action.deltas["segnet_dist_delta"]) for action in actions
    )
    archive_bytes = int(baseline.archive["bytes"]) + sum(
        int(action.deltas["archive_bytes_delta"]) for action in actions
    )
    nonnegative_components = pose >= 0.0 and seg >= 0.0 and archive_bytes >= 0
    if nonnegative_components:
        projected = _score_breakdown(pose=pose, seg=seg, archive_bytes=archive_bytes)
        score_deltas = _score_delta(baseline.score, projected)
    else:
        projected = {
            "archive_bytes": float(archive_bytes),
            "combined_no_rate": INVALID_PROJECTED_SCORE,
            "posenet_dist": float(pose),
            "posenet_score_term": INVALID_PROJECTED_SCORE,
            "rate_score_term": INVALID_PROJECTED_SCORE,
            "score": INVALID_PROJECTED_SCORE,
            "segnet_dist": float(seg),
            "segnet_score_term": INVALID_PROJECTED_SCORE,
        }
        score_deltas = {
            "archive_bytes_delta": float(archive_bytes - int(baseline.archive["bytes"])),
            "combined_no_rate_delta": INVALID_PROJECTED_SCORE,
            "posenet_dist_delta": float(pose - baseline.values["posenet"]),
            "posenet_score_delta": INVALID_PROJECTED_SCORE,
            "rate_score_delta": INVALID_PROJECTED_SCORE,
            "score_delta": INVALID_PROJECTED_SCORE,
            "segnet_dist_delta": float(seg - baseline.values["segnet"]),
            "segnet_score_delta": INVALID_PROJECTED_SCORE,
        }
    constraint_results = _combo_constraints(
        projected=projected,
        constraints=constraints,
        actions=actions,
    )
    if not nonnegative_components:
        constraint_results["score_formula"]["passed"] = False
        constraint_results["score_formula"]["nonnegative_components"] = False
    feasible = all(bool(item["passed"]) for item in constraint_results.values())
    action_ids = [action.action_id for action in actions]
    stack_id = "stack_" + _canonical_hash(action_ids)[:16]
    input_modes = sorted({action.evidence_mode for action in actions})
    blockers = [
        {
            "code": "offline_additive_response_model_only",
            "mathematical_explanation": (
                "Component deltas are added only for deterministic planning. "
                "No stack composability claim is made without a stacked exact eval."
            ),
        },
        {
            "code": "stacked_exact_eval_required",
            "mathematical_explanation": (
                "A selected stack must be built as its own archive and evaluated "
                "through archive.zip -> inflate.sh -> upstream/evaluate.py."
            ),
        },
    ]
    if any(not mode.startswith("promotable_") for mode in input_modes):
        blockers.append(
            {
                "code": "calibration_or_exact_point_input",
                "mathematical_explanation": (
                    "At least one selected action came from calibration or exact-point "
                    "evidence rather than promotable response-curve gates."
                ),
            }
        )
    return {
        "action_count": len(actions),
        "composability_claim": False,
        "constraints": constraint_results,
        "evidence_modes": input_modes,
        "feasible_under_dykstra_constraints": bool(feasible),
        "promotion_blockers": blockers,
        "promotion_eligible": False,
        "requires_stacked_exact_eval": True,
        "score_deltas": score_deltas,
        "selected_actions": [_action_to_json(action) for action in actions],
        "selected_action_ids": action_ids,
        "source_ids": [action.source_id for action in actions],
        "stack_id": stack_id,
        "stack_score_claim": False,
        "projected_components": projected,
    }


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    projected = candidate["projected_components"]
    deltas = candidate["score_deltas"]
    return (
        float(projected["score"]),
        float(deltas["score_delta"]),
        int(projected["archive_bytes"]),
        float(projected["posenet_dist"]),
        float(projected["segnet_dist"]),
        tuple(candidate["selected_action_ids"]),
    )


def _infeasible_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    failed = sum(
        1
        for result in candidate["constraints"].values()
        if not bool(result.get("passed"))
    )
    return (failed, *_candidate_sort_key(candidate))


def _enumerate_candidates(
    sources: Sequence[ResponseSource],
    *,
    baseline: BaselineState,
    constraints: Mapping[str, float | int],
    max_stack_size: int,
    max_enumerated_stacks: int,
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    count = 0

    def visit(index: int, selected: list[ResponseAction]) -> None:
        nonlocal count
        if count > max_enumerated_stacks:
            return
        if index == len(sources):
            if selected:
                count += 1
                if count <= max_enumerated_stacks:
                    candidates.append(
                        _candidate_from_actions(
                            tuple(selected),
                            baseline=baseline,
                            constraints=constraints,
                        )
                    )
            return
        visit(index + 1, selected)
        if len(selected) >= max_stack_size:
            return
        for action in sources[index].actions:
            selected.append(action)
            visit(index + 1, selected)
            selected.pop()

    visit(0, [])
    if count > max_enumerated_stacks:
        raise ComponentResponseStackOptimizerError(
            "candidate stack search exceeded max_enumerated_stacks="
            f"{max_enumerated_stacks}; reduce inputs, lower --max-stack-size, "
            "or raise the limit"
        )
    return candidates, count


def optimize_component_response_stack(
    input_paths: Sequence[Path | str],
    *,
    archive_bytes_budget: int | None = None,
    max_posenet_dist: float | None = None,
    max_segnet_dist: float | None = None,
    top_k: int = 10,
    max_stack_size: int | None = None,
    allow_calibration_inputs: bool = False,
    max_enumerated_stacks: int = 200_000,
    infeasible_top_k: int = 5,
) -> dict[str, Any]:
    if not input_paths:
        raise ComponentResponseStackOptimizerError("at least one input path is required")
    top_k = _require_int(top_k, field="top_k", minimum=1)
    infeasible_top_k = _require_int(infeasible_top_k, field="infeasible_top_k", minimum=0)
    max_enumerated_stacks = _require_int(
        max_enumerated_stacks,
        field="max_enumerated_stacks",
        minimum=1,
    )
    resolved_inputs = [Path(path).resolve() for path in input_paths]
    sources: list[ResponseSource] = []
    for index, path in enumerate(resolved_inputs):
        source_id = f"source_{index:03d}_{_slug(path.stem if path.is_file() else path.name)}"
        source = _load_source(
            path,
            source_id=source_id,
            allow_calibration_inputs=allow_calibration_inputs,
        )
        if not source.actions:
            raise ComponentResponseStackOptimizerError(f"{path}: no nonzero response points found")
        sources.append(source)

    baseline = _validate_shared_baseline(sources)
    archive_budget = (
        _require_int(archive_bytes_budget, field="archive_bytes_budget", minimum=0)
        if archive_bytes_budget is not None
        else int(baseline.archive["bytes"])
    )
    pose_gate = (
        _finite_float(max_posenet_dist, field="max_posenet_dist", minimum=0.0)
        if max_posenet_dist is not None
        else float(baseline.values["posenet"])
    )
    seg_gate = (
        _finite_float(max_segnet_dist, field="max_segnet_dist", minimum=0.0)
        if max_segnet_dist is not None
        else float(baseline.values["segnet"])
    )
    if max_stack_size is None:
        effective_max_stack_size = len(sources)
    else:
        effective_max_stack_size = _require_int(max_stack_size, field="max_stack_size", minimum=1)
    effective_max_stack_size = min(effective_max_stack_size, len(sources))
    constraints = {
        "archive_bytes_budget": archive_budget,
        "max_posenet_dist": pose_gate,
        "max_segnet_dist": seg_gate,
    }
    candidates, considered = _enumerate_candidates(
        sources,
        baseline=baseline,
        constraints=constraints,
        max_stack_size=effective_max_stack_size,
        max_enumerated_stacks=max_enumerated_stacks,
    )
    feasible = sorted(
        [
            candidate
            for candidate in candidates
            if candidate["feasible_under_dykstra_constraints"]
        ],
        key=_candidate_sort_key,
    )
    infeasible = sorted(
        [
            candidate
            for candidate in candidates
            if not candidate["feasible_under_dykstra_constraints"]
        ],
        key=_infeasible_sort_key,
    )
    input_modes = sorted({source.evidence_mode for source in sources})
    all_inputs_promotable = all(source.promotion_eligible_input for source in sources)
    atom_allocation = _atom_allocation_table(sources)
    return {
        "schema_version": 1,
        "format": FORMAT,
        "tool": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "recommendation_promotable": False,
        "composability_claim": False,
        "requires_stacked_exact_eval": True,
        "input_evidence_modes": input_modes,
        "input_evidence_promotable": bool(all_inputs_promotable),
        "non_promotable_reasons": [
            "offline optimizer emits planning candidates only",
            "selected stack must be exact-evaluated as its own archive",
        ],
        "score_formula": {
            "archive_bytes_denominator": ORIGINAL_VIDEO_BYTES,
            "formula": "100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37545489",
            "separated_terms": ["segnet_score_term", "posenet_score_term", "rate_score_term"],
        },
        "dykstra_constraints": {
            "archive_bytes_budget": archive_budget,
            "max_posenet_dist": pose_gate,
            "max_segnet_dist": seg_gate,
            "projection_order": [
                "archive_bytes_budget",
                "posenet_gate",
                "segnet_gate",
                "score_formula",
                "reproducibility_custody",
            ],
        },
        "baseline": {
            "archive": dict(baseline.archive),
            "score_terms": dict(baseline.score),
            "values": dict(baseline.values),
        },
        "optimization": {
            "actions_available": sum(len(source.actions) for source in sources),
            "candidate_stacks_considered": considered,
            "feasible_candidate_count": len(feasible),
            "infeasible_candidate_count": len(infeasible),
            "knapsack_objective": "minimize projected contest score under Dykstra-style constraints",
            "max_enumerated_stacks": max_enumerated_stacks,
            "max_stack_size": effective_max_stack_size,
            "one_action_per_source": True,
            "top_k": top_k,
        },
        "atom_allocation_table": atom_allocation,
        "input_sources": [_source_to_json(source) for source in sources],
        "recommendations": feasible[:top_k],
        "infeasible_recommendations": infeasible[:infeasible_top_k],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help=(
            "official_component_response_summary.json, response-curve JSON, "
            "directory containing those files, or an exact-point response plan"
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--archive-bytes-budget", type=int, default=None)
    parser.add_argument("--max-posenet-dist", type=float, default=None)
    parser.add_argument("--max-segnet-dist", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--infeasible-top-k", type=int, default=5)
    parser.add_argument("--max-stack-size", type=int, default=None)
    parser.add_argument("--max-enumerated-stacks", type=int, default=200_000)
    parser.add_argument(
        "--allow-calibration-inputs",
        action="store_true",
        help=(
            "Accept canonical CUDA calibration curves or exact-point plan inputs "
            "that are not promotable response-curve evidence. Output remains non-promotable."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = optimize_component_response_stack(
        args.inputs,
        archive_bytes_budget=args.archive_bytes_budget,
        max_posenet_dist=args.max_posenet_dist,
        max_segnet_dist=args.max_segnet_dist,
        top_k=args.top_k,
        max_stack_size=args.max_stack_size,
        allow_calibration_inputs=args.allow_calibration_inputs,
        max_enumerated_stacks=args.max_enumerated_stacks,
        infeasible_top_k=args.infeasible_top_k,
    )
    if args.output_json is not None:
        _write_json(args.output_json, result)
    else:
        print(_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
