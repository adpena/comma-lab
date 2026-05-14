#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Produce official component-response curves from canonical archive evals.

This is intentionally narrower than ``profile_component_sensitivity.py``. It
does not perturb renderer tensors in process and it does not load scorer models.
Each response point is an exact archive evaluated through:

    archive.zip -> inflate.sh -> upstream/evaluate.py

via ``experiments/contest_auth_eval.py`` or a pre-existing
``contest_auth_eval.json`` with matching archive custody. The output response
curves are suitable for ``build_component_sensitivity_manifest.py`` only when
all promotion gates pass. This script does not create component maps or
stability JSON; those remain separate inputs to the manifest builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("posenet", "segnet", "combined")
CONTEST_SAMPLE_COUNT = 600
RESPONSE_OUTPUT_FORMAT = "official_component_response_curves_v1"
PLAN_FORMAT = "official_component_response_plan_v1"
SCORE_EPS = 1e-12
SIGNED_DELTA_ERROR_MODE = "signed_delta"
ABSOLUTE_MAGNITUDE_ERROR_MODE = "absolute_magnitude"
DEFAULT_GATE_SPEC = {
    "zero_repro_tolerance": 1e-7,
    "external_baseline_repro_tolerance": 1e-7,
    "observed_delta_min": 1e-12,
    "holdout_error_max": 0.35,
    "require_prediction_deltas": True,
}
COMPONENT_READOUTS = {
    "posenet": "official_pose_mse",
    "segnet": "official_argmax_disagreement",
    "combined": "official_component_formula",
}
COMPONENT_UNITS = {
    "posenet": "mean_pose_mse",
    "segnet": "mean_argmax_disagreement",
    "combined": "100*segnet + sqrt(10*posenet), no rate term",
}


class OfficialComponentResponseError(ValueError):
    """Raised when official component-response curves cannot be produced."""


@dataclass(frozen=True)
class PointSpec:
    index: int
    epsilon: float
    archive: Path
    contest_auth_eval_json: Path | None
    predicted_deltas: Mapping[str, float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PointResult:
    spec: PointSpec
    contest_auth_eval_json: Path
    contest_payload: Mapping[str, Any]
    archive: Mapping[str, Any]
    contest_json: Mapping[str, Any]
    values: Mapping[str, float]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise OfficialComponentResponseError(f"{path}: invalid JSON") from exc


def _finite_float(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfficialComponentResponseError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise OfficialComponentResponseError(f"{field} must be finite")
    if minimum is not None and out < minimum:
        raise OfficialComponentResponseError(f"{field} must be >= {minimum}")
    return out


def _resolve_path(value: str | Path, *, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _require_existing_file(path: Path, *, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise OfficialComponentResponseError(f"{label} not found: {path}")
    return path


def _component_values_from_contest_eval(payload: Mapping[str, Any]) -> dict[str, float]:
    pose = _finite_float(
        payload.get("avg_posenet_dist"),
        field="contest_auth_eval.avg_posenet_dist",
        minimum=0.0,
    )
    seg = _finite_float(
        payload.get("avg_segnet_dist"),
        field="contest_auth_eval.avg_segnet_dist",
        minimum=0.0,
    )
    return {
        "posenet": pose,
        "segnet": seg,
        "combined": 100.0 * seg + math.sqrt(10.0 * pose),
    }


def _validate_contest_eval_json(
    path: Path,
    *,
    archive: Path,
    device: str,
) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise OfficialComponentResponseError(f"{path}: contest eval must be a JSON object")

    n_samples = payload.get("n_samples")
    if n_samples != CONTEST_SAMPLE_COUNT:
        raise OfficialComponentResponseError(
            f"{path}: n_samples={n_samples!r}, expected {CONTEST_SAMPLE_COUNT}"
        )

    provenance = payload.get("provenance")
    provenance_device = provenance.get("device") if isinstance(provenance, dict) else None
    if provenance_device != device:
        raise OfficialComponentResponseError(
            f"{path}: provenance.device={provenance_device!r}, expected {device!r}"
        )

    archive_meta = _file_meta(archive)
    payload_bytes = payload.get("archive_size_bytes")
    payload_sha = provenance.get("archive_sha256") if isinstance(provenance, dict) else None
    payload_sha = payload_sha or payload.get("archive_sha256")
    if payload_bytes != archive_meta["bytes"]:
        raise OfficialComponentResponseError(
            f"{path}: archive_size_bytes={payload_bytes!r} does not match "
            f"{archive} bytes={archive_meta['bytes']}"
        )
    if payload_sha != archive_meta["sha256"]:
        raise OfficialComponentResponseError(
            f"{path}: archive_sha256={payload_sha!r} does not match "
            f"{archive} sha256={archive_meta['sha256']}"
        )

    _component_values_from_contest_eval(payload)
    return payload


def _copy_existing_eval_json(src: Path, *, point_dir: Path) -> Path:
    point_dir.mkdir(parents=True, exist_ok=True)
    dest = point_dir / "contest_auth_eval.json"
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def _safe_point_label(index: int, epsilon: float) -> str:
    eps = f"{epsilon:+.12g}".replace("+", "p").replace("-", "m")
    eps = re.sub(r"[^A-Za-z0-9_.]+", "_", eps)
    return f"point_{index:03d}_eps_{eps}"


def _run_contest_auth_eval(
    *,
    archive: Path,
    point_dir: Path,
    contest_auth_eval_script: Path,
    inflate_sh: Path,
    upstream_dir: Path,
    video_names_file: Path,
    device: str,
    inflate_timeout: int,
    evaluate_timeout: int,
) -> Path:
    point_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = point_dir / "contest_auth_eval.stdout.log"
    stderr_path = point_dir / "contest_auth_eval.stderr.log"
    cmd = [
        sys.executable,
        str(contest_auth_eval_script),
        "--archive",
        str(archive),
        "--inflate-sh",
        str(inflate_sh),
        "--upstream-dir",
        str(upstream_dir),
        "--video-names-file",
        str(video_names_file),
        "--device",
        device,
        "--work-dir",
        str(point_dir),
        "--inflate-timeout",
        str(inflate_timeout),
        "--evaluate-timeout",
        str(evaluate_timeout),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    if result.returncode != 0:
        raise OfficialComponentResponseError(
            "contest_auth_eval failed for "
            f"{archive} with returncode {result.returncode}; see {stderr_path}"
        )
    out_json = point_dir / "contest_auth_eval.json"
    if not out_json.exists():
        raise OfficialComponentResponseError(
            f"contest_auth_eval completed but did not write {out_json}"
        )
    return out_json


def _prediction_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    for nested_key in ("component_deltas", "component_predictions", "predicted_delta", "predicted_deltas"):
        nested = value.get(nested_key)
        if isinstance(nested, Mapping):
            return _prediction_mapping(nested)

    out: dict[str, float] = {}
    for component in COMPONENTS:
        if component in value:
            out[component] = _finite_float(
                value[component],
                field=f"predicted_deltas.{component}",
            )
    return out


def _point_predictions(item: Mapping[str, Any]) -> dict[str, float]:
    for key in (
        "predicted_delta",
        "predicted_deltas",
        "component_predictions",
        "prediction",
    ):
        predictions = _prediction_mapping(item.get(key))
        if predictions:
            return predictions
    return {}


def _load_plan_points(
    plan_path: Path,
    *,
    baseline_archive: Path,
    baseline_contest_auth_eval_json: Path | None,
) -> tuple[dict[str, Any], list[PointSpec]]:
    payload = _load_json(plan_path)
    if isinstance(payload, list):
        plan: dict[str, Any] = {"format": PLAN_FORMAT, "points": payload}
    elif isinstance(payload, dict):
        plan = dict(payload)
    else:
        raise OfficialComponentResponseError(f"{plan_path}: plan must be a JSON object or list")

    raw_points = plan.get("points")
    if not isinstance(raw_points, list):
        raise OfficialComponentResponseError(f"{plan_path}: points must be a list")

    root = plan_path.parent
    top_baseline_json = plan.get("baseline_contest_auth_eval_json")
    if baseline_contest_auth_eval_json is None and top_baseline_json is not None:
        baseline_contest_auth_eval_json = _resolve_path(str(top_baseline_json), root=root)

    specs: list[PointSpec] = []
    has_zero = False
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping):
            raise OfficialComponentResponseError(f"{plan_path}: points[{index}] must be an object")
        epsilon = _finite_float(raw.get("epsilon"), field=f"points[{index}].epsilon")
        archive_value = raw.get("archive")
        archive = (
            baseline_archive
            if archive_value is None and abs(epsilon) <= SCORE_EPS
            else _resolve_path(str(archive_value), root=root)
            if archive_value is not None
            else None
        )
        if archive is None:
            raise OfficialComponentResponseError(
                f"{plan_path}: points[{index}].archive is required for nonzero epsilon"
            )
        _require_existing_file(archive, label=f"points[{index}].archive")

        json_value = raw.get("contest_auth_eval_json")
        eval_json = None
        if json_value is not None:
            eval_json = _resolve_path(str(json_value), root=root)
            _require_existing_file(eval_json, label=f"points[{index}].contest_auth_eval_json")
        elif abs(epsilon) <= SCORE_EPS:
            eval_json = baseline_contest_auth_eval_json

        has_zero = has_zero or abs(epsilon) <= SCORE_EPS
        specs.append(
            PointSpec(
                index=index,
                epsilon=epsilon,
                archive=archive,
                contest_auth_eval_json=eval_json,
                predicted_deltas=_point_predictions(raw),
                metadata={
                    str(key): raw[key]
                    for key in sorted(raw)
                    if key
                    not in {
                        "archive",
                        "contest_auth_eval_json",
                        "epsilon",
                        "predicted_delta",
                        "predicted_deltas",
                        "component_predictions",
                        "prediction",
                    }
                },
            )
        )

    if not has_zero:
        specs.insert(
            0,
            PointSpec(
                index=-1,
                epsilon=0.0,
                archive=baseline_archive,
                contest_auth_eval_json=baseline_contest_auth_eval_json,
                predicted_deltas={component: 0.0 for component in COMPONENTS},
                metadata={"role": "auto_inserted_baseline"},
            ),
        )

    epsilons = [spec.epsilon for spec in specs]
    rounded = [round(eps, 15) for eps in epsilons]
    if len(set(rounded)) != len(rounded):
        raise OfficialComponentResponseError("plan contains duplicate epsilon values")
    if not any(abs(eps) > SCORE_EPS for eps in epsilons):
        raise OfficialComponentResponseError("plan must contain at least one nonzero response point")
    return plan, sorted(specs, key=lambda item: (item.epsilon, item.index))


def _preflight_promotion_predictions(specs: list[PointSpec]) -> None:
    missing: list[str] = []
    for spec in specs:
        if abs(spec.epsilon) <= SCORE_EPS:
            continue
        missing_components = [
            component
            for component in ("posenet", "segnet")
            if component not in spec.predicted_deltas
        ]
        if missing_components:
            missing.append(
                f"points[{spec.index}] epsilon={spec.epsilon}: "
                f"missing {', '.join(missing_components)}"
            )
    if missing:
        raise OfficialComponentResponseError(
            "promotion preflight failed before evaluate_points: every nonzero "
            "response point requires posenet and segnet predicted_delta values; "
            + "; ".join(missing)
        )


def _force_same_run_zero_baseline(
    specs: list[PointSpec],
    *,
    baseline_archive: Path,
) -> list[PointSpec]:
    forced: list[PointSpec] = []
    for spec in specs:
        if abs(spec.epsilon) > SCORE_EPS:
            forced.append(spec)
            continue
        if spec.archive.resolve() != baseline_archive.resolve():
            raise OfficialComponentResponseError(
                "same-run eps=0 baseline requires the zero point archive to match "
                f"--baseline-archive; got {spec.archive}"
            )
        metadata = dict(spec.metadata)
        metadata["same_run_zero_baseline"] = True
        if spec.contest_auth_eval_json is not None:
            metadata["external_zero_contest_auth_eval_json_ignored_for_curve"] = str(
                spec.contest_auth_eval_json
            )
        forced.append(
            PointSpec(
                index=spec.index,
                epsilon=spec.epsilon,
                archive=spec.archive,
                contest_auth_eval_json=None,
                predicted_deltas=spec.predicted_deltas,
                metadata=metadata,
            )
        )
    return forced


def evaluate_points(
    specs: list[PointSpec],
    *,
    output_dir: Path,
    contest_auth_eval_script: Path,
    inflate_sh: Path,
    upstream_dir: Path,
    video_names_file: Path,
    device: str,
    inflate_timeout: int,
    evaluate_timeout: int,
) -> list[PointResult]:
    results: list[PointResult] = []
    for ordinal, spec in enumerate(specs):
        label = _safe_point_label(ordinal, spec.epsilon)
        point_dir = output_dir / "evals" / label
        if spec.contest_auth_eval_json is not None:
            eval_json = _copy_existing_eval_json(spec.contest_auth_eval_json, point_dir=point_dir)
        else:
            eval_json = _run_contest_auth_eval(
                archive=spec.archive,
                point_dir=point_dir,
                contest_auth_eval_script=contest_auth_eval_script,
                inflate_sh=inflate_sh,
                upstream_dir=upstream_dir,
                video_names_file=video_names_file,
                device=device,
                inflate_timeout=inflate_timeout,
                evaluate_timeout=evaluate_timeout,
            )
        payload = _validate_contest_eval_json(eval_json, archive=spec.archive, device=device)
        results.append(
            PointResult(
                spec=spec,
                contest_auth_eval_json=eval_json,
                contest_payload=payload,
                archive=_file_meta(spec.archive),
                contest_json=_file_meta(eval_json),
                values=_component_values_from_contest_eval(payload),
            )
        )
    return results


def _is_close(a: float, b: float) -> bool:
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= 1e-9 * scale


def _coverage_metadata(
    epsilons: list[float],
    *,
    allow_directional: bool,
    directional_action: Any,
) -> tuple[dict[str, Any], bool]:
    values = sorted(set(float(eps) for eps in epsilons))
    positives = [value for value in values if value > SCORE_EPS]
    negatives = [value for value in values if value < -SCORE_EPS]
    zero_present = any(abs(value) <= SCORE_EPS for value in values)
    pair_count = sum(
        1
        for positive in positives
        if any(_is_close(negative, -positive) for negative in negatives)
    )
    if zero_present and pair_count > 0:
        return (
            {
                "response_kind": "symmetric",
                "epsilon_ladder": values,
                "symmetric_epsilon_pairs": int(pair_count),
            },
            True,
        )
    if allow_directional and zero_present and (positives or negatives):
        action = directional_action or {
            "action": "archive_perturbation_plan_directional_response",
            "epsilon_values": values,
        }
        return (
            {
                "response_kind": "directional",
                "epsilon_ladder": values,
                "directional_action": action,
            },
            True,
        )
    return (
        {
            "response_kind": "directional" if positives or negatives else "invalid",
            "epsilon_ladder": values,
        },
        False,
    )


def _prediction_for_component(
    raw_predictions: Mapping[str, float],
    *,
    component: str,
    baseline: Mapping[str, float],
) -> float | None:
    if component in raw_predictions:
        return float(raw_predictions[component])
    if component != "combined":
        return None
    if "posenet" not in raw_predictions or "segnet" not in raw_predictions:
        return None
    pose_pred = float(baseline["posenet"]) + float(raw_predictions["posenet"])
    if pose_pred < 0.0:
        raise OfficialComponentResponseError(
            "combined prediction from posenet/segnet deltas would make pose_dist negative"
        )
    return (
        100.0 * float(raw_predictions["segnet"])
        + math.sqrt(10.0 * pose_pred)
        - math.sqrt(10.0 * float(baseline["posenet"]))
    )


def _prediction_error_mode(plan: Mapping[str, Any]) -> str:
    perturbation = plan.get("perturbation")
    model: Mapping[str, Any] = {}
    if isinstance(perturbation, Mapping) and isinstance(perturbation.get("prediction_model"), Mapping):
        model = perturbation["prediction_model"]
    elif isinstance(plan.get("prediction_model"), Mapping):
        model = plan["prediction_model"]

    explicit_mode = model.get("prediction_error_mode")
    if explicit_mode is not None:
        mode = str(explicit_mode)
        if mode in {SIGNED_DELTA_ERROR_MODE, ABSOLUTE_MAGNITUDE_ERROR_MODE}:
            return mode
        raise OfficialComponentResponseError(
            f"unsupported prediction_error_mode={mode!r}; expected "
            f"{SIGNED_DELTA_ERROR_MODE!r} or {ABSOLUTE_MAGNITUDE_ERROR_MODE!r}"
        )

    semantics = str(model.get("prediction_delta_semantics", "")).lower()
    sign_policy = str(model.get("sign_policy", "")).lower()
    if "nonnegative" in semantics or ("nonnegative" in sign_policy and "magnitude" in sign_policy):
        return ABSOLUTE_MAGNITUDE_ERROR_MODE
    return SIGNED_DELTA_ERROR_MODE


def _prediction_error_payload(
    *,
    predicted: float,
    observed_delta: float,
    mode: str,
) -> dict[str, Any]:
    if mode == ABSOLUTE_MAGNITUDE_ERROR_MODE:
        if predicted < -SCORE_EPS:
            return {
                "valid": False,
                "predicted_delta_for_error": predicted,
                "observed_delta_for_error": abs(observed_delta),
                "abs_error": None,
                "relative_error": None,
            }
        predicted_for_error = abs(predicted)
        observed_for_error = abs(observed_delta)
    elif mode == SIGNED_DELTA_ERROR_MODE:
        predicted_for_error = predicted
        observed_for_error = observed_delta
    else:
        raise OfficialComponentResponseError(f"unsupported prediction error mode: {mode!r}")

    abs_error = abs(predicted_for_error - observed_for_error)
    denom = max(abs(observed_for_error), 1e-12)
    return {
        "valid": True,
        "predicted_delta_for_error": predicted_for_error,
        "observed_delta_for_error": observed_for_error,
        "abs_error": abs_error,
        "relative_error": abs_error / denom,
    }


def _blocker(code: str, explanation: str) -> dict[str, str]:
    return {"code": code, "mathematical_explanation": explanation}


def _curve_for_component(
    *,
    component: str,
    results: list[PointResult],
    baseline: PointResult,
    gate_spec: Mapping[str, float],
    coverage: Mapping[str, Any],
    coverage_passed: bool,
    plan: Mapping[str, Any],
    device: str,
    external_baseline_values: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    missing_predictions = False
    invalid_prediction_semantics = False
    finite_values = True
    prediction_errors: list[float] = []
    abs_prediction_errors: list[float] = []
    nonzero_observed: list[float] = []
    zero_deltas: list[float] = []

    baseline_values = dict(baseline.values)
    prediction_error_mode = _prediction_error_mode(plan)
    for result in results:
        value = float(result.values[component])
        base_value = float(baseline_values[component])
        delta = value - base_value
        predicted = _prediction_for_component(
            result.spec.predicted_deltas,
            component=component,
            baseline=baseline_values,
        )
        if abs(result.spec.epsilon) <= SCORE_EPS and predicted is None:
            predicted = 0.0
        if abs(result.spec.epsilon) > SCORE_EPS:
            nonzero_observed.append(abs(delta))
        else:
            zero_deltas.append(abs(delta))

        prediction_payload: dict[str, Any]
        if predicted is None:
            missing_predictions = True
            prediction_payload = {
                "implemented": False,
                "predicted_delta": None,
                "abs_error": None,
                "relative_error": None,
            }
        else:
            pred = float(predicted)
            error_payload = _prediction_error_payload(
                predicted=pred,
                observed_delta=delta,
                mode=prediction_error_mode,
            )
            if not error_payload["valid"]:
                invalid_prediction_semantics = True
                abs_error = None
                relative_error = None
            else:
                abs_error = float(error_payload["abs_error"])
                relative_error = float(error_payload["relative_error"])
                prediction_errors.append(relative_error)
                abs_prediction_errors.append(abs_error)
            prediction_payload = {
                "implemented": True,
                "valid": bool(error_payload["valid"]),
                "predicted_delta": pred,
                "prediction_error_mode": prediction_error_mode,
                "predicted_delta_for_error": error_payload["predicted_delta_for_error"],
                "observed_delta_for_error": error_payload["observed_delta_for_error"],
                "abs_error": abs_error,
                "relative_error": relative_error,
            }

        finite_values = finite_values and all(
            math.isfinite(item)
            for item in (
                value,
                delta,
                *(prediction_errors[-1:] if prediction_errors else []),
                *(abs_prediction_errors[-1:] if abs_prediction_errors else []),
            )
        )
        points.append(
            {
                "epsilon": float(result.spec.epsilon),
                "value": value,
                "baseline": base_value,
                "delta": delta,
                "all_components": dict(result.values),
                "prediction": prediction_payload,
                "archive": dict(result.archive),
                "contest_auth_eval_json": dict(result.contest_json),
                "point_metadata": dict(result.spec.metadata),
            }
        )

    observed_delta_max = max(nonzero_observed, default=0.0)
    zero_repro_error = max(zero_deltas, default=None)
    zero_repro = zero_repro_error is not None and zero_repro_error <= gate_spec["zero_repro_tolerance"]
    external_baseline_value = None
    external_baseline_repro_error = None
    external_baseline_repro = True
    if external_baseline_values is not None:
        external_baseline_value = float(external_baseline_values[component])
        external_baseline_repro_error = abs(float(baseline_values[component]) - external_baseline_value)
        external_baseline_repro = (
            external_baseline_repro_error <= gate_spec["external_baseline_repro_tolerance"]
        )
    signal_present = observed_delta_max >= gate_spec["observed_delta_min"]
    prediction_error = max(prediction_errors, default=0.0)
    prediction_abs_error = max(abs_prediction_errors, default=0.0)
    prediction_error_passed = (
        not missing_predictions
        and not invalid_prediction_semantics
        and prediction_errors
        and prediction_error <= gate_spec["holdout_error_max"]
    )
    promotion_passed = bool(
        device == "cuda"
        and finite_values
        and coverage_passed
        and zero_repro
        and external_baseline_repro
        and signal_present
        and prediction_error_passed
    )

    blockers: list[dict[str, str]] = []
    if device != "cuda":
        blockers.append(
            _blocker(
                "non_cuda_official_component_response",
                "Official component-response curves are promotable only when authored on CUDA.",
            )
        )
    if not finite_values:
        blockers.append(
            _blocker(
                "nonfinite_component_response",
                "All official component values, deltas, and prediction errors must be finite.",
            )
        )
    if not coverage_passed:
        blockers.append(
            _blocker(
                "response_coverage_failed",
                "Promotion response curves require eps=0 plus a matched -eps/+eps pair, "
                "or an explicit directional action when directional mode is allowed.",
            )
        )
    if not zero_repro:
        blockers.append(
            _blocker(
                "zero_reproduction_failed",
                "The eps=0 official archive response did not reproduce the baseline component value.",
            )
        )
    if not external_baseline_repro:
        blockers.append(
            _blocker(
                "external_baseline_reproduction_failed",
                "The same-run eps=0 official archive response did not reproduce the supplied "
                "baseline contest_auth_eval.json component value. Treat this as runner/scorer "
                "calibration drift, not sensitivity evidence.",
            )
        )
    if not signal_present:
        blockers.append(
            _blocker(
                "response_signal_absent",
                "Observed official component response is below the minimum signal floor.",
            )
        )
    if missing_predictions:
        blockers.append(
            _blocker(
                "missing_prediction_deltas",
                "Every nonzero response point needs map-predicted component deltas "
                "before the curve can calibrate sensitivity-map accuracy.",
            )
        )
    if invalid_prediction_semantics:
        blockers.append(
            _blocker(
                "invalid_prediction_delta_semantics",
                "The prediction artifact declared nonnegative magnitude semantics but "
                "supplied a negative component-delta magnitude.",
            )
        )
    if not prediction_error_passed:
        blockers.append(
            _blocker(
                "prediction_error_gate_failed",
                "Map-predicted component deltas did not pass the official response "
                "relative-error threshold.",
            )
        )

    holdout_error = prediction_error if prediction_errors else observed_delta_max
    payload = {
        "schema_version": 1,
        "format": RESPONSE_OUTPUT_FORMAT,
        "tool": "experiments/profile_component_sensitivity_official.py",
        "component": component,
        "device": device,
        "promotion_eligible": promotion_passed,
        "evidence_grade": "A" if promotion_passed else "diagnostic_cuda_official_component_response",
        "official_component_response": True,
        "canonical_scorer_path": True,
        "component_response_path": "archive_zip_inflate_sh_upstream_evaluate_py",
        "passed": promotion_passed,
        "gate_spec": dict(gate_spec),
        "gate_results": {
            "finite_values": bool(finite_values),
            "coverage_passed": bool(coverage_passed),
            "zero_repro": bool(zero_repro),
            "zero_repro_error": zero_repro_error,
            "external_baseline_repro": bool(external_baseline_repro),
            "external_baseline_repro_error": external_baseline_repro_error,
            "external_baseline_value": external_baseline_value,
            "signal_present": bool(signal_present),
            "observed_delta_max": observed_delta_max,
            "prediction_error_gate_implemented": not missing_predictions,
            "prediction_error_passed": bool(prediction_error_passed),
            "prediction_error_mode": prediction_error_mode,
            "invalid_prediction_semantics": bool(invalid_prediction_semantics),
            "max_relative_prediction_error": prediction_error,
            "max_abs_prediction_error": prediction_abs_error,
            "promotion_gate_passed": promotion_passed,
        },
        "component_readout": COMPONENT_READOUTS[component],
        "component_units": dict(COMPONENT_UNITS),
        "promotion_blockers": [] if promotion_passed else blockers,
        "sensitivity_source": "official_archive_finite_difference_component_response",
        "perturbation": plan.get("perturbation", {}),
        "baseline": {
            "epsilon": float(baseline.spec.epsilon),
            "archive": dict(baseline.archive),
            "contest_auth_eval_json": dict(baseline.contest_json),
            "values": baseline_values,
        },
        "count": len(points),
        "holdout_error": holdout_error,
        "holdout_error_kind": (
            "max_relative_prediction_error_against_official_archive_response_abs_magnitude"
            if prediction_error_mode == ABSOLUTE_MAGNITUDE_ERROR_MODE
            else "max_relative_prediction_error_against_official_archive_response_signed_delta"
        ),
        "observed_holdout_delta_max": observed_delta_max,
        **dict(coverage),
        "points": points,
    }
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def produce_official_component_response_curves(
    *,
    baseline_archive: Path,
    perturbation_plan: Path,
    output_dir: Path,
    baseline_contest_auth_eval_json: Path | None,
    contest_auth_eval_script: Path,
    inflate_sh: Path,
    upstream_dir: Path,
    video_names_file: Path,
    device: str,
    inflate_timeout: int,
    evaluate_timeout: int,
    max_relative_error: float,
    zero_repro_tolerance: float,
    min_observed_delta: float,
    allow_directional: bool,
    require_passed: bool = False,
    same_run_zero_baseline: bool = False,
) -> dict[str, Any]:
    if device != "cuda":
        raise OfficialComponentResponseError("promotion official component response requires --device cuda")
    baseline_archive = _require_existing_file(baseline_archive.resolve(), label="baseline archive")
    if baseline_contest_auth_eval_json is not None:
        baseline_contest_auth_eval_json = _require_existing_file(
            baseline_contest_auth_eval_json.resolve(),
            label="baseline contest_auth_eval_json",
        )
    perturbation_plan = _require_existing_file(
        perturbation_plan.resolve(),
        label="perturbation plan",
    )
    contest_auth_eval_script = _require_existing_file(
        contest_auth_eval_script.resolve(),
        label="contest_auth_eval script",
    )
    inflate_sh = _require_existing_file(inflate_sh.resolve(), label="inflate.sh")
    upstream_dir = upstream_dir.resolve()
    if not (upstream_dir / "evaluate.py").exists():
        raise OfficialComponentResponseError(f"upstream evaluate.py not found: {upstream_dir}")
    video_names_file = _require_existing_file(video_names_file.resolve(), label="video names file")

    output_dir.mkdir(parents=True, exist_ok=True)
    plan, specs = _load_plan_points(
        perturbation_plan,
        baseline_archive=baseline_archive,
        baseline_contest_auth_eval_json=baseline_contest_auth_eval_json,
    )
    effective_same_run_zero_baseline = bool(same_run_zero_baseline or require_passed)
    if require_passed:
        _preflight_promotion_predictions(specs)
    external_baseline_contest_auth_eval_json = (
        _file_meta(baseline_contest_auth_eval_json)
        if baseline_contest_auth_eval_json is not None
        else None
    )
    if effective_same_run_zero_baseline:
        external_baseline_values = None
        if baseline_contest_auth_eval_json is not None:
            external_baseline_payload = _validate_contest_eval_json(
                baseline_contest_auth_eval_json,
                archive=baseline_archive,
                device=device,
            )
            external_baseline_values = _component_values_from_contest_eval(external_baseline_payload)
        specs = _force_same_run_zero_baseline(specs, baseline_archive=baseline_archive)
    else:
        external_baseline_values = None
    results = evaluate_points(
        specs,
        output_dir=output_dir,
        contest_auth_eval_script=contest_auth_eval_script,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
        video_names_file=video_names_file,
        device=device,
        inflate_timeout=inflate_timeout,
        evaluate_timeout=evaluate_timeout,
    )

    baseline_matches = [
        result for result in results if result.spec.archive.resolve() == baseline_archive.resolve()
    ]
    zero_matches = [result for result in results if abs(result.spec.epsilon) <= SCORE_EPS]
    if not zero_matches:
        raise OfficialComponentResponseError("internal error: no eps=0 baseline point")
    baseline = zero_matches[0]
    nonzero_baseline_matches = [
        result for result in baseline_matches if abs(result.spec.epsilon) > SCORE_EPS
    ]
    if nonzero_baseline_matches:
        raise OfficialComponentResponseError(
            "baseline archive appears at a nonzero epsilon; response baseline is ambiguous"
        )

    gate_spec = {
        **DEFAULT_GATE_SPEC,
        "holdout_error_max": float(max_relative_error),
        "zero_repro_tolerance": float(zero_repro_tolerance),
        "observed_delta_min": float(min_observed_delta),
    }
    coverage, coverage_passed = _coverage_metadata(
        [result.spec.epsilon for result in results],
        allow_directional=allow_directional,
        directional_action=plan.get("directional_action"),
    )
    response_curve_paths: dict[str, str] = {}
    curves: dict[str, Any] = {}
    for component in COMPONENTS:
        curve = _curve_for_component(
            component=component,
            results=results,
            baseline=baseline,
            gate_spec=gate_spec,
            coverage=coverage,
            coverage_passed=coverage_passed,
            plan=plan,
            device=device,
            external_baseline_values=external_baseline_values,
        )
        path = output_dir / f"{component}_official_response_curve.json"
        _write_json(path, curve)
        response_curve_paths[component] = str(path)
        curves[component] = curve

    summary = {
        "schema_version": 1,
        "format": "official_component_response_summary_v1",
        "tool": "experiments/profile_component_sensitivity_official.py",
        "device": device,
        "require_passed_mode": bool(require_passed),
        "same_run_zero_baseline": bool(effective_same_run_zero_baseline),
        "promotion_eligible": all(bool(curves[component]["passed"]) for component in COMPONENTS),
        "baseline_archive": _file_meta(baseline_archive),
        "baseline_contest_auth_eval_json": dict(baseline.contest_json),
        "external_baseline_contest_auth_eval_json": external_baseline_contest_auth_eval_json,
        "perturbation_plan": _file_meta(perturbation_plan),
        "response_curve_paths": response_curve_paths,
        "gate_spec": gate_spec,
        "external_baseline_component_values": (
            dict(external_baseline_values) if external_baseline_values is not None else None
        ),
        "response_coverage": dict(coverage),
        "response_coverage_passed": bool(coverage_passed),
        "points": [
            {
                "epsilon": float(result.spec.epsilon),
                "archive": dict(result.archive),
                "contest_auth_eval_json": dict(result.contest_json),
                "values": dict(result.values),
            }
            for result in results
        ],
    }
    summary_path = output_dir / "official_component_response_summary.json"
    _write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--baseline-contest-auth-eval-json", type=Path, default=None)
    parser.add_argument("--perturbation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contest-auth-eval-script",
        type=Path,
        default=REPO_ROOT / "experiments" / "contest_auth_eval.py",
    )
    parser.add_argument(
        "--inflate-sh",
        type=Path,
        default=REPO_ROOT / "submissions" / "robust_current" / "inflate.sh",
    )
    parser.add_argument("--upstream", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument(
        "--video-names-file",
        type=Path,
        default=REPO_ROOT / "upstream" / "public_test_video_names.txt",
    )
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--evaluate-timeout", type=int, default=1800)
    parser.add_argument("--max-relative-error", type=float, default=0.35)
    parser.add_argument("--zero-repro-tolerance", type=float, default=1e-7)
    parser.add_argument("--min-observed-delta", type=float, default=1e-12)
    parser.add_argument(
        "--allow-directional",
        action="store_true",
        help="Allow one-sided directional response metadata instead of requiring a matched -eps/+eps pair.",
    )
    parser.add_argument(
        "--same-run-zero-baseline",
        action="store_true",
        help=(
            "Run the eps=0 baseline archive in this job even when an external "
            "baseline contest_auth_eval.json is supplied. --require-passed implies this."
        ),
    )
    parser.add_argument(
        "--require-passed",
        action="store_true",
        help="Exit nonzero if any component response curve fails promotion gates.",
    )
    args = parser.parse_args(argv)
    for field in ("max_relative_error", "zero_repro_tolerance", "min_observed_delta"):
        value = getattr(args, field)
        if not math.isfinite(float(value)) or float(value) < 0.0:
            parser.error(f"--{field.replace('_', '-')} must be finite and nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = produce_official_component_response_curves(
            baseline_archive=args.baseline_archive,
            baseline_contest_auth_eval_json=args.baseline_contest_auth_eval_json,
            perturbation_plan=args.perturbation_plan,
            output_dir=args.output_dir,
            contest_auth_eval_script=args.contest_auth_eval_script,
            inflate_sh=args.inflate_sh,
            upstream_dir=args.upstream,
            video_names_file=args.video_names_file,
            device=args.device,
            inflate_timeout=args.inflate_timeout,
            evaluate_timeout=args.evaluate_timeout,
            max_relative_error=args.max_relative_error,
            zero_repro_tolerance=args.zero_repro_tolerance,
            min_observed_delta=args.min_observed_delta,
            allow_directional=args.allow_directional,
            require_passed=args.require_passed,
            same_run_zero_baseline=args.same_run_zero_baseline,
        )
    except OfficialComponentResponseError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    if args.require_passed and summary["promotion_eligible"] is not True:
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit("FATAL: official component response gates did not pass")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
