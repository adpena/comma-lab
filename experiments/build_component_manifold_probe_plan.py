#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a non-promotable component-manifold probe plan from exact eval JSONs.

The output is a planning artifact for local rate-distortion geometry. It treats
each exact archive result as a point near a baseline archive and records score
slopes, curvature estimates, geometry-basin checks, and simple interaction
terms. It never promotes, ranks, kills, or claims a score by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "experiments/build_component_manifold_probe_plan.py"
INPUT_SCHEMA = "component_manifold_probe_input_v1"
OUTPUT_SCHEMA = "component_manifold_probe_plan_v1"
SCORE_DENOMINATOR_BYTES = 37_545_489  # [contest-defined: original video bytes]
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
PLANNER_WARNING = (
    "This artifact is a derivation over exact eval JSONs. It is not a score "
    "claim and cannot promote, rank, kill, or retire a lane without a concrete "
    "archive passing exact CUDA auth eval and adjudication."
)


class ComponentManifoldProbeError(ValueError):
    """Raised when a probe plan cannot be formed rigorously."""


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


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
        raise ComponentManifoldProbeError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ComponentManifoldProbeError(f"{label} must be a JSON object: {path}")
    return payload


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComponentManifoldProbeError(f"{field} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ComponentManifoldProbeError(f"{field} must be finite")
    return out


def _optional_str(value: Any, *, field: str, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ComponentManifoldProbeError(f"{field} is required")
        return None
    if not isinstance(value, str) or not value:
        raise ComponentManifoldProbeError(f"{field} must be a non-empty string")
    return value


def _resolve_path(raw: Any, *, base_dir: Path, field: str) -> Path:
    value = _optional_str(raw, field=field, required=True)
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise ComponentManifoldProbeError(f"{field} does not exist: {path}")
    return path


def _nested_get(payload: Mapping[str, Any], keys: Iterable[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_present(payload: Mapping[str, Any], candidates: Iterable[Any]) -> Any:
    for candidate in candidates:
        if isinstance(candidate, tuple):
            value = _nested_get(payload, candidate)
        else:
            value = payload.get(candidate)
        if value is not None:
            return value
    return None


def _load_exact_eval(
    path: Path,
    *,
    label: str,
    strict_cuda: bool,
    required_samples: int,
) -> dict[str, Any]:
    payload = _load_json_object(path, label=label)
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}

    score = _first_present(
        payload,
        ("score_recomputed_from_components", "final_score", "score"),
    )
    archive_bytes = _first_present(payload, ("archive_size_bytes", "archive_bytes"))
    seg = _first_present(payload, ("avg_segnet_dist", "seg_dist", "segnet_dist"))
    pose = _first_present(payload, ("avg_posenet_dist", "pose_dist", "posenet_dist"))
    n_samples_raw = _first_present(payload, ("n_samples", "sample_count", "num_samples"))
    device = _first_present(payload, ("device", ("provenance", "device")))
    archive_sha = _first_present(
        payload,
        (
            "archive_sha256",
            ("provenance", "archive_sha256"),
            ("provenance", "archive_sha"),
        ),
    )

    score_f = _finite_float(score, field=f"{label}.score")
    archive_bytes_i = int(_finite_float(archive_bytes, field=f"{label}.archive_size_bytes"))
    seg_f = _finite_float(seg, field=f"{label}.avg_segnet_dist")
    pose_f = _finite_float(pose, field=f"{label}.avg_posenet_dist")
    n_samples_i = int(_finite_float(n_samples_raw, field=f"{label}.n_samples"))
    if n_samples_i != required_samples:
        raise ComponentManifoldProbeError(
            f"{label}.n_samples must be {required_samples}, got {n_samples_i}"
        )
    if strict_cuda and device != "cuda":
        raise ComponentManifoldProbeError(f"{label}.provenance.device must be cuda")
    if not isinstance(archive_sha, str) or len(archive_sha) != 64:
        raise ComponentManifoldProbeError(f"{label}.archive_sha256 must be a SHA-256 hex string")

    component_score = 100.0 * seg_f + math.sqrt(10.0 * pose_f)
    rate_score = 25.0 * archive_bytes_i / SCORE_DENOMINATOR_BYTES
    formula_score = component_score + rate_score
    return {
        "file": _file_meta(path),
        "score": score_f,
        "formula_score": formula_score,
        "formula_score_delta_vs_recorded": formula_score - score_f,
        "archive_size_bytes": archive_bytes_i,
        "avg_segnet_dist": seg_f,
        "avg_posenet_dist": pose_f,
        "seg_score_term": 100.0 * seg_f,
        "pose_score_term": math.sqrt(10.0 * pose_f),
        "rate_score_term": rate_score,
        "n_samples": n_samples_i,
        "archive_sha256": archive_sha,
        "device": device,
        "hardware": _first_present(payload, ("hardware", ("provenance", "hardware"))),
        "promotion_eligible": payload.get("promotion_eligible"),
        "component_gate_status": payload.get("component_gate_status"),
    }


def _geometry_basin(
    baseline: Mapping[str, Any],
    point: Mapping[str, Any],
    *,
    max_posenet_relative: float,
    max_segnet_relative: float,
) -> dict[str, Any]:
    if max_posenet_relative <= 0 or max_segnet_relative <= 0:
        raise ComponentManifoldProbeError("geometry basin relative limits must be positive")
    if baseline["avg_posenet_dist"] <= 0 or baseline["avg_segnet_dist"] <= 0:
        raise ComponentManifoldProbeError("baseline component distances must be positive")
    pose_ratio = float(point["avg_posenet_dist"]) / float(baseline["avg_posenet_dist"])
    seg_ratio = float(point["avg_segnet_dist"]) / float(baseline["avg_segnet_dist"])
    violations: list[dict[str, Any]] = []
    if pose_ratio > max_posenet_relative:
        violations.append(
            {
                "component": "posenet",
                "relative_to_baseline": pose_ratio,
                "max_relative": max_posenet_relative,
                "observed": point["avg_posenet_dist"],
                "baseline": baseline["avg_posenet_dist"],
            }
        )
    if seg_ratio > max_segnet_relative:
        violations.append(
            {
                "component": "segnet",
                "relative_to_baseline": seg_ratio,
                "max_relative": max_segnet_relative,
                "observed": point["avg_segnet_dist"],
                "baseline": baseline["avg_segnet_dist"],
            }
        )
    return {
        "passed": not violations,
        "ratios": {
            "posenet_relative": round(pose_ratio, 12),
            "segnet_relative": round(seg_ratio, 12),
        },
        "limits": {
            "max_posenet_relative": max_posenet_relative,
            "max_segnet_relative": max_segnet_relative,
        },
        "violations": violations,
    }


def _point_delta(
    baseline: Mapping[str, Any],
    point: Mapping[str, Any],
    *,
    epsilon: float | None,
) -> dict[str, Any]:
    seg_delta = float(point["seg_score_term"]) - float(baseline["seg_score_term"])
    pose_delta = float(point["pose_score_term"]) - float(baseline["pose_score_term"])
    rate_delta = float(point["rate_score_term"]) - float(baseline["rate_score_term"])
    score_delta = float(point["score"]) - float(baseline["score"])
    formula_delta = seg_delta + pose_delta + rate_delta
    out = {
        "score_delta": score_delta,
        "formula_score_delta": formula_delta,
        "recorded_minus_formula_delta": score_delta - formula_delta,
        "archive_bytes_delta": int(point["archive_size_bytes"]) - int(baseline["archive_size_bytes"]),
        "seg_score_term_delta": seg_delta,
        "pose_score_term_delta": pose_delta,
        "rate_score_term_delta": rate_delta,
        "avg_segnet_dist_delta": float(point["avg_segnet_dist"]) - float(baseline["avg_segnet_dist"]),
        "avg_posenet_dist_delta": float(point["avg_posenet_dist"]) - float(baseline["avg_posenet_dist"]),
    }
    if epsilon is not None and epsilon != 0.0:
        out["score_slope_per_epsilon"] = score_delta / epsilon
        out["formula_score_slope_per_epsilon"] = formula_delta / epsilon
        out["rate_slope_per_epsilon"] = rate_delta / epsilon
        out["component_slope_per_epsilon"] = (seg_delta + pose_delta) / epsilon
    return out


def _parse_point(
    raw: Mapping[str, Any],
    *,
    base_dir: Path,
    index: int,
    baseline: Mapping[str, Any],
    strict_cuda: bool,
    required_samples: int,
    max_posenet_relative: float,
    max_segnet_relative: float,
) -> dict[str, Any]:
    point_id = _optional_str(raw.get("point_id"), field=f"points[{index}].point_id", required=True)
    axis_id = _optional_str(raw.get("axis_id"), field=f"points[{index}].axis_id", required=True)
    family = _optional_str(raw.get("family"), field=f"points[{index}].family", required=True)
    coordinate = _optional_str(raw.get("coordinate"), field=f"points[{index}].coordinate")
    epsilon = None
    if raw.get("epsilon") is not None:
        epsilon = _finite_float(raw.get("epsilon"), field=f"points[{index}].epsilon")
    eval_path = _resolve_path(
        raw.get("contest_auth_eval_json"),
        base_dir=base_dir,
        field=f"points[{index}].contest_auth_eval_json",
    )
    point = _load_exact_eval(
        eval_path,
        label=f"points[{index}]",
        strict_cuda=strict_cuda,
        required_samples=required_samples,
    )
    basin = _geometry_basin(
        baseline,
        point,
        max_posenet_relative=max_posenet_relative,
        max_segnet_relative=max_segnet_relative,
    )
    delta = _point_delta(baseline, point, epsilon=epsilon)
    return {
        "point_id": point_id,
        "family": family,
        "axis_id": axis_id,
        "coordinate": coordinate,
        "epsilon": epsilon,
        "notes": raw.get("notes"),
        "tags": raw.get("tags", []),
        "exact_eval": point,
        "delta_vs_baseline": delta,
        "geometry_basin": basin,
        "continuation_candidate": bool(basin["passed"] and delta["score_delta"] < 0.0),
    }


def _compute_curvature(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_axis: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        epsilon = point.get("epsilon")
        if isinstance(epsilon, (int, float)) and epsilon != 0:
            by_axis[(str(point["family"]), str(point["axis_id"]))].append(point)

    curvature_records: list[dict[str, Any]] = []
    for (family, axis_id), axis_points in sorted(by_axis.items()):
        positives = [p for p in axis_points if float(p["epsilon"]) > 0]
        negatives = [p for p in axis_points if float(p["epsilon"]) < 0]
        for plus in positives:
            plus_eps = float(plus["epsilon"])
            match = min(
                negatives,
                key=lambda item: abs(abs(float(item["epsilon"])) - plus_eps),
                default=None,
            )
            if match is None or not math.isclose(abs(float(match["epsilon"])), plus_eps):
                continue
            minus = match
            denominator = plus_eps - float(minus["epsilon"])
            if denominator == 0:
                continue
            plus_delta = plus["delta_vs_baseline"]
            minus_delta = minus["delta_vs_baseline"]
            curvature_records.append(
                {
                    "family": family,
                    "axis_id": axis_id,
                    "epsilon_abs": plus_eps,
                    "minus_point_id": minus["point_id"],
                    "plus_point_id": plus["point_id"],
                    "central_score_slope": (
                        float(plus_delta["score_delta"]) - float(minus_delta["score_delta"])
                    )
                    / denominator,
                    "central_formula_score_slope": (
                        float(plus_delta["formula_score_delta"])
                        - float(minus_delta["formula_score_delta"])
                    )
                    / denominator,
                    "score_curvature_per_epsilon2": (
                        float(plus_delta["score_delta"]) + float(minus_delta["score_delta"])
                    )
                    / (plus_eps * plus_eps),
                    "formula_score_curvature_per_epsilon2": (
                        float(plus_delta["formula_score_delta"])
                        + float(minus_delta["formula_score_delta"])
                    )
                    / (plus_eps * plus_eps),
                }
            )
    return curvature_records


def _compute_interactions(
    raw_interactions: Any,
    *,
    points_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if raw_interactions is None:
        return []
    if not isinstance(raw_interactions, list):
        raise ComponentManifoldProbeError("interactions must be a list")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_interactions):
        if not isinstance(raw, Mapping):
            raise ComponentManifoldProbeError(f"interactions[{index}] must be an object")
        interaction_id = _optional_str(
            raw.get("interaction_id"),
            field=f"interactions[{index}].interaction_id",
            required=True,
        )
        combined_id = _optional_str(
            raw.get("combined_point_id"),
            field=f"interactions[{index}].combined_point_id",
            required=True,
        )
        point_ids = raw.get("point_ids")
        if not isinstance(point_ids, list) or not point_ids:
            raise ComponentManifoldProbeError(f"interactions[{index}].point_ids must be non-empty")
        if combined_id not in points_by_id:
            raise ComponentManifoldProbeError(f"unknown combined_point_id {combined_id!r}")
        missing = [point_id for point_id in point_ids if point_id not in points_by_id]
        if missing:
            raise ComponentManifoldProbeError(f"unknown interaction point_ids: {missing}")
        combined_delta = points_by_id[combined_id]["delta_vs_baseline"]

        def residual_delta(field: str) -> float:
            return float(combined_delta[field]) - sum(
                float(points_by_id[str(point_id)]["delta_vs_baseline"][field])
                for point_id in point_ids
            )

        score_residual = residual_delta("score_delta")
        records.append(
            {
                "interaction_id": interaction_id,
                "point_ids": point_ids,
                "combined_point_id": combined_id,
                "score_interaction_residual": score_residual,
                "formula_score_interaction_residual": residual_delta("formula_score_delta"),
                "rate_interaction_residual": residual_delta("rate_score_term_delta"),
                "seg_interaction_residual": residual_delta("seg_score_term_delta"),
                "pose_interaction_residual": residual_delta("pose_score_term_delta"),
                "classification": "synergy"
                if score_residual < 0
                else "antagonism"
                if score_residual > 0
                else "additive",
                "note": (
                    "Negative residual means the combined archive scored better than "
                    "the sum of standalone deltas; positive residual means antagonism."
                ),
            }
        )
    return records


def build_component_manifold_probe_plan(
    *,
    input_plan: Path,
    output_json: Path,
    strict_cuda: bool = True,
    required_samples: int = 600,
    max_posenet_relative: float = 1.25,
    max_segnet_relative: float = 1.25,
) -> dict[str, Any]:
    input_plan = input_plan.resolve()
    input_payload = _load_json_object(input_plan, label="input plan")
    if input_payload.get("schema") != INPUT_SCHEMA:
        raise ComponentManifoldProbeError(
            f"{input_plan}: expected schema {INPUT_SCHEMA!r}"
        )
    base_dir = input_plan.parent
    limits = input_payload.get("geometry_basin_limits")
    if isinstance(limits, Mapping):
        max_posenet_relative = _finite_float(
            limits.get("max_posenet_relative", max_posenet_relative),
            field="geometry_basin_limits.max_posenet_relative",
        )
        max_segnet_relative = _finite_float(
            limits.get("max_segnet_relative", max_segnet_relative),
            field="geometry_basin_limits.max_segnet_relative",
        )

    baseline_path = _resolve_path(
        input_payload.get("baseline_contest_json"),
        base_dir=base_dir,
        field="baseline_contest_json",
    )
    baseline = _load_exact_eval(
        baseline_path,
        label="baseline",
        strict_cuda=strict_cuda,
        required_samples=required_samples,
    )
    raw_points = input_payload.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ComponentManifoldProbeError("points must be a non-empty list")
    points = [
        _parse_point(
            raw,
            base_dir=base_dir,
            index=index,
            baseline=baseline,
            strict_cuda=strict_cuda,
            required_samples=required_samples,
            max_posenet_relative=max_posenet_relative,
            max_segnet_relative=max_segnet_relative,
        )
        for index, raw in enumerate(raw_points)
        if isinstance(raw, Mapping)
    ]
    if len(points) != len(raw_points):
        raise ComponentManifoldProbeError("every point must be a JSON object")
    point_ids = [str(point["point_id"]) for point in points]
    if len(set(point_ids)) != len(point_ids):
        raise ComponentManifoldProbeError("point_id values must be unique")
    points_by_id = {str(point["point_id"]): point for point in points}
    curvature = _compute_curvature(points)
    interactions = _compute_interactions(
        input_payload.get("interactions"),
        points_by_id=points_by_id,
    )
    continuation = [
        {
            "point_id": point["point_id"],
            "family": point["family"],
            "axis_id": point["axis_id"],
            "score_delta": point["delta_vs_baseline"]["score_delta"],
            "archive_bytes_delta": point["delta_vs_baseline"]["archive_bytes_delta"],
            "reason": "inside geometry basin and lower exact score than baseline",
        }
        for point in sorted(
            points,
            key=lambda item: float(item["delta_vs_baseline"]["score_delta"]),
        )
        if point["continuation_candidate"]
    ]
    blocked = [
        {
            "point_id": point["point_id"],
            "family": point["family"],
            "axis_id": point["axis_id"],
            "violations": point["geometry_basin"]["violations"],
            "score_delta": point["delta_vs_baseline"]["score_delta"],
        }
        for point in points
        if not point["geometry_basin"]["passed"]
    ]

    payload: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": PRODUCER,
        "evidence_grade": "derivation",
        "input_evidence_required": "A_or_A++_exact_cuda_archive_jsons",
        "score_claim": False,
        "promotion_eligible": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "planner_warning": PLANNER_WARNING,
        "coordinate_system": {
            "name": "contest_component_rate_distortion_manifold_v1",
            "coordinates": [
                "archive_size_bytes",
                "avg_segnet_dist",
                "avg_posenet_dist",
                "score",
                "family",
                "axis_id",
                "epsilon",
            ],
            "continuous_time_view": (
                "dtheta/dt follows projected negative score gradient while "
                "lambda terms enforce rate, SegNet, PoseNet, archive custody, "
                "inflate budget, and reproducibility constraints."
            ),
        },
        "input_plan": _file_meta(input_plan),
        "strict_cuda": strict_cuda,
        "required_samples": required_samples,
        "geometry_basin_limits": {
            "max_posenet_relative": max_posenet_relative,
            "max_segnet_relative": max_segnet_relative,
        },
        "baseline": baseline,
        "points": points,
        "curvature_estimates": curvature,
        "interactions": interactions,
        "continuation_candidates": continuation,
        "blocked_points": blocked,
        "next_dispatch_policy": {
            "allowed": bool(continuation),
            "blocked_family_count": len(blocked),
            "rule": (
                "Only points inside the geometry basin with exact-score improvement "
                "can seed another concrete archive/eval. Collapsed points update "
                "priors but cannot drive water-filling archive policies."
            ),
        },
        "resource_diversity_hooks": [
            "Python planners and trainers",
            "Rust/Zig/C deterministic decoders",
            "arithmetic or ANS/range coding atoms",
            "NeRV/INR/HNeRV latent manifolds",
            "SegMap/Q-FAITHFUL soft-LUT control signals",
            "systems-control and differential-equation lambda updates",
            "Bayesian/bandit or reinforcement learned atom selectors",
        ],
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
        },
    }
    _write_json(output_json, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-posenet-relative", type=float, default=1.25)
    parser.add_argument("--max-segnet-relative", type=float, default=1.25)
    parser.add_argument("--required-samples", type=int, default=600)
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="Allow non-CUDA JSONs for forensic/dev artifacts only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_component_manifold_probe_plan(
        input_plan=args.input_plan,
        output_json=args.output_json,
        strict_cuda=not args.allow_non_cuda,
        required_samples=args.required_samples,
        max_posenet_relative=args.max_posenet_relative,
        max_segnet_relative=args.max_segnet_relative,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
