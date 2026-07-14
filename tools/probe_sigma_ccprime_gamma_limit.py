#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Static metric-closure and optional real-n600 sigma A/B receipt analyzer.

This is a zero-launch analyzer.  ``sigma_ij`` from ``length_sigma`` is a scalar,
spatially isotropic Euclidean interface density: it is *not* a Wulff/Finsler
orientation law.  A metric sigma is a necessary static admissibility condition
for the usual multiphase pairwise-perimeter lower-semicontinuous relaxation; a
metric result here is never a proof of a Gamma-limit for this trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.boundary_math.length_sigma import (
    FITTED_20260707_MATRIX,
    FRAGILITY_20260709_MATRIX,
    PRESET_ALL_ONES,
    PRESET_FITTED_20260707,
    PRESET_FRAGILITY_20260709,
)

SCHEMA: Final[str] = "sigma_ccprime_gamma_limit_analysis.v1"
TRAJECTORY_SCHEMA: Final[str] = "sigma_ccprime_gamma_limit_trajectory.v1"
PREREGISTRATION_SCHEMA: Final[str] = "sigma_ccprime_gamma_limit_preregistration.v1"
CLASSES: Final[tuple[str, ...]] = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
STRATA: Final[tuple[str, ...]] = ("all", "hard", "easy")
RARE: Final[tuple[str, ...]] = ("Lane", "Movable")
COMMON: Final[tuple[str, ...]] = ("Road", "Undrivable", "MyCar")
REQUIRED_CUSTODY: Final[tuple[str, ...]] = (
    "seed", "order_sha256", "model_sha256", "optimizer_fingerprint",
    "curriculum_fingerprint", "init_ema_sha256", "non_treatment_config_sha256",
)


class ReceiptError(ValueError):
    """A supplied empirical receipt is absent, malformed, or lacks custody."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def preregistration_content_address(payload: Mapping[str, Any]) -> str:
    """Address a preregistration without its self-referential address field."""
    unsigned = {key: value for key, value in payload.items() if key != "content_address_sha256"}
    return canonical_sha256(unsigned)


def _finite(value: object, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ReceiptError(f"{path} must be finite" + (" and >= 0" if nonnegative else ""))
    return result


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{path} must be an object")
    return value


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReceiptError(f"{path} must be a lowercase SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ReceiptError(f"{path} must be a lowercase SHA-256 hex string") from exc
    if value != value.lower():
        raise ReceiptError(f"{path} must be a lowercase SHA-256 hex string")
    return value


def matrix_for(spec: str) -> np.ndarray:
    if spec == PRESET_ALL_ONES:
        return np.ones((5, 5), dtype=np.float64)
    if spec == PRESET_FITTED_20260707:
        return np.asarray(FITTED_20260707_MATRIX, dtype=np.float64)
    if spec == PRESET_FRAGILITY_20260709:
        return np.asarray(FRAGILITY_20260709_MATRIX, dtype=np.float64)
    raise ValueError(f"unsupported sigma preset {spec!r}")


def triangle_violations(matrix: np.ndarray, *, tolerance: float = 1e-12) -> list[dict[str, Any]]:
    """Enumerate every directed triangle inequality failure sigma_ij <= sigma_ik+sigma_kj."""
    violations: list[dict[str, Any]] = []
    for i in range(len(CLASSES)):
        for j in range(i + 1, len(CLASSES)):
            for k in range(len(CLASSES)):
                if k in (i, j):
                    continue
                direct, via = float(matrix[i, j]), float(matrix[i, k] + matrix[k, j])
                if direct > via + tolerance:
                    violations.append({
                        "direct_pair": [CLASSES[i], CLASSES[j]], "via_class": CLASSES[k],
                        "direct_sigma": direct, "via_sigma_sum": via, "excess": direct - via,
                    })
    return violations


def floyd_warshall_metric_closure(matrix: np.ndarray) -> np.ndarray:
    """Shortest-path metric relaxation, using zero diagonal (not sigma_ii's sentinel one)."""
    distance = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(distance, 0.0)
    for k in range(distance.shape[0]):
        distance = np.minimum(distance, distance[:, [k]] + distance[[k], :])
    return distance


def static_matrix_receipt(spec: str) -> dict[str, Any]:
    matrix = matrix_for(spec)
    closure = floyd_warshall_metric_closure(matrix)
    violations = triangle_violations(matrix)
    return {
        "sigma_spec": spec,
        "class_order": list(CLASSES),
        "surface_density_geometry": "scalar_pairwise_spatially_isotropic_scaled_euclidean",
        "orientation_anisotropy_status": "ABSENT_FROM_SCALAR_SIGMA_ALONE",
        "matrix": matrix.tolist(),
        "matrix_sha256": sha256_bytes(matrix.tobytes()),
        "triangle_violations": violations,
        "metric_admissibility": "BLOCKED_TRIANGLE_VIOLATION" if violations else "METRIC_ADMISSIBLE_STATIC",
        "metric_closure": closure.tolist(),
        "metric_closure_sha256": sha256_bytes(closure.tobytes()),
        "relaxation": "shortest_path_metric_closure",
        "gamma_limit_status": "NOT_PROVEN_BY_THIS_STATIC_ANALYZER",
        "gamma_limit_note": "metric admissibility is necessary here, not a trainer-specific Gamma-limit proof",
        "launch_authorized": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def _read(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"receipt missing, unreadable, or non-JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ReceiptError(f"receipt root must be an object: {path}")
    return data, sha256_bytes(raw)


def _validate_preregistration(
    receipt: Mapping[str, Any], label: str, expected_sigma: str
) -> Mapping[str, Any]:
    preregistration = _object(receipt.get("preregistration"), f"{label}.preregistration")
    if preregistration.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise ReceiptError(f"{label}.preregistration.schema_version must equal {PREREGISTRATION_SCHEMA!r}")
    experiment_id = preregistration.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ReceiptError(f"{label}.preregistration.experiment_id must be a nonempty immutable id")
    supplied = _sha256(
        preregistration.get("content_address_sha256"),
        f"{label}.preregistration.content_address_sha256",
    )
    if supplied != preregistration_content_address(preregistration):
        raise ReceiptError(f"{label}.preregistration self-address does not match declared contents")
    declared_diff = _object(
        preregistration.get("declared_treatment_only_diff"),
        f"{label}.preregistration.declared_treatment_only_diff",
    )
    if set(declared_diff) != {"changed_paths", "control", "treatment"}:
        raise ReceiptError("preregistration must declare exactly the treatment-only diff")
    if declared_diff["changed_paths"] != ["treatment.length_sigma_matrix"]:
        raise ReceiptError("preregistration must declare only treatment.length_sigma_matrix as changed")
    control = _object(declared_diff["control"], "preregistration.control")
    treatment = _object(declared_diff["treatment"], "preregistration.treatment")
    if dict(control) != {"length_sigma_matrix": PRESET_ALL_ONES}:
        raise ReceiptError("preregistration control must be exactly all-ones length sigma")
    if dict(treatment) != {"length_sigma_matrix": expected_sigma}:
        raise ReceiptError("preregistration treatment must exactly match selected sigma preset")
    return preregistration


def _validate(
    receipt: Mapping[str, Any], label: str, expected_sigma: str, declared_treatment_sigma: str
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    if receipt.get("schema_version") != TRAJECTORY_SCHEMA:
        raise ReceiptError(f"{label}.schema_version must equal {TRAJECTORY_SCHEMA!r}")
    custody = _object(receipt.get("custody"), f"{label}.custody")
    authority = _object(custody.get("authority"), f"{label}.custody.authority")
    if authority.get("cohort") != "real-n600" or authority.get("pair_count") != 600:
        raise ReceiptError(f"{label} lacks exact real-n600 / 600-pair authority")
    for key in REQUIRED_CUSTODY:
        if not isinstance(custody.get(key), str) or not custody[key]:
            raise ReceiptError(f"{label}.custody.{key} must be a nonempty fingerprint string")
    _sha256(custody.get("data_fingerprint_sha256"), f"{label}.custody.data_fingerprint_sha256")
    preregistration = _validate_preregistration(receipt, label, declared_treatment_sigma)
    treatment = _object(receipt.get("treatment"), f"{label}.treatment")
    if dict(treatment) != {"length_sigma_matrix": expected_sigma}:
        raise ReceiptError(f"{label}.treatment.length_sigma_matrix must equal {expected_sigma!r}")
    rows = receipt.get("trajectory")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ReceiptError(f"{label}.trajectory must contain >=2 observations")
    clean: list[dict[str, Any]] = []
    last_update, last_time = -1, -1.0
    for row_index, raw in enumerate(rows):
        row = _object(raw, f"{label}.trajectory[{row_index}]")
        update = row.get("update")
        if isinstance(update, bool) or not isinstance(update, int) or update <= last_update:
            raise ReceiptError(f"{label}.trajectory[{row_index}].update must be strictly increasing integer")
        seconds = _finite(row.get("wall_time_seconds"), f"{label}.trajectory[{row_index}].wall_time_seconds", nonnegative=True)
        if seconds <= last_time:
            raise ReceiptError(f"{label}.trajectory[{row_index}].wall_time_seconds must be strictly increasing")
        classes = _object(row.get("d_seg_by_class"), f"{label}.trajectory[{row_index}].d_seg_by_class")
        if set(classes) != set(CLASSES):
            raise ReceiptError(f"{label} needs exactly canonical classes {list(CLASSES)!r}; no Sky")
        values: dict[str, dict[str, float]] = {}
        for class_name in CLASSES:
            strata = _object(classes[class_name], f"{label}.{class_name}")
            if set(strata) != set(STRATA):
                raise ReceiptError(f"{label}.{class_name} needs exactly all/hard/easy d_seg strata")
            values[class_name] = {key: _finite(strata[key], f"{label}.{class_name}.{key}", nonnegative=True) for key in STRATA}
        clean.append({"update": update, "wall_time_seconds": seconds, "d_seg_by_class": values})
        last_update, last_time = update, seconds
    return clean, custody, preregistration


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    slopes = [(y[b] - y[a]) / (x[b] - x[a]) for a in range(len(x) - 1) for b in range(a + 1, len(x))]
    return -float(statistics.median(slopes))


def _rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    updates = [float(row["update"]) for row in rows]
    seconds = [float(row["wall_time_seconds"]) for row in rows]
    per_class: dict[str, Any] = {}
    for name in CLASSES:
        per_class[name] = {}
        for stratum in STRATA:
            values = [float(row["d_seg_by_class"][name][stratum]) for row in rows]
            per_class[name][stratum] = {
                "start_d_seg": values[0], "end_d_seg": values[-1],
                "theil_sen_decline_per_update": _slope(updates, values),
                "theil_sen_decline_per_wall_second": _slope(seconds, values),
            }
    return {"n_observations": len(rows), "per_class": per_class}


def _mean(items: Sequence[float]) -> float:
    return float(statistics.fmean(items))


def compare_rates(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    axis = "theil_sen_decline_per_update"
    delta = {name: {s: float(treatment["per_class"][name][s][axis]) - float(control["per_class"][name][s][axis]) for s in STRATA} for name in CLASSES}
    def gap(rates: Mapping[str, Any]) -> float:
        return _mean([float(rates["per_class"][n]["all"][axis]) for n in COMMON]) - _mean([float(rates["per_class"][n]["all"][axis]) for n in RARE])
    control_gap, treatment_gap = gap(control), gap(treatment)
    closure = None if control_gap <= 0 else 100.0 * (control_gap - treatment_gap) / control_gap
    any_harm = any(value < 0.0 for by_stratum in delta.values() for value in by_stratum.values())
    global_delta = _mean([delta[name]["all"] for name in CLASSES])
    verdict = "TRADEOFF" if any_harm or global_delta < 0 else "CONVERGENCE_GAP_REDUCED" if closure is not None and closure > 0 else "INERT_OR_GAP_NOT_REDUCED"
    return {"rate_axis_for_instance_verdict": axis, "rare_classes": list(RARE), "common_classes": list(COMMON), "rare_common_gap": {"all_ones": control_gap, "sigma": treatment_gap, "closure_percent": closure}, "global_all_rate_delta": global_delta, "per_class_stratum_rate_delta": delta, "instance_verdict": verdict}


def _output(static: dict[str, Any], *, status: str, owed: str, blocker: str | None = None, empirical: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": SCHEMA, "execution": "ZERO_LAUNCH_STATIC_AND_RECEIPT_ANALYZER", "verdict_scope": "INSTANCE", "static_sigma_analysis": static, "evidence_status": status, "owed_status": owed, "gamma_limit_status": "NOT_PROVEN_BY_THIS_ANALYZER", "launch_authorized": False, "score_claim": False, "promotion_eligible": False, "pointer_moved": False}
    if blocker:
        result["blocker"] = blocker
    if empirical:
        result["empirical_ab"] = empirical
    result["content_address_sha256"] = canonical_sha256(result)
    return result


def analyze(spec: str, all_ones_path: Path | None = None, sigma_path: Path | None = None) -> tuple[dict[str, Any], bool]:
    static = static_matrix_receipt(spec)
    if all_ones_path is None and sigma_path is None:
        return _output(static, status="DERIVED_STATIC_ONLY_NO_EMPIRICAL_CLAIM", owed="OWED"), True
    if all_ones_path is None or sigma_path is None:
        return _output(static, status="BLOCKED_NO_EMPIRICAL_CLAIM", owed="OWED", blocker="both all-ones and sigma real-n600 receipts are required"), False
    if spec == PRESET_ALL_ONES:
        return _output(static, status="BLOCKED_NO_EMPIRICAL_CLAIM", owed="OWED", blocker="empirical A/B requires a sigma treatment distinct from all-ones control"), False
    control_sha = sigma_sha = None
    try:
        control, control_sha = _read(all_ones_path)
        treatment, sigma_sha = _read(sigma_path)
        control_rows, control_custody, control_preregistration = _validate(
            control, "all_ones", PRESET_ALL_ONES, spec
        )
        treatment_rows, treatment_custody, treatment_preregistration = _validate(
            treatment, "sigma", spec, spec
        )
        for key in REQUIRED_CUSTODY:
            if control_custody[key] != treatment_custody[key]:
                raise ReceiptError(f"custody mismatch for {key}")
        if control_custody["data_fingerprint_sha256"] != treatment_custody["data_fingerprint_sha256"]:
            raise ReceiptError("data fingerprint mismatch")
        if control_preregistration["experiment_id"] != treatment_preregistration["experiment_id"]:
            raise ReceiptError("preregistration experiment_id mismatch")
        if control_preregistration["content_address_sha256"] != treatment_preregistration["content_address_sha256"]:
            raise ReceiptError("preregistration self-address mismatch")
        if [row["update"] for row in control_rows] != [row["update"] for row in treatment_rows]:
            raise ReceiptError("A/B update schedules differ; matched convergence comparison is invalid")
        control_rates, sigma_rates = _rates(control_rows), _rates(treatment_rows)
    except ReceiptError as exc:
        return _output(static, status="BLOCKED_NO_EMPIRICAL_CLAIM", owed="OWED", blocker=str(exc), empirical={"source_receipt_sha256": {"all_ones": control_sha, "sigma": sigma_sha}}), False
    return _output(static, status="MEASURED_FROM_SUPPLIED_REAL_N600_RECEIPTS", owed="CLOSED", empirical={"source_receipt_sha256": {"all_ones": control_sha, "sigma": sigma_sha}, "data_fingerprint_sha256": control_custody["data_fingerprint_sha256"], "preregistration_content_address_sha256": control_preregistration["content_address_sha256"], "rates": {"all_ones": control_rates, "sigma": sigma_rates}, "comparison": compare_rates(control_rates, sigma_rates)}), True


def write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-spec", choices=(PRESET_ALL_ONES, PRESET_FITTED_20260707, PRESET_FRAGILITY_20260709), default=PRESET_FITTED_20260707)
    parser.add_argument("--all-ones-receipt", type=Path)
    parser.add_argument("--sigma-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output, admitted = analyze(args.sigma_spec, args.all_ones_receipt, args.sigma_receipt)
    write_json_atomically(args.output, output)
    if not admitted:
        print(f"OWED: {output['blocker']}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "content_address_sha256": output["content_address_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
