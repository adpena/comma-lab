#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed receipt adjudicator for the Task #494 ordinal-loss A/B.

This program never trains, calls a scorer, or launches a provider.  It compares
two *existing* real-n600 trajectory receipts for the only permitted treatment:
``ce`` versus ``margin_hinge`` with ``margin_target_end == 0.0``.  Zero is the
DERIVED exact argmax decision boundary, not a fitted margin.

Input schema ``ordinal_perclass_convergence_trajectory.v1``::

  {
    "schema_version": "ordinal_perclass_convergence_trajectory.v1",
    "custody": {
      "authority": {"cohort": "real-n600", "pair_count": 600},
      "seed": "...", "order_sha256": "...", "model_sha256": "...",
      "optimizer_fingerprint": "...", "curriculum_fingerprint": "...",
      "init_ema_sha256": "...", "non_treatment_config_sha256": "...",
      "data_fingerprint": "...", "preregistration_sha256": "..."
    },
    "treatment": {"seg_loss": "ce"},
    "trajectory": [{
      "update": 0, "wall_time_seconds": 0.0,
      "d_seg_by_class": {
        "Road": {"all": 0.2, "hard": 0.3, "easy": 0.1}, ...
      }
    }]
  }

Every row must contain all five canonical classes and all/all-hard/easy d_seg
strata.  ``Sky`` is intentionally forbidden: it is not a separate authority
class in this probe.  Rates are robust Theil--Sen slopes of d_seg decline per
update and per wall-clock second.  A blocked receipt is evidence of missing
custody, never fixture or empirical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

SCHEMA: Final[str] = "ordinal_perclass_convergence_analysis.v1"
INPUT_SCHEMA: Final[str] = "ordinal_perclass_convergence_trajectory.v1"
CLASSES: Final[tuple[str, ...]] = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
RARE_CLASSES: Final[tuple[str, ...]] = ("Lane", "Movable")
COMMON_CLASSES: Final[tuple[str, ...]] = ("Road", "Undrivable", "MyCar")
STRATA: Final[tuple[str, ...]] = ("all", "hard", "easy")
REQUIRED_CUSTODY: Final[tuple[str, ...]] = (
    "seed",
    "order_sha256",
    "model_sha256",
    "optimizer_fingerprint",
    "curriculum_fingerprint",
    "init_ema_sha256",
    "non_treatment_config_sha256",
    "data_fingerprint",
    "preregistration_sha256",
)
SHA256_CUSTODY_FIELDS: Final[tuple[str, ...]] = (
    "order_sha256",
    "model_sha256",
    "init_ema_sha256",
    "non_treatment_config_sha256",
    "data_fingerprint",
    "preregistration_sha256",
)


class ReceiptError(ValueError):
    """A receipt is absent, malformed, or lacks required real-n600 custody."""


def _finite_number(value: object, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ReceiptError(f"{path} must be finite" + (" and >= 0" if nonnegative else ""))
    return result


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{path} must be an object")
    return value


def sha256_bytes(payload: bytes) -> str:
    """Return the immutable source-receipt fingerprint."""
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an output payload before its self-referential hash is attached."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(encoded)


def read_receipt(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReceiptError(f"receipt missing or unreadable: {path}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReceiptError(f"receipt is not JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ReceiptError(f"receipt root must be an object: {path}")
    return payload, sha256_bytes(raw)


def _validate_custody(receipt: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if receipt.get("schema_version") != INPUT_SCHEMA:
        raise ReceiptError(f"{label}.schema_version must equal {INPUT_SCHEMA!r}")
    custody = _mapping(receipt.get("custody"), f"{label}.custody")
    authority = _mapping(custody.get("authority"), f"{label}.custody.authority")
    if authority.get("cohort") != "real-n600" or authority.get("pair_count") != 600:
        raise ReceiptError(f"{label} lacks exact real-n600 / 600-pair authority")
    for field in REQUIRED_CUSTODY:
        value = custody.get(field)
        if not isinstance(value, str) or not value:
            raise ReceiptError(f"{label}.custody.{field} must be a nonempty fingerprint string")
    for field in SHA256_CUSTODY_FIELDS:
        if not _is_sha256(custody.get(field)):
            raise ReceiptError(f"{label}.custody.{field} must be a lowercase SHA-256")
    return custody


def _validate_treatment(receipt: Mapping[str, Any], label: str, expected_loss: str) -> None:
    treatment = _mapping(receipt.get("treatment"), f"{label}.treatment")
    expected_keys = (
        {"seg_loss", "margin_target_end"}
        if expected_loss == "margin_hinge"
        else {"seg_loss"}
    )
    if set(treatment) != expected_keys:
        raise ReceiptError(
            f"{label}.treatment must contain exactly {sorted(expected_keys)!r}; "
            "no confounded treatment knobs"
        )
    if treatment.get("seg_loss") != expected_loss:
        raise ReceiptError(f"{label}.treatment.seg_loss must be {expected_loss!r}")
    if expected_loss == "margin_hinge":
        margin = _finite_number(treatment.get("margin_target_end"), f"{label}.treatment.margin_target_end")
        if margin != 0.0:
            raise ReceiptError("margin_hinge must use margin_target_end == 0.0 (exact argmax boundary)")
    elif "margin_target_end" in treatment:
        raise ReceiptError("CE control must not carry margin_target_end")


def _validate_trajectory(receipt: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    trajectory = receipt.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise ReceiptError(f"{label}.trajectory must contain at least two observations")
    parsed: list[dict[str, Any]] = []
    previous_update = -1
    previous_seconds = -1.0
    for index, raw_row in enumerate(trajectory):
        row = _mapping(raw_row, f"{label}.trajectory[{index}]")
        update_value = row.get("update")
        if isinstance(update_value, bool) or not isinstance(update_value, int) or update_value <= previous_update:
            raise ReceiptError(f"{label}.trajectory[{index}].update must be strictly increasing integer")
        seconds = _finite_number(row.get("wall_time_seconds"), f"{label}.trajectory[{index}].wall_time_seconds", nonnegative=True)
        if seconds <= previous_seconds:
            raise ReceiptError(f"{label}.trajectory[{index}].wall_time_seconds must be strictly increasing")
        classes = _mapping(row.get("d_seg_by_class"), f"{label}.trajectory[{index}].d_seg_by_class")
        if set(classes) != set(CLASSES):
            raise ReceiptError(f"{label}.trajectory[{index}] needs exactly canonical classes {list(CLASSES)!r}; no Sky")
        parsed_classes: dict[str, dict[str, float]] = {}
        for class_name in CLASSES:
            strata = _mapping(classes[class_name], f"{label}.trajectory[{index}].d_seg_by_class.{class_name}")
            if set(strata) != set(STRATA):
                raise ReceiptError(f"{label}.{class_name} needs exactly all/hard/easy d_seg strata")
            parsed_classes[class_name] = {
                stratum: _finite_number(strata[stratum], f"{label}.{class_name}.{stratum}", nonnegative=True)
                for stratum in STRATA
            }
        parsed.append({"update": update_value, "wall_time_seconds": seconds, "d_seg_by_class": parsed_classes})
        previous_update = update_value
        previous_seconds = seconds
    return parsed


def validate_matched_receipts(ce_receipt: Mapping[str, Any], margin_receipt: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate custody, permitted treatment, and matched updates; return clean trajectories."""
    ce_custody = _validate_custody(ce_receipt, "ce")
    margin_custody = _validate_custody(margin_receipt, "margin")
    for field in REQUIRED_CUSTODY:
        if ce_custody[field] != margin_custody[field]:
            raise ReceiptError(f"custody mismatch for {field}")
    _validate_treatment(ce_receipt, "ce", "ce")
    _validate_treatment(margin_receipt, "margin", "margin_hinge")
    ce = _validate_trajectory(ce_receipt, "ce")
    margin = _validate_trajectory(margin_receipt, "margin")
    if [row["update"] for row in ce] != [row["update"] for row in margin]:
        raise ReceiptError("A/B update schedules differ; matched convergence comparison is invalid")
    return ce, margin


def theil_sen_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Robust median slope, refusing duplicate or underspecified abscissae."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ReceiptError("Theil-Sen needs >=2 matched finite observations")
    slopes = [
        (y_values[right] - y_values[left]) / (x_values[right] - x_values[left])
        for left in range(len(x_values) - 1)
        for right in range(left + 1, len(x_values))
        if x_values[right] > x_values[left]
    ]
    if not slopes:
        raise ReceiptError("Theil-Sen needs strictly increasing abscissae")
    return float(statistics.median(slopes))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ReceiptError("cannot average an empty value collection")
    return float(statistics.fmean(values))


def derive_rates(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive d_seg decline rates (positive is convergence) on update and wall-time axes."""
    updates = [float(row["update"]) for row in trajectory]
    seconds = [float(row["wall_time_seconds"]) for row in trajectory]
    per_class: dict[str, Any] = {}
    for class_name in CLASSES:
        metrics: dict[str, Any] = {}
        for stratum in STRATA:
            d_seg = [float(row["d_seg_by_class"][class_name][stratum]) for row in trajectory]
            metrics[stratum] = {
                "start_d_seg": d_seg[0],
                "end_d_seg": d_seg[-1],
                "theil_sen_decline_per_update": -theil_sen_slope(updates, d_seg),
                "theil_sen_decline_per_wall_second": -theil_sen_slope(seconds, d_seg),
            }
        per_class[class_name] = metrics
    return {"per_class": per_class, "n_observations": len(trajectory)}


def _rate(rates: Mapping[str, Any], class_name: str, stratum: str, axis: str) -> float:
    return float(rates["per_class"][class_name][stratum][axis])


def _rare_common_gap(rates: Mapping[str, Any], axis: str) -> dict[str, float]:
    rare = _mean(_rate(rates, item, "all", axis) for item in RARE_CLASSES)
    common = _mean(_rate(rates, item, "all", axis) for item in COMMON_CLASSES)
    return {"rare": rare, "common": common, "common_minus_rare": common - rare}


def compare_rates(ce_rates: Mapping[str, Any], margin_rates: Mapping[str, Any]) -> dict[str, Any]:
    """Compare rare/common convergence and classify the preregistered instance threshold."""
    axis = "theil_sen_decline_per_update"
    rate_delta: dict[str, dict[str, float]] = {
        class_name: {
            stratum: _rate(margin_rates, class_name, stratum, axis) - _rate(ce_rates, class_name, stratum, axis)
            for stratum in STRATA
        }
        for class_name in CLASSES
    }
    ce_gap = _rare_common_gap(ce_rates, axis)["common_minus_rare"]
    margin_gap = _rare_common_gap(margin_rates, axis)["common_minus_rare"]
    gap_closure_percent = None if ce_gap <= 0.0 else 100.0 * (ce_gap - margin_gap) / ce_gap
    rare_hard_delta = _mean(rate_delta[item]["hard"] for item in RARE_CLASSES)
    rare_easy_delta = _mean(rate_delta[item]["easy"] for item in RARE_CLASSES)
    global_delta = _mean(rate_delta[item]["all"] for item in CLASSES)
    common_regression = {item: rate_delta[item]["all"] for item in COMMON_CLASSES if rate_delta[item]["all"] < 0.0}
    any_harm = any(delta < 0.0 for by_stratum in rate_delta.values() for delta in by_stratum.values())
    if any_harm or global_delta < 0.0 or common_regression:
        verdict = "TRADEOFF"
        reason = "one or more class/stratum rates regressed"
    elif gap_closure_percent is None:
        verdict = "OWED_BASELINE_IMBALANCE"
        reason = "CE rare/common gap was non-positive, so gap closure is undefined"
    elif gap_closure_percent >= 50.0 and rare_hard_delta > 0.0 and rare_easy_delta > 0.0:
        verdict = "DOMINANT"
        reason = ">=50% rare/common gap closure, rare hard/easy improvement, no global/common regression"
    elif gap_closure_percent >= 10.0:
        verdict = "CONTRIBUTORY"
        reason = "10-<50% gap closure, or >=50% without all dominance conditions"
    else:
        verdict = "INERT"
        reason = "<10% rare/common gap closure"
    return {
        "rate_axis_for_instance_verdict": axis,
        "rare_classes": list(RARE_CLASSES),
        "common_classes": list(COMMON_CLASSES),
        "rare_common_gap": {"ce": ce_gap, "margin_hinge": margin_gap, "closure_percent": gap_closure_percent},
        "rare_common_gap_by_axis": {
            rate_axis: {
                "ce": _rare_common_gap(ce_rates, rate_axis),
                "margin_hinge": _rare_common_gap(margin_rates, rate_axis),
            }
            for rate_axis in ("theil_sen_decline_per_update", "theil_sen_decline_per_wall_second")
        },
        "rare_hard_rate_delta": rare_hard_delta,
        "rare_easy_rate_delta": rare_easy_delta,
        "global_all_rate_delta": global_delta,
        "common_all_regressions": common_regression,
        "per_class_stratum_rate_delta": rate_delta,
        "instance_verdict": verdict,
        "reason": reason,
    }


def _blocked_output(*, ce_sha256: str | None, margin_sha256: str | None, error: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution": "ZERO_LAUNCH_RECEIPT_ANALYZER",
        "verdict_scope": "INSTANCE",
        "evidence_status": "BLOCKED_NO_EMPIRICAL_CLAIM",
        "owed_status": "OWED",
        "launch_authorized": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "blocker": error,
        "source_receipt_sha256": {"ce": ce_sha256, "margin_hinge": margin_sha256},
    }
    output["content_address_sha256"] = canonical_sha256(output)
    return output


def analyze_receipts(ce_path: Path, margin_path: Path) -> tuple[dict[str, Any], bool]:
    """Return a content-addressed result and whether both arms passed custody validation."""
    ce_sha256: str | None = None
    margin_sha256: str | None = None
    try:
        ce_receipt, ce_sha256 = read_receipt(ce_path)
        margin_receipt, margin_sha256 = read_receipt(margin_path)
        ce_trajectory, margin_trajectory = validate_matched_receipts(ce_receipt, margin_receipt)
        ce_rates = derive_rates(ce_trajectory)
        margin_rates = derive_rates(margin_trajectory)
        comparison = compare_rates(ce_rates, margin_rates)
    except ReceiptError as exc:
        return _blocked_output(ce_sha256=ce_sha256, margin_sha256=margin_sha256, error=str(exc)), False
    output = {
        "schema_version": SCHEMA,
        "execution": "ZERO_LAUNCH_RECEIPT_ANALYZER",
        "verdict_scope": "INSTANCE",
        "evidence_status": "MEASURED_FROM_SUPPLIED_REAL_N600_RECEIPTS",
        "owed_status": "CLOSED",
        "launch_authorized": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "treatment": {
            "control": {"seg_loss": "ce"},
            "treatment": {"seg_loss": "margin_hinge", "margin_target_end": 0.0},
            "margin_target_end_provenance": "DERIVED exact argmax boundary",
        },
        "canonical_classes": list(CLASSES),
        "source_receipt_sha256": {"ce": ce_sha256, "margin_hinge": margin_sha256},
        "rates": {"ce": ce_rates, "margin_hinge": margin_rates},
        "comparison": comparison,
    }
    output["content_address_sha256"] = canonical_sha256(output)
    return output, True


def write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ce-receipt", type=Path, required=True, help="existing CE real-n600 trajectory receipt")
    parser.add_argument("--margin-receipt", type=Path, required=True, help="existing margin_hinge real-n600 trajectory receipt")
    parser.add_argument("--output", type=Path, required=True, help="analysis JSON to write atomically")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output, admitted = analyze_receipts(args.ce_receipt, args.margin_receipt)
    write_json_atomically(args.output, output)
    if not admitted:
        print(f"OWED: {output['blocker']}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "content_address_sha256": output["content_address_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
