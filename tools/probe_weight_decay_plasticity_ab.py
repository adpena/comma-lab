#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Zero-launch adjudicator for a preregistered real-n600 weight-decay A/B.

This tool never imports a trainer, scorer, or provider.  It consumes one
immutable preregistration and two existing arm receipts.  The external LM
plasticity result is a SPECULATIVE transfer hypothesis only; this output makes
an INSTANCE-scoped, correlational description of supplied witness receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

SCHEMA: Final[str] = "weight_decay_plasticity_ab_analysis.v1"
PREREGISTRATION_SCHEMA: Final[str] = "weight_decay_plasticity_preregistration.v1"
INPUT_SCHEMA: Final[str] = "weight_decay_plasticity_trajectory.v1"
CLASSES: Final[tuple[str, ...]] = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
RARE_CLASSES: Final[tuple[str, ...]] = ("Lane", "Movable")
MATCHED_CUSTODY: Final[tuple[str, ...]] = (
    "seed",
    "pair_order_sha256",
    "model_definition_sha256",
    "init_ema_sha256",
    "optimizer_non_weight_decay_fingerprint",
    "curriculum_fingerprint",
    "data_fingerprint",
    "non_weight_decay_config_sha256",
)
SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
RANK_KINDS: Final[frozenset[str]] = frozenset(("effective_rank", "pseudo_rank"))
LAUNCH_BLOCKERS: Final[tuple[str, ...]] = (
    "typed_weight_decay_dsl_lever_missing",
    "weight_decay_resume_guard_missing",
)


class ReceiptError(ValueError):
    """An input lacks the custody necessary for a real A/B conclusion."""


def sha256_bytes(payload: bytes) -> str:
    """Return the content address for exact source bytes."""
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON payload before its self-referential address is appended."""
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{path} must be an object")
    return value


def _finite(value: object, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ReceiptError(f"{path} must be finite" + (" and >= 0" if nonnegative else ""))
    return number


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"{path} must be a nonempty string")
    return value


def _sha256(value: object, path: str) -> str:
    text = _nonempty(value, path)
    if not SHA256_RE.fullmatch(text):
        raise ReceiptError(f"{path} must be a lowercase SHA-256")
    return text


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReceiptError(f"input missing or unreadable: {path}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReceiptError(f"input is not JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ReceiptError(f"input root must be an object: {path}")
    return payload, sha256_bytes(raw)


def _validate_preregistration(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    if preregistration.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise ReceiptError(f"preregistration.schema_version must equal {PREREGISTRATION_SCHEMA!r}")
    if preregistration.get("immutable") is not True:
        raise ReceiptError("preregistration.immutable must be true")
    if preregistration.get("only_variable") != "weight_decay":
        raise ReceiptError("preregistration.only_variable must equal 'weight_decay'")
    expected_address = preregistration.get("content_address_sha256")
    without_address = dict(preregistration)
    without_address.pop("content_address_sha256", None)
    if expected_address != canonical_sha256(without_address):
        raise ReceiptError("preregistration.content_address_sha256 does not address its pinned content")

    values: dict[str, float] = {}
    for arm in ("control", "treatment"):
        arm_data = _mapping(preregistration.get(arm), f"preregistration.{arm}")
        if arm_data.get("arm_id") != arm:
            raise ReceiptError(f"preregistration.{arm}.arm_id must equal {arm!r}")
        values[arm] = _finite(arm_data.get("weight_decay"), f"preregistration.{arm}.weight_decay", nonnegative=True)
        _nonempty(arm_data.get("weight_decay_provenance"), f"preregistration.{arm}.weight_decay_provenance")
    if values["control"] == values["treatment"]:
        raise ReceiptError("preregistration must pin distinct control and treatment weight_decay values")
    return values


def _validate_custody(receipt: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if receipt.get("schema_version") != INPUT_SCHEMA:
        raise ReceiptError(f"{label}.schema_version must equal {INPUT_SCHEMA!r}")
    custody = _mapping(receipt.get("custody"), f"{label}.custody")
    authority = _mapping(custody.get("authority"), f"{label}.custody.authority")
    if authority.get("cohort") != "real-n600" or authority.get("pair_count") != 600:
        raise ReceiptError(f"{label} lacks exact real-n600 / 600-pair authority")
    for field in MATCHED_CUSTODY:
        validator = _sha256 if field.endswith("_sha256") else _nonempty
        validator(custody.get(field), f"{label}.custody.{field}")
    return custody


def _validate_arm(
    receipt: Mapping[str, Any],
    label: str,
    *,
    preregistration_sha256: str,
    expected_weight_decay: float,
) -> list[dict[str, Any]]:
    if receipt.get("arm_id") != label:
        raise ReceiptError(f"{label}.arm_id must equal {label!r}")
    if receipt.get("preregistration_sha256") != preregistration_sha256:
        raise ReceiptError(f"{label}.preregistration_sha256 must address the supplied immutable preregistration")
    changed_fields = receipt.get("declared_changed_fields")
    if changed_fields != ["weight_decay"]:
        raise ReceiptError(f"{label}.declared_changed_fields must be exactly ['weight_decay']")
    treatment = _mapping(receipt.get("treatment"), f"{label}.treatment")
    if set(treatment) != {"weight_decay"}:
        raise ReceiptError(f"{label}.treatment may contain only weight_decay")
    if _finite(treatment["weight_decay"], f"{label}.treatment.weight_decay", nonnegative=True) != expected_weight_decay:
        raise ReceiptError(f"{label}.treatment.weight_decay differs from preregistration")
    return _validate_trajectory(receipt, label)


def _validate_trajectory(receipt: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    trajectory = receipt.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise ReceiptError(f"{label}.trajectory must contain at least two observations")
    parsed: list[dict[str, Any]] = []
    previous_update = -1
    previous_seconds = -1.0
    rank_kind: str | None = None
    archive_bytes_by_sha256: dict[str, int] = {}
    for index, raw_row in enumerate(trajectory):
        row = _mapping(raw_row, f"{label}.trajectory[{index}]")
        update = row.get("update")
        if isinstance(update, bool) or not isinstance(update, int) or update <= previous_update:
            raise ReceiptError(f"{label}.trajectory[{index}].update must be a strictly increasing integer")
        seconds = _finite(row.get("wall_time_seconds"), f"{label}.trajectory[{index}].wall_time_seconds", nonnegative=True)
        if seconds <= previous_seconds:
            raise ReceiptError(f"{label}.trajectory[{index}].wall_time_seconds must be strictly increasing")
        classes = _mapping(row.get("d_seg_by_class"), f"{label}.trajectory[{index}].d_seg_by_class")
        if set(classes) != set(CLASSES):
            raise ReceiptError(f"{label}.trajectory[{index}] needs exactly canonical classes {list(CLASSES)!r}")
        d_seg_by_class = {name: _finite(classes[name], f"{label}.{name}.d_seg", nonnegative=True) for name in CLASSES}
        rank = _mapping(row.get("trunk_weight_rank"), f"{label}.trajectory[{index}].trunk_weight_rank")
        if set(rank) != {"kind", "value", "parameter_scope"}:
            raise ReceiptError(f"{label}.trunk_weight_rank must have kind/value/parameter_scope only")
        if rank.get("kind") not in RANK_KINDS or rank.get("parameter_scope") != "trunk_weights":
            raise ReceiptError("rank must be effective_rank or pseudo_rank over trunk_weights, never code rank")
        current_rank_kind = str(rank["kind"])
        if rank_kind is None:
            rank_kind = current_rank_kind
        elif current_rank_kind != rank_kind:
            raise ReceiptError(f"{label}.trunk_weight_rank.kind must remain {rank_kind!r} throughout an arm")
        archive_bytes = row.get("archive_bytes")
        if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes < 0:
            raise ReceiptError(f"{label}.trajectory[{index}].archive_bytes must be a nonnegative integer")
        archive_sha256 = _sha256(row.get("archive_sha256"), f"{label}.trajectory[{index}].archive_sha256")
        prior_bytes = archive_bytes_by_sha256.setdefault(archive_sha256, archive_bytes)
        if prior_bytes != archive_bytes:
            raise ReceiptError("one archive_sha256 cannot be associated with multiple archive byte counts")
        rate = _finite(row.get("archive_rate_bytes_per_pair"), f"{label}.trajectory[{index}].archive_rate_bytes_per_pair", nonnegative=True)
        if not math.isclose(rate, archive_bytes / 600.0, rel_tol=0.0, abs_tol=1e-12):
            raise ReceiptError("archive_rate_bytes_per_pair must equal archive_bytes / 600")
        parsed.append(
            {
                "update": update,
                "wall_time_seconds": seconds,
                "d_seg_by_class": d_seg_by_class,
                "overall_d_seg": _finite(row.get("overall_d_seg"), f"{label}.trajectory[{index}].overall_d_seg", nonnegative=True),
                "trunk_weight_rank": {"kind": current_rank_kind, "value": _finite(rank["value"], f"{label}.trunk_weight_rank.value", nonnegative=True)},
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_sha256,
                "archive_rate_bytes_per_pair": rate,
            }
        )
        previous_update, previous_seconds = update, seconds
    return parsed


def validate_matched_receipts(
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate a genuine one-variable A/B and return parsed arm trajectories."""
    weights = _validate_preregistration(preregistration)
    control_custody, treatment_custody = _validate_custody(control, "control"), _validate_custody(treatment, "treatment")
    for field in MATCHED_CUSTODY:
        if control_custody[field] != treatment_custody[field]:
            raise ReceiptError(f"custody mismatch for {field}; arms differ by more than weight_decay")
    control_rows = _validate_arm(control, "control", preregistration_sha256=preregistration_sha256, expected_weight_decay=weights["control"])
    treatment_rows = _validate_arm(treatment, "treatment", preregistration_sha256=preregistration_sha256, expected_weight_decay=weights["treatment"])
    if [row["update"] for row in control_rows] != [row["update"] for row in treatment_rows]:
        raise ReceiptError("A/B update schedules differ; matched convergence comparison is invalid")
    if control_rows[0]["trunk_weight_rank"]["kind"] != treatment_rows[0]["trunk_weight_rank"]["kind"]:
        raise ReceiptError("A/B trunk-weight rank kinds differ; rank tradeoff is not comparable")
    return control_rows, treatment_rows


def theil_sen_slope(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Return a robust median slope over a strictly increasing independent axis."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ReceiptError("Theil-Sen needs at least two matched finite observations")
    slopes = [(y_values[j] - y_values[i]) / (x_values[j] - x_values[i]) for i in range(len(x_values) - 1) for j in range(i + 1, len(x_values))]
    return float(statistics.median(slopes))


def derive_rates(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive robust convergence, rank, and archive-rate trajectories."""
    updates = [float(row["update"]) for row in trajectory]
    seconds = [float(row["wall_time_seconds"]) for row in trajectory]
    def slopes(values: Sequence[float], *, decline: bool = False) -> dict[str, float]:
        sign = -1.0 if decline else 1.0
        return {"theil_sen_per_update": sign * theil_sen_slope(updates, values), "theil_sen_per_wall_second": sign * theil_sen_slope(seconds, values)}
    by_class = {
        name: {"start_d_seg": trajectory[0]["d_seg_by_class"][name], "end_d_seg": trajectory[-1]["d_seg_by_class"][name], **slopes([float(row["d_seg_by_class"][name]) for row in trajectory], decline=True)}
        for name in CLASSES
    }
    return {
        "n_observations": len(trajectory),
        "per_class_d_seg": by_class,
        "overall_d_seg": {"start": trajectory[0]["overall_d_seg"], "end": trajectory[-1]["overall_d_seg"], **slopes([float(row["overall_d_seg"]) for row in trajectory], decline=True)},
        "trunk_weight_rank": {"kind": trajectory[0]["trunk_weight_rank"]["kind"], "start": trajectory[0]["trunk_weight_rank"]["value"], "end": trajectory[-1]["trunk_weight_rank"]["value"], **slopes([float(row["trunk_weight_rank"]["value"]) for row in trajectory])},
        "archive": {"start_bytes": trajectory[0]["archive_bytes"], "start_sha256": trajectory[0]["archive_sha256"], "end_bytes": trajectory[-1]["archive_bytes"], "end_sha256": trajectory[-1]["archive_sha256"], "start_rate_bytes_per_pair": trajectory[0]["archive_rate_bytes_per_pair"], "end_rate_bytes_per_pair": trajectory[-1]["archive_rate_bytes_per_pair"], **slopes([float(row["archive_rate_bytes_per_pair"]) for row in trajectory])},
    }


def _keeps_learning(class_rates: Mapping[str, Any]) -> dict[str, Any]:
    update = float(class_rates["theil_sen_per_update"])
    wall = float(class_rates["theil_sen_per_wall_second"])
    return {"keeps_learning": update > 0.0 and wall > 0.0, "criterion": "positive robust d_seg decline on both update and wall-time axes", "theil_sen_decline_per_update": update, "theil_sen_decline_per_wall_second": wall}


def compare_rates(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    """Describe A/B deltas without inferring a causal mechanism from this instance."""
    rare = {name: {"control": _keeps_learning(control["per_class_d_seg"][name]), "treatment": _keeps_learning(treatment["per_class_d_seg"][name])} for name in RARE_CLASSES}
    return {
        "per_class_decline_delta_treatment_minus_control": {name: {axis: float(treatment["per_class_d_seg"][name][axis]) - float(control["per_class_d_seg"][name][axis]) for axis in ("theil_sen_per_update", "theil_sen_per_wall_second")} for name in CLASSES},
        "rare_class_keeps_learning": rare,
        "overall_d_seg_decline_delta_treatment_minus_control": {axis: float(treatment["overall_d_seg"][axis]) - float(control["overall_d_seg"][axis]) for axis in ("theil_sen_per_update", "theil_sen_per_wall_second")},
        "rank_rate_tradeoff": {
            "rank_kind": control["trunk_weight_rank"]["kind"],
            "end_trunk_weight_rank_delta_treatment_minus_control": float(treatment["trunk_weight_rank"]["end"]) - float(control["trunk_weight_rank"]["end"]),
            "end_archive_bytes_delta_treatment_minus_control": int(treatment["archive"]["end_bytes"]) - int(control["archive"]["end_bytes"]),
            "end_archive_rate_bytes_per_pair_delta_treatment_minus_control": float(treatment["archive"]["end_rate_bytes_per_pair"]) - float(control["archive"]["end_rate_bytes_per_pair"]),
        },
        "mechanism_interpretation": "SPECULATIVE transfer from LM plasticity literature; this A/B is correlational mechanism evidence only",
    }


def _blocked_output(*, preregistration_sha256: str | None, control_sha256: str | None, treatment_sha256: str | None, error: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution": "ZERO_LAUNCH_RECEIPT_ANALYZER",
        "verdict_scope": "INSTANCE",
        "evidence_status": "BLOCKED_NO_EMPIRICAL_CLAIM",
        "owed_status": "OWED",
        "launch_authorized": False,
        "launch_blockers": list(LAUNCH_BLOCKERS),
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "blocker": error,
        "source_sha256": {"preregistration": preregistration_sha256, "control": control_sha256, "treatment": treatment_sha256},
    }
    output["content_address_sha256"] = canonical_sha256(output)
    return output


def analyze_receipts(preregistration_path: Path, control_path: Path, treatment_path: Path) -> tuple[dict[str, Any], bool]:
    """Analyze supplied receipt bytes only; never launch or construct a trainer config."""
    prereg_sha256 = control_sha256 = treatment_sha256 = None
    try:
        preregistration, prereg_sha256 = read_json(preregistration_path)
        control, control_sha256 = read_json(control_path)
        treatment, treatment_sha256 = read_json(treatment_path)
        control_rows, treatment_rows = validate_matched_receipts(preregistration, prereg_sha256, control, treatment)
        control_rates, treatment_rates = derive_rates(control_rows), derive_rates(treatment_rows)
    except ReceiptError as exc:
        return _blocked_output(preregistration_sha256=prereg_sha256, control_sha256=control_sha256, treatment_sha256=treatment_sha256, error=str(exc)), False
    output: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution": "ZERO_LAUNCH_RECEIPT_ANALYZER",
        "verdict_scope": "INSTANCE",
        "evidence_status": "MEASURED_FROM_SUPPLIED_REAL_N600_RECEIPTS",
        "owed_status": "CLOSED",
        "launch_authorized": False,
        "launch_blockers": list(LAUNCH_BLOCKERS),
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "mechanism_interpretation": "SPECULATIVE transfer from LM plasticity literature; correlational mechanism only, not witness-mechanism evidence",
        "canonical_classes": list(CLASSES),
        "source_sha256": {"preregistration": prereg_sha256, "control": control_sha256, "treatment": treatment_sha256},
        "pinned_weight_decay": _validate_preregistration(preregistration),
        "rates": {"control": control_rates, "treatment": treatment_rates},
        "comparison": compare_rates(control_rates, treatment_rates),
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
    parser.add_argument("--preregistration", type=Path, required=True, help="immutable numeric weight-decay preregistration JSON")
    parser.add_argument("--control-receipt", type=Path, required=True, help="existing real-n600 control receipt")
    parser.add_argument("--treatment-receipt", type=Path, required=True, help="existing real-n600 treatment receipt")
    parser.add_argument("--output", type=Path, required=True, help="analysis JSON written atomically")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output, admitted = analyze_receipts(args.preregistration, args.control_receipt, args.treatment_receipt)
    write_json_atomically(args.output, output)
    if not admitted:
        print(f"OWED: {output['blocker']}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "content_address_sha256": output["content_address_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
