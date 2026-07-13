#!/usr/bin/env python3
"""Validate future counted receiver-close compander A/B receipts without running an A/B.

This is a pure receipt comparator. It performs no training, inflation, scoring, archive
creation, or provider dispatch. Both input receipts must already contain their custody and
realized measurements from the governed launch/evaluation path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

CANONICAL_CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
TREATMENT_FACTORY = "MarginCompandedGroundChart"
SCHEMA = "compander_receiver_close_ab.v1"


class ReceiptRefusal(ValueError):
    """Raised when a future arm lacks matched-treatment or receiver custody."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_sha256(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise ReceiptRefusal(f"{field} must be a lowercase SHA-256")
    return str(value)


def _finite_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReceiptRefusal(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ReceiptRefusal(f"{field} must be finite")
    return result


def _validate_arm(arm: dict[str, Any], label: str) -> dict[str, Any]:
    if int(arm.get("n_pairs", -1)) != 600:
        raise ReceiptRefusal(f"{label}.n_pairs must equal 600")
    steps = int(arm.get("optimizer_steps", -1))
    if steps <= 0:
        raise ReceiptRefusal(f"{label}.optimizer_steps must be positive")
    seed = arm.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ReceiptRefusal(f"{label}.seed must be a nonnegative integer")
    axis = arm.get("axis")
    if not isinstance(axis, str) or not axis.strip():
        raise ReceiptRefusal(f"{label}.axis custody is required")
    config = arm.get("config_custody")
    if not isinstance(config, dict):
        raise ReceiptRefusal(f"{label}.config_custody is required")
    config_sha = _require_sha256(config.get("sha256"), f"{label}.config_custody.sha256")

    archive = arm.get("archive_custody")
    if not isinstance(archive, dict):
        raise ReceiptRefusal(f"{label}.archive_custody is required")
    archive_bytes = int(archive.get("bytes", -1))
    if archive_bytes <= 0:
        raise ReceiptRefusal(f"{label}.archive_custody.bytes must be positive")
    archive_sha = _require_sha256(archive.get("sha256"), f"{label}.archive_custody.sha256")

    receiver = arm.get("receiver_custody")
    if not isinstance(receiver, dict):
        raise ReceiptRefusal(f"{label}.receiver_custody is required")
    if receiver.get("parseback_passed") is not True:
        raise ReceiptRefusal(f"{label} archive parse-back did not pass")
    if receiver.get("archive_sha256") != archive_sha:
        raise ReceiptRefusal(f"{label} receiver archive SHA does not match archive custody")
    decoded_sha = _require_sha256(
        receiver.get("decoded_sha256"), f"{label}.receiver_custody.decoded_sha256"
    )
    reference_sha = _require_sha256(
        receiver.get("prearchive_reference_sha256"),
        f"{label}.receiver_custody.prearchive_reference_sha256",
    )
    if decoded_sha != reference_sha:
        raise ReceiptRefusal(f"{label} receiver decode differs from pre-archive reference")
    runtime_sha = _require_sha256(
        receiver.get("runtime_sha256"), f"{label}.receiver_custody.runtime_sha256"
    )

    per_class = arm.get("per_class_d_seg")
    if not isinstance(per_class, dict):
        raise ReceiptRefusal(f"{label}.per_class_d_seg is required")
    normalized_per_class = {
        name: _finite_number(per_class.get(name), f"{label}.per_class_d_seg.{name}")
        for name in CANONICAL_CLASSES
    }
    d_pose = _finite_number(arm.get("d_pose"), f"{label}.d_pose")
    return {
        "n_pairs": 600,
        "optimizer_steps": steps,
        "seed": seed,
        "axis": axis,
        "config_sha256": config_sha,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "receiver_decoded_sha256": decoded_sha,
        "receiver_runtime_sha256": runtime_sha,
        "per_class_d_seg": normalized_per_class,
        "d_pose": d_pose,
    }


def compare_receipts(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    """Return treatment-minus-control effects after strict matched-arm validation."""

    control_checked = _validate_arm(control, "control")
    treatment_checked = _validate_arm(treatment, "treatment")
    if control_checked["optimizer_steps"] != treatment_checked["optimizer_steps"]:
        raise ReceiptRefusal("control/treatment optimizer steps are not matched")
    if control_checked["seed"] != treatment_checked["seed"]:
        raise ReceiptRefusal("control/treatment seeds are not matched")
    if control_checked["archive_bytes"] != treatment_checked["archive_bytes"]:
        raise ReceiptRefusal("control/treatment total archive bytes are not matched")
    if control_checked["axis"] != treatment_checked["axis"]:
        raise ReceiptRefusal("control/treatment authority axes are not identical")

    levers = treatment.get("dsl_lever_factories")
    if not isinstance(levers, list) or TREATMENT_FACTORY not in levers:
        raise ReceiptRefusal(f"treatment must identify DSL factory {TREATMENT_FACTORY}")
    payload = treatment.get("chart_payload_custody")
    if not isinstance(payload, dict):
        raise ReceiptRefusal("treatment.chart_payload_custody is required")
    if payload.get("counted_in_total_archive_bytes") is not True:
        raise ReceiptRefusal("video-derived chart payload is not explicitly receiver-counted")
    payload_bytes = int(payload.get("bytes", -1))
    if payload_bytes <= 0 or payload_bytes > treatment_checked["archive_bytes"]:
        raise ReceiptRefusal("chart payload bytes must be positive and inside total archive bytes")
    payload_sha = _require_sha256(
        payload.get("sha256"), "treatment.chart_payload_custody.sha256"
    )
    if payload.get("containing_archive_sha256") != treatment_checked["archive_sha256"]:
        raise ReceiptRefusal("chart payload custody does not name the treatment archive SHA")

    deltas = {
        name: (
            treatment_checked["per_class_d_seg"][name]
            - control_checked["per_class_d_seg"][name]
        )
        for name in CANONICAL_CLASSES
    }
    d_pose_delta = treatment_checked["d_pose"] - control_checked["d_pose"]
    return {
        "schema": SCHEMA,
        "verdict_scope": (
            "receiver-closed counted matched-bytes/matched-steps n600 compander A/B only"
        ),
        "authority_axis": control_checked["axis"],
        "matched": {
            "n_pairs": 600,
            "optimizer_steps": control_checked["optimizer_steps"],
            "seed": control_checked["seed"],
            "total_archive_bytes": control_checked["archive_bytes"],
        },
        "treatment_factory": TREATMENT_FACTORY,
        "chart_payload": {
            "bytes_counted_inside_archive": payload_bytes,
            "sha256": payload_sha,
            "containing_archive_sha256": treatment_checked["archive_sha256"],
        },
        "per_class_d_seg_delta_treatment_minus_control": deltas,
        "primary_lane_d_seg_delta_treatment_minus_control": deltas["Lane"],
        "d_pose_delta_treatment_minus_control": d_pose_delta,
        "pose_nonworsening": d_pose_delta <= 0.0,
        "rate_matched_exactly": True,
        "score_claim": False,
        "promotion_claim": False,
        "interpretation": (
            "negative d_seg delta is improvement; Lane is primary; rate is matched, and pose "
            "must not worsen before any promotion decision"
        ),
        "control_custody": control_checked,
        "treatment_custody": treatment_checked,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-receipt", type=Path, required=True)
    parser.add_argument("--treatment-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control_receipt.read_text(encoding="utf-8"))
    treatment = json.loads(args.treatment_receipt.read_text(encoding="utf-8"))
    result = compare_receipts(control, treatment)
    _atomic_json(args.out, result)
    print(json.dumps({"out": str(args.out), "score_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
