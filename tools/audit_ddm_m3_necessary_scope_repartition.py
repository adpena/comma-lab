#!/usr/bin/env python3
"""Fail-closed DDM M3 solve-versus-descent scope audit.

This tool re-derives only quantities supported by the landed c1, v14, v19b,
and G2 receipts.  In particular, it refuses to extrapolate the six-pair G2G2
Lane solve or the scorer-free G2 n600 atlas into a full-stratum inverse-solve
coverage claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

CLASS_ORDER = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
EXPECTED_SHA256 = {
    "c1": "14fdf1570b43df65ac949fe157e68ea328ff584f7df1331acf25cca8f900d936",
    "v14": "82d3249908d42a86575c407ab3d7acdf9b3706b31225f2e46862b2472966e5a9",
    "v19b": "4bb5d6b4b793b667c7cbe15e37cbf9a27f6c0e75451374839fb5df8ca1c1b8e8",
    "g2": "061220fd8c1ca047b210841235fc805194a96175e933ee110ba4ac8bb2077d84",
}


class AuditRefusal(ValueError):
    """Raised when source custody or semantic invariants do not match."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checked(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise AuditRefusal(f"SHA256_MISMATCH:{path}:expected={expected_sha256}:actual={actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditRefusal(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise AuditRefusal(reason)


def _v14_control_classes(v14: dict[str, Any]) -> dict[str, dict[str, int]]:
    ladder = v14.get("fixed_ladder")
    _require(isinstance(ladder, list) and ladder, "V14_FIXED_LADDER_REQUIRED")
    per_stratum = ladder[0].get("per_stratum")
    _require(isinstance(per_stratum, dict), "V14_PER_STRATUM_REQUIRED")
    result: dict[str, dict[str, int]] = {}
    for name in CLASS_ORDER:
        row = per_stratum.get(name)
        _require(isinstance(row, dict), f"V14_CLASS_REQUIRED:{name}")
        errors = row.get("errors")
        sites = row.get("sites")
        _require(
            isinstance(errors, int) and isinstance(sites, int) and sites > 0,
            f"V14_CLASS_COUNTS_INVALID:{name}",
        )
        result[name] = {"errors": errors, "sites": sites}
    return result


def _v19b_measurement_classes(v19b: dict[str, Any]) -> dict[str, dict[str, Any]]:
    n600 = v19b.get("n600")
    _require(isinstance(n600, dict), "V19B_N600_REQUIRED")
    measurement = n600.get("measurement")
    _require(isinstance(measurement, dict), "V19B_N600_MEASUREMENT_REQUIRED")
    per_class = measurement.get("per_class")
    _require(isinstance(per_class, dict), "V19B_PER_CLASS_REQUIRED")
    for name in CLASS_ORDER:
        _require(isinstance(per_class.get(name), dict), f"V19B_CLASS_REQUIRED:{name}")
    return per_class


def build_audit(
    *,
    c1: dict[str, Any],
    v14: dict[str, Any],
    v19b: dict[str, Any],
    g2: dict[str, Any],
    source_paths: dict[str, str],
) -> dict[str, Any]:
    """Build the deterministic scope receipt from already-verified JSON objects."""

    _require(c1.get("schema") == "ddm_c1_composed_candidate_spec.v1", "C1_SCHEMA")
    _require(
        v19b.get("schema") == "ddm_v19b_joint_remeasure_stack_receipt.v1",
        "V19B_SCHEMA",
    )
    _require(g2.get("schema") == "solve_diff_aggregate_ledger.v1", "G2_SCHEMA")
    _require(c1.get("score_claim") is False, "C1_SCORE_CLAIM_MUST_BE_FALSE")
    _require(v19b.get("score_claim") is False, "V19B_SCORE_CLAIM_MUST_BE_FALSE")
    _require(g2.get("score_claim") is False, "G2_SCORE_CLAIM_MUST_BE_FALSE")

    c1_debt = c1.get("debt")
    _require(isinstance(c1_debt, dict), "C1_DEBT_REQUIRED")
    assigned = c1_debt.get("integer_residual_after_perfect_lane_plus_movable")
    target_max = c1_debt.get("integer_target_error_max")
    _require(assigned == 2_377_273, "C1_ASSIGNED_RESIDUAL_DRIFT")
    _require(target_max == 136_839, "C1_TARGET_ERROR_MAX_DRIFT")

    control = _v14_control_classes(v14)
    measured = _v19b_measurement_classes(v19b)
    n600 = v19b["n600"]
    control_total = sum(row["errors"] for row in control.values())
    measured_total = sum(int(measured[name]["errors"]) for name in CLASS_ORDER)
    _require(control_total == n600["control"]["errors"] == 3_240_528, "CONTROL_SUM")
    _require(measured_total == n600["measurement"]["errors"] == 3_137_206, "V19B_SUM")

    delta_bytes = int(n600["joint_delta_vs_v15_control"]["delta_archive_bytes"])
    _require(delta_bytes == 3_884, "V19B_DELTA_BYTES_DRIFT")
    nonadditivity = v19b.get("nonadditivity")
    _require(isinstance(nonadditivity, dict), "V19B_NONADDITIVITY_REQUIRED")
    amplified_gain = float(nonadditivity["amplified_gain_total"])

    residual_bucket_names = ("Road", "Undrivable", "MyCar")
    residual_flips = 0
    role_flips = 0
    frontier: list[dict[str, Any]] = []
    for name in CLASS_ORDER:
        before = control[name]["errors"]
        after = int(measured[name]["errors"])
        sites = control[name]["sites"]
        _require(int(measured[name]["sites"]) == sites, f"CLASS_SITE_DRIFT:{name}")
        net_flips = before - after
        if name in residual_bucket_names:
            residual_flips += net_flips
        else:
            role_flips += net_flips
        solve_status = (
            "MEASURED_SUBSET_ONLY_N600_UNKNOWN"
            if name == "Lane"
            else "NOT_MEASURED_NO_FULL_STRATUM_MULTICOEFFICIENT_SOLVE"
        )
        frontier.append(
            {
                "stratum": name,
                "frame": "frame_1",
                "seg_incidence": "FULL",
                "control_errors": before,
                "control_sites": sites,
                "multicoefficient_inverse_solve": {
                    "status": solve_status,
                    "certified_closed_errors": None,
                    "certified_closed_fraction": None,
                    "complete_bytes": None,
                    "off_target_collateral": None,
                    "reason": (
                        "G2G2 measured only six selected pairs and 20 Lane "
                        "centerline coordinates; G2 n600 has no receiver delta d_seg."
                    ),
                },
                "v19b_correction_common_master": {
                    "status": "MEASURED_NET_CLASS_EFFECT",
                    "errors_after": after,
                    "net_flips": net_flips,
                    "class_conditional_delta_d_seg_after_minus_before": (-net_flips / sites),
                    "shared_stack_delta_bytes": delta_bytes,
                    "per_class_byte_allocation": None,
                    "off_target_collateral": ("NET_HARM" if net_flips < 0 else "NOT_IDENTIFIABLE_FROM_RECEIPT"),
                },
                "v19c_saturation_top_up": {
                    "status": "PENDING_RECEIPT_ABSENT",
                    "net_flips": None,
                    "bytes": None,
                },
                "frame_separability": {
                    "status": "SEG_FRAME_1_ONLY_EXACT",
                    "pose_coupling": "POSE_NET_READS_BOTH_FRAMES",
                },
                "certified_infeasible_residual": None,
                "certification_status": "NOT_CERTIFIED_COMPOSED_REACH_NOT_MEASURED",
            }
        )
        frontier.append(
            {
                "stratum": name,
                "frame": "frame_0",
                "seg_incidence": "ZERO_EXACT",
                "control_errors": 0,
                "control_sites": None,
                "multicoefficient_inverse_solve": {
                    "status": "NOT_APPLICABLE_NO_SEGNET_INCIDENCE",
                    "certified_closed_errors": 0,
                    "certified_closed_fraction": None,
                    "complete_bytes": 0,
                    "off_target_collateral": 0,
                },
                "correction_synergy": {
                    "status": "NOT_APPLICABLE_FOR_DSEG",
                    "net_flips": 0,
                    "bytes": 0,
                },
                "frame_separability": {
                    "status": "POSE_ONLY_PREIMAGE_OPEN_NOT_MEASURED_CURRENT_VEHICLE",
                    "r1_comparator": {
                        "d_pose": 0.001610,
                        "complete_bytes": 7_195,
                        "transferable_component": False,
                    },
                },
                "certified_infeasible_residual": 0,
                "certification_status": "ZERO_SEG_OBLIGATION_EXACT",
            }
        )

    _require(residual_flips == 73_945, "V19B_RESIDUAL_BUCKET_DELTA_DRIFT")
    _require(role_flips == 29_377, "V19B_ROLE_BUCKET_DELTA_DRIFT")
    bucket_receipt = n600["c1_bucket_delta_vs_v15_control"]
    _require(
        bucket_receipt["residual_bucket_Road_Undrivable_MyCar"]["realized_net_flips"] == residual_flips,
        "V19B_RESIDUAL_BUCKET_CROSSCHECK",
    )
    _require(
        bucket_receipt["role_bucket_Lane_plus_Movable"]["realized_net_flips"] == role_flips,
        "V19B_ROLE_BUCKET_CROSSCHECK",
    )

    candidate_admission = g2.get("candidate_admission")
    _require(isinstance(candidate_admission, dict), "G2_ADMISSION_REQUIRED")
    _require(
        candidate_admission.get("status") == "BLOCKED_NO_RECEIVER_DELTA_DSEG",
        "G2_ADMISSION_STATUS_DRIFT",
    )
    _require(
        candidate_admission.get("inner_encoder_jacobian") == "ABSENT",
        "G2_INNER_JACOBIAN_STATUS_DRIFT",
    )

    current_master_counterfactual = assigned - residual_flips
    evidence_percent = 100.0 * residual_flips / assigned
    return {
        "schema": "ddm_m3_necessary_scope_repartition_receipt.v1",
        "date_utc": "2026-07-23",
        "lane_id": "lane_ddm_m3_necessary_scope_repartition_20260723",
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "pointer": "0.1910828242 [contest-CPU Linux x86_64]",
        "pointer_moved": False,
        "main_landing_review_required": True,
        "source_paths": source_paths,
        "source_sha256": EXPECTED_SHA256,
        "frontier": frontier,
        "aggregate": {
            "c1_assigned_residual_errors": assigned,
            "target_error_max": target_max,
            "v19b_residual_bucket_net_flips": residual_flips,
            "v19b_role_bucket_net_flips": role_flips,
            "v19b_total_net_flips": control_total - measured_total,
            "v19b_shared_delta_bytes": delta_bytes,
            "v19b_amplified_gain_score_units": amplified_gain,
            "v19c_status": "PENDING_RECEIPT_ABSENT",
            "current_master_counterfactual_residual_after_v19b_subtraction": (current_master_counterfactual),
            "measured_stale_partition_evidence_percent": evidence_percent,
            "certified_infeasible_residual_errors": None,
            "true_necessary_scope_interval_errors": [0, assigned],
            "certified_over_scope_percent": None,
            "verdict": "NUMERIC_TRUE_SCOPE_NOT_CERTIFIABLE_CURRENT_CUSTODY",
            "verdict_scope": (
                "INSTANCE:C1_V14_V19B_G2_LANDED_RECEIPTS; no full-stratum "
                "multicoefficient solve, solve-then-correct common master, v19c "
                "saturation receipt, or current-vehicle frame0 Pose preimage"
            ),
        },
        "coverage_gaps": [
            "Road frame_1 full-stratum multicoefficient receiver solve",
            "Lane frame_1 beyond six selected pairs and 20 centerline coordinates",
            "Undrivable frame_1 full-stratum multicoefficient receiver solve",
            "Movable frame_1 full-stratum multicoefficient receiver solve",
            "MyCar frame_1 full-stratum multicoefficient receiver solve",
            "v19c correction saturation receipt",
            "solve-then-correct sequential common-master replay",
            "current-vehicle conditional frame_0 Pose preimage at n600",
        ],
        "named_missed_synergies": [
            {
                "name": "SOLVE_X_SE_CORRECTION",
                "status": "NOT_MEASURED",
                "reason": "v19b starts from v15, not a full-stratum solve output",
            },
            {
                "name": "CORRECTION_X_CORRECTION",
                "status": "MEASURED_V19B",
                "amplified_gain_score_units": amplified_gain,
            },
            {
                "name": "FRAME1_SEG_X_FRAME0_POSE",
                "status": "STRUCTURALLY_SEPARABLE_BUT_CURRENT_VEHICLE_RATE_NOT_MEASURED",
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-ledger", type=Path, required=True)
    parser.add_argument("--v14-receipt", type=Path, required=True)
    parser.add_argument("--v19b-receipt", type=Path, required=True)
    parser.add_argument("--g2-aggregate-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "c1": str(args.c1_ledger),
        "v14": str(args.v14_receipt),
        "v19b": str(args.v19b_receipt),
        "g2": str(args.g2_aggregate_ledger),
    }
    receipt = build_audit(
        c1=_load_checked(args.c1_ledger, EXPECTED_SHA256["c1"]),
        v14=_load_checked(args.v14_receipt, EXPECTED_SHA256["v14"]),
        v19b=_load_checked(args.v19b_receipt, EXPECTED_SHA256["v19b"]),
        g2=_load_checked(args.g2_aggregate_ledger, EXPECTED_SHA256["g2"]),
        source_paths=paths,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    tmp.write_text(rendered, encoding="utf-8")
    os.replace(tmp, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
