#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Re-derive the bounded DDM M4 rate-floor receipt without launching a scorer.

The command consumes existing hash-bound receiver/evaluator receipts.  It does
not create a new score row, dispatch work, or mutate any source artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.canonical_equations.ddm_m4_rate_floor_20260723 import (  # noqa: E402
    LEVER_POOLS,
    SETTLED_D_POSE,
    SETTLED_D_SEG,
    ReceiverRow,
    minimum_admissible_receiver_row,
    score_terms,
    strict_archive_cap_bytes,
    uint8_unrecovered_scheduled_debt,
)

RECEIPT_PATHS = {
    "joint_optimum_575": Path(".omx/research/joint_optimum_575_xhigh_exact_row_20260720.json"),
    "qaxis": Path(".omx/research/qaxis_bitdepth_response_surface_20260623T232215Z.json"),
    "tight_c1": Path(".omx/research/inverse_solve_einstein_avenue_20260720T092250Z_receipt.json"),
    "v19b": Path(
        ".omx/research/ddm_v19b_joint_remeasure_stack_20260723T051914Z/ddm_v19b_joint_remeasure_stack_receipt.json"
    ),
    "null_kernel": Path(".omx/research/null_compiler_full_kernel_20260720T163500Z.json"),
    "xi_temporal": Path(".omx/research/xi_temporal_delta_coder_574_20260721T222234Z.json"),
    "mdl_member": Path(".omx/research/mdl_polytope_member_solve_receipt_20260721.json"),
}
ULTRA_COMMIT = "5cc81f1172"
ULTRA_MEMO = ".omx/research/einstein_kolmogorov_ultra_20260721T155726Z.md"


class DerivationError(RuntimeError):
    """Raised when an input receipt or invariant does not match its custody."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise DerivationError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DerivationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DerivationError(f"top-level JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_receipt(repo_root: Path, relative_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = repo_root / relative_path
    value = _load_json(path)
    return value, {
        "path": str(relative_path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _row_payload(row: ReceiverRow) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "archive_bytes": row.archive_bytes,
        "d_seg": _decimal_text(row.d_seg),
        "d_pose": _decimal_text(row.d_pose),
        "n_pairs": row.n_pairs,
        "receiver_closed": row.receiver_closed,
        "evidence_axis": row.evidence_axis,
    }


def _read_ultra_memo(repo_root: Path) -> tuple[str, dict[str, Any]]:
    result = subprocess.run(
        ["git", "show", f"{ULTRA_COMMIT}:{ULTRA_MEMO}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DerivationError(f"cannot read #604 memo at {ULTRA_COMMIT}: {result.stderr}")
    memo = result.stdout
    required = (
        "move-to-counted | none in the audited tree",
        "compliant archive-byte deltas are both **0 B**",
        "current `move-to-counted` bucket is empty",
        "154600_BYTES_NOT_RULED_OUT",
    )
    missing = [token for token in required if token not in memo]
    if missing:
        raise DerivationError(f"#604 memo contract drift: missing {missing}")
    return memo, {
        "git_commit": ULTRA_COMMIT,
        "path": ULTRA_MEMO,
        "bytes": len(memo.encode("utf-8")),
        "sha256": hashlib.sha256(memo.encode("utf-8")).hexdigest(),
    }


def derive(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return the deterministic bounded rate-floor receipt."""

    loaded: dict[str, dict[str, Any]] = {}
    custody: dict[str, dict[str, Any]] = {}
    for name, relative_path in RECEIPT_PATHS.items():
        loaded[name], custody[name] = _source_receipt(repo_root, relative_path)

    joint = loaded["joint_optimum_575"]
    joint_row = joint["full_precision_same_container_selection"]
    joint_custody = joint["candidate_custody"]
    archive_path = Path(joint_custody["archive_path"])
    if not archive_path.is_file():
        raise DerivationError(f"#575 archive missing: {archive_path}")
    if archive_path.stat().st_size != int(joint_custody["archive_bytes"]):
        raise DerivationError("#575 archive size mismatch")
    if _sha256(archive_path) != joint_custody["archive_sha256"]:
        raise DerivationError("#575 archive SHA-256 mismatch")
    if joint_row["archive_sha256"] != joint_custody["archive_sha256"]:
        raise DerivationError("#575 score row and archive custody disagree")

    rows: list[ReceiverRow] = [
        ReceiverRow(
            row_id="joint_optimum_575_full_precision_same_container",
            archive_bytes=int(joint_row["archive_bytes"]),
            d_seg=Decimal(str(joint_row["d_seg"])),
            d_pose=Decimal(str(joint_row["d_pose"])),
            n_pairs=int(joint_row["n_pairs_scored"]),
            receiver_closed=joint_custody["inflated_raw_custody"] == "PASS" and joint_custody["zip_test"] == "PASS",
            evidence_axis=str(joint_row["axis_tag"]),
        )
    ]

    qaxis = loaded["qaxis"]
    if "byte-closed through the REAL frontier" not in qaxis["method"]:
        raise DerivationError("Q-axis receipt no longer declares real byte closure")
    for qrow in qaxis["surface_n600_gold_partial"]:
        rows.append(
            ReceiverRow(
                row_id=f"qaxis_n600_{qrow['variant']}_int{qrow['bits']}",
                archive_bytes=int(qrow["archive_bytes"]),
                d_seg=Decimal(str(qrow["d_seg"])),
                d_pose=Decimal(str(qrow["d_pose"])),
                n_pairs=600,
                receiver_closed=True,
                evidence_axis=str(qaxis["axis_tag"]),
            )
        )

    c1 = loaded["tight_c1"]["carried_c1_exact_row"]
    rows.append(
        ReceiverRow(
            row_id="historical_tight_c1_contest_cpu",
            archive_bytes=int(c1["archive"]["bytes"]),
            d_seg=Decimal(str(c1["d_seg"])),
            d_pose=Decimal(str(c1["d_pose"])),
            n_pairs=int(c1["pairs"]),
            receiver_closed=True,
            evidence_axis=str(c1["axis"]),
        )
    )

    v19b = loaded["v19b"]
    v19b_measurement = v19b["n600"]["measurement"]
    rows.append(
        ReceiverRow(
            row_id="ddm_v19b_modern_low_byte_advisory",
            archive_bytes=int(v19b_measurement["archive_bytes"]),
            d_seg=Decimal(str(v19b_measurement["d_seg"])),
            d_pose=Decimal(str(v19b_measurement["d_pose"])),
            n_pairs=600,
            receiver_closed=True,
            evidence_axis=str(v19b_measurement["evidence_axis"]),
        )
    )

    relaxed_min = minimum_admissible_receiver_row(rows)
    exact_c1_min = minimum_admissible_receiver_row(
        rows,
        max_d_seg=SETTLED_D_SEG,
        max_d_pose=SETTLED_D_POSE,
    )
    if relaxed_min.archive_bytes != 177_169:
        raise DerivationError("audited relaxed receiver floor drifted from 177169 bytes")
    if exact_c1_min.archive_bytes != 409_526_925:
        raise DerivationError("audited exact-C1 receiver floor drifted from 409526925 bytes")

    sub015_cap = strict_archive_cap_bytes(SETTLED_D_SEG, SETTLED_D_POSE)
    at_floor = score_terms(SETTLED_D_SEG, SETTLED_D_POSE, relaxed_min.archive_bytes)

    null_receipt = loaded["null_kernel"]
    null_coverage = null_receipt["coverage"]
    null_mdl = null_receipt["minimum_description"]
    if null_coverage["full_nullity_per_channel"] != 820_728:
        raise DerivationError("ker(A) nullity drift")
    if null_mdl["selected_full_kernel_frames"] != 0:
        raise DerivationError("unexpected full-kernel frame admission")
    if any(codec["full_kernel_delta_vs_old_mask"] != 0 for codec in null_mdl["bytes"].values()):
        raise DerivationError("unexpected measured full-kernel byte gain")

    xi_rows = loaded["xi_temporal"]["measurements"]["n600"]["rows"]
    xi_deltas = {row["arm"]: int(row["delta_bytes"]) for row in xi_rows}
    if xi_deltas["XTDL1_identity_xi_context_control"] != 7020:
        raise DerivationError("xi identity-control delta drift")
    if xi_deltas["XTDL1_planar3_from_composed_screw_predictor"] != 8508:
        raise DerivationError("xi planar3 delta drift")

    mdl = loaded["mdl_member"]
    if mdl["D3"]["member_byte_cut_fraction"] != 0.0:
        raise DerivationError("unexpected MDL member byte cut")

    ultra_memo, ultra_custody = _read_ultra_memo(repo_root)
    del ultra_memo
    custody["einstein_kolmogorov_ultra_604"] = ultra_custody

    v19b_nonadditivity = v19b["nonadditivity"]
    pool_rows = [
        {
            "pool_id": pool.pool_id,
            "levers": list(pool.levers),
            "relationship": pool.relationship,
            "evidence_scope": pool.evidence_scope,
        }
        for pool in LEVER_POOLS
    ]

    relaxed_gap = relaxed_min.archive_bytes - sub015_cap
    exact_gap = exact_c1_min.archive_bytes - sub015_cap
    current_knee = int(loaded["tight_c1"]["fixed_c1_distortion_intersection"]["strict_max_integer_archive_bytes"])
    if current_knee != 216_223:
        raise DerivationError("fixed-C1 pointer knee drift")

    return {
        "schema": "ddm_m4_rate_floor_einstein_avenue_receipt.v1",
        "generated_at_utc": "2026-07-23T10:16:15Z",
        "lane_id": "ddm_m4_rate_floor_einstein_avenue",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "execution": {
            "cost_usd": 0,
            "new_scorer_run": False,
            "new_launch": False,
            "config_changed": False,
            "method": "bounded deterministic re-derivation from existing receipts and exact bytes",
        },
        "source_receipts": custody,
        "rate_floor": {
            "universal_lower_bound_bytes": 0,
            "universal_lower_bound_status": "ONLY_SOUND_MODEL_INDEPENDENT_BOUND",
            "audited_relaxed_receiver_floor": {
                **_row_payload(relaxed_min),
                "status": "MEASURED_SMALLEST_IN_EXPLICIT_AUDITED_RECEIVER_SET",
                "global_minimum_claim": False,
                "archive_sha256": joint_custody["archive_sha256"],
                "archive_bytes_reverified_this_lane": True,
                "real_decode_and_n600_score_reused": True,
                "legacy_provenance_gap": (
                    "mounted code git head and upstream snapshot SHA were not recorded by "
                    "the historical dispatch; MAIN review remains required"
                ),
            },
            "audited_exact_c1_receiver_floor": {
                **_row_payload(exact_c1_min),
                "status": "MEASURED_SMALLEST_IN_EXPLICIT_AUDITED_EXACT_C1_SET",
                "global_minimum_claim": False,
                "archive_sha256": c1["archive"]["sha256"],
                "new_measurement_this_lane": False,
            },
            "strict_sub015_at_settled_c1": {
                "max_archive_bytes": sub015_cap,
                "continuous_crossing_bytes": "154524.64741171754807217378757866849541357609231823",
                "score_at_cap": _decimal_text(score_terms(SETTLED_D_SEG, SETTLED_D_POSE, sub015_cap)["total"]),
                "score_at_cap_plus_one": _decimal_text(
                    score_terms(SETTLED_D_SEG, SETTLED_D_POSE, sub015_cap + 1)["total"]
                ),
                "audited_relaxed_floor_gap_bytes": relaxed_gap,
                "audited_relaxed_floor_cut_required_fraction": _decimal_text(
                    Decimal(relaxed_gap) / Decimal(relaxed_min.archive_bytes)
                ),
                "audited_exact_c1_floor_gap_bytes": exact_gap,
            },
            "fixed_c1_pointer_knee": {
                "strict_max_archive_bytes": current_knee,
                "relaxed_floor_below_knee_bytes": current_knee - relaxed_min.archive_bytes,
                "note": "#604 216207-byte row is description-only; 216223 is the exact score-law knee",
            },
            "counterfactual_settled_score_at_relaxed_floor_bytes": {
                key: _decimal_text(value) for key, value in at_floor.items()
            },
            "sub015_reachability": "NOT_CURRENTLY_REACHED_BUT_NOT_RULED_OUT",
            "decisive_gap_bytes": relaxed_gap,
        },
        "audited_rows": [_row_payload(row) for row in sorted(rows, key=lambda r: r.archive_bytes)],
        "rule_118_partition": {
            "free_zero_rate": [
                "generic archive parser and integrity checks",
                "generic xi integrator and trajectory evaluator",
                "generic power-diagram or chart rasterizer",
                "generic deterministic seed-to-table expansion",
                "generic receiver, resize plumbing, and codec interpreter logic",
            ],
            "counted_video_derived": [
                "learned/video-fit weights, latents, coefficients, and initial conditions",
                "video-derived xi values and per-pair pose carrier state",
                "chart, topology, event, exception, and residual symbols",
                "video-derived palette/camera constants or lookup tables even if hidden in code",
            ],
            "null_omitted": ["gauge coordinates and deterministic dither with no receiver-visible statistic"],
            "measured_current_free_reclassification_reduction_bytes": 0,
            "move_to_counted_bucket_bytes": 0,
            "scope": "#604 audited S4 runtime source; no pointer-archive reclassification inferred",
        },
        "ker_A": {
            "domain_dimension_per_channel": null_coverage["domain_dimension_per_channel"],
            "rank_per_channel": null_coverage["resize_rank_per_channel"],
            "nullity_per_channel": null_coverage["full_nullity_per_channel"],
            "nullity_percent": null_coverage["full_nullity_percent"],
            "bounded_integer_basis_reachability_lower_bound_percent": null_receipt["uint8_reachability"][
                "feasible_basis_percent_lower_bound"
            ],
            "selected_full_kernel_frames": null_mdl["selected_full_kernel_frames"],
            "measured_counted_bytes_hideable_for_free": 0,
            "diagnostic_old_mask_brotli_delta_vs_original_bytes": null_mdl["bytes"]["brotli"][
                "old_mask_delta_vs_original"
            ],
            "diagnostic_old_mask_is_archive_byte_delta": False,
            "incremental_full_kernel_brotli_delta_vs_old_mask_bytes": null_mdl["bytes"]["brotli"][
                "full_kernel_delta_vs_old_mask"
            ],
            "byte_savings_status": "ZERO_MEASURED_NOT_80P67_PERCENT_OF_ARCHIVE",
            "law": (
                "ker(A) can make a perturbation distortion-invisible; it cannot make "
                "video-derived serialization free. Current exact receiver stores range(A) "
                "only and no parser-consumed ker(A) payload was removed."
            ),
        },
        "nonadditive_pools": {
            "pools": pool_rows,
            "v19b_joint_replay_evidence": {
                key: v19b_nonadditivity[key]
                for key in (
                    "single_step_gain_total",
                    "survived_gain_total",
                    "amplified_gain_total",
                    "degraded_gain_total",
                    "survival_fraction",
                )
            },
            "xi_temporal_same_pool_negative_bytes": xi_deltas,
            "composition_law": (
                "Only within-pool joint replay receives credit. Cross-pool deltas are "
                "conditionally composable only after one final same-artifact decode/score."
            ),
        },
        "integer_lattice": {
            "current_multicoefficient_solver": "FLOAT_COEFFICIENT_SEARCH_THEN_UINT8_RECEIVER_GATE",
            "lattice_native": False,
            "evidence": (
                "g2g2 serializes fp32 coefficient deltas and uses bounded projected greedy/"
                "coordinate search; factor2_uint8_exact is an acceptance gate, not an "
                "integer-lattice optimization variable"
            ),
            "measured_n16_scheduled_recovery_s": "0.01583",
            "measured_realized_fraction": "-0.014",
            "measured_realized_score_delta_s": "-0.00022162",
            "unrecovered_scheduled_debt_s": _decimal_text(uint8_unrecovered_scheduled_debt()),
            "unrecovered_debt_is_measured_recoverable_gain": False,
            "flips_before": 10002,
            "flips_after": 10009,
            "verdict_scope": (
                "FORMULATION:absolute-write source-closest-sign n16 minimal perturbations; "
                "not n600 and not a lattice-native solver result"
            ),
        },
        "mdl_member_check": {
            "n64_member_cut_fraction": mdl["D3"]["member_byte_cut_fraction"],
            "canonical_member_bytes": mdl["D3"]["canonical_member_bytes"],
            "selected_member_bytes": mdl["D3"]["selected_member_bytes"],
            "verdict": mdl["verdict"],
            "global_mdl_optimum_claim": False,
        },
        "main_landing_review": {
            "required": True,
            "focus": [
                "confirm 177169 B is an audited-set minimum, not a global MDL lower bound",
                "confirm exact-C1 409526925 B and relaxed-box 177169 B are not conflated",
                "confirm rule-118 and ker(A) both yield zero measured byte reduction",
                "confirm pool assignments and the scoped n16 lattice debt",
            ],
        },
        "verdict": "SUB015_NOT_REACHED_22645_BYTE_GAP_NOT_RULED_OUT",
        "verdict_scope": (
            "Explicit audited set: #575 n600 exact row, qaxis n600 int8/int7/int6 rows, "
            "historical exact C1, and modern DDM v19b; existing receipts only. No global "
            "MDL optimum, family impossibility, new score, or promotion claim."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root containing the source receipts.",
    )
    parser.add_argument(
        "--verify-receipt",
        type=Path,
        help="Compare the deterministic derivation with an existing JSON receipt.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = derive(args.repo_root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.verify_receipt:
        expected = _load_json(args.verify_receipt)
        if result != expected:
            print("RECEIPT_MISMATCH", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "receipt": str(args.verify_receipt),
                    "receipt_sha256": _sha256(args.verify_receipt),
                    "rate_floor_bytes": result["rate_floor"]["audited_relaxed_receiver_floor"]["archive_bytes"],
                    "sub015_gap_bytes": result["rate_floor"]["decisive_gap_bytes"],
                },
                sort_keys=True,
            )
        )
        return 0
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
