#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the scorer-free DDM DB1 decay/margin receipt.

This tool reads only immutable tracked/SSD receipts.  It never imports a scorer
or launcher and cannot mutate the frontier pointer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tac.analysis.ddm_db1_decay_bounds import (  # noqa: E402
    N600_SITES,
    analyze_margin_mass,
    analyze_v19c_curve,
    sha256_file,
)

DEFAULT_V19C = REPO / (
    ".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/"
    "stage_checkpoints/02_n600_saturation_curve.json"
)
DEFAULT_SN1_RECEIPT = REPO / (
    ".omx/research/ddm_sn1_segnet_telemetry_asymmetry_n600_20260723/"
    "ddm_sn1_segnet_telemetry_asymmetry_receipt.json"
)
DEFAULT_SN1_TELEMETRY = REPO / (
    ".omx/research/ddm_sn1_segnet_telemetry_asymmetry_n600_20260723/"
    "segnet_internal_telemetry_n600.jsonl"
)
DEFAULT_AT1_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/evidence/ddm_at1x_atlas_materialize_20260723T203027Z"
)
DEFAULT_WJOINT_HISTORY = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_ws3_w_joint_exact_history_20260724T132200Z/full_run_receipt.json"
)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _history_anchor(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    values = receipt["pose_finish_engage_state"]["exact_d_seg"]
    steps = receipt["pose_finish_engage_state"]["exact_verdict_steps"]
    first_delta = float(values[0]) - float(values[1])
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "exact_steps": steps,
        "exact_d_seg": values,
        "first_step_improvement": first_delta,
        "first_step_net_corrected_errors": int(round(first_delta * N600_SITES)),
        "strict_improvement_interval_count": sum(
            float(right) < float(left) for left, right in zip(values, values[1:])
        ),
        "interval_count": len(values) - 1,
        "authority": (
            "[naive-menu upper bound]; current broadcast forbids using generic proposal "
            "construction as scorer-recursive firing authority"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v19c-curve", type=Path, default=DEFAULT_V19C)
    parser.add_argument("--sn1-receipt", type=Path, default=DEFAULT_SN1_RECEIPT)
    parser.add_argument("--sn1-telemetry", type=Path, default=DEFAULT_SN1_TELEMETRY)
    parser.add_argument("--at1-manifest", type=Path, default=DEFAULT_AT1_ROOT / "atlas_manifest.json")
    parser.add_argument(
        "--at1-atlas", type=Path, default=DEFAULT_AT1_ROOT / "gaze_contraction_atlas.json"
    )
    parser.add_argument("--w-joint-history", type=Path, default=DEFAULT_WJOINT_HISTORY)
    parser.add_argument("--target-d-seg", type=float, default=8.684e-4)
    parser.add_argument("--horizon-admissions", type=int, default=450)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=603366)
    parser.add_argument("--written-at-utc", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    history = _history_anchor(args.w_joint_history)
    v19c = analyze_v19c_curve(
        args.v19c_curve,
        target=args.target_d_seg,
        horizon_additional_admissions=args.horizon_admissions,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    margin = analyze_margin_mass(
        sn1_receipt_path=args.sn1_receipt,
        telemetry_path=args.sn1_telemetry,
        at1_manifest_path=args.at1_manifest,
        at1_atlas_path=args.at1_atlas,
        operating_points={
            "W_joint_exact": 0.07051923116048177,
            "W_seg_class_card": 0.024124510023328993,
            "V19C_terminal": 0.024786978827582466,
            "card_130789B_target": args.target_d_seg,
            "card_128254B_prospective_target": 9.1e-4,
            "sub_0_15_stretch_target": 4.576e-4,
        },
        targets={
            "card_130789B": args.target_d_seg,
            "card_128254B_prospective": 9.1e-4,
            "sub_0_15_stretch": 4.576e-4,
        },
        opening_delta_d_seg=float(history["first_step_improvement"]),
    )
    selected = v19c["fits"][v19c["selected_by_aicc"]]
    receipt = {
        "schema": "ddm_db1_decay_bounds.v1",
        "run_id": "ddm_db1_decay_bounds_20260725",
        "written_at_utc": args.written_at_utc or _utc_now(),
        "runtime": {
            "python": sys.version,
            "numpy": __import__("numpy").__version__,
            "pid": os.getpid(),
        },
        "delegation_checkpoint_key": "codex_delegate:ddm_db1_decay_bounds:20260725T121605Z",
        "lane_id": "ddm_db1_decay_bounds",
        "research_only": True,
        "execution_allowed": False,
        "paid_dispatch": False,
        "score_claim": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory artifacts; no new scorer invocation]",
        "v19c_decay_fit": v19c,
        "sn1_at1_margin_mass": margin,
        "descent_opening_history": history,
        "reconciliation": {
            "v19c_selected_model": v19c["selected_by_aicc"],
            "v19c_selected_asymptote": selected["asymptote"],
            "v19c_horizon_d_seg": selected["predicted_d_seg_at_horizon"],
            "v19c_analogue_reaches_target_in_horizon": (
                selected["predicted_d_seg_at_horizon"] <= args.target_d_seg
            ),
            "margin_result": (
                "The fixed SN1 boundary support is too small to carry W_seg-class to the "
                "card target even under an all-beneficial oracle, but this is not a live "
                "descent lower bound because boundary support can be replenished."
            ),
            "fit_result": (
                "Both fitted V19C families remain above 0.023-class at the 450-admission "
                "horizon; the AICc-selected exponential asymptote is about 0.0244."
            ),
            "why_not_transfer": (
                "V19C is an accepted finite proposal-order process. SN1 is a fixed predicted-"
                "boundary atlas without target-error coordinates. Neither identifies the "
                "transport kernel or replenishment law of the scorer-recursive #366 engine."
            ),
        },
        "gate_g2": {
            "verdict": "INDETERMINATE",
            "verdict_scope": (
                "FORMULATION: current hash-bound V19C accepted proposal-order curve plus fixed "
                "SN1/AT1 predicted-boundary atlas. V19C-instance in-horizon target reach is "
                "excluded; live scorer-recursive descent, boundary replenishment, E7 residual "
                "coding, solve-derived knee states, and broader description families remain open."
            ),
            "card_e7_disposition": (
                "DO_NOT_REFUSE_FROM_DESCENT_ANALOGUE_ALONE. Optimal stop D* cannot be identified "
                "because neither marginal step transport/cost nor H(flip-field|free context) "
                "is present in the current atlas custody."
            ),
            "cheapest_resolving_measurement": (
                "Augment the already-owed bounded J10 re-smoke, not a new arm: for 10 accepted "
                "exact n600 verdict intervals from the card-selected start, persist target-error-"
                "conditioned (pair,y,x,winner,rival,margin,d2) before/after rows, gross beneficial/"
                "harmful flips, realized step seconds, and context-coded residual bits. This one "
                "same-parent trace identifies decay transport and E7's marginal step-vs-byte D*."
            ),
            "fire_authority": "MAIN_ONLY_AFTER_REVIEW_AND_ALL_CARD_GATES",
        },
        "equation_disposition": {
            "register": True,
            "equation_id": "ddm_sn1_margin_mass_duplicate_budget_bounds_v1",
            "reason": (
                "The duplicate-budget N(delta) inequality is an exact measured combinatorial "
                "law in the fixed-atlas domain. V19C decay fits remain receipt-local models and "
                "must not be registered as transferable laws."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "verdict": receipt["gate_g2"]["verdict"],
                "selected_v19c_model": v19c["selected_by_aicc"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
