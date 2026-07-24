#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Arbitrate the two bounded WS2 windows with the registered slope equation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.ddm_warm_start_slope_falsifier import (  # noqa: E402
    ObjectiveTerms,
    critical_pose_to_seg_slope_ratio,
    derive_warm_start_gap,
    evaluate_bounded_slope_window,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402

SCHEMA = "ddm_ws2_warm_start_slope_arbitration.v1"
WS3_SCHEMA = "ddm_ws3_warm_start_slope_arbitration.v1"
EXPECTED_R_STAR = 4.1215446777965665
POINTER = "0.1910828242 [contest-CPU]"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_receipt(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DirectDescriptionError(f"{label} receipt is unavailable: {path}")
    row = json.loads(path.read_bytes())
    if row.get("schema") != "ddm_joint_descent_full_run_receipt.v1":
        raise DirectDescriptionError(f"{label} full-run receipt schema differs")
    if row.get("bounded_verification") is not True:
        raise DirectDescriptionError(f"{label} is not a bounded verification")
    step = int(row.get("global_step", -1))
    if step != 4:
        decision = row.get("latest_realized_stage_decision", row.get("verdict"))
        raise DirectDescriptionError(
            f"{label} stopped at global_step={step}, not the preregistered "
            f"four-step endpoint; terminal_decision={decision}"
        )
    if row.get("baseline_verdict") is None or row.get("final_stage_verdict") is None:
        raise DirectDescriptionError(f"{label} endpoint verdict custody is incomplete")
    if row.get("score_claim") is not False:
        raise DirectDescriptionError(f"{label} score_claim custody differs")
    return row


def _terms(verdict: dict[str, Any]) -> ObjectiveTerms:
    return ObjectiveTerms.from_distortions(
        d_seg=float(verdict["d_seg"]),
        d_pose=float(verdict["d_pose"]),
    )


def _delta(start: ObjectiveTerms, end: ObjectiveTerms, steps: int) -> dict[str, float]:
    return {
        "seg_term_delta_per_step": (end.seg - start.seg) / steps,
        "pose_term_progress_per_step": (start.pose - end.pose) / steps,
        "total_distortion_term_delta": (
            end.seg + end.pose - start.seg - start.pose
        ),
    }


def arbitrate(
    *,
    producer_path: Path,
    w_seg_path: Path,
    w_joint_path: Path,
) -> dict[str, Any]:
    producer = json.loads(producer_path.read_bytes())
    if producer.get("schema") != "ddm_ws2_warm_start_custody_producer.v1":
        raise DirectDescriptionError("WS2 producer receipt schema differs")
    sealed = producer["sealed_batch16_endpoint_comparison"]
    r_star = critical_pose_to_seg_slope_ratio(
        wseg_d_seg=float(sealed["W_seg"]["sealed_batch16_d_seg"]),
        wseg_d_pose=float(sealed["W_seg"]["sealed_batch16_d_pose"]),
        wjoint_d_seg=float(sealed["W_joint"]["sealed_batch16_d_seg"]),
        wjoint_d_pose=float(sealed["W_joint"]["sealed_batch16_d_pose"]),
    )
    if abs(r_star - EXPECTED_R_STAR) > 1.0e-12:
        raise DirectDescriptionError(
            f"registered critical ratio differs: {r_star} != {EXPECTED_R_STAR}"
        )
    gap = derive_warm_start_gap(
        wseg=ObjectiveTerms.from_distortions(
            d_seg=float(sealed["W_seg"]["sealed_batch16_d_seg"]),
            d_pose=float(sealed["W_seg"]["sealed_batch16_d_pose"]),
        ),
        wjoint=ObjectiveTerms.from_distortions(
            d_seg=float(sealed["W_joint"]["sealed_batch16_d_seg"]),
            d_pose=float(sealed["W_joint"]["sealed_batch16_d_pose"]),
        ),
    )
    w_seg = _load_receipt(w_seg_path, "W_seg")
    w_joint = _load_receipt(w_joint_path, "W_joint")
    w_seg_start = _terms(w_seg["baseline_verdict"])
    w_seg_end = _terms(w_seg["final_stage_verdict"])
    w_joint_start = _terms(w_joint["baseline_verdict"])
    w_joint_end = _terms(w_joint["final_stage_verdict"])
    slope = evaluate_bounded_slope_window(
        gap=gap,
        start=w_seg_start,
        end=w_seg_end,
        steps=4,
    )
    selected = "W_seg" if slope.decision == "ADOPT_WSEG" else "W_joint"
    return {
        "schema": SCHEMA,
        "equation_id": "ddm_ws1_warm_start_slope_falsifier_v1",
        "critical_ratio": r_star,
        "registered_slope_verdict": {
            "decision": slope.decision,
            "reason": slope.reason,
            "seg_delta_per_step": slope.seg_delta_per_step,
            "pose_progress_per_step": slope.pose_progress_per_step,
            "seg_regression_per_step": slope.seg_regression_per_step,
            "observed_ratio": slope.observed_ratio,
            "predicted_pose_repayment_steps": slope.predicted_pose_repayment_steps,
            "predicted_seg_advantage_exhaustion_steps": (
                slope.predicted_seg_advantage_exhaustion_steps
            ),
        },
        "selected_warm_start": selected,
        "window_deltas": {
            "W_seg": _delta(w_seg_start, w_seg_end, 4),
            "W_joint": _delta(w_joint_start, w_joint_end, 4),
        },
        "inputs": {
            "producer_receipt": {
                "path": str(producer_path),
                "sha256": _sha(producer_path),
            },
            "W_seg_full_run_receipt": {
                "path": str(w_seg_path),
                "sha256": _sha(w_seg_path),
            },
            "W_joint_full_run_receipt": {
                "path": str(w_joint_path),
                "sha256": _sha(w_joint_path),
            },
        },
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "main_review_required": True,
    }


def arbitrate_formulation_stop(
    *,
    producer_path: Path,
    w_seg_path: Path,
    w_seg_terminal_proposal_path: Path,
    w_joint_path: Path,
) -> dict[str, Any]:
    """Apply the registered Seg-regression terminal law to reformed W_seg."""

    producer = json.loads(producer_path.read_bytes())
    if producer.get("schema") != "ddm_ws2_warm_start_custody_producer.v1":
        raise DirectDescriptionError("WS2 producer receipt schema differs")
    sealed = producer["sealed_batch16_endpoint_comparison"]
    r_star = critical_pose_to_seg_slope_ratio(
        wseg_d_seg=float(sealed["W_seg"]["sealed_batch16_d_seg"]),
        wseg_d_pose=float(sealed["W_seg"]["sealed_batch16_d_pose"]),
        wjoint_d_seg=float(sealed["W_joint"]["sealed_batch16_d_seg"]),
        wjoint_d_pose=float(sealed["W_joint"]["sealed_batch16_d_pose"]),
    )
    if abs(r_star - EXPECTED_R_STAR) > 1.0e-12:
        raise DirectDescriptionError(
            f"registered critical ratio differs: {r_star} != {EXPECTED_R_STAR}"
        )
    gap = derive_warm_start_gap(
        wseg=ObjectiveTerms.from_distortions(
            d_seg=float(sealed["W_seg"]["sealed_batch16_d_seg"]),
            d_pose=float(sealed["W_seg"]["sealed_batch16_d_pose"]),
        ),
        wjoint=ObjectiveTerms.from_distortions(
            d_seg=float(sealed["W_joint"]["sealed_batch16_d_seg"]),
            d_pose=float(sealed["W_joint"]["sealed_batch16_d_pose"]),
        ),
    )
    w_seg = json.loads(w_seg_path.read_bytes())
    if (
        w_seg.get("schema") != "ddm_joint_descent_full_run_receipt.v1"
        or w_seg.get("bounded_verification") is not True
        or int(w_seg.get("global_step", -1)) != 0
        or not str(w_seg.get("campaign_blocker", "")).startswith(
            "BLOCKED_REALIZED_NO_PURE_PRICED_DESCENT"
        )
        or w_seg.get("baseline_verdict") is None
        or w_seg.get("score_claim") is not False
    ):
        raise DirectDescriptionError("W_seg formulation-stop receipt custody differs")
    proposal = json.loads(w_seg_terminal_proposal_path.read_bytes())
    if (
        proposal.get("schema") != "ddm_joint_descent_chunked_stage_verdict.v1"
        or int(proposal.get("global_step", -1)) != 1
        or proposal.get("score_claim") is not False
        or proposal.get("pure_priced_delta", {}).get("accepted") is not True
        or proposal.get("warm_start_component_safe_residual_admitted") is not False
    ):
        raise DirectDescriptionError("W_seg exact terminal proposal custody differs")
    w_joint = _load_receipt(w_joint_path, "W_joint")
    exact_steps = w_joint.get("pose_finish_engage_state", {}).get(
        "exact_verdict_steps"
    )
    if exact_steps != [0, 1, 2, 3, 4]:
        raise DirectDescriptionError(
            f"W_joint exact verdict history differs: {exact_steps!r}"
        )
    w_seg_start = _terms(w_seg["baseline_verdict"])
    w_seg_terminal = _terms(proposal)
    w_joint_start = _terms(w_joint["baseline_verdict"])
    w_joint_end = _terms(w_joint["final_stage_verdict"])
    slope = evaluate_bounded_slope_window(
        gap=gap,
        start=w_seg_start,
        end=w_seg_terminal,
        steps=1,
    )
    if slope.decision != "KEEP_WJOINT" or slope.reason != "SEG_REGRESSION":
        raise DirectDescriptionError(
            "registered W_seg terminal law did not select KEEP_WJOINT/SEG_REGRESSION"
        )
    return {
        "schema": WS3_SCHEMA,
        "verdict": "ARBITRATED_FORMULATION_STOP_KEEP_WJOINT",
        "equation_id": "ddm_ws1_warm_start_slope_falsifier_v1",
        "critical_ratio": r_star,
        "registered_slope_verdict": {
            "decision": slope.decision,
            "reason": slope.reason,
            "seg_delta_per_step": slope.seg_delta_per_step,
            "pose_progress_per_step": slope.pose_progress_per_step,
            "seg_regression_per_step": slope.seg_regression_per_step,
            "observed_ratio": slope.observed_ratio,
            "predicted_pose_repayment_steps": slope.predicted_pose_repayment_steps,
            "predicted_seg_advantage_exhaustion_steps": (
                slope.predicted_seg_advantage_exhaustion_steps
            ),
        },
        "selected_warm_start": "W_joint",
        "window_status": {
            "W_seg": {
                "status": "FORMULATION_STOP_AT_EXACT_TERMINAL_PROPOSAL",
                "exact_steps": [0, 1],
                "accepted_steps": [0],
            },
            "W_joint": {
                "status": "COMPLETE_EXACT_FOUR_STEP_WINDOW",
                "exact_steps": exact_steps,
                "accepted_steps": [0, 1, 2, 3, 4],
            },
        },
        "window_deltas": {
            "W_seg_terminal_proposal": _delta(w_seg_start, w_seg_terminal, 1),
            "W_joint": _delta(w_joint_start, w_joint_end, 4),
        },
        "inputs": {
            "producer_receipt": {
                "path": str(producer_path),
                "sha256": _sha(producer_path),
            },
            "W_seg_full_run_receipt": {
                "path": str(w_seg_path),
                "sha256": _sha(w_seg_path),
            },
            "W_seg_terminal_proposal": {
                "path": str(w_seg_terminal_proposal_path),
                "sha256": _sha(w_seg_terminal_proposal_path),
            },
            "W_joint_full_run_receipt": {
                "path": str(w_joint_path),
                "sha256": _sha(w_joint_path),
            },
        },
        "verdict_scope": (
            "W_seg reformed-opening FORMULATION versus this exact W_joint INSTANCE; "
            "no family, paradigm, promotion, or score verdict"
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "main_review_required": True,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise DirectDescriptionError(f"immutable arbitration receipt differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _refusal_payload(
    *,
    reason: str,
    producer_path: Path,
    w_seg_path: Path,
    w_joint_path: Path,
) -> dict[str, Any]:
    def _input(path: Path) -> dict[str, Any]:
        row: dict[str, Any] = {"path": str(path), "available": path.is_file()}
        if path.is_file() and not path.is_symlink():
            row["sha256"] = _sha(path)
        return row

    return {
        "schema": SCHEMA,
        "verdict": "REFUSE_INCOMPLETE_FOUR_STEP_WINDOW",
        "reason": reason,
        "required_global_step": 4,
        "critical_ratio": EXPECTED_R_STAR,
        "equation_id": "ddm_ws1_warm_start_slope_falsifier_v1",
        "inputs": {
            "producer_receipt": _input(producer_path),
            "W_seg_full_run_receipt": _input(w_seg_path),
            "W_joint_full_run_receipt": _input(w_joint_path),
        },
        "verdict_scope": (
            "bounded warm-start slope arbitration only; no formulation, family, "
            "promotion, or score verdict"
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
        "main_review_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--w-seg-receipt", type=Path, required=True)
    parser.add_argument("--w-seg-terminal-proposal", type=Path)
    parser.add_argument("--w-joint-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.w_seg_terminal_proposal is None:
            result = arbitrate(
                producer_path=args.producer_receipt.resolve(),
                w_seg_path=args.w_seg_receipt.resolve(),
                w_joint_path=args.w_joint_receipt.resolve(),
            )
        else:
            result = arbitrate_formulation_stop(
                producer_path=args.producer_receipt.resolve(),
                w_seg_path=args.w_seg_receipt.resolve(),
                w_seg_terminal_proposal_path=(
                    args.w_seg_terminal_proposal.resolve()
                ),
                w_joint_path=args.w_joint_receipt.resolve(),
            )
        _atomic_json(args.output.resolve(), result)
    except (DirectDescriptionError, KeyError, ValueError) as exc:
        result = _refusal_payload(
            reason=str(exc),
            producer_path=args.producer_receipt.resolve(),
            w_seg_path=args.w_seg_receipt.resolve(),
            w_joint_path=args.w_joint_receipt.resolve(),
        )
        _atomic_json(args.output.resolve(), result)
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 4
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
