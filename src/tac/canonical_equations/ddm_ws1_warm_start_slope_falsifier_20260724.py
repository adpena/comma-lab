# SPDX-License-Identifier: MIT
"""Canonical critical-slope law for the two measured DDM WS1 warm starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.optimization.ddm_warm_start_slope_falsifier import (
    ObjectiveTerms,
    critical_pose_to_seg_slope_ratio,
    derive_warm_start_gap,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT = (
    ".omx/research/ddm_ws1_seglex96_filtered_warmstart_20260724T022500Z/"
    "ddm_ws1_seglex96_filtered_warmstart_receipt.json"
)
EQUATION_ID = "ddm_ws1_warm_start_slope_falsifier_v1"
MEASUREMENT_UTC = "2026-07-24T02:20:00Z"


def _load_receipt(path: Path | None = None) -> tuple[dict[str, Any], Path, str]:
    source = path or REPO_ROOT / RECEIPT
    payload_bytes = source.read_bytes()
    payload = json.loads(payload_bytes)
    if (
        payload.get("schema")
        != "ddm_ws1_seglex96_filtered_warmstart_measurement.v1"
        or payload.get("score_claim") is not False
        or payload.get("pointer_moved") is not False
    ):
        raise ValueError("WS1 receipt schema or advisory authority differs")
    return payload, source, hashlib.sha256(payload_bytes).hexdigest()


def build_ddm_ws1_warm_start_slope_falsifier_v1(
    receipt_path: Path | None = None,
) -> CanonicalEquation:
    """Build R* from the measured W_seg and W_joint task-term gaps."""

    payload, source, receipt_sha = _load_receipt(receipt_path)
    candidates = payload["warm_start_candidates"]
    wseg = candidates["W_seg"]
    wjoint = candidates["W_joint"]
    gap = derive_warm_start_gap(
        wseg=ObjectiveTerms.from_distortions(
            d_seg=wseg["d_seg"], d_pose=wseg["d_pose"]
        ),
        wjoint=ObjectiveTerms.from_distortions(
            d_seg=wjoint["d_seg"], d_pose=wjoint["d_pose"]
        ),
    )
    replay = critical_pose_to_seg_slope_ratio(
        wseg_d_seg=wseg["d_seg"],
        wseg_d_pose=wseg["d_pose"],
        wjoint_d_seg=wjoint["d_seg"],
        wjoint_d_pose=wjoint["d_pose"],
    )
    residual = abs(replay - gap.critical_ratio)
    provenance = build_provenance_for_macos_cpu_advisory(
        receipt_sha,
        RECEIPT if receipt_path is None else str(source),
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_ws1_two_warm_start_n600_term_gaps_20260724",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "pair_count": 600,
            "wseg_candidate_id": wseg["candidate_id"],
            "wjoint_candidate_id": wjoint["candidate_id"],
            "wseg_d_seg": wseg["d_seg"],
            "wseg_d_pose": wseg["d_pose"],
            "wjoint_d_seg": wjoint["d_seg"],
            "wjoint_d_pose": wjoint["d_pose"],
            "receipt_sha256": receipt_sha,
        },
        predicted_output={
            "critical_ratio_formula": "R_star = pose_debt / seg_advantage",
            "decision": (
                "adopt W_seg only when pose_progress/seg_advantage_erosion "
                "is at least R_star; Pose stall or Seg regression keeps W_joint"
            ),
        },
        empirical_output={
            "seg_advantage_score_units": gap.seg_advantage,
            "pose_debt_score_units": gap.pose_debt,
            "critical_ratio": gap.critical_ratio,
            "training_outcome": "UNMEASURED_SPEC_ONLY",
        },
        residual=residual,
        source_artifact=RECEIPT if receipt_path is None else str(source),
        measurement_method=(
            "n600 exact source-replayed V19C Seglex96 receiver through uint8 and "
            "frozen CPU-torch scorers; endpoint gaps measured, J5 smoke not launched"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=(
            "deterministic within-host endpoint replay; cross-host slope noise unmeasured"
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM WS1 bounded warm-start slope falsifier",
        one_line_summary=(
            "R* is the measured extra Pose debt of W_seg divided by its opening "
            "Seg advantage over W_joint."
        ),
        latex_form=(
            r"R^\star=\frac{\sqrt{10d_{p,s}}-\sqrt{10d_{p,j}}}"
            r"{100(d_{s,j}-d_{s,s})}"
        ),
        python_callable_module_path=(
            "tac.optimization.ddm_warm_start_slope_falsifier:"
            "critical_pose_to_seg_slope_ratio"
        ),
        domain_of_validity={
            "warm_starts": ["W_seg", "W_joint"],
            "vehicle": "J5 bounded smoke from exact measured WS1 candidates",
            "pair_count": 600,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "hard_rejects": ["Pose term stall or regression", "Seg term regression"],
            "metric_authority": (
                "margin-Fisher/rank4 Seg hyperplanes plus exact low-rank Pose "
                "quadratic; Euclidean identity-L2 control-only"
            ),
            "excluded": [
                "training result; no smoke was launched",
                "contest score or promotion",
                "CPU/CUDA equivalence",
                "cross-start optimizer-state equivalence",
            ],
            "score_claim": False,
        },
        units_in={
            "wseg_d_seg": "fraction",
            "wseg_d_pose": "mean_squared_pose_error",
            "wjoint_d_seg": "fraction",
            "wjoint_d_pose": "mean_squared_pose_error",
        },
        units_out={"critical_ratio": "dimensionless_score_term_ratio"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "endpoint_gap_ratio_replay": residual,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_warm_start_slope_falsifier."
            "evaluate_bounded_slope_window",
            ".omx/research/configs/ddm_ws1_j5_slope_falsifier_20260724.json",
        ),
        canonical_producers=(
            "tools/measure_ddm_ws1_seg_lexicographic_warmstart.py",
            RECEIPT,
        ),
        provenance=provenance,
    )


def populate_ddm_ws1_warm_start_slope_falsifier_v1(
    *,
    receipt_path: Path | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append WS1 through the locked canonical-equation registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_ws1_warm_start_slope_falsifier_v1(receipt_path)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "WS1 measured endpoint gap and typed J5 bounded-smoke falsifier; "
            "metric-active readback preregistered; training unlaunched; "
            "pointer unmoved; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_ddm_ws1_warm_start_slope_falsifier_v1",
    "populate_ddm_ws1_warm_start_slope_falsifier_v1",
]
