# SPDX-License-Identifier: MIT
"""Canonical joint-action law for the measured DDM MC1 hood reassertion."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT = (
    ".omx/research/ddm_mc1_hood_static_reassert_20260724T003346Z/"
    "ddm_mc1_hood_static_reassert_receipt.json"
)
EQUATION_ID = "ddm_mc1_static_hood_reassert_joint_action_v1"
MEASUREMENT_UTC = "2026-07-24T01:11:00Z"


def evaluate_hood_reassert_joint_delta(
    *,
    parent_d_seg: float,
    parent_d_pose: float,
    parent_bytes: int,
    child_d_seg: float,
    child_d_pose: float,
    child_bytes: int,
) -> float:
    """Return child-minus-parent contest action; negative alone is admissible."""

    values = (
        float(parent_d_seg),
        float(parent_d_pose),
        float(child_d_seg),
        float(child_d_pose),
    )
    if any(not math.isfinite(v) or v < 0.0 for v in values):
        raise ValueError("distortions must be finite and non-negative")
    if int(parent_bytes) < 0 or int(child_bytes) < 0:
        raise ValueError("byte counts must be non-negative")

    def action(d_seg: float, d_pose: float, bytes_: int) -> float:
        return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * bytes_ / 37_545_489

    return action(values[2], values[3], int(child_bytes)) - action(
        values[0], values[1], int(parent_bytes)
    )


def _load_receipt(path: Path | None = None) -> tuple[dict[str, Any], Path, str]:
    source = path or REPO_ROOT / RECEIPT
    payload_bytes = source.read_bytes()
    payload = json.loads(payload_bytes)
    if (
        payload.get("schema") != "ddm_mc1_hood_static_reassert_measurement.v1"
        or payload.get("score_claim") is not False
        or payload.get("pointer_moved") is not False
    ):
        raise ValueError("MC1 receipt authority or schema differs")
    return payload, source, hashlib.sha256(payload_bytes).hexdigest()


def build_ddm_mc1_static_hood_reassert_joint_action_v1(
    receipt_path: Path | None = None,
) -> CanonicalEquation:
    """Build the measured law and rederive every stored joint delta."""

    payload, source, receipt_sha = _load_receipt(receipt_path)
    parent = payload["input_custody"]["menu1_parent"]
    residuals = []
    for row in payload["candidates"]:
        derived = evaluate_hood_reassert_joint_delta(
            parent_d_seg=parent["d_seg"],
            parent_d_pose=parent["d_pose"],
            parent_bytes=parent["archive_bytes"],
            child_d_seg=row["d_seg"],
            child_d_pose=row["d_pose"],
            child_bytes=row["archive_bytes"],
        )
        residuals.append(abs(derived - row["delta_advisory_objective_vs_parent"]))
    residual = max(residuals)
    provenance = build_provenance_for_macos_cpu_advisory(
        receipt_sha,
        RECEIPT if receipt_path is None else str(source),
        captured_at_utc=MEASUREMENT_UTC,
    )
    winner = payload["pool_winner"]
    anchor = EmpiricalAnchor(
        anchor_id="ddm_mc1_three_support_n600_joint_reassert_20260724",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "pair_count": 600,
            "parent_candidate_id": parent["row_id"],
            "support_formulations": [
                row["support_kind"] for row in payload["candidates"]
            ],
            "frame0_byte_identical": True,
            "receipt_sha256": receipt_sha,
        },
        predicted_output={
            "admission_law": "Delta S = S(child)-S(parent); admit iff Delta S < 0",
            "staticity_prior": "single static support should dominate per-frame rate",
        },
        empirical_output={
            "verdict": payload["verdict"],
            "verdict_scope": payload["verdict_scope"],
            "pool_positive": payload["pool_positive"],
            "best_support_kind": winner["support_kind"],
            "best_delta_S": winner["delta_advisory_objective_vs_parent"],
            "best_MyCar_delta_errors_realized": winner["per_class"]["MyCar"][
                "delta_errors_realized"
            ],
            "best_d_pose": winner["d_pose"],
        },
        residual=residual,
        source_artifact=RECEIPT if receipt_path is None else str(source),
        measurement_method=(
            "exact V19C receiver; exact MENU1 frame1 winner hash; post-paint base-byte "
            "hood reassert; n600 frozen CPU-torch SegNet and official PoseNet two-frame path"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=(
            "deterministic within-host replay measured; cross-host noise floor unmeasured"
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM MC1 static hood reassert joint-action admission",
        one_line_summary=(
            "A post-paint hood reassert is admissible only when its exact Seg, Pose, and "
            "counted-byte child-minus-parent action is negative."
        ),
        latex_form=(
            r"\Delta S=100(d_{\rm seg}'-d_{\rm seg})+"
            r"\sqrt{10d_{\rm pose}'}-\sqrt{10d_{\rm pose}}+"
            r"\frac{25(B'-B)}{37{,}545{,}489}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_mc1_hood_static_reassert_20260724:"
            "evaluate_hood_reassert_joint_delta"
        ),
        domain_of_validity={
            "vehicle": "V19C plus exact MENU1 frame1 paint winner",
            "operation": "restore base V19C frame1 bytes after paint on hood support",
            "support_formulations": [
                "single_static_stored",
                "per_frame_stored",
                "decoder_semantic_free",
            ],
            "receiver": "uint8 camera bytes -> frozen SegNet and official PoseNet YUV6",
            "pair_count": 600,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "excluded": [
                "contest score or promotion",
                "static-field family closure",
                "additive composition of same-pool supports",
                "pose-stat-preserving solved hood field",
            ],
            "verdict_scope": payload["verdict_scope"],
            "score_claim": False,
        },
        units_in={
            "parent_d_seg": "fraction",
            "parent_d_pose": "mean_squared_pose_error",
            "parent_bytes": "bytes",
            "child_d_seg": "fraction",
            "child_d_pose": "mean_squared_pose_error",
            "child_bytes": "bytes",
        },
        units_out={"delta_S": "contest_score_units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"stored_joint_delta_replay": residual},
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.measure_ddm_mc1_hood_static_reassert",
            "tac.optimization.ddm_hood_static_reassert.reassert_frame1",
        ),
        canonical_producers=(
            "tools/measure_ddm_mc1_hood_static_reassert.py",
            RECEIPT,
        ),
        provenance=provenance,
    )


def populate_ddm_mc1_static_hood_reassert_joint_action_v1(
    *,
    receipt_path: Path | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the measured law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_mc1_static_hood_reassert_joint_action_v1(receipt_path)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DDM MC1 n600 static-hood reassert; Seg recovery but Pose coupling rejects "
            "the measured instance; pointer unmoved; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_ddm_mc1_static_hood_reassert_joint_action_v1",
    "evaluate_hood_reassert_joint_delta",
    "populate_ddm_mc1_static_hood_reassert_joint_action_v1",
]
