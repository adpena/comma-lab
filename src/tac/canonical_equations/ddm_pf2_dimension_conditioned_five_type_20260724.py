# SPDX-License-Identifier: MIT
"""Canonical metric-eligibility law for DDM PF2's five-type split.

Identity/Euclidean geometry is an instance-scoped control.  It cannot carry a
formulation verdict even when a final hard-scorer readback improves.  A
fixed-content coder comparison remains rate-verdict eligible only when strict
parse-back proves that both arms express identical semantic content, because
the distortion terms then cancel exactly.
"""

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
from tac.optimization.ddm_dimension_conditioned_two_type import (
    resolve_formulation_metric_disposition,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT = (
    ".omx/research/ddm_pf2_dimension_conditioned_two_type_20260724T020205Z/"
    "ddm_pf2_dimension_conditioned_two_type_receipt.json"
)
RECEIPT_SHA256 = "85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73"
EQUATION_ID = "ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1"
MEASUREMENT_UTC = "2026-07-24T02:02:05Z"


def formulation_is_admissible(
    *,
    metric_status: str,
    delta: float,
    identical_content_proven: bool,
) -> bool:
    """Return whether a negative native delta may carry a formulation verdict."""

    value = float(delta)
    if not math.isfinite(value):
        raise ValueError("delta must be finite")
    disposition = resolve_formulation_metric_disposition(
        metric_status,
        identical_content_proven=identical_content_proven,
    )
    return bool(disposition.verdict_eligible and value < 0.0)


def _load_receipt(path: Path | None = None) -> tuple[dict[str, Any], Path, str]:
    source = path or REPO_ROOT / RECEIPT
    payload_bytes = source.read_bytes()
    digest = hashlib.sha256(payload_bytes).hexdigest()
    payload = json.loads(payload_bytes)
    if (
        payload.get("schema")
        != "ddm_pf2_dimension_conditioned_two_type_measurement.v1"
        or payload.get("score_claim") is not False
        or payload.get("pointer_moved") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise ValueError("PF2 receipt authority or schema differs")
    if path is None and digest != RECEIPT_SHA256:
        raise ValueError(f"canonical PF2 receipt sha256 differs: {digest}")
    return payload, source, digest


def build_ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1(
    receipt_path: Path | None = None,
) -> CanonicalEquation:
    """Build the law and rederive all three formulation eligibility rows."""

    payload, source, receipt_sha = _load_receipt(receipt_path)
    formulations = payload["family_adjudication"]["formulations"]
    if len(formulations) != 3:
        raise ValueError("PF2 receipt must contain exactly three formulations")
    rederived = []
    for row in formulations:
        identical = row["metric_status"] == "IDENTICAL_CONTENT_CODER_CONTROL"
        accepted = formulation_is_admissible(
            metric_status=row["metric_status"],
            delta=row["delta"],
            identical_content_proven=identical,
        )
        if accepted is not row["accepted"]:
            raise ValueError(
                f"stored formulation verdict differs: {row['formulation_id']}"
            )
        rederived.append(accepted)
    family = payload["family_adjudication"]
    if (
        sum(row["verdict_eligible"] for row in formulations)
        != family["eligible_formulation_count"]
        or sum(rederived) != family["accepted_formulation_count"]
        or family["ineligible_formulation_count"] != 1
    ):
        raise ValueError("PF2 family eligibility counts differ")
    provenance = build_provenance_for_macos_cpu_advisory(
        receipt_sha,
        RECEIPT if receipt_path is None else str(source),
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_pf2_five_type_three_formulation_n600_20260724",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "pair_count": 600,
            "representation_types": [
                "SKELETON",
                "CONNECTION",
                "FIBER",
                "GAUGE",
                "RESIDUAL",
            ],
            "formulation_count": 3,
            "receipt_sha256": receipt_sha,
        },
        predicted_output={
            "identity_metric": "control_only_not_verdict_eligible",
            "identical_content_coder": (
                "rate_verdict_eligible_when_strict_parseback_closes"
            ),
            "complete_family_verdict": "requires_all_three_formulations_eligible",
        },
        empirical_output={
            "verdict": payload["verdict"],
            "verdict_scope": payload["verdict_scope"],
            "eligible_formulation_count": family["eligible_formulation_count"],
            "ineligible_formulation_count": family["ineligible_formulation_count"],
            "accepted_formulation_count": family["accepted_formulation_count"],
            "f1_delta_bytes": formulations[0]["delta"],
            "f2_identity_control_delta_S": formulations[1]["delta"],
            "f2_verdict_eligible": formulations[1]["verdict_eligible"],
            "f3_delta_bytes": formulations[2]["delta"],
            "all_routes_held": all(
                row["decision"].startswith("HOLD")
                for row in payload["route_table"]
            ),
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=RECEIPT if receipt_path is None else str(source),
        measurement_method=(
            "strict n600 frozen CPU-torch F2 control readback plus exact semantic "
            "parse-back rate controls for PF1 and event-skeleton x xi-fiber; "
            "metric-law eligibility rederived from every formulation row"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=(
            "deterministic within-host replay; contest CPU/CUDA unmeasured"
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM PF2 metric-eligible five-type formulation adjudication",
        one_line_summary=(
            "A negative formulation delta is admissible only under measured "
            "scorer geometry or strict identical-content rate cancellation."
        ),
        latex_form=(
            r"\operatorname{admit}_i=\mathbf{1}[\Delta_i<0]"
            r"\mathbf{1}[g_i=g_{\rm scorer}\ \lor\ C_i\equiv C_0];\quad "
            r"g_i=I\Rightarrow\operatorname{admit}_i=0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations."
            "ddm_pf2_dimension_conditioned_five_type_20260724:"
            "formulation_is_admissible"
        ),
        domain_of_validity={
            "vehicle": "DDM PF2 over settled V19C/MENU1, PF1, G4, #574, and DR2b",
            "representation_types": [
                "SKELETON",
                "CONNECTION",
                "FIBER",
                "GAUGE",
                "RESIDUAL",
            ],
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "included": [
                "strict identical-content coder controls",
                "identity-metric row retained as an instance-scoped control",
            ],
            "excluded": [
                "metric-active F2 PoseNet quadratic projection",
                "bucket-complete margin-Fisher lambda pricing",
                "dual-metric readback",
                "contest score, promotion, or complete family verdict",
            ],
            "verdict_scope": payload["verdict_scope"],
            "score_claim": False,
        },
        units_in={
            "metric_status": "categorical_status",
            "delta": "native_formulation_units",
            "identical_content_proven": "boolean",
        },
        units_out={"admissible": "boolean"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"stored_eligibility_replay": 0.0},
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.measure_ddm_pf2_dimension_conditioned_two_type",
            "tac.bit_allocator",
            "tac.cathedral_autopilot",
        ),
        canonical_producers=(
            "tools/measure_ddm_pf2_dimension_conditioned_two_type.py",
            RECEIPT,
        ),
        provenance=provenance,
    )


def populate_ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1(
    *,
    receipt_path: Path | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the measured law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = (
        build_ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1(
            receipt_path
        )
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DDM PF2 n600 metric-law adjudication; F2 identity control excluded; "
            "F1 exact rate control survives; F3 exact rate control negative; "
            "pointer unmoved; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "RECEIPT_SHA256",
    "build_ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1",
    "formulation_is_admissible",
    "populate_ddm_pf2_metric_eligible_five_type_formulation_adjudication_v1",
]
