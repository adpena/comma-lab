# SPDX-License-Identifier: MIT
"""Canonical realized-validity ratio for DDM v17 integer trust regions."""

from __future__ import annotations

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
from tac.provenance.builders import build_provenance_for_research_sidecar

REPO = Path(__file__).resolve().parents[3]
UTC = "2026-07-23T03:40:00Z"
AXIS = "[macOS-CPU frozen-scorer advisory]"
RECEIPT = (
    ".omx/research/ddm_a1_bounded_collateral_realized_n64_20260723T031500Z/"
    "ddm_a1_bounded_collateral_realized_receipt.json"
)
EQUATION_ID = "ddm_v17_realized_validity_ratio_uint8_v1"


def realized_validity_ratio(predicted_reduction: float, realized_reduction: float) -> float:
    """Return rho for a strictly positive local-model reduction."""

    predicted = float(predicted_reduction)
    realized = float(realized_reduction)
    if not math.isfinite(predicted) or not math.isfinite(realized) or predicted <= 0.0:
        raise ValueError("validity ratio requires finite reductions and predicted_reduction > 0")
    return realized / predicted


def _receipt(path: Path | None = None) -> tuple[dict[str, Any], str]:
    source = path or REPO / RECEIPT
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "ddm_a1_bounded_collateral_realized_receipt.v1":
        raise ValueError("DDM v17 validity receipt schema drifted")
    if payload.get("score_claim") is not False or payload.get("evidence_axis") != AXIS:
        raise ValueError("DDM v17 validity receipt authority firewall drifted")
    if payload.get("validity_radius", {}).get("stable_law_registration_eligible") is not True:
        raise ValueError("DDM v17 receipt does not carry enough finite rho anchors")
    return payload, str(source if path is not None else RECEIPT)


def build_ddm_v17_realized_validity_ratio_uint8_v1(
    receipt_path: Path | None = None,
) -> CanonicalEquation:
    payload, source = _receipt(receipt_path)
    raw = payload["validity_radius"]["raw_rows"]
    finite = [row for row in raw if row["rho"] is not None and float(row["predicted_reduction"]) > 0.0]
    residual = max(
        abs(
            realized_validity_ratio(
                float(row["predicted_reduction"]),
                float(row["realized_reduction"]),
            )
            - float(row["rho"])
        )
        for row in finite
    )
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=RECEIPT,
        reactivation_criteria=(
            "Recalibrate on a new accepted realized point, a clean KKT solve, a wider exact "
            "candidate family, or contest-axis replay; do not infer a universal trust radius."
        ),
        measurement_axis=AXIS,
        hardware_substrate="darwin_arm64_cpu_torch",
        captured_at_utc=UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="ddm_v17_probe_a_three_basis_uint8_curve_20260723",
        measurement_utc=UTC,
        inputs={
            "basis_count": 3,
            "bases": ["1x1_rowband_control", "2x2_contextual", "boundary_normal_2x2"],
            "lattice_quanta": [1, 2, 4, 8],
            "solve_exact_calls": 12,
            "finite_rho_count": len(finite),
            "pair_ids": payload["typed_config"]["pair_ids"],
            "typed_config_sha256": payload["typed_config_sha256"],
        },
        predicted_output={
            "law": "rho(q,b) = Delta_F_realized(q,b) / Delta_F_predicted(q,b)",
            "trust_rule": "rho<0 hard-shrinks; rho never confers acceptance",
        },
        empirical_output={
            "curve_by_lattice_quanta": payload["validity_radius"]["curve_by_lattice_quanta"],
            "verdict": payload["verdict"],
            "verdict_scope": payload["verdict_scope"],
            "accepted_iterations": payload["accepted_iterations"],
            "probe_a_admitted": payload["probe_a_admitted"],
        },
        residual=residual,
        source_artifact=source,
        measurement_method=(
            "fresh receiver-through-frozen-scorer M; basis-projected KKT/Babai or explicit "
            "M-preconditioned lattice fallback; exact uint8/R replay for every point"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance="within-run deterministic exact replay; cross-host floor not measured",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM v17 receiver-realized uint8 validity ratio",
        one_line_summary=(
            "For one basis-conditioned integer proposal, rho is realized Fisher-margin-debt "
            "reduction divided by positive affine-model reduction; hard acceptance remains separate."
        ),
        latex_form=(
            r"\rho(q,b)=\frac{D_F(m(u))-D_F(m(R(u+P_bq)))}"
            r"{D_F(m(u))-D_F(m(u)+M P_bq)}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_v17_validity_radius_law_20260723:realized_validity_ratio"
        ),
        domain_of_validity={
            "vehicle": "DDM v17 / Probe A initial realized point",
            "basis_conditioned": True,
            "integer_lattice_only": True,
            "receiver": "counted paint -> uint8 -> exact R -> frozen SegNet/PoseNet",
            "included_quanta": [1, 2, 4, 8],
            "excluded": [
                "universal basis-independent validity radius",
                "acceptance from rho",
                "n64 or n600 score claim",
                "clean-KKT claim",
                "contextual template family closure",
            ],
            "verdict_scope": payload["verdict_scope"],
            "score_claim": False,
        },
        units_in={
            "predicted_reduction": "Fisher_margin_debt",
            "realized_reduction": "Fisher_margin_debt",
        },
        units_out={"rho": "dimensionless"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"stored_ratio_formula_replay": residual},
        last_calibration_utc=UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_ddm_a1_bounded_collateral_realized",
            "tac.optimization.iterative_realized_trust_region.update_trust_radius",
        ),
        canonical_producers=(
            "tools/probe_ddm_a1_bounded_collateral_realized.py",
            RECEIPT,
        ),
        provenance=provenance,
    )


def populate_ddm_v17_realized_validity_ratio_uint8_v1(
    *,
    receipt_path: Path | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the measured law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_v17_realized_validity_ratio_uint8_v1(receipt_path)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DDM v17 Probe-A measured advisory basis-conditioned validity curve; "
            "initial-grid plateau; pointer unmoved; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_ddm_v17_realized_validity_ratio_uint8_v1",
    "populate_ddm_v17_realized_validity_ratio_uint8_v1",
    "realized_validity_ratio",
]
