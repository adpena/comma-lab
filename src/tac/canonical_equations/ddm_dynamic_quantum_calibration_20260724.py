# SPDX-License-Identifier: MIT
"""Dynamic realized-quantum calibration for DDM receiver coordinates."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import register_evaluator
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID: Final = "dynamic_quantum_calibration_v1"
REPO: Final = Path(__file__).resolve().parents[3]
RECEIPT: Final = REPO / (
    ".omx/research/ddm_ms7_receiver_edges_and_25bucket_reach_20260724T172249Z/ddm_ms7_receiver_edges_receipt.json"
)
DEFAULT_LATTICE: Final = (1, 2, 4, 8, 16)
MEASUREMENT_UTC: Final = "2026-07-24T18:00:27Z"


def dynamic_quantum_calibration(
    *,
    composite_r_gain: float,
    realized_uint8_deadzone: float,
    lattice: Sequence[int] = DEFAULT_LATTICE,
    validity_radius: int,
) -> dict[str, Any]:
    """Derive and validity-gate ``k*=ceil(q/(2|g|))`` on a typed lattice."""

    gain = abs(float(composite_r_gain))
    deadzone = float(realized_uint8_deadzone)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("composite_r_gain must be finite and nonzero")
    if not math.isfinite(deadzone) or deadzone <= 0.0:
        raise ValueError("realized_uint8_deadzone must be finite and positive")
    if isinstance(validity_radius, bool) or not isinstance(validity_radius, int) or validity_radius <= 0:
        raise ValueError("validity_radius must be a positive integer")
    snapped_lattice = tuple(int(value) for value in lattice)
    if (
        not snapped_lattice
        or any(value <= 0 for value in snapped_lattice)
        or tuple(sorted(set(snapped_lattice))) != snapped_lattice
    ):
        raise ValueError("lattice must be sorted, unique, and positive")
    unsnapped = math.ceil(deadzone / (2.0 * gain))
    predicted = next((value for value in snapped_lattice if value >= unsnapped), None)
    inside = predicted is not None and predicted <= validity_radius
    return {
        "formula": "ceil(q/(2*abs(g))) snapped upward to the declared lattice",
        "composite_r_gain": gain,
        "realized_uint8_deadzone": deadzone,
        "unsnapped_k_star": unsnapped,
        "predicted_k_star": predicted,
        "validity_radius": validity_radius,
        "selected_k_star": predicted if inside else None,
        "inside_measured_validity_radius": inside,
        "status": (
            "CALIBRATED_INSIDE_MEASURED_VALIDITY_RADIUS"
            if inside
            else "NULL_OUTSIDE_MEASURED_VALIDITY_RADIUS"
            if predicted is not None
            else "NULL_LATTICE_EXHAUSTED"
        ),
    }


def _evaluate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "composite_r_gain",
        "realized_uint8_deadzone",
        "lattice",
        "validity_radius",
    }
    if set(inputs) != required:
        raise ValueError("dynamic quantum calibration inputs differ")
    return dynamic_quantum_calibration(**dict(inputs))


register_evaluator(EQUATION_ID, _evaluate)


def build_dynamic_quantum_calibration(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    payload = json.loads(source_receipt.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "ddm_ms7_receiver_edges_receipt.v1"
        or payload.get("pointer_moved") is not False
    ):
        raise ValueError("dynamic calibration receipt custody differs")
    pf3 = payload.get("pf3")
    dynamic = pf3.get("dynamic_quantum_calibration") if isinstance(pf3, Mapping) else None
    calibration = dynamic.get("calibration") if isinstance(dynamic, Mapping) else None
    predicted_vs_realized = dynamic.get("predicted_vs_realized") if isinstance(dynamic, Mapping) else None
    control = pf3.get("control_identity") if isinstance(pf3, Mapping) else None
    if (
        not isinstance(calibration, Mapping)
        or not isinstance(predicted_vs_realized, Mapping)
        or not isinstance(control, Mapping)
        or calibration.get("selected_k_star") is None
        or not isinstance(predicted_vs_realized.get("realized_minimum_nonzero_uint8_level"), int)
        or not isinstance(predicted_vs_realized.get("realized_deadzone_crossed"), bool)
    ):
        raise ValueError("dynamic calibration empirical anchor is malformed")
    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Append a new exact receiver probe whenever a coordinate's measured "
            "gain, uint8 deadzone, or family-specific validity radius changes."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_threads4",
        captured_at_utc=MEASUREMENT_UTC,
    )
    predicted_crossed = bool(predicted_vs_realized.get("predicted_deadzone_crossed"))
    realized_crossed = bool(predicted_vs_realized["realized_deadzone_crossed"])
    anchor = EmpiricalAnchor(
        anchor_id="ddm_ms7_pair523_class_birth_dynamic_quantum_20260724",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "pair_id": int(control["pair_id"]),
            "bucket_id": str(control["bucket_id"]),
            "composite_r_gain": float(calibration["composite_r_gain"]),
            "realized_uint8_deadzone": float(calibration["realized_uint8_deadzone"]),
            "lattice": list(DEFAULT_LATTICE),
            "validity_radius": int(calibration["validity_radius"]),
        },
        predicted_output={
            "selected_k_star": int(calibration["selected_k_star"]),
            "deadzone_crossed": predicted_crossed,
        },
        empirical_output={
            "selected_k_star": int(calibration["selected_k_star"]),
            "minimum_nonzero_uint8_level": int(predicted_vs_realized["realized_minimum_nonzero_uint8_level"]),
            "deadzone_crossed": realized_crossed,
        },
        residual=0.0 if predicted_crossed == realized_crossed else 1.0,
        source_artifact=str(source_receipt),
        measurement_method=(
            "one scorer-recursive RG3 class-birth coordinate compiled into the SHA-bound "
            "V19C nested receiver, realized through uint8/R, and replayed exactly"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=("within-run deterministic replay exact; cross-host variance not measured"),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM dynamic receiver-quantum calibration",
        one_line_summary=(
            "Choose the smallest measured-valid lattice amplitude predicted to cross "
            "the realized uint8 deadzone for the coordinate's composite-R gain."
        ),
        latex_form=(
            r"k_i^\star=\operatorname{snap}_{\mathcal L}^{\uparrow}"
            r"\left(\left\lceil\frac{q_i}{2|g_i|}\right\rceil\right),\quad"
            r"k_i^\star\le r_i^{\rm valid}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_dynamic_quantum_calibration_20260724:dynamic_quantum_calibration"
        ),
        domain_of_validity={
            "gain": "MS4D direct composite-R adjoint L2 gain for the exact pair/bucket block",
            "deadzone": "realized uint8 one-LSB deadzone from #580/#532",
            "lattice": [1, 2, 4, 8, 16],
            "lattice_provenance": ("G2f bidirectional dyadic amplitude ladder; whole-LSB knee at 1"),
            "validity_radius": ("family-specific measured radius; never inferred across RG3 families"),
            "null_policy": ("outside validity or exhausted lattice remains NULL; no clipped fake price"),
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        },
        units_in={
            "composite_r_gain": "realized uint8 output units per coordinate quantum",
            "realized_uint8_deadzone": "uint8 levels",
            "lattice": "coordinate quanta",
            "validity_radius": "coordinate quanta",
        },
        units_out={
            "predicted_k_star": "coordinate quanta or NULL",
            "selected_k_star": "measured-valid coordinate quanta or NULL",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "deadzone_crossing_binary_mismatch": anchor.residual,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_ms4d_waterfill_admission",
            "ddm_tolerance_capped_min_score_waterfill_v1",
            "DDM MS7 R1 coordinate admission",
        ),
        canonical_producers=(
            "MS4D direct rank-4 composite-R metric bundle",
            "G2f bidirectional amplitude secants",
            "v16/v17 realized trust-radius receipts",
        ),
        provenance=provenance,
    )


def populate_dynamic_quantum_calibration(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_dynamic_quantum_calibration(source_receipt=source_receipt)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "MS7 dynamic quantum calibration; one exact predicted-vs-realized anchor; "
            "R1/R2 prices remain NULL outside measured validity; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "DEFAULT_LATTICE",
    "EQUATION_ID",
    "build_dynamic_quantum_calibration",
    "dynamic_quantum_calibration",
    "populate_dynamic_quantum_calibration",
]
