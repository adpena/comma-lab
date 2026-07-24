# SPDX-License-Identifier: MIT
"""Canonical laws for the DDM MS2 typed quotient solve.

The equations are structural admission laws, not a claim that the n600 solve
ran.  Their source receipt is expected to be fail-closed until measured
scorer-coordinate Gram/Hessian, PF2 atlas, Pose-tube, and receiver custody are
all present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.optimization.ddm_typed_quotient_solve import (
    RATE_SCORE_PER_BYTE,
    MeasuredScorerGeometry,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

if TYPE_CHECKING:
    from tac.provenance.contract import Provenance

REPO: Final = Path(__file__).resolve().parents[3]
RECEIPT: Final = ".omx/research/ddm_ms2_typed_quotient_solve_20260724_receipt.json"
RECEIPT_SCHEMA: Final = "ddm_ms2_typed_quotient_solve_repo_receipt.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
POINTER: Final = "0.1910828242 [contest-CPU]"
PF2R_CONSUMER: Final = "pf2r.metric_active_three_formulation_rerun"
PF2R_BLOCKER_ID: Final = (
    "PF2_METRIC_ACTIVE_THREE_FORMULATION_ADJUDICATION_INCOMPLETE"
)

QUOTIENT_EQUATION_ID: Final = "ddm_ms2_visible_quotient_gauge_zero_v1"
METRIC_EQUATION_ID: Final = "ddm_ms2_scorer_metric_second_order_action_v1"
DUAL_EQUATION_ID: Final = "ddm_ms2_typed_block_dual_exchange_v1"
QUANTUM_EQUATION_ID: Final = "ddm_ms2_effective_quantum_admission_v1"
CODER_RACE_EQUATION_ID: Final = "ddm_ms2_skeleton_fiber_coder_race_v1"
EQUATION_IDS: Final = (
    QUOTIENT_EQUATION_ID,
    METRIC_EQUATION_ID,
    DUAL_EQUATION_ID,
    QUANTUM_EQUATION_ID,
    CODER_RACE_EQUATION_ID,
)


def visible_quotient_counted_bytes(
    *,
    visible_counted_bytes: int,
    gauge_counted_bytes: int,
) -> int:
    """Return payload bytes only when the dropped gauge costs exactly zero."""

    for value, field in (
        (visible_counted_bytes, "visible_counted_bytes"),
        (gauge_counted_bytes, "gauge_counted_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if gauge_counted_bytes != 0:
        raise ValueError("GAUGE bytes must be zero by quotient construction")
    return visible_counted_bytes


def scorer_metric_rate_action(
    delta: np.ndarray,
    *,
    geometry: MeasuredScorerGeometry,
    counted_bytes: int,
) -> float:
    """Return ``1/2 delta^T H_score delta + 25 B/N``."""

    value = np.asarray(delta, dtype=np.float64)
    if value.shape != (geometry.dimension,) or not np.isfinite(value).all():
        raise ValueError("delta must be a finite scorer-coordinate vector")
    if isinstance(counted_bytes, bool) or not isinstance(counted_bytes, int) or counted_bytes < 0:
        raise ValueError("counted_bytes must be a nonnegative integer")
    return float(0.5 * value @ geometry.second_order_metric @ value + RATE_SCORE_PER_BYTE * counted_bytes)


def typed_block_exchange_rate(
    *,
    score_gain: float,
    exact_delta_bytes: int,
    kkt_dual: float,
    measured_source_sha256: str,
) -> dict[str, float | int | str]:
    """Return one measured, unpooled block dual/exchange row."""

    gain = float(score_gain)
    dual = float(kkt_dual)
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("score_gain must be finite and nonnegative")
    if isinstance(exact_delta_bytes, bool) or not isinstance(exact_delta_bytes, int) or exact_delta_bytes <= 0:
        raise ValueError("exact_delta_bytes must be a positive integer")
    if not math.isfinite(dual):
        raise ValueError("kkt_dual must be measured and finite")
    if (
        not isinstance(measured_source_sha256, str)
        or len(measured_source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in measured_source_sha256)
    ):
        raise ValueError("measured_source_sha256 must be a lowercase SHA-256")
    return {
        "score_gain": gain,
        "exact_delta_bytes": exact_delta_bytes,
        "score_gain_per_byte": gain / exact_delta_bytes,
        "kkt_dual": dual,
        "measured_source_sha256": measured_source_sha256,
    }


def effective_quantum(
    *,
    uint8_step: float,
    scorer_sensitivity: float,
) -> float:
    """Return the only admissible per-dimension solve quantum."""

    step = float(uint8_step)
    sensitivity = float(scorer_sensitivity)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("uint8_step must be finite and positive")
    if not math.isfinite(sensitivity) or sensitivity <= 0.0:
        raise ValueError("scorer_sensitivity must be measured, finite, and positive")
    return step * sensitivity


def skeleton_fiber_coder_race(
    *,
    skeleton_counted_bytes: int,
    fiber_counted_bytes: int,
    semantic_parseback_exact: bool,
) -> str:
    """Choose a vocabulary role only after an exact real-coder race."""

    for value, field in (
        (skeleton_counted_bytes, "skeleton_counted_bytes"),
        (fiber_counted_bytes, "fiber_counted_bytes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if semantic_parseback_exact is not True:
        raise ValueError("SKELETON-vs-FIBER race requires exact semantic parse-back")
    return (
        "SKELETON"
        if skeleton_counted_bytes <= fiber_counted_bytes
        else "FIBER"
    )


def derivation_edges() -> tuple[tuple[str, str, str], ...]:
    """Return the EQUATIONS leg consumed by the DAG and solver."""

    return (
        ("evaluate_L0_discrete_quotient", "representation_type", "SKELETON"),
        ("evaluate_L0_continuous_quadratic", "representation_type", "FIBER"),
        ("evaluate_L1_R_nullspace", "quotient_restriction", "GAUGE_zero_bytes"),
        ("rank4_margin_plus_pose6", "measured_second_order_metric", "typed_KKT_and_CVP"),
        ("uint8_step_x_sensitivity", "effective_quantum", "typed_trust_radius"),
        ("argmax_cell", "alternates_with", "within_cell_lattice"),
        ("within_cell_lattice", "alternates_with", "real_coder_price"),
        ("per_block_KKT_dual", "unpooled_exchange_rate", "train_decision_SOLVE"),
        ("SKELETON_vs_FIBER_real_coder", "measured_race", "vocabulary_role"),
    )


def _equation(
    *,
    equation_id: str,
    name: str,
    summary: str,
    latex: str,
    callable_name: str,
    units_in: dict[str, str],
    units_out: dict[str, str],
    provenance: Provenance,
    calibration_utc: str,
) -> CanonicalEquation:
    return CanonicalEquation(
        equation_id=equation_id,
        name=name,
        one_line_summary=summary,
        latex_form=latex,
        python_callable_module_path=(f"tac.canonical_equations.ddm_ms2_typed_quotient_solve_20260724:{callable_name}"),
        domain_of_validity={
            "vehicle": "DDM MS2 typed visible-quotient solve",
            "required": [
                "SHA-bound measured scorer-native geometry",
                "exact composite-R second order and inner Jacobian",
                "PF2-reconciled ten-pair typed atlas",
                "Pose tube active inside member selection",
                "exact real-coder parse-back",
            ],
            "excluded": [
                "identity-Euclidean verdicts",
                "pooled or imputed duals",
                "unlanded sister snapshots",
                "score, promotion, or lattice-family claims",
            ],
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
            "pointer": POINTER,
            "named_downstream_consumers": (
                "train-decision-table SOLVE column",
                PF2R_CONSUMER,
            ),
            "pf2r_blocker_id": PF2R_BLOCKER_ID,
        },
        units_in=units_in,
        units_out=units_out,
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=calibration_utc,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_typed_quotient_solve",
            "tools.measure_ddm_ms2_typed_quotient_solve",
            PF2R_CONSUMER,
        ),
        canonical_producers=(
            "tools/measure_ddm_ms2_typed_quotient_solve.py",
            RECEIPT,
        ),
        provenance=provenance,
    )


def build_ddm_ms2_typed_quotient_equations(
    *,
    provenance: Provenance,
    calibration_utc: str,
) -> tuple[CanonicalEquation, ...]:
    """Build all five registered structural laws from one final receipt."""

    rows = (
        _equation(
            equation_id=QUOTIENT_EQUATION_ID,
            name="DDM MS2 visible quotient and zero-byte gauge",
            summary="Only scorer-visible quotient coordinates are solved; deterministic preimage/free fill realizes the dropped gauge at exactly zero payload bytes.",
            latex=r"B_{\mathrm{total}}=B_{\mathrm{visible}}+0_{\mathrm{gauge}}",
            callable_name="visible_quotient_counted_bytes",
            units_in={"visible_counted_bytes": "bytes", "gauge_counted_bytes": "bytes"},
            units_out={"total_counted_bytes": "bytes"},
            provenance=provenance,
            calibration_utc=calibration_utc,
        ),
        _equation(
            equation_id=METRIC_EQUATION_ID,
            name="DDM MS2 scorer-metric second-order rate action",
            summary="KKT, CVP, trust, ranking, and dictionary updates use the measured rank-4/Pose6 composite-R geometry; identity is control-only.",
            latex=r"\mathcal A(\delta,B)=\frac12\delta^\top H_{\mathrm{score},R}\delta+\frac{25}{N}B",
            callable_name="scorer_metric_rate_action",
            units_in={"delta": "scorer_coordinates", "counted_bytes": "bytes"},
            units_out={"action": "score_units"},
            provenance=provenance,
            calibration_utc=calibration_utc,
        ),
        _equation(
            equation_id=DUAL_EQUATION_ID,
            name="DDM MS2 unpooled typed-block dual exchange",
            summary="Each typed block emits its measured KKT dual and exact score gain per byte; missing values remain unavailable and are never pooled.",
            latex=r"\rho_b=\Delta S_b/\Delta B_b,\quad \lambda_b\ \mathrm{unpooled}",
            callable_name="typed_block_exchange_rate",
            units_in={
                "score_gain": "score_units",
                "exact_delta_bytes": "bytes",
                "kkt_dual": "score_units_per_constraint",
            },
            units_out={"score_gain_per_byte": "score_units_per_byte"},
            provenance=provenance,
            calibration_utc=calibration_utc,
        ),
        _equation(
            equation_id=QUANTUM_EQUATION_ID,
            name="DDM MS2 per-dimension effective quantum",
            summary="Every visible dimension is admitted against its measured uint8-step times scorer sensitivity, with tolerance knees swept explicitly.",
            latex=r"q_i^{\mathrm{eff}}=\Delta u_i^{\mathrm{uint8}}\,s_i^{\mathrm{score}}",
            callable_name="effective_quantum",
            units_in={"uint8_step": "uint8_units", "scorer_sensitivity": "score_per_uint8"},
            units_out={"effective_quantum": "score_units"},
            provenance=provenance,
            calibration_utc=calibration_utc,
        ),
        _equation(
            equation_id=CODER_RACE_EQUATION_ID,
            name="DDM MS2 per-stratum skeleton-fiber coder race",
            summary="A factor enters vocabulary only after exact semantic parse-back and a measured real-coder SKELETON-versus-FIBER byte race.",
            latex=r"t_b=\arg\min_{t\in\{\mathrm{SKELETON},\mathrm{FIBER}\}} B_{b,t}",
            callable_name="skeleton_fiber_coder_race",
            units_in={"skeleton_counted_bytes": "bytes", "fiber_counted_bytes": "bytes"},
            units_out={"winner": "representation_type"},
            provenance=provenance,
            calibration_utc=calibration_utc,
        ),
    )
    if tuple(row.equation_id for row in rows) != EQUATION_IDS:
        raise RuntimeError("MS2 canonical equation ordering drifted")
    return rows


def populate_ddm_ms2_typed_quotient_equations(
    *,
    receipt_path: Path | None = None,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> tuple[CanonicalEquation, ...]:
    """Validate the final receipt, then append all laws through the lock API."""

    from tac.canonical_equations.registry import register_canonical_equation

    source = receipt_path or REPO / RECEIPT
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("DDM MS2 receipt schema drifted")
    authority = payload.get("authority", {})
    if (
        payload.get("score_claim") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("pointer_moved") is not False
        or authority.get("evidence_axis") != EVIDENCE_AXIS
    ):
        raise ValueError("DDM MS2 receipt authority firewall drifted")
    calibration_utc = str(payload["measurement"]["finished_at_utc"])
    provenance = build_provenance_for_research_sidecar(
        source,
        reactivation_criteria=(
            "Land MAIN-reviewed PF2 plus exact measured scorer Gram/composite-R Hessian/"
            "inner-Jacobian/Pose tube, then run n600 batch32 receiver closure."
        ),
        measurement_axis=EVIDENCE_AXIS,
        hardware_substrate="darwin_arm64_cpu_torch",
        captured_at_utc=calibration_utc,
    )
    equations = build_ddm_ms2_typed_quotient_equations(
        provenance=provenance,
        calibration_utc=calibration_utc,
    )
    for equation in equations:
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=(
                "DDM MS2 strict structural law; n600 refused before candidate because "
                "measured scorer geometry/PF2 custody is absent; pointer unmoved; MAIN review"
            ),
        )
    return equations


__all__ = [
    "CODER_RACE_EQUATION_ID",
    "DUAL_EQUATION_ID",
    "EQUATION_IDS",
    "METRIC_EQUATION_ID",
    "PF2R_BLOCKER_ID",
    "PF2R_CONSUMER",
    "QUANTUM_EQUATION_ID",
    "QUOTIENT_EQUATION_ID",
    "build_ddm_ms2_typed_quotient_equations",
    "derivation_edges",
    "effective_quantum",
    "populate_ddm_ms2_typed_quotient_equations",
    "scorer_metric_rate_action",
    "skeleton_fiber_coder_race",
    "typed_block_exchange_rate",
    "visible_quotient_counted_bytes",
]
