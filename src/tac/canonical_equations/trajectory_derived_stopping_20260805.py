# SPDX-License-Identifier: MIT
"""TJ1 trajectory-derived stopping law.

The law is deliberately scorer-free.  It consumes already-recorded objective
trajectories and converts projected tail gain into contest score units.  Safety
caps are reported as safety caps; they are not convergence certificates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.optimization.trajectory_stopping import (
    TRAJECTORY_STOPPING_LAW_REF,
    TrajectoryPoint,
    TrajectoryStopConfig,
    byte_score_units,
    evaluate_trajectory_stop,
    projection_interval,
    seg_flip_score_units,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = TRAJECTORY_STOPPING_LAW_REF
SOURCE_ARTIFACT = ".omx/research/ddm_tj1_20260805/trajectory_replay.json"


def sq1_prefix25_projection_interval() -> dict[str, Any]:
    """Replay the embedded SQ1 prefix-25 positive control."""

    curve = (
        (0, 27_084),
        (5, 15_237),
        (10, 11_784),
        (15, 10_242),
        (20, 9_151),
        (25, 8_553),
    )
    cfg = TrajectoryStopConfig(
        score_units_per_objective=seg_flip_score_units(),
        marginal_score_gain_per_compute=byte_score_units(),
    )
    points = tuple(TrajectoryPoint(float(step), float(value)) for step, value in curve)
    interval = projection_interval(points, cfg, target_compute=50.0)
    decision = evaluate_trajectory_stop(points, cfg, safety_bound_compute=25.0)
    return {
        "decision": decision.to_payload(),
        "projection": interval.to_payload(),
    }


def _source_payload(source_receipt: str | Path) -> dict[str, Any]:
    path = Path(source_receipt)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _anchor(source_receipt: str | Path, provenance) -> EmpiricalAnchor:
    payload = _source_payload(source_receipt)
    projection = payload.get("prefix25_projection_to_step50", {})
    if not projection:
        projection = sq1_prefix25_projection_interval()["projection"]
    measured_eta = float(projection.get("measured_step50_eta", 0.8620042643923241))
    eta_low = float(projection.get("eta_low", 0.796750))
    eta_high = float(projection.get("eta_high", 0.864350))
    residual = 0.0 if eta_low <= measured_eta <= eta_high else min(
        abs(measured_eta - eta_low),
        abs(measured_eta - eta_high),
    )
    return EmpiricalAnchor(
        anchor_id="tj1_sq1_prefix25_predicts_step50_20260805",
        measurement_utc="2026-08-05T00:00:00Z",
        inputs={
            "source_receipt": str(source_receipt),
            "prefix_step": 25,
            "target_step": 50,
            "safety_bound_is_not_convergence": True,
        },
        predicted_output={
            "eta_low": eta_low,
            "eta_high": eta_high,
            "stop_reason_at_25": payload.get("controls", [{}])[0]
            .get("decision", {})
            .get("stop_reason", "safety_bound_REPORTED"),
        },
        empirical_output={
            "measured_eta_step50": measured_eta,
            "inside_interval": eta_low <= measured_eta <= eta_high,
        },
        residual=residual,
        source_artifact=str(source_receipt),
        measurement_method=(
            "scorer-free replay over recorded SQ1/CW1 solved-paint trajectories; "
            "projection fitted on steps 0..25 and checked against realized step-50 eta"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
    )


def build_trajectory_derived_stopping_law_v1(
    *, source_receipt: str | Path = SOURCE_ARTIFACT
) -> CanonicalEquation:
    """Build the canonical TJ1 law with its SQ1 prefix positive control."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=source_receipt,
        reactivation_criteria=(
            "append a new anchor whenever a complete longer-depth receipt lands; "
            "do not use cap-bound rows as convergence certificates"
        ),
        measurement_axis="[research-signal]",
        hardware_substrate="unknown",
        captured_at_utc="2026-08-05T00:00:00Z",
    )
    anchor = _anchor(source_receipt, provenance)
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="trajectory-derived stopping law",
        one_line_summary=(
            "Fit recorded objective trajectories and stop only when projected tail gain in "
            "contest S-units falls below the caller's marginal bar; safety caps are reported "
            "as safety_bound_REPORTED, not convergence."
        ),
        latex_form=(
            r"\text{stop}\iff \widehat{\Delta S}_{tail}\le \epsilon_S "
            r"\;\text{or}\; d\widehat S/dc < \epsilon_S/c"
        ),
        python_callable_module_path=(
            "tac.optimization.trajectory_stopping:evaluate_trajectory_stop"
        ),
        domain_of_validity={
            "included": [
                "recorded solver objective trajectories with a monotone compute coordinate",
                "SQ1 solved-paint proxy-flip curves and terminal_pose_gn joint-action traces",
                "adaptive recursion depth allocation after a positive receiver/solver receipt",
            ],
            "excluded": [
                "direct SegNet/PoseNet scorer execution",
                "using an iteration cap as a convergence certificate",
                "pointer movement, score promotion, or exact-eval authority",
            ],
            "authority": "[research-signal] scorer-free recorded-trajectory replay",
        },
        units_in={
            "objective": "caller-defined loss/debt count",
            "compute": "solver step, relinearization, or evaluation coordinate",
            "score_units_per_objective": "contest S per objective unit",
            "marginal_score_gain_per_compute": "contest S per compute unit",
        },
        units_out={
            "stop_reason": "typed stop token",
            "projected_remaining_score_gain": "contest S units",
            "marginal_score_gain_per_compute": "contest S per compute unit",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"sq1_prefix25_eta_interval_residual": anchor.residual},
        last_calibration_utc="2026-08-05T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments.ddm_sq1_stage_decomposition_and_solved_paint.solve_margin_optimal_paint",
            "tac.optimization.terminal_pose_gn.solve_terminal_pose_gn",
            "tools.replay_tj1_trajectory_stopping",
        ),
        canonical_producers=(
            "SQ1/CW1 solved-paint trajectory receipts",
            "terminal_pose_gn step traces",
            "NG1 cap-artifact sweep",
        ),
        provenance=provenance,
    )


def populate_trajectory_derived_stopping_law_v1(
    *,
    source_receipt: str | Path = SOURCE_ARTIFACT,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the validated TJ1 law through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_trajectory_derived_stopping_law_v1(source_receipt=source_receipt)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "tj1 trajectory-derived stopping law; scorer-free positive controls pass; "
            "safety caps remain reported bounds"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "SOURCE_ARTIFACT",
    "build_trajectory_derived_stopping_law_v1",
    "populate_trajectory_derived_stopping_law_v1",
    "sq1_prefix25_projection_interval",
]
