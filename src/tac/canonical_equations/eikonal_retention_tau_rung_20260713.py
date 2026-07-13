# SPDX-License-Identifier: MIT
"""Canonical unified-τ retention coupling for event-native V9 sharpening.

This is a DERIVED control law, not an empirical score claim.  The persisted
``TauAdvanceController`` rung is the sole continuation coordinate; an eikonal
end weight may not be presented as active under unified-τ unless this reachable
controller consumes it.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "eikonal_retention_couples_to_tau_rung_v1"
_UTC = "2026-07-13T02:30:00Z"
_DESIGN = ".omx/research/v9_cgauge_truly_optimal_design_20260712.md"
_AXIS = "[DERIVED control law; source-inspected; no training or score claim]"


def eikonal_retention_for_rung(
    base_weight: float, end_weight: float, rung: int, n_octaves: int
) -> float:
    """Return ``λ_k = λ_0 + (λ_N-λ_0) k/N`` with fail-closed bounds."""
    n = int(n_octaves)
    k = int(rung)
    if n < 1:
        raise ValueError(f"n_octaves must be >=1, got {n_octaves!r}")
    if not (0 <= k <= n):
        raise ValueError(f"rung must be in [0,{n}], got {rung!r}")
    base = float(base_weight)
    end = float(end_weight)
    if base < 0.0 or end < 0.0:
        raise ValueError(f"eikonal weights must be non-negative, got base={base}, end={end}")
    return base + (end - base) * (k / n)


def build_eikonal_retention_couples_to_tau_rung_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_DESIGN,
        reactivation_criteria=(
            "re-derive if the tau continuation coordinate stops being a persisted discrete rung, "
            "or recalibrate the endpoint values after a matched n600 through-R treatment"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="substrate_independent_pure_control",
        captured_at_utc=_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="v9_unified_tau_rung_actuator_source_trace_20260713",
        measurement_utc=_UTC,
        inputs={
            "base_weight": 0.01,
            "end_weight": 0.05,
            "continuation_state": "persisted TauAdvanceController.rung k of N",
            "ordering": "prime lambda_k before assigning tau_k",
        },
        predicted_output={
            "endpoints": [0.01, 0.05],
            "invariant": "every lower-tau assignment observes the already-primed lambda_k",
        },
        empirical_output={
            "source_trace": "unified-tau bypasses discrete event sentinel; event rung path reachable",
            "pure_regression_tests": "endpoint, monotonic progression, invalid-orphan refusal, ordering",
            "score_effect": "UNMEASURED until matched n600 receiver-closed run",
        },
        residual=0.0,
        source_artifact="experiments/test_scheduled_eikonal_weight.py",
        measurement_method="source inspection plus deterministic pure-function regression",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Unified-tau eikonal retention coupled to the persisted tau rung",
        one_line_summary=(
            "Prime lambda_eik linearly on the committed tau rung before assigning the lower tau; "
            "refuse any armed unified-tau ramp without the event rung actuator."
        ),
        latex_form=(
            r"\lambda_{\mathrm{eik},k}=\lambda_0+(\lambda_N-\lambda_0)\frac{k}{N},\quad "
            r"\lambda_{\mathrm{eik},k}\;\prec\;\tau_k"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.eikonal_retention_tau_rung_20260713:"
            "eikonal_retention_for_rung"
        ),
        domain_of_validity={
            "included": (
                "seg-form-unify-tau",
                "tau-advance-mode event",
                "persisted deterministic geometric rung ladder",
            ),
            "excluded": (
                "clock tau under unified-tau",
                "the dissolved CE-to-tau discrete event sentinel",
                "claims about optimal endpoint magnitude or score improvement",
            ),
            "verdict_scope": "DERIVED actuation/ordering law; V9 treatment delta remains UNMEASURED",
            "authority": _AXIS,
        },
        units_in={"lambda": "loss weight", "rung": "integer", "n_octaves": "integer"},
        units_out={"lambda_eik_k": "loss weight"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"pure_function": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments.train_levelset_witness_realized_through_R_mlx",
            "tac.witness_dsl.spec_v9_cgauge",
        ),
        canonical_producers=(_DESIGN,),
        provenance=provenance,
    )


def populate_eikonal_retention_couples_to_tau_rung_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_eikonal_retention_couples_to_tau_rung_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="V9 event-native actuation law; DERIVED; score_claim=false; pointer unmoved",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_eikonal_retention_couples_to_tau_rung_v1",
    "eikonal_retention_for_rung",
    "populate_eikonal_retention_couples_to_tau_rung_v1",
]
