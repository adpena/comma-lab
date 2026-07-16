# SPDX-License-Identifier: MIT
"""Canonical law for the hybrid exact-factorized costate adjoint (task #516).

The structural law is DERIVED from three separately measured canonical equations.
Its trajectory admission anchor is advisory and scoped to the #205 telemetry; it is
not a contest score or cross-run promotion claim.
"""
from __future__ import annotations

from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "hybrid_exact_factorized_costate_adjoint_v1"
AXIS = "[macOS advisory] NON-PROMOTABLE"
SOURCE = "src/tac/witness_control/factorized_adjoint.py"
BACKTEST = ".omx/research/costate_organ_elevation_backtest_20260716.json"
SOURCE_SHA256 = "4de3b6cd5579b584c162b741b1bf484402d55e23d432928e90f616f359f0c040"
BACKTEST_SHA256 = "267fd618e2377a7b6c6b32a0be42ca3b28a3072ea972b2fb012fa274e63eda3b"


def build_hybrid_exact_factorized_costate_adjoint_v1() -> CanonicalEquation:
    structure_provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=SOURCE_SHA256,
        source_path=SOURCE,
        captured_at_utc="2026-07-16T20:30:00Z",
    )
    backtest_provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=BACKTEST_SHA256,
        source_path=BACKTEST,
        captured_at_utc="2026-07-16T20:30:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="factorized_structure_from_three_canonical_laws_20260716",
            measurement_utc="2026-07-16T20:20:00Z",
            inputs={
                "head": "segnet_head_rank4_linear_flipdist_v1",
                "resize": "realization_necessity_preimage_per_stratum_v1",
                "gain": "lane_gain_chain_composed_v1",
            },
            predicted_output={
                "law": "K = p_visible B^T diag(||dw||/G) B is rank 4 and gauge-null",
            },
            empirical_output={
                "head_rank": 4,
                "certified_zero_weight_camera_fraction": 0.226969,
                "road_lane_inverse_gain_ratio_vs_other_major_pair_median": 2.0896226415,
                "operator_rank": 4,
                "gauge_null_linf": 2.23e-16,
            },
            residual=2.23e-16,
            source_artifact=SOURCE,
            measurement_method=(
                "compose the registered head/gain/preimage constants; deterministic "
                "float64 incidence-Laplacian rank and nullspace check"),
            provenance=structure_provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="factorized_adjoint_n205_walkforward_20260716",
            measurement_utc="2026-07-16T20:30:00Z",
            inputs={
                "trajectory": "levelset_v752_baseline_20260710T185913Z",
                "intervals": 9,
                "protocol": "LOO plus past-only walk-forward plus binding AUROC",
            },
            predicted_output={"admission": "must beat persistence and AUROC >= 0.8"},
            empirical_output={
                "loo_mae_model": 0.003245520213149131,
                "loo_mae_persistence": 0.003697775590645048,
                "walkforward_mae_model": 0.00185206618604584,
                "walkforward_mae_persistence": 0.002791931483929152,
                "binding_auroc": 0.82,
                "admission": "BACKTESTED-PASS",
                "hyperparameter_selection": "POST_HOC_DEVELOPMENT_ON_205",
                "walkforward_perclass_mae_model": 0.026641247544718305,
                "walkforward_perclass_mae_persistence": 0.010822510285714293,
            },
            residual=0.000939865297883312,
            source_artifact=BACKTEST,
            measurement_method=(
                "deterministic numpy closed-form fits; outer walk-forward is past-only; "
                "per-class walk-forward LOSES and is retained as an explicit caveat"),
            provenance=backtest_provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Hybrid exact-factorized costate adjoint with gated temporal/event residual",
        one_line_summary=(
            "Exact rank-4 head x ker(A)-projected support x inverse-gain pair prior; "
            "only a closed-form temporal posterior and five-scalar amplitude residual learn."),
        latex_form=(
            r"\Lambda_\mathrm{hyb}(x,\phi,t)="
            r"\alpha_\psi(x,\phi,t)\,p_\mathrm{vis}"
            r"B^\top\operatorname{diag}(\|\Delta w_{cc'}\|/G_{cc'})B\phi,\quad "
            r"\alpha_\psi\ge0"
        ),
        python_callable_module_path=(
            "tac.witness_control.factorized_adjoint:exact_response_direction"),
        domain_of_validity={
            "scorer": "frozen contest SegNet constants from the three cited LawRefs",
            "trajectory_admission": "#205 baseline, n=9 intervals only",
            "validation_scope": (
                "development-set pass; residual ridge=10 selected during #205 build; "
                "independent compatible trajectory owed"),
            "mod32cap": "trajectory schema lacks interval-aligned d_seg_by_class; unavailable",
            "c2": "pending sufficient verdict rows; no inference from #205",
            "axis": AXIS,
            "advisory_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": (
                "scalar d_seg forecast and binding tri-gate pass on #205; per-class "
                "walk-forward loses to persistence; all unvaried levers remain duty-to-measure"),
        },
        units_in={"phi": "dimensionless lever-feature vector"},
        units_out={"lambda_response": "campaign-state rate per control-share epoch"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={
            "operator_gauge_null_linf": 2.23e-16,
            "walkforward_mae_improvement": 0.000939865297883312,
            "perclass_walkforward_regression": 0.01581873725838801,
        },
        last_calibration_utc="2026-07-16T20:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_control.shadow_controller._factorized_overlay",
            "tools/costate_digest.py section_shadow",
            "tools/witness_run_introspect.py read_controller",
        ),
        canonical_producers=(
            SOURCE,
            "tools/costate_organ_elevation_backtest.py",
        ),
        provenance=structure_provenance,
    )


def populate_hybrid_exact_factorized_costate_adjoint_equation(
    *, path: str | Path | None = None, lock_path: str | Path | None = None,
    agent: str | None = None, subagent_id: str | None = None,
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_hybrid_exact_factorized_costate_adjoint_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id)
    return eq


__all__ = [
    "AXIS", "BACKTEST", "BACKTEST_SHA256", "EQUATION_ID", "SOURCE", "SOURCE_SHA256",
    "build_hybrid_exact_factorized_costate_adjoint_v1",
    "populate_hybrid_exact_factorized_costate_adjoint_equation",
]
