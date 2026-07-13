# SPDX-License-Identifier: MIT
"""Exact fixed-quadratic delay disambiguation anchored to the Round-2 refit."""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "grokking_ridge_undertraining_disambiguation_v1"
MEASUREMENT_UTC = "2026-07-13T18:11:49.811894Z"
DAG_FEED = ".omx/research/grokking_ridge_bounds_DAG_FEED_20260713.md"
AXIS = "[macOS-CPU advisory; numpy-fp32 training-gradient evidence; no score authority]"
MEASUREMENT_RECEIPT = (
    "experiments/results/grokking_ridge_round2_refit_20260713/measurement_receipt.json"
)
MEASUREMENT_RECEIPT_SHA256 = (
    "fc8c79ef82d829f05cee79890c9b5d237e12d84e92ec83982f182de15ecb6b4d"
)


def derive_fixed_quadratic_delay_certificate(
    *,
    learning_rate: float,
    ridge_lambda: float,
    contraction_gamma: float,
    steps: int,
    initial_null_norm: float,
    terminal_gradient_norm: float,
    strong_curvature_mu: float,
) -> dict[str, float | bool | int]:
    """Derive exact delay quantities without importing the paper's assumptions.

    For ``F(W)=||XW-Y||^2/(2n)+lambda||W||^2/2``, the component in
    ``null(X)`` evolves exactly as ``(1-eta*lambda)^t``.  The global
    parameter error is bounded by strong convexity using the actually measured
    terminal gradient.  This helper does not claim population generalization.
    """

    numeric = (
        learning_rate,
        ridge_lambda,
        contraction_gamma,
        initial_null_norm,
        terminal_gradient_norm,
        strong_curvature_mu,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in numeric
    ):
        raise ValueError("delay certificate inputs must be finite numbers")
    if learning_rate <= 0.0 or ridge_lambda < 0.0:
        raise ValueError("learning rate must be positive and ridge lambda nonnegative")
    if not 0.0 <= contraction_gamma < 1.0:
        raise ValueError("contraction gamma must lie in [0,1)")
    if initial_null_norm < 0.0 or terminal_gradient_norm < 0.0:
        raise ValueError("norms must be nonnegative")
    if strong_curvature_mu <= 0.0:
        raise ValueError("strong curvature must be positive")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    null_factor = abs(1.0 - learning_rate * ridge_lambda)
    if null_factor >= 1.0:
        raise ValueError("null-space update is not contractive")
    return {
        "steps": steps,
        "null_retention_factor": null_factor**steps,
        "terminal_null_norm": (null_factor**steps) * initial_null_norm,
        "paper_slow_component_present": initial_null_norm > 0.0,
        "global_contraction_factor": contraction_gamma**steps,
        "terminal_parameter_error_bound": terminal_gradient_norm / strong_curvature_mu,
    }


def build_grokking_ridge_undertraining_disambiguation_v1() -> CanonicalEquation:
    """Build the fixed-head terminality law with the real-n600 refit anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DAG_FEED,
        reactivation_criteria=(
            "recompute for any feature chart, replay distribution, ridge ladder, initialization, "
            "dtype, or heldout target change; witness-stage transfer requires a separately measured "
            "stable Jacobian-null projector and heldout/evaluator calibration"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_macos_arm64_cpu_numpy_fp32_and_cpu_torch_teacher",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="grokking_ridge_round2_real_n600_refit_20260713",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "feature_dimension_m": 31,
            "scalar_training_rows_n": 1_474_560,
            "train_states": 480,
            "heldout_states": 120,
            "ridge_ratios_to_data_lmax": [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0],
            "initialization_variance_nu2": 0.0,
            "fit_steps": [15, 150],
            "receipt_sha256": MEASUREMENT_RECEIPT_SHA256,
        },
        predicted_output={
            "gd15_reproduces_committed_weights": True,
            "gd150_materially_changes_heldout_fidelity": False,
            "paper_eq7_directly_applicable": False,
            "fixed_objective_undertrained": False,
        },
        empirical_output={
            "gd15_reproduced_committed_weights_bitwise": True,
            "gd15_objective_gap": 0.0,
            "gd15_terminal_gradient_norm": 8.38783763098216e-15,
            "gd15_parameter_residual": 2.2703186949787158e-15,
            "gd15_parameter_residual_bound": 2.601118716541341e-15,
            "gd15_heldout_cosine": 0.001415793417951615,
            "gd150_heldout_cosine": 0.0014157934642280926,
            "gd150_minus_gd15_heldout_cosine": 4.627647760226061e-11,
            "gd150_max_abs_weight_delta": 8.881784197001252e-16,
            "best_exact_ridge_ladder_cosine": 0.007690592649965529,
            "best_exact_ridge_ladder_relative_l2": 1.0007586082750441,
            "best_exact_ridge_ratio": 1e-6,
            "eta_times_spectral_lambda": 0.6666666831905239,
            "one_minus_eta_times_spectral_lambda": 0.3333333168094761,
            "null_mode_half_life_steps": 0.6309297251026525,
            "paper_overparameterization_m_minus_n": -1_474_529,
            "synthetic_data_used": False,
            "new_heldout_teacher_calls": 120,
            "verdict": "FEATURE_POVERTY_FORMULATION_NOT_UNDERTRAINED",
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=4.627647760226061e-11,
        source_artifact=MEASUREMENT_RECEIPT,
        measurement_method=(
            "reuse 480 content-addressed exact-label training sufficient statistics; reproduce the "
            "committed 15-step NumPy-fp32 fit; extend to 150 steps; recompute 120 real heldout CPU "
            "SegNet costates; evaluate exact optima over seven ridge scales with fp64 reductions"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Fixed-quadratic undertraining versus feature-ceiling disambiguation",
        one_line_summary=(
            "Zero null initialization removes the ridge-grokking slow mode; terminal gradient and "
            "exact ridge-ladder optima separate optimizer delay from fixed linear feature poverty."
        ),
        latex_form=(
            r"W_t-W^*=(I-\eta H)^t(W_0-W^*),\quad H=X^\top X/n+\lambda I;\quad "
            r"P_{\ker X}W_t=(1-\eta\lambda)^tP_{\ker X}W_0;\quad "
            r"P_{\ker X}W_0=0\Rightarrow P_{\ker X}W_t=0;\quad "
            r"\|W_t-W^*\|_F\le\|\nabla F(W_t)\|_F/\mu"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.grokking_ridge_undertraining_disambiguation_20260713:"
            "derive_fixed_quadratic_delay_certificate"
        ),
        domain_of_validity={
            "scope_level": "formulation x instance",
            "included": [
                "fixed feature matrix and fixed replay distribution",
                "quadratic ridge objective with constant learning rate",
                "full-batch deterministic gradient descent",
                "terminal-gradient strong-convexity certificate",
                "heldout exact-optimum comparison across a declared ridge ladder",
            ],
            "excluded": [
                "population-generalization guarantees when realizability is unproved",
                "nonconvex witness training, changing curriculum objectives, AdamW, Muon, or EMA",
                "stage-advance authority for event #315, hit detector #344, or intrinsic-time clocks",
                "nonlinear heads, frozen-stem features, RFF lifts, margin-field targets, and on-policy replay",
                "evaluator score, contest-CPU/CUDA, promotion, or live-training authority",
            ],
            "paper_equation_7_applicability_to_anchor": "REFUSED: m<n and nu^2=0",
            "authority": AXIS,
            "research_only": True,
            "review_status": "self-audited-UNREVIEWED_BY_MAIN",
        },
        units_in={
            "eta": "inverse Hessian-curvature units",
            "lambda_mu": "Hessian-curvature units",
            "gradient": "objective per parameter unit",
            "parameter_norm": "Frobenius head-parameter norm",
        },
        units_out={
            "retention_and_contraction": "dimensionless",
            "terminal_parameter_error_bound": "Frobenius head-parameter norm",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "gd150_minus_gd15_heldout_cosine": 4.627647760226061e-11,
            "fixed_objective_terminality_bound_violation": 0.0,
            "source_run_mutation": 0.0,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_grokking_ridge_round2",
            "round3_surrogate_feature_admission_guard",
        ),
        canonical_producers=("tools.probe_grokking_ridge_round2",),
        provenance=provenance,
    )


def populate_grokking_ridge_undertraining_disambiguation_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append through the locked registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_grokking_ridge_undertraining_disambiguation_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="round2-grokking-autopsy; fixed-head; real-n600; research-only",
    )
    return equation


__all__ = [
    "AXIS",
    "DAG_FEED",
    "EQUATION_ID",
    "MEASUREMENT_RECEIPT",
    "MEASUREMENT_RECEIPT_SHA256",
    "MEASUREMENT_UTC",
    "build_grokking_ridge_undertraining_disambiguation_v1",
    "derive_fixed_quadratic_delay_certificate",
    "populate_grokking_ridge_undertraining_disambiguation_v1",
]

