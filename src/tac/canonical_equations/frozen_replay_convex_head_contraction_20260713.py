# SPDX-License-Identifier: MIT
"""Canonical contraction and exact-label amortization law for round 2."""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "frozen_replay_convex_head_contraction_v1"
MEASUREMENT_UTC = "2026-07-13T17:05:25.719725Z"
DAG_FEED = ".omx/research/frozen_replay_convex_head_contraction_DAG_FEED_20260713.md"
AXIS = "[macOS-CPU advisory; numpy-fp32 training-gradient evidence; no score authority]"
MEASUREMENT_RECEIPT = (
    "experiments/results/frozen_replay_convex_head_95kill_n600_20260713/"
    "measurement_receipt.json"
)
MEASUREMENT_RECEIPT_SHA256 = (
    "067ce197d30fa9e2c7c4bda48ac671af550e0a00f126289ba5b30946d44fc4b1"
)


def derive_spectral_scale_contraction(
    *, data_eigenvalue_min: float, data_eigenvalue_max: float
) -> dict[str, float]:
    """Derive ridge curvature and optimal-step contraction in Frobenius norm.

    The registered ridge is not caller tuned: ``lambda=data_eigenvalue_max``.
    For a positive-semidefinite feature covariance this makes ``kappa(H)<=2``
    and therefore ``gamma<=1/3`` even when the design matrix is rank deficient.
    """

    values = (data_eigenvalue_min, data_eigenvalue_max)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("data eigenvalues must be finite numbers")
    if data_eigenvalue_min < 0.0 or data_eigenvalue_max <= 0.0:
        raise ValueError("feature covariance must be positive semidefinite with positive scale")
    if data_eigenvalue_min > data_eigenvalue_max:
        raise ValueError("minimum eigenvalue cannot exceed maximum eigenvalue")
    ridge = float(data_eigenvalue_max)
    mu = float(data_eigenvalue_min + ridge)
    smoothness = float(data_eigenvalue_max + ridge)
    eta = 2.0 / (mu + smoothness)
    gamma = (smoothness - mu) / (smoothness + mu)
    return {
        "ridge_lambda": ridge,
        "mu": mu,
        "smoothness_L": smoothness,
        "step_size_eta": eta,
        "contraction_gamma": gamma,
        "derived_gamma_upper_bound": 1.0 / 3.0,
    }


def cached_exact_label_teacher_calls(
    *, fresh_anchor_samples: int, paired_difference_samples: int, labels_per_difference: int
) -> int:
    """Return ``C_teacher=A+c_label*D`` with fail-closed integer custody."""

    values = (fresh_anchor_samples, paired_difference_samples, labels_per_difference)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("teacher-call law accepts nonnegative integers only")
    return fresh_anchor_samples + labels_per_difference * paired_difference_samples


def build_frozen_replay_convex_head_contraction_v1() -> CanonicalEquation:
    """Build the theorem with its fixed-replay real-n600 empirical anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DAG_FEED,
        reactivation_criteria=(
            "recompute on any replay-distribution, feature-chart, ridge-policy, dtype, or exact-label "
            "custody change; a negative is scoped to the registered fixed V9 n600 formulation"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_macos_arm64_cpu_numpy_fp32_and_cpu_torch_teacher",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="frozen_replay_convex_head_v9_n600_seed455_20260713",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "n_pairs": 600,
            "train_states": 480,
            "heldout_states": 120,
            "sampled_train_rows": 1_474_560,
            "feature_count": 31,
            "fit_epochs": 15,
            "effective_cached_state_steps": 7_200,
            "seed": 455,
            "teacher_batch_size": 1,
            "gt_cache_sha256": (
                "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
            ),
            "receipt_sha256": MEASUREMENT_RECEIPT_SHA256,
        },
        predicted_output={
            "ideal_contraction_gamma_upper_bound": 1.0 / 3.0,
            "executed_parameter_ratios_must_not_exceed_derived_gamma": True,
            "teacher_call_amortization_minimum_x": 5.0,
            "heldout_cosine_minimum": -0.16153190769629602,
        },
        empirical_output={
            "verdict": "GO",
            "executed_contraction_gamma": 0.3333333461703458,
            "max_parameter_ratio_above_scale_floor": 0.32923753849768017,
            "max_objective_ratio_above_scale_floor": 0.10413857661064749,
            "mu": 3.2247038851557344,
            "smoothness_L": 6.449407796557402,
            "ridge_lambda": 3.2247040271759033,
            "executed_step_size_eta": 0.20673732459545135,
            "terminal_gradient_norm": 8.38783763098216e-15,
            "parameter_residual": 2.2703186949787158e-15,
            "parameter_residual_bound": 2.601118716541341e-15,
            "fit_prediction_rmse_residual": 1.3920579326014151e-15,
            "fit_prediction_rmse_bound": 4.670948667757252e-15,
            "per_state_gradient_variance": 7.21498595203425e-14,
            "heldout_costate_cosine": 0.0014157933865487525,
            "heldout_costate_relative_l2": 1.0000018705777456,
            "heldout_renderer_gradient_dot": 0.1096160079189985,
            "heldout_renderer_gradient_cosine": 0.017697414591996724,
            "fresh_teacher_calls": 600,
            "effective_cached_state_steps": 7_200,
            "teacher_calls_per_effective_step": 1.0 / 12.0,
            "teacher_call_amortization_x": 12.0,
            "label_difference_teacher_coefficient": 0,
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=MEASUREMENT_RECEIPT,
        measurement_method=(
            "read-only deterministic replay of three real V9 n600 checkpoints; one batch-size-1 "
            "CPU SegNet input-costate call per unique state; objective-exact fp32 X'X/X'Y/Y'Y "
            "cache; full-batch 15-step convex solve; full-grid 120-state held-out costate and "
            "matched renderer-gradient fp64 reductions; append-only verifier source amendment "
            "after a scale-floor bug, with zero repeated training teacher calls"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Frozen-replay convex-head contraction and cached-label call law",
        one_line_summary=(
            "Spectral ridge gives ideal full-batch gamma<=1/3; executed fp32 gamma is re-derived "
            "after eta rounding, and cached same-state labels make c_label=0."
        ),
        latex_form=(
            r"F(W)=\frac{\|XW-Y\|_F^2}{2n}+\frac{\lambda\|W\|_F^2}{2},\quad "
            r"H=X^\top X/n+\lambda I,\quad \lambda=\lambda_{\max}(X^\top X/n),\quad "
            r"\eta^*=\frac{2}{\mu+L},\quad \|W_{t+1}-W^*\|_F\le "
            r"\gamma_{ideal}\|W_t-W^*\|_F,\quad "
            r"\gamma_{ideal}=\frac{L-\mu}{L+\mu}\le\frac13;\quad "
            r"C_{teacher}=A+c_{label}D,\quad c_{label}=0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.frozen_replay_convex_head_contraction_20260713:"
            "derive_spectral_scale_contraction"
        ),
        domain_of_validity={
            "scope_level": "formulation x instance",
            "included": [
                "fixed replay distribution and fixed feature matrix X",
                "cached exact labels Y with content-addressed state identity",
                "full-batch deterministic NumPy-fp32 gradient descent",
                "Euclidean parameter and Frobenius head norm",
                "ridge on every trainable head coordinate",
                "separate executed-fp32 eta rounding and realized operator-norm certificate",
            ],
            "excluded": [
                "on-policy or drifting replay distributions",
                "nonlinear trainable feature extractors",
                "minibatch contraction without a separate stochastic noise-floor theorem",
                "MPS, evaluator score, contest-CPU, CUDA, promotion, or live-training authority",
            ],
            "residual_to_fidelity": (
                "||W-W*||_F<=||grad F||_F/mu and "
                "||X(W-W*)||_F/sqrt(n)<=sqrt(L-lambda)||grad F||_F/mu"
            ),
            "label_cancellation": "g_s(W)-g_s(V)=X_s'X_s(W-V)",
            "authority": AXIS,
            "research_only": True,
            "review_status": "self-audited-UNREVIEWED_BY_MAIN",
        },
        units_in={
            "X": "fixed feature units",
            "Y": "exact SegNet input-costate units per scorer-input RGB unit",
            "W": "costate units per feature unit",
            "A_D_C_teacher": "exact labeled state evaluations",
        },
        units_out={
            "gamma": "dimensionless Frobenius-norm contraction",
            "eta": "inverse Hessian-curvature units",
            "C_teacher": "exact labeled state evaluations",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "ideal_theorem_gamma_upper_bound": 1.0 / 3.0,
            "executed_gamma_rounding_above_ideal_upper_bound": (
                0.3333333461703458 - 1.0 / 3.0
            ),
            "parameter_contraction_bound_violation": 0.0,
            "objective_contraction_bound_violation": 0.0,
            "teacher_amortization_5x_shortfall": 0.0,
            "round1_early_cosine_bar_shortfall": 0.0,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.scorer_surrogate.frozen_replay_convex_head",
            "tools.probe_frozen_replay_convex_head",
            "tac.witness_dsl.frozen_replay_convex_head_policy",
        ),
        canonical_producers=("tools.probe_frozen_replay_convex_head",),
        provenance=provenance,
    )


def populate_frozen_replay_convex_head_contraction_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append only through the locked registry helper; never edit the hot registry."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_frozen_replay_convex_head_contraction_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="round2-95kill; fixed-replay; cached-exact-labels; research-only",
    )
    return equation


__all__ = [
    "AXIS",
    "DAG_FEED",
    "EQUATION_ID",
    "MEASUREMENT_RECEIPT",
    "MEASUREMENT_RECEIPT_SHA256",
    "MEASUREMENT_UTC",
    "build_frozen_replay_convex_head_contraction_v1",
    "cached_exact_label_teacher_calls",
    "derive_spectral_scale_contraction",
    "populate_frozen_replay_convex_head_contraction_v1",
]
