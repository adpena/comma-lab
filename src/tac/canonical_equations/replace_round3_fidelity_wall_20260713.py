# SPDX-License-Identifier: MIT
"""Canonical laws and empirical anchor for REPLACE round-3 fidelity wall."""

from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "replace_round3_fidelity_wall_v1"
MEASUREMENT_UTC = "2026-07-13T18:27:28.528897Z"
DAG_FEED = ".omx/research/replace_round3_fidelity_wall_DAG_FEED_20260713.md"
AXIS = "[macOS-CPU advisory; fp32 training-gradient evidence; no score authority]"
MEASUREMENT_RECEIPT = (
    "experiments/results/replace_round3_fidelity_wall_20260713/measurement_receipt.json"
)
MEASUREMENT_RECEIPT_SHA256 = (
    "83704e64d1e5a70c00cf96c19330ff8453459e1024f957bceb48f99972157d75"
)


def conditional_masked_costate_cosine(*, retained_l2_square_fraction: float) -> float:
    """Derive ``cos(lambda, M lambda)=sqrt(rho)`` for an orthogonal mask ``M``."""

    value = retained_l2_square_fraction
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError("retained L2-square fraction must be finite and in [0,1]")
    return math.sqrt(float(value))


def exact_teacher_call_economics(
    *, training_label_calls: int, validation_calls: int, effective_cached_label_uses: int
) -> dict[str, float | int]:
    """Separate label acquisition from validation in cached-teacher economics."""

    values = (training_label_calls, validation_calls, effective_cached_label_uses)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("teacher-call economics accepts nonnegative integers only")
    if training_label_calls == 0 or training_label_calls + validation_calls == 0:
        raise ValueError("amortization requires at least one training label call")
    total = training_label_calls + validation_calls
    return {
        "training_label_calls": training_label_calls,
        "validation_calls": validation_calls,
        "all_exact_labeled_calls": total,
        "effective_cached_label_uses": effective_cached_label_uses,
        "label_only_amortization_x": effective_cached_label_uses / training_label_calls,
        "inclusive_amortization_x": effective_cached_label_uses / total,
    }


def prefix_conv_compute_fraction(
    *, prefix_forward_macs: int, full_forward_macs: int
) -> dict[str, float | int]:
    """Derive prefix/full forward-plus-input-VJP convolution compute.

    Frozen convolution weights charge one forward-equivalent pass for the input
    VJP on both numerator and denominator, so the common factor of two cancels.
    """

    values = (prefix_forward_macs, full_forward_macs)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("convolution MAC custody accepts positive integers only")
    if prefix_forward_macs > full_forward_macs:
        raise ValueError("prefix MACs cannot exceed full-forward MACs")
    fraction = prefix_forward_macs / full_forward_macs
    return {
        "prefix_forward_macs": prefix_forward_macs,
        "full_forward_macs": full_forward_macs,
        "prefix_fraction_of_full_teacher_conv_flops": fraction,
        "conv_only_ideal_speedup_x": 1.0 / fraction,
    }


def build_replace_round3_fidelity_wall_v1() -> CanonicalEquation:
    """Build the scoped negative anchor and the formulation-independent laws."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=DAG_FEED,
        reactivation_criteria=(
            "recompute on any prefix depth, feature lift, target, replay distribution, seed, "
            "FORE support, dtype, or exact-label custody change; the negative is limited to "
            "the registered shallow pre-SE n600 instance"
        ),
        measurement_axis=AXIS,
        hardware_substrate="apple_macos_arm64_cpu_numpy_fp32_and_cpu_torch_teacher",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="replace_round3_fidelity_wall_v9_n600_seed455_20260713",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "n_pairs": 600,
            "train_states": 480,
            "heldout_states": 120,
            "checkpoint_epochs": [150, 251, 275],
            "base_feature_count": 42,
            "rff_feature_count": 74,
            "rff_frequency_count": 16,
            "sampled_train_rows": 1_474_560,
            "fit_epochs": 15,
            "effective_cached_label_uses": 7_200,
            "seed": 455,
            "receipt_sha256": MEASUREMENT_RECEIPT_SHA256,
        },
        predicted_output={
            "heldout_input_costate_cosine_minimum": 0.07078966932743762,
            "positive_dot_state_fraction_minimum": 0.60,
            "localizer_l2_square_mass_fraction_minimum": 0.47,
            "label_only_clean_run_amortization_x": 15.0,
            "inclusive_clean_run_amortization_x": 12.0,
        },
        empirical_output={
            "verdict": "NO_GO_REGISTERED_ROUND3_RUNGS",
            "winning_rung": "pre_se_prefix_rff",
            "linear_heldout_costate_cosine": 0.0016650255538056325,
            "rff_heldout_costate_cosine": 0.0016791964165317613,
            "rff_heldout_costate_relative_l2": 1.0000003871015077,
            "rff_positive_dot_state_fraction": 0.9166666666666666,
            "rff_renderer_gradient_cosine": 0.0857091119977912,
            "source_margin_retained_l2_square_fraction": 0.1634677541848741,
            "source_margin_conditional_exact_cosine": 0.40431145690528497,
            "rff_mass_retained_l2_square_fraction": 0.024426459564827255,
            "rff_mass_conditional_exact_cosine": 0.1562896655727027,
            "oracle_retained_l2_square_fraction": 0.5278150212253758,
            "oracle_conditional_exact_cosine": 0.72650878950318,
            "prefix_fraction_of_full_teacher_conv_flops": 0.005714118050141177,
            "clean_training_label_calls": 480,
            "clean_validation_calls": 120,
            "clean_label_only_amortization_x": 15.0,
            "clean_inclusive_amortization_x": 12.0,
            "campaign_conservative_training_starts": 626,
            "campaign_conservative_all_starts": 746,
            "campaign_conservative_label_only_amortization_x": 11.501597444089457,
            "campaign_conservative_inclusive_amortization_x": 9.651474530831099,
            "linear_executed_spectral_gamma": 0.3333333514855682,
            "linear_max_observed_parameter_ratio": 0.33369136360723056,
            "rff_executed_spectral_gamma": 0.3333333857516738,
            "rff_max_observed_parameter_ratio": 0.3358835234087279,
            "fit_residual_bounds_all_validated": True,
            "FORE_weights_applied": False,
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=0.06911047291090586,
        source_artifact=MEASUREMENT_RECEIPT,
        measurement_method=(
            "read-only deterministic replay of three real V9 n600 checkpoints; batch-size-1 "
            "CPU SegNet label calls; frozen local pre-SE prefix-adjoint ridge; one fixed RFF "
            "lift; exact local-prefix VJP; 120-state fp64 direction reductions; two 4.7-percent "
            "support localizers; campaign-wide retry accounting; certified exact-scratch cleanup"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Frozen-prefix costate chain, support identity, and teacher-call economics",
        one_line_summary=(
            "A fixed chart keeps the head convex; masked exact-costate cosine is sqrt(rho); "
            "the registered shallow linear/RFF instance does not predict costate direction."
        ),
        latex_form=(
            r"\widehat{\lambda}_x=J_\phi(x)^\top\widehat{\lambda}_\phi,\quad "
            r"\widehat{\lambda}_\phi=X_\phi W;\qquad "
            r"\rho=\|M\lambda\|_2^2/\|\lambda\|_2^2,\quad "
            r"\cos(\lambda,M\lambda)=\sqrt{\rho};\qquad "
            r"C_{teacher}=A_{label}+V_{validation}+c_{label}D,\quad c_{label}=0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.replace_round3_fidelity_wall_20260713:"
            "conditional_masked_costate_cosine"
        ),
        domain_of_validity={
            "scope_level": "formulation x instance",
            "included": [
                "fixed frozen local pre-SE feature map and exact local-prefix VJP",
                "fixed feature or fixed RFF lift with a convex ridge output head",
                "orthogonal binary support projectors on exact costates",
                "explicit separation of training-label and validation calls",
                "fixed V9 n600 replay, seed455, and local macOS CPU evidence",
            ],
            "excluded": [
                "deeper frozen prefixes or trainable nonlinear learners",
                "class-pair or multiscale heads and top-k classification targets",
                "transition-complete or on-policy FORE-weighted replay",
                "wall-speed claims under the observed host contention",
                "evaluator score, archive bytes, contest-CPU, CUDA, MPS, or promotion authority",
            ],
            "fp32_contraction_caveat": (
                "executed spectral operators are contractive and objective/residual certificates pass, "
                "but observed parameter ratios exceed gamma by 0.000358 linear and 0.002550 RFF"
            ),
            "research_only": True,
            "review_status": "self-audited-UNREVIEWED_BY_MAIN",
            "authority": AXIS,
        },
        units_in={
            "lambda": "SegNet loss per scorer-input RGB unit",
            "rho": "retained squared-L2 fraction",
            "A_label_V_validation_D": "exact labeled state calls or cached label uses",
            "prefix_full_macs": "multiply-accumulates",
        },
        units_out={
            "costate_cosine": "dimensionless",
            "teacher_amortization": "cached uses per exact labeled call",
            "conv_compute_fraction": "dimensionless",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "winning_direction_cosine_shortfall": 0.06911047291090586,
            "source_margin_mass_fraction_shortfall": 0.3065322458151259,
            "rff_mass_fraction_shortfall": 0.4455735404351727,
            "linear_observed_parameter_ratio_above_gamma": 0.0003580121216623655,
            "rff_observed_parameter_ratio_above_gamma": 0.0025501376570540857,
            "clean_label_only_amortization_shortfall": 0.0,
            "clean_inclusive_amortization_shortfall": 0.0,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.scorer_surrogate.replace_round3_fidelity_wall",
            "tools.probe_replace_round3_fidelity_wall",
            "tac.witness_dsl.replace_round3_fidelity_wall_policy",
        ),
        canonical_producers=("tools.probe_replace_round3_fidelity_wall",),
        provenance=provenance,
    )


def populate_replace_round3_fidelity_wall_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append only through the locked registry helper; main review owns shared registration."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_replace_round3_fidelity_wall_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="replace-round3; scoped-no-go; frozen-prefix-rff; research-only",
    )
    return equation


__all__ = [
    "AXIS",
    "DAG_FEED",
    "EQUATION_ID",
    "MEASUREMENT_RECEIPT",
    "MEASUREMENT_RECEIPT_SHA256",
    "MEASUREMENT_UTC",
    "build_replace_round3_fidelity_wall_v1",
    "conditional_masked_costate_cosine",
    "exact_teacher_call_economics",
    "populate_replace_round3_fidelity_wall_v1",
    "prefix_conv_compute_fraction",
]
