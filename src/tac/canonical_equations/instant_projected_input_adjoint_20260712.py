# SPDX-License-Identifier: MIT
"""INSTANT adaptive projection law for frozen pointwise-Conv2d input adjoints.

Doan, Tran, Tartaglione, Simidjievski, and Nguyen (2026), *INSTANT:
Compressing Gradients and Activations for Resource-Efficient Training*, ICLR
2026, OpenReview:P2q6Y7UweV, motivates calibrated low-rank gradient spaces.
No arXiv identifier or DOI was found on the verified OpenReview paper page.

This OSS-reconciled adaptation keeps the exact frozen-scorer forward and
approximates only eligible ungrouped 1x1-convolution input adjoints.  It does not import the
paper's empirical speedup claim into Pact; local frozen-SegNet path timing,
including refresh admission and decision guards, is the sole measured economics
authority for this lane.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

INSTANT_PROJECTED_INPUT_ADJOINT_EQUATION_ID = "instant_projected_input_adjoint_v1"
_MEMO = ".omx/research/codex_findings_master_oss_reconciliation_20260713_codex.md"
_RECEIPT = (
    "experiments/results/instant_oss_reconciliation_20260713T033944Z/"
    "measurement_receipt.json"
)
_UTC = "2026-07-13T03:43:06.355354+00:00"


def build_instant_projected_input_adjoint_v1() -> CanonicalEquation:
    """Build the isolated rank law and its measured-admission boundary."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=(
            "reopen only under a separately pre-registered nonlinear forward replacement or a materially "
            "different projection law that clears identical early, boundary, and late descent/economics gates"
        ),
        measurement_axis="[macOS-CPU advisory saved-regime local probe; NON-PROMOTABLE]",
        hardware_substrate="apple_macos_cpu_numpy_torch_with_mlx_skipped_no_metal",
        captured_at_utc=_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="instant_projected_adjoint_charged_cycle_pair0_20260713",
        measurement_utc=_UTC,
        inputs={
            "pair": 0,
            "saved_regimes": ["early_ce_boundary", "boundary_tau_boundary", "late_l7_boundary"],
            "eligible_pointwise_conv2d_layers": 90,
            "other_conv2d_layers_exact": 35,
            "retained_energy_targets": [0.90, 0.95, 0.99],
            "oversampling": 5,
            "timing_samples_per_arm": 3,
            "calibration_states": 3,
        },
        predicted_output={
            "falsifier": (
                "NO-GO unless one retained-energy target has positive renderer-gradient direction and "
                "paired median-minus-MAD speedup greater than one in every saved regime"
            )
        },
        empirical_output={
            "common_admitted_targets": [],
            "admitted_regime_arms": [],
            "late_l7_target_0_95_renderer_gradient_cosine": 0.4499864310744855,
            "late_l7_target_0_95_hot_step_speedup_lower_bound": 0.7506562830313536,
            "maximum_hot_step_speedup_lower_bound": 1.0353914790745997,
            "maximum_optimistic_charged_cycle_ratio": 0.5888733451533681,
            "charged_cycle_cadences": [2, 4, 8],
            "all_nine_arms_decisive_economic_no_go": True,
            "verdict": "NO_GO",
            "verdict_scope": (
                "n=1 pair0; three sealed CE/tau/L7 states; macOS-CPU advisory; exact frozen-SegNet "
                "forward; adaptive projection on eligible ungrouped 1x1 Conv2d input adjoints"
            ),
            "score_claim": False,
            "pointer_moved": False,
            "across_seed_variance": "UNKNOWN",
            "review_status": "fresh-eyes-reviewed(3)-CLEAN",
        },
        residual=1.0,
        source_artifact=_RECEIPT,
        measurement_method=(
            "same-state dense-vs-projected frozen-SegNet forward/backward timing; exact renderer-gradient "
            "direction; exact-teacher CE/d_seg fractional recess; three paired timing samples per arm; "
            "measured projected-candidate validation charged at K={2,4,8} with zero calibration/fallback "
            "explicitly treated as optimistic lower-cost assumptions"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=(
            "within-arm timing uses median-minus-MAD; composed across-seed floor is UNKNOWN on the "
            "single-seed spine"
        ),
    )

    return CanonicalEquation(
        equation_id=INSTANT_PROJECTED_INPUT_ADJOINT_EQUATION_ID,
        name="INSTANT adaptive-axis frozen pointwise-Conv2d input adjoint",
        one_line_summary=(
            "Project each saved pointwise-convolution output cotangent on its smaller calibrated "
            "channel or spatial axis, then apply the frozen weight in that subspace."
        ),
        latex_form=(
            r"G_\ell\in\mathbb{R}^{L_\ell\times C_{out}},\ Q_\ell Q_\ell^\top=I_r,\quad "
            r"\widehat g_{x,\ell}=\begin{cases}(G_\ell Q_\ell^\top)(Q_\ell W_\ell),"
            r"&C_{out}\le L_\ell\\Q_\ell^\top((Q_\ell G_\ell)W_\ell),&L_\ell<C_{out}\end{cases},\quad "
            r"S_K=\frac{K t_{exact}}{t_{cal}+t_{exact}+(K-1)(t_{approx}+t_{validate}+t_{fallback})}"
        ),
        python_callable_module_path=(
            "tac.boundary_math.instant_projected_adjoint:pointwise_input_adjoint_numpy"
        ),
        domain_of_validity={
            "research_only": True,
            "citation": (
                "Tuan-Kiet Doan, Trung-Hieu Tran, Enzo Tartaglione, Nikola Simidjievski, "
                "Van-Tam Nguyen (2026), INSTANT: Compressing Gradients and Activations for "
                "Resource-Efficient Training, ICLR 2026, OpenReview:P2q6Y7UweV; verified "
                "OpenReview paper page; no arXiv identifier or DOI found"
            ),
            "calibration": (
                "Q is the leading singular basis on the smaller of C_out and spatial length; "
                "rank is the smallest value reaching retained energy in {0.90,0.95,0.99}, "
                "plus the official oversampling constant 5 capped by layer geometry"
            ),
            "included": (
                "frozen ungrouped 1x1 Conv2d weights and biases with exact forward operator",
                "finite N=1 output cotangents and orthonormal adaptive-axis bases",
                "ordinary autograd for every non-1x1 convolution and all other scorer adjoints",
                "equal-refresh economics including exact labels and projection calibration",
                "one CE probe relaxation applied to early, tau, and L7 saved states",
                "global, boundary-annulus, renderer-direction, and one-step exact-teacher checks",
            ),
            "admission": (
                "one retained-energy law must show exact-teacher descent in every registered "
                "early, boundary, and late saved-state regime, changed-frame exact-teacher reuse must be "
                "validated, and at least one K in {2,4,8} must exceed one under the typed equal-refresh "
                "economics law after charging measured projected-candidate validation"
            ),
            "excluded": (
                "logit or argmax agreement used as costate evidence",
                "a universal cosine threshold",
                "cross-objective, cross-frame, cross-provider, stale, mutated, or nonfinite evidence",
                "paper-reported training speedups used as Pact timing evidence",
                "score, frontier, trainer-integration, or promotion claims",
                "stage-native tau/L7 objective claims or measured whole-trainer-step timing",
            ),
            "fallback": "any excluded or failed condition selects full_teacher",
            "authority": "derived projection law plus local saved-regime measurement; score_claim=false",
            "review_status": "fresh-eyes-reviewed(3)-CLEAN",
        },
        units_in={
            "G": "teacher_loss_per_convolution_output_unit",
            "W": "convolution_output_unit_per_input_patch_unit",
            "Q": "dimensionless_orthonormal_channel_or_spatial_projection",
            "t_exact": "seconds_per_dense_frozen_segnet_scorer_forward_backward_slice",
            "t_label": (
                "seconds_per_content-bound_exact_refresh_forward_backward_with_output-cotangent_labels; "
                "this is the cycle's exact refresh step and replaces rather than adds to t_exact"
            ),
            "t_cal": (
                "seconds per content-bound adaptive-axis calibration; zero is an explicit optimistic lower-cost assumption"
            ),
            "t_approx": "seconds_per_projected_frozen_segnet_scorer_forward_backward_slice",
            "t_validate": "seconds per exact-teacher validation on a reused projected adjoint",
            "t_fallback": "seconds per full-teacher fallback after a failed validation",
        },
        units_out={
            "g_x_hat": "teacher_loss_per_pointwise_convolution_input_unit",
            "S_K": "dimensionless_equal_refresh_speedup",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "no_common_admitted_rank_law": 1.0,
            "gap_from_best_optimistic_charged_cycle_ratio_to_one": 0.4111266548466319,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=(
            "tac.witness_dsl.scorer_gradient_policy",
            "tools.probe_instant_projected_adjoint",
        ),
        canonical_producers=("tools.probe_instant_projected_adjoint",),
        provenance=provenance,
    )


def populate_instant_projected_input_adjoint_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Idempotently append the equation through the canonical registry writer."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_instant_projected_input_adjoint_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="instant_projected_input_adjoint_probe_20260712",
    )
    return equation


__all__ = [
    "INSTANT_PROJECTED_INPUT_ADJOINT_EQUATION_ID",
    "build_instant_projected_input_adjoint_v1",
    "populate_instant_projected_input_adjoint_v1",
]
