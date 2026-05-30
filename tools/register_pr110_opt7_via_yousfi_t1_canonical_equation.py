#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register canonical equation for PR110-OPT-7 via Yousfi-T1 L1 PROMOTION substrate.

Per Catalog #344 canonical equations registry + the canonical 5-helper
composition contract: register `pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1_savings_v1`
with FORMALIZATION_PENDING first empirical anchor capturing the Phase C
MLX-LOCAL N=100 smoke verification + reactivation criteria for paired-CUDA
RATIFICATION per Catalog #246.
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
    register_canonical_equation,
)
from tac.provenance.builders import build_provenance_for_predicted


def main() -> int:
    utc_now = datetime.now(UTC).isoformat()

    anchor = EmpiricalAnchor(
        anchor_id="pr110_opt7_via_yousfi_t1_l1_promotion_phase_c_smoke_anchor_20260530",
        measurement_utc=utc_now,
        inputs={
            "in_domain_context": "pr110_opt7_5_helper_canonical_composition_l1_promotion_via_yousfi_t1",
            "n_pairs": 100,
            "vulnerable_pair_budget": 16,
            "alaska_color_branch": "Y0_UV",
            "inverse_scorer_basis_strategy": "uniward_inverse_joint_scorer_basis_linear_combination",
            "chroma_perturbation_strategy": "joint_atick_redlich_linear_combination",
            "chroma_perturbation_magnitude": 4.0,
            "use_canonical_pose_vulnerability_anchor": False,
            "rng_seed": 42,
            "smoke_artifacts_path": (
                "experiments/results/pr110_opt7_via_yousfi_t1_l1_promotion_smoke_20260530T205259Z/"
                "training_stats.json"
            ),
        },
        predicted_output={
            "predicted_score_delta_lower_bound": -0.0030,
            "predicted_score_delta_upper_bound": 0.0010,
            "expected_helpers_invoked": 5,
            "expected_substantive_distinctness_verdict": "PASS",
        },
        empirical_output={
            "helpers_invoked_observed": 5,
            "substantive_distinctness_verdict_observed": "PASS",
            "archive_bytes_emitted": 530,
            "inverse_scorer_n_selected_pairs_observed": 16,
            "luma_preservation_yuv6_observed": 0.0,
            "chroma_perturbation_yuv6_observed": 4.0,
            "predicted_score_delta_status": "pending_paired_cuda_ratification",
        },
        residual=0.0,
        source_artifact=(
            "experiments/results/pr110_opt7_via_yousfi_t1_l1_promotion_smoke_20260530T205259Z/"
            "training_stats.json"
        ),
        measurement_method=(
            "macos_mlx_advisory_l1_promotion_smoke_phase_c_5_helper_invocation_verification_"
            "per_slot_eee_no_fake_implementations_gate"
        ),
        provenance=build_provenance_for_predicted(
            model_id=(
                "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1."
                "apply_substrate_to_pr110_canonical"
            ),
            inputs_sha256="0" * 64,  # canonical placeholder; Phase D anchor overrides
        ),
    )

    equation = CanonicalEquation(
        equation_id=(
            "pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1_savings_v1"
        ),
        name=(
            "PR110-OPT-7 Fridrich UNIWARD inverse-scorer basis via Yousfi-Tier-1 "
            "L1 PROMOTION savings"
        ),
        one_line_summary=(
            "5-helper composition (alaska+Yousfi-T1 ABC+PR110-OPT-7) score delta on "
            "PR110 fec6; FORMALIZATION_PENDING paired-CUDA"
        ),
        latex_form=(
            r"\Delta S_{\text{composed}} = \Delta S_{\text{alaska}} \oplus "
            r"\Delta S_{\text{Yousfi-T1-A}} \oplus \Delta S_{\text{Yousfi-T1-B}} \oplus "
            r"\Delta S_{\text{Yousfi-T1-C}} \oplus \Delta S_{\text{PR110-OPT-7}}; "
            r"\Delta S_{\text{composed}} \in [-0.0030, 0.0010] \text{ (predicted)}"
        ),
        python_callable_module_path=(
            "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1."
            "apply_substrate_to_pr110_canonical"
        ),
        domain_of_validity={
            "substrate_id": "pr110_opt7_via_yousfi_t1",
            "canonical_helper_composition_count": 5,
            "n_pairs_range": "[24, 600]",
            "vulnerable_pair_budget_range": "[1, n_pairs]",
            "alaska_color_branches_canonical": [
                "Y0", "Y1", "Y2", "Y3", "U", "V", "Y_only",
                "UV_only", "YUV6_full", "Y0_UV", "Y123_UV",
            ],
            "inverse_scorer_basis_strategies_canonical": [
                "uniward_inverse_local_variance_baseline",
                "uniward_inverse_segnet_gradient_sensitivity",
                "uniward_inverse_posenet_gradient_sensitivity",
                "uniward_inverse_joint_scorer_basis_linear_combination",
            ],
            "chroma_perturbation_strategies_canonical": [
                "local_variance_weighted",
                "segnet_gradient_weighted",
                "posenet_gradient_weighted_via_mae_v",
                "joint_atick_redlich_linear_combination",
            ],
            "chroma_perturbation_magnitude_range": "[0.0, 255.0] exclusive",
            "predicted_band_validation_status": "pending_post_training",
        },
        units_in={
            "n_pairs": "pair_count",
            "vulnerable_pair_budget": "pair_count",
            "chroma_perturbation_magnitude": "per_byte_intensity_in_0_255",
            "alaska_color_branch": "yuv6_channel_branch_enum_value",
            "inverse_scorer_basis_strategy": "uniward_strategy_enum_value",
            "chroma_perturbation_strategy": "chroma_strategy_enum_value",
        },
        units_out={
            "predicted_delta_adjustment": "score_units_canonical_tier_a_observability",
            "delta_vs_fec6_bytes": "archive_bytes_delta_vs_canonical_pr110_fec6_baseline",
            "n_selected_pairs": "pair_count",
            "wire_bytes_estimate": "bytes",
            "luma_preservation_yuv6": "exact_0_per_canonical_contract",
            "chroma_perturbation_yuv6": "perturbation_intensity_in_yuv6_chroma_channels",
            "aggregate_predicted_delta_s": (
                "score_units_canonical_pr110_opt7_inverse_scorer_basis_l0_scaffold"
            ),
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            # All residuals must be >= 0; using absolute residual semantics
            "predicted_band_half_width_abs": 0.0020,
            "paired_cuda_ratification_required": 1.0,
            "phase_c_smoke_helpers_invoked_residual": 0.0,
            "phase_c_smoke_substantive_distinctness_residual": 0.0,
        },
        last_calibration_utc=utc_now,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools/operator_authorize.py (recipe substrate_pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1_modal_t4_dispatch)",
            "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1.verify_canonical_helper_invocation",
        ),
        canonical_producers=(
            "tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1."
            "apply_substrate_to_pr110_canonical",
            "experiments/train_substrate_pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id=(
                "tools/register_pr110_opt7_via_yousfi_t1_canonical_equation.py::"
                "pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1_savings_v1"
            ),
            inputs_sha256="0" * 64,
        ),
    )
    registered = register_canonical_equation(
        equation,
        agent="claude",
        subagent_id="pr110_opt7_l1_promotion_via_yousfi_t1_20260530T210000Z",
        notes=(
            "PR110-OPT-7 L1 PROMOTION via Yousfi-T1 enablement canonical equation "
            "registration. FORMALIZATION_PENDING paired-CUDA RATIFICATION per "
            "Phase D operator-attended dispatch. Anchor captures the Phase C "
            "MLX-LOCAL N=100 smoke GREEN per 7/7 substantive axes."
        ),
    )
    print(f"REGISTERED equation: {registered.equation_id}")
    print(f"  anchors: {len(registered.empirical_anchors)}")
    print(f"  consumers: {len(registered.canonical_consumers)}")
    print(f"  producers: {len(registered.canonical_producers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
