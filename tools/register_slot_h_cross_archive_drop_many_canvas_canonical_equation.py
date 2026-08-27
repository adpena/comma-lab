# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""SLOT H cascade item 3 — register `cross_archive_drop_many_canvas_composition_alpha_v1`.

Per Catalog #344 canonical equations registry sister discipline + operator
NON-NEGOTIABLE 2026-05-19 *"we need to formalize all of this and canonicalize
and operationalize because I am afraid we are learning but if we don't have
systems of equations and models and such we are just gaining tribal knowledge"*.

Sister of `triple_substrate_composition_alpha_v1` (#23) at the cross-archive
canvas surface. THIS equation IS the FORMALIZATION_PENDING canonical for the
84-cell 7-archive × 12-operator DROP-MANY canvas matrix landed today.
"""

from __future__ import annotations

from tac.canonical_equations import (
    CanonicalEquation,
    register_canonical_equation,
)
from tac.provenance.builders import build_provenance_for_predicted

CANONICAL_EQUATION_ID = "cross_archive_drop_many_canvas_composition_alpha_v1"

LATEX_FORM = (
    r"\alpha(X, Y) = \min(\max(F(X) \cdot A(Y), 0), 2) "
    r"\text{ where } "
    r"F(X) = \text{archive\_family\_compatibility}(X) \in [0.70, 1.00] "
    r"\text{ and } "
    r"A(Y) = \text{operator\_axis\_attack\_vector}(Y) \in [0.50, 1.10]"
)

ONE_LINE_SUMMARY = (
    "Cross-archive DROP-MANY canvas composition_alpha = clamp(F(X)*A(Y), 0, 2); "
    "84-cell 7-archive x 12-operator matrix; bands per Catalog #322 v2 cascade."
)


def build_equation() -> CanonicalEquation:
    return CanonicalEquation(
        equation_id=CANONICAL_EQUATION_ID,
        name="Cross-archive DROP-MANY canvas composition_alpha",
        one_line_summary=ONE_LINE_SUMMARY,
        latex_form=LATEX_FORM,
        python_callable_module_path=(
            "tools.append_slot_h_cross_archive_84_cell_to_substrate_composition_matrix.build_84_cell_rows"
        ),
        domain_of_validity={
            "default_context_when_legacy_caller_omits": (
                "cross_archive_drop_many_canvas_84_cell_composition_alpha_prediction"
            ),
            "empirical_anchor_status": (
                "design_only_pending_first_paired_cuda_anchor_per_catalog_246_on_HIGH_EV_cells"
            ),
            "in_domain_contexts": [
                "cross_archive_drop_many_canvas_84_cell_composition_alpha_prediction",
                "pr110_pr101_pr106_pr107_dqs1_a1_hdm8_x_12_canonical_operators_canvas",
                "f_archive_compatibility_times_a_operator_attack_vector_decomposition",
                "tier_a_observability_only_scaffold_per_catalog_341_357",
            ],
            "measurement_axes": [
                "[contest-CUDA]",
                "[contest-CPU]",
                "[predicted]",
            ],
            "out_of_domain_contexts": [
                "single_archive_only_per_existing_5d_canvas_extended_operators",
                "pairwise_substrate_alpha_only_per_existing_substrate_composition_matrix",
                "4_or_more_way_compositions_use_triple_substrate_composition_alpha_v1_sister",
            ],
            "sister_equations": [
                "triple_substrate_composition_alpha_v1 (#23 sister at TRIPLE-substrate surface)",
                "cross_substrate_top_k_byte_overlap_predicts_composition_alpha_v1 (#22)",
                "procedural_codebook_from_seed_compression_savings_v1 (#26)",
                "replace_one_via_linear_substitution_distortion_v1 (sister 8-extended-ops)",
                "replace_many_via_beam_search_per_axis_decomposition_v1",
                "merge_pair_via_rate_distortion_joint_optimization_v1",
                "reorder_pair_via_entropy_coder_context_markov_v1",
                "drop_frame_via_per_frame_master_gradient_v1",
                "synthesize_frame_via_atick_redlich_cooperative_receiver_v1",
                "motion_conditional_via_rao_ballard_predictive_coding_v1",
                "temporal_coherence_via_wyner_ziv_side_information_v1",
            ],
            "F_archive_compatibility_constraint": (
                "F(X) in [0.70, 1.00]; 1.00 for frontier-saturated CPU archives "
                "(PR110/PR101/DQS1); 0.95 for CUDA-frontier archives "
                "(PR106/HDM8); 0.80 for HNeRV-baseline (A1); 0.70 for "
                "public-reference (PR107)"
            ),
            "A_operator_attack_vector_constraint": (
                "A(Y) in [0.50, 1.10]; 0.50 for saturating DISTORTION-axis "
                "ops (FULL_DROP/REPAIR); 0.65 for masked/feathered; 0.55 for "
                "REPLACE_ONE; 0.85-0.95 for HIGH-EV DISTORTION+RATE; 1.05-"
                "1.10 for SUPER_ADDITIVE potential (SYNTHESIZE_FRAME / "
                "TEMPORAL_COHERENCE)"
            ),
            "alpha_band_classifier_per_catalog_322_v2_cascade": (
                "SUPER_ADDITIVE > 1.05 / ADDITIVE (0.7, 1.05] / SUB_ADDITIVE "
                "(0.3, 0.7] / SATURATING <= 0.3"
            ),
        },
        units_in={
            "X": "archive_family_id (PR110/PR101/PR106/PR107/DQS1/A1/HDM8)",
            "Y": "canonical_operator_id (12 members: 4 canonical + 8 extended)",
            "F_X": "dimensionless_compatibility_score_in_0p70_to_1p00",
            "A_Y": "dimensionless_attack_vector_score_in_0p50_to_1p10",
        },
        units_out={
            "alpha": "dimensionless_composition_alpha_in_0_to_2_clamped",
            "alpha_band": "categorical_SUPER_ADDITIVE_or_ADDITIVE_or_SUB_ADDITIVE_or_SATURATING",
        },
        empirical_anchors=tuple(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-05-29T05:55:00+00:00",
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=(
            ".omx/state/substrate_composition_matrix.json",
            "tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2",
            "tac.cathedral_autopilot.rank_candidates_via_three_metric_trichotomy",
            "future_cross_archive_paired_cuda_dispatch_consumers_per_catalog_246",
        ),
        canonical_producers=(
            "tools/append_slot_h_cross_archive_84_cell_to_substrate_composition_matrix.py",
            "tools/register_slot_h_cross_archive_drop_many_canvas_canonical_equation.py",
            ".omx/research/cross_archive_drop_many_canvas_7_archive_x_12_operator_84_cell_design_20260529.md",
        ),
        provenance=build_provenance_for_predicted(
            model_id="slot_h_cross_archive_drop_many_canvas_canonical_equation_v1",
            inputs_sha256="bc92414d970b857d6beacf118f276b01b263c059114b6d6b0fc063106724ed89",
            measurement_axis="[predicted]",
            hardware_substrate="scaffold_only_no_paid_dispatch",
            captured_at_utc="2026-05-29T05:55:00+00:00",
        ),
    )


def main() -> int:
    equation = build_equation()
    register_canonical_equation(
        equation,
        agent="claude",
        subagent_id=(
            "slot_h_cascade_item_3_cross_archive_composition_matrix_7_archive_drop_many_canvas_20260529_0030cst"
        ),
        notes=(
            "SLOT H cascade item 3 (of 7-item cascade) — 84-cell 7-archive × "
            "12-operator DROP-MANY canvas composition_alpha matrix. Tier A "
            "scaffold-only per Catalog #341/#357; FORMALIZATION_PENDING per "
            "Catalog #344; sister of triple_substrate_composition_alpha_v1 "
            "at cross-archive surface. Lane lane_slot_h_cascade_item_3_"
            "cross_archive_composition_matrix_7_archive_drop_many_canvas_"
            "20260529 L1."
        ),
    )
    print(f"[slot_h] Registered canonical equation: {CANONICAL_EQUATION_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
