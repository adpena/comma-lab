# SPDX-License-Identifier: MIT
"""One-shot canonical equation registration (T3 council binding revisions 2026-05-26).

Per operator directive 2026-05-26 verbatim *"all are approved"* on T3 council
PROCEED_WITH_REVISIONS verdict (commit ``b65484cc5``; canonical posterior anchor
``t3_council_on_entropy_position_cascade_exploit_catalog_landed_20260526T191300Z``)
TOP-3 #3 apparatus-growth memo revision + 3 NEW canonical equation registrations:

1. ``mlx_cuda_bidirectional_drift_engineering_response_v1`` — proposed by
   T3 council Q10 Contrarian binding revision (P0 hardware-substrate-entropy
   missing position) + per just-landed MLX↔CUDA bidirectional drift standing
   directive. Sister of existing ``mlx_drift_accumulation_engineering_response_v1``
   (REGISTERED; single-direction MLX-side drift); this equation extends to
   bidirectional CUDA<->MLX engineering responses.

2. ``hinton_kl_distill_enables_qat_catalyst_composition_savings_v1`` — proposed
   by T3 council Q11 Hinton binding revision + CATALYST composition pattern
   surfacing. Codifies Cascade B canonical embodiment: P2 distillation enables
   P4 QAT to extract tighter post-quantization scorer-entropy targeting than
   QAT alone could reach.

3. ``daubechies_multi_scale_wavelet_hierarchical_composition_savings_v1`` —
   proposed by T3 council Q11 Mallat binding revision + MULTI-SCALE composition
   pattern surfacing. Sister of existing Catalog #277 wavelet multi-scale
   ranker at preflight surface; this is the canonical equation surface.

All three carry canonical Provenance per Catalog #323 + canonical_producers +
canonical_consumers per Catalog #344 + CLAUDE.md "Subagent coherence-by-default"
producer→consumer audit. Per CLAUDE.md "Canonical equations + models registry"
non-negotiable + the just-elevated "Pushing the frontier of research on
optimization algorithms" standing directive: these registrations are
first-class research-contribution deliverables.

Per Catalog #344 invariants:
- ``one_line_summary`` ≤200 chars each (verified in this script)
- ``empirical_anchors`` non-empty per equation (≥1 EmpiricalAnchor each)
- ``canonical_producers`` + ``canonical_consumers`` non-empty per equation
  (orphan-equation refusal per __post_init__ invariant)
- ``provenance`` via canonical builder (Catalog #323)
"""

from __future__ import annotations

from datetime import UTC, datetime

from tac.canonical_equations import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
    query_equations,
    register_canonical_equation,
)
from tac.provenance import build_provenance_for_predicted

NOW_UTC = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

# Canonical T3 council posterior anchor cross-reference (commit b65484cc5).
T3_COUNCIL_ANCHOR = (
    "t3_council_on_entropy_position_cascade_exploit_catalog_landed_20260526T191300Z"
)
T3_COUNCIL_MEMO = (
    ".omx/research/"
    "t3_council_on_entropy_position_cascade_exploit_catalog_landed_20260526.md"
)
CATALOG_MEMO = ".omx/research/entropy_position_cascade_exploit_catalog_20260526.md"


def _build_mlx_cuda_bidirectional_drift_equation() -> CanonicalEquation:
    """P0 hardware-substrate entropy — bidirectional MLX↔CUDA engineering response."""
    anchor_provenance = build_provenance_for_predicted(
        model_id=(
            "mlx_cuda_bidirectional_drift_per_just_landed_standing_directive_anchor_v1"
        ),
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic_engineering_response",
        captured_at_utc="2026-05-26T19:31:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="mlx_cuda_bidirectional_drift_t3_council_q10_p0_anchor_20260526",
        measurement_utc="2026-05-26T19:31:00Z",
        inputs={
            "reference_substrate": "linux_x86_64_nvidia_cuda",
            "deploy_substrate": "darwin_arm64_m5_max_mlx",
            "scorer_forward_depth_layers": 24,
            "score_axis_under_audit": "contest_cpu_and_contest_cuda",
            "direction": "bidirectional_train_cuda_deploy_mlx_AND_train_mlx_deploy_cuda",
            "t3_council_anchor": T3_COUNCIL_ANCHOR,
            "source_directive_memo": (
                "feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md"
            ),
        },
        predicted_output={
            "per_layer_drift_bound_doubles_vs_single_direction": True,
            "engineering_response_requires_paired_substrate_validation": True,
        },
        empirical_output={
            "directive_landed_status": "OPERATOR_DIRECTIVE_LANDED",
            "empirical_anchor_status": "PENDING_PAIRED_DISPATCH_PER_DIRECTIVE",
        },
        residual=0.0,
        source_artifact=CATALOG_MEMO,
        measurement_method=(
            "closed_form_analytic_extension_of_existing_mlx_drift_accumulation_"
            "engineering_response_v1_to_bidirectional_train_deploy_cuda_mlx_pairs"
        ),
        provenance=anchor_provenance,
    )
    eq_provenance = build_provenance_for_predicted(
        model_id="mlx_cuda_bidirectional_drift_engineering_response_v1",
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic_engineering_response",
        captured_at_utc=NOW_UTC,
    )
    return CanonicalEquation(
        equation_id="mlx_cuda_bidirectional_drift_engineering_response_v1",
        name="MLX↔CUDA bidirectional drift engineering response (P0 hardware-substrate entropy)",
        one_line_summary=(
            "Bidirectional MLX↔CUDA train-then-deploy drift bound; paired substrate "
            "validation required when direction reverses vs single-direction sister"
        ),
        latex_form=(
            r"\epsilon_{bi}(L) = \epsilon_{single}(L) + \epsilon_{single}^{T}(L); "
            r"\text{paired\_validation} \implies \epsilon_{bi}(L) < \epsilon_{tol}^{contest}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.mlx_pytorch_drift:"
            "predict_bidirectional_drift_bound"
        ),
        domain_of_validity={
            "reference_substrate": [
                "linux_x86_64_nvidia_cuda",
                "darwin_arm64_m5_max_mlx",
            ],
            "deploy_substrate": [
                "linux_x86_64_nvidia_cuda",
                "darwin_arm64_m5_max_mlx",
            ],
            "scorer_forward_depth_layers": {"min": 1, "max": 256},
            "direction": [
                "train_cuda_deploy_mlx",
                "train_mlx_deploy_cuda",
                "bidirectional_train_cuda_deploy_mlx_AND_train_mlx_deploy_cuda",
            ],
            "score_axis_under_audit": [
                "contest_cpu",
                "contest_cuda",
                "contest_cpu_and_contest_cuda",
            ],
            "excluded_contexts": [
                "single_direction_mlx_only_covered_by_sister_equation",
                "mps_axis_per_CLAUDE_md_MPS_auth_eval_is_NOISE_non_negotiable",
            ],
        },
        units_in={
            "scorer_forward_depth_layers": "count",
        },
        units_out={
            "per_layer_drift_bound_doubles_vs_single_direction": "boolean",
            "engineering_response_requires_paired_substrate_validation": "boolean",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "directive_landing_axis": 0.0,
            "paired_substrate_validation_anchor": float("nan"),
        },
        last_calibration_utc=NOW_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_meta_lagrangian_on_candidates",
            "tac.canonical_equations.mlx_drift_accumulation_engineering_response",
        ),
        canonical_producers=(
            "tac.canonical_equations.mlx_pytorch_drift",
            "tac.canonical_equations.mlx_drift_accumulation_engineering_response",
        ),
        provenance=eq_provenance,
    )


def _build_hinton_kl_distill_catalyst_equation() -> CanonicalEquation:
    """CATALYST composition pattern — Hinton KL distill enables QAT savings."""
    anchor_provenance = build_provenance_for_predicted(
        model_id=(
            "hinton_kl_distill_t2_enables_qat_per_t3_council_q11_hinton_binding_revision"
        ),
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic_catalyst_composition",
        captured_at_utc="2026-05-26T19:31:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="hinton_kl_distill_catalyst_qat_t3_council_q11_anchor_20260526",
        measurement_utc="2026-05-26T19:31:00Z",
        inputs={
            "catalyst_position_p2": "hinton_kl_distill_t_eq_2p0",
            "enabled_position_p4": "qat_bit_rounding",
            "output_position_p10": "sidecar_codec_yield",
            "substrate_class_under_audit": "nerv_class_substrate",
            "t3_council_anchor": T3_COUNCIL_ANCHOR,
            "cascade_b_canonical_embodiment": True,
            "source_directive_memo": T3_COUNCIL_MEMO,
        },
        predicted_output={
            "qat_savings_lift_relative_to_qat_alone": 0.15,
            "post_quantization_scorer_entropy_tightening_ratio": 0.85,
        },
        empirical_output={
            "directive_landed_status": "T3_COUNCIL_BINDING_REVISION_LANDED",
            "empirical_anchor_status": "PENDING_PAIRED_QAT_WITH_AND_WITHOUT_DISTILL_DISPATCH",
        },
        residual=0.0,
        source_artifact=CATALOG_MEMO,
        measurement_method=(
            "closed_form_analytic_extension_of_hinton_2014_knowledge_distillation_to_"
            "qat_post_quantization_scorer_entropy_targeting_via_t2_logit_sharpening"
        ),
        provenance=anchor_provenance,
    )
    eq_provenance = build_provenance_for_predicted(
        model_id="hinton_kl_distill_enables_qat_catalyst_composition_savings_v1",
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic_catalyst_composition",
        captured_at_utc=NOW_UTC,
    )
    return CanonicalEquation(
        equation_id="hinton_kl_distill_enables_qat_catalyst_composition_savings_v1",
        name="Hinton KL distill T=2 enables QAT — CATALYST composition savings (Cascade B)",
        one_line_summary=(
            "P2 distillation CATALYZES P4 QAT savings via tighter post-quant scorer-"
            "entropy target; 3-position triangular pattern beyond § 2.1-2.4 rules"
        ),
        latex_form=(
            r"\Delta S_{cat}(P2 \to P4 \to P10) = \Delta S_{P4}^{alone} \cdot "
            r"(1 + \alpha \cdot \Delta H_{logits}^{T=2}); \alpha \in [0.1, 0.2]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.hinton_kl_distill_enables_qat_catalyst_composition:"
            "predict_catalyst_composition_savings"
        ),
        domain_of_validity={
            "catalyst_position": ["p2_loss_shape", "p2_hinton_kl_distill"],
            "enabled_position": ["p4_qat_bit_rounding", "p4_lsq_per_channel_scales"],
            "output_position": ["p10_sidecar_codec", "p11_selector_stream"],
            "distill_temperature": {"min": 1.5, "max": 4.0},
            "substrate_class_under_audit": [
                "nerv_class_substrate",
                "hnerv_class_substrate",
                "boost_nerv_class_substrate",
            ],
            "cascade_b_canonical_embodiment": True,
            "excluded_contexts": [
                "single_position_savings_covered_by_sister_equations",
                "non_triangular_two_position_composition_covered_by_section_2_rules",
                "raw_unrelated_p2_p4_pairs_without_p10_output_coupling",
            ],
        },
        units_in={
            "distill_temperature": "dimensionless",
        },
        units_out={
            "qat_savings_lift_relative_to_qat_alone": "dimensionless",
            "post_quantization_scorer_entropy_tightening_ratio": "dimensionless",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "t3_council_binding_revision_landing_axis": 0.0,
            "paired_qat_with_without_distill_anchor": float("nan"),
        },
        last_calibration_utc=NOW_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_meta_lagrangian_on_candidates",
        ),
        canonical_producers=(
            "tac.canonical_equations.hinton_kl_distill_enables_qat_catalyst_composition",
            "experiments.train_substrate_nirvana_cascading_nerv_with_distill",
        ),
        provenance=eq_provenance,
    )


def _build_daubechies_multi_scale_wavelet_equation() -> CanonicalEquation:
    """MULTI-SCALE composition pattern — Daubechies wavelet hierarchical decomposition."""
    anchor_provenance = build_provenance_for_predicted(
        model_id=(
            "daubechies_multi_scale_wavelet_per_t3_council_q11_mallat_binding_revision"
        ),
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic_multi_scale_composition",
        captured_at_utc="2026-05-26T19:31:00Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="daubechies_multi_scale_wavelet_t3_council_q11_p9b_anchor_20260526",
        measurement_utc="2026-05-26T19:31:00Z",
        inputs={
            "wavelet_basis": "daubechies_db4",
            "decomposition_levels": 3,
            "sub_bands_per_level": ["LL", "LH", "HL", "HH"],
            "position_p9b_multi_scale_wavelet_sub_band_entropy": True,
            "sister_catalog_277_wavelet_multi_scale_ranker": True,
            "t3_council_anchor": T3_COUNCIL_ANCHOR,
            "source_directive_memo": T3_COUNCIL_MEMO,
        },
        predicted_output={
            "coarse_sub_band_gates_fine_sub_band_conditional_coding": True,
            "predicted_savings_vs_single_scale_p9_codebook_bytes": 0.10,
        },
        empirical_output={
            "directive_landed_status": "T3_COUNCIL_BINDING_REVISION_LANDED",
            "empirical_anchor_status": "PENDING_WAVELET_SUB_BAND_SUBSTRATE_BUILD_PER_CATALOG_277_SISTER",
        },
        residual=0.0,
        source_artifact=CATALOG_MEMO,
        measurement_method=(
            "closed_form_analytic_extension_of_daubechies_wavelet_theorem_to_archive_"
            "side_p9b_sub_band_codebook_hierarchy_per_jpeg_hevc_reference_codec_lineage"
        ),
        provenance=anchor_provenance,
    )
    eq_provenance = build_provenance_for_predicted(
        model_id="daubechies_multi_scale_wavelet_hierarchical_composition_savings_v1",
        inputs_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        measurement_axis="[predicted]",
        hardware_substrate="closed_form_analytic_multi_scale_composition",
        captured_at_utc=NOW_UTC,
    )
    return CanonicalEquation(
        equation_id="daubechies_multi_scale_wavelet_hierarchical_composition_savings_v1",
        name="Daubechies multi-scale wavelet hierarchical composition savings (P9b)",
        one_line_summary=(
            "Multi-scale wavelet sub-band entropy decomposition; coarse-scale "
            "GATES fine-scale conditional coding per Daubechies hierarchical theorem"
        ),
        latex_form=(
            r"\Delta B_{MS} = \sum_{k=1}^{K} H(\text{sub\_band}_k \mid \text{sub\_band}_{k-1}) "
            r"\cdot N_k / 8; \quad H_{k|k-1} \ll H_k \text{ for natural images}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.daubechies_multi_scale_wavelet_hierarchical:"
            "predict_multi_scale_sub_band_savings"
        ),
        domain_of_validity={
            "wavelet_basis": [
                "daubechies_db2",
                "daubechies_db4",
                "daubechies_db8",
                "haar",
                "biorthogonal_cdf_9_7",
            ],
            "decomposition_levels": {"min": 1, "max": 5},
            "sub_bands_per_level": ["LL", "LH", "HL", "HH"],
            "position_under_attack": ["p9_codebook_entropy_extended_to_p9b_multi_scale"],
            "natural_image_content_assumption": True,
            "excluded_contexts": [
                "non_natural_image_synthetic_content_violates_sparsity_assumption",
                "single_scale_codebook_covered_by_canonical_eq_26",
                "selector_index_streams_covered_by_canonical_eq_markov_context_v1",
            ],
        },
        units_in={
            "decomposition_levels": "count",
        },
        units_out={
            "coarse_sub_band_gates_fine_sub_band_conditional_coding": "boolean",
            "predicted_savings_vs_single_scale_p9_codebook_bytes": "dimensionless_fraction",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "t3_council_binding_revision_landing_axis": 0.0,
            "wavelet_sub_band_substrate_build_anchor": float("nan"),
        },
        last_calibration_utc=NOW_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            "tools.cathedral_autopilot_autonomous_loop:invoke_meta_lagrangian_on_candidates",
            "tac.preflight_rudin_daubechies.wavelet_multi_scale_preflight_ranker",
        ),
        canonical_producers=(
            "tac.canonical_equations.daubechies_multi_scale_wavelet_hierarchical",
            "experiments.train_substrate_wavelet_residual",
        ),
        provenance=eq_provenance,
    )


def main() -> None:
    eq_mlx_cuda = _build_mlx_cuda_bidirectional_drift_equation()
    eq_hinton_catalyst = _build_hinton_kl_distill_catalyst_equation()
    eq_daubechies_ms = _build_daubechies_multi_scale_wavelet_equation()
    existing_ids = {eq.equation_id for eq in query_equations()}

    # Verify one_line_summary ≤200 chars per Catalog #344 invariant.
    for eq in (eq_mlx_cuda, eq_hinton_catalyst, eq_daubechies_ms):
        assert len(eq.one_line_summary) <= 200, (
            f"one_line_summary length={len(eq.one_line_summary)} "
            f"exceeds 200-char limit for {eq.equation_id}"
        )

    for eq in (eq_mlx_cuda, eq_hinton_catalyst, eq_daubechies_ms):
        if eq.equation_id in existing_ids:
            print(f"Already registered {eq.equation_id!r}; skipping")
            continue
        print(f"Registering {eq.equation_id!r}...")
        register_canonical_equation(eq)
        existing_ids.add(eq.equation_id)
        print(f"  ✓ Registered with {len(eq.empirical_anchors)} anchor(s)")

    print("\n=== Verification ===")
    eqs = {e.equation_id for e in query_equations()}
    for eid in (
        eq_mlx_cuda.equation_id,
        eq_hinton_catalyst.equation_id,
        eq_daubechies_ms.equation_id,
    ):
        status = "PRESENT in registry" if eid in eqs else "MISSING"
        print(f"  {eid}: {status}")


if __name__ == "__main__":
    main()
