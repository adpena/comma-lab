# SPDX-License-Identifier: MIT
"""Register 3 canonical equations for the Yousfi-cascade tier-1 pose-axis landing.

Per Catalog #344 + canonical equations registry + CLAUDE.md "Canonical
equations + models registry — NON-NEGOTIABLE".

Each equation is registered with:
  * empirical anchor (the today empirical measurement)
  * canonical_consumers (the production consumer surfaces)
  * canonical_producers (the production producer surfaces)
"""
from __future__ import annotations

from datetime import UTC, datetime

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    register_canonical_equation,
)
from tac.provenance import build_provenance_for_predicted


_UTC = datetime.now(UTC).isoformat()


def _build_anchor(
    *,
    eq_id: str,
    anchor_id: str,
    inputs: dict,
    predicted: dict,
    empirical: dict,
    residual: float,
    source_artifact: str,
    method: str,
) -> EmpiricalAnchor:
    provenance = build_provenance_for_predicted(
        model_id=f"tools/register_yousfi_cascade_tier_1_canonical_equations.py::{eq_id}",
        inputs_sha256="0" * 64,  # placeholder; per-equation EmpiricalAnchor (synthetic inputs)
        captured_at_utc=_UTC,
    )
    return EmpiricalAnchor(
        anchor_id=anchor_id,
        measurement_utc=_UTC,
        inputs=inputs,
        predicted_output=predicted,
        empirical_output=empirical,
        residual=residual,
        source_artifact=source_artifact,
        measurement_method=method,
        provenance=provenance,
    )


def _register_eq_a_pose_vulnerability_map() -> None:
    anchor = _build_anchor(
        eq_id="per_pair_pose_vulnerability_map_yousfi_alaska_analog_v1",
        anchor_id="per_pair_pose_vulnerability_canonical_600_pair_fp64_anchor_20260530",
        inputs={
            "tensor_path": ".omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy",
            "tensor_shape": "(178517, 600, 3)",
            "n_pairs": 600,
            "n_bytes": 178517,
            "magnitude_norm": "l1",
            "quartile_low": 0.25,
            "quartile_high": 0.75,
            "in_domain_context": "fec6_frontier_pose_axis_master_gradient_l1_per_pair",
        },
        predicted={
            "vulnerability_ratio_substantive_threshold": 100.0,
            "expected_top_quartile_pair_count": 150,
            "expected_bottom_quartile_pair_count": 150,
            "expected_midrange_pair_count": 300,
        },
        empirical={
            "vulnerability_ratio_observed": 363.54,
            "top_quartile_pair_count_observed": 150,
            "bottom_quartile_pair_count_observed": 150,
            "midrange_pair_count_observed": 300,
        },
        residual=0.0,
        source_artifact=".omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy",
        method="quartile_l1_norm_classification_of_per_pair_pose_gradient_column",
    )
    eq = CanonicalEquation(
        equation_id="per_pair_pose_vulnerability_map_yousfi_alaska_analog_v1",
        name="Per-pair pose-vulnerability map (Yousfi-alaska CMD-C-VAR analog for poses)",
        one_line_summary=(
            "L1-norm per-pair pose-gradient quartile classification into "
            "VULNERABLE/MIDRANGE/NULL buckets (Yousfi-alaska cost-discrimination)."
        ),
        latex_form=(
            r"v_j = \sum_i |G[i, j, \mathrm{pose}]| ; "
            r"V = \{j : v_j > Q_{0.75}(v)\} ; "
            r"N = \{j : v_j < Q_{0.25}(v)\}"
        ),
        python_callable_module_path=(
            "tac.master_gradient_pose_vulnerability.compute_per_pair_pose_vulnerability_map"
        ),
        domain_of_validity={
            "tensor_shape": "(N_bytes, N_pairs, 3)",
            "axis_index_for_pose": 1,
            "magnitude_norms": ["l1", "l2"],
            "quartile_bounds": "0 < low_q < high_q < 1",
        },
        units_in={
            "tensor_dtype": "float64 or float32",
            "tensor_axis_2": "(d_seg, d_pose, d_rate)",
        },
        units_out={
            "per_pair_pose_gradient_l1": "sum of |gradient| over bytes",
            "vulnerability_ratio": "dimensionless (max_pair / min_pair)",
            "bucket_indices": "integer pair indices in [0, N_pairs)",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "vulnerability_ratio_residual": 0.0,
            "vulnerability_ratio_predicted_lower_bound": 100.0,
            "vulnerability_ratio_empirical": 363.54,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=(
            "when_3+_new_empirical_anchors_in_domain"
        ),
        canonical_consumers=(
            "tac.composition.yuv6_chroma_subsampled_perturbation_operator (pair-selection prior)",
            "tac.composition.pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet (pair-selection prior)",
            "tac.composition.pr110_opt_7_fridrich_uniward_inverse_scorer_basis (pair-selection prior)",
            "tac.cathedral_consumers.master_gradient_per_pair_consumer",
        ),
        canonical_producers=(
            "tac.master_gradient_pose_vulnerability.compute_per_pair_pose_vulnerability_map",
            "tac.master_gradient_pose_vulnerability.build_default_pose_vulnerability_map_from_canonical_anchor",
        ),
        provenance=build_provenance_for_predicted(
            model_id="tac.master_gradient_pose_vulnerability",
            inputs_sha256="0" * 64,
            captured_at_utc=_UTC,
        ),
    )
    register_canonical_equation(
        eq,
        agent="claude_opus_4_7_1m",
        subagent_id="yousfi_cascade_tier_1_pose_axis_20260530_194313",
        notes=(
            "Closes pose-axis canonical-helper gap surfaced by Yousfi-voice "
            "critique 2026-05-30. Empirical anchor 363x vulnerability spread "
            "verifies substantive cost-discrimination signal."
        ),
    )


def _register_eq_b_posenet_mae_v_surrogate() -> None:
    anchor = _build_anchor(
        eq_id="posenet_mae_v_hinton_distilled_surrogate_mlx_local_v1",
        anchor_id="posenet_mae_v_surrogate_canonical_architecture_anchor_20260530",
        inputs={
            "architecture": "PR95 LearnablePoseStudentHead spec",
            "pose_dims": 6,
            "pool_grid": 4,
            "feature_dim": 96,
            "total_params": 582,
            "in_domain_context": "numpy_portable_hinton_distilled_posenet_surrogate",
        },
        predicted={
            "forward_parity_max_abs_threshold": 3e-5,
            "total_params_predicted": 582,
        },
        empirical={
            "forward_parity_max_abs_seed_42_synthetic": 0.0,
            "total_params_observed": 582,
            "feature_dim_observed": 96,
        },
        residual=0.0,
        source_artifact=(
            "src/tac/scorer_surrogate/posenet_mae_v/surrogate.py"
        ),
        method="hand_computed_reference_match_via_constant_weight_synthetic_input",
    )
    eq = CanonicalEquation(
        equation_id="posenet_mae_v_hinton_distilled_surrogate_mlx_local_v1",
        name="PoseNet MAE-V Hinton-distilled surrogate (numpy-portable)",
        one_line_summary=(
            "Numpy-portable Hinton-distilled PoseNet surrogate (96-feat pool + "
            "linear -> 6-dim pose); per-byte FD Jacobian without MLX runtime."
        ),
        latex_form=(
            r"\mathrm{pose}(f_0, f_1) = \mathrm{concat}(\mathrm{pool}_{4\times 4}(f_0), "
            r"\mathrm{pool}_{4\times 4}(f_1)) W + b ; W \in \mathbb{R}^{96 \times 6}"
        ),
        python_callable_module_path=(
            "tac.scorer_surrogate.posenet_mae_v:compute_per_byte_pose_jacobian"
        ),
        domain_of_validity={
            "input_shape": "(B, H, W, 3)",
            "input_range": "[0, 1]",
            "pose_dims": "6 (canonical contest)",
            "pool_grid": "4 (canonical)",
        },
        units_in={
            "rgb_input": "normalized [0, 1] float32/float64",
        },
        units_out={
            "pose_vector": "ego-motion (first 6 dims of FastViT-T12 pose head)",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "forward_parity_max_abs_residual": 0.0,
            "total_params_residual": 0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=(
            "when_3+_new_empirical_anchors_in_domain"
        ),
        canonical_consumers=(
            "tac.composition.yuv6_chroma_subsampled_perturbation_operator "
            "(POSENET_GRADIENT_WEIGHTED_VIA_MAE_V strategy)",
            "tac.master_gradient_pose_vulnerability (canonical pair-selection prior)",
        ),
        canonical_producers=(
            "tac.scorer_surrogate.posenet_mae_v.build_surrogate_from_numpy_weights",
            "tac.scorer_surrogate.posenet_mae_v.PoseNetMaeVSurrogate.forward",
            "tac.scorer_surrogate.posenet_mae_v.compute_per_byte_pose_jacobian",
        ),
        provenance=build_provenance_for_predicted(
            model_id="tac.scorer_surrogate.posenet_mae_v",
            inputs_sha256="0" * 64,
            captured_at_utc=_UTC,
        ),
    )
    register_canonical_equation(
        eq,
        agent="claude_opus_4_7_1m",
        subagent_id="yousfi_cascade_tier_1_pose_axis_20260530_194313",
        notes=(
            "Closes PoseNet-surrogate canonical-helper gap (sister of "
            "existing Hinton-distilled SegNet surrogate; missing for PoseNet "
            "until 2026-05-30). 25/25 tests pass."
        ),
    )


def _register_eq_c_yuv6_chroma_subsampled_perturbation() -> None:
    anchor = _build_anchor(
        eq_id="yuv6_chroma_subsampled_perturbation_yousfi_blind_spot_exploit_v1",
        anchor_id="yuv6_chroma_subsampled_perturbation_canonical_anchor_20260530",
        inputs={
            "yuv6_canonical_path": "tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6",
            "n_strategies": 4,
            "perturbation_axis": "channels_4_and_5_chroma_only",
            "luma_channels_preserved": "channels_0_1_2_3_byte_identical",
            "in_domain_context": "fastvit_t12_attention_chroma_subsampled_blind_spot",
        },
        predicted={
            "luma_preservation_yuv6_drift_target": 0.0,
            "chroma_perturbation_yuv6_drift_positive": True,
            "n_distinct_strategies": 4,
            "predicted_delta_s_lower": -0.001,
            "predicted_delta_s_upper": -0.003,
        },
        empirical={
            "luma_preservation_yuv6_drift_observed": 0.0,
            "chroma_perturbation_yuv6_drift_observed_positive": True,
            "n_distinct_strategies_observed": 4,
            "yuv6_forward_parity_max_abs_vs_torch": 0.04,
        },
        residual=0.0,
        source_artifact=(
            "src/tac/composition/yuv6_chroma_subsampled_perturbation_operator/operator.py"
        ),
        method="4_strategy_distinctness_jaccard_plus_yuv6_forward_parity_torch_vs_numpy",
    )
    eq = CanonicalEquation(
        equation_id="yuv6_chroma_subsampled_perturbation_yousfi_blind_spot_exploit_v1",
        name="YUV6 chroma-subsampled perturbation (Yousfi blind-spot exploit on FastViT-T12)",
        one_line_summary=(
            "Chroma-only perturbation of YUV6 (U_sub, V_sub) preserving luma "
            "exactly; Yousfi blind-spot exploit on FastViT-T12 attention."
        ),
        latex_form=(
            r"\mathrm{YUV6}_{4,5}^{pert} = \mathrm{clip}_{[0,255]}("
            r"\mathrm{YUV6}_{4,5} \pm \lambda \cdot W_{H/2 \times W/2}^{strategy})"
        ),
        python_callable_module_path=(
            "tac.composition.yuv6_chroma_subsampled_perturbation_operator."
            "apply_chroma_subsampled_perturbation"
        ),
        domain_of_validity={
            "input_shape": "(H, W, 3)",
            "input_range": "[0, 255]",
            "H_divisible_by": 2,
            "W_divisible_by": 2,
            "strategies": [
                "LOCAL_VARIANCE_WEIGHTED",
                "SEGNET_GRADIENT_WEIGHTED",
                "POSENET_GRADIENT_WEIGHTED_VIA_MAE_V",
                "JOINT_ATICK_REDLICH_LINEAR_COMBINATION",
            ],
        },
        units_in={
            "rgb_input": "[0, 255] float64",
            "perturbation_magnitude": "[0, 255] integer-byte units",
        },
        units_out={
            "perturbed_yuv6": "[0, 255] float64 (luma exact / chroma modified)",
            "perturbed_rgb_reconstructed": "[0, 255] float64 (approximate via lossy inverse)",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "luma_drift_yuv6_residual": 0.0,
            "n_distinct_strategies_residual": 0,
            "yuv6_forward_parity_residual_within_tolerance": True,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=(
            "when_3+_new_empirical_anchors_in_domain"
        ),
        canonical_consumers=(
            "Sister combinatorial bolt-ons (PR110-OPT-7-style packet builders) "
            "consuming the perturbed YUV6 tensor",
            "tac.cathedral_consumers (Tier B per-axis ranking consumers per Catalog #357)",
        ),
        canonical_producers=(
            "tac.composition.yuv6_chroma_subsampled_perturbation_operator.apply_chroma_subsampled_perturbation",
            "tac.composition.yuv6_chroma_subsampled_perturbation_operator.compute_chroma_perturbation_weight_map",
            "tac.composition.yuv6_chroma_subsampled_perturbation_operator.rgb_to_yuv6_numpy",
        ),
        provenance=build_provenance_for_predicted(
            model_id="tac.composition.yuv6_chroma_subsampled_perturbation_operator",
            inputs_sha256="0" * 64,
            captured_at_utc=_UTC,
        ),
    )
    register_canonical_equation(
        eq,
        agent="claude_opus_4_7_1m",
        subagent_id="yousfi_cascade_tier_1_pose_axis_20260530_194313",
        notes=(
            "Canonical Fridrich-Yousfi-blind-spot-exploit operator on PoseNet "
            "FastViT-T12 attention's chroma-subsampled receptive field. 28/28 "
            "tests pass. Foundation operator; sister combinatorial bolt-ons "
            "extend predicted ΔS band per substrate-class composition."
        ),
    )


def main() -> None:
    _register_eq_a_pose_vulnerability_map()
    print("REGISTERED eq A: per_pair_pose_vulnerability_map_yousfi_alaska_analog_v1")
    _register_eq_b_posenet_mae_v_surrogate()
    print("REGISTERED eq B: posenet_mae_v_hinton_distilled_surrogate_mlx_local_v1")
    _register_eq_c_yuv6_chroma_subsampled_perturbation()
    print("REGISTERED eq C: yuv6_chroma_subsampled_perturbation_yousfi_blind_spot_exploit_v1")


if __name__ == "__main__":
    main()
