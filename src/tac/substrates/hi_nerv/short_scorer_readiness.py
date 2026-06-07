# SPDX-License-Identifier: MIT
"""HiNeRV short scorer-smoke readiness contract.

This module is intentionally torch/MLX-free.  It consumes already-produced
trainer metrics and receiver-cache reports, then decides whether a short local
smoke is informative enough to unlock a longer MLX run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA = "hi_nerv_short_scorer_smoke_readiness.v1"
HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_OCCUPIED_CLASS_FRACTION = 0.400001
HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_TARGET_CLASS_COVERAGE_FRACTION = 0.8
HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_TARGET_CLASS_MIN_RATIO = 0.2
HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY = (
    "false_authority_macos_mlx_training_no_contest_score_claim"
)
HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG = "[macOS-MLX research-signal]"
HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON = 1.0e-6
HI_NERV_CONTEST_ORIGINAL_VIDEO_BYTES = 37_545_489
HI_NERV_CONTEST_RATE_SCORE_PER_BYTE = 25.0 / HI_NERV_CONTEST_ORIGINAL_VIDEO_BYTES
HI_NERV_CONTEST_SEGNET_PIXEL_SCORE_WEIGHT = 100.0
NERV_SOURCE_QUALIFIED_METRICS_SCHEMA = "nerv_source_qualified_metrics.v1"
_SEGNET_DIRECT_LIVE_SUBCONTROL_DUAL_KEYS = {
    "segnet_direct_live_class_histogram_weight": (
        "hi_nerv_segnet_direct_live_class_histogram"
    ),
    "segnet_direct_live_class_balanced_hinge_weight": (
        "hi_nerv_segnet_direct_live_class_balanced_hinge"
    ),
    "segnet_direct_live_class_balanced_ce_weight": (
        "hi_nerv_segnet_direct_live_class_balanced_ce"
    ),
    "segnet_direct_live_class_balanced_squared_hinge_weight": (
        "hi_nerv_segnet_direct_live_class_balanced_squared_hinge"
    ),
    "segnet_direct_live_class_region_recon_weight": (
        "hi_nerv_segnet_direct_live_class_region_recon"
    ),
    "segnet_direct_live_rare_class_logit_weight": (
        "hi_nerv_segnet_direct_live_rare_class_logit"
    ),
    "segnet_direct_live_target_mass_floor_weight": (
        "hi_nerv_segnet_direct_live_target_mass_floor"
    ),
    "segnet_direct_live_target_min_ratio_floor_weight": (
        "hi_nerv_segnet_direct_live_target_min_ratio_floor"
    ),
}
_SEGNET_DIRECT_LIVE_SUBCONTROL_ACTIVE_WEIGHT_KEYS = {
    "segnet_direct_live_class_histogram_weight": (
        "active_loss_weight__segnet_direct_live_class_histogram"
    ),
    "segnet_direct_live_class_balanced_hinge_weight": (
        "active_loss_weight__segnet_direct_live_class_balanced_hinge"
    ),
    "segnet_direct_live_class_balanced_ce_weight": (
        "active_loss_weight__segnet_direct_live_class_balanced_ce"
    ),
    "segnet_direct_live_class_balanced_squared_hinge_weight": (
        "active_loss_weight__segnet_direct_live_class_balanced_squared_hinge"
    ),
    "segnet_direct_live_class_region_recon_weight": (
        "active_loss_weight__segnet_direct_live_class_region_recon"
    ),
    "segnet_direct_live_rare_class_logit_weight": (
        "active_loss_weight__segnet_direct_live_rare_class_logit"
    ),
    "segnet_direct_live_target_mass_floor_weight": (
        "active_loss_weight__segnet_direct_live_target_mass_floor"
    ),
    "segnet_direct_live_target_min_ratio_floor_weight": (
        "active_loss_weight__segnet_direct_live_target_min_ratio_floor"
    ),
}
_SECTION_BYTE_PRICED_ONLY_CONSTRAINT_KEYS = frozenset(
    {
        "hi_nerv_hiv1_header_section_bytes",
        "hi_nerv_meta_json_section_bytes",
    }
)

_SECTION_BYTE_PRESSURE_KEYS_BY_KIND = {
    "decoder": (
        "coder_qat_c1a_entropy",
        "coder_qat_quant_residual",
        "coder_qat_delta",
        "coder_qat_magnitude",
    ),
    "latent": (
        "latent_qat_c1a_entropy",
        "latent_qat_quant_residual",
        "latent_qat_delta",
        "latent_qat_magnitude",
        # Current HiNeRV latent sections are packaged from the same exported
        # state. Until separate latent QAT terms are attached, decoder QAT is
        # the real differentiable pressure available to the shared loss.
        "coder_qat_c1a_entropy",
        "coder_qat_quant_residual",
        "coder_qat_delta",
        "coder_qat_magnitude",
    ),
}


def build_hinerv_short_scorer_smoke_readiness_report(
    *,
    train_time_controls: Any,
    final_loss_components: Mapping[str, Any] | None,
    post_export_quality: Mapping[str, Any] | None,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    allow_mock_scorer_teacher: bool,
    unscored_research_smoke_enabled: bool = False,
    require_section_byte_dual_ascent: bool = False,
    require_pose_direct_live_distillation: bool = False,
    decoder_weight_waterfill_plan_metadata: Mapping[str, Any] | None = None,
    output_head_target_bias_init_metadata: Mapping[str, Any] | None = None,
    min_segnet_occupied_class_fraction_for_fit_gate: float = (
        HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_OCCUPIED_CLASS_FRACTION
    ),
    min_segnet_target_class_coverage_fraction_for_fit_gate: float = (
        HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_TARGET_CLASS_COVERAGE_FRACTION
    ),
    min_segnet_target_class_min_ratio_for_fit_gate: float = (
        HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_TARGET_CLASS_MIN_RATIO
    ),
) -> dict[str, Any]:
    min_occupied = _finite_float(min_segnet_occupied_class_fraction_for_fit_gate)
    if min_occupied is None or not 0.0 <= min_occupied <= 1.0:
        min_occupied = HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_OCCUPIED_CLASS_FRACTION
    min_target_coverage = _finite_float(
        min_segnet_target_class_coverage_fraction_for_fit_gate
    )
    if min_target_coverage is None or not 0.0 <= min_target_coverage <= 1.0:
        min_target_coverage = (
            HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_TARGET_CLASS_COVERAGE_FRACTION
        )
    min_target_min_ratio = _finite_float(
        min_segnet_target_class_min_ratio_for_fit_gate
    )
    if min_target_min_ratio is None or not 0.0 <= min_target_min_ratio <= 1.0:
        min_target_min_ratio = (
            HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_TARGET_CLASS_MIN_RATIO
        )
    final_components = {
        str(key): finite
        for key, value in (final_loss_components or {}).items()
        if (finite := _finite_float(value)) is not None
    }
    actionable_blockers: list[str] = []

    def add_blocker(blocker: str) -> None:
        if blocker not in actionable_blockers:
            actionable_blockers.append(blocker)

    direct_live_weight = _control_float(
        train_time_controls,
        "segnet_direct_live_distillation_weight",
    )
    direct_live_subcontrol_metric_keys = {
        "segnet_direct_live_class_histogram_weight": (
            "loss_part_segnet_direct_live_class_histogram_loss"
        ),
        "segnet_direct_live_class_balanced_hinge_weight": (
            "loss_part_segnet_direct_live_class_balanced_hinge_loss"
        ),
        "segnet_direct_live_class_balanced_ce_weight": (
            "loss_part_segnet_direct_live_class_balanced_ce_loss"
        ),
        "segnet_direct_live_class_balanced_squared_hinge_weight": (
            "loss_part_segnet_direct_live_class_balanced_squared_hinge_loss"
        ),
        "segnet_direct_live_class_region_recon_weight": (
            "loss_part_segnet_direct_live_class_region_recon_loss"
        ),
        "segnet_direct_live_rare_class_logit_weight": (
            "loss_part_segnet_direct_live_rare_class_logit_loss"
        ),
        "segnet_direct_live_target_mass_floor_weight": (
            "loss_part_segnet_direct_live_target_mass_floor_loss"
        ),
        "segnet_direct_live_target_min_ratio_floor_weight": (
            "loss_part_segnet_direct_live_target_min_ratio_floor_loss"
        ),
    }
    direct_live_subcontrol_weights = {
        key: _control_float(train_time_controls, key)
        for key in direct_live_subcontrol_metric_keys
    }
    direct_live_subcontrol_stage_active_weights = {
        key: _finite_mapping_value(
            final_components,
            _SEGNET_DIRECT_LIVE_SUBCONTROL_ACTIVE_WEIGHT_KEYS[key],
        )
        for key in direct_live_subcontrol_metric_keys
    }
    active_direct_live_subcontrol_control_keys = {
        control_key
        for control_key in direct_live_subcontrol_metric_keys
        if _direct_live_subcontrol_active_for_stage(
            configured_weight=direct_live_subcontrol_weights[control_key],
            stage_active_weight=direct_live_subcontrol_stage_active_weights[
                control_key
            ],
        )
    }
    active_direct_live_subcontrol_metric_keys = {
        metric_key
        for control_key, metric_key in direct_live_subcontrol_metric_keys.items()
        if control_key in active_direct_live_subcontrol_control_keys
    }
    target_min_ratio_floor_control_active = (
        "segnet_direct_live_target_min_ratio_floor_weight"
        in active_direct_live_subcontrol_control_keys
    )
    target_region_debt_dynamics_gate = _target_region_debt_dynamics_gate(
        final_components
    )
    direct_live_subcontrol_enabled = bool(active_direct_live_subcontrol_metric_keys)
    direct_live_enabled = direct_live_weight > 0.0 or direct_live_subcontrol_enabled
    pose_direct_live_weight = _control_float(
        train_time_controls,
        "pose_direct_live_distillation_weight",
    )
    pose_direct_live_enabled = pose_direct_live_weight > 0.0
    segnet_weight = _finite_float(segnet_distillation_weight)
    pose_weight = _finite_float(pose_distillation_weight)
    generic_segnet_requested = bool(segnet_weight is not None and segnet_weight > 0.0)
    generic_posenet_requested = bool(pose_weight is not None and pose_weight > 0.0)
    segnet_direct_live_only = bool(direct_live_enabled and not generic_segnet_requested)
    posenet_direct_live_only = bool(
        pose_direct_live_enabled and not generic_posenet_requested
    )
    if bool(allow_mock_scorer_teacher):
        add_blocker("hi_nerv_short_smoke_mock_scorer_teacher_enabled")
    if bool(unscored_research_smoke_enabled):
        add_blocker("hi_nerv_short_smoke_unscored_research_smoke_enabled")
    if not generic_segnet_requested and not direct_live_enabled:
        add_blocker("hi_nerv_short_smoke_real_segnet_teacher_not_requested")
    if not generic_posenet_requested and not pose_direct_live_enabled:
        add_blocker("hi_nerv_short_smoke_real_posenet_teacher_not_requested")
    pose_distill_keys = (
        "loss_part_pose_score_term",
        "loss_part_pose_distill_raw_mse",
        "loss_part_pose_distill_score_marginal_wrt_raw_mse",
        "loss_part_pose_score_marginal_wrt_raw_mse",
    )
    pose_distill_metrics = {
        key: _finite_mapping_value(final_components, key)
        for key in pose_distill_keys
    }
    if generic_posenet_requested:
        required_pose_distill_keys = (
            "loss_part_pose_score_term",
            "loss_part_pose_distill_raw_mse",
            "loss_part_pose_distill_score_marginal_wrt_raw_mse",
        )
        missing_pose_distill = [
            key
            for key in required_pose_distill_keys
            if pose_distill_metrics[key] is None
        ]
        if missing_pose_distill:
            add_blocker("hi_nerv_short_smoke_missing_posenet_distill_telemetry")
    direct_live_keys = tuple(
        dict.fromkeys(
            (
                "loss_part_segnet_direct_live_distill",
                "loss_part_segnet_direct_live_argmax_disagreement",
                "loss_part_segnet_direct_live_candidate_occupied_class_fraction",
                "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction",
                "loss_part_segnet_direct_live_candidate_target_class_missing_fraction",
                "loss_part_segnet_direct_live_candidate_target_class_min_ratio",
                *sorted(active_direct_live_subcontrol_metric_keys),
            )
        )
    )
    direct_live_metrics = {
        key: _finite_mapping_value(final_components, key) for key in direct_live_keys
    }
    if not direct_live_enabled:
        add_blocker("hi_nerv_short_smoke_direct_live_segnet_distillation_disabled")
    else:
        missing_direct_live = [
            key for key, value in direct_live_metrics.items() if value is None
        ]
        if missing_direct_live:
            add_blocker("hi_nerv_short_smoke_missing_direct_live_segnet_telemetry")
        missing_subcontrol_metrics = sorted(
            key
            for key in active_direct_live_subcontrol_metric_keys
            if direct_live_metrics.get(key) is None
        )
        if missing_subcontrol_metrics:
            add_blocker(
                "hi_nerv_short_smoke_missing_direct_live_segnet_subcontrol_telemetry"
            )
        candidate_occupied = direct_live_metrics[
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction"
        ]
        if candidate_occupied is not None and _below_floor(
            candidate_occupied,
            min_occupied,
        ):
            add_blocker("hi_nerv_short_smoke_direct_live_class_occupancy_collapsed")
        candidate_target_coverage = direct_live_metrics[
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction"
        ]
        if (
            candidate_target_coverage is not None
            and _below_floor(candidate_target_coverage, min_target_coverage)
        ):
            add_blocker(
                "hi_nerv_short_smoke_direct_live_target_class_coverage_collapsed"
            )
        candidate_target_min_ratio = direct_live_metrics[
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
        ]
        if candidate_target_min_ratio is None:
            add_blocker("hi_nerv_short_smoke_direct_live_target_class_min_ratio_missing")
        elif _below_floor(candidate_target_min_ratio, min_target_min_ratio):
            add_blocker("hi_nerv_short_smoke_direct_live_target_class_mass_collapsed")
        if target_min_ratio_floor_control_active:
            if not target_region_debt_dynamics_gate["current_debt_present"]:
                add_blocker(
                    "hi_nerv_short_smoke_segnet_target_region_debt_missing"
                )
            elif target_region_debt_dynamics_gate["unresolved_debt_present"]:
                if not target_region_debt_dynamics_gate["pre_debt_present"]:
                    add_blocker(
                        "hi_nerv_short_smoke_segnet_target_region_debt_dynamics_missing"
                    )
                elif not target_region_debt_dynamics_gate[
                    "accepted_update_reduced_debt"
                ]:
                    add_blocker(
                        "hi_nerv_short_smoke_segnet_target_region_debt_not_reduced"
                    )

    pose_direct_live_keys = (
        "loss_part_pose_direct_live_score_term",
        "loss_part_pose_direct_live_raw_mse",
        "loss_part_pose_direct_live_score_marginal_wrt_raw_mse",
        "loss_part_pose_direct_live_yuv6_pair_std",
        "loss_part_pose_direct_live_input_official_yuv6_concat",
        "loss_part_pose_direct_live_input_frame0_incidence",
        "loss_part_pose_direct_live_input_frame1_incidence",
        "loss_part_pose_direct_live_input_channel_count",
        "loss_part_pose_direct_live_input_temporal_delta_proxy_authority",
        "loss_part_pose_direct_live_yuv6_pair_temporal_delta_std",
    )
    pose_direct_live_metrics = {
        key: _finite_mapping_value(final_components, key)
        for key in pose_direct_live_keys
    }
    if pose_direct_live_enabled:
        missing_pose_direct_live = [
            key for key, value in pose_direct_live_metrics.items() if value is None
        ]
        if missing_pose_direct_live:
            add_blocker("hi_nerv_short_smoke_missing_direct_live_posenet_telemetry")
    elif bool(require_pose_direct_live_distillation):
        add_blocker("hi_nerv_short_smoke_direct_live_posenet_distillation_required")

    dual_required_constraint_keys: dict[str, str] = {}
    if direct_live_enabled and direct_live_weight > 0.0:
        dual_required_constraint_keys["hi_nerv_segnet_direct_live_distill"] = (
            "loss_part_segnet_direct_live_distill"
        )
    if direct_live_enabled:
        for control_key, metric_key in direct_live_subcontrol_metric_keys.items():
            if control_key in active_direct_live_subcontrol_control_keys:
                dual_required_constraint_keys[
                    _SEGNET_DIRECT_LIVE_SUBCONTROL_DUAL_KEYS[control_key]
                ] = metric_key
        if direct_live_weight > 0.0:
            dual_required_constraint_keys[
                "hi_nerv_segnet_direct_live_argmax_disagreement"
            ] = "loss_part_segnet_direct_live_argmax_disagreement"
        if direct_live_subcontrol_weights[
            "segnet_direct_live_class_histogram_weight"
        ] > 0.0 and "segnet_direct_live_class_histogram_weight" in (
            active_direct_live_subcontrol_control_keys
        ):
            dual_required_constraint_keys[
                "hi_nerv_segnet_direct_live_target_missing_fraction_histogram"
            ] = (
                "loss_part_segnet_direct_live_candidate_target_class_missing_fraction"
            )
        if direct_live_subcontrol_weights[
            "segnet_direct_live_class_balanced_ce_weight"
        ] > 0.0 and "segnet_direct_live_class_balanced_ce_weight" in (
            active_direct_live_subcontrol_control_keys
        ):
            dual_required_constraint_keys[
                "hi_nerv_segnet_direct_live_target_missing_fraction_ce"
            ] = (
                "loss_part_segnet_direct_live_candidate_target_class_missing_fraction"
            )
        if direct_live_subcontrol_weights[
            "segnet_direct_live_class_region_recon_weight"
        ] > 0.0 and "segnet_direct_live_class_region_recon_weight" in (
            active_direct_live_subcontrol_control_keys
        ):
            dual_required_constraint_keys[
                "hi_nerv_segnet_direct_live_target_min_ratio_region_recon"
            ] = "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
        if direct_live_subcontrol_weights[
            "segnet_direct_live_rare_class_logit_weight"
        ] > 0.0 and "segnet_direct_live_rare_class_logit_weight" in (
            active_direct_live_subcontrol_control_keys
        ):
            dual_required_constraint_keys[
                "hi_nerv_segnet_direct_live_target_min_ratio_rare_class_logit"
            ] = "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
        if direct_live_subcontrol_weights[
            "segnet_direct_live_target_mass_floor_weight"
        ] > 0.0 and "segnet_direct_live_target_mass_floor_weight" in (
            active_direct_live_subcontrol_control_keys
        ):
            dual_required_constraint_keys[
                "hi_nerv_segnet_direct_live_target_min_ratio_mass_floor"
            ] = "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
        if direct_live_subcontrol_weights[
            "segnet_direct_live_target_min_ratio_floor_weight"
        ] > 0.0 and "segnet_direct_live_target_min_ratio_floor_weight" in (
            active_direct_live_subcontrol_control_keys
        ):
            dual_required_constraint_keys[
                "hi_nerv_segnet_direct_live_target_min_ratio_floor_gate"
            ] = "loss_part_segnet_direct_live_candidate_target_class_min_ratio"
    if pose_direct_live_enabled:
        dual_required_constraint_keys["hi_nerv_posenet_yuv6_pair_distill"] = (
            "loss_part_pose_direct_live_score_term"
        )
    dual_ascent_gate = _direct_live_dual_ascent_gate(
        final_components,
        required_constraint_keys=dual_required_constraint_keys,
    )
    if dual_required_constraint_keys:
        if not dual_ascent_gate["active"]:
            add_blocker(
                "hi_nerv_short_smoke_dual_ascent_inactive_for_direct_live_controls"
            )
        if dual_ascent_gate["missing_constraint_telemetry"]:
            add_blocker("hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry")
        if dual_ascent_gate["constraints_missing_observed_metric"]:
            add_blocker(
                "hi_nerv_short_smoke_direct_live_dual_ascent_missing_observed_metric"
            )
        if dual_ascent_gate["constraints_without_updates"]:
            add_blocker("hi_nerv_short_smoke_direct_live_dual_ascent_never_updated")
        if dual_ascent_gate["constraints_without_applied_weight"]:
            add_blocker(
                "hi_nerv_short_smoke_direct_live_dual_ascent_weight_not_applied"
            )
        if dual_ascent_gate["constraints_with_positive_violation_and_zero_lambda"]:
            add_blocker(
                "hi_nerv_short_smoke_direct_live_dual_ascent_lambda_not_activated"
            )

    section_byte_dual_ascent_gate = _section_byte_dual_ascent_gate(
        final_components,
        require_section_byte_dual_ascent=bool(require_section_byte_dual_ascent),
    )
    if section_byte_dual_ascent_gate["required"]:
        if not section_byte_dual_ascent_gate["section_or_archive_metric_present"]:
            add_blocker("hi_nerv_short_smoke_missing_train_time_section_byte_metrics")
        if not section_byte_dual_ascent_gate["active"]:
            add_blocker("hi_nerv_short_smoke_section_byte_dual_ascent_inactive")
        if section_byte_dual_ascent_gate["missing_constraint_telemetry"]:
            add_blocker("hi_nerv_short_smoke_missing_section_byte_dual_ascent_telemetry")
        if section_byte_dual_ascent_gate["constraints_missing_observed_metric"]:
            add_blocker(
                "hi_nerv_short_smoke_section_byte_dual_ascent_missing_observed_metric"
            )
        if section_byte_dual_ascent_gate["constraints_without_updates"]:
            add_blocker("hi_nerv_short_smoke_section_byte_dual_ascent_never_updated")
        if section_byte_dual_ascent_gate["constraints_without_applied_weight"]:
            add_blocker("hi_nerv_short_smoke_section_byte_dual_ascent_weight_not_applied")
        if section_byte_dual_ascent_gate[
            "constraints_with_positive_violation_and_zero_lambda"
        ]:
            add_blocker(
                "hi_nerv_short_smoke_section_byte_dual_ascent_lambda_not_activated"
            )

    decoder_waterfill_gate = _decoder_weight_waterfill_actuation_gate(
        final_components,
        metadata=decoder_weight_waterfill_plan_metadata,
    )
    if decoder_waterfill_gate["required"]:
        if not decoder_waterfill_gate["train_time_fake_quant_bound"]:
            add_blocker(
                "hi_nerv_short_smoke_decoder_waterfill_fake_quant_not_bound"
            )
        if not decoder_waterfill_gate["gradient_multiplier_metrics_present"]:
            add_blocker(
                "hi_nerv_short_smoke_decoder_waterfill_gradient_multiplier_not_observed"
            )
        if decoder_waterfill_gate["requested_but_unapplied"]:
            add_blocker(
                "hi_nerv_short_smoke_decoder_waterfill_gradient_multiplier_unapplied"
            )
        if decoder_waterfill_gate["missing_exact_name_count_positive"]:
            add_blocker(
                "hi_nerv_short_smoke_decoder_waterfill_gradient_multiplier_stale_name"
            )

    output_head_target_init_gate = _output_head_target_init_gate(
        output_head_target_bias_init_metadata
    )
    scorer_domain_hard_birth_gate = _scorer_domain_hard_birth_gate(
        (
            output_head_target_bias_init_metadata.get("scorer_domain_bootstrap")
            if isinstance(output_head_target_bias_init_metadata, Mapping)
            else None
        ),
        min_target_min_ratio=min_target_min_ratio,
    )
    if output_head_target_init_gate["required"]:
        if not output_head_target_init_gate["bias_init_enabled"]:
            add_blocker(
                "hi_nerv_short_smoke_output_head_target_bias_init_not_enabled"
            )
        if not output_head_target_init_gate["contrast_init_enabled"]:
            add_blocker(
                "hi_nerv_short_smoke_output_head_target_contrast_init_not_enabled"
            )
    if scorer_domain_hard_birth_gate["required"]:
        if not scorer_domain_hard_birth_gate["bootstrap_enabled"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_bootstrap_not_enabled"
            )
        if scorer_domain_hard_birth_gate["hard_birth_requested_but_not_consumed"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_hard_birth_requested_but_not_consumed"
            )
        if not scorer_domain_hard_birth_gate["hard_birth_enabled"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_hard_birth_not_enabled"
            )
        if not scorer_domain_hard_birth_gate["after_min_ratio_present"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_hard_birth_min_ratio_missing"
            )
        elif not scorer_domain_hard_birth_gate["after_min_ratio_cleared"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_hard_birth_min_ratio_collapsed"
            )
        if scorer_domain_hard_birth_gate["soft_progress_only_no_argmax_debt_move"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_hard_birth_soft_progress_only_no_argmax_debt_move"
            )
        if scorer_domain_hard_birth_gate["accepted_steps_without_argmax_debt_move"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_hard_birth_accepted_steps_without_argmax_debt_move"
            )
        if scorer_domain_hard_birth_gate["receiver_quantum_rejections_without_crossing"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_hard_birth_receiver_subquantum_updates"
            )
        if scorer_domain_hard_birth_gate["accepted_steps_without_receiver_uint8_change"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_hard_birth_accepted_steps_without_receiver_uint8_change"
            )
        if scorer_domain_hard_birth_gate["no_accepted_steps_with_remaining_debt"]:
            add_blocker(
                "hi_nerv_short_smoke_scorer_domain_hard_birth_no_accepted_steps_with_debt"
            )

    contrast_floor_weight = _control_float(
        train_time_controls,
        "scorer_input_contrast_floor_weight",
    )
    contrast_floor_enabled = contrast_floor_weight > 0.0
    contrast_floor_keys = (
        "loss_part_scorer_input_contrast_floor",
        "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio",
        "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio",
    )
    contrast_floor_metrics = {
        key: _finite_mapping_value(final_components, key) for key in contrast_floor_keys
    }
    segnet_ratio = contrast_floor_metrics[
        "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio"
    ]
    posenet_yuv6_ratio = contrast_floor_metrics[
        "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio"
    ]
    if not contrast_floor_enabled:
        add_blocker("hi_nerv_short_smoke_scorer_input_contrast_floor_disabled")
    else:
        missing_contrast = [
            key for key, value in contrast_floor_metrics.items() if value is None
        ]
        if missing_contrast:
            add_blocker("hi_nerv_short_smoke_missing_scorer_input_contrast_floor_telemetry")
        if (
            segnet_ratio is not None
            and segnet_ratio
            < _control_float(
                train_time_controls,
                "scorer_input_contrast_floor_segnet_min_std_ratio",
            )
        ):
            add_blocker("hi_nerv_short_smoke_segnet_contrast_floor_ratio_below_threshold")
        if (
            posenet_yuv6_ratio is not None
            and posenet_yuv6_ratio
            < _control_float(
                train_time_controls,
                "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio",
            )
        ):
            add_blocker("hi_nerv_short_smoke_posenet_yuv6_contrast_floor_ratio_below_threshold")

    shape_tether_weight = _control_float(
        train_time_controls,
        "scorer_input_shape_tether_weight",
    )
    shape_tether_enabled = shape_tether_weight > 0.0
    shape_tether_metric_aliases = {
        "loss": (
            "loss_part_scorer_input_shape_tether",
            "loss_part_pr95_stage_scorer_input_shape_tether",
        ),
        "segnet_last_rgb": (
            "loss_part_scorer_input_shape_tether_segnet_last_rgb",
            "loss_part_pr95_stage_scorer_input_shape_tether_segnet_last_rgb",
        ),
        "posenet_yuv6_pair": (
            "loss_part_scorer_input_shape_tether_posenet_yuv6_pair",
            "loss_part_pr95_stage_scorer_input_shape_tether_posenet_yuv6_pair",
        ),
        "posenet_yuv6_temporal_delta": (
            "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta",
            "loss_part_pr95_stage_scorer_input_shape_tether_posenet_yuv6_temporal_delta",
        ),
    }
    shape_tether_metrics = {
        name: _first_finite_mapping_value(final_components, aliases)
        for name, aliases in shape_tether_metric_aliases.items()
    }
    if not shape_tether_enabled:
        add_blocker("hi_nerv_short_smoke_scorer_input_shape_tether_disabled")
    elif any(value is None for value in shape_tether_metrics.values()):
        add_blocker("hi_nerv_short_smoke_missing_scorer_input_shape_tether_telemetry")

    temporal_floor_weight = _control_float(
        train_time_controls,
        "posenet_temporal_signal_floor_weight",
    )
    temporal_floor_enabled = temporal_floor_weight > 0.0
    temporal_floor_metric_aliases = {
        "loss": (
            "loss_part_posenet_temporal_signal_floor",
            "loss_part_pr95_stage_posenet_temporal_signal_floor",
        ),
        "mean_std_ratio": (
            "loss_part_posenet_temporal_signal_floor_mean_std_ratio",
            "loss_part_pr95_stage_posenet_temporal_signal_floor_mean_std_ratio",
        ),
        "mean_abs_ratio": (
            "loss_part_posenet_temporal_signal_floor_mean_abs_ratio",
            "loss_part_pr95_stage_posenet_temporal_signal_floor_mean_abs_ratio",
        ),
    }
    temporal_floor_metrics = {
        name: _first_finite_mapping_value(final_components, aliases)
        for name, aliases in temporal_floor_metric_aliases.items()
    }
    if not temporal_floor_enabled:
        add_blocker("hi_nerv_short_smoke_posenet_temporal_signal_floor_disabled")
    elif any(value is None for value in temporal_floor_metrics.values()):
        add_blocker(
            "hi_nerv_short_smoke_missing_posenet_temporal_signal_floor_telemetry"
        )

    receiver_cache_summary = receiver_cache_quality_manifest_summary(post_export_quality)
    receiver_surface_identity_gate = _receiver_surface_identity_gate(
        post_export_quality
    )
    source_qualified_metrics = _source_qualified_metrics_receipt(
        allow_mock_scorer_teacher=allow_mock_scorer_teacher,
        unscored_research_smoke_enabled=unscored_research_smoke_enabled,
        generic_segnet_requested=generic_segnet_requested,
        direct_live_enabled=direct_live_enabled,
        generic_posenet_requested=generic_posenet_requested,
        pose_direct_live_enabled=pose_direct_live_enabled,
        receiver_surface_identity_gate=receiver_surface_identity_gate,
        direct_live_metrics=direct_live_metrics,
        pose_direct_live_metrics=pose_direct_live_metrics,
        pose_distill_metrics=pose_distill_metrics,
    )
    segnet_argmax_probe = (
        post_export_quality.get("segnet_argmax_probe")
        if isinstance(post_export_quality, Mapping)
        else None
    )
    mlx_scorer_response_probe = (
        post_export_quality.get("mlx_scorer_response_probe")
        if isinstance(post_export_quality, Mapping)
        else None
    )
    scorer_input_distribution_gate = (
        post_export_quality.get("scorer_input_distribution_gate")
        if isinstance(post_export_quality, Mapping)
        else None
    )
    if post_export_quality is None:
        add_blocker("hi_nerv_short_smoke_receiver_cache_quality_gate_not_run")
    else:
        if not receiver_surface_identity_gate["archive_identity_present"]:
            add_blocker(
                "hi_nerv_short_smoke_receiver_surface_archive_identity_missing"
            )
        if not receiver_surface_identity_gate["direct_receiver_parseback_present"]:
            add_blocker(
                "hi_nerv_short_smoke_receiver_surface_parseback_missing"
            )
        if receiver_surface_identity_gate["archive_sha256_mismatch"]:
            add_blocker(
                "hi_nerv_short_smoke_receiver_surface_archive_sha256_mismatch"
            )
        if not receiver_surface_identity_gate["candidate_cache_manifest_bound"]:
            add_blocker(
                "hi_nerv_short_smoke_receiver_surface_cache_manifest_missing"
            )
        if not bool(post_export_quality.get("quality_gate_passed")):
            add_blocker("hi_nerv_short_smoke_receiver_cache_quality_failed")
        if not isinstance(scorer_input_distribution_gate, Mapping):
            add_blocker(
                "hi_nerv_short_smoke_receiver_cache_scorer_input_distribution_gate_missing"
            )
        else:
            distribution_blockers = {
                str(blocker)
                for blocker in scorer_input_distribution_gate.get("blockers") or []
            }
            if not bool(scorer_input_distribution_gate.get("fit_gate_passed")):
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_scorer_input_distribution_gate_failed"
                )
            if {
                "candidate_segnet_last_rgb_distribution_std_too_low",
                "candidate_segnet_last_rgb_distribution_dynamic_range_too_low",
            } & distribution_blockers:
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_segnet_rgb_distribution_degenerate"
                )
            if {
                "candidate_posenet_yuv6_pair_distribution_std_too_low",
                "candidate_posenet_yuv6_pair_distribution_dynamic_range_too_low",
            } & distribution_blockers:
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_posenet_yuv6_distribution_degenerate"
                )
            if {
                "candidate_posenet_yuv6_temporal_signal_std_too_low",
                "candidate_posenet_yuv6_temporal_signal_mean_abs_too_low",
            } & distribution_blockers:
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_posenet_yuv6_temporal_signal_degenerate"
                )
        if not isinstance(segnet_argmax_probe, Mapping):
            add_blocker("hi_nerv_short_smoke_receiver_cache_segnet_argmax_probe_missing")
        else:
            if not bool(segnet_argmax_probe.get("fit_gate_passed")):
                add_blocker("hi_nerv_short_smoke_receiver_cache_segnet_argmax_probe_failed")
            receiver_candidate_occupied = _finite_float(
                segnet_argmax_probe.get("candidate_occupied_class_fraction")
            )
            receiver_target_coverage = _finite_float(
                segnet_argmax_probe.get("candidate_target_class_coverage_fraction")
            )
            receiver_target_min_ratio = _finite_float(
                segnet_argmax_probe.get("candidate_target_class_min_ratio")
            )
            if receiver_candidate_occupied is None:
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_segnet_argmax_occupancy_missing"
                )
            elif _below_floor(receiver_candidate_occupied, min_occupied):
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_segnet_argmax_class_occupancy_collapsed"
                )
            if receiver_target_coverage is None:
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_segnet_target_class_coverage_missing"
                )
            elif _below_floor(receiver_target_coverage, min_target_coverage):
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_segnet_target_class_coverage_collapsed"
                )
            if receiver_target_min_ratio is None:
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_segnet_target_class_min_ratio_missing"
                )
            elif _below_floor(receiver_target_min_ratio, min_target_min_ratio):
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_segnet_target_class_mass_collapsed"
                )
        if bool(post_export_quality.get("mlx_scorer_response_probe_required")):
            if not isinstance(mlx_scorer_response_probe, Mapping):
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_mlx_scorer_response_probe_missing"
                )
            elif not bool(mlx_scorer_response_probe.get("fit_gate_passed")):
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_mlx_scorer_response_probe_failed"
                )

    score_dynamics_diagnosis = _score_dynamics_diagnosis(
        final_components,
        receiver_cache_summary=receiver_cache_summary,
        min_target_min_ratio=min_target_min_ratio,
        section_byte_dual_ascent_gate=section_byte_dual_ascent_gate,
    )
    ready = not actionable_blockers
    return {
        "schema": HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA,
        "authority": HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY,
        "axis_tag": HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
        "scope": "local_mlx_false_authority_short_scorer_teacher_smoke_gate",
        "short_scorer_teacher_smoke_ready": ready,
        "ready_for_long_run": ready,
        "teacher_gate": {
            "real_segnet_teacher_requested": generic_segnet_requested,
            "direct_live_segnet_requested": bool(direct_live_enabled),
            "direct_live_segnet_subcontrol_requested": bool(
                direct_live_subcontrol_enabled
            ),
            "direct_live_segnet_only": bool(segnet_direct_live_only),
            "real_posenet_teacher_requested": generic_posenet_requested,
            "direct_live_posenet_requested": bool(pose_direct_live_enabled),
            "direct_live_posenet_only": bool(posenet_direct_live_only),
            "mock_scorer_teacher_allowed": bool(allow_mock_scorer_teacher),
            "unscored_research_smoke_enabled": bool(unscored_research_smoke_enabled),
            "segnet_distillation_weight": segnet_weight,
            "pose_distillation_weight": pose_weight,
            "pose_direct_live_distillation_weight": pose_direct_live_weight,
        },
        "direct_live_segnet_gate": {
            "enabled": direct_live_enabled,
            "weight": direct_live_weight,
            "base_distillation_enabled": bool(direct_live_weight > 0.0),
            "subcontrol_enabled": bool(direct_live_subcontrol_enabled),
            "subcontrol_weights": direct_live_subcontrol_weights,
            "subcontrol_stage_active_weights": (
                direct_live_subcontrol_stage_active_weights
            ),
            "active_subcontrol_control_keys": sorted(
                active_direct_live_subcontrol_control_keys
            ),
            "active_subcontrol_metric_keys": sorted(
                active_direct_live_subcontrol_metric_keys
            ),
            "min_candidate_occupied_class_fraction_for_fit_gate": min_occupied,
            "min_candidate_target_class_coverage_fraction_for_fit_gate": (
                min_target_coverage
            ),
            "min_candidate_target_class_min_ratio_for_fit_gate": (
                min_target_min_ratio
            ),
            "metrics": direct_live_metrics,
            "target_region_debt_dynamics_gate": target_region_debt_dynamics_gate,
        },
        "direct_live_posenet_gate": {
            "enabled": pose_direct_live_enabled,
            "required": bool(require_pose_direct_live_distillation),
            "weight": pose_direct_live_weight,
            "metrics": pose_direct_live_metrics,
        },
        "posenet_distill_gate": {
            "enabled": generic_posenet_requested,
            "weight": pose_weight,
            "metrics": pose_distill_metrics,
        },
        "direct_live_dual_ascent_gate": dual_ascent_gate,
        "section_byte_dual_ascent_gate": section_byte_dual_ascent_gate,
        "decoder_weight_waterfill_actuation_gate": decoder_waterfill_gate,
        "output_head_target_init_gate": output_head_target_init_gate,
        "scorer_domain_hard_birth_bootstrap_gate": scorer_domain_hard_birth_gate,
        "scorer_input_contrast_floor_gate": {
            "enabled": contrast_floor_enabled,
            "weight": contrast_floor_weight,
            "segnet_last_rgb_min_std_ratio": _control_float(
                train_time_controls,
                "scorer_input_contrast_floor_segnet_min_std_ratio",
            ),
            "posenet_yuv6_pair_min_std_ratio": _control_float(
                train_time_controls,
                "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio",
            ),
            "metrics": contrast_floor_metrics,
        },
        "scorer_input_shape_tether_gate": {
            "enabled": shape_tether_enabled,
            "weight": shape_tether_weight,
            "metrics": shape_tether_metrics,
        },
        "posenet_temporal_signal_floor_gate": {
            "enabled": temporal_floor_enabled,
            "weight": temporal_floor_weight,
            "metrics": temporal_floor_metrics,
        },
        "score_dynamics_diagnosis": score_dynamics_diagnosis,
        "receiver_cache_quality": receiver_cache_summary,
        "receiver_surface_identity_gate": receiver_surface_identity_gate,
        "source_qualified_metrics": source_qualified_metrics,
        "final_loss_components_present": bool(final_components),
        "actionable_blockers": actionable_blockers,
        "blockers": [
            "hi_nerv_short_scorer_smoke_is_false_authority",
            *actionable_blockers,
        ],
        **FALSE_AUTHORITY,
    }


def _source_qualified_metrics_receipt(
    *,
    allow_mock_scorer_teacher: bool,
    unscored_research_smoke_enabled: bool,
    generic_segnet_requested: bool,
    direct_live_enabled: bool,
    generic_posenet_requested: bool,
    pose_direct_live_enabled: bool,
    receiver_surface_identity_gate: Mapping[str, Any],
    direct_live_metrics: Mapping[str, Any],
    pose_direct_live_metrics: Mapping[str, Any],
    pose_distill_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if bool(allow_mock_scorer_teacher):
        blockers.append("source_metrics_mock_scorer_teacher_enabled")
    if bool(unscored_research_smoke_enabled):
        blockers.append("source_metrics_unscored_research_smoke_enabled")
    seg_requested = bool(generic_segnet_requested or direct_live_enabled)
    pose_requested = bool(generic_posenet_requested or pose_direct_live_enabled)
    if not seg_requested:
        blockers.append("source_metrics_segnet_teacher_not_requested")
    if not pose_requested:
        blockers.append("source_metrics_posenet_teacher_not_requested")
    if not bool(receiver_surface_identity_gate.get("archive_identity_present")):
        blockers.append("source_metrics_archive_identity_missing")
    if not bool(receiver_surface_identity_gate.get("direct_receiver_parseback_present")):
        blockers.append("source_metrics_direct_receiver_parseback_missing")
    if bool(receiver_surface_identity_gate.get("archive_sha256_mismatch")):
        blockers.append("source_metrics_archive_sha256_mismatch")
    if not bool(receiver_surface_identity_gate.get("candidate_cache_manifest_bound")):
        blockers.append("source_metrics_candidate_cache_manifest_missing")
    if direct_live_enabled and not any(
        _finite_float(value) is not None for value in direct_live_metrics.values()
    ):
        blockers.append("source_metrics_segnet_direct_live_metrics_missing")
    pose_metric_values = (
        *pose_direct_live_metrics.values(),
        *pose_distill_metrics.values(),
    )
    if pose_requested and not any(
        _finite_float(value) is not None for value in pose_metric_values
    ):
        blockers.append("source_metrics_posenet_metrics_missing")
    source_qualified = not blockers
    return {
        "schema": NERV_SOURCE_QUALIFIED_METRICS_SCHEMA,
        "family": "hinerv",
        "source_qualified": bool(source_qualified),
        "metric_source": (
            "upstream_evaluate_geometry"
            if source_qualified
            else "blocked_source_qualification"
        ),
        "canonical_score_source": (
            "S=100*d_seg+sqrt(10*d_pose)+25*archive_bytes/37545489"
            if source_qualified
            else None
        ),
        "seg_metric_source": (
            "SegNet last-frame RGB argmax disagreement"
            if source_qualified
            else None
        ),
        "pose_metric_source": (
            "PoseNet two-frame YUV6 raw-MSE score term"
            if source_qualified
            else None
        ),
        "archive_metric_source": (
            "archive.zip charged bytes" if source_qualified else None
        ),
        "receiver_parseback_bound": bool(
            receiver_surface_identity_gate.get("direct_receiver_parseback_present")
        ),
        "archive_identity_present": bool(
            receiver_surface_identity_gate.get("archive_identity_present")
        ),
        "candidate_cache_manifest_bound": bool(
            receiver_surface_identity_gate.get("candidate_cache_manifest_bound")
        ),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def hinerv_short_scorer_smoke_readiness_summary(
    report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    return _metadata_safe(
        {
            "schema": report.get("schema"),
            "report_path": report.get("report_path"),
            "authority": report.get("authority"),
            "axis_tag": report.get("axis_tag"),
            "scope": report.get("scope"),
            "short_scorer_teacher_smoke_ready": bool(
                report.get("short_scorer_teacher_smoke_ready")
            ),
            "ready_for_long_run": bool(report.get("ready_for_long_run")),
            "teacher_gate": report.get("teacher_gate"),
            "direct_live_segnet_gate": report.get("direct_live_segnet_gate"),
            "direct_live_posenet_gate": report.get("direct_live_posenet_gate"),
            "posenet_distill_gate": report.get("posenet_distill_gate"),
            "direct_live_dual_ascent_gate": report.get(
                "direct_live_dual_ascent_gate"
            ),
            "section_byte_dual_ascent_gate": report.get(
                "section_byte_dual_ascent_gate"
            ),
            "decoder_weight_waterfill_actuation_gate": report.get(
                "decoder_weight_waterfill_actuation_gate"
            ),
            "output_head_target_init_gate": report.get(
                "output_head_target_init_gate"
            ),
            "scorer_domain_hard_birth_bootstrap_gate": report.get(
                "scorer_domain_hard_birth_bootstrap_gate"
            ),
            "scorer_input_contrast_floor_gate": report.get(
                "scorer_input_contrast_floor_gate"
            ),
            "scorer_input_shape_tether_gate": report.get(
                "scorer_input_shape_tether_gate"
            ),
            "posenet_temporal_signal_floor_gate": report.get(
                "posenet_temporal_signal_floor_gate"
            ),
            "score_dynamics_diagnosis": report.get("score_dynamics_diagnosis"),
            "receiver_cache_quality": report.get("receiver_cache_quality"),
            "actionable_blockers": [
                str(blocker) for blocker in report.get("actionable_blockers") or []
            ],
            "blockers": [str(blocker) for blocker in report.get("blockers") or []],
        }
    )


def hinerv_short_scorer_smoke_long_run_admission(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the compact trainer/runner long-run admission payload.

    The full readiness report is useful for debugging, but launch automation
    needs a small, stable actuator verdict: whether the short scorer-complete
    smoke earned a longer MLX run, plus the exact blocker atoms to clear.
    """

    long_run_admission_passed = bool(
        report is not None and report.get("ready_for_long_run")
    )
    short_scorer_teacher_smoke_passed = bool(
        report is not None and report.get("short_scorer_teacher_smoke_ready")
    )
    admission_blockers: list[str] = []
    if not long_run_admission_passed:
        admission_blockers = list(
            dict.fromkeys(
                [
                    "hi_nerv_short_scorer_smoke_not_ready_for_long_run",
                    *[
                        str(blocker)
                        for blocker in (
                            report.get("actionable_blockers")
                            if isinstance(report, Mapping)
                            else []
                        )
                        or []
                    ],
                ]
            )
        )
    return _metadata_safe(
        {
            "schema": "hi_nerv_short_scorer_smoke_long_run_admission.v1",
            "long_run_admission_passed": long_run_admission_passed,
            "short_scorer_teacher_smoke_passed": short_scorer_teacher_smoke_passed,
            "report_path": (
                str(report.get("report_path") or "")
                if isinstance(report, Mapping)
                else ""
            ),
            "admission_blockers": admission_blockers,
            "admission_blocker_count": len(admission_blockers),
            "authority": HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY,
            "axis_tag": HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
        }
    )


def _score_dynamics_diagnosis(
    metrics: Mapping[str, Any],
    *,
    receiver_cache_summary: Mapping[str, Any] | None,
    min_target_min_ratio: float,
    section_byte_dual_ascent_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the local score dynamics that matter to upstream evaluate.py.

    ``evaluate.py`` admits exactly three score axes: last-frame SegNet argmax
    pixels, two-frame PoseNet YUV6 error, and archive bytes.  This helper keeps
    that geometry visible to automation, so short smokes answer which atom is
    unsolved and whether accepted updates reduce it.
    """

    seg_atom_rows = _segnet_target_region_atom_rows(metrics)
    worst_seg_atom = seg_atom_rows[0] if seg_atom_rows else None
    seg_unsolved_score = _first_finite_mapping_value(
        metrics,
        (
            "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass",
            "loss_part_segnet_direct_live_argmax_disagreement",
        ),
    )
    if (
        seg_unsolved_score is not None
        and "loss_part_segnet_direct_live_argmax_disagreement" in metrics
        and "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass"
        not in metrics
    ):
        seg_unsolved_score *= HI_NERV_CONTEST_SEGNET_PIXEL_SCORE_WEIGHT
    pre_seg_unsolved_score = _first_finite_mapping_value(
        metrics,
        (
            "dynamics_pre_update_loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass",
            "dynamics_pre_update_loss_part_segnet_direct_live_argmax_disagreement",
        ),
    )
    if (
        pre_seg_unsolved_score is not None
        and "dynamics_pre_update_loss_part_segnet_direct_live_argmax_disagreement"
        in metrics
        and "dynamics_pre_update_loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass"
        not in metrics
    ):
        pre_seg_unsolved_score *= HI_NERV_CONTEST_SEGNET_PIXEL_SCORE_WEIGHT
    seg_delta = _delta(seg_unsolved_score, pre_seg_unsolved_score)
    target_min_ratio = _finite_mapping_value(
        metrics,
        "loss_part_segnet_direct_live_candidate_target_class_min_ratio",
    )
    target_coverage = _finite_mapping_value(
        metrics,
        "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction",
    )
    argmax_disagreement = _finite_mapping_value(
        metrics,
        "loss_part_segnet_direct_live_argmax_disagreement",
    )

    pose_direct = _finite_mapping_value(metrics, "loss_part_pose_direct_live_score_term")
    pose_proxy = _finite_mapping_value(metrics, "loss_part_pose_score_term")
    pose_score = pose_direct if pose_direct is not None else pose_proxy
    pre_pose_direct = _finite_mapping_value(
        metrics,
        "dynamics_pre_update_loss_part_pose_direct_live_score_term",
    )
    pre_pose_proxy = _finite_mapping_value(
        metrics,
        "dynamics_pre_update_loss_part_pose_score_term",
    )
    pre_pose_score = pre_pose_direct if pre_pose_direct is not None else pre_pose_proxy
    pose_delta = _delta(pose_score, pre_pose_score)

    archive_bytes = _finite_mapping_value(metrics, "train_time_archive_bytes")
    archive_rate_score = _finite_mapping_value(metrics, "train_time_archive_rate_score")
    if archive_rate_score is None and archive_bytes is not None:
        archive_rate_score = archive_bytes * HI_NERV_CONTEST_RATE_SCORE_PER_BYTE
    pre_archive_rate_score = _finite_mapping_value(
        metrics,
        "dynamics_pre_update_loss_part_train_time_archive_rate_score",
    )
    rate_delta = _delta(archive_rate_score, pre_archive_rate_score)

    local_score_proxy = _sum_finite(seg_unsolved_score, pose_score, archive_rate_score)
    pre_local_score_proxy = _sum_finite(
        pre_seg_unsolved_score,
        pre_pose_score,
        pre_archive_rate_score,
    )
    local_score_delta = _delta(local_score_proxy, pre_local_score_proxy)
    dominant_axis = _dominant_axis(
        {
            "segnet_target_region": seg_unsolved_score,
            "posenet_yuv6_pair": pose_score,
            "archive_rate": archive_rate_score,
        }
    )
    min_ratio_blocked = bool(
        target_min_ratio is not None
        and _below_floor(target_min_ratio, min_target_min_ratio)
    )
    active_byte_constraints = []
    if isinstance(section_byte_dual_ascent_gate, Mapping):
        active_byte_constraints = [
            str(key)
            for key in section_byte_dual_ascent_gate.get(
                "constraints_with_active_loss_pressure"
            )
            or []
        ]
    receiver_summary = receiver_cache_summary if isinstance(receiver_cache_summary, Mapping) else {}
    receiver_seg_score = _finite_float(receiver_summary.get("mlx_scorer_response_avg_segnet_dist"))
    receiver_pose_score = _finite_float(receiver_summary.get("mlx_scorer_response_avg_posenet_dist"))

    if min_ratio_blocked or (
        worst_seg_atom is not None
        and (worst_seg_atom.get("score_weighted_unsolved_argmax_mass") or 0.0) > 0.0
    ):
        regime = "segnet_target_region_decision_crossing_blocked"
        actuator = "target_region_margin_crossing_output_head_bias_and_region_waterfill"
    elif pose_score is not None and pose_score > 1.0:
        regime = "posenet_yuv6_pair_geometry_blocked"
        actuator = "pose_yuv6_geometry_temporal_tether_and_pair_curriculum"
    elif archive_rate_score is not None and active_byte_constraints:
        regime = "rate_dual_pressure_active"
        actuator = "section_byte_dual_qat_ablation_and_waterfill"
    else:
        regime = "no_single_dominant_score_axis_observed"
        actuator = "inspect_missing_metrics_or_optimizer_dynamics"

    return {
        "schema": "hi_nerv_evaluate_py_score_dynamics_diagnosis.v1",
        "authority": HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY,
        "axis_tag": HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
        "contest_formula": {
            "score": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/original_video_bytes",
            "original_video_bytes": HI_NERV_CONTEST_ORIGINAL_VIDEO_BYTES,
            "rate_score_per_byte": HI_NERV_CONTEST_RATE_SCORE_PER_BYTE,
            "segnet_pixel_score_weight": HI_NERV_CONTEST_SEGNET_PIXEL_SCORE_WEIGHT,
        },
        "dominant_axis": dominant_axis,
        "dynamics_regime": regime,
        "recommended_next_actuator": actuator,
        "segnet": {
            "argmax_disagreement": argmax_disagreement,
            "target_class_coverage_fraction": target_coverage,
            "target_class_min_ratio": target_min_ratio,
            "min_ratio_floor": min_target_min_ratio,
            "score_weighted_unsolved_argmax_mass": seg_unsolved_score,
            "pre_score_weighted_unsolved_argmax_mass": pre_seg_unsolved_score,
            "delta_score_weighted_unsolved_argmax_mass": seg_delta,
            "accepted_update_reduced_unsolved_mass": _negative_delta(seg_delta),
            "worst_target_region_atom": worst_seg_atom,
            "top_target_region_atoms": seg_atom_rows[:5],
            "receiver_avg_segnet_dist": receiver_seg_score,
        },
        "posenet": {
            "score_term": pose_score,
            "pre_score_term": pre_pose_score,
            "delta_score_term": pose_delta,
            "accepted_update_reduced_pose_term": _negative_delta(pose_delta),
            "direct_live_score_term": pose_direct,
            "cached_score_term": pose_proxy,
            "yuv6_pair_distribution_mae": _finite_mapping_value(
                metrics,
                "loss_part_scorer_input_distribution_guard_yuv6_pair_mae",
            ),
            "yuv6_pair_contrast_ratio": _finite_mapping_value(
                metrics,
                "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio",
            ),
            "temporal_signal_std_ratio": _first_finite_mapping_value(
                metrics,
                (
                    "loss_part_posenet_temporal_signal_floor_mean_std_ratio",
                    "loss_part_pr95_stage_posenet_temporal_signal_floor_mean_std_ratio",
                ),
            ),
            "receiver_avg_posenet_dist": receiver_pose_score,
        },
        "rate": {
            "archive_bytes": archive_bytes,
            "archive_rate_score": archive_rate_score,
            "pre_archive_rate_score": pre_archive_rate_score,
            "delta_archive_rate_score": rate_delta,
            "byte_dual_active": bool(
                section_byte_dual_ascent_gate.get("active")
                if isinstance(section_byte_dual_ascent_gate, Mapping)
                else False
            ),
            "active_loss_pressure_constraints": active_byte_constraints,
        },
        "joint": {
            "local_score_proxy": local_score_proxy,
            "pre_local_score_proxy": pre_local_score_proxy,
            "delta_local_score_proxy": local_score_delta,
            "accepted_update_reduced_local_score_proxy": _negative_delta(
                local_score_delta
            ),
            "seg_pose_tradeoff": _tradeoff_label(seg_delta, pose_delta),
        },
        **FALSE_AUTHORITY,
    }


def _segnet_target_region_atom_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    prefix = "loss_part_segnet_direct_live_target_min_ratio_floor_class_"
    suffix = "_score_weighted_unsolved_argmax_mass"
    rows: list[dict[str, Any]] = []
    for key in sorted(metrics):
        key_s = str(key)
        if not key_s.startswith(prefix) or not key_s.endswith(suffix):
            continue
        class_index = key_s[len(prefix) : -len(suffix)]
        if not class_index.isdigit():
            continue
        base = f"{prefix}{class_index}"
        score_mass = _finite_mapping_value(metrics, key_s)
        if score_mass is None:
            continue
        rows.append(
            {
                "class_index": int(class_index),
                "score_weighted_unsolved_argmax_mass": score_mass,
                "target_region_unsolved_argmax_mass": _finite_mapping_value(
                    metrics,
                    f"{base}_target_region_unsolved_argmax_mass",
                ),
                "score_weighted_crossing_loss": _finite_mapping_value(
                    metrics,
                    f"{base}_score_weighted_crossing_loss",
                ),
                "decision_crossing_score_debt_boost": _finite_mapping_value(
                    metrics,
                    f"{base}_decision_crossing_score_debt_boost",
                ),
                "target_fraction": _finite_mapping_value(
                    metrics,
                    f"{base}_target_fraction",
                ),
                "region_ratio": _finite_mapping_value(
                    metrics,
                    f"{base}_region_ratio",
                ),
                "region_deficit": _finite_mapping_value(
                    metrics,
                    f"{base}_region_deficit",
                ),
                "hard_ratio": _finite_mapping_value(metrics, f"{base}_hard_ratio"),
                "target_region_frontier_margin": _finite_mapping_value(
                    metrics,
                    f"{base}_target_region_frontier_margin",
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("score_weighted_unsolved_argmax_mass") or 0.0),
            float(row.get("score_weighted_crossing_loss") or 0.0),
        ),
        reverse=True,
    )


def _delta(value: float | None, pre_value: float | None) -> float | None:
    if value is None or pre_value is None:
        return None
    return value - pre_value


def _negative_delta(delta: float | None) -> bool | None:
    if delta is None:
        return None
    return delta < 0.0


def _sum_finite(*values: float | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _dominant_axis(values: Mapping[str, float | None]) -> str | None:
    finite = {
        str(key): float(value)
        for key, value in values.items()
        if value is not None and math.isfinite(float(value))
    }
    if not finite:
        return None
    return max(finite, key=lambda key: abs(finite[key]))


def _tradeoff_label(seg_delta: float | None, pose_delta: float | None) -> str:
    seg_improved = bool(seg_delta is not None and seg_delta < 0.0)
    seg_worsened = bool(seg_delta is not None and seg_delta > 0.0)
    pose_improved = bool(pose_delta is not None and pose_delta < 0.0)
    pose_worsened = bool(pose_delta is not None and pose_delta > 0.0)
    if seg_improved and pose_improved:
        return "segnet_and_posenet_cooperative"
    if seg_improved and pose_worsened:
        return "segnet_improved_posenet_worsened"
    if seg_worsened and pose_improved:
        return "posenet_improved_segnet_worsened"
    if seg_worsened and pose_worsened:
        return "segnet_and_posenet_both_worsened"
    if seg_delta is None or pose_delta is None:
        return "tradeoff_delta_missing"
    return "segnet_and_posenet_flat_or_mixed_small"


def _target_region_debt_dynamics_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    current = _finite_mapping_value(
        metrics,
        "loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass",
    )
    previous = _finite_mapping_value(
        metrics,
        "dynamics_pre_update_loss_part_segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass",
    )
    delta = _delta(current, previous)
    return {
        "schema": "hi_nerv_segnet_target_region_debt_dynamics_gate.v1",
        "current_score_weighted_unsolved_argmax_mass": current,
        "pre_score_weighted_unsolved_argmax_mass": previous,
        "delta_score_weighted_unsolved_argmax_mass": delta,
        "current_debt_present": current is not None,
        "pre_debt_present": previous is not None,
        "unresolved_debt_present": bool(
            current is not None
            and current > HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
        ),
        "accepted_update_reduced_debt": bool(
            delta is not None
            and delta < -HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
        ),
    }


def _scorer_domain_hard_birth_gate(
    metadata: Any,
    *,
    min_target_min_ratio: float,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {
            "schema": "hi_nerv_scorer_domain_hard_birth_bootstrap_gate.v1",
            "required": False,
            "bootstrap_enabled": False,
            "hard_birth_enabled": False,
            "after_min_ratio_present": False,
            "after_min_ratio_cleared": False,
            "no_accepted_steps_with_remaining_debt": False,
        }
    hard_birth = metadata.get("segnet_hard_birth_bootstrap")
    hard_birth = hard_birth if isinstance(hard_birth, Mapping) else {}
    before = metadata.get("metrics_before")
    before = before if isinstance(before, Mapping) else {}
    after = metadata.get("metrics_after")
    after = after if isinstance(after, Mapping) else {}
    after_min_ratio = _finite_mapping_value(
        after,
        "segnet_hard_birth_bootstrap_candidate_target_class_min_ratio",
    )
    before_min_ratio = _finite_mapping_value(
        before,
        "segnet_hard_birth_bootstrap_candidate_target_class_min_ratio",
    )
    min_ratio_delta = _finite_mapping_value(
        metadata,
        "segnet_hard_birth_bootstrap_candidate_target_class_min_ratio_delta",
    )
    if min_ratio_delta is None:
        min_ratio_delta = _delta(after_min_ratio, before_min_ratio)
    remaining_debt = _finite_mapping_value(
        after,
        "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass",
    )
    before_debt = _finite_mapping_value(
        before,
        "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass",
    )
    debt_delta = _finite_mapping_value(
        metadata,
        "segnet_hard_birth_bootstrap_score_weighted_total_unsolved_argmax_mass_delta",
    )
    if debt_delta is None:
        debt_delta = _delta(remaining_debt, before_debt)
    after_worst_debt = _finite_mapping_value(
        after,
        "segnet_hard_birth_bootstrap_score_weighted_worst_unsolved_argmax_mass",
    )
    before_worst_debt = _finite_mapping_value(
        before,
        "segnet_hard_birth_bootstrap_score_weighted_worst_unsolved_argmax_mass",
    )
    worst_debt_delta = _delta(after_worst_debt, before_worst_debt)
    loss_delta = _finite_mapping_value(
        metadata,
        "segnet_hard_birth_bootstrap_loss_delta",
    )
    accepted_step_count = _finite_mapping_value(metadata, "accepted_step_count")
    receiver_quantum_attempt_count = _finite_mapping_value(
        metadata,
        "receiver_quantum_attempt_count",
    )
    receiver_quantum_rejected_step_count = _finite_mapping_value(
        metadata,
        "receiver_quantum_rejected_step_count",
    )
    receiver_quantum_crossing_accepted_step_count = _finite_mapping_value(
        metadata,
        "receiver_quantum_crossing_accepted_step_count",
    )
    hard_birth_argmax_progress_accepted_step_count = _finite_mapping_value(
        metadata,
        "hard_birth_argmax_progress_accepted_step_count",
    )
    hard_birth_argmax_progress_rejected_step_count = _finite_mapping_value(
        metadata,
        "hard_birth_argmax_progress_rejected_step_count",
    )
    hard_birth_worst_improved_total_spill_rejected_step_count = (
        _finite_mapping_value(
            metadata,
            "hard_birth_worst_improved_total_spill_rejected_step_count",
        )
    )
    hard_birth_requested_weight = _finite_mapping_value(
        metadata,
        "segnet_hard_birth_bootstrap_requested_weight",
    )
    hard_birth_effective_weight = _finite_mapping_value(
        metadata,
        "segnet_hard_birth_bootstrap_effective_weight",
    )
    hard_birth_request_consumed_raw = metadata.get(
        "segnet_hard_birth_bootstrap_request_consumed"
    )
    hard_birth_request_consumed = (
        bool(hard_birth_request_consumed_raw)
        if hard_birth_request_consumed_raw is not None
        else not bool(
            hard_birth_requested_weight is not None
            and hard_birth_requested_weight
            > HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
            and (
                hard_birth_effective_weight is None
                or hard_birth_effective_weight
                <= HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
            )
        )
    )
    hard_birth_requested_but_not_consumed = bool(
        hard_birth_requested_weight is not None
        and hard_birth_requested_weight
        > HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
        and not hard_birth_request_consumed
    )
    max_candidate_segnet_worst_debt_reduction = _finite_mapping_value(
        metadata,
        "max_candidate_segnet_worst_debt_reduction",
    )
    max_candidate_segnet_total_debt_spill_given_worst_improvement = (
        _finite_mapping_value(
            metadata,
            "max_candidate_segnet_total_debt_spill_given_worst_improvement",
        )
    )
    max_accepted_segnet_worst_debt_reduction = _finite_mapping_value(
        metadata,
        "max_accepted_segnet_worst_debt_reduction",
    )
    max_candidate_frame1_delta_abs_uint8 = _finite_mapping_value(
        metadata,
        "max_candidate_frame1_delta_abs_uint8",
    )
    max_accepted_frame1_delta_abs_uint8 = _finite_mapping_value(
        metadata,
        "max_accepted_frame1_delta_abs_uint8",
    )
    max_candidate_frame1_receiver_uint8_changed_count = _finite_mapping_value(
        metadata,
        "max_candidate_frame1_receiver_uint8_changed_count",
    )
    max_accepted_frame1_receiver_uint8_changed_count = _finite_mapping_value(
        metadata,
        "max_accepted_frame1_receiver_uint8_changed_count",
    )
    max_candidate_frame1_receiver_uint8_changed_fraction = _finite_mapping_value(
        metadata,
        "max_candidate_frame1_receiver_uint8_changed_fraction",
    )
    max_accepted_frame1_receiver_uint8_changed_fraction = _finite_mapping_value(
        metadata,
        "max_accepted_frame1_receiver_uint8_changed_fraction",
    )
    debt_remains = bool(
        remaining_debt is not None
        and remaining_debt > HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
    )
    no_accepted_steps = bool(
        accepted_step_count is not None and accepted_step_count <= 0.0
    )
    argmax_ratio_moved = bool(
        min_ratio_delta is not None
        and min_ratio_delta > HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
    )
    score_debt_moved = bool(
        debt_delta is not None
        and debt_delta < -HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
    )
    worst_score_debt_moved = bool(
        worst_debt_delta is not None
        and worst_debt_delta < -HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
    )
    loss_moved = bool(
        loss_delta is not None
        and loss_delta > HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON
    )
    accepted_steps_present = bool(
        accepted_step_count is not None and accepted_step_count > 0.0
    )
    hard_argmax_birth_progress = bool(
        argmax_ratio_moved or score_debt_moved or worst_score_debt_moved
    )
    after_min_ratio_cleared = bool(
        after_min_ratio is not None
        and not _below_floor(after_min_ratio, min_target_min_ratio)
    )
    soft_progress_only = bool(
        loss_moved
        and not hard_argmax_birth_progress
        and not after_min_ratio_cleared
    )
    accepted_without_argmax_debt_move = bool(
        accepted_steps_present
        and debt_remains
        and not hard_argmax_birth_progress
    )
    receiver_quantum_rejections_without_crossing = bool(
        metadata.get("receiver_quantum_acceptance_enabled") is True
        and receiver_quantum_rejected_step_count is not None
        and receiver_quantum_rejected_step_count > 0.0
        and (
            receiver_quantum_crossing_accepted_step_count is None
            or receiver_quantum_crossing_accepted_step_count <= 0.0
        )
        and debt_remains
    )
    accepted_steps_without_receiver_uint8_change = bool(
        metadata.get("receiver_quantum_acceptance_enabled") is True
        and accepted_steps_present
        and (
            max_accepted_frame1_receiver_uint8_changed_count is not None
            and max_accepted_frame1_receiver_uint8_changed_count <= 0.0
        )
        and debt_remains
    )
    return {
        "schema": "hi_nerv_scorer_domain_hard_birth_bootstrap_gate.v1",
        "required": True,
        "bootstrap_enabled": bool(metadata.get("enabled")),
        "hard_birth_enabled": bool(hard_birth.get("enabled")),
        "hard_birth_requested_weight": hard_birth_requested_weight,
        "hard_birth_effective_weight": hard_birth_effective_weight,
        "hard_birth_request_consumed": hard_birth_request_consumed,
        "hard_birth_requested_but_not_consumed": (
            hard_birth_requested_but_not_consumed
        ),
        "accepted_step_count": accepted_step_count,
        "before_candidate_target_class_min_ratio": before_min_ratio,
        "after_candidate_target_class_min_ratio": after_min_ratio,
        "delta_candidate_target_class_min_ratio": min_ratio_delta,
        "min_candidate_target_class_min_ratio_for_fit_gate": min_target_min_ratio,
        "before_score_weighted_total_unsolved_argmax_mass": before_debt,
        "after_score_weighted_total_unsolved_argmax_mass": remaining_debt,
        "delta_score_weighted_total_unsolved_argmax_mass": debt_delta,
        "before_score_weighted_worst_unsolved_argmax_mass": before_worst_debt,
        "after_score_weighted_worst_unsolved_argmax_mass": after_worst_debt,
        "delta_score_weighted_worst_unsolved_argmax_mass": worst_debt_delta,
        "loss_delta": loss_delta,
        "receiver_quantum_acceptance_enabled": bool(
            metadata.get("receiver_quantum_acceptance_enabled")
        ),
        "receiver_quantum_attempt_count": receiver_quantum_attempt_count,
        "receiver_quantum_rejected_step_count": receiver_quantum_rejected_step_count,
        "receiver_quantum_crossing_accepted_step_count": (
            receiver_quantum_crossing_accepted_step_count
        ),
        "hard_birth_argmax_progress_accepted_step_count": (
            hard_birth_argmax_progress_accepted_step_count
        ),
        "hard_birth_argmax_progress_rejected_step_count": (
            hard_birth_argmax_progress_rejected_step_count
        ),
        "hard_birth_worst_improved_total_spill_rejected_step_count": (
            hard_birth_worst_improved_total_spill_rejected_step_count
        ),
        "max_candidate_segnet_worst_debt_reduction": (
            max_candidate_segnet_worst_debt_reduction
        ),
        "max_candidate_segnet_total_debt_spill_given_worst_improvement": (
            max_candidate_segnet_total_debt_spill_given_worst_improvement
        ),
        "max_accepted_segnet_worst_debt_reduction": (
            max_accepted_segnet_worst_debt_reduction
        ),
        "max_candidate_frame1_delta_abs_uint8": max_candidate_frame1_delta_abs_uint8,
        "max_accepted_frame1_delta_abs_uint8": max_accepted_frame1_delta_abs_uint8,
        "max_candidate_frame1_receiver_uint8_changed_count": (
            max_candidate_frame1_receiver_uint8_changed_count
        ),
        "max_accepted_frame1_receiver_uint8_changed_count": (
            max_accepted_frame1_receiver_uint8_changed_count
        ),
        "max_candidate_frame1_receiver_uint8_changed_fraction": (
            max_candidate_frame1_receiver_uint8_changed_fraction
        ),
        "max_accepted_frame1_receiver_uint8_changed_fraction": (
            max_accepted_frame1_receiver_uint8_changed_fraction
        ),
        "after_min_ratio_present": after_min_ratio is not None,
        "after_min_ratio_cleared": after_min_ratio_cleared,
        "remaining_debt_present": debt_remains,
        "accepted_steps_present": accepted_steps_present,
        "argmax_min_ratio_moved": argmax_ratio_moved,
        "score_weighted_total_unsolved_argmax_mass_reduced": score_debt_moved,
        "score_weighted_worst_unsolved_argmax_mass_reduced": worst_score_debt_moved,
        "hard_argmax_birth_progress": hard_argmax_birth_progress,
        "soft_loss_progress": loss_moved,
        "soft_progress_only_no_argmax_debt_move": soft_progress_only,
        "accepted_steps_without_argmax_debt_move": accepted_without_argmax_debt_move,
        "receiver_quantum_rejections_without_crossing": (
            receiver_quantum_rejections_without_crossing
        ),
        "accepted_steps_without_receiver_uint8_change": (
            accepted_steps_without_receiver_uint8_change
        ),
        "birth_progress_stage": (
            "hard_argmax_birth_or_debt_progress"
            if hard_argmax_birth_progress
            else (
                "accepted_steps_without_receiver_uint8_change"
                if accepted_steps_without_receiver_uint8_change
                else (
                "receiver_subquantum_updates_no_crossing"
                if receiver_quantum_rejections_without_crossing
                else (
                    "soft_loss_progress_only_no_argmax_debt_move"
                    if soft_progress_only
                    else (
                        "accepted_steps_without_argmax_debt_move"
                        if accepted_without_argmax_debt_move
                        else "hard_birth_progress_not_observed"
                    )
                )
                )
            )
        ),
        "no_accepted_steps_with_remaining_debt": bool(
            no_accepted_steps and debt_remains
        ),
    }


def receiver_cache_quality_manifest_summary(
    report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if report is None:
        return None
    gate = report.get("quality_gate") if isinstance(report, Mapping) else None
    gate_stats = gate.get("stats") if isinstance(gate, Mapping) else None
    crux = report.get("distortion_crux_probe") if isinstance(report, Mapping) else None
    argmax_probe = (
        report.get("segnet_argmax_probe") if isinstance(report, Mapping) else None
    )
    scorer_input_distribution_gate = (
        report.get("scorer_input_distribution_gate")
        if isinstance(report, Mapping)
        else None
    )
    distribution_blockers = (
        [str(blocker) for blocker in scorer_input_distribution_gate.get("blockers") or []]
        if isinstance(scorer_input_distribution_gate, Mapping)
        else None
    )
    return {
        "schema": "hi_nerv_receiver_cache_quality_summary.v1",
        "report_path": report.get("report_path"),
        "archive_path": report.get("archive_path"),
        "archive_sha256": report.get("archive_sha256"),
        "candidate_cache_dir": report.get("candidate_cache_dir"),
        "quality_gate_path": report.get("quality_gate_path"),
        "quality_gate_verdict": gate.get("verdict") if isinstance(gate, Mapping) else None,
        "quality_gate_passed": bool(report.get("quality_gate_passed")),
        "mlx_scorer_response_probe_path": report.get(
            "mlx_scorer_response_probe_path"
        ),
        "mlx_scorer_response_probe_required": bool(
            report.get("mlx_scorer_response_probe_required")
        ),
        "mlx_scorer_response_probe_passed": (
            bool(report.get("mlx_scorer_response_probe", {}).get("fit_gate_passed"))
            if isinstance(report.get("mlx_scorer_response_probe"), Mapping)
            else None
        ),
        "mlx_scorer_response_avg_posenet_dist": (
            report.get("mlx_scorer_response_probe", {}).get("avg_posenet_dist")
            if isinstance(report.get("mlx_scorer_response_probe"), Mapping)
            else None
        ),
        "mlx_scorer_response_avg_segnet_dist": (
            report.get("mlx_scorer_response_probe", {}).get("avg_segnet_dist")
            if isinstance(report.get("mlx_scorer_response_probe"), Mapping)
            else None
        ),
        "candidate_segnet_last_rgb_stats": (
            gate_stats.get("candidate_segnet_last_rgb")
            if isinstance(gate_stats, Mapping)
            else None
        ),
        "candidate_posenet_yuv6_pair_stats": (
            gate_stats.get("candidate_posenet_yuv6_pair")
            if isinstance(gate_stats, Mapping)
            else None
        ),
        "scorer_input_distribution_gate_path": report.get(
            "scorer_input_distribution_gate_path"
        ),
        "scorer_input_distribution_gate_passed": (
            bool(scorer_input_distribution_gate.get("fit_gate_passed"))
            if isinstance(scorer_input_distribution_gate, Mapping)
            else None
        ),
        "scorer_input_distribution_metrics": (
            {
                "segnet_last_frame_rgb": scorer_input_distribution_gate.get(
                    "segnet_last_frame_rgb"
                ),
                "posenet_yuv6_pair": scorer_input_distribution_gate.get(
                    "posenet_yuv6_pair"
                ),
                "posenet_yuv6_temporal_signal": scorer_input_distribution_gate.get(
                    "posenet_yuv6_temporal_signal"
                ),
                "thresholds": scorer_input_distribution_gate.get("thresholds"),
            }
            if isinstance(scorer_input_distribution_gate, Mapping)
            else None
        ),
        "scorer_input_distribution_blockers": distribution_blockers,
        "distance_to_reference": (
            gate.get("distance_to_reference") if isinstance(gate, Mapping) else None
        ),
        "distortion_crux_probe_path": report.get("distortion_crux_probe_path"),
        "distortion_crux_probe_passed": (
            bool(crux.get("fit_gate_passed")) if isinstance(crux, Mapping) else None
        ),
        "segnet_argmax_probe_path": report.get("segnet_argmax_probe_path"),
        "segnet_argmax_probe_passed": (
            bool(argmax_probe.get("fit_gate_passed"))
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "segnet_argmax_disagreement_rate": (
            argmax_probe.get("segnet_argmax_disagreement_rate")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "segnet_argmax_target_region_error_score_contribution": (
            argmax_probe.get("target_region_error_score_contribution")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "segnet_argmax_target_region_error_worst_class": (
            argmax_probe.get("target_region_error_worst_class")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "segnet_argmax_target_region_error_worst_score_contribution": (
            argmax_probe.get("target_region_error_worst_score_contribution")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "segnet_argmax_target_region_error_total_mismatch_pixels": (
            argmax_probe.get("target_region_error_total_mismatch_pixels")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "segnet_argmax_target_region_error_profile": (
            argmax_probe.get("target_region_error_profile")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "candidate_argmax_occupied_class_fraction": (
            argmax_probe.get("candidate_occupied_class_fraction")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "candidate_argmax_target_class_coverage_fraction": (
            argmax_probe.get("candidate_target_class_coverage_fraction")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "candidate_argmax_target_material_class_covered_count": (
            argmax_probe.get("candidate_target_material_class_covered_count")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "target_argmax_material_class_count": (
            argmax_probe.get("target_material_class_count")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "reference_argmax_occupied_class_fraction": (
            argmax_probe.get("reference_occupied_class_fraction")
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "segnet_argmax_probe_blockers": (
            [str(blocker) for blocker in argmax_probe.get("blockers") or []]
            if isinstance(argmax_probe, Mapping)
            else None
        ),
        "distortion_crux_dominant_domain": (
            crux.get("aggregate", {}).get("dominant_domain_top_k")
            if isinstance(crux, Mapping) and isinstance(crux.get("aggregate"), Mapping)
            else None
        ),
        "hard_pair_coverage": (
            crux.get("hard_pair_coverage") if isinstance(crux, Mapping) else None
        ),
        "blockers": [str(blocker) for blocker in report.get("blockers") or []],
    }


def _receiver_surface_identity_gate(report: Any) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        return {
            "schema": "hi_nerv_receiver_surface_identity_gate.v1",
            "archive_identity_present": False,
            "direct_receiver_parseback_present": False,
            "archive_sha256_mismatch": False,
            "candidate_cache_manifest_bound": False,
        }
    direct_report = report.get("direct_receiver_cache_report")
    direct_report = direct_report if isinstance(direct_report, Mapping) else {}
    cache_summary = report.get("cache_manifest_summary")
    cache_summary = cache_summary if isinstance(cache_summary, Mapping) else {}

    archive_sha256 = _hex_sha256_or_none(report.get("archive_sha256"))
    direct_archive_sha256 = _hex_sha256_or_none(direct_report.get("archive_sha256"))
    archive_bytes = _finite_mapping_value(report, "archive_bytes")
    top_zip_member = _nonempty_string_or_none(report.get("zip_member"))
    direct_zip_member = _nonempty_string_or_none(direct_report.get("zip_member"))
    cache_manifest_raw_sha256 = _hex_sha256_or_none(cache_summary.get("raw_sha256"))
    direct_raw_sha256 = _hex_sha256_or_none(
        direct_report.get("direct_render_raw_sha256")
    )
    candidate_cache_manifest_sha256 = _hex_sha256_or_none(
        report.get("candidate_cache_manifest_sha256")
    )

    archive_identity_present = bool(
        _nonempty_string_or_none(report.get("archive_path"))
        and archive_sha256 is not None
        and archive_bytes is not None
        and archive_bytes > 0.0
        and top_zip_member
    )
    direct_receiver_parseback_present = bool(
        direct_report.get("schema") == "hi_nerv_direct_receiver_cache_report.v1"
        and direct_report.get("source_family") == "hi_nerv"
        and direct_report.get("archive_magic") == "HIV1"
        and direct_archive_sha256 is not None
        and direct_zip_member
        and _finite_mapping_value(direct_report, "cached_pair_count") is not None
        and _finite_mapping_value(direct_report, "cached_pair_count") > 0.0
        and direct_raw_sha256 is not None
        and _nonempty_string_or_none(direct_report.get("identity_audit_sha256"))
        and direct_report.get("candidate_cache_identity_mode")
        == "hi_nerv_direct_receiver_render_cache_identity_audited_false_authority"
    )
    archive_sha256_mismatch = bool(
        archive_sha256 is not None
        and direct_archive_sha256 is not None
        and archive_sha256 != direct_archive_sha256
    )
    candidate_cache_manifest_bound = bool(
        _nonempty_string_or_none(report.get("candidate_cache_manifest_path"))
        and candidate_cache_manifest_sha256 is not None
        and cache_summary.get("source_kind") == "hi_nerv_direct_receiver_render"
        and cache_manifest_raw_sha256 is not None
        and direct_raw_sha256 is not None
        and cache_manifest_raw_sha256 == direct_raw_sha256
    )
    return {
        "schema": "hi_nerv_receiver_surface_identity_gate.v1",
        "archive_identity_present": archive_identity_present,
        "direct_receiver_parseback_present": direct_receiver_parseback_present,
        "archive_sha256_mismatch": archive_sha256_mismatch,
        "candidate_cache_manifest_bound": candidate_cache_manifest_bound,
        "archive_sha256": archive_sha256,
        "direct_receiver_archive_sha256": direct_archive_sha256,
        "zip_member": top_zip_member,
        "direct_receiver_zip_member": direct_zip_member,
        "cache_manifest_source_kind": cache_summary.get("source_kind"),
        "cache_manifest_raw_sha256": cache_manifest_raw_sha256,
        "direct_receiver_raw_sha256": direct_raw_sha256,
        "candidate_cache_manifest_sha256": candidate_cache_manifest_sha256,
    }


def _control_float(controls: Any, key: str) -> float:
    value = controls.get(key) if isinstance(controls, Mapping) else getattr(controls, key)
    finite = _finite_float(value)
    return 0.0 if finite is None else finite


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _nonempty_string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _hex_sha256_or_none(value: Any) -> str | None:
    text = _nonempty_string_or_none(value)
    if text is None or len(text) != 64:
        return None
    if any(ch not in "0123456789abcdefABCDEF" for ch in text):
        return None
    return text.lower()


def _below_floor(
    value: float,
    floor: float,
    *,
    epsilon: float = HI_NERV_SHORT_SCORER_SMOKE_FLOOR_COMPARISON_EPSILON,
) -> bool:
    return bool((float(value) + float(epsilon)) < float(floor))


def _finite_mapping_value(mapping: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return None
    return _finite_float(mapping.get(key))


def _direct_live_subcontrol_active_for_stage(
    *,
    configured_weight: float,
    stage_active_weight: float | None,
) -> bool:
    # Missing stage metadata means the trainer is older or the smoke failed
    # before emitting stage weights. Fail closed by requiring telemetry.
    if stage_active_weight is None:
        return configured_weight > 0.0
    return stage_active_weight > 0.0


def _first_finite_mapping_value(
    mapping: Mapping[str, Any] | None,
    keys: tuple[str, ...],
) -> float | None:
    for key in keys:
        value = _finite_mapping_value(mapping, key)
        if value is not None:
            return value
    return None


def _safe_loss_metric_key(value: str) -> str:
    return "_".join(
        part
        for part in "".join(
            ch if ch.isalnum() else "_" for ch in str(value)
        ).split("_")
        if part
    )


def _section_byte_pressure_loss_keys(constraint_key: str) -> tuple[str, ...]:
    if constraint_key == "hi_nerv_archive_total_bytes":
        return tuple(
            dict.fromkeys(
                [
                    *_SECTION_BYTE_PRESSURE_KEYS_BY_KIND["decoder"],
                    *_SECTION_BYTE_PRESSURE_KEYS_BY_KIND["latent"],
                ]
            )
        )
    section = (
        constraint_key.removeprefix("hi_nerv_")
        .removesuffix("_section_bytes")
        .strip()
    )
    if section == "decoder_state" or section.startswith("decoder"):
        return _SECTION_BYTE_PRESSURE_KEYS_BY_KIND["decoder"]
    if section.startswith("latents") or section.startswith("latent"):
        return _SECTION_BYTE_PRESSURE_KEYS_BY_KIND["latent"]
    return ()


def _section_byte_actual_loss_pressure_active(
    metrics: Mapping[str, Any],
    constraint_key: str,
) -> bool:
    for loss_key in _section_byte_pressure_loss_keys(constraint_key):
        safe_key = _safe_loss_metric_key(loss_key)
        active_weight = _finite_mapping_value(
            metrics,
            f"active_loss_weight__{safe_key}",
        )
        if active_weight is not None and active_weight > 0.0:
            return True
        weighted_part = _finite_mapping_value(
            metrics,
            f"loss_part_weighted_{loss_key}",
        )
        if weighted_part is not None and weighted_part > 0.0:
            return True
    return False


def _direct_live_dual_ascent_gate(
    metrics: Mapping[str, Any],
    *,
    required_constraint_keys: Mapping[str, str],
) -> dict[str, Any]:
    required = {
        str(constraint_key): str(metric_key)
        for constraint_key, metric_key in required_constraint_keys.items()
    }
    active_value = _finite_mapping_value(metrics, "dual_ascent_active")
    constraint_count = _finite_mapping_value(metrics, "dual_ascent_constraint_count")
    telemetry_by_constraint: dict[str, dict[str, float | None]] = {}
    missing_fields_by_constraint: dict[str, list[str]] = {}
    constraints_missing_observed_metric: list[str] = []
    constraints_without_updates: list[str] = []
    constraints_without_applied_weight: list[str] = []
    constraints_with_positive_violation_and_zero_lambda: list[str] = []

    telemetry_fields = {
        "metric": "dual_ascent_metric__{key}",
        "missing_metric": "dual_ascent_missing_metric__{key}",
        "lambda": "dual_ascent_lambda__{key}",
        "update_count": "dual_ascent_update_count__{key}",
        "weight_applied": "dual_ascent_weight_applied__{key}",
        "effective_loss_weight": "dual_ascent_effective_loss_weight__{key}",
        "violation": "dual_ascent_violation__{key}",
    }
    for constraint_key in sorted(required):
        row = {
            name: _finite_mapping_value(metrics, pattern.format(key=constraint_key))
            for name, pattern in telemetry_fields.items()
        }
        telemetry_by_constraint[constraint_key] = row
        missing_fields = [
            name
            for name in (
                "metric",
                "missing_metric",
                "lambda",
                "update_count",
                "weight_applied",
                "effective_loss_weight",
            )
            if row[name] is None
        ]
        if missing_fields:
            missing_fields_by_constraint[constraint_key] = missing_fields
            continue
        if (row["missing_metric"] or 0.0) > 0.0:
            constraints_missing_observed_metric.append(constraint_key)
        if (row["update_count"] or 0.0) <= 0.0:
            constraints_without_updates.append(constraint_key)
        if (
            (row["weight_applied"] or 0.0) <= 0.0
            or (row["effective_loss_weight"] or 0.0) <= 0.0
        ):
            constraints_without_applied_weight.append(constraint_key)
        if (
            row["violation"] is not None
            and row["violation"] > 0.0
            and (row["lambda"] or 0.0) <= 0.0
        ):
            constraints_with_positive_violation_and_zero_lambda.append(
                constraint_key
            )

    return {
        "schema": "hi_nerv_short_scorer_direct_live_dual_ascent_gate.v1",
        "required": bool(required),
        "active": bool(active_value is not None and active_value > 0.0),
        "dual_ascent_active": active_value,
        "dual_ascent_constraint_count": constraint_count,
        "required_observed_metric_keys_by_constraint": dict(sorted(required.items())),
        "telemetry_by_constraint": telemetry_by_constraint,
        "missing_constraint_telemetry": sorted(missing_fields_by_constraint),
        "missing_fields_by_constraint": missing_fields_by_constraint,
        "constraints_missing_observed_metric": constraints_missing_observed_metric,
        "constraints_without_updates": constraints_without_updates,
        "constraints_without_applied_weight": constraints_without_applied_weight,
        "constraints_with_positive_violation_and_zero_lambda": (
            constraints_with_positive_violation_and_zero_lambda
        ),
        "authority": HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY,
        "axis_tag": HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
        **FALSE_AUTHORITY,
    }


def _decoder_weight_waterfill_actuation_gate(
    metrics: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = dict(metadata) if isinstance(metadata, Mapping) else {}
    fake_quant = (
        plan.get("fake_quant_forward")
        if isinstance(plan.get("fake_quant_forward"), Mapping)
        else {}
    )
    row_count = _finite_float(plan.get("row_count")) or 0.0
    targeted_tensor_count = _finite_float(fake_quant.get("targeted_tensor_count")) or 0.0
    required = bool(plan.get("attached") is True or row_count > 0.0)
    train_time_fake_quant_bound = bool(
        plan.get("train_time_fake_quant_bound") is True
        and targeted_tensor_count > 0.0
    )
    requested_count = _finite_mapping_value(
        metrics,
        "gradient_multiplier_requested_control_count",
    )
    applied_leaf_count = _finite_mapping_value(
        metrics,
        "gradient_multiplier_applied_leaf_count",
    )
    requested_but_unapplied = _finite_mapping_value(
        metrics,
        "gradient_multiplier_requested_but_unapplied",
    )
    missing_exact_name_count = _finite_mapping_value(
        metrics,
        "gradient_multiplier_missing_exact_name_count",
    )
    gradient_multiplier_metrics_present = all(
        value is not None
        for value in (
            requested_count,
            applied_leaf_count,
            requested_but_unapplied,
            missing_exact_name_count,
        )
    )
    requested_control_absent_for_nonempty_plan = bool(
        gradient_multiplier_metrics_present
        and required
        and row_count > 0.0
        and (requested_count or 0.0) <= 0.0
    )
    return {
        "schema": "hi_nerv_short_scorer_decoder_weight_waterfill_actuation_gate.v1",
        "required": required,
        "attached": plan.get("attached") is True,
        "row_count": row_count,
        "train_time_fake_quant_bound": train_time_fake_quant_bound,
        "fake_quant_targeted_tensor_count": targeted_tensor_count,
        "gradient_multiplier_metrics_present": gradient_multiplier_metrics_present,
        "gradient_multiplier_requested_control_count": requested_count,
        "gradient_multiplier_applied_leaf_count": applied_leaf_count,
        "gradient_multiplier_requested_but_unapplied": requested_but_unapplied,
        "gradient_multiplier_missing_exact_name_count": missing_exact_name_count,
        "gradient_multiplier_requested_control_absent_for_nonempty_plan": (
            requested_control_absent_for_nonempty_plan
        ),
        "requested_but_unapplied": bool(
            gradient_multiplier_metrics_present
            and (
                (requested_but_unapplied or 0.0) > 0.0
                or (
                    (requested_count or 0.0) > 0.0
                    and (applied_leaf_count or 0.0) <= 0.0
                )
                or requested_control_absent_for_nonempty_plan
            )
        ),
        "missing_exact_name_count_positive": bool(
            gradient_multiplier_metrics_present
            and (missing_exact_name_count or 0.0) > 0.0
        ),
        "metadata": _metadata_safe(plan),
        "authority": HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY,
        "axis_tag": HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
        **FALSE_AUTHORITY,
    }


def _output_head_target_init_gate(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    required = isinstance(metadata, Mapping)
    payload = dict(metadata) if isinstance(metadata, Mapping) else {}
    contrast_payload = (
        payload.get("contrast_init")
        if isinstance(payload.get("contrast_init"), Mapping)
        else {}
    )
    bias_init_enabled = bool(payload.get("enabled") is True)
    contrast_init_enabled = bool(contrast_payload.get("enabled") is True)
    return {
        "schema": "hi_nerv_short_scorer_output_head_target_init_gate.v1",
        "required": required,
        "bias_init_present": bool(payload),
        "bias_init_enabled": bias_init_enabled,
        "contrast_init_present": bool(contrast_payload),
        "contrast_init_enabled": contrast_init_enabled,
        "metadata": _metadata_safe(payload),
        "authority": HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY,
        "axis_tag": HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
        **FALSE_AUTHORITY,
    }


def _section_byte_dual_ascent_gate(
    metrics: Mapping[str, Any],
    *,
    require_section_byte_dual_ascent: bool,
) -> dict[str, Any]:
    metric_keys = set(metrics)
    section_metric_constraints: dict[str, str] = {}
    archive_metric_present = (
        "train_time_archive_rate_score" in metric_keys
        or "dual_ascent_metric__hi_nerv_archive_total_bytes" in metric_keys
    )
    if archive_metric_present:
        section_metric_constraints["hi_nerv_archive_total_bytes"] = (
            "train_time_archive_rate_score"
        )
    for key in sorted(metric_keys):
        if key.startswith("train_time_section_rate_score__"):
            section = key.removeprefix("train_time_section_rate_score__")
            if section:
                section_metric_constraints[f"hi_nerv_{section}_section_bytes"] = key
        elif (
            key.startswith("dual_ascent_metric__hi_nerv_")
            and key.endswith("_section_bytes")
        ):
            constraint_key = key.removeprefix("dual_ascent_metric__")
            section = constraint_key.removeprefix("hi_nerv_").removesuffix(
                "_section_bytes"
            )
            if section:
                section_metric_constraints.setdefault(
                    constraint_key,
                    f"train_time_section_rate_score__{section}",
                )
    section_or_archive_metric_present = bool(section_metric_constraints)
    if not require_section_byte_dual_ascent and not section_or_archive_metric_present:
        return {
            "schema": "hi_nerv_short_scorer_section_byte_dual_ascent_gate.v1",
            "required": False,
            "section_or_archive_metric_present": False,
            "active": False,
            "required_observed_metric_keys_by_constraint": {},
            "section_constraint_count": 0,
            "archive_metric_present": False,
            "authority": HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY,
            "axis_tag": HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
            **FALSE_AUTHORITY,
        }
    telemetry_fields = {
        "metric": "dual_ascent_metric__{key}",
        "missing_metric": "dual_ascent_missing_metric__{key}",
        "lambda": "dual_ascent_lambda__{key}",
        "update_count": "dual_ascent_update_count__{key}",
        "weight_applied": "dual_ascent_weight_applied__{key}",
        "effective_loss_weight": "dual_ascent_effective_loss_weight__{key}",
        "violation": "dual_ascent_violation__{key}",
    }
    active_value = _finite_mapping_value(metrics, "dual_ascent_active")
    constraint_count = _finite_mapping_value(metrics, "dual_ascent_constraint_count")
    telemetry_by_constraint: dict[str, dict[str, float | None]] = {}
    missing_fields_by_constraint: dict[str, list[str]] = {}
    constraints_missing_observed_metric: list[str] = []
    constraints_without_updates: list[str] = []
    constraints_without_applied_weight: list[str] = []
    constraints_with_positive_violation_and_zero_lambda: list[str] = []
    slack_constraints_without_applied_weight: list[str] = []
    priced_only_constraints: list[str] = []
    constraints_with_active_loss_pressure: list[str] = []

    for constraint_key in sorted(section_metric_constraints):
        observed_key = section_metric_constraints[constraint_key]
        row = {
            name: _finite_mapping_value(metrics, pattern.format(key=constraint_key))
            for name, pattern in telemetry_fields.items()
        }
        if row["metric"] is None:
            row["metric"] = _finite_mapping_value(metrics, observed_key)
        telemetry_by_constraint[constraint_key] = row
        if constraint_key in _SECTION_BYTE_PRICED_ONLY_CONSTRAINT_KEYS:
            priced_only_constraints.append(constraint_key)
            continue
        missing_fields = [
            name
            for name in (
                "metric",
                "missing_metric",
                "lambda",
                "update_count",
                "weight_applied",
                "effective_loss_weight",
            )
            if row[name] is None
        ]
        if missing_fields:
            missing_fields_by_constraint[constraint_key] = missing_fields
            continue
        if (row["missing_metric"] or 0.0) > 0.0:
            constraints_missing_observed_metric.append(constraint_key)
        if (row["update_count"] or 0.0) <= 0.0:
            constraints_without_updates.append(constraint_key)
        positive_violation = bool(row["violation"] is not None and row["violation"] > 0.0)
        actual_loss_pressure_active = _section_byte_actual_loss_pressure_active(
            metrics,
            constraint_key,
        )
        if actual_loss_pressure_active:
            constraints_with_active_loss_pressure.append(constraint_key)
        missing_pressure = bool(
            not actual_loss_pressure_active
            and (
                (row["weight_applied"] or 0.0) <= 0.0
                or (row["effective_loss_weight"] or 0.0) <= 0.0
            )
        )
        if positive_violation and missing_pressure:
            constraints_without_applied_weight.append(constraint_key)
        elif (not positive_violation) and missing_pressure:
            slack_constraints_without_applied_weight.append(constraint_key)
        if positive_violation and (row["lambda"] or 0.0) <= 0.0:
            constraints_with_positive_violation_and_zero_lambda.append(
                constraint_key
            )
    missing_constraint_telemetry = sorted(missing_fields_by_constraint)
    return {
        "schema": "hi_nerv_short_scorer_section_byte_dual_ascent_gate.v1",
        "required": True,
        "active": bool(active_value is not None and active_value > 0.0),
        "dual_ascent_active": active_value,
        "dual_ascent_constraint_count": constraint_count,
        "required_observed_metric_keys_by_constraint": dict(
            sorted(section_metric_constraints.items())
        ),
        "telemetry_by_constraint": telemetry_by_constraint,
        "missing_constraint_telemetry": missing_constraint_telemetry,
        "missing_fields_by_constraint": missing_fields_by_constraint,
        "constraints_missing_observed_metric": constraints_missing_observed_metric,
        "constraints_without_updates": constraints_without_updates,
        "constraints_without_applied_weight": constraints_without_applied_weight,
        "constraints_with_positive_violation_and_zero_lambda": (
            constraints_with_positive_violation_and_zero_lambda
        ),
        "slack_constraints_without_applied_weight": (
            slack_constraints_without_applied_weight
        ),
        "constraints_with_active_loss_pressure": constraints_with_active_loss_pressure,
        "priced_only_constraints": priced_only_constraints,
        "section_or_archive_metric_present": section_or_archive_metric_present,
        "section_constraint_count": len(
            [
                key
                for key in section_metric_constraints
                if key.endswith("_section_bytes")
            ]
        ),
        "archive_metric_present": archive_metric_present,
        "authority": HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY,
        "axis_tag": HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
        **FALSE_AUTHORITY,
    }


def _metadata_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _metadata_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_safe(inner) for inner in value]
    return value


__all__ = [
    "HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_OCCUPIED_CLASS_FRACTION",
    "HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_TARGET_CLASS_COVERAGE_FRACTION",
    "HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA",
    "build_hinerv_short_scorer_smoke_readiness_report",
    "hinerv_short_scorer_smoke_long_run_admission",
    "hinerv_short_scorer_smoke_readiness_summary",
    "receiver_cache_quality_manifest_summary",
]
