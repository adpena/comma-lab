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
HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY = (
    "false_authority_macos_mlx_training_no_contest_score_claim"
)
HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG = "[macOS-MLX research-signal]"


def build_hinerv_short_scorer_smoke_readiness_report(
    *,
    train_time_controls: Any,
    final_loss_components: Mapping[str, Any] | None,
    post_export_quality: Mapping[str, Any] | None,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    allow_mock_scorer_teacher: bool,
    unscored_research_smoke_enabled: bool = False,
    min_segnet_occupied_class_fraction_for_fit_gate: float = (
        HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_OCCUPIED_CLASS_FRACTION
    ),
) -> dict[str, Any]:
    min_occupied = _finite_float(min_segnet_occupied_class_fraction_for_fit_gate)
    if min_occupied is None or not 0.0 <= min_occupied <= 1.0:
        min_occupied = HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_OCCUPIED_CLASS_FRACTION
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
    direct_live_enabled = direct_live_weight > 0.0
    segnet_weight = _finite_float(segnet_distillation_weight)
    pose_weight = _finite_float(pose_distillation_weight)
    if bool(allow_mock_scorer_teacher):
        add_blocker("hi_nerv_short_smoke_mock_scorer_teacher_enabled")
    if bool(unscored_research_smoke_enabled):
        add_blocker("hi_nerv_short_smoke_unscored_research_smoke_enabled")
    if (segnet_weight is None or segnet_weight <= 0.0) and not direct_live_enabled:
        add_blocker("hi_nerv_short_smoke_real_segnet_teacher_not_requested")
    if pose_weight is None or pose_weight <= 0.0:
        add_blocker("hi_nerv_short_smoke_real_posenet_teacher_not_requested")
    direct_live_keys = (
        "loss_part_segnet_direct_live_distill",
        "loss_part_segnet_direct_live_argmax_disagreement",
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction",
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
        candidate_occupied = direct_live_metrics[
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction"
        ]
        if candidate_occupied is not None and candidate_occupied < min_occupied:
            add_blocker("hi_nerv_short_smoke_direct_live_class_occupancy_collapsed")

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

    receiver_cache_summary = receiver_cache_quality_manifest_summary(post_export_quality)
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
            if receiver_candidate_occupied is None:
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_segnet_argmax_occupancy_missing"
                )
            elif receiver_candidate_occupied < min_occupied:
                add_blocker(
                    "hi_nerv_short_smoke_receiver_cache_segnet_argmax_class_occupancy_collapsed"
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

    ready = not actionable_blockers
    return {
        "schema": HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA,
        "authority": HI_NERV_SHORT_SCORER_SMOKE_AUTHORITY,
        "axis_tag": HI_NERV_SHORT_SCORER_SMOKE_AXIS_TAG,
        "scope": "local_mlx_false_authority_short_scorer_teacher_smoke_gate",
        "short_scorer_teacher_smoke_ready": ready,
        "ready_for_long_run": ready,
        "teacher_gate": {
            "real_segnet_teacher_requested": bool(
                segnet_weight is not None and segnet_weight > 0.0
            ),
            "real_posenet_teacher_requested": bool(
                pose_weight is not None and pose_weight > 0.0
            ),
            "mock_scorer_teacher_allowed": bool(allow_mock_scorer_teacher),
            "unscored_research_smoke_enabled": bool(unscored_research_smoke_enabled),
            "segnet_distillation_weight": segnet_weight,
            "pose_distillation_weight": pose_weight,
        },
        "direct_live_segnet_gate": {
            "enabled": direct_live_enabled,
            "weight": direct_live_weight,
            "min_candidate_occupied_class_fraction_for_fit_gate": min_occupied,
            "metrics": direct_live_metrics,
        },
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
        "receiver_cache_quality": receiver_cache_summary,
        "final_loss_components_present": bool(final_components),
        "actionable_blockers": actionable_blockers,
        "blockers": [
            "hi_nerv_short_scorer_smoke_is_false_authority",
            *actionable_blockers,
        ],
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
            "scorer_input_contrast_floor_gate": report.get(
                "scorer_input_contrast_floor_gate"
            ),
            "receiver_cache_quality": report.get("receiver_cache_quality"),
            "actionable_blockers": [
                str(blocker) for blocker in report.get("actionable_blockers") or []
            ],
            "blockers": [str(blocker) for blocker in report.get("blockers") or []],
        }
    )


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
        "candidate_argmax_occupied_class_fraction": (
            argmax_probe.get("candidate_occupied_class_fraction")
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


def _finite_mapping_value(mapping: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return None
    return _finite_float(mapping.get(key))


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
    "HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA",
    "build_hinerv_short_scorer_smoke_readiness_report",
    "hinerv_short_scorer_smoke_readiness_summary",
    "receiver_cache_quality_manifest_summary",
]
