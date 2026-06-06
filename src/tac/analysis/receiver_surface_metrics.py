# SPDX-License-Identifier: MIT
"""Canonical receiver-surface metric names for NeRV evaluator actions.

This module is the single receiver-surface contract for production consumers:
action effects, crux traces, servo lifts, runner summaries, and readiness DAGs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

RECEIVER_SURFACE_CONTRACT_SCHEMA = "nerv_receiver_surface_contract.v1"

RECEIVER_SURFACE_LOSS_DELTA_KEYS = ("receiver_surface_loss_delta",)
RECEIVER_SURFACE_FLOAT_RGB_DELTA_LINF_KEYS = ("receiver_surface_float_rgb_delta_linf",)
RECEIVER_SURFACE_UINT8_CHANGED_PIXELS_KEYS = (
    "receiver_surface_uint8_changed_pixels",
    "uint8_changed_pixels",
)
RECEIVER_SURFACE_UINT8_DELTA_ABS_MAX_KEYS = (
    "receiver_surface_uint8_delta_abs_max",
    "uint8_delta_abs_max",
)
RECEIVER_SURFACE_SEGNET_INPUT_DELTA_LINF_KEYS = (
    "receiver_surface_segnet_input_delta_linf",
    "segnet_input_delta_linf",
)
RECEIVER_SURFACE_MARGIN_P50_DELTA_KEYS = (
    "receiver_surface_worst_region_margin_p50_delta",
    "worst_region_margin_p50_delta",
)
RECEIVER_SURFACE_ARGMAX_FLIPPED_PIXELS_KEYS = (
    "receiver_surface_argmax_flipped_pixels",
    "receiver_surface_segnet_argmax_flipped_pixels",
    "argmax_flipped_pixels",
    "segnet_argmax_flipped_pixels",
)
RECEIVER_SURFACE_ARGMAX_CHANGED_COUNT_KEYS = (
    "receiver_surface_argmax_changed_count_region",
    "receiver_surface_argmax_changed_count",
    "argmax_changed_count_region",
)
RECEIVER_SURFACE_TARGET_HARD_WON_COUNT_KEYS = (
    "receiver_surface_target_hard_won_count",
    "receiver_surface_target_hard_won_count_region",
    "target_hard_won_count",
    "hard_won_count",
)
RECEIVER_SURFACE_TARGET_HARD_LOST_COUNT_KEYS = (
    "receiver_surface_target_hard_lost_count",
    "receiver_surface_target_hard_lost_count_region",
    "target_hard_lost_count",
)
RECEIVER_SURFACE_NET_TARGET_SUPPORT_DELTA_KEYS = (
    "receiver_surface_net_target_support_delta",
    "receiver_surface_net_target_support_delta_region",
    "net_target_support_delta",
)
RECEIVER_SURFACE_WRONG_TO_TARGET_COUNT_KEYS = (
    "receiver_surface_wrong_to_target_count",
    "wrong_to_target_count",
)
RECEIVER_SURFACE_TARGET_TO_WRONG_COUNT_KEYS = (
    "receiver_surface_target_to_wrong_count",
    "target_to_wrong_count",
)
RECEIVER_SURFACE_WRONG_TO_WRONG_COUNT_KEYS = (
    "receiver_surface_wrong_to_wrong_count",
    "wrong_to_wrong_count",
)
RECEIVER_SURFACE_POSENET_INPUT_DELTA_LINF_KEYS = (
    "receiver_surface_posenet_input_delta_linf",
    "posenet_input_delta_linf",
)
RECEIVER_SURFACE_POSE_OUTPUT_DELTA_KEYS = (
    "receiver_surface_pose_output_delta",
    "pose_output_delta",
    "pose_output_delta_l2",
)
RECEIVER_SURFACE_FAKEQUANT_TARGET_HARD_WON_COUNT_KEYS = (
    "receiver_surface_fakequant_target_hard_won_count",
    "fakequant_target_hard_won_count",
)
RECEIVER_SURFACE_FAKEQUANT_NET_TARGET_SUPPORT_DELTA_KEYS = (
    "receiver_surface_fakequant_net_target_support_delta",
    "fakequant_net_target_support_delta",
)
RECEIVER_SURFACE_FAKEQUANT_ARGMAX_FLIPPED_PIXELS_KEYS = (
    "receiver_surface_fakequant_argmax_flipped_pixels",
    "receiver_surface_fakequant_segnet_argmax_flipped_pixels",
    "fakequant_argmax_flipped_pixels",
    "fakequant_segnet_argmax_flipped_pixels",
)
RECEIVER_SURFACE_FAKEQUANT_MARGIN_DELTA_KEYS = (
    "receiver_surface_fakequant_margin_delta",
    "fakequant_margin_delta",
    "fakequant_segnet_margin_delta",
)
RECEIVER_SURFACE_FAKEQUANT_POSE_OUTPUT_DELTA_KEYS = (
    "receiver_surface_fakequant_pose_output_delta",
    "fakequant_pose_output_delta",
    "fakequant_pose_output_delta_l2",
)
RECEIVER_SURFACE_PARSEBACK_TARGET_HARD_WON_COUNT_KEYS = (
    "receiver_surface_parseback_target_hard_won_count",
    "parseback_target_hard_won_count",
)
RECEIVER_SURFACE_PARSEBACK_NET_TARGET_SUPPORT_DELTA_KEYS = (
    "receiver_surface_parseback_net_target_support_delta",
    "parseback_net_target_support_delta",
)
RECEIVER_SURFACE_PARSEBACK_ARGMAX_FLIPPED_PIXELS_KEYS = (
    "receiver_surface_parseback_argmax_flipped_pixels",
    "receiver_surface_parseback_segnet_argmax_flipped_pixels",
    "parseback_argmax_flipped_pixels",
    "parseback_segnet_argmax_flipped_pixels",
)
RECEIVER_SURFACE_PARSEBACK_MARGIN_DELTA_KEYS = (
    "receiver_surface_parseback_margin_delta",
    "parseback_margin_delta",
    "parseback_segnet_margin_delta",
)
RECEIVER_SURFACE_PARSEBACK_POSE_OUTPUT_DELTA_KEYS = (
    "receiver_surface_parseback_pose_output_delta",
    "parseback_pose_output_delta",
    "parseback_pose_output_delta_l2",
)
RECEIVER_SURFACE_INFLATED_TARGET_HARD_WON_COUNT_KEYS = (
    "receiver_surface_inflated_target_hard_won_count",
    "inflated_target_hard_won_count",
)
RECEIVER_SURFACE_INFLATED_NET_TARGET_SUPPORT_DELTA_KEYS = (
    "receiver_surface_inflated_net_target_support_delta",
    "inflated_net_target_support_delta",
)
RECEIVER_SURFACE_INFLATED_ARGMAX_FLIPPED_PIXELS_KEYS = (
    "receiver_surface_inflated_argmax_flipped_pixels",
    "receiver_surface_inflated_segnet_argmax_flipped_pixels",
    "inflated_argmax_flipped_pixels",
    "inflated_segnet_argmax_flipped_pixels",
)
RECEIVER_SURFACE_INFLATED_POSE_OUTPUT_DELTA_KEYS = (
    "receiver_surface_inflated_pose_output_delta",
    "inflated_pose_output_delta",
    "inflated_pose_output_delta_l2",
)
RECEIVER_SURFACE_FAKEQUANT_SURVIVAL_KEYS = (
    "receiver_surface_fakequant_survival",
    "fakequant_survival",
    "fakequant_survived",
)
RECEIVER_SURFACE_PARSEBACK_SURVIVAL_KEYS = (
    "receiver_surface_parseback_survival",
    "parseback_survival",
    "parseback_survived",
)
RECEIVER_SURFACE_INFLATE_SURVIVAL_KEYS = (
    "receiver_surface_inflate_survival",
    "inflate_survival",
    "inflate_survived",
)
RECEIVER_SURFACE_SURVIVAL_KEYS_BY_LABEL = {
    "fakequant": RECEIVER_SURFACE_FAKEQUANT_SURVIVAL_KEYS,
    "parseback": RECEIVER_SURFACE_PARSEBACK_SURVIVAL_KEYS,
    "inflate": RECEIVER_SURFACE_INFLATE_SURVIVAL_KEYS,
}

RECEIVER_SURFACE_SEG_TARGET_SUPPORT_KEYS = (
    *RECEIVER_SURFACE_TARGET_HARD_WON_COUNT_KEYS,
    *RECEIVER_SURFACE_NET_TARGET_SUPPORT_DELTA_KEYS,
    *RECEIVER_SURFACE_WRONG_TO_TARGET_COUNT_KEYS,
)
RECEIVER_SURFACE_FAKEQUANT_TARGET_SUPPORT_KEYS = (
    *RECEIVER_SURFACE_FAKEQUANT_TARGET_HARD_WON_COUNT_KEYS,
    *RECEIVER_SURFACE_FAKEQUANT_NET_TARGET_SUPPORT_DELTA_KEYS,
)
RECEIVER_SURFACE_PARSEBACK_TARGET_SUPPORT_KEYS = (
    *RECEIVER_SURFACE_PARSEBACK_TARGET_HARD_WON_COUNT_KEYS,
    *RECEIVER_SURFACE_PARSEBACK_NET_TARGET_SUPPORT_DELTA_KEYS,
)
RECEIVER_SURFACE_INFLATED_TARGET_SUPPORT_KEYS = (
    *RECEIVER_SURFACE_INFLATED_TARGET_HARD_WON_COUNT_KEYS,
    *RECEIVER_SURFACE_INFLATED_NET_TARGET_SUPPORT_DELTA_KEYS,
)
RECEIVER_SURFACE_SEG_SCORER_MOTION_KEYS = (
    *RECEIVER_SURFACE_ARGMAX_FLIPPED_PIXELS_KEYS,
    *RECEIVER_SURFACE_SEG_TARGET_SUPPORT_KEYS,
)
RECEIVER_SURFACE_POSE_SCORER_MOTION_KEYS = (*RECEIVER_SURFACE_POSE_OUTPUT_DELTA_KEYS,)
RECEIVER_SURFACE_FAKEQUANT_SCORER_MOTION_KEYS = (
    *RECEIVER_SURFACE_FAKEQUANT_ARGMAX_FLIPPED_PIXELS_KEYS,
    *RECEIVER_SURFACE_FAKEQUANT_TARGET_SUPPORT_KEYS,
    *RECEIVER_SURFACE_FAKEQUANT_POSE_OUTPUT_DELTA_KEYS,
)
RECEIVER_SURFACE_PARSEBACK_SCORER_MOTION_KEYS = (
    *RECEIVER_SURFACE_PARSEBACK_ARGMAX_FLIPPED_PIXELS_KEYS,
    *RECEIVER_SURFACE_PARSEBACK_TARGET_SUPPORT_KEYS,
    *RECEIVER_SURFACE_PARSEBACK_POSE_OUTPUT_DELTA_KEYS,
)
RECEIVER_SURFACE_INFLATED_SCORER_MOTION_KEYS = (
    *RECEIVER_SURFACE_INFLATED_ARGMAX_FLIPPED_PIXELS_KEYS,
    *RECEIVER_SURFACE_INFLATED_TARGET_SUPPORT_KEYS,
    *RECEIVER_SURFACE_INFLATED_POSE_OUTPUT_DELTA_KEYS,
)
RECEIVER_SURFACE_RECEIVER_VISIBLE_KEYS = (
    *RECEIVER_SURFACE_UINT8_CHANGED_PIXELS_KEYS,
    *RECEIVER_SURFACE_UINT8_DELTA_ABS_MAX_KEYS,
    *RECEIVER_SURFACE_SEGNET_INPUT_DELTA_LINF_KEYS,
    *RECEIVER_SURFACE_POSENET_INPUT_DELTA_LINF_KEYS,
    *RECEIVER_SURFACE_SEG_SCORER_MOTION_KEYS,
    *RECEIVER_SURFACE_POSE_SCORER_MOTION_KEYS,
)
RECEIVER_SURFACE_SCORER_VISIBLE_KEYS = (
    *RECEIVER_SURFACE_SEG_SCORER_MOTION_KEYS,
    *RECEIVER_SURFACE_POSE_SCORER_MOTION_KEYS,
    *RECEIVER_SURFACE_FAKEQUANT_SCORER_MOTION_KEYS,
    *RECEIVER_SURFACE_PARSEBACK_SCORER_MOTION_KEYS,
    *RECEIVER_SURFACE_INFLATED_SCORER_MOTION_KEYS,
)
RECEIVER_SURFACE_ALIAS_METRIC_ROWS = (
    (RECEIVER_SURFACE_LOSS_DELTA_KEYS, "receiver_surface_loss_delta", "loss"),
    (RECEIVER_SURFACE_FLOAT_RGB_DELTA_LINF_KEYS, "receiver_surface_float_rgb_delta_linf", "rgb"),
    (RECEIVER_SURFACE_UINT8_CHANGED_PIXELS_KEYS, "receiver_surface_uint8_changed_pixels", "receiver"),
    (RECEIVER_SURFACE_SEGNET_INPUT_DELTA_LINF_KEYS, "receiver_surface_segnet_input_delta_linf", "segnet"),
    (RECEIVER_SURFACE_MARGIN_P50_DELTA_KEYS, "receiver_surface_worst_region_margin_p50_delta", "segnet"),
    (RECEIVER_SURFACE_ARGMAX_FLIPPED_PIXELS_KEYS, "receiver_surface_argmax_flipped_pixels", "segnet"),
    (RECEIVER_SURFACE_ARGMAX_CHANGED_COUNT_KEYS, "receiver_surface_argmax_changed_count_region", "segnet"),
    (RECEIVER_SURFACE_TARGET_HARD_WON_COUNT_KEYS, "receiver_surface_target_hard_won_count", "segnet"),
    (RECEIVER_SURFACE_TARGET_HARD_LOST_COUNT_KEYS, "receiver_surface_target_hard_lost_count", "segnet"),
    (RECEIVER_SURFACE_NET_TARGET_SUPPORT_DELTA_KEYS, "receiver_surface_net_target_support_delta", "segnet"),
    (RECEIVER_SURFACE_WRONG_TO_TARGET_COUNT_KEYS, "receiver_surface_wrong_to_target_count", "segnet"),
    (RECEIVER_SURFACE_TARGET_TO_WRONG_COUNT_KEYS, "receiver_surface_target_to_wrong_count", "segnet"),
    (RECEIVER_SURFACE_WRONG_TO_WRONG_COUNT_KEYS, "receiver_surface_wrong_to_wrong_count", "segnet"),
    (RECEIVER_SURFACE_POSENET_INPUT_DELTA_LINF_KEYS, "receiver_surface_posenet_input_delta_linf", "posenet"),
    (RECEIVER_SURFACE_POSE_OUTPUT_DELTA_KEYS, "receiver_surface_pose_output_delta", "posenet"),
    (
        RECEIVER_SURFACE_FAKEQUANT_ARGMAX_FLIPPED_PIXELS_KEYS,
        "receiver_surface_fakequant_argmax_flipped_pixels",
        "fakequant",
    ),
    (RECEIVER_SURFACE_FAKEQUANT_MARGIN_DELTA_KEYS, "receiver_surface_fakequant_margin_delta", "fakequant"),
    (
        RECEIVER_SURFACE_FAKEQUANT_TARGET_HARD_WON_COUNT_KEYS,
        "receiver_surface_fakequant_target_hard_won_count",
        "fakequant",
    ),
    (
        RECEIVER_SURFACE_FAKEQUANT_NET_TARGET_SUPPORT_DELTA_KEYS,
        "receiver_surface_fakequant_net_target_support_delta",
        "fakequant",
    ),
    (
        RECEIVER_SURFACE_PARSEBACK_ARGMAX_FLIPPED_PIXELS_KEYS,
        "receiver_surface_parseback_argmax_flipped_pixels",
        "parseback",
    ),
    (RECEIVER_SURFACE_PARSEBACK_MARGIN_DELTA_KEYS, "receiver_surface_parseback_margin_delta", "parseback"),
    (
        RECEIVER_SURFACE_PARSEBACK_TARGET_HARD_WON_COUNT_KEYS,
        "receiver_surface_parseback_target_hard_won_count",
        "parseback",
    ),
    (
        RECEIVER_SURFACE_PARSEBACK_NET_TARGET_SUPPORT_DELTA_KEYS,
        "receiver_surface_parseback_net_target_support_delta",
        "parseback",
    ),
    (
        RECEIVER_SURFACE_INFLATED_ARGMAX_FLIPPED_PIXELS_KEYS,
        "receiver_surface_inflated_argmax_flipped_pixels",
        "inflate",
    ),
    (
        RECEIVER_SURFACE_INFLATED_TARGET_HARD_WON_COUNT_KEYS,
        "receiver_surface_inflated_target_hard_won_count",
        "inflate",
    ),
    (
        RECEIVER_SURFACE_INFLATED_NET_TARGET_SUPPORT_DELTA_KEYS,
        "receiver_surface_inflated_net_target_support_delta",
        "inflate",
    ),
)
RECEIVER_SURFACE_CANONICAL_METRIC_ROWS = (
    *RECEIVER_SURFACE_ALIAS_METRIC_ROWS,
    (RECEIVER_SURFACE_UINT8_DELTA_ABS_MAX_KEYS, "receiver_surface_uint8_delta_abs_max", "receiver"),
    (RECEIVER_SURFACE_FAKEQUANT_POSE_OUTPUT_DELTA_KEYS, "receiver_surface_fakequant_pose_output_delta", "fakequant"),
    (RECEIVER_SURFACE_PARSEBACK_POSE_OUTPUT_DELTA_KEYS, "receiver_surface_parseback_pose_output_delta", "parseback"),
    (RECEIVER_SURFACE_INFLATED_POSE_OUTPUT_DELTA_KEYS, "receiver_surface_inflated_pose_output_delta", "inflate"),
)
RECEIVER_SURFACE_TRACE_METRIC_ROWS = (
    *(
        ((canonical_key,), canonical_key, axis)
        for _keys, canonical_key, axis in RECEIVER_SURFACE_CANONICAL_METRIC_ROWS
    ),
    (RECEIVER_SURFACE_FAKEQUANT_SURVIVAL_KEYS, "receiver_surface_fakequant_survival", "fakequant"),
    (RECEIVER_SURFACE_PARSEBACK_SURVIVAL_KEYS, "receiver_surface_parseback_survival", "parseback"),
    (RECEIVER_SURFACE_INFLATE_SURVIVAL_KEYS, "receiver_surface_inflate_survival", "inflate"),
)
RECEIVER_SURFACE_EVIDENCE_KEYS = tuple(
    canonical_key for _keys, canonical_key, _axis in RECEIVER_SURFACE_TRACE_METRIC_ROWS
)


def normalize_receiver_surface(value: Any) -> dict[str, Any]:
    """Return a scalar-only surface with canonical receiver keys populated."""

    if not isinstance(value, Mapping):
        return {}
    surface: dict[str, Any] = {}
    for key, raw in value.items():
        if _scalar_ok(raw):
            surface[str(key)] = raw
    for keys, canonical_key, _axis in RECEIVER_SURFACE_CANONICAL_METRIC_ROWS:
        found = surface_value(surface, *keys)
        if found is not None:
            surface[canonical_key] = found
    for label in RECEIVER_SURFACE_SURVIVAL_KEYS_BY_LABEL:
        survived, _ = receiver_surface_survival_state(label, surface)
        if survived is not None:
            surface[f"receiver_surface_{label}_survival"] = survived
    return surface


def finite_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def metric_value(metrics: Sequence[Mapping[str, Any]], *keys: str) -> float | None:
    for mapping in metrics:
        for key in keys:
            found = finite_metric(mapping.get(key))
            if found is not None:
                return found
    return None


def surface_value(surface: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        found = finite_metric(surface.get(key))
        if found is not None:
            return found
    return None


def surface_has_positive(surface: Mapping[str, Any], *keys: str) -> bool:
    return any((value := surface_value(surface, key)) is not None and value > 0.0 for key in keys)


def receiver_surface_visible(surface: Mapping[str, Any]) -> bool:
    return surface_has_positive(surface, *RECEIVER_SURFACE_RECEIVER_VISIBLE_KEYS)


def receiver_surface_receiver_visible(surface: Mapping[str, Any]) -> bool:
    return receiver_surface_visible(surface)


def receiver_surface_scorer_motion(surface: Mapping[str, Any]) -> bool:
    return surface_has_positive(surface, *RECEIVER_SURFACE_SCORER_VISIBLE_KEYS)


def receiver_surface_scorer_visible(surface: Mapping[str, Any]) -> bool:
    return receiver_surface_scorer_motion(surface)


def receiver_surface_uint8_contact(surface: Mapping[str, Any]) -> bool:
    return surface_has_positive(
        surface,
        *RECEIVER_SURFACE_UINT8_CHANGED_PIXELS_KEYS,
        *RECEIVER_SURFACE_UINT8_DELTA_ABS_MAX_KEYS,
    )


def receiver_surface_target_support_breakdown(
    surface: Mapping[str, Any],
) -> dict[str, float]:
    """Return target-support birth metrics using canonical key names."""

    out: dict[str, float] = {}
    for keys, canonical_key in (
        (RECEIVER_SURFACE_TARGET_HARD_WON_COUNT_KEYS, "receiver_surface_target_hard_won_count"),
        (RECEIVER_SURFACE_TARGET_HARD_LOST_COUNT_KEYS, "receiver_surface_target_hard_lost_count"),
        (RECEIVER_SURFACE_NET_TARGET_SUPPORT_DELTA_KEYS, "receiver_surface_net_target_support_delta"),
        (RECEIVER_SURFACE_WRONG_TO_TARGET_COUNT_KEYS, "receiver_surface_wrong_to_target_count"),
        (RECEIVER_SURFACE_TARGET_TO_WRONG_COUNT_KEYS, "receiver_surface_target_to_wrong_count"),
        (RECEIVER_SURFACE_WRONG_TO_WRONG_COUNT_KEYS, "receiver_surface_wrong_to_wrong_count"),
    ):
        found = surface_value(surface, *keys)
        if found is not None:
            out[canonical_key] = found
    hard_won = out.get("receiver_surface_target_hard_won_count")
    hard_lost = out.get("receiver_surface_target_hard_lost_count")
    if "receiver_surface_net_target_support_delta" not in out and hard_won is not None:
        out["receiver_surface_net_target_support_delta"] = hard_won - float(hard_lost or 0.0)
    return out


def receiver_surface_survival_state(
    label: str,
    *sources: Mapping[str, Any],
    blocker_prefix: str | None = None,
) -> tuple[bool | None, list[str]]:
    """Merge survival booleans across proposal/surface payloads."""

    keys = RECEIVER_SURFACE_SURVIVAL_KEYS_BY_LABEL[label]
    values: list[bool] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            if key in source and isinstance(source.get(key), bool):
                values.append(bool(source[key]))
    if not values:
        return None, []
    if False in values:
        blockers = [f"{blocker_prefix}_{label}_survival_conflict"] if blocker_prefix and True in values else []
        return False, blockers
    return True, []


def target_support_breakdown_present(metrics: Sequence[Mapping[str, Any]]) -> bool:
    return (
        metric_value(metrics, *RECEIVER_SURFACE_TARGET_HARD_WON_COUNT_KEYS) is not None
        or metric_value(metrics, *RECEIVER_SURFACE_NET_TARGET_SUPPORT_DELTA_KEYS) is not None
    )


def _scalar_ok(value: Any) -> bool:
    return isinstance(value, bool | int | str) or (isinstance(value, float) and math.isfinite(value))


__all__ = [
    "RECEIVER_SURFACE_ARGMAX_CHANGED_COUNT_KEYS",
    "RECEIVER_SURFACE_ARGMAX_FLIPPED_PIXELS_KEYS",
    "RECEIVER_SURFACE_CANONICAL_METRIC_ROWS",
    "RECEIVER_SURFACE_CONTRACT_SCHEMA",
    "RECEIVER_SURFACE_EVIDENCE_KEYS",
    "RECEIVER_SURFACE_FAKEQUANT_ARGMAX_FLIPPED_PIXELS_KEYS",
    "RECEIVER_SURFACE_FAKEQUANT_MARGIN_DELTA_KEYS",
    "RECEIVER_SURFACE_FAKEQUANT_NET_TARGET_SUPPORT_DELTA_KEYS",
    "RECEIVER_SURFACE_FAKEQUANT_POSE_OUTPUT_DELTA_KEYS",
    "RECEIVER_SURFACE_FAKEQUANT_SURVIVAL_KEYS",
    "RECEIVER_SURFACE_FAKEQUANT_TARGET_HARD_WON_COUNT_KEYS",
    "RECEIVER_SURFACE_FLOAT_RGB_DELTA_LINF_KEYS",
    "RECEIVER_SURFACE_INFLATED_ARGMAX_FLIPPED_PIXELS_KEYS",
    "RECEIVER_SURFACE_INFLATED_NET_TARGET_SUPPORT_DELTA_KEYS",
    "RECEIVER_SURFACE_INFLATED_POSE_OUTPUT_DELTA_KEYS",
    "RECEIVER_SURFACE_INFLATED_TARGET_HARD_WON_COUNT_KEYS",
    "RECEIVER_SURFACE_INFLATE_SURVIVAL_KEYS",
    "RECEIVER_SURFACE_LOSS_DELTA_KEYS",
    "RECEIVER_SURFACE_MARGIN_P50_DELTA_KEYS",
    "RECEIVER_SURFACE_NET_TARGET_SUPPORT_DELTA_KEYS",
    "RECEIVER_SURFACE_PARSEBACK_ARGMAX_FLIPPED_PIXELS_KEYS",
    "RECEIVER_SURFACE_PARSEBACK_MARGIN_DELTA_KEYS",
    "RECEIVER_SURFACE_PARSEBACK_NET_TARGET_SUPPORT_DELTA_KEYS",
    "RECEIVER_SURFACE_PARSEBACK_POSE_OUTPUT_DELTA_KEYS",
    "RECEIVER_SURFACE_PARSEBACK_SURVIVAL_KEYS",
    "RECEIVER_SURFACE_PARSEBACK_TARGET_HARD_WON_COUNT_KEYS",
    "RECEIVER_SURFACE_POSENET_INPUT_DELTA_LINF_KEYS",
    "RECEIVER_SURFACE_POSE_OUTPUT_DELTA_KEYS",
    "RECEIVER_SURFACE_RECEIVER_VISIBLE_KEYS",
    "RECEIVER_SURFACE_SCORER_VISIBLE_KEYS",
    "RECEIVER_SURFACE_SEGNET_INPUT_DELTA_LINF_KEYS",
    "RECEIVER_SURFACE_TARGET_HARD_LOST_COUNT_KEYS",
    "RECEIVER_SURFACE_TARGET_HARD_WON_COUNT_KEYS",
    "RECEIVER_SURFACE_TARGET_TO_WRONG_COUNT_KEYS",
    "RECEIVER_SURFACE_TRACE_METRIC_ROWS",
    "RECEIVER_SURFACE_UINT8_CHANGED_PIXELS_KEYS",
    "RECEIVER_SURFACE_UINT8_DELTA_ABS_MAX_KEYS",
    "RECEIVER_SURFACE_WRONG_TO_TARGET_COUNT_KEYS",
    "RECEIVER_SURFACE_WRONG_TO_WRONG_COUNT_KEYS",
    "finite_metric",
    "metric_value",
    "normalize_receiver_surface",
    "receiver_surface_receiver_visible",
    "receiver_surface_scorer_motion",
    "receiver_surface_scorer_visible",
    "receiver_surface_survival_state",
    "receiver_surface_target_support_breakdown",
    "receiver_surface_uint8_contact",
    "receiver_surface_visible",
    "surface_has_positive",
    "surface_value",
    "target_support_breakdown_present",
]
