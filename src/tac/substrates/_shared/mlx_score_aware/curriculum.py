# SPDX-License-Identifier: MIT
"""Shared MLX score-aware curriculum controls.

This module is the common SNeRV/HiNeRV surface for stageable score terms.  The
actual training loop consumes :class:`CurriculumStage.loss_weights`; helpers
here keep runner CLI knobs, native substrate exports, and tests on the same
contract instead of letting each carrier invent private stage semantics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise
from typing import Any


def build_scoreaware_stage_loss_weights(
    *,
    recon: float = 1.0,
    segnet: float = 1.0,
    pose: float = 1.0,
    scorer_input_guard: float = 1.0,
    scorer_input_contrast_floor: float | None = None,
    scorer_input_shape_tether: float | None = None,
    posenet_yuv6_geometry_tether: float | None = None,
    posenet_temporal_signal_floor: float | None = None,
    segnet_direct_live: float | None = None,
    segnet_direct_live_class_histogram: float | None = None,
    segnet_direct_live_class_balanced_hinge: float | None = None,
    segnet_direct_live_class_balanced_ce: float | None = None,
    segnet_direct_live_class_balanced_squared_hinge: float | None = None,
    segnet_direct_live_class_region_recon: float | None = None,
    segnet_direct_live_rare_class_logit: float | None = None,
    segnet_direct_live_target_mass_floor: float | None = None,
    segnet_direct_live_target_min_ratio_floor: float | None = None,
    pose_direct_live: float | None = None,
) -> dict[str, float]:
    """Return canonical per-component stage weights for shared MLX training."""

    guard = float(scorer_input_guard)
    contrast = guard if scorer_input_contrast_floor is None else float(
        scorer_input_contrast_floor
    )
    shape = guard if scorer_input_shape_tether is None else float(
        scorer_input_shape_tether
    )
    pose_geometry = guard if posenet_yuv6_geometry_tether is None else float(
        posenet_yuv6_geometry_tether
    )
    temporal = guard if posenet_temporal_signal_floor is None else float(
        posenet_temporal_signal_floor
    )
    direct_live = float(segnet) if segnet_direct_live is None else float(
        segnet_direct_live
    )
    direct_live_histogram = (
        direct_live
        if segnet_direct_live_class_histogram is None
        else float(segnet_direct_live_class_histogram)
    )
    direct_live_balanced_hinge = (
        direct_live
        if segnet_direct_live_class_balanced_hinge is None
        else float(segnet_direct_live_class_balanced_hinge)
    )
    direct_live_balanced_ce = (
        direct_live
        if segnet_direct_live_class_balanced_ce is None
        else float(segnet_direct_live_class_balanced_ce)
    )
    direct_live_balanced_squared_hinge = (
        direct_live
        if segnet_direct_live_class_balanced_squared_hinge is None
        else float(segnet_direct_live_class_balanced_squared_hinge)
    )
    direct_live_region = (
        direct_live
        if segnet_direct_live_class_region_recon is None
        else float(segnet_direct_live_class_region_recon)
    )
    direct_live_rare = (
        direct_live
        if segnet_direct_live_rare_class_logit is None
        else float(segnet_direct_live_rare_class_logit)
    )
    direct_live_target_mass = (
        direct_live
        if segnet_direct_live_target_mass_floor is None
        else float(segnet_direct_live_target_mass_floor)
    )
    direct_live_target_min_ratio = (
        direct_live
        if segnet_direct_live_target_min_ratio_floor is None
        else float(segnet_direct_live_target_min_ratio_floor)
    )
    pose_live = float(pose) if pose_direct_live is None else float(pose_direct_live)
    values = {
        "recon": float(recon),
        "distill": float(segnet),
        "pose_distill": float(pose),
        "pose_direct_live_distill": pose_live,
        "scorer_input_guard": guard,
        "scorer_input_contrast_floor": contrast,
        "scorer_input_shape_tether": shape,
        "posenet_yuv6_geometry_tether": pose_geometry,
        "posenet_temporal_signal_floor": temporal,
        "segnet_direct_live_distill": direct_live,
        "segnet_direct_live_class_histogram": direct_live_histogram,
        "segnet_direct_live_class_balanced_hinge": direct_live_balanced_hinge,
        "segnet_direct_live_class_balanced_ce": direct_live_balanced_ce,
        "segnet_direct_live_class_balanced_squared_hinge": (
            direct_live_balanced_squared_hinge
        ),
        "segnet_direct_live_class_region_recon": direct_live_region,
        "segnet_direct_live_rare_class_logit": direct_live_rare,
        "segnet_direct_live_target_mass_floor": direct_live_target_mass,
        "segnet_direct_live_target_min_ratio_floor": direct_live_target_min_ratio,
    }
    bad = [
        name
        for name, value in values.items()
        if not math.isfinite(value) or value < 0.0
    ]
    if bad:
        raise ValueError(
            "score-aware stage loss weights must be finite and non-negative; "
            "invalid: " + ", ".join(bad)
        )
    if not any(value > 0.0 for value in values.values()):
        raise ValueError("at least one score-aware stage loss weight must be positive")
    return values


def coerce_scoreaware_stage_loss_weights(
    loss_weights: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Validate an optional canonical stage-weight mapping."""

    raw = dict(loss_weights or {})
    return build_scoreaware_stage_loss_weights(
        recon=float(raw.get("recon", 1.0)),
        segnet=float(raw.get("distill", raw.get("segnet", 1.0))),
        pose=float(raw.get("pose_distill", raw.get("pose", 1.0))),
        scorer_input_guard=float(
            raw.get("scorer_input_guard", raw.get("scorer_input_distribution_guard", 1.0))
        ),
        scorer_input_contrast_floor=(
            None
            if "scorer_input_contrast_floor" not in raw
            else float(raw["scorer_input_contrast_floor"])
        ),
        scorer_input_shape_tether=(
            None
            if "scorer_input_shape_tether" not in raw
            else float(raw["scorer_input_shape_tether"])
        ),
        posenet_yuv6_geometry_tether=(
            None
            if "posenet_yuv6_geometry_tether" not in raw
            else float(raw["posenet_yuv6_geometry_tether"])
        ),
        posenet_temporal_signal_floor=(
            None
            if "posenet_temporal_signal_floor" not in raw
            else float(raw["posenet_temporal_signal_floor"])
        ),
        segnet_direct_live=(
            None
            if "segnet_direct_live_distill" not in raw
            else float(raw["segnet_direct_live_distill"])
        ),
        segnet_direct_live_class_histogram=(
            None
            if "segnet_direct_live_class_histogram" not in raw
            else float(raw["segnet_direct_live_class_histogram"])
        ),
        segnet_direct_live_class_balanced_hinge=(
            None
            if "segnet_direct_live_class_balanced_hinge" not in raw
            else float(raw["segnet_direct_live_class_balanced_hinge"])
        ),
        segnet_direct_live_class_balanced_ce=(
            None
            if "segnet_direct_live_class_balanced_ce" not in raw
            else float(raw["segnet_direct_live_class_balanced_ce"])
        ),
        segnet_direct_live_class_balanced_squared_hinge=(
            None
            if "segnet_direct_live_class_balanced_squared_hinge" not in raw
            else float(raw["segnet_direct_live_class_balanced_squared_hinge"])
        ),
        segnet_direct_live_class_region_recon=(
            None
            if "segnet_direct_live_class_region_recon" not in raw
            else float(raw["segnet_direct_live_class_region_recon"])
        ),
        segnet_direct_live_rare_class_logit=(
            None
            if "segnet_direct_live_rare_class_logit" not in raw
            else float(raw["segnet_direct_live_rare_class_logit"])
        ),
        segnet_direct_live_target_mass_floor=(
            None
            if "segnet_direct_live_target_mass_floor" not in raw
            else float(raw["segnet_direct_live_target_mass_floor"])
        ),
        segnet_direct_live_target_min_ratio_floor=(
            None
            if "segnet_direct_live_target_min_ratio_floor" not in raw
            else float(raw["segnet_direct_live_target_min_ratio_floor"])
        ),
        pose_direct_live=(
            None
            if "pose_direct_live_distill" not in raw
            else float(raw["pose_direct_live_distill"])
        ),
    )


def build_scoreaware_curriculum_stages(
    *,
    substrate_id: str,
    epochs: int,
    loss_weights: Mapping[str, float],
    pose_distillation_warmup_epochs: int = 0,
    scorer_input_shape_warmup_epochs: int = 0,
    segnet_direct_live_escape_warmup_epochs: int = 0,
    segnet_direct_live_escape_class_multiplier: float = 1.0,
) -> tuple[Any, ...]:
    """Build explicit MLX score-aware stages consumed by ``run_long_training``."""

    from tac.training.long_training_canonical import CurriculumStage

    total_epochs = int(epochs)
    pose_warmup = int(pose_distillation_warmup_epochs)
    shape_warmup = int(scorer_input_shape_warmup_epochs)
    escape_warmup = int(segnet_direct_live_escape_warmup_epochs)
    escape_class_multiplier = float(segnet_direct_live_escape_class_multiplier)
    if pose_warmup < 0:
        raise ValueError("pose_distillation_warmup_epochs must be >= 0")
    if shape_warmup < 0:
        raise ValueError("scorer_input_shape_warmup_epochs must be >= 0")
    if escape_warmup < 0:
        raise ValueError("segnet_direct_live_escape_warmup_epochs must be >= 0")
    if not math.isfinite(escape_class_multiplier) or escape_class_multiplier <= 0.0:
        raise ValueError(
            "segnet_direct_live_escape_class_multiplier must be finite and > 0"
        )
    warmup_fields = {
        "pose_distillation_warmup_epochs": pose_warmup,
        "scorer_input_shape_warmup_epochs": shape_warmup,
        "segnet_direct_live_escape_warmup_epochs": escape_warmup,
    }
    for field, warmup_epochs in warmup_fields.items():
        if warmup_epochs <= 0:
            continue
        if total_epochs <= 1:
            raise ValueError(f"{field} requires epochs > 1")
        if warmup_epochs >= total_epochs:
            raise ValueError(f"{field} must be smaller than epochs")

    if pose_warmup > 0 or shape_warmup > 0 or escape_warmup > 0:
        boundaries = sorted(
            {
                0,
                total_epochs,
                *({pose_warmup} if pose_warmup > 0 else set()),
                *({shape_warmup} if shape_warmup > 0 else set()),
                *({escape_warmup} if escape_warmup > 0 else set()),
            }
        )
        stages = []
        for start_epoch, end_epoch in pairwise(boundaries):
            if start_epoch == end_epoch:
                continue
            stage_weights = dict(loss_weights)
            suppressed: list[str] = []
            if pose_warmup > 0 and start_epoch < pose_warmup:
                stage_weights["pose_distill"] = 0.0
                stage_weights["pose_direct_live_distill"] = 0.0
                suppressed.append("pose")
            if shape_warmup > 0 and start_epoch < shape_warmup:
                stage_weights["segnet_direct_live_distill"] = 0.0
                stage_weights["segnet_direct_live_class_histogram"] = 0.0
                stage_weights["segnet_direct_live_class_balanced_hinge"] = 0.0
                stage_weights["segnet_direct_live_class_balanced_ce"] = 0.0
                stage_weights["segnet_direct_live_class_balanced_squared_hinge"] = 0.0
                stage_weights["segnet_direct_live_class_region_recon"] = 0.0
                stage_weights["segnet_direct_live_rare_class_logit"] = 0.0
                stage_weights["segnet_direct_live_target_mass_floor"] = 0.0
                stage_weights["segnet_direct_live_target_min_ratio_floor"] = 0.0
                suppressed.append("direct_live")
            if escape_warmup > 0 and start_epoch < escape_warmup:
                stage_weights["segnet_direct_live_base_loss"] = 0.0
                for key in (
                    "segnet_direct_live_class_histogram",
                    "segnet_direct_live_class_balanced_hinge",
                    "segnet_direct_live_class_balanced_ce",
                    "segnet_direct_live_class_balanced_squared_hinge",
                    "segnet_direct_live_class_region_recon",
                    "segnet_direct_live_rare_class_logit",
                    "segnet_direct_live_target_mass_floor",
                    "segnet_direct_live_target_min_ratio_floor",
                ):
                    stage_weights[key] = (
                        float(stage_weights.get(key, 1.0))
                        * escape_class_multiplier
                    )
                suppressed.append("direct_live_base")
            suffix = "full" if not suppressed else "warmup_" + "_".join(suppressed)
            stages.append(
                CurriculumStage(
                    name=f"{substrate_id}_mlx_score_aware_{suffix}",
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    loss_weights=stage_weights,
                    notes=(
                        "Explicit scorer curriculum. Shape warmup suppresses "
                        "direct-live SegNet pressure so scorer-input geometry can "
                        "stabilize before class-escape pressure; direct-live "
                        "escape warmup suppresses only the base logit-matching "
                        "term while preserving class-balanced escape terms; Pose "
                        "warmup independently suppresses cached and direct-live "
                        "PoseNet pressure."
                    ),
                )
            )
        return tuple(stages)

    return (
        CurriculumStage(
            name=f"{substrate_id}_mlx_score_aware_weighted_full",
            start_epoch=0,
            end_epoch=total_epochs,
            loss_weights=dict(loss_weights),
            notes=(
                "Explicit score-aware component weights consumed directly by "
                "the shared MLX adapter train step."
            ),
        ),
    )
