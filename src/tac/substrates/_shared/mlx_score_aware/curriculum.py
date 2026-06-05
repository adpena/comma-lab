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
    segnet_direct_live: float | None = None,
) -> dict[str, float]:
    """Return canonical per-component stage weights for shared MLX training."""

    guard = float(scorer_input_guard)
    contrast = guard if scorer_input_contrast_floor is None else float(
        scorer_input_contrast_floor
    )
    shape = guard if scorer_input_shape_tether is None else float(
        scorer_input_shape_tether
    )
    direct_live = float(segnet) if segnet_direct_live is None else float(
        segnet_direct_live
    )
    values = {
        "recon": float(recon),
        "distill": float(segnet),
        "pose_distill": float(pose),
        "scorer_input_guard": guard,
        "scorer_input_contrast_floor": contrast,
        "scorer_input_shape_tether": shape,
        "segnet_direct_live_distill": direct_live,
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
        segnet_direct_live=(
            None
            if "segnet_direct_live_distill" not in raw
            else float(raw["segnet_direct_live_distill"])
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
) -> tuple[Any, ...]:
    """Build explicit MLX score-aware stages consumed by ``run_long_training``."""

    from tac.training.long_training_canonical import CurriculumStage

    total_epochs = int(epochs)
    pose_warmup = int(pose_distillation_warmup_epochs)
    shape_warmup = int(scorer_input_shape_warmup_epochs)
    escape_warmup = int(segnet_direct_live_escape_warmup_epochs)
    if pose_warmup < 0:
        raise ValueError("pose_distillation_warmup_epochs must be >= 0")
    if shape_warmup < 0:
        raise ValueError("scorer_input_shape_warmup_epochs must be >= 0")
    if escape_warmup < 0:
        raise ValueError("segnet_direct_live_escape_warmup_epochs must be >= 0")
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
                suppressed.append("pose")
            if shape_warmup > 0 and start_epoch < shape_warmup:
                stage_weights["segnet_direct_live_distill"] = 0.0
                suppressed.append("direct_live")
            if escape_warmup > 0 and start_epoch < escape_warmup:
                stage_weights["segnet_direct_live_base_loss"] = 0.0
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
                        "warmup independently suppresses PoseNet pressure."
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
