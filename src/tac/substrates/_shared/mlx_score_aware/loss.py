# SPDX-License-Identifier: MIT
"""Gradient-reachable MLX score-aware Lagrangian (separation of concerns).

This module owns ONLY the loss math: reconstruction MSE + the optional
gradient-reachable scorer-distilled surrogate + optional
substrate-specific extra terms. It is substrate-AGNOSTIC: the renderer forward
convention is decoded via :func:`decode_frames_nhwc01` so the loss never
assumes a fixed model signature.

The score-aware term is the canonical Hinton-distilled surrogate per
CLAUDE.md "eval_roundtrip -- NON-NEGOTIABLE" + Catalog #164 sister discipline:
the production teacher is the real MLX SegNet logits cache on the contest
SegNet frame (default pair frame 1, matching upstream ``x[:, -1, ...]``), the
student is a learnable head on the decoded frame, and gradient flows KL ->
decoded -> renderer params. The explicit mock path is allowed only for
scorer-blind smoke tests.

[verified-against: tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss.score_teacher_distillation_loss canonical scorer surrogate]
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from tac.substrates._shared.mlx_score_aware.device_gate import (
    require_mlx_for_harness,
)

if TYPE_CHECKING:
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle


_CORE_LOSS_WEIGHT_ALIASES: dict[str, tuple[str, ...]] = {
    "recon": ("recon", "reconstruction"),
    "distill": ("distill", "segnet_distill", "segnet"),
    "pose_distill": ("pose_distill", "posenet_distill", "pose"),
    "pose_direct_live_distill": (
        "pose_direct_live_distill",
        "posenet_direct_live",
        "direct_live_pose",
    ),
    "scorer_input_guard": (
        "scorer_input_guard",
        "scorer_input_distribution_guard",
        "value_domain_guard",
    ),
    "scorer_input_contrast_floor": (
        "scorer_input_contrast_floor",
        "contrast_floor",
    ),
    "scorer_input_shape_tether": (
        "scorer_input_shape_tether",
        "shape_tether",
    ),
    "posenet_yuv6_geometry_tether": (
        "posenet_yuv6_geometry_tether",
        "posenet_yuv6_geometry",
        "posenet_pair_geometry_tether",
        "posenet_yuv6_pair_geometry",
    ),
    "posenet_temporal_signal_floor": (
        "posenet_temporal_signal_floor",
        "posenet_temporal_delta_floor",
        "temporal_signal_floor",
    ),
    "segnet_direct_live_distill": (
        "segnet_direct_live_distill",
        "segnet_direct_live",
        "direct_live_segnet",
    ),
    "segnet_direct_live_base_loss": (
        "segnet_direct_live_base_loss",
        "segnet_direct_live_base",
    ),
    "segnet_direct_live_class_histogram": (
        "segnet_direct_live_class_histogram",
        "segnet_direct_live_histogram",
    ),
    "segnet_direct_live_class_balanced_hinge": (
        "segnet_direct_live_class_balanced_hinge",
        "segnet_direct_live_balanced_hinge",
    ),
    "segnet_direct_live_class_balanced_ce": (
        "segnet_direct_live_class_balanced_ce",
        "segnet_direct_live_balanced_ce",
    ),
    "segnet_direct_live_class_balanced_squared_hinge": (
        "segnet_direct_live_class_balanced_squared_hinge",
        "segnet_direct_live_balanced_squared_hinge",
    ),
    "segnet_direct_live_class_region_recon": (
        "segnet_direct_live_class_region_recon",
        "segnet_direct_live_region_recon",
        "segnet_direct_live_target_region_recon",
    ),
    "segnet_direct_live_rare_class_logit": (
        "segnet_direct_live_rare_class_logit",
        "segnet_direct_live_missing_class_logit",
        "segnet_direct_live_any_target_class_logit",
    ),
    "segnet_direct_live_target_mass_floor": (
        "segnet_direct_live_target_mass_floor",
        "segnet_direct_live_target_class_mass_floor",
        "segnet_direct_live_min_ratio_mass_floor",
    ),
    "segnet_direct_live_target_min_ratio_floor": (
        "segnet_direct_live_target_min_ratio_floor",
        "segnet_direct_live_target_class_min_ratio_floor",
        "segnet_direct_live_min_ratio_floor",
        "segnet_direct_live_hard_support_floor",
    ),
}
_CORE_LOSS_WEIGHT_KEYS = frozenset(
    key for aliases in _CORE_LOSS_WEIGHT_ALIASES.values() for key in aliases
)
_SEGNET_OCCUPANCY_MIN_CLASS_FRACTION = 1.0e-3
_SEGNET_OCCUPANCY_MIN_CLASS_PIXELS = 2


def _segnet_occupancy_min_fraction(pixel_count: int) -> tuple[float, int]:
    """Return the class-mass floor for nondegenerate SegNet occupancy."""

    count = max(
        _SEGNET_OCCUPANCY_MIN_CLASS_PIXELS,
        math.ceil(max(int(pixel_count), 0) * _SEGNET_OCCUPANCY_MIN_CLASS_FRACTION),
    )
    if pixel_count <= 0:
        return 1.0, count
    return float(count / pixel_count), count


def _segnet_target_argmax_from_logits_or_exact(
    target_logits: Any,
    *,
    target_argmax: Any | None,
) -> Any:
    """Return exact d_seg target labels when the teacher preserved them."""

    mx = require_mlx_for_harness()
    if target_argmax is None:
        return mx.argmax(target_logits, axis=-1)
    if tuple(target_argmax.shape) != tuple(target_logits.shape[:-1]):
        raise ValueError(
            "target_argmax must match target logits without the class axis; "
            f"target_argmax={tuple(target_argmax.shape)} "
            f"target_logits={tuple(target_logits.shape)}"
        )
    return target_argmax.astype(mx.int32)


def _exact_segnet_target_argmax_for_indices(
    scorer_teacher: Any,
    idx: Any,
    target_logits: Any,
) -> Any:
    """Use provider-preserved hard labels, falling back to logits argmax."""

    mx = require_mlx_for_harness()
    fn = getattr(scorer_teacher, "teacher_argmax_for_indices", None)
    if not callable(fn):
        return mx.argmax(target_logits, axis=-1)
    return _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=mx.stop_gradient(fn(idx)),
    )


def component_loss_weight(
    loss_weights: Mapping[str, float] | None,
    component: str,
    *,
    default: float = 1.0,
) -> float:
    """Return the stage weight for a core scorer-loss component.

    ``CurriculumStage.loss_weights`` is the canonical staging surface in the
    long-training harness. Core components use stable component names, with a
    few aliases for older NeRV planning surfaces. Missing keys preserve legacy
    behavior by returning ``default``.
    """

    if not loss_weights:
        return float(default)
    aliases = _CORE_LOSS_WEIGHT_ALIASES.get(component, (component,))
    for key in aliases:
        if key in loss_weights:
            return float(loss_weights[key])
    return float(default)


def component_config_weight_with_floor(
    loss_weights: Mapping[str, float] | None,
    component: str,
    configured_weight: float,
) -> float:
    """Return the static config weight lifted by a train-time config floor."""

    configured = float(configured_weight)
    floor = component_loss_weight(
        loss_weights,
        f"{component}_config_floor",
        default=0.0,
    )
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError(
            f"{component}_config_floor must be finite and non-negative; got {floor!r}"
        )
    return max(configured, floor)


_SEGNET_DIRECT_LIVE_SUBCONTROL_COMPONENTS: tuple[tuple[str, str], ...] = (
    ("segnet_direct_live_class_histogram", "segnet_direct_live_class_histogram_weight"),
    (
        "segnet_direct_live_class_balanced_hinge",
        "segnet_direct_live_class_balanced_hinge_weight",
    ),
    (
        "segnet_direct_live_class_balanced_ce",
        "segnet_direct_live_class_balanced_ce_weight",
    ),
    (
        "segnet_direct_live_class_balanced_squared_hinge",
        "segnet_direct_live_class_balanced_squared_hinge_weight",
    ),
    (
        "segnet_direct_live_class_region_recon",
        "segnet_direct_live_class_region_recon_weight",
    ),
    ("segnet_direct_live_rare_class_logit", "segnet_direct_live_rare_class_logit_weight"),
    (
        "segnet_direct_live_target_mass_floor",
        "segnet_direct_live_target_mass_floor_weight",
    ),
    (
        "segnet_direct_live_target_min_ratio_floor",
        "segnet_direct_live_target_min_ratio_floor_weight",
    ),
)


def _segnet_direct_live_subcontrol_active(
    bundle: RendererBundle,
    loss_weights: Mapping[str, float] | None,
) -> bool:
    """Return whether any direct-live SegNet subcontrol has non-zero pressure."""

    for component, attr in _SEGNET_DIRECT_LIVE_SUBCONTROL_COMPONENTS:
        stage_weight = component_loss_weight(loss_weights, component)
        config_weight = component_config_weight_with_floor(
            loss_weights,
            component,
            float(getattr(bundle, attr)),
        )
        if config_weight * stage_weight > 0.0:
            return True
    return False


def _extra_loss_weight_overrides(
    loss_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if not loss_weights:
        return {}
    return {
        str(key): float(value)
        for key, value in loss_weights.items()
        if str(key) not in _CORE_LOSS_WEIGHT_KEYS
    }


def source_pair_indices_for_local_batch(bundle: RendererBundle, idx: Any) -> Any:
    """Map local hydrated target rows to source-video model pair indices.

    Most harness users train on a prefix, so local target rows and source model
    rows are identical. Hard-pair training is different: the targets may be a
    compact local tensor such as rows ``[0, 1]`` while the full-video renderer
    must decode source temporal rows such as ``[417, 22]``. This helper keeps
    that distinction explicit and MLX-native.
    """

    if bundle.source_pair_indices is None:
        return idx
    mx = require_mlx_for_harness()
    source_rows = mx.array(bundle.source_pair_indices, dtype=mx.int32)
    return mx.take(source_rows, idx, axis=0)


def decode_frames_nhwc01(bundle: RendererBundle, idx: Any) -> tuple[Any, Any]:
    """Decode ``(rgb_0, rgb_1)`` as NHWC ``[0, 1]`` regardless of model convention.

    Returns two MLX float32 arrays each ``(B, H, W, 3)`` in ``[0, 1]``, ready
    for MSE against the canonical NHWC ``[0, 1]`` targets.

    Args:
        bundle: the substrate RendererBundle.
        idx: MLX int32 ``(B,)`` pair-index batch.

    Returns:
        ``(rgb_0, rgb_1)`` NHWC float32 each ``(B, H, W, 3)`` in ``[0, 1]``.
    """
    mx = require_mlx_for_harness()
    model = bundle.model
    model_idx = source_pair_indices_for_local_batch(bundle, idx)
    if bundle.forward_convention == "reconstruct_pair_nchw01":
        result = model.reconstruct_pair(model_idx)
        # The renderer may return (rgb_0, rgb_1) or (rgb_0, rgb_1, z); take the
        # first two. Each is (B, 3, H, W) in [0, 1].
        rgb_0 = result[0]
        rgb_1 = result[1]
        rgb_0 = mx.transpose(rgb_0, (0, 2, 3, 1))
        rgb_1 = mx.transpose(rgb_1, (0, 2, 3, 1))
        return rgb_0, rgb_1
    # call_b2chw_255: model(idx) -> (B, 2, 3, H, W) in [0, 255].
    pair = model(model_idx)
    pair01 = pair / 255.0
    rgb_0 = mx.transpose(pair01[:, 0], (0, 2, 3, 1))
    rgb_1 = mx.transpose(pair01[:, 1], (0, 2, 3, 1))
    return rgb_0, rgb_1


def _apply_eval_roundtrip_ste_nhwc01(bundle: RendererBundle, rgb: Any) -> Any:
    """Apply the PR95 byte-realized scorer surface to NHWC ``[0, 1]`` frames."""
    if not bundle.eval_roundtrip_ste_enabled:
        return rgb
    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_eval_roundtrip_nhwc,
    )

    roundtripped = apply_eval_roundtrip_nhwc(
        rgb * 255.0,
        camera_hw=tuple(bundle.eval_roundtrip_camera_hw),
        output_hw=(int(rgb.shape[1]), int(rgb.shape[2])),
        simulate_resize=True,
        simulate_uint8=True,
        ste_round=True,
    )
    return roundtripped / 255.0


def pose_student_inputs_nhwc(bundle: RendererBundle, rgb_0: Any, rgb_1: Any) -> tuple[Any, Any]:
    """Return pose-student inputs on the configured differentiable surface."""
    if bundle.pose_student_input_preprocess == "rgb":
        return rgb_0, rgb_1
    if bundle.pose_student_input_preprocess == "pr95_yuv6":
        from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

        return (
            rgb_to_yuv6_mlx(rgb_0 * 255.0),
            rgb_to_yuv6_mlx(rgb_1 * 255.0),
        )
    raise ValueError(
        "unsupported pose_student_input_preprocess "
        f"{bundle.pose_student_input_preprocess!r}; RendererBundle should have "
        "validated this."
    )


def posenet_yuv6_pair_nhwc255(bundle: RendererBundle, rgb_0: Any, rgb_1: Any) -> Any:
    """Return upstream PoseNet's direct-live YUV6 pair surface for candidates.

    The contest PoseNet consumes both frames after RGB->YUV6 and 2x spatial
    downsample, with the two 6-channel tensors concatenated to 12 channels.
    Candidate frames enter this helper as NHWC RGB in ``[0, 1]`` at the
    canonical scorer size ``384x512`` and leave as NHWC YUV6 byte-scale
    ``(B, 192, 256, 12)``. This is the live-score VJP surface; it is distinct
    from the lightweight pose-student input helper above.
    """

    del bundle
    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    return require_mlx_for_harness().concatenate(
        [rgb_to_yuv6_mlx(rgb_0 * 255.0), rgb_to_yuv6_mlx(rgb_1 * 255.0)],
        axis=-1,
    )


def _prepare_recon_pixel_weight(
    bundle: RendererBundle,
    frame_shape: Any,
    *,
    idx: Any | None = None,
    frame_index: int | None = None,
) -> Any:
    """Coerce + validate the bundle's ``recon_pixel_weight`` map for broadcasting.

    Accepts an MLX float32 static spatial map shaped ``(H, W)`` /
    ``(H, W, 1)`` / ``(H, W, C)`` (C in {1, 3}) / ``(1, H, W, C)``, a per-pair
    map shaped ``(N, H, W, C)``, or a per-pair/per-frame map shaped
    ``(N, 2, H, W, C)``. Static maps return broadcastable shape
    ``(1, H, W, C)``; indexed maps return ``(B, H, W, C)`` for the active
    batch. C may be 1 or 3.
    The map is gradient-blocked by the caller; this helper only normalizes shape
    and validates the spatial dims match + the map is non-negative + finite.

    Args:
        bundle: the harness RendererBundle carrying ``recon_pixel_weight``.
        frame_shape: the decoded frame's ``(B, H, W, 3)`` shape tuple.
        idx: optional pair indices for dynamic ``(N,...)`` maps.
        frame_index: required for ``(N,2,H,W,C)`` maps; 0 for frame_0, 1 for
            frame_1.

    Returns:
        an MLX float32 array broadcastable against ``(B, H, W, 3)``.

    Raises:
        MlxScoreAwareHarnessError: on spatial mismatch / negative / non-finite /
            unsupported ndim.
    """
    mx = require_mlx_for_harness()
    from tac.substrates._shared.mlx_score_aware.device_gate import (
        MlxScoreAwareHarnessError,
    )

    b, h, w, _c = frame_shape
    w_arr = bundle.recon_pixel_weight
    provider = getattr(w_arr, "recon_pixel_weight_for_batch", None)
    if callable(provider):
        if idx is None or frame_index not in (0, 1):
            raise MlxScoreAwareHarnessError(
                "recon_pixel_weight provider requires pair indices and "
                "frame_index 0 or 1"
            )
        w_arr = mx.array(
            provider(idx=idx, frame_shape=frame_shape, frame_index=int(frame_index))
        ).astype(mx.float32)
        if w_arr.ndim != 4:
            raise MlxScoreAwareHarnessError(
                "recon_pixel_weight provider must return (B,H,W,C); got "
                f"ndim={w_arr.ndim} shape={tuple(w_arr.shape)}"
            )
        lb, wh, ww, wc = w_arr.shape
        if lb not in (1, b) or (wh, ww) != (h, w) or wc not in (1, 3):
            raise MlxScoreAwareHarnessError(
                "recon_pixel_weight provider must return "
                f"(1|B,{h},{w},1|3); got {tuple(w_arr.shape)} for B={b}"
            )
        if float(mx.min(w_arr)) < 0.0:
            raise MlxScoreAwareHarnessError(
                "recon_pixel_weight must be non-negative (it is a per-pixel weight)."
            )
        if not bool(mx.all(mx.isfinite(w_arr))):
            raise MlxScoreAwareHarnessError("recon_pixel_weight must be finite.")
        if float(mx.mean(w_arr)) <= 0.0:
            raise MlxScoreAwareHarnessError(
                "recon_pixel_weight must have positive total mass; all-zero maps "
                "silently disable reconstruction instead of re-weighting it."
            )
        return w_arr
    # Coerce numpy / list -> MLX float32 (MLX arrays pass through as float32).
    w_arr = mx.array(w_arr).astype(mx.float32)
    nd = w_arr.ndim
    if nd == 2:  # (H, W) -> (1, H, W, 1)
        if w_arr.shape != (h, w):
            raise MlxScoreAwareHarnessError(
                f"recon_pixel_weight (H,W) must match decoded frame ({h},{w}); got {tuple(w_arr.shape)}"
            )
        w_arr = w_arr.reshape(1, h, w, 1)
    elif nd == 3:  # (H, W, C) -> (1, H, W, C) with C in {1, 3}
        wh, ww, wc = w_arr.shape
        if (wh, ww) != (h, w) or wc not in (1, 3):
            raise MlxScoreAwareHarnessError(
                f"recon_pixel_weight (H,W,C) must be ({h},{w},1) or ({h},{w},3); got {tuple(w_arr.shape)}"
            )
        w_arr = w_arr.reshape(1, h, w, wc)
    elif nd == 4:  # (1, H, W, C) static OR (N, H, W, C) per-pair.
        lb, wh, ww, wc = w_arr.shape
        if (wh, ww) != (h, w) or wc not in (1, 3):
            raise MlxScoreAwareHarnessError(
                f"recon_pixel_weight (B,H,W,C) must be (1|N,{h},{w},1|3); got {tuple(w_arr.shape)}"
            )
        if lb != 1:
            if idx is None:
                raise MlxScoreAwareHarnessError(
                    "per-pair recon_pixel_weight (N,H,W,C) requires pair indices"
                )
            w_arr = w_arr[idx]
    elif nd == 5:  # (N, 2, H, W, C) -> select active frame and pair batch.
        n, frames, wh, ww, wc = w_arr.shape
        if frames != 2 or (wh, ww) != (h, w) or wc not in (1, 3):
            raise MlxScoreAwareHarnessError(
                f"recon_pixel_weight (N,2,H,W,C) must be (N,2,{h},{w},1|3); got {tuple(w_arr.shape)}"
            )
        if idx is None or frame_index not in (0, 1):
            raise MlxScoreAwareHarnessError(
                "per-pair/per-frame recon_pixel_weight requires pair indices "
                "and frame_index 0 or 1"
            )
        w_arr = w_arr[idx, int(frame_index)]
    else:
        raise MlxScoreAwareHarnessError(
            "recon_pixel_weight must be (H,W) / (H,W,C) / (1|N,H,W,C) / "
            f"(N,2,H,W,C); got ndim={nd} shape={tuple(w_arr.shape)}"
        )
    # Non-negative + finite invariants (the map is a re-weight, not a mask).
    if float(mx.min(w_arr)) < 0.0:
        raise MlxScoreAwareHarnessError("recon_pixel_weight must be non-negative (it is a per-pixel weight).")
    if not bool(mx.all(mx.isfinite(w_arr))):
        raise MlxScoreAwareHarnessError("recon_pixel_weight must be finite.")
    if float(mx.mean(w_arr)) <= 0.0:
        raise MlxScoreAwareHarnessError(
            "recon_pixel_weight must have positive total mass; all-zero maps "
            "silently disable reconstruction instead of re-weighting it."
        )
    return w_arr


def _weighted_recon(bundle: RendererBundle, rgb: Any, gt: Any, weight: Any) -> Any:
    """Per-pixel weighted reconstruction MSE for one frame.

    ``mean(w * (rgb - gt)^2)`` with the canonical ``"mean"`` normalization
    dividing by ``mean(w)`` so the weighted loss is a convex re-distribution of
    the SAME total magnitude as the uniform ``mean((rgb - gt)^2)`` (keeps the
    ``recon_weight`` Lagrangian coefficient comparable across A/B arms). When
    ``recon_pixel_weight_normalize == "none"`` the raw map is applied.
    """
    mx = require_mlx_for_harness()
    sq = (rgb - gt) ** 2  # (B, H, W, 3)
    weighted = mx.mean(weight * sq)
    if bundle.recon_pixel_weight_normalize == "mean":
        weighted = weighted / (mx.mean(weight) + 1e-12)
    return weighted


def _pose_distillation_loss_and_raw_mse(
    bundle: RendererBundle,
    *,
    student_pose: Any,
    teacher_pose: Any,
    per_dim_scale: Any = None,
) -> tuple[Any, Any]:
    """Return the train-time pose loss plus exact raw MSE telemetry.

    ``pose_distillation_loss == "mse"`` returns the configured train loss:
    Mahalanobis/std-scaled when ``per_dim_scale`` is supplied, raw otherwise.
    ``"huber"`` uses an MSE-matched Huber on that same train-space diff:
    ``diff**2`` inside the delta and ``2*delta*abs(diff)-delta**2`` outside,
    so small-error curvature matches MSE while large-error gradients are
    bounded.

    The second returned value is always the unscaled upstream-contest
    ``d_pose`` proxy: raw MSE over the scored pose dims. Do not use the scaled
    train loss as ``sqrt(10*d_pose)``; that confuses optimizer conditioning with
    the contest Lagrangian and can overdrive the pose branch by orders of
    magnitude.
    """
    mx = require_mlx_for_harness()
    raw_diff = student_pose - teacher_pose
    raw_mse = mx.mean(raw_diff * raw_diff)
    diff = raw_diff
    if per_dim_scale is not None:
        diff = diff / mx.maximum(per_dim_scale, 1.0e-12)
    train_mse = mx.mean(diff * diff)
    if bundle.pose_distillation_loss == "mse":
        return train_mse, raw_mse
    delta = float(bundle.pose_distillation_huber_delta)
    abs_diff = mx.abs(diff)
    quadratic = diff * diff
    linear = 2.0 * delta * abs_diff - delta * delta
    return mx.mean(mx.where(abs_diff <= delta, quadratic, linear)), raw_mse


def _direct_live_posenet_distillation_loss_and_metrics(
    bundle: RendererBundle,
    rgb_0: Any,
    rgb_1: Any,
    idx: Any,
) -> tuple[Any, dict[str, Any]]:
    """Direct-live PoseNet VJP term on the exact upstream YUV6 pair surface."""

    mx = require_mlx_for_harness()
    if bundle.pose_scorer_teacher is None:
        raise ValueError(
            "pose_direct_live_distillation_weight > 0 without pose_scorer_teacher; "
            "RendererBundle.__post_init__ should have rejected this."
        )
    live_pose_fn = getattr(
        bundle.pose_scorer_teacher,
        "teacher_pose_for_yuv6_pair_nhwc",
        None,
    )
    if not callable(live_pose_fn):
        raise ValueError(
            "pose_direct_live_distillation_weight > 0 requires a PoseNet teacher "
            "with teacher_pose_for_yuv6_pair_nhwc(candidate_yuv6_pair)."
        )
    yuv6_pair = posenet_yuv6_pair_nhwc255(bundle, rgb_0, rgb_1)
    candidate_pose = live_pose_fn(yuv6_pair)
    teacher_pose = mx.stop_gradient(
        bundle.pose_scorer_teacher.teacher_pose_for_indices(idx)
    )
    raw_diff = candidate_pose - teacher_pose
    raw_mse = mx.mean(raw_diff * raw_diff)
    score_term = mx.sqrt(10.0 * raw_mse + 1.0e-12)
    metrics = {
        "pose_direct_live_distill": score_term,
        "pose_direct_live_raw_mse": raw_mse,
        "pose_direct_live_score_term": score_term,
        "pose_direct_live_abs_mean": mx.mean(mx.abs(raw_diff)),
        "pose_direct_live_candidate_pose_mean": mx.mean(candidate_pose),
        "pose_direct_live_candidate_pose_std": mx.std(candidate_pose),
        "pose_direct_live_target_pose_mean": mx.mean(teacher_pose),
        "pose_direct_live_target_pose_std": mx.std(teacher_pose),
        "pose_direct_live_yuv6_pair_mean": mx.mean(yuv6_pair),
        "pose_direct_live_yuv6_pair_std": mx.std(yuv6_pair),
        "pose_direct_live_yuv6_pair_temporal_delta_std": mx.std(
            yuv6_pair[..., 6:12] - yuv6_pair[..., 0:6]
        ),
    }
    return score_term, metrics


def _value_domain_distribution_guard_parts(
    candidate: Any,
    reference: Any,
) -> dict[str, Any]:
    """Match per-channel value-domain mean, std, and dynamic range.

    ``candidate`` and ``reference`` are channels-last scorer-domain tensors
    shaped ``(B,H,W,C)`` on the same numeric scale.  The reference path is
    gradient-blocked; this is a distribution tether, not a hidden target
    renderer.
    """

    mx = require_mlx_for_harness()

    cand_mean = mx.mean(candidate, axis=(1, 2))
    ref_mean = mx.stop_gradient(mx.mean(reference, axis=(1, 2)))
    mean_loss = mx.mean((cand_mean - ref_mean) ** 2)

    cand_centered = candidate - mx.mean(candidate, axis=(1, 2), keepdims=True)
    ref_centered = reference - mx.mean(reference, axis=(1, 2), keepdims=True)
    cand_std = mx.sqrt(mx.mean(cand_centered * cand_centered, axis=(1, 2)) + 1.0e-12)
    ref_std = mx.stop_gradient(
        mx.sqrt(mx.mean(ref_centered * ref_centered, axis=(1, 2)) + 1.0e-12)
    )
    std_loss = mx.mean((cand_std - ref_std) ** 2)

    cand_range = mx.max(candidate, axis=(1, 2)) - mx.min(candidate, axis=(1, 2))
    ref_range = mx.stop_gradient(
        mx.max(reference, axis=(1, 2)) - mx.min(reference, axis=(1, 2))
    )
    dynamic_range_loss = mx.mean((cand_range - ref_range) ** 2)
    total = mean_loss + std_loss + dynamic_range_loss
    return {
        "total": total,
        "mean": mean_loss,
        "std": std_loss,
        "dynamic_range": dynamic_range_loss,
    }


def _soft_saturation_guard_parts(
    bundle: RendererBundle,
    candidate_rgb01: Any,
    reference_rgb01: Any,
) -> dict[str, Any]:
    """Match near-0/near-1 RGB saturation mass on normalized RGB tensors."""

    mx = require_mlx_for_harness()
    margin = float(bundle.scorer_input_distribution_guard_saturation_margin)
    temperature = float(bundle.scorer_input_distribution_guard_temperature)

    cand_soft_sat = mx.mean(
        mx.sigmoid((margin - candidate_rgb01) / temperature)
        + mx.sigmoid((candidate_rgb01 - (1.0 - margin)) / temperature),
        axis=(1, 2),
    )
    ref_soft_sat = mx.stop_gradient(
        mx.mean(
            mx.sigmoid((margin - reference_rgb01) / temperature)
            + mx.sigmoid((reference_rgb01 - (1.0 - margin)) / temperature),
            axis=(1, 2),
        )
    )
    saturation_loss = mx.mean((cand_soft_sat - ref_soft_sat) ** 2)
    return {"total": saturation_loss, "soft_saturation": saturation_loss}


def _scorer_input_fit_guard_parts(
    candidate: Any,
    reference: Any,
) -> dict[str, Any]:
    """Direct scorer-input fit on the exact tensor consumed downstream."""

    mx = require_mlx_for_harness()
    diff = candidate - mx.stop_gradient(reference)
    mse = mx.mean(diff * diff)
    mae = mx.mean(mx.abs(diff))
    return {"total": mse + mae, "mse": mse, "mae": mae}


def _spatial_gradient_guard_parts(
    candidate: Any,
    reference: Any,
) -> dict[str, Any]:
    """Match scorer-domain local contrast with dense gradients.

    Global dynamic range is too sparse a training signal for collapsed NeRV
    renderers: only the current extrema get range gradients. The contest
    networks consume downsampled spatial structure, so match first differences
    on the same scorer-domain tensor instead of adding a perceptual image loss.
    """

    mx = require_mlx_for_harness()
    ref = mx.stop_gradient(reference)
    cand_dx = candidate[:, :, 1:, :] - candidate[:, :, :-1, :]
    ref_dx = ref[:, :, 1:, :] - ref[:, :, :-1, :]
    cand_dy = candidate[:, 1:, :, :] - candidate[:, :-1, :, :]
    ref_dy = ref[:, 1:, :, :] - ref[:, :-1, :, :]
    dx_fit = _scorer_input_fit_guard_parts(cand_dx, ref_dx)
    dy_fit = _scorer_input_fit_guard_parts(cand_dy, ref_dy)
    dx_distribution = _value_domain_distribution_guard_parts(cand_dx, ref_dx)
    dy_distribution = _value_domain_distribution_guard_parts(cand_dy, ref_dy)
    total = (
        dx_fit["total"]
        + dy_fit["total"]
        + dx_distribution["std"]
        + dy_distribution["std"]
    )
    return {
        "total": total,
        "mse": dx_fit["mse"] + dy_fit["mse"],
        "mae": dx_fit["mae"] + dy_fit["mae"],
        "std": dx_distribution["std"] + dy_distribution["std"],
    }


def _frame_distribution_guard_parts(
    bundle: RendererBundle,
    rgb: Any,
    gt: Any,
) -> dict[str, Any]:
    """Differentiable scorer-input value-domain guard for one RGB frame.

    The contest scorers never consume human perceptual quality; they consume
    byte-realized RGB/YUV tensors after fixed preprocessing. A renderer that
    saturates, range-collapses, or shifts those tensors can have excellent byte
    rate and terrible SegNet/PoseNet distortion. This guard keeps the decoded
    RGB distribution on the same local manifold as the target before scorer
    surrogates try to learn finer decision boundaries.
    """

    distribution = _value_domain_distribution_guard_parts(rgb, gt)
    saturation = _soft_saturation_guard_parts(bundle, rgb, gt)
    spatial_gradient = _spatial_gradient_guard_parts(rgb, gt)
    total = distribution["total"] + saturation["total"] + spatial_gradient["total"]
    return {
        "total": total,
        "mean": distribution["mean"],
        "std": distribution["std"],
        "dynamic_range": distribution["dynamic_range"],
        "soft_saturation": saturation["soft_saturation"],
        "spatial_gradient": spatial_gradient["total"],
        "spatial_gradient_mse": spatial_gradient["mse"],
        "spatial_gradient_mae": spatial_gradient["mae"],
        "spatial_gradient_std": spatial_gradient["std"],
    }


def _posenet_yuv6_distribution_guard_parts(
    rgb_0: Any,
    rgb_1: Any,
    gt_0: Any,
    gt_1: Any,
) -> dict[str, Any]:
    """Match the PR95 PoseNet YUV6 pair value domain and temporal signal."""

    mx = require_mlx_for_harness()
    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    yuv0 = rgb_to_yuv6_mlx(rgb_0 * 255.0) / 255.0
    yuv1 = rgb_to_yuv6_mlx(rgb_1 * 255.0) / 255.0
    ref_yuv0 = mx.stop_gradient(rgb_to_yuv6_mlx(gt_0 * 255.0) / 255.0)
    ref_yuv1 = mx.stop_gradient(rgb_to_yuv6_mlx(gt_1 * 255.0) / 255.0)
    pair = mx.concatenate([yuv0, yuv1], axis=-1)
    ref_pair = mx.concatenate([ref_yuv0, ref_yuv1], axis=-1)
    temporal_delta = yuv1 - yuv0
    ref_temporal_delta = ref_yuv1 - ref_yuv0

    pair_parts = _value_domain_distribution_guard_parts(pair, ref_pair)
    temporal_parts = _value_domain_distribution_guard_parts(
        temporal_delta,
        ref_temporal_delta,
    )
    pair_fit = _scorer_input_fit_guard_parts(pair, ref_pair)
    temporal_delta_fit = _scorer_input_fit_guard_parts(
        temporal_delta,
        ref_temporal_delta,
    )
    pair_spatial_gradient = _spatial_gradient_guard_parts(pair, ref_pair)
    total = pair_parts["total"] + temporal_parts["total"]
    total = (
        total
        + pair_fit["total"]
        + temporal_delta_fit["total"]
        + pair_spatial_gradient["total"]
    )
    return {
        "total": total,
        "pair": pair_parts["total"],
        "pair_mse": pair_fit["mse"],
        "pair_mae": pair_fit["mae"],
        "pair_mean": pair_parts["mean"],
        "pair_std": pair_parts["std"],
        "pair_dynamic_range": pair_parts["dynamic_range"],
        "pair_spatial_gradient": pair_spatial_gradient["total"],
        "pair_spatial_gradient_mse": pair_spatial_gradient["mse"],
        "pair_spatial_gradient_mae": pair_spatial_gradient["mae"],
        "pair_spatial_gradient_std": pair_spatial_gradient["std"],
        "temporal_delta": temporal_parts["total"],
        "temporal_delta_mse": temporal_delta_fit["mse"],
        "temporal_delta_mae": temporal_delta_fit["mae"],
        "temporal_delta_mean": temporal_parts["mean"],
        "temporal_delta_std": temporal_parts["std"],
        "temporal_delta_dynamic_range": temporal_parts["dynamic_range"],
    }


def posenet_yuv6_geometry_tether_loss(
    rgb_0: Any,
    rgb_1: Any,
    gt_0: Any,
    gt_1: Any,
) -> tuple[Any, dict[str, Any]]:
    """Return a dense PoseNet-only geometry tether on upstream YUV6 inputs.

    The temporal floor only asks whether the decoded pair has enough YUV6
    motion magnitude. The current HiNeRV failure can pass that floor while the
    PoseNet geometry remains wrong. This term prices the exact two-frame YUV6
    pair, temporal delta, and local pair gradients consumed by upstream
    PoseNet; it is not a human-perceptual objective.
    """

    parts = _posenet_yuv6_distribution_guard_parts(rgb_0, rgb_1, gt_0, gt_1)
    total = parts["total"]
    return total, {
        "posenet_yuv6_geometry_tether": total,
        "posenet_yuv6_geometry_tether_pair": parts["pair"],
        "posenet_yuv6_geometry_tether_pair_mse": parts["pair_mse"],
        "posenet_yuv6_geometry_tether_pair_mae": parts["pair_mae"],
        "posenet_yuv6_geometry_tether_pair_spatial_gradient": parts[
            "pair_spatial_gradient"
        ],
        "posenet_yuv6_geometry_tether_pair_spatial_gradient_mse": parts[
            "pair_spatial_gradient_mse"
        ],
        "posenet_yuv6_geometry_tether_pair_spatial_gradient_mae": parts[
            "pair_spatial_gradient_mae"
        ],
        "posenet_yuv6_geometry_tether_temporal_delta": parts["temporal_delta"],
        "posenet_yuv6_geometry_tether_temporal_delta_mse": parts[
            "temporal_delta_mse"
        ],
        "posenet_yuv6_geometry_tether_temporal_delta_mae": parts[
            "temporal_delta_mae"
        ],
        "posenet_yuv6_geometry_tether_temporal_delta_std": parts[
            "temporal_delta_std"
        ],
        "posenet_yuv6_geometry_tether_temporal_delta_dynamic_range": parts[
            "temporal_delta_dynamic_range"
        ],
    }


def scorer_input_distribution_guard_loss(
    bundle: RendererBundle,
    rgb_0: Any,
    rgb_1: Any,
    gt_0: Any,
    gt_1: Any,
) -> tuple[Any, dict[str, Any]]:
    """Return the two-frame differentiable scorer-input distribution guard."""

    parts_0 = _frame_distribution_guard_parts(bundle, rgb_0, gt_0)
    parts_1 = _frame_distribution_guard_parts(bundle, rgb_1, gt_1)
    segnet_frame1_fit = _scorer_input_fit_guard_parts(rgb_1, gt_1)
    yuv6_parts = _posenet_yuv6_distribution_guard_parts(rgb_0, rgb_1, gt_0, gt_1)
    total = parts_0["total"] + parts_1["total"]
    total = total + segnet_frame1_fit["total"] + yuv6_parts["total"]
    return total, {
        "scorer_input_distribution_guard": total,
        "scorer_input_distribution_guard_mean": parts_0["mean"] + parts_1["mean"],
        "scorer_input_distribution_guard_std": parts_0["std"] + parts_1["std"],
        "scorer_input_distribution_guard_dynamic_range": (
            parts_0["dynamic_range"] + parts_1["dynamic_range"]
        ),
        "scorer_input_distribution_guard_soft_saturation": (
            parts_0["soft_saturation"] + parts_1["soft_saturation"]
        ),
        "scorer_input_distribution_guard_spatial_gradient": (
            parts_0["spatial_gradient"] + parts_1["spatial_gradient"]
        ),
        "scorer_input_distribution_guard_spatial_gradient_mse": (
            parts_0["spatial_gradient_mse"] + parts_1["spatial_gradient_mse"]
        ),
        "scorer_input_distribution_guard_spatial_gradient_mae": (
            parts_0["spatial_gradient_mae"] + parts_1["spatial_gradient_mae"]
        ),
        "scorer_input_distribution_guard_segnet_frame1_spatial_gradient": (
            parts_1["spatial_gradient"]
        ),
        "scorer_input_distribution_guard_segnet_frame1_mse": segnet_frame1_fit[
            "mse"
        ],
        "scorer_input_distribution_guard_segnet_frame1_mae": segnet_frame1_fit[
            "mae"
        ],
        "scorer_input_distribution_guard_yuv6_pair": yuv6_parts["pair"],
        "scorer_input_distribution_guard_yuv6_pair_mean": yuv6_parts["pair_mean"],
        "scorer_input_distribution_guard_yuv6_pair_std": yuv6_parts["pair_std"],
        "scorer_input_distribution_guard_yuv6_pair_dynamic_range": yuv6_parts[
            "pair_dynamic_range"
        ],
        "scorer_input_distribution_guard_yuv6_pair_spatial_gradient": (
            yuv6_parts["pair_spatial_gradient"]
        ),
        "scorer_input_distribution_guard_yuv6_pair_spatial_gradient_mse": (
            yuv6_parts["pair_spatial_gradient_mse"]
        ),
        "scorer_input_distribution_guard_yuv6_pair_spatial_gradient_mae": (
            yuv6_parts["pair_spatial_gradient_mae"]
        ),
        "scorer_input_distribution_guard_yuv6_pair_mse": yuv6_parts["pair_mse"],
        "scorer_input_distribution_guard_yuv6_pair_mae": yuv6_parts["pair_mae"],
        "scorer_input_distribution_guard_yuv6_temporal_delta": yuv6_parts[
            "temporal_delta"
        ],
        "scorer_input_distribution_guard_yuv6_temporal_delta_mse": yuv6_parts[
            "temporal_delta_mse"
        ],
        "scorer_input_distribution_guard_yuv6_temporal_delta_mae": yuv6_parts[
            "temporal_delta_mae"
        ],
        "scorer_input_distribution_guard_yuv6_temporal_delta_mean": yuv6_parts[
            "temporal_delta_mean"
        ],
        "scorer_input_distribution_guard_yuv6_temporal_delta_std": yuv6_parts[
            "temporal_delta_std"
        ],
        "scorer_input_distribution_guard_yuv6_temporal_delta_dynamic_range": (
            yuv6_parts["temporal_delta_dynamic_range"]
        ),
    }


def _std_floor_guard_parts(
    candidate: Any,
    reference: Any,
    *,
    min_ratio: float,
) -> dict[str, Any]:
    """Hinge against contrast collapse on a scorer-domain tensor.

    ``candidate`` and ``reference`` are same-domain channels-last tensors. The
    reference is gradient-blocked; the loss only pushes the candidate out of
    the flat-input basin until its per-sample/per-channel std reaches
    ``min_ratio * reference_std``. Normalizing by ``reference_std`` keeps the
    control scale-stable between RGB and YUV6.
    """

    mx = require_mlx_for_harness()
    ref = mx.stop_gradient(reference)
    cand_centered = candidate - mx.mean(candidate, axis=(1, 2), keepdims=True)
    ref_centered = ref - mx.mean(ref, axis=(1, 2), keepdims=True)
    cand_std = mx.sqrt(mx.mean(cand_centered * cand_centered, axis=(1, 2)) + 1.0e-12)
    ref_std = mx.stop_gradient(
        mx.sqrt(mx.mean(ref_centered * ref_centered, axis=(1, 2)) + 1.0e-12)
    )
    ratio = cand_std / mx.maximum(ref_std, 1.0e-6)
    deficit = mx.maximum(float(min_ratio) - ratio, 0.0)
    loss = mx.mean(deficit * deficit)
    return {
        "total": loss,
        "mean_ratio": mx.mean(ratio),
        "min_ratio": mx.min(ratio),
        "target_min_ratio": mx.array(float(min_ratio), dtype=mx.float32),
        "candidate_std": mx.mean(cand_std),
        "reference_std": mx.mean(ref_std),
    }


def scorer_input_contrast_floor_loss(
    bundle: RendererBundle,
    rgb_0: Any,
    rgb_1: Any,
    gt_0: Any,
    gt_1: Any,
) -> tuple[Any, dict[str, Any]]:
    """Return scorer-domain std-floor loss for SegNet RGB and PoseNet YUV6.

    Upstream ``evaluate.py`` scores SegNet on only the last frame of each pair
    and PoseNet on both frames after RGB->YUV6. This term is therefore scoped
    to those exact input domains; it is not a human visual-fidelity proxy.
    """

    mx = require_mlx_for_harness()
    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    segnet_parts = _std_floor_guard_parts(
        rgb_1,
        gt_1,
        min_ratio=float(bundle.scorer_input_contrast_floor_segnet_min_std_ratio),
    )
    yuv0 = rgb_to_yuv6_mlx(rgb_0 * 255.0) / 255.0
    yuv1 = rgb_to_yuv6_mlx(rgb_1 * 255.0) / 255.0
    ref_yuv0 = mx.stop_gradient(rgb_to_yuv6_mlx(gt_0 * 255.0) / 255.0)
    ref_yuv1 = mx.stop_gradient(rgb_to_yuv6_mlx(gt_1 * 255.0) / 255.0)
    pose_parts = _std_floor_guard_parts(
        mx.concatenate([yuv0, yuv1], axis=-1),
        mx.concatenate([ref_yuv0, ref_yuv1], axis=-1),
        min_ratio=float(
            bundle.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio
        ),
    )
    total = segnet_parts["total"] + pose_parts["total"]
    return total, {
        "scorer_input_contrast_floor": total,
        "scorer_input_contrast_floor_segnet_last_rgb": segnet_parts["total"],
        "scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": (
            segnet_parts["mean_ratio"]
        ),
        "scorer_input_contrast_floor_segnet_last_rgb_min_std_ratio": (
            segnet_parts["min_ratio"]
        ),
        "scorer_input_contrast_floor_segnet_last_rgb_target_min_std_ratio": (
            segnet_parts["target_min_ratio"]
        ),
        "scorer_input_contrast_floor_segnet_last_rgb_candidate_std": (
            segnet_parts["candidate_std"]
        ),
        "scorer_input_contrast_floor_segnet_last_rgb_reference_std": (
            segnet_parts["reference_std"]
        ),
        "scorer_input_contrast_floor_posenet_yuv6_pair": pose_parts["total"],
        "scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": (
            pose_parts["mean_ratio"]
        ),
        "scorer_input_contrast_floor_posenet_yuv6_pair_min_std_ratio": (
            pose_parts["min_ratio"]
        ),
        "scorer_input_contrast_floor_posenet_yuv6_pair_target_min_std_ratio": (
            pose_parts["target_min_ratio"]
        ),
        "scorer_input_contrast_floor_posenet_yuv6_pair_candidate_std": (
            pose_parts["candidate_std"]
        ),
        "scorer_input_contrast_floor_posenet_yuv6_pair_reference_std": (
            pose_parts["reference_std"]
        ),
    }


def _mean_abs_floor_guard_parts(
    candidate: Any,
    reference: Any,
    *,
    min_ratio: float,
) -> dict[str, Any]:
    """One-sided hinge on mean absolute temporal signal magnitude."""

    mx = require_mlx_for_harness()
    ref = mx.stop_gradient(reference)
    cand_mean_abs = mx.mean(mx.abs(candidate), axis=(1, 2))
    ref_mean_abs = mx.stop_gradient(mx.mean(mx.abs(ref), axis=(1, 2)))
    ratio = cand_mean_abs / mx.maximum(ref_mean_abs, 1.0e-6)
    deficit = mx.maximum(float(min_ratio) - ratio, 0.0)
    loss = mx.mean(deficit * deficit)
    return {
        "total": loss,
        "mean_ratio": mx.mean(ratio),
        "min_ratio": mx.min(ratio),
        "target_min_ratio": mx.array(float(min_ratio), dtype=mx.float32),
        "candidate_mean_abs": mx.mean(cand_mean_abs),
        "reference_mean_abs": mx.mean(ref_mean_abs),
    }


def posenet_temporal_signal_floor_loss(
    bundle: RendererBundle,
    rgb_0: Any,
    rgb_1: Any,
    gt_0: Any,
    gt_1: Any,
) -> tuple[Any, dict[str, Any]]:
    """Return a PoseNet-specific floor on YUV6 temporal motion signal.

    Upstream PoseNet consumes both frames after RGB->YUV6.  A compact renderer can
    have adequate two-frame YUV6 distribution while making adjacent decoded frames
    almost identical, which destroys the ego-motion signal.  This one-sided floor
    prices only the missing temporal delta magnitude, leaving dense shape fitting
    to ``scorer_input_shape_tether``.
    """

    mx = require_mlx_for_harness()
    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    yuv0 = rgb_to_yuv6_mlx(rgb_0 * 255.0) / 255.0
    yuv1 = rgb_to_yuv6_mlx(rgb_1 * 255.0) / 255.0
    ref_yuv0 = mx.stop_gradient(rgb_to_yuv6_mlx(gt_0 * 255.0) / 255.0)
    ref_yuv1 = mx.stop_gradient(rgb_to_yuv6_mlx(gt_1 * 255.0) / 255.0)
    delta = yuv1 - yuv0
    ref_delta = ref_yuv1 - ref_yuv0
    std_parts = _std_floor_guard_parts(
        delta,
        ref_delta,
        min_ratio=float(bundle.posenet_temporal_signal_min_std_ratio),
    )
    mean_abs_parts = _mean_abs_floor_guard_parts(
        delta,
        ref_delta,
        min_ratio=float(bundle.posenet_temporal_signal_min_mean_abs_ratio),
    )
    total = std_parts["total"] + mean_abs_parts["total"]
    return total, {
        "posenet_temporal_signal_floor": total,
        "posenet_temporal_signal_floor_std": std_parts["total"],
        "posenet_temporal_signal_floor_mean_abs": mean_abs_parts["total"],
        "posenet_temporal_signal_floor_mean_std_ratio": std_parts["mean_ratio"],
        "posenet_temporal_signal_floor_min_std_ratio": std_parts["min_ratio"],
        "posenet_temporal_signal_floor_target_min_std_ratio": std_parts[
            "target_min_ratio"
        ],
        "posenet_temporal_signal_floor_candidate_std": std_parts["candidate_std"],
        "posenet_temporal_signal_floor_reference_std": std_parts["reference_std"],
        "posenet_temporal_signal_floor_mean_abs_ratio": mean_abs_parts["mean_ratio"],
        "posenet_temporal_signal_floor_min_mean_abs_ratio": mean_abs_parts[
            "min_ratio"
        ],
        "posenet_temporal_signal_floor_target_min_mean_abs_ratio": mean_abs_parts[
            "target_min_ratio"
        ],
        "posenet_temporal_signal_floor_candidate_mean_abs": mean_abs_parts[
            "candidate_mean_abs"
        ],
        "posenet_temporal_signal_floor_reference_mean_abs": mean_abs_parts[
            "reference_mean_abs"
        ],
    }


def _centered_variance_normalized_fit_parts(
    candidate: Any,
    reference: Any,
) -> dict[str, Any]:
    """Dense scorer-input shape fit, normalized by reference variance."""

    mx = require_mlx_for_harness()
    ref = mx.stop_gradient(reference)
    cand_centered = candidate - mx.mean(candidate, axis=(1, 2), keepdims=True)
    ref_centered = ref - mx.mean(ref, axis=(1, 2), keepdims=True)
    ref_std = mx.stop_gradient(
        mx.sqrt(
            mx.mean(ref_centered * ref_centered, axis=(1, 2), keepdims=True)
            + 1.0e-12
        )
    )
    normalized_residual = (cand_centered - ref_centered) / mx.maximum(
        ref_std,
        1.0e-6,
    )
    mse = mx.mean(normalized_residual * normalized_residual)
    mae = mx.mean(mx.abs(normalized_residual))
    cand_std = mx.sqrt(
        mx.mean(cand_centered * cand_centered, axis=(1, 2)) + 1.0e-12
    )
    reference_std = mx.sqrt(
        mx.mean(ref_centered * ref_centered, axis=(1, 2)) + 1.0e-12
    )
    return {
        "total": mse + mae,
        "mse": mse,
        "mae": mae,
        "candidate_centered_std": mx.mean(cand_std),
        "reference_centered_std": mx.mean(reference_std),
    }


def scorer_input_shape_tether_loss(
    rgb_0: Any,
    rgb_1: Any,
    gt_0: Any,
    gt_1: Any,
) -> tuple[Any, dict[str, Any]]:
    """Return dense shape tether on exact upstream scorer-input domains.

    SegNet uses only pair frame 1 RGB. PoseNet uses both frames through the
    PR95 YUV6 pair and is also sensitive to temporal motion. The loss is
    centered and normalized by the reference variance, so flat renderers get
    dense gradients toward scorer-causal local structure without adding a human
    perceptual objective.
    """

    mx = require_mlx_for_harness()
    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    segnet_parts = _centered_variance_normalized_fit_parts(rgb_1, gt_1)
    yuv0 = rgb_to_yuv6_mlx(rgb_0 * 255.0) / 255.0
    yuv1 = rgb_to_yuv6_mlx(rgb_1 * 255.0) / 255.0
    ref_yuv0 = mx.stop_gradient(rgb_to_yuv6_mlx(gt_0 * 255.0) / 255.0)
    ref_yuv1 = mx.stop_gradient(rgb_to_yuv6_mlx(gt_1 * 255.0) / 255.0)
    pose_pair = mx.concatenate([yuv0, yuv1], axis=-1)
    ref_pose_pair = mx.concatenate([ref_yuv0, ref_yuv1], axis=-1)
    pose_pair_parts = _centered_variance_normalized_fit_parts(
        pose_pair,
        ref_pose_pair,
    )
    temporal_parts = _centered_variance_normalized_fit_parts(
        yuv1 - yuv0,
        ref_yuv1 - ref_yuv0,
    )
    total = segnet_parts["total"] + pose_pair_parts["total"] + temporal_parts["total"]
    return total, {
        "scorer_input_shape_tether": total,
        "scorer_input_shape_tether_segnet_last_rgb": segnet_parts["total"],
        "scorer_input_shape_tether_segnet_last_rgb_mse": segnet_parts["mse"],
        "scorer_input_shape_tether_segnet_last_rgb_mae": segnet_parts["mae"],
        "scorer_input_shape_tether_segnet_last_rgb_candidate_centered_std": (
            segnet_parts["candidate_centered_std"]
        ),
        "scorer_input_shape_tether_segnet_last_rgb_reference_centered_std": (
            segnet_parts["reference_centered_std"]
        ),
        "scorer_input_shape_tether_posenet_yuv6_pair": pose_pair_parts["total"],
        "scorer_input_shape_tether_posenet_yuv6_pair_mse": pose_pair_parts["mse"],
        "scorer_input_shape_tether_posenet_yuv6_pair_mae": pose_pair_parts["mae"],
        "scorer_input_shape_tether_posenet_yuv6_pair_candidate_centered_std": (
            pose_pair_parts["candidate_centered_std"]
        ),
        "scorer_input_shape_tether_posenet_yuv6_pair_reference_centered_std": (
            pose_pair_parts["reference_centered_std"]
        ),
        "scorer_input_shape_tether_posenet_yuv6_temporal_delta": (
            temporal_parts["total"]
        ),
        "scorer_input_shape_tether_posenet_yuv6_temporal_delta_mse": (
            temporal_parts["mse"]
        ),
        "scorer_input_shape_tether_posenet_yuv6_temporal_delta_mae": (
            temporal_parts["mae"]
        ),
        "scorer_input_shape_tether_posenet_yuv6_temporal_delta_candidate_centered_std": (
            temporal_parts["candidate_centered_std"]
        ),
        "scorer_input_shape_tether_posenet_yuv6_temporal_delta_reference_centered_std": (
            temporal_parts["reference_centered_std"]
        ),
    }


def _segnet_argmax_surface_metrics(
    *,
    candidate_logits: Any,
    target_logits: Any,
    target_argmax: Any | None = None,
) -> dict[str, Any]:
    """Return scorer-class telemetry for a live SegNet surface.

    The scalar distillation loss can decrease while the candidate remains
    collapsed to one class.  These metrics make that failure mode observable in
    the training loop, before export or exact replay budget is spent.
    """

    mx = require_mlx_for_harness()
    class_count = int(candidate_logits.shape[-1])
    pixel_count = int(candidate_logits.size // max(class_count, 1))
    min_class_fraction, min_class_pixels = _segnet_occupancy_min_fraction(pixel_count)
    cand_argmax = mx.argmax(candidate_logits, axis=-1)
    target_argmax = _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=target_argmax,
    )
    cand_argmax_f = cand_argmax.astype(mx.float32)
    target_argmax_f = target_argmax.astype(mx.float32)
    disagreement = mx.mean((cand_argmax != target_argmax).astype(mx.float32))
    cand_mean = mx.mean(candidate_logits)
    target_mean = mx.mean(target_logits)
    metrics: dict[str, Any] = {
        "segnet_direct_live_argmax_disagreement": disagreement,
        "segnet_direct_live_candidate_argmax_mean": mx.mean(cand_argmax_f),
        "segnet_direct_live_target_argmax_mean": mx.mean(target_argmax_f),
        "segnet_direct_live_candidate_logits_mean": cand_mean,
        "segnet_direct_live_target_logits_mean": target_mean,
        "segnet_direct_live_candidate_logits_std": mx.sqrt(
            mx.mean((candidate_logits - cand_mean) ** 2)
        ),
        "segnet_direct_live_target_logits_std": mx.sqrt(
            mx.mean((target_logits - target_mean) ** 2)
        ),
    }
    any_occupied = mx.array(0.0, dtype=mx.float32)
    occupied = mx.array(0.0, dtype=mx.float32)
    target_any_occupied = mx.array(0.0, dtype=mx.float32)
    target_occupied = mx.array(0.0, dtype=mx.float32)
    target_any_covered = mx.array(0.0, dtype=mx.float32)
    target_material_covered = mx.array(0.0, dtype=mx.float32)
    candidate_target_min_ratio = mx.array(1.0, dtype=mx.float32)
    eps = mx.array(1.0e-6, dtype=mx.float32)
    for class_index in range(class_count):
        cand_fraction = mx.mean(
            (cand_argmax == class_index).astype(mx.float32)
        )
        target_fraction = mx.mean(
            (target_argmax == class_index).astype(mx.float32)
        )
        target_present = target_fraction > 0.0
        target_material = target_fraction >= min_class_fraction
        target_material_threshold = mx.minimum(
            target_fraction,
            mx.array(min_class_fraction, dtype=mx.float32),
        )
        any_covered = target_present & (cand_fraction > 0.0)
        material_covered = target_material & (
            cand_fraction >= target_material_threshold
        )
        candidate_target_ratio = mx.minimum(
            cand_fraction / mx.maximum(target_fraction, eps),
            mx.array(1.0, dtype=mx.float32),
        )
        candidate_target_min_ratio = mx.minimum(
            candidate_target_min_ratio,
            mx.where(target_present, candidate_target_ratio, candidate_target_min_ratio),
        )
        metrics[
            f"segnet_direct_live_candidate_class_{class_index}_fraction"
        ] = cand_fraction
        metrics[
            f"segnet_direct_live_target_class_{class_index}_fraction"
        ] = target_fraction
        metrics[
            f"segnet_direct_live_candidate_target_class_{class_index}_ratio"
        ] = candidate_target_ratio
        metrics[
            f"segnet_direct_live_candidate_target_class_{class_index}_any_covered"
        ] = any_covered.astype(mx.float32)
        metrics[
            f"segnet_direct_live_candidate_target_class_{class_index}_material_covered"
        ] = material_covered.astype(mx.float32)
        any_occupied = any_occupied + (cand_fraction > 0.0).astype(mx.float32)
        occupied = occupied + (cand_fraction >= min_class_fraction).astype(mx.float32)
        target_any_occupied = target_any_occupied + (
            target_fraction > 0.0
        ).astype(mx.float32)
        target_occupied = target_occupied + (
            target_fraction >= min_class_fraction
        ).astype(mx.float32)
        target_any_covered = target_any_covered + any_covered.astype(mx.float32)
        target_material_covered = target_material_covered + material_covered.astype(
            mx.float32
        )
    metrics["segnet_direct_live_candidate_any_occupied_class_fraction"] = (
        any_occupied / float(max(class_count, 1))
    )
    metrics["segnet_direct_live_candidate_occupied_class_fraction"] = (
        occupied / float(max(class_count, 1))
    )
    metrics["segnet_direct_live_target_any_occupied_class_fraction"] = (
        target_any_occupied / float(max(class_count, 1))
    )
    metrics["segnet_direct_live_target_occupied_class_fraction"] = (
        target_occupied / float(max(class_count, 1))
    )
    metrics["segnet_direct_live_candidate_target_any_class_coverage_fraction"] = (
        target_any_covered / mx.maximum(target_any_occupied, eps)
    )
    metrics["segnet_direct_live_candidate_target_class_coverage_fraction"] = (
        target_material_covered / mx.maximum(target_occupied, eps)
    )
    metrics["segnet_direct_live_candidate_target_class_missing_fraction"] = (
        mx.array(1.0, dtype=mx.float32)
        - metrics["segnet_direct_live_candidate_target_class_coverage_fraction"]
    )
    metrics["segnet_direct_live_candidate_target_class_min_ratio"] = (
        candidate_target_min_ratio
    )
    metrics["segnet_direct_live_target_any_class_count"] = target_any_occupied
    metrics["segnet_direct_live_target_material_class_count"] = target_occupied
    metrics["segnet_direct_live_candidate_target_any_class_covered_count"] = (
        target_any_covered
    )
    metrics["segnet_direct_live_candidate_target_material_class_covered_count"] = (
        target_material_covered
    )
    metrics["segnet_direct_live_occupancy_min_class_fraction"] = mx.array(
        min_class_fraction, dtype=mx.float32
    )
    metrics["segnet_direct_live_occupancy_min_class_pixel_count"] = mx.array(
        float(min_class_pixels), dtype=mx.float32
    )
    return metrics


def _segnet_class_histogram_loss_and_metrics(
    *,
    bundle: RendererBundle,
    candidate_logits: Any,
    target_logits: Any,
    target_argmax: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Match candidate class measure to the real SegNet target measure.

    ``argmax_hinge`` prices per-pixel decision correctness.  The HiNeRV collapse
    probes showed the missing orthogonal constraint: the candidate can remain
    nearly one-class globally while the hinge improves.  This term is the
    differentiable class-measure tether:

    * target measure = stop-gradient hard argmax histogram from upstream SegNet;
    * candidate measure = mean softmax mass over the live candidate logits;
    * loss = cross-entropy ``H(target_hist, candidate_soft_hist)`` plus a
      symmetric class-measure L1 term.

    It is not a visual prior and not a proxy score claim.  It is a necessary
    condition for low ``d_seg`` because a one-class candidate cannot match the
    target argmax distribution unless the target is itself one-class.  The L1
    component is deliberate: the first HiNeRV direct-live collapse probes showed
    that asymmetric cross-entropy alone underprices overproduced classes while
    the hard argmax surface remains collapsed.
    """

    mx = require_mlx_for_harness()
    class_count = int(candidate_logits.shape[-1])
    if class_count < 1:
        raise ValueError("SegNet class histogram loss requires >=1 class")
    target_idx = _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=target_argmax,
    )
    eye = mx.eye(class_count, dtype=mx.float32)
    target_one_hot = mx.take(eye, target_idx.reshape(-1), axis=0)
    target_hist = mx.stop_gradient(mx.mean(target_one_hot, axis=0))
    temperature = float(bundle.distillation_temperature)
    if temperature <= 0.0:
        raise ValueError(
            "segnet class histogram loss requires positive "
            f"distillation_temperature; got {temperature}"
        )
    logits = candidate_logits / temperature
    logits = logits - mx.max(logits, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits)
    candidate_probs = exp_logits / mx.sum(exp_logits, axis=-1, keepdims=True)
    candidate_hist = mx.mean(candidate_probs.reshape(-1, class_count), axis=0)
    candidate_idx = mx.stop_gradient(mx.argmax(candidate_logits, axis=-1))
    candidate_one_hot = mx.take(eye, candidate_idx.reshape(-1), axis=0)
    candidate_hard_hist = mx.stop_gradient(mx.mean(candidate_one_hot, axis=0))
    eps = mx.array(1.0e-6, dtype=mx.float32)
    candidate_hist_safe = mx.maximum(candidate_hist, eps)
    cross_entropy = -mx.sum(target_hist * mx.log(candidate_hist_safe))
    l1 = mx.sum(mx.abs(candidate_hist - target_hist))
    hard_l1 = mx.sum(mx.abs(candidate_hard_hist - target_hist))
    underproduction = mx.stop_gradient(mx.maximum(target_hist - candidate_hard_hist, 0.0))
    overproduction = mx.stop_gradient(mx.maximum(candidate_hard_hist - target_hist, 0.0))
    overproduction_total = mx.sum(overproduction)
    overproduction_weights = overproduction / mx.maximum(overproduction_total, eps)
    transfer_total = mx.array(0.0, dtype=mx.float32)
    transfer_occupied = mx.array(0.0, dtype=mx.float32)
    transfer_terms = []
    metrics: dict[str, Any] = {
        "segnet_direct_live_class_histogram_cross_entropy": cross_entropy,
        "segnet_direct_live_class_histogram_l1": l1,
        "segnet_direct_live_class_histogram_hard_l1": hard_l1,
        "segnet_direct_live_class_histogram_underproduction_mass": mx.sum(
            underproduction
        ),
        "segnet_direct_live_class_histogram_overproduction_mass": mx.sum(
            overproduction
        ),
    }
    for class_index in range(class_count):
        target_mask = (target_idx == class_index).astype(mx.float32)
        target_mass = mx.sum(target_mask)
        class_active = (target_mass > 0.0).astype(mx.float32)
        class_prob = candidate_probs[..., class_index]
        target_prob_mean = (
            mx.sum(target_mask * class_prob) / mx.maximum(target_mass, eps)
        )
        target_fraction = target_hist[class_index]
        candidate_hard_fraction = candidate_hard_hist[class_index]
        hard_ratio = mx.minimum(
            candidate_hard_fraction / mx.maximum(target_fraction, eps),
            mx.array(1.0, dtype=mx.float32),
        )
        under_ratio = mx.stop_gradient(mx.maximum(1.0 - hard_ratio, 0.0))
        target_prob_floor = mx.minimum(
            mx.array(0.85, dtype=mx.float32),
            mx.maximum(
                mx.array(0.55, dtype=mx.float32),
                mx.array(0.35, dtype=mx.float32) + target_fraction,
            ),
        )
        target_prob_deficit = mx.maximum(target_prob_floor - target_prob_mean, 0.0)
        overproduced_prob = mx.sum(candidate_probs * overproduction_weights, axis=-1)
        overproduced_impostor_prob = mx.maximum(
            overproduced_prob - class_prob * overproduction_weights[class_index],
            mx.array(0.0, dtype=mx.float32),
        )
        overproduced_impostor_loss = (
            mx.sum(target_mask * overproduced_impostor_prob * overproduced_impostor_prob)
            / mx.maximum(target_mass, eps)
        )
        score_mass_boost = (
            mx.array(1.0, dtype=mx.float32)
            + mx.array(32.0, dtype=mx.float32) * target_fraction
        )
        class_transfer = (
            class_active
            * under_ratio
            * score_mass_boost
            * (
                target_prob_deficit * target_prob_deficit
                + mx.array(4.0, dtype=mx.float32) * overproduced_impostor_loss
            )
        )
        transfer_total = transfer_total + class_transfer
        transfer_occupied = transfer_occupied + class_active
        transfer_terms.append(class_transfer)
        metrics[
            f"segnet_direct_live_candidate_soft_class_{class_index}_fraction"
        ] = candidate_hist[class_index]
        metrics[
            f"segnet_direct_live_target_hist_class_{class_index}_fraction"
        ] = target_hist[class_index]
        metrics[
            f"segnet_direct_live_candidate_hard_class_{class_index}_fraction"
        ] = candidate_hard_hist[class_index]
        metrics[
            f"segnet_direct_live_class_histogram_class_{class_index}_underproduction"
        ] = underproduction[class_index]
        metrics[
            f"segnet_direct_live_class_histogram_class_{class_index}_overproduction"
        ] = overproduction[class_index]
        metrics[
            f"segnet_direct_live_class_histogram_class_{class_index}_hard_ratio"
        ] = hard_ratio
        metrics[
            f"segnet_direct_live_class_histogram_class_{class_index}_target_prob_mean"
        ] = target_prob_mean
        metrics[
            f"segnet_direct_live_class_histogram_class_{class_index}_target_prob_deficit"
        ] = target_prob_deficit
        metrics[
            f"segnet_direct_live_class_histogram_class_{class_index}_overproduced_impostor_loss"
        ] = overproduced_impostor_loss
        metrics[
            f"segnet_direct_live_class_histogram_class_{class_index}_mass_transfer"
        ] = class_transfer
    transfer_stack = mx.stack(transfer_terms)
    transfer_mean = transfer_total / mx.maximum(transfer_occupied, eps)
    transfer_worst = mx.max(transfer_stack)
    mass_transfer = transfer_mean + transfer_worst
    loss = cross_entropy + l1 + mass_transfer
    metrics["segnet_direct_live_class_histogram_mass_transfer"] = mass_transfer
    metrics["segnet_direct_live_class_histogram_mass_transfer_mean"] = transfer_mean
    metrics["segnet_direct_live_class_histogram_mass_transfer_worst"] = transfer_worst
    metrics["segnet_direct_live_class_histogram_loss"] = loss
    return loss, metrics


def _segnet_class_balanced_hinge_loss_and_metrics(
    *,
    bundle: RendererBundle,
    candidate_logits: Any,
    target_logits: Any,
    target_argmax: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Class-balanced bootstrap hinge for direct-live SegNet collapse escape."""

    mx = require_mlx_for_harness()
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        _argmax_hinge_per_pixel,
    )

    class_count = int(candidate_logits.shape[-1])
    target_idx = _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=target_argmax,
    )
    hinge = _argmax_hinge_per_pixel(
        candidate_logits,
        target_logits,
        margin=float(bundle.segnet_hinge_margin),
        teacher_argmax=target_idx,
    )
    total = mx.array(0.0, dtype=mx.float32)
    occupied = mx.array(0.0, dtype=mx.float32)
    metrics: dict[str, Any] = {}
    eps = mx.array(1.0e-6, dtype=mx.float32)
    for class_index in range(class_count):
        mask = (target_idx == class_index).astype(mx.float32)
        mass = mx.sum(mask)
        class_active = (mass > 0.0).astype(mx.float32)
        class_loss = mx.sum(mask * hinge) / mx.maximum(mass, eps)
        total = total + class_active * class_loss
        occupied = occupied + class_active
        metrics[
            f"segnet_direct_live_class_balanced_hinge_class_{class_index}"
        ] = class_loss
    loss = total / mx.maximum(occupied, eps)
    metrics["segnet_direct_live_class_balanced_hinge_loss"] = loss
    metrics["segnet_direct_live_target_occupied_class_fraction"] = (
        occupied / float(max(class_count, 1))
    )
    return loss, metrics


def _segnet_class_balanced_squared_hinge_loss_and_metrics(
    *,
    bundle: RendererBundle,
    candidate_logits: Any,
    target_logits: Any,
    target_argmax: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Class-balanced squared margin for far-from-boundary collapse escape.

    The linear Crammer-Singer hinge is the right local boundary geometry for
    upstream SegNet argmax flips.  In the observed HiNeRV collapse basin the
    candidate is not local: every hard pixel is class 2, while minority target
    classes only receive soft probability mass.  Squaring the positive margin
    keeps the same decision boundary but makes large violations carry larger
    gradients until each target class is close enough for the linear hinge to
    take over.
    """

    mx = require_mlx_for_harness()
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        _argmax_hinge_per_pixel,
    )

    class_count = int(candidate_logits.shape[-1])
    target_idx = _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=target_argmax,
    )
    hinge = _argmax_hinge_per_pixel(
        candidate_logits,
        target_logits,
        margin=float(bundle.segnet_hinge_margin),
        teacher_argmax=target_idx,
    )
    squared = hinge * hinge
    total = mx.array(0.0, dtype=mx.float32)
    occupied = mx.array(0.0, dtype=mx.float32)
    metrics: dict[str, Any] = {}
    eps = mx.array(1.0e-6, dtype=mx.float32)
    for class_index in range(class_count):
        mask = (target_idx == class_index).astype(mx.float32)
        mass = mx.sum(mask)
        class_active = (mass > 0.0).astype(mx.float32)
        class_loss = mx.sum(mask * squared) / mx.maximum(mass, eps)
        total = total + class_active * class_loss
        occupied = occupied + class_active
        metrics[
            f"segnet_direct_live_class_balanced_squared_hinge_class_{class_index}"
        ] = class_loss
    loss = total / mx.maximum(occupied, eps)
    metrics["segnet_direct_live_class_balanced_squared_hinge_loss"] = loss
    metrics[
        "segnet_direct_live_class_balanced_squared_hinge_target_occupied_class_fraction"
    ] = occupied / float(max(class_count, 1))
    return loss, metrics


def _segnet_class_balanced_ce_loss_and_metrics(
    *,
    candidate_logits: Any,
    target_logits: Any,
    target_argmax: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Class-balanced hard-target CE for direct-live SegNet collapse escape.

    The contest term is hard argmax disagreement, so the target label is the
    real SegNet argmax.  In the observed HiNeRV one-class basin, hinge and
    histogram pressure moved soft mass but did not cross the hard class-2
    decision surface.  Cross-entropy gives larger gradients when the true
    target probability is crushed, while class balancing keeps minority target
    labels visible instead of letting dominant pixels absorb the loss.
    """

    mx = require_mlx_for_harness()
    class_count = int(candidate_logits.shape[-1])
    if class_count < 1:
        raise ValueError("SegNet class-balanced CE requires >=1 class")
    target_idx = _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=target_argmax,
    )
    logits = candidate_logits - mx.max(candidate_logits, axis=-1, keepdims=True)
    log_probs = logits - mx.log(mx.sum(mx.exp(logits), axis=-1, keepdims=True))
    target_log_prob = mx.squeeze(
        mx.take_along_axis(log_probs, target_idx[..., None], axis=-1),
        axis=-1,
    )
    per_pixel = -target_log_prob
    total = mx.array(0.0, dtype=mx.float32)
    occupied = mx.array(0.0, dtype=mx.float32)
    metrics: dict[str, Any] = {}
    eps = mx.array(1.0e-6, dtype=mx.float32)
    active_class_terms = []
    for class_index in range(class_count):
        mask = (target_idx == class_index).astype(mx.float32)
        mass = mx.sum(mask)
        class_active = (mass > 0.0).astype(mx.float32)
        class_loss = mx.sum(mask * per_pixel) / mx.maximum(mass, eps)
        total = total + class_active * class_loss
        occupied = occupied + class_active
        active_class_terms.append(class_active * class_loss)
        metrics[
            f"segnet_direct_live_class_balanced_ce_class_{class_index}"
        ] = class_loss
    class_balanced_mean = total / mx.maximum(occupied, eps)
    worst_active_class = mx.max(mx.stack(active_class_terms))
    loss = class_balanced_mean + worst_active_class
    metrics["segnet_direct_live_class_balanced_ce_loss"] = loss
    metrics["segnet_direct_live_class_balanced_ce_mean_loss"] = (
        class_balanced_mean
    )
    metrics["segnet_direct_live_class_balanced_ce_worst_class_loss"] = (
        worst_active_class
    )
    metrics["segnet_direct_live_class_balanced_ce_target_occupied_class_fraction"] = (
        occupied / float(max(class_count, 1))
    )
    return loss, metrics


def _segnet_class_region_recon_loss_and_metrics(
    *,
    candidate_rgb: Any,
    target_rgb: Any,
    candidate_logits: Any,
    target_logits: Any,
    target_argmax: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Class-balanced fit on upstream SegNet target regions.

    Direct-live CE/hinge move the real SegNet decision surface, but a collapsed
    compact renderer can still starve minority target regions of dense RGB
    gradient.  This term is deliberately scoped to the exact frame/regions that
    upstream ``evaluate.py`` scores: the last-frame SegNet argmax regions.  It
    is not a visual objective; it is a missing-class escape actuator that gives
    dense scorer-input gradients to target classes whose hard candidate mass is
    below the target mass.
    """

    mx = require_mlx_for_harness()
    class_count = int(candidate_logits.shape[-1])
    if class_count < 1:
        raise ValueError("SegNet class-region recon requires >=1 class")
    if tuple(candidate_rgb.shape[:3]) != tuple(candidate_logits.shape[:3]):
        raise ValueError(
            "SegNet class-region recon requires candidate RGB and logits to "
            "share batch/spatial axes; got "
            f"rgb={tuple(candidate_rgb.shape)} logits={tuple(candidate_logits.shape)}"
        )
    if tuple(target_rgb.shape) != tuple(candidate_rgb.shape):
        raise ValueError(
            "SegNet class-region recon requires target RGB to match candidate RGB; "
            f"target={tuple(target_rgb.shape)} candidate={tuple(candidate_rgb.shape)}"
        )

    target_idx = _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=target_argmax,
    )
    candidate_idx = mx.stop_gradient(mx.argmax(candidate_logits, axis=-1))
    ref_rgb = mx.stop_gradient(target_rgb)
    diff = candidate_rgb - ref_rgb
    per_pixel = mx.mean(diff * diff, axis=-1) + mx.mean(mx.abs(diff), axis=-1)

    total = mx.array(0.0, dtype=mx.float32)
    occupied = mx.array(0.0, dtype=mx.float32)
    metrics: dict[str, Any] = {}
    eps = mx.array(1.0e-6, dtype=mx.float32)
    for class_index in range(class_count):
        target_mask = (target_idx == class_index).astype(mx.float32)
        target_mass = mx.sum(target_mask)
        class_active = (target_mass > 0.0).astype(mx.float32)
        class_loss = mx.sum(target_mask * per_pixel) / mx.maximum(target_mass, eps)
        target_fraction = mx.stop_gradient(mx.mean(target_mask))
        candidate_fraction = mx.stop_gradient(
            mx.mean((candidate_idx == class_index).astype(mx.float32))
        )
        deficit = mx.maximum(target_fraction - candidate_fraction, 0.0)
        deficit_ratio = deficit / mx.maximum(target_fraction, eps)
        missing_boost = 1.0 + deficit_ratio
        total = total + class_active * missing_boost * class_loss
        occupied = occupied + class_active
        metrics[
            f"segnet_direct_live_class_region_recon_class_{class_index}"
        ] = class_loss
        metrics[
            f"segnet_direct_live_class_region_recon_class_{class_index}_boost"
        ] = missing_boost
        metrics[
            f"segnet_direct_live_class_region_recon_class_{class_index}_deficit"
        ] = deficit
    loss = total / mx.maximum(occupied, eps)
    metrics["segnet_direct_live_class_region_recon_loss"] = loss
    metrics["segnet_direct_live_class_region_recon_target_occupied_class_fraction"] = (
        occupied / float(max(class_count, 1))
    )
    return loss, metrics


def _segnet_rare_class_logit_loss_and_metrics(
    *,
    candidate_logits: Any,
    target_logits: Any,
    target_argmax: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Deficit-priced CE/soft-mass pressure for any target-present class.

    The material-occupancy gate intentionally ignores one-pixel crumbs, but the
    contest score does not: if upstream SegNet assigns a class to a pixel on
    the scored last frame, a candidate missing that class can still pay
    ``d_seg``.  Class-balanced CE already prevents dominant classes from
    swallowing the loss; this term adds the missing piece observed in HiNeRV
    smokes: hard candidate deficit and rarity must change the train-time price,
    while the candidate soft class mass must remain differentiable.
    """

    mx = require_mlx_for_harness()
    class_count = int(candidate_logits.shape[-1])
    if class_count < 1:
        raise ValueError("SegNet rare-class logit loss requires >=1 class")
    target_idx = _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=target_argmax,
    )
    candidate_idx = mx.stop_gradient(mx.argmax(candidate_logits, axis=-1))
    logits = candidate_logits - mx.max(candidate_logits, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits)
    probs = exp_logits / mx.sum(exp_logits, axis=-1, keepdims=True)
    log_probs = logits - mx.log(mx.sum(exp_logits, axis=-1, keepdims=True))

    eps = mx.array(1.0e-6, dtype=mx.float32)
    # Cap rarity pressure so tiny classes get a real actuator without turning a
    # single mislabeled pixel into an infinite-gradient curriculum.
    rare_floor = mx.array(1.0e-3, dtype=mx.float32)
    rare_cap = mx.array(4.0, dtype=mx.float32)
    total = mx.array(0.0, dtype=mx.float32)
    occupied = mx.array(0.0, dtype=mx.float32)
    active_terms = []
    metrics: dict[str, Any] = {}
    for class_index in range(class_count):
        target_mask = (target_idx == class_index).astype(mx.float32)
        target_mass = mx.sum(target_mask)
        class_active = (target_mass > 0.0).astype(mx.float32)
        target_fraction = mx.stop_gradient(mx.mean(target_mask))
        candidate_hard_fraction = mx.stop_gradient(
            mx.mean((candidate_idx == class_index).astype(mx.float32))
        )
        class_prob = probs[..., class_index]
        candidate_soft_fraction = mx.mean(class_prob)
        target_prob_mean = (
            mx.sum(target_mask * class_prob) / mx.maximum(target_mass, eps)
        )
        class_ce = -mx.sum(target_mask * log_probs[..., class_index]) / mx.maximum(
            target_mass,
            eps,
        )
        class_logit = candidate_logits[..., class_index]
        class_eye = mx.array(
            [1.0 if idx == class_index else 0.0 for idx in range(class_count)],
            dtype=candidate_logits.dtype,
        )
        impostor_logits = candidate_logits - class_eye * mx.array(
            1.0e30,
            dtype=candidate_logits.dtype,
        )
        max_impostor = mx.max(impostor_logits, axis=-1)
        margin = mx.maximum(max_impostor - class_logit + 1.0, 0.0)
        class_margin = mx.sum(target_mask * margin * margin) / mx.maximum(
            target_mass,
            eps,
        )
        masked_margin = mx.where(
            target_mask > 0.0,
            margin,
            mx.array(1.0e30, dtype=margin.dtype),
        )
        frontier_margin = mx.stop_gradient(mx.min(masked_margin))
        finite_frontier_margin = mx.where(
            class_active > 0.0,
            frontier_margin,
            mx.array(0.0, dtype=mx.float32),
        )
        shifted_margin = mx.maximum(margin - frontier_margin, 0.0)
        easy_temperature = mx.minimum(
            mx.array(2.0, dtype=mx.float32),
            mx.maximum(
                mx.array(0.25, dtype=mx.float32),
                mx.sqrt(mx.maximum(target_fraction, rare_floor)),
            ),
        )
        easy_weight = target_mask * mx.exp(
            -mx.stop_gradient(shifted_margin) / easy_temperature
        )
        easy_weight_mass = mx.sum(easy_weight)
        class_easy_margin = mx.sum(easy_weight * margin * margin) / mx.maximum(
            easy_weight_mass,
            eps,
        )
        easy_weight_peak = mx.max(easy_weight) / mx.maximum(easy_weight_mass, eps)
        hard_deficit = mx.maximum(target_fraction - candidate_hard_fraction, 0.0)
        hard_deficit_ratio = hard_deficit / mx.maximum(target_fraction, eps)
        hard_missing = (candidate_hard_fraction <= 0.0).astype(mx.float32)
        material_hard_floor = mx.minimum(
            target_fraction,
            mx.maximum(
                mx.array(0.001, dtype=mx.float32),
                mx.array(0.10, dtype=mx.float32) * target_fraction,
            ),
        )
        hard_undercovered = (
            candidate_hard_fraction + eps < material_hard_floor
        ).astype(mx.float32)
        soft_deficit = mx.maximum(target_fraction - candidate_soft_fraction, 0.0)
        soft_deficit_ratio = soft_deficit / mx.maximum(target_fraction, eps)
        rarity_boost = mx.minimum(
            rare_cap,
            1.0 / mx.sqrt(mx.maximum(target_fraction, rare_floor)),
        )
        # d_seg is a per-pixel flip rate, so a large target class that is hard
        # missing must not be hidden by equal-class averaging. Rarity still
        # gives tiny material classes a real actuator, while score-mass boost
        # keeps large missing classes priced near their true contest leverage.
        score_mass_boost = (
            mx.array(1.0, dtype=mx.float32)
            + mx.array(32.0, dtype=mx.float32)
            * target_fraction
            * (mx.array(1.0, dtype=mx.float32) + hard_undercovered)
        )
        boost = (1.0 + hard_deficit_ratio) * rarity_boost * score_mass_boost
        easy_margin_weight = (
            mx.array(2.0, dtype=mx.float32)
            + mx.array(6.0, dtype=mx.float32) * hard_deficit_ratio
            + mx.array(4.0, dtype=mx.float32) * hard_undercovered
        )
        target_prob_floor = mx.minimum(
            mx.array(0.45, dtype=mx.float32),
            mx.maximum(mx.array(0.20, dtype=mx.float32), target_fraction),
        )
        target_prob_floor_deficit = mx.maximum(
            target_prob_floor - target_prob_mean,
            0.0,
        )
        seed_mass_floor = mx.minimum(
            mx.array(0.08, dtype=mx.float32),
            mx.maximum(mx.array(0.02, dtype=mx.float32), target_fraction),
        )
        seed_mass_floor_log_ratio = mx.maximum(
            mx.log(
                (seed_mass_floor + eps)
                / mx.maximum(candidate_soft_fraction, eps)
            ),
            0.0,
        )
        seed_mass_floor_loss = (
            hard_undercovered
            * seed_mass_floor_log_ratio
            * seed_mass_floor_log_ratio
        )
        seed_weight_normalized = easy_weight / mx.maximum(easy_weight_mass, eps)
        seed_target_prob_mean = mx.sum(seed_weight_normalized * class_prob)
        frontier_island_margin = mx.sum(seed_weight_normalized * margin)
        frontier_island_crossing_loss = (
            frontier_island_margin * frontier_island_margin
        )
        # The contest SegNet term is a hard argmax flip rate.  When a target
        # class has no material candidate island, soft global mass is not enough;
        # the easiest target pixels need class-k probability high enough to win.
        seed_argmax_prob_floor = mx.minimum(
            mx.array(0.75, dtype=mx.float32),
            mx.maximum(
                mx.array(0.55, dtype=mx.float32),
                target_prob_floor + mx.array(0.20, dtype=mx.float32),
            ),
        )
        seed_argmax_prob_floor_deficit = mx.maximum(
            seed_argmax_prob_floor - seed_target_prob_mean,
            0.0,
        )
        seed_argmax_prob_loss = (
            class_active
            * hard_undercovered
            * seed_argmax_prob_floor_deficit
            * seed_argmax_prob_floor_deficit
        )
        crossing_loss = class_active * hard_undercovered * (
            mx.array(8.0, dtype=mx.float32) * frontier_island_crossing_loss
            + mx.array(16.0, dtype=mx.float32)
            * target_prob_floor_deficit
            * target_prob_floor_deficit
        )
        class_loss = (
            class_ce
            + class_margin
            + easy_margin_weight * class_easy_margin
            + 4.0 * soft_deficit_ratio * soft_deficit_ratio
            + 4.0 * seed_mass_floor_loss
            + crossing_loss
            + mx.array(8.0, dtype=mx.float32) * seed_argmax_prob_loss
        )
        boosted_loss = class_active * boost * class_loss
        total = total + boosted_loss
        occupied = occupied + class_active
        active_terms.append(boosted_loss)
        metrics[f"segnet_direct_live_rare_class_logit_class_{class_index}"] = (
            class_loss
        )
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_boost"
        ] = boost
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_rarity_boost"
        ] = rarity_boost
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_score_mass_boost"
        ] = score_mass_boost
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_hard_deficit"
        ] = hard_deficit
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_soft_deficit"
        ] = soft_deficit
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_target_fraction"
        ] = target_fraction
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_candidate_hard_fraction"
        ] = candidate_hard_fraction
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_candidate_soft_fraction"
        ] = candidate_soft_fraction
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_target_prob_mean"
        ] = target_prob_mean
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_margin"
        ] = class_margin
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_easy_margin"
        ] = class_easy_margin
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_easy_weight_peak"
        ] = easy_weight_peak
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_easy_temperature"
        ] = easy_temperature
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_frontier_margin"
        ] = finite_frontier_margin
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_frontier_island_margin"
        ] = frontier_island_margin
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_frontier_island_crossing_loss"
        ] = frontier_island_crossing_loss
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_easy_margin_weight"
        ] = easy_margin_weight
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_hard_missing"
        ] = hard_missing
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_material_hard_floor"
        ] = material_hard_floor
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_hard_undercovered"
        ] = hard_undercovered
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_target_prob_floor"
        ] = target_prob_floor
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_target_prob_floor_deficit"
        ] = target_prob_floor_deficit
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_crossing_loss"
        ] = crossing_loss
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_seed_mass_floor"
        ] = seed_mass_floor
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_seed_mass_floor_log_ratio"
        ] = seed_mass_floor_log_ratio
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_seed_mass_floor_loss"
        ] = seed_mass_floor_loss
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_seed_target_prob_mean"
        ] = seed_target_prob_mean
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_seed_argmax_prob_floor"
        ] = seed_argmax_prob_floor
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_seed_argmax_prob_floor_deficit"
        ] = seed_argmax_prob_floor_deficit
        metrics[
            f"segnet_direct_live_rare_class_logit_class_{class_index}_seed_argmax_prob_loss"
        ] = seed_argmax_prob_loss
    class_mean = total / mx.maximum(occupied, eps)
    worst_class = mx.max(mx.stack(active_terms))
    loss = class_mean + worst_class
    metrics["segnet_direct_live_rare_class_logit_loss"] = loss
    metrics["segnet_direct_live_rare_class_logit_mean_loss"] = class_mean
    metrics["segnet_direct_live_rare_class_logit_worst_class_loss"] = worst_class
    metrics["segnet_direct_live_rare_class_logit_target_occupied_class_fraction"] = (
        occupied / float(max(class_count, 1))
    )
    return loss, metrics


def _segnet_target_mass_floor_loss_and_metrics(
    *,
    candidate_logits: Any,
    target_logits: Any,
    target_argmax: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Differentiable soft-mass floor for target-present SegNet classes.

    This term attacks the HiNeRV/SNeRV post-class-birth failure mode: the
    candidate can contain a scored class somewhere, while the hard mass ratio
    for at least one target-present class remains zero.  The contest SegNet
    score is still a hard argmax flip rate; this loss supplies the pre-argmax
    probability and soft-mass pressure needed for those hard islands to grow.
    """

    mx = require_mlx_for_harness()
    class_count = int(candidate_logits.shape[-1])
    if class_count < 1:
        raise ValueError("SegNet target-mass floor loss requires >=1 class")
    target_idx = _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=target_argmax,
    )
    candidate_idx = mx.stop_gradient(mx.argmax(candidate_logits, axis=-1))
    logits = candidate_logits - mx.max(candidate_logits, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits)
    probs = exp_logits / mx.sum(exp_logits, axis=-1, keepdims=True)

    eps = mx.array(1.0e-6, dtype=mx.float32)
    total = mx.array(0.0, dtype=mx.float32)
    occupied = mx.array(0.0, dtype=mx.float32)
    active_terms = []
    metrics: dict[str, Any] = {}
    for class_index in range(class_count):
        target_mask = (target_idx == class_index).astype(mx.float32)
        target_mass = mx.sum(target_mask)
        class_active = (target_mass > 0.0).astype(mx.float32)
        target_fraction = mx.stop_gradient(mx.mean(target_mask))
        candidate_hard_fraction = mx.stop_gradient(
            mx.mean((candidate_idx == class_index).astype(mx.float32))
        )
        class_prob = probs[..., class_index]
        candidate_soft_fraction = mx.mean(class_prob)
        target_prob_mean = (
            mx.sum(target_mask * class_prob) / mx.maximum(target_mass, eps)
        )
        class_logit = candidate_logits[..., class_index]
        class_eye = mx.array(
            [1.0 if idx == class_index else 0.0 for idx in range(class_count)],
            dtype=candidate_logits.dtype,
        )
        impostor_logits = candidate_logits - class_eye * mx.array(
            1.0e30,
            dtype=candidate_logits.dtype,
        )
        max_impostor = mx.max(impostor_logits, axis=-1)
        target_region_margin = mx.maximum(
            max_impostor - class_logit + mx.array(1.0, dtype=mx.float32),
            0.0,
        )
        target_region_crossing_loss = (
            mx.sum(target_mask * target_region_margin * target_region_margin)
            / mx.maximum(target_mass, eps)
        )
        hard_ratio = candidate_hard_fraction / mx.maximum(target_fraction, eps)
        hard_deficit_ratio = mx.maximum(1.0 - hard_ratio, 0.0)
        soft_mass_floor = mx.minimum(
            target_fraction,
            mx.maximum(
                mx.array(1.0e-3, dtype=mx.float32),
                mx.array(0.35, dtype=mx.float32) * target_fraction,
            ),
        )
        target_prob_floor = mx.minimum(
            mx.array(0.70, dtype=mx.float32),
            mx.maximum(
                mx.array(0.45, dtype=mx.float32),
                mx.array(0.35, dtype=mx.float32) + target_fraction,
            ),
        )
        soft_mass_log_ratio = mx.maximum(
            mx.log((soft_mass_floor + eps) / mx.maximum(candidate_soft_fraction, eps)),
            0.0,
        )
        target_prob_deficit = mx.maximum(target_prob_floor - target_prob_mean, 0.0)
        score_mass_boost = (
            mx.array(1.0, dtype=mx.float32)
            + mx.array(32.0, dtype=mx.float32) * target_fraction
        )
        undercovered_boost = (
            mx.array(1.0, dtype=mx.float32)
            + mx.array(8.0, dtype=mx.float32) * hard_deficit_ratio
        )
        class_loss = (
            soft_mass_log_ratio * soft_mass_log_ratio
            + mx.array(8.0, dtype=mx.float32)
            * target_prob_deficit
            * target_prob_deficit
            + (
                mx.array(1.0, dtype=mx.float32)
                + mx.array(4.0, dtype=mx.float32) * hard_deficit_ratio
            )
            * target_region_crossing_loss
        )
        boosted_loss = class_active * score_mass_boost * undercovered_boost * class_loss
        total = total + boosted_loss
        occupied = occupied + class_active
        active_terms.append(boosted_loss)
        metrics[f"segnet_direct_live_target_mass_floor_class_{class_index}"] = (
            class_loss
        )
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_target_fraction"
        ] = target_fraction
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_candidate_hard_fraction"
        ] = candidate_hard_fraction
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_candidate_soft_fraction"
        ] = candidate_soft_fraction
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_target_prob_mean"
        ] = target_prob_mean
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_hard_ratio"
        ] = hard_ratio
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_hard_deficit_ratio"
        ] = hard_deficit_ratio
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_soft_mass_floor"
        ] = soft_mass_floor
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_target_prob_floor"
        ] = target_prob_floor
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_soft_mass_log_ratio"
        ] = soft_mass_log_ratio
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_target_prob_deficit"
        ] = target_prob_deficit
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_target_region_crossing_loss"
        ] = target_region_crossing_loss
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_target_region_margin_mean"
        ] = (
            mx.sum(target_mask * target_region_margin)
            / mx.maximum(target_mass, eps)
        )
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_score_mass_boost"
        ] = score_mass_boost
        metrics[
            f"segnet_direct_live_target_mass_floor_class_{class_index}_undercovered_boost"
        ] = undercovered_boost
    class_mean = total / mx.maximum(occupied, eps)
    worst_class = mx.max(mx.stack(active_terms))
    loss = class_mean + worst_class
    metrics["segnet_direct_live_target_mass_floor_loss"] = loss
    metrics["segnet_direct_live_target_mass_floor_mean_loss"] = class_mean
    metrics["segnet_direct_live_target_mass_floor_worst_class_loss"] = worst_class
    metrics["segnet_direct_live_target_mass_floor_target_occupied_class_fraction"] = (
        occupied / float(max(class_count, 1))
    )
    return loss, metrics


def _segnet_target_min_ratio_floor_loss_and_metrics(
    *,
    candidate_logits: Any,
    target_logits: Any,
    target_argmax: Any | None = None,
    min_ratio_floor: float = 0.35,
) -> tuple[Any, dict[str, Any]]:
    """Worst-class hard-support birth loss for upstream SegNet argmax geometry.

    ``_segnet_target_mass_floor_loss_and_metrics`` prices soft class mass.  The
    current HiNeRV distortion crux is sharper: after soft mass appears, at least
    one target-present class can still have zero or tiny hard argmax support.
    The contest ``d_seg`` term is a frame-1 hard argmax flip rate, so long runs
    need an explicit differentiable actuator for that minimum target-class ratio.

    The hard ratio itself is stop-gradient telemetry; it only gates/weights
    differentiable target-region logit margins, target probabilities, and a
    frontier-island seed probability.  This keeps the loss on the scorer's
    decision surface without importing human visual fidelity.
    """

    mx = require_mlx_for_harness()
    floor = float(min_ratio_floor)
    if not math.isfinite(floor) or floor < 0.0 or floor > 1.0:
        raise ValueError(
            "SegNet target-min-ratio floor must be finite in [0, 1]; "
            f"got {min_ratio_floor!r}"
        )
    class_count = int(candidate_logits.shape[-1])
    if class_count < 1:
        raise ValueError("SegNet target-min-ratio floor loss requires >=1 class")
    target_idx = _segnet_target_argmax_from_logits_or_exact(
        target_logits,
        target_argmax=target_argmax,
    )
    candidate_idx = mx.stop_gradient(mx.argmax(candidate_logits, axis=-1))
    logits = candidate_logits - mx.max(candidate_logits, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits)
    probs = exp_logits / mx.sum(exp_logits, axis=-1, keepdims=True)

    eps = mx.array(1.0e-6, dtype=mx.float32)
    floor_arr = mx.array(floor, dtype=mx.float32)
    total = mx.array(0.0, dtype=mx.float32)
    occupied = mx.array(0.0, dtype=mx.float32)
    active_terms = []
    unsolved_argmax_masses = []
    score_weighted_unsolved_argmax_masses = []
    score_weighted_crossing_losses = []
    min_ratio = mx.array(1.0, dtype=mx.float32)
    min_region_ratio = mx.array(1.0, dtype=mx.float32)
    worst_ratio_deficit = mx.array(0.0, dtype=mx.float32)
    worst_region_deficit = mx.array(0.0, dtype=mx.float32)
    metrics: dict[str, Any] = {
        "segnet_direct_live_target_min_ratio_floor_configured_floor": floor_arr,
    }
    for class_index in range(class_count):
        target_mask = (target_idx == class_index).astype(mx.float32)
        target_mass = mx.sum(target_mask)
        class_active = (target_mass > 0.0).astype(mx.float32)
        target_fraction = mx.stop_gradient(mx.mean(target_mask))
        candidate_hard_fraction = mx.stop_gradient(
            mx.mean((candidate_idx == class_index).astype(mx.float32))
        )
        target_region_hard_fraction = mx.stop_gradient(
            mx.sum(
                target_mask * (candidate_idx == class_index).astype(mx.float32)
            )
            / mx.maximum(target_mass, eps)
        )
        hard_ratio = mx.minimum(
            candidate_hard_fraction / mx.maximum(target_fraction, eps),
            mx.array(1.0, dtype=mx.float32),
        )
        region_ratio = mx.minimum(
            target_region_hard_fraction,
            mx.array(1.0, dtype=mx.float32),
        )
        min_ratio = mx.minimum(min_ratio, mx.where(class_active > 0.0, hard_ratio, min_ratio))
        min_region_ratio = mx.minimum(
            min_region_ratio,
            mx.where(class_active > 0.0, region_ratio, min_region_ratio),
        )
        ratio_deficit = mx.maximum(floor_arr - hard_ratio, 0.0)
        region_deficit = mx.maximum(floor_arr - region_ratio, 0.0)
        worst_ratio_deficit = mx.maximum(
            worst_ratio_deficit,
            mx.where(class_active > 0.0, ratio_deficit, 0.0),
        )
        worst_region_deficit = mx.maximum(
            worst_region_deficit,
            mx.where(class_active > 0.0, region_deficit, 0.0),
        )
        support_deficit = mx.maximum(ratio_deficit, region_deficit)
        ratio_active = (support_deficit > 0.0).astype(mx.float32)
        target_region_unsolved_argmax_mass = (
            class_active
            * target_fraction
            * mx.maximum(mx.array(1.0, dtype=mx.float32) - region_ratio, 0.0)
        )
        score_weighted_unsolved_argmax_mass = (
            mx.array(100.0, dtype=mx.float32) * target_region_unsolved_argmax_mass
        )
        class_prob = probs[..., class_index]
        candidate_soft_fraction = mx.mean(class_prob)
        target_prob_mean = (
            mx.sum(target_mask * class_prob) / mx.maximum(target_mass, eps)
        )
        class_logit = candidate_logits[..., class_index]
        class_eye = mx.array(
            [1.0 if idx == class_index else 0.0 for idx in range(class_count)],
            dtype=candidate_logits.dtype,
        )
        impostor_logits = candidate_logits - class_eye * mx.array(
            1.0e30,
            dtype=candidate_logits.dtype,
        )
        max_impostor = mx.max(impostor_logits, axis=-1)
        target_region_margin = mx.maximum(
            max_impostor - class_logit + mx.array(1.0, dtype=mx.float32),
            0.0,
        )
        target_region_crossing_loss = (
            mx.sum(target_mask * target_region_margin * target_region_margin)
            / mx.maximum(target_mass, eps)
        )
        score_weighted_crossing_loss = (
            mx.array(100.0, dtype=mx.float32)
            * class_active
            * target_fraction
            * target_region_crossing_loss
        )
        decision_crossing_score_debt_boost = (
            mx.array(1.0, dtype=mx.float32)
            + mx.minimum(
                mx.array(32.0, dtype=mx.float32),
                mx.stop_gradient(score_weighted_unsolved_argmax_mass),
            )
        )
        masked_region_margin = mx.where(
            target_mask > 0.0,
            target_region_margin,
            mx.array(1.0e30, dtype=target_region_margin.dtype),
        )
        raw_frontier_margin = mx.min(masked_region_margin)
        frontier_margin = mx.stop_gradient(
            mx.where(
                class_active > 0.0,
                raw_frontier_margin,
                mx.array(0.0, dtype=target_region_margin.dtype),
            )
        )
        shifted_margin = mx.maximum(
            target_region_margin - frontier_margin,
            mx.array(0.0, dtype=target_region_margin.dtype),
        )
        easy_temperature = mx.minimum(
            mx.array(2.0, dtype=mx.float32),
            mx.maximum(
                mx.array(0.25, dtype=mx.float32),
                mx.sqrt(mx.maximum(target_fraction, eps)),
            ),
        )
        easy_weight = target_mask * mx.exp(
            -mx.stop_gradient(shifted_margin) / easy_temperature
        )
        easy_weight_mass = mx.sum(easy_weight)
        easy_weight_normalized = easy_weight / mx.maximum(easy_weight_mass, eps)
        seed_target_prob_mean = mx.sum(easy_weight_normalized * class_prob)
        seed_island_mean_margin = mx.sum(easy_weight_normalized * target_region_margin)
        seed_island_crossing_loss = mx.sum(
            easy_weight_normalized * target_region_margin * target_region_margin
        )
        target_prob_floor = mx.minimum(
            mx.array(0.80, dtype=mx.float32),
            mx.maximum(
                mx.array(0.55, dtype=mx.float32),
                mx.array(0.35, dtype=mx.float32) + floor_arr,
            ),
        )
        target_prob_deficit = mx.maximum(target_prob_floor - target_prob_mean, 0.0)
        seed_prob_floor = mx.minimum(
            mx.array(0.90, dtype=mx.float32),
            mx.maximum(
                mx.array(0.70, dtype=mx.float32),
                target_prob_floor + mx.array(0.10, dtype=mx.float32),
            ),
        )
        seed_prob_deficit = mx.maximum(seed_prob_floor - seed_target_prob_mean, 0.0)
        soft_mass_floor = mx.minimum(
            target_fraction,
            mx.maximum(
                mx.array(1.0e-3, dtype=mx.float32),
                floor_arr * target_fraction,
            ),
        )
        soft_mass_log_ratio = mx.maximum(
            mx.log((soft_mass_floor + eps) / mx.maximum(candidate_soft_fraction, eps)),
            0.0,
        )
        rarity_boost = mx.minimum(
            mx.array(8.0, dtype=mx.float32),
            1.0 / mx.sqrt(mx.maximum(target_fraction, mx.array(1.0e-4, dtype=mx.float32))),
        )
        score_mass_boost = (
            mx.array(1.0, dtype=mx.float32)
            + mx.array(32.0, dtype=mx.float32) * target_fraction
        )
        ratio_boost = (
            mx.array(1.0, dtype=mx.float32)
            + mx.array(16.0, dtype=mx.float32) * support_deficit
            + mx.array(8.0, dtype=mx.float32) * region_deficit
        )
        class_loss = (
            mx.array(4.0, dtype=mx.float32)
            * soft_mass_log_ratio
            * soft_mass_log_ratio
            + mx.array(12.0, dtype=mx.float32)
            * target_prob_deficit
            * target_prob_deficit
            + mx.array(12.0, dtype=mx.float32)
            * seed_prob_deficit
            * seed_prob_deficit
            + decision_crossing_score_debt_boost
            * (
                mx.array(1.0, dtype=mx.float32)
                + mx.array(4.0, dtype=mx.float32) * region_deficit
            )
            * target_region_crossing_loss
            + mx.array(4.0, dtype=mx.float32) * seed_island_crossing_loss
        )
        boosted_loss = (
            class_active
            * ratio_active
            * rarity_boost
            * score_mass_boost
            * ratio_boost
            * class_loss
        )
        total = total + boosted_loss
        occupied = occupied + class_active
        active_terms.append(boosted_loss)
        unsolved_argmax_masses.append(target_region_unsolved_argmax_mass)
        score_weighted_unsolved_argmax_masses.append(
            score_weighted_unsolved_argmax_mass
        )
        score_weighted_crossing_losses.append(score_weighted_crossing_loss)
        metrics[f"segnet_direct_live_target_min_ratio_floor_class_{class_index}"] = (
            class_loss
        )
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_boost"
        ] = rarity_boost * score_mass_boost * ratio_boost
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_target_fraction"
        ] = target_fraction
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_candidate_hard_fraction"
        ] = candidate_hard_fraction
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_candidate_soft_fraction"
        ] = candidate_soft_fraction
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_hard_ratio"
        ] = hard_ratio
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_region_ratio"
        ] = region_ratio
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_ratio_deficit"
        ] = ratio_deficit
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_region_deficit"
        ] = region_deficit
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_target_prob_mean"
        ] = target_prob_mean
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_target_prob_floor"
        ] = target_prob_floor
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_target_prob_deficit"
        ] = target_prob_deficit
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_seed_target_prob_mean"
        ] = seed_target_prob_mean
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_seed_prob_floor"
        ] = seed_prob_floor
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_seed_prob_deficit"
        ] = seed_prob_deficit
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_soft_mass_floor"
        ] = soft_mass_floor
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_soft_mass_log_ratio"
        ] = soft_mass_log_ratio
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_target_region_crossing_loss"
        ] = target_region_crossing_loss
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_target_region_unsolved_argmax_mass"
        ] = target_region_unsolved_argmax_mass
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_score_weighted_unsolved_argmax_mass"
        ] = score_weighted_unsolved_argmax_mass
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_score_weighted_crossing_loss"
        ] = score_weighted_crossing_loss
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_decision_crossing_score_debt_boost"
        ] = decision_crossing_score_debt_boost
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_seed_island_crossing_loss"
        ] = seed_island_crossing_loss
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_seed_island_mean_margin"
        ] = seed_island_mean_margin
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_target_region_frontier_margin"
        ] = mx.where(
            class_active > 0.0,
            frontier_margin,
            mx.array(0.0, dtype=mx.float32),
        )
        metrics[
            f"segnet_direct_live_target_min_ratio_floor_class_{class_index}_ratio_active"
        ] = ratio_active
    class_mean = total / mx.maximum(occupied, eps)
    worst_class = mx.max(mx.stack(active_terms))
    loss = class_mean + worst_class
    unsolved_stack = mx.stack(unsolved_argmax_masses)
    score_weighted_unsolved_stack = mx.stack(score_weighted_unsolved_argmax_masses)
    score_weighted_crossing_stack = mx.stack(score_weighted_crossing_losses)
    metrics["segnet_direct_live_target_min_ratio_floor_loss"] = loss
    metrics["segnet_direct_live_target_min_ratio_floor_mean_loss"] = class_mean
    metrics["segnet_direct_live_target_min_ratio_floor_worst_class_loss"] = worst_class
    metrics[
        "segnet_direct_live_target_min_ratio_floor_total_target_region_unsolved_argmax_mass"
    ] = mx.sum(unsolved_stack)
    metrics[
        "segnet_direct_live_target_min_ratio_floor_worst_target_region_unsolved_argmax_mass"
    ] = mx.max(unsolved_stack)
    metrics[
        "segnet_direct_live_target_min_ratio_floor_score_weighted_total_unsolved_argmax_mass"
    ] = mx.sum(score_weighted_unsolved_stack)
    metrics[
        "segnet_direct_live_target_min_ratio_floor_score_weighted_worst_unsolved_argmax_mass"
    ] = mx.max(score_weighted_unsolved_stack)
    metrics[
        "segnet_direct_live_target_min_ratio_floor_score_weighted_total_crossing_loss"
    ] = mx.sum(score_weighted_crossing_stack)
    metrics[
        "segnet_direct_live_target_min_ratio_floor_worst_score_weighted_unsolved_argmax_class_index"
    ] = mx.argmax(score_weighted_unsolved_stack).astype(mx.float32)
    metrics["segnet_direct_live_target_min_ratio_floor_worst_ratio_deficit"] = (
        worst_ratio_deficit
    )
    metrics[
        "segnet_direct_live_target_min_ratio_floor_worst_region_deficit"
    ] = worst_region_deficit
    metrics["segnet_direct_live_target_min_ratio_floor_min_ratio"] = min_ratio
    metrics["segnet_direct_live_target_min_ratio_floor_min_region_ratio"] = (
        min_region_ratio
    )
    metrics["segnet_direct_live_target_min_ratio_floor_target_occupied_class_fraction"] = (
        occupied / float(max(class_count, 1))
    )
    return loss, metrics


def _direct_live_segnet_logit_distillation_loss_and_metrics(
    bundle: RendererBundle,
    seg_rgb_nhwc01: Any,
    idx: Any,
    *,
    target_seg_rgb_nhwc01: Any | None = None,
    loss_weights: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Score-facing direct SegNet VJP term plus class-surface telemetry.

    This is deliberately separate from the tiny learnable student head.  The
    live teacher surface runs the real ported SegNet on the decoded candidate
    frame, so gradients flow through SegNet's input Jacobian back into the
    renderer pixels.  The target logits remain the cached real SegNet response
    on the source frame.
    """

    mx = require_mlx_for_harness()
    if bundle.scorer_teacher is None:
        raise ValueError(
            "segnet_direct_live_distillation_weight > 0 requires scorer_teacher"
        )
    live_fn = getattr(bundle.scorer_teacher, "teacher_logits_for_frames_nhwc01", None)
    if not callable(live_fn):
        raise ValueError(
            "segnet_direct_live_distillation_weight > 0 requires "
            "teacher_logits_for_frames_nhwc01"
        )
    candidate_logits = live_fn(seg_rgb_nhwc01)
    target_logits = mx.stop_gradient(bundle.scorer_teacher.teacher_logits_for_indices(idx))
    target_argmax = _exact_segnet_target_argmax_for_indices(
        bundle.scorer_teacher,
        idx,
        target_logits,
    )
    if tuple(candidate_logits.shape) != tuple(target_logits.shape):
        raise ValueError(
            "direct live SegNet candidate/target logits shape mismatch: "
            f"candidate={tuple(candidate_logits.shape)} "
            f"target={tuple(target_logits.shape)}"
        )
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        HintonMlxCustomLossFnConfig,
        score_teacher_distillation_loss,
    )

    loss_cfg = HintonMlxCustomLossFnConfig(
        temperature=bundle.distillation_temperature,
        distillation_objective=bundle.segnet_distillation_objective,
        tau_boundary=bundle.segnet_tau_boundary,
        hinge_margin=bundle.segnet_hinge_margin,
        student_head_out_channels=bundle.distillation_num_classes,
    )
    loss = score_teacher_distillation_loss(
        student_logits=candidate_logits,
        teacher_logits=target_logits,
        config=loss_cfg,
        teacher_argmax=target_argmax,
    )
    base_loss = loss
    base_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_base_loss",
    )
    base_loss_weight = (
        float(bundle.segnet_direct_live_base_loss_weight) * base_stage_weight
    )
    loss = base_loss_weight * base_loss
    metrics = _segnet_argmax_surface_metrics(
        candidate_logits=candidate_logits,
        target_logits=target_logits,
        target_argmax=target_argmax,
    )
    metrics["segnet_direct_live_base_loss"] = base_loss
    metrics["segnet_direct_live_base_loss_config_weight"] = mx.array(
        float(bundle.segnet_direct_live_base_loss_weight),
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_base_loss_stage_weight"] = mx.array(
        base_stage_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_base_loss_weight"] = mx.array(
        base_loss_weight,
        dtype=mx.float32,
    )
    hist_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_class_histogram",
    )
    hist_config_weight = component_config_weight_with_floor(
        loss_weights,
        "segnet_direct_live_class_histogram",
        float(bundle.segnet_direct_live_class_histogram_weight),
    )
    hist_weight = hist_config_weight * hist_stage_weight
    metrics["segnet_direct_live_class_histogram_config_weight"] = mx.array(
        float(bundle.segnet_direct_live_class_histogram_weight),
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_histogram_effective_config_weight"] = mx.array(
        hist_config_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_histogram_stage_weight"] = mx.array(
        hist_stage_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_histogram_weight"] = mx.array(
        hist_weight,
        dtype=mx.float32,
    )
    if hist_weight > 0.0:
        hist_loss, hist_metrics = _segnet_class_histogram_loss_and_metrics(
            bundle=bundle,
            candidate_logits=candidate_logits,
            target_logits=target_logits,
            target_argmax=target_argmax,
        )
        loss = loss + hist_weight * hist_loss
        metrics.update(hist_metrics)
    balanced_hinge_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_class_balanced_hinge",
    )
    balanced_hinge_config_weight = component_config_weight_with_floor(
        loss_weights,
        "segnet_direct_live_class_balanced_hinge",
        float(bundle.segnet_direct_live_class_balanced_hinge_weight),
    )
    balanced_hinge_weight = balanced_hinge_config_weight * balanced_hinge_stage_weight
    metrics["segnet_direct_live_class_balanced_hinge_config_weight"] = mx.array(
        float(bundle.segnet_direct_live_class_balanced_hinge_weight),
        dtype=mx.float32,
    )
    metrics[
        "segnet_direct_live_class_balanced_hinge_effective_config_weight"
    ] = mx.array(
        balanced_hinge_config_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_balanced_hinge_stage_weight"] = mx.array(
        balanced_hinge_stage_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_balanced_hinge_weight"] = mx.array(
        balanced_hinge_weight,
        dtype=mx.float32,
    )
    if balanced_hinge_weight > 0.0:
        balanced_hinge, balanced_metrics = (
            _segnet_class_balanced_hinge_loss_and_metrics(
                bundle=bundle,
                candidate_logits=candidate_logits,
                target_logits=target_logits,
                target_argmax=target_argmax,
            )
        )
        loss = loss + balanced_hinge_weight * balanced_hinge
        metrics.update(balanced_metrics)
    balanced_ce_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_class_balanced_ce",
    )
    balanced_ce_config_weight = component_config_weight_with_floor(
        loss_weights,
        "segnet_direct_live_class_balanced_ce",
        float(bundle.segnet_direct_live_class_balanced_ce_weight),
    )
    balanced_ce_weight = balanced_ce_config_weight * balanced_ce_stage_weight
    metrics["segnet_direct_live_class_balanced_ce_config_weight"] = mx.array(
        float(bundle.segnet_direct_live_class_balanced_ce_weight),
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_balanced_ce_effective_config_weight"] = mx.array(
        balanced_ce_config_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_balanced_ce_stage_weight"] = mx.array(
        balanced_ce_stage_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_balanced_ce_weight"] = mx.array(
        balanced_ce_weight,
        dtype=mx.float32,
    )
    if balanced_ce_weight > 0.0:
        balanced_ce, balanced_ce_metrics = (
            _segnet_class_balanced_ce_loss_and_metrics(
                candidate_logits=candidate_logits,
                target_logits=target_logits,
                target_argmax=target_argmax,
            )
        )
        loss = loss + balanced_ce_weight * balanced_ce
        metrics.update(balanced_ce_metrics)
    squared_hinge_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_class_balanced_squared_hinge",
    )
    squared_hinge_config_weight = component_config_weight_with_floor(
        loss_weights,
        "segnet_direct_live_class_balanced_squared_hinge",
        float(bundle.segnet_direct_live_class_balanced_squared_hinge_weight),
    )
    squared_hinge_weight = squared_hinge_config_weight * squared_hinge_stage_weight
    metrics[
        "segnet_direct_live_class_balanced_squared_hinge_config_weight"
    ] = mx.array(
        float(bundle.segnet_direct_live_class_balanced_squared_hinge_weight),
        dtype=mx.float32,
    )
    metrics[
        "segnet_direct_live_class_balanced_squared_hinge_effective_config_weight"
    ] = mx.array(
        squared_hinge_config_weight,
        dtype=mx.float32,
    )
    metrics[
        "segnet_direct_live_class_balanced_squared_hinge_stage_weight"
    ] = mx.array(
        squared_hinge_stage_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_balanced_squared_hinge_weight"] = mx.array(
        squared_hinge_weight,
        dtype=mx.float32,
    )
    if squared_hinge_weight > 0.0:
        squared_hinge, squared_hinge_metrics = (
            _segnet_class_balanced_squared_hinge_loss_and_metrics(
                bundle=bundle,
                candidate_logits=candidate_logits,
                target_logits=target_logits,
                target_argmax=target_argmax,
            )
        )
        loss = loss + squared_hinge_weight * squared_hinge
        metrics.update(squared_hinge_metrics)
    region_recon_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_class_region_recon",
    )
    region_recon_config_weight = component_config_weight_with_floor(
        loss_weights,
        "segnet_direct_live_class_region_recon",
        float(bundle.segnet_direct_live_class_region_recon_weight),
    )
    region_recon_weight = region_recon_config_weight * region_recon_stage_weight
    metrics["segnet_direct_live_class_region_recon_config_weight"] = mx.array(
        float(bundle.segnet_direct_live_class_region_recon_weight),
        dtype=mx.float32,
    )
    metrics[
        "segnet_direct_live_class_region_recon_effective_config_weight"
    ] = mx.array(
        region_recon_config_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_region_recon_stage_weight"] = mx.array(
        region_recon_stage_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_class_region_recon_weight"] = mx.array(
        region_recon_weight,
        dtype=mx.float32,
    )
    if region_recon_weight > 0.0:
        if target_seg_rgb_nhwc01 is None:
            raise ValueError(
                "segnet_direct_live_class_region_recon_weight > 0 requires "
                "target_seg_rgb_nhwc01 so the loss is scoped to the scored "
                "SegNet target frame."
            )
        region_recon, region_recon_metrics = (
            _segnet_class_region_recon_loss_and_metrics(
                candidate_rgb=seg_rgb_nhwc01,
                target_rgb=target_seg_rgb_nhwc01,
                candidate_logits=candidate_logits,
                target_logits=target_logits,
                target_argmax=target_argmax,
            )
        )
        loss = loss + region_recon_weight * region_recon
        metrics.update(region_recon_metrics)
    rare_class_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_rare_class_logit",
    )
    rare_class_config_weight = component_config_weight_with_floor(
        loss_weights,
        "segnet_direct_live_rare_class_logit",
        float(bundle.segnet_direct_live_rare_class_logit_weight),
    )
    rare_class_weight = rare_class_config_weight * rare_class_stage_weight
    metrics["segnet_direct_live_rare_class_logit_config_weight"] = mx.array(
        float(bundle.segnet_direct_live_rare_class_logit_weight),
        dtype=mx.float32,
    )
    metrics[
        "segnet_direct_live_rare_class_logit_effective_config_weight"
    ] = mx.array(
        rare_class_config_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_rare_class_logit_stage_weight"] = mx.array(
        rare_class_stage_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_rare_class_logit_weight"] = mx.array(
        rare_class_weight,
        dtype=mx.float32,
    )
    if rare_class_weight > 0.0:
        rare_class, rare_class_metrics = _segnet_rare_class_logit_loss_and_metrics(
            candidate_logits=candidate_logits,
            target_logits=target_logits,
            target_argmax=target_argmax,
        )
        loss = loss + rare_class_weight * rare_class
        metrics.update(rare_class_metrics)
    target_mass_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_target_mass_floor",
    )
    target_mass_config_weight = component_config_weight_with_floor(
        loss_weights,
        "segnet_direct_live_target_mass_floor",
        float(bundle.segnet_direct_live_target_mass_floor_weight),
    )
    target_mass_weight = target_mass_config_weight * target_mass_stage_weight
    metrics["segnet_direct_live_target_mass_floor_config_weight"] = mx.array(
        float(bundle.segnet_direct_live_target_mass_floor_weight),
        dtype=mx.float32,
    )
    metrics[
        "segnet_direct_live_target_mass_floor_effective_config_weight"
    ] = mx.array(
        target_mass_config_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_target_mass_floor_stage_weight"] = mx.array(
        target_mass_stage_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_target_mass_floor_weight"] = mx.array(
        target_mass_weight,
        dtype=mx.float32,
    )
    if target_mass_weight > 0.0:
        target_mass, target_mass_metrics = (
            _segnet_target_mass_floor_loss_and_metrics(
                candidate_logits=candidate_logits,
                target_logits=target_logits,
                target_argmax=target_argmax,
            )
        )
        loss = loss + target_mass_weight * target_mass
        metrics.update(target_mass_metrics)
    target_min_ratio_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_target_min_ratio_floor",
    )
    target_min_ratio_config_weight = component_config_weight_with_floor(
        loss_weights,
        "segnet_direct_live_target_min_ratio_floor",
        float(bundle.segnet_direct_live_target_min_ratio_floor_weight),
    )
    target_min_ratio_weight = (
        target_min_ratio_config_weight * target_min_ratio_stage_weight
    )
    metrics["segnet_direct_live_target_min_ratio_floor_config_weight"] = mx.array(
        float(bundle.segnet_direct_live_target_min_ratio_floor_weight),
        dtype=mx.float32,
    )
    metrics[
        "segnet_direct_live_target_min_ratio_floor_effective_config_weight"
    ] = mx.array(
        target_min_ratio_config_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_target_min_ratio_floor_stage_weight"] = mx.array(
        target_min_ratio_stage_weight,
        dtype=mx.float32,
    )
    metrics["segnet_direct_live_target_min_ratio_floor_weight"] = mx.array(
        target_min_ratio_weight,
        dtype=mx.float32,
    )
    if target_min_ratio_weight > 0.0:
        target_min_ratio, target_min_ratio_metrics = (
            _segnet_target_min_ratio_floor_loss_and_metrics(
                candidate_logits=candidate_logits,
                target_logits=target_logits,
                target_argmax=target_argmax,
            )
        )
        loss = loss + target_min_ratio_weight * target_min_ratio
        metrics.update(target_min_ratio_metrics)
    return loss, metrics


def direct_live_segnet_logit_distillation_loss(
    bundle: RendererBundle,
    seg_rgb_nhwc01: Any,
    idx: Any,
) -> Any:
    """Score-facing direct SegNet VJP term for decoded candidate frames."""

    loss, _metrics = _direct_live_segnet_logit_distillation_loss_and_metrics(
        bundle,
        seg_rgb_nhwc01,
        idx,
    )
    return loss


def score_aware_loss(
    bundle: RendererBundle,
    idx: Any,
    *,
    recon_weight: float = 1.0,
    loss_weights: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compute the gradient-reachable MLX score-aware Lagrangian.

    The combined loss is::

        L = recon_weight * (mse(rgb_0, gt_0) + mse(rgb_1, gt_1))
            + distillation_weight * scorer_teacher_objective(student, teacher)
            + sum_k extra_weight[k] * extra_term_k

    The reconstruction MSE is over the canonical NHWC ``[0, 1]`` frames. The
    optional score-aware term is the configured scorer-teacher surrogate
    (gradient-reachable from distill -> decoded frame -> renderer params)
    per CLAUDE.md "eval_roundtrip" + Catalog #164 sister discipline.

    Args:
        bundle: the substrate RendererBundle.
        idx: MLX int32 ``(B,)`` pair-index batch.
        recon_weight: Lagrangian weight on the reconstruction MSE term.
        loss_weights: optional per-name overrides for the extra-loss terms.

    Returns:
        ``(total_loss_scalar, parts_dict)`` where ``parts_dict`` has scalar
        component values for telemetry (``total`` / ``recon`` / ``distill`` /
        per-extra).
    """
    mx = require_mlx_for_harness()
    weights = dict(bundle.extra_loss_weights)
    weights.update(_extra_loss_weight_overrides(loss_weights))
    recon_stage_weight = component_loss_weight(loss_weights, "recon")
    segnet_stage_weight = component_loss_weight(loss_weights, "distill")
    pose_stage_weight = component_loss_weight(loss_weights, "pose_distill")
    pose_direct_live_stage_weight = component_loss_weight(
        loss_weights,
        "pose_direct_live_distill",
        default=pose_stage_weight,
    )
    scorer_input_guard_stage_weight = component_loss_weight(
        loss_weights,
        "scorer_input_guard",
    )
    scorer_input_contrast_floor_stage_weight = component_loss_weight(
        loss_weights,
        "scorer_input_contrast_floor",
        default=scorer_input_guard_stage_weight,
    )
    scorer_input_shape_tether_stage_weight = component_loss_weight(
        loss_weights,
        "scorer_input_shape_tether",
        default=scorer_input_guard_stage_weight,
    )
    posenet_yuv6_geometry_tether_stage_weight = component_loss_weight(
        loss_weights,
        "posenet_yuv6_geometry_tether",
        default=scorer_input_guard_stage_weight,
    )
    posenet_temporal_signal_floor_stage_weight = component_loss_weight(
        loss_weights,
        "posenet_temporal_signal_floor",
        default=scorer_input_guard_stage_weight,
    )
    segnet_direct_live_stage_weight = component_loss_weight(
        loss_weights,
        "segnet_direct_live_distill",
        default=segnet_stage_weight,
    )

    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, idx)
    rgb_0 = _apply_eval_roundtrip_ste_nhwc01(bundle, rgb_0)
    rgb_1 = _apply_eval_roundtrip_ste_nhwc01(bundle, rgb_1)
    gt_0 = bundle.target_rgb_0[idx]
    gt_1 = bundle.target_rgb_1[idx]
    if bundle.recon_pixel_weight is None:
        # Canonical UNIFORM recon — BYTE-IDENTICAL to pre-channel runs.
        mse_0 = mx.mean((rgb_0 - gt_0) ** 2)
        mse_1 = mx.mean((rgb_1 - gt_1) ** 2)
        recon = mse_0 + mse_1
    else:
        # OPT-IN recon_pixel_weight channel (the codex-named building block):
        # re-weight the per-pixel squared error by the measured-SegNet-saliency
        # map (gradient-blocked) so the renderer spends capacity on the pixels
        # the map deems score-relevant. Applied to BOTH frames.
        weight_0 = mx.stop_gradient(
            _prepare_recon_pixel_weight(
                bundle,
                rgb_0.shape,
                idx=idx,
                frame_index=0,
            )
        )
        weight_1 = mx.stop_gradient(
            _prepare_recon_pixel_weight(
                bundle,
                rgb_1.shape,
                idx=idx,
                frame_index=1,
            )
        )
        mse_0 = _weighted_recon(bundle, rgb_0, gt_0, weight_0)
        mse_1 = _weighted_recon(bundle, rgb_1, gt_1, weight_1)
        recon = mse_0 + mse_1
    total = recon_weight * recon_stage_weight * recon
    parts: dict[str, Any] = {"recon": recon}

    if (
        bundle.scorer_input_distribution_guard_weight > 0.0
        and scorer_input_guard_stage_weight != 0.0
    ):
        guard, guard_parts = scorer_input_distribution_guard_loss(
            bundle,
            rgb_0,
            rgb_1,
            gt_0,
            gt_1,
        )
        total = (
            total
            + float(bundle.scorer_input_distribution_guard_weight)
            * scorer_input_guard_stage_weight
            * guard
        )
        parts.update(guard_parts)

    if (
        bundle.scorer_input_contrast_floor_weight > 0.0
        and scorer_input_contrast_floor_stage_weight != 0.0
    ):
        contrast_floor, contrast_floor_parts = scorer_input_contrast_floor_loss(
            bundle,
            rgb_0,
            rgb_1,
            gt_0,
            gt_1,
        )
        total = (
            total
            + float(bundle.scorer_input_contrast_floor_weight)
            * scorer_input_contrast_floor_stage_weight
            * contrast_floor
        )
        parts.update(contrast_floor_parts)

    if (
        bundle.scorer_input_shape_tether_weight > 0.0
        and scorer_input_shape_tether_stage_weight != 0.0
    ):
        shape_tether, shape_tether_parts = scorer_input_shape_tether_loss(
            rgb_0,
            rgb_1,
            gt_0,
            gt_1,
        )
        total = (
            total
            + float(bundle.scorer_input_shape_tether_weight)
            * scorer_input_shape_tether_stage_weight
            * shape_tether
        )
        parts.update(shape_tether_parts)

    if (
        bundle.posenet_yuv6_geometry_tether_weight > 0.0
        and posenet_yuv6_geometry_tether_stage_weight != 0.0
    ):
        pose_geometry, pose_geometry_parts = posenet_yuv6_geometry_tether_loss(
            rgb_0,
            rgb_1,
            gt_0,
            gt_1,
        )
        total = (
            total
            + float(bundle.posenet_yuv6_geometry_tether_weight)
            * posenet_yuv6_geometry_tether_stage_weight
            * pose_geometry
        )
        parts.update(pose_geometry_parts)

    if (
        bundle.posenet_temporal_signal_floor_weight > 0.0
        and posenet_temporal_signal_floor_stage_weight != 0.0
    ):
        temporal_floor, temporal_floor_parts = posenet_temporal_signal_floor_loss(
            bundle,
            rgb_0,
            rgb_1,
            gt_0,
            gt_1,
        )
        total = (
            total
            + float(bundle.posenet_temporal_signal_floor_weight)
            * posenet_temporal_signal_floor_stage_weight
            * temporal_floor
        )
        parts.update(temporal_floor_parts)

    if bundle.distillation_weight > 0.0 and segnet_stage_weight != 0.0:
        from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
            HintonMlxCustomLossFnConfig,
            score_teacher_distillation_loss,
        )

        loss_cfg = HintonMlxCustomLossFnConfig(
            temperature=bundle.distillation_temperature,
            distillation_objective=bundle.segnet_distillation_objective,
            tau_boundary=bundle.segnet_tau_boundary,
            hinge_margin=bundle.segnet_hinge_margin,
            student_head_out_channels=bundle.distillation_num_classes,
        )
        if bundle.scorer_teacher is not None:
            # PRODUCTION path (Catalog #164 + C6 IBPS / DreamerV3 lesson): the
            # distill term BINDS THE REAL SCORER. The student is the learnable
            # 1x1-conv head on the DECODED contest SegNet frame
            # (gradient-bearing: KL -> head(decoded) -> renderer params); the
            # teacher is the REAL contest SegNet's per-pixel class distribution
            # on this pair's TARGET SegNet frame (gradient-blocked). Backprop
            # through the FULL ported SegNet would
            # be ideal but produces NaN gradients in MLX's second-order autograd
            # composition with the renderer's PixelShuffle/bilinear backward;
            # the learnable-head-distilled-from-real-SegNet-teacher surrogate
            # gives a FINITE, genuinely scorer-bound gradient (the head learns
            # decoded-RGB -> real-SegNet-class-logits, so the renderer gradient
            # is pulled toward what the real scorer rewards, NOT toward a fixed
            # cosine of pixel means).
            head = bundle.learnable_student_head
            if head is None:  # defensive; bundle.__post_init__ already enforces.
                raise ValueError(
                    "scorer_teacher set without learnable_student_head; "
                    "RendererBundle.__post_init__ should have rejected this."
                )
            seg_rgb = rgb_1 if bundle.segnet_teacher_frame_index == 1 else rgb_0
            student_logits = head(seg_rgb)
            teacher_logits = mx.stop_gradient(bundle.scorer_teacher.teacher_logits_for_indices(idx))
            teacher_argmax = _exact_segnet_target_argmax_for_indices(
                bundle.scorer_teacher,
                idx,
                teacher_logits,
            )
            distill = score_teacher_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                config=loss_cfg,
                teacher_argmax=teacher_argmax,
            )
        else:
            # SCORER-BLIND mock fallback — reachable ONLY when
            # ``allow_mock_scorer_teacher=True`` (bundle.__post_init__ fails
            # closed otherwise). The MockTeacherLogitsProvider is a fixed cosine
            # of RGB pixel means with NO SegNet weights; the distill gradient is
            # ~parallel to the recon gradient (scorer-blind). Kept for $0
            # no-real-SegNet smokes that explicitly accept reconstruction-proxy.
            from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
                MockTeacherLogitsProvider,
            )

            provider = MockTeacherLogitsProvider(
                num_classes=bundle.distillation_num_classes,
            )
            student_logits = provider.teacher_logits(rgb_0)
            teacher_logits = mx.stop_gradient(provider.teacher_logits(gt_0))
            distill = score_teacher_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                config=loss_cfg,
            )
        total = total + bundle.distillation_weight * segnet_stage_weight * distill
        parts["distill"] = distill

    direct_live_weight = float(bundle.segnet_direct_live_distillation_weight)
    direct_live_subcontrol_active = _segnet_direct_live_subcontrol_active(
        bundle,
        loss_weights,
    )
    direct_live_active = direct_live_weight > 0.0 or direct_live_subcontrol_active
    if direct_live_active and segnet_direct_live_stage_weight != 0.0:
        seg_rgb = rgb_1 if bundle.segnet_teacher_frame_index == 1 else rgb_0
        direct_live_loss_weights: Mapping[str, float] | None = loss_weights
        direct_live_outer_weight = direct_live_weight
        if direct_live_weight <= 0.0:
            direct_live_outer_weight = 1.0
            direct_live_loss_weights = {
                **dict(loss_weights or {}),
                "segnet_direct_live_base_loss": 0.0,
            }
        (
            direct_live_distill,
            direct_live_metrics,
        ) = _direct_live_segnet_logit_distillation_loss_and_metrics(
            bundle,
            seg_rgb,
            idx,
            target_seg_rgb_nhwc01=(
                gt_1 if bundle.segnet_teacher_frame_index == 1 else gt_0
            ),
            loss_weights=direct_live_loss_weights,
        )
        total = (
            total
            + direct_live_outer_weight
            * segnet_direct_live_stage_weight
            * direct_live_distill
        )
        parts["segnet_direct_live_distill"] = direct_live_distill
        parts.update(direct_live_metrics)

    if bundle.pose_distillation_weight > 0.0 and pose_stage_weight != 0.0:
        # PRODUCTION pose path (Catalog #164 + the C6 IBPS / DreamerV3 lesson,
        # POSE axis): the pose term BINDS THE REAL POSENET. The student is the
        # learnable pose head on the DECODED frame PAIR (gradient-bearing:
        # pose-MSE -> pose_head(decoded_0, decoded_1) -> renderer params); the
        # teacher is the REAL contest PoseNet's pose on this pair's TWO TARGET
        # frames (gradient-blocked). Backprop through the full ported FastViT
        # PoseNet would NaN in MLX's second-order autograd composed with the
        # renderer's PixelShuffle/bilinear backward (identical to the SegNet
        # finding); the learnable-head surrogate gives a FINITE, scorer-bound
        # gradient. ``bundle.__post_init__`` already enforces that
        # pose_scorer_teacher + learnable_pose_student_head are both wired.
        pose_head = bundle.learnable_pose_student_head
        if pose_head is None or bundle.pose_scorer_teacher is None:
            raise ValueError(
                "pose_distillation_weight > 0 without pose_scorer_teacher + "
                "learnable_pose_student_head; RendererBundle.__post_init__ "
                "should have rejected this."
            )
        pose_rgb_0, pose_rgb_1 = pose_student_inputs_nhwc(bundle, rgb_0, rgb_1)
        student_pose = pose_head(pose_rgb_0, pose_rgb_1)
        teacher_pose = mx.stop_gradient(bundle.pose_scorer_teacher.teacher_pose_for_indices(idx))
        # Standardize per-dim by the teacher's per-dim std (canonical scale-
        # stable pose objective) when the teacher cache supplies it.
        per_dim_scale = getattr(bundle.pose_scorer_teacher, "per_dim_scale", None)
        pose_train_loss, pose_distill_raw_mse = _pose_distillation_loss_and_raw_mse(
            bundle,
            student_pose=student_pose,
            teacher_pose=teacher_pose,
            per_dim_scale=per_dim_scale,
        )
        pose_score_term = mx.sqrt(10.0 * pose_distill_raw_mse + 1.0e-12)
        total = (
            total
            + bundle.pose_distillation_weight
            * pose_stage_weight
            * pose_score_term
        )
        parts["pose_distill"] = pose_distill_raw_mse
        parts["pose_distill_train_loss"] = pose_train_loss
        parts["pose_distill_raw_mse"] = pose_distill_raw_mse
        parts["pose_score_term"] = pose_score_term

    pose_direct_live_weight = float(bundle.pose_direct_live_distillation_weight)
    if pose_direct_live_weight > 0.0 and pose_direct_live_stage_weight != 0.0:
        direct_pose, direct_pose_metrics = (
            _direct_live_posenet_distillation_loss_and_metrics(
                bundle,
                rgb_0,
                rgb_1,
                idx,
            )
        )
        total = (
            total
            + pose_direct_live_weight * pose_direct_live_stage_weight * direct_pose
        )
        parts.update(direct_pose_metrics)

    if bundle.extra_loss_terms is not None:
        extra = bundle.extra_loss_terms(bundle.model, idx)
        for name, term in extra.items():
            w = float(weights.get(name, 1.0))
            total = total + w * term
            parts[name] = term

    parts["total"] = total
    return total, parts


def build_mlx_segnet_pair_teacher(
    bundle: RendererBundle,
    *,
    upstream_dir: Any = "upstream",
    device: str = "cpu",
) -> Any:
    """Build a real-MLX-SegNet per-pair teacher cache for the harness.

    The canonical scorer-bound teacher per Catalog #164 + the C6 IBPS /
    DreamerV3 RSSM scorer-blindness lesson. Loads the real upstream PyTorch
    SegNet, ports it to MLX (pure-MLX op graph), runs ONE gradient-free SegNet
    forward per pair's TARGET SegNet frame, and caches the per-pixel class
    logits indexed by PAIR index. The cache satisfies the
    :class:`tac.substrates._shared.mlx_score_aware.bundle.ScorerTeacherProvider`
    protocol (``num_classes`` + ``teacher_logits_for_indices``) so it threads
    directly into ``RendererBundle.scorer_teacher``.

    This is the teacher target the learnable student head is distilled toward;
    the renderer gradient then flows KL -> head(decoded) -> renderer, binding
    the renderer to the REAL SegNet's class boundaries (NOT a pixel-cosine).

    NOTE on resolution: the SegNet logits are at SegNet's canonical
    ``(384, 512)`` output. The learnable student head preserves the decoded
    frame's spatial dims, so the bundle's targets MUST be ``(384, 512)`` for
    the student/teacher shapes to align (the canonical contest eval size).

    Args:
        bundle: the harness RendererBundle. Its
            ``segnet_teacher_frame_index`` selects which target frame supplies
            teacher logits; default ``1`` matches upstream SegNet last-frame
            slicing. Targets MUST be NHWC ``[0, 1]`` at SegNet size.
        upstream_dir: path to the upstream repo (contains the SegNet weights).
        device: PyTorch device for the SegNet weight load + MLX port (``cpu``
            per CLAUDE.md "MPS auth eval is NOISE" — no MPS for the teacher).

    Returns:
        a :class:`RealSegNetTeacherLogitsCache` keyed by PAIR index (so its
        ``teacher_logits_for_indices(idx)`` aligns with the harness batch).

    Raises:
        MlxScoreAwareHarnessError: targets are not at SegNet's ``(384, 512)``.
    """
    import hashlib
    from pathlib import Path

    import numpy as np

    from tac.local_acceleration.mlx_scorer_adapters import MLXSegNetAdapter
    from tac.scorer import load_default_segnet
    from tac.substrates._shared.mlx_score_aware.device_gate import (
        MlxScoreAwareHarnessError,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        RealSegNetTeacherLogitsCache,
    )

    mx = require_mlx_for_harness()
    tgt = bundle.target_rgb_1 if bundle.segnet_teacher_frame_index == 1 else bundle.target_rgb_0
    n_pairs, h, w, _c = tgt.shape
    if (h, w) != (384, 512):
        raise MlxScoreAwareHarnessError(
            f"build_mlx_segnet_pair_teacher requires targets at SegNet size "
            f"(384, 512) for student/teacher shape alignment; got ({h}, {w}). "
            "Decode the harness targets at the canonical contest eval size."
        )
    upstream_path = Path(upstream_dir)
    segnet_path = upstream_path / "models" / "segnet.safetensors"
    segnet_sha = (
        hashlib.sha256(segnet_path.read_bytes()).hexdigest()
        if segnet_path.is_file()
        else None
    )
    segnet = load_default_segnet(str(upstream_path), device=device)
    segnet.eval()
    mlx_segnet = MLXSegNetAdapter(segnet)
    # One gradient-free SegNet forward per pair target SegNet frame, chunked to
    # keep memory bounded. Store the teacher cache as fp16 and cast the active
    # batch back to fp32 at lookup time: the teacher is gradient-blocked and the
    # full-video logits cache is a major memory term for 600-pair HiNeRV/SNeRV
    # runs. SegNet preprocess expects RGB in 0..255 (no internal /255 per the
    # upstream cache builder convention), so scale the [0,1] target up.
    chunk = 16
    logits_chunks = []
    argmax_chunks = []
    for start in range(0, n_pairs, chunk):
        end = min(start + chunk, n_pairs)
        x = tgt[start:end] * 255.0  # (b, 384, 512, 3) MLX
        out = mx.stop_gradient(mlx_segnet(x))  # (b, 384, 512, K) MLX
        mx.eval(out)
        logits_fp32 = np.array(out).astype(np.float32)
        argmax_chunks.append(np.argmax(logits_fp32, axis=-1).astype(np.uint8))
        logits_chunks.append(logits_fp32.astype(np.float16))
    logits_np = np.concatenate(logits_chunks, axis=0)  # (n_pairs, 384, 512, K)
    argmax_np = np.concatenate(argmax_chunks, axis=0)  # (n_pairs, 384, 512)
    return RealSegNetTeacherLogitsCache(
        teacher_logits_thwk=mx.array(logits_np).astype(mx.float16),
        teacher_argmax_thw=mx.array(argmax_np, dtype=mx.uint8),
        frame_count=int(logits_np.shape[0]),
        height=int(logits_np.shape[1]),
        width=int(logits_np.shape[2]),
        num_classes=int(logits_np.shape[3]),
        upstream_segnet_safetensors_sha256=segnet_sha,
        live_segnet_adapter=mlx_segnet,
    )


def build_mlx_posenet_pair_teacher(
    bundle: RendererBundle,
    *,
    upstream_dir: Any = "upstream",
    device: str = "cpu",
) -> Any:
    """Build a real-PyTorch-PoseNet per-pair teacher cache for the harness.

    The POSE axis sister of :func:`build_mlx_segnet_pair_teacher` — PoseNet is
    DOMINANT at the frontier (per CLAUDE.md "SegNet vs PoseNet importance").
    Loads the real upstream PyTorch PoseNet, runs ONE gradient-free PoseNet
    forward per pair's TWO TARGET frames (the contest PoseNet consumes the
    FULL pair, not a single frame), and caches the per-pair pose vector (first
    ``bundle.pose_dims`` of the 12-dim pose head) indexed by PAIR index. The
    cache satisfies the
    :class:`tac.substrates._shared.mlx_score_aware.bundle.PoseScorerTeacherProvider`
    protocol so it threads directly into ``RendererBundle.pose_scorer_teacher``.

    This is the teacher target the learnable pose-student head is distilled
    toward (MSE); the renderer gradient then flows pose-MSE -> pose_head(decoded
    pair) -> renderer, binding the renderer to the REAL PoseNet's ego-motion
    estimate (NOT a pixel-MSE-redundant direction).

    The real PoseNet ``preprocess_input`` interpolates each frame to
    ``(384, 512)`` then applies ``rgb_to_yuv6`` per frame -> 6 channels -> a
    ``(B, 12, H', W')`` YUV6 pair. SegNet-size targets ``(384, 512)`` are the
    canonical contest eval size so the interpolate is a no-op on spatial dims.

    Args:
        bundle: the harness RendererBundle. Targets MUST be NHWC ``[0, 1]`` at
            SegNet/contest size ``(384, 512)``. ``bundle.pose_dims`` selects how
            many pose dims to cache (default 6).
        upstream_dir: path to the upstream repo (contains the PoseNet weights).
        device: PyTorch device for the PoseNet weight load + forward (``cpu``
            per CLAUDE.md "MPS auth eval is NOISE" — no MPS for the teacher;
            MPS PoseNet drift is 23x).

    Returns:
        a :class:`RealPoseNetTeacherCache` keyed by PAIR index so its
        ``teacher_pose_for_indices(idx)`` aligns with the harness batch.

    Raises:
        MlxScoreAwareHarnessError: targets are not at contest ``(384, 512)``.
    """
    import hashlib
    import time
    from pathlib import Path

    import numpy as np
    import torch

    from tac.local_acceleration.mlx_scorer_adapters import torch_posenet_to_mlx
    from tac.scorer import load_default_scorers
    from tac.substrates._shared.mlx_score_aware.device_gate import (
        MlxScoreAwareHarnessError,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        RealPoseNetTeacherCache,
    )

    mx = require_mlx_for_harness()
    n_pairs, h, w, _c = bundle.target_rgb_0.shape
    n_pairs_1, h1, w1, _c1 = bundle.target_rgb_1.shape
    if (h, w) != (384, 512) or (h1, w1) != (384, 512):
        raise MlxScoreAwareHarnessError(
            f"build_mlx_posenet_pair_teacher requires both target frames at "
            f"contest size (384, 512); got frame0 ({h}, {w}) frame1 "
            f"({h1}, {w1}). Decode the harness targets at the canonical eval size."
        )
    if n_pairs != n_pairs_1:
        raise MlxScoreAwareHarnessError(
            f"target_rgb_0 ({n_pairs}) and target_rgb_1 ({n_pairs_1}) pair counts must match."
        )
    t0 = time.time()
    upstream_path = Path(upstream_dir)
    posenet_path = upstream_path / "models" / "posenet.safetensors"
    posenet_sha = hashlib.sha256(posenet_path.read_bytes()).hexdigest() if posenet_path.is_file() else None
    posenet, _segnet = load_default_scorers(str(upstream_path), device=device)
    posenet.eval()
    mlx_posenet = None
    mlx_posenet_error = None
    try:
        mlx_posenet = torch_posenet_to_mlx(posenet)
    except Exception as exc:
        # Unit tests and some fallback teacher probes use minimal Torch
        # PoseNet shims that are valid for target-pose caching but do not carry
        # the full upstream module state required for an MLX live scorer port.
        # Returning the cache without the adapter preserves the student-pose
        # path; RendererBundle still fails closed if direct-live PoseNet is
        # enabled with this cache.
        mlx_posenet_error = f"{type(exc).__name__}: {exc}"
    pose_dims = int(bundle.pose_dims)
    chunk = 16
    pose_chunks = []
    with torch.inference_mode():
        for start in range(0, n_pairs, chunk):
            end = min(start + chunk, n_pairs)
            # Keep full-video PoseNet teacher construction chunk-local. A
            # full600 target pair is large enough that materializing both
            # frames as full NumPy float32 videos can kill concurrent MLX
            # training before the runner emits a startup report.
            f0 = np.array(bundle.target_rgb_0[start:end], dtype=np.float32) * 255.0
            f1 = np.array(bundle.target_rgb_1[start:end], dtype=np.float32) * 255.0
            # PoseNet.preprocess_input expects (b, t=2, c=3, H, W) per
            # upstream/modules.py — NCHW frames stacked over the time axis.
            f0_nchw = np.transpose(f0, (0, 3, 1, 2))  # (b, 3, 384, 512)
            f1_nchw = np.transpose(f1, (0, 3, 1, 2))
            stacked = np.stack([f0_nchw, f1_nchw], axis=1)  # (b, 2, 3, 384, 512)
            x = torch.from_numpy(stacked.astype(np.float32)).to(device)
            x_pre = posenet.preprocess_input(x)  # (b, 12, 192, 256) YUV6 pair
            out = posenet(x_pre)  # dict; 'pose' head (b, 12)
            if pose_dims > int(out["pose"].shape[-1]):
                raise MlxScoreAwareHarnessError(
                    f"bundle.pose_dims={pose_dims} exceeds PoseNet pose head width {int(out['pose'].shape[-1])}."
                )
            pose = out["pose"][..., :pose_dims]  # (b, pose_dims)
            pose_chunks.append(pose.detach().cpu().numpy().astype(np.float32))
    pose_np = np.concatenate(pose_chunks, axis=0)  # (n_pairs, pose_dims)
    # Canonical BOUNDED-AMPLIFICATION per-dim scale for the standardized pose-MSE
    # divisor (see pose_distillation_mse_loss PER-DIM SCALING). The raw per-dim
    # std spans ~3 orders of magnitude (dim 0 std ~0.9, the rotation dims ~0.001);
    # dividing by the raw std would AMPLIFY the near-constant dims ~1000x and make
    # them dominate. Floor the scale at 10% of the MAX std so the amplification
    # ratio is capped at 10x — each dim contributes comparably WITHOUT a
    # near-constant dim's tiny error blowing up the loss. This is the canonical
    # robust standardization (Mahalanobis-like with bounded condition number).
    raw_std = np.std(pose_np, axis=0).astype(np.float32)
    scale_floor = max(float(raw_std.max()) * 0.1, 1.0e-3)
    per_dim_scale = np.maximum(raw_std, scale_floor)
    return RealPoseNetTeacherCache(
        teacher_pose_np=mx.array(pose_np),
        num_pairs=int(pose_np.shape[0]),
        pose_dims=int(pose_np.shape[1]),
        per_dim_scale=mx.array(per_dim_scale),
        upstream_posenet_safetensors_sha256=posenet_sha,
        cache_build_seconds=time.time() - t0,
        live_posenet_adapter=mlx_posenet,
        live_posenet_adapter_error=mlx_posenet_error,
    )


__all__ = [
    "build_mlx_posenet_pair_teacher",
    "build_mlx_segnet_pair_teacher",
    "component_loss_weight",
    "decode_frames_nhwc01",
    "pose_student_inputs_nhwc",
    "posenet_temporal_signal_floor_loss",
    "posenet_yuv6_geometry_tether_loss",
    "score_aware_loss",
    "scorer_input_contrast_floor_loss",
    "scorer_input_distribution_guard_loss",
    "scorer_input_shape_tether_loss",
    "source_pair_indices_for_local_batch",
]

# Internal helpers exported for the channel's dedicated tests + substrate reuse.
# (kept out of __all__ so they are not part of the wildcard public surface, but
# importable by name for the recon_pixel_weight A/B confirm + downstream wiring).
