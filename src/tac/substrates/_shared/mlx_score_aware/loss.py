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
    "scorer_input_guard": (
        "scorer_input_guard",
        "scorer_input_distribution_guard",
        "value_domain_guard",
    ),
}
_CORE_LOSS_WEIGHT_KEYS = frozenset(
    key for aliases in _CORE_LOSS_WEIGHT_ALIASES.values() for key in aliases
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

    ``pose_distillation_loss == "mse"`` preserves the legacy canonical
    PoseNet-teacher objective exactly. ``"huber"`` uses an MSE-matched Huber:
    ``diff**2`` inside the delta and ``2*delta*abs(diff)-delta**2`` outside,
    so small-error curvature matches MSE while large-error gradients are
    bounded. The raw MSE is always computed from the same scaled diff for
    diagnostics and admission gates.
    """
    mx = require_mlx_for_harness()
    diff = student_pose - teacher_pose
    if per_dim_scale is not None:
        diff = diff / mx.maximum(per_dim_scale, 1.0e-12)
    raw_mse = mx.mean(diff * diff)
    if bundle.pose_distillation_loss == "mse":
        return raw_mse, raw_mse
    delta = float(bundle.pose_distillation_huber_delta)
    abs_diff = mx.abs(diff)
    quadratic = diff * diff
    linear = 2.0 * delta * abs_diff - delta * delta
    return mx.mean(mx.where(abs_diff <= delta, quadratic, linear)), raw_mse


def _frame_distribution_guard_parts(bundle: RendererBundle, rgb: Any, gt: Any) -> dict[str, Any]:
    """Differentiable scorer-input value-domain guard for one frame.

    The contest scorers never consume human perceptual quality; they consume
    byte-realized RGB/YUV tensors after fixed preprocessing. A renderer that
    saturates, range-collapses, or shifts those tensors can have excellent byte
    rate and terrible SegNet/PoseNet distortion. This guard keeps the decoded
    RGB distribution on the same local manifold as the target before scorer
    surrogates try to learn finer decision boundaries.
    """

    mx = require_mlx_for_harness()
    margin = float(bundle.scorer_input_distribution_guard_saturation_margin)
    temperature = float(bundle.scorer_input_distribution_guard_temperature)

    cand_mean = mx.mean(rgb, axis=(1, 2))
    ref_mean = mx.stop_gradient(mx.mean(gt, axis=(1, 2)))
    mean_loss = mx.mean((cand_mean - ref_mean) ** 2)

    cand_centered = rgb - mx.mean(rgb, axis=(1, 2), keepdims=True)
    ref_centered = gt - mx.mean(gt, axis=(1, 2), keepdims=True)
    cand_std = mx.sqrt(mx.mean(cand_centered * cand_centered, axis=(1, 2)) + 1.0e-12)
    ref_std = mx.stop_gradient(
        mx.sqrt(mx.mean(ref_centered * ref_centered, axis=(1, 2)) + 1.0e-12)
    )
    std_loss = mx.mean((cand_std - ref_std) ** 2)

    cand_range = mx.max(rgb, axis=(1, 2)) - mx.min(rgb, axis=(1, 2))
    ref_range = mx.stop_gradient(mx.max(gt, axis=(1, 2)) - mx.min(gt, axis=(1, 2)))
    dynamic_range_loss = mx.mean((cand_range - ref_range) ** 2)

    cand_soft_sat = mx.mean(
        mx.sigmoid((margin - rgb) / temperature)
        + mx.sigmoid((rgb - (1.0 - margin)) / temperature),
        axis=(1, 2),
    )
    ref_soft_sat = mx.stop_gradient(
        mx.mean(
            mx.sigmoid((margin - gt) / temperature)
            + mx.sigmoid((gt - (1.0 - margin)) / temperature),
            axis=(1, 2),
        )
    )
    saturation_loss = mx.mean((cand_soft_sat - ref_soft_sat) ** 2)
    total = mean_loss + std_loss + dynamic_range_loss + saturation_loss
    return {
        "total": total,
        "mean": mean_loss,
        "std": std_loss,
        "dynamic_range": dynamic_range_loss,
        "soft_saturation": saturation_loss,
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
    total = parts_0["total"] + parts_1["total"]
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
    }


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
    scorer_input_guard_stage_weight = component_loss_weight(
        loss_weights,
        "scorer_input_guard",
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
            distill = score_teacher_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                config=loss_cfg,
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
        pose_distill, pose_distill_raw_mse = _pose_distillation_loss_and_raw_mse(
            bundle,
            student_pose=student_pose,
            teacher_pose=teacher_pose,
            per_dim_scale=per_dim_scale,
        )
        total = total + bundle.pose_distillation_weight * pose_stage_weight * pose_distill
        parts["pose_distill"] = pose_distill
        if bundle.pose_distillation_loss != "mse":
            parts["pose_distill_raw_mse"] = pose_distill_raw_mse

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
    segnet = load_default_segnet(str(upstream_dir), device=device)
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
    for start in range(0, n_pairs, chunk):
        end = min(start + chunk, n_pairs)
        x = tgt[start:end] * 255.0  # (b, 384, 512, 3) MLX
        out = mx.stop_gradient(mlx_segnet(x))  # (b, 384, 512, K) MLX
        mx.eval(out)
        logits_chunks.append(np.array(out).astype(np.float16))
    logits_np = np.concatenate(logits_chunks, axis=0)  # (n_pairs, 384, 512, K)
    return RealSegNetTeacherLogitsCache(
        teacher_logits_thwk=mx.array(logits_np).astype(mx.float16),
        frame_count=int(logits_np.shape[0]),
        height=int(logits_np.shape[1]),
        width=int(logits_np.shape[2]),
        num_classes=int(logits_np.shape[3]),
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
    )


__all__ = [
    "build_mlx_posenet_pair_teacher",
    "build_mlx_segnet_pair_teacher",
    "component_loss_weight",
    "decode_frames_nhwc01",
    "pose_student_inputs_nhwc",
    "score_aware_loss",
    "scorer_input_distribution_guard_loss",
    "source_pair_indices_for_local_batch",
]

# Internal helpers exported for the channel's dedicated tests + substrate reuse.
# (kept out of __all__ so they are not part of the wildcard public surface, but
# importable by name for the recon_pixel_weight A/B confirm + downstream wiring).
