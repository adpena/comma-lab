# SPDX-License-Identifier: MIT
"""Gradient-reachable MLX score-aware Lagrangian (separation of concerns).

This module owns ONLY the loss math: reconstruction MSE + the optional
gradient-reachable Hinton-distilled KL T=2.0 scorer surrogate + optional
substrate-specific extra terms. It is substrate-AGNOSTIC: the renderer forward
convention is decoded via :func:`decode_frames_nhwc01` so the loss never
assumes a fixed model signature.

The score-aware term is the canonical Hinton-distilled surrogate per
CLAUDE.md "eval_roundtrip -- NON-NEGOTIABLE" + Catalog #164 sister discipline:
the teacher is the deterministic ``MockTeacherLogitsProvider`` projection on the
TARGET frame (stop-gradient), the student is the same projection on the DECODED
frame (gradient-bearing). Gradient flows KL -> decoded -> renderer params; the
stop-gradient on the teacher avoids the self-KL false positive.

[verified-against: tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss.hinton_distilled_kl_t2_loss canonical scorer surrogate]
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from tac.substrates._shared.mlx_score_aware.device_gate import (
    require_mlx_for_harness,
)

if TYPE_CHECKING:
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle


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
    if bundle.forward_convention == "reconstruct_pair_nchw01":
        result = model.reconstruct_pair(idx)
        # The renderer may return (rgb_0, rgb_1) or (rgb_0, rgb_1, z); take the
        # first two. Each is (B, 3, H, W) in [0, 1].
        rgb_0 = result[0]
        rgb_1 = result[1]
        rgb_0 = mx.transpose(rgb_0, (0, 2, 3, 1))
        rgb_1 = mx.transpose(rgb_1, (0, 2, 3, 1))
        return rgb_0, rgb_1
    # call_b2chw_255: model(idx) -> (B, 2, 3, H, W) in [0, 255].
    pair = model(idx)
    pair01 = pair / 255.0
    rgb_0 = mx.transpose(pair01[:, 0], (0, 2, 3, 1))
    rgb_1 = mx.transpose(pair01[:, 1], (0, 2, 3, 1))
    return rgb_0, rgb_1


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
            + distillation_weight * T**2 * KL(student || teacher)
            + sum_k extra_weight[k] * extra_term_k

    The reconstruction MSE is over the canonical NHWC ``[0, 1]`` frames. The
    optional score-aware term is the canonical Hinton-distilled KL T=2.0
    surrogate (gradient-reachable from KL -> decoded frame -> renderer params)
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
    if loss_weights:
        weights.update({k: float(v) for k, v in loss_weights.items()})

    rgb_0, rgb_1 = decode_frames_nhwc01(bundle, idx)
    gt_0 = bundle.target_rgb_0[idx]
    gt_1 = bundle.target_rgb_1[idx]
    mse_0 = mx.mean((rgb_0 - gt_0) ** 2)
    mse_1 = mx.mean((rgb_1 - gt_1) ** 2)
    recon = mse_0 + mse_1
    total = recon_weight * recon
    parts: dict[str, Any] = {"recon": recon}

    if bundle.distillation_weight > 0.0:
        from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
            MockTeacherLogitsProvider,
            hinton_distilled_kl_t2_loss,
        )

        provider = MockTeacherLogitsProvider(
            num_classes=bundle.distillation_num_classes,
        )
        # Student consumes the DECODED frame_0; teacher consumes the TARGET
        # frame_0 (stop-gradient). Gradient flows KL -> student_logits ->
        # decoded -> renderer params.
        student_logits = provider.teacher_logits(rgb_0)
        teacher_logits = mx.stop_gradient(provider.teacher_logits(gt_0))
        distill = hinton_distilled_kl_t2_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temperature=bundle.distillation_temperature,
        )
        total = total + bundle.distillation_weight * distill
        parts["distill"] = distill

    if bundle.extra_loss_terms is not None:
        extra = bundle.extra_loss_terms(bundle.model, idx)
        for name, term in extra.items():
            w = float(weights.get(name, 1.0))
            total = total + w * term
            parts[name] = term

    parts["total"] = total
    return total, parts


__all__ = [
    "decode_frames_nhwc01",
    "score_aware_loss",
]
