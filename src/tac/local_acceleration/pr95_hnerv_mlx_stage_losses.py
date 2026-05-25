# SPDX-License-Identifier: MIT
"""MLX-native PR95 HNeRV stage-loss primitives.

The public PR95 HNeRV recipe trains against frozen PoseNet/SegNet targets with
stage-specific segmentation losses plus ``sqrt(10 * pose_mse)``.  These
helpers port that math to MLX and keep it separate from scorer-network wiring:
they prove the loss formula is portable, but they do not by themselves make a
local MLX run score-authoritative.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PR95_MLX_STAGE_LOSS_CONTRACT_SCHEMA = "pr95_hnerv_mlx_stage_loss_contract.v1"
PR95_MLX_STAGE_SCORER_LOSS_SURFACE = "pr95_stage_seg_pose_scorer_surrogate"
PR95_SEG_LOSS_CE = "ce_seg_loss"
PR95_SEG_LOSS_TAU_SOFTPLUS = "tau_softplus_seg_loss"
PR95_SEG_LOSS_SMOOTH_DISAGREEMENT = "smooth_disagreement_seg_loss"
PR95_SEG_LOSS_L7_SOFTPLUS = "l7_softplus_seg_loss"
PR95_SEG_LOSS_FAMILIES: tuple[str, ...] = (
    PR95_SEG_LOSS_CE,
    PR95_SEG_LOSS_TAU_SOFTPLUS,
    PR95_SEG_LOSS_SMOOTH_DISAGREEMENT,
    PR95_SEG_LOSS_L7_SOFTPLUS,
)


def pr95_mlx_cross_entropy_seg_loss(seg_logits_nchw: Any, targets_hard_nhw: Any) -> Any:
    """Stage-1 PR95 cross-entropy over SegNet logits."""

    import mlx.core as mx

    one_hot = _targets_one_hot_nchw(seg_logits_nchw, targets_hard_nhw)
    target_logits = mx.sum(seg_logits_nchw * one_hot, axis=1)
    logsumexp = mx.squeeze(mx.logsumexp(seg_logits_nchw, axis=1, keepdims=True), axis=1)
    return mx.mean(logsumexp - target_logits)


def pr95_mlx_tau_softplus_seg_loss(
    seg_logits_nchw: Any,
    targets_hard_nhw: Any,
    *,
    tau: float = 0.3,
) -> Any:
    """Stage-2 PR95 tau-softplus margin loss."""

    margin = _target_vs_best_other_margin(seg_logits_nchw, targets_hard_nhw)
    return mx_mean_tau_softplus_negative_margin(margin, tau=tau)


def pr95_mlx_smooth_disagreement_seg_loss(
    seg_logits_nchw: Any,
    targets_hard_nhw: Any,
    *,
    tau: float = 0.3,
) -> Any:
    """Stage-3/4 PR95 smooth-disagreement margin loss."""

    import mlx.core as mx

    margin = _target_vs_best_other_margin(seg_logits_nchw, targets_hard_nhw)
    return mx.mean(mx.sigmoid(-margin / float(tau)))


def pr95_mlx_l7_softplus_seg_loss(
    seg_logits_nchw: Any,
    targets_hard_nhw: Any,
    *,
    tau: float = 0.3,
    l7_threshold: float = 1.0,
    l7_mult: float = 4.0,
) -> Any:
    """Stage-5+ PR95 L7-weighted softplus margin loss."""

    import mlx.core as mx

    margin = _target_vs_best_other_margin(seg_logits_nchw, targets_hard_nhw)
    per_pixel = float(tau) * _softplus(-margin / float(tau))
    weights = mx.where(
        margin < float(l7_threshold),
        1.0 + float(l7_mult),
        1.0,
    )
    weights = weights / mx.mean(weights)
    weights = mx.stop_gradient(weights)
    return mx.mean(per_pixel * weights)


def pr95_mlx_pose_loss(pose_pred_first6: Any, pose_target_first6: Any) -> Any:
    """PR95 pose term: ``sqrt(10 * MSE + 1e-12)``."""

    import mlx.core as mx

    residual = pose_pred_first6 - pose_target_first6
    return mx.sqrt(10.0 * mx.mean(residual * residual) + 1.0e-12)


def pr95_mlx_stage_seg_loss(
    loss_family: str,
    seg_logits_nchw: Any,
    targets_hard_nhw: Any,
    *,
    tau: float = 0.3,
    l7_threshold: float = 1.0,
    l7_mult: float = 4.0,
) -> Any:
    """Dispatch a PR95 stage segmentation loss by canonical family name."""

    family = str(loss_family)
    if family == PR95_SEG_LOSS_CE:
        return pr95_mlx_cross_entropy_seg_loss(seg_logits_nchw, targets_hard_nhw)
    if family == PR95_SEG_LOSS_TAU_SOFTPLUS:
        return pr95_mlx_tau_softplus_seg_loss(
            seg_logits_nchw,
            targets_hard_nhw,
            tau=tau,
        )
    if family == PR95_SEG_LOSS_SMOOTH_DISAGREEMENT:
        return pr95_mlx_smooth_disagreement_seg_loss(
            seg_logits_nchw,
            targets_hard_nhw,
            tau=tau,
        )
    if family == PR95_SEG_LOSS_L7_SOFTPLUS:
        return pr95_mlx_l7_softplus_seg_loss(
            seg_logits_nchw,
            targets_hard_nhw,
            tau=tau,
            l7_threshold=l7_threshold,
            l7_mult=l7_mult,
        )
    raise ValueError(
        "unknown PR95 stage loss family "
        f"{loss_family!r}; expected one of {', '.join(PR95_SEG_LOSS_FAMILIES)}"
    )


def pr95_mlx_stage_scorer_surrogate_loss(
    *,
    seg_logits_nchw: Any,
    targets_hard_nhw: Any,
    pose_pred_first6: Any,
    pose_target_first6: Any,
    loss_family: str,
    seg_weight: float = 100.0,
    pose_weight: float = 1.0,
    tau: float = 0.3,
    l7_threshold: float = 1.0,
    l7_mult: float = 4.0,
    cat_entropy_term: Any | None = None,
    cat_lambda: float = 0.0,
) -> Any:
    """PR95 stage objective excluding the decoder/scorer forward plumbing."""

    seg = pr95_mlx_stage_seg_loss(
        loss_family,
        seg_logits_nchw,
        targets_hard_nhw,
        tau=tau,
        l7_threshold=l7_threshold,
        l7_mult=l7_mult,
    )
    pose = pr95_mlx_pose_loss(pose_pred_first6, pose_target_first6)
    total = float(seg_weight) * seg + float(pose_weight) * pose
    if cat_entropy_term is not None and float(cat_lambda) > 0.0:
        total = total + float(cat_lambda) * cat_entropy_term
    return total


def pr95_mlx_stage_loss_contract_from_training_config(
    training_config: Mapping[str, Any],
    *,
    stage_index: int,
) -> dict[str, Any]:
    """Lower registry stage metadata into an executable MLX loss contract."""

    loss_family = str(training_config.get("stage_loss_family") or "")
    if loss_family not in PR95_SEG_LOSS_FAMILIES:
        raise ValueError(
            f"stage {stage_index}: unsupported PR95 loss family {loss_family!r}"
        )
    return {
        "schema": PR95_MLX_STAGE_LOSS_CONTRACT_SCHEMA,
        "source_pr": 95,
        "stage_index": int(stage_index),
        "stage_modules": list(training_config.get("stage_modules") or []),
        "loss_surface": PR95_MLX_STAGE_SCORER_LOSS_SURFACE,
        "seg_loss_family": loss_family,
        "seg_weight": float(training_config.get("stage_seg_weight", 100.0)),
        "pose_weight": float(training_config.get("stage_pose_weight", 1.0)),
        "pose_loss_family": "sqrt_10x_mse_first6",
        "tau": float(training_config.get("stage_tau", 0.3)),
        "l7_threshold": float(training_config.get("stage_l7_threshold", 1.0)),
        "l7_mult": float(training_config.get("stage_l7_mult", 4.0)),
        "cat_entropy_family": "cat_entropy_v2",
        "cat_sigma": float(training_config.get("stage_cat_sigma", 0.2)),
        "cat_lambda": float(training_config.get("stage_cat_lambda", 0.0)),
        "uses_qat": bool(training_config.get("stage_uses_qat", False)),
        "uses_muon": bool(training_config.get("stage_uses_muon", False)),
        "mlx_loss_primitives_implemented": True,
        "scorer_network_forward_gradient_wired": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def mx_mean_tau_softplus_negative_margin(margin: Any, *, tau: float = 0.3) -> Any:
    """Return ``mean(tau * softplus(-margin / tau))`` in MLX."""

    import mlx.core as mx

    return float(tau) * mx.mean(_softplus(-margin / float(tau)))


def _target_vs_best_other_margin(seg_logits_nchw: Any, targets_hard_nhw: Any) -> Any:
    import mlx.core as mx

    one_hot = _targets_one_hot_nchw(seg_logits_nchw, targets_hard_nhw)
    target_logits = mx.sum(seg_logits_nchw * one_hot, axis=1, keepdims=True)
    masked = mx.where(one_hot > 0.0, -1.0e9, seg_logits_nchw)
    best_other = mx.max(masked, axis=1, keepdims=True)
    return target_logits - best_other


def _targets_one_hot_nchw(seg_logits_nchw: Any, targets_hard_nhw: Any) -> Any:
    import mlx.core as mx

    class_count = int(seg_logits_nchw.shape[1])
    one_hot_nhwc = mx.eye(class_count)[targets_hard_nhw]
    return mx.transpose(one_hot_nhwc, (0, 3, 1, 2)).astype(seg_logits_nchw.dtype)


def _softplus(x: Any) -> Any:
    import mlx.core as mx

    return mx.log1p(mx.exp(-mx.abs(x))) + mx.maximum(x, 0.0)


__all__ = [
    "PR95_MLX_STAGE_LOSS_CONTRACT_SCHEMA",
    "PR95_MLX_STAGE_SCORER_LOSS_SURFACE",
    "PR95_SEG_LOSS_CE",
    "PR95_SEG_LOSS_FAMILIES",
    "PR95_SEG_LOSS_L7_SOFTPLUS",
    "PR95_SEG_LOSS_SMOOTH_DISAGREEMENT",
    "PR95_SEG_LOSS_TAU_SOFTPLUS",
    "mx_mean_tau_softplus_negative_margin",
    "pr95_mlx_cross_entropy_seg_loss",
    "pr95_mlx_l7_softplus_seg_loss",
    "pr95_mlx_pose_loss",
    "pr95_mlx_smooth_disagreement_seg_loss",
    "pr95_mlx_stage_loss_contract_from_training_config",
    "pr95_mlx_stage_scorer_surrogate_loss",
    "pr95_mlx_stage_seg_loss",
    "pr95_mlx_tau_softplus_seg_loss",
]
