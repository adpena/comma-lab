# SPDX-License-Identifier: MIT
"""U-DIE-KL substrate-wide loss helper — Grand Reunion symposium #5 composite.

This module is the canonical training-side substrate-agnostic bolt-on
combining three score-aware-loss families into a single torch-tensor loss:

* **U** = UNIWARD per-pixel embedding-cost weighting
  (Holub-Fridrich-Denemark *EURASIP JIS* 2014)
* **DIE** = Detector-Informed Embedding scorer-gradient weighting
  (Yousfi 2022)
* **KL** = Hinton-Vinyals-Dean 2014 SegNet logit distillation, T=2.0
  (Quantizr 0.33 archive recipe per CLAUDE.md "Quantizr intelligence")

The composite is

    total_loss = standard_loss
               + alpha * UNIWARD_weighted_loss(pred, target)
               + beta  * DIE_weighted_loss(pred, target, scorer_seg, scorer_pose)
               + gamma * KL_distill_loss(pred_seg_logits, target_seg_logits, T)

with ``alpha, beta, gamma >= 0`` and substrate-specific defaults documented
in ``.omx/research/u_die_kl_substrate_wide_loss_v1_design_20260515.md``.

Substrate trainer adoption recipe (3 lines)
===========================================

    # 1. Import (one line, in the loss-construction block)
    from tac.losses import UDIEKLLoss

    # 2. Construct (one line, after scorers are loaded; reuses canonical
    #    preprocess_input per Catalog #164)
    udie_kl_loss = UDIEKLLoss(
        scorer_seg=segnet, scorer_pose=posenet,
        alpha=0.5, beta=0.5, gamma=1.0, kl_temperature=2.0,
    )

    # 3. Add to training step (one line, augments existing loss)
    loss = standard_loss + udie_kl_loss(pred_btchw, target_btchw)

The helper honors the canonical scorer-preprocess discipline (Catalog #164:
``scorer.preprocess_input(...)`` BEFORE ``scorer(...)``) by routing through
``tac.losses.core.scorer_forward_pair``.

Lane: ``lane_u_die_kl_substrate_wide_loss_v1_20260515``
Council provenance: Fridrich + Yousfi + Hinton + Quantizr per Grand Reunion
symposium 2026-05-15 Phase D #5 composite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.losses.core import (
    _hwc_to_chw,
    _validate_kl_temperature,
    scorer_forward_pair,
)

__all__ = (
    "DEFAULT_KL_TEMPERATURE",
    "DEFAULT_UNIWARD_EPSILON",
    "DEFAULT_DIE_CACHE_INTERVAL",
    "UDIEKLConfig",
    "UDIEKLLoss",
    "compute_uniward_weight_map",
    "compute_die_weight_map",
    "kl_distill_segnet_term",
)

# ---------------------------------------------------------------------------
# Canonical defaults
# ---------------------------------------------------------------------------

# Hinton 2015 + Quantizr 0.33 deploy + CLAUDE.md "Quantizr intelligence" non-
# negotiable. KL distillation softmax temperature.
DEFAULT_KL_TEMPERATURE: float = 2.0

# UNIWARD wavelet-band-sum denominator floor (Holub-Fridrich-Denemark 2014).
# Prevents division by zero in flat regions where Sigma_b |W_b(I)(p)| ~= 0.
DEFAULT_UNIWARD_EPSILON: float = 1e-3

# DIE-map cache interval. The per-pixel scorer-gradient map is expensive
# (1 backward pass) but slowly varying because contest scorers are frozen
# and the predicted image moves smoothly under SGD; we cache for K
# iterations. K=10 is the canonical default per the symposium #5 row;
# K=1 forces full per-batch recomputation (slower but more accurate).
DEFAULT_DIE_CACHE_INTERVAL: int = 10


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_uniward_epsilon(eps: float) -> float:
    if isinstance(eps, bool):
        raise ValueError("uniward_epsilon must be a finite positive number")
    try:
        value = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("uniward_epsilon must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("uniward_epsilon must be a finite positive number")
    return value


def _validate_weight(value: float, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _validate_cache_interval(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("die_cache_interval must be a positive int")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("die_cache_interval must be a positive int") from exc
    if result < 1:
        raise ValueError("die_cache_interval must be a positive int")
    return result


# ---------------------------------------------------------------------------
# UNIWARD weight map (canonical Haar-like wavelet bands per Mallat 2009 §7.2)
# ---------------------------------------------------------------------------


def compute_uniward_weight_map(
    target_btchw: torch.Tensor,
    *,
    epsilon: float = DEFAULT_UNIWARD_EPSILON,
    detach: bool = True,
) -> torch.Tensor:
    """Compute the UNIWARD per-pixel embedding-cost weight map.

    Per the canonical UNIWARD formulation (Holub-Fridrich-Denemark 2014 §III):

        rho(p) = 1 / (epsilon + |W_h(I)(p)| + |W_v(I)(p)| + |W_d(I)(p)|)

    where ``{W_h, W_v, W_d}`` are horizontal, vertical, diagonal Haar-like
    detail bands. We inverse-steg-flip the framing for inverse-steganalysis:
    weight per-pixel reconstruction loss by ``rho(p)`` so the trainer spends
    its capacity where the scorer is MOST sensitive (flat regions; small
    perturbations are detectable) and saves bytes where the scorer is BLIND
    (textured regions; perturbations are undetectable).

    The weight is normalized per-image to ``mean=1`` so the loss scale
    matches the unweighted MSE — substrate trainers can drop alpha=1 in
    without recalibrating their LR / EMA decay.

    Args:
        target_btchw: ``(B, T, C, H, W)`` ground-truth pair tensor (any
            float dtype). Wavelet bands are computed on the luminance
            channel (mean over C); the same map is broadcast over C/T.
        epsilon: numerical floor preventing division by zero.
        detach: if True, the returned weight map has no gradient. The
            UNIWARD weighting is treated as a fixed prior on the data
            statistics; backprop through it is not the design intent
            (and would couple the loss to the target frames in a way
            that the steganalysis literature does not justify).

    Returns:
        ``(B, T, 1, H, W)`` float32 tensor; per-image-mean is ``1.0``.
    """
    if target_btchw.ndim != 5:
        raise ValueError(
            f"target_btchw must be (B, T, C, H, W); got shape {tuple(target_btchw.shape)}"
        )
    epsilon = _validate_uniward_epsilon(epsilon)
    B, T, C, H, W = target_btchw.shape

    # Compute on grayscale luminance to mirror the UNIWARD UERD-on-Y
    # discipline (Holub 2014 §III.B); aggregation over C is the caller's
    # responsibility per the canonical convention.
    x = target_btchw.to(dtype=torch.float32).mean(dim=2, keepdim=True)  # (B, T, 1, H, W)

    flat = x.reshape(B * T, 1, H, W)

    # Haar-like horizontal first-difference (HL band).
    pad_r = torch.zeros((B * T, 1, H, 1), dtype=flat.dtype, device=flat.device)
    h_band = torch.cat([flat[:, :, :, 1:] - flat[:, :, :, :-1], pad_r], dim=3)

    # Haar-like vertical first-difference (LH band).
    pad_b = torch.zeros((B * T, 1, 1, W), dtype=flat.dtype, device=flat.device)
    v_band = torch.cat([flat[:, :, 1:, :] - flat[:, :, :-1, :], pad_b], dim=2)

    # Haar-like diagonal cross-difference (HH band).
    d_partial = torch.cat([flat[:, :, 1:, :] - flat[:, :, :-1, :], pad_b], dim=2)
    d_band = torch.cat([d_partial[:, :, :, 1:] - d_partial[:, :, :, :-1], pad_r], dim=3)

    bands_abs = h_band.abs() + v_band.abs() + d_band.abs()
    rho = 1.0 / (epsilon + bands_abs)  # (B*T, 1, H, W)

    # Per-image normalization so mean=1.0 (preserves loss scale across
    # substrate trainers; alpha=1 is the canonical "drop-in" default).
    per_image_mean = rho.mean(dim=(2, 3), keepdim=True).clamp_min(1e-12)
    rho_normalized = rho / per_image_mean

    out = rho_normalized.reshape(B, T, 1, H, W)
    out = torch.nan_to_num(out, nan=1.0, posinf=1.0, neginf=1.0)
    if detach:
        out = out.detach()
    return out


# ---------------------------------------------------------------------------
# DIE weight map (scorer-gradient-weighted per Yousfi 2022 §IV)
# ---------------------------------------------------------------------------


def compute_die_weight_map(
    pred_btchw: torch.Tensor,
    target_btchw: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    *,
    detach: bool = True,
) -> torch.Tensor:
    """Compute the DIE per-pixel scorer-attention weight map.

    Per Yousfi 2022 §IV ("Detector-Informed Embedding for Steganography"):

        DIE_attention(p) = ||grad_seg(p)||_2 + ||grad_pose(p)||_2

    where ``grad_seg`` and ``grad_pose`` are the gradients of the contest
    scorers w.r.t. the predicted image at pixel ``p``. We compute the
    gradient via a single autograd backward pass on the score-domain
    distance between the predicted and target frames.

    The map is normalized per-image to ``mean=1`` so the loss scale matches
    the unweighted MSE (sister discipline to ``compute_uniward_weight_map``).

    The scorer-preprocess discipline (Catalog #164: ``preprocess_input``
    BEFORE ``forward``) is honored by routing through
    ``tac.losses.core.scorer_forward_pair``.

    Args:
        pred_btchw: ``(B, T, C, H, W)`` predicted pair tensor with
            ``requires_grad=True`` so the scorer-gradient backward pass
            populates ``.grad`` on a clone of this tensor.
        target_btchw: ``(B, T, C, H, W)`` ground-truth pair tensor (frozen).
        scorer_seg: contest SegNet (``smp.Unet`` per CLAUDE.md "Exact scorer
            architectures"). Weights MUST be frozen
            (``.requires_grad_(False)``) before construction.
        scorer_pose: contest PoseNet (FastViT-T12 hydra-head per CLAUDE.md).
            Weights MUST be frozen.
        detach: if True, the returned weight map has no gradient (canonical
            because we use it as a fixed multiplier inside the loss; the
            gradient flow we want is ``scorer-grad -> weight -> loss ->
            backward -> renderer params``, NOT a double-backward through
            the weight map itself).

    Returns:
        ``(B, T, 1, H, W)`` float32 tensor; per-image-mean is ``1.0``.
    """
    if pred_btchw.ndim != 5 or target_btchw.ndim != 5:
        raise ValueError(
            f"pred and target must be (B, T, C, H, W); got pred shape "
            f"{tuple(pred_btchw.shape)}, target shape {tuple(target_btchw.shape)}"
        )
    if pred_btchw.shape != target_btchw.shape:
        raise ValueError(
            f"pred and target shapes must match; got {tuple(pred_btchw.shape)} vs "
            f"{tuple(target_btchw.shape)}"
        )

    B, T, C, H, W = pred_btchw.shape

    # Detach the inputs and create a fresh leaf tensor for the gradient
    # probe. We do NOT want the renderer's params to receive a gradient
    # from this internal probe; we only want d(score)/d(pixel) at the
    # current pixel value.
    probe = pred_btchw.detach().clone().requires_grad_(True)

    # Run the canonical scorer-forward-pair (handles preprocess_input).
    fp_out, fs_out = scorer_forward_pair(probe, scorer_pose, scorer_seg)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(target_btchw, scorer_pose, scorer_seg)

    # Build a scalar score-domain distance to backprop from.
    pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
    # SegNet distance: simple cross-entropy on softmax(student) vs
    # softmax(teacher); we don't need the canonical surrogate here because
    # we only need a scalar to backprop from.
    seg_log_p = F.log_softmax(fs_out, dim=1)
    seg_q = F.softmax(gs_out, dim=1).detach()
    seg_dist = F.kl_div(seg_log_p, seg_q, reduction="batchmean")

    score_dist = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)
    grad = torch.autograd.grad(
        score_dist, probe, retain_graph=False, create_graph=False, only_inputs=True
    )[0]
    # grad has shape (B, T, C, H, W); pool per-pixel L2 norm over C and T.
    per_pixel = grad.pow(2).sum(dim=2, keepdim=True).sqrt()  # (B, T, 1, H, W)

    # Per-image normalization (mean=1.0 per the loss-scale-preserving
    # convention sister of compute_uniward_weight_map).
    per_image_mean = per_pixel.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-12)
    weight = per_pixel / per_image_mean

    weight = torch.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)
    if detach:
        weight = weight.detach()
    return weight


# ---------------------------------------------------------------------------
# KL distillation term (canonical Hinton-Vinyals-Dean 2014 + T^2 scaling)
# ---------------------------------------------------------------------------


def kl_distill_segnet_term(
    pred_btchw: torch.Tensor,
    target_btchw: torch.Tensor,
    scorer_seg: nn.Module,
    *,
    temperature: float = DEFAULT_KL_TEMPERATURE,
) -> torch.Tensor:
    """Hinton-style KL distillation for SegNet logits at temperature T.

    Computes

        T^2 * KL(softmax(student / T) || softmax(teacher / T))

    where ``student`` is the predicted-frame SegNet output (gradients flow)
    and ``teacher`` is the target-frame SegNet output (frozen,
    ``torch.no_grad()``). PoseNet is intentionally NOT distilled here per
    CLAUDE.md "KL distill caused PoseNet collapse as primary loss" — KL
    distill for SegNet only.

    Honors Catalog #164 by routing through ``preprocess_input`` BEFORE
    ``forward``.

    Args:
        pred_btchw: ``(B, T, C, H, W)`` predicted pair (gradients flow).
        target_btchw: ``(B, T, C, H, W)`` ground-truth pair (frozen).
        scorer_seg: contest SegNet; weights must be frozen.
        temperature: softmax temperature (default 2.0 per Quantizr canon).

    Returns:
        Scalar tensor: ``T^2 * KL_divergence``.
    """
    if pred_btchw.ndim != 5 or target_btchw.ndim != 5:
        raise ValueError(
            f"pred and target must be (B, T, C, H, W); got pred shape "
            f"{tuple(pred_btchw.shape)}, target shape {tuple(target_btchw.shape)}"
        )
    if pred_btchw.shape != target_btchw.shape:
        raise ValueError(
            f"pred and target shapes must match; got {tuple(pred_btchw.shape)} vs "
            f"{tuple(target_btchw.shape)}"
        )
    T_kl = _validate_kl_temperature(temperature)

    # Canonical preprocess + forward (Catalog #164).
    fs_in = scorer_seg.preprocess_input(pred_btchw)
    fs_logits = scorer_seg(fs_in)  # student; gradients flow
    with torch.no_grad():
        gs_in = scorer_seg.preprocess_input(target_btchw)
        gs_logits = scorer_seg(gs_in)  # teacher; frozen

    log_p = F.log_softmax(fs_logits / T_kl, dim=1)
    q = F.softmax(gs_logits / T_kl, dim=1)
    # KL per pixel; sum over class dim, mean over spatial+batch.
    kl_per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)
    seg_kl = kl_per_pixel.mean()
    # Hinton 2015 T^2 scaling: compensates for KL compression at high T so
    # gradient norms are consistent across temperature values.
    return seg_kl * (T_kl * T_kl)


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UDIEKLConfig:
    """Convex-combination weights + hyperparameters for the U-DIE-KL loss.

    Constraints (validated in __post_init__):

    * ``alpha, beta, gamma`` are finite non-negative.
    * ``kl_temperature`` is finite positive (per ``_validate_kl_temperature``).
    * ``uniward_epsilon`` is finite positive.
    * ``die_cache_interval`` is a positive int (1 = recompute every batch;
      higher values cache for K iterations to amortize the
      gradient-probe cost).
    """

    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 1.0
    kl_temperature: float = DEFAULT_KL_TEMPERATURE
    uniward_epsilon: float = DEFAULT_UNIWARD_EPSILON
    die_cache_interval: int = DEFAULT_DIE_CACHE_INTERVAL

    def __post_init__(self) -> None:
        # Use object.__setattr__ to coerce values through validators while
        # preserving frozen-dataclass semantics.
        object.__setattr__(self, "alpha", _validate_weight(self.alpha, field="alpha"))
        object.__setattr__(self, "beta", _validate_weight(self.beta, field="beta"))
        object.__setattr__(self, "gamma", _validate_weight(self.gamma, field="gamma"))
        object.__setattr__(
            self, "kl_temperature", _validate_kl_temperature(self.kl_temperature)
        )
        object.__setattr__(
            self, "uniward_epsilon", _validate_uniward_epsilon(self.uniward_epsilon)
        )
        object.__setattr__(
            self, "die_cache_interval", _validate_cache_interval(self.die_cache_interval)
        )


class UDIEKLLoss(nn.Module):
    """Substrate-wide U-DIE-KL training-side bolt-on loss.

    The composite is

        total = alpha * UNIWARD_weighted_loss(pred, target)
              + beta  * DIE_weighted_loss(pred, target, scorer_seg, scorer_pose)
              + gamma * kl_distill_segnet_term(pred, target, scorer_seg, T)

    where ``UNIWARD_weighted_loss`` and ``DIE_weighted_loss`` are
    per-pixel-weighted MSE terms (rho(p) and DIE_attention(p) respectively;
    both normalized per-image to mean=1 so the loss scale matches the
    unweighted MSE).

    Substrate trainers add this to their existing standard loss as

        loss = standard_loss + udie_kl_loss(pred_btchw, target_btchw)

    The helper honors:

    * Canonical scorer-preprocess discipline (Catalog #164: routes through
      ``scorer_forward_pair`` and ``scorer.preprocess_input``).
    * EMA compatibility per CLAUDE.md "EMA — NON-NEGOTIABLE": the helper
      adds NO learnable params (scorers are frozen by contract; UNIWARD
      and DIE weights are non-learnable per-pixel multipliers).
    * Compress-side scorer use only per CLAUDE.md "Contest compliance"
      non-negotiable: the helper is INTENDED to be used during training
      (compress side) and MUST NOT be inserted into the inflate runtime
      (the scorer-load-at-inflate violation would push ~73 MB of scorer
      weights into the archive rate).

    Args:
        scorer_seg: contest SegNet; weights MUST be frozen by the caller.
        scorer_pose: contest PoseNet; weights MUST be frozen by the caller.
        alpha: weight on the UNIWARD-weighted loss term (default 0.5).
        beta: weight on the DIE-weighted loss term (default 0.5).
        gamma: weight on the KL distillation term (default 1.0).
        kl_temperature: softmax temperature for KL distillation (default 2.0
            per Quantizr canon).
        uniward_epsilon: UNIWARD wavelet-denominator floor (default 1e-3).
        die_cache_interval: cache the DIE weight map for K iterations
            (default 10; K=1 forces full per-batch recomputation).
    """

    # Canonical instance attribute types (helps static analysis + dataclasses
    # interop without making this class a dataclass — nn.Module ownership of
    # parameters/buffers is not dataclass-friendly).
    config: UDIEKLConfig
    scorer_seg: nn.Module
    scorer_pose: nn.Module

    def __init__(
        self,
        scorer_seg: nn.Module,
        scorer_pose: nn.Module,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        kl_temperature: float = DEFAULT_KL_TEMPERATURE,
        uniward_epsilon: float = DEFAULT_UNIWARD_EPSILON,
        die_cache_interval: int = DEFAULT_DIE_CACHE_INTERVAL,
    ) -> None:
        super().__init__()
        if not isinstance(scorer_seg, nn.Module):
            raise TypeError(
                "scorer_seg must be a torch.nn.Module (e.g. the contest SegNet)"
            )
        if not isinstance(scorer_pose, nn.Module):
            raise TypeError(
                "scorer_pose must be a torch.nn.Module (e.g. the contest PoseNet)"
            )
        self.config = UDIEKLConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            kl_temperature=kl_temperature,
            uniward_epsilon=uniward_epsilon,
            die_cache_interval=die_cache_interval,
        )
        # We do NOT register the scorers as submodules — they are CALLER-
        # owned and CALLER-frozen. Storing them as plain attributes means
        # they don't appear in self.parameters(), don't get state-dict-saved
        # by the substrate trainer, and don't pollute the EMA shadow.
        # Catalog #164 still applies: the helper calls
        # scorer.preprocess_input(...) before scorer(...) every iteration.
        object.__setattr__(self, "scorer_seg", scorer_seg)
        object.__setattr__(self, "scorer_pose", scorer_pose)

        # DIE cache state (tensor-buffer; not a parameter).
        self._die_step: int = 0
        self._die_cached_weight: torch.Tensor | None = None

    def reset_die_cache(self) -> None:
        """Clear the DIE weight cache (e.g., between epochs or on LR change)."""
        self._die_step = 0
        self._die_cached_weight = None

    def _maybe_compute_die_weight(
        self, pred_btchw: torch.Tensor, target_btchw: torch.Tensor
    ) -> torch.Tensor:
        """Return cached DIE weight or recompute if interval elapsed.

        The cache is keyed implicitly by the call sequence; it is ONLY safe
        when the trainer batches in a deterministic order and the predicted
        image moves smoothly. For non-deterministic batching, set
        ``die_cache_interval=1`` to force per-batch recomputation.
        """
        interval = self.config.die_cache_interval
        if (
            self._die_cached_weight is None
            or (self._die_step % interval) == 0
            or self._die_cached_weight.shape != (
                pred_btchw.shape[0],
                pred_btchw.shape[1],
                1,
                pred_btchw.shape[3],
                pred_btchw.shape[4],
            )
            or self._die_cached_weight.device != pred_btchw.device
        ):
            self._die_cached_weight = compute_die_weight_map(
                pred_btchw, target_btchw, self.scorer_seg, self.scorer_pose, detach=True
            )
        self._die_step += 1
        return self._die_cached_weight

    def forward(
        self, pred_btchw: torch.Tensor, target_btchw: torch.Tensor
    ) -> torch.Tensor:
        """Compute the U-DIE-KL composite loss.

        Args:
            pred_btchw: ``(B, T, C, H, W)`` predicted pair tensor.
            target_btchw: ``(B, T, C, H, W)`` ground-truth pair tensor.

        Returns:
            Scalar tensor: ``alpha * UNIWARD_loss + beta * DIE_loss + gamma * KL``.
        """
        if pred_btchw.ndim != 5 or target_btchw.ndim != 5:
            raise ValueError(
                f"pred and target must be (B, T, C, H, W); got pred shape "
                f"{tuple(pred_btchw.shape)}, target shape "
                f"{tuple(target_btchw.shape)}"
            )
        if pred_btchw.shape != target_btchw.shape:
            raise ValueError(
                f"pred and target shapes must match; got {tuple(pred_btchw.shape)} vs "
                f"{tuple(target_btchw.shape)}"
            )

        cfg = self.config
        # Per-pixel residual (B, T, C, H, W).
        residual = (pred_btchw.float() - target_btchw.float()).pow(2)

        total = pred_btchw.new_zeros(())

        # alpha: UNIWARD-weighted loss
        if cfg.alpha > 0.0:
            uniward_w = compute_uniward_weight_map(
                target_btchw, epsilon=cfg.uniward_epsilon, detach=True
            )  # (B, T, 1, H, W)
            uniward_loss = (residual * uniward_w).mean()
            total = total + cfg.alpha * uniward_loss

        # beta: DIE-weighted loss
        if cfg.beta > 0.0:
            die_w = self._maybe_compute_die_weight(pred_btchw, target_btchw)
            die_loss = (residual * die_w).mean()
            total = total + cfg.beta * die_loss

        # gamma: KL distillation
        if cfg.gamma > 0.0:
            kl_term = kl_distill_segnet_term(
                pred_btchw, target_btchw, self.scorer_seg,
                temperature=cfg.kl_temperature,
            )
            total = total + cfg.gamma * kl_term

        return total
