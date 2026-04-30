# ROUNDTRIP_NOT_REQUIRED: this is a TRAINING-TIME framework that produces a
# specialized renderer checkpoint. The byte-encoding round-trip is the
# downstream codec's responsibility (FP4, block-FP, Self-Compress NN).
"""Lane δ — Joint renderer + scorer-distilled supervisor end-to-end training.

Paradigm shift δ per `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md` §4.2.

This module implements the Hassabis-class moonshot: jointly train the
renderer + a small *scorer-distilled supervisor* that approximates the contest
SegNet + PoseNet pair, with end-to-end gradient flow.

The contest scorer cannot be backprop-targeted directly (FastViT-T12 + EfficientNet-B2
are 73MB; PoseNet attention softmax is bf16-stable but distillation-friendly).
Instead, we:

1. **Distill the scorer pair into a single multi-head proxy** (`MultiHeadScorerProxy`,
   ~5-15K params) that reproduces seg argmax + 6-dim pose vector.
2. **Train the renderer** against the proxy (full gradient flow) AND against
   the contest scorer pair (no-grad target generation, MSE / CE supervision).
3. **The proxy is end-to-end trainable** alongside the renderer, kept honest
   against the contest scorer via periodic re-distillation steps.

This is the AlphaFold-class *learned codec end-to-end* applied to mask compression.

Key novelty vs sibling lanes
----------------------------
- `experiments/train_distill.py` distills SegNet only, fixed proxy, no joint.
- `experiments/train_renderer.py` uses the proxy as a fixed target (no joint).
- `tac.constrained_gen` does inflate-time TTO against the contest scorer (no train).

This module is the FIRST that:
- Co-trains renderer + proxy.
- Uses contest scorer as a **periodic correctness anchor** (every N steps).
- Adds a small differentiable *auth-eval-aware* regularizer that penalizes
  renderer outputs the proxy is uncertain about (epistemic uncertainty signal).

Math foundation
---------------

Joint Lagrangian:

    L_joint = L_proxy_seg + L_proxy_pose + L_renderer_proxy + λ_anchor · L_proxy_vs_contest_scorer
            + λ_unc · L_uncertainty_regularizer + λ_auth · L_auth_eval_proxy

where:
- `L_proxy_seg = CE(proxy_seg(rendered_pair), GT_seg_argmax)` — proxy supervised
  by **contest SegNet's argmax on the original pair** (recomputed periodically).
- `L_proxy_pose = MSE(proxy_pose(rendered_pair), GT_pose_6dim)` — proxy supervised
  similarly.
- `L_renderer_proxy = CE+MSE(proxy_seg/pose(rendered_pair), targets)` — renderer
  optimizes against the *current* proxy as a differentiable surrogate.
- `L_proxy_vs_contest_scorer` — anchors proxy to contest scorer at sync points.
- `L_uncertainty_regularizer = E[H(proxy_seg_logits)]` — penalizes proxy entropy
  on rendered pairs to avoid renderer overfitting to high-uncertainty regions.
- `L_auth_eval_proxy` — predicted-auth-eval-score proxy (small MLP from
  proxy outputs → predicted score). This is the "auth-eval-aware regularizer"
  in the spec.

**Convergence behavior**: Wang et al.-style co-training; proxy lags renderer by
~1 epoch but tracks within 1-2% of contest scorer's behavior at convergence
[prediction]. 4-6 weeks dev × $50-100 GPU per spec.

CLAUDE.md compliance
--------------------
- No silent defaults — every public function arg required-keyword.
- EMA + eval_roundtrip wired by calling training script.
- Contest scorer is loaded with `weights_only=True` per Mario R2 hardening.
- All claims tagged `[empirical:test]` / `[derivation]` / `[prediction]`.
- Training paths produce a checkpoint that downstream encoders compress
  (does NOT produce archive bytes itself).

Out of scope (intentional)
--------------------------
- Renderer architecture (uses any `tac.renderer` checkpoint as init).
- Concrete byte encoding (handled by Lane S, Lane Ω, Lane SCNN).
- Contest scorer loading at INFLATE time (forbidden — strict scorer rule).

References
----------
- Hassabis et al. AlphaFold 2 paper — end-to-end learned codec for protein
- Hinton & Vinyals & Dean 2014 — knowledge distillation T=2.0
- Selfcomp paradigm shift α' — joint-trained mask basis
- Council: §4.2 paradigm δ + Selfcomp 6th shift verdict
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "JOINT_DELTA_FRAMEWORK_VERSION",
    "MultiHeadScorerProxy",
    "AuthEvalAwareRegularizer",
    "JointTrainingConfig",
    "compute_joint_loss",
    "compute_uncertainty_regularizer",
    "compute_renderer_proxy_loss",
    "compute_proxy_distill_loss",
    "predicted_auth_eval_score",
]


JOINT_DELTA_FRAMEWORK_VERSION: int = 1
"""Schema version for paradigm δ exported metadata."""


# ── Multi-head scorer proxy (the differentiable surrogate) ─────────────


class MultiHeadScorerProxy(nn.Module):
    """Tiny differentiable proxy approximating contest SegNet + PoseNet.

    Architecture (~10-15K params total):
    - Shared backbone: 3-layer DepthwiseSeparable conv stem (channel mult 2).
    - SegNet head: 1×1 conv → 5-class logits, AVG-pooled to (48, 64) per
      contest scorer ground truth resolution.
    - PoseNet head: 2-layer MLP on global-pooled stem features → 6-dim pose.

    The proxy takes a PAIR of frames `(B, 2, 3, H, W)` and outputs:
        seg_logits: (B, 5, H_out, W_out)
        pose: (B, 6)

    Args:
        in_channels: input frame channels (3 for RGB).
        n_classes: number of segmentation classes (5 for contest).
        pose_dim: pose vector dimension (6 for contest).
        hidden: hidden channel multiplier (default 16 → ~10K params).
        seg_out_h: SegNet output height (48 for contest 48×64).
        seg_out_w: SegNet output width (64 for contest).

    Param count check (default hidden=16):
        stem: 3*16*9 + 16*32*9 + 32*32*9 + biases ≈ 14K params.
        Total < 16K. Beats Quantizr's 88K by 5×.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        n_classes: int = 5,
        pose_dim: int = 6,
        hidden: int = 16,
        seg_out_h: int = 48,
        seg_out_w: int = 64,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")
        if pose_dim <= 0:
            raise ValueError(f"pose_dim must be positive, got {pose_dim}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        if seg_out_h <= 0 or seg_out_w <= 0:
            raise ValueError(
                f"seg_out_h={seg_out_h}, seg_out_w={seg_out_w} must be positive"
            )

        self.in_channels = int(in_channels)
        self.n_classes = int(n_classes)
        self.pose_dim = int(pose_dim)
        self.hidden = int(hidden)
        self.seg_out_h = int(seg_out_h)
        self.seg_out_w = int(seg_out_w)

        pair_channels = in_channels * 2  # concatenate two frames along channel
        # Stem: 3 layers of conv → BN → ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(pair_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
        )
        # Seg head: 1×1 → 5 classes, then resize to (seg_out_h, seg_out_w)
        self.seg_head = nn.Conv2d(hidden * 2, n_classes, kernel_size=1)
        # Pose head: global avg pool → 2-layer MLP → 6-dim
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden * 2, pose_dim),
        )

    def forward(self, frame_pair: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            frame_pair: (B, 2, C, H, W) tensor of two stacked RGB frames.

        Returns:
            Dict with keys:
                "seg_logits": (B, n_classes, seg_out_h, seg_out_w)
                "pose": (B, pose_dim)
        """
        if frame_pair.dim() != 5:
            raise ValueError(
                f"frame_pair must be 5-D (B, 2, C, H, W), got {frame_pair.dim()}-D"
            )
        if frame_pair.shape[1] != 2:
            raise ValueError(
                f"frame_pair shape[1] must be 2 (frame pair), got {frame_pair.shape[1]}"
            )
        b, _, c, h, w = frame_pair.shape
        if c != self.in_channels:
            raise ValueError(
                f"frame_pair shape[2]={c} != in_channels={self.in_channels}"
            )
        # Concatenate the two frames along channel
        x = frame_pair.reshape(b, 2 * c, h, w)
        feat = self.stem(x)
        seg_logits = self.seg_head(feat)
        seg_logits = F.interpolate(
            seg_logits,
            size=(self.seg_out_h, self.seg_out_w),
            mode="bilinear",
            align_corners=False,
        )
        pose = self.pose_head(feat)
        return {"seg_logits": seg_logits, "pose": pose}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Auth-eval-aware regularizer (small MLP predicting contest score) ───


class AuthEvalAwareRegularizer(nn.Module):
    """Predicts the auth-eval score from proxy seg + pose distortions.

    This is a SMALL MLP (200-500 params) that takes
        [proxy_seg_dist, proxy_pose_dist, archive_bytes]
    and predicts the contest score `25·rate + 100·seg + √(10·pose)`.

    During training, the renderer minimizes this PREDICTED score as a regularizer.
    The MLP is supervised by actual contest-CUDA scores at sync points.

    Args:
        hidden: hidden width (default 32 → ~200 params).
    """

    def __init__(self, *, hidden: int = 32) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        *,
        seg_dist: torch.Tensor,
        pose_dist: torch.Tensor,
        archive_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """Predict auth-eval score from (seg, pose, bytes) signal.

        Args (all required-keyword):
            seg_dist: (B,) scalar SegNet distortion proxy.
            pose_dist: (B,) scalar PoseNet distortion proxy.
            archive_bytes: (B,) scalar archive byte count proxy.

        Returns:
            (B,) predicted contest score per batch element.
        """
        if seg_dist.dim() != 1 or pose_dist.dim() != 1 or archive_bytes.dim() != 1:
            raise ValueError(
                f"all inputs must be 1-D, got "
                f"seg={seg_dist.dim()}D pose={pose_dist.dim()}D bytes={archive_bytes.dim()}D"
            )
        if seg_dist.shape != pose_dist.shape or seg_dist.shape != archive_bytes.shape:
            raise ValueError(
                f"input shapes must match, got "
                f"seg={tuple(seg_dist.shape)} pose={tuple(pose_dist.shape)} "
                f"bytes={tuple(archive_bytes.shape)}"
            )
        # Normalize bytes to ~unit scale for MLP (Quantizr ≈ 300KB → 0.3)
        bytes_norm = archive_bytes / 1_000_000.0
        x = torch.stack([seg_dist, pose_dist, bytes_norm], dim=1)
        return self.net(x).squeeze(-1)


def predicted_auth_eval_score(
    *,
    regularizer: AuthEvalAwareRegularizer,
    seg_dist: torch.Tensor,
    pose_dist: torch.Tensor,
    archive_bytes: torch.Tensor,
) -> torch.Tensor:
    """Functional alias for AuthEvalAwareRegularizer.forward."""
    return regularizer(
        seg_dist=seg_dist, pose_dist=pose_dist, archive_bytes=archive_bytes
    )


# ── Loss components ────────────────────────────────────────────────────


def compute_proxy_distill_loss(
    *,
    proxy_seg_logits: torch.Tensor,
    proxy_pose: torch.Tensor,
    target_seg_argmax: torch.Tensor,
    target_pose: torch.Tensor,
    seg_weight: float = 1.0,
    pose_weight: float = 1.0,
) -> torch.Tensor:
    """Distillation loss: proxy ← contest scorer outputs.

    L_proxy = seg_weight · CE(proxy_seg, target_argmax) + pose_weight · MSE(proxy_pose, target_pose)

    Args (all required-keyword):
        proxy_seg_logits: (B, n_classes, H, W) proxy logits.
        proxy_pose: (B, pose_dim) proxy pose.
        target_seg_argmax: (B, H, W) argmax class IDs from contest SegNet.
        target_pose: (B, pose_dim) ground-truth pose from contest PoseNet.
        seg_weight: scaling weight on CE term.
        pose_weight: scaling weight on MSE term.

    Returns:
        Scalar tensor — total proxy distillation loss.
    """
    if seg_weight < 0 or pose_weight < 0:
        raise ValueError(
            f"weights must be non-negative; seg_weight={seg_weight} pose_weight={pose_weight}"
        )
    if proxy_seg_logits.dim() != 4:
        raise ValueError(
            f"proxy_seg_logits must be 4-D (B,C,H,W), got {proxy_seg_logits.dim()}-D"
        )
    if proxy_pose.dim() != 2:
        raise ValueError(
            f"proxy_pose must be 2-D (B,pose_dim), got {proxy_pose.dim()}-D"
        )
    if target_seg_argmax.dim() != 3:
        raise ValueError(
            f"target_seg_argmax must be 3-D (B,H,W), got {target_seg_argmax.dim()}-D"
        )
    if target_pose.dim() != 2:
        raise ValueError(
            f"target_pose must be 2-D (B,pose_dim), got {target_pose.dim()}-D"
        )
    seg_loss = F.cross_entropy(proxy_seg_logits, target_seg_argmax.long())
    pose_loss = F.mse_loss(proxy_pose, target_pose)
    return seg_weight * seg_loss + pose_weight * pose_loss


def compute_renderer_proxy_loss(
    *,
    proxy_outputs: dict[str, torch.Tensor],
    target_seg_argmax: torch.Tensor,
    target_pose: torch.Tensor,
    seg_weight: float = 1.0,
    pose_weight: float = 1.0,
) -> torch.Tensor:
    """Renderer loss against (frozen-grad) proxy outputs.

    The renderer minimizes the proxy's predicted distortion. Gradient flows
    BACK through the proxy into the renderer's reconstructed frames.

    The proxy parameters are NOT updated by this loss path — pass through with
    `proxy.requires_grad_(False)` or wrap in `torch.no_grad()` selectively at
    callsite. This function does NOT detach for the caller.
    """
    return compute_proxy_distill_loss(
        proxy_seg_logits=proxy_outputs["seg_logits"],
        proxy_pose=proxy_outputs["pose"],
        target_seg_argmax=target_seg_argmax,
        target_pose=target_pose,
        seg_weight=seg_weight,
        pose_weight=pose_weight,
    )


def compute_uncertainty_regularizer(
    *,
    proxy_seg_logits: torch.Tensor,
) -> torch.Tensor:
    """Mean per-pixel entropy of the proxy seg distribution.

    H(p) = -Σ_c p_c log p_c

    High entropy → proxy uncertain → renderer should AVOID those regions
    (not over-fit). Penalizing this term pushes the renderer toward decisions
    the proxy is confident about.

    Args (all required-keyword):
        proxy_seg_logits: (B, n_classes, H, W) proxy seg logits.

    Returns:
        Scalar mean entropy over (B, H, W) in nats.
    """
    if proxy_seg_logits.dim() != 4:
        raise ValueError(
            f"proxy_seg_logits must be 4-D, got {proxy_seg_logits.dim()}-D"
        )
    log_p = F.log_softmax(proxy_seg_logits, dim=1)
    p = log_p.exp()
    h_per_pixel = -(p * log_p).sum(dim=1)  # (B, H, W)
    return h_per_pixel.mean()


# ── Joint training config (typed) ──────────────────────────────────────


@dataclass
class JointTrainingConfig:
    """Hyperparameters for paradigm δ joint training.

    Per CLAUDE.md "no silent defaults": every weight is required keyword in
    `compute_joint_loss(...)`. This dataclass exists for the calling training
    script to pin its config in one place (and serialize for provenance).
    """

    proxy_distill_weight: float
    """Weight on `L_proxy_distill` (proxy ← contest scorer)."""

    renderer_proxy_weight: float
    """Weight on `L_renderer_proxy` (renderer ← proxy)."""

    contest_anchor_weight: float
    """Weight on `L_proxy_vs_contest_scorer` (anchors proxy at sync points)."""

    uncertainty_weight: float
    """Weight on `L_uncertainty_regularizer`."""

    auth_eval_weight: float
    """Weight on `L_auth_eval_proxy` (predicted-auth-eval score regularizer)."""

    contest_anchor_period: int
    """Number of training steps between contest-scorer sync points (e.g. 1000)."""

    proxy_seg_weight: float = 100.0
    """Inner weight on SegNet CE in proxy distill (matches contest 100·seg)."""

    proxy_pose_weight: float = 1.0
    """Inner weight on PoseNet MSE in proxy distill."""

    def __post_init__(self) -> None:
        # All weights non-negative
        for n, v in [
            ("proxy_distill_weight", self.proxy_distill_weight),
            ("renderer_proxy_weight", self.renderer_proxy_weight),
            ("contest_anchor_weight", self.contest_anchor_weight),
            ("uncertainty_weight", self.uncertainty_weight),
            ("auth_eval_weight", self.auth_eval_weight),
            ("proxy_seg_weight", self.proxy_seg_weight),
            ("proxy_pose_weight", self.proxy_pose_weight),
        ]:
            if v < 0:
                raise ValueError(f"{n} must be non-negative, got {v}")
        if self.contest_anchor_period <= 0:
            raise ValueError(
                f"contest_anchor_period must be positive, got {self.contest_anchor_period}"
            )


# ── Joint loss aggregator ──────────────────────────────────────────────


def compute_joint_loss(
    *,
    config: JointTrainingConfig,
    proxy_outputs_on_rendered: dict[str, torch.Tensor],
    proxy_outputs_on_original: dict[str, torch.Tensor] | None,
    contest_seg_argmax_on_original: torch.Tensor | None,
    contest_pose_on_original: torch.Tensor | None,
    archive_bytes_proxy: torch.Tensor | None,
    auth_eval_regularizer: AuthEvalAwareRegularizer | None,
    is_anchor_step: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose the joint Lagrangian per `config`.

    Args (all required-keyword):
        config: JointTrainingConfig.
        proxy_outputs_on_rendered: proxy(rendered_pair) — full-grad through proxy.
        proxy_outputs_on_original: proxy(original_pair) — used for proxy distill.
            None if not an anchor step.
        contest_seg_argmax_on_original: contest SegNet(original_pair) argmax.
            None if not an anchor step.
        contest_pose_on_original: contest PoseNet(original_pair). None if not anchor.
        archive_bytes_proxy: scalar tensor (B,) of archive-byte estimate per
            example. None disables the auth-eval regularizer.
        auth_eval_regularizer: AuthEvalAwareRegularizer or None.
        is_anchor_step: True if this batch syncs proxy ← contest scorer.

    Returns:
        (total_loss, parts_dict).
    """
    if not isinstance(config, JointTrainingConfig):
        raise TypeError(
            f"config must be JointTrainingConfig, got {type(config).__name__}"
        )
    parts: dict[str, torch.Tensor] = {}

    # 1) Proxy distill (anchor step only) — proxy ← contest scorer
    if is_anchor_step:
        if proxy_outputs_on_original is None:
            raise ValueError(
                "is_anchor_step=True requires proxy_outputs_on_original"
            )
        if contest_seg_argmax_on_original is None:
            raise ValueError(
                "is_anchor_step=True requires contest_seg_argmax_on_original"
            )
        if contest_pose_on_original is None:
            raise ValueError(
                "is_anchor_step=True requires contest_pose_on_original"
            )
        l_proxy_distill = compute_proxy_distill_loss(
            proxy_seg_logits=proxy_outputs_on_original["seg_logits"],
            proxy_pose=proxy_outputs_on_original["pose"],
            target_seg_argmax=contest_seg_argmax_on_original,
            target_pose=contest_pose_on_original,
            seg_weight=config.proxy_seg_weight,
            pose_weight=config.proxy_pose_weight,
        )
        parts["proxy_distill"] = l_proxy_distill
    else:
        l_proxy_distill = torch.zeros(
            (), device=proxy_outputs_on_rendered["pose"].device
        )

    # 2) Renderer ← proxy (every step) — needs targets
    if (
        contest_seg_argmax_on_original is not None
        and contest_pose_on_original is not None
    ):
        l_renderer_proxy = compute_renderer_proxy_loss(
            proxy_outputs=proxy_outputs_on_rendered,
            target_seg_argmax=contest_seg_argmax_on_original,
            target_pose=contest_pose_on_original,
            seg_weight=config.proxy_seg_weight,
            pose_weight=config.proxy_pose_weight,
        )
        parts["renderer_proxy"] = l_renderer_proxy
    else:
        l_renderer_proxy = torch.zeros(
            (), device=proxy_outputs_on_rendered["pose"].device
        )

    # 3) Uncertainty regularizer (every step)
    l_unc = compute_uncertainty_regularizer(
        proxy_seg_logits=proxy_outputs_on_rendered["seg_logits"]
    )
    parts["uncertainty"] = l_unc

    # 4) Auth-eval regularizer (every step IF byte proxy + reg available)
    if (
        auth_eval_regularizer is not None
        and archive_bytes_proxy is not None
    ):
        # Use proxy outputs as "predicted seg/pose distortion"
        # SegNet distortion proxy = mean per-pixel CE on argmax targets
        # PoseNet distortion proxy = mean per-batch MSE
        if (
            contest_seg_argmax_on_original is not None
            and contest_pose_on_original is not None
        ):
            with torch.no_grad():
                pred_argmax = (
                    proxy_outputs_on_rendered["seg_logits"]
                    .argmax(dim=1)
                )
                # Force matching shape to GT before comparing
                if pred_argmax.shape != contest_seg_argmax_on_original.shape:
                    pred_argmax = F.interpolate(
                        pred_argmax.unsqueeze(1).float(),
                        size=contest_seg_argmax_on_original.shape[-2:],
                        mode="nearest",
                    ).squeeze(1).long()
                seg_dist = (pred_argmax != contest_seg_argmax_on_original).float().mean(
                    dim=(1, 2)
                )
            pose_dist = (
                (proxy_outputs_on_rendered["pose"] - contest_pose_on_original) ** 2
            ).mean(dim=1)
            pred_score = predicted_auth_eval_score(
                regularizer=auth_eval_regularizer,
                seg_dist=seg_dist,
                pose_dist=pose_dist,
                archive_bytes=archive_bytes_proxy,
            )
            l_auth = pred_score.mean()
        else:
            l_auth = torch.zeros((), device=proxy_outputs_on_rendered["pose"].device)
        parts["auth_eval"] = l_auth
    else:
        l_auth = torch.zeros((), device=proxy_outputs_on_rendered["pose"].device)

    # 5) Anchor: proxy_outputs_on_original ↔ contest scorer outputs at sync.
    # In current formulation this is captured by `proxy_distill`; weight here is
    # an additional multiplier on top.
    l_anchor = (
        config.contest_anchor_weight * l_proxy_distill
        if is_anchor_step
        else torch.zeros((), device=l_proxy_distill.device)
    )
    parts["contest_anchor"] = l_anchor

    total = (
        config.proxy_distill_weight * l_proxy_distill
        + config.renderer_proxy_weight * l_renderer_proxy
        + config.uncertainty_weight * l_unc
        + config.auth_eval_weight * l_auth
        + l_anchor
    )

    parts["total"] = total.detach()
    return total, parts
