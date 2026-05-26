# SPDX-License-Identifier: MIT
"""NSCS02 score-aware loss with downsample-then-upsample composite path.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
lesson 1 (substrate must be score-aware) + lesson 8 (eval-roundtrip-aware
+ differentiable scorer-preprocess training).

Per the standing directive
``feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md``
this file FORKS the canonical
``tac.substrates.score_aware_common.score_pair_components`` because
NSCS02's optimal score requires the gradient to flow through the
low-resolution renderer and a scorer-compatible resize chain. The exact
contest chain is still a probe target because standalone inflate emits camera
resolution while the training proxy upsamples directly to scorer resolution.

Specifically:
- Canonical helper assumes the renderer outputs at scorer-native
  (384, 512) and the scorer's preprocess is a no-op when the renderer
  already matches.
- NSCS02 outputs at (192, 256). The training proxy path is:
  loss -> SegNet/PoseNet -> scorer preprocess at (384, 512) ->
  differentiable upsample -> renderer.
- The standalone submission runtime currently emits camera resolution
  (874, 1164) before the contest scorer preprocesses back to (384, 512).
  Bilinear/bicubic and direct-vs-camera-intermediate resize choices therefore
  need a no-train resizing-chain ablation before any score-band claim.

Wire-in compliance:
- ``# SCORER_PREPROCESS_HANDLED_OK:nscs02_unique_downsample_upsample_path``
  same-line waiver per Catalog #164 covers each forward call.
- Catalog #220 operational mechanism = OPERATIONAL: the upsample is
  the SCORE-IMPROVING runtime consumption that converts the
  smaller archive bytes into score-relevant frame-pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .architecture import NSCS02DownsampledDecoder

SCORER_HW: tuple[int, int] = (384, 512)


@dataclass(frozen=True)
class NSCS02LossComponents:
    """Per-pair loss components matching the contest scorer formula.

    The contest scorer is ``S = 100 * S_seg + sqrt(10 * S_pose) + 25 * R``
    per ``upstream/evaluate.py:92``. Training proxy loss approximates the
    seg + pose terms; rate is bounded at archive-build time.
    """

    seg_loss: torch.Tensor  # SegNet 5-class argmax disagreement proxy
    pose_loss: torch.Tensor  # PoseNet 6-dim pose MSE
    pixel_loss: torch.Tensor  # auxiliary pixel-MSE for early-epoch stability
    total: torch.Tensor


def compute_nscs02_score_aware_loss(
    decoder: NSCS02DownsampledDecoder,
    z: torch.Tensor,
    target_pair: torch.Tensor,
    seg_scorer: torch.nn.Module | None,
    pose_scorer: torch.nn.Module | None,
    *,
    seg_weight: float = 100.0,
    pose_weight: float = 1.0,
    pixel_weight: float = 0.1,
    upsample_mode: str = "bicubic",
) -> NSCS02LossComponents:
    """Compute NSCS02 score-aware loss for one frame-pair batch.

    Args:
        decoder: NSCS02 5-stage decoder.
        z: per-pair latent (B, latent_dim).
        target_pair: ground-truth frame-pair shaped (B, 2, 3, 384, 512)
            in [0, 255]. The training data is decoded at scorer-native
            (384, 512) per CLAUDE.md "Forbidden synthetic-data" rule.
        seg_scorer: SegNet (or distilled surrogate). May be ``None`` for
            pixel-only smoke; in that case the seg term is zero.
        pose_scorer: PoseNet (or distilled surrogate). May be ``None``
            for pixel-only smoke.

    Returns:
        ``NSCS02LossComponents`` with per-component tensors and a
        weighted-sum ``total`` for ``.backward()``.
    """
    z.shape[0]
    # Render at (192, 256), upsample to scorer-native (384, 512).
    # The composite forward is the proxy for inflate-time semantics.
    rendered_at_scorer = decoder.render_then_upsample_to_scorer(
        z, scorer_hw=SCORER_HW, mode=upsample_mode
    )  # (B, 2, 3, 384, 512)

    pixel_loss = F.mse_loss(rendered_at_scorer, target_pair)

    seg_loss = torch.zeros((), device=z.device)
    pose_loss = torch.zeros((), device=z.device)

    if seg_scorer is not None:
        # SegNet expects (B, T, C, H, W); preprocess_input slices last frame.
        # NSCS02 unique decision: feed both rendered + target through preprocess
        # so the disagreement is computed on identical scorer-side semantics.
        # # SCORER_PREPROCESS_HANDLED_OK:nscs02_unique_downsample_upsample_path
        seg_in_pred = seg_scorer.preprocess_input(rendered_at_scorer)
        with torch.no_grad():
            seg_in_target = seg_scorer.preprocess_input(target_pair)
            seg_target_logits = seg_scorer(seg_in_target)  # SCORER_PREPROCESS_HANDLED_OK:bare_seg_scorer_local_var_called_AFTER_preprocess_input_on_line_above_canonical_pattern_per_comprehensive_bug_audit_cascade_20260526
        seg_pred_logits = seg_scorer(seg_in_pred)  # SCORER_PREPROCESS_HANDLED_OK:bare_seg_scorer_local_var_called_AFTER_preprocess_input_3_lines_above_canonical_pattern_per_comprehensive_bug_audit_cascade_20260526
        # 5-class soft-argmax disagreement via KL-distill (T=2.0 per Quantizr).
        T_distill = 2.0
        log_p_pred = F.log_softmax(seg_pred_logits / T_distill, dim=1)
        p_target = F.softmax(seg_target_logits / T_distill, dim=1)
        seg_loss = F.kl_div(log_p_pred, p_target, reduction="batchmean") * (T_distill * T_distill)

    if pose_scorer is not None:
        # PoseNet expects (B, T, C, H, W) -> preprocess to (B, T*6, H/2, W/2).
        # # SCORER_PREPROCESS_HANDLED_OK:nscs02_unique_downsample_upsample_path
        pose_in_pred = pose_scorer.preprocess_input(rendered_at_scorer)
        with torch.no_grad():
            pose_in_target = pose_scorer.preprocess_input(target_pair)
            pose_target_out = pose_scorer(pose_in_target)  # SCORER_PREPROCESS_HANDLED_OK:bare_pose_scorer_local_var_called_AFTER_preprocess_input_on_line_above_canonical_pattern_per_comprehensive_bug_audit_cascade_20260526
        pose_pred_out = pose_scorer(pose_in_pred)  # SCORER_PREPROCESS_HANDLED_OK:bare_pose_scorer_local_var_called_AFTER_preprocess_input_3_lines_above_canonical_pattern_per_comprehensive_bug_audit_cascade_20260526
        # Pose distortion = MSE on first 6 pose dims (matches upstream).
        pose_loss = sum(
            (pose_pred_out[k][..., : v.shape[-1] // 2] - pose_target_out[k][..., : v.shape[-1] // 2])
            .pow(2)
            .mean()
            for k, v in pose_pred_out.items()
        )

    total = seg_weight * seg_loss + pose_weight * pose_loss + pixel_weight * pixel_loss
    return NSCS02LossComponents(
        seg_loss=seg_loss.detach(),
        pose_loss=pose_loss.detach() if isinstance(pose_loss, torch.Tensor) else torch.tensor(0.0),
        pixel_loss=pixel_loss.detach(),
        total=total,
    )


__all__ = [
    "SCORER_HW",
    "NSCS02LossComponents",
    "compute_nscs02_score_aware_loss",
]
