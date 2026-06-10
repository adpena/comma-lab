# SPDX-License-Identifier: MIT
"""The torch-frozen-scorer <-> MLX-vjp gradient bridge (the faithful score-aware loss).

This is the load-bearing piece of the 1:1 MLX port that lets the MLX decoder
train against the LIVE frozen contest scorer *without* porting EfficientNet-B2
SegNet / FastViT PoseNet to MLX (which is exactly the second-order-autograd NaN
trap that forced the broken harness's learnable-head surrogate — see
``.omx/research/inert_loop_fix_*.md``).

The mechanism (validated to fp32 epsilon by finite-difference in the test suite):

1. The MLX decoder renders the pair (NHWC, ``[0, 255]``).
2. The rendered RGB is handed to the FROZEN torch DistortionNet, which computes
   the PR95 score-aware loss (``100*seg + 1*pose``) the SAME way PR95's
   ``stages/common.py`` does — direct CE/margin through the live SegNet against
   GT-argmax targets + ``sqrt(10*MSE)`` pose. Because the scorer is FROZEN
   (``requires_grad=False``), torch needs only FIRST-order autograd w.r.t. the
   *pixels* (a leaf tensor) — no second-order, no NaN.
3. torch returns ``dL/d(pixels)`` (the cotangent on the MLX render's output).
4. ``mx.vjp`` propagates that cotangent back through the MLX decoder to the
   decoder weights + latents.

The result is the EXACT PR95 score-aware gradient (the scorer half is the exact
torch path; the decoder half is the exact MLX path), with no surrogate and no
NaN. The eval roundtrip (PR95 ``stages/common.py``: bicubic-up 874x1164 ->
bilinear-down 384x512 -> STE round) is applied torch-side so the optimizer sees
the rounded frames the scorer sees (CLAUDE.md "eval_roundtrip" non-negotiable).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training
from tac.score_aware_loop.live_segnet_loss import (
    STAGE_SEG_LOSS_FNS,
    exact_d_seg_from_logits,
    pose_loss,
)

try:  # pragma: no cover - import guard
    import mlx.core as mx
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]

# Camera / scorer resolutions (contest-exact; PR95 ``score.py`` + ``stages/common``).
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)


@dataclass
class ScoreBridgeResult:
    """Outputs of one score-aware bridge evaluation for a batch."""

    loss_value: float
    seg_loss_value: float
    pose_loss_value: float
    d_seg: float
    # cotangent dL/d(rendered pixels), MLX array shaped like the render output.
    pixel_cotangent: Any


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover
        raise RuntimeError("tac.mlx_pr95_port.score_bridge requires mlx.core.")


class TorchScorerBridge:
    """Compute the PR95 score-aware loss + ``dL/d(pixels)`` via the frozen torch scorer.

    Args:
        distortion_net: frozen upstream ``DistortionNet`` (live SegNet + PoseNet).
            All params MUST have ``requires_grad=False`` (fail closed otherwise).
        seg_targets_hard: ``(n_pairs, 384, 512)`` int64 GT SegNet argmax.
        pose_targets: ``(n_pairs, 6)`` float32 GT PoseNet output (or None).
        seg_loss_form: one of the PR95 stage seg-loss family names.
        seg_weight / pose_weight: PR95 ``100*seg + 1*pose`` aggregation.
        eval_roundtrip: apply the bicubic-up/bilinear-down/STE-round inner loop.
        scorer_hw: scorer input (H, W); default contest 384x512.
    """

    def __init__(
        self,
        distortion_net: torch.nn.Module,
        seg_targets_hard: torch.Tensor,
        pose_targets: torch.Tensor | None,
        *,
        seg_loss_form: str = "ce_seg_loss",
        seg_weight: float = 100.0,
        pose_weight: float = 1.0,
        eval_roundtrip: bool = True,
        scorer_hw: tuple[int, int] = SCORER_HW,
        seg_loss_fn: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        for p in distortion_net.parameters():
            if p.requires_grad:
                raise ValueError(
                    "distortion_net has trainable params; the scorer must be "
                    "frozen (requires_grad=False) so the gradient only updates "
                    "the carrier (CLAUDE.md 'Strict scorer rule')."
                )
        self.dnet = distortion_net
        self.seg_targets_hard = seg_targets_hard
        self.pose_targets = pose_targets
        if seg_loss_fn is not None:
            self.seg_loss_fn = seg_loss_fn
        else:
            if seg_loss_form not in STAGE_SEG_LOSS_FNS:
                raise ValueError(
                    f"unknown seg_loss_form {seg_loss_form!r}; "
                    f"known: {sorted(STAGE_SEG_LOSS_FNS)}"
                )
            self.seg_loss_fn = STAGE_SEG_LOSS_FNS[seg_loss_form]
        self.seg_weight = float(seg_weight)
        self.pose_weight = float(pose_weight)
        self.eval_roundtrip = bool(eval_roundtrip)
        self.scorer_hw = (int(scorer_hw[0]), int(scorer_hw[1]))
        self.pose_enabled = pose_targets is not None

    def set_seg_loss_form(self, seg_loss_form: str) -> None:
        """Switch the stage seg-loss (PR95 curriculum stage transition)."""
        if seg_loss_form not in STAGE_SEG_LOSS_FNS:
            raise ValueError(
                f"unknown seg_loss_form {seg_loss_form!r}; "
                f"known: {sorted(STAGE_SEG_LOSS_FNS)}"
            )
        self.seg_loss_fn = STAGE_SEG_LOSS_FNS[seg_loss_form]

    def _to_torch_leaf(self, render_n2chw: Any) -> torch.Tensor:
        """Convert the MLX render ``(B, 2, 3, h, w)`` to a torch leaf tensor."""
        np_render = np.asarray(render_n2chw, dtype=np.float32)
        return torch.tensor(np_render, dtype=torch.float32, requires_grad=True)

    def loss_and_pixel_grad(
        self, render_n2chw: Any, idx: torch.Tensor
    ) -> ScoreBridgeResult:
        """Forward the frozen scorer; return loss + cotangent on the render pixels.

        ``render_n2chw`` is the MLX decoder output ``(B, 2, 3, h, w)`` in
        ``[0, 255]`` (the PR95 N2CHW layout). ``idx`` selects the GT targets.
        """
        _require_mlx()
        leaf = self._to_torch_leaf(render_n2chw)  # (B,2,3,h,w) requires_grad
        b = leaf.shape[0]
        flat = leaf.reshape(b * 2, 3, leaf.shape[-2], leaf.shape[-1])
        # Resize to scorer size (PR95 renders at 384x512 natively; smaller test
        # carriers resize up first).
        if (int(flat.shape[-2]), int(flat.shape[-1])) != self.scorer_hw:
            flat = F.interpolate(
                flat, size=self.scorer_hw, mode="bilinear", align_corners=False
            )
        if self.eval_roundtrip:
            cam_h = max(round(self.scorer_hw[0] * 874 / 384), self.scorer_hw[0] + 1)
            cam_w = max(round(self.scorer_hw[1] * 1164 / 512), self.scorer_hw[1] + 1)
            flat = apply_eval_roundtrip_during_training(
                flat,
                simulate_uint8=True,
                simulate_resize=True,
                ste_round=True,
                target_h=cam_h,
                target_w=cam_w,
            )
        else:
            flat = flat.clamp(0.0, 255.0)
        bchw = flat.reshape(b, 2, 3, self.scorer_hw[0], self.scorer_hw[1])
        bhwc = bchw.permute(0, 1, 3, 4, 2).contiguous()  # (B,2,H,W,C)
        posenet_in, segnet_in = self.dnet.preprocess_input(bhwc)
        seg_out = self.dnet.segnet(segnet_in)
        seg_l = self.seg_loss_fn(seg_out, self.seg_targets_hard[idx])
        total = self.seg_weight * seg_l
        pose_l = torch.tensor(0.0)
        if self.pose_enabled:
            pose_out = self.dnet.posenet(posenet_in)
            pose_pred = pose_out["pose"][:, :6]
            pose_l = pose_loss(pose_pred, self.pose_targets[idx])
            total = total + self.pose_weight * pose_l
        with torch.no_grad():
            d_seg = exact_d_seg_from_logits(seg_out, self.seg_targets_hard[idx])
        total.backward()
        pixel_grad_np = leaf.grad.detach().numpy().astype(np.float32)
        return ScoreBridgeResult(
            loss_value=float(total.detach().item()),
            seg_loss_value=float(seg_l.detach().item()),
            pose_loss_value=float(pose_l.detach().item()) if self.pose_enabled else 0.0,
            d_seg=float(d_seg),
            pixel_cotangent=mx.array(pixel_grad_np),
        )

    @torch.no_grad()
    def exact_d_seg(self, render_n2chw: Any, idx: torch.Tensor) -> float:
        """Return the EXACT live-render d_seg for a batch (no gradient)."""
        _require_mlx()
        np_render = np.asarray(render_n2chw, dtype=np.float32)
        leaf = torch.tensor(np_render, dtype=torch.float32)
        b = leaf.shape[0]
        flat = leaf.reshape(b * 2, 3, leaf.shape[-2], leaf.shape[-1])
        if (int(flat.shape[-2]), int(flat.shape[-1])) != self.scorer_hw:
            flat = F.interpolate(
                flat, size=self.scorer_hw, mode="bilinear", align_corners=False
            )
        if self.eval_roundtrip:
            cam_h = max(round(self.scorer_hw[0] * 874 / 384), self.scorer_hw[0] + 1)
            cam_w = max(round(self.scorer_hw[1] * 1164 / 512), self.scorer_hw[1] + 1)
            flat = apply_eval_roundtrip_during_training(
                flat,
                simulate_uint8=True,
                simulate_resize=True,
                ste_round=True,
                target_h=cam_h,
                target_w=cam_w,
            )
        else:
            flat = flat.clamp(0.0, 255.0)
        bchw = flat.reshape(b, 2, 3, self.scorer_hw[0], self.scorer_hw[1])
        bhwc = bchw.permute(0, 1, 3, 4, 2).contiguous()
        _, segnet_in = self.dnet.preprocess_input(bhwc)
        seg_out = self.dnet.segnet(segnet_in)
        return float(exact_d_seg_from_logits(seg_out, self.seg_targets_hard[idx]))

    @torch.no_grad()
    def exact_d_pose(self, render_n2chw: Any, idx: torch.Tensor) -> float:
        """Return the EXACT live-render d_pose for a batch (the contest pose term).

        The contest PoseNet term is ``MSE(pose_pred[:, :6], pose_target)`` — NOT the
        ``sqrt(10*MSE)`` training surrogate. This returns the raw MSE d_pose so the
        trainer/test can compare against the #74/#80 tube targets directly. Fails
        closed if pose is not enabled (no PoseNet or no targets).
        """
        _require_mlx()
        if not self.pose_enabled:
            raise ValueError(
                "exact_d_pose requires a PoseNet + pose_targets (pose not enabled)."
            )
        np_render = np.asarray(render_n2chw, dtype=np.float32)
        leaf = torch.tensor(np_render, dtype=torch.float32)
        b = leaf.shape[0]
        flat = leaf.reshape(b * 2, 3, leaf.shape[-2], leaf.shape[-1])
        if (int(flat.shape[-2]), int(flat.shape[-1])) != self.scorer_hw:
            flat = F.interpolate(
                flat, size=self.scorer_hw, mode="bilinear", align_corners=False
            )
        if self.eval_roundtrip:
            cam_h = max(round(self.scorer_hw[0] * 874 / 384), self.scorer_hw[0] + 1)
            cam_w = max(round(self.scorer_hw[1] * 1164 / 512), self.scorer_hw[1] + 1)
            flat = apply_eval_roundtrip_during_training(
                flat,
                simulate_uint8=True,
                simulate_resize=True,
                ste_round=True,
                target_h=cam_h,
                target_w=cam_w,
            )
        else:
            flat = flat.clamp(0.0, 255.0)
        bchw = flat.reshape(b, 2, 3, self.scorer_hw[0], self.scorer_hw[1])
        bhwc = bchw.permute(0, 1, 3, 4, 2).contiguous()
        posenet_in, _ = self.dnet.preprocess_input(bhwc)
        pose_out = self.dnet.posenet(posenet_in)
        pose_pred = pose_out["pose"][:, :6]
        return float(F.mse_loss(pose_pred, self.pose_targets[idx]).item())


__all__ = [
    "CAMERA_HW",
    "SCORER_HW",
    "ScoreBridgeResult",
    "TorchScorerBridge",
]
