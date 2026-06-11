# SPDX-License-Identifier: MIT
"""The MLX-GPU end-to-end score-aware loss (the fast, on-GPU sibling of the bridge).

This is the throughput unlock for the 600-pair capstone gate. The standing
``TorchScorerBridge`` (``tac.mlx_pr95_port.score_bridge``) runs the frozen
EfficientNet-B2 SegNet + FastViT PoseNet forward+backward on **torch-CPU** — on
Apple-Silicon arm64 torch has no mkldnn so the depthwise convs dispatch to the
naive ``aten::_slow_conv2d_forward`` reference kernel, which is the
>97%-of-wall-clock bottleneck (~18 min/epoch at 600 pairs). This module instead
runs the SAME score-aware loss end-to-end on the **MLX GPU**: the render is
preprocessed (eval_roundtrip + per-frame resize + rgb_to_yuv6) in differentiable
MLX, the FULL MLX SegNet/PoseNet port (``MLXDistortionScorerAdapter``) computes
seg+pose loss, and ``mx.value_and_grad`` propagates ``dL/d(pixels)`` back through
the scorer to the render — never leaving the GPU.

**Authority discipline (CLAUDE.md "NO FAKE" class 8 + the MLX authority ladder).**
torch-CPU exact is the ONLY authority. The MLX scorer is a *fast training signal*,
not a score axis. Per the 2026-06-11 drift audit
(``.omx/research/mlx_scorer_port_drift_audit_20260611.md``):

  * MLX-CPU is bit-faithful to torch-CPU for the charged quantities (2 argmax
    flips / 19.66M, both at fp32-ULP ties; pose component drift ~8.7e-11). It is
    the strict parity reference.
  * MLX-GPU (Metal) drifts on d_seg by ~1.2e-5 (243 flips / 19.66M, all confined
    to SegNet decision-boundary near-ties) — negligible for the per-step training
    gradient.
  * MLX-GPU pose drifts ~2.76e-4, which can EXCEED the frontier d_pose (~3.4e-5).
    So MLX-GPU pose is a relative training signal ONLY; the ABSOLUTE d_pose near
    the frontier MUST be read back on torch-CPU.

The reusable contract therefore mirrors ``TorchScorerBridge`` exactly
(``loss_and_pixel_grad`` / ``exact_d_seg`` / ``exact_d_pose`` /
``fused_d_seg_d_pose``) so the trainer can swap backends behind a flag with NO
authority change: the MLX-GPU path drives the per-step gradient; the trainer
re-scores on the torch-CPU bridge for every reported metric (the
``authority_recheck_every`` cadence + every eval).

Reuses (does NOT re-implement): the canonical differentiable MLX preprocessing
(``tac.local_acceleration.pr95_hnerv_mlx_training.apply_eval_roundtrip_nhwc`` +
``rgb_to_yuv6_mlx`` + ``resize_nhwc_align_corners_false``), the full MLX scorer
port (``tac.local_acceleration.mlx_scorer_adapters.MLXDistortionScorerAdapter``),
and the score-aware loss family (``tac.score_aware_loop.live_segnet_loss``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch  # noqa: TC002 - used at runtime (idx.detach(), Module.parameters())

from tac.mlx_pr95_port.score_bridge import CAMERA_HW, SCORER_HW, ScoreBridgeResult

try:  # pragma: no cover - import guard
    import mlx.core as mx
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover
        raise RuntimeError("tac.mlx_pr95_port.mlx_gpu_score_bridge requires mlx.core.")


def _resolve_mlx_seg_loss_fns() -> dict[str, Callable[..., Any]]:
    """Canonical MLX stage seg-loss family (REUSE, not re-implement).

    These are the ALREADY parity-tested NCHW-layout MLX seg-losses from
    ``tac.mlx_pr95_port.mlx_losses`` (verified 1:1 with the torch PR95 family in
    ``test_torch_parity.py::test_loss_parity_all_stage_seg_losses_fp32_exact``).
    The MLX scorer adapter emits NHWC logits ``(B,H,W,C)``; this bridge transposes
    to NCHW ``(B,C,H,W)`` before calling them so the seg-loss math is the single
    canonical source of truth (CLAUDE.md UNIQUE-AND-COMPLETE: share the bolt-on
    seg-loss; the bridge is the substrate-specific wiring).
    """
    from tac.mlx_pr95_port.mlx_losses import (
        ce_seg_loss_mlx,
        l7_softplus_seg_loss_mlx,
        smooth_disagreement_seg_loss_mlx,
        tau_softplus_seg_loss_mlx,
    )

    return {
        "ce_seg_loss": ce_seg_loss_mlx,
        "tau_softplus_seg_loss": tau_softplus_seg_loss_mlx,
        "smooth_disagreement_seg_loss": smooth_disagreement_seg_loss_mlx,
        "l7_softplus_seg_loss": l7_softplus_seg_loss_mlx,
    }


# Canonical name set (mirrors STAGE_SEG_LOSS_FNS). The callables are resolved
# lazily (the import requires mlx) via :func:`_resolve_mlx_seg_loss_fns`.
MLX_STAGE_SEG_LOSS_FORMS: tuple[str, ...] = (
    "ce_seg_loss",
    "tau_softplus_seg_loss",
    "smooth_disagreement_seg_loss",
    "l7_softplus_seg_loss",
)


class MLXGpuScorerBridge:
    """Compute the PR95 score-aware loss + ``dL/d(pixels)`` end-to-end on the MLX GPU.

    Drop-in fast sibling of :class:`tac.mlx_pr95_port.score_bridge.TorchScorerBridge`.
    The constructor args + the public methods (``loss_and_pixel_grad`` /
    ``exact_d_seg`` / ``exact_d_pose`` / ``fused_d_seg_d_pose`` /
    ``set_seg_loss_form``) mirror the torch bridge so the trainer swaps backends
    behind a flag. The difference: forward AND backward run in MLX on the GPU.

    **NOT an authority.** Every number this bridge returns is
    ``[macOS-MLX research-signal]`` — a fast training signal. The trainer must
    re-score on the torch-CPU bridge for any reported/promoted metric (the
    ``authority_recheck_every`` cadence). Near the frontier, ``exact_d_pose`` MUST
    come from the torch-CPU bridge (the GPU pose drift ~2.76e-4 can swamp a
    frontier d_pose ~3.4e-5).

    Args:
        distortion_net: frozen upstream ``DistortionNet`` (live SegNet + PoseNet),
            converted ONCE to the MLX adapter. All params MUST be frozen.
        seg_targets_hard: ``(n_pairs, 384, 512)`` int64 GT SegNet argmax (torch).
        pose_targets: ``(n_pairs, 6)`` float32 GT PoseNet output (torch) or None.
        seg_loss_form / seg_weight / pose_weight / eval_roundtrip / scorer_hw:
            same semantics as ``TorchScorerBridge``.
        device_type: ``"gpu"`` (the fast path) or ``"cpu"`` (the bit-faithful
            reference path — used by the parity validation).
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
        device_type: str = "gpu",
        num_classes: int = 5,
    ) -> None:
        _require_mlx()
        from tac.local_acceleration.mlx_scorer_adapters import (
            torch_distortion_net_to_mlx,
        )

        for p in distortion_net.parameters():
            if p.requires_grad:
                raise ValueError(
                    "distortion_net has trainable params; the scorer must be "
                    "frozen (requires_grad=False) so the gradient only updates "
                    "the carrier (CLAUDE.md 'Strict scorer rule')."
                )
        if device_type not in {"cpu", "gpu"}:
            raise ValueError(f"device_type must be 'cpu' or 'gpu', got {device_type!r}")
        self._seg_loss_fns = _resolve_mlx_seg_loss_fns()
        if seg_loss_form not in self._seg_loss_fns:
            raise ValueError(
                f"unknown seg_loss_form {seg_loss_form!r}; "
                f"known: {sorted(self._seg_loss_fns)}"
            )
        self.device_type = device_type
        # Build the MLX adapter under the target device so its weight arrays live
        # there (the adapter copies torch weights into mx.array at construction).
        from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

        with temporary_mlx_device(device_type):
            self.adapter = torch_distortion_net_to_mlx(distortion_net)
        # GT targets as MLX arrays (seg int32 (N,384,512); pose float32 (N,6)).
        self.seg_targets_mx = mx.array(
            np.asarray(seg_targets_hard.detach().cpu().numpy(), dtype=np.int32)
        )
        self.pose_enabled = pose_targets is not None
        self.pose_targets_mx = (
            mx.array(np.asarray(pose_targets.detach().cpu().numpy(), dtype=np.float32))
            if pose_targets is not None
            else None
        )
        self.seg_loss_form = seg_loss_form
        self.seg_weight = float(seg_weight)
        self.pose_weight = float(pose_weight)
        self.eval_roundtrip = bool(eval_roundtrip)
        self.scorer_hw = (int(scorer_hw[0]), int(scorer_hw[1]))
        self.num_classes = int(num_classes)
        self.camera_hw = CAMERA_HW

    def set_seg_loss_form(self, seg_loss_form: str) -> None:
        """Switch the stage seg-loss (PR95 curriculum stage transition)."""
        if seg_loss_form not in self._seg_loss_fns:
            raise ValueError(
                f"unknown seg_loss_form {seg_loss_form!r}; "
                f"known: {sorted(self._seg_loss_fns)}"
            )
        self.seg_loss_form = seg_loss_form

    # -- shared differentiable MLX preprocess (the bridge's torch preamble) ----

    def _preprocess_render_to_scorer_inputs(self, render_n2chw: Any) -> tuple[Any, Any]:
        """Render ``(B,2,3,h,w)`` MLX -> (posenet_yuv6 NHWC, segnet_rgb NHWC).

        Mirrors ``TorchScorerBridge.loss_and_pixel_grad`` preamble exactly, but in
        differentiable MLX so the gradient flows back to ``render_n2chw``:

          1. N2CHW -> (B*2, h, w, 3) NHWC.
          2. resize to scorer HW (bilinear, align_corners=False) if needed.
          3. eval_roundtrip: bicubic-up to camera, bilinear-down to scorer, uint8
             STE (``apply_eval_roundtrip_nhwc``); else clip to [0,255].
          4. SegNet input = frame1 RGB only ``(B, H, W, 3)`` (upstream uses the
             LAST frame ``x[:, -1]``).
          5. PoseNet input = per-frame rgb_to_yuv6 then concat the two frames' 6
             channels -> ``(B, H//2, W//2, 12)`` (upstream
             ``rearrange '(b t) c h w -> b (t c) h w'``).
        """
        from tac.local_acceleration.pr95_hnerv_mlx_training import (
            apply_eval_roundtrip_nhwc,
            resize_nhwc_align_corners_false,
            rgb_to_yuv6_mlx,
        )

        b = int(render_n2chw.shape[0])
        h = int(render_n2chw.shape[-2])
        w = int(render_n2chw.shape[-1])
        # N2CHW -> (B,2,h,w,3) -> (B*2,h,w,3)
        nhwc = mx.transpose(render_n2chw, (0, 1, 3, 4, 2))
        flat = mx.reshape(nhwc, (b * 2, h, w, 3))
        sh, sw = self.scorer_hw
        if (h, w) != (sh, sw):
            flat = resize_nhwc_align_corners_false(flat, size=(sh, sw), mode="bilinear")
        if self.eval_roundtrip:
            flat = apply_eval_roundtrip_nhwc(
                flat,
                camera_hw=self.camera_hw,
                output_hw=(sh, sw),
                simulate_resize=True,
                simulate_uint8=True,
                ste_round=True,
            )
        else:
            flat = mx.clip(flat, 0.0, 255.0)
        # (B*2, H, W, 3) -> (B, 2, H, W, 3)
        pair_nhwc = mx.reshape(flat, (b, 2, sh, sw, 3))
        # SegNet: last frame only.
        segnet_rgb = pair_nhwc[:, 1, :, :, :]  # (B, H, W, 3)
        # PoseNet: per-frame yuv6 then concat the 2 frames' 6 channels -> 12.
        yuv6 = rgb_to_yuv6_mlx(pair_nhwc)  # (B, 2, H//2, W//2, 6)
        h2 = int(yuv6.shape[2])
        w2 = int(yuv6.shape[3])
        # (B, 2, h2, w2, 6) -> (B, h2, w2, 2, 6) -> (B, h2, w2, 12)
        posenet_yuv6 = mx.reshape(
            mx.transpose(yuv6, (0, 2, 3, 1, 4)), (b, h2, w2, 12)
        )
        return posenet_yuv6, segnet_rgb

    def _seg_loss(self, segnet_rgb: Any, idx: Any) -> Any:
        seg_logits_nhwc = self.adapter.segnet(segnet_rgb)  # NHWC (B,H,W,C)
        # NHWC -> NCHW for the canonical NCHW MLX seg-loss family (single source
        # of truth; verified 1:1 with torch PR95 in test_torch_parity).
        seg_logits_nchw = mx.transpose(seg_logits_nhwc, (0, 3, 1, 2))
        targets = self.seg_targets_mx[idx]  # (B,H,W)
        return self._seg_loss_fns[self.seg_loss_form](seg_logits_nchw, targets)

    def _pose_loss(self, posenet_yuv6: Any, idx: Any) -> Any:
        from tac.mlx_pr95_port.mlx_losses import pose_loss_mlx

        pose_out = self.adapter.posenet(posenet_yuv6)["pose"][..., :6]
        return pose_loss_mlx(pose_out, self.pose_targets_mx[idx])

    # -- public contract (mirrors TorchScorerBridge) ---------------------------

    def loss_and_pixel_grad(
        self, render_n2chw: Any, idx: torch.Tensor
    ) -> ScoreBridgeResult:
        """Forward the MLX scorer; return loss + cotangent on the render pixels.

        ``render_n2chw`` is the MLX decoder output ``(B,2,3,h,w)`` in ``[0,255]``.
        ``idx`` selects the GT targets. The pixel cotangent is ``dL/d(render)`` —
        the SAME quantity the torch bridge returns — computed via
        ``mx.value_and_grad`` through the full MLX scorer.
        """
        _require_mlx()
        from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

        idx_mx = mx.array(np.asarray(idx.detach().cpu().numpy(), dtype=np.int32))
        seg_weight = self.seg_weight
        pose_weight = self.pose_weight
        pose_enabled = self.pose_enabled
        targets = self.seg_targets_mx[idx_mx]

        from tac.mlx_pr95_port.mlx_losses import pose_loss_mlx

        # The gradient closure returns ONLY the scalar ``total`` (the optimizer
        # objective). MLX 0.31.1 has no ``has_aux``; capturing the seg/pose
        # sub-scalars from inside ``loss_fn`` and evaluating them AFTER
        # ``value_and_grad`` would RE-RUN the forward graph (a 2nd full scorer
        # forward — the source of the slow-test timeout). So telemetry (seg/pose
        # breakdown + d_seg) is computed in ONE separate ``stop_gradient`` forward
        # below — exactly one extra forward, not three recomputes.
        def loss_fn(render: Any) -> Any:
            posenet_yuv6, segnet_rgb = self._preprocess_render_to_scorer_inputs(render)
            seg_logits_nhwc = self.adapter.segnet(segnet_rgb)
            seg_logits_nchw = mx.transpose(seg_logits_nhwc, (0, 3, 1, 2))
            total = seg_weight * self._seg_loss_fns[self.seg_loss_form](
                seg_logits_nchw, targets
            )
            if pose_enabled:
                pose_out = self.adapter.posenet(posenet_yuv6)["pose"][..., :6]
                total = total + pose_weight * pose_loss_mlx(
                    pose_out, self.pose_targets_mx[idx_mx]
                )
            return total

        with temporary_mlx_device(self.device_type):
            loss_val, pixel_grad = mx.value_and_grad(loss_fn)(render_n2chw)
            mx.eval(loss_val, pixel_grad)
            # ONE no-grad telemetry forward for the seg/pose/d_seg breakdown.
            posenet_yuv6, segnet_rgb = self._preprocess_render_to_scorer_inputs(
                mx.stop_gradient(render_n2chw)
            )
            seg_logits_nhwc = mx.stop_gradient(self.adapter.segnet(segnet_rgb))
            seg_logits_nchw = mx.transpose(seg_logits_nhwc, (0, 3, 1, 2))
            seg_l = self._seg_loss_fns[self.seg_loss_form](seg_logits_nchw, targets)
            seg_pred = mx.argmax(seg_logits_nhwc, axis=-1)
            d_seg_arr = mx.mean((seg_pred != targets).astype(mx.float32))
            if pose_enabled:
                pose_out = mx.stop_gradient(
                    self.adapter.posenet(posenet_yuv6)["pose"][..., :6]
                )
                pose_l = pose_loss_mlx(pose_out, self.pose_targets_mx[idx_mx])
            else:
                pose_l = mx.array(0.0)
            mx.eval(seg_l, pose_l, d_seg_arr)
            seg_loss_value = float(np.asarray(seg_l))
            pose_loss_value = float(np.asarray(pose_l)) if pose_enabled else 0.0
            d_seg = float(np.asarray(d_seg_arr))
        return ScoreBridgeResult(
            loss_value=float(np.asarray(loss_val)),
            seg_loss_value=seg_loss_value,
            pose_loss_value=pose_loss_value,
            d_seg=d_seg,
            pixel_cotangent=pixel_grad,
        )

    def _exact_d_seg_inner(self, render_n2chw: Any, idx_mx: Any) -> float:
        """Exact SegNet argmax-disagreement rate on the MLX render (no grad)."""
        _, segnet_rgb = self._preprocess_render_to_scorer_inputs(render_n2chw)
        seg_logits = mx.stop_gradient(self.adapter.segnet(segnet_rgb))
        pred = mx.argmax(seg_logits, axis=-1)  # (B,H,W)
        targets = self.seg_targets_mx[idx_mx]
        mx.eval(pred)
        diff = (pred != targets).astype(mx.float32)
        return float(np.asarray(mx.mean(diff)))

    def exact_d_seg(self, render_n2chw: Any, idx: torch.Tensor) -> float:
        """Return the MLX-render d_seg for a batch (research-signal, NOT authority)."""
        _require_mlx()
        from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

        idx_mx = mx.array(np.asarray(idx.detach().cpu().numpy(), dtype=np.int32))
        with temporary_mlx_device(self.device_type):
            return self._exact_d_seg_inner(render_n2chw, idx_mx)

    def exact_d_pose(self, render_n2chw: Any, idx: torch.Tensor) -> float:
        """Return the MLX-render raw PoseNet MSE d_pose (research-signal ONLY).

        WARNING: near the frontier (d_pose ~ 3e-5) the MLX-GPU pose drift
        (~2.76e-4) can EXCEED the signal. Use the torch-CPU bridge for the
        absolute d_pose authority. This exists for fast relative monitoring.
        """
        _require_mlx()
        if not self.pose_enabled:
            raise ValueError(
                "exact_d_pose requires a PoseNet + pose_targets (pose not enabled)."
            )
        from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

        idx_mx = mx.array(np.asarray(idx.detach().cpu().numpy(), dtype=np.int32))
        with temporary_mlx_device(self.device_type):
            posenet_yuv6, _ = self._preprocess_render_to_scorer_inputs(render_n2chw)
            pose_out = mx.stop_gradient(
                self.adapter.posenet(posenet_yuv6)["pose"][..., :6]
            )
            mse = mx.mean(mx.square(pose_out - self.pose_targets_mx[idx_mx]))
            mx.eval(mse)
            return float(np.asarray(mse))

    def fused_d_seg_d_pose(
        self, render_n2chw: Any, idx: torch.Tensor
    ) -> tuple[float, float]:
        """Return ``(d_seg, d_pose)`` from one shared MLX preprocess (research-signal)."""
        _require_mlx()
        if not self.pose_enabled:
            raise ValueError(
                "fused_d_seg_d_pose requires a PoseNet + pose_targets (pose not "
                "enabled); use exact_d_seg for the seg-only path."
            )
        from tac.local_acceleration.mlx_scorer_adapters import temporary_mlx_device

        idx_mx = mx.array(np.asarray(idx.detach().cpu().numpy(), dtype=np.int32))
        with temporary_mlx_device(self.device_type):
            posenet_yuv6, segnet_rgb = self._preprocess_render_to_scorer_inputs(
                render_n2chw
            )
            seg_logits = mx.stop_gradient(self.adapter.segnet(segnet_rgb))
            pred = mx.argmax(seg_logits, axis=-1)
            diff = (pred != self.seg_targets_mx[idx_mx]).astype(mx.float32)
            d_seg = mx.mean(diff)
            pose_out = mx.stop_gradient(
                self.adapter.posenet(posenet_yuv6)["pose"][..., :6]
            )
            d_pose = mx.mean(mx.square(pose_out - self.pose_targets_mx[idx_mx]))
            mx.eval(d_seg, d_pose)
            return float(np.asarray(d_seg)), float(np.asarray(d_pose))


__all__ = [
    "MLX_STAGE_SEG_LOSS_FORMS",
    "MLXGpuScorerBridge",
]
