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


def configure_torch_cpu_threads(num_threads: int | None = None) -> int:
    """Pin the torch-CPU scorer thread count to the measured-optimal value.

    The frozen contest scorer (EfficientNet-B2 SegNet + FastViT PoseNet) is the
    >97%-of-wall-clock bottleneck of the capstone training step (the MLX render /
    numpy copy / eval_roundtrip are <3%). On Apple-Silicon arm64 torch has NO
    mkldnn / NO MKL (``torch.backends.mkldnn.is_available()`` is False), so the
    depthwise convolutions dispatch to ``aten::_slow_conv2d_forward`` — the naive
    reference kernel — and the ONLY genuinely-fast axes are (a) scorer-batch
    amortization (caller-side: render more pairs per scorer call) and (b) the
    torch thread count, tuned here.

    Measured on the 6-performance-core M5 Max (``tools/profile_capstone_training_
    throughput.py`` thread sweep, full fwd+bwd step): the per-step wall-clock is
    flat from ~6 to ~10 threads and DEGRADES at >=14 (cross-core cache thrash on
    the slow-conv path makes 14-18 threads SLOWER than 6). So the sane policy is
    to pin to the performance-core count (default torch picks 6 on this box,
    which is already near-optimal) and NEVER let it run up to the full logical
    core count.

    This is a numerics-preserving knob: thread count changes only the float
    reduction order inside the conv/matmul kernels (sub-ULP), never the SegNet
    argmax (the exact d_seg is bit-stable across thread counts — verified in the
    bridge test suite). A daemon should call this ONCE at startup before the
    first scorer forward.

    Args:
        num_threads: explicit thread count to pin. ``None`` (default) resolves to
            ``min(performance_core_count, 8)`` — the measured sweet spot — falling
            back to the current ``torch.get_num_threads()`` if the perf-core count
            cannot be read.

    Returns:
        The thread count that was set (the resolved value).
    """
    if num_threads is None:
        perf_cores: int | None = None
        try:  # macOS perf-core count (the cores the scorer should run on).
            import subprocess

            perf_cores = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel0.physicalcpu"]
                ).decode().strip()
            )
        except Exception:  # pragma: no cover - non-macOS / sysctl missing
            perf_cores = None
        resolved = (
            min(perf_cores, 8)
            if perf_cores and perf_cores > 0
            else int(torch.get_num_threads())
        )
    else:
        resolved = max(1, int(num_threads))
    torch.set_num_threads(resolved)
    return resolved


class Yuv6NotPatchedError(RuntimeError):
    """Raised when the pose path's upstream ``rgb_to_yuv6`` is not differentiable."""


def _posenet_routes_through_upstream_yuv6(distortion_net: Any) -> bool:
    """True iff this scorer's PoseNet is the REAL upstream one (routes via the global).

    Only the upstream ``modules.PoseNet`` calls the module-level
    ``modules.rgb_to_yuv6`` in ``preprocess_input`` (``upstream/modules.py:74``).
    Proto / stand-in PoseNets in the test suite have their OWN preprocess and do
    NOT depend on the global patch, so the fail-closed check must not fire on them
    (a false positive would break the proto-scorer fixtures and force a worse
    workaround). We detect the real one by class identity against the upstream
    ``modules.PoseNet`` (the only consumer of the global ``rgb_to_yuv6``).
    """
    from tac.differentiable_eval_roundtrip import _resolve_upstream_modules

    posenet = getattr(distortion_net, "posenet", None)
    if posenet is None:
        return False
    _frame_utils, modules = _resolve_upstream_modules()
    upstream_posenet_cls = getattr(modules, "PoseNet", None) if modules else None
    if upstream_posenet_cls is None:
        return False
    return isinstance(posenet, upstream_posenet_cls)


def _assert_yuv6_patched_for_pose_gradient(distortion_net: Any) -> None:
    """Fail closed unless ``modules.rgb_to_yuv6`` is the differentiable patch ([C1]).

    Upstream ``PoseNet.preprocess_input`` (``upstream/modules.py:74``) calls the
    module-level ``rgb_to_yuv6`` that ``modules.py`` imported from ``frame_utils``
    at its own import time. Upstream's implementation is ``@torch.no_grad()`` /
    in-place, so the pose pixel-gradient is SEVERED — the loss is silently
    pose-inert. ``patch_upstream_yuv6_globally`` (called by
    ``tac.score_aware_loop.targets.load_frozen_distortion_net``) overwrites
    ``modules.rgb_to_yuv6`` with ``differentiable_rgb_to_yuv6``. This asserts that
    swap is in place so a caller that builds the bridge over an un-patched REAL
    upstream scorer gets a clear error instead of a silently broken pose objective.

    The check ONLY applies when the scorer's PoseNet is the real upstream
    ``modules.PoseNet`` (the only consumer of the global). Proto / stand-in
    PoseNets (their own preprocess; no global dependency) are exempt — they cannot
    be severed by the un-patched global, so enforcing it on them would be a false
    positive (CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    the gate must be precise, not over-broad).
    """
    if not _posenet_routes_through_upstream_yuv6(distortion_net):
        return  # proto / non-upstream PoseNet: not severable by the global.
    from tac.differentiable_eval_roundtrip import (
        _resolve_upstream_modules,
        differentiable_rgb_to_yuv6,
    )

    _frame_utils, modules = _resolve_upstream_modules()
    if modules is None or not hasattr(modules, "rgb_to_yuv6"):  # pragma: no cover
        return
    if modules.rgb_to_yuv6 is not differentiable_rgb_to_yuv6:
        raise Yuv6NotPatchedError(
            "TorchScorerBridge: pose is enabled and the scorer is the REAL upstream "
            "PoseNet, but `modules.rgb_to_yuv6` is NOT the differentiable patch (it "
            f"is {getattr(modules.rgb_to_yuv6, '__name__', modules.rgb_to_yuv6)!r}). "
            "PoseNet.preprocess_input would sever the pose pixel-gradient, making "
            "the loss silently pose-inert. Call "
            "`tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally()` "
            "(or build the scorer via `load_frozen_distortion_net`, which patches) "
            "BEFORE constructing the bridge. (CLAUDE.md 'eval_roundtrip' + "
            "'Comment-only contracts FORBIDDEN'.)"
        )


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
        # [C1] Fail closed if the upstream yuv6 is NOT the differentiable patch when
        # pose is enabled. PoseNet.preprocess_input calls the module-level
        # ``modules.rgb_to_yuv6`` (upstream/modules.py:74); upstream's is
        # ``@torch.no_grad()`` / in-place, which SEVERS the pose pixel-grad — a
        # silently pose-INERT loss. ``patch_upstream_yuv6_globally`` (done by
        # ``load_frozen_distortion_net``) swaps in ``differentiable_rgb_to_yuv6``.
        # A future caller with an un-patched scorer must get a clear error, not a
        # silent regression (CLAUDE.md "Comment-only contracts FORBIDDEN").
        if pose_targets is not None:
            _assert_yuv6_patched_for_pose_gradient(distortion_net)
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

    def _eval_preprocess(self, render_n2chw: Any) -> tuple[Any, Any]:
        """Shared eval-side preprocess (resize -> eval_roundtrip -> NHWC -> split).

        Returns ``(posenet_in, segnet_in)`` for the frozen scorer. This is the
        SAME ladder ``exact_d_seg`` / ``exact_d_pose`` apply (bilinear resize to
        scorer HW if needed, then the bicubic-up/bilinear-down/STE-uint8
        eval_roundtrip, then NHWC + ``preprocess_input``), factored out so a fused
        eval can run SegNet AND PoseNet on ONE preprocess instead of two. Pure
        refactor of the existing per-method code (byte-identical to running each
        method's preamble) — no behavior change.
        """
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
        return self.dnet.preprocess_input(bhwc)

    @torch.inference_mode()
    def fused_d_seg_d_pose(
        self, render_n2chw: Any, idx: torch.Tensor
    ) -> tuple[float, float]:
        """Return ``(d_seg, d_pose)`` from ONE shared preprocess (eval throughput).

        Equivalent to calling ``exact_d_seg`` and ``exact_d_pose`` on the same
        ``render_n2chw`` / ``idx``, but runs the resize + eval_roundtrip +
        ``preprocess_input`` ONCE and feeds the result to BOTH SegNet and
        PoseNet, instead of the separate-call path that re-renders + re-preprocesses
        for each. The avoided work (one preprocess + one eval_roundtrip per batch)
        is small relative to the SegNet forward, but it is real and the result is
        numerics-faithful: ``d_seg`` is bit-identical to ``exact_d_seg`` (the same
        SegNet argmax on the same preprocessed frames) and ``d_pose`` is identical
        to ``exact_d_pose`` (the same PoseNet MSE). Uses ``torch.inference_mode``
        so NO autograd graph is built (eval-only; the loss path keeps its graph).

        Fails closed if pose is not enabled (no PoseNet / no targets) — a fused
        d_seg/d_pose has no meaning without the pose half.
        """
        _require_mlx()
        if not self.pose_enabled:
            raise ValueError(
                "fused_d_seg_d_pose requires a PoseNet + pose_targets (pose not "
                "enabled); use exact_d_seg for the seg-only path."
            )
        posenet_in, segnet_in = self._eval_preprocess(render_n2chw)
        seg_out = self.dnet.segnet(segnet_in)
        d_seg = float(exact_d_seg_from_logits(seg_out, self.seg_targets_hard[idx]))
        pose_out = self.dnet.posenet(posenet_in)
        pose_pred = pose_out["pose"][:, :6]
        d_pose = float(F.mse_loss(pose_pred, self.pose_targets[idx]).item())
        return d_seg, d_pose


__all__ = [
    "CAMERA_HW",
    "SCORER_HW",
    "ScoreBridgeResult",
    "TorchScorerBridge",
    "Yuv6NotPatchedError",
    "configure_torch_cpu_threads",
]
