# SPDX-License-Identifier: MIT
"""The WORKING score-aware loop — PR95-faithful, reusable, well-conditioned.

This is the fix task #76 produces for the INERT loop. It is a small, reusable
surface that #74 (distillation), #63 (loss hinge), and lever-C/#65 (retrained
smaller renderer) consume against their own carriers.

The contract (1:1 with PR95 ``hnerv_muon/src/stages/common.py``, the proven 0.193
loop, NOT the MLX learnable-head surrogate that the elephant audit diagnosed as
inert):

1. The seg objective is computed by forwarding the LIVE frozen SegNet on the
   rendered frame and applying a margin/CE loss against the GT SegNet argmax
   targets. There is NO learnable student head and NO recon-MSE.
2. The pose objective is ``sqrt(10*MSE)`` of the LIVE frozen PoseNet output vs
   the GT PoseNet output (the upstream YUV6 path is gradient-reachable via the
   canonical ``patch_upstream_yuv6_globally`` token, per CLAUDE.md
   "eval_roundtrip" non-negotiable — the upstream ``rgb_to_yuv6`` is
   ``@torch.no_grad`` / in-place and otherwise severs the pose gradient).
3. The eval roundtrip (bicubic-up 874x1164 -> bilinear-down 384x512 -> STE
   round) is in the inner loop, so the optimizer sees the rounded frames the
   scorer sees.
4. Aggregation is the PR95 score-domain weighting ``100*seg + 1*pose`` — NOT a
   dual-ascent Lagrangian over a dozen guard/tether/floor terms (that surrogate
   stack is what made grad_norm explode to 1e6 and drowned the seg signal).
5. AdamW (decoder lr, latents 10x), cosine schedule, grad-clip 1.0 (which is
   RARELY hit because the loss is well-scaled — the diagnostic for a working
   loop is that the clip fires on a small fraction of steps, not 100%).
6. EMA shadow (decay 0.999) is the inference checkpoint per CLAUDE.md "EMA".

The decisive observable is :func:`ScoreAwareTrainer.exact_d_seg` — the EXACT
SegNet argmax-disagreement on the LIVE render. A working loop drives it BELOW the
~0.50 mean-field wall; the inert harness pinned it at 0.5048.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training
from tac.score_aware_loop.live_segnet_loss import (
    CURRICULUM_SEG_LOSS_FORMS,
    STAGE_SEG_LOSS_FNS,
    exact_d_seg_from_logits,
    pose_loss,
)

# Scorer input size: SegNet/PoseNet consume (384, 512) HxW (camera-eval size).
SCORER_HW = (384, 512)


@dataclass
class ScoreAwareLoopConfig:
    """Configuration for the PR95-faithful score-aware loop.

    Defaults mirror PR95's stage-1 (the CE stage that pulls the decoder to
    argmax-correct frames). The aggregation weights are the PR95 score-domain
    ``100*seg + 1*pose``.
    """

    epochs: int = 60
    batch_size: int = 8
    seg_loss_form: str = "ce_seg_loss"
    seg_weight: float = 100.0
    pose_weight: float = 1.0
    decoder_lr: float = 1.0e-3
    latent_lr_mult: float = 10.0
    grad_clip: float = 1.0
    lr_floor_ratio: float = 5.0e-6
    ema_decay: float = 0.999
    eval_every: int = 10
    pose_enabled: bool = True
    eval_roundtrip: bool = True
    seed: int = 0
    # Scorer input resolution (H, W). Default is the contest camera-eval size; a
    # smaller value is permitted ONLY for fast tests/proofs (the real scorer + the
    # real eval roundtrip consume 384x512). Production runs MUST keep the default.
    scorer_hw: tuple[int, int] = SCORER_HW
    # Telemetry sink (epoch dict appended). Optional.
    telemetry: list[dict[str, Any]] = field(default_factory=list)


class _EMA:
    """Param-wise EMA shadow (CLAUDE.md "EMA" canonical snapshot+restore)."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self._num_updates = 0
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        # [B4-FIX 2026-06-11] warmup decay (the ONE canonical schedule) so the
        # shadow tracks the live weights on short runs (the poisoned-tree fix).
        from tac.ema_warmup import warmup_ema_decay

        self._num_updates += 1
        d = warmup_ema_decay(self._num_updates, self.decay)
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if s.is_floating_point():
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v.detach())

    def apply_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA shadow into ``model``; return the originals for restore."""
        orig = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return orig


class ScoreAwareTrainer:
    """PR95-faithful score-aware trainer over a per-pair RGB carrier.

    Args:
        model: a carrier with ``reconstruct_pair(idx) -> (rgb_0, rgb_1)`` each
            ``(B, 3, H, W)`` in ``[0, 255]`` (e.g. :class:`TinyPairCarrier`).
        distortion_net: the upstream frozen ``DistortionNet`` (live SegNet +
            PoseNet); its parameters MUST have ``requires_grad=False``.
        seg_targets_hard: GT SegNet argmax ``(n_pairs, H, W)`` int64.
        pose_targets: GT PoseNet 6-dim ``(n_pairs, 6)`` float32.
        config: loop config.
        seg_loss_fn: optional override of the stage seg-loss fn (defaults to the
            registry entry for ``config.seg_loss_form``).

    The carrier latent count must match ``seg_targets_hard.shape[0]``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        distortion_net: torch.nn.Module,
        seg_targets_hard: torch.Tensor,
        pose_targets: torch.Tensor | None,
        config: ScoreAwareLoopConfig,
        *,
        seg_loss_fn: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.dnet = distortion_net
        self.seg_targets_hard = seg_targets_hard
        self.pose_targets = pose_targets
        self.cfg = config
        if seg_loss_fn is not None:
            self.seg_loss_fn = seg_loss_fn
        else:
            if config.seg_loss_form not in STAGE_SEG_LOSS_FNS:
                raise ValueError(
                    f"unknown seg_loss_form {config.seg_loss_form!r}; "
                    f"known: {sorted(STAGE_SEG_LOSS_FNS)}"
                )
            self.seg_loss_fn = STAGE_SEG_LOSS_FNS[config.seg_loss_form]
        # Fail closed: the scorer must be frozen (CLAUDE.md "Strict scorer").
        for p in self.dnet.parameters():
            if p.requires_grad:
                raise ValueError(
                    "distortion_net has trainable params; the scorer must be "
                    "frozen (requires_grad=False) so the gradient only updates "
                    "the carrier."
                )
        self.n_pairs = int(seg_targets_hard.shape[0])

    # -- the loss (PR95-faithful, direct-through-live-scorer) ----------------

    def _render_pair_scorer_input(self, idx: torch.Tensor) -> torch.Tensor:
        """Render the pair and produce the scorer ``(B, 2, H, W, 3)`` BHWC input.

        Up/down-samples to the exact scorer resolution via the canonical eval
        roundtrip (bicubic-up -> bilinear-down -> STE round), exactly the surface
        ``evaluate.py`` applies. Returns BHWC in ``[0, 255]`` (the layout
        ``DistortionNet.preprocess_input`` consumes).
        """
        scorer_hw = self.cfg.scorer_hw
        rgb0, rgb1 = self.model.reconstruct_pair(idx)  # (B,3,h,w) in [0,255]
        b = rgb0.shape[0]
        pair = torch.stack([rgb0, rgb1], dim=1)  # (B,2,3,h,w)
        flat = pair.reshape(b * 2, 3, rgb0.shape[-2], rgb0.shape[-1])
        # Resize to scorer size first (PR95 renders at 384x512 natively; tiny
        # carriers render small, so resize to scorer res before the roundtrip).
        flat = F.interpolate(flat, size=scorer_hw, mode="bilinear", align_corners=False)
        if self.cfg.eval_roundtrip:
            # The eval roundtrip upsamples to camera resolution then back. For the
            # default 384x512 scorer this is the contest 874x1164. For smaller
            # test resolutions, scale the camera target proportionally (~2.27x).
            cam_h = max(round(scorer_hw[0] * 874 / 384), scorer_hw[0] + 1)
            cam_w = max(round(scorer_hw[1] * 1164 / 512), scorer_hw[1] + 1)
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
        bchw = flat.reshape(b, 2, 3, scorer_hw[0], scorer_hw[1])
        # DistortionNet.preprocess_input expects (B, T, H, W, C).
        return bchw.permute(0, 1, 3, 4, 2).contiguous()

    def compute_loss(self, idx: torch.Tensor) -> dict[str, Any]:
        """Forward the LIVE frozen scorer and compute the PR95-faithful loss.

        Returns a dict with ``total`` (scalar tensor), ``seg`` / ``pose``
        (scalar tensors), and ``d_seg`` (float, the EXACT argmax disagreement on
        the live render for this batch).
        """
        bhwc = self._render_pair_scorer_input(idx)
        posenet_in, segnet_in = self.dnet.preprocess_input(bhwc)
        seg_out = self.dnet.segnet(segnet_in)
        # Measure the exact per-batch d_seg first so the hard-pixel curriculum
        # can gate its phase on it (the curriculum forms take ``current_d_seg``).
        with torch.no_grad():
            d_seg_batch = exact_d_seg_from_logits(seg_out, self.seg_targets_hard[idx])
        if self.cfg.seg_loss_form in CURRICULUM_SEG_LOSS_FORMS:
            seg_l = self.seg_loss_fn(
                seg_out, self.seg_targets_hard[idx], current_d_seg=d_seg_batch
            )
        else:
            seg_l = self.seg_loss_fn(seg_out, self.seg_targets_hard[idx])
        total = self.cfg.seg_weight * seg_l
        out: dict[str, Any] = {"seg": seg_l}
        out["d_seg"] = d_seg_batch
        if self.cfg.pose_enabled and self.pose_targets is not None:
            pose_out = self.dnet.posenet(posenet_in)
            pose_pred = pose_out["pose"][:, :6]
            pose_l = pose_loss(pose_pred, self.pose_targets[idx])
            total = total + self.cfg.pose_weight * pose_l
            out["pose"] = pose_l
        out["total"] = total
        return out

    # -- the optimizer (well-conditioned, AdamW + cosine + grad-clip 1.0) ----

    def _build_optimizer(self) -> tuple[torch.optim.Optimizer, Any, list]:
        cfg = self.cfg
        latent_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if "latent" in name.lower():
                latent_params.append(p)
            else:
                other_params.append(p)
        groups = [{"params": other_params, "lr": cfg.decoder_lr}]
        if latent_params:
            groups.append(
                {"params": latent_params, "lr": cfg.decoder_lr * cfg.latent_lr_mult}
            )
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
        eta_min_ratio = max(cfg.lr_floor_ratio / max(cfg.decoder_lr, 1e-12), 1e-3)

        def lr_lambda(epoch: int) -> float:
            return max(
                0.5 * (1 + math.cos(math.pi * epoch / max(cfg.epochs, 1))),
                eta_min_ratio,
            )

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        all_params = other_params + latent_params
        return opt, sched, all_params

    def train(self) -> dict[str, Any]:
        """Run the loop. Returns a summary dict with the descent trajectory.

        The summary records, per eval epoch, the EXACT live-render d_seg, the seg
        loss, the grad norm, and the fraction of steps where grad-clip fired —
        the well-conditioning diagnostic.
        """
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        opt, sched, all_params = self._build_optimizer()
        ema = _EMA(self.model, cfg.ema_decay)
        self._ema_for_eval = ema

        d_seg_initial = self.exact_d_seg(use_ema=False)
        trajectory: list[dict[str, Any]] = []
        best_d_seg = float("inf")
        clip_fired_steps = 0
        total_steps = 0

        for epoch in range(cfg.epochs):
            perm = torch.randperm(self.n_pairs)
            ep_seg = 0.0
            ep_grad_norm = 0.0
            nb = 0
            for start in range(0, self.n_pairs, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                opt.zero_grad(set_to_none=True)
                loss_parts = self.compute_loss(idx)
                loss_parts["total"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
                gn = float(grad_norm)
                if gn > cfg.grad_clip:
                    clip_fired_steps += 1
                total_steps += 1
                opt.step()
                ema.update(self.model)
                ep_seg += float(loss_parts["seg"].detach())
                ep_grad_norm += gn
                nb += 1
            sched.step()

            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                d_seg = self.exact_d_seg(use_ema=True)
                best_d_seg = min(best_d_seg, d_seg)
                row = {
                    "epoch": epoch + 1,
                    "exact_d_seg_ema": d_seg,
                    "loss_seg_mean": ep_seg / max(nb, 1),
                    "grad_norm_mean": ep_grad_norm / max(nb, 1),
                    "clip_fired_fraction": clip_fired_steps / max(total_steps, 1),
                    "lr": opt.param_groups[0]["lr"],
                }
                trajectory.append(row)
                cfg.telemetry.append(row)

        return {
            "d_seg_initial": d_seg_initial,
            "d_seg_final_ema": trajectory[-1]["exact_d_seg_ema"] if trajectory else None,
            "d_seg_best_ema": best_d_seg,
            "descended": (
                trajectory[-1]["exact_d_seg_ema"] < d_seg_initial - 1e-4
                if trajectory
                else False
            ),
            "clip_fired_fraction_final": (
                trajectory[-1]["clip_fired_fraction"] if trajectory else None
            ),
            "trajectory": trajectory,
        }

    @torch.no_grad()
    def exact_d_seg(self, *, use_ema: bool = True) -> float:
        """Return the mean EXACT live-render d_seg over all pairs.

        The decisive, non-proxy observable. When ``use_ema`` the EMA shadow is
        evaluated (the inference checkpoint per CLAUDE.md "EMA").
        """
        ema_orig = None
        if use_ema and hasattr(self, "_ema_for_eval"):
            ema_orig = self._ema_for_eval.apply_to(self.model)
        was_training = self.model.training
        self.model.eval()
        try:
            total = 0.0
            n = 0
            for start in range(0, self.n_pairs, self.cfg.batch_size):
                idx = torch.arange(
                    start, min(start + self.cfg.batch_size, self.n_pairs)
                )
                bhwc = self._render_pair_scorer_input(idx)
                _, segnet_in = self.dnet.preprocess_input(bhwc)
                seg_out = self.dnet.segnet(segnet_in)
                d = exact_d_seg_from_logits(seg_out, self.seg_targets_hard[idx])
                total += d * len(idx)
                n += len(idx)
            return total / max(n, 1)
        finally:
            if ema_orig is not None:
                self.model.load_state_dict(ema_orig)
            if was_training:
                self.model.train()
