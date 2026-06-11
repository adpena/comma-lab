# SPDX-License-Identifier: MIT
"""Cross-hardware-robust margin hinge for the capstone (the numpy-portability guard).

The capstone is trained on macOS (MLX render + torch-CPU scorer), exported to a
pure-numpy archive, and scored on a DIFFERENT host (Linux x86_64 contest CPU,
NVIDIA CUDA). The SegNet ``d_seg`` the score charges is a per-pixel ARGMAX, and
the per-host fp32 reduction-order drift moves the SegNet logits by up to ~0.096
(``.omx/research/arch_override_fp32_exact_gpu_training_scorer_20260611.md`` RUNG A
+ the drift ladder). A boundary pixel whose ``target_logit - runnerup_logit``
margin is INSIDE that drift band can FLIP its argmax across the host boundary —
so a local-sub-0.15 advisory can EVAPORATE at the contest (the lever-map L9
failure mode).

The fix (lever-map L7, named ESSENTIAL by the capstone spec): a NEW loss TERM
that pushes the boundary-pixel margin PAST the cross-hardware logit drift. It is
DISTINCT from the existing weight-boost ``l7_softplus_seg_loss`` (which reweights
the stage surrogate to concentrate gradient on hard pixels): this enforces a
margin FLOOR — ``mean(relu(margin_floor - margin))`` over the pixels whose margin
is below the floor (the boundary set). At the optimum every kept pixel has
``margin >= margin_floor``, so its argmax survives any logit perturbation smaller
than ``margin_floor`` — the numpy/Linux/CUDA transfer guarantee.

The hinge is ADDED to whatever PR95 stage seg-loss is active (CE / tau-softplus /
smooth-disagreement / L7-weighted), so it stacks with the curriculum rather than
replacing it. A capstone-owned WRAPPER (this module) keeps it collision-free with
the shared ``mlx_pr95_port`` loss surface: the trainer installs the wrapper into
the bridge's ``seg_loss_fn`` and re-wraps the (possibly stage-switched) base loss
on every curriculum transition.

NO-FAKE: the hinge is a REAL loss term with a REAL gradient — the test asserts it
penalizes small-margin-correct pixels (raises the loss + produces a non-zero
pixel gradient that PUSHES the margin up), and that with ``hinge_weight=0`` it is
byte-identical to the bare stage loss. A constant / no-op would FAIL those tests.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from tac.score_aware_loop.live_segnet_loss import _target_minus_runnerup_margin


def margin_floor_hinge(
    seg_logits: torch.Tensor,
    targets_hard: torch.Tensor,
    *,
    margin_floor: float,
) -> torch.Tensor:
    """Cross-hardware margin-floor hinge: ``mean(relu(margin_floor - margin))``.

    ``margin = target_logit - max_competing_logit`` (the SAME per-pixel margin the
    PR95 surrogates + the exact d_seg use). The hinge is zero for any pixel whose
    margin already clears ``margin_floor`` and grows linearly as the margin drops
    below it (including the argmax-WRONG pixels, ``margin < 0``, which it also
    pushes toward correct + past the floor). Mean over ALL pixels (the relu is the
    boundary selector — interior pixels with large margin contribute exactly 0).

    Args:
        seg_logits: live SegNet logits ``(B, C, H, W)``.
        targets_hard: GT SegNet argmax ``(B, H, W)`` int.
        margin_floor: the required margin (anchor ~0.1 > the measured ~0.096
            cross-hardware logit drift). Must be > 0.

    Returns:
        Scalar hinge loss (``relu(margin_floor - margin)`` averaged over pixels).
    """
    if margin_floor <= 0.0:
        raise ValueError(f"margin_floor must be > 0; got {margin_floor}")
    margin = _target_minus_runnerup_margin(seg_logits, targets_hard)  # (B,1,H,W)
    return torch.clamp(margin_floor - margin, min=0.0).mean()


class CrossHwMarginHingeSegLoss:
    """A seg-loss callable = base stage loss + ``hinge_weight * margin_floor_hinge``.

    Wraps the ACTIVE PR95 stage seg-loss (``base_loss_fn``) and adds the
    cross-hardware margin-floor hinge. The trainer installs an instance as the
    bridge's ``seg_loss_fn`` and calls :meth:`set_base_loss_fn` on every PR95
    curriculum stage transition so the hinge always stacks on the CURRENT stage
    surrogate (CE -> tau-softplus -> ... -> L7), never a stale one.

    ``hinge_weight=0`` makes the call byte-identical to the bare base loss (the
    opt-out / off path), so the default-off behaviour is provably a no-op.
    """

    def __init__(
        self,
        base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        margin_floor: float,
        hinge_weight: float,
    ) -> None:
        if margin_floor <= 0.0:
            raise ValueError(f"margin_floor must be > 0; got {margin_floor}")
        if hinge_weight < 0.0:
            raise ValueError(f"hinge_weight must be >= 0; got {hinge_weight}")
        self.base_loss_fn = base_loss_fn
        self.margin_floor = float(margin_floor)
        self.hinge_weight = float(hinge_weight)

    def set_base_loss_fn(
        self, base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        """Re-point the wrapped base loss (called on a curriculum stage switch)."""
        self.base_loss_fn = base_loss_fn

    def __call__(
        self, seg_logits: torch.Tensor, targets_hard: torch.Tensor
    ) -> torch.Tensor:
        base = self.base_loss_fn(seg_logits, targets_hard)
        if self.hinge_weight == 0.0:
            return base
        hinge = margin_floor_hinge(
            seg_logits, targets_hard, margin_floor=self.margin_floor
        )
        return base + self.hinge_weight * hinge


__all__ = ["CrossHwMarginHingeSegLoss", "margin_floor_hinge"]
