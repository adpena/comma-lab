# SPDX-License-Identifier: MIT
"""PR95-faithful score-aware losses: direct margin/CE through the LIVE frozen SegNet.

This module is the *working* loss surface that task #76 produces — the fix for
the INERT score-aware loop diagnosed in
``.omx/research/pr95_elephant_audit_20260610T185556Z.md`` (task #75).

The elephant: our MLX harness computed a *learnable-student-head KL distillation*
surrogate (``loss.py::score_aware_loss`` default path), where the gradient to the
renderer flows through a learnable 1x1 head that is *itself* still learning to
mimic the SegNet — so the renderer gradient is a moving, weakly-coupled target
and the exact ``d_seg`` (SegNet argmax disagreement on the LIVE render) never
descends (pinned ~0.5048 across all 8 stages of the B1 clean run).

PR95's PROVEN loop (``hnerv_muon/src/losses.py`` + ``stages/common.py``) does
something structurally different and well-conditioned:

    seg_out = distortion_net.segnet(segnet_in)         # LIVE frozen SegNet forward
    seg_l   = seg_loss_fn(seg_out, gt_argmax_targets)  # CE/softplus ON the live logits

There is **NO learnable student head** and **NO recon-MSE**. The targets are the
GT scorer's own per-pixel argmax (literally the d_seg reference). The gradient
flows ``render -> frozen SegNet -> render params`` directly, which is exactly the
quantity ``evaluate.py`` charges. The four stage losses are the canonical PR95
margin surrogates.

These functions take SegNet logits ``(B, C, H, W)`` and hard targets
``(B, H, W)`` and return a scalar — a 1:1 port of PR95's ``losses.py`` so a
trainer (or the diagnosis harness) can call the EXACT proven objective.

[verified-against: hnerv_muon/src/losses.py ce_seg_loss / tau_softplus_seg_loss /
 smooth_disagreement_seg_loss / l7_softplus_seg_loss / pose_loss — bit-for-bit
 port; see .omx/research/inert_loop_fix_*.md head-to-head verdict]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _target_minus_runnerup_margin(
    seg_logits: torch.Tensor, targets_hard: torch.Tensor
) -> torch.Tensor:
    """Return per-pixel ``(target_logit - max_other_class_logit)`` margin.

    This is the shared core of the PR95 margin surrogates. ``margin > 0`` means
    the pixel is argmax-correct (the d_seg reference class wins); ``margin < 0``
    means a SegNet argmax disagreement. The surrogates squash this margin so the
    gradient peaks near the decision boundary (margin ~ 0) — pushing boundary
    pixels across the class line, which is exactly what lowers d_seg.

    Args:
        seg_logits: live SegNet logits ``(B, C, H, W)``.
        targets_hard: GT SegNet argmax labels ``(B, H, W)`` int64 in ``[0, C)``.

    Returns:
        ``(B, 1, H, W)`` per-pixel margin (target minus best competing class).
    """
    if seg_logits.dim() != 4:
        raise ValueError(
            f"seg_logits must be (B, C, H, W); got {tuple(seg_logits.shape)}"
        )
    if targets_hard.dim() != 3:
        raise ValueError(
            f"targets_hard must be (B, H, W); got {tuple(targets_hard.shape)}"
        )
    if targets_hard.shape != (seg_logits.shape[0], *seg_logits.shape[2:]):
        raise ValueError(
            "targets_hard spatial shape must match seg_logits without the class "
            f"axis; targets={tuple(targets_hard.shape)} "
            f"logits={tuple(seg_logits.shape)}"
        )
    target_logits = seg_logits.gather(1, targets_hard.unsqueeze(1).long())
    masked = seg_logits.clone()
    masked.scatter_(1, targets_hard.unsqueeze(1).long(), -1e9)
    runnerup = masked.max(dim=1, keepdim=True)[0]
    return target_logits - runnerup


def ce_seg_loss(seg_logits: torch.Tensor, targets_hard: torch.Tensor) -> torch.Tensor:
    """Stage-1 cross-entropy on the LIVE SegNet logits vs GT argmax targets.

    Identical to PR95 ``ce_seg_loss``: ``F.cross_entropy(logits, hard)``. This
    descends toward ~0 as the rendered frame's SegNet argmax matches the GT's —
    i.e. as d_seg drops. A working loop shows this number FALLING; the inert
    harness showed it RISING (1.16 -> 1.61).
    """
    return F.cross_entropy(seg_logits, targets_hard.long())


def tau_softplus_seg_loss(
    seg_logits: torch.Tensor, targets_hard: torch.Tensor, tau: float = 0.3
) -> torch.Tensor:
    """Stage-2 tau-softplus margin surrogate (PR95 ``tau_softplus_seg_loss``)."""
    margin = _target_minus_runnerup_margin(seg_logits, targets_hard)
    return (tau * F.softplus(-margin / tau)).mean()


def smooth_disagreement_seg_loss(
    seg_logits: torch.Tensor, targets_hard: torch.Tensor, tau: float = 0.3
) -> torch.Tensor:
    """Stage-3/4 smooth-disagreement surrogate (PR95 ``smooth_disagreement_seg_loss``).

    ``sigmoid(-margin/tau)`` — the soft 0/1 disagreement indicator. At the
    optimum it equals the true argmax-disagreement RATE (the exact d_seg), so
    minimizing it directly minimizes d_seg. Bell-curve gradient peaks at
    margin=0.
    """
    margin = _target_minus_runnerup_margin(seg_logits, targets_hard)
    return torch.sigmoid(-margin / tau).mean()


def l7_softplus_seg_loss(
    seg_logits: torch.Tensor,
    targets_hard: torch.Tensor,
    tau: float = 0.3,
    l7_threshold: float = 1.0,
    l7_mult: float = 4.0,
) -> torch.Tensor:
    """Stage-5+ L7-weighted softplus (PR95 ``l7_softplus_seg_loss``).

    Hard pixels (margin < threshold) get a ``(1 + l7_mult)`` boost, renormalized
    to mean 1, concentrating gradient on the pixels still near/over the line.
    """
    margin = _target_minus_runnerup_margin(seg_logits, targets_hard)
    per_pixel = tau * F.softplus(-margin / tau)
    with torch.no_grad():
        weights = 1.0 + l7_mult * (margin < l7_threshold).float()
        weights = weights / weights.mean()
    return (per_pixel * weights).mean()


def margin_polytope_hinge_seg_loss(
    seg_logits: torch.Tensor,
    targets_hard: torch.Tensor,
    margin_target: float = 0.5,
    hard_pixel_only: bool = False,
) -> torch.Tensor:
    """Margin-polytope hinge: push each pixel to the CHEAPEST-interior side.

    The hypothesis under test (seg_core #52/#63, FIELDS-MEDAL decisive test
    2026-06-11): CE optimizes the full per-pixel class probability simplex
    (driving the target logit arbitrarily high), which spends optimizer capacity
    on already-correct pixels and leaves the DIFFUSE argmax-instability near the
    SegNet decision boundary unresolved. The margin polytope is the convex region
    ``{margin >= margin_target}`` (correct side of the boundary plus a small
    slack). The hinge ``relu(margin_target - margin)`` is the L1 distance to that
    polytope: it is ZERO once a pixel clears the boundary by ``margin_target`` and
    applies gradient ONLY to pixels that are wrong or marginally-correct. This is
    the "cheapest interior" objective — it stops optimizing a pixel the moment it
    is safely on the correct side, concentrating all capacity on flip-prone
    boundary pixels (exactly the source of d_seg, per the diffuse-instability
    deep-profile).

    ``margin = target_logit - max_other_class_logit`` (the shared PR95 core). The
    gradient w.r.t. the logits is a constant ``+1`` on the target class and
    ``-1`` on the current runner-up for every pixel inside the hinge band, and 0
    outside — a clean, well-conditioned push across the class line.

    Args:
        seg_logits: live SegNet logits ``(B, C, H, W)``.
        targets_hard: GT SegNet argmax labels ``(B, H, W)`` int64.
        margin_target: required positive margin (slack). 0.5 = clear the boundary
            by half a logit unit. Larger = push harder past the boundary.
        hard_pixel_only: if True, mask the hinge to pixels currently argmax-WRONG
            (``margin <= 0``) — the curriculum mode (see
            :func:`hard_pixel_curriculum_seg_loss`). Note: with this mask the
            already-correct-but-marginal pixels (0 < margin < margin_target) are
            DROPPED, so use it only once d_seg is near floor (where the
            aggregate-surrogate signal is noise but the per-flipped-pixel signal
            is not, per #92).

    Returns:
        scalar mean hinge loss.
    """
    margin = _target_minus_runnerup_margin(seg_logits, targets_hard)
    hinge = F.relu(margin_target - margin)
    if hard_pixel_only:
        with torch.no_grad():
            flipped = (margin <= 0.0).float()
        denom = flipped.sum().clamp_min(1.0)
        return (hinge * flipped).sum() / denom
    return hinge.mean()


def hard_pixel_curriculum_seg_loss(
    seg_logits: torch.Tensor,
    targets_hard: torch.Tensor,
    margin_target: float = 0.5,
    curriculum_d_seg_threshold: float = 3.0e-3,
    current_d_seg: float | None = None,
) -> torch.Tensor:
    """Margin hinge with a d_seg-gated hard-pixel curriculum.

    While d_seg is far from floor (``current_d_seg >= threshold`` or unknown),
    the full margin-polytope hinge runs over ALL pixels (broad descent). Once
    d_seg drops below ``curriculum_d_seg_threshold``, the loss MASKS to the
    currently-flipped pixels (``margin <= 0``) so the remaining optimizer capacity
    is spent only on the residual disagreements — where, per #92, the aggregate
    surrogate is dominated by easy-pixel noise but the per-boundary-pixel signal
    is still informative.

    Args:
        current_d_seg: the EXACT live-render d_seg for THIS batch (the trainer
            passes ``compute_loss``'s measured value). If ``None`` the curriculum
            stays in the broad (all-pixel) phase.
    """
    in_hard_phase = (
        current_d_seg is not None and current_d_seg < curriculum_d_seg_threshold
    )
    return margin_polytope_hinge_seg_loss(
        seg_logits,
        targets_hard,
        margin_target=margin_target,
        hard_pixel_only=in_hard_phase,
    )


def pose_loss(pose_pred: torch.Tensor, pose_target: torch.Tensor) -> torch.Tensor:
    """sqrt(10*MSE) pose surrogate, constant across PR95 stages (``losses.py``).

    Concave-in-MSE so small errors are emphasized more than plain MSE — matches
    the contest ``sqrt(10*d_pose)`` Lagrangian term.
    """
    mse = F.mse_loss(pose_pred, pose_target)
    return torch.sqrt(10.0 * mse + 1e-12)


# Canonical stage -> seg-loss-fn registry, mirroring PR95's 8-stage progression.
# Keyed by the canonical ``loss_form`` token used in B1 telemetry so the
# diagnosis harness and any future trainer agree on the name<->fn mapping.
STAGE_SEG_LOSS_FNS = {
    "ce_seg_loss": ce_seg_loss,
    "tau_softplus_seg_loss": tau_softplus_seg_loss,
    "smooth_disagreement_seg_loss": smooth_disagreement_seg_loss,
    "l7_softplus_seg_loss": l7_softplus_seg_loss,
    "margin_polytope_hinge_seg_loss": margin_polytope_hinge_seg_loss,
    "hard_pixel_curriculum_seg_loss": hard_pixel_curriculum_seg_loss,
}

# Loss forms that accept the trainer's measured per-batch d_seg via a
# ``current_d_seg`` kwarg (the hard-pixel curriculum gate). The trainer threads
# the value only for these forms so the registry stays uniform.
CURRICULUM_SEG_LOSS_FORMS = frozenset({"hard_pixel_curriculum_seg_loss"})


def exact_d_seg_from_logits(
    seg_logits: torch.Tensor, targets_hard: torch.Tensor
) -> float:
    """Return the EXACT SegNet argmax-disagreement rate (the d_seg the scorer charges).

    This is the headline diagnostic: ``mean(argmax(live_logits) != gt_argmax)``.
    It is NOT a proxy — it is the per-pixel quantity ``upstream/evaluate.py``
    computes for the SegNet term. A working loop drives this BELOW 0.5; the inert
    harness pinned it at ~0.5048.
    """
    with torch.no_grad():
        pred = seg_logits.argmax(dim=1)
        return float((pred != targets_hard.long()).float().mean().item())
