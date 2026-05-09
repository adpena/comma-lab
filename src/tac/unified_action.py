"""GR-style unified action principle for the contest pipeline (T7-T22 + Lane-12-v2 + A1).

Per ``feedback_unified_lagrangian_action_principle_GR_style_20260509`` the
council's eureka was that the contest objective decomposes as a single scalar
**action**

.. math::

    S_{\\text{total}}(\\theta;\\,b,\\,h) = \\int \\big(\\,
        \\mathcal{L}_{\\text{seg}}
      + \\mathcal{L}_{\\text{pose}}
      + \\mathcal{L}_{\\text{rate}}
      + \\sum_k \\mathcal{L}_{\\text{regularizer}}^{(k)}
      \\big)\\,dt

with each ``L`` carrying explicit Lagrange multipliers (dual variables). The
individual tracks landed in 2026-05-09's lateral-leap session each correspond to
ONE named contribution to the action — there is no separate "loss function" per
track; the existing trainers already realize variational solutions to subsets of
this action. The unified module makes the decomposition first-class and
solver-actionable.

This module is a **scaffold** — existing trainers (Phase 1 Ballé hyperprior,
Lane 12-v2 NeRV-as-renderer, etc.) keep working as-is; ``Action`` is the
migration target. Each per-track field is OPTIONAL and `None`-by-default so a
trainer can opt into one contribution at a time.

The migration path is documented as an `Action.migration_status()` table.

Cross-references
----------------

- ``feedback_unified_lagrangian_action_principle_GR_style_20260509.md`` — the meta-principle
- ``feedback_t11_t13_t19_free_lateral_leaps_landed_20260509.md`` — T11/T13/T19 landings
- ``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`` — non-arbitrariness
- ``tac.optimizer.meta_lagrangian.MetaLagrangianSearch`` — existing planning-only Lagrangian search
- ``tac.joint_admm_coordinator.run_admm`` — existing ADMM coordinator
- ``tac.score_geometry_shannon_floor`` — T7 closed-form floor

CLAUDE.md compliance
--------------------

- Per "Meta-Lagrangian/Pareto solver" non-negotiable: every track is a typed
  field on ``Action`` so the planner can consume it as a row.
- Per "Forbidden score claims": this module never claims a score. It returns a
  scalar ``S_total`` tagged ``[predicted; unified-action; closed-form
  weighted-sum]``. Authoritative scores come from ``upstream/evaluate.py`` on
  exact archive bytes.
- Per CLAUDE.md "FORBIDDEN device-selection defaults": no MPS fallback. Caller
  passes ``device`` explicitly; defaults to CPU for the smoke/test path only.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

ACTION_SCHEMA_VERSION = "tac_unified_action_v1"
ACTION_EVIDENCE_GRADE = "[predicted; unified-action; closed-form weighted-sum]"


class TrackKind(str, Enum):
    """Canonical track identifiers landed via the lateral-leap session."""

    SEG_BASELINE = "seg_baseline"  # 100 * argmax-disagreement (contest scorer canonical)
    POSE_BASELINE = "pose_baseline"  # sqrt(10 * pose_avg) (contest scorer canonical)
    RATE_BASELINE = "rate_baseline"  # 25 * archive_bytes / PR106_TOTAL_RATE_DENOM
    T7_FISHER_RAO = "t7_fisher_rao"  # Riemannian distance (seg-axis refinement)
    T8_SINKHORN_W2 = "t8_sinkhorn_w2"  # Wasserstein-2 (seg-axis refinement)
    T11_LOVASZ_HINGE = "t11_lovasz_hinge"  # Convex IoU envelope (seg-axis)
    T13_JOINT_SOURCE_RD = "t13_joint_source_rd"  # Rate-axis floor refinement
    T19_ADAPTIVE_RHO = "t19_adaptive_rho"  # Numerical-solver step (no L contribution; consumed by ADMM)
    T20_KL_POSE_DISTILL = "t20_kl_pose_distill"  # Pose-axis training-time refinement
    T22_TEMPORAL_CONSISTENCY = "t22_temporal_consistency"  # Temporal smoothness regularizer
    LANE_12_V2 = "lane_12_v2_nerv_as_renderer"  # Architecture-class atom


@dataclass(frozen=True)
class DualVariables:
    """Per-track dual variables (Lagrange multipliers) λ_k.

    Defaults are ``1.0`` (uniform Boyd-style equal-weight init); a trainer
    that stacks tracks should adapt these via ADMM (see
    :class:`tac.joint_admm_coordinator`) or fix them per the council
    deliberation.
    """

    lambda_seg: float = 1.0
    lambda_pose: float = 1.0
    lambda_rate: float = 1.0
    lambda_t7: float = 0.0  # Off by default; activate per-experiment
    lambda_t8: float = 0.0
    lambda_t11: float = 0.0
    lambda_t13: float = 0.0  # T13 is a CONSTRAINT not a soft term; usually unused
    lambda_t20: float = 0.0
    lambda_t22: float = 0.0


@dataclass
class Action:
    """A single GR-style action carrying every track Lagrangian as a named field.

    Each ``L_<track>`` is a ``Callable[[theta], torch.Tensor]`` that returns a
    scalar tensor (the per-track contribution to the action). Setting a field
    to ``None`` drops that track from the action.

    Mutable fields (the callables) are intentionally NOT frozen — a trainer
    may swap them between phases.
    """

    L_seg: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_pose: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_rate: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t7: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t8: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t11: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t13: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t20: Callable[[torch.Tensor], torch.Tensor] | None = None
    L_t22: Callable[[torch.Tensor], torch.Tensor] | None = None

    duals: DualVariables = field(default_factory=DualVariables)
    metadata: dict[str, Any] = field(default_factory=dict)

    def active_tracks(self) -> list[TrackKind]:
        """Return the list of TrackKind whose contribution is currently active."""
        out: list[TrackKind] = []
        mapping = {
            TrackKind.SEG_BASELINE: self.L_seg,
            TrackKind.POSE_BASELINE: self.L_pose,
            TrackKind.RATE_BASELINE: self.L_rate,
            TrackKind.T7_FISHER_RAO: self.L_t7,
            TrackKind.T8_SINKHORN_W2: self.L_t8,
            TrackKind.T11_LOVASZ_HINGE: self.L_t11,
            TrackKind.T13_JOINT_SOURCE_RD: self.L_t13,
            TrackKind.T20_KL_POSE_DISTILL: self.L_t20,
            TrackKind.T22_TEMPORAL_CONSISTENCY: self.L_t22,
        }
        for kind, L in mapping.items():
            if L is None:
                continue
            # T7/T8/T11/T13/T20/T22 are inactive when their dual is exactly 0
            dual_attr = {
                TrackKind.T7_FISHER_RAO: "lambda_t7",
                TrackKind.T8_SINKHORN_W2: "lambda_t8",
                TrackKind.T11_LOVASZ_HINGE: "lambda_t11",
                TrackKind.T13_JOINT_SOURCE_RD: "lambda_t13",
                TrackKind.T20_KL_POSE_DISTILL: "lambda_t20",
                TrackKind.T22_TEMPORAL_CONSISTENCY: "lambda_t22",
            }.get(kind)
            if dual_attr is not None and getattr(self.duals, dual_attr) == 0.0:
                continue
            out.append(kind)
        return out

    def S_total(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute the scalar action.

        Returns a 0-D tensor with autograd if ``theta.requires_grad``.
        """
        contributions = self._per_track_contributions(theta)
        if not contributions:
            # Degenerate: no active tracks. Return a 0-tensor sharing theta's
            # device/dtype/grad so downstream gradient calls don't crash.
            return torch.zeros((), device=theta.device, dtype=theta.dtype)
        return sum(contributions.values(), start=torch.zeros((), device=theta.device, dtype=theta.dtype))

    def _per_track_contributions(self, theta: torch.Tensor) -> dict[TrackKind, torch.Tensor]:
        """Return per-track {TrackKind -> scalar tensor} for ACTIVE tracks only."""
        contributions: dict[TrackKind, torch.Tensor] = {}
        # Baseline contest contributions ALWAYS use lambda=1.0 weighting since they
        # ARE the contest scorer. The duals only modulate REFINEMENT tracks.
        if self.L_seg is not None:
            contributions[TrackKind.SEG_BASELINE] = self.duals.lambda_seg * self.L_seg(theta)
        if self.L_pose is not None:
            contributions[TrackKind.POSE_BASELINE] = self.duals.lambda_pose * self.L_pose(theta)
        if self.L_rate is not None:
            contributions[TrackKind.RATE_BASELINE] = self.duals.lambda_rate * self.L_rate(theta)
        if self.L_t7 is not None and self.duals.lambda_t7 != 0.0:
            contributions[TrackKind.T7_FISHER_RAO] = self.duals.lambda_t7 * self.L_t7(theta)
        if self.L_t8 is not None and self.duals.lambda_t8 != 0.0:
            contributions[TrackKind.T8_SINKHORN_W2] = self.duals.lambda_t8 * self.L_t8(theta)
        if self.L_t11 is not None and self.duals.lambda_t11 != 0.0:
            contributions[TrackKind.T11_LOVASZ_HINGE] = self.duals.lambda_t11 * self.L_t11(theta)
        if self.L_t13 is not None and self.duals.lambda_t13 != 0.0:
            contributions[TrackKind.T13_JOINT_SOURCE_RD] = self.duals.lambda_t13 * self.L_t13(theta)
        if self.L_t20 is not None and self.duals.lambda_t20 != 0.0:
            contributions[TrackKind.T20_KL_POSE_DISTILL] = self.duals.lambda_t20 * self.L_t20(theta)
        if self.L_t22 is not None and self.duals.lambda_t22 != 0.0:
            contributions[TrackKind.T22_TEMPORAL_CONSISTENCY] = self.duals.lambda_t22 * self.L_t22(theta)
        return contributions

    def gradient(self, theta: torch.Tensor) -> dict[TrackKind, torch.Tensor]:
        """Per-track gradient contributions ``∂L_k/∂θ`` via autograd.

        Each entry is a tensor of the SAME shape as ``theta`` (the gradient of
        that track's contribution alone). Useful for adversarial review of
        which track is dominating the descent direction.

        Requires ``theta.requires_grad``.
        """
        if not theta.requires_grad:
            raise ValueError(
                "Action.gradient requires theta.requires_grad=True; got False"
            )
        contributions = self._per_track_contributions(theta)
        grads: dict[TrackKind, torch.Tensor] = {}
        for kind, contribution in contributions.items():
            # Short-circuit when the contribution is detached / constant — there
            # is no autograd graph to back-propagate through (track does not
            # depend on theta).
            if not contribution.requires_grad or contribution.grad_fn is None:
                grads[kind] = torch.zeros_like(theta)
                continue
            grad = torch.autograd.grad(
                contribution,
                theta,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                # Track does not depend on theta (e.g. constant rate-axis term)
                grad = torch.zeros_like(theta)
            grads[kind] = grad.detach()
        return grads

    def step(
        self,
        theta: torch.Tensor,
        lr: float,
        dual_update: Callable[[DualVariables, dict[TrackKind, torch.Tensor]], DualVariables] | None = None,
    ) -> tuple[torch.Tensor, DualVariables]:
        """One variational step: gradient descent on theta + optional dual update.

        Returns (theta_new, duals_new). The dual_update callable, if provided,
        is called with the OLD duals + the per-track gradient norms (NOT the
        gradient tensors themselves) so it can implement Boyd-style adaptive
        ρ on the dual variables. If ``None``, duals are held constant.

        Note: this is a SIMPLE variational step — production trainers should
        run inside their own optimizer (Adam/SGD/AdamW) and use ``S_total``
        as the loss. ``Action.step`` is here for the smoke-test / scaffold path.
        """
        if not theta.requires_grad:
            raise ValueError("Action.step requires theta.requires_grad=True; got False")
        if not (lr > 0.0):
            raise ValueError(f"lr must be > 0; got {lr}")

        per_track_grads = self.gradient(theta)
        # Sum the per-track gradients to get the total gradient.
        total_grad = sum(
            per_track_grads.values(),
            start=torch.zeros_like(theta),
        )
        with torch.no_grad():
            theta_new = (theta - lr * total_grad).detach().requires_grad_(theta.requires_grad)

        # Optional dual update.
        if dual_update is None:
            duals_new = self.duals
        else:
            grad_norms = {k: float(g.norm().item()) for k, g in per_track_grads.items()}
            duals_new = dual_update(self.duals, grad_norms)
            if not isinstance(duals_new, DualVariables):
                raise TypeError(
                    f"dual_update must return DualVariables; got {type(duals_new).__name__}"
                )

        return theta_new, duals_new

    def migration_status(self) -> dict[str, Any]:
        """Report which tracks are active + which trainer is the canonical migration target.

        Returns a dict suitable for JSON serialization to a research ledger.
        """
        return {
            "schema": ACTION_SCHEMA_VERSION,
            "evidence_grade": ACTION_EVIDENCE_GRADE,
            "active_tracks": [k.value for k in self.active_tracks()],
            "duals": {
                "lambda_seg": self.duals.lambda_seg,
                "lambda_pose": self.duals.lambda_pose,
                "lambda_rate": self.duals.lambda_rate,
                "lambda_t7": self.duals.lambda_t7,
                "lambda_t8": self.duals.lambda_t8,
                "lambda_t11": self.duals.lambda_t11,
                "lambda_t13": self.duals.lambda_t13,
                "lambda_t20": self.duals.lambda_t20,
                "lambda_t22": self.duals.lambda_t22,
            },
            "canonical_trainer_migration_targets": {
                "phase_1_balle_hyperprior": "experiments/train_score_gradient_pr101_finetune.py + tac.paradigm_delta_epsilon_zeta",
                "lane_12_v2_nerv_as_renderer": "tac.lane_12_v2_nerv_as_renderer.train_step",
                "score_gradient_pr101_finetune": "experiments/train_score_gradient_pr101_finetune.py",
            },
            "metadata": dict(self.metadata),
        }


def make_action_from_track_callables(
    seg: Callable | None = None,
    pose: Callable | None = None,
    rate: Callable | None = None,
    *,
    t7_fisher_rao: Callable | None = None,
    t8_sinkhorn_w2: Callable | None = None,
    t11_lovasz_hinge: Callable | None = None,
    t13_joint_source_rd: Callable | None = None,
    t20_kl_pose_distill: Callable | None = None,
    t22_temporal_consistency: Callable | None = None,
    duals: DualVariables | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Action:
    """Convenience factory that takes per-track callables + DualVariables.

    Example::

        from tac.unified_action import make_action_from_track_callables, DualVariables

        action = make_action_from_track_callables(
            seg=lambda theta: torch.sum(theta**2),
            duals=DualVariables(lambda_seg=1.0),
        )
        S = action.S_total(theta)
    """
    return Action(
        L_seg=seg,
        L_pose=pose,
        L_rate=rate,
        L_t7=t7_fisher_rao,
        L_t8=t8_sinkhorn_w2,
        L_t11=t11_lovasz_hinge,
        L_t13=t13_joint_source_rd,
        L_t20=t20_kl_pose_distill,
        L_t22=t22_temporal_consistency,
        duals=duals or DualVariables(),
        metadata=dict(metadata) if metadata else {},
    )


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "ACTION_EVIDENCE_GRADE",
    "TrackKind",
    "DualVariables",
    "Action",
    "make_action_from_track_callables",
]
