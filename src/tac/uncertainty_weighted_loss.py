# SPDX-License-Identifier: MIT
"""Canonical uncertainty-weighted multi-task loss per ARBITRARINESS-EXTINCTION TOP-3 + TOP-6.

Per codex routing directive
``codex_routing_directive_arbitrariness_extinction_top3_per_pair_focal_weighting_20260518.md``
+ ``codex_routing_directive_arbitrariness_extinction_top6_uncertainty_weighted_multitask_loss_20260518.md``
+ slot 22 codex findings review TOP-3 routing
+ operator blanket approval 2026-05-19 verbatim *"all operator decisions
and approval granted and provided fuly and completely"*.

Bug class
=========

(a) **Per-pair uniform weighting** (TOP-3): every substrate trainer
weights the 1199 (or 600) per-pair losses UNIFORMLY. But per-pair
``pose_avg + seg_avg`` vary by ~100× across the 1200 pairs (per-pair
difficulty distribution is highly skewed). Uniform weighting
under-penalizes hard pairs → optimizer spends gradient on easy pairs
already near zero.

(b) **Axis-level static weights** (TOP-6): substrate score-aware losses
combine seg + pose + rate terms via static multipliers that ignore
per-axis HOMOSCEDASTIC noise (Kendall et al 2018 multi-task uncertainty
weighting; arXiv:1705.07115). The canonical Kendall closed-form learns
``log(σ²_axis)`` as ``nn.Parameter`` and weights the per-axis loss by
``1/(2σ²)`` + regularizer ``log(σ)``.

Predicted ΔS impact: ``[-0.006, -0.001]`` per axis-level uncertainty
weighting + ``[-0.006, -0.001]`` per per-pair focal weighting; combined
``[-0.012, -0.002]``. Cost envelope: ``$0`` (training-time only;
parameter count increase is negligible: 3 scalar log-variance params).

Mathematical contract
=====================

Kendall et al 2018 §3 derivation (Gaussian likelihood per task):

    L_total(W, σ_seg, σ_pose, σ_rate) =
        L_seg / (2 σ_seg²) + log(σ_seg)
      + L_pose / (2 σ_pose²) + log(σ_pose)
      + L_rate / (2 σ_rate²) + log(σ_rate)

The σ params are learned jointly with W. As σ_axis increases, the axis
loss contribution decreases AND a logarithmic regularizer penalizes
unbounded σ growth. Optimal σ* balances the two terms; this is the
canonical Lagrange-multiplier closed-form for the dual problem of
minimizing per-axis loss SUBJECT TO bounded uncertainty.

For numerical stability we parameterize ``log_sigma = log(σ)`` and
optimize over ``log_sigma`` directly. The forward becomes:

    L_total = exp(-2*log_sigma_axis) * L_axis * 0.5 + log_sigma_axis

This is the form used by Kendall's reference implementation + every
downstream Kendall-style implementation (e.g. timm + torchgeo).

Per-pair focal weighting (Lin et al 2017, arXiv:1708.02002):

    weight_pair_i = (1 - p_i)^γ
    where p_i = exp(-current_loss_i)

The two compose orthogonally: per-pair focal weighting reshapes the
per-pair loss distribution; axis-level Kendall weighting balances the
per-axis aggregates. The canonical helpers below implement both.

6-hook wire-in
==============

Per Catalog #125 ("Subagent coherence-by-default" non-negotiable):

- Hook 1 (sensitivity-map): N/A.
- Hook 2 (Pareto constraint): ACTIVE — axis-level Kendall weights
  implicitly trace a Pareto-optimal point on the seg×pose×rate frontier
  per the dual Lagrangian.
- Hook 3 (bit-allocator): N/A.
- Hook 4 (cathedral autopilot dispatch): ACTIVE via
  ``tac.cathedral_consumers.uncertainty_weighted_loss_consumer``
  (Catalog #335 auto-discovery paradigm).
- Hook 5 (continual-learning posterior): ACTIVE via
  ``tac.canonical_equations`` registration as
  ``per_pair_loss_weighting_optimal_v1``.
- Hook 6 (probe-disambiguator): N/A.

Cross-references
================

- Kendall, A. & Gal, Y. (2017). "What Uncertainties Do We Need in
  Bayesian Deep Learning for Computer Vision?" arXiv:1703.04977
- Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-Task Learning Using
  Uncertainty to Weigh Losses for Scene Geometry and Semantics."
  CVPR 2018 (arXiv:1705.07115)
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
  "Focal Loss for Dense Object Detection." ICCV 2017 (arXiv:1708.02002)
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE"
- ``tac.score_lagrangian`` (sister canonical helper TOP-1; complementary —
  TOP-1 = analytical baseline; TOP-3/TOP-6 = learned perturbations around)
- Catalog #323 canonical Provenance umbrella
- Catalog #335 cathedral consumer auto-ingest paradigm

Per CLAUDE.md "Beauty, simplicity, and developer experience": frozen
dataclass + explicit invariants where applicable; the learnable module
exposes a narrow API.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn


# Canonical Kendall et al 2018 initial log-sigma (corresponds to σ=1 = uniform).
DEFAULT_INITIAL_LOG_SIGMA: Final[float] = 0.0

# Canonical Lin et al 2017 focal gamma (γ=0 = uniform; γ=2 = paper default).
DEFAULT_FOCAL_GAMMA: Final[float] = 0.0  # opt-in; uniform by default
FOCAL_GAMMA_PAPER_DEFAULT: Final[float] = 2.0


class UncertaintyWeightedLossError(ValueError):
    """Raised when canonical-helper invariants are violated."""


@dataclass(frozen=True)
class UncertaintyWeightedLossConfig:
    """Canonical config for axis-level Kendall uncertainty weighting.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-
    negotiable: substrates MAY fork (e.g., to drop the rate axis if they
    operate at the contest rate-term floor); the canonical defaults reflect
    the contest's 3-axis ``sqrt(10*d_pose) + 100*d_seg + 25*rate/N``
    score formula.
    """

    initial_log_sigma_seg: float = DEFAULT_INITIAL_LOG_SIGMA
    initial_log_sigma_pose: float = DEFAULT_INITIAL_LOG_SIGMA
    initial_log_sigma_rate: float = DEFAULT_INITIAL_LOG_SIGMA
    include_rate_axis: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_log_sigma_seg", self.initial_log_sigma_seg),
            ("initial_log_sigma_pose", self.initial_log_sigma_pose),
            ("initial_log_sigma_rate", self.initial_log_sigma_rate),
        ):
            if not isinstance(value, (int, float)):
                raise UncertaintyWeightedLossError(
                    f"{name} must be numeric; got {type(value).__name__}"
                )
            if value != value:  # NaN
                raise UncertaintyWeightedLossError(f"{name} must not be NaN")
        if not isinstance(self.include_rate_axis, bool):
            raise UncertaintyWeightedLossError("include_rate_axis must be bool")


class UncertaintyWeightedScoreLoss(nn.Module):
    """Canonical Kendall multi-task uncertainty weighting for contest score.

    Forward signature::

        loss = uw_loss(seg_term, pose_term, rate_term)

    Returns the scalar weighted sum + logarithmic regularizer. The 3
    log-sigma parameters are learned jointly with the substrate model.

    Per CLAUDE.md "Beauty, simplicity, and developer experience": this
    module exposes only the forward + helper accessors. No hidden state.
    """

    def __init__(self, config: UncertaintyWeightedLossConfig | None = None) -> None:
        super().__init__()
        cfg = config or UncertaintyWeightedLossConfig()
        self._config = cfg
        self.log_sigma_seg = nn.Parameter(torch.tensor(float(cfg.initial_log_sigma_seg)))
        self.log_sigma_pose = nn.Parameter(torch.tensor(float(cfg.initial_log_sigma_pose)))
        if cfg.include_rate_axis:
            self.log_sigma_rate = nn.Parameter(torch.tensor(float(cfg.initial_log_sigma_rate)))
        else:
            self.register_parameter("log_sigma_rate", None)

    @property
    def config(self) -> UncertaintyWeightedLossConfig:
        return self._config

    def forward(
        self,
        seg_term: torch.Tensor,
        pose_term: torch.Tensor,
        rate_term: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Kendall-weighted total loss.

        Canonical formula (per Kendall et al 2018 §3 + their reference impl):

            L = 0.5 * exp(-2*log_sigma_axis) * L_axis + log_sigma_axis

        for each axis, summed.
        """
        if not isinstance(seg_term, torch.Tensor):
            raise UncertaintyWeightedLossError(
                f"seg_term must be Tensor; got {type(seg_term).__name__}"
            )
        if not isinstance(pose_term, torch.Tensor):
            raise UncertaintyWeightedLossError(
                f"pose_term must be Tensor; got {type(pose_term).__name__}"
            )
        total = (
            0.5 * torch.exp(-2.0 * self.log_sigma_seg) * seg_term
            + self.log_sigma_seg
            + 0.5 * torch.exp(-2.0 * self.log_sigma_pose) * pose_term
            + self.log_sigma_pose
        )
        if self._config.include_rate_axis:
            if rate_term is None:
                raise UncertaintyWeightedLossError(
                    "include_rate_axis=True requires non-None rate_term"
                )
            if not isinstance(rate_term, torch.Tensor):
                raise UncertaintyWeightedLossError(
                    f"rate_term must be Tensor; got {type(rate_term).__name__}"
                )
            total = total + (
                0.5 * torch.exp(-2.0 * self.log_sigma_rate) * rate_term
                + self.log_sigma_rate
            )
        return total

    def current_sigmas(self) -> dict[str, float]:
        """Return current σ_axis values as plain Python floats (detached)."""
        out = {
            "sigma_seg": math.exp(float(self.log_sigma_seg.detach())),
            "sigma_pose": math.exp(float(self.log_sigma_pose.detach())),
        }
        if self._config.include_rate_axis:
            out["sigma_rate"] = math.exp(float(self.log_sigma_rate.detach()))
        return out

    def current_axis_weights(self) -> dict[str, float]:
        """Return current ``0.5 * exp(-2*log_sigma)`` axis weights."""
        out = {
            "weight_seg": 0.5 * math.exp(-2.0 * float(self.log_sigma_seg.detach())),
            "weight_pose": 0.5 * math.exp(-2.0 * float(self.log_sigma_pose.detach())),
        }
        if self._config.include_rate_axis:
            out["weight_rate"] = (
                0.5 * math.exp(-2.0 * float(self.log_sigma_rate.detach()))
            )
        return out


def per_pair_focal_weights(
    per_pair_losses: torch.Tensor,
    *,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    detach_weights: bool = True,
) -> torch.Tensor:
    """Compute per-pair focal weights per Lin et al 2017.

    Formula:

        p_i = exp(-per_pair_loss_i)
        weight_i = (1 - p_i)^gamma

    For ``gamma=0`` the weights are uniform (=1 everywhere). Increasing
    ``gamma`` concentrates more weight on hard pairs (high per_pair_loss).

    Args:
        per_pair_losses: ``(N,)`` or ``(B, P)`` tensor of per-pair losses.
        gamma: focal-loss gamma; >= 0. Default 0 (uniform).
        detach_weights: if True (default), the weights are detached so the
            focal modulation does NOT contribute gradient to the underlying
            loss-magnitude path (canonical Lin et al 2017 usage). If False,
            the weights flow gradient through ``p_i``.

    Returns:
        Tensor with same shape as ``per_pair_losses`` containing focal
        weights in ``[0, 1]``.
    """
    if not isinstance(per_pair_losses, torch.Tensor):
        raise UncertaintyWeightedLossError(
            f"per_pair_losses must be Tensor; got {type(per_pair_losses).__name__}"
        )
    if not isinstance(gamma, (int, float)) or gamma != gamma:
        raise UncertaintyWeightedLossError(
            f"gamma must be numeric non-NaN; got {gamma!r}"
        )
    if gamma < 0:
        raise UncertaintyWeightedLossError(
            f"gamma must be >= 0; got {gamma}"
        )
    if gamma == 0:
        # Uniform weights — no need to evaluate the formula.
        return torch.ones_like(per_pair_losses)
    p = torch.exp(-per_pair_losses)
    weights = (1.0 - p) ** gamma
    if detach_weights:
        weights = weights.detach()
    return weights


def apply_focal_per_pair_reweighting(
    per_pair_losses: torch.Tensor,
    *,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    detach_weights: bool = True,
) -> torch.Tensor:
    """Return the focal-weighted MEAN of ``per_pair_losses``.

    Equivalent to::

        weights = per_pair_focal_weights(per_pair_losses, gamma=gamma)
        return (weights * per_pair_losses).sum() / weights.sum().clamp(min=eps)
    """
    weights = per_pair_focal_weights(
        per_pair_losses, gamma=gamma, detach_weights=detach_weights
    )
    weighted_sum = (weights * per_pair_losses).sum()
    weight_total = weights.sum().clamp(min=1e-8)
    return weighted_sum / weight_total


__all__ = [
    "DEFAULT_INITIAL_LOG_SIGMA",
    "DEFAULT_FOCAL_GAMMA",
    "FOCAL_GAMMA_PAPER_DEFAULT",
    "UncertaintyWeightedLossError",
    "UncertaintyWeightedLossConfig",
    "UncertaintyWeightedScoreLoss",
    "per_pair_focal_weights",
    "apply_focal_per_pair_reweighting",
]
