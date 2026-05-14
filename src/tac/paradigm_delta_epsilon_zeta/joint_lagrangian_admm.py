# SPDX-License-Identifier: MIT
# ADMM_WAIVED:B4-reviewed historical/planning naming; docstrings or delegated coordinator code clarify whether this is Lagrangian, bridge, or actual iterative ADMM.
"""Joint Lagrangian-ADMM coordinator for PARADIGM-δεζ T1.

Boyd-style ADMM (Boyd et al. 2011 §3) coordinates four constraint axes for
T1's joint training of (frozen-A1-encoder + 128K decoder + Ballé hyperprior):

  Augmented Lagrangian:
    L_ρ = D + λ_R · (R - R_target)
              + (ρ/2) · (R - R_target + u_R)²
              + λ_S · (s_loss - s_target)
              + (ρ/2) · (s_loss - s_target + u_S)²
              + λ_P · (p_loss - p_target)
              + (ρ/2) · (p_loss - p_target + u_P)²

  Dual updates (Boyd ADMM):
    u_R ← u_R + (R - R_target)
    u_S ← u_S + (s_loss - s_target)
    u_P ← u_P + (p_loss - p_target)
    λ_R ← max(0, λ_R + ρ * (R - R_target))   (only ≥ when constraint active)
    λ_S, λ_P analogous

  Adaptive ρ (Boyd §3.4.1): if primal residual >> 10× dual residual for K
  steps, ρ ← 2ρ; if dual >> 10× primal, ρ ← ρ/2. Bounded to [ρ_min, ρ_max].

This is **score-domain** Lagrangian — D, s_loss, p_loss are the contest
score components (rate, seg, pose) directly, NOT proxy losses. The R-target
should be sourced from :mod:`tac.joint_source_rd_bound` for principled
floors.

CLAUDE.md compliance
--------------------

- All lambdas are clipped to ≥ 0 (proper inequality multipliers).
- ρ is bounded; runaway ρ is the most common ADMM failure mode and the
  config enforces ``rho_max >= rho_min > 0``.
- The coordinator is **stateless** wrt the underlying nn.Modules — it owns
  the dual state but does not hold references to the optimiser. The trainer
  wires ρ-scaled augmented terms into its own backward pass.
- Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag", no
  predicted score is recorded in this module's docstring; predictions live
  in the trainer + dispatcher with explicit ``[predicted; Phase 1 scaffold]``
  tags.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class JointLagrangianADMMConfig:
    """Configuration for the joint Lagrangian-ADMM coordinator.

    Attributes
    ----------
    rate_target_bytes : float
        Target rate in BYTES (NOT bits). The coordinator converts to bits
        internally for the augmented term. T1 default: 80,000 (a 128KB-class
        archive after FP4 + Brotli).
    seg_target : float
        Target average SegNet distortion. Default: 0.0007 (matches A1 anchor).
    pose_target : float
        Target average PoseNet distortion. Default: 0.00017 (matches A1 anchor).
    rho_init : float
        Initial ρ (penalty parameter). Default: 1.0.
    rho_min, rho_max : float
        Bounds on ρ for adaptive update. Default: (0.01, 100.0).
    adaptive_rho_window : int
        Number of recent steps to inspect for primal/dual residual ratio.
        Default: 16.
    adaptive_rho_ratio : float
        Trigger threshold for the primal/dual ratio (Boyd §3.4.1 default 10).
    lambda_init : float
        Initial Lagrange multiplier for each constraint. Default: 1.0.
    lambda_max : float
        Cap on each Lagrange multiplier (prevents pathological blow-up).
        Default: 1e6.
    use_t19_adaptive_rho : bool
        If True, the coordinator routes its embedded adaptive-ρ rule through
        the standalone :func:`tac.joint_admm_coordinator.adaptive_rho_step`
        helper (the T19 deliverable per memory
        ``feedback_t11_t13_t19_free_lateral_leaps_landed_20260509``). The
        canonical Boyd §3.4.1 / He-Yang 2000 form is used at every step
        instead of the legacy windowed-average form. Default: False
        (backward-compat). The trainer flips this on with
        ``--enable-t19-adaptive-rho``.
    t19_tau_grow, t19_tau_shrink : float
        Multiplicative grow/shrink factors when ``use_t19_adaptive_rho`` is
        active. Defaults are Boyd's canonical 2.0 / 0.5. Must satisfy
        ``t19_tau_grow > 1`` and ``t19_tau_shrink in (0, 1)``.
    """

    rate_target_bytes: float = 80_000.0
    seg_target: float = 7e-4
    pose_target: float = 1.7e-4
    rho_init: float = 1.0
    # Default ρ band kept at the legacy (0.01, 100.0) for backward-compat.
    # T19's brief recommends a wider (1e-3, 1e3) band for the standalone
    # helper; opt-in callers (e.g. the trainer's --enable-t19-adaptive-rho
    # path) override these explicitly.
    rho_min: float = 0.01
    rho_max: float = 100.0
    adaptive_rho_window: int = 16
    adaptive_rho_ratio: float = 10.0
    lambda_init: float = 1.0
    lambda_max: float = 1e6
    use_t19_adaptive_rho: bool = False
    t19_tau_grow: float = 2.0
    t19_tau_shrink: float = 0.5

    def __post_init__(self) -> None:
        if self.rho_min <= 0:
            raise ValueError(f"rho_min must be > 0, got {self.rho_min}")
        if self.rho_max < self.rho_min:
            raise ValueError(
                f"rho_max ({self.rho_max}) must be >= rho_min ({self.rho_min})"
            )
        if not (self.rho_min <= self.rho_init <= self.rho_max):
            raise ValueError(
                f"rho_init ({self.rho_init}) must be in [rho_min, rho_max]"
            )
        if self.adaptive_rho_window < 1:
            raise ValueError("adaptive_rho_window must be >= 1")
        if self.adaptive_rho_ratio <= 1.0:
            raise ValueError(
                f"adaptive_rho_ratio must be > 1, got {self.adaptive_rho_ratio}"
            )
        if self.lambda_max <= 0:
            raise ValueError("lambda_max must be > 0")
        if self.t19_tau_grow <= 1.0:
            raise ValueError(
                f"t19_tau_grow must be > 1, got {self.t19_tau_grow}"
            )
        if not (0.0 < self.t19_tau_shrink < 1.0):
            raise ValueError(
                f"t19_tau_shrink must be in (0, 1), got {self.t19_tau_shrink}"
            )


@dataclass
class LagrangianStepResult:
    """Result of one ADMM step.

    Attributes
    ----------
    augmented_lagrangian : torch.Tensor
        Scalar tensor (with grad) the trainer must call ``.backward()`` on.
    primal_residuals : dict[str, float]
        Constraint violations at this step (rate / seg / pose).
    dual_residuals : dict[str, float]
        Lambda updates (rate / seg / pose).
    rho : float
        Current ρ AFTER any adaptive update.
    lambdas : dict[str, float]
        Current λ AFTER updates (post-step).
    rho_changed : bool
        True if ρ was adjusted by the adaptive rule this step.
    """

    augmented_lagrangian: torch.Tensor
    primal_residuals: dict[str, float]
    dual_residuals: dict[str, float]
    rho: float
    lambdas: dict[str, float]
    rho_changed: bool = False


class JointLagrangianADMM:
    """Boyd-style augmented-Lagrangian coordinator for T1's three constraints.

    Examples
    --------
    >>> coord = JointLagrangianADMM()
    >>> # In the trainer's step:
    >>> distortion = compute_pixel_l1(decoder_out, target)
    >>> rate_bits = balle_out['rate_total_bits']
    >>> seg_loss = compute_segnet_distortion(...)
    >>> pose_loss = compute_posenet_distortion(...)
    >>> result = coord.step(
    ...     distortion=distortion,
    ...     rate_bits=rate_bits,
    ...     seg_loss=seg_loss,
    ...     pose_loss=pose_loss,
    ... )
    >>> result.augmented_lagrangian.backward()
    """

    def __init__(self, config: JointLagrangianADMMConfig | None = None):
        self.config = config or JointLagrangianADMMConfig()
        self.rho: float = float(self.config.rho_init)
        self.lambdas: dict[str, float] = {
            "rate": float(self.config.lambda_init),
            "seg": float(self.config.lambda_init),
            "pose": float(self.config.lambda_init),
        }
        # Scaled dual variable u (Boyd ADMM scaled form).
        self.u: dict[str, float] = {"rate": 0.0, "seg": 0.0, "pose": 0.0}
        # Residual histories for adaptive ρ.
        self._primal_history: deque[float] = deque(
            maxlen=self.config.adaptive_rho_window
        )
        self._dual_history: deque[float] = deque(
            maxlen=self.config.adaptive_rho_window
        )
        self.step_count: int = 0
        # T19 trajectory log: each entry is a dict with
        # {step, rho_before, rho_after, direction, ratio, primal, dual,
        #  source: "t19" | "legacy"}.
        # The trainer flushes this to a side log when --enable-t19-adaptive-rho
        # is active. Always populated (cheap; few entries per epoch).
        self.rho_trajectory: list[dict[str, Any]] = []

    @property
    def rate_target_bits(self) -> float:
        return float(self.config.rate_target_bytes) * 8.0

    def _rate_residual(self, rate_bits: torch.Tensor) -> torch.Tensor:
        # Normalise to dimensionless [-1, 1] band: (R - R_target) / R_target
        return (rate_bits - self.rate_target_bits) / max(self.rate_target_bits, 1.0)

    def _seg_residual(self, seg_loss: torch.Tensor) -> torch.Tensor:
        return (seg_loss - self.config.seg_target) / max(self.config.seg_target, 1e-12)

    def _pose_residual(self, pose_loss: torch.Tensor) -> torch.Tensor:
        return (pose_loss - self.config.pose_target) / max(self.config.pose_target, 1e-12)

    def step(
        self,
        *,
        distortion: torch.Tensor,
        rate_bits: torch.Tensor,
        seg_loss: torch.Tensor,
        pose_loss: torch.Tensor,
    ) -> LagrangianStepResult:
        """Compute the augmented Lagrangian + advance the dual updates.

        Returns
        -------
        LagrangianStepResult
            Carries the scalar tensor for ``.backward()`` plus updated state.
        """
        cfg = self.config
        rho = self.rho

        r_rate = self._rate_residual(rate_bits)
        r_seg = self._seg_residual(seg_loss)
        r_pose = self._pose_residual(pose_loss)

        # Augmented Lagrangian (scaled-dual form):
        #   L = D + Σ_c [ λ_c · r_c + (ρ/2) · (r_c + u_c)² ]
        u_rate = self.u["rate"]
        u_seg = self.u["seg"]
        u_pose = self.u["pose"]
        lam_rate = self.lambdas["rate"]
        lam_seg = self.lambdas["seg"]
        lam_pose = self.lambdas["pose"]

        aug = (
            distortion
            + lam_rate * r_rate
            + 0.5 * rho * (r_rate + u_rate) ** 2
            + lam_seg * r_seg
            + 0.5 * rho * (r_seg + u_seg) ** 2
            + lam_pose * r_pose
            + 0.5 * rho * (r_pose + u_pose) ** 2
        )

        # Dual updates use the DETACHED residuals (the .item() values).
        # The augmented term carries the gradient through the residual, but
        # the Lagrange multiplier itself is updated WITHOUT gradient.
        with torch.no_grad():
            r_rate_val = float(r_rate.detach().item())
            r_seg_val = float(r_seg.detach().item())
            r_pose_val = float(r_pose.detach().item())

            self.u["rate"] = float(u_rate + r_rate_val)
            self.u["seg"] = float(u_seg + r_seg_val)
            self.u["pose"] = float(u_pose + r_pose_val)

            # λ updates: project to [0, lambda_max] for inequality constraints.
            new_lam_rate = max(0.0, min(cfg.lambda_max, lam_rate + rho * r_rate_val))
            new_lam_seg = max(0.0, min(cfg.lambda_max, lam_seg + rho * r_seg_val))
            new_lam_pose = max(0.0, min(cfg.lambda_max, lam_pose + rho * r_pose_val))
            dual_residuals = {
                "rate": new_lam_rate - lam_rate,
                "seg": new_lam_seg - lam_seg,
                "pose": new_lam_pose - lam_pose,
            }
            self.lambdas = {
                "rate": new_lam_rate,
                "seg": new_lam_seg,
                "pose": new_lam_pose,
            }

            primal_norm = math.sqrt(r_rate_val ** 2 + r_seg_val ** 2 + r_pose_val ** 2)
            dual_norm = math.sqrt(
                dual_residuals["rate"] ** 2
                + dual_residuals["seg"] ** 2
                + dual_residuals["pose"] ** 2
            )
            self._primal_history.append(primal_norm)
            self._dual_history.append(dual_norm)

            rho_changed = self._maybe_adapt_rho()

        self.step_count += 1
        return LagrangianStepResult(
            augmented_lagrangian=aug,
            primal_residuals={
                "rate": r_rate_val,
                "seg": r_seg_val,
                "pose": r_pose_val,
            },
            dual_residuals=dual_residuals,
            rho=self.rho,
            lambdas=dict(self.lambdas),
            rho_changed=rho_changed,
        )

    def _maybe_adapt_rho(self) -> bool:
        """Adaptive-ρ rule (Boyd §3.4.1, scaled-dual form).

        Two backends:

        * Legacy (default; ``cfg.use_t19_adaptive_rho == False``):
          windowed-average grow/shrink rule. If
          ``primal / dual > τ`` over the recent window, increase ρ;
          if ``dual / primal > τ`` over the recent window, decrease ρ.
        * T19 (``cfg.use_t19_adaptive_rho == True``): per-step Boyd
          §3.4.1 / He-Yang 2000 update via
          :func:`tac.joint_admm_coordinator.adaptive_rho_step`. The
          standalone helper is the canonical form recommended by the
          coherence council 2026-05-09 (see memory
          ``feedback_t11_t13_t19_free_lateral_leaps_landed_20260509`` and
          ``feedback_grand_council_portfolio_coherence_journal_grade_20260509``).
        Both backends are bounded to ``[rho_min, rho_max]`` and append a
        ``{step, rho_before, rho_after, direction, ratio, primal, dual,
        source}`` entry to ``self.rho_trajectory`` whenever ρ changes.
        """
        cfg = self.config
        old_rho = self.rho

        if cfg.use_t19_adaptive_rho:
            # Lazy import to keep this module importable when
            # tac.joint_admm_coordinator dependencies are not present.
            from tac.joint_admm_coordinator import (  # noqa: WPS433
                adaptive_rho_step,
            )
            if not self._primal_history or not self._dual_history:
                return False
            primal_curr = self._primal_history[-1]
            dual_curr = self._dual_history[-1]
            t19 = adaptive_rho_step(
                rho_curr=self.rho,
                primal_residual=primal_curr,
                dual_residual=dual_curr,
                mu=cfg.adaptive_rho_ratio,
                tau_grow=cfg.t19_tau_grow,
                tau_shrink=cfg.t19_tau_shrink,
                rho_min=cfg.rho_min,
                rho_max=cfg.rho_max,
            )
            self.rho = float(t19.rho_next)
            if self.rho != old_rho:
                self.rho_trajectory.append({
                    "step": self.step_count,
                    "rho_before": old_rho,
                    "rho_after": self.rho,
                    "direction": t19.direction,
                    "ratio": t19.ratio,
                    "primal": primal_curr,
                    "dual": dual_curr,
                    "source": "t19",
                })
                # Reset history after a ρ change so the next adaptation
                # measures residuals at the new ρ scale.
                self._primal_history.clear()
                self._dual_history.clear()
                return True
            return False

        # Legacy backend.
        if len(self._primal_history) < self._primal_history.maxlen:
            return False
        primal_avg = sum(self._primal_history) / len(self._primal_history)
        dual_avg = sum(self._dual_history) / len(self._dual_history)
        direction = "hold"
        if dual_avg > 0 and primal_avg / max(dual_avg, 1e-12) > cfg.adaptive_rho_ratio:
            self.rho = min(cfg.rho_max, self.rho * 2.0)
            direction = "grow"
        elif primal_avg > 0 and dual_avg / max(primal_avg, 1e-12) > cfg.adaptive_rho_ratio:
            self.rho = max(cfg.rho_min, self.rho * 0.5)
            direction = "shrink"
        else:
            return False
        if self.rho != old_rho:
            self.rho_trajectory.append({
                "step": self.step_count,
                "rho_before": old_rho,
                "rho_after": self.rho,
                "direction": direction,
                "ratio": (primal_avg / max(dual_avg, 1e-12)),
                "primal": primal_avg,
                "dual": dual_avg,
                "source": "legacy",
            })
        # Reset history after a ρ change so we measure on the new scale.
        self._primal_history.clear()
        self._dual_history.clear()
        return self.rho != old_rho

    def state_dict(self) -> dict[str, Any]:
        return {
            "rho": self.rho,
            "lambdas": dict(self.lambdas),
            "u": dict(self.u),
            "step_count": self.step_count,
            "primal_history": list(self._primal_history),
            "dual_history": list(self._dual_history),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rho = float(state["rho"])
        self.lambdas = {k: float(v) for k, v in state["lambdas"].items()}
        self.u = {k: float(v) for k, v in state["u"].items()}
        self.step_count = int(state["step_count"])
        self._primal_history = deque(
            (float(x) for x in state.get("primal_history", [])),
            maxlen=self.config.adaptive_rho_window,
        )
        self._dual_history = deque(
            (float(x) for x in state.get("dual_history", [])),
            maxlen=self.config.adaptive_rho_window,
        )
