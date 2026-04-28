"""Lagrangian-controlled KL-distillation auxiliary weight.

Replaces the hand-derived ``--kl-distill-weight`` constant in
``experiments/optimize_poses.py`` with a Lagrangian dual-ascent rule
that maintains a TARGET signal-to-noise ratio (KL contribution / scorer
contribution).

Problem statement
-----------------
Let ``s_t`` denote the scorer loss at step ``t`` (the canonical
``seg_weight * seg_loss + pose_weight * pose_loss`` residual that drives
pose TTO) and ``k_t`` denote the raw KL-distillation auxiliary loss
(``tac.losses.kl_distill_segnet_only`` value at step ``t``). The aim of
Lane G is to use the KL signal as an *auxiliary guide* — strong enough
to nudge the renderer toward Quantizr-style soft-label SegNet logits but
weak enough that the primary scorer gradient still dominates.

We formalise that goal as the constrained optimisation::

    min_{θ}  E_t[s_t(θ)]                       (primary objective)
    s.t.     E_t[w · k_t(θ)] = ρ · E_t[s_t(θ)] (SNR-target equality)

where ``w`` is the auxiliary KL weight (the parameter to control) and
``ρ`` (``snr_target``) is the operator-supplied "auxiliary signal-to-
noise ratio" (default ``0.10`` = KL term should be 10% of scorer term,
per the canonical Hinton 2015 auxiliary-distill regime).

The Lagrangian is::

    L(θ, λ) = E_t[s_t(θ)] + λ · (E_t[w · k_t(θ)] - ρ · E_t[s_t(θ)])

Per Boyd & Vandenberghe (2004) §5.4, when the per-step proxy
``snr_t = (w_t · k_t) / max(s_t, ε)`` is positive (always — both
terms are non-negative) the geometric dual update::

    log w_{t+1} = log w_t + η · log(ρ / snr_t)
    snr_t       = (w_t · k_t) / (s_t + ε)

is gradient ascent on the strongly-concave KL-projected dual surrogate
``Φ(log w) = ρ · log w - log(E[k/s] · w)``. ``Φ`` is concave (linear
plus -log of a positive linear functional) and its unique stationary
point is at ``E[w · k / s] = ρ``, i.e. ``E[snr] = ρ``. Geometric (vs
arithmetic) convergence holds because Δ(log w) is proportional to
``log(distance-from-target)`` — the same convergence rate as the
exponentiated-gradient family (Kivinen & Warmuth 1997) for ratio
constraints. With constant ``η ∈ (0, 1]`` and bounded iterates the
Robbins-Monro diminishing-step regime (Robbins & Monro 1951) is
satisfied in expectation.

The multiplicative update keeps ``w > 0`` automatically and is the
canonical exponential-gradient (Kivinen & Warmuth 1997) move on the
positive cone — an explicit projection is unnecessary.

Stability guards
----------------
1. We bound ``log w`` to ``[log w_min, log w_max]`` to prevent runaway
   when ``s_t`` is transiently near zero (project onto a compact set ⇒
   strong duality holds trivially per Boyd §5.4).
2. We add ``ε = 1e-8`` to the denominator to avoid division by zero on
   the very first steps (when both losses are tiny).
3. The very first step uses the supplied ``initial_weight`` without a
   dual update (no observation available yet).

References
----------
Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge.
    See §5.4 (strong duality, Slater's condition).
Kivinen, J. & Warmuth, M. (1997). *Exponentiated Gradient versus
    Gradient Descent for Linear Predictors*. Information & Computation.
Robbins, H. & Monro, S. (1951). *A Stochastic Approximation Method*.
    Annals of Mathematical Statistics 22(3).
Hinton, G., Vinyals, O. & Dean, J. (2015). *Distilling the Knowledge in
    a Neural Network*. NIPS Deep Learning Workshop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["LearnableKLWeight"]


@dataclass
class LearnableKLWeight:
    """Lagrangian dual-ascent controller for the KL-distill auxiliary weight.

    Parameters
    ----------
    snr_target:
        Target ratio of KL contribution to scorer contribution. Default
        ``0.10`` = KL is 10% of the scorer loss (Hinton 2015 auxiliary
        regime).
    initial_weight:
        Starting weight ``w_0``. Default ``0.002`` matches the Lane G v3
        hand-derived constant (so the controller starts from the
        previously-shipped operating point and only moves if the
        observed SNR drifts).
    eta:
        Dual-ascent step size in log-space (multiplier on
        ``log(ρ / snr_t)``). Default ``0.5`` — for ``η ≤ 1`` the
        geometric update is contractive on log-distance; ``η = 0.5``
        approaches the target by 50% per step in log-space when the
        observation is stationary, giving ~10 steps from a 1000× off-
        target initial weight.
    log_weight_min, log_weight_max:
        Bounds on ``log w`` to enforce stability when the scorer loss
        is transiently tiny. Defaults span 6 orders of magnitude
        around 1.0 (``e^-9 ≈ 1.2e-4`` → ``e^3 ≈ 20``).
    eps:
        Numerical stabiliser added to ``s_t`` before division.
    """

    snr_target: float = 0.10
    initial_weight: float = 0.002
    eta: float = 0.5
    log_weight_min: float = -9.0
    log_weight_max: float = 3.0
    eps: float = 1e-8
    # Internal state — initialised in __post_init__.
    _log_weight: float = field(init=False, default=0.0)
    _step: int = field(init=False, default=0)
    _last_snr: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.snr_target <= 0:
            raise ValueError(
                f"snr_target must be > 0 (got {self.snr_target}); "
                f"a non-positive target leaves the dual update without a "
                f"well-defined sign."
            )
        if self.initial_weight <= 0:
            raise ValueError(
                f"initial_weight must be > 0 (got {self.initial_weight}); "
                f"the multiplicative update operates in log-space and "
                f"requires a strictly-positive starting iterate."
            )
        if self.eta <= 0:
            raise ValueError(
                f"eta must be > 0 (got {self.eta}); a non-positive step "
                f"size halts dual ascent."
            )
        if self.log_weight_max <= self.log_weight_min:
            raise ValueError(
                f"log_weight_max ({self.log_weight_max}) must exceed "
                f"log_weight_min ({self.log_weight_min}); the projection "
                f"set is otherwise empty."
            )
        self._log_weight = max(
            self.log_weight_min,
            min(self.log_weight_max, math.log(self.initial_weight)),
        )

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def weight(self) -> float:
        """Current auxiliary weight ``w_t`` (always > 0)."""
        return math.exp(self._log_weight)

    @property
    def last_snr(self) -> float | None:
        """Most recently observed signal-to-noise ratio, or ``None``
        before the first :py:meth:`update` call."""
        return self._last_snr

    @property
    def step_count(self) -> int:
        """Number of completed :py:meth:`update` calls."""
        return self._step

    def update(self, kl_value: float, scorer_value: float) -> float:
        """Apply one dual-ascent step using the latest observation.

        Parameters
        ----------
        kl_value:
            Raw KL loss value ``k_t`` (NOT pre-multiplied by ``w``).
        scorer_value:
            Scorer loss value ``s_t`` (the ``seg_weight * seg_loss +
            pose_weight * pose_loss`` residual driving pose TTO).

        Returns
        -------
        float
            The updated weight ``w_{t+1}`` (which the caller should use
            on the NEXT step). We deliberately delay the application by
            one step so the SNR observation matches the weight that
            produced it (causal control).
        """
        if not (kl_value >= 0 and math.isfinite(kl_value)):
            raise ValueError(
                f"kl_value must be a finite non-negative float "
                f"(got {kl_value!r}); KL divergence is non-negative by "
                f"definition."
            )
        if not math.isfinite(scorer_value):
            raise ValueError(
                f"scorer_value must be finite (got {scorer_value!r})."
            )
        # Compute observed SNR with the CURRENT weight (the one that
        # produced this observation). Add eps in the denominator only.
        observed_snr = (self.weight * kl_value) / (scorer_value + self.eps)
        self._last_snr = float(observed_snr)
        # Geometric (exponentiated-gradient) dual ascent on the ratio
        # constraint: log w_{t+1} = log w_t + η · log(ρ / snr_t).
        # Geometric (vs arithmetic) so that the convergence rate is
        # uniform in scale — w can move by orders of magnitude in a
        # bounded number of steps regardless of where it started.
        # Floor observed_snr at eps to avoid log(0) on a degenerate
        # opening step where kl == 0 exactly.
        snr_for_log = max(observed_snr, self.eps)
        delta = self.eta * math.log(self.snr_target / snr_for_log)
        new_log_weight = self._log_weight + delta
        # Project onto the compact bound (Boyd §5.4 strong duality).
        new_log_weight = max(self.log_weight_min, min(self.log_weight_max, new_log_weight))
        self._log_weight = new_log_weight
        self._step += 1
        return self.weight

    def state_dict(self) -> dict:
        """Serialize controller state for resume."""
        return {
            "snr_target": self.snr_target,
            "initial_weight": self.initial_weight,
            "eta": self.eta,
            "log_weight_min": self.log_weight_min,
            "log_weight_max": self.log_weight_max,
            "eps": self.eps,
            "_log_weight": self._log_weight,
            "_step": self._step,
            "_last_snr": self._last_snr,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a previous :py:meth:`state_dict`."""
        for key in (
            "snr_target", "initial_weight", "eta",
            "log_weight_min", "log_weight_max", "eps",
            "_log_weight", "_step", "_last_snr",
        ):
            if key in state:
                setattr(self, key, state[key])
