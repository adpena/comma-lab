# SPDX-License-Identifier: MIT
"""Langevin SDE optimizer for substrate polish phase (beat-PR95 design Idea 2).

This module implements the discretized Langevin stochastic differential equation
(SDE) as a ``torch.optim.Optimizer`` for use in the polish phase of HNeRV-family
substrate training. The mathematical formulation:

    dθ_t = -∇L(θ_t) dt + sqrt(2 T_t) dW_t

where ``dW_t`` is multivariate standard Brownian motion and ``T_t`` is a
deterministic temperature schedule from ``T_init`` down to ``T_final``.

Discretization via Euler-Maruyama:

    θ_{t+1} = θ_t - lr · ∇L(θ_t) + sqrt(2 T_t · lr) · ξ,    ξ ~ N(0, I)

**Why Langevin for the polish phase?** At fixed T, the SDE's stationary
distribution is the Gibbs measure ``p(θ) ∝ exp(-L(θ)/T)``. As T → 0, mass
concentrates on ``argmin L``. With slow-enough annealing (Geman & Geman 1984
logarithmic schedule), convergence to GLOBAL minimum is provable. Faster
schedules (cosine, exponential) provably converge to local-min only — but
empirically dominate pure-gradient methods on non-convex landscapes
(Welling & Teh 2011 SGLD).

PR95's Stage 8 progression AdamW → Muon is the empirical workaround for HNeRV's
plateau-and-sharp-minimum coexistence. Langevin is the first-principles correct
answer: thermal fluctuation escapes plateaus.

**Apples-to-apples evidence discipline**: this optimizer carries no score claim
in itself. Any empirical contest-score result requires the standard dispatch
discipline (CUDA + CPU paired auth eval on 1:1 contest-CI hardware per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA").

**Cross-references**:
- Design memo: ``.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md``
- Memory anchor: ``feedback_beat_pr95_curriculum_substrate_training_design_landed_20260513.md``
- Prior per-pixel GARCH-Langevin: ``tac.contrib.finance_optimizers``
  (NOT a substitute — that variant operates in pixel space, not parameter space)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch

__all__ = [
    "LangevinOptimizer",
    "cosine_temperature_schedule",
    "exponential_temperature_schedule",
    "geman_geman_log_schedule",
]

ScheduleFn = Callable[[int, int, float, float], float]
"""Signature: ``schedule(step, n_steps, T_init, T_final) -> T_t``."""


def cosine_temperature_schedule(
    step: int, n_steps: int, T_init: float, T_final: float
) -> float:
    """Cosine annealing from ``T_init`` at step 0 to ``T_final`` at step ``n_steps``.

    Convergence: local-min only (faster than Geman-Geman). Default for the polish
    phase where the curriculum has already driven θ to a well-prepared starting
    point.
    """
    if n_steps <= 0:
        return T_final
    t = max(0, min(step, n_steps))
    ratio = t / float(n_steps)
    return float(T_final + 0.5 * (T_init - T_final) * (1.0 + math.cos(math.pi * ratio)))


def geman_geman_log_schedule(
    step: int, n_steps: int, T_init: float, T_final: float
) -> float:
    """Geman & Geman (1984) logarithmic cooling: ``T_t = c / log(2 + t)``.

    Convergence: GLOBAL minimum (provably, when SDE is integrated for infinite
    time). Practical cost: extremely slow — included for completeness and for the
    probe-disambiguator design tension B (cosine vs log) per CLAUDE.md Catalog
    #125 hook 6.

    The constant ``c`` is chosen so that ``T(0) = T_init`` and
    ``T(n_steps) = T_final``:

        c = (T_init - T_final) / (1/log(2) - 1/log(2 + n_steps))
        T_final = T(n_steps) actual

    For practical use, we APPROXIMATE by interpolating between the two endpoint
    values using a normalized log curve, ensuring monotone decrease.
    """
    if n_steps <= 0:
        return T_final
    if step <= 0:
        return T_init
    if step >= n_steps:
        return T_final
    # Normalized log: log(2 + step) maps [0, n_steps] to [log(2), log(2+n_steps)]
    log_0 = math.log(2.0)
    log_n = math.log(2.0 + n_steps)
    log_t = math.log(2.0 + step)
    # Map log_t in [log_0, log_n] to [0, 1] then invert (smaller t → larger T)
    progress = (log_t - log_0) / (log_n - log_0)
    return float(T_init + (T_final - T_init) * progress)


def exponential_temperature_schedule(
    step: int, n_steps: int, T_init: float, T_final: float
) -> float:
    """Exponential decay from ``T_init`` to ``T_final``."""
    if n_steps <= 0:
        return T_final
    t = max(0, min(step, n_steps))
    if T_init <= 0.0 or T_final <= 0.0:
        # Fall back to linear when log is ill-defined
        return float(T_init + (T_final - T_init) * (t / float(n_steps)))
    ratio = math.exp(math.log(T_final / T_init) * (t / float(n_steps)))
    return float(T_init * ratio)


SCHEDULES: dict[str, ScheduleFn] = {
    "cosine": cosine_temperature_schedule,
    "log": geman_geman_log_schedule,
    "geman_geman": geman_geman_log_schedule,
    "exp": exponential_temperature_schedule,
    "exponential": exponential_temperature_schedule,
}


class LangevinOptimizer(torch.optim.Optimizer):
    """Discretized Langevin SDE optimizer (Euler-Maruyama integration).

    Update rule per parameter ``p`` with gradient ``g`` at step ``t``:

        p ← p − lr · g  +  sqrt(2 · T_t · lr) · ξ,    ξ ~ N(0, I)

    where ``T_t`` is from the annealing schedule.

    **Important**: the noise term ``sqrt(2 T lr)`` (NOT ``sqrt(2 T lr · dt²)``)
    is the correct Euler-Maruyama discretization with ``dt = lr``, NOT a
    separately-parameterized timestep. This matches the SGLD literature
    (Welling & Teh 2011 eq. 6) and the textbook reference (Kloeden & Platen
    1992 §10.2 for SDE numerics).

    **Quantization-aware integration**: when used in the polish phase of QAT
    training (PR95 Stage 5+), the Langevin noise amplitude
    ``sqrt(2 T_final · lr) ≈ 1e-4`` is BELOW the INT8 quantization grid (1/127
    ≈ 7.9e-3). The noise gets ROUNDED AWAY by the next fake_quantize call,
    bounding the worst-case adversarial-steganalysis surface increment to zero
    on quantized weights. Apply Langevin to FP32 shadow weights ONLY.

    Args:
        params: iterable of parameter tensors (or parameter groups, per the
            standard ``torch.optim.Optimizer`` API)
        lr: step size ``η`` (also the discretization ``dt``). Default 1e-4.
        T_init: initial temperature for the annealing schedule. Default 1.0.
        T_final: final temperature for the annealing schedule. Default 1e-4.
        n_steps: total number of steps the schedule runs over. Default 2000.
        weight_decay: L2 weight decay coefficient (added to gradient BEFORE
            the noise injection). Default 0.0.
        schedule: temperature-schedule name; one of {"cosine", "log",
            "exp"}. Default "cosine".
        noise_seed: optional torch.Generator seed for determinism in tests.
            Default None (uses global RNG state).

    Example:
        >>> model = torch.nn.Linear(10, 1)
        >>> opt = LangevinOptimizer(model.parameters(), lr=1e-4,
        ...                          T_init=1.0, T_final=1e-4, n_steps=2000)
        >>> loss_fn = torch.nn.MSELoss()
        >>> for step in range(2000):
        ...     opt.zero_grad()
        ...     y = model(torch.randn(8, 10))
        ...     loss = loss_fn(y, torch.zeros(8, 1))
        ...     loss.backward()
        ...     opt.step()
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-4,
        T_init: float = 1.0,
        T_final: float = 1e-4,
        n_steps: int = 2000,
        weight_decay: float = 0.0,
        schedule: str = "cosine",
        noise_seed: int | None = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if T_init < T_final:
            raise ValueError(
                f"T_init ({T_init}) must be >= T_final ({T_final}); "
                "annealing temperatures must monotone-decrease"
            )
        if T_init < 0.0 or T_final < 0.0:
            raise ValueError(
                f"temperatures must be non-negative, got T_init={T_init}, T_final={T_final}"
            )
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if schedule not in SCHEDULES:
            raise ValueError(
                f"unknown schedule {schedule!r}; must be one of {list(SCHEDULES)}"
            )
        defaults = {
            "lr": float(lr),
            "T_init": float(T_init),
            "T_final": float(T_final),
            "n_steps": int(n_steps),
            "weight_decay": float(weight_decay),
            "schedule": schedule,
        }
        super().__init__(params, defaults)
        self._step_count: int = 0
        self._schedule_fn: ScheduleFn = SCHEDULES[schedule]
        if noise_seed is not None:
            # Per-instance generator for deterministic tests; not used to update
            # the global RNG state
            self._generator: torch.Generator | None = torch.Generator()
            self._generator.manual_seed(int(noise_seed))
        else:
            self._generator = None

    @property
    def current_step(self) -> int:
        """Number of completed ``step()`` calls."""
        return self._step_count

    def current_temperature(self) -> float:
        """Temperature at the CURRENT step (before the next update)."""
        d = self.defaults
        return self._schedule_fn(
            self._step_count, d["n_steps"], d["T_init"], d["T_final"]
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> Any:
        """Perform one Euler-Maruyama Langevin update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            n_steps = group["n_steps"]
            T_init = group["T_init"]
            T_final = group["T_final"]
            T_t = self._schedule_fn(self._step_count, n_steps, T_init, T_final)
            noise_scale = math.sqrt(max(2.0 * T_t * lr, 0.0))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)
                # Gradient descent component
                p.add_(grad, alpha=-lr)
                # Brownian noise component (skip if T_t = 0 or noise_scale = 0)
                if noise_scale > 0.0:
                    if self._generator is not None:
                        # Deterministic per-instance noise (test-only)
                        noise = torch.empty_like(p).normal_(
                            mean=0.0, std=1.0, generator=self._generator
                        )
                    else:
                        noise = torch.randn_like(p)
                    p.add_(noise, alpha=noise_scale)

        self._step_count += 1
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Serialize state (parent class state + the step counter)."""
        state = super().state_dict()
        state["_step_count"] = self._step_count
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state. Resumes from the saved step counter."""
        step_count = state_dict.pop("_step_count", 0)
        super().load_state_dict(state_dict)
        self._step_count = int(step_count)
