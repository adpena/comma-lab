# SPDX-License-Identifier: MIT
"""Riemannian SGD on SE(3) for Lane RM pose TTO.

This optimiser implements the Riemannian gradient descent recipe of
Bonnabel (2013) on the SE(3) manifold:

    1. Compute the Euclidean gradient of the loss with respect to the
       (axis-angle ω, translation t) parameterisation of each pose.
    2. Project that gradient onto the tangent space at the current point
       (with the canonical left-invariant metric on SE(3) the projection
       is the identity — see ``riemannian_gradient_se3``).
    3. Take a geodesic step via the SE(3) exponential map; the rotation
       update is multiplicative (R_new = R · exp(η̂)), guaranteeing that
       the rotation factor remains in SO(3) up to floating-point
       round-off.
    4. (Optional) Riemannian momentum: parallel-transport the previous
       velocity vector along the geodesic before adding it to the new
       gradient; for SE(3) with the left-invariant metric the parallel
       transport along a left-multiplication geodesic is a no-op on the
       Lie-algebra coordinates (Sola Eq. 174), so we implement momentum
       as a vector accumulator in se(3) coordinates.

References
----------
* Bonnabel, *Stochastic gradient descent on Riemannian manifolds*,
  IEEE TAC 58(9), 2013 — convergence rate matches Euclidean SGD.
* Boumal, *An Introduction to Optimization on Smooth Manifolds*,
  Cambridge UP, 2023 — Chapter 10 (Riemannian SGD), §10.5 (momentum).
* Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix
  Manifolds*, Princeton UP, 2008 — Chapter 4 (line-search), §4.1
  (retractions and convergence).
* Sola, Deray, Atchuthan, *A micro Lie theory for state estimation in
  robotics*, arXiv:1812.01537, 2018 — closed-form SE(3) primitives we
  call into (``src/tac/se3.py``).

Pose-tensor contract
--------------------
This optimiser interprets each row of a parameter tensor of shape
``(N, 6)`` (or batched ``(..., 6)``) as a single SE(3) element split as
``(ω ∈ ℝ³, t ∈ ℝ³)``; ``ω`` is axis-angle (so ``omega = log_map_so3(R)``)
and ``t`` is the Cartesian translation. This matches the comma.ai PoseNet
6-vector output, where the first three dims are angular and the last
three are translational. Tensors with a last-dim other than 6 are
rejected at ``step()`` time to fail loud rather than silently corrupt a
LoRA / radial-zoom parameterisation.

Why a custom optimiser (instead of running plain SGD then re-projecting)
-----------------------------------------------------------------------
* Correctness: a Euclidean SGD step on (ω, t) does NOT correspond to a
  geodesic on SO(3) — it accumulates orthogonality error which compounds
  across steps. The exponential-map composition does so by construction.
* Convergence: Bonnabel (2013) shows Riemannian SGD attains the same
  asymptotic convergence rate as Euclidean SGD on smooth manifolds, but
  with a constant factor improved by the manifold curvature when the
  parameter space has nontrivial structure (which SO(3) does — its
  diameter under the Frobenius metric is bounded, while ℝ³ is not).
* Specifically for the pose TTO loop: per
  ``project_posenet_rank1_discovery``, dim 0 of the pose carries 99.8%
  of the variance and lies on a near-trivial manifold, while dims 1-5
  are nuisance directions on a compact submanifold (effectively SO(3)).
  A Riemannian optimiser stays on that submanifold by construction;
  Euclidean SGD wanders off it.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer

from .se3 import (
    batched_geodesic_step_axis_angle,
    inverse_left_jacobian_so3,
    mark_tangent_as_lie_algebra,
)


# ──────────────────────────────────────────────────────────────────────────
# Public optimiser
# ──────────────────────────────────────────────────────────────────────────


class RiemannianSGD(Optimizer):
    """Riemannian SGD on SE(3) with optional Lie-algebra momentum.

    Parameters
    ----------
    params : iterable of parameter tensors
        Each tensor's last dim must be 6 — interpreted as
        ``(ω_x, ω_y, ω_z, t_x, t_y, t_z)``.
    lr : float
        Geodesic step size; descent direction is ``-grad`` so the
        effective step is ``-lr · grad``.
    momentum : float, default 0.0
        Riemannian momentum coefficient β. The velocity buffer lives in
        se(3) coordinates: ``v_{k+1} = β · v_k + grad`` and the geodesic
        step uses ``-lr · v_{k+1}`` (Polyak heavy-ball formulation). With
        the left-invariant metric, parallel transport on the Lie algebra
        is the identity (Sola §6), so no transport correction is needed
        between steps.
    weight_decay : float, default 0.0
        L2 penalty on the (ω, t) coordinates added to the gradient before
        the momentum update — exactly as in ``torch.optim.SGD``. Note that
        an L2 penalty in the chart is not a manifold-intrinsic regulariser;
        it biases the optimiser toward the origin of the chart, which for
        axis-angle is the identity rotation. Default 0 leaves behaviour
        unchanged.

    Notes
    -----
    * If a parameter has no gradient (``param.grad is None``), it is
      skipped — same convention as ``torch.optim.SGD``.
    * The optimiser is *non-differentiable*: ``step()`` mutates parameters
      in-place under ``torch.no_grad`` and detaches the result. Do not
      backprop through it.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401 — match torch.optim signature
        """Perform a single geodesic optimisation step.

        Parameters
        ----------
        closure : callable, optional
            Same semantics as ``torch.optim.Optimizer.step``: if provided,
            it is evaluated under ``torch.enable_grad`` to recompute the
            loss, and its return value is returned to the caller.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            weight_decay: float = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.shape[-1] != 6:
                    raise ValueError(
                        "RiemannianSGD parameters must have last dim == 6 "
                        f"(interpreted as (omega, t)); got {tuple(param.shape)}"
                    )

                grad = param.grad
                if weight_decay != 0.0:
                    grad = grad + weight_decay * param

                # Riemannian momentum: keep a velocity buffer in se(3)
                # coordinates. The buffer lives in optimiser state per
                # parameter tensor (matching torch.optim.SGD's pattern).
                if momentum != 0.0:
                    state = self.state[param]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(grad)
                        state["momentum_buffer"] = buf
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                # Split into rotation / translation tangent components.
                # The descent direction is -grad, so we pass step = -lr to
                # the batched geodesic helper.
                grad_omega = grad[..., 0:3]
                grad_t_cartesian = grad[..., 3:6]

                omega_current = param[..., 0:3]
                t_current = param[..., 3:6]

                # Round 14 finding 2 (R15) — project the Cartesian
                # translation gradient through ``J_l(ω)^-1`` so the
                # increment lives in se(3) coordinates, matching the
                # Sola §4.5 retraction convention that
                # ``batched_geodesic_step_axis_angle`` applies. Without
                # this projection the batched helper composes two J_l's
                # and silently mis-updates the translation. Mirrors the
                # per-element ``riemannian_gradient_se3`` projection.
                Jl_inv = inverse_left_jacobian_so3(omega_current)  # (..., 3, 3)
                grad_v = (Jl_inv @ grad_t_cartesian.unsqueeze(-1)).squeeze(-1)
                # Stamp the Lie-algebra sentinel so the batched helper's
                # contract check accepts the tangent without raising.
                mark_tangent_as_lie_algebra(grad_v)

                omega_new, t_new = batched_geodesic_step_axis_angle(
                    omega_current,
                    t_current,
                    grad_omega,
                    grad_v,
                    step_size=-lr,
                )

                # In-place write back. Using copy_ keeps the parameter
                # identity stable so downstream consumers (and the autograd
                # graph that recomputes the loss next step) keep their
                # references.
                param[..., 0:3].copy_(omega_new)
                param[..., 3:6].copy_(t_new)

        return loss
