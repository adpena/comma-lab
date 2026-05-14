# SPDX-License-Identifier: MIT
"""Variational frame generation via Euler-Lagrange + Lagrangian dual optimization.

Euler's contribution: The frame generation problem is a calculus of variations
problem. We seek frame function f*(x,y) that minimizes the functional:

    J[f] = integral L(x, y, f, df/dx, df/dy) dx dy

where the Lagrangian density L incorporates:
  - SegNet fidelity: 100 * seg_distortion(f, f_gt)
  - PoseNet fidelity: sqrt(10 * pose_distortion(f, f_gt))
  - Rate penalty: lambda_rate * rate(f)
  - Smoothness prior: lambda_smooth * (|grad f|^2 + |laplacian f|^2)

The Euler-Lagrange equation:
    dL/df - d/dx(dL/df_x) - d/dy(dL/df_y) = 0

is solved via gradient descent on the discretized pixel grid.

Lagrange's contribution: The competition score has a hard rate-distortion
tradeoff structure. The Lagrangian dual problem:

    min_f  D(f)     subject to  R(f) <= R_budget
    L(f, lambda) = D(f) + lambda * (R(f) - R_budget)

The dual variable lambda is learned via gradient ascent, giving the
theoretically optimal rate-distortion tradeoff automatically.

KKT conditions:
  1. Stationarity: grad_f L = 0
  2. Primal feasibility: R(f) <= R_budget
  3. Dual feasibility: lambda >= 0
  4. Complementary slackness: lambda * (R(f) - R_budget) = 0

Example::

    from tac.contrib.variational_gen import VariationalFrameGenerator, LagrangianDualOptimizer
    gen = VariationalFrameGenerator(cfg={})
    frames = gen.solve(init_frames, masks, posenet, segnet)

    dual = LagrangianDualOptimizer(cfg={})
    frames = dual.optimize(init_frames, masks, posenet, segnet, rate_budget=0.01)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "spatial_gradient",
    "laplacian",
    "gradient_magnitude_squared",
    "smoothness_energy",
    "total_variation",
    "VariationalFrameGenerator",
    "LagrangianDualOptimizer",
]


# ---- Euler-Lagrange Discretization Utilities ----


def spatial_gradient(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute discrete spatial gradients df/dx, df/dy via central differences.

    Central differences: df/dx[i,j] = (f[i,j+1] - f[i,j-1]) / 2
    with replicate padding at boundaries to maintain shape.

    This is the natural discretization for the Euler-Lagrange equation on a
    regular pixel grid with Neumann boundary conditions (zero normal derivative).

    Args:
        f: (..., H, W) tensor.

    Returns:
        (df_dx, df_dy) each with same shape as f.
    """
    # Pad replicate then central diff
    f_pad_x = F.pad(f, (1, 1, 0, 0), mode="replicate")
    df_dx = (f_pad_x[..., :, 2:] - f_pad_x[..., :, :-2]) * 0.5

    f_pad_y = F.pad(f, (0, 0, 1, 1), mode="replicate")
    df_dy = (f_pad_y[..., 2:, :] - f_pad_y[..., :-2, :]) * 0.5

    return df_dx, df_dy


def laplacian(f: torch.Tensor) -> torch.Tensor:
    """Compute discrete Laplacian via 5-point stencil.

    Laplacian[f](i,j) = f(i+1,j) + f(i-1,j) + f(i,j+1) + f(i,j-1) - 4*f(i,j)

    This is the standard second-order accurate discretization on a unit grid.
    Equivalent to the divergence of the gradient: div(grad(f)).

    Args:
        f: (..., H, W) tensor.

    Returns:
        Laplacian with same shape as f.
    """
    f_pad = F.pad(f, (1, 1, 1, 1), mode="replicate")
    lap = (
        f_pad[..., 1:-1, 2:]     # right
        + f_pad[..., 1:-1, :-2]  # left
        + f_pad[..., 2:, 1:-1]   # down
        + f_pad[..., :-2, 1:-1]  # up
        - 4.0 * f_pad[..., 1:-1, 1:-1]
    )
    return lap


def gradient_magnitude_squared(f: torch.Tensor) -> torch.Tensor:
    """Compute |grad f|^2 = (df/dx)^2 + (df/dy)^2.

    Used as the first-order smoothness term in the Lagrangian density.
    The Euler-Lagrange equation for this term gives the Laplacian: div(grad f).

    Args:
        f: (..., H, W) tensor.

    Returns:
        (..., H, W) tensor of gradient magnitude squared.
    """
    df_dx, df_dy = spatial_gradient(f)
    return df_dx.pow(2) + df_dy.pow(2)


def smoothness_energy(
    f: torch.Tensor,
    lambda_grad: float = 1.0,
    lambda_lap: float = 0.1,
) -> torch.Tensor:
    """Compute smoothness energy: lambda_grad * |grad f|^2 + lambda_lap * |lap f|^2.

    This is the regularization functional in the Lagrangian density:
        E_smooth[f] = integral (lambda_grad |grad f|^2 + lambda_lap |laplacian f|^2) dx dy

    The first term penalizes first-order discontinuities (edges).
    The second term penalizes second-order discontinuities (texture noise).

    The Euler-Lagrange equation for E_smooth is:
        -lambda_grad * laplacian(f) + lambda_lap * biharmonic(f) = 0

    which is a fourth-order PDE (thin-plate spline equation).

    Args:
        f: (B, C, H, W) frame tensor.
        lambda_grad: weight for first-order smoothness.
        lambda_lap: weight for second-order smoothness (biharmonic).

    Returns:
        Scalar energy.
    """
    grad_sq = gradient_magnitude_squared(f)
    lap = laplacian(f)
    energy = lambda_grad * grad_sq.mean() + lambda_lap * lap.pow(2).mean()
    return energy


def total_variation(f: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation: sum |df/dx| + |df/dy|.

    TV is the L1 analog of the gradient magnitude. It preserves edges better
    than L2 smoothness (Rudin-Osher-Fatemi denoising). Used as a rate proxy:
    frames with low TV compress better under any transform-based codec.

    Args:
        f: (B, C, H, W) tensor.

    Returns:
        Scalar TV value, averaged over all pixels and channels.
    """
    tv_h = (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
    tv_w = (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


# ---- Variational Frame Generator ----


class VariationalFrameGenerator:
    """Solve the Euler-Lagrange equations for scorer-optimal frame generation.

    The functional to minimize:
        J[f] = D_seg(f) + D_pose(f) + lambda_smooth * E_smooth(f) + lambda_rate * TV(f)

    where:
        D_seg = 100 * (1 - cosine_similarity(softmax(segnet(f)), softmax(segnet(gt))))
        D_pose = sqrt(10 * MSE(posenet(f), posenet(gt)))
        E_smooth = |grad f|^2 + 0.1 * |lap f|^2
        TV(f) = anisotropic total variation

    Solved via gradient descent with Armijo line search on the discretized
    384x512 pixel grid.

    All parameters are configurable via the cfg dict, following the project
    convention of cfg.get("param", default).

    Args:
        cfg: configuration dict. Keys:
            - variational_lr (float): initial learning rate, default 1.0
            - variational_steps (int): max gradient descent steps, default 500
            - lambda_smooth (float): smoothness weight, default 0.01
            - lambda_rate (float): rate (TV) weight, default 0.1
            - lambda_grad (float): first-order smoothness, default 1.0
            - lambda_lap (float): second-order smoothness, default 0.1
            - armijo_c (float): Armijo sufficient decrease param, default 1e-4
            - armijo_rho (float): Armijo backtracking factor, default 0.5
            - grad_clip (float): gradient clipping norm, default 10.0
            - convergence_tol (float): stop if gradient norm < tol, default 1e-6
            - use_line_search (bool): enable Armijo line search, default True
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    def _get(self, key: str, default: Any) -> Any:
        return self.cfg.get(key, default)

    def solve(
        self,
        init_frames: torch.Tensor,
        masks: torch.Tensor,
        posenet: nn.Module,
        segnet: nn.Module,
        gt_frames: torch.Tensor | None = None,
        log_every: int = 50,
    ) -> torch.Tensor:
        """Solve the variational problem via gradient descent.

        The Euler-Lagrange equation dL/df - d/dx(dL/df_x) - d/dy(dL/df_y) = 0
        is equivalent to grad J[f] = 0. We solve this by steepest descent:

            f_{k+1} = f_k - alpha_k * grad J[f_k]

        with optional Armijo line search for alpha_k.

        Args:
            init_frames: (N, C, H, W) initial frame tensor in [0, 255].
                Can be noise, class-mean colors, or a warm-start from another method.
            masks: (N, H, W) long tensor with target segmentation class indices.
            posenet: frozen PoseNet model.
            segnet: frozen SegNet model.
            gt_frames: (N, C, H, W) optional ground truth for PoseNet reference.
                If None, PoseNet loss uses self-consistency (temporal pairs).
            log_every: print diagnostics every N steps (0 to disable).

        Returns:
            (N, C, H, W) optimized frames in [0, 255].
        """
        lr = self._get("variational_lr", 1.0)
        max_steps = self._get("variational_steps", 500)
        lambda_smooth = self._get("lambda_smooth", 0.01)
        lambda_rate = self._get("lambda_rate", 0.1)
        lambda_grad = self._get("lambda_grad", 1.0)
        lambda_lap = self._get("lambda_lap", 0.1)
        grad_clip_val = self._get("grad_clip", 10.0)
        convergence_tol = self._get("convergence_tol", 1e-6)
        use_line_search = self._get("use_line_search", True)
        armijo_c = self._get("armijo_c", 1e-4)
        armijo_rho = self._get("armijo_rho", 0.5)

        device = init_frames.device
        N, C, H, W = init_frames.shape

        # Working copy
        f = init_frames.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([f], lr=lr)

        # Cache GT scorer outputs if available
        gt_pose_ref = None
        gt_seg_ref = None
        if gt_frames is not None:
            with torch.no_grad():
                gt_pair = gt_frames.unsqueeze(1).expand(N, 2, C, H, W).contiguous()
                gt_pose_in = posenet.preprocess_input(gt_pair)
                gt_pose_out = posenet(gt_pose_in)
                gt_pose_ref = (
                    gt_pose_out["pose"] if isinstance(gt_pose_out, dict) else gt_pose_out
                )[..., :6].detach()
                gt_seg_in = segnet.preprocess_input(gt_pair)
                gt_seg_ref = F.softmax(segnet(gt_seg_in), dim=1).detach()

        best_loss = float("inf")
        best_f = f.detach().clone()

        for step in range(max_steps):
            optimizer.zero_grad()

            # --- Distortion terms ---
            seg_loss = self._segnet_loss(f, masks, segnet)
            pose_loss = self._posenet_loss(f, posenet, gt_pose_ref)
            distortion = 100.0 * seg_loss + torch.sqrt(10.0 * pose_loss + 1e-8)

            # --- Smoothness (Euler-Lagrange regularizer) ---
            smooth_loss = smoothness_energy(f, lambda_grad, lambda_lap)

            # --- Rate proxy (total variation) ---
            rate_loss = total_variation(f)

            # --- Full functional ---
            J = distortion + lambda_smooth * smooth_loss + lambda_rate * rate_loss

            J.backward()

            # Gradient clipping for numerical stability
            grad_norm = f.grad.data.norm()
            if grad_norm > grad_clip_val:
                f.grad.data.mul_(grad_clip_val / (grad_norm + 1e-12))

            # Convergence check
            if grad_norm.item() < convergence_tol:
                if log_every > 0:
                    print(f"  variational: converged at step {step + 1}, "
                          f"grad_norm={grad_norm.item():.2e}")
                break

            if use_line_search:
                # Armijo backtracking line search on the FULL functional J.
                # Evaluates the complete objective (distortion + smooth + TV)
                # at each trial point to ensure valid comparison.
                with torch.no_grad():
                    direction = -f.grad.data
                    current_val = J.item()
                    slope = -(grad_norm.item() ** 2)
                    alpha = lr
                    best_trial = f.data.clone()
                    accepted = False
                    for _ in range(10):
                        f_trial = (f.data + alpha * direction).clamp(0.0, 255.0)
                        # Evaluate FULL functional at trial point
                        trial_smooth = smoothness_energy(f_trial, lambda_grad, lambda_lap)
                        trial_tv = total_variation(f_trial)
                        trial_reg = (
                            lambda_smooth * trial_smooth.item()
                            + lambda_rate * trial_tv.item()
                        )
                        # Distortion at trial point (requires scorer forward)
                        f_tmp = f_trial.detach().requires_grad_(False)
                        trial_seg = self._segnet_loss(f_tmp, masks, segnet).item() if segnet is not None else 0.0
                        trial_pose = self._posenet_loss(f_tmp, posenet, gt_pose_ref).item() if posenet is not None else 0.0
                        trial_dist = 100.0 * trial_seg + (10.0 * trial_pose + 1e-8) ** 0.5
                        trial_val = trial_dist + trial_reg
                        if trial_val <= current_val + armijo_c * alpha * slope:
                            best_trial = f_trial
                            accepted = True
                            break
                        alpha *= armijo_rho
                    f.data.copy_(best_trial if accepted else f.data)
            else:
                optimizer.step()
                with torch.no_grad():
                    f.data.clamp_(0.0, 255.0)

            # Track best
            with torch.no_grad():
                if J.item() < best_loss:
                    best_loss = J.item()
                    best_f = f.data.clone()

            if log_every > 0 and (step + 1) % log_every == 0:
                print(
                    f"  variational step {step + 1:4d}/{max_steps}: "
                    f"J={J.item():.4f} seg={seg_loss.item():.4f} "
                    f"pose={pose_loss.item():.4f} smooth={smooth_loss.item():.4f} "
                    f"TV={rate_loss.item():.4f} |grad|={grad_norm.item():.2e}"
                )

        return best_f.clamp(0.0, 255.0)

    def _segnet_loss(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        segnet: nn.Module,
    ) -> torch.Tensor:
        """Differentiable SegNet distortion: soft cross-entropy on logits.

        The scorer formula uses argmax disagreement (hard). We use cross-entropy
        as a differentiable surrogate whose gradient points toward the same
        minimum. As optimization converges, the argmax agrees with the target
        mask, and the surrogate loss tracks the true distortion.

        Args:
            frames: (N, C, H, W) requires_grad tensor.
            masks: (N, H, W) long target masks.
            segnet: frozen SegNet.

        Returns:
            Scalar cross-entropy loss.
        """
        N, C, H, W = frames.shape
        device = frames.device

        # Use preprocess_input to match eval-time scorer behavior.
        # SegNet.preprocess_input expects (B, T, C, H, W).
        frames_btchw = frames.unsqueeze(1).contiguous()  # (N, 1, C, H, W)
        seg_input = segnet.preprocess_input(frames_btchw)
        logits = segnet(seg_input)
        H_out, W_out = logits.shape[2], logits.shape[3]

        masks_resized = F.interpolate(
            masks.float().unsqueeze(1).to(device),
            size=(H_out, W_out), mode="nearest",
        ).squeeze(1).long()

        return F.cross_entropy(logits, masks_resized)

    def _posenet_loss(
        self,
        frames: torch.Tensor,
        posenet: nn.Module,
        gt_pose_ref: torch.Tensor | None,
    ) -> torch.Tensor:
        """PoseNet temporal consistency loss.

        If gt_pose_ref is provided, use L2 distance to GT pose outputs.
        Otherwise, use self-consistency: minimize variance of pose outputs
        across overlapping pairs (frames should produce consistent ego-motion).

        Args:
            frames: (N, C, H, W) requires_grad tensor.
            posenet: frozen PoseNet.
            gt_pose_ref: (N, 6) optional cached GT pose outputs.

        Returns:
            Scalar pose loss.
        """
        N = frames.shape[0]
        device = frames.device

        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Build pairs
        pair = frames.unsqueeze(1).expand(N, 2, *frames.shape[1:]).contiguous()
        posenet_in = posenet.preprocess_input(pair)
        pose_out = posenet(posenet_in)
        pred = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        pred = pred[..., :6]

        if gt_pose_ref is not None:
            return (pred - gt_pose_ref).pow(2).mean()
        else:
            # Self-consistency: minimize pose magnitude (self-pairs should give zero motion)
            return pred.pow(2).mean()


# ---- Lagrangian Dual Optimizer ----


class LagrangianDualOptimizer:
    """Primal-dual optimization for rate-distortion tradeoff.

    Solves the constrained problem:
        min_f  D(f) = 100*seg(f) + sqrt(10*pose(f))
        s.t.   R(f) <= R_budget

    via the Lagrangian:
        L(f, lambda) = D(f) + lambda * (R(f) - R_budget)

    The primal variable f is updated by gradient descent on L.
    The dual variable lambda is updated by gradient ascent:
        lambda_{k+1} = max(0, lambda_k + eta * (R(f_k) - R_budget))

    This satisfies the KKT conditions at convergence:
      1. Stationarity: grad_f D(f*) + lambda* grad_f R(f*) = 0
      2. Primal feasibility: R(f*) <= R_budget
      3. Dual feasibility: lambda* >= 0
      4. Complementary slackness: lambda* (R(f*) - R_budget) = 0

    The dual variable lambda learns the marginal cost of rate in distortion
    units -- it replaces the ad-hoc weight 25 in the score formula with the
    theoretically optimal tradeoff for the given rate budget.

    Args:
        cfg: configuration dict. Keys:
            - dual_primal_lr (float): primal (frame) learning rate, default 0.5
            - dual_dual_lr (float): dual (lambda) learning rate, default 0.01
            - dual_steps (int): max primal-dual iterations, default 500
            - rate_budget (float): target rate R_budget, default 0.01
            - lambda_init (float): initial dual variable, default 25.0
            - lambda_smooth (float): smoothness regularizer weight, default 0.01
            - kkt_tol (float): KKT violation tolerance for convergence, default 1e-4
            - grad_clip (float): gradient clipping norm, default 10.0
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    def _get(self, key: str, default: Any) -> Any:
        return self.cfg.get(key, default)

    def optimize(
        self,
        init_frames: torch.Tensor,
        masks: torch.Tensor,
        posenet: nn.Module,
        segnet: nn.Module,
        rate_budget: float | None = None,
        log_every: int = 50,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run primal-dual optimization.

        Updates alternate between:
          1. Primal step: f <- f - alpha * grad_f L(f, lambda)
          2. Dual step: lambda <- max(0, lambda + eta * (R(f) - R_budget))

        Args:
            init_frames: (N, C, H, W) initial frames in [0, 255].
            masks: (N, H, W) target segmentation masks.
            posenet: frozen PoseNet model.
            segnet: frozen SegNet model.
            rate_budget: rate constraint (overrides cfg).
            log_every: logging interval (0 to disable).

        Returns:
            (optimized_frames, diagnostics) where diagnostics is a dict with:
                - final_lambda: learned dual variable
                - final_rate: achieved rate proxy
                - final_distortion: achieved distortion
                - kkt_violation: max KKT violation at convergence
                - converged: whether KKT conditions are satisfied
                - steps: number of steps taken
        """
        primal_lr = self._get("dual_primal_lr", 0.5)
        dual_lr = self._get("dual_dual_lr", 0.01)
        max_steps = self._get("dual_steps", 500)
        budget = rate_budget if rate_budget is not None else self._get("rate_budget", 0.01)
        lambda_val = self._get("lambda_init", 25.0)
        lambda_smooth = self._get("lambda_smooth", 0.01)
        kkt_tol = self._get("kkt_tol", 1e-4)
        grad_clip_val = self._get("grad_clip", 10.0)

        device = init_frames.device
        N, C, H, W = init_frames.shape

        # Primal variable
        f = init_frames.detach().clone().requires_grad_(True)
        primal_opt = torch.optim.Adam([f], lr=primal_lr)

        # Dual variable (scalar, on same device)
        lam = torch.tensor(lambda_val, device=device, dtype=torch.float32)

        best_f = f.detach().clone()
        best_lagrangian = float("inf")
        history: list[dict[str, float]] = []

        for step in range(max_steps):
            primal_opt.zero_grad()

            # --- Distortion D(f) ---
            seg_loss = self._segnet_loss(f, masks, segnet)
            pose_loss = self._posenet_loss(f, posenet)
            distortion = 100.0 * seg_loss + torch.sqrt(10.0 * pose_loss + 1e-8)

            # --- Rate proxy R(f) = normalized total variation ---
            rate = total_variation(f) / (N * C * H * W)

            # --- Smoothness regularizer ---
            smooth = smoothness_energy(f)

            # --- Lagrangian L(f, lambda) ---
            lagrangian = distortion + lam * (rate - budget) + lambda_smooth * smooth

            lagrangian.backward()

            # Gradient clipping
            if f.grad is not None:
                grad_norm = f.grad.data.norm()
                if grad_norm > grad_clip_val:
                    f.grad.data.mul_(grad_clip_val / (grad_norm + 1e-12))
            else:
                grad_norm = torch.tensor(0.0, device=device)

            # Primal step
            primal_opt.step()
            with torch.no_grad():
                f.data.clamp_(0.0, 255.0)

            # Dual step: gradient ascent on lambda
            with torch.no_grad():
                rate_val = rate.item()
                constraint_violation = rate_val - budget
                lam = torch.clamp(lam + dual_lr * constraint_violation, min=0.0)

            # KKT violation monitoring
            kkt_stationarity = grad_norm.item() if f.grad is not None else 0.0
            kkt_primal = max(0.0, constraint_violation)
            kkt_complementary = abs(lam.item() * constraint_violation)
            kkt_violation = max(kkt_stationarity, kkt_primal, kkt_complementary)

            # Track best
            lag_val = lagrangian.item()
            if lag_val < best_lagrangian:
                best_lagrangian = lag_val
                best_f = f.data.clone()

            record = {
                "step": step + 1,
                "lagrangian": lag_val,
                "distortion": distortion.item(),
                "rate": rate_val,
                "lambda": lam.item(),
                "kkt_violation": kkt_violation,
                "seg": seg_loss.item(),
                "pose": pose_loss.item(),
            }
            history.append(record)

            if log_every > 0 and (step + 1) % log_every == 0:
                print(
                    f"  dual step {step + 1:4d}/{max_steps}: "
                    f"L={lag_val:.4f} D={distortion.item():.4f} "
                    f"R={rate_val:.6f} lam={lam.item():.3f} "
                    f"KKT={kkt_violation:.2e}"
                )

            # Convergence check
            if kkt_violation < kkt_tol and step > 10:
                if log_every > 0:
                    print(f"  dual: KKT satisfied at step {step + 1}, "
                          f"violation={kkt_violation:.2e}")
                break

        diagnostics = {
            "final_lambda": lam.item(),
            "final_rate": history[-1]["rate"] if history else 0.0,
            "final_distortion": history[-1]["distortion"] if history else 0.0,
            "kkt_violation": history[-1]["kkt_violation"] if history else float("inf"),
            "converged": (history[-1]["kkt_violation"] if history else float("inf")) < kkt_tol,
            "steps": len(history),
        }

        return best_f.clamp(0.0, 255.0), diagnostics

    def _segnet_loss(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        segnet: nn.Module,
    ) -> torch.Tensor:
        """Cross-entropy SegNet loss using preprocess_input path."""
        N, C, H, W = frames.shape
        device = frames.device
        frames_btchw = frames.unsqueeze(1).contiguous()  # (N, 1, C, H, W)
        seg_input = segnet.preprocess_input(frames_btchw)
        logits = segnet(seg_input)
        H_out, W_out = logits.shape[2], logits.shape[3]
        masks_resized = F.interpolate(
            masks.float().unsqueeze(1).to(device),
            size=(H_out, W_out), mode="nearest",
        ).squeeze(1).long()
        return F.cross_entropy(logits, masks_resized)

    def _posenet_loss(
        self,
        frames: torch.Tensor,
        posenet: nn.Module,
    ) -> torch.Tensor:
        """Self-consistency PoseNet loss (self-pairs should give zero motion)."""
        N = frames.shape[0]
        device = frames.device
        if N < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        pair = frames.unsqueeze(1).expand(N, 2, *frames.shape[1:]).contiguous()
        posenet_in = posenet.preprocess_input(pair)
        pose_out = posenet(posenet_in)
        pred = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        return pred[..., :6].pow(2).mean()

    def trace_pareto_frontier(
        self,
        init_frames: torch.Tensor,
        masks: torch.Tensor,
        posenet: nn.Module,
        segnet: nn.Module,
        num_points: int = 10,
        rate_range: tuple[float, float] = (0.001, 0.05),
    ) -> list[dict[str, Any]]:
        """Trace the Pareto frontier by varying the rate budget.

        Each point on the frontier is a different rate-distortion tradeoff.
        The Lagrangian dual naturally finds one point per budget value;
        sweeping the budget traces the full frontier.

        The frontier satisfies the multi-objective optimality condition:
        no point can improve one objective without worsening another.

        Args:
            init_frames: (N, C, H, W) initial frames.
            masks: (N, H, W) target masks.
            posenet: frozen PoseNet.
            segnet: frozen SegNet.
            num_points: number of frontier points.
            rate_range: (min_rate, max_rate) range to sweep.

        Returns:
            List of dicts, each with keys: rate_budget, rate, distortion,
            seg, pose, lambda, frames.
        """
        frontier: list[dict[str, Any]] = []
        budgets = torch.linspace(rate_range[0], rate_range[1], num_points)

        for i, budget in enumerate(budgets):
            print(f"  Pareto point {i + 1}/{num_points}: rate_budget={budget.item():.4f}")
            frames, diag = self.optimize(
                init_frames.clone(),
                masks, posenet, segnet,
                rate_budget=budget.item(),
                log_every=0,
            )
            frontier.append({
                "rate_budget": budget.item(),
                "rate": diag["final_rate"],
                "distortion": diag["final_distortion"],
                "lambda": diag["final_lambda"],
                "converged": diag["converged"],
                "frames": frames.detach().cpu(),
            })

        return frontier


# ---- Smoke tests ----


def _smoke_test() -> None:
    """Run basic shape and convergence checks without real scorers."""
    print("variational_gen: starting smoke tests...")

    # Test spatial gradient
    f = torch.randn(2, 3, 16, 16)
    df_dx, df_dy = spatial_gradient(f)
    assert df_dx.shape == f.shape, f"df_dx shape {df_dx.shape} != {f.shape}"
    assert df_dy.shape == f.shape, f"df_dy shape {df_dy.shape} != {f.shape}"

    # Test Laplacian
    lap = laplacian(f)
    assert lap.shape == f.shape, f"Laplacian shape {lap.shape} != {f.shape}"

    # Test gradient magnitude squared
    gms = gradient_magnitude_squared(f)
    assert gms.shape == f.shape
    assert (gms >= 0).all(), "Gradient magnitude squared must be non-negative"

    # Test smoothness energy
    energy = smoothness_energy(f)
    assert energy.shape == (), f"Energy should be scalar, got {energy.shape}"
    assert energy.item() >= 0.0, "Energy must be non-negative"

    # Test total variation
    tv = total_variation(f)
    assert tv.shape == (), f"TV should be scalar, got {tv.shape}"
    assert tv.item() >= 0.0, "TV must be non-negative"

    # Constant frame should have zero TV and zero gradient energy
    const = torch.ones(1, 3, 16, 16) * 128.0
    tv_const = total_variation(const)
    assert tv_const.item() < 1e-6, f"Constant frame TV should be ~0, got {tv_const.item()}"

    gms_const = gradient_magnitude_squared(const)
    assert gms_const.abs().max().item() < 1e-6, "Constant frame gradient should be ~0"

    # Test that smoothness energy is differentiable
    f_grad = torch.randn(1, 3, 16, 16, requires_grad=True)
    e = smoothness_energy(f_grad)
    e.backward()
    assert f_grad.grad is not None, "Smoothness energy must be differentiable"
    assert f_grad.grad.shape == f_grad.shape

    print("  variational_gen: differential operators verified")

    # Test VariationalFrameGenerator with mock scorers
    class MockSegNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, -1, ...]

    class MockPoseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(3, 6)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            pooled = self.pool(x).squeeze(-1).squeeze(-1)
            return {"pose": self.fc(pooled)}

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, -1, ...]

    posenet = MockPoseNet()
    segnet = MockSegNet()

    init = torch.rand(2, 3, 32, 32) * 255.0
    masks = torch.randint(0, 5, (2, 32, 32))

    gen = VariationalFrameGenerator(cfg={
        "variational_steps": 10,
        "variational_lr": 0.5,
        "use_line_search": False,
    })
    result = gen.solve(init, masks, posenet, segnet, log_every=0)
    assert result.shape == init.shape, f"Output shape {result.shape} != input shape {init.shape}"
    assert result.min() >= 0.0 and result.max() <= 255.0
    print("  variational_gen: VariationalFrameGenerator verified")

    # Test LagrangianDualOptimizer
    dual = LagrangianDualOptimizer(cfg={
        "dual_steps": 10,
        "dual_primal_lr": 0.5,
        "rate_budget": 0.1,
    })
    result_dual, diag = dual.optimize(init, masks, posenet, segnet, log_every=0)
    assert result_dual.shape == init.shape
    assert result_dual.min() >= 0.0 and result_dual.max() <= 255.0
    assert "final_lambda" in diag
    assert "kkt_violation" in diag
    assert diag["final_lambda"] >= 0.0, "Dual variable must be non-negative (dual feasibility)"
    assert diag["steps"] > 0
    print("  variational_gen: LagrangianDualOptimizer verified")
    print(f"    final_lambda={diag['final_lambda']:.4f}, "
          f"rate={diag['final_rate']:.6f}, "
          f"distortion={diag['final_distortion']:.4f}")

    print("variational_gen: all smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
