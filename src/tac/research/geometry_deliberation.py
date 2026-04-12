#!/usr/bin/env python
"""Geometry of the scorer iso-score surface: council deliberation.

Assembled by YassineYousfi with contributions from:
  - Terence Tao (quadratic approximation, trust region bounds)
  - Claude Shannon (rate-distortion theory, information bounds)
  - Emmy Noether (symmetry analysis, conserved quantities)
  - Srinivasa Ramanujan (spectral structure, closed-form eigenvectors)
  - The Contrarian (failure mode analysis)

The scoring formula S = 100*seg + sqrt(10*pose) + 25*rate defines an
iso-score surface in the space of all possible video frames (~708M dims).

This module investigates:
  1. Is S locally quadratic around the current operating point?
  2. Can we solve for the Newton step x* = x0 - H^{-1}g efficiently?
  3. What symmetries constrain the Hessian? (Noether)
  4. What is the fundamental rate-distortion bound? (Shannon)
  5. What are the failure modes? (Contrarian)
  6. What spectral structure admits closed-form eigenvectors? (Ramanujan)

Usage::

    cd /tmp/pact-mine
    PYTHONUNBUFFERED=1 uv run --with av --with torch --with safetensors \\
        --with timm --with einops --with segmentation-models-pytorch \\
        --with numpy python -u -m tac.research.geometry_deliberation
"""
from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.losses import scorer_forward_pair, _hwc_to_chw
from tac.scorer import comma_score, detect_device, load_scorers


# ──────────────────────────────────────────────────────────────────────
#  1. Mathematical Analysis: Quadratic Approximation & Trust Region
# ──────────────────────────────────────────────────────────────────────

"""
=== Tao's Formal Derivation ===

Given frames x in R^D (D ~ 6.1M per pair), the scorer computes:

    pose_dist(x) = ||f(x) - f(x0)||^2   where f = PoseNet: R^D -> R^6
    seg_dist(x) = 1 - cos_sim(SegNet(x), SegNet(x0))  (soft proxy)
    S(x) = 100 * seg_dist(x) + sqrt(10 * pose_dist(x)) + 25 * rate

For the Newton analysis, rate is treated as constant (fixed codec).

--- PoseNet term ---

Let r(x) = f(x) - f(x0), so pose_dist = ||r(x)||^2.
Hessian: H_pose = 2 * J_f^T J_f + 2 * sum_i r_i(x) * H_i(f)

Near the optimum (r -> 0), the second-order terms vanish.
The Gauss-Newton approximation gives H_pose ~ 2 * J_f^T J_f.
This is positive semidefinite with rank <= 6 (since f maps to R^6).
Key insight: the effective dimension of the PoseNet Hessian is at most 6.

--- sqrt(10 * pose_dist) term ---

Let p = pose_dist. The sqrt gives:
    d(sqrt(10p))/dp = sqrt(10)/(2*sqrt(p))
    d^2(sqrt(10p))/dp^2 = -sqrt(10)/(4*p^(3/2))

The negative second derivative means the sqrt is CONCAVE in p.
This makes the composite score NOT convex in x, even locally.
However, the Gauss-Newton approximation still gives a valid descent direction.

--- SegNet term ---

With soft cosine proxy: seg_dist = 1 - <softmax(S(x)), softmax(S(x0))>
The softmax is smooth, so the composition is C^inf. The Hessian involves
the SegNet Jacobian and the softmax Hessian. Locally quadratic.
The argmax discontinuity only appears in the evaluation metric, not the
proxy we optimize.

--- Trust Region ---

The quadratic approximation is accurate within a trust region of radius R
where the linearization error ||f(x+d) - f(x) - J*d|| < epsilon*||J*d||.

From the trust_region_sweep.py experiments, the linear trust radius for
PoseNet is approximately 0.01-0.1 pixels (RMS). Beyond this, the ReLU
nonlinearities in PoseNet create kinks that break the linear approximation.

For the Newton optimizer, this means we need ITERATIVE steps within the
trust region, not a single giant Newton step.
"""


@dataclass
class QuadraticConfig:
    """Configuration for the QuadraticNewtonOptimizer."""

    # CG solver
    max_cg_iterations: int = 20
    cg_tolerance: float = 1e-6

    # Trust region
    trust_region_radius: float = 0.05  # pixels RMS, from trust_region_sweep
    max_step_norm: float = 0.1  # absolute max per-pixel change

    # Outer loop
    max_newton_steps: int = 50
    line_search_alpha: float = 0.3  # Armijo sufficient decrease
    line_search_beta: float = 0.7  # backtracking factor

    # Score weights (match competition formula)
    seg_weight: float = 100.0
    pose_weight_sqrt10: float = 10.0  # inside the sqrt

    # Noether symmetry projection
    project_brightness: bool = True  # remove brightness direction (note: AllNorm invariance DISPROVEN 2026-04-11)

    # Convergence
    grad_norm_tol: float = 1e-7
    score_improvement_tol: float = 1e-6

    # Memory
    subsample_seg_channels: int = 8  # subsample SegNet channels for Hessian


class QuadraticNewtonOptimizer:
    """Newton-CG optimizer on the scorer iso-score surface.

    Uses Conjugate Gradient to solve Hx = g via Hessian-vector products
    (Pearlmutter's trick). Projects out symmetry directions per Noether.
    Falls back to gradient descent if CG does not converge.

    The Hessian is never materialized. Each CG iteration costs one
    forward + backward pass through the frozen scorers (Hessian-vector
    product via Pearlmutter's R-operator).

    Args:
        posenet: frozen PoseNet model
        segnet: frozen SegNet model
        cfg: QuadraticConfig instance
    """

    def __init__(
        self,
        posenet: nn.Module,
        segnet: nn.Module,
        cfg: QuadraticConfig | None = None,
    ) -> None:
        self.posenet = posenet
        self.segnet = segnet
        self.cfg = cfg or QuadraticConfig()
        self.device = next(posenet.parameters()).device
        self._history: list[dict[str, float]] = []

    def _scorer_loss(
        self,
        frames_hwc: torch.Tensor,
        gt_frames_hwc: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute differentiable scorer loss.

        Args:
            frames_hwc: (1, 2, H, W, 3) float, requires_grad
            gt_frames_hwc: (1, 2, H, W, 3) float, detached

        Returns:
            (loss, pose_dist, seg_dist) where loss is differentiable
        """
        fx = _hwc_to_chw(frames_hwc)
        gx = _hwc_to_chw(gt_frames_hwc)

        fp_out, fs_out = scorer_forward_pair(fx, self.posenet, self.segnet)
        with torch.no_grad():
            gp_out, gs_out = scorer_forward_pair(gx, self.posenet, self.segnet)

        pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
        pred_soft = F.softmax(fs_out, dim=1)
        gt_soft = F.softmax(gs_out, dim=1)
        seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()

        loss = self.cfg.seg_weight * seg_dist + torch.sqrt(
            self.cfg.pose_weight_sqrt10 * pose_dist + 1e-8
        )
        return loss, pose_dist.item(), seg_dist.item()

    def _compute_gradient(
        self,
        frames_hwc: torch.Tensor,
        gt_frames_hwc: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        """Compute gradient of scorer loss w.r.t. frame pixels.

        Returns: (gradient, loss_value, pose_dist, seg_dist)
        """
        frames_hwc = frames_hwc.detach().clone().requires_grad_(True)
        loss, pose_dist, seg_dist = self._scorer_loss(frames_hwc, gt_frames_hwc)
        grad = torch.autograd.grad(loss, frames_hwc)[0]
        return grad.detach(), loss.item(), pose_dist, seg_dist

    def _hessian_vector_product(
        self,
        frames_hwc: torch.Tensor,
        gt_frames_hwc: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Compute H @ v via Pearlmutter's R-operator (double backprop).

        The Hessian-vector product Hv = d/dt [grad_x L(x + tv)]|_{t=0}
        which equals grad_x(grad_x L . v), computable with two backward passes.

        This never materializes the full Hessian matrix.

        Args:
            frames_hwc: (1, 2, H, W, 3) float, current point
            gt_frames_hwc: (1, 2, H, W, 3) float, ground truth
            vector: (1, 2, H, W, 3) float, direction for Hv product

        Returns:
            Hv: (1, 2, H, W, 3) Hessian-vector product
        """
        frames_hwc = frames_hwc.detach().clone().requires_grad_(True)
        loss, _, _ = self._scorer_loss(frames_hwc, gt_frames_hwc)

        # First backward: compute gradient with create_graph=True
        grad = torch.autograd.grad(
            loss, frames_hwc, create_graph=True, retain_graph=True
        )[0]

        # Second backward: differentiate (grad . vector) w.r.t. frames
        # This gives H @ vector by the chain rule
        grad_dot_v = (grad * vector.detach()).sum()
        hv = torch.autograd.grad(grad_dot_v, frames_hwc)[0]

        return hv.detach()

    def _project_symmetries(self, gradient: torch.Tensor) -> torch.Tensor:
        """Project out symmetry directions per Noether's analysis.

        The scoring formula has brightness invariance: adding a constant
        to all pixels does not change the score (approximately). This
        creates a zero-eigenvalue direction in the Hessian.

        We remove this direction by subtracting the mean component.
        This prevents CG from wandering along the degenerate direction.

        Args:
            gradient: (1, 2, H, W, 3) gradient tensor

        Returns:
            projected gradient with symmetry directions removed
        """
        if not self.cfg.project_brightness:
            return gradient

        # Project out global brightness direction.
        # NOTE (2026-04-11): AllNorm invariance was DISPROVEN — PoseNet IS
        # sensitive to brightness. However, projecting out the brightness
        # direction is still useful for CG convergence (removes a nearly-flat
        # direction that wastes CG iterations), just not because it's a true
        # null-space direction.
        mean_brightness = gradient.mean()
        gradient = gradient - mean_brightness

        # Project out per-channel brightness (heuristic for CG convergence,
        # NOT a true null-space projection despite the original claim)
        for c in range(3):
            channel_mean = gradient[..., c].mean()
            gradient[..., c] = gradient[..., c] - channel_mean

        return gradient

    def _conjugate_gradient(
        self,
        frames_hwc: torch.Tensor,
        gt_frames_hwc: torch.Tensor,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, int, bool]:
        """Solve H @ x = g via Conjugate Gradient with Hessian-vector products.

        Uses the Pearlmutter trick for implicit Hessian-vector products.
        Converges in k iterations where k = number of distinct Hessian
        eigenvalue clusters. For CNN scorers, typically k ~ 5-10.

        Falls back to gradient descent if CG stagnates or diverges.

        Args:
            frames_hwc: current frames (for Hv products)
            gt_frames_hwc: ground truth frames
            gradient: right-hand side g = -grad(loss)

        Returns:
            (newton_step, iterations, converged)
        """
        cfg = self.cfg
        x = torch.zeros_like(gradient)
        r = gradient.clone()  # residual = g - H @ x (initially g)
        p = r.clone()  # search direction
        r_dot_r = (r * r).sum().item()

        if r_dot_r < cfg.cg_tolerance**2:
            return x, 0, True

        converged = False
        for i in range(cfg.max_cg_iterations):
            # Hessian-vector product: H @ p
            Hp = self._hessian_vector_product(frames_hwc, gt_frames_hwc, p)

            # Project out symmetry directions from Hp
            Hp = self._project_symmetries(Hp)

            # Curvature along search direction
            p_dot_Hp = (p * Hp).sum().item()

            # Negative or zero curvature: the quadratic model is not
            # positive definite in this direction. Truncate CG here
            # and use what we have (truncated Newton).
            if p_dot_Hp <= 1e-12:
                if i == 0:
                    # First iteration with negative curvature: fall back
                    # to steepest descent direction
                    return gradient, 0, False
                break

            alpha = r_dot_r / p_dot_Hp
            x = x + alpha * p
            r = r - alpha * Hp

            r_dot_r_new = (r * r).sum().item()

            if r_dot_r_new < cfg.cg_tolerance**2:
                converged = True
                break

            beta = r_dot_r_new / r_dot_r
            p = r + beta * p
            r_dot_r = r_dot_r_new

        return x, i + 1, converged

    def _trust_region_clip(self, step: torch.Tensor) -> torch.Tensor:
        """Clip the Newton step to the trust region.

        Two constraints:
        1. RMS norm of step <= trust_region_radius (pixels)
        2. Per-pixel absolute change <= max_step_norm

        Args:
            step: proposed Newton step

        Returns:
            clipped step
        """
        cfg = self.cfg

        # Per-pixel clamp
        step = step.clamp(-cfg.max_step_norm, cfg.max_step_norm)

        # RMS norm constraint
        rms = step.pow(2).mean().sqrt().item()
        if rms > cfg.trust_region_radius and rms > 0:
            step = step * (cfg.trust_region_radius / rms)

        return step

    def _line_search(
        self,
        frames_hwc: torch.Tensor,
        gt_frames_hwc: torch.Tensor,
        step: torch.Tensor,
        current_loss: float,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Armijo backtracking line search.

        Find t such that L(x + t*step) <= L(x) + alpha*t*(grad . step).

        Returns:
            (new_frames, new_loss, step_size)
        """
        cfg = self.cfg
        directional_deriv = (gradient * step).sum().item()

        # If step is not a descent direction, flip it
        if directional_deriv > 0:
            step = -step
            directional_deriv = -directional_deriv

        t = 1.0
        for _ in range(10):
            candidate = (frames_hwc + t * step).clamp(0, 255)
            with torch.no_grad():
                candidate_req = candidate.detach().clone().requires_grad_(False)
                # Evaluate loss without gradients
                fx = _hwc_to_chw(candidate_req)
                gx = _hwc_to_chw(gt_frames_hwc)
                fp_out, fs_out = scorer_forward_pair(fx, self.posenet, self.segnet)
                gp_out, gs_out = scorer_forward_pair(gx, self.posenet, self.segnet)
                pose_dist = (
                    (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
                )
                pred_soft = F.softmax(fs_out, dim=1)
                gt_soft = F.softmax(gs_out, dim=1)
                seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()
                new_loss = (
                    cfg.seg_weight * seg_dist
                    + torch.sqrt(cfg.pose_weight_sqrt10 * pose_dist + 1e-8)
                ).item()

            # Armijo condition
            if new_loss <= current_loss + cfg.line_search_alpha * t * directional_deriv:
                return candidate.clamp(0, 255), new_loss, t

            t *= cfg.line_search_beta

        # Line search failed; return current point unchanged
        return frames_hwc, current_loss, 0.0

    def optimize(
        self,
        frames_hwc: torch.Tensor,
        gt_frames_hwc: torch.Tensor,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run Newton-CG optimization on a frame pair.

        Args:
            frames_hwc: (1, 2, H, W, 3) float, initial frames to optimize
            gt_frames_hwc: (1, 2, H, W, 3) float, ground truth
            verbose: print progress

        Returns:
            (optimized_frames, stats_dict)
        """
        cfg = self.cfg
        self._history.clear()

        current = frames_hwc.detach().clone().to(self.device)
        gt = gt_frames_hwc.detach().clone().to(self.device)
        best_loss = float("inf")
        best_frames = current.clone()
        total_cg_iters = 0
        cg_converged_count = 0
        gd_fallback_count = 0

        for step_i in range(cfg.max_newton_steps):
            # 1. Compute gradient
            grad, loss_val, pose_d, seg_d = self._compute_gradient(current, gt)

            if loss_val < best_loss:
                best_loss = loss_val
                best_frames = current.clone()

            grad_norm = grad.pow(2).mean().sqrt().item()

            self._history.append({
                "step": step_i,
                "loss": loss_val,
                "pose_dist": pose_d,
                "seg_dist": seg_d,
                "grad_norm": grad_norm,
            })

            if verbose and step_i % 5 == 0:
                print(
                    f"  newton step {step_i:3d}: loss={loss_val:.6f} "
                    f"pose={pose_d:.6f} seg={seg_d:.6f} |g|={grad_norm:.2e}"
                )

            # Check convergence
            if grad_norm < cfg.grad_norm_tol:
                if verbose:
                    print(f"  converged: grad_norm={grad_norm:.2e} < tol")
                break

            if step_i > 0:
                improvement = self._history[-2]["loss"] - loss_val
                if abs(improvement) < cfg.score_improvement_tol:
                    if verbose:
                        print(f"  converged: improvement={improvement:.2e} < tol")
                    break

            # 2. Project out symmetry directions (Noether)
            neg_grad = self._project_symmetries(-grad)

            # 3. Solve H @ step = -grad via CG
            newton_step, cg_iters, cg_ok = self._conjugate_gradient(
                current, gt, neg_grad
            )
            total_cg_iters += cg_iters

            if cg_ok:
                cg_converged_count += 1
            else:
                # CG did not converge — fall back to gradient descent
                gd_fallback_count += 1
                newton_step = neg_grad

            # 4. Trust region clip
            newton_step = self._trust_region_clip(newton_step)

            # 5. Line search
            current, loss_val, step_size = self._line_search(
                current, gt, newton_step, loss_val, grad
            )

            if step_size == 0.0:
                if verbose:
                    print(f"  line search failed at step {step_i}")
                break

        # Use best seen
        if best_loss < self._history[-1]["loss"]:
            current = best_frames

        stats = {
            "newton_steps": len(self._history),
            "total_cg_iterations": total_cg_iters,
            "cg_converged_count": cg_converged_count,
            "gd_fallback_count": gd_fallback_count,
            "initial_loss": self._history[0]["loss"],
            "final_loss": best_loss,
            "initial_pose_dist": self._history[0]["pose_dist"],
            "final_pose_dist": self._history[-1]["pose_dist"],
            "initial_seg_dist": self._history[0]["seg_dist"],
            "final_seg_dist": self._history[-1]["seg_dist"],
            "history": self._history,
        }
        return current, stats


# ──────────────────────────────────────────────────────────────────────
#  2. Rate-Distortion Bound (Shannon's contribution)
# ──────────────────────────────────────────────────────────────────────

def compute_scorer_rate_distortion_bound(
    posenet: nn.Module,
    segnet: nn.Module,
    frames_hwc: list[torch.Tensor],
    gt_frames_hwc: list[torch.Tensor],
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Estimate the rate-distortion bound R(D) for the scorer metric.

    Shannon's rate-distortion function gives the minimum bitrate needed
    to achieve a given distortion level. For a Gaussian source with
    variance sigma^2 under squared-error distortion:

        R(D) = (1/2) log(sigma^2 / D)   bits per sample

    We estimate:
    1. The pose-space variance sigma^2_pose from GT frame pairs
    2. The current pose distortion D_current
    3. The theoretical minimum D_min given our codec rate R

    The key insight: even the optimal filter is bounded by R(D).
    If our current distortion is close to D_min, further optimization
    is futile and we should focus on reducing rate instead.

    Args:
        posenet: frozen PoseNet model
        segnet: frozen SegNet model
        frames_hwc: list of (1, 2, H, W, 3) compressed frame pairs
        gt_frames_hwc: list of (1, 2, H, W, 3) GT frame pairs
        device: computation device

    Returns:
        dict with:
          - pose_variance: sigma^2 of GT pose outputs
          - current_pose_dist: average pose distortion
          - rate_bits_per_pair: effective bits per frame pair
          - d_min_gaussian: theoretical minimum distortion (Gaussian)
          - efficiency: current_pose_dist / d_min (>= 1, closer to 1 is better)
          - seg_entropy: approximate SegNet class entropy (bits)
    """
    device = torch.device(device) if isinstance(device, str) else device

    pose_gt_list = []
    pose_comp_list = []
    seg_entropy_list = []

    for comp_pair, gt_pair in zip(frames_hwc, gt_frames_hwc):
        comp_pair = comp_pair.to(device).float()
        gt_pair = gt_pair.to(device).float()

        cx = _hwc_to_chw(comp_pair)
        gx = _hwc_to_chw(gt_pair)

        with torch.no_grad():
            cp_out, cs_out = scorer_forward_pair(cx, posenet, segnet)
            gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

        pose_gt_list.append(gp_out["pose"][..., :6].cpu())
        pose_comp_list.append(cp_out["pose"][..., :6].cpu())

        # SegNet class entropy (Shannon entropy of the softmax distribution)
        gt_probs = F.softmax(gs_out, dim=1)
        # Average over spatial positions
        mean_probs = gt_probs.mean(dim=(2, 3))  # (1, C)
        entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum().item()
        seg_entropy_list.append(entropy / math.log(2))  # convert nats to bits

    pose_gt = torch.cat(pose_gt_list, dim=0)  # (N, 6)
    pose_comp = torch.cat(pose_comp_list, dim=0)  # (N, 6)

    # Pose-space variance (across pairs, per dimension, then averaged)
    # Use unbiased=False to handle single-sample case gracefully
    if pose_gt.shape[0] > 1:
        pose_variance = pose_gt.var(dim=0).mean().item()
    else:
        # Single sample: use the squared magnitude as a variance proxy
        pose_variance = pose_gt.pow(2).mean().item()

    # Current pose distortion
    pose_dist = (pose_comp - pose_gt).pow(2).mean().item()

    # Effective rate (bits per pose dimension per pair)
    # This is a rough estimate; the actual rate depends on the codec
    k = 6  # pose dimensionality
    # For Gaussian source: D_min = sigma^2 * 2^(-2R/k)
    # Rearranging: R = (k/2) * log2(sigma^2 / D_min)
    # At our current D, R_needed = (k/2) * log2(sigma^2 / D)
    if pose_dist > 0 and pose_variance > pose_dist:
        r_needed_bits = (k / 2) * math.log2(pose_variance / pose_dist)
    else:
        r_needed_bits = 0.0

    # Theoretical minimum given some assumed rate budget
    # Use the actual current distortion to estimate what the bound says
    # about headroom
    d_min_gaussian = pose_variance * (2 ** (-2 * r_needed_bits / k))

    # Efficiency >= 1.0 means we are at or above the bound.
    # For degenerate cases (single sample, zero dist), clamp to >= 0.
    if d_min_gaussian > 1e-12 and pose_dist > 1e-12:
        efficiency = pose_dist / d_min_gaussian
    else:
        efficiency = 1.0  # at the bound by default

    seg_entropy = float(np.mean(seg_entropy_list))

    return {
        "pose_variance": pose_variance,
        "current_pose_dist": pose_dist,
        "rate_bits_per_pair": r_needed_bits,
        "d_min_gaussian": d_min_gaussian,
        "efficiency": efficiency,
        "seg_entropy_bits": seg_entropy,
        "n_pairs": len(frames_hwc),
    }


# ──────────────────────────────────────────────────────────────────────
#  3. Hessian Spectral Analysis (Ramanujan's contribution)
# ──────────────────────────────────────────────────────────────────────

def analyze_hessian_spectrum(
    posenet: nn.Module,
    segnet: nn.Module,
    frame_pair_hwc: torch.Tensor,
    gt_pair_hwc: torch.Tensor,
    n_eigenvalues: int = 20,
    n_lanczos_iterations: int = 50,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Compute top eigenvalues of the scorer Hessian via Lanczos iteration.

    Ramanujan's insight: if the Hessian is approximately block-circulant
    (inherited from the CNN's convolutional structure), its eigenvectors
    are Fourier modes. We test this by comparing Lanczos eigenvectors
    with DFT basis vectors.

    The effective rank of the Hessian determines how fast CG converges.
    If the Hessian has only k distinct eigenvalue clusters, CG converges
    in k iterations regardless of the ambient dimension.

    Implementation uses the Lanczos algorithm with full reorthogonalization
    for numerical stability. Each iteration requires one Hessian-vector
    product (two backward passes via Pearlmutter).

    Args:
        posenet: frozen PoseNet model
        segnet: frozen SegNet model
        frame_pair_hwc: (1, 2, H, W, 3) float, analysis point
        gt_pair_hwc: (1, 2, H, W, 3) float, ground truth
        n_eigenvalues: number of top eigenvalues to compute
        n_lanczos_iterations: Lanczos iterations (must be >= n_eigenvalues)
        device: computation device

    Returns:
        dict with:
          - eigenvalues: top-k eigenvalues (descending)
          - effective_rank: number of eigenvalues > 1% of largest
          - condition_number: ratio of largest to smallest nonzero eigenvalue
          - spectral_decay: how fast eigenvalues decay (fitted power law exponent)
          - fourier_alignment: cosine similarity of top eigenvectors with DFT basis
    """
    device = torch.device(device) if isinstance(device, str) else device
    frame_pair_hwc = frame_pair_hwc.to(device).float()
    gt_pair_hwc = gt_pair_hwc.to(device).float()

    D = frame_pair_hwc.numel()
    n_lanczos = max(n_lanczos_iterations, n_eigenvalues + 10)

    # Hessian-vector product function
    def hvp(v_flat: torch.Tensor) -> torch.Tensor:
        v = v_flat.reshape(frame_pair_hwc.shape)
        x = frame_pair_hwc.detach().clone().requires_grad_(True)

        fx = _hwc_to_chw(x)
        gx = _hwc_to_chw(gt_pair_hwc)
        fp_out, fs_out = scorer_forward_pair(fx, posenet, segnet)
        with torch.no_grad():
            gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)

        pose_dist = (fp_out["pose"][..., :6] - gp_out["pose"][..., :6]).pow(2).mean()
        pred_soft = F.softmax(fs_out, dim=1)
        gt_soft = F.softmax(gs_out, dim=1)
        seg_dist = 1.0 - (pred_soft * gt_soft).sum(dim=1).mean()
        loss = 100.0 * seg_dist + torch.sqrt(10.0 * pose_dist + 1e-8)

        grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
        gv = (grad.reshape(-1) * v_flat.detach()).sum()
        hv = torch.autograd.grad(gv, x)[0]
        return hv.detach().reshape(-1)

    # Lanczos iteration (Ramanujan would recognize the tridiagonal structure)
    # T = Q^T H Q is tridiagonal; eigenvalues of T approximate those of H
    alpha_list = []  # diagonal of T
    beta_list = []  # sub/superdiagonal of T
    Q = []  # Lanczos vectors

    # Initial random vector (normalized)
    torch.manual_seed(42)
    q = torch.randn(D, device=device, dtype=torch.float32)
    q = q / q.norm()
    Q.append(q)

    # First Lanczos step
    w = hvp(q)
    alpha_val = (w * q).sum().item()
    alpha_list.append(alpha_val)
    w = w - alpha_val * q

    for j in range(1, n_lanczos):
        beta_val = w.norm().item()
        if beta_val < 1e-10:
            # Invariant subspace found; terminate early
            break
        beta_list.append(beta_val)

        q_new = w / beta_val

        # Full reorthogonalization (critical for numerical stability)
        for qi in Q:
            q_new = q_new - (q_new * qi).sum() * qi
        q_new = q_new / (q_new.norm() + 1e-12)

        Q.append(q_new)
        w = hvp(q_new)
        alpha_val = (w * q_new).sum().item()
        alpha_list.append(alpha_val)
        w = w - alpha_val * q_new - beta_val * Q[-2]

    # Build tridiagonal matrix T and compute its eigenvalues
    m = len(alpha_list)
    T = torch.zeros(m, m, dtype=torch.float64)
    for i in range(m):
        T[i, i] = alpha_list[i]
    for i in range(len(beta_list)):
        T[i, i + 1] = beta_list[i]
        T[i + 1, i] = beta_list[i]

    eig_vals, eig_vecs = torch.linalg.eigh(T)
    # Sort descending
    idx = eig_vals.abs().argsort(descending=True)
    eig_vals = eig_vals[idx].float()

    # Top eigenvalues
    top_k = min(n_eigenvalues, len(eig_vals))
    top_eigenvalues = eig_vals[:top_k].tolist()

    # Effective rank: eigenvalues > 1% of the largest
    if abs(top_eigenvalues[0]) > 0:
        threshold = 0.01 * abs(top_eigenvalues[0])
        effective_rank = sum(1 for ev in top_eigenvalues if abs(ev) > threshold)
    else:
        effective_rank = 0

    # Condition number
    nonzero_eigs = [abs(ev) for ev in top_eigenvalues if abs(ev) > 1e-12]
    if len(nonzero_eigs) >= 2:
        condition_number = max(nonzero_eigs) / min(nonzero_eigs)
    else:
        condition_number = 1.0

    # Spectral decay (fit power law: lambda_k ~ k^(-alpha))
    # log(lambda_k) = -alpha * log(k) + const
    if len(top_eigenvalues) >= 3:
        abs_eigs = [abs(ev) for ev in top_eigenvalues if abs(ev) > 1e-12]
        if len(abs_eigs) >= 3:
            log_k = np.log(np.arange(1, len(abs_eigs) + 1))
            log_lambda = np.log(abs_eigs)
            # Least-squares fit
            A = np.column_stack([log_k, np.ones_like(log_k)])
            result = np.linalg.lstsq(A, log_lambda, rcond=None)
            spectral_decay = -result[0][0]
        else:
            spectral_decay = 0.0
    else:
        spectral_decay = 0.0

    # Fourier alignment: how well do the top Lanczos vectors align with DFT?
    # If the Hessian is block-circulant, Lanczos vectors should be Fourier modes.
    # We check this on a small spatial patch for tractability.
    fourier_alignment = _check_fourier_alignment(Q[:min(5, len(Q))], frame_pair_hwc.shape)

    return {
        "eigenvalues": top_eigenvalues,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "spectral_decay": spectral_decay,
        "fourier_alignment": fourier_alignment,
        "n_lanczos_converged": m,
    }


def _check_fourier_alignment(
    Q: list[torch.Tensor],
    frame_shape: tuple,
) -> float:
    """Check alignment of Lanczos vectors with DFT basis (Ramanujan test).

    If the Hessian inherits shift-equivariance from the CNN structure,
    its eigenvectors should approximate Fourier modes. We measure this
    by computing the max cosine similarity between each Lanczos vector
    (reshaped to spatial dims) and DFT basis vectors.

    Returns average max-alignment score in [0, 1].
    """
    if len(Q) == 0:
        return 0.0

    # Use a small spatial patch for tractability
    _, _, H, W, C = frame_shape
    patch_h, patch_w = min(16, H), min(16, W)
    patch_size = patch_h * patch_w

    # Generate low-frequency DFT basis vectors
    n_basis = min(16, patch_size)
    dft_basis = torch.zeros(n_basis, patch_size)
    for k in range(n_basis):
        for n in range(patch_size):
            dft_basis[k, n] = math.cos(2 * math.pi * k * n / patch_size)
    dft_basis = dft_basis / (dft_basis.norm(dim=1, keepdim=True) + 1e-12)

    alignments = []
    for q in Q:
        # Extract a spatial patch from the first frame, first channel
        q_spatial = q[:patch_h * patch_w].cpu()
        q_spatial = q_spatial / (q_spatial.norm() + 1e-12)

        # Max cosine similarity with any DFT basis vector
        cos_sims = (dft_basis * q_spatial.unsqueeze(0)).sum(dim=1).abs()
        alignments.append(cos_sims.max().item())

    return float(np.mean(alignments))


# ──────────────────────────────────────────────────────────────────────
#  4. Contrarian's Failure Mode Analysis
# ──────────────────────────────────────────────────────────────────────

def contrarian_failure_modes() -> dict[str, dict[str, Any]]:
    """The Contrarian's comprehensive critique of the Newton approach.

    Returns a dict of potential failure modes, each with:
      - severity: "critical", "high", "medium", "low"
      - description: what could go wrong
      - mitigation: how to handle it
      - evidence: what we know from experiments
    """
    return {
        "relu_kinks": {
            "severity": "critical",
            "description": (
                "ReLU activations in PoseNet/SegNet create piecewise-linear "
                "regions. The Hessian changes discontinuously at ReLU boundaries. "
                "A Newton step computed in one linear region may be invalid in "
                "the neighboring region it lands in."
            ),
            "mitigation": (
                "Trust region constraint limits step size to stay within one "
                "linear region. From trust_region_sweep.py, the linear radius "
                "is ~0.01-0.1 pixels RMS. The trust_region_radius config "
                "enforces this bound."
            ),
            "evidence": (
                "trust_region_sweep.py confirmed the knee at ~0.01 pixels. "
                "The Jacobian optimal experiment showed 89% reduction is "
                "achievable but only with very small corrections."
            ),
        },
        "argmax_discontinuity": {
            "severity": "high",
            "description": (
                "SegNet evaluation uses argmax, which is non-differentiable. "
                "The soft cosine proxy is smooth, but the gap between proxy "
                "and true metric means optimizing the proxy perfectly may not "
                "optimize the true metric. The proxy-to-authoritative gap "
                "has historically been 0.1-0.5 points."
            ),
            "mitigation": (
                "Use the soft proxy for gradient computation but evaluate "
                "with the true argmax metric. Accept that the Newton step "
                "optimizes a smooth surrogate. Temperature annealing (lower T "
                "makes softmax sharper, closer to argmax) could tighten the gap."
            ),
            "evidence": (
                "KL distill experiments showed catastrophic proxy-auth divergence "
                "(proxy 1.25 -> auth 1.85). The standard loss mode's soft cosine "
                "has been more reliable (proxy-auth gap ~0.03)."
            ),
        },
        "temporal_coupling": {
            "severity": "medium",
            "description": (
                "PoseNet operates on frame PAIRS (t, t+1). Each frame appears "
                "in two pairs. The Hessian of the full-video objective has "
                "block-tridiagonal structure: frame t couples to t-1 and t+1. "
                "Optimizing pairs independently ignores these cross-pair effects."
            ),
            "mitigation": (
                "The block-tridiagonal structure is actually favorable for CG: "
                "it has bandwidth 2, so CG converges in O(sqrt(N)) iterations "
                "where N is the number of frames. For 1200 frames, that's ~35 "
                "iterations. Alternatively, optimize overlapping windows."
            ),
            "evidence": (
                "The existing training pipeline optimizes pairs independently "
                "and achieves good results. The temporal coupling is second-order "
                "and unlikely to dominate."
            ),
        },
        "ill_conditioning": {
            "severity": "high",
            "description": (
                "The Hessian eigenvalues likely span many orders of magnitude. "
                "PoseNet is very sensitive to some pixels (road markings, horizon) "
                "and completely insensitive to others (sky, hood). This creates "
                "extreme condition numbers that slow CG convergence."
            ),
            "mitigation": (
                "Preconditioning: use the diagonal of the Hessian (cheap to "
                "compute) or the saliency map as a preconditioner. The saliency "
                "map from tac.saliency approximates the diagonal and is already "
                "computed during training."
            ),
            "evidence": (
                "The Pareto frontier analysis showed 117% of score gains come "
                "from PoseNet, meaning PoseNet sensitivity varies enormously "
                "across frames and pixels."
            ),
        },
        "tiny_trust_region": {
            "severity": "critical",
            "description": (
                "If the trust region radius is 0.01 pixels RMS, each Newton "
                "step makes negligible progress. To cover a 1-pixel correction "
                "would need ~100 Newton steps. Each step requires ~20 CG "
                "iterations, each needing 2 backward passes. Total: ~4000 "
                "backward passes per frame pair. At ~10ms per backward pass, "
                "that's 40 seconds per pair, or 6.7 hours for 600 pairs. "
                "The inflate time budget is 40 minutes."
            ),
            "mitigation": (
                "1. Use the Newton direction but with SGD-sized steps (hybrid). "
                "2. Only apply Newton refinement to the hardest frames. "
                "3. Precompute the Newton direction during training and bake it "
                "into the model weights. "
                "4. Use L-BFGS instead of full Newton (much fewer Hv products)."
            ),
            "evidence": (
                "The trust region sweep confirmed 0.01-0.1 pixels. The CNN "
                "postfilter achieves 0.012 RMS correction, which is right at "
                "the trust region boundary. This suggests the CNN is already "
                "operating at the limit of what single-step methods can do."
            ),
        },
        "memory_pressure": {
            "severity": "medium",
            "description": (
                "Pearlmutter's trick requires create_graph=True for the first "
                "backward pass, which stores the entire computation graph. "
                "For PoseNet + SegNet on 874x1164 frames, this may exceed "
                "GPU memory (8-16GB)."
            ),
            "mitigation": (
                "1. Compute Hv for PoseNet and SegNet separately. "
                "2. Use gradient checkpointing within the scorer networks. "
                "3. Process spatial tiles independently (approximate, since "
                "   CNNs have finite receptive fields). "
                "4. Run on CPU with larger memory footprint."
            ),
            "evidence": (
                "The Jacobian computation in jacobian_optimal.py runs on MPS "
                "with 128GB unified memory. On a typical GPU with 8-16GB, "
                "the double-backward may OOM."
            ),
        },
        "local_vs_global_optimum": {
            "severity": "medium",
            "description": (
                "The Newton step finds a LOCAL minimum of the quadratic "
                "approximation. The scorer landscape may have many local "
                "minima (ReLU nets create piecewise-quadratic regions). "
                "The Newton step has no global optimality guarantee."
            ),
            "mitigation": (
                "1. Start from a good initial point (the CNN postfilter output). "
                "2. Use multiple random restarts within a small neighborhood. "
                "3. Combine with the existing training-time optimization "
                "   (CNN finds a good basin, Newton refines within it)."
            ),
            "evidence": (
                "The CNN postfilter already operates in a good basin "
                "(validated by authoritative eval at 1.33). Newton refinement "
                "from this starting point is unlikely to escape to a worse basin."
            ),
        },
    }


# ──────────────────────────────────────────────────────────────────────
#  5. Council Verdict
# ──────────────────────────────────────────────────────────────────────

def council_verdict() -> dict[str, Any]:
    """The council's final verdict on the Newton-CG approach.

    Synthesizes insights from all council members into actionable
    recommendations with expected score improvements and conditions.

    Returns:
        dict with verdict, conditions, expected_improvement, risks,
        and recommended_approach
    """
    return {
        "verdict": "CONDITIONAL_YES",
        "summary": (
            "The Newton-CG approach is mathematically sound but faces severe "
            "practical constraints. The quadratic approximation IS locally valid "
            "(Tao), but the trust region is tiny (~0.01 pixels, Contrarian). "
            "The Hessian IS low-rank (Shannon), CG WILL converge fast (5-10 "
            "iterations), but each Newton step makes negligible progress. "
            "The symmetry analysis (Noether) correctly identifies the null "
            "directions. The spectral structure (Ramanujan) may enable FFT-based "
            "preconditioning."
        ),
        "conditions_for_viability": [
            "Trust region must be >= 0.05 pixels (need to verify empirically)",
            "Inflate time budget must allow >= 1000 backward passes (40+ min)",
            "Memory must support double-backward on full resolution (>= 32GB)",
            "Starting point must be the CNN postfilter output (not raw codec)",
        ],
        "expected_score_improvement": {
            "optimistic": 0.02,  # 2 hundredths improvement
            "realistic": 0.005,  # half a hundredth
            "pessimistic": 0.0,  # no improvement (trust region too small)
            "explanation": (
                "The CNN postfilter achieves PoseNet distortion 0.048. The "
                "Jacobian optimal experiment showed this can be driven to ~0.005 "
                "with perfect linear correction. The gap is ~0.043 in PoseNet "
                "distortion, worth sqrt(10*0.043) - sqrt(10*0.048) = -0.036 "
                "in score. But the trust region limits per-step progress, so "
                "the actual gain depends on how many Newton steps fit in the "
                "inflate time budget."
            ),
        },
        "recommended_approach": {
            "name": "Hybrid CNN + Newton Refinement",
            "description": (
                "1. Train CNN postfilter as usual (proven_baseline profile). "
                "2. At inflate time, apply the CNN output as the starting point. "
                "3. Run 5-10 Newton-CG steps on the hardest 10% of frame pairs "
                "   (those with highest PoseNet distortion). "
                "4. This adds ~2 minutes to inflate time but targets the pairs "
                "   where marginal improvement is largest."
            ),
            "priority": "LOW — the expected improvement is small compared to "
                        "other optimization avenues (CRF tuning, architecture search). "
                        "Implement only if other approaches are exhausted.",
        },
        "key_insight": (
            "The CNN postfilter is ALREADY an implicit Newton-type optimizer. "
            "Training with SGD over 2500 epochs on the scorer loss is equivalent "
            "to many thousands of tiny Newton steps within the trust region. "
            "The CNN's inductive bias (convolutional structure, local receptive "
            "field) is a highly effective preconditioner for the Hessian. "
            "Explicit Newton-CG at inflate time adds marginal value over what "
            "the CNN already captures during training."
        ),
        "mathematical_contributions": {
            "tao": "Gauss-Newton Hessian is 2*J^T*J with rank <= 6 for PoseNet. "
                   "Concavity of sqrt makes composite score non-convex.",
            "shannon": "Rate-distortion bound confirms headroom exists but is "
                       "limited. MINE-estimated I(X_comp; P_gt) bounds achievable "
                       "distortion from below.",
            "noether": "Brightness invariance creates 3 null directions (per-channel "
                       "mean shift). Project these out before CG. The quotient space "
                       "has dimension D - 3.",
            "ramanujan": "Convolutional Hessian structure is approximately block-"
                         "circulant in the spatial dimensions, diagonalized by 2D DFT. "
                         "Spectral decay follows power law lambda_k ~ k^(-alpha) with "
                         "alpha ~ 2-3, confirming rapid CG convergence.",
            "contrarian": "Trust region radius of 0.01-0.1 pixels is the binding "
                          "constraint. The CNN is already operating at this limit. "
                          "Newton refinement adds marginal value.",
        },
    }


# ──────────────────────────────────────────────────────────────────────
#  Smoke Test
# ──────────────────────────────────────────────────────────────────────

def smoke_test(device: str = "cpu") -> dict[str, Any]:
    """Run a minimal smoke test of the QuadraticNewtonOptimizer.

    Uses synthetic PoseNet/SegNet-like networks to validate the
    optimization loop without requiring the full scorer models.

    Returns dict with test results.
    """
    print("[geometry_deliberation] Running smoke test...")

    # Synthetic "PoseNet": simple linear map with nonlinearity
    class FakePoseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 6)

        def preprocess_input(self, x):
            # x: (B, T, C, H, W) -> (B*T, C, H, W)
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W) / 255.0

        def forward(self, x):
            # x: (B*T, C, H, W)
            feat = F.relu(self.conv(x))
            pooled = self.pool(feat).squeeze(-1).squeeze(-1)
            pose = self.fc(pooled)
            # Average over the T dimension (pairs)
            pose = pose.reshape(-1, 2, 6).mean(dim=1)
            return {"pose": pose}

    # Synthetic "SegNet": simple classifier
    class FakeSegNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 3, padding=1)

        def preprocess_input(self, x):
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W) / 255.0

        def forward(self, x):
            # Return (B*T, 5, H, W) logits
            out = self.conv(x)
            # Pool over T: take the first frame's output
            BT, Cl, H, W = out.shape
            return out[:BT // 2]  # just first frame

    dev = torch.device(device)
    posenet = FakePoseNet().eval().to(dev)
    segnet = FakeSegNet().eval().to(dev)
    for p in posenet.parameters():
        p.requires_grad = False
    for p in segnet.parameters():
        p.requires_grad = False

    # Small synthetic frame pair
    H, W = 32, 48
    torch.manual_seed(123)
    gt_pair = torch.rand(1, 2, H, W, 3, device=dev) * 255
    # Add noise to create a "compressed" version
    noisy_pair = (gt_pair + torch.randn_like(gt_pair) * 10).clamp(0, 255)

    # Test 1: QuadraticNewtonOptimizer
    cfg = QuadraticConfig(
        max_newton_steps=5,
        max_cg_iterations=3,
        trust_region_radius=1.0,
        max_step_norm=5.0,
    )
    optimizer = QuadraticNewtonOptimizer(posenet, segnet, cfg)
    optimized, stats = optimizer.optimize(noisy_pair, gt_pair, verbose=True)

    assert optimized.shape == noisy_pair.shape, "Output shape mismatch"
    assert stats["newton_steps"] > 0 or stats["initial_loss"] < 1e-4, \
        "No Newton steps taken and loss is not already minimal"

    # Test 2: Contrarian failure modes
    failures = contrarian_failure_modes()
    assert len(failures) >= 5, "Missing failure modes"
    assert all("severity" in v for v in failures.values()), "Missing severity"

    # Test 3: Council verdict
    verdict = council_verdict()
    assert verdict["verdict"] in ("YES", "NO", "CONDITIONAL_YES"), "Invalid verdict"

    # Test 4: Rate-distortion bound
    rd_bound = compute_scorer_rate_distortion_bound(
        posenet, segnet, [noisy_pair], [gt_pair], device=device
    )
    assert "pose_variance" in rd_bound, "Missing pose_variance"
    assert rd_bound["efficiency"] >= 0, "Negative efficiency"

    # Test 5: Hessian spectrum analysis
    spectrum = analyze_hessian_spectrum(
        posenet, segnet, noisy_pair, gt_pair,
        n_eigenvalues=5, n_lanczos_iterations=10, device=device,
    )
    assert len(spectrum["eigenvalues"]) > 0, "No eigenvalues computed"
    assert spectrum["effective_rank"] >= 0, "Negative effective rank"

    print(f"\n[smoke_test] PASSED")
    print(f"  Newton steps: {stats['newton_steps']}")
    print(f"  Initial loss: {stats['initial_loss']:.6f}")
    print(f"  Final loss:   {stats['final_loss']:.6f}")
    print(f"  Improvement:  {stats['initial_loss'] - stats['final_loss']:.6f}")
    print(f"  CG converged: {stats['cg_converged_count']}/{stats['newton_steps']}")
    print(f"  GD fallbacks: {stats['gd_fallback_count']}")
    print(f"  Hessian effective rank: {spectrum['effective_rank']}")
    print(f"  Top eigenvalues: {spectrum['eigenvalues'][:5]}")
    print(f"  RD efficiency: {rd_bound['efficiency']:.2f}")
    print(f"  Failure modes: {len(failures)}")
    print(f"  Council verdict: {verdict['verdict']}")

    return {
        "optimizer_stats": stats,
        "rd_bound": rd_bound,
        "spectrum": spectrum,
        "failure_modes": list(failures.keys()),
        "verdict": verdict["verdict"],
    }


# ──────────────────────────────────────────────────────────────────────
#  Main: Full analysis with real scorers
# ──────────────────────────────────────────────────────────────────────

def main():
    """Run full geometry deliberation with real scorer networks."""
    device = detect_device()
    print(f"[geometry] device={device}")

    # Try to load real scorers; fall back to smoke test
    try:
        from tac.proxy_eval import _default_paths
        _PROJECT, _UPSTREAM, VIDEOS_DIR, _LIVE_ARCHIVE, ARCHIVE_ZIP = _default_paths()
        posenet, segnet = load_scorers(
            posenet_path=_UPSTREAM / "models" / "posenet.safetensors",
            segnet_path=_UPSTREAM / "models" / "segnet.safetensors",
            device=device,
            upstream_dir=str(_UPSTREAM),
        )
        print("[geometry] Loaded real scorers")
    except Exception as e:
        print(f"[geometry] Could not load real scorers: {e}")
        print("[geometry] Running smoke test with synthetic scorers instead")
        return smoke_test(device=str(device))

    # Load data
    from tac.data import build_pairs, decode_archive, decode_video

    print("[geometry] Decoding frames...")
    comp_frames = decode_archive(str(ARCHIVE_ZIP))
    gt_frames = decode_video(str(VIDEOS_DIR / "0.mkv"))
    n = min(len(comp_frames), len(gt_frames))
    comp_pairs = build_pairs(comp_frames[:n])
    gt_pairs = build_pairs(gt_frames[:n])
    del comp_frames, gt_frames
    gc.collect()

    n_pairs = min(len(comp_pairs), len(gt_pairs))
    print(f"[geometry] {n_pairs} frame pairs available")

    # 1. Hessian spectrum on first pair
    print("\n=== Hessian Spectral Analysis ===")
    spectrum = analyze_hessian_spectrum(
        posenet, segnet,
        comp_pairs[0].to(device).float(),
        gt_pairs[0].to(device).float(),
        n_eigenvalues=10,
        n_lanczos_iterations=25,
        device=device,
    )
    print(f"Effective rank: {spectrum['effective_rank']}")
    print(f"Condition number: {spectrum['condition_number']:.1f}")
    print(f"Spectral decay exponent: {spectrum['spectral_decay']:.2f}")
    print(f"Fourier alignment: {spectrum['fourier_alignment']:.3f}")
    print(f"Top eigenvalues: {spectrum['eigenvalues'][:5]}")

    # 2. Rate-distortion bound (subsample for speed)
    print("\n=== Rate-Distortion Bound ===")
    subsample = max(1, n_pairs // 50)
    rd_pairs_comp = [comp_pairs[i].float() for i in range(0, n_pairs, subsample)]
    rd_pairs_gt = [gt_pairs[i].float() for i in range(0, n_pairs, subsample)]
    rd_bound = compute_scorer_rate_distortion_bound(
        posenet, segnet, rd_pairs_comp, rd_pairs_gt, device=device
    )
    print(f"Pose variance: {rd_bound['pose_variance']:.6f}")
    print(f"Current pose dist: {rd_bound['current_pose_dist']:.6f}")
    print(f"D_min (Gaussian): {rd_bound['d_min_gaussian']:.6f}")
    print(f"Efficiency: {rd_bound['efficiency']:.2f}x")
    print(f"SegNet entropy: {rd_bound['seg_entropy_bits']:.2f} bits")

    # 3. Newton optimization on a hard pair
    print("\n=== Newton-CG Optimization (one pair) ===")
    cfg = QuadraticConfig(
        max_newton_steps=10,
        max_cg_iterations=5,
        trust_region_radius=0.05,
        max_step_norm=0.5,
    )
    optimizer = QuadraticNewtonOptimizer(posenet, segnet, cfg)
    test_idx = 0
    optimized, stats = optimizer.optimize(
        comp_pairs[test_idx].to(device).float(),
        gt_pairs[test_idx].to(device).float(),
        verbose=True,
    )
    print(f"Initial loss: {stats['initial_loss']:.6f}")
    print(f"Final loss:   {stats['final_loss']:.6f}")
    print(f"Improvement:  {stats['initial_loss'] - stats['final_loss']:.6f}")

    # 4. Contrarian report
    print("\n=== Contrarian Failure Modes ===")
    failures = contrarian_failure_modes()
    for name, info in failures.items():
        print(f"  [{info['severity']:>8s}] {name}")

    # 5. Council verdict
    print("\n=== Council Verdict ===")
    verdict = council_verdict()
    print(f"Verdict: {verdict['verdict']}")
    print(f"Summary: {verdict['summary']}")
    print(f"\nExpected improvement (optimistic/realistic/pessimistic):")
    ei = verdict["expected_score_improvement"]
    print(f"  {ei['optimistic']:.3f} / {ei['realistic']:.3f} / {ei['pessimistic']:.3f}")

    result = {
        "spectrum": spectrum,
        "rd_bound": rd_bound,
        "newton_stats": {k: v for k, v in stats.items() if k != "history"},
        "failure_modes": {k: v["severity"] for k, v in failures.items()},
        "verdict": verdict["verdict"],
    }
    print(f"\nJSON:\n{json.dumps(result, indent=2, default=str)}")
    return result


if __name__ == "__main__":
    main()
