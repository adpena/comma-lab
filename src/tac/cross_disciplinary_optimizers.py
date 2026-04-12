"""Cross-disciplinary optimization algorithms for scorer-optimal frame generation.

Implements 16 optimization algorithms from physics, biology, chemistry, geophysics,
climate science, astrophysics, and quantum computing — all adapted to the constrained
frame optimization problem: minimize S = 100*seg_distortion + sqrt(10*pose_distortion) + 25*rate
by directly optimizing pixel values against frozen scorer networks.

The optimization landscape is 384x512x3 = 589,824 dimensional pixel space. Scorer
networks (PoseNet, SegNet) are frozen. We optimize pixel values directly. The loss
landscape is highly non-convex with many local minima.

Each optimizer is a self-contained class that takes frames (as torch tensors), scorer
models, and config, and returns optimized frames. All hyperparameters are configurable.

Usage::

    from tac.cross_disciplinary_optimizers import optimizer_factory, ensemble_optimize

    opt = optimizer_factory("simulated_annealing", {"T0": 100.0, "alpha": 0.99})
    optimized = opt.optimize(frames, posenet, segnet, masks, expected_pose)

    # Or run an ensemble:
    best = ensemble_optimize(frames, posenet, segnet, masks, expected_pose,
                             optimizers=["simulated_annealing", "basin_hopping", "cma_es"],
                             config={"num_steps": 500})
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "CrossDisciplinaryOptimizer",
    "SimulatedAnnealing",
    "HamiltonianMonteCarlo",
    "LangevinDynamics",
    "ReplicaExchange",
    "CMAES",
    "DifferentialEvolution",
    "ParticleSwarmOptimization",
    "Metadynamics",
    "BasinHopping",
    "FullWaveformInversion",
    "SeismicMultiGrid",
    "EnsembleKalmanFilter",
    "FourDVar",
    "NestedSampling",
    "MultigridRelaxation",
    "QuantumAnnealingSimulation",
    "optimizer_factory",
    "ensemble_optimize",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _scorer_loss_fn(
    frames: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    compress_weight: float = 1.0,
) -> torch.Tensor:
    """Unified loss: 100*seg + sqrt(10*pose) + compress_weight*TV.

    Handles the full forward pass through both scorers and returns a
    differentiable scalar loss. All inputs must be on the same device.

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255], requires_grad.
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        masks: (N, H_seg, W_seg) long tensor of target class indices.
        expected_pose: (P, 6) float tensor of expected pose outputs (P = N-1).
        seg_weight: weight on segmentation distortion (default 100).
        pose_weight: multiplier inside sqrt for pose (default 10).
        compress_weight: weight on total-variation compressibility term.

    Returns:
        Scalar loss tensor with gradient graph attached.
    """
    from tac.constrained_gen import (
        compute_compressibility_loss,
        compute_posenet_constraint_loss,
        compute_segnet_constraint_loss,
    )

    seg_loss = compute_segnet_constraint_loss(frames, masks, segnet)
    pose_loss = compute_posenet_constraint_loss(frames, expected_pose, posenet)
    compress_loss = compute_compressibility_loss(frames)

    total = seg_weight * seg_loss + torch.sqrt(pose_weight * pose_loss + 1e-8) + compress_weight * compress_loss
    return total


def _scorer_loss_no_grad(
    frames: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    compress_weight: float = 1.0,
) -> float:
    """Evaluate loss without gradient computation. Returns Python float."""
    with torch.no_grad():
        loss = _scorer_loss_fn(
            frames, posenet, segnet, masks, expected_pose,
            seg_weight, pose_weight, compress_weight,
        )
    return loss.item()


def _compute_gradient(
    frames: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    compress_weight: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Compute gradient of scorer loss w.r.t. frames.

    Returns:
        (gradient tensor same shape as frames, loss value as float)
    """
    frames_opt = frames.detach().clone().requires_grad_(True)
    loss = _scorer_loss_fn(
        frames_opt, posenet, segnet, masks, expected_pose,
        seg_weight, pose_weight, compress_weight,
    )
    loss.backward()
    return frames_opt.grad.detach(), loss.item()


def _clamp_frames(frames: torch.Tensor) -> torch.Tensor:
    """Clamp pixel values to valid [0, 255] range."""
    return frames.clamp(0.0, 255.0)


def _pca_reduce(frames_flat: torch.Tensor, jacobian_eigvecs: torch.Tensor) -> torch.Tensor:
    """Project high-dimensional frame space into PCA-reduced subspace.

    Args:
        frames_flat: (N, D) flattened frames.
        jacobian_eigvecs: (D, K) top-K eigenvectors of scorer Jacobian.

    Returns:
        (N, K) reduced coordinates.
    """
    return frames_flat @ jacobian_eigvecs


def _pca_reconstruct(reduced: torch.Tensor, jacobian_eigvecs: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """Reconstruct frames from PCA-reduced coordinates.

    Args:
        reduced: (N, K) reduced coordinates.
        jacobian_eigvecs: (D, K) top-K eigenvectors.
        mean: (D,) mean vector for centering.

    Returns:
        (N, D) reconstructed flattened frames.
    """
    return reduced @ jacobian_eigvecs.T + mean


def _compute_jacobian_eigvecs(
    frames: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    n_components: int = 64,
    n_samples: int = 32,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
) -> torch.Tensor:
    """Estimate top eigenvectors of the scorer Jacobian via random probing.

    Uses randomized power iteration: perturb frames in random directions,
    measure gradient change, build approximate Jacobian subspace.

    Args:
        frames: (N, H, W, 3) reference frames.
        n_components: number of top eigenvectors to extract.
        n_samples: number of random probes for estimation.

    Returns:
        (D, n_components) matrix of top eigenvectors, where D = N*H*W*3.
    """
    device = frames.device
    shape = frames.shape
    D = frames.numel()
    n_components = min(n_components, D, n_samples)

    # Collect gradient samples from random perturbations
    grad_samples = []
    for _ in range(n_samples):
        perturbed = frames + torch.randn_like(frames) * 5.0
        perturbed = _clamp_frames(perturbed)
        grad, _ = _compute_gradient(perturbed, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
        grad_samples.append(grad.reshape(-1))

    G = torch.stack(grad_samples, dim=0)  # (n_samples, D)
    # SVD on the gradient matrix to find principal directions
    # Use economy SVD: only compute n_components singular vectors
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    # Vh rows are right singular vectors of G = principal gradient directions
    eigvecs = Vh[:n_components].T  # (D, n_components)
    return eigvecs


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CrossDisciplinaryOptimizer:
    """Base class for all cross-disciplinary optimizers.

    Subclasses must implement ``optimize()`` which takes frames and scorer
    models and returns optimized frames.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.diagnostics: dict[str, Any] = {}

    def optimize(
        self,
        frames: torch.Tensor,
        posenet: torch.nn.Module,
        segnet: torch.nn.Module,
        masks: torch.Tensor,
        expected_pose: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Optimize frames to minimize scorer loss.

        Args:
            frames: (N, H, W, 3) float tensor in [0, 255].
            posenet: frozen PoseNet model.
            segnet: frozen SegNet model.
            masks: (N, H_seg, W_seg) long tensor of target class indices.
            expected_pose: (P, 6) float tensor where P = N-1.

        Returns:
            (N, H, W, 3) optimized frames in [0, 255].
        """
        raise NotImplementedError

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic information from the last optimization run."""
        return self.diagnostics


# ===========================================================================
# FROM PHYSICS
# ===========================================================================


class SimulatedAnnealing(CrossDisciplinaryOptimizer):
    """Simulated Annealing for frame optimization (Kirkpatrick et al. 1983).

    Reference: Kirkpatrick, Gelatt & Vecchi, "Optimization by Simulated Annealing",
    Science 220(4598):671-680, 1983.

    Why it helps: The scorer loss landscape has many local minima due to the
    non-convex interaction between SegNet (discrete argmax) and PoseNet (continuous
    regression). Gradient descent gets trapped in these basins. SA escapes via
    stochastic acceptance of worse solutions with probability exp(-deltaE/T),
    controlled by a cooling schedule. As temperature decreases, the algorithm
    transitions from global exploration to local refinement.

    Configurable: T0 (initial temperature), alpha (cooling rate), cooling_schedule
    (exponential/linear/logarithmic), num_steps, perturbation_scale.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        T0 = cfg.get("T0", 100.0)
        alpha = cfg.get("alpha", 0.995)
        num_steps = cfg.get("num_steps", 500)
        cooling = cfg.get("cooling_schedule", "exponential")
        perturb_scale = cfg.get("perturbation_scale", 5.0)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device
        current = frames.clone()
        current_loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)

        best = current.clone()
        best_loss = current_loss

        losses = []
        accepts = 0

        for step in range(num_steps):
            # Temperature schedule
            if cooling == "exponential":
                T = T0 * (alpha ** step)
            elif cooling == "linear":
                T = T0 * max(1e-8, 1.0 - step / num_steps)
            elif cooling == "logarithmic":
                T = T0 / (1.0 + math.log(1.0 + step))
            else:
                T = T0 * (alpha ** step)

            # Perturbation scaled by temperature
            noise = torch.randn_like(current) * perturb_scale * math.sqrt(T / T0)
            candidate = _clamp_frames(current + noise)

            candidate_loss = _scorer_loss_no_grad(candidate, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)

            # Metropolis criterion
            delta = candidate_loss - current_loss
            if delta < 0 or (T > 1e-12 and torch.rand(1).item() < math.exp(-delta / T)):
                current = candidate
                current_loss = candidate_loss
                accepts += 1

            if current_loss < best_loss:
                best = current.clone()
                best_loss = current_loss

            losses.append(current_loss)

        self.diagnostics = {
            "final_loss": best_loss,
            "acceptance_rate": accepts / max(1, num_steps),
            "loss_history": losses,
            "num_steps": num_steps,
        }
        return best


class HamiltonianMonteCarlo(CrossDisciplinaryOptimizer):
    """Hamiltonian Monte Carlo for frame optimization (Duane et al. 1987).

    Reference: Duane, Kennedy, Pendleton & Roweth, "Hybrid Monte Carlo",
    Physics Letters B 195(2):216-222, 1987.

    Why it helps: HMC uses the geometric structure of the scorer loss landscape
    (via gradients) to propose distant moves that still have high acceptance
    probability. The leapfrog integrator conserves the Hamiltonian (approximately),
    so proposals traverse long distances along iso-loss contours instead of
    making small random steps. This is far more efficient than random-walk MCMC
    in 589K-dimensional space.

    q = pixel values, p = momentum, H(q,p) = scorer_loss(q) + 0.5*p^T*M^{-1}*p.
    The mass matrix M can be learned from scorer Jacobian eigenvalues.

    Configurable: step_size, num_leapfrog_steps, mass_matrix_type, num_steps.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        step_size = cfg.get("step_size", 0.01)
        num_leapfrog = cfg.get("num_leapfrog_steps", 10)
        num_steps = cfg.get("num_steps", 100)
        mass_type = cfg.get("mass_matrix_type", "identity")
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device
        current_q = frames.clone()
        best = current_q.clone()
        best_loss = _scorer_loss_no_grad(current_q, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)

        # Mass matrix: diagonal, either identity or learned from gradient norms
        if mass_type == "learned":
            grad, _ = _compute_gradient(current_q, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
            mass_inv = 1.0 / (grad.abs().clamp(min=1e-6) + 1.0)
        else:
            mass_inv = torch.ones_like(current_q)

        losses = []
        accepts = 0

        for step in range(num_steps):
            # Sample momentum
            p = torch.randn_like(current_q) / torch.sqrt(mass_inv)

            # Current Hamiltonian
            current_grad, current_U = _compute_gradient(
                current_q, posenet, segnet, masks, expected_pose, seg_weight, pose_weight,
            )
            current_K = 0.5 * (p * p * mass_inv).sum().item()

            # Leapfrog integration
            q = current_q.clone()
            p = p - 0.5 * step_size * current_grad

            for lf in range(num_leapfrog):
                q = _clamp_frames(q + step_size * p * mass_inv)
                grad, _ = _compute_gradient(q, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
                if lf < num_leapfrog - 1:
                    p = p - step_size * grad
                else:
                    p = p - 0.5 * step_size * grad

            # Proposed Hamiltonian
            proposed_U = _scorer_loss_no_grad(q, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
            proposed_K = 0.5 * (p * p * mass_inv).sum().item()

            # Metropolis accept/reject
            delta_H = (proposed_U + proposed_K) - (current_U + current_K)
            if delta_H < 0 or torch.rand(1).item() < math.exp(-min(delta_H, 500.0)):
                current_q = q
                accepts += 1

            loss = _scorer_loss_no_grad(current_q, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
            if loss < best_loss:
                best = current_q.clone()
                best_loss = loss
            losses.append(loss)

        self.diagnostics = {
            "final_loss": best_loss,
            "acceptance_rate": accepts / max(1, num_steps),
            "loss_history": losses,
        }
        return best


class LangevinDynamics(CrossDisciplinaryOptimizer):
    """Stochastic Gradient Langevin Dynamics (Welling & Teh 2011).

    Reference: Welling & Teh, "Bayesian Learning via Stochastic Gradient Langevin
    Dynamics", ICML 2011.

    Why it helps: SGLD adds calibrated noise to gradient descent, sampling from
    the posterior over valid frames rather than finding a single MAP estimate.
    The noise prevents collapse to sharp local minima (which often correspond to
    scorer adversarial examples that don't generalize). Annealing the inverse
    temperature beta transitions from exploration to exploitation.

    Update: x_{t+1} = x_t - eta * grad_loss(x_t) + sqrt(2*eta/beta) * epsilon.

    Configurable: lr, beta_start, beta_end, num_steps, noise_schedule.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        lr = cfg.get("lr", 0.1)
        beta_start = cfg.get("beta_start", 0.1)
        beta_end = cfg.get("beta_end", 100.0)
        num_steps = cfg.get("num_steps", 500)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        current = frames.clone()
        best = current.clone()
        best_loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
        losses = []

        for step in range(num_steps):
            # Anneal inverse temperature (log-linear)
            t_frac = step / max(1, num_steps - 1)
            beta = beta_start * ((beta_end / beta_start) ** t_frac)

            grad, loss_val = _compute_gradient(
                current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight,
            )

            # SGLD update
            noise_scale = math.sqrt(2.0 * lr / max(beta, 1e-8))
            noise = torch.randn_like(current) * noise_scale
            current = _clamp_frames(current - lr * grad + noise)

            loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
            if loss < best_loss:
                best = current.clone()
                best_loss = loss
            losses.append(loss)

        self.diagnostics = {"final_loss": best_loss, "loss_history": losses}
        return best


class ReplicaExchange(CrossDisciplinaryOptimizer):
    """Replica Exchange / Parallel Tempering (Swendsen & Wang 1986).

    Reference: Swendsen & Wang, "Replica Monte Carlo Simulation of Spin-Glasses",
    Physical Review Letters 57(21):2607, 1986.

    Why it helps: Runs K replicas of the optimization at different temperatures
    simultaneously. Hot replicas explore the scorer landscape freely (accepting
    worse solutions), while cold replicas exploit local structure. Periodic swaps
    between adjacent temperature levels transfer global information (good basins
    discovered by hot replicas) to cold replicas for refinement.

    Configurable: num_replicas, T_min, T_max, swap_every, steps_per_sweep.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        K = cfg.get("num_replicas", 4)
        T_min = cfg.get("T_min", 0.1)
        T_max = cfg.get("T_max", 100.0)
        swap_every = cfg.get("swap_every", 10)
        num_steps = cfg.get("num_steps", 200)
        perturb_scale = cfg.get("perturbation_scale", 5.0)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device

        # Geometric temperature ladder
        temps = [T_min * ((T_max / T_min) ** (k / max(1, K - 1))) for k in range(K)]

        # Initialize replicas
        replicas = [frames.clone() for _ in range(K)]
        replica_losses = [
            _scorer_loss_no_grad(r, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
            for r in replicas
        ]

        best = frames.clone()
        best_loss = min(replica_losses)
        swaps = 0
        losses = []

        for step in range(num_steps):
            # Each replica does one SA step at its temperature
            for k in range(K):
                T = temps[k]
                noise = torch.randn_like(replicas[k]) * perturb_scale * math.sqrt(T / T_max)
                candidate = _clamp_frames(replicas[k] + noise)
                cand_loss = _scorer_loss_no_grad(candidate, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)

                delta = cand_loss - replica_losses[k]
                if delta < 0 or (T > 1e-12 and torch.rand(1).item() < math.exp(-delta / T)):
                    replicas[k] = candidate
                    replica_losses[k] = cand_loss

            # Swap adjacent replicas
            if step % swap_every == 0 and step > 0:
                for k in range(K - 1):
                    # Metropolis criterion for swap
                    beta_k = 1.0 / max(temps[k], 1e-12)
                    beta_k1 = 1.0 / max(temps[k + 1], 1e-12)
                    delta = (beta_k - beta_k1) * (replica_losses[k + 1] - replica_losses[k])
                    if delta > 0 or torch.rand(1).item() < math.exp(delta):
                        replicas[k], replicas[k + 1] = replicas[k + 1], replicas[k]
                        replica_losses[k], replica_losses[k + 1] = replica_losses[k + 1], replica_losses[k]
                        swaps += 1

            # Track best from coldest replica
            if replica_losses[0] < best_loss:
                best = replicas[0].clone()
                best_loss = replica_losses[0]
            losses.append(best_loss)

        self.diagnostics = {
            "final_loss": best_loss,
            "num_swaps": swaps,
            "replica_losses": replica_losses,
            "loss_history": losses,
        }
        return best


# ===========================================================================
# FROM BIOLOGY
# ===========================================================================


class CMAES(CrossDisciplinaryOptimizer):
    """CMA-ES — Covariance Matrix Adaptation Evolution Strategy (Hansen 2001).

    Reference: Hansen & Ostermeier, "Completely Derandomized Self-Adaptation in
    Evolution Strategies", Evolutionary Computation 9(2):159-195, 2001.

    Why it helps: CMA-ES adapts the search distribution (mean + covariance) to
    the local curvature of the loss landscape. It is derivative-free, so it
    handles the non-smooth SegNet argmax loss naturally. By operating in a
    PCA-reduced subspace of the scorer Jacobian, we make the covariance matrix
    tractable (K x K instead of 589K x 589K).

    Configurable: pop_size, n_components, sigma0, num_generations.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        pop_size = cfg.get("pop_size", 16)
        n_comp = cfg.get("n_components", 32)
        sigma0 = cfg.get("sigma0", 10.0)
        num_gen = cfg.get("num_steps", 100)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device
        shape = frames.shape
        D = frames.numel()
        n_comp = min(n_comp, D)
        mu_select = pop_size // 2  # number of parents

        # Compute PCA basis from Jacobian
        eigvecs = _compute_jacobian_eigvecs(
            frames, posenet, segnet, masks, expected_pose,
            n_components=n_comp, n_samples=max(n_comp + 4, 32),
            seg_weight=seg_weight, pose_weight=pose_weight,
        )  # (D, n_comp)

        # Initialize in reduced space
        mean_full = frames.reshape(-1)
        mean_z = (mean_full @ eigvecs).clone()  # (n_comp,)
        sigma = sigma0
        C = torch.eye(n_comp, device=device, dtype=frames.dtype)
        p_sigma = torch.zeros(n_comp, device=device, dtype=frames.dtype)
        p_c = torch.zeros(n_comp, device=device, dtype=frames.dtype)

        # CMA-ES weights
        weights = torch.tensor(
            [math.log(mu_select + 0.5) - math.log(i + 1) for i in range(mu_select)],
            device=device, dtype=frames.dtype,
        )
        weights = weights / weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum().item()

        # Learning rates
        c_sigma = (mu_eff + 2.0) / (n_comp + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n_comp + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n_comp) / (n_comp + 4.0 + 2.0 * mu_eff / n_comp)
        c1 = 2.0 / ((n_comp + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n_comp + 2.0) ** 2 + mu_eff))
        chi_n = math.sqrt(n_comp) * (1.0 - 1.0 / (4.0 * n_comp) + 1.0 / (21.0 * n_comp ** 2))

        best = frames.clone()
        best_loss = _scorer_loss_no_grad(frames, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
        losses = []

        for gen in range(num_gen):
            # Sample population
            try:
                L = torch.linalg.cholesky(C)
            except torch.linalg.LinAlgError:
                C = C + 1e-4 * torch.eye(n_comp, device=device, dtype=frames.dtype)
                L = torch.linalg.cholesky(C)

            z_samples = torch.randn(pop_size, n_comp, device=device, dtype=frames.dtype)
            y_samples = z_samples @ L.T  # (pop_size, n_comp)
            x_samples = mean_z.unsqueeze(0) + sigma * y_samples  # (pop_size, n_comp)

            # Evaluate fitness
            fitnesses = []
            for i in range(pop_size):
                full_vec = x_samples[i] @ eigvecs.T + (mean_full - mean_z @ eigvecs.T)
                candidate = _clamp_frames(full_vec.reshape(shape))
                fit = _scorer_loss_no_grad(candidate, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
                fitnesses.append(fit)

            # Sort by fitness (lower is better)
            indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
            best_idx = indices[0]

            if fitnesses[best_idx] < best_loss:
                full_vec = x_samples[best_idx] @ eigvecs.T + (mean_full - mean_z @ eigvecs.T)
                best = _clamp_frames(full_vec.reshape(shape))
                best_loss = fitnesses[best_idx]

            # Recombination: weighted mean of best mu_select
            old_mean = mean_z.clone()
            mean_z = torch.zeros_like(mean_z)
            for rank, idx in enumerate(indices[:mu_select]):
                mean_z += weights[rank] * x_samples[idx]

            # Update evolution paths
            y_mean = (mean_z - old_mean) / sigma
            z_mean = torch.linalg.solve(L, y_mean)

            p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * z_mean
            h_sigma = 1.0 if p_sigma.norm().item() / math.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2.0 / (n_comp + 1.0)) * chi_n else 0.0

            p_c = (1.0 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_mean

            # Covariance matrix update
            artmp = torch.stack([(x_samples[indices[j]] - old_mean) / sigma for j in range(mu_select)], dim=0)
            rank_mu_update = sum(weights[j] * artmp[j].unsqueeze(1) @ artmp[j].unsqueeze(0) for j in range(mu_select))

            C = (
                (1.0 - c1 - c_mu) * C
                + c1 * (p_c.unsqueeze(1) @ p_c.unsqueeze(0) + (1.0 - h_sigma) * c_c * (2.0 - c_c) * C)
                + c_mu * rank_mu_update
            )

            # Step-size update
            sigma *= math.exp((c_sigma / d_sigma) * (p_sigma.norm().item() / chi_n - 1.0))
            sigma = max(1e-8, min(sigma, 1000.0))

            losses.append(best_loss)

        self.diagnostics = {"final_loss": best_loss, "final_sigma": sigma, "loss_history": losses}
        return best


class DifferentialEvolution(CrossDisciplinaryOptimizer):
    """Differential Evolution for frame optimization (Storn & Price 1997).

    Reference: Storn & Price, "Differential Evolution — A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces", Journal of Global
    Optimization 11(4):341-359, 1997.

    Why it helps: DE maintains a diverse population of candidate frames. The
    mutation operator (a + F*(b-c)) creates new candidates by combining
    differences between existing population members, automatically adapting
    the search scale to the landscape. Population diversity prevents premature
    convergence to scorer adversarial examples.

    Configurable: pop_size, F (scale factor), CR (crossover rate), num_steps.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        pop_size = cfg.get("pop_size", 20)
        F_scale = cfg.get("F", 0.8)
        CR = cfg.get("CR", 0.9)
        num_steps = cfg.get("num_steps", 200)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device
        shape = frames.shape

        # Initialize population with perturbations around input
        population = [_clamp_frames(frames + torch.randn_like(frames) * 10.0) for _ in range(pop_size)]
        population[0] = frames.clone()  # Keep original as one member

        fitnesses = [
            _scorer_loss_no_grad(p, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
            for p in population
        ]

        best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
        best = population[best_idx].clone()
        best_loss = fitnesses[best_idx]
        losses = []

        for step in range(num_steps):
            for i in range(pop_size):
                # Select 3 distinct random indices != i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = [candidates[int(torch.randint(len(candidates), (1,)).item())] for _ in range(3)]

                # Mutation: donor = a + F*(b - c)
                donor = _clamp_frames(population[a] + F_scale * (population[b] - population[c]))

                # Crossover: binomial
                mask = torch.rand_like(frames) < CR
                # Ensure at least one dimension is from donor
                j_rand = torch.randint(frames.numel(), (1,)).item()
                mask_flat = mask.reshape(-1)
                mask_flat[j_rand] = True
                mask = mask_flat.reshape(shape)

                trial = torch.where(mask, donor, population[i])
                trial = _clamp_frames(trial)

                trial_fit = _scorer_loss_no_grad(trial, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)

                if trial_fit <= fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fit

                    if trial_fit < best_loss:
                        best = trial.clone()
                        best_loss = trial_fit

            losses.append(best_loss)

        self.diagnostics = {"final_loss": best_loss, "loss_history": losses}
        return best


class ParticleSwarmOptimization(CrossDisciplinaryOptimizer):
    """Particle Swarm Optimization (Kennedy & Eberhart 1995).

    Reference: Kennedy & Eberhart, "Particle Swarm Optimization", IEEE ICNN 1995.

    Why it helps: PSO uses swarm intelligence — particles share information about
    good regions via the global best, while maintaining individual exploration via
    personal bests. Operates in PCA-reduced space (top-K scorer Jacobian
    eigenvectors) to make the swarm tractable in high-dimensional frame space.

    Configurable: num_particles, w (inertia), c1 (cognitive), c2 (social),
    n_components, num_steps.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        n_particles = cfg.get("num_particles", 16)
        w = cfg.get("w", 0.7)
        c1 = cfg.get("c1", 1.5)
        c2 = cfg.get("c2", 1.5)
        n_comp = cfg.get("n_components", 32)
        num_steps = cfg.get("num_steps", 200)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device
        shape = frames.shape
        D = frames.numel()
        n_comp = min(n_comp, D)

        # Compute PCA basis
        eigvecs = _compute_jacobian_eigvecs(
            frames, posenet, segnet, masks, expected_pose,
            n_components=n_comp, n_samples=max(n_comp + 4, 32),
            seg_weight=seg_weight, pose_weight=pose_weight,
        )

        mean_full = frames.reshape(-1)

        def to_full(z):
            return _clamp_frames((z @ eigvecs.T + mean_full).reshape(shape))

        def evaluate(z):
            return _scorer_loss_no_grad(to_full(z), posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)

        # Initialize particles
        positions = torch.randn(n_particles, n_comp, device=device, dtype=frames.dtype) * 10.0
        positions[0] = torch.zeros(n_comp, device=device, dtype=frames.dtype)
        velocities = torch.randn(n_particles, n_comp, device=device, dtype=frames.dtype) * 1.0

        # Personal and global bests
        pbest_pos = positions.clone()
        pbest_fit = torch.tensor([evaluate(positions[i]) for i in range(n_particles)], device=device)

        gbest_idx = pbest_fit.argmin().item()
        gbest_pos = pbest_pos[gbest_idx].clone()
        gbest_fit = pbest_fit[gbest_idx].item()

        losses = []

        for step in range(num_steps):
            r1 = torch.rand(n_particles, n_comp, device=device, dtype=frames.dtype)
            r2 = torch.rand(n_particles, n_comp, device=device, dtype=frames.dtype)

            velocities = (
                w * velocities
                + c1 * r1 * (pbest_pos - positions)
                + c2 * r2 * (gbest_pos.unsqueeze(0) - positions)
            )

            positions = positions + velocities

            # Evaluate and update bests
            for i in range(n_particles):
                fit = evaluate(positions[i])
                if fit < pbest_fit[i].item():
                    pbest_pos[i] = positions[i].clone()
                    pbest_fit[i] = fit
                    if fit < gbest_fit:
                        gbest_pos = positions[i].clone()
                        gbest_fit = fit

            losses.append(gbest_fit)

        self.diagnostics = {"final_loss": gbest_fit, "loss_history": losses}
        return to_full(gbest_pos)


# ===========================================================================
# FROM CHEMISTRY
# ===========================================================================


class Metadynamics(CrossDisciplinaryOptimizer):
    """Metadynamics for frame optimization (Laio & Parrinello 2002).

    Reference: Laio & Parrinello, "Escaping free-energy minima", PNAS 99(20):
    12562-12566, 2002.

    Why it helps: Metadynamics adds repulsive Gaussian bias potentials at
    previously visited points in collective-variable space (here: scorer outputs
    = pose_dist, seg_dist). This progressively fills in local minima, forcing
    the optimizer to explore new regions. Unlike SA which relies on random
    fluctuations, metadynamics systematically eliminates known basins.

    Configurable: gaussian_height, gaussian_width, deposit_every, lr, num_steps.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        gauss_h = cfg.get("gaussian_height", 1.0)
        gauss_w = cfg.get("gaussian_width", 0.1)
        deposit_every = cfg.get("deposit_every", 10)
        lr = cfg.get("lr", 0.1)
        num_steps = cfg.get("num_steps", 500)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device

        current = frames.clone()
        best = current.clone()
        best_loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)

        # Bias history: list of (collective_variable_center, height)
        bias_centers: list[torch.Tensor] = []
        losses = []

        for step in range(num_steps):
            # Compute gradient of original loss
            grad, loss_val = _compute_gradient(
                current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight,
            )

            # Compute collective variables (low-dim summary of state)
            # Use a hash of the frame's scorer response as CV
            with torch.no_grad():
                cv = self._compute_cv(current, posenet, segnet, masks, expected_pose)

            # Add bias gradient: repel from previously visited CVs
            if bias_centers:
                bias_grad = torch.zeros_like(current)
                for center in bias_centers:
                    diff = cv - center
                    dist_sq = (diff ** 2).sum()
                    # Gradient of Gaussian bias in CV space, projected back
                    gauss_val = gauss_h * torch.exp(-dist_sq / (2.0 * gauss_w ** 2))
                    # Approximate: push away from visited region proportional to proximity
                    bias_grad += gauss_val * grad * 0.1  # use loss gradient direction as proxy

                grad = grad - bias_grad  # bias pushes away from visited minima

            current = _clamp_frames(current - lr * grad)

            # Deposit bias
            if step % deposit_every == 0:
                with torch.no_grad():
                    cv = self._compute_cv(current, posenet, segnet, masks, expected_pose)
                bias_centers.append(cv.clone())

            loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
            if loss < best_loss:
                best = current.clone()
                best_loss = loss
            losses.append(loss)

        self.diagnostics = {"final_loss": best_loss, "num_bias_deposits": len(bias_centers), "loss_history": losses}
        return best

    @staticmethod
    def _compute_cv(
        frames: torch.Tensor,
        posenet: torch.nn.Module,
        segnet: torch.nn.Module,
        masks: torch.Tensor,
        expected_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Compute collective variables: (seg_loss, pose_loss) as a 2D summary."""
        from tac.constrained_gen import (
            compute_posenet_constraint_loss,
            compute_segnet_constraint_loss,
        )
        seg_l = compute_segnet_constraint_loss(frames, masks, segnet)
        pose_l = compute_posenet_constraint_loss(frames, expected_pose, posenet)
        return torch.stack([seg_l.detach(), pose_l.detach()])


class BasinHopping(CrossDisciplinaryOptimizer):
    """Basin Hopping for frame optimization (Wales & Doye 1997).

    Reference: Wales & Doye, "Global Optimization by Basin-Hopping and the
    Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110
    Atoms", Journal of Physical Chemistry A 101(28):5111-5116, 1997.

    Why it helps: Basin hopping transforms the scorer loss landscape into a
    staircase by alternating random perturbation with local gradient descent.
    This separates basin-to-basin transitions (random jumps) from within-basin
    optimization (deterministic gradient descent), making each phase more
    efficient than either alone. The Metropolis criterion at the basin level
    enables systematic exploration of distinct local minima.

    Configurable: num_hops, local_steps, local_lr, perturbation_scale, temperature.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        num_hops = cfg.get("num_hops", 50)
        local_steps = cfg.get("local_steps", 20)
        local_lr = cfg.get("local_lr", 0.1)
        perturb_scale = cfg.get("perturbation_scale", 10.0)
        T = cfg.get("temperature", 10.0)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)
        num_steps = cfg.get("num_steps", num_hops)  # alias

        device = frames.device
        current = frames.clone()

        # Local minimization
        def local_minimize(x):
            for _ in range(local_steps):
                grad, _ = _compute_gradient(x, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
                x = _clamp_frames(x - local_lr * grad)
            return x

        current = local_minimize(current)
        current_loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)

        best = current.clone()
        best_loss = current_loss
        losses = []
        accepts = 0

        for hop in range(num_steps):
            # Random perturbation
            candidate = _clamp_frames(current + torch.randn_like(current) * perturb_scale)

            # Local minimization
            candidate = local_minimize(candidate)
            cand_loss = _scorer_loss_no_grad(candidate, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)

            # Metropolis at basin level
            delta = cand_loss - current_loss
            if delta < 0 or (T > 1e-12 and torch.rand(1).item() < math.exp(-delta / T)):
                current = candidate
                current_loss = cand_loss
                accepts += 1

            if current_loss < best_loss:
                best = current.clone()
                best_loss = current_loss

            losses.append(best_loss)

        self.diagnostics = {
            "final_loss": best_loss,
            "acceptance_rate": accepts / max(1, num_steps),
            "loss_history": losses,
        }
        return best


# ===========================================================================
# FROM GEOPHYSICS
# ===========================================================================


class FullWaveformInversion(CrossDisciplinaryOptimizer):
    """Full Waveform Inversion / Adjoint Method (Tarantola 1984).

    Reference: Tarantola, "Inversion of seismic reflection data in the
    acoustic approximation", Geophysics 49(8):1259-1266, 1984.

    Why it helps: Our problem IS an inverse problem — given scorer observations,
    find the model parameters (pixel values). FWI uses the adjoint-state method
    (automatic differentiation) for efficient gradient computation, combined with
    multi-scale optimization: start at 1/4 resolution (captures coarse SegNet
    semantics), refine to 1/2 (mid-frequency structure), then full resolution
    (fine PoseNet-sensitive details). Regularization prevents overfitting to
    scorer artifacts.

    Configurable: scales (resolution levels), steps_per_scale, lr, regularization
    (tikhonov_weight, tv_weight).
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        scales = cfg.get("scales", [0.25, 0.5, 1.0])
        steps_per_scale = cfg.get("steps_per_scale", 100)
        lr = cfg.get("lr", 0.1)
        tikhonov_weight = cfg.get("tikhonov_weight", 0.01)
        tv_weight = cfg.get("tv_weight", 0.1)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        num_steps = cfg.get("num_steps", steps_per_scale * len(scales))

        device = frames.device
        N, H, W, C = frames.shape
        current = frames.clone()
        best = current.clone()
        best_loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)

        steps_done = 0
        max_steps = num_steps
        losses = []

        for scale in scales:
            sH, sW = max(4, int(H * scale)), max(4, int(W * scale))

            # Downscale to current resolution
            working = F.interpolate(
                current.permute(0, 3, 1, 2),  # NCHW
                size=(sH, sW), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)  # NHWC

            scale_steps = min(steps_per_scale, max_steps - steps_done)
            for step in range(scale_steps):
                # Upscale working frames to full resolution for scorer eval
                full_res = F.interpolate(
                    working.permute(0, 3, 1, 2),
                    size=(H, W), mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                full_res = _clamp_frames(full_res)

                # Compute scorer gradient at full resolution
                grad_full, loss_val = _compute_gradient(
                    full_res, posenet, segnet, masks, expected_pose, seg_weight, pose_weight,
                )

                # Downscale gradient to working resolution
                grad_scaled = F.interpolate(
                    grad_full.permute(0, 3, 1, 2),
                    size=(sH, sW), mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)

                # Tikhonov regularization: L2 penalty on deviation from initial
                init_scaled = F.interpolate(
                    frames.permute(0, 3, 1, 2),
                    size=(sH, sW), mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                tikhonov_grad = tikhonov_weight * (working - init_scaled)

                # TV regularization: edge-preserving smoothness
                tv_grad = torch.zeros_like(working)
                if working.shape[1] > 1:
                    tv_grad[:, :-1, :, :] += torch.sign(working[:, 1:, :, :] - working[:, :-1, :, :]) * (-tv_weight)
                    tv_grad[:, 1:, :, :] += torch.sign(working[:, 1:, :, :] - working[:, :-1, :, :]) * tv_weight
                if working.shape[2] > 1:
                    tv_grad[:, :, :-1, :] += torch.sign(working[:, :, 1:, :] - working[:, :, :-1, :]) * (-tv_weight)
                    tv_grad[:, :, 1:, :] += torch.sign(working[:, :, 1:, :] - working[:, :, :-1, :]) * tv_weight

                working = working - lr * (grad_scaled + tikhonov_grad + tv_grad)
                working = working.clamp(0.0, 255.0)
                steps_done += 1

                # Track best at full resolution
                full_candidate = F.interpolate(
                    working.permute(0, 3, 1, 2),
                    size=(H, W), mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                full_candidate = _clamp_frames(full_candidate)
                loss = _scorer_loss_no_grad(full_candidate, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
                if loss < best_loss:
                    best = full_candidate.clone()
                    best_loss = loss
                losses.append(loss)

            # Upscale working result for next scale
            current = F.interpolate(
                working.permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
            current = _clamp_frames(current)

        self.diagnostics = {"final_loss": best_loss, "scales": scales, "loss_history": losses}
        return best


class SeismicMultiGrid(CrossDisciplinaryOptimizer):
    """Seismic Tomography Multi-Grid V-cycle (Virieux & Operto 2009).

    Reference: Virieux & Operto, "An overview of full-waveform inversion in
    exploration geophysics", Geophysics 74(6):WCC1-WCC26, 2009.

    Why it helps: V-cycle multigrid converges O(N) for smooth error components.
    Coarse-grid optimization captures SegNet-sensitive coarse semantics first,
    then fine-grid smoothing adds PoseNet-sensitive fine details. The restriction
    (fine→coarse) and prolongation (coarse→fine) operators match the scorer
    sensitivity hierarchy naturally.

    Resolution levels: 96x128 → 192x256 → 384x512 (V-cycle).

    Configurable: levels (resolution sequence), smooth_steps, num_cycles.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        levels = cfg.get("levels", [(96, 128), (192, 256), (384, 512)])
        smooth_steps = cfg.get("smooth_steps", 20)
        num_cycles = cfg.get("num_cycles", 5)
        lr = cfg.get("lr", 0.1)
        num_steps = cfg.get("num_steps", num_cycles * smooth_steps * len(levels))
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)

        device = frames.device
        N, H, W, C = frames.shape
        current = frames.clone()
        best = current.clone()
        best_loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
        losses = []

        def smooth(x, steps, target_h, target_w):
            """Gauss-Seidel-like smoothing at a given resolution."""
            working = F.interpolate(
                x.permute(0, 3, 1, 2), size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

            for _ in range(steps):
                full = F.interpolate(
                    working.permute(0, 3, 1, 2), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                full = _clamp_frames(full)

                grad_full, _ = _compute_gradient(full, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
                grad_scaled = F.interpolate(
                    grad_full.permute(0, 3, 1, 2), size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)

                working = (working - lr * grad_scaled).clamp(0.0, 255.0)

            return F.interpolate(
                working.permute(0, 3, 1, 2), size=(H, W),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        for cycle in range(num_cycles):
            # Down leg: coarse to fine
            for lH, lW in levels:
                current = _clamp_frames(smooth(current, smooth_steps, lH, lW))
                loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
                if loss < best_loss:
                    best = current.clone()
                    best_loss = loss
                losses.append(loss)

            # Up leg: fine to coarse (correction sweep)
            for lH, lW in reversed(levels[:-1]):
                current = _clamp_frames(smooth(current, smooth_steps // 2, lH, lW))
                loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
                if loss < best_loss:
                    best = current.clone()
                    best_loss = loss
                losses.append(loss)

        self.diagnostics = {"final_loss": best_loss, "num_cycles": num_cycles, "loss_history": losses}
        return best


# ===========================================================================
# FROM CLIMATE SCIENCE
# ===========================================================================


class EnsembleKalmanFilter(CrossDisciplinaryOptimizer):
    """Ensemble Kalman Filter for frame optimization (Evensen 1994).

    Reference: Evensen, "Sequential data assimilation with a nonlinear
    quasi-geostrophic model using Monte Carlo methods to forecast error
    statistics", Journal of Geophysical Research 99(C5):10143-10162, 1994.

    Why it helps: EnKF maintains an ensemble of candidate frames, each with
    implicit uncertainty (the ensemble spread). The prediction step evolves
    frames via gradient descent, while the update step assimilates scorer
    "observations" using the Kalman gain. This naturally balances model
    prediction (gradient information) with scorer feedback (actual loss
    values), handling uncertainty in the optimization landscape.

    Configurable: ensemble_size, observation_noise, prediction_steps, lr, num_steps.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        N_ens = cfg.get("ensemble_size", 16)
        obs_noise = cfg.get("observation_noise", 1.0)
        pred_steps = cfg.get("prediction_steps", 5)
        lr = cfg.get("lr", 0.05)
        num_steps = cfg.get("num_steps", 100)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device
        shape = frames.shape

        # Initialize ensemble
        ensemble = [_clamp_frames(frames + torch.randn_like(frames) * 5.0) for _ in range(N_ens)]
        ensemble[0] = frames.clone()

        best = frames.clone()
        best_loss = _scorer_loss_no_grad(frames, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
        losses = []

        for step in range(num_steps):
            # Prediction step: evolve each ensemble member via gradient descent
            for i in range(N_ens):
                for _ in range(pred_steps):
                    grad, _ = _compute_gradient(
                        ensemble[i], posenet, segnet, masks, expected_pose, seg_weight, pose_weight,
                    )
                    ensemble[i] = _clamp_frames(ensemble[i] - lr * grad)

            # Compute ensemble mean and perturbations
            ens_stack = torch.stack([e.reshape(-1) for e in ensemble], dim=0)  # (N_ens, D)
            ens_mean = ens_stack.mean(dim=0)  # (D,)
            ens_pert = ens_stack - ens_mean.unsqueeze(0)  # (N_ens, D)

            # Observation: scorer loss for each member (scalar per member)
            obs = torch.tensor(
                [_scorer_loss_no_grad(e, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight) for e in ensemble],
                device=device, dtype=frames.dtype,
            )  # (N_ens,)
            obs_mean = obs.mean()

            # "H" operator maps state to observation (scorer loss)
            # Hx_pert = obs - obs_mean for each member
            Hx_pert = obs - obs_mean  # (N_ens,)

            # Kalman gain in ensemble space
            # P_f*H^T = (1/(N-1)) * ens_pert^T * Hx_pert
            PfHT = (ens_pert.T @ Hx_pert) / max(N_ens - 1, 1)  # (D,)
            # HPfH^T + R = (1/(N-1)) * Hx_pert^T * Hx_pert + R
            HPfHT_R = (Hx_pert @ Hx_pert) / max(N_ens - 1, 1) + obs_noise ** 2

            # Kalman gain K = PfHT / HPfHT_R (scalar observation)
            K = PfHT / max(HPfHT_R.item(), 1e-8)  # (D,)

            # Update: push each member toward lower-loss regions
            target_obs = obs.min()  # target: best observed loss
            for i in range(N_ens):
                innovation = target_obs - obs[i] + torch.randn(1, device=device).item() * obs_noise
                ensemble[i] = _clamp_frames(
                    (ens_stack[i] + K * innovation).reshape(shape)
                )

            # Track best
            for i in range(N_ens):
                loss = _scorer_loss_no_grad(ensemble[i], posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
                if loss < best_loss:
                    best = ensemble[i].clone()
                    best_loss = loss
            losses.append(best_loss)

        self.diagnostics = {"final_loss": best_loss, "ensemble_size": N_ens, "loss_history": losses}
        return best


class FourDVar(CrossDisciplinaryOptimizer):
    """4D-Var Data Assimilation (Le Dimet & Talagrand 1986).

    Reference: Le Dimet & Talagrand, "Variational algorithms for analysis and
    assimilation of meteorological observations: theoretical aspects", Tellus A
    38(2):97-110, 1986.

    Why it helps: 4D-Var optimizes ALL frames simultaneously over the temporal
    window, enforcing temporal coherence as a hard constraint rather than a soft
    penalty. The adjoint model (backprop) computes the gradient w.r.t. all frames
    in one backward pass. This is critical for PoseNet which evaluates frame
    PAIRS — optimizing frames independently can create temporal discontinuities
    that PoseNet penalizes heavily.

    Configurable: temporal_weight, lr, num_steps, temporal_order (1 or 2).
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        temporal_weight = cfg.get("temporal_weight", 10.0)
        lr_val = cfg.get("lr", 0.1)
        num_steps = cfg.get("num_steps", 500)
        temporal_order = cfg.get("temporal_order", 1)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device
        current = frames.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([current], lr=lr_val)

        best = frames.clone()
        best_loss = _scorer_loss_no_grad(frames, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
        losses = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Scorer loss over all frames
            scorer_l = _scorer_loss_fn(
                current.clamp(0.0, 255.0), posenet, segnet, masks, expected_pose,
                seg_weight, pose_weight, compress_weight,
            )

            # Temporal coherence constraint
            if current.shape[0] > 1:
                # First-order: L2 between consecutive frames
                temporal_l1 = ((current[1:] - current[:-1]) ** 2).mean()
                temporal_l = temporal_weight * temporal_l1

                if temporal_order >= 2 and current.shape[0] > 2:
                    # Second-order: penalize acceleration (jerk)
                    accel = current[2:] - 2 * current[1:-1] + current[:-2]
                    temporal_l2 = (accel ** 2).mean()
                    temporal_l = temporal_l + temporal_weight * 0.5 * temporal_l2
            else:
                temporal_l = torch.tensor(0.0, device=device)

            total_loss = scorer_l + temporal_l
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                current.clamp_(0.0, 255.0)
                loss = _scorer_loss_no_grad(current.data, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
                if loss < best_loss:
                    best = current.data.clone()
                    best_loss = loss
                losses.append(loss)

        self.diagnostics = {"final_loss": best_loss, "loss_history": losses}
        return best


# ===========================================================================
# FROM ASTROPHYSICS / COSMOLOGY
# ===========================================================================


class NestedSampling(CrossDisciplinaryOptimizer):
    """Nested Sampling for frame optimization (Skilling 2004).

    Reference: Skilling, "Nested Sampling for General Bayesian Computation",
    Bayesian Analysis 1(4):833-860, 2006. (conference version 2004.)

    Why it helps: Nested sampling maintains N "live points" in frame space,
    each with likelihood L = exp(-scorer_loss). It systematically contracts the
    prior volume by replacing the worst point with a new sample constrained to
    have higher likelihood. This naturally estimates the volume of good solutions
    (the evidence), telling us how much freedom we have in frame space. It also
    provides the best solution found as a byproduct.

    Configurable: num_live, num_iterations, likelihood_scale.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        num_live = cfg.get("num_live", 16)
        num_iter = cfg.get("num_steps", 200)
        lik_scale = cfg.get("likelihood_scale", 1.0)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device

        # Initialize live points
        live_points = [_clamp_frames(frames + torch.randn_like(frames) * 10.0) for _ in range(num_live)]
        live_points[0] = frames.clone()
        live_losses = [
            _scorer_loss_no_grad(p, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
            for p in live_points
        ]

        best = frames.clone()
        best_loss = min(live_losses)
        best_idx = live_losses.index(best_loss)
        best = live_points[best_idx].clone()

        log_evidence = -float("inf")
        dead_points_log_weights = []
        losses = []

        for it in range(num_iter):
            # Find worst live point
            worst_idx = max(range(num_live), key=lambda i: live_losses[i])
            worst_loss = live_losses[worst_idx]
            L_min = worst_loss  # threshold: new point must be better than this

            # Log-weight of the dead point
            # log(prior_volume_shrinkage) ~ -it/num_live
            log_weight = -it / num_live - worst_loss * lik_scale
            dead_points_log_weights.append(log_weight)

            # Replace worst: sample from constrained prior (L > L_min)
            # Use MCMC walk from a random live point
            donor_idx = torch.randint(num_live, (1,)).item()
            while donor_idx == worst_idx:
                donor_idx = torch.randint(num_live, (1,)).item()

            new_point = live_points[donor_idx].clone()
            for _ in range(20):  # MCMC steps to move away from donor
                candidate = _clamp_frames(new_point + torch.randn_like(new_point) * 5.0)
                cand_loss = _scorer_loss_no_grad(candidate, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
                if cand_loss < L_min:
                    new_point = candidate
                    break  # Accept first valid point

            new_loss = _scorer_loss_no_grad(new_point, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
            live_points[worst_idx] = new_point
            live_losses[worst_idx] = new_loss

            if new_loss < best_loss:
                best = new_point.clone()
                best_loss = new_loss

            losses.append(best_loss)

        # Estimate log evidence
        if dead_points_log_weights:
            max_lw = max(dead_points_log_weights)
            log_evidence = max_lw + math.log(sum(math.exp(lw - max_lw) for lw in dead_points_log_weights))

        self.diagnostics = {
            "final_loss": best_loss,
            "log_evidence": log_evidence,
            "loss_history": losses,
        }
        return best


class MultigridRelaxation(CrossDisciplinaryOptimizer):
    """Multigrid Relaxation (Brandt 1977, used in cosmological N-body).

    Reference: Brandt, "Multi-Level Adaptive Solutions to Boundary-Value
    Problems", Mathematics of Computation 31(138):333-390, 1977.

    Why it helps: Applies Jacobi/Gauss-Seidel-style relaxation on the scorer
    gradient field with red-black ordering for parallel updates. The multigrid
    V-cycle achieves O(N) convergence for smooth error components — smooth
    errors (coarse SegNet semantics) are damped on coarse grids while
    oscillatory errors (fine PoseNet details) are damped on fine grids.

    Configurable: num_v_cycles, pre_smooth, post_smooth, coarsest_size, lr.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        num_v_cycles = cfg.get("num_v_cycles", 5)
        pre_smooth = cfg.get("pre_smooth", 5)
        post_smooth = cfg.get("post_smooth", 5)
        coarsest_h = cfg.get("coarsest_h", 48)
        coarsest_w = cfg.get("coarsest_w", 64)
        lr = cfg.get("lr", 0.1)
        num_steps = cfg.get("num_steps", num_v_cycles * (pre_smooth + post_smooth) * 4)
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)

        device = frames.device
        N, H, W, C = frames.shape
        current = frames.clone()
        best = current.clone()
        best_loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
        losses = []

        # Build resolution hierarchy
        levels = []
        h, w = H, W
        while h >= coarsest_h and w >= coarsest_w:
            levels.append((h, w))
            h, w = h // 2, w // 2
        if not levels:
            levels = [(H, W)]

        def relax(x, steps, res_h, res_w):
            """Red-black Gauss-Seidel relaxation at given resolution."""
            working = F.interpolate(
                x.permute(0, 3, 1, 2), size=(res_h, res_w),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

            for s in range(steps):
                full = F.interpolate(
                    working.permute(0, 3, 1, 2), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                full = _clamp_frames(full)

                grad_full, _ = _compute_gradient(full, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
                grad_res = F.interpolate(
                    grad_full.permute(0, 3, 1, 2), size=(res_h, res_w),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)

                # Red-black ordering: update even pixels first, then odd
                rb_mask = torch.zeros(1, res_h, res_w, 1, device=device, dtype=frames.dtype)
                for i in range(res_h):
                    for j in range(res_w):
                        if (i + j) % 2 == s % 2:
                            rb_mask[0, i, j, 0] = 1.0

                working = (working - lr * grad_res * rb_mask).clamp(0.0, 255.0)

            return F.interpolate(
                working.permute(0, 3, 1, 2), size=(H, W),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)

        for cycle in range(num_v_cycles):
            # Down leg: pre-smoothing at each level
            for lH, lW in levels:
                current = _clamp_frames(relax(current, pre_smooth, lH, lW))

            # Up leg: post-smoothing at each level (reversed)
            for lH, lW in reversed(levels):
                current = _clamp_frames(relax(current, post_smooth, lH, lW))

            loss = _scorer_loss_no_grad(current, posenet, segnet, masks, expected_pose, seg_weight, pose_weight)
            if loss < best_loss:
                best = current.clone()
                best_loss = loss
            losses.append(loss)

        self.diagnostics = {"final_loss": best_loss, "num_v_cycles": num_v_cycles, "loss_history": losses}
        return best


# ===========================================================================
# FROM QUANTUM COMPUTING (classical simulation)
# ===========================================================================


class QuantumAnnealingSimulation(CrossDisciplinaryOptimizer):
    """Quantum Annealing Simulation (Kadowaki & Nishimori 1998).

    Reference: Kadowaki & Nishimori, "Quantum annealing in the transverse
    Ising model", Physical Review E 58(5):5355, 1998.

    Why it helps: Simulates quantum tunneling through energy barriers in the
    scorer loss landscape. Classical SA can only go over barriers (requiring high
    temperature), but quantum tunneling allows crossing through barriers with
    probability that depends on barrier width rather than height. This is
    approximated by adding a "tunneling" term: occasional discrete jumps to
    distant points in a low-energy direction, with probability decreasing as
    the transverse field Gamma is annealed to zero.

    Configurable: Gamma_start, Gamma_end, num_steps, tunneling_scale, lr.
    """

    def optimize(self, frames, posenet, segnet, masks, expected_pose, **kwargs):
        cfg = self.config
        Gamma_start = cfg.get("Gamma_start", 50.0)
        Gamma_end = cfg.get("Gamma_end", 0.1)
        num_steps = cfg.get("num_steps", 500)
        tunneling_scale = cfg.get("tunneling_scale", 20.0)
        lr = cfg.get("lr", 0.1)
        n_paths = cfg.get("n_paths", 4)  # Suzuki-Trotter slices
        seg_weight = cfg.get("seg_weight", 100.0)
        pose_weight = cfg.get("pose_weight", 10.0)
        compress_weight = cfg.get("compress_weight", 1.0)

        device = frames.device

        # Initialize Trotter slices (replicas representing quantum state)
        paths = [_clamp_frames(frames + torch.randn_like(frames) * 3.0) for _ in range(n_paths)]
        paths[0] = frames.clone()

        best = frames.clone()
        best_loss = _scorer_loss_no_grad(frames, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
        losses = []

        for step in range(num_steps):
            # Anneal transverse field
            t_frac = step / max(1, num_steps - 1)
            Gamma = Gamma_start * ((Gamma_end / Gamma_start) ** t_frac)

            for k in range(n_paths):
                # Classical potential gradient
                grad, loss_val = _compute_gradient(
                    paths[k], posenet, segnet, masks, expected_pose, seg_weight, pose_weight,
                )

                # Coupling between adjacent Trotter slices (quantum kinetic term)
                coupling = torch.zeros_like(paths[k])
                if n_paths > 1:
                    k_prev = (k - 1) % n_paths
                    k_next = (k + 1) % n_paths
                    coupling = (paths[k_prev] + paths[k_next] - 2.0 * paths[k])

                # Quantum tunneling: stochastic discrete jump
                tunnel_prob = Gamma / (Gamma + 1.0)
                if torch.rand(1).item() < tunnel_prob:
                    # Jump along negative gradient direction with large step
                    jump = -grad / (grad.norm() + 1e-8) * tunneling_scale * Gamma / Gamma_start
                    paths[k] = _clamp_frames(paths[k] + jump)
                else:
                    # Standard gradient update + inter-slice coupling
                    coupling_strength = Gamma * 0.1
                    paths[k] = _clamp_frames(
                        paths[k] - lr * grad + coupling_strength * coupling
                    )

            # Track best across all paths
            for k in range(n_paths):
                loss = _scorer_loss_no_grad(paths[k], posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
                if loss < best_loss:
                    best = paths[k].clone()
                    best_loss = loss
            losses.append(best_loss)

        self.diagnostics = {"final_loss": best_loss, "final_Gamma": Gamma, "loss_history": losses}
        return best


# ===========================================================================
# Factory and ensemble
# ===========================================================================

_OPTIMIZER_REGISTRY: dict[str, type[CrossDisciplinaryOptimizer]] = {
    # Physics
    "simulated_annealing": SimulatedAnnealing,
    "hmc": HamiltonianMonteCarlo,
    "langevin": LangevinDynamics,
    "replica_exchange": ReplicaExchange,
    # Biology
    "cma_es": CMAES,
    "differential_evolution": DifferentialEvolution,
    "pso": ParticleSwarmOptimization,
    # Chemistry
    "metadynamics": Metadynamics,
    "basin_hopping": BasinHopping,
    # Geophysics
    "fwi": FullWaveformInversion,
    "seismic_multigrid": SeismicMultiGrid,
    # Climate science
    "enkf": EnsembleKalmanFilter,
    "4dvar": FourDVar,
    # Astrophysics
    "nested_sampling": NestedSampling,
    "multigrid_relaxation": MultigridRelaxation,
    # Quantum
    "quantum_annealing": QuantumAnnealingSimulation,
}


def optimizer_factory(name: str, config: dict[str, Any] | None = None) -> CrossDisciplinaryOptimizer:
    """Create an optimizer by name.

    Args:
        name: One of the registered optimizer names (see _OPTIMIZER_REGISTRY).
        config: Configuration dict passed to the optimizer constructor.

    Returns:
        Instantiated optimizer.

    Raises:
        KeyError: If name is not in the registry.
    """
    if name not in _OPTIMIZER_REGISTRY:
        available = ", ".join(sorted(_OPTIMIZER_REGISTRY.keys()))
        raise KeyError(f"Unknown optimizer {name!r}. Available: {available}")
    return _OPTIMIZER_REGISTRY[name](config or {})


def ensemble_optimize(
    frames: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    masks: torch.Tensor,
    expected_pose: torch.Tensor,
    optimizers: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run multiple optimizers and return the best result.

    Each optimizer runs independently from the same starting point. The result
    with the lowest scorer loss wins.

    Args:
        frames: (N, H, W, 3) float tensor in [0, 255].
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        masks: (N, H_seg, W_seg) long tensor of target class indices.
        expected_pose: (P, 6) float tensor where P = N-1.
        optimizers: list of optimizer names to run. Defaults to all.
        config: shared configuration dict (individual optimizer configs can
            be nested under the optimizer name key).

    Returns:
        (N, H, W, 3) best optimized frames.
    """
    config = config or {}
    if optimizers is None:
        optimizers = list(_OPTIMIZER_REGISTRY.keys())

    best_frames = frames.clone()
    best_loss = float("inf")
    results: dict[str, dict] = {}

    seg_weight = config.get("seg_weight", 100.0)
    pose_weight = config.get("pose_weight", 10.0)
    compress_weight = config.get("compress_weight", 1.0)

    for name in optimizers:
        # Allow per-optimizer config override
        opt_config = {**config}
        if name in config:
            opt_config.update(config[name])

        opt = optimizer_factory(name, opt_config)
        try:
            result = opt.optimize(frames, posenet, segnet, masks, expected_pose)
            loss = _scorer_loss_no_grad(result, posenet, segnet, masks, expected_pose, seg_weight, pose_weight, compress_weight)
            results[name] = {"loss": loss, **opt.get_diagnostics()}

            if loss < best_loss:
                best_frames = result
                best_loss = loss
        except Exception as e:
            results[name] = {"error": str(e)}

    return best_frames


# ===========================================================================
# Smoke test
# ===========================================================================

def _smoke_test():
    """Run each optimizer for 10 steps on a tiny 2x32x32 test case.

    Uses mock scorer models that return differentiable dummy outputs.
    This verifies that all optimizers can:
    1. Run without errors
    2. Accept the correct input format
    3. Return valid output tensors
    4. Reduce loss (at least not crash)
    """
    import sys

    print("Cross-disciplinary optimizers smoke test")
    print("=" * 60)

    device = "cpu"
    N, H, W, C = 2, 32, 32, 3

    # Mock scorer models
    class MockPoseNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(6, 16, 3, padding=1)
            self.fc = torch.nn.Linear(16, 6)

        def preprocess_input(self, x):
            # x: (B, T, C, H, W) -> take mean across T, resize to small
            B, T, Ci, Hi, Wi = x.shape
            merged = x.mean(dim=1)  # (B, C, H, W)
            # Simple: tile to 6 channels
            if Ci == 3:
                merged = torch.cat([merged, merged], dim=1)  # (B, 6, H, W)
            return merged

        def forward(self, x):
            # x: (B, 6, H, W) -> pose (B, 6)
            feat = F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).squeeze(-1).squeeze(-1)
            pose = self.fc(feat)
            return {"pose": pose}

    class MockSegNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 5, 3, padding=1)  # 5 classes

        def preprocess_input(self, x):
            B, T, Ci, Hi, Wi = x.shape
            return x[:, -1, :, :, :]  # last frame: (B, C, H, W)

        def forward(self, x):
            return self.conv(x)  # (B, 5, H, W)

    posenet = MockPoseNet().eval()
    segnet = MockSegNet().eval()

    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)

    frames = torch.rand(N, H, W, C, device=device) * 255.0
    masks = torch.randint(0, 5, (N, H, W), device=device)
    expected_pose = torch.randn(N - 1, 6, device=device) * 0.1

    passed = 0
    failed = 0

    for name in sorted(_OPTIMIZER_REGISTRY.keys()):
        smoke_config = {
            "num_steps": 10,
            "num_hops": 10,
            "num_v_cycles": 2,
            "pop_size": 4,
            "num_particles": 4,
            "num_live": 4,
            "num_replicas": 2,
            "ensemble_size": 4,
            "n_components": 4,
            "n_paths": 2,
            "steps_per_scale": 3,
            "scales": [0.5, 1.0],
            "levels": [(16, 16), (32, 32)],
            "smooth_steps": 3,
            "pre_smooth": 2,
            "post_smooth": 2,
            "coarsest_h": 8,
            "coarsest_w": 8,
            "local_steps": 3,
            "prediction_steps": 2,
            "num_generations": 5,
        }

        try:
            opt = optimizer_factory(name, smoke_config)
            result = opt.optimize(frames, posenet, segnet, masks, expected_pose)

            assert result.shape == frames.shape, f"Shape mismatch: {result.shape} vs {frames.shape}"
            assert result.min() >= 0.0, f"Negative pixel values: {result.min()}"
            assert result.max() <= 255.0, f"Pixel values > 255: {result.max()}"
            assert not torch.isnan(result).any(), "NaN in result"

            diag = opt.get_diagnostics()
            final_loss = diag.get("final_loss", "N/A")
            print(f"  PASS  {name:30s}  loss={final_loss}")
            passed += 1

        except Exception as e:
            print(f"  FAIL  {name:30s}  {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    success = _smoke_test()
    import sys
    sys.exit(0 if success else 1)
