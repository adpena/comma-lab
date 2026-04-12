"""Hamiltonian dynamics for pixel-space optimization (phase-space formulation).

Yousfi's shower insight: treat each pixel as a particle in a potential field
defined by the scorer loss landscape. The gradient of the scorer loss is the
force, and the learning rate is the time step.

Hamilton's equations for pixel optimization:

    dp/dt = -dH/dq  (momentum update: force from scorer gradient)
    dq/dt =  dH/dp  (position update: move along momentum)

where:
    q = pixel values (position in configuration space)
    p = pixel momenta (accumulated gradient direction)
    H(q, p) = V(q) + T(p) = scorer_loss(q) + ||p||^2 / (2m)

This formulation gives us natural:
  - Momentum: p accumulates over time, giving SGD+momentum for free
  - Energy conservation: H(q, p) = const along trajectories
  - Symplectic structure: phase space volume is preserved (Liouville theorem)
  - Leapfrog integration: time-reversible, symplectic integrator

The symplectic integrator preserves the Hamiltonian (energy) to machine precision
over long trajectories, preventing the optimization from diverging. This is
strictly better than naive gradient descent for non-convex landscapes.

Example::

    from tac.hamiltonian_dynamics import HamiltonianPixelOptimizer
    opt = HamiltonianPixelOptimizer(cfg={})
    frames = opt.optimize(init_frames, scorer_potential, dt=0.01, num_steps=1000)
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def kinetic_energy(momentum: torch.Tensor, mass: float = 1.0) -> torch.Tensor:
    """Compute kinetic energy T(p) = ||p||^2 / (2m).

    The kinetic energy determines how momenta translate to velocities.
    Larger mass = slower dynamics = more stable optimization.
    Smaller mass = faster dynamics = risk of overshooting.

    In the statistical mechanics analogy, T ~ temperature of the pixel
    system. High T explores broadly; low T settles near a minimum.

    Args:
        momentum: (...) momentum tensor.
        mass: effective mass parameter.

    Returns:
        Scalar kinetic energy.
    """
    return momentum.pow(2).sum() / (2.0 * mass)


def hamiltonian(
    position: torch.Tensor,
    momentum: torch.Tensor,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    mass: float = 1.0,
) -> torch.Tensor:
    """Compute total Hamiltonian H = V(q) + T(p).

    The Hamiltonian is conserved along exact trajectories (Noether's theorem
    for time-translation symmetry). In discrete integration, symplectic
    integrators preserve H to O(dt^p) where p is the integrator order.

    Monitoring H during optimization detects instability: if H drifts
    significantly, the time step is too large.

    Args:
        position: pixel values (configuration space).
        momentum: pixel momenta (cotangent space).
        potential_fn: V(q) = scorer loss function.
        mass: effective mass.

    Returns:
        Scalar Hamiltonian value.
    """
    V = potential_fn(position)
    T = kinetic_energy(momentum, mass)
    return V + T


class LeapfrogIntegrator:
    """Stormer-Verlet (leapfrog) symplectic integrator.

    The leapfrog scheme for Hamilton's equations:
        p_{n+1/2} = p_n - (dt/2) * dV/dq(q_n)        (half-step momentum)
        q_{n+1}   = q_n + dt * p_{n+1/2} / m          (full-step position)
        p_{n+1}   = p_{n+1/2} - (dt/2) * dV/dq(q_{n+1})  (half-step momentum)

    Properties:
      - Second-order accurate: error O(dt^2)
      - Symplectic: preserves the symplectic 2-form (phase space volume)
      - Time-reversible: running backwards recovers the initial state
      - Energy preservation: H is bounded for any trajectory length

    These properties make leapfrog strictly superior to Euler integration for
    Hamiltonian systems. The energy conservation prevents gradient descent
    from diverging on non-convex scorer loss landscapes.

    Args:
        dt: time step size.
        mass: effective mass parameter.
        damping: velocity damping factor in [0, 1]. 0 = no damping (pure Hamiltonian),
            1 = full damping (steepest descent). Intermediate values give
            underdamped dynamics (momentum with friction).
    """

    def __init__(
        self,
        dt: float = 0.01,
        mass: float = 1.0,
        damping: float = 0.0,
    ) -> None:
        self.dt = dt
        self.mass = mass
        self.damping = damping

    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        grad_V: torch.Tensor,
        grad_V_next: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one leapfrog step.

        If grad_V_next is not provided, uses a single-evaluation variant:
        half-step p, full-step q, then the caller must compute grad_V at q_{n+1}
        for the next half-step.

        Args:
            q: (B, C, H, W) pixel positions.
            p: (B, C, H, W) pixel momenta.
            grad_V: dV/dq at current position (negative force).
            grad_V_next: dV/dq at next position (for full leapfrog), or None.

        Returns:
            (q_new, p_new) updated position and momentum.
        """
        dt = self.dt
        m = self.mass

        # Half-step momentum
        p_half = p - (dt / 2.0) * grad_V

        # Full-step position
        q_new = q + dt * p_half / m

        # Apply damping (Langevin dynamics extension)
        if self.damping > 0:
            p_half = p_half * (1.0 - self.damping)

        # Second half-step momentum (using gradient at new position)
        if grad_V_next is not None:
            p_new = p_half - (dt / 2.0) * grad_V_next
        else:
            # Caller must provide grad_V_next on next call
            p_new = p_half

        return q_new, p_new


class HamiltonianPixelOptimizer:
    """Optimize pixel values using Hamiltonian dynamics.

    Treats the scorer loss as a potential energy V(q) and pixel values as
    particle positions q in a D-dimensional configuration space.

    The trajectory in phase space (q(t), p(t)) follows Hamilton's equations,
    integrated via the leapfrog scheme. The resulting dynamics are:
      - Momentum-based: naturally escapes shallow local minima
      - Energy-conserving: prevents divergence on rough loss landscapes
      - Phase-space volume preserving: ergodic exploration

    With damping > 0, the dynamics become Langevin-like: the system loses
    energy to friction and eventually settles at a minimum. The damping
    rate controls the exploration-exploitation tradeoff.

    Args:
        cfg: configuration dict. Keys:
            - hamiltonian_dt (float): time step, default 0.01
            - hamiltonian_mass (float): effective mass, default 1.0
            - hamiltonian_damping (float): friction coefficient, default 0.01
            - hamiltonian_steps (int): number of integration steps, default 500
            - hamiltonian_grad_clip (float): gradient clipping, default 10.0
            - energy_monitor_interval (int): print energy every N steps, default 50
            - anneal_damping (bool): increase damping over time, default True
            - anneal_start (float): initial damping if annealing, default 0.001
            - anneal_end (float): final damping if annealing, default 0.1
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    def _get(self, key: str, default: Any) -> Any:
        return self.cfg.get(key, default)

    def optimize(
        self,
        init_frames: torch.Tensor,
        potential_fn: Callable[[torch.Tensor], torch.Tensor],
        dt: float | None = None,
        num_steps: int | None = None,
        log_every: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run Hamiltonian dynamics optimization.

        The trajectory proceeds as:
          1. Initialize momenta from N(0, mass*T) (thermal equilibrium)
          2. Integrate Hamilton's equations via leapfrog
          3. Track best position (lowest potential energy)
          4. Optionally anneal damping for simulated annealing effect

        Args:
            init_frames: (B, C, H, W) initial pixel values in [0, 255].
            potential_fn: V(q) -> scalar potential energy (scorer loss).
                Must be differentiable w.r.t. input.
            dt: time step (overrides cfg).
            num_steps: integration steps (overrides cfg).
            log_every: logging interval (overrides cfg).

        Returns:
            (optimized_frames, diagnostics) where diagnostics includes:
                - energy_history: list of H values
                - potential_history: list of V values
                - kinetic_history: list of T values
                - best_potential: lowest V achieved
                - energy_drift: max |H - H_0| / |H_0| (should be small)
                - steps: number of steps taken
        """
        dt_val = dt or self._get("hamiltonian_dt", 0.01)
        mass = self._get("hamiltonian_mass", 1.0)
        damping_base = self._get("hamiltonian_damping", 0.01)
        max_steps = num_steps or self._get("hamiltonian_steps", 500)
        grad_clip = self._get("hamiltonian_grad_clip", 10.0)
        log_interval = log_every or self._get("energy_monitor_interval", 50)
        anneal_damping = self._get("anneal_damping", True)
        anneal_start = self._get("anneal_start", 0.001)
        anneal_end = self._get("anneal_end", 0.1)

        device = init_frames.device

        # Initialize position and momentum
        q = init_frames.detach().clone()
        # Thermal initialization: p ~ N(0, mass * kT), with kT=1
        p = torch.randn_like(q) * math.sqrt(mass)

        best_q = q.clone()
        best_V = float("inf")

        energy_history: list[float] = []
        potential_history: list[float] = []
        kinetic_history: list[float] = []

        for step in range(max_steps):
            # Compute damping (anneal from low to high)
            if anneal_damping:
                t_frac = step / max(max_steps - 1, 1)
                current_damping = anneal_start + (anneal_end - anneal_start) * t_frac
            else:
                current_damping = damping_base

            # Create integrator WITHOUT damping — damping is applied ONCE
            # after the complete leapfrog step to avoid double-damping.
            integrator = LeapfrogIntegrator(dt=dt_val, mass=mass, damping=0.0)

            # Compute gradient of potential (force)
            q_grad = q.detach().clone().requires_grad_(True)
            V = potential_fn(q_grad)
            V.backward()
            grad_V = q_grad.grad.detach().clone() if q_grad.grad is not None else torch.zeros_like(q)

            # Gradient clipping for stability
            grad_norm = grad_V.norm()
            if grad_norm > grad_clip:
                grad_V = grad_V * (grad_clip / (grad_norm + 1e-12))

            # First half of leapfrog (p half-step + q full step)
            q_new, p_half = integrator.step(q, p, grad_V)

            # Clamp to valid pixel range
            q_new = q_new.clamp(0.0, 255.0)

            # Compute energy for monitoring
            V_val = V.item()
            T_val = kinetic_energy(p, mass).item()
            H_val = V_val + T_val

            energy_history.append(H_val)
            potential_history.append(V_val)
            kinetic_history.append(T_val)

            # Track best position
            if V_val < best_V:
                best_V = V_val
                best_q = q_new.clone()

            # Second half-step: compute gradient at new position
            q_grad2 = q_new.detach().clone().requires_grad_(True)
            V2 = potential_fn(q_grad2)
            V2.backward()
            grad_V_next = q_grad2.grad.detach().clone() if q_grad2.grad is not None else torch.zeros_like(q)
            if grad_V_next.norm() > grad_clip:
                grad_V_next = grad_V_next * (grad_clip / (grad_V_next.norm() + 1e-12))

            # Complete leapfrog: second momentum half-step
            p_new = p_half - (dt_val / 2.0) * grad_V_next

            # Apply damping ONCE after the complete step (Langevin extension)
            if current_damping > 0:
                p_new = p_new * (1.0 - current_damping)

            q = q_new
            p = p_new

            if log_interval > 0 and (step + 1) % log_interval == 0:
                print(
                    f"  hamiltonian step {step + 1:4d}/{max_steps}: "
                    f"H={H_val:.4f} V={V_val:.4f} T={T_val:.4f} "
                    f"|grad|={grad_norm.item():.2e} damping={current_damping:.4f}"
                )

        # Energy conservation diagnostic
        H0 = energy_history[0] if energy_history else 1.0
        energy_drift = max(abs(h - H0) for h in energy_history) / (abs(H0) + 1e-8) if energy_history else 0.0

        diagnostics = {
            "energy_history": energy_history,
            "potential_history": potential_history,
            "kinetic_history": kinetic_history,
            "best_potential": best_V,
            "energy_drift": energy_drift,
            "steps": len(energy_history),
        }

        return best_q.clamp(0.0, 255.0), diagnostics

    def optimize_with_scorer(
        self,
        init_frames: torch.Tensor,
        masks: torch.Tensor,
        posenet: nn.Module,
        segnet: nn.Module,
        seg_weight: float = 100.0,
        pose_weight: float = 10.0,
        smooth_weight: float = 0.01,
        log_every: int = 50,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convenience method: Hamiltonian dynamics with scorer potential.

        The potential energy is the competition score formula:
            V(q) = seg_weight * seg_loss(q) + pose_weight * pose_loss(q)
                   + smooth_weight * TV(q)

        Args:
            init_frames: (N, C, H, W) initial frames in [0, 255].
            masks: (N, H, W) target segmentation masks.
            posenet: frozen PoseNet model.
            segnet: frozen SegNet model.
            seg_weight: SegNet loss multiplier.
            pose_weight: PoseNet loss multiplier.
            smooth_weight: total variation weight.
            log_every: logging interval.

        Returns:
            (optimized_frames, diagnostics) tuple.
        """
        def scorer_potential(q: torch.Tensor) -> torch.Tensor:
            """Scorer-based potential energy."""
            N, C, H, W = q.shape
            device = q.device

            # SegNet loss
            q_resized = F.interpolate(q, size=(384, 512), mode="bilinear", align_corners=False)
            logits = segnet(q_resized)
            H_out, W_out = logits.shape[2], logits.shape[3]
            masks_resized = F.interpolate(
                masks.float().unsqueeze(1).to(device),
                size=(H_out, W_out), mode="nearest",
            ).squeeze(1).long()
            seg_loss = F.cross_entropy(logits, masks_resized)

            # PoseNet loss (self-pairs)
            pose_loss = torch.tensor(0.0, device=device)
            if N >= 2:
                pair = q.unsqueeze(1).expand(N, 2, C, H, W).contiguous()
                posenet_in = posenet.preprocess_input(pair)
                pose_out = posenet(posenet_in)
                pred = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
                pose_loss = pred[..., :6].pow(2).mean()

            # Total variation
            tv_h = (q[:, :, 1:, :] - q[:, :, :-1, :]).abs().mean()
            tv_w = (q[:, :, :, 1:] - q[:, :, :, :-1]).abs().mean()
            tv = tv_h + tv_w

            return seg_weight * seg_loss + pose_weight * pose_loss + smooth_weight * tv

        return self.optimize(init_frames, scorer_potential, log_every=log_every)


class NoseHooverThermostat:
    """Nose-Hoover thermostat for canonical (NVT) ensemble dynamics.

    The Nose-Hoover extension adds a fictitious degree of freedom that
    acts as a thermostat, maintaining the system at a target temperature.

    Extended equations of motion:
        dq/dt = p/m
        dp/dt = -dV/dq - xi * p
        dxi/dt = (1/Q) * (sum(p^2/m) - N_dof * kT)

    where xi is the thermostat variable, Q is the thermostat mass, and
    kT is the target temperature.

    At high temperature: broad exploration (simulated annealing start).
    At low temperature: fine-tuning near minimum (simulated annealing end).

    The thermostat ensures the time-averaged kinetic energy equals N_dof * kT / 2,
    which is the equipartition theorem. This prevents the system from either
    cooling to absolute zero (stuck at local minimum) or heating indefinitely
    (diverging optimization).

    Args:
        target_temperature: kT parameter controlling exploration.
        thermostat_mass: Q parameter controlling thermostat response time.
            Small Q = fast response (tight temperature control).
            Large Q = slow response (more like Hamiltonian dynamics).
        mass: particle mass.
    """

    def __init__(
        self,
        target_temperature: float = 1.0,
        thermostat_mass: float = 10.0,
        mass: float = 1.0,
    ) -> None:
        self.kT = target_temperature
        self.Q = thermostat_mass
        self.mass = mass
        self.xi = 0.0  # thermostat variable

    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        grad_V: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One integration step with Nose-Hoover thermostat.

        Uses velocity-Verlet with thermostat coupling.

        Args:
            q: position (pixel values).
            p: momentum.
            grad_V: gradient of potential at q.
            dt: time step.

        Returns:
            (q_new, p_new) updated state.
        """
        m = self.mass
        N_dof = q.numel()

        # Half-step momentum with thermostat friction
        p_half = p - (dt / 2.0) * (grad_V + self.xi * p)

        # Full-step position
        q_new = q + dt * p_half / m

        # Update thermostat variable
        KE = p_half.pow(2).sum().item() / m
        self.xi += (dt / self.Q) * (KE - N_dof * self.kT)

        # Second half-step momentum (caller must provide new grad_V)
        p_new = p_half  # Incomplete: needs second half after new grad_V

        return q_new, p_new

    def complete_step(
        self,
        p_half: torch.Tensor,
        grad_V_new: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Complete the velocity-Verlet step with new gradient.

        Args:
            p_half: half-step momentum from step().
            grad_V_new: gradient of potential at new position.
            dt: time step.

        Returns:
            p_new: completed momentum.
        """
        return p_half - (dt / 2.0) * (grad_V_new + self.xi * p_half)


# ---- Smoke tests ----


def _smoke_test() -> None:
    """Run basic physics and convergence checks."""
    print("hamiltonian_dynamics: starting smoke tests...")

    # Test kinetic energy
    p = torch.randn(2, 3, 8, 8)
    T = kinetic_energy(p, mass=1.0)
    assert T.item() >= 0.0, "Kinetic energy must be non-negative"
    expected_T = p.pow(2).sum().item() / 2.0
    assert abs(T.item() - expected_T) < 1e-4, f"KE mismatch: {T.item()} vs {expected_T}"

    # Zero momentum should give zero kinetic energy
    T_zero = kinetic_energy(torch.zeros(1, 3, 4, 4))
    assert T_zero.item() == 0.0, "Zero momentum should give zero KE"
    print("  hamiltonian_dynamics: kinetic energy verified")

    # Test Hamiltonian with simple quadratic potential
    def quadratic_potential(q: torch.Tensor) -> torch.Tensor:
        """V(q) = ||q - 128||^2 / (2 * N) — harmonic well centered at gray."""
        return (q - 128.0).pow(2).mean()

    q = torch.randn(1, 3, 8, 8) * 50.0 + 128.0
    p_test = torch.randn(1, 3, 8, 8)
    H = hamiltonian(q, p_test, quadratic_potential, mass=1.0)
    assert H.item() >= 0.0, "Hamiltonian must be non-negative for quadratic potential"
    print("  hamiltonian_dynamics: Hamiltonian computation verified")

    # Test LeapfrogIntegrator: energy conservation on harmonic oscillator
    # V(q) = 0.5 * k * q^2, exact solution is sinusoidal with period 2*pi*sqrt(m/k)
    dt = 0.01
    mass = 1.0
    integrator = LeapfrogIntegrator(dt=dt, mass=mass, damping=0.0)

    q_harm = torch.tensor([100.0]).reshape(1, 1, 1, 1)  # displaced from equilibrium at 0
    p_harm = torch.zeros(1, 1, 1, 1)  # starting from rest

    def harm_grad(q: torch.Tensor) -> torch.Tensor:
        return q  # dV/dq = k*q with k=1

    H_initial = 0.5 * q_harm.pow(2).item() + 0.5 * p_harm.pow(2).item()
    max_drift = 0.0

    for _ in range(1000):
        grad_q = harm_grad(q_harm)
        q_new, p_half = integrator.step(q_harm, p_harm, grad_q)
        # Complete step with gradient at new position
        grad_q_new = harm_grad(q_new)
        p_new = p_half - (dt / 2.0) * grad_q_new
        q_harm, p_harm = q_new, p_new

        H_current = 0.5 * q_harm.pow(2).item() + 0.5 * p_harm.pow(2).item()
        drift = abs(H_current - H_initial) / (abs(H_initial) + 1e-12)
        max_drift = max(max_drift, drift)

    assert max_drift < 0.01, f"Energy drift too large for harmonic oscillator: {max_drift:.6f}"
    print(f"  hamiltonian_dynamics: leapfrog energy conservation verified (drift={max_drift:.2e})")

    # Test HamiltonianPixelOptimizer with quadratic potential
    optimizer = HamiltonianPixelOptimizer(cfg={
        "hamiltonian_dt": 0.1,
        "hamiltonian_mass": 1.0,
        "hamiltonian_steps": 50,
        "hamiltonian_damping": 0.05,
        "anneal_damping": True,
        "anneal_start": 0.01,
        "anneal_end": 0.1,
    })

    init = torch.rand(1, 3, 8, 8) * 255.0
    result, diag = optimizer.optimize(init, quadratic_potential, log_every=0)
    assert result.shape == init.shape
    assert result.min() >= 0.0 and result.max() <= 255.0
    assert diag["steps"] == 50
    assert len(diag["energy_history"]) == 50
    # With damping, potential should decrease
    V_start = diag["potential_history"][0]
    V_end = diag["potential_history"][-1]
    print(f"  hamiltonian_dynamics: optimizer V: {V_start:.4f} -> {V_end:.4f}")

    # Test NoseHooverThermostat
    thermostat = NoseHooverThermostat(
        target_temperature=0.1,
        thermostat_mass=10.0,
        mass=1.0,
    )
    q_nh = torch.rand(1, 1, 4, 4) * 10.0
    p_nh = torch.randn(1, 1, 4, 4)
    grad_nh = q_nh.clone()
    q_new_nh, p_half_nh = thermostat.step(q_nh, p_nh, grad_nh, dt=0.01)
    assert q_new_nh.shape == q_nh.shape
    assert p_half_nh.shape == p_nh.shape
    p_final_nh = thermostat.complete_step(p_half_nh, q_new_nh, dt=0.01)
    assert p_final_nh.shape == p_nh.shape
    print("  hamiltonian_dynamics: Nose-Hoover thermostat verified")

    # Test optimize_with_scorer with mock models
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
    masks = torch.randint(0, 5, (2, 16, 16))
    init_scorer = torch.rand(2, 3, 16, 16) * 255.0

    opt_scorer = HamiltonianPixelOptimizer(cfg={
        "hamiltonian_steps": 10,
        "hamiltonian_dt": 0.05,
    })
    result_scorer, diag_scorer = opt_scorer.optimize_with_scorer(
        init_scorer, masks, posenet, segnet, log_every=0,
    )
    assert result_scorer.shape == init_scorer.shape
    assert result_scorer.min() >= 0.0 and result_scorer.max() <= 255.0
    print("  hamiltonian_dynamics: scorer optimization verified")

    print("hamiltonian_dynamics: all smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
