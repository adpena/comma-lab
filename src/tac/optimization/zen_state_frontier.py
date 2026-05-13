"""Zen-state frontier optimization primitives.

This module converts the 2026-05-13 zen-state frontier ledgers into reusable
code. It intentionally carries no score authority: every function emits
optimizer/math artifacts only. A Pact score changes only after a byte-closed
archive/runtime packet passes the canonical exact-eval path.

Implemented primitives:

* information-geometric Langevin optimizer with diagonal Fisher preconditioning;
* Onsager-style pair/sample weights from score sensitivity;
* diagonal-Gaussian Wasserstein barycenters for checkpoint soups;
* one-dimensional Brenier/quantile quantization for scalar payloads;
* entropic Sinkhorn transport plans for codebook/latent assignment probes;
* tensor-train/MPS decomposition for weight-compression probes;
* tropical residual adapter forward pass for TD-LoRA-style probes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from tac.optimization.langevin_optimizer import (
    cosine_temperature_schedule,
    exponential_temperature_schedule,
    geman_geman_log_schedule,
)

ScheduleFn = Callable[[int, int, float, float], float]


class InformationGeometricLangevinOptimizer(torch.optim.Optimizer):
    """Diagonal-Fisher natural-gradient Langevin optimizer.

    The update is the Euler-Maruyama discretization of

    ``dtheta = -F^-1 grad L dt + sqrt(2 T F^-1) dW``.

    ``fisher_diag`` may be supplied per parameter tensor. If omitted, the
    optimizer uses an exponential moving average of squared gradients, which is
    the Sophia/Adam-style diagonal empirical Fisher approximation. The update is
    deterministic when ``T_init == T_final == 0``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-4,
        T_init: float = 0.0,
        T_final: float = 0.0,
        n_steps: int = 1_000,
        schedule: str | ScheduleFn = "cosine",
        fisher_diag: Sequence[torch.Tensor] | None = None,
        fisher_beta: float = 0.95,
        fisher_damping: float = 1e-6,
        weight_decay: float = 0.0,
        noise_seed: int = 0,
    ) -> None:
        if lr <= 0 or not math.isfinite(lr):
            raise ValueError(f"lr must be positive finite; got {lr!r}")
        if fisher_damping <= 0 or not math.isfinite(fisher_damping):
            raise ValueError(
                f"fisher_damping must be positive finite; got {fisher_damping!r}"
            )
        if not 0.0 <= fisher_beta < 1.0:
            raise ValueError(f"fisher_beta must be in [0, 1); got {fisher_beta!r}")
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive; got {n_steps!r}")

        self._schedule_fn = _resolve_schedule(schedule)
        self._step_index = 0
        self._generator = torch.Generator()
        self._generator.manual_seed(int(noise_seed))
        params_list = list(params)
        defaults = {
            "lr": float(lr),
            "T_init": float(T_init),
            "T_final": float(T_final),
            "n_steps": int(n_steps),
            "fisher_beta": float(fisher_beta),
            "fisher_damping": float(fisher_damping),
            "weight_decay": float(weight_decay),
        }
        super().__init__(params_list, defaults)

        if fisher_diag is not None:
            flat_params = [p for group in self.param_groups for p in group["params"]]
            if len(fisher_diag) != len(flat_params):
                raise ValueError(
                    f"fisher_diag length {len(fisher_diag)} does not match "
                    f"parameter count {len(flat_params)}"
                )
            for p, f in zip(flat_params, fisher_diag, strict=True):
                if tuple(f.shape) != tuple(p.shape):
                    raise ValueError(
                        f"fisher_diag shape {tuple(f.shape)} does not match "
                        f"parameter shape {tuple(p.shape)}"
                    )
                state = self.state[p]
                state["fisher_diag"] = f.detach().to(device=p.device, dtype=p.dtype)
                state["external_fisher"] = True

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = float(group["lr"])
            n_steps = int(group["n_steps"])
            T = self._schedule_fn(
                self._step_index,
                n_steps,
                float(group["T_init"]),
                float(group["T_final"]),
            )
            fisher_beta = float(group["fisher_beta"])
            damping = float(group["fisher_damping"])
            weight_decay = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if weight_decay:
                    grad = grad.add(p.detach(), alpha=weight_decay)
                state = self.state[p]
                if state.get("external_fisher") is True:
                    fisher = state["fisher_diag"]
                else:
                    fisher = state.get("fisher_diag")
                    if fisher is None:
                        fisher = torch.zeros_like(p)
                    fisher.mul_(fisher_beta).addcmul_(grad, grad, value=1.0 - fisher_beta)
                    state["fisher_diag"] = fisher
                inv_fisher = torch.rsqrt(fisher.clamp_min(0).add(damping).square())
                p.addcmul_(grad, inv_fisher, value=-lr)
                if T > 0.0:
                    std = torch.sqrt(torch.as_tensor(2.0 * T * lr, device=p.device, dtype=p.dtype))
                    noise = torch.randn(
                        p.shape,
                        generator=self._generator if p.device.type == "cpu" else None,
                        device=p.device,
                        dtype=p.dtype,
                    )
                    p.add_(noise * std * torch.sqrt(inv_fisher))
        self._step_index += 1
        return loss


def _resolve_schedule(schedule: str | ScheduleFn) -> ScheduleFn:
    if callable(schedule):
        return schedule
    normalized = schedule.lower().strip()
    if normalized == "cosine":
        return cosine_temperature_schedule
    if normalized in {"exp", "exponential"}:
        return exponential_temperature_schedule
    if normalized in {"log", "geman_geman"}:
        return geman_geman_log_schedule
    raise ValueError(f"unknown temperature schedule: {schedule!r}")


def onsager_importance_weights(
    sensitivities: Sequence[float] | np.ndarray,
    *,
    floor: float = 1e-12,
    normalize_sum: float = 1.0,
) -> np.ndarray:
    """Return sample weights proportional to score-sensitivity energy.

    This is the implementable form of the Onsager-regression/sample-variance
    argument from the statistical-physics ledger: high-response pairs get more
    optimizer mass, but every pair keeps nonzero probability.
    """

    values = np.asarray(sensitivities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("sensitivities must be a non-empty 1D array")
    if floor <= 0 or not math.isfinite(floor):
        raise ValueError(f"floor must be positive finite; got {floor!r}")
    if normalize_sum <= 0 or not math.isfinite(normalize_sum):
        raise ValueError(
            f"normalize_sum must be positive finite; got {normalize_sum!r}"
        )
    clipped = np.maximum(values, 0.0) + floor
    total = float(clipped.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("sensitivities produced non-finite total weight")
    return clipped * (normalize_sum / total)


def wasserstein_barycenter_diagonal_gaussians(
    means: Sequence[Sequence[float]] | np.ndarray,
    variances: Sequence[Sequence[float]] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute W2 barycenter of diagonal Gaussian checkpoints.

    For 1D Gaussians the barycenter standard deviation is the weighted average
    of standard deviations. A diagonal Gaussian is the product of independent
    1D Gaussians, so this applies coordinate-wise.
    """

    m = np.asarray(means, dtype=np.float64)
    v = np.asarray(variances, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("means must have shape (k, d)")
    if v.shape != m.shape:
        raise ValueError(f"variances shape {v.shape} must match means shape {m.shape}")
    if np.any(v < 0) or not np.all(np.isfinite(v)):
        raise ValueError("variances must be finite and non-negative")
    w = _normalize_weights(weights, m.shape[0])
    mean_star = np.sum(m * w[:, None], axis=0)
    std_star = np.sum(np.sqrt(v) * w[:, None], axis=0)
    return mean_star, std_star * std_star


def brenier_quantile_quantize_1d(
    values: Sequence[float] | np.ndarray,
    *,
    n_levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize 1D values by monotone quantile transport.

    The returned levels are per-bin means over sorted values. Assignment is
    monotone in the input value, which is the 1D Brenier-map property.
    """

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1; got {n_levels!r}")
    n_levels = min(int(n_levels), int(x.size))
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    bins = np.array_split(sorted_x, n_levels)
    levels = np.array([float(chunk.mean()) for chunk in bins], dtype=np.float64)
    q_sorted = np.concatenate(
        [np.full(len(chunk), levels[i], dtype=np.float64) for i, chunk in enumerate(bins)]
    )
    quantized = np.empty_like(x)
    quantized[order] = q_sorted
    return quantized, levels


def sinkhorn_transport_plan(
    cost: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_weights: Sequence[float] | np.ndarray | None = None,
    target_weights: Sequence[float] | np.ndarray | None = None,
    epsilon: float = 0.05,
    n_iters: int = 200,
    tol: float = 1e-9,
) -> np.ndarray:
    """Compute an entropic OT plan with Sinkhorn scaling."""

    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or 0 in c.shape:
        raise ValueError("cost must be a non-empty 2D array")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError(f"epsilon must be positive finite; got {epsilon!r}")
    a = _normalize_weights(source_weights, c.shape[0])
    b = _normalize_weights(target_weights, c.shape[1])
    kernel = np.exp(-c / epsilon)
    kernel = np.maximum(kernel, np.finfo(np.float64).tiny)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max(1, int(n_iters))):
        prev_u = u.copy()
        u = a / (kernel @ v)
        v = b / (kernel.T @ u)
        if np.max(np.abs(u - prev_u)) < tol:
            break
    plan = (u[:, None] * kernel) * v[None, :]
    return plan


@dataclass(frozen=True)
class TensorTrain:
    """Tensor-train/MPS decomposition."""

    cores: tuple[np.ndarray, ...]
    original_shape: tuple[int, ...]

    @property
    def ranks(self) -> tuple[int, ...]:
        if not self.cores:
            return ()
        ranks = [int(self.cores[0].shape[0])]
        ranks.extend(int(core.shape[-1]) for core in self.cores)
        return tuple(ranks)


def mps_decompose(
    tensor: Sequence[float] | np.ndarray,
    *,
    max_rank: int | None = None,
    relative_tolerance: float = 0.0,
) -> TensorTrain:
    """Tensor-train decomposition by sequential SVD."""

    x = np.asarray(tensor, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("tensor must have rank >= 2 for MPS decomposition")
    if max_rank is not None and max_rank < 1:
        raise ValueError(f"max_rank must be >= 1; got {max_rank!r}")
    if relative_tolerance < 0 or not math.isfinite(relative_tolerance):
        raise ValueError(
            f"relative_tolerance must be non-negative finite; got {relative_tolerance!r}"
        )
    shape = tuple(int(s) for s in x.shape)
    cores: list[np.ndarray] = []
    rank_left = 1
    unfolding = x.reshape(shape[0], -1)
    norm = float(np.linalg.norm(x))
    tol_abs = relative_tolerance * norm
    for mode, dim in enumerate(shape[:-1]):
        unfolding = unfolding.reshape(rank_left * dim, -1)
        u, s, vh = np.linalg.svd(unfolding, full_matrices=False)
        rank = _choose_svd_rank(s, max_rank=max_rank, tol_abs=tol_abs)
        u = u[:, :rank]
        s = s[:rank]
        vh = vh[:rank, :]
        cores.append(u.reshape(rank_left, dim, rank))
        unfolding = s[:, None] * vh
        rank_left = rank
        if mode + 1 < len(shape) - 1:
            unfolding = unfolding.reshape(rank_left * shape[mode + 1], -1)
    cores.append(unfolding.reshape(rank_left, shape[-1], 1))
    return TensorTrain(cores=tuple(cores), original_shape=shape)


def mps_reconstruct(train: TensorTrain) -> np.ndarray:
    """Reconstruct a dense tensor from a tensor-train/MPS decomposition."""

    if not train.cores:
        raise ValueError("TensorTrain has no cores")
    out = train.cores[0]
    for core in train.cores[1:]:
        out = np.tensordot(out, core, axes=([-1], [0]))
    return np.squeeze(out, axis=(0, -1)).reshape(train.original_shape)


def tropical_lora_forward(
    base_output: torch.Tensor,
    residual_candidates: Sequence[torch.Tensor],
    *,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Apply a tropical/max-plus residual adapter.

    ``temperature=0`` uses exact max-plus selection. Positive temperature uses
    log-sum-exp smoothing so the adapter can be trained by gradient descent.
    """

    if not residual_candidates:
        return base_output
    stacked = torch.stack([base_output, *[base_output + r for r in residual_candidates]])
    if temperature <= 0.0:
        return torch.amax(stacked, dim=0)
    if not math.isfinite(temperature):
        raise ValueError(f"temperature must be finite; got {temperature!r}")
    return torch.logsumexp(stacked / temperature, dim=0) * temperature


def _normalize_weights(
    weights: Sequence[float] | np.ndarray | None,
    n: int,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights shape {w.shape} must be ({n},)")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return w / total


def _choose_svd_rank(
    singular_values: np.ndarray,
    *,
    max_rank: int | None,
    tol_abs: float,
) -> int:
    if singular_values.size == 0:
        return 1
    rank = int(singular_values.size)
    if tol_abs > 0.0:
        tail_energy = np.cumsum(singular_values[::-1] ** 2)[::-1]
        keep = np.nonzero(np.sqrt(tail_energy) > tol_abs)[0]
        rank = int(keep[-1] + 1) if keep.size else 1
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return max(1, rank)


__all__ = [
    "InformationGeometricLangevinOptimizer",
    "TensorTrain",
    "brenier_quantile_quantize_1d",
    "mps_decompose",
    "mps_reconstruct",
    "onsager_importance_weights",
    "sinkhorn_transport_plan",
    "tropical_lora_forward",
    "wasserstein_barycenter_diagonal_gaussians",
]
