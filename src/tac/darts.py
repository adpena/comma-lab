"""DARTS (Differentiable Architecture Search) — generic framework.

Implements the bilevel-optimization formulation from Liu, Simonyan, Yang
(ICLR 2019, "DARTS: Differentiable Architecture Search"):

    min_α  L_val(w*(α), α)
    s.t.   w*(α) = argmin_w L_train(w, α)

We use the **first-order approximation** (Liu et al. §2.3): instead of
back-propagating through the inner argmin via the implicit-function
theorem (which is O(|w|·|α|) and fragile), we treat the current weights
``w`` as a stationary surrogate for ``w*(α)``. Liu et al. report this
approximation matches the second-order solution within 0.3% on CIFAR-10
while being ~3× cheaper. The convergence guarantee carries over from
Bengio (2000, "Gradient-based optimization of hyperparameters", Neural
Computation) Theorem 1: alternating gradient descent on a bilevel system
with a smooth inner objective converges to a local stationary point of
the upper-level objective when the lower-level optimum is approximated
to first order.

Public API
----------

* :class:`DARTSCell` — N candidate ops + softmax-weighted mixture.
* :class:`DARTSOptimizer` — a thin :class:`torch.optim.Adam` wrapper that
  *only* steps on the architecture parameters ``α``. Keep this strictly
  separate from your model-weight optimizer (Liu et al. §2.2:
  "alternating gradient descent" — never mix the two parameter groups in
  a single optimizer).
* :func:`alpha_kl_to_uniform` — convergence diagnostic. Returns the KL
  divergence between the final softmax(α) distribution and the uniform
  distribution. > 2.0 nats = decisive convergence; < 1.0 nats = the
  search was inconclusive (CLAUDE.md "no premature kills" — DOCUMENT
  the inconclusive result rather than picking a winner from noise).

Temperature annealing
---------------------

We anneal a softmax temperature ``T`` from 5.0 → 0.1 across training:

    softmax(α / T)_i = exp(α_i / T) / Σ_j exp(α_j / T)

At ``T=5.0`` the mixture is near-uniform — every candidate carries
roughly equal gradient signal (uniform exploration). At ``T=0.1`` the
mixture is near-discrete — the search has effectively committed to its
top-α candidate (exploitation). The linear anneal schedule (vs cosine,
log) is justified by Hinton et al. 2015 "Distilling the Knowledge in a
Neural Network" §2: the temperature schedule's only requirement is
monotonicity; the rate is empirical. We use linear because it is the
simplest schedule with a clear interpretation (epoch-fraction of search
that has occurred).

Why first-order DARTS for our setting
-------------------------------------

Our DARTS application here is to **arch-knob** search (ratios, channel
dims, hidden dims) rather than the original DARTS *operation-search*
(conv vs sep-conv vs identity vs zero etc.). Knob-DARTS has a much
simpler bilevel landscape because every candidate produces an output of
the same shape — we never need the second-order zero-op trick or the
DARTS-V2 implicit-grad correction. First-order is the right tool.

This module does NOT implement:
  * second-order DARTS (too expensive for our 12h budget),
  * PC-DARTS partial-channel sampling (no benefit at our small N),
  * progressive shrinking (we have ≤ 16 candidates, no need to shrink),
  * DARTS+ early-stopping (we run to fixed epoch budget for
    reproducibility — convergence signal goes through alpha_kl_to_uniform
    and is recorded in provenance.json, not used to terminate the run).

References
----------

* Liu, Simonyan, Yang. "DARTS: Differentiable Architecture Search."
  ICLR 2019. https://arxiv.org/abs/1806.09055
* Bengio. "Gradient-based optimization of hyperparameters." Neural
  Computation 12.8 (2000): 1889-1900.
* Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network."
  NeurIPS-W 2015.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn


__all__ = [
    "DARTSCell",
    "DARTSOptimizer",
    "DARTSAnnealSchedule",
    "alpha_kl_to_uniform",
    "alpha_softmax",
    "discrete_arch_index",
]


# ── Temperature schedule ────────────────────────────────────────────────


@dataclass(frozen=True)
class DARTSAnnealSchedule:
    """Linear temperature anneal from ``T_start`` → ``T_end``.

    Defaults (T_start=5.0, T_end=0.1) mirror the rationale in the module
    docstring: start uniform, end near-discrete.
    """

    T_start: float = 5.0
    T_end: float = 0.1

    def __post_init__(self) -> None:
        if self.T_start <= 0 or self.T_end <= 0:
            raise ValueError(
                f"DARTSAnnealSchedule temperatures must be > 0; got "
                f"T_start={self.T_start}, T_end={self.T_end}"
            )
        if self.T_start <= self.T_end:
            raise ValueError(
                f"DARTSAnnealSchedule must anneal DOWN (start > end); got "
                f"T_start={self.T_start} ≤ T_end={self.T_end}. The whole "
                f"point of the schedule is exploration → exploitation."
            )

    def temperature(self, epoch: int, total_epochs: int) -> float:
        """Linear anneal. Clamped to [T_end, T_start]."""
        if total_epochs <= 1:
            return self.T_end
        # Clamp epoch to [0, total_epochs - 1] so callers cannot accidentally
        # produce a temperature outside [T_end, T_start] by passing
        # epoch ≥ total_epochs.
        e = max(0, min(int(epoch), total_epochs - 1))
        frac = e / (total_epochs - 1)  # 0.0 at start, 1.0 at end
        return float(self.T_start + frac * (self.T_end - self.T_start))


# ── Softmax + diagnostics ───────────────────────────────────────────────


def alpha_softmax(alpha: torch.Tensor, temperature: float) -> torch.Tensor:
    """Numerically-stable softmax(α / T).

    PyTorch's :func:`torch.softmax` already subtracts the max for stability,
    so we simply divide by T. Guarded against T<=0 (which would silently
    produce inf/nan)."""
    if temperature <= 0:
        raise ValueError(f"DARTS softmax temperature must be > 0, got {temperature}")
    return torch.softmax(alpha / temperature, dim=-1)


def alpha_kl_to_uniform(alpha: torch.Tensor, temperature: float = 1.0) -> float:
    """KL( softmax(α/T) || Uniform(N) ) in nats.

    KL(p||u) = Σ p_i log(p_i · N) = log(N) + Σ p_i log(p_i) = log(N) - H(p)

    where H(p) is the Shannon entropy of p in nats and N is the number of
    candidates. KL → 0 means p is uniform (search is inconclusive); KL →
    log(N) means p is a Dirac on a single candidate (search has fully
    committed). Per the module docstring: > 2.0 nats = decisive,
    < 1.0 nats = inconclusive.

    Args:
        alpha: shape (N,) — the unnormalized arch logits.
        temperature: T to apply before softmax (typically 1.0 for the
            *final* diagnostic — we want to measure the discrete-decision
            quality, not the smoothed mixture-quality).

    Returns:
        KL divergence in nats (always ≥ 0).
    """
    if alpha.dim() != 1:
        raise ValueError(f"alpha_kl_to_uniform expects 1-D α, got shape {tuple(alpha.shape)}")
    p = alpha_softmax(alpha.detach(), temperature)
    n = alpha.numel()
    # Numerically guard against p_i == 0 (would produce -inf in log).
    log_n = math.log(n)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum().item()
    kl = log_n - entropy
    # KL is always ≥ 0 (Gibbs' inequality); clamp tiny float noise.
    return max(0.0, kl)


def discrete_arch_index(alpha: torch.Tensor) -> int:
    """argmax(α) — the *discovered* architecture for retrain-from-scratch."""
    if alpha.dim() != 1:
        raise ValueError(f"discrete_arch_index expects 1-D α, got shape {tuple(alpha.shape)}")
    return int(alpha.detach().argmax().item())


# ── DARTSCell ────────────────────────────────────────────────────────────


class DARTSCell(nn.Module):
    """Mixture of N candidate ops with learnable architecture parameters α.

    Forward pass::

        out = Σ_i softmax(α / T)_i · op_i(x)

    The N candidate ops MUST all produce outputs of the same shape (this
    is a knob-search cell, not a topology-search cell — see module
    docstring for the design rationale).

    α is initialized to **zeros** so that softmax(α/T) starts uniform
    regardless of T. This avoids biasing the search toward any candidate
    before any data has been seen.

    Args:
        ops: an iterable of N candidate ``nn.Module``s. Each is called as
            ``op(x)`` in the forward pass.
        anneal: temperature schedule (defaults to ``DARTSAnnealSchedule()``).
        names: optional human-readable name per op for logging /
            provenance. Length must match ``ops``.

    Attributes:
        alpha: ``nn.Parameter`` of shape (N,). Steered by a separate
            optimizer (see :class:`DARTSOptimizer`).
        ops: ``nn.ModuleList`` of the N candidates.
        names: tuple of N strings.
    """

    def __init__(
        self,
        ops: Iterable[nn.Module],
        anneal: DARTSAnnealSchedule | None = None,
        names: Sequence[str] | None = None,
    ):
        super().__init__()
        self.ops = nn.ModuleList(list(ops))
        if len(self.ops) < 2:
            raise ValueError(
                f"DARTSCell needs ≥ 2 candidate ops to be a search; got {len(self.ops)}"
            )
        # α starts at zeros — softmax(0/T) = uniform regardless of T, so
        # initialization is independent of the schedule (Liu et al. §2.2).
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))
        self.anneal = anneal if anneal is not None else DARTSAnnealSchedule()
        if names is not None:
            names = tuple(str(n) for n in names)
            if len(names) != len(self.ops):
                raise ValueError(
                    f"names length {len(names)} != ops length {len(self.ops)}"
                )
            self.names: tuple[str, ...] = names
        else:
            self.names = tuple(f"op_{i}" for i in range(len(self.ops)))
        # Current temperature; updated by :meth:`temperature_anneal`.
        # Stored as a float (not a Parameter) — the schedule is operator-
        # controlled, not gradient-controlled.
        self._current_T: float = float(self.anneal.T_start)

    # ── Temperature schedule ────────────────────────────────────────────

    def temperature_anneal(self, epoch: int, total_epochs: int) -> float:
        """Update the current temperature from the linear anneal schedule."""
        self._current_T = self.anneal.temperature(epoch, total_epochs)
        return self._current_T

    @property
    def current_temperature(self) -> float:
        return self._current_T

    # ── Forward + introspection ─────────────────────────────────────────

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Mixture forward: Σ softmax(α/T)_i · op_i(x).

        Extra ``*args`` / ``**kwargs`` are forwarded to every candidate op
        unchanged (so callers can pass e.g. conditioning tensors that
        every op shares).
        """
        weights = alpha_softmax(self.alpha, self._current_T)
        out: torch.Tensor | None = None
        for w_i, op in zip(weights, self.ops):
            y = op(x, *args, **kwargs)
            term = w_i * y
            out = term if out is None else out + term
        # `out` is guaranteed non-None because we asserted len(ops) ≥ 2.
        assert out is not None
        return out

    def discrete_arch(self) -> int:
        """argmax(α) — the discovered architecture index."""
        return discrete_arch_index(self.alpha)

    def discrete_arch_name(self) -> str:
        """Human-readable name of the discovered architecture."""
        return self.names[self.discrete_arch()]

    def alpha_kl_nats(self, temperature: float = 1.0) -> float:
        """Convergence diagnostic. See :func:`alpha_kl_to_uniform`."""
        return alpha_kl_to_uniform(self.alpha, temperature)

    def alpha_softmax_distribution(self, temperature: float | None = None) -> torch.Tensor:
        """Detached softmax(α/T) snapshot for logging."""
        T = float(temperature) if temperature is not None else self._current_T
        return alpha_softmax(self.alpha.detach(), T)

    # ── Parameter-group helpers ─────────────────────────────────────────

    def arch_parameters(self) -> list[nn.Parameter]:
        """Just the α parameter — for the DARTS (architecture) optimizer."""
        return [self.alpha]

    def weight_parameters(self) -> list[nn.Parameter]:
        """Everything EXCEPT α — for the model-weight optimizer.

        Use :func:`split_arch_weight_params` at the *full-supernet* level
        if your supernet has multiple :class:`DARTSCell` instances; this
        method only handles a single cell."""
        own_alpha = id(self.alpha)
        return [p for p in self.parameters() if id(p) != own_alpha]


# ── Param-group splitter for full supernets ─────────────────────────────


def split_arch_weight_params(
    supernet: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Walk a supernet and return (arch_params, weight_params).

    arch_params = every ``alpha`` :class:`nn.Parameter` of every
    :class:`DARTSCell` in the supernet. weight_params = everything else.

    This is the canonical way to wire two optimizers (one for α, one for
    weights) when the supernet contains multiple :class:`DARTSCell`s.
    """
    arch_ids: set[int] = set()
    arch_params: list[nn.Parameter] = []
    for module in supernet.modules():
        if isinstance(module, DARTSCell):
            arch_ids.add(id(module.alpha))
            arch_params.append(module.alpha)
    weight_params = [p for p in supernet.parameters() if id(p) not in arch_ids]
    return arch_params, weight_params


# ── DARTSOptimizer (Adam on α) ──────────────────────────────────────────


class DARTSOptimizer(torch.optim.Adam):
    """Adam optimizer specialized for the architecture parameters α.

    Liu et al. §2.2: α is updated with Adam at lr=3e-4, β=(0.5, 0.999),
    weight_decay=1e-3. We mirror those defaults here. Override at the
    call site if you have a justified reason.

    Treat this as a *strictly separate* optimizer from your model-weight
    optimizer. Alternating gradient descent::

        for batch in train_loader:
            # 1. Update α on the val batch (frozen weights)
            arch_opt.zero_grad()
            loss_val = model(val_batch)
            loss_val.backward()
            arch_opt.step()

            # 2. Update w on the train batch (frozen α)
            weight_opt.zero_grad()
            loss_train = model(train_batch)
            loss_train.backward()
            weight_opt.step()

    Note that PyTorch's autograd will populate ``alpha.grad`` from the
    first backward and ``weight.grad`` from the second, but each
    optimizer only steps on its own parameter group, so the alternation
    is bookkeeping-clean as long as you ``zero_grad()`` consistently.
    """

    def __init__(
        self,
        arch_params: Iterable[nn.Parameter],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.5, 0.999),
        weight_decay: float = 1e-3,
    ):
        params = list(arch_params)
        if not params:
            raise ValueError(
                "DARTSOptimizer needs ≥ 1 arch parameter — pass alpha "
                "from each DARTSCell in the supernet."
            )
        # All parameter tensors must be 1-D (the α convention) — guard so
        # that no one accidentally hands us a model weight tensor.
        for p in params:
            if p.dim() != 1:
                raise ValueError(
                    f"DARTSOptimizer arch params must be 1-D (the α "
                    f"vector); got tensor of shape {tuple(p.shape)}"
                )
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)


# ── Trajectory recorder (for provenance) ────────────────────────────────


class DARTSAlphaTrajectory:
    """Append-only record of (epoch, T, alpha-snapshot) tuples.

    The CLAUDE.md "algorithmic rigor" requirement says: *each search
    reports the FULL α evolution trajectory (per-epoch α values, not
    just final)*. This class is the canonical container; serialize the
    list to JSON in your provenance file.
    """

    def __init__(self, op_names: Sequence[str]):
        self.op_names: tuple[str, ...] = tuple(op_names)
        self.records: list[dict] = []

    def record(
        self,
        epoch: int,
        cell: DARTSCell,
        *,
        train_loss: float | None = None,
        val_loss: float | None = None,
    ) -> None:
        if cell.alpha.numel() != len(self.op_names):
            raise ValueError(
                f"DARTSAlphaTrajectory op_names length {len(self.op_names)} "
                f"does not match cell.alpha numel {cell.alpha.numel()}"
            )
        snapshot = {
            "epoch": int(epoch),
            "temperature": float(cell.current_temperature),
            "alpha": [float(v) for v in cell.alpha.detach().cpu().tolist()],
            "softmax": [
                float(v)
                for v in cell.alpha_softmax_distribution(temperature=1.0).cpu().tolist()
            ],
            "argmax_index": cell.discrete_arch(),
            "argmax_name": cell.discrete_arch_name(),
            "kl_nats_to_uniform": cell.alpha_kl_nats(temperature=1.0),
        }
        if train_loss is not None:
            snapshot["train_loss"] = float(train_loss)
        if val_loss is not None:
            snapshot["val_loss"] = float(val_loss)
        self.records.append(snapshot)

    def to_dict(self) -> dict:
        if not self.records:
            return {"op_names": list(self.op_names), "records": [], "discovered": None}
        final = self.records[-1]
        return {
            "op_names": list(self.op_names),
            "records": self.records,
            "discovered": {
                "argmax_index": final["argmax_index"],
                "argmax_name": final["argmax_name"],
                "alpha_final": final["alpha"],
                "softmax_final": final["softmax"],
                "kl_nats_final": final["kl_nats_to_uniform"],
                # Convergence verdict per CLAUDE.md "algorithmic rigor":
                #   > 2.0 nats = decisive, < 1.0 nats = inconclusive.
                "convergence_verdict": _convergence_verdict(
                    final["kl_nats_to_uniform"]
                ),
            },
        }


def _convergence_verdict(kl_nats: float) -> str:
    if kl_nats >= 2.0:
        return "decisive"
    if kl_nats >= 1.0:
        return "moderate"
    return "inconclusive"


# ── Search-loop helper (alternating SGD) ────────────────────────────────


def darts_search_step(
    supernet: nn.Module,
    val_loss_fn: Callable[[], torch.Tensor],
    train_loss_fn: Callable[[], torch.Tensor],
    arch_opt: DARTSOptimizer,
    weight_opt: torch.optim.Optimizer,
) -> tuple[float, float]:
    """One alternating-SGD step (first-order DARTS, Liu et al. §2.3).

    Args:
        supernet: the model containing the :class:`DARTSCell`(s).
        val_loss_fn: callable returning a scalar val-batch loss tensor.
            Called with α only (do NOT take the inner argmin step here).
        train_loss_fn: callable returning a scalar train-batch loss
            tensor. Called with weights only.
        arch_opt: :class:`DARTSOptimizer` over the α parameters.
        weight_opt: any standard optimizer over the weight parameters.

    Returns:
        (val_loss, train_loss) as Python floats — for logging.
    """
    # Step 1: arch update on val batch.
    arch_opt.zero_grad(set_to_none=True)
    val_loss = val_loss_fn()
    val_loss.backward()
    arch_opt.step()

    # Step 2: weight update on train batch.
    weight_opt.zero_grad(set_to_none=True)
    train_loss = train_loss_fn()
    train_loss.backward()
    weight_opt.step()

    return float(val_loss.detach().item()), float(train_loss.detach().item())
