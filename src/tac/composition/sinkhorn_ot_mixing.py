# SPDX-License-Identifier: MIT
"""Sinkhorn-Cuturi Entropic OT Mixing — composition/stacking primitive.

Cuturi 2013 ("Sinkhorn Distances: Lightspeed Computation of Optimal Transport")
introduced entropic regularisation of the optimal-transport problem:

    OT_λ(μ, ν) = min_{T ∈ Π(μ, ν)} ⟨T, C⟩ - (1/λ) H(T)                  (S.1)

where ``Π(μ, ν)`` is the set of joint distributions with marginals ``μ`` and
``ν``, ``C`` is the cost matrix, ``H(T)`` is the entropy of ``T``, and
``λ > 0`` is the regularisation. The Sinkhorn-Knopp algorithm solves this in
``O(n² · iters)`` via alternating row/column normalisation of the kernel
``K = exp(-λ C)``.

This module exposes a **differentiable** Sinkhorn solver that returns the
transport plan ``T*`` and the regularised OT cost ``⟨T*, C⟩``. The plan can
be used to MIX parameter sets (or any pair of finite measures over a feature
space) by pulling source mass through ``T*`` to the target support.

Source memos:
- ``.omx/research/zen_state_frontier_optimal_transport_20260513.md``
- Sister primitive: ``tac.composition.wbce_mera`` (Wasserstein barycenter).

Cross-references
----------------
- Bregman-divergence mixing (``tac.composition.bregman_mixing``) — KL is the
  entropic prior, so Sinkhorn-OT is Bregman projection onto the transport
  polytope.
- Stack-of-stacks integration: ``tac.composition.stack_of_stacks.compose``.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
This module produces a forward-pass module + serialisable spec; it does not
modify archive bytes by itself. Substrate integration must register a
parser-section manifest entry per CLAUDE.md Catalog #124 before any
``score_claim=True``. Until paired ``[contest-CUDA]`` + ``[contest-CPU]``
anchors land on a Sinkhorn-OT-equipped substrate, every result is
``score_claim=False``, ``promotion_eligible=False``,
``ready_for_exact_eval_dispatch=False``.

HNeRV parity discipline (13 lessons)
------------------------------------
1. Score-aware: the solver is fully differentiable; trainer drives
   apply_eval_roundtrip + scorer gradient.
2. Export-first: :meth:`SinkhornOTMixer.serialize_state` is deterministic.
3-6. Substrate concerns; not violated.
7. Bolt-on ≤ 300 LOC.
8-13. Standard substrate concerns; not violated.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import torch

from tac.composition.frontier_primitives import (
    sinkhorn_transport_plan as _canonical_sinkhorn_transport_plan,
)

SINKHORN_MAGIC = b"SKH1"
SINKHORN_SCHEMA_VERSION = 1

DEFAULT_REG = 0.1
DEFAULT_MAX_ITERS = 200
DEFAULT_TOL = 1e-6
DEFAULT_EPS = 1e-12


class SinkhornError(ValueError):
    """Raised when a SinkhornOTMixer spec or input is invalid."""


@dataclass(frozen=True)
class SinkhornOTMixerSpec:
    """Specification for a Sinkhorn-OT mixer.

    Args:
        reg: entropic regularisation strength ``1/λ`` in (S.1). Smaller →
            sharper transport plan, more iterations, less stable.
        max_iters: maximum Sinkhorn-Knopp iterations.
        tol: convergence tolerance on the L1 norm of the row-marginal
            residual.
        eps: numerical floor to avoid log(0) inside the log-domain
            iteration.
        log_domain: if True, run the iteration in log-space for numerical
            stability at small ``reg``. Slightly slower but recommended
            for reg < 0.05.
    """

    reg: float = DEFAULT_REG
    max_iters: int = DEFAULT_MAX_ITERS
    tol: float = DEFAULT_TOL
    eps: float = DEFAULT_EPS
    log_domain: bool = True

    def __post_init__(self) -> None:
        if self.reg <= 0 or not math.isfinite(float(self.reg)):
            raise SinkhornError(f"reg must be positive, got {self.reg}")
        if self.max_iters <= 0:
            raise SinkhornError(f"max_iters must be positive, got {self.max_iters}")
        if self.tol <= 0 or not math.isfinite(float(self.tol)):
            raise SinkhornError(f"tol must be positive, got {self.tol}")
        if self.eps <= 0 or not math.isfinite(float(self.eps)):
            raise SinkhornError(f"eps must be positive, got {self.eps}")


def sinkhorn_solve(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    *,
    reg: float = DEFAULT_REG,
    max_iters: int = DEFAULT_MAX_ITERS,
    tol: float = DEFAULT_TOL,
    eps: float = DEFAULT_EPS,
    log_domain: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Solve the entropic OT problem (Eq. S.1) by Sinkhorn-Knopp iteration.

    Args:
        a: source marginal of shape ``(n,)``; must be non-negative.
        b: target marginal of shape ``(m,)``; must be non-negative.
        cost: cost matrix of shape ``(n, m)``.
        reg: entropic regularisation strength.
        max_iters, tol, eps: solver controls.
        log_domain: numerical-stability mode.

    Returns:
        ``(plan, cost_value, iterations)`` where ``plan`` is shape
        ``(n, m)`` with row-sums ≈ ``a`` and column-sums ≈ ``b``, and
        ``cost_value = ⟨plan, cost⟩``.
    """
    if reg <= 0 or not math.isfinite(float(reg)):
        raise SinkhornError("reg must be positive and finite")
    if max_iters <= 0:
        raise SinkhornError("max_iters must be positive")
    if tol <= 0 or not math.isfinite(float(tol)):
        raise SinkhornError("tol must be positive and finite")
    if eps <= 0 or not math.isfinite(float(eps)):
        raise SinkhornError("eps must be positive and finite")
    if a.dim() != 1 or b.dim() != 1:
        raise SinkhornError("a and b must be 1-D tensors")
    if cost.dim() != 2:
        raise SinkhornError("cost must be a 2-D tensor")
    if cost.shape != (a.shape[0], b.shape[0]):
        raise SinkhornError(
            f"cost shape {cost.shape} != ({a.shape[0]}, {b.shape[0]})"
        )
    _require_finite_tensor(a, field_name="a")
    _require_finite_tensor(b, field_name="b")
    _require_finite_tensor(cost, field_name="cost")
    if torch.any(a < 0) or torch.any(b < 0):
        raise SinkhornError("marginals must be non-negative")
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum <= 0 or b_sum <= 0:
        raise SinkhornError("marginals must have positive total mass")
    # Normalise so transport plan is doubly-stochastic in mass-units.
    a_n = a / a_sum
    b_n = b / b_sum

    if not log_domain:
        try:
            result = _canonical_sinkhorn_transport_plan(
                a,
                b,
                cost,
                epsilon=reg,
                max_iters=max_iters,
                tol=tol,
            )
        except ValueError as exc:
            raise SinkhornError(str(exc)) from exc
        scaled_plan = result.plan.to(dtype=cost.dtype, device=cost.device) * a_sum
        cost_value = (scaled_plan * cost).sum()
        return scaled_plan, cost_value, result.iterations

    if log_domain:
        # log-domain Sinkhorn (numerically stable for small reg)
        log_a = torch.log(a_n.clamp_min(eps))
        log_b = torch.log(b_n.clamp_min(eps))
        log_K = -cost / reg
        f = torch.zeros_like(a_n)
        g = torch.zeros_like(b_n)
        iters = 0
        for it in range(max_iters):
            iters = it + 1
            # f update: log a - logsumexp(log_K + g[None, :], dim=1)
            f_new = log_a - torch.logsumexp(log_K + g[None, :], dim=1)
            g_new = log_b - torch.logsumexp(log_K + f_new[:, None], dim=0)
            err = (f_new - f).abs().sum() + (g_new - g).abs().sum()
            f = f_new
            g = g_new
            if float(err.detach()) < tol:
                break
        log_plan = log_K + f[:, None] + g[None, :]
        plan = log_plan.exp()
    # Scale plan back to original mass.
    scaled_plan = plan * a_sum  # plan rows ~= a_n; scaling by a_sum gives a.
    cost_value = (scaled_plan * cost).sum()
    return scaled_plan, cost_value, iters


class SinkhornOTMixer:
    """Mix source vectors into a target support via entropic OT plan.

    Given K source vectors ``{x_k}`` with weights ``α_k`` and M target
    "anchor" vectors ``{y_j}`` with weights ``β_j``, the mixer computes
    the Sinkhorn plan ``T*`` minimising

        ⟨T, C⟩ - (1/λ) H(T)  s.t. T 1 = α, T^T 1 = β

    where ``C[k, j] = ||x_k - y_j||²``, then returns the *barycentric
    projection* of source through the plan:

        z_j = Σ_k T*[k, j] · x_k / β_j

    Example
    -------
    >>> import torch
    >>> from tac.composition.sinkhorn_ot_mixing import (
    ...     SinkhornOTMixer, SinkhornOTMixerSpec,
    ... )
    >>> mixer = SinkhornOTMixer(SinkhornOTMixerSpec(reg=0.1))
    >>> src = torch.tensor([[0.0], [2.0]])
    >>> tgt_anchors = torch.tensor([[1.0]])
    >>> z = mixer.transport(src, tgt_anchors)
    >>> z.shape
    torch.Size([1, 1])
    """

    def __init__(self, spec: SinkhornOTMixerSpec) -> None:
        self.spec = spec

    def transport(
        self,
        source: torch.Tensor,
        target_anchors: torch.Tensor,
        *,
        source_weights: torch.Tensor | None = None,
        target_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return barycentric projection ``(M, D)`` of source through plan."""
        if source.dim() != 2 or target_anchors.dim() != 2:
            raise SinkhornError("source and target_anchors must be 2-D")
        if source.shape[1] != target_anchors.shape[1]:
            raise SinkhornError(
                "source and target_anchors must have same feature dim"
            )
        _require_finite_tensor(source, field_name="source")
        _require_finite_tensor(target_anchors, field_name="target_anchors")
        n, m = source.shape[0], target_anchors.shape[0]
        if source_weights is None:
            a = torch.ones(n, dtype=source.dtype, device=source.device) / n
        else:
            if source_weights.shape != (n,):
                raise SinkhornError(
                    f"source_weights shape {source_weights.shape} != ({n},)"
                )
            _require_finite_tensor(source_weights, field_name="source_weights")
            a = source_weights
        if target_weights is None:
            b = torch.ones(m, dtype=source.dtype, device=source.device) / m
        else:
            if target_weights.shape != (m,):
                raise SinkhornError(
                    f"target_weights shape {target_weights.shape} != ({m},)"
                )
            _require_finite_tensor(target_weights, field_name="target_weights")
            b = target_weights
        # Cost = squared Euclidean.
        cost = torch.cdist(source, target_anchors, p=2).pow(2)
        plan, _, _ = sinkhorn_solve(
            a,
            b,
            cost,
            reg=self.spec.reg,
            max_iters=self.spec.max_iters,
            tol=self.spec.tol,
            eps=self.spec.eps,
            log_domain=self.spec.log_domain,
        )
        # Barycentric projection: z_j = Σ_k T[k, j] x_k / col_sum.
        col_sums = plan.sum(dim=0, keepdim=True).clamp_min(self.spec.eps)
        weights = plan / col_sums  # shape (n, m)
        z = weights.t() @ source  # (m, D)
        return z

    def cost(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        source_weights: torch.Tensor | None = None,
        target_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the regularised OT cost ``⟨T*, C⟩`` (0-D tensor)."""
        if source.dim() != 2 or target.dim() != 2:
            raise SinkhornError("source and target must be 2-D")
        if source.shape[1] != target.shape[1]:
            raise SinkhornError("source and target must have same feature dim")
        _require_finite_tensor(source, field_name="source")
        _require_finite_tensor(target, field_name="target")
        n, m = source.shape[0], target.shape[0]
        a = (
            source_weights
            if source_weights is not None
            else torch.ones(n, dtype=source.dtype, device=source.device) / n
        )
        b = (
            target_weights
            if target_weights is not None
            else torch.ones(m, dtype=source.dtype, device=source.device) / m
        )
        if source_weights is not None:
            _require_finite_tensor(source_weights, field_name="source_weights")
        if target_weights is not None:
            _require_finite_tensor(target_weights, field_name="target_weights")
        cost_mat = torch.cdist(source, target, p=2).pow(2)
        _, cost_value, _ = sinkhorn_solve(
            a,
            b,
            cost_mat,
            reg=self.spec.reg,
            max_iters=self.spec.max_iters,
            tol=self.spec.tol,
            eps=self.spec.eps,
            log_domain=self.spec.log_domain,
        )
        return cost_value

    def serialize_state(self) -> bytes:
        """Deterministic spec serialisation."""
        s = self.spec
        body = bytearray()
        body += SINKHORN_MAGIC
        body += struct.pack("<H", SINKHORN_SCHEMA_VERSION)
        body += struct.pack("<d", s.reg)
        body += struct.pack("<I", s.max_iters)
        body += struct.pack("<d", s.tol)
        body += struct.pack("<d", s.eps)
        body += struct.pack("<B", 1 if s.log_domain else 0)
        return bytes(body)

    @classmethod
    def deserialize_state(cls, payload: bytes) -> SinkhornOTMixer:
        """Inverse of :meth:`serialize_state`."""
        if len(payload) < 4 or payload[:4] != SINKHORN_MAGIC:
            raise SinkhornError(f"bad magic: {payload[:4]!r}")
        off = 4
        (version,) = struct.unpack_from("<H", payload, off)
        off += 2
        if version != SINKHORN_SCHEMA_VERSION:
            raise SinkhornError(f"unsupported schema version: {version}")
        (reg,) = struct.unpack_from("<d", payload, off)
        off += 8
        (max_iters,) = struct.unpack_from("<I", payload, off)
        off += 4
        (tol,) = struct.unpack_from("<d", payload, off)
        off += 8
        (eps,) = struct.unpack_from("<d", payload, off)
        off += 8
        (log_domain_byte,) = struct.unpack_from("<B", payload, off)
        return cls(
            SinkhornOTMixerSpec(
                reg=reg,
                max_iters=max_iters,
                tol=tol,
                eps=eps,
                log_domain=bool(log_domain_byte),
            )
        )


def estimate_param_bytes(spec: SinkhornOTMixerSpec) -> int:
    """Spec serialisation byte estimate."""
    return 4 + 2 + 8 + 4 + 8 + 8 + 1  # = 35


def _require_finite_tensor(tensor: torch.Tensor, *, field_name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise SinkhornError(f"{field_name} must be torch.Tensor")
    if not torch.isfinite(tensor).all():
        raise SinkhornError(f"{field_name} must contain finite values")


__all__ = [
    "DEFAULT_EPS",
    "DEFAULT_MAX_ITERS",
    "DEFAULT_REG",
    "DEFAULT_TOL",
    "SINKHORN_MAGIC",
    "SINKHORN_SCHEMA_VERSION",
    "SinkhornError",
    "SinkhornOTMixer",
    "SinkhornOTMixerSpec",
    "estimate_param_bytes",
    "sinkhorn_solve",
]
