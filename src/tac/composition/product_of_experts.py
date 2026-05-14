# SPDX-License-Identifier: MIT
"""Product-of-Experts Gating — composition/stacking primitive.

Hinton 1999 ("Products of Experts", https://www.cs.toronto.edu/~hinton/poe.html)
introduced compositions of distributions via the product rule:

    p(x) = (1/Z) · ∏_k p_k(x)^{α_k}                                      (P.1)

Each expert ``p_k`` is a "soft constraint" on the data; the product is
"sharpened" — only points HIGH in EVERY expert's density survive. Compare
to mixture models (``p(x) = Σ_k λ_k p_k(x)``) which add (any expert
suffices); products multiply (every expert must agree).

For weighted log-density combination (which is what we usually want):

    log p(x) = Σ_k α_k log p_k(x) - log Z(α)                            (P.2)

This is the canonical pattern when combining K renderers, each producing
RGB-shaped output: pick the per-pair expert assignment via a soft gate, OR
average log-densities for ensembling.

Source memos:
- Hinton 1999 PoE concept.
- ``.omx/research/ancient_elder_polymath_research_20260513.md`` cross-ref.

Cross-references
----------------
- Bregman mixing (``tac.composition.bregman_mixing``) — KL projection vs
  product combination.
- Hypernetwork (``tac.composition.hypernetwork``) — provides the experts.
- Distillation chain (``tac.composition.distillation_chain``) — chains
  PoE outputs.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
This module produces a forward-pass module + serialisable spec; it does not
modify archive bytes by itself. Substrate integration must register a
parser-section manifest entry per CLAUDE.md Catalog #124 before any
``score_claim=True``. Until paired ``[contest-CUDA]`` + ``[contest-CPU]``
anchors land on a PoE-equipped substrate, every result is
``score_claim=False``, ``promotion_eligible=False``,
``ready_for_exact_eval_dispatch=False``.

HNeRV parity discipline (13 lessons)
------------------------------------
1. Score-aware: PoE forward is differentiable; trainer drives
   apply_eval_roundtrip + scorer gradient.
2. Export-first: :meth:`ProductOfExpertsComposer.serialize_state`.
3-6. Substrate concerns; not violated.
7. Bolt-on ≤ 250 LOC.
8-13. Standard substrate concerns; not violated.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

POE_MAGIC = b"POE1"
POE_SCHEMA_VERSION = 1
DEFAULT_TEMPERATURE = 1.0


class ProductOfExpertsError(ValueError):
    """Raised when a PoE spec or input is invalid."""


@dataclass(frozen=True)
class ProductOfExpertsSpec:
    """Specification for a Product-of-Experts composer.

    Args:
        num_experts: number of experts ``K`` (must be ≥ 1).
        per_expert_alpha: optional fixed weights ``α_k``. ``None`` →
            uniform (each weight = 1). Length must equal ``num_experts``.
        temperature: gating temperature for soft-gating mode (smaller →
            sharper). Must be > 0.
        mode: one of {"log_density_sum", "soft_gate", "hard_gate"}.
            - ``"log_density_sum"``: combine via Eq. P.2 (additive in log).
            - ``"soft_gate"``: assign per-position soft expert weights via
              a softmax over per-expert log-likelihoods.
            - ``"hard_gate"``: per-position argmax (non-differentiable;
              use only at inference).
    """

    num_experts: int = 2
    per_expert_alpha: tuple[float, ...] | None = None
    temperature: float = DEFAULT_TEMPERATURE
    mode: str = "log_density_sum"

    def __post_init__(self) -> None:
        if self.num_experts < 1:
            raise ProductOfExpertsError(
                f"num_experts must be ≥ 1, got {self.num_experts}"
            )
        if self.per_expert_alpha is not None:
            if len(self.per_expert_alpha) != self.num_experts:
                raise ProductOfExpertsError(
                    f"per_expert_alpha length {len(self.per_expert_alpha)} "
                    f"!= num_experts {self.num_experts}"
                )
            if any(not math.isfinite(float(a)) for a in self.per_expert_alpha):
                raise ProductOfExpertsError("per_expert_alpha entries must be finite")
            if any(a < 0 for a in self.per_expert_alpha):
                raise ProductOfExpertsError(
                    "per_expert_alpha entries must be non-negative"
                )
        if self.temperature <= 0 or not math.isfinite(float(self.temperature)):
            raise ProductOfExpertsError(
                f"temperature must be positive, got {self.temperature}"
            )
        if self.mode not in {"log_density_sum", "soft_gate", "hard_gate"}:
            raise ProductOfExpertsError(f"unknown mode: {self.mode!r}")


class ProductOfExpertsComposer(nn.Module):
    """Combine K experts via product-of-experts rule.

    The composer accepts per-expert log-densities or per-expert outputs and
    fuses them per the spec mode.

    Example
    -------
    >>> import torch
    >>> from tac.composition.product_of_experts import (
    ...     ProductOfExpertsComposer, ProductOfExpertsSpec,
    ... )
    >>> spec = ProductOfExpertsSpec(num_experts=2, mode="log_density_sum")
    >>> composer = ProductOfExpertsComposer(spec)
    >>> log_p1 = torch.tensor([0.0, -1.0])
    >>> log_p2 = torch.tensor([-2.0, 0.0])
    >>> composer.combine_log_densities([log_p1, log_p2])
    tensor([-2., -1.])
    """

    def __init__(self, spec: ProductOfExpertsSpec) -> None:
        super().__init__()
        self.spec = spec
        if spec.per_expert_alpha is None:
            alpha = torch.ones(spec.num_experts)
        else:
            alpha = torch.tensor(spec.per_expert_alpha, dtype=torch.float32)
        # Learnable alpha defaults from spec; tag as buffer so deterministic.
        self.register_buffer("alpha", alpha, persistent=True)

    def combine_log_densities(
        self,
        log_densities: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Return ``Σ_k α_k log p_k(x)`` (Eq. P.2 without log Z normaliser).

        The normaliser is omitted because it depends on the support
        partition; for ranking/scoring it is constant and irrelevant.
        """
        if len(log_densities) != self.spec.num_experts:
            raise ProductOfExpertsError(
                f"expected {self.spec.num_experts} log_densities, got {len(log_densities)}"
            )
        _validate_tensor_sequence(
            log_densities,
            field_name="log_densities",
            require_float=True,
        )
        stacked = torch.stack(list(log_densities), dim=0)  # (K, ...)
        alpha = self.alpha.to(dtype=stacked.dtype, device=stacked.device)
        view = alpha.view((-1,) + (1,) * (stacked.dim() - 1))
        return (view * stacked).sum(dim=0)

    def soft_gate(
        self,
        per_expert_log_likelihood: Sequence[torch.Tensor],
        per_expert_outputs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Per-position soft mixture using PoE-derived gating weights.

        Gating weights:
            ``g_k(x) = softmax_k(α_k log p_k(x) / T)``

        Output:
            ``y(x) = Σ_k g_k(x) · f_k(x)``

        Args:
            per_expert_log_likelihood: K tensors of shape ``(...,)``.
            per_expert_outputs: K tensors of shape ``(..., F)`` where F
                may be 1 or arbitrary (must agree across experts).

        Returns:
            Mixture output of shape ``(..., F)``.
        """
        if len(per_expert_log_likelihood) != self.spec.num_experts:
            raise ProductOfExpertsError(
                f"expected {self.spec.num_experts} log-likelihoods, got "
                f"{len(per_expert_log_likelihood)}"
            )
        if len(per_expert_outputs) != self.spec.num_experts:
            raise ProductOfExpertsError(
                f"expected {self.spec.num_experts} outputs, got "
                f"{len(per_expert_outputs)}"
            )
        logp_shape = _validate_tensor_sequence(
            per_expert_log_likelihood,
            field_name="per_expert_log_likelihood",
            require_float=True,
        )
        output_shape = _validate_tensor_sequence(
            per_expert_outputs,
            field_name="per_expert_outputs",
            require_float=True,
        )
        _validate_output_shape_matches_logp(output_shape, logp_shape)
        stacked_logp = torch.stack(list(per_expert_log_likelihood), dim=0)  # (K, ...)
        stacked_outs = torch.stack(list(per_expert_outputs), dim=0)  # (K, ..., F)
        alpha = self.alpha.to(dtype=stacked_logp.dtype, device=stacked_logp.device)
        view = alpha.view((-1,) + (1,) * (stacked_logp.dim() - 1))
        weighted = view * stacked_logp / self.spec.temperature
        gates = torch.softmax(weighted, dim=0)  # (K, ...)
        gate_view = gates.unsqueeze(-1)  # (K, ..., 1)
        return (gate_view * stacked_outs).sum(dim=0)

    def hard_gate(
        self,
        per_expert_log_likelihood: Sequence[torch.Tensor],
        per_expert_outputs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Per-position argmax expert selection (non-differentiable)."""
        if len(per_expert_log_likelihood) != self.spec.num_experts:
            raise ProductOfExpertsError(
                f"expected {self.spec.num_experts} log-likelihoods, got "
                f"{len(per_expert_log_likelihood)}"
            )
        if len(per_expert_outputs) != self.spec.num_experts:
            raise ProductOfExpertsError(
                f"expected {self.spec.num_experts} outputs, got "
                f"{len(per_expert_outputs)}"
            )
        logp_shape = _validate_tensor_sequence(
            per_expert_log_likelihood,
            field_name="per_expert_log_likelihood",
            require_float=True,
        )
        output_shape = _validate_tensor_sequence(
            per_expert_outputs,
            field_name="per_expert_outputs",
            require_float=True,
        )
        _validate_output_shape_matches_logp(output_shape, logp_shape)
        stacked_logp = torch.stack(list(per_expert_log_likelihood), dim=0)
        stacked_outs = torch.stack(list(per_expert_outputs), dim=0)
        alpha = self.alpha.to(dtype=stacked_logp.dtype, device=stacked_logp.device)
        view = alpha.view((-1,) + (1,) * (stacked_logp.dim() - 1))
        weighted = view * stacked_logp
        idx = weighted.argmax(dim=0)  # (...,)
        # Gather along K dim.
        idx_expanded = idx.unsqueeze(0).unsqueeze(-1).expand(1, *idx.shape, stacked_outs.shape[-1])
        return torch.gather(stacked_outs, dim=0, index=idx_expanded).squeeze(0)

    def serialize_state(self) -> bytes:
        """Deterministic serialisation."""
        body = bytearray()
        body += POE_MAGIC
        body += struct.pack("<H", POE_SCHEMA_VERSION)
        body += struct.pack("<I", self.spec.num_experts)
        if self.spec.per_expert_alpha is None:
            body += struct.pack("<H", 0)
        else:
            body += struct.pack("<H", len(self.spec.per_expert_alpha))
            for a in self.spec.per_expert_alpha:
                body += struct.pack("<f", float(a))
        body += struct.pack("<d", self.spec.temperature)
        mode_bytes = self.spec.mode.encode("ascii")
        body += struct.pack("<H", len(mode_bytes))
        body += mode_bytes
        return bytes(body)

    @classmethod
    def deserialize_state(cls, payload: bytes) -> ProductOfExpertsComposer:
        """Inverse of :meth:`serialize_state`."""
        if len(payload) < 4 or payload[:4] != POE_MAGIC:
            raise ProductOfExpertsError(f"bad magic: {payload[:4]!r}")
        off = 4
        (version,) = struct.unpack_from("<H", payload, off)
        off += 2
        if version != POE_SCHEMA_VERSION:
            raise ProductOfExpertsError(f"unsupported schema version: {version}")
        (num_experts,) = struct.unpack_from("<I", payload, off)
        off += 4
        (num_alpha,) = struct.unpack_from("<H", payload, off)
        off += 2
        if num_alpha == 0:
            alpha: tuple[float, ...] | None = None
        else:
            alpha = tuple(
                struct.unpack_from("<f", payload, off + 4 * i)[0]
                for i in range(num_alpha)
            )
            off += 4 * num_alpha
        (temperature,) = struct.unpack_from("<d", payload, off)
        off += 8
        (mode_len,) = struct.unpack_from("<H", payload, off)
        off += 2
        mode = payload[off : off + mode_len].decode("ascii")
        spec = ProductOfExpertsSpec(
            num_experts=num_experts,
            per_expert_alpha=alpha,
            temperature=temperature,
            mode=mode,
        )
        return cls(spec)


def _validate_tensor_sequence(
    tensors: Sequence[torch.Tensor],
    *,
    field_name: str,
    require_float: bool,
) -> torch.Size:
    if not tensors:
        raise ProductOfExpertsError(f"{field_name} must be non-empty")
    shape = tensors[0].shape
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise ProductOfExpertsError(f"{field_name}[{i}] must be torch.Tensor")
        if tensor.shape != shape:
            raise ProductOfExpertsError(
                f"{field_name}[{i}].shape {tensor.shape} != "
                f"{field_name}[0].shape {shape}"
            )
        if require_float and not torch.is_floating_point(tensor):
            raise ProductOfExpertsError(f"{field_name}[{i}] must be floating-point")
        if not torch.isfinite(tensor).all():
            raise ProductOfExpertsError(f"{field_name}[{i}] must contain finite values")
    return shape


def _validate_output_shape_matches_logp(
    output_shape: torch.Size,
    logp_shape: torch.Size,
) -> None:
    if len(output_shape) != len(logp_shape) + 1:
        raise ProductOfExpertsError(
            "per_expert_outputs must have shape log_likelihood_shape + (F,); "
            f"got output shape {tuple(output_shape)} and log shape {tuple(logp_shape)}"
        )
    if tuple(output_shape[:-1]) != tuple(logp_shape):
        raise ProductOfExpertsError(
            "per_expert_outputs leading dims must match log-likelihood shape; "
            f"got {tuple(output_shape[:-1])} vs {tuple(logp_shape)}"
        )
    if output_shape[-1] <= 0:
        raise ProductOfExpertsError("per_expert_outputs feature dim must be positive")


__all__ = [
    "POE_MAGIC",
    "POE_SCHEMA_VERSION",
    "ProductOfExpertsComposer",
    "ProductOfExpertsError",
    "ProductOfExpertsSpec",
]
