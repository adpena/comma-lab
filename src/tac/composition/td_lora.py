"""Tropical-Decomposition LoRA (TD-LoRA) — composition/stacking primitive.

Tropical-Decomposition LoRA replaces the vanilla LoRA adapter

    W'(x) = W·x + A·B^T·x

with a *tropical max-plus* combination over ``k`` rank-(r/k) branches:

    W'(x) = max( W·x, W·x + Δ_1(x), …, W·x + Δ_k(x) )                   (E2.1)

where ``W`` is the frozen base and each ``Δ_i`` is a trainable low-rank
residual branch. This residual form is intentional: a freshly initialized
adapter must be an exact no-op against the frozen base so score movement is
apples-to-apples. Each adapter branch is parameterised in low-rank form with
rank ``r/k`` so the TOTAL trainable parameter count is identical to vanilla
LoRA at rank ``r``. The tropical maximum captures *piecewise-linear
refinements* that vanilla LoRA cannot: by
Zhang-Naitzat-Lim 2018, any continuous PWL function can be written as a
difference of tropical polynomials, and the minimum tropical
representation of an empirical fine-tune is typically more compact than
the dense LoRA representation.

Source memo: ``.omx/research/zen_state_frontier_deep_math_research_20260513.md``
§E2 ("Tropical-Decomposition LoRA on PR95 (TD-LoRA)").

Cross-references
----------------
- Tropical algebra derivations:
  ``.omx/research/zen_state_frontier_tropical_algebra_20260513.md``
- PR95 LoRA-DoRA scaffold (drop-in target):
  ``tac.substrates.pr95_lora_dora``
- Stack-of-stacks integration:
  ``tac.composition.stack_of_stacks.compose``

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
* This module produces a forward-pass module + serializable spec; it does
  NOT modify archive bytes by itself. Substrate integration must register
  a parser-section manifest entry per CLAUDE.md Catalog #124 before any
  ``score_claim=True``.
* Until paired ``[contest-CUDA]`` + ``[contest-CPU]`` anchors land on a
  TD-LoRA-equipped substrate, every result is ``score_claim=False``,
  ``promotion_eligible=False``, ``ready_for_exact_eval_dispatch=False``.

HNeRV parity discipline (13 lessons)
------------------------------------
1. Score-aware: the forward is differentiable; substrate trainer drives
   ``apply_eval_roundtrip=True`` and gradient-through-SegNet/PoseNet.
2. Export-first: the spec serialises deterministically; the substrate
   archive writer pulls bytes from ``serialize_state``.
3. Archive grammar: declared by the consuming substrate, not here.
4. Inflate: the inflate-side forward is the same nn.Module (≤ 350 LOC).
5. Architecture-as-renderer: this is a parameter ADAPTER; substrate
   integration must keep the renderer RGB-shaped at the outer boundary.
6. Score-domain Lagrangian: untouched.
7. Bolt-on ≤ 350 LOC: yes.
8. eval_roundtrip-aware: trainer responsibility.
9. Runtime closure: torch only (already required by substrate).
10-13. Standard substrate concerns; not violated by this primitive.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

# Wire-format magic for serialisation.
TD_LORA_MAGIC = b"TDL1"
TD_LORA_SCHEMA_VERSION = 1
TD_LORA_MAX_BRANCHES = 16
TD_LORA_MIN_BRANCHES = 1
TD_LORA_FLAG_BASE_WEIGHTS = 1 << 0
TD_LORA_TIE_GRAD_TEMPERATURE = 1.0


class TropicalLoRAError(ValueError):
    """Raised when a TD-LoRA spec or input shape is invalid."""


@dataclass(frozen=True)
class TropicalLoRASpec:
    """Specification for a Tropical-Decomposition LoRA adapter.

    Args:
        in_features: input dimensionality of the wrapped linear layer.
        out_features: output dimensionality.
        rank: total LoRA rank summed across the ``num_branches`` adapter
            branches. Each branch carries ``rank // num_branches``.
            ``rank == 0`` collapses TD-LoRA to the base linear (no
            adapters); kept as an explicit no-op for ablations.
        num_branches: ``k`` in equation E2.1 above. Must satisfy
            ``1 ≤ k ≤ 16`` and ``rank % k == 0`` (clean rank split).
        alpha: LoRA scaling factor (multiplies the adapter contribution
            uniformly before the tropical max). Defaults to ``rank /
            num_branches`` per the LoRA paper convention.
        include_base: whether the BASE linear ``W·x`` is one of the
            tropical-max branches. Defaults to ``True`` (matches the
            ``E2.1`` formulation; the base "branch 0" is always present
            when this is True).
    """

    in_features: int
    out_features: int
    rank: int
    num_branches: int = 2
    alpha: float | None = None
    include_base: bool = True

    def __post_init__(self) -> None:
        if self.in_features <= 0:
            raise TropicalLoRAError(
                f"in_features must be positive; got {self.in_features}"
            )
        if self.out_features <= 0:
            raise TropicalLoRAError(
                f"out_features must be positive; got {self.out_features}"
            )
        if self.rank < 0:
            raise TropicalLoRAError(
                f"rank must be ≥ 0; got {self.rank}"
            )
        if not (TD_LORA_MIN_BRANCHES <= self.num_branches <= TD_LORA_MAX_BRANCHES):
            raise TropicalLoRAError(
                f"num_branches must be in [{TD_LORA_MIN_BRANCHES}, "
                f"{TD_LORA_MAX_BRANCHES}]; got {self.num_branches}"
            )
        if self.rank % self.num_branches != 0:
            raise TropicalLoRAError(
                f"rank={self.rank} must divide evenly into "
                f"num_branches={self.num_branches}"
            )
        if self.alpha is not None and not math.isfinite(float(self.alpha)):
            raise TropicalLoRAError(f"alpha must be finite; got {self.alpha}")

    @property
    def per_branch_rank(self) -> int:
        return self.rank // self.num_branches

    @property
    def effective_alpha(self) -> float:
        if self.alpha is not None:
            return float(self.alpha)
        if self.num_branches == 0:
            return 1.0
        return float(self.per_branch_rank) if self.per_branch_rank > 0 else 1.0


class TropicalLoRAAdapter(nn.Module):
    """Tropical-Decomposition LoRA adapter.

    Wraps a (frozen) linear base ``W ∈ R^{out × in}`` with ``k``
    low-rank adapter branches and combines via tropical max-plus.

    Forward (with ``include_base=True``):

        y_base = (x @ W^T) + b_W                         # frozen
        Δ_i    = α · (x @ A_i^T) @ B_i^T + b_i           # branch i ∈ [1, k]
        y      = max(y_base, y_base + Δ_i, ...)          # broadcast over batch

    Note the maximum is *elementwise* over the output feature dimension —
    consistent with the tropical semiring (max as +, + as ·).
    """

    def __init__(self, spec: TropicalLoRASpec, base_linear: nn.Linear | None = None):
        super().__init__()
        self.spec = spec
        in_f = spec.in_features
        out_f = spec.out_features
        r = spec.per_branch_rank
        k = spec.num_branches

        if base_linear is not None:
            if base_linear.in_features != in_f or base_linear.out_features != out_f:
                raise TropicalLoRAError(
                    "base_linear shape must match spec "
                    f"(in={in_f}, out={out_f}); got "
                    f"(in={base_linear.in_features}, out={base_linear.out_features})"
                )
            self.base = base_linear
        else:
            self.base = nn.Linear(in_f, out_f, bias=True)

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.adapters_A = nn.ParameterList()
        self.adapters_B = nn.ParameterList()
        self.adapters_bias = nn.ParameterList()
        for _ in range(k):
            if r == 0:
                self.adapters_A.append(nn.Parameter(torch.zeros(in_f, 0)))
                self.adapters_B.append(nn.Parameter(torch.zeros(0, out_f)))
            else:
                A = torch.empty(in_f, r)
                B = torch.zeros(r, out_f)
                # LoRA-standard init: A ~ kaiming, B = 0 (so initial Δ = 0).
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                self.adapters_A.append(nn.Parameter(A))
                self.adapters_B.append(nn.Parameter(B))
            self.adapters_bias.append(nn.Parameter(torch.zeros(out_f)))

    @property
    def include_base(self) -> bool:
        return self.spec.include_base

    @property
    def num_branches(self) -> int:
        return self.spec.num_branches

    def adapter_branch_output(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Compute the i-th adapter branch output (no base, no max)."""
        A = self.adapters_A[i]
        B = self.adapters_B[i]
        bias = self.adapters_bias[i]
        if A.shape[1] == 0:  # rank-0 branch
            return bias.expand(*x.shape[:-1], -1)
        h = x @ A  # (..., in) @ (in, r) -> (..., r)
        h = h @ B  # (..., r) @ (r, out) -> (..., out)
        return self.spec.effective_alpha * h + bias

    def _tropical_residual(self, residuals: list[torch.Tensor]) -> torch.Tensor:
        """Hard tropical max with smooth tie gradients for trainable residuals."""

        stacked = torch.stack(residuals, dim=0)
        hard = stacked.max(dim=0).values
        if self.training and self.spec.rank > 0 and stacked.requires_grad:
            # Forward remains the exact hard tropical max. Backward uses a
            # centered log-sum-exp surrogate so the all-zero no-op init still
            # seeds gradients into adapter branches instead of tying to the
            # frozen base branch only.
            tau = TD_LORA_TIE_GRAD_TEMPERATURE
            soft = tau * (
                torch.logsumexp(stacked / tau, dim=0) - math.log(stacked.shape[0])
            )
            return hard + (soft - soft.detach())
        return hard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply tropical max-plus over base plus residual adapter branches."""
        if x.shape[-1] != self.spec.in_features:
            raise TropicalLoRAError(
                f"input last dim={x.shape[-1]} mismatches in_features="
                f"{self.spec.in_features}"
            )
        if self.include_base:
            base_out = self.base(x)
            residuals = [torch.zeros_like(base_out)]
            for i in range(self.num_branches):
                residuals.append(self.adapter_branch_output(x, i))
            return base_out + self._tropical_residual(residuals)

        branch_outputs = [
            self.adapter_branch_output(x, i) for i in range(self.num_branches)
        ]
        return torch.stack(branch_outputs, dim=0).max(dim=0).values

    def serialize_state(self, *, include_base_weights: bool | None = None) -> bytes:
        """Deterministic byte serialisation of the spec + adapter weights.

        Format:
            magic (4s) + version (B) + num_branches (B) + rank (H) +
            in_features (I) + out_features (I) + include_base (B) +
            flags (B) + alpha (f) + optional base_payload + adapter_payload

        ``include_base_weights`` defaults to ``True`` when the base branch is
        part of the tropical max. That preserves deterministic forward parity
        after deserialisation. Legacy/no-base payloads must be restored with
        an explicit caller-supplied ``base_linear``.
        """
        base_payload_enabled = (
            self.spec.include_base
            if include_base_weights is None
            else bool(include_base_weights)
        )
        flags = TD_LORA_FLAG_BASE_WEIGHTS if base_payload_enabled else 0
        header = struct.pack(
            "<4sBBHIIBBf",
            TD_LORA_MAGIC,
            TD_LORA_SCHEMA_VERSION,
            self.spec.num_branches,
            self.spec.rank,
            self.spec.in_features,
            self.spec.out_features,
            1 if self.spec.include_base else 0,
            flags,
            self.spec.effective_alpha,
        )
        base_bytes = bytearray()
        if base_payload_enabled:
            base_bytes.extend(
                self.base.weight.detach().cpu().contiguous().numpy().astype(
                    "float32"
                ).tobytes()
            )
            if self.base.bias is None:
                bias = torch.zeros(self.spec.out_features, dtype=torch.float32)
            else:
                bias = self.base.bias.detach().cpu().contiguous()
            base_bytes.extend(bias.numpy().astype("float32").tobytes())
        adapter_bytes = bytearray()
        for i in range(self.num_branches):
            adapter_bytes.extend(
                self.adapters_A[i].detach().cpu().contiguous().numpy().astype(
                    "float32"
                ).tobytes()
            )
            adapter_bytes.extend(
                self.adapters_B[i].detach().cpu().contiguous().numpy().astype(
                    "float32"
                ).tobytes()
            )
            adapter_bytes.extend(
                self.adapters_bias[i].detach().cpu().contiguous().numpy().astype(
                    "float32"
                ).tobytes()
            )
        return bytes(header) + bytes(base_bytes) + bytes(adapter_bytes)

    @classmethod
    def deserialize_state(
        cls,
        payload: bytes,
        *,
        base_linear: nn.Linear | None = None,
    ) -> TropicalLoRAAdapter:
        """Inverse of :meth:`serialize_state`."""
        header_struct = struct.Struct("<4sBBHIIBBf")
        if len(payload) < header_struct.size:
            raise TropicalLoRAError(
                f"payload too short ({len(payload)} < {header_struct.size})"
            )
        magic, version, k, r, in_f, out_f, include_base_u8, flags, alpha = (
            header_struct.unpack_from(payload, 0)
        )
        if magic != TD_LORA_MAGIC:
            raise TropicalLoRAError(
                f"magic mismatch: expected {TD_LORA_MAGIC!r} got {magic!r}"
            )
        if version != TD_LORA_SCHEMA_VERSION:
            raise TropicalLoRAError(
                f"version mismatch: expected {TD_LORA_SCHEMA_VERSION} got {version}"
            )
        spec = TropicalLoRASpec(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            num_branches=k,
            alpha=alpha,
            include_base=bool(include_base_u8),
        )
        offset = header_struct.size
        has_base_weights = bool(flags & TD_LORA_FLAG_BASE_WEIGHTS)
        unknown_flags = flags & ~TD_LORA_FLAG_BASE_WEIGHTS
        if unknown_flags:
            raise TropicalLoRAError(f"unsupported TD-LoRA payload flags: {unknown_flags}")
        if spec.include_base and not has_base_weights and base_linear is None:
            raise TropicalLoRAError(
                "payload omits base weights for include_base=True; pass "
                "base_linear=... to restore deterministic forward parity"
            )
        module = cls(spec, base_linear=None if has_base_weights else base_linear)
        if has_base_weights:
            base_weight, offset = _read_float32_tensor(
                payload,
                offset,
                (out_f, in_f),
                field_name="base.weight",
            )
            base_bias, offset = _read_float32_tensor(
                payload,
                offset,
                (out_f,),
                field_name="base.bias",
            )
            with torch.no_grad():
                module.base.weight.copy_(base_weight)
                if module.base.bias is not None:
                    module.base.bias.copy_(base_bias)
        per_branch_rank = spec.per_branch_rank
        for i in range(k):
            A, offset = _read_float32_tensor(
                payload,
                offset,
                (in_f, per_branch_rank),
                field_name=f"adapters_A[{i}]",
            )
            B, offset = _read_float32_tensor(
                payload,
                offset,
                (per_branch_rank, out_f),
                field_name=f"adapters_B[{i}]",
            )
            bias, offset = _read_float32_tensor(
                payload,
                offset,
                (out_f,),
                field_name=f"adapters_bias[{i}]",
            )
            if per_branch_rank > 0:
                with torch.no_grad():
                    module.adapters_A[i].copy_(A)
                    module.adapters_B[i].copy_(B)
            with torch.no_grad():
                module.adapters_bias[i].copy_(bias)
        if offset != len(payload):
            raise TropicalLoRAError(
                f"payload has {len(payload) - offset} trailing bytes"
            )
        return module


def estimate_param_bytes(spec: TropicalLoRASpec, dtype_bytes: int = 4) -> int:
    """Return number of bytes consumed by the trainable adapter weights.

    Matches LoRA's standard cost ``2 · in · r + r · out`` for vanilla
    (the rank goes through one A and one B); for TD-LoRA, each branch
    has its own (A_i, B_i, bias) and the TOTAL is
    ``k · (in · r/k + r/k · out + out) = in · r + r · out + k · out``
    (the extra ``k · out`` term is the per-branch bias).
    """
    r = spec.rank
    k = spec.num_branches
    in_f = spec.in_features
    out_f = spec.out_features
    return (in_f * r + r * out_f + k * out_f) * dtype_bytes


def tropical_max_plus(branches: Iterable[torch.Tensor]) -> torch.Tensor:
    """Tropical max-plus reduction over a list of pre-broadcasted tensors.

    Convenience function for callers who already have per-branch
    pre-bias forwards and want to apply the tropical max manually
    (e.g. for inspection / ablation harnesses).
    """
    stacked = torch.stack(list(branches), dim=0)
    return stacked.max(dim=0).values


def _read_float32_tensor(
    payload: bytes,
    offset: int,
    shape: tuple[int, ...],
    *,
    field_name: str,
) -> tuple[torch.Tensor, int]:
    numel = math.prod(shape)
    nbytes = numel * 4
    end = offset + nbytes
    if end > len(payload):
        raise TropicalLoRAError(
            f"payload truncated while reading {field_name}: "
            f"need {nbytes} bytes at offset {offset}, have {len(payload) - offset}"
        )
    if numel == 0:
        return torch.empty(shape, dtype=torch.float32), end
    tensor = torch.frombuffer(
        bytearray(payload[offset:end]), dtype=torch.float32
    ).reshape(shape).clone()
    return tensor, end


__all__ = [
    "TD_LORA_FLAG_BASE_WEIGHTS",
    "TD_LORA_MAGIC",
    "TD_LORA_MAX_BRANCHES",
    "TD_LORA_MIN_BRANCHES",
    "TD_LORA_SCHEMA_VERSION",
    "TropicalLoRAAdapter",
    "TropicalLoRAError",
    "TropicalLoRASpec",
    "estimate_param_bytes",
    "tropical_max_plus",
]
