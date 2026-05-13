"""Canonical composition primitives for checkpoint and vector mixtures.

This module is intentionally small and substrate-agnostic.  It provides the
math surfaces that several frontier lanes need before they can become archive
or trainer integrations:

- diagonal-Gaussian W2 barycenters for checkpoint/model-soup mixtures;
- MERA-style hierarchy metadata for multi-scale checkpoint contracts;
- Bregman and Sinkhorn helpers for deterministic tensor/vector mixing.

None of these helpers claims a contest score or marks a candidate dispatchable.
They produce typed tensors and metadata that a substrate can wire into an
export, trainer, or planner after it supplies archive/runtime custody.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from tac.composition.registry import (
    PLANNING_ONLY,
    PROMOTION_ELIGIBLE,
    READY_FOR_EXACT_EVAL_DISPATCH,
    SCORE_CLAIM,
)

FRONTIER_PRIMITIVES_SCHEMA_VERSION = "tac_composition_frontier_primitives_v1"

BregmanDivergence = Literal[
    "squared_euclidean",
    "kl_forward",
    "kl_reverse",
]


class CompositionPrimitiveError(ValueError):
    """Raised when composition primitive inputs are malformed."""


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes for metadata payloads.

    The function is strict about NaN/Inf so serialized metadata can be hashed
    and compared across machines without accidental JSON extensions.
    """

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor including dtype, shape, and contiguous value bytes."""

    if not isinstance(tensor, torch.Tensor):
        raise CompositionPrimitiveError(
            f"tensor_sha256 expects torch.Tensor, got {type(tensor).__name__}"
        )
    t = tensor.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(t.dtype), "shape": list(t.shape)}
    )
    h = hashlib.sha256()
    h.update(header)
    raw = (
        t.view(torch.int16).numpy().tobytes()
        if t.dtype is torch.bfloat16
        else t.numpy().tobytes()
    )
    h.update(raw)
    return h.hexdigest()


def metadata_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a deterministic JSON metadata payload."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def normalize_weights(
    weights: Sequence[float] | None,
    count: int,
    *,
    field_name: str = "weights",
) -> tuple[float, ...]:
    """Normalize non-negative finite weights to sum to one.

    ``None`` means uniform weights.  Zero-total, negative, non-finite, and
    length-mismatched weights fail closed.
    """

    if count <= 0:
        raise CompositionPrimitiveError(f"count must be positive; got {count}")
    if weights is None:
        return tuple(1.0 / count for _ in range(count))
    if len(weights) != count:
        raise CompositionPrimitiveError(
            f"{field_name} length {len(weights)} != expected count {count}"
        )
    vals = tuple(float(w) for w in weights)
    if any((not math.isfinite(w)) for w in vals):
        raise CompositionPrimitiveError(f"{field_name} must be finite")
    if any(w < 0.0 for w in vals):
        raise CompositionPrimitiveError(f"{field_name} must be non-negative")
    total = sum(vals)
    if total <= 0.0:
        raise CompositionPrimitiveError(f"{field_name} must have positive sum")
    return tuple(w / total for w in vals)


def _require_finite_tensor(tensor: torch.Tensor, *, field_name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise CompositionPrimitiveError(
            f"{field_name} must be torch.Tensor, got {type(tensor).__name__}"
        )
    if tensor.numel() == 0:
        raise CompositionPrimitiveError(f"{field_name} must be non-empty")
    if not torch.isfinite(tensor).all():
        raise CompositionPrimitiveError(f"{field_name} must contain only finite values")


def _validate_same_shape(
    tensors: Sequence[torch.Tensor],
    *,
    field_name: str,
) -> tuple[int, ...]:
    if not tensors:
        raise CompositionPrimitiveError(f"{field_name} must be non-empty")
    ref_shape = tuple(tensors[0].shape)
    for idx, tensor in enumerate(tensors):
        _require_finite_tensor(tensor, field_name=f"{field_name}[{idx}]")
        if tuple(tensor.shape) != ref_shape:
            raise CompositionPrimitiveError(
                f"{field_name}[{idx}] shape {tuple(tensor.shape)} != {ref_shape}"
            )
    return ref_shape


@dataclass(frozen=True, slots=True)
class DiagonalGaussian:
    """Diagonal Gaussian approximation for one checkpoint/tensor vector."""

    mean: torch.Tensor
    variance: torch.Tensor
    label: str = ""

    def __post_init__(self) -> None:
        _require_finite_tensor(self.mean, field_name="mean")
        _require_finite_tensor(self.variance, field_name="variance")
        if tuple(self.mean.shape) != tuple(self.variance.shape):
            raise CompositionPrimitiveError(
                f"mean shape {tuple(self.mean.shape)} != variance shape "
                f"{tuple(self.variance.shape)}"
            )
        if (self.variance < 0).any():
            raise CompositionPrimitiveError("variance must be non-negative")

    def to_metadata(self) -> dict[str, Any]:
        """Return deterministic metadata without embedding raw tensor values."""

        return {
            "label": self.label,
            "mean_sha256": tensor_sha256(self.mean),
            "shape": list(self.mean.shape),
            "variance_sha256": tensor_sha256(self.variance),
        }


@dataclass(frozen=True, slots=True)
class CheckpointBarycenter:
    """Barycenter result for a checkpoint state-dict family."""

    mean_state: Mapping[str, torch.Tensor]
    variance_state: Mapping[str, torch.Tensor]
    weights: tuple[float, ...]
    source_labels: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        keys = tuple(sorted(self.mean_state))
        return {
            "schema_version": FRONTIER_PRIMITIVES_SCHEMA_VERSION,
            "kind": "checkpoint_diagonal_gaussian_barycenter",
            "planning_only": PLANNING_ONLY,
            "score_claim": SCORE_CLAIM,
            "promotion_eligible": PROMOTION_ELIGIBLE,
            "ready_for_exact_eval_dispatch": READY_FOR_EXACT_EVAL_DISPATCH,
            "keys": list(keys),
            "source_labels": list(self.source_labels),
            "weights": list(self.weights),
            "mean_sha256": {k: tensor_sha256(self.mean_state[k]) for k in keys},
            "variance_sha256": {
                k: tensor_sha256(self.variance_state[k]) for k in keys
            },
        }

    def metadata_sha256(self) -> str:
        return metadata_sha256(self.to_metadata())


def wasserstein_diagonal_gaussian_barycenter(
    gaussians: Sequence[DiagonalGaussian],
    weights: Sequence[float] | None = None,
) -> DiagonalGaussian:
    """Compute the W2 barycenter of diagonal Gaussians.

    For diagonal Gaussians, the W2 barycenter has weighted-mean center and
    per-coordinate standard deviation equal to the weighted mean of standard
    deviations.  Variance is therefore ``(sum_i w_i sqrt(var_i)) ** 2``.
    """

    if not gaussians:
        raise CompositionPrimitiveError("gaussians must be non-empty")
    shape = tuple(gaussians[0].mean.shape)
    for idx, gaussian in enumerate(gaussians):
        if tuple(gaussian.mean.shape) != shape:
            raise CompositionPrimitiveError(
                f"gaussians[{idx}] shape {tuple(gaussian.mean.shape)} != {shape}"
            )
    w = normalize_weights(weights, len(gaussians))
    device = gaussians[0].mean.device
    dtype = gaussians[0].mean.dtype
    w_t = torch.tensor(w, dtype=torch.float64, device=device)
    means = torch.stack([g.mean.to(device=device, dtype=torch.float64) for g in gaussians])
    variances = torch.stack(
        [g.variance.to(device=device, dtype=torch.float64) for g in gaussians]
    )
    view_shape = (-1,) + (1,) * (means.ndim - 1)
    mean = (means * w_t.view(view_shape)).sum(dim=0)
    std = variances.clamp_min(0.0).sqrt()
    variance = (std * w_t.view(view_shape)).sum(dim=0).square()
    return DiagonalGaussian(
        mean=mean.to(dtype=dtype),
        variance=variance.to(dtype=dtype),
        label="w2_diagonal_barycenter",
    )


def checkpoint_diagonal_gaussian_barycenter(
    checkpoints: Sequence[Mapping[str, torch.Tensor]],
    variances: Sequence[Mapping[str, torch.Tensor]] | None = None,
    weights: Sequence[float] | None = None,
    source_labels: Sequence[str] | None = None,
) -> CheckpointBarycenter:
    """Compute a deterministic per-key diagonal-Gaussian checkpoint barycenter."""

    if not checkpoints:
        raise CompositionPrimitiveError("checkpoints must be non-empty")
    keys = tuple(sorted(checkpoints[0]))
    if not keys:
        raise CompositionPrimitiveError("checkpoints must contain at least one tensor")
    for idx, checkpoint in enumerate(checkpoints):
        if tuple(sorted(checkpoint)) != keys:
            raise CompositionPrimitiveError(
                f"checkpoints[{idx}] keys do not match reference keys"
            )
    if variances is not None:
        if len(variances) != len(checkpoints):
            raise CompositionPrimitiveError("variances length must match checkpoints")
        for idx, variance in enumerate(variances):
            if tuple(sorted(variance)) != keys:
                raise CompositionPrimitiveError(
                    f"variances[{idx}] keys do not match reference keys"
                )
    labels = (
        tuple(source_labels)
        if source_labels is not None
        else tuple(f"checkpoint_{i}" for i in range(len(checkpoints)))
    )
    if len(labels) != len(checkpoints):
        raise CompositionPrimitiveError("source_labels length must match checkpoints")
    w = normalize_weights(weights, len(checkpoints))

    mean_state: dict[str, torch.Tensor] = {}
    variance_state: dict[str, torch.Tensor] = {}
    for key in keys:
        gaussians: list[DiagonalGaussian] = []
        for idx, checkpoint in enumerate(checkpoints):
            tensor = checkpoint[key]
            variance = (
                torch.zeros_like(tensor, dtype=tensor.dtype)
                if variances is None
                else variances[idx][key]
            )
            gaussians.append(DiagonalGaussian(tensor, variance, label=labels[idx]))
        bary = wasserstein_diagonal_gaussian_barycenter(gaussians, w)
        mean_state[key] = bary.mean
        variance_state[key] = bary.variance
    return CheckpointBarycenter(
        mean_state=mean_state,
        variance_state=variance_state,
        weights=w,
        source_labels=labels,
    )


@dataclass(frozen=True, slots=True)
class MERALevelMetadata:
    """One level in a MERA-style hierarchy contract."""

    tensor_name: str
    level: int
    input_shape: tuple[int, ...]
    effective_shape: tuple[int, ...]
    bond_dim: int

    def __post_init__(self) -> None:
        if not self.tensor_name:
            raise CompositionPrimitiveError("tensor_name cannot be empty")
        if self.level < 0:
            raise CompositionPrimitiveError("level must be non-negative")
        if not self.input_shape or any(dim <= 0 for dim in self.input_shape):
            raise CompositionPrimitiveError("input_shape must contain positive dims")
        if not self.effective_shape or any(dim <= 0 for dim in self.effective_shape):
            raise CompositionPrimitiveError(
                "effective_shape must contain positive dims"
            )
        if self.bond_dim <= 0:
            raise CompositionPrimitiveError("bond_dim must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bond_dim": self.bond_dim,
            "effective_shape": list(self.effective_shape),
            "input_shape": list(self.input_shape),
            "level": self.level,
            "tensor_name": self.tensor_name,
        }


@dataclass(frozen=True, slots=True)
class MERAHierarchyMetadata:
    """Serializable MERA-style hierarchy contract for checkpoint mixtures."""

    hierarchy_id: str
    source_checkpoint_ids: tuple[str, ...]
    levels: tuple[MERALevelMetadata, ...]
    research_only: bool = True
    score_claim: bool = SCORE_CLAIM
    promotion_eligible: bool = PROMOTION_ELIGIBLE
    ready_for_exact_eval_dispatch: bool = READY_FOR_EXACT_EVAL_DISPATCH
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.hierarchy_id:
            raise CompositionPrimitiveError("hierarchy_id cannot be empty")
        if not self.source_checkpoint_ids:
            raise CompositionPrimitiveError("source_checkpoint_ids cannot be empty")
        if not self.levels:
            raise CompositionPrimitiveError("levels cannot be empty")
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise CompositionPrimitiveError(
                "MERA hierarchy metadata cannot claim score or dispatch readiness"
            )
        seen: set[tuple[str, int]] = set()
        for level in self.levels:
            key = (level.tensor_name, level.level)
            if key in seen:
                raise CompositionPrimitiveError(
                    f"duplicate MERA level for tensor={level.tensor_name!r} "
                    f"level={level.level}"
                )
            seen.add(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FRONTIER_PRIMITIVES_SCHEMA_VERSION,
            "hierarchy_id": self.hierarchy_id,
            "kind": "mera_hierarchy_metadata",
            "levels": [level.to_dict() for level in self.levels],
            "metadata": dict(self.metadata),
            "planning_only": PLANNING_ONLY,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "source_checkpoint_ids": list(self.source_checkpoint_ids),
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    def sha256(self) -> str:
        return metadata_sha256(self.to_dict())


def build_mera_hierarchy_metadata(
    tensor_shapes: Mapping[str, Sequence[int]],
    *,
    source_checkpoint_ids: Sequence[str],
    max_bond_dim: int,
    hierarchy_id: str = "mera_checkpoint_mixture",
    metadata: Mapping[str, str] | None = None,
) -> MERAHierarchyMetadata:
    """Build deterministic MERA-style hierarchy metadata from tensor shapes."""

    if max_bond_dim <= 0:
        raise CompositionPrimitiveError("max_bond_dim must be positive")
    levels: list[MERALevelMetadata] = []
    for tensor_name in sorted(tensor_shapes):
        shape = tuple(int(dim) for dim in tensor_shapes[tensor_name])
        if not shape or any(dim <= 0 for dim in shape):
            raise CompositionPrimitiveError(
                f"invalid shape for tensor {tensor_name!r}: {shape}"
            )
        depth = max(1, math.ceil(math.log2(max(shape))))
        for level in range(depth):
            scale = 2**level
            effective = tuple(max(1, math.ceil(dim / scale)) for dim in shape)
            bond_dim = min(max_bond_dim, max(1, min(effective)))
            levels.append(
                MERALevelMetadata(
                    tensor_name=tensor_name,
                    level=level,
                    input_shape=shape,
                    effective_shape=effective,
                    bond_dim=bond_dim,
                )
            )
    return MERAHierarchyMetadata(
        hierarchy_id=hierarchy_id,
        source_checkpoint_ids=tuple(source_checkpoint_ids),
        levels=tuple(levels),
        metadata=tuple(sorted((metadata or {}).items())),
    )


def bregman_barycenter(
    vectors: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    divergence: BregmanDivergence = "squared_euclidean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute a deterministic Bregman barycenter for equal-shaped tensors.

    ``squared_euclidean`` is the weighted arithmetic mean.
    ``kl_forward`` minimizes ``sum_i w_i KL(x || p_i)`` on the simplex and
    returns the normalized weighted geometric mean.
    ``kl_reverse`` minimizes ``sum_i w_i KL(p_i || x)`` on the simplex and
    returns the normalized weighted arithmetic mean of normalized inputs.
    """

    _validate_same_shape(vectors, field_name="vectors")
    if eps <= 0.0 or not math.isfinite(eps):
        raise CompositionPrimitiveError("eps must be positive and finite")
    w = normalize_weights(weights, len(vectors))
    device = vectors[0].device
    dtype = vectors[0].dtype
    w_t = torch.tensor(w, dtype=torch.float64, device=device)
    stacked = torch.stack([v.to(device=device, dtype=torch.float64) for v in vectors])
    view_shape = (-1,) + (1,) * (stacked.ndim - 1)
    if divergence == "squared_euclidean":
        return (stacked * w_t.view(view_shape)).sum(dim=0).to(dtype=dtype)
    if divergence not in ("kl_forward", "kl_reverse"):
        raise CompositionPrimitiveError(f"unsupported divergence {divergence!r}")
    if (stacked < 0).any():
        raise CompositionPrimitiveError("KL barycenter inputs must be non-negative")
    flat = stacked.reshape(stacked.shape[0], -1)
    totals = flat.sum(dim=1, keepdim=True)
    if (totals <= 0).any():
        raise CompositionPrimitiveError("KL barycenter inputs must have positive mass")
    probs = (flat / totals).clamp_min(eps)
    probs = probs / probs.sum(dim=1, keepdim=True)
    if divergence == "kl_reverse":
        out = (probs * w_t.view(-1, 1)).sum(dim=0)
    else:
        out = torch.exp((probs.log() * w_t.view(-1, 1)).sum(dim=0))
    out = out / out.sum().clamp_min(eps)
    return out.reshape_as(vectors[0]).to(dtype=dtype)


@dataclass(frozen=True, slots=True)
class SinkhornResult:
    """Result of entropic optimal-transport mixing."""

    plan: torch.Tensor
    row_marginal: torch.Tensor
    col_marginal: torch.Tensor
    iterations: int
    converged: bool


def _normalize_mass_vector(vector: torch.Tensor, *, field_name: str) -> torch.Tensor:
    _require_finite_tensor(vector, field_name=field_name)
    if vector.ndim != 1:
        raise CompositionPrimitiveError(f"{field_name} must be 1-D")
    if (vector < 0).any():
        raise CompositionPrimitiveError(f"{field_name} must be non-negative")
    total = vector.sum()
    if total <= 0:
        raise CompositionPrimitiveError(f"{field_name} must have positive mass")
    return vector.to(dtype=torch.float64) / total.to(dtype=torch.float64)


def sinkhorn_transport_plan(
    source: torch.Tensor,
    target: torch.Tensor,
    cost: torch.Tensor,
    *,
    epsilon: float = 0.05,
    max_iters: int = 200,
    tol: float = 1e-9,
) -> SinkhornResult:
    """Compute a deterministic entropic OT plan with Sinkhorn iterations."""

    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise CompositionPrimitiveError("epsilon must be positive and finite")
    if max_iters < 1:
        raise CompositionPrimitiveError("max_iters must be >= 1")
    if tol <= 0.0 or not math.isfinite(tol):
        raise CompositionPrimitiveError("tol must be positive and finite")
    a = _normalize_mass_vector(source, field_name="source")
    b = _normalize_mass_vector(target, field_name="target")
    _require_finite_tensor(cost, field_name="cost")
    if tuple(cost.shape) != (a.numel(), b.numel()):
        raise CompositionPrimitiveError(
            f"cost shape {tuple(cost.shape)} != {(a.numel(), b.numel())}"
        )
    c = cost.to(dtype=torch.float64)
    kernel = torch.exp(-c / float(epsilon)).clamp_min(torch.finfo(torch.float64).tiny)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    converged = False
    last_err = math.inf
    for iteration in range(1, max_iters + 1):
        u = a / (kernel @ v).clamp_min(torch.finfo(torch.float64).tiny)
        v = b / (kernel.t() @ u).clamp_min(torch.finfo(torch.float64).tiny)
        if iteration % 10 == 0 or iteration == max_iters:
            plan = u[:, None] * kernel * v[None, :]
            row_err = (plan.sum(dim=1) - a).abs().max().item()
            col_err = (plan.sum(dim=0) - b).abs().max().item()
            last_err = max(row_err, col_err)
            if last_err <= tol:
                converged = True
                break
    plan = u[:, None] * kernel * v[None, :]
    if not converged and last_err <= tol:
        converged = True
    return SinkhornResult(
        plan=plan.to(dtype=cost.dtype),
        row_marginal=a.to(dtype=cost.dtype),
        col_marginal=b.to(dtype=cost.dtype),
        iterations=iteration,
        converged=converged,
    )


__all__ = [
    "FRONTIER_PRIMITIVES_SCHEMA_VERSION",
    "BregmanDivergence",
    "CheckpointBarycenter",
    "CompositionPrimitiveError",
    "DiagonalGaussian",
    "MERAHierarchyMetadata",
    "MERALevelMetadata",
    "SinkhornResult",
    "bregman_barycenter",
    "build_mera_hierarchy_metadata",
    "canonical_json_bytes",
    "checkpoint_diagonal_gaussian_barycenter",
    "metadata_sha256",
    "normalize_weights",
    "sinkhorn_transport_plan",
    "tensor_sha256",
    "wasserstein_diagonal_gaussian_barycenter",
]
