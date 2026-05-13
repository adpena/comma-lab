"""Adapter-composition primitives for LoRA/DoRA and tropical gates.

The objects here are deterministic contracts around adapter records.  They do
not own training state, do not infer score authority, and do not dispatch work.
They are meant to sit between substrate-specific adapter codecs such as
``tac.substrates.pr95_lora_dora`` and higher-level composition planners.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from tac.composition.frontier_primitives import (
    canonical_json_bytes,
    metadata_sha256,
    tensor_sha256,
)
from tac.composition.registry import (
    PLANNING_ONLY,
    PROMOTION_ELIGIBLE,
    READY_FOR_EXACT_EVAL_DISPATCH,
    SCORE_CLAIM,
)

ADAPTER_COMPOSITION_SCHEMA_VERSION = "tac_composition_adapters_v1"

AdapterKind = Literal["lora", "dora", "td_lora"]
TropicalSelectionMode = Literal["hard_max", "softmax"]
HypernetworkActivation = Literal["identity", "relu", "tanh"]


class AdapterCompositionError(ValueError):
    """Raised when adapter composition inputs are malformed."""


def _require_finite_tensor(tensor: torch.Tensor, *, field_name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise AdapterCompositionError(
            f"{field_name} must be torch.Tensor, got {type(tensor).__name__}"
        )
    if tensor.numel() == 0:
        raise AdapterCompositionError(f"{field_name} must be non-empty")
    if not torch.isfinite(tensor).all():
        raise AdapterCompositionError(f"{field_name} must contain finite values")


def _flatten_weight(weight: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    _require_finite_tensor(weight, field_name="weight")
    original_shape = tuple(weight.shape)
    if weight.ndim == 2:
        return weight, original_shape
    if weight.ndim == 4:
        return weight.reshape(weight.shape[0], -1), original_shape
    raise AdapterCompositionError(
        f"weight must be 2-D or 4-D for adapter folding; got {original_shape}"
    )


@dataclass(frozen=True, slots=True)
class AdapterRecord:
    """Canonical LoRA/DoRA/TD-LoRA adapter record.

    The tensor shape convention matches PR95 LoRA/DoRA trailer records:
    ``A`` has shape ``(rank, in_dim)`` and ``B`` has shape ``(out_dim, rank)``.
    DoRA records additionally carry ``magnitude`` with shape ``(out_dim,)``.
    """

    name: str
    kind: AdapterKind
    rank: int
    alpha: float
    A: torch.Tensor
    B: torch.Tensor
    magnitude: torch.Tensor | None = None
    adapter_id: str | None = None
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise AdapterCompositionError("adapter name cannot be empty")
        if self.kind not in ("lora", "dora", "td_lora"):
            raise AdapterCompositionError(f"unsupported adapter kind {self.kind!r}")
        if self.rank < 1:
            raise AdapterCompositionError("rank must be >= 1")
        if not math.isfinite(float(self.alpha)):
            raise AdapterCompositionError("alpha must be finite")
        _require_finite_tensor(self.A, field_name="A")
        _require_finite_tensor(self.B, field_name="B")
        if self.A.ndim != 2:
            raise AdapterCompositionError(f"A must be 2-D, got {tuple(self.A.shape)}")
        if self.B.ndim != 2:
            raise AdapterCompositionError(f"B must be 2-D, got {tuple(self.B.shape)}")
        if self.A.shape[0] != self.rank:
            raise AdapterCompositionError(
                f"A first dim {self.A.shape[0]} != rank {self.rank}"
            )
        if self.B.shape[1] != self.rank:
            raise AdapterCompositionError(
                f"B second dim {self.B.shape[1]} != rank {self.rank}"
            )
        if self.kind == "dora":
            if self.magnitude is None:
                raise AdapterCompositionError("DoRA records require magnitude")
            _require_finite_tensor(self.magnitude, field_name="magnitude")
            if tuple(self.magnitude.shape) != (self.B.shape[0],):
                raise AdapterCompositionError(
                    f"magnitude shape {tuple(self.magnitude.shape)} != "
                    f"({self.B.shape[0]},)"
                )
        elif self.magnitude is not None:
            raise AdapterCompositionError("only DoRA records may carry magnitude")

    @property
    def resolved_adapter_id(self) -> str:
        return self.adapter_id or f"{self.kind}:{self.name}:r{self.rank}"

    @property
    def target_weight_name(self) -> str:
        return self.name if self.name.endswith(".weight") else f"{self.name}.weight"

    def to_metadata(self) -> dict[str, Any]:
        payload = {
            "A_sha256": tensor_sha256(self.A),
            "B_sha256": tensor_sha256(self.B),
            "adapter_id": self.resolved_adapter_id,
            "alpha": float(self.alpha),
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "name": self.name,
            "planning_only": PLANNING_ONLY,
            "promotion_eligible": PROMOTION_ELIGIBLE,
            "rank": self.rank,
            "ready_for_exact_eval_dispatch": READY_FOR_EXACT_EVAL_DISPATCH,
            "schema_version": ADAPTER_COMPOSITION_SCHEMA_VERSION,
            "score_claim": SCORE_CLAIM,
            "target_weight_name": self.target_weight_name,
        }
        if self.magnitude is not None:
            payload["magnitude_sha256"] = tensor_sha256(self.magnitude)
        return payload


def adapter_record_from_pr95(record: Mapping[str, Any]) -> AdapterRecord:
    """Normalize a PR95 LoRA/DoRA trailer dict to :class:`AdapterRecord`."""

    try:
        name = str(record["name"])
        kind = record["kind"]
        rank = int(record["rank"])
        alpha = float(record["alpha"])
        A = record["A"]
        B = record["B"]
    except KeyError as exc:
        raise AdapterCompositionError(f"missing adapter record key {exc.args[0]!r}") from exc
    if kind not in ("lora", "dora", "td_lora"):
        raise AdapterCompositionError(f"unsupported adapter kind {kind!r}")
    magnitude = record.get("magnitude")
    adapter_id = record.get("adapter_id")
    metadata_obj = record.get("metadata", ())
    if isinstance(metadata_obj, Mapping):
        metadata = tuple(sorted((str(k), str(v)) for k, v in metadata_obj.items()))
    else:
        metadata = tuple((str(k), str(v)) for k, v in metadata_obj)
    return AdapterRecord(
        name=name,
        kind=kind,
        rank=rank,
        alpha=alpha,
        A=A,
        B=B,
        magnitude=magnitude,
        adapter_id=str(adapter_id) if adapter_id is not None else None,
        metadata=metadata,
    )


def adapter_delta_matrix(adapter: AdapterRecord | Mapping[str, Any]) -> torch.Tensor:
    """Return ``(alpha / rank) * B @ A`` for a LoRA-style adapter."""

    rec = adapter_record_from_pr95(adapter) if isinstance(adapter, Mapping) else adapter
    scale = float(rec.alpha) / float(rec.rank)
    return scale * (rec.B.float() @ rec.A.float())


def fold_adapter_record_into_weight(
    weight: torch.Tensor,
    adapter: AdapterRecord | Mapping[str, Any],
) -> torch.Tensor:
    """Fold one LoRA/DoRA adapter into a 2-D or 4-D base weight tensor."""

    rec = adapter_record_from_pr95(adapter) if isinstance(adapter, Mapping) else adapter
    flat, original_shape = _flatten_weight(weight)
    flat_f32 = flat.float()
    delta = adapter_delta_matrix(rec).to(device=flat_f32.device)
    if tuple(delta.shape) != tuple(flat_f32.shape):
        raise AdapterCompositionError(
            f"adapter delta shape {tuple(delta.shape)} != flattened weight "
            f"shape {tuple(flat_f32.shape)} for {rec.name!r}"
        )
    if rec.kind in ("lora", "td_lora"):
        folded = flat_f32 + delta
    elif rec.kind == "dora":
        assert rec.magnitude is not None
        candidate = flat_f32 + delta
        norm = torch.linalg.norm(candidate, dim=1, keepdim=True).clamp_min(1e-12)
        folded = rec.magnitude.to(device=flat_f32.device).float().unsqueeze(1)
        folded = folded * (candidate / norm)
    else:  # pragma: no cover - AdapterRecord validates kind.
        raise AdapterCompositionError(f"unsupported adapter kind {rec.kind!r}")
    return folded.reshape(original_shape).to(dtype=weight.dtype, device=weight.device)


def fold_adapter_chain(
    state_dict: Mapping[str, torch.Tensor],
    adapters: Sequence[AdapterRecord | Mapping[str, Any]],
    *,
    on_missing: Literal["raise", "ignore"] = "raise",
) -> dict[str, torch.Tensor]:
    """Fold a sequence of adapters into a copied state dict.

    The default is fail-closed on missing targets.  ``on_missing='ignore'`` is
    available for runtime compatibility with historical PR95 inflate behavior.
    """

    if on_missing not in ("raise", "ignore"):
        raise AdapterCompositionError("on_missing must be 'raise' or 'ignore'")
    out = dict(state_dict)
    for adapter in adapters:
        rec = adapter_record_from_pr95(adapter) if isinstance(adapter, Mapping) else adapter
        target = rec.target_weight_name
        if target not in out and rec.name in out:
            target = rec.name
        if target not in out:
            if on_missing == "ignore":
                continue
            raise AdapterCompositionError(f"adapter target {rec.target_weight_name!r} missing")
        out[target] = fold_adapter_record_into_weight(out[target], rec)
    return out


@dataclass(frozen=True, slots=True)
class TropicalSelectionResult:
    """Deterministic tropical adapter-selection output."""

    branch_ids: tuple[str, ...]
    logits: torch.Tensor
    weights: torch.Tensor
    selected_branch_id: str
    mode: TropicalSelectionMode

    def to_metadata(self) -> dict[str, Any]:
        return {
            "branch_ids": list(self.branch_ids),
            "logits_sha256": tensor_sha256(self.logits),
            "mode": self.mode,
            "planning_only": PLANNING_ONLY,
            "promotion_eligible": PROMOTION_ELIGIBLE,
            "ready_for_exact_eval_dispatch": READY_FOR_EXACT_EVAL_DISPATCH,
            "schema_version": ADAPTER_COMPOSITION_SCHEMA_VERSION,
            "score_claim": SCORE_CLAIM,
            "selected_branch_id": self.selected_branch_id,
            "weights": [float(x) for x in self.weights.detach().cpu().tolist()],
        }

    def sha256(self) -> str:
        return metadata_sha256(self.to_metadata())


def stable_softmax(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    """Numerically stable softmax with validation."""

    _require_finite_tensor(logits, field_name="logits")
    if logits.ndim != 1:
        raise AdapterCompositionError("logits must be 1-D")
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise AdapterCompositionError("temperature must be positive and finite")
    z = logits.float() / float(temperature)
    z = z - z.max()
    weights = torch.exp(z)
    return weights / weights.sum().clamp_min(1e-12)


def tropical_adapter_weights(
    logits: torch.Tensor,
    branch_ids: Sequence[str],
    *,
    mode: TropicalSelectionMode = "hard_max",
    temperature: float = 1.0,
) -> TropicalSelectionResult:
    """Select or softly mix adapter branches with tropical max-plus semantics."""

    if mode not in ("hard_max", "softmax"):
        raise AdapterCompositionError(f"unsupported tropical selection mode {mode!r}")
    _require_finite_tensor(logits, field_name="logits")
    if logits.ndim != 1:
        raise AdapterCompositionError("logits must be 1-D")
    ids = tuple(str(branch_id) for branch_id in branch_ids)
    if len(ids) != logits.numel():
        raise AdapterCompositionError("branch_ids length must match logits")
    if len(set(ids)) != len(ids):
        raise AdapterCompositionError("branch_ids must be unique")
    if mode == "hard_max":
        index = int(torch.argmax(logits).item())
        weights = torch.zeros_like(logits, dtype=torch.float32)
        weights[index] = 1.0
    else:
        weights = stable_softmax(logits, temperature=temperature)
        index = int(torch.argmax(weights).item())
    return TropicalSelectionResult(
        branch_ids=ids,
        logits=logits.detach().clone(),
        weights=weights,
        selected_branch_id=ids[index],
        mode=mode,
    )


def compose_tropical_adapter_delta(
    deltas: Sequence[torch.Tensor],
    logits: torch.Tensor,
    branch_ids: Sequence[str],
    *,
    mode: TropicalSelectionMode = "hard_max",
    temperature: float = 1.0,
) -> tuple[torch.Tensor, TropicalSelectionResult]:
    """Compose same-shaped adapter deltas by hard tropical max or softmax mix."""

    if not deltas:
        raise AdapterCompositionError("deltas must be non-empty")
    ref_shape = tuple(deltas[0].shape)
    for idx, delta in enumerate(deltas):
        _require_finite_tensor(delta, field_name=f"deltas[{idx}]")
        if tuple(delta.shape) != ref_shape:
            raise AdapterCompositionError(
                f"deltas[{idx}] shape {tuple(delta.shape)} != {ref_shape}"
            )
    result = tropical_adapter_weights(
        logits,
        branch_ids,
        mode=mode,
        temperature=temperature,
    )
    stack = torch.stack([d.float() for d in deltas], dim=0)
    view_shape = (-1,) + (1,) * (stack.ndim - 1)
    mixed = (stack * result.weights.to(device=stack.device).view(view_shape)).sum(dim=0)
    return mixed.to(dtype=deltas[0].dtype, device=deltas[0].device), result


def fold_tropical_adapter_metadata(
    *,
    target_name: str,
    selection: TropicalSelectionResult,
    output_delta: torch.Tensor,
) -> dict[str, Any]:
    """Return deterministic metadata for a tropical adapter fold."""

    if not target_name:
        raise AdapterCompositionError("target_name cannot be empty")
    _require_finite_tensor(output_delta, field_name="output_delta")
    payload = {
        "branch_ids": list(selection.branch_ids),
        "kind": "tropical_adapter_fold",
        "mode": selection.mode,
        "output_delta_sha256": tensor_sha256(output_delta),
        "planning_only": PLANNING_ONLY,
        "promotion_eligible": PROMOTION_ELIGIBLE,
        "ready_for_exact_eval_dispatch": READY_FOR_EXACT_EVAL_DISPATCH,
        "schema_version": ADAPTER_COMPOSITION_SCHEMA_VERSION,
        "score_claim": SCORE_CLAIM,
        "selected_branch_id": selection.selected_branch_id,
        "selection_sha256": selection.sha256(),
        "target_name": target_name,
    }
    payload["metadata_sha256"] = metadata_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class FrozenLinearLayer:
    """Explicit linear layer for deterministic hypernetwork composition."""

    weight: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        _require_finite_tensor(self.weight, field_name="weight")
        _require_finite_tensor(self.bias, field_name="bias")
        if self.weight.ndim != 2:
            raise AdapterCompositionError("hypernetwork weight must be 2-D")
        if tuple(self.bias.shape) != (self.weight.shape[0],):
            raise AdapterCompositionError(
                f"bias shape {tuple(self.bias.shape)} != ({self.weight.shape[0]},)"
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight.to(device=x.device, dtype=x.dtype) @ x + self.bias.to(
            device=x.device,
            dtype=x.dtype,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "bias_sha256": tensor_sha256(self.bias),
            "out_dim": int(self.weight.shape[0]),
            "in_dim": int(self.weight.shape[1]),
            "weight_sha256": tensor_sha256(self.weight),
        }


@dataclass(frozen=True, slots=True)
class HypernetworkWeights:
    """Adapter weights emitted by a deterministic hypernetwork composer."""

    adapter_ids: tuple[str, ...]
    logits: torch.Tensor
    weights: torch.Tensor

    def as_mapping(self) -> dict[str, float]:
        return {
            adapter_id: float(weight)
            for adapter_id, weight in zip(self.adapter_ids, self.weights, strict=True)
        }


@dataclass(frozen=True, slots=True)
class DeterministicHypernetworkComposer:
    """Tiny explicit hypernetwork mapping context features to adapter weights."""

    adapter_ids: tuple[str, ...]
    layers: tuple[FrozenLinearLayer, ...]
    activation: HypernetworkActivation = "tanh"
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not self.adapter_ids:
            raise AdapterCompositionError("adapter_ids cannot be empty")
        if len(set(self.adapter_ids)) != len(self.adapter_ids):
            raise AdapterCompositionError("adapter_ids must be unique")
        if not self.layers:
            raise AdapterCompositionError("hypernetwork requires at least one layer")
        if self.activation not in ("identity", "relu", "tanh"):
            raise AdapterCompositionError(f"unsupported activation {self.activation!r}")
        if self.temperature <= 0.0 or not math.isfinite(float(self.temperature)):
            raise AdapterCompositionError("temperature must be positive and finite")
        for idx in range(len(self.layers) - 1):
            if self.layers[idx].weight.shape[0] != self.layers[idx + 1].weight.shape[1]:
                raise AdapterCompositionError(
                    f"layer {idx} out_dim does not match layer {idx + 1} in_dim"
                )
        if self.layers[-1].weight.shape[0] != len(self.adapter_ids):
            raise AdapterCompositionError(
                "last hypernetwork layer out_dim must match adapter_ids"
            )

    def __call__(self, context_features: torch.Tensor) -> HypernetworkWeights:
        _require_finite_tensor(context_features, field_name="context_features")
        if context_features.ndim != 1:
            raise AdapterCompositionError("context_features must be 1-D")
        if context_features.shape[0] != self.layers[0].weight.shape[1]:
            raise AdapterCompositionError(
                f"context_features dim {context_features.shape[0]} != "
                f"{self.layers[0].weight.shape[1]}"
            )
        x = context_features.float()
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                if self.activation == "relu":
                    x = torch.relu(x)
                elif self.activation == "tanh":
                    x = torch.tanh(x)
        weights = stable_softmax(x, temperature=self.temperature)
        return HypernetworkWeights(
            adapter_ids=self.adapter_ids,
            logits=x.detach().clone(),
            weights=weights,
        )

    def to_metadata(self) -> dict[str, Any]:
        payload = {
            "activation": self.activation,
            "adapter_ids": list(self.adapter_ids),
            "kind": "deterministic_hypernetwork_composer",
            "layers": [layer.to_metadata() for layer in self.layers],
            "planning_only": PLANNING_ONLY,
            "promotion_eligible": PROMOTION_ELIGIBLE,
            "ready_for_exact_eval_dispatch": READY_FOR_EXACT_EVAL_DISPATCH,
            "schema_version": ADAPTER_COMPOSITION_SCHEMA_VERSION,
            "score_claim": SCORE_CLAIM,
            "temperature": float(self.temperature),
        }
        payload["metadata_sha256"] = metadata_sha256(payload)
        return payload

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_metadata()).decode("utf-8")


@dataclass(frozen=True, slots=True)
class PoEGatingResult:
    """Product-of-experts log-domain gating result."""

    log_weights: torch.Tensor
    weights: torch.Tensor


def _validate_log_weights(log_values: torch.Tensor, *, field_name: str) -> None:
    if not isinstance(log_values, torch.Tensor):
        raise AdapterCompositionError(f"{field_name} must be torch.Tensor")
    if log_values.numel() == 0:
        raise AdapterCompositionError(f"{field_name} must be non-empty")
    if torch.isnan(log_values).any() or torch.isposinf(log_values).any():
        raise AdapterCompositionError(f"{field_name} cannot contain NaN or +Inf")


def _log_softmax_allow_neginf(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    if torch.isneginf(values).all(dim=dim).any():
        raise AdapterCompositionError("log weights cannot be all -Inf on any row")
    return values - torch.logsumexp(values, dim=dim, keepdim=True)


def product_of_experts_gating(
    expert_log_weights: torch.Tensor,
    *,
    reliabilities: Sequence[float] | None = None,
    prior_log_weights: torch.Tensor | None = None,
) -> PoEGatingResult:
    """Combine expert gates in log space with stable normalization.

    ``expert_log_weights`` is shaped ``(n_experts, n_choices)`` and may be
    logits or log-probabilities.  Each row is normalized before the product is
    formed so one expert cannot win by using a different logit offset.
    """

    _validate_log_weights(expert_log_weights, field_name="expert_log_weights")
    if expert_log_weights.ndim != 2:
        raise AdapterCompositionError("expert_log_weights must be 2-D")
    n_experts, n_choices = expert_log_weights.shape
    if n_experts < 1 or n_choices < 1:
        raise AdapterCompositionError("expert_log_weights must be non-empty")
    rel = (
        tuple(1.0 for _ in range(n_experts))
        if reliabilities is None
        else tuple(float(x) for x in reliabilities)
    )
    if len(rel) != n_experts:
        raise AdapterCompositionError("reliabilities length must match experts")
    if any((not math.isfinite(x)) for x in rel) or any(x < 0.0 for x in rel):
        raise AdapterCompositionError("reliabilities must be finite and non-negative")
    if sum(rel) <= 0.0:
        raise AdapterCompositionError("at least one reliability must be positive")
    logits = expert_log_weights.float()
    normalized = _log_softmax_allow_neginf(logits, dim=1)
    rel_t = torch.tensor(rel, dtype=normalized.dtype, device=normalized.device)
    combined = (normalized * rel_t[:, None]).sum(dim=0)
    if prior_log_weights is not None:
        _validate_log_weights(prior_log_weights, field_name="prior_log_weights")
        if tuple(prior_log_weights.shape) != (n_choices,):
            raise AdapterCompositionError(
                f"prior_log_weights shape {tuple(prior_log_weights.shape)} != "
                f"({n_choices},)"
        )
        prior = _log_softmax_allow_neginf(prior_log_weights.float(), dim=0)
        combined = combined + prior.to(device=combined.device)
    if torch.isneginf(combined).all():
        raise AdapterCompositionError("product of experts assigns zero mass to all choices")
    log_weights = combined - torch.logsumexp(combined, dim=0)
    weights = torch.exp(log_weights)
    return PoEGatingResult(log_weights=log_weights, weights=weights)


__all__ = [
    "ADAPTER_COMPOSITION_SCHEMA_VERSION",
    "AdapterCompositionError",
    "AdapterKind",
    "AdapterRecord",
    "DeterministicHypernetworkComposer",
    "FrozenLinearLayer",
    "HypernetworkWeights",
    "PoEGatingResult",
    "TropicalSelectionMode",
    "TropicalSelectionResult",
    "adapter_delta_matrix",
    "adapter_record_from_pr95",
    "compose_tropical_adapter_delta",
    "fold_adapter_chain",
    "fold_adapter_record_into_weight",
    "fold_tropical_adapter_metadata",
    "product_of_experts_gating",
    "stable_softmax",
    "tropical_adapter_weights",
]
