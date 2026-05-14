# SPDX-License-Identifier: MIT
"""Per-layer CUDA-vs-CPU drift comparator.

Given two :class:`IntrospectionRecord` objects captured for the same architecture
on two different devices (typically CUDA vs CPU, but any two devices the toolkit
supports), compute per-layer drift metrics:

- ``l2_relative_error`` = ``L2(out_a - out_b) / max(L2(out_a), eps)``
- ``max_abs_error`` = ``max|out_a - out_b|``
- ``kl_divergence`` (only meaningful for logit-like outputs)
- ``rank_drift`` (top-K disagreement count for argmax-like outputs)

Plus a closed-form compounding-factor test:

- ``compounding_factor(layers) -> float`` returns ``prod(1 + per_layer_eps)``,
  the geometric "(1+ε)^L" prediction the operator can verify on real data.

All numbers are tagged ``[diagnostic-not-score]`` per CLAUDE.md.

Limitations
-----------

This module does NOT dispatch CUDA. Per the task spec, the CUDA path is
designed but not invoked here. Operators dispatch the CUDA capture separately
(via the ``experiments/dump_scorer_activations.py`` CLI on a Linux x86_64 GPU
instance) and pass both records into :func:`compute_layer_drift` locally.

Per CLAUDE.md "MPS auth eval is NOISE": MPS-vs-CUDA drift comparisons are
recorded as ``[advisory only]`` and MUST NEVER be used for promotion.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .scorer_introspection import IntrospectionRecord, LayerRecord, LayerStats


@dataclass
class DriftMetrics:
    """Per-layer drift summary."""

    layer_name: str
    module_type: str
    output_index: int
    l2_relative_error: float
    max_abs_error: float
    mean_abs_error: float
    kl_divergence: float | None
    rank_top1_disagreement: float | None
    has_full_tensors: bool
    fingerprint_only_l2_proxy: float | None  # used when full tensors absent
    fingerprint_only_max_proxy: float | None
    note: str = ""


def _l2_relative(diff: torch.Tensor, ref: torch.Tensor) -> float:
    n_diff = float(torch.linalg.vector_norm(diff.flatten().float()).item())
    n_ref = float(torch.linalg.vector_norm(ref.flatten().float()).item())
    return n_diff / max(n_ref, 1e-12)


def _kl_divergence_logit(a: torch.Tensor, b: torch.Tensor) -> float:
    """KL(softmax(a) || softmax(b)) along the last axis, mean-reduced."""
    a = a.float()
    b = b.float()
    if a.dim() < 2 or a.shape != b.shape:
        return float("nan")
    log_pa = torch.log_softmax(a, dim=-1)
    log_pb = torch.log_softmax(b, dim=-1)
    pa = log_pa.exp()
    return float((pa * (log_pa - log_pb)).sum(dim=-1).mean().item())


def _rank_top1_disagreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of positions where argmax(a) != argmax(b) along last axis."""
    if a.dim() < 2 or a.shape != b.shape:
        return float("nan")
    da = a.argmax(dim=-1)
    db = b.argmax(dim=-1)
    return float((da != db).float().mean().item())


def _drift_from_full_tensors(
    a: torch.Tensor, b: torch.Tensor, layer_type: str
) -> tuple[float, float, float, float | None, float | None]:
    diff = a.float() - b.float()
    l2_rel = _l2_relative(diff, a)
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.abs().mean().item()) if diff.numel() else 0.0
    # KL/rank only meaningful for logit-shaped tensors. We treat the layer as
    # "logit-like" if its module type ends in ``Linear`` or contains ``head``.
    kl = None
    rank = None
    cls_lower = layer_type.lower()
    if any(tok in cls_lower for tok in ("linear", "head", "classifier", "logit")):
        try:
            kl = _kl_divergence_logit(a, b)
            rank = _rank_top1_disagreement(a, b)
        except Exception:  # pragma: no cover - defensive
            kl = None
            rank = None
    return l2_rel, max_abs, mean_abs, kl, rank


def _drift_from_fingerprints(s_a: LayerStats, s_b: LayerStats) -> tuple[float, float]:
    """Best-effort scalar drift proxies when only fingerprints are kept.

    These are NOT a substitute for full-tensor L2 — they only indicate whether
    the gross magnitude/spread of the layer's output is different. A zero
    proxy does NOT prove identity. Use ``capture_mode="full"`` for layers you
    care to compare exactly.
    """
    if s_a.numel == 0 or s_b.numel == 0:
        return 0.0, 0.0
    # L2 proxy: |L2_a - L2_b| / max(L2_a, eps).
    l2_proxy = abs(s_a.l2_norm - s_b.l2_norm) / max(s_a.l2_norm, 1e-12)
    # Max proxy: difference of max magnitudes.
    max_proxy = abs(max(abs(s_a.min), abs(s_a.max)) - max(abs(s_b.min), abs(s_b.max)))
    return l2_proxy, max_proxy


def compute_layer_drift(
    record_a: IntrospectionRecord,
    record_b: IntrospectionRecord,
) -> dict[str, list[DriftMetrics]]:
    """Compute per-layer drift between two introspection records.

    Returns ``{layer_name: [DriftMetrics, ...]}`` where the list ranges over
    the layer's output tensor indices (most layers have a single output and
    the list will be of length 1).

    Layers present in only one record are skipped with a stable warning into
    the metrics dict via a ``note``.
    """
    if record_a.model_kind != record_b.model_kind:
        raise ValueError(
            f"records describe different models: {record_a.model_kind} vs "
            f"{record_b.model_kind}"
        )

    by_name_b: dict[str, LayerRecord] = {layer.name: layer for layer in record_b.layers}

    out: dict[str, list[DriftMetrics]] = {}
    for layer_a in record_a.layers:
        layer_b = by_name_b.get(layer_a.name)
        if layer_b is None:
            out[layer_a.name] = [
                DriftMetrics(
                    layer_name=layer_a.name,
                    module_type=layer_a.module_type,
                    output_index=0,
                    l2_relative_error=float("nan"),
                    max_abs_error=float("nan"),
                    mean_abs_error=float("nan"),
                    kl_divergence=None,
                    rank_top1_disagreement=None,
                    has_full_tensors=False,
                    fingerprint_only_l2_proxy=None,
                    fingerprint_only_max_proxy=None,
                    note="missing in record_b",
                )
            ]
            continue

        per_output: list[DriftMetrics] = []
        n_outputs = max(len(layer_a.output_stats), len(layer_b.output_stats))
        for i in range(n_outputs):
            stats_a = layer_a.output_stats[i] if i < len(layer_a.output_stats) else None
            stats_b = layer_b.output_stats[i] if i < len(layer_b.output_stats) else None
            full_a = (
                layer_a.full_output[i]
                if layer_a.full_output is not None and i < len(layer_a.full_output)
                else None
            )
            full_b = (
                layer_b.full_output[i]
                if layer_b.full_output is not None and i < len(layer_b.full_output)
                else None
            )

            if full_a is not None and full_b is not None and full_a.shape == full_b.shape:
                l2_rel, max_abs, mean_abs, kl, rank = _drift_from_full_tensors(
                    full_a, full_b, layer_a.module_type
                )
                per_output.append(
                    DriftMetrics(
                        layer_name=layer_a.name,
                        module_type=layer_a.module_type,
                        output_index=i,
                        l2_relative_error=l2_rel,
                        max_abs_error=max_abs,
                        mean_abs_error=mean_abs,
                        kl_divergence=kl,
                        rank_top1_disagreement=rank,
                        has_full_tensors=True,
                        fingerprint_only_l2_proxy=None,
                        fingerprint_only_max_proxy=None,
                        note="",
                    )
                )
                continue

            if stats_a is None or stats_b is None:
                per_output.append(
                    DriftMetrics(
                        layer_name=layer_a.name,
                        module_type=layer_a.module_type,
                        output_index=i,
                        l2_relative_error=float("nan"),
                        max_abs_error=float("nan"),
                        mean_abs_error=float("nan"),
                        kl_divergence=None,
                        rank_top1_disagreement=None,
                        has_full_tensors=False,
                        fingerprint_only_l2_proxy=None,
                        fingerprint_only_max_proxy=None,
                        note="missing output index",
                    )
                )
                continue

            l2_proxy, max_proxy = _drift_from_fingerprints(stats_a, stats_b)
            per_output.append(
                DriftMetrics(
                    layer_name=layer_a.name,
                    module_type=layer_a.module_type,
                    output_index=i,
                    l2_relative_error=float("nan"),
                    max_abs_error=float("nan"),
                    mean_abs_error=float("nan"),
                    kl_divergence=None,
                    rank_top1_disagreement=None,
                    has_full_tensors=False,
                    fingerprint_only_l2_proxy=l2_proxy,
                    fingerprint_only_max_proxy=max_proxy,
                    note="fingerprint-only (capture_mode='fingerprint' or layer too large)",
                )
            )
        out[layer_a.name] = per_output

    return out


def drift_to_dict(drift: dict[str, list[DriftMetrics]]) -> list[dict]:
    """Flatten the drift mapping for JSON serialization."""
    flat: list[dict] = []
    for entries in drift.values():
        for d in entries:
            flat.append(asdict(d))
    return flat


def compounding_factor(per_layer_eps: Iterable[float]) -> float:
    """Closed-form ``∏ (1 + ε_i)`` for a sequence of per-layer relative errors.

    Used to test the "(1 + ε)^L compounds to PR102's CUDA-CPU 5x pose drift"
    hypothesis. If 12 RepMixer blocks each contribute ε ≈ 0.14, the compound
    factor is ``1.14 ** 12 ≈ 4.82``, close to the empirical 5x multiplier.

    Returns the (1 + ε)-product. The caller decides how to interpret it
    relative to the observed final-output drift.
    """
    factor = 1.0
    for eps in per_layer_eps:
        factor *= 1.0 + float(eps)
    return factor


def estimate_compounding_for_path(
    drift: dict[str, list[DriftMetrics]],
    layer_name_filter: str = "vision.stages.",
) -> dict[str, float]:
    """Aggregate compound factors along a named substructure.

    Default filter selects the FastViT vision-tower stages so the operator can
    test the per-block compounding hypothesis directly:

    >>> drift = compute_layer_drift(record_cuda, record_cpu)
    >>> estimate_compounding_for_path(drift)
    {'compound_factor_l2_rel': 4.78, 'num_layers': 12, ...}
    """
    eps_values: list[float] = []
    for name, entries in drift.items():
        if layer_name_filter not in name:
            continue
        # Keep only the RepMixerBlock-level entries to avoid double-counting
        # the inner conv/fc layers that are children of the block.
        if not name.endswith(tuple(f".blocks.{i}" for i in range(20))):
            continue
        for entry in entries:
            if math.isnan(entry.l2_relative_error):
                continue
            eps_values.append(entry.l2_relative_error)
    return {
        "compound_factor_l2_rel": compounding_factor(eps_values),
        "num_layers_in_path": len(eps_values),
        "mean_eps": (sum(eps_values) / len(eps_values)) if eps_values else 0.0,
        "max_eps": max(eps_values) if eps_values else 0.0,
        "filter": layer_name_filter,
    }


def compute_score_jacobian_per_layer(
    *args, **kwargs
):  # pragma: no cover - intentional stub
    """Placeholder for score-Jacobian-per-layer estimation.

    Computing ``∂score/∂(layer activation)`` requires a full forward+backward
    on the contest scorer with the contest distortion definition, AND running
    the result through the contest evaluation. This is non-trivial in CPU-only
    mode (the auth eval is a CUDA path on Linux x86_64 per CLAUDE.md
    "Submission auth eval — BOTH CPU AND CUDA"), and the task spec explicitly
    flags this as a follow-up.

    Calling this raises ``NotImplementedError`` so callers cannot accidentally
    consume a placeholder number as an exploit signal. Tracked in
    ``feedback_scorer_introspection_toolkit_20260508.md`` as deferred.
    """
    raise NotImplementedError(
        "compute_score_jacobian_per_layer is deferred. The CUDA-faithful "
        "Jacobian requires a contest-CUDA forward+backward path; capture the "
        "CUDA introspection record first via experiments/dump_scorer_"
        "activations.py on a Linux x86_64 GPU instance, then operator will "
        "wire this in a follow-up. [diagnostic-not-score]"
    )
