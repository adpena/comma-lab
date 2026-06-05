# SPDX-License-Identifier: MIT
"""Coder-aware QAT terms for compact MLX receiver training.

This module keeps the rate pressure where the evidence says it belongs for the
compact NeRV/HPRC family: inside decoder-weight training. The terms are
advisory MLX-local loss components only; archive custody, receiver proof, and
contest CPU/CUDA exact eval remain the promotion surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
    require_mlx_for_harness,
)

DEFAULT_DECODER_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "latent_embed",
    "blocks",
    "feature_grids",
    "convnext_blocks",
    "injector",
    "head",
    "decoder",
    "hfr_",
    "mfu_",
)
DEFAULT_DECODER_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "latents",
    "codebook",
    "ema",
    "student",
    "teacher",
    "quantizer",
)
CODER_QAT_AUTHORITY = "false_macos_mlx_research_signal"
_MIN_POSITIVE_FP16_SCALE = 5.960464477539063e-08


@dataclass(frozen=True)
class CoderAwareQATConfig:
    """Configuration for score-training pressure toward cheap decoder bytes.

    The loss is deliberately substrate-agnostic: select trainable decoder
    tensors by stable parameter-name substrings, then penalize distance to a
    symmetric fixed-point grid plus simple entropy-friendly shape priors. The
    hard archive/exporter still decides actual bytes.
    """

    enabled: bool = False
    quant_bits: int = 8
    quant_residual_weight: float = 1.0e-4
    magnitude_weight: float = 0.0
    delta_weight: float = 0.0
    c1a_entropy_weight: float = 0.0
    c1a_sigma: float = 0.2
    c1a_sample_size: int = 512
    include_substrings: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_DECODER_INCLUDE_SUBSTRINGS
    )
    exclude_substrings: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_DECODER_EXCLUDE_SUBSTRINGS
    )
    eps: float = 1.0e-8

    def validated(self) -> CoderAwareQATConfig:
        """Return ``self`` after fail-closed validation."""

        if int(self.quant_bits) < 1 or int(self.quant_bits) > 16:
            raise MlxScoreAwareHarnessError(
                "coder-aware QAT quant_bits must be in [1, 16]"
            )
        if float(self.quant_residual_weight) < 0.0:
            raise MlxScoreAwareHarnessError(
                "coder-aware QAT quant_residual_weight must be non-negative"
            )
        if float(self.magnitude_weight) < 0.0:
            raise MlxScoreAwareHarnessError(
                "coder-aware QAT magnitude_weight must be non-negative"
            )
        if float(self.delta_weight) < 0.0:
            raise MlxScoreAwareHarnessError(
                "coder-aware QAT delta_weight must be non-negative"
            )
        if float(self.c1a_entropy_weight) < 0.0:
            raise MlxScoreAwareHarnessError(
                "coder-aware QAT c1a_entropy_weight must be non-negative"
            )
        if float(self.c1a_sigma) <= 0.0:
            raise MlxScoreAwareHarnessError(
                "coder-aware QAT c1a_sigma must be positive"
            )
        if int(self.c1a_sample_size) <= 0:
            raise MlxScoreAwareHarnessError(
                "coder-aware QAT c1a_sample_size must be positive"
            )
        includes = tuple(str(item) for item in self.include_substrings if str(item))
        excludes = tuple(str(item) for item in self.exclude_substrings if str(item))
        if not includes:
            raise MlxScoreAwareHarnessError(
                "coder-aware QAT include_substrings must not be empty"
            )
        if float(self.eps) <= 0.0:
            raise MlxScoreAwareHarnessError("coder-aware QAT eps must be positive")
        return CoderAwareQATConfig(
            enabled=bool(self.enabled),
            quant_bits=int(self.quant_bits),
            quant_residual_weight=float(self.quant_residual_weight),
            magnitude_weight=float(self.magnitude_weight),
            delta_weight=float(self.delta_weight),
            c1a_entropy_weight=float(self.c1a_entropy_weight),
            c1a_sigma=float(self.c1a_sigma),
            c1a_sample_size=int(self.c1a_sample_size),
            include_substrings=includes,
            exclude_substrings=excludes,
            eps=float(self.eps),
        )


def coder_qat_loss_weights(cfg: CoderAwareQATConfig) -> dict[str, float]:
    """Return loss-weight entries consumable by ``RendererBundle``."""

    c = cfg.validated()
    if not c.enabled:
        return {}
    return {
        "coder_qat_quant_residual": float(c.quant_residual_weight),
        "coder_qat_magnitude": float(c.magnitude_weight),
        "coder_qat_delta": float(c.delta_weight),
        "coder_qat_c1a_entropy": float(c.c1a_entropy_weight),
    }


def coder_qat_metadata(cfg: CoderAwareQATConfig) -> dict[str, Any]:
    """Machine-readable metadata for reports without score authority leakage."""

    c = cfg.validated()
    return {
        "schema": "coder_aware_decoder_qat.v1",
        "enabled": bool(c.enabled),
        "quant_bits": int(c.quant_bits),
        "quant_residual_weight": float(c.quant_residual_weight),
        "magnitude_weight": float(c.magnitude_weight),
        "delta_weight": float(c.delta_weight),
        "c1a_entropy_weight": float(c.c1a_entropy_weight),
        "c1a_sigma": float(c.c1a_sigma),
        "c1a_sample_size": int(c.c1a_sample_size),
        "c1a_source": (
            "PR95 cat_entropy_v2 soft categorical entropy adapted to selected "
            "decoder weights"
        ),
        "quantizer_geometry": (
            "symmetric_signed_axis0_fp16_scale_for_matrix_conv_weights_"
            "per_tensor_fp16_scale_for_biases"
        ),
        "include_substrings": list(c.include_substrings),
        "exclude_substrings": list(c.exclude_substrings),
        "target": (
            "decoder weight distributions only; latents/codebooks/selectors are "
            "priced by archive sections and scorer replay"
        ),
        "authority": CODER_QAT_AUTHORITY,
        "authority_status": "advisory_training_loss_only_not_archive_or_score_authority",
    }


def _iter_named_arrays(model: Any) -> list[tuple[str, Any]]:
    """Flatten MLX module parameters into stable ``(name, array)`` entries."""

    try:
        from mlx.utils import tree_flatten
    except Exception as exc:  # pragma: no cover - exercised only without MLX.
        raise MlxScoreAwareHarnessError(
            "coder-aware QAT requires mlx.utils.tree_flatten"
        ) from exc

    params = model.parameters()
    named: list[tuple[str, Any]] = []
    for key, value in tree_flatten(params):
        name = ".".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
        if hasattr(value, "shape"):
            named.append((name, value))
    return named


def _selected_arrays(model: Any, cfg: CoderAwareQATConfig) -> list[tuple[str, Any]]:
    c = cfg.validated()
    rows = []
    for name, value in _iter_named_arrays(model):
        if any(token in name for token in c.include_substrings) and not any(
            token in name for token in c.exclude_substrings
        ):
            rows.append((name, value))
    if c.enabled and not rows:
        raise MlxScoreAwareHarnessError(
            "coder-aware QAT selected no decoder parameters; adjust include/exclude substrings"
        )
    return rows


def _archive_quant_scale(values: Any, *, quant_bits: int) -> tuple[Any, Any]:
    """Return archive-shaped ``(normalized, scale)`` for decoder QAT pressure."""

    mx = require_mlx_for_harness()
    levels = max(1, (1 << (int(quant_bits) - 1)) - 1)
    abs_values = mx.abs(values)
    if len(values.shape) >= 2 and int(values.shape[0]) > 1:
        reduce_axes = tuple(range(1, len(values.shape)))
        abs_max = mx.max(abs_values, axis=reduce_axes, keepdims=True)
    else:
        abs_max = mx.max(abs_values)
    raw_scale = abs_max / float(levels)
    scale32 = mx.where(
        abs_max > 0.0,
        mx.maximum(raw_scale, _MIN_POSITIVE_FP16_SCALE),
        1.0,
    )
    scale = mx.stop_gradient(scale32.astype(mx.float16).astype(mx.float32))
    return values / scale, scale


def build_decoder_coder_qat_terms(
    model: Any,
    cfg: CoderAwareQATConfig,
) -> dict[str, Any]:
    """Build differentiable coder-aware QAT terms for an MLX renderer.

    The fixed-point target is gradient-blocked. Gradients therefore push weights
    toward cheap quantized values without pretending that the relaxed loss is a
    byte-closed archive result.
    """

    mx = require_mlx_for_harness()
    c = cfg.validated()
    zero = mx.array(0.0, dtype=mx.float32)
    if not c.enabled:
        return {}

    arrays = _selected_arrays(model, c)
    quant_terms: list[Any] = []
    magnitude_terms: list[Any] = []
    delta_terms: list[Any] = []

    for _name, arr in arrays:
        values = arr.astype(mx.float32)
        normalized, _scale = _archive_quant_scale(values, quant_bits=c.quant_bits)
        levels = max(1, (1 << (int(c.quant_bits) - 1)) - 1)
        quantized = mx.stop_gradient(
            mx.clip(mx.round(normalized), -float(levels), float(levels))
        )
        quant_terms.append(mx.mean((normalized - quantized) ** 2))
        magnitude_terms.append(mx.mean(mx.abs(normalized)))
        flat = mx.reshape(normalized, (-1,))
        if int(flat.shape[0]) > 1:
            delta_terms.append(mx.mean(mx.abs(flat[1:] - flat[:-1])))

    return {
        "coder_qat_quant_residual": mx.mean(mx.stack(quant_terms))
        if quant_terms
        else zero,
        "coder_qat_magnitude": mx.mean(mx.stack(magnitude_terms))
        if magnitude_terms
        else zero,
        "coder_qat_delta": mx.mean(mx.stack(delta_terms)) if delta_terms else zero,
        "coder_qat_c1a_entropy": build_decoder_c1a_entropy_term(
            model,
            c,
            sigma=float(c.c1a_sigma),
            sample_size=int(c.c1a_sample_size),
        ),
    }


def build_decoder_c1a_entropy_term(
    model: Any,
    cfg: CoderAwareQATConfig,
    *,
    sigma: float,
    sample_size: int = 2000,
) -> Any:
    """Build the PR95 C1a soft-categorical entropy term for decoder weights.

    This mirrors PR95's ``cat_entropy_v2`` shape: selected decoder tensors are
    normalized to the symmetric INT grid, softly assigned to integer bins with
    Gaussian bandwidth ``sigma``, and charged by size-weighted categorical
    entropy. The max-abs scale is gradient-blocked, as in fake-quant QAT; the
    entropy itself remains differentiable with respect to decoder weights.
    """

    import math

    import numpy as np

    mx = require_mlx_for_harness()
    c = cfg.validated()
    if not c.enabled:
        return mx.array(0.0, dtype=mx.float32)
    if float(sigma) <= 0.0:
        raise MlxScoreAwareHarnessError("C1a entropy sigma must be positive")
    if int(sample_size) <= 0:
        raise MlxScoreAwareHarnessError("C1a entropy sample_size must be positive")

    arrays = _selected_arrays(model, c)
    levels = max(1, (1 << (int(c.quant_bits) - 1)) - 1)
    bins = mx.arange(-levels, levels + 1, dtype=mx.float32)
    total_numel = 0
    weighted_entropy = mx.array(0.0, dtype=mx.float32)
    log2 = math.log(2.0)

    for _name, arr in arrays:
        values = arr.astype(mx.float32)
        numel = int(np.prod(tuple(int(v) for v in values.shape)))
        if numel <= 0:
            continue
        scale = mx.stop_gradient(mx.max(mx.abs(values)) / float(levels) + float(c.eps))
        normalized = mx.reshape(values / scale, (-1,))
        if int(normalized.shape[0]) > int(sample_size):
            sample_idx = np.linspace(
                0,
                int(normalized.shape[0]) - 1,
                int(sample_size),
                dtype=np.int32,
            )
            normalized = normalized[mx.array(sample_idx)]
        diff = (mx.expand_dims(normalized, 1) - mx.expand_dims(bins, 0)) / float(sigma)
        soft_assign = mx.exp(-0.5 * (diff**2))
        soft_assign = soft_assign / (
            mx.sum(soft_assign, axis=1, keepdims=True) + float(c.eps)
        )
        bin_prob = mx.mean(soft_assign, axis=0)
        bin_prob = bin_prob / (mx.sum(bin_prob) + float(c.eps))
        entropy = -mx.sum(bin_prob * (mx.log(bin_prob + float(c.eps)) / log2))
        weighted_entropy = weighted_entropy + float(numel) * entropy
        total_numel += numel

    if total_numel <= 0:
        return mx.array(0.0, dtype=mx.float32)
    return weighted_entropy / float(total_numel)
