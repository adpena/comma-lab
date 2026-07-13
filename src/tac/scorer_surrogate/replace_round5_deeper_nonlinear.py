# SPDX-License-Identifier: MIT
"""Post-SE feature charts and deterministic nonlinear heads for REPLACE round 5."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from tac.scorer_surrogate.replace_round3_fidelity_wall import prefix_feature_matrix
from tac.scorer_surrogate.replace_round4_support_ranking import (
    ORDERED_PAIR_COUNT,
    SOURCE_CLASS_SENSITIVITY,
    ordered_class_pair_ids,
)

SCHEMA = "replace_round5_deeper_nonlinear.v1"
AUTHORITY_SCOPE = "local CPU frozen-SegNet costate research evidence; no score authority"
RESEARCH_ONLY = True
BLOCK2_CHANNELS = 24
BLOCK3_CHANNELS = 48
DEEP_FEATURE_COUNT = 42 + BLOCK2_CHANNELS + BLOCK3_CHANNELS + 2


class Round5LocalizationError(ValueError):
    """A feature, cost, nonlinear, or calibration invariant failed."""


def _finite(value: Any, *, name: str, dtype: Any = np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise Round5LocalizationError(f"{name} contains nonfinite values")
    return array


def resize_bilinear_align_corners_false(value_chw: Any, height: int, width: int) -> np.ndarray:
    """Deterministic NumPy reference for Torch bilinear, ``align_corners=False``."""

    value = _finite(value_chw, name="value_chw")
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise Round5LocalizationError("bilinear input must be CHW or singleton NCHW")
    if isinstance(height, bool) or isinstance(width, bool) or height < 1 or width < 1:
        raise Round5LocalizationError("bilinear output dimensions must be positive integers")
    source_h, source_w = value.shape[1:]
    y = (np.arange(height, dtype=np.float64) + 0.5) * source_h / height - 0.5
    x = (np.arange(width, dtype=np.float64) + 0.5) * source_w / width - 0.5
    y = np.clip(y, 0.0, source_h - 1.0)
    x = np.clip(x, 0.0, source_w - 1.0)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_h - 1)
    x1 = np.minimum(x0 + 1, source_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    top = value[:, y0, :][:, :, x0] * (1.0 - wx)[None, None, :]
    top += value[:, y0, :][:, :, x1] * wx[None, None, :]
    bottom = value[:, y1, :][:, :, x0] * (1.0 - wx)[None, None, :]
    bottom += value[:, y1, :][:, :, x1] * wx[None, None, :]
    output = top * (1.0 - wy)[None, :, None] + bottom * wy[None, :, None]
    return np.ascontiguousarray(output, dtype=np.float32)


def deeper_pair_block_features(
    prefix_nchw: Any,
    block2_nchw: Any,
    block3_nchw: Any,
    labels_hw: Any,
    margins_hw: Any,
    pair_ids_hw: Any,
    *,
    checkpoint_index: int,
    checkpoint_count: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the sealed 116-column post-SE class-pair block chart."""

    prefix = _finite(prefix_nchw, name="prefix_nchw")
    if prefix.ndim == 4 and prefix.shape[0] == 1:
        prefix = prefix[0]
    if prefix.ndim != 3 or prefix.shape[0] != 32:
        raise Round5LocalizationError("prefix must have shape (1,32,H,W) or (32,H,W)")
    height, width = prefix.shape[1:]
    block2 = _finite(block2_nchw, name="block2_nchw")
    block3 = _finite(block3_nchw, name="block3_nchw")
    if block2.ndim == 4 and block2.shape[0] == 1:
        block2 = block2[0]
    if block3.ndim == 4 and block3.shape[0] == 1:
        block3 = block3[0]
    if block2.shape != (BLOCK2_CHANNELS, height // 2, width // 2):
        raise Round5LocalizationError("block-2 feature geometry drifted")
    if block3.shape != (BLOCK3_CHANNELS, height // 4, width // 4):
        raise Round5LocalizationError("block-3 feature geometry drifted")
    base = prefix_feature_matrix(
        prefix,
        labels_hw,
        margins_hw,
        checkpoint_index=checkpoint_index,
        checkpoint_count=checkpoint_count,
        stride=stride,
    ).astype(np.float32, copy=False)
    block2_up = resize_bilinear_align_corners_false(block2, height, width)
    block3_up = resize_bilinear_align_corners_false(block3, height, width)
    labels = np.asarray(labels_hw)
    margins = _finite(margins_hw, name="margins_hw")
    if labels.shape != (2 * height, 2 * width) or margins.shape != labels.shape:
        raise Round5LocalizationError("source labels/margins must match the input grid")
    pair_ids = np.asarray(pair_ids_hw)
    sampled_shape = (len(range(0, height, stride)), len(range(0, width, stride)))
    if pair_ids.shape == labels.shape:
        pair_rows = pair_ids[::2, ::2][::stride, ::stride].reshape(-1)
    elif pair_ids.shape == (height, width):
        pair_rows = pair_ids[::stride, ::stride].reshape(-1)
    elif pair_ids.shape == sampled_shape:
        pair_rows = pair_ids.reshape(-1)
    else:
        raise Round5LocalizationError("pair ids do not match input, prefix, or sampled geometry")
    pair_rows = pair_rows.astype(np.int64, copy=False)
    if pair_rows.size != base.shape[0] or np.any(
        (pair_rows < 0) | (pair_rows >= ORDERED_PAIR_COUNT)
    ):
        raise Round5LocalizationError("ordered pair rows drifted")
    sampled_labels = labels[::2, ::2][::stride, ::stride].reshape(-1).astype(np.int64)
    sampled_margin = np.tanh(
        margins[::2, ::2][::stride, ::stride].reshape(-1), dtype=np.float32
    )
    sensitivity = np.log1p(SOURCE_CLASS_SENSITIVITY[sampled_labels]).astype(np.float32)
    rows = np.concatenate(
        (
            base,
            block2_up[:, ::stride, ::stride].reshape(BLOCK2_CHANNELS, -1).T,
            block3_up[:, ::stride, ::stride].reshape(BLOCK3_CHANNELS, -1).T,
            sensitivity[:, None],
            (sensitivity * sampled_margin)[:, None],
        ),
        axis=1,
    )
    if rows.shape[1] != DEEP_FEATURE_COUNT:
        raise Round5LocalizationError("deeper feature width drifted")
    return (
        np.ascontiguousarray(rows, dtype=np.float32),
        np.ascontiguousarray(pair_rows, dtype=np.int16),
    )


def deep_feature_snapshot(segnet: Any, frame_nchw: Any) -> tuple[Any, Any, Any]:
    """Execute deterministically through encoder block 3 and return detached cuts."""

    frame = frame_nchw.detach().requires_grad_(True)
    model = segnet.encoder.model
    value = model.bn1(model.conv_stem(frame))
    first = model.blocks[0][0]
    prefix = first.aa(first.bn1(first.conv_dw(value)))
    value = model.blocks[0](value)
    block2 = model.blocks[1](value)
    block3 = model.blocks[2](block2)
    expected = ((32, 192, 256), (24, 96, 128), (48, 48, 64))
    observed = tuple(tuple(int(v) for v in tensor.shape[1:]) for tensor in (prefix, block2, block3))
    if observed != expected:
        raise Round5LocalizationError(f"post-SE feature geometry drifted: {observed}")
    return prefix.detach().clone(), block2.detach().clone(), block3.detach().clone()


def capture_round5_teacher(
    *, segnet: Any, frame_nchw: Any, labels: Any
) -> tuple[Any, Any, Any, Any, np.ndarray, dict[str, float], float]:
    """One exact teacher call yielding all post-SE cuts and the RGB costate."""

    import torch
    import torch.nn.functional as functional

    frame = frame_nchw.detach().requires_grad_(True)
    captured: dict[str, Any] = {}

    def hook(name: str) -> Any:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if name in captured:
                raise Round5LocalizationError(f"feature hook {name} fired twice")
            captured[name] = output.detach().clone()

        return capture

    modules = {
        "prefix": segnet.encoder.model.blocks[0][0].bn1,
        "block2": segnet.encoder.model.blocks[1],
        "block3": segnet.encoder.model.blocks[2],
    }
    handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
    started = time.perf_counter()
    try:
        logits = segnet(frame)
        loss = functional.cross_entropy(logits, labels)
        input_costate = torch.autograd.grad(loss, frame, retain_graph=False)[0]
    finally:
        for handle in handles:
            handle.remove()
    if captured.keys() != modules.keys():
        raise Round5LocalizationError("not every registered deeper feature hook fired")
    tensors = (*captured.values(), input_costate, logits)
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise Round5LocalizationError("round-5 exact teacher produced nonfinite tensors")
    pair_ids = ordered_class_pair_ids(labels.detach().cpu().numpy()[0], logits.detach().cpu().numpy())
    elapsed = time.perf_counter() - started
    metrics = {
        "ce": float(loss.detach().item()),
        "dseg": float((logits.argmax(1) != labels).float().mean().detach().item()),
    }
    return (
        captured["prefix"],
        captured["block2"],
        captured["block3"],
        input_costate.detach(),
        pair_ids,
        metrics,
        elapsed,
    )


class DeepCutCostLedger:
    """Observe real tensor shapes and derive post-SE cut cost/tileability receipts."""

    _CUT_PREFIXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "block2-post-se": (
            "encoder.model.conv_stem",
            "encoder.model.blocks.0.",
            "encoder.model.blocks.1.",
        ),
        "block3-post-se": (
            "encoder.model.conv_stem",
            "encoder.model.blocks.0.",
            "encoder.model.blocks.1.",
            "encoder.model.blocks.2.",
        ),
    }

    def __init__(self, segnet: Any) -> None:
        self.segnet = segnet
        self._handles: list[Any] = []
        self._conv_rows: list[dict[str, int | str]] = []
        self._se_rows: list[dict[str, int | str]] = []

    def __enter__(self) -> DeepCutCostLedger:
        import torch

        for name, module in self.segnet.named_modules():
            if isinstance(module, torch.nn.Conv2d):

                def conv_hook(
                    _module: Any, _inputs: Any, output: Any, *, module_name: str = name
                ) -> None:
                    batch, channels_out, height, width = (int(v) for v in output.shape)
                    kernel_h, kernel_w = (int(v) for v in _module.kernel_size)
                    macs = (
                        batch
                        * channels_out
                        * height
                        * width
                        * (int(_module.in_channels) // int(_module.groups))
                        * kernel_h
                        * kernel_w
                    )
                    self._conv_rows.append({"module": module_name, "forward_macs": macs})

                self._handles.append(module.register_forward_hook(conv_hook))
            if type(module).__name__ == "SqueezeExcite":

                def se_pre_hook(
                    _module: Any, inputs: Any, *, module_name: str = name
                ) -> None:
                    tensor = inputs[0]
                    batch, channels, height, width = (int(v) for v in tensor.shape)
                    reductions = batch * channels * ((height * width - 1) + 1)
                    self._se_rows.append(
                        {
                            "module": module_name,
                            "channels": channels,
                            "height": height,
                            "width": width,
                            "global_pool_forward_flops": reductions,
                            "gate_scalars": batch * channels,
                        }
                    )

                self._handles.append(module.register_forward_pre_hook(se_pre_hook))
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @staticmethod
    def _under_cut(module: str, prefixes: tuple[str, ...]) -> bool:
        return any(module == prefix or module.startswith(prefix) for prefix in prefixes)

    def summary(self) -> dict[str, Any]:
        if not self._conv_rows or not self._se_rows:
            raise Round5LocalizationError("cost ledger observed no convolution or SE reduction")
        names = [str(row["module"]) for row in self._conv_rows]
        if len(names) != len(set(names)):
            raise Round5LocalizationError("a convolution executed more than once")
        full_conv_macs = sum(int(row["forward_macs"]) for row in self._conv_rows)
        cuts: dict[str, Any] = {}
        for cut, prefixes in self._CUT_PREFIXES.items():
            conv_macs = sum(
                int(row["forward_macs"])
                for row in self._conv_rows
                if self._under_cut(str(row["module"]), prefixes)
            )
            se_rows = [
                row
                for row in self._se_rows
                if self._under_cut(str(row["module"]), prefixes)
            ]
            pool_forward_flops = sum(int(row["global_pool_forward_flops"]) for row in se_rows)
            se_mlp_macs = sum(
                int(row["forward_macs"])
                for row in self._conv_rows
                if self._under_cut(str(row["module"]), prefixes) and ".se." in str(row["module"])
            )
            forward_plus_vjp_conv_flops = 4 * conv_macs
            forward_plus_vjp_pool_flops = 2 * pool_forward_flops
            total = forward_plus_vjp_conv_flops + forward_plus_vjp_pool_flops
            cuts[cut] = {
                "status": "DERIVED_FROM_MEASURED_REAL_TENSOR_SHAPES",
                "forward_conv_macs": conv_macs,
                "forward_plus_input_vjp_conv_flops": forward_plus_vjp_conv_flops,
                "fraction_of_full_teacher_conv_flops": conv_macs / full_conv_macs,
                "se_reduction_count": len(se_rows),
                "global_pool_forward_flops": pool_forward_flops,
                "global_pool_forward_plus_vjp_flops": forward_plus_vjp_pool_flops,
                "global_pool_fraction_of_cut_conv_plus_pool_flops": (
                    forward_plus_vjp_pool_flops / total
                ),
                "se_mlp_forward_macs": se_mlp_macs,
                "se_mlp_fraction_of_cut_forward_conv_macs": se_mlp_macs / conv_macs,
                "global_gate_scalars_per_frame": sum(int(row["gate_scalars"]) for row in se_rows),
                "tileability_verdict": "not-independently-tileable-after-first-se",
                "exact_tile_recovery_requirement": (
                    "full-frame global means and SE gates must be computed and broadcast at every "
                    "registered SE boundary before any downstream tile can be exact"
                ),
            }
        return {
            "schema": "replace_round5_deep_cut_cost.v1",
            "convention": "one multiply-add is one MAC and two FLOPs",
            "full_forward_conv_macs": full_conv_macs,
            "full_forward_plus_input_backward_conv_flops": 4 * full_conv_macs,
            "cuts": cuts,
            "global_pool_operation_convention": (
                "per channel: H*W-1 additions plus one division; VJP charged one matching pass"
            ),
            "omitted_from_flop_model": [
                "batch normalization",
                "pointwise activation and sigmoid",
                "decoder interpolation",
                "loss and argmax",
                "localizer matrix multiplies",
                "autograd bookkeeping",
            ],
            "per_conv_forward_macs": self._conv_rows,
            "per_se_global_reduction": self._se_rows,
        }


@dataclass(frozen=True)
class PairGatedMLPWeights:
    input_weight: np.ndarray
    input_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    def validate(self) -> None:
        w1 = _finite(self.input_weight, name="input_weight")
        b1 = _finite(self.input_bias, name="input_bias")
        w2 = _finite(self.output_weight, name="output_weight")
        b2 = _finite(self.output_bias, name="output_bias")
        if w1.ndim != 2 or w1.shape[1] != DEEP_FEATURE_COUNT:
            raise Round5LocalizationError("MLP input weight geometry drifted")
        if b1.shape != (w1.shape[0],):
            raise Round5LocalizationError("MLP input bias geometry drifted")
        if w2.shape != (ORDERED_PAIR_COUNT, w1.shape[0]):
            raise Round5LocalizationError("MLP pair-gated output weight geometry drifted")
        if b2.shape != (ORDERED_PAIR_COUNT,):
            raise Round5LocalizationError("MLP pair-gated output bias geometry drifted")


def pair_gated_logits_numpy(
    features: Any, pair_ids: Any, weights: PairGatedMLPWeights
) -> np.ndarray:
    """Portable NumPy-fp32 inference reference for the nonlinear rung."""

    weights.validate()
    x = _finite(features, name="features")
    pair = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != pair.size or x.shape[1] != DEEP_FEATURE_COUNT:
        raise Round5LocalizationError("MLP inference rows drifted")
    if np.any((pair < 0) | (pair >= ORDERED_PAIR_COUNT)):
        raise Round5LocalizationError("MLP pair ids drifted")
    hidden = np.maximum(
        np.float32(0.0), x @ weights.input_weight.T + weights.input_bias[None]
    )
    logits = np.sum(hidden * weights.output_weight[pair], axis=1) + weights.output_bias[pair]
    if not np.isfinite(logits).all():
        raise Round5LocalizationError("MLP inference produced nonfinite logits")
    return np.ascontiguousarray(logits, dtype=np.float32)


def sigmoid_probabilities(logits: Any) -> np.ndarray:
    value = _finite(logits, name="logits")
    output = np.empty_like(value, dtype=np.float32)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def disagreement_query_audit(
    seed_probabilities: Any,
    exact_support: Any,
    *,
    seed: int,
    targeted_fraction: float = 0.04,
    random_audit_fraction: float = 0.01,
) -> dict[str, float | int | bool]:
    """Audit epistemic disagreement with a positive-propensity random floor."""

    probabilities = _finite(seed_probabilities, name="seed_probabilities")
    support = np.asarray(exact_support, dtype=np.bool_).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[1] != support.size:
        raise Round5LocalizationError("seed probabilities and support rows disagree")
    if probabilities.shape[0] < 2:
        raise Round5LocalizationError("disagreement requires multiple deterministic seeds")
    if not 0.0 < targeted_fraction < 1.0 or not 0.0 < random_audit_fraction < 1.0:
        raise Round5LocalizationError("query fractions must lie in (0,1)")
    mean = probabilities.mean(axis=0, dtype=np.float64)
    disagreement = probabilities.std(axis=0, dtype=np.float64)
    error = np.abs(mean - support.astype(np.float64))
    count = support.size
    quintile = max(1, math.ceil(0.2 * count))
    flat = np.arange(count, dtype=np.int64)
    descending = np.lexsort((flat, -disagreement))
    high_error = float(error[descending[:quintile]].mean())
    low_error = float(error[descending[-quintile:]].mean())
    ratio = high_error / max(low_error, np.finfo(np.float64).tiny)
    disagreement_rank = _rankdata_average(disagreement)
    error_rank = _rankdata_average(error)
    if disagreement_rank.std() == 0.0 or error_rank.std() == 0.0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(disagreement_rank, error_rank)[0, 1])
    targeted_count = max(1, math.ceil(targeted_fraction * count))
    targeted = descending[:targeted_count]
    remaining = np.setdiff1d(flat, targeted, assume_unique=True)
    audit_count = max(1, math.ceil(random_audit_fraction * count))
    generator = np.random.default_rng(seed)
    audit = np.sort(generator.choice(remaining, size=audit_count, replace=False))
    queried = np.concatenate((targeted, audit))
    query_error = float(error[queried].sum(dtype=np.float64))
    total_error = float(error.sum(dtype=np.float64))
    return {
        "cell_count": count,
        "targeted_count": targeted_count,
        "random_audit_count": audit_count,
        "queried_count": int(queried.size),
        "realized_query_fraction": queried.size / count,
        "random_audit_positive_propensity": audit_count / remaining.size,
        "high_disagreement_mean_absolute_error": high_error,
        "low_disagreement_mean_absolute_error": low_error,
        "high_to_low_error_ratio": ratio,
        "spearman_disagreement_vs_absolute_error": spearman,
        "queried_absolute_error_fraction": query_error / max(total_error, np.finfo(np.float64).tiny),
        "disagreement_rank_gate_pass": bool(ratio >= 1.25 and spearman > 0.0),
    }


__all__ = [
    "AUTHORITY_SCOPE",
    "BLOCK2_CHANNELS",
    "BLOCK3_CHANNELS",
    "DEEP_FEATURE_COUNT",
    "RESEARCH_ONLY",
    "DeepCutCostLedger",
    "PairGatedMLPWeights",
    "Round5LocalizationError",
    "capture_round5_teacher",
    "deep_feature_snapshot",
    "deeper_pair_block_features",
    "disagreement_query_audit",
    "pair_gated_logits_numpy",
    "resize_bilinear_align_corners_false",
    "sigmoid_probabilities",
]
