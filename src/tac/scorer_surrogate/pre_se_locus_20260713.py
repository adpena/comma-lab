# SPDX-License-Identifier: MIT
"""PRE-SE feature charts for the final untested Round-5 localization cell."""

from __future__ import annotations

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
from tac.scorer_surrogate.replace_round5_deeper_nonlinear import (
    DeepCutCostLedger,
    resize_bilinear_align_corners_false,
)
from tac.witness_dsl.pre_se_locus_policy_20260713 import (
    LOCUS_SPECS,
    PreSELocusSpec,
)

SCHEMA = "pre_se_locus_20260713.v1"
AUTHORITY_SCOPE = "local CPU frozen-SegNet costate research evidence; no score authority"
RESEARCH_ONLY = True


class PreSELocusError(ValueError):
    """A PRE-SE feature, graph, cost, or inference invariant failed."""


def _finite(value: Any, *, name: str, dtype: Any = np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise PreSELocusError(f"{name} contains nonfinite values")
    return array


def locus_by_name(name: str) -> PreSELocusSpec:
    for spec in LOCUS_SPECS:
        if spec.name == name:
            return spec
    raise PreSELocusError(f"unknown PRE-SE locus {name!r}")


def _normalize_feature_tensor(value: Any, *, name: str) -> np.ndarray:
    tensor = _finite(value, name=name)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise PreSELocusError(f"{name} must be CHW or singleton NCHW")
    return tensor


def pre_se_pair_block_features(
    prefix_nchw: Any,
    locus_nchw: Any,
    labels_hw: Any,
    margins_hw: Any,
    pair_ids_hw: Any,
    *,
    locus: str,
    checkpoint_index: int,
    checkpoint_count: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one Round-5-equivalent chart with exactly one PRE-SE source."""

    spec = locus_by_name(locus)
    prefix = _normalize_feature_tensor(prefix_nchw, name="prefix_nchw")
    if prefix.shape[0] != 32:
        raise PreSELocusError("prefix channel geometry drifted")
    height, width = prefix.shape[1:]
    feature = _normalize_feature_tensor(locus_nchw, name=locus)
    expected = (spec.channels, height // spec.spatial_divisor, width // spec.spatial_divisor)
    if feature.shape != expected:
        raise PreSELocusError(f"{locus} geometry drifted: {feature.shape} != {expected}")
    base = prefix_feature_matrix(
        prefix,
        labels_hw,
        margins_hw,
        checkpoint_index=checkpoint_index,
        checkpoint_count=checkpoint_count,
        stride=stride,
    ).astype(np.float32, copy=False)
    upsampled = resize_bilinear_align_corners_false(feature, height, width)
    labels = np.asarray(labels_hw)
    margins = _finite(margins_hw, name="margins_hw")
    if labels.shape != (2 * height, 2 * width) or margins.shape != labels.shape:
        raise PreSELocusError("source labels/margins must match the input grid")
    pair_ids = np.asarray(pair_ids_hw)
    sampled_shape = (len(range(0, height, stride)), len(range(0, width, stride)))
    if pair_ids.shape == labels.shape:
        pair_rows = pair_ids[::2, ::2][::stride, ::stride].reshape(-1)
    elif pair_ids.shape == (height, width):
        pair_rows = pair_ids[::stride, ::stride].reshape(-1)
    elif pair_ids.shape == sampled_shape:
        pair_rows = pair_ids.reshape(-1)
    else:
        raise PreSELocusError("pair ids do not match input, prefix, or sampled geometry")
    pair_rows = pair_rows.astype(np.int64, copy=False)
    if pair_rows.size != base.shape[0] or np.any(
        (pair_rows < 0) | (pair_rows >= ORDERED_PAIR_COUNT)
    ):
        raise PreSELocusError("ordered pair rows drifted")
    sampled_labels = labels[::2, ::2][::stride, ::stride].reshape(-1).astype(np.int64)
    sampled_margin = np.tanh(
        margins[::2, ::2][::stride, ::stride].reshape(-1), dtype=np.float32
    )
    sensitivity = np.log1p(SOURCE_CLASS_SENSITIVITY[sampled_labels]).astype(np.float32)
    rows = np.concatenate(
        (
            base,
            upsampled[:, ::stride, ::stride].reshape(spec.channels, -1).T,
            sensitivity[:, None],
            (sensitivity * sampled_margin)[:, None],
        ),
        axis=1,
    )
    if rows.shape[1] != spec.feature_count:
        raise PreSELocusError(f"{locus} feature width drifted")
    return (
        np.ascontiguousarray(rows, dtype=np.float32),
        np.ascontiguousarray(pair_rows, dtype=np.int16),
    )


def _feature_modules(segnet: Any) -> dict[str, Any]:
    model = segnet.encoder.model
    return {
        "prefix": model.blocks[0][0].bn1,
        **{
            spec.name: model.blocks[spec.stage_index][spec.block_index].se
            for spec in LOCUS_SPECS
        },
    }


def _capture_feature_hooks(segnet: Any) -> tuple[dict[str, Any], list[Any]]:
    captured: dict[str, Any] = {}
    handles = []
    for name, module in _feature_modules(segnet).items():
        if name == "prefix":

            def hook(
                _module: Any, _inputs: Any, output: Any, *, feature_name: str = name
            ) -> None:
                if feature_name in captured:
                    raise PreSELocusError(f"feature hook {feature_name} fired twice")
                captured[feature_name] = output.detach().clone()

            handles.append(module.register_forward_hook(hook))
        else:

            def pre_hook(
                _module: Any, inputs: Any, *, feature_name: str = name
            ) -> None:
                if feature_name in captured:
                    raise PreSELocusError(f"feature hook {feature_name} fired twice")
                captured[feature_name] = inputs[0].detach().clone()

            handles.append(module.register_forward_pre_hook(pre_hook))
    return captured, handles


def _validate_captured(captured: dict[str, Any]) -> None:
    expected = {
        "prefix": (32, 192, 256),
        "block2-pre-se": (144, 96, 128),
        "block3-pre-se": (288, 48, 64),
    }
    observed = {
        name: tuple(int(value) for value in tensor.shape[1:])
        for name, tensor in captured.items()
    }
    if observed != expected:
        raise PreSELocusError(f"PRE-SE feature geometry drifted: {observed}")


def pre_se_feature_snapshot(segnet: Any, frame_nchw: Any) -> tuple[Any, Any, Any]:
    """Execute through stage 3 and capture both last-MBConv PRE-SE inputs."""

    captured, handles = _capture_feature_hooks(segnet)
    model = segnet.encoder.model
    value = frame_nchw.detach().requires_grad_(True)
    try:
        value = model.bn1(model.conv_stem(value))
        value = model.blocks[0](value)
        value = model.blocks[1](value)
        model.blocks[2](value)
    finally:
        for handle in handles:
            handle.remove()
    _validate_captured(captured)
    return (
        captured["prefix"],
        captured["block2-pre-se"],
        captured["block3-pre-se"],
    )


def capture_pre_se_teacher(
    *, segnet: Any, frame_nchw: Any, labels: Any
) -> tuple[Any, Any, Any, Any, np.ndarray, dict[str, float], float]:
    """One exact teacher call yielding both PRE-SE cuts and the RGB costate."""

    import torch
    import torch.nn.functional as functional

    frame = frame_nchw.detach().requires_grad_(True)
    captured, handles = _capture_feature_hooks(segnet)
    started = time.perf_counter()
    try:
        logits = segnet(frame)
        loss = functional.cross_entropy(logits, labels)
        input_costate = torch.autograd.grad(loss, frame, retain_graph=False)[0]
    finally:
        for handle in handles:
            handle.remove()
    _validate_captured(captured)
    tensors = (*captured.values(), input_costate, logits)
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise PreSELocusError("PRE-SE exact teacher produced nonfinite tensors")
    pair_ids = ordered_class_pair_ids(
        labels.detach().cpu().numpy()[0], logits.detach().cpu().numpy()
    )
    elapsed = time.perf_counter() - started
    metrics = {
        "ce": float(loss.detach().item()),
        "dseg": float((logits.argmax(1) != labels).float().mean().detach().item()),
    }
    return (
        captured["prefix"],
        captured["block2-pre-se"],
        captured["block3-pre-se"],
        input_costate.detach(),
        pair_ids,
        metrics,
        elapsed,
    )


def verify_pre_se_taps(segnet: Any, frame_nchw: Any) -> dict[str, Any]:
    """Prove the hook is before its own SE and audit every upstream SE dependency."""

    import torch

    model = segnet.encoder.model
    execution_order: list[str] = []
    all_handles = []
    for module_name, module in segnet.named_modules():
        if type(module).__name__ == "SqueezeExcite":

            def order_hook(
                _module: Any, _inputs: Any, *, name: str = module_name
            ) -> None:
                execution_order.append(name)

            all_handles.append(module.register_forward_pre_hook(order_hook))
    captures: dict[str, dict[str, Any]] = {}
    for spec in LOCUS_SPECS:
        block = model.blocks[spec.stage_index][spec.block_index]
        captures[spec.name] = {}

        def aa_hook(
            _module: Any, _inputs: Any, output: Any, *, name: str = spec.name
        ) -> None:
            captures[name]["depthwise_activation"] = output.detach().clone()

        def se_pre_hook(
            _module: Any, inputs: Any, *, name: str = spec.name
        ) -> None:
            captures[name]["pre_se"] = inputs[0].detach().clone()

        def se_post_hook(
            _module: Any, _inputs: Any, output: Any, *, name: str = spec.name
        ) -> None:
            captures[name]["post_se"] = output.detach().clone()

        all_handles.extend(
            (
                block.aa.register_forward_hook(aa_hook),
                block.se.register_forward_pre_hook(se_pre_hook),
                block.se.register_forward_hook(se_post_hook),
            )
        )
    try:
        with torch.no_grad():
            segnet(frame_nchw.detach())
    finally:
        for handle in all_handles:
            handle.remove()
    result: dict[str, Any] = {}
    for spec in LOCUS_SPECS:
        observed = captures[spec.name]
        depthwise = observed["depthwise_activation"]
        pre_se = observed["pre_se"]
        post_se = observed["post_se"]
        own_pre_equal = bool(torch.equal(depthwise, pre_se))
        if not own_pre_equal:
            raise PreSELocusError(f"{spec.name} hook is not the depthwise activation")
        target_name = spec.module
        try:
            ordinal = execution_order.index(target_name)
        except ValueError as error:
            raise PreSELocusError(f"target SE did not execute: {target_name}") from error
        upstream = execution_order[:ordinal]
        result[spec.name] = {
            "status": "MEASURED_CONSTRUCTION_PROOF",
            "hook_module": target_name,
            "captured_shape_nchw": [int(value) for value in pre_se.shape],
            "equals_depthwise_activation_immediately_before_own_se": own_pre_equal,
            "own_se_applied_to_capture": False,
            "own_se_changes_tensor_max_abs": float((post_se - pre_se).abs().max().item()),
            "upstream_se_global_reduction_count": len(upstream),
            "upstream_se_modules": upstream,
            "locally_computable_from_mbconv_input": True,
            "strict_end_to_end_independently_tileable_from_rgb": len(upstream) == 0,
            "tileability_verdict": (
                "independently-tileable"
                if not upstream
                else "not-independently-tileable-after-upstream-se"
            ),
        }
    return result


class PreSECutCostLedger(DeepCutCostLedger):
    """Round-5 cost ledger stopped immediately before each target MBConv SE."""

    _CUTS: ClassVar[dict[str, tuple[int, int]]] = {
        "block2-pre-se": (1, 2),
        "block3-pre-se": (2, 2),
    }

    @staticmethod
    def _under_pre_se_cut(module: str, stage: int, block: int) -> bool:
        stem = "encoder.model.conv_stem"
        if module == stem:
            return True
        prefix = "encoder.model.blocks."
        if not module.startswith(prefix):
            return False
        parts = module[len(prefix) :].split(".")
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            return False
        module_stage = int(parts[0])
        module_block = int(parts[1])
        if module_stage < stage:
            return True
        if module_stage > stage:
            return False
        if module_block < block:
            return True
        if module_block > block:
            return False
        suffix = ".".join(parts[2:])
        return suffix in {"conv_pw", "conv_dw"}

    def summary(self) -> dict[str, Any]:
        if not self._conv_rows or not self._se_rows:
            raise PreSELocusError("cost ledger observed no convolution or SE reduction")
        names = [str(row["module"]) for row in self._conv_rows]
        if len(names) != len(set(names)):
            raise PreSELocusError("a convolution executed more than once")
        full_conv_macs = sum(int(row["forward_macs"]) for row in self._conv_rows)
        cuts: dict[str, Any] = {}
        for cut_name, (stage, block) in self._CUTS.items():
            conv_rows = [
                row
                for row in self._conv_rows
                if self._under_pre_se_cut(str(row["module"]), stage, block)
            ]
            se_rows = [
                row
                for row in self._se_rows
                if self._under_pre_se_cut(str(row["module"]), stage, block)
            ]
            conv_macs = sum(int(row["forward_macs"]) for row in conv_rows)
            pool_forward_flops = sum(int(row["global_pool_forward_flops"]) for row in se_rows)
            forward_plus_vjp_conv_flops = 4 * conv_macs
            forward_plus_vjp_pool_flops = 2 * pool_forward_flops
            total = forward_plus_vjp_conv_flops + forward_plus_vjp_pool_flops
            cuts[cut_name] = {
                "status": "DERIVED_FROM_MEASURED_REAL_TENSOR_SHAPES",
                "stop_point": f"encoder.model.blocks.{stage}.{block}.se:forward_pre",
                "forward_conv_macs": conv_macs,
                "forward_plus_input_vjp_conv_flops": forward_plus_vjp_conv_flops,
                "fraction_of_full_teacher_conv_flops": conv_macs / full_conv_macs,
                "upstream_se_reduction_count": len(se_rows),
                "global_pool_forward_flops": pool_forward_flops,
                "global_pool_forward_plus_vjp_flops": forward_plus_vjp_pool_flops,
                "global_pool_fraction_of_cut_conv_plus_pool_flops": (
                    forward_plus_vjp_pool_flops / total
                ),
                "global_gate_scalars_per_frame": sum(
                    int(row["gate_scalars"]) for row in se_rows
                ),
                "own_se_included": False,
                "strict_end_to_end_independently_tileable_from_rgb": len(se_rows) == 0,
                "tileability_verdict": (
                    "independently-tileable"
                    if not se_rows
                    else "not-independently-tileable-after-upstream-se"
                ),
            }
        return {
            "schema": "pre_se_locus_cut_cost.v1",
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
class PreSEPairGatedMLPWeights:
    """Feature-width-agnostic Round-5 pair-gated MLP weights."""

    input_weight: np.ndarray
    input_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    def validate(self) -> None:
        w1 = _finite(self.input_weight, name="input_weight")
        b1 = _finite(self.input_bias, name="input_bias")
        w2 = _finite(self.output_weight, name="output_weight")
        b2 = _finite(self.output_bias, name="output_bias")
        if w1.ndim != 2 or w1.shape[1] < 1:
            raise PreSELocusError("MLP input weight geometry drifted")
        if b1.shape != (w1.shape[0],):
            raise PreSELocusError("MLP input bias geometry drifted")
        if w2.shape != (ORDERED_PAIR_COUNT, w1.shape[0]):
            raise PreSELocusError("MLP pair-gated output weight geometry drifted")
        if b2.shape != (ORDERED_PAIR_COUNT,):
            raise PreSELocusError("MLP pair-gated output bias geometry drifted")


def pre_se_pair_gated_logits_numpy(
    features: Any, pair_ids: Any, weights: PreSEPairGatedMLPWeights
) -> np.ndarray:
    """Portable NumPy-fp32 inference for either PRE-SE locus width."""

    weights.validate()
    rows = _finite(features, name="features")
    pair = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    feature_count = int(weights.input_weight.shape[1])
    if rows.ndim != 2 or rows.shape != (pair.size, feature_count):
        raise PreSELocusError("MLP inference rows drifted")
    if np.any((pair < 0) | (pair >= ORDERED_PAIR_COUNT)):
        raise PreSELocusError("MLP pair ids drifted")
    hidden = np.maximum(
        np.float32(0.0), rows @ weights.input_weight.T + weights.input_bias[None]
    )
    logits = np.sum(hidden * weights.output_weight[pair], axis=1) + weights.output_bias[pair]
    if not np.isfinite(logits).all():
        raise PreSELocusError("MLP inference produced nonfinite logits")
    return np.ascontiguousarray(logits, dtype=np.float32)


__all__ = [
    "AUTHORITY_SCOPE",
    "RESEARCH_ONLY",
    "SCHEMA",
    "PreSECutCostLedger",
    "PreSELocusError",
    "PreSEPairGatedMLPWeights",
    "capture_pre_se_teacher",
    "locus_by_name",
    "pre_se_feature_snapshot",
    "pre_se_pair_block_features",
    "pre_se_pair_gated_logits_numpy",
    "verify_pre_se_taps",
]
