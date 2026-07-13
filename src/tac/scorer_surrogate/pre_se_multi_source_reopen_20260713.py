# SPDX-License-Identifier: MIT
"""Portable feature composition and cheap-global accounting for #484."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import MethodType
from typing import Any

import numpy as np

from tac.scorer_surrogate.pre_se_locus_20260713 import (
    PreSEPairGatedMLPWeights,
    pre_se_pair_block_features,
    pre_se_pair_gated_logits_numpy,
)

SCHEMA = "pre_se_multi_source_reopen_20260713.v1"
AUTHORITY_SCOPE = "local CPU frozen-SegNet costate research evidence; no score authority"
RESEARCH_ONLY = True
BASE_FEATURE_COUNT = 42
BLOCK2_FEATURE_COUNT = 144
BLOCK3_FEATURE_COUNT = 288
SENSITIVITY_FEATURE_COUNT = 2
MULTI_SOURCE_FEATURE_COUNT = 476


class PreSEMultiSourceError(ValueError):
    """The composition, donated-gate proof, or accounting failed closed."""


def multi_source_pair_block_features(
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
    """Concatenate the two protected charts while deduplicating shared columns."""

    block2, pairs2 = pre_se_pair_block_features(
        prefix_nchw,
        block2_nchw,
        labels_hw,
        margins_hw,
        pair_ids_hw,
        locus="block2-pre-se",
        checkpoint_index=checkpoint_index,
        checkpoint_count=checkpoint_count,
        stride=stride,
    )
    block3, pairs3 = pre_se_pair_block_features(
        prefix_nchw,
        block3_nchw,
        labels_hw,
        margins_hw,
        pair_ids_hw,
        locus="block3-pre-se",
        checkpoint_index=checkpoint_index,
        checkpoint_count=checkpoint_count,
        stride=stride,
    )
    if not np.array_equal(pairs2, pairs3):
        raise PreSEMultiSourceError("source pair rows drifted")
    if not np.array_equal(block2[:, :BASE_FEATURE_COUNT], block3[:, :BASE_FEATURE_COUNT]):
        raise PreSEMultiSourceError("shared shallow chart differs across source builders")
    if not np.array_equal(
        block2[:, -SENSITIVITY_FEATURE_COUNT:],
        block3[:, -SENSITIVITY_FEATURE_COUNT:],
    ):
        raise PreSEMultiSourceError("shared sensitivity chart differs across source builders")
    rows = np.concatenate(
        (
            block2[:, :BASE_FEATURE_COUNT],
            block2[:, BASE_FEATURE_COUNT:-SENSITIVITY_FEATURE_COUNT],
            block3[:, BASE_FEATURE_COUNT:-SENSITIVITY_FEATURE_COUNT],
            block2[:, -SENSITIVITY_FEATURE_COUNT:],
        ),
        axis=1,
    )
    if rows.shape != (pairs2.size, MULTI_SOURCE_FEATURE_COUNT):
        raise PreSEMultiSourceError("multi-source feature geometry drifted")
    return np.ascontiguousarray(rows, dtype=np.float32), pairs2


def compose_protected_feature_rows(
    block2_rows: Any, block3_rows: Any
) -> np.ndarray:
    """Compose aligned protected nonlinear chunks without copying duplicated columns."""

    left = np.asarray(block2_rows, dtype=np.float32)
    right = np.asarray(block3_rows, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise PreSEMultiSourceError("protected feature chunk row geometry drifted")
    if left.shape[1] != 188 or right.shape[1] != 332:
        raise PreSEMultiSourceError("protected single-locus widths drifted")
    if not np.array_equal(left[:, :BASE_FEATURE_COUNT], right[:, :BASE_FEATURE_COUNT]):
        raise PreSEMultiSourceError("protected base columns are not aligned")
    if not np.array_equal(
        left[:, -SENSITIVITY_FEATURE_COUNT:], right[:, -SENSITIVITY_FEATURE_COUNT:]
    ):
        raise PreSEMultiSourceError("protected sensitivity columns are not aligned")
    result = np.concatenate(
        (
            left[:, :BASE_FEATURE_COUNT],
            left[:, BASE_FEATURE_COUNT:-SENSITIVITY_FEATURE_COUNT],
            right[:, BASE_FEATURE_COUNT:-SENSITIVITY_FEATURE_COUNT],
            left[:, -SENSITIVITY_FEATURE_COUNT:],
        ),
        axis=1,
    )
    return np.ascontiguousarray(result, dtype=np.float32)


def derive_receptive_field_to_block3_pre_se(segnet: Any) -> dict[str, Any]:
    """Derive the exact local receptive-field size from executable convolutions."""

    model = segnet.encoder.model
    receptive_field = 1
    jump = 1
    rows: list[dict[str, int | str]] = []

    def charge(name: str, conv: Any) -> None:
        nonlocal receptive_field, jump
        kernel = int(conv.kernel_size[0])
        stride = int(conv.stride[0])
        dilation = int(conv.dilation[0])
        receptive_field += (kernel - 1) * dilation * jump
        jump *= stride
        rows.append(
            {
                "module": name,
                "kernel": kernel,
                "stride": stride,
                "dilation": dilation,
                "receptive_field": receptive_field,
                "output_stride": jump,
            }
        )

    charge("encoder.model.conv_stem", model.conv_stem)
    reached = False
    for stage_index in range(3):
        for block_index, block in enumerate(model.blocks[stage_index]):
            # DepthwiseSeparableConv starts at conv_dw; InvertedResidual has a 1x1 conv_pw first.
            if type(block).__name__ == "InvertedResidual":
                charge(f"encoder.model.blocks.{stage_index}.{block_index}.conv_pw", block.conv_pw)
            charge(f"encoder.model.blocks.{stage_index}.{block_index}.conv_dw", block.conv_dw)
            if (stage_index, block_index) == (2, 2):
                reached = True
                break
            projection = "conv_pwl" if hasattr(block, "conv_pwl") else "conv_pw"
            charge(
                f"encoder.model.blocks.{stage_index}.{block_index}.{projection}",
                getattr(block, projection),
            )
        if reached:
            break
    if not reached:
        raise PreSEMultiSourceError("block3 PRE-SE target was not reached")
    radius = (receptive_field - 1) // 2
    aligned_halo = ((radius + jump - 1) // jump) * jump
    return {
        "status": "DERIVED_FROM_EXECUTABLE_GRAPH",
        "receptive_field_input_pixels": receptive_field,
        "radius_input_pixels": radius,
        "output_stride_input_pixels": jump,
        "aligned_halo_input_pixels": aligned_halo,
        "convolution_recurrence": "r_next = r + (kernel-1)*dilation*jump; jump_next = jump*stride",
        "modules": rows,
    }


def ordered_upstream_se_modules(segnet: Any) -> tuple[str, ...]:
    """Return the unique execution-order SE ancestors of block3 PRE-SE."""

    target = segnet.encoder.model.blocks[2][2].se
    names = {id(module): name for name, module in segnet.named_modules()}
    rows: list[str] = []
    for stage_index in range(3):
        for _block_index, block in enumerate(segnet.encoder.model.blocks[stage_index]):
            if block.se is target:
                if len(rows) != len(set(rows)):
                    raise PreSEMultiSourceError("SE ancestor list contains duplicates")
                return tuple(rows)
            rows.append(names[id(block.se)])
    raise PreSEMultiSourceError("block3 target SE was not found")


def capture_full_frame_se_gates(
    segnet: Any, frame_nchw: Any, module_names: Sequence[str]
) -> dict[str, Any]:
    """Measure the requested SE gates once on the complete frame."""

    import torch

    modules = dict(segnet.named_modules())
    captured: dict[str, Any] = {}
    handles = []
    for name in module_names:
        if name not in modules or type(modules[name]).__name__ != "SqueezeExcite":
            raise PreSEMultiSourceError(f"invalid SE module {name}")

        def hook(_module: Any, _inputs: Any, output: Any, *, key: str = name) -> None:
            if key in captured:
                raise PreSEMultiSourceError(f"SE gate {key} executed twice")
            captured[key] = output.detach().clone()

        handles.append(modules[name].gate.register_forward_hook(hook))
    try:
        with torch.no_grad():
            segnet(frame_nchw.detach())
    finally:
        for handle in handles:
            handle.remove()
    if tuple(captured) != tuple(module_names):
        raise PreSEMultiSourceError("not every registered full-frame SE gate was captured")
    return captured


@contextmanager
def donated_se_gates(
    segnet: Any, gates: Mapping[str, Any]
) -> Iterator[None]:
    """Replace registered SE reductions with exact broadcast of donated gates."""

    modules = dict(segnet.named_modules())
    originals: dict[str, Any] = {}
    try:
        for name, gate in gates.items():
            module = modules.get(name)
            if module is None or type(module).__name__ != "SqueezeExcite":
                raise PreSEMultiSourceError(f"invalid donated-gate module {name}")
            originals[name] = module.forward

            def forward(self: Any, value: Any, *, fixed: Any = gate) -> Any:
                if value.ndim != 4 or fixed.shape != (value.shape[0], value.shape[1], 1, 1):
                    raise PreSEMultiSourceError("donated SE gate geometry drifted")
                return value * fixed.to(device=value.device, dtype=value.dtype)

            module.forward = MethodType(forward, module)
        yield
    finally:
        for name, original in originals.items():
            modules[name].forward = original


def capture_variable_pre_se(segnet: Any, frame_nchw: Any) -> tuple[Any, Any]:
    """Capture block2/block3 PRE-SE tensors for any aligned tile geometry."""

    import torch

    model = segnet.encoder.model
    captured: dict[str, Any] = {}
    handles = []
    for name, module in (
        ("block2-pre-se", model.blocks[1][2].se),
        ("block3-pre-se", model.blocks[2][2].se),
    ):
        def hook(_module: Any, inputs: Any, *, key: str = name) -> None:
            captured[key] = inputs[0].detach().clone()

        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            value = model.bn1(model.conv_stem(frame_nchw.detach()))
            value = model.blocks[0](value)
            value = model.blocks[1](value)
            model.blocks[2](value)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"block2-pre-se", "block3-pre-se"}:
        raise PreSEMultiSourceError("variable PRE-SE capture failed")
    return captured["block2-pre-se"], captured["block3-pre-se"]


def cheap_global_cost_accounting(
    cut_cost_model: Mapping[str, Any],
    *,
    upstream_se_modules: Sequence[str],
    tile_count: int,
    tiled_local_conv_macs: int,
) -> dict[str, Any]:
    """Charge the unique SE reductions/MLPs once and amortize over local tiles."""

    if tile_count < 1 or tiled_local_conv_macs < 1:
        raise PreSEMultiSourceError("invalid tile cost input")
    if len(upstream_se_modules) != len(set(upstream_se_modules)):
        raise PreSEMultiSourceError("upstream SE accounting contains duplicates")
    module_names = set(upstream_se_modules)
    se_rows = [
        row
        for row in cut_cost_model["per_se_global_reduction"]
        if str(row["module"]) in module_names
    ]
    if {str(row["module"]) for row in se_rows} != module_names:
        raise PreSEMultiSourceError("cost ledger does not cover every upstream SE module")
    se_mlp_macs = sum(
        int(row["forward_macs"])
        for row in cut_cost_model["per_conv_forward_macs"]
        if any(str(row["module"]).startswith(f"{name}.") for name in module_names)
    )
    pool_forward_flops = sum(int(row["global_pool_forward_flops"]) for row in se_rows)
    global_forward_plus_vjp = 4 * se_mlp_macs + 2 * pool_forward_flops
    local_forward_plus_vjp = 4 * tiled_local_conv_macs
    total = local_forward_plus_vjp + global_forward_plus_vjp
    return {
        "status": "DERIVED_FROM_MEASURED_REAL_TENSOR_SHAPES",
        "tile_count": tile_count,
        "unique_upstream_se_reduction_count": len(se_rows),
        "branch_incidence_count": 11,
        "branch_incidence_note": "4 block2 plus 7 block3 incidences; seven unique because block2 ancestors are shared",
        "global_gate_scalars_once": sum(int(row["gate_scalars"]) for row in se_rows),
        "SE_MLP_forward_macs_once": se_mlp_macs,
        "global_pool_forward_flops_once": pool_forward_flops,
        "global_forward_plus_vjp_flops_once": global_forward_plus_vjp,
        "tiled_local_conv_forward_macs_sum": tiled_local_conv_macs,
        "tiled_local_conv_forward_plus_vjp_flops_sum": local_forward_plus_vjp,
        "total_tiled_forward_plus_vjp_flops": total,
        "true_average_per_tile_forward_plus_vjp_flops_including_amortized_globals": total / tile_count,
        "amortized_global_forward_plus_vjp_flops_per_tile": global_forward_plus_vjp / tile_count,
        "convention": "one multiply-add is one MAC and two FLOPs; input VJP charges 2x forward conv FLOPs",
        "omitted": ["batch norm", "pointwise activation/sigmoid", "interpolation", "autograd bookkeeping"],
    }


__all__ = [
    "AUTHORITY_SCOPE",
    "MULTI_SOURCE_FEATURE_COUNT",
    "SCHEMA",
    "PreSEMultiSourceError",
    "PreSEPairGatedMLPWeights",
    "capture_full_frame_se_gates",
    "capture_variable_pre_se",
    "cheap_global_cost_accounting",
    "compose_protected_feature_rows",
    "derive_receptive_field_to_block3_pre_se",
    "donated_se_gates",
    "multi_source_pair_block_features",
    "ordered_upstream_se_modules",
    "pre_se_pair_gated_logits_numpy",
]
