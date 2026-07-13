# SPDX-License-Identifier: MIT
"""Round-3 frozen-prefix costate heads and target reformulations.

The trainable objects in this module are convex ridge heads on frozen features.
The primary formulation predicts the adjoint at a local pre-squeeze-excite
EfficientNet-B2 prefix, then applies the *exact* frozen-prefix VJP to obtain an
input RGB costate.  No live trainer, evaluator, or archive surface is touched.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.scorer_surrogate.frozen_replay_convex_head import (
    ReplaySufficientStatistics,
    array_sha256,
    fit_cached_convex_head,
    vector_fidelity,
)

SCHEMA = "replace_round3_fidelity_wall.v1"
AUTHORITY_SCOPE = "local macOS-CPU fp32 training-gradient research evidence; no score authority"
RESEARCH_ONLY = True
PREFIX_MODULE = "encoder.model.blocks.0.0.bn1"
PREFIX_CONV_MODULES = (
    "encoder.model.conv_stem",
    "encoder.model.blocks.0.0.conv_dw",
)
BASE_FEATURE_COUNT = 42
RFF_FREQUENCY_COUNT = 16
RFF_FEATURE_COUNT = BASE_FEATURE_COUNT + 2 * RFF_FREQUENCY_COUNT


class Round3FidelityError(ValueError):
    """A formulation, tensor, or custody invariant failed closed."""


def _float32(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise Round3FidelityError(f"{name} contains nonfinite values")
    return array


def _prefix_arrays(
    prefix_nchw: Any, labels_hw: Any, margins_hw: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = _float32(prefix_nchw, name="prefix_nchw")
    if prefix.ndim == 4 and prefix.shape[0] == 1:
        prefix = prefix[0]
    if prefix.ndim != 3 or prefix.shape[0] != 32:
        raise Round3FidelityError("registered prefix must have shape (1,32,h,w) or (32,h,w)")
    labels = np.asarray(labels_hw)
    margins = _float32(margins_hw, name="margins_hw")
    height, width = prefix.shape[1:]
    if labels.shape != (2 * height, 2 * width) or margins.shape != labels.shape:
        raise Round3FidelityError("source labels/margins must be exactly twice the prefix grid")
    if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() >= 5:
        raise Round3FidelityError("source labels must be integer class ids in [0,5)")
    return prefix, labels, margins


def prefix_feature_matrix(
    prefix_nchw: Any,
    labels_hw: Any,
    margins_hw: Any,
    *,
    checkpoint_index: int,
    checkpoint_count: int,
    stride: int,
) -> np.ndarray:
    """Build the sealed 42-column local frozen-prefix chart."""

    prefix, labels, margins = _prefix_arrays(prefix_nchw, labels_hw, margins_hw)
    if checkpoint_count != 3 or checkpoint_index not in range(checkpoint_count):
        raise Round3FidelityError("registered chart requires exactly three checkpoint stages")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise Round3FidelityError("stride must be an integer >=1")
    height, width = prefix.shape[1:]
    label_prefix = labels[::2, ::2]
    margin_prefix = margins[::2, ::2]
    if label_prefix.shape != (height, width):
        raise Round3FidelityError("source-grid decimation did not match prefix geometry")
    one_hot = np.eye(5, dtype=np.float32)[label_prefix.astype(np.int64)].transpose(2, 0, 1)
    stage = np.zeros((checkpoint_count, height, width), dtype=np.float32)
    stage[checkpoint_index] = np.float32(1.0)
    channels = np.concatenate(
        (
            np.ones((1, height, width), dtype=np.float32),
            prefix,
            one_hot,
            np.tanh(margin_prefix, dtype=np.float32)[None],
            stage,
        ),
        axis=0,
    )
    if channels.shape[0] != BASE_FEATURE_COUNT:
        raise Round3FidelityError("prefix feature implementation and sealed width disagree")
    rows = channels[:, ::stride, ::stride].reshape(BASE_FEATURE_COUNT, -1).T
    return np.ascontiguousarray(rows, dtype=np.float32)


def deterministic_rff_parameters(
    *, feature_count_without_bias: int = BASE_FEATURE_COUNT - 1, seed: int = 455
) -> np.ndarray:
    """Return the one preregistered, untuned random Fourier frequency matrix."""

    if feature_count_without_bias != BASE_FEATURE_COUNT - 1 or seed != 455:
        raise Round3FidelityError("changing RFF input width or seed creates a new instance")
    generator = np.random.default_rng(seed)
    omega = generator.standard_normal(
        (feature_count_without_bias, RFF_FREQUENCY_COUNT), dtype=np.float32
    )
    omega /= np.float32(math.sqrt(feature_count_without_bias))
    return np.ascontiguousarray(omega, dtype=np.float32)


def rff_lift(base_features: Any, *, seed: int = 455) -> np.ndarray:
    """Append fixed sine/cosine features; the fitted ridge remains convex."""

    base = _float32(base_features, name="base_features")
    if base.ndim != 2 or base.shape[1] != BASE_FEATURE_COUNT:
        raise Round3FidelityError("base feature matrix must have the sealed 42 columns")
    omega = deterministic_rff_parameters(seed=seed)
    # Torch can leave stale IEEE status flags on the worker thread.  Suppress
    # warnings from those flags, then fail on actual nonfinite output below.
    with np.errstate(all="ignore"):
        phase = np.ascontiguousarray(base[:, 1:] @ omega, dtype=np.float32)
    if not np.isfinite(phase).all():
        raise Round3FidelityError("RFF projection contains nonfinite values")
    scale = np.float32(math.sqrt(1.0 / RFF_FREQUENCY_COUNT))
    lifted = np.concatenate(
        (base, scale * np.cos(phase), scale * np.sin(phase)), axis=1
    )
    if lifted.shape[1] != RFF_FEATURE_COUNT:
        raise Round3FidelityError("RFF implementation and sealed width disagree")
    return np.ascontiguousarray(lifted, dtype=np.float32)


def sampled_prefix_target_rows(target_nchw: Any, *, stride: int) -> np.ndarray:
    target = _float32(target_nchw, name="target_nchw")
    if target.ndim == 4 and target.shape[0] == 1:
        target = target[0]
    if target.ndim != 3:
        raise Round3FidelityError("prefix target must be NCHW or CHW")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise Round3FidelityError("stride must be an integer >=1")
    return np.ascontiguousarray(
        target[:, ::stride, ::stride].reshape(target.shape[0], -1).T,
        dtype=np.float32,
    )


def prefix_cell_costate_l2(input_costate_nchw: Any) -> np.ndarray:
    """Reduce the exact RGB input costate to 2x2 prefix-cell L2 norms."""

    costate = _float32(input_costate_nchw, name="input_costate_nchw")
    if costate.ndim == 3:
        costate = costate[None]
    if costate.ndim != 4 or costate.shape[0] != 1 or costate.shape[1] != 3:
        raise Round3FidelityError("input costate must have shape (1,3,H,W)")
    height, width = costate.shape[2:]
    if height % 2 or width % 2:
        raise Round3FidelityError("input grid must be divisible into 2x2 prefix cells")
    cell_energy = np.square(costate, dtype=np.float32).reshape(
        1, 3, height // 2, 2, width // 2, 2
    ).sum(axis=(1, 3, 5), dtype=np.float32)[0]
    return np.sqrt(cell_energy, dtype=np.float32)


def log_costate_mass_target_rows(input_costate_nchw: Any, *, stride: int) -> np.ndarray:
    mass = prefix_cell_costate_l2(input_costate_nchw)
    log_mass = np.log(np.maximum(mass, np.float32(np.finfo(np.float32).tiny)))
    return np.ascontiguousarray(log_mass[::stride, ::stride].reshape(-1, 1), dtype=np.float32)


@dataclass(frozen=True)
class MultiTargetStateStatistics:
    gram: np.ndarray
    rhs: np.ndarray
    target_square_by_channel: np.ndarray
    row_count: int
    feature_sha256: str
    target_sha256: str

    def validate(self) -> None:
        gram = _float32(self.gram, name="gram")
        rhs = _float32(self.rhs, name="rhs")
        square = np.asarray(self.target_square_by_channel, dtype=np.float64)
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
            raise Round3FidelityError("gram must be square")
        if rhs.ndim != 2 or rhs.shape[0] != gram.shape[0]:
            raise Round3FidelityError("rhs feature dimension disagrees with gram")
        if square.shape != (rhs.shape[1],) or not np.isfinite(square).all() or np.any(square < 0):
            raise Round3FidelityError("target-square channel custody is invalid")
        if self.row_count < 1:
            raise Round3FidelityError("state row count must be positive")
        if not np.allclose(gram, gram.T, rtol=0.0, atol=8.0 * np.finfo(np.float32).eps):
            raise Round3FidelityError("gram is not symmetric at the fp32 floor")


def cache_multi_target_sufficient_statistics(
    features: Any, targets: Any
) -> MultiTargetStateStatistics:
    x = _float32(features, name="features")
    y = _float32(targets, name="targets")
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 1:
        raise Round3FidelityError("features and targets must be row-aligned matrices")
    # Torch can leave stale IEEE flags on this worker thread.  Silence only
    # those warnings and let the record validator refuse actual nonfinite data.
    with np.errstate(all="ignore"):
        gram = np.ascontiguousarray(x.T @ x, dtype=np.float32)
        rhs = np.ascontiguousarray(x.T @ y, dtype=np.float32)
    gram = np.ascontiguousarray(np.float32(0.5) * (gram + gram.T), dtype=np.float32)
    record = MultiTargetStateStatistics(
        gram=gram,
        rhs=rhs,
        target_square_by_channel=np.sum(
            np.square(y.astype(np.float64)), axis=0, dtype=np.float64
        ),
        row_count=x.shape[0],
        feature_sha256=array_sha256(x),
        target_sha256=array_sha256(y),
    )
    record.validate()
    return record


@dataclass(frozen=True)
class MultiTargetReplayStatistics:
    gram: np.ndarray
    rhs: np.ndarray
    target_square_by_channel: np.ndarray
    row_count: int
    state_count: int
    per_state_gram: np.ndarray
    per_state_rhs: np.ndarray
    per_state_rows: np.ndarray

    def validate(self) -> None:
        gram = _float32(self.gram, name="aggregate_gram")
        rhs = _float32(self.rhs, name="aggregate_rhs")
        per_gram = _float32(self.per_state_gram, name="per_state_gram")
        per_rhs = _float32(self.per_state_rhs, name="per_state_rhs")
        rows = np.asarray(self.per_state_rows)
        if gram.shape != (rhs.shape[0], rhs.shape[0]):
            raise Round3FidelityError("aggregate gram/rhs shapes disagree")
        if per_gram.shape != (self.state_count, rhs.shape[0], rhs.shape[0]):
            raise Round3FidelityError("per-state gram shape disagrees")
        if per_rhs.shape != (self.state_count, *rhs.shape):
            raise Round3FidelityError("per-state rhs shape disagrees")
        if rows.shape != (self.state_count,) or self.row_count != int(rows.sum()):
            raise Round3FidelityError("per-state row counts do not reconcile")


def aggregate_multi_target_statistics(
    records: Sequence[MultiTargetStateStatistics],
) -> MultiTargetReplayStatistics:
    if not records:
        raise Round3FidelityError("at least one cached state is required")
    for record in records:
        record.validate()
    shape = records[0].rhs.shape
    if any(record.rhs.shape != shape or record.gram.shape != (shape[0], shape[0]) for record in records):
        raise Round3FidelityError("cached states use different feature/target charts")
    per_gram = np.stack([record.gram for record in records]).astype(np.float32, copy=False)
    per_rhs = np.stack([record.rhs for record in records]).astype(np.float32, copy=False)
    rows = np.asarray([record.row_count for record in records], dtype=np.int64)
    result = MultiTargetReplayStatistics(
        gram=np.ascontiguousarray(np.sum(per_gram, axis=0, dtype=np.float32)),
        rhs=np.ascontiguousarray(np.sum(per_rhs, axis=0, dtype=np.float32)),
        target_square_by_channel=np.sum(
            np.stack([record.target_square_by_channel for record in records]),
            axis=0,
            dtype=np.float64,
        ),
        row_count=int(rows.sum()),
        state_count=len(records),
        per_state_gram=np.ascontiguousarray(per_gram),
        per_state_rhs=np.ascontiguousarray(per_rhs),
        per_state_rows=rows,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class MultiTargetFit:
    weights: np.ndarray
    block_fits: tuple[Any, ...]

    def summary(self) -> dict[str, Any]:
        certificates = [fit.certificate.to_dict() for fit in self.block_fits]
        if any(certificate != certificates[0] for certificate in certificates[1:]):
            raise Round3FidelityError("separable output blocks derived different Hessians")
        parameter_ratios = [
            float(row["parameter_contraction_ratio"])
            for fit in self.block_fits
            for row in fit.trace
            if row["parameter_contraction_ratio"] is not None
        ]
        objective_ratios = [
            float(row["objective_gap_ratio"])
            for fit in self.block_fits
            for row in fit.trace
            if row["objective_gap_ratio"] is not None
        ]
        return {
            "feature_count": int(self.weights.shape[0]),
            "target_count": int(self.weights.shape[1]),
            "three_channel_block_count": len(self.block_fits),
            "certificate": certificates[0],
            "max_admitted_parameter_contraction_ratio": max(parameter_ratios, default=None),
            "max_admitted_objective_gap_ratio": max(objective_ratios, default=None),
            "terminal_gradient_norm_max": max(
                float(fit.terminal_gradient_norm) for fit in self.block_fits
            ),
            "residual_bounds_all_validated": all(
                bool(fit.residual_bounds_validated) for fit in self.block_fits
            ),
            "weights_array_sha256": array_sha256(self.weights),
        }


def fit_multi_target_ridge(
    stats: MultiTargetReplayStatistics, *, epochs: int
) -> MultiTargetFit:
    """Fit arbitrary output width as separable 3-channel round-2 ridge blocks."""

    stats.validate()
    fits: list[Any] = []
    weight_blocks: list[np.ndarray] = []
    for start in range(0, stats.rhs.shape[1], 3):
        stop = min(start + 3, stats.rhs.shape[1])
        width = stop - start
        rhs = np.zeros((stats.rhs.shape[0], 3), dtype=np.float32)
        rhs[:, :width] = stats.rhs[:, start:stop]
        per_rhs = np.zeros((stats.state_count, stats.rhs.shape[0], 3), dtype=np.float32)
        per_rhs[:, :, :width] = stats.per_state_rhs[:, :, start:stop]
        block_stats = ReplaySufficientStatistics(
            gram=stats.gram,
            rhs=rhs,
            target_square_sum=float(stats.target_square_by_channel[start:stop].sum()),
            row_count=stats.row_count,
            state_count=stats.state_count,
            per_state_gram=stats.per_state_gram,
            per_state_rhs=per_rhs,
            per_state_rows=stats.per_state_rows,
        )
        fit = fit_cached_convex_head(block_stats, epochs=epochs)
        fits.append(fit)
        weight_blocks.append(fit.weights[:, :width])
    weights = np.ascontiguousarray(np.concatenate(weight_blocks, axis=1), dtype=np.float32)
    result = MultiTargetFit(weights=weights, block_fits=tuple(fits))
    result.summary()
    return result


def predict_prefix_adjoint(
    features: Any, weights: Any, *, height: int, width: int, channels: int = 32
) -> np.ndarray:
    x = _float32(features, name="features")
    w = _float32(weights, name="weights")
    if x.ndim != 2 or w.shape != (x.shape[1], channels):
        raise Round3FidelityError("feature/head shapes disagree")
    if x.shape[0] != height * width:
        raise Round3FidelityError("full prefix-grid row count disagrees with geometry")
    with np.errstate(all="ignore"):
        rows = np.ascontiguousarray(x @ w, dtype=np.float32)
    if not np.isfinite(rows).all():
        raise Round3FidelityError("predicted prefix adjoint contains nonfinite values")
    return np.ascontiguousarray(rows.T.reshape(1, channels, height, width), dtype=np.float32)


def local_prefix_activation(segnet: Any, frame_nchw: Any) -> Any:
    """Execute exactly the registered local prefix, ending before first SE."""

    model = segnet.encoder.model
    block = model.blocks[0][0]
    value = model.bn1(model.conv_stem(frame_nchw))
    value = block.aa(block.bn1(block.conv_dw(value)))
    if value.ndim != 4 or value.shape[1] != 32:
        raise Round3FidelityError("local prefix geometry drifted")
    return value


def local_prefix_feature_snapshot(segnet: Any, frame_nchw: Any) -> Any:
    """Return a detached chart tensor through the teacher's grad-enabled path.

    On the registered CPU backend, ``Conv2d`` can select a numerically distinct
    implementation when its input does not require gradients.  The frozen
    chart is defined by the exact-teacher path, so every cached/later rung must
    reconstruct it with ``requires_grad=True`` even when no VJP follows.
    """

    frame = frame_nchw.detach().requires_grad_(True)
    return local_prefix_activation(segnet, frame).detach()


def capture_exact_teacher_with_prefix_adjoint(
    *, segnet: Any, frame_nchw: Any, labels: Any
) -> tuple[Any, Any, Any, dict[str, float], float]:
    """One exact label call yielding input and registered-prefix costates."""

    import torch
    import torch.nn.functional as functional

    frame = frame_nchw.detach().requires_grad_(True)
    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if "prefix_graph" in captured:
            raise Round3FidelityError("registered prefix executed more than once")
        # Some downstream EfficientNet primitives reuse activation storage.
        # Keep the graph object for the exact adjoint and an immutable snapshot
        # for the fixed feature chart; never let later in-place work rewrite X.
        captured["prefix_graph"] = output
        captured["prefix_snapshot"] = output.detach().clone()

    prefix_module = segnet.encoder.model.blocks[0][0].bn1
    handle = prefix_module.register_forward_hook(hook)
    started = time.perf_counter()
    try:
        logits = segnet(frame)
        if "prefix_graph" not in captured:
            raise Round3FidelityError("registered prefix hook did not fire")
        loss = functional.cross_entropy(logits, labels)
        prefix_adjoint, input_costate = torch.autograd.grad(
            loss, (captured["prefix_graph"], frame), retain_graph=False
        )
    finally:
        handle.remove()
    elapsed = time.perf_counter() - started
    tensors = (captured["prefix_snapshot"], prefix_adjoint, input_costate)
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise Round3FidelityError("exact teacher produced a nonfinite prefix/input tensor")
    metrics = {
        "ce": float(loss.detach().item()),
        "dseg": float((logits.argmax(1) != labels).float().mean().detach().item()),
    }
    return (
        captured["prefix_snapshot"],
        prefix_adjoint.detach(),
        input_costate.detach(),
        metrics,
        elapsed,
    )


def exact_prefix_vjp(
    *, segnet: Any, frame_nchw: Any, prefix_adjoint_nchw: Any
) -> tuple[Any, float]:
    """Map a predicted prefix adjoint to RGB with the exact local-prefix VJP."""

    import torch

    frame = frame_nchw.detach().requires_grad_(True)
    adjoint = torch.as_tensor(prefix_adjoint_nchw, dtype=torch.float32)
    started = time.perf_counter()
    prefix = local_prefix_activation(segnet, frame)
    if tuple(prefix.shape) != tuple(adjoint.shape):
        raise Round3FidelityError("predicted prefix adjoint geometry drifted")
    input_costate = torch.autograd.grad(prefix, frame, grad_outputs=adjoint)[0].detach()
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(input_costate).all()):
        raise Round3FidelityError("prefix VJP produced a nonfinite input costate")
    return input_costate, elapsed


def source_margin_risk_scores(margins_hw: Any) -> np.ndarray:
    """Cheapest canonical localizer: smallest absolute source margin first."""

    margins = _float32(margins_hw, name="margins_hw")
    if margins.ndim != 2 or margins.shape[0] % 2 or margins.shape[1] % 2:
        raise Round3FidelityError("margin grid must be an even two-dimensional array")
    cell_abs_margin = np.abs(margins).reshape(
        margins.shape[0] // 2, 2, margins.shape[1] // 2, 2
    ).min(axis=(1, 3))
    return np.ascontiguousarray(-cell_abs_margin, dtype=np.float32)


def mass_localization_metrics(
    input_costate_nchw: Any, prefix_scores_hw: Any, *, area_fraction: float
) -> dict[str, float | int]:
    """Measure L2 mass in a deterministic top-area prefix-cell mask."""

    costate = _float32(input_costate_nchw, name="input_costate_nchw")
    if costate.ndim == 3:
        costate = costate[None]
    scores = _float32(prefix_scores_hw, name="prefix_scores_hw")
    cell_mass = np.square(prefix_cell_costate_l2(costate), dtype=np.float64)
    if scores.shape != cell_mass.shape:
        raise Round3FidelityError("localizer scores and prefix-cell mass disagree")
    if not 0.0 < area_fraction < 1.0:
        raise Round3FidelityError("area_fraction must lie strictly between zero and one")
    count = max(1, math.ceil(area_fraction * scores.size))
    flat_index = np.arange(scores.size, dtype=np.int64)
    selected = np.lexsort((flat_index, -scores.reshape(-1).astype(np.float64)))[:count]
    oracle = np.lexsort((flat_index, -cell_mass.reshape(-1)))[:count]
    total = float(cell_mass.sum(dtype=np.float64))
    if total <= 0.0:
        raise Round3FidelityError("exact input costate has zero L2 mass")
    retained = float(cell_mass.reshape(-1)[selected].sum(dtype=np.float64) / total)
    oracle_retained = float(cell_mass.reshape(-1)[oracle].sum(dtype=np.float64) / total)
    retained_square = float(cell_mass.reshape(-1)[selected].sum(dtype=np.float64))
    oracle_square = float(cell_mass.reshape(-1)[oracle].sum(dtype=np.float64))
    realized_area = count / float(scores.size)
    return {
        "selected_prefix_cells": count,
        "prefix_cell_count": int(scores.size),
        "realized_input_area_fraction": realized_area,
        "exact_costate_l2_square": total,
        "retained_exact_costate_l2_square": retained_square,
        "oracle_retained_exact_costate_l2_square": oracle_square,
        "retained_exact_costate_l2_mass_fraction": retained,
        "oracle_retained_exact_costate_l2_mass_fraction": oracle_retained,
        "uplift_over_uniform_area": retained / realized_area,
        "conditional_masked_exact_costate_cosine": math.sqrt(retained),
    }


class ConvMacLedger:
    """Observe one real forward and derive the frozen input-VJP FLOP ratio."""

    def __init__(self, segnet: Any) -> None:
        self.segnet = segnet
        self._handles: list[Any] = []
        self._rows: list[dict[str, int | str]] = []

    def __enter__(self) -> ConvMacLedger:
        import torch

        for name, module in self.segnet.named_modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue

            def hook(_module: Any, inputs: Any, output: Any, *, module_name: str = name) -> None:
                if not hasattr(output, "shape") or len(output.shape) != 4:
                    raise Round3FidelityError("Conv2d output geometry is not NCHW")
                batch, channels_out, height, width = (int(value) for value in output.shape)
                channels_in = int(_module.in_channels)
                kernel_h, kernel_w = (int(value) for value in _module.kernel_size)
                macs = (
                    batch
                    * channels_out
                    * height
                    * width
                    * (channels_in // int(_module.groups))
                    * kernel_h
                    * kernel_w
                )
                self._rows.append({"module": module_name, "forward_macs": macs})

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self) -> dict[str, Any]:
        if not self._rows:
            raise Round3FidelityError("Conv MAC ledger observed no forward")
        names = [str(row["module"]) for row in self._rows]
        if len(names) != len(set(names)):
            raise Round3FidelityError("a Conv2d module executed more than once in the observed forward")
        full_forward_macs = sum(int(row["forward_macs"]) for row in self._rows)
        prefix_forward_macs = sum(
            int(row["forward_macs"])
            for row in self._rows
            if row["module"] in PREFIX_CONV_MODULES
        )
        if not all(name in names for name in PREFIX_CONV_MODULES):
            raise Round3FidelityError("registered prefix Conv modules were not both observed")
        # Frozen weights require only grad-input, so the conventional convolution
        # operation model charges one forward-equivalent pass for the VJP.
        full_teacher_macs = 2 * full_forward_macs
        prefix_prediction_macs = 2 * prefix_forward_macs
        return {
            "status": "DERIVED_FROM_MEASURED_REAL_TENSOR_SHAPES",
            "convention": "one multiply-add is one MAC and two FLOPs",
            "full_forward_conv_macs": full_forward_macs,
            "full_forward_conv_flops": 2 * full_forward_macs,
            "full_forward_plus_input_backward_conv_macs": full_teacher_macs,
            "full_forward_plus_input_backward_conv_flops": 2 * full_teacher_macs,
            "prefix_forward_conv_macs": prefix_forward_macs,
            "prefix_forward_plus_input_vjp_conv_macs": prefix_prediction_macs,
            "prefix_forward_plus_input_vjp_conv_flops": 2 * prefix_prediction_macs,
            "prefix_fraction_of_full_teacher_conv_flops": (
                prefix_prediction_macs / full_teacher_macs
            ),
            "omitted_from_derived_flop_model": [
                "batch normalization",
                "activation",
                "pooling and interpolation",
                "loss and argmax",
                "head matrix multiply",
                "autograd bookkeeping",
            ],
            "per_conv_forward_macs": self._rows,
        }


def direction_admission(
    *, cosine: float, positive_dot_state_fraction: float, cosine_bar: float, fraction_bar: float
) -> dict[str, Any]:
    values = (cosine, positive_dot_state_fraction, cosine_bar, fraction_bar)
    if any(not math.isfinite(float(value)) for value in values):
        raise Round3FidelityError("direction admission accepts finite values only")
    passed = cosine >= cosine_bar and positive_dot_state_fraction >= fraction_bar
    return {
        "verdict": "PASS" if passed else "FAIL",
        "heldout_input_costate_cosine": cosine,
        "cosine_bar": cosine_bar,
        "cosine_multiple_of_round2_noise": cosine / 0.0014157933865487525,
        "positive_dot_state_fraction": positive_dot_state_fraction,
        "positive_dot_state_fraction_bar": fraction_bar,
        "both_gates_required": True,
    }


__all__ = [
    "AUTHORITY_SCOPE",
    "BASE_FEATURE_COUNT",
    "PREFIX_MODULE",
    "RESEARCH_ONLY",
    "RFF_FEATURE_COUNT",
    "ConvMacLedger",
    "MultiTargetFit",
    "MultiTargetReplayStatistics",
    "MultiTargetStateStatistics",
    "Round3FidelityError",
    "aggregate_multi_target_statistics",
    "array_sha256",
    "cache_multi_target_sufficient_statistics",
    "capture_exact_teacher_with_prefix_adjoint",
    "deterministic_rff_parameters",
    "direction_admission",
    "exact_prefix_vjp",
    "fit_multi_target_ridge",
    "local_prefix_activation",
    "local_prefix_feature_snapshot",
    "log_costate_mass_target_rows",
    "mass_localization_metrics",
    "predict_prefix_adjoint",
    "prefix_feature_matrix",
    "rff_lift",
    "sampled_prefix_target_rows",
    "source_margin_risk_scores",
    "vector_fidelity",
]
